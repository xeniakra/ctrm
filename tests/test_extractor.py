import json
from pathlib import Path

import pytest

from optiflow_ctrm.extractor import ExtractionFailedError, RecapExtractor
from optiflow_ctrm.models import ExtractionResult, ModelEvidenceClaim, extraction_json_schema
from optiflow_ctrm.provenance import EvidenceValidationError, locate_model_evidence
from optiflow_ctrm.runtime.base import InferenceResult


ROOT = Path(__file__).parents[1]


def test_voice_note_recovers_explicit_terms_even_when_model_abstains():
    fixture = json.loads((ROOT / 'evals/cases/06_fixed_fx.json').read_text())
    source = (ROOT / fixture['source']['path']).read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture['expected'])), source)
    generated['deal']['fx'] = None
    generated['deal']['quality_spec'] = None
    generated['deal']['pricing']['differential'] = {'value': None, 'unit': None}
    generated['deal']['inspection'].update(company='buyer', cost_split='buyer_pays')
    actual = RecapExtractor(FakeRuntime(generated, evidence)).extract(source)
    assert actual.model_dump(mode='json', by_alias=True)['deal'] == fixture['expected']['deal']


def test_unrelated_price_is_not_used_for_enrichment_or_normalization():
    fixture = json.loads((ROOT / 'evals/cases/17_deal_distraction.json').read_text())
    source = (ROOT / fixture['source']['path']).read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture['expected'])), source)
    generated['deal']['pricing']['quote_code'] = None
    generated['deal']['pricing']['differential']['value'] = 8
    actual = RecapExtractor(FakeRuntime(generated, evidence)).extract(source)
    assert actual.model_dump(mode='json', by_alias=True)['deal'] == fixture['expected']['deal']


def _inference(value: object) -> InferenceResult:
    return InferenceResult(
        value=value,
        raw_text=json.dumps(value),
        model="qwen-test",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        tokens_per_second=1.0,
    )


class FakeRuntime:
    def __init__(self, *values: object) -> None:
        self.values = list(values)
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def generate_json(self, messages, json_schema):
        self.calls.append((list(messages), dict(json_schema)))
        return _inference(self.values.pop(0))


def _generated(expected: dict, source: str) -> tuple[dict, dict]:
    generated = {key: value for key, value in expected.items() if key != "provenance"}
    grouped: dict[tuple[int, int], set[str]] = {}
    for entry in expected["provenance"]:
        if entry["field"] in {
            "/deal/product/product_code",
            "/deal/product/reference_name",
        }:
            continue
        field = _evidence_field(entry["field"])
        for evidence in entry["evidence"]:
            if evidence["kind"] == "source":
                fragment = source[evidence["start"] : evidence["end"]]
                if "\n" in fragment or "\r" in fragment:
                    offset = evidence["start"]
                    for line in fragment.splitlines(keepends=True):
                        quote = line.rstrip("\r\n")
                        if quote:
                            grouped.setdefault((offset, offset + len(quote)), set()).add(field)
                        offset += len(line)
                else:
                    start = source.rfind("\n", 0, evidence["start"]) + 1
                    end = source.find("\n", evidence["end"])
                    if end < 0:
                        end = len(source)
                    grouped.setdefault((start, end), set()).add(field)
    evidence_values: dict[str, list[str]] = {}
    for (start, end), fields in sorted(grouped.items()):
        for field in fields:
            quote = source[start:end]
            if quote not in evidence_values.setdefault(field, []):
                evidence_values[field].append(quote)
    evidence = {"evidence": dict(sorted(evidence_values.items()))}
    return generated, evidence


def _evidence_field(path: str) -> str:
    grouped_paths = (
        "/deal/quantity",
        "/deal/delivery_term",
        "/deal/delivery_window",
        "/deal/pricing/period",
        "/deal/pricing/differential",
        "/deal/payment_terms",
        "/deal/fx",
        "/deal/inspection",
        "/deal/demurrage",
        "/deal/notes",
    )
    return next((prefix for prefix in grouped_paths if path.startswith(prefix)), path)


@pytest.mark.parametrize(
    ("recap_name", "case_name"),
    [
        ("recap_01.txt", "01_clean_telegram.json"),
        # ("recap_02.txt", "02_forwarded_amendments.json"),
        # ("recap_03.txt", "03_incomplete_must_abstain.json"),
    ],
)
def test_supplied_recaps_use_canonical_schema_without_changing_source(
    recap_name: str, case_name: str
) -> None:
    recap = (ROOT / "data" / "recaps" / recap_name).read_text()
    expected = json.loads((ROOT / "evals" / "cases" / case_name).read_text())["expected"]
    runtime = FakeRuntime(*_generated(expected, recap))

    result = RecapExtractor(runtime).extract(recap)

    assert isinstance(result, ExtractionResult)
    assert len(runtime.calls) == 2
    messages, schema = runtime.calls[0]
    assert messages[-1]["content"].endswith(recap)
    assert schema == extraction_json_schema()
    assert "verified quotations" in messages[0]["content"].lower()
    buyer_evidence = next(
        entry.evidence for entry in result.provenance if entry.field == "/deal/buyer"
    )
    span = next(item for item in buyer_evidence if item.kind == "source")
    assert result.deal.buyer in recap[span.start : span.end]


def test_invalid_first_result_gets_exactly_one_repair_attempt() -> None:
    expected = json.loads((ROOT / "evals" / "cases" / "01_clean_telegram.json").read_text())[
        "expected"
    ]
    recap = (ROOT / "data" / "recaps" / "recap_01.txt").read_text()
    runtime = FakeRuntime({"deal": {}}, *_generated(expected, recap))
    result = RecapExtractor(runtime).extract(recap)

    assert result.deal.buyer == "Helvig Energy AG"
    assert len(runtime.calls) == 3
    repair_messages, _ = runtime.calls[1]
    assert "one final repair" in repair_messages[-1]["content"].lower()
    assert recap in repair_messages[1]["content"]


def test_second_invalid_result_fails_safely() -> None:
    runtime = FakeRuntime({"deal": {}}, {"deal": {}})

    with pytest.raises(ExtractionFailedError, match="after one repair"):
        RecapExtractor(runtime).extract("raw recap")

    assert len(runtime.calls) == 2


def test_reference_codes_are_resolved_by_deterministic_csv_lookup() -> None:
    expected = json.loads((ROOT / "evals" / "cases" / "01_clean_telegram.json").read_text())[
        "expected"
    ]
    recap = (ROOT / "data" / "recaps" / "recap_01.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(expected)), recap)
    generated["deal"]["product"]["product_code"] = None
    generated["deal"]["product"]["reference_name"] = None
    generated["deal"]["pricing"]["quote_code"] = "Platts FOB MED Italy Gasoil 0.1%"

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.deal.product.product_code == "GO01"
    assert result.deal.product.reference_name == "Gasoil 0.1% S"
    assert result.deal.pricing.quote_code == "PLATTS_FOB_MED_ITALY_GASOIL_01"
    for field in (
        "/deal/product/product_code",
        "/deal/product/reference_name",
        "/deal/pricing/quote_code",
    ):
        evidence = next(entry.evidence for entry in result.provenance if entry.field == field)
        assert {item.kind for item in evidence} == {"reference", "source"}


def test_catalog_code_alias_splits_product_from_explicit_quality_spec() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "04_calendar_low.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_04.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture["expected"])), recap)
    generated["deal"]["product"] = {
        "as_written": "ULSD 10 ppm, EN 590",
        "product_code": None,
        "reference_name": None,
    }
    generated["deal"]["quality_spec"] = "10 ppm, EN 590"

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.deal.product.as_written == "ULSD 10 ppm"
    assert result.deal.product.product_code == "ULSD10"
    assert result.deal.product.reference_name == "Ultra Low Sulphur Diesel 10 ppm"
    assert result.deal.quality_spec == "EN 590"
    evidence_by_field = {entry.field: entry.evidence for entry in result.provenance}
    assert "ULSD 10 ppm" in {
        recap[item.start : item.end]
        for item in evidence_by_field["/deal/product/as_written"]
        if item.kind == "source"
    }
    assert "EN 590" in {
        recap[item.start : item.end]
        for item in evidence_by_field["/deal/quality_spec"]
        if item.kind == "source"
    }


def test_reference_enrichment_does_not_remove_fields_from_the_evidence_prompt() -> None:
    fixture = json.loads(
        (ROOT / "evals" / "cases" / "02_forwarded_amendments.json").read_text()
    )
    recap = (ROOT / "data" / "recaps" / "recap_02.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture["expected"])), recap)
    generated["deal"]["product"]["as_written"] = (
        "naphtha full range, spec per our standard NAPH-FR-2 sheet"
    )
    generated["deal"]["product"]["product_code"] = None
    generated["deal"]["product"]["reference_name"] = None
    generated["deal"]["quality_spec"] = "per our standard NAPH-FR-2 sheet"
    runtime = FakeRuntime(generated, evidence)

    RecapExtractor(runtime).extract(recap)

    evidence_properties = runtime.calls[1][1]["properties"]["evidence"]["properties"]
    assert "/deal/product/as_written" in evidence_properties
    assert "/deal/quality_spec" in evidence_properties


def test_mixed_units_canonicalizes_closed_terms_without_wording_specific_notes() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "05_mixed_units.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_05.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture["expected"])), recap)
    generated["deal"]["notes"] = ["laycan 18–20 Oct 26"]
    generated["deal"]["pricing"]["period"]["expression"] = "BL-2 thru BL+2"
    generated["deal"]["product"]["product_code"] = None
    generated["deal"]["product"]["reference_name"] = None
    generated["deal"]["quality_spec"] = "Jet A-1"
    generated["deal"]["quantity"]["tolerance_option"] = "buyer"
    evidence["evidence"]["/deal/notes"] = ["laycan 18–20 Oct 26"]
    evidence["evidence"]["/deal/product/as_written"] = ["Jet A-1"]
    repair = {"evidence": {"/deal/product/as_written": ["Jet A-1"]}}

    result = RecapExtractor(FakeRuntime(generated, evidence, repair)).extract(recap)

    assert result.deal.notes == []
    assert result.deal.pricing.period.expression == "BL-2..BL+2"
    assert result.deal.product.as_written == "Jet A-1"
    assert result.deal.product.product_code == "JETA1"
    assert result.deal.product.reference_name == "Jet A-1"
    assert result.deal.quality_spec is None
    assert result.deal.quantity.tolerance_option == "seller"


def test_calendar_period_expression_is_serialized_from_typed_dates() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "04_calendar_low.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_04.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture["expected"])), recap)
    generated["deal"]["pricing"]["period"]["expression"] = "20-24 October 2026"

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.deal.pricing.period.expression == "2026-10-20..2026-10-24"


def test_explicit_pricing_syntax_is_normalized_before_validation() -> None:
    expected = json.loads((ROOT / "evals" / "cases" / "01_clean_telegram.json").read_text())[
        "expected"
    ]
    recap = (ROOT / "data" / "recaps" / "recap_01.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(expected)), recap)
    generated["deal"]["pricing"]["period"] = None
    generated["deal"]["pricing"]["differential"]["value"] = 0
    generated["questions"] = [{"field": "/pricing/period", "question": "What is the period?"}]

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.questions == []
    assert result.deal.pricing.period.expression == "BL+0..BL+3"
    assert result.deal.pricing.differential.value == -12.5
    differential_evidence = next(
        entry.evidence
        for entry in result.provenance
        if entry.field == "/deal/pricing/differential/value"
    )
    assert "minus USD 12.50/MT" in {
        recap[item.start : item.end] for item in differential_evidence if item.kind == "source"
    }


def test_ambiguous_recap_values_abstain_and_get_canonical_questions() -> None:
    expected = json.loads(
        (ROOT / "evals" / "cases" / "03_incomplete_must_abstain.json").read_text()
    )["expected"]
    recap = (ROOT / "data" / "recaps" / "recap_03.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(expected)), recap)
    generated["deal"]["buyer"] = "us"
    generated["deal"]["quantity"]["value"] = 15_000
    generated["deal"]["delivery_window"] = {
        "from": "2026-10-15",
        "to": "2026-10-31",
    }
    generated["deal"]["payment_terms"] = {
        "as_written": "as usual",
        "payments": [
            {
                "portion": {"kind": None, "percentage": None},
                "stage": None,
                "instrument": None,
                "irrevocable": None,
                "due": None,
                "document_condition": None,
            }
        ],
        "late_interest": None,
    }
    generated["deal"]["quality_spec"] = "0.5"
    generated["questions"] = [{"field": "/pricing/currency", "question": "What is the currency?"}]

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.deal.buyer is None
    assert result.deal.quantity.value is None
    assert result.deal.delivery_window.from_date is None
    assert result.deal.delivery_window.to_date is None
    assert result.deal.payment_terms is None
    assert result.deal.quality_spec is None
    assert result.deal.notes == ["2-3 cargoes of about 5,000 mt each"]
    assert all(question.field.startswith("/deal/") for question in result.questions)
    assert {question.field for question in result.questions} == {
        question["field"] for question in expected["questions"]
    }


def test_forwarded_recap_terms_are_canonicalized() -> None:
    expected = json.loads((ROOT / "evals" / "cases" / "02_forwarded_amendments.json").read_text())[
        "expected"
    ]
    recap = (ROOT / "data" / "recaps" / "recap_02.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(expected)), recap)
    generated["questions"] = [
        {"field": "/deal/pricing/quote_code", "question": "Which quotation applies?"}
    ]
    generated["deal"]["pricing"]["quote_code"] = None
    generated["deal"]["fx"]["basis"] = "fixed agreed rate"
    generated["deal"]["inspection"]["location"] = "at discharge"
    generated["deal"]["notes"] = [
        "inspector is Saybolt not Intertek, appointed by us, at our cost as agreed"
    ]
    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    assert result.questions == []
    assert result.deal.fx.basis == "fixed_agreed"
    assert result.deal.inspection.location == "discharge"
    assert result.deal.notes == ["Rest as per our frame agreement of 14.03.2026."]
    assert result.deal.payment_terms is not None
    assert len(result.deal.payment_terms.payments) == 2
    payment_evidence = next(
        entry.evidence
        for entry in result.provenance
        if entry.field == "/deal/payment_terms"
    )
    payment_quotes = [
        recap[item.start : item.end]
        for item in payment_evidence
        if item.kind == "source"
    ]
    assert len(payment_quotes) == 2
    assert all("\n" not in quote for quote in payment_quotes)
    assert any("90 pct provisional" in quote for quote in payment_quotes)
    assert any("balance within 5 days" in quote for quote in payment_quotes)


def test_payment_parser_removes_model_inferences_not_present_in_recap() -> None:
    fixture = json.loads(
        (ROOT / "evals" / "cases" / "02_forwarded_amendments.json").read_text()
    )
    recap = (ROOT / "data" / "recaps" / "recap_02.txt").read_text()
    generated, evidence = _generated(json.loads(json.dumps(fixture["expected"])), recap)
    first, balance = generated["deal"]["payment_terms"]["payments"]
    first["instrument"] = "cash"
    first["irrevocable"] = False
    balance["portion"] = {"kind": "percentage", "percentage": 10}
    balance["instrument"] = "cash"
    balance["due"]["day_basis"] = "banking_days"

    result = RecapExtractor(FakeRuntime(generated, evidence)).extract(recap)

    first_result, balance_result = result.deal.payment_terms.payments
    assert first_result.instrument is None
    assert first_result.irrevocable is None
    assert balance_result.portion.kind == "balance"
    assert balance_result.portion.percentage is None
    assert balance_result.instrument is None
    assert balance_result.due.day_basis == "days"


@pytest.mark.parametrize(
    "case_path",
    sorted((ROOT / "evals" / "cases").glob("*.json")),
    ids=lambda path: path.stem,
)
def test_verbatim_evidence_builds_provenance_for_every_eval_recap(
    case_path: Path,
) -> None:
    fixture = json.loads(case_path.read_text())
    recap = (ROOT / "data" / "recaps" / Path(fixture["source"]["path"]).name).read_text()
    result = RecapExtractor(FakeRuntime(*_generated(fixture["expected"], recap))).extract(recap)
    actual = result.model_dump(mode="json", by_alias=True)

    assert actual["deal"] == fixture["expected"]["deal"]
    assert {question["field"] for question in actual["questions"]} == {
        question["field"] for question in fixture["expected"]["questions"]
    }
    assert set(actual) == {"deal", "provenance", "questions"}
    for entry in actual["provenance"]:
        for evidence in entry["evidence"]:
            if evidence["kind"] == "source":
                assert recap[evidence["start"] : evidence["end"]]


def test_non_verbatim_model_evidence_gets_one_bounded_repair() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "01_clean_telegram.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_01.txt").read_text()
    generated, good = _generated(fixture["expected"], recap)
    bad = json.loads(json.dumps(good))
    bad["evidence"]["/deal/buyer"][0] += " [not in source]"
    repaired = {"evidence": {"/deal/buyer": good["evidence"]["/deal/buyer"]}}
    runtime = FakeRuntime(generated, bad, repaired)

    result = RecapExtractor(runtime).extract(recap)

    assert result.deal.buyer == "Helvig Energy AG"
    assert len(runtime.calls) == 3
    assert "one evidence repair attempt" in runtime.calls[2][0][0]["content"].lower()


def test_one_invalid_fragment_repairs_the_whole_multiline_payment_claim() -> None:
    fixture = json.loads(
        (ROOT / "evals" / "cases" / "02_forwarded_amendments.json").read_text()
    )
    recap = (ROOT / "data" / "recaps" / "recap_02.txt").read_text()
    generated, good = _generated(fixture["expected"], recap)
    bad = json.loads(json.dumps(good))
    payment_quotes = bad["evidence"]["/deal/payment_terms"]
    assert len(payment_quotes) == 2
    payment_quotes[1] += " [not in source]"
    repaired = {
        "evidence": {
            "/deal/payment_terms": good["evidence"]["/deal/payment_terms"],
        }
    }

    result = RecapExtractor(FakeRuntime(generated, bad, repaired)).extract(recap)

    assert result.deal.payment_terms is not None
    payment_entry = next(
        entry for entry in result.provenance if entry.field == "/deal/payment_terms"
    )
    assert len([item for item in payment_entry.evidence if item.kind == "source"]) == 2


def test_evidence_fragment_cannot_cross_a_physical_source_line() -> None:
    source = "first payment clause\nsecond payment clause"
    claim = ModelEvidenceClaim(
        quote=source,
        fields=["/deal/payment_terms"],
    )

    with pytest.raises(EvidenceValidationError, match="physical source line"):
        locate_model_evidence([claim], source)


def test_invalid_evidence_repair_rejects_the_extraction() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "01_clean_telegram.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_01.txt").read_text()
    generated, good = _generated(fixture["expected"], recap)
    bad = {"evidence": {"/deal/buyer": good["evidence"]["/deal/buyer"]}}
    bad["evidence"]["/deal/buyer"][0] += " [not in source]"
    runtime = FakeRuntime(generated, bad, bad)

    with pytest.raises(ExtractionFailedError, match="evidence was invalid"):
        RecapExtractor(runtime).extract(recap)

    assert len(runtime.calls) == 3


def test_normalized_natural_language_dates_keep_their_original_source_span() -> None:
    fixture = json.loads((ROOT / "evals" / "cases" / "06_fixed_fx.json").read_text())
    recap = (ROOT / "data" / "recaps" / "recap_06.txt").read_text()

    result = RecapExtractor(FakeRuntime(*_generated(fixture["expected"], recap))).extract(recap)

    evidence = next(
        entry.evidence for entry in result.provenance if entry.field == "/deal/delivery_window/from"
    )
    source_text = " ".join(
        recap[item.start : item.end] for item in evidence if item.kind == "source"
    )
    assert "twenty-second through the twenty-fifth of October 2026" in source_text
