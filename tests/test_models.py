from datetime import date

import pytest
from pydantic import ValidationError

from optiflow_ctrm.models import ExtractionResult


def _minimal_complete() -> dict:
    return {
        "deal": {
            "recap_date": "2026-08-11",
            "buyer": "Buyer Ltd",
            "seller": "Seller Ltd",
            "product": {
                "as_written": "Gasoil",
                "product_code": None,
                "reference_name": None,
            },
            "quantity": {
                "value": 30_000,
                "unit": "MT",
                "tolerance_pct": 10,
                "tolerance_option": "seller",
            },
            "delivery_term": {"as_written": "FOB Augusta", "incoterm": "FOB", "place": "Augusta"},
            "delivery_window": {"from": "2026-09-02", "to": "2026-09-06"},
            "pricing": {
                "quote_code": "QUOTE_1",
                "statistic": "mean",
                "period": {
                    "kind": "bl_relative",
                    "expression": "BL+0..BL+3",
                    "from_offset_days": 0,
                    "to_offset_days": 3,
                },
                "differential": {"value": -12.5, "unit": "USD/MT"},
            },
            "payment_terms": {
                "as_written": "five days from B/L",
                "payments": [
                    {
                        "portion": {"kind": None, "percentage": None},
                        "stage": None,
                        "instrument": None,
                        "irrevocable": None,
                        "due": {
                            "qualifier": None,
                            "days": 5,
                            "day_basis": "days",
                            "relation": "from",
                            "event": "bill_of_lading",
                        },
                        "document_condition": None,
                    }
                ],
                "late_interest": None,
            },
            "currency": "USD",
            "fx": None,
            "quality_spec": None,
            "inspection": None,
            "demurrage": None,
            "law": None,
            "arbitration": None,
            "vessel": None,
            "notes": [],
        },
        "questions": [],
        "provenance": [],
    }


def test_complete_deal_parses_typed_dates_and_preserves_units() -> None:
    result = ExtractionResult.model_validate(_minimal_complete())

    assert result.deal.recap_date == date(2026, 8, 11)
    assert result.deal.quantity.unit == "MT"
    assert result.deal.pricing.differential.value == -12.5


def test_missing_mandatory_value_requires_matching_question() -> None:
    value = _minimal_complete()
    value["deal"]["buyer"] = None

    with pytest.raises(ValidationError, match="/deal/buyer"):
        ExtractionResult.model_validate(value)

    value["questions"] = [{"field": "/deal/buyer", "question": "What is the buyer's legal name?"}]
    assert ExtractionResult.model_validate(value).deal.buyer is None


def test_non_usd_deal_requires_fx_or_questions() -> None:
    value = _minimal_complete()
    value["deal"]["currency"] = "EUR"
    value["questions"] = [
        {"field": "/deal/fx/rate", "question": "What agreed FX rate applies?"},
        {"field": "/deal/fx/basis", "question": "What is the agreed FX basis?"},
    ]

    assert ExtractionResult.model_validate(value).deal.fx is None


def test_rejects_invalid_incoterm_and_backwards_windows() -> None:
    value = _minimal_complete()
    value["deal"]["delivery_term"]["incoterm"] = "EX TANK"
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(value)

    value = _minimal_complete()
    value["deal"]["delivery_window"] = {"from": "2026-09-07", "to": "2026-09-06"}
    with pytest.raises(ValidationError, match="delivery window"):
        ExtractionResult.model_validate(value)


def test_constrained_schema_requires_every_canonical_output_key() -> None:
    from optiflow_ctrm.models import (
        canonical_json_schema,
        evidence_json_schema,
        extraction_json_schema,
    )

    schema = extraction_json_schema()
    assert set(schema["properties"]) == {"deal", "questions"}
    assert set(schema["required"]) == set(schema["properties"])
    for definition in schema["$defs"].values():
        if "properties" in definition:
            assert set(definition["required"]) == set(definition["properties"])

    evidence_schema = evidence_json_schema()
    assert set(evidence_schema["properties"]) == {"evidence"}
    assert evidence_schema["properties"]["evidence"]["maxProperties"] == 21
    targeted = evidence_json_schema(["/deal/buyer", "/deal/seller"])
    targeted_evidence = targeted["properties"]["evidence"]
    assert set(targeted_evidence["properties"]) == {"/deal/buyer", "/deal/seller"}
    assert set(targeted_evidence["required"]) == {"/deal/buyer", "/deal/seller"}
    assert targeted_evidence["additionalProperties"] is False
    assert targeted_evidence["properties"]["/deal/buyer"]["type"] == "array"
    assert targeted_evidence["properties"]["/deal/buyer"]["minItems"] == 1
    assert set(canonical_json_schema()["properties"]) == {
        "deal",
        "provenance",
        "questions",
    }
    assert "evidence" not in canonical_json_schema()["properties"]


def test_provenance_is_a_constrained_list_not_an_open_ended_object() -> None:
    from optiflow_ctrm.models import canonical_json_schema, extraction_json_schema

    inference_schema = extraction_json_schema()
    assert "provenance" not in inference_schema["properties"]
    schema = canonical_json_schema()
    assert schema["properties"]["provenance"]["type"] == "array"
    assert schema["properties"]["provenance"]["items"]["$ref"].endswith("/ProvenanceEntry")


def test_clarification_fields_are_constrained_to_canonical_deal_paths() -> None:
    from optiflow_ctrm.models import extraction_json_schema

    schema = extraction_json_schema()
    field_schema = schema["$defs"]["ClarificationQuestion"]["properties"]["field"]

    assert "/deal/pricing/period" in field_schema["enum"]
    assert "/pricing/period" not in field_schema["enum"]
    assert "/product/product_code" not in field_schema["enum"]
