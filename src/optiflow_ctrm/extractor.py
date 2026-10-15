"""Extraction orchestration.

This module assembles the prompt, calls the local runtime, validates the typed
result, and coordinates at most one repair attempt. It must not calculate prices
or silently fill omitted commercial values.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from optiflow_ctrm.models import (
    GeneratedExtraction,
    ExtractionResult,
    ModelEvidenceClaim,
    ModelEvidenceResponse,
    evidence_json_schema,
    extraction_json_schema,
)
from optiflow_ctrm.provenance import (
    EvidenceValidationError,
    SourceSpanMap,
    add_source_span,
    build_source_provenance,
    find_unique_source_span,
    locate_model_evidence,
    locate_wrapped_evidence,
    commercial_source,
    merge_source_spans,
)
from optiflow_ctrm.references import ReferenceCatalog
from optiflow_ctrm.runtime import (
    InferenceProtocolError,
    InferenceRuntime,
    create_default_runtime,
)
from optiflow_ctrm.validation import normalize_generated_output


SYSTEM_PROMPT = """You extract one physical oil deal from a recap into the supplied JSON schema.

Safety and extraction rules:
- Use only facts explicitly present in RECAP SOURCE. Never supply a customary or likely value.
- Every schema key must be returned. Use null for an absent or genuinely ambiguous value.
- For every null mandatory value, add one concise, field-specific question that a trader can
  answer in one line.
- Preserve quantity and differential units exactly as agreed. Do not convert BBL and MT.
- A price discount is negative and a premium is positive, but do not invent either.
- Only EXW, FAS, FOB, CFR, CIF, DAP, and DDP are Incoterms. Preserve other wording in
  delivery_term.as_written, set incoterm to null, and ask for clarification.
- In a thread, later explicit amendments win. Ignore unrelated deals mentioned in signatures,
  forwards, or side discussions.
- Do not invent product or quotation codes. Put the product wording in product.as_written and
  the quotation name/code as written in pricing.quote_code; deterministic code resolves them.
- Do not emit evidence, provenance, or offsets in this call. Deterministic application code
  requests compact verbatim evidence separately and converts verified quotations into source
  offsets.
- product.as_written contains only the product wording. Put its specification in quality_spec.
- payment_terms preserves the complete wording in as_written and splits each explicitly stated
  payment into payments. Do not derive a percentage for "balance" or infer an unstated payment
  instrument, day basis, stage, deadline qualifier, or document type.
- notes contains only commercially material terms; omit confirmations and paperwork requests.
- Return JSON only. Do not include Markdown or commentary.
"""

SOURCE_MARKER = "RECAP SOURCE STARTS ON THE NEXT LINE (offset 0):\n"

EVIDENCE_SYSTEM_PROMPT = """You locate evidence for an already extracted oil deal.

Return only the supplied evidence JSON object. Every TARGET FIELD is a required key under
evidence, and its value is an array of one or more supporting quotations. Each quotation must be
a unique, exact, case-sensitive, verbatim substring copied from RECAP SOURCE. Use multiple array
items when a value is supported by multiple physical source lines. Never paraphrase, normalise,
add punctuation, or include a field that is not in TARGET FIELDS. Reuse quotations for fields
supported by the same clause. Every individual quotation must stay within one physical source
line: never remove or replace a line break. Do not return the deal itself.
"""


class ExtractionFailedError(RuntimeError):
    """Constrained extraction or its single bounded repair failed validation."""


class RecapExtractor:
    """Generate and validate one canonical extraction from unmodified recap text."""

    def __init__(
        self,
        runtime: InferenceRuntime | None = None,
        references: ReferenceCatalog | None = None,
    ) -> None:
        self.runtime = runtime or create_default_runtime()
        self.references = references or ReferenceCatalog.load()

    def extract(self, text: str) -> ExtractionResult:
        if not isinstance(text, str):
            raise TypeError("recap text must be a string")
        if not text:
            raise ValueError("recap text must not be empty")

        schema = extraction_json_schema()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SOURCE_MARKER + text},
        ]

        first_value: Any | None = None
        try:
            first = self.runtime.generate_json(messages, schema)
            first_value = first.value
            normalized, parser_spans = normalize_generated_output(first.value, text)
            parsed = GeneratedExtraction.model_validate(normalized)
        except (ValidationError, InferenceProtocolError) as first_error:
            repair_messages = self._repair_messages(messages, first_value, first_error)
            try:
                repaired = self.runtime.generate_json(repair_messages, schema)
                normalized, parser_spans = normalize_generated_output(repaired.value, text)
                parsed = GeneratedExtraction.model_validate(normalized)
            except (ValidationError, InferenceProtocolError) as second_error:
                raise ExtractionFailedError(
                    "model output remained invalid after one repair attempt"
                ) from second_error

        try:
            return self._finalize(parsed, text, parser_spans)
        except (ValidationError, InferenceProtocolError, EvidenceValidationError) as error:
            raise ExtractionFailedError("model evidence was invalid") from error

    def _finalize(
        self,
        extraction: GeneratedExtraction,
        text: str,
        parser_spans: SourceSpanMap,
    ) -> ExtractionResult:
        value = extraction.model_dump(mode="json", by_alias=True)
        source_spans = merge_source_spans(parser_spans)
        enrichment_spans: SourceSpanMap = {}
        product_code, quote_code = self._enrich_references(value, text, enrichment_spans)
        value, source_spans = normalize_generated_output(value, text, source_spans)
        targets = _evidence_targets(value["deal"], source_spans)
        source_spans = merge_source_spans(source_spans, enrichment_spans)
        already_proven = {
            field for field in targets if source_spans.get(field)
        }
        if targets:
            evidence_messages = _evidence_messages(targets, text)
            evidence_result = self.runtime.generate_json(
                evidence_messages, evidence_json_schema(targets)
            )
            evidence = ModelEvidenceResponse.model_validate(evidence_result.value)
            model_spans = _locate_valid_claims(evidence, set(targets), text)
            missing = set(targets) - set(model_spans) - already_proven
            if missing:
                repair_targets = {field: targets[field] for field in sorted(missing)}
                repaired_result = self.runtime.generate_json(
                    _evidence_messages(repair_targets, text, repair=True),
                    evidence_json_schema(repair_targets),
                )
                repaired = ModelEvidenceResponse.model_validate(repaired_result.value)
                _validate_evidence_targets(repaired, missing)
                repaired_spans = _locate_valid_claims(repaired, missing, text)
                model_spans = merge_source_spans(model_spans, repaired_spans)
                unresolved = missing - set(repaired_spans)
                if unresolved:
                    _reject_unproven_fields(value["deal"], unresolved, source_spans)
                    value, source_spans = normalize_generated_output(
                        value, text, source_spans
                    )
                    if "/deal/product/as_written" in unresolved:
                        product_code = None
                    if "/deal/pricing/quote_code" in unresolved:
                        quote_code = None
            source_spans = merge_source_spans(source_spans, model_spans)
        _remove_structured_duplicate_notes(value["deal"], text, source_spans)
        value["provenance"] = build_source_provenance(
            value,
            text,
            source_spans,
            grouped_paths={"/deal/payment_terms"},
        )
        if product_code is not None:
            _add_reference_evidence(
                value,
                "/deal/product/product_code",
                "data/products.csv",
                product_code,
                copy_source_from="/deal/product/as_written",
            )
            _add_reference_evidence(
                value,
                "/deal/product/reference_name",
                "data/products.csv",
                product_code,
                copy_source_from="/deal/product/as_written",
            )
        if quote_code is not None:
            _add_reference_evidence(
                value,
                "/deal/pricing/quote_code",
                "data/price_quotes.csv",
                quote_code,
            )
        return ExtractionResult.model_validate(value)

    def _enrich_references(
        self,
        value: dict[str, Any],
        text: str,
        source_spans: SourceSpanMap,
    ) -> tuple[str | None, str | None]:
        text = commercial_source(text)
        product_value = value["deal"]["product"]["as_written"]
        product = self.references.resolve_product(product_value)
        product_split: tuple[str, str, int] | None = None
        if isinstance(product_value, str):
            for separator in re.finditer(r",\s*", product_value):
                product_wording = product_value[: separator.start()].rstrip()
                specification = product_value[separator.end() :].strip()
                prefix_product = self.references.resolve_product(product_wording)
                if prefix_product is not None and specification:
                    product = prefix_product
                    specification_at = product_value.find(specification, separator.end())
                    product_split = (product_wording, specification, specification_at)
                    break
        if product is not None:
            if isinstance(product_value, str) and product_split is not None:
                product_wording, specification, specification_at = product_split
                value["deal"]["product"]["as_written"] = product_wording
                value["deal"]["quality_spec"] = specification
                combined_span = find_unique_source_span(text, product_value)
                if combined_span is not None:
                    start, _ = combined_span
                    add_source_span(
                        source_spans,
                        "/deal/product/as_written",
                        (start, start + len(product_wording)),
                    )
                    add_source_span(
                        source_spans,
                        "/deal/quality_spec",
                        (start + specification_at, start + specification_at + len(specification)),
                    )
            value["deal"]["product"]["product_code"] = product.code
            value["deal"]["product"]["reference_name"] = product.name
            if not source_spans.get("/deal/product/as_written"):
                product_wording = value["deal"]["product"]["as_written"]
                if isinstance(product_wording, str):
                    product_span = _find_product_clause_span(text, product_wording)
                    if product_span is not None:
                        add_source_span(
                            source_spans,
                            "/deal/product/as_written",
                            product_span,
                        )

        quote_value = value["deal"]["pricing"]["quote_code"]
        quote = self.references.resolve_quote(quote_value)
        if quote is None and "price tbc" not in text.casefold():
            quote = self.references.resolve_quote(text)
        if quote is not None:
            value["deal"]["pricing"]["quote_code"] = quote.code
            if not source_spans.get("/deal/pricing/quote_code"):
                span = find_unique_source_span(text, quote.name)
                if span is not None:
                    add_source_span(source_spans, "/deal/pricing/quote_code", span)

        return (
            product.code if product is not None else None,
            quote.code if quote is not None else None,
        )

    @staticmethod
    def _repair_messages(
        original_messages: list[dict[str, str]],
        invalid_value: Any | None,
        error: ValidationError | InferenceProtocolError | EvidenceValidationError,
    ) -> list[dict[str, str]]:
        messages = list(original_messages)
        if invalid_value is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(invalid_value, ensure_ascii=False, separators=(",", ":")),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Make one final repair. Return the entire corrected JSON object and change only "
                    "what is needed to satisfy these validation errors:\n"
                    + _safe_error_summary(error)
                ),
            }
        )
        return messages


def _safe_error_summary(
    error: ValidationError | InferenceProtocolError | EvidenceValidationError,
) -> str:
    """Report actionable validation failures without echoing Pydantic input data."""

    if isinstance(error, ValidationError):
        details = [
            {
                "location": "/".join(str(part) for part in item["loc"]),
                "type": item["type"],
                "message": item["msg"],
            }
            for item in error.errors(include_url=False, include_context=False, include_input=False)
        ]
        return json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    return str(error)


def _find_product_clause_span(source: str, product_wording: str) -> tuple[int, int] | None:
    """Locate a unique product occurrence in a product-shaped physical line."""

    pattern = re.compile(
        r"^[ \t]*(?:>[ \t]*)*(?:(?:product|grade)\s*[:=—-]\s*)?"
        rf"(?P<product>{re.escape(product_wording)})(?=[ \t]*(?:[/,;]|$))",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        return None
    return matches[0].span("product")


def _remove_structured_duplicate_notes(
    deal: dict[str, Any], source: str, source_spans: SourceSpanMap
) -> None:
    """Drop notes whose exact evidence span is already owned by a structured field."""

    notes = deal.get("notes")
    if not isinstance(notes, list) or not notes:
        return
    structured_spans: set[tuple[int, int]] = set()
    for field, spans in source_spans.items():
        if field.startswith("/deal/notes"):
            continue
        owner = _structured_evidence_owner(field)
        if _fully_populated(_pointer_value(deal, owner)):
            structured_spans.update(spans)
    note_evidence = set(source_spans.get("/deal/notes", []))
    kept: list[Any] = []
    removed_spans: set[tuple[int, int]] = set()
    for note in notes:
        # A shared quotation can contain a structured fact AND an unstructured
        # dependency. Evidence overlap alone cannot justify deleting the latter.
        if isinstance(note, str) and re.search(
            r"\b(?:subject\s+to|if|unless|once|pending|conditional|after|until)\b", note, re.I
        ):
            kept.append(note)
            continue
        span = find_unique_source_span(source, note) if isinstance(note, str) else None
        if span is not None and span in note_evidence and span in structured_spans:
            removed_spans.add(span)
        else:
            kept.append(note)
    if len(kept) == len(notes):
        return
    deal["notes"] = kept
    remaining = [
        span for span in source_spans.get("/deal/notes", []) if span not in removed_spans
    ]
    if remaining:
        source_spans["/deal/notes"] = remaining
    else:
        source_spans.pop("/deal/notes", None)


def _structured_evidence_owner(field: str) -> str:
    groups = (
        "/deal/quantity",
        "/deal/delivery_term",
        "/deal/delivery_window",
        "/deal/pricing/period",
        "/deal/pricing/differential",
        "/deal/payment_terms",
        "/deal/fx",
        "/deal/inspection",
        "/deal/demurrage",
    )
    return next((group for group in groups if field.startswith(group)), field)


def _fully_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(value) and all(_fully_populated(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_fully_populated(item) for item in value)
    return True


def _evidence_targets(
    deal: dict[str, Any], source_spans: SourceSpanMap | None = None
) -> dict[str, Any]:
    """Return compact semantic field groups and their extracted values."""

    targets: dict[str, Any] = {}
    for path, value in _flatten_deal(deal, "/deal").items():
        if value is None or path in {
            "/deal/product/product_code",
            "/deal/product/reference_name",
        }:
            continue
        field = _evidence_field(path)
        if source_spans and source_spans.get(field):
            continue
        targets.setdefault(field, _pointer_value(deal, field))
    return dict(sorted(targets.items()))


def _validate_evidence_targets(
    evidence: ModelEvidenceResponse, targets: set[str]
) -> None:
    claimed = set(evidence.evidence)
    missing = sorted(targets - claimed)
    unexpected = sorted(claimed - targets)
    if missing:
        raise EvidenceValidationError(f"model evidence omitted target fields: {missing}")
    if unexpected:
        raise EvidenceValidationError(
            f"model evidence included fields that were not requested: {unexpected}"
        )


def _locate_valid_claims(
    evidence: ModelEvidenceResponse,
    targets: set[str],
    source: str,
) -> SourceSpanMap:
    """Keep valid first-pass claims; missing or malformed claims are repaired once."""

    spans: SourceSpanMap = {}
    source = commercial_source(source)
    for field, quotes in evidence.evidence.items():
        if field not in targets:
            continue
        field_spans: SourceSpanMap = {}
        try:
            for quote in quotes:
                claim = ModelEvidenceClaim(quote=quote, fields=[field])
                try:
                    located = locate_model_evidence([claim], source)
                except EvidenceValidationError:
                    wrapped = locate_wrapped_evidence(quote, source)
                    if not wrapped:
                        raise
                    located = {field: wrapped}
                field_spans = merge_source_spans(field_spans, located)
        except EvidenceValidationError:
            continue
        for span in field_spans[field]:
            add_source_span(spans, field, span)
    return spans


def _evidence_messages(
    targets: dict[str, Any], text: str, *, repair: bool = False
) -> list[dict[str, str]]:
    instruction = "MISSING OR INVALID TARGET FIELDS" if repair else "TARGET FIELDS"
    system_prompt = EVIDENCE_SYSTEM_PROMPT
    if repair:
        system_prompt += (
            "\nThis is the one evidence repair attempt. Return quotations only for the listed "
            "missing or invalid fields. Check every character and preserve line breaks exactly."
        )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                instruction
                + ":\n"
                + json.dumps(targets, ensure_ascii=False, separators=(",", ":"))
                + "\n\n"
                + SOURCE_MARKER
                + text
            ),
        },
    ]


def _reject_unproven_fields(
    deal: dict[str, Any], fields: set[str], source_spans: SourceSpanMap | None = None
) -> None:
    """Abstain only from field groups still lacking verbatim evidence after repair."""

    structured_groups = {
        "/deal/quantity",
        "/deal/delivery_term",
        "/deal/delivery_window",
        "/deal/pricing/differential",
    }
    nullable_groups = {"/deal/fx", "/deal/inspection", "/deal/demurrage"}
    source_spans = source_spans or {}
    for field in fields:
        if source_spans.get(field):
            continue
        relative = field.removeprefix("/deal/")
        parts = relative.split("/")
        parent: Any = deal
        for part in parts[:-1]:
            parent = parent.get(part) if isinstance(parent, dict) else None
        if not isinstance(parent, dict):
            continue
        key = parts[-1]
        if field in structured_groups | nullable_groups and isinstance(parent.get(key), dict):
            parent[key] = {
                item: content if source_spans.get(f"{field}/{item}") else None
                for item, content in parent[key].items()
            }
            if field in nullable_groups and all(item is None for item in parent[key].values()):
                parent[key] = None
        elif field == "/deal/notes":
            parent[key] = []
        elif field in nullable_groups:
            parent[key] = None
        else:
            parent[key] = None

    if "/deal/product/as_written" in fields:
        deal["product"]["product_code"] = None
        deal["product"]["reference_name"] = None


def _evidence_field(path: str) -> str:
    grouped = (
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
    return next((prefix for prefix in grouped if path.startswith(prefix)), path)


def _flatten_deal(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            flattened.update(_flatten_deal(item, f"{path}/{escaped}"))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_deal(item, f"{path}/{index}"))
        return flattened
    return {path: value}


def _pointer_value(deal: dict[str, Any], field: str) -> Any:
    value: Any = deal
    for part in field.removeprefix("/deal/").split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def _add_reference_evidence(
    extraction: dict[str, Any],
    field: str,
    file: str,
    key: str,
    *,
    copy_source_from: str | None = None,
) -> None:
    entries = extraction["provenance"]
    entry = next((item for item in entries if item["field"] == field), None)
    if entry is None:
        evidence = []
        if copy_source_from is not None:
            source_entry = next(
                (item for item in entries if item["field"] == copy_source_from), None
            )
            if source_entry is not None:
                evidence.extend(
                    item for item in source_entry["evidence"] if item["kind"] == "source"
                )
        entry = {"field": field, "evidence": evidence}
        entries.append(entry)
    reference = {"kind": "reference", "file": file, "key": key}
    if reference not in entry["evidence"]:
        entry["evidence"].append(reference)
    entries.sort(key=lambda item: item["field"])


def extract_recap(
    text: str,
    runtime: InferenceRuntime | None = None,
) -> ExtractionResult:
    """Convenience entry point shared by CLI/MCP callers."""

    return RecapExtractor(runtime).extract(text)


# Task terminology and MCP tool naming use both verbs; keep them on one path.
parse_recap = extract_recap
