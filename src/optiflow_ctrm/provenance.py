"""Resolve verbatim evidence against the unchanged recap and build provenance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias
import re

from optiflow_ctrm.models import ModelEvidenceClaim


SourceSpan: TypeAlias = tuple[int, int]
SourceSpanMap: TypeAlias = dict[str, list[SourceSpan]]


class EvidenceValidationError(ValueError):
    """A generated value lacks a unique verbatim quotation in the source."""


def commercial_source(source: str) -> str:
    """Mask an explicitly excluded trailing section without shifting any offsets."""
    marker = re.search(r"(?im)^UNRELATED[^\r\n]*(?:do not merge|not part of)[^\r\n]*", source)
    if marker is None:
        return source
    return source[:marker.start()] + re.sub(r"[^\r\n]", " ", source[marker.start():])


def physical_spans(source: str, start: int, end: int) -> list[SourceSpan]:
    return [(start + m.start(), start + m.end())
            for m in re.finditer(r"[^\r\n]+", source[start:end]) if m.group().strip()]


def locate_wrapped_evidence(quote: str, source: str) -> list[SourceSpan]:
    """Recover only whitespace/quote-prefix wrapping, never changed words or case.

    Match against the original text and return physical spans. Paragraph boundaries
    and message headers cannot be silently traversed.
    """
    words = re.split(r"\s+", quote.strip())
    if not words:
        return []
    join = r"(?:[ \t]+|[ \t]*\r?\n[ \t]*(?:>[ \t]*)*)"
    matches = list(re.finditer(join.join(re.escape(word) for word in words), source))
    if len(matches) != 1:
        return []
    match = matches[0]
    spans = physical_spans(source, *match.span())
    result = []
    for start, end in spans:
        prefix = re.match(r"[ \t]*(?:>[ \t]*)*", source[start:end])
        start += prefix.end()
        end = start + len(source[start:end].rstrip())
        if end > start:
            result.append((start, end))
    return result


def locate_model_evidence(claims: Iterable[ModelEvidenceClaim], source: str) -> SourceSpanMap:
    """Convert model-provided verbatim quotations to offsets deterministically."""

    spans: SourceSpanMap = {}
    for claim in claims:
        if "\n" in claim.quote or "\r" in claim.quote:
            raise EvidenceValidationError(
                "model evidence quotation must stay within one physical source line"
            )
        occurrences = _occurrences(source, claim.quote)
        if not occurrences:
            raise EvidenceValidationError(
                f"model evidence quote was not found verbatim in recap: {claim.quote!r}"
            )
        if len(occurrences) > 1:
            raise EvidenceValidationError(
                f"model evidence quote is not unique in recap: {claim.quote!r}"
            )
        start = occurrences[0]
        span = (start, start + len(claim.quote))
        for field in claim.fields:
            add_source_span(spans, field, span)
    return spans


def build_source_provenance(
    extraction: Mapping[str, Any],
    source: str,
    source_spans: Mapping[str, list[SourceSpan]],
    *,
    grouped_paths: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build leaf-level provenance from verified model and parser spans."""

    entries: list[dict[str, Any]] = []
    groups = set(grouped_paths)
    for path in sorted(groups):
        value = _pointer_value(extraction, path)
        if value is None:
            continue
        spans = _spans_for_path(source_spans, path)
        _validate_spans(path, spans, source)
        entries.append(_provenance_entry(path, spans))

    for path, value in sorted(_flatten(extraction.get("deal"), "/deal").items()):
        if value is None or path in {
            "/deal/product/product_code",
            "/deal/product/reference_name",
        }:
            continue
        if any(path.startswith(f"{group}/") for group in groups):
            continue
        spans = _spans_for_path(source_spans, path)
        _validate_spans(path, spans, source)
        entries.append(_provenance_entry(path, spans))
    return sorted(entries, key=lambda item: item["field"])


def _validate_spans(path: str, spans: list[SourceSpan], source: str) -> None:
    if not spans:
        raise EvidenceValidationError(f"no verified source evidence for {path}")
    for start, end in spans:
        if start < 0 or end <= start or end > len(source):
            raise EvidenceValidationError(f"source evidence is outside recap for {path}")


def _provenance_entry(path: str, spans: list[SourceSpan]) -> dict[str, Any]:
    return {
        "field": path,
        "evidence": [
            {"kind": "source", "start": start, "end": end} for start, end in spans
        ],
    }


def _pointer_value(extraction: Mapping[str, Any], path: str) -> Any:
    value: Any = extraction
    for part in path.lstrip("/").split("/"):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part.replace("~1", "/").replace("~0", "~"))
    return value


def merge_source_spans(*span_maps: Mapping[str, list[SourceSpan]]) -> SourceSpanMap:
    """Merge verified span maps without changing source coordinates."""

    merged: SourceSpanMap = {}
    for spans in span_maps:
        for field, field_spans in spans.items():
            for span in field_spans:
                add_source_span(merged, field, span)
    return merged


def add_source_span(spans: SourceSpanMap, field: str, span: SourceSpan) -> None:
    values = spans.setdefault(field, [])
    if span not in values:
        values.append(span)
        values.sort()


def find_unique_source_span(source: str, quotation: str) -> SourceSpan | None:
    """Return a case-insensitive unique exact-text span for a reference name."""

    folded_source = source.casefold()
    folded_quote = quotation.casefold()
    starts = _occurrences(folded_source, folded_quote)
    if len(starts) != 1:
        return None
    start = starts[0]
    return start, start + len(quotation)


def _occurrences(source: str, quotation: str) -> list[int]:
    starts: list[int] = []
    offset = 0
    while True:
        start = source.find(quotation, offset)
        if start < 0:
            return starts
        starts.append(start)
        offset = start + 1


def _spans_for_path(source_spans: Mapping[str, list[SourceSpan]], path: str) -> list[SourceSpan]:
    candidate = path
    while candidate.startswith("/deal"):
        spans = source_spans.get(candidate)
        if spans:
            return list(spans)
        candidate = candidate.rsplit("/", 1)[0]
    return []


def _flatten(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.update(_flatten(item, f"{path}/{escaped}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{path}/{index}"))
        return result
    return {path: value}
