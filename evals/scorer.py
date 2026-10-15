"""Deterministic field-level extraction and abstention scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import ValidationError

from evals.normalization import values_equal
from optiflow_ctrm.models import ExtractionResult


_MISSING = object()


@dataclass(slots=True)
class ExtractionScore:
    total_fields: int = 0
    correct_values: int = 0
    incorrect_values: int = 0
    correct_abstentions: int = 0
    wrong_abstentions: int = 0
    unsupported_non_null: int = 0
    schema_valid: bool = False
    provenance_valid: bool = False
    expected_questions: int = 0
    matched_questions: int = 0
    incorrect_paths: list[str] = field(default_factory=list)
    wrong_abstention_paths: list[str] = field(default_factory=list)
    unsupported_paths: list[str] = field(default_factory=list)
    provenance_errors: list[str] = field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        if not self.total_fields:
            return 0.0
        return (self.correct_values + self.correct_abstentions) / self.total_fields

    @property
    def question_recall(self) -> float:
        if not self.expected_questions:
            return 1.0
        return self.matched_questions / self.expected_questions

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["field_accuracy"] = self.field_accuracy
        value["question_recall"] = self.question_recall
        return value


def score_extraction(
    expected: dict[str, Any], actual: dict[str, Any] | None, source: str
) -> ExtractionScore:
    """Compare one actual result to human-reviewed ground truth.

    ``unsupported_non_null`` is deliberately separate from ordinary incorrect
    values: it identifies a model supplying a value where ground truth requires
    abstention.
    """

    score = ExtractionScore()
    expected_fields = _flatten(expected.get("deal"), "/deal")
    score.total_fields = len(expected_fields)
    score.expected_questions = len(_question_paths(expected))
    if actual is None:
        for path, expected_value in expected_fields.items():
            if expected_value is not None:
                score.wrong_abstentions += 1
                score.wrong_abstention_paths.append(path)
        return score

    try:
        ExtractionResult.model_validate(actual)
        score.schema_valid = True
    except ValidationError:
        score.schema_valid = False

    for path, expected_value in expected_fields.items():
        actual_value = _pointer_get(actual, path)
        if actual_value is _MISSING:
            actual_value = None
        if expected_value is None:
            if actual_value is None:
                score.correct_abstentions += 1
            else:
                score.unsupported_non_null += 1
                score.unsupported_paths.append(path)
        elif actual_value is None:
            score.wrong_abstentions += 1
            score.wrong_abstention_paths.append(path)
        elif values_equal(expected_value, actual_value):
            score.correct_values += 1
        else:
            score.incorrect_values += 1
            score.incorrect_paths.append(path)

    expected_question_paths = _question_paths(expected)
    actual_question_paths = _question_paths(actual)
    score.matched_questions = len(expected_question_paths & actual_question_paths)
    score.provenance_errors = validate_provenance(actual, source)
    score.provenance_valid = not score.provenance_errors
    return score


def field_differences(
    expected: dict[str, Any], actual: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return a path-sorted, human-readable whole-result comparison."""

    differences: list[dict[str, Any]] = []
    expected_fields = _flatten(expected, "")
    for path, expected_value in sorted(expected_fields.items()):
        actual_value = _pointer_get(actual, path) if actual is not None else _MISSING
        if actual_value is _MISSING:
            actual_value = None
        if values_equal(expected_value, actual_value):
            continue
        if expected_value is None and actual_value is not None:
            classification = "unsupported_non_null"
        elif expected_value is not None and actual_value is None:
            classification = "wrong_abstention"
        else:
            classification = "incorrect_value"
        differences.append(
            {
                "actual": actual_value,
                "classification": classification,
                "expected": expected_value,
                "path": path,
            }
        )
    return differences


def validate_provenance(actual: dict[str, Any], source: str) -> list[str]:
    """Check source-span bounds and evidence coverage against original text."""

    errors: list[str] = []
    provenance = _provenance_map(actual.get("provenance"))
    if provenance is None:
        return ["provenance must be a list of field/evidence objects"]

    for path, evidence_items in provenance.items():
        if not isinstance(evidence_items, list) or not evidence_items:
            errors.append(f"{path}: evidence must be a non-empty list")
            continue
        for item in evidence_items:
            if not isinstance(item, dict):
                errors.append(f"{path}: evidence item must be an object")
                continue
            if item.get("kind") != "source":
                continue
            start, end = item.get("start"), item.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > len(source)
            ):
                errors.append(f"{path}: source span is outside the original recap")

    deal = actual.get("deal")
    if isinstance(deal, dict):
        for path, value in _flatten_with_lists(deal, "/deal").items():
            if value is None or path in {
                "/deal/product/product_code",
                "/deal/product/reference_name",
            }:
                continue
            evidence = _evidence_for_path(provenance, path)
            if not any(
                isinstance(item, dict) and item.get("kind") == "source" for item in evidence
            ):
                errors.append(f"{path}: non-null extracted value has no source evidence")
    return errors


def _flatten(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {path: value}
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{path}/{_escape(key)}"))
        return result
    # Lists such as notes are scored as one ordered commercial field.
    return {path: value}


def _flatten_with_lists(value: Any, path: str) -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten_with_lists(item, f"{path}/{_escape(key)}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_with_lists(item, f"{path}/{index}"))
        return result
    return {path: value}


def _pointer_get(document: Any, pointer: str) -> Any:
    value = document
    for encoded in pointer.lstrip("/").split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _question_paths(document: dict[str, Any]) -> set[str]:
    questions = document.get("questions", [])
    if not isinstance(questions, list):
        return set()
    return {
        question["field"]
        for question in questions
        if isinstance(question, dict) and isinstance(question.get("field"), str)
    }


def _evidence_for_path(provenance: dict[str, Any], path: str) -> list[Any]:
    candidate = path
    while candidate.startswith("/deal"):
        evidence = provenance.get(candidate)
        if isinstance(evidence, list):
            return evidence
        candidate = candidate.rsplit("/", 1)[0]
    return []


def _provenance_map(value: Any) -> dict[str, Any] | None:
    # Mapping support keeps old stored runs inspectable; new canonical outputs
    # always use the constrained list form.
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        return None
    result: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("field"), str):
            return None
        result[entry["field"]] = entry.get("evidence")
    return result


def _escape(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")
