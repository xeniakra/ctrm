import copy
import json
from pathlib import Path

from evals.run import EvaluationObservation, evaluate_cases, load_cases, write_report
from evals.scorer import field_differences, score_extraction


ROOT = Path(__file__).parents[1]


def _fixture(name: str) -> tuple[dict, str]:
    case = json.loads((ROOT / "evals" / "cases" / name).read_text())
    source = (ROOT / "data" / "recaps" / Path(case["source"]["path"]).name).read_text()
    return case["expected"], source


def test_perfect_extraction_scores_all_fields_and_provenance() -> None:
    expected, source = _fixture("01_clean_telegram.json")

    score = score_extraction(expected, copy.deepcopy(expected), source)

    assert score.field_accuracy == 1.0
    assert score.incorrect_values == 0
    assert score.wrong_abstentions == 0
    assert score.unsupported_non_null == 0
    assert score.provenance_valid is True
    assert score.question_recall == 1.0


def test_hallucinated_value_is_scored_as_unsupported_non_null() -> None:
    expected, source = _fixture("03_incomplete_must_abstain.json")
    actual = copy.deepcopy(expected)
    actual["deal"]["quantity"]["value"] = 15_000

    score = score_extraction(expected, actual, source)

    assert score.unsupported_non_null == 1
    assert "/deal/quantity/value" in score.unsupported_paths


def test_stated_value_left_null_is_wrong_abstention() -> None:
    expected, source = _fixture("01_clean_telegram.json")
    actual = copy.deepcopy(expected)
    actual["deal"]["buyer"] = None

    score = score_extraction(expected, actual, source)

    assert score.wrong_abstentions == 1
    assert "/deal/buyer" in score.wrong_abstention_paths


def test_loader_hash_checks_sources_and_report_persists_actual_outputs(tmp_path: Path) -> None:
    cases = load_cases(ROOT / "evals" / "cases", ROOT)
    outputs = {case.id: copy.deepcopy(case.expected) for case in cases}

    report = evaluate_cases(cases, lambda case: outputs[case.id], metadata={"seed": 42})
    output_path = write_report(report, tmp_path / "result.json")
    persisted = json.loads(output_path.read_text())
    per_case = json.loads((tmp_path / "result.cases" / "01_clean_telegram.json").read_text())

    assert len(persisted["cases"]) == len(cases)
    assert persisted["cases"][0]["actual"] is not None
    assert persisted["summary"]["field_accuracy"] == 1.0
    assert persisted["metadata"]["seed"] == 42
    assert per_case == outputs["01_clean_telegram"]
    assert set(per_case) == {"deal", "questions", "provenance"}
    assert persisted["cases"][0]["expected"] == outputs["01_clean_telegram"]


def test_report_aggregates_weighted_generation_and_end_to_end_throughput() -> None:
    cases = load_cases(ROOT / "evals" / "cases", ROOT)[:1]

    def observe(case):
        return EvaluationObservation(
            actual=copy.deepcopy(case.expected),
            attempts=[
                {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                    "tokens_per_second": 20.0,
                    "request_latency_ms": 6000.0,
                },
                {
                    "prompt_tokens": 80,
                    "completion_tokens": 50,
                    "total_tokens": 130,
                    "tokens_per_second": 10.0,
                    "request_latency_ms": 6000.0,
                },
            ],
        )

    summary = evaluate_cases(cases, observe)["summary"]

    assert summary["inference_attempts"] == 2
    assert summary["prompt_tokens"] == 280
    assert summary["completion_tokens"] == 150
    assert summary["total_tokens"] == 430
    assert summary["aggregate_tokens_per_second"] == 12.5
    assert summary["throughput_attempts"] == 2
    assert summary["aggregate_generation_tokens_per_second"] == 15.0
    assert summary["generation_rate_attempts"] == 2


def test_failed_validation_keeps_latest_generated_output_for_manual_review(tmp_path: Path) -> None:
    cases = load_cases(ROOT / "evals" / "cases", ROOT)

    def fail(_case):
        from evals.run import EvaluationCallError

        raise EvaluationCallError(
            "invalid model output",
            [{"raw_value": {"buyer": "flat output"}, "raw_text": "..."}],
        )

    result = evaluate_cases(cases[:1], fail)

    assert result["cases"][0]["actual"] is None
    assert result["cases"][0]["attempts"][0]["raw_value"]["buyer"] == "flat output"
    assert "validated_actual" not in result["cases"][0]
    write_report(result, tmp_path / "failed.json")
    assert json.loads((tmp_path / "failed.cases" / f"{cases[0].id}.json").read_text()) is None
    persisted = json.loads((tmp_path / "failed.json").read_text())
    assert persisted["cases"][0]["attempts"] == result["cases"][0]["attempts"]


def test_expected_deal_keys_are_alphabetically_sorted_for_review() -> None:
    for path in sorted((ROOT / "evals" / "cases").glob("*.json")):
        fixture = json.loads(path.read_text(), object_pairs_hook=dict)
        assert set(fixture) == {"expected", "id", "source"}
        assert set(fixture["expected"]) == {"deal", "provenance", "questions"}
        keys = list(fixture["expected"]["deal"])
        assert keys == sorted(keys)


def test_manual_differences_include_required_properties_outside_deal() -> None:
    expected, _ = _fixture("01_clean_telegram.json")
    actual = copy.deepcopy(expected)
    actual["questions"] = [{"field": "/deal/buyer", "question": "What is the buyer?"}]

    differences = field_differences(expected, actual)

    assert differences == [
        {
            "actual": [{"field": "/deal/buyer", "question": "What is the buyer?"}],
            "classification": "incorrect_value",
            "expected": [],
            "path": "/questions",
        }
    ]
