"""Run every committed case against local Qwen and persist actual outputs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

from evals.scorer import ExtractionScore, field_differences, score_extraction
from optiflow_ctrm.extractor import RecapExtractor
from optiflow_ctrm.models import ExtractionResult
from optiflow_ctrm.runtime.base import InferenceResult, InferenceRuntime
from optiflow_ctrm.runtime.big_model import (
    MODEL_QUANTIZATION as BIG_MODEL_QUANTIZATION,
    MODEL_REPOSITORY as BIG_MODEL_REPOSITORY,
    MODEL_REVISION as BIG_MODEL_REVISION,
    Qwen30BVllmClient,
    Qwen30BVllmConfig,
)
from optiflow_ctrm.runtime.small_model import (
    GGUF_MODEL_FILE,
    GGUF_MODEL_REPOSITORY,
    GGUF_MODEL_REVISION,
    GGUF_MODEL_SHA256,
    SOURCE_MODEL_REPOSITORY,
    LlamaCppClient,
    LlamaCppConfig,
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    source_path: Path
    source_text: str
    source_sha256: str
    expected: dict[str, Any]


@dataclass(slots=True)
class EvaluationObservation:
    actual: ExtractionResult | Mapping[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)


class EvaluationCallError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


class RecordingRuntime:
    """Transparent runtime wrapper retaining outputs and inference measurements."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self.runtime = runtime
        self.results: list[InferenceResult] = []
        self.request_latencies_ms: list[float] = []

    def generate_json(
        self,
        messages: Sequence[Mapping[str, str]],
        json_schema: Mapping[str, Any],
    ) -> InferenceResult:
        started = time.perf_counter()
        result = self.runtime.generate_json(messages, json_schema)
        request_latency_ms = (time.perf_counter() - started) * 1000
        self.results.append(result)
        self.request_latencies_ms.append(request_latency_ms)
        return result


class LiveExtractor:
    def __init__(self, runtime: InferenceRuntime) -> None:
        self.recording_runtime = RecordingRuntime(runtime)
        self.extractor = RecapExtractor(self.recording_runtime)

    def __call__(self, case: EvalCase) -> EvaluationObservation:
        start = len(self.recording_runtime.results)
        try:
            actual = self.extractor.extract(case.source_text)
        except Exception as exc:
            attempts = [
                _serialize_attempt(result, latency)
                for result, latency in zip(
                    self.recording_runtime.results[start:],
                    self.recording_runtime.request_latencies_ms[start:],
                    strict=True,
                )
            ]
            raise EvaluationCallError(f"{type(exc).__name__}: {exc}", attempts) from exc
        attempts = [
            _serialize_attempt(result, latency)
            for result, latency in zip(
                self.recording_runtime.results[start:],
                self.recording_runtime.request_latencies_ms[start:],
                strict=True,
            )
        ]
        return EvaluationObservation(actual=actual, attempts=attempts)


def load_cases(case_dir: Path, repository_root: Path) -> list[EvalCase]:
    """Load fixtures, resolve their source files, and reject source drift."""

    cases: list[EvalCase] = []
    for fixture_path in sorted(case_dir.glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source_info = fixture["source"]
        source_path = repository_root / source_info["path"]
        if not source_path.is_file():
            # The supplied files now live under data/recaps; retain compatibility
            # with fixtures authored before that directory was introduced.
            source_path = repository_root / "data" / "recaps" / Path(source_info["path"]).name
        if not source_path.is_file():
            raise FileNotFoundError(
                f"source for {fixture_path.name} not found: {source_info['path']}"
            )
        source_bytes = source_path.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        if digest != source_info["sha256"]:
            raise ValueError(
                f"source hash mismatch for {fixture['id']}: expected {source_info['sha256']}, got {digest}"
            )
        cases.append(
            EvalCase(
                id=fixture["id"],
                source_path=source_path,
                source_text=source_bytes.decode("utf-8"),
                source_sha256=digest,
                expected=fixture["expected"],
            )
        )
    if not cases:
        raise ValueError(f"no evaluation cases found in {case_dir}")
    return cases


def evaluate_cases(
    cases: Sequence[EvalCase],
    extract: Callable[[EvalCase], ExtractionResult | Mapping[str, Any] | EvaluationObservation],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run and score cases without stopping the suite after one model failure."""

    case_results: list[dict[str, Any]] = []
    scores: list[ExtractionScore] = []
    for case in cases:
        started = time.perf_counter()
        actual: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        error: str | None = None
        try:
            observed = extract(case)
            if isinstance(observed, EvaluationObservation):
                actual_value = observed.actual
                attempts = observed.attempts
            else:
                actual_value = observed
            actual = _model_or_mapping_to_dict(actual_value)
        except EvaluationCallError as exc:
            error = str(exc)
            attempts = exc.attempts
        except Exception as exc:  # keep remaining eval cases measurable
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        score = score_extraction(case.expected, actual, case.source_text)
        scores.append(score)
        case_results.append(
            {
                "id": case.id,
                "source": {
                    "path": str(case.source_path),
                    "sha256": case.source_sha256,
                },
                "latency_ms": latency_ms,
                "actual": actual,
                "differences": field_differences(case.expected, actual),
                "attempts": attempts,
                "error": error,
                "expected": case.expected,
                "score": score.to_dict(),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
        "summary": _summarize(
            scores,
            [item["latency_ms"] for item in case_results],
            case_results,
        ),
        "cases": case_results,
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> Path:
    """Write diagnostics to the report and extraction-only JSON to .cases files.

    A failed extraction is stored as JSON null; its error and raw attempts stay
    in the aggregate report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    case_dir = output_path.parent / f"{output_path.stem}.cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for case in report.get("cases", []):
        case_path = case_dir / f"{case['id']}.json"
        case_path.write_text(
            json.dumps(
                case.get("actual"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return output_path


def _summarize(
    scores: Sequence[ExtractionScore],
    latencies: Sequence[float],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_fields = sum(score.total_fields for score in scores)
    correct = sum(score.correct_values + score.correct_abstentions for score in scores)
    expected_questions = sum(score.expected_questions for score in scores)
    matched_questions = sum(score.matched_questions for score in scores)
    summary = {
        "cases": len(scores),
        "schema_valid_cases": sum(score.schema_valid for score in scores),
        "provenance_valid_cases": sum(score.provenance_valid for score in scores),
        "total_fields": total_fields,
        "correct_values": sum(score.correct_values for score in scores),
        "incorrect_values": sum(score.incorrect_values for score in scores),
        "correct_abstentions": sum(score.correct_abstentions for score in scores),
        "wrong_abstentions": sum(score.wrong_abstentions for score in scores),
        "unsupported_non_null": sum(score.unsupported_non_null for score in scores),
        "field_accuracy": correct / total_fields if total_fields else 0.0,
        "question_recall": matched_questions / expected_questions if expected_questions else 1.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    summary.update(_summarize_inference(case_results))
    return summary


def _summarize_inference(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate token counts and throughput across every successful model call.

    The primary rate uses measured HTTP call time, so it is available and
    comparable across runtimes. The optional model-reported generation rate
    excludes prompt evaluation and is combined by reconstructing generation
    time for each call rather than by taking an unweighted mean.
    """

    attempts = [
        attempt
        for case in case_results
        for attempt in case.get("attempts", [])
        if isinstance(attempt, Mapping)
    ]
    prompt_tokens = _sum_int_values(attempts, "prompt_tokens")
    completion_tokens = _sum_int_values(attempts, "completion_tokens")
    total_tokens = _sum_int_values(attempts, "total_tokens")

    reported_tokens = 0
    reported_seconds = 0.0
    reported_rate_attempts = 0
    end_to_end_tokens = 0
    end_to_end_seconds = 0.0
    end_to_end_attempts = 0
    for attempt in attempts:
        tokens = attempt.get("completion_tokens")
        rate = attempt.get("tokens_per_second")
        if (
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens > 0
            and isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and rate > 0
        ):
            reported_tokens += tokens
            reported_seconds += tokens / float(rate)
            reported_rate_attempts += 1

        request_latency_ms = attempt.get("request_latency_ms")
        if (
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and tokens >= 0
            and isinstance(request_latency_ms, (int, float))
            and not isinstance(request_latency_ms, bool)
            and request_latency_ms > 0
        ):
            end_to_end_tokens += tokens
            end_to_end_seconds += float(request_latency_ms) / 1000
            end_to_end_attempts += 1

    return {
        "inference_attempts": len(attempts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "aggregate_tokens_per_second": (
            end_to_end_tokens / end_to_end_seconds if end_to_end_seconds else None
        ),
        "aggregate_tokens_per_second_method": (
            "completion_tokens / summed successful inference HTTP request duration"
        ),
        "throughput_attempts": end_to_end_attempts,
        "aggregate_generation_tokens_per_second": (
            reported_tokens / reported_seconds if reported_seconds else None
        ),
        "generation_rate_attempts": reported_rate_attempts,
    }


def _sum_int_values(values: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(
        value
        for item in values
        if isinstance((value := item.get(key)), int) and not isinstance(value, bool)
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _model_or_mapping_to_dict(value: ExtractionResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, ExtractionResult):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("extractor must return ExtractionResult or a mapping")


def _serialize_attempt(result: InferenceResult, request_latency_ms: float) -> dict[str, Any]:
    return {
        "raw_value": result.value,
        "raw_text": result.raw_text,
        "model": result.model,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "tokens_per_second": result.tokens_per_second,
        "request_latency_ms": request_latency_ms,
        "timings": dict(result.timings),
    }


def _print_table(report: Mapping[str, Any]) -> None:
    headers = ("case", "accuracy", "wrong-null", "unsupported", "schema", "provenance", "ms")
    rows = []
    for item in report["cases"]:
        score = item["score"]
        rows.append(
            (
                item["id"],
                f"{score['field_accuracy']:.1%}",
                str(score["wrong_abstentions"]),
                str(score["unsupported_non_null"]),
                "yes" if score["schema_valid"] else "no",
                "yes" if score["provenance_valid"] else "no",
                f"{item['latency_ms']:.0f}",
            )
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    summary = report["summary"]
    aggregate_rate = summary["aggregate_tokens_per_second"]
    rate_text = f"{aggregate_rate:.1f}" if aggregate_rate is not None else "n/a"
    print(
        f"\nOverall accuracy {summary['field_accuracy']:.1%}; "
        f"wrong abstentions {summary['wrong_abstentions']}; "
        f"unsupported values {summary['unsupported_non_null']}; "
        f"p50/p95 {summary['latency_ms_p50']:.0f}/{summary['latency_ms_p95']:.0f} ms; "
        f"aggregate inference {rate_text} completion tokens/s"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        choices=("llama-cpp", "vllm"),
        default="llama-cpp",
        help="inference server API to use (default: llama-cpp)",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model", help="served model name; defaults to the selected runtime model")
    parser.add_argument("--case-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-version", default=os.getenv("LLAMA_CPP_VERSION"))
    parser.add_argument("--model-revision")
    parser.add_argument("--gguf-sha256", default=os.getenv("QWEN_GGUF_SHA256", GGUF_MODEL_SHA256))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    cases = load_cases(args.case_dir, repository_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.runtime == "vllm":
        config = Qwen30BVllmConfig(
            base_url=args.base_url or "http://127.0.0.1:8000",
            model=args.model or BIG_MODEL_REPOSITORY,
            api_key=os.getenv("VLLM_API_KEY"),
        )
        generation = asdict(config)
        generation.pop("api_key", None)
        metadata = {
            "source_model_repository": BIG_MODEL_REPOSITORY,
            "model_revision": args.model_revision or BIG_MODEL_REVISION,
            "quantization": BIG_MODEL_QUANTIZATION,
            "runtime": "vLLM",
            "runtime_version": args.runtime_version,
            "generation": generation,
        }
        runtime: InferenceRuntime = Qwen30BVllmClient(config)
        default_name = f"{timestamp}_qwen3-30b-a3b-fp8.json"
    else:
        config = LlamaCppConfig(
            base_url=args.base_url or "http://127.0.0.1:8080",
            model=args.model or SOURCE_MODEL_REPOSITORY,
        )
        metadata = {
            "source_model_repository": SOURCE_MODEL_REPOSITORY,
            "model_revision": args.model_revision
            or os.getenv("QWEN_MODEL_REVISION", GGUF_MODEL_REVISION),
            "gguf_repository": GGUF_MODEL_REPOSITORY,
            "gguf_file": GGUF_MODEL_FILE,
            "gguf_sha256": args.gguf_sha256,
            "runtime": "llama.cpp",
            "runtime_version": args.runtime_version,
            "generation": asdict(config),
        }
        runtime = LlamaCppClient(config)
        default_name = f"{timestamp}_qwen3-4b-q4_k_m.json"
    report = evaluate_cases(cases, LiveExtractor(runtime), metadata)
    output = args.output or repository_root / "evals" / "results" / default_name
    write_report(report, output)
    _print_table(report)
    print(f"Actual outputs written to {output}")
    print(f"Per-recap extraction outputs written to {output.parent / f'{output.stem}.cases'}")
    return 0 if report["summary"]["schema_valid_cases"] == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
