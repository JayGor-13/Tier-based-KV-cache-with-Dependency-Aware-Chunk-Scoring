"""Audit experiment JSON/CSV artifacts without modifying their source files.

Schema-v2 artifacts are checked against the executable result contract. Legacy
artifacts are inspected for numerical corruption and obvious structural errors,
but are deliberately classified as paper-ineligible because their provenance and
sample-level identities cannot be reconstructed reliably.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.io_utils import write_json_atomic
from benchmarks.numerical_validation import find_nonfinite_paths, generation_health
from benchmarks.qualification import recompute_declared_qualification
from benchmarks.result_schema import SCHEMA_VERSION, validate_result_payload


SUMMARY_IDENTITY_FIELDS = (
    "model",
    "dataset",
    "method",
    "requested_retention_ratio",
    "theta",
    "alpha",
    "beta",
    "recent_window",
    "chunking_strategy",
    "fixed_chunk_size",
    "min_chunk_tokens",
    "max_chunk_tokens",
    "dependency_top_k",
    "tier1_score_mode",
    "attention_mode",
    "layer_weighting",
    "protect_sink",
    "protect_recent",
    "allow_level2_fallback",
    "decode_policy",
)

CSV_FLOAT_FIELDS = {
    "requested_retention_ratio",
    "requested_compression_ratio",
    "requested_compression_multiplier",
    "theta",
    "alpha",
    "beta",
    "actual_retention_ratio",
    "actual_compression_ratio",
    "actual_compression_multiplier",
    "budget_utilization",
    "sequence_length",
    "kept_tokens",
    "removed_tokens",
    "num_chunks",
    "avg_chunk_size",
    "kv_gib_before",
    "kv_gib_after",
    "kv_gib_saved",
    "prefill_ms",
    "scoring_ms",
    "policy_ms",
    "decode_ms",
    "method_specific_ms",
    "decode_tokens_per_second",
    "decode_ms_per_token",
    "evidence_token_retention",
    "evidence_chunk_survival",
    "tier0_chunks",
    "tier1_chunks",
    "tier2_chunks",
    "quality_mean",
}

CSV_INTEGER_FIELDS = {"samples", "runs", "budget_shortfall_max", "budget_overflow_max"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load_strictly_observed(path: Path) -> Any:
    # Retain non-standard NaN/Infinity long enough to report their exact paths.
    return json.loads(path.read_text(encoding="utf-8"))


def _record_issue(
    issues: list[dict[str, Any]],
    *,
    index: int | str,
    code: str,
    detail: str,
) -> None:
    issues.append({"record": index, "code": code, "detail": detail})


def _audit_rows(rows: Iterable[Any]) -> tuple[int, list[dict[str, Any]], list[str]]:
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: dict[str, Mapping[str, Any]] = {}
    count = 0
    for index, raw in enumerate(rows):
        count += 1
        if not isinstance(raw, Mapping):
            _record_issue(
                issues,
                index=index,
                code="not_an_object",
                detail="run row is not a JSON object",
            )
            continue
        row = dict(raw)
        nonfinite = find_nonfinite_paths(row)
        if nonfinite:
            _record_issue(
                issues,
                index=index,
                code="nonfinite",
                detail=f"non-finite numeric values at {nonfinite[:10]}",
            )
        token_ids = row.get("generated_token_ids")
        if isinstance(token_ids, list):
            try:
                observed = generation_health(
                    token_ids,
                    max_new_tokens=(row.get("config") or {}).get("max_new_tokens"),
                )
            except (TypeError, ValueError, OverflowError) as exc:
                _record_issue(
                    issues,
                    index=index,
                    code="invalid_token_ids",
                    detail=str(exc),
                )
            else:
                if observed["degenerate_repetition"]:
                    _record_issue(
                        issues,
                        index=index,
                        code="degenerate_generation",
                        detail="generation reached its limit with >=95% one token",
                    )
        run_key = row.get("run_key")
        if isinstance(run_key, str):
            if run_key in seen and dict(seen[run_key]) != row:
                _record_issue(
                    issues,
                    index=index,
                    code="conflicting_duplicate",
                    detail=f"run_key {run_key!r} has different payloads",
                )
            elif run_key in seen:
                warnings.append(f"record {index} is an exact duplicate of run_key {run_key}")
            seen[run_key] = row
    return count, issues, warnings


def audit_json(path: Path) -> dict[str, Any]:
    try:
        payload = _json_load_strictly_observed(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "format": "json",
            "artifact_class": "unreadable",
            "records": 0,
            "corrupt_records": 0,
            "paper_eligible": False,
            "blockers": [{"record": "$", "code": "unreadable", "detail": str(exc)}],
            "warnings": [],
        }

    if isinstance(payload, Mapping) and isinstance(payload.get("runs"), list):
        count, row_issues, warnings = _audit_rows(payload["runs"])
        if payload.get("schema_version") == SCHEMA_VERSION:
            schema_failures = validate_result_payload(payload, require_complete=True)
            blockers = [
                {"record": "$", "code": "schema", "detail": failure}
                for failure in schema_failures
            ] + row_issues
            recomputed = recompute_declared_qualification(dict(payload))
            paper_eligible = not blockers and recomputed.get("passed") is True
            if recomputed.get("passed") is not True:
                blockers.extend(
                    {
                        "record": "$",
                        "code": "not_qualified",
                        "detail": failure,
                    }
                    for failure in recomputed.get("failures", [])
                )
            artifact_class = "schema_v2"
        else:
            blockers = row_issues + [
                {
                    "record": "$",
                    "code": "legacy_schema",
                    "detail": "artifact lacks the schema-v2 identity/provenance contract",
                }
            ]
            paper_eligible = False
            artifact_class = "legacy_result"
    elif isinstance(payload, list):
        count, row_issues, warnings = _audit_rows(payload)
        blockers = row_issues + [
            {
                "record": "$",
                "code": "legacy_schema",
                "detail": "top-level row lists lack schema-v2 provenance and coverage",
            }
        ]
        paper_eligible = False
        artifact_class = "legacy_rows"
    else:
        count = 0
        warnings = []
        blockers = [
            {
                "record": "$",
                "code": "unsupported_shape",
                "detail": "expected a result object with `runs` or a list of run rows",
            }
        ]
        paper_eligible = False
        artifact_class = "unsupported_json"

    corrupt_codes = {
        "unreadable",
        "nonfinite",
        "invalid_token_ids",
        "degenerate_generation",
        "conflicting_duplicate",
        "not_an_object",
    }
    corrupt_records = len(
        {
            str(issue["record"])
            for issue in blockers
            if issue["code"] in corrupt_codes
        }
    )
    return {
        "format": "json",
        "artifact_class": artifact_class,
        "records": count,
        "corrupt_records": corrupt_records,
        "paper_eligible": paper_eligible,
        "blockers": blockers,
        "warnings": warnings,
    }


def _parse_csv_number(
    row: Mapping[str, str],
    field: str,
    *,
    integer: bool = False,
) -> int | float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field} is non-finite: {raw!r}")
    if integer:
        if not value.is_integer():
            raise ValueError(f"{field} must be an integer: {raw!r}")
        return int(value)
    return value


def audit_csv(path: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = set(reader.fieldnames or [])
    except (OSError, UnicodeError, csv.Error) as exc:
        return {
            "format": "csv",
            "artifact_class": "unreadable",
            "records": 0,
            "corrupt_records": 0,
            "paper_eligible": False,
            "blockers": [{"record": "$", "code": "unreadable", "detail": str(exc)}],
            "warnings": [],
        }

    is_summary = {"quality_mean", "samples", "runs"}.issubset(fields)
    duplicate_identities: dict[tuple[str, ...], tuple[int, dict[str, str]]] = {}
    for index, row in enumerate(rows, start=2):
        for field in sorted(CSV_FLOAT_FIELDS & fields):
            try:
                _parse_csv_number(row, field)
            except (ValueError, OverflowError) as exc:
                _record_issue(issues, index=index, code="invalid_number", detail=str(exc))
        for field in sorted(CSV_INTEGER_FIELDS & fields):
            try:
                _parse_csv_number(row, field, integer=True)
            except (ValueError, OverflowError) as exc:
                _record_issue(issues, index=index, code="invalid_number", detail=str(exc))

        for field in ("requested_retention_ratio", "actual_retention_ratio", "quality_mean"):
            if field not in fields:
                continue
            try:
                value = _parse_csv_number(row, field)
            except (ValueError, OverflowError):
                continue
            if value is not None and not 0.0 <= float(value) <= 1.0:
                _record_issue(
                    issues,
                    index=index,
                    code="out_of_range",
                    detail=f"{field}={value} is outside [0, 1]",
                )

        if is_summary:
            identity = tuple(row.get(field, "") for field in SUMMARY_IDENTITY_FIELDS)
            previous = duplicate_identities.get(identity)
            if previous is not None and previous[1] != row:
                _record_issue(
                    issues,
                    index=index,
                    code="conflicting_duplicate",
                    detail=f"same summary configuration differs from row {previous[0]}",
                )
            elif previous is not None:
                warnings.append(f"CSV row {index} exactly duplicates row {previous[0]}")
            duplicate_identities[identity] = (index, dict(row))

    if is_summary and rows:
        qualities = []
        for row in rows:
            try:
                value = _parse_csv_number(row, "quality_mean")
            except (ValueError, OverflowError):
                continue
            if value is not None:
                qualities.append(float(value))
        if qualities and all(value == 0.0 for value in qualities):
            warnings.append(
                "every reported quality_mean is zero; inspect raw predictions and parsing"
            )

    issues.append(
        {
            "record": "$",
            "code": "legacy_summary" if is_summary else "legacy_csv",
            "detail": (
                "aggregate CSV lacks sample identities, model revisions, protocol "
                "fingerprints, raw predictions, and qualification evidence"
            ),
        }
    )
    corrupt_codes = {"invalid_number", "out_of_range", "conflicting_duplicate"}
    corrupt_records = len(
        {
            str(issue["record"])
            for issue in issues
            if issue["code"] in corrupt_codes
        }
    )
    return {
        "format": "csv",
        "artifact_class": "legacy_summary" if is_summary else "legacy_csv",
        "records": len(rows),
        "corrupt_records": corrupt_records,
        "paper_eligible": False,
        "blockers": issues,
        "warnings": warnings,
    }


def audit_path(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "exists": False,
            "format": resolved.suffix.lower().lstrip(".") or "unknown",
            "artifact_class": "missing",
            "records": 0,
            "corrupt_records": 0,
            "paper_eligible": False,
            "blockers": [
                {"record": "$", "code": "missing", "detail": "file does not exist"}
            ],
            "warnings": [],
        }
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        report = audit_json(resolved)
    elif suffix in {".csv", ".tsv"}:
        report = audit_csv(resolved)
    else:
        report = {
            "format": suffix.lstrip(".") or "unknown",
            "artifact_class": "unsupported",
            "records": 0,
            "corrupt_records": 0,
            "paper_eligible": False,
            "blockers": [
                {
                    "record": "$",
                    "code": "unsupported_format",
                    "detail": f"unsupported file extension {suffix!r}",
                }
            ],
            "warnings": [],
        }
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": _sha256(resolved),
        **report,
    }


def audit_results(paths: Iterable[str | Path]) -> dict[str, Any]:
    sources = [audit_path(path) for path in paths]
    return {
        "audit_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "record_count": sum(int(source.get("records", 0)) for source in sources),
        "corrupt_record_count": sum(
            int(source.get("corrupt_records", 0)) for source in sources
        ),
        "paper_eligible_source_count": sum(
            1 for source in sources if source.get("paper_eligible") is True
        ),
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSON/CSV artifacts to audit")
    parser.add_argument("--output", help="Optional strict-JSON audit report path")
    parser.add_argument(
        "--fail-on",
        choices=("never", "corrupt", "paper-ineligible"),
        default="corrupt",
        help="Exit policy; source files are never modified",
    )
    options = parser.parse_args()
    report = audit_results(options.paths)
    if options.output:
        write_json_atomic(options.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if options.fail_on == "corrupt" and report["corrupt_record_count"]:
        raise SystemExit(2)
    if options.fail_on == "paper-ineligible" and (
        report["paper_eligible_source_count"] != report["source_count"]
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
