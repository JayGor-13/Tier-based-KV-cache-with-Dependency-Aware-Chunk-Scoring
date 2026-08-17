"""Check target-model access and print immutable Hugging Face revision pins."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.model_preflight import hub_model_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated model IDs")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--output", default="protocol/model_revisions.json")
    args = parser.parse_args()

    token = os.getenv(args.hf_token_env) if args.hf_token_env else None
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not models:
        raise ValueError("At least one model is required.")

    reports = [
        hub_model_preflight(model_name=model, revision=None, token=token)
        for model in models
    ]
    failures = [report for report in reports if report.get("passed") is not True]
    payload = {
        "models": {
            report["model"]: report.get("resolved_revision") for report in reports
        },
        "reports": reports,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if failures:
        for report in failures:
            print(f"FAILED {report['model']}: {'; '.join(report['issues'])}")
        raise SystemExit(2)

    pins = ";".join(
        f"{model}={revision}" for model, revision in payload["models"].items()
    )
    print(f'--model-revisions "{pins}"')
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
