#!/usr/bin/env bash
# Compatibility wrapper for the complete resumable paper suite.
set -euo pipefail

python scripts/run_paper_suite.py --profile "${1:-main}" "${@:2}"
