"""Select and freeze the best configuration from a tuning result JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tuning import select_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/paper/selected_config.json")
    args = parser.parse_args()
    selection = select_from_file(args.input, args.output)
    print(f"Selected configuration: {selection['selected']['configuration']}")
    print(f"Selection written to {args.output}")


if __name__ == "__main__":
    main()
