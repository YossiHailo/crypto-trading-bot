"""Run one paper-trading cycle against the configured exchange and exit.

Usage (from the repository root):

    python scripts/dry_run.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import amain  # noqa: E402


def main() -> None:
    raise SystemExit(asyncio.run(amain(["--once"])))


if __name__ == "__main__":
    main()
