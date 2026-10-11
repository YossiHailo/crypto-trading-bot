"""Docker health check: exit 0 when the heartbeat is fresh, 1 otherwise.

    python -m core.healthcheck

Fresh means younger than three loop intervals (at least 10 minutes), read
from LOOP_INTERVAL_MINUTES and DATA_DIR in the environment.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    interval = float(os.environ.get("LOOP_INTERVAL_MINUTES", "15"))
    max_age = max(600.0, interval * 60 * 3)
    try:
        stamp = datetime.fromisoformat((data_dir / "heartbeat").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        print("no heartbeat yet")
        return 1
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if age > max_age:
        print(f"heartbeat is {age:.0f}s old (limit {max_age:.0f}s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
