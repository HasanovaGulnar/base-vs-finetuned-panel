"""Append-only JSONL logs for runs and votes.

One line per event keeps the files safe to append from multiple Streamlit
sessions and trivially inspectable after the demo.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    with _LOCK:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def log_run(path: str | Path, record: dict) -> None:
    payload = {**record, "logged_at": now_iso()}
    append_jsonl(path, payload)


def log_vote(path: str | Path, record: dict) -> None:
    payload = {**record, "voted_at": now_iso()}
    append_jsonl(path, payload)
