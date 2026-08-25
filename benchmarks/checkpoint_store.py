"""Transactional SQLite progress storage for long experiment grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


def is_sqlite_checkpoint(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".sqlite", ".sqlite3", ".db"}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class SQLiteCheckpointStore:
    """Append-only run storage with keyed parity replacement and strict resume."""

    def __init__(
        self,
        path: str | Path,
        *,
        fingerprint: str,
        resume: bool,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        metadata = self._metadata()
        if resume:
            if not existed or not metadata:
                self.close()
                raise FileNotFoundError(
                    f"Resume requires an initialized SQLite checkpoint: {self.path}"
                )
            if metadata.get("fingerprint") != str(fingerprint):
                self.close()
                raise ValueError(
                    "Checkpoint configuration does not match the requested experiment."
                )
            if metadata.get("schema_version") != str(CHECKPOINT_SCHEMA_VERSION):
                self.close()
                raise ValueError("Unsupported SQLite checkpoint schema version.")
        else:
            if metadata or self._has_rows():
                self.close()
                raise FileExistsError(
                    f"Checkpoint already exists; use resume or a new path: {self.path}"
                )
            with self.connection:
                self._set_metadata("schema_version", str(CHECKPOINT_SCHEMA_VERSION))
                self._set_metadata("fingerprint", str(fingerprint))
                self._set_metadata("state", "in_progress")

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    row_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parity (
                    parity_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def _metadata(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in self.connection.execute("SELECT key, value FROM metadata")
        }

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (str(key), str(value)),
        )

    def _has_rows(self) -> bool:
        runs = self.connection.execute("SELECT EXISTS(SELECT 1 FROM runs)").fetchone()[0]
        parity = self.connection.execute("SELECT EXISTS(SELECT 1 FROM parity)").fetchone()[0]
        return bool(runs or parity)

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        run_key = row.get("run_key")
        if isinstance(run_key, str) and run_key:
            return f"run:{run_key}"
        digest = hashlib.sha256(_json(row).encode("utf-8")).hexdigest()
        return f"diagnostic:{digest}"

    @staticmethod
    def _parity_key(row: dict[str, Any]) -> str:
        values = [row.get("model"), row.get("dataset"), row.get("sample_id")]
        return hashlib.sha256(_json(values).encode("utf-8")).hexdigest()

    def append_run(self, row: dict[str, Any]) -> None:
        payload = _json(row)
        row_key = self._row_key(row)
        existing = self.connection.execute(
            "SELECT payload_json FROM runs WHERE row_key = ?", (row_key,)
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload:
                raise ValueError(f"Conflicting checkpoint row for {row_key}.")
            return
        with self.connection:
            self.connection.execute(
                "INSERT INTO runs(row_key, status, payload_json) VALUES (?, ?, ?)",
                (row_key, str(row.get("status")), payload),
            )

    def upsert_parity(self, row: dict[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO parity(parity_key, payload_json) VALUES (?, ?)",
                (self._parity_key(row), _json(row)),
            )

    def load(self) -> dict[str, Any]:
        metadata = self._metadata()
        runs = [
            json.loads(payload)
            for (payload,) in self.connection.execute(
                "SELECT payload_json FROM runs ORDER BY sequence"
            )
        ]
        parity_records = [
            json.loads(payload)
            for (payload,) in self.connection.execute(
                "SELECT payload_json FROM parity ORDER BY parity_key"
            )
        ]
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state": metadata.get("state"),
            "fingerprint": metadata.get("fingerprint"),
            "runs": runs,
            "parity_records": parity_records,
        }

    def mark_complete(self) -> None:
        with self.connection:
            self._set_metadata("state", "complete")

    def close(self) -> None:
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "SQLiteCheckpointStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "SQLiteCheckpointStore",
    "is_sqlite_checkpoint",
]
