"""SQLite-backed run history, baselines, and response cache.

Every run is persisted so that regressions can be detected by comparing runs, and
so `regressa trend` can show score movement over time. SQLite is deliberate: no
service to run, and the file is easy to commit as a CI artifact or throw away.
"""

from __future__ import annotations

import functools
import json
import os
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from regressa.results import CaseResult, RunSummary

DEFAULT_DB = Path(".regressa") / "regressa.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    suite        TEXT NOT NULL,
    suite_hash   TEXT NOT NULL,
    label        TEXT,
    git_sha      TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    total        INTEGER NOT NULL DEFAULT 0,
    passed       INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,
    errors       INTEGER NOT NULL DEFAULT 0,
    score        REAL NOT NULL DEFAULT 0,
    cost_usd     REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    case_id        TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    passed         INTEGER NOT NULL,
    score          REAL NOT NULL,
    output         TEXT,
    latency_ms     REAL NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    cached         INTEGER NOT NULL DEFAULT 0,
    stop_reason    TEXT,
    error          TEXT,
    tags           TEXT,
    assertions     TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_suite ON runs(suite, started_at);

CREATE TABLE IF NOT EXISTS baselines (
    suite   TEXT PRIMARY KEY,
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    set_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def new_run_id() -> str:
    """Sortable, human-typable run id: 20260811T134502-a1b2c3."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def git_sha() -> str | None:
    """Current commit, recorded so a regression can be traced to a change."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


def _synchronized(method: Callable[..., Any]) -> Callable[..., Any]:
    """Serialise a Store method against the shared connection.

    The runner calls `cache_get` / `cache_put` from worker threads while a run is
    in flight. A single sqlite3 connection tolerates use from several threads,
    but concurrent `with self.conn:` blocks interleave their implicit
    transactions and the driver raises `InterfaceError: bad parameter or other
    API misuse`. One lock around every connection touch removes the interleaving;
    contention is irrelevant next to the API latency these calls sit behind.
    """

    @functools.wraps(method)
    def wrapper(self: Store, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class Store:
    """Thin wrapper over the SQLite file. Usable as a context manager.

    Safe to share across threads: every method that touches the connection is
    serialised by `_lock`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.environ.get("REGRESSA_DB") or DEFAULT_DB)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reentrant: resolve_run -> get_run -> _hydrate all re-acquire.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @_synchronized
    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #

    @_synchronized
    def save_run(self, run: RunSummary) -> None:
        run.recompute()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (id, suite, suite_hash, label, git_sha, started_at, finished_at,
                     total, passed, failed, errors, score, cost_usd)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run.run_id,
                    run.suite,
                    run.suite_hash,
                    run.label,
                    run.git_sha,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.total,
                    run.passed,
                    run.failed,
                    run.errors,
                    run.score,
                    run.cost_usd,
                ),
            )
            self.conn.execute("DELETE FROM results WHERE run_id = ?", (run.run_id,))
            self.conn.executemany(
                """
                INSERT INTO results
                    (run_id, case_id, target_id, passed, score, output, latency_ms,
                     input_tokens, output_tokens, cost_usd, cached, stop_reason,
                     error, tags, assertions)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        run.run_id,
                        r.case_id,
                        r.target_id,
                        int(r.passed),
                        r.score,
                        r.output,
                        r.latency_ms,
                        r.usage.input_tokens,
                        r.usage.output_tokens,
                        r.usage.cost_usd,
                        int(r.cached),
                        r.stop_reason,
                        r.error,
                        json.dumps(r.tags),
                        json.dumps([a.model_dump() for a in r.assertions]),
                    )
                    for r in run.results
                ],
            )

    @_synchronized
    def get_run(self, run_id: str) -> RunSummary | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._hydrate(row)

    @_synchronized
    def latest_run(self, suite: str | None = None) -> RunSummary | None:
        if suite:
            row = self.conn.execute(
                "SELECT * FROM runs WHERE suite = ? ORDER BY started_at DESC LIMIT 1",
                (suite,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return self._hydrate(row) if row else None

    @_synchronized
    def resolve_run(self, ref: str, suite: str | None = None) -> RunSummary | None:
        """Resolve a run reference.

        Accepts a run id, `latest`, `baseline`, or `latest~N` for the Nth most
        recent run — so `regressa diff latest~1 latest` works without copying ids.
        """
        if ref == "baseline":
            return self.get_baseline(suite) if suite else None
        if ref == "latest":
            return self.latest_run(suite)
        if ref.startswith("latest~"):
            try:
                offset = int(ref.split("~", 1)[1])
            except ValueError:
                return None
            runs = self.list_runs(suite=suite, limit=offset + 1)
            return runs[offset] if len(runs) > offset else None
        return self.get_run(ref)

    @_synchronized
    def list_runs(self, suite: str | None = None, limit: int = 20) -> list[RunSummary]:
        if suite:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE suite = ? ORDER BY started_at DESC LIMIT ?",
                (suite, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._hydrate(row, with_results=False) for row in rows]

    @_synchronized
    def delete_run(self, run_id: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0

    @_synchronized
    def suites(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT suite FROM runs ORDER BY suite").fetchall()
        return [row["suite"] for row in rows]

    @_synchronized
    def _hydrate(self, row: sqlite3.Row, with_results: bool = True) -> RunSummary:
        run = RunSummary(
            run_id=row["id"],
            suite=row["suite"],
            suite_hash=row["suite_hash"],
            label=row["label"],
            git_sha=row["git_sha"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=(
                datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            total=row["total"],
            passed=row["passed"],
            failed=row["failed"],
            errors=row["errors"],
            score=row["score"],
            cost_usd=row["cost_usd"],
        )
        if with_results:
            run.results = list(self._results_for(row["id"]))
        return run

    def _results_for(self, run_id: str) -> Iterator[CaseResult]:
        rows = self.conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY case_id, target_id",
            (run_id,),
        ).fetchall()
        for row in rows:
            yield CaseResult.model_validate(
                {
                    "case_id": row["case_id"],
                    "target_id": row["target_id"],
                    "passed": bool(row["passed"]),
                    "score": row["score"],
                    "output": row["output"] or "",
                    "latency_ms": row["latency_ms"],
                    "usage": {
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "cost_usd": row["cost_usd"],
                    },
                    "cached": bool(row["cached"]),
                    "stop_reason": row["stop_reason"],
                    "error": row["error"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "assertions": json.loads(row["assertions"] or "[]"),
                }
            )

    # ------------------------------------------------------------------ #
    # Baselines
    # ------------------------------------------------------------------ #

    @_synchronized
    def set_baseline(self, suite: str, run_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO baselines (suite, run_id, set_at) VALUES (?,?,?)",
                (suite, run_id, datetime.now(timezone.utc).isoformat()),
            )

    @_synchronized
    def get_baseline(self, suite: str) -> RunSummary | None:
        row = self.conn.execute(
            "SELECT run_id FROM baselines WHERE suite = ?", (suite,)
        ).fetchone()
        return self.get_run(row["run_id"]) if row else None

    @_synchronized
    def clear_baseline(self, suite: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM baselines WHERE suite = ?", (suite,))
        return cur.rowcount > 0

    # ------------------------------------------------------------------ #
    # Response cache
    # ------------------------------------------------------------------ #

    @_synchronized
    def cache_get(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row["response"]) if row else None

    @_synchronized
    def cache_put(self, key: str, response: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?,?,?)",
                (key, json.dumps(response), datetime.now(timezone.utc).isoformat()),
            )

    @_synchronized
    def cache_clear(self) -> int:
        with self.conn:
            cur = self.conn.execute("DELETE FROM cache")
        return cur.rowcount

    # ------------------------------------------------------------------ #
    # Trends
    # ------------------------------------------------------------------ #

    @_synchronized
    def trend(self, suite: str, limit: int = 30) -> list[dict[str, Any]]:
        """Oldest-first score/pass-rate history, for charts and `regressa trend`."""
        rows = self.conn.execute(
            """
            SELECT id, started_at, label, git_sha, total, passed, score, cost_usd
            FROM runs WHERE suite = ? ORDER BY started_at DESC LIMIT ?
            """,
            (suite, limit),
        ).fetchall()
        return [
            {
                "run_id": row["id"],
                "started_at": row["started_at"],
                "label": row["label"],
                "git_sha": row["git_sha"],
                "total": row["total"],
                "passed": row["passed"],
                "pass_rate": (row["passed"] / row["total"]) if row["total"] else 0.0,
                "score": row["score"],
                "cost_usd": row["cost_usd"],
            }
            for row in reversed(rows)
        ]

    @_synchronized
    def case_history(self, suite: str, case_id: str, target_id: str, limit: int = 30
                     ) -> list[dict[str, Any]]:
        """Per-case history — useful for spotting a flaky case vs a real regression."""
        rows = self.conn.execute(
            """
            SELECT r.id AS run_id, r.started_at, res.passed, res.score
            FROM results res JOIN runs r ON r.id = res.run_id
            WHERE r.suite = ? AND res.case_id = ? AND res.target_id = ?
            ORDER BY r.started_at DESC LIMIT ?
            """,
            (suite, case_id, target_id, limit),
        ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "started_at": row["started_at"],
                "passed": bool(row["passed"]),
                "score": row["score"],
            }
            for row in reversed(rows)
        ]
