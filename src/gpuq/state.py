"""SQLite-backed durable state for jobs and leases."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

JobState = str  # queued | starting | running | done | failed | cancelled

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    cmd TEXT NOT NULL,
    cwd TEXT NOT NULL,
    env TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    pid INTEGER,
    exit_code INTEGER,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    parent_id TEXT REFERENCES jobs(id),
    description TEXT
);
CREATE INDEX IF NOT EXISTS jobs_state_priority
    ON jobs(state, priority DESC, submitted_at);

CREATE TABLE IF NOT EXISTS leases (
    id TEXT PRIMARY KEY,
    reason TEXT,
    granted_at TEXT,
    expires_at TEXT,
    released_at TEXT
);
"""

_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("parent_id", "ALTER TABLE jobs ADD COLUMN parent_id TEXT REFERENCES jobs(id)"),
    ("description", "ALTER TABLE jobs ADD COLUMN description TEXT"),
)


@dataclass(frozen=True)
class JobRecord:
    id: str
    cmd: list[str]
    cwd: str
    env: dict[str, str]
    tag: str
    priority: int
    state: JobState
    pid: int | None
    exit_code: int | None
    submitted_at: str
    started_at: str | None
    finished_at: str | None
    parent_id: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class LeaseRecord:
    id: str
    reason: str | None
    granted_at: str | None
    expires_at: str | None
    released_at: str | None


def _row_to_job(r: sqlite3.Row) -> JobRecord:
    keys = r.keys()
    return JobRecord(
        id=r["id"],
        cmd=json.loads(r["cmd"]),
        cwd=r["cwd"],
        env=json.loads(r["env"]),
        tag=r["tag"],
        priority=r["priority"],
        state=r["state"],
        pid=r["pid"],
        exit_code=r["exit_code"],
        submitted_at=r["submitted_at"],
        started_at=r["started_at"],
        finished_at=r["finished_at"],
        parent_id=r["parent_id"] if "parent_id" in keys else None,
        description=r["description"] if "description" in keys else None,
    )


def _row_to_lease(r: sqlite3.Row) -> LeaseRecord:
    return LeaseRecord(
        id=r["id"],
        reason=r["reason"],
        granted_at=r["granted_at"],
        expires_at=r["expires_at"],
        released_at=r["released_at"],
    )


class JobRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._c = conn

    def insert(self, j: JobRecord) -> None:
        self._c.execute(
            "INSERT INTO jobs(id,cmd,cwd,env,tag,priority,state,pid,exit_code,"
            "submitted_at,started_at,finished_at,parent_id,description) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                j.id,
                json.dumps(j.cmd),
                j.cwd,
                json.dumps(j.env),
                j.tag,
                j.priority,
                j.state,
                j.pid,
                j.exit_code,
                j.submitted_at,
                j.started_at,
                j.finished_at,
                j.parent_id,
                j.description,
            ),
        )

    def delete(self, jid: str) -> bool:
        cur = self._c.execute("DELETE FROM jobs WHERE id=?", (jid,))
        return cur.rowcount > 0

    def get(self, jid: str) -> JobRecord | None:
        r = self._c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return _row_to_job(r) if r else None

    def find(
        self,
        *,
        state: str | None = None,
        states: list[str] | None = None,
        tag: str | None = None,
        since: str | None = None,
    ) -> list[JobRecord]:
        sql = "SELECT * FROM jobs WHERE 1=1"
        args: list = []
        if state:
            sql += " AND state=?"
            args.append(state)
        if states:
            placeholders = ",".join("?" for _ in states)
            sql += f" AND state IN ({placeholders})"
            args.extend(states)
        if tag:
            sql += " AND tag=?"
            args.append(tag)
        if since:
            sql += " AND submitted_at >= ?"
            args.append(since)
        sql += " ORDER BY priority DESC, submitted_at ASC"
        return [_row_to_job(r) for r in self._c.execute(sql, args)]

    def update_state(
        self,
        jid: str,
        *,
        state: str,
        pid: int | None = None,
        exit_code: int | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        sets = ["state=?"]
        args: list = [state]
        if pid is not None:
            sets.append("pid=?")
            args.append(pid)
        if exit_code is not None:
            sets.append("exit_code=?")
            args.append(exit_code)
        if started_at is not None:
            sets.append("started_at=?")
            args.append(started_at)
        if finished_at is not None:
            sets.append("finished_at=?")
            args.append(finished_at)
        args.append(jid)
        self._c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", args)

    def max_queued_priority(self, *, exclude: str | None = None) -> int | None:
        sql = "SELECT MAX(priority) AS m FROM jobs WHERE state='queued'"
        args: list = []
        if exclude is not None:
            sql += " AND id != ?"
            args.append(exclude)
        r = self._c.execute(sql, args).fetchone()
        return r["m"] if r and r["m"] is not None else None

    def bump(self, jid: str) -> bool:
        rec = self.get(jid)
        if rec is None or rec.state != "queued":
            return False
        top = self.max_queued_priority(exclude=jid)
        new_prio = (top + 1) if top is not None else 1
        self._c.execute("UPDATE jobs SET priority=? WHERE id=?", (new_prio, jid))
        return True

    def next_queued(self) -> JobRecord | None:
        r = self._c.execute(
            "SELECT * FROM jobs WHERE state='queued' "
            "ORDER BY priority DESC, submitted_at ASC LIMIT 1"
        ).fetchone()
        return _row_to_job(r) if r else None

    def running(self) -> list[JobRecord]:
        return [
            _row_to_job(r)
            for r in self._c.execute("SELECT * FROM jobs WHERE state IN ('starting','running')")
        ]


class LeaseRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._c = conn

    def insert(self, lr: LeaseRecord) -> None:
        self._c.execute(
            "INSERT INTO leases(id,reason,granted_at,expires_at,released_at) " "VALUES(?,?,?,?,?)",
            (lr.id, lr.reason, lr.granted_at, lr.expires_at, lr.released_at),
        )

    def get(self, lid: str) -> LeaseRecord | None:
        r = self._c.execute("SELECT * FROM leases WHERE id=?", (lid,)).fetchone()
        return _row_to_lease(r) if r else None

    def release(self, lid: str, when: str) -> None:
        self._c.execute("UPDATE leases SET released_at=? WHERE id=?", (when, lid))


class Store:
    def __init__(self, db: Path) -> None:
        self.db = db
        self._conn: sqlite3.Connection | None = None
        self.jobs: JobRepo
        self.leases: LeaseRepo

    def init(self) -> None:
        self.db.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None enables autocommit; required for our update_state
        # writes to be visible immediately to the test reader.
        self._conn = sqlite3.connect(self.db, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        for col, ddl in _MIGRATIONS:
            if col not in existing:
                self._conn.execute(ddl)
        self.jobs = JobRepo(self._conn)
        self.leases = LeaseRepo(self._conn)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
