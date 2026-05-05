from datetime import UTC, datetime

import pytest

from gpuq.state import JobRecord, LeaseRecord, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "state.db")
    s.init()
    yield s
    s.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _job(id_: str, *, prio: int = 0, t: int = 0, state: str = "queued") -> JobRecord:
    return JobRecord(
        id=id_,
        cmd=["x"],
        cwd="/",
        env={},
        tag="",
        priority=prio,
        state=state,
        pid=None,
        exit_code=None,
        submitted_at=f"2026-01-01T00:00:{t:02d}",
        started_at=None,
        finished_at=None,
    )


def test_insert_and_fetch_job(store):
    rec = JobRecord(
        id="j_abc123",
        cmd=["echo", "hi"],
        cwd="/tmp",
        env={"A": "1"},
        tag="t",
        priority=0,
        state="queued",
        pid=None,
        exit_code=None,
        submitted_at=_now(),
        started_at=None,
        finished_at=None,
    )
    store.jobs.insert(rec)
    assert store.jobs.get("j_abc123") == rec


def test_list_jobs_filtered_by_state(store):
    store.jobs.insert(_job("j_a", state="queued"))
    store.jobs.insert(_job("j_b", state="running"))
    store.jobs.insert(_job("j_c", state="done"))
    assert {j.id for j in store.jobs.find(state="queued")} == {"j_a"}


def test_update_job_state(store):
    store.jobs.insert(_job("j_x"))
    store.jobs.update_state("j_x", state="running", pid=99, started_at=_now())
    got = store.jobs.get("j_x")
    assert got.state == "running"
    assert got.pid == 99


def test_pop_next_queued_priority_then_fifo(store):
    for i, prio in enumerate([0, 5, 5, 1]):
        store.jobs.insert(_job(f"j_{i:06d}", prio=prio, t=i))
    order = []
    while nxt := store.jobs.next_queued():
        order.append(nxt.id)
        store.jobs.update_state(nxt.id, state="done")
    assert order == ["j_000001", "j_000002", "j_000003", "j_000000"]


def test_lease_insert_and_release(store):
    rec = LeaseRecord(id="l_x", reason="r", granted_at=_now(), expires_at=None, released_at=None)
    store.leases.insert(rec)
    store.leases.release("l_x", _now())
    assert store.leases.get("l_x").released_at is not None


def test_env_and_cmd_roundtrip(store):
    rec = JobRecord(
        id="j_env",
        cmd=["python", "-c", "print(1)"],
        cwd="/",
        env={"K": "v with spaces", "N": "42"},
        tag="",
        priority=0,
        state="queued",
        pid=None,
        exit_code=None,
        submitted_at=_now(),
        started_at=None,
        finished_at=None,
    )
    store.jobs.insert(rec)
    got = store.jobs.get("j_env")
    assert got.env == {"K": "v with spaces", "N": "42"}
    assert got.cmd == ["python", "-c", "print(1)"]


def test_max_queued_priority_none_when_empty(store):
    assert store.jobs.max_queued_priority() is None


def test_max_queued_priority_ignores_non_queued(store):
    store.jobs.insert(_job("j_a", prio=3, state="queued"))
    store.jobs.insert(_job("j_b", prio=10, state="running"))
    store.jobs.insert(_job("j_c", prio=99, state="done"))
    assert store.jobs.max_queued_priority() == 3


def test_bump_sets_priority_above_queued_max(store):
    store.jobs.insert(_job("j_a", prio=0, t=0))
    store.jobs.insert(_job("j_b", prio=5, t=1))
    store.jobs.insert(_job("j_c", prio=2, t=2))
    assert store.jobs.bump("j_c") is True
    assert store.jobs.get("j_c").priority == 6


def test_bump_only_queued_job_sets_priority_to_one(store):
    store.jobs.insert(_job("j_a", prio=0, t=0))
    assert store.jobs.bump("j_a") is True
    # max_queued_priority excluding self is None → priority becomes 1
    assert store.jobs.get("j_a").priority == 1


def test_bump_refuses_non_queued(store):
    store.jobs.insert(_job("j_a", state="running"))
    assert store.jobs.bump("j_a") is False


def test_bump_missing_returns_false(store):
    assert store.jobs.bump("j_nope") is False


def test_parent_id_and_description_round_trip(store):
    rec = JobRecord(
        id="j_child",
        cmd=["x"],
        cwd="/",
        env={},
        tag="",
        priority=0,
        state="queued",
        pid=None,
        exit_code=None,
        submitted_at=_now(),
        started_at=None,
        finished_at=None,
        parent_id="j_parent",
        description="run a thing",
    )
    store.jobs.insert(rec)
    got = store.jobs.get("j_child")
    assert got.parent_id == "j_parent"
    assert got.description == "run a thing"


def test_find_supports_states_list_and_since(store):
    store.jobs.insert(_job("j_a", state="queued", t=1))
    store.jobs.insert(_job("j_b", state="failed", t=2))
    store.jobs.insert(_job("j_c", state="done", t=3))
    ids = {j.id for j in store.jobs.find(states=["queued", "failed"])}
    assert ids == {"j_a", "j_b"}
    ids = {j.id for j in store.jobs.find(since="2026-01-01T00:00:02")}
    assert ids == {"j_b", "j_c"}


def test_delete_job_removes_row(store):
    store.jobs.insert(_job("j_a"))
    assert store.jobs.delete("j_a") is True
    assert store.jobs.get("j_a") is None
    assert store.jobs.delete("j_a") is False


def test_migration_idempotent_on_existing_db(tmp_path):
    db = tmp_path / "state.db"
    s1 = Store(db)
    s1.init()
    s1.close()
    s2 = Store(db)
    s2.init()
    cols = {row["name"] for row in s2._conn.execute("PRAGMA table_info(jobs)")}
    assert "parent_id" in cols and "description" in cols
    s2.close()


def test_running_lists_starting_and_running(store):
    store.jobs.insert(_job("j_q", state="queued"))
    store.jobs.insert(_job("j_s", state="starting"))
    store.jobs.insert(_job("j_r", state="running"))
    ids = {j.id for j in store.jobs.running()}
    assert ids == {"j_s", "j_r"}
