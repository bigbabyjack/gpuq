from datetime import UTC, datetime, timedelta

import pytest

from gpuq.daemon.queue import Queue, wedge_signature
from gpuq.state import JobRecord, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db.sqlite")
    s.init()
    yield s
    s.close()


def _job(id_: str, prio: int = 0, t: int = 0) -> JobRecord:
    return JobRecord(
        id=id_,
        cmd=["x"],
        cwd="/",
        env={},
        tag="",
        priority=prio,
        state="queued",
        pid=None,
        exit_code=None,
        submitted_at=f"2026-01-01T00:00:{t:02d}",
        started_at=None,
        finished_at=None,
    )


def test_enqueue_and_pop_priority(store):
    q = Queue(store)
    q.enqueue(_job("j_a", prio=0, t=0))
    q.enqueue(_job("j_b", prio=5, t=1))
    first = q.pop()
    assert first is not None
    assert first.id == "j_b"
    # Mark first one done so next_queued moves on (queue.pop sets to "starting")
    store.jobs.update_state("j_b", state="done")
    second = q.pop()
    assert second is not None
    assert second.id == "j_a"
    store.jobs.update_state("j_a", state="done")
    assert q.pop() is None


def test_position(store):
    q = Queue(store)
    q.enqueue(_job("j_a", prio=0, t=0))
    q.enqueue(_job("j_b", prio=5, t=1))
    assert q.position("j_b") == 1
    assert q.position("j_a") == 2


def test_cancel_removes_queued(store):
    q = Queue(store)
    q.enqueue(_job("j_a"))
    assert q.cancel("j_a") is True
    assert q.pop() is None
    assert store.jobs.get("j_a").state == "cancelled"


def test_cancel_running_returns_false(store):
    q = Queue(store)
    q.enqueue(_job("j_a"))
    q.pop()  # j_a is now "starting"
    assert q.cancel("j_a") is False


def _running_job(*, pid, started_at, **kw) -> JobRecord:
    return JobRecord(
        id=kw.get("id", "j_x"),
        cmd=["x"],
        cwd="/",
        env={},
        tag="",
        priority=0,
        state="running",
        pid=pid,
        exit_code=None,
        submitted_at="2026-01-01T00:00:00+00:00",
        started_at=started_at,
        finished_at=None,
    )


def test_wedge_no_pid_no_logs_old_enough(tmp_path):
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=120)).isoformat()
    job = _running_job(pid=None, started_at=started)
    sig = wedge_signature(job, log_dir=tmp_path, now=now, alive=lambda _: False)
    assert sig == "wedged-no-pid"


def test_wedge_dead_pid(tmp_path):
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=120)).isoformat()
    job = _running_job(pid=999999, started_at=started)
    sig = wedge_signature(job, log_dir=tmp_path, now=now, alive=lambda _: False)
    assert sig == "wedged-dead-pid"


def test_not_wedged_when_pid_alive(tmp_path):
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=120)).isoformat()
    job = _running_job(pid=1, started_at=started)
    assert wedge_signature(job, log_dir=tmp_path, now=now, alive=lambda _: True) is None


def test_not_wedged_when_logs_nonempty(tmp_path):
    (tmp_path / "combined.log").write_text("hello\n")
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=120)).isoformat()
    job = _running_job(pid=None, started_at=started)
    assert wedge_signature(job, log_dir=tmp_path, now=now, alive=lambda _: False) is None


def test_not_wedged_when_under_min_age(tmp_path):
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=10)).isoformat()
    job = _running_job(pid=None, started_at=started)
    assert wedge_signature(job, log_dir=tmp_path, now=now, alive=lambda _: False) is None


def test_not_wedged_when_state_not_running(tmp_path):
    now = datetime(2026, 1, 1, 0, 5, 0, tzinfo=UTC)
    started = (now - timedelta(seconds=120)).isoformat()
    job = _running_job(pid=None, started_at=started)
    job_q = JobRecord(**{**job.__dict__, "state": "queued"})
    assert wedge_signature(job_q, log_dir=tmp_path, now=now, alive=lambda _: False) is None
