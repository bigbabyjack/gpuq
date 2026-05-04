import pytest

from gpuq.daemon.queue import Queue
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
