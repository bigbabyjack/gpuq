import re

from gpuq.ids import new_job_id, new_lease_id


def test_job_id_format():
    jid = new_job_id()
    assert re.fullmatch(r"j_[0-9a-z]{6}", jid)


def test_job_id_unique():
    assert len({new_job_id() for _ in range(1000)}) == 1000


def test_lease_id_format():
    assert re.fullmatch(r"l_[0-9a-z]{6}", new_lease_id())
