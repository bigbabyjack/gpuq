import asyncio

import pytest

from gpuq.logs import JobLogs


def test_append_and_read_combined(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    j.append("stdout", "hello\n")
    j.append("stderr", "err\n")
    j.close()
    assert (tmp_path / "j_x" / "combined.log").read_text() == "hello\nerr\n"
    assert (tmp_path / "j_x" / "stdout.log").read_text() == "hello\n"
    assert (tmp_path / "j_x" / "stderr.log").read_text() == "err\n"


def test_tail_last_n(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    for i in range(50):
        j.append("stdout", f"line {i}\n")
    j.close()
    tail = JobLogs(tmp_path / "j_x").tail(n=5)
    assert tail == [("stdout", f"line {i}\n") for i in range(45, 50)]


def test_read_all_preserves_stream_and_order(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    j.append("stdout", "a\n")
    j.append("stderr", "b\n")
    j.append("stdout", "c\n")
    j.close()
    assert JobLogs(tmp_path / "j_x").read_all() == [
        ("stdout", "a\n"),
        ("stderr", "b\n"),
        ("stdout", "c\n"),
    ]


@pytest.mark.asyncio
async def test_follow_yields_new_lines(tmp_path):
    j = JobLogs(tmp_path / "j_x")
    j.open()
    out: list[tuple[str, str]] = []

    async def consume():
        async for stream, line in j.follow():
            out.append((stream, line))

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    j.append("stdout", "live\n")
    await asyncio.sleep(0.01)
    j.eof()
    await consumer
    j.close()
    assert ("stdout", "live\n") in out
