from __future__ import annotations

import pytest


@pytest.fixture
def gpuq_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
async def daemon(gpuq_home):
    from gpuq.daemon.server import Server

    server = Server()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
