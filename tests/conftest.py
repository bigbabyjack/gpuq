from __future__ import annotations

import pytest


@pytest.fixture
async def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from gpuq.daemon.server import Server

    server = Server()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
