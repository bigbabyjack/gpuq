"""Library-side lease context manager."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from gpuq import paths, protocol as p


async def _acquire(reason: str) -> str:
    r, w = await asyncio.open_unix_connection(str(paths.socket_path()))
    w.write(p.encode_request(p.Lease(reason=reason)).encode())
    await w.drain()
    line = await r.readline()
    ev = p.decode_event(line.decode())
    w.close()
    await w.wait_closed()
    assert isinstance(ev, p.ResultEvent)
    return ev.payload["lease_id"]


async def _release(lid: str) -> None:
    r, w = await asyncio.open_unix_connection(str(paths.socket_path()))
    w.write(p.encode_request(p.Release(lease_id=lid)).encode())
    await w.drain()
    await r.readline()
    w.close()
    await w.wait_closed()


@contextmanager
def lease(reason: str = ""):
    """Block until the GPU is free, hold it for the duration of the with-block."""
    lid = asyncio.run(_acquire(reason))
    try:
        yield lid
    finally:
        asyncio.run(_release(lid))
