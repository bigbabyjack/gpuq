"""Per-job log directory: split + combined files, append, tail, async follow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


class JobLogs:
    """Captures a job's stdout/stderr to disk and broadcasts to async followers.

    On disk:
        stdout.log     stdout only, plain text
        stderr.log     stderr only, plain text
        combined.log   merged plain text (what `gpuq logs -f` streams to users)
        index.jsonl    one {"stream","line"} record per append, for replay
    """

    def __init__(self, dirpath: Path) -> None:
        self.dir = dirpath
        self._stdout = None
        self._stderr = None
        self._combined = None
        self._index = None
        self._waiters: list[asyncio.Queue[tuple[str, str] | None]] = []

    def open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._stdout = (self.dir / "stdout.log").open("a", buffering=1)
        self._stderr = (self.dir / "stderr.log").open("a", buffering=1)
        self._combined = (self.dir / "combined.log").open("a", buffering=1)
        self._index = (self.dir / "index.jsonl").open("a", buffering=1)

    def append(self, stream: str, line: str) -> None:
        target = self._stdout if stream == "stdout" else self._stderr
        assert target is not None and self._combined is not None and self._index is not None
        target.write(line)
        self._combined.write(line)
        self._index.write(json.dumps({"stream": stream, "line": line}) + "\n")
        for q in list(self._waiters):
            q.put_nowait((stream, line))

    def eof(self) -> None:
        for q in list(self._waiters):
            q.put_nowait(None)

    def close(self) -> None:
        for f in (self._stdout, self._stderr, self._combined, self._index):
            if f:
                f.close()

    def read_all(self) -> list[tuple[str, str]]:
        idx = self.dir / "index.jsonl"
        if not idx.exists():
            return []
        out: list[tuple[str, str]] = []
        with idx.open() as f:
            for raw in f:
                rec = json.loads(raw)
                out.append((rec["stream"], rec["line"]))
        return out

    def tail(self, *, n: int = 100) -> list[tuple[str, str]]:
        return self.read_all()[-n:]

    async def follow(self):
        q: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self._waiters.append(q)
        try:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            if q in self._waiters:
                self._waiters.remove(q)
