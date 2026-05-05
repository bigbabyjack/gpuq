"""Asyncio unix-socket server. Owns the queue, runs jobs serially, fans out events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from datetime import UTC, datetime
from datetime import timedelta as _timedelta
from typing import Any

from gpuq import paths
from gpuq import protocol as p
from gpuq.daemon.queue import Queue, recent_mean_runtime, wedge_signature
from gpuq.daemon.runner import Runner
from gpuq.ids import new_job_id, new_lease_id
from gpuq.logs import JobLogs
from gpuq.ollama import OllamaController
from gpuq.state import JobRecord, LeaseRecord, Store

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _timedelta_seconds(s: float) -> _timedelta:
    return _timedelta(seconds=s)


class Bus:
    """Fan-out per-job events to subscribed asyncio.Queues."""

    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, jid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[jid].append(q)
        return q

    def unsubscribe(self, jid: str, q: asyncio.Queue) -> None:
        if q in self._subs.get(jid, []):
            self._subs[jid].remove(q)

    def publish(self, jid: str, ev: p.Event) -> None:
        for q in list(self._subs.get(jid, [])):
            q.put_nowait(ev)

    def close(self, jid: str) -> None:
        for q in list(self._subs.get(jid, [])):
            q.put_nowait(None)


class Server:
    def __init__(self) -> None:
        self.store = Store(paths.db_path())
        self.queue: Queue
        self.bus = Bus()
        self.ollama = OllamaController()
        self._server: asyncio.base_events.Server | None = None
        self._scheduler_task: asyncio.Task | None = None
        self._current_runner: Runner | None = None
        self._current_job_id: str | None = None
        self._lease_held = asyncio.Event()
        self._lease_id: str | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        paths.ensure()
        sock = paths.socket_path()
        if sock.exists():
            sock.unlink()
        self.store.init()
        self.queue = Queue(self.store)
        self._reconcile_on_startup()
        self._server = await asyncio.start_unix_server(self._handle, path=str(sock))
        self._scheduler_task = asyncio.create_task(self._scheduler())

    async def stop(self) -> None:
        self._stopping.set()
        self.queue.notify.set()
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._scheduler_task
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self.store.close()
        sock = paths.socket_path()
        if sock.exists():
            sock.unlink()

    def _reconcile_on_startup(self) -> None:
        for j in self.store.jobs.running():
            self.store.jobs.update_state(j.id, state="failed", exit_code=-1, finished_at=_now())

    async def _scheduler(self) -> None:
        while not self._stopping.is_set():
            await self.queue.notify.wait()
            if self._stopping.is_set():
                return
            if self._lease_held.is_set():
                await self._wait_lease_released()
                continue
            job = self.queue.pop()
            if job is None:
                continue
            try:
                await self._run_job(job)
            except Exception:
                log.exception("scheduler: _run_job crashed for %s", job.id)
                with contextlib.suppress(Exception):
                    self.store.jobs.update_state(
                        job.id, state="failed", exit_code=-1, finished_at=_now()
                    )

    async def _wait_lease_released(self) -> None:
        while self._lease_held.is_set() and not self._stopping.is_set():
            await asyncio.sleep(0.02)

    async def _run_job(self, job: JobRecord) -> None:
        evicted = await self.ollama.evict()
        if evicted:
            self.bus.publish(job.id, p.OllamaEvent(action="evicted", models=evicted))

        job_logs = JobLogs(paths.log_dir(job.id))
        job_logs.open()
        runner = Runner()
        self._current_runner = runner
        self._current_job_id = job.id

        async def mirror():
            async for stream, line in job_logs.follow():
                self.bus.publish(
                    job.id,
                    p.LogEvent(id=job.id, stream=stream, line=line),
                )

        mirror_task = asyncio.create_task(mirror())
        rc: int = -1
        dur: float = 0.0
        spawn_error: str | None = None
        try:
            self.store.jobs.update_state(job.id, state="running", started_at=_now())
            self.bus.publish(
                job.id,
                p.StateEvent(id=job.id, state="running", started_at=_now()),
            )
            try:
                run_task = asyncio.create_task(
                    runner.run(cmd=job.cmd, cwd=job.cwd, env=job.env, logs=job_logs)
                )
                for _ in range(50):
                    if runner.pid is not None:
                        self.store.jobs.update_state(job.id, state="running", pid=runner.pid)
                        break
                    await asyncio.sleep(0.01)
                rc, dur = await run_task
            except Exception as e:
                spawn_error = repr(e)
                log.exception("job %s: runner crashed", job.id)
        finally:
            job_logs.eof()
            with contextlib.suppress(Exception):
                await mirror_task
            job_logs.close()
            self._current_runner = None
            self._current_job_id = None

            final_state = "done" if (rc == 0 and spawn_error is None) else "failed"
            with contextlib.suppress(Exception):
                self.store.jobs.update_state(
                    job.id,
                    state=final_state,
                    exit_code=rc,
                    finished_at=_now(),
                )
                self.bus.publish(
                    job.id,
                    p.StateEvent(
                        id=job.id,
                        state=final_state,
                        exit_code=rc,
                        duration_s=dur,
                    ),
                )
                self.bus.close(job.id)

            if self.store.jobs.next_queued() is None and not self._lease_held.is_set():
                with contextlib.suppress(Exception):
                    await self.ollama.restore()

    # -- connection handler ---------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                req = p.decode_request(line.decode())
            except p.ProtocolError as e:
                _send(writer, p.ErrorEvent(message=str(e)))
                return
            await self._dispatch(req, writer)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            log.exception("handler error")
            with contextlib.suppress(Exception):
                _send(writer, p.ErrorEvent(message=f"server error: {e}"))
        finally:
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _dispatch(self, req: p.Request, writer: asyncio.StreamWriter) -> None:
        if isinstance(req, p.Submit):
            await self._handle_submit(req, writer)
        elif isinstance(req, p.Bump):
            await self._handle_bump(req, writer)
        elif isinstance(req, p.Attach):
            await self._handle_attach(req, writer)
        elif isinstance(req, p.Ps):
            await self._handle_ps(req, writer)
        elif isinstance(req, p.Show):
            await self._handle_show(req, writer)
        elif isinstance(req, p.Cancel):
            await self._handle_cancel(req, writer)
        elif isinstance(req, p.Doctor):
            await self._handle_doctor(req, writer)
        elif isinstance(req, p.Retry):
            await self._handle_retry(req, writer)
        elif isinstance(req, p.Prune):
            await self._handle_prune(req, writer)
        elif isinstance(req, p.Lease):
            await self._handle_lease(req, writer)
        elif isinstance(req, p.Release):
            await self._handle_release(req, writer)
        elif isinstance(req, p.EvictOllama):
            evicted = await self.ollama.evict()
            _send(
                writer,
                p.ResultEvent(
                    payload={
                        "evicted": bool(evicted),
                        "was_loaded": evicted,
                    }
                ),
            )
        elif isinstance(req, p.RestoreOllama):
            await self.ollama.restore()
            _send(writer, p.ResultEvent(payload={"ok": True}))

    async def _handle_submit(self, req: p.Submit, writer: asyncio.StreamWriter) -> None:
        jid = new_job_id()
        if req.next_:
            top = self.store.jobs.max_queued_priority()
            priority = (top + 1) if top is not None else 1
        else:
            priority = req.priority
        rec = JobRecord(
            id=jid,
            cmd=req.cmd,
            cwd=req.cwd,
            env=req.env,
            tag=req.tag,
            priority=priority,
            state="queued",
            pid=None,
            exit_code=None,
            submitted_at=_now(),
            started_at=None,
            finished_at=None,
            parent_id=req.parent_id or None,
            description=req.description or None,
        )
        sub = self.bus.subscribe(jid)
        self.queue.enqueue(rec)
        pos = self.queue.position(jid)
        eta_running, eta_start = self._compute_eta(rec, pos)
        _send(
            writer,
            p.StateEvent(
                id=jid,
                state="queued",
                position=pos,
                eta_running_s=eta_running,
                eta_start_s=eta_start,
            ),
        )
        await writer.drain()
        if req.detach:
            self.bus.unsubscribe(jid, sub)
            return
        await self._stream_until_terminal(jid, sub, writer)

    async def _stream_until_terminal(
        self, jid: str, sub: asyncio.Queue, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                ev = await sub.get()
                if ev is None:
                    return
                _send(writer, ev)
                await writer.drain()
                if isinstance(ev, p.StateEvent) and ev.state in ("done", "failed", "cancelled"):
                    return
        finally:
            self.bus.unsubscribe(jid, sub)

    async def _handle_attach(self, req: p.Attach, writer: asyncio.StreamWriter) -> None:
        sub = self.bus.subscribe(req.id)
        log_path = paths.log_dir(req.id)
        if log_path.exists() and req.from_ in ("start", "tail"):
            jl = JobLogs(log_path)
            history = jl.read_all() if req.from_ == "start" else jl.tail(n=req.tail)
            for stream, line in history:
                _send(writer, p.LogEvent(id=req.id, stream=stream, line=line))
            await writer.drain()
        if req.follow:
            await self._stream_until_terminal(req.id, sub, writer)
        else:
            self.bus.unsubscribe(req.id, sub)

    async def _handle_ps(self, req: p.Ps, writer: asyncio.StreamWriter) -> None:
        jobs = self.store.jobs.find(
            state=req.state,
            states=req.states or None,
            tag=req.tag,
            since=req.since,
        )
        _send(
            writer,
            p.ResultEvent(
                payload={
                    "jobs": [_job_dict(j) for j in jobs],
                }
            ),
        )

    async def _handle_show(self, req: p.Show, writer: asyncio.StreamWriter) -> None:
        rec = self.store.jobs.get(req.id)
        _send(writer, p.ResultEvent(payload={"job": _job_dict(rec)}))

    async def _handle_bump(self, req: p.Bump, writer: asyncio.StreamWriter) -> None:
        ok = self.store.jobs.bump(req.id)
        payload: dict[str, Any] = {"ok": ok}
        if ok:
            payload["job"] = _job_dict(self.store.jobs.get(req.id))
        _send(writer, p.ResultEvent(payload=payload))

    async def _handle_cancel(self, req: p.Cancel, writer: asyncio.StreamWriter) -> None:
        rec = self.store.jobs.get(req.id)
        if rec is None:
            _send(writer, p.ResultEvent(payload={"ok": False, "reason": "unknown"}))
            return
        if rec.state == "queued":
            self.queue.cancel(req.id)
            self.bus.publish(req.id, p.StateEvent(id=req.id, state="cancelled"))
            self.bus.close(req.id)
            _send(
                writer,
                p.ResultEvent(payload={"ok": True, "job": _job_dict(self.store.jobs.get(req.id))}),
            )
            return
        if rec.state in ("running", "starting") and self._current_job_id == req.id:
            assert self._current_runner is not None
            timeout = max(req.timeout, 0.0)
            await self._current_runner.signal(req.signal)
            transitioned = await self._wait_terminal(req.id, timeout)
            if not transitioned and req.signal != "KILL":
                await self._current_runner.signal("KILL")
                transitioned = await self._wait_terminal(req.id, min(timeout, 5.0))
            if transitioned:
                _send(
                    writer,
                    p.ResultEvent(
                        payload={"ok": True, "job": _job_dict(self.store.jobs.get(req.id))}
                    ),
                )
            else:
                _send(writer, p.ResultEvent(payload={"ok": False, "reason": "wedged"}))
            return
        if rec.state in ("done", "failed", "cancelled"):
            _send(writer, p.ResultEvent(payload={"ok": False, "reason": "already_terminal"}))
            return
        _send(writer, p.ResultEvent(payload={"ok": False, "reason": "not_current"}))

    def _compute_eta(self, rec: JobRecord, pos: int | None) -> tuple[float | None, float | None]:
        mean = recent_mean_runtime(self.store, tag=rec.tag)
        if mean is None:
            mean = recent_mean_runtime(self.store, tag="")
        if mean is None:
            return None, None
        running_remaining = 0.0
        if self._current_job_id is not None:
            cur = self.store.jobs.get(self._current_job_id)
            if cur and cur.started_at:
                try:
                    age = (
                        datetime.now(UTC) - datetime.fromisoformat(cur.started_at)
                    ).total_seconds()
                except ValueError:
                    age = 0.0
                running_remaining = max(mean - age, 0.0)
        eta_running = running_remaining
        ahead = max((pos or 1) - 1, 0)
        eta_start = running_remaining + ahead * mean
        return eta_running, eta_start

    async def _wait_terminal(self, jid: str, timeout: float) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            rec = self.store.jobs.get(jid)
            if rec and rec.state in ("done", "failed", "cancelled"):
                return True
            if self._current_job_id != jid:
                rec = self.store.jobs.get(jid)
                if rec and rec.state in ("done", "failed", "cancelled"):
                    return True
            await asyncio.sleep(0.05)
        rec = self.store.jobs.get(jid)
        return bool(rec and rec.state in ("done", "failed", "cancelled"))

    async def _handle_doctor(self, req: p.Doctor, writer: asyncio.StreamWriter) -> None:
        now = datetime.now(UTC)
        candidates = self.store.jobs.find(state="running")
        if req.ids:
            ids = set(req.ids)
            candidates = [j for j in candidates if j.id in ids]
        wedged: list[dict[str, Any]] = []
        for j in candidates:
            sig = wedge_signature(
                j,
                log_dir=paths.log_dir(j.id),
                now=now,
                min_age_s=req.min_age_s,
            )
            if sig is None:
                continue
            entry = {
                "id": j.id,
                "signature": sig,
                "pid": j.pid,
                "started_at": j.started_at,
            }
            if req.fix:
                # Don't reap the currently-running job — it's not actually wedged
                # if the daemon thinks it owns it.
                if self._current_job_id == j.id:
                    entry["fixed"] = False
                    entry["reason"] = "current_job"
                else:
                    self.store.jobs.update_state(
                        j.id,
                        state="failed",
                        exit_code=-1,
                        finished_at=_now(),
                    )
                    entry["fixed"] = True
            wedged.append(entry)
        _send(writer, p.ResultEvent(payload={"wedged": wedged, "fix": req.fix}))

    async def _handle_retry(self, req: p.Retry, writer: asyncio.StreamWriter) -> None:
        rec = self.store.jobs.get(req.id)
        if rec is None:
            _send(writer, p.ResultEvent(payload={"ok": False, "reason": "unknown"}))
            return
        allowed = {"failed", "cancelled"} if not req.force else {"failed", "cancelled", "done"}
        if rec.state not in allowed:
            _send(
                writer,
                p.ResultEvent(payload={"ok": False, "reason": f"state={rec.state}"}),
            )
            return
        new_id = new_job_id()
        new_rec = JobRecord(
            id=new_id,
            cmd=rec.cmd,
            cwd=rec.cwd,
            env=rec.env,
            tag=rec.tag,
            priority=rec.priority,
            state="queued",
            pid=None,
            exit_code=None,
            submitted_at=_now(),
            started_at=None,
            finished_at=None,
            parent_id=rec.id,
            description=rec.description,
        )
        self.queue.enqueue(new_rec)
        _send(
            writer,
            p.ResultEvent(payload={"ok": True, "job": _job_dict(self.store.jobs.get(new_id))}),
        )

    async def _handle_prune(self, req: p.Prune, writer: asyncio.StreamWriter) -> None:
        import shutil

        states = req.states or ["done", "failed", "cancelled"]
        states = [s for s in states if s not in ("queued", "running", "starting")]
        cutoff = (datetime.now(UTC) - _timedelta_seconds(req.older_than_s)).isoformat()
        candidates = self.store.jobs.find(states=states)
        deleted: list[str] = []
        for j in candidates:
            if req.keep_tag and j.tag == req.keep_tag:
                continue
            stamp = j.finished_at or j.submitted_at
            if stamp >= cutoff:
                continue
            deleted.append(j.id)
            if not req.dry_run:
                self.store.jobs.delete(j.id)
                log_dir = paths.log_dir(j.id)
                if log_dir.exists():
                    shutil.rmtree(log_dir, ignore_errors=True)
        _send(
            writer,
            p.ResultEvent(payload={"deleted": deleted, "dry_run": req.dry_run}),
        )

    async def _handle_lease(self, req: p.Lease, writer: asyncio.StreamWriter) -> None:
        while self._current_job_id is not None or self._lease_held.is_set():
            await asyncio.sleep(0.02)
        await self.ollama.evict()
        lid = new_lease_id()
        granted = _now()
        self.store.leases.insert(
            LeaseRecord(
                id=lid,
                reason=req.reason or None,
                granted_at=granted,
                expires_at=None,
                released_at=None,
            )
        )
        self._lease_held.set()
        self._lease_id = lid
        _send(
            writer,
            p.ResultEvent(
                payload={
                    "lease_id": lid,
                    "granted_at": granted,
                    "expires_at": None,
                }
            ),
        )

    async def _handle_release(self, req: p.Release, writer: asyncio.StreamWriter) -> None:
        if self._lease_id == req.lease_id:
            self.store.leases.release(req.lease_id, _now())
            self._lease_held.clear()
            self._lease_id = None
            self.queue.notify.set()
            _send(writer, p.ResultEvent(payload={"ok": True}))
        else:
            _send(writer, p.ResultEvent(payload={"ok": False}))


def _job_dict(j: JobRecord | None) -> dict[str, Any] | None:
    if j is None:
        return None
    return {
        "id": j.id,
        "cmd": j.cmd,
        "cwd": j.cwd,
        "tag": j.tag,
        "priority": j.priority,
        "state": j.state,
        "pid": j.pid,
        "exit_code": j.exit_code,
        "submitted_at": j.submitted_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "parent_id": j.parent_id,
        "description": j.description,
    }


def _send(writer: asyncio.StreamWriter, ev: p.Event) -> None:
    writer.write(p.encode_event(ev).encode())
