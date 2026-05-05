"""gpuq CLI (click-based, thin client over the unix socket)."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import click

from gpuq import paths
from gpuq import protocol as p


def _run(coro):
    return asyncio.run(coro)


def _parse_duration(s: str) -> float:
    """Parse '1d', '6h', '30m', '45s', or raw seconds → seconds."""
    s = s.strip()
    if not s:
        raise click.UsageError("empty duration")
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in units:
        try:
            return float(s[:-1]) * units[s[-1]]
        except ValueError as e:
            raise click.UsageError(f"bad duration: {s!r}") from e
    return float(s)


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s / 60)}m"
    return f"{s / 3600:.1f}h"


def _since_to_iso(s: str) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime.now(UTC) - timedelta(seconds=_parse_duration(s))).isoformat()


async def _open():
    sock = paths.socket_path()
    if not sock.exists():
        click.echo(f"[gpuq] daemon not running at {sock}", err=True)
        click.echo("[gpuq] start it with: systemctl --user start gpuq", err=True)
        sys.exit(64)
    return await asyncio.open_unix_connection(str(sock))


async def _send(req: p.Request):
    reader, writer = await _open()
    writer.write(p.encode_request(req).encode())
    await writer.drain()
    return reader, writer


async def _recv_one(reader) -> p.Event | None:
    line = await reader.readline()
    return p.decode_event(line.decode()) if line else None


@click.group()
def main() -> None:
    """gpuq — single-GPU job scheduler."""


@main.command()
@click.option("-t", "--tag", default="")
@click.option("-p", "--priority", type=int, default=0)
@click.option("--next", "next_", is_flag=True, help="Jump ahead of all queued jobs.")
@click.option("--detach", is_flag=True)
@click.option("--describe", default="", help="Long-form description for history queries.")
@click.argument("cmd", nargs=-1, required=True)
def submit(
    tag: str,
    priority: int,
    next_: bool,
    detach: bool,
    describe: str,
    cmd: tuple[str, ...],
) -> None:
    """Submit a command to the queue. Streams output and exits with its code."""
    if next_ and priority != 0:
        raise click.UsageError("--next and --priority are mutually exclusive")

    async def run():
        req = p.Submit(
            cmd=list(cmd),
            cwd=os.getcwd(),
            env=dict(os.environ),
            tag=tag,
            priority=priority,
            detach=detach,
            next_=next_,
            description=describe,
        )
        reader, writer = await _send(req)
        exit_code = 0
        try:
            while (ev := await _recv_one(reader)) is not None:
                if isinstance(ev, p.LogEvent):
                    target = sys.stdout if ev.stream == "stdout" else sys.stderr
                    target.write(ev.line)
                    target.flush()
                elif isinstance(ev, p.StateEvent):
                    parts = [f"[gpuq] {ev.state} {ev.id}"]
                    if ev.position is not None:
                        parts.append(f"pos={ev.position}")
                    if ev.eta_running_s is not None:
                        parts.append(f"eta_running={_fmt_dur(ev.eta_running_s)}")
                    if ev.eta_start_s is not None:
                        parts.append(f"eta_start={_fmt_dur(ev.eta_start_s)}")
                    if ev.exit_code is not None:
                        parts.append(f"exit={ev.exit_code}")
                    click.echo(" ".join(parts), err=True)
                    if ev.exit_code is not None:
                        exit_code = ev.exit_code
                elif isinstance(ev, p.OllamaEvent):
                    click.echo(f"[gpuq] ollama {ev.action} {ev.models}", err=True)
                elif isinstance(ev, p.ErrorEvent):
                    click.echo(f"[gpuq] error: {ev.message}", err=True)
                    exit_code = 1
        finally:
            writer.close()
        sys.exit(exit_code)

    _run(run())


@main.command()
@click.option("--tag", default=None)
@click.option("--state", default=None, help="Single state filter (back-compat).")
@click.option(
    "--states",
    "states_csv",
    default=None,
    help="Comma-separated states (e.g. running,queued,failed).",
)
@click.option("-a", "--all", "all_", is_flag=True, help="Include terminal jobs.")
@click.option(
    "--since",
    default=None,
    help="Only jobs submitted since this duration (e.g. 1d, 6h, 30m).",
)
@click.option("--json", "as_json", is_flag=True)
def ps(
    tag: str | None,
    state: str | None,
    states_csv: str | None,
    all_: bool,
    since: str | None,
    as_json: bool,
) -> None:
    """List jobs. By default shows queued + running; --all for history."""

    async def run():
        states: list[str] = []
        if states_csv:
            states = [s.strip() for s in states_csv.split(",") if s.strip()]
        elif state:
            pass  # use single-state field
        elif not all_:
            states = ["queued", "starting", "running"]
        since_iso = _since_to_iso(since) if since else None
        reader, writer = await _send(p.Ps(tag=tag, state=state, states=states, since=since_iso))
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"jobs": []}
        if as_json:
            click.echo(json.dumps(payload))
        else:
            for j in payload.get("jobs", []):
                click.echo(
                    f"{j['id']}  {j['state']:9}  p={j['priority']}  "
                    f"{j.get('tag', '')}  {' '.join(j['cmd'])}"
                )

    _run(run())


@main.command()
@click.argument("id")
@click.option("--json", "as_json", is_flag=True)
def show(id: str, as_json: bool) -> None:
    """Show one job."""

    async def run():
        reader, writer = await _send(p.Show(id=id))
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"job": None}
        click.echo(json.dumps(payload, indent=None if as_json else 2))

    _run(run())


@main.command()
@click.argument("id")
@click.option("-f", "--follow", is_flag=True)
@click.option("--from", "from_", type=click.Choice(["start", "tail"]), default="tail")
@click.option("-n", "--tail", "tail", type=int, default=200, help="Last N lines (default 200).")
def logs(id: str, follow: bool, from_: str, tail: int) -> None:
    """Stream job logs."""

    async def run():
        # Probe for empty-output state up front, so we can emit a placeholder.
        log_path = paths.log_dir(id)
        combined = log_path / "combined.log"
        empty = (not combined.exists()) or combined.stat().st_size == 0
        if empty:
            show_reader, show_writer = await _send(p.Show(id=id))
            ev = await _recv_one(show_reader)
            show_writer.close()
            job = (
                ev.payload.get("job")
                if isinstance(ev, p.ResultEvent) and ev.payload.get("job")
                else None
            )
            state = job["state"] if job else "unknown"
            started = job.get("started_at") if job else None
            click.echo(
                f"[gpuq] no output yet for {id} (state={state}, started={started})",
                err=True,
            )

        reader, writer = await _send(p.Attach(id=id, follow=follow, from_=from_, tail=tail))
        try:
            while (ev := await _recv_one(reader)) is not None:
                if isinstance(ev, p.LogEvent):
                    target = sys.stdout if ev.stream == "stdout" else sys.stderr
                    target.write(ev.line)
                    target.flush()
        finally:
            writer.close()

    _run(run())


@main.command()
@click.argument("id")
@click.option("--kill", is_flag=True)
def cancel(id: str, kill: bool) -> None:
    """Cancel a job."""

    async def run():
        reader, writer = await _send(
            p.Cancel(
                id=id,
                signal="KILL" if kill else "TERM",
            )
        )
        ev = await _recv_one(reader)
        writer.close()
        if isinstance(ev, p.ResultEvent) and ev.payload.get("ok"):
            click.echo(f"[gpuq] cancelled {id}", err=True)
        else:
            click.echo(f"[gpuq] could not cancel {id}", err=True)
            sys.exit(1)

    _run(run())


@main.group()
def ollama() -> None:
    """Ollama controls."""


@ollama.command("evict")
def ollama_evict() -> None:
    async def run():
        reader, writer = await _send(p.EvictOllama())
        ev = await _recv_one(reader)
        writer.close()
        click.echo(json.dumps(ev.payload if isinstance(ev, p.ResultEvent) else {}))

    _run(run())


@ollama.command("restore")
def ollama_restore() -> None:
    async def run():
        reader, writer = await _send(p.RestoreOllama())
        await _recv_one(reader)
        writer.close()

    _run(run())


@main.command()
@click.argument("id")
def bump(id: str) -> None:
    """Move a queued job to the front of the queue."""

    async def run():
        reader, writer = await _send(p.Bump(id=id))
        ev = await _recv_one(reader)
        writer.close()
        if isinstance(ev, p.ResultEvent) and ev.payload.get("ok"):
            click.echo(f"[gpuq] bumped {id}", err=True)
        else:
            click.echo(f"[gpuq] could not bump {id} (not queued?)", err=True)
            sys.exit(1)

    _run(run())


@main.command()
@click.argument("id")
@click.option("--force", is_flag=True, help="Allow retry of done jobs.")
def retry(id: str, force: bool) -> None:
    """Resubmit a failed/cancelled job."""

    async def run():
        reader, writer = await _send(p.Retry(id=id, force=force))
        ev = await _recv_one(reader)
        writer.close()
        payload: dict = ev.payload if isinstance(ev, p.ResultEvent) else {"ok": False}
        if not payload.get("ok"):
            click.echo(
                f"[gpuq] could not retry {id}: {payload.get('reason', 'unknown')}",
                err=True,
            )
            sys.exit(1)
        job = payload.get("job") or {}
        click.echo(job.get("id", ""))

    _run(run())


@main.command()
@click.option("--older-than", "older_than", default="7d", help="Age cutoff (e.g. 1d, 6h).")
@click.option("--state", "states_csv", default=None, help="Comma-separated states to prune.")
@click.option("--keep-tag", default="", help="Skip jobs with this tag.")
@click.option("--dry-run", is_flag=True)
def prune(older_than: str, states_csv: str | None, keep_tag: str, dry_run: bool) -> None:
    """Delete done/failed/cancelled jobs older than the cutoff."""

    async def run():
        states = []
        if states_csv:
            states = [s.strip() for s in states_csv.split(",") if s.strip()]
        secs = _parse_duration(older_than)
        reader, writer = await _send(
            p.Prune(
                older_than_s=secs,
                states=states,
                keep_tag=keep_tag,
                dry_run=dry_run,
            )
        )
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"deleted": []}
        ids = payload.get("deleted", [])
        verb = "would delete" if payload.get("dry_run") else "deleted"
        click.echo(f"[gpuq] {verb} {len(ids)} job(s)", err=True)
        for jid in ids:
            click.echo(jid)

    _run(run())


@main.command()
@click.option("--fix", is_flag=True, help="Transition wedged jobs to failed.")
@click.option("--id", "id_", default=None, help="Restrict to a single job id.")
@click.option("--min-age", "min_age", default=60.0, type=float, help="Min age in seconds.")
@click.option("--json", "as_json", is_flag=True)
def doctor(fix: bool, id_: str | None, min_age: float, as_json: bool) -> None:
    """Detect (and optionally clear) wedged jobs."""

    async def run():
        ids = [id_] if id_ else []
        reader, writer = await _send(p.Doctor(fix=fix, ids=ids, min_age_s=min_age))
        ev = await _recv_one(reader)
        writer.close()
        payload = ev.payload if isinstance(ev, p.ResultEvent) else {"wedged": []}
        if as_json:
            click.echo(json.dumps(payload))
            return
        wedged = payload.get("wedged", [])
        if not wedged:
            click.echo("[gpuq] no wedged jobs", err=True)
            return
        for w in wedged:
            click.echo(
                f"{w['id']}  running  pid={w['pid']}  started {w['started_at']}\n"
                f"  signature: {w['signature']}" + ("  fixed" if w.get("fixed") else "")
            )
        if not fix:
            click.echo(
                f"\n{len(wedged)} job(s) wedged. Re-run with --fix to transition to failed.",
                err=True,
            )

    _run(run())


@main.command()
def daemon() -> None:
    """Run the daemon in the foreground (used by systemd)."""
    from gpuq.daemon.main import main as daemon_main

    daemon_main()
