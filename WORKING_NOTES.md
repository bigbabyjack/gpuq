# Working Notes — gpuq

Last updated: **2026-05-04**

## Where we are

v0.1 implementation complete on `main`. 50 tests passing, pre-commit clean.

```
src/gpuq/
  paths.py            XDG paths
  ids.py              short job/lease ids
  protocol.py         JSON-line wire protocol
  state.py            sqlite Store + JobRepo + LeaseRepo
  logs.py             per-job log dir (stdout/stderr/combined + index.jsonl)
  ollama.py           OllamaController.evict()
  lease.py            sync `gpuq.lease()` context manager
  cli.py              click verbs (submit, ps, show, logs, cancel, ollama, daemon)
  daemon/
    queue.py          priority FIFO over JobRepo
    runner.py         spawn + capture
    server.py         asyncio unix-socket server, scheduler, bus
    main.py           foreground daemon entrypoint
scripts/gpuq.service  systemd --user unit
docs/superpowers/plans/2026-05-04-gpuq.md   implementation plan
```

## Active task

(none — ready to dogfood)

## Recent decisions (and why)

- **State dir = `$XDG_STATE_HOME/gpuq/`** (default `~/.local/state/gpuq/`),
  not `~/.local/share/`. Mutable runtime state belongs in XDG_STATE_HOME.
  Spec said "share"; we diverged on purpose.
- **`JobRepo.find(...)` not `.list(...)`** — shadowing builtin confuses ty.
- **`logs.eof()` is signalled by the server, not the runner**: server can
  then await the bus mirror task before publishing the terminal state event,
  preserving end-to-end log/state ordering.
- **Scheduler waits on lease release via 20 ms poll** — simple, bounded.
- **Ollama restore is a no-op**: relies on `OLLAMA_KEEP_ALIVE` lazy-load.

## Open threads

- `attach from_=cursor` is in the protocol but only tail/start are implemented.
- No starvation guard for low-priority jobs (single-user load).
- `lease --duration` is wired but we never expire.
- `gpu`-marked test for a real CUDA op is deferred (CI has no GPU).

## Don't lose

- Pre-commit blocks suppression markers — never silence; fix the cause.
- Vulture flags pytest fixtures consumed by name. Use
  `@pytest.mark.usefixtures("…")` instead of an unused parameter.
