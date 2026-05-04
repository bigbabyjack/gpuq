# Decisions Log — gpuq

Append-only ADR-lite. Dated entries. Never edited, never reordered.

Format: `## YYYY-MM-DD — Title` then *Context*, *Decision*, *Rationale*, *Consequences*.

---

## 2026-05-04 — Project scaffolded

**Context.** New project bootstrapped via `~/code/agentic-research/scripts/new-project.py`.

**Decision.** Astral stack (uv, ruff, ty, vulture). Harness files seeded from canonical template. Committer symlinked. AGENTS.md is the canonical instruction file; `CLAUDE.md` is a symlink to it.

**Rationale.** Default cross-project setup per `~/code/agentic-research/agent-harness.md`.

**Consequences.** Standard discipline applies: hooks unbypassable, commits via committer, AGENTS.md telegraph, two-tier persistent state (this file + `WORKING_NOTES.md`).

---

## 2026-05-04 — v0.1 implementation

**Context.** Spec at `docs/superpowers/specs/2026-05-04-gpuq-design.md`
called for a single-GPU job scheduler daemon with a thin CLI, ollama
eviction, and a cooperative lease.

**Decision.** Built it in 10 commits over one session. Asyncio unix-socket
daemon, sqlite for durable job/lease state, per-job log dir with sidecar
`index.jsonl` for stream-tagged replay, click-based CLI streaming
JSON-line frames. `OllamaController.evict()` shells out to `ollama ps`
+ `ollama stop`; restore is a no-op (lazy-load via `OLLAMA_KEEP_ALIVE`).

**Rationale.** Stayed inside the Python stdlib for the runtime (no aiohttp,
no async-pg, etc.) — single-box single-user load doesn't justify it.
Each daemon module has one responsibility (queue / runner / ollama / server),
so failure modes can be reasoned about file-by-file.

**Consequences.** No multi-daemon, no GPU-aware co-scheduling, no preemption
of running jobs. `attach from_=cursor`, lease expiry, and starvation guards
are deliberate v2 hooks — wires exist in the protocol, behaviour does not.
State dir is `$XDG_STATE_HOME/gpuq/` (the spec said `share/`; state belongs
in `state/`).

---
