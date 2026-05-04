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
