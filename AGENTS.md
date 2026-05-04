# AGENTS.md

Project-scoped rules. Universal rules (commits, no-suppression, TDD, verification loop, persistent state, anti-patterns) live in `~/.claude/CLAUDE.md` — read those first.

## Tooling — Astral stack

```bash
uv add <pkg>           # runtime dep
uv add --dev <pkg>     # dev dep
uv sync                # install/refresh
uv run <cmd>           # run anything
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check        # types
```

## Pre-commit

`uv run pre-commit install` once after `uv sync`. Hooks: ruff (check + format), ty, vulture, pygrep blocking suppression markers (`# noqa`, `# type: ignore`, `# pyright: ignore`, `# fmt: skip|off|on`), hygiene. No `--no-verify`. Ever.

If vulture flags an entry used dynamically (f-string field refs, `getattr` lookups, dataclass fields read via `config.<field>`), regenerate `vulture_whitelist.py` as a `refactor:` prep commit and continue — don't pause to ask. Procedure: `uv run vulture --make-whitelist src/ tests/ > vulture_whitelist.py`.

## Commits

Always via `scripts/committer "type(scope): subject" path1 path2`. Conventional Commits enforced; bypass flags refused; soft-warn on diffs > 400 lines. Symlink to `~/code/agentic-research/scripts/committer`.

## pytest markers

`--strict-markers` is on. Standard: `slow`, `integration`, `gpu`. Declare new markers in `pyproject.toml` before using.

## Project-specific anti-patterns

Living catalog. Append as patterns emerge.

(none yet)
