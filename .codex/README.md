# Codex Worker Notes for md_to_rag

This directory records the local Codex worker convention for this repo.

Follow `AGENTS.md` for every worker run. The current startup routine is:

1. Read `AGENTS.md`.
2. Read `.hermes/project-status.md` if present.
3. Run `git status --short --branch`.
4. Restate the active repo, branch, files in scope, and whether sibling repos are off-limits.
5. Read the active plan under `docs/plans/` when the task names one, then edit only in-scope files.

After scoped implementation, required verification, and the Pre-PR Codex Review Gate pass, routine commit, push, and PR creation/update are authorized by the project policy. For the active managed-PR program authorized on 2026-06-17, after checks and valid Copilot/remote comments are resolved, the controller may merge to `main` and delete that PR's remote/local task branch. Outside this scoped program, destructive actions such as branch deletion, force-push, history rewrite, broad cleanup, or deleting unrelated work still require fresh explicit approval.
