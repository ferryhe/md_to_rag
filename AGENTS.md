# Project Agent Boundary: md_to_rag

## Identity

- Project: `md_to_rag`
- Repo path: `/home/ec2-user/work/md_to_rag`
- Remote: `https://github.com/ferryhe/md_to_rag`
- Worker role: Markdown -> RAG artifact CLI，负责 ingest/chunk/embed/index/query/inspect。

## Hard Boundaries

1. Treat this repository as the only writable workspace for this project.
2. Do not read, edit, or infer requirements from sibling repositories unless the current task explicitly names them.
3. Do not copy secrets, local `.env` files, generated credentials, or unreviewed artifacts between projects.
4. After scoped implementation and required verification, including the Pre-PR Codex Review Gate below, pass, automatically commit, push, and create a PR if one does not already exist; do not pause for routine commit/push/PR approval. Destructive actions such as deleting branches, rewriting history, removing files outside scope, or force-pushing still require explicit user approval.
5. After creating a PR, perform one follow-up pass about 15 minutes later to check GitHub checks and remote review/Copilot comments. Evaluate comments on merit, automatically fix only confirmed-safe issues, rerun focused/full validation as appropriate, commit and push fixes, then report the final state.
6. Before editing, run `git status --short --branch` and identify unrelated local changes. Do not stage or commit unrelated changes; isolate them from the task or stop and report if they cannot be safely isolated.
7. Keep changes narrow and project-scoped. For cross-project contracts, edit only this repo's side unless the task explicitly covers multiple repos.

## Branch Policy

- `main` is treated as the clean baseline tracking `origin/main`.
- New work starts from latest `main` on a task branch.
- Do not implement directly on `main`.
- Commit/push automatically after implementation and required verification, including the Pre-PR Codex Review Gate below, pass, and create a PR only if the task branch does not already have one.
- About 15 minutes after PR creation, perform one follow-up pass to evaluate GitHub checks and remote review/Copilot comments, apply only confirmed-safe fixes, rerun validation, and push follow-up commits.
- Destructive actions such as force-push, branch deletion, history rewrite, or broad cleanup still require explicit approval from 北老师.

## Required Startup Routine

Every Codex worker run must:

1. Read this `AGENTS.md`.
2. Read `.hermes/project-status.md` if present.
3. Run `git status --short --branch`.
4. Restate the active repo, branch, files in scope, and whether sibling repos are off-limits.
5. Only then edit files.

## Verification Policy

- Prefer focused tests for touched code.
- Required verification means the focused/full checks identified by this Verification Policy for the files touched plus the Pre-PR Codex Review Gate below; if required checks fail or cannot run, do not treat verification as passed, and record the exact blocker and next command.
- For CLI contract changes, verify `--help`, `--json` output, schema fixtures, and idempotent reruns.
- For frontend/API work, run local service/build checks and browser smoke when UI behavior changes.
- If checks cannot run, record the exact blocker and the next command to run.

## Pre-PR Codex Review Gate

After development and local verification are complete, but before creating or updating a PR, run a separate Codex CLI review of the current branch against the PR base. Treat this as a mandatory local review gate.

Use the PR base branch as the diff base; if there is no open PR yet, use `main` unless the task explicitly names another base:

```bash
BASE_BRANCH=${BASE_BRANCH:-main}
git fetch origin "$BASE_BRANCH"
git diff --no-ext-diff "origin/$BASE_BRANCH"...HEAD > /tmp/codex-pr.diff
codex exec --sandbox read-only "Review /tmp/codex-pr.diff for technically correct, in-scope issues that should block this PR."
```

Evaluate Codex findings the same way as remote review comments: accept only technically correct, in-scope findings; make the necessary fixes; rerun the focused/full verification; then create or update the PR. If Codex CLI cannot run because of authentication or tooling, record the blocker explicitly in the final report before proceeding.

## Reporting Format

End every run with:

- Project and branch
- Files changed
- Checks run and results
- Uncommitted/untracked files noticed
- Blockers or decisions needed
- Recommended next action

Also update `.hermes/project-status.md` with the latest state.
