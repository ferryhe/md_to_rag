# md_to_rag 独立 CLI 与 RAG Artifact v1 Implementation Plan

> **For Hermes/Codex:** Use the project-isolated Codex worker pattern. Read `AGENTS.md` and `.hermes/project-status.md` before each run. Do not commit, push, or open PRs without explicit approval.

**Goal:** 建立领域无关的 Markdown corpus 到可检索知识库 artifact 的 CLI。

**Architecture:** CLI-first、JSONL-first、manifest-first；核心分 source/documents/chunks/embeddings/index/query 六层；第一版实现 native + FAISS，CocoIndex 只作为设计参考/未来 backend。

**Tech Stack:** Project-native stack plus CLI-first JSON/JSONL manifests. Python projects should use Typer/Pydantic where already present; TypeScript projects should preserve pnpm/OpenAPI workflow.

---

## Context

This repository is one module in the broader agent-operated knowledge pipeline:

```text
web_listening -> doc_to_md -> md_to_rag -> rag_to_agent/domain adapters -> ai_interface
```

Current project role: Markdown -> RAG artifact CLI，负责 ingest/chunk/embed/index/query/inspect。

Current planning scope: 新建独立 md_to_rag，先 native engine + FAISS，保留 CocoIndex-style 增量与 backend 扩展点。

## Non-Negotiable Contracts

1. CLI outputs must be machine-readable and stable (`--json` where applicable).
2. Artifacts must be path-portable and manifest-driven.
3. Reruns must be idempotent.
4. Every derived artifact must preserve provenance back to its input.
5. Secrets/API keys must never be written into manifests or committed files.
6. Cross-repo integration happens through files/manifests/tool specs, not hidden imports.

## Proposed Tasks

### Task 1: 初始化 Python 包骨架

**Objective:** 创建 pyproject.toml、src/md_to_rag/cli.py、schemas.py、manifest.py、tests。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** `md-to-rag --help` 可运行，pytest 通过。

### Task 2: 实现 init/inspect

**Objective:** `md-to-rag init PROJECT` 创建标准目录；`inspect --json` 读取 manifest/status。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试重复 init 幂等。

### Task 3: 实现 ingest

**Objective:** 读取 Markdown 文件夹或 doc_to_md manifest，生成 source/documents.jsonl 与 corpus_manifest.json。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试 doc_id/hash/path 稳定。

### Task 4: 实现 chunk

**Objective:** 移植/简化 AI_actuarial semantic chunking，输出 chunks/chunks.jsonl、chunk_profile.json。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试 heading_path、start/end line、content_hash。

### Task 5: 实现 embed cache

**Objective:** provider 抽象先支持 local dummy 和 OpenAI compatible；embedding_profile 入 manifest，不保存 secrets。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试相同 chunk 复用 cache，profile 改变触发重算。

### Task 6: 实现 FAISS index/query

**Objective:** 保存 indexes/faiss/index.faiss + metadata.jsonl + index_manifest.json；query 返回 citations。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试 top-k、threshold、missing index error。

### Task 7: 实现 diff/rebuild

**Objective:** 按 source/chunk/embedding/index hash 做增量判断。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** 测试改一个 md 只重算对应 doc/chunks。

### Task 8: 预留 CocoIndex adapter ADR

**Objective:** 新增 docs/adr/0001-native-vs-cocoindex-engine.md，定义未来 `--engine cocoindex` 边界。

**Files:**
- Modify/Create project-specific files identified during the task.
- Update tests or fixtures for the changed contract.

**Steps:**
1. Inspect the current implementation and write down exact files touched.
2. Add or update the smallest contract/test fixture first.
3. Implement the minimal change.
4. Run the focused verification command.
5. Update `.hermes/project-status.md` with result and next action.

**Verification:** ADR 明确不影响 v1 native contract。


---

## Acceptance Criteria

- A Codex worker can understand this repo's boundary from `AGENTS.md`.
- A future implementation branch can start from this plan without needing cross-chat context.
- The module's input/output contract is explicit enough for the next module in the chain.
- All new behavior is testable through CLI commands and fixture manifests.

## Recommended First PR

Start with documentation/contracts and fixture-only changes. Do not implement all runtime behavior in the first PR. The first PR should make the intended contract reviewable before code follows.
