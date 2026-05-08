---
title: "HuggingFace Hub — HuggingFace hf CLI：搜索/下载/上传模型、数据集"
sidebar_label: "HuggingFace Hub"
description: "HuggingFace hf CLI：搜索/下载/上传模型、数据集"
---
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
# HuggingFace Hub
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
HuggingFace hf CLI: search/download/upload models, datasets.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
## 技能元数据
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/huggingface-hub` |
| Version | `1.0.0` |
| Author | Hugging Face |
| License | MIT |
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
## 参考：完整 SKILL.md
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
# HuggingFace Hub
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
The `hf` command is the modern command-line interface for interacting with the Hugging Face Hub, providing tools to manage repositories, models, datasets, and Spaces.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
> **IMPORTANT:** The `hf` command replaces the now deprecated `huggingface-cli` command.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
## Quick Start
*   **Installation:** `curl -LsSf https://hf.co/cli/install.sh | bash -s`
*   **Help:** Use `hf --help` to view all available functions and real-world examples.
*   **Authentication:** Recommended via `HF_TOKEN` environment variable or the `--token` flag.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
---

## Core Commands

### General Operations
*   `hf download REPO_ID`: Download files from the Hub.
*   `hf upload REPO_ID`: Upload files/folders (recommended for single-commit).
*   `hf upload-large-folder REPO_ID LOCAL_PATH`: Recommended for resumable uploads of large directories.
*   `hf sync`: Sync files between a local directory and a bucket.
*   `hf env` / `hf version`: View environment and version details.

### Authentication (`hf auth`)
*   `login` / `logout`: Manage sessions using tokens from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
*   `list` / `switch`: Manage and toggle between multiple stored access tokens.
*   `whoami`: Identify the currently logged-in account.

### Repository Management (`hf repos`)
*   `create` / `delete`: Create or permanently remove repositories.
*   `duplicate`: Clone a model, dataset, or Space to a new ID.
*   `move`: Transfer a repository between namespaces.
*   `branch` / `tag`: Manage Git-like references.
*   `delete-files`: Remove specific files using patterns.

---
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
## Specialized Hub Interactions
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
### Datasets & Models
*   **Datasets:** `hf datasets list`, `info`, and `parquet` (list parquet URLs).
*   **SQL Queries:** `hf datasets sql SQL` — Execute raw SQL via DuckDB against dataset parquet URLs.
*   **Models:** `hf models list` and `info`.
*   **Papers:** `hf papers list` — View daily papers.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
### Discussions & Pull Requests (`hf discussions`)
*   Manage the lifecycle of Hub contributions: `list`, `create`, `info`, `comment`, `close`, `reopen`, and `rename`.
*   `diff`: View changes in a PR.
*   `merge`: Finalize pull requests.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
### Infrastructure & Compute
*   **Endpoints:** Deploy and manage Inference Endpoints (`deploy`, `pause`, `resume`, `scale-to-zero`, `catalog`).
*   **Jobs:** Run compute tasks on HF infrastructure. Includes `hf jobs uv` for running Python scripts with inline dependencies and `stats` for resource monitoring.
*   **Spaces:** Manage interactive apps. Includes `dev-mode` and `hot-reload` for Python files without full restarts.
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
### Storage & Automation
*   **Buckets:** Full S3-like bucket management (`create`, `cp`, `mv`, `rm`, `sync`).
*   **Cache:** Manage local storage with `list`, `prune` (remove detached revisions), and `verify` (checksum checks).
*   **Webhooks:** Automate workflows by managing Hub webhooks (`create`, `watch`, `enable`/`disable`).
*   **Collections:** Organize Hub items into collections (`add-item`, `update`, `list`).
HuggingFace hf CLI：搜索/下载/上传模型、数据集。
---

## Advanced Usage & Tips

### Global Flags
*   `--format json`: Produces machine-readable output for automation.
*   `-q` / `--quiet`: Limits output to IDs only.

### Extensions & Skills
*   **Extensions:** Extend CLI functionality via GitHub repositories using `hf extensions install REPO_ID`.
*   **Skills:** Manage AI assistant skills with `hf skills add`.
