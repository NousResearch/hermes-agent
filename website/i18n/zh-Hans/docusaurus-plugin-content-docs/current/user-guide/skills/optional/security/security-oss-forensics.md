---
title: "Oss Forensics — GitHub 仓库的供应链调查、证据恢复和取证分析"
sidebar_label: "Oss Forensics"
description: "GitHub 仓库的供应链调查、证据恢复和取证分析"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Oss Forensics

GitHub 仓库的供应链调查、证据恢复和取证分析。
涵盖删除提交恢复、强推检测、IOC 提取、多源证据收集、假设形成/验证和结构化取证报告。
灵感来自 RAPTOR 的 1800+ 行 OSS 取证系统。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/security/oss-forensics` 安装 |
| 路径 | `optional-skills/security/oss-forensics` |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# OSS 安全取证技能

用于研究开源供应链攻击的 7 阶段多智能体调查框架。
改编自 RAPTOR 的取证系统。涵盖 GitHub Archive、Wayback Machine、GitHub API、本地 git 分析、IOC 提取、证据支持的假设形成和验证，以及最终取证报告生成。

---

## ⚠️ 反幻觉防护栏

在每个调查步骤之前阅读。违反这些会使报告无效。

1. **证据优先规则**：任何报告、假设或摘要中的每个声明必须引用至少一个证据 ID（`EV-XXXX`）。禁止没有引用的声明。
2. **保持专注**：每个子智能体（调查员）只有一个数据源。不要混合来源。GH Archive 调查员不查询 GitHub API，反之亦然。角色边界是硬性的。
3. **事实与假设分离**：用 `[HYPOTHESIS]` 标记所有未验证的推论。只有针对原始来源验证的陈述才能作为事实陈述。
4. **不制造证据**：假设验证者必须机械检查每个引用的证据 ID 实际上存在于证据存储中，然后才能接受假设。
5. **需要证据才能反驳**：假设不能在没有具体、证据支持的反驳论点的情况下被Dismiss。"未找到证据"不足以反驳 — 它只会使假设不确定。
6. **SHA/URL 双重验证**：作为证据引用的任何提交 SHA、URL 或外部标识符必须在被标记为已验证之前从至少两个来源独立确认。
7. **可疑代码规则**：永远不要在本地运行调查仓库中找到的代码。仅静态分析，或在沙箱环境中使用 `execute_code`。
8. **密钥编辑**：在最终报告中必须编辑发现的任何 API 密钥、令牌或凭证。仅在内部记录它们。

---

## 示例场景

- **场景 A：依赖混淆**：恶意包 `internal-lib-v2` 以高于内部版本号的版本上传到 NPM。调查员必须追踪此包首次出现的时间，以及目标仓库中是否有任何 PushEvents 将 `package.json` 更新到此版本。
- **场景 B：维护者接管**：长期贡献者的账户被用来推送带有后门的 `.github/workflows/build.yml`。调查员查找此用户在长时间不活动后或来自新 IP/位置（如果可通过 BigQuery 检测）的 PushEvents。
- **场景 C：强推隐藏**：开发者意外提交了生产密钥，然后强推来"修复"。调查员使用 `git fsck` 和 GH Archive 恢复原始提交 SHA 并验证泄露了什么。

---

> **路径约定**：在整个技能中，`SKILL_DIR` 指的是此技能安装目录的根（包含此 `SKILL.md` 的文件夹）。当技能加载时，将 `SKILL_DIR` 解析为实际路径 — 例如 `~/.hermes/skills/security/oss-forensics/` 或 `optional-skills/` 等效路径。所有脚本和模板引用都相对于它。

## 阶段 0：初始化

1. 创建调查工作目录：
   ```bash
   mkdir investigation_$(echo "REPO_NAME" | tr '/' '_')
   cd investigation_$(echo "REPO_NAME" | tr '/' '_')
   ```
2. 初始化证据存储：
   ```bash
   python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list
   ```
3. 复制取证报告模板：
   ```bash
   cp SKILL_DIR/templates/forensic-report.md ./investigation-report.md
   ```
4. 创建 `iocs.md` 文件以跟踪发现的妥协指标。
5. 记录调查开始时间、目标仓库和陈述的调查目标。

---

## 阶段 1：提示解析和 IOC 提取

**目标**：从用户请求中提取所有结构化调查目标。

**操作**：
- 解析用户提示并提取：
  - 目标仓库（`owner/repo`）
  - 目标参与者（GitHub 句柄、邮箱地址）
  - 感兴趣的时间窗口（提交日期范围、PR 时间戳）
  - 提供的妥协指标：提交 SHA、文件路径、包名、IP 地址、域名、API 密钥/令牌、恶意 URL
  - 任何链接的供应商安全报告或博客文章

**工具**：仅推理，或 `execute_code` 用于从大文本块进行正则提取。

**输出**：用提取的 IOC 填充 `iocs.md`。每个 IOC 必须有：
- 类型（来自：COMMIT_SHA、FILE_PATH、API_KEY、SECRET、IP_ADDRESS、DOMAIN、PACKAGE_NAME、ACTOR_USERNAME、MALICIOUS_URL、OTHER）
- 值
- 来源（用户提供、推断）

**参考**：参见 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) 了解 IOC 分类法。

---

## 阶段 2：并行证据收集

使用 `delegate_task`（批量模式，最多 3 个并发）生成最多 5 个专业调查员子智能体。每个调查员有**一个数据源**，不得混合来源。

> **编排者注意**：在每个委托任务的 `context` 字段中传递阶段 1 的 IOC 列表和调查时间窗口。

---

### 调查员 1：本地 Git 调查员

**角色边界**：您仅查询本地 GIT 仓库。不要调用任何外部 API。

**操作**：
```bash
# 克隆仓库
git clone https://github.com/OWNER/REPO.git target_repo && cd target_repo

# 带统计的完整提交日志
git log --all --full-history --stat --format="%H|%ae|%an|%ai|%s" > ../git_log.txt

# 检测强推证据（孤立/悬空提交）
git fsck --lost-found --unreachable 2>&1 | grep commit > ../dangling_commits.txt

# 检查重写历史的 reflog
git reflog --all > ../reflog.txt

# 列出所有分支包括已删除的远程引用
git branch -a -v > ../branches.txt

# 查找可疑的大型二进制文件添加
git log --all --diff-filter=A --name-only --format="%H %ai" -- "*.so" "*.dll" "*.exe" "*.bin" > ../binary_additions.txt

# 检查 GPG 签名异常
git log --show-signature --format="%H %ai %aN" > ../signature_check.txt 2>&1
```

**要收集的证据**（通过 `python3 SKILL_DIR/scripts/evidence-store.py add` 添加）：
- 每个悬空提交 SHA → 类型：`git`
- 强推证据（显示历史重写的 reflog）→ 类型：`git`
- 来自已验证贡献者的未签名提交 → 类型：`git`
- 可疑二进制文件添加 → 类型：`git`

**参考**：参见 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) 了解访问强推提交的方法。

---

### 调查员 2：GitHub API 调查员

**角色边界**：您仅查询 GITHUB REST API。不要在本地运行 git 命令。

**操作**：
```bash
# 提交（分页）
curl -s "https://api.github.com/repos/OWNER/REPO/commits?per_page=100" > api_commits.json

# Pull Requests 包括已关闭/已删除
curl -s "https://api.github.com/repos/OWNER/REPO/pulls?state=all&per_page=100" > api_prs.json

# Issues
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=all&per_page=100" > api_issues.json

# 贡献者和协作者变更
curl -s "https://api.github.com/repos/OWNER/REPO/contributors" > api_contributors.json

# 仓库事件（最近 300 个）
curl -s "https://api.github.com/repos/OWNER/REPO/events?per_page=100" > api_events.json

# 检查特定可疑提交 SHA 详情
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" > commit_detail.json

# Releases
curl -s "https://api.github.com/repos/OWNER/REPO/releases?per_page=100" > api_releases.json

# 检查特定提交是否存在（强推提交可能在 commits/ 上 404 但在 git/commits/ 上成功）
curl -s "https://api.github.com/repos/OWNER/REPO/commits/SHA" | jq .sha
```

**交叉引用目标**（将差异标记为证据）：
- PR 在存档中存在但 API 中缺失 → 删除的证据
- 贡献者在存档事件中但不在贡献者列表中 → 权限撤销的证据
- 提交在存档 PushEvents 中但不在 API 提交列表中 → 强推/删除的证据

**参考**：参见 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) 了解 GH 事件类型。

---

### 调查员 3：Wayback Machine 调查员

**角色边界**：您仅查询 WAYBACK MACHINE CDX API。不要使用 GitHub API。

**目标**：恢复已删除的 GitHub 页面（README、issues、PR、releases、wiki 页面）。

**操作**：
```bash
# 搜索仓库主页的存档快照
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO&output=json&limit=100&from=YYYYMMDD&to=YYYYMMDD" > wayback_main.json

# 搜索特定已删除 issue
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUM&output=json&limit=50" > wayback_issue_NUM.json

# 搜索特定已删除 PR
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/pull/NUM&output=json&limit=50" > wayback_pr_NUM.json

# 获取页面的最佳快照
# 使用 Wayback Machine URL：https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
# 示例：https://web.archive.org/web/20240101000000*/github.com/OWNER/REPO

# 高级：搜索已删除 releases/tags
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/releases/tag/*&output=json" > wayback_tags.json

# 高级：搜索历史 wiki 变更
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/wiki/*&output=json" > wayback_wiki.json
```

**要收集的证据**：
- 已删除 issues/PR 的存档快照及其内容
- 显示变更的历史 README 版本
- 存在于存档中但当前 GitHub 状态中缺失的内容证据

**参考**：参见 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) 了解 CDX API 参数。

---

### 调查员 4：GH Archive / BigQuery 调查员

**角色边界**：您仅通过 BIGQUERY 查询 GITHUB ARCHIVE。这是一个防篡改的所有公共 GitHub 事件记录。

> **先决条件**：需要具有 BigQuery 访问权限的 Google Cloud 凭据（`gcloud auth application-default login`）。如果不可用，跳过此调查员并在报告中注明。

**成本优化规则**（强制）：
1. 每个查询前始终运行 `--dry_run` 以估计成本。
2. 使用 `_TABLE_SUFFIX` 按日期范围过滤以最小化扫描数据。
3. 仅 SELECT 您需要的列。
4. 除非聚合，否则添加 LIMIT。

```bash
# 模板：安全 BigQuery 查询 OWNER/REPO 的 PushEvents
bq query --use_legacy_sql=false --dry_run "
SELECT created_at, actor.login, payload.commits, payload.before, payload.head,
       payload.size, payload.distinct_size
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'PushEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 1000
"
# 如果成本可接受，重新运行不带 --dry_run

# 检测强推：零 distinct_size PushEvents 意味着提交被强制擦除
# payload.distinct_size = 0 AND payload.size > 0 → 强推指标

# 检查已删除分支事件
bq query --use_legacy_sql=false "
SELECT created_at, actor.login, payload.ref, payload.ref_type
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'DeleteEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 200
"
```

**要收集的证据**：
- 强推事件（payload.size > 0, payload.distinct_size = 0）
- 分支/标签的 DeleteEvents
- 可疑 CI/CD 自动化的 WorkflowRunEvents
- 在 git 日志中"间隔"之前的 PushEvents（重写的证据）

**参考**：参见 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) 了解所有 12 种事件类型和查询模式。

---

### 调查员 5：IOC 丰富调查员

**角色边界**：您仅使用被动公共来源丰富阶段 1 的现有 IOC。不要执行目标仓库中的任何代码。

**操作**：
- 对于每个提交 SHA：通过直接 GitHub URL 尝试恢复（`github.com/OWNER/REPO/commit/SHA.patch`）
- 对于每个域名/IP：通过公共 WHOIS 服务的 `web_extract` 检查被动 DNS、WHOIS 记录
- 对于每个包名：检查 npm/PyPI 是否有匹配的恶意包报告
- 对于每个参与者用户名：检查 GitHub 个人资料、贡献历史、账户年龄
- 使用 3 种方法恢复强推提交（参见 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md)）

---

## 阶段 3：证据整合

所有调查员完成后：

1. 运行 `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list` 查看所有收集的证据。
2. 对于每条证据，验证 `content_sha256` 哈希与原始来源匹配。
3. 按以下方式分组证据：
   - **时间线**：按时间顺序排列所有带时间戳的证据
   - **参与者**：按 GitHub 句柄或邮箱分组
   - **IOC**：将证据链接到与其相关的 IOC
4. 识别**差异**：存在于一个来源但另一个来源中缺失的项目（关键删除指标）。
5. 将证据标记为 `[VERIFIED]`（从 2+ 独立来源确认）或 `[UNVERIFIED]`（仅单一来源）。

---

## 阶段 4：假设形成

假设必须：
- 陈述具体声明（例如"参与者 X 于 DATE 强推到 BRANCH 以擦除提交 SHA"）
- 引用至少 2 个支持它的证据 ID（`EV-XXXX`、`EV-YYYY`）
- 识别什么证据会反驳它
- 在验证前标记为 `[HYPOTHESIS]`

**常见假设模板**（参见 [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md)）：
- 维护者妥协：合法账户在接管后用于注入恶意代码
- 依赖混淆：包名占用以拦截安装
- CI/CD 注入：在构建期间运行代码的恶意工作流变更
- 打字稿：针对拼写错误者的几乎相同的包名
- 凭证泄露：令牌/密钥意外提交然后强推擦除

对于每个假设，生成 `delegate_task` 子智能体，在确认之前尝试找到反驳证据。

---

## 阶段 5：假设验证

验证者子智能体必须机械检查：

1. 对于每个假设，提取所有引用的证据 ID。
2. 验证每个 ID 存在于 `evidence.json` 中（任何 ID 缺失都是硬失败 → 假设因可能被伪造而拒绝）。
3. 验证每条 `[VERIFIED]` 证据从 2+ 来源确认。
4. 检查逻辑一致性：证据描绘的时间线是否支持假设？
5. 检查替代解释：相同的证据模式是否可能由良性原因引起？

**输出**：
- `VALIDATED`：所有证据被引用、验证、逻辑一致、没有合理的替代解释。
- `INCONCLUSIVE`：证据支持假设但存在替代解释或证据不足。
- `REJECTED`：缺失的证据 ID、被引用为事实的未验证证据、检测到的逻辑不一致。

被拒绝的假设反馈到阶段 4 进行细化（最多 3 次迭代）。

---

## 阶段 6：最终报告生成

使用 [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) 中的模板填充 `investigation-report.md`。

**强制部分**：
- 执行摘要：一段裁决（已妥协/干净/不确定），置信度
- 时间线：所有重大事件按时间顺序重建，带证据引用
- 验证的假设：每个假设及其状态和支持证据 ID
- 证据注册表：所有 `EV-XXXX` 条目的表格，包含来源、类型和验证状态
- IOC 列表：所有提取和丰富的妥协指标
- 监管链：证据如何收集、从什么来源、在什么时间戳
- 建议：如果检测到妥协，立即缓解；监控建议

**报告规则**：
- 每个事实声明必须有至少一个 `[EV-XXXX]` 引用
- 执行摘要必须声明置信度（高/中/低）
- 所有密钥/凭证必须编辑为 `[REDACTED]`

---

## 阶段 7：完成

1. 运行最终证据计数：`python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`
2. 归档完整的调查目录。
3. 如果确认妥协：
   - 列出立即缓解措施（轮换凭证、固定依赖哈希、通知受影响的用户）
   - 识别受影响的版本/包
   - 注意披露义务（如果是公共包：与包注册机构协调）
4. 向用户呈现最终的 `investigation-report.md`。

---

## 道德使用指南

此技能专为**防御性安全调查**设计 — 保护开源软件免受供应链攻击。不得用于：

- **骚扰或跟踪**贡献者或维护者
- **人肉搜索** — 为恶意目的将 GitHub 活动与真实身份关联
- **竞争情报** — 未经授权调查专有或内部仓库
- **虚假指控** — 在没有验证证据的情况下发布调查结果（参见反幻觉防护栏）

调查应以**最小侵入**原则进行：仅收集验证或反驳假设所必需的证据。发布结果时，遵循负责任的披露实践，并在公开披露之前与受影响的维护者协调。

如果调查揭示了真正的妥协，遵循协调的漏洞披露流程：
1. 首先私下通知仓库维护者
2. 允许合理的修复时间（通常 90 天）
3. 如果发布的包受影响，与包注册机构（npm、PyPI 等）协调
4. 如果适当，提交 CVE

---

## API 速率限制

GitHub REST API 强制执行速率限制，如果不管理会中断大型调查。

**认证请求**：5,000/小时（需要 `GITHUB_TOKEN` 环境变量或 `gh` CLI 认证）
**未认证请求**：60/小时（调查不可用）

**最佳实践**：
- 始终认证：`export GITHUB_TOKEN=ghp_...` 或使用 `gh` CLI（自动认证）
- 使用条件请求（`If-None-Match` / `If-Modified-Since` 头）避免对未更改数据消耗配额
- 对于分页端点，按顺序获取所有页面 — 不要针对同一端点并行化
- 检查 `X-RateLimit-Remaining` 头；如果低于 100，在 `X-RateLimit-Reset` 时间戳处暂停
- BigQuery 有自己的配额（免费套餐 10 TiB/天）— 始终先试运行
- Wayback Machine CDX API：没有正式速率限制，但请礼貌（最多 1-2 请求/秒）

如果在调查中途遇到速率限制，将部分结果记录到证据存储中并在报告中注明限制。

---

## 参考材料

- [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) — BigQuery 查询、CDX API、12 种事件类型
- [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) — IOC 分类法、证据来源类型、观察类型
- [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) — 恢复已删除的提交、PR、issues
- [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) — 每种攻击类型的预建假设模板
- [evidence-store.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/scripts/evidence-store.py) — 管理证据 JSON 存储的 CLI 工具
- [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) — 结构化报告模板
