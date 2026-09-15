---
title: "Oss Forensics — GitHub 仓库的供应链调查、证据恢复与取证分析"
sidebar_label: "Oss Forensics"
description: "GitHub 仓库的供应链调查、证据恢复与取证分析"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Oss Forensics

GitHub 仓库的供应链调查、证据恢复与取证分析。
涵盖已删除提交的恢复、强制推送检测、IOC 提取、多源证据收集、
假设形成与验证，以及结构化取证报告生成。
灵感来源于 RAPTOR 的 1800+ 行 OSS Forensics 系统。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/security/oss-forensics` 安装 |
| 路径 | `optional-skills/security/oss-forensics` |
| 平台 | linux, macos, windows |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# OSS 安全取证 Skill

一个用于研究开源供应链攻击的 7 阶段多 agent 调查框架。
改编自 RAPTOR 的取证系统。涵盖 GitHub Archive、Wayback Machine、GitHub API、
本地 git 分析、IOC 提取、基于证据的假设形成与验证，以及最终取证报告生成。

---

## ⚠️ 反幻觉（Anti-Hallucination）防护规则

在每个调查步骤前必须阅读这些规则。违反这些规则将使报告失效。

1. **证据优先原则**：任何报告、假设或摘要中的每一项声明都必须引用至少一个证据 ID（`EV-XXXX`）。禁止无引用的断言。
2. **职责边界**：每个子 agent（调查员）只有一个数据源，不得混用。GH Archive 调查员不查询 GitHub API，反之亦然。职责边界是硬性规定。
3. **事实与假设分离**：所有未经验证的推断必须标注 `[HYPOTHESIS]`。只有经原始来源验证的陈述才可作为事实表述。
4. **禁止捏造证据**：假设验证器必须机械地检查每个被引用的证据 ID 在证据库中确实存在，然后才能接受假设。
5. **反驳需有证据**：驳斥一个假设必须提供具体的、有证据支撑的反驳论点。"未找到证据"不足以推翻假设——这只能使假设变为不确定状态。
6. **SHA/URL 双重验证**：任何作为证据引用的提交 SHA、URL 或外部标识符，必须在被标记为已验证之前从至少两个来源独立确认。
7. **可疑代码规则**：绝不在本地运行被调查仓库中发现的代码。仅进行静态分析，或在沙箱环境中使用 `execute_code`。
8. **密钥脱敏**：调查过程中发现的任何 API 密钥、token 或凭据必须在最终报告中脱敏处理，仅在内部日志中记录。

---

## 示例场景

- **场景 A：依赖混淆**：恶意包 `internal-lib-v2` 以更高版本号上传至 NPM，高于内部版本。调查员需追踪该包首次出现的时间，以及目标仓库中是否有 PushEvent 将 `package.json` 更新为该版本。
- **场景 B：维护者账户接管**：一名长期贡献者的账户被用于推送带有后门的 `.github/workflows/build.yml`。调查员在该用户长期不活跃或来自新 IP/位置（如可通过 BigQuery 检测）之后，查找其 PushEvent。
- **场景 C：强制推送隐藏**：开发者意外提交了生产环境密钥，随后强制推送以"修复"。调查员使用 `git fsck` 和 GH Archive 恢复原始提交 SHA，并验证泄露内容。

---

> **路径约定**：在本 skill 中，`SKILL_DIR` 指本 skill 安装目录的根目录（包含此 `SKILL.md` 的文件夹）。加载 skill 时，请将 `SKILL_DIR` 解析为实际路径——例如 `~/.hermes/skills/security/oss-forensics/` 或对应的 `optional-skills/` 路径。所有脚本和模板引用均相对于该目录。

## 阶段 0：初始化

1. 创建调查工作目录：
   ```bash
   mkdir investigation_$(echo "REPO_NAME" | tr '/' '_')
   cd investigation_$(echo "REPO_NAME" | tr '/' '_')
   ```
2. 初始化证据库：
   ```bash
   python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list
   ```
3. 复制取证报告模板：
   ```bash
   cp SKILL_DIR/templates/forensic-report.md ./investigation-report.md
   ```
4. 创建 `iocs.md` 文件，用于追踪发现的入侵指标（Indicators of Compromise，IOC）。
5. 记录调查开始时间、目标仓库及调查目标说明。

---

## 阶段 1：Prompt 解析与 IOC 提取

**目标**：从用户请求中提取所有结构化调查目标。

**操作**：
- 解析用户 prompt（提示词），提取：
  - 目标仓库（`owner/repo`）
  - 目标参与者（GitHub 用户名、电子邮件地址）
  - 关注的时间窗口（提交日期范围、PR 时间戳）
  - 提供的入侵指标：提交 SHA、文件路径、包名、IP 地址、域名、API 密钥/token、恶意 URL
  - 任何关联的供应商安全报告或博客文章

**工具**：仅推理，或对大段文本使用 `execute_code` 进行正则提取。

**输出**：将提取的 IOC 填入 `iocs.md`。每个 IOC 必须包含：
- 类型（从以下选择：COMMIT_SHA、FILE_PATH、API_KEY、SECRET、IP_ADDRESS、DOMAIN、PACKAGE_NAME、ACTOR_USERNAME、MALICIOUS_URL、OTHER）
- 值
- 来源（用户提供、推断得出）

**参考**：IOC 分类法见 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)。

---

## 阶段 2：并行证据收集

使用 `delegate_task`（批量模式，最多 3 个并发）派生最多 5 个专业调查员子 agent。每个调查员只有**一个数据源**，不得混用。

> **编排器注意**：在每个委托任务的 `context` 字段中传入阶段 1 的 IOC 列表和调查时间窗口。

---

### 调查员 1：本地 Git 调查员

**职责边界**：仅查询**本地 Git 仓库**，不调用任何外部 API。

**操作**：
```bash
# 克隆仓库
git clone https://github.com/OWNER/REPO.git target_repo && cd target_repo

# 完整提交日志（含统计信息）
git log --all --full-history --stat --format="%H|%ae|%an|%ai|%s" > ../git_log.txt

# 检测强制推送证据（孤立/悬空提交）
git fsck --lost-found --unreachable 2>&1 | grep commit > ../dangling_commits.txt

# 检查 reflog 中的历史重写
git reflog --all > ../reflog.txt

# 列出所有分支，包括已删除的远程引用
git branch -a -v > ../branches.txt

# 查找可疑的大型二进制文件添加
git log --all --diff-filter=A --name-only --format="%H %ai" -- "*.so" "*.dll" "*.exe" "*.bin" > ../binary_additions.txt

# 检查 GPG 签名异常
git log --show-signature --format="%H %ai %aN" > ../signature_check.txt 2>&1
```

**需收集的证据**（通过 `python3 SKILL_DIR/scripts/evidence-store.py add` 添加）：
- 每个悬空提交 SHA → 类型：`git`
- 强制推送证据（reflog 显示历史重写）→ 类型：`git`
- 已验证贡献者的未签名提交 → 类型：`git`
- 可疑二进制文件添加 → 类型：`git`

**参考**：访问强制推送提交的方法见 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md)。

---

### 调查员 2：GitHub API 调查员

**职责边界**：仅查询 **GitHub REST API**，不在本地运行 git 命令。

**操作**：
```bash
# 提交（分页）
curl -s "https://api.github.com/repos/OWNER/REPO/commits?per_page=100" > api_commits.json

# Pull Request（含已关闭/已删除）
curl -s "https://api.github.com/repos/OWNER/REPO/pulls?state=all&per_page=100" > api_prs.json

# Issues
curl -s "https://api.github.com/repos/OWNER/REPO/issues?state=all&per_page=100" > api_issues.json

# 贡献者及协作者变更
curl -s "https://api.github.com/repos/OWNER/REPO/contributors" > api_contributors.json

# 仓库事件（最近 300 条）
curl -s "https://api.github.com/repos/OWNER/REPO/events?per_page=100" > api_events.json

# 查看特定可疑提交 SHA 的详情
curl -s "https://api.github.com/repos/OWNER/REPO/git/commits/SHA" > commit_detail.json

# Releases
curl -s "https://api.github.com/repos/OWNER/REPO/releases?per_page=100" > api_releases.json

# 检查特定提交是否存在（强制推送的提交在 commits/ 可能返回 404，但在 git/commits/ 可能成功）
curl -s "https://api.github.com/repos/OWNER/REPO/commits/SHA" | jq .sha
```

**交叉比对目标**（将差异标记为证据）：
- PR 存在于归档中但 API 中缺失 → 删除证据
- 贡献者出现在归档事件中但不在贡献者列表中 → 权限撤销证据
- 提交出现在归档 PushEvent 中但不在 API 提交列表中 → 强制推送/删除证据

**参考**：GH 事件类型见 [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md)。

---

### 调查员 3：Wayback Machine 调查员

**职责边界**：仅查询 **Wayback Machine CDX API**，不使用 GitHub API。

**目标**：恢复已删除的 GitHub 页面（README、issues、PR、releases、wiki 页面）。

**操作**：
```bash
# 搜索仓库主页的归档快照
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO&output=json&limit=100&from=YYYYMMDD&to=YYYYMMDD" > wayback_main.json

# 搜索特定已删除 issue
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/issues/NUM&output=json&limit=50" > wayback_issue_NUM.json

# 搜索特定已删除 PR
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/pull/NUM&output=json&limit=50" > wayback_pr_NUM.json

# 获取页面的最佳快照
# 使用 Wayback Machine URL：https://web.archive.org/web/TIMESTAMP/ORIGINAL_URL
# 示例：https://web.archive.org/web/20240101000000*/github.com/OWNER/REPO

# 高级：搜索已删除的 releases/tags
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/releases/tag/*&output=json" > wayback_tags.json

# 高级：搜索历史 wiki 变更
curl -s "https://web.archive.org/cdx/search/cdx?url=github.com/OWNER/REPO/wiki/*&output=json" > wayback_wiki.json
```

**需收集的证据**：
- 已删除 issue/PR 的归档快照及其内容
- 显示变更的历史 README 版本
- 存在于归档中但在当前 GitHub 状态中缺失的内容证据

**参考**：CDX API 参数见 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)。

---

### 调查员 4：GH Archive / BigQuery 调查员

**职责边界**：仅通过 **BigQuery** 查询 **GitHub Archive**。这是所有公开 GitHub 事件的防篡改记录。

> **前提条件**：需要具有 BigQuery 访问权限的 Google Cloud 凭据（`gcloud auth application-default login`）。如不可用，跳过此调查员并在报告中注明。

**成本优化规则**（强制执行）：
1. 每次查询前必须先运行 `--dry_run` 以估算成本。
2. 使用 `_TABLE_SUFFIX` 按日期范围过滤，最小化扫描数据量。
3. 只 SELECT 所需列。
4. 除非进行聚合，否则添加 LIMIT。

```bash
# 模板：安全的 BigQuery 查询，用于查询 OWNER/REPO 的 PushEvent
bq query --use_legacy_sql=false --dry_run "
SELECT created_at, actor.login, payload.commits, payload.before, payload.head,
       payload.size, payload.distinct_size
FROM \`githubarchive.month.*\`
WHERE _TABLE_SUFFIX BETWEEN 'YYYYMM' AND 'YYYYMM'
  AND type = 'PushEvent'
  AND repo.name = 'OWNER/REPO'
LIMIT 1000
"
# 如果成本可接受，去掉 --dry_run 重新运行

# 检测强制推送：distinct_size 为零的 PushEvent 表示提交被强制擦除
# payload.distinct_size = 0 AND payload.size > 0 → 强制推送指标

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

**需收集的证据**：
- 强制推送事件（payload.size > 0，payload.distinct_size = 0）
- 分支/标签的 DeleteEvent
- 可疑 CI/CD 自动化的 WorkflowRunEvent
- 在 git 日志出现"空白"之前的 PushEvent（历史重写证据）

**参考**：所有 12 种事件类型及查询模式见 [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md)。

---

### 调查员 5：IOC 富化调查员

**职责边界**：仅使用**被动公开来源**对阶段 1 中的**现有 IOC** 进行富化。不执行目标仓库中的任何代码。

**操作**：
- 对每个提交 SHA：尝试通过直接 GitHub URL（`github.com/OWNER/REPO/commit/SHA.patch`）恢复
- 对每个域名/IP：检查被动 DNS、WHOIS 记录（通过 `web_extract` 访问公开 WHOIS 服务）
- 对每个包名：检查 npm/PyPI 中是否有匹配的恶意包报告
- 对每个 actor 用户名：检查 GitHub 个人资料、贡献历史、账户注册时间
- 使用 3 种方法恢复强制推送的提交（见 [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md)）

---

## 阶段 3：证据整合

所有调查员完成后：

1. 运行 `python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list` 查看所有已收集证据。
2. 对每条证据，验证 `content_sha256` 哈希值与原始来源一致。
3. 按以下维度对证据分组：
   - **时间线**：将所有带时间戳的证据按时间顺序排列
   - **参与者**：按 GitHub 用户名或电子邮件分组
   - **IOC**：将证据与其关联的 IOC 链接
4. 识别**差异**：存在于一个来源但在另一个来源中缺失的条目（关键删除指标）。
5. 将证据标记为 `[VERIFIED]`（已从 2 个以上独立来源确认）或 `[UNVERIFIED]`（仅单一来源）。

---

## 阶段 4：假设形成

一个假设必须：
- 陈述具体声明（例如："参与者 X 于某日期对 BRANCH 进行强制推送以擦除提交 SHA"）
- 引用至少 2 个支持它的证据 ID（`EV-XXXX`、`EV-YYYY`）
- 指明哪些证据可以推翻它
- 在验证之前标注 `[HYPOTHESIS]`

**常见假设模板**（见 [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md)）：
- 维护者账户被攻陷：合法账户在被接管后用于注入恶意代码
- 依赖混淆：包名抢注以拦截安装
- CI/CD 注入：恶意 workflow 变更以在构建期间运行代码
- 仿冒命名（Typosquatting）：针对拼写错误者的高度相似包名
- 凭据泄露：token/密钥意外提交后强制推送以擦除

对每个假设，派生一个 `delegate_task` 子 agent，在确认之前尝试寻找反驳证据。

---

## 阶段 5：假设验证

验证器子 agent 必须机械地检查：

1. 对每个假设，提取所有被引用的证据 ID。
2. 验证每个 ID 在 `evidence.json` 中存在（如有任何 ID 缺失则硬性失败 → 假设因可能捏造而被拒绝）。
3. 验证每条 `[VERIFIED]` 证据已从 2 个以上来源确认。
4. 检查逻辑一致性：证据所描绘的时间线是否支持该假设？
5. 检查替代解释：相同的证据模式是否可能源于良性原因？

**输出**：
- `VALIDATED`：所有证据已引用、已验证、逻辑一致，且不存在合理的替代解释。
- `INCONCLUSIVE`：证据支持假设，但存在替代解释或证据不足。
- `REJECTED`：证据 ID 缺失、将未验证证据作为事实引用、检测到逻辑不一致。

被拒绝的假设反馈至阶段 4 进行修正（最多 3 次迭代）。

---

## 阶段 6：最终报告生成

使用 [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) 中的模板填写 `investigation-report.md`。

**必填章节**：
- 执行摘要：一段式结论（已被攻陷 / 干净 / 不确定），含置信度等级
- 时间线：所有重要事件的时间顺序重建，含证据引用
- 已验证假设：每条假设含状态及支持证据 ID
- 证据注册表：所有 `EV-XXXX` 条目的表格，含来源、类型和验证状态
- IOC 列表：所有提取和富化的入侵指标
- 证据保管链：证据的收集方式、来源及收集时间戳
- 建议：如检测到攻陷，提供即时缓解措施；以及监控建议

**报告规则**：
- 每项事实声明必须至少有一个 `[EV-XXXX]` 引用
- 执行摘要必须说明置信度等级（高 / 中 / 低）
- 所有密钥/凭据必须脱敏为 `[REDACTED]`

---

## 阶段 7：完成

1. 运行最终证据统计：`python3 SKILL_DIR/scripts/evidence-store.py --store evidence.json list`
2. 归档完整调查目录。
3. 如确认存在攻陷：
   - 列出即时缓解措施（轮换凭据、固定依赖哈希、通知受影响用户）
   - 识别受影响的版本/包
   - 注明披露义务（如为公开包：与包注册表协调）
4. 向用户呈现最终 `investigation-report.md`。

---

## 道德使用准则

本 skill 专为**防御性安全调查**而设计——保护开源软件免受供应链攻击。不得用于：

- **骚扰或跟踪**贡献者或维护者
- **人肉搜索（Doxing）**——将 GitHub 活动与真实身份关联用于恶意目的
- **竞争情报**——未经授权调查专有或内部仓库
- **虚假指控**——在没有经过验证的证据的情况下发布调查结果（参见反幻觉防护规则）

调查应遵循**最小侵入原则**：仅收集验证或反驳假设所必需的证据。发布结果时，遵循负责任披露实践，在公开披露前与受影响的维护者协调。

如果调查揭示了真实的攻陷，请遵循协调漏洞披露流程：
1. 首先私下通知仓库维护者
2. 给予合理的修复时间（通常为 90 天）
3. 如涉及已发布包，与包注册表（npm、PyPI 等）协调
4. 如适用，提交 CVE

---

## API 速率限制

GitHub REST API 强制执行速率限制，如不加以管理，将中断大型调查。

**已认证请求**：5,000 次/小时（需要 `GITHUB_TOKEN` 环境变量或 `gh` CLI 认证）
**未认证请求**：60 次/小时（不适用于调查）

**最佳实践**：
- 始终进行认证：`export GITHUB_TOKEN=ghp_...` 或使用 `gh` CLI（自动认证）
- 使用条件请求（`If-None-Match` / `If-Modified-Since` 请求头），避免对未变更数据消耗配额
- 对分页端点，按顺序获取所有页面——不要对同一端点并行请求
- 检查 `X-RateLimit-Remaining` 响应头；如低于 100，暂停至 `X-RateLimit-Reset` 时间戳
- BigQuery 有其自身配额（免费层每日 10 TiB）——始终先进行 dry-run
- Wayback Machine CDX API：无正式速率限制，但请保持礼貌（最多 1-2 次请求/秒）

如在调查中途遭遇速率限制，将部分结果记录到证据库中，并在报告中注明该限制。

---

## 参考资料

- [github-archive-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/github-archive-guide.md) — BigQuery 查询、CDX API、12 种事件类型
- [evidence-types.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/evidence-types.md) — IOC 分类法、证据来源类型、观察类型
- [recovery-techniques.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/recovery-techniques.md) — 恢复已删除的提交、PR、issues
- [investigation-templates.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/references/investigation-templates.md) — 按攻击类型预置的假设模板
- [evidence-store.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/scripts/evidence-store.py) — 用于管理证据 JSON 库的 CLI 工具
- [forensic-report.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/security/oss-forensics/templates/forensic-report.md) — 结构化报告模板