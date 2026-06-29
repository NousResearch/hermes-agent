---
title: "Github Code Review — 通过 gh 或 REST 审查 PR：差异对比、行内评论"
sidebar_label: "Github Code Review"
description: "通过 gh 或 REST 审查 PR：差异对比、行内评论"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Github Code Review

通过 gh 或 REST 审查 PR：差异对比、行内评论。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/github/github-code-review` |
| 版本 | `1.1.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `GitHub`, `Code-Review`, `Pull-Requests`, `Git`, `Quality` |
| 相关 skill | [`github-auth`](/user-guide/skills/bundled/github/github-github-auth), [`github-pr-workflow`](/user-guide/skills/bundled/github/github-github-pr-workflow) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# GitHub Code Review

在推送前对本地变更执行代码审查，或审查 GitHub 上的开放 PR。此 skill 大部分功能使用纯 `git` 命令——`gh`/`curl` 的区别仅在 PR 级别的交互中才有意义。

## 前置条件

- 已通过 GitHub 身份验证（参见 `github-auth` skill）
- 位于 git 仓库内部

### 设置（用于 PR 交互）

```bash
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  AUTH="gh"
else
  AUTH="git"
  if [ -z "$GITHUB_TOKEN" ]; then
    if [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
      GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
    elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
      GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
    fi
  fi
fi

REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(echo "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
OWNER=$(echo "$OWNER_REPO" | cut -d/ -f1)
REPO=$(echo "$OWNER_REPO" | cut -d/ -f2)
```

---

## 1. 审查本地变更（推送前）

此部分为纯 `git` 操作——适用于所有环境，无需 API。

### 获取差异

```bash
# 已暂存的变更（即将提交的内容）
git diff --staged

# 相对于 main 的所有变更（PR 将包含的内容）
git diff main...HEAD

# 仅显示文件名
git diff main...HEAD --name-only

# 统计摘要（每个文件的插入/删除行数）
git diff main...HEAD --stat
```

### 审查策略

1. **先了解全局：**

```bash
git diff main...HEAD --stat
git log main..HEAD --oneline
```

2. **逐文件审查**——使用 `read_file` 查看已变更文件的完整上下文，并通过差异了解具体改动：

```bash
git diff main...HEAD -- src/auth/login.py
```

3. **检查常见问题：**

```bash
# 遗留的调试语句、TODO、console.log 等
git diff main...HEAD | grep -n "print(\|console\.log\|TODO\|FIXME\|HACK\|XXX\|debugger"

# 意外暂存的大文件
git diff main...HEAD --stat | sort -t'|' -k2 -rn | head -10

# 密钥或凭据模式
git diff main...HEAD | grep -in "password\|secret\|api_key\|token.*=\|private_key"

# 合并冲突标记
git diff main...HEAD | grep -n "<<<<<<\|>>>>>>\|======="
```

4. **向用户呈现结构化反馈。**

### 审查输出格式

审查本地变更时，按以下结构呈现结果：

```
## Code Review Summary

### Critical
- **src/auth.py:45** — SQL injection: user input passed directly to query.
  Suggestion: Use parameterized queries.

### Warnings
- **src/models/user.py:23** — Password stored in plaintext. Use bcrypt or argon2.
- **src/api/routes.py:112** — No rate limiting on login endpoint.

### Suggestions
- **src/utils/helpers.py:8** — Duplicates logic in `src/core/utils.py:34`. Consolidate.
- **tests/test_auth.py** — Missing edge case: expired token test.

### Looks Good
- Clean separation of concerns in the middleware layer
- Good test coverage for the happy path
```

---

## 2. 审查 GitHub 上的 Pull Request

### 查看 PR 详情

**使用 gh：**

```bash
gh pr view 123
gh pr diff 123
gh pr diff 123 --name-only
```

**使用 git + curl：**

```bash
PR_NUMBER=123

# 获取 PR 详情
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "
import sys, json
pr = json.load(sys.stdin)
print(f\"Title: {pr['title']}\")
print(f\"Author: {pr['user']['login']}\")
print(f\"Branch: {pr['head']['ref']} -> {pr['base']['ref']}\")
print(f\"State: {pr['state']}\")
print(f\"Body:\n{pr['body']}\")"

# 列出已变更文件
curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/files \
  | python3 -c "
import sys, json
for f in json.load(sys.stdin):
    print(f\"{f['status']:10} +{f['additions']:-4} -{f['deletions']:-4}  {f['filename']}\")"
```

### 在本地检出 PR 进行完整审查

此操作使用纯 `git`——无需 `gh`：

```bash
# 获取 PR 分支并检出
git fetch origin pull/123/head:pr-123
git checkout pr-123

# 现在可以使用 read_file、search_files、运行测试等

# 查看与基础分支的差异
git diff main...pr-123
```

**使用 gh（快捷方式）：**

```bash
gh pr checkout 123
```

### 在 PR 上留下评论

**通用 PR 评论——使用 gh：**

```bash
gh pr comment 123 --body "Overall looks good, a few suggestions below."
```

**通用 PR 评论——使用 curl：**

```bash
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/issues/$PR_NUMBER/comments \
  -d '{"body": "Overall looks good, a few suggestions below."}'
```

### 留下行内审查评论

**单条行内评论——使用 gh（通过 API）：**

```bash
HEAD_SHA=$(gh pr view 123 --json headRefOid --jq '.headRefOid')

gh api repos/$OWNER/$REPO/pulls/123/comments \
  --method POST \
  -f body="This could be simplified with a list comprehension." \
  -f path="src/auth/login.py" \
  -f commit_id="$HEAD_SHA" \
  -f line=45 \
  -f side="RIGHT"
```

**单条行内评论——使用 curl：**

```bash
# 获取 head commit SHA
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments \
  -d "{
    \"body\": \"This could be simplified with a list comprehension.\",
    \"path\": \"src/auth/login.py\",
    \"commit_id\": \"$HEAD_SHA\",
    \"line\": 45,
    \"side\": \"RIGHT\"
  }"
```

### 提交正式审查（批准 / 请求变更）

**使用 gh：**

```bash
gh pr review 123 --approve --body "LGTM!"
gh pr review 123 --request-changes --body "See inline comments."
gh pr review 123 --comment --body "Some suggestions, nothing blocking."
```

**使用 curl——原子性提交包含多条评论的审查：**

```bash
HEAD_SHA=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"COMMENT\",
    \"body\": \"Code review from Hermes Agent\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"Use parameterized queries to prevent SQL injection.\"},
      {\"path\": \"src/models/user.py\", \"line\": 23, \"body\": \"Hash passwords with bcrypt before storing.\"},
      {\"path\": \"tests/test_auth.py\", \"line\": 1, \"body\": \"Add test for expired token edge case.\"}
    ]
  }"
```

事件值：`"APPROVE"`、`"REQUEST_CHANGES"`、`"COMMENT"`

`line` 字段指文件*新版本*中的行号。对于已删除的行，使用 `"side": "LEFT"`。

---

## 3. 审查清单

执行代码审查（本地或 PR）时，系统性地检查以下内容：

### 正确性
- 代码是否实现了其声称的功能？
- 边界情况是否已处理（空输入、null、大数据、并发访问）？
- 错误路径是否优雅处理？

### 安全性
- 无硬编码的密钥、凭据或 API key
- 对用户输入进行验证
- 无 SQL 注入、XSS 或路径遍历
- 在需要的地方进行身份验证/授权检查

### 代码质量
- 命名清晰（变量、函数、类）
- 无不必要的复杂性或过早抽象
- DRY——无应提取的重复逻辑
- 函数职责单一

### 测试
- 新代码路径是否已测试？
- 正常路径和错误情况是否已覆盖？
- 测试是否可读且可维护？

### 性能
- 无 N+1 查询或不必要的循环
- 在适当位置使用缓存
- 异步代码路径中无阻塞操作

### 文档
- 公共 API 已文档化
- 非显而易见的逻辑有注释说明"为什么"
- 若行为发生变化，README 已更新

---

## 4. 推送前审查工作流

当用户要求"审查代码"或"推送前检查"时：

1. `git diff main...HEAD --stat`——了解变更范围
2. `git diff main...HEAD`——阅读完整差异
3. 对每个已变更的文件，如需更多上下文则使用 `read_file`
4. 应用上述审查清单
5. 按结构化格式呈现结果（Critical / Warnings / Suggestions / Looks Good）
6. 若发现严重问题，在用户推送前主动提出修复

---

## 5. PR 审查工作流（端到端）

当用户要求"审查 PR #N"、"查看这个 PR"，或提供 PR URL 时，按以下步骤执行：

### 第一步：设置环境

```bash
source "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/gh-env.sh"
# 或运行本 skill 顶部的内联设置代码块
```

### 第二步：收集 PR 上下文

获取 PR 元数据、描述和已变更文件列表，在深入代码之前了解变更范围。

**使用 gh：**
```bash
gh pr view 123
gh pr diff 123 --name-only
gh pr checks 123
```

**使用 curl：**
```bash
PR_NUMBER=123

# PR 详情（标题、作者、描述、分支）
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER

# 带行数统计的已变更文件
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/files
```

### 第三步：在本地检出 PR

这样可以完整使用 `read_file`、`search_files`，以及运行测试的能力。

```bash
git fetch origin pull/$PR_NUMBER/head:pr-$PR_NUMBER
git checkout pr-$PR_NUMBER
```

### 第四步：阅读差异并理解变更

```bash
# 与基础分支的完整差异
git diff main...HEAD

# 对于大型 PR，逐文件查看
git diff main...HEAD --name-only
# 然后对每个文件：
git diff main...HEAD -- path/to/file.py
```

对每个已变更的文件，使用 `read_file` 查看变更周围的完整上下文——仅凭差异可能遗漏只有在周围代码中才能发现的问题。

### 第五步：在本地运行自动化检查（如适用）

```bash
# 若有测试套件，运行测试
python -m pytest 2>&1 | tail -20
# 或：npm test, cargo test, go test ./..., 等

# 若已配置，运行 linter
ruff check . 2>&1 | head -30
# 或：eslint, clippy, 等
```

### 第六步：应用审查清单（第 3 节）

逐一检查每个类别：正确性、安全性、代码质量、测试、性能、文档。

### 第七步：将审查结果发布到 GitHub

汇总结果并以正式审查形式提交，附带行内评论。

**使用 gh：**
```bash
# 若无问题——批准
gh pr review $PR_NUMBER --approve --body "Reviewed by Hermes Agent. Code looks clean — good test coverage, no security concerns."

# 若发现问题——请求变更并附行内评论
gh pr review $PR_NUMBER --request-changes --body "Found a few issues — see inline comments."
```

**使用 curl——原子性提交包含多条行内评论的审查：**
```bash
HEAD_SHA=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['head']['sha'])")

# 构建审查 JSON——event 为 APPROVE、REQUEST_CHANGES 或 COMMENT
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$GH_OWNER/$GH_REPO/pulls/$PR_NUMBER/reviews \
  -d "{
    \"commit_id\": \"$HEAD_SHA\",
    \"event\": \"REQUEST_CHANGES\",
    \"body\": \"## Hermes Agent Review\n\nFound 2 issues, 1 suggestion. See inline comments.\",
    \"comments\": [
      {\"path\": \"src/auth.py\", \"line\": 45, \"body\": \"🔴 **Critical:** User input passed directly to SQL query — use parameterized queries.\"},
      {\"path\": \"src/models.py\", \"line\": 23, \"body\": \"⚠️ **Warning:** Password stored without hashing.\"},
      {\"path\": \"src/utils.py\", \"line\": 8, \"body\": \"💡 **Suggestion:** This duplicates logic in core/utils.py:34.\"}
    ]
  }"
```

### 第八步：同时发布摘要评论

除行内评论外，还需留下顶层摘要，让 PR 作者一目了然地了解全貌。使用 `references/review-output-template.md` 中的审查输出格式。

**使用 gh：**
```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
## Code Review Summary

**Verdict: Changes Requested** (2 issues, 1 suggestion)

### 🔴 Critical
- **src/auth.py:45** — SQL injection vulnerability

### ⚠️ Warnings
- **src/models.py:23** — Plaintext password storage

### 💡 Suggestions
- **src/utils.py:8** — Duplicated logic, consider consolidating

### ✅ Looks Good
- Clean API design
- Good error handling in the middleware layer

---
*Reviewed by Hermes Agent*
EOF
)"
```

### 第九步：清理

```bash
git checkout main
git branch -D pr-$PR_NUMBER
```

### 决策：批准 vs 请求变更 vs 评论

- **批准（Approve）**——无严重或警告级别的问题，仅有次要建议或完全通过
- **请求变更（Request Changes）**——存在任何在合并前应修复的严重或警告级别问题
- **评论（Comment）**——有观察和建议，但无阻塞性问题（在不确定或 PR 为草稿时使用）