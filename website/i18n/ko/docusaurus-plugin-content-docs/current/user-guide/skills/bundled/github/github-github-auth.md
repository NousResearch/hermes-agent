---
title: "Github Auth — GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인"
sidebar_label: "Github Auth"
description: "GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Github Auth

GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함(기본 설치됨) |
| 경로 | `skills/github/github-auth` |
| 버전 | `1.1.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `GitHub`, `Authentication`, `Git`, `gh-cli`, `SSH`, `Setup` |
| 관련 스킬 | [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow), [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review), [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues), [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# GitHub 인증 설정

이 스킬은 에이전트가 GitHub 리포지토리, PR, 이슈, CI에서 작업할 수 있도록 인증을 설정합니다. 다음 두 가지 방법을 다룹니다.

- **`git` (항상 사용 가능)** — HTTPS 개인 액세스 토큰 또는 SSH 키 사용
- **`gh` CLI (설치된 경우)** — 더 풍부한 GitHub API 액세스와 더 간단한 인증 흐름 제공

## 감지 흐름

사용자가 GitHub 작업을 요청하면 먼저 다음 검사를 실행합니다.

```bash
# Check what's available
git --version
gh --version 2>/dev/null || echo "gh not installed"

# Check if already authenticated
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**의사 결정 트리:**
1. `gh auth status`가 인증됨을 표시하면 → 준비가 완료된 것이므로 모든 작업에 `gh` 사용
2. `gh`가 설치되어 있지만 인증되지 않았다면 → 아래의 "gh auth" 방법 사용
3. `gh`가 설치되어 있지 않다면 → 아래의 "git 전용" 방법 사용(sudo 불필요)

---

## 방법 1: Git 전용 인증(gh 및 sudo 불필요)

이 방법은 `git`이 설치된 모든 컴퓨터에서 작동합니다. 루트 액세스가 필요하지 않습니다.

### 옵션 A: 개인 액세스 토큰을 사용한 HTTPS(권장)

가장 이식성이 높은 방법입니다. SSH 설정 없이 어디서나 작동합니다.

**1단계: 개인 액세스 토큰 생성**

사용자에게 **https://github.com/settings/tokens**로 이동하도록 안내합니다.

- "Generate new token (classic)" 클릭
- "hermes-agent"와 같은 이름 지정
- 범위 선택:
  - `repo` (전체 리포지토리 액세스 — 읽기, 쓰기, 푸시, PR)
  - `workflow` (GitHub Actions 트리거 및 관리)
  - `read:org` (조직 리포지토리로 작업하는 경우)
- 만료 기간 설정(90일을 기본값으로 권장)
- 토큰 복사 — 다시 표시되지 않습니다

**2단계: 토큰을 저장하도록 git 구성**

```bash
# Set up the credential helper to cache credentials
# "store" saves to ~/.git-credentials in plaintext (simple, persistent)
git config --global credential.helper store

# Now do a test operation that triggers auth — git will prompt for credentials
# Username: <their-github-username>
# Password: <paste the personal access token, NOT their GitHub password>
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

한 번 인증 정보를 입력하면 저장되어 이후 모든 작업에서 재사용됩니다.

**대안: 캐시 헬퍼(인증 정보가 메모리에서 만료됨)**

```bash
# Cache in memory for 8 hours (28800 seconds) instead of saving to disk
git config --global credential.helper 'cache --timeout=28800'
```

**대안: 원격 URL에 토큰 직접 설정(리포지토리별)**

```bash
# Embed token in the remote URL (avoids credential prompts entirely)
git remote set-url origin https://<username>:<token>@github.com/<owner>/<repo>.git
```

**3단계: git ID 구성**

```bash
# Required for commits — set name and email
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**4단계: 확인**

```bash
# Test push access (this should work without any prompts now)
git ls-remote https://github.com/<their-username>/<any-repo>.git

# Verify identity
git config --global user.name
git config --global user.email
```

### 옵션 B: SSH 키 인증

SSH를 선호하거나 이미 키를 설정한 사용자에게 적합합니다.

**1단계: 기존 SSH 키 확인**

```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**2단계: 필요한 경우 키 생성**

```bash
# Generate an ed25519 key (modern, secure, fast)
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519 -N ""

# Display the public key for them to add to GitHub
cat ~/.ssh/id_ed25519.pub
```

사용자에게 **https://github.com/settings/keys**에서 공개 키를 추가하도록 안내합니다.
- "New SSH key" 클릭
- 공개 키 내용 붙여넣기
- "hermes-agent-&lt;machine-name>"과 같은 제목 지정

**3단계: 연결 테스트**

```bash
ssh -T git@github.com
# Expected: "Hi <username>! You've successfully authenticated..."
```

**4단계: GitHub에서 SSH를 사용하도록 git 구성**

```bash
# Rewrite HTTPS GitHub URLs to SSH automatically
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

**5단계: git ID 구성**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

---

## 방법 2: gh CLI 인증

`gh`가 설치되어 있으면 한 번에 API 액세스와 git 인증 정보를 모두 처리합니다.

### 대화형 브라우저 로그인(데스크톱)

```bash
gh auth login
# Select: GitHub.com
# Select: HTTPS
# Authenticate via browser
```

### 토큰 기반 로그인(헤드리스/SSH 서버)

```bash
echo "<THEIR_TOKEN>" | gh auth login --with-token

# Set up git credentials through gh
gh auth setup-git
```

### 확인

```bash
gh auth status
```

---

## gh 없이 GitHub API 사용

`gh`를 사용할 수 없어도 개인 액세스 토큰과 함께 `curl`을 사용하면 전체 GitHub API에 액세스할 수 있습니다. 다른 GitHub 스킬은 폴백으로 이 방식을 구현합니다.

### API 호출을 위한 토큰 설정

```bash
# Option 1: Export as env var (preferred — keeps it out of commands)
export GITHUB_TOKEN="<token>"

# Then use in curl calls:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### Git 인증 정보에서 토큰 추출

git 인증 정보가 이미 구성되어 있으면(credential.helper store를 통해) 토큰을 추출할 수 있습니다.

```bash
# Read from git credential store
uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py"
```

### 헬퍼: 인증 방법 감지

모든 GitHub 워크플로의 시작 부분에 다음 패턴을 사용합니다.

```bash
# Try gh first, fall back to git + curl
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif _hermes_env="${HERMES_HOME:-$HOME/.hermes}/.env"; [ -f "$_hermes_env" ] && grep -q "^GITHUB_TOKEN=" "$_hermes_env"; then
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" "$_hermes_env" | head -1 | cut -d= -f2 | tr -d '\n\r')
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(uv run python3 "${HERMES_HOME:-$HOME/.hermes}/skills/github/github-auth/scripts/git-credential-token.py")
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| `git push` asks for password | GitHub는 비밀번호 인증을 비활성화했습니다. 비밀번호로 개인 액세스 토큰을 사용하거나 SSH로 전환하세요 |
| `remote: Permission to X denied` | 토큰에 `repo` 범위가 없을 수 있습니다 — 올바른 범위로 다시 생성하세요 |
| `fatal: Authentication failed` | 캐시된 인증 정보가 오래되었을 수 있습니다 — `git credential reject`를 실행한 후 다시 인증하세요 |
| `ssh: connect to host github.com port 22: Connection refused` | HTTPS 포트를 통한 SSH를 시도하세요: `Host github.com`과 `Port 443`, `Hostname ssh.github.com`을 `~/.ssh/config`에 추가하세요 |
| Credentials not persisting | `git config --global credential.helper`를 확인하세요 — `store` 또는 `cache`여야 합니다 |
| Multiple GitHub accounts | `~/.ssh/config`에서 호스트 별칭마다 다른 키를 사용하는 SSH 또는 리포지토리별 인증 정보 URL을 사용하세요 |
| `gh: command not found` + no sudo | 위의 Git 전용 방법 1을 사용하세요 — 설치할 필요가 없습니다 |
