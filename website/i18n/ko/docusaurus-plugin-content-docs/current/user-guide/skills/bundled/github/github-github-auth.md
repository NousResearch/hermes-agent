---
title: "Github Auth — GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인"
sidebar_label: "Github Auth"
description: "GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Github Auth

GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/github/github-auth` |
| Version | `1.1.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `GitHub`, `Authentication`, `Git`, `gh-cli`, `SSH`, `Setup` |
| Related skills | [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow), [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review), [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues), [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# GitHub 인증 설정 (GitHub Authentication Setup)

이 스킬은 에이전트가 GitHub 리포지토리, PR, 이슈 및 CI로 작업할 수 있도록 인증을 설정합니다. 다음 두 가지 경로를 다룹니다:

- **`git` (항상 사용 가능)** — HTTPS 개인 액세스 토큰(Personal Access Token) 또는 SSH 키 사용
- **`gh` CLI (설치된 경우)** — 더 간단한 인증 흐름을 가진 풍부한 GitHub API 액세스

## 감지 흐름 (Detection Flow)

사용자가 GitHub와 관련된 작업을 요청할 때, 먼저 다음 확인을 실행하세요:

```bash
# 사용 가능한 것 확인
git --version
gh --version 2>/dev/null || echo "gh not installed"

# 이미 인증되었는지 확인
gh auth status 2>/dev/null || echo "gh not authenticated"
git config --global credential.helper 2>/dev/null || echo "no git credential helper"
```

**결정 트리(Decision tree):**
1. `gh auth status`가 인증되었다고 표시하는 경우 → 준비 완료, 모든 작업에 `gh` 사용
2. `gh`가 설치되어 있지만 인증되지 않은 경우 → 아래 "gh 인증" 방법 사용
3. `gh`가 설치되지 않은 경우 → 아래 "git 전용" 방법 사용 (sudo 불필요)

---

## 방법 1: Git 전용 인증 (gh 없음, sudo 없음)

이 방법은 `git`이 설치된 모든 머신에서 작동합니다. 루트 액세스가 필요하지 않습니다.

### 옵션 A: 개인 액세스 토큰을 사용한 HTTPS (권장)

이것은 가장 이식성이 높은 방법입니다 — 어디서나 작동하며, SSH 설정이 필요하지 않습니다.

**1단계: 개인 액세스 토큰 생성**

사용자에게 다음 링크로 이동하도록 안내합니다: **https://github.com/settings/tokens**

- "Generate new token (classic)" 클릭
- "hermes-agent"와 같은 이름 지정
- 권한(scopes) 선택:
  - `repo` (전체 리포지토리 액세스 — 읽기, 쓰기, 푸시, PR)
  - `workflow` (GitHub Actions 트리거 및 관리)
  - `read:org` (조직 리포지토리로 작업하는 경우)
- 만료일 설정 (90일이 기본값으로 좋음)
- 토큰 복사 — 다시 표시되지 않음

**2단계: 토큰을 저장하도록 git 구성**

```bash
# 자격 증명을 캐시하도록 자격 증명 헬퍼 설정
# "store"는 ~/.git-credentials에 일반 텍스트로 저장합니다 (간단하고 영구적임)
git config --global credential.helper store

# 이제 인증을 트리거하는 테스트 작업을 수행 — git이 자격 증명을 묻는 프롬프트를 표시함
# Username: <사용자의-github-사용자이름>
# Password: <사용자의 GitHub 비밀번호가 아닌, 복사한 개인 액세스 토큰 붙여넣기>
git ls-remote https://github.com/<their-username>/<any-repo>.git
```

자격 증명을 한 번 입력하면, 저장되고 모든 향후 작업에 재사용됩니다.

**대안: 캐시 헬퍼 (자격 증명이 메모리에서 만료됨)**

```bash
# 디스크에 저장하는 대신 8시간 (28800초) 동안 메모리에 캐시
git config --global credential.helper 'cache --timeout=28800'
```

**대안: 원격 URL에 토큰을 직접 설정 (리포지토리별)**

```bash
# 원격 URL에 토큰 포함 (자격 증명 프롬프트를 완전히 방지)
git remote set-url origin https://<username>:<token>@github.com/<owner>/<repo>.git
```

**3단계: git ID 구성**

```bash
# 커밋에 필요 — 이름과 이메일 설정
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

**4단계: 확인**

```bash
# 푸시 액세스 테스트 (이제 프롬프트 없이 작동해야 함)
git ls-remote https://github.com/<their-username>/<any-repo>.git

# ID 확인
git config --global user.name
git config --global user.email
```

### 옵션 B: SSH 키 인증

SSH를 선호하거나 이미 키가 설정된 사용자에게 좋습니다.

**1단계: 기존 SSH 키 확인**

```bash
ls -la ~/.ssh/id_*.pub 2>/dev/null || echo "No SSH keys found"
```

**2단계: 필요한 경우 키 생성**

```bash
# ed25519 키 생성 (최신, 안전함, 빠름)
ssh-keygen -t ed25519 -C "their-email@example.com" -f ~/.ssh/id_ed25519 -N ""

# 사용자가 GitHub에 추가할 수 있도록 공개 키 표시
cat ~/.ssh/id_ed25519.pub
```

사용자에게 다음 링크에서 공개 키를 추가하도록 안내합니다: **https://github.com/settings/keys**
- "New SSH key" 클릭
- 공개 키 내용 붙여넣기
- "hermes-agent-&lt;machine-name>"과 같은 제목 지정

**3단계: 연결 테스트**

```bash
ssh -T git@github.com
# 예상: "Hi <username>! You've successfully authenticated..."
```

**4단계: GitHub에 SSH를 사용하도록 git 구성**

```bash
# HTTPS GitHub URL을 자동으로 SSH로 다시 쓰기
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

**5단계: git ID 구성**

```bash
git config --global user.name "Their Name"
git config --global user.email "their-email@example.com"
```

---

## 방법 2: gh CLI 인증

`gh`가 설치된 경우, API 액세스와 git 자격 증명을 한 번에 처리합니다.

### 대화형 브라우저 로그인 (데스크탑)

```bash
gh auth login
# 선택: GitHub.com
# 선택: HTTPS
# 브라우저를 통해 인증
```

### 토큰 기반 로그인 (헤드리스 / SSH 서버)

```bash
echo "<THEIR_TOKEN>" | gh auth login --with-token

# gh를 통해 git 자격 증명 설정
gh auth setup-git
```

### 확인

```bash
gh auth status
```

---

## gh 없이 GitHub API 사용하기

`gh`를 사용할 수 없는 경우, 개인 액세스 토큰과 함께 `curl`을 사용하여 전체 GitHub API에 액세스할 수 있습니다. 이것이 다른 GitHub 스킬들이 폴백을 구현하는 방법입니다.

### API 호출을 위한 토큰 설정

```bash
# 옵션 1: 환경 변수로 내보내기 (권장 — 명령에서 제외시킴)
export GITHUB_TOKEN="<token>"

# 그런 다음 curl 호출에서 사용:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user
```

### Git 자격 증명에서 토큰 추출

git 자격 증명이 이미 구성된 경우 (credential.helper store를 통해), 토큰을 추출할 수 있습니다:

```bash
# git 자격 증명 저장소에서 읽기
grep "github.com" ~/.git-credentials 2>/dev/null | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|'
```

### 헬퍼: 인증 방법 감지

GitHub 워크플로우를 시작할 때 이 패턴을 사용하세요:

```bash
# gh를 먼저 시도하고, git + curl로 폴백
if command -v gh &>/dev/null && gh auth status &>/dev/null; then
  echo "AUTH_METHOD=gh"
elif [ -n "$GITHUB_TOKEN" ]; then
  echo "AUTH_METHOD=curl"
elif [ -f ~/.hermes/.env ] && grep -q "^GITHUB_TOKEN=" ~/.hermes/.env; then
  export GITHUB_TOKEN=$(grep "^GITHUB_TOKEN=" ~/.hermes/.env | head -1 | cut -d= -f2 | tr -d '\n\r')
  echo "AUTH_METHOD=curl"
elif grep -q "github.com" ~/.git-credentials 2>/dev/null; then
  export GITHUB_TOKEN=$(grep "github.com" ~/.git-credentials | head -1 | sed 's|https://[^:]*:\([^@]*\)@.*|\1|')
  echo "AUTH_METHOD=curl"
else
  echo "AUTH_METHOD=none"
  echo "Need to set up authentication first"
fi
```

---

## 문제 해결 (Troubleshooting)

| 문제 | 해결책 |
|---------|----------|
| `git push` 가 비밀번호를 물어봄 | GitHub가 비밀번호 인증을 비활성화했습니다. 비밀번호 대신 개인 액세스 토큰을 사용하거나, SSH로 전환하세요 |
| `remote: Permission to X denied` | 토큰에 `repo` 권한이 없을 수 있습니다 — 올바른 권한으로 다시 생성하세요 |
| `fatal: Authentication failed` | 캐시된 자격 증명이 오래되었을 수 있습니다 — `git credential reject`를 실행한 후 다시 인증하세요 |
| `ssh: connect to host github.com port 22: Connection refused` | HTTPS 포트를 통한 SSH를 시도하세요: `~/.ssh/config`에 `Port 443`과 `Hostname ssh.github.com`이 있는 `Host github.com`을 추가하세요 |
| 자격 증명이 유지되지 않음 | `git config --global credential.helper` 확인 — `store` 또는 `cache`여야 합니다 |
| 여러 GitHub 계정 | `~/.ssh/config`에서 호스트 별칭(alias)별로 다른 키와 함께 SSH를 사용하거나, 리포지토리별 자격 증명 URL을 사용하세요 |
| `gh: command not found` + sudo 없음 | 위의 git 전용 방법 1을 사용하세요 — 설치가 필요하지 않습니다 |
