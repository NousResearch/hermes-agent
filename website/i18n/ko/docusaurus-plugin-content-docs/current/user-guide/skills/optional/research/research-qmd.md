---
title: "Qmd — 노트, 문서, 트랜스크립트에 대한 하이브리드 로컬 검색"
sidebar_label: "Qmd"
description: "노트, 문서, 트랜스크립트에 대한 하이브리드 로컬 검색"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 소스 SKILL.md를 편집하세요. */}

# Qmd

노트, 문서, 트랜스크립트에 대한 하이브리드 로컬 검색입니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/research/qmd`로 설치 |
| 경로 | `optional-skills/research/qmd` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | macos, linux |
| 태그 | `Search`, `Knowledge-Base`, `RAG`, `Notes`, `MCP`, `Local-AI` |
| 관련 스킬 | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지침으로 보는 내용입니다.
:::

# QMD — 마크업 문서 질의

개인 지식 기반을 위한 로컬 온디바이스 검색 엔진입니다. 마크다운 노트,
회의 트랜스크립트, 문서 및 텍스트 기반 파일을 색인한 다음 키워드 일치,
의미 이해, LLM 기반 재순위를 결합한 하이브리드 검색을 제공합니다. 모든
기능은 클라우드 의존성 없이 로컬에서 실행됩니다.

[Tobi Lütke](https://github.com/tobi/qmd)가 만들었습니다. MIT 라이선스입니다.

## 사용 시점

- 사용자가 자신의 노트, 문서, 지식 기반 또는 회의 트랜스크립트 검색을 요청할 때
- 대규모 마크다운/텍스트 파일 모음에서 무언가를 찾고 싶어 할 때
- 단순한 키워드 grep이 아닌 의미 검색("X 개념에 관한 노트 찾기")을 원할 때
- 이미 qmd 컬렉션을 설정했으며 이를 질의하고 싶을 때
- 로컬 지식 기반 또는 문서 검색 시스템 설정을 요청할 때
- 키워드: "search my notes", "find in my docs", "knowledge base", "qmd"

## 사전 요구 사항

### Node.js >= 22 (필수)

```bash
# Check version
node --version  # must be >= 22

# macOS — install or upgrade via Homebrew
brew install node@22

# Linux — use NodeSource or nvm
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
# or with nvm:
nvm install 22 && nvm use 22
```

### 확장 기능을 지원하는 SQLite (macOS 전용)

macOS 시스템 SQLite는 확장 기능 로딩을 지원하지 않습니다. Homebrew를 통해 설치하세요.

```bash
brew install sqlite
```

### qmd 설치

```bash
npm install -g @tobilu/qmd
# or with Bun:
bun install -g @tobilu/qmd
```

첫 실행 시 로컬 GGUF 모델 3개(총 약 2GB)를 자동으로 다운로드합니다.

| 모델 | 용도 | 크기 |
|-------|------|------|
| embeddinggemma-300M-Q8_0 | 벡터 임베딩 | ~300MB |
| qwen3-reranker-0.6b-q8_0 | 결과 재순위 지정 | ~640MB |
| qmd-query-expansion-1.7B | 질의 확장 | ~1.1GB |

### 설치 확인

```bash
qmd --version
qmd status
```

## 빠른 참조

| 명령 | 기능 | 속도 |
|---------|-------------|-------|
| `qmd search "query"` | BM25 키워드 검색(모델 없음) | ~0.2초 |
| `qmd vsearch "query"` | 의미 벡터 검색(모델 1개) | ~3초 |
| `qmd query "query"` | 하이브리드 + 재순위 지정(모델 3개 모두 사용) | 웜 상태 ~2~3초, 콜드 상태 ~19초 |
| `qmd get <docid>` | 전체 문서 내용 검색 | 즉시 |
| `qmd multi-get "glob"` | 여러 파일 검색 | 즉시 |
| `qmd collection add <path> --name <n>` | 디렉터리를 컬렉션으로 추가 | 즉시 |
| `qmd context add <path> "description"` | 검색 결과 개선을 위한 컨텍스트 메타데이터 추가 | 즉시 |
| `qmd embed` | 벡터 임베딩 생성/업데이트 | 가변 |
| `qmd status` | 색인 상태 및 컬렉션 정보 표시 | 즉시 |
| `qmd mcp` | MCP 서버 시작(stdio) | 지속 실행 |
| `qmd mcp --http --daemon` | MCP 서버 시작(HTTP, 웜 모델) | 지속 실행 |

## 설정 워크플로

### 1. 컬렉션 추가

문서가 들어 있는 디렉터리를 qmd에 지정하세요.

```bash
# Add a notes directory
qmd collection add ~/notes --name notes

# Add project docs
qmd collection add ~/projects/myproject/docs --name project-docs

# Add meeting transcripts
qmd collection add ~/meetings --name meetings

# List all collections
qmd collection list
```

### 2. 컨텍스트 설명 추가

컨텍스트 메타데이터는 검색 엔진이 각 컬렉션의 내용을 이해하는 데 도움이 됩니다. 검색 품질이 크게 향상됩니다.

```bash
qmd context add qmd://notes "Personal notes, ideas, and journal entries"
qmd context add qmd://project-docs "Technical documentation for the main project"
qmd context add qmd://meetings "Meeting transcripts and action items from team syncs"
```

### 3. 임베딩 생성

```bash
qmd embed
```

모든 컬렉션의 모든 문서를 처리하고 벡터 임베딩을 생성합니다. 새 문서나 컬렉션을 추가한 후 다시 실행하세요.

### 4. 확인

```bash
qmd status   # shows index health, collection stats, model info
```

## 검색 패턴

### 빠른 키워드 검색(BM25)

적합한 경우: 정확한 용어, 코드 식별자, 이름, 알려진 구문.
모델을 로드하지 않으므로 거의 즉시 결과가 나옵니다.

```bash
qmd search "authentication middleware"
qmd search "handleError async"
```

### 의미 벡터 검색

적합한 경우: 자연어 질문, 개념적 질의.
첫 질의에서 임베딩 모델을 로드합니다(~3초).

```bash
qmd vsearch "how does the rate limiter handle burst traffic"
qmd vsearch "ideas for improving onboarding flow"
```

### 재순위 지정을 포함한 하이브리드 검색(최고 품질)

적합한 경우: 품질이 가장 중요한 질의.
질의 확장, 병렬 BM25+벡터 검색, 재순위 지정에 모델 3개를 모두 사용합니다.

```bash
qmd query "what decisions were made about the database migration"
```

### 구조화된 다중 모드 질의

정밀도를 위해 한 질의에서 여러 검색 유형을 결합합니다.

```bash
# BM25 for exact term + vector for concept
qmd query $'lex: rate limiter\nvec: how does throttling work under load'

# With query expansion
qmd query $'expand: database migration plan\nlex: "schema change"'
```

### 질의 구문(lex/BM25 모드)

| 구문 | 효과 | 예시 |
|--------|---------|---------|
| `term` | 접두사 일치 | `perf`는 "performance"와 일치 |
| `"phrase"` | 정확한 구문 | `"rate limiter"` |
| `-term` | 용어 제외 | `performance -sports` |

### HyDE(가상 문서 임베딩)

복잡한 주제에서는 답변이 어떤 모습일지 예상하여 작성하세요.

```bash
qmd query $'hyde: The migration plan involves three phases. First, we add the new columns without dropping the old ones. Then we backfill data. Finally we cut over and remove legacy columns.'
```

### 컬렉션 범위 지정

```bash
qmd search "query" --collection notes
qmd query "query" --collection project-docs
```

### 출력 형식

```bash
qmd search "query" --json        # JSON output (best for parsing)
qmd search "query" --limit 5     # Limit results
qmd get "#abc123"                # Get by document ID
qmd get "path/to/file.md"       # Get by file path
qmd get "file.md:50" -l 100     # Get specific line range
qmd multi-get "journals/*.md" --json  # Batch retrieve by glob
```

## MCP 통합(권장)

qmd는 Hermes Agent의 네이티브 MCP 클라이언트를 통해 검색 도구를 직접 제공하는 MCP 서버를 노출합니다. 권장되는 통합 방식입니다. 한 번 설정하면 이 스킬을 로드하지 않아도 에이전트가 qmd 도구를 자동으로 받습니다.

### 옵션 A: Stdio 모드(간단)

`~/.hermes/config.yaml`에 추가하세요.

```yaml
mcp_servers:
  qmd:
    command: "qmd"
    args: ["mcp"]
    timeout: 30
    connect_timeout: 45
```

다음 도구가 등록됩니다: `mcp_qmd_search`, `mcp_qmd_vsearch`,
`mcp_qmd_deep_search`, `mcp_qmd_get`, `mcp_qmd_status`.

**절충점:** 첫 검색 호출 시 모델을 로드하므로(~19초의 콜드 스타트), 이후 세션 동안 웜 상태로 유지됩니다. 가끔 사용하는 경우에는 충분합니다.

### 옵션 B: HTTP 데몬 모드(고속, 대량 사용에 권장)

qmd 데몬을 별도로 시작하면 메모리에 모델을 웜 상태로 유지합니다.

```bash
# Start daemon (persists across agent restarts)
qmd mcp --http --daemon

# Runs on http://localhost:8181 by default
```

그런 다음 Hermes Agent가 HTTP를 통해 연결하도록 설정합니다.

```yaml
mcp_servers:
  qmd:
    url: "http://localhost:8181/mcp"
    timeout: 30
```

**절충점:** 실행 중 약 2GB RAM을 사용하지만 모든 질의가 빠릅니다(~2~3초). 자주 검색하는 사용자에게 가장 적합합니다.

### 데몬 실행 유지

#### macOS(launchd)

```bash
cat > ~/Library/LaunchAgents/com.qmd.daemon.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.qmd.daemon</string>
  <key>ProgramArguments</key>
  <array>
    <string>qmd</string>
    <string>mcp</string>
    <string>--http</string>
    <string>--daemon</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/qmd-daemon.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/qmd-daemon.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.qmd.daemon.plist
```

#### Linux(systemd 사용자 서비스)

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/qmd-daemon.service << 'EOF'
[Unit]
Description=QMD MCP Daemon
After=network.target

[Service]
ExecStart=qmd mcp --http --daemon
Restart=on-failure
RestartSec=10
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now qmd-daemon
systemctl --user status qmd-daemon
```

### MCP 도구 참조

연결되면 다음 도구를 `mcp_qmd_*`로 사용할 수 있습니다.

| MCP 도구 | 매핑 대상 | 설명 |
|----------|---------|-------------|
| `mcp_qmd_search` | `qmd search` | BM25 키워드 검색 |
| `mcp_qmd_vsearch` | `qmd vsearch` | 의미 벡터 검색 |
| `mcp_qmd_deep_search` | `qmd query` | 하이브리드 검색 + 재순위 지정 |
| `mcp_qmd_get` | `qmd get` | ID 또는 경로로 문서 검색 |
| `mcp_qmd_status` | `qmd status` | 색인 상태 및 통계 |

MCP 도구는 다중 모드 검색을 위한 구조화된 JSON 질의를 받습니다.

```json
{
  "searches": [
    {"type": "lex", "query": "authentication middleware"},
    {"type": "vec", "query": "how user login is verified"}
  ],
  "collections": ["project-docs"],
  "limit": 10
}
```

## CLI 사용(MCP 없이)

MCP가 설정되지 않은 경우 터미널에서 qmd를 직접 사용하세요.

```
terminal(command="qmd query 'what was decided about the API redesign' --json", timeout=30)
```

설정 및 관리 작업에는 항상 터미널을 사용하세요.

```
terminal(command="qmd collection add ~/Documents/notes --name notes")
terminal(command="qmd context add qmd://notes 'Personal research notes and ideas'")
terminal(command="qmd embed")
terminal(command="qmd status")
```

## 검색 파이프라인 작동 방식

내부 동작을 이해하면 적절한 검색 모드를 선택하는 데 도움이 됩니다.

1. **질의 확장** — 미세 조정된 1.7B 모델이 대체 질의 2개를 생성합니다. 원래 질의에는 융합 시 2배 가중치가 적용됩니다.
2. **병렬 검색** — 모든 질의 변형에 대해 BM25(SQLite FTS5)와 벡터 검색이 동시에 실행됩니다.
3. **RRF 융합** — 상호 순위 융합(Reciprocal Rank Fusion, k=60)이 결과를 병합합니다. 최상위 순위 보너스: 1위는 +0.05, 2~3위는 +0.02입니다.
4. **LLM 재순위 지정** — qwen3-reranker가 상위 후보 30개의 점수를 매깁니다(0.0~1.0).
5. **위치 인식 블렌딩** — 1~3위: 검색 75% / 재순위 지정 25%. 4~10위: 60/40. 11위 이상: 40/60(긴 꼬리에 대해서는 재순위 지정기를 더 신뢰).

**스마트 청킹:** 문서는 자연스러운 중단점(제목, 코드 블록, 빈 줄)에서 약 900토큰을 목표로 15% 중첩하여 분할됩니다. 코드 블록은 블록 중간에서 절대 분할되지 않습니다.

## 모범 사례

1. **항상 컨텍스트 설명을 추가하세요** — `qmd context add`는 검색 정확도를 크게 향상합니다. 각 컬렉션에 무엇이 들어 있는지 설명하세요.
2. **문서 추가 후 다시 임베딩하세요** — 컬렉션에 새 파일을 추가하면 `qmd embed`를 다시 실행해야 합니다.
3. **속도에는 `qmd search`를 사용하세요** — 빠른 키워드 조회(코드 식별자, 정확한 이름)가 필요할 때 BM25는 즉시 결과를 내며 모델이 필요하지 않습니다.
4. **품질에는 `qmd query`를 사용하세요** — 질문이 개념적이거나 사용자가 가능한 최상의 결과를 원할 때 하이브리드 검색을 사용하세요.
5. **MCP 통합을 우선하세요** — 한 번 설정하면 매번 이 스킬을 로드하지 않아도 에이전트가 네이티브 도구를 받습니다.
6. **자주 사용하는 사용자에게는 데몬 모드를 사용하세요** — 사용자가 지식 기반을 정기적으로 검색한다면 HTTP 데몬 설정을 권장하세요.
7. **구조화된 검색의 첫 질의에는 2배 가중치가 적용됩니다** — lex와 vec을 결합할 때 가장 중요하고 확실한 질의를 먼저 배치하세요.

## 문제 해결

### "첫 실행 시 모델을 다운로드합니다"

정상적인 동작입니다 — qmd는 처음 사용할 때 약 2GB의 GGUF 모델을 자동으로 다운로드합니다. 한 번만 수행됩니다.

### 콜드 스타트 지연(~19초)

모델이 메모리에 로드되지 않았을 때 발생합니다. 해결 방법:
- HTTP 데몬 모드(`qmd mcp --http --daemon`)를 사용하여 웜 상태 유지
- 모델이 필요하지 않을 때 `qmd search`(BM25 전용) 사용
- MCP stdio 모드는 첫 검색 시 모델을 로드하고 세션 동안 웜 상태로 유지

### macOS: "확장 기능을 로드할 수 없음"

Homebrew SQLite를 설치하세요: `brew install sqlite`
그런 다음 시스템 SQLite보다 먼저 PATH에 있는지 확인하세요.

### "컬렉션을 찾을 수 없음"

`qmd collection add <path> --name <name>`을 실행하여 디렉터리를 추가한 다음, `qmd embed`로 색인하세요.

### 임베딩 모델 재정의(CJK/다국어)

영어가 아닌 콘텐츠에는 `QMD_EMBED_MODEL` 환경 변수를 설정하세요.
```bash
export QMD_EMBED_MODEL="your-multilingual-model"
```

## 데이터 저장

- **색인 및 벡터:** `~/.cache/qmd/index.sqlite`
- **모델:** 첫 실행 시 로컬 캐시에 자동 다운로드
- **클라우드 의존성 없음** — 모든 기능이 로컬에서 실행됨

## 참고 자료

- [GitHub: tobi/qmd](https://github.com/tobi/qmd)
- [QMD 변경 로그](https://github.com/tobi/qmd/blob/main/CHANGELOG.md)
