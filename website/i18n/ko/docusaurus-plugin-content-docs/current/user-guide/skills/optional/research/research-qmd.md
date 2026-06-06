---
title: "Qmd"
sidebar_label: "Qmd"
description: "BM25, 벡터 검색, LLM 리랭킹(reranking)을 갖춘 하이브리드 검색 엔진인 qmd를 사용하여 로컬에서 개인 지식 베이스, 노트, 문서 및 회의록을 검색합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Qmd

BM25, 벡터 검색 및 LLM 리랭킹을 갖춘 하이브리드 검색 엔진인 qmd를 사용하여 개인 지식 베이스, 노트, 문서 및 회의록을 로컬에서 검색합니다. CLI 및 MCP 통합을 지원합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/qmd` 명령으로 설치 |
| Path | `optional-skills/research/qmd` |
| Version | `1.0.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Platforms | macos, linux |
| Tags | `Search`, `Knowledge-Base`, `RAG`, `Notes`, `MCP`, `Local-AI` |
| Related skills | [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian), [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# QMD — Query Markup Documents

개인 지식 베이스를 위한 로컬 온디바이스(on-device) 검색 엔진입니다. 마크다운 노트, 회의록, 문서 및 기타 텍스트 기반 파일을 인덱싱한 다음 키워드 일치, 의미적(semantic) 이해, LLM 기반 리랭킹(reranking)을 결합한 하이브리드 검색을 제공합니다 — 이 모든 것이 클라우드 종속성 없이 로컬에서 실행됩니다.

[Tobi Lütke](https://github.com/tobi/qmd)가 제작했습니다. MIT 라이선스를 따릅니다.

## 사용 시기 (When to Use)

- 사용자가 노트, 문서, 지식 베이스 또는 회의록을 검색해 달라고 요청할 때
- 대량의 마크다운/텍스트 파일 모음에서 무언가를 찾고자 할 때
- 단순한 키워드 grep이 아닌 의미적(semantic) 검색("X 개념에 대한 노트 찾기")을 원할 때
- 사용자가 이미 qmd 컬렉션을 설정했고, 그것들을 쿼리하고 싶어 할 때
- 사용자가 로컬 지식 베이스나 문서 검색 시스템 설정을 요청할 때
- 키워드: "search my notes", "find in my docs", "knowledge base", "qmd"

## 전제 조건 (Prerequisites)

### Node.js >= 22 (필수)

```bash
# 버전 확인
node --version  # 22 이상이어야 함

# macOS — Homebrew를 통한 설치 또는 업그레이드
brew install node@22

# Linux — NodeSource 또는 nvm 사용
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
# 또는 nvm으로:
nvm install 22 && nvm use 22
```

### 확장(Extension) 지원이 포함된 SQLite (macOS 전용)

macOS 시스템에 내장된 SQLite에는 확장 로딩 기능이 없습니다. Homebrew로 설치하세요:

```bash
brew install sqlite
```

### qmd 설치

```bash
npm install -g @tobilu/qmd
# 또는 Bun 사용:
bun install -g @tobilu/qmd
```

처음 실행 시 3개의 로컬 GGUF 모델(총 약 2GB)을 자동 다운로드합니다:

| Model | Purpose | Size |
|-------|---------|------|
| embeddinggemma-300M-Q8_0 | 벡터 임베딩 (Vector embeddings) | ~300MB |
| qwen3-reranker-0.6b-q8_0 | 결과 리랭킹 (Result reranking) | ~640MB |
| qmd-query-expansion-1.7B | 쿼리 확장 (Query expansion) | ~1.1GB |

### 설치 확인 (Verify Installation)

```bash
qmd --version
qmd status
```

## 빠른 참조 (Quick Reference)

| Command | What It Does | Speed |
|---------|-------------|-------|
| `qmd search "query"` | BM25 키워드 검색 (모델 없음) | ~0.2s |
| `qmd vsearch "query"` | 의미적 벡터 검색 (1개 모델) | ~3s |
| `qmd query "query"` | 하이브리드 + 리랭킹 (3개 모델 모두) | 웜(warm) 상태 ~2-3s, 콜드(cold) 상태 ~19s |
| `qmd get <docid>` | 전체 문서 내용 가져오기 | 즉시(instant) |
| `qmd multi-get "glob"` | 여러 파일 가져오기 | 즉시(instant) |
| `qmd collection add <path> --name <n>` | 디렉토리를 컬렉션으로 추가 | 즉시(instant) |
| `qmd context add <path> "description"` | 검색 개선을 위한 컨텍스트 메타데이터 추가 | 즉시(instant) |
| `qmd embed` | 벡터 임베딩 생성/업데이트 | 다름(varies) |
| `qmd status` | 인덱스 상태 및 컬렉션 정보 표시 | 즉시(instant) |
| `qmd mcp` | MCP 서버 시작 (stdio) | 지속(persistent) |
| `qmd mcp --http --daemon` | MCP 서버 시작 (HTTP, warm 모델) | 지속(persistent) |

## 설정 워크플로우 (Setup Workflow)

### 1. 컬렉션 추가 (Add Collections)

문서가 있는 디렉토리를 qmd에 지정합니다:

```bash
# 노트 디렉토리 추가
qmd collection add ~/notes --name notes

# 프로젝트 문서 추가
qmd collection add ~/projects/myproject/docs --name project-docs

# 회의록 추가
qmd collection add ~/meetings --name meetings

# 모든 컬렉션 나열
qmd collection list
```

### 2. 컨텍스트 설명 추가 (Add Context Descriptions)

컨텍스트 메타데이터는 검색 엔진이 각 컬렉션에 어떤 내용이 들어있는지 이해하는 데 도움을 줍니다. 이는 검색 품질을 크게 향상시킵니다:

```bash
qmd context add qmd://notes "Personal notes, ideas, and journal entries"
qmd context add qmd://project-docs "Technical documentation for the main project"
qmd context add qmd://meetings "Meeting transcripts and action items from team syncs"
```

### 3. 임베딩 생성 (Generate Embeddings)

```bash
qmd embed
```

이 명령어는 모든 컬렉션의 모든 문서를 처리하고 벡터 임베딩을 생성합니다. 새 문서나 컬렉션을 추가한 후 다시 실행하세요.

### 4. 확인 (Verify)

```bash
qmd status   # 인덱스 상태, 컬렉션 통계, 모델 정보 표시
```

## 검색 패턴 (Search Patterns)

### 빠른 키워드 검색 (Fast Keyword Search - BM25)

가장 적합한 대상: 정확한 용어, 코드 식별자, 이름, 알려진 구절.
로드되는 모델이 없으므로 거의 즉각적인 결과가 나옵니다.

```bash
qmd search "authentication middleware"
qmd search "handleError async"
```

### 의미적 벡터 검색 (Semantic Vector Search)

가장 적합한 대상: 자연어 질문, 개념적 쿼리.
임베딩 모델을 로드합니다 (첫 쿼리 시 ~3초).

```bash
qmd vsearch "how does the rate limiter handle burst traffic"
qmd vsearch "ideas for improving onboarding flow"
```

### 리랭킹을 포함한 하이브리드 검색 (가장 높은 품질 - Hybrid Search)

가장 적합한 대상: 품질이 가장 중요한 쿼리.
3개 모델(쿼리 확장, 병렬 BM25+벡터, 리랭킹)을 모두 사용합니다.

```bash
qmd query "what decisions were made about the database migration"
```

### 구조화된 다중 모드 쿼리 (Structured Multi-Mode Queries)

정밀도를 높이기 위해 단일 쿼리에서 여러 검색 유형을 결합합니다:

```bash
# 정확한 용어를 위한 BM25 + 개념을 위한 벡터 결합
qmd query $'lex: rate limiter\nvec: how does throttling work under load'

# 쿼리 확장 포함
qmd query $'expand: database migration plan\nlex: "schema change"'
```

### 쿼리 구문 (Query Syntax - lex/BM25 mode)

| Syntax | Effect | Example |
|--------|--------|---------|
| `term` | 접두사(Prefix) 매치 | `perf`는 "performance"와 매치됨 |
| `"phrase"` | 정확한 구절(Exact phrase) | `"rate limiter"` |
| `-term` | 용어 제외 | `performance -sports` |

### HyDE (Hypothetical Document Embeddings - 가상 문서 임베딩)

복잡한 주제의 경우, 답변이 어떤 모습일지 예상하여 작성해 보세요:

```bash
qmd query $'hyde: The migration plan involves three phases. First, we add the new columns without dropping the old ones. Then we backfill data. Finally we cut over and remove legacy columns.'
```

### 컬렉션으로 범위 제한 (Scoping to Collections)

```bash
qmd search "query" --collection notes
qmd query "query" --collection project-docs
```

### 출력 형식 (Output Formats)

```bash
qmd search "query" --json        # JSON 출력 (파싱에 가장 좋음)
qmd search "query" --limit 5     # 결과 제한
qmd get "#abc123"                # 문서 ID로 가져오기
qmd get "path/to/file.md"       # 파일 경로로 가져오기
qmd get "file.md:50" -l 100     # 특정 줄 범위 가져오기
qmd multi-get "journals/*.md" --json  # glob 패턴으로 일괄 가져오기
```

## MCP 연동 (권장 - MCP Integration)

qmd는 네이티브 MCP 클라이언트를 통해 Hermes Agent에 직접 검색 도구를 제공하는 MCP 서버를 노출합니다. 이 방법이 권장되며, 한 번 설정되면 이 스킬을 로드하지 않고도 에이전트가 qmd 도구들을 자동으로 얻게 됩니다.

### 옵션 A: Stdio 모드 (간단함)

`~/.hermes/config.yaml`에 다음을 추가하세요:

```yaml
mcp_servers:
  qmd:
    command: "qmd"
    args: ["mcp"]
    timeout: 30
    connect_timeout: 45
```

이것은 다음 도구들을 등록합니다: `mcp_qmd_search`, `mcp_qmd_vsearch`, `mcp_qmd_deep_search`, `mcp_qmd_get`, `mcp_qmd_status`.

**트레이드오프:** 첫 검색 호출 시 모델이 로드되며(~19초 콜드 스타트), 이후 세션 동안 메모리에 유지됩니다. 가끔 사용하는 용도로 적합합니다.

### 옵션 B: HTTP 데몬 모드 (빠름, 헤비 유저 권장)

qmd 데몬을 별도로 시작합니다 — 이 데몬은 모델들을 메모리에 따뜻한(warm) 상태로 유지합니다:

```bash
# 데몬 시작 (에이전트 재시작 간에도 유지됨)
qmd mcp --http --daemon

# 기본적으로 http://localhost:8181에서 실행됨
```

그런 다음 Hermes Agent가 HTTP를 통해 연결되도록 설정합니다:

```yaml
mcp_servers:
  qmd:
    url: "http://localhost:8181/mcp"
    timeout: 30
```

**트레이드오프:** 실행하는 동안 약 2GB의 RAM을 사용하지만 모든 쿼리가 빠릅니다(~2-3초). 검색을 자주 하는 사용자에게 가장 좋습니다.

### 데몬을 계속 실행 상태로 유지하기 (Keeping the Daemon Running)

#### macOS (launchd)

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

#### Linux (systemd user service)

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

### MCP 도구 참조 (MCP Tools Reference)

연결되면 이 도구들은 `mcp_qmd_*` 형태로 사용 가능합니다:

| MCP Tool | Maps To | Description |
|----------|---------|-------------|
| `mcp_qmd_search` | `qmd search` | BM25 키워드 검색 |
| `mcp_qmd_vsearch` | `qmd vsearch` | 의미적 벡터 검색 |
| `mcp_qmd_deep_search` | `qmd query` | 하이브리드 검색 + 리랭킹 |
| `mcp_qmd_get` | `qmd get` | ID 또는 경로로 문서 가져오기 |
| `mcp_qmd_status` | `qmd status` | 인덱스 상태 및 통계 |

MCP 도구는 다중 모드 검색을 위해 구조화된 JSON 쿼리를 허용합니다:

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

## CLI 사용법 (MCP 없이 사용 시)

MCP가 구성되지 않은 경우 터미널을 통해 직접 qmd를 사용하세요:

```
terminal(command="qmd query 'what was decided about the API redesign' --json", timeout=30)
```

설정 및 관리 작업의 경우 항상 터미널을 사용하세요:

```
terminal(command="qmd collection add ~/Documents/notes --name notes")
terminal(command="qmd context add qmd://notes 'Personal research notes and ideas'")
terminal(command="qmd embed")
terminal(command="qmd status")
```

## 검색 파이프라인 작동 원리 (How the Search Pipeline Works)

내부 구조를 이해하면 적절한 검색 모드를 선택하는 데 도움이 됩니다:

1. **쿼리 확장(Query Expansion)** — 미세 조정(fine-tuned)된 1.7B 모델이 2개의 대안 쿼리를 생성합니다. 원래 쿼리는 퓨전(fusion) 시 2배의 가중치를 갖습니다.
2. **병렬 검색(Parallel Retrieval)** — 모든 쿼리 변형에 대해 BM25(SQLite FTS5)와 벡터 검색이 동시에 실행됩니다.
3. **RRF 퓨전(RRF Fusion)** — Reciprocal Rank Fusion(k=60)으로 결과를 병합합니다. 상위 순위 보너스: 1위는 +0.05, 2-3위는 +0.02의 가중치를 얻습니다.
4. **LLM 리랭킹(LLM Reranking)** — qwen3-reranker가 상위 30개 후보의 점수(0.0-1.0)를 매깁니다.
5. **위치 인식 블렌딩(Position-Aware Blending)** — 1~3순위: 검색(retrieval) 75% / 리랭커(reranker) 25%. 4~10순위: 60/40. 11순위 이상: 40/60 (롱테일(long tail)일수록 리랭커를 더 신뢰함).

**스마트 청킹(Smart Chunking):** 문서는 15%의 겹침과 함께 약 900 토큰을 목표로 하여 자연스러운 끊어짐 지점(제목, 코드 블록, 빈 줄)에서 분할됩니다. 코드 블록은 중간에 분할되지 않습니다.

## 모범 사례 (Best Practices)

1. **항상 컨텍스트 설명을 추가하세요** — `qmd context add`는 검색 정확도를 극적으로 향상시킵니다. 각 컬렉션에 어떤 내용이 들어있는지 설명하세요.
2. **문서 추가 후 다시 임베딩하세요** — 새 파일이 컬렉션에 추가되면 `qmd embed`를 반드시 다시 실행해야 합니다.
3. **속도를 위해서는 `qmd search`를 사용하세요** — 빠른 키워드 찾기(코드 식별자, 정확한 이름)가 필요할 때 BM25는 즉각적이며 모델이 필요하지 않습니다.
4. **품질을 위해서는 `qmd query`를 사용하세요** — 질문이 개념적이거나 사용자가 최상의 결과를 원할 때는 하이브리드 검색을 사용하세요.
5. **MCP 통합을 선호하세요** — 한 번 구성되면 에이전트가 매번 이 스킬을 로드할 필요 없이 기본 도구로 사용할 수 있습니다.
6. **검색을 자주 하는 사용자는 데몬 모드** — 사용자가 지식 베이스를 정기적으로 검색하는 경우 HTTP 데몬 설정을 권장하세요.
7. **구조화된 검색의 첫 쿼리는 2배의 가중치를 갖습니다** — lex(키워드)와 vec(벡터)를 결합할 때 가장 중요하거나 확실한 쿼리를 첫 번째에 넣으세요.

## 문제 해결 (Troubleshooting)

### "첫 실행 시 모델 다운로드 중 (Models downloading on first run)"
정상입니다 — qmd는 처음 사용할 때 약 2GB의 GGUF 모델을 자동으로 다운로드합니다. 이 작업은 1회성입니다.

### 콜드 스타트 지연 (~19초)
모델이 메모리에 로드되지 않았을 때 발생합니다. 해결책:
- HTTP 데몬 모드(`qmd mcp --http --daemon`)를 사용하여 항상 웜(warm) 상태를 유지합니다.
- 모델이 필요 없을 때는 `qmd search` (BM25 전용)를 사용합니다.
- MCP stdio 모드는 첫 번째 검색 시 모델을 로드하고 세션 동안 웜 상태를 유지합니다.

### macOS: "unable to load extension"
Homebrew SQLite를 설치합니다: `brew install sqlite`
그 후 이 설치 경로가 시스템 SQLite보다 먼저 PATH에 오도록 설정합니다.

### "No collections found"
`qmd collection add <path> --name <name>`을 실행하여 디렉토리를 추가한 후 `qmd embed`를 실행하여 인덱싱합니다.

### 임베딩 모델 재정의 (CJK/다국어 지원)
영어가 아닌 콘텐츠를 다룰 경우 `QMD_EMBED_MODEL` 환경 변수를 설정하세요:
```bash
export QMD_EMBED_MODEL="your-multilingual-model"
```

## 데이터 저장소 (Data Storage)

- **인덱스 및 벡터:** `~/.cache/qmd/index.sqlite`
- **모델:** 첫 실행 시 로컬 캐시에 자동 다운로드됨
- **클라우드 종속성 없음** — 모든 것이 로컬에서 실행됨

## 참고 자료 (References)

- [GitHub: tobi/qmd](https://github.com/tobi/qmd)
- [QMD Changelog](https://github.com/tobi/qmd/blob/main/CHANGELOG.md)
