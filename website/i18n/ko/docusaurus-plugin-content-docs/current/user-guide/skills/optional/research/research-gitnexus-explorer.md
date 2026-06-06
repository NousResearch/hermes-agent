---
title: "Gitnexus Explorer"
sidebar_label: "Gitnexus Explorer"
description: "GitNexus로 코드베이스를 인덱싱하고 웹 UI + Cloudflare 터널을 통해 인터랙티브 지식 그래프를 제공합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Gitnexus Explorer

GitNexus로 코드베이스를 인덱싱하고 웹 UI + Cloudflare 터널을 통해 인터랙티브 지식 그래프를 제공합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/gitnexus-explorer` 명령으로 설치 |
| Path | `optional-skills/research/gitnexus-explorer` |
| Version | `1.0.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `gitnexus`, `code-intelligence`, `knowledge-graph`, `visualization` |
| Related skills | [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# GitNexus Explorer

모든 코드베이스를 지식 그래프로 인덱싱하고 기호, 호출 체인, 클러스터, 실행 흐름을 탐색할 수 있는 대화형 웹 UI를 제공합니다. 원격 액세스를 위해 Cloudflare를 통해 터널링됩니다.

## 사용 시기 (When to Use)

- 사용자가 코드베이스의 아키텍처를 시각적으로 탐색하고 싶어 할 때
- 사용자가 저장소의 지식 그래프 / 종속성 그래프를 요청할 때
- 사용자가 대화형 코드베이스 탐색기를 다른 사람과 공유하고 싶어 할 때

## 전제 조건 (Prerequisites)

- **Node.js** (v18+) — GitNexus와 프록시에 필요합니다
- **git** — 저장소에 `.git` 디렉토리가 있어야 합니다
- **cloudflared** — 터널링에 필요합니다 (누락된 경우 ~/.local/bin에 자동 설치됨)

## 크기 경고 (Size Warning)

웹 UI는 브라우저에서 모든 노드를 렌더링합니다. 약 5,000개 미만의 파일이 있는 저장소에서 잘 작동합니다. 대규모 저장소(30k+ 노드)의 경우 브라우저 탭이 느려지거나 다운될 수 있습니다. CLI/MCP 도구는 모든 규모에서 작동합니다 — 이 제한은 웹 시각화에만 적용됩니다.

## 단계 (Steps)

### 1. GitNexus 클론 및 빌드 (1회성 설정)

```bash
GITNEXUS_DIR="${GITNEXUS_DIR:-$HOME/.local/share/gitnexus}"

if [ ! -d "$GITNEXUS_DIR/gitnexus-web/dist" ]; then
  git clone https://github.com/abhigyanpatwari/GitNexus.git "$GITNEXUS_DIR"
  cd "$GITNEXUS_DIR/gitnexus-shared" && npm install && npm run build
  cd "$GITNEXUS_DIR/gitnexus-web" && npm install
fi
```

### 2. 원격 액세스를 위한 웹 UI 패치

웹 UI는 API 호출에 `localhost:4747`을 기본값으로 사용합니다. 터널/프록시를 통해 작동하도록 동일 출처(same-origin)를 사용하도록 패치합니다:

**파일: `$GITNEXUS_DIR/gitnexus-web/src/config/ui-constants.ts`**
변경 전:
```typescript
export const DEFAULT_BACKEND_URL = 'http://localhost:4747';
```
변경 후:
```typescript
export const DEFAULT_BACKEND_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.origin : 'http://localhost:4747';
```

**파일: `$GITNEXUS_DIR/gitnexus-web/vite.config.ts`**
`server: { }` 블록 내에 `allowedHosts: true`를 추가합니다 (프로덕션 빌드 대신 개발 모드를 실행할 때만 필요):
```typescript
server: {
    allowedHosts: true,
    // ... existing config
},
```

그런 다음 프로덕션 번들을 빌드합니다:
```bash
cd "$GITNEXUS_DIR/gitnexus-web" && npx vite build
```

### 3. 대상 저장소 인덱싱

```bash
cd /path/to/target-repo
npx gitnexus analyze --skip-agents-md
rm -rf .claude/    # Claude Code 전용 아티팩트 제거
```

시맨틱 검색을 위해 `--embeddings`를 추가할 수 있습니다 (더 느림 — 초 단위 대신 분 단위 소요).

인덱스는 저장소 내의 `.gitnexus/`에 저장됩니다 (auto-gitignored).

### 4. 프록시 스크립트 생성

다음 내용을 파일(예: `$GITNEXUS_DIR/proxy.mjs`)에 작성합니다. 이 스크립트는 프로덕션 웹 UI를 제공하고 `/api/*` 요청을 GitNexus 백엔드로 프록시합니다 — 동일 출처, CORS 문제 없음, sudo 필요 없음, nginx 필요 없음.

```javascript
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const API_PORT = parseInt(process.env.API_PORT || '4747');
const DIST_DIR = process.argv[2] || './dist';
const PORT = parseInt(process.argv[3] || '8888');

const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff',
  '.wasm': 'application/wasm',
};

function proxyToApi(req, res) {
  const opts = {
    hostname: '127.0.0.1', port: API_PORT,
    path: req.url, method: req.method, headers: req.headers,
  };
  const proxy = http.request(opts, (upstream) => {
    res.writeHead(upstream.statusCode, upstream.headers);
    upstream.pipe(res, { end: true });
  });
  proxy.on('error', () => { res.writeHead(502); res.end('Backend unavailable'); });
  req.pipe(proxy, { end: true });
}

function serveStatic(req, res) {
  let filePath = path.join(DIST_DIR, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  if (!fs.existsSync(filePath)) filePath = path.join(DIST_DIR, 'index.html');
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';
  try {
    const data = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'public, max-age=3600' });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
}

http.createServer((req, res) => {
  if (req.url.startsWith('/api')) proxyToApi(req, res);
  else serveStatic(req, res);
}).listen(PORT, () => console.log(`GitNexus proxy on http://localhost:${PORT}`));
```

### 5. 서비스 시작

```bash
# 터미널 1: GitNexus 백엔드 API
npx gitnexus serve &

# 터미널 2: 프록시 (웹 UI + API를 하나의 포트에서)
node "$GITNEXUS_DIR/proxy.mjs" "$GITNEXUS_DIR/gitnexus-web/dist" 8888 &
```

확인: `curl -s http://localhost:8888/api/repos`가 인덱싱된 저장소(들)를 반환해야 합니다.

### 6. Cloudflare를 통한 터널링 (선택 사항 — 원격 액세스용)

```bash
# 필요한 경우 cloudflared 설치 (sudo 불필요)
if ! command -v cloudflared &>/dev/null; then
  mkdir -p ~/.local/bin
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o ~/.local/bin/cloudflared
  chmod +x ~/.local/bin/cloudflared
  export PATH="$HOME/.local/bin:$PATH"
fi

# 터널 시작 (--config /dev/null은 기존 명명된 터널과의 충돌을 방지합니다)
cloudflared tunnel --config /dev/null --url http://localhost:8888 --no-autoupdate --protocol http2
```

터널 URL(예: `https://random-words.trycloudflare.com`)이 stderr에 인쇄됩니다.
이것을 공유하세요 — 링크가 있는 누구나 그래프를 탐색할 수 있습니다.

### 7. 정리 (Cleanup)

```bash
# 서비스 중지
pkill -f "gitnexus serve"
pkill -f "proxy.mjs"
pkill -f cloudflared

# 대상 저장소에서 인덱스 제거
cd /path/to/target-repo
npx gitnexus clean
rm -rf .claude/
```

## 주의사항 (Pitfalls)

- 사용자가 `~/.cloudflared/config.yml`에 기존 명명된 터널 설정을 가지고 있는 경우 **cloudflared를 위해 `--config /dev/null`이 필수입니다**. 이것이 없으면 설정의 포괄적(catch-all) 인그레스 규칙이 모든 퀵 터널 요청에 대해 404를 반환합니다.

- **터널링에는 프로덕션 빌드가 필수입니다.** Vite 개발 서버는 기본적으로 localhost가 아닌 호스트를 차단합니다(`allowedHosts`). 프로덕션 빌드 + Node 프록시는 이를 완전히 방지합니다.

- **웹 UI는 `.claude/` 또는 `CLAUDE.md`를 생성하지 않습니다.** 이것들은 `npx gitnexus analyze`에 의해 생성됩니다. 마크다운 파일을 생략하려면 `--skip-agents-md`를 사용하고, 나머지 파일들은 `rm -rf .claude/`로 삭제하세요. 이들은 hermes-agent 사용자에게는 필요 없는 Claude Code 연동 파일입니다.

- **브라우저 메모리 제한.** 웹 UI는 전체 그래프를 브라우저 메모리에 로드합니다. 5,000개 이상의 파일이 있는 저장소는 느려질 수 있습니다. 30,000개 이상의 파일은 탭을 다운시킬 가능성이 큽니다.

- **임베딩은 선택 사항입니다.** `--embeddings`는 시맨틱 검색을 활성화하지만 대규모 저장소에서는 분 단위가 소요됩니다. 빠른 탐색을 위해서는 생략하고, AI 채팅 패널을 통한 자연어 쿼리를 원할 경우 추가하세요.

- **다중 저장소.** `gitnexus serve`는 인덱싱된 "모든" 저장소를 제공합니다. 여러 저장소를 인덱싱한 후 serve를 한 번 시작하면, 웹 UI에서 이들 간에 전환할 수 있습니다.
