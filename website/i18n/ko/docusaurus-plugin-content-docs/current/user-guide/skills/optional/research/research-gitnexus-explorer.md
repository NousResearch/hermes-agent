---
title: "Gitnexus Explorer — 대화형 코드베이스 지식 그래프 웹 UI 제공"
sidebar_label: "Gitnexus Explorer"
description: "대화형 코드베이스 지식 그래프 웹 UI 제공"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Gitnexus Explorer

코드베이스 지식 그래프를 탐색할 수 있는 대화형 웹 UI를 제공합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/gitnexus-explorer`로 설치 |
| 경로 | `optional-skills/research/gitnexus-explorer` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `gitnexus`, `code-intelligence`, `knowledge-graph`, `visualization` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# GitNexus Explorer

모든 코드베이스를 지식 그래프로 색인하고 심볼, 호출 체인, 클러스터 및 실행 흐름을 탐색할 수 있는 대화형 웹 UI를 제공합니다. 원격 액세스를 위해 Cloudflare를 통한 터널을 사용합니다.

## 사용 시점

- 사용자가 코드베이스 아키텍처를 시각적으로 탐색하려는 경우
- 사용자가 저장소의 지식 그래프/의존성 그래프를 요청하는 경우
- 사용자가 다른 사람과 대화형 코드베이스 탐색기를 공유하려는 경우

## 사전 요구 사항

- **Node.js** (v18+) — GitNexus와 프록시에 필요
- **git** — 저장소에 `.git` 디렉터리가 있어야 함
- **cloudflared** — 터널링에 필요(없으면 `~/.local/bin`에 자동 설치)

## 크기 경고

웹 UI는 브라우저에서 모든 노드를 렌더링합니다. 약 5,000개 미만의 파일을 가진 저장소는 원활하게 작동합니다. 대규모 저장소(30k+ 노드)는 브라우저 탭이 느려지거나 충돌할 수 있습니다. CLI/MCP 도구는 어떤 규모에서도 작동하며, 이 제한은 웹 시각화에만 적용됩니다.

## 단계

### 1. GitNexus 복제 및 빌드(최초 한 번 설정)

```bash
GITNEXUS_DIR="${GITNEXUS_DIR:-$HOME/.local/share/gitnexus}"

if [ ! -d "$GITNEXUS_DIR/gitnexus-web/dist" ]; then
  git clone https://github.com/abhigyanpatwari/GitNexus.git "$GITNEXUS_DIR"
  cd "$GITNEXUS_DIR/gitnexus-shared" && npm install && npm run build
  cd "$GITNEXUS_DIR/gitnexus-web" && npm install
fi
```

### 2. 원격 액세스를 위한 웹 UI 패치

웹 UI는 API 호출에 기본적으로 `localhost:4747`을 사용합니다. 터널/프록시를 통해 작동하도록 동일 출처를 사용하게 패치합니다.

**파일: `$GITNEXUS_DIR/gitnexus-web/src/config/ui-constants.ts`**
다음을 변경합니다:
```typescript
export const DEFAULT_BACKEND_URL = 'http://localhost:4747';
```
다음으로 변경합니다:
```typescript
export const DEFAULT_BACKEND_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.origin : 'http://localhost:4747';
```

**파일: `$GITNEXUS_DIR/gitnexus-web/vite.config.ts`**
`server: { }` 블록 안에 `allowedHosts: true`를 추가합니다(프로덕션 빌드 대신 개발 모드로 실행할 때만 필요).
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

### 3. 대상 저장소 색인

```bash
cd /path/to/target-repo
npx gitnexus analyze --skip-agents-md
rm -rf .claude/    # remove Claude Code-specific artifacts
```

의미 검색을 사용하려면 `--embeddings`를 추가합니다(더 느림 — 수 초가 아닌 수 분 소요).

색인은 저장소 내부의 `.gitnexus/`에 저장됩니다(자동으로 git에서 무시됨).

### 4. 프록시 스크립트 생성

이를 파일(예: `$GITNEXUS_DIR/proxy.mjs`)에 작성합니다. 프로덕션 웹 UI를 제공하고 `/api/*`를 GitNexus 백엔드로 프록시합니다 — 동일 출처이므로 CORS 문제가 없고, sudo나 nginx도 필요하지 않습니다.

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
# Terminal 1: GitNexus backend API
npx gitnexus serve &

# Terminal 2: Proxy (web UI + API on one port)
node "$GITNEXUS_DIR/proxy.mjs" "$GITNEXUS_DIR/gitnexus-web/dist" 8888 &
```

확인: `curl -s http://localhost:8888/api/repos`는 색인된 저장소를 반환해야 합니다.

### 6. Cloudflare로 터널링(선택 사항 — 원격 액세스용)

```bash
# Install cloudflared if needed (no sudo)
if ! command -v cloudflared &>/dev/null; then
  mkdir -p ~/.local/bin
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o ~/.local/bin/cloudflared
  chmod +x ~/.local/bin/cloudflared
  export PATH="$HOME/.local/bin:$PATH"
fi

# Start tunnel (--config /dev/null avoids conflicts with existing named tunnels)
cloudflared tunnel --config /dev/null --url http://localhost:8888 --no-autoupdate --protocol http2
```

터널 URL(예: `https://random-words.trycloudflare.com`)은 stderr에 출력됩니다. 링크가 있는 누구나 그래프를 탐색할 수 있도록 공유하세요.

### 7. 정리

```bash
# Stop services
pkill -f "gitnexus serve"
pkill -f "proxy.mjs"
pkill -f cloudflared

# Remove index from the target repo
cd /path/to/target-repo
npx gitnexus clean
rm -rf .claude/
```

## 문제점 및 주의 사항

- 사용자가 기존의 이름 있는 터널 설정을 `~/.cloudflared/config.yml`에 가지고 있다면 cloudflared에 `--config /dev/null`이 필요합니다. 이 옵션이 없으면 설정의 포괄 ingress 규칙이 모든 빠른 터널 요청에 404를 반환합니다.

- 터널링에는 프로덕션 빌드가 필수입니다. Vite 개발 서버는 기본적으로 로컬호스트가 아닌 호스트를 차단합니다(`allowedHosts`). 프로덕션 빌드와 Node 프록시를 사용하면 이 문제를 완전히 피할 수 있습니다.

- 웹 UI는 `.claude/`나 `CLAUDE.md`를 만들지 않습니다. `npx gitnexus analyze`가 이를 생성합니다. 마크다운 파일을 억제하려면 `--skip-agents-md`를 사용하고, 나머지는 `rm -rf .claude/`로 제거합니다. 이는 hermes-agent 사용자에게 필요하지 않은 Claude Code 통합입니다.

- 브라우저 메모리 제한. 웹 UI는 전체 그래프를 브라우저 메모리에 로드합니다. 5k+ 파일의 저장소는 느려질 수 있습니다. 30k+ 파일은 탭이 충돌할 가능성이 높습니다.

- 임베딩은 선택 사항입니다. `--embeddings`는 의미 검색을 활성화하지만 대규모 저장소에서는 수 분이 걸립니다. 빠르게 탐색하려면 생략하고, AI 채팅 패널에서 자연어 쿼리를 사용하려면 추가합니다.

- 여러 저장소. `gitnexus serve`는 색인된 모든 저장소를 제공합니다. 여러 저장소를 색인한 뒤 serve를 한 번 시작하면 웹 UI에서 저장소를 전환할 수 있습니다.
