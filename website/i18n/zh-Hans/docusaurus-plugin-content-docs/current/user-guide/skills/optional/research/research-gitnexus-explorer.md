---
title: "Gitnexus Explorer"
sidebar_label: "Gitnexus Explorer"
description: "使用 GitNexus 为代码库建立索引，并通过 Web UI + Cloudflare 隧道提供交互式知识图谱服务"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Gitnexus Explorer

使用 GitNexus 为代码库建立索引，并通过 Web UI + Cloudflare 隧道提供交互式知识图谱服务。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/research/gitnexus-explorer` 安装 |
| 路径 | `optional-skills/research/gitnexus-explorer` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `gitnexus`, `code-intelligence`, `knowledge-graph`, `visualization` |
| 相关 skill | [`hermes-agent`](/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`codebase-inspection`](/user-guide/skills/bundled/github/github-codebase-inspection) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# GitNexus Explorer

将任意代码库索引为知识图谱，并提供交互式 Web UI，用于探索符号、调用链、聚类和执行流。通过 Cloudflare 隧道实现远程访问。

## 适用场景

- 用户希望可视化探索代码库架构
- 用户请求生成某个仓库的知识图谱/依赖图
- 用户希望与他人共享交互式代码库浏览器

## 前置条件

- **Node.js**（v18+）— GitNexus 和代理所需
- **git** — 仓库必须包含 `.git` 目录
- **cloudflared** — 用于隧道（如缺失，自动安装至 `~/.local/bin`）

## 规模警告

Web UI 在浏览器中渲染所有节点。文件数不超过约 5,000 的仓库运行良好。大型仓库（30k+ 节点）会导致浏览器标签页卡顿或崩溃。CLI/MCP 工具在任何规模下均可正常工作——仅 Web 可视化存在此限制。

## 步骤

### 1. 克隆并构建 GitNexus（一次性设置）

```bash
GITNEXUS_DIR="${GITNEXUS_DIR:-$HOME/.local/share/gitnexus}"

if [ ! -d "$GITNEXUS_DIR/gitnexus-web/dist" ]; then
  git clone https://github.com/abhigyanpatwari/GitNexus.git "$GITNEXUS_DIR"
  cd "$GITNEXUS_DIR/gitnexus-shared" && npm install && npm run build
  cd "$GITNEXUS_DIR/gitnexus-web" && npm install
fi
```

### 2. 为远程访问修补 Web UI

Web UI 默认使用 `localhost:4747` 进行 API 调用。将其修补为使用同源地址，以便通过隧道/代理正常工作：

**文件：`$GITNEXUS_DIR/gitnexus-web/src/config/ui-constants.ts`**
将：
```typescript
export const DEFAULT_BACKEND_URL = 'http://localhost:4747';
```
改为：
```typescript
export const DEFAULT_BACKEND_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.origin : 'http://localhost:4747';
```

**文件：`$GITNEXUS_DIR/gitnexus-web/vite.config.ts`**
在 `server: { }` 块内添加 `allowedHosts: true`（仅在使用开发模式而非生产构建时需要）：
```typescript
server: {
    allowedHosts: true,
    // ... existing config
},
```

然后构建生产包：
```bash
cd "$GITNEXUS_DIR/gitnexus-web" && npx vite build
```

### 3. 为目标仓库建立索引

```bash
cd /path/to/target-repo
npx gitnexus analyze --skip-agents-md
rm -rf .claude/    # remove Claude Code-specific artifacts
```

添加 `--embeddings` 可启用语义搜索（速度较慢——需要数分钟而非数秒）。

索引存储在仓库内的 `.gitnexus/` 目录中（已自动加入 `.gitignore`）。

### 4. 创建代理脚本

将以下内容写入文件（例如 `$GITNEXUS_DIR/proxy.mjs`）。它提供生产 Web UI 服务，并将 `/api/*` 代理至 GitNexus 后端——同源，无 CORS 问题，无需 sudo，无需 nginx。

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

### 5. 启动服务

```bash
# Terminal 1: GitNexus backend API
npx gitnexus serve &

# Terminal 2: Proxy (web UI + API on one port)
node "$GITNEXUS_DIR/proxy.mjs" "$GITNEXUS_DIR/gitnexus-web/dist" 8888 &
```

验证：`curl -s http://localhost:8888/api/repos` 应返回已索引的仓库。

### 6. 通过 Cloudflare 建立隧道（可选——用于远程访问）

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

隧道 URL（例如 `https://random-words.trycloudflare.com`）将输出至 stderr。分享该链接——任何拥有链接的人均可探索图谱。

### 7. 清理

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

## 注意事项

- **`cloudflared` 必须使用 `--config /dev/null`**：若用户在 `~/.cloudflared/config.yml` 中存在已命名的隧道配置，则不加此参数时，配置中的兜底 ingress 规则会对所有快速隧道请求返回 404。

- **隧道必须使用生产构建。** Vite 开发服务器默认阻止非 localhost 主机（`allowedHosts`）。使用生产构建 + Node 代理可完全规避此问题。

- **Web UI 不会创建 `.claude/` 或 `CLAUDE.md`。** 这些文件由 `npx gitnexus analyze` 创建。使用 `--skip-agents-md` 可抑制 markdown 文件的生成，再用 `rm -rf .claude/` 清除其余内容。这些是 Claude Code 集成产物，Hermes Agent 用户无需使用。

- **浏览器内存限制。** Web UI 将整个图谱加载至浏览器内存。文件数超过 5k 的仓库可能出现卡顿，超过 30k 文件的仓库很可能导致标签页崩溃。

- **Embedding（嵌入）为可选项。** `--embeddings` 可启用语义搜索，但在大型仓库上需要数分钟。如需快速探索可跳过；若希望通过 AI 对话面板进行自然语言查询，则可添加此选项。

- **多仓库支持。** `gitnexus serve` 会服务所有已索引的仓库。可先为多个仓库建立索引，再启动一次 serve，Web UI 支持在各仓库间切换。