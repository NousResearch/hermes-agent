---
title: "Gitnexus Explorer"
sidebar_label: "Gitnexus Explorer"
description: "使用 GitNexus 索引代码库，并通过 Web UI + Cloudflare 隧道提供交互式知识图谱"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Gitnexus Explorer

使用 GitNexus 索引代码库，并通过 Web UI + Cloudflare 隧道提供交互式知识图谱。

## 技能元数据

| | |
|---|---|
| 来源 | 可选技能 — 使用 `hermes skills install official/research/gitnexus-explorer` 安装 |
| 路径 | `optional-skills/research/gitnexus-explorer` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Teknium |
| 许可证 | MIT |
| 标签 | `gitnexus`, `code-intelligence`, `knowledge-graph`, `visualization` |
| 相关技能 | [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) |

## 参考：完整的 SKILL.md

:::info
以下是 Hermes 加载此技能时使用的完整技能定义。这是技能激活时智能体看到的指令。
:::

# GitNexus Explorer

将任何代码库索引到知识图谱中，并通过交互式 Web UI 提供服务，用于探索符号、调用链、聚类和执行流。通过 Cloudflare 隧道访问。

## 何时使用

- 用户想要直观地探索代码库的架构
- 用户请求代码库的知识图谱/依赖图
- 用户想要与他人分享交互式代码库浏览器

## 先决条件

- **Node.js** (v18+) — GitNexus 和代理必需
- **git** — 仓库必须有 `.git` 目录
- **cloudflared** — 用于隧道（缺失时自动安装到 ~/.local/bin）

## 大小警告

Web UI 在浏览器中渲染所有节点。约 5000 个文件以下的仓库运行良好。大型仓库（30k+ 节点）会变慢或导致浏览器崩溃。CLI/MCP 工具可以在任何规模下工作 — 只有 Web 可视化有此限制。

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

### 2. 修补 Web UI 以实现远程访问

Web UI 默认为 `localhost:4747` 进行 API 调用。修补它使用同源，这样它可以通过隧道/代理工作：

**文件：`$GITNEXUS_DIR/gitnexus-web/src/config/ui-constants.ts`**
修改：
```typescript
export const DEFAULT_BACKEND_URL = 'http://localhost:4747';
```
为：
```typescript
export const DEFAULT_BACKEND_URL = typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? window.location.origin : 'http://localhost:4747';
```

**文件：`$GITNEXUS_DIR/gitnexus-web/vite.config.ts`**
在 `server: { }` 块内添加 `allowedHosts: true`（仅在使用开发模式而非生产构建时需要）：
```typescript
server: {
    allowedHosts: true,
    // ... 现有配置
},
```

然后构建生产包：
```bash
cd "$GITNEXUS_DIR/gitnexus-web" && npx vite build
```

### 3. 索引目标仓库

```bash
cd /path/to/target-repo
npx gitnexus analyze --skip-agents-md
rm -rf .claude/    # 删除 Claude Code 特定工件
```

添加 `--embeddings` 以支持语义搜索（较慢 — 分钟而非秒）。

索引位于仓库内的 `.gitnexus/` 中（自动 gitignore）。

### 4. 创建代理脚本

将此写入文件（例如 `$GITNEXUS_DIR/proxy.mjs`）。它提供生产 Web UI 并将 `/api/*` 代理到 GitNexus 后端 — 同源、无 CORS 问题、无需 sudo、无需 nginx。

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
# 终端 1：GitNexus 后端 API
npx gitnexus serve &

# 终端 2：代理（Web UI + API 在一个端口上）
node "$GITNEXUS_DIR/proxy.mjs" "$GITNEXUS_DIR/gitnexus-web/dist" 8888 &
```

验证：`curl -s http://localhost:8888/api/repos` 应返回索引的仓库。

### 6. 使用 Cloudflare 隧道（可选 — 用于远程访问）

```bash
# 如需要安装 cloudflared（无需 sudo）
if ! command -v cloudflared &>/dev/null; then
  mkdir -p ~/.local/bin
  curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o ~/.local/bin/cloudflared
  chmod +x ~/.local/bin/cloudflared
  export PATH="$HOME/.local/bin:$PATH"
fi

# 启动隧道（--config /dev/null 避免与现有命名隧道配置冲突）
cloudflared tunnel --config /dev/null --url http://localhost:8888 --no-autoupdate --protocol http2
```

隧道 URL（例如 `https://random-words.trycloudflare.com`）打印到 stderr。
分享它 — 任何有链接的人都可以探索图谱。

### 7. 清理

```bash
# 停止服务
pkill -f "gitnexus serve"
pkill -f "proxy.mjs"
pkill -f cloudflared

# 从目标仓库中删除索引
cd /path/to/target-repo
npx gitnexus clean
rm -rf .claude/
```

## 陷阱

- **cloudflared 必须使用 `--config /dev/null`** 如果用户在 `~/.cloudflared/config.yml` 有现有命名隧道配置。没有它，配置中的 catch-all 入口规则会为所有快速隧道请求返回 404。

- **生产构建是隧道传输的强制要求。** Vite 开发服务器默认阻止非本地主机（`allowedHosts`）。生产构建 + Node 代理完全避免这个问题。

- **Web UI 不会创建 `.claude/` 或 `CLAUDE.md`。** 这些由 `npx gitnexus analyze` 创建。使用 `--skip-agents-md` 抑制 markdown 文件，然后用 `rm -rf .claude/` 删除其余。这些是 hermes-agent 用户不需要的 Claude Code 集成。

- **浏览器内存限制。** Web UI 将整个图谱加载到浏览器内存中。5000+ 文件的仓库可能会变慢。30000+ 文件可能会导致标签页崩溃。

- **嵌入是可选的。** `--embeddings` 启用语义搜索但在大型仓库上需要数分钟。跳过它进行快速探索；如果想要通过 AI 聊天面板进行自然语言查询则添加它。

- **多个仓库。** `gitnexus serve` 为所有索引的仓库提供服务。索引几个仓库，一次启动服务，Web UI 允许在它们之间切换。
