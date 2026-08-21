---
title: "Mcporter — 터미널에서 MCP 서버/도구 나열, 인증, 호출"
sidebar_label: "Mcporter"
description: "터미널에서 MCP 서버/도구를 나열하고, 인증하고, 호출"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Mcporter

터미널에서 MCP 서버와 도구를 나열하고, 인증하고, 호출합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mcp/mcporter`로 설치 |
| 경로 | `optional-skills/mcp/mcporter` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `MCP`, `Tools`, `API`, `Integrations`, `Interop` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 불러오는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# mcporter

`mcporter`를 사용해 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 서버와 도구를 터미널에서 직접 검색하고, 호출하고, 관리합니다.

## 사전 요구 사항

Node.js가 필요합니다:
```bash
# No install needed (runs via npx)
npx mcporter list

# Or install globally
npm install -g mcporter
```

## 빠른 시작

```bash
# List MCP servers already configured on this machine
mcporter list

# List tools for a specific server with schema details
mcporter list <server> --schema

# Call a tool
mcporter call <server.tool> key=value
```

## MCP 서버 검색

mcporter는 컴퓨터에 있는 다른 MCP 클라이언트(Claude Desktop, Cursor 등)가 구성한 서버를 자동으로 검색합니다. 사용할 새 서버를 찾으려면 [mcpfinder.dev](https://mcpfinder.dev) 또는 [mcp.so](https://mcp.so)와 같은 레지스트리를 살펴본 다음 임시로 연결하세요:

```bash
# Connect to any MCP server by URL (no config needed)
mcporter list --http-url https://some-mcp-server.com --name my_server

# Or run a stdio server on the fly
mcporter list --stdio "npx -y @modelcontextprotocol/server-filesystem" --name fs
```

## 도구 호출

```bash
# Key=value syntax
mcporter call linear.list_issues team=ENG limit:5

# Function syntax
mcporter call "linear.create_issue(title: \"Bug fix needed\")"

# Ad-hoc HTTP server (no config needed)
mcporter call https://api.example.com/mcp.fetch url=https://example.com

# Ad-hoc stdio server
mcporter call --stdio "bun run ./server.ts" scrape url=https://example.com

# JSON payload
mcporter call <server.tool> --args '{"limit": 5}'

# Machine-readable output (recommended for Hermes)
mcporter call <server.tool> key=value --output json
```

## 인증 및 구성

```bash
# OAuth login for a server
mcporter auth <server | url> [--reset]

# Manage config
mcporter config list
mcporter config get <key>
mcporter config add <server>
mcporter config remove <server>
mcporter config import <path>
```

구성 파일 위치: `./config/mcporter.json` (`--config`으로 재정의).

## 데몬

지속적인 서버 연결을 사용하려면:
```bash
mcporter daemon start
mcporter daemon status
mcporter daemon stop
mcporter daemon restart
```

## 코드 생성

```bash
# Generate a CLI wrapper for an MCP server
mcporter generate-cli --server <name>
mcporter generate-cli --command <url>

# Inspect a generated CLI
mcporter inspect-cli <path> [--json]

# Generate TypeScript types/client
mcporter emit-ts <server> --mode client
mcporter emit-ts <server> --mode types
```

## 참고

- 구조화된 출력으로 더 쉽게 분석하려면 `--output json`을 사용하세요.
- 임시 서버(HTTP URL 또는 `--stdio` 명령)는 구성 없이 작동하므로 일회성 호출에 유용합니다.
- OAuth 인증에는 대화형 브라우저 흐름이 필요할 수 있습니다 — 필요한 경우 `terminal(command="mcporter auth <server>", pty=true)`를 사용하세요.
