---
title: "Mcporter"
sidebar_label: "Mcporter"
description: "mcporter CLI를 사용하여 MCP 서버/도구들의 목록을 조회, 설정, 인증, 직접 호출(HTTP 또는 stdio)을 할 수 있으며 임시 서버 사용, 구성 편집, CLI/타입 생성 등이 가능합니다."
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Mcporter

mcporter CLI를 사용하여 MCP 서버/도구들을 조회, 구성, 인증 및 직접(HTTP 또는 stdio) 호출하세요. 임시 서버 실행, 구성 편집, CLI/타입 생성을 포함합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mcp/mcporter`로 설치 |
| 경로 | `optional-skills/mcp/mcporter` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `MCP`, `Tools`, `API`, `Integrations`, `Interop` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# mcporter

`mcporter`를 사용하여 터미널에서 직접 [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) 서버와 도구를 검색, 호출 및 관리할 수 있습니다.

## 필수 조건

Node.js가 필요합니다:
```bash
# 설치 없이 실행 (npx 사용)
npx mcporter list

# 또는 전역 설치
npm install -g mcporter
```

## 빠른 시작

```bash
# 현재 시스템에 이미 구성되어 있는 MCP 서버 목록 조회
mcporter list

# 스키마 정보를 포함하여 특정 서버의 도구 목록 조회
mcporter list <server> --schema

# 도구 호출
mcporter call <server.tool> key=value
```

## MCP 서버 검색

mcporter는 이 장비에 있는 다른 MCP 클라이언트들(Claude Desktop, Cursor 등)이 구성한 서버들을 자동 검색합니다. 사용할 새 서버를 찾으려면 [mcpfinder.dev](https://mcpfinder.dev) 또는 [mcp.so](https://mcp.so) 같은 레지스트리를 탐색한 후 임시(ad-hoc)로 연결할 수 있습니다:

```bash
# URL을 통해 임의의 MCP 서버에 연결 (구성 불필요)
mcporter list --http-url https://some-mcp-server.com --name my_server

# 또는 즉시(on the fly) stdio 서버 실행
mcporter list --stdio "npx -y @modelcontextprotocol/server-filesystem" --name fs
```

## 도구 호출

```bash
# Key=value 문법
mcporter call linear.list_issues team=ENG limit:5

# 함수 문법
mcporter call "linear.create_issue(title: \"Bug fix needed\")"

# 임시 HTTP 서버 사용 (구성 불필요)
mcporter call https://api.example.com/mcp.fetch url=https://example.com

# 임시 stdio 서버 사용
mcporter call --stdio "bun run ./server.ts" scrape url=https://example.com

# JSON 페이로드(payload)
mcporter call <server.tool> --args '{"limit": 5}'

# 기계가 읽기 쉬운 출력 형식 (Hermes에 권장)
mcporter call <server.tool> key=value --output json
```

## 인증과 설정 (Auth and Config)

```bash
# 서버 OAuth 로그인
mcporter auth <server | url> [--reset]

# 설정 관리
mcporter config list
mcporter config get <key>
mcporter config add <server>
mcporter config remove <server>
mcporter config import <path>
```

설정 파일 위치: `./config/mcporter.json` (`--config`로 덮어쓸 수 있습니다).

## 데몬 (Daemon)

지속적인 서버 연결을 위한 명령어:
```bash
mcporter daemon start
mcporter daemon status
mcporter daemon stop
mcporter daemon restart
```

## 코드 생성

```bash
# MCP 서버에 대한 CLI 래퍼(wrapper) 생성
mcporter generate-cli --server <name>
mcporter generate-cli --command <url>

# 생성된 CLI 검사
mcporter inspect-cli <path> [--json]

# TypeScript 타입/클라이언트(client) 생성
mcporter emit-ts <server> --mode client
mcporter emit-ts <server> --mode types
```

## 참고사항

- 더욱 파싱하기 쉬운 구조화된 출력을 원한다면 `--output json` 옵션을 사용하세요.
- 임시 서버 (HTTP URL 또는 `--stdio` 명령 사용)는 별다른 구성 없이 바로 동작합니다 — 단발성 호출에 유용합니다.
- OAuth 인증은 브라우저를 통한 대화형 흐름이 필요할 수 있습니다 — 필요한 경우 `terminal(command="mcporter auth <server>", pty=true)` 형식을 사용하세요.
