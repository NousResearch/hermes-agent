---
sidebar_position: 6
title: "MCP (Model Context Protocol)"
description: "표준화된 MCP 서버를 통해 새로운 도구, 리소스 및 기능을 에이전트와 통합합니다."
---

# 모델 컨텍스트 프로토콜 (Model Context Protocol, MCP)

[Model Context Protocol](https://modelcontextprotocol.io/)은 AI 애플리케이션을 데이터 소스 및 도구에 연결하는 개방형 표준입니다. Hermes는 MCP 클라이언트를 내장하여 모든 표준 MCP 서버의 도구를 에이전트에 통합할 수 있습니다.

## 작동 방식

Hermes는 시작 시 MCP 서버에 연결하고 해당 서버의 도구와 리소스를 검색합니다. 그런 다음 이러한 도구는 내장된 도구와 원활하게 통합되어 동일한 컨텍스트와 속도 제한을 공유하며 LLM에 함께 노출됩니다.

도구와 리소스라는 두 가지 기능이 지원됩니다:
- **도구(Tools)**: 에이전트가 데이터 검색, 파일 쓰기 또는 API 호출과 같은 작업을 수행하기 위해 호출할 수 있는 함수입니다.
- **리소스(Resources)**: 에이전트가 로드하고 읽을 수 있는 데이터 조각(예: 문서, 데이터베이스 스키마)입니다. 읽기 전용 컨텍스트의 경우, 에이전트는 `mcp_read_resource` 도구를 사용하여 노출된 모든 리소스 콘텐츠를 가져올 수 있습니다. 프롬프트/템플릿/에이전트는 아직 지원되지 않습니다(클로드 데스크톱과 일치).

## 설정

`~/.hermes/mcp.yaml` 파일을 편집하여 MCP 서버를 구성할 수 있습니다:

```yaml
mcpServers:
  sqlite:
    command: "uvx"
    args: ["mcp-server-sqlite", "--db-path", "/Users/naen/test.db"]
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "your-token-here"
```

구성은 Claude Desktop과 정확히 동일한 스키마를 사용합니다.

서버가 구성되면 다음과 같이 할 수 있습니다:
1. `hermes status`를 실행하여 MCP 서버 연결을 확인합니다.
2. `hermes`를 시작하면 서버에 자동으로 연결됩니다.
3. 에이전트가 새로운 도구와 리소스를 보고 상호작용합니다.

:::tip
구성 후 `hermes mcp refresh`를 실행하여 런타임에 에이전트를 다시 시작하지 않고도 서버 구성을 다시 로드할 수 있습니다.
:::

## 샘플링 (Sampling) — 서버-투-에이전트 LLM 호출

Hermes는 공식 [MCP 샘플링 사양](https://spec.modelcontextprotocol.io/specification/client/sampling/)을 완전히 구현합니다. 이를 통해 **MCP 서버가 자체 목적을 위해 에이전트에 LLM 완성을 요청할 수 있습니다**.

일반적으로 상호작용은 다음과 같이 흐릅니다: 에이전트가 도구를 호출함 → 서버가 작업을 수행함 → 서버가 도구 결과를 반환함.
샘플링을 사용하면 중간 단계가 허용됩니다: 에이전트가 도구를 호출함 → 서버가 작업을 수행함 → **서버가 에이전트에 "이 프롬프트를 나를 위해 완료해 줘"라고 요청함** → 에이전트가 완료를 반환함 → 서버가 작업을 마침 → 서버가 도구 결과를 반환함.

Hermes는 샘플링 요청을 주 에이전트(또는 위임된 하위 에이전트)의 현재 대화, 작업 및 모델 설정에 바인딩합니다:
- **라우팅:** MCP 도구 실행의 일부로 발생하는 샘플링 요청은 해당 도구를 호출한 것과 동일한 에이전트/하위 에이전트로 라우팅됩니다. 이를 통해 분리된 하위 에이전트가 여전히 올바르게 샘플링을 제공할 수 있습니다.
- **제공자:** 샘플링 요청은 에이전트가 현재 사용 중인 것과 동일한 LLM 제공자/모델을 사용하여 백그라운드 호출을 통해 투명하게 수행됩니다.
- **도구 및 권한 부여:** 사양에 따라 샘플링 요청은 *도구가 없는(tool-less)* 보조 호출입니다. 서버는 도구 사용을 요청할 수 없으며 에이전트의 다른 도구(또는 자체 도구)에 액세스할 수 없습니다.

특별한 구성은 필요하지 않습니다 — 샘플링을 요청하는 모든 MCP 서버는 즉시 예상대로 작동합니다.

## 도구 필터링 (Tool Filtering)

에이전트에 노출하고 싶지 않은 도구가 서버에 있는 경우 `config.yaml`의 `mcp` 섹션에 `allowed_tools` 또는 `blocked_tools` 목록을 구성할 수 있습니다.

```yaml
# ~/.hermes/config.yaml
mcp:
  # 모든 서버에서 글로벌하게 차단된 도구
  blocked_tools:
    - fetch_url
    - execute_query
```

서버별로 필터링할 수도 있습니다:

```yaml
# ~/.hermes/config.yaml
mcp:
  servers:
    github:
      # 이 서버에서 허용되는 도구 목록 (다른 도구는 모두 차단됨)
      allowed_tools:
        - search_repositories
        - get_issue
    sqlite:
      # 이 서버에서 차단할 도구
      blocked_tools:
        - execute_query
```

> **참고**: `allowed_tools`와 `blocked_tools`를 모두 지정하는 경우 **양쪽 모두** 통과해야 허용됩니다. 즉, 허용 목록에 있지만 차단 목록에도 있는 도구는 차단됩니다.
