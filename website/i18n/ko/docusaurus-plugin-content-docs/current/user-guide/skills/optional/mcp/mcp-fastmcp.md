---
title: "Fastmcp — Python MCP 서버 빌드, 테스트 및 배포"
sidebar_label: "Fastmcp"
description: "Python MCP 서버 빌드, 테스트 및 배포"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Fastmcp

FastMCP로 Python MCP 서버를 빌드하고, 테스트하고, 배포합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mcp/fastmcp`로 설치 |
| 경로 | `optional-skills/mcp/fastmcp` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `MCP`, `FastMCP`, `Python`, `Tools`, `Resources`, `Prompts`, `Deployment` |
| 관련 스킬 | [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# FastMCP

FastMCP로 Python에서 MCP 서버를 빌드하고, 로컬에서 검증하고, MCP 클라이언트에 설치하고, HTTP 엔드포인트로 배포합니다.

## 사용 시점

다음 작업을 할 때 이 스킬을 사용합니다.

- Python으로 새 MCP 서버 만들기
- API, 데이터베이스, CLI 또는 파일 처리 워크플로를 MCP 도구로 래핑하기
- 도구 외에 리소스나 프롬프트 노출하기
- Hermes나 다른 클라이언트에 연결하기 전에 FastMCP CLI로 서버 스모크 테스트하기
- Claude Code, Claude Desktop, Cursor 또는 유사한 MCP 클라이언트에 서버 설치하기
- FastMCP 서버 저장소를 HTTP 배포에 맞게 준비하기

서버가 이미 존재하고 Hermes에 연결하기만 하면 `native-mcp`를 사용합니다. 새 서버를 빌드하는 대신 기존 MCP 서버에 임시 CLI로 접근하려는 경우에는 `mcporter`를 사용합니다.

## 사전 요구 사항

먼저 작업 환경에 FastMCP를 설치합니다.

```bash
pip install fastmcp
fastmcp version
```

API 템플릿에는 아직 설치되지 않은 경우 `httpx`를 설치합니다.

```bash
pip install httpx
```

## 포함된 파일

### 템플릿

- `templates/api_wrapper.py` - 인증 헤더를 지원하는 REST API 래퍼
- `templates/database_server.py` - 읽기 전용 SQLite 쿼리 서버
- `templates/file_processor.py` - 텍스트 파일 검사 및 검색 서버

### 스크립트

- `scripts/scaffold_fastmcp.py` - 시작 템플릿을 복사하고 서버 이름 플레이스홀더를 교체

### 참조 자료

- `references/fastmcp-cli.md` - FastMCP CLI 워크플로, 설치 대상 및 배포 확인

## 워크플로

### 1. 가장 작은 실행 가능한 서버 형태 선택

먼저 유용한 표면을 가장 좁게 잡습니다.

- API 래퍼: API 전체가 아니라 가치가 높은 엔드포인트 1~3개로 시작
- 데이터베이스 서버: 읽기 전용 검사와 제약된 쿼리 경로 노출
- 파일 프로세서: 명시적인 경로 인수를 사용하는 결정적 작업 노출
- 프롬프트/리소스: 클라이언트에 재사용 가능한 프롬프트 템플릿이나 검색 가능한 문서가 필요할 때만 추가

모호한 도구가 많은 대형 서버보다 이름, 독스트링, 스키마가 잘 갖춰진 얇은 서버를 우선합니다.

### 2. 템플릿에서 스캐폴딩

템플릿을 직접 복사하거나 스캐폴드 도우미를 사용합니다.

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py \
  --template api_wrapper \
  --name "Acme API" \
  --output ./acme_server.py
```

사용 가능한 템플릿은 다음과 같습니다.

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py --list
```

수동으로 복사하는 경우 `__SERVER_NAME__`을 실제 서버 이름으로 교체합니다.

### 3. 먼저 도구 구현

리소스나 프롬프트를 추가하기 전에 `@mcp.tool` 함수를 구현합니다.

도구 설계 규칙:

- 모든 도구에 구체적인 동사 기반 이름 부여
- 독스트링을 사용자에게 표시되는 도구 설명으로 작성
- 매개변수를 명시적이고 타입이 지정된 상태로 유지
- 가능한 경우 구조화된 JSON 안전 데이터를 반환
- 안전하지 않은 입력은 일찍 검증
- 첫 버전은 기본적으로 읽기 전용 동작 우선

좋은 도구 예시:

- `get_customer`
- `search_tickets`
- `describe_table`
- `summarize_text_file`

좋지 않은 도구 예시:

- `run`
- `process`
- `do_thing`

### 4. 도움이 될 때만 리소스와 프롬프트 추가

클라이언트가 스키마, 정책 문서, 생성된 보고서처럼 안정적인 읽기 전용 콘텐츠를 가져오는 데 도움이 될 때 `@mcp.resource`를 추가합니다.

서버가 알려진 워크플로에 재사용 가능한 프롬프트 템플릿을 제공해야 할 때 `@mcp.prompt`를 추가합니다.

모든 문서를 프롬프트로 만들지 마세요. 다음을 우선합니다.

- 작업에는 도구
- 데이터/문서 검색에는 리소스
- 재사용 가능한 LLM 지침에는 프롬프트

### 5. 어디에든 통합하기 전에 서버 테스트

로컬 검증에는 FastMCP CLI를 사용합니다.

```bash
fastmcp inspect acme_server.py:mcp
fastmcp list acme_server.py --json
fastmcp call acme_server.py search_resources query=router limit=5 --json
```

빠르게 반복하며 디버깅하려면 서버를 로컬에서 실행합니다.

```bash
fastmcp run acme_server.py:mcp
```

HTTP 전송을 로컬에서 테스트하려면 다음을 실행합니다.

```bash
fastmcp run acme_server.py:mcp --transport http --host 127.0.0.1 --port 8000
fastmcp list http://127.0.0.1:8000/mcp --json
fastmcp call http://127.0.0.1:8000/mcp search_resources query=router --json
```

서버가 작동한다고 주장하기 전에 새 도구마다 실제 `fastmcp call`을 최소 한 번은 실행합니다.

### 6. 로컬 검증이 통과하면 클라이언트에 설치

FastMCP는 지원되는 MCP 클라이언트에 서버를 등록할 수 있습니다.

```bash
fastmcp install claude-code acme_server.py
fastmcp install claude-desktop acme_server.py
fastmcp install cursor acme_server.py -e .
```

`fastmcp discover`를 사용하여 컴퓨터에 이미 설정된 이름 있는 MCP 서버를 검사합니다.

Hermes 통합이 목적이라면 다음 중 하나를 선택합니다.

- `native-mcp` 스킬을 사용하여 `~/.hermes/config.yaml`에 서버 설정
- 인터페이스가 안정화될 때까지 개발 중 FastMCP CLI 명령 계속 사용

### 7. 로컬 계약이 안정화된 후 배포

관리형 호스팅에서는 Prefect Horizon이 FastMCP가 가장 직접적으로 문서화하는 경로입니다. 배포 전에 다음을 실행합니다.

```bash
fastmcp inspect acme_server.py:mcp
```

저장소에 다음이 포함되어 있는지 확인합니다.

- FastMCP 서버 객체가 있는 Python 파일
- `requirements.txt` 또는 `pyproject.toml`
- 배포에 필요한 환경 변수 문서

일반 HTTP 호스팅의 경우 먼저 HTTP 전송을 로컬에서 검증한 뒤 서버 포트를 노출할 수 있는 Python 호환 플랫폼에 배포합니다.

## 일반적인 패턴

### API 래퍼 패턴

REST 또는 HTTP API를 MCP 도구로 노출할 때 사용합니다.

권장하는 첫 범위:

- 읽기 경로 하나
- 목록/검색 경로 하나
- 선택적 상태 확인

구현 참고 사항:

- 인증 정보는 하드코딩하지 말고 환경 변수에 보관
- 요청 로직을 하나의 도우미에 중앙화
- 간결한 맥락과 함께 API 오류 노출
- 반환 전에 일관되지 않은 상위 시스템 페이로드 정규화

`templates/api_wrapper.py`에서 시작합니다.

### 데이터베이스 패턴

안전한 쿼리 및 검사 기능을 노출할 때 사용합니다.

권장하는 첫 범위:

- `list_tables`
- `describe_table`
- 제약된 읽기 쿼리 도구 하나

구현 참고 사항:

- 기본적으로 읽기 전용 DB 접근
- 초기 버전에서는 `SELECT`가 아닌 SQL 거부
- 행 수 제한
- 열 이름과 함께 행 반환

`templates/database_server.py`에서 시작합니다.

### 파일 프로세서 패턴

필요할 때 파일을 검사하거나 변환해야 하는 서버에 사용합니다.

권장하는 첫 범위:

- 파일 콘텐츠 요약
- 파일 내부 검색
- 결정적 메타데이터 추출

구현 참고 사항:

- 명시적인 파일 경로 수락
- 파일 누락과 인코딩 실패 확인
- 미리보기와 결과 수 제한
- 특정 외부 도구가 필요한 경우가 아니면 셸 명령 실행을 피함

`templates/file_processor.py`에서 시작합니다.

## 품질 기준

FastMCP 서버를 인계하기 전에 다음을 모두 확인합니다.

- 서버가 부작용 없이 정상적으로 import됨
- `fastmcp inspect <file.py:mcp>` 성공
- `fastmcp list <server spec> --json` 성공
- 새 도구마다 실제 `fastmcp call`이 최소 한 번 존재
- 환경 변수가 문서화됨
- 추측 없이 이해할 수 있을 만큼 도구 표면이 작음

## 문제 해결

### FastMCP 명령을 찾을 수 없음

활성 환경에 패키지를 설치합니다.

```bash
pip install fastmcp
fastmcp version
```

### `fastmcp inspect` 실패

다음을 확인합니다.

- 충돌을 일으키는 부작용 없이 파일이 import되는지
- `<file.py:object>`에서 FastMCP 인스턴스 이름이 올바른지
- 템플릿의 선택적 의존성이 설치되었는지

### Python에서는 작동하지만 CLI를 통하지 않으면 작동하지 않음

다음을 실행합니다.

```bash
fastmcp list server.py --json
fastmcp call server.py your_tool_name --json
```

이렇게 하면 일반적으로 이름 불일치, 누락된 필수 인수 또는 직렬화할 수 없는 반환값이 드러납니다.

### Hermes가 배포된 서버를 볼 수 없음

서버 빌드 부분은 올바르지만 Hermes 설정이 잘못되었을 수 있습니다. `native-mcp` 스킬을 로드하고 `~/.hermes/config.yaml`에 서버를 설정한 다음 Hermes를 재시작합니다.

## 참조 자료

CLI 세부 사항, 설치 대상 및 배포 확인은 `references/fastmcp-cli.md`를 읽으세요.
