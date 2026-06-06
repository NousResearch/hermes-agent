---
title: "Fastmcp — Python에서 FastMCP를 사용하여 MCP 서버 빌드, 테스트, 검사, 설치 및 배포하기"
sidebar_label: "Fastmcp"
description: "Python에서 FastMCP를 사용하여 MCP 서버 빌드, 테스트, 검사, 설치 및 배포하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Fastmcp

Python에서 FastMCP를 사용하여 MCP 서버를 빌드, 테스트, 검사, 설치 및 배포합니다. 새로운 MCP 서버를 생성하거나, API나 데이터베이스를 MCP 도구로 래핑하거나, 리소스나 프롬프트를 노출하거나, Claude Code, Cursor 또는 HTTP 배포용으로 FastMCP 서버를 준비할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mcp/fastmcp`를 사용하여 설치 |
| 경로 | `optional-skills/mcp/fastmcp` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `MCP`, `FastMCP`, `Python`, `Tools`, `Resources`, `Prompts`, `Deployment` |
| 관련 스킬 | [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`mcporter`](/docs/user-guide/skills/optional/mcp/mcp-mcporter) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 내용은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보게 되는 내용입니다.
:::

# FastMCP

FastMCP를 사용하여 Python으로 MCP 서버를 빌드하고, 로컬에서 유효성을 검사하고, MCP 클라이언트에 설치하며, HTTP 엔드포인트로 배포합니다.

## 사용 시기

작업 내용이 다음과 같을 때 이 스킬을 사용하세요:

- Python으로 새로운 MCP 서버 생성하기
- API, 데이터베이스, CLI 또는 파일 처리 워크플로우를 MCP 도구로 래핑하기
- 도구 외에 리소스나 프롬프트 노출하기
- Hermes나 다른 클라이언트에 연결하기 전에 FastMCP CLI로 서버 스모크 테스트하기
- Claude Code, Claude Desktop, Cursor 또는 유사한 MCP 클라이언트에 서버 설치하기
- HTTP 배포를 위해 FastMCP 서버 레포지토리 준비하기

서버가 이미 존재하고 Hermes에 연결하기만 하면 될 때는 `native-mcp`를 사용하세요. 직접 MCP 서버를 빌드하는 대신 기존 MCP 서버에 대한 임시 CLI 액세스가 목적일 때는 `mcporter`를 사용하세요.

## 전제 조건

먼저 작업 환경에 FastMCP를 설치하세요:

```bash
pip install fastmcp
fastmcp version
```

API 템플릿의 경우, 아직 설치되지 않았다면 `httpx`를 설치하세요:

```bash
pip install httpx
```

## 포함된 파일

### 템플릿

- `templates/api_wrapper.py` - 인증 헤더를 지원하는 REST API 래퍼
- `templates/database_server.py` - 읽기 전용 SQLite 쿼리 서버
- `templates/file_processor.py` - 텍스트 파일 검사 및 검색 서버

### 스크립트

- `scripts/scaffold_fastmcp.py` - 스타터 템플릿을 복사하고 서버 이름 자리 표시자를 대체

### 참조

- `references/fastmcp-cli.md` - FastMCP CLI 워크플로우, 설치 대상 및 배포 검사

## 워크플로우

### 1. 사용 가능한 최소 크기의 서버 형태 선택

가장 좁고 유용한 영역부터 먼저 선택하세요:

- API 래퍼: 전체 API가 아닌 1-3개의 가치가 높은 엔드포인트부터 시작
- 데이터베이스 서버: 읽기 전용 인트로스펙션 및 제한된 쿼리 경로 노출
- 파일 프로세서: 명시적 경로 인수를 사용하는 결정론적 작업 노출
- 프롬프트/리소스: 클라이언트에 재사용 가능한 프롬프트 템플릿이나 검색 가능한 문서가 필요할 때만 추가

모호한 도구가 있는 큰 서버보다 좋은 이름, 독스트링(docstring) 및 스키마가 있는 얇은 서버를 선호합니다.

### 2. 템플릿에서 스캐폴딩

템플릿을 직접 복사하거나 스캐폴드 헬퍼를 사용하세요:

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py \
  --template api_wrapper \
  --name "Acme API" \
  --output ./acme_server.py
```

사용 가능한 템플릿:

```bash
python ~/.hermes/skills/mcp/fastmcp/scripts/scaffold_fastmcp.py --list
```

수동으로 복사하는 경우, `__SERVER_NAME__`을 실제 서버 이름으로 바꾸세요.

### 3. 도구를 먼저 구현

리소스나 프롬프트를 추가하기 전에 `@mcp.tool` 함수부터 시작하세요.

도구 설계 규칙:

- 모든 도구에 동사 기반의 구체적인 이름을 부여
- 사용자 대면 도구 설명으로 독스트링(docstring) 작성
- 매개변수를 명시적으로 유지하고 타입을 지정
- 가능한 경우 구조화된 JSON 안전(JSON-safe) 데이터 반환
- 안전하지 않은 입력은 조기에 검증
- 첫 번째 버전에서는 기본적으로 읽기 전용 동작을 선호

좋은 도구 예시:

- `get_customer`
- `search_tickets`
- `describe_table`
- `summarize_text_file`

나쁜 도구 예시:

- `run`
- `process`
- `do_thing`

### 4. 도움이 될 때만 리소스 및 프롬프트 추가

클라이언트가 스키마, 정책 문서 또는 생성된 보고서와 같이 안정적인 읽기 전용 콘텐츠를 가져오는 것이 유익한 경우 `@mcp.resource`를 추가하세요.

알려진 워크플로우에 대해 서버가 재사용 가능한 프롬프트 템플릿을 제공해야 할 때 `@mcp.prompt`를 추가하세요.

모든 문서를 프롬프트로 바꾸지 마세요. 다음을 선호하세요:

- 작업을 위한 도구
- 데이터/문서 검색을 위한 리소스
- 재사용 가능한 LLM 지침을 위한 프롬프트

### 5. 아무 곳에나 통합하기 전에 서버 테스트

로컬 유효성 검사를 위해 FastMCP CLI를 사용하세요:

```bash
fastmcp inspect acme_server.py:mcp
fastmcp list acme_server.py --json
fastmcp call acme_server.py search_resources query=router limit=5 --json
```

빠른 반복 디버깅을 위해 로컬에서 서버를 실행하세요:

```bash
fastmcp run acme_server.py:mcp
```

로컬에서 HTTP 전송을 테스트하려면:

```bash
fastmcp run acme_server.py:mcp --transport http --host 127.0.0.1 --port 8000
fastmcp list http://127.0.0.1:8000/mcp --json
fastmcp call http://127.0.0.1:8000/mcp search_resources query=router --json
```

서버가 작동한다고 주장하기 전에 최소한 하나의 실제 `fastmcp call`을 각 새로운 도구에 대해 항상 실행하세요.

### 6. 로컬 유효성 검사 통과 시 클라이언트에 설치

FastMCP는 지원되는 MCP 클라이언트에 서버를 등록할 수 있습니다:

```bash
fastmcp install claude-code acme_server.py
fastmcp install claude-desktop acme_server.py
fastmcp install cursor acme_server.py -e .
```

`fastmcp discover`를 사용하여 머신에 이미 구성된 이름 지정된 MCP 서버를 검사하세요.

Hermes 통합이 목적인 경우, 다음 중 하나를 수행하세요:

- `native-mcp` 스킬을 사용하여 `~/.hermes/config.yaml`에 서버를 구성하거나
- 인터페이스가 안정화될 때까지 개발 중에 FastMCP CLI 명령을 계속 사용

### 7. 로컬 컨트랙트가 안정된 후 배포

관리형 호스팅의 경우, Prefect Horizon이 FastMCP에서 가장 직접적으로 문서화하는 경로입니다. 배포하기 전에:

```bash
fastmcp inspect acme_server.py:mcp
```

레포지토리에 다음이 포함되어 있는지 확인하세요:

- FastMCP 서버 객체가 포함된 Python 파일
- `requirements.txt` 또는 `pyproject.toml`
- 배포에 필요한 모든 환경 변수 문서

일반 HTTP 호스팅의 경우, 먼저 로컬에서 HTTP 전송의 유효성을 검사한 다음, 서버 포트를 노출할 수 있는 Python 호환 플랫폼에 배포하세요.

## 일반적인 패턴

### API 래퍼 패턴

REST 또는 HTTP API를 MCP 도구로 노출할 때 사용합니다.

권장하는 첫 번째 조각:

- 하나의 읽기 경로
- 하나의 목록/검색 경로
- 선택적 상태 확인(health check)

구현 참고 사항:

- 인증은 하드코딩하지 않고 환경 변수에 보관
- 하나의 헬퍼에 요청 로직 중앙화
- 간결한 컨텍스트와 함께 API 오류 노출
- 반환하기 전에 일관성 없는 업스트림 페이로드를 정규화

`templates/api_wrapper.py`에서 시작하세요.

### 데이터베이스 패턴

안전한 쿼리 및 검사 기능을 노출할 때 사용합니다.

권장하는 첫 번째 조각:

- `list_tables`
- `describe_table`
- 하나의 제한된 읽기 쿼리 도구

구현 참고 사항:

- 기본적으로 읽기 전용 DB 액세스 허용
- 초기 버전에서는 비 `SELECT` SQL 거부
- 행 수 제한
- 행과 함께 열 이름 반환

`templates/database_server.py`에서 시작하세요.

### 파일 프로세서 패턴

서버가 필요에 따라 파일을 검사하거나 변환해야 할 때 사용합니다.

권장하는 첫 번째 조각:

- 파일 내용 요약
- 파일 내 검색
- 결정론적 메타데이터 추출

구현 참고 사항:

- 명시적 파일 경로 허용
- 누락된 파일 및 인코딩 실패 확인
- 미리보기 및 결과 수 제한
- 특정 외부 도구가 필요한 경우가 아니면 셸아웃(shelling out) 회피

`templates/file_processor.py`에서 시작하세요.

## 품질 기준

FastMCP 서버를 인계하기 전에 다음 사항을 모두 확인하세요:

- 서버가 깨끗하게 임포트(import) 됨
- `fastmcp inspect <file.py:mcp>` 성공
- `fastmcp list <server spec> --json` 성공
- 모든 새로운 도구에 최소한 하나의 실제 `fastmcp call`이 있음
- 환경 변수가 문서화됨
- 추측 없이 이해할 수 있을 만큼 도구 표면이 작음

## 문제 해결

### FastMCP 명령 누락

활성 환경에 패키지를 설치하세요:

```bash
pip install fastmcp
fastmcp version
```

### `fastmcp inspect` 실패

다음 사항을 확인하세요:

- 크래시를 유발하는 부작용 없이 파일이 임포트(import) 됨
- FastMCP 인스턴스의 이름이 `<file.py:object>`에 올바르게 지정됨
- 템플릿의 선택적 종속성이 설치됨

### 도구가 Python에서는 작동하지만 CLI를 통해서는 작동하지 않음

실행하세요:

```bash
fastmcp list server.py --json
fastmcp call server.py your_tool_name --json
```

이는 일반적으로 이름 불일치, 누락된 필수 인수 또는 직렬화할 수 없는 반환 값을 노출합니다.

### Hermes가 배포된 서버를 볼 수 없음

서버 빌드 부분은 올바를 수 있지만 Hermes 구성이 잘못되었을 수 있습니다. `native-mcp` 스킬을 로드하고 `~/.hermes/config.yaml`에 서버를 구성한 다음, Hermes를 재시작하세요.

## 참조

CLI 세부 정보, 설치 대상 및 배포 검사에 대해서는 `references/fastmcp-cli.md`를 읽어보세요.
