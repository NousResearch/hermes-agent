---
title: 결과물 제공 모드 (채팅 내 아티팩트)
sidebar_label: 결과물 제공 모드
description: 에이전트가 생성한 차트, PDF, 스프레드시트 및 기타 파일을 메시징 플랫폼의 네이티브 첨부 파일 형태로 전달하는 방법.
---

# 결과물 제공 모드 (Deliverable Mode)

Hermes 에이전트가 메시징 게이트웨이(Slack, Discord, Telegram, WhatsApp, Signal 등) 내에서 실행될 때, 생성된 파일을 사용자가 복사해야 하는 경로가 아닌 네이티브 첨부 파일 형태로 채팅방에 직접 전달할 수 있습니다.

차트는 인라인 이미지로 표시됩니다. PDF 보고서는 파일 다운로드 형태로 표시됩니다. 스프레드시트는 `.xlsx`로 업로드됩니다. 에이전트는 `MEDIA:` 태그를 쓰거나 특별한 작업을 수행할 필요가 없습니다 — 단순히 파일을 생성하고 응답 내에 그 절대 경로를 언급하기만 하면 됩니다. 게이트웨이는 텍스트에서 경로를 추출하고, 사용자에게 보이는 메시지에서는 이를 제거한 뒤, 파일을 네이티브 형태로 업로드합니다.

## 작동 방식 (How it works)

세 가지 요소가 결합되어 작동합니다:

1. **에이전트가 파일을 생성하는 도구를 갖추고 있습니다.** matplotlib를 통한 차트 생성을 위한 `execute_code`, PDF를 위한 `latex-pdf-report` 스킬, 발표 자료를 위한 `powerpoint` 스킬, 이미지를 위한 `image_generate`, 오디오를 위한 `text_to_speech` 등이 있습니다.

2. **게이트웨이가 에이전트의 응답에서 파일 경로를 스캔합니다.** 지원되는 확장자로 끝나는 절대 경로(`/tmp/...`) 또는 홈 디렉토리 상대 경로(`~/...`)가 추출됩니다. 코드 블록과 인라인 코드 안의 경로는 무시되므로 예제 코드가 훼손되는 일은 발생하지 않습니다.

3. **게이트웨이가 파일 유형에 따라 디스패치합니다.** 플랫폼이 지원하는 경우 이미지는 인라인으로 삽입되고, 비디오도 인라인으로 삽입되며, 오디오는 음성/오디오 첨부 파일로 라우팅되고, 그 외의 모든 것은 파일 첨부 형태로 업로드됩니다.

## 지원되는 파일 확장자 (Supported file extensions)

| 카테고리 | 확장자 | 전송 방식 |
|---|---|---|
| 이미지 | `.png .jpg .jpeg .gif .webp .bmp .tiff .svg` | 인라인 삽입 |
| 비디오 | `.mp4 .mov .avi .mkv .webm` | 인라인 삽입 (지원되는 경우) |
| 오디오 | `.mp3 .wav .ogg .m4a .flac` | 음성 / 오디오 첨부 파일 |
| 문서 | `.pdf .docx .doc .odt .rtf .txt .md` | 파일 업로드 |
| 데이터 | `.xlsx .xls .csv .tsv .json .xml .yaml .yml` | 파일 업로드 |
| 프레젠테이션 | `.pptx .ppt .odp` | 파일 업로드 |
| 압축 파일 | `.zip .tar .gz .tgz .bz2 .7z` | 파일 업로드 |
| 웹 | `.html .htm` | 파일 업로드 |

에이전트가 임의의 소스 파일을 자동으로 전송하지 않도록 `.py`, `.log` 및 기타 소스 파일 확장자는 의도적으로 제외되었습니다. 코드를 사용자에게 보내려면 코드 블록을 사용하세요.

## 에이전트가 아티팩트를 생성하도록 유도하기 (Encouraging the agent to produce artifacts)

에이전트는 기본적으로 아티팩트를 적극 활용하지 않습니다 — 그 방법을 알아야 합니다. 이를 유도하는 두 가지 방법이 있습니다:

**세션 단위:** 명시적으로 요청하거나 ("비교 결과를 차트로 보내줘", "데이터를 CSV로 반환해 줘"), 메시징 플랫폼에서 아티팩트 스타일의 답변을 선호하도록 맞춤 지시사항 / 성격(personality) 항목을 작성하세요.

**프로젝트 수준:** 에이전트가 작업하는 프로젝트의 `AGENTS.md` / `CLAUDE.md` / `.cursorrules`에 이 성향을 추가하거나, `~/.hermes/SOUL.md`의 전역 페르소나에 추가하거나, `~/.hermes/config.yaml`의 `agent.personalities` 아래 이름이 지정된 사전 설정으로 추가하세요 (`/personality`를 통해 세션별로 전환 가능).

에이전트가 사용해야 하는 메커니즘은 간단합니다: 파일을 절대 경로(예: `/tmp/q3-revenue.png`)로 렌더링하고 그 경로를 답변에 일반 텍스트로 언급하면 됩니다. 나머지는 게이트웨이가 알아서 처리합니다. 펜스 코드 블록이나 백틱(`) 안의 경로는 무시되므로 예제 코드가 훼손되지 않습니다.

## 칸반: 완료 알림에 포함되는 아티팩트 (Kanban: artifacts ride completion notifications)

Hermes의 칸반(kanban) 다중 에이전트 워크플로우를 사용하는 경우, 워커(worker)는 `kanban_complete` 호출 시 결과물 파일을 첨부할 수 있습니다:

```python
kanban_complete(
    summary="rendered Q3 revenue chart and report",
    artifacts=[
        "/tmp/q3-revenue.png",
        "/tmp/q3-report.pdf",
    ],
)
```

게이트웨이 알리미가 Slack/Telegram 등에서 해당 작업을 구독한 사용자에게 "작업 완료" 메시지를 전달할 때, 각 아티팩트도 해당 채팅방에 네이티브 첨부 파일로 함께 업로드합니다. 사용자는 결과물과 요약 내용을 한곳에서 받아볼 수 있습니다.

알리미가 실행될 때 디스크에 존재하지 않는 파일은 조용히 건너뜁니다.

## MCP를 통해 더 많은 서비스 연결 (Connecting more services with MCP)

아티팩트 전송 파이프라인 외에도 에이전트는 MCP(Model Context Protocol)를 통해 다른 서비스에 접근할 수 있습니다. MCP 생태계는 대부분의 널리 사용되는 도구들을 위한 커뮤니티 서버를 제공합니다 — 필요한 것을 설치하세요:

| 서비스 | 활용 가능한 기능 |
|---|---|
| **Notion** | Notion 페이지/데이터베이스 읽기 및 쓰기, 워크스페이스 쿼리 |
| **GitHub** | 이슈, PR, 댓글 관리, gh CLI를 넘어선 저장소 검색 |
| **Linear** | 티켓, 프로젝트, 사이클 관리 |
| **Slack** | 워크스페이스 전체 검색, 다른 채널 읽기 |
| **Gmail** | 받은 편지함 분류, 메일 전송, 라벨 관리 |
| **Salesforce** | 리드, 기회, 계정 데이터 확인 |
| **Snowflake / BigQuery** | 데이터 웨어하우스에 대한 SQL 실행 |
| **Google Drive** | 파일 검색, 내용 확인, 공유 관리 |

`~/.hermes/config.yaml`의 `mcp_servers` 섹션을 통해 MCP 서버를 설치하세요. 전체 설정 가이드는 [MCP 연동](./mcp.md)을 참조하세요.

## Slack의 Perplexity Computer와의 비교 (Comparison to Perplexity Computer in Slack)

Perplexity Computer의 Slack 연동은 동일한 아이디어를 기반으로 합니다: 에이전트가 결과물(차트, PDF, 슬라이드 덱)을 생성하고 이를 스레드에 네이티브 첨부 파일 형태로 다시 게시합니다. Hermes 에이전트의 결과물 제공 모드는 사용자 측면에서 로컬로 동일한 패턴을 제공합니다:

- 생성 과정은 원격 테넌트가 아닌 사용자의 자체 venv / 샌드박스에서 발생합니다.
- 파일은 동일한 Slack `files.uploadV2` API를 통해 채팅방에 업로드됩니다.
- 400여 개의 선별된 호스팅 연동 카탈로그 대신 MCP를 통해 광범위한 커넥터를 제공합니다 — 실제로 사용하는 커넥터만 설치하세요.

OAuth 토큰은 원격 토큰 저장소 없이 사용자의 장비인 `auth.json` / `.env`에 보관됩니다. 다중 테넌트(multi-tenant) 마이크로 VM도 사용하지 않습니다. 하지만 최종 결과는 동일합니다.
