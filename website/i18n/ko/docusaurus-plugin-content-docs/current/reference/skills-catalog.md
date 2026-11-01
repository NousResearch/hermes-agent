---
sidebar_position: 5
title: "번들 스킬 카탈로그"
description: "Hermes Agent에 포함되어 제공되는 스킬 카탈로그"
---

# 번들 스킬 카탈로그

Hermes는 설치 시 `~/.hermes/skills/`에 복사되는 대규모 기본 제공 스킬 라이브러리와 함께 제공됩니다. 아래의 각 스킬은 전체 정의, 설정 및 사용법이 담긴 전용 페이지로 연결됩니다.

Hermes는 `hermes update`를 실행할 때 번들 스킬도 동기화하지만, 동기화 매니페스트는 로컬에서 삭제된 항목과 사용자가 수정한 내용을 존중합니다. 프로필의 `~/.hermes/skills/` 트리에 이 목록의 스킬이 없다면, 해당 스킬은 여전히 Hermes에 포함되어 있는 것이므로 `hermes skills reset <name> --restore`로 복원할 수 있습니다.

이 목록에는 없지만 저장소에 있는 스킬이 있다면, 카탈로그는 `website/scripts/generate-skill-docs.py`에서 다시 생성됩니다.

## apple

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`apple-notes`](/docs/user-guide/skills/bundled/apple/apple-apple-notes) | memo CLI로 Apple Notes 관리: 생성, 검색, 편집. | `apple/apple-notes` |
| [`apple-reminders`](/docs/user-guide/skills/bundled/apple/apple-apple-reminders) | remindctl로 Apple Reminders 관리: 추가, 목록 조회, 완료 처리. | `apple/apple-reminders` |
| [`findmy`](/docs/user-guide/skills/bundled/apple/apple-findmy) | macOS의 FindMy.app을 통해 Apple 기기/AirTag 추적. | `apple/findmy` |
| [`imessage`](/docs/user-guide/skills/bundled/apple/apple-imessage) | macOS에서 imsg CLI로 iMessage/SMS 송수신. | `apple/imessage` |

## autonomous-ai-agents

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code) | Claude Code CLI에 코딩 위임(기능, PR). | `autonomous-ai-agents/claude-code` |
| [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex) | OpenAI Codex CLI에 코딩 위임(기능, PR). | `autonomous-ai-agents/codex` |
| [`computer-use`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-computer-use) | 포커스를 빼앗지 않고 백그라운드에서 데스크톱 조작. | `autonomous-ai-agents/computer-use` |
| [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | Hermes Agent 사용, 설정, 테마 적용, 확장 및 오케스트레이션. | `autonomous-ai-agents/hermes-agent` |
| [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) | OpenCode CLI에 코딩 위임(기능, PR 검토). | `autonomous-ai-agents/opencode` |

## creative

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) | HTML로 제작하는 어두운 테마의 SVG 아키텍처/클라우드/인프라 다이어그램. | `creative/architecture-diagram` |
| [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art) | ASCII 아트: pyfiglet, cowsay, boxes, 이미지-to-ASCII. | `creative/ascii-art` |
| [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) | ASCII 동영상: 비디오/오디오를 색상 ASCII MP4/GIF로 변환. | `creative/ascii-video` |
| [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic) | 인포그래픽: 21개 레이아웃 x 21개 스타일(정보 그래픽, 시각화). | `creative/baoyu-infographic` |
| [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design) | 일회성 HTML 결과물(랜딩 페이지, 프레젠테이션, 프로토타입) 디자인. | `creative/claude-design` |
| [`comfyui`](/docs/user-guide/skills/bundled/creative/creative-comfyui) | 디퓨전 워크플로를 통해 이미지, 동영상 및 오디오 생성. | `creative/comfyui` |
| [`design-md`](/docs/user-guide/skills/bundled/creative/creative-design-md) | Google의 DESIGN.md 토큰 사양 파일 작성/검증/내보내기. | `creative/design-md` |
| [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) | 손으로 그린 느낌의 Excalidraw JSON 다이어그램(아키텍처, 플로, 시퀀스). | `creative/excalidraw` |
| [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) | 텍스트를 자연스럽게 다듬기: AI투의 표현을 제거하고 실제 목소리를 더함. | `creative/humanizer` |
| [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) | Manim CE 애니메이션: 3Blue1Brown 스타일의 수학/알고리즘. | `creative/manim-video` |
| [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js) | p5.js 스케치: 생성 예술, 셰이더, 인터랙티브, 3D. | `creative/p5js` |
| [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs) | HTML/CSS로 구현하는 54개의 실제 디자인 시스템(Stripe, Linear, Vercel). | `creative/popular-web-designs` |
| [`pretext`](/docs/user-guide/skills/bundled/creative/creative-pretext) | DOM 없는 텍스트 레이아웃으로 창의적인 브라우저 데모 제작. | `creative/pretext` |
| [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch) | 비교를 위한 일회성 HTML 목업: 2~3개의 디자인 변형. | `creative/sketch` |
| [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | 작곡 기법과 Suno AI 음악 프롬프트. | `creative/songwriting-and-ai-music` |
| [`touchdesigner-mcp`](/docs/user-guide/skills/bundled/creative/creative-touchdesigner-mcp) | twozero MCP를 통해 TouchDesigner 제어. | `creative/touchdesigner-mcp` |

## email

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`email-inbox-triage`](/docs/user-guide/skills/bundled/email/email-email-inbox-triage) | 받은편지함 분류: 스레드 우선순위 지정, 안전한 답장 초안 작성. | `email/email-inbox-triage` |
| [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) | Himalaya CLI: 터미널에서 IMAP/SMTP 이메일 사용. | `email/himalaya` |

## github

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) | pygount로 코드베이스 검사: LOC, 언어, 비율. | `github/codebase-inspection` |
| [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth) | GitHub 인증 설정: HTTPS 토큰, SSH 키, gh CLI 로그인. | `github/github-auth` |
| [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review) | PR 검토: gh 또는 REST를 통한 diff, 인라인 댓글. | `github/github-code-review` |
| [`github-issue-to-pr`](/docs/user-guide/skills/bundled/github/github-github-issue-to-pr) | 정직한 CI 상태와 함께 GitHub 이슈를 검증된 PR로 진행. | `github/github-issue-to-pr` |
| [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues) | gh 또는 REST로 GitHub 이슈 생성, 분류, 라벨 지정, 할당. | `github/github-issues` |
| [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) | GitHub PR 생명주기: 브랜치, 커밋, 개설, CI, 병합. | `github/github-pr-workflow` |
| [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) | 저장소 복제/생성/포크; 리모트와 릴리스 관리. | `github/github-repo-management` |

## media

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search) | curl + jq로 Tenor에서 GIF 검색/다운로드. | `media/gif-search` |
| [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee) | CLI를 통한 오디오 스펙트로그램/특징(멜, 크로마, MFCC). | `media/songsee` |
| [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) | YouTube 트랜스크립트를 요약, 스레드, 블로그로 변환. | `media/youtube-content` |

## mlops

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`evaluating-llms-harness`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-evaluating-llms-harness) | lm-eval-harness: LLM 벤치마크(MMLU, GSM8K 등). | `mlops/evaluation/evaluating-llms-harness` |
| [`huggingface-hub`](/docs/user-guide/skills/bundled/mlops/mlops-huggingface-hub) | HuggingFace hf CLI: 모델, 데이터셋 검색/다운로드/업로드. | `mlops/huggingface-hub` |
| [`llama-cpp`](/docs/user-guide/skills/bundled/mlops/mlops-inference-llama-cpp) | llama.cpp 로컬 GGUF 추론 + HF Hub 모델 탐색. | `mlops/inference/llama-cpp` |
| [`serving-llms-vllm`](/docs/user-guide/skills/bundled/mlops/mlops-inference-serving-llms-vllm) | vLLM: 고처리량 LLM 서빙, OpenAI API, 양자화. | `mlops/inference/serving-llms-vllm` |
| [`weights-and-biases`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-weights-and-biases) | W&B: ML 실험, 스윕, 모델 레지스트리, 대시보드 기록. | `mlops/evaluation/weights-and-biases` |

## note-taking

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) | Obsidian 볼트의 노트 읽기, 검색, 생성 및 편집. | `note-taking/obsidian` |

## productivity

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable) | curl을 통한 Airtable REST API. 레코드 CRUD, 필터, 업서트. | `productivity/airtable` |
| [`box`](/docs/user-guide/skills/bundled/productivity/productivity-box) | Box로 클라우드 파일, 공유, 검색 및 메타데이터 관리. | `productivity/box` |
| [`document-to-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-document-to-action-items) | 문서에서 인용된 의무, 기한, 작업 추출. | `productivity/document-to-action-items` |
| [`docx`](/docs/user-guide/skills/bundled/productivity/productivity-docx) | Word `.docx` 파일 생성, 읽기, 편집 및 템플릿화. | `productivity/docx` |
| [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) | gws CLI 또는 Python으로 Gmail, Calendar, Drive, Docs, Sheets 사용. | `productivity/google-workspace` |
| [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) | OpenStreetMap/OSRM을 통한 지오코딩, POI, 경로, 시간대. | `productivity/maps` |
| [`meeting-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-meeting-action-items) | 회의 노트를 인용된 결정 사항, 담당자, 티켓으로 변환. | `productivity/meeting-action-items` |
| [`nano-pdf`](/docs/user-guide/skills/bundled/productivity/productivity-nano-pdf) | 자연어 프롬프트로 기존 PDF의 텍스트 편집. | `productivity/nano-pdf` |
| [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) | Notion API + ntn CLI: 페이지, 데이터베이스, 마크다운, Workers. | `productivity/notion` |
| [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) | PDF/스캔 문서에서 텍스트 추출(pymupdf, marker-pdf). | `productivity/ocr-and-documents` |
| [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) | PDF 생성, 읽기, 병합, 작성 및 보안 설정. | `productivity/pdf` |
| [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) | python-pptx로 `.pptx` 프레젠테이션 생성, 읽기, 편집. | `productivity/powerpoint` |
| [`product-price-monitor`](/docs/user-guide/skills/bundled/productivity/productivity-product-price-monitor) | 제품, 항공편 또는 상품 등록 가격 감시; 목표 가격 알림. | `productivity/product-price-monitor` |
| [`session-librarian`](/docs/user-guide/skills/bundled/productivity/productivity-session-librarian) | 프롬프트에 따라 세션 정리: 찾기, 이름 변경, 보관, 정리. | `productivity/session-librarian` |
| [`teams-meeting-pipeline`](/docs/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | Teams 회의 요약, 작업 재생, Graph 구독. | `productivity/teams-meeting-pipeline` |
| [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning) | 주간 리셋: 약속, 정체된 작업, 다음 주 계획. | `productivity/weekly-review-planning` |
| [`xlsx`](/docs/user-guide/skills/bundled/productivity/productivity-xlsx) | Excel `.xlsx` 통합 문서 및 CSV 생성, 읽기, 편집. | `productivity/xlsx` |

## research

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) | 키워드, 저자, 분류 또는 ID로 arXiv 논문 검색. | `research/arxiv` |
| [`blocked-page-recovery`](/docs/user-guide/skills/bundled/research/research-blocked-page-recovery) | 웹 추출기 또는 브라우저가 403/429/챌린지 페이지, 페이월 또는 봇 탐지 중간 페이지에 도달했을 때 아카이브 스냅샷과 리더 대체 수단으로 차단/유료 페이지 복구. | `research/blocked-page-recovery` |
| [`blogwatcher`](/docs/user-guide/skills/bundled/research/research-blogwatcher) | blogwatcher-cli 도구로 블로그와 RSS/Atom 피드 모니터링. | `research/blogwatcher` |
| [`competitor-news-monitor`](/docs/user-guide/skills/bundled/research/research-competitor-news-monitor) | 지정한 기업의 중대한 뉴스 감시; 출처가 인용된 다이제스트. | `research/competitor-news-monitor` |
| [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations) | 인용 가능하고 검증 가능한 출처를 바탕으로 답변과 문서 작성. | `research/grounded-citations` |
| [`llm-wiki`](/docs/user-guide/skills/bundled/research/research-llm-wiki) | Karpathy의 LLM Wiki: 상호 연결된 마크다운 지식 기반 구축/질의. | `research/llm-wiki` |
| [`research-paper-writing`](/docs/user-guide/skills/bundled/research/research-research-paper-writing) | NeurIPS/ICML/ICLR용 ML 논문 작성: 설계→제출. | `research/research-paper-writing` |

## smart-home

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`openhue`](/docs/user-guide/skills/bundled/smart-home/smart-home-openhue) | OpenHue CLI로 Philips Hue 조명, 장면, 방 제어. | `smart-home/openhue` |

## social-media

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) | xurl CLI로 X/Twitter 사용: 원문 게시물 검색, 게시, DM, 미디어. | `social-media/xurl` |

## software-development

| 스킬 | 설명 | 경로 |
|-------|-------------|------|
| [`dogfood`](/docs/user-guide/skills/bundled/software-development/software-development-dogfood) | 웹 앱 탐색적 QA: 버그, 증거, 보고서 발견. | `software-development/dogfood` |
| [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) | 저장소 내 SKILL.md 파일 작성: 프런트매터와 구조. | `software-development/hermes-agent-skill-authoring` |
| [`inspecting-hermes-desktop-dom`](/docs/user-guide/skills/bundled/software-development/software-development-inspecting-hermes-desktop-dom) | CDP를 통해 실행 중인 Hermes 데스크톱 DOM/CSS 읽기. | `software-development/inspecting-hermes-desktop-dom` |
| [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) | --inspect + Chrome DevTools Protocol CLI로 Node.js 디버깅. | `software-development/node-inspect-debugger` |
| [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) | `.hermes/plans/`에 마크다운 계획 작성; 실행하지 않음. | `software-development/plan` |
| [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy) | Python 디버깅: pdb REPL + debugpy 원격(DAP). | `software-development/python-debugpy` |
| [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | 커밋 전 검토: 보안 검사, 품질 게이트, 자동 수정. | `software-development/requesting-code-review` |
| [`simplify-code`](/docs/user-guide/skills/bundled/software-development/software-development-simplify-code) | 최근 코드 변경 사항을 병렬 4개 에이전트로 정리. | `software-development/simplify-code` |
| [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike) | 빌드 전에 아이디어를 검증하는 일회성 실험. | `software-development/spike` |
| [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | 4단계 근본 원인 디버깅: 수정 전에 버그 이해. | `software-development/systematic-debugging` |
| [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD: RED-GREEN-REFACTOR, 코드 전에 테스트 작성 강제. | `software-development/test-driven-development` |
