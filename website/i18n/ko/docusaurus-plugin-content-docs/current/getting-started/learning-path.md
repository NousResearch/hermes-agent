---
sidebar_position: 3
title: '학습 경로'
description: '경험 수준과 목표에 따라 Hermes Agent 문서를 학습하는 경로를 선택해 보세요.'
---

# 학습 경로

Hermes Agent는 CLI 어시스턴트, 텔레그램/디스코드 봇, 작업 자동화, 강화 학습(RL) 훈련 등 다양한 기능을 제공합니다. 이 페이지는 여러분의 경험 수준과 달성하고자 하는 목표에 따라 어디서부터 시작하고 무엇을 읽어야 할지 결정하는 데 도움을 줍니다.

:::tip 여기서 시작하세요
아직 Hermes Agent를 설치하지 않으셨다면, [설치 가이드](/getting-started/installation)부터 시작하여 [빠른 시작](/getting-started/quickstart)을 진행해 보세요. 아래의 모든 내용은 작동하는 설치 환경이 준비되어 있음을 전제로 합니다.
:::

:::tip 최초 제공자(provider) 설정
처음 사용하는 사용자는 거의 항상 `hermes setup --portal` 명령어가 필요합니다. 하나의 OAuth로 모델 및 4가지 Tool Gateway 도구(검색/이미지/TTS/브라우저)를 모두 커버할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 이 페이지 활용 방법

- **자신의 수준을 알고 계신가요?** [경험 수준별 표](#경험-수준별)로 이동하여 해당 등급의 권장 독서 순서를 따르세요.
- **특정 목표가 있으신가요?** [사용 사례별](#사용-사례별) 섹션으로 건너뛰어 해당하는 시나리오를 찾아보세요.
- **가볍게 둘러보시는 중인가요?** [주요 기능 한눈에 보기](#주요-기능-한눈에-보기) 표를 확인하여 Hermes Agent가 지원하는 모든 기능을 빠르게 훑어보세요.

## 경험 수준별

| 수준 | 목표 | 권장 문서 | 예상 소요 시간 |
|---|---|---|---|
| **초급자** | 설치 및 실행, 기본적인 대화 수행, 내장 도구 사용 | [설치](/getting-started/installation) → [빠른 시작](/getting-started/quickstart) → [CLI 사용법](/user-guide/cli) → [설정](/user-guide/configuration) | ~1시간 |
| **중급자** | 메시징 봇 설정, 메모리, 크론 작업, 스킬 등 고급 기능 활용 | [세션](/user-guide/sessions) → [메시징](/user-guide/messaging) → [도구](/user-guide/features/tools) → [스킬](/user-guide/features/skills) → [메모리](/user-guide/features/memory) → [크론](/user-guide/features/cron) | ~2–3시간 |
| **고급자** | 커스텀 도구 구축, 스킬 생성, 강화 학습(RL)을 통한 모델 훈련, 프로젝트 기여 | [아키텍처](/developer-guide/architecture) → [도구 추가하기](/developer-guide/adding-tools) → [스킬 생성하기](/developer-guide/creating-skills) → [기여하기](/developer-guide/contributing) | ~4–6시간 |

## 사용 사례별

원하는 시나리오를 선택해 보세요. 각 시나리오별로 읽어야 할 순서대로 관련 문서를 링크해 두었습니다.

### "CLI 코딩 어시스턴트로 사용하고 싶습니다"

코드 작성, 검토 및 실행을 위한 대화형 터미널 어시스턴트로 Hermes Agent를 사용해 보세요.

1. [설치](/getting-started/installation)
2. [빠른 시작](/getting-started/quickstart)
3. [CLI 사용법](/user-guide/cli)
4. [코드 실행](/user-guide/features/code-execution)
5. [컨텍스트 파일](/user-guide/features/context-files)
6. [팁 & 요령](/guides/tips)

:::tip
컨텍스트 파일을 사용하여 대화에 직접 파일을 전달할 수 있습니다. Hermes Agent는 프로젝트 내의 코드를 읽고, 편집하고, 실행할 수 있습니다.
:::

### "텔레그램/디스코드 봇을 만들고 싶습니다"

원하는 메시징 플랫폼에 Hermes Agent를 봇으로 배포해 보세요.

1. [설치](/getting-started/installation)
2. [설정](/user-guide/configuration)
3. [메시징 개요](/user-guide/messaging)
4. [텔레그램 설정](/user-guide/messaging/telegram)
5. [디스코드 설정](/user-guide/messaging/discord)
6. [음성 모드](/user-guide/features/voice-mode)
7. [Hermes에서 음성 모드 사용하기](/guides/use-voice-mode-with-hermes)
8. [보안](/user-guide/security)

전체 프로젝트 예시는 다음을 참고하세요:
- [일일 브리핑 봇](/guides/daily-briefing-bot)
- [팀 텔레그램 어시스턴트](/guides/team-telegram-assistant)

### "작업을 자동화하고 싶습니다"

주기적인 작업을 예약하거나, 배치 작업을 실행하거나, 에이전트 액션을 서로 연결할 수 있습니다.

1. [빠른 시작](/getting-started/quickstart)
2. [크론 예약](/user-guide/features/cron)
3. [배치 프로세싱](/user-guide/features/batch-processing)
4. [위임(Delegation)](/user-guide/features/delegation)
5. [훅(Hooks)](/user-guide/features/hooks)

:::tip
크론 작업을 이용하면 사용자가 자리를 비워도 Hermes Agent가 일정에 따라 일일 요약, 주기적 점검, 자동 보고서 생성 등의 작업을 실행할 수 있습니다.
:::

### "커스텀 도구/스킬을 개발하고 싶습니다"

자신만의 도구와 재사용 가능한 스킬 패키지로 Hermes Agent의 기능을 확장해 보세요.

1. [플러그인](/user-guide/features/plugins)
2. [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin)
3. [도구 개요](/user-guide/features/tools)
4. [스킬 개요](/user-guide/features/skills)
5. [MCP (Model Context Protocol)](/user-guide/features/mcp)
6. [아키텍처](/developer-guide/architecture)
7. [도구 추가하기](/developer-guide/adding-tools)
8. [스킬 생성하기](/developer-guide/creating-skills)

:::tip
대부분의 커스텀 도구 개발은 플러그인부터 시작하시는 것이 좋습니다. [도구 추가하기](/developer-guide/adding-tools) 페이지는 일반적인 사용자/커스텀 도구 개발 경로가 아니라, Hermes 코어에 내장 도구를 추가하는 개발자를 위한 문서입니다.
:::

### "모델을 훈련하고 싶습니다"

[Atropos](https://github.com/NousResearch/atropos)를 기반으로 하는 Hermes Agent의 RL 훈련 파이프라인을 통해 강화 학습을 사용하여 모델 동작을 미세 조정할 수 있습니다.

1. [빠른 시작](/getting-started/quickstart)
2. [설정](/user-guide/configuration)
3. [Atropos RL 환경](https://github.com/NousResearch/atropos) (외부 링크)
4. [제공자 라우팅](/user-guide/features/provider-routing)
5. [아키텍처](/developer-guide/architecture)

:::tip
RL 훈련은 Hermes Agent가 대화와 도구 호출을 처리하는 기본적인 방식을 이미 이해하고 있을 때 가장 효과적입니다. 처음이시라면 초급자 경로를 먼저 완료해 보세요.
:::

### "Python 라이브러리로 사용하고 싶습니다"

프로그램 방식으로 Hermes Agent를 직접 작성 중인 Python 애플리케이션에 통합해 보세요.

1. [설치](/getting-started/installation)
2. [빠른 시작](/getting-started/quickstart)
3. [Python 라이브러리 가이드](/guides/python-library)
4. [아키텍처](/developer-guide/architecture)
5. [도구](/user-guide/features/tools)
6. [세션](/user-guide/sessions)

## 주요 기능 한눈에 보기

어떤 기능들이 제공되는지 잘 모르시겠나요? 주요 기능들을 한눈에 볼 수 있는 요약 디렉토리입니다:

| 기능 | 설명 | 링크 |
|---|---|---|
| **도구 (Tools)** | 에이전트가 호출할 수 있는 내장 도구 (파일 I/O, 검색, 쉘 등) | [도구](/user-guide/features/tools) |
| **스킬 (Skills)** | 새로운 기능들을 추가해 주는 설치 가능한 플러그인 패키지 | [스킬](/user-guide/features/skills) |
| **메모리 (Memory)** | 세션 간 지속되는 메모리 | [메모리](/user-guide/features/memory) |
| **컨텍스트 파일 (Context Files)** | 대화에 파일 및 디렉토리 정보 입력 | [컨텍스트 파일](/user-guide/features/context-files) |
| **MCP** | Model Context Protocol을 통해 외부 도구 서버에 연결 | [MCP](/user-guide/features/mcp) |
| **크론 (Cron)** | 에이전트의 주기적인 예약 작업 설정 | [크론](/user-guide/features/cron) |
| **위임 (Delegation)** | 병렬 작업을 위해 서브 에이전트 실행 | [위임](/user-guide/features/delegation) |
| **코드 실행 (Code Execution)** | 프로그램 방식으로 Hermes 도구를 호출하는 Python 스크립트 실행 | [코드 실행](/user-guide/features/code-execution) |
| **브라우저 (Browser)** | 웹 브라우징 및 스크래핑 | [브라우저](/user-guide/features/browser) |
| **훅 (Hooks)** | 이벤트 기반 콜백 및 미들웨어 | [훅](/user-guide/features/hooks) |
| **배치 프로세싱 (Batch Processing)** | 여러 입력을 대량으로 한꺼번에 처리 | [배치 프로세싱](/user-guide/features/batch-processing) |
| **제공자 라우팅 (Provider Routing)** | 다양한 LLM 제공자 간의 요청 라우팅 | [제공자 라우팅](/user-guide/features/provider-routing) |

## 다음에 읽을 내용

현재 진행 상태에 따라 권장하는 단계입니다:

- **설치를 방금 마치셨나요?** → [빠른 시작](/getting-started/quickstart)으로 이동하여 첫 대화를 시작해 보세요.
- **빠른 시작을 완료하셨나요?** → 환경을 커스터마이징하려면 [CLI 사용법](/user-guide/cli) 및 [설정](/user-guide/configuration)을 읽어보세요.
- **기본적인 사용법에 익숙해지셨나요?** → 에이전트의 기능을 최대로 활용하기 위해 [도구](/user-guide/features/tools), [스킬](/user-guide/features/skills), [메모리](/user-guide/features/memory)를 살펴보세요.
- **팀을 위한 구성을 하고 계신가요?** → 권한 제어와 대화 관리를 이해하기 위해 [보안](/user-guide/security) 및 [세션](/user-guide/sessions)을 읽어보세요.
- **직접 개발할 준비가 되셨나요?** → 내부 구조를 파악하고 기여를 시작하기 위해 [개발자 가이드](/developer-guide/architecture)로 이동하세요.
- **실전 예시를 원하시나요?** → 실제 적용 프로젝트와 팁이 담긴 [가이드](/guides/tips) 섹션을 확인해 보세요.

:::tip
모든 문서를 다 읽으실 필요는 없습니다. 목표에 맞는 경로를 선택하고 순서대로 링크를 따라가다 보면 빠르게 생산성을 높일 수 있습니다. 언제든지 이 페이지로 돌아와 다음 단계를 찾아보세요.
:::
