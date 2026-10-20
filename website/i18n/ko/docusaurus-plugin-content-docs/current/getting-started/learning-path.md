---
sidebar_position: 3
title: '학습 경로'
description: '경험 수준과 목표에 따라 Hermes Agent 문서에서 학습 경로를 선택하세요.'
---

# 학습 경로

Hermes Agent는 CLI 도우미, Telegram/Discord 봇, 작업 자동화, RL 학습 등 다양한 일을 할 수 있습니다. 이 페이지에서는 경험 수준과 하려는 작업에 따라 어디서 시작하고 무엇을 읽을지 안내합니다.

:::tip 여기서 시작하세요
아직 Hermes Agent를 설치하지 않았다면 [설치 가이드](/getting-started/installation)부터 시작한 다음 [빠른 시작](/getting-started/quickstart)을 따라 하세요. 아래 내용은 정상적으로 설치되어 있다고 가정합니다.
:::

:::tip 처음 제공자 설정하기
처음 사용하는 사람은 대부분 `hermes setup --portal`을 원합니다. 하나의 OAuth로 모델과 네 가지 Tool Gateway 도구(검색/이미지/TTS/브라우저)를 설정할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 이 페이지 사용법

- **수준을 알고 있나요?** [경험 수준별 표](#by-experience-level)로 이동해 자신의 단계에 맞는 읽기 순서를 따르세요.
- **특정 목표가 있나요?** [사용 사례별](#by-use-case)로 이동해 상황에 맞는 항목을 찾으세요.
- **그냥 둘러보는 중인가요?** [주요 기능](#key-features-at-a-glance) 표에서 Hermes Agent가 할 수 있는 일을 빠르게 살펴보세요.

## 경험 수준별

| 수준 | 목표 | 권장 읽기 | 예상 시간 |
|---|---|---|---|
| **초급** | 실행 환경 구성, 기본 대화, 내장 도구 사용 | [설치](/getting-started/installation) → [빠른 시작](/getting-started/quickstart) → [CLI 사용법](/user-guide/cli) → [설정](/user-guide/configuration) | 약 1시간 |
| **중급** | 메시징 봇 설정, 메모리·cron 작업·스킬 같은 고급 기능 사용 | [세션](/user-guide/sessions) → [메시징](/user-guide/messaging) → [도구](/user-guide/features/tools) → [스킬](/user-guide/features/skills) → [메모리](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | 약 2~3시간 |
| **고급** | 사용자 지정 도구 제작, 스킬 생성, RL로 모델 학습, 프로젝트 기여 | [아키텍처](/developer-guide/architecture) → [도구 추가](/developer-guide/adding-tools) → [스킬 만들기](/developer-guide/creating-skills) → [기여하기](/developer-guide/contributing) | 약 4~6시간 |

## 사용 사례별

하고 싶은 일에 맞는 시나리오를 선택하세요. 각 항목은 관련 문서를 읽어야 할 순서대로 연결합니다.

### "CLI 코딩 도우미를 원합니다"

대화형 터미널 도우미로 코드를 작성하고, 검토하고, 실행하세요.

1. [설치](/getting-started/installation)
2. [빠른 시작](/getting-started/quickstart)
3. [CLI 사용법](/user-guide/cli)
4. [코드 실행](/user-guide/features/code-execution)
5. [컨텍스트 파일](/user-guide/features/context-files)
6. [팁과 요령](/guides/tips)

:::tip
컨텍스트 파일로 파일을 대화에 직접 전달하세요. Hermes Agent는 프로젝트의 코드를 읽고, 편집하고, 실행할 수 있습니다.
:::

### "Telegram/Discord 봇을 원합니다"

자주 사용하는 메시징 플랫폼에서 Hermes Agent를 봇으로 배포하세요.

1. [설치](/getting-started/installation)
2. [설정](/user-guide/configuration)
3. [메시징 개요](/user-guide/messaging)
4. [Telegram 설정](/user-guide/messaging/telegram)
5. [Discord 설정](/user-guide/messaging/discord)
6. [음성 모드](/user-guide/features/voice-mode)
7. [Hermes에서 음성 모드 사용](/guides/use-voice-mode-with-hermes)
8. [보안](/user-guide/security)

전체 프로젝트 예제는 다음을 참고하세요.
- [일일 브리핑 봇](/guides/daily-briefing-bot)
- [팀 Telegram 어시스턴트](/guides/team-telegram-assistant)

### "작업을 자동화하고 싶습니다"

반복 작업을 예약하고, 일괄 작업을 실행하고, 에이전트 작업을 함께 연결하세요.

1. [빠른 시작](/getting-started/quickstart)
2. [Cron 예약](/user-guide/features/cron)
3. [일괄 처리](/user-guide/features/batch-processing)
4. [위임](/user-guide/features/delegation)
5. [훅](/user-guide/features/hooks)

:::tip
Cron 작업을 사용하면 매일 요약, 주기적 확인, 자동 보고서 같은 작업을 사용자가 없어도 일정에 따라 실행할 수 있습니다.
:::

### "사용자 지정 도구/스킬을 만들고 싶습니다"

직접 만든 도구와 재사용 가능한 스킬 패키지로 Hermes Agent를 확장하세요.

1. [플러그인](/user-guide/features/plugins)
2. [Hermes 플러그인 만들기](/developer-guide/plugins)
3. [도구 개요](/user-guide/features/tools)
4. [스킬 개요](/user-guide/features/skills)
5. [MCP(Model Context Protocol)](/user-guide/features/mcp)
6. [아키텍처](/developer-guide/architecture)
7. [도구 추가](/developer-guide/adding-tools)
8. [스킬 만들기](/developer-guide/creating-skills)

:::tip
대부분의 사용자 지정 도구 제작은 플러그인부터 시작하세요. [도구 추가](/developer-guide/adding-tools) 페이지는 일반적인 사용자/사용자 지정 도구 경로가 아니라 Hermes 내장 코어 개발을 위한 문서입니다.
:::

### "모델을 학습시키고 싶습니다"

Hermes Agent의 RL 학습 파이프라인([Atropos](https://github.com/NousResearch/atropos) 기반)을 사용해 강화 학습으로 모델 동작을 미세 조정하세요.

1. [빠른 시작](/getting-started/quickstart)
2. [설정](/user-guide/configuration)
3. [Atropos RL 환경](https://github.com/NousResearch/atropos) (외부)
4. [제공자 라우팅](/user-guide/features/provider-routing)
5. [아키텍처](/developer-guide/architecture)

:::tip
Hermes Agent가 대화와 도구 호출을 처리하는 기본 방식을 이미 이해하고 있을 때 RL 학습이 가장 잘 작동합니다. 처음이라면 먼저 초급 경로를 따라 하세요.
:::

### "Python 라이브러리로 사용하고 싶습니다"

Hermes Agent를 프로그래밍 방식으로 자신의 Python 애플리케이션에 통합하세요.

1. [설치](/getting-started/installation)
2. [빠른 시작](/getting-started/quickstart)
3. [Python 라이브러리 가이드](/guides/python-library)
4. [아키텍처](/developer-guide/architecture)
5. [도구](/user-guide/features/tools)
6. [세션](/user-guide/sessions)

## 주요 기능 한눈에 보기

무엇을 사용할 수 있는지 확실하지 않나요? 주요 기능을 간단히 정리하면 다음과 같습니다.

| 기능 | 하는 일 | 링크 |
|---|---|---|
| **도구** | 에이전트가 호출할 수 있는 내장 도구(파일 I/O, 검색, 셸 등) | [도구](/user-guide/features/tools) |
| **스킬** | 새로운 기능을 추가하는 설치 가능한 플러그인 패키지 | [스킬](/user-guide/features/skills) |
| **메모리** | 세션 간 영구 메모리 | [메모리](/user-guide/features/memory) |
| **컨텍스트 파일** | 파일과 디렉터리를 대화에 제공 | [컨텍스트 파일](/user-guide/features/context-files) |
| **MCP** | Model Context Protocol을 통해 외부 도구 서버에 연결 | [MCP](/user-guide/features/mcp) |
| **Cron** | 반복 에이전트 작업 예약 | [Cron](/user-guide/features/cron) |
| **위임** | 병렬 작업을 위한 서브 에이전트 생성 | [위임](/user-guide/features/delegation) |
| **코드 실행** | Hermes 도구를 프로그래밍 방식으로 호출하는 Python 스크립트 실행 | [코드 실행](/user-guide/features/code-execution) |
| **브라우저** | 웹 브라우징 및 스크래핑 | [브라우저](/user-guide/features/browser) |
| **Hooks** | 이벤트 기반 콜백 및 미들웨어 | [Hooks](/user-guide/features/hooks) |
| **일괄 처리** | 여러 입력을 대량으로 처리 | [일괄 처리](/user-guide/features/batch-processing) |
| **제공자 라우팅** | 여러 LLM 제공자로 요청 라우팅 | [제공자 라우팅](/user-guide/features/provider-routing) |

## 다음에 읽을 문서

현재 위치에 따라 다음 문서를 읽으세요.

- **설치를 막 마쳤나요?** → [빠른 시작](/getting-started/quickstart)으로 이동해 첫 대화를 실행하세요.
- **빠른 시작을 끝냈나요?** → [CLI 사용법](/user-guide/cli)과 [설정](/user-guide/configuration)을 읽고 환경을 사용자 지정하세요.
- **기본 사항에 익숙한가요?** → [도구](/user-guide/features/tools), [스킬](/user-guide/features/skills), [메모리](/user-guide/features/memory)를 살펴보고 에이전트의 기능을 최대한 활용하세요.
- **팀용으로 설정하나요?** → [보안](/user-guide/security)과 [세션](/user-guide/sessions)을 읽고 접근 제어와 대화 관리를 이해하세요.
- **만들 준비가 되었나요?** → [개발자 가이드](/developer-guide/architecture)로 이동해 내부 구조를 이해하고 기여를 시작하세요.
- **실용적인 예제가 필요하나요?** → [가이드](/guides/tips)에서 실제 프로젝트와 팁을 확인하세요.

:::tip
모든 문서를 읽을 필요는 없습니다. 목표에 맞는 경로를 선택하고 링크를 순서대로 따라가면 빠르게 생산성을 높일 수 있습니다. 다음 단계가 필요할 때 언제든 이 페이지로 돌아오세요.
:::
