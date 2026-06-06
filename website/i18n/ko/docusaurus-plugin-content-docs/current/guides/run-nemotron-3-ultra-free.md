---
sidebar_position: 0
title: "Hermes Agent에서 Nemotron 3 Ultra 무료로 실행하기"
description: "Nous Portal에서 NVIDIA Nemotron 3 Ultra 체험 — 6월 4일~18일 무료 제공 — Hermes Agent에서 당일(Day 0) 지원"
---

# Hermes Agent에서 Nemotron 3 Ultra 무료로 실행하기

Nous Research가 **NVIDIA**와 협력하여 개방형 첨단 파운데이션 모델을 발전시키기 위해 선도적인 AI 연구소들의 **Nemotron Coalition**에 합류하게 되었습니다. 이를 기념하여 파트너인 **Nebius**와 함께 [Nous Portal](https://portal.nousresearch.com)에서 2주간(**6월 4일 ~ 6월 18일**) **Nemotron 3 Ultra**를 무료로 제공합니다. 아래 지침을 따라 오늘 여러분의 Hermes Agent에서 이 모델을 직접 경험해 보세요.

:::info 기간 한정 제공
`nvidia/nemotron-3-ultra:free` 티어는 **6월 4일부터 6월 18일까지** 이용할 수 있습니다. 무료 플랜을 유지하려면 `:free` 태그가 붙은 해당 변형(variant)을 정확히 선택해야 합니다.
:::

원하는 설치 방법을 선택하세요. **데스크톱 앱**이 가장 쉬운 방법이며 터미널이 필요하지 않습니다. 터미널 환경이 편하시다면, 바로 아래에 있는 **명령줄(CLI)** 설치 방법을 이용하세요.

## 옵션 A — 데스크톱 앱 (권장)

가장 간단한 방법: 클릭 한 번으로 설치가 완료되며, 화면 안내에 따라 설정이 진행됩니다. 터미널이 전혀 필요 없습니다.

### 1. 다운로드 및 설치

macOS 또는 Windows용 [Hermes Desktop 설치 프로그램 다운로드](https://hermes-agent.nousresearch.com/desktop) 후, 파일을 엽니다. 첫 실행 시 설정이 마무리되며, 보통 1분 이내에 완료됩니다.

### 2. Nous Portal 연결

앱이 열리면 "Let's get you set up(설정을 시작합시다)" 화면이 나타납니다. **Recommended(권장)**로 표시된 **Nous Portal**을 클릭하세요. 브라우저가 열리면 [Nous Portal](https://portal.nousresearch.com) 계정을 생성하거나 로그인하고, **Free(무료)** 플랜을 선택한 뒤 Hermes를 승인(authorize)합니다. 앱이 자동으로 연결됩니다.

### 3. 무료 Nemotron 3 Ultra 모델 선택

연결 후, 앱에 **Default model(기본 모델)** 카드가 표시됩니다. **Change(변경)**를 클릭하고, **nemotron 3 ultra**를 검색한 후 **Free tier(무료 티어)** 태그가 있는 다음 항목을 선택하세요:

```
nvidia/nemotron-3-ultra:free
```

`:free` 태그가 있어야 무료 티어로 유지되므로, 반드시 해당 항목을 선택해야 합니다.

### 4. 대화 시작

**Start chatting(대화 시작)**을 클릭하세요. 이것으로 끝입니다 — 이제 Nemotron 3 Ultra와 무료로 대화를 시작할 수 있습니다.

## 옵션 B — 명령줄 (CLI)

터미널을 선호하시나요?

### 1. Hermes Agent 설치

macOS/Linux/WSL2/Android의 경우 다음을 실행하세요:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Windows의 경우 다음을 실행하세요:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

먼저 스크립트를 검토하고 싶다면 [`install.sh`](https://hermes-agent.nousresearch.com/install.sh)를 다운로드하여 내용을 확인한 후 실행하세요.

완료 후, 셸(shell)을 다시 로드합니다:

```bash
source ~/.bashrc   # 또는 source ~/.zshrc
```

### 2. 빠른 설정(Quick Setup) 실행

```bash
hermes setup
```

**Quick Setup(빠른 설정)**을 선택하세요. Hermes가 브라우저 탭을 열고 다음 단계가 완료될 때까지 기다립니다.

### 3. Nous Portal 계정 생성

브라우저에서 [Nous Portal](https://portal.nousresearch.com) 계정을 생성하거나 로그인하고 **Free(무료)** 플랜을 선택합니다.

### 4. 계정 연결

Hermes Agent에 계정을 연결할지 묻는 메시지가 나타나면 **Connect(연결)**를 클릭합니다. 연동이 완료되면 확인 메시지가 나타납니다.

### 5. 무료 Nemotron 3 Ultra 모델 선택

터미널로 돌아가서 모델 목록 중 다음을 선택합니다:

```
nvidia/nemotron-3-ultra:free
```

`:free` 태그가 있어야 무료 티어로 유지되므로, 반드시 해당 항목을 선택해야 합니다.

### 6. 대화 시작

남은 빠른 설정 과정을 완료한 후 다음을 실행합니다:

```bash
hermes
```

이것으로 끝입니다 — 이제 Nemotron 3 Ultra와 무료로 대화를 시작할 수 있습니다.

## 나중에 모델 변경하기

이미 다른 모델로 설정하셨나요?

- **데스크톱 앱:** 모델 선택기를 열고 **nemotron 3 ultra**를 검색한 다음 **Free tier** 항목을 선택하세요.
- **CLI / TUI:** 세션 내부에서 언제든지 `/model nvidia/nemotron-3-ultra:free`를 입력하여 전환하거나, `/model`을 실행하여 선택기를 열고 목록에서 선택하세요.

## 문제 해결

- **목록에 모델이 보이지 않나요?** Nous Portal 연결을 완료했는지, 그리고 **Free** 플랜을 선택했는지 확인하세요. CLI에서 `hermes portal info`를 입력하면 로그인 상태와 라우팅이 Nous를 통해 이루어지는지 확인할 수 있습니다.
- **잘못된 항목을 선택했나요?** `nvidia/nemotron-3-ultra:free`를 다시 선택하세요 — 무료 티어를 유지하려면 `:free` 접미사가 필수적입니다.
- **브라우저가 열리지 않거나 원격 호스트(CLI) 환경인가요?** 포트 포워딩 및 수동 붙여넣기 해결 방법은 [SSH / 원격 호스트에서의 OAuth](/guides/oauth-over-ssh)를 참조하세요.

## 함께 보기

- **[데스크톱 앱](/user-guide/desktop)** — 클릭 한 번으로 설치되는 네이티브 앱 (macOS, Windows, Linux)
- **[Nous Portal로 Hermes Agent 실행하기](/guides/run-hermes-with-nous-portal)** — 전체 Portal 워크스루: 모델, Tool Gateway, 검증
- **[Nous Portal 통합](/integrations/nous-portal)** — 구독에 포함된 기능 알아보기
- **[빠른 시작 (Quickstart)](/getting-started/quickstart)** — 설치에서 대화까지 5분 완성
