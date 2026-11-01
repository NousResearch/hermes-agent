---
sidebar_position: 0
title: "Hermes Agent에서 Nemotron 3 Ultra를 무료로 실행하기"
description: "Nous Portal에서 NVIDIA Nemotron 3 Ultra를 무료로 사용해 보세요 — 6월 4일~18일 — Hermes Agent에서 day 0 지원"
---

# Hermes Agent에서 Nemotron 3 Ultra를 무료로 실행하기

Nous Research는 **NVIDIA**와 함께 오픈 프런티어 기반 모델을 발전시키는 주요 AI 연구소 연합인 **Nemotron Coalition**에 합류했습니다. 이를 기념하여 [Nous Portal](https://portal.nousresearch.com)에서 **Nemotron 3 Ultra**를 2주 동안(**6월 4일~6월 18일**) 무료로 제공하도록 **Nebius**와 협력했습니다. 아래 지침에 따라 오늘 Hermes Agent에서 모델을 사용해 보세요.

:::info 기간 한정 혜택
`nvidia/nemotron-3-ultra:free` 티어는 **6월 4일부터 6월 18일까지** 이용할 수 있습니다. 무료 요금제로 유지하려면 `:free` 태그가 필요하므로 정확히 이 변형을 선택하세요.
:::

자신에게 맞는 설치 방법을 선택하세요. **데스크톱 앱**이 가장 쉽고 터미널이 필요하지 않습니다. 터미널을 선호한다면 바로 아래에 **명령줄** 설치 방법이 있습니다.

## 옵션 A — 데스크톱 앱(권장)

가장 간단한 방법은 안내에 따라 클릭만 하면 되는 원클릭 설치 프로그램입니다. 터미널이 필요하지 않습니다.

### 1. 다운로드 및 설치

macOS 또는 Windows용 [Hermes Desktop 설치 프로그램을 다운로드](https://hermes-agent.nousresearch.com/)한 후 실행합니다. 처음 실행하면 자체 설정이 완료됩니다(보통 1분 이내).

### 2. Nous Portal 연결

앱이 열리면 "설정을 시작해 보세요" 화면이 표시됩니다. **Nous Portal**(**권장**으로 표시됨)을 클릭합니다. 브라우저가 열리면 [Nous Portal](https://portal.nousresearch.com) 계정을 만들거나 로그인하고, **Free** 요금제를 선택한 다음 Hermes를 승인합니다. 앱이 자동으로 연결됩니다.

### 3. 무료 Nemotron 3 Ultra 모델 선택

연결되면 앱에 **기본 모델** 카드가 표시됩니다. **변경**을 클릭하고 **nemotron 3 ultra**를 검색한 다음 **Free tier** 태그가 붙은 변형을 선택합니다.

```
nvidia/nemotron-3-ultra:free
```

무료 티어로 유지하려면 `:free` 태그가 필요하므로 해당 변형을 선택하세요.

### 4. 채팅 시작

**채팅 시작**을 클릭합니다. 이제 무료로 Nemotron 3 Ultra와 대화할 수 있습니다.

## 옵션 B — 명령줄

터미널을 선호하시나요?

### 1. Hermes Agent 설치

macOS/Linux/WSL2/Android에서는 다음을 실행합니다.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Windows에서는 다음을 실행합니다.

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

먼저 검토하고 싶나요? [`install.sh`](https://hermes-agent.nousresearch.com/install.sh)를 다운로드하여 내용을 확인한 다음 실행합니다.

완료되면 셸을 다시 로드합니다.

```bash
source ~/.bashrc   # or source ~/.zshrc
```

### 2. 빠른 설정 실행

```bash
hermes setup
```

**빠른 설정**을 선택합니다. Hermes가 브라우저 탭을 열고 다음 단계를 완료할 때까지 기다립니다.

### 3. Nous Portal 계정 만들기

브라우저에서 [Nous Portal](https://portal.nousresearch.com) 계정을 만들거나 로그인하고 **Free** 요금제를 선택합니다.

### 4. 계정 연결

Hermes Agent에 계정을 연결하라는 메시지가 표시되면 **연결**을 클릭합니다. 연결되면 확인 메시지가 표시됩니다.

### 5. 무료 Nemotron 3 Ultra 모델 선택

터미널로 돌아갑니다. 모델 목록에서 다음을 선택합니다.

```
nvidia/nemotron-3-ultra:free
```

무료 요금제로 유지하려면 `:free` 태그가 필요하므로 해당 변형을 선택해야 합니다.

### 6. 채팅 시작

남은 빠른 설정 안내를 완료한 다음 다음을 실행합니다.

```bash
hermes
```

이제 무료로 Nemotron 3 Ultra와 대화할 수 있습니다.

## 나중에 이 모델로 전환하기

이미 다른 모델로 설정했나요?

- **데스크톱 앱:** 모델 선택기를 열고 **nemotron 3 ultra**를 검색한 다음 **Free tier** 변형을 선택합니다.
- **CLI / TUI:** 세션 안에서 `/model nvidia/nemotron-3-ultra:free`를 사용해 언제든 전환하거나, `/model`을 실행하여 선택기를 열고 목록에서 선택합니다.

## 문제 해결

- **목록에 모델이 보이지 않나요?** Nous Portal 연결을 완료했고 **Free** 요금제를 사용 중인지 확인합니다. CLI에서 `hermes portal info`를 실행하면 로그인되어 있고 Nous를 통해 라우팅되는지 확인할 수 있습니다.
- **잘못된 변형을 선택했나요?** `nvidia/nemotron-3-ultra:free`를 다시 선택합니다 — 무료 티어를 유지하려면 `:free` 접미사가 필요합니다.
- **브라우저가 열리지 않거나 원격 호스트에서 CLI를 사용 중인가요?** 포트 전달 방법은 [OAuth over SSH / 원격 호스트](/guides/oauth-over-ssh)를 참조하세요.

## 함께 보기

- **[데스크톱 앱](/user-guide/desktop)** — macOS, Windows, Linux용 기본 원클릭 앱
- **[Nous Portal로 Hermes Agent 실행하기](/guides/run-hermes-with-nous-portal)** — 모델, Tool Gateway, 확인 절차를 포함한 전체 Portal 안내
- **[Nous Portal 통합](/integrations/nous-portal)** — 구독에 포함된 내용
- **[빠른 시작](/getting-started/quickstart)** — 5분 이내에 설치부터 채팅까지
