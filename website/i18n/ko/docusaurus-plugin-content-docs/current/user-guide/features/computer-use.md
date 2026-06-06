---
title: "컴퓨터 사용 (macOS) (Computer Use (macOS))"
description: "Hermes Agent가 사용자 몰래 백그라운드에서 macOS 데스크탑을 조작하게 만들기"
sidebar_label: "컴퓨터 사용 (macOS)"
sidebar_position: 8
---

# 컴퓨터 사용 (macOS) (Computer Use (macOS))

Hermes Agent는 클릭, 타이핑, 스크롤, 드래그 등 Mac 데스크탑을 **백그라운드**에서 구동할 수 있습니다. 사용자의 마우스 커서가 움직이거나, 키보드 포커스가 바뀌거나, macOS의 스페이스 화면이 전환되는 일은 발생하지 않습니다. 사용자와 에이전트가 같은 기기에서 동시에 작업을 수행할 수 있습니다.

대부분의 컴퓨터-조작 연동 기능들과 달리, 이 기능은 Claude, GPT, Gemini, 또는 로컬 vLLM 엔드포인트 상의 오픈 모델 등 **도구(tool)를 다룰 줄 아는 모델이라면 어떤 것**이든 함께 작동합니다. 특정한 Anthropic 전용 스키마에 신경 쓸 필요가 없습니다.

## 작동 원리 (How it works)

`computer_use` 도구 세트는 stdio를 통해 MCP 통신 방식으로 [`cua-driver`](https://github.com/trycua/cua)에 연결됩니다. 이 드라이버는 macOS 전용으로 설계되었으며, SkyLight 내부의 비공개 기능(`SLEventPostToPid`, `SLPSPostEventRecordTo`)과 `_AXObserverAddNotificationAndCheckRemote` 접근성(accessibility) 인터페이스를 사용하여 다음과 같은 일들을 수행합니다:

- HID 이벤트 변조나 강제적인 커서 위치 변경 없이 대상 프로세스에 직접 이벤트를 주입합니다.
- 창(Window)들을 맨 앞으로 띄우거나 스페이스 화면을 전환시키지 않은 채 AppKit의 활성화 상태(active-state)를 전환시킵니다.
- 창이 가려져 있더라도 Chromium이나 Electron 기반 앱들의 접근성 트리(accessibility trees) 정보들을 생생하게 살려둡니다.

이러한 조합은 OpenAI의 Codex "background computer-use"가 제공하는 기능과 동일합니다. cua-driver는 이에 대응하는 오픈 소스 결과물입니다.

## 활성화 방법 (Enabling)

가장 편한 방법을 고르세요 — 둘 다 동일한 업스트림(upstream) 설치 프로그램을 실행합니다:

**옵션 1: 전용 CLI 명령어 사용 (가장 직접적인 방법).**

```
hermes computer-use install
```

이 명령어는 업스트림 cua-driver 설치 스크립트(`curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh`)를 가져와 실행시킵니다.
설치가 잘 되었는지 확인하려면 `hermes computer-use status`를 치면 됩니다.

**옵션 2: 대화형 방식으로 도구 활성화.**

1. `hermes tools`를 실행하고, `🖱️ Computer Use (macOS)` → `cua-driver (background)`를 선택합니다.
2. 설정 과정에서 업스트림 설치 프로그램이 실행됩니다 (옵션 1과 동일).

어떤 경로를 택했든, 설치 후에는 다음을 수행해야 합니다:

3. 권한 부여 요청이 뜨면 macOS 설정에서 접근을 허용합니다:
   - **시스템 설정 → 개인정보 보호 및 보안 → 손쉬운 사용** → 터미널(또는 Hermes 앱) 허용.
   - **시스템 설정 → 개인정보 보호 및 보안 → 화면 기록** → 동일하게 허용.
4. 해당 도구가 켜진 상태로 세션을 시작합니다:
   ```
   hermes -t computer_use chat
   ```
   또는 `~/.hermes/config.yaml` 파일의 활성화 도구(enabled toolsets) 항목에 `computer_use`를 추가해 둬도 됩니다.

## cua-driver 최신 상태 유지하기 (Keeping cua-driver up to date)

cua-driver 프로젝트는 정기적으로 수정 사항을 배포합니다 (예: v0.1.6 에서는 UTM 사용 흐름에 영향을 끼치던 Safari 창 포커스 버그가 수정되었습니다). 여러분이 낡은 버전 속에 갇히는 일이 없도록 Hermes는 다음 두 가지 지점에서 바이너리 파일들을 갱신시킵니다:

- **`hermes update`** — Hermes 프로그램 자체를 업데이트할 때, PATH 경로상에 `cua-driver`가 존재한다면 업데이트 과정 마지막에 업스트림 설치 프로그램을 다시금 실행합니다. macOS 환경이 아니거나 cua-driver가 깔리지 않은 유저들에게선 동작하지 않습니다.
- **`hermes computer-use install --upgrade`** — 수동 강제 갱신입니다. cua-driver의 설치 여부와 무관하게 업스트림 설치기를 다시 실행합니다. 다음 에이전트 업데이트 패치를 기다릴 필요 없이 가장 최신의 버그 픽스를 적용하고 싶을 때 사용하세요.

`hermes computer-use status` 명령을 통해 바이너리 경로명 옆에 적혀진 버전을 알아볼 수 있습니다.

## 짧은 사용 예시 (Quick example)

사용자의 지시어(Prompt): *"Stripe에서 온 가장 최근 이메일을 찾고, 그들이 나에게 무엇을 요구하는지 요약해 줘."*

에이전트의 계획:

1. `computer_use(action="capture", mode="som", app="Mail")` — 사이드바의 항목들, 툴바 버튼, 편지 행들 하나하나에 일일이 번호(SOM)가 매겨진 Mail 앱의 스크린샷 캡처본을 얻습니다.
2. `computer_use(action="click", element=14)` — (캡처본의 14번에 해당하는) 검색 입력창을 클릭합니다.
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` — 엔터를 치고 새로운 캡처 스크린샷을 확보합니다.
5. 최상단 결과물을 클릭하고 본문을 읽은 뒤 요약합니다.

이 모든 과정 동안 당신의 마우스 커서는 원래 있던 자리에 그대로 머물며, Mail 앱이 맨 앞 화면으로 튀어나오는 일도 일어나지 않습니다.

## 제공자 호환성 (Provider compatibility)

| 모델 제공자 (Provider) | 비전 가능? (Vision?) | 지원 여부? (Works?) | 비고 (Notes) |
|---|---|---|---|
| Anthropic (Claude Sonnet/Opus 3+) | ✅ | ✅ | 종합적으로 최고 수준; SOM + 절대 좌표 모두 처리. |
| OpenRouter (비전 기능 모델들) | ✅ | ✅ | 다중 부분(Multi-part) 도구 메시지가 지원됨. |
| OpenAI (GPT-4+, GPT-5) | ✅ | ✅ | 위와 동일. |
| Local vLLM / LM Studio (비전 모델) | ✅ | ✅ | 해당 모델이 다중 부분 도구의 내용을 소화할 수 있을 시에만. |
| 오직 문자만 인식하는 모델들 | ❌ | ✅ (기능 축소) | 오로지 접근성 트리 구조(accessibility-tree)에서만 일하는 `mode="ax"` 옵션을 켜고 쓰세요. |

스크린샷 이미지들은 도구의 결과물과 함께 OpenAI 스타일의 `image_url` 형태로 보내집니다. Anthropic 모델을 쓸 땐 어댑터가 이를 자체적인 `tool_result` 이미지 블록 형태로 고쳐 보냅니다.

## 안전성 (Safety)

Hermes는 다층적인 방어벽 구조를 가동합니다:

- 파괴적인 기능들(클릭, 타이핑, 스크롤, 드래그, 키 입력, 창 활성화)은 모두 승인을 요합니다 — 터미널상의 대화형 승인 절차를 거치거나 메신저 창에서 승인 버튼을 눌러야 합니다.
- 시스템 도구 단계부터 치명적인 조합키들의 입력을 굳게 막아놓았습니다: 휴지통 비우기, 강제 삭제, 화면 잠금, 로그아웃, 강제 로그아웃.
- 다음의 문자열 타이핑 입력 구조들을 철저히 막습니다: `curl | bash`, `sudo rm -rf /`, 포크 폭탄(fork bombs) 등.
- 에이전트에게 하달된 시스템 기본 명령은 분명하게 규정합니다: 허가 다이얼로그(permission dialogs) 클릭 금지, 비밀번호 직접 타이핑 금지, 스크린샷 속에 파묻혀 숨겨진 지시들 따르지 말 것.

만일 모든 동작에 일일이 승인 도장을 찍고 싶다면 `~/.hermes/config.yaml` 파일 내에 `approvals.mode: manual` 설정을 곁들여 사용하세요.

## 토큰 효율성 (Token efficiency)

스크린샷 전송엔 값비싼 대가가 따릅니다. 때문에 Hermes는 다음 네 단계의 최적화 작업을 실시합니다:

- **스크린샷 밀어내기 (Screenshot eviction)** — Anthropic 어댑터는 오직 가장 최근의 스크린샷 3장만을 남겨놓습니다; 시간이 지난 사진들은 `[screenshot removed to save context]` 란 문구가 적힌 빈 껍데기가 됩니다.
- **클라이언트 측의 압축 가지치기 (Client-side compression pruning)** — 컨텍스트 압축 시스템이 시각 정보들이 오간 결과물들을 감별해 낸 뒤 과거 사진 자료들을 직접 오려냅니다.
- **이미지 최적화 토큰 계산법 (Image-aware token estimation)** — 각 이미지는 그 엄청난 길이의 base64 텍스트 그대로 처리되지 않고 ~1500 토큰(Anthropic 지정 일괄 수치) 수준으로 계산 처리됩니다.
- **서버 측의 대화 내용 지우기 (Server-side context editing (Anthropic 전용))** — Anthropic API가 구형 도구 결과물들을 서버 자체에서 직접 비워버릴 수 있도록 어댑터가 `context_management` 기능을 통해 `clear_tool_uses_20250919`를 활성화시킵니다.

1568×900 크기의 화면에서 20번의 상호작용이 일어난 한 세션의 경우 보통 60만(600K) 토큰이 소모되는 것이 아닌 약 3만(30K) 정도의 스크린샷 토큰만이 지출됩니다.

## 한계 (Limitations)

- **macOS에서만 지원됩니다.** cua-driver가 사용하는 비공개 Apple SPI들은 Linux나 Windows 환경엔 아예 존재하지 않습니다. 여러 OS를 넘나들며 시각적인 창 제어를 원한다면 `browser` 도구를 활용하세요.
- **비공개 시스템 기능 조작에 따른 부작용 (Private SPI risk).** Apple은 어느 순간 OS 업데이트를 내어 이 SkyLight의 심볼 체계들을 확 바꿔버릴 수 있습니다. macOS가 한 차례 엎어진 이후에도 계속 안전하게 구동시키고 싶다면 `HERMES_CUA_DRIVER_VERSION` 환경변수를 통해 현재 드라이버의 버전을 딱 고정해 놓으세요.
- **속도 (Performance).** 백그라운드 운용 방식은 원래 눈에 띄게 돌아가는 방식보다 조금 더 느립니다 — HID 이벤트들을 직접 던져버리는 때와 비교 시 SkyLight를 돌아가는 이벤트들은 약 5~20ms 정도 늦습니다. 마우스 클릭 명령을 내릴 땐 잘 모르지만 스피드-런 기록을 찍어보려 한다면 체감이 될 것입니다.
- **키보드를 통한 비밀번호 수기 입력 불가 (No keyboard password entry).** 커맨드 쉘에 텍스트들을 쳐 넣는 `type` 행위에는 강력한 제약 블록들이 처져 있습니다; 비밀번호를 넣어야 할 땐 시스템에 내장된 자동 입력(autofill) 기능을 사용하세요.

## 설정 (Configuration)

드라이버 바이너리 파일 경로를 재설정 (테스트나 CI 빌드를 위해):

```
HERMES_CUA_DRIVER_CMD=/opt/homebrew/bin/cua-driver
HERMES_CUA_DRIVER_VERSION=0.5.0    # 선택사항으로 버전 고정 가능
```

동작 시스템부 자체를 통째로 끄고 갈아 끼우기 (테스트 목적):

```
HERMES_COMPUTER_USE_BACKEND=noop   # 호출 기록만 남기고, 실제 기기 조작은 하지 않음
```

## 문제 해결 (Troubleshooting)

**`computer_use backend unavailable: cua-driver is not installed`** — `hermes computer-use install`을 쳐서 cua-driver 바이너리를 설치하거나 `hermes tools` 명령어를 통해 Computer Use 툴셋 자체를 켜주세요.

**클릭 기능이 안 먹히는 거 같아요** — 화면 캡처본을 떠보고 살펴보세요. 보지 못했던 어떤 모달(알람이나 설정창)이 마우스 입력을 틀어막고 있을 수도 있습니다. `escape`나 닫기 버튼으로 그것들을 꺼버리세요.

**엘리먼트 요소(Elements) 숫자들이 엉뚱해요** — SOM(번호표기) 숫자들은 바로 다음번 `capture` 스크린샷 화면까지 유효합니다. 무엇이든 상태 변화를 줄 행동을 했다면 매번 화면을 새롭게 다시 캡처하세요.

**"blocked pattern in type text" 가 뜹니다** — 당신이 입력하려 시도한 `type` 문자열 안에 위험 요소로 등재된 쉘 패턴이 포함되어 있습니다. 문자열을 토막 내어 나누든가 아니면 다시 한번 생각해보시길 권합니다.

## 참고 항목 (See also)

- [통용 기술: `macos-computer-use`](https://github.com/NousResearch/hermes-agent/blob/main/skills/apple/macos-computer-use/SKILL.md)
- [cua-driver 소스 코드 (trycua/cua)](https://github.com/trycua/cua)
- 운영체제에 구애받지 않는 웹 전용 조작은 [브라우저 자동화 (Browser automation)](./browser.md) 참고.
