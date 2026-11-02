---
title: 컴퓨터 사용
sidebar_position: 16
---

# 컴퓨터 사용

Hermes Agent는 **macOS, Windows, Linux**에서 **백그라운드**로 데스크톱을
제어할 수 있습니다. 클릭, 입력, 스크롤, 드래그 등을 수행할 수 있습니다.
커서가 움직이지 않고, 키보드 포커스가 바뀌지 않으며, 가상 데스크톱 /
Spaces도 전환되지 않습니다. 사용자와 에이전트가 같은 컴퓨터에서 함께
작업합니다.

대부분의 컴퓨터 사용 통합과 달리, 이 기능은 Claude, GPT, Gemini 또는 로컬
OpenAI 호환 엔드포인트의 오픈 모델 등 **도구를 사용할 수 있는 모든
모델**에서 작동합니다. Anthropic 전용 스키마를 걱정할 필요가 없습니다.

## 작동 방식

`computer_use` 도구 세트는 오픈 소스 백그라운드
컴퓨터 사용 드라이버인 [`cua-driver`](https://github.com/trycua/cua)와
stdio를 통해 MCP로 통신합니다. 각 플랫폼은 내부적으로 적절한 접근성 +
입력 스택을 사용합니다.

| 플랫폼 | 접근성 트리 | 입력 디스패치 |
|---|---|---|
| macOS | AX (비공개 SkyLight SPI) | `SLPSPostEventRecordTo` — pid 범위 지정, 커서 이동 없음 |
| Windows | UIAutomation | `SendInput` + `PostMessage` — 포커스 탈취 없음 |
| Linux | AT-SPI (X11 + Wayland) | XTest (X11) / virtual-keyboard (Wayland) |

모든 플랫폼에서 결과는 같습니다. 에이전트는 표시 중인 모든 창의 접근성
트리를 읽고, 해당 창을 앞으로 가져오거나, 가상 데스크톱을 전환하거나,
실제 OS 커서를 움직이지 않은 채 합성 이벤트를 전송할 수 있습니다.

기반 계약 — *백그라운드 모드가 중요한 이유*, 포그라운드 없음 불변 조건,
클릭 디스패치 내부 동작 — 은
**[cua.ai/docs/explanation/the-no-foreground-contract](https://cua.ai/docs/explanation/the-no-foreground-contract)**를
참조하세요.

## 활성화

**새로 설치하면 드라이버가 이미 포함되어 있습니다.** Hermes 설치 프로그램
(`install.sh` / `install.ps1`)은 `cua-driver`를 사전 설치합니다(최선의
노력 방식이며, `--skip-computer-use` / `-SkipComputerUse`를 전달하면
선택적으로 제외할 수 있습니다). 따라서 Computer Use를 활성화하는 일은
설정 하나만 바꾸면 됩니다.

- **`hermes tools`** → `🖱️  Computer Use` 선택 — 아직 드라이버가 없으면 자동으로 설치합니다.
- **대시보드 / 데스크톱 앱** → Computer Use 도구 세트 전환 — 드라이버가 없으면 전환 시 백그라운드에서 자동으로 설치를 시작합니다(도구 세트 패널에서 진행 상황을 확인하세요).

**수동 대체 방법(이전 설치 또는 설치 단계 건너뛰기):**

```
hermes computer-use install
```

이 명령은 업스트림 cua-driver 설치 프로그램을 가져와 실행합니다 —
macOS/Linux에서는 `install.sh`, Windows에서는 `install.ps1`을 사용합니다.
설치 여부를 확인하려면 `hermes computer-use
status`를 사용하세요.

설치한 후에는 어떤 경로를 사용했는지와 관계없이 플랫폼에 맞는 사전
요구 사항을 허용하세요.

| 플랫폼 | 사전 요구 사항 |
|---|---|
| **macOS** | 시스템 설정 → 개인정보 보호 및 보안 → **손쉬운 사용** + **화면 기록** → 터미널(또는 Hermes 앱)을 허용합니다. `hermes computer-use doctor`가 어떤 권한이 누락되었는지 알려줍니다. |
| **Windows** | 설치 시에는 없습니다. SSH를 통해 제어하는 경우(RDP / 콘솔이 아님), 자동 시작 패턴이 필요합니다 — Session 0 ↔ Session 1+ 프록시는 [cua.ai/docs/how-to-guides/driver/windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh)를 참조하세요. |
| **Linux** | 연결 가능한 디스플레이 서버: X11에는 `DISPLAY` 설정, Wayland에는 `XDG_SESSION_TYPE=wayland` 설정이 필요합니다. Wayland 세션은 캡처를 위해 XWayland 브리지가 필요합니다. AT-SPI는 GNOME/KDE/Xfce에서 기본적으로 활성화되어 있어야 합니다. |

그런 다음 도구 세트를 활성화하여 세션을 시작하세요.

```
hermes -t computer_use chat
```

또는 `~/.hermes/config.yaml`의 활성화된 도구 세트에 `computer_use`를
추가하세요.

## 권한 모드와 로그인된 브라우저 프로필

Hermes는 기존 승인 UX를 cua-driver 0.10의 변경 불가능한 데몬 모드에
매핑합니다. 동기화해야 할 두 번째 권한 전환은 없습니다.

| Hermes 세션 | cua-driver 모드 | 사람의 개입 | `existing_profile` |
|---|---|---|---|
| 수동 또는 스마트 승인(기본값) | `standard` | 일반 Hermes 승인; Cua는 보호 경계에서 중지 | 인증된 보호 호스트가 없으면 거부; Hermes는 현재 그런 호스트를 주장하지 않음 |
| `--yolo`, `/yolo` 또는 `approvals.mode: off` | 비공개 `unrestricted` 데몬 | 명시적인 Hermes 위험 수락 한 번; 런타임 Cua 프롬프트 없음 | Cua에 내장된 정책, 관리 정책, 사용자 정책의 한도 내에서 허용 |

unrestricted 데몬은 해당 Hermes 세션에만 비공개로 사용됩니다. `/yolo`를
끄거나, 세션을 재설정/닫거나, 취소 정리를 수행하거나, 프로세스가 종료되면
Cua 세션이 끝나고 데몬이 중지됩니다. 시스템 전체 데몬의 모드를 변경하지
않으며, 다른 Hermes 대화에 동일한 권한을 부여하지도 않습니다.

`smart` 승인은 `standard`로 유지됩니다. LLM 분류는 사람의 보호된 동의가
아니기 때문입니다. Cua의 `bounded` 매니페스트 모드도 스마트 승인이나
일반 도구 확인에서 추론되지 않습니다. 정확한 매니페스트를 검토하고
실행하는 별도의 신뢰된 호스트가 필요합니다.

<div class="alert alert--warning">

YOLO/unrestricted 모드는 프롬프트 인젝션이나 의도하지 않은 입력을
방어하지 않습니다. 일회용 VM에서 사용하거나, 완전히 침해되어도 감수할 수
있는 계정과 데이터에만 사용하세요.

</div>

## `hermes computer-use doctor` — 첫 번째 문제 분류 지점

`hermes computer-use doctor`는 cua-driver의 구조화된
`health_report` MCP 도구를 실행하고 검사별 행렬을 출력합니다. 어떤 작업이
작동하지 않는 **이유**를 알아내는 가장 빠른 방법입니다.

```
$ hermes computer-use doctor
⚠️  cua-driver 0.5.8 on darwin — degraded
  ✅ binary_version: cua-driver 0.5.8
  ✅ platform_supported: macOS 26.4.1 (arm64)
  ✅ session_active: MCP session is active.
  ❌ bundle_identity: Process has no CFBundleIdentifier.
      → Run the binary inside CuaDriver.app so TCC grants attribute correctly.
  ✅ tcc_accessibility: Accessibility is granted.
  ✅ tcc_screen_recording: Screen Recording is granted.
  ✅ ax_capability: AX is trusted and reachable.
  ✅ screen_capture_capability: ScreenCaptureKit reachable; 1 display(s) shareable.
```

- 전체 상태가 `ok`이면 **종료 코드 0** — 모든 연결이 완료되었습니다.
- `degraded` 또는 `failed`이면 **종료 코드 1** — 하나 이상의 검사가 실패했습니다. 각 실패 항목의 힌트에서 수정 방법을 알려줍니다.
- cua-driver 바이너리 자체에 연결할 수 없으면 **종료 코드 2**입니다.

유용한 플래그:

- `--include CHECK` — 나열한 검사만 실행합니다(여러 개를 반복 지정 가능).
- `--skip CHECK` — 검사를 건너뜁니다(`--include`보다 우선).
- `--json` — 원시 구조화 페이로드를 출력합니다. `tools/call health_report` MCP 응답과 같은 형태입니다.

검사 행렬은 플랫폼을 인식합니다. `bundle_identity` / `tcc_*`는 해당
개념이 적용되지 않는 Windows + Linux에서 `skip`입니다.
`ax_capability`는 macOS에서 AX, Windows에서 UIA, Linux에서 AT-SPI를
검사하며, 연결할 수 없을 때 각각에 맞는 진단 힌트를 제공합니다.

## 에이전트 커서와 세션

에이전트가 동작하면 각 클릭 / 입력 / 스크롤이 도착하는 위치까지 **색조가
입혀진 오버레이 커서**가 미끄러지듯 이동합니다. 실제 OS 커서는 움직이지
않습니다. 오버레이는 "에이전트가 여기서 동작 중"임을 알려주는 시각적
신호입니다. 각 Hermes 실행은 자체 cua-driver **세션 ID**(예:
`hermes-3a7b9c14d2e8`)를 선언합니다. 커서의 정체성은 이 세션에 연결되므로,
동시에 실행되는 여러 실행 / 하위 에이전트도 서로의 커서를 건드리지 않고
각자의 커서를 사용할 수 있습니다.

커서는 `cua-driver`의 CLI 플래그 또는 런타임
`set_agent_cursor_style` MCP 도구로 조정할 수 있습니다. 전체 메뉴(내장
`arrow`와 `teardrop` 실루엣, `--cursor-icon`을 통한 사용자 지정 SVG /
PNG / ICO, 런타임 그라데이션 색상, bloom halo)는
[cua.ai/docs/how-to-guides/driver/personalize-cursor](https://cua.ai/docs/how-to-guides/driver/personalize-cursor)를
참조하세요.

## 더 깊이 알아보기 — cua-driver 스킬 팩

Hermes는 자체 스킬(`skills/autonomous-ai-agents/computer-use/SKILL.md`)을
Hermes 측 `computer_use` 동작 어휘에 집중시킵니다. 에이전트가 로드하는
단일 정보 출처입니다. 플랫폼별 심화 설명, 녹화 의미론, 브라우저 페이지
상호작용 등 더 깊은 내용을 보려면 cua-driver 팀이 직접 제공하고 유지하는
cua-driver 스킬 팩을 에이전트 하니스에 설치하세요.

```
cua-driver skills install
```

이 명령은 팩을 에이전트 하니스의 스킬 디렉터리에 심볼릭 링크합니다.
실행하면 에이전트가 다음 항목에 접근할 수 있습니다.

| 파일 | 주제 |
|---|---|
| `SKILL.md` | 크로스 플랫폼 핵심(스냅샷 불변 조건, 포그라운드 없음 계약, 클릭 디스패치, AX 트리 메커니즘) |
| `MACOS.md` | macOS 세부 사항: 포그라운드 없음 계약, AXMenuBar 탐색, SkyLight 클릭 디스패치, Apple Events JS 브리지 |
| `WINDOWS.md` | Windows 세부 사항: UIA 트리, UWP / `ApplicationFrameHost` 호스팅, Session 0 격리, 자동 시작 패턴 |
| `LINUX.md` | Linux 세부 사항: AT-SPI 트리, X11 / Wayland, 터미널 에뮬레이터 감지 |
| `RECORDING.md` | 궤적 + 동영상 녹화 의미론 |
| `WEB_APPS.md` | 브라우저 페이지 상호작용 팁 |
| `TESTS.md` | 궤적에 의한 재생 워크플로 |

이 항목들은 Hermes 스킬의 중복이 아니라 **플랫폼별 심화 설명**입니다.
에이전트가 "Windows에서 클릭이 잘못된 요소에 도착했다"고 보고하면,
UIA / UWP 맥락을 확인하여 원인과 다르게 처리해야 하는 방법을 이해하기
위해 `WINDOWS.md`를 읽습니다.

Hermes가 설치되어 있는 에이전트 하니스 자동 감지는 아직
`trycua/cua`의 후속 작업으로 계획되어 있습니다. 그때까지는
`cua-driver skills install`을 한 번 실행하고, 생성된
`~/.cua-driver/skills/cua-driver` 디렉터리를 하니스에 지정하거나 평소
사용하는 스킬 공간에 심볼릭 링크하세요.

## 간단한 예

사용자 프롬프트: *"Stripe에서 온 가장 최근 이메일을 찾아서 내가 무엇을
해야 하는지 요약해 줘."*

에이전트의 계획(이 형태는 macOS / Windows / Linux에서 동일하며, 모델은
플랫폼에 맞는 관용적인 단축키와 앱 이름으로 대체합니다)은 다음과
같습니다.

1. `computer_use(action="capture", mode="som", app="Mail")` — 모든 사이드바 항목, 도구 모음 버튼, 메시지 행에 번호를 매긴 이메일 앱의 스크린샷을 가져옵니다.
2. `computer_use(action="click", element=14)` — 검색 필드를 클릭합니다.
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` — 제출하고 새 스크린샷을 가져옵니다.
5. 상단 결과를 클릭하고 본문을 읽은 뒤 요약합니다.

이 모든 과정에서 커서는 사용자가 둔 위치에 그대로 있고 이메일 앱은
앞으로 나오지 않습니다.

## 제공자 호환성

| 제공자 | 비전? | 작동? | 참고 |
|---|---|---|---|
| Anthropic (Claude Sonnet/Opus 3+) | ✅ | ✅ | 전반적으로 가장 좋음; SOM + 원시 좌표. |
| OpenRouter (모든 비전 모델) | ✅ | ✅ | 다중 파트 도구 메시지 지원. |
| OpenAI (GPT-4+, GPT-5) | ✅ | ✅ | 위와 동일. |
| Google (Gemini 2+) | ✅ | ✅ | 도구 호출 + 비전 모두 지원. |
| 로컬 vLLM / LM Studio / Ollama (비전 모델) | ✅ | ✅ | 모델이 다중 파트 도구 콘텐츠를 지원하는 경우. |
| 텍스트 전용 모델 | ❌ | ✅ (성능 저하) | 접근성 트리만 사용하는 작업에는 `mode="ax"`를 사용하세요. |

스크린샷은 OpenAI 스타일 `image_url` 파트로 도구 결과에 인라인으로
전송됩니다. Anthropic의 경우 어댑터가 이를 네이티브 `tool_result`
이미지 블록으로 변환합니다. 이미지 MIME 타입은 cua-driver의 명시적인
`mimeType` 필드(`image/png` 또는 `image/jpeg`)에서 가져옵니다 — 클라이언트
측 매직 바이트 스니핑은 없습니다.

## 안전

Hermes는 여러 계층의 보호 장치를 적용합니다.

- 파괴적 작업(클릭, 입력, 드래그, 스크롤, 키, focus_app)은 승인이 필요합니다 — CLI 대화상자에서 직접 승인하거나 메시징 플랫폼의 승인 버튼을 사용할 수 있습니다.
- 도구 수준에서 강제 차단되는 키 조합: 휴지통 비우기, 강제 삭제, 화면 잠금, 로그아웃, 강제 로그아웃.
- 강제 차단되는 입력 패턴: `curl | bash`, `sudo rm -rf /`, fork bomb 등.
- 에이전트의 시스템 프롬프트는 권한 대화상자를 클릭하지 않고, 비밀번호를 입력하지 않으며, 스크린샷에 삽입된 지시를 따르지 말라고 명시합니다.

모든 작업을 확인하고 싶다면 `~/.hermes/config.yaml`에서
`approvals.mode: manual`과 함께 사용하세요.

## 토큰 효율

스크린샷은 비용이 큽니다. Hermes는 네 계층의 최적화를 적용합니다.

- **스크린샷 제거** — Anthropic 어댑터는 컨텍스트에 가장 최근 스크린샷 3개만 유지하며, 이전 항목은 `[screenshot removed]` 자리 표시자가 됩니다.
- **클라이언트 측 압축 가지치기** — 컨텍스트 압축기가 멀티모달 도구 결과를 감지하고 이전 결과에서 이미지 파트를 제거합니다.
- **이미지 인식 토큰 추정** — 각 이미지는 base64 문자 길이가 아니라 약 1500 토큰(Anthropic의 정액 요금)으로 계산됩니다.
- **서버 측 컨텍스트 편집(Anthropic만 해당)** — 활성화되면 어댑터가 `context_management`를 통해 `clear_tool_uses_20250919`를 활성화하여 Anthropic API가 서버 측에서 오래된 도구 결과를 지우도록 합니다.

1568×900 디스플레이에서 20개 작업을 수행하는 세션은 일반적으로 스크린샷
컨텍스트에 약 30K 토큰이 들며, 약 600K 토큰이 들지 않습니다.

## 제한 사항

- **성능.** 백그라운드 모드는 포그라운드보다 느립니다 — 접근성으로 라우팅되는 이벤트는 직접 HID를 전송하는 것과 비교해 macOS에서 약 5–20ms, Windows UIA에서 약 3–10ms, Linux AT-SPI에서 약 5–15ms가 걸립니다. 에이전트 속도의 클릭에서는 눈에 띄지 않지만, 속도 주행을 녹화하려 하면 눈에 띕니다.
- **키보드로 비밀번호 입력 불가.** `type`에는 명령 셸 페이로드에 대한 차단 패턴이 있습니다. 비밀번호에는 시스템 자동 입력(macOS Keychain / Windows Credential Manager / GNOME Keyring / KWallet)을 사용하세요.
- **일부 앱은 접근성 트리를 노출하지 않습니다.** Windows의 최신 UWP 앱, Linux의 Electron < 28, 사용자 지정 그리기를 사용하는 일부 macOS 앱(Logic, Final Cut, 일부 게임)은 AX 트리가 드물거나 비어 있습니다. 트리가 비어 있으면 픽셀 좌표로 대체하거나 작업 전체를 건너뛰세요.
- **Windows: 관리자 권한 창은 일반 에이전트에서 제어할 수 없습니다.** Windows UIPI(User Interface Privilege Isolation)는 무결성 수준 경계를 적용합니다. 즉, Medium 무결성 프로세스(기본 Hermes 에이전트)는 High 무결성(Administrator) 프로세스가 소유한 창의 UIA 트리를 열거하거나 해당 창에 마우스 입력을 주입할 수 없습니다. 증상은 스크린샷은 정상적으로 렌더링되는데도(GDI 캡처는 무결성 검사보다 하위에서 동작함) `capture(mode='som')`이 요소 0개를 반환하고 `click(...)`이 아무것도 하지 않으면서 성공을 보고하는 것입니다. 키보드 이벤트는 UIPI를 부분적으로 우회하므로 Tab / Enter로 관리자 권한 대화상자를 계속 탐색할 수 있습니다. 이는 OS 제약이며 cua-driver 버그가 아닙니다 — 모든 Windows 자동화 스택에 영향을 줍니다. 관리자 권한 창을 제어하려면 Hermes 에이전트 자체를 높은 무결성으로 실행하세요(권한 상승 터미널에서 실행). 그렇지 않으면 관리자 권한이 아닌 창을 대상으로 하세요.
- **플랫폼별 배포 시 주의 사항:**
  - **macOS**는 비공개 SkyLight SPI를 사용합니다. Apple은 OS 업데이트에서 이를 변경할 수 있습니다. Hermes는 설치된 cua-driver가 테스트한 버전보다 오래되면 경고합니다.
  - **Windows** SSH 세션은 대화형 데스크톱이 없는 **Session 0**에서 실행됩니다. RDP / 콘솔 세션 내부에서 Hermes를 구동하거나 cua-driver의 자동 시작 예약 작업을 설정하세요 — [windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh)에 방법이 있습니다.
  - **Linux**에는 연결 가능한 디스플레이 서버가 필요합니다. 헤드리스 서버는 `computer_use`가 이벤트를 캡처하거나 주입하기 전에 Xvfb(`Xvfb :99 -screen 0 1920x1080x24`)가 필요합니다. 순수 Wayland 세션은 화면 캡처를 위해 XWayland 브리지가 필요합니다(cua-driver의 Wayland 입력 경로는 입력을 독립적으로 처리합니다).

데스크톱 오버헤드 없이(그리고 TCC / Session 0 / X11 설정 없이) 크로스
플랫폼 GUI 자동화를 수행하려면 `browser` 도구 세트를 사용하세요. 웹 전용
작업에는 실제 헤드리스 Chromium을 사용하므로 올바른 선택입니다.

## 구성

드라이버 바이너리 경로를 재정의합니다(테스트 / CI / 로컬 빌드).

```
HERMES_CUA_DRIVER_CMD=/path/to/your/cua-driver
```

백엔드를 통째로 교체합니다(테스트용).

```
HERMES_COMPUTER_USE_BACKEND=noop   # records calls, no side effects
```

### 텔레메트리

cua-driver는 업스트림에서 익명 사용 텔레메트리(PostHog)를 기본적으로
활성화한 상태로 제공됩니다. **Hermes는 이를 대신 비활성화합니다** — 모든
cua-driver 호출(MCP 백엔드, `status`, `doctor`, 설치)에서 Hermes는 드라이버
환경에 `CUA_DRIVER_RS_TELEMETRY_ENABLED=0`을 설정합니다.

다시 선택하려면(cua-driver가 자체 기본값을 사용하고 텔레메트리를 보내도록
허용하려면) `config.yaml`에 다음을 설정하세요.

```yaml
computer_use:
  cua_telemetry: true   # default: false (telemetry off)
```

활성화하면 `hermes computer-use doctor`가 `telemetry: enabled`를
보고하고, 비활성화하면(기본값) `telemetry: disabled via
CUA_DRIVER_RS_TELEMETRY_ENABLED`를 보고합니다.

## 로컬 cua-driver 빌드 테스트

cua-driver 자체를 개발하거나 아직 출시되지 않은 수정 사항을 테스트하려면
게시된 릴리스 대신 소스에서 빌드한 바이너리를 사용하도록 Hermes를
지정하세요. Hermes는 `shutil.which("cua-driver")`로 드라이버를 확인하며
`HERMES_CUA_DRIVER_VERSION`을 **강제하지 않습니다**. 따라서
`0.0.0-local-*`로 보고되는 로컬 빌드는 그대로 허용됩니다. 두 가지 방법이
있습니다.

### 방법 A — `install-local`(빌드 + PATH에 배치)

`trycua/cua` 체크아웃에서 업스트림 로컬 설치 프로그램을 실행하세요.
릴리스 모드로 Rust 백엔드를 빌드하고 `cua-driver`를 운영 환경 설치
프로그램과 동일한 설치 레이아웃에 배치하며, bin 디렉터리를 PATH에
추가합니다.

```powershell
# Windows (PowerShell), from the cua repo root
./libs/cua-driver/scripts/install-local.ps1 -NoAutoStart
```

```bash
# macOS / Linux, from the cua repo root  (defaults to a debug build without --release)
./libs/cua-driver/scripts/install-local.sh --release
```

- Windows는 빌드를 `%USERPROFILE%\.cua-driver\packages\…` 아래에 임시 배치하고 `%LOCALAPPDATA%\Programs\Cua\cua-driver\bin`을 여기에 연결하는 junction을 만듭니다(User PATH에 추가됨). macOS/Linux는 `cua-driver`를 `~/.local/bin`에 심볼릭 링크합니다(`--bin-dir <path>`로 재정의 가능).
- `-NoAutoStart`는 `cua-driver-serve` 로그온 데몬 등록을 건너뜁니다 — Hermes 테스트에는 필요하지 않습니다(참고 사항 참조).

그런 다음 새 셸을 열어 PATH 변경 사항이 보이게 하고 다음을 확인하세요.

```
cua-driver --version                 # local builds report 0.0.0-local-release
# Windows:      (Get-Command cua-driver).Source
# macOS/Linux:  which cua-driver
```

### 방법 B — Hermes가 빌드된 바이너리를 바로 가리키도록 설정(가장 빠른 반복)

설치 절차를 완전히 건너뛰세요. `cargo build`를 실행하고
`HERMES_CUA_DRIVER_CMD`를 결과 바이너리로 설정합니다. 빠른
편집/빌드/테스트에 가장 적합합니다.

```bash
cargo build -p cua-driver            # add --release for a release build; run from libs/cua-driver/rust
```

```
# Windows (.env)
HERMES_CUA_DRIVER_CMD=C:\path\to\cua\libs\cua-driver\rust\target\debug\cua-driver.exe
# macOS / Linux (.env)
HERMES_CUA_DRIVER_CMD=/path/to/cua/libs/cua-driver/rust/target/debug/cua-driver
```

### Hermes가 빌드를 사용 중인지 확인

- `hermes computer-use status`는 확인된 바이너리 경로와 버전을 출력합니다.
- `hermes computer-use doctor`는 바이너리에 연결할 수 있는지 확인하고 전체 MCP 경로를 처음부터 끝까지 실행합니다.
- 세션에서 `computer_use(action="capture")`는 생성된 `cua-driver mcp` 자식 프로세스를 실행합니다.

### 참고 사항 및 주의점

- **Hermes는 `cua-driver mcp` stdio 프록시를 생성합니다.** 일반 세션에서 프록시는 표준 시스템 데몬에 연결하고(필요하면 시작할 수 있음), 명시적 Hermes YOLO에서는 Hermes가 비공개 `cua-driver serve --embedded` 자식을 소유하고 프록시를 비공개 소켓 또는 명명된 파이프로 연결합니다. SSH에서 대화형 Session 1+ 입력을 위해서는 Windows 자동 시작/UIAccess 패턴도 여전히 중요합니다 — 제한 사항 섹션을 참조하세요.
- **Windows에서 바이너리가 잠김.** 실행 중인 `cua-driver-serve` 데몬이 `cua-driver.exe`를 점유하여 빌드 시 덮어쓰기를 막을 수 있습니다. `install-local.ps1`은 잠긴 바이너리의 이름을 자동으로 변경합니다. 수동으로 `cargo build`하는 경우(방법 B) 먼저 `cua-driver autostart disable`(또는 `schtasks /End /TN cua-driver-serve`)으로 중지하세요.
- **재빌드 반복.** cua-driver 소스를 편집한 후 방법 A에서는 `install-local`을 다시 실행하고(빌드, 다시 배치, `current` junction 전환), 방법 B에서는 `cargo build`만 다시 실행하면 됩니다 — 어느 경우에도 Hermes를 변경할 필요는 없습니다.
- **로컬 빌드는 버전 검사를 건너뜁니다.** Hermes는 설치된 cua-driver가 OS별 테스트 기준선보다 오래되면 경고하지만, `0.0.0-local-*` 개발 빌드는 제외하므로 로컬 빌드에서는 해당 경고가 발생하지 않습니다.

## 문제 해결

**무언가 이상할 때 첫 번째 조치: `hermes computer-use doctor`를
실행하세요.** 검사별 구조화 행렬에서 무엇이 잘못되었는지 정확히
알려주므로, 디버깅을 돕는 에이전트도 같은 정보를 확인할 수 있습니다.

doctor가 잡아내지 못하는 특정 실패 모드:

**`computer_use backend unavailable: cua-driver is not installed`** —
`hermes computer-use install`을 실행하여 cua-driver 바이너리를 가져오거나,
`hermes tools`를 실행하고 Computer Use 도구 세트를 활성화하세요.

**클릭해도 아무 효과가 없는 것처럼 보임** — 캡처하여 확인하세요. 보지
못한 모달 창이 입력을 차단하고 있을 수 있습니다. `escape` 또는 닫기
버튼으로 닫으세요.

**요소 인덱스가 오래됨** — SOM 인덱스는 다음 `capture` 전까지만 유효합니다.
상태를 변경하는 작업을 수행한 후 다시 캡처하세요. 래퍼는 오래된 요소를
감지하기 위해 불투명한 `element_token`을 전달하므로, 잘못된 클릭 대신
명시적인 오류가 표시됩니다.

**"blocked pattern in type text"** — `type`에 입력하려 한 텍스트가
위험한 셸 패턴 목록과 일치합니다. 명령을 나누거나 다시 검토하세요.

**Linux에서 캡처가 비어 있음** — `DISPLAY`가 설정되지 않았거나 XWayland
브리지 없는 순수 Wayland를 사용 중입니다. `hermes computer-use doctor`가
`ax_capability: fail`과 `Set DISPLAY (X11)…` 힌트를 표시합니다.

**SSH를 통한 Windows에서 캡처가 비어 있음** — 서비스 세션인 Session 0에
있습니다. RDP / 콘솔에서 직접 구동하거나 자동 시작 패턴을 설정하세요 —
[cua.ai/docs/how-to-guides/driver/windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh)를
참조하세요.

## 함께 보기

- **Hermes 측 스킬** — `skills/autonomous-ai-agents/computer-use/SKILL.md` — Hermes `computer_use` 동작 어휘를 가르칩니다. 에이전트가 로드하는 항목입니다.
- **cua-driver 스킬 팩** — 플랫폼별 심화 설명(macOS 포그라운드 없음 계약, Windows UIA + Session 0, Linux AT-SPI + X11/Wayland, 녹화, 브라우저 페이지)을 보려면 `cua-driver skills install`을 실행하고 `MACOS.md` / `WINDOWS.md` / `LINUX.md` / `RECORDING.md` / `WEB_APPS.md`를 읽으세요. cua-driver의 `skills install`이 Hermes를 자동 감지하게 되면(계획된 후속 작업), 설치 시 자동으로 처리됩니다.
- **cua.ai/docs** — cua-driver 프로젝트의 문서:
  - [컴퓨터 사용이란?](https://cua.ai/docs/explanation/what-is-computer-use) — 개념 소개
  - [포그라운드 없음 계약](https://cua.ai/docs/explanation/the-no-foreground-contract) — *백그라운드 모드가 중요한 이유*
  - [설치 참고](https://cua.ai/docs/how-to-guides/driver/install) — 크로스 플랫폼 설치 세부 사항
  - [에이전트 커서 맞춤 설정](https://cua.ai/docs/how-to-guides/driver/personalize-cursor) — 내장 형태, 사용자 지정 에셋, 런타임 재정의
  - [SSH를 통해 Windows 제어](https://cua.ai/docs/how-to-guides/driver/windows-ssh) — Session 0 → Session 1+ 자동 시작 패턴
  - [cua-driver 계속 실행](https://cua.ai/docs/how-to-guides/driver/keep-running) — 자동 시작 / 데몬 수명 주기
  - [에이전트 연결](https://cua.ai/docs/how-to-guides/driver/connect-your-agent) — 다양한 하니스(Hermes 포함)에 cua-driver 등록
- [cua-driver 소스(trycua/cua)](https://github.com/trycua/cua)
- 네이티브 앱을 제어할 필요가 없는 크로스 플랫폼 웹 작업은 [브라우저 자동화](./browser.md)를 참조하세요.
