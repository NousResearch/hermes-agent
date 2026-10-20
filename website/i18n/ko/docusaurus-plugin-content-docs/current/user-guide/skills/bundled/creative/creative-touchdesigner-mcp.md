---
title: "TouchDesigner MCP — twozero MCP로 TouchDesigner 제어하기"
sidebar_label: "TouchDesigner MCP"
description: "twozero MCP로 TouchDesigner 제어하기"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# TouchDesigner MCP

twozero MCP로 TouchDesigner를 제어합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본 설치됨) |
| 경로 | `skills/creative/touchdesigner-mcp` |
| 버전 | `1.1.0` |
| 작성자 | kshitijk4poor |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `TouchDesigner`, `MCP`, `twozero`, `creative-coding`, `real-time-visuals`, `generative-art`, `audio-reactive`, `VJ`, `installation`, `GLSL` |
| 관련 스킬 | [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# TouchDesigner 통합 (twozero MCP)

## 중요 규칙

1. **매개변수 이름을 절대 추측하지 마세요.** 먼저 해당 OP 유형에 대해 `td_get_par_info`를 호출하세요. TD 2025.32에서는 학습 데이터가 올바르지 않습니다.
2. **`tdAttributeError`가 발생하면 중지하세요.** 계속하기 전에 실패한 노드에서 `td_get_operator_info`를 호출하세요.
3. **스크립트 콜백에서 절대 절대 경로를 하드코딩하지 마세요.** `me.parent()` / `scriptOp.parent()`를 사용하세요.
4. **`td_execute_python`보다 네이티브 MCP 도구를 우선하세요.** `td_create_operator`, `td_set_operator_pars`, `td_get_errors` 등을 사용하세요. 복잡한 다단계 로직에만 `td_execute_python`으로 대체하세요.
5. **빌드하기 전에 `td_get_hints`를 호출하세요.** 작업 중인 OP 유형에 맞는 패턴을 반환합니다.

## 아키텍처

```
Hermes Agent -> MCP (Streamable HTTP) -> twozero.tox (port 40404) -> TD Python
```

네이티브 도구 36개. 무료 플러그인 (결제/라이선스 없음 — 2026년 4월 확인).
컨텍스트 인식 (선택된 OP와 현재 네트워크를 인식).
허브 상태 확인: `GET http://localhost:40404/mcp`는 인스턴스 PID, 프로젝트 이름, TD 버전이 포함된 JSON을 반환합니다.

## 설정 (자동화)

모든 작업을 처리하려면 설정 스크립트를 실행하세요:

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/skills/creative/touchdesigner-mcp/scripts/setup.sh"
```

스크립트가 수행하는 작업:
1. TD가 실행 중인지 확인
2. 아직 캐시되지 않았다면 twozero.tox 다운로드
3. Hermes 설정에 `twozero_td` MCP 서버 추가 (없는 경우)
4. 포트 40404에서 MCP 연결 테스트
5. 남은 수동 단계 보고 (.tox를 TD로 드래그하고 MCP 토글 활성화)

### 수동 단계 (1회만 필요하며 자동화할 수 없음)

1. **`~/Downloads/twozero.tox`를 TD 네트워크 편집기로 드래그** → 설치 클릭
2. **MCP 활성화:** twozero 아이콘 클릭 → 설정 → mcp → "auto start MCP" → 예
3. **Hermes 세션 재시작**하여 새 MCP 서버 반영

설정 후 다음으로 확인하세요:
```bash
nc -z 127.0.0.1 40404 && echo "twozero MCP: READY"
```

## 환경 참고 사항

- **비상업용 TD**는 해상도를 1280×1280으로 제한합니다. `outputresolution = 'custom'`을 사용하고 너비/높이를 명시적으로 설정하세요.
- **코덱:** `prores` (macOS에서 권장) 또는 대체 코덱으로 `mjpa`. H.264/H.265/AV1에는 Commercial 라이선스가 필요합니다.
- 매개변수를 설정하기 전에 항상 `td_get_par_info`를 호출하세요 — 이름은 TD 버전에 따라 다릅니다 (중요 규칙 #1 참조).

## 워크플로

### 0단계: 탐색 (무엇이든 빌드하기 전에)

```
Call td_get_par_info with op_type for each type you plan to use.
Call td_get_hints with the topic you're building (e.g. "glsl", "audio reactive", "feedback").
Call td_get_focus to see where the user is and what's selected.
Call td_get_network to see what already exists.
```

임시 노드를 만들지 말고 정리도 하지 마세요. 기존의 탐색 절차 전체를 대체합니다.

### 1단계: 정리 + 빌드

**중요: 정리와 생성을 별도의 MCP 호출로 분리하세요.** 하나의 `td_execute_python` 스크립트에서 같은 이름의 노드를 삭제하고 다시 만들면 "Invalid OP object" 오류가 발생합니다. 함정 #11b를 참조하세요.

각 노드에는 `td_create_operator`를 사용하세요 (뷰포트 위치를 자동으로 처리합니다):

```
td_create_operator(type="noiseTOP", parent="/project1", name="bg", parameters={"resolutionw": 1280, "resolutionh": 720})
td_create_operator(type="levelTOP", parent="/project1", name="brightness")
td_create_operator(type="nullTOP", parent="/project1", name="out")
```

대량 생성이나 연결에는 `td_execute_python`을 사용하세요:

```python
# td_execute_python script:
root = op('/project1')
nodes = []
for name, optype in [('bg', noiseTOP), ('fx', levelTOP), ('out', nullTOP)]:
    n = root.create(optype, name)
    nodes.append(n.path)
# Wire chain
for i in range(len(nodes)-1):
    op(nodes[i]).outputConnectors[0].connect(op(nodes[i+1]).inputConnectors[0])
result = {'created': nodes}
```

### 2단계: 매개변수 설정

네이티브 도구를 우선하세요 (매개변수를 검증하며 충돌하지 않습니다):

```
td_set_operator_pars(path="/project1/bg", parameters={"roughness": 0.6, "monochrome": true})
```

표현식이나 모드에는 `td_execute_python`을 사용하세요:

```python
op('/project1/time_driver').par.colorr.expr = "absTime.seconds % 1000.0"
```

### 3단계: 연결

`td_execute_python`을 사용하세요 — 네이티브 연결 도구는 없습니다:

```python
op('/project1/bg').outputConnectors[0].connect(op('/project1/fx').inputConnectors[0])
```

### 4단계: 확인

```
td_get_errors(path="/project1", recursive=true)
td_get_perf()
td_get_operator_info(path="/project1/out", detail="full")
```

### 5단계: 표시 / 캡처

```
td_get_screenshot(path="/project1/out")
```

또는 스크립트로 창을 여세요:

```python
win = op('/project1').create(windowCOMP, 'display')
win.par.winop = op('/project1/out').path
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

## MCP 도구 빠른 참조

**핵심 (주로 사용할 도구):**
| 도구 | 설명 |
|------|------|
| `td_execute_python` | TD에서 임의의 Python 실행. 전체 API에 접근 가능. |
| `td_create_operator` | 매개변수를 사용해 노드 생성 + 자동 배치 |
| `td_set_operator_pars` | 매개변수를 안전하게 설정 (검증하며 충돌하지 않음) |
| `td_get_operator_info` | 단일 노드 검사: 연결, 매개변수, 오류 |
| `td_get_operators_info` | 한 번의 호출로 여러 노드 검사 |
| `td_get_network` | 경로의 네트워크 구조 확인 |
| `td_get_errors` | 재귀적으로 오류/경고 찾기 |
| `td_get_par_info` | OP 유형의 매개변수 이름 가져오기 (탐색 대체) |
| `td_get_hints` | 빌드 전 패턴/팁 가져오기 |
| `td_get_focus` | 열려 있는 네트워크와 선택된 항목 확인 |

**읽기/쓰기:**
| 도구 | 설명 |
|------|------|
| `td_read_dat` | DAT 텍스트 콘텐츠 읽기 |
| `td_write_dat` | DAT 콘텐츠 쓰기/패치 |
| `td_read_chop` | CHOP 채널 값 읽기 |
| `td_read_textport` | TD 콘솔 출력 읽기 |

**시각:**
| 도구 | 설명 |
|------|------|
| `td_get_screenshot` | 한 OP 뷰어를 파일로 캡처 |
| `td_get_screenshots` | 여러 OP를 한 번에 캡처 |
| `td_get_screen_screenshot` | TD를 통해 실제 화면 캡처 |
| `td_navigate_to` | 네트워크 편집기를 OP로 이동 |

**검색:**
| 도구 | 설명 |
|------|------|
| `td_find_op` | 프로젝트 전체에서 이름/유형으로 OP 찾기 |
| `td_search` | 코드, 표현식, 문자열 매개변수 검색 |

**시스템:**
| 도구 | 설명 |
|------|------|
| `td_get_perf` | 성능 프로파일링 (FPS, 느린 OP) |
| `td_list_instances` | 실행 중인 모든 TD 인스턴스 나열 |
| `td_get_docs` | TD 주제에 대한 심층 문서 |
| `td_agents_md` | 컴포넌트별 markdown 문서 읽기/쓰기 |
| `td_reinit_extension` | 코드 편집 후 확장 기능 다시 로드 |
| `td_clear_textport` | 디버그 세션 전에 콘솔 지우기 |

**입력 자동화:**
| 도구 | 설명 |
|------|------|
| `td_input_execute` | TD에 마우스/키보드 입력 보내기 |
| `td_input_status` | 입력 큐 상태 조회 |
| `td_input_clear` | 입력 자동화 중지 |
| `td_op_screen_rect` | 노드의 화면 좌표 가져오기 |
| `td_click_screen_point` | 스크린샷의 한 지점 클릭 |
| `td_screen_point_to_global` | 스크린샷 픽셀을 절대 좌표로 변환 |

위 표에는 일반적인 창작 워크플로에 사용되는 도구 32개가 나와 있습니다. 나머지 4개 도구(`td_project_quit`, `td_test_session`, `td_dev_log`, `td_clear_dev_log`)는 관리자/개발 모드 유틸리티입니다 — 전체 36개 도구 참조와 완전한 매개변수 스키마는 `references/mcp-tools.md`를 참조하세요.

## 주요 구현 규칙

**GLSL 시간:** GLSL TOP에는 `uTDCurrentTime`이 없습니다. Values 페이지를 사용하세요:
```python
# Call td_get_par_info(op_type="glslTOP") first to confirm param names
td_set_operator_pars(path="/project1/shader", parameters={"value0name": "uTime"})
# Then set expression via script:
# op('/project1/shader').par.value0.expr = "absTime.seconds"
# In GLSL: uniform float uTime;
```

대체 방법: `rgba32float` 형식의 Constant TOP (8비트는 0-1로 제한되어 셰이더가 멈춥니다).

**Feedback TOP:** 직접 입력 와이어가 아니라 `top` 매개변수 참조를 사용하세요. "Not enough sources"는 첫 번째 cook 이후 해결됩니다. "Cook dependency loop" 경고는 예상된 동작입니다.

**해상도:** 비상업용은 1280×1280으로 제한됩니다. `outputresolution = 'custom'`을 사용하세요.

**대형 셰이더:** GLSL을 `/tmp/file.glsl`에 쓴 다음 `td_write_dat` 또는 `td_execute_python`을 사용해 로드하세요.

**Vertex/Point 접근 (TD 2025.32):** `point.P[0]`, `point.P[1]`, `point.P[2]` — `.x`, `.y`, `.z`는 사용하지 마세요.

**확장 기능:** CONSTANT 모드에서 `ext0object` 형식은 `"op('./datName').module.ClassName(me)"`입니다. `td_write_dat`로 확장 기능 코드를 편집한 후 `td_reinit_extension`을 호출하세요.

**스크립트 콜백:** `me.parent()` / `scriptOp.parent()`를 사용해 항상 상대 경로를 지정하세요.

**노드 정리:** 반복하기 전에 항상 `list(root.children)`을 사용하고 + `child.valid`를 확인하세요.

## 동영상 녹화 / 내보내기

```python
# via td_execute_python:
root = op('/project1')
rec = root.create(moviefileoutTOP, 'recorder')
op('/project1/out').outputConnectors[0].connect(rec.inputConnectors[0])
rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'prores'  # Apple ProRes — NOT license-restricted on macOS
rec.par.record = True   # start
# rec.par.record = False  # stop (call separately later)
```

H.264/H.265/AV1에는 Commercial 라이선스가 필요합니다. macOS에서는 `prores`를 사용하고, 대체 코덱으로 `mjpa`를 사용하세요.
프레임 추출: `ffmpeg -i /tmp/output.mov -vframes 120 /tmp/frames/frame_%06d.png`

**TOP.save()는 애니메이션에 쓸모가 없습니다** — 매번 같은 GPU 텍스처를 캡처합니다. 항상 MovieFileOut을 사용하세요.

### 녹화 전: 체크리스트

1. `td_get_perf`로 **FPS > 0인지 확인**하세요. FPS=0이면 녹화 파일이 비어 있습니다. 함정 #38-39를 참조하세요.
2. `td_get_screenshot`으로 **셰이더 출력이 검은색이 아닌지 확인**하세요. 검은색 출력 = 셰이더 오류 또는 입력 누락입니다. 함정 #8, #40을 참조하세요.
3. **오디오와 함께 녹화하는 경우:** 먼저 오디오를 시작하도록 큐에 넣고, 그다음 녹화를 3프레임 지연하세요. 함정 #19를 참조하세요.
4. **녹화를 시작하기 전에 출력 경로를 설정**하세요 — 같은 스크립트에서 둘 다 설정하면 경쟁 상태가 발생할 수 있습니다.

## 오디오 반응형 GLSL (검증된 레시피)

### 올바른 신호 체인 (2026년 4월 테스트됨)

```
AudioFileIn CHOP (playmode=sequential)
  → AudioSpectrum CHOP (FFT=512, outputmenu=setmanually, outlength=256, timeslice=ON)
  → Math CHOP (gain=10)
  → CHOP to TOP (dataformat=r, layout=rowscropped)
  → GLSL TOP input 1 (spectrum texture, 256x2)

Constant TOP (rgba32float, time) → GLSL TOP input 0
GLSL TOP → Null TOP → MovieFileOut
```

### 핵심 오디오 반응형 규칙 (경험적으로 검증됨)

1. AudioSpectrum의 **TimeSlice는 반드시 ON으로 유지**하세요. OFF = 전체 오디오 파일을 처리 → 24000+ 샘플 → CHOP to TOP 오버플로.
2. **Output Length를 수동으로 256으로 설정**하세요. `outputmenu='setmanually'`와 `outlength=256`을 사용합니다. 기본 출력은 22050 샘플입니다.
3. **스펙트럼 평활화에 Lag CHOP을 사용하지 마세요.** Lag CHOP은 timeslice 모드에서 작동하며 256개 샘플을 2400개 이상으로 확장하고 모든 값을 거의 0으로 평균 냅니다 (~1e-06). 셰이더는 사용할 수 있는 데이터를 받지 못합니다. 테스트에서 오디오 동기화 실패의 가장 큰 원인이었습니다.
4. **Filter CHOP도 사용하지 마세요** — 스펙트럼 데이터에서 동일한 timeslice 확장 문제가 발생합니다.
5. 필요한 경우 **GLSL 셰이더에서 평활화**하세요. 피드백 텍스처를 사용한 시간적 lerp인 `mix(prevValue, newValue, 0.3)`를 사용할 수 있습니다. 파이프라인 지연 없이 프레임 단위로 정확히 동기화됩니다.
6. **CHOP to TOP dataformat = 'r'**, layout = 'rowscropped'. 스펙트럼 출력은 256x2 (스테레오)입니다. 첫 번째 채널은 y=0.25에서 샘플링하세요.
7. **Math gain = 10** (5가 아님). 원시 스펙트럼 값은 저음 범위에서 약 0.19입니다. gain 10이면 셰이더에서 사용할 수 있는 약 5.0을 얻습니다.
8. **Resample CHOP은 필요하지 않습니다.** AudioSpectrum의 `outlength` 매개변수로 출력 크기를 직접 제어하세요.

### GLSL 스펙트럼 샘플링

```glsl
// Input 0 = time (1x1 rgba32float), Input 1 = spectrum (256x2)
float iTime = texture(sTD2DInputs[0], vec2(0.5)).r;

// Sample multiple points per band and average for stability:
// NOTE: y=0.25 for first channel (stereo texture is 256x2, first row center is 0.25)
float bass = (texture(sTD2DInputs[1], vec2(0.02, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.05, 0.25)).r) / 2.0;
float mid  = (texture(sTD2DInputs[1], vec2(0.2, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.35, 0.25)).r) / 2.0;
float hi   = (texture(sTD2DInputs[1], vec2(0.6, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.8, 0.25)).r) / 2.0;
```

완전한 빌드 스크립트와 셰이더 코드는 `references/network-patterns.md`를 참조하세요.

## 오퍼레이터 빠른 참조

| 계열 | 색상 | Python 클래스 / MCP 유형 | 접미사 |
|--------|-------|-------------|--------|
| TOP | 보라색 | noiseTOP, glslTOP, compositeTOP, levelTop, blurTOP, textTOP, nullTOP | TOP |
| CHOP | 녹색 | audiofileinCHOP, audiospectrumCHOP, mathCHOP, lfoCHOP, constantCHOP | CHOP |
| SOP | 파란색 | gridSOP, sphereSOP, transformSOP, noiseSOP | SOP |
| DAT | 흰색 | textDAT, tableDAT, scriptDAT, webserverDAT | DAT |
| MAT | 노란색 | phongMAT, pbrMAT, glslMAT, constMAT | MAT |
| COMP | 회색 | geometryCOMP, containerCOMP, cameraCOMP, lightCOMP, windowCOMP | COMP |

## 보안 참고 사항

- MCP는 localhost에서만 실행됩니다 (포트 40404). 인증이 없으므로 모든 로컬 프로세스가 명령을 보낼 수 있습니다.
- `td_execute_python`은 TD 프로세스 사용자로서 TD Python 환경과 파일 시스템에 제한 없이 접근합니다.
- `setup.sh`는 공식 404zero.com URL에서 twozero.tox를 다운로드합니다. 우려된다면 다운로드를 확인하세요.
- 이 스킬은 localhost 외부로 데이터를 전송하지 않습니다. 모든 MCP 통신은 로컬에서 이루어집니다.

## 참조 자료

| 파일 | 내용 |
|------|------|
| `references/pitfalls.md` | 실제 세션에서 얻은 검증된 교훈 |
| `references/operators.md` | 모든 오퍼레이터 계열과 매개변수 및 사용 사례 |
| `references/network-patterns.md` | 레시피: 오디오 반응형, 생성형, GLSL, 인스턴싱 |
| `references/mcp-tools.md` | 전체 twozero MCP 도구 매개변수 스키마 |
| `references/python-api.md` | TD Python: op(), 스크립팅, 확장 기능 |
| `references/troubleshooting.md` | 연결 진단, 디버깅 |
| `references/glsl.md` | GLSL 유니폼, 내장 함수, 셰이더 템플릿 |
| `references/postfx.md` | Post-FX: 블룸, CRT, 색수차, 피드백 글로우 |
| `references/layout-compositor.md` | HUD 레이아웃 패턴, 패널 그리드, BSP 스타일 레이아웃 |
| `references/operator-tips.md` | 와이어프레임 렌더링, 피드백 TOP 설정 |
| `references/geometry-comp.md` | Geometry COMP: 인스턴싱, POP와 SOP, 모핑 |
| `references/audio-reactive.md` | 오디오 대역 추출, 비트 감지, 엔벌로프 추적 |
| `references/animation.md` | LFO, 타이머, 키프레임, 이징, 표현식 기반 모션 |
| `references/midi-osc.md` | MIDI/OSC 컨트롤러, TouchOSC, 다중 시스템 동기화 |
| `references/particles.md` | POP 및 레거시 particleSOP — 방출, 힘, 충돌 |
| `references/projection-mapping.md` | 다중 창 출력, 코너 핀, 메시 워프, 에지 블렌딩 |
| `references/external-data.md` | HTTP, WebSocket, MQTT, Serial, TCP, webserverDAT |
| `references/panel-ui.md` | 사용자 지정 매개변수, 패널 COMP, 버튼/슬라이더/필드, panelExecuteDAT |
| `references/replicator.md` | replicatorCOMP — 데이터 기반 복제, 레이아웃, 콜백 |
| `references/dat-scripting.md` | Execute DAT 계열 — chop/dat/parameter/panel/op/executeDAT |
| `references/3d-scene.md` | 조명 리그, 그림자, IBL/큐브맵, 다중 카메라, PBR |
| `scripts/setup.sh` | 자동 설정 스크립트 |

---

> 코드를 작성하는 게 아닙니다. 빛을 지휘하는 겁니다.
