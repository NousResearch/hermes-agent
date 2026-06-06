---
title: "Touchdesigner Mcp"
sidebar_label: "Touchdesigner Mcp"
description: "twozero MCP를 통해 실행 중인 TouchDesigner 인스턴스 제어 — 오퍼레이터 생성, 매개변수 설정, 와이어 연결, 파이썬 실행, 실시간 비주얼 구축"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Touchdesigner Mcp

twozero MCP를 통해 실행 중인 TouchDesigner 인스턴스 제어 — 오퍼레이터 생성, 매개변수 설정, 와이어 연결, 파이썬 실행, 실시간 비주얼 구축. 36개의 기본 도구 포함.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/creative/touchdesigner-mcp` |
| Version | `1.1.0` |
| Author | kshitijk4poor |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `TouchDesigner`, `MCP`, `twozero`, `creative-coding`, `real-time-visuals`, `generative-art`, `audio-reactive`, `VJ`, `installation`, `GLSL` |
| Related skills | [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp), [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video), [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video), `hermes-video` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# TouchDesigner 통합 (twozero MCP)

## 핵심 규칙 (CRITICAL RULES)

1. **절대 파라미터 이름을 추측하지 마세요.** 가장 먼저 오퍼레이터(op) 유형에 대해 `td_get_par_info`를 호출하세요. TD 2025.32의 경우 당신의 훈련 데이터는 잘못되었습니다.
2. **`tdAttributeError`가 발생하면 중단하세요.** 계속 진행하기 전에 실패한 노드에 대해 `td_get_operator_info`를 호출하세요.
3. **스크립트 콜백에 절대 경로를 하드코딩하지 마세요.** `me.parent()` / `scriptOp.parent()`를 사용하세요.
4. **`td_execute_python`보다 네이티브 MCP 도구를 선호하세요.** `td_create_operator`, `td_set_operator_pars`, `td_get_errors` 등을 사용하세요. 복잡한 다단계 논리를 위해 차선책으로만 `td_execute_python`을 사용하세요.
5. **빌드하기 전에 `td_get_hints`를 호출하세요.** 이것은 작업하려는 op 유형에 특화된 패턴을 반환합니다.

## 아키텍처 (Architecture)

```
Hermes Agent -> MCP (스트리밍 가능 HTTP) -> twozero.tox (포트 40404) -> TD Python
```

36개의 기본 도구. 무료 플러그인 (결제/라이선스 없음 — 2026년 4월 확인).
컨텍스트 인식 (선택된 OP, 현재 네트워크를 앎).
허브 상태 확인: `GET http://localhost:40404/mcp` 는 인스턴스 PID, 프로젝트 이름, TD 버전을 포함한 JSON을 반환합니다.

## 설정 (자동화됨)

설정 스크립트를 실행하여 모든 것을 처리합니다:

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/skills/creative/touchdesigner-mcp/scripts/setup.sh"
```

이 스크립트는 다음을 수행합니다:
1. TD가 실행 중인지 확인
2. 캐시되지 않은 경우 twozero.tox 다운로드
3. Hermes config에 `twozero_td` MCP 서버 추가 (없는 경우)
4. 포트 40404에서 MCP 연결 테스트
5. 남은 수동 단계가 무엇인지 보고 (TD로 .tox 드래그, MCP 토글 켜기)

### 수동 단계 (일회성, 자동화할 수 없음)

1. **`~/Downloads/twozero.tox`를 TD 네트워크 편집기로 드래그** → Install 클릭
2. **MCP 활성화:** twozero 아이콘 클릭 → Settings → mcp → "auto start MCP" → Yes
3. **Hermes 세션 재시작**하여 새로운 MCP 서버를 인식

설정 후 확인:
```bash
nc -z 127.0.0.1 40404 && echo "twozero MCP: READY"
```

## 환경 참고 사항

- **비상업용 TD**는 해상도를 1280×1280으로 제한합니다. `outputresolution = 'custom'`을 사용하고 너비/높이를 명시적으로 설정하세요.
- **코덱:** `prores` (macOS 권장) 또는 대체품으로 `mjpa`. H.264/H.265/AV1은 상업용 라이선스가 필요합니다.
- 파라미터를 설정하기 전에 항상 `td_get_par_info`를 호출하세요 — 이름은 TD 버전에 따라 다릅니다 (핵심 규칙 #1 참조).

## 워크플로우

### 0단계: 발견 (Discover) (무엇이든 빌드하기 전)

```
사용하려는 각 유형에 대해 op_type과 함께 td_get_par_info를 호출하세요.
빌드하려는 주제(예: "glsl", "audio reactive", "feedback")와 함께 td_get_hints를 호출하세요.
사용자의 위치와 선택된 항목을 보려면 td_get_focus를 호출하세요.
이미 존재하는 것을 보려면 td_get_network를 호출하세요.
```

임시 노드 없음, 정리 없음. 이것은 이전의 발견 단계를 완전히 대체합니다.

### 1단계: 정리 + 빌드 (Clean + Build)

**중요: 정리(cleanup)와 생성(creation)을 개별 MCP 호출로 분리하세요.** 하나의 `td_execute_python` 스크립트에서 같은 이름의 노드를 파괴하고 다시 생성하면 "Invalid OP object" 오류가 발생합니다. 문제점(pitfalls) #11b를 참조하세요.

각 노드에 대해 `td_create_operator`를 사용하세요 (뷰포트 위치 지정을 자동으로 처리합니다):

```
td_create_operator(type="noiseTOP", parent="/project1", name="bg", parameters={"resolutionw": 1280, "resolutionh": 720})
td_create_operator(type="levelTOP", parent="/project1", name="brightness")
td_create_operator(type="nullTOP", parent="/project1", name="out")
```

대량 생성 또는 연결에는 `td_execute_python`을 사용하세요:

```python
# td_execute_python 스크립트:
root = op('/project1')
nodes = []
for name, optype in [('bg', noiseTOP), ('fx', levelTOP), ('out', nullTOP)]:
    n = root.create(optype, name)
    nodes.append(n.path)
# 와이어 체인
for i in range(len(nodes)-1):
    op(nodes[i]).outputConnectors[0].connect(op(nodes[i+1]).inputConnectors[0])
result = {'created': nodes}
```

### 2단계: 매개변수 설정 (Set Parameters)

기본 도구를 우선 사용하세요 (매개변수 유효성 검사, 충돌 없음):

```
td_set_operator_pars(path="/project1/bg", parameters={"roughness": 0.6, "monochrome": true})
```

표현식이나 모드의 경우 `td_execute_python`을 사용하세요:

```python
op('/project1/time_driver').par.colorr.expr = "absTime.seconds % 1000.0"
```

### 3단계: 와이어 연결 (Wire)

`td_execute_python`을 사용하세요 — 연결을 위한 기본 도구가 없습니다:

```python
op('/project1/bg').outputConnectors[0].connect(op('/project1/fx').inputConnectors[0])
```

### 4단계: 검증 (Verify)

```
td_get_errors(path="/project1", recursive=true)
td_get_perf()
td_get_operator_info(path="/project1/out", detail="full")
```

### 5단계: 표시 / 캡처 (Display / Capture)

```
td_get_screenshot(path="/project1/out")
```

또는 스크립트를 통해 창을 엽니다:

```python
win = op('/project1').create(windowCOMP, 'display')
win.par.winop = op('/project1/out').path
win.par.winw = 1280; win.par.winh = 720
win.par.winopen.pulse()
```

## MCP 도구 빠른 참조

**핵심 도구 (가장 많이 사용):**
| 도구 | 설명 |
|------|------|
| `td_execute_python` | TD에서 임의의 Python 실행. 전체 API 액세스. |
| `td_create_operator` | 파라미터 + 자동 위치 지정으로 노드 생성 |
| `td_set_operator_pars` | 파라미터를 안전하게 설정 (유효성 검사, 충돌 없음) |
| `td_get_operator_info` | 단일 노드 검사: 연결, 파라미터, 오류 |
| `td_get_operators_info` | 한 번의 호출로 여러 노드 검사 |
| `td_get_network` | 특정 경로의 네트워크 구조 확인 |
| `td_get_errors` | 재귀적으로 오류/경고 찾기 |
| `td_get_par_info` | OP 유형에 대한 파라미터 이름 가져오기 (발견 대체) |
| `td_get_hints` | 빌드 전 패턴/팁 가져오기 |
| `td_get_focus` | 어떤 네트워크가 열려 있는지, 무엇이 선택되었는지 |

**읽기/쓰기 (Read/Write):**
| 도구 | 설명 |
|------|------|
| `td_read_dat` | DAT 텍스트 내용 읽기 |
| `td_write_dat` | DAT 내용 쓰기/패치 |
| `td_read_chop` | CHOP 채널 값 읽기 |
| `td_read_textport` | TD 콘솔 출력 읽기 |

**시각 도구 (Visual):**
| 도구 | 설명 |
|------|------|
| `td_get_screenshot` | 하나의 OP 뷰어를 파일로 캡처 |
| `td_get_screenshots` | 한 번에 여러 OP 캡처 |
| `td_get_screen_screenshot` | TD를 통해 실제 화면 캡처 |
| `td_navigate_to` | 네트워크 편집기를 특정 OP로 점프 |

**검색 (Search):**
| 도구 | 설명 |
|------|------|
| `td_find_op` | 프로젝트 전체에서 이름/유형별로 op 찾기 |
| `td_search` | 코드, 표현식, 문자열 파라미터 검색 |

**시스템 (System):**
| 도구 | 설명 |
|------|------|
| `td_get_perf` | 성능 프로파일링 (FPS, 느린 op) |
| `td_list_instances` | 실행 중인 모든 TD 인스턴스 나열 |
| `td_get_docs` | TD 주제에 대한 심층 문서 |
| `td_agents_md` | COMP별 마크다운 문서 읽기/쓰기 |
| `td_reinit_extension` | 코드 편집 후 확장(extension) 다시 로드 |
| `td_clear_textport` | 디버그 세션 전에 콘솔 지우기 |

**입력 자동화 (Input Automation):**
| 도구 | 설명 |
|------|------|
| `td_input_execute` | 마우스/키보드를 TD로 전송 |
| `td_input_status` | 입력 대기열 상태 폴링 |
| `td_input_clear` | 입력 자동화 중지 |
| `td_op_screen_rect` | 노드의 화면 좌표 가져오기 |
| `td_click_screen_point` | 스크린샷 내의 점 클릭 |
| `td_screen_point_to_global` | 스크린샷 픽셀을 절대 화면 좌표로 변환 |

위의 표는 일반적인 창작 워크플로우에서 사용되는 32개의 도구를 다룹니다. 나머지 4개 도구(`td_project_quit`, `td_test_session`, `td_dev_log`, `td_clear_dev_log`)는 관리자/개발자 모드 유틸리티입니다. 전체 매개변수 스키마를 포함한 36개 도구에 대한 전체 참조는 `references/mcp-tools.md`를 참조하세요.

## 핵심 구현 규칙

**GLSL 시간:** GLSL TOP에 `uTDCurrentTime`은 없습니다. Values 페이지를 사용하세요:
```python
# 파라미터 이름을 확인하기 위해 먼저 td_get_par_info(op_type="glslTOP")를 호출합니다.
td_set_operator_pars(path="/project1/shader", parameters={"value0name": "uTime"})
# 그런 다음 스크립트를 통해 표현식을 설정합니다:
# op('/project1/shader').par.value0.expr = "absTime.seconds"
# GLSL에서: uniform float uTime;
```

대체 방안: `rgba32float` 형식의 Constant TOP (8비트는 0-1로 고정되어 셰이더가 멈춥니다).

**Feedback TOP:** 직접 입력으로 연결하지 말고 `top` 매개변수 참조를 사용하세요. "Not enough sources"는 첫 번째 cook 이후 해결됩니다. "Cook dependency loop" 경고는 정상입니다.

**해상도:** 비상업용(Non-Commercial) 버전은 1280×1280으로 제한됩니다. `outputresolution = 'custom'`을 사용하세요.

**대용량 셰이더:** GLSL을 `/tmp/file.glsl`에 작성한 다음, `td_write_dat` 또는 `td_execute_python`을 사용하여 로드합니다.

**버텍스/포인트 접근 (TD 2025.32):** `point.P[0]`, `point.P[1]`, `point.P[2]` — `.x`, `.y`, `.z`가 아닙니다.

**확장(Extensions):** `ext0object` 형식은 CONSTANT 모드에서 `"op('./datName').module.ClassName(me)"` 입니다. `td_write_dat`로 확장 코드를 편집한 후, `td_reinit_extension`을 호출하세요.

**스크립트 콜백:** 항상 `me.parent()` / `scriptOp.parent()`를 통해 상대 경로를 사용하세요.

**노드 정리:** 항상 순회(iterate)하기 전에 `list(root.children)` + `child.valid` 검사를 수행하세요.

## 녹화 / 비디오 내보내기 (Recording / Exporting Video)

```python
# td_execute_python을 통해:
root = op('/project1')
rec = root.create(moviefileoutTOP, 'recorder')
op('/project1/out').outputConnectors[0].connect(rec.inputConnectors[0])
rec.par.type = 'movie'
rec.par.file = '/tmp/output.mov'
rec.par.videocodec = 'prores'  # Apple ProRes — macOS에서 라이선스 제한이 없음
rec.par.record = True   # 시작
# rec.par.record = False  # 정지 (나중에 별도로 호출)
```

H.264/H.265/AV1은 상업용(Commercial) 라이선스가 필요합니다. macOS에서는 `prores`를 사용하거나 대체품으로 `mjpa`를 사용하세요.
프레임 추출: `ffmpeg -i /tmp/output.mov -vframes 120 /tmp/frames/frame_%06d.png`

**TOP.save()는 애니메이션에 쓸모가 없습니다** — 매번 동일한 GPU 텍스처를 캡처합니다. 항상 MovieFileOut을 사용하세요.

### 녹화 전: 체크리스트

1. `td_get_perf`를 통해 **FPS가 0보다 큰지 확인하세요.** FPS가 0이면 녹화가 비어 있게 됩니다. 문제점(pitfalls) #38-39를 참조하세요.
2. `td_get_screenshot`을 통해 **셰이더 출력이 검은색이 아닌지 확인하세요.** 검은색 출력 = 셰이더 오류 또는 누락된 입력. 문제점 #8, #40을 참조하세요.
3. **오디오와 함께 녹화하는 경우:** 오디오가 먼저 시작되도록 대기한 다음, 녹화를 3프레임 지연시킵니다. 문제점 #19를 참조하세요.
4. **녹화를 시작하기 전에 출력 경로를 설정하세요** — 동일한 스크립트에서 둘 다 설정하면 경합(race)이 발생할 수 있습니다.

## 오디오 반응형 GLSL (검증된 레시피)

### 올바른 신호 체인 (2026년 4월 테스트 완료)

```
AudioFileIn CHOP (playmode=sequential)
  → AudioSpectrum CHOP (FFT=512, outputmenu=setmanually, outlength=256, timeslice=ON)
  → Math CHOP (gain=10)
  → CHOP to TOP (dataformat=r, layout=rowscropped)
  → GLSL TOP input 1 (스펙트럼 텍스처, 256x2)

Constant TOP (rgba32float, time) → GLSL TOP input 0
GLSL TOP → Null TOP → MovieFileOut
```

### 중요한 오디오 반응형 규칙 (경험적으로 검증됨)

1. AudioSpectrum에 대해 **TimeSlice는 ON 상태를 유지해야 합니다.** OFF = 전체 오디오 파일을 처리함 → 24000개 이상의 샘플 → CHOP to TOP 오버플로우.
2. `outputmenu='setmanually'` 및 `outlength=256`을 통해 **출력 길이(Output Length)를 256으로 수동 설정합니다.** 기본값은 22050 샘플을 출력합니다.
3. **스펙트럼 평활화(smoothing)에 Lag CHOP을 절대 사용하지 마세요.** Lag CHOP은 타임슬라이스 모드에서 작동하여 256개 샘플을 2400개 이상으로 확장하고, 모든 값을 거의 0(~1e-06)에 가깝게 평균화합니다. 셰이더는 사용 가능한 데이터를 받지 못합니다. 이것은 테스트에서 오디오 동기화 실패의 주요 원인이었습니다.
4. **Filter CHOP도 사용하지 마세요** — 스펙트럼 데이터에서 동일한 타임슬라이스 확장 문제가 발생합니다.
5. 필요한 경우 **평활화는 피드백 텍스처를 사용한 시간적 보간(temporal lerp)을 통해 GLSL 셰이더에서 수행해야 합니다:** `mix(prevValue, newValue, 0.3)`. 이렇게 하면 파이프라인 지연 시간이 0인 프레임 단위의 완벽한 동기화가 제공됩니다.
6. **CHOP to TOP dataformat = 'r'**, layout = 'rowscropped'. 스펙트럼 출력은 256x2 (스테레오)입니다. 첫 번째 채널에 대해 y=0.25에서 샘플링합니다.
7. **Math gain = 10** (5가 아님). 원시 스펙트럼 값은 저음(bass) 범위에서 약 0.19입니다. 10의 게인은 셰이더에 사용 가능한 약 5.0을 제공합니다.
8. **Resample CHOP이 필요하지 않습니다.** AudioSpectrum의 `outlength` 파라미터를 통해 출력 크기를 직접 제어하세요.

### GLSL 스펙트럼 샘플링

```glsl
// Input 0 = time (1x1 rgba32float), Input 1 = spectrum (256x2)
float iTime = texture(sTD2DInputs[0], vec2(0.5)).r;

// 안정을 위해 밴드당 여러 지점을 샘플링하고 평균화:
// 참고: 첫 번째 채널의 경우 y=0.25 (스테레오 텍스처는 256x2이고, 첫 번째 행의 중심은 0.25임)
float bass = (texture(sTD2DInputs[1], vec2(0.02, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.05, 0.25)).r) / 2.0;
float mid  = (texture(sTD2DInputs[1], vec2(0.2, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.35, 0.25)).r) / 2.0;
float hi   = (texture(sTD2DInputs[1], vec2(0.6, 0.25)).r +
              texture(sTD2DInputs[1], vec2(0.8, 0.25)).r) / 2.0;
```

전체 빌드 스크립트 + 셰이더 코드는 `references/network-patterns.md`를 참조하세요.

## 오퍼레이터 빠른 참조

| 제품군 (Family) | 색상 | Python 클래스 / MCP 유형 | 접미사 (Suffix) |
|--------|-------|-------------|--------|
| TOP | 보라색 | noiseTOP, glslTOP, compositeTOP, levelTop, blurTOP, textTOP, nullTOP | TOP |
| CHOP | 녹색 | audiofileinCHOP, audiospectrumCHOP, mathCHOP, lfoCHOP, constantCHOP | CHOP |
| SOP | 파란색 | gridSOP, sphereSOP, transformSOP, noiseSOP | SOP |
| DAT | 흰색 | textDAT, tableDAT, scriptDAT, webserverDAT | DAT |
| MAT | 노란색 | phongMAT, pbrMAT, glslMAT, constMAT | MAT |
| COMP | 회색 | geometryCOMP, containerCOMP, cameraCOMP, lightCOMP, windowCOMP | COMP |

## 보안 참고 사항

- MCP는 로컬 호스트(포트 40404)에서만 실행됩니다. 인증이 없으므로 모든 로컬 프로세스가 명령을 보낼 수 있습니다.
- `td_execute_python`은 TD 프로세스 사용자 권한으로 TD Python 환경 및 파일 시스템에 무제한 액세스할 수 있습니다.
- `setup.sh`는 공식 404zero.com URL에서 twozero.tox를 다운로드합니다. 우려되는 경우 다운로드된 내용을 확인하세요.
- 이 스킬은 절대로 로컬 호스트 외부로 데이터를 보내지 않습니다. 모든 MCP 통신은 로컬입니다.

## 참조 문서

| 파일 | 내용 |
|------|------|
| `references/pitfalls.md` | 실제 세션에서 얻은 귀중한 교훈 |
| `references/operators.md` | 파라미터 및 사용 사례가 있는 모든 오퍼레이터 제품군 |
| `references/network-patterns.md` | 레시피: 오디오 반응형, 제너레이티브, GLSL, 인스턴싱 |
| `references/mcp-tools.md` | 전체 twozero MCP 도구 파라미터 스키마 |
| `references/python-api.md` | TD Python: op(), 스크립팅, 확장(extensions) |
| `references/troubleshooting.md` | 연결 진단, 디버깅 |
| `references/glsl.md` | GLSL 유니폼, 내장 함수, 셰이더 템플릿 |
| `references/postfx.md` | Post-FX: 블룸, CRT, 색수차(chromatic aberration), 피드백 글로우 |
| `references/layout-compositor.md` | HUD 레이아웃 패턴, 패널 그리드, BSP 스타일 레이아웃 |
| `references/operator-tips.md` | 와이어프레임 렌더링, 피드백 TOP 설정 |
| `references/geometry-comp.md` | Geometry COMP: 인스턴싱, POP vs SOP, 모핑 |
| `references/audio-reactive.md` | 오디오 밴드 추출, 비트 감지, 엔벨로프 팔로워 |
| `references/animation.md` | LFO, 타이머, 키프레임, 이징(easing), 표현식 구동 모션 |
| `references/midi-osc.md` | MIDI/OSC 컨트롤러, TouchOSC, 다중 장비 동기화 |
| `references/particles.md` | POP 및 레거시 particleSOP — 방출, 힘(forces), 충돌 |
| `references/projection-mapping.md` | 다중 창 출력, 코너 핀, 메시 워프, 엣지 블렌딩 |
| `references/external-data.md` | HTTP, WebSocket, MQTT, Serial, TCP, webserverDAT |
| `references/panel-ui.md` | 사용자 정의 파라미터, panel COMP, 버튼/슬라이더/필드, panelExecuteDAT |
| `references/replicator.md` | replicatorCOMP — 데이터 기반 클로닝, 레이아웃, 콜백 |
| `references/dat-scripting.md` | Execute DAT 제품군 — chop/dat/parameter/panel/op/executeDAT |
| `references/3d-scene.md` | 조명 장비, 그림자, IBL/큐브맵, 다중 카메라, PBR |
| `scripts/setup.sh` | 자동 설정 스크립트 |

---

> 당신은 코드를 작성하는 것이 아닙니다. 빛을 지휘하는 것입니다.
