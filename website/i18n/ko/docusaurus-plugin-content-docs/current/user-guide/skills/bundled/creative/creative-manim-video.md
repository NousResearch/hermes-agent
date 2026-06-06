---
title: "Manim Video — Manim CE 애니메이션: 3Blue1Brown 수학/알고리즘 비디오"
sidebar_label: "Manim Video"
description: "Manim CE 애니메이션: 3Blue1Brown 수학/알고리즘 비디오"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py 스크립트에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Manim Video

Manim CE 애니메이션: 3Blue1Brown 수학/알고리즘 비디오.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/creative/manim-video` |
| Version | `1.0.0` |
| Platforms | linux, macos, windows |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 것입니다.
:::

# Manim 비디오 제작 파이프라인

## 언제 사용해야 하나

사용자가 다음과 같은 것을 요청할 때 사용합니다: 애니메이션이 포함된 설명, 수학 애니메이션, 개념 시각화, 알고리즘 연습, 기술적 설명 영상, 3Blue1Brown 스타일 비디오, 또는 기하학적/수학적 콘텐츠가 포함된 모든 프로그래밍 방식 애니메이션. Manim Community Edition을 사용하여 3Blue1Brown 스타일의 설명 비디오, 알고리즘 시각화, 방정식 유도, 아키텍처 다이어그램 및 데이터 스토리를 만듭니다.

## 크리에이티브 기준 (Creative Standard)

이것은 교육용 시네마(educational cinema)입니다. 모든 프레임이 가르침을 줍니다. 모든 애니메이션이 구조를 보여줍니다.

**코드를 단 한 줄이라도 작성하기 전에** 내러티브 아크(narrative arc)를 명확히 하세요. 이것이 어떤 오해를 바로잡아 줄 수 있을까요? "아하" 하는 깨달음의 순간은 언제입니까? 어떤 시각적 스토리가 시청자를 혼란에서 이해로 이끌어 줄까요? 사용자의 프롬프트는 시작점일 뿐입니다 — 교육적인 야망을 가지고 이를 해석하세요.

**대수학 전에 기하학을 먼저 보여주세요.** 도형을 먼저 보여주고, 수식은 그 다음에 보여주세요. 시각적 기억은 기호 기억보다 더 빨리 암호화됩니다. 시청자가 공식 이전에 기하학적 패턴을 보게 되면, 방정식은 더 와닿게 됩니다.

**첫 번째 렌더링에서 탁월함을 보여주는 것은 절대 타협할 수 없는 조건입니다.** 결과물은 수정 과정 없이 시각적으로 명확하고 미적으로 일관되어야 합니다. 어수선해 보이거나, 타이밍이 맞지 않거나, "AI가 만든 슬라이드"처럼 보인다면, 그것은 잘못된 것입니다.

**불투명도 레이어링(Opacity layering)은 주의를 집중시킵니다.** 절대로 모든 것을 최대 밝기로 보여주지 마세요. 기본 요소는 1.0, 맥락적 요소는 0.4, 구조적 요소(축, 격자)는 0.15로 설정하세요. 뇌는 시각적 중요도(salience)를 여러 층으로 나눠서 처리합니다.

**여유 공간(Breathing room).** 모든 애니메이션 뒤에는 `self.wait()`가 필요합니다. 시청자는 방금 나타난 것을 흡수할 시간이 필요합니다. 애니메이션에서 다른 애니메이션으로 서둘러 넘어가지 마세요. 주요 내용이 나타난 후의 2초간의 멈춤은 결코 낭비되는 시간이 아닙니다.

**응집력 있는 시각적 언어.** 모든 씬(scene)은 색상 팔레트, 일관된 타이포그래피 크기, 일치하는 애니메이션 속도를 공유합니다. 모든 씬에서 무작위로 다른 색상을 사용하는 영상은 기술적으로 완벽하더라도 미학적 실패입니다.

## 전제 조건

모든 종속성을 확인하려면 `scripts/setup.sh`를 실행하세요. 요구 사항: Python 3.10+, Manim Community Edition v0.20+ (`pip install manim`), LaTeX (Linux에서는 `texlive-full`, macOS에서는 `mactex`), 및 ffmpeg. 참조 문서는 Manim CE v0.20.1을 기준으로 테스트되었습니다.

## 모드 (Modes)

| 모드 | 입력 | 출력 | 참조 |
|------|-------|--------|-----------|
| **개념 설명 (Concept explainer)** | 주제/개념 | 기하학적 직관이 포함된 애니메이션 설명 | `references/scene-planning.md` |
| **방정식 유도 (Equation derivation)** | 수학 표현식 | 단계별 애니메이션 증명 | `references/equations.md` |
| **알고리즘 시각화 (Algorithm visualization)** | 알고리즘 설명 | 데이터 구조와 함께 단계별 실행 | `references/graphs-and-data.md` |
| **데이터 스토리 (Data story)** | 데이터/메트릭 | 애니메이션화된 차트, 비교, 카운터 | `references/graphs-and-data.md` |
| **아키텍처 다이어그램 (Architecture diagram)** | 시스템 설명 | 연결을 빌드하는 컴포넌트들 | `references/mobjects.md` |
| **논문 해설 (Paper explainer)** | 연구 논문 | 주요 결과 및 방법에 대한 애니메이션 | `references/scene-planning.md` |
| **3D 시각화 (3D visualization)** | 3D 개념 | 회전하는 표면, 매개변수 곡선, 공간 기하학 | `references/camera-and-3d.md` |

## 스택 (Stack)

프로젝트당 단일 파이썬 스크립트를 사용합니다. 브라우저, Node.js, GPU가 필요하지 않습니다.

| 계층 | 도구 | 목적 |
|-------|------|---------|
| 코어 (Core) | Manim Community Edition | 씬 렌더링, 애니메이션 엔진 |
| 수학 (Math) | LaTeX (texlive/MiKTeX) | `MathTex`를 통한 방정식 렌더링 |
| 비디오 I/O | ffmpeg | 씬 이어붙이기, 형식 변환, 오디오 먹싱(muxing) |
| TTS | ElevenLabs / Qwen3-TTS (선택 사항) | 내레이션 음성 안내(voiceover) |

## 파이프라인

```
계획(PLAN) --> 코드(CODE) --> 렌더링(RENDER) --> 이어붙이기(STITCH) --> 오디오(AUDIO - 선택) --> 검토(REVIEW)
```

1. **계획(PLAN)** — 내러티브 아크, 씬 목록, 시각적 요소, 색상 팔레트, 보이스오버 대본이 포함된 `plan.md` 작성
2. **코드(CODE)** — 독립적으로 렌더링 가능한 클래스가 씬당 하나씩 포함된 `script.py` 작성
3. **렌더링(RENDER)** — 초안용은 `manim -ql script.py Scene1 Scene2 ...`, 최종 프로덕션용은 `-qh`
4. **이어붙이기(STITCH)** — 씬 클립을 ffmpeg concat을 사용하여 `final.mp4`로 결합
5. **오디오(AUDIO)** (선택) — ffmpeg를 통해 보이스오버 및/또는 배경 음악 추가. `references/rendering.md` 참조
6. **검토(REVIEW)** — 미리보기 스틸 컷(preview stills) 렌더링, 계획과 대조 확인, 조정

## 프로젝트 구조

```
project-name/
  plan.md                # 내러티브 아크, 씬 분류
  script.py              # 모든 씬이 들어있는 단일 파일
  concat.txt             # ffmpeg 씬 목록
  final.mp4              # 이어 붙인 최종 출력
  media/                 # Manim에서 자동 생성됨
    videos/script/480p15/
```

## 크리에이티브 디렉션

### 색상 팔레트

| 팔레트 | 배경 (Background) | 1차 색상 (Primary) | 2차 색상 (Secondary) | 강조 색상 (Accent) | 사용 사례 |
|---------|-----------|---------|-----------|--------|----------|
| **클래식 3B1B (Classic 3B1B)** | `#1C1C1C` | `#58C4DD` (BLUE) | `#83C167` (GREEN) | `#FFFF00` (YELLOW) | 일반적인 수학/CS |
| **따뜻한 학술용 (Warm academic)** | `#2D2B55` | `#FF6B6B` | `#FFD93D` | `#6BCB77` | 접근하기 쉬운 |
| **네온 테크 (Neon tech)** | `#0A0A0A` | `#00F5FF` | `#FF00FF` | `#39FF14` | 시스템, 아키텍처 |
| **모노크롬 (Monochrome)** | `#1A1A2E` | `#EAEAEA` | `#888888` | `#FFFFFF` | 미니멀리스트 |

### 애니메이션 속도

| 문맥 | run_time | 애니메이션 후 self.wait() |
|---------|----------|-------------------|
| 제목/인트로 등장 | 1.5s | 1.0s |
| 주요 방정식 공개 | 2.0s | 2.0s |
| 변환/모핑(morph) | 1.5s | 1.5s |
| 보조 라벨 | 0.8s | 0.5s |
| 화면 정리 (FadeOut) | 0.5s | 0.3s |
| "아하" 하는 깨달음의 순간 공개 | 2.5s | 3.0s |

### 타이포그래피 스케일

| 역할 | 폰트 크기 | 용도 |
|------|-----------|-------|
| 제목 (Title) | 48 | 씬 제목, 시작 텍스트 |
| 헤딩 (Heading) | 36 | 씬 내의 섹션 헤더 |
| 본문 (Body) | 30 | 설명 텍스트 |
| 라벨 (Label) | 24 | 주석, 축 라벨 |
| 캡션 (Caption) | 20 | 자막, 작은 글씨 |

### 폰트

**모든 텍스트에 고정폭(monospace) 폰트를 사용하세요.** Manim의 Pango 렌더러는 가변폭(proportional) 폰트를 사용할 경우 모든 크기에서 커닝(kerning) 문제가 발생합니다. 전체 권장 사항은 `references/visual-design.md`를 참조하세요.

```python
MONO = "Menlo"  # 파일의 상단에 한 번 정의

Text("Fourier Series", font_size=48, font=MONO, weight=BOLD)  # 제목들
Text("n=1: sin(x)", font_size=20, font=MONO)                  # 라벨들
MathTex(r"\nabla L")                                            # 수학 기호 (LaTeX 사용)
```

가독성을 위해 최소 `font_size=18`을 유지하세요.

### 씬(Scene)별 변화

모든 씬에 동일한 구성을 사용하지 마세요. 각 씬에 대해 다음을 고려하세요:
- 팔레트에서 **다른 주요 색상** 사용
- **다른 레이아웃** — 항상 모든 것을 가운데 정렬하지 마세요.
- **다른 애니메이션 등장 방식** — Write, FadeIn, GrowFromCenter, Create를 번갈아 사용하세요.
- **다른 시각적 무게감** — 일부 씬은 밀도 있게, 다른 씬은 여유롭게 구성하세요.

## 워크플로우

### 1단계: 계획 (plan.md)

코드를 작성하기 전에 `plan.md`를 작성하세요. 포괄적인 템플릿은 `references/scene-planning.md`를 참조하세요.

### 2단계: 코드 (script.py)

씬 하나당 하나의 클래스를 만듭니다. 모든 씬은 독립적으로 렌더링할 수 있습니다.

```python
from manim import *

BG = "#1C1C1C"
PRIMARY = "#58C4DD"
SECONDARY = "#83C167"
ACCENT = "#FFFF00"
MONO = "Menlo"

class Scene1_Introduction(Scene):
    def construct(self):
        self.camera.background_color = BG
        title = Text("Why Does This Work?", font_size=48, color=PRIMARY, weight=BOLD, font=MONO)
        self.add_subcaption("Why does this work?", duration=2)
        self.play(Write(title), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), run_time=0.5)
```

핵심 패턴:
- 모든 애니메이션의 **자막**: `self.add_subcaption("text", duration=N)` 또는 `self.play()`의 `subcaption="text"`
- 여러 씬에 걸친 일관성을 위해 파일 상단의 **공유 색상 상수** 사용
- 모든 씬에 설정된 **`self.camera.background_color`**
- **깔끔한 퇴장** — 씬 마지막에 모든 mobject를 FadeOut: `self.play(FadeOut(Group(*self.mobjects)))`

### 3단계: 렌더링

```bash
manim -ql script.py Scene1_Introduction Scene2_CoreConcept  # 초안
manim -qh script.py Scene1_Introduction Scene2_CoreConcept  # 프로덕션
```

### 4단계: 이어붙이기 (Stitch)

```bash
cat > concat.txt << 'EOF'
file 'media/videos/script/480p15/Scene1_Introduction.mp4'
file 'media/videos/script/480p15/Scene2_CoreConcept.mp4'
EOF
ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

### 5단계: 검토

```bash
manim -ql --format=png -s script.py Scene2_CoreConcept  # 미리보기 스틸 컷
```

## 핵심 구현 참고 사항

### LaTeX용 원시 문자열 (Raw Strings)
```python
# 잘못된 예: MathTex("\frac{1}{2}")
# 올바른 예:
MathTex(r"\frac{1}{2}")
```

### 가장자리 텍스트(Edge Text)를 위한 buff >= 0.5 설정
```python
label.to_edge(DOWN, buff=0.5)  # 0.5 미만은 절대 사용 금지
```

### 텍스트를 교체하기 전에 FadeOut
```python
self.play(ReplacementTransform(note1, note2))  # 그 위에 Write(note2) 하지 않음
```

### 화면에 추가되지 않은 Mobject는 절대로 애니메이션 하지 마세요
```python
self.play(Create(circle))  # 반드시 먼저 화면에 추가(add) 해야 함
self.play(circle.animate.set_color(RED))  # 그런 다음 애니메이션 적용
```

## 성능 목표 (Performance Targets)

| 품질 | 해상도 | FPS | 속도 |
|---------|-----------|-----|-------|
| `-ql` (초안) | 854x480 | 15 | 씬당 5-15초 |
| `-qm` (중간) | 1280x720 | 30 | 씬당 15-60초 |
| `-qh` (프로덕션) | 1920x1080 | 60 | 씬당 30-120초 |

항상 `-ql` 해상도에서 반복 작업 하세요. 최종 출력시에만 `-qh`로 렌더링 하세요.

## 참조 문서 (References)

| 파일 | 내용 |
|------|----------|
| `references/animations.md` | 핵심 애니메이션, 속도 함수, 구성(composition), `.animate` 구문, 타이밍 패턴 |
| `references/mobjects.md` | Text, shapes, VGroup/Group, 위치 지정, 스타일링, 커스텀 mobject |
| `references/visual-design.md` | 12가지 디자인 원칙, 불투명도 레이어링, 레이아웃 템플릿, 색상 팔레트 |
| `references/equations.md` | Manim에서의 LaTeX, TransformMatchingTex, 방정식 유도 패턴 |
| `references/graphs-and-data.md` | Axes, 플로팅(plotting), BarChart, 애니메이션 데이터, 알고리즘 시각화 |
| `references/camera-and-3d.md` | MovingCameraScene, ThreeDScene, 3D 곡면, 카메라 제어 |
| `references/scene-planning.md` | 내러티브 아크, 레이아웃 템플릿, 씬 전환, 계획 템플릿 |
| `references/rendering.md` | CLI 참조, 품질 사전 설정(presets), ffmpeg, 보이스오버 워크플로우, GIF 내보내기 |
| `references/troubleshooting.md` | LaTeX 오류, 애니메이션 오류, 흔히 하는 실수, 디버깅 |
| `references/animation-design-thinking.md` | 애니메이션을 쓸 때와 정적인 이미지를 보여줄 때, 분해(decomposition), 페이싱, 내레이션 동기화 |
| `references/updaters-and-trackers.md` | ValueTracker, add_updater, always_redraw, 시간 기반 업데이터, 패턴 |
| `references/paper-explainer.md` | 연구 논문을 애니메이션으로 바꾸기 — 워크플로우, 템플릿, 도메인 패턴 |
| `references/decorations.md` | SurroundingRectangle, Brace, 화살표, DashedLine, Angle, 주석의 라이프사이클 |
| `references/production-quality.md` | 코드 작성 전, 렌더링 전, 렌더링 후 체크리스트, 공간 레이아웃, 색상, 템포 |

---

## 크리에이티브 방향 전환 (Creative Divergence) - 사용자가 실험적/창조적/독특한 결과를 요구할 때만 사용하세요.

사용자가 창의적이고, 실험적이거나 기존과는 다른 설명 방식을 요구할 경우, 애니메이션을 설계하기 **전에** 전략을 선택하고 충분한 사고 과정을 거치세요.

- **SCAMPER** — 사용자가 기존에 알고 있는 표준적인 설명에 신선한 방식을 원할 때
- **가정 뒤집기 (Assumption Reversal)** — 사용자가 특정 주제를 가르치는 전통적인 방식에 이의를 제기하고 싶을 때

### SCAMPER 변환
표준적인 수학적/기술적 시각화를 가져와 다음과 같이 변환합니다:
- **대체 (Substitute)**: 표준적인 시각적 은유를 교체합니다 (예: 수직선 → 구불구불한 길, 행렬 → 도시 격자).
- **결합 (Combine)**: 두 가지 설명 방식을 동시에 융합합니다 (예: 대수적 설명 + 기하학적 설명 동시 진행).
- **역방향 (Reverse)**: 역으로 추론합니다 — 결과에서 시작하여 기본 원리(axioms)까지 해체해 나갑니다.
- **수정 (Modify)**: 파라미터를 과장하여 왜 이것이 중요한지 보여줍니다 (예: 학습률 10배, 샘플 크기 1000배).
- **제거 (Eliminate)**: 모든 표기법을 완전히 제거합니다 — 오직 애니메이션과 공간적 관계만으로 설명합니다.

### 가정 뒤집기 (Assumption Reversal)
1. 이 주제가 시각화되는 방식 중 "표준"으로 간주되는 것들을 나열합니다 (예: 왼쪽에서 오른쪽으로, 2D, 개별 단계, 공식 표기법).
2. 가장 근본적인 가정을 하나 고릅니다.
3. 이를 뒤집습니다 (예: 오른쪽에서 왼쪽으로 유도, 2D 개념의 3D화, 단계별이 아닌 연속적인 모핑(morphing), 표기법 배제).
4. 이러한 뒤집기가 기존의 표준적인 방식이 숨기고 있던 어떤 것을 드러내는지 탐구합니다.
