---
title: "Pixel Art — 시대별 팔레트를 사용하는 픽셀 아트(NES, Game Boy, PICO-8)"
sidebar_label: "Pixel Art"
description: "시대별 팔레트를 사용하는 픽셀 아트(NES, Game Boy, PICO-8)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Pixel Art

시대별 팔레트를 사용하는 픽셀 아트(NES, Game Boy, PICO-8)입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/pixel-art`로 설치 |
| 경로 | `optional-skills/creative/pixel-art` |
| 버전 | `2.0.0` |
| 작성자 | dodo-reach |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `creative`, `pixel-art`, `arcade`, `snes`, `nes`, `gameboy`, `retro`, `image`, `video` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# Pixel Art

모든 이미지를 레트로 픽셀 아트로 변환한 다음, 시대에 맞는 효과(비, 반딧불이, 눈, 불씨)를 사용해 짧은 MP4 또는 GIF로 애니메이션화할 수 있습니다.

이 스킬에는 두 가지 스크립트가 포함되어 있습니다.

- `scripts/pixel_art.py` — 사진 → 픽셀 아트 PNG(Floyd-Steinberg 디더링)
- `scripts/pixel_art_video.py` — 픽셀 아트 PNG → 애니메이션 MP4(+ 선택적 GIF)

시대에 정확한 색상(NES, Game Boy, PICO-8 등)이 필요하면 프리셋을 하드웨어 팔레트에 맞출 수 있으며, 아케이드/SNES 스타일에는 적응형 N색 양자화를 사용할 수 있습니다.

## 사용 시점

- 사용자가 원본 이미지에서 레트로 픽셀 아트를 원하는 경우
- 사용자가 NES / Game Boy / PICO-8 / C64 / 아케이드 / SNES 스타일을 요청하는 경우
- 사용자가 짧은 반복 애니메이션(비 내리는 장면, 밤하늘, 눈 등)을 원하는 경우
- 포스터, 앨범 커버, 소셜 게시물, 스프라이트, 캐릭터, 아바타

## 워크플로

생성하기 전에 사용자에게 스타일을 확인합니다. 프리셋마다 결과가 크게 다르고 재생성에는 비용이 듭니다.

### 1단계 — 스타일 제안

대표 프리셋 4개와 함께 `clarify`를 호출합니다. 사용자가 요청한 내용에 따라 세트를 선택하며, 14개를 모두 나열하지 않습니다.

사용자 의도가 명확하지 않을 때의 기본 메뉴:

```python
clarify(
    question="Which pixel-art style do you want?",
    choices=[
        "arcade — bold, chunky 80s cabinet feel (16 colors, 8px)",
        "nes — Nintendo 8-bit hardware palette (54 colors, 8px)",
        "gameboy — 4-shade green Game Boy DMG",
        "snes — cleaner 16-bit look (32 colors, 4px)",
    ],
)
```

사용자가 이미 시대(예: "80s arcade", "Gameboy")를 지정했다면 `clarify`를 건너뛰고 일치하는 프리셋을 직접 사용합니다.

### 2단계 — 애니메이션 제안(선택 사항)

사용자가 비디오/GIF를 요청했거나 결과에 움직임을 더하면 좋을 것 같다면 어떤 장면인지 묻습니다.

```python
clarify(
    question="Want to animate it? Pick a scene or skip.",
    choices=[
        "night — stars + fireflies + leaves",
        "urban — rain + neon pulse",
        "snow — falling snowflakes",
        "skip — just the image",
    ],
)
```

`clarify`를 연속으로 두 번 넘게 호출하지 않습니다. 스타일에 이어 장면을 한 번씩 호출합니다. 사용자가 메시지에서 특정 스타일과 장면을 명시했다면 `clarify`를 완전히 건너뜁니다.

### 3단계 — 생성

먼저 `pixel_art()`를 실행하고, 애니메이션이 요청되었다면 결과에 이어 `pixel_art_video()`를 실행합니다.

## 프리셋 카탈로그

| 프리셋 | 시대 | 팔레트 | 블록 | 적합한 용도 |
|--------|-----|---------|------|----------|
| `arcade` | 80년대 아케이드 | 적응형 16 | 8px | 대담한 포스터, 히어로 아트 |
| `snes` | 16비트 | 적응형 32 | 4px | 캐릭터, 세밀한 장면 |
| `nes` | 8비트 | NES (54) | 8px | 진정한 NES 느낌 |
| `gameboy` | DMG 휴대기기 | 녹색 4단계 | 8px | 흑백 Game Boy |
| `gameboy_pocket` | Pocket 휴대기기 | 회색 4단계 | 8px | 흑백 GB Pocket |
| `pico8` | PICO-8 | 고정 16 | 6px | 판타지 콘솔 느낌 |
| `c64` | Commodore 64 | 고정 16 | 8px | 8비트 가정용 컴퓨터 |
| `apple2` | Apple II 고해상도 | 고정 6 | 10px | 극단적인 레트로, 6색 |
| `teletext` | BBC Teletext | 순수 8 | 10px | 굵은 원색 |
| `mspaint` | Windows MS Paint | 고정 24 | 8px | 향수를 불러일으키는 데스크톱 |
| `mono_green` | CRT 형광체 | 녹색 2 | 6px | 터미널/CRT 미학 |
| `mono_amber` | CRT 앰버 | 앰버 2 | 6px | 앰버 모니터 느낌 |
| `neon` | 사이버펑크 | 네온 10 | 6px | 베이퍼웨이브/사이버 |
| `pastel` | 부드러운 파스텔 | 파스텔 10 | 6px | 카와이 / 부드러운 느낌 |

이름이 지정된 팔레트는 `scripts/palettes.py`에 있습니다(`references/palettes.md`에서 전체 목록 확인 — 이름이 지정된 팔레트 총 28개). 모든 프리셋은 재정의할 수 있습니다.

```python
pixel_art("in.png", "out.png", preset="snes", palette="PICO_8", block=6)
```

## 장면 카탈로그(비디오용)

| 장면 | 효과 |
|-------|---------|
| `night` | 반짝이는 별 + 반딧불이 + 흘러가는 나뭇잎 |
| `dusk` | 반딧불이 + 반짝임 |
| `tavern` | 먼지 입자 + 따뜻한 반짝임 |
| `indoor` | 먼지 입자 |
| `urban` | 비 + 네온 펄스 |
| `nature` | 나뭇잎 + 반딧불이 |
| `magic` | 반짝임 + 반딧불이 |
| `storm` | 비 + 번개 |
| `underwater` | 거품 + 빛의 반짝임 |
| `fire` | 불씨 + 반짝임 |
| `snow` | 눈송이 + 반짝임 |
| `desert` | 아지랑이 + 먼지 |

## 호출 패턴

### Python(가져오기)

```python
import sys
sys.path.insert(0, "/home/teknium/.hermes/skills/creative/pixel-art/scripts")
from pixel_art import pixel_art
from pixel_art_video import pixel_art_video

# 1. Convert to pixel art
pixel_art("/path/to/photo.jpg", "/tmp/pixel.png", preset="nes")

# 2. Animate (optional)
pixel_art_video(
    "/tmp/pixel.png",
    "/tmp/pixel.mp4",
    scene="night",
    duration=6,
    fps=15,
    seed=42,
    export_gif=True,
)
```

### CLI

```bash
cd /home/teknium/.hermes/skills/creative/pixel-art/scripts

python pixel_art.py in.jpg out.png --preset gameboy
python pixel_art.py in.jpg out.png --preset snes --palette PICO_8 --block 6

python pixel_art_video.py out.png out.mp4 --scene night --duration 6 --gif
```

## 파이프라인의 근거

**픽셀 변환:**
1. 대비/색상/선명도 향상(팔레트가 작을수록 더 강하게)
2. 양자화 전에 색조 영역을 단순화하도록 포스터화
3. `Image.NEAREST`로 `block` 크기에 맞춰 축소(보간 없는 선명한 픽셀)
4. 적응형 N색 팔레트 또는 이름이 지정된 하드웨어 팔레트에 대해 Floyd-Steinberg 디더링으로 양자화
5. `Image.NEAREST`로 다시 확대

축소 후 양자화하면 디더링이 최종 픽셀 격자에 맞춰집니다. 그 전에 양자화하면 사라질 세부 요소에 오류 확산이 낭비됩니다.

**비디오 오버레이:**
- 각 틱마다 기본 프레임을 복사(정적 배경)
- 프레임마다 상태 없이 파티클을 그려 오버레이(효과마다 함수 하나)
- ffmpeg `libx264 -pix_fmt yuv420p -crf 18`로 인코딩
- `palettegen` + `paletteuse`를 통한 선택적 GIF

## 종속성

- Python 3.9+
- Pillow (`pip install Pillow`)
- PATH에 있는 ffmpeg(비디오에만 필요 — Hermes가 이 패키지를 설치함)

## 문제점 및 주의 사항

- 팔레트 키는 대소문자를 구분합니다(`"NES"`, `"PICO_8"`, `"GAMEBOY_ORIGINAL"`).
- 매우 작은 원본(&lt;100px 너비)은 8-10px 블록에서 뭉개집니다. 원본이 작다면 먼저 확대합니다.
- 소수 `block` 또는 `palette`는 양자화를 중단시킵니다 — 양의 정수로 유지합니다.
- 애니메이션 파티클 수는 약 640x480 캔버스에 맞춰 조정되어 있습니다. 매우 큰 이미지에서는 밀도를 위해 다른 시드로 두 번째 패스를 실행하는 것이 좋습니다.
- `mono_green` / `mono_amber`는 `color=0.0`을 강제합니다(채도 제거). 재정의하면서 색도를 유지하면 2색 팔레트가 부드러운 영역에 줄무늬를 만들 수 있습니다.
- `clarify` 루프: 턴당 최대 두 번(스타일, 장면) 호출합니다. 사용자에게 선택지를 너무 많이 요구하지 않습니다.

## 검증

- 출력 경로에 PNG가 생성됨
- 프리셋의 블록 크기에서 선명한 정사각형 픽셀이 보임
- 색상 수가 프리셋과 일치함(이미지를 눈으로 확인하거나 `Image.open(p).getcolors()` 실행)
- 유효한 MP4 비디오임(`ffprobe`로 열 수 있음) 및 크기가 0이 아님

## 저작자 표시

이름이 지정된 하드웨어 팔레트와 `pixel_art_video.py`의 절차적 애니메이션 루프는 [pixel-art-studio](https://github.com/Synero/pixel-art-studio)(MIT)에서 포팅되었습니다. 자세한 내용은 이 스킬 디렉터리의 `ATTRIBUTION.md`를 참조하세요.
