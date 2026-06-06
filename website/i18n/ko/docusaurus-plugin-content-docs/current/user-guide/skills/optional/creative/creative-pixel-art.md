---
title: "Pixel Art — 시대별 팔레트를 사용한 픽셀 아트 (NES, Game Boy, PICO-8)"
sidebar_label: "Pixel Art"
description: "시대별 팔레트를 사용한 픽셀 아트 (NES, Game Boy, PICO-8)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pixel Art

시대별 팔레트를 사용한 픽셀 아트 (NES, Game Boy, PICO-8).

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/creative/pixel-art`로 설치 |
| Path | `optional-skills/creative/pixel-art` |
| Version | `2.0.0` |
| Author | dodo-reach |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `creative`, `pixel-art`, `arcade`, `snes`, `nes`, `gameboy`, `retro`, `image`, `video` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Pixel Art

모든 이미지를 레트로 픽셀 아트 형식으로 변환하고, 그 후 선택적으로 시대에 맞는 효과(비, 반딧불이, 눈, 불씨)가 있는 짧은 MP4 또는 GIF로 애니메이션화합니다.

이 스킬에는 두 개의 스크립트가 제공됩니다:

- `scripts/pixel_art.py` — 사진 → 픽셀 아트 PNG (Floyd-Steinberg 디더링)
- `scripts/pixel_art_video.py` — 픽셀 아트 PNG → 애니메이션 MP4 (+ 선택적 GIF)

각각을 임포트(import)하거나 직접 실행할 수 있습니다. 특정 시대의 정확한 색상(NES, Game Boy, PICO-8 등)을 원할 때 하드웨어 팔레트에 맞추는 프리셋을 사용하거나, 아케이드/SNES 스타일을 위한 적응형 N-색상 양자화를 사용할 수 있습니다.

## 사용 시기

- 사용자가 원본 이미지에서 레트로 픽셀 아트를 원할 때
- 사용자가 NES / Game Boy / PICO-8 / C64 / 아케이드 / SNES 스타일을 요청할 때
- 사용자가 짧은 반복 애니메이션(비 오는 장면, 밤하늘, 눈 등)을 원할 때
- 포스터, 앨범 커버, 소셜 게시물, 스프라이트, 캐릭터, 아바타

## 워크플로우

생성하기 전에 사용자에게 스타일을 확인하세요. 다양한 프리셋이 매우 다른 결과를 생성하며, 다시 생성하는 데는 비용이 많이 듭니다.

### 1단계 — 스타일 제안

4개의 대표적인 프리셋으로 `clarify`를 호출하세요. 사용자가 요청한 내용에 기반해 세트를 선택하고 14개 전체를 한 번에 쏟아내지 마세요.

사용자의 의도가 불분명할 때의 기본 메뉴:

```python
clarify(
    question="어떤 픽셀 아트 스타일을 원하십니까?",
    choices=[
        "arcade — 굵고 투박한 80년대 캐비닛 느낌 (16색, 8px)",
        "nes — 닌텐도 8비트 하드웨어 팔레트 (54색, 8px)",
        "gameboy — 4가지 녹색 음영의 게임보이 DMG",
        "snes — 더 깔끔한 16비트 느낌 (32색, 4px)",
    ],
)
```

사용자가 이미 시대를 명시한 경우(예: "80s arcade", "Gameboy") `clarify`를 건너뛰고 일치하는 프리셋을 직접 사용합니다.

### 2단계 — 애니메이션 제안 (선택 사항)

사용자가 비디오/GIF를 요청했거나 결과물에 모션이 들어가면 좋을 경우 어떤 장면(scene)을 원하는지 묻습니다:

```python
clarify(
    question="애니메이션을 추가하시겠습니까? 장면을 선택하거나 건너뛰세요.",
    choices=[
        "night — 별 + 반딧불이 + 나뭇잎",
        "urban — 비 + 네온 펄스",
        "snow — 떨어지는 눈송이",
        "skip — 이미지만",
    ],
)
```

한 턴에 `clarify`를 연달아 두 번 이상 호출하지 마세요. (하나는 스타일에 대해, 또 하나는 애니메이션 가능 시 장면에 대해). 사용자가 메시지에서 명시적으로 특정 스타일과 장면을 요청했다면 `clarify`를 완전히 건너뜁니다.

### 3단계 — 생성

먼저 `pixel_art()`를 실행합니다. 애니메이션이 요청된 경우, 그 결과를 연달아 `pixel_art_video()`에 전달합니다.

## 프리셋 카탈로그

| 프리셋 | 시대 | 팔레트 | 블록 | 권장 용도 |
|--------|-----|---------|-------|----------|
| `arcade` | 80s 아케이드 | 적응형 16색 | 8px | 굵은 포스터, 메인 아트 |
| `snes` | 16-bit | 적응형 32색 | 4px | 캐릭터, 디테일한 장면 |
| `nes` | 8-bit | NES (54색) | 8px | 진정한 NES 느낌 |
| `gameboy` | DMG 휴대용 | 4가지 녹색 음영 | 8px | 모노크롬 게임보이 |
| `gameboy_pocket` | 포켓 휴대용 | 4가지 회색 음영 | 8px | 모노 GB 포켓 |
| `pico8` | PICO-8 | 16색 고정 | 6px | 판타지 콘솔 느낌 |
| `c64` | Commodore 64 | 16색 고정 | 8px | 8비트 가정용 컴퓨터 |
| `apple2` | Apple II 고해상도 | 6색 고정 | 10px | 극한의 레트로, 6색 |
| `teletext` | BBC Teletext | 8 순색 | 10px | 투박한 원색 |
| `mspaint` | Windows MS Paint | 24색 고정 | 8px | 향수 어린 데스크탑 |
| `mono_green` | CRT 형광 물질 | 2가지 녹색 | 6px | 터미널/CRT 미학 |
| `mono_amber` | CRT 호박색 | 2가지 호박색 | 6px | 호박색 모니터 느낌 |
| `neon` | 사이버펑크 | 10가지 네온 | 6px | 베이퍼웨이브/사이버 |
| `pastel` | 부드러운 파스텔 | 10가지 파스텔 | 6px | 카와이(Kawaii) / 부드러운 |

지정된 팔레트는 `scripts/palettes.py`에 위치해 있습니다 (총 28개의 팔레트 전체 목록은 `references/palettes.md` 참조). 어떤 프리셋이든 재정의할 수 있습니다:

```python
pixel_art("in.png", "out.png", preset="snes", palette="PICO_8", block=6)
```

## 장면 카탈로그 (비디오용)

| 장면 | 효과 |
|-------|---------|
| `night` | 반짝이는 별 + 반딧불이 + 떠다니는 나뭇잎 |
| `dusk` | 반딧불이 + 반짝임 |
| `tavern` | 먼지 티끌 + 따뜻한 반짝임 |
| `indoor` | 먼지 티끌 |
| `urban` | 비 + 네온 펄스 |
| `nature` | 나뭇잎 + 반딧불이 |
| `magic` | 반짝임 + 반딧불이 |
| `storm` | 비 + 번개 |
| `underwater` | 거품 + 빛 반짝임 |
| `fire` | 불씨 + 반짝임 |
| `snow` | 눈송이 + 반짝임 |
| `desert` | 아지랑이 + 먼지 |

## 호출 패턴

### Python (임포트)

```python
import sys
sys.path.insert(0, "/home/teknium/.hermes/skills/creative/pixel-art/scripts")
from pixel_art import pixel_art
from pixel_art_video import pixel_art_video

# 1. 픽셀 아트로 변환
pixel_art("/path/to/photo.jpg", "/tmp/pixel.png", preset="nes")

# 2. 애니메이션화 (선택 사항)
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

## 파이프라인 설계 근거

**픽셀 변환:**
1. 대비/색상/선명도 향상 (팔레트가 작을수록 더 강하게 적용)
2. 양자화 전 톤(tone) 영역을 단순화하기 위해 포스터화(Posterize)
3. `Image.NEAREST`를 사용하여 `block` 단위로 축소 (하드 픽셀, 보간 없음)
4. Floyd-Steinberg 디더링을 통한 양자화 — 적응형 N-색상 팔레트 또는 명명된 하드웨어 팔레트 기반
5. `Image.NEAREST`를 사용해 다시 확대

축소 이후에 양자화를 수행하면 디더링이 최종 픽셀 그리드와 정렬된 상태로 유지됩니다. 양자화를 먼저 수행하면 나중에 사라질 디테일에 오류 확산(error-diffusion)을 낭비하게 됩니다.

**비디오 오버레이:**
- 매 틱마다 기본 프레임 복사 (정적인 배경)
- 프레임 단위의 무상태 파티클 그리기 오버레이 (효과당 하나의 함수)
- ffmpeg `libx264 -pix_fmt yuv420p -crf 18`을 통해 인코딩
- `palettegen` + `paletteuse`를 통한 선택적 GIF 생성

## 종속성

- Python 3.9+
- Pillow (`pip install Pillow`)
- PATH의 ffmpeg (비디오에만 필요 — Hermes 패키지 내장 설치)

## 주의 사항

- 팔레트 키는 대소문자를 구분합니다 (`"NES"`, `"PICO_8"`, `"GAMEBOY_ORIGINAL"`).
- 아주 작은 원본(너비 100px 미만)은 8-10px 블록 하에서 뭉개집니다. 원본이 너무 작다면 먼저 크기를 키우세요.
- 분수 형태의 `block`이나 `palette`는 양자화를 깨뜨리므로 양의 정수를 유지하세요.
- 애니메이션 입자(particle) 수는 약 640x480 캔버스에 맞춰 조정되어 있습니다. 매우 큰 이미지에서는 입자 밀도를 위해 다른 시드(seed)로 두 번째 패스를 수행해야 할 수도 있습니다.
- `mono_green` / `mono_amber`는 `color=0.0`(채도 제거)을 강제합니다. 이를 무시하고 크로마(채도)를 유지하면 2가지 색상 팔레트가 부드러운 영역에서 줄무늬를 만들 수 있습니다.
- `clarify` 루프: 한 턴에 최대 두 번 호출하세요 (스타일, 그다음 장면). 사용자에게 더 이상의 선택을 쏟아내지 마세요.

## 검증

- 지정된 출력 경로에 PNG 생성
- 설정된 블록 크기로 명확한 사각형 픽셀 블록이 보임
- 색상 수가 프리셋과 일치함 (눈으로 확인하거나 `Image.open(p).getcolors()` 실행)
- 비디오가 크기가 0이 아닌 유효한 MP4 파일임 (`ffprobe`로 열 수 있음)

## 저작자 표시 (Attribution)

명명된 하드웨어 팔레트와 `pixel_art_video.py`의 절차적 애니메이션 루프는 [pixel-art-studio](https://github.com/Synero/pixel-art-studio) (MIT)에서 포팅되었습니다. 자세한 내용은 이 스킬 디렉토리의 `ATTRIBUTION.md`를 참조하세요.
