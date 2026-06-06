---
title: "Songsee — CLI를 통한 오디오 스펙트로그램/특성(mel, chroma, MFCC) 시각화"
sidebar_label: "Songsee"
description: "CLI를 통한 오디오 스펙트로그램/특성(mel, chroma, MFCC) 시각화"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Songsee

CLI를 통해 오디오 스펙트로그램/특성(mel, chroma, MFCC)을 시각화합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/media/songsee` |
| Version | `1.0.0` |
| Author | community |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Audio`, `Visualization`, `Spectrogram`, `Music`, `Analysis` |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# songsee

오디오 파일에서 스펙트로그램과 멀티 패널 오디오 특성 시각화를 생성합니다.

## 전제 조건 (Prerequisites)

[Go](https://go.dev/doc/install)가 필요합니다:
```bash
go install github.com/steipete/songsee/cmd/songsee@latest
```

선택 사항: WAV/MP3 이외의 포맷을 위한 `ffmpeg`.

## 빠른 시작 (Quick Start)

```bash
# 기본 스펙트로그램 (Basic spectrogram)
songsee track.mp3

# 특정 파일로 저장 (Save to specific file)
songsee track.mp3 -o spectrogram.png

# 멀티 패널 시각화 그리드 (Multi-panel visualization grid)
songsee track.mp3 --viz spectrogram,mel,chroma,hpss,selfsim,loudness,tempogram,mfcc,flux

# 특정 시간 구간 자르기 (Time slice - 12.5초에서 시작, 8초 길이)
songsee track.mp3 --start 12.5 --duration 8 -o slice.jpg

# 표준 입력에서 읽기 (From stdin)
cat track.mp3 | songsee - --format png -o out.png
```

## 시각화 유형 (Visualization Types)

`--viz` 파라미터와 함께 쉼표로 구분된 값을 사용하세요:

| Type | Description |
|------|-------------|
| `spectrogram` | 표준 주파수 스펙트로그램 (Standard frequency spectrogram) |
| `mel` | 멜 스케일 스펙트로그램 (Mel-scaled spectrogram) |
| `chroma` | 피치 클래스 분포 (Pitch class distribution) |
| `hpss` | 하모닉/퍼커시브 분리 (Harmonic/percussive separation) |
| `selfsim` | 자기 유사성 행렬 (Self-similarity matrix) |
| `loudness` | 시간에 따른 음량 (Loudness over time) |
| `tempogram` | 템포 추정 (Tempo estimation) |
| `mfcc` | 멜-주파수 셉스트럼 계수 (Mel-frequency cepstral coefficients) |
| `flux` | 스펙트럴 플럭스, 시작점 감지 (Spectral flux, onset detection) |

여러 개의 `--viz` 유형은 단일 이미지 내의 그리드로 렌더링됩니다.

## 공통 플래그 (Common Flags)

| Flag | Description |
|------|-------------|
| `--viz` | 시각화 유형 (쉼표로 구분) |
| `--style` | 컬러 팔레트: `classic`, `magma`, `inferno`, `viridis`, `gray` |
| `--width` / `--height` | 출력 이미지 크기(dimensions) |
| `--window` / `--hop` | FFT 윈도우 및 홉(hop) 크기 |
| `--min-freq` / `--max-freq` | 주파수 범위 필터 |
| `--start` / `--duration` | 오디오의 시간 구간 |
| `--format` | 출력 포맷: `jpg` 또는 `png` |
| `-o` | 출력 파일 경로 |

## 참고 (Notes)

- WAV 및 MP3는 네이티브로 디코딩됩니다; 다른 포맷은 `ffmpeg`가 필요합니다.
- 출력 이미지는 자동화된 오디오 분석을 위해 `vision_analyze`로 분석될 수 있습니다.
- 오디오 출력을 비교하거나, 합성을 디버깅하거나, 오디오 처리 파이프라인을 문서화하는 데 유용합니다.
