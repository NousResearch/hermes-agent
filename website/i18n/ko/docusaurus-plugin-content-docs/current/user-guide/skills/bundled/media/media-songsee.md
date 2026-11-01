---
title: "Songsee — CLI를 통한 오디오 스펙트로그램/특징(mel, chroma, MFCC)"
sidebar_label: "Songsee"
description: "CLI를 통한 오디오 스펙트로그램/특징(mel, chroma, MFCC)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 skill의 SKILL.md에서 자동으로 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Songsee

CLI를 통해 오디오 스펙트로그램과 여러 오디오 특징 시각화를 생성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 기본 설치에 포함 |
| 경로 | `skills/media/songsee` |
| 버전 | `1.0.0` |
| 작성자 | community |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Audio`, `Visualization`, `Spectrogram`, `Music`, `Analysis` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화된 상태에서 에이전트가 보는 내용입니다.
:::

# songsee

오디오 파일에서 스펙트로그램과 여러 패널로 구성된 오디오 특징 시각화를 생성합니다.

## 사전 요구 사항

[Go](https://go.dev/doc/install)가 필요합니다:
```bash
go install github.com/steipete/songsee/cmd/songsee@latest
```

선택 사항: WAV/MP3 이외의 형식을 지원하려면 `ffmpeg`가 필요합니다.

## 빠른 시작

```bash
# Basic spectrogram
songsee track.mp3

# Save to specific file
songsee track.mp3 -o spectrogram.png

# Multi-panel visualization grid
songsee track.mp3 --viz spectrogram,mel,chroma,hpss,selfsim,loudness,tempogram,mfcc,flux

# Time slice (start at 12.5s, 8s duration)
songsee track.mp3 --start 12.5 --duration 8 -o slice.jpg

# From stdin
cat track.mp3 | songsee - --format png -o out.png
```

## 시각화 유형

쉼표로 구분한 값을 `--viz`와 함께 사용합니다:

| 유형 | 설명 |
|------|-------------|
| `spectrogram` | 표준 주파수 스펙트로그램 |
| `mel` | Mel 스케일 스펙트로그램 |
| `chroma` | 피치 클래스 분포 |
| `hpss` | 하모닉/퍼커시브 분리 |
| `selfsim` | 자기 유사도 행렬 |
| `loudness` | 시간에 따른 음량 |
| `tempogram` | 템포 추정 |
| `mfcc` | Mel 주파수 켑스트럼 계수 |
| `flux` | 스펙트럼 플럭스(온셋 감지) |

여러 `--viz` 유형을 지정하면 하나의 이미지에 그리드로 렌더링됩니다.

## 일반 플래그

| 플래그 | 설명 |
|------|-------------|
| `--viz` | 시각화 유형(쉼표로 구분) |
| `--style` | 색상 팔레트: `classic`, `magma`, `inferno`, `viridis`, `gray` |
| `--width` / `--height` | 출력 이미지 크기 |
| `--window` / `--hop` | FFT 윈도우 및 홉 크기 |
| `--min-freq` / `--max-freq` | 주파수 범위 필터 |
| `--start` / `--duration` | 오디오의 시간 구간 |
| `--format` | 출력 형식: `jpg` 또는 `png` |
| `-o` | 출력 파일 경로 |

## 참고 사항

- WAV와 MP3는 기본적으로 디코딩되며, 다른 형식에는 `ffmpeg`가 필요합니다.
- 자동화된 오디오 분석에는 `vision_analyze`를 사용해 출력 이미지를 검사할 수 있습니다.
- 오디오 출력 비교, 합성 디버깅 또는 오디오 처리 파이프라인 문서화에 유용합니다.
