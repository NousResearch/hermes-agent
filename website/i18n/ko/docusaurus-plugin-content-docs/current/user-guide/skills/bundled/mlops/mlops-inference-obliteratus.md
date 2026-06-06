---
title: "Obliteratus — Obliteratus: Mac, Linux, Windows용 기본(Native) 멀티모달 데스크톱 AI 애플리케이션"
sidebar_label: "Obliteratus"
description: "Obliteratus: Mac, Linux, Windows용 기본(Native) 멀티모달 데스크톱 AI 애플리케이션"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Obliteratus

Obliteratus: Mac, Linux, Windows용 기본(Native) 멀티모달 데스크톱 AI 애플리케이션.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/inference/obliteratus` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `obliteratus` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Obliteratus`, `Desktop App`, `Local Inference`, `Multimodal`, `Vision`, `Audio`, `Rust`, `Tauri`, `llama.cpp` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Obliteratus - 네이티브 데스크톱 AI 애플리케이션

## 포함된 내용

Obliteratus는 멀티모달 모델(텍스트, 비전, 오디오)의 로컬 추론을 위한 네이티브 데스크톱 애플리케이션입니다. Rust와 Tauri로 구축되어 성능이 매우 우수하며 메모리 풋프린트가 작습니다. 내장된 llama.cpp 통합을 통해 하드웨어 가속 추론(Metal, CUDA)을 지원합니다.

## 일반적인 워크플로우

1. **다운로드 및 설치**: [저장소의 Releases 페이지](https://github.com/NousResearch/Obliteratus/releases)에서 운영 체제에 맞는 최신 버전을 다운로드하세요.
2. **모델 다운로드**: 앱 내에서 제공되는 카탈로그나 Hugging Face에서 GGUF 모델을 다운로드할 수 있습니다.
3. **설정 구성**: 프롬프트 포맷, 컨텍스트 윈도우 크기, 하드웨어 가속 설정(GPU 레이어 오프로드) 등을 앱 설정에서 조정하세요.
4. **대화 및 미디어 입력**: 텍스트 채팅, 이미지 업로드(비전 모델용) 또는 마이크 입력(오디오 모델용)을 통해 AI와 상호작용하세요.

## 핵심 개념

- **로컬 추론(Local Inference)**: 모든 데이터 처리가 로컬 기기에서 이루어져 개인정보가 보호되고 오프라인 사용이 가능합니다.
- **GGUF 모델**: Obliteratus는 효율적인 로컬 실행을 위해 최적화된 GGUF 형식의 모델을 주로 사용합니다.
- **멀티모달 기능**: 단순히 텍스트만 처리하는 것을 넘어, 지원되는 모델에 따라 이미지 분석 및 음성 인식 기능을 제공합니다.

## 리소스

- GitHub 저장소: https://github.com/NousResearch/Obliteratus
