---
title: "Pytorch Fsdp — 대규모 모델을 위한 완전 샤딩 데이터 병렬 학습"
sidebar_label: "Pytorch Fsdp"
description: "대규모 모델을 위한 완전 샤딩 데이터 병렬 학습"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pytorch Fsdp

대규모 모델을 위한 완전 샤딩 데이터 병렬 학습입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/pytorch-fsdp`로 설치 |
| 경로 | `optional-skills/mlops/pytorch-fsdp` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `torch>=2.0`, `transformers` |
| 플랫폼 | linux, macos |
| 태그 | `Distributed Training`, `PyTorch`, `FSDP`, `Data Parallel`, `Sharding`, `Mixed Precision`, `CPU Offloading`, `FSDP2`, `Large-Scale Training` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Pytorch-Fsdp Skill

공식 문서에서 생성된 pytorch-fsdp 개발 지원입니다.

## 이 스킬을 사용하는 시점

이 스킬은 다음과 같은 경우 트리거되어야 합니다:
- pytorch-fsdp를 사용할 때
- pytorch-fsdp 기능 또는 API에 대해 질문할 때
- pytorch-fsdp 솔루션을 구현할 때
- pytorch-fsdp 코드를 디버깅할 때
- pytorch-fsdp 모범 사례를 학습할 때

## 빠른 참조

실행 가능한 FSDP 스니펫으로 구성된 전체 공통 패턴 카탈로그(약 157k자)는
`references/common-patterns.md`에 있습니다 — 래핑, 샤딩 전략, 체크포인트 또는 혼합 정밀도 예제가 필요할 때 `read_file`로 로드하세요. FSDP 명령을 기억에 의존해 재구성하기보다 여기서 시작하세요.

## 참고 파일

이 스킬에는 `references/`에 포괄적인 문서가 포함되어 있습니다:

- **other.md** - 기타 문서

세부 정보가 필요할 때 `view`를 사용해 특정 참고 파일을 읽으세요.

## 이 스킬 사용 방법

### 초보자용
기초 개념을 익히려면 getting_started 또는 tutorials 참고 파일부터 시작하세요.

### 특정 기능용
자세한 내용은 적절한 범주 참고 파일(api, guides 등)을 사용하세요.

### 코드 예제용
위의 빠른 참조 섹션에는 공식 문서에서 추출한 일반적인 패턴이 포함되어 있습니다.

## 리소스

### references/
공식 출처에서 추출한 문서가 체계적으로 정리되어 있습니다. 이 파일에는 다음이 포함됩니다:
- 자세한 설명
- 언어 주석이 포함된 코드 예제
- 원본 문서로 연결되는 링크
- 빠른 탐색을 위한 목차

### scripts/
일반적인 자동화를 위한 도우미 스크립트를 여기에 추가하세요.

### assets/
템플릿, 보일러플레이트 또는 예제 프로젝트를 여기에 추가하세요.

## 참고

- 이 스킬은 공식 문서에서 자동으로 생성되었습니다
- 참고 파일은 원본 문서의 구조와 예제를 보존합니다
- 코드 예제에는 더 나은 구문 강조를 위한 언어 감지가 포함됩니다

## 업데이트

이 스킬을 최신 문서로 갱신하려면:
1. 동일한 구성으로 스크레이퍼를 다시 실행합니다
2. 스킬이 최신 문서로 다시 빌드됩니다
