---
title: "Axolotl — Axolotl: YAML LLM 미세 조정(LoRA, DPO, GRPO)"
sidebar_label: "Axolotl"
description: "Axolotl: YAML LLM 미세 조정(LoRA, DPO, GRPO)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Axolotl

Axolotl: YAML LLM 미세 조정(LoRA, DPO, GRPO).

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/axolotl`로 설치 |
| 경로 | `optional-skills/mlops/training/axolotl` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `axolotl`, `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `deepspeed` |
| 플랫폼 | linux, macos |
| 태그 | `Fine-Tuning`, `Axolotl`, `LLM`, `LoRA`, `QLoRA`, `DPO`, `KTO`, `ORPO`, `GRPO`, `YAML`, `HuggingFace`, `DeepSpeed`, `Multimodal` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Axolotl Skill

## 포함 내용

Axolotl로 LLM을 미세 조정하기 위한 전문가 지침입니다. YAML 설정, 100개 이상의 모델, LoRA/QLoRA, DPO/KTO/ORPO/GRPO, 멀티모달 지원을 다룹니다.

Axolotl 개발, 생성된 공식 문서에서 제공하는 지원입니다.

## 이 스킬을 사용하는 시점

다음과 같은 경우 이 스킬이 활성화되어야 합니다:
- axolotl을 사용할 때
- axolotl 기능 또는 API를 질문할 때
- axolotl 솔루션을 구현할 때
- axolotl 코드를 디버깅할 때
- axolotl 모범 사례를 학습할 때

## 빠른 참조

### 일반적인 패턴

**패턴 1:** 학습 작업에 허용 가능한 데이터 전송 속도가 나오는지 검증하려면 NCCL Tests를 실행해 병목 지점을 찾을 수 있습니다. 예를 들면 다음과 같습니다:

```
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 3
```

**패턴 2:** Axolotl yaml에서 FSDP를 사용하도록 모델을 구성합니다. 예를 들면 다음과 같습니다:

```
fsdp_version: 2
fsdp_config:
  offload_params: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

**패턴 3:** context_parallel_size는 전체 GPU 수의 약수여야 합니다. 예를 들면 다음과 같습니다:

```
context_parallel_size
```

**패턴 4:** 예를 들면 다음과 같습니다: - GPU 8개와 시퀀스 병렬화 없음: 스텝마다 서로 다른 배치 8개 처리 - GPU 8개와 context_parallel_size=4: 서로 다른 배치 2개만 처리(각 배치를 GPU 4개에 걸쳐 분할) - GPU당 micro_batch_size가 2이면 전역 배치 크기가 16에서 4로 감소

```
context_parallel_size=4
```

**패턴 5:** 구성에서 save_compressed: true를 설정하면 압축 형식으로 모델을 저장할 수 있으며, 다음과 같은 효과가 있습니다: - 디스크 공간 사용량을 약 40% 줄임 - 가속 추론을 위해 vLLM과의 호환성 유지 - 추가 최적화(예: 양자화)를 위해 llmcompressor와의 호환성 유지

```
save_compressed: true
```

**패턴 6:** 참고 통합을 integrations 폴더에 배치할 필요는 없습니다. Python 환경의 패키지에 설치되어 있기만 하면 어느 위치에나 둘 수 있습니다. 예시는 이 저장소를 참조하세요: https://github.com/axolotl-ai-cloud/diff-transformer

```
integrations
```

**패턴 7:** 단일 예제와 배치 데이터를 모두 처리합니다. - 단일 예제: sample[‘input_ids’]는 list[int] - 배치 데이터: sample[‘input_ids’]는 list[list[int]]

```
utils.trainer.drop_long_seq(sample, sequence_len=2048, min_sequence_len=2)
```

### 코드 패턴 예시

**예시 1** (python):
```python
cli.cloud.modal_.ModalCloud(config, app=None)
```

**예시 2** (python):
```python
cli.cloud.modal_.run_cmd(cmd, run_folder, volumes=None)
```

**예시 3** (python):
```python
core.trainers.base.AxolotlTrainer(
    *_args,
    bench_data_collator=None,
    eval_data_collator=None,
    dataset_tags=None,
    **kwargs,
)
```

**예시 4** (python):
```python
core.trainers.base.AxolotlTrainer.log(logs, start_time=None)
```

**예시 5** (python):
```python
prompt_strategies.input_output.RawInputOutputPrompter()
```

## 참조 파일

이 스킬에는 `references/`에 종합 문서가 포함되어 있습니다:

- **api.md** - API 문서
- **dataset-formats.md** - 데이터셋 형식 문서
- **other.md** - 기타 문서

자세한 정보가 필요하면 특정 참조 파일을 읽을 때 `view`를 사용하세요.

## 이 스킬 사용하기

### 초보자용
기초 개념을 익히려면 getting_started 또는 tutorials 참조 파일부터 시작하세요.

### 특정 기능용
자세한 내용은 적절한 카테고리 참조 파일(api, guides 등)을 사용하세요.

### 코드 예시용
위의 빠른 참조 섹션에는 공식 문서에서 추출한 일반적인 패턴이 포함되어 있습니다.

## 리소스

### references/
정리된 문서가 들어 있습니다. 이 파일에는 다음이 포함됩니다:
- 자세한 설명
- 언어 주석이 포함된 코드 예시
- 원본 문서 링크
- 빠른 탐색을 위한 목차

### scripts/
일반적인 자동화를 위한 도우미 스크립트를 여기에 추가하세요.

### assets/
템플릿, 보일러플레이트 또는 예시 프로젝트를 여기에 추가하세요.

## 참고 사항

- 이 스킬은 공식 문서에서 자동으로 생성되었습니다.
- 참조 파일은 문서 구조와 예시를 그대로 보존합니다.
- 더 나은 구문 강조를 위해 코드 예시에는 언어 감지가 포함됩니다.
- 빠른 참조 패턴은 문서에서 자주 사용되는 내용에서 추출됩니다.

## 업데이트

이 스킬을 최신 문서로 갱신하려면:
1. 동일한 설정으로 스크래퍼를 다시 실행합니다.
2. 스킬이 최신 내용으로 다시 빌드됩니다.
