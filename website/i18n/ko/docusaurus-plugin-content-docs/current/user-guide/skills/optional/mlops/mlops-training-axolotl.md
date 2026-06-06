---
title: "Axolotl — Axolotl: YAML LLM 파인튜닝 (LoRA, DPO, GRPO)"
sidebar_label: "Axolotl"
description: "Axolotl: YAML LLM 파인튜닝 (LoRA, DPO, GRPO)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Axolotl

Axolotl: YAML 기반의 LLM 파인튜닝 (LoRA, DPO, GRPO).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/axolotl`로 설치 |
| 경로 | `optional-skills/mlops/training/axolotl` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `axolotl`, `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `deepspeed` |
| 플랫폼 | linux, macos |
| 태그 | `Fine-Tuning`, `Axolotl`, `LLM`, `LoRA`, `QLoRA`, `DPO`, `KTO`, `ORPO`, `GRPO`, `YAML`, `HuggingFace`, `DeepSpeed`, `Multimodal` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Axolotl 스킬

## 포함된 내용

Axolotl을 사용한 LLM 파인튜닝에 대한 전문가 가이드 — YAML 구성, 100개 이상의 모델, LoRA/QLoRA, DPO/KTO/ORPO/GRPO, 멀티모달 지원.

공식 문서로부터 생성된 axolotl 개발에 대한 포괄적인 지원.

## 이 스킬을 사용하는 경우

이 스킬은 다음과 같은 경우에 트리거해야 합니다:
- axolotl 관련 작업을 할 때
- axolotl 기능이나 API에 대해 질문할 때
- axolotl 솔루션을 구현할 때
- axolotl 코드를 디버깅할 때
- axolotl 모범 사례를 배울 때

## 빠른 참조

### 일반적인 패턴

**패턴 1:** 학습 작업에 허용 가능한 데이터 전송 속도가 존재하는지 확인하려면, NCCL 테스트를 실행하여 병목 지점을 찾을 수 있습니다. 예를 들면:

```
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 3
```

**패턴 2:** Axolotl yaml에서 FSDP를 사용하도록 모델을 구성합니다. 예를 들면:

```
fsdp_version: 2
fsdp_config:
  offload_params: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

**패턴 3:** context_parallel_size는 전체 GPU 수의 약수여야 합니다. 예를 들면:

```
context_parallel_size
```

**패턴 4:** 예를 들면: - 8개의 GPU를 사용하고 시퀀스 병렬 처리가 없는 경우: 스텝 당 8개의 다른 배치가 처리됩니다 - 8개의 GPU를 사용하고 context_parallel_size=4인 경우: 스텝 당 2개의 다른 배치만 처리됩니다 (각각 4개의 GPU에 분할됨) - GPU 당 micro_batch_size가 2인 경우, 글로벌 배치 크기는 16에서 4로 감소합니다.

```
context_parallel_size=4
```

**패턴 5:** 구성에서 save_compressed: true 로 설정하면 압축된 형식으로 모델 저장을 활성화하며, 이는: - 디스크 공간 사용량을 약 40% 줄입니다 - 가속화된 추론을 위해 vLLM과의 호환성을 유지합니다 - 추가 최적화(예: 양자화)를 위해 llmcompressor와의 호환성을 유지합니다

```
save_compressed: true
```

**패턴 6:** 참고: 여러분의 연동(integration) 코드를 반드시 integrations 폴더에 넣을 필요는 없습니다. 파이썬 환경의 패키지에 설치되어 있기만 하면 어느 위치에 있어도 상관없습니다. 예시는 다음 저장소를 확인하세요: https://github.com/axolotl-ai-cloud/diff-transformer

```
integrations
```

**패턴 7:** 단일 예제와 일괄(batched) 데이터를 모두 처리합니다. - 단일 예제: sample['input_ids']는 list[int] 입니다 - 일괄 데이터: sample['input_ids']는 list[list[int]] 입니다

```
utils.trainer.drop_long_seq(sample, sequence_len=2048, min_sequence_len=2)
```

### 예제 코드 패턴

**예제 1** (python):
```python
cli.cloud.modal_.ModalCloud(config, app=None)
```

**예제 2** (python):
```python
cli.cloud.modal_.run_cmd(cmd, run_folder, volumes=None)
```

**예제 3** (python):
```python
core.trainers.base.AxolotlTrainer(
    *_args,
    bench_data_collator=None,
    eval_data_collator=None,
    dataset_tags=None,
    **kwargs,
)
```

**예제 4** (python):
```python
core.trainers.base.AxolotlTrainer.log(logs, start_time=None)
```

**예제 5** (python):
```python
prompt_strategies.input_output.RawInputOutputPrompter()
```

## 참고 파일

이 스킬에는 `references/` 에 포괄적인 문서가 포함되어 있습니다:

- **api.md** - Api 문서
- **dataset-formats.md** - Dataset-Formats 문서
- **other.md** - 기타 문서

상세한 정보가 필요할 때 특정 참고 파일을 읽기 위해 `view`를 사용하세요.

## 이 스킬로 작업하기

### 초보자를 위해
기본적인 개념을 익히기 위해 getting_started 또는 tutorials 참고 파일을 시작으로 사용하세요.

### 특정 기능을 위해
상세한 정보를 위해 알맞은 카테고리 참고 파일(api, guides 등)을 사용하세요.

### 코드 예제를 위해
위의 빠른 참조 섹션에는 공식 문서에서 추출한 일반적인 사용 패턴이 포함되어 있습니다.

## 리소스

### references/
공식 출처에서 추출된 체계적인 문서입니다. 이 파일들은 다음을 포함합니다:
- 상세한 설명
- 향상된 구문 강조를 위한 언어 주석이 포함된 코드 예제
- 원본 문서로의 링크
- 빠른 탐색을 위한 목차 (Table of contents)

### scripts/
일반적인 자동화 작업을 위한 도우미 스크립트를 여기에 추가하세요.

### assets/
템플릿, 보일러플레이트, 또는 예제 프로젝트를 여기에 추가하세요.

## 참고

- 이 스킬은 공식 문서에서 자동으로 생성되었습니다.
- 참고 파일은 원본 문서의 구조와 예제를 유지합니다.
- 향상된 구문 강조를 위해 코드 예제에 언어 감지 기능이 포함되어 있습니다.
- 문서의 일반적인 사용 예시에서 빠른 참조 패턴을 추출합니다.

## 업데이트

이 스킬의 문서를 최신 정보로 갱신하려면:
1. 동일한 설정으로 스크래퍼를 다시 실행합니다.
2. 스킬이 최신 정보로 다시 빌드됩니다.
