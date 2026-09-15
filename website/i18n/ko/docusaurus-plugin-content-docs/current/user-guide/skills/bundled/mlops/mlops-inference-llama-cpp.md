---
title: "Llama Cpp — llama.cpp 로컬 GGUF 추론 + HF Hub 모델 검색"
sidebar_label: "Llama Cpp"
description: "llama.cpp 로컬 GGUF 추론 + HF Hub 모델 검색"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llama Cpp

llama.cpp 로컬 GGUF 추론 및 HF Hub 모델 검색.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 제공(기본 설치) |
| 경로 | `skills/mlops/inference/llama-cpp` |
| 버전 | `2.1.2` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `llama-cpp-python>=0.2.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `llama.cpp`, `GGUF`, `Quantization`, `Hugging Face Hub`, `CPU Inference`, `Apple Silicon`, `Edge Deployment`, `AMD GPUs`, `Intel GPUs`, `NVIDIA`, `URL-first` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# llama.cpp + GGUF

llama.cpp를 위한 로컬 GGUF 추론, 양자화 선택 또는 Hugging Face 저장소 검색에 이 스킬을 사용하세요.

## 사용 시점

- CPU, Apple Silicon, CUDA, ROCm 또는 Intel GPU에서 로컬 모델 실행
- 특정 Hugging Face 저장소에 맞는 GGUF 찾기
- Hub에서 `llama-server` 또는 `llama-cli` 명령 만들기
- 이미 llama.cpp를 지원하는 모델을 Hub에서 검색
- 저장소에서 사용 가능한 `.gguf` 파일과 크기 열거
- 사용자의 RAM 또는 VRAM에 맞는 Q4/Q5/Q6/IQ 변형 선택

## 모델 검색 워크플로

`hf`, Python 또는 사용자 지정 스크립트를 요청하기 전에 URL 워크플로를 우선하세요.

1. Hub에서 후보 저장소 검색:
   - 기본값: `https://huggingface.co/models?apps=llama.cpp&sort=trending`
   - 모델 제품군 검색에는 `search=<term>` 추가
   - 사용자가 크기 제약을 제시한 경우 `num_parameters=min:0,max:24B` 또는 유사한 값 추가
2. llama.cpp 로컬 앱 보기로 저장소 열기:
   - `https://huggingface.co/<repo>?local-app=llama.cpp`
3. 로컬 앱 스니펫이 표시되면 이를 단일 기준으로 취급:
   - 정확한 `llama-server` 또는 `llama-cli` 명령 복사
   - HF에 표시된 권장 양자화를 정확히 보고
4. 동일한 `?local-app=llama.cpp` URL을 페이지 텍스트 또는 HTML로 읽고 `Hardware compatibility` 아래 섹션 추출:
   - 일반 표보다 정확한 양자화 레이블과 크기를 우선
   - `UD-Q4_K_M` 또는 `IQ4_NL_XL` 같은 저장소별 레이블 유지
   - 가져온 페이지 소스에 해당 섹션이 표시되지 않으면 그 사실을 밝히고 tree API 및 일반적인 양자화 지침으로 대체
5. tree API를 조회하여 실제로 존재하는 항목 확인:
   - `https://huggingface.co/api/models/<repo>/tree/main?recursive=true`
   - `type`이 `file`이고 `path`가 `.gguf`로 끝나는 항목 유지
   - 파일명과 바이트 크기의 기준으로 `path`와 `size` 사용
   - 양자화된 체크포인트를 `mmproj-*.gguf` 프로젝터 파일 및 `BF16/` 샤드 파일과 분리
   - `https://huggingface.co/<repo>/tree/main`은 사람이 확인할 때만 사용
6. 로컬 앱 스니펫이 텍스트로 표시되지 않으면 저장소와 선택한 양자화로 명령 재구성:
   - 간단한 양자화 선택: `llama-server -hf <repo>:<QUANT>`
   - 정확한 파일 대체 방법: `llama-server --hf-repo <repo> --hf-file <filename.gguf>`
7. 저장소에서 이미 GGUF 파일을 제공하지 않는 경우에만 Transformers 가중치에서 변환하도록 제안

## 빠른 시작

### llama.cpp 설치

```bash
# macOS / Linux (simplest)
brew install llama.cpp
```

```bash
winget install llama.cpp
```

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### Hugging Face Hub에서 직접 실행

```bash
llama-cli -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

```bash
llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
```

### Hub에서 정확한 GGUF 파일 실행

tree API에 사용자 지정 파일 이름이 표시되거나 정확한 HF 스니펫이 없는 경우 사용하세요.

```bash
llama-server \
    --hf-repo microsoft/Phi-3-mini-4k-instruct-gguf \
    --hf-file Phi-3-mini-4k-instruct-q4.gguf \
    -c 4096
```

### OpenAI 호환 서버 확인

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a limerick about Python exceptions"}
    ]
  }'
```

## Python 바인딩(llama-cpp-python)

`pip install llama-cpp-python`(CUDA: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`; Metal: `CMAKE_ARGS="-DGGML_METAL=on" ...`).

### 기본 생성

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,     # 0 for CPU, 99 to offload everything
    n_threads=8,
)

out = llm("What is machine learning?", max_tokens=256, temperature=0.7)
print(out["choices"][0]["text"])
```

### 채팅 + 스트리밍

```python
llm = Llama(
    model_path="./model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,
    chat_format="llama-3",   # or "chatml", "mistral", etc.
)

resp = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ],
    max_tokens=256,
)
print(resp["choices"][0]["message"]["content"])

# Streaming
for chunk in llm("Explain quantum computing:", max_tokens=256, stream=True):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

### 임베딩

```python
llm = Llama(model_path="./model-q4_k_m.gguf", embedding=True, n_gpu_layers=35)
vec = llm.embed("This is a test sentence.")
print(f"Embedding dimension: {len(vec)}")
```

Hub에서 GGUF를 바로 로드할 수도 있습니다.

```python
llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=35,
)
```

## 양자화 선택

먼저 Hub 페이지를 사용하고, 그다음 일반적인 휴리스틱을 사용하세요.

- HF가 사용자의 하드웨어 프로필에 호환된다고 표시한 정확한 양자화를 우선하세요.
- 일반적인 채팅은 `Q4_K_M`으로 시작하세요.
- 코드 또는 기술 작업에는 메모리가 허용되는 경우 `Q5_K_M` 또는 `Q6_K`를 우선하세요.
- RAM 예산이 매우 빠듯한 경우, 사용자가 품질보다 적합성을 명시적으로 우선할 때만 `Q3_K_M`, `IQ` 변형 또는 `Q2` 변형을 고려하세요.
- 멀티모달 저장소에서는 `mmproj-*.gguf`를 별도로 언급하세요. 프로젝터는 주 모델 파일이 아닙니다.
- 저장소 고유 레이블을 정규화하지 마세요. 페이지에 `UD-Q4_K_M`이라고 표시되면 `UD-Q4_K_M`으로 보고하세요.

## 저장소에서 사용 가능한 GGUF 추출

사용자가 어떤 GGUF가 존재하는지 물으면 다음을 반환하세요.

- 파일명
- 파일 크기
- 양자화 레이블
- 주 모델인지 보조 프로젝터인지

요청하지 않는 한 다음은 무시하세요.

- README
- BF16 샤드 파일
- imatrix blob 또는 보정 아티팩트

이 단계에는 tree API를 사용하세요.

- `https://huggingface.co/api/models/<repo>/tree/main?recursive=true`

`unsloth/Qwen3.6-35B-A3B-GGUF` 같은 저장소의 경우 로컬 앱 페이지에 `UD-Q4_K_M`, `UD-Q5_K_M`, `UD-Q6_K`, `Q8_0` 같은 양자화 칩이 표시될 수 있으며, tree API는 `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` 및 `Qwen3.6-35B-A3B-Q8_0.gguf` 같은 정확한 파일 경로와 바이트 크기를 노출합니다. tree API를 사용하여 양자화 레이블을 정확한 파일명으로 변환하세요.

## 검색 패턴

다음 URL 형식을 직접 사용하세요.

```text
https://huggingface.co/models?apps=llama.cpp&sort=trending
https://huggingface.co/models?search=<term>&apps=llama.cpp&sort=trending
https://huggingface.co/models?search=<term>&apps=llama.cpp&num_parameters=min:0,max:24B&sort=trending
https://huggingface.co/<repo>?local-app=llama.cpp
https://huggingface.co/api/models/<repo>/tree/main?recursive=true
https://huggingface.co/<repo>/tree/main
```

## 출력 형식

검색 요청에 답할 때 다음과 같이 간결하고 구조화된 결과를 우선하세요.

```text
Repo: <repo>
Recommended quant from HF: <label> (<size>)
llama-server: <command>
Other GGUFs:
- <filename> - <size>
- <filename> - <size>
Source URLs:
- <local-app URL>
- <tree API URL>
```

## 참조 자료

- **[hub-discovery.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/hub-discovery.md)** - URL 전용 Hugging Face 워크플로, 검색 패턴, GGUF 추출 및 명령 재구성
- **[advanced-usage.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/advanced-usage.md)** — speculative decoding, batched inference, grammar-constrained generation, LoRA, multi-GPU, custom builds, benchmark scripts
- **[quantization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/quantization.md)** — 양자화 품질 트레이드오프, Q4/Q5/Q6/IQ 사용 시점, 모델 크기 확장, imatrix
- **[server.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/server.md)** — Hub에서 직접 서버 실행, OpenAI API 엔드포인트, Docker 배포, NGINX 로드 밸런싱, 모니터링
- **[optimization.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/optimization.md)** — CPU 스레딩, BLAS, GPU 오프로딩 휴리스틱, 배치 튜닝, 벤치마크
- **[troubleshooting.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/inference/llama-cpp/references/troubleshooting.md)** — 설치/변환/양자화/추론/서버 문제, Apple Silicon, 디버깅

## 리소스

- **GitHub**: https://github.com/ggml-org/llama.cpp
- **Hugging Face GGUF + llama.cpp 문서**: https://huggingface.co/docs/hub/gguf-llamacpp
- **Hugging Face Local Apps 문서**: https://huggingface.co/docs/hub/main/local-apps
- **Hugging Face Local Agents 문서**: https://huggingface.co/docs/hub/agents-local
- **로컬 앱 예시 페이지**: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?local-app=llama.cpp
- **예시 tree API**: https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main?recursive=true
- **llama.cpp 검색 예시**: https://huggingface.co/models?num_parameters=min:0,max:24B&apps=llama.cpp&sort=trending
- **라이선스**: MIT
