---
sidebar_position: 2
title: "AI 프로바이더"
description: "로컬 AI, OpenRouter, Anthropic, OpenAI, LiteLLM, vLLM, SGLang, Ollama 등의 백엔드 인퍼런스 프로바이더를 구성하세요."
---

# AI 프로바이더 구성

Hermes Agent는 모델 제공(serving)을 처리하지 않으며, 백엔드 "인퍼런스 프로바이더"에 연결됩니다. 단일 모델(예: 로컬 Ollama 또는 Anthropic)에 직접 연결하거나, 수백 개의 모델이 포함된 라우팅 프록시(OpenRouter 또는 LiteLLM)를 통해 연결할 수 있습니다.

`~/.hermes/config.yaml`의 기본 설정은 다음과 같습니다:

```yaml
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4.6
```

이 가이드에서는 클라우드 API, 자체 호스팅 모델, 데스크톱 앱 및 사용자 정의 엔드포인트 등 지원되는 모든 프로바이더를 연결하는 방법을 설명합니다.

:::tip CLI를 사용하여 구성하기
`hermes model`을 실행하면 대화형 마법사가 열립니다. 이 도구는 지원되는 모든 프로바이더를 검색하고, 사용 가능한 모델을 가져오며, `config.yaml`과 환경 변수를 자동으로 구성합니다. 이 가이드의 모든 내용은 이 도구를 사용하여 설정할 수 있습니다.
:::

## 권장 프로바이더

빠르게 시작하고 싶다면 이 옵션 중 하나를 선택하세요:

### 1. OpenRouter (기본값)
[OpenRouter](https://openrouter.ai/)는 단일 API 키로 수백 개의 모델(Claude, GPT, Gemini, Llama, DeepSeek)에 대한 액세스를 제공합니다.
- **장점:** 설정 없이 모델 간 전환(예: 코딩을 위한 Sonnet 4.6, 비용 절감을 위한 Llama-3-70B). 가장 넓은 컨텍스트 지원, 라우팅 폴백(fallback).
- **설정:** `hermes auth add openrouter` (브라우저에서 로그인) 또는 환경 변수 `OPENROUTER_API_KEY` 설정.

### 2. Nous Portal (권장)
[Nous Portal](https://portal.nousresearch.com)은 Hermes를 실행하기 위해 권장되는 방법입니다. OpenRouter 모델 카탈로그와 [Tool Gateway](/integrations/nous-portal) (웹 검색, 브라우저 자동화, 텍스트-음성 변환, 이미지 생성)를 단일 구독 청구로 제공합니다.
- **장점:** 5개의 도구 계정을 따로 가입할 필요 없이 하나로 통합됩니다.
- **설정:** `hermes setup --portal` 실행.

### 3. Ollama (로컬)
[Ollama](https://ollama.com/)는 데스크톱에서 오픈소스 모델(Llama, Qwen, Mistral)을 실행하기 위한 가장 쉬운 방법입니다.
- **장점:** 무료, 프라이빗, 오프라인.
- **설정:** 앱을 다운로드하고 `ollama run qwen2.5-coder:32b`를 실행한 후 `hermes model`에서 Ollama를 선택하세요.
- **참고:** 에이전트 용도로 사용하려면 구성이 필요합니다. 아래 [Ollama 섹션](#ollama--로컬-데스크톱-인퍼런스)을 참조하세요.

---

## 클라우드 API 프로바이더

### Anthropic
- **지원 모델:** Claude 4.6 (Opus/Sonnet), Claude 4.5 (Haiku), Claude 4
- **설정:** 환경 변수 `ANTHROPIC_API_KEY` 설정.
- **장점:** 네이티브 도구 호출(tool calling), 최고의 코딩 모델(Sonnet 4.6).

### OpenAI
- **지원 모델:** GPT-5.5, GPT-5.4, GPT-5.3 시리즈
- **설정:** 환경 변수 `OPENAI_API_KEY` 설정.
- **장점:** 일관성, 구조화된 출력(structured output), 네이티브 함수 호출.

### Google Gemini
- **지원 모델:** Gemini 3 Pro/Flash 시리즈, Gemini 3.1 Pro/Flash/Flash-Lite 시리즈
- **설정:** 환경 변수 `GEMINI_API_KEY` 설정.
- **장점:** 방대한 컨텍스트 윈도우(2M+), 비디오 및 오디오 이해를 위한 강력한 멀티모달 기능.

---

## 로컬 및 자체 호스팅 서버

자체 하드웨어나 클라우드 인스턴스에서 모델을 호스팅하는 경우, Hermes는 `base_url` 재정의를 통해 OpenAI 호환 엔드포인트에 연결할 수 있습니다.

### Ollama — 로컬 데스크톱 인퍼런스

[Ollama](https://ollama.com/)를 사용하면 백그라운드 서비스에서 로컬 모델을 쉽게 실행할 수 있습니다. 코딩 및 일반 에이전트 작업에는 **Qwen 2.5 Coder 32B** (`qwen2.5-coder:32b`) 또는 **Llama 3 70B** (`llama-3-70b-instruct`)를 권장합니다. 8B 모델은 에이전트 작업을 수행하기에 너무 작아서 계속 반복적인 헛소리(hallucinate)를 할 것입니다.

```bash
hermes model
# Ollama 선택
# 모델 선택
```

:::warning Ollama 컨텍스트 윈도우 한계
Ollama는 기본적으로 **2048개의 토큰**만 메모리에 로드합니다. 에이전트 작업은 보통 시작할 때 시스템 프롬프트만으로 4000개 이상의 토큰을 소모하므로, Ollama는 Hermes의 프롬프트 절반을 무시하고 대화 기록도 잘라냅니다! 이로 인해 엉뚱한 행동, 루프 발생, 도구 실행 실패가 발생할 수 있습니다.

**해결 방법:** `~/.hermes/config.yaml`의 모델 구성에 `num_ctx`를 추가하세요. (최소 **32768**, VRAM이 허용한다면 **65536** 권장):

```yaml
model:
  default: "qwen2.5-coder:32b"
  provider: ollama
  ollama:
    num_ctx: 65536
```
:::

Ollama가 다른 머신에 있나요? `config.yaml`에 호스트를 설정하세요:

```yaml
model:
  default: "qwen2.5-coder:32b"
  provider: ollama
  base_url: "http://192.168.1.100:11434/v1"
```

*참고:* 다른 머신의 연결을 허용하려면 호스트 시스템에서 `OLLAMA_HOST=0.0.0.0`을 설정해야 합니다.

---

### vLLM — 프로덕션급 GPU 서빙

[vLLM](https://github.com/vllm-project/vllm)은 데이터 센터 또는 멀티 GPU 워크스테이션에서 가장 인기 있는 고성능 서빙 엔진입니다. 대량 처리(Prefix caching, PagedAttention)에 최적화되어 있습니다.

```bash
# OpenAI 호환 서버 시작
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

그 다음 Hermes를 구성하세요:

```bash
hermes model
# "Custom endpoint (self-hosted / VLLM / etc.)" 선택
# URL 입력: http://localhost:8000/v1
# 모델 이름 입력: meta-llama/Llama-3.1-70B-Instruct
```

:::caution 도구 호출 파서(Tool Call Parser) 필수
vLLM에서는 `--enable-auto-tool-choice`와 `--tool-call-parser` 플래그를 사용하여 명시적으로 도구 호출을 활성화해야 합니다. 그렇지 않으면 일반 텍스트로 응답합니다. 모델에 따라 적절한 파서(예: `hermes`, `llama3_json`)를 선택하세요.
:::

---

### SGLang — RadixAttention을 통한 빠른 서빙

[SGLang](https://github.com/sgl-project/sglang)은 KV 캐시 재사용을 위한 RadixAttention 기능을 제공하는 vLLM의 대안입니다. 멀티턴 대화, 제한된 디코딩(constrained decoding), 구조화된 출력에 최적화되어 있습니다.

```bash
pip install "sglang[all]"
python -m sglang.launch_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 \
  --tool-call-parser qwen
```

그 다음 Hermes를 구성하세요:

```bash
hermes model
# "Custom endpoint (self-hosted / VLLM / etc.)" 선택
# URL 입력: http://localhost:30000/v1
# 모델 이름 입력: meta-llama/Llama-3.1-70B-Instruct
```

:::caution SGLang의 기본 출력 토큰 수
응답이 잘린 경우 `max_tokens`를 요청에 추가하거나 서버에 `--default-max-tokens`를 설정하세요. SGLang의 기본값은 128 토큰입니다.
:::

---

### llama.cpp / llama-server — CPU 및 Metal 인퍼런스

[llama.cpp](https://github.com/ggml-org/llama.cpp)는 CPU, Apple Silicon(Metal) 및 소비자용 GPU에서 양자화된 모델을 실행합니다.

```bash
# llama-server 빌드 및 시작
cmake -B build && cmake --build build --config Release
./build/bin/llama-server \
  --jinja -fa \
  -c 64000 \
  -ngl 99 \
  -m models/qwen2.5-coder-32b-instruct-Q4_K_M.gguf \
  --port 8080 --host 0.0.0.0
```

:::caution `--jinja` 플래그는 도구 호출에 필수입니다
`--jinja`가 없으면 llama-server는 `tools` 매개변수를 완전히 무시합니다.
:::

---

### LM Studio — 로컬 모델을 위한 데스크톱 앱

[LM Studio](https://lmstudio.ai/)는 로컬 모델을 GUI로 실행하기 위한 데스크톱 앱입니다.

```bash
lms server start                        # 포트 1234에서 시작
lms load qwen2.5-coder --context-length 64000
```

그 다음 Hermes를 구성하세요:

```bash
hermes model
# "LM Studio" 선택
# Enter 키를 눌러 http://localhost:1234/v1 사용
# 감지된 모델 중 하나를 선택
```

---

### WSL2 네트워킹 (Windows 사용자)

Hermes Agent를 WSL2 내부에서 실행하고 모델 서버(Ollama, LM Studio 등)가 **Windows 호스트**에 있는 경우, 네트워크 연결을 설정해야 합니다.

#### 옵션 1: 미러링 네트워킹 모드 (권장, Windows 11 22H2+)

1. `%USERPROFILE%\.wslconfig` 파일을 생성하거나 수정합니다:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
2. PowerShell에서 WSL을 재시작합니다: `wsl --shutdown`

#### 옵션 2: Windows 호스트 IP 사용 (Windows 10 / 이전 빌드)

WSL2 내에서 Windows 호스트 IP를 찾아 `base_url`로 사용합니다:
```bash
ip route show | grep -i default | awk '{ print $3 }'
```
모델 서버 설정에서 수신(Bind) 주소를 `0.0.0.0`으로 설정하고 방화벽 규칙을 추가해야 할 수도 있습니다.

---

### LiteLLM Proxy — 다중 프로바이더 게이트웨이

[LiteLLM](https://docs.litellm.ai/)은 100개 이상의 LLM 프로바이더를 단일 API 뒤에 통합하는 프록시입니다.

```bash
pip install "litellm[proxy]"
litellm --config litellm_config.yaml --port 4000
```
Hermes의 사용자 정의 엔드포인트 URL을 `http://localhost:4000/v1`로 설정하세요.

---

### ClawRouter — 비용 최적화 라우팅

[ClawRouter](https://github.com/BlockRunAI/ClawRouter)는 블록체인 결제(USDC)를 사용하여 모델을 자동으로 선택하고 라우팅하는 프록시입니다.
```bash
npx @blockrun/clawrouter
```
사용자 정의 엔드포인트 URL을 `http://localhost:8402/v1`, 모델 이름을 `blockrun/auto`로 설정하세요.

---

### 기타 호환 프로바이더

OpenAI 호환 API를 지원하는 서비스는 모두 사용할 수 있습니다. `config.yaml`에 추가하거나 마법사를 통해 설정하세요:

| 프로바이더 | Base URL |
|----------|----------|
| Together AI | `https://api.together.xyz/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` |

---

### 컨텍스트 길이 감지

**`context_length`**는 대화 기록에 허용되는 전체 토큰 예산을 의미합니다 (예: 200,000). **`model.max_tokens`**는 단일 응답의 출력 토큰 한도입니다.

Hermes는 API, 캐시, 설정 파일 등의 여러 소스를 확인하여 컨텍스트 길이를 자동으로 감지합니다. 이 값이 잘못 감지된 경우 명시적으로 설정할 수 있습니다:

```yaml
model:
  default: "qwen3.5:9b"
  base_url: "http://localhost:8080/v1"
  context_length: 131072
```

---

### 이름이 지정된 사용자 정의 프로바이더 (Named Custom Providers)

여러 사용자 정의 엔드포인트를 `config.yaml`에 정의하여 세션 중에 빠르게 전환할 수 있습니다:

```yaml
custom_providers:
  - name: local
    base_url: http://localhost:8080/v1
  - name: work
    base_url: https://gpu-server.internal.corp/v1
    key_env: CORP_API_KEY
    api_mode: chat_completions
```

전환 방법:
```
/model custom:local:qwen-2.5
/model custom:work:llama3-70b
```

---

## 추가 정보 및 요리책 (Cookbook)

Together AI, Groq, Perplexity와 같은 프로바이더 설정 및 다중 프로바이더 구성의 전체 요리책 예시는 `~/.hermes/config.yaml`과 환경 변수 설정을 통해 가능합니다.

| 상황 | 권장 |
|----------|-------------|
| 설정 없이 시작 | OpenRouter (기본값) 또는 Nous Portal |
| 로컬 모델, 쉬운 설정 | Ollama |
| 프로덕션 GPU 서빙 | vLLM 또는 SGLang |
| Mac / GPU 없음 | Ollama 또는 llama.cpp |

자세한 내용은 [구성(Configuration)](/user-guide/configuration) 및 [대체 프로바이더(Fallback Providers)](/user-guide/features/fallback-providers) 가이드를 참조하세요.
