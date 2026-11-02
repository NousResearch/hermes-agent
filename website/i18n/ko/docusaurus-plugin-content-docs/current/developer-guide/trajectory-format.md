# 트라젝토리 형식

Hermes Agent는 학습 데이터, 디버깅 산출물 및 강화 학습 데이터셋에 사용하기 위해 ShareGPT 호환 JSONL 형식으로 대화 트라젝토리를 저장합니다.

소스 파일: `agent/trajectory.py`, `run_agent.py`(`_save_trajectory` 검색), `batch_runner.py`


## 파일 이름 규칙

트라젝토리는 현재 작업 디렉터리의 파일에 기록됩니다.

| 파일 | 시점 |
|------|------|
| `trajectory_samples.jsonl` | 성공적으로 완료된 대화(`completed=True`) |
| `failed_trajectories.jsonl` | 실패했거나 중단된 대화(`completed=False`) |

배치 러너(`batch_runner.py`)는 배치별 사용자 지정 출력 파일(예: `batch_001_output.jsonl`)에 추가 메타데이터 필드와 함께 기록합니다.


## JSONL 항목 형식

파일의 각 줄은 자체적으로 완결된 JSON 객체입니다. 두 가지 변형이 있습니다.

### CLI/대화형 형식(`_save_trajectory`에서 생성)

```json
{
  "conversations": [ ... ],
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```

### 배치 러너 형식(`batch_runner.py`에서 생성)

```json
{
  "prompt_index": 42,
  "conversations": [ ... ],
  "metadata": { "prompt_source": "gsm8k", "difficulty": "hard" },
  "completed": true,
  "partial": false,
  "api_calls": 7,
  "toolsets_used": ["code_tools", "file_tools"],
  "tool_stats": {
    "terminal": {"count": 3, "success": 3, "failure": 0},
    "read_file": {"count": 2, "success": 2, "failure": 0},
    "write_file": {"count": 0, "success": 0, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0,
    "write_file": 0
  }
}
```

`tool_stats` 및 `tool_error_counts` 딕셔너리는 HuggingFace 데이터셋 로딩 시 항목 간 스키마가 일관되도록 모든 가능한 도구(`model_tools.TOOL_TO_TOOLSET_MAP`에서 가져옴)를 0 기본값과 함께 포함하도록 정규화됩니다.


## 대화 배열(ShareGPT 형식)

`conversations` 배열은 ShareGPT 역할 규칙을 사용합니다.

| API 역할 | ShareGPT `from` |
|----------|-----------------|
| system | `"system"` |
| user | `"human"` |
| assistant | `"gpt"` |
| tool | `"tool"` |

### 전체 예시

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. If available tools are not relevant in assisting with user query, just respond in natural conversational language. Don't make assumptions about what values to plug into functions. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags. Here are the available tools:\n<tools>\n[{\"name\": \"terminal\", \"description\": \"Execute shell commands\", \"parameters\": {\"type\": \"object\", \"properties\": {\"command\": {\"type\": \"string\"}}}, \"required\": null}]\n</tools>\nFor each function call return a JSON object, with the following pydantic model json schema for each:\n{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\nEach function call should be enclosed within <tool_call> </tool_call> XML tags.\nExample:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    },
    {
      "from": "human",
      "value": "What Python version is installed?"
    },
    {
      "from": "gpt",
      "value": "<think>\nThe user wants to know the Python version. I should run python3 --version.\n</think>\n<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"python3 --version\"}}\n</tool_call>"
    },
    {
      "from": "tool",
      "value": "<tool_response>\n{\"tool_call_id\": \"call_abc123\", \"name\": \"terminal\", \"content\": \"Python 3.11.6\"}\n</tool_response>"
    },
    {
      "from": "gpt",
      "value": "<think>\nGot the version. I can now answer the user.\n</think>\nPython 3.11.6 is installed on this system."
    }
  ],
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```


## 정규화 규칙

### 추론 콘텐츠 마크업

트라젝토리 변환기는 모델이 원래 추론을 생성한 방식과 관계없이 모든 추론을 `<think>` 태그로 정규화합니다.

1. **네이티브 사고 토큰**(Anthropic, OpenAI o-series와 같은 제공자의 `msg["reasoning"]` 필드): `<think>\n{reasoning}\n</think>\n`으로 감싸 콘텐츠 앞에 추가합니다.

2. **REASONING_SCRATCHPAD XML**(네이티브 사고가 비활성화되고 모델이 시스템 프롬프트에서 지시한 XML을 통해 추론하는 경우): `convert_scratchpad_to_think()`를 통해 `<REASONING_SCRATCHPAD>` 태그를 `<think>`로 변환합니다.

3. **비어 있는 think 블록**: 모든 `gpt` 턴은 `<think>` 블록을 반드시 포함합니다. 추론이 생성되지 않았다면 빈 블록을 삽입합니다: `<think>\n</think>\n` — 이를 통해 학습 데이터의 형식을 일관되게 유지합니다.

### 도구 호출 정규화

API 형식의 도구 호출(`tool_call_id`, 함수 이름, JSON 문자열인 인수를 포함)은 XML로 감싼 JSON으로 변환됩니다.

```
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>
```

- 인수는 JSON 문자열에서 다시 객체로 파싱됩니다(이중 인코딩되지 않음).
- JSON 파싱에 실패하면(대화 중 검증되므로 발생하지 않아야 함) 경고를 기록하고 빈 `{}`를 사용합니다.
- 한 assistant 턴에 여러 도구 호출이 있으면 하나의 `gpt` 메시지 안에 여러 `<tool_call>` 블록을 생성합니다.

### 도구 응답 정규화

assistant 메시지 뒤에 오는 모든 도구 결과는 XML로 감싼 JSON 응답을 포함하는 하나의 `tool` 턴으로 그룹화됩니다.

```
<tool_response>
{"tool_call_id": "call_abc123", "name": "terminal", "content": "output here"}
</tool_response>
```

- 도구 콘텐츠가 JSON처럼 보이면(`{` 또는 `[`로 시작), 파싱하여 콘텐츠 필드가 문자열이 아닌 JSON 객체/배열을 포함하도록 합니다.
- 여러 도구 결과는 하나의 메시지에서 줄바꿈으로 연결됩니다.
- 도구 이름은 상위 assistant의 `tool_calls` 배열에서 위치를 기준으로 매칭됩니다.

### 시스템 메시지

시스템 메시지는 대화에서 가져오지 않고 저장 시 생성됩니다. 다음 항목을 포함하는 Hermes 함수 호출 프롬프트 템플릿을 따릅니다.

- 함수 호출 프로토콜을 설명하는 서문
- JSON 도구 정의를 포함하는 `<tools>` XML 블록
- `FunctionCall` 객체의 스키마 참조
- `<tool_call>` 예시

도구 정의에는 `name`, `description`, `parameters`, `required`가 포함되며, `required`는 표준 형식에 맞추기 위해 `null`로 설정됩니다.


## 트라젝토리 로딩

트라젝토리는 표준 JSONL이므로 모든 JSON-lines 리더로 로드할 수 있습니다.

```python
import json

def load_trajectories(path: str):
    """Load trajectory entries from a JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

# Filter to successful completions only
successful = [e for e in load_trajectories("trajectory_samples.jsonl")
              if e.get("completed")]

# Extract just the conversations for training
training_data = [e["conversations"] for e in successful]
```

### HuggingFace Datasets용 로딩

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="trajectory_samples.jsonl")
```

정규화된 `tool_stats` 스키마는 모든 항목이 동일한 열을 갖도록 보장하여, 데이터셋을 로드할 때 Arrow 스키마 불일치 오류를 방지합니다.


## 트라젝토리 저장 제어

트라젝토리 저장은 `run_agent.py`/라이브러리 수준의 스위치이며, `hermes` CLI는 이를 위한 구성 키나 플래그를 제공하지 않습니다.

```bash
python run_agent.py --save_trajectories --query='your question here'
```

또는 프로그래밍 방식으로 설정합니다: `AIAgent(..., save_trajectories=True)` / `initialize_agent(..., save_trajectories=True)`. 활성화하면 각 대화 턴이 끝날 때 `_save_trajectory()` 메서드가 호출됩니다.

배치 러너는 항상 트라젝토리를 저장합니다(이것이 배치 러너의 주요 목적입니다).

모든 턴에서 추론이 0인 샘플은 추론이 없는 예시로 학습 데이터를 오염시키지 않도록 배치 러너가 자동으로 폐기합니다.
