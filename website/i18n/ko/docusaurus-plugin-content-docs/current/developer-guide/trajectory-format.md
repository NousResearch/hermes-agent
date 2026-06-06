# 궤적 형식 (Trajectory Format)

Hermes Agent는 훈련 데이터, 디버깅 아티팩트 및 강화학습(RL) 데이터셋으로 사용하기 위해 대화 궤적을 ShareGPT 호환 JSONL 형식으로 저장합니다.

소스 파일: `agent/trajectory.py`, `run_agent.py` (`_save_trajectory` 검색), `batch_runner.py`

## 파일 명명 규칙

궤적은 현재 작업 디렉터리의 파일에 기록됩니다:

| 파일 | 조건 |
|------|------|
| `trajectory_samples.jsonl` | 성공적으로 완료된 대화 (`completed=True`) |
| `failed_trajectories.jsonl` | 실패하거나 중단된 대화 (`completed=False`) |

일괄 실행기(batch runner)인 `batch_runner.py`는 배치를 위한 사용자 지정 출력 파일(예: `batch_001_output.jsonl`)에 추가 메타데이터 필드와 함께 기록합니다.

`save_trajectory()`의 `filename` 매개변수를 통해 파일 이름을 재정의할 수 있습니다.

## JSONL 항목 형식

파일의 각 줄은 독립적인 JSON 객체입니다. 두 가지 변형이 있습니다:

### CLI/대화형 형식 (`_save_trajectory` 사용)

```json
{
  "conversations": [ ... ],
  "timestamp": "2026-03-30T14:22:31.456789",
  "model": "anthropic/claude-sonnet-4.6",
  "completed": true
}
```

### Batch Runner 형식 (`batch_runner.py` 사용)

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

`tool_stats` 및 `tool_error_counts` 딕셔너리는 HuggingFace 데이터셋 로딩을 위한 항목 간 스키마 일관성을 보장하기 위해 기본값을 0으로 설정하여 모든 가능한 도구(`model_tools.TOOL_TO_TOOLSET_MAP` 참고)를 포함하도록 정규화됩니다.

## 대화 배열 (ShareGPT 형식)

`conversations` 배열은 ShareGPT 역할 규칙을 사용합니다:

| API 역할 | ShareGPT `from` |
|----------|-----------------|
| system | `"system"` |
| user | `"human"` |
| assistant | `"gpt"` |
| tool | `"tool"` |

### 전체 예제

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

## 정규화 규칙 (Normalization Rules)

### 추론 내용(Reasoning Content) 마크업

궤적 변환기는 모델이 원래 생성한 방식과 관계없이 모든 추론을 `<think>` 태그로 정규화합니다:

1. **네이티브 추론 토큰(Native thinking tokens)** (Anthropic, OpenAI o-시리즈 등의 `msg["reasoning"]` 필드): `<think>\n{reasoning}\n</think>\n` 형식으로 감싸며, 내용 앞에 추가됩니다.

2. **REASONING_SCRATCHPAD XML** (네이티브 추론이 비활성화되어 있고 모델이 시스템 프롬프트 지시 XML을 통해 추론할 때): `<REASONING_SCRATCHPAD>` 태그는 `convert_scratchpad_to_think()`에 의해 `<think>`로 변환됩니다.

3. **빈 추론 블록**: 모든 `gpt` 턴은 반드시 `<think>` 블록을 갖도록 보장됩니다. 생성된 추론 내용이 없는 경우, 훈련 데이터의 일관된 형식을 보장하기 위해 빈 블록 `<think>\n</think>\n` 이 삽입됩니다.

### 도구 호출 정규화 (Tool Call Normalization)

API 형식의 도구 호출(`tool_call_id`, 함수 이름, JSON 문자열 형태의 인자 포함)은 XML로 감싸진 JSON으로 변환됩니다:

```
<tool_call>
{"name": "terminal", "arguments": {"command": "ls -la"}}
</tool_call>
```

- 인자(arguments)는 JSON 문자열에서 다시 객체로 파싱됩니다 (이중 인코딩되지 않음).
- JSON 파싱에 실패하는 경우(대화 중 유효성을 검사하므로 발생해서는 안 됨), 빈 객체 `{}`가 사용되고 경고가 기록됩니다.
- 한 어시스턴트 턴의 다중 도구 호출은 단일 `gpt` 메시지에서 여러 `<tool_call>` 블록을 생성합니다.

### 도구 응답 정규화 (Tool Response Normalization)

어시스턴트 메시지 뒤에 오는 모든 도구 결과는 XML로 감싸진 JSON 응답과 함께 단일 `tool` 턴으로 그룹화됩니다:

```
<tool_response>
{"tool_call_id": "call_abc123", "name": "terminal", "content": "output here"}
</tool_response>
```

- 도구 내용이 JSON 형태( `{` 또는 `[`로 시작)인 경우 문자열이 아닌 JSON 객체/배열을 콘텐츠 필드에 포함하도록 파싱됩니다.
- 다중 도구 결과는 단일 메시지에서 줄 바꿈으로 연결됩니다.
- 도구 이름은 상위 어시스턴트의 `tool_calls` 배열을 기준으로 위치에 맞게 매칭됩니다.

### 시스템 메시지 (System Message)

시스템 메시지는 궤적 저장 시점에 생성됩니다 (대화에서 가져오지 않음). 이는 다음을 포함하는 Hermes 함수 호출 프롬프트 템플릿을 따릅니다:

- 함수 호출 프로토콜을 설명하는 서문
- JSON 도구 정의를 포함하는 `<tools>` XML 블록
- `FunctionCall` 객체를 위한 스키마 참조
- `<tool_call>` 예제

도구 정의에는 `name`, `description`, `parameters`, `required`(표준 형식과 일치하도록 `null`로 설정됨)가 포함됩니다.

## 궤적 불러오기 (Loading Trajectories)

궤적은 표준 JSONL 형식이므로 어떤 JSON-lines 리더로든 불러올 수 있습니다:

```python
import json

def load_trajectories(path: str):
    """JSONL 파일에서 궤적 항목 불러오기."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

# 성공적인 완료 항목만 필터링
successful = [e for e in load_trajectories("trajectory_samples.jsonl")
              if e.get("completed")]

# 훈련을 위해 대화(conversations)만 추출
training_data = [e["conversations"] for e in successful]
```

### HuggingFace 데이터셋용으로 불러오기

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="trajectory_samples.jsonl")
```

정규화된 `tool_stats` 스키마는 모든 항목이 동일한 열을 가지도록 보장하여 데이터셋을 불러오는 동안 Arrow 스키마 불일치 오류를 방지합니다.

## 궤적 저장 제어 (Controlling Trajectory Saving)

CLI에서 궤적 저장은 다음과 같이 제어됩니다:

```yaml
# config.yaml
agent:
  save_trajectories: true  # 기본값: false
```

또는 `--save-trajectories` 플래그를 통해 제어할 수 있습니다. 에이전트가 `save_trajectories=True`로 초기화되면 매 대화 턴이 끝날 때마다 `_save_trajectory()` 메서드가 호출됩니다.

일괄 실행기(batch runner)는 항상 궤적을 저장합니다 (이것이 주 목적입니다).

모든 턴에 걸쳐 추론 내용이 전혀 없는 샘플은 훈련 데이터가 오염되는 것을 방지하기 위해 일괄 실행기에 의해 자동으로 삭제됩니다.
