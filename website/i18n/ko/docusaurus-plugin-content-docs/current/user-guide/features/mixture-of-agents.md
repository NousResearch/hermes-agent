---
sidebar_position: 7
title: "Mixture of Agents"
description: "Mixture of Agents에서 선택 가능한 모델로 표시되는 이름 있는 MoA 프리셋 생성"
---

# Mixture of Agents

Mixture of Agents는 가상 모델 제공자입니다. 이름이 지정된 각 MoA 프리셋은 `moa` 제공자 아래에서 선택 가능한 모델로 표시됩니다.

MoA 프리셋을 선택하면 해당 프리셋의 애그리게이터가 실제로 작동하는 모델이 됩니다. 이 모델이 어시스턴트 응답을 작성하고 도구 호출을 생성합니다. 참조 모델은 먼저 실행되어 애그리게이터가 사용할 분석을 제공합니다.

여러 모델의 관점이 도움이 되는 어려운 작업을 수행하면서도 도구 호출, 후속 반복, 중단, 대화 기록 유지, 다른 메시지와 동일한 세션 컨텍스트 등 Hermes의 일반 에이전트 루프가 필요한 경우 MoA를 사용하세요.

## 모델로 MoA 프리셋 선택

일반적인 모델 선택 화면에서 프리셋을 선택할 수 있습니다:

```bash
/model default --provider moa
/model review --provider moa
```

MoA는 모델 시스템의 일반 제공자이므로 **모든 Hermes 화면**에서 MoA 프리셋을 선택할 수 있습니다:

- **CLI / gateway / TUI `/model`** — `/model <preset> --provider moa` 또는 기본 프리셋에는 `/model --provider moa`를 사용합니다. 구성된 프리셋과 이름이 정확히 일치하면 일반적인 `/model <preset>`도 작동합니다.
- **`hermes model` 및 **Dashboard 모델 선택기** — 프리셋 이름을 모델로 표시하는 `Mixture of Agents` 제공자 행이 나타납니다.
- **Desktop GUI 앱** — 모델 드롭다운에 `MoA presets` 섹션이 표시됩니다. 프리셋(`MoA: <preset>`)을 선택하면 활성 모델이 해당 프리셋으로 전환됩니다. Desktop 설정 패널에서도 프리셋을 생성하고 편집할 수 있습니다.

따라서 구성된 프리셋은 다른 모델을 선택할 수 있는 모든 곳에 표시됩니다.

## 슬래시 명령 단축키

`/moa`는 한 번만 사용하는 편의용 문법입니다. **기본** MoA 프리셋을 통해 단일 프롬프트를 실행한 다음, 사용 중이던 모델을 복원합니다:

```bash
/moa design and implement a migration plan for this flaky test cluster
```

Hermes는 해당 한 턴 동안 기본 MoA 프리셋으로 임시 전환하고 프롬프트를 보낸 다음, 이후 이전 모델을 복원합니다. 전체 인수가 프롬프트이며, `/moa`는 더 이상 이를 프리셋 이름으로 해석하지 않습니다.

```bash
/moa
```

프롬프트 없이 `/moa`만 입력하면 사용법만 출력합니다.

세션의 나머지 시간 동안 MoA 프리셋으로 **전환**하려면 모델 선택기에서 해당 프리셋을 선택하세요. 위에서 설명한 대로 모든 모델 선택 화면의 `Mixture of Agents` 제공자 아래에 MoA 프리셋이 표시됩니다. `/moa`는 의도적으로 모델 전환 명령이 아니므로 일반 프롬프트가 실수로 모델을 변경하는 일은 없습니다.

## 에이전트 루프에서의 작동 방식

제공자 `moa`를 선택한 경우 각 주 모델 호출마다 Hermes는 다음을 수행합니다:

1. 이름으로 선택한 프리셋을 확인합니다.
2. 도구 스키마 없이 구성된 참조 모델을 실행합니다(참조 모델에는 대화의 사용자/어시스턴트 텍스트만 전달되며 Hermes 시스템 프롬프트나 도구 호출 기록은 전달되지 않습니다. 따라서 참조 호출 비용을 낮추고 엄격한 제공자의 거부를 방지합니다).
3. 참조 모델의 출력을 애그리게이터가 사용할 비공개 컨텍스트로 추가합니다.
4. 일반 Hermes 도구 스키마와 함께 구성된 애그리게이터를 호출합니다.
5. 애그리게이터의 응답을 실제 모델 응답으로 처리합니다.
6. 애그리게이터가 도구를 호출하면 Hermes가 평소와 같이 해당 도구를 실행합니다.
7. 다음 모델 반복에서 업데이트된 대화에 대해 동일한 MoA 프로세스를 다시 실행합니다. 여기에는 도구 결과도 포함됩니다.

MoA는 일반 모델 시스템을 통해 선택되므로 `/goal`, gateway 세션, TUI 세션, Desktop 채팅과 자동으로 함께 사용할 수 있습니다.

## 프리셋 구성

다음 위치에서 이름 있는 MoA 프리셋을 구성할 수 있습니다:

- Dashboard → Models → Model Settings → Mixture of Agents
- Desktop 앱 → Settings → Model → Mixture of Agents
- `hermes moa configure [name]`
- `config.yaml`

구성에는 명시적인 제공자/모델 쌍이 저장되므로 제공자를 섞고 동일한 제공자의 여러 모델을 사용할 수 있습니다:

```yaml
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      # Optional: pin sampling temperatures. When omitted (the default),
      # temperature is NOT sent and each model uses its provider default —
      # the same behavior as a single-model Hermes agent.
      # reference_temperature: 0.6
      # aggregator_temperature: 0.4
      max_tokens: 4096
      enabled: true
```

기본 프리셋:

- 참조: `openai-codex:gpt-5.5`
- 참조: `openrouter:deepseek/deepseek-v4-pro`
- 애그리게이터 / 작동 모델: `openrouter:anthropic/claude-opus-4.8`

### `reference_max_tokens`로 자문 속도 조정

각 턴마다 MoA는 참조 모델(자문 모델)을 병렬로 실행한 다음 애그리게이터가 작동합니다. 자문 모델의 생성이 턴당 지연 시간의 주된 원인입니다. 자문 모델이 출력을 작성하는 데 사용하는 토큰 수가 많을수록 턴이 가장 느린 자문 모델의 완료를 기다리기 때문에 턴의 전체 소요 시간이 강하게 연관됩니다. 기본적으로 자문 모델은 **제한이 없습니다**(`reference_max_tokens`가 설정되지 않음). 따라서 에세이처럼 긴 조언을 작성할 수 있습니다.

프리셋에서 `reference_max_tokens`를 설정하여 자문 모델 출력에 상한을 두고 간결한 조언을 제공하도록 하세요. 애그리게이터에는 각 자문 모델 판단의 요지만 필요하므로 상한(예: `600`)을 두면 품질에 미치는 영향은 적으면서 턴당 전체 소요 시간을 측정 가능하게 줄일 수 있습니다. 상한은 **자문 모델에만** 적용되며, 사용자가 보는 응답을 생성하는 작동 애그리게이터의 출력에는 절대 적용되지 않습니다.

```yaml
moa:
  presets:
    fast:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
        - provider: openrouter
          model: openai/gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      reference_max_tokens: 600   # concise advice → faster turns
```

이전의 제한 없는 동작을 유지하려면 설정하지 않거나(`0` 또는 빈 값) 그대로 두세요.

### `fanout`을 사용한 자문 주기

기본적으로 자문 모델은 **사용자 턴당 한 번**(`fanout: user_turn`) 실행됩니다. 첫 메시지에서 계획 수준의 조언을 종합한 다음, 작동 애그리게이터가 나머지 도구 루프를 단독으로 처리합니다. 도구 호출 수에 따라 자문 비용이 증가하지 않으므로 가장 저렴한 주기입니다. 비용과 조언의 최신성 사이에서 다른 절충을 제공하는 두 가지 주기도 있습니다:

- `fanout: per_iteration` — 자문 모델이 **모든 도구 반복**마다 다시 실행되므로 최신 도구 결과를 항상 반영하지만, 한 턴의 도구 호출 수만큼 자문 지연 시간과 비용이 증가합니다.
- `fanout: every_n:3` — 중간 지점입니다. 자문 모델은 각 사용자 턴의 **첫 번째** 반복과 이후 **3번째 도구 반복마다** 실행됩니다(어떤 `N >= 2`도 사용할 수 있음). 그 사이의 반복에서는 마지막 자문 실행에서 캐시된 지침을 재사용하므로 애그리게이터는 여전히 모든 단계에서 조언을 받습니다. 다만 매 단계가 아니라 N단계마다 새로 고쳐집니다. 새 사용자 메시지가 들어올 때마다 카운터가 초기화되므로 모든 턴은 최신 조언으로 시작합니다. `fanout: {mode: every_n, n: 3}` 매핑 형식도 허용되며 문자열 형식으로 정규화됩니다.

```yaml
moa:
  presets:
    fresh:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
      aggregator:
        provider: openrouter
        model: openai/gpt-5.5
      fanout: per_iteration   # advisors refresh on every tool iteration
```

알 수 없거나 잘못된 값은 `user_turn`으로 대체됩니다.

:::note 기본값 변경
2026년 7월 이전에는 기본 주기가 `per_iteration`이었습니다. 이제 기본값은 `user_turn`입니다. 모드별 벤치마크에서 더 비싼 기본값이 정당화될 때까지 비용과 영향이 가장 낮은 주기를 사용합니다. 단계별 자문을 다시 사용하려는 프리셋은 `fanout: per_iteration`을 명시적으로 설정하세요.
:::

### 자문 출력의 개인정보 보호 필터

자문 모델의 출력에는 대화의 민감한 데이터(이메일, 형식이 지정된 전화번호, API 키, JWT)가 포함될 수 있으며, 이러한 데이터가 UI에 표시되는 참조 블록, 저장된 MoA 추적 기록, 애그리게이터 프롬프트에 복사될 수 있습니다. 기본적으로 꺼져 있는 `moa.privacy_filter`는 다음 표면에서 해당 데이터를 가립니다:

```yaml
moa:
  privacy_filter: display   # or: full
```

- `display` — **사용자에게 표시되는 표면만** 가립니다. UI에 렌더링되는 라벨이 붙은 참조 블록과 `save_traces`가 기록하는 레코드가 대상입니다. 애그리게이터는 여전히 원본 자문 텍스트를 받아 응답 품질에는 영향을 주지 않습니다.
- `full` — 애그리게이터 프롬프트에 주입되는 자문 텍스트(및 한 번만 실행되는 `/moa` 합성 입력)도 추가로 가립니다.

자격 증명 형식(API 키 접두사, JWT, 비공개 키, DB 연결 문자열)은 Hermes의 중앙 비밀정보 리다이렉터가 마스킹합니다. MoA 필터는 여기에 이메일과 명확한 형식의 전화번호 마스킹을 추가합니다. 패턴은 코드 리뷰 방식의 조언을 위해 의도적으로 보수적으로 적용됩니다. 숫자만 있는 문자열, 줄 번호, 타임스탬프, git SHA, IP 주소는 절대 건드리지 않으며 `(555) 123-4567` 또는 `555-123-4567`처럼 구분된 전화번호 형식만 일치합니다.

### 슬롯별 추론 강도

참조 및 애그리게이터 슬롯에는 `reasoning_effort`도 설정할 수 있습니다. 동일한 모델이 서로 다른 깊이로 기여하도록 하거나 애그리게이터가 자문 참조 모델보다 더 깊이 생각하도록 할 때 사용하세요. 유효한 값은 Hermes의 일반 추론 제어와 일치합니다: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`.

```yaml
moa:
  presets:
    deep_review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: low
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: xhigh
        - provider: xai-oauth
          model: grok-4.5
      aggregator:
        provider: openai-codex
        model: gpt-5.6-sol
        reasoning_effort: high
```

해당 슬롯에서 제공자/Hermes 기본값을 사용하려면 `reasoning_effort`를 생략하세요.

## 터미널 프리셋 관리

```bash
hermes moa list
hermes moa configure              # update the default preset
hermes moa configure review       # create or update a named preset
hermes moa delete review
```

## 벤치마크

HermesBench에서 `gpt-5.5` 참조 모델을 기반으로 `claude-opus-4.8`이 애그리게이트하는 두 모델 MoA 프리셋은 어느 모델을 단독으로 실행한 것보다 높은 점수를 기록합니다:

| 모델 | HermesBench 점수 |
|---|---|
| **Opus 애그리게이터(opus-4.8 + gpt-5.5 참조) — MoA** | **0.8202** |
| `anthropic/claude-opus-4.8` | 0.7607 |
| `openai/gpt-5.5` | 0.7412 |

MoA 구성은 가장 강력한 구성 요소(opus-4.8)보다 약 6점 높은 점수를 기록합니다. 이는 어려운 작업에서 두 모델의 출력을 단순히 평균내는 것이 아니라 두 번째 관점을 집계함으로써 품질이 향상된다는 것을 확인합니다.

## 프롬프트 캐싱

MoA는 **주 대화의 프롬프트 캐시가 절대 깨지지 않도록** 설계되었습니다. MoA 프리셋을 선택하는 것은 일반적인 모델 선택입니다. 대화 중간에 과거 컨텍스트를 변경하거나 도구 세트를 교체하거나 시스템 프롬프트를 다시 생성하지 않습니다. 대화 기록, 시스템 프롬프트, 도구 스키마는 바이트 단위로 안정적으로 유지되므로 다른 모델이 의존하는 모든 턴의 캐시된 접두사는 일반 모델을 사용할 때와 정확히 동일하게 보존됩니다. MoA 프리셋으로 전환하거나 프리셋에서 벗어날 때 드는 비용은 다른 `/model` 전환과 동일한 캐시 무효화 비용이며, 더 크지 않습니다.

두 내부 호출 유형 모두 정상적으로 캐시됩니다:

- **참조 모델**은 대화의 잘라낸 결정적 뷰를 받습니다(시스템 프롬프트와 도구 기록은 위 루프에서 설명한 대로 제거됨). 이 뷰는 안정적인 기록의 안정적인 함수이므로 참조 모델의 프롬프트 접두사는 반복마다 재사용되고 정상적으로 캐시됩니다. 참조 모델은 도구가 없는 짧은 자문 호출입니다.
- **애그리게이터**는 작동 모델입니다. 참조 모델의 출력은 비공개 지침으로 최신 사용자 턴의 *끝*에 추가됩니다. 이 텍스트는 안정적인 접두사 전체(시스템 프롬프트 + 이전 기록) 아래쪽의 끝에 위치하므로 캐시된 접두사를 무효화하지 않습니다. 애그리게이터는 주입된 부분 위의 모든 내용에서 캐시 적중을 얻으며 새로 추가된 끝부분만 새로 처리합니다. 이는 새 사용자 메시지도 캐시되지 않은 끝 토큰으로 처리되는 모든 일반 턴의 동작과 정확히 같습니다.

따라서 MoA는 두 호출 유형 모두에서 프롬프트 캐싱을 희생하지 않습니다. 유일한 실제 비용은 반복마다 추가되는 참조 호출입니다. 캐시가 깨지는 비용이 아니라 여러 모델의 관점을 얻는 비용을 지불하는 것입니다. 나머지 Hermes와 공유하는 장기 대화 접두사는 완전히 유지됩니다.

## 참고 사항

- MoA는 더 이상 `hermes tools`에 나열되지 않으며 활성화할 `moa` 도구 세트도 없습니다.
- 프리셋에서 `enabled: false`를 설정하면 해당 프리셋의 참조 팬아웃이 비활성화됩니다. 애그리게이터가 단독으로 작동하므로 일반 모델로 선택한 것과 정확히 같은 동작입니다. 대시보드와 Desktop 설정에 이 프리셋별 끄기 스위치가 표시됩니다.
- 프리셋의 애그리게이터는 다른 MoA 프리셋이 될 수 없습니다. 재귀적인 MoA 트리는 의도적으로 차단됩니다.
- 참조 모델 하나에서 자격 증명 오류가 발생해도 턴이 중단되지 않습니다. Hermes는 해당 오류를 참조 컨텍스트에 포함하고 응답을 반환한 모델의 결과로 계속 진행합니다.
- MoA는 모델 호출 수를 늘립니다. 단일 모델 반복에서 여러 참조 호출과 애그리게이터 호출이 발생할 수 있습니다.
