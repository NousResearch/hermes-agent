---
sidebar_position: 11
title: 모델 카탈로그
description: OpenRouter와 Nous Portal의 엄선된 모델 선택기 목록을 구동하는 원격 호스팅 매니페스트입니다.
---

# 모델 카탈로그

Hermes는 문서 사이트와 함께 호스팅되는 JSON 매니페스트에서 **OpenRouter** 및 **Nous Portal**용 엄선된 모델 목록을 가져옵니다. 이를 통해 유지 관리자는 새 `hermes-agent` 릴리스를 배포하지 않고도 선택기 목록을 업데이트할 수 있습니다.

매니페스트에 연결할 수 없는 경우(오프라인, 네트워크 차단, 호스팅 장애) Hermes는 CLI에 포함된 저장소 내 스냅샷으로 조용히 대체합니다. 매니페스트가 선택기를 중단시키는 일은 없습니다. 최악의 경우 설치된 버전에 번들된 목록이 표시됩니다.

## 실시간 매니페스트 URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

기존 `deploy-site.yml` GitHub Pages 파이프라인을 통해 `main`에 병합될 때마다 게시됩니다. 단일 진실 공급원은 저장소의 `website/static/api/model-catalog.json`입니다.

## 스키마

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2",         "description": "default", "default": true},
        {"id": "moonshotai/kimi-k3",   "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2", "default": true},
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k3"}
      ]
    }
  }
}
```

필드 설명:

- **`version`** — 정수형 스키마 버전입니다. 향후 스키마가 변경되면 이 값이 증가합니다. Hermes는 이해할 수 없는 버전의 매니페스트를 거부하고 하드코딩된 스냅샷으로 대체합니다.
- **`metadata`** — 매니페스트, 공급자, 모델 수준의 자유 형식 dict입니다. 어떤 키든 사용할 수 있습니다. Hermes는 알 수 없는 필드를 무시하므로 스키마 변경을 조율하지 않고도 항목에(`"tier": "paid"`, `"tags": [...]` 등) 주석을 달 수 있습니다.
- **`description`** — OpenRouter 전용입니다. 선택기 배지 텍스트(`"recommended"`, `"free"`, `"default"` 또는 빈 값)를 결정합니다. Nous Portal은 이를 사용하지 않으며, 무료 티어 게이팅은 Portal의 가격 엔드포인트에서 실시간으로 결정됩니다.
- **`default`** — 공급자마다 정확히 하나의 항목에만 `"default": true`가 있을 수 있습니다. 해당 모델이 **무음 기본값**입니다. 즉 사용자가 모델을 선택하지 않았을 때 Hermes가 사용하는 모델입니다(GUI 온보딩 확인 카드, `model` 없이 구성된 `provider`, 빈 `model.default`). 런타임에서는 캐시만 읽으므로(`get_default_model_from_cache`) 빠른 확인 경로가 네트워크에 접근하지 않습니다. 캐시된 매니페스트가 없으면 Hermes는 저장소 내 `PREFERRED_SILENT_DEFAULT_MODEL` 상수로 대체하며, 이 상수는 레이블이 지정된 항목과 일치해야 합니다. 이를 통해 유지 관리자는 릴리스를 배포하지 않고도 무음 기본값을 교체할 수 있습니다. 의도적으로 성능이 좋고 비용이 낮은 모델이며, 가장 비싼 플래그십 모델은 사용하지 않습니다.
- **가격과 컨텍스트 길이**는 매니페스트에 없습니다. 가져올 때 실시간 공급자 API(`/v1/models` 엔드포인트, models.dev)에서 가져옵니다.

## 가져오기 동작

| 시점 | 동작 |
|---|---|
| `/model` 또는 `hermes model` | 디스크 캐시가 오래되었으면 가져오고, 그렇지 않으면 캐시 사용 |
| 디스크 캐시가 최신 상태(< TTL) | 네트워크 요청 없음 |
| 캐시가 있는 상태에서 네트워크 실패 | 캐시로 조용히 대체하고 로그 한 줄 기록 |
| 캐시가 없는 상태에서 네트워크 실패 | 저장소 내 스냅샷으로 조용히 대체 |
| 매니페스트 스키마 검증 실패 | 연결할 수 없는 것으로 처리 |

캐시 위치: `~/.hermes/cache/model_catalog.json`.

## 구성

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 1
  providers: {}
```

원격 가져오기를 완전히 비활성화하고 항상 저장소 내 스냅샷을 사용하려면 `enabled: false`로 설정합니다.

### 공급자별 재정의 URL

제3자는 동일한 스키마를 사용해 자체 큐레이션 목록을 호스팅할 수 있습니다. 공급자를 사용자 지정 URL로 지정합니다.

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

재정의 매니페스트에는 해당 공급자 블록만 채우면 됩니다. 다른 공급자는 계속 마스터 URL을 기준으로 확인됩니다.

### 선택기에서 공급자 숨기기

`excluded_providers`를 사용하면 유효한 자격 증명이 있더라도 `/model` 선택기에서 특정 공급자를 숨길 수 있습니다. 일반적으로 사용하지 않아야 하는 레거시 또는 테스트 공급자의 자격 증명이 있는 경우 유용합니다(예: `auth.json`에 여전히 캐시되어 있거나 `gh` CLI를 통해 검색된 예전 Copilot 또는 OpenRouter 토큰).

제외 항목은 공급자가 표시될 수 있는 모든 키와 대소문자를 구분하지 않고 비교됩니다. 여기에는 Hermes id와 models.dev id(내장 매핑 공급자), overlay pid와 확인된 Hermes slug(overlay 공급자), canonical slug(canonical 공급자)가 포함됩니다. 따라서 `copilot` 같은 단일 항목으로 어느 섹션에서 공급자를 내보내든 숨길 수 있습니다. 모든 `/model` 선택기 화면에서 적용됩니다: 게이트웨이 대화형/텍스트 선택기, TUI 선택기, 대화형 `hermes model` CLI 선택기입니다. 빈 목록(또는 키 생략)은 아무런 영향을 주지 않습니다.

```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai
```

## 매니페스트 업데이트

유지 관리자는 다음을 실행합니다.

```bash
# Re-generate from the in-repo hardcoded lists (keeps manifest in sync after
# editing OPENROUTER_MODELS or _PROVIDER_MODELS["nous"] in hermes_cli/models.py).
python scripts/build_model_catalog.py
```

그런 다음 결과 변경 사항을 `website/static/api/model-catalog.json`에 반영해 `main`으로 PR을 보냅니다. 문서 사이트는 병합 시 자동으로 배포되며 새 매니페스트는 몇 분 안에 적용됩니다.

저장소 내 스냅샷에 속하지 않는 세밀한 메타데이터 변경은 JSON을 직접 수정할 수도 있습니다. 생성기 스크립트는 편의 기능일 뿐, 유일한 진실 공급원은 아닙니다.
