---
sidebar_position: 11
title: 모델 카탈로그 (Model Catalog)
description: OpenRouter 및 Nous Portal용 선별된 모델 선택기 목록을 제공하는 원격 호스팅 매니페스트.
---

# 모델 카탈로그 (Model Catalog)

Hermes는 문서 사이트와 함께 호스팅되는 JSON 매니페스트에서 **OpenRouter** 및 **Nous Portal**을 위해 선별된 모델 목록을 가져옵니다. 이를 통해 메인테이너는 새로운 `hermes-agent` 릴리스를 배포하지 않고도 선택기 목록을 업데이트할 수 있습니다.

매니페스트에 접근할 수 없는 경우(오프라인, 네트워크 차단, 호스팅 실패 등), Hermes는 CLI와 함께 제공되는 저장소 내 스냅샷으로 조용히 폴백(fallback)합니다. 매니페스트로 인해 선택기가 중단되는 일은 절대 없으며, 최악의 경우 설치된 버전에 번들된 목록이 표시됩니다.

## 실시간 매니페스트 URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

기존의 `deploy-site.yml` GitHub Pages 파이프라인을 통해 `main`에 병합될 때마다 게시됩니다. 신뢰할 수 있는 소스(source of truth)는 저장소의 `website/static/api/model-catalog.json`에 있습니다.

## 스키마 (Schema)

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "moonshotai/kimi-k2.6", "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k2.6"}
      ]
    }
  }
}
```

필드 참고:

- **`version`** — 정수형 스키마 버전. 향후 스키마는 이 값을 증가시킵니다. Hermes는 이해하지 못하는 버전의 매니페스트를 거부하고 하드코딩된 스냅샷으로 폴백합니다.
- **`metadata`** — 매니페스트, 프로바이더, 모델 수준의 자유 형식(free-form) 딕셔너리입니다. 어떤 키든 사용할 수 있습니다. Hermes는 알 수 없는 필드를 무시하므로 스키마 변경을 조율할 필요 없이 항목에 주석을 달 수 있습니다(`"tier": "paid"`, `"tags": [...]` 등).
- **`description`** — OpenRouter 전용. 선택기 배지 텍스트를 제어합니다(`"recommended"`, `"free"`, 또는 비어 있음). Nous Portal은 이를 사용하지 않습니다. 무료 티어 접근 제어는 Portal의 가격 책정 엔드포인트에서 실시간으로 결정됩니다.
- **가격 및 컨텍스트 길이**는 매니페스트에 포함되지 않습니다. 이 정보는 가져올 때 실시간 프로바이더 API(`/v1/models` 엔드포인트, models.dev)에서 가져옵니다.

## 가져오기 동작 (Fetch behavior)

| 발생 시점 | 발생하는 동작 |
|---|---|
| `/model` 또는 `hermes model` | 디스크 캐시가 오래된 경우 가져오고, 그렇지 않으면 캐시 사용 |
| 디스크 캐시가 최신임 (< TTL) | 네트워크 접근 없음 |
| 네트워크 실패, 캐시 있음 | 캐시로 조용히 폴백, 한 줄 로그 기록 |
| 네트워크 실패, 캐시 없음 | 저장소 내 스냅샷으로 조용히 폴백 |
| 매니페스트 스키마 검증 실패 | 접근할 수 없는 것으로 처리됨 |

캐시 위치: `~/.hermes/cache/model_catalog.json`.

## 설정 (Config)

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 24
  providers: {}
```

항상 저장소 내 스냅샷을 사용하도록 원격 가져오기를 완전히 비활성화하려면 `enabled: false`로 설정하세요.

### 프로바이더별 재정의 URL (Per-provider override URLs)

제3자는 동일한 스키마를 사용하여 자체 큐레이션 목록을 자체 호스팅할 수 있습니다. 프로바이더를 커스텀 URL로 지정하세요:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

재정의된 매니페스트는 대상 프로바이더 블록만 채우면 됩니다. 다른 프로바이더들은 마스터 URL을 계속 참조합니다.

## 매니페스트 업데이트하기

메인테이너:

```bash
# 저장소 내 하드코딩된 목록에서 다시 생성합니다 (hermes_cli/models.py에서
# OPENROUTER_MODELS 또는 _PROVIDER_MODELS["nous"]를 편집한 후
# 매니페스트를 동기화된 상태로 유지합니다).
python scripts/build_model_catalog.py
```

그런 다음 생성된 결과인 `website/static/api/model-catalog.json`에 대한 변경 사항을 `main`에 PR 하세요. 문서 사이트는 병합 시 자동으로 배포되며 새 매니페스트는 몇 분 안에 활성화됩니다.

저장소 내 스냅샷에 포함되어서는 안 되는 세밀한 메타데이터 변경을 위해 JSON을 직접 수동으로 편집할 수도 있습니다. 생성 스크립트는 편의를 위한 것이며, 유일한 신뢰할 수 있는 소스는 아닙니다.
