---
title: "Actual 설정 — Hermes에서 Actual Computer(actual.inc) 추론 설정"
sidebar_label: "Actual 설정"
description: "Hermes에서 Actual Computer(actual.inc) 추론 설정"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Actual 설정

Hermes에서 Actual Computer(actual.inc) 추론을 설정합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Optional — `hermes skills install official/devops/actual-setup`으로 설치 |
| 경로 | `optional-skills/devops/actual-setup` |
| 버전 | `2.0.0` |
| 작성자 | shl0ms + Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `actual`, `actual-inc`, `provider`, `local-inference`, `relay`, `gguf`, `setup` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침이기도 합니다.
:::

# Actual Computer 설정 스킬

[actual.inc](https://actual.inc)(Actual Computer)를 Hermes 추론 프로바이더로 설정합니다. Actual은 사용자의 하드웨어를 비공개 추론 클러스터로 전환하고, 두 가지 방식으로 OpenAI 호환 API를 제공합니다. 하나는 `ac_` 키로 인증하는 호스팅 종단 간 암호화 릴레이 `https://api.actual.inc`이고, 다른 하나는 로컬 온디바이스 데몬 `http://127.0.0.1:8080`(루프백에서는 인증 없음)입니다. 이 스킬은 사용자를 대신해 Actual 데몬을 설치하지 않습니다. 기기 인증에는 브라우저를 사용하는 사람이 필요합니다.

## 사용 시점

- 사용자가 actual.inc를 추론 프로바이더로 추가하려는 경우
- 사용자에게 `ac_` 키가 있고 Hermes를 Actual 클러스터를 통해 라우팅하려는 경우
- 사용자가 완전한 로컬 온디바이스 추론을 위해 Actual 데몬을 사용하려는 경우
- 문제 해결: Actual 요청이 이해하기 어려운 400 오류나 빈 스트림으로 실패하는 경우

## 사전 요구 사항

- Hermes는 **최초 제공되는 `actual` 프로바이더 지원을 갖추고 있습니다**(프로바이더 ID `actual`, 별칭 `actual-computer`, `actualcomputer`, `aci`). 현재 Hermes에서 Actual을 `custom_providers` / `providers.actual.*` 항목으로 설정하지 마세요. 내장 프로바이더가 이 이름을 소유하고 base URL 정규화, Responses 전송 방식, 로컬 무인증을 자동으로 처리합니다.
- 릴레이 모드: Actual 계정과 https://actual.inc/user/keys에서 발급한 `ac_` 추론 키
- 로컬 모드: 사용자가 데몬을 설치하고(`curl -fsSL "https://actual.inc/install" | bash`) `actual`을 한 번 실행한 뒤 출력된 `https://actual.inc/device?code=...` URL을 브라우저에서 열어 기기 인증을 완료해야 합니다. 그 URL을 사용자에게 전달하고 기다리세요. 이메일을 지어내거나 사용자를 대신해 인증하지 마세요. 코드는 5분 후 만료되므로 `actual`을 다시 실행해 새 코드를 받으세요.

## 실행 방법

### 릴레이 / API 모드

1. 키를 `.env`에 넣습니다(비밀 정보만 — `config.yaml`에는 넣지 않음):
   `~/.hermes/.env`에 `ACTUAL_API_KEY=ac_...`를 추가합니다.
2. `terminal`로 키를 확인하고 모델을 검색합니다.
   ```bash
   curl -s https://api.actual.inc/v1/models -H "Authorization: Bearer $ACTUAL_API_KEY"
   ```
3. 프로바이더와 모델을 선택합니다.
   ```bash
   hermes config set model.provider actual
   hermes config set model.default "MODEL_ID_FROM_DISCOVERY"
   ```
4. 종단 간으로 확인합니다.
   ```bash
   hermes chat -Q -q "Reply with exactly: ACTUAL_OK" --provider actual -m MODEL_ID
   ```

### 로컬 모드

1. 사람이 데몬을 설치하고 인증을 완료합니다(사전 요구 사항 참조).
2. 모델을 다운로드하고 로드합니다(한 번 인증하면 스크립트로 실행 가능).
   ```bash
   actual models search "qwen2.5 0.5b instruct gguf" --limit 8 --no-prompt
   # Downloads REQUIRE an explicit quantization (409 ambiguous_model_download otherwise):
   actual models download "Qwen/Qwen2.5-0.5B-Instruct-GGUF/Q4_K_M"
   actual models list        # note the INSTALLED name (differs from download id)
   actual models load "qwen2.5-0.5b-instruct-q4_k_m"   # load by installed name
   ```
3. Hermes가 데몬을 사용하도록 지정합니다. 루프백 호스트가 포함된 `ACTUAL_BASE_URL`은 내장 프로바이더를 자동으로 로컬 무인증 모드로 전환하므로 키가 필요하지 않습니다. `~/.hermes/.env`에 `ACTUAL_BASE_URL=http://127.0.0.1:8080`을 추가한 다음 다음을 실행합니다.
   ```bash
   hermes config set model.provider actual
   hermes config set model.default "INSTALLED_MODEL_NAME"
   ```
4. 확인합니다(도구 세트 축소 — 아래 컨텍스트 창 함정 참조).
   ```bash
   hermes chat -Q -q "Reply with exactly: LOCAL_OK" --provider actual -m INSTALLED_NAME -t file,web
   ```

## 빠른 참조

| 항목 | 값 |
|---|---|
| 호스팅 릴레이 | `https://api.actual.inc/v1` (일반 호스트에서 자동 정규화) |
| 로컬 데몬 | `http://127.0.0.1:8080/v1` (루프백에서 무인증) |
| 키 환경 변수 | `ACTUAL_API_KEY` (`ac_...`) |
| 기본 URL 환경 변수 | `ACTUAL_BASE_URL` (루프백 호스트 ⇒ 로컬 무인증 모드) |
| 프로바이더 ID / 별칭 | `actual` / `actual-computer`, `actualcomputer`, `aci` |
| 전송 방식 | Responses API(`codex_responses`) — 내장 방식이며 재정의하지 않음 |
| 클러스터 고정 | config.yaml의 `providers.actual.extra_headers`를 통한 `X-Cluster-ID` 헤더 |
| 모델 크기 안내 | 0.5B Q4_K_M 약 470MB(장난감), 7-8B Q4_K_M 약 4.5GB(일상용), 32B 약 20GB |

## 함정

1. **reasoning_effort 함정(최초 제공 프로바이더이므로 Hermes가 처리).** Actual의 SGLang/vLLM 백엔드는 `none/low/medium/high/max`만 허용합니다. `xhigh`/`ultra`는 이해하기 어려운 `Expecting value: line 1 column 1 (char 0)` 오류(래핑된 HTTP 400)를 일으켰습니다. 내장 프로바이더는 전송 시 `xhigh→high`, `ultra→max`로 제한합니다. 이전 Hermes에서 이런 방식으로 계속 400이 발생하면 config.yaml에서 모델별 제한을 `agent.reasoning_overrides.<model>: high`로 설정하세요.
2. **소형 로컬 모델의 컨텍스트 창 오버플로.** Hermes의 기본 도구 세트는 약 26k 토큰의 스키마와 약 9k 토큰의 시스템 프롬프트로 구성됩니다. 32k 컨텍스트로 로드한 모델은 첫 턴 전에 오버플로되고 llama.cpp 계열 서버는 `data: [DONE]`만 출력합니다. Hermes는 이를 `Provider returned an empty stream with no finish_reason`으로 보고합니다. 이는 SSE 버그가 **아닙니다**. 해결 방법은 도구를 제한하거나(`-t file,web`), 더 큰 `n_ctx`로 모델을 로드하거나, 전체 도구 세트에 대해 컨텍스트가 64k 이상인 모델을 선택하는 것입니다. 업스트림 추적 이슈: #51448(새 이슈를 만들지 말고 해당 이슈에 증거를 추가). 관련 있지만 별개의 이슈: #65631(400을 담은 HTTP-200 SSE), #56516(추론 전용 스트림).
3. **다운로드 ID와 설치된 이름.** `actual models download`는 `repo/QUANT`를 받고 명시적인 양자화가 없으면 409를 반환합니다. `actual models load`는 `actual models list`의 설치된 이름을 받습니다.
4. **빈 콘텐츠를 반환하는 추론 모델.** GLM/Qwen 추론 변형은 별도의 `reasoning` 필드에 사고 과정을 출력하며, 작은 `max_tokens`를 추론에 전부 사용할 수 있습니다. 실패라고 판단하기 전에 `max_tokens`를 넉넉하게 지정하세요.
5. **`actual`이라는 사용자 지정 프로바이더를 만들지 마세요.** 이전의 최초 지원 전 설정 가이드는 `providers.actual.*` 설정 블록을 작성했습니다. 현재 Hermes에서는 내장 프로바이더가 해당 이름을 차지하므로 오래된 사용자 지정 블록은 무시되거나 충돌합니다. 해당 블록을 삭제하고 위의 환경 변수 + model.provider 흐름을 사용하세요.

## 확인

```bash
# Relay:
hermes chat -Q -q "Reply with exactly: ACTUAL_OK" --provider actual -m MODEL
# Local (small model — reduced toolset):
hermes chat -Q -q "Reply with exactly: LOCAL_OK" --provider actual -m MODEL -t file,web
# Provider status (local no-auth shows key_source=local-offline):
hermes status
```

다른 OpenAI 호환 클라이언트(예: OpenCode)는 `references/opencode.md`를 참조하세요.
