---
sidebar_position: 5
title: "xAI Grok (OAuth 지원)"
description: "Hermes Agent에서 API 키 또는 X(Twitter) OAuth를 통해 xAI의 Grok 모델에 연결하기"
---

# xAI Grok (OAuth 지원)

Hermes Agent는 이제 API 키나 [X(이전의 Twitter)](https://x.com)를 통한 OAuth 인증을 사용하여 [xAI](https://x.ai)의 **Grok** 모델을 최고 수준으로 지원합니다.

두 가지 연결 방법이 있습니다:

1. **xAI API (`xai` 프로바이더)** — [console.x.ai](https://console.x.ai)의 API 키를 사용하는 전통적인 개발자용입니다. 도구 호출(tool calling)과 비전(vision)을 완벽하게 지원합니다.
2. **X OAuth (`xai-oauth` 프로바이더)** — 유효한 X Premium+ 또는 Grok 구독이 필요합니다. X 계정으로 로그인하여 API 키 없이 대화할 수 있습니다. 이 방법은 Hermes Agent 도구 호출과 아직 호환되지 않습니다.

---

## 방법 1: xAI API (권장)

이 방법은 도구 호출 및 비전 분석을 포함한 Hermes Agent의 전체 기능을 제공합니다. 일반적인 개발자 설정입니다.

### 1단계: API 키 발급
1. [console.x.ai](https://console.x.ai)로 이동합니다.
2. API 키를 생성합니다.
3. 청구(Billing) 탭에서 인퍼런스 요금을 지불하기 위한 크레딧이 있는지 확인하세요.

### 2단계: 환경 변수 설정
`~/.hermes/.env` 파일에 API 키를 추가합니다:

```env
XAI_API_KEY=xai-your-api-key-here
```

*(선택 사항)* Grok은 X(Twitter) 데이터에 접근하여 최신 검색을 수행할 수 있습니다. 이 기능을 활성화하려면 X의 OAuth도 연결해야 합니다(아래의 방법 2 참고). 환경 변수에 `X_OAUTH_TOKEN`이 구성되어 있으면 `xai` 프로바이더는 웹 쿼리에 대해 기본 제공 도구 대신 Grok의 기본 X 검색을 자동으로 활용합니다.

### 3단계: Hermes 구성
CLI 명령어를 사용하거나 `config.yaml`을 직접 편집하세요.

**명령어 사용:**
```bash
hermes model
# 프로바이더 목록에서 "xAI"를 선택합니다
# "grok-beta" 또는 "grok-vision-beta"를 선택합니다
```

**수동 구성 (`~/.hermes/config.yaml`):**
```yaml
model:
  provider: xai
  default: grok-beta
```

---

## 방법 2: X OAuth (구독자용, Tools 미지원)

이 방법을 사용하면 API 사용량에 대한 비용을 지불하지 않고도 X Premium+ 또는 Grok 독립형 구독을 활용하여 Hermes 내에서 Grok과 채팅할 수 있습니다. **현재, 이 방법은 Hermes Agent의 도구 호출 메커니즘을 지원하지 않습니다.** 에이전트 기능을 실행하지 않고 텍스트 기반으로 Grok과 대화하는 경우에만 유용합니다.

### 1단계: OAuth로 로그인
브라우저 기반의 로그인 흐름을 시작하려면 터미널에서 다음을 실행하세요:

```bash
hermes auth add xai-oauth
```

브라우저가 열리면 X 계정에 권한을 부여하도록 요청합니다. 확인이 완료되면 Hermes는 안전한 통신을 위해 `~/.hermes/auth.json`에 새로 고침 토큰을 저장합니다.

*(원격 호스트에서 Hermes를 실행 중이거나 브라우저를 열 수 없는 경우, [SSH/원격 호스트에서의 OAuth 가이드](/guides/oauth-over-ssh)를 참조하여 `--manual-paste` 옵션을 사용하세요.)*

### 2단계: Hermes 구성
OAuth를 통해 인증했다면 `xai-oauth` 프로바이더를 사용하도록 Hermes를 구성해야 합니다.

**명령어 사용:**
```bash
hermes model
# 프로바이더 목록에서 "xAI (OAuth)"를 선택합니다
```

**수동 구성 (`~/.hermes/config.yaml`):**
```yaml
model:
  provider: xai-oauth
  default: grok-latest  # OAuth는 모델 버전이 다르게 지정되는 경우가 많습니다
```

### 참고 사항 및 제한 사항
* **도구 호출 불가:** OAuth 엔드포인트는 복잡한 함수 호출 매개변수를 지원하지 않으므로, Hermes Agent는 이 모드에서 셸 명령어를 실행하거나 파일을 편집할 수 없습니다. 단순한 대화형 인터페이스로 작동합니다.
* **토큰 수명:** 저장된 OAuth 새로 고침 토큰은 X(Twitter)에서 세션을 수동으로 취소하지 않는 한 유효하게 유지됩니다. 만료된 경우 `hermes auth add xai-oauth`를 다시 실행하세요.
* **구독 필요:** OAuth 방법이 작동하려면 연결된 X 계정에 Grok을 이용할 수 있는 등급의 활성 구독이 있어야 합니다. 그렇지 않으면 403 Forbidden 또는 401 Unauthorized 오류가 반환됩니다.
