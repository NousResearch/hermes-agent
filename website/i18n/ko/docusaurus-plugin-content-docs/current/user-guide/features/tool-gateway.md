---
sidebar_position: 17
title: "도구 게이트웨이"
description: "Nous Portal을 통해 브라우저 자동화,웹 검색, TTS, 이미지 생성을 하나의 제공자 키로 관리합니다."
---

# 도구 게이트웨이 (Tool Gateway)

도구 게이트웨이는 [Nous Portal](https://portal.nousresearch.com/) 구독과 통합되어, 복잡한 설정이나 여러 공급업체 계정을 관리할 필요 없이 에이전트의 강력한 기능(웹 검색, 브라우저 자동화, 음성, 이미지 생성)을 즉시 제공합니다.

`hermes setup --portal`을 실행하면 여러 개의 흩어진 API 키(`BROWSERBASE_API_KEY`, `TAVILY_API_KEY`, `ELEVENLABS_API_KEY`, `FAL_KEY`)를 찾고 비용을 지불하는 대신, 게이트웨이가 이 모든 트래픽을 기존 Nous Portal 구독을 통해 자동으로 라우팅합니다.

## 지원되는 도구

게이트웨이는 도구를 에이전트에게 투명하게 제공합니다. 코드를 변경할 필요 없이 구성만 변경하면 됩니다.

- **[웹 검색 (Web Search)](tools.md#web_search)** — 실시간 인터넷 검색 및 웹 페이지 추출 (이전의 Tavily 또는 Brave Search 백엔드 사용)
- **[브라우저 자동화 (Browser Automation)](browser.md)** — 전체 클라우드 브라우저 인스턴스를 통한 탐색, 클릭, 타이핑 (이전의 Browserbase 사용)
- **[음성 및 TTS (Voice & TTS)](tts.md)** — 고품질 음성 합성 (이전의 ElevenLabs 사용)
- **[이미지 생성 (Image Generation)](image-generation.md)** — 텍스트 기반 시각화 생성 (이전의 FAL.ai 사용)
- **[모델 라우팅 (Model Routing)](provider-routing.md)** — (선택 사항) 도구와 동일한 구독을 사용하여 최상위 모델에 접근

## 활성화

게이트웨이를 켜는 가장 쉬운 방법은 대화형 설정 도구를 사용하는 것입니다:

```bash
hermes setup --portal
```

이 명령은 브라우저를 열어 Nous Portal에서 사용자를 인증한 후, 로컬 구성(`~/.hermes/config.yaml`)을 업데이트하여 지원되는 기능들이 게이트웨이 백엔드를 사용하도록 합니다.

기존에 설정한 환경 변수(예: `TAVILY_API_KEY`)를 수동으로 제거할 필요는 없습니다. `config.yaml`에 명시적으로 설정된 공급자가 환경 변수의 기본값보다 우선합니다. 언제든지 `hermes config set web_search.provider tavily` 등을 실행하여 로컬 키를 사용하는 방식으로 돌아갈 수 있습니다.

## 백그라운드 작동 방식

게이트웨이 활성화 시 `~/.hermes/config.yaml`에 기록되는 구성은 다음과 같습니다:

```yaml
web_search:
  provider: nous_gateway
browser:
  cloud_provider: nous_gateway
tts:
  provider: nous_gateway
image_generation:
  provider: nous_gateway
```

Hermes가 실행될 때마다:
1. 에이전트가 `web_search("latest AI news")`를 호출합니다.
2. Hermes는 `web_search.provider`가 `nous_gateway`로 설정된 것을 확인합니다.
3. 요청은 사용자의 세션 토큰으로 서명되어 Hermes 클라이언트에서 Nous Tool Gateway 엔드포인트(`https://gateway.nousresearch.com/...`)로 안전하게 터널링됩니다.
4. 게이트웨이는 구독 한도를 확인하고, 기본 서비스 제공자(예: Tavily)로 요청을 처리한 후 에이전트에게 동일한 형태의 데이터를 반환합니다.

## 한도 및 크레딧

도구 게이트웨이 사용량은 Nous Portal 계정에 연결됩니다. 각 도구 사용(검색, 브라우저 세션 시작, 이미지 생성)은 해당 도구에 대해 명시된 크레딧 비용만큼 월간 한도에서 차감됩니다.

포털 대시보드(portal dashboard)에서 현재 사용량, 한도 및 구독 계층(tier)을 확인할 수 있습니다. 속도 제한이나 한도 소진에 도달하면 해당 도구 호출은 에이전트에게 명확한 오류 메시지를 반환하여, 에이전트가 (루프에 빠지거나 충돌하는 대신) 사용자에게 할당량을 다 썼다고 알릴 수 있게 합니다.
