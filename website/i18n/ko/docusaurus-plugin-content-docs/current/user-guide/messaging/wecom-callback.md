---
sidebar_position: 15
---

# WeCom Callback (자체 구축 앱)

콜백/웹훅 모델을 사용하여 자체 구축 기업 애플리케이션으로 Hermes를 WeCom(기업용 WeChat)에 연결합니다.

:::info WeCom 봇 vs WeCom 콜백
Hermes는 두 가지 WeCom 통합 모드를 지원합니다:
- **[WeCom Bot](wecom.md)** — 봇 스타일, WebSocket을 통해 연결됩니다. 설정이 더 간단하고 그룹 채팅에서 작동합니다.
- **WeCom Callback** (현재 페이지) — 자체 구축 앱, 암호화된 XML 콜백을 수신합니다. 사용자의 WeCom 사이드바에 독립적인 앱으로 표시됩니다. 다중 기업(multi-corp) 라우팅을 지원합니다.
:::

봇 스타일 통합에 대해서는 [WeCom Bot](./wecom.md) 문서도 참고하세요.

> `hermes gateway setup`을 실행하고 안내에 따른 설정을 위해 **WeCom Callback**을 선택하세요.

## 작동 방식

1. WeCom 관리자 콘솔에서 자체 구축 애플리케이션을 등록합니다.
2. WeCom은 암호화된 XML을 여러분의 HTTP 콜백 엔드포인트로 푸시합니다.
3. Hermes는 메시지를 복호화하고 에이전트를 위해 대기열에 넣습니다.
4. 즉시 확인 응답을 보냅니다 (사용자에게는 아무것도 표시되지 않음).
5. 에이전트가 요청을 처리합니다 (일반적으로 3~30분 소요).
6. 답변은 WeCom `message/send` API를 통해 주도적으로 전송됩니다.

## 전제 조건

- 관리자 권한이 있는 WeCom 기업 계정
- `aiohttp` 및 `httpx` Python 패키지 (기본 설치에 포함됨)
- 콜백 URL을 위한 공개적으로 접근 가능한 서버 (또는 ngrok과 같은 터널)

## 설정

### 1. WeCom에서 자체 구축 앱 생성

1. [WeCom Admin Console](https://work.weixin.qq.com/) → **Applications** → **Create App**으로 이동
2. **Corp ID** 기록 (관리자 콘솔 상단에 표시됨)
3. 앱 설정에서 **Corp Secret** 생성
4. 앱의 개요 페이지에서 **Agent ID** 기록
5. **Receive Messages** 아래에 콜백 URL 구성:
   - URL: `http://YOUR_PUBLIC_IP:8645/wecom/callback`
   - Token: 임의의 토큰 생성 (WeCom에서 제공)
   - EncodingAESKey: 키 생성 (WeCom에서 제공)

### 2. 환경 변수 구성

`.env` 파일에 추가하세요:

```bash
WECOM_CALLBACK_CORP_ID=your-corp-id
WECOM_CALLBACK_CORP_SECRET=your-corp-secret
WECOM_CALLBACK_AGENT_ID=1000002
WECOM_CALLBACK_TOKEN=your-callback-token
WECOM_CALLBACK_ENCODING_AES_KEY=your-43-char-aes-key

# 선택 사항
WECOM_CALLBACK_HOST=0.0.0.0
WECOM_CALLBACK_PORT=8645
WECOM_CALLBACK_ALLOWED_USERS=user1,user2
```

### 3. 게이트웨이 시작

```bash
hermes gateway
```

(`hermes gateway start`는 `hermes gateway install`을 통해 systemd/launchd 서비스를 등록한 후에만 사용하세요.)

콜백 어댑터는 구성된 포트에서 HTTP 서버를 시작합니다. WeCom은 GET 요청을 통해 콜백 URL을 확인한 다음 POST를 통해 메시지를 보내기 시작합니다.

## 구성 참조

`config.yaml`의 `platforms.wecom_callback.extra` 아래에 이들을 설정하거나 환경 변수를 사용하세요:

| 설정 | 기본값 | 설명 |
|---------|---------|-------------|
| `corp_id` | — | WeCom 기업 Corp ID (필수) |
| `corp_secret` | — | 자체 구축 앱의 Corp secret (필수) |
| `agent_id` | — | 자체 구축 앱의 Agent ID (필수) |
| `token` | — | 콜백 검증 토큰 (필수) |
| `encoding_aes_key` | — | 콜백 암호화를 위한 43자 AES 키 (필수) |
| `host` | `0.0.0.0` | HTTP 콜백 서버를 위한 바인드 주소 |
| `port` | `8645` | HTTP 콜백 서버를 위한 포트 |
| `path` | `/wecom/callback` | 콜백 엔드포인트의 URL 경로 |

## 다중 앱 라우팅

여러 자체 구축 앱을 실행하는 기업의 경우(예: 여러 부서나 자회사에 걸쳐), `config.yaml`에 `apps` 목록을 구성하세요:

```yaml
platforms:
  wecom_callback:
    enabled: true
    extra:
      host: "0.0.0.0"
      port: 8645
      apps:
        - name: "dept-a"
          corp_id: "ww_corp_a"
          corp_secret: "secret-a"
          agent_id: "1000002"
          token: "token-a"
          encoding_aes_key: "key-a-43-chars..."
        - name: "dept-b"
          corp_id: "ww_corp_b"
          corp_secret: "secret-b"
          agent_id: "1000003"
          token: "token-b"
          encoding_aes_key: "key-b-43-chars..."
```

크로스-기업 충돌을 방지하기 위해 사용자는 `corp_id:user_id` 단위로 범위가 지정됩니다. 사용자가 메시지를 보낼 때, 어댑터는 사용자가 속한 앱(기업)을 기록하고 올바른 앱의 액세스 토큰을 통해 응답을 라우팅합니다.

## 접근 제어

앱과 상호작용할 수 있는 사용자를 제한하세요:

```bash
# 특정 사용자 허용 목록 지정
WECOM_CALLBACK_ALLOWED_USERS=zhangsan,lisi,wangwu

# 또는 모든 사용자 허용
WECOM_CALLBACK_ALLOW_ALL_USERS=true
```

## 엔드포인트

어댑터는 다음을 노출합니다:

| 메서드 | 경로 | 목적 |
|--------|------|---------|
| GET | `/wecom/callback` | URL 검증 핸드셰이크 (설정 시 WeCom이 전송함) |
| POST | `/wecom/callback` | 암호화된 메시지 콜백 (WeCom이 사용자 메시지를 여기로 전송함) |
| GET | `/health` | 헬스 체크 — `{"status": "ok"}` 반환 |

## 암호화

모든 콜백 페이로드는 EncodingAESKey를 사용하는 AES-CBC로 암호화됩니다. 어댑터는 다음을 처리합니다:

- **인바운드**: XML 페이로드 복호화, SHA1 서명 검증
- **아웃바운드**: 능동적 API를 통해 응답 전송 (암호화된 콜백 응답 아님)

이 암호화 구현은 Tencent의 공식 WXBizMsgCrypt SDK와 호환됩니다.

## 제한 사항

- **스트리밍 없음** — 에이전트 처리가 완료된 후 완전한 메시지 형태로 답변이 도착합니다.
- **타이핑 표시 없음** — 콜백 모델은 타이핑 상태를 지원하지 않습니다.
- **텍스트 전용** — 현재 입력으로 텍스트 메시지만 지원합니다; 이미지/파일/음성 입력은 아직 구현되지 않았습니다. 에이전트는 WeCom 플랫폼 힌트를 통해 아웃바운드 미디어 기능(이미지, 문서, 비디오, 음성)을 알고 있습니다.
- **응답 지연** — 에이전트 세션은 보통 3~30분이 소요됩니다; 사용자는 처리가 완료되어야 답글을 봅니다.

## 문제 해결

**서명 검증 실패.**
WeCom은 관리자 콘솔에 등록한 **Token**으로 모든 요청에 서명합니다. Hermes에 구성된 토큰과 관리자 콘솔이 기대하는 토큰이 불일치하는 것이 가장 흔한 원인입니다. 관리자 콘솔에서 **Token**과 **EncodingAESKey**를 모두 다시 복사하세요 — 쉽게 잘릴 수 있습니다. `~/.hermes/.env`의 `=` 주변 공백은 서명 확인을 망가뜨릴 수 있습니다. 수정 후 `hermes gateway run`을 재시작하세요.

**콜백 URL에 연결할 수 없거나 / 검증 단계 실패.**
WeCom은 등록된 공개 URL로 요청을 보냅니다. 다음을 확인하세요:
1. 역방향 프록시 / 터널이 `/wecom/callback`을 게이트웨이의 포트로 포워딩하는지 확인하세요.
2. 관리자 콘솔의 URL이 HTTPS인지 확인하세요 (WeCom은 일반 HTTP를 거부합니다).
3. 외부 네트워크에서 `curl -i https://<your-domain>/wecom/callback`이 타임아웃 이외의 값을 반환하는지 확인하세요 (쿼리 파라미터가 없는 4xx 응답도 괜찮습니다 — 이는 리스너에 접근 가능하다는 의미입니다).

**포트에 연결할 수 없거나 / 리스너가 바인딩되지 않음.**
바인딩된 호스트/포트는 `hermes gateway run` 로그를 확인하세요. 어댑터가 `127.0.0.1`에 바인딩된 경우, 반드시 역방향 프록시나 터널을 앞에 두어야 합니다 — WeCom의 서버는 로컬 호스트(loopback)에 접근할 수 없습니다. `config.yaml`에 `extra.host: 0.0.0.0`을 설정하거나(직접 노출 시 `allowed_source_cidrs` 추가), 루프백을 유지하고 Cloudflare Tunnel / nginx와 같은 터널을 사용하세요.
