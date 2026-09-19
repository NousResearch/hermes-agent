---
sidebar_position: 15
---

# WeCom 콜백 (자체 구축 앱)

콜백/웹훅 모델을 사용해 Hermes를 WeCom(기업용 WeChat)의 자체 구축 기업 애플리케이션으로 연결합니다.

:::info WeCom 봇과 WeCom 콜백
Hermes는 두 가지 WeCom 통합 모드를 지원합니다.
- **[WeCom 봇](wecom.md)** — WebSocket을 통해 연결되는 봇 방식입니다. 설정이 더 간단하고 그룹 채팅에서 작동합니다.
- **WeCom 콜백**(이 페이지) — 암호화된 XML 콜백을 수신하는 자체 구축 앱입니다. 사용자의 WeCom 사이드바에 완전한 앱으로 표시됩니다. 여러 기업 라우팅을 지원합니다.
:::

참고: 봇 방식 통합은 [WeCom 봇](./wecom.md)을 참조하세요.

> 안내에 따라 설정하려면 `hermes gateway setup`을 실행하고 **WeCom 콜백**을 선택하세요.

## 작동 방식

1. WeCom 관리자 콘솔에서 자체 구축 애플리케이션을 등록합니다.
2. WeCom이 암호화된 XML을 HTTP 콜백 엔드포인트로 푸시합니다.
3. Hermes가 메시지를 복호화하고 에이전트 대기열에 추가합니다.
4. 즉시 응답합니다(무음 — 사용자에게 아무것도 표시되지 않음).
5. 에이전트가 요청을 처리합니다(일반적으로 3~30분).
6. WeCom `message/send` API를 통해 답변을 능동적으로 전달합니다.

## 사전 요구 사항

- 관리자 액세스 권한이 있는 WeCom 기업 계정
- `aiohttp` 및 `httpx` Python 패키지(기본 설치에 포함)
- 콜백 URL에 공개적으로 접근할 수 있는 서버(또는 ngrok 같은 터널)

## 설정

### 1. WeCom에서 자체 구축 앱 만들기

1. [WeCom 관리자 콘솔](https://work.weixin.qq.com/) → **애플리케이션** → **앱 만들기**로 이동합니다.
2. **기업 ID(Corp ID)**를 기록합니다(관리자 콘솔 상단에 표시됨).
3. 앱 설정에서 **기업 시크릿(Corp Secret)**을 생성합니다.
4. 앱 개요 페이지에서 **Agent ID**를 기록합니다.
5. **메시지 수신**에서 콜백 URL을 구성합니다.
   - URL: `http://YOUR_PUBLIC_IP:8645/wecom/callback`
   - Token: 무작위 토큰을 생성합니다(WeCom에서 하나를 제공합니다).
   - EncodingAESKey: 키를 생성합니다(WeCom에서 하나를 제공합니다).

### 2. 환경 변수 구성

`.env` 파일에 추가합니다.

```bash
WECOM_CALLBACK_CORP_ID=your-corp-id
WECOM_CALLBACK_CORP_SECRET=your-corp-secret
WECOM_CALLBACK_AGENT_ID=1000002
WECOM_CALLBACK_TOKEN=your-callback-token
WECOM_CALLBACK_ENCODING_AES_KEY=your-43-char-aes-key

# Optional
# WECOM_CALLBACK_HOST=  # optional pin; unset = dual-stack (all interfaces, IPv4+IPv6)
WECOM_CALLBACK_PORT=8645
WECOM_CALLBACK_ALLOWED_USERS=user1,user2
```

### 3. 게이트웨이 시작

```bash
hermes gateway
```

(`hermes gateway install`을 실행해 systemd/launchd 서비스를 등록한 후에만 `hermes gateway start`를 사용하세요.)

콜백 어댑터가 구성된 포트에서 HTTP 서버를 시작합니다. WeCom은 GET 요청으로 콜백 URL을 확인한 후 POST를 통해 메시지를 보내기 시작합니다.

## 구성 참조

`config.yaml`의 `platforms.wecom_callback.extra` 아래에 설정하거나 환경 변수를 사용하세요.

| 설정 | 기본값 | 설명 |
|---------|---------|-------------|
| `corp_id` | — | WeCom 기업의 Corp ID(필수) |
| `corp_secret` | — | 자체 구축 앱의 기업 시크릿(필수) |
| `agent_id` | — | 자체 구축 앱의 Agent ID(필수) |
| `token` | — | 콜백 확인 토큰(필수) |
| `encoding_aes_key` | — | 콜백 암호화에 사용하는 43자 AES 키(필수) |
| `host` | 설정되지 않음(듀얼 스택: 모든 인터페이스, IPv4+IPv6) | HTTP 콜백 서버의 바인드 주소 |
| `port` | `8645` | HTTP 콜백 서버의 포트 |
| `path` | `/wecom/callback` | 콜백 엔드포인트의 URL 경로 |

## 여러 앱 라우팅

여러 자체 구축 앱을 운영하는 기업(예: 여러 부서 또는 자회사)의 경우 `config.yaml`에서 `apps` 목록을 구성합니다.

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

기업 간 충돌을 방지하기 위해 사용자를 `corp_id:user_id`로 구분합니다. 사용자가 메시지를 보내면 어댑터가 사용자가 속한 앱(기업)을 기록하고 해당 앱의 액세스 토큰을 통해 답변을 라우팅합니다.

## 액세스 제어

앱과 상호작용할 수 있는 사용자를 제한합니다.

```bash
# Allowlist specific users
WECOM_CALLBACK_ALLOWED_USERS=zhangsan,lisi,wangwu

# Or allow all users
WECOM_CALLBACK_ALLOW_ALL_USERS=true
```

## 엔드포인트

어댑터가 다음 엔드포인트를 노출합니다.

| 메서드 | 경로 | 용도 |
|--------|------|---------|
| GET | `/wecom/callback` | URL 확인 핸드셰이크(WeCom이 설정 중 전송) |
| POST | `/wecom/callback` | 암호화된 메시지 콜백(WeCom이 사용자의 메시지를 이곳으로 전송) |
| GET | `/health` | 상태 확인 — `{"status": "ok"}` 반환 |

## 암호화

모든 콜백 페이로드는 EncodingAESKey를 사용해 AES-CBC로 암호화됩니다. 어댑터가 다음을 처리합니다.

- **인바운드**: XML 페이로드 복호화, SHA1 서명 확인
- **아웃바운드**: 능동적 API를 통해 답변 전송(암호화된 콜백 응답 아님)

암호화 구현은 Tencent의 공식 WXBizMsgCrypt SDK와 호환됩니다.

## 제한 사항

- **스트리밍 없음** — 에이전트가 작업을 완료한 후 답변이 완성된 메시지로 도착합니다.
- **입력 중 표시 없음** — 콜백 모델은 입력 중 상태를 지원하지 않습니다.
- **텍스트만 지원** — 현재 입력은 텍스트 메시지만 지원하며 이미지/파일/음성 입력은 아직 구현되지 않았습니다. WeCom 플랫폼 힌트를 통해 에이전트는 아웃바운드 미디어 기능(이미지, 문서, 동영상, 음성)을 인식합니다.
- **응답 지연** — 에이전트 세션은 3~30분이 걸리며 처리가 완료되면 사용자가 답변을 확인할 수 있습니다.

## 문제 해결

**서명 확인에 실패합니다.**
WeCom은 관리자 콘솔에 등록한 **Token**으로 모든 요청에 서명합니다.
Hermes에 구성된 토큰과 관리자 콘솔이 기대하는 토큰의 불일치가 가장 일반적인 원인입니다. 관리자 콘솔에서 **Token**과
**EncodingAESKey**를 모두 다시 복사하세요. 쉽게 잘릴 수 있습니다.
`~/.hermes/.env` 값에서 `=` 주변의 공백도 서명 확인을 방해합니다. 수정한 후 `hermes gateway run`을 다시 시작하세요.

**콜백 URL에 연결할 수 없거나 확인 단계가 실패합니다.**
WeCom은 등록한 공개 URL에 접속합니다. 다음을 확인하세요.
1. 리버스 프록시/터널이 `/wecom/callback`을 게이트웨이 포트로 전달하는지 확인합니다.
2. 관리자 콘솔의 URL이 HTTPS인지 확인합니다(WeCom은 일반 HTTP를 거부합니다).
3. 네트워크 외부에서 `curl -i https://<your-domain>/wecom/callback`을 실행했을 때
   시간 초과가 아닌 응답이 반환되는지 확인합니다(쿼리 매개변수가 없을 때 4xx여도 괜찮습니다 — 리스너에 연결할 수 있다는 의미입니다).

**포트에 연결할 수 없거나 리스너가 바인딩되지 않습니다.**
바인딩된 호스트/포트는 `hermes gateway run` 로그에서 확인하세요. 어댑터가
`127.0.0.1`에 바인딩되었다면 리버스 프록시 또는 터널 앞에 배치해야 합니다 — WeCom 서버는 루프백에 연결할 수 없습니다. `extra.host`를 설정하지 않아 기본 듀얼 스택
바인딩(모든 인터페이스, IPv4+IPv6)을 적용하거나, `config.yaml`에서 인터페이스를 지정하세요(직접 노출하는 경우
`allowed_source_cidrs`도 함께 설정). 또는 루프백을 유지하고 Cloudflare Tunnel/nginx 같은 터널을 사용하세요.
