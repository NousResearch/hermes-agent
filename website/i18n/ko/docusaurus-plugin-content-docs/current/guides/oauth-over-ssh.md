---
sidebar_position: 17
title: "SSH / 원격 호스트를 통한 OAuth"
description: "Hermes가 원격 컴퓨터, 컨테이너 또는 점프 박스 뒤에서 실행될 때 브라우저 기반 OAuth(xAI, Spotify, MCP 서버)를 완료하는 방법"
---

# SSH / 원격 호스트를 통한 OAuth

일부 Hermes 제공자 — **xAI Grok OAuth**, **Spotify**, **원격 MCP 서버**(Linear, Sentry, Atlassian, Asana, Figma 등) — 는 *루프백 리디렉션(loopback redirect)* OAuth 흐름을 사용합니다. 인증 서버는 브라우저를 `http://127.0.0.1:<port>/callback`으로 리디렉션하여 Hermes가 시작한 작은 HTTP 리스너가 인증 코드를 가져올 수 있도록 합니다.

이것은 Hermes와 브라우저가 동일한 컴퓨터에 있을 때 완벽하게 작동합니다. 그러나 이들이 서로 다른 순간 문제가 발생합니다: 랩탑의 브라우저는 **당신의 랩탑**의 `127.0.0.1`에 연결을 시도하지만, 리스너는 **원격 서버**의 `127.0.0.1`에 바인딩되어 있기 때문입니다.

해결책은 한 줄의 SSH 로컬 포워딩입니다 — **또는**, 실제 SSH 클라이언트(GCP Cloud Shell, GitHub Codespaces, EC2 Instance Connect, Gitpod, 브라우저 기반 웹 IDE)가 없는 경우 [#26923](https://github.com/NousResearch/hermes-agent/issues/26923)에 도입된 새로운 `--manual-paste` 플래그를 사용하는 것입니다.

## 요약 (TL;DR)

```bash
# 로컬 컴퓨터(랩탑)의 별도 터미널에서:
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 원격 컴퓨터의 기존 SSH 세션에서:
hermes auth add xai-oauth --no-browser
# → Hermes가 인증 URL을 출력합니다. 랩탑의 브라우저에서 이를 엽니다.
# → 브라우저가 127.0.0.1:56121/callback으로 리디렉션되고, 터널이 요청을
#   원격 리스너로 전달하여 로그인이 완료됩니다.
```

포트 `56121`은 xAI OAuth가 사용하는 포트입니다. Spotify의 경우 이를 `43827`로 변경하세요. Hermes는 `Waiting for callback on ...` 라인에 바인딩된 정확한 포트를 출력하므로 이를 복사하여 사용하세요.

## 브라우저 전용 원격 환경 (Cloud Shell / Codespaces / EC2 Instance Connect)

GCP Cloud Shell, GitHub Codespaces, AWS EC2 Instance Connect, Gitpod 또는 다른 브라우저 기반 콘솔 내에서 Hermes를 실행하여 일반적인 SSH 클라이언트가 없는 경우 위의 SSH 터널을 사용할 수 없습니다. 대신 `--manual-paste`를 사용하세요:

```bash
hermes auth add xai-oauth --manual-paste
# → Hermes가 인증 URL을 출력합니다. 랩탑의 브라우저에서 엽니다.
# → 브라우저에서 승인합니다. 127.0.0.1:56121/callback으로의 리디렉션이 실패하여
#   로드되지 않습니다 — 이는 정상적인 현상입니다.
# → 실패한 페이지의 주소 표시줄에서 전체 URL(FULL URL)을 복사합니다.
# → 터미널의 "Callback URL:" 프롬프트에 붙여넣습니다.
```

동일한 플래그가 통합 모델 선택기를 위한 `hermes model --manual-paste`에서도 작동합니다. Hermes는 전체 URL, 단순히 쿼리 부분인 `?code=...&state=...`, 또는 — 업스트림 동의 페이지가 리디렉션하는 대신 페이지 내에서 인증 코드를 렌더링하는 경우(브라우저 기반 콘솔에서 xAI의 현재 동작) — 단순히 코드 값 자체를 붙여넣는 세 가지 형식을 교환하여 허용합니다.

Hermes는 두 경로 모두에서 **동일한 PKCE 검증기, 상태(state), 및 논스(nonce)**를 사용하므로 업스트림 OAuth 흐름은 바이트 단위로 동일합니다 — `--manual-paste`는 콜백 홉에 대한 순수 전송 방식의 변경일 뿐이며 보안 다운그레이드가 아닙니다.

## 이를 필요로 하는 제공자들

| 제공자 | 루프백 포트 | 터널 필요 여부 |
|----------|---------------|----------------|
| `xai-oauth` (Grok SuperGrok) | `56121` | 예, Hermes가 원격인 경우 |
| Spotify | `43827` | 예, Hermes가 원격인 경우 |
| MCP 서버 (`auth: oauth`) | 서버별로 자동 선택됨 | 예, Hermes가 원격인 경우 |
| `anthropic` (Claude Pro/Max) | 해당 없음 | 아니오 — 코드 붙여넣기 방식 |
| `openai-codex` (ChatGPT Plus/Pro) | 해당 없음 | 아니오 — 디바이스 코드 방식 |
| `minimax`, `nous-portal` | 해당 없음 | 아니오 — 디바이스 코드 방식 |

여러분의 제공자가 위 표에 없다면 터널이 필요하지 않습니다.

## MCP 서버

원격 MCP 서버(Linear, Sentry, Atlassian, Asana, Figma 등)는 동일한 루프백 리디렉션 흐름을 사용합니다. Hermes는 서버당 무료 포트를 자동으로 선택하고 시작 시(새 서버가 `mcp_servers:`에 나타날 때) 또는 `hermes mcp login <server>`를 실행할 때 OAuth 흐름이 시작되면 인증 URL을 출력합니다.

원격 호스트에서 이 프로세스를 완료하는 방법은 두 가지입니다:

**옵션 1 — 리디렉션 URL 붙여넣기 (설정 불필요, 어디서든 작동).** 대화형 터미널에서 Hermes는 로컬 리스너를 실행하는 것과 함께 리디렉션 URL을 붙여넣으라는 프롬프트를 표시합니다. 브라우저에서 승인한 후 `http://127.0.0.1:<port>/callback`으로의 리디렉션은 연결 오류를 표시합니다 — 이는 정상입니다. **브라우저의 주소 표시줄에서 전체 URL**을 복사하여 Hermes 프롬프트에 붙여넣으세요:

```
  MCP OAuth: authorization required.
  Open this URL in your browser:

    https://mcp.linear.app/authorize?response_type=code&...

  Or paste the redirect URL here (or the ?code=...&state=... portion) and press Enter:
> https://mcp.linear.app/callback?code=abc123&state=xyz
  Got authorization code from paste — completing flow.
```

순수한 `?code=...&state=...` 쿼리 문자열도 허용됩니다. 이것은 `auth: oauth`를 사용하는 모든 MCP 서버에서 작동하며 SSH 구성 변경이 필요하지 않습니다.

**옵션 2 — SSH 포트 포워딩 (xAI / Spotify와 동일).** Hermes는 SSH 세션 힌트에 정확히 바인딩된 포트를 출력합니다. 랩탑에서 별도의 터미널을 엽니다:

```bash
ssh -N -L <port>:127.0.0.1:<port> user@remote-host
```

그런 다음 브라우저에서 평소처럼 인증 URL을 엽니다. 리디렉션이 터널을 통과하고 리스너가 이를 수신합니다. 이 방법은 무인 환경에서 흐름을 완료해야 할 때(예: 대화형으로 붙여넣을 수 없는 스크립트 기반 재인증) 사용하세요.

**주의사항 — 30초 구성 재로드 경합.** 실행 중인 Hermes 세션 내에서 `~/.hermes/config.yaml`을 편집하여 OAuth MCP 서버를 추가하면 CLI는 30초 타임아웃과 함께 MCP 연결을 자동으로 재로드합니다. 대화형 OAuth 흐름을 완료하기에는 시간이 충분하지 않으므로 재로드가 중단됩니다. 대신 새 터미널에서 `hermes mcp login <server>`를 사용하세요 — 이 명령어는 제한이 없으며 붙여넣을 때까지 전체 5분을 기다립니다.

## 리스너가 0.0.0.0으로 바인딩할 수 없는 이유

xAI와 Spotify 모두 허용 목록에 대해 `redirect_uri` 매개변수를 검증합니다. 두 서비스 모두 루프백 형식(`http://127.0.0.1:<exact-port>/callback`)을 요구합니다. 리스너를 `0.0.0.0` 또는 다른 포트에 바인딩하면 인증 서버가 redirect_uri 불일치로 요청을 거부합니다. SSH 터널은 처음부터 끝까지 루프백 URI를 온전하게 유지합니다.

## 단계별 가이드: 단일 SSH 홉

### 1. 로컬 컴퓨터에서 터널 시작

```bash
# xAI Grok OAuth (포트 56121)
ssh -N -L 56121:127.0.0.1:56121 user@remote-host

# 또는 Spotify의 경우 (포트 43827)
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N`은 "원격 셸을 열지 않고, 단순히 터널만 유지하라"는 의미입니다. 로그인이 완료될 때까지 이 터널을 계속 열어두세요.

### 2. 별도의 SSH 세션에서 인증 명령어 실행

```bash
ssh user@remote-host
hermes auth add xai-oauth --no-browser
# 또는 Spotify의 경우:
# hermes auth add spotify --no-browser
```

Hermes는 SSH 세션을 감지하여 브라우저 자동 열기를 건너뛰고, 인증 URL과 함께 `Waiting for callback on http://127.0.0.1:<port>/callback` 라인을 출력합니다.

### 3. 로컬 브라우저에서 URL 열기

원격 터미널에서 인증 URL을 복사하여 랩탑의 브라우저에 붙여넣습니다. 동의 화면을 승인합니다. 인증 서버는 `http://127.0.0.1:<port>/callback`으로 리디렉션합니다. 브라우저가 터널에 닿으면 요청이 원격 리스너로 전달되고 Hermes가 `Login successful!`을 출력합니다.

성공 라인을 보게 되면 터널을 중지할 수 있습니다(첫 번째 터널에서 Ctrl+C를 누릅니다).

## 단계별 가이드: 점프 박스를 통한 접속

베스천(bastion) / 점프 호스트를 통해 Hermes에 접속하는 경우, SSH의 내장 `-J` (ProxyJump)를 사용하세요:

```bash
ssh -N -L 56121:127.0.0.1:56121 -J jump-user@jump-host user@final-host
```

이는 점프 박스 자체에 루프백 포트를 두지 않고 점프 호스트를 통과하여 SSH 연결을 묶습니다(chain). 랩탑의 로컬 `127.0.0.1:56121`은 최종 원격 호스트의 `127.0.0.1:56121`로 직접 터널링됩니다.

`-J`를 지원하지 않는 구형 OpenSSH의 경우 긴 형태는 다음과 같습니다:

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 56121:127.0.0.1:56121 \
    user@final-host
```

## Mosh, tmux, ssh ControlMaster

터널은 기본 SSH 연결의 속성입니다. mosh 세션 위에 `tmux` 내부에서 Hermes를 실행하는 경우, mosh 로밍은 `-L` 포워딩을 전달하지 않습니다. `-L` 터널을 위해서만 **별도의** 일반 SSH 세션을 여세요 — 인증 흐름 중에 그 연결이 유지되어야 합니다. 대화형 mosh/tmux 세션은 평소대로 Hermes를 계속 실행할 수 있습니다.

`ssh -o ControlMaster=auto`를 사용하는 경우 다중 연결의 포트 포워딩은 마스터의 수명을 공유합니다. 터널이 생성되지 않으면 마스터를 다시 시작하세요:

```bash
ssh -O exit user@remote-host
ssh -N -L 56121:127.0.0.1:56121 user@remote-host
```

## 문제 해결

### `bind [127.0.0.1]:56121: Address already in use`

랩탑의 다른 프로그램이 이미 해당 포트를 사용하고 있습니다. 이전 터널이 정상적으로 닫히지 않았거나, 로컬 Hermes가 해당 포트에서 수신 대기 중일 수 있습니다. 원인을 찾아 종료하세요:

```bash
# macOS / Linux
lsof -iTCP:56121 -sTCP:LISTEN
kill <PID>
```

그런 다음 `ssh -L` 명령을 다시 시도하세요.

### "Could not establish connection. We couldn't reach your app." (xAI)

xAI의 인증 페이지는 `127.0.0.1:<port>/callback`으로의 리디렉션이 리스너에 도달하지 못할 때 이 메시지를 표시합니다. 터널이 실행되지 않았거나, 포트가 잘못되었거나, Hermes가 이전 실행에서 출력한 포트를 사용하고 있을 수 있습니다(선호하는 포트가 사용 중이면 포트가 자동으로 변경될 수 있으므로 항상 최신 `Waiting for callback on ...` 라인을 확인하세요).

### `xAI authorization timed out waiting for the local callback`

위와 같은 근본적인 원인입니다 — 리디렉션이 돌아오지 않았습니다. 터널이 여전히 활성화되어 있는지 확인하고(`ssh -N`은 출력을 표시하지 않으므로 터널을 시작한 터미널을 확인하세요), 필요한 경우 터널을 다시 시작한 다음 `hermes auth add xai-oauth --no-browser`를 다시 실행하세요.

### 토큰이 잘못된 `~/.hermes`에 저장됨

토큰은 `hermes auth add ...` 명령을 실행한 Linux 사용자 아래에 기록됩니다. 게이트웨이 / systemd 서비스가 다른 사용자(예: `root` 또는 전용 `hermes` 사용자)로 실행되는 경우 토큰이 해당 사용자의 `~/.hermes/auth.json`에 저장되도록 **해당** 사용자로 인증하세요. `sudo -u hermes -i` 또는 이와 동등한 명령을 사용하세요.

## 참고 항목

- [xAI Grok OAuth](./xai-grok-oauth.md)
- [Spotify (`Running over SSH`)](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [기본 MCP 클라이언트 (OAuth 섹션)](../user-guide/features/mcp.md#oauth-authenticated-http-servers)
- [SSH `-J` / ProxyJump (매뉴얼 페이지)](https://man.openbsd.org/ssh#J)
