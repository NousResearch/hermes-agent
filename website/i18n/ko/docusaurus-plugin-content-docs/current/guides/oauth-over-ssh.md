---
sidebar_position: 17
title: "SSH를 통한 OAuth / 원격 호스트"
description: "Hermes가 원격 머신, 컨테이너 또는 점프 박스 뒤에서 실행될 때 브라우저 기반 OAuth(Spotify, MCP 서버)를 완료하는 방법"
---

# SSH를 통한 OAuth / 원격 호스트

일부 Hermes provider — **Spotify**와 **remote MCP servers**(Linear, Sentry, Atlassian, Asana, Figma, …) — 는 *loopback redirect* OAuth 흐름을 사용합니다. 인증 서버는 브라우저를 `http://127.0.0.1:<port>/callback`으로 리디렉션하므로 Hermes가 시작한 작은 HTTP 리스너가 인증 코드를 가져올 수 있습니다.

Hermes와 브라우저가 같은 머신에 있으면 완벽하게 작동합니다. 둘이 다른 순간 문제가 발생합니다. 노트북의 브라우저는 **노트북**의 `127.0.0.1`에 연결하려고 하지만, 리스너는 **원격 서버**의 `127.0.0.1`에 바인딩되어 있기 때문입니다.

해결 방법은 한 줄짜리 SSH 로컬 포워딩입니다. 대화형 터미널의 MCP 서버에서는 대신 리디렉션 URL을 붙여 넣을 수도 있습니다(터널 불필요).

**xAI Grok OAuth (`xai-oauth`)는 loopback callback이 아니라 OAuth device code를 사용합니다** — 출력된 verification URL을 아무 브라우저에서나 열면 Hermes가 승인될 때까지 폴링합니다. SSH 터널은 필요하지 않습니다. [xAI Grok OAuth](./xai-grok-oauth.md)를 참조하세요.

## 요약

```bash
# On your local machine (laptop), in a separate terminal:
ssh -N -L 43827:127.0.0.1:43827 user@remote-host

# In your existing SSH session on the remote machine:
hermes auth spotify --no-browser
# → Hermes prints an authorize URL. Open it in a browser on your laptop.
# → Your browser redirects to 127.0.0.1:43827/callback, the tunnel forwards
#   the request to the remote listener, login completes.
```

Hermes는 `Waiting for callback on ...` 줄에 바인딩한 정확한 포트를 출력합니다 — 여기서 복사하세요. Spotify의 기본 포트는 `43827`입니다.

## 이 기능이 필요한 Provider

| Provider | Loopback 포트 | 터널 필요 여부 |
|----------|---------------|----------------|
| Spotify | `43827` (기본값) | Hermes가 원격에 있을 때 필요 |
| MCP 서버 (`auth: oauth`) | 서버별 자동 선택 | Hermes가 원격에 있을 때 필요(또는 리디렉션 URL 붙여넣기) |
| `xai-oauth` (Grok SuperGrok) | n/a | 아니요 — device code 흐름 |
| `anthropic` (Claude Pro/Max) | n/a | 아니요 — 코드 붙여넣기 흐름 |
| `openai-codex` (ChatGPT Plus/Pro) | n/a | 아니요 — device code 흐름 |
| `minimax`, `nous-portal` | n/a | 아니요 — device code 흐름 |

Provider가 표에 없다면 터널이 필요하지 않습니다.

## MCP 서버

원격 MCP 서버(Linear, Sentry, Atlassian, Asana, Figma 등)는 동일한 loopback redirect 흐름을 사용합니다. Hermes는 서버마다 사용 가능한 포트를 자동으로 선택하고 OAuth 흐름이 시작될 때 인증 URL을 출력합니다 — 시작 시(`mcp_servers:`에 새 서버가 나타날 때) 또는 `hermes mcp login <server>`를 실행할 때입니다.

원격 호스트에서 이를 완료하는 방법은 두 가지입니다.

**옵션 1 — 리디렉션 URL을 붙여넣기(설정 불필요, 어디서나 작동).** 대화형 터미널에서 Hermes는 로컬 리스너를 실행하는 동시에 리디렉션 URL을 붙여넣으라는 메시지를 표시합니다. 브라우저에서 승인한 후 `http://127.0.0.1:<port>/callback`으로 리디렉션되면 연결 오류가 표시됩니다 — 이는 정상입니다. 브라우저 주소 표시줄의 **전체 URL**을 복사해 Hermes 프롬프트에 붙여넣으세요.

```
  MCP OAuth: authorization required.
  Open this URL in your browser:

    https://mcp.linear.app/authorize?response_type=code&...

  Or paste the redirect URL here (or the ?code=...&state=... portion) and press Enter:
> https://mcp.linear.app/callback?code=abc123&state=xyz
  Got authorization code from paste — completing flow.
```

`?code=...&state=...`만 있는 쿼리 문자열도 허용됩니다. `auth: oauth`를 사용하는 모든 MCP 서버에서 작동하며 SSH 설정을 변경할 필요가 없습니다.

**옵션 2 — SSH 포트 포워딩(Spotify와 동일).** Hermes는 SSH 세션 안내에 바인딩한 정확한 포트를 출력합니다. 노트북에서 별도의 터미널을 열고 다음을 실행하세요.

```bash
ssh -N -L <port>:127.0.0.1:<port> user@remote-host
```

그런 다음 평소처럼 인증 URL을 브라우저에서 여세요. 리디렉션이 터널을 통과하고 리스너가 이를 수신합니다. 이 방법은 흐름을 무인으로 완료해야 할 때(예: 대화형으로 붙여넣을 수 없는 스크립트 기반 재인증) 사용하세요.

**주의 사항 — 30s 구성 재로드 경쟁 상태.** 실행 중인 Hermes 세션 내부에서 `~/.hermes/config.yaml`을 편집해 OAuth MCP 서버를 추가하면 CLI가 30초의 제한 시간으로 MCP 연결을 자동 재로드합니다. 이는 대화형 OAuth 흐름을 완료하기에 충분하지 않으며 재로드가 포기하게 됩니다. 대신 새 터미널에서 `hermes mcp login <server>`를 사용하세요 — 이 명령에는 이러한 제한이 없고 붙여넣을 때까지 전체 5분 동안 기다립니다.

## 리스너가 그냥 0.0.0.0에 바인딩할 수 없는 이유

Spotify와 대부분의 MCP OAuth 서버는 `redirect_uri` 매개변수를 허용 목록과 비교합니다. 두 서버 모두 루프백 형식(`http://127.0.0.1:<exact-port>/callback`)을 요구합니다. 리스너를 `0.0.0.0` 또는 다른 포트에 바인딩하면 인증 서버가 redirect_uri 불일치로 요청을 거부합니다. SSH 터널은 처음부터 끝까지 루프백 URI를 그대로 유지합니다.

## 단계별 안내: 단일 SSH 홉

### 1. 로컬 머신에서 터널 시작

```bash
# Spotify (port 43827)
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N`은 "원격 셸을 열지 않고 터널만 계속 유지"한다는 의미입니다. 로그인하는 동안 이 터미널을 계속 실행해 두세요.

### 2. 별도의 SSH 세션에서 인증 명령 실행

```bash
ssh user@remote-host
hermes auth spotify --no-browser
```

Hermes는 SSH 세션을 감지해 브라우저 자동 열기를 건너뛰고, 인증 URL과 `Waiting for callback on http://127.0.0.1:<port>/callback` 줄을 출력합니다.

### 3. 로컬 브라우저에서 URL 열기

원격 터미널에서 인증 URL을 복사해 노트북의 브라우저에 붙여넣으세요. 동의 화면에서 승인합니다. 인증 서버는 `http://127.0.0.1:<port>/callback`으로 리디렉션합니다. 브라우저가 터널에 연결되고 요청이 원격 리스너로 전달되면 Hermes가 `Login successful!`을 출력합니다.

성공 줄이 표시되면 터널을 종료해도 됩니다(첫 번째 터미널에서 Ctrl+C).

## 단계별 안내: 점프 박스를 거치는 경우

Bastion / jump host를 통해 Hermes에 접속한다면 SSH에 내장된 `-J`(ProxyJump)를 사용하세요.

```bash
ssh -N -L 43827:127.0.0.1:43827 -J jump-user@jump-host user@final-host
```

이 명령은 점프 호스트를 통해 SSH 연결을 연결하며, 루프백 포트를 점프 박스 자체에 노출하지 않습니다. 노트북의 로컬 `127.0.0.1:43827`은 최종 원격 호스트의 `127.0.0.1:43827`로 곧바로 터널링됩니다.

`-J`를 지원하지 않는 구형 OpenSSH에서는 긴 형식을 사용하세요.

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 43827:127.0.0.1:43827 \
    user@final-host
```

## Mosh, tmux, ssh ControlMaster

터널은 기반 SSH 연결의 속성입니다. mosh 세션을 통해 `tmux` 안에서 Hermes를 실행하는 경우 mosh의 로밍은 `-L` 포워딩을 전달하지 않습니다. **`-L` 터널 전용으로 별도의 일반 SSH 세션**을 여세요 — 인증 흐름 동안 계속 유지해야 하는 연결이 바로 이것입니다. 대화형 mosh/tmux 세션에서는 Hermes를 평소처럼 계속 실행할 수 있습니다.

`ssh -o ControlMaster=auto`를 사용하는 경우 멀티플렉싱된 연결의 포트 포워딩은 마스터의 수명을 공유합니다. 터널이 시작되지 않으면 마스터를 재시작하세요.

```bash
ssh -O exit user@remote-host
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

## 문제 해결

### `bind [127.0.0.1]:43827: Address already in use`

노트북에서 이미 무언가가 해당 포트를 사용하고 있습니다. 이전 터널이 정상적으로 종료되지 않았거나 로컬 Hermes도 해당 포트에서 수신 중일 수 있습니다. 문제를 일으킨 프로세스를 찾아 종료하세요.

```bash
# macOS / Linux
lsof -iTCP:43827 -sTCP:LISTEN
kill <PID>
```

그런 다음 `ssh -L` 명령을 다시 시도하세요.

### Authorization timed out waiting for the local callback

리디렉션이 원격 리스너로 돌아오지 않았습니다. 터널이 아직 실행 중인지 확인하고(`ssh -N`은 출력을 표시하지 않으므로 시작한 터미널을 확인하세요), 최신 `Waiting for callback on ...` 줄의 포트를 사용했는지 확인하세요(Hermes는 선호 포트가 사용 중이면 포트 번호를 자동으로 올릴 수 있습니다). 필요하면 터널을 재시작하고 인증 명령을 다시 실행하세요.

### Tokens land in the wrong `~/.hermes`

토큰은 `hermes auth add ...`를 실행한 Linux 사용자 아래에 기록됩니다. 게이트웨이 / systemd 서비스가 다른 사용자(예: `root` 또는 전용 `hermes` 사용자)로 실행된다면 **해당 사용자로** 인증해 토큰이 해당 사용자의 `~/.hermes/auth.json`에 저장되도록 하세요. `sudo -u hermes -i` 또는 이에 상응하는 명령을 사용합니다.

## 관련 문서

- [xAI Grok OAuth](./xai-grok-oauth.md) — 디바이스 코드 방식; SSH 터널 없음
- [Spotify (`Running over SSH`)](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [Native MCP client (OAuth 섹션)](../user-guide/features/mcp.md#oauth-authenticated-http-servers)
- [SSH `-J` / ProxyJump (매뉴얼 페이지)](https://man.openbsd.org/ssh#J)
