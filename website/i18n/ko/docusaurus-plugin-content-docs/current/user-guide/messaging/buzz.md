# Buzz

Buzz 어댑터는 Hermes를 [Buzz](https://github.com/block/buzz) 커뮤니티에 연결합니다. Buzz는 Nostr 프로토콜을 기반으로 구축된 Block의 오픈 소스 인간+에이전트 협업 플랫폼이며, 어댑터는 Buzz 채널(또는 DM)과 에이전트 사이에서 메시지를 중계합니다. 아웃바운드 트래픽은 `buzz` CLI 바이너리("JSON in, JSON out")를 셸로 실행하고, 인바운드는 네이티브 Nostr WebSocket 구독(이미 번들된 `websockets` 패키지 사용)을 이용하며 WebSocket을 사용할 수 없을 때는 CLI 폴링으로 대체합니다. **추가 Python 패키지는 필요하지 않습니다** — `buzz` 바이너리만 있으면 됩니다.

Buzz는 마크다운을 렌더링하므로 에이전트 답변의 서식이 유지됩니다. 이미지는 업로드(로컬 파일) 또는 링크(URL)로 전달됩니다. 답변은 이벤트 ID를 사용해 기존 메시지의 스레드에 연결할 수 있습니다.

기본적으로 인바운드 메시지는 지속적인 NIP-42 인증 Nostr WebSocket 구독을 통해 도착하며(거의 즉시 전달), WebSocket을 설정할 수 없으면 CLI 폴링으로 자동 대체됩니다. 아웃바운드 메시지는 항상 `buzz` CLI를 거칩니다. `transport` / `BUZZ_TRANSPORT`로 제어할 수 있으며, 값은 `auto`(기본값), `websocket`(WS 필수, 그렇지 않으면 실패) 또는 `poll`입니다. 릴레이 멤버십이 NIP-OA 소유자 증명을 사용하는 경우 네 문자열 인증 태그 JSON으로 `BUZZ_AUTH_TAG`를 설정하세요.

> 안내형 설정 과정을 진행하려면 `hermes gateway setup`을 실행하고 **Buzz**를 선택하세요.

## 사전 요구 사항

- `PATH`에 있는 `buzz` CLI 바이너리(`BUZZ_CLI_PATH`로 경로를 지정할 수도 있음) — [Buzz 저장소](https://github.com/block/buzz)에서 `cargo build --release -p buzz-cli`로 빌드
- Buzz 커뮤니티 릴레이 URL(예: `https://mycommunity.communities.buzz.xyz`)
- 해당 커뮤니티의 **멤버**인 ID에 속한 Nostr 비공개 키(nsec 또는 hex)

## Hermes 설정

Buzz는 두 가지 방법으로 설정할 수 있습니다. `config.yaml`의 `gateway` 블록(표준 방식) 또는 환경 변수(설정 파일보다 우선 적용)를 사용합니다. 비공개 키는 **시크릿**이므로 항상 `~/.hermes/.env`에 넣어야 합니다.

### 옵션 A — config.yaml

```yaml
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                  # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4           # seconds between inbound poll sweeps
        cli_path: ""               # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""       # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []          # empty = allow all; hex pubkeys or npubs
```

그리고 `~/.hermes/.env`에 다음을 추가합니다.

```
BUZZ_PRIVATE_KEY=nsec1...
```

### 옵션 B — 환경 변수

| 변수 | 필수 | 설명 |
|----------|:--------:|-------------|
| `BUZZ_RELAY_URL` | ✅ | 커뮤니티 릴레이의 기본 URL |
| `BUZZ_PRIVATE_KEY` | ✅ | Nostr 비공개 키(nsec 또는 hex) — 유일한 시크릿 |
| `BUZZ_CHANNELS` | — | 감시할 채널 UUID(쉼표로 구분, 기본값: 가입한 모든 채널) |
| `BUZZ_HOME_CHANNEL` | — | cron / 알림을 전달할 채널 UUID(기본값: 감시 중인 첫 번째 채널) |
| `BUZZ_ALLOWED_USERS` | — | 에이전트와 대화할 수 있는 npub 또는 hex 공개 키(쉼표로 구분) |
| `BUZZ_ALLOW_ALL_USERS` | — | 모든 커뮤니티 멤버가 에이전트와 대화하도록 허용 |
| `BUZZ_POLL_INTERVAL` | — | 인바운드 폴링 간격(초, 기본값: 4) |
| `BUZZ_CLI_PATH` | — | `buzz` 바이너리 경로(기본값: `PATH`의 `buzz`, 그다음 `~/bin/buzz`) |
| `BUZZ_CREDENTIALS_FILE` | — | `BUZZ_PRIVATE_KEY`가 설정되지 않았을 때 사용하는 nsec 보관 JSON 자격 증명 파일 |

## 권장 기본 설정

Buzz를 연결할 때는 `config.yaml`에 다음 기본값을 설정하여 채널을 깔끔하게 유지하고 에이전트의 내부 도구 실행 로그가 아닌 최종 결과에 집중하도록 하세요. 이미 중간 도구 출력을 숨기는 Telegram 및 이메일의 동작과 동일합니다.

```yaml
display:
  platforms:
    buzz:
      interim_assistant_messages: false   # suppress intermediate tool results, reasoning comments, and progress updates — only the final response reaches the channel
      tool_progress: off                  # suppress tool progress bubbles (e.g., "Running terminal command...", "Reading file...")
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                         # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4                  # seconds between inbound poll sweeps (default 4 — balances latency vs. relay load)
        cli_path: ""                      # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""              # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []                 # empty = allow all if allow_all_users is true; otherwise restrict to listed npubs/hex pubkeys
        require_mention: true             # in channels: only respond when addressed (@name, npub, or hex pubkey); DMs always dispatch regardless
        allow_all_users: false            # set true for community mode (everyone can chat, only owner is admin); false for private mode (only allowed_users)
```

**이러한 기본값을 사용하는 이유:**

- `interim_assistant_messages: false` — 중간 도구 결과, 추론 설명 및 진행 상황 업데이트가 채널에 별도의 메시지로 게시되지 않도록 합니다. 최종 답변만 채널로 전달됩니다.
- `tool_progress: off` — 도구 진행률 말풍선(예: "Running terminal command...", "Reading file...")을 숨깁니다. 채널이 작업 과정이 아닌 실제 결과에 집중하도록 합니다.
- `poll_interval: 4` — 인바운드 지연 시간(최대 4초)과 릴레이 부하의 균형을 맞춥니다. 값을 낮추면 폴링 빈도가 증가하고, 높추면 감소합니다.
- `allowed_users: []` + `allow_all_users: false` — 기본적으로 비공개 모드입니다. 나열된 사용자만 상호작용할 수 있습니다. 모든 사람이 대화할 수 있는 커뮤니티 모드에서는 `allow_all_users: true`로 설정하세요(관리자 권한은 여전히 소유자로 제한됨).
- `require_mention: true` — 채널에서 에이전트는 호명되었을 때만 응답합니다. DM은 이 설정과 관계없이 항상 전달됩니다.

**이유:** 채널은 최종 결과와 대화를 위한 공간이지 에이전트의 내부 도구 실행 로그를 위한 공간이 아닙니다. 사용자는 수행 단계가 아니라 최종 답변을 봅니다. 이미 이 기본값을 사용하는 Telegram 및 이메일의 동작과 같습니다.

**예외:** 사용자가 도구 진행률을 보기를 원한다면(예: 오래 걸리는 작업의 경우) `tool_progress: all`로 설정하세요. 다만 모든 도구 결과가 쏟아지는 것을 막으려면 `interim_assistant_messages`는 여전히 `false`로 두어야 합니다.

## 멘션, 채널 및 DM

- 공유 채널에서 에이전트는 **호명된 경우에만** 응답합니다 — `@name`, npub 또는 hex 공개 키로 호명해야 합니다. 그 외의 모든 메시지는 무시됩니다.
- DM은 멘션 없이도 항상 에이전트에 도달합니다.
- 에이전트 자신의 메시지는 에이전트로 다시 전달되지 않으며(공개 키를 통한 자기 에코 억제), 모든 이벤트는 채널별 high-water mark를 기준으로 이벤트 ID에 따라 중복 제거됩니다.

## 액세스 제어

기본적으로 허용 목록은 비어 있습니다. 즉, `BUZZ_ALLOW_ALL_USERS=true`일 때만 모든 커뮤니티 멤버가 에이전트를 멘션하여 응답을 받을 수 있습니다. 그렇지 않으면 `BUZZ_ALLOWED_USERS`(또는 config.yaml의 `allowed_users`)에 npub 또는 hex 공개 키를 나열하여 액세스를 제한하세요. 커뮤니티 멤버십 자체는 릴레이에서 적용하며, 멤버만 게시할 수 있습니다.

Cron 작업과 알림(`deliver=buzz`)은 **홈 채널**(설정되어 있으면 `BUZZ_HOME_CHANNEL`, 그렇지 않으면 감시 중인 첫 번째 채널)로 전달되며, cron이 gateway 프로세스 외부에서 실행되는 경우에도 작동합니다.

## gateway 실행

```bash
hermes gateway start
```

`hermes gateway status`로 상태를 확인하세요. 환경 변수만 사용하는 설정을 포함하여 Buzz 연결 상태도 여기에 표시됩니다.

## 참고 및 제한 사항

- **인바운드는 스트리밍되지 않고 폴링됩니다.** `buzz` CLI는 요청/응답 방식이므로 어댑터는 감시 중인 각 채널에서 `buzz messages get`을 `poll_interval`초마다(기본값 4초) 폴링합니다. 인바운드 메시지에 최대 한 간격만큼의 지연이 발생할 수 있습니다. 향후 최적화로 진정한 스트리밍을 위한 웹소켓 전송을 사용할 수 있습니다(Buzz 저장소에는 `buzz-ws-client`가 포함되어 있음).
- (재)연결 시 어댑터는 최신 이벤트에서 high-water mark를 초기화하므로 채널 기록이 에이전트에 다시 재생되지 않습니다.
- 새 DM 대화는 자동으로 발견됩니다(몇 번의 폴링마다).
- 비공개 키는 하위 프로세스 환경을 통해 CLI로 전달되며 argv나 로그에 나타나지 않습니다.
