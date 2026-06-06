---
title: "Spotify"
description: "Hermes Agent로 Spotify 재생, 플레이리스트, 라이브러리를 제어하세요."
sidebar_position: 9
---

# Spotify

Hermes는 PKCE OAuth가 포함된 공식 Spotify Web API를 사용하여 재생, 대기열, 검색, 플레이리스트, 저장된 트랙/앨범 및 청취 기록 등 Spotify를 직접 제어할 수 있습니다. 토큰은 `~/.hermes/auth.json`에 저장되며 401 오류 발생 시 자동으로 갱신됩니다. 기기당 한 번만 로그인하면 됩니다.

Hermes에 내장된 다른 OAuth 연동(Google, GitHub Copilot, Codex)과 달리 Spotify는 모든 사용자가 자신의 가벼운 개발자 앱을 직접 등록해야 합니다. Spotify는 타사에서 누구나 사용할 수 있는 공용 OAuth 앱을 출시하는 것을 허용하지 않기 때문입니다. 이 과정은 약 2분이 소요되며 `hermes auth spotify` 명령어를 통해 안내받을 수 있습니다.

## 사전 요구 사항 (Prerequisites)

- Spotify 계정. **무료** 계정은 검색, 플레이리스트, 라이브러리 및 활동 도구에 사용할 수 있습니다. 재생 제어(재생, 일시 정지, 건너뛰기, 탐색, 볼륨, 대기열 추가, 전송)를 위해서는 **Premium** 계정이 필요합니다.
- 설치 및 실행 중인 Hermes Agent.
- 재생 도구의 경우: **활성화된 Spotify Connect 기기** — Web API가 제어할 대상이 있도록 최소 하나 이상의 기기(휴대폰, 데스크톱, 웹 플레이어, 스피커)에 Spotify 앱이 열려 있어야 합니다. 활성화된 기기가 없으면 "no active device" 메시지와 함께 `403 Forbidden` 오류가 발생합니다. 아무 기기에서나 Spotify를 열고 다시 시도하세요.

## 설정 (Setup)

### 한 번에 해결하기 (One-shot): `hermes tools` 또는 최초 실행 설정

가장 빠른 방법입니다. 다음을 실행하세요:

```bash
hermes tools
```

`🎵 Spotify`로 스크롤한 다음, 스페이스바를 눌러 켜고 `s`를 눌러 저장합니다. 처음 실행 시 `hermes setup` / `hermes setup tools` 흐름 중에도 동일한 토글을 사용할 수 있습니다. Spotify는 선택적 기능이므로 해당 단계에서 활성화하면 `hermes tools`와 동일한 제공자 기반(provider-aware) 구성이 실행됩니다.

Hermes는 즉시 OAuth 흐름으로 들어갑니다. 아직 Spotify 앱을 등록하지 않았다면, 그 자리에서 바로 앱을 생성할 수 있도록 안내합니다. 완료하면 한 번의 패스로 도구 세트가 활성화되고 인증됩니다.

단계를 분리해서 진행하거나 나중에 다시 인증하려면 아래의 2단계 흐름을 사용하세요.

### 2단계 흐름 (Two-step flow)

#### 1. 도구 세트 활성화

```bash
hermes tools
```

`🎵 Spotify`를 켜고 저장한 후 인라인 마법사가 열리면 무시하고 닫습니다 (Ctrl+C). 도구 세트는 켜진 상태로 유지되며 인증 단계만 연기됩니다.

#### 2. 로그인 마법사 실행

```bash
hermes auth spotify
```

7개의 Spotify 도구는 1단계를 거친 후에만 에이전트의 도구 세트에 나타납니다. 원하지 않는 사용자가 모든 API 호출에 불필요한 도구 스키마를 포함하지 않도록 기본적으로 꺼져 있습니다.

`HERMES_SPOTIFY_CLIENT_ID`가 설정되어 있지 않으면 Hermes가 앱 등록 과정을 안내합니다:

1. 브라우저에서 `https://developer.spotify.com/dashboard`를 엽니다.
2. Spotify의 "Create app" 양식에 붙여넣을 정확한 값을 출력합니다.
3. 발급받은 Client ID를 입력하라는 프롬프트를 표시합니다.
4. 나중에 이 단계를 건너뛸 수 있도록 이를 `~/.hermes/.env`에 저장합니다.
5. OAuth 동의 흐름으로 바로 이어집니다.

승인 후 토큰은 `~/.hermes/auth.json`의 `providers.spotify` 아래에 저장됩니다. 활성 추론 제공자(inference provider)는 변경되지 **않습니다** — Spotify 인증은 LLM 제공자와 독립적입니다.

### Spotify 앱 생성하기 (마법사가 요구하는 것)

대시보드가 열리면 **Create app**을 클릭하고 다음을 입력합니다:

| 필드 (Field) | 값 (Value) |
|-------|-------|
| App name | 아무거나 (예: `hermes-agent`) |
| App description | 아무거나 (예: `personal Hermes integration`) |
| Website | 비워둠 |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | **Web API** 체크 |

약관에 동의하고 **Save**를 클릭합니다. 다음 페이지에서 **Settings**를 클릭하고 → **Client ID**를 복사하여 Hermes 프롬프트에 붙여넣습니다. Hermes가 필요로 하는 값은 그것뿐입니다. PKCE는 클라이언트 시크릿을 사용하지 않습니다.

### SSH 원격 접속 / 헤드리스 환경에서 실행하기

`SSH_CLIENT` 또는 `SSH_TTY`가 설정되어 있으면 Hermes는 마법사 및 OAuth 단계 중에 브라우저를 자동으로 열지 않습니다. Hermes가 출력하는 대시보드 URL과 권한 부여 URL을 복사하여 로컬 머신의 브라우저에서 열고 정상적으로 진행하세요. 로컬 HTTP 리스너는 원격 호스트의 `43827` 포트에서 여전히 실행됩니다. SSH 로컬 포트 포워딩 없이는 노트북의 브라우저가 원격 루프백에 연결할 수 없습니다:

```bash
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

점프 박스(jump-box) / 배스천(bastion) 설정 및 기타 문제점(mosh, tmux, 포트 충돌)에 대해서는 [SSH 원격 접속 시 OAuth 인증 (OAuth over SSH / Remote Hosts)](../../guides/oauth-over-ssh.md)를 참조하세요.

## 확인 (Verify)

```bash
hermes auth status spotify
```

토큰이 있는지, 그리고 액세스 토큰이 언제 만료되는지 표시합니다. 갱신은 자동으로 이루어집니다. 어떤 Spotify API 호출이라도 401을 반환하면 클라이언트는 새로 고침 토큰(refresh token)을 교환하고 한 번 재시도합니다. 새로 고침 토큰은 Hermes를 재시작해도 유지되므로, Spotify 계정 설정에서 앱 권한을 취소하거나 `hermes auth logout spotify`를 실행하지 않는 한 다시 인증할 필요가 없습니다.

## 사용법 (Using it)

로그인하면 에이전트는 7개의 Spotify 도구에 접근할 수 있습니다. 에이전트와 자연스럽게 대화하면 에이전트가 올바른 도구와 액션을 선택합니다. 최상의 동작을 위해 에이전트는 올바른 사용 패턴(단일 검색 후 재생, `get_state`를 미리 실행하지 않아야 할 때 등)을 가르치는 컴패니언 스킬을 로드합니다.

```
> 마일스 데이비스 노래 좀 틀어줘
> 지금 무슨 노래 듣고 있어?
> 이 트랙을 내 'Late Night Jazz' 플레이리스트에 추가해
> 다음 곡으로 넘겨
> "Focus 2026"이라는 새 플레이리스트를 만들고 방금 들은 세 곡을 추가해
> 내 저장된 앨범 중에 라디오헤드(Radiohead) 앨범이 뭐야?
> Blackbird 어쿠스틱 커버 검색해 줘
> 부엌 스피커로 재생 넘겨
```

### 도구 참조 (Tool reference)

재생 상태를 변경하는 모든 액션은 특정 기기를 타겟팅하기 위해 선택적 `device_id`를 허용합니다. 생략하면 Spotify는 현재 활성 기기를 사용합니다.

#### `spotify_playback`
재생 제어 및 검사, 최근 들은 트랙 기록 가져오기.

| 액션 (Action) | 목적 (Purpose) | Premium 필수? |
|--------|---------|----------|
| `get_state` | 전체 재생 상태 (트랙, 기기, 진행률, 셔플/반복) | 아니요 |
| `get_currently_playing` | 현재 트랙만 (204에서는 빈 값 반환 — 아래 참조) | 아니요 |
| `play` | 재생 시작/재개. 선택 사항: `context_uri`, `uris`, `offset`, `position_ms` | 예 |
| `pause` | 재생 일시 정지 | 예 |
| `next` / `previous` | 트랙 건너뛰기 | 예 |
| `seek` | `position_ms`로 점프 | 예 |
| `set_repeat` | `state` = `track` / `context` / `off` | 예 |
| `set_shuffle` | `state` = `true` / `false` | 예 |
| `set_volume` | `volume_percent` = 0-100 | 예 |
| `recently_played` | 최근 들은 트랙. 선택 사항: `limit`, `before`, `after` (Unix ms) | 아니요 |

#### `spotify_devices`
| 액션 (Action) | 목적 (Purpose) |
|--------|---------|
| `list` | 계정에 보이는 모든 Spotify Connect 기기 목록 |
| `transfer` | 재생을 `device_id`로 이동. 선택적 `play: true`는 전송 시 재생 시작 |

### Home Assistant 관리 스피커

Home Assistant가 이미 Spotify Connect를 지원하는 스피커(예: Sonos, Echo, Nest 또는 기타 Connect 지원 스피커)를 관리하는 경우, Spotify가 해당 스피커를 볼 수 있으면 자동으로 `spotify_devices list`에 나타납니다. Hermes는 이를 위해 Home Assistant ↔ Spotify 브리지가 필요하지 않습니다. Spotify가 기기 라우팅을 기본적으로 처리합니다.

Hermes에게 스피커의 표시 이름으로 재생 전송을 요청하거나 (예: "Spotify를 부엌 스피커로 전송해 줘"), 스크립팅할 때 `spotify_devices list`를 호출하여 정확한 `device_id`를 `spotify_devices transfer`에 전달하세요. 스피커가 누락된 경우 Spotify 앱이나 스피커의 Spotify 통합 환경을 한 번 열어서 Spotify가 이를 활성 Connect 대상으로 등록하도록 하세요.

#### `spotify_queue`
| 액션 (Action) | 목적 (Purpose) | Premium 필수? |
|--------|---------|----------|
| `get` | 현재 대기열에 있는 트랙 | 아니요 |
| `add` | 큐에 `uri` 추가 | 예 |

#### `spotify_search`
카탈로그를 검색합니다. `query`는 필수입니다. 선택 사항: `types` (배열: `track` / `album` / `artist` / `playlist` / `show` / `episode`), `limit`, `offset`, `market`.

#### `spotify_playlists`
| 액션 (Action) | 목적 (Purpose) | 필수 인수 (Required args) |
|--------|---------|---------------|
| `list` | 사용자의 플레이리스트 | — |
| `get` | 하나의 플레이리스트 + 트랙들 | `playlist_id` |
| `create` | 새 플레이리스트 생성 | `name` (+ 선택 사항: `description`, `public`, `collaborative`) |
| `add_items` | 트랙 추가 | `playlist_id`, `uris` (선택 사항: `position`) |
| `remove_items` | 트랙 제거 | `playlist_id`, `uris` (+ 선택 사항: `snapshot_id`) |
| `update_details` | 이름 변경 / 편집 | `playlist_id` + `name`, `description`, `public`, `collaborative` 중 하나 |

#### `spotify_albums`
| 액션 (Action) | 목적 (Purpose) | 필수 인수 (Required args) |
|--------|---------|---------------|
| `get` | 앨범 메타데이터 | `album_id` |
| `tracks` | 앨범 트랙 목록 | `album_id` |

#### `spotify_library`
저장된 트랙 및 앨범에 대한 통합 접근. `kind` 인수를 사용하여 컬렉션을 선택합니다.

| 액션 (Action) | 목적 (Purpose) |
|--------|---------|
| `list` | 페이지가 매겨진 라이브러리 목록 |
| `save` | 라이브러리에 `ids` / `uris` 추가 |
| `remove` | 라이브러리에서 `ids` / `uris` 제거 |

필수: `kind` = `tracks` 또는 `albums`, 그리고 `action`.

### 기능 비교: Free vs Premium

읽기 전용 도구는 Free 계정에서 작동합니다. 재생이나 대기열을 조작하는 모든 작업에는 Premium이 필요합니다.

| Free 계정에서 동작 | Premium 필요 |
|---------------|------------------|
| `spotify_search` (모두) | `spotify_playback` — play, pause, next, previous, seek, set_repeat, set_shuffle, set_volume |
| `spotify_playback` — get_state, get_currently_playing, recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists` (모두) | |
| `spotify_albums` (모두) | |
| `spotify_library` (모두) | |

## 스케줄링: Spotify + Cron

Spotify 도구는 일반 Hermes 도구이므로 Hermes 세션에서 실행되는 Cron 작업은 원하는 스케줄에 따라 재생을 트리거할 수 있습니다. 새로운 코드가 필요하지 않습니다.

### 아침 기상 플레이리스트

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "재생을 내 주방 스피커로 옮기고 'Morning Commute' 플레이리스트를 시작해. 볼륨은 40으로. 셔플 켜."
```

평일 아침 7시에 일어나는 일:
1. Cron이 헤드리스 Hermes 세션을 시작합니다.
2. 에이전트가 프롬프트를 읽고, 이름으로 "주방 스피커"를 찾기 위해 `spotify_devices list`를 호출한 다음, `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play` 순서로 진행합니다.
3. 대상 스피커에서 음악이 시작됩니다. 총비용: 한 번의 세션, 몇 번의 도구 호출, 사람의 입력 없음.

### 취침 시간

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Spotify 일시 정지해. 그리고 내일 다시 시작할 때 조용하게 볼륨을 20으로 설정해."
```

### 주의 사항 (Gotchas)

- **Cron이 실행될 때 활성화된 기기가 존재해야 합니다.** Spotify 클라이언트(휴대폰/데스크톱/Connect 스피커)가 실행 중이지 않으면 재생 작업에서 `403 no active device`를 반환합니다. 아침 플레이리스트의 경우 휴대폰보다는 항상 켜져 있는 기기(Sonos, Echo, 스마트 스피커)를 타겟팅하는 것이 좋습니다.
- **재생을 조작하는 모든 작업에는 Premium이 필요합니다** — 재생, 일시 정지, 건너뛰기, 볼륨, 전송. 읽기 전용 cron 작업(예약된 "내 최근 재생 트랙 이메일로 보내줘")은 Free 계정에서도 잘 작동합니다.
- **Cron 에이전트는 활성화된 도구 세트를 상속합니다.** Cron 세션이 Spotify 도구를 보려면 `hermes tools`에서 Spotify가 활성화되어 있어야 합니다.
- **Cron 작업은 `skip_memory=True`로 실행되므로** 메모리 저장소에 기록하지 않습니다.

전체 Cron 참조: [Cron 작업 (Cron Jobs)](./cron).

## 로그아웃 (Sign out)

```bash
hermes auth logout spotify
```

`~/.hermes/auth.json`에서 토큰을 제거합니다. 앱 구성도 함께 지우려면 `~/.hermes/.env`에서 `HERMES_SPOTIFY_CLIENT_ID` (그리고 설정했다면 `HERMES_SPOTIFY_REDIRECT_URI`)를 삭제하거나 마법사를 다시 실행하세요.

Spotify 쪽에서 앱 권한을 취소하려면 [계정에 연결된 앱 (Apps connected to your account)](https://www.spotify.com/account/apps/)을 방문하여 **REMOVE ACCESS**를 클릭하세요.

## 문제 해결 (Troubleshooting)

**`403 Forbidden — Player command failed: No active device found`** — 최소 한 대의 기기에서 Spotify가 실행되고 있어야 합니다. 휴대폰, 데스크톱 또는 웹 플레이어에서 Spotify 앱을 열고 아무 트랙이나 1초 정도 재생하여 등록한 후 다시 시도하세요. `spotify_devices list`는 현재 표시되는 기기를 보여줍니다.

**`403 Forbidden — Premium required`** — 재생 조작 액션을 사용하려는 Free 계정 사용자입니다. 위의 기능 비교표를 참조하세요.

**`get_currently_playing`에서 `204 No Content` 발생** — 현재 어떤 기기에서도 아무것도 재생되고 있지 않습니다. 이것은 Spotify의 정상적인 응답이며 오류가 아닙니다. Hermes는 이를 설명 가능한 빈 결과(`is_playing: false`)로 표출합니다.

**`INVALID_CLIENT: Invalid redirect URI`** — Spotify 앱 설정의 리디렉션 URI가 Hermes가 사용하는 것과 일치하지 않습니다. 기본값은 `http://127.0.0.1:43827/spotify/callback`입니다. 이를 앱의 허용된 리디렉션 URI에 추가하거나, 등록한 URI에 맞게 `~/.hermes/.env`에 `HERMES_SPOTIFY_REDIRECT_URI`를 설정하세요.

**`429 Too Many Requests`** — Spotify의 속도 제한입니다. Hermes가 친절한 오류를 반환합니다. 1분 정도 기다렸다가 다시 시도하세요. 만약 지속된다면 스크립트에서 짧은 주기로 반복 호출을 하고 있을 가능성이 높습니다. Spotify의 할당량은 대략 30초마다 재설정됩니다.

**`401 Unauthorized`가 계속 나타납니다** — 새로 고침 토큰이 취소되었습니다 (보통 계정에서 앱을 제거했거나 앱이 삭제된 경우). `hermes auth spotify`를 다시 실행하세요.

**마법사에서 브라우저가 열리지 않습니다** — SSH를 통한 접속이거나 디스플레이가 없는 컨테이너 내부인 경우, Hermes가 이를 감지하고 자동 열기를 건너뜁니다. 콘솔에 출력된 대시보드 URL을 복사하여 수동으로 엽니다.

## 고급: 사용자 지정 스코프 (Custom scopes)

기본적으로 Hermes는 제공되는 모든 도구에 필요한 스코프를 요청합니다. 접근을 제한하고 싶다면 재정의하세요:

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

스코프 참조: [Spotify Web API 스코프](https://developer.spotify.com/documentation/web-api/concepts/scopes). 도구가 필요로 하는 것보다 적은 스코프를 요청하면 해당 도구의 호출은 403 오류로 실패합니다.

## 고급: 사용자 지정 Client ID / Redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

또는 `~/.hermes/.env`에 영구적으로 설정하세요:

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

리디렉션 URI는 Spotify 앱의 설정에서 허용 목록에 추가되어 있어야 합니다. 포트 43827이 이미 사용 중인 경우가 아니면 기본값으로 모든 사람에게 작동하므로 변경하지 마세요.

## 파일 위치 (Where things live)

| 파일 (File) | 내용 (Contents) |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | 액세스 토큰, 새로 고침 토큰, 만료 기한, 스코프, 리디렉션 URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`, (선택 사항) `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify 앱 | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)에서 본인이 소유함; Client ID와 허용된 리디렉션 URI 목록 포함 |
