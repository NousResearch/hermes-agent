# Spotify

Hermes는 Spotify의 공식 Web API와 PKCE OAuth를 사용해 재생, 대기열, 검색, 플레이리스트, 저장한 트랙/앨범, 청취 기록을 직접 제어할 수 있습니다. 토큰은 `~/.hermes/auth.json`에 저장되며 401 응답이 오면 자동으로 갱신됩니다. 컴퓨터마다 한 번만 로그인하면 됩니다(갱신 토큰은 약 6개월 후 만료되므로 만료되면 `hermes auth spotify`를 다시 실행하세요).

Hermes의 기본 제공 OAuth 통합(Google, GitHub Copilot, Codex)과 달리 Spotify에서는 모든 사용자가 직접 간단한 개발자 앱을 등록해야 합니다. Spotify는 누구나 사용할 수 있는 공개 OAuth 앱을 제3자가 제공하는 것을 허용하지 않습니다. 등록에는 약 2분이 걸리며 `hermes auth spotify`가 전체 과정을 안내합니다.

## 사전 요구 사항

- Spotify 계정. **Free** 계정은 검색, 플레이리스트, 라이브러리, 활동 도구를 사용할 수 있습니다. 재생 제어(재생, 일시 중지, 건너뛰기, 탐색, 볼륨, 대기열 추가, 전송)에는 **Premium**이 필요합니다.
- Hermes Agent가 설치되어 실행 중이어야 합니다.
- 재생 도구를 사용하려면 **활성 Spotify Connect 기기**가 필요합니다. Web API가 제어할 대상을 찾을 수 있도록 Spotify 앱이 하나 이상의 기기(휴대폰, 데스크톱, 웹 플레이어, 스피커)에서 열려 있어야 합니다. 활성 기기가 없으면 "no active device" 메시지와 함께 `403 Forbidden`이 반환됩니다. 아무 기기에서나 Spotify를 열고 다시 시도하세요.

## 설정

### 한 번에 설정하기: `hermes tools` 또는 최초 실행 설정

가장 빠른 방법입니다. 다음을 실행하세요.

```bash
hermes tools
```

`🎵 Spotify`로 스크롤하고 스페이스 키를 눌러 켠 다음 `s`를 눌러 저장하세요. 최초 실행 시 `hermes setup` / `hermes setup tools` 과정에서도 동일한 토글을 사용할 수 있습니다. Spotify는 옵트인 방식으로 유지되므로 여기서 활성화하면 `hermes tools`와 동일한 제공자 인식 설정이 실행됩니다.

Hermes는 바로 OAuth 과정으로 들어갑니다. 아직 Spotify 앱이 없다면 인라인으로 앱 생성을 안내합니다. 완료하면 도구 모음이 활성화되고 인증도 한 번에 끝납니다.

단계를 따로 진행하고 싶거나 나중에 다시 인증하려면 아래의 2단계 흐름을 사용하세요.

### 2단계 흐름

#### 1. 도구 모음 활성화

```bash
hermes tools
```

`🎵 Spotify`를 켜고 저장한 뒤 인라인 마법사가 열리면 종료하세요(Ctrl+C). 도구 모음은 켜진 상태로 유지되고 인증 단계만 미뤄집니다.

#### 2. 로그인 마법사 실행

```bash
hermes auth spotify
```

1단계가 끝난 뒤에만 에이전트의 도구 모음에 7개의 Spotify 도구가 표시됩니다. 기본적으로 꺼져 있으므로 사용하지 않는 사용자는 모든 API 호출마다 추가 도구 스키마를 전송하지 않습니다.

`HERMES_SPOTIFY_CLIENT_ID`가 설정되어 있지 않으면 Hermes가 인라인으로 앱 등록을 안내합니다.

1. 브라우저에서 `https://developer.spotify.com/dashboard`를 엽니다.
2. Spotify의 "Create app" 양식에 붙여 넣을 정확한 값을 출력합니다.
3. 발급받은 Client ID를 입력하라는 메시지를 표시합니다.
4. Client ID를 `~/.hermes/.env`에 저장하여 다음 실행부터 이 단계를 건너뜁니다.
5. OAuth 동의 과정으로 바로 이어집니다.

승인하면 토큰이 `~/.hermes/auth.json`의 `providers.spotify` 아래에 기록됩니다. 현재 활성화된 추론 제공자는 변경되지 않습니다. Spotify 인증은 LLM 제공자와 독립적입니다.

### Spotify 앱 생성(마법사가 요청하는 항목)

대시보드가 열리면 **Create app**을 클릭하고 다음과 같이 입력하세요.

| 항목 | 값 |
|-------|-------|
| 앱 이름 | 아무 값이나(예: `hermes-agent`) |
| 앱 설명 | 아무 값이나(예: `personal Hermes integration`) |
| 웹사이트 | 비워 둡니다 |
| 리디렉션 URI | `http://127.0.0.1:43827/spotify/callback` |
| 사용할 API/SDK | **Web API** 선택 |

약관에 동의하고 **Save**를 클릭하세요. 다음 페이지에서 **Settings**를 클릭한 뒤 **Client ID**를 복사하여 Hermes 프롬프트에 붙여 넣습니다. Hermes에 필요한 값은 이것뿐이며 PKCE에서는 client secret을 사용하지 않습니다.

### SSH / 헤드리스 환경에서 실행

`SSH_CLIENT` 또는 `SSH_TTY`가 설정되어 있으면 마법사와 OAuth 단계에서 자동 브라우저 열기를 건너뜁니다. Hermes가 출력하는 대시보드 URL과 인증 URL을 복사해 로컬 컴퓨터의 브라우저에서 열고 평소처럼 진행하세요. 로컬 HTTP 리스너는 원격 호스트의 `43827` 포트에서 계속 실행됩니다. 노트북의 브라우저가 원격 루프백에 접근하려면 SSH 로컬 포워딩이 필요합니다.

```bash
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

점프 박스/배스천 설정 및 기타 문제(mosh, tmux, 포트 충돌)는 [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md)를 참고하세요.

## 확인

```bash
hermes auth status spotify
```

토큰이 있는지와 액세스 토큰의 만료 시점을 보여 줍니다. 갱신은 자동입니다. Spotify API 호출이 401을 반환하면 클라이언트가 갱신 토큰으로 교환한 뒤 한 번 다시 시도합니다. 갱신 토큰은 Hermes를 다시 시작해도 유지되므로 Spotify 계정 설정에서 앱을 철회했거나 `hermes auth logout spotify`를 실행한 경우에만 다시 인증하면 됩니다.

## 사용

로그인하면 에이전트가 7개의 Spotify 도구에 접근할 수 있습니다. 에이전트에게 자연스럽게 말하면 적절한 도구와 작업을 선택합니다. 가장 좋은 동작을 위해 에이전트는 표준 사용 패턴(한 번 검색한 뒤 재생하기, `get_state`를 미리 호출하지 않아야 하는 경우 등)을 알려 주는 보조 스킬을 불러옵니다.

```
> play some miles davis
> what am I listening to
> add this track to my Late Night Jazz playlist
> skip to the next song
> make a new playlist called "Focus 2026" and add the last three songs I played
> which of my saved albums are by Radiohead
> search for acoustic covers of Blackbird
> transfer playback to my kitchen speaker
```

### 도구 참조

재생을 변경하는 모든 작업은 특정 기기를 대상으로 지정할 수 있는 선택적 `device_id`를 받습니다. 지정하지 않으면 Spotify가 현재 활성 기기를 사용합니다.

#### `spotify_playback`
재생을 제어하고 확인하며 최근 재생 기록을 가져옵니다.

| 작업 | 용도 | Premium? |
|--------|---------|----------|
| `get_state` | 전체 재생 상태(트랙, 기기, 진행률, 셔플/반복) | 아니요 |
| `get_currently_playing` | 현재 트랙만 반환(204이면 비어 있음 — 아래 참조) | 아니요 |
| `play` | 재생 시작/재개. 선택 사항: `context_uri`, `uris`, `offset`, `position_ms` | 예 |
| `pause` | 재생 일시 중지 | 예 |
| `next` / `previous` | 트랙 건너뛰기 | 예 |
| `seek` | `position_ms` 위치로 이동 | 예 |
| `set_repeat` | `state` = `track` / `context` / `off` | 예 |
| `set_shuffle` | `state` = `true` / `false` | 예 |
| `set_volume` | `volume_percent` = 0-100 | 예 |
| `recently_played` | 최근 재생 트랙. 선택 사항: `limit`, `before`, `after`(Unix ms) | 아니요 |

#### `spotify_devices`
| 작업 | 용도 |
|--------|---------|
| `list` | 계정에서 확인할 수 있는 모든 Spotify Connect 기기 |
| `transfer` | 재생을 `device_id`로 이동. 선택 사항인 `play: true`는 이동 후 재생을 시작합니다 |

### Home Assistant로 관리하는 스피커

Home Assistant가 이미 Spotify Connect를 지원하는 스피커(예: Sonos, Echo, Nest 또는 기타 Connect 지원 스피커)를 관리한다면 Spotify가 해당 스피커를 확인할 수 있을 때 `spotify_devices list`에 자동으로 표시됩니다. 이 경로에서는 Hermes에 Home Assistant ↔ Spotify 브리지가 필요하지 않습니다. Spotify가 기기 라우팅을 기본적으로 처리합니다.

스피커의 표시 이름으로 Hermes에 재생을 전송해 달라고 요청하거나(예: “transfer Spotify to the kitchen speaker”), `spotify_devices list`를 호출한 뒤 스크립트에서 정확한 `device_id`를 `spotify_devices transfer`에 전달하세요. 스피커가 보이지 않으면 Spotify 앱 또는 스피커의 Spotify 통합을 한 번 열어 Spotify가 이를 활성 Connect 대상으로 등록하도록 하세요.

#### `spotify_queue`
| 작업 | 용도 | Premium? |
|--------|---------|----------|
| `get` | 현재 대기열의 트랙 | 아니요 |
| `add` | `uri`를 대기열에 추가 | 예 |

#### `spotify_search`
카탈로그를 검색합니다. `query`는 필수입니다. 선택 사항: `types`(`track` / `album` / `artist` / `playlist` / `show` / `episode` 배열), `limit`, `offset`, `market`.

#### `spotify_playlists`
| 작업 | 용도 | 필수 인수 |
|--------|---------|---------------|
| `list` | 사용자의 플레이리스트 | — |
| `get` | 플레이리스트 하나와 트랙 | `playlist_id` |
| `create` | 새 플레이리스트 | `name`(및 선택 사항인 `description`, `public`, `collaborative`) |
| `add_items` | 트랙 추가 | `playlist_id`, `uris`(선택 사항인 `position`) |
| `remove_items` | 트랙 제거 | `playlist_id`, `uris`(선택 사항인 `snapshot_id`) |
| `update_details` | 이름 변경 / 편집 | `playlist_id` 및 `name`, `description`, `public`, `collaborative` 중 하나 이상 |

#### `spotify_albums`
| 작업 | 용도 | 필수 인수 |
|--------|---------|---------------|
| `get` | 앨범 메타데이터 | `album_id` |
| `tracks` | 앨범 트랙 목록 | `album_id` |

#### `spotify_library`
저장한 트랙과 저장한 앨범에 통합적으로 접근합니다. `kind` 인수로 컬렉션을 선택하세요.

| 작업 | 용도 |
|--------|---------|
| `list` | 페이지 단위 라이브러리 목록 |
| `save` | 라이브러리에 `ids` / `uris` 추가 |
| `remove` | 라이브러리에서 `ids` / `uris` 제거 |

필수: `kind` = `tracks` 또는 `albums`와 `action`.

### 기능 매트릭스: Free와 Premium

읽기 전용 도구는 Free 계정에서 작동합니다. 재생 또는 대기열을 변경하는 작업에는 Premium이 필요합니다.

| Free에서 작동 | Premium 필요 |
|---------------|------------------|
| `spotify_search` (전체) | `spotify_playback` — play, pause, next, previous, seek, set_repeat, set_shuffle, set_volume |
| `spotify_playback` — get_state, get_currently_playing, recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists` (전체) | |
| `spotify_albums` (전체) | |
| `spotify_library` (전체) | |

## 예약 실행: Spotify + cron

Spotify 도구는 일반 Hermes 도구이므로 Hermes 세션에서 실행되는 cron 작업이 어떤 일정으로든 재생을 시작할 수 있습니다. 새 코드는 필요하지 않습니다.

### 아침 기상 플레이리스트

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

평일 오전 7시에 다음과 같이 실행됩니다.

1. cron이 헤드리스 Hermes 세션을 시작합니다.
2. 에이전트가 프롬프트를 읽고 `spotify_devices list`를 호출해 이름으로 "kitchen speaker"를 찾은 다음, `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play`를 호출합니다.
3. 대상 스피커에서 음악이 시작됩니다. 총 비용은 세션 하나와 몇 번의 도구 호출이며 사람의 입력은 필요하지 않습니다.

### 밤에 긴장 풀기

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Pause Spotify. Then set volume to 20 so it's quiet when I start it again tomorrow."
```

### 주의 사항

- **cron이 실행될 때 활성 기기가 있어야 합니다.** Spotify 클라이언트(휴대폰/데스크톱/Connect 스피커)가 실행 중이지 않으면 재생 작업이 `403 no active device`를 반환합니다. 아침 플레이리스트에는 휴대폰 대신 항상 켜져 있는 기기(Sonos, Echo, 스마트 스피커)를 대상으로 지정하는 것이 좋습니다.
- **재생을 변경하는 모든 작업에는 Premium이 필요합니다.** 재생, 일시 중지, 건너뛰기, 볼륨, 전송이 이에 해당합니다. 읽기 전용 cron 작업(예정된 "최근 재생 트랙을 이메일로 보내기")은 Free에서도 문제없이 작동합니다.
- **cron 에이전트는 활성 도구 모음을 상속합니다.** cron 세션에서 Spotify 도구를 보려면 `hermes tools`에서 Spotify를 활성화해야 합니다.
- **cron 작업은 `skip_memory=True`로 실행되므로 메모리 저장소에 기록하지 않습니다.**

전체 cron 참조: [Cron Jobs](./cron).

## 로그아웃

```bash
hermes auth logout spotify
```

`~/.hermes/auth.json`에서 토큰을 제거합니다. 앱 설정도 지우려면 `~/.hermes/.env`에서 `HERMES_SPOTIFY_CLIENT_ID`(설정했다면 `HERMES_SPOTIFY_REDIRECT_URI`도)를 삭제하거나 마법사를 다시 실행하세요.

Spotify 측에서 앱을 철회하려면 [계정에 연결된 앱](https://www.spotify.com/account/apps/)을 방문하여 **REMOVE ACCESS**를 클릭하세요.

## 문제 해결

**`403 Forbidden — Player command failed: No active device found`** — 하나 이상의 기기에서 Spotify를 실행해야 합니다. 휴대폰, 데스크톱 또는 웹 플레이어에서 Spotify 앱을 열고 트랙을 잠시 재생해 기기를 등록한 다음 다시 시도하세요. `spotify_devices list`로 현재 표시되는 기기를 확인할 수 있습니다.

**`403 Forbidden — Premium required`** — 재생을 변경하는 작업을 Free 계정에서 사용하고 있습니다. 위의 기능 매트릭스를 참고하세요.

**`get_currently_playing`에서 `204 No Content`** — 어떤 기기에서도 현재 재생 중인 항목이 없습니다. 이는 Spotify의 정상 응답이며 오류가 아닙니다. Hermes는 이를 설명이 포함된 빈 결과(`is_playing: false`)로 표시합니다.

**`INVALID_CLIENT: Invalid redirect URI`** — Spotify 앱 설정의 리디렉션 URI가 Hermes가 사용하는 URI와 일치하지 않습니다. 기본값은 `http://127.0.0.1:43827/spotify/callback`입니다. 앱의 허용된 리디렉션 URI에 해당 값을 추가하거나 `~/.hermes/.env`에서 `HERMES_SPOTIFY_REDIRECT_URI`를 등록한 값으로 설정하세요.

**`429 Too Many Requests`** — Spotify의 요청 제한입니다. Hermes가 알기 쉬운 오류를 반환하므로 1분 기다렸다가 다시 시도하세요. 계속되면 스크립트에서 짧은 간격으로 반복 실행 중일 가능성이 높습니다. Spotify의 할당량은 약 30초마다 초기화됩니다.

**`401 Unauthorized`가 계속 발생함** — 갱신 토큰이 철회되었습니다(대개 계정에서 앱을 제거했거나 앱이 삭제된 경우). `hermes auth spotify`를 다시 실행하세요.

**마법사가 브라우저를 열지 않음** — SSH를 사용 중이거나 디스플레이가 없는 컨테이너 안에 있으면 Hermes가 이를 감지하고 자동 열기를 건너뜁니다. 출력된 대시보드 URL을 복사해 수동으로 여세요.

## 고급: 사용자 지정 scopes

기본적으로 Hermes는 제공되는 모든 도구에 필요한 scopes를 요청합니다. 접근을 제한하고 싶다면 재정의할 수 있습니다.

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

Scope 참조: [Spotify Web API scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes). 도구에 필요한 것보다 적은 scopes를 요청하면 해당 도구의 호출이 403으로 실패합니다.

## 고급: 사용자 지정 client ID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

또는 `~/.hermes/.env`에 영구적으로 설정하세요.

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

리디렉션 URI는 Spotify 앱 설정에서 허용 목록에 있어야 합니다. 기본값은 거의 모든 사용자에게 작동하므로 포트 43827이 사용 중일 때만 변경하세요.

## 파일 위치

| 파일 | 내용 |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | 액세스 토큰, 갱신 토큰, 만료 시점, scope, 리디렉션 URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`, 선택 사항인 `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify 앱 | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)의 본인 소유 앱. Client ID와 리디렉션 URI 허용 목록이 포함됩니다 |
