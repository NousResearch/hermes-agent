# Spotify

Hermes can control Spotify directly — playback, queue, search, playlists, saved tracks/albums, and listening history — using Spotify's official Web API with PKCE OAuth. Tokens are stored in `~/.hermes/auth.json` and refreshed automatically on 401; you only log in once per machine.

Unlike Hermes' built-in OAuth integrations (Google, GitHub Copilot, Codex), Spotify requires every user to register their own lightweight developer app. Spotify does not let third parties ship a public OAuth app that anyone can use. It takes about two minutes and `hermes auth spotify` walks you through it.

## Prerequisites

- A Spotify account. **Free** works for search, playlist, library, and activity tools. **Premium** is required for playback control (play, pause, skip, seek, volume, queue add, transfer).
- Hermes Agent installed and running.
- For playback tools: an **active Spotify Connect device** — the Spotify app must be open on at least one device (phone, desktop, web player, speaker) so the Web API has something to control. If nothing is active you'll get a `403 Forbidden` with a "no active device" message; open Spotify on any device and retry.

## Setup

### 1. Enable the toolset

```bash
hermes tools
```

Scroll to `🎵 Spotify`, press space to toggle it on, then `s` to save. The 9 Spotify tools only appear in the agent's toolset after this — they're off by default so users who don't want them don't ship extra tool schemas on every API call.

### 2. Run the login wizard

```bash
hermes auth spotify
```

If no `HERMES_SPOTIFY_CLIENT_ID` is set, Hermes walks you through the app registration inline:

1. Opens `https://developer.spotify.com/dashboard` in your browser
2. Prints the exact values to paste into Spotify's "Create app" form
3. Prompts you for the Client ID you get back
4. Saves it to `~/.hermes/.env` so future runs skip this step
5. Continues straight into the OAuth consent flow

After you approve, tokens are written under `providers.spotify` in `~/.hermes/auth.json`. The active inference provider is NOT changed — Spotify auth is independent of your LLM provider.

### Creating the Spotify app (what the wizard asks for)

When the dashboard opens, click **Create app** and fill in:

| Field | Value |
|-------|-------|
| App name | anything (e.g. `hermes-agent`) |
| App description | anything (e.g. `personal Hermes integration`) |
| Website | leave blank |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | check **Web API** |

Agree to the terms and click **Save**. On the next page click **Settings** → copy the **Client ID** and paste it into the Hermes prompt. That's the only value Hermes needs — PKCE doesn't use a client secret.

### Running over SSH / in a headless environment

If `SSH_CLIENT` or `SSH_TTY` is set, Hermes skips the automatic browser open during both the wizard and the OAuth step. Copy the dashboard URL and the authorization URL Hermes prints, open them in a browser on your local machine, and proceed normally — the local HTTP listener still runs on the remote host on port 43827. If you need to reach it through an SSH tunnel, forward that port: `ssh -L 43827:127.0.0.1:43827 remote`.

## Verify

```bash
hermes auth status spotify
```

Shows whether tokens are present and when the access token expires. Refresh is automatic: when any Spotify API call returns 401, the client exchanges the refresh token and retries once. Refresh tokens persist across Hermes restarts, so you only re-auth if you revoke the app in your Spotify account settings or run `hermes auth logout spotify`.

## Using it

Once logged in, the agent has access to 7 Spotify tools. You talk to the agent naturally — it picks the right tool and action. For the best behavior, the agent loads a companion skill that teaches canonical usage patterns (single-search-then-play, when not to preflight `get_state`, etc.).

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

### Tool reference

All playback-mutating actions accept an optional `device_id` to target a specific device. If omitted, Spotify uses the currently active device.

#### `spotify_playback`
Control and inspect playback, plus fetch recently played history.

| Action | Purpose | Premium? |
|--------|---------|----------|
| `get_state` | Full playback state (track, device, progress, shuffle/repeat) | No |
| `get_currently_playing` | Just the current track (returns empty on 204 — see below) | No |
| `play` | Start/resume playback. Optional: `context_uri`, `uris`, `offset`, `position_ms` | Yes |
| `pause` | Pause playback | Yes |
| `next` / `previous` | Skip track | Yes |
| `seek` | Jump to `position_ms` | Yes |
| `set_repeat` | `state` = `track` / `context` / `off` | Yes |
| `set_shuffle` | `state` = `true` / `false` | Yes |
| `set_volume` | `volume_percent` = 0-100 | Yes |
| `recently_played` | Last played tracks. Optional `limit`, `before`, `after` (Unix ms) | No |

#### `spotify_devices`
| Action | Purpose |
|--------|---------|
| `list` | Every Spotify Connect device visible to your account |
| `transfer` | Move playback to `device_id`. Optional `play: true` starts playback on transfer |

#### `spotify_queue`
| Action | Purpose | Premium? |
|--------|---------|----------|
| `get` | Currently queued tracks | No |
| `add` | Append `uri` to the queue | Yes |

#### `spotify_search`
Search the catalog. `query` is required. Optional: `types` (array of `track` / `album` / `artist` / `playlist` / `show` / `episode`), `limit`, `offset`, `market`.

#### `spotify_playlists`
| Action | Purpose | Required args |
|--------|---------|---------------|
| `list` | User's playlists | — |
| `get` | One playlist + tracks | `playlist_id` |
| `create` | New playlist | `name` (+ optional `description`, `public`, `collaborative`) |
| `add_items` | Add tracks | `playlist_id`, `uris` (optional `position`) |
| `remove_items` | Remove tracks | `playlist_id`, `uris` (+ optional `snapshot_id`) |
| `update_details` | Rename / edit | `playlist_id` + any of `name`, `description`, `public`, `collaborative` |

#### `spotify_albums`
| Action | Purpose | Required args |
|--------|---------|---------------|
| `get` | Album metadata | `album_id` |
| `tracks` | Album track list | `album_id` |

#### `spotify_library`
Unified access to saved tracks and saved albums. Pick the collection with the `kind` arg.

| Action | Purpose |
|--------|---------|
| `list` | Paginated library listing |
| `save` | Add `ids` / `uris` to library |
| `remove` | Remove `ids` / `uris` from library |

Required: `kind` = `tracks` or `albums`, plus `action`.

### Feature matrix: Free vs Premium

Read-only tools work on Free accounts. Anything that mutates playback or the queue requires Premium.

| Works on Free | Premium required |
|---------------|------------------|
| `spotify_search` (all) | `spotify_playback` — play, pause, next, previous, seek, set_repeat, set_shuffle, set_volume |
| `spotify_playback` — get_state, get_currently_playing, recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists` (all) | |
| `spotify_albums` (all) | |
| `spotify_library` (all) | |

## Sign out

```bash
hermes auth logout spotify
```

Removes tokens from `~/.hermes/auth.json`. To also clear the app config, delete `HERMES_SPOTIFY_CLIENT_ID` (and `HERMES_SPOTIFY_REDIRECT_URI` if you set it) from `~/.hermes/.env`, or run the wizard again.

To revoke the app on Spotify's side, visit [Apps connected to your account](https://www.spotify.com/account/apps/) and click **REMOVE ACCESS**.

## Troubleshooting

**`403 Forbidden — Player command failed: No active device found`** — You need Spotify running on at least one device. Open the Spotify app on your phone, desktop, or web player, start any track for a second to register it, and retry. `spotify_devices list` shows what's currently visible.

**`403 Forbidden — Premium required`** — You're on a Free account trying to use a playback-mutating action. See the feature matrix above.

**`204 No Content` on `get_currently_playing`** — nothing is currently playing on any device. This is Spotify's normal response, not an error; Hermes surfaces it as an explanatory empty result (`is_playing: false`).

**`INVALID_CLIENT: Invalid redirect URI`** — the redirect URI in your Spotify app settings doesn't match what Hermes is using. The default is `http://127.0.0.1:43827/spotify/callback`. Either add that to your app's allowed redirect URIs, or set `HERMES_SPOTIFY_REDIRECT_URI` in `~/.hermes/.env` to whatever you registered.

**`429 Too Many Requests`** — Spotify's rate limit. Hermes returns a friendly error; wait a minute and retry. If this persists, you're probably running a tight loop in a script — Spotify's quota resets roughly every 30 seconds.

**`401 Unauthorized` keeps coming back** — Your refresh token was revoked (usually because you removed the app from your account, or the app was deleted). Run `hermes auth spotify` again.

**Wizard doesn't open the browser** — If you're over SSH or in a container without a display, Hermes detects it and skips the auto-open. Copy the dashboard URL it prints and open it manually.

## Advanced: custom scopes

By default Hermes requests the scopes needed for every shipped tool. Override if you want to restrict access:

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

Scope reference: [Spotify Web API scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes). If you request fewer scopes than a tool needs, that tool's calls will fail with 403.

## Advanced: custom client ID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

Or set them permanently in `~/.hermes/.env`:

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

The redirect URI must be allow-listed in your Spotify app's settings. The default works for almost everyone — only change it if port 43827 is taken.

## Where things live

| File | Contents |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | access token, refresh token, expiry, scope, redirect URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`, optional `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify app | owned by you at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard); contains the Client ID and the redirect URI allow-list |
