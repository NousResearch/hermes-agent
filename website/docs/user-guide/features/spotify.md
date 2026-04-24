# Spotify

Hermes can control Spotify directly — playback, queue, search, playlists, saved tracks/albums, and listening history — using Spotify's official Web API with PKCE OAuth.

Unlike most Hermes integrations, Spotify requires every user to register their own lightweight developer app. Spotify does not let third parties ship a public OAuth app that anyone can use. The whole thing takes about two minutes.

## Prerequisites

- A Spotify account (Free works for most tools; **playback control requires Premium**)
- Hermes Agent installed and running

## Setup

### 1. Enable the toolset

```bash
hermes tools
```

Scroll to `🎵 Spotify`, press space to toggle it on, then `s` to save.

### 2. Run the login wizard

```bash
hermes auth spotify
```

If you don't have a Spotify app yet, Hermes walks you through creating one:

1. Opens the Spotify developer dashboard in your browser
2. Tells you exactly what values to paste into the Spotify form
3. Prompts you for the `Client ID` you get back
4. Saves it to `~/.hermes/.env` and continues straight into the OAuth flow

After the Spotify consent page, tokens are saved under `providers.spotify` in `~/.hermes/auth.json` and the integration is live.

### Creating the Spotify app (what the wizard asks for)

When you land on the dashboard, click **Create app** and fill in:

| Field | Value |
|-------|-------|
| App name | anything (e.g. `hermes-agent`) |
| App description | anything (e.g. `personal Hermes integration`) |
| Website | leave blank |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | **Web API** |

Agree to the terms, click **Save**. On the next screen click **Settings** → copy the **Client ID**. That's the only value Hermes needs (no client secret — PKCE doesn't use one).

## Verify

```bash
hermes auth status spotify
```

Shows whether tokens are present and when the access token expires. Hermes automatically refreshes on 401.

## Using it

Once logged in, the agent has access to 9 Spotify tools:

| Tool | Actions |
|------|---------|
| `spotify_playback` | play, pause, skip, seek, volume, now playing, playback state |
| `spotify_devices` | list devices, transfer playback |
| `spotify_queue` | inspect queue, add tracks to queue |
| `spotify_search` | search tracks, albums, artists, playlists |
| `spotify_playlists` | list, get, create, update, add/remove tracks |
| `spotify_albums` | get album, list album tracks |
| `spotify_saved_tracks` | list, save, remove |
| `spotify_saved_albums` | list, save, remove |
| `spotify_activity` | recently played, now playing |

The agent picks the right tool automatically. Ask it to "play some Miles Davis," "what am I listening to," "add the current track to my starred playlist," etc.

## Sign out

```bash
hermes auth logout spotify
```

Removes tokens from `~/.hermes/auth.json`. To also clear the app config, delete `HERMES_SPOTIFY_CLIENT_ID` (and optionally `HERMES_SPOTIFY_REDIRECT_URI`) from `~/.hermes/.env`.

## Troubleshooting

**`403 Forbidden` on playback endpoints** — Spotify requires Premium for `play`, `pause`, `skip`, and volume control. Search, playlists, and library reads work on Free.

**`204 No Content` on `now_playing`** — nothing is currently playing; expected behavior, not an error.

**`INVALID_CLIENT: Invalid redirect URI`** — the redirect URI registered in your Spotify app doesn't match what Hermes is using. Default is `http://127.0.0.1:43827/spotify/callback`. If you picked something else, set `HERMES_SPOTIFY_REDIRECT_URI` in `~/.hermes/.env` to match.

**`429 Too Many Requests`** — Spotify rate limit. Hermes surfaces this as a friendly error; wait a minute and retry.

## Advanced: custom scopes

By default Hermes requests the scopes needed for every shipped tool. To override:

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

See Spotify's [scope reference](https://developer.spotify.com/documentation/web-api/concepts/scopes) for available values.

## Advanced: custom client ID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

Or set them permanently in `~/.hermes/.env`:

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

The redirect URI must be allow-listed in your Spotify app's settings.
