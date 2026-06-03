---
title: "Spotify — Spotify：播放、搜索、队列、管理播放列表和设备"
sidebar_label: "Spotify"
description: "Spotify：播放、搜索、队列、管理播放列表和设备"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Spotify

Spotify：播放、搜索、队列、管理播放列表和设备。

## Skill 元数据

| | |
|---|---|
| 来源 | 内置（默认安装） |
| 路径 | `skills/media/spotify` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `spotify`, `music`, `playback`, `playlists`, `media` |
| 相关 skill | [`gif-search`](/user-guide/skills/bundled/media/media-gif-search) |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Spotify

通过 Hermes Spotify 工具集（7 个工具）控制用户的 Spotify 账户。设置指南：https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify

## 何时使用此 skill

用户说出类似以下内容时："play X"、"pause"、"skip"、"queue up X"、"what's playing"、"search for X"、"add to my X playlist"、"make a playlist"、"save this to my library" 等。

## 7 个工具

- `spotify_playback` — play、pause、next、previous、seek、set_repeat、set_shuffle、set_volume、get_state、get_currently_playing、recently_played
- `spotify_devices` — list、transfer
- `spotify_queue` — get、add
- `spotify_search` — 搜索曲库
- `spotify_playlists` — list、get、create、add_items、remove_items、update_details
- `spotify_albums` — get、tracks
- `spotify_library` — 使用 `kind: "tracks"|"albums"` 进行 list/save/remove

修改播放状态的操作需要 Spotify Premium；搜索/曲库/播放列表操作在免费版上也可使用。

## 规范模式（最小化工具调用次数）

### "Play &lt;artist/track/album>"
一次搜索，然后通过 URI 播放。除非用户要求选项，否则**不要**循环遍历搜索结果并逐一描述。

```
spotify_search({"query": "miles davis kind of blue", "types": ["album"], "limit": 1})
→ got album URI spotify:album:1weenld61qoidwYuZ1GESA
spotify_playback({"action": "play", "context_uri": "spotify:album:1weenld61qoidwYuZ1GESA"})
```

对于"play some &lt;artist>"（无特定歌曲），优先使用 `types: ["artist"]` 并播放艺术家的 context URI — Spotify 会自动处理智能随机播放。如果用户说"the song"或"that track"，则搜索 `types: ["track"]` 并将 `uris: [track_uri]` 传给 play。

### "What's playing?" / "What am I listening to?"
单次调用——不要在 get_currently_playing 之后再链式调用 get_state。

```
spotify_playback({"action": "get_currently_playing"})
```

如果返回 204/空（`is_playing: false`），告知用户当前没有播放内容。不要重试。

### "Pause" / "Skip" / "Volume 50"
直接执行操作，无需预先检查状态。

```
spotify_playback({"action": "pause"})
spotify_playback({"action": "next"})
spotify_playback({"action": "set_volume", "volume_percent": 50})
```

### "Add to my &lt;playlist name> playlist"
1. 用 `spotify_playlists list` 按名称查找播放列表 ID
2. 获取曲目 URI（来自当前播放，或通过搜索）
3. 用 playlist_id 和 URI 调用 `spotify_playlists add_items`

```
spotify_playlists({"action": "list"})
→ found "Late Night Jazz" = 37i9dQZF1DX4wta20PHgwo
spotify_playback({"action": "get_currently_playing"})
→ current track uri = spotify:track:0DiWol3AO6WpXZgp0goxAV
spotify_playlists({"action": "add_items",
                   "playlist_id": "37i9dQZF1DX4wta20PHgwo",
                   "uris": ["spotify:track:0DiWol3AO6WpXZgp0goxAV"]})
```

### "Create a playlist called X and add the last 3 songs I played"
```
spotify_playback({"action": "recently_played", "limit": 3})
spotify_playlists({"action": "create", "name": "Focus 2026"})
→ got playlist_id back in response
spotify_playlists({"action": "add_items", "playlist_id": <id>, "uris": [<3 uris>]})
```

### "Save / unsave / is this saved?"
使用 `spotify_library` 并指定正确的 `kind`。

```
spotify_library({"kind": "tracks", "action": "save", "uris": ["spotify:track:..."]})
spotify_library({"kind": "albums", "action": "list", "limit": 50})
```

### "Transfer playback to my &lt;device>"
```
spotify_devices({"action": "list"})
→ pick the device_id by matching name/type
spotify_devices({"action": "transfer", "device_id": "<id>", "play": true})
```

## 关键失败模式

**`403 Forbidden — No active device found`** 出现在任何播放操作上，意味着 Spotify 在任何地方都未运行。告知用户："请先在手机/桌面/网页播放器上打开 Spotify，随便播放一首曲目几秒钟，然后重试。"不要盲目重试工具调用——结果会完全相同。可以调用 `spotify_devices list` 确认；空列表意味着没有活跃设备。

**`403 Forbidden — Premium required`** 意味着用户使用的是免费版，并尝试修改播放状态。不要重试；告知用户此操作需要 Premium。读取操作仍然有效（搜索、播放列表、曲库、get_state）。

**`get_currently_playing` 返回 `204 No Content`** 不是错误——它表示当前没有播放内容。工具返回 `is_playing: false`。直接将此情况告知用户即可。

**`429 Too Many Requests`** = 速率限制。等待后重试一次。如果持续发生，说明你在循环——停止。

**`401 Unauthorized` 重试后仍出现** — 刷新令牌已被撤销。告知用户重新运行 `hermes auth spotify`。

## URI 和 ID 格式

Spotify 使用三种可互换的 ID 格式。工具接受所有三种并会自动规范化：

- URI：`spotify:track:0DiWol3AO6WpXZgp0goxAV`（推荐）
- URL：`https://open.spotify.com/track/0DiWol3AO6WpXZgp0goxAV`
- 裸 ID：`0DiWol3AO6WpXZgp0goxAV`

如有疑问，使用完整 URI。搜索结果在 `uri` 字段中返回 URI——直接传入即可。

实体类型：`track`、`album`、`artist`、`playlist`、`show`、`episode`。请为操作使用正确的类型——`spotify_playback.play` 的 `context_uri` 期望 album/playlist/artist；`uris` 期望曲目 URI 数组。

## 禁止事项

- **不要在每次操作前调用 `get_state`。** Spotify 接受 play/pause/skip 而无需预检。仅在用户询问"what's playing"或需要推断设备/曲目时才检查状态。
- **除非被要求，否则不要描述搜索结果。** 如果用户说"play X"，搜索、获取排名第一的 URI、播放。如果播放错了，他们自己会听出来。
- **不要在 `403 Premium required` 或 `403 No active device` 时重试。** 在用户采取行动之前，这些错误是永久性的。
- **不要用 `spotify_search` 按名称查找播放列表** — 那会搜索 Spotify 公开曲库。用户播放列表来自 `spotify_playlists list`。
- **不要在 `spotify_library` 中将 `kind: "tracks"` 与专辑 URI 混用**（反之亦然）。工具会规范化 ID，但 API 端点不同。