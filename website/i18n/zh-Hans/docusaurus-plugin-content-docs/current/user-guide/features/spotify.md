# Spotify

Hermes 可以直接控制 Spotify —— 包括播放、队列、搜索、播放列表、已保存的曲目/专辑和听歌历史 —— 通过 Spotify 官方 Web API 配合 PKCE OAuth。Token 存储在 `~/.hermes/auth.json` 中，遇到 401 时自动刷新；每台机器只需登录一次。

与 Hermes 内置的 OAuth 集成（Google、GitHub Copilot、Codex）不同，Spotify 要求每位用户自行注册一个轻量级开发者应用。Spotify 不允许第三方发布可供任何人使用的公共 OAuth 应用。注册大约需要两分钟，`hermes auth spotify` 会引导你完成。

## 前置条件

- 一个 Spotify 账号。**免费版**即可使用搜索、播放列表、曲库和活动工具。**Premium** 需要用于播放控制（播放、暂停、跳过、跳转、音量、添加到队列、转移设备）。
- 已安装并运行 Hermes Agent。
- 使用播放工具时：需要一个**活跃的 Spotify Connect 设备** —— 至少有一台设备（手机、桌面端、网页播放器、音箱）打开 Spotify 应用，Web API 才有控制目标。如果没有活跃设备，你会收到 `403 Forbidden` 并提示 "no active device"；在任何设备上打开 Spotify 后重试即可。

## 设置

### 一键完成：`hermes tools`

最快的方式。运行：

```bash
hermes tools
```

滚动到 `🎵 Spotify`，按空格键开启，然后按 `s` 保存。Hermes 会直接带你进入 OAuth 流程 —— 如果你还没有 Spotify 应用，它会引导你在线创建。完成后，工具集和认证一步搞定。

如果你希望分步操作（或之后重新认证），请使用下面的两步流程。

### 两步流程

#### 1. 启用工具集

```bash
hermes tools
```

开启 `🎵 Spotify`，保存，当内联向导弹出时，按 Ctrl+C 关闭它。工具集保持开启状态；仅认证步骤被延后。

#### 2. 运行登录向导

```bash
hermes auth spotify
```

7 个 Spotify 工具只有在完成第 1 步后才会出现在 agent 的工具集中 —— 默认关闭，这样不需要它们的用户不会在每次 API 调用时携带额外的工具 schema。

如果未设置 `HERMES_SPOTIFY_CLIENT_ID`，Hermes 会内联引导你完成应用注册：

1. 在浏览器中打开 `https://developer.spotify.com/dashboard`
2. 打印需要粘贴到 Spotify "Create app" 表单中的精确值
3. 提示你输入返回的 Client ID
4. 将其保存到 `~/.hermes/.env`，以便后续运行跳过此步骤
5. 直接进入 OAuth 授权流程

授权完成后，token 会写入 `~/.hermes/auth.json` 的 `providers.spotify` 下。当前推理 provider **不会**改变 —— Spotify 认证与你的 LLM provider 相互独立。

### 创建 Spotify 应用（向导要求的内容）

当 Dashboard 打开后，点击 **Create app** 并填写：

| 字段 | 值 |
|-------|-------|
| App name | 任意（例如 `hermes-agent`） |
| App description | 任意（例如 `personal Hermes integration`） |
| Website | 留空 |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | 勾选 **Web API** |

同意条款并点击 **Save**。在下一页点击 **Settings** → 复制 **Client ID** 并粘贴到 Hermes 提示符中。这是 Hermes 唯一需要的值 —— PKCE 不使用 client secret。

### 通过 SSH / 无头环境运行

如果设置了 `SSH_CLIENT` 或 `SSH_TTY`，Hermes 在向导和 OAuth 步骤中都会跳过自动打开浏览器。复制 Hermes 打印的 Dashboard URL 和授权 URL，在本地机器的浏览器中打开，然后正常继续 —— 本地 HTTP 监听器仍在远程主机的 43827 端口运行。如果你需要通过 SSH 隧道访问该端口，请转发：`ssh -L 43827:127.0.0.1:43827 remote`。

## 验证

```bash
hermes auth status spotify
```

显示 token 是否存在以及 access token 的过期时间。刷新是自动的：当任何 Spotify API 调用返回 401 时，客户端会交换 refresh token 并重试一次。Refresh token 在 Hermes 重启后仍然有效，因此只有在你的 Spotify 账号设置中撤销应用或运行 `hermes auth logout spotify` 时才需要重新认证。

## 使用方法

登录后，agent 可以访问 7 个 Spotify 工具。你可以自然地与 agent 对话 —— 它会选择合适的工具和执行动作。为了获得最佳行为，agent 会加载一个配套 skill，教授规范的使用模式（单次搜索后播放、何时不要预检 `get_state` 等）。

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

### 工具参考

所有会改变播放状态的动作都接受可选的 `device_id` 参数以定位特定设备。如果省略，Spotify 会使用当前活跃设备。

#### `spotify_playback`
控制并查看播放状态，同时获取最近播放历史。

| Action | Purpose | Premium? |
|--------|---------|----------|
| `get_state` | 完整播放状态（曲目、设备、进度、随机/重复） | No |
| `get_currently_playing` | 仅当前曲目（204 时返回空 —— 见下文） | No |
| `play` | 开始/恢复播放。可选：`context_uri`、`uris`、`offset`、`position_ms` | Yes |
| `pause` | 暂停播放 | Yes |
| `next` / `previous` | 跳过曲目 | Yes |
| `seek` | 跳转到 `position_ms` | Yes |
| `set_repeat` | `state` = `track` / `context` / `off` | Yes |
| `set_shuffle` | `state` = `true` / `false` | Yes |
| `set_volume` | `volume_percent` = 0-100 | Yes |
| `recently_played` | 最近播放的曲目。可选 `limit`、`before`、`after`（Unix ms） | No |

#### `spotify_devices`
| Action | Purpose |
|--------|---------|
| `list` | 你账号下可见的所有 Spotify Connect 设备 |
| `transfer` | 将播放转移到 `device_id`。可选 `play: true` 在转移后开始播放 |

#### `spotify_queue`
| Action | Purpose | Premium? |
|--------|---------|----------|
| `get` | 当前队列中的曲目 | No |
| `add` | 将 `uri` 添加到队列 | Yes |

#### `spotify_search`
搜索曲库。`query` 是必需的。可选：`types`（`track` / `album` / `artist` / `playlist` / `show` / `episode` 的数组）、`limit`、`offset`、`market`。

#### `spotify_playlists`
| Action | Purpose | Required args |
|--------|---------|---------------|
| `list` | 用户的播放列表 | — |
| `get` | 单个播放列表 + 曲目 | `playlist_id` |
| `create` | 新建播放列表 | `name`（可选 `description`、`public`、`collaborative`） |
| `add_items` | 添加曲目 | `playlist_id`、`uris`（可选 `position`） |
| `remove_items` | 移除曲目 | `playlist_id`、`uris`（可选 `snapshot_id`） |
| `update_details` | 重命名 / 编辑 | `playlist_id` + `name`、`description`、`public`、`collaborative` 中的任意项 |

#### `spotify_albums`
| Action | Purpose | Required args |
|--------|---------|---------------|
| `get` | 专辑元数据 | `album_id` |
| `tracks` | 专辑曲目列表 | `album_id` |

#### `spotify_library`
统一访问已保存的曲目和专辑。使用 `kind` 参数选择集合。

| Action | Purpose |
|--------|---------|
| `list` | 分页曲库列表 |
| `save` | 将 `ids` / `uris` 添加到曲库 |
| `remove` | 从曲库中移除 `ids` / `uris` |

必需：`kind` = `tracks` 或 `albums`，加上 `action`。

### 功能矩阵：免费版 vs Premium

只读工具在免费版账号上可用。任何会改变播放状态或队列的操作都需要 Premium。

| 免费版可用 | 需要 Premium |
|---------------|------------------|
| `spotify_search`（全部） | `spotify_playback` — play、pause、next、previous、seek、set_repeat、set_shuffle、set_volume |
| `spotify_playback` — get_state、get_currently_playing、recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists`（全部） | |
| `spotify_albums`（全部） | |
| `spotify_library`（全部） | |

## 定时任务：Spotify + cron

由于 Spotify 工具是普通的 Hermes 工具，在 Hermes 会话中运行的 cron job 可以按任意计划触发播放。无需编写新代码。

### 早间唤醒播放列表

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

每个工作日上午 7 点会发生什么：
1. Cron 启动一个无头 Hermes 会话。
2. Agent 读取提示，调用 `spotify_devices list` 按名称查找 "kitchen speaker"，然后 `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play`。
3. 音乐在目标音箱上开始播放。总成本：一个会话、几次工具调用、无需人工输入。

### 夜间放松

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Pause Spotify. Then set volume to 20 so it's quiet when I start it again tomorrow."
```

### 注意事项

- **Cron 触发时必须存在活跃设备。** 如果没有 Spotify 客户端在运行（手机/桌面/Connect 音箱），播放动作会返回 `403 no active device`。对于早间播放列表，诀窍是定位一台始终在线的设备（Sonos、Echo、智能音箱）而不是你的手机。
- **任何会改变播放状态的操作都需要 Premium** —— play、pause、skip、volume、transfer。只读的 cron job（例如定时 "email me my recently played tracks"）在免费版上运行正常。
- **Cron agent 继承你当前启用的工具集。** Spotify 必须在 `hermes tools` 中启用，cron 会话才能看到 Spotify 工具。
- **Cron job 以 `skip_memory=True` 运行**，因此不会写入你的 memory store。

完整的 cron 参考：[Cron Jobs](./cron)。

## 退出登录

```bash
hermes auth logout spotify
```

从 `~/.hermes/auth.json` 中移除 token。如果要同时清除应用配置，请从 `~/.hermes/.env` 中删除 `HERMES_SPOTIFY_CLIENT_ID`（以及如果你设置过的 `HERMES_SPOTIFY_REDIRECT_URI`），或再次运行向导。

要在 Spotify 侧撤销应用授权，请访问 [Apps connected to your account](https://www.spotify.com/account/apps/) 并点击 **REMOVE ACCESS**。

## 故障排除

**`403 Forbidden — Player command failed: No active device found`** —— 你需要在至少一台设备上运行 Spotify。在手机上、桌面端或网页播放器中打开 Spotify 应用，播放任意曲目一秒钟以注册设备，然后重试。`spotify_devices list` 会显示当前可见的设备。

**`403 Forbidden — Premium required`** —— 你正在使用免费版账号尝试执行会改变播放状态的动作。参见上方的功能矩阵。

**`204 No Content` on `get_currently_playing`** —— 当前没有任何设备在播放。这是 Spotify 的正常响应，不是错误；Hermes 会将其作为解释性的空结果返回（`is_playing: false`）。

**`INVALID_CLIENT: Invalid redirect URI`** —— 你的 Spotify 应用设置中的 redirect URI 与 Hermes 使用的不匹配。默认值是 `http://127.0.0.1:43827/spotify/callback`。要么将其添加到你应用的允许 redirect URI 列表中，要么在 `~/.hermes/.env` 中设置 `HERMES_SPOTIFY_REDIRECT_URI` 为你注册的值。

**`429 Too Many Requests`** —— Spotify 的速率限制。Hermes 会返回友好的错误；等待一分钟后重试。如果持续出现，你可能在脚本中运行了紧凑循环 —— Spotify 的配额大约每 30 秒重置一次。

**`401 Unauthorized` 反复出现** —— 你的 refresh token 已被撤销（通常是因为你从账号中移除了应用，或应用被删除）。再次运行 `hermes auth spotify`。

**向导没有打开浏览器** —— 如果你通过 SSH 连接或在没有显示输出的容器中，Hermes 会检测到并跳过自动打开。复制它打印的 Dashboard URL 手动打开。

## 高级：自定义 scopes

默认情况下，Hermes 会请求所有已发布工具所需的 scopes。如果你想限制访问，可以覆盖：

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

Scope 参考：[Spotify Web API scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes)。如果你请求的 scope 少于某个工具所需，该工具的调用会失败并返回 403。

## 高级：自定义 client ID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

或在 `~/.hermes/.env` 中永久设置：

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

Redirect URI 必须在你 Spotify 应用的设置中列入白名单。默认值适用于绝大多数用户 —— 只有 43827 端口被占用时才需要更改。

## 文件位置

| 文件 | 内容 |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | access token、refresh token、过期时间、scope、redirect URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`、可选的 `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify app | 由你在 [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) 拥有；包含 Client ID 和 redirect URI 白名单 |
