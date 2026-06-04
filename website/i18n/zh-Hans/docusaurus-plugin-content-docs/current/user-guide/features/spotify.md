# Spotify

Hermes 可以直接控制 Spotify——播放、队列、搜索、播放列表、已保存的曲目/专辑以及收听历史——通过 Spotify 官方 Web API 配合 PKCE OAuth 实现。Token（令牌）存储在 `~/.hermes/auth.json` 中，遇到 401 时自动刷新；每台机器只需登录一次。

与 Hermes 内置的 OAuth 集成（Google、GitHub Copilot、Codex）不同，Spotify 要求每位用户自行注册一个轻量级开发者应用。Spotify 不允许第三方发布可供所有人使用的公共 OAuth 应用。整个过程大约需要两分钟，`hermes auth spotify` 会全程引导你完成。

## 前提条件

- 一个 Spotify 账号。**免费版**可使用搜索、播放列表、音乐库和活动工具。**Premium 版**才能使用播放控制（播放、暂停、跳曲、定位、音量、添加队列、切换设备）。
- 已安装并运行 Hermes Agent。
- 使用播放工具时：需要一个**活跃的 Spotify Connect 设备**——至少一台设备（手机、桌面端、网页播放器、音箱）上必须打开 Spotify 应用，Web API 才有对象可控制。若无活跃设备，将收到 `403 Forbidden` 并提示"no active device"；在任意设备上打开 Spotify 后重试即可。

## 设置

### 一键完成：`hermes tools` 或首次运行设置

最快捷的方式。运行：

```bash
hermes tools
```

滚动到 `🎵 Spotify`，按空格键启用，再按 `s` 保存。同样的开关也可在首次运行 `hermes setup` / `hermes setup tools` 流程中找到。Spotify 默认为可选启用，在此处启用会触发与 `hermes tools` 相同的提供商感知配置流程。

Hermes 会直接进入 OAuth 流程——如果你还没有 Spotify 应用，它会内联引导你创建一个。完成后，工具集即被启用并完成认证，一步到位。

如果你希望分步操作（或稍后重新认证），请使用下方的两步流程。

### 两步流程

#### 1. 启用工具集

```bash
hermes tools
```

启用 `🎵 Spotify`，保存，当内联向导弹出时关闭它（Ctrl+C）。工具集保持开启状态，仅跳过认证步骤。

#### 2. 运行登录向导

```bash
hermes auth spotify
```

7 个 Spotify 工具只有在完成第 1 步后才会出现在 agent 的工具集中——它们默认关闭，以避免不需要它们的用户在每次 API 调用时额外传输工具 schema。

若未设置 `HERMES_SPOTIFY_CLIENT_ID`，Hermes 会内联引导你完成应用注册：

1. 在浏览器中打开 `https://developer.spotify.com/dashboard`
2. 打印需要粘贴到 Spotify "Create app" 表单中的确切值
3. 提示你输入获得的 Client ID
4. 将其保存到 `~/.hermes/.env`，后续运行时跳过此步骤
5. 直接进入 OAuth 授权流程

授权完成后，token 将写入 `~/.hermes/auth.json` 的 `providers.spotify` 下。当前推理提供商不会改变——Spotify 认证与你的 LLM 提供商无关。

### 创建 Spotify 应用（向导所需内容）

当 dashboard 打开后，点击 **Create app** 并填写：

| 字段 | 值 |
|-------|-------|
| App name | 任意（例如 `hermes-agent`） |
| App description | 任意（例如 `personal Hermes integration`） |
| Website | 留空 |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | 勾选 **Web API** |

同意条款并点击 **Save**。在下一页点击 **Settings** → 复制 **Client ID** 并粘贴到 Hermes 提示中。这是 Hermes 唯一需要的值——PKCE 不使用 client secret。

### 通过 SSH / 在无头环境中运行

若设置了 `SSH_CLIENT` 或 `SSH_TTY`，Hermes 在向导和 OAuth 步骤中均会跳过自动打开浏览器。复制 Hermes 打印的 dashboard URL 和授权 URL，在本地机器的浏览器中打开，然后正常操作——本地 HTTP 监听器仍在远程主机的 `43827` 端口运行。你的笔记本浏览器无法直接访问远程回环地址，需要通过 SSH 本地端口转发：

```bash
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

关于跳板机/堡垒机设置及其他注意事项（mosh、tmux、端口冲突），请参阅 [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md)。

## 验证

```bash
hermes auth status spotify
```

显示 token 是否存在以及 access token 的过期时间。刷新是自动的：当任何 Spotify API 调用返回 401 时，客户端会用 refresh token 换取新 token 并重试一次。Refresh token 在 Hermes 重启后仍然有效，只有在你的 Spotify 账号设置中撤销该应用，或运行 `hermes auth logout spotify` 后才需要重新认证。

## 使用方法

登录后，agent 可访问 7 个 Spotify 工具。你用自然语言与 agent 交流——它会选择正确的工具和操作。为获得最佳效果，agent 会加载一个配套技能，教授规范的使用模式（先搜索再播放、何时不需要预先调用 `get_state` 等）。

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

所有会修改播放状态的操作都接受可选的 `device_id` 参数以指定目标设备。若省略，Spotify 将使用当前活跃设备。

#### `spotify_playback`
控制和查看播放状态，以及获取最近播放历史。

| 操作 | 用途 | 需要 Premium？ |
|--------|---------|----------|
| `get_state` | 完整播放状态（曲目、设备、进度、随机/循环） | 否 |
| `get_currently_playing` | 仅当前曲目（204 时返回空——见下文） | 否 |
| `play` | 开始/恢复播放。可选：`context_uri`、`uris`、`offset`、`position_ms` | 是 |
| `pause` | 暂停播放 | 是 |
| `next` / `previous` | 跳曲 | 是 |
| `seek` | 跳转到 `position_ms` | 是 |
| `set_repeat` | `state` = `track` / `context` / `off` | 是 |
| `set_shuffle` | `state` = `true` / `false` | 是 |
| `set_volume` | `volume_percent` = 0-100 | 是 |
| `recently_played` | 最近播放的曲目。可选 `limit`、`before`、`after`（Unix 毫秒） | 否 |

#### `spotify_devices`
| 操作 | 用途 |
|--------|---------|
| `list` | 你账号下所有可见的 Spotify Connect 设备 |
| `transfer` | 将播放切换到 `device_id`。可选 `play: true` 在切换时立即开始播放 |

### Home Assistant 管理的音箱

如果 Home Assistant 管理的音箱本身支持 Spotify Connect（例如 Sonos、Echo、Nest 或其他支持 Connect 的音箱），只要 Spotify 能识别它们，它们就会自动出现在 `spotify_devices list` 中。Hermes 不需要 Home Assistant ↔ Spotify 桥接——Spotify 原生处理设备路由。

通过音箱的显示名称让 Hermes 切换播放（例如"transfer Spotify to the kitchen speaker"），或在脚本中调用 `spotify_devices list` 获取确切的 `device_id` 后传给 `spotify_devices transfer`。若音箱未出现，请在 Spotify 应用或音箱的 Spotify 集成中打开一次，让 Spotify 将其注册为活跃的 Connect 目标。

#### `spotify_queue`
| 操作 | 用途 | 需要 Premium？ |
|--------|---------|----------|
| `get` | 当前队列中的曲目 | 否 |
| `add` | 将 `uri` 追加到队列 | 是 |

#### `spotify_search`
搜索曲库。`query` 为必填项。可选：`types`（`track` / `album` / `artist` / `playlist` / `show` / `episode` 的数组）、`limit`、`offset`、`market`。

#### `spotify_playlists`
| 操作 | 用途 | 必填参数 |
|--------|---------|---------------|
| `list` | 用户的播放列表 | — |
| `get` | 单个播放列表及其曲目 | `playlist_id` |
| `create` | 新建播放列表 | `name`（可选 `description`、`public`、`collaborative`） |
| `add_items` | 添加曲目 | `playlist_id`、`uris`（可选 `position`） |
| `remove_items` | 移除曲目 | `playlist_id`、`uris`（可选 `snapshot_id`） |
| `update_details` | 重命名/编辑 | `playlist_id` + `name`、`description`、`public`、`collaborative` 中的任意项 |

#### `spotify_albums`
| 操作 | 用途 | 必填参数 |
|--------|---------|---------------|
| `get` | 专辑元数据 | `album_id` |
| `tracks` | 专辑曲目列表 | `album_id` |

#### `spotify_library`
统一访问已保存的曲目和专辑。通过 `kind` 参数选择集合类型。

| 操作 | 用途 |
|--------|---------|
| `list` | 分页列出音乐库 |
| `save` | 将 `ids` / `uris` 添加到音乐库 |
| `remove` | 从音乐库移除 `ids` / `uris` |

必填：`kind` = `tracks` 或 `albums`，以及 `action`。

### 功能矩阵：免费版 vs Premium 版

只读工具在免费账号上可用。任何修改播放状态或队列的操作都需要 Premium。

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

由于 Spotify 工具是普通的 Hermes 工具，在 Hermes 会话中运行的 cron 任务可以按任意计划触发播放，无需编写额外代码。

### 早晨唤醒播放列表

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

每个工作日早上 7 点发生的事情：
1. Cron 启动一个无头 Hermes 会话。
2. Agent 读取 prompt（提示词），调用 `spotify_devices list` 按名称找到"kitchen speaker"，然后依次调用 `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play`。
3. 音乐在目标音箱上开始播放。总计：一个会话，几次工具调用，无需人工干预。

### 夜间收尾

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Pause Spotify. Then set volume to 20 so it's quiet when I start it again tomorrow."
```

### 注意事项

- **cron 触发时必须存在活跃设备。** 若没有 Spotify 客户端在运行（手机/桌面端/Connect 音箱），播放操作将返回 `403 no active device`。对于早晨播放列表，建议指定一个始终开机的设备（Sonos、Echo、智能音箱），而非手机。
- **任何修改播放状态的操作都需要 Premium**——播放、暂停、跳曲、音量、切换设备。只读 cron 任务（例如定时"发送我最近播放的曲目"）在免费版上可正常使用。
- **cron agent 继承你的活跃工具集。** Spotify 必须在 `hermes tools` 中启用，cron 会话才能看到 Spotify 工具。
- **Cron 任务以 `skip_memory=True` 运行**，不会写入你的记忆存储。

完整 cron 参考：[Cron Jobs](./cron)。

## 退出登录

```bash
hermes auth logout spotify
```

从 `~/.hermes/auth.json` 中移除 token。若还需清除应用配置，请从 `~/.hermes/.env` 中删除 `HERMES_SPOTIFY_CLIENT_ID`（以及 `HERMES_SPOTIFY_REDIRECT_URI`，如果你设置了的话），或重新运行向导。

若要在 Spotify 侧撤销应用，请访问[已连接到你账号的应用](https://www.spotify.com/account/apps/)并点击 **REMOVE ACCESS**。

## 故障排查

**`403 Forbidden — Player command failed: No active device found`** — 你需要在至少一台设备上运行 Spotify。在手机、桌面端或网页播放器上打开 Spotify 应用，随便播放一首曲目以注册设备，然后重试。`spotify_devices list` 可显示当前可见的设备。

**`403 Forbidden — Premium required`** — 你使用的是免费账号，但尝试执行需要 Premium 的播放操作。请参阅上方的功能矩阵。

**`get_currently_playing` 返回 `204 No Content`** — 当前所有设备上均无内容播放。这是 Spotify 的正常响应，不是错误；Hermes 将其呈现为说明性的空结果（`is_playing: false`）。

**`INVALID_CLIENT: Invalid redirect URI`** — 你的 Spotify 应用设置中的 redirect URI 与 Hermes 使用的不匹配。默认值为 `http://127.0.0.1:43827/spotify/callback`。请将其添加到应用的允许 redirect URI 列表中，或在 `~/.hermes/.env` 中将 `HERMES_SPOTIFY_REDIRECT_URI` 设置为你注册的值。

**`429 Too Many Requests`** — Spotify 的速率限制。Hermes 会返回友好的错误提示；等待一分钟后重试。若持续出现，你可能在脚本中运行了紧密循环——Spotify 的配额大约每 30 秒重置一次。

**`401 Unauthorized` 持续出现** — 你的 refresh token 已被撤销（通常是因为你从账号中移除了该应用，或应用被删除）。重新运行 `hermes auth spotify`。

**向导未打开浏览器** — 若你通过 SSH 连接或在没有显示器的容器中运行，Hermes 会检测到并跳过自动打开。复制它打印的 dashboard URL 并手动打开。

## 进阶：自定义 scope

默认情况下，Hermes 会请求所有已发布工具所需的 scope。若需限制访问权限，可覆盖默认值：

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

Scope 参考：[Spotify Web API scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes)。若请求的 scope 少于某个工具所需，该工具的调用将以 403 失败。

## 进阶：自定义 client ID / redirect URI

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

或在 `~/.hermes/.env` 中永久设置：

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

Redirect URI 必须在你的 Spotify 应用设置中加入白名单。默认值适用于绝大多数情况——只有在 43827 端口被占用时才需要更改。

## 文件位置

| 文件 | 内容 |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | access token、refresh token、过期时间、scope、redirect URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`，可选 `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify 应用 | 由你在 [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) 管理；包含 Client ID 和 redirect URI 白名单 |