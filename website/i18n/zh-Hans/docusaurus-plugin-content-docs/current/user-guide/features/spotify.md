# Spotify

Hermes 可以直接控制 Spotify——播放、队列、搜索、播放列表、保存的曲目/专辑和收听历史——使用 Spotify 官方 Web API 和 PKCE OAuth。令牌存储在 `~/.hermes/auth.json` 中，收到 401 时自动刷新；每台机器只需登录一次。

与 Hermes 内置的 OAuth 集成（Google、GitHub Copilot、Codex）不同，Spotify 要求每个用户注册自己的轻量级开发者应用。Spotify 不允许第三方发布任何人都可以使用的公共 OAuth 应用。这需要大约两分钟，`hermes auth spotify` 会引导你完成。

## 前置条件

- Spotify 账户。**免费**账户可以使用搜索、播放列表、资料库和活动工具。**Premium** 是播放控制（播放、暂停、跳过、搜索、音量、队列添加、传输）所必需的。
- Hermes Agent 已安装并运行。
- 对于播放工具：**需要一个活动的 Spotify Connect 设备**——Spotify 应用必须在至少一个设备（手机、台式机、网络播放器、扬声器）上打开，以便 Web API 有东西可以控制。如果没有活动设备，你会得到 `403 Forbidden` 和"no active device"消息；在任何设备上打开 Spotify 并重试。

## 设置

### 一次性：`hermes tools`

最快的方式。运行：

```bash
hermes tools
```

滚动到 `🎵 Spotify`，按空格键将其打开，然后按 `s` 保存。Hermes 直接进入 OAuth 流程——如果你还没有 Spotify 应用，它会引导你在线创建一个。完成后，工具集在一遍中启用并认证。

### 两步流程

#### 1. 启用工具集

```bash
hermes tools
```

将 `🎵 Spotify` 打开，保存，当内联向导打开时，关闭它（Ctrl+C）。工具集保持打开；只有认证步骤被推迟。

#### 2. 运行登录向导

```bash
hermes auth spotify
```

## 验证

```bash
hermes auth status spotify
```

显示令牌是否存在以及访问令牌何时过期。刷新是自动的：当任何 Spotify API 调用返回 401 时，客户端交换刷新令牌并重试一次。

## 使用

登录后，agent 可以访问 7 个 Spotify 工具。你可以自然地与 agent 对话——它会选择正确的工具和操作。

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

#### `spotify_playback`
控制和检查播放，加上获取最近播放历史。

| 操作 | 用途 | Premium? |
|--------|---------|----------|
| `get_state` | 完整播放状态（曲目、设备、进度、随机/重复） | 否 |
| `get_currently_playing` | 仅当前曲目（返回空时为 204） | 否 |
| `play` | 开始/恢复播放 | 是 |
| `pause` | 暂停播放 | 是 |
| `next` / `previous` | 跳过曲目 | 是 |
| `seek` | 跳转到 `position_ms` | 是 |
| `set_repeat` | `state` = `track` / `context` / `off` | 是 |
| `set_shuffle` | `state` = `true` / `false` | 是 |
| `set_volume` | `volume_percent` = 0-100 | 是 |
| `recently_played` | 最近播放的曲目 | 否 |

#### `spotify_devices`
| 操作 | 用途 |
|--------|---------|
| `list` | 你的账户可见的每个 Spotify Connect 设备 |
| `transfer` | 将播放转移到 `device_id` |

#### `spotify_queue`
| 操作 | 用途 | Premium? |
|--------|---------|---------|
| `get` | 当前排队的曲目 | 否 |
| `add` | 将 `uri` 追加到队列 | 是 |

#### `spotify_search`
搜索目录。必需：`query`。

#### `spotify_playlists`
| 操作 | 用途 |
|--------|---------|
| `list` | 用户的播放列表 |
| `get` | 一个播放列表 + 曲目 |
| `create` | 新播放列表 |
| `add_items` | 添加曲目 |
| `remove_items` | 移除曲目 |
| `update_details` | 重命名/编辑 |

#### `spotify_library`
统一访问保存的曲目和保存的专辑。

| 操作 | 用途 |
|--------|---------|
| `list` | 分页资料库列表 |
| `save` | 将 `ids` / `uris` 添加到资料库 |
| `remove` | 从资料库移除 `ids` / `uris` |

## 功能对比：免费 vs Premium

只读工具在免费账户上可用。任何改变播放或队列的操作都需要 Premium。

## 计划任务：Spotify + cron

由于 Spotify 工具是常规 Hermes 工具，在 Hermes 会话中运行的 cron 作业可以按任何计划触发播放。

### 早晨唤醒播放列表

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

## 故障排除

**`403 Forbidden — Player command failed: No active device found`** — 你需要在至少一个设备上运行 Spotify。打开 Spotify 应用，重试。

**`403 Forbidden — Premium required`** — 你使用的是免费账户，试图使用改变播放的操作。

**`204 No Content` on `get_currently_playing`** — 当前没有在任何设备上播放。这是 Spotify 的正常响应，不是错误。

**`INVALID_CLIENT: Invalid redirect URI`** — 你的 Spotify 应用设置中的 redirect URI 与 Hermes 使用的不匹配。默认值是 `http://127.0.0.1:43827/spotify/callback`。

**`429 Too Many Requests`** — Spotify 的速率限制。等待一分钟后重试。

**`401 Unauthorized` 持续出现** — 你的刷新令牌被撤销了（通常因为你从账户中移除了应用）。再次运行 `hermes auth spotify`。

## 文件位置

| 文件 | 内容 |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | 访问令牌、刷新令牌、过期时间、范围、重定向 URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID` |
