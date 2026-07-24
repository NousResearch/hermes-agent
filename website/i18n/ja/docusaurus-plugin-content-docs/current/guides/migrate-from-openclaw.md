---
sidebar_position: 10
title: "OpenClaw から移行する"
description: "OpenClaw / Clawdbot のセットアップを Hermes Agent に移行するための完全ガイド — 何が移行されるか、設定がどうマッピングされるか、移行後に何を確認すべきか。"
---

# OpenClaw から移行する

`hermes claw migrate` は、OpenClaw（または旧 Clawdbot/Moldbot）のセットアップを Hermes にインポートします。このガイドでは、何が移行されるか、設定キーのマッピング、移行後に何を確認すべきかを正確に説明します。

## クイックスタート

```bash
# プレビューしてから移行（常に最初にプレビューを表示し、確認を求めます）
hermes claw migrate

# プレビューのみ、変更なし
hermes claw migrate --dry-run

# APIキーを含む完全な移行、確認をスキップ
hermes claw migrate --preset full --migrate-secrets --yes
```

移行は、変更を加える前に、インポートされる内容の完全なプレビューを常に表示します。リストを確認してから、続行を確定してください。

デフォルトでは `~/.openclaw/` から読み込みます。旧 `~/.clawdbot/` または `~/.moltbot/` ディレクトリは自動的に検出されます。旧設定ファイル名（`clawdbot.json`、`moltbot.json`）も同様です。

## オプション

| オプション | 説明 |
|--------|-------------|
| `--dry-run` | プレビューのみ — 移行される内容を表示した後に停止します。 |
| `--preset <name>` | `full`（互換性のあるすべての設定）または `user-data`（インフラ設定を除く）。どちらのプリセットもデフォルトではシークレットをインポートしません。`--migrate-secrets` を明示的に渡してください。 |
| `--overwrite` | 競合時に既存の Hermes ファイルを上書き（デフォルト: プランに競合がある場合は適用を拒否）。 |
| `--migrate-secrets` | APIキーを含めます。`--preset full` の下でも必須です。どのプリセットも黙ってシークレットをインポートしません。 |
| `--no-backup` | 移行前の `~/.hermes/` の zip スナップショットをスキップ（デフォルトでは、適用前に単一の復元ポイントアーカイブが `~/.hermes/backups/pre-migration-*.zip` の下に書き込まれます。`hermes import` で復元可能）。 |
| `--source <path>` | カスタムの OpenClaw ディレクトリ。 |
| `--workspace-target <path>` | `AGENTS.md` を配置する場所。 |
| `--skill-conflict <mode>` | `skip`（デフォルト）、`overwrite`、または `rename`。 |
| `--yes` | プレビュー後の確認プロンプトをスキップ。 |

## 何が移行されるか

### ペルソナ、メモリ、指示

| 内容 | OpenClaw のソース | Hermes の宛先 | 備考 |
|------|----------------|-------------------|-------|
| ペルソナ | `workspace/SOUL.md` | `~/.hermes/SOUL.md` | そのままコピー |
| ワークスペースの指示 | `workspace/AGENTS.md` | `--workspace-target` 内の `AGENTS.md` | `--workspace-target` フラグが必要 |
| 長期メモリ | `workspace/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | エントリにパースされ、既存とマージ、重複排除。`§` 区切り文字を使用。 |
| ユーザープロファイル | `workspace/USER.md` | `~/.hermes/memories/USER.md` | メモリと同じエントリマージロジック。 |
| 日次メモリファイル | `workspace/memory/*.md` | `~/.hermes/memories/MEMORY.md` | すべての日次ファイルがメインメモリにマージされます。 |

ワークスペースファイルは、フォールバックパスとして `workspace.default/` と `workspace-main/` でも確認されます（OpenClaw は最近のバージョンで `workspace/` を `workspace-main/` にリネームし、マルチエージェント構成では `workspace-{agentId}` を使用します）。

### スキル（4つのソース）

| ソース | OpenClaw の場所 | Hermes の宛先 |
|--------|------------------|-------------------|
| ワークスペーススキル | `workspace/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 管理／共有スキル | `~/.openclaw/skills/` | `~/.hermes/skills/openclaw-imports/` |
| 個人のクロスプロジェクト | `~/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |
| プロジェクトレベルの共有 | `workspace/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |

スキルの競合は `--skill-conflict` で処理されます。`skip` は既存の Hermes スキルを残し、`overwrite` は置き換え、`rename` は `-imported` のコピーを作成します。

### モデルとプロバイダーの設定

| 内容 | OpenClaw の設定パス | Hermes の宛先 | 備考 |
|------|---------------------|-------------------|-------|
| デフォルトモデル | `agents.defaults.model` | `config.yaml` → `model` | 文字列または `{primary, fallbacks}` オブジェクト |
| カスタムプロバイダー | `models.providers.*` | `config.yaml` → `custom_providers` | `baseUrl`、`apiType`/`api` をマッピング — 短形式（"openai"、"anthropic"）とハイフン形式（"openai-completions"、"anthropic-messages"、"google-generative-ai"）の両方の値を処理 |
| プロバイダーの APIキー | `models.providers.*.apiKey` | `~/.hermes/.env` | `--migrate-secrets` が必要。下記の [APIキーの解決](#api-key-resolution) を参照。 |

### エージェントの動作

| 内容 | OpenClaw の設定パス | Hermes の設定パス | マッピング |
|------|---------------------|-------------------|---------|
| 最大ターン数 | `agents.defaults.timeoutSeconds` | `agent.max_turns` | `timeoutSeconds / 10`、上限200 |
| 詳細モード | `agents.defaults.verboseDefault` | `agent.verbose` | "off" / "on" / "full" |
| 推論の労力 | `agents.defaults.thinkingDefault` | `agent.reasoning_effort` | "always"/"high"/"xhigh" → "high"、"auto"/"medium"/"adaptive" → "medium"、"off"/"low"/"none"/"minimal" → "low" |
| 圧縮 | `agents.defaults.compaction.mode` | `compression.enabled` | "off" → false、それ以外 → true |
| 圧縮モデル | `agents.defaults.compaction.model` | `compression.summary_model` | 文字列をそのままコピー |
| 人間らしい遅延 | `agents.defaults.humanDelay.mode` | `human_delay.mode` | "natural" / "custom" / "off" |
| 人間らしい遅延のタイミング | `agents.defaults.humanDelay.minMs` / `.maxMs` | `human_delay.min_ms` / `.max_ms` | そのままコピー |
| タイムゾーン | `agents.defaults.userTimezone` | `timezone` | 文字列をそのままコピー |
| 実行タイムアウト | `tools.exec.timeoutSec` | `terminal.timeout` | そのままコピー（フィールドは `timeout` ではなく `timeoutSec`） |
| Docker サンドボックス | `agents.defaults.sandbox.backend` | `terminal.backend` | "docker" → "docker" |
| Docker イメージ | `agents.defaults.sandbox.docker.image` | `terminal.docker_image` | そのままコピー |

### セッションリセットポリシー

| OpenClaw の設定パス | Hermes の設定パス | 備考 |
|---------------------|-------------------|-------|
| `session.reset.mode` | `session_reset.mode` | "daily"、"idle"、または両方 |
| `session.reset.atHour` | `session_reset.at_hour` | 日次リセットの時刻（0〜23） |
| `session.reset.idleMinutes` | `session_reset.idle_minutes` | 非アクティブの分数 |

注: OpenClaw には `session.resetTriggers`（`["daily", "idle"]` のような単純な文字列配列）もあります。構造化された `session.reset` が存在しない場合、移行は `resetTriggers` からの推論にフォールバックします。

### MCP サーバー

| OpenClaw のフィールド | Hermes のフィールド | 備考 |
|----------------|-------------|-------|
| `mcp.servers.*.command` | `mcp_servers.*.command` | Stdio トランスポート |
| `mcp.servers.*.args` | `mcp_servers.*.args` | |
| `mcp.servers.*.env` | `mcp_servers.*.env` | |
| `mcp.servers.*.cwd` | `mcp_servers.*.cwd` | |
| `mcp.servers.*.url` | `mcp_servers.*.url` | HTTP/SSE トランスポート |
| `mcp.servers.*.tools.include` | `mcp_servers.*.tools.include` | ツールのフィルタリング |
| `mcp.servers.*.tools.exclude` | `mcp_servers.*.tools.exclude` | |

### TTS（テキスト読み上げ）

TTS 設定は、次の優先順位で **2つの** OpenClaw の設定場所から読み込まれます。

1. `messages.tts.providers.{provider}.*`（正規の場所）
2. トップレベルの `talk.providers.{provider}.*`（フォールバック）
3. 旧式のフラットキー `messages.tts.{provider}.*`（最も古い形式）

| 内容 | Hermes の宛先 |
|------|-------------------|
| プロバイダー名 | `config.yaml` → `tts.provider` |
| ElevenLabs ボイスID | `config.yaml` → `tts.elevenlabs.voice_id` |
| ElevenLabs モデルID | `config.yaml` → `tts.elevenlabs.model_id` |
| OpenAI モデル | `config.yaml` → `tts.openai.model` |
| OpenAI ボイス | `config.yaml` → `tts.openai.voice` |
| Edge TTS ボイス | `config.yaml` → `tts.edge.voice`（OpenClaw は "edge" を "microsoft" にリネーム — 両方とも認識されます） |
| TTS アセット | `~/.hermes/tts/`（ファイルコピー） |

### メッセージングプラットフォーム

| プラットフォーム | OpenClaw の設定パス | Hermes の `.env` 変数 | 備考 |
|----------|---------------------|----------------------|-------|
| Telegram | `channels.telegram.botToken` または `.accounts.default.botToken` | `TELEGRAM_BOT_TOKEN` | トークンは文字列または [SecretRef](#secretref-handling)。フラットレイアウトと accounts レイアウトの両方をサポート。 |
| Telegram | `credentials/telegram-default-allowFrom.json` | `TELEGRAM_ALLOWED_USERS` | `allowFrom[]` 配列からカンマ結合 |
| Discord | `channels.discord.token` または `.accounts.default.token` | `DISCORD_BOT_TOKEN` | |
| Discord | `channels.discord.allowFrom` または `.accounts.default.allowFrom` | `DISCORD_ALLOWED_USERS` | |
| Slack | `channels.slack.botToken` または `.accounts.default.botToken` | `SLACK_BOT_TOKEN` | |
| Slack | `channels.slack.appToken` または `.accounts.default.appToken` | `SLACK_APP_TOKEN` | |
| Slack | `channels.slack.allowFrom` または `.accounts.default.allowFrom` | `SLACK_ALLOWED_USERS` | |
| WhatsApp | `channels.whatsapp.allowFrom` または `.accounts.default.allowFrom` | `WHATSAPP_ALLOWED_USERS` | Baileys QR ペアリングによる認証 — 移行後に再ペアリングが必要 |
| Signal | `channels.signal.account` または `.accounts.default.account` | `SIGNAL_ACCOUNT` | |
| Signal | `channels.signal.httpUrl` または `.accounts.default.httpUrl` | `SIGNAL_HTTP_URL` | |
| Signal | `channels.signal.allowFrom` または `.accounts.default.allowFrom` | `SIGNAL_ALLOWED_USERS` | |
| Matrix | `channels.matrix.accessToken` または `.accounts.default.accessToken` | `MATRIX_ACCESS_TOKEN` | `botToken` ではなく `accessToken` を使用 |
| Mattermost | `channels.mattermost.botToken` または `.accounts.default.botToken` | `MATTERMOST_BOT_TOKEN` | |

### その他の設定

| 内容 | OpenClaw のパス | Hermes のパス | 備考 |
|------|-------------|-------------|-------|
| 承認モード | `approvals.exec.mode` | `config.yaml` → `approvals.mode` | "auto"→"off"、"always"→"manual"、"smart"→"smart" |
| コマンド許可リスト | `exec-approvals.json` | `config.yaml` → `command_allowlist` | パターンをマージし重複排除 |
| ブラウザ CDP URL | `browser.cdpUrl` | `config.yaml` → `browser.cdp_url` | |
| ブラウザのヘッドレス | `browser.headless` | `config.yaml` → `browser.headless` | |
| Brave 検索キー | `tools.web.search.brave.apiKey` | `.env` → `BRAVE_API_KEY` | `--migrate-secrets` が必要 |
| ゲートウェイ認証トークン | `gateway.auth.token` | `.env` → `HERMES_GATEWAY_TOKEN` | `--migrate-secrets` が必要 |
| 作業ディレクトリ | `agents.defaults.workspace` | `.env` → `MESSAGING_CWD` | |

### アーカイブ（直接の Hermes 相当物なし）

これらは手動レビュー用に `~/.hermes/migration/openclaw/<timestamp>/archive/` に保存されます。

| 内容 | アーカイブファイル | Hermes での再現方法 |
|------|-------------|--------------------------|
| `IDENTITY.md` | `archive/workspace/IDENTITY.md` | `SOUL.md` にマージ |
| `TOOLS.md` | `archive/workspace/TOOLS.md` | Hermes には組み込みのツール指示があります |
| `HEARTBEAT.md` | `archive/workspace/HEARTBEAT.md` | 定期的なタスクには cronジョブを使用 |
| `BOOTSTRAP.md` | `archive/workspace/BOOTSTRAP.md` | コンテキストファイルまたはスキルを使用 |
| cronジョブ | `archive/cron-config.json` | `hermes cron create` で再作成 |
| プラグイン | `archive/plugins-config.json` | [プラグインガイド](/docs/user-guide/features/hooks) を参照 |
| フック／Webhook | `archive/hooks-config.json` | `hermes webhook` またはゲートウェイフックを使用 |
| メモリバックエンド | `archive/memory-backend-config.json` | `hermes honcho` で設定 |
| スキルレジストリ | `archive/skills-registry-config.json` | `hermes skills config` を使用 |
| UI／アイデンティティ | `archive/ui-identity-config.json` | `/skin` コマンドを使用 |
| ロギング | `archive/logging-diagnostics-config.json` | `config.yaml` の logging セクションで設定 |
| マルチエージェントリスト | `archive/agents-list.json` | Hermes プロファイルを使用 |
| チャネルバインディング | `archive/bindings.json` | プラットフォームごとに手動セットアップ |
| 複雑なチャネル | `archive/channels-deep-config.json` | 手動でプラットフォーム設定 |

## APIキーの解決 {#api-key-resolution}

`--migrate-secrets` が有効な場合、APIキーは優先順位に従って **4つのソース** から収集されます。

1. **設定値** — `openclaw.json` 内の `models.providers.*.apiKey` と TTS プロバイダーキー
2. **環境ファイル** — `~/.openclaw/.env`（`OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` などのキー）
3. **設定の env サブオブジェクト** — `openclaw.json` → `"env"` または `"env"."vars"`（一部の構成では別の `.env` ファイルではなくここにキーを保存します）
4. **認証プロファイル** — `~/.openclaw/agents/main/agent/auth-profiles.json`（エージェントごとの認証情報）

設定値が優先されます。後続の各ソースが、残りのギャップを埋めます。

### サポートされるキーのターゲット

`OPENROUTER_API_KEY`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`、`GEMINI_API_KEY`、`ZAI_API_KEY`、`MINIMAX_API_KEY`、`ELEVENLABS_API_KEY`、`TELEGRAM_BOT_TOKEN`、`VOICE_TOOLS_OPENAI_KEY`

この許可リストにないキーは決してコピーされません。

## SecretRef の処理 {#secretref-handling}

トークンと APIキーの OpenClaw 設定値は、3つの形式のいずれかになります。

```json
// プレーン文字列
"channels": { "telegram": { "botToken": "123456:ABC-DEF..." } }

// 環境変数テンプレート
"channels": { "telegram": { "botToken": "${TELEGRAM_BOT_TOKEN}" } }

// SecretRef オブジェクト
"channels": { "telegram": { "botToken": { "source": "env", "id": "TELEGRAM_BOT_TOKEN" } } }
```

移行はこれら3つの形式すべてを解決します。env テンプレートと `source: "env"` の SecretRef オブジェクトについては、`~/.openclaw/.env` と `openclaw.json` の env サブオブジェクトで値を検索します。`source: "file"` または `source: "exec"` の SecretRef オブジェクトは自動的に解決できません。移行はこれらについて警告し、それらの値は `hermes config set` を介して手動で Hermes に追加する必要があります。

## 移行後

1. **移行レポートを確認する** — 完了時に、移行済み、スキップ、競合した項目の数とともに出力されます。

2. **アーカイブされたファイルをレビューする** — `~/.hermes/migration/openclaw/<timestamp>/archive/` 内のものは手動での対応が必要です。

3. **新しいセッションを開始する** — インポートされたスキルとメモリエントリは、現在のセッションではなく新しいセッションで有効になります。

4. **APIキーを検証する** — `hermes status` を実行してプロバイダーの認証を確認します。

5. **メッセージングをテストする** — プラットフォームトークンを移行した場合は、ゲートウェイを再起動します: `systemctl --user restart hermes-gateway`

6. **セッションポリシーを確認する** — `hermes config get session_reset` が期待どおりか検証します。

7. **WhatsApp を再ペアリングする** — WhatsApp はトークン移行ではなく QRコードペアリング（Baileys）を使用します。`hermes whatsapp` を実行してペアリングします。

8. **アーカイブのクリーンアップ** — すべてが機能することを確認した後、`hermes claw cleanup` を実行して、残った OpenClaw ディレクトリを `.pre-migration/` にリネームします（状態の混乱を防ぎます）。

## トラブルシューティング

### 「OpenClaw directory not found」

移行は `~/.openclaw/`、次に `~/.clawdbot/`、次に `~/.moltbot/` を確認します。インストールが別の場所にある場合は、`--source /path/to/your/openclaw` を使用してください。

### 「No provider API keys found」

OpenClaw のバージョンによって、キーはいくつかの場所に保存されている可能性があります。`openclaw.json` 内の `models.providers.*.apiKey` にインライン、`~/.openclaw/.env`、`openclaw.json` の `"env"` サブオブジェクト、または `agents/main/agent/auth-profiles.json` です。移行はこれら4つすべてを確認します。キーが `source: "file"` または `source: "exec"` の SecretRef を使用している場合、自動的に解決できません。`hermes config set` を介して追加してください。

### 移行後にスキルが表示されない

インポートされたスキルは `~/.hermes/skills/openclaw-imports/` に配置されます。新しいセッションを開始して有効にするか、`/skills` を実行して読み込まれているか確認してください。

### TTS のボイスが移行されない

OpenClaw は TTS 設定を2か所に保存します。`messages.tts.providers.*` とトップレベルの `talk` 設定です。移行は両方を確認します。ボイスID が OpenClaw の UI を介して（別のパスに保存されて）設定された場合、手動で設定する必要があるかもしれません: `hermes config set tts.elevenlabs.voice_id YOUR_VOICE_ID`。
