---
sidebar_position: 4
title: "チュートリアル: チーム向けTelegramアシスタント"
description: "コードのヘルプ、リサーチ、システム管理などをチーム全員で利用できるTelegramボットをセットアップする手順ガイド"
---

# チーム向けTelegramアシスタントをセットアップする

このチュートリアルでは、Hermes Agentを基盤とした、複数のチームメンバーが利用できるTelegramボットのセットアップ手順を説明します。最後まで進めると、チームはコード、リサーチ、システム管理、その他あらゆる用途で相談できる共有AIアシスタントを手に入れられます。しかもユーザーごとの認可によってセキュアに保護されています。

## 何を作るのか

次のようなTelegramボットです:

- **認可された任意のチームメンバー**がDMで相談できる — コードレビュー、リサーチ、シェルコマンド、デバッグ
- **あなたのサーバー上で動作**し、ツールへのフルアクセスを持つ — ターミナル、ファイル編集、ウェブ検索、コード実行
- **ユーザーごとのセッション** — 各人がそれぞれの会話コンテキストを持つ
- **デフォルトでセキュア** — 承認されたユーザーだけが操作でき、認可方法は2種類
- **スケジュールタスク** — 日次スタンドアップ、ヘルスチェック、リマインダーをチームチャンネルに配信

---

## 前提条件

始める前に、次を用意してください:

- **Hermes Agentがインストール済み**のサーバーまたはVPS（ノートPCではなく — ボットは常時稼働し続ける必要があります）。まだの場合は[インストールガイド](/docs/getting-started/installation)に従ってください。
- あなた自身の（ボットオーナー用の）**Telegramアカウント**
- **LLMプロバイダーの設定** — 最低限、OpenAI、Anthropic、またはその他のサポート対象プロバイダーのAPIキーを `~/.hermes/.env` に設定

:::tip
月額5ドルのVPSでもゲートウェイを動かすには十分です。Hermes自体は軽量で — コストがかかるのはLLMのAPI呼び出しであり、それはリモートで発生します。
:::

---

## ステップ1: Telegramボットを作成する

すべてのTelegramボットは **@BotFather** から始まります — これはボットを作成するためのTelegram公式ボットです。

1. **Telegramを開き**、`@BotFather` を検索するか、[t.me/BotFather](https://t.me/BotFather) にアクセスします

2. **`/newbot` を送信** — BotFatherが2つのことを尋ねてきます:
   - **表示名** — ユーザーに表示される名前（例: `Team Hermes Assistant`）
   - **ユーザー名** — `bot` で終わる必要があります（例: `myteam_hermes_bot`）

3. **ボットトークンをコピー** — BotFatherは次のような返信をします:
   ```
   Use this token to access the HTTP API:
   7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...
   ```
   このトークンは次のステップで必要になるので保存しておきます。

4. **説明を設定**（任意ですが推奨）:
   ```
   /setdescription
   ```
   ボットを選択し、次のような文言を入力します:
   ```
   Team AI assistant powered by Hermes Agent. DM me for help with code, research, debugging, and more.
   ```

5. **ボットコマンドを設定**（任意 — ユーザーにコマンドメニューを提供します）:
   ```
   /setcommands
   ```
   ボットを選択し、次を貼り付けます:
   ```
   new - Start a fresh conversation
   model - Show or change the AI model
   status - Show session info
   help - Show available commands
   stop - Stop the current task
   ```

:::warning
ボットトークンは秘密に保ってください。トークンを持つ者は誰でもボットを操作できます。漏洩した場合は、BotFatherで `/revoke` を使って新しいトークンを生成してください。
:::

---

## ステップ2: ゲートウェイを設定する

2つの選択肢があります: 対話型セットアップウィザード（推奨）か、手動設定です。

### 選択肢A: 対話型セットアップ（推奨）

```bash
hermes gateway setup
```

矢印キーで選択しながら、すべてを順に設定できます。**Telegram** を選び、ボットトークンを貼り付け、プロンプトが表示されたらユーザーIDを入力します。

### 選択肢B: 手動設定

次の行を `~/.hermes/.env` に追加します:

```bash
# BotFatherから取得したTelegramボットトークン
TELEGRAM_BOT_TOKEN=7123456789:AAH1bGciOiJSUzI1NiIsInR5cCI6Ikp...

# あなたのTelegramユーザーID（数値）
TELEGRAM_ALLOWED_USERS=123456789
```

### ユーザーIDを調べる

TelegramのユーザーIDは数値です（ユーザー名ではありません）。調べる方法:

1. Telegramで [@userinfobot](https://t.me/userinfobot) にメッセージを送る
2. すぐに数値のユーザーIDが返信されます
3. その番号を `TELEGRAM_ALLOWED_USERS` にコピーします

:::info
TelegramのユーザーIDは `123456789` のような永続的な番号です。変更可能な `@username` とは異なります。許可リストには必ず数値のIDを使ってください。
:::

---

## ステップ3: ゲートウェイを起動する

### 簡易テスト

まずはすべて正しく動作するか確認するため、ゲートウェイをフォアグラウンドで実行します:

```bash
hermes gateway
```

次のような出力が表示されるはずです:

```
[Gateway] Starting Hermes Gateway...
[Gateway] Telegram adapter connected
[Gateway] Cron scheduler started (tick every 60s)
```

Telegramを開いてボットを見つけ、メッセージを送信します。返信があれば成功です。`Ctrl+C` で停止します。

### 本番: サービスとしてインストール

再起動後も生き残る永続的なデプロイには:

```bash
hermes gateway install
sudo hermes gateway install --system   # Linuxのみ: 起動時に立ち上がるシステムサービス
```

これによりバックグラウンドサービスが作成されます: Linuxではデフォルトでユーザーレベルの **systemd** サービス、macOSでは **launchd** サービス、`--system` を渡した場合はLinuxの起動時システムサービスになります。

```bash
# Linux — デフォルトのユーザーサービスを管理
hermes gateway start
hermes gateway stop
hermes gateway status

# ライブログを表示
journalctl --user -u hermes-gateway -f

# SSHログアウト後も実行し続ける
sudo loginctl enable-linger $USER

# Linuxサーバー — システムサービス用の明示的なコマンド
sudo hermes gateway start --system
sudo hermes gateway status --system
journalctl -u hermes-gateway -f
```

```bash
# macOS — サービスを管理
hermes gateway start
hermes gateway stop
tail -f ~/.hermes/logs/gateway.log
```

:::tip macOSのPATH
launchdのplistは、ゲートウェイのサブプロセスがNode.jsやffmpegといったツールを見つけられるように、インストール時のシェルPATHを取り込みます。後から新しいツールをインストールした場合は、`hermes gateway install` を再実行してplistを更新してください。
:::

### 稼働を確認する

```bash
hermes gateway status
```

その後、Telegramのボットにテストメッセージを送信します。数秒以内に応答が返ってくるはずです。

---

## ステップ4: チームアクセスをセットアップする

それでは、チームメイトにアクセスを付与しましょう。方法は2つあります。

### 方法A: 静的な許可リスト

各チームメンバーのTelegramユーザーIDを集め（[@userinfobot](https://t.me/userinfobot) にメッセージを送ってもらいます）、カンマ区切りのリストとして追加します:

```bash
# ~/.hermes/.env 内
TELEGRAM_ALLOWED_USERS=123456789,987654321,555555555
```

変更後はゲートウェイを再起動します:

```bash
hermes gateway stop && hermes gateway start
```

### 方法B: DMペアリング（チームには推奨）

DMペアリングはより柔軟です — ユーザーIDを事前に集める必要がありません。仕組みは次のとおりです:

1. **チームメイトがボットにDMを送る** — 許可リストにないため、ボットはワンタイムのペアリングコードを返信します:
   ```
   🔐 Pairing code: XKGH5N7P
   Send this code to the bot owner for approval.
   ```

2. **チームメイトがあなたにそのコードを送る**（Slack、メール、対面など任意の手段で）

3. **あなたがサーバー上で承認する**:
   ```bash
   hermes pairing approve telegram XKGH5N7P
   ```

4. **承認完了** — ボットはすぐにそのユーザーのメッセージへの応答を開始します

**ペアリング済みユーザーの管理:**

```bash
# 保留中および承認済みの全ユーザーを表示
hermes pairing list

# 特定ユーザーのアクセスを取り消す
hermes pairing revoke telegram 987654321

# 期限切れの保留中コードをクリア
hermes pairing clear-pending
```

:::tip
DMペアリングは、新しいユーザーを追加するときにゲートウェイを再起動する必要がないため、チームに最適です。承認は即座に有効になります。
:::

### セキュリティ上の考慮事項

- ターミナルアクセスを持つボットでは、**`GATEWAY_ALLOW_ALL_USERS=true` を決して設定しないでください** — ボットを見つけた者なら誰でもサーバー上でコマンドを実行できてしまいます
- ペアリングコードは **1時間** で期限切れになり、暗号学的乱数を使用します
- レート制限により総当たり攻撃を防ぎます: ユーザーごとに10分あたり1リクエスト、プラットフォームごとに最大3つの保留中コード
- 承認の試行が5回失敗すると、そのプラットフォームは1時間のロックアウト状態になります
- すべてのペアリングデータは `chmod 0600` の権限で保存されます

---

## ステップ5: ボットを設定する

### ホームチャンネルを設定する

**ホームチャンネル**は、ボットがcronジョブの結果やプロアクティブなメッセージを配信する場所です。これがないと、スケジュールタスクは出力の送り先を持ちません。

**選択肢1:** ボットがメンバーとなっている任意のTelegramグループまたはチャットで `/sethome` コマンドを使います。

**選択肢2:** `~/.hermes/.env` で手動設定します:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="Team Updates"
```

チャンネルIDを調べるには、グループに [@userinfobot](https://t.me/userinfobot) を追加します — そのグループのチャットIDが報告されます。

### ツール進捗の表示を設定する

ボットがツールを使う際に表示する詳細度を制御します。`~/.hermes/config.yaml` で:

```yaml
display:
  tool_progress: new    # off | new | all | verbose
```

| モード | 表示される内容 |
|------|-------------|
| `off` | 応答のみ — ツールの動作は表示されない |
| `new` | 新しいツール呼び出しごとに簡潔なステータスを表示（メッセージング用途に推奨） |
| `all` | すべてのツール呼び出しを詳細付きで表示 |
| `verbose` | コマンド結果を含むツール出力をすべて表示 |

ユーザーはチャット内で `/verbose` コマンドを使い、セッションごとにこれを変更することもできます。

### SOUL.mdでパーソナリティをセットアップする

`~/.hermes/SOUL.md` を編集して、ボットのコミュニケーション方法をカスタマイズします:

詳しいガイドは [HermesでSOUL.mdを使う](/docs/guides/use-soul-with-hermes) を参照してください。

```markdown
# Soul
You are a helpful team assistant. Be concise and technical.
Use code blocks for any code. Skip pleasantries — the team
values directness. When debugging, always ask for error logs
before guessing at solutions.
```

### プロジェクトコンテキストを追加する

チームが特定のプロジェクトに取り組んでいる場合は、ボットがスタックを把握できるようにコンテキストファイルを作成します:

```markdown
<!-- ~/.hermes/AGENTS.md -->
# Team Context
- We use Python 3.12 with FastAPI and SQLAlchemy
- Frontend is React with TypeScript
- CI/CD runs on GitHub Actions
- Production deploys to AWS ECS
- Always suggest writing tests for new code
```

:::info
コンテキストファイルは、すべてのセッションのシステムプロンプトに注入されます。簡潔に保ってください — 1文字ごとにトークン予算を消費します。
:::

---

## ステップ6: スケジュールタスクをセットアップする

ゲートウェイを稼働させた状態で、結果をチームチャンネルに配信する定期タスクをスケジュールできます。

### 日次スタンドアップサマリー

Telegramでボットにメッセージを送ります:

```
Every weekday at 9am, check the GitHub repository at
github.com/myorg/myproject for:
1. Pull requests opened/merged in the last 24 hours
2. Issues created or closed
3. Any CI/CD failures on the main branch
Format as a brief standup-style summary.
```

エージェントは自動的にcronジョブを作成し、依頼したチャット（またはホームチャンネル）に結果を配信します。

### サーバーヘルスチェック

```
Every 6 hours, check disk usage with 'df -h', memory with 'free -h',
and Docker container status with 'docker ps'. Report anything unusual —
partitions above 80%, containers that have restarted, or high memory usage.
```

### スケジュールタスクを管理する

```bash
# CLIから
hermes cron list          # スケジュール済みジョブをすべて表示
hermes cron status        # スケジューラが稼働中か確認

# Telegramチャットから
/cron list                # ジョブを表示
/cron remove <job_id>     # ジョブを削除
```

:::warning
cronジョブのプロンプトは、過去の会話の記憶を持たない完全に新しいセッションで実行されます。各プロンプトには、エージェントが必要とする **すべての** コンテキスト — ファイルパス、URL、サーバーアドレス、明確な指示 — が含まれていることを確認してください。
:::

---

## 本番運用のヒント

### 安全のためにDockerを使う

共有のチームボットでは、エージェントのコマンドがホスト上ではなくコンテナ内で実行されるよう、ターミナルバックエンドとしてDockerを使います:

```bash
# ~/.hermes/.env 内
TERMINAL_BACKEND=docker
TERMINAL_DOCKER_IMAGE=nikolaik/python-nodejs:python3.11-nodejs20
```

または `~/.hermes/config.yaml` で:

```yaml
terminal:
  backend: docker
  container_cpu: 1
  container_memory: 5120
  container_persistent: true
```

こうすれば、誰かがボットに破壊的な操作を依頼したとしても、ホストシステムは保護されます。

### ゲートウェイを監視する

```bash
# ゲートウェイが稼働中か確認
hermes gateway status

# ライブログを監視（Linux）
journalctl --user -u hermes-gateway -f

# ライブログを監視（macOS）
tail -f ~/.hermes/logs/gateway.log
```

### Hermesを最新に保つ

Telegramからボットに `/update` を送ると、最新版を取得して再起動します。あるいはサーバーから:

```bash
hermes update
hermes gateway stop && hermes gateway start
```

### ログの場所

| 対象 | 場所 |
|------|------|
| ゲートウェイログ | `journalctl --user -u hermes-gateway`（Linux）または `~/.hermes/logs/gateway.log`（macOS） |
| cronジョブの出力 | `~/.hermes/cron/output/{job_id}/{timestamp}.md` |
| cronジョブの定義 | `~/.hermes/cron/jobs.json` |
| ペアリングデータ | `~/.hermes/pairing/` |
| セッション履歴 | `~/.hermes/sessions/` |

---

## さらに進めるには

これで動作するチーム向けTelegramアシスタントが手に入りました。次のステップをいくつか紹介します:

- **[セキュリティガイド](/docs/user-guide/security)** — 認可、コンテナの分離、コマンド承認の詳細
- **[メッセージングゲートウェイ](/docs/user-guide/messaging)** — ゲートウェイのアーキテクチャ、セッション管理、チャットコマンドの完全リファレンス
- **[Telegramセットアップ](/docs/user-guide/messaging/telegram)** — ボイスメッセージやTTSを含むプラットフォーム固有の詳細
- **[スケジュールタスク](/docs/user-guide/features/cron)** — 配信オプションやcron式を使った高度なcronスケジューリング
- **[コンテキストファイル](/docs/user-guide/features/context-files)** — プロジェクト知識のためのAGENTS.md、SOUL.md、.cursorrules
- **[パーソナリティ](/docs/user-guide/features/personality)** — 組み込みのパーソナリティプリセットとカスタムペルソナの定義
- **プラットフォームを追加する** — 同じゲートウェイで [Discord](/docs/user-guide/messaging/discord)、[Slack](/docs/user-guide/messaging/slack)、[WhatsApp](/docs/user-guide/messaging/whatsapp) を同時に動かせます

---

*質問や問題がありますか？ GitHubでissueを開いてください — コントリビューションを歓迎します。*
