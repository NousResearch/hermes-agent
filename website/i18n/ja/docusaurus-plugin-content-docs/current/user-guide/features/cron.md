---
sidebar_position: 5
title: "スケジュールタスク（Cron）"
description: "自然言語で自動タスクをスケジュールし、1つのcronツールで管理し、1つ以上のスキルを紐づける"
---

# スケジュールタスク（Cron）

自然言語またはcron式でタスクを自動実行するようにスケジュールできます。Hermesはcron管理を、個別のschedule/list/removeツールではなく、アクションスタイルの操作を持つ単一の`cronjob`ツールを通じて公開します。

## cronが今できること

cronジョブは次のことができます。

- ワンショットまたは繰り返しのタスクをスケジュールする
- ジョブの一時停止、再開、編集、トリガー、削除を行う
- ジョブに0個、1個、または複数のスキルを紐づける
- 結果を発信元のチャット、ローカルファイル、または設定済みのプラットフォームターゲットに配信する
- 通常の静的ツールリストを持つ新規エージェントセッションで実行する
- **no-agentモード**で実行する — スケジュールに沿って動くスクリプトであり、その標準出力はそのまま配信され、LLMは一切関与しません（後述の[no-agentモード](#no-agent-mode-script-only-jobs)のセクションを参照）

これらすべては`cronjob`ツールを通じてHermes自身からも利用できるため、平易な言葉で依頼するだけでジョブの作成、一時停止、編集、削除が可能です — CLIは不要です。

:::warning
cron実行中のセッションは、再帰的にさらにcronジョブを作成することはできません。Hermesは暴走するスケジューリングループを防ぐため、cron実行内ではcron管理ツールを無効化します。
:::

## スケジュールタスクの作成

### チャットで `/cron` を使う

```bash
/cron add 30m "Remind me to check the build"
/cron add "every 2h" "Check server status"
/cron add "every 1h" "Summarize new feed items" --skill blogwatcher
/cron add "every 1h" "Use both skills and combine the result" --skill blogwatcher --skill maps
```

### スタンドアロンCLIから

```bash
hermes cron create "every 2h" "Check server status"
hermes cron create "every 1h" "Summarize new feed items" --skill blogwatcher
hermes cron create "every 1h" "Use both skills and combine the result" \
  --skill blogwatcher \
  --skill maps \
  --name "Skill combo"
```

### 自然な会話を通じて

通常どおりHermesに依頼します。

```text
Every morning at 9am, check Hacker News for AI news and send me a summary on Telegram.
```

Hermesは内部で統一された`cronjob`ツールを使用します。

## スキルを使ったcronジョブ

cronジョブは、プロンプトを実行する前に1つ以上のスキルをロードできます。

### 単一スキル

```python
cronjob(
    action="create",
    skill="blogwatcher",
    prompt="Check the configured feeds and summarize anything new.",
    schedule="0 9 * * *",
    name="Morning feeds",
)
```

### 複数スキル

スキルは順番にロードされます。プロンプトは、それらのスキルの上に重ねられるタスク指示になります。

```python
cronjob(
    action="create",
    skills=["blogwatcher", "maps"],
    prompt="Look for new local events and interesting nearby places, then combine them into one short brief.",
    schedule="every 6h",
    name="Local brief",
)
```

これは、完全なスキルテキストをcronプロンプト自体に詰め込むことなく、スケジュールされたエージェントに再利用可能なワークフローを継承させたい場合に便利です。

## プロジェクトディレクトリ内でジョブを実行する

cronジョブはデフォルトで、いかなるリポジトリからも切り離されて実行されます — `AGENTS.md`、`CLAUDE.md`、`.cursorrules`はロードされず、ターミナル / ファイル / コード実行ツールはゲートウェイが起動した作業ディレクトリから動作します。これを変更するには、`--workdir`（CLI）または`workdir=`（ツール呼び出し）を渡します。

```bash
# スタンドアロンCLI（scheduleとpromptは位置引数）
hermes cron create "every 1d at 09:00" \
  "Audit open PRs, summarize CI health, and post to #eng" \
  --workdir /home/me/projects/acme
```

```python
# チャットから、cronjobツール経由で
cronjob(
    action="create",
    schedule="every 1d at 09:00",
    workdir="/home/me/projects/acme",
    prompt="Audit open PRs, summarize CI health, and post to #eng",
)
```

`workdir`が設定されている場合は次のようになります。

- そのディレクトリの`AGENTS.md`、`CLAUDE.md`、`.cursorrules`がシステムプロンプトに注入されます（対話型CLIと同じ検出順序）
- `terminal`、`read_file`、`write_file`、`patch`、`search_files`、`execute_code`はすべて、そのディレクトリを作業ディレクトリとして使用します（`TERMINAL_CWD`経由）
- パスは存在する絶対ディレクトリでなければなりません — 相対パスや存在しないディレクトリは作成 / 更新時に拒否されます
- 編集時に`--workdir ""`（またはツール経由で`workdir=""`）を渡すと、これをクリアして以前の動作に戻せます

:::note Serialization
`workdir`を持つジョブは、並列プールではなく、スケジューラのtick上で逐次実行されます。これは意図的なものです — `TERMINAL_CWD`はプロセスグローバルであるため、2つのworkdirジョブが同時に実行されると互いのcwdを破壊してしまいます。workdirを持たないジョブは、従来どおり並列で実行されます。
:::

## ジョブの編集

ジョブを変更するためだけに、削除して再作成する必要はありません。

### チャット

```bash
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Use the revised task"
/cron edit <job_id> --skill blogwatcher --skill maps
/cron edit <job_id> --remove-skill blogwatcher
/cron edit <job_id> --clear-skills
```

### スタンドアロンCLI

```bash
hermes cron edit <job_id> --schedule "every 4h"
hermes cron edit <job_id> --prompt "Use the revised task"
hermes cron edit <job_id> --skill blogwatcher --skill maps
hermes cron edit <job_id> --add-skill maps
hermes cron edit <job_id> --remove-skill blogwatcher
hermes cron edit <job_id> --clear-skills
```

補足:

- `--skill`を繰り返すと、ジョブに紐づいたスキルリストが置き換えられます
- `--add-skill`は既存のリストを置き換えずに追加します
- `--remove-skill`は特定の紐づいたスキルを削除します
- `--clear-skills`は紐づいたスキルをすべて削除します

## ライフサイクルアクション

cronジョブには、単なる作成/削除よりも充実したライフサイクルが備わっています。

### チャット

```bash
/cron list
/cron pause <job_id>
/cron resume <job_id>
/cron run <job_id>
/cron remove <job_id>
```

### スタンドアロンCLI

```bash
hermes cron list
hermes cron pause <job_id>
hermes cron resume <job_id>
hermes cron run <job_id>
hermes cron remove <job_id>
hermes cron status
hermes cron tick
```

各アクションの動作:

- `pause` — ジョブを残したままスケジューリングを停止する
- `resume` — ジョブを再有効化し、次回の将来の実行を計算する
- `run` — 次のスケジューラtickでジョブをトリガーする
- `remove` — 完全に削除する

## 仕組み

**cronの実行はゲートウェイデーモンが処理します。** ゲートウェイは60秒ごとにスケジューラをtickさせ、実行予定のジョブを分離されたエージェントセッションで実行します。

```bash
hermes gateway install     # ユーザーサービスとしてインストール
sudo hermes gateway install --system   # Linux: サーバー向けのブート時システムサービス
hermes gateway             # またはフォアグラウンドで実行

hermes cron list
hermes cron status
```

### ゲートウェイスケジューラの動作

各tickでHermesは次のことを行います。

1. `~/.hermes/cron/jobs.json`からジョブをロードする
2. `next_run_at`を現在時刻と照合する
3. 実行予定の各ジョブに対して新規の`AIAgent`セッションを開始する
4. オプションで、その新規セッションに1つ以上の紐づいたスキルを注入する
5. プロンプトを完了まで実行する
6. 最終応答を配信する
7. 実行メタデータと次回スケジュール時刻を更新する

`~/.hermes/cron/.tick.lock`のファイルロックにより、重複するスケジューラtickが同じジョブバッチを二重実行することを防ぎます。

## 配信オプション

ジョブをスケジュールする際に、出力先を指定します。

| オプション | 説明 | 例 |
|--------|-------------|---------|
| `"origin"` | ジョブが作成された場所に返す | メッセージングプラットフォームでのデフォルト |
| `"local"` | ローカルファイルにのみ保存（`~/.hermes/cron/output/`） | CLIでのデフォルト |
| `"telegram"` | Telegramのホームチャンネル | `TELEGRAM_HOME_CHANNEL`を使用 |
| `"telegram:123456"` | IDで指定した特定のTelegramチャット | 直接配信 |
| `"telegram:-100123:17585"` | 特定のTelegramトピック | `chat_id:thread_id`形式 |
| `"discord"` | Discordのホームチャンネル | `DISCORD_HOME_CHANNEL`を使用 |
| `"discord:#engineering"` | 特定のDiscordチャンネル | チャンネル名で指定 |
| `"slack"` | Slackのホームチャンネル | |
| `"whatsapp"` | WhatsAppのホーム | |
| `"signal"` | Signal | |
| `"matrix"` | Matrixのホームルーム | |
| `"mattermost"` | Mattermostのホームチャンネル | |
| `"email"` | Email | |
| `"sms"` | Twilio経由のSMS | |
| `"homeassistant"` | Home Assistant | |
| `"dingtalk"` | DingTalk | |
| `"feishu"` | Feishu/Lark | |
| `"wecom"` | WeCom | |
| `"weixin"` | Weixin（WeChat） | |
| `"bluebubbles"` | BlueBubbles（iMessage） | |
| `"qqbot"` | QQ Bot（Tencent QQ） | |
| `"all"` | 接続済みのすべてのホームチャンネルにファンアウトする | 発火時に解決 |
| `"telegram,discord"` | 特定のチャンネル群にファンアウトする | カンマ区切りリスト |
| `"origin,all"` | 発信元に**加えて**、接続済みの他のすべてのチャンネルに配信する | 任意のトークンを組み合わせる |

エージェントの最終応答は自動的に配信されます。cronプロンプト内で`send_message`を呼び出す必要はありません。

### ルーティングの意図（`all`）

`all`を使うと、設定済みのすべてのメッセージングチャンネルに対して、名前で列挙することなく1つのcronジョブを送れます。これは**発火時に解決される**ため、Telegramを接続する前に作成されたジョブでも、`TELEGRAM_HOME_CHANNEL`を設定した後の次のtickでTelegramを拾います。

セマンティクス: `all`はホームチャンネルが設定されたすべてのプラットフォームに展開されます。0個でも問題ありません。その場合、ジョブは単に配信ターゲットを生成せず、上流で配信失敗として記録されます。

`all`は明示的なターゲットと組み合わせられます。`origin,all`は発信元のチャット*に加えて*接続済みの他のすべてのホームチャンネルに配信し、`(platform, chat_id, thread_id)`で重複を排除します。

### 応答のラッピング

デフォルトでは、配信されるcron出力は、受信者がスケジュールタスク由来であると分かるように、ヘッダーとフッターでラップされます。

```
Cronjob Response: Morning feeds
-------------

<agent output here>

Note: The agent cannot see this message, and therefore cannot respond to it.
```

ラッパーなしで生のエージェント出力を配信するには、`cron.wrap_response`を`false`に設定します。

```yaml
# ~/.hermes/config.yaml
cron:
  wrap_response: false
```

### サイレント抑制

エージェントの最終応答が`[SILENT]`で始まる場合、配信は完全に抑制されます。出力は監査用にローカル（`~/.hermes/cron/output/`内）に保存されますが、配信ターゲットへのメッセージは送信されません。

これは、問題があるときのみ報告すべき監視ジョブに便利です。

```text
Check if nginx is running. If everything is healthy, respond with only [SILENT].
Otherwise, report the issue.
```

失敗したジョブは`[SILENT]`マーカーに関わらず常に配信されます — サイレント化できるのは成功した実行だけです。

## スクリプトのタイムアウト

実行前スクリプト（`script`パラメータで紐づけたもの）には、デフォルトで120秒のタイムアウトがあります。スクリプトにより長い時間が必要な場合 — 例えば、ボットのようなタイミングパターンを避けるためにランダムな遅延を含める場合 — この値を増やせます。

```yaml
# ~/.hermes/config.yaml
cron:
  script_timeout_seconds: 300   # 5分
```

または`HERMES_CRON_SCRIPT_TIMEOUT`環境変数を設定します。解決順序は、環境変数 → config.yaml → デフォルトの120秒です。

## no-agentモード（スクリプトのみのジョブ） {#no-agent-mode-script-only-jobs}

LLMの推論を必要としない繰り返しジョブ — 古典的なウォッチドッグ、ディスク/メモリのアラート、ハートビート、CIのpingなど — の場合、作成時に`no_agent=True`を渡します。スケジューラはスクリプトをスケジュールどおりに実行し、その標準出力を直接配信し、エージェントを完全にスキップします。

```bash
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"
```

セマンティクス:

- スクリプトの標準出力（トリムされたもの）→ メッセージとしてそのまま配信されます。
- **空の標準出力 → サイレントtick**、配信なし。これがウォッチドッグのパターンです。「何か問題があるときだけ何かを言う」。
- 0以外の終了コードまたはタイムアウト → エラーアラートが配信されるため、壊れたウォッチドッグがサイレントに失敗することはありません。
- 最終行の`{"wakeAgent": false}` → サイレントtick（LLMジョブが使うのと同じゲート）。
- トークンなし、モデルなし、プロバイダーフォールバックなし — ジョブは推論レイヤーに一切触れません。

`.sh` / `.bash`ファイルは`/bin/bash`の下で実行され、それ以外は現在のPythonインタプリタ（`sys.executable`）の下で実行されます。スクリプトは`~/.hermes/scripts/`に置く必要があります（実行前スクリプトゲートと同じサンドボックスルール）。

### エージェントがこれらをセットアップしてくれる

`cronjob`ツールのスキーマは`no_agent`をHermesに直接公開しているため、チャットでウォッチドッグを記述してエージェントに組み立ててもらえます。

```text
Ping me on Telegram if RAM is over 85%, every 5 minutes.
```

Hermesは`write_file`経由でチェックスクリプトを`~/.hermes/scripts/`に書き込み、次を呼び出します。

```python
cronjob(action="create", schedule="every 5m",
        script="memory-watchdog.sh", no_agent=True,
        deliver="telegram", name="memory-watchdog")
```

メッセージ内容がスクリプトによって完全に決定される場合（ウォッチドッグ、しきい値アラート、ハートビート）、`no_agent=True`を自動的に選択します。同じツールでエージェントはジョブの一時停止、再開、編集、削除も行えるため、誰もCLIに触れることなく、ライフサイクル全体がチャット駆動になります。

実例については[スクリプトのみのcronジョブガイド](/docs/guides/cron-script-only)を参照してください。

## `context_from` でジョブを連鎖させる

cronジョブは、過去の実行の記憶を持たない分離されたセッションで実行されます。しかし、あるジョブの出力がまさに次のジョブに必要なものである場合があります。`context_from`パラメータはその接続を自動的に配線します — ジョブBのプロンプトには、実行時にジョブAの最新の出力がコンテキストとして先頭に付加されます。

```python
# ジョブ1: 生データを収集
cronjob(
    action="create",
    prompt="Fetch the top 10 AI/ML stories from Hacker News. Save them to ~/.hermes/data/briefs/raw.md in markdown format with title, URL, and score.",
    schedule="0 7 * * *",
    name="AI News Collector",
)

# ジョブ2: トリアージ — ジョブ1の出力をコンテキストとして受け取る
# ジョブ1のIDは次から取得: cronjob(action="list")
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/raw.md. Score each story 1–10 for engagement potential and novelty. Output the top 5 to ~/.hermes/data/briefs/ranked.md.",
    schedule="30 7 * * *",
    context_from="<job1_id>",
    name="AI News Triage",
)

# ジョブ3: 出荷 — ジョブ2の出力をコンテキストとして受け取る
cronjob(
    action="create",
    prompt="Read ~/.hermes/data/briefs/ranked.md. Write 3 tweet drafts (hook + body + hashtags). Deliver to telegram:7976161601.",
    schedule="0 8 * * *",
    context_from="<job2_id>",
    name="AI News Brief",
)
```

**仕組み:**

- ジョブ2が発火すると、Hermesは`~/.hermes/cron/output/{job1_id}/*.md`からジョブ1の最新の出力を読み取ります
- その出力はジョブ2のプロンプトに自動的に先頭付加されます
- ジョブ2は「このファイルを読め」とハードコードする必要はありません — その内容をコンテキストとして受け取ります
- 連鎖はどんな長さでも構いません: ジョブ1 → ジョブ2 → ジョブ3 → …

**`context_from` が受け付けるもの:**

| 形式 | 例 |
|--------|---------|
| 単一のジョブID（文字列） | `context_from="a1b2c3d4"` |
| 複数のジョブID（リスト） | `context_from=["job_a", "job_b"]` |

出力は列挙された順に連結されます。

**使いどころ:**

- 多段パイプライン（収集 → フィルタ → 整形 → 配信）
- ステップNの作業がステップN−1の出力に依存する依存タスク
- 1つのジョブが他の複数のジョブの結果を集約するファンアウト/ファンインのパターン

## プロバイダーのリカバリー

cronジョブは、設定済みのフォールバックプロバイダーと認証情報プールのローテーションを継承します。プライマリのAPIキーがレート制限されている場合やプロバイダーがエラーを返した場合、cronエージェントは次のことができます。

- `config.yaml`に`fallback_providers`（またはレガシーの`fallback_model`）を設定していれば、**代替プロバイダーにフォールバック**する
- 同じプロバイダーについて、[認証情報プール](/docs/user-guide/configuration#credential-pool-strategies)内の**次の認証情報にローテーション**する

これは、高頻度またはピーク時間帯に実行されるcronジョブがより堅牢になることを意味します — 単一のレート制限されたキーが実行全体を失敗させることはありません。

## スケジュール形式

エージェントの最終応答は自動的に配信されます — 同じ宛先について、cronプロンプトに`send_message`を含める必要は**ありません**。cron実行が、スケジューラが既に配信する宛先とまったく同じターゲットに対して`send_message`を呼び出すと、Hermesはその重複送信をスキップし、ユーザー向けの内容を最終応答に入れるようモデルに指示します。`send_message`は、追加の宛先や別の宛先に対してのみ使用してください。

### 相対遅延（ワンショット）

```text
30m     → 30分後に1回実行
2h      → 2時間後に1回実行
1d      → 1日後に1回実行
```

### 間隔（繰り返し）

```text
every 30m    → 30分ごと
every 2h     → 2時間ごと
every 1d     → 毎日
```

### cron式

```text
0 9 * * *       → 毎日午前9時
0 9 * * 1-5     → 平日の午前9時
0 */6 * * *     → 6時間ごと
30 8 1 * *      → 毎月1日の午前8時30分
0 0 * * 0       → 毎週日曜日の深夜0時
```

### ISOタイムスタンプ

```text
2026-03-15T09:00:00    → 2026年3月15日午前9時に1回限り
```

## 繰り返しの動作

| スケジュールの種類 | デフォルトの繰り返し | 動作 |
|--------------|----------------|----------|
| ワンショット（`30m`、タイムスタンプ） | 1 | 1回実行 |
| 間隔（`every 2h`） | 無期限 | 削除されるまで実行 |
| cron式 | 無期限 | 削除されるまで実行 |

これは上書きできます。

```python
cronjob(
    action="create",
    prompt="...",
    schedule="every 2h",
    repeat=5,
)
```

## プログラムによるジョブ管理

エージェント向けのAPIは1つのツールです。

```python
cronjob(action="create", ...)
cronjob(action="list")
cronjob(action="update", job_id="...")
cronjob(action="pause", job_id="...")
cronjob(action="resume", job_id="...")
cronjob(action="run", job_id="...")
cronjob(action="remove", job_id="...")
```

`update`では、紐づいたスキルをすべて削除するために`skills=[]`を渡します。

## cronジョブが利用できるツールセット

cronは各ジョブを、チャットプラットフォームが紐づいていない新規エージェントセッションで実行します。デフォルトでは、cronエージェントは**`hermes tools`で`cron`プラットフォーム向けに設定したツールセット**を取得します — CLIのデフォルトでも、ありとあらゆるものでもありません。

```bash
hermes tools
# → cursesのUIで「cron」プラットフォームを選ぶ
# → Telegram/Discord等と同じようにツールセットをオン/オフする
```

ジョブごとのより厳密な制御は、`cronjob.create`の`enabled_toolsets`フィールド（または既存ジョブに対しては`cronjob.update`）で利用できます。

```text
cronjob(action="create", name="weekly-news-summary",
        schedule="every sunday 9am",
        enabled_toolsets=["web", "file"],      # web + fileのみ、terminal/browser等なし
        prompt="Summarize this week's AI news: ...")
```

ジョブに`enabled_toolsets`が設定されている場合はそれが優先されます。そうでない場合は`hermes tools`のcronプラットフォーム設定が優先され、それもなければHermesは組み込みのデフォルトにフォールバックします。これはコスト管理にとって重要です。`moa`、`browser`、`delegation`を小さな「ニュース取得」ジョブごとに持ち込むと、すべてのLLM呼び出しでツールスキーマのプロンプトが肥大化します。

### エージェントを完全にスキップする: `wakeAgent`

cronジョブが事前チェックスクリプト（`script=`経由）を紐づけている場合、スクリプトは実行時にHermesがそもそもエージェントを起動すべきかどうかを判断できます。次の形式の最終標準出力行を出力します。

```text
{"wakeAgent": false}
```

…すると、cronはこのtickでのエージェント実行を完全にスキップします。状態が実際に変化したときだけLLMを起こせばよい高頻度のポーリング（1〜5分ごと）に便利です — そうしないと、中身のないエージェントターンに何度も課金されることになります。

```python
# 事前チェックスクリプト
import json, sys
latest = fetch_latest_issue_count()
prev = read_state("issue_count")
if latest == prev:
    print(json.dumps({"wakeAgent": False}))   # このtickをスキップ
    sys.exit(0)
write_state("issue_count", latest)
print(json.dumps({"wakeAgent": True, "context": {"new_issues": latest - prev}}))
```

`wakeAgent`が省略された場合、デフォルトは`true`（通常どおりエージェントを起こす）です。

### ジョブの連鎖: `context_from`

cronジョブは、`context_from`に他の1つ以上のジョブの名前（またはID）を列挙することで、それらの最新の成功出力を取り込めます。

```text
cronjob(action="create", name="daily-digest",
        schedule="every day 7am",
        context_from=["ai-news-fetch", "github-prs-fetch"],
        prompt="Write the daily digest using the outputs above.")
```

参照されたジョブの最新の完了出力が、この実行のためのコンテキストとしてプロンプトの上に注入されます。各上流エントリは有効なジョブIDまたは名前でなければなりません（`cronjob action="list"`を参照）。注意: 連鎖は*最新の完了*出力を読み取ります — 同じtickで実行中の上流ジョブを待つことはありません。

## ジョブのストレージ

ジョブは`~/.hermes/cron/jobs.json`に保存されます。ジョブ実行の出力は`~/.hermes/cron/output/{job_id}/{timestamp}.md`に保存されます。

ジョブは`model`と`provider`を`null`として保存することがあります。これらのフィールドが省略されている場合、Hermesは実行時にグローバル設定からそれらを解決します。これらはジョブごとのオーバーライドが設定されている場合にのみジョブレコードに現れます。

ストレージはアトミックなファイル書き込みを使用するため、書き込みが中断されても、部分的に書き込まれたジョブファイルが残ることはありません。

## 自己完結したプロンプトは依然として重要

:::warning Important
cronジョブは完全に新規のエージェントセッションで実行されます。プロンプトには、紐づいたスキルによってまだ提供されていない、エージェントが必要とするすべてを含める必要があります。
:::

**悪い例:** `"Check on that server issue"`

**良い例:** `"SSH into server 192.168.1.100 as user 'deploy', check if nginx is running with 'systemctl status nginx', and verify https://example.com returns HTTP 200."`

## セキュリティ

スケジュールタスクのプロンプトは、作成時と更新時にプロンプトインジェクションや認証情報の流出パターンがないかスキャンされます。不可視のUnicodeトリック、SSHバックドアの試み、明白なシークレット流出ペイロードを含むプロンプトはブロックされます。
