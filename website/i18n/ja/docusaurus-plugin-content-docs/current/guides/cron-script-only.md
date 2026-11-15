---
sidebar_position: 13
title: "スクリプトのみの Cron ジョブ（LLM なし）"
description: "LLM を完全にスキップする、古典的な監視犬型 cron ジョブ — スクリプトがスケジュールどおりに実行され、その標準出力がメッセージングプラットフォームに配信されます。メモリアラート、ディスクアラート、CI ping、定期的なヘルスチェック。"
---

# スクリプトのみの Cron ジョブ

送りたいメッセージが何かをすでに正確に分かっていることがあります。それについて推論するためのエージェントは不要で、必要なのはタイマーで実行されるスクリプトと、その出力（あれば）を Telegram / Discord / Slack / Signal に届けることだけです。

Hermes はこれを **no-agent モード** と呼びます。LLM を抜いた cron システムです。

<!-- ascii-guard-ignore -->
```
   ┌──────────────────┐          ┌──────────────────┐
   │ scheduler tick   │  every   │ run script       │
   │ (every N minutes)│ ──────▶ │ (bash or python) │
   └──────────────────┘          └──────────────────┘
                                          │
                                          │ stdout
                                          ▼
                                 ┌──────────────────┐
                                 │ delivery router  │
                                 │ (telegram/disc…) │
                                 └──────────────────┘
```
<!-- ascii-guard-ignore-end -->

- **LLM 呼び出しなし。** ゼロトークン、ゼロのエージェントループ、ゼロのモデル費用。
- **スクリプトがジョブそのもの。** スクリプトがアラートするかどうかを決めます。出力を出す → メッセージが送信される。何も出さない → サイレントなティック。
- **Bash または Python。** `.sh` / `.bash` ファイルは、利用可能であれば `PATH` 上の `bash` で、そうでなければ `/bin/bash` で実行され、それ以外の拡張子は現在の Python インタプリタで実行されます。Python スクリプトは `--interpreter` で **ユーザー管理の venv** を固定することもできます（[独自の Python 環境を使う](#using-your-own-python-environment) を参照）。パスは `~/.hermes/scripts/` 内に解決される必要があります（そのディレクトリ内にとどまる限り、相対パス、絶対パス、`~` 形式のいずれも可）。cron スクリプトは Hermes プロセスの環境からプロバイダーの認証情報を **継承しません**。
- **同じスケジューラー。** LLM ジョブと並んで `cronjob` に存在し、一時停止、再開、一覧表示、ログ、配信ターゲティングはすべて同じように機能します。

## いつ使うか

no-agent モードは次の用途に使います。

- **メモリ／ディスク／GPU の監視犬。** 5分ごとに実行し、しきい値を超えたときだけアラートします。
- **CI フック。** デプロイ完了 → コミット SHA を投稿。ビルド失敗 → ログの最後の100行を送信。
- **定期的なメトリクス。** 「毎日午前9時に Stripe の売上」をシンプルな API 呼び出し + 整形で。
- **外部イベントのポーラー。** API をチェックし、状態変化時にアラート。
- **ハートビート。** N 分ごとにダッシュボードに ping し、ホストが生きていることを証明。

エージェントに何を言うべきかを **判断** させる必要がある場合は、通常の（LLM 駆動の）cron ジョブを使います。長いドキュメントの要約、フィードから興味深い項目の選択、人間にやさしいメッセージの起草などです。no-agent パスは、スクリプトの標準出力がすでにメッセージそのものである場合のためのものです。

## チャットから1つ作成する

no-agent モードの真の利点は、エージェント自身があなたのために監視犬をセットアップできることです。エディタも、シェルも、CLI フラグを覚える必要もありません。望むことを説明すれば、Hermes がスクリプトを書き、スケジュールし、いつ発火するかを教えてくれます。

### トランスクリプトの例

> **あなた:** ping me on telegram if RAM is over 85% every 5 minutes
>
> **Hermes:** *(`~/.hermes/scripts/memory-watchdog.sh` を書き、`no_agent=true` で `cronjob(...)` を呼び出す)*
>
> Set up. Runs every 5 min, alerts Telegram only when RAM is over 85%. Script: `memory-watchdog.sh`. Job ID: `abc123`.

裏側で、エージェントは2つのツール呼び出しを行います。

```python
# 1. Write the check script
write_file(
    path="~/.hermes/scripts/memory-watchdog.sh",
    content='''#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  echo "RAM ${ram_pct}% on $(hostname)"
fi
# Empty stdout = silent tick; no message sent.
''',
)

# 2. Schedule it — no_agent=True skips the LLM on every tick
cronjob(
    action="create",
    schedule="every 5m",
    script="memory-watchdog.sh",
    no_agent=True,
    deliver="telegram",
    name="memory-watchdog",
)
```

それ以降、すべてのティックは無料です。スケジューラーがスクリプトを実行し、空でなければその標準出力を Telegram にパイプし、モデルには一切触れません。

### エージェントがあなたの代わりに決めること

「X のときにアラートして」や「N 分ごとに Y をチェックして Z なら教えて」のようにリクエストを言い表すと、Hermes の `cronjob_manage` ツールの説明は、メッセージの内容がスクリプトによって完全に決定される場合は `no_agent=True` を使うように指示します。リクエストに推論が必要な場合（*「新しい issue を要約して」*、*「最も興味深い見出しを選んで」*、*「フレンドリーなリマインダーを起草して」*）は、通常の LLM 駆動のパスにフォールバックします。

`--no-agent` を自分で指定する必要はありません。動作を説明するだけです。

### チャットから監視犬を管理する

エージェントは、作成するのと同じ方法でジョブを一時停止、再開、編集、削除できます。

> **あなた:** stop the memory watchdog for tonight
>
> **Hermes:** *(`cronjob(action='pause', job_id='abc123')` を呼び出す)*
>
> Paused. Resume with "turn it back on" or via `hermes cron resume abc123`.

> **あなた:** change it to every 15 minutes
>
> **Hermes:** *(`cronjob(action='update', job_id='abc123', schedule='every 15m')` を呼び出す)*

完全なライフサイクル（作成 / 一覧 / 更新 / 一時停止 / 再開 / 即時実行 / 削除）が、CLI コマンドを学ばなくてもエージェントから利用できます。

## CLI から1つ作成する

シェルがお好みですか？ CLI パスでは、3つのコマンドで同じ結果が得られます。

```bash
# 1. Write your script
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
# Alert when RAM usage is over 85%. Silent otherwise.
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "⚠ RAM ${RAM_PCT}% on $(hostname)"
fi
# Empty stdout = silent run; no message sent.
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2. Schedule it
hermes cron create "every 5m" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram \
  --name "memory-watchdog"

# 3. Verify
hermes cron list
hermes cron run <job_id>    # fire it once to test
```

これで全部です。プロンプトも、スキルも、モデルもありません。


## スクリプト出力が配信にどうマッピングされるか

| スクリプトの動作 | 結果 |
|-----------------|--------|
| Exit 0、空でない標準出力 | 標準出力がそのまま配信される |
| Exit 0、空の標準出力 | サイレントなティック — 配信なし |
| Exit 0、標準出力の最終行に `{"wakeAgent": false}` を含む | サイレントなティック（LLM ジョブと共有のゲート） |
| 非ゼロの終了コード | エラーアラートが配信される（壊れた監視犬がサイレントに失敗しないように） |
| スクリプトのタイムアウト | エラーアラートが配信される |

「空のときはサイレント」という動作が、古典的な監視犬パターンの鍵です。スクリプトは毎分自由に実行できますが、チャネルが実際に注意を要する何かがあるときだけメッセージを目にします。

## スクリプトのルール

スクリプトは `~/.hermes/scripts/` 内に解決されなければなりません。これは実行時に強制されます。解決後のターゲットがそのディレクトリ内にとどまる場合、相対名、絶対パス、`~` で始まるパスは受け付けられます。パストラバーサルやシンボリックリンクによる脱出は拒否されます。同じディレクトリは、LLM ジョブで使われる事前チェックスクリプトのゲートと共有されます。

インタプリタの選択はファイル拡張子によります。

| 拡張子 | インタプリタ |
|-----------|-------------|
| `.sh`、`.bash` | `PATH` 上の `bash`（フォールバックは `/bin/bash`） |
| それ以外 | `sys.executable`（現在の Python）、または [設定されたインタプリタ](#using-your-own-python-environment) |

`#!/...` シバンは意図的に尊重 **しません** 。インタプリタのセットを明示的かつ小さく保つことで、スケジューラーが信頼する面が減ります。

### 独自の Python 環境を使う {#using-your-own-python-environment}

デフォルトでは、Python の cron スクリプトは Hermes 自身の Python 環境で実行されます。この環境には Hermes 自身の依存関係しか含まれていないため、`openpyxl`、データベースドライバー、その他あなたがインストールしたパッケージをインポートするスクリプトは `ModuleNotFoundError` で失敗します。

代わりに `--interpreter` を使って、ジョブを **ユーザー管理の venv** に向けることができます。

```bash
# 1. Create a venv you own — it survives Hermes reinstalls/rebuilds.
uv venv ~/venvs/hermes-reporting --python 3.11
uv pip install --python ~/venvs/hermes-reporting/bin/python openpyxl

# 2. Schedule the job with that interpreter.
hermes cron create "0 8 * * *" \
  --no-agent \
  --script daily-report.py \
  --interpreter ~/venvs/hermes-reporting/bin/python \
  --deliver telegram
```

`--model` と同様、これはユーザーが所有する設定です。`hermes cron create/edit` で設定します。エージェントの `cronjob` ツールでは設定できません。

ルール:

- venv は **ユーザー管理** です。Hermes はそれを作成、フリーズ、復元したり、パッケージをインストールしたりしません — 指定されたパスを呼び出すだけです。
- パスは **絶対パスまたは `~` で始まるパス** でなければなりません（例: `~/venvs/reporting/bin/python3`）。`python3` のような裸の名前は、`PATH` の変更に対して安定しないため拒否されます。
- **Python の実行ファイル**（`python`、`python3`、`python3.12` など。シンボリックリンクのターゲットも含む）でなければなりません — `/bin/bash` やその他のインタプリタは拒否されます。
- **Python スクリプトにのみ** 適用されます。`.sh` / `.bash` は常に bash で実行されます。
- ジョブレベルの設定は、`script` と `monitor_script` が Python ファイルである場合、その両方に適用されます。
- 作成時ではなく **実行時に** 検証されます — cron ジョブは長期間存続するため、ジョブを作成してから発火するまでの間に venv が再構築されたり移動されたりする可能性があります。存在しない、または実行できないインタプリタは、明確なスクリプト失敗となり、他のエラーと同様に配信されます。
- 後でクリアするには: `hermes cron edit <job_id> --interpreter ""`。

## スケジュール構文

他のすべての cron ジョブと同じです。

```bash
hermes cron create "every 5m"        # interval
hermes cron create "every 2h"
hermes cron create "0 9 * * *"       # standard cron: 9am daily
hermes cron create "30m"             # one-shot: run once in 30 minutes
```

完全な構文については [cron 機能リファレンス](../user-guide/features/cron.md) を参照してください。

## 配信ターゲット

`--deliver` は、ゲートウェイが知っているすべてを受け付けます。いくつかの一般的な形:

```bash
--deliver telegram                       # platform home channel
--deliver telegram:-1001234567890        # specific chat
--deliver telegram:-1001234567890:17585  # specific Telegram forum topic
--deliver discord:#ops
--deliver slack:#engineering
--deliver signal:+15551234567
--deliver local                          # just save to ~/.hermes/cron/output/
```

ボットトークン型のプラットフォーム（Telegram、Discord、Slack、Signal、SMS、WhatsApp）では、スクリプト実行時にゲートウェイが稼働している必要はありません。ツールは、`~/.hermes/.env` / `~/.hermes/config.yaml` にすでにある認証情報を使って、各プラットフォームの REST エンドポイントを直接呼び出します。

## 編集とライフサイクル

```bash
hermes cron list                                    # see all jobs
hermes cron pause <job_id>                          # stop firing, keep definition
hermes cron resume <job_id>
hermes cron edit <job_id> --schedule "every 10m"    # adjust cadence
hermes cron edit <job_id> --agent                   # flip to LLM mode
hermes cron edit <job_id> --no-agent --script …     # flip back
hermes cron remove <job_id>                         # delete it
```

LLM ジョブで機能するすべて（一時停止、再開、手動トリガー、配信ターゲットの変更）は、no-agent ジョブでも機能します。

## 実践例: ディスク容量アラート

```bash
cat > ~/.hermes/scripts/disk-alert.sh <<'EOF'
#!/usr/bin/env bash
# Alert when / or /home is over 90% full.
THRESHOLD=90
df -h / /home 2>/dev/null | awk -v t="$THRESHOLD" '
  NR > 1 && $5+0 >= t {
    printf "⚠ Disk %s full on %s\n", $5, $6
  }
'
EOF
chmod +x ~/.hermes/scripts/disk-alert.sh

hermes cron create "*/15 * * * *" \
  --no-agent \
  --script disk-alert.sh \
  --deliver telegram \
  --name "disk-alert"
```

両方のファイルシステムが 90% 未満のときはサイレントで、いずれかが埋まると、しきい値を超えたファイルシステムごとにちょうど1行発火します。

## 他のパターンとの比較

| アプローチ | 何が実行されるか | いつ使うか |
|----------|-----------|-------------|
| `cronjob --no-agent`（このページ） | Hermes のスケジュールで動くあなたのスクリプト | 推論を必要としない、繰り返しの監視犬／アラート／メトリクス |
| `cronjob_manage`（デフォルト、LLM） | 任意の事前チェックスクリプト付きのエージェント | メッセージ内容がデータに対する推論を必要とする場合 |
| OS の cron + [webhook サブスクリプション](../user-guide/messaging/webhooks.md) への `curl` | OS のスケジュールで動くあなたのスクリプト | Hermes 自体が不調かもしれない場合（監視している当の対象） |

*ゲートウェイがダウンしていても* 発火しなければならない重要なシステムヘルスの監視犬には、OS レベルの cron と Hermes の webhook サブスクリプション（または任意の外部アラートエンドポイント）への素の `curl` を使ってください。これらは独立した OS プロセスとして実行され、Hermes が稼働していることに依存しません。ゲートウェイ内のスケジューラーは、監視対象が外部にある場合に適した選択肢です。

## 関連

- [Cron であらゆることを自動化する](./automate-with-cron.md) — LLM 駆動の cron パターン。
- [スケジュールタスク（Cron）リファレンス](../user-guide/features/cron.md) — 完全なスケジュール構文、ライフサイクル、配信ルーティング。
- [Webhook サブスクリプション](../user-guide/messaging/webhooks.md) — 外部スケジューラー向けの撃ちっぱなし HTTP エントリポイント。
- [ゲートウェイの内部](/docs/developer-guide/gateway-internals) — 配信ルーターの内部。
