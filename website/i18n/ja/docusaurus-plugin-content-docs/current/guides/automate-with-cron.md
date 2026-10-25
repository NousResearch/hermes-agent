---
sidebar_position: 11
title: "Cronで何でも自動化する"
description: "Hermes cronを使った実践的な自動化パターン — モニタリング、レポート、パイプライン、マルチスキルワークフロー"
---

# Cronで何でも自動化する

[デイリーブリーフィングボットのチュートリアル](/docs/guides/daily-briefing-bot)では基本を扱いました。このガイドはさらに先へ進みます — あなた自身のワークフローに応用できる、5つの実践的な自動化パターンです。

機能の完全なリファレンスについては、[スケジュールタスク（Cron）](/docs/user-guide/features/cron)を参照してください。

:::info 重要な概念
cronジョブは、現在のチャットの記憶を持たない新しいエージェントセッションで実行されます。プロンプトは**完全に自己完結**していなければなりません — エージェントが知る必要のあるすべてを含めてください。
:::

:::tip LLMは不要？ no-agentモードを使いましょう。
スクリプトがすでに送りたい正確なメッセージを生成している定期的なウォッチドッグ（メモリ警告、ディスク警告、CIのping、ハートビート）の場合は、[スクリプトのみのcronジョブ](/docs/guides/cron-script-only)でLLMを完全にスキップしましょう。トークンはゼロ、スケジューラは同じです。チャットでHermesに依頼してセットアップしてもらうこともできます — `cronjob` ツールは `no_agent=True` を選ぶべきタイミングを心得ており、あなたのためにスクリプトを書きます。
:::

---

## パターン1: ウェブサイトの変更モニター

URLの変更を監視し、何かが異なるときだけ通知を受け取ります。

ここでの秘密兵器は `script` パラメータです。Pythonスクリプトが各実行の前に動き、その標準出力がエージェントのコンテキストになります。スクリプトは機械的な作業（取得、差分）を担い、エージェントは推論（この変更は興味深いか？）を担います。

監視スクリプトを作成します。

```bash
mkdir -p ~/.hermes/scripts
```

```python title="~/.hermes/scripts/watch-site.py"
import hashlib, json, os, urllib.request

URL = "https://example.com/pricing"
STATE_FILE = os.path.expanduser("~/.hermes/scripts/.watch-site-state.json")

# 現在のコンテンツを取得
req = urllib.request.Request(URL, headers={"User-Agent": "Hermes-Monitor/1.0"})
content = urllib.request.urlopen(req, timeout=30).read().decode()
current_hash = hashlib.sha256(content.encode()).hexdigest()

# 以前の状態を読み込み
prev_hash = None
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev_hash = json.load(f).get("hash")

# 現在の状態を保存
with open(STATE_FILE, "w") as f:
    json.dump({"hash": current_hash, "url": URL}, f)

# エージェント向けの出力
if prev_hash and prev_hash != current_hash:
    print(f"CHANGE DETECTED on {URL}")
    print(f"Previous hash: {prev_hash}")
    print(f"Current hash: {current_hash}")
    print(f"\nCurrent content (first 2000 chars):\n{content[:2000]}")
else:
    print("NO_CHANGE")
```

cronジョブをセットアップします。

```bash
/cron add "every 1h" "If the script output says CHANGE DETECTED, summarize what changed on the page and why it might matter. If it says NO_CHANGE, respond with just [SILENT]." --script ~/.hermes/scripts/watch-site.py --name "Pricing monitor" --deliver telegram
```

:::tip [SILENT] のトリック
エージェントの最終応答に `[SILENT]` が含まれている場合、配信は抑制されます。つまり、実際に何かが起きたときだけ通知され、静かな時間帯にスパムが来ることはありません。
:::

---

## パターン2: 週次レポート

複数のソースから情報をまとめて、整形された要約にします。これは週に一度実行され、ホームチャンネルに配信されます。

```bash
/cron add "0 9 * * 1" "Generate a weekly report covering:

1. Search the web for the top 5 AI news stories from the past week
2. Search GitHub for trending repositories in the 'machine-learning' topic
3. Check Hacker News for the most discussed AI/ML posts

Format as a clean summary with sections for each source. Include links.
Keep it under 500 words — highlight only what matters." --name "Weekly AI digest" --deliver telegram
```

CLIから:

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly report covering the top AI news, trending ML GitHub repos, and most-discussed HN posts. Format with sections, include links, keep under 500 words." \
  --name "Weekly AI digest" \
  --deliver telegram
```

`0 9 * * 1` は標準的なcron式で、毎週月曜の午前9:00を意味します。

---

## パターン3: GitHubリポジトリウォッチャー

新しいissue、PR、またはリリースがないかリポジトリを監視します。

```bash
/cron add "every 6h" "Check the GitHub repository NousResearch/hermes-agent for:
- New issues opened in the last 6 hours
- New PRs opened or merged in the last 6 hours
- Any new releases

Use the terminal to run gh commands:
  gh issue list --repo NousResearch/hermes-agent --state open --json number,title,author,createdAt --limit 10
  gh pr list --repo NousResearch/hermes-agent --state all --json number,title,author,createdAt,mergedAt --limit 10

Filter to only items from the last 6 hours. If nothing new, respond with [SILENT].
Otherwise, provide a concise summary of the activity." --name "Repo watcher" --deliver discord
```

:::warning 自己完結したプロンプト
プロンプトに正確な `gh` コマンドが含まれている点に注目してください。cronエージェントは以前の実行やあなたの好みの記憶を持ちません — すべてを明示的に書き出しましょう。
:::

---

## パターン4: データ収集パイプライン

一定間隔でデータをスクレイピングし、ファイルに保存し、時間の経過に伴うトレンドを検出します。このパターンは、（収集用の）スクリプトと（分析用の）エージェントを組み合わせます。

```python title="~/.hermes/scripts/collect-prices.py"
import json, os, urllib.request
from datetime import datetime

DATA_DIR = os.path.expanduser("~/.hermes/data/prices")
os.makedirs(DATA_DIR, exist_ok=True)

# 現在のデータを取得（例: 暗号通貨の価格）
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
data = json.loads(urllib.request.urlopen(url, timeout=30).read())

# 履歴ファイルに追記
entry = {"timestamp": datetime.now().isoformat(), "prices": data}
history_file = os.path.join(DATA_DIR, "history.jsonl")
with open(history_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

# 分析用に直近の履歴を読み込み
lines = open(history_file).readlines()
recent = [json.loads(l) for l in lines[-24:]]  # 直近24個のデータポイント

# エージェント向けの出力
print(f"Current: BTC=${data['bitcoin']['usd']}, ETH=${data['ethereum']['usd']}")
print(f"Data points collected: {len(lines)} total, showing last {len(recent)}")
print(f"\nRecent history:")
for r in recent[-6:]:
    print(f"  {r['timestamp']}: BTC=${r['prices']['bitcoin']['usd']}, ETH=${r['prices']['ethereum']['usd']}")
```

```bash
/cron add "every 1h" "Analyze the price data from the script output. Report:
1. Current prices
2. Trend direction over the last 6 data points (up/down/flat)
3. Any notable movements (>5% change)

If prices are flat and nothing notable, respond with [SILENT].
If there's a significant move, explain what happened." \
  --script ~/.hermes/scripts/collect-prices.py \
  --name "Price tracker" \
  --deliver telegram
```

スクリプトが機械的な収集を行い、エージェントが推論のレイヤーを追加します。

---

## パターン5: マルチスキルワークフロー

複雑なスケジュールタスクのために、スキルを連鎖させます。スキルはプロンプトが実行される前に順番に読み込まれます。

```bash
# arxiv スキルで論文を探し、obsidian スキルでノートを保存する
/cron add "0 8 * * *" "Search arXiv for the 3 most interesting papers on 'language model reasoning' from the past day. For each paper, create an Obsidian note with the title, authors, abstract summary, and key contribution." \
  --skill arxiv \
  --skill obsidian \
  --name "Paper digest"
```

ツールから直接:

```python
cronjob(
    action="create",
    skills=["arxiv", "obsidian"],
    prompt="Search arXiv for papers on 'language model reasoning' from the past day. Save the top 3 as Obsidian notes.",
    schedule="0 8 * * *",
    name="Paper digest",
    deliver="local"
)
```

スキルは順番に読み込まれます — まず `arxiv`（論文の検索方法をエージェントに教える）、次に `obsidian`（ノートの書き方を教える）。プロンプトがそれらを結びつけます。

---

## ジョブを管理する

```bash
# アクティブなすべてのジョブを一覧表示
/cron list

# ジョブを即座にトリガー（テスト用）
/cron run <job_id>

# 削除せずにジョブを一時停止
/cron pause <job_id>

# 実行中のジョブのスケジュールやプロンプトを編集
/cron edit <job_id> --schedule "every 4h"
/cron edit <job_id> --prompt "Updated task description"

# 既存のジョブからスキルを追加または削除
/cron edit <job_id> --skill arxiv --skill obsidian
/cron edit <job_id> --clear-skills

# ジョブを完全に削除
/cron remove <job_id>
```

---

## 配信先

`--deliver` フラグは結果の送り先を制御します。

| 配信先 | 例 | ユースケース |
|--------|---------|----------|
| `origin` | `--deliver origin` | ジョブを作成したのと同じチャット（デフォルト） |
| `local` | `--deliver local` | ローカルファイルにのみ保存 |
| `telegram` | `--deliver telegram` | あなたのTelegramホームチャンネル |
| `discord` | `--deliver discord` | あなたのDiscordホームチャンネル |
| `slack` | `--deliver slack` | あなたのSlackホームチャンネル |
| 特定のチャット | `--deliver telegram:-1001234567890` | 特定のTelegramグループ |
| スレッド指定 | `--deliver telegram:-1001234567890:17585` | 特定のTelegramトピックスレッド |

---

## ヒント

**プロンプトを自己完結させる。** cronジョブ内のエージェントはあなたの会話の記憶を持ちません。URL、リポジトリ名、形式の好み、配信指示をプロンプトに直接含めてください。

**`[SILENT]` を積極的に使う。** 監視ジョブでは、常に「何も変わっていなければ `[SILENT]` で応答せよ」といった指示を含めましょう。これにより通知のノイズが防げます。

**データ収集にはスクリプトを使う。** `script` パラメータを使えば、退屈な部分（HTTPリクエスト、ファイルI/O、状態の追跡）をPythonスクリプトに任せられます。エージェントはスクリプトの標準出力だけを見て、それに推論を適用します。これはエージェント自身に取得をさせるより安価で信頼性が高いです。

**`/cron run` でテストする。** スケジュールのトリガーを待つ前に、`/cron run <job_id>` で即座に実行し、出力が正しく見えることを確認しましょう。

**スケジュール式。** サポートされる形式: 相対的な遅延（`30m`）、間隔（`every 2h`）、標準的なcron式（`0 9 * * *`）、ISOタイムスタンプ（`2025-06-15T09:00:00`）。`daily at 9am` のような自然言語はサポートされていません — 代わりに `0 9 * * *` を使ってください。

---

*完全なcronリファレンス — すべてのパラメータ、エッジケース、内部の仕組み — については、[スケジュールタスク（Cron）](/docs/user-guide/features/cron)を参照してください。*
