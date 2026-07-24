---
sidebar_position: 5
title: "Hermes を Python ライブラリとして使う"
description: "AIAgent を独自の Python スクリプト、Web アプリ、自動化パイプラインに組み込む — CLI 不要"
---

# Hermes を Python ライブラリとして使う

Hermes は単なる CLI ツールではありません。`AIAgent` を直接インポートして、独自の Python
スクリプト、Web アプリケーション、自動化パイプラインの中でプログラム的に使えます。このガイドでは、
その方法を説明します。

---

## インストール

Hermes をリポジトリから直接インストールします。

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

または [uv](https://docs.astral.sh/uv/) で。

```bash
uv pip install git+https://github.com/NousResearch/hermes-agent.git
```

`requirements.txt` にピン留めすることもできます。

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
Hermes をライブラリとして使う場合、CLI が使うのと同じ環境変数が必要です。最低限、
`OPENROUTER_API_KEY`（直接プロバイダーアクセスを使う場合は `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY`）を設定してください。
:::

---

## 基本的な使い方

Hermes を使う最も簡単な方法は `chat()` メソッドです。メッセージを渡すと、文字列が返ってきます。

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)
response = agent.chat("What is the capital of France?")
print(response)
```

`chat()` は会話ループ全体（ツール呼び出し、リトライ、すべて）を内部で処理し、最終的なテキスト応答
だけを返します。

:::warning
Hermes を自分のコードに組み込むときは、常に `quiet_mode=True` を設定してください。これを指定しないと、
エージェントは CLI のスピナー、進捗インジケーター、その他のターミナル出力を表示し、アプリケーションの
出力を散らかしてしまいます。
:::

---

## 会話の完全な制御

会話をより細かく制御するには、`run_conversation()` を直接使います。これは、完全な応答、メッセージ
履歴、メタデータを含む辞書を返します。

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="Search for recent Python 3.13 features",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"Messages exchanged: {len(result['messages'])}")
```

返される辞書には次が含まれます。

- **`final_response`** — エージェントの最終的なテキスト応答
- **`messages`** — 完全なメッセージ履歴（system、user、assistant、ツール呼び出し）

（渡した `task_id` は VM 分離のためにエージェントインスタンスに保存されますが、返り値の辞書には
エコーバックされません。）

その呼び出しに対して、一時的なシステムプロンプトを上書きするカスタムシステムメッセージを渡すことも
できます。

```python
result = agent.run_conversation(
    user_message="Explain quicksort",
    system_message="You are a computer science tutor. Use simple analogies.",
)
```

---

## ツールの設定

`enabled_toolsets` または `disabled_toolsets` を使って、エージェントがアクセスできるツールセットを
制御します。

```python
# web ツール（ブラウジング、検索）のみを有効化
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# terminal アクセス以外のすべてを有効化
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
最小限でロックダウンされたエージェントが欲しいとき（例: リサーチボット用の web 検索のみ）は
`enabled_toolsets` を使います。ほとんどの機能は欲しいが特定のものを制限する必要があるとき（例:
共有環境での terminal アクセス禁止）は `disabled_toolsets` を使います。
:::

---

## マルチターンの会話

メッセージ履歴を渡し直すことで、複数のターンにわたって会話の状態を維持します。

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
)

# 最初のターン
result1 = agent.run_conversation("My name is Alice")
history = result1["messages"]

# 2 番目のターン — エージェントはコンテキストを覚えている
result2 = agent.run_conversation(
    "What's my name?",
    conversation_history=history,
)
print(result2["final_response"])  # "Your name is Alice."
```

`conversation_history` パラメータは、前の結果の `messages` リストを受け取ります。エージェントは
それを内部でコピーするので、元のリストが変更されることはありません。

---

## トラジェクトリの保存

トラジェクトリの保存を有効にすると、会話を ShareGPT 形式でキャプチャできます — トレーニング
データの生成やデバッグに便利です。

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("Write a Python function to sort a list")
# trajectory_samples.jsonl に ShareGPT 形式で保存される
```

各会話は単一の JSONL 行として追記されるため、自動実行からデータセットを簡単に収集できます。

---

## カスタムシステムプロンプト

`ephemeral_system_prompt` を使うと、エージェントの挙動を導きつつ、トラジェクトリファイルには
**保存されない**カスタムシステムプロンプトを設定できます（トレーニングデータをクリーンに保ちます）。

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="You are a SQL expert. Only answer database questions.",
    quiet_mode=True,
)

response = agent.chat("How do I write a JOIN query?")
print(response)
```

これは、専門化されたエージェント（コードレビュアー、ドキュメントライター、SQL アシスタント）を、
すべて同じ基盤のツールを使って構築するのに最適です。

---

## バッチ処理

多数のプロンプトを並列で実行するために、Hermes には `batch_runner.py` が含まれています。これは、
適切なリソース分離を備えた並行 `AIAgent` インスタンスを管理します。

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

各プロンプトは独自の `task_id` と分離された環境を取得します。カスタムのバッチロジックが必要な場合は、
`AIAgent` を直接使って独自に構築できます。

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "Explain recursion",
    "What is a hash table?",
    "How does garbage collection work?",
]

def process_prompt(prompt):
    # スレッドセーフのためにタスクごとに新しいエージェントを作成
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        quiet_mode=True,
        skip_memory=True,
    )
    return agent.chat(prompt)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

:::warning
**スレッドまたはタスクごとに、新しい `AIAgent` インスタンスを必ず作成してください**。エージェントは
内部状態（会話履歴、ツールセッション、イテレーションカウンター）を保持しており、これを共有するのは
スレッドセーフではありません。
:::

---

## 統合の例

### FastAPI エンドポイント

```python
from fastapi import FastAPI
from pydantic import BaseModel
from run_agent import AIAgent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "anthropic/claude-sonnet-4"

@app.post("/chat")
async def chat(request: ChatRequest):
    agent = AIAgent(
        model=request.model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    response = agent.chat(request.message)
    return {"response": response}
```

### Discord ボット

```python
import discord
from run_agent import AIAgent

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!hermes "):
        query = message.content[8:]
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
        )
        response = agent.chat(query)
        await message.channel.send(response[:2000])

client.run("YOUR_DISCORD_TOKEN")
```

### CI/CD パイプラインのステップ

```python
#!/usr/bin/env python3
"""CI step: auto-review a PR diff."""
import subprocess
from run_agent import AIAgent

diff = subprocess.check_output(["git", "diff", "main...HEAD"]).decode()

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    disabled_toolsets=["terminal", "browser"],
)

review = agent.chat(
    f"Review this PR diff for bugs, security issues, and style problems:\n\n{diff}"
)
print(review)
```

---

## 主なコンストラクタパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|---------|-------------|
| `model` | `str` | `"anthropic/claude-opus-4.6"` | OpenRouter 形式のモデル |
| `quiet_mode` | `bool` | `False` | CLI 出力を抑制する |
| `enabled_toolsets` | `List[str]` | `None` | 特定のツールセットをホワイトリストにする |
| `disabled_toolsets` | `List[str]` | `None` | 特定のツールセットをブラックリストにする |
| `save_trajectories` | `bool` | `False` | 会話を JSONL に保存する |
| `ephemeral_system_prompt` | `str` | `None` | カスタムシステムプロンプト（トラジェクトリには保存されない） |
| `max_iterations` | `int` | `90` | 会話ごとのツール呼び出しの最大イテレーション数 |
| `skip_context_files` | `bool` | `False` | AGENTS.md ファイルの読み込みをスキップする |
| `skip_memory` | `bool` | `False` | 永続メモリの読み書きを無効にする |
| `api_key` | `str` | `None` | API キー（環境変数にフォールバック） |
| `base_url` | `str` | `None` | カスタム API エンドポイント URL |
| `platform` | `str` | `None` | プラットフォームのヒント（`"discord"`、`"telegram"` など） |

---

## 重要な注意

:::tip
- 作業ディレクトリの `AGENTS.md` ファイルをシステムプロンプトに読み込みたくない場合は、
  **`skip_context_files=True`** を設定します。
- エージェントが永続メモリを読み書きしないようにするには **`skip_memory=True`** を設定します —
  ステートレスな API エンドポイントには推奨です。
- `platform` パラメータ（例: `"discord"`、`"telegram"`）は、プラットフォーム固有のフォーマット
  ヒントを注入し、エージェントが出力スタイルを適応させるようにします。
:::

:::warning
- **スレッドセーフ**: スレッドまたはタスクごとに 1 つの `AIAgent` を作成してください。インスタンスを
  並行呼び出し間で共有しないでください。
- **リソースのクリーンアップ**: エージェントは、会話が終了すると自動的にリソース（ターミナル
  セッション、ブラウザインスタンス）をクリーンアップします。長時間稼働するプロセスで実行している
  場合は、各会話が正常に完了するようにしてください。
- **イテレーション制限**: デフォルトの `max_iterations=90` は寛大です。単純な Q&A のユースケースでは、
  暴走するツール呼び出しループを防ぎコストを抑えるために、低くする（例: `max_iterations=10`）ことを
  検討してください。
:::
