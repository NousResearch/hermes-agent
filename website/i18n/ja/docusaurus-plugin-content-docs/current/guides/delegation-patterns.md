---
sidebar_position: 13
title: "委譲と並列作業"
description: "サブエージェント委譲をいつどう使うか — 並列調査、コードレビュー、複数ファイル作業のためのパターン"
---

# 委譲と並列作業

Hermesは、独立した子エージェントを起動して、タスクに並列で取り組ませることができます。各サブエージェントは独自の会話、ターミナルセッション、ツールセットを持ちます。返ってくるのは最終的な要約のみで — 中間のツール呼び出しがあなたのコンテキストウィンドウに入ることはありません。

機能の完全なリファレンスについては、[サブエージェント委譲](/docs/user-guide/features/delegation)を参照してください。

---

## いつ委譲するか

**委譲に向いている候補:**
- 推論が重いサブタスク（デバッグ、コードレビュー、調査の統合）
- 中間データであなたのコンテキストをあふれさせるようなタスク
- 並列で独立したワークストリーム（AとBを同時に調査）
- バイアスなしにエージェントに取り組ませたい、新しいコンテキストのタスク

**別の手段を使う:**
- 単一のツール呼び出し → ツールを直接使うだけ
- ステップ間にロジックのある機械的な複数ステップ作業 → `execute_code`
- ユーザーとのやり取りが必要なタスク → サブエージェントは `clarify` を使えません
- 素早いファイル編集 → 直接行う
- 現在のターンを超えて存続しなければならない、長時間実行される永続的な作業 → `cronjob` または `terminal(background=True, notify_on_complete=True)`。`delegate_task` は**同期的**です: 親のターンが中断されると、アクティブな子はキャンセルされ、その作業は破棄されます。

---

## パターン: 並列調査

3つのトピックを同時に調査し、構造化された要約を受け取ります。

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

舞台裏では、Hermesは次を使います。

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress",
        "toolsets": ["web"]
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem",
        "toolsets": ["web"]
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies",
        "toolsets": ["web"]
    }
])
```

3つすべてが並行して実行されます。各サブエージェントは独立してウェブを検索し、要約を返します。その後、親エージェントがそれらを統合して一貫したブリーフィングにまとめます。

---

## パターン: コードレビュー

先入観なしにコードに取り組む、新しいコンテキストのサブエージェントにセキュリティレビューを委譲します。

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

鍵となるのは `context` フィールドです — サブエージェントが必要とするすべてを含めなければなりません。

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass.""",
    toolsets=["terminal", "file"]
)
```

:::warning コンテキストの問題
サブエージェントはあなたの会話について**まったく何も**知りません。完全にゼロから始まります。「さっき話していたバグを直して」と委譲しても、サブエージェントはどのバグのことか分かりません。ファイルパス、エラーメッセージ、プロジェクト構造、制約を常に明示的に渡してください。
:::

---

## パターン: 代替案の比較

同じ問題に対する複数のアプローチを並列で評価し、最良のものを選びます。

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

各サブエージェントは1つの選択肢を独立して調査します。互いに分離されているため、相互汚染はありません — それぞれの評価が独自の長所に基づいて成立します。親エージェントは3つすべての要約を受け取り、比較を行います。

---

## パターン: 複数ファイルのリファクタリング

大きなリファクタリングタスクを並列のサブエージェントに分割し、それぞれがコードベースの別の部分を担当します。

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py""",
        "toolsets": ["terminal", "file"]
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema.""",
        "toolsets": ["terminal", "file"]
    }
])
```

:::tip
各サブエージェントは独自のターミナルセッションを持ちます。同じプロジェクトディレクトリで、互いに踏み合うことなく作業できます — 別々のファイルを編集している限りは。2つのサブエージェントが同じファイルに触れる可能性がある場合は、並列作業の完了後に、そのファイルは自分で処理してください。
:::

---

## パターン: 収集してから分析

機械的なデータ収集には `execute_code` を使い、その後、推論の重い分析を委譲します。

```python
# ステップ1: 機械的な収集（推論が不要なのでここでは execute_code が良い）
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# 最も関連性の高い上位5件から全文を抽出
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# 分析ステップ用に保存
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# ステップ2: 推論の重い分析（ここでは委譲が良い）
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M.""",
    toolsets=["terminal", "file"]
)
```

これはしばしば最も効率的なパターンです: `execute_code` が10回以上の連続したツール呼び出しを安価に処理し、その後、サブエージェントがクリーンなコンテキストで唯一の高コストな推論タスクを行います。

---

## ツールセットの選択

サブエージェントが必要とするものに基づいてツールセットを選びます。

| タスクの種類 | ツールセット | 理由 |
|-----------|----------|-----|
| ウェブ調査 | `["web"]` | web_search + web_extract のみ |
| コード作業 | `["terminal", "file"]` | シェルアクセス + ファイル操作 |
| フルスタック | `["terminal", "file", "web"]` | メッセージング以外すべて |
| 読み取り専用の分析 | `["file"]` | ファイルの読み取りのみ、シェルなし |

ツールセットを制限すると、サブエージェントの焦点が保たれ、偶発的な副作用（調査用サブエージェントがシェルコマンドを実行するなど）を防げます。

---

## 制約

- **デフォルトは3つの並列タスク**: バッチはデフォルトで3つの同時サブエージェント（config.yaml の `delegation.max_concurrent_children` で設定可能、ハードな上限はなく、下限が1のみ）
- **ネストした委譲はオプトイン**: リーフのサブエージェント（デフォルト）は `delegate_task`、`clarify`、`memory`、`send_message`、`execute_code` を呼べません。オーケストレーターのサブエージェント（`role="orchestrator"`）はさらなる委譲のために `delegate_task` を保持しますが、それは `delegation.max_spawn_depth` がデフォルトの1より上げられている場合（1-3に対応）のみです。他の4つはブロックされたままです。グローバルに無効化するには `delegation.orchestrator_enabled: false` を設定します。

### 並列度と深さのチューニング

| 設定 | デフォルト | 範囲 | 効果 |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | `delegate_task` 呼び出しごとの並列バッチサイズ |
| `max_spawn_depth` | 1 | 1-3 | さらに起動できる委譲レベルの数 |

例: ネストしたサブエージェントで30の並列ワーカーを実行する:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **別々のターミナル** — 各サブエージェントは、別々の作業ディレクトリと状態を持つ独自のターミナルセッションを得ます
- **会話履歴なし** — サブエージェントは、親エージェントが `delegate_task` を呼び出すときに渡す `goal` と `context` だけを見ます
- **デフォルトは50イテレーション** — 単純なタスクではコストを抑えるために `max_iterations` を低く設定します
- **永続的ではない** — `delegate_task` は同期的で、親のターン内で実行されます。親が中断された場合（新しいユーザーメッセージ、`/stop`、`/new`）、アクティブなすべての子はキャンセルされ（`status="interrupted"`）、その作業は破棄されます。現在のターンを超えて存続しなければならない作業には、`cronjob` または `terminal(background=True, notify_on_complete=True)` を使用してください。

---

## ヒント

**ゴールを具体的にする。** 「バグを直して」は曖昧すぎます。「api/handlers.py の47行目の、process_request() が parse_body() から None を受け取る TypeError を直して」なら、サブエージェントが作業を進めるのに十分です。

**ファイルパスを含める。** サブエージェントはあなたのプロジェクト構造を知りません。関連するファイル、プロジェクトのルート、テストコマンドへの絶対パスを常に含めてください。

**コンテキスト分離のために委譲を使う。** 新しい視点が欲しいことがあります。委譲はあなたに問題を明確に言語化させることを強制し、サブエージェントはあなたの会話で積み上がった前提なしにそれに取り組みます。

**結果を確認する。** サブエージェントの要約はあくまで要約です。サブエージェントが「バグを直し、テストは通った」と言っても、自分でテストを実行するかdiffを読んで検証してください。

---

*完全な委譲リファレンス — すべてのパラメータ、ACP連携、高度な設定 — については、[サブエージェント委譲](/docs/user-guide/features/delegation)を参照してください。*
