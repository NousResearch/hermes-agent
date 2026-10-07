---
sidebar_position: 8
title: "コード実行"
description: "RPCツールアクセスを備えたプログラム的なPython実行 — 複数ステップのワークフローを1ターンに集約する"
---

# コード実行（プログラム的なツール呼び出し）

`execute_code` ツールは、エージェントがHermesのツールをプログラム的に呼び出すPythonスクリプトを書けるようにし、複数ステップのワークフローを1回のLLMターンに集約します。スクリプトはエージェントホスト上の子プロセスで実行され、UnixドメインソケットのRPCを通じてHermesと通信します。

## 仕組み

1. エージェントは `from hermes_tools import ...` を使ったPythonスクリプトを書きます
2. Hermesは、RPC関数を備えた `hermes_tools.py` スタブモジュールを生成します
3. HermesはUnixドメインソケットを開き、RPCリスナースレッドを開始します
4. スクリプトは子プロセスで実行されます — ツール呼び出しはソケットを通じてHermesに戻ります
5. LLMに返されるのはスクリプトの `print()` 出力のみで、中間的なツール結果はコンテキストウィンドウに入りません

```python
# エージェントは次のようなスクリプトを書けます:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... フィルタリングと処理 ...
print(summary)
```

**スクリプト内で利用可能なツール:** `web_search`、`web_extract`、`read_file`、`write_file`、`search_files`、`patch`、`terminal`（フォアグラウンドのみ）。

## エージェントがこれを使うとき

エージェントは、次のような場合に `execute_code` を使用します。

- 間に処理ロジックを挟む**3回以上のツール呼び出し**
- 大量データのフィルタリングまたは条件分岐
- 結果に対するループ

主な利点: 中間的なツール結果はコンテキストウィンドウに入らず — 最終的な `print()` 出力のみが返ってくるため、トークン使用量を劇的に削減します。

## 実践的な例

### データ処理パイプライン

```python
from hermes_tools import search_files, read_file
import json

# すべての設定ファイルを見つけてデータベース設定を抽出する
matches = search_files("database", path=".", file_glob="*.yaml", limit=20)
configs = []
for match in matches.get("matches", []):
    content = read_file(match["path"])
    configs.append({"file": match["path"], "preview": content["content"][:200]})

print(json.dumps(configs, indent=2))
```

### 複数ステップのWebリサーチ

```python
from hermes_tools import web_search, web_extract
import json

# 検索、抽出、要約を1ターンで行う
results = web_search("Rust async runtime comparison 2025", limit=5)
summaries = []
for r in results["data"]["web"]:
    page = web_extract([r["url"]])
    for p in page.get("results", []):
        if p.get("content"):
            summaries.append({
                "title": r["title"],
                "url": r["url"],
                "excerpt": p["content"][:500]
            })

print(json.dumps(summaries, indent=2))
```

### 一括ファイルリファクタリング

```python
from hermes_tools import search_files, read_file, patch

# 非推奨APIを使っているすべてのPythonファイルを見つけて修正する
matches = search_files("old_api_call", path="src/", file_glob="*.py")
fixed = 0
for match in matches.get("matches", []):
    result = patch(
        path=match["path"],
        old_string="old_api_call(",
        new_string="new_api_call(",
        replace_all=True
    )
    if "error" not in str(result):
        fixed += 1

print(f"Fixed {fixed} files out of {len(matches.get('matches', []))} matches")
```

### ビルドとテストのパイプライン

```python
from hermes_tools import terminal, read_file
import json

# テストを実行し、結果を解析してレポートする
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# テスト出力を解析する
passed = output.count(" passed")
failed = output.count(" failed")
errors = output.count(" error")

report = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "exit_code": result.get("exit_code", -1),
    "summary": output[-500:] if len(output) > 500 else output
}

print(json.dumps(report, indent=2))
```

## 実行モード

`execute_code` には、`~/.hermes/config.yaml` の `code_execution.mode` で制御される2つの実行モードがあります。

| モード | 作業ディレクトリ | Pythonインタープリター |
|------|-------------------|--------------------|
| **`project`**（デフォルト） | セッションの作業ディレクトリ（`terminal()` と同じ） | アクティブな `VIRTUAL_ENV` / `CONDA_PREFIX` のpython、なければHermes自身のpythonにフォールバック |
| `strict` | ユーザーのプロジェクトから隔離された一時ステージングディレクトリ | `sys.executable`（Hermes自身のpython） |

**`project` のままにすべきとき:** `import pandas`、`from my_project import foo`、または `open(".env")` のような相対パスを、`terminal()` の中と同じように動作させたい場合。これはほとんどの場合に望ましい挙動です。

**`strict` に切り替えるべきとき:** 最大限の再現性が必要な場合 — ユーザーがどのvenvをアクティブにしたかに関係なく、毎セッション同じインタープリターを使いたい、かつスクリプトをプロジェクトツリーから隔離したい（相対パス経由でプロジェクトファイルを誤って読み込むリスクがない）場合。

```yaml
# ~/.hermes/config.yaml
code_execution:
  mode: project   # または "strict"
```

`project` モードでのフォールバックの挙動: `VIRTUAL_ENV` / `CONDA_PREFIX` が未設定、壊れている、または3.8より古いPythonを指している場合、リゾルバーはクリーンに `sys.executable` にフォールバックします — エージェントが動作するインタープリターを持たない状態になることは決してありません。

セキュリティ上重要な不変条件は、両モードで同一です。

- 環境のスクラブ（APIキー、トークン、認証情報の除去）
- ツールのホワイトリスト（スクリプトは `execute_code` を再帰的に呼び出したり、`delegate_task` やMCPツールを呼び出したりできません）
- リソース制限（タイムアウト、stdout上限、ツール呼び出し上限）

モードを切り替えると、スクリプトがどこで実行され、どのインタープリターが実行するかが変わりますが、どの認証情報が見えるか、どのツールを呼び出せるかは変わりません。

## リソース制限

| リソース | 制限 | 備考 |
|----------|-------|-------|
| **タイムアウト** | 5分（300秒） | スクリプトはSIGTERMで終了され、5秒の猶予後にSIGKILLされます |
| **Stdout** | 50 KB | 出力は `[output truncated at 50KB]` の注記とともに切り詰められます |
| **Stderr** | 10 KB | デバッグ用に、非ゼロ終了時に出力へ含められます |
| **ツール呼び出し** | 実行あたり50回 | 制限に達するとエラーが返されます |

すべての制限は `config.yaml` で設定可能です。

```yaml
# ~/.hermes/config.yaml 内
code_execution:
  mode: project      # project（デフォルト） | strict
  timeout: 300       # スクリプトあたりの最大秒数（デフォルト: 300）
  max_tool_calls: 50 # 実行あたりの最大ツール呼び出し回数（デフォルト: 50）
```

## スクリプト内でのツール呼び出しの仕組み

スクリプトが `web_search("query")` のような関数を呼び出すと、次のようになります。

1. 呼び出しはJSONにシリアライズされ、Unixドメインソケットを通じて親プロセスに送られます
2. 親は標準の `handle_function_call` ハンドラを通じてディスパッチします
3. 結果はソケットを通じて返されます
4. 関数は解析された結果を返します

これは、スクリプト内のツール呼び出しが通常のツール呼び出しと同一に振る舞うことを意味します — 同じレート制限、同じエラー処理、同じ機能です。唯一の制約は、`terminal()` がフォアグラウンド専用である（`background` や `pty` パラメータが使えない）ことです。

## エラー処理

スクリプトが失敗すると、エージェントは構造化されたエラー情報を受け取ります。

- **非ゼロ終了コード**: stderrが出力に含まれるため、エージェントは完全なトレースバックを見られます
- **タイムアウト**: スクリプトは終了され、エージェントは `"Script timed out after 300s and was killed."` を見ます
- **中断**: 実行中にユーザーが新しいメッセージを送信した場合、スクリプトは終了され、エージェントは `[execution interrupted — user sent a new message]` を見ます
- **ツール呼び出し制限**: 50回の呼び出し制限に達すると、以降のツール呼び出しはエラーメッセージを返します

レスポンスには常に `status`（success/error/timeout/interrupted）、`output`、`tool_calls_made`、`duration_seconds` が含まれます。

## セキュリティ

:::danger セキュリティモデル
子プロセスは**最小限の環境**で実行されます。APIキー、トークン、認証情報はデフォルトで除去されます。スクリプトはRPCチャネルを通じてのみツールにアクセスします — 明示的に許可されない限り、環境変数からシークレットを読み取ることはできません。
:::

名前に `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL`、`PASSWD`、`AUTH` を含む環境変数は除外されます。安全なシステム変数（`PATH`、`HOME`、`LANG`、`SHELL`、`PYTHONPATH`、`VIRTUAL_ENV` など）のみが渡されます。

### スキルの環境変数パススルー

スキルがそのfrontmatterで `required_environment_variables` を宣言している場合、それらの変数はスキルが読み込まれた後に `execute_code` と `terminal` の両方の子プロセスへ**自動的に渡されます**。これにより、任意のコードに対するセキュリティ姿勢を弱めることなく、スキルが宣言したAPIキーを使えるようになります。

スキル以外のユースケースでは、`config.yaml` で変数を明示的に許可リストに登録できます。

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

詳細は[セキュリティガイド](/docs/user-guide/security#environment-variable-passthrough)を参照してください。

Hermesは常に、スクリプトと自動生成された `hermes_tools.py` RPCスタブを、実行後にクリーンアップされる一時ステージングディレクトリに書き込みます。`strict` モードでは、スクリプトもそこで*実行されます*。`project` モードでは、セッションの作業ディレクトリで実行されます（ステージングディレクトリは `PYTHONPATH` に残るため、importは引き続き解決されます）。子プロセスは独自のプロセスグループで実行されるため、タイムアウトや中断時にクリーンに終了できます。

## execute_code vs terminal

| ユースケース | execute_code | terminal |
|----------|-------------|----------|
| 間にツール呼び出しを挟む複数ステップのワークフロー | ✅ | ❌ |
| 単純なシェルコマンド | ❌ | ✅ |
| 大きなツール出力のフィルタリング/処理 | ✅ | ❌ |
| ビルドやテストスイートの実行 | ❌ | ✅ |
| 検索結果に対するループ処理 | ✅ | ❌ |
| インタラクティブ/バックグラウンドプロセス | ❌ | ✅ |
| 環境にAPIキーが必要 | ⚠️ [パススルー](/docs/user-guide/security#environment-variable-passthrough)経由のみ | ✅（ほとんどがパススルーされる） |

**目安:** 呼び出しの間にロジックを挟んでHermesのツールをプログラム的に呼び出す必要があるときは `execute_code` を使います。シェルコマンド、ビルド、プロセスの実行には `terminal` を使います。

## プラットフォームサポート

コード実行にはUnixドメインソケットが必要で、**LinuxとmacOSでのみ**利用できます。Windowsでは自動的に無効化されます — エージェントは通常の逐次的なツール呼び出しにフォールバックします。
