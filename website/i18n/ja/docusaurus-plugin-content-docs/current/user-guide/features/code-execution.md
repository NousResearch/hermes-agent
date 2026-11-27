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
# The agent can write scripts like:
from hermes_tools import web_search, web_extract

results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # ... filter and process ...
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

# Find all config files and extract database settings
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

# Search, extract, and summarize in one turn
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

# Find all Python files using deprecated API and fix them
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

# Run tests, parse results, and report
result = terminal("cd /project && python -m pytest --tb=short -q 2>&1", timeout=120)
output = result.get("output", "")

# Parse test output
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
  mode: project   # or "strict"
```

`project` モードでのフォールバックの挙動: `VIRTUAL_ENV` / `CONDA_PREFIX` が未設定、壊れている、または3.8より古いPythonを指している場合、リゾルバーはクリーンに `sys.executable` にフォールバックします — エージェントが動作するインタープリターを持たない状態になることは決してありません。

セキュリティ上重要な不変条件は、両モードで同一です。

- 環境のスクラブ（APIキー、トークン、認証情報の除去）
- ツールのホワイトリスト（スクリプトは `execute_code` を再帰的に呼び出したり、`delegate_task` やMCPツールを呼び出したりできません）
- リソース制限（タイムアウト、stdout上限、ツール呼び出し上限）

モードを切り替えると、スクリプトがどこで実行され、どのインタープリターが実行するかが変わりますが、どの認証情報が見えるか、どのツールを呼び出せるかは変わりません。

## 永続セッションカーネル

同じセッション、実行モード、インタープリター、作業ディレクトリ、ツールセットに対する呼び出しでは、
Python の子プロセスが再利用されます。import、変数、読み込んだデータはセル間で
保持され得ます。子プロセスの環境はカーネルの起動時に固定されます。

そのカーネルの状態を破棄するには `reset: true` を渡します。タイムアウトや中断されたカーネルでも
状態が失われることがあります。後からターミナルの環境を変更しても、既存のカーネル内で
すでに見えているとは考えないでください。旧来の `code_execution.kernel_mode`
設定は、もはや独立したスイッチではありません。

## リソース制限

| リソース | 制限 | 備考 |
|----------|-------|-------|
| **タイムアウト** | 5分（300秒） | スクリプトはSIGTERMで終了され、5秒の猶予後にSIGKILLされます |
| **Stdout** | 50 KB | 先頭と末尾がインラインで表示されます。完全な出力は `~/.hermes/cache/exec/` に保存され、そのパスが結果に含まれます |
| **Stderr** | 10 KB | デバッグ用に、非ゼロ終了時に出力へ含められます |
| **ツール呼び出し** | 実行あたり50回 | 制限に達するとエラーが返されます |

すべての制限は `config.yaml` で設定可能です。

```yaml
# In ~/.hermes/config.yaml
code_execution:
  mode: project      # project (default) | strict
  timeout: 300       # Max seconds per script (default: 300)
  max_tool_calls: 50 # Max tool calls per execution (default: 50)
```

## 呼び出し間の状態（セッションカーネル）

ローカルのターミナルバックエンドでは、`execute_code` は呼び出しのたびに新しいインタープリターを起動するわけではありません。各セッションは永続的な Python カーネルを持つため、ある呼び出しでの変数、import、読み込んだデータを次の呼び出しで利用できます。エージェントはデータセットを毎回読み直す代わりに、1回読み込んで複数のターンにわたって照会できます。サブエージェントは独自のカーネルを持ち、カーネルがセッション間で共有されることはありません。サブエージェントのカーネルはちょうどそのサブエージェントが存在する間だけ生存します: サブエージェントの実行中はライブカーネルの上限から除外され（広いファンアウトでも、兄弟のカーネルがタスク途中で追い出されることはなくなりました）、サブエージェントが終了すると破棄されます。

カーネルが終了する条件:

- **タイムアウトまたは中断。** タイムアウトに達した（または中断された）セルはカーネルプロセスを終了させ、その状態は意図的に失われます。結果にその旨が示され、次の呼び出しでは新しいカーネルが起動します。
- **`reset=true`。** エージェントは `reset: true` を渡してカーネルの状態を破棄し、クリーンな状態から始められます。これは環境の変更を反映させる方法でもあります: カーネルの環境は起動時に固定されるため、新たに許可リストに追加されたパススルー変数は、カーネルがリセットされるまで見えません。
- **アイドルタイムアウトと追い出し。** カーネルは、セッションの終了時、`code_execution.kernel_idle_timeout` 秒（デフォルト 1800）アイドル状態が続いた後、または `code_execution.max_session_kernels`（デフォルト 4）を超えるトップレベルセッションのカーネルが生存していて最も古いものが追い出されたときに終了します（実行中のサブエージェントのカーネルは上限に数えられません）。

**編集されたモジュール。** カーネルはインポートしたすべてのモジュールを保持するため、プロジェクトのファイルが編集された後も、次のセルは以前に読み込まれたコードを実行し続けます。その場合、結果には編集されたモジュール名を示す警告が付き（ローカルカーネルでは `warning`、リモートバックエンドでは `kernel.warning`）、エージェントは `importlib.reload()` でそれらを再読み込みするか、`reset: true` を渡すことができます。このためにカーネルの状態が破棄されることはありません。標準ライブラリとインストール済みパッケージは監視されません。

セキュリティの枠組みはワンショットのスクリプトと同じです: 環境のスクラブ、ツールのホワイトリスト、呼び出しごとのツール予算はすべてのセルに適用され、ツール呼び出しの権限（承認、セッション、許可リスト）はセルごとに再バインドされます。

```yaml
# ~/.hermes/config.yaml
code_execution:
  kernel_idle_timeout: 1800   # seconds a kernel may sit idle before it is reaped
  max_session_kernels: 4      # kernels kept alive at once; oldest is evicted past this
```

**リモートバックエンド**（Docker、SSH、Modal）は、同じ契約でリモートのセッションカーネルを実行します。バックエンド上でカーネルを起動できない場合、Hermes は各呼び出しを単独のスクリプトとして実行する方式にフォールバックし、その旨を結果に示します。

**大きな出力。** 50 KB を超える stdout は先頭と末尾がインラインで表示され、完全なテキストは `~/.hermes/cache/exec/` 配下に保存されてそのパスが結果に含まれるため、エージェントはスクリプトを再実行する代わりに `read_file` でページ送りしながら読めます。

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

承認: ゲートウェイのセッションでは、スクリプトは実行前に全体として承認されます。ローカルの CLI、TUI、Desktop、ACP のセッションでは、あなたが入力したターンにはスクリプト全体に対するプロンプトは表示されません: スクリプト内の Python レベルのファイル操作やプロセス操作はプロンプトなしで実行され、`terminal()` の呼び出しだけが確認を求めます。注入されたターンや自動化のターンでは、スクリプト全体に対するプロンプトが維持されます。[`execute_code` スクリプト](../security.md#execute-code-scripts)を参照してください。

### スキルの環境変数パススルー

スキルがそのfrontmatterで `required_environment_variables` を宣言している場合、それらの変数はスキルが読み込まれた後に `execute_code` と `terminal` の両方の子プロセスへ**自動的に渡されます**。これにより、任意のコードに対するセキュリティ姿勢を弱めることなく、スキルが宣言したAPIキーを使えるようになります。

スキル以外のユースケースでは、`config.yaml` で変数を明示的に許可リストに登録できます。

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

詳細は[セキュリティガイド](../security.md#environment-variable-passthrough)を参照してください。

### 子プロセス内の `HERMES_*` 変数

子プロセスが受け取る運用上の `HERMES_*` 変数は、完全一致の名前による小さな固定セットだけです。

- `HERMES_HOME`
- `HERMES_PROFILE`
- `HERMES_CONFIG`
- `HERMES_ENV`

（加えて、RPC チャネルが機能するように Hermes が明示的に注入する `HERMES_RPC_DIR` / `HERMES_RPC_SOCKET` / `TZ` / `HOME`）。

:::note 挙動の変更
以前のバージョンでは、名前が `HERMES_` で始まる**あらゆる**変数が子プロセスに
渡されていました。この広いプレフィックスは、セキュリティ強化のために削除されました: シークレットの部分文字列に一致しない
`HERMES_*` という名前の設定（たとえば `HERMES_BASE_URL`、`HERMES_KANBAN_DB`、`HERMES_*_WEBHOOK`
エンドポイントなど）が、任意のサンドボックス化されたコードに漏れる可能性があったためです。

`execute_code` スクリプト — またはそれが import 時に読み込むリポジトリ/プラグインのモジュール —
が上記4つの運用上の名前以外の `HERMES_*` 変数に依存していた場合、子プロセス内ではその変数が
**未設定**になります。この削除は意図的なものであり、
バグではありません。
:::

**回避策 — 変数を明示的に許可する。** どちらの方法も、変数を
`execute_code` *と* `terminal` の両方の子プロセスに渡し、どちらもシークレット除去の保証を
弱めません（Hermes が管理するプロバイダーの認証情報が、この方法で再許可されることは
決してありません）。

1. **マシンごと、`config.yaml` で** — 正確な変数名を
   パススルーの許可リストに追加します。

   ```yaml
   terminal:
     env_passthrough:
       - HERMES_KANBAN_DB
       - HERMES_BASE_URL
   ```

2. **スキルごと、スキルの frontmatter で** — 宣言しておくと、そのスキルが
   読み込まれるたびに自動的に登録されます。

   ```yaml
   required_environment_variables:
     - HERMES_KANBAN_DB
   ```

**診断方法。** 子プロセスが許可リストにない `HERMES_*` 変数を1つ以上削除すると、
Hermes はそれらの名前を示し、`env_passthrough` という逃げ道を案内する1行の `debug` ログを出力します。
スクリプトが `HERMES_*` 変数が欠けているかのように振る舞う場合は、デバッグログを有効にして実行し（`hermes logs --level
DEBUG`、または `~/.hermes/logs/agent.log` を確認）、
`execute_code: dropped N non-allowlisted HERMES_* var(s)` を探してください。

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
| 環境にAPIキーが必要 | ⚠️ [パススルー](../security.md#environment-variable-passthrough)経由のみ | ✅（ほとんどがパススルーされる） |

**目安:** 呼び出しの間にロジックを挟んでHermesのツールをプログラム的に呼び出す必要があるときは `execute_code` を使います。シェルコマンド、ビルド、プロセスの実行には `terminal` を使います。

## プラットフォームサポート

コード実行は **Linux、macOS、Windows** で利用できます。Linux と macOS では、RPC チャネルに Unix ドメインソケットを使います。`AF_UNIX` が信頼できない Windows では、Hermes はサンドボックスの RPC トランスポートとして自動的にループバックの TCP ソケットにフォールバックします。リモートのターミナルバックエンド（Docker/SSH/Modal など）は代わりにファイルベースの RPC トランスポートを使い、さらにバックエンド内に Python 3 が必要です。
