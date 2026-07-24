---
sidebar_position: 9
sidebar_label: "プラグインを作る"
title: "Hermesプラグインを作る"
description: "ツール、フック、データファイル、スキルを備えた完全なHermesプラグインをステップバイステップで構築するガイド"
---

# Hermesプラグインを作る

このガイドでは、完全なHermesプラグインをゼロから構築する手順を解説します。最後には、複数のツール、ライフサイクルフック、同梱されたデータファイル、バンドルされたスキルを備えた動作するプラグインが完成します。プラグインシステムがサポートするすべての機能が揃います。

:::info どのガイドが必要か分からない場合
Hermesにはいくつかの異なるプラガブルインターフェースがあります。Pythonの`register_*` APIを使うものもあれば、設定駆動やドロップインディレクトリ方式のものもあります。まずはこのマップを使ってください。

| 追加したいもの | 読むべきガイド |
|---|---|
| カスタムツール、フック、スラッシュコマンド、スキル、CLIサブコマンド | **このガイド**（汎用プラグインサーフェス） |
| **LLM / 推論バックエンド**（新しいプロバイダー） | [Model Provider Plugins](/docs/developer-guide/model-provider-plugin) |
| **ゲートウェイチャネル**（Discord/Telegram/IRC/Teams など） | [Adding Platform Adapters](/docs/developer-guide/adding-platform-adapters) |
| **メモリバックエンド**（Honcho/Mem0/Supermemory など） | [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) |
| **コンテキスト圧縮エンジン** | [Context Engine Plugins](/docs/developer-guide/context-engine-plugin) |
| **画像生成バックエンド** | [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) |
| **TTSバックエンド**（任意のCLI — Piper、VoxCPM、Kokoro、ボイスクローニングなど） | [TTS custom command providers](/docs/user-guide/features/tts#custom-command-providers) — 設定駆動、Python不要 |
| **STTバックエンド**（カスタムwhisper / ASR CLI） | [Voice Message Transcription](/docs/user-guide/features/tts#voice-message-transcription-stt) — `HERMES_LOCAL_STT_COMMAND`をシェルテンプレートに設定 |
| **MCP経由の外部ツール**（filesystem、GitHub、Linear、任意のMCPサーバー） | [MCP](/docs/user-guide/features/mcp) — `config.yaml`に`mcp_servers.<name>`を宣言 |
| **ゲートウェイイベントフック**（起動時、セッションイベント、コマンドで発火） | [Event Hooks](/docs/user-guide/features/hooks#gateway-event-hooks) — `~/.hermes/hooks/<name>/`に`HOOK.yaml` + `handler.py`を配置 |
| **シェルフック**（イベント時にシェルコマンドを実行） | [Shell Hooks](/docs/user-guide/features/hooks#shell-hooks) — `config.yaml`の`hooks:`配下に宣言 |
| **追加のスキルソース**（カスタムGitHubリポジトリ、プライベートスキルインデックス） | [Skills](/docs/user-guide/features/skills) — `hermes skills tap add <repo>` · [tapの公開](/docs/user-guide/features/skills#publishing-a-custom-skill-tap) |
| ファーストクラスの**コア**推論プロバイダー（プラグインではない） | [Adding Providers](/docs/developer-guide/adding-providers) |

設定駆動（TTS、STT、MCP、シェルフック）やドロップインディレクトリ（ゲートウェイフック）スタイルを含む、すべての拡張サーフェスを一覧でまとめた[Pluggable interfaces table](/docs/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)も参照してください。
:::

## 何を作るのか

2つのツールを持つ**電卓（calculator）**プラグインです。
- `calculate` — 数式を評価する（`2**16`、`sqrt(144)`、`pi * 5**2`）
- `unit_convert` — 単位を変換する（`100 F → 37.78 C`、`5 km → 3.11 mi`）

さらに、すべてのツール呼び出しをログに記録するフックと、バンドルされたスキルファイルも追加します。

## ステップ1: プラグインディレクトリを作成する

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

## ステップ2: マニフェストを書く

`plugin.yaml`を作成します。

```yaml
name: calculator
version: 1.0.0
description: Math calculator — evaluate expressions and convert units
provides_tools:
  - calculate
  - unit_convert
provides_hooks:
  - post_tool_call
```

これはHermesに次のように伝えます。「私はcalculatorというプラグインで、ツールとフックを提供します」。`provides_tools`と`provides_hooks`フィールドは、プラグインが登録するものを列挙したリストです。

追加できるオプションフィールド:
```yaml
author: Your Name
requires_env:          # env変数で読み込みをゲートする。インストール時にプロンプト表示
  - SOME_API_KEY       # シンプル形式 — 未設定ならプラグインは無効化される
  - name: OTHER_KEY    # リッチ形式 — インストール時に説明/URLを表示
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
```

## ステップ3: ツールスキーマを書く

`schemas.py`を作成します。これは、LLMがいつツールを呼び出すか判断するために読むものです。

```python
"""Tool schemas — what the LLM sees."""

CALCULATE = {
    "name": "calculate",
    "description": (
        "Evaluate a mathematical expression and return the result. "
        "Supports arithmetic (+, -, *, /, **), functions (sqrt, sin, cos, "
        "log, abs, round, floor, ceil), and constants (pi, e). "
        "Use this for any math the user asks about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2**10', 'sqrt(144)')",
            },
        },
        "required": ["expression"],
    },
}

UNIT_CONVERT = {
    "name": "unit_convert",
    "description": (
        "Convert a value between units. Supports length (m, km, mi, ft, in), "
        "weight (kg, lb, oz, g), temperature (C, F, K), data (B, KB, MB, GB, TB), "
        "and time (s, min, hr, day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "The numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'km', 'lb', 'F', 'GB')",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit (e.g., 'mi', 'kg', 'C', 'MB')",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
}
```

**スキーマが重要な理由:** `description`フィールドは、LLMがツールをいつ使うか判断する手がかりになります。何をするツールで、いつ使うのかを具体的に書きましょう。`parameters`は、LLMが渡す引数を定義します。

## ステップ4: ツールハンドラを書く

`tools.py`を作成します。これは、LLMがツールを呼び出したときに実際に実行されるコードです。

```python
"""Tool handlers — the code that runs when the LLM calls each tool."""

import json
import math

# 式評価のための安全なグローバル — ファイル/ネットワークアクセスなし
_SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "log": math.log, "log2": math.log2, "log10": math.log10,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
    "factorial": math.factorial,
}


def calculate(args: dict, **kwargs) -> str:
    """数式を安全に評価する。

    ハンドラのルール:
    1. args（dict）を受け取る — LLMが渡したパラメータ
    2. 処理を行う
    3. JSON文字列を返す — 常に、エラー時でも
    4. 前方互換性のため **kwargs を受け取る
    """
    expression = args.get("expression", "").strip()
    if not expression:
        return json.dumps({"error": "No expression provided"})

    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_MATH)
        return json.dumps({"expression": expression, "result": result})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero"})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Invalid: {e}"})


# 変換テーブル — 値は基本単位
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}


def _convert_temp(value, from_u, to_u):
    # 摂氏に正規化
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # ターゲットに変換
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)


def unit_convert(args: dict, **kwargs) -> str:
    """単位を変換する。"""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # 温度
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # 比率ベースの変換
        for table in (_LENGTH, _WEIGHT, _DATA, _TIME):
            lc = {k.lower(): v for k, v in table.items()}
            if from_unit.lower() in lc and to_unit.lower() in lc:
                result = float(value) * lc[from_unit.lower()] / lc[to_unit.lower()]
                return json.dumps({"input": f"{value} {from_unit}",
                                 "result": round(result, 6),
                                 "output": f"{round(result, 6)} {to_unit}"})

        return json.dumps({"error": f"Cannot convert {from_unit} → {to_unit}"})
    except Exception as e:
        return json.dumps({"error": f"Conversion failed: {e}"})
```

**ハンドラの主要ルール:**
1. **シグネチャ:** `def my_handler(args: dict, **kwargs) -> str`
2. **戻り値:** 常にJSON文字列。成功時もエラー時も同様です。
3. **例外を投げない:** すべての例外をキャッチし、代わりにエラーJSONを返します。
4. **`**kwargs`を受け取る:** Hermesは将来、追加のコンテキストを渡す可能性があります。

## ステップ5: 登録処理を書く

`__init__.py`を作成します。これはスキーマとハンドラを結びつけます。

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# フック経由でツール使用状況を追跡する
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """フック: すべてのツール呼び出し後に実行される（自分のものだけではない）。"""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def register(ctx):
    """スキーマをハンドラに結びつけ、フックを登録する。"""
    ctx.register_tool(name="calculate",    toolset="calculator",
                      schema=schemas.CALCULATE,    handler=tools.calculate)
    ctx.register_tool(name="unit_convert", toolset="calculator",
                      schema=schemas.UNIT_CONVERT, handler=tools.unit_convert)

    # このフックは自分のものだけでなく、すべてのツール呼び出しで発火する
    ctx.register_hook("post_tool_call", _on_post_tool_call)
```

**`register()`が行うこと:**
- 起動時に正確に1回だけ呼び出されます
- `ctx.register_tool()`はツールをレジストリに登録します — モデルは即座にそれを認識します
- `ctx.register_hook()`はライフサイクルイベントを購読します
- `ctx.register_cli_command()`はCLIサブコマンドを登録します（例: `hermes my-plugin <subcommand>`）
- `ctx.register_command()`はセッション内スラッシュコマンドを登録します（例: CLI / ゲートウェイチャット内での`/myplugin <args>`） — 後述の[スラッシュコマンドを登録する](#register-slash-commands)を参照
- `ctx.dispatch_tool(name, arguments)` — 親エージェントのコンテキスト（承認、認証情報、task_id）を自動的に結びつけた状態で、他の任意のツール（組み込みまたは別プラグインのもの）を呼び出します。`terminal`、`read_file`、その他のツールを、モデルが直接呼び出したかのように呼び出す必要があるスラッシュコマンドハンドラから便利に使えます。
- この関数がクラッシュした場合、プラグインは無効化されますが、Hermesは問題なく動作を続けます

**`dispatch_tool`の例 — ツールを実行するスラッシュコマンド:**

```python
def handle_scan(ctx, argstr):
    """terminalツールをレジストリ経由で呼び出して /scan を実装する。"""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{argstr}'"})
    return result  # 呼び出し元のチャットUIに返される

def register(ctx):
    ctx.register_command("scan", handle_scan, help="Find files matching a glob")
```

ディスパッチされたツールは、通常の承認、リダクション、バジェットのパイプラインを通過します。これらを迂回するショートカットではなく、本物のツール呼び出しです。

## ステップ6: テストする

Hermesを起動します。

```bash
hermes
```

バナーのツールリストに`calculator: calculate, unit_convert`が表示されるはずです。

次のようなプロンプトを試してみましょう。
```
What's 2 to the power of 16?
Convert 100 fahrenheit to celsius
What's the square root of 2 times pi?
How many gigabytes is 1.5 terabytes?
```

プラグインの状態を確認します。
```
/plugins
```

出力:
```
Plugins (1):
  ✓ calculator v1.0.0 (2 tools, 1 hooks)
```

### プラグイン検出のデバッグ

プラグインが表示されない場合、または表示されるが読み込まれない場合は、`HERMES_PLUGINS_DEBUG=1`を設定して、stderrに詳細な検出ログを出力させます。

```bash
HERMES_PLUGINS_DEBUG=1 hermes plugins list
```

すべてのプラグインソース（bundled、user、project、entry-points）について、次の情報が表示されます。

- どのディレクトリがスキャンされ、それぞれが何個のマニフェストを生成したか
- マニフェストごと: 解決されたキー、name、kind、source、ディスク上のパス
- スキップ理由: `disabled via config`、`not enabled in config`、`exclusive plugin`、`no plugin.yaml, depth cap reached`
- 読み込み時: インポートされているプラグインと、`register(ctx)`が登録した内容（ツール、フック、スラッシュコマンド、CLIコマンド）の1行サマリー
- パース失敗時: 例外の完全なトレースバック（YAMLスキャナーエラーなど）
- `register()`失敗時: `__init__.py`内で例外を発生させた行を指す完全なトレースバック

同じログは、env変数が設定されているとき、常に`~/.hermes/logs/agent.log`にWARNINGレベル（失敗のみ）とDEBUGレベル（すべて）で書き込まれます。そのため、env変数を付けて実行できない場合（例: ゲートウェイの内部から）は、代わりにログファイルをtailしてください。

```bash
hermes logs --level WARNING | grep -i plugin
```

プラグインが表示されない一般的な理由:

- **設定で有効化されていない** — プラグインはオプトインです。`hermes plugins enable <name>`を実行してください（nameは`plugins list`の出力から取得します。ネストレイアウトの場合は`<category>/<plugin>`になることがあります）。
- **ディレクトリレイアウトが間違っている** — `~/.hermes/plugins/<plugin-name>/plugin.yaml`（フラット）または`~/.hermes/plugins/<category>/<plugin-name>/plugin.yaml`（最大1階層のカテゴリネスト）でなければなりません。それより深いものは無視されます。
- **`__init__.py`がない** — プラグインディレクトリには`plugin.yaml`と、`register(ctx)`関数を持つ`__init__.py`の両方が必要です。
- **`kind`が間違っている** — ゲートウェイアダプターはマニフェストに`kind: platform`が必要です。メモリプロバイダーは`kind: exclusive`として自動検出され、`plugins.enabled`ではなく`memory.provider`設定を通じてルーティングされます。

## プラグインの最終的な構成

```
~/.hermes/plugins/calculator/
├── plugin.yaml      # 「私はcalculator、ツールとフックを提供します」
├── __init__.py      # 配線: スキーマ → ハンドラ、フックの登録
├── schemas.py       # LLMが読むもの（説明 + パラメータ仕様）
└── tools.py         # 実行されるもの（calculate, unit_convert 関数）
```

4つのファイル、明確な分離:
- **マニフェスト**はプラグインが何であるかを宣言する
- **スキーマ**はLLM向けにツールを説明する
- **ハンドラ**は実際のロジックを実装する
- **登録処理**はすべてを接続する

## プラグインは他に何ができるのか

### データファイルを同梱する

任意のファイルをプラグインディレクトリに配置し、インポート時に読み込めます。

```python
# tools.py または __init__.py 内
from pathlib import Path

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = yaml.safe_load(f)
```

### スキルをバンドルする

プラグインは、エージェントが`skill_view("plugin:skill")`で読み込むスキルファイルを同梱できます。`__init__.py`で登録します。

```
~/.hermes/plugins/my-plugin/
├── __init__.py
├── plugin.yaml
└── skills/
    ├── my-workflow/
    │   └── SKILL.md
    └── my-checklist/
        └── SKILL.md
```

```python
from pathlib import Path

def register(ctx):
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
```

エージェントは、名前空間付きの名前でスキルを読み込めるようになります。

```python
skill_view("my-plugin:my-workflow")   # → プラグインのバージョン
skill_view("my-workflow")              # → 組み込みバージョン（変更なし）
```

**主な特性:**
- プラグインスキルは**読み取り専用**です — `~/.hermes/skills/`には入らず、`skill_manage`で編集できません。
- プラグインスキルはシステムプロンプトの`<available_skills>`インデックスに**載りません** — 明示的にオプトインで読み込むものです。
- 素のスキル名は影響を受けません — 名前空間が組み込みスキルとの衝突を防ぎます。
- エージェントがプラグインスキルを読み込むと、同じプラグインの兄弟スキルを列挙したバンドルコンテキストバナーが先頭に付加されます。

:::tip レガシーパターン
古い`shutil.copy2`パターン（スキルを`~/.hermes/skills/`にコピーする方法）も依然として動作しますが、組み込みスキルとの名前衝突リスクを生みます。新しいプラグインでは`ctx.register_skill()`を推奨します。
:::

### 環境変数でゲートする

プラグインがAPIキーを必要とする場合:

```yaml
# plugin.yaml — シンプル形式（後方互換）
requires_env:
  - WEATHER_API_KEY
```

`WEATHER_API_KEY`が設定されていない場合、プラグインは明確なメッセージとともに無効化されます。クラッシュもエージェント内のエラーもなく、ただ「Plugin weather disabled (missing: WEATHER_API_KEY)」と表示されるだけです。

ユーザーが`hermes plugins install`を実行すると、未設定の`requires_env`変数について**対話的にプロンプト**が表示されます。値は自動的に`.env`に保存されます。

より良いインストール体験のためには、説明とサインアップURLを含むリッチ形式を使ってください。

```yaml
# plugin.yaml — リッチ形式
requires_env:
  - name: WEATHER_API_KEY
    description: "API key for OpenWeather"
    url: "https://openweathermap.org/api"
    secret: true
```

| フィールド | 必須 | 説明 |
|-------|----------|-------------|
| `name` | はい | 環境変数名 |
| `description` | いいえ | インストールプロンプト時にユーザーに表示される |
| `url` | いいえ | 認証情報の取得先 |
| `secret` | いいえ | `true`の場合、入力が隠される（パスワードフィールドのように） |

両方の形式を同じリスト内で混在させることができます。すでに設定されている変数は静かにスキップされます。

### 条件付きのツール利用可否

オプションのライブラリに依存するツールの場合:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = ツールはモデルから隠される
)
```

### 複数のフックを登録する

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### フックリファレンス

各フックは、**[Event Hooks reference](/docs/user-guide/features/hooks#plugin-hooks)**で詳しく説明されています（コールバックシグネチャ、パラメータ表、各フックが正確にいつ発火するか、例）。以下はサマリーです。

| フック | 発火するタイミング | コールバックシグネチャ | 戻り値 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](/docs/user-guide/features/hooks#pre_tool_call) | 任意のツールが実行される前 | `tool_name: str, args: dict, task_id: str` | 無視される |
| [`post_tool_call`](/docs/user-guide/features/hooks#post_tool_call) | 任意のツールが返った後 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 無視される |
| [`pre_llm_call`](/docs/user-guide/features/hooks#pre_llm_call) | ターンごとに1回、ツール呼び出しループの前 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [コンテキストインジェクション](#pre_llm_call-context-injection) |
| [`post_llm_call`](/docs/user-guide/features/hooks#post_llm_call) | ターンごとに1回、ツール呼び出しループの後（成功したターンのみ） | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 無視される |
| [`on_session_start`](/docs/user-guide/features/hooks#on_session_start) | 新しいセッションが作成されたとき（最初のターンのみ） | `session_id: str, model: str, platform: str` | 無視される |
| [`on_session_end`](/docs/user-guide/features/hooks#on_session_end) | すべての`run_conversation`呼び出しの終了時 + CLI終了時 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 無視される |
| [`on_session_finalize`](/docs/user-guide/features/hooks#on_session_finalize) | CLI/ゲートウェイがアクティブなセッションを破棄するとき | `session_id: str \| None, platform: str` | 無視される |
| [`on_session_reset`](/docs/user-guide/features/hooks#on_session_reset) | ゲートウェイが新しいセッションキーに切り替えるとき（`/new`、`/reset`） | `session_id: str, platform: str` | 無視される |

ほとんどのフックは、結果を返すだけのfire-and-forget型のオブザーバーです — 戻り値は無視されます。例外は`pre_llm_call`で、これは会話にコンテキストをインジェクトできます。

すべてのコールバックは、前方互換性のため`**kwargs`を受け取るべきです。フックコールバックがクラッシュした場合、ログに記録されてスキップされます。他のフックとエージェントは通常どおり続行します。

### `pre_llm_call`のコンテキストインジェクション {#pre_llm_call-context-injection}

これは、戻り値が意味を持つ唯一のフックです。`pre_llm_call`コールバックが`"context"`キーを持つdict（または素の文字列）を返すと、Hermesはそのテキストを**現在のターンのユーザーメッセージ**にインジェクトします。これは、メモリプラグイン、RAG連携、ガードレール、そしてモデルに追加コンテキストを提供する必要があるあらゆるプラグインのための仕組みです。

#### 戻り値の形式

```python
# context キーを持つ dict
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# 素の文字列（上記の dict 形式と等価）
return "Recalled memories:\n- User prefers dark mode"

# None を返すか、何も返さない → インジェクションなし（オブザーバーのみ）
return None
```

`"context"`キーを持つNoneでない空でない戻り値（または素の空でない文字列）はすべて収集され、現在のターンのユーザーメッセージに追加されます。

#### インジェクションの仕組み

インジェクトされたコンテキストは、システムプロンプトではなく**ユーザーメッセージ**に追加されます。これは意図的な設計上の選択です。

- **プロンプトキャッシュの保持** — システムプロンプトはターンをまたいで同一のままです。AnthropicとOpenRouterはシステムプロンプトのプレフィックスをキャッシュするため、それを安定させることでマルチターン会話の入力トークンを75%以上削減できます。プラグインがシステムプロンプトを変更すると、すべてのターンがキャッシュミスになってしまいます。
- **一時的** — インジェクションはAPI呼び出し時のみ発生します。会話履歴内の元のユーザーメッセージが変更されることはなく、セッションデータベースにも何も永続化されません。
- **システムプロンプトはHermesの領域** — そこにはモデル固有のガイダンス、ツール強制ルール、パーソナリティ指示、キャッシュされたスキルコンテンツが含まれます。プラグインは、エージェントのコア指示を変更するのではなく、ユーザーの入力と並べてコンテキストを提供します。

#### 例: メモリリコールプラグイン

```python
"""Memory plugin — recalls relevant context from a vector store."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """各LLMターンの前に呼び出される。リコールしたメモリを返す。"""
    try:
        resp = httpx.post(f"{MEMORY_API}/recall", json={
            "session_id": session_id,
            "query": user_message,
        }, timeout=3)
        memories = resp.json().get("results", [])
        if not memories:
            return None  # インジェクトするものがない

        text = "Recalled context from previous sessions:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # 静かに失敗し、エージェントを壊さない

def register(ctx):
    ctx.register_hook("pre_llm_call", recall_context)
```

#### 例: ガードレールプラグイン

```python
"""Guardrails plugin — enforces content policies."""

POLICY = """You MUST follow these content policies for this session:
- Never generate code that accesses the filesystem outside the working directory
- Always warn before executing destructive operations
- Refuse requests involving personal data extraction"""

def inject_guardrails(**kwargs):
    """すべてのターンにポリシーテキストをインジェクトする。"""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### 例: オブザーバーのみのフック（インジェクションなし）

```python
"""Analytics plugin — tracks turn metadata without injecting context."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """各LLM呼び出しの前に発火する。None を返す — コンテキストインジェクションなし。"""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # 戻り値なし → インジェクションなし

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### 複数のプラグインがコンテキストを返す場合

複数のプラグインが`pre_llm_call`からコンテキストを返す場合、それらの出力は二重改行で結合され、まとめてユーザーメッセージに追加されます。順序はプラグインの検出順（プラグインディレクトリ名のアルファベット順）に従います。

### CLIコマンドを登録する

プラグインは独自の`hermes <plugin>`サブコマンドツリーを追加できます。

```python
def _my_command(args):
    """hermes my-plugin <subcommand> のハンドラ。"""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """hermes my-plugin の argparse ツリーを構築する。"""
    subs = subparser.add_subparsers(dest="my_command")
    subs.add_parser("status", help="Show plugin status")
    subs.add_parser("config", help="Show plugin config")
    subparser.set_defaults(func=_my_command)

def register(ctx):
    ctx.register_tool(...)
    ctx.register_cli_command(
        name="my-plugin",
        help="Manage my plugin",
        setup_fn=_setup_argparse,
        handler_fn=_my_command,
    )
```

登録後、ユーザーは`hermes my-plugin status`、`hermes my-plugin config`などを実行できます。

**メモリプロバイダープラグイン**は、代わりに規約ベースのアプローチを使います。プラグインの`cli.py`ファイルに`register_cli(subparser)`関数を追加してください。メモリプラグインの検出システムが自動的に見つけるため、`ctx.register_cli_command()`の呼び出しは不要です。詳細は[Memory Provider Plugin guide](/docs/developer-guide/memory-provider-plugin#adding-cli-commands)を参照してください。

**アクティブプロバイダーによるゲート:** メモリプラグインのCLIコマンドは、そのプロバイダーが設定内でアクティブな`memory.provider`である場合にのみ表示されます。ユーザーがあなたのプロバイダーをセットアップしていない場合、あなたのCLIコマンドはヘルプ出力を散らかしません。

### スラッシュコマンドを登録する {#register-slash-commands}

プラグインはセッション内スラッシュコマンドを登録できます。これは、ユーザーが会話中に入力するコマンドです（`/lcm status`や`/ping`など）。CLIとゲートウェイ（Telegram、Discordなど）の両方で動作します。

```python
def _handle_status(raw_args: str) -> str:
    """/mystatus のハンドラ — コマンド名の後ろのすべてが渡される。"""
    if raw_args.strip() == "help":
        return "Usage: /mystatus [help|check]"
    return "Plugin status: all systems nominal"

def register(ctx):
    ctx.register_command(
        "mystatus",
        handler=_handle_status,
        description="Show plugin status",
    )
```

登録後、ユーザーは任意のセッションで`/mystatus`と入力できます。コマンドはオートコンプリート、`/help`の出力、Telegramボットメニューに表示されます。

**シグネチャ:** `ctx.register_command(name: str, handler: Callable, description: str = "")`

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `name` | `str` | 先頭のスラッシュを除いたコマンド名（例: `"lcm"`、`"mystatus"`） |
| `handler` | `Callable[[str], str \| None]` | 生の引数文字列とともに呼び出される。`async`でもよい。 |
| `description` | `str` | `/help`、オートコンプリート、Telegramボットメニューに表示される |

**`register_cli_command()`との主な違い:**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 呼び出し方 | セッション内で`/name` | ターミナルで`hermes name` |
| 動作する場所 | CLIセッション、Telegram、Discordなど | ターミナルのみ |
| ハンドラが受け取るもの | 生の引数文字列 | argparseの`Namespace` |
| ユースケース | 診断、ステータス、クイックアクション | 複雑なサブコマンドツリー、セットアップウィザード |

**衝突保護:** プラグインが組み込みコマンド（`help`、`model`、`new`など）と衝突する名前を登録しようとした場合、登録は静かに拒否され、ログに警告が出ます。組み込みコマンドが常に優先されます。

**非同期ハンドラ:** ゲートウェイのディスパッチは非同期ハンドラを自動的に検出してawaitするため、同期・非同期のどちらの関数も使えます。

```python
async def _handle_check(raw_args: str) -> str:
    result = await some_async_operation()
    return f"Check result: {result}"

def register(ctx):
    ctx.register_command("check", handler=_handle_check, description="Run async check")
```

### スラッシュコマンドからツールをディスパッチする

ツールをオーケストレーションする必要があるスラッシュコマンドハンドラ（`delegate_task`でサブエージェントを起動する、`file_edit`を呼び出すなど）は、フレームワークの内部に手を伸ばすのではなく、`ctx.dispatch_tool()`を使うべきです。親エージェントのコンテキスト（ワークスペースのヒント、スピナー、モデルの継承）が自動的に結びつけられます。

```python
def register(ctx):
    def _handle_deliver(raw_args: str):
        result = ctx.dispatch_tool(
            "delegate_task",
            {
                "goal": raw_args,
                "toolsets": ["terminal", "file", "web"],
            },
        )
        return result

    ctx.register_command(
        "deliver",
        handler=_handle_deliver,
        description="Delegate a goal to a subagent",
    )
```

**シグネチャ:** `ctx.dispatch_tool(name: str, args: dict, *, parent_agent=None) -> str`

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `name` | `str` | ツールレジストリに登録されているツール名（例: `"delegate_task"`、`"file_edit"`） |
| `args` | `dict` | ツール引数。モデルが送るのと同じ形状 |
| `parent_agent` | `Agent \| None` | オプションの上書き。省略した場合、現在のCLIエージェントから解決される（ゲートウェイモードではグレースフルに劣化する） |

**実行時の動作:**

- **CLIモード:** `parent_agent`はアクティブなCLIエージェントから解決され、ワークスペースのヒント、スピナー、モデル選択が期待どおりに継承されます。
- **ゲートウェイモード:** CLIエージェントが存在しないため、ツールはグレースフルに劣化します — ワークスペースは`TERMINAL_CWD`から読み取られ、スピナーは表示されません。
- **明示的な上書き:** 呼び出し側が`parent_agent=`を明示的に渡した場合、それが尊重され、上書きされません。

これは、プラグインコマンドからのツールディスパッチのための公開された安定インターフェースです。プラグインは`ctx._cli_ref.agent`などのプライベートな状態に手を伸ばすべきではありません。

:::tip
このガイドは**汎用プラグイン**（ツール、フック、スラッシュコマンド、CLIコマンド）を扱います。以下のセクションでは、各専門プラグインタイプの作成パターンを概説します。それぞれがフィールドリファレンスと例について、完全なガイドへのリンクを示しています。
:::

## 専門プラグインタイプ

Hermesには、汎用サーフェスを超えた5つの専門プラグインタイプがあります。それぞれは、`plugins/<category>/<name>/`（bundled）または`~/.hermes/plugins/<category>/<name>/`（user）配下のディレクトリとして提供されます。コントラクトはカテゴリによって異なります。必要なものを選んでから、その完全なガイドを読んでください。

### モデルプロバイダープラグイン — LLMバックエンドを追加する

プロファイルを`plugins/model-providers/<name>/`に配置します。

```python
# plugins/model-providers/acme/__init__.py
from providers import register_provider
from providers.base import ProviderProfile

register_provider(ProviderProfile(
    name="acme",
    aliases=("acme-inference",),
    display_name="Acme Inference",
    env_vars=("ACME_API_KEY", "ACME_BASE_URL"),
    base_url="https://api.acme.example.com/v1",
    auth_type="api_key",
    default_aux_model="acme-small-fast",
    fallback_models=("acme-large-v3", "acme-medium-v3"),
))
```

```yaml
# plugins/model-providers/acme/plugin.yaml
name: acme-provider
kind: model-provider
version: 1.0.0
description: Acme Inference — OpenAI-compatible direct API
```

何かが`get_provider_profile()`または`list_providers()`を初めて呼び出したときに遅延検出されます — `auth.py`、`config.py`、`doctor.py`、`models.py`、`runtime_provider.py`、そしてchat_completionsトランスポートが自動的に結びつきます。ユーザープラグインは、同じ名前のbundledプラグインを上書きします。

**完全なガイド:** [Model Provider Plugins](/docs/developer-guide/model-provider-plugin) — フィールドリファレンス、オーバーライド可能なフック（`prepare_messages`、`build_extra_body`、`build_api_kwargs_extras`、`fetch_models`）、api_modeの選択、認証タイプ、テスト。

### プラットフォームプラグイン — ゲートウェイチャネルを追加する

アダプターを`plugins/platforms/<name>/`に配置します。

```python
# plugins/platforms/myplatform/adapter.py
from gateway.platforms.base import BasePlatformAdapter

class MyPlatformAdapter(BasePlatformAdapter):
    async def connect(self): ...
    async def send(self, chat_id, text): ...
    async def disconnect(self): ...

def check_requirements():
    import os
    return bool(os.environ.get("MYPLATFORM_TOKEN"))

def _env_enablement():
    import os
    tok = os.getenv("MYPLATFORM_TOKEN", "").strip()
    if not tok:
        return None
    return {"token": tok}

def register(ctx):
    ctx.register_platform(
        name="myplatform",
        label="MyPlatform",
        adapter_factory=lambda cfg: MyPlatformAdapter(cfg),
        check_fn=check_requirements,
        required_env=["MYPLATFORM_TOKEN"],
        # env変数から PlatformConfig.extra を自動入力し、env のみのセットアップが
        # SDK のインスタンス化なしで `hermes gateway status` に表示されるようにする。
        env_enablement_fn=_env_enablement,
        # cron 配信にオプトイン: `deliver=myplatform` がこの変数にルーティングされる。
        cron_deliver_env_var="MYPLATFORM_HOME_CHANNEL",
        emoji="💬",
        platform_hint="You are chatting via MyPlatform. Keep responses concise.",
    )
```

```yaml
# plugins/platforms/myplatform/plugin.yaml
name: myplatform-platform
label: MyPlatform
kind: platform
version: 1.0.0
description: MyPlatform gateway adapter
requires_env:
  - name: MYPLATFORM_TOKEN
    description: "Bot token from the MyPlatform console"
    password: true
optional_env:
  - name: MYPLATFORM_HOME_CHANNEL
    description: "Default channel for cron delivery"
    password: false
```

**完全なガイド:** [Adding Platform Adapters](/docs/developer-guide/adding-platform-adapters) — 完全な`BasePlatformAdapter`コントラクト、メッセージルーティング、認証ゲート、セットアップウィザード連携。stdlibのみで動作する例としては`plugins/platforms/irc/`を見てください。

### メモリプロバイダープラグイン — セッションをまたいだ知識バックエンドを追加する

`MemoryProvider`の実装を`plugins/memory/<name>/`に配置します。

```python
# plugins/memory/my-memory/__init__.py
from agent.memory_provider import MemoryProvider

class MyMemoryProvider(MemoryProvider):
    @property
    def name(self) -> str:
        return "my-memory"

    def is_available(self) -> bool:
        import os
        return bool(os.environ.get("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_message, assistant_response, **kwargs) -> None:
        ...

    def prefetch(self, query: str, **kwargs) -> str | None:
        ...

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

メモリプロバイダーは単一選択です — 一度にアクティブになれるのは1つだけで、`config.yaml`の`memory.provider`で選択します。

**完全なガイド:** [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) — 完全な`MemoryProvider` ABC、スレッディングコントラクト、プロファイル分離、`cli.py`経由のCLIコマンド登録。

### コンテキストエンジンプラグイン — コンテキスト圧縮器を置き換える

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def should_compress(self, messages, model) -> bool: ...
    def compress(self, messages, model) -> list[dict]: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

コンテキストエンジンは単一選択です — `config.yaml`の`context.engine`で選択します。

**完全なガイド:** [Context Engine Plugins](/docs/developer-guide/context-engine-plugin)。

### 画像生成バックエンド

プロバイダーを`plugins/image_gen/<name>/`に配置します。

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, **kwargs) -> str: ...   # 画像パスを返す

def register(ctx):
    ctx.register_image_gen_provider(MyImageGenProvider())
```

```yaml
# plugins/image_gen/my-imggen/plugin.yaml
name: my-imggen
kind: backend
version: 1.0.0
description: Custom image generation backend
```

**完全なガイド:** [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) — 完全な`ImageGenProvider` ABC、`list_models()` / `get_setup_schema()`メタデータ、`success_response()`/`error_response()`ヘルパー、base64 vs URL出力、ユーザーによる上書き、pip配布。

**リファレンス例:** `plugins/image_gen/openai/`（OpenAI SDK経由のDALL-E / GPT-Image）、`plugins/image_gen/openai-codex/`、`plugins/image_gen/xai/`（Grok画像生成）。

## Python以外の拡張サーフェス

Hermesは、Pluginではない拡張も受け付けます。これらは[Pluggable interfaces table](/docs/user-guide/features/plugins#pluggable-interfaces--where-to-go-for-each)に示されています。以下のセクションでは、各作成スタイルを簡単に概説します。

### MCPサーバー — 外部ツールを登録する

Model Context Protocol（MCP）サーバーは、Pluginなしで独自のツールをHermesに登録します。`~/.hermes/config.yaml`で宣言します。

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    timeout: 120

  linear:
    url: "https://mcp.linear.app/sse"
    auth:
      type: "oauth"
```

Hermesは起動時に各サーバーに接続し、そのツールを一覧表示し、組み込みツールと並べて登録します。LLMは、それらを他の任意のツールとまったく同じように認識します。**完全なガイド:** [MCP](/docs/user-guide/features/mcp)。

### ゲートウェイイベントフック — ライフサイクルイベントで発火する

マニフェスト + ハンドラを`~/.hermes/hooks/<name>/`に配置します。

```yaml
# ~/.hermes/hooks/long-task-alert/HOOK.yaml
name: long-task-alert
description: Send a push notification when a long task finishes
events:
  - agent:end
```

```python
# ~/.hermes/hooks/long-task-alert/handler.py
async def handle(event_type: str, context: dict) -> None:
    if context.get("duration_seconds", 0) > 120:
        # 通知を送る …
        pass
```

イベントには、`gateway:startup`、`session:start`、`session:end`、`session:reset`、`agent:start`、`agent:step`、`agent:end`、そしてワイルドカードの`command:*`が含まれます。フック内のエラーはキャッチされてログに記録されます — メインパイプラインをブロックすることはありません。

**完全なガイド:** [Gateway Event Hooks](/docs/user-guide/features/hooks#gateway-event-hooks)。

### シェルフック — ツール呼び出し時にシェルコマンドを実行する

ツールが発火したときにスクリプトを実行したいだけの場合（通知、監査ログ、デスクトップアラート、自動フォーマッタ）は、`config.yaml`のシェルフックを使ってください — Pythonは不要です。

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

Pythonプラグインフックと同じすべてのイベント（`pre_tool_call`、`post_tool_call`、`pre_llm_call`、`post_llm_call`、`on_session_start`、`on_session_end`、`pre_gateway_dispatch`）に加えて、`pre_tool_call`のブロッキング判定のための構造化JSON出力をサポートします。

**完全なガイド:** [Shell Hooks](/docs/user-guide/features/hooks#shell-hooks)。

### スキルソース — カスタムスキルレジストリを追加する

スキルのGitHubリポジトリを保守している場合（または組み込みソースを超えてコミュニティインデックスから取得したい場合）は、それを**tap**として追加します。

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

独自のtapを公開するのは、`skills/<skill-name>/SKILL.md`ディレクトリを持つGitHubリポジトリを用意するだけです — サーバーやレジストリへのサインアップは不要です。

**完全なガイド:** [Skills Hub](/docs/user-guide/features/skills#skills-hub) · [カスタムtapの公開](/docs/user-guide/features/skills#publishing-a-custom-skill-tap)（リポジトリレイアウト、最小限の例、非デフォルトパス、信頼レベル）。

### コマンドテンプレート経由のTTS / STT

オーディオやテキストを読み書きする任意のCLIは、`config.yaml`を通じて組み込めます — Pythonコードは不要です。

```yaml
tts:
  provider: voxcpm
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      voice_compatible: true
```

STTについては、`HERMES_LOCAL_STT_COMMAND`をシェルテンプレートに向けてください。サポートされるプレースホルダー: `{input_path}`、`{output_path}`、`{format}`、`{voice}`、`{model}`、`{speed}`（TTS）、`{input_path}`、`{output_dir}`、`{language}`、`{model}`（STT）。パスを扱う任意のCLIが、自動的にプラグインになります。

**完全なガイド:** [TTS custom command providers](/docs/user-guide/features/tts#custom-command-providers) · [STT](/docs/user-guide/features/tts#voice-message-transcription-stt)。

## pip経由で配布する

プラグインを公開して共有するには、Pythonパッケージにエントリポイントを追加します。

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

```bash
pip install hermes-plugin-calculator
# 次回の hermes 起動時にプラグインが自動検出される
```

## NixOS向けに配布する

NixOSユーザーは、エントリポイントを持つ`pyproject.toml`を提供すれば、あなたのプラグインを宣言的にインストールできます。

**エントリポイントプラグイン**（配布に推奨）:
```nix
# ユーザーの configuration.nix
services.hermes-agent.extraPythonPackages = [
  (pkgs.python312Packages.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ pkgs.python312Packages.setuptools ];
  })
];
```

**ディレクトリプラグイン**（`pyproject.toml`不要）:
```nix
services.hermes-agent.extraPlugins = [
  (pkgs.fetchFromGitHub {
    owner = "you";
    repo = "hermes-my-plugin";
    rev = "v1.0.0";
    hash = "sha256-...";
  })
];
```

オーバーレイの使い方や衝突チェックを含む完全なドキュメントについては、[Nix Setup guide](/docs/getting-started/nix-setup#plugins)を参照してください。

## よくある間違い

**ハンドラがJSON文字列を返していない:**
```python
# 誤り — dict を返している
def handler(args, **kwargs):
    return {"result": 42}

# 正しい — JSON 文字列を返している
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**ハンドラのシグネチャに`**kwargs`がない:**
```python
# 誤り — Hermes が追加コンテキストを渡すと壊れる
def handler(args):
    ...

# 正しい
def handler(args, **kwargs):
    ...
```

**ハンドラが例外を投げる:**
```python
# 誤り — 例外が伝播し、ツール呼び出しが失敗する
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# 正しい — キャッチしてエラー JSON を返す
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**スキーマの説明が曖昧すぎる:**
```python
# 悪い — モデルがいつ使えばよいか分からない
"description": "Does stuff"

# 良い — モデルがいつどのように使うか正確に分かる
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
