---
sidebar_position: 11
sidebar_label: "プラグイン"
title: "プラグイン"
description: "プラグインシステムを通じて、カスタムツール、フック、統合で Hermes を拡張する"
---

# プラグイン

Hermes には、コアコードを変更せずにカスタムツール、フック、統合を追加するためのプラグインシステムが
あります。

自分用、チーム用、または 1 つのプロジェクト用にカスタムツールを作成したい場合は、通常これが正しい
道です。開発者ガイドの [ツールの追加](/docs/developer-guide/adding-tools) ページは、`tools/` と
`toolsets.py` に存在する組み込みの Hermes コアツール向けです。

**→ [Hermes プラグインを作る](/docs/guides/build-a-hermes-plugin)** — 完全な動作例を含む
ステップバイステップのガイドです。

## クイック概要

`plugin.yaml` と Python コードを含むディレクトリを `~/.hermes/plugins/` に置くだけです。

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # マニフェスト
├── __init__.py      # register() — スキーマをハンドラに接続する
├── schemas.py       # ツールスキーマ（LLM が見るもの）
└── tools.py         # ツールハンドラ（呼び出されたときに実行されるもの）
```

Hermes を起動すると、あなたのツールが組み込みツールと並んで表示されます。モデルはすぐにそれらを
呼び出せます。

### 最小の動作例

以下は、`hello_world` ツールを追加し、フックを通じてすべてのツール呼び出しをログ出力する完全な
プラグインです。

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Minimal Hermes plugin — registers a tool and a hook."""

import json


def register(ctx):
    # --- Tool: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Returns a friendly greeting for the given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
        description="Return a friendly greeting for the given name.",
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

両方のファイルを `~/.hermes/plugins/hello-world/` に置いて Hermes を再起動すると、モデルはすぐに
`hello_world` を呼び出せます。フックは、すべてのツール呼び出しの後にログ行を出力します。

`./.hermes/plugins/` 配下のプロジェクトローカルなプラグインは、デフォルトで無効です。信頼できる
リポジトリでのみ、Hermes を起動する前に `HERMES_ENABLE_PROJECT_PLUGINS=true` を設定して有効に
してください。

## プラグインができること

以下のすべての `ctx.*` API は、プラグインの `register(ctx)` 関数内で利用できます。

| 機能 | 方法 |
|-----------|-----|
| ツールを追加する | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| フックを追加する | `ctx.register_hook("post_tool_call", callback)` |
| スラッシュコマンドを追加する | `ctx.register_command(name, handler, description)` — CLI とゲートウェイのセッションに `/name` を追加する |
| コマンドからツールをディスパッチする | `ctx.dispatch_tool(name, args)` — 親エージェントのコンテキストを自動接続して登録済みツールを呼び出す |
| CLI コマンドを追加する | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — `hermes <plugin> <subcommand>` を追加する |
| メッセージを注入する | `ctx.inject_message(content, role="user")` — [メッセージの注入](#injecting-messages) を参照 |
| データファイルを同梱する | `Path(__file__).parent / "data" / "file.yaml"` |
| スキルをバンドルする | `ctx.register_skill(name, path)` — `plugin:skill` として名前空間化され、`skill_view("plugin:skill")` 経由で読み込まれる |
| 環境変数でゲートする | plugin.yaml の `requires_env: [API_KEY]` — `hermes plugins install` 中にプロンプトされる |
| pip で配布する | `[project.entry-points."hermes_agent.plugins"]` |
| ゲートウェイプラットフォームを登録する（Discord、Telegram、IRC など） | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — [プラットフォームアダプターの追加](/docs/developer-guide/adding-platform-adapters) を参照 |
| 画像生成バックエンドを登録する | `ctx.register_image_gen_provider(provider)` — [画像生成プロバイダープラグイン](/docs/developer-guide/image-gen-provider-plugin) を参照 |
| コンテキスト圧縮エンジンを登録する | `ctx.register_context_engine(engine)` — [コンテキストエンジンプラグイン](/docs/developer-guide/context-engine-plugin) を参照 |
| メモリバックエンドを登録する | `plugins/memory/<name>/__init__.py` で `MemoryProvider` をサブクラス化する — [メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin) を参照（別の検出システムを使用） |
| ホスト所有の LLM 呼び出しを実行する | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — ユーザーのアクティブなモデル＋認証を借りて、オプションの JSON スキーマ検証付きで一発の補完を行う。[プラグインの LLM アクセス](/docs/developer-guide/plugin-llm-access) を参照 |
| 推論バックエンド（LLM プロバイダー）を登録する | `plugins/model-providers/<name>/__init__.py` で `register_provider(ProviderProfile(...))` — [モデルプロバイダープラグイン](/docs/developer-guide/model-provider-plugin) を参照（別の検出システムを使用） |

## プラグインの検出

| ソース | パス | ユースケース |
|--------|------|----------|
| バンドル | `<repo>/plugins/` | Hermes に同梱 — [組み込みプラグイン](/docs/user-guide/features/built-in-plugins) を参照 |
| ユーザー | `~/.hermes/plugins/` | 個人用プラグイン |
| プロジェクト | `.hermes/plugins/` | プロジェクト固有のプラグイン（`HERMES_ENABLE_PROJECT_PLUGINS=true` が必要） |
| pip | `hermes_agent.plugins` entry_points | 配布パッケージ |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS の宣言的インストール — [Nix セットアップ](/docs/getting-started/nix-setup#plugins) を参照 |

名前が衝突した場合、後のソースが先のソースを上書きします。そのため、バンドルされたプラグインと同じ
名前のユーザープラグインはそれを置き換えます。

### プラグインのサブカテゴリ

各ソース内で、Hermes はプラグインを専門の検出システムにルーティングするサブカテゴリのディレクトリも
認識します。

| サブディレクトリ | 保持する内容 | 検出システム |
|---|---|---|
| `plugins/`（ルート） | 一般プラグイン — ツール、フック、スラッシュコマンド、CLI コマンド、バンドルされたスキル | `PluginManager`（kind: `standalone` または `backend`） |
| `plugins/platforms/<name>/` | ゲートウェイチャネルアダプター（`ctx.register_platform()`） | `PluginManager`（kind: `platform`、1 階層深い） |
| `plugins/image_gen/<name>/` | 画像生成バックエンド（`ctx.register_image_gen_provider()`） | `PluginManager`（kind: `backend`、1 階層深い） |
| `plugins/memory/<name>/` | メモリプロバイダー（`MemoryProvider` をサブクラス化） | `plugins/memory/__init__.py` の**独自ローダー**（kind: `exclusive` — 同時にアクティブなのは 1 つ） |
| `plugins/context_engine/<name>/` | コンテキスト圧縮エンジン（`ctx.register_context_engine()`） | `plugins/context_engine/__init__.py` の**独自ローダー**（同時にアクティブなのは 1 つ） |
| `plugins/model-providers/<name>/` | LLM プロバイダープロファイル（`register_provider(ProviderProfile(...))`） | `providers/__init__.py` の**独自ローダー**（初回の `get_provider_profile()` 呼び出し時に遅延スキャン） |

`~/.hermes/plugins/model-providers/<name>/` と `~/.hermes/plugins/memory/<name>/` にある
ユーザープラグインは、同名のバンドルプラグインを上書きします — `register_provider()` /
`register_memory_provider()` では後勝ちです。ディレクトリを置くだけで、リポジトリを一切編集せずに
組み込みを置き換えます。

## プラグインはオプトイン（いくつかの例外あり）

**一般プラグインとユーザーがインストールしたバックエンドは、デフォルトで無効です** — 検出はそれらを
見つける（そのため `hermes plugins` と `/plugins` に表示される）のですが、プラグインの名前を
`~/.hermes/config.yaml` の `plugins.enabled` に追加するまで、フックやツールを持つものは何も
読み込まれません。これにより、明示的な同意なしにサードパーティのコードが実行されるのを防ぎます。

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # オプションの拒否リスト — 名前が両方に現れた場合は常にこちらが勝つ
    - noisy-plugin
```

状態を切り替える 3 つの方法:

```bash
hermes plugins                    # インタラクティブなトグル（スペースでチェック／チェック解除）
hermes plugins enable <name>      # 許可リストに追加
hermes plugins disable <name>     # 許可リストから削除 ＋ disabled に追加
```

`hermes plugins install owner/repo` の後、`Enable 'name' now? [y/N]` と尋ねられます — デフォルトは
no です。スクリプト化されたインストールでは `--enable` または `--no-enable` でプロンプトをスキップ
します。

### 許可リストがゲートしないもの

いくつかのカテゴリのプラグインは `plugins.enabled` をバイパスします — それらは Hermes の組み込み
サーフェスの一部であり、デフォルトでゲートオフされると基本機能が壊れてしまうためです。

| プラグインの種類 | 代わりにどう有効化されるか |
|---|---|
| **バンドルされたプラットフォームプラグイン**（`plugins/platforms/` 配下の IRC、Teams など） | 自動的に読み込まれ、同梱されたすべてのゲートウェイチャネルが利用可能になります。実際のチャネルは `config.yaml` の `gateway.platforms.<name>.enabled` で有効になります。 |
| **バンドルされたバックエンド**（`plugins/image_gen/` 配下の画像生成プロバイダーなど） | 自動的に読み込まれ、デフォルトのバックエンドが「そのまま動く」ようになります。選択は `config.yaml` の `<category>.provider`（例: `image_gen.provider: openai`）で行われます。 |
| **メモリプロバイダー**（`plugins/memory/`） | すべて検出され、ちょうど 1 つがアクティブになります。`config.yaml` の `memory.provider` で選択されます。 |
| **コンテキストエンジン**（`plugins/context_engine/`） | すべて検出され、1 つがアクティブになります。`config.yaml` の `context.engine` で選択されます。 |
| **モデルプロバイダー**（`plugins/model-providers/`） | `plugins/model-providers/` 配下のバンドルされたすべてのプロバイダーは、最初の `get_provider_profile()` 呼び出し時に検出・登録されます。ユーザーは `--provider` または `config.yaml` で一度に 1 つを選びます。 |
| **pip でインストールされた `backend` プラグイン** | `plugins.enabled` でオプトイン（一般プラグインと同じ）。 |
| **ユーザーがインストールしたプラットフォーム**（`~/.hermes/plugins/platforms/` 配下） | `plugins.enabled` でオプトイン — サードパーティのゲートウェイアダプターには明示的な同意が必要です。 |

要するに、**バンドルされた「常に動く」インフラは自動的に読み込まれ、サードパーティの一般プラグインは
オプトイン**です。`plugins.enabled` の許可リストは、ユーザーが `~/.hermes/plugins/` に置く任意の
コードに特化したゲートです。

### 既存ユーザー向けのマイグレーション

オプトインプラグインを持つバージョンの Hermes（設定スキーマ v21 以降）にアップグレードすると、
`~/.hermes/plugins/` 配下にすでにインストールされていて `plugins.disabled` に含まれていなかった
ユーザープラグインは、**自動的に `plugins.enabled` に祖父継承（grandfather）されます**。既存の設定は
そのまま動作し続けます。バンドルされたスタンドアロンプラグインは祖父継承されません — 既存ユーザーで
あっても明示的にオプトインする必要があります。（バンドルされたプラットフォーム／バックエンド
プラグインは決してゲートされなかったため、祖父継承は不要でした。）

## 利用可能なフック

プラグインは、これらのライフサイクルイベントに対してコールバックを登録できます。完全な詳細、
コールバックのシグネチャ、例については **[イベントフックのページ](/docs/user-guide/features/hooks#plugin-hooks)** を
参照してください。

| フック | 発火するタイミング |
|------|-----------|
| [`pre_tool_call`](/docs/user-guide/features/hooks#pre_tool_call) | 任意のツールが実行される前 |
| [`post_tool_call`](/docs/user-guide/features/hooks#post_tool_call) | 任意のツールが返った後 |
| [`pre_llm_call`](/docs/user-guide/features/hooks#pre_llm_call) | ターンごとに 1 回、LLM ループの前 — `{"context": "..."}` を返して [ユーザーメッセージにコンテキストを注入](/docs/user-guide/features/hooks#pre_llm_call) できる |
| [`post_llm_call`](/docs/user-guide/features/hooks#post_llm_call) | ターンごとに 1 回、LLM ループの後（成功したターンのみ） |
| [`on_session_start`](/docs/user-guide/features/hooks#on_session_start) | 新しいセッションが作成された（最初のターンのみ） |
| [`on_session_end`](/docs/user-guide/features/hooks#on_session_end) | すべての `run_conversation` 呼び出しの終了 ＋ CLI 終了ハンドラ |
| [`on_session_finalize`](/docs/user-guide/features/hooks#on_session_finalize) | CLI／ゲートウェイがアクティブなセッションを破棄する（`/new`、GC、CLI 終了） |
| [`on_session_reset`](/docs/user-guide/features/hooks#on_session_reset) | ゲートウェイが新しいセッションキーに切り替える（`/new`、`/reset`、`/clear`、アイドルローテーション） |
| [`subagent_stop`](/docs/user-guide/features/hooks#subagent_stop) | `delegate_task` 終了後、子ごとに 1 回 |
| [`pre_gateway_dispatch`](/docs/user-guide/features/hooks#pre_gateway_dispatch) | ゲートウェイがユーザーメッセージを受信、認証＋ディスパッチの前。`{"action": "skip" \| "rewrite" \| "allow", ...}` を返してフローに影響を与えられる。 |

## プラグインの種類

Hermes には 4 種類のプラグインがあります。

| 種類 | 何をするか | 選択方法 | 場所 |
|------|-------------|-----------|----------|
| **一般プラグイン** | ツール、フック、スラッシュコマンド、CLI コマンドを追加する | 複数選択（有効／無効） | `~/.hermes/plugins/` |
| **メモリプロバイダー** | 組み込みメモリを置き換えるか拡張する | 単一選択（1 つがアクティブ） | `plugins/memory/` |
| **コンテキストエンジン** | 組み込みのコンテキスト圧縮器を置き換える | 単一選択（1 つがアクティブ） | `plugins/context_engine/` |
| **モデルプロバイダー** | 推論バックエンドを宣言する（OpenRouter、Anthropic など） | 複数登録、`--provider` / `config.yaml` で選択 | `plugins/model-providers/` |

メモリプロバイダーとコンテキストエンジンは**プロバイダープラグイン**です — 各種類のうち、同時に
アクティブにできるのは 1 つだけです。モデルプロバイダーもプラグインですが、多数が同時に読み込まれ、
ユーザーは `--provider` または `config.yaml` で一度に 1 つを選びます。一般プラグインは任意の組み合わせ
で有効にできます。

## プラグイン可能なインターフェース — それぞれの行き先

上の表は 4 つのプラグインカテゴリを示していますが、「一般プラグイン」の中で `PluginContext` は
いくつかの異なる拡張ポイントを公開しています。さらに Hermes は、Python プラグインシステムの外にある
拡張（設定駆動のバックエンド、シェルフックのコマンド、外部サーバーなど）も受け入れます。作りたいものに
対する正しいドキュメントを見つけるには、この表を使ってください。

| 追加したいもの… | 方法 | 作成ガイド |
|---|---|---|
| LLM が呼び出せる**ツール** | Python プラグイン — `ctx.register_tool()` | [Hermes プラグインを作る](/docs/guides/build-a-hermes-plugin) · [ツールの追加](/docs/developer-guide/adding-tools) |
| **ライフサイクルフック**（LLM の前後、セッションの開始／終了、ツールフィルタ） | Python プラグイン — `ctx.register_hook()` | [フックリファレンス](/docs/user-guide/features/hooks) · [Hermes プラグインを作る](/docs/guides/build-a-hermes-plugin) |
| CLI／ゲートウェイ向けの**スラッシュコマンド** | Python プラグイン — `ctx.register_command()` | [Hermes プラグインを作る](/docs/guides/build-a-hermes-plugin) · [CLI の拡張](/docs/developer-guide/extending-the-cli) |
| `hermes <thing>` 向けの**サブコマンド** | Python プラグイン — `ctx.register_cli_command()` | [CLI の拡張](/docs/developer-guide/extending-the-cli) |
| プラグインが同梱する**スキル** | Python プラグイン — `ctx.register_skill()` | [スキルの作成](/docs/developer-guide/creating-skills) |
| **推論バックエンド**（LLM プロバイダー: OpenAI 互換、Codex、Anthropic-Messages、Bedrock） | プロバイダープラグイン — `plugins/model-providers/<name>/` で `register_provider(ProviderProfile(...))` | **[モデルプロバイダープラグイン](/docs/developer-guide/model-provider-plugin)** · [プロバイダーの追加](/docs/developer-guide/adding-providers) |
| **ゲートウェイチャネル**（Discord / Telegram / IRC / Teams など） | プラットフォームプラグイン — `plugins/platforms/<name>/` で `ctx.register_platform()` | [プラットフォームアダプターの追加](/docs/developer-guide/adding-platform-adapters) |
| **メモリバックエンド**（Honcho、Mem0、Supermemory など） | メモリプラグイン — `plugins/memory/<name>/` で `MemoryProvider` をサブクラス化 | [メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin) |
| **コンテキスト圧縮戦略** | コンテキストエンジンプラグイン — `ctx.register_context_engine()` | [コンテキストエンジンプラグイン](/docs/developer-guide/context-engine-plugin) |
| **画像生成バックエンド**（DALL·E、SDXL など） | バックエンドプラグイン — `ctx.register_image_gen_provider()` | [画像生成プロバイダープラグイン](/docs/developer-guide/image-gen-provider-plugin) |
| **TTS バックエンド**（任意の CLI — Piper、VoxCPM、Kokoro、xtts、音声クローンスクリプトなど） | 設定駆動 — `config.yaml` で `tts.providers.<name>` に `type: command` で宣言 | [TTS セットアップ](/docs/user-guide/features/tts#custom-command-providers) |
| **STT バックエンド**（カスタムの whisper バイナリ、ローカル ASR CLI） | 設定駆動 — `HERMES_LOCAL_STT_COMMAND` 環境変数をシェルテンプレートに設定 | [音声メッセージの文字起こし（STT）](/docs/user-guide/features/tts#voice-message-transcription-stt) |
| **MCP 経由の外部ツール**（filesystem、GitHub、Linear、Notion、任意の MCP サーバー） | 設定駆動 — `config.yaml` で `mcp_servers.<name>` を `command:` / `url:` で宣言。Hermes はサーバーのツールを自動検出し、組み込みと並べて登録します。 | [MCP](/docs/user-guide/features/mcp) |
| **追加のスキルソース**（カスタム GitHub リポジトリ、プライベートなスキルインデックス） | CLI — `hermes skills tap add <repo>` | [スキルハブ](/docs/user-guide/features/skills#skills-hub) · [カスタムタップの公開](/docs/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **ゲートウェイイベントフック**（`gateway:startup`、`session:start`、`agent:end`、`command:*` で発火） | `HOOK.yaml` ＋ `handler.py` を `~/.hermes/hooks/<name>/` に置く | [イベントフック](/docs/user-guide/features/hooks#gateway-event-hooks) |
| **シェルフック**（イベント発生時にシェルコマンドを実行 — 通知、監査ログ、デスクトップ通知） | 設定駆動 — `config.yaml` で `hooks:` 配下に宣言 | [シェルフック](/docs/user-guide/features/hooks#shell-hooks) |

:::note
すべてが Python プラグインというわけではありません。一部の拡張サーフェスは、意図的に**設定駆動の
シェルコマンド**（TTS、STT、シェルフック）を使うので、すでに持っている任意の CLI が Python を
書かずにプラグインになります。他のもの（MCP）は**外部サーバー**で、エージェントが接続してそこから
ツールを自動登録します。そしていくつか（ゲートウェイフック）は、独自のマニフェスト形式を持つ
**ドロップインのディレクトリ**です。ユースケースに合った統合スタイルに対して、正しいサーフェスを
選んでください。上の表の作成ガイドはそれぞれ、プレースホルダー、検出、例を扱っています。
:::

## NixOS の宣言的プラグイン

NixOS では、モジュールのオプションを通じてプラグインを宣言的にインストールできます — `hermes
plugins install` は不要です。完全な詳細は **[Nix セットアップガイド](/docs/getting-started/nix-setup#plugins)** を
参照してください。

```nix
services.hermes-agent = {
  # Directory plugin (source tree with plugin.yaml)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin (pip package)
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # Enable in config
  settings.plugins.enabled = [ "my-plugin" ];
};
```

宣言的プラグインは `nix-managed-` プレフィックス付きでシンボリックリンクされます — 手動で
インストールしたプラグインと共存し、Nix の設定から削除されると自動的にクリーンアップされます。

## プラグインの管理

```bash
hermes plugins                               # 統合インタラクティブ UI
hermes plugins list                          # テーブル: enabled / disabled / not enabled
hermes plugins install user/repo             # Git からインストールし、その後 Enable? [y/N] とプロンプト
hermes plugins install user/repo --enable    # インストールして有効化（プロンプトなし）
hermes plugins install user/repo --no-enable # インストールするが無効のまま（プロンプトなし）
hermes plugins update my-plugin              # 最新を取得
hermes plugins remove my-plugin              # アンインストール
hermes plugins enable my-plugin              # 許可リストに追加
hermes plugins disable my-plugin             # 許可リストから削除 ＋ disabled に追加
```

### インタラクティブ UI

引数なしで `hermes plugins` を実行すると、複合的なインタラクティブ画面が開きます。

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **General Plugins セクション** — チェックボックス、SPACE でトグル。チェック済み ＝
  `plugins.enabled` にある、チェックなし ＝ `plugins.disabled` にある（明示的にオフ）。
- **Provider Plugins セクション** — 現在の選択を表示します。ENTER を押すとラジオピッカーに入り、
  アクティブなプロバイダーを 1 つ選びます。
- バンドルプラグインは、`[bundled]` タグ付きで同じ一覧に表示されます。

プロバイダープラグインの選択は `config.yaml` に保存されます。

```yaml
memory:
  provider: "honcho"      # 空文字列 = 組み込みのみ

context:
  engine: "compressor"    # デフォルトの組み込み compressor
```

### enabled vs. disabled vs. どちらでもない

プラグインは次の 3 つの状態のいずれかになります。

| 状態 | 意味 | `plugins.enabled` にある? | `plugins.disabled` にある? |
|---|---|---|---|
| `enabled` | 次のセッションで読み込まれる | はい | いいえ |
| `disabled` | 明示的にオフ — `enabled` にも含まれていても読み込まれない | （無関係） | はい |
| `not enabled` | 検出されたがオプトインされていない | いいえ | いいえ |

新しくインストールされた、またはバンドルされたプラグインのデフォルトは `not enabled` です。
`hermes plugins list` は 3 つの異なる状態をすべて表示するので、明示的にオフにされたものと、有効化を
待っているだけのものを区別できます。

実行中のセッションでは、`/plugins` が現在読み込まれているプラグインを表示します。

## メッセージの注入 {#injecting-messages}

プラグインは、`ctx.inject_message()` を使ってアクティブな会話にメッセージを注入できます。

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**シグネチャ:** `ctx.inject_message(content: str, role: str = "user") -> bool`

仕組み:

- エージェントが**アイドル**（ユーザー入力待ち）の場合、メッセージは次の入力としてキューに入れられ、
  新しいターンを開始します。
- エージェントが**ターンの途中**（実行中）の場合、メッセージは現在の操作を中断します — ユーザーが
  新しいメッセージを入力して Enter を押すのと同じです。
- `"user"` 以外のロールの場合、内容には `[role]`（例: `[system] ...`）が前置きされます。
- メッセージが正常にキューに入れられた場合は `True` を、CLI 参照が利用できない場合（例:
  ゲートウェイモード）は `False` を返します。

これにより、リモートコントロールのビューア、メッセージングブリッジ、Webhook レシーバーといった
プラグインが、外部ソースから会話にメッセージを供給できるようになります。

:::note
`inject_message` は CLI モードでのみ利用できます。ゲートウェイモードでは CLI 参照がなく、メソッドは
`False` を返します。
:::

ハンドラの契約、スキーマの形式、フックの挙動、エラー処理、よくある間違いについては、
**[完全なガイド](/docs/guides/build-a-hermes-plugin)** を参照してください。
