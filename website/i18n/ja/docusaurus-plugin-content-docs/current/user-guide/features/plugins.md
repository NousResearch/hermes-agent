---
sidebar_position: 11
sidebar_label: "プラグイン"
title: "プラグイン"
description: "プラグインシステムを通じて、カスタムツール、フック、統合で Hermes を拡張する"
---

# プラグイン

Hermes には、コアコードを変更せずにカスタムツール、フック、統合を追加するためのプラグインシステムがあります。

自分用、チーム用、または 1 つのプロジェクト用にカスタムツールを作成したい場合は、通常これが正しい道です。開発者ガイドの [ツールの追加](/docs/developer-guide/adding-tools) ページは、`tools/` と `toolsets.py` に存在する組み込みの Hermes コアツール向けです。

**→ [Hermes プラグインを作る](../../developer-guide/plugins/index.md)** — 完全な動作例を含むステップバイステップのガイドです。

## クイック概要

`plugin.yaml` と Python コードを含むディレクトリを `~/.hermes/plugins/` に置くだけです。

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # manifest
├── __init__.py      # register() — wires schemas to handlers
├── schemas.py       # tool schemas (what the LLM sees)
└── tools.py         # tool handlers (what runs when called)
```

Hermes を起動すると、あなたのツールが組み込みツールと並んで表示されます。モデルはすぐにそれらを呼び出せます。

### 最小の動作例

以下は、`hello_world` ツールを追加し、フックを通じてすべてのツール呼び出しをログ出力する完全なプラグインです。

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
provides_tools:
  - hello_world
provides_hooks:
  - post_tool_call
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
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, args, result, **kwargs):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

両方のファイルを `~/.hermes/plugins/hello-world/` に置いて Hermes を再起動すると、モデルはすぐに `hello_world` を呼び出せます。フックは、すべてのツール呼び出しの後にログ行を出力します。

モデル向けのツール説明は `schema["description"]` に書きます。オプションの `ctx.register_tool(description=...)` の値は別物で、`ToolEntry` レジストリのメタデータです。省略するとスキーマの説明がデフォルトになりますが、Hermes は `description` を持たないスキーマにこの値をコピーし戻すことはしません。テキストはスキーマで一度だけ定義することを推奨します。両方の値を指定する場合は同期させてください。モデルが見るのはスキーマの値です。

### プロジェクトローカルなプラグイン（`./.hermes/plugins/`） {#project-local-plugins}

ゲートウェイはプロジェクトローカルなプラグインを読み込みません。すべての Hermes セッション（`hermes`、`hermes --tui`、Desktop アプリ、メッセージング、cron）は 1 つのゲートウェイ内で実行され、ゲートウェイはプロセス全体に対して一度だけプラグインを検出し、セッションを開始したフォルダーから検出することはありません。そのため `HERMES_ENABLE_PROJECT_PLUGINS=true` を設定してもセッションには効果がありません。ゲートウェイのマシン上の CLI または TUI セッションが、この変数を設定した状態で `./.hermes/plugins/` にプラグインがあるフォルダーで開始されると、Hermes はそのセッションの開始時に警告を表示し、同じ行を `gateway.log` に書き込みます。

プラグインを使うには、ユーザープラグインとしてインストールし（`~/.hermes/plugins/` または `hermes plugins install`）、`hermes plugins enable <name>` で有効にしてください。

## プラグインができること

以下のすべての `ctx.*` API は、プラグインの `register(ctx)` 関数内で利用できます。

| 機能 | 方法 |
|-----------|-----|
| ツールを追加する | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| フックを追加する | `ctx.register_hook("post_tool_call", callback)` |
| スラッシュコマンドを追加する | `ctx.register_command(name, handler, description)` — CLI とゲートウェイのセッションに `/name` を追加する |
| コマンドからツールをディスパッチする | `ctx.dispatch_tool(name, args)` — 親エージェントのコンテキストを自動接続して登録済みツールを呼び出す |
| CLI コマンドを追加する | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — `hermes <plugin> <subcommand>` を追加する |
| メッセージを注入する | `ctx.inject_message(content, role="user", session_key=... \| origin=...)` - [メッセージの注入](#injecting-messages) を参照 |
| ステータスメッセージを投稿・編集する | `handle = ctx.post_message(text)` の後に `handle.edit(text)` — [ライブメッセージの投稿](#posting-a-live-message) を参照 |
| データファイルを同梱する | `Path(__file__).parent / "data" / "file.yaml"` |
| スキルをバンドルする | `ctx.register_skill(name, path)` — `plugin:skill` として名前空間化され、`skill_view("plugin:skill")` 経由で読み込まれる |
| 環境変数でゲートする | plugin.yaml の `requires_env: [API_KEY]` — `hermes plugins install` 中にプロンプトされる |
| pip で配布する | `[project.entry-points."hermes_agent.plugins"]` |
| ゲートウェイプラットフォームを登録する（Discord、Telegram、IRC など） | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — [プラットフォームアダプターの追加](/docs/developer-guide/adding-platform-adapters) を参照 |
| 画像生成バックエンドを登録する | `ctx.register_image_gen_provider(provider)` — [画像生成プロバイダープラグイン](/docs/developer-guide/image-gen-provider-plugin) を参照 |
| 委任戦略を登録する（各 `delegate_task` サブエージェントのルーティング方法） | `ctx.register_delegation_provider(provider)` — [委任プロバイダープラグイン](/docs/developer-guide/delegation-provider-plugin) を参照 |
| 動画生成バックエンドを登録する | `ctx.register_video_gen_provider(provider)` — [動画生成プロバイダープラグイン](/docs/developer-guide/video-gen-provider-plugin) を参照 |
| コンテキスト圧縮エンジンを登録する | `ctx.register_context_engine(engine)` — [コンテキストエンジンプラグイン](/docs/developer-guide/context-engine-plugin) を参照 |
| ターミナル実行バックエンド（クラウドサンドボックス）を登録する | `ctx.register_terminal_environment_provider(provider)` — [ターミナル環境プラグイン](/docs/developer-guide/terminal-environment-plugin) を参照 |
| コンピューター操作ドライバーを登録する | `ctx.register_computer_use_provider(provider)` — `computer_use.backend` で選択。[コンピューター操作バックエンドプラグイン](../../developer-guide/plugins/index.md#computer-use-backend-plugins) を参照 |
| 人間による承認プロンプトをルーティングする | `ctx.register_approval_transport(name, present_fn)` — [承認トランスポート](#approval-transports) を参照 |
| メモリバックエンドを登録する | `plugins/memory/<name>/__init__.py` で `MemoryProvider` をサブクラス化する — [メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin) を参照（別の検出システムを使用） |
| ホスト所有の LLM 呼び出しを実行する | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — ユーザーのアクティブなモデル＋認証を借りて、オプションの JSON スキーマ検証付きで一発の補完を行う。[プラグインの LLM アクセス](/docs/developer-guide/plugin-llm-access) を参照 |
| MCP ツールを呼び出す（ケイパビリティでゲート） | `ctx.call_mcp(server, tool, arguments, timeout=30)` — [プラグインから MCP サーバーを呼び出す](#calling-mcp-servers-from-plugins) を参照 |
| 推論バックエンド（LLM プロバイダー）を登録する | `plugins/model-providers/<name>/__init__.py` で `register_provider(ProviderProfile(...))` — [モデルプロバイダープラグイン](/docs/developer-guide/model-provider-plugin) を参照（別の検出システムを使用） |

:::note スラッシュコマンドのハンドラはゲートウェイで実行される
`ctx.register_command()` で登録した `/command` は、セッションを所有するゲートウェイプロセス内で、プラグインのフック、ツール、MCP サーバーと並んで実行されます。従来の `hermes chat` CLI で入力したコマンドも含みます。そのため、ハンドラは次のようにすべきです。

- `print()` を呼ぶ代わりに**テキストを返す**。出力された内容はユーザーのチャットではなく、ゲートウェイの stdout に送られます。
- **セッションの作業ディレクトリは `agent.runtime_cwd.resolve_agent_cwd()` で読む**。`os.getcwd()` はゲートウェイ自身のディレクトリです。ハンドラの実行中、ゲートウェイはセッションのディレクトリを固定します。
- **呼び出し元のセッションをハンドラのキーワード引数として受け取る**: `session_id`、`session_key`、`platform`、`source`（送信者の `SessionSource`）を宣言すると Hermes が渡します。引数 1 つのハンドラは変更なしで動作し続けます。同じ値がハンドラの周囲で `HERMES_SESSION_*` としてバインドされるので（`os.environ` ではなく `from gateway.session_context import get_session_env` で読む）、ハンドラが開始したバックグラウンドジョブはそのセッションを親として記録します。

従来の CLI は、ハンドラが実行されている間ずっと応答を待ちます。Ctrl+C で止まるのは待機であってハンドラではありません。ハンドラはゲートウェイで完了し、その応答は表示されません。コマンドはゲートウェイへの独自の接続上で実行されるので、実行中もセッションのプロンプト、割り込み、その他のコマンドが止められることはありません。

[スラッシュコマンドの登録](../../developer-guide/plugins/index.md#register-slash-commands) を参照してください。
:::

## プラグインの検出

| ソース | パス | ユースケース |
|--------|------|----------|
| バンドル | `<repo>/plugins/` | Hermes に同梱 — [組み込みプラグイン](./built-in-plugins.md) を参照 |
| ユーザー | `~/.hermes/plugins/` | 個人用プラグイン |
| プロジェクト | `.hermes/plugins/` | セッションを実行するゲートウェイには読み込まれない。[プロジェクトローカルなプラグイン](#project-local-plugins) を参照 |
| pip | `hermes_agent.plugins` entry_points | 配布パッケージ |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS の宣言的インストール — [Nix セットアップ](/docs/getting-started/nix-setup#plugins) を参照 |

名前が衝突した場合、後のソースが先のソースを上書きします。そのため、バンドルされたプラグインと同じ名前のユーザープラグインはそれを置き換えます。

### プラグインのサブカテゴリ

各ソース内で、Hermes はプラグインを専門の検出システムにルーティングするサブカテゴリのディレクトリも認識します。

| サブディレクトリ | 保持する内容 | 検出システム |
|---|---|---|
| `plugins/`（ルート） | 一般プラグイン — ツール、フック、スラッシュコマンド、CLI コマンド、バンドルされたスキル | `PluginManager`（kind: `standalone` または `backend`） |
| `plugins/platforms/<name>/` | ゲートウェイチャネルアダプター（`ctx.register_platform()`） | `PluginManager`（kind: `platform`、1 階層深い） |
| `plugins/image_gen/<name>/` | 画像生成バックエンド（`ctx.register_image_gen_provider()`） | `PluginManager`（kind: `backend`、1 階層深い） |
| `plugins/delegation/<name>/` | 委任戦略（`ctx.register_delegation_provider()`）。`default` は現在のルーティング | `PluginManager`（kind: `backend`、1 階層深い） |
| `plugins/memory/<name>/` | メモリプロバイダー（`MemoryProvider` をサブクラス化） | `plugins/memory/__init__.py` の**独自ローダー**（kind: `exclusive` — 同時にアクティブなのは 1 つ） |
| `plugins/context_engine/<name>/` | コンテキスト圧縮エンジン（`ctx.register_context_engine()`） | `plugins/context_engine/__init__.py` の**独自ローダー**（同時にアクティブなのは 1 つ） |
| `plugins/model-providers/<name>/` | LLM プロバイダープロファイル（`register_provider(ProviderProfile(...))`） | `providers/__init__.py` の**独自ローダー**（初回の `get_provider_profile()` 呼び出し時に遅延スキャン） |

`~/.hermes/plugins/model-providers/<name>/` にあるユーザープラグインは、同名のバンドルされたモデルプロバイダーを上書きします（`register_provider()` では後勝ち）。そのため、リポジトリを一切編集せずに組み込みのプロバイダープロファイルを置き換えられます。メモリプロバイダーは逆の順序で解決されます。`~/.hermes/plugins/memory/<name>/` では名前が衝突すると**バンドル**のプロバイダーが勝つ（バンドル、ユーザー、プロジェクト、エントリポイントの順で、最初に見つかったものが勝つ）ため、ユーザーのメモリプロバイダーには独自の一意な名前が必要です。

## プラグインはオプトイン（いくつかの例外あり）

**一般プラグインとユーザーがインストールしたバックエンドは、デフォルトで無効です** — 検出はそれらを見つける（そのため `hermes plugins` と `/plugins` に表示される）のですが、プラグインの名前を `~/.hermes/config.yaml` の `plugins.enabled` に追加するまで、フックやツールを持つものは何も読み込まれません。これにより、明示的な同意なしにサードパーティのコードが実行されるのを防ぎます。

:::note `plugins.enabled` が管理するのはプラグインだけ
`~/.hermes/hooks/<name>/` 配下の [ゲートウェイイベントフック](./hooks.md#gateway-event-hooks) はプラグインではなく、`plugins.enabled` や `plugins.disabled` でゲートされることは**ありません**。このディレクトリは配置によって信頼されます。有効な `HOOK.yaml` ＋ `handler.py` を含むサブディレクトリはすべて起動時にゲートウェイがインポートし、そこにファイルを置くことがオプトインになります。[ゲートウェイフックの信頼モデル](./hooks.md#gateway-hook-trust) を参照してください。
:::

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # optional deny-list — always wins if a name appears in both
    - noisy-plugin
  # Optional: deadline (seconds) for each Git clone, fetch or checkout
  # during plugin installation, including automatic memory-provider migration.
  # Default 300; values above 3600 are clamped. A subdirectory install
  # (owner/repo/path/to/plugin) downloads only that folder's files.
  clone_timeout_seconds: 300
  # Optional: wall-clock cap (seconds) for timeout-bounded in-process Python
  # plugin hook callbacks (hot-path observers + pre_tool_call). Default 30;
  # set 0 to disable; values above 600 are clamped. Timed-out pre_tool_call
  # callbacks fail closed (block the tool). Caller-thread hooks such as
  # subagent_stop are never moved onto a timeout worker.
  # Shell hooks keep their own per-entry timeout under the top-level hooks: key.
  hook_callback_timeout: 30
  # Optional: deadline (seconds) for one plugin's import + register() at load.
  # A plugin that overruns it is skipped with the reason "load timed out after
  # Ns" (reported like any other load failure: the startup warning and the
  # in-session `/plugins` listing) and the remaining plugins keep loading; the
  # stuck thread is abandoned. Default 10; set 0 to disable; values above 600
  # are clamped.
  load_timeout_seconds: 10
```

状態を切り替える 3 つの方法:

```bash
hermes plugins                    # interactive toggle (space to check/uncheck)
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove from allow-list + add to disabled
```

ツールの選択を保存している場合（`hermes tools` により `platform_toolsets` にツールセットが明示的に列挙されている場合）、プラグインを無効にするとそのツールセットも保存済みのリストから外れ、有効にすると戻ります。CLI と Desktop/TUI/ダッシュボードのスイッチは同じ挙動です。保存された選択がない場合、リストには手を加えません。プラグインのツールセットはデフォルトでオンです。

`hermes plugins install owner/repo` の後、`Enable 'name' now? [y/N]` と尋ねられます — デフォルトは no です。スクリプト化されたインストールでは `--enable` または `--no-enable` でプロンプトをスキップします。メモリプロバイダー（`__init__.py` が `MemoryProvider` を登録するプラグイン）の場合は、代わりに `Use 'name' as the memory provider now?` と尋ねられます。yes（または `--enable`）は `memory.provider` を設定します。これがプロバイダーを有効化する唯一のスイッチです。no の場合は `hermes memory setup` に任されます。

再現可能なインストールのためには、完全で不変なコミットを固定してください（タグ、ブランチ、短縮 SHA は受け付けられません）。

```bash
hermes plugins install owner/repo --ref 0123456789abcdef0123456789abcdef01234567
```

Hermes はそのコミットを detached でチェックアウトし、`HEAD` が要求された SHA と完全に一致することを検証し、正規のソース、インストールされたリビジョン、固定状態を現在のプロファイルに記録します。`hermes plugins update` は固定されたプラグインを動かすことを拒否します。新しいコミットは `hermes plugins install <source> --force --ref <new-commit>` で明示的に指定してください。更新と同様に、プラグインのインストール元と同じソースからの強制再インストールはコードを置き換えますが、あなたのファイルは保持します。追跡されていないファイルと git で無視されたファイルはその場に残り、追跡されているファイルへの編集は `~/.hermes/plugins-backup/<name>-<sha>/` にコピーされます。別のソースからの再インストールはクリーンな状態から始まります。プラグインを完全にリセットするには、先に `hermes plugins remove` してください。プロファイルローカルのインストールメタデータには、設定値、環境変数の値、シークレット、ケイパビリティの付与は含まれません。

同じ固定は Hermes Desktop でも利用できます。**Skills → Plugins → Install from Git** には完全な 40 文字の SHA を受け付ける *Pin to commit* フィールドがあり、プラグイン一覧では固定されたインストールすべてに `pinned @ <sha8>` バッジが表示されるので、チーム全員が同じコミットを実行していることを確認できます。`hermes plugins list` は Source 列に固定を表示します（`git pinned@<sha8>`）。固定は、以下で説明する保存済みの認証情報を通じて、プライベートリポジトリでも機能します。

### プライベートリポジトリからのインストール

`hermes plugins install` は非対話的にクローンする（ユーザー名やパスワードを尋ねることはない）ため、プライベートリポジトリには Hermes が自力で見つけられる認証情報が必要です。すべてのクローン、固定された `--ref` のフェッチ、`hermes plugins update` のプルは、まず匿名で試行されます — 公開リポジトリにはあなたの認証情報が送られることはないので、古いトークンや失効したトークンが公開インストールを壊すことはありません。リモートが匿名アクセスを拒否した場合にのみ、Hermes は認証情報を探します。`https://` のソースでは、次の順で試します。

1. `.env` の `GITHUB_TOKEN` または `GH_TOKEN`（GitHub ホストのみ）。
2. `gh` CLI のログイン（`gh auth login`）、GitHub ホストのみ。
3. そのホスト用の git 認証ヘルパー（`git credential fill`） — 認証情報がすでに保存されていれば、GitLab、Bitbucket、セルフホストのサーバーでも動作します。

認証情報は、そのインストールまたは更新のための一回限りの HTTP ヘッダーとして送られます。プラグインの `.git/config` やインストールメタデータに書き込まれることはありません。SSH のソース（`git@host:owner/repo.git`）は従来どおり ssh-agent で認証します。同じ解決方法が `hermes plugins update`、git からのカタログ MCP インストール、git URL から取得されるプロファイルディストリビューションにも適用されます。

`hermes doctor` は設定された `GITHUB_TOKEN`/`GH_TOKEN` を `api.github.com` に送り（**API Connectivity** の下）、GitHub がそれを拒否した場合は、期限切れのトークンを含む変数名と `.env` ファイルを示すので、削除または置き換えられます。

### 許可リストがゲートしないもの

いくつかのカテゴリのプラグインは `plugins.enabled` をバイパスします — それらは Hermes の組み込みサーフェスの一部であり、デフォルトでゲートオフされると基本機能が壊れてしまうためです。

| プラグインの種類 | 代わりにどう有効化されるか |
|---|---|
| **バンドルされたプラットフォームプラグイン**（`plugins/platforms/` 配下の IRC、Teams など） | 自動的に読み込まれ、同梱されたすべてのゲートウェイチャネルが利用可能になります。実際のチャネルは `config.yaml` の `gateway.platforms.<name>.enabled` で有効になります。 |
| **バンドルされたバックエンド**（`plugins/image_gen/` 配下の画像生成プロバイダーなど） | 自動的に読み込まれ、デフォルトのバックエンドが「そのまま動く」ようになります。選択は `config.yaml` の `<category>.provider`（例: `image_gen.provider: openai`）で行われます。 |
| **メモリプロバイダー**（`plugins/memory/`） | すべて検出され、ちょうど 1 つがアクティブになります。`config.yaml` の `memory.provider` で選択されます。 |
| **コンテキストエンジン**（`plugins/context_engine/`） | すべて検出され、1 つがアクティブになります。`config.yaml` の `context.engine` で選択されます。 |
| **モデルプロバイダー**（`plugins/model-providers/`） | `plugins/model-providers/` 配下のバンドルされたすべてのプロバイダーは、最初の `get_provider_profile()` 呼び出し時に検出・登録されます。ユーザーは `--provider` または `config.yaml` で一度に 1 つを選びます。 |
| **pip でインストールされた `backend` プラグイン** | `plugins.enabled` でオプトイン（一般プラグインと同じ）。 |
| **ユーザーがインストールしたプラットフォーム**（`~/.hermes/plugins/platforms/` 配下） | `plugins.enabled` でオプトイン — サードパーティのゲートウェイアダプターには明示的な同意が必要です。 |

要するに、**バンドルされた「常に動く」インフラは自動的に読み込まれ、サードパーティの一般プラグインはオプトイン**です。`plugins.enabled` の許可リストは、ユーザーが `~/.hermes/plugins/` に置く任意のコードに特化したゲートです。

### 承認トランスポート {#approval-transports}

承認トランスポートは、既存の Hermes のツール承認リクエストを**人間がどこで見て応答するか**を変更します。コマンドに承認が必要かどうかを決めるものではなく、認可ポリシーの API でもありません。

```python
def present(request):
    # Deliver request.command and request.description to your UI, wait for
    # its authenticated human response, then return a request-bound decision.
    choice = send_to_my_ui_and_wait(request)  # once/session/always/deny
    return request.respond(choice)


def register(ctx):
    ctx.register_approval_transport("my-ui", present)
```

`present` は同期でも非同期でもかまいません。Hermes はこれを上限付きのワーカーで実行し、プラグインが守らなくても正規の `approvals.timeout` を強制します。リクエストは不変で、マスク済みの表示テキスト、ホストの表示クラス（`cli` または `gateway`）、ホストのタイムアウト、許可された選択肢、不透明なリクエスト ID/ダイジェストを含みます。`request.respond(choice)` の結果を返してください。バインドされていない辞書や、古いまたは変更されたリクエスト ID/ダイジェストは拒否されます。プラグインはホストが提示しなかったスコープを返すことはできません（例えば、一回限りのリクエストに対する `always`）。

登録だけでは何も起こりません。プラグインの有効化と、そのトランスポートの明示的な選択は、別々の同意ステップです。

```yaml
plugins:
  enabled: [my-approval-plugin]

security:
  approval:
    transport: my-ui
    transport_fallback: deny     # default
```

トランスポートの例外、タイムアウト、利用できない登録、無効な選択肢、古い応答は、デフォルトで拒否されます。選択したトランスポートが失敗したときに、通常の CLI/TUI/ゲートウェイ/ACP のサーフェスで意図的にプロンプトを表示するには、`transport_fallback: builtin` を設定してください。この明示的なオプトインがない限り、Hermes が別のサーフェスにプロンプトを表示することはありません。

ハードラインのブロック、sudo-stdin の保護、ユーザーの拒否ルール、リクエストのバインド、許可されたスコープ、永続化、フック、最終的な認可は、引き続き Hermes が所有します。ハードラインのコマンドは、トランスポートのコールバックより前にブロックされます。このインターフェースには、意図的に**プラグインの承認ポリシー、自動許可コールバック、必須の `pre_tool_call` ポリシーはありません**。将来の承認ポリシーのケイパビリティはプラグインのケイパビリティ同意モデルを使う可能性がありますが、トランスポートの選択がそれを付与することはありません。

### 既存ユーザー向けのマイグレーション

オプトインプラグインを持つバージョンの Hermes（設定スキーマ v21 以降）にアップグレードすると、`~/.hermes/plugins/` 配下にすでにインストールされていて `plugins.disabled` に含まれていなかったユーザープラグインは、**自動的に `plugins.enabled` に祖父継承（grandfather）されます**。既存の設定はそのまま動作し続けます。バンドルされたスタンドアロンプラグインは祖父継承されません — 既存ユーザーであっても明示的にオプトインする必要があります。（バンドルされたプラットフォーム／バックエンドプラグインは決してゲートされなかったため、祖父継承は不要でした。）

## 利用可能なフック

プラグインは、現在 `hermes_cli.plugins.VALID_HOOKS` が受け付ける 27 のライフサイクルイベントを登録できます。正確なタイミング、戻り値の扱い、ペイロードのフィールド、プライバシーに関する注意は、**[イベントフックのカタログ](./hooks.md#shipped-plugin-hook-catalog)** が正式な情報源です。

| 説明上のカテゴリ | 同梱フック |
|---|---|
| **ディレクティブ／制御** | `pre_tool_call`, `pre_llm_call`, `pre_verify`, `pre_gateway_dispatch` |
| **変換** | `transform_tool_result`, `transform_terminal_output`, `transform_llm_output`, `pre_transcription` |
| **オブザーバー** | `post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`, `api_request_error`, `pre_auxiliary_call`, `post_auxiliary_call`, `on_stream_start`, `on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `post_compaction`, `post_curator_run`, `on_session_delete`, `agent_loop_stopped`, `on_skill_lifecycle`, `subagent_start`, `subagent_stop`, `pre_approval_request`, `post_approval_response`, `on_human_input_request`, `on_human_input_resolved`, `pre_command`, `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked` |

これらのカテゴリは現在の挙動を説明するものであり、将来の命名規則を定めるものではありません。プラグインのミドルウェアは別のレジストリ／サーフェスのままです。
## プラグインの種類

Hermes には 4 種類のプラグインがあります。

| 種類 | 何をするか | 選択方法 | 場所 |
|------|-------------|-----------|----------|
| **一般プラグイン** | ツール、フック、スラッシュコマンド、CLI コマンドを追加する | 複数選択（有効／無効） | `~/.hermes/plugins/` |
| **メモリプロバイダー** | 組み込みメモリを置き換えるか拡張する | 単一選択（1 つがアクティブ） | `plugins/memory/` |
| **コンテキストエンジン** | 組み込みのコンテキスト圧縮器を置き換える | 単一選択（1 つがアクティブ） | `plugins/context_engine/` |
| **モデルプロバイダー** | 推論バックエンドを宣言する（OpenRouter、Anthropic など） | 複数登録、`--provider` / `config.yaml` で選択 | `plugins/model-providers/` |

メモリプロバイダーとコンテキストエンジンは**プロバイダープラグイン**です — 各種類のうち、同時にアクティブにできるのは 1 つだけです。モデルプロバイダーもプラグインですが、多数が同時に読み込まれ、ユーザーは `--provider` または `config.yaml` で一度に 1 つを選びます。一般プラグインは任意の組み合わせで有効にできます。

## プラグイン可能なインターフェース — それぞれの行き先 {#pluggable-interfaces--where-to-go-for-each}

上の表は 4 つのプラグインカテゴリを示していますが、「一般プラグイン」の中で `PluginContext` はいくつかの異なる拡張ポイントを公開しています。さらに Hermes は、Python プラグインシステムの外にある拡張（設定駆動のバックエンド、シェルフックのコマンド、外部サーバーなど）も受け入れます。作りたいものに対する正しいドキュメントを見つけるには、この表を使ってください。

| 追加したいもの… | 方法 | 作成ガイド |
|---|---|---|
| LLM が呼び出せる**ツール** | Python プラグイン — `ctx.register_tool()` | [Hermes プラグインを作る](../../developer-guide/plugins/index.md) · [ツールの追加](/docs/developer-guide/adding-tools) |
| **ライフサイクルフック**（LLM の前後、セッションの開始／終了、ツールフィルタ） | Python プラグイン — `ctx.register_hook()` | [フックリファレンス](./hooks.md) · [Hermes プラグインを作る](../../developer-guide/plugins/index.md) |
| CLI／ゲートウェイ向けの**スラッシュコマンド** | Python プラグイン — `ctx.register_command()` | [Hermes プラグインを作る](../../developer-guide/plugins/index.md) · [CLI の拡張](/docs/developer-guide/extending-the-cli) |
| `hermes <thing>` 向けの**サブコマンド** | Python プラグイン — `ctx.register_cli_command()` | [CLI の拡張](/docs/developer-guide/extending-the-cli) |
| プラグインが同梱する**スキル** | Python プラグイン — `ctx.register_skill()` | [スキルの作成](/docs/developer-guide/creating-skills) |
| **推論バックエンド**（LLM プロバイダー: OpenAI 互換、Codex、Anthropic-Messages、Bedrock） | プロバイダープラグイン — `plugins/model-providers/<name>/` で `register_provider(ProviderProfile(...))` | **[モデルプロバイダープラグイン](/docs/developer-guide/model-provider-plugin)** · [プロバイダーの追加](/docs/developer-guide/adding-providers) |
| **ゲートウェイチャネル**（Discord / Telegram / IRC / Teams など） | プラットフォームプラグイン — `plugins/platforms/<name>/` で `ctx.register_platform()` | [プラットフォームアダプターの追加](/docs/developer-guide/adding-platform-adapters) |
| **メモリバックエンド**（Honcho、Mem0、Supermemory など） | メモリプラグイン — `plugins/memory/<name>/` で `MemoryProvider` をサブクラス化 | [メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin) |
| **コンテキスト圧縮戦略** | コンテキストエンジンプラグイン — `ctx.register_context_engine()` | [コンテキストエンジンプラグイン](/docs/developer-guide/context-engine-plugin) |
| **画像生成バックエンド**（DALL·E、SDXL など） | バックエンドプラグイン — `ctx.register_image_gen_provider()` | [画像生成プロバイダープラグイン](/docs/developer-guide/image-gen-provider-plugin) |
| **委任戦略**（サブエージェントごとのプロファイルまたは名前付きモデル） | バックエンドプラグイン — `ctx.register_delegation_provider()`、`delegation.backend` で選択 | [委任プロバイダープラグイン](/docs/developer-guide/delegation-provider-plugin) |
| **動画生成バックエンド**（Veo、Kling、Pixverse、Grok-Imagine、Runway など） | バックエンドプラグイン — `ctx.register_video_gen_provider()` | [動画生成プロバイダープラグイン](/docs/developer-guide/video-gen-provider-plugin) |
| **コンピューター操作ドライバー**（`computer_use` ツールの背後にあるデスクトップ操作） | 単一選択のプロバイダープラグイン — `ctx.register_computer_use_provider()`、`computer_use.backend` で選択 | [コンピューター操作バックエンドプラグイン](../../developer-guide/plugins/index.md#computer-use-backend-plugins) |
| **TTS バックエンド**（任意の CLI — Piper、VoxCPM、Kokoro、xtts、音声クローンスクリプトなど） | 設定駆動（推奨） — `config.yaml` で `tts.providers.<name>` に `type: command` で宣言。または Python バックエンドプラグイン — シェルテンプレート以上のものが必要な Python SDK／ストリーミングエンジン向けに `ctx.register_tts_provider()`。 | [TTS セットアップ](./tts.md#custom-command-providers) · [Python プラグインガイド](./tts.md#python-plugin-providers) |
| **STT バックエンド**（任意の CLI — whisper.cpp、カスタムの whisper バイナリ、ローカル ASR CLI） | 設定駆動（推奨） — `config.yaml` で `stt.providers.<name>` に `type: command` で宣言するか、レガシーな単一コマンドの抜け道として `HERMES_LOCAL_STT_COMMAND` を設定。または Python バックエンドプラグイン — Python SDK エンジン（OpenRouter、SenseAudio、Gemini-STT など）向けに `ctx.register_transcription_provider()`。 | [STT セットアップ](./tts.md#stt-custom-command-providers) · [Python プラグインガイド](./tts.md#python-plugin-providers-stt) |
| **MCP 経由の外部ツール**（filesystem、GitHub、Linear、Notion、任意の MCP サーバー） | 設定駆動 — `config.yaml` で `mcp_servers.<name>` を `command:` / `url:` で宣言。Hermes はサーバーのツールを自動検出し、組み込みと並べて登録します。 | [MCP](./mcp.md) |
| **追加のスキルソース**（カスタム GitHub リポジトリ、プライベートなスキルインデックス） | CLI — `hermes skills tap add <repo>` | [スキルハブ](./skills.md#skills-hub) · [カスタムタップの公開](./skills.md#publishing-a-custom-skill-tap) |
| **ゲートウェイイベントフック**（`gateway:startup`、`session:start`、`agent:end`、`command:*` で発火） | `HOOK.yaml` ＋ `handler.py` を `~/.hermes/hooks/<name>/` に置く | [イベントフック](./hooks.md#gateway-event-hooks) |
| **シェルフック**（イベント発生時にシェルコマンドを実行 — 通知、監査ログ、デスクトップ通知） | 設定駆動 — `config.yaml` で `hooks:` 配下に宣言 | [シェルフック](./hooks.md#shell-hooks) |

:::note
すべてが Python プラグインというわけではありません。一部の拡張サーフェスは、意図的に**設定駆動のシェルコマンド**（TTS、STT、シェルフック）を使うので、すでに持っている任意の CLI が Python を書かずにプラグインになります。他のもの（MCP）は**外部サーバー**で、エージェントが接続してそこからツールを自動登録します。そしていくつか（ゲートウェイフック）は、独自のマニフェスト形式を持つ**ドロップインのディレクトリ**です。ユースケースに合った統合スタイルに対して、正しいサーフェスを選んでください。上の表の作成ガイドはそれぞれ、プレースホルダー、検出、例を扱っています。
:::

## NixOS の宣言的プラグイン

NixOS では、モジュールのオプションを通じてプラグインを宣言的にインストールできます — `hermes plugins install` は不要です。完全な詳細は **[Nix セットアップガイド](/docs/getting-started/nix-setup#plugins)** を参照してください。

```nix
services.hermes-agent = {
  # Directory plugin (source tree with plugin.yaml)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin (pip package)
  extraPythonPackages = [ (config.services.hermes-agent.package.python.pkgs.buildPythonPackage { ... }) ];
  # Enable in config
  settings.plugins.enabled = [ "my-plugin" ];
};
```

宣言的プラグインは `nix-managed-` プレフィックス付きでシンボリックリンクされます — 手動でインストールしたプラグインと共存し、Nix の設定から削除されると自動的にクリーンアップされます。

## プラグインの管理

```bash
hermes plugins                               # unified interactive UI
hermes plugins list                          # table: enabled / disabled / not enabled (bundled backends,
                                             # platforms and the live memory.provider count as enabled)
hermes plugins search <term>                 # search the Hermes plugin catalog
hermes plugins install <name>                # install a catalog entry (repo @ reviewed pinned SHA)
hermes plugins install user/repo             # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable    # request enable; dependency consent still applies
hermes plugins install user/repo --no-enable # install but leave disabled (no prompt)
hermes plugins update my-plugin              # pull latest (local edits are autostashed and re-applied)
hermes plugins update my-plugin --force      # same, accepting a caution security-scan verdict without the question
hermes plugins update                        # list every plugin with an update, then "Update all N? [Y/n]"
hermes plugins update --yes                  # same, without the question (scripts, cron; required without a terminal)
hermes plugins remove my-plugin              # uninstall; also drops it from plugins.enabled/disabled/entries
                                             # and resets memory.provider when it was the live provider
hermes plugins enable my-plugin              # add to allow-list
hermes plugins disable my-plugin             # remove from allow-list + add to disabled (bundled platforms:
                                             # either spelling works, e.g. photon-platform or platforms/photon)
hermes plugins capabilities [my-plugin]      # declared vs granted capabilities
hermes plugins idle-tasks [my-plugin]        # background idle jobs plugins registered: last run, next run
hermes plugins idle-tasks my-plugin --pause suggest   # stop one until --resume
hermes plugins check-updates                 # read-only: is any installed plugin outdated?
hermes plugins adopt my-plugin               # track a self-cloned plugin dir (read its git origin)
hermes plugins trust-update-url my-plugin    # confirm a changed update_url after review
```

### 更新チェックと出所の記録

Hermes は Git のインストール元とリビジョンを `.install-metadata.json` に記録します。固定されていない追跡対象のインストールは、保存されたソースのリモート HEAD、または一致する保存済みの `update_url` フィードと比較されます。固定されたインストールは固定されたままです。自分でクローンしたディレクトリは、追跡対象のインストールになる前に `hermes plugins adopt NAME` が必要です。手動でコピーされたディレクトリや出所がずれたディレクトリには、診断のガイダンスが表示されます。pip のエントリポイントプラグインは、所有するディストリビューションの利用可能なバージョンを報告できますが、そのチェックによって Git 管理のインストールになるわけではありません。

`hermes plugins check-updates` はプラグインのファイルを変更しません。スケジュールされたゲートウェイのチェックは、`plugins.auto_update_check_hours` の期限が来たときに実行されます。デフォルトは 24 時間で、`0` で無効になります。その結果は `hermes pm status` とデスクトップの同期ステータス画面で確認できます。別の間隔を設定した場合、これは厳密な 1 日 1 回の制限ではありません。

名前なしの `hermes plugins update` は同じチェックを実行し、移動できるプラグインを一覧表示します。カタログのインストール（レビュー済みのカタログコミットに再固定）と、追跡対象の Git インストール（プル）です。スキップするプラグインはすべて理由とともに示されます。固定済み、`--ref` で保持、修正が必要、状態不明、または Hermes がアップグレードしない pip ディストリビューションです。ターミナルでは一度だけ尋ね、その後 `hermes plugins update NAME` と同じ経路で各プラグインを更新するので、新しいケイパビリティ、広がったカタログの固定、スキャンの指摘は引き続きプラグインごとにプロンプトされます。1 つが失敗しても残りは止まりません。サマリーには、更新されたもの、すでに最新だったもの、失敗したものとその理由が表示されます（いずれかが失敗した場合は終了コード 1）。ターミナルがない場合は一覧を表示するだけで、`--yes` を渡さない限り何も更新しません。`--yes` を付けた場合、同意プロンプトはあらゆる非対話的な更新と同様にフェイルクローズします（新しいケイパビリティは付与されないまま、広がったカタログの固定は適用されません）。

デフォルトでは、プラグインは `hermes plugins update` を実行したときにのみ更新されます。自動更新（`plugins.auto_apply: true`）をオンにすると、ゲートウェイのスケジュールされたチェックが、同じカタログの再固定と追跡対象の Git プルを自動で適用します。どちらの経路も更新時のセキュリティスキャンを使い、無人の更新は、広がったカタログの固定や Git 更新（新しいツール、フック、ケイパビリティ）、新しいケイパビリティ、Git 更新に対する caution のスキャン判定を決して受け入れません。それらはあなたが自分で更新を実行するのを待ちます。（カタログの再固定は caution の判定を受け入れます。その固定はカタログに受け入れられたときにレビュアーが読んでいるためです。）自動更新は、固定済み、手動、ずれた、または pip ディストリビューションの行を管理しません。また、カタログの固定とは別のコミットで `--ref` を付けてインストールしたカタログプラグインも管理しません。それはあなたが名前を指定する（`hermes plugins update <name>`）まで選んだコミットに留まり、すべての Update all（名前なしの `hermes plugins update` を含む）はそれをスキップします。カタログの固定と等しい `--ref` は通常のカタログインストールとして記録されるので、自動更新はカタログとともにそれを動かします。自動更新はゲートウェイのハウスキーピングスレッドで実行されるので、セッションの開始を遅らせることはなく、オフライン時のチェックは警告 1 つで済みます。スケジュールされたチェックが更新したコードは新しいセッションで読み込まれます。実行中のセッションはすでにインポートしたコードを保持します。この設定はプロファイルごとです。

同じスイッチと「Update all」アクションは、すべてのサーフェスにあります。

| サーフェス | Update all | 自動更新 |
|---|---|---|
| CLI | `hermes plugins update` | `hermes plugins auto-update [on\|off]` |
| チャット（従来の CLI と TUI） | `/plugins update` | `/plugins auto-update [on\|off]` |
| デスクトップアプリ | Plugins タブ、**Update all** | Plugins タブ、**Auto-update** スイッチ |
| Web ダッシュボード | Plugins ページ、**Update all** | Plugins ページ、**Auto-update** スイッチ |

`/plugins update`、Desktop、ダッシュボードはプラグインごとに尋ねずに更新するので、レビューが必要なプラグイン（広がった固定や Git 更新、Git 更新の caution スキャン）は何も変更されずに「needs review」として一覧表示されます。レビューするには名前を指定して更新してください。更新された各プラグインは、プラグイン自身の Update アクションからの更新と同様に、ただちに再読み込みされます。その MCP サーバーとスキルは開いているチャットにすぐ反映され、Python ツールは新しいセッションで読み込まれます。同じプラグインの 2 つの更新が同時に実行されることはありません。2 つ目は拒否され、何も変更しません。

`hermes update` はカタログの固定を動かすことがあります。インストール済みのカタログプラグインは更新されるまで古い固定に留まるので、更新はすべてのプロファイルについてそれらを示し、そのプロファイルのゲートウェイに次のティックで再チェックするよう依頼します（自動更新がオンならそこで適用されます）。

マニフェストが `update_url` を変更または新たに導入した場合、`hermes plugins trust-update-url NAME` で承認するまで Hermes は新しいアドレスを拒否します。これはフィードのソースのチェックであり、すでに信頼しているプラグインのコードに対するサンドボックスではありません。

### 依存関係の準備と保持 {#dependency-preparation-and-preservation}

Python の依存関係のインストールには、別個の同意／受け入れステップがあります。`plugins install --enable` はそのステップをバイパスしません。拒否された、または非対話的な依存関係のインストールでは、プラグインはインストールされたまま無効になることがあります。Node のサイドカーの依存関係には別のプロンプトがあり、プラグインローカルに留まります。

PM は、新しい環境と設定を公開する前に、コアと有効なプラグインのセットで Python の依存関係を準備します。解決に失敗した場合は、以前の選択が保持されます。新しく選択された環境が実行中のプロセスでまだアクティブでない場合は、Hermes を再起動してください。

有効なセットは、デフォルトのホームと `profiles/` 配下の**すべてのプロファイル**の和集合で、各 `config.yaml`（`plugins.enabled`、`plugins.disabled`、`memory.provider`）から読み取られます。PM は読めないホームについて推測することを拒否します。有効な YAML でない、マッピングでない、`plugins.enabled`/`plugins.disabled` がリストでない、または `memory.provider` が文字列でない `config.yaml` があると、そのプロファイルのプラグインを次の環境から黙って落とすのではなく、**すべての**ホームの依存関係の準備が失敗します（`could not parse plugin selection: <path>`）。問題のファイルを修正または削除してください。空の `config.yaml` は問題ありません。

通常の Hermes アプリケーションの更新は、ラッパーファイルや外部のサイドカーリンクを含め、ユーザーのプラグインディレクトリを保持します。明示的なプラグインの更新や削除は、それらのファイルを変更することがあります。[パッケージ管理](/docs/reference/package-management) と [プラグイン作成ガイド](../../developer-guide/plugins/index.md#lazy-install-optional-python-dependencies) を参照してください。
### ワンクリックインストールリンク（Desktop） {#one-click-install-links-desktop}

Hermes Desktop は `hermes://` URL スキームを登録するので、Web サイト、README、チャットメッセージからプラグインのインストールに直接リンクできます。

```
hermes://plugin/install?catalog=NAME               # catalog entry, installs the reviewed pin
hermes://plugin/install?repo=owner/repo            # any git repo
hermes://plugin/install?repo=owner/repo&enable=1   # enable the agent plugin after install
hermes://plugin/install?repo=owner/repo&force=1    # replace an existing install
hermes://plugin/install?catalog=<name>             # reviewed catalog entry at its pinned commit
```

`catalog=<name>` の形式は、すべての [プラグインカタログ](/docs/user-guide/features/plugin-catalog) のカードにある **Open in Hermes Desktop** ボタンが使うものです。Desktop はライブのカタログ（**Capabilities → Plugins** ピッカーが表示するのと同じフィード）に対して名前を解決し、アプリ内で選んだときと同じ**レビュー済みカタログエントリ**のダイアログを開きます。エージェント側はブランチの先端ではなく、カタログの固定されたコミットでインストールされます。リンクにはリポジトリの URL が含まれず、カタログにない名前はエラーのトーストを表示するだけで何もしません — git のパスとして解釈し直されることはないので、リンクが見慣れた名前の陰にレビューされていないリポジトリを紛れ込ませることはできません。

`repo=` のリンクの場合、クリックすると Hermes が開いて**確認ダイアログ**が表示されます — リポジトリ ID、「Before you install」の注記、GitHub の閲覧とクローンのリンクです — その後、リポジトリをシャロークローンして何が同梱されているかを検出します（**エージェントプラグイン** — バックエンドの Python、**デスクトッププラグイン** — アプリの UI、またはその両方）。チェックボックスでコンポーネントを選び、確定します。確定するまで何もインストールされません。ディープリンクが自動インストールすることはなく、エージェントプラグインのインストールは `hermes plugins install` と同じ [インストール時のセキュリティスキャン](#install-time-security-scanning) を通ります。

ハイブリッドなリポジトリ（1 つのリポジトリにエージェント側とデスクトップ側がある）は、1 つのリンクと 1 つのダイアログを使います。同じモーダルには、リンクなしで **Capabilities → Plugins → Install from Git** からもアクセスできます。レガシーな `hermes://plugin-agent/…` と `hermes://plugin-desktop/…` の URL も同じダイアログにルーティングされます。開発ビルド（`pnpm run dev`）では、スキームは `hermes-dev://` です。

Web サイトに SDK は不要です — 通常のアンカーで動作します。

```html
<a href="hermes://plugin/install?repo=owner/repo&enable=1">Install in Hermes</a>
```

MCP サーバーにも同等のリンク形式があります — [Add to Hermes リンク](../../reference/mcp-config-reference.md#add-to-hermes-link) を参照してください。

### プラグインのケイパビリティと同意

プラグインは、必要とする特権的なホストのサーフェスを `plugin.yaml` で宣言できます。

```yaml
name: my-plugin
capabilities:
  - tools.override        # replace built-in tools
  - llm.model_override    # pick the model for host-owned LLM calls
```

プラグインがケイパビリティを宣言すると、`hermes plugins install`（および `hermes plugins enable`）は 1 行のリスク説明付きでその一覧を表示し、一度だけ尋ねます。同意すると、付与が同意ハッシュとタイムスタンプとともに `plugins.entries.<id>.granted_capabilities` に記録されます。拒否すると、プラグインは有効なまま、それらのケイパビリティはオフになります — 行儀のよいプラグインは `ctx.has_capability()` で確認し、穏やかに機能を落とします。

**更新時の再同意:** プラグインの更新があなたが付与していないケイパビリティを宣言した場合、`hermes plugins update` は追加分を示して再度尋ねます。新しいケイパビリティは同意するまでオフのままです — プラグインの更新が黙ってアクセス範囲を広げることは決してありません。カタログの再固定はさらに一歩進んでいます。新しい固定が、インストール済みのバージョンになかったツール、フック、Python の依存関係、ホストのケイパビリティ、プロバイダーのサインイン（`requires_auth`）、新規または変更された OAuth クライアント（`oauth:`）、Desktop の UI 部分を追加する場合、CLI は何かを動かす前に差分を表示して `y/N` を尋ね、Desktop／ダッシュボードの **Update** ボタンも同じ確認を開きます。拒否した場合（または非対話的なセッションの場合）、プラグインは古い固定に留まります。

**非対話的なセッションはフェイルクローズ:** TTY なしでインストールまたは更新するとインストールは完了しますが、宣言されたケイパビリティは付与*されません*。後で付与するには、対話的に `hermes plugins enable <id>` を実行してください。

状態はいつでも確認できます。

```bash
hermes plugins capabilities             # all plugins with declared/granted capabilities
hermes plugins capabilities my-plugin   # one plugin, declared vs granted
```

ケイパビリティ ID は、以前の機能ごとの設定ゲートと 1 対 1 で対応します。それらは引き続き動作しますが、同意フローに置き換わるため**非推奨**です。

| ケイパビリティ | レガシーなキー（`plugins.entries.<id>.…`） |
|---|---|
| `tools.override` | `allow_tool_override` |
| `llm.provider_override` | `llm.allow_provider_override` |
| `llm.model_override` | `llm.allow_model_override` |
| `llm.agent_id_override` | `llm.allow_agent_id_override` |
| `llm.profile_override` | `llm.allow_profile_override` |
| `llm.task_override` | `llm.allow_task_override` |
| `gateway.platform_actions` | `allow_platform_actions` |

ケイパビリティが付与されている*か*、レガシーなキーが設定されている*か*の*いずれか*でゲートは開きます — 既存の設定は変更なしで動作し続けます。`llm.side_question` と `llm.session_fork` にはレガシーなキーがありません。それぞれ、同意フローを通じて宣言かつ付与された場合にのみ開きます。

:::warning サンドボックスではありません
ケイパビリティは**同意と監査のレイヤー**であり、隔離ではありません。プラグインは通常のインプロセスの Python として実行されるので、悪意のあるプラグインはここにあるすべてのゲートを無視できます。ケイパビリティを付与することはプラグインの作者への信頼の表明であり、コード監査ではありません。また Hermes はプラグインのコードをレビューしていません。信頼できるソースのプラグインだけをインストールしてください。
:::

### プラットフォームアクション {#platform-actions}

`ctx.platform_actions` は、ライブのゲートウェイアダプターレジストリを通じて、接続されたチャットプラットフォーム上で動作するための、最小限でケイパビリティでゲートされた動詞のセットをプラグインに与えます — アダプターをモンキーパッチする代わりの、公認の手段です。**デフォルトではオフです**。すべての呼び出しで `gateway.platform_actions` ケイパビリティ（レガシーなキーは `plugins.entries.<id>.allow_platform_actions`）が再確認され、付与されていない呼び出しは動作せずに構造化されたエラーを返します。

v1 の動詞（どちらも `async` で、どちらもプレーンな dict を返し、どちらもフックのディスパッチに例外を送出することはありません）:

```python
result = await ctx.platform_actions.add_reaction(
    platform="telegram", chat_id="-100123", message_id="456", emoji="👍",
)
result = await ctx.platform_actions.set_thread_title(
    platform="discord", chat_id="123", thread_id="456", title="New title",
)
if not result["ok"]:
    print(result["error"], result.get("detail"))
```

成功は `{"ok": True, "action": <verb>}` です。失敗は `{"ok": False, "error": <code>, "detail": <str>}` で、安定したエラーコードを持ちます: `capability_not_granted`、`invalid_argument`、`gateway_unavailable`、`unknown_platform`、`adapter_not_registered`、`adapter_disconnected`、`unsupported_platform_action`、`action_failed`。アクションは、動作する前に対象のアダプターが存在し接続されていることを検証します。切断された、または存在しないアダプターは例外ではなく、構造化されたエラーに落ちます。

v1 でサポートされるプラットフォームは Telegram と Discord です。Telegram の `add_reaction` はボットのリアクションを*設定*します（Bot API は以前のボットのリアクションを積み重ねるのではなく置き換えます）。すべてのアクションは — 許可されたものも拒否されたものも — プラグイン ID、動詞、プラットフォーム、結果とともにログに書き込まれます。

:::warning セキュリティに関する注意
プラットフォームアクションは**ボットとしてメッセージングする権限**です。付与されたプラグインは、フックをトリガーしたチャットだけでなく、ゲートウェイのボットが到達できるあらゆるチャットでリアクションしたりスレッド名を変更したりできます。`gateway.platform_actions` は信頼できるプラグインにだけ付与し、どのアクションを行うかを正確に文書化しているプラグインを選んでください。生のプラットフォーム SDK のペイロード／ハンドルへのアクセスは、意図的にこのサーフェスに**含まれていません** — #64176 のラウンド 2 の設計修正により、それには「安定性の保証なし」というラベル付きの独自のケイパビリティ（`gateway.raw_events`）と別の設計が必要で、まだ出荷されていません。
:::

### プラグインを見つける — Hermes プラグインカタログ

`hermes plugins search <term>` は **Hermes プラグインカタログ** を検索します — hermes-agent リポジトリ（`plugin-catalog/`）で管理されている、厳選された SHA 固定のカタログです。マッチングはエントリ名、説明、宣言されたツールを対象とします。

```bash
hermes plugins search telegram    # search the catalog
hermes plugins browse             # browse every entry
hermes plugins info <name>        # full details for one entry
```

プラグインが見つかったら、名前だけでインストールします — 名前はエントリのリポジトリの**固定されたコミット SHA** に解決され、カタログの出所が記録されるので、カタログが動いたときに `hermes plugins update` が再固定できます。

```bash
hermes plugins install <catalog-name>
```

明示的な `owner/repo` や Git URL の識別子はカタログに触れることがなく、カスタム（未レビュー）のソースとしてフラグが立てられます。明示的な `--ref <40-char commit SHA>` はカスタムインストールを固定します。

完全な信頼モデル、受け入れの CI、提出のワークフローについては、[プラグインカタログ](/docs/user-guide/features/plugin-catalog) を参照してください。

:::warning カタログ掲載 ≠ 監査済み
カタログのエントリは、エントリのメタデータと宣言されたケイパビリティが受け入れ時にレビューされたことを意味します — **コード監査ではありません**。インストールは引き続き通常の同意フローを通ります（プラグインはデフォルトで無効の状態でインストールされ、有効化は明示的なステップであり、ツールの上書き権限には別途付与が必要です）。有効にする前に、プラグインのソースをレビューしてください。
:::

### プラグインパック

**プラグインパック**は、プラグインのセットを固定する、宣言的で共有可能な YAML ファイル（`hermes-pack.yaml`）です — modpack を共有するようなものです。パックのインストールは通常の固定インストールに展開されるだけで、実行時に新しいものは何も存在しません。

```yaml
name: voice-assistant-pack
description: STT + streaming TTS + approval relay
author: hyper
version: 1.0.0
plugins:
  - name: hermes-telegram-business       # bare plugin-catalog name…
    ref: e905f3bc5eeaa5a9dab9bc5155601b3ebec75757
  - repo: owner/approval-relay           # …or explicit owner/repo (or git URL)
    ref: 8f3c2d1a9b4e5f6071829304a5b6c7d8e9f00112
    subdir: plugins/relay                # optional monorepo path
config:                                  # optional, non-secret seeds only
  hermes-media-studio:
    default_model: flux-3
skills: []                               # declared list only (not auto-installed yet)
```

```bash
hermes plugins pack show ./hermes-pack.yaml     # dry-run review
hermes plugins pack install ./hermes-pack.yaml  # review → confirm → install
hermes plugins pack export > hermes-pack.yaml   # snapshot the current install
hermes plugins pack export --enabled-only       # only plugins.enabled
```

**サプライチェーンの姿勢。** すべてのエントリの `ref` は正確な 40 文字のコミット SHA でなければなりません — タグやブランチ名は、エントリ名を示すエラーで拒否されます。これはプラグインカタログと同じルールです。パックのインストールは `hermes plugins install --ref <sha>` とまったく同じ固定インストールの経路を通り、同じ出所を `plugins/.install-metadata.json` に記録するので、同じパックの 2 回のインストールは同一に解決されます。パックは [マニフェスト v2 のフィールド](../../developer-guide/plugins/index.md)（`manifest_version`、`api_version`、`requires_plugins`）の上に構築されます — 各プラグイン自身のマニフェストは、引き続き通常のインストール経路で検証されます。

**同意が一括で付与されることはありません。** `pack install` は必須のレビュー画面（すべてのプラグイン、ソース、固定された ref、宣言されたケイパビリティ）を表示し、パックの内容について**1 回**確認を求めます。その後、各プラグインが宣言したケイパビリティは、標準のプラグインごとのケイパビリティ同意プロンプトを通ります — 単一の `hermes plugins install` と同じです。`--yes` はなく、非対話的なセッションではパックをインストールできません。

**シークレットがパックで運ばれることはありません。** `config:` のシードは、シークレットではない `plugins.entries.<id>` のキーに限られます — シークレットのようなキー名（`*token*`、`*key*`、`*password*` など）、ケイパビリティの付与、非推奨の `allow_*` 信頼ゲートは、インストール時に拒否され、エクスポート時に取り除かれます。シークレットが必要なプラグインは自身の `requires_env` でそれを宣言し、通常どおりインストール中にプロンプトされます。`plugins.entries.<id>` にある既存のユーザーの値は、常にパックのシードより優先されます。

**部分的な失敗。** 各プラグインは独立してインストールされます。失敗はプラグインごとに報告され、残りは続行し、いずれかのプラグインが失敗した場合はコマンドが非ゼロで終了します。

**エクスポートの注意点。** `pack export` には、既知の Git の出所を持つプラグイン（`hermes plugins install` でインストールされたもの）だけが含まれます。ローカルのみのプラグインは、出力される YAML にインストール可能なエントリとしてではなく、警告コメントとして列挙されます。

`skills:` のリストはインストール時に解析・表示されますが、まだ自動インストールはされません — 今のところ手動でインストールしてください（`hermes skills`）。スキルハブの ID をパックのインストールに接続することは、文書化された今後の拡張ポイントです。

### インストール時のセキュリティスキャン {#install-time-security-scanning}

すべての `hermes plugins install` と `hermes plugins update` は、プラグインが有効化される前に、プラグインのツリーに対して静的なセキュリティスキャンを実行します（Claude Cowork のスキル＆プラグインのセキュリティスキャンに着想を得ています）。スキャナーは [スキルハブのガード](./skills.md) と同じ脅威パターンエンジンを再利用します — 認証情報ストアの持ち出し、リバースシェル、破壊的なコマンド、永続化の仕組み、難読化された実行、ドキュメントファイル内のプロンプトインジェクション — プラグインを考慮した除外付きです。プロバイダープラグインが環境から**自身の** API キーを読む（文書化された `requires_env` パターン）ことはフラグされません。

Cowork の pass/warn/fail に対応する 3 つの判定:

| 判定 | 挙動 |
|---|---|
| **safe** | 通常どおりインストールされ、追加の出力はない |
| **caution** | 指摘が表示され、`Install anyway? [y/N]` を確認する（または `--force` を渡す） |
| **dangerous** | ブロックされる。`--force` でも上書き**できない** |

`hermes plugins update` は、インストール済みのコピーを置き換える前に同じルールで新しいコードをスキャンするので、更新で危険なコードを取り込んだプラグインは実行される前にフラグされます。Git 更新に対する caution の判定は `Update anyway? [y/N]` と尋ね、`hermes plugins update NAME --force` は質問なしでそれを受け入れます。Hermes Desktop は同じ指摘を **Update anyway** ボタン付きのダイアログで表示します。カタログの更新はレビュー済みの固定に移動するので、そこでの caution の判定は尋ねません。いずれの場合も、更新が受け入れられるまで何も変わりません。

Hermes Desktop は、直近の更新チェックで新しいコミットが見つかった Git インストールのプラグインに **Update** ボタンを表示します。ゲートウェイは `plugins.auto_update_check_hours`（デフォルト 24）ごとにそのチェックを実行し、`hermes plugins check-updates` はオンデマンドで実行します。

`hermes plugins update` で、更新後のツリーに対する dangerous の判定は、あなたが指摘をレビューして再有効化するまでプラグインを無効にします。dangerous のブロックは、それを引き起こした重大な指摘を示すので（例: `1 critical of 42 findings (destructive_root_rm)`）、ブロックの原因となった 1 行が合計の陰に隠れることはありません。

インストール時にホスト上で実行できないテキストは、プラグインの振る舞いではなく**文脈**として評価されるので、指摘を下げることはあっても削除することはありません — すべての指摘はファイルと行とともにレポートに残ります。

- **ドキュメントの文章**（`README.md`、`AGENTS.md`、`docs/**/*.md`、`.txt`、`.rst`、`.html`）は、それ単独では決して **dangerous** を生みません。そこに引用されたコマンドや認証情報のパス（アンインストール手順、`~/.ssh` を挙げた拒否リスト）は重大度が 1 段階下がり、プラグイン**自身の**インストールディレクトリを削除する README（`rm -rf "$HOME/.hermes/plugins/<name>"`）は note になります。エージェント向けの形は完全な重大度を保ちます — プロンプトインジェクション、Markdown による持ち出し、エージェント設定の編集、`curl … | sh` のワンライナー、`authorized_keys` への追記、漏洩したプロバイダーキー — また、エージェントが指示として読む、同梱された `skills/` ツリー配下や `after-install.md` 内のものも同様です。
- **テストツリーとフィクスチャ**（プラグインのルートの `tests/`、`test/`、`testing/`、`spec/`、`specs/`、`fixtures/`。任意の深さの `__tests__/` と `__fixtures__/`。`*.test.*`、`*.spec.*`、`test_*.py`、`*_test.*`）は引き続きスキャンされます — プラグインの `__init__.py` がそこからインポートできるためです — が、引用されているだけの悪意ある文字列（`verdict_for("rm -rf /")`、偽の `sk-…` キーを含むマスキング用コーパス）は note になり、インポート時に実行されるテストコード（`os.system('rm -rf /')`）は **caution** が上限になります。他のファイル（`setup.sh`、`src/spec/…`）での同じ指摘は引き続き **dangerous** です。
- **行全体のコメントと `CHANGELOG.md`** は防御について説明するものなので、文章と同様に評価されます。
- **メディアヘッダーにデコードされる Base64**（data URI や JSON の背景データ内の PNG/JPEG/GIF/WOFF/PDF など）は情報扱いです。テキストフィルター（`grep`、`jq`）にパイプされた `base64 -d` は note、シェルやインタープリターにパイプされたものは完全な重大度を保ちます。正規表現リテラルの選択肢の一部としての `sudo` / `env|`（`/approval|sudo|secret/`、マスキングのパターン）は note ですが、コマンド文字列内（`subprocess.run("sudo …")`）ではそうなりません。

同様に、実行時の `.py` ファイルの `if __name__ == "__main__":` 自己テストブロック内にある一般的なサンプルトークン（`hardcoded_secret`）は **caution** が上限になります — ローダーはプラグインをインポートするだけで、そのブロックを実行することはないためです — が、その中の他のすべての指摘（破壊的なコマンド、`sk-…` のようなプロバイダー形式のキー）と、ガードより上にある同じトークンは完全な重大度を保ちます。

スキャンはデフォルトでオンです。`config.yaml` で無効にできます。

```yaml
plugins:
  scan_on_install: false
```

### プラグインをプロセス外で実行する（`plugins.isolation`） {#running-plugins-out-of-process-pluginsisolation}

デフォルトでは、サードパーティの Python プラグインはこれまでどおり Hermes のプロセスにインポートされます。`plugins.isolation: host` を設定すると、それらは**プラグインホスト**に移されます。これはプロファイルごとに 1 つの別個の Python プロセスで、必要に応じて起動され、プロファイルのユーザーがインストールしたプラグインをインポートし、プライベートなパイプで Hermes と通信します。

```yaml
plugins:
  isolation: host        # default: in_process
  host:
    launcher: []         # optional argv prefix for the host, e.g. a sandbox runner
```

プラグイン自体は変わりません。同じ `ctx` を受け取り、ツール、フック、スラッシュコマンド、スキル、プロバイダーオブジェクト（画像／動画生成、Web 検索、ブラウザー、TTS/STT、メモリ、コンテキストエンジン、シークレットソース、`@` コンテキスト参照、モデルプロバイダーのプロファイル）を以前とまったく同じように登録します。Hermes は自分の側に、ホストを呼び出す対応するエントリを登録し、値はプロバイダーの基底クラスが宣言する型としてやり取りされます（シークレットソースの `fetch()` は `Path` を受け取り `FetchResult` を返します）。ダッシュボードのプラグイン API もホストが提供します。バンドルされたプラグインはインプロセスで実行され続けます。インプロセスのダッシュボード API ハンドラはゲートウェイのイベントループを共有するので、それをブロックしてはいけません（[イベントループをブロックしない](extending-the-dashboard.md#dont-block-the-event-loop) を参照）。

複数のプロファイルを提供するゲートウェイでは、ゲートウェイが起動したプロファイル以外のすべてのプロファイルが、`plugins.isolation` の設定にかかわらず、自身のダッシュボードプラグイン（自身の `plugins/` フォルダーまたはルートの `~/.hermes/plugins/` にインストールされ、同名の衝突ではプロファイル自身のコピーが勝ち、自身の設定で有効化されたもの）を自身のプラグインホストで実行します。別のプロファイルのプラグインコードがゲートウェイ内で実行されることはありません。ホストはまだ WebSocket を扱わないので、そのようなプラグインの WebSocket ルートはそのプロファイルでは拒否されます（HTTP 501）。HTTP ルートは動作します。遅いホスト側ハンドラが止めるのは自身のプロファイルのプラグインリクエストだけで、ホストが 60 秒経っても応答しないリクエストには HTTP 504 が返ります。このようなホストは、そのプロファイルの最初のダッシュボードプラグインリクエストで起動し、ゲートウェイが終了するまで実行されます。ダッシュボードプラグインを使ったプロファイルごとに、追加の Python プロセスが 1 つです。

`host` モードで変わること:

- **インタープリターを共有しない。** プラグインのモジュールが Hermes のプロセスに入ることはないので、メモリから別のプロファイルのデータを読んだり、Hermes の内部をパッチしたりできません。マルチプレックスのゲートウェイでは、すべてのプロファイルが自身のホストを持ち、そのプロファイルの環境とシークレットだけで起動されます。
- **クラッシュは封じ込められる。** クラッシュまたは終了したプラグインは Hermes ではなく自身のホストを終了させます。実行中の呼び出しはツールエラーを返し、Hermes はホストを再起動してプラグインを再読み込みします（再試行回数に上限あり）。
- **いくつかのサーフェスはインプロセスのコードを必要とし**、読み込む代わりに明確な理由でそのプラグインを失敗させます: ゲートウェイのプラットフォームアダプター（`register_platform`）、承認トランスポート、Telegram／プラットフォームのハンドラ、独自の SDK クライアントを構築するモデルプロバイダーのプロファイル（`create_client`）、ターミナルバックエンド（バックエンドのコマンドがこの理由で失敗します。`singularity` プラグインがその一例です）、ストリーミングと WebSocket のダッシュボードエンドポイント、Hermes のモジュールをモンキーパッチするプラグイン。これらは `isolation: in_process` で実行してください。

**共有デプロイメントでのロック。** `plugins.isolation` は通常のプロファイル設定なので、プロファイルの `config.yaml` を編集できる人なら誰でもオフにできます。プロファイルが互いに隔離したい人々のものである場合は、代わりに [管理スコープ](/docs/user-guide/managed-scope) で固定してください。管理された値はすべてのプロファイル自身の設定より優先され、`hermes config set` はそれを変更することを拒否します。

```yaml
# /etc/hermes/config.yaml (root-owned, read by every profile on the machine)
plugins:
  isolation: host
  host:
    launcher: [...]      # pin the sandbox runner too, if you use one
```

エージェント自身がオペレーターのファイルに到達できないよう、エージェントのターミナルも隔離されたバックエンド（Docker、SSH など）で実行してください。

`hermes plugins validate <dir>` と `hermes plugins show <name>` は、プラグインがホストで実行されるかどうか、されない場合はその理由を報告します。執筆時点では、プラグインカタログの 348 エントリのうち 299 が変更なしでホストで実行されます。

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

- **General Plugins セクション** — チェックボックス、SPACE でトグル。プラグインが現在アクティブな場合、行はチェックされた状態で開きます。`plugins.enabled` に列挙されているもの、バンドルされたプラットフォーム、バックエンド、モデルプロバイダー（リストのエントリなしでオン）、またはカテゴリの選択されたプロバイダーです。終了時に書き込まれるのは切り替えた行だけです。チェックを外すとプラグインが `plugins.disabled` に追加され（明示的にオフ）、チェックすると `plugins.enabled` に追加されて古い無効化が解除されます。ピッカーを開いてそのまま閉じても何も変わりません。
- **Provider Plugins セクション** — 現在の選択を表示します。ENTER を押すとラジオピッカーに入り、アクティブなプロバイダーを 1 つ選びます。
- バンドルプラグインは、`[bundled]` タグ付きで同じ一覧に表示されます。

プロバイダープラグインの選択は `config.yaml` に保存されます。

```yaml
memory:
  provider: "honcho"      # empty string = built-in only

context:
  engine: "compressor"    # default built-in compressor
```

### enabled vs. disabled vs. どちらでもない

プラグインは次の 3 つの状態のいずれかになります。

| 状態 | 意味 | `plugins.enabled` にある? | `plugins.disabled` にある? |
|---|---|---|---|
| `enabled` | 次のセッションで読み込まれる | はい | いいえ |
| `disabled` | 明示的にオフ — `enabled` にも含まれていても読み込まれない | （無関係） | はい |
| `not enabled` | 検出されたがオプトインされていない | いいえ | いいえ |

新しくインストールされた、またはバンドルされたプラグインのデフォルトは `not enabled` です。`hermes plugins list` は 3 つの異なる状態をすべて表示するので、明示的にオフにされたものと、有効化を待っているだけのものを区別できます。

実行中のセッションでは、`/plugins` が現在読み込まれているプラグインを表示します。

## ライブメッセージの投稿 {#posting-a-live-message}

プラグインは、自分が作業している会話に 1 つのメッセージを投稿し、後でその同じメッセージを更新できます。どのチャネルでも、プラットフォームごとのコードは不要です。ライブの進捗表示（長いツールの実行中のチェックリスト、計画の改訂に合わせて変わる計画カード）に使ってください。

```python
def run_migration(args, **kwargs):
    status = ctx.post_message("Migrating: 0/3 tables")
    for i, table in enumerate(TABLES, 1):
        migrate(table)
        status.edit(f"Migrating: {i}/3 tables")
    status.edit("Migration done ✓")
    return tool_result({"migrated": len(TABLES)})
```

**シグネチャ:** `ctx.post_message(text: str) -> PluginMessage` と `PluginMessage.edit(text: str) -> dict`。すでにゲートウェイのイベントループ上で実行されている `async` のコード（非同期のフックやスラッシュコマンドのハンドラ）からは、代わりに `await ctx.apost_message(text)` と `await handle.aedit(text)` を使ってください。同期版はそこでは、ループをデッドロックさせる代わりに `error: "on_event_loop"` で拒否します。

**どの会話か。** 現在の会話とは、呼び出し元のコードにバインドされたセッションです。そのセッションのターン内で実行されるツールハンドラやフック、そのスラッシュコマンド、またはそのセッションのために発火したフックです。プラグインがチャットを指定することはないので、他の場所に投稿することはできません。プラグイン自身が開始したスレッドはそのバインドを継承しません（継承させるなら `contextvars.copy_context().run(...)` で開始してください）。また、タイマー、cron の実行、ライブのゲートウェイがないプロセスには、会話がまったくありません。

**例外を送出しない。** すべての呼び出しが結果を返します。`handle.ok`、`handle.message_id`、`handle.error` は直前の操作を表します。`edit()` は `{"ok", "message_id", "edited"}` を返し、新しいメッセージを投稿しなければならなかった場合は `"fallback": True` が加わります。エラーコード: `no_conversation`（呼び出し元のコンテキストに何もバインドされていない、セッションがライブでない、または別のプロファイルに属している）、`gateway_unavailable`、`adapter_unavailable`、`send_failed`、`edit_failed`、`on_event_loop`、`timeout`。拒否された投稿は、`edit()` も同様に拒否されるハンドルを返すので、正常系に分岐は不要です。`timeout` はゲートウェイが 30 秒以内に応答しなかったことを意味します。操作は実行され続け、その結果は引き続きハンドルに反映され（その間 `handle.pending` は true）、次の `edit()` はまずそれを待つので、遅い投稿がハンドルで編集できないメッセージを残すことはありません。プラットフォームが拒否した、または例外を送出した編集は、編集をサポートしないプラットフォームと同じように新しいメッセージにフォールバックします。すべての投稿と編集は `plugin_message plugin=<id> op=post|edit ...` としてログに記録されます。

各サーフェスの動作:

| サーフェス | `post_message` | `edit` |
|---|---|---|
| 編集できるメッセージングプラットフォーム（Telegram、Discord、Slack、Matrix など） | チャット／スレッドへのアダプターの通常の `send` | 同じメッセージをその場で編集する |
| 編集をサポートしないプラットフォーム（SMS、メール、`edit_message` を実装していないプラグインアダプター） | `send` | 新しいメッセージを送信し、ハンドルはそれを追う（`edited: False, fallback: True`） |
| Desktop と Ink TUI | セッションの閲覧者へのキー付き通知（Desktop のトースト、TUI のステータスライン） | その通知をその場で置き換える |
| 従来の CLI（`hermes chat`） | 通知行としてキューに入れられ、次のターンの終わりに出力される（従来の CLI にはライブの通知枠がない） | 編集のたびに別の行がキューに入る |

テキストはプラットフォームのメッセージ上限に合わせて省略記号付きで切り詰められます（分割されることはありません）。プレーンテキストです。アダプターが独自の書式でレンダリングすることはありますが、プラグインはそれに依存すべきではありません。メッセージはすでにプラグインを呼び出した会話にしか届かないので、権限は不要です。`ctx.post_message` は、プロセス外のプラグインホスト（`plugins.isolation: host`）で実行されるプラグインでは利用できません。ハンドルはゲートウェイのイベントループとライブのアダプターにバインドされているためです。

## メッセージの注入 {#injecting-messages}

プラグインは、`ctx.inject_message()` を使って、CLI の会話、既知のゲートウェイセッション、または指定したチャットの新しいゲートウェイセッションにメッセージを注入できます。

```python
# The chat the plugin is working for (see "Without a session key" below; prefer passing session_key)
ctx.inject_message("New data arrived from the webhook", role="user")

# Existing gateway conversation
ctx.inject_message(
    "New data arrived from the webhook",
    role="user",
    session_key="agent:main:telegram:dm:123456789",
)

# Start (or continue) the session for a chat/thread, e.g. a Discord forum post the plugin just opened
ctx.inject_message(
    "Review the attached spec and post a plan here.",
    origin={
        "platform": "discord",
        "chat_id": "1290000000000000000",   # the forum post (thread) id
        "chat_type": "thread",
        "thread_id": "1290000000000000000",
        "user_id": "123456789012345678",    # who the turn runs as; must pass the gateway's allowlist
        "user_name": "Desk Orchestrator",
    },
)
```

**シグネチャ:** `ctx.inject_message(content: str, role: str = "user", *, session_key: str | None = None, origin: Mapping[str, Any] | None = None) -> bool`

プラグインはすべてのローカルサーフェス（従来の CLI、TUI、Desktop）でゲートウェイ内で実行されるので、注入されたメッセージが進行中の作業をキャンセルすることはありません。

- 対象のセッションが**アイドル**の場合、メッセージは新しいターンを開始します。
- **ターンの途中**の場合、メッセージはそのセッションのキューで待機し、現在のターンが終わった後に次のターンとして実行されます。実行中のツールは中断されません。
- `"user"` 以外のロールの場合、内容には `[role]`（例: `[system] ...`）が前置きされます。
- ホストがメッセージを受け入れた場合は `True` を返します（以下のホストごとのルールを参照）。

セッションキーなし（`session_key` と `origin` の両方を省略）で、プラグインがゲートウェイ（またはそのプラグインホスト）内で実行されている場合 — 従来の CLI、TUI、Desktop のセッションではプラグインはそこで実行されます — ゲートウェイがチャットを選びます。

1. プラグインがローカルセッション自身の呼び出し（そのスラッシュコマンド、そのセッションのために発火したフック、またはそのターン内のツールやフック）を処理している場合、メッセージはそのセッションに送られます。開いているセッションがいくつあっても、推測することはありません。
2. 呼び出し元のコンテキストにセッションがない場合（タイマー、バックグラウンドスレッド — プラグインが開始したスレッドは呼び出しのコンテキストを継承しません）、プラグインのプロファイルのローカルセッションのうち、従来の CLI（`hermes chat`）で開始され閲覧者が接続している唯一のセッションに送られます。TUI と Desktop のセッション、メッセージングのチャット、他のプロファイルのセッションがこの方法で選ばれることはありません。
3. それ以外の場合（該当するセッションがない、または複数ある）は拒否されます。`False` が返り、どこにも何も注入されず、ゲートウェイのログにプラグイン名とセッションキーを渡すべき旨が記録されます。

選ばれたセッションは、キー付きの注入と同じようにテキストをキューに入った入力として受け取ります。以下の `allow_gateway_injection` 権限が必要です。それが従来の CLI のセッションである場合、ターンとその応答はセッションのトランスクリプトに含まれますが、接続している従来の CLI には表示されません。メッセージングのチャットのためにキーなしで行われた呼び出しは、従来どおり拒否されます。`session_key` を渡すことを推奨します。キーなしの呼び出しはチャットが曖昧なときに拒否されることがあり、キーなしの注入は将来廃止される可能性があります。

キー付きのゲートウェイモード:

- `session_key` は既存のゲートウェイセッションを識別するものでなければなりません。CLI のセッション ID ではなく、安定したルーティングキーです。
- Hermes はそのセッションに保存されたプラットフォーム、チャット、スレッド、プロファイル、会話履歴を再利用します。プラグインは `session_key` を通じてそのルートを変更できません。
- セッションはプラグイン自身のプロファイルに属していなければなりません。別のプロファイルのチャットやローカルセッションのキーには即座に `False` が返り、理由がゲートウェイのログに記録されます。
- Hermes はディスパッチの前に、保存されたルートをゲートウェイの現在の認可ルールに照らして再確認します。
- アダプター時点または上流の認可判断だけに依存していたルートは、Hermes が現在のコアの許可リスト、ペアリング、または明示的な全許可の設定から再検証できない限り拒否されます。
- 注入されたテキストは常に会話の入力です。スラッシュコマンドを呼び出したり、ツールを承認したり、保留中の確認や明確化のプロンプトを解決したりすることはできません。
- ディスパッチが保留中の間、ルートと会話は固定されます。トピックの復元によってルートが変わった場合、または処理が始まる前にセッションがローテーションした場合、Hermes はリクエストを破棄します。
- リクエストはプラットフォームアダプターの通常のメッセージ経路に入ります。アクティブなセッションでは、競合するターンを開始するのではなく、既存のビジーセッションのキューを使います。
- ライブのゲートウェイが非同期ディスパッチのためにリクエストを受け入れた場合は `True` を返します。これはエージェントのターンやプラットフォームへの配信が完了したことを確認するものではありません。
- `session_key` と `origin` の両方が指定された場合、どちらも指定されずチャットが選ばれなかった場合（上記）、権限が付与されていない場合、またはリクエストを受け入れられるライブのホストがない場合は `False` を返します。非同期の受け入れの後に判明した不明またはルーティング不能なセッションキーは、ゲートウェイのログに書き込まれます。

`origin` でセッションを開始する（ゲートウェイモードのみ）:

- `origin` は保存されたセッションの origin（`SessionSource.to_dict()`）と同じ形を使います。`platform` と `chat_id` は必須で、`chat_type`、`thread_id`、`user_id`、`user_name`、`chat_name` およびその他の origin フィールドは任意です。まだセッションが存在しないときに使ってください — 例えば、ボットが作成したフォーラム投稿で、Discord のようなプラットフォームはそれを受信メッセージとしてボットに返すことがありません。
- セッションは通常のセッション経路を通じて**プラグイン自身のプロファイル**（プラグインマネージャーがそれを読み込んだプロファイル）に、新しいプロンプトと空の履歴で作成され、応答はそのチャット／スレッドに配信されます。そのチャットがそのプロファイルですでにセッションを持っている場合、テキストは人間のメッセージとまったく同じようにそこに加わる（ビジーセッションのキューが適用される）ので、同じスレッドでの人間のフォローアップは同じセッションを継続します。
- ゲートウェイのセッション権限の下（ゲートウェイが所有するすべてのセッション）では、プラグインのターンはそのセッションの永続的な FIFO にコミットされ、そこですでに受け入れられた人間のメッセージのコネクターの出所を借ります。そのため、チャットに**まだ人間のメッセージが受け入れられていない**（セッションがない、またはコミットされた人間のターンがないセッション）`origin` は拒否されます。`inject_message` は即座に `False` を返し、ゲートウェイのログには "has no human message admitted yet" と記録されます。Hermes が、プラットフォームが配信していないチャットのためにプラットフォームの出所を作り出すことはありません。そのチャットの最初のメッセージはプラットフォーム（または人間）から送られるようにし、その後 `origin` または `session_key` で注入してください。
- プラグインが別のプロファイルを対象にすることは決してできません。origin が別の `profile` を指定している場合、ゲートウェイ自身のプロファイルルーティングがそのチャットを別のプロファイルに送る場合、プラグインのプロファイルがこのゲートウェイで提供されていない場合、またはそのプロファイルに `platform` の接続済みアダプターがない場合は `False` が返されます（理由はゲートウェイのログに記録）— 別のプロファイルのボットが借用されることはありません。マルチプレックスのゲートウェイでは、ターンはプラグインのプロファイルのホームとシークレットの下で実行されます。
- 認可はバイパスされません。origin のユーザーは、そのプロファイルについてゲートウェイの現在の許可リスト、ペアリング、または全許可の設定を通過しなければなりません。`session_key` の経路と同じチェックです。認可された `user_id` を選んでください。そうでない場合、リクエストはディスパッチ時に破棄され、ログに記録されます。
- 同じテキストのルールが適用されます。スラッシュコマンド、承認、プロンプトの解決はできません。以下の権限は、プラグイン自身のプロファイルの `config.yaml` から読み取られます。

Ink TUI（`hermes --tui`）とデスクトップ／ダッシュボードのチャットは 3 つ目のホストです。従来の CLI の参照を設定せず、メッセージングゲートウェイのインジェクターにも登録しません — この 2 つのホストは分離されているので、ライブのゲートウェイが TUI を上書きする（またはその逆）ことはありません。一時的な UI のセッション ID ではなく、セッションの永続的な `session_key`（`ses_…` の ID）を渡してください。Hermes はそのセッションのプロンプトキューにテキストを入れます。ビジーなセッションは次のターンのためにメッセージを保持し、アイドルなセッションはターンを開始します。ライブの TUI セッションではないキーは、メッセージングゲートウェイが実行中であればそちらに任され、別のチャットに再ルーティングされることはありません。

これにより、リモートコントロールのビューア、メッセージングブリッジ、Webhook レシーバーといったプラグインが、外部ソースから会話にメッセージを供給できるようになります。

ゲートウェイへの注入は、外部のメッセージングプラットフォームにエージェントの応答を送ることがあります。これはすべてのプラグインでデフォルトで無効です。`config.yaml` でプラグインごとに付与してください。

```yaml
plugins:
  entries:
    my-plugin:
      allow_gateway_injection: true
```

:::warning
ゲートウェイへの注入は信頼できるプラグインにだけ付与してください。Hermes はこのホスト API の権限を確認し、既存のセッションルートまたはプラグイン自身のプロファイル内の新しいセッションに制限し、現在の認可を適用しますが、Python プラグインはインプロセスで実行され、この設定はサンドボックスではありません。
:::

:::note
このプラグイン API は、外部プロセス向けの公開 HTTP エンドポイントや CLI コマンドを公開しません。プラグインは対象のゲートウェイの `session_key` をすでに知っている必要があります。例えば、自身の信頼できる設定や、以前に保持したセッション状態からです。
:::

## ゲートウェイ内のプラグインコード

すべてのチャットとすべてのローカルセッション（CLI、TUI、Desktop）のフック、ツール、注入されたターンは、起動したクライアントではなく 1 つのゲートウェイプロセス内で実行されます。従来の CLI でのプロセスレベルの習慣のいくつかは、そのままでは通用しません。

- `os.getcwd()` はゲートウェイ自身のディレクトリで、すべてのセッションで共有されます。Hermes がそれを変更することはありません。フックやツールが実行されているセッションの作業ディレクトリには `ctx.session_cwd()` を使ってください。CLI/TUI/Desktop のセッションが開始されたワークスペース、またはチャットの場合はプロファイルのターミナルの cwd です（ない場合は `""`）。サブプロセスには `cwd=` として渡してください。これは実行中のターンからセッションを読み取るので、セッションのために実行されるコード（ターンのフック、ツールハンドラ、承認フック）で動作します。`pre_gateway_dispatch` のような受信時のフックは、メッセージがいずれかのセッションに受け入れられる前に実行されます。そこではプロファイルのターミナルの cwd を返し、設定されていない場合は `""` を返します。
- `ctx.session_surface()` は呼び出し元のセッションのクライアントを示します: `"cli"`、`"tui"`、`"desktop"`、`"acp"`、またはメッセージングプラットフォーム（`"telegram"` など）。セッション外では `""` です。
- `hermes_cli.plugin_session.session_client()` は、呼び出し元のローカルセッションを作成したクライアントを、`session.create` で自ら名乗ったとおりに返します（`{"name": "my-client", "version": "1.2.0"}`。`version` は送られた場合のみ）。または `None` を返します: セッション外、メッセージングのチャット、名前を送らなかったクライアント（TUI、Desktop、従来の CLI、ACP は現在名前を送りません）の場合です。`plugins.isolation: host` ではプラグインのプロセスにセッションのコンテキストがないので、代わりにシステムプロンプトのセクションで `info["client"]` を読んでください。名前はクライアントが選んだラベルであり、認証ではありません。それを理由に何かを付与したり、チェックを省略したりしてはいけません。
- `print()` の出力はターミナルではなく、ゲートウェイの `gateway-stdio.log` に送られます。代わりに `logging` を使ってください。
- `input()` は end-of-file を受け取ります。ゲートウェイにはターミナルがありません。ツールの結果やスラッシュコマンドを通じて尋ねてください。

従来の CLI のプロセス自体の中（例えばプラグインのスラッシュコマンド）では、`ctx.session_cwd()` は起動ディレクトリで、`ctx.session_surface()` は `"cli"` です。

## 独自のチャットクライアントを持つプラグイン

プラグインは、独自のチャットフロントエンドを起動するコマンド（`ctx.register_cli_command` による `hermes <name>`）を追加できます。フロントエンドは `hermes gateway attach --json` でプロファイルの共有ゲートウェイに接続します（[プログラムによる統合](/docs/developer-guide/programmatic-integration) を参照）。

このコマンドは、プラグインが Hermes のプロセス内で実行される場合（`plugins.isolation: in_process`、デフォルト）にのみ存在します。`plugins.isolation: host` では Hermes は `ctx.register_cli_command` をスキップし、`hermes <name>` はコマンドになりません。

3 つのインポートが、Hermes のターミナルクライアントのランチャーが自身のクライアントの周りで行うことを担います。

- `hermes_cli.gateway_chat.shell_client_env()` は、コマンドを実行したシェルの許可リストに含まれる変数を dict として返します。環境を変更する前にランチャー内で呼び出し、その dict を `session.create` の `client_env` として送ってください。するとセッションのコマンドは、ユーザーの `PATH`、仮想環境、ツールチェーンの変数とともに実行されます。認証情報に見える名前は除外され、ゲートウェイも独自のルールで dict を再度フィルタリングします。
- `hermes_cli.chat_launch.hermes_command(args)` は、この Hermes のインストールを `args` 付きで実行する argv を返します。`PYTHONPATH` は不要で、子プロセスは継承した `HERMES_HOME` が指定するプロファイルを提供します。`PATH` 上の `hermes`（別のインストールかもしれない）や `sys.executable`（更新で置き換わるかもしれない）の代わりに、`hermes gateway attach` 用としてフロントエンドに渡してください。
- `hermes_cli.chat_launch.print_exit_summary(session_id, command=...)` は、チャットの終了時に Hermes が表示する行を出力します: セッションを再開する方法、その ID、タイトル、メッセージ数、トークン数です。フロントエンドの終了後に、自分のコマンド名を付けて呼び出してください。メッセージのないセッションでは何も出力しません。これらの行は再開方法として `--resume <id>`（短い ID）と `-c "<title>"` の 2 つを示すので、コマンドは両方を受け付ける必要があります。

```python
import json
import os
import subprocess
import tempfile


def _arguments(parser):
    # The exit summary prints `hermes my-client --resume <id>` and `hermes my-client -c "<title>"`.
    parser.add_argument("--resume", "-r", metavar="ID")
    parser.add_argument("--continue", "-c", dest="title", metavar="TITLE")


def _session_to_resume(args):
    wanted = args.resume or args.title
    if not wanted:
        return None  # a new chat: a new profile has no session store to open yet
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    db = SessionDB(db_path=get_hermes_home() / "state.db", read_only=True)
    try:
        # The summary prints a short id: resolve_session_id takes a unique prefix.
        found = db.resolve_session_id(args.resume) if args.resume else db.resolve_session_by_title(args.title)
    finally:
        db.close()
    if not found:
        raise SystemExit(f"hermes my-client: no session matches {wanted!r}")
    return found


def _run_chat(args):
    from hermes_cli.chat_launch import hermes_command, print_exit_summary
    from hermes_cli.gateway_chat import shell_client_env

    env = dict(os.environ)
    env["MY_CLIENT_ENV"] = json.dumps(shell_client_env())
    env["MY_ATTACH_COMMAND"] = json.dumps(hermes_command(["gateway", "attach", "--json", "--start"]))
    env["MY_RESUME"] = _session_to_resume(args) or ""  # the frontend sends it with session.resume
    # One file for this launch: the frontend writes {"session_id": ...} there for the session it shows.
    fd, session_file = tempfile.mkstemp(prefix="my-client-", suffix=".json")
    os.close(fd)
    env["MY_SESSION_FILE"] = session_file
    try:
        subprocess.run(["my-client"], env=env)
        with open(session_file, encoding="utf-8") as f:
            shown = f.read()
    finally:
        os.unlink(session_file)
    if shown:  # empty when the client left before it opened a session
        print_exit_summary(json.loads(shown)["session_id"], command="hermes my-client")


def register(ctx):
    ctx.register_cli_command("my-client", "Chat in my client", _arguments, _run_chat)
```

### `-w`: 新しいワークツリーでセッションを実行する

同じランチャーで、`hermes --tui -w` と同じように `-w` を提供できます。`hermes_cli.session_worktree` の 3 つの関数を呼び出します。

- クライアントの起動前に `create()`。ユーザーがいるリポジトリの新しい git ワークツリーを、`<repo>/.worktrees/` 配下に、独自の `hermes/` ブランチで作成します。`{"path", "branch", "repo_root", "base"}` を返すか、ツリーを作成できなかった場合は `None` を返します（理由はすでに出力済みです）。`path` を `session.create` の `cwd` として送ってください。
- 再開したセッションにクライアントが接続する前に `restore(session_id)`。完全なセッション ID を渡します。そのセッションが、以前の終了時に削除された `-w` のツリーで実行されていた場合、同じパスにツリーを再作成してそのパスを返します。それ以外の場合は `None` を返します。これがないと、ゲートウェイはそのセッションのすべてのターンを `workspace_unavailable` で拒否します。
- クライアントの終了後、`finally` ブロック内で `finish(info, session_id, ask=...)`。これにより、起動に失敗したクライアントがツリーを残すことはありません。正常終了後は `ask=True` を渡してください。するとコミットされていない、またはプッシュされていない作業があるツリーについて、ターミナルで keep / archive / discard の質問が出ます。クライアントがターミナルを返した後にだけ呼び出してください。もう何も使っていないクリーンなツリーは、ブランチとともに削除されます。保持されたツリーは `session_id` に引き渡されるので、そのセッションが再開可能な間は Hermes が保持します。

`git worktree add` で自分でツリーを作らないでください。Hermes は `.worktrees/` 配下の古いアイドルなツリーを定期的に削除します。ロックされたツリーを保持するのは、ロックの理由が Hermes の書いたものである場合だけなので、保存されたセッションがまだそこで実行されていても、あなたのツリーを削除してしまう可能性があります。

```python
import os

from hermes_cli.session_worktree import create, finish, restore

resume = _session_to_resume(args)  # the full session id, from the launcher above
info = create() if args.worktree else None
if args.worktree and info is None:
    raise SystemExit(1)
if resume and not args.worktree:
    restore(resume)
code, session_id = None, resume
try:
    code, session_id = run_my_client(cwd=info["path"] if info else os.getcwd(), resume=resume)
finally:
    if info:  # also when the client did not start, or Ctrl+C reached the launcher
        finish(info, session_id, ask=code == 0)
```

## プラグインから MCP サーバーを呼び出す {#calling-mcp-servers-from-plugins}

`ctx.call_mcp()` を使うと、プラグインはユーザーが設定した MCP サーバーのツールを — 任意のフックやツールハンドラから同期的に — 呼び出せます。呼び出しは Hermes の既存のネイティブ MCP クライアントを経由します（モデルが呼び出す MCP ツールと同じ接続、信頼ティアのゲート、サーキットブレーカー、再接続ロジック。並行するクライアントを使うことは決してありません）。

```python
result = ctx.call_mcp(
    "knowledge_rag",            # server name from mcp.servers
    "query_knowledge",          # tool on that server
    {"query": "deploy runbook"},
    timeout=30,                 # seconds; clamped to 1–600
)
if result["ok"]:
    print(result["result"])
else:
    print("MCP error:", result["error"])
```

**シグネチャ:** `ctx.call_mcp(server: str, tool: str, arguments: dict | None = None, timeout: float = 30) -> dict`

安定したエンベロープを返します: `{"ok": True, "result": ...}`（サーバーが提供する場合は `structuredContent` も含む）または `{"ok": False, "error": "..."}`。約 64 KB を超える結果は切り詰められ、`"truncated": True` のフラグが立てられます。

### セキュリティ: デフォルトオフ、サーバーごとの許可リスト

プラグインは**デフォルトでは MCP にアクセスできません**。オペレーターは `config.yaml` で各サーバーを明示的に付与する必要があります。

```yaml
plugins:
  entries:
    my-plugin:
      mcp_allowlist: ["knowledge_rag", "github"]
```

- リストにないサーバーを呼び出すと、設定すべき正確な設定キーを示す `PermissionError` が送出されます。
- 付与はサーバーごと・プラグインごとです — 設定されたすべてのサーバーに対する包括的な権限になることはなく、`"*"` のワイルドカードは受け付けられません。
- すべての呼び出しにはタイムアウトが強制されるので（デフォルト 30 秒）、ハングした MCP サーバーが、それを呼び出したフックやツールのパイプラインを止めることはできません。
- MCP サーバーは信頼できないコンテンツを返します。`result` は指示ではなくデータとして扱ってください — 検証なしで特権的な判断（承認、コマンドの実行）に渡してはいけません。

:::warning
`mcp_allowlist` を付与すると、プラグインはその MCP サーバーに対してモデルと同じアクセス権を持ちます — サーバーが公開する書き込み可能なツールも含みます（サーバーの `trust` ティアのゲートに従います）。プラグインが本当に必要とするサーバーだけを付与してください。
:::

ハンドラの契約、スキーマの形式、フックの挙動、エラー処理、よくある間違いについては、**[完全なガイド](../../developer-guide/plugins/index.md)** を参照してください。
