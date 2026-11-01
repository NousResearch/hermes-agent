---
sidebar_label: "プラグインを作る"
slug: /developer-guide/plugins
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
| **ネイティブデスクトップアプリ**の拡張（ペイン、ページ、ステータスバー、パレット、テーマ） | [Desktop Plugin SDK](../desktop-plugin-sdk.md) |
| **Webダッシュボード**の拡張（タブ、シェルスロット、テーマ） | [Extending the Dashboard](../../user-guide/features/extending-the-dashboard.md) |
| **LLM / 推論バックエンド**（新しいプロバイダー） | [Model Provider Plugins](../model-provider-plugin.md) |
| **ゲートウェイチャネル**（Discord/Telegram/IRC/Teams など） | [Adding Platform Adapters](../adding-platform-adapters.md) |
| **メモリバックエンド**（Honcho/Mem0/Supermemory など） | [Memory Provider Plugins](../memory-provider-plugin.md) |
| **コンテキスト圧縮エンジン** | [Context Engine Plugins](../context-engine-plugin.md) |
| **画像生成バックエンド** | [Image Generation Provider Plugins](../image-gen-provider-plugin.md) |
| **動画生成バックエンド** | [Video Generation Provider Plugins](../video-gen-provider-plugin.md) |
| **Web検索 / 抽出バックエンド** | [Web Search Provider Plugins](../web-search-provider-plugin.md) |
| **クラウドブラウザバックエンド**（Browserbase風のCDPセッションプロバイダー） | [Browser Provider Plugins](../browser-provider-plugin.md) |
| **computer-useドライバー**（`computer_use`ツールの背後にあるデスクトップ操作） | [Computer-useバックエンドプラグイン](#computer-use-backend-plugins) — `ctx.register_computer_use_provider()` |
| **シークレットマネージャーバックエンド**（vault / パスワードマネージャー / OSキーストア） | [Secret Source Plugins](../secret-source-plugin.md) |
| **ダッシュボードのOIDC/認証プロバイダー** | [Web Dashboard — custom providers](../../user-guide/features/web-dashboard.md#custom-providers) — `ctx.register_dashboard_auth_provider()` |
| **TTSバックエンド**（任意のCLI — Piper、VoxCPM、Kokoro、ボイスクローニングなど） | [TTS custom command providers](../../user-guide/features/tts.md#custom-command-providers) — 設定駆動、Python不要 |
| **STTバックエンド**（カスタムwhisper / ASR CLI） | [Voice Message Transcription](../../user-guide/features/tts.md#voice-message-transcription-stt) — `HERMES_LOCAL_STT_COMMAND`をargvトークン化されたテンプレートに設定 |
| **MCP経由の外部ツール**（filesystem、GitHub、Linear、任意のMCPサーバー） | [MCP](../../user-guide/features/mcp.md) — `config.yaml`に`mcp_servers.<name>`を宣言 |
| **ゲートウェイイベントフック**（起動時、セッションイベント、コマンドで発火） | [Event Hooks](../../user-guide/features/hooks.md#gateway-event-hooks) — `~/.hermes/hooks/<name>/`に`HOOK.yaml` + `handler.py`を配置 |
| **シェルフック**（イベント時にシェルコマンドを実行） | [Shell Hooks](../../user-guide/features/hooks.md#shell-hooks) — `config.yaml`の`hooks:`配下に宣言 |
| **追加のスキルソース**（カスタムGitHubリポジトリ、プライベートスキルインデックス） | [Skills](../../user-guide/features/skills.md) — `hermes skills tap add <repo>` · [tapの公開](../../user-guide/features/skills.md#publishing-a-custom-skill-tap) |
| ファーストクラスの**コア**推論プロバイダー（プラグインではない） | [Adding Providers](../adding-providers.md) |

設定駆動（TTS、STT、MCP、シェルフック）やドロップインディレクトリ（ゲートウェイフック）スタイルを含む、すべての拡張サーフェスを一覧でまとめた[Pluggable interfaces table](../../user-guide/features/plugins.md#pluggable-interfaces--where-to-go-for-each)も参照してください。
:::

:::caution サードパーティ製品のプラグインはコアツリーではなくスタンドアロンで配布する
**他者の製品やプロジェクト**を統合するプラグイン — オブザーバビリティ/メトリクスバックエンド、ベンダーSaaSコネクタ、アナリティクスダッシュボード、有料サービスとの連携 — は、`NousResearch/hermes-agent`にマージするのではなく、**スタンドアロンのプラグインリポジトリ**として構築・配布されます。ユーザーはそれらを`~/.hermes/plugins/`に、またはpipのエントリポイント経由でインストールします。このガイドの内容はすべて、スタンドアロンリポジトリからでも同じように動作します。これは結合度と保守性に関する判断（コアは速く進化し、私たちはあなたのバックエンドを所有していません）であり、品質の基準ではありません — 優れたプラグインであっても、独自のリポジトリに属するべき場合があります。Nous Research Discordの`#plugins-skills-and-skins`チャネルで宣伝してください。ポリシーについては[CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)を参照してください。
:::

## Portable Agent Plugins v1パッケージ {#portable-agent-plugins-v1-packages}

Hermesは、Agent Plugins v1.0.0形式をターゲットとするディレクトリパッケージもインストール・読み込みできます。これは、Hermesがすでに持っているポータブルなコンポーネントのための互換アダプターです。ネイティブの`plugin.yaml` + `register(ctx)`プラグインを置き換えるものではありません。

```text
my-portable-plugin/
├── plugin.json
├── skills/
│   └── summarize/
│       ├── SKILL.md
│       └── references/
└── mcp.json
```

ポータブルパッケージは通常のワークフローでインストールして有効化します。

```bash
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
```

ポータブルパッケージは、明示的に有効化しない限りインストール後は無効のままです。有効化されたパッケージは、直下の`skills/*/SKILL.md`ディレクトリと、ルートの`mcp.json`にあるstdio MCPサーバーを提供できます。スキルは読み取り専用で名前空間付きであり、`skills_list`と`skill_view`を通じて読み込まれます。MCPコマンドは、1つの実行ファイルトークンと別個の引数リストとして渡され、シェルを経由することはありません。完全修飾スキル名を調べるには`skills_list`を使ってください。ポータブルスキルの名前空間は`agent-plugin-<slug>-<hash>`という決定論的な形式を持ち、検出されたプラグインキーから導出されるため、サニタイズ後の名前が衝突することはありません。ポータブルパッケージのMCPサーバーは、ユーザー自身の`mcp_servers`ブロックと同じルールで、`mcp.json`で付けられた名前をそのまま保持します。そのため、モデルに見える`mcp__<server>__<tool>`という名前は、ツールの動詞部分をプロバイダーの64文字上限内に収めたままになります。サーバー名の重複は読み込み時の衝突として扱われます。`config.yaml`のサーバーがパッケージより優先され、先に読み込まれたパッケージが次のパッケージより優先されます。負けた側は、両者の名前を示す警告とともにスキップされます。

Hermesは、`plugin.json`、Agent Skillsのフロントマター、固定されたコンポーネントの配置場所、`mcp.json`、解決済みパス、シンボリックリンクの封じ込めをローカルで検証します。パッケージの読み込み中にJSONスキーマを取得することはありません。不正なスキルやMCPエントリは、有効な兄弟コンポーネントが読み込める場合、それぞれの境界でスキップされます。`PLUGIN_ROOT`は解決済みのパッケージルートを指します。`PLUGIN_DATA`は、Hermesが管理するプロファイル単位の書き込み可能ディレクトリを指します。ポータブルMCPの`env`で宣言された値は、可視のパッケージデータであり、シークレットの保存手段ではありません。`mcp.json`に認証情報を置かないでください。

現在のポータブルサブセットは、stdioとStreamable HTTPのMCPエントリをサポートしています。ポータブルな`streamable-http`エントリは、Hermesの既存のネイティブリモートMCPクライアント（URLベースの`mcp_servers`設定を支えているのと同じランタイム）を通じてルーティングされ、v1の境界ルールが適用されます。URLはユーザー情報やフラグメントを含まない絶対http(s)でなければならず、平文のHTTPは`localhost`/ループバックホストに対してのみ受け付けられ、設定されたヘッダーはクロスオリジンのリダイレクトをまたいで転送されることはありません。レガシーの`sse`エントリは報告されたうえでスキップされます。Agent Plugins v1は、信頼、権限、来歴、サンドボックスを定義していません。パッケージを有効化すると、その指示とローカル実行ファイルに対して、インストール済みの他のHermesプラグインと同じフルトラストの扱いが与えられます。

パッケージは、ユーザーが`config.yaml`で`trust: untrusted`を指定するのと同じように、自身のMCPサーバーの1つをゲートするようHermesに求めることができます。お金を使う、取引する、メッセージを送る、アカウントを変更するといったツールを持つサーバーに使ってください。そうすれば、スキルの指示だけに頼るのではなく、書き込み可能な呼び出しごとにユーザーが承認するようになります。

```json
{
  "extensions": {
    "com.nousresearch.hermes": {
      "servers": {
        "trade": { "trust": "untrusted" }
      }
    }
  }
}
```

サーバー名は`mcp.json`のエントリと一致していなければなりません。`untrusted`の場合、そのサーバーへのツール呼び出しのうち`readOnlyHint: true`の注釈がないものはすべて、先にユーザーに確認し、誰も応答できない場合（cron、無人実行）はフェイルクローズします。他に受け付けられる値はデフォルトの`full`のみなので、パッケージはアクセスを狭めることはできても、広げることはできません。同じ名前の`config.yaml`サーバーは、trustを含めてパッケージのエントリを置き換えます。他のハーネスはこの拡張を無視します。`trust`は、同じサーバーエントリ内で`app`、`requires`、`liveness`と並べて置けます（[Application declarations](./application-declarations.md)を参照）。

[レンダリングされた仕様](https://agent-plugins.org/specification)では現在v1.0.0をWorking Draftとしていますが、[バージョン管理された仕様リポジトリ](https://github.com/agentplugins/agent-plugins-spec/blob/main/spec/1.0.0.md)ではPublishedとして記録されています。Hermesは、変わりうるどちらのステータスラベルでもなく、正規のv1.0.0スキーマ識別子と規範テキストに基づいて動作を決定します。これは明示的にサポートされたサブセットであり、Agent Pluginsへの完全準拠を主張するものではありません。

## ネイティブプラグインの互換性コントラクト {#native-plugin-compatibility-contract}

ネイティブの`plugin.yaml` + `register(ctx)`プラグインは、単一のグローバルなプラグインAPI番号ではなく、振る舞いによって保護されています。Hermesは`PLUGIN_API_VERSION`を公開せず、マニフェスト全体での`api:`の一致を要求せず、無関係な値にAPIバージョンを付けることもありません。ドキュメント化された振る舞いを使うプラグインは、通常のHermesアップグレード後も動作し続けるはずです。

互換性ルールは次のとおりです。

- **追加的に進化する。** ドキュメント化された`PluginContext`メソッドが削除されたり名前が変わったりすることはありません。新しいパラメータはオプションでデフォルト値を持ち、キーワード専用にすべきです。既存の戻り値フィールドが削除されたり、黙って型が変わったりすることはありません。
- **フックのペイロードはキーワードペイロード。** 新しいフックデータはキーワードフィールドとして追加され、既存フィールドの意味や位置を変えることはありません。Hermesはコールバックのシグネチャを調べます。レガシーなコールバックは自身が宣言したフィールドを受け取り、`**kwargs`を持つコールバックは現在のペイロード全体を受け取ります。新しいプラグインは`**kwargs`を受け取るべきです。そうすれば、シグネチャを変更することなく追加されたデータを利用できます。
- **マニフェストは追加に対して開かれている。** 未知の`plugin.yaml`フィールドは無視されます。そのため、古いHermesリリースでも、新しいリリースで導入されたメタデータを含むマニフェストを持つプラグインを読み込めます。ただし、プラグインのコード自体がサポートされているランタイムの振る舞いを使っていることが前提です。
- **プロバイダーインターフェースはデフォルトを通じて拡張される。** 新しいプロバイダーメソッドにはデフォルト実装があります。新しいコールバックコンテキストはオプションであり、シグネチャの検査によってプロバイダーがそれを受け付けると分かった場合にのみ転送されます。抽象メソッドや無条件に転送される引数を追加するには、一斉切り替えのシグネチャ変更ではなく、移行期間が必要です。
- **境界をまたぐコントラクトにバージョンを付ける。** ケイパビリティは、ワイヤーペイロードや永続化フォーマットを定義する場合（例えばオブザーバーのペイロードやsecret-sourceの状態）、独自のスキーマバージョンを持つことができます。そのローカルなスキーマ内ではフィールドを追加的に保ってください。永続化されたプラグインの状態と設定は読み取り可能なままにするか、明示的なマイグレーションを同梱しなければなりません。古いフォーマットで書かれた再開セッションも、引き続きリプレイできなければなりません。無関係なコールバックやコンテキストの値にバージョンリテラルを追加しないでください。

このコントラクトは、ドキュメント化されたサーフェスのみを対象とします。コアの関数、メソッド、モジュール属性、プライベートなテーブルを実行時に置き換えたりラップしたりすること（`AIAgent.<method>`への代入、Hermesモジュールへの`setattr`、`sys.modules`やコアのdictへの書き込み）は、サポートされた拡張ポイントではありません。内部が変わるたびに壊れ、同じ継ぎ目にパッチを当てる他のすべてのプラグインと衝突します。プラグインカタログは、受け入れ時にこれを拒否します（`hermes plugins
validate`の`no core override`チェック）。必要な公開フックが欠けている場合は、それを説明するissueを作成してください。

### 非推奨化ポリシー {#deprecation-policy}

ドキュメント化されたネイティブプラグインの振る舞いは、以下のすべてを満たす場合にのみ非推奨にできます。

1. プラグインガイドとリリースノートに、代替手段と移行手順があること
2. 代替手段と最も早い削除リリースを示す警告が、プロセスごとに最大1回出力されること
3. 古い振る舞いが、少なくとも後続の2つのマイナーリリースを通じてサポートされること
4. その期間を通じて、レガシーのパスと代替手段の両方について、振る舞いベースの互換性カバレッジがあること

期間終了後の削除には、永続化されたデータや再開可能なセッションに必要なマイグレーションを含めなければなりません。実際には、削除よりも追加的なエイリアスやアダプターが好まれます。

Hermesは、隔離された`HERMES_HOME`から検出される凍結された外部プラグインのフィクスチャを用いて、このコントラクトを強制しています。これらのテストは`PluginManager`を通じてプラグインを読み込んで呼び出し、内部シンボルのリストやソースコードの形ではなく、実際の登録結果とコールバックの結果をアサートします。

### 2026年9月のモジュール分割: 旧インポートパスの削除 {#sep-2026-module-decomposition-old-import-paths-removed}

Hermesの内部は、2026年9月に`<stem>_<topic>`という兄弟モジュール群に分割されました（PR #102117）。**内部のインポートパスは、上記のプラグインコントラクトの一部であったことは一度もありません。** 一時的な互換レイヤーが2026-09-14まで古いパスを解決し続けていましたが、それは削除されました。そのため、古いパスをまだインポートしているプラグインは`ImportError`で読み込みに失敗します（理由は`hermes plugins list`に表示されます）。

そのようなプラグインを修正するには、現在その名前を定義しているモジュールからインポートするか、より良い方法として、内部ではなく`ctx`とドキュメント化されたABCを使ってください。旧パスから新パスへの完全な対応表は、互換レイヤーを含んでいた最後のコミットの[`COMPAT_MANIFEST.md`](https://github.com/NousResearch/hermes-agent/blob/5912ed81ed9/COMPAT_MANIFEST.md)にあります。

## 何を作るのか {#what-youre-building}

2つのツールを持つ**電卓（calculator）**プラグインです。
- `calculate` — 数式を評価する（`2**16`、`sqrt(144)`、`pi * 5**2`）
- `unit_convert` — 単位を変換する（`100 F → 37.78 C`、`5 km → 3.11 mi`）

さらに、すべてのツール呼び出しをログに記録するフックと、バンドルされたスキルファイルも追加します。

## ステップ1: プラグインディレクトリを作成する {#step-1-create-the-plugin-directory}

ディレクトリを作成し、ステップ2に進みます。

```bash
mkdir -p ~/.hermes/plugins/calculator
cd ~/.hermes/plugins/calculator
```

### Plugin Doctorで検証する {#validate-with-plugin-doctor}

`hermes plugins doctor [path-or-id]`は、Hermes自身が使っているのと同じディレクトリ検出、マニフェストパーサー、名前空間付きインポート、`register(ctx)`、フックレジストリ、ツールレジストリを実行します。無効なフック名、`**kwargs`を受け取らないコールバック、登録の失敗、そして宣言されたツール/フックと実際に登録されたものとのずれを報告します。`--ci`を渡すと、エラー時に非ゼロで終了します。

```bash
hermes plugins doctor . --ci
```

Doctorは一時的な`HERMES_HOME`を使い、チェック後にプラグインの登録状態を元に戻し、登録の実行中に意図しないネットワークアクセスを検出するため、Pythonからの直接のソケット接続をブロックします。これはサンドボックスではありません。プラグインのコードは現在のユーザーの権限でプロセス内で実行され、サブプロセスを起動することもできます。そのため、インポートしてもよいと信頼できるコードに対してのみDoctorを実行してください。

## ステップ2: マニフェストを書く {#step-2-write-the-manifest}

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
capabilities:          # 要求する特権的なホストサーフェス（同意フロー）
  - tools.override     # 組み込みツールを置き換える（ユーザーの同意が必要）
  - llm.model_override # ホスト所有のLLM呼び出しのモデルを選ぶ
```

### ケイパビリティを宣言する {#declaring-capabilities}

プラグインが特権的なホストサーフェス — 組み込みツールの上書き、`ctx.llm`呼び出しのモデル選択など — を必要とする場合は、`capabilities:`で宣言してください。インストール/有効化の時点でユーザーにそのリストが表示され、一度だけ同意します。後のバージョンでケイパビリティが追加された場合、更新フローでは追加分についてのみ再度確認します。宣言されていない、または同意されていないケイパビリティは単にオフになります（フェイルクローズ）。そのため、**使う前に確認し、グレースフルに劣化させてください**。

```python
def register(ctx):
    if ctx.has_capability("tools.override"):
        ctx.register_tool(..., override=True)
    else:
        ctx.register_tool(...)   # 衝突しない名前で登録する
```

既知のケイパビリティID: `tools.override`、`llm.provider_override`、`llm.model_override`、`llm.agent_id_override`、`llm.profile_override`、`llm.task_override`（正規のレジストリは`hermes_cli/plugin_capabilities.py`を参照）。未知のIDは無視されます。以前のケイパビリティごとの設定キー（`plugins.entries.<id>.allow_tool_override`など）も引き続き動作しますが非推奨です — ユーザーが単一の監査可能な同意画面を得られるよう、代わりにケイパビリティを宣言してください。ケイパビリティは同意 + 監査であり、**サンドボックスではありません**。ホストのAPIサーフェスをゲートするだけで、それ以上のことはしません。

**pipで配布されるプラグイン**は、インストール後に`plugin.yaml`ディレクトリを持たないため、代わりにディストリビューションのメタデータで、対になる`hermes_agent.plugin_capabilities`エントリポイントグループを通じてケイパビリティを宣言します。各宣言は`<plugin-id>.<capability-id>`という名前を持ち、`hermes_agent.plugins`エントリポイントと同じオブジェクトを指します。

```toml
[project.entry-points."hermes_agent.plugins"]
calculator = "my_pkg:register"

[project.entry-points."hermes_agent.plugin_capabilities"]
"calculator.tools.override" = "my_pkg:register"
```

Hermesはコードをインポートせずにインストール済みのメタデータからこれらを読み取るため、pipインストールでも`hermes plugins capabilities`と同意フローは正確なままです。

### マニフェストv2リファレンス {#manifest-v2-reference}

`plugin.yaml`は、追加的な**v2スキーマ**（#64165）もサポートしています。すべてのフィールドはオプションです。`manifest_version`のないマニフェストはv1マニフェストであり、今後も完全にサポートされ続けます。未知のフィールドが読み込みを壊すことはありません — 警告とともに無視されます（前方互換性）。また、このHermesが理解するものより新しい`manifest_version`も、警告付きで読み込まれます。

| フィールド | 型 | 意味 |
|---|---|---|
| `manifest_version` | int | マニフェストの**ファイル形式**のバージョン。省略時 = `1`。現在の最大値: `2`。`api_version`とは独立。 |
| `api_version` | int | プラグインがターゲットとするランタイムの**プラグインAPI世代**（ctxサーフェス / フックシグネチャ）。意図的に`manifest_version`とは別の軸になっています — `api_version: 1`のプラグインがv2マニフェストを使うこともできます。 |
| `requires_plugins` | list | プラグイン間の依存関係: `- id: other-plugin`に、オプションで`version_range: ">=1.0,<2"`。**助言的**: 依存関係が欠けている場合は明確な警告がログに出ますが、プラグインは読み込まれます — 実行時に`ctx.has_plugin("other-plugin")`で確認してください。読み込み**順序**はこれらのエッジに従います。AがBを必要とする場合、Bの`register()`がAより先に実行されます（トポロジカルソート、同順位はアルファベット順。循環がある場合は警告を出してアルファベット順にフォールバック）。 |
| `python_dependencies` | list of str | 宣言されたPython要件（例: `"requests>=2.0,<3"`）。インストール時に同意を求めます。有効化時には、既存のコア、エクストラ、有効化済みプラグインの和集合とともに、候補をPMを通じて受け入れます。準備が成功すると環境と設定がトランザクショナルに公開され、失敗すると以前の選択と有効化セットが保持されます。同意しなかった場合、インストールされたプラグインは無効のままです。上限をピン留めしてください。 |
| `python_runtime` | str | `external` — プラグインが独自のインタープリタ/venvを管理します（サイドカーパターン）。Hermesは何もインストールせず、`pyproject.toml`にも手を付けません。 |
| `config_schema` | mapping | `plugins.entries.<id>.settings`配下のキーをJSONスキーマ風に記述したもの: `api_url: {type: str, default: "", description: "...", required: false}`。読み込み時に検証されます。不一致があっても、キー名と期待される型を示す実用的な警告がログに出るだけで、読み込み失敗にはなりません。型: `str`、`int`、`float`、`bool`、`list`、`dict`（およびJSONスキーマのエイリアス）と`secret`。デスクトップのPluginsタブの設定フォームもこれで駆動されます — [デスクトップの設定フォーム](#settings-form-in-the-desktop)を参照。 |
| `license` | str | SPDX形式のライセンスID（例: `MIT`）。 |
| `homepage` | str | プロジェクトのURL。 |
| `tags` | list of str | 自由形式の検出用タグ（例: `[gateway, telegram]`）。 |
| `provides_locales` | list | 言語パックの宣言: ID（`- pl`）または`{id, endonym, rtl}`マッピング。ローダーは対応する`locales/<id>[.tui\|.desktop].yaml`を自動的に登録します — [言語パックを同梱する](#ship-a-language-pack)を参照。 |

```yaml
# plugin.yaml — マニフェストv2の例
name: my-plugin
version: 1.2.0
manifest_version: 2
api_version: 1
license: MIT
homepage: https://github.com/owner/my-plugin
tags: [gateway, demo]
requires_plugins:
  - id: other-plugin
    version_range: ">=1.0,<2"
python_dependencies:
  - "somepkg>=1.0,<2"     # PMによる受け入れの前に同意を求める
config_schema:
  api_url: {type: str, default: "", description: "Service endpoint"}
```

:::note 共有の依存関係の受け入れ
プラグインのインストール時には、Python依存関係への同意を求めます。プラグインを有効化すると、コアの依存関係、エクストラ、有効化済みプラグインとともに、その要件がPMを通じて準備されます。パックの有効化も同じ受け入れトランザクションを使います。アクティブなプラグインを再インストールする場合は、公開前にステージングされた宣言に対して同意を求めます。拒否した場合、インストール済みのプラグインと選択された環境は保持されます。

インストーラーとPMの受け入れ処理は、サポートされていない`manifest_version`の値と、満たされていない`requires_hermes`制約を公開前に拒否します。
:::

### Python依存関係 {#python-dependencies}

ディレクトリプラグインは、独自のPyPIパッケージを持ち込めます。マニフェスト（上記の`python_dependencies`）か、より望ましくは`plugin.yaml`の隣に置いた`pyproject.toml`で宣言してください。

```toml
[project]
name = "my-plugin"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "somepkg>=1.0,<2",
    "other[extra]>=3.11",
]
```

両方が存在する場合は`pyproject.toml`が優先されます。Hermesがそれらをどう扱うか:

- **インストール / 有効化** — PMは、カスタムの`HERMES_HOME`ルートを含め、依存関係ホームを共有するすべてのプロファイルにわたって、コア、選択されたエクストラ、有効化済みプラグインの和集合を解決します。新しいプラグインは無効の状態でダウンロードされ、有効化の前にPython依存関係への同意が求められます。`pyproject.toml`は`python_dependencies`やレガシーの`pip_dependencies`より優先されます。
- **アトミックな公開** — PMは、有効化やアクティブなプラグインの置き換えを公開する前に、新しい環境世代を準備します。解決、ダウンロード、ビルドのいずれかが失敗した場合、以前の環境とプラグインの選択が保持されます。既存のプラグインが犠牲になることはありません。
- **更新でも和集合を保持** — `hermes update`は、新しい世代を準備する際に有効化済みプラグインを含めます。更新後のpip再インストールはありません。`hermes plugins update`は、アクティブな置き換えを準備してから、コードと依存関係の世代をまとめて入れ替えます。
- **要件の衛生管理** — 不正なPEP 508要件は拒否されます。環境マーカーは、ターゲットのインタープリタが評価できるようそのまま保持されます。`hermes-agent`への自己依存は、チェックアウトがHermesを提供するため省略されます。直接URLの要件は管理されません。それらにはプラグイン所有の外部ランタイムを使ってください。
- **`--no-deps`**は、依存関係への同意なしに新しいプラグインをダウンロードし、`--enable`を付けても無効のままにします。アクティブなプラグインを置き換える際のPMによる受け入れを迂回することはできません。
- **`--yes-deps`**は依存関係に関する質問に前もって回答するため、ヘッドレスなインストール（CI、SSHによる自動化、コンテナのエントリポイント）でも、拒否されることなく宣言された依存関係が準備されます。`--no-deps`とは同時に指定できません。
- **`python_runtime: external`**は、サイドカーの依存関係を共有の和集合から除外します。HermesはそのPythonランタイムをインストールせず、その宣言も変更しません。
- **読み込むものがないのはエラー** — `hermes plugins validate`は、隣に`__init__.py`、`desktop/plugin.js`、`plugin.json`のいずれもない`plugin.yaml`を拒否します。pipレイアウトのパッケージには、ディレクトリプラグインのラッパーが必要です。
- `security.allow_lazy_installs: false`は、オンデマンドでの取得をブロックします。明示的な依存関係への同意と明示的な有効化がPMによる準備を許可します。検出によってインストールが行われることはありません。

`HERMES_HOME/plugins/`は`hermes update`とデスクトップの更新を経ても残ります。アップデーターが再構築するのはvenvとチェックアウトだけで、ホームディレクトリには手を付けません。

### 依存関係のセキュリティポリシー {#dependency-security-policy}

Hermesは**自身の**依存関係を隔離しています。チェックアウトの`[tool.uv] exclude-newer = "14 days"`により、Hermes自身が依存するパッケージの公開されたばかりのリリースは、2週間のあいだ`hermes update`や組み込みの遅延インストールに取り込まれません。そのため、乗っ取られたアップロードはユーザーに届く前に上流で発見されます。**この隔離はあなたのプラグインの依存関係には適用されません。** Hermesがあなたのプラグインを自身の環境に解決する際、このカットオフはHermes自身がロックしているパッケージにのみ適用され、それ以外には適用されません。そのため、プラグインは昨日公開されたリリースを下限にして今日インストールすることができ、それが何を取り込むかについては、Hermesではなくプラグインの作者が責任を負います。（Hermes自身が依存するパッケージのより新しいバージョンを必要とするプラグインは、そのパッケージの期間が明けるのを引き続き待つことになります。）

独自のポリシーを定め、それを守ってください。強く推奨される事項:

- **すべての依存関係に上限を付ける** — 安定版のパッケージには`>=floor,<next_major`、1.0未満のパッケージには`>=0.29,<0.32`。素の`>=X.Y`は、将来のすべてのリリースをレビューなしで採用してしまいます。
- **下限は今週のリリースではなく、API互換性のある最も古いバージョンにする。** 新しいwheelを下限にすると、それが公開された日にすべてのインストーラーがそれを使うことを強制されます。`>=old,!=broken,<next`なら、広い範囲を保ちつつ問題のある1つのリリースだけをスキップできます。
- **独自の新規リリース隔離を採用する** — 下限を新しいリリースに上げるまで約14日待ち、自身のCIでは`uv --exclude-newer "14 days"`（または`UV_EXCLUDE_NEWER`）で解決してください。そうすれば、テストしたロックがそのままユーザーに届きます。
- **ロックをピン留めし、更新をレビューする。** 依存関係の更新はコードの変更として扱ってください。上流の差分を読んでから、ピンを付け直します。

プラグインカタログのレビューでは、ピン留めされたSHAの時点の依存関係リスト（`plugin.yaml`または`pyproject.toml`）を読み、素の下限や上限の欠落を指摘します。下限が単に最近のものであるという理由でエントリが保留されることはありません。

## ステップ3: ツールスキーマを書く {#step-3-write-the-tool-schemas}

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

## ステップ4: ツールハンドラを書く {#step-4-write-the-tool-handlers}

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
    """Evaluate a math expression safely.

    Rules for handlers:
    1. Receive args (dict) — the parameters the LLM passed
    2. Do the work
    3. Return a JSON string — ALWAYS, even on error
    4. Accept **kwargs for forward compatibility
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
    """Convert between units."""
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
4. **`**kwargs`を受け取る:** Hermesはコンテキストキーワード（`task_id`、`session_id`、`user_task`、`parent_agent`など）を注入しますが、シグネチャで名前が指定されたものだけを転送します。そのため`def handler(args)`でも動作します。`**kwargs`は、追加的に拡張されていくコンテキスト全体を受け取るための手段です。

## ステップ5: 登録処理を書く {#step-5-write-the-registration}

`__init__.py`を作成します。これはスキーマとハンドラを結びつけます。

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# フック経由でツール使用状況を追跡する
_call_log = []

def _on_post_tool_call(tool_name, args, result, task_id, **kwargs):
    """Hook: runs after every tool call (not just ours)."""
    _call_log.append({"tool": tool_name, "session": task_id})
    if len(_call_log) > 100:
        _call_log.pop(0)
    logger.debug("Tool called: %s (session %s)", tool_name, task_id)


def register(ctx):
    """Wire schemas to handlers and register hooks."""
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
- `ctx.get_config()` / `ctx.set_config()`は、このプラグイン自身の設定名前空間にのみアクセスします。`ctx.state`は、プラグインが所有するランタイムデータをアクティブなプロファイル配下に保存します。
- この関数がクラッシュした場合、プラグインは無効化されますが、Hermesは問題なく動作を続けます

**`dispatch_tool`の例 — ツールを実行するスラッシュコマンド:**

```python
def handle_scan(ctx, raw_args: str):
    """Implement /scan by invoking the terminal tool through the registry."""
    result = ctx.dispatch_tool("terminal", {"command": f"find . -name '{raw_args}'"})
    return result  # 呼び出し元のチャットUIに返される

def register(ctx):
    # ハンドラは raw_args 文字列を1つだけ受け取る。ctx は lambda でクロージャに閉じ込める。
    ctx.register_command(
        "scan",
        lambda raw: handle_scan(ctx, raw),
        description="Find files matching a glob",
    )
```

ディスパッチされたツールは、通常の承認、リダクション、バジェットのパイプラインを通過します。これらを迂回するショートカットではなく、本物のツール呼び出しです。

### 設定とランタイム状態を保存する {#store-settings-and-runtime-state}

ユーザーに見える振る舞いには、プラグイン相対の設定キーを使ってください。Hermesはそれらを`plugins.entries.<plugin-id>.settings`配下で解決し、グローバルなパス、プラグインをまたぐパス、ディレクトリトラバーサルのパスを拒否します。

```python
def register(ctx):
    endpoint = ctx.get_config("endpoint", default="https://example.invalid")
    retries = ctx.get_config("retry.attempts", default=3)

    ctx.set_config("endpoint", endpoint)
    ctx.set_config("retry.attempts", retries)
```

プラグインが所有するカーソル、キャッシュ、重複排除データには、ランタイムの管理情報を`config.yaml`に置くのではなく、`ctx.state`を使ってください。

```python
def register(ctx):
    cursor = ctx.state.get("cursor", default={"page": 0})
    ctx.state.set("cursor", {"page": cursor["page"] + 1})
```

状態はプロファイル単位で、アトミックに置き換えられ、同時に書き込む複数のライターに対しても安全であり、プラグインごとに10 MiBまでに制限されています。ポータブルパッケージは、自身の`PLUGIN_DATA`と同じディレクトリを共有します。ネイティブプラグインには、衝突しにくくWindowsでも安全な名前空間が与えられます。既存の状態が不正な形式だった場合は、報告されたうえで保持されます。

設定と状態は所有者が異なります。設定は`config.yaml`にあるユーザーに見える振る舞いであり、状態は`<HERMES_HOME>/plugin-data/`配下にあるプラグイン所有のランタイムデータです。どちらのAPIも、他のプラグインの名前空間を公開することはありません。

### デスクトップの設定フォーム {#settings-form-in-the-desktop}

マニフェストの`config_schema`で宣言したすべてのキーは、デスクトップアプリの**Settings → Plugins**配下にあるプラグイン専用ページの1行として表示されます（プラグインのCapabilities → Plugins行にある歯車アイコンで開きます）。デスクトップ側のコードは不要です。バックエンドの`plugins.manage list`がスキーマと各キーの現在値を返し、保存は`ctx.set_config()`と同じライターを通じて書き込まれます。そのため、プラグインが読み戻すのは`plugins.entries.<id>.settings.<key>`です。フォームは`type`に基づくテーブル駆動です。

| マニフェストの`type` | フィールド | 追加のキー |
|---|---|---|
| `str`（デフォルト） | テキスト入力 | `choices: [a, b]`（または`enum:`）でドロップダウンになる |
| `int`、`float` | 数値入力 | |
| `bool` | スイッチ | |
| `list`、`dict` | JSONエディタ | |
| `secret` | マスクされた入力 | `env: MY_PLUGIN_TOKEN` — 値が保存される`.env`変数（デフォルトは大文字スネークケースの`<PLUGIN_ID>_<KEY>`） |

どのエントリも、`label`（または`title`。指定しない場合はキーがセンテンスケースで表示され、`maps_api_key` → "Maps API key"となります）、`description`（ラベルの下に表示されるヘルプテキスト）、`default`、`required`（行に**Required**の印を付ける）を受け付けます。

```yaml
config_schema:
  api_url: {type: str, default: "https://api.example.com", label: "API URL", description: "Service endpoint"}
  retries: {type: int, default: 3}
  mode: {type: str, choices: [fast, careful], default: fast}
  api_key: {type: secret, env: MY_PLUGIN_API_KEY, description: "Personal access token"}
```

**シークレットが`config.yaml`に触れることはありません。** `secret`フィールドが持つのは`.env`の変数名と値が設定済みかどうかだけです。デスクトップは、プロバイダーのAPIキーと同じ認証情報ルート（`PUT /api/env`）を通じて値を保存し、プラグインは`requires_env`エントリとまったく同じように`os.environ.get("MY_PLUGIN_API_KEY")`でそれを読み取ります。`plugins.manage settings`アクションは、シークレットキーと、型や`choices`がスキーマと一致しない値を拒否します。

## ステップ6: テストする {#step-6-test-it}

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

### プラグイン検出のデバッグ {#debugging-plugin-discovery}

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

同じログは常に`~/.hermes/logs/agent.log`に書き込まれます。WARNINGレベル（失敗のみ）で、env変数が設定されているときはDEBUGレベル（すべて）でも書き込まれます。そのため、env変数を付けて実行できない場合（例: ゲートウェイの内部から）は、代わりにログファイルをtailしてください。

```bash
hermes logs --level WARNING | grep -i plugin
```

プラグインが表示されない一般的な理由:

- **設定で有効化されていない** — プラグインはオプトインです。`hermes plugins enable <name>`を実行してください（nameは`plugins list`の出力から取得します。ネストレイアウトの場合は`<category>/<plugin>`になることがあります）。
- **ディレクトリレイアウトが間違っている:** ネイティブパッケージは`~/.hermes/plugins/<plugin-name>/plugin.yaml`（フラット）またはカテゴリ1階層を使います。ポータブルパッケージは、同じ場所でルートの`plugin.json`を使います。それより深いものは無視されます。
- **`__init__.py`がない:** ネイティブパッケージには`plugin.yaml`と、`register(ctx)`関数を持つ`__init__.py`の両方が必要です。ポータブルパッケージはPythonをインポートしないため、`__init__.py`は不要です。
- **`kind`が間違っている** — ゲートウェイアダプターはマニフェストに`kind: platform`が必要です。メモリプロバイダーは`kind: exclusive`として自動検出され、`plugins.enabled`ではなく`memory.provider`設定を通じてルーティングされます。

## プラグインの最終的な構成 {#your-plugins-final-structure}

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

## プラグインは他に何ができるのか {#what-else-can-plugins-do}

### データファイルを同梱する {#ship-data-files}

任意のファイルをプラグインディレクトリに配置し、インポート時に読み込めます。

```python
# tools.py または __init__.py 内
from pathlib import Path
from ruamel.yaml import YAML

_PLUGIN_DIR = Path(__file__).parent
_DATA_FILE = _PLUGIN_DIR / "data" / "languages.yaml"

with open(_DATA_FILE) as f:
    _DATA = YAML(typ="safe").load(f)
```

これは*同梱する*ファイルの場合です。*書き込む*状態は別物です — 次のセクションを参照してください。

### 永続的な状態を保存する {#store-durable-state}

ランタイムの状態をプラグインディレクトリに書き込まないでください。そこはインストールツリーであり、`hermes plugins update` / `remove`がgit pullしたり削除したりします — ユーザーのデータもそれと一緒に失われます。正式な保存場所はプラグインごとのデータルートで、これは両方の操作を経ても残り、アクティブなプロファイルに従います。

```python
from plugins.plugin_storage import plugin_data_dir, plugin_db

# <hermes home>/plugin-data/<name>/ — 初回使用時に作成される
state_file = plugin_data_dir("my-plugin") / "state.json"

# または <data dir>/data.db にある SQLite データベース（WAL モード、スレッドフレンドリー）
conn = plugin_db("my-plugin")
conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY)")
```

プラグインごとに1つのディレクトリがあるため、すべてのプラグインのデータを予測可能な1つの場所で確認できます。シークレットはここに置くべきではありません — 認証情報の読み取りは、他のすべての場所と同様に、標準の`.env` / シークレットスコープのパスを通じて行います。

### スキルを同梱する {#bundle-skills}

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

### 言語パックを同梱する {#ship-a-language-pack}

プラグインは、UI言語を追加したり、既存の言語の文言を上書きしたりできます。これは、Python（`agent.i18n.t()`: 承認プロンプト、ゲートウェイの返信、ツールの動詞、ヒント）、`hermes --tui`インターフェース、デスクトップアプリのすべてのサーフェスに一度に適用されます。`provides_locales`を宣言してYAMLを同梱するだけで、**Pythonは不要**です。

```
~/.hermes/plugins/hermes-lang-pl/
├── plugin.yaml
└── locales/
    ├── pl.yaml            # コア（Python）の文字列 — 同梱の locales/en.yaml と同じキーツリー
    ├── pl.tui.yaml        # オプション: TUI の文字列（キーは locales/_keys.tui.json）
    └── pl.desktop.yaml    # オプション: デスクトップの文字列（キーは locales/_keys.desktop.json）
```

```yaml
name: hermes-lang-pl
version: 1.0.0
description: Polish language pack
provides_locales:
  - id: pl              # 小文字の BCP-47 形式の ID: pl, pt-br, zh-hant
    endonym: Polski     # 言語切り替え UI に表示される名前
    rtl: false
```

`provides_locales`が宣言されている場合、ローダーは`register()`の前に`ctx.register_locale_dir(<plugin>/locales)`を呼び出します（`__init__.py`のないマニフェストのみのパックは、マニフェストのみのデスクトッププラグインと同様に読み込まれます）。カタログは**階層的かつ部分的**です: パック → ユーザーオーバーレイ（`<HERMES_HOME>/locales/`） → 同梱 → 英語 → キー。パックには変更するキーだけがあればよく、キーごとに最後に読み込まれたパックが優先されます。コアの値は英語と同じ名前付きの`{placeholders}`を保持します。英語の値が関数になっているTUI/デスクトップのエントリは、位置指定の`{0}`、`{1}`プレースホルダーを持つ文字列として記述します。

コードを持つプラグインは、プログラムから登録できます。ハンドルは`PluginRegistration`なので、プラグインをアンロードするとそのレイヤーは削除されます。また、登録によって`display.language`が変わることはありません。

```python
def register(ctx):
    here = Path(__file__).parent
    ctx.register_locale("pl", here / "locales" / "pl.yaml", endonym="Polski")        # YAML のパス
    ctx.register_locale("pl", {"approval": {"denied": "      ✗ Odrzucono"}})          # マッピング（ネストまたはフラット）
    ctx.register_locale("pl", here / "locales" / "pl.tui.yaml", surface="tui")       # core | tui | desktop
    ctx.register_locale_dir(here / "locales")                                         # すべての <lang>[.surface].yaml
```

`hermes plugins validate`は、宣言された各IDについて、パース可能でテキストのみの`locales/<id>.yaml`があるかをチェックし（テキスト以外のリーフはエラー）、そのサーフェスの英語カタログに存在しないキーの名前を**警告**します。レンダラーは、`i18n.languages` / `i18n.catalog` RPCを通じてパックのレイヤーを取得します。ユーザー向けガイド: [Language Packs](../../user-guide/features/language-packs.md)。

### 環境変数でゲートする {#gate-on-environment-variables}

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

### オプションのPython依存関係を遅延インストールする {#lazy-install-optional-python-dependencies}

HermesプロジェクトのエクストラでカバーされているSDKについては、それを必要とする操作の箇所で`pm.ensure_import`を使ってください。読み取り専用の利用可否チェックには`pm.available`を使います。頻繁にポーリングされる`check_fn`から依存関係をインストールしないでください。

この例は、既存の`bedrock`エクストラを要求します。

```python
from pm import InstallError, ensure_import

def my_tool_handler(args, **kwargs):
    try:
        ensure_import("bedrock")
    except InstallError as exc:
        return {"error": str(exc)}

    import boto3
    # ここで SDK を使う。
```

引数は`pyproject.toml`のエクストラ名です。任意のパッケージ指定やプラグイン修飾されたキーではありません。古い`LAZY_DEPS`レジストリと`FeatureUnavailable`例外はもう存在しません。

新しい環境が選択された場合、ヘルパーは再起動が必要であることを報告することがあります。実行中のプロセス内で2つ目の環境からインポートするのではなく、そのエラーを返してください。すでに利用可能な依存関係は、`security.allow_lazy_installs`がfalseであってもインストールの必要はありません。

ディレクトリプラグイン自身のPython依存関係については、その`pyproject.toml`の`[project]`配下に`dependencies`を宣言してください。作者が書いたプロジェクトファイルがない場合、PMは`plugin.yaml`または`plugin.yml`にあるレガシーの`pip_dependencies`リストと`python_dependencies`リストを結合します。PMが以前に生成したプロジェクトファイルは、これらのリストを上書きしません。同意、ワークスペースへの所属、最新性のチェックには同じ宣言が使われます。PMは、プラグインを有効化する前に、コアの要件とともにそれらの依存関係を準備します。生成されたワークスペースが、プラグインディレクトリや同梱のロックファイルを書き換えることはありません。依存関係が衝突した場合は受け入れが拒否され、以前の選択が保持されます。PMが他のプラグインを自動的に無効化することはありません。

pipで手動インストールした依存関係は、永続的なPMの宣言ではありません。後で環境が置き換えられた際に保持されるとは限りません。Pythonランタイムが PMの外側にあるラッパープラグインは、[memory-provider survival contract](../memory-provider-plugin.md#hermes_home-survival-contract-what-wrappers-can-rely-on)を利用できます。ランタイムのレイアウトと遅延インストールのポリシーについては、[Package management](../../reference/package-management.md)を参照してください。



### スレッドセーフな遅延シングルトン {#thread-safe-lazy-singletons}

プラグインは、SDKクライアント、HTTPセッション、コネクションプールといった高コストなオブジェクトを、初回使用時に構築されるモジュールレベルの変数にキャッシュすることがよくあります。

```python
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    _client = ExpensiveClient(...)   # ← TOCTOU 競合
    return _client
```

これは落とし穴です。Hermesは1つのプロセス内で複数のスレッド（委譲されたツール呼び出し、バックグラウンドワーカー、自己改善フォーク）を実行するため、`_client`が設定される前に2つのスレッドが`get_client()`に到達し、**両方**が`is not None`チェックを通過し、**両方**が高コストな構築を実行し、2番目の書き込みが1番目を上書きしてしまう可能性があります — その結果、負けた側が開いたリソース（コネクション、ファイルハンドル、バックグラウンドスレッド）がリークします。

ロックを自前で書かないでください。`plugins/plugin_utils.py`のヘルパーを使います。

```python
from plugins.plugin_utils import lazy_singleton, SingletonSlot

# 引数なしのアクセサー → デコレートする:
@lazy_singleton
def get_client():
    return ExpensiveClient(load_config())   # 正確に1回だけ実行される

client = get_client()    # スレッド間で安全
get_client.reset()       # インスタンスを破棄する（テスト / 後始末）


# 構築用の引数を取るアクセサー → スロットを使う:
_slot: SingletonSlot = SingletonSlot()

def get_client(config=None):
    return _slot.get(lambda: ExpensiveClient(resolve(config)))

def reset_client():
    _slot.reset()
```

どちらもダブルチェックロッキングで並行する初回呼び出しを直列化し、ファクトリを最大1回だけ実行します。ファクトリが例外を発生させた場合は何もキャッシュされず、次の呼び出しで再試行されます。[Honchoメモリプラグイン](https://github.com/plastic-labs/honcho/tree/main/hermes-plugin-honcho)（`client.py`）がリファレンスとなる利用例です。

> 経験則: `global _something`の後に`is None`チェックと構築処理を書くときは、いつでも代わりにこれらのどちらかを使ってください。



### 条件付きのツール利用可否 {#conditional-tool-availability}

オプションのライブラリに依存するツールの場合:

```python
ctx.register_tool(
    name="my_tool",
    schema={...},
    handler=my_handler,
    check_fn=lambda: _has_optional_lib(),  # False = ツールはモデルから隠される
)
```

### 組み込みツールを上書きする {#overriding-a-built-in-tool}

組み込みツールを独自の実装に置き換えるには（例えば、デフォルトのブラウザツールをheaded ChromeのCDPバックエンドに差し替える、`web_search`を社内のカスタムインデックスに置き換えるなど）、`override=True`を渡します。

```python
def register(ctx):
    ctx.register_tool(
        name="browser_navigate",             # 組み込みと同じ名前
        toolset="plugin_my_browser",         # 独自のツールセット名前空間
        schema={...},
        handler=my_custom_navigate,
        override=True,                       # 明示的なオプトイン
    )
```

`override=True`がない場合、レジストリは別のツールセットの既存ツールを覆い隠すような登録をすべて拒否します — これにより意図しない上書きを防ぎます。**組み込み**ツールを上書きするには、さらにオペレーターが`config.yaml`の`plugins.entries.<plugin_id>.allow_tool_override: true`でオプトインする必要があります。このゲートがない場合、`register_tool(override=True)`は`PluginToolOverrideError`を発生させます。上書きはログに記録されるため、`~/.hermes/logs/agent.log`で監査できます。プラグインは組み込みツールの後に読み込まれるため、登録順序は正しく、あなたのハンドラが組み込みのものを置き換えます。

**同梱されていないプラグインにはオペレーターの許可も必要です。** Hermesコアに同梱されていないプラグイン（user、project、pipソース）では、既存の組み込みツールに対する`override=True`に、さらに`config.yaml`でのプラグインごとのオプトインが必要です。

```yaml
plugins:
  entries:
    my-plugin:                    # `hermes plugins list` に表示されるプラグインのレジストリキー
      allow_tool_override: true
```

許可がない場合、`ctx.register_tool(..., override=True)`は`PluginToolOverrideError`を発生させます。`register()`の例外はローダーによってキャッチされるため、プラグインは無効化され、Hermesは動作を続けます。このゲートが存在するのは、`shell_exec`や`write_file`のような特権的な組み込みツールを黙って置き換える有効化済みプラグインが、モデルがそれを通じて行うすべての操作を傍受できてしまうためです。同梱プラグインは対象外です。そこでの上書きはメンテナーの判断だからです。設定を読み込めない場合、ゲートはフェイルクローズします。

通常、このキーを手で編集することはありません。`hermes plugins enable <name>`は、プラグインのマニフェストが`capabilities:`配下でそのケイパビリティを宣言している場合にのみ、それを許可するかどうかを尋ねます（同意画面。デフォルトは「いいえ」）。ケイパビリティを宣言していないプラグインは、許可のプロンプトなしで有効化されます。`--allow-tool-override` / `--no-allow-tool-override`フラグは、どちらの場合でも明示的に許可を設定または取り消します。スクリプトによるインストールや、まだマニフェストブロックを採用していないプラグインを事前に承認する場合に使います。同じ許可は`deregister()`もゲートします。許可がない場合、プラグインは自分が所有していないツールを削除できません（さもなければ上書きチェックを回避する手段になってしまいます）。

### 複数のフックを登録する {#register-multiple-hooks}

```python
def register(ctx):
    ctx.register_hook("pre_tool_call", before_any_tool)
    ctx.register_hook("post_tool_call", after_any_tool)
    ctx.register_hook("pre_llm_call", inject_memory)
    ctx.register_hook("on_session_start", on_new_session)
    ctx.register_hook("on_session_end", on_session_end)
```

### フックリファレンス {#hook-reference}

各フックは、**[Event Hooks reference](../../user-guide/features/hooks.md#plugin-hooks)**で詳しく説明されています（コールバックシグネチャ、パラメータ表、各フックが正確にいつ発火するか、例）。以下はサマリーです。

| フック | 発火するタイミング | コールバックシグネチャ | 戻り値 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](../../user-guide/features/hooks.md#pre_tool_call) | 任意のツールが実行される前 | `tool_name: str, args: dict, task_id: str` | オプションのディレクティブ: `{"action": "block", "message": ...}`で呼び出しを拒否、`{"action": "approve", "message": ...}`で人間による承認ゲートにエスカレート |
| [`post_tool_call`](../../user-guide/features/hooks.md#post_tool_call) | 任意のツールが返った後 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 無視される |
| [`pre_llm_call`](../../user-guide/features/hooks.md#pre_llm_call) | ターンごとに1回、ツール呼び出しループの前 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [コンテキストインジェクション](#pre_llm_call-context-injection) |
| [`post_llm_call`](../../user-guide/features/hooks.md#post_llm_call) | ターンごとに1回、ツール呼び出しループの後（成功したターンのみ） | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str` | 無視される |
| `pre_api_request` | プロバイダーへの各生APIリクエストの前（モデルがツールを呼び出す場合はターンごとに複数回） | `session_id: str, model: str, provider: str, base_url: str, api_mode: str, api_call_count: int, message_count: int, tool_count: int, approx_input_tokens: int, max_tokens: int, request: dict` | 無視される |
| `post_api_request` | プロバイダーへの各生APIリクエストが返った後 | `pre_api_request`のフィールドに加えて`api_duration: float, finish_reason: str, response_model: str \| None, usage: dict, response: dict, assistant_content_chars: int, assistant_tool_call_count: int` | 無視される |
| `api_request_error` | プロバイダーAPI呼び出しが例外を発生させたとき | 相関フィールドに加えて`status_code: int \| None, retry_count: int \| None, max_retries: int \| None, retryable: bool \| None, reason: str \| None, error: dict, request: dict` | 無視される |
| `pre_auxiliary_call` | 補助LLM呼び出し（タイトル付け、圧縮、MoA、ビジョン、承認など）の各プロバイダー試行の前。`pre_api_request`ではない | `aux_task: str`に加えて`pre_api_request`のフィールド（`session_id`/`task_id`/`turn_id`は親ターンのものか空、`api_request_id: str`、`retry_count: int`、`streaming: bool`、`request: dict`） | 無視される |
| `post_auxiliary_call` | その試行が返った後、または例外を発生させた後 | `pre_auxiliary_call`のフィールドに加えて`api_duration: float, finish_reason, response_model, usage: dict \| None, response: dict \| None, error: str \| None, error_type: str \| None` | 無視される |
| [`on_session_start`](../../user-guide/features/hooks.md#on_session_start) | 新しいセッションが作成されたとき（最初のターンのみ） | `session_id: str, model: str, platform: str` | 無視される |
| [`on_session_end`](../../user-guide/features/hooks.md#on_session_end) | すべての`run_conversation`呼び出しの終了時 + CLI終了時 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 無視される |
| [`on_session_finalize`](../../user-guide/features/hooks.md#on_session_finalize) | CLI/ゲートウェイがアクティブなセッションを破棄するとき | `session_id: str \| None, platform: str` | 無視される |
| [`on_session_reset`](../../user-guide/features/hooks.md#on_session_reset) | ゲートウェイが新しいセッションキーに切り替えるとき（`/new`、`/reset`） | `session_id: str, platform: str` | 無視される |
| [`gateway_platform_event`](../../user-guide/features/hooks.md#gateway_platform_event) | 認可されたプラットフォームネイティブのイベントがゲートウェイの境界で正規化されたとき（現在はTelegramのリアクション） | `platform: str, event_type: str, payload: dict` | 無視される |
| `kanban_task_claimed` | kanbanタスクが取得されたとき（ディスパッチャープロセス、ワーカーの起動前） | `task_id: str, board: str \| None, assignee: str \| None, run_id: int \| None, profile_name: str` | 無視される |
| `kanban_task_completed` | kanbanタスクが完了したとき（ワーカープロセス） | `task_id, board, assignee, run_id, profile_name, summary: str \| None` | 無視される |
| `kanban_task_blocked` | kanbanタスクがブロックされたとき（ワーカープロセス） | `task_id, board, assignee, run_id, profile_name, reason: str \| None` | 無視される |

ほとんどのフックは、fire-and-forget型のオブザーバーです — 戻り値は無視されます。例外は、会話にコンテキストをインジェクトできる`pre_llm_call`と、block/approveディレクティブを返せる`pre_tool_call`です。

すべてのコールバックは、前方互換性のため`**kwargs`を受け取るべきです。フックコールバックがクラッシュした場合、ログに記録されてスキップされます。他のフックとエージェントは通常どおり続行します。

kanbanのライフサイクルフックはボードDBの変更がコミットされた**後**に発火するため、コールバックは常に永続化された状態を参照でき、SQLiteの書き込みロックを保持することはありません。kanbanワーカーは別個の`hermes -p <profile> chat -q`サブプロセスとして実行されるため、`kanban_task_claimed`は**ディスパッチャー**プロセスで発火し、`kanban_task_completed` / `kanban_task_blocked`は**ワーカー**プロセスで発火します — すべての遷移を一元的に観測するにはディスパッチャーで、タスクごとのセッション内コンテキストを得るにはワーカーでフックしてください。

**APIリクエストフック**は、プロバイダーへの生リクエストを観測するオブザーバーで、ターン単位の`pre_llm_call` / `post_llm_call`ペアより1段下のレベルにあります。ツールを呼び出す1つのターンは複数のAPIリクエストを行い、これらのフックはそれぞれのリクエストの前後で発火します。これらはオブザーバビリティプラグイン（トレーシング、コスト計算、レイテンシダッシュボード）のために存在します。`request`と`response`のkwargsは、プロバイダーペイロードをサニタイズしてサイズ上限を設けたJSONビューであり（機密キーはリダクトされ、長い文字列は切り詰められ、SDKオブジェクトは正規化されます）、`usage`はプレーンなトークンサマリーのdictです。すべてのペイロードは相関フィールド`turn_id`、`api_request_id`、`task_id`、`session_id`、`api_call_count`を持つため、プラグインはリクエスト、ツール呼び出し、ターンを結びつけることができます。`api_request_error`はプロバイダー呼び出しが例外を発生させたときに発火し、`status_code`、`retry_count` / `max_retries`、`retryable`、`reason`、そして`type`と`message`を持つ`error` dictを追加します。

### `pre_llm_call`のコンテキストインジェクション {#pre_llm_call-context-injection}

これは、戻り値が意味を持つ唯一のフックです。`pre_llm_call`コールバックが`"context"`キーを持つdict（または素の文字列）を返すと、Hermesはそのテキストを**現在のターンのユーザーメッセージ**にインジェクトします。これは、メモリプラグイン、RAG連携、ガードレール、そしてモデルに追加コンテキストを提供する必要があるあらゆるプラグインのための仕組みです。

#### 戻り値の形式 {#return-format}

```python
# context キーを持つ dict
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# 素の文字列（上記の dict 形式と等価）
return "Recalled memories:\n- User prefers dark mode"

# None を返すか、何も返さない → インジェクションなし（オブザーバーのみ）
return None
```

`"context"`キーを持つNoneでない空でない戻り値（または素の空でない文字列）はすべて収集され、現在のターンのユーザーメッセージに追加されます。

#### サイズ超過コンテキストの退避 {#oversized-context-spill}

フックごとのコンテキストは、デフォルトで`10,000`文字に制限されています。上限を超えた部分は`$HERMES_HOME/hook_outputs/<session_id>/<uuid>.txt`に書き出され、先頭/末尾のプレビューと保存先パスに置き換えられます。モデルは本当に必要な場合、`read_file`や`terminal`で完全な内容を読むことができます。これにより、暴走したプラグインが後続のすべてのターンのプロンプトを肥大化させ、プロンプトキャッシュのプレフィックスを台無しにするのを防ぎます。`config.yaml`で調整できます。

```yaml
hooks:
  output_spill:
    enabled: true          # デフォルト: true
    max_chars: 10000       # デフォルト。退避を無効にするには大きな値に設定
    preview_head: 500      # プレビューの先頭に表示する文字数
    preview_tail: 500      # プレビューの末尾に表示する文字数
    # directory: null      # デフォルト: $HERMES_HOME/hook_outputs
```

#### インジェクションの仕組み {#how-injection-works}

インジェクトされたコンテキストは、システムプロンプトではなく**ユーザーメッセージ**に追加されます。これは意図的な設計上の選択です。

- **プロンプトキャッシュの保持** — システムプロンプトはターンをまたいで同一のままです。AnthropicとOpenRouterはシステムプロンプトのプレフィックスをキャッシュするため、それを安定させることでマルチターン会話の入力トークンを75%以上削減できます。プラグインがシステムプロンプトを変更すると、すべてのターンがキャッシュミスになってしまいます。
- **一時的** — インジェクションはAPI呼び出し時のみ発生します。会話履歴内の元のユーザーメッセージが変更されることはなく、セッションデータベースにも何も永続化されません。
- **システムプロンプトはHermesの領域** — そこにはモデル固有のガイダンス、ツール強制ルール、パーソナリティ指示、キャッシュされたスキルコンテンツが含まれます。プラグインは、エージェントのコア指示を変更するのではなく、ユーザーの入力と並べてコンテキストを提供します。

#### 例: メモリリコールプラグイン {#example-memory-recall-plugin}

```python
"""Memory plugin — recalls relevant context from a vector store."""

import httpx

MEMORY_API = "https://your-memory-api.example.com"

def recall_context(session_id, user_message, is_first_turn, **kwargs):
    """Called before each LLM turn. Returns recalled memories."""
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

#### 例: ガードレールプラグイン {#example-guardrails-plugin}

```python
"""Guardrails plugin — enforces content policies."""

POLICY = """You MUST follow these content policies for this session:
- Never generate code that accesses the filesystem outside the working directory
- Always warn before executing destructive operations
- Refuse requests involving personal data extraction"""

def inject_guardrails(**kwargs):
    """Injects policy text into every turn."""
    return {"context": POLICY}

def register(ctx):
    ctx.register_hook("pre_llm_call", inject_guardrails)
```

#### 例: オブザーバーのみのフック（インジェクションなし） {#example-observer-only-hook-no-injection}

```python
"""Analytics plugin — tracks turn metadata without injecting context."""

import logging
logger = logging.getLogger(__name__)

def log_turn(session_id, user_message, model, is_first_turn, **kwargs):
    """Fires before each LLM call. Returns None — no context injected."""
    logger.info("Turn: session=%s model=%s first=%s msg_len=%d",
                session_id, model, is_first_turn, len(user_message or ""))
    # 戻り値なし → インジェクションなし

def register(ctx):
    ctx.register_hook("pre_llm_call", log_turn)
```

#### 複数のプラグインがコンテキストを返す場合 {#multiple-plugins-returning-context}

複数のプラグインが`pre_llm_call`からコンテキストを返す場合、それらの出力は二重改行で結合され、まとめてユーザーメッセージに追加されます。順序はプラグインの検出順（プラグインディレクトリ名のアルファベット順）に従います。

### ミドルウェア: 起きることを変える {#middleware-change-what-happens}

フックはエージェントループを観測します（上記でドキュメント化されたいくつかの操舵の形を除く）。**ミドルウェアは起きることそのものを変えます**。リクエストミドルウェアは、下流の何かがそれを参照する前に実効ペイロードを書き換え、実行ミドルウェアは実際の呼び出しをラップします。同じ`register(ctx)`エントリポイントから登録します。

```python
def cap_find_output(tool_name, args, **kwargs):
    """Rewrite terminal find commands to cap their output."""
    command = args.get("command", "")
    if tool_name == "terminal" and command.startswith("find "):
        return {
            "args": {**args, "command": command + " | head -100"},
            "source": "my-plugin",
            "reason": "cap find output",
        }
    return None  # 呼び出しを変更しない

def register(ctx):
    ctx.register_middleware("tool_request", cap_find_output)
```

種類の正規のリストは、`hermes_cli/middleware.py`の`VALID_MIDDLEWARE`です。

| 種類 | 受け取るもの | 戻り値のコントラクト |
|------|----------|-----------------|
| `tool_request` | `tool_name`、`args`、`original_args`、コンテキストkwargs | `{"args": {...}}`を返すと、フック、ガードレール、承認、実行がそれを参照する前に、実効的なツール引数を置き換えます。呼び出しを変更しない場合は`None`を返します。 |
| `llm_request` | `request`、`original_request`、コンテキストkwargs | `{"request": {...}}`を返すと、Hermesが送信する前に実効的なプロバイダーkwargsを置き換えます。 |
| `tool_execution` | ペイロードと`next_call` | ツールの実行をラップします。下流のチェーンを実行するために`next_call(payload)`を正確に1回呼び出し（またはショートサーキットするために呼び出さず）、結果を返します。 |
| `llm_execution` | ペイロードと`next_call` | 同じ形で、プロバイダー呼び出しをラップします。 |

**実際に重要なルール:**

- リクエストミドルウェアはチェーンされます。各コールバックは、それより前のコールバックによって書き換えられたペイロードを参照しますが、`original_args` / `original_request`には常にミドルウェア適用前のコピーが入っています。ペイロードはコールバック間でコピーされるため、自由に変更してかまいません。
- 返すdictには、`source`、`reason`、`name`の文字列を含めることができます。これらはミドルウェアトレースに記録され、下流のオブザーバーフックは`middleware_trace` kwargとしてそれを受け取ります。
- 実行ミドルウェアの`next_call`は**1回限り**です。2回呼び出すと例外が発生します。プロバイダーやツールを再実行してしまうことになるためです。
- 例外を発生させたミドルウェアコールバックは、ログに記録されてスキップされ、チェーンは続行されます。`next_call`の後に発生した下流の失敗は、そのまま伝播します。ミドルウェアがベースのランタイムパスを壊すことは決してありません。
- ミドルウェアのペイロードは、オブザーバーのテレメトリフィールドと並んで`middleware_schema_version`（`hermes.middleware.v1`）を持ちます。
- 未知の種類は失敗せず警告付きで登録されるため、新しいHermes向けに書かれたプラグインも古いHermesで読み込めます。

### CLIコマンドを登録する {#register-cli-commands}

プラグインは独自の`hermes <plugin>`サブコマンドツリーを追加できます。

```python
def _my_command(args):
    """Handler for hermes my-plugin <subcommand>."""
    sub = getattr(args, "my_command", None)
    if sub == "status":
        print("All good!")
    elif sub == "config":
        print("Current config: ...")
    else:
        print("Usage: hermes my-plugin <status|config>")

def _setup_argparse(subparser):
    """Build the argparse tree for hermes my-plugin."""
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

**メモリプロバイダープラグイン**は、代わりに規約ベースのアプローチを使います。プラグインの`cli.py`ファイルに`register_cli(subparser)`関数を追加してください。メモリプラグインの検出システムが自動的に見つけるため、`ctx.register_cli_command()`の呼び出しは不要です。詳細は[Memory Provider Plugin guide](../memory-provider-plugin.md#adding-cli-commands)を参照してください。

**アクティブプロバイダーによるゲート:** メモリプラグインのCLIコマンドは、そのプロバイダーが設定内でアクティブな`memory.provider`である場合にのみ表示されます。ユーザーがあなたのプロバイダーをセットアップしていない場合、あなたのCLIコマンドがヘルプ出力を散らかすことはありません。

### スラッシュコマンドを登録する {#register-slash-commands}

プラグインはセッション内スラッシュコマンドを登録できます。これは、ユーザーが会話中に入力するコマンドです（`/lcm status`や`/ping`など）。CLIとゲートウェイ（Telegram、Discordなど）の両方で動作します。

```python
def _handle_status(raw_args: str) -> str:
    """Handler for /mystatus — called with everything after the command name."""
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

**シグネチャ:** `ctx.register_command(name: str, handler: Callable, description: str = "", args_hint: str = "")`

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

### スラッシュコマンドからツールをディスパッチする {#dispatch-tools-from-slash-commands}

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
- **ゲートウェイモード:** CLIエージェントが存在しないため、ツールはグレースフルに劣化します — ワークスペースは設定されたターミナルの作業ディレクトリから読み取られ、スピナーは表示されません。
- **明示的な上書き:** 呼び出し側が`parent_agent=`を明示的に渡した場合、それが尊重され、上書きされません。

これは、プラグインコマンドからのツールディスパッチのための公開された安定インターフェースです。プラグインは`ctx._cli_ref.agent`などのプライベートな状態に手を伸ばすべきではありません。

### フックの内部から操作する（プロファイル + ツール） {#act-from-inside-a-hook-profile--tools}

`ctx._cli_ref`が設定されるのは、**対話型CLI**セッションの場合だけです。ゲートウェイ、非対話の`hermes chat -q`実行、**kanbanが起動したワーカーセッション**では`None`です — そのため、`_cli_ref`を経由するプラグインのロジックは、まさにそれらのコンテキストで黙って何もしなくなります。フックが実際に必要とするものは、セッションに依存しない2つの安定したAPIでカバーされます。

- **`ctx.profile_name`** — アクティブなプロファイル名（例: `"default"`、またはkanbanワーカーでは担当者のプロファイル）。`HERMES_HOME`から導出されるため、`_cli_ref`に依存せずどこでも動作します。
- **`ctx.dispatch_tool(name, args)`** — `kanban_*`ツール、`delegate_task`、`terminal`、`read_file`などを含め、登録済みの任意のツール（組み込みまたはプラグイン）を呼び出します。フックがどのプロセスで発火するかに関係なく、フックのコールバックから動作します。

これらを組み合わせると、kanbanのライフサイクルフックが遷移を観測し、フレームワークの内部に触れることなくボードに対して操作できます。

```python
def register(ctx):
    def on_blocked(*, task_id, reason=None, **kw):
        # ワーカープロセスで実行される。ここでは ctx._cli_ref は None。
        ctx.dispatch_tool("kanban_comment", {
            "task_id": task_id,
            "comment": f"[{ctx.profile_name}] auto-noted block: {reason}",
        })
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

完全な`hermes <subcommand>`（例: `hermes kanban show`）を実行するには、`ctx.dispatch_tool("terminal", {"command": "hermes kanban show ..."})`で`terminal`ツールを使ってシェルに処理を委ねてください — ヘッドレスなワーカーセッション向けのプロセス内スラッシュコマンドブリッジはなく、フックからHermesを駆動するためのサポートされた方法はツールです。

### エージェントのブラウザに生のCDPコマンドを送る {#send-raw-cdp-commands-to-the-agents-browser}

ブラウザツールより低いレベルでエージェントのブラウザを操作しなければならないプラグインは、`SUPERVISOR_REGISTRY.capture(task_id)`で、タスクのライブなスーパーバイザー接続に固定されたCDPハンドルを取得し、`call(method, params=None, *, session_id=None, timeout=10.0)`でコマンドを送ることができます。このハンドルは、その接続が失われると`CapturedCDPInvalid`を発生させ、より新しい接続に対象を切り替えることはありません。カタログが拒否する`CDPSupervisor`のプライベート属性の読み書きの代わりに、これを使ってください。これは信頼されたプロセス内のトランスポートにすぎず、オリジン、同意、所有権のチェックは一切行いません。完全なコントラクト: [Browser CDP Supervisor](../browser-supervisor.md#trusted-plugin-cdp-access)。

### どのcron実行の中にいるかを知る {#know-which-cron-run-you-are-in}

`ctx.current_cron_execution()`は、現在のコードが実行されているスケジュール実行を返し、cronの外では`None`を返します。実行中に発火する任意のフック（`pre_tool_call`、`post_tool_call`、`pre_llm_call`など）とツールハンドラから動作します。値は凍結された`CronExecution`です。

| フィールド | 意味 |
|---|---|
| `job_id`、`job_name` | cronジョブ。 |
| `execution_id` | 実行台帳（`hermes cron runs`）におけるこの実行の行。 |
| `source` | どの経路で実行が発火したか: `"builtin"`（組み込みスケジューラー）、`"direct"`（その外で発火した実行。例: `hermes cron run`）、または外部スケジューラーの名前。 |
| `scheduled_instant` | この実行が発火したスケジュール上の時点。手動実行やその他のスケジュール外の実行では`None`になるため、スケジュール実行と手動実行を区別するにはこのフィールドを確認してください。 |
| `started_at` | 実行が開始された時刻。 |
| `profile` | ジョブを所有するプロファイル。 |

スケジューラーは、実行が実行権の獲得に成功した後にのみこれを設定し、実行が終わるとクリアします。実行ごとの値なので、同時に発火する2つのジョブやプロファイルが互いの値を見ることはありません。モデルがこれを偽造することはできません。フックの引数はHermesから来るものであり、ツールのサブプロセス（terminal、`execute_code`）は別のインタープリタで実行されるためです。`delegate_task`で起動されたサブエージェントは、スケジュール実行そのものではないため`None`を受け取ります。

```python
def register(ctx):
    def guard(*, tool_name, args, **kw):
        run = ctx.current_cron_execution()
        if tool_name == "deploy" and (run is None or run.scheduled_instant is None):
            return {"action": "block", "message": "deploy only runs from its scheduled cron job"}
    ctx.register_hook("pre_tool_call", guard)
```

### Automation Blueprintsを追加する {#add-automation-blueprints}

`ctx.register_automation_blueprint(key, *, title, description, schedule_template, prompt_template, category="general", slots=(), deliver_default="origin", skills=(), tags=())`は、空欄を埋めるだけで使える自動化を[Automation Blueprintsカタログ](../../reference/automation-blueprints-catalog.mdx)に追加します。それは`/blueprint`（CLI、TUI、メッセンジャー）、ダッシュボードのBlueprintsタブ、デスクトップのCronページで、組み込みのものと並んで、プラグイン名のラベル付きで表示されます。引数は組み込みの`AutomationBlueprint`（`cron/blueprint_catalog.py`）のフィールドで、各スロットは`BlueprintSlot`のフィールドからなるdictです。

```python
def register(ctx):
    ctx.register_automation_blueprint(
        "standup",
        title="Team standup digest",
        description="Every weekday, summarize what changed in a repo since yesterday.",
        category="work",
        schedule_template="{minute} {hour} * * 1-5",
        prompt_template="Summarize yesterday's commits and open PRs for {repo} as a short standup digest.",
        slots=[
            {"name": "repo", "type": "text", "label": "Which repo?", "default": "acme/app"},
            {"name": "time", "type": "time", "label": "What time?", "default": "09:15"},
            {"name": "deliver", "type": "enum", "label": "Where to deliver?", "default": "origin",
             "options": ["origin", "local"], "strict": False},
        ],
        tags=["work", "daily"],
    )
```

その後、ユーザーは`/blueprint teamtools:standup`（または単に`/blueprint standup`）を実行するか、デスクトップ/ダッシュボードのフォームでそれを選びます。

- **キーは名前空間付き。** カタログのキーは`<plugin>:<key>`（`teamtools:standup`）なので、プラグインが組み込みや他のプラグインのblueprintを置き換えることは決してできません。素のキーを渡してください。`:`を含むキーは拒否されます。
- **スロットとテンプレート。** スロットの`type`は`time`（`HH:MM`）、`enum`、`weekdays`、`text`のいずれかです。`prompt_template`では任意のスロットを`{name}`として使えます。`schedule_template`では、スロット名に加えて`{minute}`/`{hour}`（`time`という名前のスロットから）と`{dow}`（`recurrence`/`day`スロットから。なければ`*`）を使えます。`deliver`という名前のスロットには、デスクトップとダッシュボードのフォームでプロファイルの実際の配信先が表示されます。
- **登録時に検証される。** 未知のプレースホルダー、不正なスロットの型、重複したキー、パースできないスケジュールがあると、警告がログに出てそのblueprintはスキップされますが、プラグインの残りの部分は読み込まれます。すべてのスロットにデフォルトがある場合、Hermesは読み込み時に一度blueprintを埋めてみるため、壊れたテンプレートはユーザーが最初に*Schedule it*を押したときではなく、その時点で失敗します。
- **プロファイルごと。** プラグインはプロファイルごとに読み込まれるため、blueprintはプラグインが有効化されているプロファイルでのみ一覧に表示されます。
- **ジョブはプラグインより長く存続する。** blueprintをスケジュールすると、プロンプトとスケジュールがすでに埋められた通常のcronジョブが作成されます。プラグインを無効化またはアンインストールすると、blueprintはカタログから削除されますが、それらのジョブは実行され続けます。ジョブの`skills`がプラグインの同梱するスキルを指定している場合、プラグインが無効な間、ジョブはそのスキルをスキップします（そしてそれをログに記録します）。

これにはマニフェストのケイパビリティは不要です。スロットはプレーンなdictなので、プラグインホストの分離下でも動作します。

### SlackのBlock Kitボタンのクリックを処理する {#handle-slack-block-kit-button-clicks}

インタラクティブな要素（ボタン、オーバーフローメニュー、日付ピッカーなど）を持つBlock Kitメッセージを投稿するプラグインは、クリックハンドラをSlackアダプターに直接登録できます — `slack_bolt.AsyncApp`へのモンキーパッチは不要です。

```python
def register(ctx):
    async def _on_approve(ack, body, action):
        # 3秒以内に ack する — slack_bolt の要件。
        await ack()
        # body["channel"]["id"], body["user"]["id"], body["message"]["ts"]
        # action["action_id"], action["value"]
        sweep_id = (action.get("value") or "").split("|", 1)[-1]
        # ...決定論的な処理を行い、フォローアップを投稿する。

    ctx.register_slack_action_handler("inbox_sweep_approve", _on_approve)
```

**シグネチャ:** `ctx.register_slack_action_handler(action_id, callback) -> None`

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `action_id` | `str \| re.Pattern \| dict` | `slack_bolt.App.action()`が受け付けるもの: リテラルの`action_id`、複数のIDにマッチするコンパイル済み正規表現、または`{"action_id": "...", "block_id": "..."}`のような制約dict |
| `callback` | 非同期callable | slack_boltの規約に従って`(ack, body, action)`を受け取る |

**実行時の動作:**

- ハンドラはプラグインの読み込み時にキューに入れられ、Slackプラットフォームが接続したときにアダプターの`slack_bolt.AsyncApp`に結びつけられます。
- 各コールバックは防御的にラップされます。ハンドラが例外を発生させた場合、ゲートウェイはエラーをログに記録し、Slackがリトライを止めるようにベストエフォートでクリックをackします。
- 標準のslack_boltのルールが適用されます — 3秒以内に`await ack()`し、その後で時間のかかる処理を行ってください。
- マルチワークスペースのデプロイでは、ハンドラは接続されている任意のワークスペースからのクリックで発火します。振る舞いのスコープを限定する必要がある場合は`body["team"]["id"]`を使ってください。

これは、プラグインがSlackのインタラクティビティに参加するための公開された方法です。古いプラグインは`SlackAdapter.connect`にパッチを当てているかもしれませんが、代わりにこのAPIを使ってください。slack_boltの全サーフェス（Block Kitのアクションだけでなく、イベント、ショートカット、コマンド）については、後述の汎用的な`register_platform_handler("slack", ...)`を使ってください。

### ネイティブのプラットフォームハンドラを登録する（任意のプラットフォーム） {#register-native-platform-handlers-any-platform}

コアのアダプターがルーティングしないプラットフォームイベント — 追加の更新タイプ、ネイティブのボタンコールバック、リアクション/メンバーイベント、Webhookルート — を受け取る必要があるプラグインは、プラットフォームのアダプターが接続時に呼び出すハンドラファクトリを登録できます。これは**すべての**ゲートウェイプラットフォームで動作します。

```python
def register(ctx):
    def _wire(native, adapter):
        # native: プラットフォームのクライアント/アプリオブジェクト（下の表を参照）
        # adapter: プラットフォームアダプターのインスタンス（読み取り専用として扱う）
        # register() が SDK なしでも動作するよう、プラットフォーム SDK はここでインポートする。
        ...

    ctx.register_platform_handler("discord", _wire)
```

**シグネチャ:** `ctx.register_platform_handler(platform, factory) -> None`

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `platform` | `str` | 小文字のゲートウェイプラットフォーム名（`"telegram"`、`"discord"`、`"slack"`、`"matrix"`など） |
| `factory` | callable | 接続時に`(native, adapter)`を受け取る |

**プラットフォームごとの`native`の内容:**

| プラットフォーム | `native`オブジェクト | 典型的なフック |
|----------|-----------------|---------------|
| `telegram` | PTBの`Application` | `add_handler` — 任意の更新タイプ、パターンでスコープされたコールバック |
| `discord` | `discord.ext.commands.Bot` | `add_listener` — リアクション、メンバーイベント、スレッド、ボイス |
| `slack` | `slack_bolt.AsyncApp` | `app.event()` / `app.action()` / `app.command()` |
| `matrix` | Matrixクライアント | イベントコールバック |
| `teams` | Teamsの`App` | `on_message` / `on_card_action`デコレーター |
| `dingtalk` | `DingTalkStreamClient` | 他のストリームトピック用の`register_callback_handler` |
| `feishu` | lark_oapiクライアント | API呼び出し、イベントルーティング |
| `line`、`api_server`、`msgraph_webhook` | aiohttpの`web.Application` | `router.add_get/post` — カスタムルート（ルーターが凍結される前に結びつけられる） |
| その他すべて（whatsapp、signal、irc、email、sms、ntfy、wecom、weixin、bluebubbles、yuanbaoなど） | `None` | 接続時フック。`adapter`ハンドルを通じて処理する |

**実行時の動作:**

- ファクトリはプラグインの読み込み時にキューに入れられ、プラットフォームが接続したときに呼び出されます — ディスパッチ順序が重要なプラットフォーム（Telegram、Slack、Teams、aiohttpルーター）では、コアのハンドラが登録される**前に**実行されるため、スコープされたプラグインのハンドラが優先され、それ以外はすべて通過します。
- **先にマッチしたものが処理されるディスパッチテーブルに追加するハンドラは、必ずスコープを限定してください。** Telegramでは`CallbackQueryHandler(..., pattern=r"^myplugin:")`を使ってください — スコープのないハンドラは、コアのボタンフロー（実行の承認、モデルピッカー、確認プロンプト）を飲み込んでしまいます。
- 各ファクトリは分離されています。例外を発生させた場合、エラーはログに記録され、プラットフォームは接続を続けます。
- プラットフォームSDKはモジュールレベルではなく、ファクトリの本体内でインポートしてください — `register()`はSDKがインストールされていない場合でも動作しなければなりません。
- 1つのプラグインが複数のプラットフォーム向けにファクトリを登録できます。それぞれは自分のプラットフォームが接続したときにのみ発火します。

**Telegramのエイリアス:** `ctx.register_telegram_handler(factory)`は、`ctx.register_platform_handler("telegram", factory)`の後方互換エイリアスです。

例 — Telegram、パターンでスコープされたインラインボタン:

```python
def register(ctx):
    def _wire(application, adapter):
        from telegram.ext import CallbackQueryHandler

        async def _on_button(update, context):
            query = update.callback_query
            await query.answer()
            # ..."myplugin:*" コールバックを処理する

        application.add_handler(
            CallbackQueryHandler(_on_button, pattern=r"^myplugin:")
        )

    ctx.register_platform_handler("telegram", _wire)
```

例 — Discord、リアクションイベント:

```python
def register(ctx):
    def _wire(bot, adapter):
        async def on_raw_reaction_add(payload):
            ...  # 例: リアクションによる投票 / モデレーション

        bot.add_listener(on_raw_reaction_add, "on_raw_reaction_add")

    ctx.register_platform_handler("discord", _wire)
```

### 実行中のプラグイン読み込み: すぐに有効になるものと次のセッションからのもの {#mid-run-plugin-loading-what-activates-now-vs-next-session}

プラグインは、ゲートウェイ（またはTUI/デスクトップサーバー）がすでに実行中のときにも読み込まれることがあります: `hermes plugins
install`/`enable`、デスクトップやダッシュボードからのインストール、カタログの再ピン留め、ツールによって引き起こされる強制的な再検出。これらの経路はすべて**実際の強制再スキャン**（`discover_plugins(force=True)`）を実行し、その内部から`PluginManager.on_plugin_loaded(callback)`が、**新たに**読み込まれたプラグインごとに1つのサマリーとともに発火します（`hermes_cli/plugins_activation.py`）。

```python
{"name": "late-mcp", "key": "late-mcp",
 "activated_now": {"gateway_commands": ["late"], "callbacks": ["telegram"]},
 "deferred": {"tools": ["late_tool"], "prompt": ["late.section"], "mcp_servers": ["worker"]}}
```

- **即座に有効** — ゲートウェイのスラッシュコマンド、ゲートウェイの変換フック / その他のフック、プラットフォームのコールバック。ゲートウェイランナーは起動時に購読し、すべてのライブなアダプターの冪等な`rewire_plugin_handlers()`を呼び出すため、後から読み込まれたプラグインが登録した`register_platform_handler`ファクトリ（またはSlackのアクションハンドラ）も再起動なしで結びつけられます。再結合はネイティブクライアントごとに`(plugin, factory
  qualname)`で重複排除されます。Telegramでは、後から読み込まれたハンドラは、コアのキャッチオールの`filters.COMMAND` / `CallbackQueryHandler`より前に引き上げられます（PTBはグループごとに最初にマッチしたものをディスパッチします）。これは接続時に配置される場合とまったく同じ位置です。
- **遅延** — `tools`と`prompt`のセクションは**次のセッション**から適用されます（実行中のセッションのプロンプト/ツールスキーマはキャッシュ安定性のため固定されており、`/skills install`と同じルールです）。`mcp_servers`（プラグインの`mcp.json`のサーバー。mcp.jsonでの名前のまま）は、`mcp.reload`時または次のセッションで接続されます。
- 結合の解除はありません: 実行中にプラグインを無効化しても、すでに結びつけられたハンドラはゲートウェイが再起動するまで残り、各サーフェスはそのことを表示します。

インストール系のサーフェスは、まさにこの区分を報告します: `hermes plugins install/enable`は実行中のゲートウェイに通知（`reload-plugins`コントロールソケットの動詞）した後にそれを表示し、`plugins.manage install/toggle/update`は`activation` + `gateway_reloaded`を返します（`restart_required`は、どのゲートウェイも応答しなかった場合にのみtrueになります）。

:::tip
このガイドは**汎用プラグイン**（ツール、フック、スラッシュコマンド、CLIコマンド）を扱います。以下のセクションでは、各専門プラグインタイプの作成パターンを概説します。それぞれがフィールドリファレンスと例について、完全なガイドへのリンクを示しています。
:::

## 専門プラグインタイプ {#specialized-plugin-types}

Hermesには、汎用サーフェスを超えた5つの専門プラグインタイプがあります。それぞれは、`plugins/<category>/<name>/`（bundled）または`~/.hermes/plugins/<category>/<name>/`（user）配下のディレクトリとして提供されます。コントラクトはカテゴリによって異なります。必要なものを選んでから、その完全なガイドを読んでください。

### モデルプロバイダープラグイン — LLMバックエンドを追加する {#model-provider-plugins--add-an-llm-backend}

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

**完全なガイド:** [Model Provider Plugins](../model-provider-plugin.md) — フィールドリファレンス、オーバーライド可能なフック（`prepare_messages`、`build_extra_body`、`build_api_kwargs_extras`、`fetch_models`）、api_modeの選択、認証タイプ、テスト。

### プラットフォームプラグイン — ゲートウェイチャネルを追加する {#platform-plugins--add-a-gateway-channel}

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
        # env 変数から PlatformConfig.extra を自動入力し、env のみのセットアップが
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

**完全なガイド:** [Adding Platform Adapters](../adding-platform-adapters.md) — 完全な`BasePlatformAdapter`コントラクト、メッセージルーティング、認証ゲート、セットアップウィザード連携。stdlibのみで動作する例としては`plugins/platforms/irc/`を見てください。

### メモリプロバイダープラグイン — セッションをまたいだ知識バックエンドを追加する {#memory-provider-plugins--add-a-cross-session-knowledge-backend}

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

    def sync_turn(self, user_content, assistant_content, *,
                  session_id="", messages=None) -> None:
        ...

    def prefetch(self, query, *, session_id="") -> str:
        ...

    def get_tool_schemas(self) -> list[dict]:
        return []   # 必須の @abstractmethod — 完全なガイドを参照

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

メモリプロバイダーは単一選択です — 一度にアクティブになれるのは1つだけで、`config.yaml`の`memory.provider`で選択します。

プロバイダーが汎用プラグインとしても読み込まれる場合、そのライフサイクルフックは汎用の検出処理が所有します。メモリローダーがフックを提供するのは、同じプラグインソースが汎用の検出処理を通じて正常に読み込まれるまでのフォールバックとしてのみです。プロバイダーが繰り返し読み込まれると、フォールバックのフックグループが置き換えられます。グループ内の異なるコールバックは保持されます。これは、異なるプラグインソースからのフックの重複を排除したり、プロバイダーの有効化を変更したりするものではありません。

**完全なガイド:** [Memory Provider Plugins](../memory-provider-plugin.md) — 完全な`MemoryProvider` ABC、スレッディングコントラクト、プロファイル分離、`cli.py`経由のCLIコマンド登録。

### コンテキストエンジンプラグイン — コンテキスト圧縮器を置き換える {#context-engine-plugins--replace-the-context-compressor}

```python
# plugins/context_engine/my-engine/__init__.py
from agent.context_engine import ContextEngine

class MyContextEngine(ContextEngine):
    @property
    def name(self) -> str:
        return "my-engine"

    def update_from_response(self, usage) -> None: ...
    def should_compress(self, prompt_tokens: int = None) -> bool: ...
    def compress(self, messages, current_tokens=None, focus_topic=None,
                 force=False, memory_context="") -> list: ...

def register(ctx):
    ctx.register_context_engine(MyContextEngine())
```

コンテキストエンジンは単一選択です — `config.yaml`の`context.engine`で選択します。

**完全なガイド:** [Context Engine Plugins](../context-engine-plugin.md)。

### 画像生成バックエンド {#image-generation-backends}

プロバイダーを`plugins/image_gen/<name>/`に配置します。

```python
# plugins/image_gen/my-imggen/__init__.py
from agent.image_gen_provider import ImageGenProvider

class MyImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "my-imggen"

    def is_available(self) -> bool: ...
    def generate(self, prompt: str, aspect_ratio="landscape", **kwargs) -> dict:
        # success_response(...) / error_response(...) を返す
        ...

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

**完全なガイド:** [Image Generation Provider Plugins](../image-gen-provider-plugin.md) — 完全な`ImageGenProvider` ABC、`list_models()` / `get_setup_schema()`メタデータ、`success_response()`/`error_response()`ヘルパー、base64 vs URL出力、ユーザーによる上書き、pip配布。

**リファレンス例:** `plugins/image_gen/openai/`（OpenAI SDK経由のDALL-E / GPT-Image）、`plugins/image_gen/openai-codex/`、`plugins/image_gen/xai/`（Grok画像生成）。

### Computer-useバックエンドプラグイン {#computer-use-backend-plugins}

`computer_use`ツールは、`ComputerUseBackend` ABC（`tools/computer_use/backend.py`）を通じて、ちょうど1つの**ドライバー**とやり取りします。それを選ぶのは`config.yaml`の`computer_use.backend`です。デフォルトは組み込みのcua-driverで、これは`plugins/computer_use/cua/`にある通常のプロバイダーとして同梱されています。別のドライバーは、`~/.hermes/plugins/<name>/`にあり、その`register(ctx)`が`ctx.register_computer_use_provider()`を呼び出すプロバイダープラグインです。ディレクトリ名が、`computer_use.backend`に設定する値になります。computer-useプロバイダーは、メモリプロバイダーやコンテキストエンジンと同様に単一選択です。

```python
# ~/.hermes/plugins/my-driver/__init__.py
from tools.computer_use.backend import (
    ActionResult, CaptureResult, ComputerUseBackend, ComputerUseProvider,
)

class MyBackend(ComputerUseBackend):
    def start(self): ...                    # ドライバーセッションを開く
    def stop(self): ...                     # セッションを破棄する（終了時にも呼ばれる）
    def is_available(self): return True
    def capture(self, mode="som", app=None, pid=None, window_id=None):
        return CaptureResult(mode=mode, width=1920, height=1080, png_b64=...)
    def click(self, **kw): return ActionResult(ok=True, action="click")
    def drag(self, **kw): return ActionResult(ok=True, action="drag")
    def scroll(self, **kw): return ActionResult(ok=True, action="scroll")
    def type_text(self, text, **kw): return ActionResult(ok=True, action="type")
    def key(self, keys, **kw): return ActionResult(ok=True, action="key")
    def list_apps(self): return []
    def focus_app(self, app, raise_window=False): return ActionResult(ok=True, action="focus_app")
    def set_value(self, value, element=None):
        # アクションを実行できないドライバーは呼び出しごとにそう伝える。ツールスキーマは決して変わらない。
        return ActionResult(ok=False, action="set_value", code="unsupported_action",
                            message="my-driver has no accessibility value setter")

class MyDriverProvider(ComputerUseProvider):
    name = "my-driver"
    display_name = "My driver"

    def create_backend(self, *, permission_mode):
        return MyBackend()                  # permission_mode: standard | bounded | unrestricted

    def is_available(self):                 # ツールをゲートする。軽量で、ネットワークアクセスなし
        return True

    def doctor(self):                       # オプション: `hermes computer-use doctor`
        print("my-driver: ok")
        return 0

def register(ctx):
    ctx.register_computer_use_provider(MyDriverProvider())
```

```yaml
# ~/.hermes/plugins/my-driver/plugin.yaml
name: my-driver
version: 1.0.0
description: Alternative computer-use driver   # ピッカーに表示される
```

`hermes tools` → Computer Useで選択するか（インストール済みのプロバイダーは、CLIとデスクトップのツールセットパネルでcua-driverの下に一覧表示されます）、直接設定します。`plugins.enabled`のエントリは不要です。

```yaml
# config.yaml
computer_use:
  backend: my-driver     # デフォルト: cua
```

ルール:

- **一度にアクティブなプロバイダーは1つだけ。** インポートされインスタンス化されるのは、選択されたものだけです。選択されていないインストール済みのプロバイダーはディスク上に残り、（`plugin.yaml`から）選択肢として一覧表示されるだけです。プロバイダーが自ら有効化されることはありません。
- **選択はプロファイルごとで、セッションがドライバーを起動するときに読み取られる。** 設定された名前が解決できない場合、その名前とインストール済みのプロバイダーを示すエラーで呼び出しが失敗します。Hermesが別のドライバーにフォールバックすることは決してありません。
- **モデルに見えるスキーマはすべてのプロバイダーで同じ**なので、ドライバーを差し替えてもプロンプトキャッシュは維持されます。対応できないものは、アクションごとに`ActionResult(ok=False, code="unsupported_action", ...)`で報告してください。
- 承認ゲート、Bot Desktopのリース、スクリーンショットの重複排除、要素数の上限、ビジョンへのルーティングはすべて、バックエンドより上のツール内で実行されます。バックエンドは画面を操作するだけです。
- `hermes computer-use status`/`doctor`/`permissions`がcua-driverのチェックを実行するのは、`cua`が選択されている場合だけです。それ以外のプロバイダーでは、その`is_available()`を報告し、`doctor()`があればそれを実行します。
- プロバイダーはライブなドライバーセッションを返すため、プロセス内でのみ読み込まれます。`plugins.isolation: host`の下では、ユーザーのプロバイダーは拒否され、呼び出しはそれを読み込めなかったことを報告します。

## Python以外の拡張サーフェス {#non-python-extension-surfaces}

Hermesは、Pythonプラグインではない拡張も受け付けます。これらは[Pluggable interfaces table](../../user-guide/features/plugins.md#pluggable-interfaces--where-to-go-for-each)に示されています。以下のセクションでは、各作成スタイルを簡単に概説します。

### MCPサーバー — 外部ツールを登録する {#mcp-servers--register-external-tools}

Model Context Protocol（MCP）サーバーは、Pythonプラグインなしで独自のツールをHermesに登録します。`~/.hermes/config.yaml`で宣言します。

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

Hermesは起動時に各サーバーに接続し、そのツールを一覧表示し、組み込みツールと並べて登録します。LLMは、それらを他の任意のツールとまったく同じように認識します。**完全なガイド:** [MCP](../../user-guide/features/mcp.md)。

### ゲートウェイイベントフック — ライフサイクルイベントで発火する {#gateway-event-hooks--fire-on-lifecycle-events}

マニフェスト + ハンドラを`~/.hermes/hooks/<name>/`に配置します。プラグインとは異なり`plugins.enabled`の手順はありません。ゲートウェイは起動時に有効なフックディレクトリをすべてインポートするため、ファイルを配置すること**自体が**オプトインになります（[信頼モデル](../../user-guide/features/hooks.md#gateway-hook-trust)）。

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

**完全なガイド:** [Gateway Event Hooks](../../user-guide/features/hooks.md#gateway-event-hooks)。

### シェルフック — ツール呼び出し時にシェルコマンドを実行する {#shell-hooks--run-a-shell-command-on-tool-calls}

ツールが発火したときにスクリプトを実行したいだけの場合（通知、監査ログ、デスクトップアラート、自動フォーマッタ）は、`config.yaml`のシェルフックを使ってください — Pythonは不要です。

```yaml
hooks:
  - event: post_tool_call
    command: "notify-send 'Tool ran: {tool_name}'"
    when:
      tools: [terminal, patch, write_file]
```

Pythonプラグインフックと同じすべてのイベント（`pre_tool_call`、`post_tool_call`、`pre_llm_call`、`post_llm_call`、`on_session_start`、`on_session_end`、`pre_gateway_dispatch`）に加えて、`pre_tool_call`のブロッキング判定のための構造化JSON出力をサポートします。

**完全なガイド:** [Shell Hooks](../../user-guide/features/hooks.md#shell-hooks)。

### スキルソース — カスタムスキルレジストリを追加する {#skill-sources--add-a-custom-skill-registry}

スキルのGitHubリポジトリを保守している場合（または組み込みソースを超えてコミュニティインデックスから取得したい場合）は、それを**tap**として追加します。

```bash
hermes skills tap add myorg/skills-repo
hermes skills search my-workflow --source myorg/skills-repo
hermes skills install myorg/skills-repo/my-workflow
```

独自のtapを公開するのは、`skills/<skill-name>/SKILL.md`ディレクトリを持つGitHubリポジトリを用意するだけです — サーバーやレジストリへのサインアップは不要です。

**完全なガイド:** [Skills Hub](../../user-guide/features/skills.md#skills-hub) · [カスタムtapの公開](../../user-guide/features/skills.md#publishing-a-custom-skill-tap)（リポジトリレイアウト、最小限の例、非デフォルトパス、信頼レベル）。

### コマンドテンプレート経由のTTS / STT {#tts--stt-via-command-templates}

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

STTについては、`HERMES_LOCAL_STT_COMMAND`をargvトークン化されたテンプレートに向けてください。これは暗黙のシェル解釈なしで実行されます。信頼できるローカルコマンドがシェル構文を必要とする場合は、明示的に`sh -c`、`cmd /c`、またはPowerShellでラップしてください。サポートされるプレースホルダー: `{input_path}`、`{output_path}`、`{format}`、`{voice}`、`{model}`、`{speed}`（TTS）、`{input_path}`、`{output_dir}`、`{language}`、`{model}`（STT）。パスを扱う任意のCLIが、自動的にプラグインになります。

**完全なガイド:** [TTS custom command providers](../../user-guide/features/tts.md#custom-command-providers) · [STT](../../user-guide/features/tts.md#voice-message-transcription-stt)。

## pip経由で配布する {#distribute-via-pip}

プラグインを公開して共有するには、Pythonパッケージにエントリポイントを追加します。

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-plugin = "my_plugin_package"
```

インストールの所有者が提供する環境（例えばNixのderivation）にディストリビューションが存在する場合、エントリポイントによる検出は引き続きサポートされます。これは検出であって、PMが選択した世代にパッケージを注入する許可ではありません。管理されたインストールでは、`pyproject.toml`またはマニフェストのPython要件を持つディレクトリプラグインとして配布し、PMがトランザクショナルに受け入れられるよう`hermes plugins install` / `enable`を使ってください。新しい環境が選択された後はHermesを再起動してください。`hermes pm install`が受け付けるのは管理対象のツール名であり、任意のPyPIパッケージではありません。

## NixOS向けに配布する {#distribute-for-nixos}

:::warning Nixは明示的なサポート対象ではなくなりました
Nix/NixOSは、明示的にサポートされるインストール方法ではなくなりました（ベストエフォートのみ） — [Nix Setup](../../getting-started/nix-setup.md)を参照してください。このセクションは、すでにNixOS上でデプロイしているユーザーのために残されています。
:::

NixOSユーザーは、エントリポイントを持つ`pyproject.toml`を提供すれば、あなたのプラグインを宣言的にインストールできます。

**エントリポイントプラグイン**（配布に推奨）:
```nix
# ユーザーの configuration.nix
services.hermes-agent.extraPythonPackages = [
  (config.services.hermes-agent.package.python.pkgs.buildPythonPackage {
    pname = "my-plugin";
    version = "1.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "you";
      repo = "hermes-my-plugin";
      rev = "v1.0.0";
      hash = "sha256-...";  # nix-prefetch-url --unpack
    };
    format = "pyproject";
    build-system = [ config.services.hermes-agent.package.python.pkgs.setuptools ];
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

オーバーレイの使い方や衝突チェックを含む完全なドキュメントについては、[Nix Setup guide](../../getting-started/nix-setup.md#plugins)を参照してください。

## よくある間違い {#common-mistakes}

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
# 動作する — ディスパッチャーはシグネチャで名前が指定されたコンテキストキーワードだけを転送する
def handler(args):
    ...

# より良い — 注入されるすべてのコンテキストフィールド（task_id, session_id, parent_agent, ...）を受け取る
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
