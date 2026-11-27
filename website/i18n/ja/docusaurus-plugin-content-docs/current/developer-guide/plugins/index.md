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
| **ネイティブデスクトップアプリ**の拡張（ペイン、ページ、ステータスバー、パレット、テーマ） | [Desktop Plugin SDK](/docs/developer-guide/desktop-plugin-sdk) |
| **Webダッシュボード**の拡張（タブ、シェルスロット、テーマ） | [Extending the Dashboard](../../user-guide/features/extending-the-dashboard.md) |
| **LLM / 推論バックエンド**（新しいプロバイダー） | [Model Provider Plugins](/docs/developer-guide/model-provider-plugin) |
| **ゲートウェイチャネル**（Discord/Telegram/IRC/Teams など） | [Adding Platform Adapters](/docs/developer-guide/adding-platform-adapters) |
| **メモリバックエンド**（Honcho/Mem0/Supermemory など） | [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) |
| **コンテキスト圧縮エンジン** | [Context Engine Plugins](/docs/developer-guide/context-engine-plugin) |
| **画像生成バックエンド** | [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) |
| **動画生成バックエンド** | [Video Generation Provider Plugins](/docs/developer-guide/video-gen-provider-plugin) |
| **Web検索 / 抽出バックエンド** | [Web Search Provider Plugins](/docs/developer-guide/web-search-provider-plugin) |
| **クラウドブラウザバックエンド**（Browserbase風のCDPセッションプロバイダー） | [Browser Provider Plugins](/docs/developer-guide/browser-provider-plugin) |
| **委任戦略**（サブエージェントごとにプロファイルや名前付きモデルを選ぶ） | [Delegation Provider Plugins](/docs/developer-guide/delegation-provider-plugin) — `ctx.register_delegation_provider()` |
| **computer-useドライバー**（`computer_use`ツールの背後にあるデスクトップ操作） | [Computer-useバックエンドプラグイン](#computer-use-backend-plugins) — `ctx.register_computer_use_provider()` |
| **シークレットマネージャーバックエンド**（vault / パスワードマネージャー / OSキーストア） | [Secret Source Plugins](/docs/developer-guide/secret-source-plugin) |
| **ダッシュボードのOIDC/認証プロバイダー** | [Web Dashboard — custom providers](../../user-guide/features/web-dashboard.md#custom-providers) — `ctx.register_dashboard_auth_provider()` |
| **TTSバックエンド**（任意のCLI — Piper、VoxCPM、Kokoro、ボイスクローニングなど） | [TTS custom command providers](../../user-guide/features/tts.md#custom-command-providers) — 設定駆動、Python不要 |
| **STTバックエンド**（カスタムwhisper / ASR CLI） | [Voice Message Transcription](../../user-guide/features/tts.md#voice-message-transcription-stt) — `HERMES_LOCAL_STT_COMMAND`をargvトークン化されたテンプレートに設定 |
| **MCP経由の外部ツール**（filesystem、GitHub、Linear、任意のMCPサーバー） | [MCP](../../user-guide/features/mcp.md) — `config.yaml`に`mcp_servers.<name>`を宣言 |
| **ゲートウェイイベントフック**（起動時、セッションイベント、コマンドで発火） | [Event Hooks](../../user-guide/features/hooks.md#gateway-event-hooks) — `~/.hermes/hooks/<name>/`に`HOOK.yaml` + `handler.py`を配置 |
| **シェルフック**（イベント時にシェルコマンドを実行） | [Shell Hooks](../../user-guide/features/hooks.md#shell-hooks) — `config.yaml`の`hooks:`配下に宣言 |
| **追加のスキルソース**（カスタムGitHubリポジトリ、プライベートスキルインデックス） | [Skills](../../user-guide/features/skills.md) — `hermes skills tap add <repo>` · [tapの公開](../../user-guide/features/skills.md#publishing-a-custom-skill-tap) |
| ファーストクラスの**コア**推論プロバイダー（プラグインではない） | [Adding Providers](/docs/developer-guide/adding-providers) |

設定駆動（TTS、STT、MCP、シェルフック）やドロップインディレクトリ（ゲートウェイフック）スタイルを含む、すべての拡張サーフェスを一覧でまとめた[Pluggable interfaces table](../../user-guide/features/plugins.md#pluggable-interfaces--where-to-go-for-each)も参照してください。
:::

:::caution サードパーティ製品のプラグインはコアツリーではなくスタンドアロンで配布する
**他者の製品やプロジェクト**を統合するプラグイン — オブザーバビリティ/メトリクスバックエンド、ベンダーSaaSコネクタ、アナリティクスダッシュボード、有料サービスとの連携 — は、`NousResearch/hermes-agent`にマージするのではなく、**スタンドアロンのプラグインリポジトリ**として構築・配布されます。ユーザーはそれらを`~/.hermes/plugins/`に、またはpipのエントリポイント経由でインストールします。このガイドの内容はすべて、スタンドアロンリポジトリからでも同じように動作します。これは結合度と保守性に関する判断（コアは速く進化し、私たちはあなたのバックエンドを所有していません）であり、品質の基準ではありません — 優れたプラグインであっても、独自のリポジトリに属するべき場合があります。Nous Research Discordの`#plugins-skills-and-skins`チャネルで宣伝してください。ポリシーについては[CONTRIBUTING.md](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)を参照してください。
:::

:::tip すでに組み込まれています: 自作する前にここを確認してください
Hermesは、プラグイン作者向けに以下をすでに提供しています。

- **カタログチェックをローカルで実行する:** `hermes plugins validate /path/to/your-plugin --install-deps`は、カタログのCIが実行するのと同じチェックを実行します。[Submitting to the Plugin Catalog](/docs/developer-guide/plugins/catalog-submission)を参照してください。
- **プラグインを隔離してテストする:** `hermes plugins doctor [path-or-id]`は、一時的な`HERMES_HOME`を使って、Hermesが使うのと同じ検出、マニフェストパーサー、`register(ctx)`、レジストリを実行します。[Plugin Doctorで検証する](#validate-with-plugin-doctor)を参照してください。
- **更新をまたいで状態を保持する:** `plugin_data_dir()`と`plugin_db()`は、`hermes plugins update`や`remove`を経ても残り、アクティブなプロファイルに従うデータディレクトリをプラグインに提供します。[永続的な状態を保存する](#store-durable-state)を参照してください。
- **Python依存関係を宣言する:** `plugin.yaml`の`python_dependencies`配下か、その隣の`pyproject.toml`に列挙します。[Python依存関係](#python-dependencies)を参照してください。
- **プラグインと一緒にスキルを配布する:** `ctx.register_skill()`で登録します。[スキルを同梱する](#bundle-skills)を参照してください。
- **特権的なホストサーフェスを要求する:** ユーザーが単一の同意画面を得られるよう、`capabilities:`配下に宣言します。[ケイパビリティを宣言する](#declaring-capabilities)を参照してください。
- **LLM呼び出しを行う:** ホスト所有の認証情報とフェイルクローズの信頼ゲートを備えた`ctx.llm`を使います。[Plugin LLM Access](/docs/developer-guide/plugin-llm-access)を参照してください。
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

ポータブルパッケージは、明示的に有効化しない限りインストール後は無効のままです。有効化されたパッケージは、直下の`skills/*/SKILL.md`ディレクトリと、ルートの`mcp.json`にあるstdio MCPサーバーを提供できます。スキルは読み取り専用で名前空間付きであり、`skills_list`と`skill_view`を通じて読み込まれます。MCPコマンドは、1つの実行ファイルトークンと別個の引数リストとして渡され、シェルを経由することはありません。ポータブルパッケージのスキルは、ネイティブプラグインと同じ形式で、プラグイン名にちなんで`<plugin-name>:<skill>`（`blender:blender`）と名付けられます。読み込まれた2つのプラグインが同じ修飾名を登録しようとした場合、2つ目のスキルは両者の名前を示す警告とともにスキップされます。データディレクトリ`plugin-data/agent-plugin-<slug>-<hash>`は、検出されたプラグインキーから導出された名前を保持するため、サニタイズ後の名前がディスク上で衝突することはありません。ポータブルパッケージのMCPサーバーは、ユーザー自身の`mcp_servers`ブロックと同じルールで、`mcp.json`で付けられた名前をそのまま保持します。そのため、モデルに見える`mcp__<server>__<tool>`という名前は、ツールの動詞部分をプロバイダーの64文字上限内に収めたままになります。サーバー名の重複は読み込み時の衝突として扱われます。`config.yaml`のサーバーがパッケージより優先され、先に読み込まれたパッケージが次のパッケージより優先されます。負けた側は、両者の名前を示す警告とともにスキップされます。

Hermesは、`plugin.json`、Agent Skillsのフロントマター、固定されたコンポーネントの配置場所、`mcp.json`、解決済みパス、シンボリックリンクの封じ込めをローカルで検証します。パッケージの読み込み中にJSONスキーマを取得することはありません。不正なスキルやMCPエントリは、有効な兄弟コンポーネントが読み込める場合、それぞれの境界でスキップされます。`PLUGIN_ROOT`は解決済みのパッケージルートを指します。`PLUGIN_DATA`は、Hermesが管理するプロファイル単位の書き込み可能ディレクトリを指します。ポータブルMCPの`env`で宣言された値は、可視のパッケージデータであり、シークレットの保存手段ではありません。`mcp.json`に認証情報を置かないでください。ユーザーの変数をstdioサーバーにそのまま渡すには（そうしない場合、HermesはMCPサーバーにフィルタリングされた環境を渡します）、`"NAME": "${NAME}"`として宣言してください。値はユーザーのシェルまたはプロファイルの`.env`から取得されます。変数が未設定（または空）の場合、そのエントリは存在しないものとして扱われます。サーバーがリテラルの`${NAME}`を受け取ることはなく、`NAME`はその行がなかった場合にサーバーが持つ値（通常は未設定）のままになります。オプションの変数も安全に転送できます。

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

サーバー名は`mcp.json`のエントリと一致していなければなりません。`untrusted`の場合、そのサーバーへのツール呼び出しのうち`readOnlyHint: true`の注釈がないものはすべて、先にユーザーに確認し（ツールごとにOnce、Session、Alwaysのいずれか）、誰も応答できない場合（cron、無人実行）はフェイルクローズします。他に受け付けられる値はデフォルトの`full`のみなので、パッケージはアクセスを狭めることはできても、広げることはできません。同じ名前の`config.yaml`サーバーは、trustを含めてパッケージのエントリを置き換えますが、独自に設定しない限り、パッケージが宣言したコストと支出上限（後述）は保持します。他のハーネスはこの拡張を無視します。`trust`は、同じサーバーエントリ内で`app`、`requires`、`liveness`と並べて置けます（[Application declarations](/docs/developer-guide/plugins/application-declarations)を参照）。

ツールがお金やクレジットを消費するサーバーは、各呼び出しのコストとチャットごとの予算を宣言できます。するとHermesは、承認プロンプトでそのコストをチャットの累計とともに示します。また、チャットの累計が`spend_limit`を超えることになる呼び出しは、ユーザーがそのサーバーを信頼済みにしていても、先に確認します。

```json
{
  "extensions": {
    "com.nousresearch.hermes": {
      "servers": {
        "renders": {
          "trust": "untrusted",
          "costs": {
            "create_image": 20,
            "animate": { "credits": [5, 20], "note": "per animation" }
          },
          "spend_limit": 100,
          "cost_unit": "credits"
        }
      }
    }
  }
}
```

`costs`のキーは、サーバーが列挙するとおりのMCPツール名です。コストは、非負の数値、`[min, max]`の範囲、または`credits`（数値または範囲）と短い`note`を持つオブジェクトのいずれかです。`spend_limit`はオプション（`0`または省略 = 上限なし）で、`cost_unit`のデフォルトは`credits`です。テーブルが不正な場合、パッケージはそのキーを示すエラーとともに読み込みに失敗します。これらはネイティブの`tool_costs`、`spend_limit`、`cost_unit`キーに対応します（[Tool costs and spend limits](../../user-guide/features/mcp.md#tool-costs-and-spend-limits)を参照）。コストはサーバーに関する事実なので、ユーザーの同名の`config.yaml`エントリ（ユーザーがtrustを広げる方法）があっても、そのエントリ自身がキーを設定しない限り保持されます。`spend_limit`はユーザーのものが優先され、`0`にして無効化することもできます。

長時間実行されるサーバーは、他のサーバーのデフォルトを変えることなく、Hermes固有のツール呼び出し`timeout`（秒単位）を`plugin.json`で宣言することもできます。

```json
{
  "extensions": {
    "com.nousresearch.hermes": {
      "servers": { "analysis": { "timeout": 1000 } }
    }
  }
}
```

値は、プラットフォームのスレッド待機上限内の正の有限数でなければなりません。これはネイティブのサーバーごとの`timeout`に対応し、省略した場合は既存の`timeouts.mcp.tool_call`へのフォールバックが維持されます。同名のネイティブ`mcp_servers`エントリは、タイムアウトを含めてパッケージのエントリ全体を引き続き置き換えます。これは待機の予算を延ばすものであり、キャンセルや権限のポリシーを変えるものではありません。サーバー内部の解析期限は別途設定し、その結果がクライアントに届くまでの時間を残しておいてください。他のハーネスはこの拡張を無視します。

このHermesが知らないサーバーエントリのキー（より新しいリリースで追加されたもの）は警告とともに無視され、`hermes plugins validate`がそれを列挙しますが、パッケージはインストールされます。このルールより前のリリースでは代わりにパッケージが拒否され、また、キーを無視するリリースは意図したより安全性の低い振る舞いをする可能性があるため、新しいキーはカタログの`requires_hermes`の下限とあわせて宣言してください。`plugin.json`には`requires_hermes`フィールドがないため、そのカタログの下限がポータブルパッケージにとって唯一の下限です。カタログ名によるインストールは古いHermesでは拒否され、`hermes plugins update`はHermesが更新されるまでインストール済みのピンを保持します。

明示的なサーバーごとのMCP呼び出し予算は、Hermesの外側のツール実行器にも届きます。予算が共有の並行バッチ期限を超える呼び出しは、直接呼ばれた場合でもツール検索経由の場合でも、逐次実行のバリアになります。その呼び出しは、同じモデル応答内の無関係なツールを延長することなく、独自の予算を得ます。既存のより短いMCP呼び出しは通常の並列受け入れポリシーを維持し、中断と承認の扱いは変わりません。

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

`hermes plugins doctor [path-or-id]`は、Hermes自身が使っているのと同じディレクトリ検出、マニフェストパーサー、名前空間付きインポート、`register(ctx)`、フックレジストリ、ツールレジストリを実行します。無効なフック名、`**kwargs`を受け取らないコールバック、登録の失敗、そして宣言されたツール/フックと実際に登録されたものとのずれを報告します。`desktop/plugin.js`を持つパッケージについては、デスクトップアプリがその古いコピーを実行している場合にも警告します（[Developing a unified package](/docs/developer-guide/desktop-plugin-sdk#developing-a-unified-package)を参照）。`--ci`を渡すと、エラー時に非ゼロで終了します。

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

`register()`が登録するすべてのツールを`provides_tools`に列挙してください。このフィールドは、ユーザーがインストールしたプラグインのツールが読み込まれるかどうかを決めるものでは**ありません**。プラグインが有効化されれば、宣言されているかどうかにかかわらず、`register()`が登録したものはすべて利用可能になります。このフィールドが実際に影響するもの:

- **`hermes plugins validate`**: 登録されたツールがリストと一致しない場合、「declared tools」チェックが失敗し、カタログへの受け入れがブロックされます。
- **カタログの一覧表示**: カタログおよびダッシュボード/デスクトップのPluginsページにおける「N tools」チップとツール名検索。
- **ダッシュボードの認証ヒント**: 「needs auth」と`hermes auth <name>`コマンドの表示には、宣言されたツールの可用性チェックのみが使われます。
- **バンドルされた`kind: platform`プラグインのみ**: このフィールドは、アダプターが遅延されたまま、CLI/TUIセッションで`tools.py`を読み込むスイッチになります。[Outbound client tools](/docs/developer-guide/adding-platform-adapters#outbound-client-tools-provides_tools)を参照してください。

追加できるオプションフィールド:
```yaml
author: Your Name
requires_env:          # gate loading on env vars; prompted during install
  - SOME_API_KEY       # simple format — plugin disabled if missing
  - name: OTHER_KEY    # rich format — shows description/url during install
    description: "Key for the Other service"
    url: "https://other.com/keys"
    secret: true
capabilities:          # privileged host surfaces you request (consent flow)
  - tools.override     # replace built-in tools (needs user consent)
  - llm.model_override # choose the model for host-owned LLM calls
```

### ケイパビリティを宣言する {#declaring-capabilities}

プラグインが特権的なホストサーフェス — 組み込みツールの上書き、`ctx.llm`呼び出しのモデル選択など — を必要とする場合は、`capabilities:`で宣言してください。インストール/有効化の時点でユーザーにそのリストが表示され、一度だけ同意します。後のバージョンでケイパビリティが追加された場合、更新フローでは追加分についてのみ再度確認します。宣言されていない、または同意されていないケイパビリティは単にオフになります（フェイルクローズ）。そのため、**使う前に確認し、グレースフルに劣化させてください**。

```python
def register(ctx):
    if ctx.has_capability("tools.override"):
        ctx.register_tool(..., override=True)
    else:
        ctx.register_tool(...)   # register under a non-conflicting name
```

既知のケイパビリティID: `tools.override`、`llm.provider_override`、`llm.model_override`、`llm.agent_id_override`、`llm.profile_override`、`llm.task_override`、`llm.side_question`（ターンの後にセッション自身のエージェントに質問する。[Plugin LLM Access](/docs/developer-guide/plugin-llm-access#asking-the-sessions-own-agent-ctxside_question)を参照）、`llm.session_fork`（セッションのエージェントの範囲を限定したバックグラウンドコピーを実行する。[Plugin LLM Access](/docs/developer-guide/plugin-llm-access#forking-the-sessions-own-agent-ctxfork_session)を参照）、`gateway.platform_actions`（正規のレジストリは`hermes_cli/plugin_capabilities.py`を参照）。未知のIDは無視されます。以前のケイパビリティごとの設定キー（`plugins.entries.<id>.allow_tool_override`など）も引き続き動作しますが非推奨です — ユーザーが単一の監査可能な同意画面を得られるよう、代わりにケイパビリティを宣言してください。ケイパビリティは同意 + 監査であり、**サンドボックスではありません**。ホストのAPIサーフェスをゲートするだけで、それ以上のことはしません。

**pipで配布されるプラグイン**は、インストール後に`plugin.yaml`ディレクトリを持たないため、代わりにディストリビューションのメタデータで、対になる`hermes_agent.plugin_capabilities`エントリポイントグループを通じてケイパビリティを宣言します。各宣言は`<plugin-id>.<capability-id>`という名前を持ち、`hermes_agent.plugins`エントリポイントと同じオブジェクトを指します。

```toml
[project.entry-points."hermes_agent.plugins"]
calculator = "my_pkg:register"

[project.entry-points."hermes_agent.plugin_capabilities"]
"calculator.tools.override" = "my_pkg:register"
```

Hermesはコードをインポートせずにインストール済みのメタデータからこれらを読み取るため、pipインストールでも`hermes plugins capabilities`と同意フローは正確なままです。

### ユーザーのCodexサインインを使う（コアが行うリクエスト） {#use-the-users-codex-sign-in-core-made-requests}

ユーザーのChatGPT / Codexサブスクリプションを必要とするプラグイン（OpenAI Realtimeのクライアントシークレットの発行、Codexバックエンドエンドポイントの呼び出し）は、`~/.codex/auth.json`やHermesの`auth.json`を読み取ったり、リフレッシュしたり、書き換えたりしては**いけません**。Codexのリフレッシュトークンは使い捨てです。独自にリフレッシュするプラグインはHermesのロック付きリフレッシュと競合し、負けた側のローテーションチェーンは失効させられ、ユーザーはサインアウトされてしまいます。代わりに、Hermesにリクエストを行わせてください。Hermesは自身のリフレッシュパスを通じて認証情報を解決し、`Authorization`を自ら付与し、レスポンスだけを返します。あなたのコードがトークンを目にすることはありません。

`plugin.yaml`でサインインを宣言します。

```yaml
name: live-voice
requires_auth:
  - openai-codex
```

次に、公開ヘルパーを呼び出します。これは普通の関数なので、`register(ctx)`のコードだけでなく、ダッシュボードプラグイン（`dashboard/plugin_api.py`）やベンダリングされたモジュールからもインポートできます。

```python
from hermes_cli.plugin_provider_requests import ProviderNotSignedIn, credentialed_provider_request

def mint_client_secret(session: dict) -> dict:
    try:
        resp = credentialed_provider_request(
            "openai-codex", "POST", "https://api.openai.com/v1/realtime/client_secrets",
            json={"session": session}, timeout=15)
    except ProviderNotSignedIn as exc:
        return {"ok": False, "error": str(exc)}  # tells the user how to sign in
    if resp.status != 200:
        return {"ok": False, "error": f"client secret failed ({resp.status}): {resp.text[:300]}"}
    return {"ok": True, **resp.json()}
```

これは、`~/.codex/auth.json`を読み取り、トークンの期限が近づくと`auth.openai.com/oauth/token`にPOSTし、結果を書き戻すような自作ヘルパーを置き換えるものです。そのコードは削除してください。プラグインに残されたリフレッシュ手順はありません。

`credentialed_provider_request(provider, method, url, *, json=None, headers=None, timeout=30.0)`は`ProviderResponse`（`status`、`headers`、`body`、`.text`、`.json()`）を返します。ルールは次のとおりです。

- **宣言したプラグインのみ。** 呼び出し元は、コールスタック上にファイルがあるプラグインです。Hermesはそれをpluginsディレクトリ配下で見つけ、その`plugin.yaml`を読み取ります。`requires_auth`にプロバイダーを列挙していないプラグインや、インストール済みのどのプラグインにも属さないコードは`PermissionError`を受け取ります。ヘルパーは自分の関数の中から呼び出してください。素のヘルパーを`run_in_executor`に渡すと、スタック上にプラグインのフレームが残らないため、呼び出しは拒否されます。
- **プロバイダーのオリジンのみ。** `openai-codex`の場合、トークンは`https://chatgpt.com`と`https://api.openai.com`にのみ送信されます。それ以外のURLは、リクエストが行われる前に`PermissionError`を発生させます。Codexの認証情報がカスタムゲートウェイにルーティングされるプロファイルも拒否されます。そのキーはゲートウェイのものだからです。
- **リダイレクトなし。** 3xxレスポンスはそのまま返され、追従されることはないため、ヘッダーが第三のホストに届くことはありません。
- **プロファイル単位。** 認証情報は、アクティブなプロファイルのCodexサインインであり、チャットが使うものと同じです（`hermes auth add openai-codex`、またはデスクトップアプリでのCodexサインイン）。サインインがない場合は`ProviderNotSignedIn`を受け取り、そのメッセージにはサインインコマンドが示されます。
- 独自の`headers`も送信されますが、Hermesが付与する`Authorization`（またはアカウント）ヘッダーを置き換えるものは除きます。

`requires_auth`は`hermes plugins show <name>`とインストール時に表示され、これを追加するカタログの再ピン留めでは、更新前にユーザーに確認します。ケイパビリティと同様に、これは同意と可視性のためのものです。**サンドボックスではありません**。プロセス内のプラグインコードは依然としてユーザーが読める任意のファイルを読めますが、そうするプラグインはレビュアーに拒否されます。

**使用量パネル:** 使用量エンドポイントを自分で呼び出さないでください。`agent.account_usage.fetch_account_usage("openai-codex", read_only=True)`を使ってください。これは`AccountUsageSnapshot`（プラン、`used_percent`と`reset_at`を持つセッション/週単位のウィンドウ）を返し、認証情報をリフレッシュしたりローテーションしたりすることはありません。**サインアウト**はプラグインではなくHermesの役割です。認証情報ファイルを削除するのではなく、ユーザーに`hermes auth logout openai-codex`を案内してください。

### 独自のOAuthクライアントを使う（Googleおよびその他のAPI） {#use-your-own-oauth-client-google-and-other-apis}

ユーザー自身のOAuthクライアント（例えばGoogle Cloudの「Desktop app」クライアント）でサードパーティAPIを呼び出すプラグインは、同意フローを実行したり、リフレッシュトークンを保存したり、それをリフレッシュしたりしません。クライアントを宣言すれば、Hermesがグラントを管理します。

```yaml
name: gemini-voice
requires_env:
  - name: GEMINI_VOICE_CLIENT_ID
  - name: GEMINI_VOICE_CLIENT_SECRET
    secret: true
oauth:
  - id: google
    authorize_url: https://accounts.google.com/o/oauth2/v2/auth
    token_url: https://oauth2.googleapis.com/token
    scopes:
      - https://www.googleapis.com/auth/calendar.events.owned
      - https://www.googleapis.com/auth/documents
    client_id_env: GEMINI_VOICE_CLIENT_ID
    client_secret_env: GEMINI_VOICE_CLIENT_SECRET  # optional: a public PKCE client has none
    hosts: [www.googleapis.com, docs.googleapis.com]
    authorize_params: {access_type: offline, prompt: consent}  # Google issues a refresh token only with these
```

ユーザーは、ターミナルから`hermes auth add gemini-voice:google`（プラグインのキー、コロン、エントリの`id`）で一度だけサインインします。これはモデルプロバイダーのヘルパーが実行するのと同じPKCEログインで、`127.0.0.1`へのループバックリダイレクトを使い、Hermesがリモートホストで動いている場合はSSHトンネルのヒントを表示します。IdPが宣言より少ないスコープしか付与しなかった場合、何も保存されません。`hermes auth status|refresh|logout gemini-voice:google`で、グラントの表示、リフレッシュ、削除ができます。グラントはアクティブなプロファイルの`auth.json`に保存されます。Hermesは残り時間が2分未満になると使用時にリフレッシュし、ゲートウェイやcronの下でサインインを開始することはありません。

プラグインは、エントリの`id`をプロバイダーとして、同じヘルパーを通じてAPIを呼び出します。

```python
from hermes_cli.auth_constants import AuthError
from hermes_cli.plugin_provider_requests import ProviderNotSignedIn, credentialed_provider_request

def create_document(title: str) -> dict:
    try:
        resp = credentialed_provider_request(
            "google", "POST", "https://docs.googleapis.com/v1/documents", json={"title": title}, timeout=20)
    except ProviderNotSignedIn as exc:
        return {"ok": False, "error": str(exc)}  # names `hermes auth add gemini-voice:google`
    except AuthError as exc:
        return {"ok": False, "error": f"{exc.code}: {exc}"}
    return {"ok": resp.status < 300, "status": resp.status, "body": resp.json()}
```

上記のCodexのルールに加えて、次のルールが適用されます。

- **HTTPSのみ。** `authorize_url`、`token_url`、およびすべての`hosts`エントリは`https`です（ループバックの例外はありません）。それ以外は`hermes plugins validate`とカタログの受け入れで失敗します。
- **宣言したホストのみ。** ベアラートークンは、エントリの`hosts`に完全一致（サブドメインなし）する`https://<host>`（ポート443）に送られます。それ以外のURL、`http`、ユーザー情報を含むURLは、リクエストが行われる前に`PermissionError`を発生させます。サインインプロバイダーのドメイン外のホストは、レビュアーが確認する警告になります。
- **独自のクライアント変数。** `client_id_env`と`client_secret_env`はプラグイン自身の変数を指定します。Hermes自身が使う変数（プロバイダーのAPIキー、プラットフォームのトークン）は拒否されます。Hermesはこれらを`token_url`に送信するためです。
- **プラグイン自身のグラント。** グラントは呼び出し元プラグインのキーの下に保存されるため、同じ`id`を宣言した別のプラグインはそれを使えません。また、プラグイン自身のエントリが、同名のHermesのサインインより優先されます。
- **グラントは宣言に従う。** グラントは作成時のエントリを記憶しています。トークンURL、ホスト、スコープ、クライアント変数を変更したプラグインのバージョンは、ユーザーが再度サインインするまで`ProviderNotSignedIn`を受け取ります。また、`oauth`エントリを追加したり、その認可ホスト、トークンホスト、ホスト、スコープを変更したりするカタログの再ピン留めでは、更新前に確認します。
- **エラー。** `ProviderNotSignedIn`: サインインしていない、失効した、またはサインイン後にエントリが変更された。`AuthError`のコード: `oauth_client_id_missing`と`oauth_client_secret_missing`（変数が設定されていない）、`invalid_client`、`unauthorized_client`、`invalid_scope`（IdPがクライアントを拒否した。ユーザーが修正する必要がある）、`oauth_refresh_failed`（ネットワークまたはサーバーのエラー。後で再試行する）。失効したグラント（`invalid_grant`）は廃棄され、次のサインインまでネットワークには何も送られません。エラーはトレースバック、チェーン、リクエストオブジェクトなしでプラグインに届くため、トークン、リフレッシュトークン、クライアントシークレットにたどり着くものはありません。
- **サインインと削除。** `hermes auth add`は、ブラウザを開く前に、コード、クライアントシークレット、トークンをどのホストが受け取るかを表示し、ゲートウェイ、cron、エージェントセッションからの実行を拒否します。`hermes plugins remove`はプラグインのグラントを削除し、`hermes auth logout <plugin>:<id>`はプラグインがなくなった後も引き続き動作します。

`hermes plugins validate`は、`oauth:`ブロックとそのクライアント変数が宣言されていることをチェックし、プラグインのコードがトークンURLを指定している場合に警告します。`requires_auth`と同様に、これは同意と可視性のためのものであり、サンドボックスではありません。

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
| `config_keys` | list | コードが読み取る、Hermes自身の設定セクション内のドット区切りパス（例: `memory.retaindb`）。各パスはそれ自身とその配下すべてをカバーするため、`hermes config get/set`は「not a recognized config key」と警告する代わりにそれらのキーを認識します。ただし、プラグインがアクティブな間（有効化されている、または選択された`memory.provider`である）に限ります。パスは少なくとも2階層の深さが必要で、Hermesがすでに定義しているパスは無視されるため、宣言によってセクション内の他の場所のタイプミスが隠れることはありません。自分のプラグイン専用の設定は`config_schema`に置いてください。これは、プラグインが読み取るコアのセクションにのみ使います。ポータブルな`plugin.json`パッケージは、これを`extensions."com.nousresearch.hermes".config_keys`配下に置きます。 |
| `license` | str | SPDX形式のライセンスID（例: `MIT`）。 |
| `homepage` | str | プロジェクトのURL。 |
| `tags` | list of str | 自由形式の検出用タグ（例: `[gateway, telegram]`）。 |
| `provides_locales` | list | 言語パックの宣言: ID（`- pl`）または`{id, endonym, rtl}`マッピング。ローダーは対応する`locales/<id>[.tui\|.desktop].yaml`を自動的に登録します — [言語パックを同梱する](#ship-a-language-pack)を参照。 |
| `requires_auth` | list of str | プラグインがHermesを通じてサインインを利用するプロバイダー（現在は`openai-codex`）。`credentialed_provider_request`に必須で、インストール時と`hermes plugins show`で表示されます。[ユーザーのCodexサインインを使う](#use-the-users-codex-sign-in-core-made-requests)を参照。 |
| `oauth` | list of mappings | プラグインがHermesを通じて使うOAuthクライアント: `id`、`authorize_url`、`token_url`、`scopes`、`client_id_env`、オプションの`client_secret_env`、`hosts`、オプションの`authorize_params`。`hermes auth add <plugin>:<id>`でサインインし、`credentialed_provider_request`を通じて使います。[独自のOAuthクライアントを使う](#use-your-own-oauth-client-google-and-other-apis)を参照。 |

```yaml
# plugin.yaml — manifest v2 example
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
  - "somepkg>=1.0,<2"     # consent before PM admission
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
- **何もインストールしない`pyproject.toml`**（`dependencies = []`で、`[build-system]`、エクストラ、依存関係グループがない）は宣言なしとして扱われます。依存関係に関する質問はなく、プラグインを有効化または更新しても依存関係の環境はそのままです。
- **`python_runtime: external`**は、サイドカーの依存関係を共有の和集合から除外します。HermesはそのPythonランタイムをインストールせず、その宣言も変更しません。
- **読み込むものがないのはエラー** — `hermes plugins validate`は、隣に`__init__.py`、`desktop/plugin.js`、`plugin.json`のいずれもない`plugin.yaml`を拒否します。pipレイアウトのパッケージには、ディレクトリプラグインのラッパーが必要です。
- `security.allow_lazy_installs: false`は、オンデマンドでの取得をブロックします。明示的な依存関係への同意と明示的な有効化がPMによる準備を許可します。検出によってインストールが行われることはありません。

`HERMES_HOME/plugins/`は`hermes update`とデスクトップの更新を経ても残ります。アップデーターが再構築するのはvenvとチェックアウトだけで、ホームディレクトリには手を付けません。

### 依存関係のセキュリティポリシー {#dependency-security-policy}

Hermesは、**自身の**依存関係をロックする際にその経過日数をチェックします。チェックアウトの`[tool.uv] exclude-newer = "3 days"`により、公開から3日未満のリリースは`uv.lock`に入りません。そのため、乗っ取られたアップロードがユーザーに届く前に上流で発見される時間が確保されます。ユーザーはその検証済みのロックからインストールし、ユーザーのマシンで日付が再チェックされることはありません。**このチェックはあなたのプラグインの依存関係には適用されません。** Hermesがあなたのプラグインを自身の環境に解決する際、Hermesの厳密なピンはそのまま維持され、Hermesが使う他のパッケージはHermesが宣言する範囲内で動く可能性があり、それ以外はすべてあなたの要件に従います。そのため、プラグインは昨日公開されたリリースを下限にして今日インストールすることができます。それが何を取り込むかについては、Hermesではなくプラグインの作者が責任を負います。Hermesが厳密にピン留めしているパッケージの別バージョンを要求するプラグインは、Hermesのピンが動くまでインストールに失敗します。

独自のポリシーを定め、それを守ってください。強く推奨される事項:

- **すべての依存関係に上限を付ける** — 安定版のパッケージには`>=floor,<next_major`、1.0未満のパッケージには`>=0.29,<0.32`。素の`>=X.Y`は、将来のすべてのリリースをレビューなしで採用してしまいます。
- **下限は今週のリリースではなく、API互換性のある最も古いバージョンにする。** 新しいwheelを下限にすると、それが公開された日にすべてのインストーラーがそれを使うことを強制されます。`>=old,!=broken,<next`なら、広い範囲を保ちつつ問題のある1つのリリースだけをスキップできます。
- **独自のリリース経過日数チェックを採用する**: 下限を新しいリリースに上げるまで数日待ち、自身のCIでは`uv --exclude-newer "3 days"`（または`UV_EXCLUDE_NEWER`）で解決してください。そうすれば、テストしたロックがそのままユーザーに届きます。
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

# Safe globals for expression evaluation — no file/network access
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


# Conversion tables — values are in base units
_LENGTH = {"m": 1, "km": 1000, "mi": 1609.34, "ft": 0.3048, "in": 0.0254, "cm": 0.01}
_WEIGHT = {"kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
_DATA = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
_TIME = {"s": 1, "ms": 0.001, "min": 60, "hr": 3600, "day": 86400}


def _convert_temp(value, from_u, to_u):
    # Normalize to Celsius
    c = {"F": (value - 32) * 5/9, "K": value - 273.15}.get(from_u, value)
    # Convert to target
    return {"F": c * 9/5 + 32, "K": c + 273.15}.get(to_u, c)


def unit_convert(args: dict, **kwargs) -> str:
    """Convert between units."""
    value = args.get("value")
    from_unit = args.get("from_unit", "").strip()
    to_unit = args.get("to_unit", "").strip()

    if value is None or not from_unit or not to_unit:
        return json.dumps({"error": "Need value, from_unit, and to_unit"})

    try:
        # Temperature
        if from_unit.upper() in {"C","F","K"} and to_unit.upper() in {"C","F","K"}:
            result = _convert_temp(float(value), from_unit.upper(), to_unit.upper())
            return json.dumps({"input": f"{value} {from_unit}", "result": round(result, 4),
                             "output": f"{round(result, 4)} {to_unit}"})

        # Ratio-based conversions
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
4. **`**kwargs`を受け取る:** Hermesはコンテキストキーワード（`task_id`、`session_id`、`conversation_id`、`user_task`、`parent_agent`など）を注入しますが、シグネチャで名前が指定されたものだけを転送します。そのため`def handler(args)`でも動作します。`**kwargs`は、追加的に拡張されていくコンテキスト全体を受け取るための手段です。
   プラグインの[セッションフォーク](/docs/developer-guide/plugin-llm-access#forking-the-sessions-own-agent-ctxfork_session)の内部では、ハンドラは`turn_origin="plugin_fork:<plugin id>"`も受け取ります。ライブのターンでの呼び出しには付きません。
5. **会話ごとの状態は`conversation_id`をキーにする:** `session_id`は現在のトランスクリプトのセグメントを指し、ローテーションする圧縮（`compression.in_place: false`）によって会話の途中で置き換えられます。`conversation_id`は会話の最初のセッションのIDであり、変わりません。`/new`と`/branch`は新しい会話を開始します。委任されたサブエージェントの呼び出しは、自身の`session_id`と委任元の`conversation_id`を持ちます。呼び出しにセッションがない場合（例えば`execute_code`スクリプト内からのツール呼び出し）は、どちらも空です。

## ステップ5: 登録処理を書く {#step-5-write-the-registration}

`__init__.py`を作成します。これはスキーマとハンドラを結びつけます。

```python
"""Calculator plugin — registration."""

import logging

from . import schemas, tools

logger = logging.getLogger(__name__)

# Track tool usage via hooks
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

    # This hook fires for ALL tool calls, not just ours
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
    return result  # returned to the caller's chat UI

def register(ctx):
    # Handlers receive a single raw_args string; close over ctx via a lambda.
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

**シークレットが`config.yaml`に触れることはありません。** `secret`フィールドが持つのは`.env`の変数名と値が設定済みかどうかだけです。デスクトップは、プロバイダーのAPIキーと同じ認証情報ルート（`PUT /api/env`）を通じて値を保存し、プラグインは`requires_env`エントリとまったく同じように`agent.secret_scope`の`get_secret("MY_PLUGIN_API_KEY")`でそれを読み取ります。`.env`、`auth.json`、その他のツールの認証情報ファイルを自分で読み取ることは決してしないでください（カタログのルール11）。`plugins.manage settings`アクションは、シークレットキーと、型や`choices`がスキーマと一致しない値を拒否します。

**ポータブルパッケージ（`plugin.json`）**は、Agent Plugins v1が未知のトップレベルフィールドを拒否するため、同じスキーマをHermesのベンダー名前空間の下に宣言します。

```json
{
  "extensions": {
    "com.nousresearch.hermes": {
      "config_schema": {
        "api_key": { "type": "secret", "env": "MY_PLUGIN_API_KEY" },
        "show_chip": { "type": "bool", "default": true, "label": "Show the status-bar chip" }
      }
    }
  }
}
```

`env`が`mcp.json`でパススルーされる変数（`"env": { "MY_PLUGIN_API_KEY": "${MY_PLUGIN_API_KEY}" }`）を指しているシークレットは、すぐにMCPサーバーに届きます。保存後、Hermesは解決済みの起動入力が変わったプラグインのサーバーを再接続します（新しいキーは該当し、トグルは該当しません）。そのため、このフォームはユーザーがサーバーのキーを設定する場所にもなります。

**ポーリングせずに保存に反応する。** 保存のたびに`plugin.<id>.settings.changed`がブロードキャストされます（ペイロードは`{ written: [...keys] }`。シークレットのみの保存では空のリストが送られます）。そのため、デスクトップ側はすぐに設定を読み直せます: `ctx.onEvent('plugin.my-plugin.settings.changed', reload)`。値は、自分の`plugin_api.py`から`hermes_cli.config`の`load_config_readonly()`（`plugins.entries.<id>.settings`）で読み取り、スキーマのデフォルトにフォールバックしてください。

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
├── plugin.yaml      # "I'm calculator, I provide tools and hooks"
├── __init__.py      # Wiring: schemas → handlers, register hooks
├── schemas.py       # What the LLM reads (descriptions + parameter specs)
└── tools.py         # What runs (calculate, unit_convert functions)
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
# In tools.py or __init__.py
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

# <hermes home>/plugin-data/<name>/ — created on first use
state_file = plugin_data_dir("my-plugin") / "state.json"

# Or a SQLite database at <data dir>/data.db (WAL mode, thread-friendly)
conn = plugin_db("my-plugin")
conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY)")
```

プラグインごとに1つのディレクトリがあるため、すべてのプラグインのデータを予測可能な1つの場所で確認できます。シークレットはここに置くべきではありません — 認証情報の読み取りは、他のすべての場所と同様に、標準の`.env` / シークレットスコープのパスを通じて行います。

### テキストを保存・送信する前にシークレットをリダクトする {#redact-secrets-before-storing-or-sending-text}

プラグインが保存またはアップロードするテキスト（トランスクリプト、差分、ログ）には、ユーザーが貼り付けたAPIキーが含まれている可能性があります。Hermesの内部をインポートするのではなく、まず`ctx.redact_text()`に通してください。

```python
safe = ctx.redact_text(transcript, force=True)  # force: this text leaves the machine
```

これはHermes自身の出力と同じスクラバーを、最も強い設定で使います。プレフィックス付きのキーは`«redacted:sk-…»`のようなセンチネルになり、キーの文字は一切残りません。また、`API_KEY=…`のような代入はマスクされます。`ctx.register_redaction_patterns()`で登録されたパターンも適用されます。`force`を付けない場合、`security.redact_secrets: false`を設定したユーザーには、テキストが変更されずに返されます。

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
skill_view("my-plugin:my-workflow")   # → plugin's version
skill_view("my-workflow")              # → built-in version (unchanged)
```

**主な特性:**
- プラグインスキルは**読み取り専用**です — `~/.hermes/skills/`には入らず、`skill_manage`で編集できません。
- プラグインスキルはシステムプロンプトの`<available_skills>`インデックスに**載りません** — 明示的にオプトインで読み込むものです。
- 素のスキル名は影響を受けません — 名前空間が組み込みスキルとの衝突を防ぎます。
- エージェントがプラグインスキルを読み込むと、同じプラグインの兄弟スキルを列挙したバンドルコンテキストバナーが先頭に付加されます。

:::tip レガシーパターン
古い`shutil.copy2`パターン（スキルを`~/.hermes/skills/`にコピーする方法）も依然として動作しますが、組み込みスキルとの名前衝突リスクを生みます。新しいプラグインでは`ctx.register_skill()`を推奨します。
:::

### スキルカタログを読み取る {#read-the-skill-catalog}

エージェントがどのスキルを参照できるかを表示するプラグイン（カタログビュー、「このスキルを開く」、インストール済みコピーとのずれのチェック）は、スキルのディレクトリを自分で走査するのではなく、`skills_list`および`skill_view`ツールが使うのと同じヘルパーを呼び出すべきです。これらの名前は公開されており、互換性コントラクトの対象です。

| インポート | 得られるもの |
|---|---|
| `tools.skills_tool.find_all_skills(*, skip_disabled=False)` | 読み込み可能なスキルを`{"name", "description", "category"}`のdictとして返します。`skills_list`が使う、キャッシュ付き・先勝ち・プラットフォーム/環境でゲートされた走査です。`name`は`skill_view()`が受け付けるものです。 |
| `tools.skills_tool.sort_skills(skills)` | 唯一の一覧の並び順（カテゴリ、次に名前）。 |
| `tools.skills_tool.skill_search_dirs()` | `(roots, active_dir)`: 優先順位順（project、local、`create_dir`、external）の`(tier, dir)`ルートと、ライブプロファイルのskillsディレクトリ。 |
| `tools.skills_tool.skill_lookup_path_error(name)` | `name`が絶対パスであるか`..`を含む場合はエラー文字列、それ以外は`None`。検索の前に必ずチェックしてください。 |
| `tools.skills_tool.locate_skill(name, local_category_name, roots)` | `roots`全体にわたる1つの名前に対する`(error_json, skill_dir, skill_md)`: 階層の優先順位、同一階層での衝突の拒否、プロジェクトスキルの隔離ゲートを適用します。`local_category_name`は追加で試す相対パスであり（カテゴリフィルタではありません）、必要でなければ`None`を渡してください。`skill_md`は`error_json`が`None`の場合にのみ設定されます。 |
| `tools.skills_sync.dir_hash(directory)` | バンドルされたスキルの同期処理が保存するパッケージハッシュ。ユーザーのコピーがインストールされたものとまだ一致しているかを判断できます。 |
| `tools.skill_usage.read_skill_name(skill_md, fallback)` | `SKILL.md`のフロントマターの`name:`、なければ`fallback`。 |

```python
from tools.skills_tool import find_all_skills, locate_skill, skill_lookup_path_error, skill_search_dirs, sort_skills

def list_skills():
    return sort_skills(find_all_skills())

def open_skill(name):
    if err := skill_lookup_path_error(name):
        return err
    roots, _active = skill_search_dirs()
    error, _skill_dir, skill_md = locate_skill(name, None, roots)
    return error or skill_md.read_text(encoding="utf-8-sig", errors="replace")
```

プラグインが登録したスキル（`ctx.register_skill`）は、これらのルート配下のディスク上には存在しません。`find_all_skills()`はディスク上のカタログのみを返します。

### Hermes自身のヘルパーでファイルパスをチェックし、YAMLを読み取る {#check-file-paths-and-read-yaml-with-hermess-own-helpers}

モデルが選んだパス（添付ファイル、エクスポート先、要約するファイル）を読み書きするプラグインは、Hermes自身のファイルツールが拒否するのと同じパスを拒否すべきです。具体的には、認証情報ストア、skills-hubのキャッシュ、プロジェクトの`.env`ファイル、デバイスや`/proc`のプロセスファイル、そして`HERMES_WRITE_SAFE_ROOT`の外にあるものすべてです。独自の拒否リストを保持するのではなく、以下を呼び出してください。

| インポート | 得られるもの |
|---|---|
| `agent.file_safety.get_read_block_error(path)` | Hermesのファイルツールが`path`の読み取りを拒否する場合はエラー文字列、それ以外は`None`。絶対パスを渡してください。 |
| `agent.file_safety.raise_if_read_blocked(path)` | 同じチェックで、ブロック時に`ValueError`を発生させます。 |
| `agent.file_safety.get_write_denied_error(path)` | Hermesのファイルツールが`path`への書き込みを拒否する場合はエラー文字列、それ以外は`None`。 |
| `agent.file_safety.is_write_denied(path)` | その書き込みが拒否される場合は`True`。 |
| `hermes_yaml.safe_load(text)` / `hermes_yaml.safe_dump(data)` | Hermesが自身の設定に使うYAMLのリーダーとライター。HermesはPyYAMLを同梱していないため、プラグインが宣言しない限り`import yaml`は失敗します。 |

```python
from agent.file_safety import get_read_block_error
from hermes_yaml import safe_load

def load_recipe(path):
    if err := get_read_block_error(path):
        return err
    with open(path, encoding="utf-8") as fh:
        return safe_load(fh)
```

上記のように、名前そのものをインポートしてください（`from hermes_yaml import safe_load`）。`hermes plugins validate`はそれらの書き方を受け付け、素の`import agent.file_safety`はプライベートインポートとして報告します。Hermes自身の設定を読み取るには、`config.yaml`を自分でパースするのではなく、`hermes_cli.config`の`load_config_readonly()`を使ってください。

### リダイレクト時に漏らさずに認証情報付きHTTPリクエストを送る {#send-a-credentialed-http-request-without-leaking-it-on-redirect}

素の`urllib.request.urlopen`は、`Authorization`ヘッダーを付けたまま別のホストへのリダイレクトに追従します。`hermes_cli.urllib_security.open_credentialed_url(request, timeout=...)`は、Hermes自身のプロバイダー呼び出しと同じ方法で`urllib.request.Request`を開きます。リダイレクトが元のオリジンを離れると認証情報ヘッダーは削除され、プロセスにインストールされたオープナー（プロキシ、TLSの信頼設定）は維持されます。

```python
import urllib.request

from hermes_cli.urllib_security import open_credentialed_url

def fetch_models(base_url, api_key):
    request = urllib.request.Request(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
    with open_credentialed_url(request, timeout=10) as response:
        return response.read()
```

### 言語パックを同梱する {#ship-a-language-pack}

プラグインは、UI言語を追加したり、既存の言語の文言を上書きしたりできます。これは、Python（`agent.i18n.t()`: 承認プロンプト、ゲートウェイの返信、ツールの動詞、ヒント）、`hermes --tui`インターフェース、デスクトップアプリのすべてのサーフェスに一度に適用されます。`provides_locales`を宣言してYAMLを同梱するだけで、**Pythonは不要**です。

```
~/.hermes/plugins/hermes-lang-pl/
├── plugin.yaml
└── locales/
    ├── pl.yaml            # core (Python) strings — same key tree as the bundled locales/en.yaml
    ├── pl.tui.yaml        # optional: TUI strings (keys in locales/_keys.tui.json)
    └── pl.desktop.yaml    # optional: Desktop strings (keys in locales/_keys.desktop.json)
```

```yaml
name: hermes-lang-pl
version: 1.0.0
description: Polish language pack
provides_locales:
  - id: pl              # lowercase BCP-47-style id: pl, pt-br, zh-hant
    endonym: Polski     # what language switchers show
    rtl: false
```

`provides_locales`が宣言されている場合、ローダーは`register()`の前に`ctx.register_locale_dir(<plugin>/locales)`を呼び出します（`__init__.py`のないマニフェストのみのパックは、マニフェストのみのデスクトッププラグインと同様に読み込まれます）。カタログは**階層的かつ部分的**です: パック → ユーザーオーバーレイ（`<HERMES_HOME>/locales/`） → 同梱 → 英語 → キー。パックには変更するキーだけがあればよく、キーごとに最後に読み込まれたパックが優先されます。コアの値は英語と同じ名前付きの`{placeholders}`を保持します。英語の値が関数になっているTUI/デスクトップのエントリは、位置指定の`{0}`、`{1}`プレースホルダーを持つ文字列として記述します。

コードを持つプラグインは、プログラムから登録できます。ハンドルは`PluginRegistration`なので、プラグインをアンロードするとそのレイヤーは削除されます。また、登録によって`display.language`が変わることはありません。

```python
def register(ctx):
    here = Path(__file__).parent
    ctx.register_locale("pl", here / "locales" / "pl.yaml", endonym="Polski")        # YAML path
    ctx.register_locale("pl", {"approval": {"denied": "      ✗ Odrzucono"}})          # mapping, nested or flat
    ctx.register_locale("pl", here / "locales" / "pl.tui.yaml", surface="tui")       # core | tui | desktop
    ctx.register_locale_dir(here / "locales")                                         # every <lang>[.surface].yaml
```

`hermes plugins validate`は、宣言された各IDについて、パース可能でテキストのみの`locales/<id>.yaml`があるかをチェックし（テキスト以外のリーフはエラー）、そのサーフェスの英語カタログに存在しないキーの名前を**警告**します。レンダラーは、`i18n.languages` / `i18n.catalog` RPCを通じてパックのレイヤーを取得します。ユーザー向けガイド: [Language Packs](/docs/user-guide/features/language-packs)。

### 環境変数でゲートする {#gate-on-environment-variables}

プラグインがAPIキーを必要とする場合:

```yaml
# plugin.yaml — simple format (backwards-compatible)
requires_env:
  - WEATHER_API_KEY
```

`WEATHER_API_KEY`が設定されていない場合、プラグインは明確なメッセージとともに無効化されます。クラッシュもエージェント内のエラーもなく、ただ「Plugin weather disabled (missing: WEATHER_API_KEY)」と表示されるだけです。

ユーザーが`hermes plugins install`を実行すると、未設定の`requires_env`変数について**対話的にプロンプト**が表示されます。値は自動的に`.env`に保存されます。

より良いインストール体験のためには、説明とサインアップURLを含むリッチ形式を使ってください。

```yaml
# plugin.yaml — rich format
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
    # Use the SDK here.
```

引数は`pyproject.toml`のエクストラ名です。任意のパッケージ指定やプラグイン修飾されたキーではありません。古い`LAZY_DEPS`レジストリと`FeatureUnavailable`例外はもう存在しません。

新しい環境が選択された場合、ヘルパーは再起動が必要であることを報告することがあります。実行中のプロセス内で2つ目の環境からインポートするのではなく、そのエラーを返してください。すでに利用可能な依存関係は、`security.allow_lazy_installs`がfalseであってもインストールの必要はありません。

ディレクトリプラグイン自身のPython依存関係については、その`pyproject.toml`の`[project]`配下に`dependencies`を宣言してください。作者が書いたプロジェクトファイルがない場合、PMは`plugin.yaml`または`plugin.yml`にあるレガシーの`pip_dependencies`リストと`python_dependencies`リストを結合します。PMが以前に生成したプロジェクトファイルは、これらのリストを上書きしません。同意、ワークスペースへの所属、最新性のチェックには同じ宣言が使われます。PMは、プラグインを有効化する前に、コアの要件とともにそれらの依存関係を準備します。生成されたワークスペースが、プラグインディレクトリや同梱のロックファイルを書き換えることはなく、インストールするインタープリタのプラットフォームに対してのみ解決されます。別のプラットフォーム向けのwheelを提供しない依存関係が、プラグインをブロックすることはありません。依存関係が衝突した場合は受け入れが拒否され、以前の選択が保持されます。PMが他のプラグインを自動的に無効化することはありません。

pipで手動インストールした依存関係は、永続的なPMの宣言ではありません。後で環境が置き換えられた際に保持されるとは限りません。Pythonランタイムが PMの外側にあるラッパープラグインは、[memory-provider survival contract](/docs/developer-guide/memory-provider-plugin#hermes_home-survival-contract-what-wrappers-can-rely-on)を利用できます。ランタイムのレイアウトと遅延インストールのポリシーについては、[Package management](/docs/reference/package-management)を参照してください。



### スレッドセーフな遅延シングルトン {#thread-safe-lazy-singletons}

プラグインは、SDKクライアント、HTTPセッション、コネクションプールといった高コストなオブジェクトを、初回使用時に構築されるモジュールレベルの変数にキャッシュすることがよくあります。

```python
_client = None

def get_client():
    global _client
    if _client is not None:
        return _client
    _client = ExpensiveClient(...)   # ← TOCTOU race
    return _client
```

これは落とし穴です。Hermesは1つのプロセス内で複数のスレッド（委譲されたツール呼び出し、バックグラウンドワーカー、自己改善フォーク）を実行するため、`_client`が設定される前に2つのスレッドが`get_client()`に到達し、**両方**が`is not None`チェックを通過し、**両方**が高コストな構築を実行し、2番目の書き込みが1番目を上書きしてしまう可能性があります — その結果、負けた側が開いたリソース（コネクション、ファイルハンドル、バックグラウンドスレッド）がリークします。

ロックを自前で書かないでください。`plugins/plugin_utils.py`のヘルパーを使います。

```python
from plugins.plugin_utils import lazy_singleton, SingletonSlot

# Zero-arg accessor → decorate it:
@lazy_singleton
def get_client():
    return ExpensiveClient(load_config())   # runs exactly once

client = get_client()    # safe across threads
get_client.reset()       # drop the instance (tests / teardown)


# Accessor that takes a build argument → use a slot:
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
    check_fn=lambda: _has_optional_lib(),  # False = tool hidden from model
)
```

### 組み込みツールを上書きする {#overriding-a-built-in-tool}

組み込みツールを独自の実装に置き換えるには（例えば、`web_search`を社内のカスタムインデックスに置き換えるなど）、`override=True`を渡します。

```python
def register(ctx):
    ctx.register_tool(
        name="web_search",                   # same name as the built-in
        toolset="plugin_my_search",          # your own toolset namespace
        schema={...},
        handler=my_corporate_search,
        override=True,                       # explicit opt-in
    )
```

`override=True`がない場合、レジストリは別のツールセットの既存ツールを覆い隠すような登録をすべて拒否します — これにより意図しない上書きを防ぎます。**組み込み**ツールを上書きするには、さらにオペレーターが`config.yaml`の`plugins.entries.<plugin_id>.allow_tool_override: true`でオプトインする必要があります。このゲートがない場合、`register_tool(override=True)`は`PluginToolOverrideError`を発生させます。上書きはログに記録されるため、`~/.hermes/logs/agent.log`で監査できます。プラグインは組み込みツールの後に読み込まれるため、登録順序は正しく、あなたのハンドラが組み込みのものを置き換えます。

**同梱されていないプラグインにはオペレーターの許可も必要です。** Hermesコアに同梱されていないプラグイン（user、project、pipソース）では、既存の組み込みツールに対する`override=True`に、さらに`config.yaml`でのプラグインごとのオプトインが必要です。

```yaml
plugins:
  entries:
    my-plugin:                    # the plugin's registry key from `hermes plugins list`
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

### ユーザーが目にする前に返信を書き換える {#rewrite-the-reply-before-the-user-sees-it}

`transform_llm_output`フックは完成した返信を書き換えますが、ユーザーは通常、それがストリーミングされるのをすでに見ています。プラグインが先に返信を確認する必要がある場合（コンテンツをリダクト、フィルタリング、ブロックする場合）は、`hold_stream=True`を渡してください。

```python
def register(ctx):
    ctx.register_hook("transform_llm_output", redact, hold_stream=True)
```

すると、そのプラグインを読み込むプロファイル内のすべてのセッションで、返信のストリーミングが停止します。リファレンスに列挙されたサーフェスでは、ユーザーには入力中または思考中のインジケーターが表示され、その後に書き換えられた最終版だけが表示されます。それらのターンでは、ツール呼び出しの合間のコメンタリーは破棄され、フックが例外を発生させたりタイムアウトしたりした場合、ユーザーには元の返信の代わりに短い「reply withheld」通知が表示されます。装飾的な変換（フッター、スタイリング）ではこのフラグをオフのままにして、返信のストリーミングを維持してください。サーフェスごとの詳細: [Holding the reply until it is rewritten](../../user-guide/features/hooks.md#hold-stream)。

### フックリファレンス {#hook-reference}

各フックは、**[Event Hooks reference](../../user-guide/features/hooks.md#plugin-hooks)**で詳しく説明されています（コールバックシグネチャ、パラメータ表、各フックが正確にいつ発火するか、例）。以下はサマリーです。

| フック | 発火するタイミング | コールバックシグネチャ | 戻り値 |
|------|-----------|-------------------|---------|
| [`pre_tool_call`](../../user-guide/features/hooks.md#pre_tool_call) | 任意のツールが実行される前 | `tool_name: str, args: dict, task_id: str` | オプションのディレクティブ: `{"action": "block", "message": ...}`で呼び出しを拒否、`{"action": "approve", "message": ...}`で人間による承認ゲートにエスカレート |
| [`post_tool_call`](../../user-guide/features/hooks.md#post_tool_call) | 任意のツールが返った後 | `tool_name: str, args: dict, result: str, task_id: str, duration_ms: int` | 無視される |
| [`pre_llm_call`](../../user-guide/features/hooks.md#pre_llm_call) | ターンごとに1回、ツール呼び出しループの前 | `session_id: str, user_message: str, conversation_history: list, is_first_turn: bool, model: str, platform: str` | [コンテキストインジェクション](#pre_llm_call-context-injection) |
| [`post_llm_call`](../../user-guide/features/hooks.md#post_llm_call) | ターンごとに1回、ツール呼び出しループの後（成功したターンのみ） | `session_id: str, user_message: str, assistant_response: str, conversation_history: list, model: str, platform: str, side_question_token: str \| None, review_signals: dict` | 無視される |
| `pre_api_request` | プロバイダーへの各生APIリクエストの前（モデルがツールを呼び出す場合はターンごとに複数回） | `session_id: str, model: str, provider: str, base_url: str, api_mode: str, api_call_count: int, message_count: int, tool_count: int, approx_input_tokens: int, max_tokens: int, request: dict` | 無視される |
| `post_api_request` | プロバイダーへの各生APIリクエストが返った後 | `pre_api_request`のフィールドに加えて`api_duration: float, finish_reason: str, response_model: str \| None, usage: dict, response: dict, assistant_content_chars: int, assistant_tool_call_count: int` | 無視される |
| `api_request_error` | プロバイダーAPI呼び出しが例外を発生させたとき | 相関フィールドに加えて`status_code: int \| None, retry_count: int \| None, max_retries: int \| None, retryable: bool \| None, reason: str \| None, error: dict, request: dict` | 無視される |
| `pre_auxiliary_call` | 補助LLM呼び出し（タイトル付け、圧縮、MoA、ビジョン、承認など）の各プロバイダー試行の前。`pre_api_request`ではない | `aux_task: str`に加えて`pre_api_request`のフィールド（`session_id`/`task_id`/`turn_id`は親ターンのものか空、`api_request_id: str`、`retry_count: int`、`streaming: bool`、`request: dict`） | 無視される |
| `post_auxiliary_call` | その試行が返った後、または例外を発生させた後 | `pre_auxiliary_call`のフィールドに加えて`api_duration: float, finish_reason, response_model, usage: dict \| None, response: dict \| None, error: str \| None, error_type: str \| None` | 無視される |
| [`on_session_start`](../../user-guide/features/hooks.md#on_session_start) | 新しいセッションが作成されたとき（最初のターンのみ） | `session_id: str, model: str, platform: str` | 無視される |
| [`on_session_end`](../../user-guide/features/hooks.md#on_session_end) | すべての`run_conversation`呼び出しの終了時 + CLI終了時 | `session_id: str, completed: bool, interrupted: bool, model: str, platform: str` | 無視される |
| [`on_session_finalize`](../../user-guide/features/hooks.md#on_session_finalize) | CLI/ゲートウェイがアクティブなセッションを破棄するとき | `session_id: str \| None, platform: str` | 無視される |
| [`on_session_reset`](../../user-guide/features/hooks.md#on_session_reset) | ゲートウェイが新しいセッションキーに切り替えるとき（`/new`、`/reset`） | `session_id: str, platform: str` | 無視される |
| [`on_session_delete`](../../user-guide/features/hooks.md#on_session_delete) | セッションの削除または意図的なアーカイブがコミットされたとき（すべてのサーフェス、スイープ、prune） | `session_id: str, reason: str, profile: str \| None`、オプションの`deleted_ids: list[str]`、`sessions_dir: str` | 無視される |
| [`post_compaction`](../../user-guide/features/hooks.md#post_compaction) | コンパクションがコミットされたとき（任意のパス: auto、`/compress`、ゲートウェイ、Codex、micro、prune） | `session_id: str, old_session_id: str, kind: str, reason: str, before_tokens: int \| None, after_tokens: int \| None, messages_removed: int \| None` | 無視される |
| [`post_curator_run`](../../user-guide/features/hooks.md#post_curator_run) | キュレーターのパスがレポートディレクトリを書き込んだとき | `run_dir: str, run_json_path: str, report_md_path: str, dry_run: bool` | 無視される |
| [`gateway_platform_event`](../../user-guide/features/hooks.md#gateway_platform_event) | 認可されたプラットフォームネイティブのイベントがゲートウェイの境界で正規化されたとき（Telegramのリアクションとメッセージ編集、Discordのメッセージ編集と削除、スレッドの作成と名前変更。[イベントごとのコントラクト](../../user-guide/features/hooks.md#gateway_platform_event)を参照） | `platform: str, event_type: str, payload: dict` | 無視される |
| `kanban_task_claimed` | kanbanタスクが取得されたとき（ディスパッチャープロセス、ワーカーの起動前） | `task_id: str, board: str \| None, assignee: str \| None, run_id: int \| None, profile_name: str` | 無視される |
| `kanban_task_completed` | kanbanタスクが完了したとき（ワーカープロセス） | `task_id, board, assignee, run_id, profile_name, summary: str \| None` | 無視される |
| `kanban_task_blocked` | kanbanタスクがブロックされたとき（ワーカープロセス） | `task_id, board, assignee, run_id, profile_name, reason: str \| None` | 無視される |

ほとんどのフックは、fire-and-forget型のオブザーバーです — 戻り値は無視されます。例外は、会話にコンテキストをインジェクトしたり、モデルに届く前にターンをブロックしたりできる`pre_llm_call`と、block/approveディレクティブを返せる`pre_tool_call`です。

すべてのコールバックは、前方互換性のため`**kwargs`を受け取るべきです。フックコールバックがクラッシュした場合、ログに記録されてスキップされます。他のフックとエージェントは通常どおり続行します。

kanbanのライフサイクルフックはボードDBの変更がコミットされた**後**に発火するため、コールバックは常に永続化された状態を参照でき、SQLiteの書き込みロックを保持することはありません。kanbanワーカーは別個の`hermes -p <profile> chat -q`サブプロセスとして実行されるため、`kanban_task_claimed`は**ディスパッチャー**プロセスで発火し、`kanban_task_completed` / `kanban_task_blocked`は**ワーカー**プロセスで発火します — すべての遷移を一元的に観測するにはディスパッチャーで、タスクごとのセッション内コンテキストを得るにはワーカーでフックしてください。

**APIリクエストフック**は、プロバイダーへの生リクエストを観測するオブザーバーで、ターン単位の`pre_llm_call` / `post_llm_call`ペアより1段下のレベルにあります。ツールを呼び出す1つのターンは複数のAPIリクエストを行い、これらのフックはそれぞれのリクエストの前後で発火します。これらはオブザーバビリティプラグイン（トレーシング、コスト計算、レイテンシダッシュボード）のために存在します。`request`と`response`のkwargsは、プロバイダーペイロードをサニタイズしてサイズ上限を設けたJSONビューであり（機密キーはリダクトされ、長い文字列は切り詰められ、SDKオブジェクトは正規化されます）、`usage`はプレーンなトークンサマリーのdictです。すべてのペイロードは相関フィールド`turn_id`、`api_request_id`、`task_id`、`session_id`、`api_call_count`を持つため、プラグインはリクエスト、ツール呼び出し、ターンを結びつけることができます。`api_request_error`はプロバイダー呼び出しが例外を発生させたときに発火し、`status_code`、`retry_count` / `max_retries`、`retryable`、`reason`、そして`type`と`message`を持つ`error` dictを追加します。

### 各フックが発火する場所 {#hook-fire-sites}

`hermes_cli.plugins.VALID_HOOKS`のすべての名前には、少なくとも1つのコアの発火箇所があります。この表は、それぞれを発火させる関数を示しているため、正確な呼び出しとペイロードを読むことができます。パスはリポジトリルートからの相対パスです。フックを追加するときは、同じ変更でここに行を追加してください。

| フック | 発火させる箇所 |
|------|----------|
| `pre_tool_call` | `hermes_cli/plugins.py` `_dispatch_pre_tool_call_hooks`。`agent/tool_executor.py` `_pre_tool_block`、`agent/agent_runtime_helpers.py` `_pre_tool_block_message`、`model_tools.py` `_pre_dispatch_guards`から呼び出される |
| `post_tool_call` | `model_tools.py` `_emit_post_tool_call_hook` |
| `transform_terminal_output` | `tools/terminal_tool_result.py` `_apply_output_transform_hook` |
| `transform_tool_result` | `model_tools.py` `_apply_transform_tool_result_hook` |
| `transform_llm_output` | `agent/turn_finalizer.py` `apply_llm_output_transform` |
| `pre_llm_call` | `agent/turn_context.py` `_collect_pre_llm_call_context` |
| `post_llm_call` | `agent/turn_finalizer.py` `_apply_output_hooks` |
| `on_stream_start` / `on_stream_end` | `agent/stream_delivery.py` `StreamDeliveryMixin._emit_stream_start` / `_emit_stream_end` |
| `on_stream_delta` | `agent/stream_delivery.py` `StreamDeliveryMixin._fire_stream_delta`（テキスト）と`_fire_reasoning_delta`（推論） |
| `on_interim_message` | `agent/stream_delivery.py` `StreamDeliveryMixin._emit_interim_assistant_message` |
| `pre_verify` | `hermes_cli/plugins.py` `get_pre_verify_continue_message`。`agent/turn_stop_gates.py` `_pre_verify_nudge`から呼び出される |
| `pre_api_request` | `agent/turn_api_request.py` `_fire_pre_api_request_hook` |
| `post_api_request` | `agent/turn_response_intake.py` `_fire_post_api_request_hook` |
| `api_request_error` | `agent/api_request_hooks.py` `ApiRequestHooksMixin._invoke_api_request_error_hook` |
| `pre_auxiliary_call` / `post_auxiliary_call` | `agent/auxiliary_hooks.py` `_AuxCallHooks.pre` / `_AuxCallHooks.post` |
| `transform_api_error_classification` | `hermes_cli/plugins.py` `get_plugin_error_classification`。`agent/error_classifier.py` `_plugin_verdict`から呼び出される |
| `on_session_start` | `agent/conversation_loop.py` `_restore_or_build_system_prompt` |
| `on_session_end` | `agent/turn_finalizer.py` `finalize_turn`。中断による終了: `hermes_cli/cli_shutdown.py` `_invoke_interrupted_session_end`、`tui_gateway/session_lifecycle.py` `_finalize_session` |
| `on_session_finalize` | `hermes_cli/lifecycle.py` `finalize_session`。`hermes_cli/cli_session_mixin.py` `CLISessionMixin._notify_session_boundary`、`hermes_cli/cli_shutdown.py` `_notify_session_finalize`、`tui_gateway/session_lifecycle.py` `_notify_session_boundary`、`gateway/run_shutdown_session_end.py` `GatewaySessionEndMixin._finalize_session_off_loop`から呼び出される |
| `on_session_reset` | `hermes_cli/cli_session_mixin.py` `CLISessionMixin._notify_session_boundary`、`tui_gateway/session_registry.py` `_start_session_services`、`gateway/slash_commands_session.py` `GatewaySessionCommandsMixin._handle_reset_command`、`gateway/run_shutdown_session_end.py` `GatewaySessionEndMixin._reset_session_off_loop` |
| `on_session_delete` | `hermes_state_common.py` `emit_session_deleted` |
| `on_session_open` | `gateway/session_open.py` `_Open._deliver` |
| `on_skill_lifecycle` | `tools/skill_usage.py` `_emit_skill_lifecycle` |
| `subagent_start` | `tools/delegate_tool.py` `_build_child_agent` |
| `subagent_stop` | `tools/delegate_tool_results.py` `_fire_subagent_stop_hooks` |
| `pre_gateway_dispatch` | `gateway/run_inbound_hooks.py` `GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook`。webhookとリレーの受信: `gateway/platforms/webhook_ingress.py` `_admit_once` |
| `post_gateway_admission` | `gateway/run_inbound_consumer.py` `run_post_admission_hook`、`gateway/platforms/webhook_ingress.py` `_admit_once` |
| `agent_loop_stopped` | `gateway/run_agent_cache.py` `GatewayAgentCacheMixin._interrupt_and_clear_session`、`tui_gateway/session_lifecycle.py` `_interrupt_session_turn` |
| `pre_approval_request` / `post_approval_response` | `tools/approval_context.py` `_fire_approval_hook`。`tools/approval.py`、`tools/approval_prompt.py`、`tools/approval_gateway_wait.py`、`tools/approval_smart.py`、`tools/file_tools_write_guards.py`から呼び出される |
| `on_human_input_request` / `on_human_input_resolved` | `tools/human_input_hooks.py` `human_input_request` |
| `on_room_member_activity` | `tui_gateway/hosted_room_member_activity.py` `emit_room_member_activity` |
| `pre_transcription` | `tools/transcription_command.py` `_apply_pre_transcription_hook` |
| `kanban_task_claimed` | `hermes_cli/kanban_db.py` `claim_task`（`_fire_task_hook`経由） |
| `kanban_task_completed` | `hermes_cli/kanban_db.py` `complete_task`（`_fire_task_hook`経由）、`hermes_cli/kanban_swarm.py` `create_swarm` |
| `kanban_task_blocked` | `hermes_cli/kanban_db.py` `block_task`（`_fire_task_hook`経由） |
| `on_kanban_worker_spawned` | `hermes_cli/kanban_db.py` `_fire_worker_spawned_hook` |
| `on_kanban_worker_exited` | `hermes_cli/kanban_db_dispatch.py` `detect_crashed_workers` |
| `on_kanban_worker_stale_claim` | `hermes_cli/kanban_db.py` `release_stale_claims` |
| `on_kanban_task_updated` | `hermes_cli/kanban_db.py` `notify_task_updated` |
| `on_kanban_dispatch_tick` | `hermes_cli/kanban_db.py` `_fire_dispatch_tick_hook` |
| `gateway_platform_event` | `gateway/run_adapters.py` `GatewayAdapterLifecycleMixin._handle_gateway_platform_event`（イベントはTelegramとDiscordのアダプターから来る） |
| `pre_command` | `hermes_cli/plugins.py` `fire_pre_command_hook`。`cli.py` `HermesCLI.process_command`、`gateway/run_inbound_hooks.py` `GatewayInboundHooksMixin._hm_command_hooks`、`gateway/session_commands.py`から呼び出される |
| `post_compaction` | `agent/compaction_events.py` `fire_post_compaction` |
| `post_curator_run` | `agent/curator.py` `_notify_post_curator_run` |

### `pre_llm_call`のコンテキストインジェクション {#pre_llm_call-context-injection}

`pre_llm_call`コールバックが`"context"`キーを持つdict（または素の文字列）を返すと、Hermesはそのテキストを**現在のターンのユーザーメッセージ**にインジェクトします。これは、メモリプラグイン、RAG連携、ガードレール、そしてモデルに追加コンテキストを提供する必要があるあらゆるプラグインのための仕組みです。

#### 戻り値の形式 {#return-format}

```python
# Dict with context key
return {"context": "Recalled memories:\n- User prefers dark mode\n- Last project: hermes-agent"}

# Plain string (equivalent to the dict form above)
return "Recalled memories:\n- User prefers dark mode"

# Block the turn before it reaches the model (DLP / policy plugins)
return {"action": "block", "message": "Blocked by policy: prompt contains a customer ID."}

# Return None or don't return → no injection (observer-only)
return None
```

`"context"`キーを持つNoneでない空でない戻り値（または素の空でない文字列）はすべて収集され、現在のターンのユーザーメッセージに追加されます。

`{"action": "block", "message": "..."}`を返すと、**推論リクエストの前に**ターンが拒否されます。メッセージはモデルの返信の代わりにユーザーに表示され、アシスタントの返信として保存されます（そのためトランスクリプトは通常の`user → assistant`のペアを保ち、`--resume`も動作します）。ターンはモデル呼び出しを一切行わず、結果は`turn_exit_reason: "blocked_by_plugin_pre_llm_call"`を持ちます。最初のブロックが優先され、メッセージのないブロックは無視されます。

ブロックされたプロンプトは`[Message withheld: a pre_llm_call policy hook blocked it.]`として保存されます。後続のターンがリプレイし、コンパクションのサマリー、`session_search`、外部メモリプロバイダー、バックグラウンドのメモリレビューが読み取るのはこのプレースホルダーです。そのため、プロンプトがこのターンでも後のターンでもモデルに届くことはありません。ユーザーが入力した内容は表示用メタデータとしてのみ保持されるため、CLIの`--resume`の要約、TUI、デスクトップアプリ、ダッシュボードには引き続き表示されます。フックはターン開始時のコンパクションの前に実行され、ブロックされたターンではコンパクション、タイトル付け、外部メモリのプリフェッチ/同期がスキップされます。

ブロックは、コールバックがそれを返した場合にのみ適用されます。例外を発生させたコールバックや、`plugins.hook_callback_timeout`（デフォルトは30秒）を超えて実行されたコールバックは、ログに記録されてスキップされ、ターンはそのまま進みます。したがって、ポリシーフックは自身のエラーをキャッチし、判断できない場合はブロックを返すべきです。モデルのメタデータの検索（新しいセッションでの`/models`のコンテキスト長プローブなど）は、引き続きフックの前に実行される場合がありますが、それらはメッセージの内容を含みません。

#### サイズ超過コンテキストの退避 {#oversized-context-spill}

フックごとのコンテキストは、デフォルトで`10,000`文字に制限されています。上限を超えた部分は`$HERMES_HOME/hook_outputs/<session_id>/<uuid>.txt`に書き出され、先頭/末尾のプレビューと保存先パスに置き換えられます。モデルは本当に必要な場合、`read_file`や`terminal`で完全な内容を読むことができます。これにより、暴走したプラグインが後続のすべてのターンのプロンプトを肥大化させ、プロンプトキャッシュのプレフィックスを台無しにするのを防ぎます。`config.yaml`で調整できます。

```yaml
hooks:
  output_spill:
    enabled: true          # default: true
    max_chars: 10000       # default; set higher to opt out of spilling
    preview_head: 500      # chars shown at the top of the preview
    preview_tail: 500      # chars shown at the bottom of the preview
    # directory: null      # default: $HERMES_HOME/hook_outputs
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
            return None  # nothing to inject

        text = "Recalled context from previous sessions:\n"
        text += "\n".join(f"- {m['text']}" for m in memories)
        return {"context": text}
    except Exception:
        return None  # fail silently, don't break the agent

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
    # No return → no injection

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
    return None  # leave the call unchanged

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

### 各セッションのコマンドに独自の環境変数を与える {#give-each-sessions-commands-its-own-environment-variables}

ツールは、セッションを提供するプロセス内で実行されます。共有ゲートウェイの場合、それはゲートウェイデーモンであるため、会話を表示しているターミナルやペインの変数は、エージェントが実行するコマンドの環境には含まれません。`ctx.register_session_env(names, resolver)`は、コマンドを変更することなくこのギャップを埋めます。

```python
def pane_env(session_id):
    pane = find_pane_showing(session_id)  # your plugin's own lookup
    if pane is None:
        return {}  # the commands keep the gateway's own values
    return {"MY_PANE": pane.id, "MY_PANE_SOCKET": pane.socket or None}

def register(ctx):
    ctx.register_session_env(["MY_PANE", "MY_PANE_SOCKET"], pane_env)
```

- Hermesは、セッションのためにローカルプロセスを起動するたびに`resolver(session_id)`を呼び出します。対象は、フォアグラウンドまたはバックグラウンドの`terminal`コマンド、`execute_code`からの`terminal()`呼び出し、そしてセッションの`HERMES_SESSION_*`変数を受け取るその他のサブプロセスです。値はその時点で読み取られるため、変更（会話が別のペインに移動するなど）は次のコマンドに適用されます。
- `session_id`は、そのプロセスが実行されるセッションです。サブエージェントのコマンドはサブエージェントのIDを受け取ります。親の値を適用する必要がある場合は、リゾルバー内でそれを親セッションにマッピングしてください。
- 宣言した名前から文字列へのマップを返してください。値が`None`の場合、その名前はプロセス環境から削除されます。省略した名前はゲートウェイ自身の値を保持します。宣言していない名前は無視されます。
- 宣言された名前は共有シェルのスナップショットに保存されることがないため、あるセッションの値が別のセッションの後続コマンドに届くことはなく、これらの名前の`export`が次のコマンドに引き継がれることもありません。これはプラグイン自身のプロファイルにのみ適用されます。同じゲートウェイが提供する別のプロファイルは、その名前について独自のexportを保持します。
- 各名前は、プロファイルごとに1つのプラグインが持ちます。同じプロファイルで読み込まれた別のプラグインがすでに宣言している名前を宣言すると、そのプラグインの名前を示す`ValueError`が発生します。プラグインをアンロードするとその名前は解放され、同じプラグインをリロードすると再び宣言されます。自分のプラグインは、1つの名前を複数回の呼び出しで宣言できます。後の登録の値が優先され、それを破棄すると前の登録の値が戻ります。
- 名前は有効な環境変数名でなければなりません。`HERMES_*`の名前は拒否されます（コアに属するため）。サブプロセスのスクラブで削除されるプロバイダーやHermesの認証情報の名前も、あなたの値から削除されます。
- 例外を発生させたリゾルバーはログに記録され、何も提供しません。コマンドは引き続き実行されます。すべての起動時に実行されるため、高速かつローカルに保ってください。
- 値はローカルのターミナルバックエンドにのみ適用されます。リモートバックエンド（Docker、SSH、Modal、その他のサンドボックス）はそれらを受け取りません。

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

**メモリプロバイダープラグイン**は、代わりに規約ベースのアプローチを使います。プラグインの`cli.py`ファイルに`register_cli(subparser)`関数を追加してください。メモリプラグインの検出システムが自動的に見つけるため、`ctx.register_cli_command()`の呼び出しは不要です。詳細は[Memory Provider Plugin guide](/docs/developer-guide/memory-provider-plugin#adding-cli-commands)を参照してください。

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
| `handler` | `Callable[[str], str \| None]` | 生の引数文字列と、宣言した呼び出し元セッションのキーワード（後述）とともに呼び出される。`async`でもよい。 |
| `description` | `str` | `/help`、オートコンプリート、Telegramボットメニューに表示される |

**`register_cli_command()`との主な違い:**

| | `register_command()` | `register_cli_command()` |
|---|---|---|
| 呼び出し方 | セッション内で`/name` | ターミナルで`hermes name` |
| 動作する場所 | CLIセッション、Telegram、Discordなど | ターミナルのみ |
| ハンドラが受け取るもの | 生の引数文字列（+ オプトインのセッションコンテキスト） | argparseの`Namespace` |
| ユースケース | 診断、ステータス、クイックアクション | 複雑なサブコマンドツリー、セットアップウィザード |

**衝突保護:** プラグインが組み込みコマンド（`help`、`model`、`new`など）と衝突する名前を登録しようとした場合、登録は静かに拒否され、ログに警告が出ます。組み込みコマンドが常に優先されます。

**呼び出し元セッションのコンテキスト:** ハンドラは、以下のキーワードパラメータのいずれか（またはすべてを受け取る`**kwargs`）を宣言することで、どのチャットとセッションから呼び出されたかを知ることができます。何も宣言しないハンドラは、1引数の`handler(raw_args)`呼び出しのままなので、既存のプラグインは影響を受けません。

| キーワード | 値 |
|---|---|
| `session_id` | 呼び出し元セッションのID（後に`on_session_reset`が`old_session_id`として報告するもの）。チャットにまだセッションがない場合は空。 |
| `session_key` | 呼び出し元チャットのルーティングキー。 |
| `platform` | コマンドがどこから来たか: `telegram`、`discord`、…、`local`（ゲートウェイが所有するCLI/デスクトップのセッション）、`cli`、`tui`、`desktop`。 |
| `source` | ゲートウェイがコマンドをディスパッチした場合は、アダプターから導出された`gateway.session.SessionSource`（`user_id`、`chat_id`、`chat_type`、`thread_id`、…）、それ以外は`None`。 |

すべてのサーフェス（メッセージングプラットフォーム、クラシックCLI、TUI、デスクトップ）がこれらを渡します。同期ハンドラはゲートウェイのエグゼキューターで実行され、非同期ハンドラはこれまでどおりawaitされます。

```python
def _handle_whoami(raw_args: str, *, session_id: str = "", platform: str = "", source=None) -> str:
    user = source.user_id if source is not None else "local user"
    return f"{user} on {platform}, session {session_id or '(none yet)'}"

def register(ctx):
    ctx.register_command("whoami", handler=_handle_whoami, description="Show the calling session")
```

`source`はトランスポート上の識別情報であり、認可の付与ではありません。認可は安定したID（`source.user_id`、`source.chat_id`）に基づいて行い、表示名やコマンドのテキストに基づいてはいけません。また、機密性の高いコマンドが`source=None`を受け取った場合はフェイルクローズしてください。ハンドラの例外はそのまま伝播します。Hermesが少ない引数で呼び出しを再試行することはありません。

**ハンドラが実行される場所:** スラッシュコマンドハンドラは、セッションを所有するゲートウェイプロセス内で実行されます。ユーザーがクラシックな`hermes chat` CLIでコマンドを入力した場合も同様です。ハンドラはユーザーのターミナルやディレクトリで実行されているわけではないため、次の点に注意してください。

- 表示するテキストを返してください。`print()`はゲートウェイのstdoutに書き込まれ、ユーザーには表示されません。
- セッションの作業ディレクトリは、ゲートウェイ自身のディレクトリである`os.getcwd()`ではなく、`agent.runtime_cwd.resolve_agent_cwd()`から取得してください。相対パスはそれを基準に解決します。
- セッションの識別情報は、`os.environ`ではなく、`get_session_env("HERMES_SESSION_PLATFORM")`、`get_session_env("HERMES_SESSION_CHAT_ID")`、その他の`HERMES_SESSION_*`の名前で読み取ってください。関数自体は`from gateway.session_context import get_session_env`でインポートしてください。その名前が公開の継ぎ目であり、`hermes plugins validate`は`import gateway.session_context`をプライベートインポートとして報告します。
- 「このターンを誰が依頼したか」（認可、来歴）については、`HERMES_SESSION_ACTOR_ID` / `HERMES_SESSION_ACTOR_NAME`を読み取ってください。ターンに単一のアクターがいない場合（別のメンバーのメッセージが合流した場合）は、プロセス内でもターミナルや`execute_code`の子プロセスでも、どちらも空になります。そのため、空は特定の誰でもないものとして扱ってください。`HERMES_SESSION_USER_ID`はセッションのルートであり、共有チャット（Telegramのobserveモード）では空になります。そのため、そこから配信先を再構築するコードは共有の会話にとどまります。

```python
from agent.runtime_cwd import resolve_agent_cwd
from gateway.session_context import get_session_env

def _handle_todo(raw_args: str) -> str:
    todo = resolve_agent_cwd() / "TODO.md"  # the session's folder, not the gateway's
    return todo.read_text() if todo.exists() else "No TODO.md here."

def _handle_where(raw_args: str) -> str:
    platform = get_session_env("HERMES_SESSION_PLATFORM")  # unbound: falls back to os.environ, then ""
    return f"{platform or 'local'} chat {get_session_env('HERMES_SESSION_CHAT_ID') or '(none)'}"
```

### ユーザーに選択肢を選んでもらう {#ask-the-user-to-pick-an-option}

同期のスラッシュコマンドハンドラは、`ctx.ask_choice(question, choices)`で、コマンドを実行したユーザーに選択肢を1つ選んでもらうことができます。ユーザーが回答するまでブロックし、選ばれた選択肢の`value`、または`None`を返します。各サーフェスは、エージェントの`clarify`の質問と同じ方法でこれを表示します。

| サーフェス | ユーザーに表示されるもの |
|---|---|
| Telegram、Discord、Slack | 選択肢ごとに1つのボタン（と「Other」）を持つカード |
| その他のメッセージングプラットフォーム | 番号付きリスト。ユーザーは番号またはラベルで返信する |
| クラシックCLI、Ink TUI、デスクトップ | clarifyピッカー |

選択肢は、文字列（ラベル兼値）またはdict `{"value", "label", "description"}`です。descriptionはラベルの後に表示されます。値またはラベルを（大文字小文字を問わず）入力しても、その選択肢を選べます。`ask_choice`は、ユーザーがキャンセルした場合、選択肢でないものを入力した場合、clarifyのタイムアウト（`agent.clarify_timeout`）が経過した場合に`None`を返します。また、質問できる手段がない場合（フック、cronジョブ、非同期ハンドラ。ブロックすると回答を配信するイベントループが停止してしまうため）にも`None`を返します。その場合、ハンドラはテキストでの使い方にフォールバックします。

```python
_MODES = [
    {"value": "default", "label": "Default", "description": "delegation.model / parent model"},
    {"value": "model_map", "label": "Model routes", "description": "per-task model map"},
    {"value": "profile_routes", "label": "Profiles", "description": "delegate to other profiles"},
]

def register(ctx):
    def delegation(raw_args: str) -> str:
        mode = raw_args.strip() or ctx.ask_choice("Delegation mode", _MODES)
        if mode not in {m["value"] for m in _MODES}:
            return "Usage: /delegation default|model_map|profile_routes"
        ctx.set_config("mode", mode)
        return f"Delegation mode: {mode}"

    ctx.register_command("delegation", delegation, description="Switch the delegation mode",
                         args_hint="[mode]")
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
        # Runs in the worker process; ctx._cli_ref is None here.
        ctx.dispatch_tool("kanban_comment", {
            "task_id": task_id,
            "comment": f"[{ctx.profile_name}] auto-noted block: {reason}",
        })
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

完全な`hermes <subcommand>`（例: `hermes kanban show`）を実行するには、`ctx.dispatch_tool("terminal", {"command": "hermes kanban show ..."})`で`terminal`ツールを使ってシェルに処理を委ねてください — ヘッドレスなワーカーセッション向けのプロセス内スラッシュコマンドブリッジはなく、フックからHermesを駆動するためのサポートされた方法はツールです。

### 独自のフロントエンドと通信する: プラグインRPCメソッドとセッションイベント {#talk-to-your-own-frontend-plugin-rpc-methods-and-session-events}

ゲートウェイのJSON-RPCクライアントであるフロントエンド（共有ゲートウェイにアタッチされたターミナルフロントエンド、`/api/ws`上のカスタムWebクライアント）は、自身のPython側を呼び出すことができ、Python側は1つのセッションを表示しているクライアントにイベントをプッシュできます。

```python
def register(ctx):
    def pane_state(params, *, session_id, **kwargs):
        # session_id is the session the call named (None for a sessionless call), already authorized.
        state = load_state(session_id)
        ctx.emit_to_session(session_id, "pane.changed", {"rows": len(state)})
        return {"rows": state}

    ctx.register_rpc("pane.state", pane_state)
```

クライアントは任意のJSONパラメータ（`session_id`でセッションを指定）で`plugin.<plugin_id>.pane.state`を呼び出し、型が`plugin.<plugin_id>.pane.changed`のイベントフレームを待ち受けます。

```json
{"jsonrpc": "2.0", "id": 7, "method": "plugin.my-plugin.pane.state", "params": {"session_id": "local-…", "page": 2}}
```

- **名前は強制的にあなたの名前空間に入れられます**（`<plugin_id>`は`ctx.plugin_id`）。`ctx.register_rpc(name, handler)`は`plugin.<plugin_id>.<name>`を提供し、`ctx.emit_to_session(session_id, event, payload=None)`は`plugin.<plugin_id>.<event>`を送信します。`name`と`event`は、`broadcast_plugin_event`と同様に、素のドット区切りの名前（各セグメントは`[A-Za-z0-9_-]`）です。IDが`[a-z0-9_-]{1,64}`でないプラグイン（`image_gen/x`のようなカテゴリプラグイン）は登録できず、呼び出しは`ValueError`を発生させます。
- **ハンドラ**は、ゲートウェイのループの外で、所有するプロファイルのスコープ内で`handler(params, session_id=...)`として呼び出されます。`async def`でもかまいません。その場合は、プラグインのスラッシュコマンドと同様に、ゲートウェイのループ上ではなく、そのワーカースレッド内の独自のイベントループで実行されます。`**kwargs`を受け取ってください。今後のリリースでより多くのコンテキストが渡される可能性があります。戻り値はJSONエンコード可能でなければならず、それが呼び出しの`result`になります。例外、またはJSONでエンコードできない値の場合、呼び出しにはエラー`4001`、理由`plugin_rpc_failed`、そして`error.data.detail`に例外のテキストが返されます。呼び出しは戻るまでそのクライアントのリクエストレーンを占有するため、短く保ってください。長い処理はスレッドで開始し、`emit_to_session`を通じて報告します。
- **アンロードするとメソッドは削除されます。** プラグインが無効化または削除されると、そのメソッドは未知のメソッドと同様に`-32601`を返します。
- **セッションイベントはそのセッションの閲覧者にのみ送られます**。送られるのはそのセッションのイベントストリーム上で、現在アタッチしているクライアントに届き、再接続したクライアントはそれらをリプレイします。`session_id`は会話のIDでも、そのエージェントの現在のID（圧縮によってエージェントは新しいIDを得ます）でもかまいません。そのため、ツールやフックは受け取った`session_id`をそのまま渡せます。`payload`はJSONのdictです。呼び出しは、ライブの閲覧者がフレームを受け取った場合は`True`を、閲覧しているクライアントがいない場合や、このプロセスがそのようなセッションを保持していない場合は`False`を返します。`ctx.emit_to_session`は、自分で開始したスレッドからであっても、プラグイン自身のプロファイル内でセッションを検索します。`hermes_cli.plugin_events.emit_plugin_session_event(plugin_id, session_id, event, payload)`は、`ctx`を持たないコード向けの同じ呼び出しです。これは呼び出し元のプロファイルスコープを使うため、複数のプロファイルを提供するゲートウェイ上で自分のスレッドから呼ぶ場合は、`ctx.emit_to_session`を使ってください。
- **`plugins.isolation: host`:** ハンドラはプラグインホストプロセス内で実行され、両方のemit呼び出しはゲートウェイプロセスに転送され、ホストのプロファイル内で検索されます。そこでは、`emit_plugin_session_event`は呼び出しを行ったコードのプラグインとして送信されます。`plugin_id`がそのプラグイン自身のものでない場合や、コードがその後アンロードされた読み込みに由来する場合（リロードをまたいで実行され続けているタスクも含む）は、`False`を返して何も送信しません。また、自分で開始したスレッドからの場合も同様になることがあるため（ホストはそれがどのプラグインに属するか分からない場合があります）、そこでは`ctx.emit_to_session`を使ってください。プロセス内では、ヘルパーは`plugin_id`を与えられたとおりに受け取ります。
- **動作する場所:** セッションを所有するゲートウェイ（そのセッションオーソリティ。アタッチされたフロントエンドが使う共有の`hermes gateway run`ゲートウェイなど）上です。それを持たないバックエンドは、メソッドに`-32601`を返します。それを持たないプロセス（`hermes chat`、cron、`dashboard.turn_isolation`の計算用子プロセス）では、`emit_to_session`は`False`を返します。
- **まず検出してください。** プラグインメソッドを提供するオーナーは、`runtime.describe`の`capabilities`に`plugin-rpc-v1`を列挙します。それがある場合、`-32601`はそのプラグインメソッドが読み込まれていない（プラグインが無効、未インストール、またはそのようなメソッドがない）ことを意味します。`plugin-rpc-v1`がない場合、クライアントはプラグインRPCがサポートされていると仮定してはいけません（古いオーナーはケイパビリティを列挙せずに提供している可能性があります）。コントラクトは、名前空間を`plugin.{plugin_id}.{name}`として一度だけ宣言します（生成されたTypeScriptでは`CanonicalRpcNamespaces`、OpenRPCドキュメントでは`x-canonical-namespaces`）。paramsはオプションの`session_id`を持つオープンなオブジェクトで、resultはハンドラが返す任意のJSONです。

**セキュリティ。** プラグインメソッドには、そのゲートウェイ上のプラグインのスラッシュコマンドと同じゲートが適用されます。接続には`session:control`ケイパビリティが必要で、`session_id`を指定する呼び出しは、そのセッションの認可（正しいプロファイル、ローカルセッションの所有者、置き換えられた会話は読み取り専用）を通過しなければなりません。それ以外では、ゲートウェイが認証した任意のクライアントが、任意のパラメータで任意のプラグインメソッドを呼び出せます。Hermesはメソッドごとに呼び出し元をチェックしません（`SECURITY.md` §2.6、ルール4）。プラグインメソッドはプラグインツールと同様に扱い、パラメータを検証し、プラグイン自身のデータとアクションを超えるものを決して公開しないでください。

### ユーザーの自己学習のペースに従う（`review_signals`） {#follow-the-users-self-learning-cadence-review_signals}

成功したターンのたびに、`post_llm_call`は`review_signals`を持ちます。これは、Hermes自身のバックグラウンドのメモリ/スキルレビューの背後にあるカウンター（`skills.creation_nudge_interval`、`memory.nudge_interval`）と、このターンでそのレビューが期限を迎えていて有効かどうかです。ターン後に独自の処理（例えば、ターンが触れたスキルの評価）を持つプラグインは、ツール呼び出しを自分で数える代わりに、ユーザーがすでに調整したのと同じペースで実行できます。

```python
def register(ctx):
    def on_post_llm_call(review_signals=None, **kwargs):
        if not review_signals or not review_signals["background_review_enabled"]:
            return  # cron run, subagent, or the user turned background review off
        if review_signals["skill_review_due"]:
            ...  # the same turn Hermes reviews skills on

    ctx.register_hook("post_llm_call", on_post_llm_call)
```

フックはHermesが判断する前に発火し、期限フラグはその判断が使う値です。キーとセマンティクス: [`post_llm_call`](../../user-guide/features/hooks.md#review_signals)。Hermes自身のレビューと同じ方法（範囲を絞ったツールリストを持つ、セッションのエージェントのバックグラウンドコピー）でその処理を行うには、このシグナルを[`ctx.fork_session`](/docs/developer-guide/plugin-llm-access#forking-the-sessions-own-agent-ctxfork_session)と組み合わせてください。

### Hermesがアイドル状態のときにジョブを実行する（`ctx.register_idle_task`） {#run-a-job-when-hermes-is-idle-ctxregister_idle_task}

実験的機能: シグネチャは今後のリリースで変更される可能性があります（[詳細](/docs/developer-guide/plugin-llm-access#idle-tasks-ctxregister_idle_task)）。

一部の処理には、ぶら下がるべきターンがありません。1日に1回プラグインのデータを再スコアリングする、クリーンアップを提案する、キャッシュをコンパクションする、といったものです。`ctx.register_idle_task`は、スキルの[キュレーター](../../user-guide/features/curator.md)と同じスケジューラー上で、同じ場所（提供されるすべてのプロファイルに対するゲートウェイのハウスキーピング、CLI/TUIの起動時、`hermes serve`のメンテナンスティック）からそれを実行します。

```python
def register(ctx):
    def suggest(run):
        # run.name, run.idle_for_seconds, run.last_run_at (ISO or None), run.run_count
        candidates = ctx.state.get("candidates", [])
        if not candidates:
            return None
        result = ctx.llm.complete(
            messages=[{"role": "user", "content": f"Rank these for promotion: {candidates}"}],
            task="suggest",  # a task this plugin registered with ctx.register_auxiliary_task
        )
        ctx.state.set("ranking", result.text)
        return f"ranked {len(candidates)} skills"  # optional summary, shown by `hermes plugins idle-tasks`

    ctx.register_idle_task("suggest", suggest, interval_hours=24, min_idle_hours=2)
```

`register_idle_task(name, fn, *, interval_hours, min_idle_hours=2, first_run="defer")`は`PluginRegistration`を返します（破棄するとポーリングが停止します）。タスクは次の場合に実行されます。

- ホストが`min_idle_hours`の間アイドル状態である（ゲートウェイのハウスキーピングとCLIの起動は完全なアイドルとみなされ、`hermes serve`はチャットの非アクティブ時間を測定します）。
- 前回の実行から`interval_hours`（下限は1）が経過している。最初のポーリングでは、キュレーターと同様に「現在時刻」を記録して1インターバル待つため、インストールや更新の直後にジョブが実行されることはありません。`first_run="immediate"`を指定すると、代わりに最初のアイドルポーリングで実行されます。
- 一時停止されていない（`hermes plugins idle-tasks <plugin> --pause <task>`）。
- このプロセスでも別のプロセスでも、そのタスクの他の実行が進行中でない。

`fn`は、それを登録したプロファイル内の独自のスレッドで実行され、ホストはそれを待ちません。ポーリングはすぐに戻り、実行中の間はそのタスクの2回目の実行は開始されません。1回の実行は短く、繰り返しても安全なものにしてください。ゲートウェイが停止したとき、まだ進行中の実行は何も記録せず、後で再度実行されます（[シャットダウンのコントラクト](/docs/developer-guide/plugin-llm-access#idle-tasks-ctxregister_idle_task)）。例外はログに記録され、`last_status: error`として記録され、ホストに届くことはありません。状態はプラグインのデータディレクトリ`<data dir>/idle-tasks/<name>.json`（`last_run_at`、`run_count`、`paused`、`last_summary`、`last_status`、`last_error`）に保存されるため、各プロファイルが独自のコピーをスケジュールします。

これをゲートするケイパビリティはありません。スケジューラーはあなた自身の関数を呼び出すだけです。その関数が行う特権的な処理は、それぞれ独自のゲートを通過します（モデル呼び出しは[`ctx.llm`の信頼ゲート](/docs/developer-guide/plugin-llm-access#trust-gate)を通過する、など）。

### エージェントのブラウザに生のCDPコマンドを送る {#send-raw-cdp-commands-to-the-agents-browser}

ブラウザツールより低いレベルでエージェントのブラウザを操作しなければならないプラグインは、`SUPERVISOR_REGISTRY.capture(task_id)`で、タスクのライブなスーパーバイザー接続に固定されたCDPハンドルを取得し、`call(method, params=None, *, session_id=None, timeout=10.0)`でコマンドを送ることができます。このハンドルは、その接続が失われると`CapturedCDPInvalid`を発生させ、より新しい接続に対象を切り替えることはありません。カタログが拒否する`CDPSupervisor`のプライベート属性の読み書きの代わりに、これを使ってください。これは信頼されたプロセス内のトランスポートにすぎず、オリジン、同意、所有権のチェックは一切行いません。完全なコントラクト: [Browser CDP Supervisor](/docs/developer-guide/browser-supervisor#trusted-plugin-cdp-access)。

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

`ctx.register_automation_blueprint(key, *, title, description, schedule_template, prompt_template, category="general", slots=(), deliver_default="origin", skills=(), tags=())`は、空欄を埋めるだけで使える自動化を[Automation Blueprintsカタログ](/docs/reference/automation-blueprints-catalog)に追加します。それは`/blueprint`（CLI、TUI、メッセンジャー）、ダッシュボードのBlueprintsタブ、デスクトップのCronページで、組み込みのものと並んで、プラグイン名のラベル付きで表示されます。引数は組み込みの`AutomationBlueprint`（`cron/blueprint_catalog.py`）のフィールドで、各スロットは`BlueprintSlot`のフィールドからなるdictです。

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
        # ack within 3 seconds — slack_bolt requirement.
        await ack()
        # body["channel"]["id"], body["user"]["id"], body["message"]["ts"]
        # action["action_id"], action["value"]
        sweep_id = (action.get("value") or "").split("|", 1)[-1]
        # ...do the deterministic work, then post a follow-up.

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
        # native: the platform's client/app object (see table below)
        # adapter: the platform adapter instance (treat as read-only)
        # Import platform SDKs HERE so register() works without them.
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
            # ...handle "myplugin:*" callbacks

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
            ...  # e.g. reaction-based voting / moderation

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

**完全なガイド:** [Model Provider Plugins](/docs/developer-guide/model-provider-plugin) — フィールドリファレンス、オーバーライド可能なフック（`prepare_messages`、`build_extra_body`、`build_api_kwargs_extras`、`fetch_models`）、api_modeの選択、認証タイプ、テスト。

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
    from agent.secret_scope import get_secret
    return bool(get_secret("MYPLATFORM_TOKEN"))

def _env_enablement():
    from agent.secret_scope import get_secret
    tok = (get_secret("MYPLATFORM_TOKEN") or "").strip()
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
        # Auto-populate PlatformConfig.extra from env so env-only setups
        # show up in `hermes gateway status` without SDK instantiation.
        env_enablement_fn=_env_enablement,
        # Opt in to cron delivery: `deliver=myplatform` routes to this var.
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
        from agent.secret_scope import get_secret
        return bool(get_secret("MY_MEMORY_API_KEY"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def sync_turn(self, user_content, assistant_content, *,
                  session_id="", messages=None) -> None:
        ...

    def prefetch(self, query, *, session_id="") -> str:
        ...

    def get_tool_schemas(self) -> list[dict]:
        return []   # required @abstractmethod — see full guide

def register(ctx):
    ctx.register_memory_provider(MyMemoryProvider())
```

メモリプロバイダーは単一選択です — 一度にアクティブになれるのは1つだけで、`config.yaml`の`memory.provider`で選択します。

プロバイダーが汎用プラグインとしても読み込まれる場合、そのライフサイクルフックは汎用の検出処理が所有します。メモリローダーがフックを提供するのは、同じプラグインソースが汎用の検出処理を通じて正常に読み込まれるまでのフォールバックとしてのみです。プロバイダーが繰り返し読み込まれると、フォールバックのフックグループが置き換えられます。グループ内の異なるコールバックは保持されます。これは、異なるプラグインソースからのフックの重複を排除したり、プロバイダーの有効化を変更したりするものではありません。

**完全なガイド:** [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin) — 完全な`MemoryProvider` ABC、スレッディングコントラクト、プロファイル分離、`cli.py`経由のCLIコマンド登録。

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

**完全なガイド:** [Context Engine Plugins](/docs/developer-guide/context-engine-plugin)。

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
        # returns success_response(...) / error_response(...)
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

**完全なガイド:** [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin) — 完全な`ImageGenProvider` ABC、`list_models()` / `get_setup_schema()`メタデータ、`success_response()`/`error_response()`ヘルパー、base64 vs URL出力、ユーザーによる上書き、pip配布。

**リファレンス例:** `plugins/image_gen/openai/`（OpenAI SDK経由のDALL-E / GPT-Image）、`plugins/image_gen/openai-codex/`、`plugins/image_gen/xai/`（Grok画像生成）。

### Computer-useバックエンドプラグイン {#computer-use-backend-plugins}

`computer_use`ツールは、`ComputerUseBackend` ABC（`tools/computer_use/backend.py`）を通じて、ちょうど1つの**ドライバー**とやり取りします。それを選ぶのは`config.yaml`の`computer_use.backend`です。デフォルトは組み込みのcua-driverで、これは`plugins/computer_use/cua/`にある通常のプロバイダーとして同梱されています。別のドライバーは、`~/.hermes/plugins/<name>/`にあり、その`register(ctx)`が`ctx.register_computer_use_provider()`を呼び出すプロバイダープラグインです。ディレクトリ名が、`computer_use.backend`に設定する値になります。computer-useプロバイダーは、メモリプロバイダーやコンテキストエンジンと同様に単一選択です。

```python
# ~/.hermes/plugins/my-driver/__init__.py
from tools.computer_use.backend import (
    ActionResult, CaptureResult, ComputerUseBackend, ComputerUseProvider,
)

class MyBackend(ComputerUseBackend):
    def start(self): ...                    # open the driver session
    def stop(self): ...                     # tear it down (also called at exit)
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
        # A driver that can't do an action says so per call; the tool schema never changes.
        return ActionResult(ok=False, action="set_value", code="unsupported_action",
                            message="my-driver has no accessibility value setter")

class MyDriverProvider(ComputerUseProvider):
    name = "my-driver"
    display_name = "My driver"

    def create_backend(self, *, permission_mode):
        return MyBackend()                  # permission_mode: standard | bounded | unrestricted

    def is_available(self):                 # gates the tool; cheap, no network
        return True

    def doctor(self):                       # optional: `hermes computer-use doctor`
        print("my-driver: ok")
        return 0

def register(ctx):
    ctx.register_computer_use_provider(MyDriverProvider())
```

```yaml
# ~/.hermes/plugins/my-driver/plugin.yaml
name: my-driver
version: 1.0.0
description: Alternative computer-use driver   # shown in the picker
```

`hermes tools` → Computer Useで選択するか（インストール済みのプロバイダーは、CLIとデスクトップのツールセットパネルでcua-driverの下に一覧表示されます）、直接設定します。`plugins.enabled`のエントリは不要です。

```yaml
# config.yaml
computer_use:
  backend: my-driver     # default: cua
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
        # send notification …
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
Nix/NixOSは、明示的にサポートされるインストール方法ではなくなりました（ベストエフォートのみ） — [Nix Setup](/docs/getting-started/nix-setup)を参照してください。このセクションは、すでにNixOS上でデプロイしているユーザーのために残されています。
:::

NixOSユーザーは、エントリポイントを持つ`pyproject.toml`を提供すれば、あなたのプラグインを宣言的にインストールできます。

**エントリポイントプラグイン**（配布に推奨）:
```nix
# User's configuration.nix
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

オーバーレイの使い方や衝突チェックを含む完全なドキュメントについては、[Nix Setup guide](/docs/getting-started/nix-setup#plugins)を参照してください。

## よくある間違い {#common-mistakes}

**ハンドラがJSON文字列を返していない:**
```python
# Wrong — returns a dict
def handler(args, **kwargs):
    return {"result": 42}

# Right — returns a JSON string
def handler(args, **kwargs):
    return json.dumps({"result": 42})
```

**ハンドラのシグネチャに`**kwargs`がない:**
```python
# Works — the dispatcher only forwards the context keywords a signature names
def handler(args):
    ...

# Better — receives every injected context field (task_id, session_id, parent_agent, ...)
def handler(args, **kwargs):
    ...
```

**ハンドラが例外を投げる:**
```python
# Wrong — exception propagates, tool call fails
def handler(args, **kwargs):
    result = 1 / int(args["value"])  # ZeroDivisionError!
    return json.dumps({"result": result})

# Right — catch and return error JSON
def handler(args, **kwargs):
    try:
        result = 1 / int(args.get("value", 0))
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

**スキーマの説明が曖昧すぎる:**
```python
# Bad — model doesn't know when to use it
"description": "Does stuff"

# Good — model knows exactly when and how
"description": "Evaluate a mathematical expression. Use for arithmetic, trig, logarithms. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e."
```
