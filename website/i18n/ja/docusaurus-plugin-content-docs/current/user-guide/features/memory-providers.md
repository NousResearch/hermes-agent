---
sidebar_position: 4
title: "メモリプロバイダー"
description: "外部メモリプロバイダープラグイン — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# メモリプロバイダー

Hermes Agentには、組み込みのMEMORY.mdおよびUSER.mdを超えて、エージェントに永続的でセッションをまたぐ知識を与える2つの外部メモリプロバイダープラグイン（RetainDBとByteRover）が付属しており、さらに多くのもの（Honcho、Hindsight、Supermemory、Holographicなど）が[プラグインカタログ](./plugins.md)から利用できます。一度にアクティブにできる外部プロバイダーは**1つ**だけです — 組み込みメモリは常にそれと並行してアクティブです。

## クイックスタート

```bash
hermes memory setup      # interactive picker + configuration
hermes memory status     # check what's active
hermes memory off        # disable external provider
```

`hermes plugins` → Provider Plugins → Memory Provider からアクティブなメモリプロバイダーを選択することもできます。

または`~/.hermes/config.yaml`で手動設定します。

```yaml
memory:
  provider: retaindb  # or byterover,
                      # or honcho / hindsight / supermemory / mem0 / openviking / holographic
                      # (plugin catalog — run `hermes plugins install <name>` first)
```

## 仕組み

メモリプロバイダーがアクティブな場合、Hermesは自動的に次のことを行います。

1. システムプロンプトに**プロバイダーのコンテキストを注入**する（プロバイダーが知っていること）
2. 各ターンの前に**関連するメモリをプリフェッチ**する（バックグラウンド、非ブロッキング）
3. 各応答の後に会話ターンをプロバイダーに**同期**する
4. **セッション終了時にメモリを抽出**する（それをサポートするプロバイダーの場合）
5. 組み込みメモリへの書き込みを外部プロバイダーに**ミラーリング**する
6. エージェントがメモリを検索、保存、管理できるよう**プロバイダー固有のツールを追加**する

組み込みメモリ（MEMORY.md / USER.md）は、これまでとまったく同じように動作し続けます。外部プロバイダーは追加的なものです。

Hermesがプロバイダーに渡すすべてのもの（ターン、ツール出力を含むトランスクリプト、想起クエリ、セッション終了時と圧縮前のトランスクリプト、ミラーリングされたメモリへの書き込み、委譲の結果、プロバイダー自身のツールの引数）は、まずチャットプラットフォームやcronの配信と同じシークレットのスクラブを通過するため、ツール出力にエコーされたキーはプロバイダーが保存する前にマスクされます。その配信時のスクラブと同様に、これは `security.redact_secrets: false` でも適用されます。認識可能な形を持たない認証情報（任意のパスワード、`key=value` や `Bearer` の文脈の外にある不透明なトークン）は検出されません。ローカルのトランスクリプトには元のテキストが残ります。

## 利用可能なプロバイダー

### Honcho

:::info プラグインカタログ
Honchoは[Plastic Labs](https://github.com/plastic-labs/honcho/tree/main/hermes-plugin-honcho)によってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。以前ツリー内で提供されていたのと同じプロバイダーです: ツール、設定ファイル、`hermes honcho` コマンドは変わっていません。
:::

弁証的推論、セッションスコープのコンテキスト注入、セマンティック検索、永続的な結論を備えた、AIネイティブなセッションをまたぐユーザーモデリング。ベースコンテキストには、ユーザー表現とピアカードに加えてセッションサマリーが含まれるようになり、エージェントは既に議論された内容を把握できます。

| | |
|---|---|
| **最適な用途** | セッションをまたぐコンテキストを持つマルチエージェントシステム、ユーザーとエージェントの整合 |
| **必要なもの** | `hermes plugins install honcho`（`honcho-ai` SDKも一緒にインストールされます）。[APIキー](https://app.honcho.dev) またはセルフホストインスタンス |
| **データストレージ** | Honcho Cloud またはセルフホスト |
| **コスト** | Honchoの料金（クラウド） / 無料（セルフホスト） |

**ツール（5個）:** `honcho_profile`（ピアカードの読み取り/更新）、`honcho_search`（セマンティック検索）、`honcho_context`（セッションコンテキスト — サマリー、表現、カード、メッセージ）、`honcho_reasoning`（LLMで合成）、`honcho_conclude`（結論の作成/削除）

**アーキテクチャ:** 2層のコンテキスト注入 — ベース層（セッションサマリー + 表現 + ピアカード、`contextCadence`でリフレッシュ）に加えて、弁証的な補足（LLM推論、`dialecticCadence`でリフレッシュ）。弁証法は、ベースコンテキストが存在するかどうかに基づいて、コールドスタート用プロンプト（一般的なユーザー事実）とウォーム用プロンプト（セッションスコープのコンテキスト）を自動的に選択します。

**3つの直交する設定ノブ**がコストと深さを独立して制御します。

- `contextCadence` — ベース層がリフレッシュされる頻度（API呼び出しの頻度）
- `dialecticCadence` — 弁証法のLLMが発火する頻度（LLM呼び出しの頻度）
- `dialecticDepth` — 弁証法の呼び出しごとの`.chat()`のパス回数（1〜3、推論の深さ）

自動注入される弁証法は、クエリの長さに応じて推論レベルも調整します（クエリが長いほど推論が深くなり、`reasoningLevelCap` で上限が設定されます）。[クエリ適応型の推論レベル](./honcho.md#query-adaptive-reasoning-level) を参照してください。

**セットアップウィザード:**
```bash
hermes plugins install honcho   # from the plugin catalog
hermes memory setup             # select "honcho" — runs the Honcho-specific post-setup
```

レガシーの`hermes honcho setup`コマンドはまだ動作しますが（現在は`hermes memory setup`にリダイレクトされます）、Honchoがアクティブなメモリプロバイダーとして選択された後にのみ登録されます。

**ヘッドレス / リモートマシン:** ブラウザのないマシン（SSH、リモートVM）でクラウド認証を行うには、ウィザードの認証方式のプロンプトで **device** を選びます。CLIが短いコードと検証用リンクを表示するので、別の任意のマシンのブラウザでリンクを開いて承認すれば、セットアップが完了します — APIキーのコピー&ペーストは不要です。使用可能なローカルブラウザが検出されない場合、ウィザードは自動的にこのオプションをデフォルトにします。

**設定:** `$HERMES_HOME/honcho.json`（プロファイルローカル）または`~/.honcho/config.json`（グローバル）。解決順序: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`。[プラグインのREADME](https://github.com/plastic-labs/honcho/tree/main/hermes-plugin-honcho)と[Honcho統合ガイド](https://docs.honcho.dev/v3/guides/integrations/hermes)を参照してください。

<details>
<summary>完全な設定リファレンス</summary>

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `apiKey` | -- | [app.honcho.dev](https://app.honcho.dev)から取得するAPIキー |
| `baseUrl` | -- | セルフホストHonchoのベースURL |
| `peerName` | -- | ユーザーピアのアイデンティティ |
| `aiPeer` | ホストキー | AIピアのアイデンティティ（プロファイルごとに1つ） |
| `workspace` | ホストキー | 共有ワークスペースID |
| `contextTokens` | `null`（上限なし） | ターンごとに自動注入されるコンテキストのトークン予算。単語境界で切り詰める |
| `contextCadence` | `1` | `context()`API呼び出し間の最小ターン数（ベース層のリフレッシュ） |
| `dialecticCadence` | `2` | `peer.chat()`のLLM呼び出し間の最小ターン数。推奨1〜5。`hybrid`/`context`モードにのみ適用 |
| `dialecticDepth` | `1` | 弁証法の呼び出しごとの`.chat()`のパス回数。1〜3にクランプ。パス0: コールド/ウォームプロンプト、パス1: 自己監査、パス2: 調整 |
| `dialecticDepthLevels` | `null` | パスごとの推論レベルの任意の配列、例: `["minimal", "low", "medium"]`。比例的なデフォルトを上書き |
| `dialecticReasoningLevel` | `'low'` | ベース推論レベル: `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | `true`のとき、モデルはツールパラメータ経由で呼び出しごとに推論レベルを上書きできる |
| `dialecticMaxChars` | `600` | システムプロンプトに注入される弁証法結果の最大文字数 |
| `recallMode` | `'hybrid'` | `hybrid`（自動注入 + ツール）、`context`（注入のみ）、`tools`（ツールのみ） |
| `writeFrequency` | `'async'` | メッセージをフラッシュするタイミング: `async`（バックグラウンドスレッド）、`turn`（同期）、`session`（終了時にバッチ）、または整数N |
| `saveMessages` | `true` | メッセージをHoncho APIに永続化するかどうか |
| `observationMode` | `'directional'` | `directional`（すべてオン）または`unified`（共有プール）。`observation`オブジェクトで上書き |
| `messageMaxChars` | `25000` | メッセージごとの最大文字数（超過するとチャンク化） |
| `dialecticMaxInputChars` | `10000` | `peer.chat()`への弁証法クエリ入力の最大文字数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session`、`global` |
| `pinUserPeer` | `false` | ゲートウェイのみ。`true`のとき、エージェント以外のすべてのゲートウェイユーザーが`peerName`に集約される。このピン留めはすべてのエイリアスより優先される |
| `userPeerAliases` | `{}` | ゲートウェイのみ。ランタイムIDをピアにマッピングする（`{"7654321": "alice"}`）。多対一 |
| `runtimePeerPrefix` | `""` | ゲートウェイのみ。どのエイリアスにも一致しない未知のランタイムIDに名前空間を付ける（`telegram_7654321`） |

</details>

<details>
<summary>最小のhoncho.json（クラウド）</summary>

```json
{
  "apiKey": "your-key-from-app.honcho.dev",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

<details>
<summary>最小のhoncho.json（セルフホスト）</summary>

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

:::tip `hermes honcho` からの移行
以前に`hermes honcho setup`を使っていた場合、あなたの設定とすべてのサーバー側データはそのまま残っています。新しいシステム経由で再有効化するには、セットアップウィザードでもう一度有効化するか、`memory.provider: honcho`を手動で設定してください。
:::

**マルチピアのセットアップ:**

Honchoは会話を、メッセージを交換するピアとしてモデル化します — 1つのユーザーピアと、Hermesプロファイルごとに1つのAIピアがあり、すべてが1つのワークスペースを共有します。ワークスペースは共有環境です。ユーザーピアはプロファイルをまたいでグローバルであり、各AIピアはそれ自体のアイデンティティです。各AIピアは自身の観測から独立した表現 / カードを構築するため、`coder`プロファイルはコード指向のまま、`writer`プロファイルは同じユーザーに対して編集者寄りのままになります。

対応関係:

| 概念 | それが何か |
|---------|-----------|
| **ワークスペース** | 共有環境。1つのワークスペース配下のすべてのHermesプロファイルは同じユーザーアイデンティティを見る。 |
| **ユーザーピア**（`peerName`） | 人間。ワークスペース内のプロファイルをまたいで共有される。 |
| **AIピア**（`aiPeer`） | Hermesプロファイルごとに1つ。ホストキー`hermes`→デフォルト。その他は`hermes.<profile>`。 |
| **観測** | Honchoが誰のメッセージから何をモデル化するかを制御するピアごとのトグル。`directional`（デフォルト、4つすべてオン）または`unified`（単一観測者プール）。 |

### 新しいプロファイル、新規のHonchoピア

```bash
hermes profile create coder --clone
```

`--clone`（`memory.provider: honcho` を引き継ぎます）により、Honchoプラグインは、`aiPeer: "coder"`、共有の`workspace`、継承された`peerName`、`recallMode`、`writeFrequency`、`observation`などを持つ`hermes_coder`ホストブロックを`honcho.json`に作成できます。AIピアは最初のメッセージの前に存在するよう、Honchoで先行して作成されます。`hermes profile rename` では、プラグインがそのホストブロックのキーを付け直し、ピアは保持されます。どちらも、プロファイルフックを備えたバージョンのHonchoプラグインが必要です。

### 既存のプロファイル、Honchoピアのバックフィル

```bash
hermes honcho sync
```

すべてのHermesプロファイルをスキャンし、ホストブロックを持たないプロファイルにホストブロックを作成し、デフォルトの`hermes`ブロックから設定を継承し、新しいAIピアを先行して作成します。冪等です — 既にホストブロックを持つプロファイルはスキップします。`hermes update` はもうこれを自動では行いません: Honchoのセットアップ前に作成したプロファイルに対して一度実行してください。

### プロファイルごとの観測

各ホストブロックは観測設定を独立して上書きできます。例: AIピアがユーザーを観測するが自己モデル化しない、コード重視のプロファイル:

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**観測トグル（ピアごとに1セット）:**

| トグル | 効果 |
|--------|--------|
| `observeMe` | Honchoがこのピアの表現を、そのピア自身のメッセージから構築する |
| `observeOthers` | このピアが他方のピアのメッセージを観測する（ピア間推論に供給される） |

`observationMode`によるプリセット:

- **`"directional"`**（デフォルト） — 4つのフラグすべてオン。完全な相互観測。ピア間の弁証法を有効化。
- **`"unified"`** — ユーザーは`observeMe: true`、AIは`observeOthers: true`、残りはfalse。単一観測者プール。AIはユーザーをモデル化するが自身はモデル化せず、ユーザーピアは自己モデル化のみ。

[Honchoダッシュボード](https://app.honcho.dev)経由で設定されたサーバー側のトグルは、ローカルのデフォルトより優先されます — セッション初期化時に同期されて戻ります。

完全な観測リファレンスについては[Honchoページ](./honcho.md#observation-directional-vs-unified)を参照してください。

### ゲートウェイのアイデンティティマッピング

上記のピアモデルは、すべての会話が`peerName`に解決されるCLI、TUI、デスクトップのセッションを対象としています。[ゲートウェイ](/docs/developer-guide/gateway-internals)は2つ目の軸を追加します: ユーザーはプラットフォーム固有のランタイムID（TelegramのUID、Discordのsnowflake、Slackのユーザー）で到着し、3つのキーが各IDをどのピアに解決するかを決めます。

| キー | 効果 |
|-----|--------|
| `pinUserPeer: true` | エージェント以外のすべてのゲートウェイユーザーが`peerName`に集約されます。ピン留めが最初にチェックされるため、すべてのエイリアスより優先されます — ユーザー側のアイデンティティがどれも独自のピアを必要としない場合にのみ選んでください |
| `userPeerAliases` | 特定のランタイムIDをピアにマッピングします（`{"7654321": "alice"}`）。個別のアイデンティティをルーティングするための場所です — それぞれ独自のピアを持つエージェントも含みます |
| `runtimePeerPrefix` | マッピングされていないランタイムIDに名前空間を付け（`telegram_7654321`）、同じ形のIDを持つプラットフォーム同士が衝突しないようにします |

ゲートウェイの外では、これらのキーは何もしません。`hermes memory setup` は、接続済みのゲートウェイプラットフォームを検出した場合にのみこれらを尋ねます。リゾルバーの順序とセットアップの流れについては[Honchoページ](./honcho.md#gateway-identity-mapping)を参照してください。

<details>
<summary>完全なhoncho.jsonの例（マルチプロファイル）</summary>

```json
{
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "dialecticReasoningLevel": "low",
      "dialecticDynamic": true,
      "dialecticCadence": 2,
      "dialecticDepth": 1,
      "dialecticMaxChars": 600,
      "contextCadence": 1,
      "messageMaxChars": 25000,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "tools",
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    },
    "hermes.writer": {
      "enabled": true,
      "aiPeer": "writer",
      "workspace": "hermes",
      "peerName": "eri"
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```

</details>

[プラグインのREADME](https://github.com/plastic-labs/honcho/tree/main/hermes-plugin-honcho)と[Honcho統合ガイド](https://docs.honcho.dev/v3/guides/integrations/hermes)を参照してください。

#### バンドル版Honchoからのアップグレード {#upgrading-from-the-bundled-honcho}

以前のHermesリリースでは、Honchoはツリー内で提供されていました。プロファイルにまだ `memory.provider: honcho` がある場合、Hermesは次回の起動時または `hermes update` の実行時にカタログプラグインを自動的にインストールします — `~/.honcho/config.json`（または `$HERMES_HOME/honcho.json`）、ホストブロック、ピア、セッションのマッピングはこれまでとまったく同じように読み込まれるため、メモリが失われることはありません。手動で行う場合や、起動時にネットワークにアクセスできないマシンでは、`hermes plugins install honcho` を実行してください。


---

### OpenViking

:::info プラグインカタログ
OpenVikingはVolcengineによってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。ソース: [volcengine/OpenViking — examples/hermes-plugin](https://github.com/volcengine/OpenViking/tree/main/examples/hermes-plugin)。既存のセットアップは自動的に移行されます — [バンドル版OpenVikingからの移行](#migrating-from-bundled-openviking) を参照してください。
:::

Volcengine（ByteDance）によるコンテキストデータベース。ファイルシステム風の知識階層、段階的な取得、6つのカテゴリへの自動メモリ抽出を備えています。

| | |
|---|---|
| **最適な用途** | 構造化されたブラウジングを伴うセルフホストの知識管理 |
| **必要なもの** | `hermes plugins install openviking`。OpenVikingが初期化、検証され、稼働していること |
| **データストレージ** | セルフホスト（ローカルまたはクラウド） |
| **コスト** | 無料（オープンソース、AGPL-3.0） |

**ツール（6個）:** `viking_search`（セマンティック検索）、`viking_read`（段階的: abstract/overview/full）、`viking_browse`（ファイルシステムナビゲーション）、`viking_remember`（事実の保存）、`viking_forget`（正確な `viking://` URIでメモリファイルを削除）、`viking_add_resource`（URL/ドキュメントの取り込み）

**セットアップ:**
```bash
# Prepare OpenViking first
openviking-server init
openviking-server doctor
openviking-server

# Then install and configure the Hermes plugin
hermes plugins install openviking
hermes memory setup    # select "openviking"
# Or manually:
hermes config set memory.provider openviking
```

`hermes memory setup` は、`~/.openviking/ovcli.conf` から接続値を再利用またはコピーできます。
手動でのセットアップでは、アクティブなプロファイルの `.env` ファイルを使用します。
デフォルトプロファイルでは `~/.hermes/.env`、名前付きプロファイルでは
`~/.hermes/profiles/<profile>/.env` を使ってください。

```text
OPENVIKING_ENDPOINT=http://127.0.0.1:1933
# OPENVIKING_API_KEY=...
# OPENVIKING_ACCOUNT=default
# OPENVIKING_USER=default
```

OpenVikingサーバーの設定は `ov.conf`（`--config`、
`OPENVIKING_CONFIG_FILE`、または `~/.openviking/ov.conf`）にあります。クライアントの接続値は
`ovcli.conf`（`OPENVIKING_CLI_CONFIG_FILE` または
`~/.openviking/ovcli.conf`）にあります。

エンドポイントがローカルで、何もリッスンしていない場合、プラグインは
`openviking-server` をバックグラウンドで起動します。カタログに登録されているバージョンでは、そのサーバーは
Hermesプロセスの環境全体（`PYTHONPATH` を除く）を継承し、そこに設定された
ボット、ゲートウェイ、リレーのトークンも含まれます。これが問題になる場合は、Hermesより前に
自分で `openviking-server` を起動して、プラグインがそれを起動しないようにしてください。
この修正はアップストリームの
[volcengine/OpenViking#5553](https://github.com/volcengine/OpenViking/pull/5553) で追跡されています。

**主な機能:**
- 段階的なコンテキストロード: L0（約100トークン）→ L1（約2k）→ L2（フル）
- セッションコミット時の自動メモリ抽出（プロファイル、設定、エンティティ、イベント、ケース、パターン）
- 階層的な知識ブラウジングのための`viking://` URIスキーム

`OPENVIKING_ACCOUNT` と `OPENVIKING_USER` は、ローカル/信頼モードで使用されます。
ピアのアイデンティティは任意です。デフォルトでは、HermesはピアIDを送信せず、
明示的なメモリを `viking://user/<user>/memories/...` に書き込みます。セットアップでは
ピアIDを尋ねません。アシスタントのコンテキストを分けるには、`config.yaml` で
`memory.openviking.agent: work-assistant` を設定してください。

空でない既存のピア設定では、ピアスコープの書き込みと想起が維持されます。
これには、`OPENVIKING_AGENT` や、リンクされたOpenViking設定内の `actor_peer_id` またはレガシーな
`agent_id` が含まれます。既存のメモリが移動されたり削除されたりすることはありません。
ピアIDがない場合、デフォルトの検索は同じOpenVikingユーザー配下のユーザーメモリと
既存のピアメモリを対象にします。古いピアメモリは既存のパスで引き続き検索できます。
どのメモリが返されるかは、ランキングと結果の上限によって決まります。
以前のピアスコープの書き込みに戻すには、`memory.openviking.agent: hermes` を設定してください。
この変更以前にユーザースコープで書き込まれたメモリはそのまま残り、
引き続き検索できます。この設定は今後の書き込みを変えるもので、既存のメモリの場所は変えません。

Hermesは、OpenVikingへのリクエストで `User-Agent: openviking-memory-hermes/<version>` を送信します。
この標準のハーネス識別子にはユーザーごとの識別子は含まれず、
別のリクエストを追加することもありません。

### バンドル版OpenVikingからの移行 {#migrating-from-bundled-openviking}

OpenVikingは以前Hermesのツリー内で提供されていました。`config.yaml` にすでに `memory.provider: openviking` がある場合、ほとんどのユーザーは何もする必要がありません:

- `hermes update` は、そのプロバイダーを指定しているすべてのプロファイルホームにカタログプラグインをインストールします。
- 最初のエージェント起動時（`hermes chat`、ゲートウェイ、Desktop）にプラグインがまだない場合、Hermesはそれをインストールし、インストールしたことを伝えます。
- `security.allow_lazy_installs: false` の場合、エージェント起動時の経路では何もインストールされず、代わりに正確な `hermes plugins install openviking` コマンドが表示されます。

`memory.provider`、`memory.openviking.*`、`.env` 内の `OPENVIKING_*` キー、`~/.openviking/`、OpenVikingサーバー上のメモリには手が加えられません。`hermes memory status` と `hermes plugins list` で確認してください。

---

### Mem0

:::info プラグインカタログ
Mem0はMem0によってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。ソース: [mem0ai/mem0 — integrations/hermes-plugin-mem0](https://github.com/mem0ai/mem0/tree/main/integrations/hermes-plugin-mem0)。既存のセットアップは自動的に移行されます — [バンドル版Mem0からの移行](#migrating-from-bundled-mem0) を参照してください。
:::

セマンティック検索、再ランキング、自動重複排除を備えた、サーバー側のLLM事実抽出。3つの接続モードがあります: **Platform**（Mem0 Cloud）、**セルフホストのダッシュボード**（Docker経由で自分で動かすMem0サーバー）、**OSS**（独自のLLM + ベクトルストアを使う、プロセス内のMem0）。

| | |
|---|---|
| **最適な用途** | 手間のかからないメモリ管理 — Mem0が抽出を自動的に処理 |
| **必要なもの** | `hermes plugins install mem0`（プラグインとともにMem0 SDKをインストールします）。APIキー（Platform）、稼働中のMem0サーバー（セルフホストのダッシュボード）、またはLLM + ベクトルストア（OSS） |
| **データストレージ** | Mem0 Cloud（Platform）、自分のMem0サーバー（セルフホストのダッシュボード）、またはプロセス内（OSS） |
| **コスト** | Mem0の料金（Platform） / 無料（セルフホストまたはOSS） |

mem0ai SDKは、シェルまたは `~/.hermes/.env` で `MEM0_TELEMETRY=false` が設定されていない限り、
匿名のPostHog使用状況テレメトリを送信します。

**ツール（4個）:** `mem0_search`（セマンティック検索。Platformモードでは任意で再ランキング、デフォルトはオフ）、`mem0_add`（事実をそのまま保存）、`mem0_update`（IDで更新）、`mem0_delete`（IDで削除）

**セットアップ（Platform）:**
```bash
hermes plugins install mem0   # from the plugin catalog
hermes memory setup    # select "mem0" → "Platform"
# Or manually:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**セットアップ（OSS）:**
```bash
hermes memory setup    # select "mem0" → "Open Source (self-hosted)"
# Or via flags:
hermes memory setup mem0 --mode oss --oss-llm openai --oss-llm-key sk-... --oss-vector qdrant
```

ファイルを書き込まずにプレビューする:
```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-... --dry-run
```

**セットアップ（セルフホストのダッシュボード）:** Docker経由で動かしているMem0サーバー（ダッシュボードのREST API）に接続します:

```bash
hermes memory setup    # select "mem0" → "Self-hosted server"
# Or via flags:
hermes memory setup mem0 --mode selfhosted --host http://localhost:8888 --api-key your-admin-api-key
```

または手動で設定します — 環境変数として:

```bash
echo "MEM0_HOST=http://localhost:8888" >> ~/.hermes/.env
echo "MEM0_API_KEY=your-admin-api-key" >> ~/.hermes/.env
```

または `mem0.json` で:

```json
{ "host": "http://localhost:8888", "api_key": "your-admin-api-key" }
```

プラグインは `X-API-Key` で認証し、サーバーの `/search` / `/memories` ルートを使用します。`api_key` は任意です（`AUTH_DISABLED` のサーバーの場合にのみ省略してください）。`mode: oss` は設定しないでください — `host` より優先されてしまいます。

**設定:** `$HERMES_HOME/mem0.json`（動作設定）。`~/.hermes/.env` に置くべきなのはシークレットの `MEM0_API_KEY` だけです。

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `mode` | `platform` | `platform`（Mem0 Cloud）または `oss`（自己管理、プロセス内） |
| `host` | — | セルフホストのMem0サーバーのURL（Dockerダッシュボード）。`X-API-Key` 付きのHTTPでルーティングします。`mode: oss` と組み合わせないでください |
| `user_id` | `hermes-user` | ユーザー識別子 |
| `agent_id` | `hermes` | エージェント識別子 |
| `rerank` | `false` | 関連性のために検索結果を再ランキングする（Platformモードのみ） |
| `sync_max_chars` | `450` | 各ターンを事実抽出に送る前に適用される、メッセージごとの文字数上限。最後の文の境界で切り詰めます。デフォルトは512トークンのエンベッダー（Ollamaの `bge-small-zh-v1.5`、`all-minilm`）に合わせたものです。`text-embedding-3-small`、`jina-embeddings-v3`、`bge-m3` のような8kトークンのエンベッダーでは値を上げてください（例: `6000`） |

**OSSでサポートされるプロバイダー:**

| コンポーネント | プロバイダー |
|-----------|-----------|
| LLM | openai, ollama |
| Embedder | openai, ollama |
| Vector Store | qdrant (local/server), pgvector |

**モードの切り替え:** `hermes memory setup mem0 --mode <platform|selfhosted|oss>` を再実行するか、`mem0.json` を直接編集します。

### バンドル版Mem0からの移行 {#migrating-from-bundled-mem0}

Mem0は以前Hermesのツリー内で（そして `hermes-agent[mem0]` のpip extraとして）提供されていました。`config.yaml` にすでに `memory.provider: mem0` がある場合、ほとんどのユーザーは何もする必要がありません:

- `hermes update` は、そのプロバイダーを指定しているすべてのプロファイルホームにカタログプラグインをインストールします。
- 最初のエージェント起動時（`hermes chat`、ゲートウェイ、Desktop）にプラグインがまだない場合、Hermesはそれをインストールし、インストールしたことを伝えます。
- `security.allow_lazy_installs: false` の場合、エージェント起動時の経路では何もインストールされず、代わりに正確な `hermes plugins install mem0` コマンドが表示されます。

`memory.provider`、`$HERMES_HOME/mem0.json`、`.env` 内の `MEM0_*` キー、そしてMem0 Cloud、Mem0サーバー、ローカルのOSSストアに保存されたメモリには手が加えられません。`hermes memory status` と `hermes plugins list` で確認してください。

---

### Hindsight

:::info プラグインカタログ
Hindsightは[vectorize-io](https://github.com/vectorize-io/hindsight)によってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。セットアップの詳細はアップストリームのドキュメントにあります: [hindsight.vectorize.io/sdks/integrations/hermes](https://hindsight.vectorize.io/sdks/integrations/hermes)。
:::

ナレッジグラフ、エンティティ解決、マルチストラテジー取得を備えた長期メモリ。`hindsight_reflect`ツールは、他のどのプロバイダーも提供しないメモリ横断の合成を提供します。セッションレベルのドキュメント追跡とともに、完全な会話ターン（ツール呼び出しを含む）を自動的に保持します。

| | |
|---|---|
| **最適な用途** | エンティティ関係を伴うナレッジグラフベースの想起 |
| **必要なもの** | `hermes plugins install hindsight`。クラウド: [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)のAPIキー。ローカル: LLMのAPIキー（OpenAI、Groq、OpenRouterなど） |
| **データストレージ** | Hindsight Cloud、ローカル組み込みPostgreSQL、または外部のローカルHindsightサーバー |
| **コスト** | Hindsightの料金（クラウド）または無料（ローカル） |

**ツール:** `hindsight_retain`（エンティティ抽出を伴う保存）、`hindsight_recall`（マルチストラテジー検索）、`hindsight_reflect`（メモリ横断の合成）

**セットアップ:**
```bash
hermes plugins install hindsight   # from the plugin catalog; answer "Use 'hindsight' as the memory provider now?"
hermes memory setup                # select "hindsight"
# Or manually:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

プラグインは `~/.hermes/plugins/hindsight/`（プロファイルホームごと）に配置されます。メモリプロバイダーは `plugins.enabled` ではなく、`config.yaml` の `memory.provider` によって有効化されます: インストール時のプロンプトを承認する（または `--enable` を渡す）とそれが設定され、拒否した場合は変更されないため、後で `hermes memory setup` で選べます。`hermes plugins enable hindsight` はメモリプロバイダーを有効化しません。`hermes memory setup`、`hermes memory status`、`hermes plugins list`、ダッシュボードのMemory設定はいずれも、カタログからインストールしたプラグインで動作します。ローカル組み込みモードでは、プラグインは初回使用時にHermesの遅延インストールの経路を通じて `hindsight-all` をインストールします。この経路は `security.allow_lazy_installs` を尊重します。

**ローカルモードのUI:** `hindsight-embed -p hermes ui start`

**設定:** `$HERMES_HOME/hindsight/config.json`

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`、`local_embedded`、または `local_external` |
| `bank_id` | `hermes` | メモリバンク識別子 |
| `recall_budget` | `mid` | 想起の徹底度: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid`（コンテキスト + ツール）、`context`（自動注入のみ）、`tools`（ツールのみ） |
| `auto_retain` | `true` | 会話ターンを自動的に保持 |
| `auto_recall` | `true` | 各ターンの前にメモリを自動的に想起 |
| `retain_async` | `true` | 保持処理をサーバー上で非同期に行う |
| `retain_context` | `conversation between Hermes Agent and the User` | 保持されたメモリのコンテキストラベル |
| `retain_tags` | — | 保持されたメモリに適用されるデフォルトのタグ。呼び出しごとのツールのタグとマージされる |
| `retain_source` | — | 保持されたメモリに付与される任意の`metadata.source` |
| `retain_user_prefix` | `User` | 自動保持されたトランスクリプトでユーザーターンの前に使われるラベル |
| `retain_assistant_prefix` | `Assistant` | 自動保持されたトランスクリプトでアシスタントターンの前に使われるラベル |
| `recall_tags` | — | 想起時にフィルタするタグ |

完全な設定リファレンスについては[アップストリームのHermes連携ドキュメント](https://hindsight.vectorize.io/sdks/integrations/hermes)を参照してください。

#### バンドル版Hindsightからの移行 {#migrating-from-bundled-hindsight}

Hindsightは以前Hermesのツリー内で（そして `hermes-agent[hindsight]` のpip extraとして）提供されていました。`config.yaml` にすでに `memory.provider: hindsight` がある場合、ほとんどのユーザーは何もする必要がありません:

- `hermes update` は、そのプロバイダーを指定しているすべてのプロファイルホームにカタログプラグインをインストールします。各行には、対象のプロファイル名が表示されます。ターミナルでは、プラグインのPython依存関係を準備する前に確認を求めます。複数のプロファイルがそのプロバイダーを使っている場合、質問は1回だけ行われ、その回答がすべてに適用されます。ターミナルがない場合（Desktopアプリ、スクリプト、サービス）は誰も回答できないため、`security.allow_lazy_installs` がオン（デフォルト）の各プロファイルは無人で依存関係を準備します。オフのプロファイルには代わりに正確な `hermes -p <profile> plugins install hindsight` コマンドが表示され、他のプロファイルは引き続き移行されます。
- 最初のエージェント起動時（`hermes chat`、Desktop、ゲートウェイ、…）にプラグインがまだない場合、Hermesは依存関係を含めてそれをインストールし、``✓ Memory provider 'hindsight' moved out of core — installed its plugin from the catalog (memory.provider and your stored memories are unchanged; check its settings with `hermes memory status`).`` と表示します。メッセージングプラットフォームでは、最初の返信とともにこの行が届きます。
- エージェント起動時のインストールができない場合、外部メモリなしで黙って実行されるのではなく、その理由が表示されます: `security.allow_lazy_installs: false` の場合、警告にはそのプロファイル用のインストールコマンドが示されます。オフラインや拒否によるインストールでは、エラーと同じコマンドが表示されます。試行が失敗した後の1時間は、エージェントの起動時にインストールをスキップし、コマンドを繰り返し表示するだけです。`hermes update` は常に再試行します。
- エージェントの起動時に質問が行われることはありません（チャットのプロンプトがターミナルを占有しているため）。そのため、Hermesに同梱されたことのないカタログプロバイダー（例えば `mnemosyne`）は、エージェント起動時にはインストールされません: 警告にはそのプロファイル用の `hermes plugins install <name>` が示されるので、それを実行するか、ダッシュボード/DesktopのPluginsページからインストールしてください。

ディスク上で変わること: プラグインが `~/.hermes/plugins/hindsight/` に現れ、`config.yaml` に `plugins.enabled: [hindsight]` が追加されます。`memory.provider`、`$HERMES_HOME/hindsight/config.json`、`.env` 内の `HINDSIGHT_API_KEY`、メモリバンクのデータには手が加えられません。`hermes memory status`（プロバイダーがアクティブか）と `hermes plugins list`（プラグインがインストールされ有効か）で確認してください。

:::warning プラグインが設定を読む場所
Hindsightは、`$HERMES_HOME/hindsight/config.json`（プロファイルホームごと）、`~/.hindsight/config.json`（レガシーな共有設定）、および `.env` 内の `HINDSIGHT_*` 変数を読み取ります。`config.yaml` の `memory.hindsight` セクションは読み取りません: そこにある `memory.hindsight.*` キーは無視されます。プラグイン自身の `config.json`（上のキーの表）を編集するか、`hermes memory setup` を使ってください。
:::

---

### Holographic

:::info プラグインカタログ
Holographicは[itpartypattaya](https://github.com/itpartypattaya/hermes-plugin-holographic)によってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。以前ツリー内で提供されていたプロバイダーで、プロバイダー名、設定、データベースも同じです。既存のセットアップは自動的に移行されます — [バンドル版Holographicからの移行](#migrating-from-bundled-holographic) を参照してください。
:::

FTS5全文検索、信頼スコアリング、構成的代数クエリのためのHRR（Holographic Reduced Representations）を備えた、ローカルのSQLite事実ストア。

| | |
|---|---|
| **最適な用途** | 高度な取得を伴うローカル専用メモリ、外部依存なし |
| **必要なもの** | なし（SQLiteは常に利用可能）。HRR代数にはNumPyが任意。 |
| **データストレージ** | ローカルSQLite |
| **コスト** | 無料 |

**ツール:** `fact_store`（9つのアクション: add、search、probe、related、reason、contradict、update、remove、list）、`fact_feedback`（信頼スコアを訓練する有用/無用の評価）

**セットアップ:**
```bash
hermes plugins install holographic   # from the plugin catalog
hermes memory setup                  # select "holographic"
# Or manually:
hermes config set memory.provider holographic
```

**設定:** `config.yaml`の`plugins.hermes-memory-store`の下

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLiteデータベースのパス |
| `auto_extract` | `false` | セッション終了時に事実を自動抽出 |
| `default_trust` | `0.5` | デフォルトの信頼スコア（0.0〜1.0） |

**独自の機能:**
- `probe` — エンティティ固有の代数的想起（人物/物事に関するすべての事実）
- `reason` — 複数のエンティティにまたがる構成的なANDクエリ
- `contradict` — 矛盾する事実の自動検出
- 非対称なフィードバックを伴う信頼スコアリング（有用 +0.05 / 無用 -0.10）

カタログプラグインには、このリスト以外にも独自のリリースノートとオプションがあります。現在のツールのアクションと設定については、その[README](https://github.com/itpartypattaya/hermes-plugin-holographic)を参照してください。

#### バンドル版Holographicからの移行 {#migrating-from-bundled-holographic}

Holographicは以前Hermesのツリー内で提供されていました。`config.yaml` にすでに `memory.provider: holographic` がある場合、何もする必要はありません:

- `hermes update` は、そのプロバイダーを指定しているすべてのプロファイルホームにカタログプラグインをインストールします。
- 最初のエージェント起動時（`hermes chat`、ゲートウェイ、Desktop）にプラグインがまだない場合、Hermesはそれをインストールし、インストールしたことを伝えます。
- `security.allow_lazy_installs: false` の場合、エージェント起動時の経路では何もインストールされず、代わりに正確な `hermes plugins install holographic` コマンドが表示されます。

あなたの事実（`$HERMES_HOME/memory_store.db`、または設定した `db_path`）と `plugins.hermes-memory-store` の設定はこれまでとまったく同じように読み込まれるため、メモリが失われることはありません。`hermes memory status` と `hermes plugins list` で確認してください。

Nous Researchの暫定コピーをURLでインストールしていた場合（`NousResearch/hermes-plugin-holographic`、現在はアーカイブ済み）、`hermes plugins update holographic` でそれをカタログのエントリに移行できます。データはそのままの場所に残ります。

---

### RetainDB

ハイブリッド検索（ベクトル + BM25 + 再ランキング）、7つのメモリタイプ、デルタ圧縮を備えたクラウドメモリAPI。

| | |
|---|---|
| **最適な用途** | 既にRetainDBのインフラを使っているチーム |
| **必要なもの** | RetainDBアカウント + APIキー |
| **データストレージ** | RetainDB Cloud |
| **コスト** | 月額20ドル |

**ツール（10個）:** `retaindb_profile`（ユーザープロファイル）、`retaindb_search`（セマンティック検索）、`retaindb_context`（タスク関連のコンテキスト）、`retaindb_remember`（タイプ + 重要度を伴う保存）、`retaindb_forget`（メモリの削除）、さらにファイルツール: `retaindb_upload_file`、`retaindb_list_files`、`retaindb_read_file`、`retaindb_ingest_file`、`retaindb_delete_file`

**セットアップ:**
```bash
hermes memory setup    # select "retaindb"
# Or manually:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

`brv` CLI経由の永続メモリ — 段階的な取得（ファジーテキスト → LLM駆動の検索）を伴う階層的な知識ツリー。ローカルファーストで、任意のクラウド同期付き。

| | |
|---|---|
| **最適な用途** | CLIを伴うポータブルでローカルファーストなメモリを求める開発者 |
| **必要なもの** | ByteRover CLI（`npm install -g byterover-cli` または[インストールスクリプト](https://byterover.dev)） |
| **データストレージ** | ローカル（デフォルト）または ByteRover Cloud（任意の同期） |
| **コスト** | 無料（ローカル）または ByteRoverの料金（クラウド） |

**ツール:** `brv_query`（知識ツリーの検索）、`brv_curate`（事実/決定/パターンの保存）、`brv_status`（CLIバージョン + ツリー統計）

**セットアップ:**
```bash
# Install the CLI first
curl -fsSL https://byterover.dev/install.sh | sh

# Then configure Hermes
hermes memory setup    # select "byterover"
# Or manually:
hermes config set memory.provider byterover
```

**主な機能:**
- 圧縮前の自動抽出（コンテキスト圧縮が破棄する前にインサイトを保存する）
- `$HERMES_HOME/byterover/`に保存される知識ツリー（プロファイルスコープ）
- SOC2 Type II認証のクラウド同期（任意）

---

### Supermemory

:::info プラグインカタログ
SupermemoryはSupermemoryによってメンテナンスされており、Hermesに同梱されるのではなく[プラグインカタログ](./plugins.md)からインストールされます。ソースと完全な設定リファレンス: [supermemoryai/hermes-supermemory](https://github.com/supermemoryai/hermes-supermemory)。既存のセットアップは自動的に移行されます — [バンドル版Supermemoryからの移行](#migrating-from-bundled-supermemory) を参照してください。
:::

プロファイル想起、セマンティック検索、明示的なメモリツール、ターンごとの会話キャプチャ（セッションごと4時間ウィンドウごとに1ドキュメント）を備えた、セマンティックな長期メモリ。

| | |
|---|---|
| **最適な用途** | ユーザープロファイリングとセッションレベルのグラフ構築を伴うセマンティック想起 |
| **必要なもの** | `hermes plugins install supermemory`（プラグインとともにSupermemory SDKをインストールします）。[クラウドのAPIキー](http://app.supermemory.ai/integrations?connect=hermes)、または[セルフホストのサーバー](https://supermemory.ai/docs/self-hosting/overview) |
| **データストレージ** | Supermemory Cloud またはセルフホスト |
| **コスト** | Supermemoryの料金（クラウド） / 無料（セルフホスト） |

**ツール:** `supermemory_store`（明示的なメモリの保存）、`supermemory_search`（セマンティック類似度検索）、`supermemory_forget`（IDまたは最良一致クエリで忘れる）、`supermemory_profile`（永続的なプロファイル + 最近のコンテキスト）

**セットアップ:**
```bash
hermes plugins install supermemory   # from the plugin catalog
hermes memory setup                  # select "supermemory"
# Or manually:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

セルフホストのセットアップ:

```bash
npx supermemory local
```

`hermes plugins install supermemory` の後、`hermes memory setup` を実行する前に、
`$HERMES_HOME/supermemory.json` で `base_url` を設定します:

```json
{
  "base_url": "http://localhost:6767"
}
```

その後 `hermes memory setup` を実行し、ローカルサーバーが表示したAPIキーを入力します。
先にエンドポイントを設定しておくことで、セットアップ時の接続プローブも
ローカルに留まるようになります。

**設定:** `$HERMES_HOME/supermemory.json`

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | ホスト型またはセルフホストのSupermemoryのAPIエンドポイント。`SUPERMEMORY_BASE_URL` より優先されます。 |
| `container_tag` | `hermes` | 検索と書き込みに使われるコンテナタグ。プロファイルスコープのタグ用に`{identity}`テンプレートをサポート。 |
| `auto_recall` | `true` | ターンの前に関連するメモリコンテキストを注入 |
| `auto_capture` | `true` | 各応答の後にクリーンアップされたユーザー・アシスタントのターンを保存 |
| `max_recall_results` | `10` | コンテキストに整形する想起アイテムの最大数 |
| `profile_frequency` | `50` | 最初のターンとNターンごとにプロファイル事実を含める |
| `capture_mode` | `all` | デフォルトで小さいまたは些細なターンをスキップ |
| `search_mode` | `hybrid` | 検索モード: `hybrid`、`memories`、`documents` |
| `api_timeout` | `5.0` | SDKリクエストのタイムアウト |

**環境変数:** `SUPERMEMORY_API_KEY`（必須）、`SUPERMEMORY_BASE_URL`（`base_url` が設定されていない場合の互換性のためのフォールバック）、`SUPERMEMORY_CONTAINER_TAG`（設定を上書き）。

ベースURLの優先順位は `supermemory.json` → `SUPERMEMORY_BASE_URL` → `https://api.supermemory.ai` です。SDKの操作とセットアップ/ステータスのプローブは、すべて解決されたエンドポイントを使用します。

**主な機能:**
- 自動コンテキストフェンシング — 想起されたメモリをキャプチャされたターンから取り除き、再帰的なメモリ汚染を防ぐ
- ターンごとのキャプチャ — 完了した各ターンはその場で書き込まれ、セッションごと4時間ウィンドウごとに1ドキュメントになる
- 失敗したターンの書き込みは、次のターン、セッション終了、`/reset`、またはシャットダウン時に再試行される（at-least-once）
- エンドツーエンドのセルフホストのルーティング — SDKとプローブのリクエストは同じ設定済みエンドポイントを使う
- 最初のターンと設定可能な間隔で注入されるプロファイル事実
- **プロファイルスコープのコンテナ** — `container_tag`で`{identity}`を使い（例: `hermes-{identity}`→`hermes-coder`）、Hermesプロファイルごとにメモリを分離する
- **マルチコンテナモード** — `custom_containers`リストとともに`enable_custom_container_tags`を有効化し、エージェントが名前付きコンテナをまたいで読み書きできるようにする。自動操作はプライマリコンテナに留まる。

<details>
<summary>マルチコンテナの例</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding context."
}
```

</details>

**サポート:** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

### バンドル版Supermemoryからの移行 {#migrating-from-bundled-supermemory}

Supermemoryは以前Hermesのツリー内で（そして `hermes-agent[supermemory]` のpip extraとして）提供されていました。`config.yaml` にすでに `memory.provider: supermemory` がある場合、ほとんどのユーザーは何もする必要がありません:

- `hermes update` は、そのプロバイダーを指定しているすべてのプロファイルホームにカタログプラグインをインストールします（`security.allow_lazy_installs` が `false` の場合でも実行されます）。
- 最初のエージェント起動時（`hermes chat`、ゲートウェイ、…）にプラグインがまだない場合、Hermesはそれをインストールし、`✓ Memory provider 'supermemory' moved out of core — installed its plugin from the catalog (your memory.supermemory settings and data are unchanged).` と表示します。
- `security.allow_lazy_installs: false` の場合、エージェント起動時の経路では代わりに1行 — ``Memory provider 'supermemory' is not installed; security.allow_lazy_installs is off — run `hermes plugins install supermemory`.`` — がログに記録されるので、自分で `hermes plugins install supermemory` を実行します。

ディスク上で変わること: プラグインが `~/.hermes/plugins/supermemory/` に現れ、`config.yaml` に `plugins.enabled: [supermemory]` が追加されます。Supermemory SDKはプラグイン自身のパッケージメタデータからインストールされるため、`hermes-agent[supermemory]` extraは不要になりました。`memory.provider`、`$HERMES_HOME/supermemory.json`、`.env` 内の `SUPERMEMORY_*` キー、Supermemoryアカウントに保存されたメモリには手が加えられません。`hermes memory status`（プロバイダーがアクティブか）と `hermes plugins list`（プラグインがインストールされ有効か）で確認してください。

### Memori

Memori Cloudを使った構造化された長期メモリ。完了したターンのバックグラウンドでのキャプチャ、ツールを考慮したターンのコンテキスト、そして事実、サマリー、クォータ、サインアップ、フィードバックのための明示的な想起ツールを備えています。

| | |
|---|---|
| **最適な用途** | 構造化されたプロジェクトとセッションの帰属を伴う、エージェント主導の想起 |
| **必要なもの** | 外部から提供される `hermes-memori` CLIとプロバイダー連携 + [MemoriのAPIキー](https://app.memorilabs.ai/signup) |
| **データストレージ** | Memori Cloud |
| **コスト** | Memoriの料金 |

**ツール:** `memori_recall`（長期メモリの検索）、`memori_recall_summary`（要約されたコンテキスト）、`memori_quota`（使用量/クォータ）、`memori_signup`（サインアップメールのリクエスト）、`memori_feedback`（連携に関するフィードバックの送信）

**セットアップ:**

`hermes-memori` は外部の連携であり、マネージドなPMツール名ではありません。
その公開元の手順に従って、独立した環境にCLIをインストールしてください。
インストーラーを実行する前に、それが意図したHermesホームを対象にしており、
Python依存関係を宣言したプロバイダーを提供することを確認してください。外部の
インストーラーにHermesの選択された環境へpip installさせないでください。CLIが利用可能であるだけでは、
Hermes内でPythonプロバイダーが利用可能にはなりません。エントリポイントのみの
ディストリビューションには、それを含むオーナー管理のビルドが必要です。

```bash
# Run only after confirming the external installer's integration contract above.
hermes-memori install
hermes config set memory.provider memori
hermes memory setup
```

インストーラーがPM管理のディレクトリプロバイダーの受け入れに対応していない場合は、
`hermes pm install` のパッケージコマンドを勝手に作るのではなく、公開元にその連携を
依頼してください。依存関係の準備が成功したら、Hermesを再起動してください。

---

## プロバイダーの比較

| プロバイダー | ストレージ | コスト | ツール | 依存関係 | 独自の機能 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho**（プラグインカタログ） | クラウド/セルフホスト | 有料/無料 | 5 | `hermes plugins install honcho` | 弁証的ユーザーモデリング + セッションスコープのコンテキスト |
| **OpenViking**（プラグインカタログ） | セルフホスト | 無料 | 6 | `hermes plugins install openviking` + サーバー | ファイルシステム階層 + 段階的ロード |
| **Mem0**（プラグインカタログ） | クラウド/セルフホスト | 無料/有料 | 4 | `hermes plugins install mem0` | サーバー側のLLM抽出 + セルフホスト/OSSモード |
| **Hindsight**（プラグインカタログ） | クラウド/ローカル | 無料/有料 | 3 | `hermes plugins install hindsight` | ナレッジグラフ + reflect合成 |
| **Holographic**（プラグインカタログ） | ローカル | 無料 | 2 | `hermes plugins install holographic` | HRR代数 + 信頼スコアリング |
| **RetainDB** | クラウド | 月額20ドル | 10 | `requests` | デルタ圧縮 |
| **ByteRover** | ローカル/クラウド | 無料/有料 | 3 | `brv` CLI | 圧縮前の抽出 |
| **Supermemory**（プラグインカタログ） | クラウド/セルフホスト | 無料/有料 | 4 | `hermes plugins install supermemory` | コンテキストフェンシング + セッショングラフ取り込み + マルチコンテナ |
| **Memori** | クラウド | 無料/有料 | 5 | `hermes-memori` | ツールを考慮したメモリ + 構造化された想起 |

## プロファイルの分離

各プロバイダーのデータは[プロファイル](../profiles.md)ごとに分離されます。

- **ローカルストレージプロバイダー**（Holographic、ByteRover）は、プロファイルごとに異なる`$HERMES_HOME/`パスを使用します
- **設定ファイルプロバイダー**（Honcho、Mem0、Hindsight、Supermemory）は設定を`$HERMES_HOME/`に保存するため、各プロファイルが独自の認証情報を持ちます
- **クラウドプロバイダー**（RetainDB）は、プロファイルスコープのプロジェクト名を自動的に導出します
- **環境変数プロバイダー**（OpenViking）は、各プロファイルの`.env`ファイル経由で設定されます

## プラグインカタログへ移行するプロバイダー

メモリプロバイダーは、Hermesのツリーからそれぞれのメンテナー自身のリポジトリへと移り、
[プラグインカタログ](./plugins.md)を通じて公開されるようになっています。最初に移行したのはHindsightで（
[バンドル版Hindsightからの移行](#migrating-from-bundled-hindsight)を参照）、続いてHoncho（
[バンドル版Honchoからのアップグレード](#upgrading-from-the-bundled-honcho)を参照）、Supermemory（
[バンドル版Supermemoryからの移行](#migrating-from-bundled-supermemory)を参照）、Mem0（
[バンドル版Mem0からの移行](#migrating-from-bundled-mem0)を参照）、OpenViking（
[バンドル版OpenVikingからの移行](#migrating-from-bundled-openviking)を参照）、そしてHolographic（現在は
itpartypattayaがメンテナンス。
[バンドル版Holographicからの移行](#migrating-from-bundled-holographic)を参照）が移行しました。RetainDBと
ByteRoverは2026年10月15日にコアから外れます。それらの独立したリポジトリはメンテナンスされておらず、
新しいメンテナーを募集しています。あなたにとって変わることは何もありません:
プロバイダー名、それが読み取る設定、データディレクトリ、ツールはそのままです。
設定済みのプロバイダーがHermesに同梱されなくなると、`hermes update` はそれを指定しているすべてのプロファイルに
カタログプラグインをインストールします。Desktopアプリを通じて更新する場合は、
エージェントが初回起動時に同じことを行います。すべての結果は表示されます — ターミナル、
Desktop、またはメッセージングプラットフォームでは最初の返信とともに。インストールができない場合
（`security.allow_lazy_installs: false`、オフライン、拒否）、警告には正確な
`hermes plugins install <name>` コマンドが含まれます。

## メモリプロバイダーを作る

独自のものを作る方法については[開発者ガイド: メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin)を参照してください。
