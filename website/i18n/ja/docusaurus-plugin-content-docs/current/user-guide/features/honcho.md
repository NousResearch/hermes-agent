---
sidebar_position: 99
title: "Honcho Memory"
description: "Honcho による AI ネイティブの永続メモリ — 弁証法的推論、マルチエージェントのユーザーモデリング、深いパーソナライゼーション"
---

# Honcho Memory

[Honcho](https://github.com/plastic-labs/honcho) は、Hermesの組み込みメモリシステムの上に弁証法的推論と深いユーザーモデリングを追加する、AIネイティブのメモリバックエンドです。単純なキーバリューストレージの代わりに、Honchoは会話が起きた後にそれについて推論することで、ユーザーが誰であるか — その好み、コミュニケーションスタイル、目標、パターン — の動的なモデルを保持します。

:::info Honchoはカタログのメモリプロバイダープラグインです
HonchoはPlastic Labsによってメンテナンスされており、[プラグインカタログ](./plugins.md)からインストールされます（`hermes plugins install honcho`）。ソースは[plastic-labs/honcho](https://github.com/plastic-labs/honcho/tree/main/hermes-plugin-honcho)にあります。Honchoは[メモリプロバイダー](./memory-providers.md)システムに統合されているため、以下のすべての機能は、統一されたメモリプロバイダーインターフェースを通じて利用できます。Honchoを同梱していたリリースからアップグレードしたホームには、プラグインが自動的にインストールされます — 設定とメモリはそのまま引き継がれます。
:::

## Honchoが追加するもの

| 機能 | 組み込みメモリ | Honcho |
|-----------|----------------|--------|
| セッションをまたいだ永続化 | ✔ ファイルベースのMEMORY.md/USER.md | ✔ APIによるサーバーサイド |
| ユーザープロファイル | ✔ エージェントによる手動キュレーション | ✔ 自動的な弁証法的推論 |
| セッションサマリー | — | ✔ セッションスコープのコンテキスト注入 |
| マルチエージェントの分離 | — | ✔ ピアごとのプロファイル分離 |
| 観測モード | — | ✔ 統合観測または方向性観測 |
| 結論（導出された洞察） | — | ✔ パターンについてのサーバーサイド推論 |
| 履歴全体の検索 | ✔ FTS5セッション検索 | ✔ 結論に対するセマンティック検索 |

**弁証法的推論**: 各会話ターンの後（`dialecticCadence` でゲートされる）、Honchoはそのやり取りを分析し、ユーザーの好み、習慣、目標についての洞察を導き出します。これらは時間とともに蓄積され、ユーザーが明示的に述べたことを超えた、深まる理解をエージェントに与えます。弁証法はマルチパスの深さ（1〜3パス）をサポートし、自動的なコールド/ウォームのプロンプト選択を行います — コールドスタートのクエリは一般的なユーザーの事実に焦点を当て、ウォームのクエリはセッションスコープのコンテキストを優先します。

**セッションスコープのコンテキスト**: ベースコンテキストには、ユーザー表現とピアカードに加えて、セッションサマリーが含まれるようになりました。これにより、エージェントは現在のセッションで既に話し合われた内容を認識でき、繰り返しを減らし、継続性を可能にします。

**マルチエージェントプロファイル**: 複数のHermesインスタンスが同じユーザーと話す場合（例: コーディングアシスタントとパーソナルアシスタント）、Honchoは別々の「ピア」プロファイルを保持します。各ピアは自身の観測と結論のみを見るため、コンテキストの相互汚染を防ぎます。

## セットアップ

```bash
hermes plugins install honcho   # from the plugin catalog
hermes memory setup             # select "honcho" from the provider list
```

または手動で設定します。

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo 'HONCHO_API_KEY=***' >> ~/.hermes/.env
```

APIキーは [honcho.dev](https://honcho.dev) で取得してください。

## アーキテクチャ

### 2層のコンテキスト注入

毎ターン（`hybrid` または `context` モードで）、Honchoはシステムプロンプトに注入される2層のコンテキストを組み立てます。

1. **ベースコンテキスト** — セッションサマリー、ユーザー表現、ユーザーピアカード、AIの自己表現、AIアイデンティティカード。`contextCadence` でリフレッシュされます。これは「このユーザーは誰か」の層です。
2. **弁証法的サプリメント** — ユーザーの現在の状態とニーズについての、LLMが合成した推論。`dialecticCadence` でリフレッシュされます。これは「今何が重要か」の層です。

両方の層は連結され、`contextTokens` 予算（設定されている場合）に切り詰められます。

### コールド/ウォームのプロンプト選択

弁証法は、2つのプロンプト戦略を自動的に選択します。

- **コールドスタート**（ベースコンテキストがまだない）: 一般的なクエリ — 「この人物は誰か？ その好み、目標、作業スタイルは何か？」
- **ウォームセッション**（ベースコンテキストが存在する）: セッションスコープのクエリ — 「このセッションでこれまでに話し合われたことを踏まえて、このユーザーについて最も関連性の高いコンテキストは何か？」

これは、ベースコンテキストが投入されているかどうかに基づいて自動的に行われます。

### 3つの直交する設定ノブ

コストと深さは、3つの独立したノブによって制御されます。

| ノブ | 制御するもの | デフォルト |
|------|----------|---------|
| `contextCadence` | `context()` API呼び出し（ベース層のリフレッシュ）の間のターン数 | `1` |
| `dialecticCadence` | `peer.chat()` LLM呼び出し（弁証法層のリフレッシュ）の間のターン数 | `2`（推奨1〜5） |
| `dialecticDepth` | 弁証法の起動ごとの `.chat()` パス数（1〜3） | `1` |

これらは直交しています — 頻繁なコンテキストリフレッシュと低頻度の弁証法を組み合わせたり、低頻度で深いマルチパスの弁証法を行ったりできます。例: `contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2` は、毎ターンベースコンテキストをリフレッシュし、5ターンごとに弁証法を実行し、各弁証法実行で2パスを行います。

### 弁証法の深さ（マルチパス）

`dialecticDepth` > 1 のとき、各弁証法の起動は複数の `.chat()` パスを実行します。

- **パス0**: コールドまたはウォームのプロンプト（上記参照）
- **パス1**: 自己監査 — 初期評価のギャップを特定し、最近のセッションから証拠を合成する
- **パス2**: 調整 — 前のパス間の矛盾をチェックし、最終的な合成を生成する

各パスは比例的な推論レベルを使用します（早いパスは軽く、メインのパスはベースレベル）。パスごとのレベルは `dialecticDepthLevels` で上書きします — 例: 深さ3の実行には `["minimal", "medium", "high"]`。

前のパスが強いシグナル（長く構造化された出力）を返した場合、パスは早期に打ち切られるため、深さ3が必ずしも3回のLLM呼び出しを意味するわけではありません。

### セッション開始時のプリウォーム

セッションの初期化時、Honchoは設定された完全な `dialecticDepth` でバックグラウンドで弁証法呼び出しを発火させ、その結果をターン1のコンテキスト組み立てに直接渡します。コールドなピアに対する単一パスのプリウォームは、しばしば薄い出力を返します — マルチパスの深さは、ユーザーが話す前に監査/調整のサイクルを実行します。ターン1までにプリウォームが届かない場合、ターン1は境界付きのタイムアウトを伴う同期呼び出しにフォールバックします。

### クエリ適応型の推論レベル {#query-adaptive-reasoning-level}

自動注入される弁証法は、クエリの長さによって `dialecticReasoningLevel` をスケールします: 120文字以上で+1レベル、400文字以上で+2、`reasoningLevelCap`（デフォルト `"high"`）でクランプされます。`reasoningHeuristic: false` で無効化すると、すべての自動呼び出しを `dialecticReasoningLevel` に固定します。利用可能なレベル: `minimal`、`low`、`medium`、`high`、`max`。

## 設定オプション

Honchoは、`~/.honcho/config.json`（グローバル）または `$HERMES_HOME/honcho.json`（プロファイルローカル）で設定されます。セットアップウィザードがこれを処理してくれます。

### 認証付きのセルフホストHoncho

HermesをセルフホストのHonchoサーバーに向ける場合、`hermes honcho setup`（および `hermes memory setup`）はベースURLの後に**ローカルのJWT / ベアラートークン**を尋ねます。認証付きアクセスを有効にするには、サーバーの `AUTH_JWT_SECRET`（Honchoのcompose環境変数）で署名されたJWTを貼り付けます。`AUTH_USE_AUTH=false` で動作しているサーバーの場合は空欄のままにします。ローカルトークンはホストブロック（`honcho.json` の `hosts.<host>.apiKey`）の下に、クラウドの `apiKey` とは別に保存されるため、後で `Cloud or local?` プロンプトを `cloud` に戻しても、どちらの認証情報も失われません。

### 完全な設定リファレンス

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `contextTokens` | `null`（上限なし） | ターンあたりの自動注入コンテキストのトークン予算。上限を設けるには整数（例: 1200）を設定。単語境界で切り詰められます |
| `contextCadence` | `1` | `context()` API呼び出し（ベース層のリフレッシュ）の間の最小ターン数 |
| `dialecticCadence` | `2` | `peer.chat()` LLM呼び出し（弁証法層）の間の最小ターン数。推奨1〜5。`tools` モードでは無関係 — モデルが明示的に呼び出します |
| `dialecticDepth` | `1` | 弁証法の起動ごとの `.chat()` パス数。1〜3にクランプ |
| `dialecticDepthLevels` | `null` | パスごとの推論レベルのオプションの配列、例: `["minimal", "low", "medium"]`。比例的なデフォルトを上書きします |
| `dialecticReasoningLevel` | `'low'` | ベースの推論レベル: `minimal`、`low`、`medium`、`high`、`max` |
| `dialecticDynamic` | `true` | `true` のとき、モデルはツールパラメータで呼び出しごとに推論レベルを上書きできます |
| `dialecticMaxChars` | `600` | システムプロンプトに注入される弁証法結果の最大文字数 |
| `recallMode` | `'hybrid'` | `hybrid`（自動注入 + ツール）、`context`（注入のみ）、`tools`（ツールのみ） |
| `initOnSessionStart` | `false` | `tools` モードのみ。`true` はHonchoセッションを**セッション開始時に同期的に**作成するため、最初のツール呼び出しの前に準備が整います。`false`（デフォルト）は最初の `honcho_*` 呼び出しまで作成を遅延させます。下記の起動に関する注記を参照してください |
| `timeout` | `null`（SDKのデフォルト） | 各Honcho SDK呼び出しに許可される秒数（`requestTimeout` と環境変数 `HONCHO_TIMEOUT` も受け付けます）。到達不能なサーバーが呼び出しを保持できる時間に上限を設けます。上記の早期初期化も含みます |
| `writeFrequency` | `'async'` | メッセージをフラッシュするタイミング: `async`（バックグラウンドスレッド）、`turn`（同期）、`session`（終了時にバッチ）、または整数N |
| `saveMessages` | `true` | メッセージをHoncho APIに永続化するかどうか |
| `observationMode` | `'directional'` | `directional`（すべてオン）または `unified`（共有プール）。`observation` オブジェクトで上書きして粒度の細かい制御 |
| `messageMaxChars` | `25000` | `add_messages()` 経由で送信されるメッセージあたりの最大文字数。超過するとチャンク分割されます |
| `dialecticMaxInputChars` | `10000` | `peer.chat()` への弁証法クエリ入力の最大文字数 |
| `sessionStrategy` | `'per-directory'` | `per-directory`、`per-repo`、`per-session`、または `global` |
| `pinUserPeer` | `false` | ゲートウェイのみ。`true` のとき、すべてのプラットフォームユーザーが `peerName` に集約されます |
| `userPeerAliases` | `{}` | ゲートウェイのみ。ランタイムIDからピアへのマップ（`{"7654321": "alice"}`）。多対一 |
| `runtimePeerPrefix` | `""` | ゲートウェイのみ。どのエイリアスにも一致しない場合に、未知のランタイムIDに名前空間を付けます（`telegram_7654321`） |
| `a2aSessions` | `true` | 他のボットからのDMを、人間のセッションではなく、送信元ボットごとの独自のHonchoセッションに書き込みます。`false` はボットが作成したターンを完全にスキップします |

**セッション戦略**は、Honchoセッションがあなたの作業にどうマッピングされるかを制御します。
- `per-session` — 各 `hermes` 実行ごとに新しいセッションを取得。クリーンなスタート、ツール経由のメモリ。新規ユーザーに推奨。
- `per-directory` — 作業ディレクトリごとに1つのHonchoセッション。コンテキストが実行をまたいで蓄積されます。
- `per-repo` — gitリポジトリごとに1つのセッション。
- `global` — すべてのディレクトリをまたいだ単一のセッション。

ディレクトリベースの戦略と手動の `sessions` マッピングは、バックエンドの起動ディレクトリではなく、論理的なセッションの作業ディレクトリを使用します。Desktop/TUIのプロジェクトワークスペースとACPのセッションディレクトリは、遅延ビルドを含め、エージェントの構築時に渡されます。ディレクトリが指定されない場合、Honchoはランタイムのcwdリゾルバーを使用します: セッションコンテキスト、次にスコープ付きの `terminal.cwd` 設定、次にプロセスの起動ディレクトリの順です。

メッセージングゲートウェイは、戦略やタイトルに関係なく、チャットごとの安定したセッションキーを維持します。その他のセッションでは、`per-session` のアイデンティティが優先され、次に手動のディレクトリマッピング、明示的なタイトル、設定された戦略の順になります。

自動生成されたHermesのタイトル（`derived` または `llm`）は、Desktopのブランチのリネージタイトルも含めて表示用のメタデータであり、`sessionStrategy` を上書きしません。明示的なユーザータイトルは、手動マッピングのない、ゲートウェイ以外かつ `per-session` 以外のセッションに対して、意図的なセッション名の上書きとして引き続き機能します。

タイトルの出所が記録されるようになる前に作成されたセッションは、従来の動作を保持します: 古い自動タイトルと古いユーザータイトルを区別できないため、出所のないタイトルは明示的な上書きとして扱われます。

**Recallモード**は、メモリが会話にどう流れ込むかを制御します。
- `hybrid` — コンテキストがシステムプロンプトに自動注入され、かつツールも利用可能（いつクエリするかはモデルが決定）。
- `context` — 自動注入のみ、ツールは非表示。
- `tools` — ツールのみ、自動注入なし。エージェントは `honcho_reasoning`、`honcho_search` などを明示的に呼び出す必要があります。

**Recallモードごとの設定:**

| 設定 | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | メッセージをフラッシュ | メッセージをフラッシュ | メッセージをフラッシュ |
| `contextCadence` | ベースコンテキストのリフレッシュをゲート | ベースコンテキストのリフレッシュをゲート | 無関係 — 注入なし |
| `dialecticCadence` | 自動LLM呼び出しをゲート | 自動LLM呼び出しをゲート | 無関係 — モデルが明示的に呼び出す |
| `dialecticDepth` | 起動ごとのマルチパス | 起動ごとのマルチパス | 無関係 — モデルが明示的に呼び出す |
| `contextTokens` | 注入に上限を設ける | 注入に上限を設ける | 無関係 — 注入なし |
| `dialecticDynamic` | モデルの上書きをゲート | 該当なし（ツールなし） | モデルの上書きをゲート |

`tools` モードでは、モデルが完全に制御します — 望むときに、選んだ任意の `reasoning_level` で `honcho_reasoning` を呼び出します。Cadenceと予算の設定は、自動注入のあるモード（`hybrid` と `context`）にのみ適用されます。

**起動時の動作と `initOnSessionStart`。** `hybrid` と `context` モードでは、セッションはバックグラウンドスレッドで作成され、Honchoが遅いかダウンしている場合でも起動はフェイルオープンします。`tools` モードは設計上異なります: `initOnSessionStart: false`（デフォルト）では、最初の `honcho_*` ツール呼び出しまで何もHonchoに触れません。`initOnSessionStart: true` では、セッションは**エージェントの構築中に同期的に**作成されるため、ターン1のツール呼び出しが初期化途中のセッションと競合することはありません。この保証は、起動がHonchoを待つことを意味します: サーバーに到達できない場合、その早期パスのすべてのSDK呼び出しが接続/`timeout` の上限まで実行されてからエージェントの準備が整います。これはDesktopでは `request timed out: session.resume` / `prompt.submit` として現れます（レンダラーは30秒で諦めます）。Desktopでは、またHonchoが稼働していない可能性のあるローカルサービスである場合は常に、`initOnSessionStart` を `false` のままにし、有効にする場合は `honcho.json` で `timeout`（秒）を設定して各呼び出しに上限を設けてください。

## ゲートウェイのアイデンティティマッピング {#gateway-identity-mapping}

これらの設定は、[Hermesゲートウェイ](/docs/developer-guide/gateway-internals)を実行する場合にのみ意味があります — ゲートウェイは、ユーザーがプラットフォームネイティブのランタイムID（TelegramのUID、Discordのsnowflake、Slackのユーザー）を持って到着する唯一のエントリポイントです。CLI、TUI、デスクトップのセッションにはランタイムIDがなく、常に `peerName` に解決されるため、ゲートウェイ以外ではこれらのキーは何もしません。

セットアップウィザードは、ゲートウェイプラットフォームが接続されているかどうかを検出し、接続されていなければこのステップを完全にスキップします。実行される場合は、1つの質問 — *このゲートウェイと誰が話しますか？* — を尋ね、キーを導出します。

| 回答 | 結果 |
|--------|--------|
| **自分だけ** | `pinUserPeer: true` — エージェント以外のすべてのゲートウェイユーザーがあなたのピアに集約されます。ピン留めはすべてのエイリアスより優先されるため、ユーザー側のアイデンティティが独自のピアを必要としない場合にのみ選択してください。別々のエージェントがゲートウェイに到達し、それぞれが個別のピアを必要とする場合は、ピン留め**しないでください** — `pinUserPeer: false` のままにし、代わりに `userPeerAliases`（`[e]` エディター）でマッピングします |
| **自分 + 他の人**（プール） | `pinUserPeer: false` + あなたのランタイムIDを `peerName` にマッピングする `userPeerAliases` — あなたは共有の履歴にとどまり、他の人は独自のピアを得ます |
| **他の人だけ** | `pinUserPeer: false`、オプションで `runtimePeerPrefix` — 各ユーザーが独自のピアを得ます |

代わりに3つのキーを直接設定するには、プロンプトで `[e]` を選択します。

リゾルバーはキーを上から順に試し、最初に一致したものが採用されます: `pinUserPeer` → `userPeerAliases[id]` → `runtimePeerPrefix + id` → 生のランタイムID → `peerName` → セッションキーのフォールバック。

各ターンはそれを書いた人のピアの下に書き込まれるため、共有チャットで最初にメッセージを送った人が他の全員の事実を集めてしまうことはもうありません。別のボットからのターン（`bot:<profile>` タグ付きのBot ModeのDM、またはアダプターがボットとしてフラグを立てたプラットフォームアカウント）は常に独自のピアを得ます: `pinUserPeer` は1人の人物のアカウントを統合するものであり、ボットは決して統合しません。そのターンは送信者ごとの `a2a` セッション（`a2aSessions` を参照）に入り、結論とカードは人間を記述するものであるため、そのようなターンの実行中は `honcho_conclude` / `honcho_profile` が書き込みを拒否します。

:::warning ピン留めの解除はプールされたメモリを孤立させます
`pinUserPeer` を `true` から `false` に切り替えてもデータは移行されません — `peerName` の下に蓄積されたメモリはそこにとどまり、プラットフォームユーザーは新しい空のピアに解決されます。自分の継続性を保つには、**プール**のパスを選んで、あなたのランタイムIDが `peerName` にエイリアスされるようにしてください。ウィザードはこの遷移を検出すると、自動的にこの誘導を提示します。
:::

:::note 非推奨のキー
`pinPeerName` は `pinUserPeer` のレガシーエイリアスです — 後方互換性のために引き続き読み取られます（両方が設定されている場合は `pinUserPeer` が優先）が、書き込まれることはありません。セットアップを再実行すると、正規のキーに移行されます。
:::

## 観測（方向性 vs. 統合） {#observation-directional-vs-unified}

Honchoは、会話をメッセージを交換するピアとしてモデル化します。各ピアには、Honchoの `SessionPeerConfig` に1対1でマッピングされる2つの観測トグルがあります。

| トグル | 効果 |
|--------|--------|
| `observeMe` | Honchoがこのピアの表現を、そのピア自身のメッセージから構築する |
| `observeOthers` | このピアが他方のピアのメッセージを観測する（クロスピア推論に供給） |

2ピア × 2トグル = 4つのフラグ。`observationMode` はショートハンドのプリセットです。

| プリセット | ユーザーフラグ | AIフラグ | セマンティクス |
|--------|-----------|----------|-----------|
| `"directional"`（デフォルト） | me: on, others: on | me: on, others: on | 完全な相互観測。クロスピアの弁証法を可能にする — 「ユーザーが言ったこととAIが返答したことに基づいて、AIはユーザーについて何を知っているか」。 |
| `"unified"` | me: on, others: off | me: off, others: on | 共有プールのセマンティクス — AIはユーザーのメッセージのみを観測し、ユーザーピアは自己モデル化のみ。単一観測者プール。 |

ピアごとの制御のために、明示的な `observation` ブロックでプリセットを上書きします。

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

よくあるパターン:

| 意図 | 設定 |
|--------|--------|
| 完全な観測（ほとんどのユーザー） | `"observationMode": "directional"` |
| AIが自身の返答からユーザーを再モデル化すべきでない | `"ai": {"observeMe": true, "observeOthers": false}` |
| AIピアが自己観測から更新すべきでない強いペルソナ | `"ai": {"observeMe": false, "observeOthers": true}` |

[Honchoダッシュボード](https://app.honcho.dev)で設定したサーバーサイドのトグルは、ローカルのデフォルトより優先されます — Hermesはセッション初期化時にそれらを同期して戻します。

## ツール

Honchoがメモリプロバイダーとしてアクティブな場合、5つのツールが利用可能になります。

| ツール | 目的 |
|------|---------|
| `honcho_profile` | ピアカードの読み取りまたは更新 — 更新には `card`（事実のリスト）を渡し、読み取りには省略する |
| `honcho_search` | コンテキストに対するセマンティック検索 — 生の抜粋、LLMによる合成なし |
| `honcho_context` | 完全なセッションコンテキスト — サマリー、表現、カード、最近のメッセージ |
| `honcho_reasoning` | HonchoのLLMによる合成された回答 — 深さを制御するために `reasoning_level`（minimal/low/medium/high/max）を渡す |
| `honcho_conclude` | 結論の作成または削除 — 作成には `conclusion` を、削除（PIIのみ）には `delete_id` を渡す |

## CLIコマンド

`hermes honcho` サブコマンドは、**Honchoがアクティブなメモリプロバイダーである場合にのみ登録されます**（`config.yaml` の `memory.provider: honcho`）。新規インストールでは、`hermes memory setup honcho` でHonchoを直接設定してください（または `hermes memory setup` を実行してリストから選択します）。その後、`hermes honcho` サブコマンドが次回の起動時に現れます。

```bash
hermes memory setup honcho    # Configure Honcho directly (works before activation)
hermes honcho status          # Connection status, config, and key settings
hermes honcho setup           # Redirects to `hermes memory setup` (post-activation alias)
hermes honcho strategy        # Show or set session strategy (per-session/per-directory/per-repo/global)
hermes honcho peer            # Show or update peer names + dialectic reasoning level
hermes honcho mode            # Show or set recall mode (hybrid/context/tools)
hermes honcho tokens          # Show or set token budget for context and dialectic
hermes honcho identity        # Seed or show the AI peer's Honcho identity
hermes honcho sync            # Sync Honcho config to all existing profiles
hermes honcho peers           # Show peer identities across all profiles
hermes honcho sessions        # List known Honcho session mappings
hermes honcho map             # Map current directory to a Honcho session name
hermes honcho enable          # Enable Honcho for the active profile
hermes honcho disable         # Disable Honcho for the active profile
hermes honcho migrate         # Step-by-step migration guide from openclaw-honcho
```

## `hermes honcho` からの移行

以前にスタンドアロンの `hermes honcho setup` を使っていた場合は次のとおりです。

1. 既存の設定（`honcho.json` または `~/.honcho/config.json`）は保持されます
2. サーバーサイドのデータ（メモリ、結論、ユーザープロファイル）はそのまま残ります
3. config.yamlで `memory.provider: honcho` を設定して再アクティブ化します

再ログインや再セットアップは不要です。`hermes memory setup` を実行して「honcho」を選択してください — ウィザードが既存の設定を検出します。

## 完全なドキュメント

完全なリファレンスについては、[メモリプロバイダー — Honcho](./memory-providers.md#honcho)を参照してください。
