---
sidebar_position: 4
title: "メモリプロバイダー"
description: "外部メモリプロバイダープラグイン — Honcho、OpenViking、Mem0、Hindsight、Holographic、RetainDB、ByteRover、Supermemory"
---

# メモリプロバイダー

Hermes Agentには、組み込みのMEMORY.mdおよびUSER.mdを超えて、エージェントに永続的でセッションをまたぐ知識を与える8つの外部メモリプロバイダープラグインが付属しています。一度にアクティブにできる外部プロバイダーは**1つ**だけです — 組み込みメモリは常にそれと並行してアクティブです。

## クイックスタート

```bash
hermes memory setup      # 対話型のピッカー + 設定
hermes memory status     # 何がアクティブか確認
hermes memory off        # 外部プロバイダーを無効化
```

`hermes plugins` → Provider Plugins → Memory Provider からアクティブなメモリプロバイダーを選択することもできます。

または`~/.hermes/config.yaml`で手動設定します。

```yaml
memory:
  provider: openviking   # または honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
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

## 利用可能なプロバイダー

### Honcho

弁証的推論、セッションスコープのコンテキスト注入、セマンティック検索、永続的な結論を備えた、AIネイティブなセッションをまたぐユーザーモデリング。ベースコンテキストには、ユーザー表現とピアカードに加えてセッションサマリーが含まれるようになり、エージェントは既に議論された内容を把握できます。

| | |
|---|---|
| **最適な用途** | セッションをまたぐコンテキストを持つマルチエージェントシステム、ユーザーとエージェントの整合 |
| **必要なもの** | `pip install honcho-ai` + [APIキー](https://app.honcho.dev) またはセルフホストインスタンス |
| **データストレージ** | Honcho Cloud またはセルフホスト |
| **コスト** | Honchoの料金（クラウド） / 無料（セルフホスト） |

**ツール（5個）:** `honcho_profile`（ピアカードの読み取り/更新）、`honcho_search`（セマンティック検索）、`honcho_context`（セッションコンテキスト — サマリー、表現、カード、メッセージ）、`honcho_reasoning`（LLMで合成）、`honcho_conclude`（結論の作成/削除）

**アーキテクチャ:** 2層のコンテキスト注入 — ベース層（セッションサマリー + 表現 + ピアカード、`contextCadence`でリフレッシュ）に加えて、弁証的な補足（LLM推論、`dialecticCadence`でリフレッシュ）。弁証法は、ベースコンテキストが存在するかどうかに基づいて、コールドスタート用プロンプト（一般的なユーザー事実）とウォーム用プロンプト（セッションスコープのコンテキスト）を自動的に選択します。

**3つの直交する設定ノブ**がコストと深さを独立して制御します。

- `contextCadence` — ベース層がリフレッシュされる頻度（API呼び出しの頻度）
- `dialecticCadence` — 弁証法のLLMが発火する頻度（LLM呼び出しの頻度）
- `dialecticDepth` — 弁証法の呼び出しごとの`.chat()`のパス回数（1〜3、推論の深さ）

**セットアップウィザード:**
```bash
hermes memory setup        # 「honcho」を選択 — Honcho固有のセットアップ後処理を実行
```

レガシーの`hermes honcho setup`コマンドはまだ動作しますが（現在は`hermes memory setup`にリダイレクトされます）、Honchoがアクティブなメモリプロバイダーとして選択された後にのみ登録されます。

**設定:** `$HERMES_HOME/honcho.json`（プロファイルローカル）または`~/.honcho/config.json`（グローバル）。解決順序: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`。[設定リファレンス](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md)と[Honcho統合ガイド](https://docs.honcho.dev/v3/guides/integrations/hermes)を参照してください。

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

Honchoは会話を、メッセージを交換するピアとしてモデル化します — Hermesプロファイルごとに1つのユーザーピアと1つのAIピアがあり、すべてが1つのワークスペースを共有します。ワークスペースは共有環境です。ユーザーピアはプロファイルをまたいでグローバルであり、各AIピアはそれ自体のアイデンティティです。各AIピアは自身の観測から独立した表現 / カードを構築するため、`coder`プロファイルはコード指向のまま、`writer`プロファイルは同じユーザーに対して編集者寄りのままになります。

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

`--clone`は、`aiPeer: "coder"`、共有の`workspace`、継承された`peerName`、`recallMode`、`writeFrequency`、`observation`などを持つ`hermes.coder`ホストブロックを`honcho.json`に作成します。AIピアは最初のメッセージの前に存在するよう、Honchoで先行して作成されます。

### 既存のプロファイル、Honchoピアのバックフィル

```bash
hermes honcho sync
```

すべてのHermesプロファイルをスキャンし、ホストブロックを持たないプロファイルにホストブロックを作成し、デフォルトの`hermes`ブロックから設定を継承し、新しいAIピアを先行して作成します。冪等です — 既にホストブロックを持つプロファイルはスキップします。

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

[設定リファレンス](https://github.com/hermes-ai/hermes-agent/blob/main/plugins/memory/honcho/README.md)と[Honcho統合ガイド](https://docs.honcho.dev/v3/guides/integrations/hermes)を参照してください。


---

### OpenViking

Volcengine（ByteDance）によるコンテキストデータベース。ファイルシステム風の知識階層、段階的な取得、6つのカテゴリへの自動メモリ抽出を備えています。

| | |
|---|---|
| **最適な用途** | 構造化されたブラウジングを伴うセルフホストの知識管理 |
| **必要なもの** | `pip install openviking` + 稼働中のサーバー |
| **データストレージ** | セルフホスト（ローカルまたはクラウド） |
| **コスト** | 無料（オープンソース、AGPL-3.0） |

**ツール:** `viking_search`（セマンティック検索）、`viking_read`（段階的: abstract/overview/full）、`viking_browse`（ファイルシステムナビゲーション）、`viking_remember`（事実の保存）、`viking_add_resource`（URL/ドキュメントの取り込み）

**セットアップ:**
```bash
# まずOpenVikingサーバーを起動
pip install openviking
openviking-server

# 次にHermesを設定
hermes memory setup    # 「openviking」を選択
# または手動で:
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
```

**主な機能:**
- 段階的なコンテキストロード: L0（約100トークン）→ L1（約2k）→ L2（フル）
- セッションコミット時の自動メモリ抽出（プロファイル、設定、エンティティ、イベント、ケース、パターン）
- 階層的な知識ブラウジングのための`viking://` URIスキーム

---

### Mem0

セマンティック検索、再ランキング、自動重複排除を備えた、サーバー側のLLM事実抽出。

| | |
|---|---|
| **最適な用途** | 手間のかからないメモリ管理 — Mem0が抽出を自動的に処理 |
| **必要なもの** | `pip install mem0ai` + APIキー |
| **データストレージ** | Mem0 Cloud |
| **コスト** | Mem0の料金 |

**ツール:** `mem0_profile`（保存されたすべてのメモリ）、`mem0_search`（セマンティック検索 + 再ランキング）、`mem0_conclude`（事実をそのまま保存）

**セットアップ:**
```bash
hermes memory setup    # 「mem0」を選択
# または手動で:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**設定:** `$HERMES_HOME/mem0.json`

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `user_id` | `hermes-user` | ユーザー識別子 |
| `agent_id` | `hermes` | エージェント識別子 |

---

### Hindsight

ナレッジグラフ、エンティティ解決、マルチストラテジー取得を備えた長期メモリ。`hindsight_reflect`ツールは、他のどのプロバイダーも提供しないメモリ横断の合成を提供します。セッションレベルのドキュメント追跡とともに、完全な会話ターン（ツール呼び出しを含む）を自動的に保持します。

| | |
|---|---|
| **最適な用途** | エンティティ関係を伴うナレッジグラフベースの想起 |
| **必要なもの** | クラウド: [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)のAPIキー。ローカル: LLMのAPIキー（OpenAI、Groq、OpenRouterなど） |
| **データストレージ** | Hindsight Cloud またはローカル組み込みPostgreSQL |
| **コスト** | Hindsightの料金（クラウド）または無料（ローカル） |

**ツール:** `hindsight_retain`（エンティティ抽出を伴う保存）、`hindsight_recall`（マルチストラテジー検索）、`hindsight_reflect`（メモリ横断の合成）

**セットアップ:**
```bash
hermes memory setup    # 「hindsight」を選択
# または手動で:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

セットアップウィザードは依存関係を自動的にインストールし、選択したモードに必要なものだけをインストールします（クラウド用は`hindsight-client`、ローカル用は`hindsight-all`）。`hindsight-client >= 0.4.22`が必要です（古い場合はセッション開始時に自動アップグレード）。

**ローカルモードのUI:** `hindsight-embed -p hermes ui start`

**設定:** `$HERMES_HOME/hindsight/config.json`

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud`または`local` |
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

完全な設定リファレンスについては[プラグインのREADME](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md)を参照してください。

---

### Holographic

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
hermes memory setup    # 「holographic」を選択
# または手動で:
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

---

### RetainDB

ハイブリッド検索（ベクトル + BM25 + 再ランキング）、7つのメモリタイプ、デルタ圧縮を備えたクラウドメモリAPI。

| | |
|---|---|
| **最適な用途** | 既にRetainDBのインフラを使っているチーム |
| **必要なもの** | RetainDBアカウント + APIキー |
| **データストレージ** | RetainDB Cloud |
| **コスト** | 月額20ドル |

**ツール:** `retaindb_profile`（ユーザープロファイル）、`retaindb_search`（セマンティック検索）、`retaindb_context`（タスク関連のコンテキスト）、`retaindb_remember`（タイプ + 重要度を伴う保存）、`retaindb_forget`（メモリの削除）

**セットアップ:**
```bash
hermes memory setup    # 「retaindb」を選択
# または手動で:
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
# まずCLIをインストール
curl -fsSL https://byterover.dev/install.sh | sh

# 次にHermesを設定
hermes memory setup    # 「byterover」を選択
# または手動で:
hermes config set memory.provider byterover
```

**主な機能:**
- 圧縮前の自動抽出（コンテキスト圧縮が破棄する前にインサイトを保存する）
- `$HERMES_HOME/byterover/`に保存される知識ツリー（プロファイルスコープ）
- SOC2 Type II認証のクラウド同期（任意）

---

### Supermemory

プロファイル想起、セマンティック検索、明示的なメモリツール、SupermemoryグラフAPI経由のセッション終了時の会話取り込みを備えた、セマンティックな長期メモリ。

| | |
|---|---|
| **最適な用途** | ユーザープロファイリングとセッションレベルのグラフ構築を伴うセマンティック想起 |
| **必要なもの** | `pip install supermemory` + [APIキー](https://supermemory.ai) |
| **データストレージ** | Supermemory Cloud |
| **コスト** | Supermemoryの料金 |

**ツール:** `supermemory_store`（明示的なメモリの保存）、`supermemory_search`（セマンティック類似度検索）、`supermemory_forget`（IDまたは最良一致クエリで忘れる）、`supermemory_profile`（永続的なプロファイル + 最近のコンテキスト）

**セットアップ:**
```bash
hermes memory setup    # 「supermemory」を選択
# または手動で:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

**設定:** `$HERMES_HOME/supermemory.json`

| キー | デフォルト | 説明 |
|-----|---------|-------------|
| `container_tag` | `hermes` | 検索と書き込みに使われるコンテナタグ。プロファイルスコープのタグ用に`{identity}`テンプレートをサポート。 |
| `auto_recall` | `true` | ターンの前に関連するメモリコンテキストを注入 |
| `auto_capture` | `true` | 各応答の後にクリーンアップされたユーザー・アシスタントのターンを保存 |
| `max_recall_results` | `10` | コンテキストに整形する想起アイテムの最大数 |
| `profile_frequency` | `50` | 最初のターンとNターンごとにプロファイル事実を含める |
| `capture_mode` | `all` | デフォルトで小さいまたは些細なターンをスキップ |
| `search_mode` | `hybrid` | 検索モード: `hybrid`、`memories`、`documents` |
| `api_timeout` | `5.0` | SDKと取り込みリクエストのタイムアウト |

**環境変数:** `SUPERMEMORY_API_KEY`（必須）、`SUPERMEMORY_CONTAINER_TAG`（設定を上書き）。

**主な機能:**
- 自動コンテキストフェンシング — 想起されたメモリをキャプチャされたターンから取り除き、再帰的なメモリ汚染を防ぐ
- より豊かなグラフレベルの知識構築のためのセッション終了時の会話取り込み
- 最初のターンと設定可能な間隔で注入されるプロファイル事実
- 些細なメッセージのフィルタリング（「ok」「thanks」などをスキップ）
- **プロファイルスコープのコンテナ** — `container_tag`で`{identity}`を使い（例: `hermes-{identity}`→`hermes-coder`）、Hermesプロファイルごとにメモリを分離する
- **マルチコンテナモード** — `custom_containers`リストとともに`enable_custom_container_tags`を有効化し、エージェントが名前付きコンテナをまたいで読み書きできるようにする。自動操作（同期、プリフェッチ）はプライマリコンテナに留まる。

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

---

## プロバイダーの比較

| プロバイダー | ストレージ | コスト | ツール | 依存関係 | 独自の機能 |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | クラウド | 有料 | 5 | `honcho-ai` | 弁証的ユーザーモデリング + セッションスコープのコンテキスト |
| **OpenViking** | セルフホスト | 無料 | 5 | `openviking` + サーバー | ファイルシステム階層 + 段階的ロード |
| **Mem0** | クラウド | 有料 | 3 | `mem0ai` | サーバー側のLLM抽出 |
| **Hindsight** | クラウド/ローカル | 無料/有料 | 3 | `hindsight-client` | ナレッジグラフ + reflect合成 |
| **Holographic** | ローカル | 無料 | 2 | なし | HRR代数 + 信頼スコアリング |
| **RetainDB** | クラウド | 月額20ドル | 5 | `requests` | デルタ圧縮 |
| **ByteRover** | ローカル/クラウド | 無料/有料 | 3 | `brv` CLI | 圧縮前の抽出 |
| **Supermemory** | クラウド | 有料 | 4 | `supermemory` | コンテキストフェンシング + セッショングラフ取り込み + マルチコンテナ |

## プロファイルの分離

各プロバイダーのデータは[プロファイル](/docs/user-guide/profiles)ごとに分離されます。

- **ローカルストレージプロバイダー**（Holographic、ByteRover）は、プロファイルごとに異なる`$HERMES_HOME/`パスを使用します
- **設定ファイルプロバイダー**（Honcho、Mem0、Hindsight、Supermemory）は設定を`$HERMES_HOME/`に保存するため、各プロファイルが独自の認証情報を持ちます
- **クラウドプロバイダー**（RetainDB）は、プロファイルスコープのプロジェクト名を自動的に導出します
- **環境変数プロバイダー**（OpenViking）は、各プロファイルの`.env`ファイル経由で設定されます

## メモリプロバイダーを作る

独自のものを作る方法については[開発者ガイド: メモリプロバイダープラグイン](/docs/developer-guide/memory-provider-plugin)を参照してください。
