---
sidebar_position: 12
sidebar_label: "組み込みプラグイン"
title: "組み込みプラグイン"
description: "Hermes Agentに同梱され、ライフサイクルフック経由で自動的に実行されるプラグイン — disk-cleanupとその仲間"
---

# 組み込みプラグイン

Hermesには、リポジトリに同梱された少数のプラグインが付属しています。これらは `<repo>/plugins/<name>/` 配下にあり、`~/.hermes/plugins/` のユーザーインストール済みプラグインと並んで自動的に読み込まれます。これらはサードパーティプラグインと同じプラグイン面（フック、ツール、スラッシュコマンド）を使用し、ツリー内でメンテナンスされる点だけが異なります。

一般的なプラグインシステムについては[プラグイン](/docs/user-guide/features/plugins)ページを、独自のものを書くには[Hermesプラグインを構築する](/docs/guides/build-a-hermes-plugin)を参照してください。

## ディスカバリーの仕組み

`PluginManager` は4つのソースを順にスキャンします。

1. **同梱** — `<repo>/plugins/<name>/`（このページが説明する対象）
2. **ユーザー** — `~/.hermes/plugins/<name>/`
3. **プロジェクト** — `./.hermes/plugins/<name>/`（`HERMES_ENABLE_PROJECT_PLUGINS=1` が必要）
4. **Pipエントリポイント** — `hermes_agent.plugins`

名前が衝突した場合、後のソースが優先されます。`disk-cleanup` という名前のユーザープラグインは、同梱のものを置き換えます。

`plugins/memory/` と `plugins/context_engine/` は、同梱スキャンから意図的に除外されています。メモリプロバイダーとコンテキストエンジンは、設定の `hermes memory setup` / `context.engine` を通じて設定される単一選択のプロバイダーであるため、それらのディレクトリは独自のディスカバリーパスを使用します。

## 同梱プラグインはオプトイン

同梱プラグインは無効の状態で出荷されます。ディスカバリーはそれらを見つけます（`hermes plugins list` と対話的な `hermes plugins` UIに表示されます）が、明示的に有効化するまでは何も読み込まれません。

```bash
hermes plugins enable disk-cleanup
```

または `~/.hermes/config.yaml` を介して:

```yaml
plugins:
  enabled:
    - disk-cleanup
```

これはユーザーインストール済みプラグインが使用するのと同じメカニズムです。同梱プラグインが自動的に有効化されることはありません。新規インストール時も、新しいHermesにアップグレードする既存ユーザーに対してもです。常に明示的にオプトインします。

同梱プラグインを再び無効化するには:

```bash
hermes plugins disable disk-cleanup
# または: config.yaml の plugins.enabled から削除する
```

## 現在出荷されているもの

リポジトリは `plugins/` 配下に次の同梱プラグインを出荷しています。すべてオプトインです。`hermes plugins enable <name>` で有効化してください。

| プラグイン | 種類 | 目的 |
|---|---|---|
| `disk-cleanup` | フック + スラッシュコマンド | 一時ファイルを自動追跡し、セッション終了時にクリーンアップする |
| `observability/langfuse` | フック | ターン / LLM呼び出し / ツールを[Langfuse](https://langfuse.com)にトレースする |
| `spotify` | バックエンド（7ツール） | ネイティブなSpotifyの再生、キュー、検索、プレイリスト、アルバム、ライブラリ |
| `google_meet` | スタンドアロン | Meet通話への参加、ライブキャプションの文字起こし、任意のリアルタイム双方向音声 |
| `image_gen/openai` | 画像バックエンド | OpenAI `gpt-image-2` 画像生成バックエンド（FALの代替） |
| `image_gen/openai-codex` | 画像バックエンド | Codex OAuth経由のOpenAI画像生成 |
| `image_gen/xai` | 画像バックエンド | xAI `grok-2-image` バックエンド |
| `hermes-achievements` | ダッシュボードタブ | 実際のHermesセッション履歴から生成されるSteam風の収集可能バッジ |
| `kanban/dashboard` | ダッシュボードタブ | マルチエージェントディスパッチャー用のかんばんボードUI — タスク、コメント、ファンアウト、ボード切り替え。[かんばんマルチエージェント](./kanban.md)を参照。 |

メモリプロバイダー（`plugins/memory/*`）とコンテキストエンジン（`plugins/context_engine/*`）は、[メモリプロバイダー](./memory-providers.md)に別途列挙されています。これらはそれぞれ `hermes memory` と `hermes plugins` を通じて管理されます。2つの長期稼働するフックベースのプラグインのプラグインごとの詳細は次のとおりです。

### disk-cleanup

セッション中に作成される一時ファイル（テストスクリプト、一時出力、cronログ、古いchromeプロファイル）を、エージェントがツールを呼び出すのを覚えておく必要なしに、自動的に追跡して削除します。

**動作の仕組み:**

| フック | 動作 |
|---|---|
| `post_tool_call` | `write_file` / `terminal` / `patch` が、`HERMES_HOME` または `/tmp/hermes-*` 内で `test_*`、`tmp_*`、`*.test.*` に一致するファイルを作成すると、それを `test` / `temp` / `cron-output` として静かに追跡する。 |
| `on_session_end` | ターン中にテストファイルが自動追跡された場合、安全な `quick` クリーンアップを実行し、1行の要約をログに記録する。それ以外の場合は沈黙を保つ。 |

**削除ルール:**

| カテゴリ | しきい値 | 確認 |
|---|---|---|
| `test` | セッション終了ごと | なし |
| `temp` | 追跡から7日超 | なし |
| `cron-output` | 追跡から14日超 | なし |
| HERMES_HOME配下の空ディレクトリ | 常時 | なし |
| `research` | 30日超かつ最新10件を超える分 | あり（deepのみ） |
| `chrome-profile` | 追跡から14日超 | あり（deepのみ） |
| 500 MB超のファイル | 自動では決して削除しない | あり（deepのみ） |

**スラッシュコマンド** — `/disk-cleanup` はCLIとゲートウェイの両方のセッションで利用可能:

```
/disk-cleanup status                     # 内訳 + 最大の上位10件
/disk-cleanup dry-run                    # 削除せずにプレビュー
/disk-cleanup quick                      # 今すぐ安全なクリーンアップを実行
/disk-cleanup deep                       # quick + 確認が必要な項目を列挙
/disk-cleanup track <path> <category>    # 手動追跡
/disk-cleanup forget <path>              # 追跡を停止（削除はしない）
```

**状態** — すべては `$HERMES_HOME/disk-cleanup/` にあります:

| ファイル | 内容 |
|---|---|
| `tracked.json` | カテゴリ、サイズ、タイムスタンプ付きの追跡パス |
| `tracked.json.bak` | 上記のアトミック書き込みバックアップ |
| `cleanup.log` | すべての追跡 / スキップ / 拒否 / 削除の追記専用監査証跡 |

**安全性** — クリーンアップは `HERMES_HOME` または `/tmp/hermes-*` 配下のパスにのみ影響します。Windowsマウント（`/mnt/c/...`）は拒否されます。よく知られたトップレベルの状態ディレクトリ（`logs/`、`memories/`、`sessions/`、`cron/`、`cache/`、`skills/`、`plugins/`、`disk-cleanup/` 自体）は、空であっても決して削除されません。新規インストールが初回のセッション終了で台無しになることはありません。

**有効化:** `hermes plugins enable disk-cleanup`（または `hermes plugins` でチェックボックスをオンにする）。

**再び無効化:** `hermes plugins disable disk-cleanup`。

### observability/langfuse

Hermesのターン、LLM呼び出し、ツール呼び出しを、オープンソースのLLM可観測性プラットフォームである[Langfuse](https://langfuse.com)にトレースします。ターンごとに1スパン、API呼び出しごとに1生成、ツール呼び出しごとに1ツール観測です。使用量の合計、タイプごとのトークン数、コスト見積もりは、Hermesの正規の `agent.usage_pricing` の数値から得られるため、Langfuseダッシュボードは `hermes logs` に表示されるのと同じ内訳（input / output / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`）を見ます。

このプラグインはフェイルオープンです。SDKがインストールされていない、認証情報がない、またはLangfuseの一時的なエラー — これらはすべてフック内で静かなノーオペレーションになります。エージェントループが影響を受けることはありません。

**セットアップ（対話的 — 推奨）:**

```bash
hermes tools          # → Langfuse Observability → Cloud or Self-Hosted
```

ウィザードはあなたのキーを収集し、`langfuse` SDKを `pip install` し、`observability/langfuse` を `plugins.enabled` に追加してくれます。Hermesを再起動すると、次のターンでトレースが送信されます。

**セットアップ（手動）:**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

その後、認証情報を `~/.hermes/.env` に入れます。

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # またはセルフホストのURL
```

**動作の仕組み:**

| フック | 動作 |
|---|---|
| `pre_api_request` / `pre_llm_call` | ターンごとのルートスパン "Hermes turn" を開く（または再利用する）。シリアライズされた最近のメッセージを入力として、このAPI呼び出しの `generation` 子観測を開始する。 |
| `post_api_request` / `post_llm_call` | 生成を閉じ、`usage_details`、`cost_details`、`finish_reason`、アシスタント出力 + ツール呼び出しを付加する。ツール呼び出しがなく、コンテンツが空でない場合、ターンを閉じる。 |
| `pre_tool_call` | サニタイズされた `args` で `tool` 子観測を開始する。 |
| `post_tool_call` | サニタイズされた `result` でツール観測を閉じる。`read_file` のペイロードは要約され（先頭 + 末尾 + 省略行数）、巨大なファイル読み取りが `HERMES_LANGFUSE_MAX_CHARS` 未満に収まるようにする。 |

セッションのグルーピングは、`langfuse.propagate_attributes` を介してHermesのセッションID（サブエージェントの場合はタスクID）に基づくため、単一の `hermes chat` セッション内のすべてが1つのLangfuseセッションの下に存在します。

**確認:**

```bash
hermes plugins list                 # observability/langfuse が "enabled" と表示されるはず
hermes chat -q "hello"              # Langfuse UIで "Hermes turn" トレースを確認する
```

**任意のチューニング**（`.env` 内）:

| 変数 | デフォルト | 目的 |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | トレースの環境タグ（`production`、`staging`、…） |
| `HERMES_LANGFUSE_RELEASE` | — | リリース/バージョンのタグ |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | SDKに渡されるサンプリングレート（0.0〜1.0） |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | メッセージコンテンツ / ツール引数 / ツール結果のフィールドごとの切り詰め |
| `HERMES_LANGFUSE_DEBUG` | `false` | `agent.log` への詳細なプラグインログ |

Hermesプレフィックス付きと標準SDKの環境変数（`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL`）の両方が受け付けられます。両方が設定されている場合はHermesプレフィックス付きが優先されます。

**パフォーマンス:** Langfuseクライアントは最初のフック呼び出しの後にキャッシュされます。認証情報またはSDKが欠けている場合、その判定もキャッシュされ、以降のフックは環境変数の再確認や設定の再読み込みなしに高速に返ります。

**無効化:** `hermes plugins disable observability/langfuse`。プラグインモジュールは引き続き検出されますが、再び有効化するまでモジュールのコードは実行されません。

### google_meet

エージェントが**Google Meet通話に参加し、文字起こしし、参加できる**ようにします。ミーティングのメモを取り、後でやり取りを要約し、特定のポイントをフォローアップし、（任意で）TTS経由で通話に返答を発話します。

**追加される機能:**

- ブラウザ自動化を使ってMeet URLに参加するヘッドレスの仮想参加者
- 設定されたSTTプロバイダー経由のミーティング音声のライブ文字起こし
- 聞いた内容に基づいて動作するためにエージェントが呼び出す `meet_summarize` / `meet_speak` / `meet_followup` ツールセット
- `~/.hermes/cache/google_meet/<meeting_id>/` 配下に保存されるミーティング後のアーティファクト（トランスクリプト、話者帰属付きメモ、アクションアイテム）

**セットアップ:**

```bash
hermes plugins enable google_meet
# 初回使用時にプラグインのOAuthフロー経由でのサインインを求められます。
# Meetアクセス権を持つGoogleアカウントが必要です。ミーティングが
# 「招待された参加者のみ参加可能」を強制している場合、主催者の承認が必要な場合があります。
```

チャットからの使い方:

> 「meet.google.com/abc-defg-hij に参加してメモを取って。通話後、アクションアイテム付きの要約を送って。」

エージェントはミーティングへの参加を開始し、通話の進行に合わせて文字起こしをコンテキストにストリーミングし、ミーティングが終了したとき（または停止を指示したとき）に構造化された要約を生成します。

**いつ使うか:** 非同期の参加者のためにボットに文字起こし + 要約をしてほしい定例スタンドアップ、構造化されたメモが欲しい証言録取スタイルのインタビュー、Fireflies / Otter / Grain が必要になるあらゆるケース。AIに聞かれたくない場合は有効化しないでください。

**無効化:** `hermes plugins disable google_meet`。キャッシュされたトランスクリプトと録画は、削除するまで `~/.hermes/cache/google_meet/` に残ります。

### hermes-achievements

**ダッシュボードにSteam風のアチーブメントタブを追加します**。実際のHermesセッション履歴から生成される、60以上の収集可能でティア分けされたバッジです。ツールチェーンの偉業、デバッグパターン、バイブコーディングの連続記録、スキル/メモリの使用、モデル/プロバイダーの多様性、ライフスタイルの癖（週末や夜のセッション）。元々は[@PCinkusz](https://github.com/PCinkusz)が外部プラグインとして作成したもので、Hermesの機能変更と歩調を合わせるためにツリー内に取り込まれました。

**動作の仕組み:**

- ダッシュボードバックエンドで `~/.hermes/state.db` のセッション履歴全体をスキャンします
- セッションごとの統計は `(started_at, last_active)` のフィンガープリントでキャッシュされるため、以降のスキャンでは新規または変更されたセッションのみが再分析されます
- 初回のスキャンはバックグラウンドスレッドで実行されます。ダッシュボードは、数千のセッションを持つデータベースでも、それを待ってブロックすることは決してありません
- アンロック状態は `$HERMES_HOME/plugins/hermes-achievements/state.json` に永続化されます

**ティアの進行:** Copper → Silver → Gold → Diamond → Olympian。各カードには、追跡されている正確な指標を列挙する「What counts」セクションがあります。

**アチーブメントの状態:**

| 状態 | 意味 |
|---|---|
| Unlocked | 少なくとも1つのティアを達成 |
| Discovered | 既知のアチーブメント、進捗は表示されるがまだ獲得していない |
| Secret | Hermesが履歴で最初の関連シグナルを検出するまで非表示 |

**API** — ルートは `/api/plugins/hermes-achievements/` の下にマウントされます:

| エンドポイント | 目的 |
|---|---|
| `GET /achievements` | バッジごとのアンロック状態付きの全カタログ（最初のコールドスキャンが実行中の間は保留中のプレースホルダーを返す） |
| `GET /scan-status` | バックグラウンドスキャナーの状態: `idle` / `running` / `failed`、直近の所要時間、実行回数 |
| `GET /recent-unlocks` | 直近にアンロックされた20件のバッジ、新しい順 |
| `GET /sessions/{id}/badges` | 特定の1セッションで主に獲得されたバッジ |
| `POST /rescan` | 手動の同期的再スキャン（ブロックする。ユーザーが再スキャンボタンをクリックしたときに使用） |
| `POST /reset-state` | アンロック履歴とキャッシュされたスナップショットをクリアする |

**状態ファイル** — `$HERMES_HOME/plugins/hermes-achievements/` 配下にあります:

| ファイル | 内容 |
|---|---|
| `state.json` | アンロック履歴: どのバッジをいつ獲得したか。Hermesの更新をまたいで安定。 |
| `scan_snapshot.json` | 最後に完了したスキャンのペイロード（ダッシュボード読み込み時に即座に提供） |
| `scan_checkpoint.json` | フィンガープリントをキーとするセッションごとの統計キャッシュ（ウォーム再スキャンを高速化） |

**パフォーマンスに関する注意:**

- 約8,000セッションのコールドスキャンには数分かかります。最初のダッシュボードリクエストでバックグラウンドスレッドで実行されます。UIは保留中のプレースホルダーを表示し、`/scan-status` をポーリングします。
- **コールドスキャン中の増分結果** — スキャナーは約250セッションごとに部分的なスナップショットを公開するため、スキャンが進むにつれて各ダッシュボードの更新でより多くのバッジがアンロックされて表示されます。ゼロを何分も見つめることはありません。
- ウォーム再スキャンは、`started_at` + `last_active` のフィンガープリントがチェックポイントと一致するすべてのセッションについて、セッションごとの統計を再利用します。大きな履歴でも数秒で完了します。
- メモリ内スナップショットのTTLは120秒です。古いリクエストは即座に古いスナップショットを提供し、バックグラウンド更新を起動します。TTLが切れただけでスピナーを待つことは決してありません。

**有効化:** 有効化するものはありません。`hermes-achievements` はダッシュボード専用のプラグインです（ライフサイクルフックなし、モデルから見えるツールなし）。初回起動時に `hermes dashboard` のタブとして自動登録されます。`plugins.enabled` の設定はライフサイクル/ツールのプラグインのみをゲートします。ダッシュボードプラグインは、その `dashboard/manifest.json` を通じて純粋に検出されます。

**オプトアウト:** `plugins/hermes-achievements/dashboard/manifest.json` を削除または名前変更するか、ダッシュボードを出荷しない同名のユーザープラグインを `~/.hermes/plugins/hermes-achievements/` に置いて上書きします。`$HERMES_HOME/plugins/hermes-achievements/` 配下のプラグインの状態ファイルは残ります。再インストールしてもアンロック履歴は保持されます。

## 同梱プラグインを追加する

同梱プラグインは、他のあらゆるHermesプラグインとまったく同じように書かれます。[Hermesプラグインを構築する](/docs/guides/build-a-hermes-plugin)を参照してください。唯一の違いは次のとおりです。

- ディレクトリが `~/.hermes/plugins/<name>/` ではなく `<repo>/plugins/<name>/` にある
- マニフェストのソースが `hermes plugins list` で `bundled` として報告される
- 同名のユーザープラグインが同梱版を上書きする

プラグインが同梱の良い候補となるのは、次の場合です。

- オプションの依存関係がない（または、それらがすでに `pip install .[all]` の依存関係である）
- その動作がほとんどのユーザーにとって有益であり、オプトインではなくオプトアウトである
- ロジックが、エージェントがそうでなければ呼び出すのを覚えておく必要があるライフサイクルフックに結びついている
- モデルから見えるツール面を拡大せずに、コア機能を補完する

反例 — 同梱ではなく、ユーザーがインストール可能なプラグインのままにすべきもの: APIキーを伴うサードパーティ連携、ニッチなワークフロー、大きな依存関係ツリー、デフォルトでエージェントの動作を大きく変えるもの。
