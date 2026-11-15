---
sidebar_position: 12
sidebar_label: "組み込みプラグイン"
title: "組み込みプラグイン"
description: "Hermes Agentに同梱され、ライフサイクルフック経由で自動的に実行されるプラグイン — disk-cleanupとその仲間"
---

# 組み込みプラグイン

Hermesには、リポジトリに同梱された少数のプラグインが付属しています。これらは `<repo>/plugins/<name>/` 配下にあり、`~/.hermes/plugins/` のユーザーインストール済みプラグインと並んで自動的に読み込まれます。これらはサードパーティプラグインと同じプラグイン面（フック、ツール、スラッシュコマンド）を使用し、ツリー内でメンテナンスされる点だけが異なります。

一般的なプラグインシステムについては[プラグイン](./plugins.md)ページを、独自のものを書くには[Hermesプラグインを構築する](../../developer-guide/plugins/index.md)を参照してください。

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
# or: remove it from plugins.enabled in config.yaml
```

## 現在出荷されているもの


リポジトリは `plugins/` 配下に次の同梱プラグインを出荷しています。すべてオプトインです。`hermes plugins enable <name>` で有効化してください。

| プラグイン | 種類 | 目的 |
|---|---|---|
| `disk-cleanup` | フック + スラッシュコマンド | 一時ファイルを自動追跡し、セッション終了時にクリーンアップする |
| `security-guidance` | フック | `write_file`/`patch` で危険なコードをパターンマッチし、セキュリティ警告を付加する（またはブロックする） — 25ルール（Anthropicの `claude-plugins-official` パターンのApache-2.0フォーク） |
| `teams_pipeline` | スタンドアロン | Microsoft Teamsミーティングパイプライン — Graphを基盤とした、トランスクリプト優先のミーティング要約 |
| `google_meet` | スタンドアロン | Meet通話への参加、ライブキャプションの文字起こし、任意のリアルタイム双方向音声 |
| `image_gen/openai` | 画像バックエンド | OpenAI GPT Image 2 および 2.5 Flare/Sunburst による画像の生成と編集（APIキー） |
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
| `post_tool_call` | `write_file` / `terminal` / `patch` が、`HERMES_HOME` または `/tmp/hermes-*` 内で `test_*`、`tmp_*`、`*.test.*` に一致するファイルを作成すると、それを `test` / `temp` / `cron-output` として静かに追跡する。 | <!-- no-tmp: ok — documents the plugin's scope guard -->
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
/disk-cleanup status                     # breakdown + top-10 largest
/disk-cleanup dry-run                    # preview without deleting
/disk-cleanup quick                      # run safe cleanup now
/disk-cleanup deep                       # quick + list items needing confirmation
/disk-cleanup track <path> <category>    # manual tracking
/disk-cleanup forget <path>              # stop tracking (does not delete)
```

**状態** — すべては `$HERMES_HOME/disk-cleanup/` にあります:

| ファイル | 内容 |
|---|---|
| `tracked.json` | カテゴリ、サイズ、タイムスタンプ付きの追跡パス |
| `tracked.json.bak` | 上記のアトミック書き込みバックアップ |
| `cleanup.log` | すべての追跡 / スキップ / 拒否 / 削除の追記専用監査証跡 |

<!-- no-tmp: ok — documents the plugin's scope guard -->
**安全性** — クリーンアップは `HERMES_HOME` または `/tmp/hermes-*` 配下のパスにのみ影響します。Windowsマウント（`/mnt/c/...`）は拒否されます。よく知られたトップレベルの状態ディレクトリ（`logs/`、`memories/`、`sessions/`、`cron/`、`cache/`、`skills/`、`plugins/`、`disk-cleanup/` 自体）は、空であっても決して削除されません。新規インストールが初回のセッション終了で台無しになることはありません。ユーザーのプロジェクトツリー（`workspace/`、`projects/`、`plans/`、`home/`）は、追跡もスイープもまったくされません。プロジェクト内の `test_*.py` や `tmp_*` ファイルはソースコードであり、スクラッチではないからです。`kanban/`（タスクの添付ファイルとワークスペース）も追跡されることはなく、`cache/` のような保護されたトップレベル配下で追跡された*ディレクトリ*が削除されることもありません。期限切れになるのはその中のファイルだけです。

**有効化:** `hermes plugins enable disk-cleanup`（または `hermes plugins` でチェックボックスをオンにする）。

**再び無効化:** `hermes plugins disable disk-cleanup`。

### security-guidance

ファイル書き込み時の、高速なパターンマッチによるセキュリティ警告です。エージェントの `write_file` / `patch` / `skill_manage` 呼び出しが、既知の危険なコードパターン（`pickle.load`、`SafeLoader` なしの `yaml.load`、`eval(`、`os.system`、`subprocess(...,  shell=True)`、JSの `child_process.exec`、Reactの `dangerouslySetInnerHTML`、生の `.innerHTML =` / `.outerHTML =` / `document.write`、Nodeの `crypto.createCipher`、AES ECBモード、TLS検証の無効化、XXEを起こしやすい `xml.etree` / `minidom` パーサー、SRIなしの `<script src="//..." >`、`weights_only=True` なしの `torch.load`、GitHub Actionsの `${{ github.event.* }}` インジェクション）に一致するコンテンツを含む場合、プラグインはツールの結果に `⚠️ Security guidance` ブロックを付加します。

ファイルは引き続き書き込まれます。モデルは次のターンのツールメッセージで警告を読み、コードを修正するか、その構文がこのコンテキストで安全である理由を文書化できます。パターンマッチングには無視できない誤検知率があるため、デフォルトはブロックではなく警告です。

**カバレッジ:** 合計25ルールで、安全でないデシリアライズ、コマンドインジェクション、XSSシンク、暗号の落とし穴、XXE、サプライチェーン（SRI）、CI/CDワークフローインジェクションをカバーします。パターンデータは[Anthropicの `claude-plugins-official`](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/security-guidance/hooks)をそのままApache-2.0でフォークしたものです。帰属表示については、プラグインの `LICENSE` および `NOTICE` ファイルを参照してください。

**モード:**

| 環境変数 | 効果 |
|---|---|
| （未設定） | **警告モード**（デフォルト） — ファイルは書き込まれ、警告が結果に付加される |
| `SECURITY_GUIDANCE_BLOCK=1` | **ブロックモード** — 書き込みが拒否され、警告がブロック理由として返される |
| `SECURITY_GUIDANCE_DISABLE=1` | キルスイッチ — プラグインは読み込まれるが何もしない |

**有効化:** `hermes plugins enable security-guidance`（または `hermes plugins` でチェックボックスをオンにする）。

**再び無効化:** `hermes plugins disable security-guidance`。

**（まだ）行わないこと:** 上流のAnthropicプラグインにはさらに2つのレイヤーがあります。ファイルに触れた各エージェントターンでのLLMによる差分レビューと、ファイルをまたいでデータフローを追跡するエージェント的なコミット時レビューです。どちらも移植されていません。エージェントは `delegate_task` を介して、これらのレビューをオンデマンドですでに実行できます。

### Langfuse（プラグインカタログに移動）

LangfuseトレーシングはHermesに同梱されなくなりました。同じフック、`HERMES_LANGFUSE_*`
環境変数、キャプチャモードを備えた公式の
[`langfuse`](https://github.com/NousResearch/hermes-official-plugins/tree/main/langfuse) カタログプラグインとして提供されています:

```bash
hermes plugins install langfuse
```

`observability/langfuse` を有効にしていたプロファイルでは、`hermes update`（または更新後の初回起動時）によって
プラグインが自動的にインストールされて有効化され、同じLangfuseプロジェクトへのトレースが継続されます。セットアップ、チューニング、キャプチャモードは
[プラグインのREADME](https://github.com/NousResearch/hermes-official-plugins/tree/main/langfuse)に記載されています。

### NeMo Relayネイティブ統合（移行に関する注意） {#nemo-relay-native-integration-migration-note}

NeMo Relayは同梱のHermesプラグインではなくなりました。`hermes plugins enable observability/nemo_relay` は実行しないでください。Relayのセッション、ターン、LLM、ツールのライフサイクルは、現在Hermesコアが管理しています。

Relayのミドルウェアやエクスポーターは、標準のRelay `plugins.toml` で設定します。HermesはRelayのユーザー設定（`~/.config/nemo-relay/plugins.toml`）を読み込み、次にマシン全体のシステム設定（`/etc/nemo-relay/plugins.toml`、Windowsでは `%ProgramData%\nemo-relay\plugins.toml`）を読み込みます。明示的なファイルでユーザー設定を置き換えたい場合にのみ、Hermesを起動する前に `HERMES_NEMO_RELAY_PLUGINS_TOML` を設定してください。その場合もシステム設定のほうが優先順位は高くなります。このポリシーはプロセス全体に適用され、そのHermesプロセスがホストするすべてのプロファイルに及びます。どのファイルが適用されるかは `hermes doctor` で確認できます。ATOF、ATIF、OpenTelemetryのオプションについては、[NeMo Relayの可観測性設定](https://docs.nvidia.com/nemo/relay/configure-plugins/observability/about)を参照してください。

従来の `HERMES_NEMO_RELAY_ATOF_*` および `HERMES_NEMO_RELAY_ATIF_*` 設定では、エクスポーターを設定できなくなりました。`HERMES_NEMO_RELAY_PLUGINS_TOML` が未設定の場合、ゲートウェイは残っている従来の変数について警告し、`hermes doctor` がそれらを報告します。独立して検出されたRelayのユーザーまたはシステムのエクスポーターは引き続き適用されます。

**自動移行。** `hermes update`（および `hermes migrate relay`、またはすべてのプロファイルホームを対象とする `hermes migrate relay --all-profiles`）は、従来の変数を `<hermes home>/relay-plugins.toml` に変換し、そのプロファイルの `.env` に `HERMES_NEMO_RELAY_PLUGINS_TOML` を設定し、従来の行をコメントアウトします（何も削除されません）。多重化されたゲートウェイでは、各プロファイルホームにそれぞれのファイルが作成されます。生成されたファイルは書き込まれる前にRelayで検証されます。生成される形は次のとおりです（`type = "file"` のシンク識別子に注意してください。これがないシンクは拒否されます）:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4
enable_full_payloads = false

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "/home/you/.hermes/telemetry/nemo-relay/atof"
filename = "hermes-atof.jsonl"
mode = "append"

[components.config.atif]
enabled = true
agent_name = "Hermes Agent"
model_name = "unknown"
output_directory = "/home/you/.hermes/telemetry/nemo-relay/atif"
filename_template = "trajectory-{session_id}.json"

[components.config.opentelemetry]
enabled = false
endpoints = []

[components.config.policy]
unknown_component = "warn"
unknown_field = "warn"
unsupported_value = "error"
```

その後、`.env` に `HERMES_NEMO_RELAY_PLUGINS_TOML=/home/you/.hermes/relay-plugins.toml` を追加し、ゲートウェイを再起動します。

#### セッションスパンの分割（継続セッション）

Relayはスコープが閉じたときにスパンをエクスポートします。継続的なゲートウェイセッションでは、各ターンのスパンが通常どおりエクスポートされていても、セッションスパンが何日も開いたままになることがあります。オプションの分割機能は、ターンの境界でセッションスコープだけをローテーションします:

```yaml
gateway:
  telemetry:
    session_segments:
      on_compaction: false  # rotate after context compaction
      max_turns: 0          # 0 = unlimited; N = turns per segment
```

| キー | デフォルト | 動作 |
|---|---:|---|
| `on_compaction` | `false` | 圧縮の完了後、次のターン境界でローテーションする。 |
| `max_turns` | `0` | N回のターンが完了するごとにローテーションする。`0` で上限を無効化。 |

どちらのデフォルトも、セッション全体で1つのセッションスコープを維持します。ローテーションされたスパンは同じ `session_id` を保持し、`hermes.session.segment` と `hermes.session.segment_reason`（`compaction` または `max_turns`）を追加します。

### google_meet

エージェントが**Google Meet通話に参加し、文字起こしし、参加できる**ようにします。ミーティングのメモを取り、後でやり取りを要約し、特定のポイントをフォローアップし、（任意で）TTS経由で通話に返答を発話します。

**追加される機能:**

- ブラウザ自動化を使ってMeet URLに参加するヘッドレスの仮想参加者
- Meet自身のライブキャプションから得られるライブ文字起こし（ボットはミーティング音声をデコードしないため、STTの課金は発生しません。ただし、キャプションは情報が欠落しやすく、英語に偏っています）
- 通話への参加、ライブトランスクリプトのポーリング、聞いた内容に基づく動作のためにエージェントが呼び出す `meet_join` / `meet_status` / `meet_transcript` / `meet_leave` / `meet_say` ツールセット
- `~/.hermes/workspace/meetings/<meeting_id>/` 配下に保存されるミーティング後のアーティファクト（トランスクリプト、ステータス）

**セットアップ:**

```bash
hermes plugins enable google_meet
hermes meet setup   # preflight: playwright, chromium, auth file
hermes meet auth    # opens a browser to sign into Google and saves session state —
                    # needs a Google account with Meet access. Host approval may be
                    # required if the meeting enforces "only invited participants can join".
```

チャットからの使い方:

> 「meet.google.com/abc-defg-hij に参加してメモを取って。通話後、アクションアイテム付きの要約を送って。」

エージェントはミーティングへの参加を開始し、通話の進行に合わせて文字起こしをコンテキストにストリーミングし、ミーティングが終了したとき（または停止を指示したとき）に構造化された要約を生成します。

**リアルタイムモード（`mode='realtime'`）は、音声面では発話専用です。** ボットの返答はOpenAI Realtimeで合成され、仮想マイクを通じて通話に再生されます。ボットが*聞いている*のは引き続きキャプションストリームであり、ミーティング音声ではありません。通話の内容がRealtimeセッションに送られることはありません。`meet_status` は `micState`（`unmuted`、参加承認後にボットが自分でミュートを解除する必要があった場合は `unmuted_clicked`、Meetのトグルが見つからなかった場合は `unknown`）を報告するため、ボットが無音の場合に原因を診断できます。

**いつ使うか:** 非同期の参加者のためにボットに文字起こし + 要約をしてほしい定例スタンドアップ、構造化されたメモが欲しい証言録取スタイルのインタビュー、Fireflies / Otter / Grain が必要になるあらゆるケース。AIに聞かれたくない場合は有効化しないでください。

**無効化:** `hermes plugins disable google_meet`。保存されたトランスクリプトは、削除するまで `~/.hermes/workspace/meetings/` に残ります。

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

同梱プラグインは、他のあらゆるHermesプラグインとまったく同じように書かれます。[Hermesプラグインを構築する](../../developer-guide/plugins/index.md)を参照してください。唯一の違いは次のとおりです。

- ディレクトリが `~/.hermes/plugins/<name>/` ではなく `<repo>/plugins/<name>/` にある
- マニフェストのソースが `hermes plugins list` で `bundled` として報告される
- 同名のユーザープラグインが同梱版を上書きする

プラグインが同梱の良い候補となるのは、次の場合です。

- オプションの依存関係がない（または、それらがすでに宣言済みの `all` extra に含まれている）
- その動作がほとんどのユーザーにとって有益であり、オプトインではなくオプトアウトである
- ロジックが、エージェントがそうでなければ呼び出すのを覚えておく必要があるライフサイクルフックに結びついている
- モデルから見えるツール面を拡大せずに、コア機能を補完する

反例 — 同梱ではなく、ユーザーがインストール可能なプラグインのままにすべきもの: APIキーを伴うサードパーティ連携、ニッチなワークフロー、大きな依存関係ツリー、デフォルトでエージェントの動作を大きく変えるもの。
