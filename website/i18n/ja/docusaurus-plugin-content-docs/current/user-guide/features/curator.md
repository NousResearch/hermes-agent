---
sidebar_position: 3
title: "Curator"
description: "エージェントが作成したスキルのバックグラウンド保守 — 利用状況の追跡、陳腐化、アーカイブ、LLMによるレビュー"
---

# Curator

Curatorは、**エージェントが作成したスキル**に対するバックグラウンド保守パスです。各スキルが閲覧・使用・パッチされた頻度を追跡し、長期間使われていないスキルを `active → stale → archived` の状態へと移行させ、定期的に短い補助モデルによるレビューを起動して、統合の提案やドリフトのパッチを行います。

これは、[自己改善ループ](./skills.md#agent-managed-skills-skill_manage-tool)を通じて作成されたスキルが際限なく積み上がらないようにするために存在します。エージェントが新しい問題を解決してスキルを保存するたびに、そのスキルは `~/.hermes/skills/` に保存されます。保守を行わないと、カタログを汚染しトークンを浪費する、わずかに異なる狭い範囲の準重複スキルが数十個も溜まってしまいます。

デフォルトでは、Curatorはエージェントが作成したスキルのみを管理します。`curator.prune_builtins: true` を設定すると、**使われていないバンドルの組み込みスキル**（リポジトリに同梱されたもの）も、`archive_after_days` の間使われなかった後にアーカイブできるようになります。同梱スキルが `skills_list` から黙って消えると、インストールの破損と誤解されやすいため、これはオプトインになっています。ハブからインストールされたスキル（[agentskills.io](https://agentskills.io) 由来のもの）は常に対象外です。また、Curatorは**自動削除も一切行いません** — 最悪の結果でも `~/.hermes/skills/.archive/` へのアーカイブにとどまり、これは復元可能です。

[issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) で追跡されています。

## 動作の仕組み

Curatorは、cronジョブではなく非アクティブチェックによってトリガーされます。CLIセッションの開始時、ゲートウェイのハウスキーピング中（ローカルのDesktopはゲートウェイにアタッチするため、これも対象になります）、および `hermes serve` のメンテナンスタイマーで、Hermesは次の条件を確認します。

1. 最後のCurator実行から十分な時間が経過しているか（`interval_hours`、デフォルト**7日**）、そして
2. エージェントが十分な時間アイドル状態であるか（`min_idle_hours`、デフォルト**2時間**）。

`hermes serve` / `hermes dashboard` のバックエンド（例えばリモートのDesktop接続）は、cronジョブとは独立に、既存の1時間ごとのメンテナンスタイマー（最初のポーリングは90秒後）を共有します。これらは、プロセスの起動時刻と、同じプロファイル内での直近のチャットアクティビティから非アクティブ時間を測定し、セッションが閉じられたり回収されたりした後もそのアクティビティのタイムスタンプを保持し、そのプロファイルでターンが実行中の間はCuratorをスキップします。接続されているが非アクティブなウィンドウは、メンテナンスをブロックしません。同じプロファイルに対してメッセージングゲートウェイが実行中の場合は、代わりにそちらがこれらの作業を担当します。

タイマーは、そのバックエンドのプロファイルを対象に処理します。実行中のメンテナンスはワーカースレッドで動作し、バックエンドを閉じてもそのパスが協調的に中断されることはありません。同じプロファイルに対して複数の独立したserveプロセスを起動すると、Curatorのインターバルチェックが競合する可能性は依然としてあります。

両方が真であれば、`AIAgent` のバックグラウンドフォークを起動します — これはメモリ/スキルの自己改善ナッジで使われるものと同じパターンです。このフォークは独自のプロンプトキャッシュ内で実行され、アクティブな会話には一切触れません。

:::info 初回実行時の挙動
新規インストール直後（または `hermes update` 後にCurator導入前のインストールが初めてティックする場合）、Curatorは**すぐには実行されません**。最初の観測で `last_run_at` を「現在」にシードし、最初の本番パスを `interval_hours` の1周期分だけ先送りします。これにより、Curatorが手を触れる前に、スキルライブラリをレビューし、重要なものをピン留めしたり、完全にオプトアウトしたりするための1周期分の余裕が得られます。

Curatorが本番実行する前に*何をしようとしているか*を確認したい場合は、`hermes curator run --dry-run` を実行してください — ライブラリを変更することなく、同じレビューレポートを生成します。
:::

1回の実行には2つのフェーズがあります。

1. **自動遷移**（決定論的、LLM不要）。`stale_after_days`（14）の間使われていないスキルは `stale` になり、`archive_after_days`（30）の間使われていないスキルは `~/.hermes/skills/.archive/` に移動されます。これは常時有効な間引きの挙動です — Curatorが有効であれば常に実行され、補助モデルのコストはかかりません。
   - **ピン留めされたスキル**と**いずれかのcronジョブから参照されているスキル**（一時停止中/無効化されたジョブを含む）は完全にスキップされます — 自動遷移においてはピン留めと同様に扱われるため、遅いスケジュールや一時停止中のスケジュールのせいで、ジョブが使うスキルがアーカイブされてしまうことはありません。統合時にアンブレラへマージする際には、cronのスキル参照も書き換えられます。
   - **一度も使われていないスキル**（`use_count == 0`）には猶予期間があります: 少なくとも `stale_after_days` が経過するまではアーカイブされません。使用回数ゼロは証拠がないということであり、そのスキルが不要であることの証明ではありません。
2. **LLMによる統合**（高いイテレーション上限を持つ補助モデルによる単一パス — 完全なキュレーションスイープには通常50〜100回のAPI呼び出しがかかります） — **デフォルトではOFF**です。`curator.consolidate: true` の場合、フォークされたエージェントがエージェント作成スキルを調査し、`skill_view` でそれらのいずれかを読み、スキルごとに維持・パッチ（`skill_manage` 経由）・重複するスキルのクラスレベルのアンブレラへの統合・アーカイブ（`absorbed_into` のターゲットを指定した `skill_manage action=delete` 経由 — このパスにはterminalがありません）のいずれを行うかを決定します。渡される候補リストには、実際に読み書きできるスキルのみが含まれます: バンドルの組み込みスキル（`prune_builtins: true` であっても同様で、これは決定論的なアーカイブパスにのみ影響します）と `skills.disabled` 内のスキル（`skill_view` が拒否します）は決して提示されないため、このパスが拒否される読み書きでツール予算を浪費することはありません。統合ではスキルを1つの完全なパッケージとして扱います: スキルに `references/`、`templates/`、`scripts/`、`assets/`、またはそれらのパスへの相対リンクがある場合、Curatorはそれを単独のまま維持するか、必要なサポートファイルを移設してパスを書き換えるか、パッケージ全体を変更せずにアーカイブするかのいずれかを行わなければなりません — `SKILL.md` だけを別のスキルの `references/` ファイルに平坦化してはいけません。

:::info 統合はオプトイン
デフォルトでは、Curatorは**間引き**のみを行います — 決定論的な非アクティブパスがスキルをstaleとしてマークし、長期間使われていないものをアーカイブします。主観を伴うLLMによる**統合**パス（アンブレラの構築、重複するスキルのマージ）は、実行のたびに補助モデルのトークンを消費し、ライブラリに広範な構造変更を加えるため、デフォルトではオフです。`curator.consolidate: true` で有効にするか、`hermes curator run --consolidate` でオンデマンドに1回だけ実行してください。
:::

ピン留めされたスキルはCuratorの自動遷移の対象外となり、エージェントの `skill_manage` ツールはそれらの**削除**を拒否します。編集やパッチは引き続き行えます。エージェントにスキルを一切変更させたくない場合は、代わりにロックしてください。下記の[スキルのピン留め](#pinning-a-skill)と[スキルのロック](#locking-a-skill)を参照してください。

## 設定

すべての設定は `config.yaml` の `curator:` 配下に置かれます（`.env` ではありません — これは秘密情報ではないため）。デフォルト値は次のとおりです。

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 days
  min_idle_hours: 2
  stale_after_days: 14
  archive_after_days: 30
  consolidate: false           # LLM umbrella-building pass — opt-in (prune-only by default)
  prune_builtins: false        # opt in to archiving unused bundled built-in skills too (hub skills always exempt)
```

完全に無効化するには、`curator.enabled: false` を設定します。常時有効な間引きはそのままに、LLMによる統合をオプトインするには、`curator.consolidate: true` を設定します。

### より安価な補助モデルでレビューを実行する

CuratorのLLMレビューパスは、Vision、Compression、Session Searchなどと並ぶ通常の補助タスクスロット（`auxiliary.curator`）です。「Auto」は「メインのチャットモデルを使う」という意味です。レビューパスに特定のプロバイダー＋モデルをピン留めするには、このスロットを上書きしてください。

**最も簡単 — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

同じピッカーは、Webダッシュボードの**Models**タブからも利用できます。

**config.yaml を直接編集（同等）:**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

`provider: auto`（デフォルト）のままにすると、他のすべての補助タスクと同じ挙動で、レビューパスはメインのチャットモデルを通じてルーティングされます。

:::note レガシー設定
以前のリリースでは、一度限りの `curator.auxiliary.{provider,model}` ブロックを使用していました。このパスは引き続き機能しますが、非推奨のログ行を出力します — Curatorが他のすべての補助タスクと同じ仕組み（`hermes model`、ダッシュボードのModelsタブ、`base_url`、`api_key`、`timeout`、`extra_body`）を共有するよう、上記の `auxiliary.curator` への移行をお願いします。
:::

## CLI

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a run now (blocks until done). Prune-only unless curator.consolidate: true
hermes curator run --consolidate # force the LLM consolidation pass on for this run, overriding the config default
hermes curator run --background  # fire-and-forget: start the run in a background thread
hermes curator run --dry-run  # preview only — report without any mutations
hermes curator backup         # take a manual snapshot of ~/.hermes/skills/
hermes curator rollback       # restore from the newest snapshot
hermes curator rollback --list     # list available snapshots
hermes curator rollback --id <ts>  # restore a specific snapshot
hermes curator rollback -y         # skip the confirmation prompt
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator lock <skill>   # the agent may not change it
hermes curator unlock <skill>
hermes curator adopt <skill>    # hand an unmanaged skill to the curator
hermes curator adopt --all-unmanaged   # hand over every unmanaged skill
hermes curator list-unmanaged   # itemize skills with no provenance marker
hermes curator restore <skill>  # move an archived skill back to active
hermes curator list-archived    # list skills currently in ~/.hermes/skills/.archive/
hermes curator archive <skill>  # manually archive a single skill now
hermes curator prune [--days N] # bulk-archive agent-created skills idle >= N days (default: `archive_after_days`, 30)
hermes curator ledger           # list the per-mutation audit ledger (all actors)
hermes curator ledger --skill <name> --limit 50  # filter/paginate ledger entries
hermes curator rollback <entry-id>  # undo a single mutation from the ledger
hermes curator purge [--days N] [--dry-run]  # delete archived skills older than the TTL (explicit only)
```

## バックアップとロールバック

統合パス（`consolidate: true`、スキルの内容をその場で書き換える唯一のパス）の前に、Hermesは `~/.hermes/skills/` のtar.gzスナップショットを `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz` に取得します。スナップショットの対象はライブのスキルツリーのみです: `.archive/`、監査台帳、`.hub/`、およびバックアップ自体は決して含まれず、ロールバックでそれらが巻き戻されることもありません（古いコピーではアーカイブ済みスキルや台帳エントリが失われるため）。あるパスが、手を触れてほしくなかったものをアーカイブまたは統合してしまった場合、1つのコマンドで実行全体を取り消せます。

```bash
hermes curator rollback        # restore newest snapshot (with confirmation)
hermes curator rollback -y     # skip the prompt
hermes curator rollback --list # see all snapshots with reason + size
```

ロールバック自体も可逆です。スキルツリーを置き換える前に、Hermesは `pre-rollback to <target-id>` というタグの付いた別のスナップショットを取得するため、誤ったロールバックは `--id` でそのスナップショットへロールフォワードすることで取り消せます。

`hermes curator backup --reason "before-refactor"` を使えば、いつでも手動でスナップショットを取得できます。`--reason` 文字列はスナップショットの `manifest.json` に記録され、`--list` で表示されます。

デフォルトの間引きのみのパスではスナップショットを取得しません: このパスはディレクトリ全体を `.archive/` に移動するだけであり、それ自体が取り消し手段（`hermes curator restore`）になっていて、すべての変更は下記の台帳に記録されるためです。スナップショットは、ディスク使用量を抑えるために、パスのたびに `curator.backup.keep`（デフォルト2）まで間引かれます。

```yaml
curator:
  backup:
    enabled: true
    keep: 2
```

`curator.backup.enabled: false` を設定すると、自動スナップショットを無効化できます。バックアップが無効化されているときに手動の `hermes curator backup` コマンドが機能するのは、先に `enabled: true` を設定した場合のみです — このフラグは両方のパスを対称的にゲートするため、変更を伴う実行で本番前スナップショットを誤ってスキップする方法はありません。

`hermes curator status` は、最も長く使われていないスキル5件も一覧表示します — 次に陳腐化しそうなものを素早く把握する方法です。

同じサブコマンドは、実行中のセッション内（CLIまたはゲートウェイプラットフォーム）で `/curator` スラッシュコマンドとしても利用できます。

## 監査台帳と単一編集のロールバック

実行全体のスナップショットは「直前のCuratorパスが行ったことをすべて取り消す」ためのものですが、*誰が何を変更したか*を知り、ちょうど1つの変更だけを取り消したい場合もあります。すべてのスキル変更 — Curatorの自動遷移、エージェントの `skill_manage` 呼び出し、そしてあなた自身のCLIによるarchive/restore/purge — は、`~/.hermes/skills/.curator_ledger.jsonl` にある追記専用のJSONL台帳に1エントリを追加します。

- **actor** — `curator`（バックグラウンドレビューのフォーク/自動遷移）、`agent`（フォアグラウンドのエージェントのツール呼び出し）、または `user`（CLIコマンド）
- **action** — `create`、`edit`、`patch`、`delete`、`write_file`、`remove_file`、`archive`、`restore`、`purge`、`rollback`
- **evidence** — 削除の意図（統合の場合は `absorbed_into`、間引きの場合は空、および復元可能なアーカイブのパスで処理されたかどうか）、利用可能な場合はトリガーとなったセッションID
- **before/after** — ファイルごとの `{path, sha256}` マニフェスト。ファイルの内容は `~/.hermes/.curator_backups/blobs/` 配下にコンテンツアドレス方式（ハッシュで重複排除）で保存されるため、同じ未変更ファイルに触れる100件のエントリでもblobは1つで済みます。

```bash
hermes curator ledger                  # newest 20 entries
hermes curator ledger --skill my-skill --limit 50
hermes curator rollback <entry-id>     # restore that one mutation's before-state
```

単一エントリのロールバックは、その変更が触れたファイルだけを（作成されたファイルは削除して）blobストアから正確に復元します — スキルツリー内のそれ以外のものは一切動きません。ツリー全体のロールバックと同様に、まず現在の状態の安全用台帳エントリを取得し、**フェイルクローズ**で動作します: 安全用のキャプチャを書き込めない場合、何も変更されません。フォアグラウンドの削除も台帳に記録されるため、`hermes curator rollback <entry-id>` でハード削除されたスキルを復活させることもできます。

台帳はテレメトリであり、決してゲートではありません — エントリの書き込みに失敗しても、変更はそのまま実行されます。無効化するには次のようにします。

```yaml
skills:
  ledger: false
```

このファイルにはサイズの上限もあります: `skills.ledger_max_bytes`（デフォルト5 MB）を超えると、次の変更時にまず未変更ファイルの重複排除によって書き直され（`hermes curator ledger --compact` が行うことと全く同じです）、それでも実際に差異のあるエントリが上限を超える場合は、形式に関係なく最も古いものから削除されます — 最新のエントリは常に残り、保持される末尾部分の行が書き換えられたりパースされたりすることはなく（そこにある不正な形式の行もそのまま残ります）、このスイープでは、もはや何からも参照されておらず、かつ1時間より古いblobが解放されます（参照されていない新しいblobは、別のプロセスがまだ記録していないキャプチャに属している可能性があるため）。`0` に設定すると、台帳を永久に追記専用のまま維持します。

```yaml
skills:
  ledger_max_bytes: 5242880   # 0 = never auto-maintain
```

## アーカイブのTTLパージ

アーカイブされたスキルは、デフォルトでは永久に保持されます。`~/.hermes/skills/.archive/` のサイズを制限したい場合は、TTLを設定して明示的にパージしてください — パージが自動で実行されることはなく、パージされるすべてのスキルは先に（blobとともに）台帳に記録されるため、パージであっても監査可能で復元可能な痕跡が残ります。

```yaml
curator:
  archive_ttl_days: 180   # 0 (default) = never purge
```

```bash
hermes curator purge --dry-run   # preview what would be deleted
hermes curator purge             # delete archives older than the TTL (with confirmation)
hermes curator purge --days 90   # one-off TTL override
```

## 「エージェント作成」の意味

Curatorは、`~/.hermes/skills/.usage.json` で明示的に**エージェント作成**としてマークされたスキルのみを管理します。
次のすべてが真である場合に、スキルは対象となります。

1. その名前が `~/.hermes/skills/.bundled_manifest`（リポジトリに同梱されたバンドルスキル）に**含まれていない**。
2. その名前が `~/.hermes/skills/.hub/lock.json`（ハブインストールのスキル）に**含まれていない**。
3. その `.usage.json` エントリに `"created_by": "agent"` または `"agent_created": true` がある。

現在、このマーカーを設定するのは**バックグラウンドの自己改善レビューフォーク**のみです
— 定期的なレビューパス（約10エージェントターンごと）の中で新しいアンブレラスキルを作成する場合です。
バックグラウンドフォークは `"background_review"` という書き込みオリジン（`tools/skill_provenance.py` 経由）で
実行され、これが `skill_manage` 内の `mark_agent_created()` 呼び出しをトリガーする唯一のパスです。

会話中にフォアグラウンドのエージェントが `skill_manage(action="create")` で作成したスキル
（`/learn` を含む）は、エージェント作成として**マークされません** — それらは `created_by: learn` として
記録され、これにより[学習の軌跡](./memory.md#learning-journey-journey)にすぐ表示されますが、
Curatorへのオプトインではありません。それらはユーザーの指示によるものと見なされ、Curatorは意図的に手を触れません。

:::warning 手書きのスキルはキュレーションされません
手動で `SKILL.md` を作成した場合や、Hermesに外部スキルディレクトリを指定した場合、
そのスキルの `.usage.json` エントリは `created_by: null`（またはフィールドなし）になります。
Curatorはそれに手を触れません。あなたの依頼でフォアグラウンドのエージェントが作成したスキル
（`created_by: learn`）も同様です。

**Curatorが実際に管理しているスキルを確認するには**、`hermes curator status` を実行してください。
エージェント作成のカウントが0の場合、現在Curatorの管轄下にあるスキルはありません
— LLMレビューパスはスキップされ、レポートには `Duration: 0s` とともに
`Model: (not resolved) via (not resolved)` と表示されます。
:::

### 管理外のスキルの引き受け（adopt）

`hermes curator status` は、管理対象のカウントと並んで**管理外（unmanaged）**のカウントも
報告します。

```
curator-managed skills: 43 total  (agent-created=43  bundled=0)
  active     41
  stale       2
  archived    0

unmanaged (no provenance marker): 112 total
  pre-dates marker    34
  foreground-created  78
  never auto-staled or archived — `hermes curator adopt <name>` hands one over
```

この112件はキュレーションの*対象になりうる*ものの、次の2つの理由のいずれかにより、
ライフサイクルからは永久に見えない状態にあります。

- **pre-dates marker** — `created_by` が存在する前に記録が書き込まれたため、
  出自を示すシグナルを一切持っていません。記録から作成者を知ることは本質的に
  不可能です。
- **foreground-created** — フォアグラウンドの `skill_manage(create)` が設計どおり
  `created_by: learn`（古い記録では未設定）を記録したものです。あなたが依頼したスキルは
  あなたのものだからです。

そのため、大きなライブラリは、その大部分が手を付けられない状態であるにもかかわらず、
完全にキュレーションされているように見えることがあります。`adopt` は、**宣言**によってそのギャップを埋めます。

```bash
hermes curator list-unmanaged                    # itemize them, with reasons
hermes curator adopt <name> [<name> ...]         # hand specific skills over
hermes curator adopt --all-unmanaged --dry-run   # preview the full list
hermes curator adopt --all-unmanaged             # hand over everything (prompts)
hermes curator adopt --all-unmanaged --yes       # skip the prompt
```

adoptは、バックグラウンドレビューフォークが書き込むのと同じ `created_by: agent` マーカーを
書き込みます。非アクティブ時間のカウントは**リセットされません** — adoptされたスキルは既存の
`last_activity_at` を保持するため、すでに使わなくなったライブラリを引き渡しても、新たな90日間の猶予が
得られるわけではありません。長期間アイドル状態だったスキルは、adopt後の次のパスで `stale`（または `archived`）に
なると考えてください。それこそが狙いです。

adoptが左右するのは*アーカイブ*のみです。バックグラウンドの自己改善レビューは、学習元となる
あらゆるスキル（手書き、URLインストール、バンドル、ハブインストール、外部、ピン留め済み）に対して、
パッチ、書き換え、サポートファイルの追加を行えます。そのため、誰が書いたかに関係なく、教訓はそのタスクを
司るスキルに反映されます。そのような書き込みはすべてスキル台帳に記録されます。アーカイブ（削除）できるのは
Curator管理下のスキルのみで、ピン留めされたスキルは決して対象になりません。それらの書き込みを反映前にレビュー
したい場合は、`skills.write_approval` を有効にしてください（[エージェントのスキル書き込みのゲート](./skills.md#gating-agent-skill-writes-skillswrite_approval)を参照）。
レビューが編集したバンドルまたはハブインストールのスキルはローカルで変更されたものと見なされるため、
`hermes update` / `hermes skills update` はそれを上書きしなくなります。

:::note `created_by` は出自の主張ではなくポリシーフラグです
保存されるフィールド名は `created_by` ですが、これは「このファイルを誰が書いたか」ではなく、
「自律的なキュレーションがこれに手を触れてよいか」として扱われます。これらは別の問いであり、
マーカー以前の記録については、作成者の答えは単純に復元不可能です。この名前は、すでにすべての
`.usage.json` のディスク上に存在するため維持されています。ポリシーとして読んでください。
`hermes curator adopt` はポリシーを変更するものであり、ファイルの作成者については何も述べません。
:::

:::note 出自は推測されず、常に宣言されます
adoptは意図的に手動になっています。テレメトリでは作成者を確定できません: 何千回もパッチされたスキルは、
エージェントがそれを**保守**していることを証明しますが、エージェントがそれを**書いた**ことは証明しません
— Hermesはあなたに代わってユーザー作成のスキルを常に編集しています。「エージェントが作ったように見えるのでadoptする」
という自動ヒューリスティックは、いずれあなたが手書きしたものをアーカイブしてしまうでしょう。`adopt` は、
あなた以外の所有者を持つバンドル、ハブインストール、外部、および保護された組み込みスキルを拒否します。
:::

エージェント作成のスキルは、完全なライフサイクルに従います。

- `active` →（30日未使用）`stale` →（90日未使用）`archived`
- ピン留めされたスキルはすべての自動遷移を回避します
- アーカイブは `hermes curator restore <name>` で復元可能です

エージェント作成スキルをアーカイブから保護したい場合は、
`hermes curator pin <name>` を使ってください。エージェントにスキルを一切変更させたくない場合 —
例えば頼りにしている手書きのスキルなど — は、`hermes curator lock <name>` を使ってください。
どちらも以下で説明します。

## スキルのピン留め {#pinning-a-skill}

ピン留めは、スキルを削除から保護します — Curatorの自動アーカイブパスと、エージェントの `skill_manage(action="delete")` ツール呼び出しの両方からです。スキルがピン留めされると、次のようになります。

- **Curator**は、自動遷移（`active → stale → archived`）の際にそれをスキップし、LLMレビューパスはそれをそのまま残すよう指示されます。
- **エージェントの `skill_manage` ツール**は、それに対する `delete` を拒否し、ユーザーに `hermes curator unpin <name>` を案内します。パッチや編集は引き続き行えるため、エージェントはピン留め/解除/再ピン留めの手間をかけずに、落とし穴が見つかったときにピン留めされたスキルの内容を改善できます。

ピン留めと解除は次のように行います。

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

このフラグは、`~/.hermes/skills/.usage.json` のスキルエントリに `"pinned": true` として保存されるため、セッションをまたいで保持されます。

いずれかのcronジョブの `skills:` リストに名前があるスキルは、**自動遷移**について同じように保護されます（参照が残っている間、Curatorはそれらをstale/アーカイブしません）。これはジョブが一時停止中または無効化されている場合でも同様です。`skill_manage delete` もブロックしたい場合は、明示的なピン留めを優先してください。

ピン留めできるのは**エージェント作成**スキルのみです — バンドルおよびハブインストールのスキルに対して `hermes curator pin` を試みると、説明付きのメッセージとともに拒否されます。ハブインストールのスキルは、決してCuratorの変更対象になりません。バンドルの組み込みスキルは、`curator.prune_builtins: true` でオプトインした場合にのみ扱われ、その場合でも `archive_after_days` の間使われなかった後にアーカイブされるだけです — パッチ、統合、削除されることは決してありません。

少数の**保護された組み込みスキル**は、`curator.prune_builtins`、ピン留めの状態、LLMの判断に関係なく、アーカイブも統合も決してされないものとしてハードコードできます。これらは重要なUXを支えるものであり、黙ってアーカイブされると、そのスラッシュコマンドが何の兆候もなく「Unknown command」エラーになってしまいます。（このセットは現在空です — 元のメンバーだった `plan` は、ディスク上にスキルを持たない組み込みの `/plan` コマンドに昇格しました。）保護された組み込みスキルはCuratorの候補リストから完全に除外されるため、統合パスがそれらを目にすることはありません。

「削除しない」より強い保証 — エージェントが引き続き読み込んで使用する一方で、スキルの内容を凍結すること — が必要な場合は、ロックしてください（次のセクション）。ピン留めもロックも、あなた自身の編集を妨げることはありません: お使いのエディタやDesktopのスキルエディタは常に機能します。

## スキルのロック {#locking-a-skill}

ピン留めはアーカイブを止めます。ロックは**エージェントによるスキルの変更を一切**止めます。その間もスキルは読み取り可能で、スキルインデックスに掲載され、自動読み込み可能で、これまでとまったく同じように使用できます。

```bash
hermes curator lock <skill>
hermes curator unlock <skill>
hermes curator status          # lists locked skills
```

Desktopでは、**Capabilities → Skills** を開き、スキルを選択してそのロックスイッチを使ってください。

スキルがロックされると、エージェントが書き込むあらゆる手段がそれを拒否します。これは通常のターン、cronジョブ、サブエージェント、バックグラウンドの自己改善レビュー、Curatorパスのいずれでも同様で、`--yolo` の下でも変わりません。

- **`skill_manage`** は、単一操作とバッチ形式の両方で、`patch`、`edit`、`write_file`、`remove_file`、`delete` を拒否します。
- **`write_file` と `patch`** は、スキルのディレクトリ内のあらゆるパス（`SKILL.md`、`references/`、`scripts/`、…）を拒否します。
- **`terminal`** は、そこに書き込むコマンドを拒否します: リダイレクトと `tee`、`cp`/`mv`/`rm`/`touch`/`mkdir`/`chmod`、`sed -i`/`perl -i`、スキルのツリー内での `git` による変更、そこへのアーカイブ展開、そしてそれを名指しするインラインの `python -c`/`bash -c` コード（スキル内へ `cd` した後も含む）です。スキルの読み取りとそのスクリプトの実行は引き続き機能します。
- Curatorはロックされたスキルを決してアーカイブしません。
- エージェントはロックを解除できません: `terminal` は `hermes curator unlock` を拒否し、いずれかのスキルがロックされている間は、ロックの記録（`skills/.usage.json`）も同じように保護されます。拒否メッセージは、エージェントに望んでいた変更を説明するよう伝えるため、あなたが判断できます。

あなた自身の編集はゲートされません: エディタやDesktopのスキルエディタでファイルを編集することは常に機能します。エージェントに再び変更を許可するには、`hermes curator unlock <skill>` を実行するか、スイッチを切り替えてください。

このフラグは、`pinned` と同じサイドカーである `~/.hermes/skills/.usage.json` のスキルエントリに `"locked": true` として保存されるため、**プロファイルごと**になります: `hermes -p work curator lock deploy-runbook` は、そのプロファイルのコピーのみをロックします。プロファイル内で解決されるスキルであれば、どれでもロックできます（手書き、エージェント作成、バンドル、ハブインストール）。`lock` は解決できない名前を拒否します。

ロックではないもの:

- これは**ポリシーチェックであり、サンドボックスではありません。** terminalのチェックはコマンドの書き込み先を読み取ります。独自のロジックでスキルに書き込むスクリプトや `execute_code` は検査されません。厳格な境界が必要な場合は、ファイルシステムの権限と組み合わせてください。
- `.usage.json` が破損している場合、バックグラウンドのパスは拒否されます（フェイルクローズ）が、通常のターンは続行されるため、1つの壊れたファイルですべてのスキル編集が凍結されることはありません。修復するか、`hermes curator unlock <skill>` を実行して書き直してください。
- あなたが変更していないバンドルまたはハブのスキルを `hermes update` / `hermes skills update` が更新することは**止めません**。これらはあなたのコマンドであり、エージェントによる書き込みではないからです。

## 利用テレメトリ

Curatorは、`~/.hermes/skills/.usage.json` にサイドカーを保持し、スキルごとに1エントリを記録します。

```json
{
  "my-skill": {
    "use_count": 12,
    "view_count": 34,
    "last_used_at": "2026-04-24T18:12:03Z",
    "last_viewed_at": "2026-04-23T09:44:17Z",
    "patch_count": 3,
    "last_patched_at": "2026-04-20T22:01:55Z",
    "created_at": "2026-03-01T14:20:00Z",
    "state": "active",
    "pinned": false,
    "archived_at": null
  }
}
```

カウンターは次のときに増加します。

- `view_count`: エージェントがスキルに対して `skill_view` を呼び出したとき。
- `use_count`: スキルが会話のプロンプトに読み込まれたとき。
- `patch_count`: スキルに対して `skill_manage patch/edit/write_file/remove_file` が実行されたとき。

バンドルおよびハブインストールのスキルは、テレメトリの書き込みから明示的に除外されます。

## 実行ごとのレポート

すべてのCurator実行は、`~/.hermes/logs/curator/` 配下にタイムスタンプ付きのディレクトリを書き込みます。

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md` は、特定の実行が何をしたか — どのスキルが遷移し、LLMレビュアーが何と言い、どのスキルをパッチしたか — を素早く確認する方法です。`agent.log` をgrepすることなく監査するのに適しています。

:::note 候補がない場合、レポートには `(not resolved)` と表示されます
Curatorがレビューすべき**エージェント作成スキルを持たない**場合、LLMレビューパスは
完全にスキップされます。レポートのヘッダーには `Duration: 0s` とともに
`Model: (not resolved) via (not resolved)` と表示されます — これは設定エラーやモデル解決の失敗を
示すもの**ではありません**。単に候補がなかったため、モデルが一度も呼び出されなかったことを意味します。
自動遷移フェーズは引き続き実行され、そのカウントを通常どおり報告します。
:::

### サマリー内のリネームマップ

ある実行で複数のスキルがアンブレラの下に統合された（または準重複がマージされた）場合、実行の最後に表示されるユーザー向けのサマリーには、Curatorが適用したすべての `old-name → new-name` のペアを示す明示的なリネームマップが含まれます。これはスキルごとの遷移行に加えて表示されるため、大量のリネームが発生したときでも、JSONレポートの差分を取らずに一目で把握できます。このヒントは `hermes curator pin` の下にも表示されるため、新しいラベルを固定したい場合は、すぐにアンブレラ名をピン留めできます。

## アーカイブされたスキルの復元

Curatorがまだ必要なものをアーカイブしてしまった場合は、次のようにします。

```bash
hermes curator restore <skill-name>
```

これは、スキルを `~/.hermes/skills/.archive/` からアクティブツリーに戻し、その状態を `active` にリセットします。それ以降に同じ名前でバンドルまたはハブインストールのスキルがインストールされている場合（上流をシャドウすることになるため）、復元は拒否されます。

## 環境ごとの無効化

Curatorはデフォルトで有効です。オフにするには次のようにします。

- **1つのプロファイルのみの場合:** `~/.hermes/config.yaml`（またはアクティブなプロファイルの設定）を編集し、`curator.enabled: false` を設定します。
- **1回の実行のみの場合:** `hermes curator pause` — 一時停止はセッションをまたいで保持されます。再有効化するには `resume` を使います。

Curatorは `min_idle_hours` が経過していない場合も実行を拒否するため、アクティブな開発マシンでは、自然と静かな時間帯にのみ実行されます。

## 関連項目

- [スキルシステム](./skills.md) — スキルの一般的な仕組みと、それを作成する自己改善ループ
- [メモリ](./memory.md) — 長期メモリを保守する並行のバックグラウンドレビュー
- [バンドルスキルカタログ](../../reference/skills-catalog.md)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — 元の提案と設計の議論
