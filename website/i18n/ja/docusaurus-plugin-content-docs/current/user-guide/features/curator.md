---
sidebar_position: 3
title: "Curator"
description: "エージェントが作成したスキルのバックグラウンド保守 — 利用状況の追跡、陳腐化、アーカイブ、LLMによるレビュー"
---

# Curator

Curatorは、**エージェントが作成したスキル**に対するバックグラウンド保守パスです。各スキルが閲覧・使用・パッチされた頻度を追跡し、長期間使われていないスキルを `active → stale → archived` の状態へと移行させ、定期的に短い補助モデルによるレビューを起動して、統合の提案やドリフトのパッチを行います。

これは、[自己改善ループ](/docs/user-guide/features/skills#agent-managed-skills-skill_manage-tool)を通じて作成されたスキルが際限なく積み上がらないようにするために存在します。エージェントが新しい問題を解決してスキルを保存するたびに、そのスキルは `~/.hermes/skills/` に保存されます。保守を行わないと、カタログを汚染しトークンを浪費する、わずかに異なる狭い範囲の準重複スキルが数十個も溜まってしまいます。

Curatorは、バンドルされたスキル（リポジトリに同梱されたもの）やハブからインストールされたスキル（[agentskills.io](https://agentskills.io) 由来のもの）には**一切手を触れません**。エージェント自身が作成したスキルのみをレビューします。また**自動削除も一切行いません** — 最悪の結果でも `~/.hermes/skills/.archive/` へのアーカイブにとどまり、これは復元可能です。

[issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) で追跡されています。

## 動作の仕組み

Curatorは、cronデーモンではなく非アクティブチェックによってトリガーされます。CLIセッションの開始時、およびゲートウェイのcronティッカースレッド内の定期的なティックで、Hermesは次の条件を確認します。

1. 最後のCurator実行から十分な時間が経過しているか（`interval_hours`、デフォルト**7日**）、そして
2. エージェントが十分な時間アイドル状態であるか（`min_idle_hours`、デフォルト**2時間**）。

両方が真であれば、`AIAgent` のバックグラウンドフォークを起動します — これはメモリ/スキルの自己改善ナッジで使われるものと同じパターンです。このフォークは独自のプロンプトキャッシュ内で実行され、アクティブな会話には一切触れません。

:::info 初回実行時の挙動
新規インストール直後（または `hermes update` 後にCurator導入前のインストールが初めてティックする場合）、Curatorは**すぐには実行されません**。最初の観測で `last_run_at` を「現在」にシードし、最初の本番パスを `interval_hours` の1周期分だけ先送りします。これにより、Curatorが手を触れる前に、スキルライブラリをレビューし、重要なものをピン留めしたり、完全にオプトアウトしたりするための1周期分の余裕が得られます。

Curatorが本番実行する前に*何をしようとしているか*を確認したい場合は、`hermes curator run --dry-run` を実行してください — ライブラリを変更することなく、同じレビューレポートを生成します。
:::

1回の実行には2つのフェーズがあります。

1. **自動遷移**（決定論的、LLM不要）。`stale_after_days`（30）の間使われていないスキルは `stale` になり、`archive_after_days`（90）の間使われていないスキルは `~/.hermes/skills/.archive/` に移動されます。
2. **LLMレビュー**（補助モデルによる単一パス、`max_iterations=8`）。フォークされたエージェントがエージェント作成スキルを調査し、`skill_view` でそれらを読み、スキルごとに維持・パッチ（`skill_manage` 経由）・重複する複数のスキルの統合・terminalツールによるアーカイブのいずれを行うかを決定します。

ピン留めされたスキルは、Curatorの自動遷移とエージェント自身の `skill_manage` ツールの両方の対象外です。下記の[スキルのピン留め](#pinning-a-skill)を参照してください。

## 設定

すべての設定は `config.yaml` の `curator:` 配下に置かれます（`.env` ではありません — これは秘密情報ではないため）。デフォルト値は次のとおりです。

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7日
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
```

完全に無効化するには、`curator.enabled: false` を設定します。

### より安価な補助モデルでレビューを実行する

CuratorのLLMレビューパスは、Vision、Compression、Session Searchなどと並ぶ通常の補助タスクスロット（`auxiliary.curator`）です。「Auto」は「メインのチャットモデルを使う」という意味です。レビューパスに特定のプロバイダー＋モデルをピン留めするには、このスロットを上書きしてください。

**最も簡単 — `hermes model`:**

```bash
hermes model                   # → 「Auxiliary models — side-task routing」
                               # → 「Curator」を選択 → プロバイダーを選択 → モデルを選択
```

同じピッカーは、Webダッシュボードの**Models**タブからも利用できます。

**config.yaml を直接編集（同等）:**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # 余裕を持って — レビューには数分かかることがあります
```

`provider: auto`（デフォルト）のままにすると、他のすべての補助タスクと同じ挙動で、レビューパスはメインのチャットモデルを通じてルーティングされます。

:::note レガシー設定
以前のリリースでは、一度限りの `curator.auxiliary.{provider,model}` ブロックを使用していました。このパスは引き続き機能しますが、非推奨のログ行を出力します — Curatorが他のすべての補助タスクと同じ仕組み（`hermes model`、ダッシュボードのModelsタブ、`base_url`、`api_key`、`timeout`、`extra_body`）を共有するよう、上記の `auxiliary.curator` への移行をお願いします。
:::

## CLI

```bash
hermes curator status         # 最終実行、各種カウント、ピン留めリスト、LRU上位5件
hermes curator run            # 今すぐレビューをトリガー（LLMパスが終わるまでブロック）
hermes curator run --background  # 投げっぱなし: LLMパスをバックグラウンドスレッドで開始
hermes curator run --dry-run  # プレビューのみ — 変更を伴わないレポート
hermes curator backup         # ~/.hermes/skills/ の手動スナップショットを取得
hermes curator rollback       # 最新のスナップショットから復元
hermes curator rollback --list     # 利用可能なスナップショットを一覧表示
hermes curator rollback --id <ts>  # 特定のスナップショットを復元
hermes curator rollback -y         # 確認プロンプトをスキップ
hermes curator pause          # 再開されるまで実行を停止
hermes curator resume
hermes curator pin <skill>    # このスキルを自動遷移の対象外にする
hermes curator unpin <skill>
hermes curator restore <skill>  # アーカイブされたスキルをactiveに戻す
```

## バックアップとロールバック

すべての本番Curatorパスの前に、Hermesは `~/.hermes/skills/` のtar.gzスナップショットを `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz` に取得します。あるパスが、手を触れてほしくなかったものをアーカイブまたは統合してしまった場合、1つのコマンドで実行全体を取り消せます。

```bash
hermes curator rollback        # 最新のスナップショットを復元（確認あり）
hermes curator rollback -y     # プロンプトをスキップ
hermes curator rollback --list # 理由とサイズ付きですべてのスナップショットを表示
```

ロールバック自体も可逆です。スキルツリーを置き換える前に、Hermesは `pre-rollback to <target-id>` というタグの付いた別のスナップショットを取得するため、誤ったロールバックは `--id` でそのスナップショットへロールフォワードすることで取り消せます。

`hermes curator backup --reason "before-refactor"` を使えば、いつでも手動でスナップショットを取得できます。`--reason` 文字列はスナップショットの `manifest.json` に記録され、`--list` で表示されます。

スナップショットは、ディスク使用量を抑えるために `curator.backup.keep`（デフォルト5）まで間引かれます。

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

`curator.backup.enabled: false` を設定すると、自動スナップショットを無効化できます。バックアップが無効化されているときに手動の `hermes curator backup` コマンドが機能するのは、先に `enabled: true` を設定した場合のみです — このフラグは両方のパスを対称的にゲートするため、変更を伴う実行で本番前スナップショットを誤ってスキップする方法はありません。

`hermes curator status` は、最も長く使われていないスキル5件も一覧表示します — 次に陳腐化しそうなものを素早く把握する方法です。

同じサブコマンドは、実行中のセッション内（CLIまたはゲートウェイプラットフォーム）で `/curator` スラッシュコマンドとしても利用できます。

## 「エージェント作成」の意味

スキルは、その名前が次のいずれにも**含まれていない**場合に、エージェント作成と見なされます。

- `~/.hermes/skills/.bundled_manifest`（インストール時にリポジトリからコピーされたスキル）、および
- `~/.hermes/skills/.hub/lock.json`（`hermes skills install` 経由でインストールされたスキル）。

`~/.hermes/skills/` 内のそれ以外のすべては、Curatorの対象になります。これには次が含まれます。

- 会話中にエージェントが `skill_manage(action="create")` で保存したスキル。
- 手書きの `SKILL.md` で手動作成したスキル。
- Hermesに指定した外部スキルディレクトリ経由で追加されたスキル。

:::warning 手書きのスキルはエージェント保存のものと見分けがつきません
ここでの出自は**二値的**です（バンドル/ハブ vs. それ以外すべて）。Curatorは、プライベートなワークフローのために頼りにしている手書きのスキルと、自己改善ループがセッション途中で保存したスキルを区別できません。どちらも「エージェント作成」のバケットに入ります。

最初の本番パス（デフォルトではインストールから7日後）の前に、少し時間を取って次を行ってください。

1. `hermes curator run --dry-run` を実行して、Curatorが何を提案するか正確に確認する。
2. `hermes curator pin <name>` を使って、手を触れてほしくないものを囲い込む。
3. ライブラリを自分で管理したい場合は、`config.yaml` で `curator.enabled: false` を設定する。

アーカイブは `hermes curator restore <name>` で常に復元可能ですが、事後に統合を追いかけるよりも、事前にピン留めするほうが簡単です。
:::

特定のスキルを絶対に触られないように保護したい場合 — 例えば頼りにしている手書きのスキルなど — は、`hermes curator pin <name>` を使ってください。次のセクションを参照してください。

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

ピン留めできるのは**エージェント作成**スキルのみです — バンドルおよびハブインストールのスキルは、そもそもCuratorの変更対象になることがなく、`hermes curator pin` で試みると説明付きのメッセージとともに拒否されます。

「削除しない」より強い保証が必要な場合 — 例えば、エージェントが引き続き読み込む一方でスキルの内容を完全に凍結したい場合 — は、お使いのエディタで `~/.hermes/skills/<name>/SKILL.md` を直接編集してください。ピンが守るのはツール駆動の削除であり、あなた自身のファイルシステムアクセスではありません。

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
    ├── run.json      # 機械可読: 完全な忠実度、統計、LLM出力
    └── REPORT.md     # 人間可読のサマリー
```

`REPORT.md` は、特定の実行が何をしたか — どのスキルが遷移し、LLMレビュアーが何と言い、どのスキルをパッチしたか — を素早く確認する方法です。`agent.log` をgrepすることなく監査するのに適しています。

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

- [スキルシステム](/docs/user-guide/features/skills) — スキルの一般的な仕組みと、それを作成する自己改善ループ
- [メモリ](/docs/user-guide/features/memory) — 長期メモリを保守する並行のバックグラウンドレビュー
- [バンドルスキルカタログ](/docs/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — 元の提案と設計の議論
