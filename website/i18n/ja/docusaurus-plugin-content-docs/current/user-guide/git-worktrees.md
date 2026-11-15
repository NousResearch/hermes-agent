---
sidebar_position: 3
sidebar_label: "Git Worktree"
title: "Git Worktree"
description: "git worktree と分離されたチェックアウトを使い、同じリポジトリ上で複数の Hermes エージェントを安全に実行する"
---

# Git Worktree

Hermes Agent は、大規模で長く存続するリポジトリで使われることがよくあります。次のようなことをしたいとき:

- 同じプロジェクトで**複数のエージェントを並列に実行**する、または
- 実験的なリファクタリングをメインブランチから分離して保つ、

Git の **worktree** は、リポジトリ全体を複製することなく、各エージェントに独自のチェックアウトを与える最も安全な方法です。

このページでは、各セッションがクリーンで分離された作業ディレクトリを持つように、worktree と Hermes を組み合わせる方法を示します。

## なぜ Hermes で worktree を使うのか？

Hermes は**カレントワーキングディレクトリ**をプロジェクトルートとして扱います:

- CLI: `hermes` または `hermes chat` を実行するディレクトリ
- メッセージングゲートウェイ: `~/.hermes/config.yaml` の `terminal.cwd` で設定されたディレクトリ

**同じチェックアウト**で複数のエージェントを実行すると、それらの変更が互いに干渉する可能性があります:

- 一方のエージェントが、もう一方が使っているファイルを削除または書き換えるかもしれません。
- どの変更がどの実験に属するのか理解しづらくなります。

worktree を使うと、各エージェントは次を得ます:

- **独自のブランチと作業ディレクトリ**
- `/rollback` 用の**独自のチェックポイントマネージャー履歴**

関連項目: [チェックポイントと /rollback](./checkpoints-and-rollback.md)。

## クイックスタート: worktree の作成

### セッション内から: `/worktree new`

最も手早い方法です（Copilot CLI の `/worktree new` にインスパイアされています）。対話型の CLI セッションから次を実行します:

```
/worktree new my-experiment
```

Hermes はリポジトリ内に `.worktrees/my-experiment/` を作成し（ブランチは `hermes/my-experiment`。`worktree_sync: false` でない限り、直前にフェッチしたリモートの先端をベースにします）、セッションのターミナルツールとファイルツールの対象をその中へ切り替えます — 再起動は不要です。名前を省略すると、ランダムな `hermes-<id>` ツリーになります。`/worktree` だけを実行するとアクティブなツリーを表示し、`/worktree list` ですべてのツリーを一覧表示します。終了時、クリーンでプッシュ済みのツリーは削除されます。未コミットの変更や未プッシュのコミットがあるツリーでは、`hermes -w` と同じ「保持 / アーカイブ / 削除」の質問が表示されます。

### git で手動で作成する

メインリポジトリ（`.git/` を含む）から、フィーチャーブランチ用の新しい worktree を作成します:

```bash
# From the main repo root
cd /path/to/your/repo

# Create a new branch and worktree in ../repo-feature
git worktree add ../repo-feature feature/hermes-experiment
```

これにより次が作成されます:

- 新しいディレクトリ: `../repo-feature`
- 新しいブランチ: そのディレクトリにチェックアウトされた `feature/hermes-experiment`

これで新しい worktree に `cd` して、そこで Hermes を実行できます:

```bash
cd ../repo-feature

# Start Hermes in the worktree
hermes
```

Hermes は次のようにします:

- `../repo-feature` をプロジェクトルートとして認識します。
- そのディレクトリをコンテキストファイル、コード編集、ツールに使用します。
- この worktree にスコープされた `/rollback` 用の**別個のチェックポイント履歴**を使用します。

## 複数のエージェントを並列に実行する

それぞれ独自のブランチを持つ複数の worktree を作成できます:

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

別々のターミナルで:

```bash
# Terminal 1
cd ../repo-experiment-a
hermes

# Terminal 2
cd ../repo-experiment-b
hermes
```

各 Hermes プロセスは:

- 独自のブランチ（`feature/hermes-a` vs `feature/hermes-b`）で作業します。
- 異なるシャドウリポジトリのハッシュ（worktree のパスから導出）配下にチェックポイントを書き込みます。
- もう一方に影響を与えずに `/rollback` を独立して使用できます。

これは特に次の場合に便利です:

- バッチリファクタリングを実行するとき。
- 同じタスクに対して異なるアプローチを試すとき。
- 同じ上流リポジトリに対して CLI ＋ ゲートウェイのセッションをペアリングするとき。

## worktree を安全にクリーンアップする

実験が終わったら:

1. 作業を残すか破棄するかを決めます。
2. 残したい場合:
   - 通常どおりブランチをメインブランチにマージします。
3. worktree を削除します:

```bash
cd /path/to/your/repo

# Remove the worktree directory and its reference
git worktree remove ../repo-feature
```

注意:

- `git worktree remove` は、強制しない限りコミットされていない変更がある worktree の削除を拒否します。
- worktree を削除しても、ブランチは**自動的には**削除されません。通常の `git branch` コマンドでブランチを削除または保持できます。
- `~/.hermes/checkpoints/` 配下の Hermes チェックポイントデータは、worktree を削除しても自動的にはプルーニングされませんが、通常は非常に小さいものです。

## ベストプラクティス

- **Hermes の実験ごとに 1 つの worktree**
  - 実質的な変更ごとに専用のブランチ/worktree を作成します。
  - これにより diff が焦点を保ち、PR が小さくレビューしやすくなります。
- **実験にちなんでブランチに名前を付ける**
  - 例: `feature/hermes-checkpoints-docs`、`feature/hermes-refactor-tests`。
- **頻繁にコミットする**
  - 高レベルのマイルストーンには git コミットを使います。
  - その間のツール駆動の編集に対するセーフティネットとして[チェックポイントと /rollback](./checkpoints-and-rollback.md)を使います。
- **worktree を使うときは bare リポジトリのルートから Hermes を実行しない**
  - 代わりに worktree ディレクトリを優先し、各エージェントが明確なスコープを持つようにします。

## `hermes -w` を使う（自動 worktree モード）

Hermes には、独自のブランチを持つ**使い捨ての git worktree を自動的に作成する** `-w` フラグが組み込まれています。worktree を手動でセットアップする必要はありません — リポジトリに `cd` して実行するだけです:

```bash
cd /path/to/your/repo
hermes -w
```

Hermes は次のようにします:

- リポジトリ内の `.worktrees/` 配下に一時的な worktree を作成します。
- 分離されたブランチ（例: `hermes/hermes-<hash>`）をチェックアウトします。
- その worktree 内で完全な CLI セッションを実行します。

Hermes は、リポジトリ独自の git 設定を実行せずに worktree をチェックアウトします: そのチェックアウトではフック、`core.fsmonitor`、clean/smudge フィルターが無効になります。これは Hermes がカンバンのタスクやサブエージェント用に作成する worktree と同じです。Git LFS でファイルを保存しているリポジトリでは、そこにはポインターファイルが置かれます。セッションでそれらの内容が必要な場合は、worktree 内で `git lfs pull` を実行してください。

これが worktree の分離を得る最も簡単な方法です。単一のクエリと組み合わせることもできます:

```bash
hermes -w -z "Fix issue #123"
```

並列エージェントの場合、複数のターミナルを開いてそれぞれで `hermes -w` を実行します — 各呼び出しが自動的に独自の worktree とブランチを取得します。

### 未完了の作業を残してセッションを終了する

対話型の `hermes -w` セッション（クラシック CLI または `--tui`）が終了する際、その worktree に未コミットの変更や未プッシュのコミットがある場合、Hermes は次のように尋ねます:

- **worktree を保持する**（デフォルト。Enter/Esc でもこれが選ばれます） — 何も変更されません。
- **アーカイブして削除する** — 作業状態全体がアーカイブされ（後述）、セッションの終了が完了する前にツリーが削除されるため、大きなツリーが途中で打ち切られることはありません。Hermes は `hermes worktree restore` コマンドを表示します。アーカイブに失敗した場合、ツリーは保持されます。
- **削除して変更を破棄する** — 明示的に選択した場合のみ実行されます。

ワンショット実行（`-q`、`-z`）、cron ジョブ、ゲートウェイセッション、およびターミナルのない終了では質問は表示されません: ツリーは保持され、後で古い worktree の一掃処理が扱います。コミットがプッシュ済みのクリーンなツリーは、質問なしで削除されます。

## 古い worktree の一掃と復元 {#stale-worktree-sweep-and-restore}

Hermes が worktree を作成したすべてのリポジトリ（`hermes -w`、`/worktree new`、サブエージェントの worktree 分離、デスクトップアプリの git ペイン）は、アクティブなプロファイルの `<HERMES_HOME>/worktree_repos.json` に記録されます。`hermes -w` の起動時と、cron ティッカーによる 6 時間ごとに、Hermes はそれらの各リポジトリの `.worktrees/` を一掃します:

| ツリー | 判定 |
|---|---|
| 階層ごとの期間（`hermes-*`/`subagent-*` は 24 時間、名前付きツリーは 72 時間）を超えてアイドル状態で、クリーンかつマージ済みまたはプッシュ済み | 削除 |
| 階層ごとの期間を超えてアイドル状態で、未コミットの変更や未プッシュのコミットがある | **アーカイブしてから削除** |
| 生存中の hermes ロック所有者、または cwd（Linux では開いているファイルも）が内部にある生存中のプロセスが存在する | 一切触れない |
| ファイル、`HEAD`、または reflog のアクティビティが階層ごとの期間より新しい（削除されたファイルはそのフォルダを通じてカウントされます） | 一切触れない |
| カンバンのタスクツリー（`t_<hex>`） | 一切触れない（カンバンの GC が管理します） |
| 任意のサブモジュール: インデックス内の gitlink、またはツリーの `modules/` 配下に保持されたサブモジュールリポジトリ（`git submodule deinit` 後も含む） | 保持（`keep-submodule-work`: アーカイブにはサブモジュールのポインターしか記録されないため）。クリーンなマージ済みツリーは、サブモジュールが一度も初期化されていない場合にのみ削除されます |
| 内部のどこかに埋め込まれたリポジトリ（gitignore されているかどうかを問わない。例: `build/tool/.git`） | 保持（`keep-unarchivable`） |
| worktree ごとの ref（`refs/worktree/*`、`refs/bisect/*`、`refs/rewritten/*`）だけが保持しているコミット | 保持（`keep-worktree-refs`: それらの ref はツリーと一緒に削除されるため） |
| `worktrees.archive_max_file_mb` より大きい未追跡ファイル | 保持（`keep-large-untracked`） |

**無視されたファイル。** 無視されたファイル（`.env` のコピー、キャッシュ、ビルドやテストの出力）はアーカイブされず、ツリーと一緒に削除されます。唯一の例外は無視されたフォルダ内のネストされた git リポジトリで、その場合ツリーは保持されます。

同じチェックはクリーンなマージ済みツリーにも適用されます。また、すべての dirty チェックは、`status.showUntrackedFiles=no` で隠れてしまうファイルや、バイト内容がインデックスと異なる skip-worktree / assume-unchanged のファイルもカウントします。

アーカイブは `refs/hermes/archive/<tree>` に保存されるコミットで、ツリーの `HEAD`、ステージ済みの変更、未ステージの変更、未追跡ファイルを保持します（`node_modules/` や `.venv/` など、`.gitignore` で除外されたファイルは含まれません）。アーカイブは一時インデックスを使って作成されるため、ツリー自体が変更されることはありません。アーカイブを作成できない場合、ツリーは保持されます。アーカイブの作成中にツリーに入られたり、ステージ済みの変更が動いたり（`git add`、`git apply --cached`）した場合も、アーカイブは保存され、ツリーは保持されます。ファイルのバイト内容は正確に保存・復元されます: `.gitattributes`、`info/attributes`、`core.attributesFile` の改行コードやフィルターのルールは適用されません。`worktrees.archive_max_file_mb` より大きい未追跡ファイルはアーカイブから除外されてアーカイブ内に一覧として記録され、その場合ツリーは保持されます: 終了時の **Archive** とサブエージェントの結果でその旨が表示されます。リモートに存在するブランチ（オープン中の PR）は保持されます。それ以外の場合、アーカイブがそのコミットを保持し、ローカルブランチは削除されます。

```bash
hermes worktree list                 # trees plus archives (name, age, branch, reason)
hermes worktree prune --dry-run      # every archive/remove/skip decision and why; changes nothing
hermes worktree restore lane-x       # recreate .worktrees/lane-x with its branch, staged and untracked files
hermes worktree restore lane-x --path ../lane-x-review
```

復元では、ブランチが存在しないか、まだアーカイブされた `HEAD` を指している場合にブランチを元に戻します。それ以外の場合、ツリーはそのコミットで detached 状態として復元されます。`worktrees.archive_retention_days` より古いアーカイブは一掃処理で削除されます。

```yaml
worktrees:
  archive_dirty: true          # false = never remove dirty/unpushed trees (the old behaviour)
  archive_max_file_mb: 50      # untracked files above this are not archived; their tree is kept
  archive_retention_days: 30   # 0 keeps archives forever
  max_total_gb: 0              # optional budget for idle trees; oldest are archived + removed first
  install_state_retention_days: 14  # dependency state of a missing checkout outside .worktrees/ and
                               # the data root, unused this long, is reclaimed; 0 keeps it forever.
                               # Read from the root profile's config: its profiles share installs/
```

存在できるツリーの数に上限はありません: 何百もの並列サブエージェントが、それぞれ作業している間はツリーを保持します。`max_total_gb` が削除するのは、階層ごとの期間を超えたアイドル状態のツリーだけです。

サブエージェントの worktree（`delegation.worktree_isolation: true`）も、子エージェントの終了時に同じルールに従います: クリーンなツリーは削除され、コミットや編集のあるツリーはアーカイブして削除されますが、親がマージできるようにブランチは保持されます。

## まとめ

- **git worktree** を使って各 Hermes セッションに独自のクリーンなチェックアウトを与えます。
- **ブランチ**を使って実験の高レベルの履歴を捉えます。
- **チェックポイント ＋ `/rollback`** を使って各 worktree 内のミスから回復します。

この組み合わせにより、次が得られます:

- 異なるエージェントや実験が互いに干渉しないという強力な保証。
- 不適切な編集から簡単に回復できる、高速な反復サイクル。
- クリーンでレビューしやすいプルリクエスト。

## 複数の worktree にまたがる UI 部分の開発

TypeScript の部分（`ui-tui/`、`apps/desktop/`）はそれぞれ `node_modules` を必要とし、worktree ごとに `pnpm install` を実行すると、ブランチごとに同じインストールを繰り返すことになります。複数の worktree から TUI やデスクトップアプリを開発する場合は、シンボリックリンクで 1 つのインストールを共有する `htui` / `hgui` ヘルパーについて [worktree からの TUI とデスクトップ開発](/docs/developer-guide/worktree-ui-dev) を参照してください。
