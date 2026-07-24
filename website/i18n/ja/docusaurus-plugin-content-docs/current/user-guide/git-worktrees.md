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
- メッセージングゲートウェイ: `MESSAGING_CWD` で設定されたディレクトリ

**同じチェックアウト**で複数のエージェントを実行すると、それらの変更が互いに干渉する可能性があります:

- 一方のエージェントが、もう一方が使っているファイルを削除または書き換えるかもしれません。
- どの変更がどの実験に属するのか理解しづらくなります。

worktree を使うと、各エージェントは次を得ます:

- **独自のブランチと作業ディレクトリ**
- `/rollback` 用の**独自のチェックポイントマネージャー履歴**

関連項目: [チェックポイントと /rollback](./checkpoints-and-rollback.md)。

## クイックスタート: worktree の作成

メインリポジトリ（`.git/` を含む）から、フィーチャーブランチ用の新しい worktree を作成します:

```bash
# メインリポジトリのルートから
cd /path/to/your/repo

# 新しいブランチと worktree を ../repo-feature に作成
git worktree add ../repo-feature feature/hermes-experiment
```

これにより次が作成されます:

- 新しいディレクトリ: `../repo-feature`
- 新しいブランチ: そのディレクトリにチェックアウトされた `feature/hermes-experiment`

これで新しい worktree に `cd` して、そこで Hermes を実行できます:

```bash
cd ../repo-feature

# worktree で Hermes を起動
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
# ターミナル 1
cd ../repo-experiment-a
hermes

# ターミナル 2
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

# worktree ディレクトリとその参照を削除
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

これが worktree の分離を得る最も簡単な方法です。単一のクエリと組み合わせることもできます:

```bash
hermes -w -q "Fix issue #123"
```

並列エージェントの場合、複数のターミナルを開いてそれぞれで `hermes -w` を実行します — 各呼び出しが自動的に独自の worktree とブランチを取得します。

## まとめ

- **git worktree** を使って各 Hermes セッションに独自のクリーンなチェックアウトを与えます。
- **ブランチ**を使って実験の高レベルの履歴を捉えます。
- **チェックポイント ＋ `/rollback`** を使って各 worktree 内のミスから回復します。

この組み合わせにより、次が得られます:

- 異なるエージェントや実験が互いに干渉しないという強力な保証。
- 不適切な編集から簡単に回復できる、高速な反復サイクル。
- クリーンでレビューしやすいプルリクエスト。
