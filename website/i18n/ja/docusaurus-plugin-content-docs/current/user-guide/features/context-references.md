---
sidebar_position: 9
sidebar_label: "コンテキスト参照"
title: "コンテキスト参照"
description: "ファイル、フォルダ、git diff、URL をメッセージに直接添付するためのインライン @ 構文"
---

# コンテキスト参照

`@` に続けて参照を入力すると、コンテンツがメッセージに直接挿入されます。Hermes は参照をインラインで展開し、`--- Attached Context ---` セクションの下にコンテンツを追加します。

## サポートされている参照

| 構文 | 説明 |
|--------|-------------|
| `@file:path/to/file.py` | ファイルの内容を挿入します |
| `@file:path/to/file.py:10-25` | 特定の行範囲を挿入します（1 始まり、両端を含む） |
| `@folder:path/to/dir` | ファイルメタデータ付きのディレクトリツリー一覧を挿入します |
| `@diff` | `git diff`（ステージされていない作業ツリーの変更）を挿入します |
| `@staged` | `git diff --staged`（ステージされた変更）を挿入します |
| `@git:5` | 直近 N 件のコミットをパッチ付きで挿入します（最大 10） |
| `@url:https://example.com` | Web ページのコンテンツを取得して挿入します |

## 使用例

```text
Review @file:src/main.py and suggest improvements

What changed? @diff

Compare @file:old_config.yaml and @file:new_config.yaml

What's in @folder:src/components?

Summarize this article @url:https://arxiv.org/abs/2301.00001
```

1 つのメッセージ内で複数の参照を使用できます:

```text
Check @file:main.py, and also @file:test.py.
```

参照値の末尾の句読点（`,`、`.`、`;`、`!`、`?`）は自動的に除去されます。

## CLI タブ補完

インタラクティブ CLI では、`@` を入力すると自動補完がトリガーされます:

- `@` ですべての参照タイプ（`@diff`、`@staged`、`@file:`、`@folder:`、`@git:`、`@url:`）が表示されます
- `@file:` と `@folder:` は、ファイルサイズのメタデータ付きでファイルシステムのパス補完をトリガーします
- 単独の `@` に続けて部分的なテキストを入力すると、現在のディレクトリから一致するファイルとフォルダが表示されます

## 行範囲

`@file:` 参照は、正確なコンテンツ挿入のために行範囲をサポートします:

```text
@file:src/main.py:42        # 42 行目のみ
@file:src/main.py:10-25     # 10 行目から 25 行目まで（両端を含む）
```

行は 1 始まりです。無効な範囲は黙って無視されます（ファイル全体が返されます）。

## サイズ制限

コンテキスト参照は、モデルのコンテキストウィンドウを圧迫しないように上限が設けられています:

| しきい値 | 値 | 動作 |
|-----------|-------|----------|
| ソフトリミット | コンテキスト長の 25% | 警告が追加され、展開は続行されます |
| ハードリミット | コンテキスト長の 50% | 展開は拒否され、元のメッセージが変更されずに返されます |
| フォルダエントリ | 最大 200 ファイル | 超過分のエントリは `- ...` に置き換えられます |
| Git コミット | 最大 10 | `@git:N` は範囲 [1, 10] にクランプされます |

## セキュリティ

### 機密パスのブロック

認証情報の漏洩を防ぐため、これらのパスは常に `@file:` 参照からブロックされます:

- SSH キーと設定: `~/.ssh/id_rsa`、`~/.ssh/id_ed25519`、`~/.ssh/authorized_keys`、`~/.ssh/config`
- シェルプロファイル: `~/.bashrc`、`~/.zshrc`、`~/.profile`、`~/.bash_profile`、`~/.zprofile`
- 認証情報ファイル: `~/.netrc`、`~/.pgpass`、`~/.npmrc`、`~/.pypirc`
- Hermes の env: `$HERMES_HOME/.env`

これらのディレクトリは完全にブロックされます（内部のすべてのファイル）:
- `~/.ssh/`、`~/.aws/`、`~/.gnupg/`、`~/.kube/`、`$HERMES_HOME/skills/.hub/`

### パストラバーサル保護

すべてのパスは作業ディレクトリを基準に解決されます。許可されたワークスペースのルート外に解決される参照は拒否されます。

### バイナリファイルの検出

バイナリファイルは、MIME タイプと NULL バイトのスキャンによって検出されます。既知のテキスト拡張子（`.py`、`.md`、`.json`、`.yaml`、`.toml`、`.js`、`.ts` など）は MIME ベースの検出をバイパスします。バイナリファイルは警告とともに拒否されます。

## プラットフォームの可用性

コンテキスト参照は主に **CLI の機能**です。インタラクティブ CLI で動作し、`@` がタブ補完をトリガーし、参照はメッセージがエージェントに送信される前に展開されます。

**メッセージングプラットフォーム**（Telegram、Discord など）では、`@` 構文はゲートウェイによって展開されません — メッセージはそのまま渡されます。エージェント自体は、`read_file`、`search_files`、`web_extract` ツールを介して引き続きファイルを参照できます。

## コンテキスト圧縮との相互作用

会話のコンテキストが圧縮されると、展開された参照コンテンツが圧縮サマリーに含まれます。これは以下を意味します:

- `@file:` 経由で挿入された大きなファイルの内容は、コンテキストの使用量に影響します
- 会話が後で圧縮された場合、ファイルの内容は要約されます（逐語的に保持されるわけではありません）
- 非常に大きなファイルの場合は、関連するセクションのみを挿入するために行範囲（`@file:main.py:100-200`）の使用を検討してください

## 一般的なパターン

```text
# コードレビューのワークフロー
Review @diff and check for security issues

# コンテキスト付きでデバッグ
This test is failing. Here's the test @file:tests/test_auth.py
and the implementation @file:src/auth.py:50-80

# プロジェクトの探索
What does this project do? @folder:src @file:README.md

# リサーチ
Compare the approaches in @url:https://arxiv.org/abs/2301.00001
and @url:https://arxiv.org/abs/2301.00002
```

## エラー処理

無効な参照は、失敗ではなくインライン警告を生成します:

| 条件 | 動作 |
|-----------|----------|
| ファイルが見つからない | 警告: "file not found" |
| バイナリファイル | 警告: "binary files are not supported" |
| フォルダが見つからない | 警告: "folder not found" |
| Git コマンドが失敗 | git の stderr 付きの警告 |
| URL がコンテンツを返さない | 警告: "no content extracted" |
| 機密パス | 警告: "path is a sensitive credential file" |
| ワークスペース外のパス | 警告: "path is outside the allowed workspace" |
