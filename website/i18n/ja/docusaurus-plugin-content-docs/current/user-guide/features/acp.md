---
sidebar_position: 11
title: "ACP エディタ統合"
description: "VS Code、Zed、JetBrains などの ACP 互換エディタ内で Hermes Agent を使う"
---

# ACP エディタ統合

Hermes Agent は ACP サーバーとして実行でき、ACP 互換のエディタが stdio 経由で Hermes と通信し、次をレンダリングできるようにします:

- チャットメッセージ
- ツールアクティビティ
- ファイル差分
- ターミナルコマンド
- 承認プロンプト
- ストリーミングされる思考 / 応答のチャンク

ACP は、Hermes をスタンドアロンの CLI やメッセージングボットではなく、エディタネイティブのコーディングエージェントのように振る舞わせたいときに最適です。

## ACP モードで Hermes が公開するもの

Hermes は、エディタワークフロー向けに厳選された `hermes-acp` ツールセットで実行されます。これには次が含まれます:

- ファイルツール: `read_file`、`write_file`、`patch`、`search_files`
- ターミナルツール: `terminal`、`process`
- web/ブラウザツール
- メモリ、todo、セッション検索
- スキル
- execute_code と delegate_task
- vision

メッセージング配信や cronjob 管理など、典型的なエディタの UX に合わないものは意図的に除外されています。

## インストール

通常どおり Hermes をインストールし、その後 ACP のエクストラを追加します:

```bash
pip install -e '.[acp]'
```

これにより `agent-client-protocol` 依存関係がインストールされ、次が有効になります:

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## ACP サーバーの起動

次のいずれかで Hermes が ACP モードで起動します:

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

Hermes は stderr にログを出力するため、stdout は ACP の JSON-RPC トラフィック用に予約されたままになります。

## エディタのセットアップ

### VS Code

[ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client) 拡張機能をインストールします。

接続するには:

1. アクティビティバーから ACP Client パネルを開きます。
2. 組み込みのエージェントリストから **Hermes Agent** を選択します。
3. 接続してチャットを開始します。

Hermes を手動で定義したい場合は、VS Code の設定の `acp.agents` の下で追加します:

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed

設定スニペットの例:

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"],
    },
  },
}
```

### JetBrains

ACP 互換のプラグインを使い、次を指すようにします:

```text
/path/to/hermes-agent/acp_registry
```

## レジストリマニフェスト

ACP レジストリマニフェストは次の場所にあります:

```text
acp_registry/agent.json
```

これは、起動コマンドが次であるコマンドベースのエージェントをアドバタイズします:

```text
hermes acp
```

## 設定とクレデンシャル

ACP モードは CLI と同じ Hermes 設定を使用します:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

プロバイダーの解決は Hermes の通常のランタイムリゾルバを使用するため、ACP は現在設定されているプロバイダーとクレデンシャルを継承します。

## セッションの挙動

ACP セッションは、サーバーの実行中、ACP アダプターのインメモリセッションマネージャによって追跡されます。

各セッションは次を保存します:

- セッション ID
- 作業ディレクトリ
- 選択されたモデル
- 現在の会話履歴
- キャンセルイベント

基盤となる `AIAgent` は依然として Hermes の通常の永続化/ログのパスを使用しますが、ACP の `list/load/resume/fork` は現在実行中の ACP サーバープロセスにスコープされます。

## 作業ディレクトリの挙動

ACP セッションは、エディタの cwd を Hermes のタスク ID にバインドするため、ファイルツールとターミナルツールはサーバープロセスの cwd ではなく、エディタのワークスペースを基準に実行されます。

## 承認

危険なターミナルコマンドは、承認プロンプトとしてエディタに送り返すことができます。ACP の承認オプションは CLI のフローよりシンプルです:

- 一度だけ許可
- 常に許可
- 拒否

タイムアウトまたはエラーの場合、承認ブリッジはリクエストを拒否します。

## トラブルシューティング

### ACP エージェントがエディタに表示されない

確認すること:

- エディタが正しい `acp_registry/` のパスを指している
- Hermes がインストールされ、PATH 上にある
- ACP のエクストラがインストールされている（`pip install -e '.[acp]'`）

### ACP は起動するがすぐにエラーになる

次のチェックを試してください:

```bash
hermes doctor
hermes status
hermes acp
```

### クレデンシャルがない

ACP モードには独自のログインフローがありません。Hermes の既存のプロバイダーセットアップを使用します。次でクレデンシャルを設定します:

```bash
hermes model
```

または `~/.hermes/.env` を編集します。

## 関連項目

- [ACP Internals](../../developer-guide/acp-internals.md)
- [Provider Runtime Resolution](../../developer-guide/provider-runtime.md)
- [Tools Runtime](../../developer-guide/tools-runtime.md)
