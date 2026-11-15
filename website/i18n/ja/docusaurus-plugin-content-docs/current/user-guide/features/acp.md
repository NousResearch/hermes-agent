---
sidebar_position: 11
title: "ACP ホスト統合"
description: "ACP 互換のエディタやコラボレーションプラットフォーム内で Hermes Agent を使う"
---

# ACP ホスト統合

このページの Python 依存関係コマンドは、
[PM で準備したソースチェックアウト](/docs/reference/package-management#developer-workflow)を使用します。
依存関係を変更した後は、チェックアウトを再アクティブ化して Hermes を再起動してください。

Hermes Agent は ACP サーバーとして実行でき、ACP 互換のホストが stdio 経由で
Hermes と通信できるようにします。エディタは次をレンダリングできます:

- チャットメッセージ
- ツールアクティビティ
- ファイル差分
- ターミナルコマンド
- 承認プロンプト
- ストリーミングされる思考 / 応答のチャンク

他のホストも同じプロトコルを使って、コラボレーションのイベントを Hermes に
ルーティングできます。ACP は、別のアプリケーションが会話のトランスポートを担いつつ、
Hermes には既存のアイデンティティ、プロバイダー設定、メモリ、スキル、ツールを
維持させたいときに最適です。

## ACP モードで Hermes が公開するもの {#what-hermes-exposes-in-acp-mode}

Hermes は、エディタワークフロー向けに厳選された `hermes-acp` ツールセットで実行されます。これには次が含まれます:

- ファイルツール: `read_file`、`write_file`、`patch`、`search_files`
- ターミナルツール: `terminal`、`process`
- web/ブラウザツール
- メモリ、todo、セッション検索
- スキル
- execute_code と delegate_task
- vision

メッセージング配信や cronjob 管理など、典型的なエディタの UX に合わないものは意図的に除外されています。

ツールセットは、同じプラットフォーム設定に対するメッセージングゲートウェイと同じ方法で
解決されます。これには、有効なプラグインツールセットなど、ゲートウェイがリストに
追加するエクストラも含まれるため、ACP セッションもそれらを取得します。
`platform_toolsets.acp` は `hermes-acp` のデフォルトを置き換え、
`agent.disabled_toolsets` はすべての ACP セッションからツールセットを取り除きます。
`mcp_servers` の MCP サーバーも同じルールに従います。デフォルトでは、ACP は
有効なすべてのサーバーを取得します。`platform_toolsets.acp` にサーバー名を列挙すると、
それらのサーバーのみが含まれ、`no_mcp` はすべてを除外します。`hermes tools`
には ACP の項目がないため、`config.yaml` を直接編集してください:

```yaml
platform_toolsets:
  acp: [file, web, skills, github]   # only the github MCP server
agent:
  disabled_toolsets: [code_execution]
```

エディタが `session/new` で送信する MCP サーバーは別扱いです。クライアントが
セッションごとにそれらを要求し、常に追加されます。

## エディタが提供する MCP サーバー {#editor-provided-mcp-servers}

`acp-session-mcp-v1` をアドバタイズするゲートウェイでは、エディタの `mcpServers` は
ACP ビューアープロセスではなく、正規のセッションに属します。ゲートウェイは、2 つの
エディタが同じサーバー名を指定した場合でも、別々のレジストリ名前空間を使用します。
エディタを閉じても、別のセッションの MCP 実行がキャンセルされることはありません。

ディスカバリーは、最初のモデルターンの前に、即時かつ上限付きで行われます。選択された
ツールスキーマは会話の間固定されたままで、サーバーのツールリストの変更には新しい
セッションが必要です。MCP トランスポートは、兄弟セッションや共有 MCP ループを
停止することなく、各実行の後に解放されます。

ゲートウェイは、エディタのトランスポート仕様（コマンド引数、環境変数の値、HTTP
ヘッダーを含む）をメモリ内にのみ保持します。永続的なセッションポリシーが保持するのは
フィンガープリントと選択されたスキーマであり、それらのクレデンシャルのコピーではありません。
ゲートウェイの再起動後は、エディタが load/resume 時に同じ `mcpServers` を指定する必要があります。
クレデンシャルが欠けている場合は実行が拒否され、仕様やシークレットが変更された場合は、
セッションを黙って変更するのではなく再アタッチが拒否されます。サーバー設定を変更する
ときは新しいセッションを開始してください。コマンドとパスはゲートウェイホスト上で実行されます。

## インストール

通常どおり Hermes をインストールし、その後インストール先のチェックアウトから ACP のエクストラを追加します:

```bash
cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['acp'], explicit=True)"
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

非対話的なチェックには:

```bash
hermes acp --version
hermes acp --check
```

### ブラウザツール（オプション）

ブラウザツール（`browser_exec` と CDP ツールの `browser_cdp`、
`browser_dialog`、`browser_vault`）にはローカルの Chromium が必要ですが、これは
Python wheel には含まれていません。次でインストールします:

```bash
hermes acp --setup-browser           # interactive (prompts before ~400 MB download)
hermes acp --setup-browser --yes     # accept the download non-interactively
```

これはスタンドアロンのコマンドです。ターミナル認証フロー（`hermes acp --setup`）でも、モデル選択の後に続く質問としてブラウザのブートストラップが提示されるため、ほとんどのユーザーは `--setup-browser` を直接実行する必要はありません。

これが行うこと:

- Hermes が固定バージョンで指定している Chromium（`chromium` パッケージ）をパッケージマネージャー経由でインストールします。Node.js や npm は関与しません
- `browser_exec` が最初にブラウザを必要としたときに、Hermes 自身がその Chromium を起動します

ブートストラップは冪等です — 再実行は高速で、すでに完了している作業はスキップされます。

## ホストのセットアップ

### Buzz チャンネル（リレーブリッジ） {#buzz-channels-relay-bridge}

[Buzz](https://github.com/block/buzz) は、人とエージェントのための Nostr ベースの
コラボレーションプラットフォームです。その `buzz-acp` ハーネスは、stdio 経由で Buzz
チャンネルを任意の ACP エージェントに接続します:

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

これはトランスポートの統合であり、2 つ目の Hermes インストールではありません。
`buzz-acp` が起動するサブプロセスは、そのホスト上の `hermes` と同じ Hermes の設定、
クレデンシャル、メモリ、スキル、状態を使用します。

（これは、Hermes をプリセットのハーネスとしてローカルで起動する
[Buzz Desktop のマネージドランタイム](#buzz-desktop)とは別物です。リレーブリッジは、
通常はサーバー上で、エージェントのアイデンティティとして Buzz の*チャンネル*に参加するためのものです。）

前提条件:

- 上記の ACP のインストールと `hermes acp --check` を完了しておきます。
- [Buzz リポジトリ](https://github.com/block/buzz)から `buzz-acp` と `buzz` CLI を
  ビルドします（`cargo build --release -p buzz-acp`）。
- Hermes 専用の Nostr キーペアを発行し（`buzz-admin generate-key`）、リレーの
  メンバーとして登録します（`buzz-admin add-member`）。各エージェントには
  独自のアイデンティティが必要です — 人間のキーペアを再利用しないでください。
- そのアイデンティティを、対象の Buzz チャンネルに追加します。

次でブリッジを起動します:

```bash
export BUZZ_RELAY_URL="wss://community.example.com"
export BUZZ_PRIVATE_KEY="..."
export BUZZ_API_TOKEN="..."
export BUZZ_ACP_AGENT_COMMAND="hermes"
export BUZZ_ACP_AGENT_ARGS="acp"

buzz-acp
```

`BUZZ_API_TOKEN` は、リレーがトークン認証を強制する場合にのみ必要です。
秘密鍵や API トークンをコミットしたり貼り付けたりしないでください。

永続的なサーバーデプロイでは、対象の Hermes ホームを所有するのと同じ OS ユーザーで、
サービスマネージャーの下で `buzz-acp` を実行します。セットアップ、鍵の生成、
チャンネルのディスカバリー、エージェントごとのオプションは
[buzz-acp README](https://github.com/block/buzz/tree/main/crates/buzz-acp) に記載されています。

ブリッジは、Hermes のアイデンティティがメンバーになっているすべての Buzz チャンネルを
検出し、別のチャンネルに追加されると自動的に購読します。したがって、Buzz チャンネルの
メンバーシップがアクセス境界のままとなり、Hermes は自身の設定に別途チャンネルリストを
持つ必要がありません。

Hermes の ACP アクティビティをオーナーの Buzz Desktop に公開するには、次を追加します:

```bash
export BUZZ_ACP_RELAY_OBSERVER="true"
```

これにより、エージェントのオーナー宛ての暗号化された kind `24200` オブザーバーフレーム
（Buzz の NIP-AO）が発行されます。Desktop は、ライフサイクル、ツール、応答、使用量の
ライブストリームをエージェントの **Activity log** にレンダリングします。リレーはこれらの
フレームを一時的なものとして扱うため、ターンが始まる前に Desktop がオンラインである
必要があります。Desktop のローカルのオブザーバーアーカイブが、オーナー側の永続的な履歴になります。

ヘッドレスのブリッジは、承認ダイアログを表示するエディタが存在しないため、ACP の
権限リクエストに自ら応答します —
[Buzz エージェントをオーナー専用に保つ](#keep-buzz-agents-owner-only)を参照してください。
ブリッジは特権的な自動化として扱ってください: 専用の OS アカウントを使用し、
エージェントにプロンプトを送れる Buzz ユーザーを制限し（`buzz-acp` は
`BUZZ_ACP_AGENT_OWNER` によるオーナー専用の応答ゲートをサポートしています）、
Hermes が作業することを想定したチャンネルにのみメンバーシップを付与します。

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

Zed の設定で、Hermes をカスタムエージェントサーバーとして設定します:

1. Agent Panel を開きます。
2. 次の設定でカスタムエージェントサーバーを追加します:

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

3. 新しい Hermes の外部エージェントスレッドを開始します。

前提条件:

- 先に `hermes model` で Hermes のプロバイダーのクレデンシャルを設定するか、`~/.hermes/.env` / `~/.hermes/config.yaml` で設定しておきます。

### JetBrains

ACP 互換のプラグインを使い、`hermes acp` または `hermes-acp` を指すようにします。

### Buzz Desktop {#buzz-desktop}

[Buzz](https://github.com/block/buzz) は、Hermes Agent をプリセットのランタイムとして同梱しています。
Hermes を通常の方法でインストールしていれば、Buzz は自動的にそれを検出します —
**Settings → Runtimes** を開くと、ランタイムの中に Hermes が表示されます。

検出に失敗する場合（古いインストール）は、ACP ランチャーがログインシェルの PATH 上で
解決されることを確認してください:

```bash
command -v hermes-acp || command -v hermes
```

最近のインストールでは、`hermes` と `hermes-acp` の両方のランチャーが
`~/.local/bin` に書き込まれます。古いインストールでは、`hermes update` を実行すると
`hermes-acp` ランチャーが追加されます。手動のフォールバックとして、Buzz のエージェントコマンドを
`hermes`、引数を `["acp"]` に設定してください。

#### モデルピッカー

Buzz Desktop（v0.5.1+）は、エージェントのランタイム設定に Hermes の完全なモデルメニューを
レンダリングします。このリストは ACP 経由で Hermes 自身から取得されます: Hermes で認証済みの
プロバイダーのすべてのモデル（`hermes model` や `/model` コマンドの背後にあるのと同じ
インベントリ）が表示されるため、メニューにモデルがない場合は、そのプロバイダーの
クレデンシャルが Hermes 側で設定されていないことを意味します。

エントリ ID は `provider:model`（例: `openrouter:z-ai/glm-5.1`）、または
`config.yaml` で定義したカスタムの OpenAI 互換エンドポイントの場合は
`custom:<name>:<model>` の形式です。モデルの選択はそのエージェントのセッションに適用され、
Hermes 全体のデフォルトは変更しません — それには `hermes model` を使用してください。

#### Buzz エージェントをオーナー専用に保つ {#keep-buzz-agents-owner-only}

Buzz は、すべてのエージェントを **Who can talk to this agent** が `Owner only` の状態で作成します。
ランタイムが Hermes の場合は、そのままにしておいてください。

この経路では 2 つの挙動が組み合わさります。`hermes-acp` ツールセットには `terminal`
と `execute_code` が含まれており、Buzz の ACP ブリッジは Hermes の権限リクエストを
表示せずに、自ら `allow_once` で応答します。そのため、Buzz 内の Hermes エージェントは、
プロンプトを出さずにホスト上でシェルコマンドを実行します。筆者がエージェントにスクラッチ
ディレクトリに対して `rm -rf` を実行するよう頼んだところ、どこにもプロンプトが出ることなく削除されました。

`Anyone` を選択すると、チャンネルに到達できるすべての投稿者に同じシェルアクセスを
渡すことになります。Buzz はそれを選んでも警告しません。

`approvals.mode: manual` は役に立ちません: Hermes は権限リクエストを発行しますが、
Buzz がそれを自動承認するため、コマンドはそのまま実行されます。シェルを取り除くには、
代わりにツールセットを絞り込んでください: `platform_toolsets.acp` を `terminal` と
`code_execution` を含まないリストに設定するか、それらを `agent.disabled_toolsets` に追加します。
空の `platform_toolsets.acp: []` でも有効なプラグインツールセットは追加されるため、
取り除きたいプラグインツールセットは `agent.disabled_toolsets` に指定してください。

オーナーからの `!shutdown` は、どのモードでもエージェントを停止します。Buzz は
他の人からのこのコマンドを無視します。

## 設定とクレデンシャル

ACP モードは CLI と同じ Hermes 設定を使用します:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

プロバイダーの解決は Hermes の通常のランタイムリゾルバを使用するため、ACP は現在設定されているプロバイダーとクレデンシャルを継承します。Hermes は、初回実行の ACP クライアント向けにターミナル認証方式（`--setup`）もアドバタイズします。これにより Hermes の対話的なモデル/プロバイダーのセットアップが開きます。

### サービスティアピッカー

`configOptions` をレンダリングする ACP クライアントは、サポートされているモデルルートについて
**Service tier** を選択できます: **Normal**、**Fast**、**Auto**、**Cold**、そしてモデルと
アカウントの適格性チェックが許可する場合は **Ultrafast** です。ゲートウェイが選択肢を提供し、
各選択を検証します。サポートされていないルートではピッカーは表示されません。

選択はそのセッションの次のターンに適用され、すべての Hermes セッションに適用されるわけではなく、
`config.yaml` にも書き込まれません。同じゲートウェイセッションが生きている間は、エディタを
再読み込みしても選択は保持されます。アタッチされた別のクライアントからの変更も ACP の
ピッカーに反映されます。上限付きの Auto と Cold モードについては
[`/fast` リファレンス](../../reference/slash-commands.md)を参照してください。

## ホスト統合 {#host-integration}

古い ACP ホストは、起動する Hermes サブプロセスにまだこの変数を設定する場合があります。
これはユーザー設定ではありません — `.env` や `config.yaml` に手動で設定しないでください。

| 変数 | 値 | 効果 |
|----------|-------|--------|
| `HERMES_ACP_SKIP_CONFIGURED_MCP` | `1` | **効果なし。** Hermes はもうこれを読み取りません。`hermes acp` / `hermes-acp` はゲートウェイデーモンのビューアーとして実行され、`config.yaml` の MCP サーバーを自ら起動することはないため、ACP 側でスキップすべき起動処理がありません。 |

ACP プロセスは MCP ディスカバリーを行いません。ゲートウェイデーモンは、ゲートウェイの起動時に
一度だけ、自身のプロファイルの `config.yaml` にある `mcp_servers` を起動し、ACP
セッションは[上記](#what-hermes-exposes-in-acp-mode)の `platform_toolsets.acp` のルールを
通じてそれらのサーバーを取得します。ACP サブプロセスやゲートウェイにこの変数を設定しても、
それは変わりません。設定済みのサーバーを ACP セッションから除外するには、
`platform_toolsets.acp` または `no_mcp` を使用してください。

エディタの `session/new` の `mcpServers` は、変数が設定されているかどうかにかかわらず、
ゲートウェイに渡されてその正規のセッションにアタッチされます
（[エディタが提供する MCP サーバー](#editor-provided-mcp-servers)を参照）。

## セッションの挙動

ACP セッションは、サーバーの実行中、ACP アダプターのインメモリセッションマネージャによって追跡されます。

各セッションは次を保存します:

- セッション ID
- 作業ディレクトリ
- 選択されたモデル
- 現在の会話履歴
- キャンセルイベント

会話は Hermes のセッションデータベースに永続化され、ACP サーバーの再起動後も一覧表示、
読み込み、再開、フォークができます。プロンプトなしで新しいセッションを開いた場合は、
メモリ内にのみ保持されます: モデルディスカバリーのプローブが空の履歴行を作成することはありません。
空でないフォークはただちに永続化され、既存のセッションのメタデータは、現在の履歴が空であっても
引き続き更新できます。

古いバージョンからの既存の空の行は自動的には削除されません。開いている ACP の行は、
そのクライアントが切断されたことを証明するものではありません。関連するエディタのセッションを
閉じた後、`hermes sessions show <id>` で不要な行を調べ、不要であると確認できた
セッションのみを `hermes sessions delete <id>` で削除してください。

## 作業ディレクトリの挙動

ACP セッションは、エディタの cwd を Hermes のタスク ID にバインドするため、ファイルツールとターミナルツールはサーバープロセスの cwd ではなく、エディタのワークスペースを基準に実行されます。

## 承認

危険なターミナルコマンドは、承認プロンプトとしてエディタに送り返すことができます。ACP の承認オプションは CLI のフローよりシンプルです:

- 一度だけ許可
- 常に許可
- 拒否

実際にプロンプトが表示されるかどうかはホスト次第です。ホストは、リクエストをあなたに
表示する代わりに、プログラムで応答してもかまいません。その場合、これらのオプションは
通信上は存在しますが、人間に届くことはありません。Buzz Desktop はこれを行うため、
`approvals` の設定にかかわらず、その経路は無人実行として扱ってください。

タイムアウトまたはエラーの場合、承認ブリッジはリクエストを拒否します。待機時間は
`config.yaml` の `approvals.timeout`（デフォルト 300 秒）で、CLI やゲートウェイの
プロンプトが使うのと同じ設定です — エディタが承認カードをより長く開いたままにする場合は値を引き上げてください。

### セッションスコープの編集自動承認

ACP は、*一度だけ許可*と*常に許可*の間に 3 つ目の段階を公開しています: **Allow for session** です。エディタの権限プロンプトでこれを選ぶと、承認は現在の ACP セッション内にのみ記録されます — そのセッション内で以降に一致するコマンドはすべてプロンプトなしで通過しますが、新しい ACP セッション（またはエディタの再起動）ではリセットされ、最初の 1 回で再びプロンプトが表示されます。

| オプション | エディタのラベル | スコープ | 再起動をまたいで永続化 |
|---|---|---|---|
| `allow_once` | Allow once | この 1 回のツール呼び出し | いいえ |
| `allow_session` | Allow for session | この ACP セッション内で一致するすべての呼び出し | いいえ — セッション終了時にクリア |
| `allow_always` | Allow always | 今後のすべてのセッション | はい（Hermes の永続的な許可リストに書き込まれる） |
| `deny` | Deny | この 1 回のツール呼び出し | いいえ |

`allow_session` は、タスクの間はエージェントを信頼するものの、長期間有効な許可リストのエントリは付与したくないエディタワークフローにとって適切なデフォルトです。安全性のトレードオフは単純です: スコープが広いほどエディタに中断されることは減りますが、誤動作するエージェント（またはプロンプトインジェクション）が、気づく前に与えうる被害は大きくなります。見慣れないコマンドには `allow_once` から始め、エージェントが同じパターンを何度か正しく実行するのを確認したら `allow_session` に昇格させ、`allow_always` は永続的に信頼できる真に冪等なコマンド（例: `git status`）のために取っておいてください。

ACP ブリッジは、これらのオプションを Hermes の内部の承認セマンティクスにマッピングします — `allow_always` は CLI と同じ方法で永続的な許可リストのエントリを書き込み、`allow_session` は現在の ACP セッションのプロセス内承認キャッシュにのみ影響します。

## トラブルシューティング

### ACP エージェントがエディタに表示されない

確認すること:

- 手動/ローカル開発の場合、ホストのコマンドが `hermes acp` を指していることを確認する
- Hermes がインストールされ、PATH 上にある
- ACP のエクストラがインストールされている（`cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['acp'], explicit=True)"`）

### ACP は起動するがすぐにエラーになる

次のチェックを試してください:

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### クレデンシャルがない

ACP モードは Hermes の既存のプロバイダーセットアップを使用します。次でクレデンシャルを設定します:

```bash
hermes model
```

または `~/.hermes/.env` を編集します。ターミナル認証フロー（`hermes acp --setup`）からも、対話的なプロバイダー/モデルのセットアップを起動できます。

## 関連項目

- [Buzz ACP harness](https://github.com/block/buzz/tree/main/crates/buzz-acp)
- [ACP Internals](/docs/developer-guide/acp-internals)
- [Provider Runtime Resolution](/docs/developer-guide/provider-runtime)
- [Tools Runtime](/docs/developer-guide/tools-runtime)
