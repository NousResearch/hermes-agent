---
sidebar_position: 15
title: "MiniMax OAuth"
description: "ブラウザ OAuth で MiniMax にログインし、API キー不要で Hermes Agent から MiniMax-M2.7 モデルを使う"
---

# MiniMax OAuth

Hermes Agent は、[MiniMax ポータル](https://www.minimax.io) と同じクレデンシャルを使ったブラウザベースの OAuth ログインフローを通じて **MiniMax** をサポートします。API キーやクレジットカードは不要です。一度ログインすれば、Hermes が自動的にセッションを更新します。

トランスポートは `anthropic_messages` アダプターを再利用します（MiniMax は `/anthropic` で Anthropic Messages 互換のエンドポイントを公開しています）。そのため、既存のツールコール、ストリーミング、コンテキスト機能はアダプターを一切変更せずに動作します。

## 概要

| 項目 | 値 |
|------|-----|
| プロバイダー ID | `minimax-oauth` |
| 表示名 | MiniMax (OAuth) |
| 認証タイプ | ブラウザ OAuth（PKCE デバイスコードフロー） |
| トランスポート | Anthropic Messages 互換（`anthropic_messages`） |
| モデル | `MiniMax-M2.7`、`MiniMax-M2.7-highspeed` |
| グローバルエンドポイント | `https://api.minimax.io/anthropic` |
| 中国エンドポイント | `https://api.minimaxi.com/anthropic` |
| 必要な環境変数 | なし（`MINIMAX_API_KEY` はこのプロバイダーでは**使用されません**） |

## 前提条件

- Python 3.9+
- Hermes Agent がインストールされていること
- [minimax.io](https://www.minimax.io)（グローバル）または [minimaxi.com](https://www.minimaxi.com)（中国）の MiniMax アカウント
- ローカルマシンで利用可能なブラウザ（またはリモートセッション用に `--no-browser` を使用）

## クイックスタート

```bash
# プロバイダーとモデルのピッカーを起動
hermes model
# → プロバイダーリストから "MiniMax (OAuth)" を選択
# → Hermes が MiniMax 認可ページをブラウザで開く
# → ブラウザでアクセスを承認
# → モデルを選択（MiniMax-M2.7 または MiniMax-M2.7-highspeed）
# → チャット開始

hermes
```

初回ログイン後、クレデンシャルは `~/.hermes/auth.json` 配下に保存され、各セッションの前に自動的に更新されます。

## 手動でのログイン

モデルピッカーを経由せずにログインをトリガーできます:

```bash
hermes auth add minimax-oauth
```

### 中国リージョン

アカウントが中国プラットフォーム（`minimaxi.com`）にある場合は、代わりに中国リージョンの OAuth プロバイダー id `minimax-cn` を使うか、OAuth をスキップして `MINIMAX_CN_API_KEY` / `MINIMAX_CN_BASE_URL` を直接設定してください。古いドキュメントで説明されている `--region cn` フラグは CLI の引数パーサーに**配線されていません**。代わりに `minimax-cn` プロバイダーを使用してください:

```bash
hermes auth add minimax-cn --type oauth   # CN アカウントで OAuth がサポートされている場合
# またはよりシンプルに:
echo 'MINIMAX_CN_API_KEY=your-key' >> ~/.hermes/.env
```

### リモート / ヘッドレスセッション

ブラウザが利用できないサーバーやコンテナでは:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes は検証用 URL とユーザーコードを出力します。任意のデバイスで URL を開き、プロンプトが表示されたらコードを入力してください。

## OAuth フロー

Hermes は MiniMax の OAuth エンドポイントに対して PKCE デバイスコードフローを実装しています:

1. Hermes が PKCE verifier / challenge のペアとランダムな state 値を生成します。
2. challenge を付けて `{base_url}/oauth/code` に POST し、`user_code` と `verification_uri` を受け取ります。
3. ブラウザが `verification_uri` を開きます。プロンプトが表示されたら `user_code` を入力します。
4. Hermes はトークンが到着するまで（または期限が過ぎるまで）`{base_url}/oauth/token` をポーリングします。
5. トークン（`access_token`、`refresh_token`、有効期限）が `~/.hermes/auth.json` の `minimax-oauth` キー配下に保存されます。

トークンの更新（標準的な OAuth `refresh_token` グラント）は、アクセストークンの有効期限が 60 秒以内になったときに、各セッション開始時に自動的に実行されます。

## ログイン状態の確認

```bash
hermes doctor
```

`◆ Auth Providers` セクションには次のように表示されます:

```
✓ MiniMax OAuth  (logged in, region=global)
```

または、ログインしていない場合:

```
⚠ MiniMax OAuth  (not logged in)
```

## モデルの切り替え

```bash
hermes model
# → "MiniMax (OAuth)" を選択
# → モデルリストから選択
```

または、モデルを直接設定します:

```bash
hermes config set model MiniMax-M2.7
hermes config set provider minimax-oauth
```

## 設定リファレンス

ログイン後、`~/.hermes/config.yaml` には次のようなエントリが含まれます:

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### リージョンエンドポイント

| プロバイダー id | ポータル | 推論エンドポイント |
|-------------|--------|-------------------|
| `minimax-oauth`（グローバル） | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `minimax-cn`（中国） | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### プロバイダーエイリアス

次のすべては `minimax-oauth` に解決されます:

```bash
hermes --provider minimax-oauth    # 正式名
hermes --provider minimax-portal   # エイリアス
hermes --provider minimax-global   # エイリアス
hermes --provider minimax_oauth    # エイリアス（アンダースコア形式）
```

## 環境変数

`minimax-oauth` プロバイダーは `MINIMAX_API_KEY` や `MINIMAX_BASE_URL` を**使用しません**。これらの変数は、API キーベースの `minimax` および `minimax-cn` プロバイダー専用です。

| 変数 | 効果 |
|----------|--------|
| `MINIMAX_API_KEY` | `minimax` プロバイダーのみで使用 — `minimax-oauth` では無視される |
| `MINIMAX_CN_API_KEY` | `minimax-cn` プロバイダーのみで使用 — `minimax-oauth` では無視される |

実行時に `minimax-oauth` プロバイダーを強制するには:

```bash
HERMES_INFERENCE_PROVIDER=minimax-oauth hermes
```

## モデル

| モデル | 最適な用途 |
|-------|----------|
| `MiniMax-M2.7` | 長コンテキスト推論、複雑なツールコール |
| `MiniMax-M2.7-highspeed` | より低レイテンシ、軽量タスク、補助的な呼び出し |

両モデルとも最大 200,000 トークンのコンテキストをサポートします。

`MiniMax-M2.7-highspeed` は、`minimax-oauth` がプライマリプロバイダーである場合、ビジョンおよび委譲タスクの補助モデルとしても自動的に使用されます。

## トラブルシューティング

### トークンの有効期限切れ — 自動で再ログインされない

Hermes は、アクセストークンの有効期限が 60 秒以内であれば、各セッション開始時にトークンを更新します。アクセストークンがすでに有効期限切れの場合（例えば長時間オフラインだった後など）、更新は次のリクエスト時に自動的に行われます。更新が `refresh_token_reused` または `invalid_grant` で失敗した場合、Hermes はそのセッションを再ログインが必要としてマークします。

**対処:** `hermes auth add minimax-oauth` を再実行して、新しいログインを開始してください。

### 認可がタイムアウトした

デバイスコードフローには有限の有効期限ウィンドウがあります。時間内にログインを承認しないと、Hermes はタイムアウトエラーを発生させます。

**対処:** `hermes auth add minimax-oauth`（または `hermes model`）を再実行してください。フローが最初からやり直されます。

### state の不一致（CSRF の可能性）

Hermes は、認可サーバーから返された `state` 値が送信したものと一致しないことを検出しました。

**対処:** ログインを再実行してください。それでも続く場合は、OAuth レスポンスを変更しているプロキシやリダイレクトがないか確認してください。

### リモートサーバーからのログイン

`hermes` がブラウザウィンドウを開けない場合は、`--no-browser` を使用してください:

```bash
hermes auth add minimax-oauth --no-browser
```

Hermes が URL とコードを出力します。任意のデバイスで URL を開き、そこでフローを完了してください。

### 実行時の「Not logged into MiniMax OAuth」エラー

認証ストアに `minimax-oauth` のクレデンシャルがありません。まだログインしていないか、クレデンシャルファイルが削除されています。

**対処:** `hermes model` を実行して MiniMax (OAuth) を選択するか、`hermes auth add minimax-oauth` を実行してください。

## ログアウト

保存された MiniMax OAuth のクレデンシャルを削除するには:

```bash
hermes auth remove minimax-oauth
```

## 関連項目

- [AI Providers reference](../integrations/providers.md)
- [Environment Variables](../reference/environment-variables.md)
- [Configuration](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
