---
title: "Microsoft Graph アプリケーションの登録"
description: "Teams 会議パイプラインを支えるアプリ登録を作成するための Azure ポータルのウォークスルー"
---

# Microsoft Graph アプリケーションの登録

Teams 会議パイプラインは、**アプリのみ**（デーモン）認証を使って Microsoft Graph から会議の文字起こし、録画、関連成果物を読み取ります — ユーザーのサインインや会議ごとの対話的な同意は不要です。そのためには、管理者の同意を得たアプリケーション権限を持つ Azure AD アプリケーション登録が必要です。

このガイドでは次の手順を解説します:

1. アプリ登録の作成
2. クライアントシークレットの作成
3. パイプラインが必要とする Graph API 権限の付与
4. それらの権限への管理者同意
5. （任意）アプリケーションアクセスポリシーによる特定ユーザーへのアプリのスコープ設定

これを完了するには**テナント管理者権限**（または代わりに同意を付与してくれる管理者）が必要です。収集した値はブックマークしておいてください — 最後に `~/.hermes/.env` に入力します。

## 前提条件

- 会議の文字起こしと録画を生成する Teams Premium または Teams ライセンスを持つ Microsoft 365 テナント
- [entra.microsoft.com](https://entra.microsoft.com) の Azure ポータルへの管理者アクセス
- Graph の変更通知用に公開到達可能な HTTPS エンドポイント（後の Webhook リスナー手順で設定）

## ステップ 1: アプリ登録を作成する

1. [entra.microsoft.com](https://entra.microsoft.com) にテナント管理者としてサインインします。
2. **Identity → Applications → App registrations** に移動します。
3. **New registration** をクリックします。
4. 次を入力します:
   - **Name:** `Hermes Teams Meeting Pipeline`（または認識できる任意の名前）。
   - **Supported account types:** *Accounts in this organizational directory only (Single tenant)*。
   - **Redirect URI:** 空欄のまま — アプリのみ認証には不要です。
5. **Register** をクリックします。

アプリの概要ページに到達します。2 つの値をコピーします:

- **Application (client) ID** → `MSGRAPH_CLIENT_ID`
- **Directory (tenant) ID** → `MSGRAPH_TENANT_ID`

## ステップ 2: クライアントシークレットを作成する

1. 左ナビで **Certificates & secrets** を開きます。
2. **New client secret** をクリックします。
3. **Description:** `hermes-graph-secret`。**Expires:** ローテーションポリシーに合った値を選びます（6〜24 か月が一般的）。
4. **Add** をクリックします。
5. **Value** 列を直ちにコピーします — 一度しか表示されません。その値が `MSGRAPH_CLIENT_SECRET` です。

> **Secret ID** 列はシークレットではありません。必要なのは **Value** 列です。

## ステップ 3: Graph API 権限を付与する

パイプラインは最小限のアプリケーション権限セットを使用します。必要なものだけを追加してください。各権限は、アプリがテナント全体で読み取れる範囲を広げます。

1. 左ナビで **API permissions** を開きます。
2. **Add a permission** → **Microsoft Graph** → **Application permissions** をクリックします。
3. パイプラインに行わせたいことに合致する権限を下の表から追加します。
4. 追加後、**Grant admin consent for `<your tenant>`** をクリックします。Status 列はすべての権限で緑のチェックマークに変わるはずです。

### 文字起こし優先の要約に必須

| 権限 | アプリにできること |
|------------|--------------------------|
| `OnlineMeetings.Read.All` | Teams オンライン会議のメタデータ（件名、参加者、参加 URL）を読む。 |
| `OnlineMeetingTranscript.Read.All` | Teams が生成した会議の文字起こしを読む。 |

### 録画フォールバックに必須（文字起こしが利用できない場合）

| 権限 | アプリにできること |
|------------|--------------------------|
| `OnlineMeetingRecording.Read.All` | オフライン STT 処理のために Teams 会議の録画をダウンロードする。 |
| `CallRecords.Read.All` | 参加 URL のみが分かっている場合に通話レコードから会議を解決する。 |

### 送信要約の配信に必須（Graph モードのみ）

`platforms.teams.extra.delivery_mode` が `graph` の場合、パイプラインは Graph API 経由で Teams チャネルまたはチャットに要約を投稿します。代わりに `incoming_webhook` 配信モードを使う場合はこれらをスキップしてください。

| 権限 | アプリにできること |
|------------|--------------------------|
| `ChannelMessage.Send` | アプリに代わって Teams チャネルにメッセージを投稿する。 |
| `Chat.ReadWrite.All` | 1:1 およびグループチャットにメッセージを投稿する（配信先として `chat_id` を設定する場合のみ）。 |

### 非推奨

- `OnlineMeetings.ReadWrite.All` / `.All` なしの `Chat.ReadWrite` — パイプラインに必要な範囲より広すぎます。
- 委任権限 — パイプラインはアプリのみ（クライアント資格情報）フローを使うため、委任権限はユーザーサインインなしでは機能しません。

## ステップ 4:（推奨）アプリケーションアクセスポリシーでアプリをスコープ設定する

デフォルトでは、`OnlineMeetings.Read.All` のようなアプリケーション権限は、テナント内の**すべての**会議へのアクセスをアプリに付与します。パートナーデモや開発テナントではそれで問題ありませんが、本番環境では、アプリが読み取れる会議のユーザーをほぼ確実に制限したいはずです。

Microsoft はまさにこのために Teams 向けの**アプリケーションアクセスポリシー**を提供しています。このポリシーは PowerShell のみのサーフェスで、ポータル UI はありません。

MicrosoftTeams モジュールをインストールして接続した管理者 PowerShell（`Connect-MicrosoftTeams`）から:

```powershell
# Hermes アプリにスコープを限定したポリシーを作成
New-CsApplicationAccessPolicy `
  -Identity "Hermes-Meeting-Pipeline-Policy" `
  -AppIds "<MSGRAPH_CLIENT_ID>" `
  -Description "Restrict Hermes meeting pipeline to allow-listed users"

# パイプラインが会議を読み取れる特定ユーザーにポリシーを付与
Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "alice@example.com"

Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "bob@example.com"
```

付与後の反映には最大 30 分かかることがあります。次で検証します:

```powershell
Test-CsApplicationAccessPolicy -Identity "alice@example.com" -AppId "<MSGRAPH_CLIENT_ID>"
```

ポリシーがないと、**任意の**ユーザーの会議が読み取り可能になります — それが権限が技術的に付与する範囲です。本番テナントではこのステップをスキップしないでください。

## ステップ 5: 資格情報を Env ファイルに書き込む

収集した 3 つの値を `~/.hermes/.env` に入れます:

```bash
MSGRAPH_TENANT_ID=<directory-tenant-id>
MSGRAPH_CLIENT_ID=<application-client-id>
MSGRAPH_CLIENT_SECRET=<client-secret-value>
```

自分だけがシークレットを読めるようファイル権限を設定します:

```bash
chmod 600 ~/.hermes/.env
```

## ステップ 6: トークンフローを検証する

Hermes には Graph 認証のスモークテストが同梱されています。Hermes のインストール先から:

```python
python -c "
import asyncio
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
provider = MicrosoftGraphTokenProvider.from_env()
token = asyncio.run(provider.get_access_token())
print('Token acquired, length:', len(token))
print(provider.inspect_token_health())
"
```

成功すると、長いトークン文字列と、`cached: True` および 3600 に近い `expires_in_seconds` 値を示すヘルス dict が出力されます。失敗すると、Azure のエラーコードを伴う `MicrosoftGraphTokenError` が生成されます — 最も一般的なものは:

| Azure エラー | 意味 | 対処 |
|-------------|---------|-----|
| `AADSTS7000215: Invalid client secret` | シークレット値が不一致または期限切れ。 | ステップ 2 で新しいシークレットを生成し、`.env` を更新。 |
| `AADSTS700016: Application not found` | `MSGRAPH_CLIENT_ID` またはテナントが誤り。 | ステップ 1 の値が同じアプリのものか再確認。 |
| `AADSTS90002: Tenant not found` | `MSGRAPH_TENANT_ID` のタイプミス。 | アプリ概要から Directory (tenant) ID を再度コピー。 |
| 呼び出し時（トークン取得時ではない）の `insufficient_claims` | トークンは取得できるが Graph が 401/403 を返す。 | ステップ 3 の管理者同意をスキップしたか、権限を追加したが再同意していない。API permissions に戻り、再度 **Grant admin consent** をクリック。 |

## クライアントシークレットのローテーション

Azure のクライアントシークレットには厳格な有効期限があります。期限切れになる前に:

1. 最初のものを削除せずに、ステップ 2 で 2 つ目のクライアントシークレットを作成します。
2. `~/.hermes/.env` の `MSGRAPH_CLIENT_SECRET` を新しい値で更新します。
3. 新しいシークレットが読み込まれるようゲートウェイを再起動します: `hermes gateway restart`。
4. 上記のスモークテストで検証します。
5. 古いシークレットを Azure ポータルから削除します。

## 次のステップ

資格情報がクリーンに検証できたら、次に進みます:

- **Webhook リスナーのセットアップ** — Graph の変更通知を受け取る `msgraph_webhook` ゲートウェイプラットフォームを立ち上げます。
- **パイプライン設定** — Teams 会議パイプラインのランタイムとオペレーター CLI を設定します。
- **送信配信** — 要約を Teams チャネルまたはチャットへ配線します。

これらのページは、対応するランタイムを追加する PR と並んで提供されます。この資格情報セットアップは独立した前提条件であり、事前に完了しておいても安全です。
