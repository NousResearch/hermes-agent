---
sidebar_position: 16
title: "Google Gemini"
description: "Hermes AgentをGoogle Geminiで使う — ネイティブAI Studio API、APIキーのセットアップ、OAuthオプション、ツール呼び出し、ストリーミング、クォータガイダンス"
---

# Google Gemini

Hermes Agentは、**Google AI Studio / Gemini API** を使ってGoogle Geminiをネイティブプロバイダーとしてサポートします（OpenAI互換エンドポイントではありません）。これにより、Hermesは内部のOpenAI形式のメッセージとツールループを、ツール呼び出し、ストリーミング、マルチモーダル入力、Gemini固有のレスポンスメタデータを保持しながら、Geminiのネイティブ `generateContent` APIに変換できます。

Hermesは、GoogleのGemini CLIと同じCloud Code Assistバックエンドを使用する別の **Google Gemini（OAuth）** プロバイダーもサポートしています。最もリスクの低い公式APIパスには、APIキープロバイダー（`gemini`）を使用してください。

## 前提条件

- **Google AI Studio APIキー** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)で作成します
- **課金が有効なGoogle Cloudプロジェクト** — エージェント利用には推奨されます。Geminiの無料枠は、Hermesがユーザーのターンごとに複数のモデル呼び出しを行う可能性があるため、長時間実行のエージェントセッションには小さすぎます。
- **Hermesがインストールされていること** — ネイティブGeminiプロバイダーには追加のPythonパッケージは不要です。

:::tip APIキーのパス
`GOOGLE_API_KEY` または `GEMINI_API_KEY` を設定してください。Hermesは `gemini` プロバイダーに対して両方の名前を確認します。
:::

## クイックスタート

```bash
# Gemini APIキーを追加する
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# プロバイダーとしてGeminiを選択する
hermes model
# → "More providers..." → "Google AI Studio" を選択
# → Hermesがキーのティアを確認し、Geminiモデルを表示
# → モデルを選択

# チャットを開始する
hermes chat
```

直接設定を編集したい場合は、ネイティブGemini APIのベースURLを使用します。

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## 設定

`hermes model` を実行すると、`~/.hermes/config.yaml` には次が含まれます。

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

そして `~/.hermes/.env` には:

```bash
GOOGLE_API_KEY=...
```

### ネイティブGemini API

推奨されるエンドポイントは次のとおりです。

```text
https://generativelanguage.googleapis.com/v1beta
```

Hermesはこのエンドポイントを検出し、ネイティブGeminiアダプターを作成します。内部的には、Hermesはエージェントループを引き続きOpenAI形式のメッセージで保持し、各リクエストをGeminiのネイティブスキーマに変換します。

- `messages[]` → Gemini `contents[]`
- システムプロンプト → Gemini `systemInstruction`
- ツールスキーマ → Gemini `functionDeclarations`
- ツール結果 → Gemini `functionResponse` パーツ
- ストリーミング応答 → Hermesループ用のOpenAI形式のストリームチャンク

:::note Gemini 3のthought signature
Gemini 3のツール使用では、Hermesは関数呼び出しパーツに付随する `thoughtSignature` の値を保持し、次のツールターンで再生します。これは複数ステップのエージェントワークフローにおける検証上重要なパスをカバーします。

Gemini 3は、他のレスポンスパーツにもthought signatureを付加する場合があります。Hermesのネイティブアダプターは現在エージェントのツールループに最適化されているため、ツール呼び出し以外のすべてのシグネチャをパーツレベルで完全な忠実度で再生するわけではまだありません。
:::

### ネイティブエンドポイントを優先する

GoogleはOpenAI互換のエンドポイントも公開しています。

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

Hermesのエージェントセッションでは、上記のネイティブGeminiエンドポイントを優先してください。Hermesにはネイティブのアダプターが含まれているため、複数ターンのツール使用、ツール呼び出しの結果、ストリーミング、マルチモーダル入力、Geminiのレスポンスメタデータを、Geminiの `generateContent` APIに直接マッピングできます。OpenAI互換エンドポイントは、OpenAI APIの互換性が特に必要な場合に依然として有用です。

以前 `GEMINI_BASE_URL` を `/openai` のURLに設定していた場合は、それを削除するか変更してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### OAuthプロバイダー

Hermesには `google-gemini-cli` プロバイダーもあります。

```bash
hermes model
# → "Google Gemini (OAuth)" を選択
```

これはブラウザPKCEログインとCloud Code Assistバックエンドを使用します。Gemini CLIスタイルのOAuthを望むユーザーには有用な場合がありますが、GoogleがサードパーティソフトウェアからのGemini CLI OAuthクライアントの使用をポリシー違反とみなす可能性があるため、Hermesは明示的な警告を表示します。本番環境または最もリスクの低い利用には、上記のAPIキープロバイダーを優先してください。

## 利用可能なモデル

`hermes model` のピッカーは、Hermesのプロバイダーレジストリで管理されているGeminiモデルを表示します。一般的な選択肢には次が含まれます。

| モデル | ID | 備考 |
|-------|----|-------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | 利用可能な場合、最も高性能なプレビューモデル |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 強力な推論とコーディングのモデル |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | 推奨されるデフォルト。速度と性能のバランス |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | 利用可能な場合、最速/最低コストのオプション |

モデルの利用可否は時間とともに変化します。モデルが消えたり、あなたのキーで有効化されていない場合は、`hermes model` を再度実行し、現在のリストから1つ選んでください。

:::info モデルID
`provider: gemini` の場合は、`google/gemini-3-flash-preview` のようなOpenRouterスタイルのIDではなく、`gemini-3-flash-preview` のようなGeminiのネイティブモデルIDを使用してください。
:::

### 最新エイリアス

Googleは、ProとFlashのGeminiファミリー向けに移動するエイリアスを公開しています。`gemini-pro-latest` と `gemini-flash-latest` は、Hermesの設定を変更せずにGoogleにモデルを自動的に進化させてほしい場合に便利です。

| エイリアス | 現在の追跡対象 | 備考 |
|-------|------------------|-------|
| `gemini-pro-latest` | 最新のGemini Proモデル | Googleの現在のProデフォルトを使いたい場合に最適 |
| `gemini-flash-latest` | 最新のGemini Flashモデル | Googleの現在のFlashデフォルトを使いたい場合に最適 |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

厳密な再現性が必要な場合は、`gemini-3.1-pro-preview` や `gemini-3-flash-preview` のような明示的なモデルIDを優先してください。

### Gemini API経由のGemma

GoogleはGemma モデルもGemini API経由で公開しています。HermesはこれらをGoogleモデルとして認識しますが、新規ユーザーが長時間実行のエージェントセッションに評価ティアのモデルを誤って選択しないように、非常にスループットの低いGemmaエントリをデフォルトのモデルピッカーから隠しています。

有用な評価用IDには次が含まれます。

| モデル | ID | 備考 |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | より大きなGemmaモデル。互換性と品質の評価に有用 |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | 利用可能な場合、より小さなアクティブパラメータのバリアント |

これらのモデルは、Gemini APIキー上の評価オプションとして扱うのが最適です。GoogleのGemma APIの料金は無料枠のみで、利用上限は本番のGeminiモデルと比べて低いため、Hermesエージェントの持続的な利用は通常、有料のGeminiモデル、セルフホスト型のデプロイ、または適切なクォータを持つ別のプロバイダーに移行すべきです。

ピッカーから隠されているGemmaモデルを使用するには、直接設定します。

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## セッション中のモデル切り替え

会話中に `/model` コマンドを使用します。

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

まだGeminiを設定していない場合は、セッションを終了して先に `hermes model` を実行してください。`/model` は、すでに設定済みのプロバイダーとモデルの間で切り替えます。新しいAPIキーを収集することはありません。

## 診断

```bash
hermes doctor
```

doctorは次を確認します。

- `GOOGLE_API_KEY` または `GEMINI_API_KEY` が利用可能かどうか
- `google-gemini-cli` 用のGemini OAuth認証情報が存在するかどうか
- 設定されたプロバイダーの認証情報が解決できるかどうか

OAuthのクォータ使用状況については、Hermesセッション内で次を実行してください。

```text
/gquota
```

`/gquota` は `google-gemini-cli` のOAuthプロバイダーに適用され、AI StudioのAPIキープロバイダーには適用されません。

## ゲートウェイ（メッセージングプラットフォーム）

Geminiは、すべてのHermesゲートウェイプラットフォーム（Telegram、Discord、Slack、WhatsApp、LINE、Feishuなど）で機能します。Geminiをプロバイダーとして設定し、通常どおりゲートウェイを起動します。

```bash
hermes gateway setup
hermes gateway start
```

ゲートウェイは `config.yaml` を読み取り、同じGeminiプロバイダー設定を使用します。

## トラブルシューティング

### 「Gemini native client requires an API key」

Hermesが使用可能なAPIキーを見つけられませんでした。次のいずれかを `~/.hermes/.env` に追加してください。

```bash
GOOGLE_API_KEY=...
# または
GEMINI_API_KEY=...
```

その後、`hermes model` を再度実行してください。

### 「This Google API key is on the free tier」

Hermesはセットアップ中にGemini APIキーをプローブします。ツール使用、再試行、圧縮、補助タスクが複数のモデル呼び出しを必要とする可能性があるため、無料枠のクォータはわずか数ターンのエージェント操作で枯渇する場合があります。

キーに紐づくGoogle Cloudプロジェクトで課金を有効化し、必要に応じてキーを再生成してから、次を実行してください。

```bash
hermes model
```

### 「404 model not found」

選択したモデルは、あなたのアカウント、リージョン、またはキーで利用できません。`hermes model` を再度実行し、現在のリストから別のGeminiモデルを選んでください。

### `hermes model` にGemmaモデルが表示されない

Hermesは、スループットの低いGemmaモデルをデフォルトでピッカーから隠す場合があります。意図的に評価したい場合は、`~/.hermes/config.yaml` でモデルIDを直接設定してください。

### Gemmaで「429 quota exceeded」

Gemini API経由で公開されるGemmaモデルは評価に有用ですが、そのGemini API無料枠の上限は低いです。互換性テストに使用し、その後、持続的なエージェントセッションには有料のGeminiモデルまたは別のプロバイダーに切り替えてください。

### OpenAI互換エンドポイントが設定されている

`~/.hermes/.env` で次を確認してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

ネイティブエンドポイントに変更するか、オーバーライドを削除してください。

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### OAuthログインの警告

`google-gemini-cli` プロバイダーは、Gemini CLI / Cloud Code AssistのOAuthフローを使用します。これは公式のAI Studio APIキーのパスとは異なるため、Hermesは開始前に警告します。公式のAPIキー連携には、`GOOGLE_API_KEY` を伴う `provider: gemini` を使用してください。

### ツール呼び出しがスキーマエラーで失敗する

Hermesをアップグレードし、`hermes model` を再実行してください。ネイティブGeminiアダプターは、Geminiのより厳格な関数宣言形式に合わせてツールスキーマをサニタイズします。古いビルドやカスタムエンドポイントではそうでない場合があります。

## 関連

- [AIプロバイダー](/docs/integrations/providers)
- [設定](/docs/user-guide/configuration)
- [フォールバックプロバイダー](/docs/user-guide/features/fallback-providers)
- [AWS Bedrock](/docs/guides/aws-bedrock) — AWS認証情報を使用したネイティブなクラウドプロバイダー連携
