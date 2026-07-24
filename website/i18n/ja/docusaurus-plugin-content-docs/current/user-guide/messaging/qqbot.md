# QQ Bot

**公式 QQ Bot API（v2）** 経由で Hermes を QQ に接続します — プライベート（C2C）、グループの @ メンション、ギルド、ダイレクトメッセージを音声文字起こしとともにサポートします。

## 概要

QQ Bot アダプターは [公式 QQ Bot API](https://bot.q.qq.com/wiki/develop/api-v2/) を使用して以下を行います:

- QQ ゲートウェイへの永続的な **WebSocket** 接続を介してメッセージを受信します
- **REST API** 経由でテキストおよび Markdown の返信を送信します
- 画像、ボイスメッセージ、ファイル添付をダウンロードして処理します
- Tencent の組み込み ASR または設定可能な STT プロバイダーを使用してボイスメッセージを文字起こしします

## 前提条件

1. **QQ Bot アプリケーション** — [q.qq.com](https://q.qq.com) で登録します:
   - 新しいアプリケーションを作成し、**App ID** と **App Secret** をメモします
   - 必要なインテントを有効にします: C2C メッセージ、グループ @ メッセージ、ギルドメッセージ
   - テスト用にボットをサンドボックスモードで設定するか、本番用に公開します

2. **依存関係** — アダプターには `aiohttp` と `httpx` が必要です:
   ```bash
   pip install aiohttp httpx
   ```

## 設定

### インタラクティブセットアップ

```bash
hermes gateway setup
```

プラットフォームリストから **QQ Bot** を選択し、プロンプトに従います。

### 手動設定

`~/.hermes/.env` で必要な環境変数を設定します:

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## 環境変数

| 変数 | 説明 | デフォルト |
|---|---|---|
| `QQ_APP_ID` | QQ Bot App ID（必須） | — |
| `QQ_CLIENT_SECRET` | QQ Bot App Secret（必須） | — |
| `QQBOT_HOME_CHANNEL` | cron/通知配信用の OpenID | — |
| `QQBOT_HOME_CHANNEL_NAME` | ホームチャンネルの表示名 | `Home` |
| `QQ_ALLOWED_USERS` | DM アクセス用のカンマ区切りユーザー OpenID | open（すべてのユーザー） |
| `QQ_GROUP_ALLOWED_USERS` | グループアクセス用のカンマ区切りグループ OpenID | — |
| `QQ_ALLOW_ALL_USERS` | すべての DM を許可するには `true` に設定 | `false` |
| `QQ_PORTAL_HOST` | QQ ポータルホストを上書き（サンドボックスルーティングには `sandbox.q.qq.com` に設定） | `q.qq.com` |
| `QQ_STT_API_KEY` | 音声テキスト変換プロバイダーの API キー | — |
| `QQ_STT_BASE_URL` | （直接読み込まれません — 代わりに `config.yaml` で `platforms.qqbot.extra.stt.baseUrl` を設定してください） | 該当なし |
| `QQ_STT_MODEL` | STT モデル名 | `glm-asr` |

## 詳細設定

きめ細かい制御のために、`~/.hermes/config.yaml` にプラットフォーム設定を追加します:

```yaml
platforms:
  qqbot:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # QQ markdown を有効化（msg_type 2）。設定のみ。環境変数の同等物なし。
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled
      group_allow_from:
        - "group_openid_1"
      stt:
        provider: "zai"          # zai (GLM-ASR)、openai (Whisper) など
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## ボイスメッセージ（STT）

音声の文字起こしは 2 段階で動作します:

1. **QQ 組み込み ASR**（無料、常に最初に試行） — QQ はボイスメッセージの添付ファイルに `asr_refer_text` を提供し、これは Tencent 独自の音声認識を使用します
2. **設定された STT プロバイダー**（フォールバック） — QQ の ASR がテキストを返さない場合、アダプターは OpenAI 互換の STT API を呼び出します:

   - **Zhipu/GLM (zai)**: デフォルトプロバイダー、`glm-asr` モデルを使用します
   - **OpenAI Whisper**: `QQ_STT_BASE_URL` と `QQ_STT_MODEL` を設定します
   - 任意の OpenAI 互換 STT エンドポイント

## トラブルシューティング

### ボットが即座に切断される（クイック切断）

これは通常、次のことを意味します:
- **無効な App ID / Secret** — q.qq.com で認証情報を再確認してください
- **権限の欠落** — ボットに必要なインテントが有効になっていることを確認してください
- **サンドボックス専用ボット** — ボットがサンドボックスモードの場合、QQ のサンドボックステストチャンネルからのメッセージのみを受信できます

### ボイスメッセージが文字起こしされない

1. QQ 組み込みの `asr_refer_text` が添付ファイルのデータに存在するか確認してください
2. カスタム STT プロバイダーを使用している場合は、`QQ_STT_API_KEY` が正しく設定されていることを確認してください
3. STT エラーメッセージについてゲートウェイログを確認してください

### メッセージが配信されない

- q.qq.com でボットの **インテント** が有効になっていることを確認してください
- DM アクセスが制限されている場合は `QQ_ALLOWED_USERS` を確認してください
- グループメッセージの場合、ボットが **@ メンション** されていることを確認してください（グループポリシーで許可リスト化が必要な場合があります）
- cron/通知配信については `QQBOT_HOME_CHANNEL` を確認してください

### 接続エラー

- `aiohttp` と `httpx` がインストールされていることを確認してください: `pip install aiohttp httpx`
- `api.sgroup.qq.com` と WebSocket ゲートウェイへのネットワーク接続を確認してください
- 詳細なエラーメッセージと再接続の動作についてゲートウェイログを確認してください
