# Computer Use（macOS）

Hermes Agent は Mac のデスクトップを操作できます — クリック、入力、スクロール、
ドラッグを**バックグラウンド**で行います。カーソルは動かず、キーボードフォーカスも
変わらず、macOS が勝手に Space を切り替えることもありません。あなたとエージェントは
同じマシン上で共同作業します。

ほとんどの computer-use 連携と異なり、これは**ツール対応の任意のモデル**で動作します
— Claude、GPT、Gemini、またはローカルの vLLM エンドポイント上のオープンモデルなど。
Anthropic ネイティブのスキーマを気にする必要はありません。

## 仕組み

`computer_use` ツールセットは、stdio 上で MCP を話して [`cua-driver`](https://github.com/trycua/cua)
と通信します。これは SkyLight のプライベート SPI（`SLEventPostToPid`、
`SLPSPostEventRecordTo`）と `_AXObserverAddNotificationAndCheckRemote`
アクセシビリティ SPI を使う macOS ドライバーで、次を行います:

- 合成イベントをターゲットプロセスに直接ポスト — HID イベントタップも
  カーソルのワープもなし。
- ウィンドウを前面に出さずに AppKit のアクティブ状態を切り替え — Space の
  切り替えなし。
- ウィンドウが隠れているときも Chromium/Electron のアクセシビリティツリーを
  生かし続ける。

この組み合わせは、OpenAI の Codex の「バックグラウンド computer-use」が出荷しているものです。
cua-driver はそのオープンソース版です。

## 有効化

最も都合のよいパスを選んでください — どちらも同じ上流のインストーラーを実行します:

**オプション 1: 専用 CLI コマンド（最も直接的）。**

```
hermes computer-use install
```

これは上流の cua-driver インストーラーを取得して実行します:
`curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh`。
インストールの確認には `hermes computer-use status` を使います。

**オプション 2: ツールセットを対話的に有効化する。**

1. `hermes tools` を実行し、`🖱️ Computer Use (macOS)` → `cua-driver (background)` を選択。
2. セットアップが上流のインストーラーを実行します（オプション 1 と同じ）。

インストール後は、どちらのパスを取ったかに関わらず:

3. プロンプトが出たら macOS の権限を付与します:
   - **システム設定 → プライバシーとセキュリティ → アクセシビリティ** → ターミナル
     （または Hermes アプリ）を許可。
   - **システム設定 → プライバシーとセキュリティ → 画面収録** → 同じものを許可。
4. ツールセットを有効にしてセッションを開始します:
   ```
   hermes -t computer_use chat
   ```
   または `~/.hermes/config.yaml` の有効なツールセットに `computer_use` を追加します。

## クイック例

ユーザープロンプト: *「Stripe からの最新のメールを見つけて、何をしてほしいのか要約して。」*

エージェントの計画:

1. `computer_use(action="capture", mode="som", app="Mail")` — すべてのサイドバー
   項目、ツールバーボタン、メッセージ行に番号が振られた Mail のスクリーンショットを
   取得。
2. `computer_use(action="click", element=14)` — 検索フィールド（キャプチャの
   要素 #14）をクリック。
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` — 送信して
   新しいスクリーンショットを取得。
5. 上位の結果をクリックし、本文を読み、要約する。

この間ずっと、カーソルはあなたが置いた場所にとどまり、Mail が前面に出ることは
ありません。

## プロバイダー互換性

| プロバイダー | Vision? | 動作? | 備考 |
|---|---|---|---|
| Anthropic（Claude Sonnet/Opus 3+） | ✅ | ✅ | 総合的に最良；SOM ＋生の座標。 |
| OpenRouter（任意の vision モデル） | ✅ | ✅ | マルチパートのツールメッセージをサポート。 |
| OpenAI（GPT-4+、GPT-5） | ✅ | ✅ | 上記と同じ。 |
| ローカル vLLM / LM Studio（vision モデル） | ✅ | ✅ | モデルがマルチパートのツールコンテンツをサポートする場合。 |
| テキストのみのモデル | ❌ | ✅（機能制限） | アクセシビリティツリーのみの操作には `mode="ax"` を使用。 |

スクリーンショットはツール結果と一緒に OpenAI スタイルの `image_url`
パートとしてインラインで送信されます。Anthropic の場合、アダプターはそれらを
ネイティブの `tool_result` 画像ブロックに変換します。

## 安全性

Hermes は多層のガードレールを適用します:

- 破壊的なアクション（click、type、drag、scroll、key、focus_app）には承認が
  必要です — CLI ダイアログ経由で対話的に、またはメッセージングプラットフォームの
  承認ボタン経由のいずれかです。
- ツールレベルでハードブロックされるキーの組み合わせ: ゴミ箱を空にする、強制削除、
  画面ロック、ログアウト、強制ログアウト。
- ハードブロックされる入力パターン: `curl | bash`、`sudo rm -rf /`、フォーク
  ボムなど。
- エージェントのシステムプロンプトは明示的にこう伝えます: 権限ダイアログをクリック
  しない、パスワードを入力しない、スクリーンショットに埋め込まれた指示に従わない。

すべてのアクションを確認したい場合は、`~/.hermes/config.yaml` の `approvals.mode: manual` と組み合わせてください。

## トークン効率

スクリーンショットは高コストです。Hermes は 4 層の最適化を適用します:

- **スクリーンショットの退避** — Anthropic アダプターは最新の 3 枚のスクリーンショット
  のみをコンテキストに保持します。古いものは `[screenshot removed
  to save context]` のプレースホルダーになります。
- **クライアント側の圧縮プルーニング** — コンテキスト圧縮器がマルチモーダルな
  ツール結果を検出し、古いものから画像パートを除去します。
- **画像対応のトークン推定** — 各画像は base64 の文字数ではなく、約 1500 トークン
  （Anthropic の定額レート）として数えられます。
- **サーバー側のコンテキスト編集（Anthropic のみ）** — 有効な場合、アダプターは
  `context_management` 経由で `clear_tool_uses_20250919` を有効化し、Anthropic の
  API が古いツール結果をサーバー側でクリアします。

1568×900 のディスプレイでの 20 アクションのセッションは、典型的に約 600K トークン
ではなく約 30K トークンのスクリーンショットコンテキストで済みます。

## 制限事項

- **macOS のみ。** cua-driver は Linux や Windows には存在しない Apple のプライベート
  SPI を使います。クロスプラットフォームの GUI 自動化には `browser`
  ツールセットを使ってください。
- **プライベート SPI のリスク。** Apple は OS アップデートで SkyLight のシンボル
  サーフェスをいつでも変更できます。macOS のバージョンアップをまたいで再現性が
  欲しい場合は、`HERMES_CUA_DRIVER_VERSION`
  env 変数でドライバーのバージョンを固定してください。
- **パフォーマンス。** バックグラウンドモードはフォアグラウンドより遅くなります —
  SkyLight 経由のイベントは直接の HID ポストの約 5〜20ms に対して時間がかかります。
  エージェント速度のクリックでは気づきませんが、スピードランを記録しようとすると
  気づきます。
- **キーボードによるパスワード入力なし。** `type` はコマンドシェルのペイロードに
  対してハードブロックパターンを持ちます。パスワードにはシステムの自動入力を
  使ってください。

## 設定

ドライバーのバイナリパスを上書き（テスト / CI）:

```
HERMES_CUA_DRIVER_CMD=/opt/homebrew/bin/cua-driver
HERMES_CUA_DRIVER_VERSION=0.5.0    # 任意の固定
```

バックエンドを完全に差し替え（テスト用）:

```
HERMES_COMPUTER_USE_BACKEND=noop   # 呼び出しを記録するが副作用なし
```

## トラブルシューティング

**`computer_use backend unavailable: cua-driver is not installed`** —
`hermes computer-use install` を実行して cua-driver バイナリを取得するか、
`hermes tools` を実行して Computer Use ツールセットを有効化してください。

**クリックが効いていないように見える** — キャプチャして確認してください。見えて
いないモーダルが入力をブロックしているかもしれません。`escape` または閉じる
ボタンで閉じてください。

**要素インデックスが古い** — SOM インデックスは次の `capture` までしか有効で
ありません。状態を変えるアクションの後は再キャプチャしてください。

**「blocked pattern in type text」** — `type` しようとしたテキストが危険な
シェルパターンのリストに一致しています。コマンドを分割するか、再考してください。

## 関連項目

- [ユニバーサルスキル: `macos-computer-use`](https://github.com/NousResearch/hermes-agent/blob/main/skills/apple/macos-computer-use/SKILL.md)
- [cua-driver ソース（trycua/cua）](https://github.com/trycua/cua)
- クロスプラットフォームの Web タスクには[ブラウザ自動化](./browser.md)。
