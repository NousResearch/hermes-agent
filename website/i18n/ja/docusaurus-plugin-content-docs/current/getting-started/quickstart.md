---
sidebar_position: 1
title: "クイックスタート"
description: "Hermes Agent との最初の対話 — インストールから 5 分以内にチャットを始める"
---

# クイックスタート

このガイドは、ゼロの状態から、実運用に耐える Hermes のセットアップまで導きます。インストールし、プロバイダーを選び、チャットが動くことを確認し、何かが壊れたときに何をすべきかを正確に把握しましょう。

## 動画で見たい場合

**Onchain AI Garage** が、インストール、セットアップ、基本コマンドのマスタークラス解説をまとめています。動画で一緒に進めたい場合は、このページの良い補完になります。さらに詳しくは、[Hermes Agent Tutorials & Use Cases](https://www.youtube.com/channel/UCqB1bhMwGsW-yefBxYwFCCg) のプレイリスト全体を参照してください。

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## こんな方向け

- まったくの初めてで、動くセットアップまでの最短経路がほしい
- プロバイダーを切り替えていて、設定ミスで時間を無駄にしたくない
- チーム、ボット、常時稼働ワークフロー向けに Hermes をセットアップしたい
- 「インストールはできたのに、まだ何も動かない」状態にうんざりしている

## 最短経路

自分の目的に合う行を選んでください:

| 目的 | まずこれを | 次にこれを |
|---|---|---|
| とにかく自分のマシンで Hermes を動かしたい | `hermes setup` | 実際のチャットを実行し、応答することを確認 |
| 使うプロバイダーはもう分かっている | `hermes model` | 設定を保存してからチャットを開始 |
| ボットや常時稼働のセットアップがほしい | CLI が動いた後に `hermes gateway setup` | Telegram、Discord、Slack などのプラットフォームを接続 |
| ローカルまたはセルフホストのモデルを使いたい | `hermes model` → カスタムエンドポイント | エンドポイント、モデル名、コンテキスト長を確認 |
| 複数プロバイダーのフォールバックがほしい | まず `hermes model` | 基本のチャットが動いた後にだけルーティングとフォールバックを追加 |

**目安:** Hermes が通常のチャットを完了できないなら、まだ機能を追加しないでください。まずクリーンな対話を 1 つ動かし、その後にゲートウェイ、cron、スキル、音声、ルーティングを重ねていきます。

---

## 1. Hermes Agent のインストール

ワンラインインストーラーを実行します:

```bash
# Linux / macOS / WSL2 / Android (Termux)
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

:::tip Android / Termux
スマートフォンにインストールする場合は、テスト済みの手動経路、対応 extra、現在の Android 固有の制限について、専用の [Termux ガイド](./termux.md)を参照してください。
:::

:::tip Windows ユーザー
まず [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) をインストールし、その後 WSL2 ターミナル内で上記のコマンドを実行してください。
:::

完了したら、シェルを再読み込みします:

```bash
source ~/.bashrc   # または source ~/.zshrc
```

詳細なインストールオプション、前提条件、トラブルシューティングについては、[インストールガイド](./installation.md)を参照してください。

## 2. プロバイダーの選択

最も重要なセットアップ手順です。`hermes model` を使って対話的に選択を進めます:

```bash
hermes model
```

おすすめの既定:

| プロバイダー | 概要 | セットアップ方法 |
|----------|-----------|---------------|
| **Nous Portal** | サブスクリプション制、設定不要 | `hermes model` 経由の OAuth ログイン |
| **OpenAI Codex** | ChatGPT OAuth、Codex モデルを使用 | `hermes model` 経由のデバイスコード認証 |
| **Anthropic** | Claude モデルを直接 — Max プラン + 追加利用クレジット（OAuth）、またはトークン従量課金用の API キー | `hermes model` → OAuth ログイン（Max + 追加クレジットが必要）、または Anthropic API キー |
| **OpenRouter** | 多数のモデルにまたがるマルチプロバイダールーティング | API キーを入力 |
| **Z.AI** | GLM / Zhipu ホストのモデル | `GLM_API_KEY` / `ZAI_API_KEY` を設定 |
| **Kimi / Moonshot** | Moonshot ホストのコーディング・チャットモデル | `KIMI_API_KEY`（または Kimi-Coding 専用の `KIMI_CODING_API_KEY`）を設定 |
| **Kimi / Moonshot China** | 中国リージョンの Moonshot エンドポイント | `KIMI_CN_API_KEY` を設定 |
| **Arcee AI** | Trinity モデル | `ARCEEAI_API_KEY` を設定 |
| **GMI Cloud** | マルチモデルのダイレクト API | `GMI_API_KEY` を設定 |
| **MiniMax (OAuth)** | ブラウザ OAuth 経由の MiniMax-M2.7 — API キー不要 | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | 国際版 MiniMax エンドポイント | `MINIMAX_API_KEY` を設定 |
| **MiniMax China** | 中国リージョンの MiniMax エンドポイント | `MINIMAX_CN_API_KEY` を設定 |
| **Alibaba Cloud** | DashScope 経由の Qwen モデル | `DASHSCOPE_API_KEY` を設定 |
| **Hugging Face** | 統合ルーター経由の 20 以上のオープンモデル（Qwen、DeepSeek、Kimi など） | `HF_TOKEN` を設定 |
| **AWS Bedrock** | ネイティブ Converse API 経由の Claude、Nova、Llama、DeepSeek | IAM ロールまたは `aws configure`（[ガイド](../guides/aws-bedrock.md)） |
| **Kilo Code** | KiloCode ホストのモデル | `KILOCODE_API_KEY` を設定 |
| **OpenCode Zen** | 厳選モデルへの従量課金アクセス | `OPENCODE_ZEN_API_KEY` を設定 |
| **OpenCode Go** | オープンモデル向けの月額 10 ドルサブスクリプション | `OPENCODE_GO_API_KEY` を設定 |
| **DeepSeek** | DeepSeek API への直接アクセス | `DEEPSEEK_API_KEY` を設定 |
| **NVIDIA NIM** | build.nvidia.com またはローカル NIM 経由の Nemotron モデル | `NVIDIA_API_KEY` を設定（任意: `NVIDIA_BASE_URL`） |
| **GitHub Copilot** | GitHub Copilot サブスクリプション（GPT-5.x、Claude、Gemini など） | `hermes model` 経由の OAuth、または `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Copilot ACP エージェントバックエンド（ローカルの `copilot` CLI を起動） | `hermes model`（`copilot` CLI + `copilot login` が必要） |
| **Vercel AI Gateway** | Vercel AI Gateway ルーティング | `AI_GATEWAY_API_KEY` を設定 |
| **Custom Endpoint** | VLLM、SGLang、Ollama、または任意の OpenAI 互換 API | ベース URL + API キーを設定 |

ほとんどの初回ユーザーへ: プロバイダーを選び、変更する理由が分からない限り既定を受け入れてください。環境変数とセットアップ手順を含む完全なプロバイダーカタログは、[プロバイダー](../integrations/providers.md)ページにあります。

:::caution 最小コンテキスト: 64K トークン
Hermes Agent には、少なくとも **64,000 トークン**のコンテキストを持つモデルが必要です。これより小さいウィンドウのモデルは、複数ステップのツール呼び出しワークフローに必要な作業メモリを維持できず、起動時に拒否されます。ほとんどのホスト型モデル（Claude、GPT、Gemini、Qwen、DeepSeek）はこれを容易に満たします。ローカルモデルを動かしている場合は、コンテキストサイズを少なくとも 64K に設定してください（例: llama.cpp なら `--ctx-size 65536`、Ollama なら `-c 65536`）。
:::

:::tip
`hermes model` でいつでもプロバイダーを切り替えられます。ロックインはありません。対応する全プロバイダーとセットアップの詳細の一覧は、[AI プロバイダー](../integrations/providers.md)を参照してください。
:::

### 設定の保存方法

Hermes はシークレットを通常の設定から分離します:

- **シークレットとトークン** → `~/.hermes/.env`
- **シークレットでない設定** → `~/.hermes/config.yaml`

値を正しく設定する最も簡単な方法は CLI 経由です:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

適切な値が自動的に適切なファイルに振り分けられます。

## 3. 最初のチャットを実行

```bash
hermes            # クラシック CLI
hermes --tui      # モダンな TUI（推奨）
```

モデル、利用可能なツール、スキルを示すウェルカムバナーが表示されます。具体的で検証しやすいプロンプトを使いましょう:

:::tip インターフェースを選ぶ
Hermes には 2 つのターミナルインターフェースが付属します: クラシックな `prompt_toolkit` CLI と、モーダルオーバーレイ、マウス選択、ノンブロッキング入力を備えた新しい [TUI](../user-guide/tui.md) です。どちらも同じセッション、スラッシュコマンド、設定を共有します。`hermes` と `hermes --tui` でそれぞれ試してみてください。
:::

```
このリポジトリを 5 つの箇条書きで要約して、メインのエントリポイントが何か教えて。
```

```
現在のディレクトリを確認して、メインのプロジェクトファイルらしきものを教えて。
```

```
このコードベース向けにクリーンな GitHub PR ワークフローを設定するのを手伝って。
```

**成功の状態:**

- バナーに選択したモデル／プロバイダーが表示される
- Hermes がエラーなく応答する
- 必要に応じてツール（ターミナル、ファイル読み取り、Web 検索）を使える
- 対話が 1 ターンを超えて正常に続く

これが動けば、最も難しい部分は越えています。

## 4. セッションが機能することを確認

先に進む前に、再開が機能することを確認します:

```bash
hermes --continue    # 最新のセッションを再開
hermes -c            # 短縮形
```

これで、たった今行ったセッションに戻れるはずです。戻れない場合は、同じプロファイルにいるか、セッションが実際に保存されたかを確認してください。これは後で複数のセットアップやマシンを扱うときに重要になります。

## 5. 主要機能を試す

### ターミナルを使う

```
❯ ディスク使用量は？ 最も大きいディレクトリの上位 5 件を表示して。
```

エージェントがあなたの代わりにターミナルコマンドを実行し、結果を表示します。

### スラッシュコマンド

`/` を入力すると、すべてのコマンドのオートコンプリートドロップダウンが表示されます:

| コマンド | 動作 |
|---------|-------------|
| `/help` | 利用可能なすべてのコマンドを表示 |
| `/tools` | 利用可能なツールを一覧表示 |
| `/model` | モデルを対話的に切り替え |
| `/personality pirate` | 楽しい人格を試す |
| `/save` | 対話を保存 |

### 複数行入力

`Alt+Enter`、`Ctrl+J`、または `Shift+Enter` で改行を追加できます。`Shift+Enter` は、それを独立したシーケンスとして送信するターミナルが必要です（既定では Kitty / foot / WezTerm / Ghostty。iTerm2 / Alacritty / VS Code ターミナルは Kitty キーボードプロトコルを有効にすれば対応）。`Alt+Enter` と `Ctrl+J` はすべてのターミナルで動作します。

### エージェントを中断する

エージェントの処理が長すぎる場合は、新しいメッセージを入力して Enter を押してください。現在のタスクを中断して、新しい指示に切り替わります。`Ctrl+C` も使えます。

## 6. 次のレイヤーを追加する

基本のチャットが動いた後にだけ行ってください。必要なものを選びます:

### ボットまたは共有アシスタント

```bash
hermes gateway setup    # 対話的なプラットフォーム設定
```

[Telegram](/docs/user-guide/messaging/telegram)、[Discord](/docs/user-guide/messaging/discord)、[Slack](/docs/user-guide/messaging/slack)、[WhatsApp](/docs/user-guide/messaging/whatsapp)、[Signal](/docs/user-guide/messaging/signal)、[Email](/docs/user-guide/messaging/email)、[Home Assistant](/docs/user-guide/messaging/homeassistant)、[Microsoft Teams](/docs/user-guide/messaging/teams) を接続します。

### 自動化とツール

- `hermes tools` — プラットフォームごとにツールアクセスを調整
- `hermes skills` — 再利用可能なワークフローを閲覧・インストール
- Cron — ボットまたは CLI のセットアップが安定した後にだけ

### サンドボックス化されたターミナル

安全のため、エージェントを Docker コンテナまたはリモートサーバー上で実行します:

```bash
hermes config set terminal.backend docker    # Docker 分離
hermes config set terminal.backend ssh       # リモートサーバー
```

### ボイスモード

```bash
# Hermes のインストールディレクトリから（curl インストーラーは
# Linux/macOS では ~/.hermes/hermes-agent、Windows では %LOCALAPPDATA%\hermes\hermes-agent に配置します）:
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# 無料のローカル音声認識用に faster-whisper を含みます
```

その後 CLI で: `/voice on`。`Ctrl+B` を押して録音します。[ボイスモード](../user-guide/features/voice-mode.md)を参照してください。

### スキル

```bash
hermes skills search kubernetes
hermes skills install openai/skills/k8s
```

またはチャットセッション内で `/skills` を使います。

### MCP サーバー

```yaml
# ~/.hermes/config.yaml に追加
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### エディタ連携（ACP）

ACP サポートは標準の `[all]` extra に含まれているため、curl インストーラーにはすでに含まれています。次を実行するだけです:

```bash
hermes acp
```

（`[all]` なしでインストールした場合は、先に `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"` を実行してください。）

[ACP エディタ連携](../user-guide/features/acp.md)を参照してください。

---

## よくある失敗パターン

最も時間を浪費する問題は次のとおりです:

| 症状 | 考えられる原因 | 修正 |
|---|---|---|
| Hermes は開くが、空または壊れた応答を返す | プロバイダー認証またはモデル選択が誤っている | `hermes model` を再実行し、プロバイダー、モデル、認証を確認 |
| カスタムエンドポイントは「動く」が、おかしな出力を返す | ベース URL やモデル名が誤っている、または実際には OpenAI 互換でない | まず別のクライアントでエンドポイントを確認 |
| ゲートウェイは起動するが、誰もメッセージを送れない | ボットトークン、許可リスト、またはプラットフォーム設定が不完全 | `hermes gateway setup` を再実行し、`hermes gateway status` を確認 |
| `hermes --continue` が古いセッションを見つけられない | プロファイルを切り替えた、またはセッションが保存されなかった | `hermes sessions list` を確認し、正しいプロファイルにいることを確認 |
| モデルが利用不可、または奇妙なフォールバック動作 | プロバイダールーティングまたはフォールバック設定が積極的すぎる | ベースプロバイダーが安定するまでルーティングをオフのままにする |
| `hermes doctor` が設定の問題を指摘する | 設定値が欠落している、または古い | 設定を修正し、機能を追加する前にプレーンなチャットを再テスト |

## リカバリツールキット

何かおかしいと感じたら、この順番で使います:

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

このシーケンスで、「なんとなく壊れている」状態から既知の状態へ素早く戻せます。

---

## クイックリファレンス

| コマンド | 説明 |
|---------|-------------|
| `hermes` | チャットを開始 |
| `hermes model` | LLM プロバイダーとモデルを選択 |
| `hermes tools` | プラットフォームごとに有効なツールを設定 |
| `hermes setup` | フルセットアップウィザード（すべてを一度に設定） |
| `hermes doctor` | 問題を診断 |
| `hermes update` | 最新バージョンに更新 |
| `hermes gateway` | メッセージングゲートウェイを起動 |
| `hermes --continue` | 直前のセッションを再開 |

## 次のステップ

- **[CLI ガイド](../user-guide/cli.md)** — ターミナルインターフェースを使いこなす
- **[設定](../user-guide/configuration.md)** — セットアップをカスタマイズ
- **[メッセージングゲートウェイ](../user-guide/messaging/index.md)** — Telegram、Discord、Slack、WhatsApp、Signal、Email、Home Assistant、Teams などを接続
- **[ツールとツールセット](../user-guide/features/tools.md)** — 利用可能な機能を探る
- **[AI プロバイダー](../integrations/providers.md)** — 完全なプロバイダー一覧とセットアップの詳細
- **[スキルシステム](../user-guide/features/skills.md)** — 再利用可能なワークフローと知識
- **[ヒントとベストプラクティス](../guides/tips.md)** — パワーユーザー向けのヒント
