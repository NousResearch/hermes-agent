---
sidebar_position: 9
title: "OllamaでHermesをローカル実行 — API費用ゼロ"
description: "Ollamaと、Gemma 4のようなオープンウェイトモデルを使って、Hermes Agentを完全に自分のマシンで動かすステップバイステップガイド。クラウドのAPIキーや有料サブスクリプションは不要"
---

# OllamaでHermesをローカル実行 — API費用ゼロ

## 課題

クラウドのLLM APIはトークンごとに課金されます。重いコーディングセッションでは5〜20ドルかかることもあります。個人プロジェクト、学習、プライバシーに配慮した作業では、これが積み重なります — しかもすべての会話をサードパーティに送信することになります。

## このガイドで解決すること

このガイドでは、[Ollama](https://ollama.com)をモデルバックエンドとして使い、Hermes Agentを完全に自分のハードウェア上で動かす方法を解説します。APIキーもサブスクリプションも不要で、データがマシンの外に出ることもありません。一度設定すれば、HermesはOpenRouterやAnthropicを使うときと全く同じように動作します — ターミナルコマンド、ファイル編集、Webブラウジング、委譲（delegation） — ただしモデルはローカルで動きます。

最後には、次のものが手に入ります：

- 1つ以上のオープンウェイトモデルを提供するOllama
- カスタムエンドポイントとしてOllamaに接続されたHermes
- ファイルを編集し、コマンドを実行し、Webを閲覧できる動作するローカルエージェント
- オプション: 完全に自分のハードウェアで動くTelegram/Discordボット

## 必要なもの

| コンポーネント | 最低限 | 推奨 |
|-----------|---------|-------------|
| **RAM** | 8 GB（3Bモデル向け） | 32 GB以上（27B以上のモデル向け） |
| **ストレージ** | 空き5 GB | 30 GB以上（複数モデル向け） |
| **CPU** | 4コア | 8コア以上（AMD EPYC、Ryzen、Intel Xeon） |
| **GPU** | 不要 | VRAM 8 GB以上のNVIDIA GPUで大幅に高速化 |

:::tip CPUのみでも動作しますが、応答は遅くなります
OllamaはCPUのみのサーバーでも動作します。最新の8コアCPUで9Bモデルなら約10トークン/秒です。CPUで31Bモデルはさらに遅く（約2〜5トークン/秒）、各応答に30〜120秒かかりますが、動作はします。GPUがあればこれが劇的に改善します。CPUのみの構成では、環境変数でAPIタイムアウトを広げてください（これは `config.yaml` のキーではありません）：

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30分 — 遅いローカルモデルに対して余裕を持たせる
```
:::

## ステップ1: Ollamaをインストールする

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

実行中であることを確認します：

```bash
ollama --version
curl http://localhost:11434/api/tags   # {"models":[]} を返すはず
```

## ステップ2: モデルをプルする

ハードウェアに応じて選択します：

| モデル | ディスク上のサイズ | 必要なRAM | ツール呼び出し | 最適な用途 |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | 約20 GB | 24 GB以上 | あり | 最高品質 — 強力なツール利用と推論 |
| `gemma2:27b` | 約16 GB | 20 GB以上 | なし | 会話タスク、ツール利用なし |
| `gemma2:9b` | 約5 GB | 8 GB以上 | なし | 高速なチャット、Q&A — ツール呼び出し不可 |
| `llama3.2:3b` | 約2 GB | 4 GB以上 | なし | 軽量な簡易回答のみ |

:::warning ツール呼び出しが重要
Hermesは**エージェント型**のアシスタントです — ツール呼び出しを通じてファイルを編集し、コマンドを実行し、Webを閲覧します。ツール呼び出しに対応していないモデルはチャットしかできず、アクションを実行できません。Hermesの完全な体験を得るには、ツールに対応したモデル（`gemma4:31b` など）を使用してください。
:::

選んだモデルをプルします：

```bash
ollama pull gemma4:31b
```

:::info 複数のモデル
複数のモデルをプルして、Hermes内で `/model` を使って切り替えられます。Ollamaはアクティブなモデルを必要に応じてメモリに読み込み、アイドル状態のものは自動的にアンロードします。
:::

モデルが動作することを確認します：

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

モデルの応答を含むJSONレスポンスが表示されるはずです。

## ステップ3: Hermesを設定する

Hermesのセットアップウィザードを実行します：

```bash
hermes setup
```

プロバイダーを尋ねられたら、**Custom Endpoint** を選択し、次を入力します：

- **Base URL:** `http://localhost:11434/v1`
- **API Key:** 空のままにするか `no-key` と入力（Ollamaは不要）
- **Model:** `gemma4:31b`（またはプルしたモデル）

あるいは、`~/.hermes/config.yaml` を直接編集します：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## ステップ4: Hermesを使い始める

```bash
hermes
```

これだけです。これで完全にローカルなエージェントが動いています。試してみましょう：

```
You: List all Python files in this directory and count the lines of code in each

You: Read the README.md and summarize what this project does

You: Create a Python script that fetches the weather for Ho Chi Minh City
```

Hermesはターミナルツール、ファイル操作、そしてローカルモデルを使用します — クラウド呼び出しはありません。

## ステップ5: タスクに合った適切なモデルを選ぶ

すべてのタスクに最大のモデルが必要なわけではありません。実用的なガイドを示します：

| タスク | 推奨モデル | 理由 |
|------|-------------------|-----|
| ファイル編集、コード、ターミナルコマンド | `gemma4:31b` | 信頼できるツール呼び出しが可能な唯一のモデル |
| 簡単なQ&A（ツール利用不要） | `gemma2:9b` | 会話タスク向けの高速な応答 |
| 軽量なチャット | `llama3.2:3b` | 最速だが能力は非常に限られる |

:::note
完全なエージェント作業（ファイル編集、コマンド実行、ブラウジング）には、現在 `gemma4:31b` がツール呼び出しに対応した最良のローカルオプションです。より新しいモデルについては[Ollamaのモデルライブラリ](https://ollama.com/library)を確認してください — ツール呼び出しのサポートは急速に拡大しています。
:::

セッション内でモデルをその場で切り替えます：

```
/model gemma2:9b
```

## ステップ6: 速度を最適化する {#step-6-optimize-for-speed}

### Ollamaのコンテキストウィンドウを拡大する

デフォルトでは、Ollamaは2048トークンのコンテキストを使用します。エージェント作業（ツール呼び出し、長い会話）には、より多くが必要です：

```bash
# コンテキストを拡張するModelfileを作成
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 16384
EOF

ollama create gemma4-16k -f /tmp/Modelfile
```

その後、Hermesの設定を更新して `gemma4-16k` をモデル名として使用します。

### モデルを読み込んだままにする

デフォルトでは、Ollamaは5分間アイドル状態が続くとモデルをアンロードします。常駐するゲートウェイボットの場合は、読み込んだままにします：

```bash
# keep-aliveを24時間に設定
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

または、Ollamaの環境でグローバルに設定します：

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### GPUオフロードを使う（利用可能な場合）

NVIDIA GPUがある場合、Ollamaは自動的にレイヤーをGPUにオフロードします。次で確認します：

```bash
ollama ps   # 読み込まれているモデルとGPUレイヤー数を表示
```

12 GBのGPUで31Bモデルを動かす場合、部分的なオフロード（約40レイヤーがGPU、残りがCPU）になりますが、それでも大幅に高速化します。

## ステップ7: ゲートウェイボットとして実行する（オプション）

HermesがCLIでローカルに動作するようになったら、Telegramボットやdiscordボットとして公開できます — それでも完全に自分のハードウェアで動作します。

### Telegram

1. [@BotFather](https://t.me/BotFather)経由でボットを作成し、トークンを取得します
2. `~/.hermes/config.yaml` に追加します：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

platforms:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
```

3. ゲートウェイを起動します：

```bash
hermes gateway
```

これでTelegramでボットにメッセージを送ると、ローカルモデルを使って応答します。

### Discord

1. [discord.com/developers](https://discord.com/developers/applications)でDiscordアプリケーションを作成します
2. 設定に追加します：

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. 起動します： `hermes gateway`

## ステップ8: フォールバックを設定する（オプション）

ローカルモデルは複雑なタスクに苦労することがあります。ローカルモデルが失敗したときだけ起動するクラウドフォールバックを設定します：

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

これにより、利用の90%は無料（ローカル）で、難しいタスクだけが有料APIを使います。

## トラブルシューティング

### 起動時に「Connection refused」

Ollamaが実行されていません。起動します：

```bash
sudo systemctl start ollama
# または
ollama serve
```

### 応答が遅い

- **モデルサイズとRAMを確認:** モデルが利用可能なRAMより多くを必要とする場合、ディスクにスワップします。より小さいモデルを使うかRAMを追加してください。
- **`ollama ps` を確認:** GPUレイヤーがオフロードされていない場合、応答はCPUバウンドになります。CPUのみのサーバーでは正常です。
- **コンテキストを減らす:** 大きな会話は推論を遅くします。定期的に `/compress` を使うか、設定でより低い圧縮しきい値を設定してください。

### モデルがツール呼び出しに従わない

小さいモデル（3B、7B）は、ツール呼び出しの指示を無視して、構造化された関数呼び出しの代わりにプレーンテキストを生成することがあります。解決策：

- **より大きいモデルを使う** — `gemma4:31b` や `gemma2:27b` は3B/7Bモデルよりはるかにうまくツール呼び出しを処理します。
- **Hermesには自動修復がある** — 不正なツール呼び出しを検出し、自動的に修正を試みます。
- **フォールバックを設定する** — ローカルモデルが3回失敗すると、Hermesはクラウドプロバイダーにフォールバックします。

### コンテキストウィンドウのエラー

デフォルトのOllamaコンテキスト（2048トークン）は、エージェント作業には小さすぎます。拡大する方法については[ステップ6](#step-6-optimize-for-speed)を参照してください。

## コスト比較

典型的なコーディングセッション（入力約100Kトークン、出力約20Kトークン）に基づいて、ローカル実行がクラウドAPIと比べてどれだけ節約できるかを示します：

| プロバイダー | セッションあたりのコスト | 月額（毎日利用） |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | 約$0.80 | 約$24 |
| OpenRouter (GPT-4o) | 約$0.60 | 約$18 |
| **Ollama（ローカル）** | **$0.00** | **$0.00** |

唯一のコストは電気代です — ハードウェアによってセッションあたりおよそ$0.01〜0.05です。

## ローカルでうまく動くもの

- **ファイル編集とコード生成** — 9B以上のモデルでうまく処理できます
- **ターミナルコマンド** — Hermesはコマンドをラップして実行し、モデルに関わらず出力を読み取ります
- **Webブラウジング** — ブラウザツールが取得を行い、モデルは結果を解釈するだけです
- **cronジョブとスケジュールタスク** — クラウド構成と同じように動作します
- **マルチプラットフォームゲートウェイ** — Telegram、Discord、Slackすべてがローカルモデルで動作します

## クラウドモデルの方が優れているもの

- **非常に複雑な多段階の推論** — 70B以上やClaude Opusのようなクラウドモデルは明らかに優れています
- **長いコンテキストウィンドウ** — クラウドモデルは100K〜1Mトークンを提供します。ローカルモデルは通常8K〜32Kです
- **大きな応答での速度** — 長い生成では、クラウド推論はCPUのみのローカルより高速です

最適なバランス: 日常的なタスクにはローカルを使い、難しいものにはクラウドフォールバックを設定する。
