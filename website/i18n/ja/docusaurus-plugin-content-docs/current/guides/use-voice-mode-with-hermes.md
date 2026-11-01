---
sidebar_position: 8
title: "Hermesでボイスモードを使う"
description: "CLI、Telegram、Discord、DiscordのボイスチャンネルにわたってHermesのボイスモードをセットアップして使うための実践ガイド"
---

# Hermesでボイスモードを使う

このガイドは、[ボイスモード機能リファレンス](/docs/user-guide/features/voice-mode)の実践的な手引きです。

機能ページがボイスモードで何ができるかを説明するのに対し、このガイドは実際にそれをうまく使う方法を示します。

## ボイスモードが役立つ場面

ボイスモードは特に次のような場合に便利です。
- ハンズフリーのCLIワークフローが欲しい
- TelegramまたはDiscordで音声による応答が欲しい
- DiscordのボイスチャンネルにHermesを座らせてライブ会話をしたい
- 入力する代わりに、歩き回りながら素早いアイデアの記録、デバッグ、やり取りをしたい

## ボイスモードのセットアップを選ぶ

Hermesには実際には3つの異なる音声体験があります。

| モード | 最適な用途 | プラットフォーム |
|---|---|---|
| 対話型マイクループ | コーディングやリサーチ中の個人的なハンズフリー利用 | CLI |
| チャットでの音声返信 | 通常のメッセージングと並行した音声応答 | Telegram、Discord |
| ライブボイスチャンネルボット | VCでのグループまたは個人のライブ会話 | Discordのボイスチャンネル |

おすすめの進め方:
1. まずテキストを動作させる
2. 次に音声返信を有効にする
3. 完全な体験が欲しい場合は、最後にDiscordのボイスチャンネルに移行する

## ステップ1: まず通常のHermesが動作することを確認する

ボイスモードに触れる前に、次のことを確認してください。
- Hermesが起動する
- プロバイダーが設定されている
- エージェントが通常どおりテキストプロンプトに応答できる

```bash
hermes
```

何か簡単なことを尋ねます。

```text
What tools do you have available?
```

それがまだ安定していない場合は、まずテキストモードを修正してください。

## ステップ2: 適切な追加パッケージをインストールする

### CLIマイク + 再生

```bash
pip install "hermes-agent[voice]"
```

### メッセージングプラットフォーム

```bash
pip install "hermes-agent[messaging]"
```

### プレミアムElevenLabs TTS

```bash
pip install "hermes-agent[tts-premium]"
```

### ローカルNeuTTS（オプション）

```bash
python -m pip install -U neutts[all]
```

### すべて

```bash
pip install "hermes-agent[all]"
```

## ステップ3: システム依存関係をインストールする

### macOS

```bash
brew install portaudio ffmpeg opus
brew install espeak-ng
```

### Ubuntu / Debian

```bash
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng
```

これらが重要な理由:
- `portaudio` → CLIボイスモード用のマイク入力 / 再生
- `ffmpeg` → TTSとメッセージング配信のための音声変換
- `opus` → Discordのボイスコーデックのサポート
- `espeak-ng` → NeuTTS用のphonemizerバックエンド

## ステップ4: STTとTTSのプロバイダーを選ぶ

Hermesはローカルとクラウドの両方の音声スタックをサポートします。

### 最も簡単 / 最も安価なセットアップ

ローカルSTTと無料のEdge TTSを使用します。
- STTプロバイダー: `local`
- TTSプロバイダー: `edge`

これは通常、始めるのに最適な場所です。

### 環境ファイルの例

`~/.hermes/.env` に追加します。

```bash
# クラウドSTTのオプション（localはキー不要）
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# プレミアムTTS（オプション）
ELEVENLABS_API_KEY=***
```

### プロバイダーの推奨

#### 音声からテキスト（Speech-to-text）

- `local` → プライバシーとゼロコスト利用に最適なデフォルト
- `groq` → 非常に高速なクラウド文字起こし
- `openai` → 良質な有料フォールバック

#### テキストから音声（Text-to-speech）

- `edge` → 無料で、ほとんどのユーザーには十分な品質
- `neutts` → 無料のローカル/オンデバイスTTS
- `elevenlabs` → 最高品質
- `openai` → 良い中間的な選択肢
- `mistral` → 多言語、ネイティブのOpus

### `hermes setup` を使う場合

セットアップウィザードでNeuTTSを選択すると、Hermesは `neutts` がすでにインストールされているかどうかを確認します。見つからない場合、ウィザードはNeuTTSにPythonパッケージ `neutts` とシステムパッケージ `espeak-ng` が必要であることを伝え、それらをインストールするか提案し、プラットフォームのパッケージマネージャーで `espeak-ng` をインストールしてから、次を実行します。

```bash
python -m pip install -U neutts[all]
```

そのインストールをスキップした場合や失敗した場合、ウィザードはEdge TTSにフォールバックします。

## ステップ5: 推奨設定

```yaml
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0

stt:
  provider: "local"
  local:
    model: "base"

tts:
  provider: "edge"
  edge:
    voice: "en-US-AriaNeural"
```

これはほとんどの人にとって良い保守的なデフォルトです。

代わりにローカルTTSが欲しい場合は、`tts` ブロックを次のように切り替えます。

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## ユースケース1: CLIボイスモード

## 有効にする

Hermesを起動します。

```bash
hermes
```

CLI内で:

```text
/voice on
```

### 録音の流れ

デフォルトのキー:
- `Ctrl+B`

ワークフロー:
1. `Ctrl+B` を押す
2. 話す
3. 無音検出が自動的に録音を停止するのを待つ
4. Hermesが文字起こしして応答する
5. TTSがオンの場合、答えを読み上げる
6. 連続利用のためにループが自動的に再開できる

### 便利なコマンド

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### 良いCLIワークフロー

#### ウォークアップデバッグ

こう言います。

```text
I keep getting a docker permission error. Help me debug it.
```

その後、ハンズフリーで続けます。
- "Read the last error again"
- "Explain the root cause in simpler terms"
- "Now give me the exact fix"

#### リサーチ / ブレインストーミング

次のような場合に最適です。
- 考えながら歩き回る
- まとまっていないアイデアを口述する
- Hermesにリアルタイムで思考を構造化してもらう

#### アクセシビリティ / 入力を減らしたセッション

入力が不便な場合、ボイスモードは完全なHermesのループにとどまる最速の方法の1つです。

## CLIの動作を調整する

### 無音しきい値（silence threshold）

Hermesの開始/停止が過度に積極的な場合は、次を調整します。

```yaml
voice:
  silence_threshold: 250
```

しきい値が高い = 感度が低い。

### 無音の継続時間（silence duration）

文と文の間で頻繁に間を置く場合は、次を増やします。

```yaml
voice:
  silence_duration: 4.0
```

### 録音キー

`Ctrl+B` がターミナルやtmuxの習慣と競合する場合は:

```yaml
voice:
  record_key: "ctrl+space"
```

## ユースケース2: TelegramまたはDiscordでの音声返信

このモードは完全なボイスチャンネルよりもシンプルです。

Hermesは通常のチャットボットのままですが、返信を読み上げることができます。

### ゲートウェイを起動する

```bash
hermes gateway
```

### 音声返信を有効にする

TelegramまたはDiscord内で:

```text
/voice on
```

または

```text
/voice tts
```

### モード

| モード | 意味 |
|---|---|
| `off` | テキストのみ |
| `voice_only` | ユーザーが音声を送ったときのみ読み上げる |
| `all` | すべての返信を読み上げる |

### どのモードをいつ使うか

- 音声起源のメッセージに対してのみ音声返信が欲しい場合は `/voice on`
- 常に完全な音声アシスタントが欲しい場合は `/voice tts`

### 良いメッセージングワークフロー

#### スマートフォン上のTelegramアシスタント

次のような場合に使用します。
- マシンから離れている
- ボイスメモを送って素早い音声返信を得たい
- Hermesをポータブルなリサーチまたは運用アシスタントのように機能させたい

#### 音声出力付きのDiscord DM

サーバーチャンネルのメンション動作なしでプライベートなやり取りをしたい場合に便利です。

## ユースケース3: Discordのボイスチャンネル

これは最も高度なモードです。

HermesはDiscordのVCに参加し、ユーザーの発話を聞き、文字起こしし、通常のエージェントパイプラインを実行し、返信をチャンネルに読み上げます。

## 必要なDiscordの権限

通常のテキストボットのセットアップに加えて、ボットが次の権限を持っていることを確認してください。
- Connect
- Speak
- できれば Use Voice Activity

また、Developer Portalで特権インテントを有効にします。
- Presence Intent
- Server Members Intent
- Message Content Intent

## 参加と退出

ボットが存在するDiscordのテキストチャンネルで:

```text
/voice join
/voice leave
/voice status
```

### 参加したときに何が起こるか

- ユーザーがVCで話す
- Hermesが発話の境界を検出する
- トランスクリプトが関連するテキストチャンネルに投稿される
- Hermesがテキストと音声で応答する
- テキストチャンネルは `/voice join` が実行された場所のもの

### Discord VC利用のベストプラクティス

- `DISCORD_ALLOWED_USERS` を厳しく保つ
- 最初は専用のボット/テスト用チャンネルを使う
- VCモードを試す前に、通常のテキストチャットの音声モードでSTTとTTSが動作することを確認する

## 音声品質の推奨

### 最高品質のセットアップ

- STT: ローカルの `large-v3` またはGroqの `whisper-large-v3`
- TTS: ElevenLabs

### 最高の速度 / 利便性のセットアップ

- STT: ローカルの `base` またはGroq
- TTS: Edge

### 最高のゼロコストセットアップ

- STT: local
- TTS: Edge

## よくある失敗モード

### 「No audio device found」

`portaudio` をインストールします。

### 「ボットは参加するが何も聞こえない」

確認すること:
- あなたのDiscordユーザーIDが `DISCORD_ALLOWED_USERS` に含まれている
- ミュートされていない
- 特権インテントが有効になっている
- ボットにConnect/Speakの権限がある

### 「文字起こしはするが読み上げない」

確認すること:
- TTSプロバイダーの設定
- ElevenLabsまたはOpenAIのAPIキー / クォータ
- Edge変換パス用の `ffmpeg` のインストール

### 「Whisperがゴミを出力する」

試すこと:
- より静かな環境
- より高い `silence_threshold`
- 別のSTTプロバイダー/モデル
- より短く明瞭な発話

### 「DMでは動作するがサーバーチャンネルでは動作しない」

これはしばしばメンションポリシーが原因です。

デフォルトでは、特に設定されていない限り、ボットはDiscordサーバーのテキストチャンネルで `@mention` を必要とします。

## 最初の1週間のおすすめセットアップ

成功への最短経路が欲しい場合:

1. テキストのHermesを動作させる
2. `hermes-agent[voice]` をインストールする
3. ローカルSTT + Edge TTSでCLIボイスモードを使う
4. 次にTelegramまたはDiscordで `/voice on` を有効にする
5. その後でのみ、Discord VCモードを試す

この進め方なら、デバッグの範囲を小さく保てます。

## 次に読むもの

- [ボイスモード機能リファレンス](/docs/user-guide/features/voice-mode)
- [メッセージングゲートウェイ](/docs/user-guide/messaging)
- [Discordのセットアップ](/docs/user-guide/messaging/discord)
- [Telegramのセットアップ](/docs/user-guide/messaging/telegram)
- [設定](/docs/user-guide/configuration)
