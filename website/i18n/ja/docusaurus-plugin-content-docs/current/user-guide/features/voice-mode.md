---
sidebar_position: 10
title: "ボイスモード"
description: "Hermes Agent とのリアルタイム音声会話 — CLI、Telegram、Discord（DM、テキストチャンネル、ボイスチャンネル）"
---

# ボイスモード

Hermes Agent は、CLI とメッセージングプラットフォームの両方で完全な音声インタラクションをサポートします。マイクを使ってエージェントに話しかけ、音声による返答を聞き、Discord のボイスチャンネルでライブの音声会話を行えます。

推奨設定や実際の利用パターンを含む実践的なセットアップ手順については、[Hermes でボイスモードを使う](/docs/guides/use-voice-mode-with-hermes)を参照してください。

## 前提条件

音声機能を使う前に、以下を確認してください:

1. **Hermes Agent がインストール済みであること** — `pip install hermes-agent`（[インストール](/docs/getting-started/installation)を参照）
2. **LLM プロバイダーが設定済みであること** — `hermes model` を実行するか、`~/.hermes/.env` に希望するプロバイダーの認証情報を設定する
3. **動作するベースセットアップがあること** — `hermes` を実行し、音声を有効化する前にエージェントがテキストに応答することを確認する

:::tip
`~/.hermes/` ディレクトリとデフォルトの `config.yaml` は、初めて `hermes` を実行したときに自動的に作成されます。API キー用の `~/.hermes/.env` のみ手動で作成する必要があります。
:::

## 概要

| 機能 | プラットフォーム | 説明 |
|---------|----------|-------------|
| **インタラクティブボイス** | CLI | Ctrl+B を押して録音、エージェントが無音を自動検出して応答 |
| **自動ボイス返信** | Telegram、Discord | エージェントがテキスト応答とあわせて音声を送信 |
| **ボイスチャンネル** | Discord | ボットが VC に参加し、ユーザーの発話を聞き取り、返答を話す |

## 要件

### Python パッケージ

```bash
# CLI ボイスモード（マイク + 音声再生）
pip install "hermes-agent[voice]"

# Discord + Telegram メッセージング（VC サポート用の discord.py[voice] を含む）
pip install "hermes-agent[messaging]"

# プレミアム TTS（ElevenLabs）
pip install "hermes-agent[tts-premium]"

# ローカル TTS（NeuTTS、オプション）
python -m pip install -U neutts[all]

# すべてまとめて
pip install "hermes-agent[all]"
```

| エクストラ | パッケージ | 必要な用途 |
|-------|----------|-------------|
| `voice` | `sounddevice`, `numpy` | CLI ボイスモード |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Discord & Telegram ボット |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS プロバイダー |

オプションのローカル TTS プロバイダー: `neutts` は `python -m pip install -U neutts[all]` で個別にインストールします。初回使用時にモデルが自動的にダウンロードされます。

:::info
`discord.py[voice]` は **PyNaCl**（音声暗号化用）と **opus バインディング**を自動的にインストールします。これは Discord ボイスチャンネルのサポートに必要です。
:::

### システム依存関係

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # NeuTTS 用

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # NeuTTS 用
```

| 依存関係 | 目的 | 必要な用途 |
|-----------|---------|-------------|
| **PortAudio** | マイク入力と音声再生 | CLI ボイスモード |
| **ffmpeg** | 音声フォーマット変換（MP3 → Opus、PCM → WAV） | すべてのプラットフォーム |
| **Opus** | Discord 音声コーデック | Discord ボイスチャンネル |
| **espeak-ng** | Phonemizer バックエンド | ローカル NeuTTS プロバイダー |

### API キー

`~/.hermes/.env` に追加します:

```bash
# Speech-to-Text — ローカルプロバイダーはキーが一切不要
# pip install faster-whisper          # 無料、ローカル実行、推奨
GROQ_API_KEY=your-key                 # Groq Whisper — 高速、無料枠あり（クラウド）
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — 有料（クラウド）

# Text-to-Speech（オプション — Edge TTS と NeuTTS はキーなしで動作）
ELEVENLABS_API_KEY=***           # ElevenLabs — プレミアム品質
# 上記の VOICE_TOOLS_OPENAI_KEY は OpenAI TTS も有効化します
```

:::tip
`faster-whisper` がインストールされていれば、ボイスモードは STT に **API キーゼロ**で動作します。モデル（`base` で約 150 MB）は初回使用時に自動的にダウンロードされます。
:::

---

## CLI ボイスモード

ボイスモードは **クラシック CLI**（`hermes chat`）と **TUI**（`hermes --tui`）の両方で利用できます。動作はどちらも同一です — 同じスラッシュコマンド、同じ VAD 無音検出、同じストリーミング TTS、同じハルシネーションフィルター。TUI はさらにクラッシュフォレンジックログを `~/.hermes/logs/` に転送するため、特殊なオーディオバックエンドでのプッシュトゥトークの失敗を、静かに消えるのではなく完全なスタックトレースとともに報告できます。

### クイックスタート

CLI を起動してボイスモードを有効化します:

```bash
hermes                # インタラクティブ CLI を起動
```

次に、CLI 内で以下のコマンドを使います:

```
/voice          ボイスモードのオン/オフを切り替え
/voice on       ボイスモードを有効化
/voice off      ボイスモードを無効化
/voice tts      TTS 出力を切り替え
/voice status   現在の状態を表示
```

### 仕組み

1. `hermes` で CLI を起動し、`/voice on` でボイスモードを有効化する
2. **Ctrl+B を押す** — ビープ音（880Hz）が鳴り、録音が開始される
3. **話す** — ライブのオーディオレベルバーが入力を表示: `● [▁▂▃▅▇▇▅▂] ❯`
4. **話すのをやめる** — 3 秒間の無音の後、録音が自動的に停止する
5. **ビープ音が 2 回**鳴り（660Hz）、録音終了を確認する
6. 音声は Whisper で文字起こしされ、エージェントに送信される
7. TTS が有効な場合、エージェントの返答が音声で読み上げられる
8. 録音が **自動的に再開される** — キーを押さずにもう一度話せる

このループは、録音中に **Ctrl+B** を押す（連続モードを終了）か、3 回連続で発話が検出されないまで続きます。

:::tip
録音キーは `~/.hermes/config.yaml` の `voice.record_key` で設定できます（デフォルト: `ctrl+b`）。
:::

### 無音検出

2 段階のアルゴリズムで発話が終わったタイミングを検出します:

1. **発話確認** — RMS しきい値（200）を超えるオーディオが少なくとも 0.3 秒間続くのを待ちます。音節間の短い途切れは許容されます
2. **終了検出** — 発話が確認されると、3.0 秒間継続して無音になった時点でトリガーされます

15 秒間まったく発話が検出されない場合、録音は自動的に停止します。

`silence_threshold` と `silence_duration` はどちらも `config.yaml` で設定できます。また、`voice.beep_enabled: false` で録音開始/停止のビープ音を無効化できます。

### ストリーミング TTS

TTS が有効な場合、エージェントはテキストを生成しながら **文単位で** 返答を話します — 応答全体を待つ必要はありません:

1. テキストのデルタを完全な文にバッファリングします（最小 20 文字）
2. Markdown 書式と `<think>` ブロックを除去します
3. 文ごとに音声をリアルタイムで生成・再生します

### ハルシネーションフィルター

Whisper は無音や背景ノイズから幻のテキスト（"Thank you for watching"、"Subscribe" など）を生成することがあります。エージェントは、複数言語にわたる 26 種類の既知のハルシネーションフレーズのセットと、繰り返しのバリエーションを捕捉する正規表現パターンを使ってこれらをフィルタリングします。

---

## ゲートウェイのボイス返信（Telegram & Discord）

メッセージングボットをまだセットアップしていない場合は、プラットフォーム別のガイドを参照してください:
- [Telegram セットアップガイド](../messaging/telegram.md)
- [Discord セットアップガイド](../messaging/discord.md)

メッセージングプラットフォームに接続するためにゲートウェイを起動します:

```bash
hermes gateway        # ゲートウェイを起動（設定済みのプラットフォームに接続）
hermes gateway setup  # 初回設定用のインタラクティブセットアップウィザード
```

### Discord: チャンネル vs DM

ボットは Discord で 2 つのインタラクションモードをサポートします:

| モード | 話し方 | メンション必須 | セットアップ |
|------|------------|-----------------|-------|
| **ダイレクトメッセージ（DM）** | ボットのプロフィールを開く → 「メッセージ」 | 不要 | すぐに動作 |
| **サーバーチャンネル** | ボットがいるテキストチャンネルに入力 | 必要（`@botname`） | ボットをサーバーに招待する必要あり |

**DM（個人利用に推奨）:** ボットとの DM を開いて入力するだけ — @メンションは不要です。ボイス返信とすべてのコマンドはチャンネルと同様に動作します。

**サーバーチャンネル:** ボットは @メンションされたときのみ応答します（例: `@hermesbyt4 hello`）。メンションのポップアップでは、同名のロールではなく **ボットユーザー** を選択してください。

:::tip
サーバーチャンネルでメンション必須を無効化するには、`~/.hermes/.env` に追加します:
```bash
DISCORD_REQUIRE_MENTION=false
```
または、特定のチャンネルをフリーレスポンス（メンション不要）に設定します:
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### コマンド

これらは Telegram と Discord（DM とテキストチャンネル）の両方で動作します:

```
/voice          ボイスモードのオン/オフを切り替え
/voice on       ボイスメッセージを送信したときのみボイス返信
/voice tts      すべてのメッセージにボイス返信
/voice off      ボイス返信を無効化
/voice status   現在の設定を表示
```

### モード

| モード | コマンド | 動作 |
|------|---------|----------|
| `off` | `/voice off` | テキストのみ（デフォルト） |
| `voice_only` | `/voice on` | ボイスメッセージを送信したときのみ返答を話す |
| `all` | `/voice tts` | すべてのメッセージに返答を話す |

ボイスモードの設定はゲートウェイの再起動をまたいで保持されます。

### プラットフォーム別の配信

| プラットフォーム | フォーマット | 備考 |
|----------|--------|-------|
| **Telegram** | ボイスバブル（Opus/OGG） | チャット内でインライン再生。必要に応じて ffmpeg が MP3 → Opus に変換 |
| **Discord** | ネイティブボイスバブル（Opus/OGG） | ユーザーのボイスメッセージのようにインライン再生。ボイスバブル API が失敗した場合はファイル添付にフォールバック |

---

## Discord ボイスチャンネル

最も没入感のある音声機能: ボットが Discord のボイスチャンネルに参加し、ユーザーの発話を聞き取り、その発話を文字起こしし、エージェントで処理し、ボイスチャンネルで返答を話します。

### セットアップ

#### 1. Discord ボットの権限

テキスト用に Discord ボットをすでにセットアップしている場合（[Discord セットアップガイド](../messaging/discord.md)を参照）、音声権限を追加する必要があります。

[Discord Developer Portal](https://discord.com/developers/applications) → あなたのアプリケーション → **Installation** → **Default Install Settings** → **Guild Install** に移動します:

**既存のテキスト権限に以下の権限を追加します:**

| 権限 | 目的 | 必須 |
|-----------|---------|----------|
| **Connect** | ボイスチャンネルに参加 | はい |
| **Speak** | ボイスチャンネルで TTS 音声を再生 | はい |
| **Use Voice Activity** | ユーザーの発話を検出 | 推奨 |

**更新後の権限整数:**

| レベル | 整数 | 含まれる内容 |
|-------|---------|----------------|
| テキストのみ | `274878286912` | チャンネル表示、メッセージ送信、履歴読み取り、埋め込み、添付、スレッド、リアクション |
| テキスト + 音声 | `274881432640` | 上記すべて + Connect、Speak |

更新後の権限 URL で **ボットを再招待** します:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274881432640
```

`YOUR_APP_ID` を Developer Portal のアプリケーション ID に置き換えてください。

:::warning
ボットがすでに参加しているサーバーに再招待しても、ボットを削除せずに権限が更新されます。データや設定が失われることはありません。
:::

#### 2. 特権ゲートウェイインテント

[Developer Portal](https://discord.com/developers/applications) → あなたのアプリケーション → **Bot** → **Privileged Gateway Intents** で、3 つすべてを有効にします:

| インテント | 目的 |
|--------|---------|
| **Presence Intent** | ユーザーのオンライン/オフラインステータスを検出 |
| **Server Members Intent** | `DISCORD_ALLOWED_USERS` のユーザー名を数値 ID に解決（条件付き） |
| **Message Content Intent** | チャンネルのテキストメッセージ内容を読み取り |

**Message Content Intent** は必須です。**Server Members Intent** は `DISCORD_ALLOWED_USERS` リストでユーザー名を使う場合にのみ必要です — 数値ユーザー ID を使う場合はオフのままで構いません。ボイスチャンネルの SSRC → user_id マッピングは、音声 websocket 上の Discord の SPEAKING オペコードから得られるため、Server Members Intent は **不要** です。

#### 3. Opus コーデック

ゲートウェイを実行するマシンに Opus コーデックライブラリがインストールされている必要があります:

```bash
# macOS（Homebrew）
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

ボットは以下からコーデックを自動的にロードします:
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. 環境変数

```bash
# ~/.hermes/.env

# Discord ボット（テキスト用にすでに設定済み）
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — ローカルプロバイダーはキー不要（pip install faster-whisper）
# GROQ_API_KEY=your-key            # 代替: クラウドベース、高速、無料枠あり

# TTS — オプション。Edge TTS と NeuTTS はキー不要。
# ELEVENLABS_API_KEY=***      # プレミアム品質
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### ゲートウェイの起動

```bash
hermes gateway        # 既存の設定で起動
```

ボットは数秒以内に Discord でオンラインになるはずです。

### コマンド

ボットがいる Discord のテキストチャンネルで以下を使います:

```
/voice join      ボットが現在のボイスチャンネルに参加
/voice channel   /voice join のエイリアス
/voice leave     ボットがボイスチャンネルから切断
/voice status    ボイスモードと接続中のチャンネルを表示
```

:::info
`/voice join` を実行する前に、自分がボイスチャンネルに入っている必要があります。ボットはあなたがいるのと同じ VC に参加します。
:::

### 仕組み

ボットがボイスチャンネルに参加すると、次のことを行います:

1. 各ユーザーのオーディオストリームを個別に **聞き取る**
2. **無音を検出** — 少なくとも 0.5 秒の発話の後、1.5 秒の無音で処理がトリガーされる
3. Whisper STT（ローカル、Groq、または OpenAI）で音声を **文字起こし** する
4. エージェントパイプライン全体（セッション、ツール、メモリ）で **処理** する
5. TTS でボイスチャンネルに返答を **話す**

### テキストチャンネル連携

ボットがボイスチャンネルにいるとき:

- 文字起こしがテキストチャンネルに表示されます: `[Voice] @user: あなたが話した内容`
- エージェントの応答はチャンネルにテキストとして送信され、かつ VC で読み上げられます
- テキストチャンネルは `/voice join` が実行されたチャンネルです

### エコー防止

ボットは TTS 返答を再生している間、オーディオリスナーを自動的に一時停止し、自分自身の出力を聞き取って再処理することを防ぎます。

### アクセス制御

`DISCORD_ALLOWED_USERS` にリストされているユーザーのみが音声でやり取りできます。それ以外のユーザーの音声は静かに無視されます。

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## 設定リファレンス

### config.yaml

```yaml
# 音声録音（CLI）
voice:
  record_key: "ctrl+b"            # 録音の開始/停止キー
  max_recording_seconds: 120       # 最大録音時間
  auto_tts: false                  # ボイスモード開始時に TTS を自動有効化
  beep_enabled: true               # 録音開始/停止のビープ音を再生
  silence_threshold: 200           # これ未満を無音とみなす RMS レベル（0-32767）
  silence_duration: 3.0            # 自動停止までの無音秒数

# Speech-to-Text
stt:
  provider: "local"                  # "local"（無料） | "groq" | "openai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
  # model: "whisper-1"              # レガシー: provider が未設定のときに使用

# Text-to-Speech
tts:
  provider: "edge"                 # "edge"（無料） | "elevenlabs" | "openai" | "neutts" | "minimax"
  edge:
    voice: "en-US-AriaNeural"      # 322 種類の音声、74 言語
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # オプション: セルフホストや OpenAI 互換エンドポイント用に上書き
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### 環境変数

```bash
# Speech-to-Text プロバイダー（local はキー不要）
# pip install faster-whisper        # 無料のローカル STT — API キー不要
GROQ_API_KEY=...                    # Groq Whisper（高速、無料枠あり）
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper（有料）

# STT 上級者向け上書き（オプション）
STT_GROQ_MODEL=whisper-large-v3-turbo    # デフォルトの Groq STT モデルを上書き
STT_OPENAI_MODEL=whisper-1               # デフォルトの OpenAI STT モデルを上書き
GROQ_BASE_URL=https://api.groq.com/openai/v1     # カスタム Groq エンドポイント
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # カスタム OpenAI STT エンドポイント

# Text-to-Speech プロバイダー（Edge TTS と NeuTTS はキー不要）
ELEVENLABS_API_KEY=***             # ElevenLabs（プレミアム品質）
# 上記の VOICE_TOOLS_OPENAI_KEY は OpenAI TTS も有効化します

# Discord ボイスチャンネル
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### STT プロバイダー比較

| プロバイダー | モデル | 速度 | 品質 | コスト | API キー |
|----------|-------|-------|---------|------|---------|
| **Local** | `base` | 高速（CPU/GPU に依存） | 良好 | 無料 | 不要 |
| **Local** | `small` | 中程度 | より良い | 無料 | 不要 |
| **Local** | `large-v3` | 低速 | 最良 | 無料 | 不要 |
| **Groq** | `whisper-large-v3-turbo` | 非常に高速（約 0.5 秒） | 良好 | 無料枠 | 必要 |
| **Groq** | `whisper-large-v3` | 高速（約 1 秒） | より良い | 無料枠 | 必要 |
| **OpenAI** | `whisper-1` | 高速（約 1 秒） | 良好 | 有料 | 必要 |
| **OpenAI** | `gpt-4o-transcribe` | 中程度（約 2 秒） | 最良 | 有料 | 必要 |

プロバイダーの優先順位（自動フォールバック）: **local** > **groq** > **openai**

### TTS プロバイダー比較

| プロバイダー | 品質 | コスト | レイテンシ | キー必須 |
|----------|---------|------|---------|-------------|
| **Edge TTS** | 良好 | 無料 | 約 1 秒 | 不要 |
| **ElevenLabs** | 非常に良い | 有料 | 約 2 秒 | 必要 |
| **OpenAI TTS** | 良好 | 有料 | 約 1.5 秒 | 必要 |
| **NeuTTS** | 良好 | 無料 | CPU/GPU に依存 | 不要 |

NeuTTS は上記の `tts.neutts` 設定ブロックを使用します。

---

## トラブルシューティング

### 「No audio device found」（CLI）

PortAudio がインストールされていません:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

### Discord サーバーチャンネルでボットが応答しない

ボットはサーバーチャンネルでデフォルトで @メンションを必要とします。以下を確認してください:

1. `@` を入力し、同名の **ロール** ではなく **ボットユーザー**（#ディスクリミネーター付き）を選択する
2. または代わりに DM を使う — メンションは不要
3. または `~/.hermes/.env` で `DISCORD_REQUIRE_MENTION=false` を設定する

### ボットが VC に参加するが私の声を聞き取らない

- あなたの Discord ユーザー ID が `DISCORD_ALLOWED_USERS` に含まれているか確認する
- Discord でミュートになっていないか確認する
- ボットはオーディオをマッピングする前に Discord からの SPEAKING イベントを必要とします — 参加後数秒以内に話し始めてください

### ボットが私の声を聞き取るが応答しない

- STT が利用可能か確認する: `faster-whisper` をインストール（キー不要）するか、`GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY` を設定する
- LLM モデルが設定され、アクセス可能か確認する
- ゲートウェイログを確認する: `tail -f ~/.hermes/logs/gateway.log`

### ボットがテキストでは応答するがボイスチャンネルでは応答しない

- TTS プロバイダーが失敗している可能性があります — API キーとクォータを確認する
- Edge TTS（無料、キー不要）がデフォルトのフォールバックです
- TTS エラーがないかログを確認する

### Whisper がでたらめなテキストを返す

ハルシネーションフィルターがほとんどのケースを自動的に捕捉します。それでも幻の文字起こしが発生する場合:

- より静かな環境を使う
- 設定の `silence_threshold` を調整する（高いほど感度が下がる）
- 別の STT モデルを試す
