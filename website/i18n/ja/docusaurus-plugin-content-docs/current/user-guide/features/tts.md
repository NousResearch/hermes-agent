---
sidebar_position: 9
title: "音声＆TTS"
description: "すべてのプラットフォームにわたるテキスト読み上げと音声メッセージの文字起こし"
---

# 音声＆TTS

Hermes Agentは、すべてのメッセージングプラットフォームで、テキスト読み上げ（TTS）出力と音声メッセージの文字起こしの両方をサポートします。

:::tip Nous Subscribers
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、OpenAI TTSは別途のOpenAI APIキーなしで **[Tool Gateway](tool-gateway.md)** 経由で利用できます。`hermes model` または `hermes tools` を実行して有効にしてください。
:::

## テキスト読み上げ（Text-to-Speech） {#text-to-speech}

10種類のプロバイダーでテキストを音声に変換します:

| プロバイダー | 品質 | コスト | APIキー |
|----------|---------|------|---------|
| **Edge TTS**（デフォルト） | 良好 | 無料 | 不要 |
| **ElevenLabs** | 優秀 | 有料 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 良好 | 有料 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | 優秀 | 有料 | `MINIMAX_API_KEY` |
| **Mistral（Voxtral TTS）** | 優秀 | 有料 | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | 優秀 | 無料枠あり | `GEMINI_API_KEY` |
| **xAI TTS** | 優秀 | 有料 | `XAI_API_KEY` |
| **NeuTTS** | 良好 | 無料（ローカル） | 不要 |
| **KittenTTS** | 良好 | 無料（ローカル） | 不要 |
| **Piper** | 良好 | 無料（ローカル） | 不要 |

### プラットフォームでの配信

| プラットフォーム | 配信方法 | フォーマット |
|----------|----------|--------|
| Telegram | 音声バブル（インライン再生） | Opus `.ogg` |
| Discord | 音声バブル（Opus/OGG）。失敗時はファイル添付にフォールバック | Opus/MP3 |
| WhatsApp | 音声ファイルの添付 | MP3 |
| CLI | `~/.hermes/audio_cache/` に保存 | MP3 |

### 設定

```yaml
# ~/.hermes/config.yaml 内
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # グローバルな速度倍率（プロバイダー固有の設定が上書きする）
  edge:
    voice: "en-US-AriaNeural"   # 322音声、74言語
    speed: 1.0                  # レート百分率（+/-%）に変換される
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # OpenAI互換のTTSエンドポイント用にオーバーライド
    speed: 1.0                  # 0.25 - 4.0
  minimax:
    model: "speech-2.8-hd"     # speech-2.8-hd（デフォルト）、speech-2.8-turbo
    voice_id: "English_Graceful_Lady"  # https://platform.minimax.io/faq/system-voice-id を参照
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral（デフォルト）
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # または gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30種類のプリビルド音声: Zephyr, Puck, Kore, Enceladus, Gacrux など
  xai:
    voice_id: "eve"             # またはカスタム音声ID — 下記のドキュメントを参照
    language: "en"              # ISO 639-1コード
    sample_rate: 24000          # 22050 / 24000（デフォルト） / 44100 / 48000
    bit_rate: 128000            # MP3ビットレート。codec=mp3 のときのみ適用
    # base_url: "https://api.x.ai/v1"   # XAI_BASE_URL 環境変数でオーバーライド
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8。他に: kitten-tts-micro-0.8（41MB）、kitten-tts-mini-0.8（80MB）
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # 数字、通貨、単位を展開する
  piper:
    voice: en_US-lessac-medium                  # 音声名（自動ダウンロード）または .onnx への絶対パス
    # voices_dir: ''                            # デフォルト: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # onnxruntime-gpu が必要
    # length_scale: 1.0                         # 2.0 = 2倍遅く
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = 音量半分
    # normalize_audio: true
```

**速度制御**: グローバルな `tts.speed` の値は、デフォルトですべてのプロバイダーに適用されます。各プロバイダーは独自の `speed` 設定（例: `tts.openai.speed: 1.5`）で上書きできます。プロバイダー固有の速度がグローバルな値より優先されます。デフォルトは `1.0`（通常速度）です。


### 入力長の上限

各プロバイダーには、リクエストごとの入力文字数の上限が定められています。Hermesはプロバイダーを呼び出す前にテキストを切り詰めるため、リクエストが長さエラーで失敗することはありません:

| プロバイダー | デフォルト上限（文字数） |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 5000 |
| ElevenLabs | モデルに応じて変動（下記参照） |
| NeuTTS | 2000 |
| KittenTTS | 2000 |

**ElevenLabs** は、設定された `model_id` から上限を選びます:

| `model_id` | 上限（文字数） |
|------------|-------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2`（デフォルト）、`eleven_multilingual_v1`、`eleven_english_sts_v2`、`eleven_english_sts_v1` | 10000 |
| `eleven_v3`、`eleven_ttv_v3` | 5000 |
| 不明なモデル | プロバイダーのデフォルト（10000）にフォールバック |

**プロバイダーごとにオーバーライド** するには、TTS設定のプロバイダーセクションの下に `max_text_length:` を指定します:

```yaml
tts:
  openai:
    max_text_length: 8192   # プロバイダーの上限を引き上げる、または引き下げる
```

正の整数のみが有効です。ゼロ、負の値、非数値、ブール値はプロバイダーのデフォルトにフォールスルーするため、設定が壊れていても誤って切り詰めを無効にしてしまうことはありません。

### Telegramの音声バブルとffmpeg

Telegramの音声バブルにはOpus/OGGの音声フォーマットが必要です:

- **OpenAI、ElevenLabs、Mistral** はOpusをネイティブに生成 — 追加のセットアップ不要
- **Edge TTS**（デフォルト）はMP3を出力するため、変換に **ffmpeg** が必要:
- **MiniMax TTS** はMP3を出力するため、Telegramの音声バブル用に変換するには **ffmpeg** が必要
- **Google Gemini TTS** は生のPCMを出力し、**ffmpeg** を使ってTelegramの音声バブル用にOpusへ直接エンコード
- **xAI TTS** はMP3を出力するため、Telegramの音声バブル用に変換するには **ffmpeg** が必要
- **NeuTTS** はWAVを出力するため、こちらもTelegramの音声バブル用に変換するには **ffmpeg** が必要
- **KittenTTS** はWAVを出力するため、こちらもTelegramの音声バブル用に変換するには **ffmpeg** が必要
- **Piper** はWAVを出力するため、こちらもTelegramの音声バブル用に変換するには **ffmpeg** が必要

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

ffmpegがない場合、Edge TTS、MiniMax TTS、NeuTTS、KittenTTS、Piperの音声は通常の音声ファイルとして送信されます（再生は可能ですが、音声バブルではなく長方形のプレーヤーとして表示されます）。

:::tip
ffmpegをインストールせずに音声バブルを使いたい場合は、OpenAI、ElevenLabs、またはMistralのプロバイダーに切り替えてください。
:::

### xAIのカスタム音声（ボイスクローニング）

xAIは、自分の声をクローンしてTTSで使うことをサポートします。[xAI Console](https://console.x.ai/team/default/voice/voice-library) でカスタム音声を作成し、得られた `voice_id` を設定に指定します:

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # あなたのカスタム音声ID
```

録音、サポートされるフォーマット、制限の詳細については、[xAI Custom Voices ドキュメント](https://docs.x.ai/developers/model-capabilities/audio/custom-voices) を参照してください。

### Piper（ローカル、44言語）

Piperは、Open Home Foundation（Home Assistantのメンテナー）による高速なローカルニューラルTTSエンジンです。完全にCPU上で動作し、事前学習済み音声で **44言語** をサポートし、APIキーは不要です。

**`hermes tools` でインストール** → Voice & TTS → Piper — Hermesが `pip install piper-tts` を実行してくれます。あるいは手動でインストール: `pip install piper-tts`。

**Piperに切り替える:**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

ローカルにキャッシュされていない音声で初めてTTSを呼び出すと、Hermesは `python -m piper.download_voices <name>` を実行し、モデル（品質ティアに応じて約20～90MB）を `~/.hermes/cache/piper-voices/` にダウンロードします。それ以降の呼び出しはキャッシュされたモデルを再利用します。

**音声の選び方。** [完全な音声カタログ](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) には、英語、スペイン語、フランス語、ドイツ語、イタリア語、オランダ語、ポルトガル語、ロシア語、ポーランド語、トルコ語、中国語、アラビア語、ヒンディー語などが収録されており、それぞれに `x_low` / `low` / `medium` / `high` の品質ティアがあります。[rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/) で音声のサンプルを試せます。

**事前にダウンロードした音声を使う。** `tts.piper.voice` を、`.onnx` で終わる絶対パスに設定します:

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**高度なつまみ**（`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`、`use_cuda`）は、Piperの `SynthesisConfig` と1対1で対応します。古い `piper-tts` のバージョンでは無視されます。

### カスタムコマンドプロバイダー

使いたいTTSエンジンがネイティブにサポートされていない場合（VoxCPM、MLX-Kokoro、XTTS CLI、ボイスクローニングスクリプト、CLIを公開しているその他のもの）でも、Pythonを一切書かずに **コマンド型プロバイダー** として組み込めます。Hermesは入力テキストを一時的なUTF-8ファイルに書き込み、あなたのシェルコマンドを実行し、そのコマンドが生成した音声ファイルを読み込みます。

`tts.providers.<name>` の下に1つ以上のプロバイダーを宣言し、`tts.provider: <name>` で切り替えます — `edge` や `openai` といった組み込みプロバイダーを切り替えるのと同じ方法です。

```yaml
tts:
  provider: voxcpm                 # tts.providers の下の任意の名前を選ぶ
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # Telegramの音声バブルとして配信を試みる

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # ネイティブPiperも tts.piper.voice 経由でカスタム .onnx をサポート
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

#### 例: Doubao（中国語 seed-tts-2.0） {#example-doubao-chinese-seed-tts-20}

ByteDanceの [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) 双方向ストリーミングAPIによる高品質な中国語TTSには、[`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPIパッケージをインストールし、コマンドプロバイダーとして組み込みます:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
```

```yaml
tts:
  provider: doubao
  providers:
    doubao:
      type: command
      command: "doubao-speech say --text-file {input_path} --out {output_path}"
      output_format: mp3
      max_text_length: 1024
      timeout: 30
```

認証情報は、シェル環境（`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`）または `~/.doubao-speech/config.yaml` から取得されます。`--voice zh-female-warm`（または `doubao-speech list-voices` のその他のエイリアス）をコマンドに追加して音声を選びます。`doubao-speech` はストリーミングASRも同梱しています — Hermesとの統合については [下記のSTTセクション](#example-doubao--volcengine-asr) を参照してください。ソースと完全なドキュメント: [github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech)。

#### プレースホルダー

コマンドテンプレートでは、次のプレースホルダーを参照できます。Hermesはレンダリング時にそれらを置換し、各値を周囲のコンテキスト（裸／シングルクォート／ダブルクォート）に合わせてシェルクォートするため、スペースやその他のシェルにとって特別な文字を含むパスでも安全です。

| プレースホルダー      | 意味                                              |
|------------------|------------------------------------------------------|
| `{input_path}`   | Hermesが書き込んだ一時UTF-8テキストファイルのパス        |
| `{text_path}`    | `{input_path}` のエイリアス                             |
| `{output_path}`  | コマンドが音声を書き込む必要があるパス                 |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                       |
| `{voice}`        | `tts.providers.<name>.voice`。未設定なら空       |
| `{model}`        | `tts.providers.<name>.model`                         |
| `{speed}`        | 解決された速度倍率（プロバイダーまたはグローバル）       |

リテラルの中括弧には `{{` と `}}` を使います。

#### 任意のキー

| キー                | デフォルト | 意味                                                                                                    |
|--------------------|---------|------------------------------------------------------------------------------------------------------------|
| `timeout`          | `120`   | 秒。期限切れになるとプロセスツリーが終了される（Unixは `killpg`、Windowsは `taskkill /T`）。                       |
| `output_format`    | `mp3`   | `mp3` / `wav` / `ogg` / `flac` のいずれか。Hermesがパスを選ぶ場合は出力拡張子から自動推測される。      |
| `voice_compatible` | `false` | `true` のとき、HermesはMP3/WAV出力をffmpegでOpus/OGGに変換し、Telegramが音声バブルをレンダリングできるようにする。      |
| `max_text_length`  | `5000`  | コマンドをレンダリングする前に、入力がこの長さに切り詰められる。                                             |
| `voice` / `model`  | 空   | プレースホルダーの値としてのみコマンドに渡される。                                                           |

#### 挙動に関する注意

- **組み込みの名前が常に優先される。** `tts.providers.openai` のエントリがネイティブのOpenAIプロバイダーを覆い隠すことは決してないため、ユーザー設定が組み込みを密かに置き換えることはできません。
- **デフォルトの配信はドキュメント。** コマンドプロバイダーは、すべてのプラットフォームで通常の音声添付として配信します。プロバイダーごとに `voice_compatible: true` で音声バブル配信をオプトインします。
- **コマンドの失敗はエージェントに表出する。** ゼロ以外の終了、空の出力、タイムアウトはすべて、コマンドのstderr/stdoutを含むエラーを返すため、会話の中からプロバイダーをデバッグできます。
- **`command:` が設定されている場合、`type: command` がデフォルト。** `type: command` を明示的に書くのは良い習慣ですが必須ではありません。空でない `command` 文字列を持つエントリはコマンドプロバイダーとして扱われます。
- **`{input_path}` / `{text_path}` は互換。** コマンドの中で読みやすいほうを使ってください。

#### セキュリティ

コマンド型プロバイダーは、あなたが設定した任意のシェルコマンドを、あなたのユーザーの権限で実行します。Hermesはプレースホルダーの値をクォートし、設定されたタイムアウトを強制しますが、コマンドテンプレート自体は信頼されたローカル入力です — PATH上のシェルスクリプトと同じように扱ってください。

## 音声メッセージの文字起こし（STT）

Telegram、Discord、WhatsApp、Slack、Signalで送信された音声メッセージは、自動的に文字起こしされ、テキストとして会話に注入されます。エージェントはその文字起こしを通常のテキストとして認識します。

| プロバイダー | 品質 | コスト | APIキー |
|----------|---------|------|---------| 
| **Local Whisper**（デフォルト） | 良好 | 無料 | 不要 |
| **Groq Whisper API** | 良好～最高 | 無料枠あり | `GROQ_API_KEY` |
| **OpenAI Whisper API** | 良好～最高 | 有料 | `VOICE_TOOLS_OPENAI_KEY` または `OPENAI_API_KEY` |

:::info Zero Config
ローカルの文字起こしは、`faster-whisper` がインストールされていれば、追加設定なしで動作します。それが利用できない場合、Hermesは一般的なインストール場所（`/opt/homebrew/bin` など）にあるローカルの `whisper` CLIや、`HERMES_LOCAL_STT_COMMAND` 経由のカスタムコマンドも使えます。
:::

### 設定

```yaml
# ~/.hermes/config.yaml 内
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"             # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-stt"         # xAI Grok STT
```

### プロバイダーの詳細

**Local（faster-whisper）** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) を介してローカルでWhisperを実行します。デフォルトではCPUを使い、利用可能な場合はGPUを使います。モデルサイズ:

| モデル | サイズ | 速度 | 品質 |
|-------|------|-------|---------|
| `tiny` | 約75 MB | 最速 | 基本的 |
| `base` | 約150 MB | 高速 | 良好（デフォルト） |
| `small` | 約500 MB | 中速 | より良い |
| `medium` | 約1.5 GB | やや遅い | 優秀 |
| `large-v3` | 約3 GB | 最遅 | 最高 |

**Groq API** — `GROQ_API_KEY` が必要です。無料のホスト型STTオプションが欲しいときの、良いクラウドフォールバックです。

**OpenAI API** — まず `VOICE_TOOLS_OPENAI_KEY` を受け付け、`OPENAI_API_KEY` にフォールバックします。`whisper-1`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe` をサポートします。

**Mistral API（Voxtral Transcribe）** — `MISTRAL_API_KEY` が必要です。Mistralの [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) モデルを使います。13言語、話者ダイアライゼーション、単語レベルのタイムスタンプをサポートします。`pip install hermes-agent[mistral]` でインストールします。

**xAI Grok STT** — `XAI_API_KEY` が必要です。`https://api.x.ai/v1/stt` にmultipart/form-dataとしてPOSTします。チャットやTTSですでにxAIを使っていて、すべてを1つのAPIキーで済ませたい場合に良い選択肢です。自動検出の順序ではGroqの後に置かれます — 強制するには明示的に `stt.provider: xai` を設定してください。

**カスタムローカルCLIフォールバック** — Hermesにローカルの文字起こしコマンドを直接呼び出させたい場合は、`HERMES_LOCAL_STT_COMMAND` を設定します。コマンドテンプレートは `{input_path}`、`{output_dir}`、`{language}`、`{model}` のプレースホルダーをサポートします。コマンドは、`{output_dir}` の下のどこかに `.txt` の文字起こしを書き込む必要があります。

#### 例: Doubao / Volcengine ASR {#example-doubao--volcengine-asr}

Doubao TTSに [`doubao-speech`](https://pypi.org/project/doubao-speech/) を使っている場合（[上記](#example-doubao-chinese-seed-tts-20) を参照）、同じパッケージがローカルコマンドのSTT表面を介して音声からテキストへの変換も処理します:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

```yaml
stt:
  provider: local_command
```

Hermesは受信した音声メッセージを `{input_path}` に書き込み、コマンドを実行し、`{output_dir}` の下に生成された `.txt` ファイルを読み込みます。言語はVolcengineのbigmodelエンドポイントによって自動検出されます。

### フォールバックの挙動

設定したプロバイダーが利用できない場合、Hermesは自動的にフォールバックします:
- **ローカルのfaster-whisperが利用不可** → クラウドプロバイダーの前に、ローカルの `whisper` CLIまたは `HERMES_LOCAL_STT_COMMAND` を試す
- **Groqキーが未設定** → ローカルの文字起こし、次にOpenAIにフォールバック
- **OpenAIキーが未設定** → ローカルの文字起こし、次にGroqにフォールバック
- **Mistralのキー／SDKが未設定** → 自動検出でスキップ。次に利用可能なプロバイダーにフォールスルー
- **何も利用できない** → 音声メッセージは、ユーザーへの正確な注記とともにそのまま通過する
