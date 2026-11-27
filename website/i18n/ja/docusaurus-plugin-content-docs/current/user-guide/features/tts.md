---
sidebar_position: 9
title: "音声＆TTS"
description: "すべてのプラットフォームにわたるテキスト読み上げと音声メッセージの文字起こし"
---

# 音声＆TTS

このページのPython依存関係コマンドは、
[PMで準備したソースチェックアウト](/docs/reference/package-management#developer-workflow) を使います。
依存関係を変更した後は、チェックアウトを再アクティベートしてHermesを再起動してください。

Hermes Agentは、すべてのメッセージングプラットフォームで、テキスト読み上げ（TTS）出力と音声メッセージの文字起こしの両方をサポートします。

:::tip Nous Subscribers
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、OpenAI TTSは別途のOpenAI APIキーなしで **[Tool Gateway](tool-gateway.md)** 経由で利用できます。新規インストールでは `hermes setup --portal` を実行すると、ログインしてすべてのゲートウェイツールを一度に有効にできます。既存のインストールでは、`hermes model` または `hermes tools` で **Nous Subscription** を選ぶと、TTSだけを有効にできます。
:::

## テキスト読み上げ（Text-to-Speech） {#text-to-speech}

11種類のプロバイダーでテキストを音声に変換します:

| プロバイダー | 品質 | コスト | APIキー |
|----------|---------|------|---------|
| **Edge TTS**（デフォルト） | 良好 | 無料 | 不要 |
| **ElevenLabs** | 優秀 | 有料 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 良好 | 有料 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | 優秀 | 有料 | `MINIMAX_API_KEY` または `MINIMAX_CN_API_KEY` |
| **Mistral（Voxtral TTS）** | 優秀 | 有料 | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | 優秀 | 無料枠あり | `GEMINI_API_KEY` |
| **xAI TTS** | 優秀 | 有料 | `XAI_API_KEY` |
| **DeepInfra TTS** | 良好 | 有料 | `DEEPINFRA_API_KEY` |
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
# In ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "deepinfra" | "neutts" | "kittentts" | "piper" — or "nous" for the managed Tool Gateway (written when you pick Nous Subscription in `hermes tools`)
  speed: 1.0                    # Global speed multiplier (provider-specific settings override this)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Converted to rate percentage (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
    pcm_sample_rate: 24000      # Streaming PCM rate expected from the endpoint (auto-overridden by X-Audio-Sample-Rate)
    speed: 1.0                  # 0.25 - 4.0
    # language: "es"            # Sent as lang_code — only for OpenAI-compatible endpoints that support it (e.g. Kokoro)
    # consent_attestation: "I have the speaker's consent"  # Required by some OpenAI-compatible servers for cloned voices
  streaming:
    min_len: 20                 # Streaming TTS: shortest first sentence (chars) spoken on its own; CJK setups use ~6
  minimax:
    region: "global"           # "global" or "cn"; see selection rules below
    model: "speech-02-hd"     # speech-02-hd (default), speech-02-turbo
    voice_id: "English_expressive_narrator"  # See https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
    # base_url: "https://tts.example/v1/t2a_v2"  # Optional endpoint override for the selected region
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, Gacrux, etc.
    audio_tags: false           # Enable hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # or a custom voice ID — see docs below
    language: "en"              # BCP-47 code (e.g. "en", "pt-BR") or "auto" for detection
    speed: 1.0                  # 0.7–1.5, playback speed (default: 1.0)
    auto_speech_tags: false     # insert expressive audio tags via LLM rewrite
    text_normalization: false   # normalize numbers/abbreviations/symbols to spoken form
    optimize_streaming_latency: 0  # 0–2, trades quality for lower latency (default: 0)
    sample_rate: 24000          # 22050 / 24000 (default) / 44100 / 48000
    bit_rate: 128000            # MP3 bitrate; only applies when codec=mp3
    # base_url: "https://api.x.ai/v1"   # Override via XAI_BASE_URL env var
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8; also: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # Expand numbers, currencies, units
  piper:
    voice: en_US-lessac-medium                  # voice name (auto-downloaded) OR absolute path to .onnx
    # voices_dir: ''                            # default: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # requires onnxruntime-gpu
    # length_scale: 1.0                         # 2.0 = twice as slow
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = half as loud
    # normalize_audio: true
```

KittenTTSはIntel macOSおよびWindows ARM64では利用できません。依存関係がこれらのプラットフォーム向けの `onnxruntime` やPyTorchのwheelを公開していないためです。これらのプラットフォームで選択すると、プロバイダーは利用不可と報告されます。

MiniMax TTSは、リージョン、エンドポイント、認証情報をまとめて選択します:

- `region: "global"` は `https://api.minimax.io/v1/t2a_v2` を `MINIMAX_API_KEY` とともに使います。
- `region: "cn"` は `https://api.minimaxi.com/v1/t2a_v2` を `MINIMAX_CN_API_KEY` とともに使います。
- `region` を省略した場合、後方互換性のため `MINIMAX_API_KEY` が優先されます。`MINIMAX_CN_API_KEY` だけが設定されている場合、Hermesは `cn` を選択します。
- 明示的に選択したリージョンには、対応する認証情報が必要です。Hermesがもう一方のリージョンのキーを借用することはありません。`base_url` のオーバーライドは選択された認証情報を変更せず、もう一方のリージョンの公式エンドポイントを指すオーバーライドは拒否されます。

**速度制御**: グローバルな `tts.speed` の値は、デフォルトですべてのプロバイダーに適用されます。各プロバイダーは独自の `speed` 設定（例: `tts.openai.speed: 1.5`）で上書きできます。プロバイダー固有の速度がグローバルな値より優先されます。デフォルトは `1.0`（通常速度）です。

### Geminiのペルソナプロンプト

Gemini TTSは、自然言語による演技の指示に従うことができます。`tts.gemini.persona_prompt_file` に、音声のペルソナを記述したローカルのMarkdownまたはテキストファイルを設定します。ファイルには、`AUDIO PROFILE`、`SCENE`、`DIRECTOR'S NOTES`、`SAMPLE CONTEXT`、`TRANSCRIPT` といったGemini形式のセクションを含めることができます。

ファイルに `{transcript}` または `{{ transcript }}` が含まれている場合、Hermesはそのプレースホルダーを実際のTTSテキストで置き換えます。含まれていない場合は、ラベル付きの `TRANSCRIPT` セクションを自動的に末尾に追加します。ペルソナプロンプトはローカルにとどまり、チャットの返信には表示されません。

```yaml
tts:
  provider: gemini
  gemini:
    voice: Algieba
    persona_prompt_file: ~/.hermes/tts/butler-voice.md
```

### オーディオタグ（Gemini、xAI）

GoogleのGemini 3.1 Flash TTSとxAIのGrok TTSは、`[whispers]`、`[excitedly]`、`[very slow]`、`[laughs]` など、自由形式の角括弧によるオーディオタグや、その他の表現豊かな話し方の指示をサポートします。`tts.gemini.audio_tags` または `tts.xai.auto_speech_tags` を有効にすると、HermesはTTSの前に隠れた書き換えパスを実行します。この書き換えはTTSスクリプトにのみインラインタグを挿入し、表示されるチャットの返信は変わりません。

```yaml
tts:
  provider: gemini
  gemini:
    model: gemini-3.1-flash-tts-preview
    audio_tags: true
  xai: 
    auto_speech_tags: true
```

書き換えには `auxiliary.tts_audio_tags` が使われ、デフォルトではメインのチャットモデルが使われます。タグ挿入をより安価または高速なモデルで処理したい場合は、この補助タスクをオーバーライドしてください。

**ストリーミングのサンプルレート（OpenAI互換エンドポイント）**: ストリーミング再生はヘッダーのない生のPCMを受信するため、Hermesはそのサンプルレートを知る必要があります。公式のOpenAI APIは24 kHzを出力します。レートを報告する互換サーバー（`X-Audio-Sample-Rate` レスポンスヘッダー、または `Content-Type` 内の `rate=`（`audio/pcm; rate=44100`））は自動的に尊重されます。レスポンスが届くと、スピーカー、一時WAVプレーヤー、ゲートウェイの音声ストリームはすべて報告されたレートで開かれます。何も報告しないサーバーの場合は、`tts.openai.pcm_sample_rate` をエンドポイントの出力レート（例: Piperベースのサーバーなら `22050`）に設定してください。そうしないと、音声が誤った速度とピッチで再生されます。無効な値は警告をログに記録し、`24000` にフォールバックします。

**言語（OpenAI互換エンドポイント）**: `tts.openai.language` は、`lang_code` リクエストパラメーターとしてエンドポイントに転送されます。これは `lang_code` をサポートするOpenAI互換TTSサーバー向けです。たとえば [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) では、`language: "es"` を指定すると英語のデフォルトの代わりにスペイン語のフォネマイザーが選択されます。このパラメーターを受け付けない公式のOpenAI APIを使う場合は未設定のままにしてください。未設定の場合、追加のものは何も送信されません。

**クローン音声の同意（OpenAI互換エンドポイント）**: 一部のセルフホスト型OpenAI互換TTSサーバーは、リクエストに `consent_attestation` フィールドが含まれていない限り、クローン音声を `400 consent_required` で拒否します。`tts.openai.consent_attestation` に、サーバーが期待する同意の証明テキストを設定してください。Hermesは、すべてのOpenAI互換のパス（ファイル全体の合成、ストリーミング、デスクトップのクライアント直接音声）で、それをリクエストボディにそのまま転送します。公式のOpenAI APIでは未設定のままにしてください。未設定の場合、このフィールドは送信されません。


### 入力長の上限

各プロバイダーには、リクエストごとの入力文字数の上限が定められています。Hermesはプロバイダーを呼び出す前に、長い返信を文を考慮した順序付きのチャンクに分割するため、正規化されたテキスト全体が黙って切り詰められることなく保持されます:

| プロバイダー | デフォルト上限（文字数） |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 32000 |
| ElevenLabs | モデルに応じて変動（下記参照） |
| NeuTTS | 2000 |
| KittenTTS | 2000 |
| Piper | 5000 |

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
    max_text_length: 8192   # raise or lower the provider cap
```

正の整数のみが有効です。ゼロ、負の値、非数値、ブール値はプロバイダーのデフォルトにフォールスルーするため、設定が壊れていても誤ってプロバイダーのリクエスト上限を回避してしまうことはありません。

### Telegramの音声バブルとffmpeg

Telegramの音声バブルにはOpus/OGGの音声フォーマットが必要です:

- **OpenAI、ElevenLabs、Mistral** はOpusをネイティブに生成 — 追加のセットアップ不要
- **Opusに対応していないOpenAI互換サーバー**（Speaches/Kokoro、OpenRouterの音声ルート）: サーバーが `response_format: opus` を400/422で拒否した場合、Hermesはもう一度MP3を要求し、**ffmpeg** で変換
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

### ElevenLabs: エージェントが選ぶ音声

`tts.provider: elevenlabs` と `ELEVENLABS_API_KEY` を設定すると、エージェントは `text_to_speech` の呼び出しごとに異なる音声を選べます。たとえば、台本の登場人物ごとに1つの音声を割り当てられます。ElevenLabs版の `text_to_speech` には、他のプロバイダーには見えない2つの追加フィールドがあります（他のプロバイダーは設定された音声を使い続けます）:

- `voice_id` は、この呼び出しの音声を選びます。省略すると `tts.elevenlabs.voice_id` が使われます。設定が書き換えられることはありません。
- `find_voices` は、読み上げる代わりに音声を検索します。デフォルトで1ページあたり20件（最大100件）の音声を返し、次のページ用に `next_page` を返します:
  - `source: library`（デフォルト）は公開のVoice Libraryを検索します。数千のコミュニティ音声があり、よく使われているものから順に並びます。Library音声は事前に保存しなくても `voice_id` で直接使えます。ただしElevenLabsがAPI経由での利用を許可しているのは有料プランのみで、一部はより高いクレジットレート（`credit_rate` として表示）が適用されます。
  - `source: account` は、アカウント上の音声を一覧表示します。プリメイドの音声に加えて、保存、クローン、デザインしたものすべてが含まれます。
  - フィルター: 自由テキストの `query`、`gender`、`age`、`accent`、`language`（`en` のようなISOコード）、およびライブラリ用の `use_case`（例: `characters_animation`）と `sort`（`usage_character_count_1y`、`trending`、`cloned_by_count`、`created_date`）。

ライブラリ検索は、音声の名前と説明に含まれる単語を文字どおりに照合します。「orc」ではオークの音声が見つかりますが、「dwarf」では何も見つからないため、エージェントには音の特徴を記述する方法（`age: old` と `use_case: characters_animation` を伴う「gravelly old」）にフォールバックするよう指示されています。

ElevenLabsのAPIキーには、読み上げに **Text to Speech** 権限、検索に **Voices: Read** 権限が必要です。

エージェントは音声を聞くことができません。各音声のラベルと説明から選択します。すべての結果には `preview_url` が付いているため、決定する前に開いて音声を聞くことができます。

### xAIのカスタム音声（ボイスクローニング）

xAIは、自分の声をクローンしてTTSで使うことをサポートします。[xAI Console](https://console.x.ai/team/default/voice/voice-library) でカスタム音声を作成し、得られた `voice_id` を設定に指定します:

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # your custom voice ID
```

録音、サポートされるフォーマット、制限の詳細については、[xAI Custom Voices ドキュメント](https://docs.x.ai/developers/model-capabilities/audio/custom-voices) を参照してください。

### Piper（ローカル、44言語）

Piperは、Open Home Foundation（Home Assistantのメンテナー）による高速なローカルニューラルTTSエンジンです。完全にCPU上で動作し、事前学習済み音声で **44言語** をサポートし、APIキーは不要です。

**`hermes tools` でインストール** → Voice & TTS → Piper。HermesはPMを通じて
`piper` extraを要求します。準備済みのソースチェックアウトから実行する場合の明示的なコマンドは
`python -c "import pm; pm.sync_venv(['piper'], explicit=True)"` です。プラットフォームマーカーは引き続き適用されます。

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

### 読み上げトグルによるウォームアップとアンロード（ローカルエンジン）

ローカルエンジン（Piper、KittenTTS）はモデルを遅延ロードするため、何もしないと、読み上げをオンにした後の *最初の* 音声返信で、モデルのロード全体（新規インストールでは音声のダウンロードも）が最初の単語の前の無音として発生します。Hermesは、読み上げ出力のトグルをTTSがまもなく必要になるシグナルとして扱います:

- **デスクトップ** — **Read replies aloud**（コンポーザーのボタン、または Settings → Voice の **Read replies aloud in this app**）は、開いているすべてのDesktopウィンドウが従うデスクトップの設定です。オフにすると、読み上げ中の返信も停止します。Settings → Voice の **Read Responses Aloud** は、編集中のプロファイルのメッセージングとCLI用の `voice.auto_tts` です。これは一度だけデスクトップのトグルに移行され、その後の変更がデスクトップのトグルを上書きすることはありません。ローカルストレージがいっぱいか利用できない場合でも、選択はこのウィンドウでは維持されます。リロードをまたいだ永続化はベストエフォートのままです。**Read replies aloud** をオンにするか **voice conversation** を開始すると、設定されたエンジンがすぐにバックグラウンドでプリロードされます。両方を再びオフにすると、常駐モデル（Piperの音声は数十MB、KittenTTSは最大約80MB）がアンロードされ、無駄にRAMに置かれたままになりません。
- **CLI / TUI** — `/voice tts`（および `voice.auto_tts` が設定されている場合の `/voice on`）も同じことを行い、`/voice off` で解放されます。共有ゲートウェイに接続されたTUIは `/voice tts` を拒否します。ゲートウェイはそのターミナルで音声を再生できないためです。

各トグルはエンジンに対する *リース* を保持し、モデルはすべてのサーフェスにわたる最後のリースが解放されたときにのみアンロードされます。そのため、あるDesktopウィンドウで読み上げをオフにしても、別のウィンドウで実行中の会話から音声が取り上げられることはありません。アンロードは最後の解放から `tts.keep_warm_seconds`（デフォルト `60`）待機し、その間にいずれかのトグルが読み上げを再びオンにすると、ロード済みのモデルが維持されます。そのため、ウェイクワードのループやすぐに再開された音声会話で毎回音声を再ロードすることはありません。すぐにアンロードするには `0` に設定します。クラウドプロバイダーには保持するモデルがないため、トグルは遅延インストールされるSDK（edge-tts、ElevenLabs、Mistral）が存在することを確認するだけです。ウォームアップはベストエフォートです。エンジンをロードできない場合でもトグルは成功し、最初の返信は従来どおりオンデマンドでのロードにフォールバックします。

Desktopは `{"lease": "<name>", "active": true|false}` を指定して `POST /api/audio/tts-lease` を呼び出します。他のフロントエンドも同じエンドポイントを使えます。

同じリースはユーザーが宣言したプロバイダーにも届くため、セルフホスト型のTTSサーバーはトグルに応じてモデルをプリロード／アンロードできます。[コマンドプロバイダー](#custom-command-providers) は任意の `warm_command` / `release_command` を実行し、[Pythonプラグインプロバイダー](#python-plugin-providers) は `warm()` / `release()` を受け取ります。

### カスタムコマンドプロバイダー {#custom-command-providers}

使いたいTTSエンジンがネイティブにサポートされていない場合（VoxCPM、MLX-Kokoro、XTTS CLI、ボイスクローニングスクリプト、CLIを公開しているその他のもの）でも、Pythonを一切書かずに **コマンド型プロバイダー** として組み込めます。Hermesは入力テキストを一時的なUTF-8ファイルに書き込み、あなたのシェルコマンドを実行し、そのコマンドが生成した音声ファイルを読み込みます。

`tts.providers.<name>` の下に1つ以上のプロバイダーを宣言し、`tts.provider: <name>` で切り替えます — `edge` や `openai` といった組み込みプロバイダーを切り替えるのと同じ方法です。

```yaml
tts:
  provider: voxcpm                 # pick any name under tts.providers
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # try to deliver as a Telegram voice bubble

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # native Piper also supports custom .onnx via tts.piper.voice
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

**サポートされる `output_format` の値:** `mp3`（デフォルト）、`wav`、`ogg`、`flac`、`m4a`、`aac`、`amr`、`opus`。コマンドは実際にそのフォーマットを生成する必要があります（例: `ffmpeg` を使う）。Hermesは宣言された値を検証し、それに応じて出力ファイルに名前を付けるだけです。不明な値は `mp3` にフォールバックします。選択されたフォーマットは `{format}` プレースホルダーとしてコマンドにも公開されます。

**サブプロセスの環境:** コマンドプロバイダー（TTSとSTT）は、子プロセスの環境からHermesのシークレットを除去した状態で実行されます。ゲートウェイのボットトークン、LLMプロバイダーのAPIキー、内部リレーの認証情報は削除され、`PATH`、`HOME`、ロケール、その他の通常の変数は保持されます。コマンドテンプレートが環境から独自のAPIキーを必要とする場合（例: `curl` のワンライナー）は、プロバイダー設定の `env_passthrough` の下に変数名を列挙します:

```yaml
tts:
  providers:
    mycloud:
      type: command
      command: 'curl -s -H "Authorization: Bearer $MYCLOUD_API_KEY" ... -o {output_path}'
      env_passthrough: [MYCLOUD_API_KEY]
```


#### 例: Doubao（中国語 seed-tts-2.0） {#example-doubao-chinese-seed-tts-20}

ByteDanceの [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) 双方向ストリーミングAPIによる高品質な中国語TTSには、[`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPIパッケージをインストールし、コマンドプロバイダーとして組み込みます:

この外部コマンドプロバイダーは、HermesのPython環境ではなく、専用のツール環境にインストールしてください。
その実行ファイルを `PATH` 上で利用できるようにしてください。

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
| `timeout`          | `120`   | アイドル秒数。stdoutまたはstderrへの出力で期限がリセットされる。非アクティブ状態が続くとプロセスツリーが終了される（Unixは `killpg`、Windowsは `taskkill /T`）。 |
| `output_format`    | `mp3`   | `mp3` / `wav` / `ogg` / `flac` のいずれか。Hermesがパスを選ぶ場合は出力拡張子から自動推測される。      |
| `voice_compatible` | `false` | `true` のとき、HermesはMP3/WAV出力をffmpegでOpus/OGGに変換し、Telegramが音声バブルをレンダリングできるようにする。      |
| `max_text_length`  | `5000`  | コマンド呼び出し1回あたりの最大入力文字数。それより長いテキストは順序付きのチャンクに分割される。                  |
| `voice` / `model`  | 空   | プレースホルダーの値としてのみコマンドに渡される。                                                           |
| `warm_command` / `release_command` | 未設定 | サーフェスが読み上げ出力をオンにしたとき／すべてのサーフェスにわたる最後のリースが解放されたときに実行されるシェルコマンド。例: ローカルTTSサーバーをプリロードする `curl -s localhost:5002/load?model={model}` と、それに対応する `unload`。ベストエフォートかつノンブロッキングで、`command` と同じ `timeout`、`env_passthrough`、`{voice}` / `{model}` / `{speed}` プレースホルダーを使ってバックグラウンドで実行される。出力は破棄され、失敗はdebugレベルでログに記録されるだけ。 |

#### 挙動に関する注意

- **組み込みの名前が常に優先される。** `tts.providers.openai` のエントリがネイティブのOpenAIプロバイダーを覆い隠すことは決してないため、ユーザー設定が組み込みを密かに置き換えることはできません。
- **デフォルトの配信はドキュメント。** コマンドプロバイダーは、すべてのプラットフォームで通常の音声添付として配信します。プロバイダーごとに `voice_compatible: true` で音声バブル配信をオプトインします。
- **コマンドの失敗はエージェントに表出する。** ゼロ以外の終了、空の出力、タイムアウトはすべて、コマンドのstderr/stdoutを含むエラーを返すため、会話の中からプロバイダーをデバッグできます。
- **`command:` が設定されている場合、`type: command` がデフォルト。** `type: command` を明示的に書くのは良い習慣ですが必須ではありません。空でない `command` 文字列を持つエントリはコマンドプロバイダーとして扱われます。
- **`{input_path}` / `{text_path}` は互換。** コマンドの中で読みやすいほうを使ってください。

#### セキュリティ

コマンド型プロバイダーは、あなたが設定した任意のシェルコマンドを、あなたのユーザーの権限で実行します。Hermesはプレースホルダーの値をクォートし、設定されたタイムアウトを強制しますが、コマンドテンプレート自体は信頼されたローカル入力です — PATH上のシェルスクリプトと同じように扱ってください。

### Pythonプラグインプロバイダー {#python-plugin-providers}

単一のシェルコマンドでは表現できないTTSエンジン（CLIのないPython SDK、ストリーミングエンジン、音声一覧API、OAuthでリフレッシュする認証）には、`ctx.register_tts_provider()` を使ってPythonプラグインを登録します。プラグインは [カスタムコマンドプロバイダー](#custom-command-providers) のレジストリと **共存** します（置き換えるものではありません）。エンジンに合ったほうを選んでください。

#### どちらを選ぶか

| バックエンドが持っているもの | 使うもの |
|---|---|
| ファイル／stdinからテキストを読み、ファイル／stdoutに音声を書き込む単一のCLI | **コマンドプロバイダー**（Python不要） |
| シェルのパイプでつないだ2～3個のCLI | **コマンドプロバイダー** |
| Python SDKのみ — CLIなし | **プラグイン** |
| 合成しながら音声を出力するエンジンで、音声返信を最初の文から開始したい | **プラグイン**（`streams_pcm` を設定し、`stream()` をオーバーライド） |
| `hermes setup` で使われる音声一覧API | **プラグイン**（`list_voices()` をオーバーライド） |
| OAuthのリフレッシュフロー（静的なベアラートークンではない） | **プラグイン** |

組み込みが常に優先され、コマンドプロバイダーは同名のプラグインより優先されます。そのため、既存の設定を覆い隠す心配をせずに、組み込み以外の任意の名前でプラグインを安全に登録できます。

#### 最小限のプラグイン

これを `~/.hermes/plugins/my-tts/` に配置します:

`plugin.yaml`:
```yaml
name: my-tts
version: 0.1.0
description: "My custom Python TTS backend"
```

`__init__.py`:
```python
from agent.tts_provider import TTSProvider


class MyTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "my-tts"  # what tts.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom TTS"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_TTS_API_KEY"))

    def synthesize(self, text, output_path, *, voice=None, model=None,
                   speed=None, format="mp3", **extra) -> str:
        # Write audio bytes to output_path, return the path.
        # Raise on failure — the dispatcher converts exceptions to a
        # standard error envelope.
        import my_tts_sdk
        client = my_tts_sdk.Client()
        audio_bytes = client.synthesize(text=text, voice=voice or "default")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path


def register(ctx):
    ctx.register_tts_provider(MyTTSProvider())
```

有効化し（`hermes plugins enable my-tts`）、`tts.provider` をそれに向ける（`config.yaml` で `tts.provider: my-tts`）と、`text_to_speech` ツールはあなたのプラグインを経由してルーティングされます。

#### 任意のフック

より充実した統合のために、プロバイダークラスで次のものをオーバーライドします:

- `list_voices()` → `hermes tools` に表示される `{id, display, language, gender, preview_url}` の辞書のリスト。
- `list_models()` → `{id, display, languages, max_text_length}` の辞書のリスト。
- `get_setup_schema()` → `{name, badge, tag, env_vars: [{key, prompt, url}]}` を返し、`hermes tools` / `hermes setup` のピッカー行を表示します。これがなくてもプラグインは動作しますが、ピッカーでの行は最小限になります。
- `stream(text, *, voice, model, format, **extra)` → 音声バイトをyieldするイテレーター（デフォルトでは `NotImplementedError` を送出）。
- `streams_pcm = True` + `stream_sample_rate`（Hz） → ストリーミング音声パス（CLI/TUIの音声モード、デスクトップの読み上げ、ゲートウェイのストリーミング音声）に参加します。Hermesは `synthesize()` が受け取るのと同じ `tts.voice` / `tts.model` / `tts.speed` を使って `stream(text, format="pcm", voice=..., model=..., speed=...)` を呼び出し、そのレートの生のint16リトルエンディアン・モノラルPCMを期待します。両方の属性と `is_available()` は返信が始まるたびに読み取られるため、ライブの状態を反映するプロパティにできます。`synthesize()` と異なり、`stream()` は前の音声が再生されている間に最大3つの連続した文に対して同時に呼び出される可能性があるため、スレッドセーフである必要があります。正の `stream_sample_rate` がない場合、または `is_available()` が `False` の場合、Hermesは引き続き1文ずつ合成します。
- `voice_compatible` プロパティ → 出力がOpus互換で、ゲートウェイが音声バブルとして配信すべき場合は `True` に設定します（デフォルトは `False` = 通常の音声添付）。
- `warm()` / `release()` → あなたのプロバイダーが設定された `tts.provider` である間に、サーフェスが読み上げ出力をオンにしたとき／すべてのサーフェスにわたる最後のリースが解放されたときに呼び出されます。ここでローカルのモデルサーバーをプリロードまたはアンロードします。どちらもデフォルトでは何もしません。例外はdebugレベルでログに記録され、トグルを失敗させることはありません。

docstringを含む完全なABCについては、`agent/tts_provider.py` を参照してください。

## 音声メッセージの文字起こし（STT） {#voice-message-transcription-stt}

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
# In ~/.hermes/config.yaml
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra"
  language: "en"              # Global hint; blank falls back to HERMES_LOCAL_STT_LANGUAGE, then auto-detect
  local:
    model: "base"             # tiny, base, small, medium, large-v3, turbo
    language: ""              # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
  groq:
    language: ""              # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-voice-transcribe-2.0"  # or grok-voice-transcribe-1.0
    language: ""              # blank inherits stt.language, then HERMES_LOCAL_STT_LANGUAGE
```

この例で空になっているプロバイダーの言語設定は、グローバルな `"en"` を継承します。自動検出を選択するわけではありません。優先順位、自動検出の例、コマンドバックエンドの制限については、[STTの言語解決](../configuration.md#speech-to-text-stt) を参照してください。

### プロバイダーの詳細

**Local（faster-whisper）** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) を介してローカルでWhisperを実行します。デフォルトではCPUを使い、利用可能な場合はGPUを使います。モデルサイズ:

| モデル | サイズ | 速度 | 品質 |
|-------|------|-------|---------|
| `tiny` | 約75 MB | 最速 | 基本的 |
| `base` | 約150 MB | 高速 | 良好（デフォルト） |
| `small` | 約500 MB | 中速 | より良い |
| `medium` | 約1.5 GB | やや遅い | 優秀 |
| `large-v3` | 約3 GB | 最遅 | 最高 |

初回使用時に、選択したモデルが `huggingface.co` からダウンロードされます。以降のロードではローカルキャッシュが優先され、オンラインでのリビジョンチェックは不要です。Hubが利用できないネットワークでは、Hermesを起動するシェルまたはサービスで、アクセス可能なミラーをexportしてください:

```bash
HF_ENDPOINT=https://your-hugging-face-mirror.example
HF_HUB_DISABLE_XET=1
```

`HF_HUB_DISABLE_XET=1` は、`HF_ENDPOINT` を尊重しないXet CASホストに接続する代わりに、ダウンロードをミラーの通常のHTTPパスに保ちます。

**Groq API** — `GROQ_API_KEY` が必要です。無料のホスト型STTオプションが欲しいときの、良いクラウドフォールバックです。プロバイダー固有の言語ヒントには `stt.groq.language` を、グローバルなヒントには `stt.language` を使います。`HERMES_LOCAL_STT_LANGUAGE` は、どちらの設定も空でないヒントを提供しない場合にのみ使われます。解決後にヒントが残っていない場合、Groq Whisperは自動検出します。

**OpenAI API** — まず `VOICE_TOOLS_OPENAI_KEY` を受け付け、`OPENAI_API_KEY` にフォールバックします。`whisper-1`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe`、`gpt-transcribe` をサポートします。

**Mistral API（Voxtral Transcribe）** — `MISTRAL_API_KEY` が必要です。Mistralの [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) モデルを使います。13言語、話者ダイアライゼーション、単語レベルのタイムスタンプをサポートします。`cd ~/.hermes/hermes-agent && python -c "import pm; pm.sync_venv(['mistral'], explicit=True)"` でインストールします。

**xAI Grok STT** — `XAI_API_KEY`（またはxAI OAuth）が必要です。`https://api.x.ai/v1/stt` にmultipart/form-dataとしてPOSTし、`model`（デフォルトは `grok-voice-transcribe-2.0`。`stt.xai.model` または `STT_XAI_MODEL` で `grok-voice-transcribe-1.0` に固定可能）を送信します。Hermesは常にモデルを指定するため、サーバー側のデフォルトが変わっても、実行するものが黙って切り替わることはありません。`stt.language: ""`（自動検出）の場合、xAIはテキストのフォーマットに言語を必要とするため、`format` フラグは省略されます。チャットやTTSですでにxAIを使っていて、すべてを1つのAPIキーで済ませたい場合に良い選択肢です。自動検出の順序ではMistralの後に置かれます — 強制するには明示的に `stt.provider: xai` を設定してください。

**カスタムローカルCLIフォールバック** — Hermesにローカルの文字起こしコマンドを直接呼び出させたい場合は、`HERMES_LOCAL_STT_COMMAND` を設定します。コマンドテンプレートは `{input_path}`、`{output_dir}`、`{language}`、`{model}` のプレースホルダーをサポートします。Hermesはレンダリングしたテンプレートを引数リストにトークン化し、シェルを介さずに実行するため、`|`、`>`、`&&`、`;` などの演算子はリテラルの引数として渡されます。コマンドは、`{output_dir}` の下のどこかに `.txt` の文字起こしを書き込む必要があります。

#### 例: Doubao / Volcengine ASR {#example-doubao--volcengine-asr}

Doubao TTSに [`doubao-speech`](https://pypi.org/project/doubao-speech/) を使っている場合（[上記](#example-doubao-chinese-seed-tts-20) を参照）、同じパッケージがローカルコマンドのSTT表面を介して音声からテキストへの変換も処理します:

この外部コマンドプロバイダーは、HermesのPython環境ではなく、専用のツール環境にインストールしてください。
その実行ファイルを `PATH` 上で利用できるようにしてください。

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

信頼できるローカルテンプレートで意図的にパイプ、リダイレクト、その他のシェル機能が必要な場合は、シェルを明示的に呼び出してください。動的なパスはシェルプログラムの外に置き、位置引数として渡します:

```bash
export HERMES_LOCAL_STT_COMMAND='sh -c '\''whisper "$1" --output_format txt --output_dir "$2" | tee "$2/whisper.log"'\'' _ {input_path} {output_dir}'
```

Windowsでは、代わりに明示的な `cmd /c` またはPowerShellのラッパーを使ってください。明示的なラッパーにより、シェルによる解釈は、すべてのローカルSTTテンプレートの暗黙の性質ではなく、設定されたargvのオプトインの一部になります。

```yaml
stt:
  provider: local_command
```

Hermesは受信した音声メッセージを `{input_path}` に書き込み、コマンドを実行し、`{output_dir}` の下に生成された `.txt` ファイルを読み込みます。言語はVolcengineのbigmodelエンドポイントによって自動検出されます。

### フォールバックの挙動

**明示的な** `stt.provider` の選択（`config.yaml` に書き込まれたもの。例: `hermes tools` 経由）は厳密に尊重されます — そのプロバイダーが実行できない場合、黙ってエンジンを切り替えるのではなく、明確なエラー（`stt is configured to use <provider> (set via hermes tools), but <failure>. Run 'hermes tools' to change it.`）で文字起こしが失敗します。設定に書き込まれた `stt.provider: local` も明示的な選択とみなされる点に注意してください。

**プロバイダーが一度も選択されていない** 場合、Hermesは利用可能なものから自動検出します:
- **ローカルのfaster-whisperが利用不可** → クラウドプロバイダーの前に、ローカルの `whisper` CLIまたは `HERMES_LOCAL_STT_COMMAND` を試す
- **Groqキーが未設定** → スキップ。次に利用可能なプロバイダーへ
- **OpenAIキーが未設定** → スキップ。次に利用可能なプロバイダーへ
- **Mistralのキー／SDKが未設定** → 自動検出でスキップ。次に利用可能なプロバイダーにフォールスルー
- **何も利用できない** → 音声メッセージは、ユーザーへの正確な注記とともにそのまま通過する

### STTのカスタムコマンドプロバイダー {#stt-custom-command-providers}

使いたいSTTエンジンがネイティブにサポートされていない場合（Doubao ASR、NVIDIA Parakeet、whisper.cppのビルド、オープンソースのSenseVoice CLI、シェルコマンドを公開しているその他のもの）でも、Pythonを一切書かずに **コマンド型プロバイダー** として組み込めます。Hermesは音声ファイルに対してあなたのシェルコマンドを実行し、文字起こしを読み戻します。

`stt.providers.<name>` の下に1つ以上のプロバイダーを宣言し、`stt.provider: <name>` で切り替えます — TTSの [コマンドプロバイダーレジストリ](#custom-command-providers) と同じ形で、入力=音声 → 出力=文字起こしという方向に合わせたものです。

```yaml
stt:
  provider: parakeet                # pick any name under stt.providers
  providers:
    parakeet:
      type: command
      command: "parakeet-asr --model nvidia/parakeet-tdt-0.6b-v2 --in {input_path} --out {output_path}"
      format: txt
      language: en
      timeout: 300

    whispercpp:
      type: command
      command: "whisper-cli -m ~/models/ggml-large-v3.bin -f {input_path} -otxt -of {output_dir}/transcript"
      format: txt

    sensevoice:
      type: command
      command: "sensevoice-cli {input_path} --json | tee {output_path}"
      format: json
```

これは、組み込みの `local_command` パスを介したレガシーの `HERMES_LOCAL_STT_COMMAND` エスケープハッチを補完するものです。シェル駆動のコマンドプロバイダーレジストリとは異なり、レガシーのテンプレートはargvにトークン化され、暗黙のシェル解釈なしで実行されます。シェル駆動のSTTエンジンを **複数** 使いたい場合、`stt.provider` で選べる名前が欲しい場合、またはプロバイダーごとの `language` / `model` / `timeout` が必要な場合は、`stt.providers.<name>` を使ってください。

#### STTのプレースホルダー

コマンドテンプレートでは、次のプレースホルダーを参照できます。Hermesはレンダリング時にそれらを置換し、各値を周囲のコンテキスト（裸／シングルクォート／ダブルクォート）に合わせてシェルクォートするため、スペースを含むパスでも安全です。

| プレースホルダー       | 意味                                                              |
|-------------------|----------------------------------------------------------------------|
| `{input_path}`    | 入力音声ファイルの絶対パス（元の場所、読み取り専用。`normalize: true` のときは16 kHzモノラルのm4a） |
| `{output_path}`   | コマンドが文字起こしを書き込むべき絶対パス             |
| `{output_dir}`    | `{output_path}` の親ディレクトリ（whisper系のツールで便利）  |
| `{format}`        | 設定された出力フォーマット: `txt` / `json` / `srt` / `vtt`             |
| `{language}`      | 設定された言語コード（デフォルトは `en`）                          |
| `{model}`         | `stt.providers.<name>.model`。未設定なら空                       |

リテラルの中括弧には `{{` と `}}` を使います（コマンドにJSONスニペットを埋め込むときに便利です）。

#### 文字起こしの読み戻し方

コマンドが正常に終了した後:

1. `{output_path}` が存在し、空でない場合 → HermesはそれをUTF-8テキストとして読み込みます。
2. そうでなく、コマンドがstdoutに書き込んだ場合 → Hermesはそれを使います。
3. それ以外の場合 → エラー: "Command STT provider wrote no output file and produced no stdout"。

これにより、ファイルに書き込むCLI（`whisper-cli`、`parakeet-asr`）と、文字起こしをstdoutに出力するcurl風のワンライナー（`curl … | jq -r .text`）の両方にレジストリを使えます。

`format: json` / `srt` / `vtt` の場合、Hermesは生のファイル内容を `transcript` フィールドとして返します。JSONから `.text` を抽出することはランナーの対象外です — `format: txt` を設定するか、下流でJSONを後処理してください。

#### STTコマンドプロバイダーの任意のキー

| キー             | デフォルト | 意味                                                                                              |
|-----------------|---------|------------------------------------------------------------------------------------------------------|
| `timeout`       | `300`   | 秒。期限切れになるとプロセスツリーが終了される（Unixは `start_new_session`、Windowsは `taskkill /T`）。     |
| `format`        | `txt`   | `txt` / `json` / `srt` / `vtt` のいずれか。`{output_path}` の拡張子を設定する。                       |
| `language`      | `en`    | `{language}` に転送される。[STTの言語解決](../configuration.md#speech-to-text-stt) に従い、最終的には `en` にフォールバックする。 |
| `model`         | 空   | `{model}` に転送される。`transcribe_audio()` への `model=` 引数がこれを上書きする。                |
| `normalize`     | `false` | コマンドの実行前に入力を16 kHzモノラルのm4aにトランスコードする（ffmpeg）。その場合 `{input_path}` は正規化されたファイルを指す。 |

#### STTコマンドプロバイダーの挙動に関する注意

- **組み込みが常に優先される。** `stt.providers.openai: type: command` を宣言しても、本物のOpenAI Whisperハンドラーは上書きされません。組み込みの名前は、コマンドプロバイダーのリゾルバーが実行される前に短絡されます。
- **プロセスツリーのクリーンアップ。** `timeout` を超えて実行されたコマンドは、シェルのラッパーだけでなく、プロセスツリー全体が終了されます。モデルをロードするサブプロセスをforkする長時間実行のASRパイプラインも確実に回収されます。
- **シェルクォートは自動。** `'…'` の中のプレースホルダーにはシングルクォート安全なエスケープが、`"…"` の中には `$`/`` ` ``/`"` のエスケープが、クォートの外には `shlex.quote` が適用されます。プレースホルダーの値を事前にクォートしないでください。

#### STTコマンドプロバイダーのセキュリティ

シェルコマンドはHermesと同じユーザーで、ファイルシステムへの完全なアクセス権を持って実行されます — `tts.providers.<name>: type: command` や `HERMES_LOCAL_STT_COMMAND` と同じ信頼モデルです。信頼できるソースからのコマンドプロバイダーのみを宣言してください。

### Pythonプラグインプロバイダー（STT） {#python-plugin-providers-stt}

組み込みではなく、かつシェルコマンドでも表現できないSTTエンジン（Python SDK、OAuthでリフレッシュする認証、ストリーミングチャンクなどが必要なもの）には、`ctx.register_transcription_provider()` を使ってPythonプラグインを登録します。プラグインは8つの組み込みプロバイダー（`local`、`local_command`、`groq`、`openai`、`mistral`、`xai`、`elevenlabs`、`deepinfra`）および `stt.providers.<name>: type: command` レジストリと **共存** します — 組み込みはネイティブの実装を維持し、名前が衝突した場合は常に優先されます。コマンドプロバイダーは同名のプラグインより優先されます（設定はプラグインのインストールよりもローカルであるため）。

#### どちらを選ぶか（STT）

| バックエンドが持っているもの                                                 | 使うもの                                                              |
|--------------------------------------------------------------|------------------------------------------------------------------|
| 音声ファイルを受け取ってテキストを出力する単一のシェルコマンド | `stt.providers.<name>: type: command`（Python不要）        |
| レガシーの単一コマンドのエスケープハッチだけが欲しい        | `HERMES_LOCAL_STT_COMMAND` 環境変数（argvにトークン化。暗黙のシェルなし） |
| CLIのないPython SDK                                     | `register_transcription_provider()` プラグイン                      |
| OAuthでリフレッシュする認証、ストリーミングチャンク、音声一覧のメタデータ | `register_transcription_provider()` プラグイン                      |
| 組み込みですでにカバーされている（`local`、`groq`、`openai`、…）  | `stt.provider: <name>` を設定 — 組み込みはインライン               |

#### 解決の順序

1. **`stt.provider` が組み込みの名前** → 組み込みのディスパッチ。**常に優先。**
2. **`stt.provider` が `command:` の設定された `stt.providers.<name>` に一致** → コマンドプロバイダーのランナー（[STTのカスタムコマンドプロバイダー](#stt-custom-command-providers) を参照）。同名のプラグインより優先。
3. **`stt.provider` がプラグインで登録された `TranscriptionProvider` に一致** → プラグインのディスパッチ:
   - プラグインの `is_available()` が `False` を返す場合（認証情報やSDKの欠如）、呼び出しはプラグインを特定する利用不可のエラーエンベロープを返します — 汎用の "No STT provider available" メッセージでは **ありません**。
   - それ以外の場合、プラグインの `transcribe()` が `model`（公開の `model=` 引数から。なければ `stt.<provider>.model` にフォールバック）と、解決された `language` ヒント（[STTの言語解決](../configuration.md#speech-to-text-stt) を参照）とともに呼び出されます。
4. **一致なし** → "No STT provider available" エラー。

#### プロバイダーごとの設定名前空間

プラグインは、組み込みが `stt.openai.model` / `stt.mistral.model` を読むのと同じように、`config.yaml` の `stt.<provider>` からプロバイダーごとの設定を読み込みます:

```yaml
stt:
  provider: my-stt
  my-stt:
    model: whisper-large-v3
    language: ja          # forwarded as language= to transcribe()
    # any other plugin-specific keys go here; read them via your
    # own config.yaml access in __init__/is_available/transcribe
```

その他のプラグイン固有のキーは、プラグイン自身が読み込みます。

#### 最小限のプラグイン

これを `~/.hermes/plugins/my-stt/` に配置します:

`plugin.yaml`:
```yaml
name: my-stt
version: 0.1.0
description: "My custom Python STT backend"
```

`__init__.py`:
```python
from agent.transcription_provider import TranscriptionProvider


class MySTTProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "my-stt"  # what stt.provider matches against

    @property
    def display_name(self) -> str:
        return "My Custom STT"

    def is_available(self) -> bool:
        # Return False when credentials/deps are missing — picker skips
        # this row but the dispatcher still routes here on explicit config.
        import os
        return bool(os.environ.get("MY_STT_API_KEY"))

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        # Return the standard transcribe envelope:
        #   {"success": bool, "transcript": str, "provider": str, "error": str}
        # Do NOT raise — convert exceptions to the error envelope so the
        # gateway/CLI caller sees a consistent shape on failure.
        try:
            import my_stt_sdk
            client = my_stt_sdk.Client()
            text = client.transcribe(open(file_path, "rb"))
            return {
                "success": True,
                "transcript": text,
                "provider": "my-stt",
            }
        except Exception as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"my-stt failed: {exc}",
                "provider": "my-stt",
            }


def register(ctx):
    ctx.register_transcription_provider(MySTTProvider())
```

有効化し（`hermes plugins enable my-stt`）、`config.yaml` で `stt.provider: my-stt` を設定すると、音声メッセージの文字起こしはあなたのプラグインを経由してルーティングされます。

#### 任意のフック

より充実した統合のために、プロバイダークラスで次のものをオーバーライドします:

- `list_models()` → `{id, display, languages, max_audio_seconds}` の辞書のリスト。
- `default_model()` → ユーザーがモデルを上書きしない場合に返される文字列。
- `get_setup_schema()` → `{name, badge, tag, env_vars: [{key, prompt, url}]}` を返し、`hermes tools` / `hermes setup` のピッカー行を表示します（STT用のピッカーカテゴリはまだ提供されていません — このメタデータは将来の互換性のためにプラグインで利用できます）。

docstringを含む完全なABCについては、`agent/transcription_provider.py` を参照してください。
