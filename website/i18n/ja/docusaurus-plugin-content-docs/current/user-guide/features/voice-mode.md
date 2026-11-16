---
sidebar_position: 10
title: "ボイスモード"
description: "Hermes Agent とのリアルタイム音声会話 — CLI、Telegram、Discord（DM、テキストチャンネル、ボイスチャンネル）"
---

# ボイスモード

Hermes Agent は、CLI とメッセージングプラットフォームの両方で完全な音声インタラクションをサポートします。マイクを使ってエージェントに話しかけ、音声による返答を聞き、Discord のボイスチャンネルでライブの音声会話を行えます。

推奨設定や実際の利用パターンを含む実践的なセットアップ手順については、[Hermes でボイスモードを使う](../../guides/use-voice-mode-with-hermes.md)を参照してください。

ハンズフリーでのセッション開始 — 「hey hermes」（または任意のフレーズ）と話しかけて、CLI、TUI、デスクトップアプリで新しい音声セッションを開く — については、[ウェイクワード](/docs/user-guide/features/wake-word)を参照してください。

## 前提条件

音声機能を使う前に、以下を確認してください:

1. **Hermes Agent がインストール済みであること** — インストールスクリプト経由（[インストール](../../getting-started/installation.md)を参照）
2. **LLM プロバイダーが設定済みであること** — `hermes model` を実行するか、`~/.hermes/.env` に希望するプロバイダーの認証情報を設定する
3. **動作するベースセットアップがあること** — `hermes` を実行し、音声を有効化する前にエージェントがテキストに応答することを確認する

:::tip
`~/.hermes/` ディレクトリとデフォルトの `config.yaml` は、初めて `hermes` を実行したときに自動的に作成されます。API キー用の `~/.hermes/.env` のみ手動で作成する必要があります。
:::

:::tip Nous Portal で両方をカバー
有料の [Nous Portal](./tool-gateway.md) サブスクリプションは、LLM（手順 2）**と**、Tool Gateway 経由の OpenAI TTS の両方を提供します — 別途 OpenAI キーは不要です。新規インストールでは、`hermes setup --portal` で両方を一度に設定できます。
:::

## 概要

| 機能 | プラットフォーム | 説明 |
|---------|----------|-------------|
| **インタラクティブボイス** | CLI | Ctrl+B を押して録音、エージェントが無音を自動検出して応答 |
| **自動ボイス返信** | Telegram、Discord | エージェントがテキスト応答とあわせて音声を送信 |
| **ボイスチャンネル** | Discord | ボットが VC に参加し、ユーザーの発話を聞き取り、返答を話す |

## 要件

### Python パッケージ

ボイスプロバイダーの設定には `hermes tools` を使います。不足している組み込み機能の依存関係は、`security.allow_lazy_installs` とターゲットの依存関係サポートに従って PM 経由で導入されます。選択した依存関係の環境が変わった場合は、Hermes を再起動してください。

バンドル版アプリには、サポートされるエンジンの依存関係が含まれています。Docker には厳選されたサブセットが含まれ、オンデマンドのインストールは無効になっています。署名済みのペイロードやシステムの Python を pip で変更しないでください。手動の開発環境では、[開発環境のセットアップ](/docs/developer-guide/contributing#development-setup)で必要なエクストラを選択してください。

| エクストラ | パッケージ | 必要な用途 |
|-------|----------|-------------|
| `voice` | `sounddevice`、`numpy`、およびサポートされる環境では Faster-Whisper | CLI のオーディオとオプションのローカル STT |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Discord & Telegram ボット |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS プロバイダー |

ネイティブ Windows ARM64 と Intel macOS では、ローカルの Faster-Whisper は除外されます。これらのターゲットでは、クラウドまたはコマンドベースの STT プロバイダーを使ってください。`audio-io` には、ローカル STT を除いたマイク/再生の依存関係が含まれます。`all` エクストラは、すべての音声エンジンやウェイクエンジンを意味するわけではありません。

NeuTTS は別のオプションのランタイムで、初回使用時にモデルをダウンロードします。その依存関係を署名済みのアプリやシステムの Python にインストールしないでください。

:::info
`discord.py[voice]` は **PyNaCl**（音声暗号化用）と **opus バインディング**を自動的にインストールします。これは Discord ボイスチャンネルのサポートに必要です。
:::

### システム依存関係

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # for NeuTTS

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # for NeuTTS
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
# Speech-to-Text — local provider needs NO key at all
# PM prepares local Faster-Whisper on supported targets; no STT API key is needed.
GROQ_API_KEY=your-key                 # Groq Whisper — fast, free tier (cloud)
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — paid (cloud)

# Text-to-Speech (optional — Edge TTS and NeuTTS work without any key)
ELEVENLABS_API_KEY=***           # ElevenLabs — premium quality
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS
```

:::tip
`faster-whisper` がインストールされていれば、ボイスモードは STT に **API キーゼロ**で動作します。モデル（`base` で約 150 MB）は初回使用時に自動的にダウンロードされます。
:::

初回のダウンロードは通常 `huggingface.co` から行われます。ネットワーク上でこのホストが利用できない場合は、Hermes を起動するシェルまたはサービスで、アクセス可能なミラーを export してください:

```bash
HF_ENDPOINT=https://your-hugging-face-mirror.example
HF_HUB_DISABLE_XET=1
```

Xet を無効にすると、ミラー使用時に Xet の別の CAS ホストで発生する認証エラーを回避できます。モデルがキャッシュされた後は、Hermes はオンラインでのリビジョン確認なしにそのスナップショットをロードします。

---

## CLI ボイスモード

ボイスモードは **クラシック CLI**（`hermes chat`）と **TUI**（`hermes --tui`）の両方で利用できます。動作はどちらも同一です — 同じスラッシュコマンド、同じ VAD 無音検出、同じストリーミング TTS、同じハルシネーションフィルター。TUI はさらにクラッシュフォレンジックログを `~/.hermes/logs/` に転送するため、特殊なオーディオバックエンドでのプッシュトゥトークの失敗を、静かに消えるのではなく完全なスタックトレースとともに報告できます。

### クイックスタート

CLI を起動してボイスモードを有効化します:

```bash
hermes                # Start the interactive CLI
```

次に、CLI 内で以下のコマンドを使います:

```
/voice          Toggle voice mode on/off
/voice on       Enable voice mode
/voice off      Disable voice mode
/voice tts      Toggle TTS output
/voice status   Show current state
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

### 音声でボイスチャットを終了する

**「stop」** とだけ話すと、ハンズフリーで音声会話を終了できます。一致判定は意図的に厳格です: 発話全体（大文字と小文字を区別せず、前後の句読点は無視）が設定されたフレーズと一致する必要があるため、「stop doing that and try X instead」のような発話は通常どおりエージェントに届きます。フレーズのリストは `config.yaml` の `voice.stop_phrases` でカスタマイズでき（例: `["stop", "goodbye hermes"]`）、`[]` に設定すると無効になります。フレーズはどの言語でも構いません（例: `stt.language: ru` と組み合わせた `["отбой", "стоп"]`）。デスクトップアプリも同じリストに従います。`voice.stop_phrases` がデフォルトのままの場合は、いくつかの英語の追加フレーズ（「goodbye」、「never mind」、「cancel」など）も受け付けますが、カスタマイズしたリストはそれらを置き換えます。ボイスチャットは、3 回連続で無音のサイクル（発話が検出されない）が続いた場合にも自動的に終了します。

ボイスチャット中に停止フレーズだけを **入力** した場合も、すべてのサーフェス（CLI、TUI、デスクトップ）で同じように動作します: そのメッセージはエージェントに送信されず、ボイスチャットを終了します。ボイスチャット以外では、入力された「stop」は通常のメッセージです。

### ライブ文字起こし（ストリーミング STT）

`stt.streaming: true` を設定すると、話し終えた後ではなく話している最中に文字起こしが行われます。話している間に途中のテキストが表示され（クラシック CLI と TUI の入力プレースホルダー、およびデスクトップのディクテーションピル）、全体のアップロードを待つことなく、話し終えた時点で文字起こしが完了しています。

```yaml
stt:
  provider: openai        # or xai, elevenlabs
  streaming: true
  openai:
    streaming_model: gpt-live-transcribe   # the default; the one OpenAI model that streams text mid-utterance
```

| プロバイダー | ライブエンドポイント | 備考 |
|---|---|---|
| `openai` | Realtime 文字起こしセッション | 自分の `OPENAI_API_KEY` が必要。Nous が管理するオーディオゲートウェイはファイルの文字起こしのみ対応 |
| `xai` | `wss://api.x.ai/v1/stt` | `XAI_API_KEY` が必要（Grok の OAuth ログインはライブ STT には使われません） |
| `elevenlabs` | Scribe v2 realtime | `ELEVENLABS_API_KEY` |
| プラグイン | `TranscriptionProvider.streaming_capable` | プラグインは `open_stream_session()` でオプトイン |

ライブ文字起こしは、CLI と TUI のボイスモード、およびデスクトップのディクテーションに対応します。ローカルの whisper、Groq、Mistral、DeepInfra は引き続きファイル経由の方式を使います。ライブセッションを開けない場合や録音中に失敗した場合は、Hermes が通常どおり録音を文字起こしするため、これを有効にしても録音が失われることはありません。

### ストリーミング TTS

TTS が有効な場合、エージェントはテキストを生成しながら **文単位で** 返答を話します — 応答全体を待つ必要はありません。これは **すべての TTS プロバイダー** で動作します:

1. テキストのデルタを完全な文にバッファリングします（最小 20 文字）
2. Markdown 書式、絵文字、`<think>` ブロックを除去します
3. 文ごとに音声をリアルタイムで再生します — チャンク化された PCM API を持つプロバイダー（ElevenLabs、OpenAI）は最初の単語までの時間を最短にするために生のオーディオをストリーミングし、それ以外のすべてのプロバイダー（デフォルトの Edge を含む）は文が完成するたびにその文を合成して再生します

同じパイプラインが、クラシック CLI、TUI、デスクトップアプリで動作します。デスクトップの音声会話では、モデルが生成する返答テキストが返答ごとの音声 WebSocket に **ライブで** 送られるため、音声の再生と生成が並行します — 返答ごとに 1 つのソケットと 1 つのオーディオクロックを使い、文ごとの接続の切れ目はありません。

### デスクトップのリモート接続: クライアント直結の音声（最短経路）

Hermes Desktop が **リモートゲートウェイ** に接続されている場合、オーディオをゲートウェイ経由で中継する必要はまったくありません。音声セッションの開始時に、デスクトップは認証済みの REST チャンネル（`GET /api/audio/voice-config`）を通じて、アクティブなプロファイルの解決済み STT/TTS 設定（プロバイダー、モデル、言語/音声、認証情報）をゲートウェイから取得し、その後プロバイダーを **直接** 呼び出します:

- **ディクテーション / 音声入力:** マイクの録音はデスクトップからプロファイルの STT プロバイダーに直接送られ、結果の *テキスト* だけがプロンプトとしてゲートウェイに送信されます。
- **音声による返答:** 返答テキストはすでにチャットソケット経由でデスクトップにストリーミングされているため、デスクトップはプロファイルの TTS プロバイダーでローカルに合成して再生します — ゲートウェイとの接続がオーディオを運ぶことはありません。

クライアント側で設定するものはありません: 話しかけているプロファイルが、ゲートウェイ自身が処理した場合とまったく同じように、プロバイダーとキーの唯一の情報源になります。キーはセッションの間だけデスクトップのメモリに保持され、クライアントのディスクに書き込まれることはありません。

ゲートウェイホスト上でしか実行できないプロバイダー（ローカルの whisper、`edge` TTS、コマンドプロバイダー、プラグイン）は、このエンドポイントを持たない古いバックエンドと同様に、自動的に中継経路（`/api/audio/transcribe` と音声 WebSocket）にフォールバックします。Webapp（ブラウザで動作するデスクトップ UI）は常に中継します: ブラウザのタブはクロスオリジンリクエストを許可しないプロバイダー（Nous が管理するオーディオなど）を呼び出せず、キーはホスト上に留まります。すべてのプロバイダーで中継を強制するには、次のように設定します:

```yaml
voice:
  client_direct: false
```

クライアント直結の対応状況: OpenAI 互換の形式による OpenAI（Nous が管理するオーディオを含む）、Groq、Mistral、DeepInfra、xAI Grok STT、ElevenLabs STT + TTS。OAuth で設定された xAI は中継のままです（OAuth のベアラートークンはサーバー側で更新されるため）。

### デスクトップ: GPT-Live ボイスチャットモード（全二重、Hermes に委譲）

上記の連鎖ループは、デスクトップアプリにある 2 つのボイスチャットモードのうちの 1 つです。もう 1 つは、STT → ターン → TTS の連鎖全体を **1 つの全二重音声モデル**、OpenAI の `gpt-live-1` に置き換えます: 話しながら聞き取り、割り込み、相づち、背景ノイズを自ら処理し、**独自のツールは持ちません**。実際の作業を依頼するたびに Hermes に *委譲* し、Hermes は通常どおり — セッションで選択されているモデルとプロバイダー、フルツールセット、メモリ、承認を使って — 回答し、音声がその回答を言い換えて読み上げます。

```yaml
voice:
  voice_chat_mode: gpt-live     # chained (default) | gpt-live
  gpt_live:
    voice: marin                # marin, cedar, quartz, ripple, vesper, willow, stone, gleam, meridian, …
    instructions: ""            # optional extra persona sentences (tone, pace, language)
```

要件: OpenAI の API キー（`OPENAI_API_KEY`、`VOICE_TOOLS_OPENAI_KEY`、または `voice.gpt_live.api_key`）。音声レイヤーは OpenAI によって **セッション時間 1 分あたり $0.05** で課金されます（アイドル時間も含まれます）。Hermes のターンは、これまでどおりそのプロバイダーで課金されます。このモードは Settings → Voice → *Voice Chat Mode* でも設定できます。

仕組み: 音声ボタンを押すと、デスクトップから GPT-Live への WebRTC セッションが開かれます。デスクトップが受け取るのはセッション ID と SDP アンサーだけです — キーはゲートウェイホストに留まり、ゲートウェイホストがセッションの作成（`POST /api/audio/voice-live/session`）を行います。各 `session.delegation.created` は、開いているチャットでの通常のターンになります（バブルにはあなたが話した内容が表示されます。直近の音声でのやり取りは、システムプロンプトではなくターンごとの注記としてモデル入力に含まれるため、返答は読み上げやすい文章になります）。ツールのアクティビティは控えめなコンテキスト（「Hermes is working: terminal」）として音声に渡されるため、尋ねれば何が起きているかを教えてくれます。最終的な回答は文単位でストリーミングされて返されます。停止フレーズを話すと会話が終了します。`gpt-live` が選択されているのにキーが解決できない場合、ボタンは通知とともに連鎖モードにフォールバックします。

このモードでサポートされないもの: Nous が管理するオーディオプロキシ（直接のキーのみ）、CLI/TUI（`/voice` は連鎖ループのまま）、`tts` ツール（引き続き `tts.provider` を使います）。

### ボイスチャットモデル {#voice-chat-model}

音声によるターンは、入力するときとは別の（通常はより高速な）モデルで実行できます。`voice_chat` 補助スロットを、デスクトップの Settings → Models → Auxiliary models、`hermes model` → Auxiliary models、または config.yaml で設定します:

```yaml
auxiliary:
  voice_chat:
    provider: openrouter          # "auto" = the session's model (default)
    model: google/gemini-3-flash-preview   # empty with a provider = that provider's fast model
    reasoning_effort: none        # default: reasoning off on voice turns (see below)
```

音声ターンでは、スロットが `auto` のままでセッションのモデルが回答する場合も含めて、デフォルトで推論がオフになります。推論をオフにできないモデル（gpt-6-astra、思考が必須の Claude、カタログで必須とされているルート）には、代わりに受け付けられる最低のレベルが使われます。また、実行時に無効化を拒否したルートは、次の音声ターンのために記憶されます。任意のレベルを設定するか、`""` でセッションの effort を使います。

これは、CLI と TUI のボイスモード、デスクトップの音声会話、メッセージングプラットフォームのボイスノートなど、すべての連鎖型の音声ターンに適用されます。音声ターンはフルツールセットを持ち、変わるのは回答するモデルだけです。次に入力したメッセージはセッションのモデルに戻り、ターン後のメモリとスキルのレビューも同様です。使用量は `voice_chat` タスクとして記録されるため、セッションは選択したモデルを自身のモデルとして保持します。

音声モデルが圧縮を強制することはありません: 会話がすでにそのコンテキストウィンドウより大きい場合、そのターンはセッションのモデルで実行され、一度だけその旨の通知が表示されます。GPT-Live ボイスチャットはこのスロットを無視します。そこでは音声レイヤー自体がすでに高速なモデルであり、実際の作業はセッションのモデルに委譲されるためです。

### 割り込み（バージイン）

エージェントのターン中、いつでも割り込めます — マイクは話し終えた瞬間から返答の再生が完全に終わるまで有効なままです（全二重）:

- **考え中に割り込む** — 連続ボイスモードでは、LLM の生成中（音声が再生される前）に話すと進行中のターンが中断され、割り込んだ発話が次のメッセージになります。実行中のターンに対して入力するのと同じです。
- **話をさえぎる** — エージェントの返答の再生中に話すと、話し始めた瞬間に再生が止まり、話した内容が送信されます。検出器はターン開始時の *静かな室内* に対してノイズフロアを調整するため（再生音そのものに対しては調整しません）、スピーカーからの音漏れで聞こえなくなることはなく、通常の発話で確実にトリガーされます。
- **入力するか録音キーを押す** — 新しいメッセージを送信するか、プッシュトゥトークのキーを押すと、すべてのサーフェスで即座に再生が止まります。
- **「stop」と話す** — 停止フレーズは両方のフェーズで機能します: 生成中はターンを中断し、かつボイスチャットを終了します。再生中は音声を止めてチャットを終了します。

調整（config.yaml）: `voice.barge_in: false` で無効になります。`voice.barge_in_threshold_multiplier`（デフォルト `3.0`）は、静かな室内のフロアに対する発話トリガーの倍率で、低いほど感度が上がります。デスクトップアプリは再生フェーズのトリガーにもこの倍率を適用するため、返答に割り込めない静かな Bluetooth ヘッドセットでは、例えば `1.5` を使えます。`voice.barge_in_grace_seconds`（デフォルト `0.5`）は、再生開始直後のトリガーを抑制します。`HERMES_VOICE_DEBUG=1` を設定すると、ライブ調整用にブロックごとの VAD 診断情報（調整されたフロア、RMS、トリガーの判定）が stderr にストリーミングされます。

エージェントは割り込まれたことを **認識** します: 次のメッセージには、音声での返答が途中で切られたことをモデルに伝える短い注記が含まれるため、気づかないままでいるのではなく、自然に反応したり（「失礼な！」）、中断したところから続けたりできます。

### ハルシネーションフィルター

Whisper は無音や背景ノイズから幻のテキスト（"Thank you for watching"、"Subscribe" など）を生成することがあります。エージェントは、複数言語にわたる 26 種類の既知のハルシネーションフレーズのセットと、繰り返しのバリエーションを捕捉する正規表現パターンを使ってこれらをフィルタリングします。

---

## ゲートウェイのボイス返信（Telegram & Discord）

メッセージングボットをまだセットアップしていない場合は、プラットフォーム別のガイドを参照してください:
- [Telegram セットアップガイド](../messaging/telegram.md)
- [Discord セットアップガイド](../messaging/discord.md)

メッセージングプラットフォームに接続するためにゲートウェイを起動します:

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
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
/voice          Toggle voice mode on/off
/voice on       Voice replies only when you send a voice message
/voice tts      Voice replies for ALL messages
/voice off      Disable voice replies
/voice status   Show current setting
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
| テキストのみ | `309237763136` | チャンネル表示、メッセージ送信、履歴読み取り、埋め込み、添付、スレッド、リアクション、公開スレッドの作成 |
| テキスト + 音声 | `309240908864` | 上記すべて + Connect、Speak |

更新後の権限 URL で **ボットを再招待** します:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=309240908864
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
# macOS (Homebrew)
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

# Discord bot (already configured for text)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# PM prepares local Faster-Whisper on supported targets; no STT API key is needed.
# GROQ_API_KEY=your-key            # Alternative: cloud-based, fast, free tier

# TTS — optional. Edge TTS and NeuTTS need no key.
# ELEVENLABS_API_KEY=***      # Premium quality
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### ゲートウェイの起動

```bash
hermes gateway        # Start with existing configuration
```

ボットは数秒以内に Discord でオンラインになるはずです。

### コマンド

ボットがいる Discord のテキストチャンネルで以下を使います:

```
/voice join      Bot joins your current voice channel
/voice channel   Alias for /voice join
/voice leave     Bot disconnects from voice channel
/voice status    Show voice mode and connected channel
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
- 別のテキストチャンネルから `/voice join` を実行すると、バインドがそちらに移ります。移動前に取得された発話は、文字起こし中かまだ終わっていないかにかかわらず破棄され、新しいチャンネルには投稿されません

### エコー防止

ボットは TTS 返答を再生している間、オーディオリスナーを自動的に一時停止し、自分自身の出力を聞き取って再処理することを防ぎます。

### アクセス制御

`DISCORD_ALLOWED_USERS` または `DISCORD_ALLOWED_ROLES` で許可されたユーザーのみが音声でやり取りできます。ロールは、話者が話すたびにチェックされます。それ以外のユーザーの音声は静かに無視されます。

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## 設定リファレンス

### config.yaml

```yaml
# Voice recording (CLI)
voice:
  record_key: "ctrl+b"            # Key to start/stop recording
  max_recording_seconds: 120       # Maximum recording length
  auto_tts: false                  # Auto-enable TTS when voice mode starts
  beep_enabled: true               # Play record start/stop beeps
  silence_threshold: 200           # RMS level (0-32767) below which counts as silence
  silence_duration: 3.0            # Seconds of silence before auto-stop
  stop_phrases: ["stop"]           # Saying exactly one of these ends the voice chat; [] disables

# Speech-to-Text
stt:
  enabled: true                     # set to false to skip auto-transcription —
                                    # the gateway still caches the audio file and
                                    # passes its path to the agent as part of the
                                    # inbound message, useful for custom pipelines
                                    # (diarization, alignment, archival, etc.).
                                    # When enabled, the agent gets the quoted
                                    # transcript followed by a
                                    # "[User sent audio: <path>]" line, so the
                                    # recording stays reachable after STT succeeds
  provider: "local"                  # "local" (free) | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra"
  language: "en"                     # global hint for every provider; set "" to auto-detect
  local:
    model: "base"                    # tiny, base, small, medium, large-v3, turbo
    language: ""                     # optional ISO-639-1 hint; blank = stt.language, then HERMES_LOCAL_STT_LANGUAGE, then auto-detect
  groq:
    language: ""                     # optional ISO-639-1 hint; blank = stt.language, then HERMES_LOCAL_STT_LANGUAGE, then auto-detect
  # model: "whisper-1"              # Legacy: used when provider is not set

# Text-to-Speech
tts:
  provider: "edge"                 # "edge" (free) | "elevenlabs" | "openai" | "neutts" | "minimax" | "mistral" | "gemini" | "xai" | "kittentts" | "piper"
  edge:
    voice: "en-US-AriaNeural"      # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"   # or eleven_v3, eleven_flash_v2_5, ... (Desktop Settings → Voice accepts any model id)
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # optional: override for self-hosted or OpenAI-compatible endpoints
    # The `text_to_speech` tool accepts an optional per-call `instructions`
    # argument (tone, emotion, pacing, accent, whispering) that is forwarded
    # to `gpt-4o-mini-tts` and to OpenAI-compatible voice-design servers
    # (e.g. Qwen3-TTS-VoiceDesign via oMLX). See OpenAI's voice-design guide:
    # https://platform.openai.com/docs/guides/text-to-speech
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

プロバイダーの言語設定が空の場合は、環境変数を参照する前にグローバルの `stt.language` の値（デフォルト `"en"`）を継承します。自動検出の条件とコマンドバックエンドの制限については、[STT の言語解決](../configuration.md#speech-to-text-stt)を参照してください。

### 環境変数

```bash
# Speech-to-Text providers (local needs no key)
# PM prepares local Faster-Whisper on supported targets; no STT API key is needed.
GROQ_API_KEY=...                    # Groq Whisper (fast, free tier)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (paid)

# STT advanced overrides (optional)
STT_GROQ_MODEL=whisper-large-v3-turbo    # Override default Groq STT model
STT_OPENAI_MODEL=whisper-1               # Override default OpenAI STT model
GROQ_BASE_URL=https://api.groq.com/openai/v1     # Custom Groq endpoint
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # Custom OpenAI STT endpoint

# Text-to-Speech providers (Edge TTS and NeuTTS need no key)
ELEVENLABS_API_KEY=***             # ElevenLabs (premium quality)
# VOICE_TOOLS_OPENAI_KEY above also enables OpenAI TTS

# Discord voice channel
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
| **OpenAI** | `gpt-transcribe` | 高速 | 最良 | 有料（$0.0045/分） | 必要 |
| **Mistral** | `voxtral-mini-latest` | 高速 | 良好 | 有料 | 必要 |
| **xAI** | `grok-voice-transcribe-2.0` | 高速 | 最良 | 有料（バッチで $0.10/時間） | 必要 |

プロバイダーの優先順位（自動フォールバック）: **local** > **groq** > **openai**

### 長時間の録音とアップロードの制限 {#long-recordings-and-upload-limits}

クラウドプロバイダーは 1 回のリクエストに上限を設けています: OpenAI と Groq は 25 MB、Mistral は 500 MB（60 分）、xAI は 500 MB、ElevenLabs は 5 GB 弱まで受け付けます。一部の OpenAI モデルには、リクエストあたりの実質的な長さの制限もあります: `gpt-4o-transcribe` と `gpt-4o-mini-transcribe` は約 7.5 分（出力の上限が 2,000 トークンのため）、`whisper-1` は 10 分（各リクエストがデフォルトの 60 秒のタイムアウト内に終わるようにするため）です。ボイスノート、音声の添付ファイル、ボイスモードの録音がアクティブなプロバイダーの制限を超えている場合、Hermes はまずコンパクトな 16 kHz モノラル AAC に再エンコードし、ほとんどの録音を 1 回のリクエストに収めます。それでも大きすぎる場合は、Hermes が無音の箇所で分割し、各部分を順番に文字起こししてテキストを結合します。ローカルプロバイダーにはアップロードの制限がなく、分割されることはありません。どちらのステップにも ffmpeg が必要です。

### TTS プロバイダー比較

| プロバイダー | 品質 | コスト | レイテンシ | キー必須 |
|----------|---------|------|---------|-------------|
| **Edge TTS** | 良好 | 無料 | 約 1 秒 | 不要 |
| **ElevenLabs** | 非常に良い | 有料 | 約 2 秒 | 必要 |
| **OpenAI TTS** | 良好 | 有料 | 約 1.5 秒 | 必要 |
| **NeuTTS** | 良好 | 無料 | CPU/GPU に依存 | 不要 |

NeuTTS は上記の `tts.neutts` 設定ブロックを使用します。

`openai` では、`text_to_speech` ツールがオプションの `instructions` 引数を受け付け、`gpt-4o-mini-tts` の voice-design 機能（口調、感情、ペース、アクセント、ささやき声）を利用できます。同じフィールドは、`tts.openai.base_url` で指定した OpenAI 互換の voice-design サーバー（例: oMLX 経由の Qwen3-TTS-VoiceDesign）にもルーティングされます。

---

## トラブルシューティング

### 「No audio device found」（CLI）

PortAudio がインストールされていません:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

Linux デスクトップ上の Docker 内で Hermes を実行している場合、コンテナはホストのオーディオソケットにもアクセスする必要があります。PulseAudio/PipeWire 互換のセットアップについては、[Docker オーディオブリッジ](../docker.md#optional-linux-desktop-audio-bridge)の注記を参照してください。

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
