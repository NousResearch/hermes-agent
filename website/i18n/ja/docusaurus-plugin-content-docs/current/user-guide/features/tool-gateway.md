---
title: "Nous Tool Gateway"
description: "1 つのサブスクリプションで、すべてのツールを。Web 検索、画像生成、TTS、クラウドブラウザ — すべて Nous Portal 経由でルーティングされ、追加の API キーは不要。"
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway

**1 つのサブスクリプション。すべてのツールが組み込み。**

Tool Gateway は、有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションすべてに含まれています。Hermes のツール呼び出し（Web 検索、画像生成、テキスト読み上げ、クラウドブラウザ自動化）を、Nous がすでに運用しているインフラ経由でルーティングします。そのため、エージェントを役立たせるためだけに Web 検索ベンダー、FAL、OpenAI、Browser Use などへ個別に登録する必要がありません。

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', margin: '1.5rem 0'}}>
  <a href="https://portal.nousresearch.com/manage-subscription" style={{background: 'var(--ifm-color-primary)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '6px', textDecoration: 'none', fontWeight: 'bold'}}>サブスクリプションを開始・管理 →</a>
</div>

## 含まれるもの

| | ツール | できること |
|---|---|---|
| 🔍 | **Web 検索 & 抽出** | Nous が管理する、エージェントグレードな Web 検索とフルページ抽出。レート制限を気にする必要はありません。スケーリングはゲートウェイが処理します。 |
| 🎨 | **画像生成** | FAL カタログ、ネイティブの Krea モデル、Nous Portal の画像モデルにまたがる 1 つの **Nous Subscription** 行: **Krea 3**、**FLUX 2**、**GPT Image**、**Nano Banana**、**Seedream**、**Ideogram**、**Recraft**、**Qwen**、**Krea 2** など。モデルは `hermes tools` で一度選ぶだけです。そこには最新のリストが表示されます。 |
| 🔊 | **テキスト読み上げ** | `text_to_speech` ツールに組み込まれた OpenAI TTS の音声。Telegram にボイスノートを送り、パイプライン用の音声を生成し、あらゆるものをナレーションできます。 |
| 🌐 | **クラウドブラウザ自動化** | Browser Use 経由のヘッドレスなクラウド Chromium セッション。CDP 上で `browser_exec` によって駆動され、Browserbase アカウントは不要です。 |

4 つすべてが、Nous サブスクリプションに対して従量課金されます。任意の組み合わせで使えます。Web と画像にはゲートウェイを使いつつ、TTS には自分の ElevenLabs キーを使い続けることも、すべてを Nous 経由でルーティングすることもできます。

## なぜこれがあるのか

実際に*物事をこなせる*エージェントを作るには、5 つ以上の API サブスクリプションをつなぎ合わせる必要があります。それぞれに独自の登録、レート制限、課金、クセがあります。ゲートウェイはそれを 1 つのアカウントにまとめます:

- **1 つの請求。** Nous に支払えば、残りは私たちが処理します。
- **1 つの登録。** 管理すべき Web 検索、FAL、Browser Use、OpenAI audio のアカウントはありません。
- **1 つのキー。** あなたの Nous Portal OAuth がすべてのツールをカバーします。
- **同じ品質。** ダイレクトキー経路が使うのと同じバックエンド。私たちが前面に立つだけです。

いつでも自分のキーを持ち込めます。ツール単位で、好きなときに。ゲートウェイはロックインではなく、ショートカットです。

## はじめる

入り口は 3 つあります。今の状況に合うものを選んでください:

```bash
hermes setup --portal     # Fresh install: Nous OAuth + set Nous as provider + turn on the Tool Gateway in one go
```

```bash
hermes model              # Switch your inference provider to Nous Portal — Hermes then offers to turn on the gateway for all tools
```

```bash
hermes tools              # Enable the gateway per-tool — pick "Nous Subscription" for any tool you want
```

`hermes setup --portal` と `hermes model` は一括の経路です: 一度ログインし、必要ならすべてのツールをゲートウェイに切り替えます。`hermes tools` はアラカルトの経路で、必要なツールだけを 1 つずつ有効にします。

**先にログインする必要はありません。** `hermes tools` では、Nous が管理するバックエンド（Web 検索、画像、動画、TTS、ブラウザ）は、Nous Portal に一度もサインインしたことがなくても常に一覧に表示されます。いずれかを選ぶと、まだ認証していなければ Hermes がその場で Portal ログインを実行します。事前に `hermes model` を実行する必要はありません。Nous OAuth がすでに有効であれば、バックエンドを選ぶとすぐに有効になり、追加のプロンプトはありません。この経路はログインと選んだ 1 つのツールの有効化だけを行います — 推論プロバイダーを切り替えることは**なく**、他のすべてのツールでゲートウェイを有効にするよう求めることも**ありません**。

何が有効かはいつでも確認できます:

```bash
hermes portal info        # Portal auth + Tool Gateway routing summary
hermes portal tools       # Gateway catalog with current routing per tool
hermes status --full      # Full system status (Tool Gateway is one section)
```

`hermes portal info` では次のようなセクションが表示されます:

```
◆ Nous Tool Gateway
  Nous Portal     ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
```

「active via Nous subscription」と表示されたツールはゲートウェイを経由しています。それ以外は自分のキーを使っています。

## 利用資格

Tool Gateway は**有料サブスクリプション**機能です。ゲートウェイを利用するには [プランをアップグレード](https://portal.nousresearch.com/manage-subscription)してください。

一部のアカウントには **無料ツールプール** も付与されます。これは、有料サブスクリプションなしでゲートウェイのツール呼び出しをカバーする、少量のマネージドツール枠です。無料プールが利用可能な場合、ゲートウェイはそれを提示し、初回使用時にセットアップのプロンプトを表示するため、オプトインしてすぐにマネージドツールを使い始められます。

## 有効化チェックリスト

Nous のモデルを選ぶ（`hermes model`）と、ゲートウェイのバックエンドのツールごとのチェックリストが提示されます。その挙動は既存のセットアップを尊重します:

- 明示的に別のバックエンドを指定したツール（例: `web.backend: searxng`、`browser.cloud_provider: browserbase`）は**提示されません** — あなたの選択が誤って上書きされることはありません。
- 環境変数だけで設定されたツール（例: `SEARXNG_URL`、`BROWSER_USE_API_KEY`）は、自分のバックエンドを維持する旨のラベル付きで**チェックなし**で提示されます。
- 本当に未設定のツールだけが事前にチェックされた状態になります。
- 辞退は維持されます: ツールのチェックを外したままチェックリストを送信すると、以後の Nous モデルへの切り替えでは事前にチェックされません（`config.yaml` の `tool_gateway_declined_tools` に保存されます。後でチェックすると辞退はクリアされます）。

## 組み合わせて使う

ゲートウェイはツール単位です。必要なものだけ有効にできます:

- **すべてのツールを Nous 経由で** — 最も簡単。1 つのサブスクリプションで完了。
- **Web + 画像はゲートウェイ、TTS は自前** — ElevenLabs の音声を維持し、残りは Nous に任せる。
- **キーを持っていないものだけゲートウェイで** — 「Browserbase にはすでに支払っているが、Web 検索のアカウントは作りたくない」も問題なく機能します。

任意のツールはいつでも次のコマンドで切り替えられます:

```bash
hermes tools          # Interactive picker for each tool category
```

ツールを選択し、プロバイダーとして **Nous Subscription**（または好みの任意のダイレクトプロバイダー）を選びます。設定の編集は不要です。まだ Nous Portal にログインしていない場合、**Nous Subscription** を選ぶとその場で Portal ログインが始まります — 先に `hermes model` で認証する必要はありません。

## 個別の画像モデルを使う

モデルは `hermes tools` → Image Generation で一度選び、`config.yaml` に `image_gen.model` として保存されます。すべての `image_generate` 呼び出しはその保存されたモデルを使います — このツールにはモデルのパラメータがないため、呼び出しごとの上書きはありません。未設定のままにすると、有料サブスクリプションでは Krea 3、無料ツールプールでは FLUX 2 Klein 9B が使われます。

**Nous Subscription** 行には、すべてのモデルを 1 回ずつ列挙する 1 つのモデルピッカーがあります。保存された ID によって、どのゲートウェイがリクエストを処理するかが決まります: ネイティブの Krea の ID（`krea-3`、`krea-2-medium`、`krea-2-large`、`krea-2-medium-turbo`）は Krea ゲートウェイへ、FAL カタログの ID は FAL へ、それ以外の ID は Nous Portal へ送られます。選択内容は依然として `image_gen.provider: nous` とモデル ID だけです。

**Krea 3**（生成と既存画像の編集 — 最大 4 枚の参照画像、`thinking` と `resolution` の制御）、**Krea 2**（Medium、Large、Medium Turbo — 最大 10 枚のスタイル参照画像、任意の Enhance アップスケール）、および Nous Portal の画像モデルは、それぞれ専用の行ではなく、同じピッカーの中にあります。これらには有料サブスクリプションが必要です — 無料ツールプールがカバーするのは FAL のモデルだけです。FAL 経由の Krea の ID（`fal-ai/krea/v2/...`）は、ダイレクトな FAL.ai の行に残ります。Krea ゲートウェイがジョブの開始前にリクエストを拒否した場合（到達不能、またはゲートウェイ自身のリトライ後の 429）、Hermes は FAL の既定モデルでそれを再実行し、結果の `fallback_from` にその旨が示されます。これを無効にするには `image_gen.krea.fallback_to_fal: false` を設定します。ソース画像を伴うリクエストはフォールバックしません。

モデル ID、速度、価格は [画像生成](./image-generation.md#supported-models) ページにあります。セットは進化します。`hermes tools` → Image Generation で現在の最新リストが表示されます。

## 複数のコネクタアカウント {#multiple-connector-accounts}

コネクタは、エージェントが Nous のツールゲートウェイ経由で利用する Gmail などのホスト型アプリです（[Tool Search → Connectors](/docs/user-guide/features/tool-search#connectors-remote-tools) を参照）。1 つのアプリに 2 つ以上のアカウントを接続できます。たとえば Gmail の `home` と Gmail の `work` です。

- **名前。** 各アカウントには 1〜32 文字の名前があります: 小文字の英字、数字、`-` で構成され、英字または数字で始まります。名前のないアカウントには、アプリ自身のアカウントラベルが表示されます。
- **デフォルトなし。** デフォルトのアカウントはありません。名前付きアカウントが複数あるアプリを使うようエージェントに頼むときは、どれを使うか指定する必要があります。
- **新しいアカウントにはエージェントが名前を付けます。** あなたの発言から名前を選び、確信が持てなければ尋ねます。その後、選んだ名前を伝えます。たとえば「Hermes named this Gmail account work. Rename it anytime.」です。デスクトップアプリ、TUI、クラシック CLI にその行が表示されます。
- **名前の変更。** アカウントの名前はいつでも変更でき、エージェントに頼むこともできます。名前の変更には、それをサポートする Nous Portal のバージョンが必要です。それまでは「This Nous Portal cannot rename connector accounts yet.」と表示されます。
- **再接続しても何も削除されません。** 壊れたアカウントを再接続すると、同じ名前の新しいアカウントに置き換えられます。古いものは退役（無効）として保持されます。退役したアカウントは一覧に残りますが、折りたたまれます。
- **削除はあなたが行います。** アカウントの削除はあなた自身が 1 つずつ行います。エージェントがアカウントを削除することはありません。

コネクタに 2 つ以上の有効なアカウントがあると、そのコネクタのすべてのツールが `connector_alias` フィールドを受け取り、その許容値はあなたのアカウント名になります。このフィールドは、すべての有効なアカウントに名前がある場合は必須で、一部に名前がない場合は任意です。

複数のアカウントに名前があるのにエージェントがこれを省略した場合、または存在しない名前を指定した場合、呼び出しは失敗し、エラーに有効な名前が列挙されます。結果には、名前付きアカウントで実行された場合はその `account` が示されます。名前のないアカウントの場合、このフィールドはありません。

アカウントは次のいずれかの場所で管理できます:

- デスクトップアプリの **Connectors** ページ。
- TUI: `/connectors`（[スラッシュコマンド](../../reference/slash-commands.md#tools--skills)を参照）。
- CLI: `hermes connectors`（[CLI コマンド](../../reference/cli-commands.md#hermes-connectors)を参照）。
- [Nous Portal](https://portal.nousresearch.com)。

---

## 設定リファレンス

ほとんどのユーザーはここに触れる必要はありません。`hermes model` と `hermes tools` があらゆるワークフローを対話的にカバーします。このセクションは、config.yaml を直接書いたり、セットアップをスクリプト化したりする場合向けです。

### ツールカテゴリごとに 1 つの選択キー

各ツールカテゴリには、`hermes tools` のピッカー（またはデスクトップ GUI）が書き込む、プロバイダー選択用のキーが 1 つだけあります。**Nous Subscription** 行を選ぶと値 `nous` が保存され、そのカテゴリはマネージドな Tool Gateway 経由でルーティングされます。BYOK の行を選ぶとベンダー名（`fal`、`openai`、`firecrawl`、`browser-use` など）が保存され、自分の認証情報で直接接続します:

```yaml
web:
  backend: nous          # web search/extract via the Tool Gateway

image_gen:
  provider: nous         # image generation via the Tool Gateway

tts:
  provider: nous         # TTS via the Tool Gateway

stt:
  provider: nous         # speech-to-text via the Tool Gateway

browser:
  cloud_provider: nous   # cloud browser via the Tool Gateway
```

ランタイムは**常に保存された選択を使います** — 認証情報の有無によってカテゴリが選択されたり経路が変わったりすることはありません。`image_gen.provider: nous` の間は、`.env` にある `FAL_KEY` は無視されます。逆に、`FAL_KEY` が設定されていない状態で `image_gen.provider: fal` にすると、黙ってゲートウェイにフォールバックするのではなく、明確なエラーになります:

```
image_gen is configured to use fal (set via hermes tools), but FAL_KEY is not set. Run 'hermes tools' to change it.
```

**一度も設定したことのない**カテゴリ（選択キーが一度も書き込まれていない）は、これまでどおり利用可能な認証情報から自動検出されます。しかし一度選択が存在すれば、`.env` にキーを追加しても経路は変わりません — 変えられるのは `hermes tools`（または選択キーの編集）だけです。

### 自分のキーに戻す

```bash
hermes tools    # pick the tool → choose a direct provider (e.g. Firecrawl)
```

または選択キーを直接設定します:

```yaml
web:
  backend: firecrawl   # Hermes now uses FIRECRAWL_API_KEY from .env
```

### 自分のキーとゲートウェイを混在させる（Web のみ） {#mixing-your-own-key-and-the-gateway-web-only}

Web 検索と Web 抽出は、それぞれ独自の経路を選べます。デスクトップアプリで **Capabilities → Tools → Web Search & Scraping** を開きます: **Nous Subscription** 行の **Use for Search** / **Use for Extract** はその機能をゲートウェイ経由にし、ダイレクトな行（例: **Firecrawl**）の同じボタンは自分のキー経由にします。上部の **Search:** / **Extract:** のピルには、それぞれが現在どの経路を使っているかが表示されます。`config.yaml` では、マネージドの選択は機能ごとのキーの値 `nous` です:

```yaml
web:
  search_backend: firecrawl   # your FIRECRAWL_API_KEY
  extract_backend: nous       # the Nous Tool Gateway (managed Firecrawl)
```

ツール全体で **Nous Subscription** を選ぶ（または `hermes tools` を実行する）と、機能ごとのキーは両方ともクリアされ、検索と抽出の両方がゲートウェイ経由に戻ります。

### レガシーの `use_gateway` フラグ（非推奨）

古いバージョンの Hermes では、ゲートウェイ経由でルーティングするためにツールごとの `use_gateway: true` ブール値を使っていました。このフラグは**レガシー**です: もう書き込まれることはなく、`hermes tools` のピッカーは選択を書き換えるときにカテゴリの設定からこれを削除します。`use_gateway: true` をまだ含む古い設定は、読み込み時に `nous` の選択として解釈されるため、既存のセットアップは引き続き動作します。新しい設定では `use_gateway` を設定せず、代わりに `hermes tools` でプロバイダーを選んでください。

### セルフホスト型ゲートウェイ（上級者向け）

独自の Nous 互換ゲートウェイを運用していますか？ `~/.hermes/.env` でエンドポイントを上書きします:

```bash
TOOL_GATEWAY_DOMAIN=your-domain.example.com
TOOL_GATEWAY_SCHEME=https
TOOL_GATEWAY_USER_TOKEN=your-token        # normally auto-populated from Portal login
FIRECRAWL_GATEWAY_URL=https://...         # override one endpoint specifically
TOOL_GATEWAY_URL=http://127.0.0.1:3009    # pin the shared managed origin exactly
CONNECTOR_GATEWAY_URL=http://127.0.0.1:3009 # pin the connectors origin exactly
```

すべてのホストは `{label}-gateway.<domain>` という名前で、`TOOL_GATEWAY_DOMAIN` / `TOOL_GATEWAY_SCHEME` はそれら**すべて**を組み替えます。`{LABEL}_GATEWAY_URL` は 1 つのホストを正確に固定し、導出をスキップします:

- `{vendor}-gateway.<domain>` — ベンダーごとのパススルー（Firecrawl、BFL など）。
- `tool-gateway.<domain>` — 共有のマネージドオリジン: ゲートウェイ自体でホストされるベンダーとメディアのアップロード。
- `connector-gateway.<domain>` — コネクタ API（`/v1/connectors/*`）。独立したデプロイメントです。[Tool Search → Connectors](/docs/user-guide/features/tool-search#connectors-remote-tools) を参照してください。

これらのつまみは、カスタムインフラ構成（エンタープライズデプロイ、開発環境）向けに存在します。通常のサブスクライバーが設定することはありません。

## FAQ

### Telegram / Discord / その他のメッセージングゲートウェイで動作しますか？

はい。Tool Gateway は CLI ではなく、ツール実行レイヤーで動作します。ツールを呼び出せるすべてのインターフェース（CLI、Telegram、Discord、Slack、IRC、Teams、API サーバーなど）が、透過的にその恩恵を受けます。

### サブスクリプションが失効したらどうなりますか？

ゲートウェイ経由でルーティングされたツールは、更新するか `hermes tools` でダイレクト API キーに差し替えるまで動作を停止します。Hermes は Portal を指す明確なエラーを表示します。

### ツールごとの使用量やコストを確認できますか？

はい。[Nous Portal ダッシュボード](https://portal.nousresearch.com)が使用量をツールごとに分解するので、何が請求を押し上げているか分かります。

### Modal（サーバーレスターミナル）は含まれますか？

Modal は、既定の Tool Gateway バンドルの一部ではなく、Nous サブスクリプションを通じた**オプションのアドオン**として利用できます。シェル実行用のリモートサンドボックスが必要なときは、`hermes setup terminal` 経由、または `config.yaml` で直接設定してください。

### ゲートウェイを有効にするとき、既存の API キーを削除する必要がありますか？

いいえ。`.env` に残しておいてください。ツールの選択が **Nous Subscription** の間は、そのツールのダイレクトキーは単に無視されます。`hermes tools` で再びダイレクトプロバイダーを選べば、あなたのキーが再びソースになります。ゲートウェイはロックインではありません。
