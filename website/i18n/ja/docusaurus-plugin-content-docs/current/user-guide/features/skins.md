---
sidebar_position: 10
title: "スキンとテーマ"
description: "組み込みおよびユーザー定義のスキンでHermes CLIをカスタマイズする"
---

# スキンとテーマ

スキンはHermes CLIの**視覚的な表示**を制御します。バナーの色、スピナーの顔と動詞、応答ボックスのラベル、ブランディングのテキスト、ツールアクティビティのプレフィックスなどです。

会話のスタイルと視覚的なスタイルは別々の概念です。

- **パーソナリティ**はエージェントのトーンと言い回しを変えます。
- **スキン**はCLIの外観を変えます。

## スキンを変更する

```bash
/skin                # 現在のスキンを表示し、利用可能なスキンを一覧表示する
/skin ares           # 組み込みスキンに切り替える
/skin mytheme        # ~/.hermes/skins/mytheme.yaml のカスタムスキンに切り替える
```

または `~/.hermes/config.yaml` でデフォルトのスキンを設定します。

```yaml
display:
  skin: default
```

## 組み込みスキン

| スキン | 説明 | エージェントブランディング | 視覚的な特徴 |
|------|-------------|----------------|------------------|
| `default` | クラシックなHermes — ゴールドとかわいさ | `Hermes Agent` | 温かみのあるゴールドの枠線、コーンシルク色のテキスト、スピナーのかわいい顔。おなじみのカドゥケウスのバナー。クリーンで親しみやすい。 |
| `ares` | 軍神テーマ — クリムゾンとブロンズ | `Ares Agent` | ブロンズのアクセントを伴う深いクリムゾンの枠線。攻撃的なスピナーの動詞（"forging"、"marching"、"tempering steel"）。剣と盾のカスタムASCIIアートバナー。 |
| `mono` | モノクローム — クリーンなグレースケール | `Hermes Agent` | すべてグレー — 色なし。枠線は `#555555`、テキストは `#c9d1d9`。ミニマルなターミナル構成や画面録画に最適。 |
| `slate` | クールブルー — 開発者向け | `Hermes Agent` | ロイヤルブルーの枠線（`#4169e1`）、柔らかなブルーのテキスト。落ち着いていてプロフェッショナル。カスタムスピナーはなく、デフォルトの顔を使用。 |
| `daylight` | 明るいターミナル向けのライトテーマ、ダークなテキストとクールブルーのアクセント | `Hermes Agent` | 白や明るいターミナル向けに設計。ブルーの枠線を伴うダークスレートのテキスト、淡いステータス面、ライトなターミナルプロファイルでも読みやすいライトな補完メニュー。 |
| `warm-lightmode` | ライトなターミナル背景向けの温かいブラウン/ゴールドのテキスト | `Hermes Agent` | ライトなターミナル向けの温かい羊皮紙のトーン。サドルブラウンのアクセントを伴うダークブラウンのテキスト、クリーム色のステータス面。よりクールな daylight テーマに代わる素朴な選択肢。 |
| `poseidon` | 海神テーマ — 深いブルーとシーフォーム | `Poseidon Agent` | 深いブルーからシーフォームへのグラデーション。海をテーマにしたスピナー（"charting currents"、"sounding the depth"）。三叉槍のASCIIアートバナー。 |
| `sisyphus` | シーシュポステーマ — 厳粛なグレースケールと粘り強さ | `Sisyphus Agent` | くっきりとしたコントラストのライトグレー。岩をテーマにしたスピナー（"pushing uphill"、"resetting the boulder"、"enduring the loop"）。岩と丘のASCIIアートバナー。 |
| `charizard` | 火山テーマ — バーントオレンジと残り火 | `Charizard Agent` | 温かいバーントオレンジから残り火へのグラデーション。火をテーマにしたスピナー（"banking into the draft"、"measuring burn"）。ドラゴンのシルエットのASCIIアートバナー。 |

## 設定可能なキーの完全な一覧

### 色（`colors:`）

CLI全体のすべての色の値を制御します。値は16進数の色文字列です。

| キー | 説明 | デフォルト（`default` スキン） |
|-----|-------------|--------------------------|
| `banner_border` | 起動バナーを囲むパネルの枠線 | `#CD7F32`（ブロンズ） |
| `banner_title` | バナーのタイトルテキストの色 | `#FFD700`（ゴールド） |
| `banner_accent` | バナーのセクション見出し（Available Tools など） | `#FFBF00`（アンバー） |
| `banner_dim` | バナーの控えめなテキスト（区切り、補助ラベル） | `#B8860B`（ダークゴールデンロッド） |
| `banner_text` | バナーの本文テキスト（ツール名、スキル名） | `#FFF8DC`（コーンシルク） |
| `ui_accent` | 一般的なUIのアクセント色（ハイライト、アクティブ要素） | `#FFBF00` |
| `ui_label` | UIのラベルとタグ | `#4dd0e1`（ティール） |
| `ui_ok` | 成功インジケーター（チェックマーク、完了） | `#4caf50`（グリーン） |
| `ui_error` | エラーインジケーター（失敗、ブロック） | `#ef5350`（レッド） |
| `ui_warn` | 警告インジケーター（注意、承認プロンプト） | `#ffa726`（オレンジ） |
| `prompt` | 対話プロンプトのテキストの色 | `#FFF8DC` |
| `input_rule` | 入力エリアの上の水平線 | `#CD7F32` |
| `response_border` | エージェントの応答ボックスを囲む枠線（ANSIエスケープ） | `#FFD700` |
| `session_label` | セッションラベルの色 | `#DAA520` |
| `session_border` | セッションIDの控えめな枠線の色 | `#8B8682` |
| `status_bar_bg` | TUIのステータス/使用量バーの背景色 | `#1a1a2e` |
| `voice_status_bg` | 音声モードのステータスバッジの背景色 | `#1a1a2e` |
| `selection_bg` | TUIのマウス選択ハイライターの背景色。未設定の場合は `completion_menu_current_bg` にフォールバックします。 | `#333355` |
| `completion_menu_bg` | 補完メニューリストの背景色 | `#1a1a2e` |
| `completion_menu_current_bg` | アクティブな補完行の背景色 | `#333355` |
| `completion_menu_meta_bg` | 補完メタ列の背景色 | `#1a1a2e` |
| `completion_menu_meta_current_bg` | アクティブな補完メタ列の背景色 | `#333355` |

### スピナー（`spinner:`）

API応答を待つ間に表示されるアニメーションスピナーを制御します。

| キー | 型 | 説明 | 例 |
|-----|------|-------------|---------|
| `waiting_faces` | 文字列のリスト | API応答を待つ間に循環する顔 | `["(⚔)", "(⛨)", "(▲)"]` |
| `thinking_faces` | 文字列のリスト | モデルの推論中に循環する顔 | `["(⚔)", "(⌁)", "(<>)"]` |
| `thinking_verbs` | 文字列のリスト | スピナーのメッセージに表示される動詞 | `["forging", "plotting", "hammering plans"]` |
| `wings` | [左, 右] ペアのリスト | スピナーを囲む装飾的なブラケット | `[["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"]]` |

スピナーの値が空の場合（`default` や `mono` のように）、`display.py` のハードコードされたデフォルトが使用されます。

### ブランディング（`branding:`）

CLIインターフェース全体で使用されるテキスト文字列。

| キー | 説明 | デフォルト |
|-----|-------------|---------|
| `agent_name` | バナーのタイトルとステータス表示に表示される名前 | `Hermes Agent` |
| `welcome` | CLI起動時に表示されるウェルカムメッセージ | `Welcome to Hermes Agent! Type your message or /help for commands.` |
| `goodbye` | 終了時に表示されるメッセージ | `Goodbye! ⚕` |
| `response_label` | 応答ボックスのヘッダーのラベル | ` ⚕ Hermes ` |
| `prompt_symbol` | ユーザー入力プロンプトの前の記号（裸のトークン、レンダラーが末尾にスペースを追加） | `❯` |
| `help_header` | `/help` コマンド出力のヘッダーテキスト | `(^_^)? Available Commands` |

### その他のトップレベルキー

| キー | 型 | 説明 | デフォルト |
|-----|------|-------------|---------|
| `tool_prefix` | string | CLIのツール出力行の先頭に付ける文字 | `┊` |
| `tool_emojis` | dict | スピナーと進捗用のツールごとの絵文字の上書き（`{tool_name: emoji}`） | `{}` |
| `banner_logo` | string | Richマークアップ形式のASCIIアートロゴ（デフォルトの HERMES_AGENT バナーを置き換える） | `""` |
| `banner_hero` | string | Richマークアップ形式のヒーローアート（デフォルトのカドゥケウスのアートを置き換える） | `""` |

## カスタムスキン

`~/.hermes/skins/` 配下にYAMLファイルを作成します。ユーザースキンは、不足している値を組み込みの `default` スキンから継承するため、変更したいキーだけを指定すれば十分です。

### カスタムスキンの完全なYAMLテンプレート

```yaml
# ~/.hermes/skins/mytheme.yaml
# 完全なスキンテンプレート — すべてのキーを表示。不要なものは削除してよい。
# 不足している値は 'default' スキンから自動的に継承される。

name: mytheme
description: My custom theme

colors:
  banner_border: "#CD7F32"
  banner_title: "#FFD700"
  banner_accent: "#FFBF00"
  banner_dim: "#B8860B"
  banner_text: "#FFF8DC"
  ui_accent: "#FFBF00"
  ui_label: "#4dd0e1"
  ui_ok: "#4caf50"
  ui_error: "#ef5350"
  ui_warn: "#ffa726"
  prompt: "#FFF8DC"
  input_rule: "#CD7F32"
  response_border: "#FFD700"
  session_label: "#DAA520"
  session_border: "#8B8682"
  status_bar_bg: "#1a1a2e"
  voice_status_bg: "#1a1a2e"
  selection_bg: "#333355"
  completion_menu_bg: "#1a1a2e"
  completion_menu_current_bg: "#333355"
  completion_menu_meta_bg: "#1a1a2e"
  completion_menu_meta_current_bg: "#333355"

spinner:
  waiting_faces:
    - "(⚔)"
    - "(⛨)"
    - "(▲)"
  thinking_faces:
    - "(⚔)"
    - "(⌁)"
    - "(<>)"
  thinking_verbs:
    - "processing"
    - "analyzing"
    - "computing"
    - "evaluating"
  wings:
    - ["⟪⚡", "⚡⟫"]
    - ["⟪●", "●⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome to My Agent! Type your message or /help for commands."
  goodbye: "See you later! ⚡"
  response_label: " ⚡ My Agent "
  prompt_symbol: "⚡"
  help_header: "(⚡) Available Commands"

tool_prefix: "┊"

# ツールごとの絵文字の上書き（任意）
tool_emojis:
  terminal: "⚔"
  web_search: "🔮"
  read_file: "📄"

# カスタムASCIIアートバナー（任意、Richマークアップ対応）
# banner_logo: |
#   [bold #FFD700] MY AGENT [/]
# banner_hero: |
#   [#FFD700]  Custom art here  [/]
```

### 最小限のカスタムスキンの例

すべてが `default` から継承されるため、最小限のスキンでは異なる部分だけを変更すれば済みます。

```yaml
name: cyberpunk
description: Neon terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

## Hermes Mod — ビジュアルスキンエディター

[Hermes Mod](https://github.com/cocktailpeanut/hermes-mod) は、スキンを視覚的に作成・管理するためのコミュニティ製のWeb UIです。YAMLを手で書く代わりに、ライブプレビュー付きのポイント＆クリックのエディターが使えます。

![Hermes Mod skin editor](https://raw.githubusercontent.com/cocktailpeanut/hermes-mod/master/nous.png)

**できること:**

- すべての組み込みスキンとカスタムスキンを一覧表示
- 任意のスキンを、すべてのHermesスキンフィールド（色、スピナー、ブランディング、ツールプレフィックス、ツール絵文字）を備えたビジュアルエディターで開く
- テキストプロンプトから `banner_logo` のテキストアートを生成
- アップロードした画像（PNG、JPG、GIF、WEBP）を、複数のレンダースタイル（点字、ASCIIランプ、ブロック、ドット）で `banner_hero` のASCIIアートに変換
- `~/.hermes/skins/` に直接保存
- `~/.hermes/config.yaml` を更新してスキンを有効化
- 生成されたYAMLとライブプレビューを表示

### インストール

**オプション1 — Pinokio（1クリック）:**

[pinokio.computer](https://pinokio.computer) で見つけて、ワンクリックでインストールします。

**オプション2 — npx（ターミナルから最速）:**

```bash
npx -y hermes-mod
```

**オプション3 — 手動:**

```bash
git clone https://github.com/cocktailpeanut/hermes-mod.git
cd hermes-mod/app
npm install
npm start
```

### 使い方

1. アプリを起動します（Pinokioまたはターミナル経由）。
2. **Skin Studio** を開きます。
3. 編集する組み込みスキンまたはカスタムスキンを選びます。
4. テキストからロゴを生成し、必要に応じてヒーローアート用の画像をアップロードします。レンダースタイルと幅を選びます。
5. 色、スピナー、ブランディング、その他のフィールドを編集します。
6. **Save** をクリックして、スキンのYAMLを `~/.hermes/skins/` に書き込みます。
7. **Activate** をクリックして、それを現在のスキンとして設定します（`config.yaml` の `display.skin` を更新します）。

Hermes Mod は `HERMES_HOME` 環境変数を尊重するため、[プロファイル](/docs/user-guide/profiles)とも連携します。

## 運用上のメモ

- 組み込みスキンは `hermes_cli/skin_engine.py` から読み込まれます。
- 不明なスキンは自動的に `default` にフォールバックします。
- `/skin` は現在のセッションのアクティブなCLIテーマを即座に更新します。
- `~/.hermes/skins/` のユーザースキンは、同名の組み込みスキンより優先されます。
- `/skin` によるスキンの変更はセッション限りです。スキンを恒久的なデフォルトにするには、`config.yaml` で設定してください。
- `banner_logo` と `banner_hero` フィールドは、カラーのASCIIアート用にRichコンソールマークアップ（例: `[bold #FF0000]text[/]`）に対応しています。
