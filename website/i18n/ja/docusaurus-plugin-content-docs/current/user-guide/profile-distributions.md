---
sidebar_position: 3
---

# プロファイル配布: エージェント一式を共有する

**プロファイル配布**は、完全なHermesエージェント — パーソナリティ、スキル、cronジョブ、MCP接続、設定 — をgitリポジトリとしてパッケージ化したものです。リポジトリにアクセスできる人なら誰でも、1つのコマンドでエージェント一式をインストールし、その場で更新でき、しかも自分のメモリ、セッション、APIキーには一切手を触れずに済みます。

[プロファイル](./profiles.md)がローカルなエージェントだとすれば、配布はそのエージェントを共有可能にしたものです。

## これが意味すること

配布がなかった頃は、Hermesエージェントを共有するには、相手に次のものを送る必要がありました:

1. あなたのSOUL.md
2. インストールすべきスキルのリスト
3. シークレットを除いた config.yaml
4. どのMCPサーバーを接続したかの説明
5. スケジュールしたcronジョブ
6. どの環境変数を設定すべきかの手順

…そして、相手がそれを正しく組み立ててくれることを願うしかありませんでした。バージョンアップやバグ修正のたびに、この受け渡しを繰り返す必要がありました。

配布では、それらすべてが1つのgitリポジトリに収まります:

```
my-research-agent/
├── distribution.yaml    # manifest: name, version, env-var requirements
├── SOUL.md              # the agent's personality / system prompt
├── config.yaml          # model, temperature, reasoning, tool defaults
├── skills/              # bundled skills that come with the agent
├── cron/                # scheduled tasks the agent runs
└── mcp.json             # MCP servers the agent connects to
```

受け取った人は次を実行します:

```bash
hermes profile install github.com/you/my-research-agent --alias
```

…これでエージェント一式が手に入ります。自分のAPIキーを記入し（`.env.EXAMPLE` → `.env`）、`my-research-agent chat` を実行したり、Telegram / Discord / Slack / 任意のゲートウェイプラットフォームを通じてアドレス指定したりできます。あなたが新しいバージョンをプッシュすると、相手は `hermes profile update my-research-agent` を実行してあなたの変更を取り込みます — 相手のメモリとセッションはそのまま残ります。

## なぜgitか？

tarball、HTTPアーカイブ、独自フォーマットを検討しました。どれもgitには勝てませんでした:

- **作者にとってビルドステップが不要。** GitHubにプッシュすれば、利用者がインストールできます。「これをパックして、あれをアップロードして、インデックスを更新して」というループはありません。
- **タグ、ブランチ、コミットがすでにバージョニングシステムになっている。** タグのプッシュは、他のツールでの「パック + リリースのアップロード」に相当することを実現します。
- **更新はフェッチで済む。** アーカイブ全体の再ダウンロードではありません。
- **透明性。** ユーザーはリポジトリを閲覧し、バージョン間の差分を読み、それに対してissueを立て、カスタマイズのためにフォークできます。
- **プライベートリポジトリも無料で機能する。** SSHキー、`git credential` ヘルパー、GitHub CLIに保存された認証情報など、あなたのターミナルがすでに設定している認証方法がそのまま透過的に適用されます。
- **再現性はコミットSHAそのもの。** pipやnpmが記録するのと同じものです。

トレードオフ: 受け取る人はgitがインストールされている必要があります。2026年にHermesを動かしているマシンなら、それはすでに満たされています。

## どんなときに配布を使うべきか？

向いているケース:

- **特化したエージェントを共有する場合** — コンプライアンス監視、コードレビュアー、リサーチアシスタント、カスタマーサポートボットなど — をチームやコミュニティと共有するとき。
- **同じエージェントを複数のマシンにデプロイする場合**で、毎回手動でファイルをコピーしたくないとき。
- **エージェントを反復改善している場合**で、受け取る人に1つのコマンドで新しいバージョンを取り込んでもらいたいとき。
- **エージェントをプロダクトとして構築している場合** — こだわりのデフォルト、厳選したスキル、チューニングしたプロンプト — を、他の人が出発点として使うべきものとして提供するとき。

向いていないケース:

- **自分のマシンでプロファイルをバックアップしたいだけの場合。** [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) を使ってください — それがそのためのものです。
- **エージェントと一緒にAPIキーを共有したい場合。** `auth.json` と `.env` は、配布から意図的に除外されています。インストールする人それぞれが自分の認証情報を持参します。
- **メモリ / セッション / 会話履歴を共有したい場合。** それらはユーザーデータであり、配布のコンテンツではありません。決して同梱されません。

## ライフサイクル: 作者からインストール、そして更新まで

以下が、エンドツーエンドの全体的な流れです。気になる側を選んでください。

---

## 作者向け: 配布を公開する

### ステップ1 — 動作するプロファイルから始める

他のプロファイルと同じように、エージェントを構築・改善します:

```bash
hermes profile create research-bot
research-bot setup                    # configure model, API keys
# Edit ~/.hermes/profiles/research-bot/SOUL.md
# Install skills, wire up MCP servers, schedule cron jobs, etc.
research-bot chat                     # dogfood until it feels right
```

### ステップ2 — `distribution.yaml` を追加する

`~/.hermes/profiles/research-bot/distribution.yaml` を作成します:

```yaml
name: research-bot
version: 1.0.0
description: "Autonomous research assistant with arXiv and web tools"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# Tell installers which env vars the agent needs. These are checked against
# the installer's shell and existing .env file so they don't get nagged
# about keys they already have configured.
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key (for model access)"
    required: true
  - name: SERPAPI_KEY
    description: "SerpAPI key for web search"
    required: false
    default: ""
```

これがマニフェストのすべてです。`name` 以外のすべてのフィールドには、妥当なデフォルトがあります。

### ステップ3 — gitリポジトリにプッシュする

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

これでリポジトリが配布になりました。アクセスできる人なら誰でもインストールできます。

:::note
gitリポジトリには、**配布からすでに除外されているものを除き、プロファイルディレクトリ内のすべて**が含まれます: `auth.json`、`.env`、`memories/`、`sessions/`、`state.db*`、`logs/`、`workspace/`、`*_cache/`、`local/`。これらはあなたのマシンに残ります。追加のパスを除外したい場合は、`.gitignore` を追加することもできます。
:::

### ステップ4 — バージョン付きリリースをタグ付けする

エージェントが安定したポイントに達するたびに、バージョンを上げてタグを付けます:

```bash
# Edit distribution.yaml: version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

`hermes profile update research-bot` を実行する受け取り手は、最新版を取り込みます。

### リポジトリの見た目

作成された完全な配布:

```
research-bot/
├── distribution.yaml            # required
├── SOUL.md                      # strongly recommended
├── config.yaml                  # model, provider, tool defaults
├── mcp.json                     # MCP server connections
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # scheduled tasks
└── README.md                    # human-facing description (optional)
```

### 配布所有 vs ユーザー所有

インストールした人が新しいバージョンに更新すると、置き換えられるもの（作者の領域）と、そのまま残るもの（インストールした人の領域）があります。デフォルト:

| カテゴリ | パス | 更新時 |
|---|---|---|
| **配布所有** | `SOUL.md`、`config.yaml`、`mcp.json`、`skills/`、`cron/`、`distribution.yaml` | 新しいクローンから置き換え |
| **設定オーバーライド** | `config.yaml` | 実際にはデフォルトで保持されます — インストールした人がモデルやプロバイダーをチューニングしている可能性があるためです。更新時に `--force-config` を渡すとリセットされます。 |
| **ユーザー所有** | `memories/`、`sessions/`、`state.db*`、`auth.json`、`.env`、`logs/`、`workspace/`、`plans/`、`home/`、`*_cache/`、`local/` | 一切手を触れない |

マニフェストで配布所有のリストをオーバーライドできます:

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # only my research skills; other installed skills stay
  - cron/digest.json
```

省略した場合は、上記のデフォルトが適用されます — ほとんどの配布が望むのはこれです。

---

## インストールする人向け: 配布を使う

### インストール

```bash
hermes profile install github.com/you/research-bot --alias
```

何が起こるか:

1. リポジトリを一時ディレクトリにクローンします。
2. `distribution.yaml` を読み取り、マニフェスト（名前、バージョン、説明、作者、必要な環境変数）を表示します。
3. 必要な各環境変数を、あなたのシェル環境と対象プロファイルの既存の `.env` に対してチェックします。それぞれを `✓ set` または `needs setting` とマークするので、何を設定すべきか正確に分かります。
4. 確認を求めます。`-y` / `--yes` を渡すとスキップできます。
5. 配布所有のファイルを `~/.hermes/profiles/research-bot/`（またはマニフェストの `name` が解決される場所）にコピーします。
6. 必要なキーをコメントアウトした状態で `.env.EXAMPLE` を書き出します — `.env` にコピーして記入してください。
7. `--alias` を付けると、`research-bot chat` を直接実行できるようにラッパーを作成します。

### ソースの種類

任意のgit URLが機能します:

```bash
# GitHub shorthand
hermes profile install github.com/you/research-bot

# Full HTTPS
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# Self-hosted, GitLab, Gitea, Forgejo — any Git host
hermes profile install https://git.example.com/team/research-bot.git

# Private repo using your configured git auth
hermes profile install git@github.com:your-org/internal-bot.git

# Local directory during development (no git push needed)
hermes profile install ~/my-profile-in-progress/
```

### プロファイル名のオーバーライド

同じ配布を異なるプロファイル名で使いたい2人のユーザー:

```bash
# Alice
hermes profile install github.com/acme/support-bot --name support-us --alias
# Bob (same distribution, different local name)
hermes profile install github.com/acme/support-bot --name support-eu --alias
```

### 環境変数を記入する

インストール後、エージェントのプロファイルには `.env.EXAMPLE` が含まれます:

```
# Environment variables required by this Hermes distribution.
# Copy to `.env` and fill in your own values before running.

# OpenAI API key (for model access)
# (required)
OPENAI_API_KEY=

# SerpAPI key for web search
# (optional)
# SERPAPI_KEY=
```

コピーします:

```bash
cp ~/.hermes/profiles/research-bot/.env.EXAMPLE ~/.hermes/profiles/research-bot/.env
# Edit .env, paste your real keys
```

すでにシェル環境にあった必須キー（例えば `~/.zshrc` でエクスポートされている `OPENAI_API_KEY`）は、インストール時に `✓ set` とマークされます — `.env` で重複させる必要はありません。

### インストールしたものを確認する

```bash
hermes profile info research-bot
```

次が表示されます:

```
Distribution: research-bot
Version:      1.0.0
Description:  Autonomous research assistant with arXiv and web tools
Author:       Your Name
Requires:     Hermes >=0.12.0
Source:       https://github.com/you/research-bot
Installed:    2026-05-08T17:04:32+00:00

Environment variables:
  OPENAI_API_KEY (required) — OpenAI API key (for model access)
  SERPAPI_KEY (optional) — SerpAPI key for web search
```

`hermes profile list` も `Distribution` カラムを表示するので、どのプロファイルがリポジトリ由来で、どれを手作りしたのかが一目で分かります:

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆default         claude-sonnet-4              stopped      —            —
  coder           gpt-5                        stopped      coder        —
  research-bot    claude-opus-4                stopped      research-bot research-bot@1.0.0
  telemetry       claude-sonnet-4              running      telemetry    telemetry@2.3.1
```

### 更新

```bash
hermes profile update research-bot
```

何が起こるか:

1. 記録されたソースURLからリポジトリを再クローンします。
2. 配布所有のファイル（SOUL、skills、cron、mcp.json）を置き換えます。
3. あなたの `config.yaml` を**保持します** — モデル、temperature、その他の設定をチューニングしている可能性があるためです。上書きするには `--force-config` を渡します。
4. ユーザーデータには**一切手を触れません**: メモリ、セッション、認証、`.env`、ログ、状態。

アーカイブ全体の再ダウンロードはありません。設定へのローカルな変更を踏みつぶすこともありません。会話履歴を削除することもありません。

### 削除

```bash
hermes profile delete research-bot
```

削除のプロンプトは、確認を求める前に配布情報を表示します:

```
Profile: research-bot
Path:    ~/.hermes/profiles/research-bot
Model:   claude-opus-4 (anthropic)
Skills:  12
Distribution: research-bot@1.0.0
Installed from: https://github.com/you/research-bot

This will permanently delete:
  • All config, API keys, memories, sessions, skills, cron jobs
  • Command alias (~/.local/bin/research-bot)

Type 'research-bot' to confirm:
```

そのため、エージェントがどこから来たのか分からないまま、あるいは再インストールできないまま、誤って削除してしまうことはありません。

---

## ユースケースとパターン

### 個人: 1つのエージェントを複数マシンで同期する

ノートPCでリサーチアシスタントを構築しました。同じエージェントをワークステーションでも使いたいとします。

```bash
# Laptop
cd ~/.hermes/profiles/research-bot
git init && git add . && git commit -m "initial"
git remote add origin git@github.com:you/research-bot.git
git push -u origin main

# Workstation
hermes profile install github.com/you/research-bot --alias
# Fill in .env. Done.
```

ノートPCでの反復改善（`git commit && push`）は、`hermes profile update research-bot` でワークステーションに取り込まれます。メモリはマシンごとに残ります — ノートPCは自分の会話を、ワークステーションは自分の会話を覚えており、衝突しません。

### チーム: レビュー済みの社内エージェントを配布する

エンジニアリングチームが、特定のSOUL、特定のスキル、そしてすべてのPRをそれに通すcronを備えた、共有のPRレビューボットを欲しがっているとします。

```bash
# Engineering lead
cd ~/.hermes/profiles/pr-reviewer
# ... build and tune ...
git init && git add . && git commit -m "v1.0 PR reviewer"
git tag v1.0.0
git push -u origin main --tags    # push to your company's internal Git host

# Each engineer
hermes profile install git@github.com:your-org/pr-reviewer.git --alias
# Fill in .env with their own API key (billed to them), .env.EXAMPLE points at what's required
pr-reviewer chat
```

リードがv1.1（より良いSOUL、新しいスキル）を出すと、エンジニアは `hermes profile update pr-reviewer` を実行し、数分以内に全員が新しいバージョンになります。

### コミュニティ: 公開エージェントを公開する

何か新しいものを構築しました — 「Polymarketトレーダー」、「学術論文要約ツール」、「Minecraftサーバー運用アシスタント」など。それを共有したいとします。

```bash
# You
cd ~/.hermes/profiles/polymarket-trader
# Write a solid README.md at the repo root — GitHub shows it on the repo page
git init && git add . && git commit -m "v1.0"
git tag v1.0.0
# Publish to a public GitHub repo
git remote add origin https://github.com/you/hermes-polymarket-trader.git
git push -u origin main --tags

# Anyone
hermes profile install github.com/you/hermes-polymarket-trader --alias
```

インストールコマンドをツイートしましょう。試した人があなたにissueやPRを送ってくれます。誰かがカスタマイズしたければフォークします — 誰もがすでに知っている、同じgitワークフローです。

### プロダクト: こだわりのエージェントを提供する

Hermesの上に何かを構築しました — コンプライアンス監視ハーネス、カスタマーサポートスタック、ドメイン特化のリサーチプラットフォームなど。それをプロダクトとして配布したいとします。

```yaml
# distribution.yaml
name: telemetry-harness
version: 2.3.1
description: "Compliance telemetry harness — monitors and reviews regulated workflows"
hermes_requires: ">=0.13.0"
author: "Acme Compliance Inc."
license: "Commercial"

env_requires:
  - name: ACME_API_KEY
    description: "Your Acme Compliance license key (email support@acme.com)"
    required: true
  - name: OPENAI_API_KEY
    description: "OpenAI API key for model access"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "URL for your Graphiti knowledge graph instance"
    required: false
    default: "http://127.0.0.1:8000/sse"
```

顧客は単一のコマンドでインストールします。インストールのプレビューが、どのキーを用意すべきかを正確に伝えます。更新はあなたが新しいリリースをタグ付けした瞬間に展開されます。そして顧客のコンプライアンスデータ（`memories/`、`sessions/`）が顧客のマシンを離れることは決してありません。

### 一時利用: 共有インフラ上での使い捨てスクリプト

あなたは運用リードです。本番インシデントを診断する一時的なエージェント — 適切なツールとMCP接続を備えた既製のSOUL — が欲しく、それを今後1週間、3人のオンコールエンジニアのノートPCで動かしたいとします。

```bash
# You
# Build the profile, commit, push a private repo
git push -u origin main

# Each on-call
hermes profile install git@github.com:your-org/incident-2026-q2.git --alias

# Incident resolved — tear it down
hermes profile delete incident-2026-q2
```

インストールと削除のサイクルは、使い捨てにできるほど安価です。

---

## レシピ

### 特定のバージョンに固定する

:::note
git refの固定（`#v1.2.0`）は予定されていますが、初回リリースには含まれていません — 現在、インストールはデフォルトブランチを追跡します。`hermes profile info <name>` でインストール済みバージョンを追跡し、準備が整うまで更新を控えてください。
:::

### 自分のバージョンと最新版を比較する

```bash
# Your installed version
hermes profile info research-bot | grep Version

# Latest upstream (without installing)
git ls-remote --tags https://github.com/you/research-bot | tail -5
```

### 更新をまたいでローカルの設定カスタマイズを維持する

デフォルトの更新動作はすでにこれを行っています: `config.yaml` は保持されます。念のため、ローカルの調整は配布が所有しないファイルに書き込んでください:

```yaml
# ~/.hermes/profiles/research-bot/local/my-overrides.yaml
# (distribution never touches local/)
```

…そして必要に応じて `config.yaml` やSOULからそれを参照します。

### クリーンな再インストールを強制する

```bash
# Nuke and re-install from scratch (loses memories/sessions too)
hermes profile delete research-bot --yes
hermes profile install github.com/you/research-bot --alias

# Update to current main but reset config.yaml to the distribution's default
hermes profile update research-bot --force-config --yes
```

### フォークしてカスタマイズする

標準のgitワークフローです — 配布はただのリポジトリです:

```bash
# Fork the repo on GitHub, then install your fork
hermes profile install github.com/yourname/forked-research-bot --alias

# Iterate locally in ~/.hermes/profiles/forked-research-bot/
# Edit SOUL.md, commit, push to your fork
# Upstream changes: pull them into your fork the usual way
```

### プッシュする前に配布をテストする

作者のマシンから:

```bash
# Install from a local directory (no git push needed)
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test --alias

# Tweak, delete, re-install until it's right
hermes profile delete research-bot-test --yes
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test
```

---

## 配布に（決して）含まれないもの

インストーラーは、作者が誤って同梱した場合でも、次のパスをハードに除外します。これをオーバーライドできる設定オプションはありません — この安全ガードは、リグレッションテスト済みの不変条件です:

- `auth.json` — OAuthトークン、プラットフォームの認証情報
- `.env` — APIキー、シークレット
- `memories/` — 会話メモリ
- `sessions/` — 会話履歴
- `state.db`、`state.db-shm`、`state.db-wal` — セッションのメタデータ
- `logs/` — エージェントとエラーのログ
- `workspace/` — 生成された作業ファイル
- `plans/` — スクラッチのプラン
- `home/` — Dockerバックエンドにおけるユーザーのホームマウント
- `*_cache/` — 画像 / 音声 / ドキュメントのキャッシュ
- `local/` — ユーザー予約のカスタマイズ名前空間

配布をクローンすると、これらは単純に存在しません。更新しても、そのまま残ります。同じ配布を5台のマシンにインストールした場合、このデータの分離された5つのセット — マシンごとに1つ — を持つことになります。

## セキュリティと信頼

プロファイル配布は、デフォルトでは署名されていません。あなたは次を信頼することになります:

- **gitホスト**（GitHub / GitLab / その他）が、作者がプッシュしたバイトを提供してくれること。
- **作者**が、悪意のあるSOUL、スキル、cronジョブを同梱しないこと。

配布のcronジョブは**自動的にはスケジュールされません** — インストーラーが `hermes -p <name> cron list` を表示し、あなたが明示的に有効化します。SOUL.mdとスキルは、そのプロファイルとチャットを始めた時点で有効になるため、知らない人からインストールする場合は、最初の実行前にそれらを読んでください。

大まかな例え: 配布のインストールは、ブラウザ拡張機能やVS Code拡張機能のインストールに似ています。摩擦は少なく、力は大きく、ソースを信頼します。社内の配布には、プライベートリポジトリと普段のgit認証を使ってください — 新たに設定するものはありません。

将来のバージョンでは、署名、解決済みコミットSHAを記録するロックファイル（`.distribution-lock.yaml`）、そして更新を適用する前に差分を表示する `--dry-run` フラグが追加されるかもしれません。いずれもまだ提供されていません。

## 内部の仕組み

実装の詳細、正確なCLIの挙動、すべてのフラグについては、[プロファイルコマンドリファレンス](../reference/profile-commands.md#distribution-commands)を参照してください。

要点:

- `install`、`update`、`info` は `hermes profile` の中にあります — 並行したコマンドツリーではありません。
- マニフェスト形式はYAMLで、ごく小さな必須スキーマ（`name` のみ）を持ちます。
- インストーラーはクローンにあなたのローカルの `git` バイナリを使用するため、あなたのシェルがすでに処理している認証（SSHキー、認証情報ヘルパー）が透過的に機能します。
- クローン後、`.git/` は取り除かれます — インストールされたプロファイル自体はgitチェックアウトではないため、「しまった、配布のgit履歴に `.env` を誤ってコミットしてしまった」という罠を避けられます。
- 予約済みのプロファイル名（`hermes`、`test`、`tmp`、`root`、`sudo`）は、一般的なバイナリとの衝突を避けるため、インストール時に拒否されます。

## 関連項目

- [プロファイル: 複数エージェントの実行](./profiles.md) — 基本概念
- [プロファイルコマンドリファレンス](../reference/profile-commands.md) — すべてのフラグ、すべてのオプション
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — ローカルのバックアップ / リストア（配布ではありません）
- [HermesでSOULを使う](../guides/use-soul-with-hermes.md) — パーソナリティの作成
- [パーソナリティとSOUL](./features/personality.md) — SOULがエージェントにどう組み込まれるか
- [スキルカタログ](../reference/skills-catalog.md) — 同梱できるスキル
