---
sidebar_position: 3
---

# プロファイル配布: エージェント一式を共有する

**プロファイル配布**は、完全なHermesエージェント — パーソナリティ、スキル、cronジョブ、MCP接続、設定 — をgitリポジトリとしてパッケージ化したものです。リポジトリにアクセスできる人なら誰でも、1つのコマンドでエージェント一式をインストールし、その場で更新でき、しかも自分のメモリ、セッション、APIキーには一切手を触れずに済みます。

[プロファイル](./profiles.md)がローカルなエージェントだとすれば、配布はそのエージェントを共有可能にしたものです。

## プロファイルを共有する2つの方法 {#two-ways-to-share-a-profile}

Hermesには2つの共有方法があり、それぞれ異なる問いに答えます。配布は永続的な方法で、エクスポートファイルは手軽な方法です。

| | **配布**（gitリポジトリ） | **エクスポートファイル**（`.tar.gz`） |
|---|---|---|
| 届け方 | `hermes profile install <repo>` | ファイルを送る — チャット、AirDrop、USB、メール |
| 受け取る人に必要なもの | git と、リポジトリへのアクセス | そのファイル |
| 更新 | `hermes profile update` が新しいバージョンを取り込む | ファイルを再送する |
| バージョニング | タグ、ブランチ、コミットSHA | なし — ある時点のスナップショット |
| 作者のセットアップコスト | `distribution.yaml` + `.gitignore` + リポジトリ | なし — コマンド1つ |
| 含まれるもの | SOUL、config、skills、cron、MCP、plugins | 同じもの **に加えて**、デスクトップのテーマとレイアウト |
| 作成方法 | `hermes profile install` / `update` | `/export` と `/import`、または `hermes profile export` / `import` |

エージェントが今後も改善し続けるプロダクトで、他の人が追従すべきものであれば **配布** を選びます: チームのレビュー済み社内エージェント、コミュニティ向けリリース、5台のマシンにデプロイする同じエージェントなどです。

今すぐ誰かに自分のセットアップを渡したいだけの場合や、新しいノートPCに移行する場合は **エクスポートファイル** を選びます。リポジトリもマニフェストも不要です — チャットで `/export` を実行してファイルを渡せば、相手は `/import` を実行します。[プロファイルファイルのエクスポートとインポート](#export-and-import-a-profile-file) を参照してください。

この2つは排他的ではありません。多くの作者は、プロファイルを自分で使い込み、同僚に `/export` で渡して意見をもらい、バージョン管理する価値が出てきたら配布として公開しています。

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

- **今すぐ一度だけ、誰かに自分のセットアップを渡したい場合。** 配布にはリポジトリ、マニフェスト、`.gitignore` が必要です。`/export` にはそのどれも不要です — [プロファイルファイルのエクスポートとインポート](#export-and-import-a-profile-file) を参照してください。プロファイルのバックアップや新しいマシンへの移行も同様です。
- **デスクトップのテーマとレイアウトを共有したい場合。** 配布が運ぶのはエージェント — SOUL、config、skills、cron、MCP、plugins — です。デスクトップアプリから作成したエクスポートは、見た目も運びます: スキン、ライト/ダークモード、カスタムテーマ、レールの色、ウィンドウのレイアウトです。
- **エージェントと一緒にAPIキーを共有したい場合。** `auth.json`、`.env`、およびHermesがプロファイルから読み取るその他の認証情報ストア（`.op.env`、`npmrc`、OAuthとボットトークンのファイル、`honcho.json`、`mem0.json`、`teams_pipeline_store.json`、`mcp-tokens/`、`vault/`、`proxy/`、ブラウザプロファイル、プラットフォームのセッション、そしてルートまたはスキル配下にネストされた `.ssh/`、`.aws/`、`.gnupg/`、`.kube/`、`.docker/`、`.azure/`、`.config/gh/`、`.config/gcloud/`、`.envrc`）は、配布から意図的に除外されています。インストールする人それぞれが自分の認証情報を持参します。（エクスポートファイルもこれらを取り除きます。）
- **メモリ / セッション / 会話履歴を共有したい場合。** それらはユーザーデータであり、配布のコンテンツではありません。決して同梱されません。（エクスポートファイルはここが異なります — 送る前に [エクスポートに含まれるもの](#what-an-export-file-contains) を読んでください。）

:::caution
**Hermesはgitを制御しません。** このページで説明するファイルの除外は、誰かが `hermes profile install` または `hermes profile update` を実行したときに **インストーラー** によって適用されます。`git add` や `git commit` を実行したときには **適用されません**。
:::

## ライフサイクル: 作者からインストール、そして更新まで

以下が、エンドツーエンドの全体的な流れです。気になる側を選んでください。

---

## 作者向け: 配布を公開する {#for-authors-publishing-a-distribution}

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

### ステップ3 — 最初のコミットの前に `.gitignore` を作成する {#step-3--create-a-gitignore-before-the-first-commit}

:::warning
これは `git init` や `git add` を実行する **前に** 行ってください。すでにそのプロファイルとチャットしたり、セットアップを実行したり、その他の方法で使用したりしている場合、ディレクトリには同梱してはいけないファイルが含まれています: `.env`、`auth.json`、`memories/`、`sessions/`、`state.db*`、`logs/` などです。 
:::

`~/.hermes/profiles/research-bot/.gitignore` を、少なくとも次の内容で作成します:

```gitignore
# Credentials & secrets — NEVER commit
auth.json
.env
.env.EXAMPLE    # generated by install, not authorship domain

# Runtime databases & state
state.db
state.db-shm
state.db-wal
hermes_state.db
response_store.db
response_store.db-shm
response_store.db-wal
gateway.pid
gateway_state.json
processes.json
auth.lock
active_profile
.update_check

# User data — NEVER commit
memories/
sessions/
logs/
plans/
workspace/
home/

# Caches & generated artifacts
image_cache/
audio_cache/
document_cache/
browser_screenshots/
cache/

# Infrastructure (should not be in profile dir, but safe to exclude)
hermes-agent/
.worktrees/
profiles/
bin/
node_modules/

# User customization namespace — your local overrides
local/

# Checkpoints & backups (can be huge)
checkpoints/
sandboxes/
backups/

# Logs
errors.log
.hermes_history

# Distribution-authored scheduler/skills state only
cron/*
!cron/jobs.json
skills/.*
```

これは、インストーラー側で取り除かれる [ハード除外パス](#whats-not-in-a-distribution-ever) を反映したものです。リポジトリから除外したいその他のもの（スクラッチファイル、大きなアセット、ローカル専用のスキル）も、ここに追加してください。

cronについては、`cron/jobs.json` だけをコミットします。Hermesは `cron/` 配下のそれ以外のすべてのエントリをランタイム状態（ロック、台帳、出力、提案、将来のスケジューラーのサイドカー）として扱い、配布からインストールしたり置き換えたりすることはありません。`skills/` 直下の隠しエントリも同様にHermesのローカルな管理情報です。作成されたスキルディレクトリ内の隠しファイルは、そのスキルの一部として扱われます。

同梱するジョブは独自の `timezone` をピン留めでき（例えば、すべてのインストール先で同じ瞬間に届く、09:00 `America/New_York` の市況アップデート）、その他のジョブは各ユーザーのプロファイルのタイムゾーンに従います。[同梱プロファイルにおけるピン留めジョブとローカル時刻のジョブ](./features/cron.md#pinned-and-local-time-jobs-in-a-shipped-profile) を参照してください。

### ステップ4 — gitリポジトリにプッシュする

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
インストーラーは、作者が何らかの理由で [ハード除外パス](#whats-not-in-a-distribution-ever) を同梱してしまった場合でも、それらをさらに取り除きます — ただし、それが守るのはインストールする人であり、作者ではありません。 
:::

### ステップ5 — バージョン付きリリースをタグ付けする

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
├── .gitignore                   # excludes secrets & user data (see Step 3)
├── distribution.yaml            # required
├── SOUL.md                      # strongly recommended
├── config.yaml                  # model, provider, tool defaults
├── mcp.json                     # MCP server connections
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── jobs.json                # cron definitions; installed paused
└── README.md                    # human-facing description (optional)
```

### 配布所有 vs ユーザー所有

インストールした人が新しいバージョンに更新すると、置き換えられるもの（作者の領域）と、そのまま残るもの（インストールした人の領域）があります。デフォルト:

| カテゴリ | パス | 更新時 |
|---|---|---|
| **配布所有** | `SOUL.md`、`config.yaml`、`mcp.json`、`skills/`、`cron/jobs.json`、`distribution.yaml` | ファイルは新しいクローンから置き換えられます。スキルディレクトリはエントリごとにマージされます。`cron/jobs.json` はジョブidごとにマージされます: 同梱された定義はその場で更新され、あなたのローカルのジョブと各ジョブの一時停止/有効の状態は残り、新たに同梱されたジョブは一時停止状態で届きます。その他の `cron/` のファイルと `skills/` 直下の隠しメタデータはランタイム状態であり、ローカルに残ります。 |
| **設定オーバーライド** | `config.yaml` | 実際にはデフォルトで保持されます — インストールした人がモデルやプロバイダーをチューニングしている可能性があるためです。更新時に `--force-config` を渡すとリセットされます。 |
| **ユーザー所有** | `memories/`、`sessions/`、`state.db*`、`auth.json`、`.env` とその他の認証情報ストア（`platforms/pairing/` や `platforms/whatsapp/session/` のように配布所有ディレクトリの配下にあるものを含む）と、Hermesがそれらについて保持する復旧用のコピー（`state-snapshots/`、`auth.json.corrupt`、`.env.bak-*`）、`logs/`、`workspace/`、`plans/`、`home/`、`*_cache/`、`local/` | 一切手を触れません。また、`distribution_owned` で認証情報ストアを所有対象にすることはできません。あなたのプロファイルで認証情報ストアを含むディレクトリがある場所にファイルを同梱する更新（`platforms` という名前のファイルなど）は、何かが書き込まれる前に拒否されます |

マニフェストで配布所有のリストをオーバーライドできます:

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # only my research skills; other installed skills stay
  - cron/jobs.json              # canonical cron store; merged job by job
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
5. 配布所有のファイルを `~/.hermes/profiles/research-bot/`（またはマニフェストの `name` が解決される場所）にコピーします。[ハード除外パス](#whats-not-in-a-distribution-ever) は、作者が誤ってリポジトリに残していた場合でも、このコピーの際に取り除かれます。
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
2. 配布所有のファイル（SOUL、mcp.json）と同梱されたスキルを置き換え、その後 `cron/jobs.json` をジョブidごとにマージします。あなた自身のジョブは残り、同梱されたジョブの定義はあなたの一時停止/有効の選択を変えずに更新され、新たに同梱されたジョブは一時停止状態で届きます。
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
# Laptop — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/research-bot
git init && git add . && git status   # confirm no secrets staged
git commit -m "initial"
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
# Engineering lead — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/pr-reviewer
# ... build and tune ...
git init && git add . && git status   # confirm no secrets staged
git commit -m "v1.0 PR reviewer"
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
# You — create .gitignore first (see "For authors" Step 3), then:
cd ~/.hermes/profiles/polymarket-trader
# Write a solid README.md at the repo root — GitHub shows it on the repo page
git init && git add . && git status   # confirm no secrets staged
git commit -m "v1.0"
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
# You — create .gitignore first (see "For authors" Step 3), then:
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

## プロファイルファイルのエクスポートとインポート {#export-and-import-a-profile-file}

バージョニングが不要なら、リポジトリは省略できます。`/export` はプロファイルを1つの `.tar.gz` にまとめ、`/import` は受け取った側でそれを新しいプロファイルとして展開します。認証情報は書き出しの際に取り除かれます。

### エクスポート

CLI、TUI、またはデスクトップのチャットで:

```
/export                          # the active profile → managed profile-exports/<name>-<timestamp>.tar.gz
/export research-bot             # a named profile
/export research-bot -o ~/Desktop/research-bot.tar.gz
```

`-o` を指定しない場合、CLIとTUIはアーカイブを現在の作業ディレクトリではなく、
デフォルトのHermesホーム配下にあるHermes管理の `profile-exports/` ディレクトリに置きます。
これにより、日常的なエクスポートがソースのチェックアウトに紛れ込むことを防ぎ、
生成されたプロファイルのスナップショットがリポジトリのソースファイルと誤認されることを
防ぎます。Hermesホーム自体がGitのチェックアウト内にある場合（一部の
Docker/カスタムデプロイメント）、アーカイブは `~/.hermes-profile-exports/` に、
それができなければOSの一時ディレクトリ配下のユーザーごとのディレクトリに置かれます — チェックアウト内に
置かれることは決してありません。安全な自動の保存先がまったく存在しない場合（すべての候補が
Gitのチェックアウト内にある場合）、エクスポートは「No safe automatic
export destination」として拒否されるため、チェックアウトの外のパスを `-o` で
指定する必要があります。明示的な `-o` のパスは、アーカイブの保存先を意図的に
選んだ場合には引き続き尊重されます。

または、シェルから同じ仕組みで:

```bash
hermes profile export research-bot
hermes profile export research-bot -o ./research-bot.tar.gz
```

**デスクトップアプリ** には3つの入口があり、いずれもネイティブの保存ダイアログが開きます:

- **⌘K → Export profile…**
- サイドバーのレールにあるプロファイルの四角を右クリック → **Export profile…**
- レールの **+** の隣にあるインポートボタンが逆方向を担います

デスクトップからのエクスポートには、CLIにはない追加のファイルが1つ含まれます: `desktop.json` で、スキン、ライト/ダークモード、スキンが必要とするカスタムテーマの定義、プロファイルのレールの色、ウィンドウのレイアウトを運びます。デスクトップから共有されたプロファイルが、あなたのものと同じように *振る舞う* だけでなく、同じように *見える* のはこのためです。

### インポート

```
/import ~/Downloads/research-bot.tar.gz
/import ~/Downloads/research-bot.tar.gz --name research-bot-2
```

```bash
hermes profile import ./research-bot.tar.gz
hermes profile import ./research-bot.tar.gz --name research-bot-2
```

`--name` を渡さない限り、プロファイル名はアーカイブから推測されます。既存のプロファイルへの上書きインポートは拒否されます — 先に古いものをリネームするか削除してください。名前が既存のコマンドと衝突しない場合は、シェルラッパー（`research-bot` → `hermes -p research-bot`）が作成されます。

デスクトップアプリでインポートすると、`desktop.json` のオーバーレイも適用され、新しいプロファイルの新しいチャットに移動します。デスクトップで作成したアーカイブをCLIからインポートしても問題ありません — オーバーレイのファイルはディスク上に一緒に置かれ、次にデスクトップでそのプロファイルを開いたときに適用されます。

:::note
`default` としてインポートすることはできません — その名前は組み込みのルートプロファイル（`~/.hermes`）です。`--name something-else` を渡してください。
:::

### エクスポートファイルに含まれるもの {#what-an-export-file-contains}

どちらの種類のプロファイルでも常に除外されるもの: `auth.json`、`.env`、およびHermesがプロファイルから読み取るその他の認証情報ストア（WeChatの `weixin/accounts/` などのOAuthとボットトークンのファイル、`honcho.json`、`mcp-tokens/`、`vault/`、`proxy/` 内のiron-proxyのキー、`chrome-debug/` 内の `/browser connect` 用のものを含むブラウザプロファイル、プラットフォームのセッションとペアリングストア、Graph webhookの `clientState` を含むTeamsパイプラインの `teams_pipeline_store.json`）、そしてルートまたはスキル配下にネストされた `.ssh/`、`.aws/`、`.gnupg/`、`.kube/`、`.docker/`、`.azure/`、`.config/gh/`、`.config/gcloud/`、`.envrc`。あなたのAPIキーがマシンの外に出ることはありません。`honcho.json` と `mem0.json` は設定の隣にプロバイダーのAPIキーを保持していることがあるため、インポート後は `hermes honcho setup` または `hermes memory setup` を再度実行してください。

**デフォルトプロファイル**（`~/.hermes`）は許可リストを通じてエクスポートされます — 既知のHermesの成果物のみであるため、ホームディレクトリに置かれた無関係なファイルが紛れ込むことはありません:

`config.yaml`、`SOUL.md`、`MEMORY.md`、`USER.md`、`todo.json`、`system_prompt.md`、`AGENTS.md`、`CLAUDE.md`、`.cursorrules`、`skills/`、`plugins/`、`cron/`、`scripts/`、`sessions/`、`memories/`、`knowledge/`、`preferences/`、そしてデスクトップが用意した場合は `desktop.json`。

**名前付きプロファイル**（`~/.hermes/profiles/<name>`）は、それらの認証情報ストア、`home/`（Hermesのツールサブプロセスの `HOME` で、`git`、`ssh`、`gh`、`npm`、スキルのCLIが認証情報を保持する場所）、そしてHermesがそれらについて保持する復旧用のコピー（`backups/`、`state-snapshots/`、すべての `auth.json.*` や `.env.bak*`、`hermes update` が書き出す `config.yaml.bak-<timestamp>` のコピー）を除いたディレクトリ全体をコピーします。`config.yaml.bak-my-note` のように自分で名前を付けたコピーはエクスポートされ、そのシークレットは `config.yaml` と同様に伏せ字にされます。こちらは範囲が広く — プロファイルに `state.db`、ログ、キャッシュがあれば、それらもアーカイブに含まれ、ファイルは大きくなります。

:::caution 送る前にアーカイブを確認する
エクスポートはプロファイルのスナップショットであり、厳選されたリリースではありません。配布とは異なり、`memories/`、`sessions/`、`USER.md` を **含むことがあります** — そして、スキル、メモリ、ペルソナに書き込んだ個人的な内容をスキャンするものは何もありません。認証情報はファイル名でフィルタリングされますが、内容はフィルタリングされません。

他の人と共有する前に、中身を一覧表示してください:

```bash
tar -tzf research-bot.tar.gz | less
```

渡したくない会話履歴が含まれている場合は、代わりに [配布](#for-authors-publishing-a-distribution) を公開してください — 配布がメモリやセッションを同梱することはありません。
:::

## 配布に（決して）含まれないもの {#whats-not-in-a-distribution-ever}

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

インストールする人として配布をクローンすると、これらはプロファイルディレクトリに単純にコピーされません。更新しても、あなたのコピーはそのまま残ります。同じ配布を5台のマシンにインストールした場合、このデータの分離された5つのセット — マシンごとに1つ — を持つことになります。

:::caution
この除外は、**インストールする人のマシン上でのインストール / 更新時** に実行されます。作者が機密性のある/不要なファイルをコミットすることを防ぐものでは **ありません**。作者は [`.gitignore`](#step-3--create-a-gitignore-before-the-first-commit) を使って、シークレットをリポジトリから除外する必要があります。
:::

## セキュリティと信頼

プロファイル配布は、デフォルトでは署名されていません。あなたは次を信頼することになります:

- **gitホスト**（GitHub / GitLab / その他）が、作者がプッシュしたバイトを提供してくれること。
- **作者**が、悪意のあるSOUL、スキル、cronジョブを同梱しないこと。

配布のcronジョブは**自動的にはスケジュールされません** — 新たに同梱されたジョブは一時停止状態でインストールされます。`hermes -p <name> cron list` で確認し、信頼できるジョブだけを再開してください。SOUL.mdとスキルは、そのプロファイルとチャットを始めた時点で有効になるため、知らない人からインストールする場合は、最初の実行前にそれらを読んでください。

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
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — [エクスポートファイル](#export-and-import-a-profile-file) のCLI版
- [スラッシュコマンドリファレンス](../reference/slash-commands.md) — `/export`、`/import`、およびその他すべてのチャット内コマンド
- [HermesでSOULを使う](../guides/use-soul-with-hermes.md) — パーソナリティの作成
- [パーソナリティとSOUL](./features/personality.md) — SOULがエージェントにどう組み込まれるか
- [スキルカタログ](../reference/skills-catalog.md) — 同梱できるスキル
