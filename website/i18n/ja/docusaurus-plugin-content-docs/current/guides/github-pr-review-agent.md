---
sidebar_position: 10
title: "チュートリアル: GitHub PRレビューエージェント"
description: "リポジトリを監視し、プルリクエストをレビューし、フィードバックを届ける自動AIコードレビュアーを構築する — 完全に自動で"
---

# チュートリアル: GitHub PRレビューエージェントを構築する

**課題:** あなたのチームは、レビューが追いつかない速さでPRを開きます。PRは目を通してもらえるまで何日も放置されます。誰もチェックする時間がないため、ジュニア開発者がバグをマージしてしまいます。あなたは構築に時間を使う代わりに、午前中をdiffの消化に費やします。

**解決策:** 四六時中リポジトリを見張り、すべての新しいPRをバグ・セキュリティ問題・コード品質についてレビューし、サマリーを送ってくれるAIエージェント — そうすれば、本当に人間の判断が必要なPRにだけ時間を使えます。

**構築するもの:**

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│   Cron Timer  ──▶  Hermes Agent  ──▶  GitHub API  ──▶  Review     │
│   (every 2h)       + gh CLI           (PR diffs)       delivery   │
│                    + skill                             (Telegram, │
│                    + memory                            Discord,   │
│                                                        local)     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

このガイドでは、スケジュールに従ってPRをポーリングするために**cronジョブ**を使います — サーバーや公開エンドポイントは不要です。NATやファイアウォールの背後でも動作します。

:::tip 代わりにリアルタイムのレビューが欲しいですか？
公開エンドポイントが利用できる場合は、[WebhookによるGitHub PRコメントの自動投稿](./webhook-github-pr-review.md)を参照してください — PRが開かれたり更新されたりすると、GitHubが即座にHermesにイベントをプッシュします。
:::

---

## 前提条件

- **Hermes Agentがインストール済み** — [インストールガイド](/docs/getting-started/installation)を参照
- cronジョブ用に**ゲートウェイが実行中**：
  ```bash
  hermes gateway install   # サービスとしてインストール
  # または
  hermes gateway           # フォアグラウンドで実行
  ```
- **GitHub CLI（`gh`）がインストール・認証済み**：
  ```bash
  # インストール
  brew install gh        # macOS
  sudo apt install gh    # Ubuntu/Debian

  # 認証
  gh auth login
  ```
- **メッセージングの設定**（オプション） — [Telegram](/docs/user-guide/messaging/telegram)または[Discord](/docs/user-guide/messaging/discord)

:::tip メッセージングがなくても問題なし
`deliver: "local"` を使うと、レビューを `~/.hermes/cron/output/` に保存できます。通知を接続する前のテストに最適です。
:::

---

## ステップ1: セットアップを確認する

HermesがGitHubにアクセスできることを確認します。チャットを開始します：

```bash
hermes
```

シンプルなコマンドでテストします：

```
Run: gh pr list --repo NousResearch/hermes-agent --state open --limit 3
```

オープンなPRのリストが表示されるはずです。これが動作すれば、準備完了です。

---

## ステップ2: 手動レビューを試す

チャットのまま、実際のPRをレビューするようHermesに依頼します：

```
Review this pull request. Read the diff, check for bugs, security issues,
and code quality. Be specific about line numbers and quote problematic code.

Run: gh pr diff 3888 --repo NousResearch/hermes-agent
```

Hermesは次を行います：
1. `gh pr diff` を実行してコード変更を取得する
2. diff全体を読み通す
3. 具体的な指摘を伴う構造化されたレビューを生成する

品質に満足できたら、自動化の時間です。

---

## ステップ3: レビュースキルを作成する

スキルは、セッションやcron実行をまたいで永続する一貫したレビューガイドラインをHermesに与えます。スキルがないと、レビュー品質はばらつきます。

```bash
mkdir -p ~/.hermes/skills/code-review
```

`~/.hermes/skills/code-review/SKILL.md` を作成します：

```markdown
---
name: code-review
description: Review pull requests for bugs, security issues, and code quality
---

# Code Review Guidelines

When reviewing a pull request:

## What to Check
1. **Bugs** — Logic errors, off-by-one, null/undefined handling
2. **Security** — Injection, auth bypass, secrets in code, SSRF
3. **Performance** — N+1 queries, unbounded loops, memory leaks
4. **Style** — Naming conventions, dead code, missing error handling
5. **Tests** — Are changes tested? Do tests cover edge cases?

## Output Format
For each finding:
- **File:Line** — exact location
- **Severity** — Critical / Warning / Suggestion
- **What's wrong** — one sentence
- **Fix** — how to fix it

## Rules
- Be specific. Quote the problematic code.
- Don't flag style nitpicks unless they affect readability.
- If the PR looks good, say so. Don't invent problems.
- End with: APPROVE / REQUEST_CHANGES / COMMENT
```

読み込まれたことを確認します — `hermes` を起動すると、起動時のスキルリストに `code-review` が表示されるはずです。

---

## ステップ4: あなたの規約を教える

これがレビュアーを実際に役立つものにする部分です。セッションを開始し、Hermesにチームの標準を教えます：

```
Remember: In our backend repo, we use Python with FastAPI.
All endpoints must have type annotations and Pydantic models.
We don't allow raw SQL — only SQLAlchemy ORM.
Test files go in tests/ and must use pytest fixtures.
```

```
Remember: In our frontend repo, we use TypeScript with React.
No `any` types allowed. All components must have props interfaces.
We use React Query for data fetching, never useEffect for API calls.
```

これらのメモリは永久に永続します — レビュアーは、毎回指示されなくてもあなたの規約を強制します。

---

## ステップ5: 自動cronジョブを作成する

これで全体を組み合わせます。2時間ごとに実行されるcronジョブを作成します：

```bash
hermes cron create "0 */2 * * *" \
  "Check for new open PRs and review them.

Repos to monitor:
- myorg/backend-api
- myorg/frontend-app

Steps:
1. Run: gh pr list --repo REPO --state open --limit 5 --json number,title,author,createdAt
2. For each PR created or updated in the last 4 hours:
   - Run: gh pr diff NUMBER --repo REPO
   - Review the diff using the code-review guidelines
3. Format output as:

## PR Reviews — today

### [repo] #[number]: [title]
**Author:** [name] | **Verdict:** APPROVE/REQUEST_CHANGES/COMMENT
[findings]

If no new PRs found, say: No new PRs to review." \
  --name "pr-review" \
  --deliver telegram \
  --skill code-review
```

スケジュールされたことを確認します：

```bash
hermes cron list
```

### その他の便利なスケジュール

| スケジュール | タイミング |
|----------|------|
| `0 */2 * * *` | 2時間ごと |
| `0 9,13,17 * * 1-5` | 平日のみ1日3回 |
| `0 9 * * 1` | 毎週月曜の朝のまとめ |
| `30m` | 30分ごと（トラフィックの多いリポジトリ） |

---

## ステップ6: オンデマンドで実行する

スケジュールを待ちたくないですか？ 手動でトリガーします：

```bash
hermes cron run pr-review
```

またはチャットセッション内から：

```
/cron run pr-review
```

---

## さらに進める

### レビューをGitHubに直接投稿する

Telegramに配信する代わりに、エージェントにPR自体へコメントさせます：

cronプロンプトに以下を追加します：

```
After reviewing, post your review:
- For issues: gh pr review NUMBER --repo REPO --comment --body "YOUR_REVIEW"
- For critical issues: gh pr review NUMBER --repo REPO --request-changes --body "YOUR_REVIEW"
- For clean PRs: gh pr review NUMBER --repo REPO --approve --body "Looks good"
```

:::caution
`gh` が `repo` スコープを持つトークンを使っていることを確認してください。レビューは `gh` が認証されているユーザーとして投稿されます。
:::

### 週次PRダッシュボード

すべてのリポジトリの月曜朝の概要を作成します：

```bash
hermes cron create "0 9 * * 1" \
  "Generate a weekly PR dashboard:
- myorg/backend-api
- myorg/frontend-app
- myorg/infra

For each repo show:
1. Open PR count and oldest PR age
2. PRs merged this week
3. Stale PRs (older than 5 days)
4. PRs with no reviewer assigned

Format as a clean summary." \
  --name "weekly-dashboard" \
  --deliver telegram
```

### 複数リポジトリの監視

プロンプトにリポジトリを追加してスケールアップします。エージェントはそれらを順番に処理します — 追加のセットアップは不要です。

---

## トラブルシューティング

### 「gh: command not found」
ゲートウェイは最小限の環境で実行されます。`gh` がシステムのPATHにあることを確認し、ゲートウェイを再起動してください。

### レビューが汎用的すぎる
1. `code-review` スキルを追加する（ステップ3）
2. メモリ経由でHermesにあなたの規約を教える（ステップ4）
3. スタックについて持っているコンテキストが多いほど、レビューは良くなります

### cronジョブが実行されない
```bash
hermes gateway status    # ゲートウェイは実行中か？
hermes cron list         # ジョブは有効か？
```

### レート制限
GitHubは認証済みユーザーに対して1時間あたり5,000 APIリクエストを許可します。各PRレビューは約3〜5リクエスト（リスト + diff + オプションのコメント）を使います。1日100件のPRをレビューしても、制限内に十分収まります。

---

## 次のステップ

- **[WebhookベースのPRレビュー](./webhook-github-pr-review.md)** — PRが開かれたときに即座にレビューを受け取る（公開エンドポイントが必要）
- **[デイリーブリーフィングボット](/docs/guides/daily-briefing-bot)** — PRレビューを朝のニュースダイジェストと組み合わせる
- **[プラグインを構築する](/docs/guides/build-a-hermes-plugin)** — レビューロジックを共有可能なプラグインにラップする
- **[プロファイル](/docs/user-guide/profiles)** — 独自のメモリと設定を持つ専用のレビュアープロファイルを実行する
- **[フォールバックプロバイダー](/docs/user-guide/features/fallback-providers)** — あるプロバイダーがダウンしてもレビューが実行されるようにする
