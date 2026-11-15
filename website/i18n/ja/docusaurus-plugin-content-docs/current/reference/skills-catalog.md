---
sidebar_position: 5
title: "バンドルスキルカタログ"
description: "Hermes Agent に同梱されるバンドルスキルのカタログ"
---

# バンドルスキルカタログ

Hermes はインストール時に `~/.hermes/skills/` へコピーされる大規模な組み込みスキルライブラリを同梱しています。以下の各スキルは、その完全な定義・セットアップ・使い方を記載した専用ページにリンクしています。

Hermes は `hermes update` でバンドルスキルも同期しますが、同期マニフェストはローカルでの削除やユーザーによる編集を尊重します。ここに記載されているスキルがプロファイルの `~/.hermes/skills/` ツリーに見当たらない場合でも、それは依然として Hermes に同梱されています。`hermes skills reset <name> --restore` で復元してください。

このリストにないがリポジトリには存在するスキルがある場合、カタログは `website/scripts/generate-skill-docs.py` によって再生成されます。

## apple

| Skill | 説明 | Path |
|-------|-------------|------|
| [`apple-notes`](/docs/user-guide/skills/bundled/apple/apple-apple-notes) | memo CLI 経由で Apple Notes を管理: 作成、検索、編集。 | `apple/apple-notes` |
| [`apple-reminders`](/docs/user-guide/skills/bundled/apple/apple-apple-reminders) | remindctl 経由で Apple Reminders を操作: 追加、一覧、完了。 | `apple/apple-reminders` |
| [`findmy`](/docs/user-guide/skills/bundled/apple/apple-findmy) | macOS の FindMy.app 経由で Apple デバイス/AirTags を追跡。 | `apple/findmy` |
| [`imessage`](/docs/user-guide/skills/bundled/apple/apple-imessage) | macOS の imsg CLI 経由で iMessage/SMS を送受信。 | `apple/imessage` |

## autonomous-ai-agents

| Skill | 説明 | Path |
|-------|-------------|------|
| [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code) | コーディングを Claude Code CLI に委譲（機能、PR）。 | `autonomous-ai-agents/claude-code` |
| [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex) | コーディングを OpenAI Codex CLI に委譲（機能、PR）。 | `autonomous-ai-agents/codex` |
| [`computer-use`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-computer-use) | デスクトップをバックグラウンド優先で操作し、シグナルに応じてエスカレート。 | `autonomous-ai-agents/computer-use` |
| [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | Hermes Agent の利用、設定、テーマ変更、拡張、オーケストレーション。 | `autonomous-ai-agents/hermes-agent` |

## creative

| Skill | 説明 | Path |
|-------|-------------|------|
| [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) | ダークテーマの SVG アーキテクチャ/クラウド/インフラ図を HTML として生成。 | `creative/architecture-diagram` |
| [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) | ASCII 動画: 動画/音声をカラー ASCII の MP4/GIF に変換。 | `creative/ascii-video` |
| [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic) | インフォグラフィック: 21 レイアウト × 21 スタイル（信息图、可视化）。 | `creative/baoyu-infographic` |
| [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design) | 単発の HTML 成果物をデザイン（ランディング、デッキ、プロトタイプ）。 | `creative/claude-design` |
| [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) | テキストを人間らしく: AI 臭さを除去し、本物の声を加える。 | `creative/humanizer` |
| [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) | Manim CE アニメーション: 3Blue1Brown 風の数学/アルゴリズム動画。 | `creative/manim-video` |
| [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js) | p5.js スケッチ: ジェネラティブアート、シェーダー、インタラクティブ、3D。 | `creative/p5js` |
| [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs) | 54 種の実在デザインシステム（Stripe、Linear、Vercel）を HTML/CSS で。 | `creative/popular-web-designs` |
| [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | 作詞の技術と Suno AI 音楽のプロンプト。 | `creative/songwriting-and-ai-music` |

## devops

| Skill | 説明 | Path |
|-------|-------------|------|
| [`sdlc-review`](/docs/user-guide/skills/bundled/devops/devops-sdlc-review) | Kanban のハンドオフをレビューし、検証済みの成果をルーティング。 | `devops/sdlc-review` |

## email

| Skill | 説明 | Path |
|-------|-------------|------|
| [`email-inbox-triage`](/docs/user-guide/skills/bundled/email/email-email-inbox-triage) | 受信トレイのトリアージ: スレッドの優先順位付け、安全な返信の下書き。 | `email/email-inbox-triage` |
| [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) | Himalaya CLI: ターミナルから IMAP/SMTP メール。 | `email/himalaya` |

## media

| Skill | 説明 | Path |
|-------|-------------|------|
| [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) | YouTube の文字起こしを要約・スレッド・ブログに変換。 | `media/youtube-content` |

## note-taking

| Skill | 説明 | Path |
|-------|-------------|------|
| [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) | Obsidian vault 内のノートを読む・検索・作成・編集。 | `note-taking/obsidian` |

## productivity

| Skill | 説明 | Path |
|-------|-------------|------|
| [`document-to-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-document-to-action-items) | ドキュメントから出典付きで義務・期限・タスクを抽出。 | `productivity/document-to-action-items` |
| [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) | gws CLI または Python で Gmail、Calendar、Drive、Docs、Sheets。 | `productivity/google-workspace` |
| [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) | OpenStreetMap/OSRM 経由のジオコーディング、POI、経路、タイムゾーン。 | `productivity/maps` |
| [`meeting-action-items`](/docs/user-guide/skills/bundled/productivity/productivity-meeting-action-items) | 会議メモを出典付きの決定事項・担当者・チケットに変換。 | `productivity/meeting-action-items` |
| [`microsoft-office`](/docs/user-guide/skills/bundled/productivity/productivity-microsoft-office) | Word、PowerPoint、Excel ファイルの作成・読み取り・編集。 | `productivity/microsoft-office` |
| [`pdf`](/docs/user-guide/skills/bundled/productivity/productivity-pdf) | PDF ファイル: 作成、読み取り、結合、フォーム入力、OCR、テキスト編集。 | `productivity/pdf` |
| [`product-price-monitor`](/docs/user-guide/skills/bundled/productivity/productivity-product-price-monitor) | 商品・航空券・出品の価格を監視し、目標価格で通知。 | `productivity/product-price-monitor` |
| [`teams-meeting-pipeline`](/docs/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | Teams 会議の要約、ジョブの再実行、Graph サブスクリプション。 | `productivity/teams-meeting-pipeline` |
| [`weekly-review-planning`](/docs/user-guide/skills/bundled/productivity/productivity-weekly-review-planning) | 週次リセット: コミットメント、停滞中の作業、翌週の計画。 | `productivity/weekly-review-planning` |

## research

| Skill | 説明 | Path |
|-------|-------------|------|
| [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) | キーワード、著者、カテゴリ、ID で arXiv 論文を検索。 | `research/arxiv` |
| [`competitor-news-monitor`](/docs/user-guide/skills/bundled/research/research-competitor-news-monitor) | 指定した企業の重要ニュースを監視し、出典付きダイジェストを作成。 | `research/competitor-news-monitor` |
| [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations) | 回答やドキュメントを、検証可能な出典付きの情報源に基づかせる。 | `research/grounded-citations` |

## social-media

| Skill | 説明 | Path |
|-------|-------------|------|
| [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) | xurl CLI 経由の X/Twitter: 生の投稿検索、投稿、DM、メディア。 | `social-media/xurl` |

## software-development

| Skill | 説明 | Path |
|-------|-------------|------|
| [`codebase-inspection`](/docs/user-guide/skills/bundled/software-development/software-development-codebase-inspection) | pygount でコードベースを検査: LOC、言語、比率。 | `software-development/codebase-inspection` |
| [`dogfood`](/docs/user-guide/skills/bundled/software-development/software-development-dogfood) | Web アプリの探索的 QA: バグ・証跡・レポートの発見。 | `software-development/dogfood` |
| [`github`](/docs/user-guide/skills/bundled/software-development/software-development-github) | gh CLI 経由の GitHub: PR、issue、レビュー、リポジトリ、認証。 | `software-development/github` |
| [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | コミット前レビュー: セキュリティスキャン、品質ゲート、自動修正。 | `software-development/requesting-code-review` |
| [`simplify-code`](/docs/user-guide/skills/bundled/software-development/software-development-simplify-code) | 4 エージェント並列で直近のコード変更をクリーンアップ。 | `software-development/simplify-code` |
| [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike) | 構築前にアイデアを検証する使い捨ての実験。 | `software-development/spike` |
| [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | 4 フェーズの根本原因デバッグ: 修正前にバグを理解する。 | `software-development/systematic-debugging` |
| [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD: RED-GREEN-REFACTOR の徹底、コードより先にテスト。 | `software-development/test-driven-development` |

## web

| Skill | 説明 | Path |
|-------|-------------|------|
| [`blocked-page-recovery`](/docs/user-guide/skills/bundled/web/web-blocked-page-recovery) | フェッチ失敗時に使用: 403/429、ペイウォール、WAF、ボット対策の壁。 | `web/blocked-page-recovery` |
