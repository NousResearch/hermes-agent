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
| [`macos-computer-use`](/docs/user-guide/skills/bundled/apple/apple-macos-computer-use) | macOS デスクトップをバックグラウンドで操作 — スクリーンショット、マウス、キーボード、スクロール、ドラッグ — ユーザーのカーソル・キーボードフォーカス・Space を奪わずに実行。ツール対応の任意のモデルで動作。`computer_use` ツールが... | `apple/macos-computer-use` |

## autonomous-ai-agents

| Skill | 説明 | Path |
|-------|-------------|------|
| [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code) | コーディングを Claude Code CLI に委譲（機能、PR）。 | `autonomous-ai-agents/claude-code` |
| [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex) | コーディングを OpenAI Codex CLI に委譲（機能、PR）。 | `autonomous-ai-agents/codex` |
| [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) | Hermes Agent の設定、拡張、貢献を行う。 | `autonomous-ai-agents/hermes-agent` |
| [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) | コーディングを OpenCode CLI に委譲（機能、PR レビュー）。 | `autonomous-ai-agents/opencode` |

## creative

| Skill | 説明 | Path |
|-------|-------------|------|
| [`architecture-diagram`](/docs/user-guide/skills/bundled/creative/creative-architecture-diagram) | ダークテーマの SVG アーキテクチャ/クラウド/インフラ図を HTML として生成。 | `creative/architecture-diagram` |
| [`ascii-art`](/docs/user-guide/skills/bundled/creative/creative-ascii-art) | ASCII アート: pyfiglet、cowsay、boxes、画像から ASCII への変換。 | `creative/ascii-art` |
| [`ascii-video`](/docs/user-guide/skills/bundled/creative/creative-ascii-video) | ASCII 動画: 動画/音声をカラー ASCII の MP4/GIF に変換。 | `creative/ascii-video` |
| [`baoyu-comic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-comic) | ナレッジコミック（知识漫画）: 教育、伝記、チュートリアル。 | `creative/baoyu-comic` |
| [`baoyu-infographic`](/docs/user-guide/skills/bundled/creative/creative-baoyu-infographic) | インフォグラフィック: 21 レイアウト × 21 スタイル（信息图、可视化）。 | `creative/baoyu-infographic` |
| [`claude-design`](/docs/user-guide/skills/bundled/creative/creative-claude-design) | 単発の HTML 成果物をデザイン（ランディング、デッキ、プロトタイプ）。 | `creative/claude-design` |
| [`comfyui`](/docs/user-guide/skills/bundled/creative/creative-comfyui) | ComfyUI で画像・動画・音声を生成 — インストール、起動、ノード/モデルの管理、パラメータ注入を伴うワークフローの実行。ライフサイクルには公式の comfy-cli を、実行には REST/WebSocket API を直接利用。 | `creative/comfyui` |
| [`ideation`](/docs/user-guide/skills/bundled/creative/creative-creative-ideation) | 創造的な制約を通じてプロジェクトのアイデアを生成。 | `creative/creative-ideation` |
| [`design-md`](/docs/user-guide/skills/bundled/creative/creative-design-md) | Google の DESIGN.md トークン仕様ファイルの作成/検証/エクスポート。 | `creative/design-md` |
| [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) | 手描き風の Excalidraw JSON 図（アーキテクチャ、フロー、シーケンス）。 | `creative/excalidraw` |
| [`humanizer`](/docs/user-guide/skills/bundled/creative/creative-humanizer) | テキストを人間らしく: AI 臭さを除去し、本物の声を加える。 | `creative/humanizer` |
| [`manim-video`](/docs/user-guide/skills/bundled/creative/creative-manim-video) | Manim CE アニメーション: 3Blue1Brown 風の数学/アルゴリズム動画。 | `creative/manim-video` |
| [`p5js`](/docs/user-guide/skills/bundled/creative/creative-p5js) | p5.js スケッチ: ジェネラティブアート、シェーダー、インタラクティブ、3D。 | `creative/p5js` |
| [`pixel-art`](/docs/user-guide/skills/bundled/creative/creative-pixel-art) | 時代別パレット（NES、Game Boy、PICO-8）のピクセルアート。 | `creative/pixel-art` |
| [`popular-web-designs`](/docs/user-guide/skills/bundled/creative/creative-popular-web-designs) | 54 種の実在デザインシステム（Stripe、Linear、Vercel）を HTML/CSS で。 | `creative/popular-web-designs` |
| [`pretext`](/docs/user-guide/skills/bundled/creative/creative-pretext) | @chenglou/pretext を使ったクリエイティブなブラウザデモの構築時に使用 — ASCII アート向けの DOM フリーなテキストレイアウト、障害物を回り込むタイポグラフィのフロー、テキストを幾何学として扱うゲーム、キネティックタイポグラフィ、テキスト駆動のジェネラティブアート。単一ファイルの HT... | `creative/pretext` |
| [`sketch`](/docs/user-guide/skills/bundled/creative/creative-sketch) | 使い捨ての HTML モックアップ: 比較用に 2〜3 種のデザインバリアント。 | `creative/sketch` |
| [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) | 作詞の技術と Suno AI 音楽のプロンプト。 | `creative/songwriting-and-ai-music` |
| [`touchdesigner-mcp`](/docs/user-guide/skills/bundled/creative/creative-touchdesigner-mcp) | twozero MCP 経由で実行中の TouchDesigner インスタンスを制御 — オペレーターの作成、パラメータ設定、接続の配線、Python の実行、リアルタイムビジュアルの構築。36 のネイティブツール。 | `creative/touchdesigner-mcp` |

## data-science

| Skill | 説明 | Path |
|-------|-------------|------|
| [`jupyter-live-kernel`](/docs/user-guide/skills/bundled/data-science/data-science-jupyter-live-kernel) | ライブ Jupyter カーネル（hamelnb）による反復的な Python 実行。 | `data-science/jupyter-live-kernel` |

## devops

| Skill | 説明 | Path |
|-------|-------------|------|
| [`kanban-orchestrator`](/docs/user-guide/skills/bundled/devops/devops-kanban-orchestrator) | Kanban を通じて作業をルーティングするオーケストレータープロファイル向けの分解プレイブック＋誘惑回避ルール。「自分で作業をしない」ルールと基本的なライフサイクルは、すべての kanban ワーカーのシステムプロンプトに自動注入されます。このスキルは... | `devops/kanban-orchestrator` |
| [`kanban-worker`](/docs/user-guide/skills/bundled/devops/devops-kanban-worker) | Hermes Kanban ワーカー向けの落とし穴・例・エッジケース。ライフサイクル自体は KANBAN_GUIDANCE として（agent/prompt_builder.py から）各ワーカーのシステムプロンプトに自動注入されます。このスキルは、より深い詳細が必要なときに読み込むものです... | `devops/kanban-worker` |
| [`webhook-subscriptions`](/docs/user-guide/skills/bundled/devops/devops-webhook-subscriptions) | Webhook サブスクリプション: イベント駆動のエージェント実行。 | `devops/webhook-subscriptions` |

## dogfood

| Skill | 説明 | Path |
|-------|-------------|------|
| [`dogfood`](/docs/user-guide/skills/bundled/dogfood/dogfood-dogfood) | Web アプリの探索的 QA: バグ・証跡・レポートの発見。 | `dogfood` |

## email

| Skill | 説明 | Path |
|-------|-------------|------|
| [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya) | Himalaya CLI: ターミナルから IMAP/SMTP メール。 | `email/himalaya` |

## gaming

| Skill | 説明 | Path |
|-------|-------------|------|
| [`minecraft-modpack-server`](/docs/user-guide/skills/bundled/gaming/gaming-minecraft-modpack-server) | Mod 入り Minecraft サーバーをホスト（CurseForge、Modrinth）。 | `gaming/minecraft-modpack-server` |
| [`pokemon-player`](/docs/user-guide/skills/bundled/gaming/gaming-pokemon-player) | ヘッドレスエミュレータ＋RAM 読み取りで Pokemon をプレイ。 | `gaming/pokemon-player` |

## github

| Skill | 説明 | Path |
|-------|-------------|------|
| [`codebase-inspection`](/docs/user-guide/skills/bundled/github/github-codebase-inspection) | pygount でコードベースを検査: LOC、言語、比率。 | `github/codebase-inspection` |
| [`github-auth`](/docs/user-guide/skills/bundled/github/github-github-auth) | GitHub 認証のセットアップ: HTTPS トークン、SSH 鍵、gh CLI ログイン。 | `github/github-auth` |
| [`github-code-review`](/docs/user-guide/skills/bundled/github/github-github-code-review) | PR レビュー: gh または REST 経由で diff・インラインコメント。 | `github/github-code-review` |
| [`github-issues`](/docs/user-guide/skills/bundled/github/github-github-issues) | gh または REST 経由で GitHub issue の作成・トリアージ・ラベル付け・アサイン。 | `github/github-issues` |
| [`github-pr-workflow`](/docs/user-guide/skills/bundled/github/github-github-pr-workflow) | GitHub PR のライフサイクル: ブランチ、コミット、オープン、CI、マージ。 | `github/github-pr-workflow` |
| [`github-repo-management`](/docs/user-guide/skills/bundled/github/github-github-repo-management) | リポジトリのクローン/作成/フォーク、リモートとリリースの管理。 | `github/github-repo-management` |

## mcp

| Skill | 説明 | Path |
|-------|-------------|------|
| [`native-mcp`](/docs/user-guide/skills/bundled/mcp/mcp-native-mcp) | MCP クライアント: サーバーへの接続、ツールの登録（stdio/HTTP）。 | `mcp/native-mcp` |

## media

| Skill | 説明 | Path |
|-------|-------------|------|
| [`gif-search`](/docs/user-guide/skills/bundled/media/media-gif-search) | curl ＋ jq で Tenor から GIF を検索/ダウンロード。 | `media/gif-search` |
| [`heartmula`](/docs/user-guide/skills/bundled/media/media-heartmula) | HeartMuLa: 歌詞＋タグから Suno 風の楽曲生成。 | `media/heartmula` |
| [`songsee`](/docs/user-guide/skills/bundled/media/media-songsee) | CLI 経由の音声スペクトログラム/特徴量（mel、chroma、MFCC）。 | `media/songsee` |
| [`spotify`](/docs/user-guide/skills/bundled/media/media-spotify) | Spotify: 再生、検索、キュー、プレイリストとデバイスの管理。 | `media/spotify` |
| [`youtube-content`](/docs/user-guide/skills/bundled/media/media-youtube-content) | YouTube の文字起こしを要約・スレッド・ブログに変換。 | `media/youtube-content` |

## mlops

| Skill | 説明 | Path |
|-------|-------------|------|
| [`audiocraft-audio-generation`](/docs/user-guide/skills/bundled/mlops/mlops-models-audiocraft) | AudioCraft: MusicGen テキスト→音楽、AudioGen テキスト→音響。 | `mlops/models/audiocraft` |
| [`dspy`](/docs/user-guide/skills/bundled/mlops/mlops-research-dspy) | DSPy: 宣言的な LM プログラム、プロンプトの自動最適化、RAG。 | `mlops/research/dspy` |
| [`huggingface-hub`](/docs/user-guide/skills/bundled/mlops/mlops-huggingface-hub) | HuggingFace hf CLI: モデル・データセットの検索/ダウンロード/アップロード。 | `mlops/huggingface-hub` |
| [`llama-cpp`](/docs/user-guide/skills/bundled/mlops/mlops-inference-llama-cpp) | llama.cpp によるローカル GGUF 推論＋HF Hub モデル探索。 | `mlops/inference/llama-cpp` |
| [`evaluating-llms-harness`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-lm-evaluation-harness) | lm-eval-harness: LLM のベンチマーク（MMLU、GSM8K など）。 | `mlops/evaluation/lm-evaluation-harness` |
| [`obliteratus`](/docs/user-guide/skills/bundled/mlops/mlops-inference-obliteratus) | OBLITERATUS: LLM の拒否応答を除去（diff-in-means）。 | `mlops/inference/obliteratus` |
| [`segment-anything-model`](/docs/user-guide/skills/bundled/mlops/mlops-models-segment-anything) | SAM: 点・ボックス・マスクによるゼロショット画像セグメンテーション。 | `mlops/models/segment-anything` |
| [`serving-llms-vllm`](/docs/user-guide/skills/bundled/mlops/mlops-inference-vllm) | vLLM: 高スループットな LLM サービング、OpenAI API、量子化。 | `mlops/inference/vllm` |
| [`weights-and-biases`](/docs/user-guide/skills/bundled/mlops/mlops-evaluation-weights-and-biases) | W&B: ML 実験・スイープ・モデルレジストリ・ダッシュボードのログ。 | `mlops/evaluation/weights-and-biases` |

## note-taking

| Skill | 説明 | Path |
|-------|-------------|------|
| [`obsidian`](/docs/user-guide/skills/bundled/note-taking/note-taking-obsidian) | Obsidian vault 内のノートを読む・検索・作成・編集。 | `note-taking/obsidian` |

## productivity

| Skill | 説明 | Path |
|-------|-------------|------|
| [`airtable`](/docs/user-guide/skills/bundled/productivity/productivity-airtable) | curl 経由の Airtable REST API。レコードの CRUD、フィルタ、upsert。 | `productivity/airtable` |
| [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace) | gws CLI または Python で Gmail、Calendar、Drive、Docs、Sheets。 | `productivity/google-workspace` |
| [`linear`](/docs/user-guide/skills/bundled/productivity/productivity-linear) | Linear: GraphQL ＋ curl で issue・プロジェクト・チームを管理。 | `productivity/linear` |
| [`maps`](/docs/user-guide/skills/bundled/productivity/productivity-maps) | OpenStreetMap/OSRM 経由のジオコーディング、POI、経路、タイムゾーン。 | `productivity/maps` |
| [`nano-pdf`](/docs/user-guide/skills/bundled/productivity/productivity-nano-pdf) | nano-pdf CLI（自然言語プロンプト）で PDF のテキスト/誤字/タイトルを編集。 | `productivity/nano-pdf` |
| [`notion`](/docs/user-guide/skills/bundled/productivity/productivity-notion) | curl 経由の Notion API: ページ、データベース、ブロック、検索。 | `productivity/notion` |
| [`ocr-and-documents`](/docs/user-guide/skills/bundled/productivity/productivity-ocr-and-documents) | PDF/スキャンからのテキスト抽出（pymupdf、marker-pdf）。 | `productivity/ocr-and-documents` |
| [`powerpoint`](/docs/user-guide/skills/bundled/productivity/productivity-powerpoint) | .pptx デッキ、スライド、ノート、テンプレートの作成・読み取り・編集。 | `productivity/powerpoint` |
| [`teams-meeting-pipeline`](/docs/user-guide/skills/bundled/productivity/productivity-teams-meeting-pipeline) | Hermes CLI 経由で Teams 会議要約パイプラインを操作 — 会議の要約、パイプラインステータスの確認、ジョブの再実行、Microsoft Graph サブスクリプションの管理。 | `productivity/teams-meeting-pipeline` |

## red-teaming

| Skill | 説明 | Path |
|-------|-------------|------|
| [`godmode`](/docs/user-guide/skills/bundled/red-teaming/red-teaming-godmode) | LLM の脱獄: Parseltongue、GODMODE、ULTRAPLINIAN。 | `red-teaming/godmode` |

## research

| Skill | 説明 | Path |
|-------|-------------|------|
| [`arxiv`](/docs/user-guide/skills/bundled/research/research-arxiv) | キーワード、著者、カテゴリ、ID で arXiv 論文を検索。 | `research/arxiv` |
| [`blogwatcher`](/docs/user-guide/skills/bundled/research/research-blogwatcher) | blogwatcher-cli ツールでブログや RSS/Atom フィードを監視。 | `research/blogwatcher` |
| [`llm-wiki`](/docs/user-guide/skills/bundled/research/research-llm-wiki) | Karpathy の LLM Wiki: 相互リンクされた Markdown ナレッジベースの構築/クエリ。 | `research/llm-wiki` |
| [`polymarket`](/docs/user-guide/skills/bundled/research/research-polymarket) | Polymarket のクエリ: マーケット、価格、板情報、履歴。 | `research/polymarket` |
| [`research-paper-writing`](/docs/user-guide/skills/bundled/research/research-research-paper-writing) | NeurIPS/ICML/ICLR 向けの ML 論文執筆: 設計→投稿。 | `research/research-paper-writing` |

## smart-home

| Skill | 説明 | Path |
|-------|-------------|------|
| [`openhue`](/docs/user-guide/skills/bundled/smart-home/smart-home-openhue) | OpenHue CLI で Philips Hue のライト、シーン、部屋を制御。 | `smart-home/openhue` |

## social-media

| Skill | 説明 | Path |
|-------|-------------|------|
| [`xurl`](/docs/user-guide/skills/bundled/social-media/social-media-xurl) | xurl CLI 経由の X/Twitter: 投稿、検索、DM、メディア、v2 API。 | `social-media/xurl` |

## software-development

| Skill | 説明 | Path |
|-------|-------------|------|
| [`debugging-hermes-tui-commands`](/docs/user-guide/skills/bundled/software-development/software-development-debugging-hermes-tui-commands) | Hermes TUI スラッシュコマンドのデバッグ: Python、ゲートウェイ、Ink UI。 | `software-development/debugging-hermes-tui-commands` |
| [`hermes-agent-skill-authoring`](/docs/user-guide/skills/bundled/software-development/software-development-hermes-agent-skill-authoring) | リポジトリ内 SKILL.md の作成: frontmatter、バリデータ、構造。 | `software-development/hermes-agent-skill-authoring` |
| [`node-inspect-debugger`](/docs/user-guide/skills/bundled/software-development/software-development-node-inspect-debugger) | --inspect ＋ Chrome DevTools Protocol CLI で Node.js をデバッグ。 | `software-development/node-inspect-debugger` |
| [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan) | プランモード: Markdown プランを .hermes/plans/ に書き出し、実行はしない。 | `software-development/plan` |
| [`python-debugpy`](/docs/user-guide/skills/bundled/software-development/software-development-python-debugpy) | Python のデバッグ: pdb REPL ＋ debugpy リモート（DAP）。 | `software-development/python-debugpy` |
| [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) | コミット前レビュー: セキュリティスキャン、品質ゲート、自動修正。 | `software-development/requesting-code-review` |
| [`spike`](/docs/user-guide/skills/bundled/software-development/software-development-spike) | 構築前にアイデアを検証する使い捨ての実験。 | `software-development/spike` |
| [`subagent-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-subagent-driven-development) | delegate_task サブエージェントでプランを実行（2 段階レビュー）。 | `software-development/subagent-driven-development` |
| [`systematic-debugging`](/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging) | 4 フェーズの根本原因デバッグ: 修正前にバグを理解する。 | `software-development/systematic-debugging` |
| [`test-driven-development`](/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development) | TDD: RED-GREEN-REFACTOR の徹底、コードより先にテスト。 | `software-development/test-driven-development` |
| [`writing-plans`](/docs/user-guide/skills/bundled/software-development/software-development-writing-plans) | 実装プランの作成: 細分化されたタスク、パス、コード。 | `software-development/writing-plans` |

## yuanbao

| Skill | 説明 | Path |
|-------|-------------|------|
| [`yuanbao`](/docs/user-guide/skills/bundled/yuanbao/yuanbao-yuanbao) | Yuanbao（元宝）グループ: ユーザーの @メンション、情報/メンバーのクエリ。 | `yuanbao` |
