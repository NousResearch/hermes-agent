---
sidebar_position: 9
title: "オプションスキルカタログ"
description: "hermes-agent に同梱される公式オプションスキル — hermes skills install official/<category>/<skill> でインストール"
---

# オプションスキルカタログ

オプションスキルは hermes-agent に `optional-skills/` 配下で同梱されますが、**デフォルトでは有効化されていません**。次のように明示的にインストールしてください:

```bash
hermes skills install official/<category>/<skill>
```

例:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

以下の各スキルは、その完全な定義・セットアップ・使い方を記載した専用ページにリンクしています。

アンインストールするには:

```bash
hermes skills uninstall <skill-name>
```

## autonomous-ai-agents

| Skill | 説明 |
|-------|-------------|
| [**agent-merge-conflict-arbiter**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-agent-merge-conflict-arbiter) | 2 つのエージェント間のマージコンフリクトを裁定する中立な仲裁役。 |
| [**antigravity-cli**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-antigravity-cli) | Antigravity CLI（agy）を操作: プラグイン、認証、サンドボックス。 |
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | コーディングタスクを Blackbox AI のマルチモデル CLI に委譲。 |
| [**dynamic-workflow**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-dynamic-workflow) | コードで計画するファンアウト、敵対的検証、ウェーブ実行。 |
| [**grok**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok) | コーディングを xAI Grok Build CLI に委譲（機能、PR）。 |
| [**opencode**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-opencode) | コーディングを OpenCode CLI に委譲（機能、PR レビュー）。 |
| [**openhands**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-openhands) | コーディングを OpenHands CLI に委譲（モデル非依存、LiteLLM）。 |

## blockchain

| Skill | 説明 |
|-------|-------------|
| [**evm**](/docs/user-guide/skills/optional/blockchain/blockchain-evm) | 読み取り専用の EVM クライアント: 8 チェーンのウォレット、トークン、ガス。 |
| [**hyperliquid**](/docs/user-guide/skills/optional/blockchain/blockchain-hyperliquid) | Hyperliquid のマーケットデータ、アカウント履歴、トレードのレビュー。 |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | Solana のウォレット、トークン、トランザクション、NFT を USD 換算でクエリ。 |

## communication

| Skill | 説明 |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | 1-3-1 の意思決定ブリーフ: 問題、3 つの選択肢、1 つの推奨。 |

## creative

| Skill | 説明 |
|-------|-------------|
| [**ai-presenter-video**](/docs/user-guide/skills/optional/creative/creative-ai-presenter-video) | 台本＋画像から検証済みの AI プレゼンター動画を作成。 |
| [**anatomy**](/docs/user-guide/skills/optional/creative/creative-anatomy) | アイデアをインタラクティブなアイソメトリックの機械図として解説。 |
| [**archify**](/docs/user-guide/skills/optional/creative/creative-archify) | 検証済みのインタラクティブな HTML 図（アップストリームで保守）。 |
| [**ascii-art**](/docs/user-guide/skills/optional/creative/creative-ascii-art) | ASCII アート: pyfiglet、cowsay、boxes、画像から ASCII への変換。 |
| [**auteur**](/docs/user-guide/skills/optional/creative/creative-auteur) | 映画的で受賞レベルの Web ページをデザイン・構築。 |
| [**baoyu-article-illustrator**](/docs/user-guide/skills/optional/creative/creative-baoyu-article-illustrator) | 記事用イラスト: タイプ × スタイル × パレットの一貫性。 |
| [**baoyu-comic**](/docs/user-guide/skills/optional/creative/creative-baoyu-comic) | ナレッジコミック（知识漫画）: 教育、伝記、チュートリアル。 |
| [**brag**](/docs/user-guide/skills/optional/creative/creative-brag) | Hyperframes によるプロジェクトのローンチ動画（アップストリームで保守）。 |
| [**brag-slim**](/docs/user-guide/skills/optional/creative/creative-brag-slim) | プロジェクトまたは URL からローンチ動画を作成（アップストリームで保守）。 |
| [**comfyui**](/docs/user-guide/skills/optional/creative/creative-comfyui) | 拡散ワークフローで画像・動画・音声を生成。 |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | フラットでミニマルな教育用 SVG ビジュアルを HTML として生成。 |
| [**creative-ideation**](/docs/user-guide/skills/optional/creative/creative-creative-ideation) | クリエイティブ実践の名前付き手法でアイデアを生成。 |
| [**design-md**](/docs/user-guide/skills/optional/creative/creative-design-md) | Google の DESIGN.md トークン仕様ファイルの作成/検証/エクスポート。 |
| [**draw-your-font**](/docs/user-guide/skills/optional/creative/creative-draw-your-font) | 手書きの写真をインストール可能な TTF フォントに変換。 |
| [**dream-loop**](/docs/user-guide/skills/optional/creative/creative-dream-loop) | コンセプトアートとの忠実度ループで見事な 3D シーンを構築。 |
| [**excalidraw**](/docs/user-guide/skills/optional/creative/creative-excalidraw) | 手描き風の Excalidraw JSON 図（アーキテクチャ、フロー、シーケンス）。 |
| [**heartmula**](/docs/user-guide/skills/optional/creative/creative-heartmula) | HeartMuLa: 歌詞＋タグから Suno 風の楽曲生成。 |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | HTML コンポジションから MP4/WebM 動画をレンダリング。 |
| [**impeccable**](/docs/user-guide/skills/optional/creative/creative-impeccable) | フロントエンドデザインのガイダンス（アップストリームで保守、impeccable）。 |
| [**ip-as-logo**](/docs/user-guide/skills/optional/creative/creative-ip-as-logo) | 32px でも判読できる、ミニマルでかわいい IP マスコットマークをデザイン。 |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | マルチエージェントの動画制作パイプラインを計画・実行。 |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | テンプレートと Pillow のテキストオーバーレイでミーム PNG を作成。 |
| [**mono-color**](/docs/user-guide/skills/optional/creative/creative-mono-color) | 1 色または 2 色刷りのエディトリアルなポスター画像を生成。 |
| [**pixel-art**](/docs/user-guide/skills/optional/creative/creative-pixel-art) | 時代別パレット（NES、Game Boy、PICO-8）のピクセルアート。 |
| [**pretext**](/docs/user-guide/skills/optional/creative/creative-pretext) | DOM フリーなテキストレイアウトでクリエイティブなブラウザデモを構築。 |
| [**simple-english**](/docs/user-guide/skills/optional/creative/creative-simple-english) | テキストを ASD-STE100 Simplified Technical English に書き換え。 |
| [**sketch**](/docs/user-guide/skills/optional/creative/creative-sketch) | 使い捨ての HTML モックアップ: 比較用に 2〜3 種のデザインバリアント。 |
| [**social-media-content-calendar**](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) | マルチプラットフォームの SNS キャンペーンを計画: ブリーフから投稿まで。 |
| [**system-atlas**](/docs/user-guide/skills/optional/creative/creative-system-atlas) | 探索可能なアイソメトリックのアーキテクチャアトラスを HTML として構築。 |
| [**tldraw-offline**](/docs/user-guide/skills/optional/creative/creative-tldraw-offline) | エージェントで tldraw のオフラインキャンバスを操作・スクリプト化。 |
| [**unreal-mcp**](/docs/user-guide/skills/optional/creative/creative-unreal-mcp) | Unreal Engine エディタのシーン、アクター、レンダリングを自動化。 |

## data-science

| Skill | 説明 |
|-------|-------------|
| [**jupyter-notebook**](/docs/user-guide/skills/optional/data-science/data-science-jupyter-notebook) | ライブ Jupyter カーネル（hamelnb）による反復的な Python 実行。 |

## devops

| Skill | 説明 |
|-------|-------------|
| [**actual-setup**](/docs/user-guide/skills/optional/devops/devops-actual-setup) | Hermes で Actual Computer（actual.inc）の推論をセットアップ。 |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Docker のコンテナ、イメージ、ボリューム、Compose を管理。 |
| [**hermes-s6-container-supervision**](/docs/user-guide/skills/optional/devops/devops-hermes-s6-container-supervision) | Hermes Docker イメージ内の s6 サービスを変更・デバッグ。 |
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-inference-sh-cli) | inference.sh CLI で 150 以上の AI アプリ（画像、動画、LLM）を実行。 |
| [**pinggy-tunnel**](/docs/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Pinggy による SSH 経由のインストール不要な localhost トンネル。 |
| [**setup-wizard-generator**](/docs/user-guide/skills/optional/devops/devops-setup-wizard-generator) | 人間による手動セットアップを案内する bash ウィザードを生成。 |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | RSS、JSON API、GitHub をウォーターマークによる重複排除付きでポーリング。 |

## dogfood

| Skill | 説明 |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | 敵対的なユーザーをロールプレイし、UX の痛点を発見・トリアージ。 |

## email

| Skill | 説明 |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | エージェントに AgentMail CLI のメール受信箱が必要なときに使用。 |

## finance

| Skill | 説明 |
|-------|-------------|
| [**3-statement-model**](/docs/user-guide/skills/optional/finance/finance-3-statement-model) | 統合された PL/BS/CF の財務ワークブックを Excel で構築。 |
| [**comps-analysis**](/docs/user-guide/skills/optional/finance/finance-comps-analysis) | 類似企業比較による評価ワークブックを Excel で構築。 |
| [**dcf-model**](/docs/user-guide/skills/optional/finance/finance-dcf-model) | DCF（割引キャッシュフロー）評価ワークブックを Excel で構築。 |
| [**excel-author**](/docs/user-guide/skills/optional/finance/finance-excel-author) | openpyxl で監査可能な財務ワークブックをヘッドレスに構築。 |
| [**lbo-model**](/docs/user-guide/skills/optional/finance/finance-lbo-model) | IRR/MOIC 付きのレバレッジド・バイアウトのワークブックを Excel で構築。 |
| [**merger-model**](/docs/user-guide/skills/optional/finance/finance-merger-model) | M&A の増益/希薄化ワークブックを Excel で構築。 |
| [**polymarket**](/docs/user-guide/skills/optional/finance/finance-polymarket) | Polymarket のクエリ: マーケット、価格、板情報、履歴。 |
| [**pptx-author**](/docs/user-guide/skills/optional/finance/finance-pptx-author) | python-pptx で PowerPoint デッキをヘッドレスに構築。 |
| [**stocks**](/docs/user-guide/skills/optional/finance/finance-stocks) | Yahoo 経由の株価、履歴、検索、比較、暗号資産。 |

## gaming

| Skill | 説明 |
|-------|-------------|
| [**minecraft-modpack-server**](/docs/user-guide/skills/optional/gaming/gaming-minecraft-modpack-server) | Mod 入り Minecraft サーバーをホスト（CurseForge、Modrinth）。 |
| [**pokemon-player**](/docs/user-guide/skills/optional/gaming/gaming-pokemon-player) | ヘッドレスエミュレータ＋RAM 読み取りで Pokemon をプレイ。 |

## health

| Skill | 説明 |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | wger/USDA によるワークアウト計画、マクロ、身体指標。 |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | NeuroSkill からのライブな BCI の認知・気分状態を活用。 |

## mcp

| Skill | 説明 |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Python の MCP サーバーを構築・テスト・デプロイ。 |
| [**mcp-oauth-remote-gateway**](/docs/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway) | ヘッドレスなゲートウェイ上のリモート MCP サーバー向けの手動 OAuth。 |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | ターミナルから MCP サーバー/ツールを一覧表示・認証・呼び出し。 |

## media

| Skill | 説明 |
|-------|-------------|
| [**gif-search**](/docs/user-guide/skills/optional/media/media-gif-search) | curl ＋ jq で Tenor から GIF を検索/ダウンロード。 |
| [**songsee**](/docs/user-guide/skills/optional/media/media-songsee) | CLI 経由の音声スペクトログラム/特徴量（mel、chroma、MFCC）。 |

## mlops

| Skill | 説明 |
|-------|-------------|
| [**accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | 最小限の変更で PyTorch の学習を複数 GPU で実行。 |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: YAML による LLM ファインチューニング（LoRA、DPO、GRPO）。 |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | RAG とセマンティック検索のための埋め込みデータベース。 |
| [**dspy**](/docs/user-guide/skills/optional/mlops/mlops-research-dspy) | DSPy: 宣言的な LM プログラム、プロンプトの自動最適化、RAG。 |
| [**evaluating-llms-harness**](/docs/user-guide/skills/optional/mlops/mlops-evaluation-evaluating-llms-harness) | lm-eval-harness: LLM のベンチマーク（MMLU、GSM8K など）。 |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | 数十億規模の高速なベクトル類似度検索。 |
| [**flash-attention**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | 長いシーケンスでの Transformer の学習と推論を高速化。 |
| [**lambda-labs**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | ML 学習向けのオンデマンド GPU クラウドインスタンス。 |
| [**llama-cpp**](/docs/user-guide/skills/optional/mlops/mlops-inference-llama-cpp) | llama.cpp によるローカル GGUF 推論＋HF Hub モデル探索。 |
| [**modal**](/docs/user-guide/skills/optional/mlops/mlops-modal) | ML ジョブとモデル API のためのサーバーレス GPU クラウド。 |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | LLM 学習データのキュレーション: 重複排除、フィルタリング、PII マスキング。 |
| [**obliteratus**](/docs/user-guide/skills/optional/mlops/mlops-obliteratus) | OBLITERATUS: LLM の拒否応答を除去（diff-in-means）。 |
| [**peft**](/docs/user-guide/skills/optional/mlops/mlops-peft) | 限られた GPU メモリで大規模 LLM を LoRA でファインチューニング。 |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | 大規模モデル向けの Fully Sharded Data Parallel 学習。 |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | 分散サポートを内蔵したクリーンな学習ループ。 |
| [**saelens**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | モデルの特徴を解釈するための Sparse Autoencoder を学習。 |
| [**segment-anything-model**](/docs/user-guide/skills/optional/mlops/mlops-models-segment-anything-model) | SAM: 点・ボックス・マスクによるゼロショット画像セグメンテーション。 |
| [**serving-llms-vllm**](/docs/user-guide/skills/optional/mlops/mlops-inference-serving-llms-vllm) | vLLM: 高スループットな LLM サービング、OpenAI API、量子化。 |
| [**simpo**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | リファレンス不要な選好アライメント。DPO よりシンプル。 |
| [**slime**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Megatron と SGLang による LLM の RL ポストトレーニング。 |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | NVIDIA GPU での高スループットな LLM 推論。 |
| [**torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | PyTorch の 4D 並列で LLM を大規模に事前学習。 |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: 2〜5 倍高速な LoRA/QLoRA ファインチューニング、より少ない VRAM。 |

## payments

| Skill | 説明 |
|-------|-------------|
| [**stripe-link-cli**](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli) | Stripe Link によるエージェント決済 — カード、SPT、承認。 |
| [**stripe-projects**](/docs/user-guide/skills/optional/payments/payments-stripe-projects) | Stripe Projects で SaaS サービスをプロビジョニングし認証情報を同期。 |

## productivity

| Skill | 説明 |
|-------|-------------|
| [**box**](/docs/user-guide/skills/optional/productivity/productivity-box) | Box でクラウドファイル、共有、検索、メタデータを管理。 |
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | API トークンで Canvas LMS のコースと課題を取得。 |
| [**decision-questionnaire**](/docs/user-guide/skills/optional/productivity/productivity-decision-questionnaire) | 答えの出ない意思決定を質問票ドキュメントに変換。 |
| [**here-now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | サイトを &#123;slug&#125;.here.now に公開し、ファイルを Drive に保存。 |
| [**live-dashboard**](/docs/user-guide/skills/optional/productivity/productivity-live-dashboard) | ライブなソースから自己更新するダッシュボードを構築。 |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | 間隔反復のフラッシュカード: 作成、復習、クイズ、エクスポート。 |
| [**property-listings**](/docs/user-guide/skills/optional/productivity/productivity-property-listings) | 不動産・賃貸物件の情報をデスクトップカードとして表示。 |
| [**shop**](/docs/user-guide/skills/optional/productivity/productivity-shop) | Shop のカタログ検索、チェックアウト、注文追跡、返品。 |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | API 経由で SiYuan ナレッジベースをクエリ・編集。 |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | Twilio 番号のプロビジョニング、SMS/MMS、AI によるアウトバウンド通話。 |

## research

| Skill | 説明 |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | 400 以上のゲノミクス・計算生物学スキルへのゲートウェイ。 |
| [**blogwatcher**](/docs/user-guide/skills/optional/research/research-blogwatcher) | blogwatcher-cli ツールでブログや RSS/Atom フィードを監視。 |
| [**darwinian-evolver**](/docs/user-guide/skills/optional/research/research-darwinian-evolver) | Imbue の進化ループでプロンプト/正規表現/SQL/コードを進化させる。 |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | サブドメイン、SSL 証明書、WHOIS、DNS の受動的な偵察。 |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | 創薬: ChEMBL 検索、薬らしさ、相互作用。 |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | インタラクティブなコードベースのナレッジグラフ Web UI を提供。 |
| [**llm-wiki**](/docs/user-guide/skills/optional/research/research-llm-wiki) | Karpathy の LLM Wiki: 相互リンクされた Markdown ナレッジベースの構築/クエリ。 |
| [**osint-investigation**](/docs/user-guide/skills/optional/research/research-osint-investigation) | 公開記録と制裁データから資金の流れを追跡。 |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | エージェントネイティブな Web 検索、ディープリサーチ、エンリッチメント。 |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | ノート、ドキュメント、文字起こしを対象としたハイブリッドなローカル検索。 |
| [**research-paper-writing**](/docs/user-guide/skills/optional/research/research-research-paper-writing) | NeurIPS/ICML/ICLR 向けの ML 論文執筆: 設計→投稿。 |
| [**rss-feeds**](/docs/user-guide/skills/optional/research/research-rss-feeds) | RSS、Atom、JSON フィードを読み、ページの背後にあるフィードを発見。 |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | ステルスブラウジングと Cloudflare バイパスでサイトをスクレイピング。 |

## security

| Skill | 説明 |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | op CLI のセットアップ、サインイン、シークレットの読み取りや注入。 |
| [**godmode**](/docs/user-guide/skills/optional/security/security-godmode) | LLM の脱獄: Parseltongue、GODMODE、ULTRAPLINIAN。 |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | GitHub サプライチェーンのフォレンジック: 復旧、IOC、レポート。 |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | 400 以上のプラットフォームでユーザー名のアカウントを検索。 |
| [**unbroker**](/docs/user-guide/skills/optional/security/security-unbroker) | 名前や電話番号をもとにデータブローカーサイトから個人情報を削除。 |
| [**web-pentest**](/docs/user-guide/skills/optional/security/security-web-pentest) | 許可された Web ペネトレーションテスト: 偵察、証拠ベースのエクスプロイト、レポート。 |

## smart-home

| Skill | 説明 |
|-------|-------------|
| [**openhue**](/docs/user-guide/skills/optional/smart-home/smart-home-openhue) | OpenHue CLI で Philips Hue のライト、シーン、部屋を制御。 |

## social-media

| Skill | 説明 |
|-------|-------------|
| [**reddit-reading**](/docs/user-guide/skills/optional/social-media/social-media-reddit-reading) | Reddit を読む: サブレディット、検索、スレッド、ユーザー。ブラウザ不要。 |

## software-development

| Skill | 説明 |
|-------|-------------|
| [**ast-grep**](/docs/user-guide/skills/optional/software-development/software-development-ast-grep) | ast-grep による AST を考慮した構造的なコード検索と書き換え。 |
| [**code-wiki**](/docs/user-guide/skills/optional/software-development/software-development-code-wiki) | 任意のコードベースの Wiki ドキュメント＋Mermaid 図を生成。 |
| [**grill-me**](/docs/user-guide/skills/optional/software-development/software-development-grill-me) | 実装前に計画を厳しく問い詰める敵対的インタビュー。 |
| [**hermes-agent-skill-authoring**](/docs/user-guide/skills/optional/software-development/software-development-hermes-agent-skill-authoring) | リポジトリ内 SKILL.md の作成: frontmatter と構造。 |
| [**inspecting-hermes-desktop-dom**](/docs/user-guide/skills/optional/software-development/software-development-inspecting-hermes-desktop-dom) | CDP 経由でライブな Hermes デスクトップの DOM/CSS を読み取る。 |
| [**node-inspect-debugger**](/docs/user-guide/skills/optional/software-development/software-development-node-inspect-debugger) | --inspect ＋ Chrome DevTools Protocol CLI で Node.js をデバッグ。 |
| [**pr-lens**](/docs/user-guide/skills/optional/software-development/software-development-pr-lens) | コード変更をアニメーション付きのアーキテクチャ/データフロー SVG として描画。 |
| [**python-debugpy**](/docs/user-guide/skills/optional/software-development/software-development-python-debugpy) | Python のデバッグ: pdb REPL ＋ debugpy リモート（DAP）。 |
| [**rest-graphql-debug**](/docs/user-guide/skills/optional/software-development/software-development-rest-graphql-debug) | REST/GraphQL API のデバッグ: ステータスコード、認証、スキーマ、再現。 |
| [**subagent-driven-development**](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development) | delegate_task サブエージェントでプランを実行（2 段階レビュー）。 |

## web-development

| Skill | 説明 |
|-------|-------------|
| [**cloudflare-temporary-deploy**](/docs/user-guide/skills/optional/web-development/web-development-cloudflare-temporary-deploy) | wrangler --temporary でアカウント不要のまま Worker をライブデプロイ。 |
| [**har-derived-api-client**](/docs/user-guide/skills/optional/web-development/web-development-har-derived-api-client) | サイトの XHR を HAR に記録し、HTTP クライアントを導出。 |
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | Web アプリにページ内の自然言語 GUI コパイロットを埋め込む。 |
| [**publish-site**](/docs/user-guide/skills/optional/web-development/web-development-publish-site) | GitHub/Cloudflare/Netlify Pages へのバージョン管理されたサイトデプロイ。 |
| [**scrollcraft**](/docs/user-guide/skills/optional/web-development/web-development-scrollcraft) | スクロール駆動のプレミアムなランディングページ。スクロール＝タイムライン。 |

---

## オプションスキルへの貢献

リポジトリに新しいオプションスキルを追加するには:

1. `optional-skills/<category>/<skill-name>/` 配下にディレクトリを作成
2. 標準的な frontmatter（name、description、version、author）を持つ `SKILL.md` を追加
3. 補助ファイルは `references/`、`templates/`、`scripts/` サブディレクトリに配置
4. プルリクエストを送信 — マージされると、スキルはこのカタログに表示され、専用のドキュメントページが付与されます
