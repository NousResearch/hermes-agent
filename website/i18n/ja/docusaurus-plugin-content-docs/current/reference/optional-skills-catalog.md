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
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | コーディングタスクを Blackbox AI CLI エージェントに委譲。複数の LLM でタスクを実行し最良の結果を選ぶ、組み込みジャッジ付きのマルチモデルエージェント。blackbox CLI と Blackbox AI API キーが必要です。 |
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Hermes で Honcho メモリを設定・利用 — セッションをまたぐユーザーモデリング、マルチプロファイルのピア分離、観測設定、弁証法的推論、セッション要約、コンテキスト予算の強制。Honcho のセットアップやトラブルシュー... |

## blockchain

| Skill | 説明 |
|-------|-------------|
| [**base**](/docs/user-guide/skills/optional/blockchain/blockchain-base) | Base（Ethereum L2）のブロックチェーンデータを USD 価格付きでクエリ — ウォレット残高、トークン情報、トランザクション詳細、ガス分析、コントラクト検査、クジラ検出、リアルタイムのネットワーク統計。Base RPC ＋ CoinGecko を使用。API キー不要。 |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | Solana のブロックチェーンデータを USD 価格付きでクエリ — ウォレット残高、評価額付きトークンポートフォリオ、トランザクション詳細、NFT、クジラ検出、リアルタイムのネットワーク統計。Solana RPC ＋ CoinGecko を使用。API キー不要。 |

## communication

| Skill | 説明 |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | 技術提案とトレードオフ分析のための構造化された意思決定フレームワーク。複数のアプローチ（アーキテクチャ決定、ツール選定、リファクタリング戦略、移行パス）の中から選択を迫られたとき、このスキルが... |

## creative

| Skill | 説明 |
|-------|-------------|
| [**blender-mcp**](/docs/user-guide/skills/optional/creative/creative-blender-mcp) | blender-mcp アドオンへのソケット接続を介して Hermes から Blender を直接制御。3D オブジェクト、マテリアル、アニメーションの作成、任意の Blender Python（bpy）コードの実行。Blender で何かを作成・変更したいときに使用。 |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | フラットでミニマルな、ライト/ダーク対応の SVG 図を単一の HTML ファイルとして生成。9 種のセマンティックカラーランプ、文頭大文字のタイポグラフィ、自動ダークモードを備えた統一的な教育用ビジュアル言語を使用。教育用途や... に最適。 |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | HyperFrames を使って、HTML ベースの動画コンポジション、アニメーションタイトルカード、ソーシャルオーバーレイ、字幕付きトーキングヘッド動画、オーディオリアクティブビジュアル、シェーダートランジションを作成。HTML が動画の真実の源。ユーザーが... を求めるときに使用。 |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | Hermes Kanban を基盤とするマルチエージェント動画制作パイプラインの計画・構築・監視。ユーザーがあらゆる動画 — 物語映画、製品/マーケティング、ミュージックビデオ、解説、ASCII/ターミナルアート、抽象/ジェネラティブのループ... を作りたいときに使用。 |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | テンプレートを選び Pillow でテキストを重ねて、本物のミーム画像を生成。実際の .png ミームファイルを生成します。 |

## devops

| Skill | 説明 |
|-------|-------------|
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-cli) | inference.sh CLI（infsh）で 150 以上の AI アプリを実行 — 画像生成、動画作成、LLM、検索、3D、ソーシャル自動化。terminal ツールを使用。トリガー: inference.sh、infsh、ai apps、flux、veo、画像生成、動画生成、seedrea... |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Docker コンテナ、イメージ、ボリューム、ネットワーク、Compose スタックを管理 — ライフサイクル操作、デバッグ、クリーンアップ、Dockerfile 最適化。 |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | RSS、JSON API、GitHub をウォーターマークによる重複排除付きでポーリング。 |

## dogfood

| Skill | 説明 |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | あなたの製品にとって最も扱いづらく技術に抵抗するユーザーをロールプレイ。そのペルソナとしてアプリを閲覧し、あらゆる UX の痛点を見つけ、苦情を実用主義のレイヤーで濾過して本物の問題とノイズを切り分ける。実行可能なチケットを作成... |

## email

| Skill | 説明 |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | AgentMail を通じてエージェントに専用のメール受信箱を付与。エージェント所有のメールアドレス（例: hermes-agent@agentmail.to）を使って、自律的にメールを送受信・管理。 |

## finance

| Skill | 説明 |
|-------|-------------|
| [**3-statement-model**](/docs/user-guide/skills/optional/finance/finance-3-statement-model) | 運転資本スケジュール、減価償却ロールフォワード、デットスケジュール、現金と利益剰余金を整合させるプラグを備えた、完全統合の 3 表モデル（PL、BS、CF）を Excel で構築。excel-author と組み合わせて使用。 |
| [**comps-analysis**](/docs/user-guide/skills/optional/finance/finance-comps-analysis) | 類似企業分析を Excel で構築 — 営業指標、評価倍率、ピアセットとの統計的ベンチマーク。excel-author と組み合わせて使用。上場企業評価、IPO 価格設定、セクターベンチマーク、外れ値検出に。 |
| [**dcf-model**](/docs/user-guide/skills/optional/finance/finance-dcf-model) | 機関投資家品質の DCF 評価モデルを Excel で構築 — 売上予測、FCF 構築、WACC、ターミナルバリュー、ベア/ベース/ブルシナリオ、5×5 感応度テーブル。excel-author と組み合わせて使用。本源的価値の株式分析に。 |
| [**excel-author**](/docs/user-guide/skills/optional/finance/finance-excel-author) | openpyxl で監査可能な Excel ワークブックをヘッドレスに構築 — 青/黒/緑のセル規約、ハードコードより数式、名前付き範囲、バランスチェック、感応度テーブル。財務モデル、監査出力、照合に。 |
| [**lbo-model**](/docs/user-guide/skills/optional/finance/finance-lbo-model) | レバレッジド・バイアウトモデルを Excel で構築 — 資金調達と使途、デットスケジュール、キャッシュスイープ、エグジット倍率、IRR/MOIC 感応度。excel-author と組み合わせて使用。PE スクリーニング、スポンサーケース評価、ピッチでの例示 LBO に。 |
| [**merger-model**](/docs/user-guide/skills/optional/finance/finance-merger-model) | 希薄化/増益（合併）モデルを Excel で構築 — プロフォーマ PL、シナジー、資金調達構成、EPS への影響。excel-author と組み合わせて使用。M&A ピッチ、取締役会資料、ディール評価に。 |
| [**pptx-author**](/docs/user-guide/skills/optional/finance/finance-pptx-author) | python-pptx で PowerPoint デッキをヘッドレスに構築。excel-author と組み合わせ、すべての数値がワークブックのセルにトレースできるモデル裏付けのデッキに。ピッチデッキ、IC メモ、決算ノートに。 |

## health

| Skill | 説明 |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | ジムのワークアウトプランナーと栄養トラッカー。wger 経由で 690 以上のエクササイズを筋肉・器具・カテゴリで検索。USDA FoodData Central 経由で 380,000 以上の食品のマクロとカロリーを参照。BMI、TDEE、1RM、マクロ配分、体... を計算。 |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | 実行中の NeuroSkill インスタンスに接続し、ユーザーのリアルタイムの認知・感情状態（集中、リラックス、気分、認知負荷、眠気、心拍数、HRV、睡眠ステージング、40 以上の派生 EXG スコア）を応答に取り込む... |

## mcp

| Skill | 説明 |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Python の FastMCP で MCP サーバーを構築・テスト・検査・インストール・デプロイ。新しい MCP サーバーの作成、API やデータベースを MCP ツールとしてラップ、リソースやプロンプトの公開、Claude Code・Cur... 向けの FastMCP サーバー準備時に使用。 |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | mcporter CLI を使って MCP サーバー/ツールを直接（HTTP または stdio で）一覧表示・設定・認証・呼び出し。アドホックなサーバー、設定編集、CLI/型生成を含む。 |

## migration

| Skill | 説明 |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | ユーザーの OpenClaw カスタマイズ資産を Hermes Agent に移行。~/.openclaw から Hermes 互換のメモリ、SOUL.md、コマンド許可リスト、ユーザースキル、選択したワークスペース資産をインポートし、移行できなかった... ものを正確に報告。 |

## mlops

| Skill | 説明 |
|-------|-------------|
| [**huggingface-accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | 最もシンプルな分散学習 API。任意の PyTorch スクリプトに 4 行で分散サポートを追加。DeepSpeed/FSDP/Megatron/DDP の統一 API。自動デバイス配置、混合精度（FP16/BF16/FP8）。対話的な設定、単一の起動コマ... |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: YAML による LLM ファインチューニング（LoRA、DPO、GRPO）。 |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | AI アプリケーション向けのオープンソース埋め込みデータベース。埋め込みとメタデータの保存、ベクトル検索と全文検索、メタデータによるフィルタリング。シンプルな 4 関数 API。ノートブックから本番クラスタまでスケール。セマンティック検索、RAG... に使用。 |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | 視覚と言語を結ぶ OpenAI のモデル。ゼロショット画像分類、画像とテキストのマッチング、クロスモーダル検索を可能にする。4 億の画像とテキストのペアで学習。画像検索、コンテンツモデレーション、視覚言語タスク... に使用。 |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | 密ベクトルの効率的な類似度検索とクラスタリングのための Facebook のライブラリ。数十億のベクトル、GPU アクセラレーション、各種インデックス（Flat、IVF、HNSW）をサポート。高速な k-NN 検索、大規模ベクトル検索、また... に使用。 |
| [**optimizing-attention-flash**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | Flash Attention で Transformer の注意機構を最適化し、2〜4 倍の高速化と 10〜20 倍のメモリ削減を実現。長いシーケンス（512 トークン超）で Transformer を学習/実行する際、注意機構で GPU メモリの問題に遭遇する際、より高速な... が必要な際に使用。 |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | Microsoft Research の制約付き生成フレームワーク Guidance で、正規表現と文法による LLM 出力の制御、妥当な JSON/XML/コードの生成保証、構造化フォーマットの強制、多段階ワークフローの構築を行う。 |
| [**hermes-atropos-environments**](/docs/user-guide/skills/optional/mlops/mlops-hermes-atropos-environments) | Atropos 学習向けの Hermes Agent RL 環境の構築・テスト・デバッグ。HermesAgentBaseEnv インターフェース、報酬関数、エージェントループ統合、ツールを用いた評価、wandb ロギング、3 つの CLI モード（serve/process/eva...）を扱う。 |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | 研究と本番向けに最適化された高速トークナイザー。Rust ベースの実装で 1GB を 20 秒未満でトークナイズ。BPE、WordPiece、Unigram アルゴリズムをサポート。カスタム語彙の学習、アライメントの追跡、パディング/トランケーションの処理。Integ... |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | 実戦で鍛えられた構造化出力ライブラリ Instructor で、Pydantic 検証による LLM 応答からの構造化データ抽出、失敗した抽出の自動リトライ、型安全な複雑 JSON のパース、部分結果のストリーミングを行う。 |
| [**lambda-labs-gpu-cloud**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | ML の学習と推論のための予約型・オンデマンド型 GPU クラウドインスタンス。シンプルな SSH アクセス、永続ファイルシステム、または大規模学習向けの高性能マルチノードクラスタを備えた専用 GPU インスタンスが必要なときに使用。 |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | Large Language and Vision Assistant。視覚的指示チューニングと画像ベースの会話を可能にする。CLIP の視覚エンコーダと Vicuna/LLaMA 言語モデルを組み合わせる。マルチターン画像チャット、視覚的質問応答、指示... をサポート。 |
| [**modal-serverless-gpu**](/docs/user-guide/skills/optional/mlops/mlops-modal) | ML ワークロード実行のためのサーバーレス GPU クラウドプラットフォーム。インフラ管理なしのオンデマンド GPU アクセス、ML モデルの API デプロイ、自動スケーリングを伴うバッチジョブの実行が必要なときに使用。 |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | LLM 学習向けの GPU アクセラレーテッドなデータキュレーション。テキスト/画像/動画/音声をサポート。ファジー重複排除（16 倍高速）、品質フィルタリング（30 以上のヒューリスティック）、セマンティック重複排除、PII マスキング、NSFW 検出を備える。GPU をまたいでスケール... |
| [**outlines**](/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: 構造化された JSON/正規表現/Pydantic による LLM 生成。 |
| [**peft-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-peft) | LoRA、QLoRA、25 以上の手法を用いた LLM のパラメータ効率的ファインチューニング。限られた GPU メモリで大規模モデル（7B〜70B）をファインチューニングする際、精度低下を最小限にパラメータの 1% 未満を学習する必要がある際、マルチアダプタの... に使用。 |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | 本番 AI アプリケーション向けのマネージドベクトルデータベース。フルマネージドで自動スケーリング、ハイブリッド検索（密＋疎）、メタデータフィルタリング、名前空間を備える。低レイテンシ（p95 100ms 未満）。本番 RAG、レコメンドシステム、また... に使用。 |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | PyTorch FSDP による Fully Sharded Data Parallel 学習の専門的ガイダンス — パラメータシャーディング、混合精度、CPU オフロード、FSDP2。 |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | Trainer クラス、自動分散学習（DDP/FSDP/DeepSpeed）、コールバックシステム、最小限のボイラープレートを備えた高レベルの PyTorch フレームワーク。同じコードでラップトップからスーパーコンピュータまでスケール。クリーンな学習ループが欲しいとき... に使用。 |
| [**qdrant-vector-search**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | RAG とセマンティック検索のための高性能ベクトル類似度検索エンジン。高速な最近傍探索を要する本番 RAG システム、フィルタリング付きハイブリッド検索、または Rust 製の高性能なスケーラブルなベクトルストレージを構築する際に使用。 |
| [**sparse-autoencoder-training**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | SAELens を用いてニューラルネットワークの活性化を解釈可能な特徴に分解する、Sparse Autoencoder（SAE）の学習と分析のガイダンスを提供。解釈可能な特徴の発見、重ね合わせの分析、または研究... の際に使用。 |
| [**simpo-training**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | LLM アライメントのための Simple Preference Optimization。DPO のリファレンス不要な代替で、より高い性能（AlpacaEval 2.0 で +6.4 ポイント）。リファレンスモデル不要で DPO より効率的。よりシンプルな... を求めるときの選好アライメントに使用。 |
| [**slime-rl-training**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Megatron+SGLang フレームワーク slime を用いた RL による LLM ポストトレーニングのガイダンスを提供。GLM モデルの学習、カスタムデータ生成ワークフローの実装、または RL スケーリングのための緊密な Megatron-LM 統合が必要なときに使用。 |
| [**stable-diffusion-image-generation**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | HuggingFace Diffusers 経由の Stable Diffusion モデルによる最先端のテキスト→画像生成。テキストプロンプトからの画像生成、image-to-image 変換、インペインティング、またはカスタム拡散パイプラインの構築時に使用。 |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | NVIDIA TensorRT で LLM 推論を最適化し、最大スループットと最低レイテンシを実現。NVIDIA GPU（A100/H100）での本番デプロイ、PyTorch より 10〜100 倍高速な推論が必要なとき、または量子化付きモデルのサービングに使用。 |
| [**distributed-llm-pretraining-torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | torchtitan を用いた 4D 並列（FSDP2、TP、PP、CP）による PyTorch ネイティブの分散 LLM 事前学習を提供。Llama 3.1、DeepSeek V3、またはカスタムモデルを 8 から 512 以上の GPU で、Float8、torch.compile、dist... を用いて大規模に事前学習する際に使用。 |
| [**fine-tuning-with-trl**](/docs/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: LLM RLHF のための SFT、DPO、PPO、GRPO、報酬モデリング。 |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | Unsloth: 2〜5 倍高速な LoRA/QLoRA ファインチューニング、より少ない VRAM。 |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | OpenAI の汎用音声認識モデル。99 言語、文字起こし、英語への翻訳、言語識別をサポート。tiny（3900 万パラメータ）から large（15 億 5000 万パラメータ）まで 6 つのモデルサイズ。音声テキスト変換、ポッドキャスト... に使用。 |

## productivity

| Skill | 説明 |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | Canvas LMS 連携 — API トークン認証を使って履修中のコースと課題を取得。 |
| [**here.now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | 静的サイトを &#123;slug&#125;.here.now に公開し、エージェント間の受け渡しのためにプライベートファイルをクラウド Drive に保存。 |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | 間隔反復のフラッシュカードシステム。事実やテキストからカードを作成し、エージェントが採点する自由記述の回答でフラッシュカードと対話し、YouTube の文字起こしからクイズを生成し、適応的スケジューリングで期限のカードを復習し、エクスポート/イン... |
| [**shop-app**](/docs/user-guide/skills/optional/productivity/productivity-shop-app) | Shop.app: 商品検索、注文追跡、返品、再注文。 |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | curl 経由の Shopify Admin & Storefront GraphQL API。商品、注文、顧客、在庫、メタフィールド。 |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | curl 経由でセルフホストのナレッジベース内のブロックとドキュメントを検索・読み取り・作成・管理する SiYuan Note API。 |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | コアツールを変更せずに Hermes に電話機能を付与。Twilio 番号のプロビジョニングと永続化、SMS/MMS の送受信、直接通話、Bland.ai または Vapi 経由の AI 駆動アウトバウンド通話。 |

## research

| Skill | 説明 |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | bioSkills と ClawBio による 400 以上のバイオインフォマティクススキルへのゲートウェイ。ゲノミクス、トランスクリプトミクス、シングルセル、バリアントコール、ファーマコゲノミクス、メタゲノミクス、構造生物学などをカバー。ドメイン固有の参照資料を取得... |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | Python 標準ライブラリを用いた受動的なドメイン偵察。サブドメイン探索、SSL 証明書検査、WHOIS 照会、DNS レコード、ドメイン利用可否チェック、一括マルチドメイン分析。API キー不要。 |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | 創薬ワークフロー向けの製薬研究アシスタント。ChEMBL で生理活性化合物を検索、薬らしさ（Lipinski Ro5、QED、TPSA、合成容易性）を計算、OpenFDA 経由で薬物間相互作用を照会、ADMET を解釈... |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | DuckDuckGo 経由の無料 Web 検索 — テキスト、ニュース、画像、動画。API キー不要。インストール済みなら `ddgs` CLI を優先。現在のランタイムで `ddgs` が利用可能であることを確認してから Python の DDGS ライブラリを使用すること。 |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | GitNexus でコードベースをインデックス化し、Web UI ＋ Cloudflare トンネル経由で対話的なナレッジグラフを提供。 |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | Parallel CLI 向けのオプションベンダースキル — エージェントネイティブな Web 検索、抽出、ディープリサーチ、エンリッチメント、FindAll、監視。JSON 出力と非対話フローを優先。 |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | BM25、ベクトル検索、LLM リランキングを備えたハイブリッド検索エンジン qmd を使って、個人のナレッジベース、ノート、ドキュメント、会議の文字起こしをローカルで検索。CLI と MCP 統合をサポート。 |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | Scrapling による Web スクレイピング — CLI と Python による HTTP 取得、ステルスブラウザ自動化、Cloudflare バイパス、スパイダークロール。 |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | SearXNG 経由の無料メタ検索 — 70 以上の検索エンジンの結果を集約。セルフホストまたは公開インスタンスを利用。API キー不要。web search ツールセットが利用できないとき自動でフォールバック。 |

## security

| Skill | 説明 |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | 1Password CLI（op）のセットアップと利用。CLI のインストール、デスクトップアプリ統合の有効化、サインイン、コマンドへのシークレットの読み取り/注入時に使用。 |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | GitHub リポジトリのサプライチェーン調査、証拠復旧、フォレンジック分析。削除されたコミットの復旧、force-push の検出、IOC 抽出、複数ソースの証拠収集、仮説の形成/検証、st... をカバー。 |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | 400 以上のソーシャルネットワークをまたぐユーザー名の OSINT 検索。ユーザー名でソーシャルメディアアカウントを追跡。 |

## web-development

| Skill | 説明 |
|-------|-------------|
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | alibaba/page-agent を自分の Web アプリケーションに埋め込む — 単一の &lt;script> タグまたは npm パッケージとして提供される純粋な JavaScript のページ内 GUI エージェントで、サイトのエンドユーザーが自然言語（「login をクリック、userna... を入力」）で UI を操作できる。 |

---

## オプションスキルへの貢献

リポジトリに新しいオプションスキルを追加するには:

1. `optional-skills/<category>/<skill-name>/` 配下にディレクトリを作成
2. 標準的な frontmatter（name、description、version、author）を持つ `SKILL.md` を追加
3. 補助ファイルは `references/`、`templates/`、`scripts/` サブディレクトリに配置
4. プルリクエストを送信 — マージされると、スキルはこのカタログに表示され、専用のドキュメントページが付与されます
