---
sidebar_position: 12
title: "バッチ処理"
description: "エージェントの軌跡を大規模に生成 — 並列処理、チェックポイント、ツールセット分布"
---

# バッチ処理

バッチ処理を使うと、数百から数千のプロンプトにわたって Hermes エージェントを並列に実行し、構造化された軌跡（trajectory）データを生成できます。これは主に**学習データの生成**に使用されます。ファインチューニングや評価に利用できる、ツール使用統計を含む ShareGPT 形式の軌跡を生成します。

## 概要

バッチランナー（`batch_runner.py`）は、プロンプトの JSONL データセットを処理し、各プロンプトをツールアクセス付きの完全なエージェントセッションを通じて実行します。各プロンプトは独自の隔離された環境を取得します。出力は、完全な会話履歴、ツールコール統計、推論カバレッジメトリクスを含む構造化された軌跡データです。

## クイックスタート

```bash
# 基本的なバッチ実行
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# 中断された実行を再開
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume

# 利用可能なツールセット分布を一覧表示
python batch_runner.py --list_distributions
```

## データセット形式

入力データセットは JSONL ファイル（1 行に 1 つの JSON オブジェクト）です。各エントリには `prompt` フィールドが必要です:

```jsonl
{"prompt": "Write a Python function that finds the longest palindromic substring"}
{"prompt": "Create a REST API endpoint for user authentication using Flask"}
{"prompt": "Debug this error: TypeError: cannot unpack non-iterable NoneType object"}
```

エントリにはオプションで次を含められます:
- `image` または `docker_image`: このプロンプトのサンドボックスで使用するコンテナイメージ（Docker、Modal、Singularity バックエンドで動作）
- `cwd`: タスクのターミナルセッションの作業ディレクトリの上書き

## 設定オプション

| パラメーター | デフォルト | 説明 |
|-----------|---------|------|
| `--dataset_file` | （必須） | JSONL データセットへのパス |
| `--batch_size` | （必須） | バッチあたりのプロンプト数 |
| `--run_name` | （必須） | この実行の名前（出力ディレクトリとチェックポイントに使用） |
| `--distribution` | `"default"` | サンプリング元のツールセット分布 |
| `--model` | `claude-sonnet-4.6` | 使用するモデル |
| `--base_url` | `https://openrouter.ai/api/v1` | API ベース URL |
| `--api_key` | （環境変数） | モデルの API キー |
| `--max_turns` | `10` | プロンプトあたりの最大ツールコールイテレーション数 |
| `--num_workers` | `4` | 並列ワーカープロセス数 |
| `--resume` | `false` | チェックポイントから再開 |
| `--verbose` | `false` | 詳細ログを有効化 |
| `--max_samples` | all | データセットの先頭 N サンプルのみ処理 |
| `--max_tokens` | モデルのデフォルト | モデル応答あたりの最大トークン数 |

### プロバイダールーティング（OpenRouter）

| パラメーター | 説明 |
|-----------|------|
| `--providers_allowed` | 許可するプロバイダーのカンマ区切り（例: `"anthropic,openai"`） |
| `--providers_ignored` | 無視するプロバイダーのカンマ区切り（例: `"together,deepinfra"`） |
| `--providers_order` | 優先するプロバイダー順序のカンマ区切り |
| `--provider_sort` | `"price"`、`"throughput"`、`"latency"` でソート |

### 推論制御

| パラメーター | 説明 |
|-----------|------|
| `--reasoning_effort` | エフォートレベル: `none`、`minimal`、`low`、`medium`、`high`、`xhigh` |
| `--reasoning_disabled` | 推論/思考トークンを完全に無効化 |

### 高度なオプション

| パラメーター | 説明 |
|-----------|------|
| `--ephemeral_system_prompt` | 実行中に使用されるが軌跡には保存されないシステムプロンプト |
| `--log_prefix_chars` | ログプレビューで表示する文字数（デフォルト: 100） |
| `--prefill_messages_file` | few-shot プライミング用のプレフィルメッセージを含む JSON ファイルへのパス |

## ツールセット分布

各プロンプトは、**分布**からランダムにサンプリングされたツールセットのセットを取得します。これにより、学習データが多様なツールの組み合わせをカバーすることが保証されます。`--list_distributions` を使うと、利用可能なすべての分布を確認できます。

現在の実装では、分布は**個々のツールセットそれぞれ**に確率を割り当てます。サンプラーは各ツールセットを独立してフリップし、その後少なくとも 1 つのツールセットが有効になることを保証します。これは、手作業で作成された事前構築済みの組み合わせのテーブルとは異なります。

## 出力形式

すべての出力は `data/<run_name>/` に格納されます:

```text
data/my_run/
├── trajectories.jsonl    # 結合された最終出力（すべてのバッチをマージ）
├── batch_0.jsonl         # 個々のバッチ結果
├── batch_1.jsonl
├── ...
├── checkpoint.json       # 再開用チェックポイント
└── statistics.json       # ツール使用統計の集計
```

### 軌跡形式

`trajectories.jsonl` の各行は JSON オブジェクトです:

```json
{
  "prompt_index": 42,
  "conversations": [
    {"from": "human", "value": "Write a function..."},
    {"from": "gpt", "value": "I'll create that function...",
     "tool_calls": [...]},
    {"from": "tool", "value": "..."},
    {"from": "gpt", "value": "Here's the completed function..."}
  ],
  "metadata": {
    "batch_num": 2,
    "timestamp": "2026-01-15T10:30:00",
    "model": "anthropic/claude-sonnet-4.6"
  },
  "completed": true,
  "partial": false,
  "api_calls": 3,
  "toolsets_used": ["terminal", "file"],
  "tool_stats": {
    "terminal": {"count": 2, "success": 2, "failure": 0},
    "read_file": {"count": 1, "success": 1, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0
  }
}
```

`conversations` フィールドは `from` と `value` フィールドを持つ ShareGPT 風の形式を使用します。ツール統計は、ゼロのデフォルト値ですべての可能なツールを含むように正規化されており、HuggingFace データセット互換のためにエントリ間で一貫したスキーマが保証されます。

## チェックポイント

バッチランナーは耐障害性のための堅牢なチェックポイント機構を備えています:

- **チェックポイントファイル:** 各バッチの完了後に保存され、どのプロンプトインデックスが完了したかを追跡します
- **コンテンツベースの再開:** `--resume` 時、ランナーは既存のバッチファイルをスキャンし、完了したプロンプトを実際のテキストコンテンツ（インデックスだけでなく）でマッチングします。これにより、データセットの順序が変わってもリカバリが可能です
- **失敗したプロンプト:** 正常に完了したプロンプトのみが完了としてマークされます — 失敗したプロンプトは再開時に再試行されます
- **バッチのマージ:** 完了時、すべてのバッチファイル（前回の実行のものを含む）が単一の `trajectories.jsonl` にマージされます

### 再開の仕組み

1. すべての `batch_*.jsonl` ファイルから完了したプロンプトをスキャン（コンテンツマッチングによる）
2. データセットをフィルタリングして、すでに完了したプロンプトを除外
3. 残りのプロンプトを再バッチ化
4. 残りのプロンプトのみを処理
5. すべてのバッチファイル（旧 + 新）を最終出力にマージ

## 品質フィルタリング

バッチランナーは自動的な品質フィルタリングを適用します:

- **推論なしフィルター:** アシスタントターンのいずれにも推論（`<REASONING_SCRATCHPAD>` やネイティブの思考トークン）が含まれないサンプルは破棄されます
- **破損エントリフィルター:** 幻覚されたツール名（有効なツールリストにないもの）を含むエントリは、最終マージ時にフィルタリングされます
- **推論統計:** 実行全体で推論あり/なしのターンの割合を追跡します

## 統計

完了後、ランナーは包括的な統計を出力します:

- **ツール使用:** ツールごとの呼び出し回数、成功/失敗率
- **推論カバレッジ:** 推論を含むアシスタントターンの割合
- **破棄されたサンプル:** 推論を欠いたためにフィルタリングされたサンプル数
- **継続時間:** 総処理時間

統計は、プログラムによる分析のために `statistics.json` にも保存されます。

## ユースケース

### 学習データの生成

ファインチューニング用の多様なツール使用軌跡を生成します:

```bash
python batch_runner.py \
    --dataset_file=data/coding_prompts.jsonl \
    --batch_size=20 \
    --run_name=coding_v1 \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=8 \
    --distribution=default \
    --max_turns=15
```

### モデル評価

標準化されたプロンプト全体でモデルがどれだけうまくツールを使用するかを評価します:

```bash
python batch_runner.py \
    --dataset_file=data/eval_suite.jsonl \
    --batch_size=10 \
    --run_name=eval_gpt4 \
    --model=openai/gpt-4o \
    --num_workers=4 \
    --max_turns=10
```

### プロンプトごとのコンテナイメージ

特定の環境を必要とするベンチマークでは、各プロンプトが独自のコンテナイメージを指定できます:

```jsonl
{"prompt": "Install numpy and compute eigenvalues of a 3x3 matrix", "image": "python:3.11-slim"}
{"prompt": "Compile this Rust program and run it", "image": "rust:1.75"}
{"prompt": "Set up a Node.js Express server", "image": "node:20-alpine", "cwd": "/app"}
```

バッチランナーは、各プロンプトを実行する前に Docker イメージがアクセス可能かどうかを検証します。
