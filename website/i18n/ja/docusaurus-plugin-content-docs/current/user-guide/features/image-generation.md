---
title: 画像生成
description: FAL.ai 経由で画像を生成 — FLUX 2、GPT Image（1.5 & 2）、Nano Banana Pro、Ideogram、Recraft V4 Pro などを含む 9 モデルに対応。`hermes tools` で選択可能。
sidebar_label: 画像生成
sidebar_position: 6
---

# 画像生成

Hermes Agent は FAL.ai 経由でテキストプロンプトから画像を生成します。9 つのモデルがそのまま使え、それぞれ速度、品質、コストのトレードオフが異なります。アクティブなモデルは `hermes tools` で設定でき、`config.yaml` に保存されます。

## 対応モデル

| モデル | 速度 | 強み | 価格 |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(既定)* | `<1s` | 高速、くっきりしたテキスト | $0.006/MP |
| `fal-ai/flux-2-pro` | 約 6s | スタジオ品質のフォトリアリズム | $0.03/MP |
| `fal-ai/z-image/turbo` | 約 2s | 英中バイリンガル、6B パラメータ | $0.005/MP |
| `fal-ai/nano-banana-pro` | 約 8s | Gemini 3 Pro、推論の深さ、テキスト描画 | $0.15/画像（1K） |
| `fal-ai/gpt-image-1.5` | 約 15s | プロンプト追従 | $0.034/画像 |
| `fal-ai/gpt-image-2` | 約 20s | SOTA のテキスト描画 + CJK、世界を認識したフォトリアリズム | $0.04〜0.06/画像 |
| `fal-ai/ideogram/v3` | 約 5s | 最高のタイポグラフィ | $0.03〜0.09/画像 |
| `fal-ai/recraft/v4/pro/text-to-image` | 約 8s | デザイン、ブランドシステム、本番対応 | $0.25/画像 |
| `fal-ai/qwen-image` | 約 12s | LLM ベース、複雑なテキスト | $0.02/MP |

価格は執筆時点での FAL の料金です。最新の数値は [fal.ai](https://fal.ai/) で確認してください。

## セットアップ

:::tip Nous サブスクライバー
有料の [Nous Portal](https://portal.nousresearch.com) サブスクリプションをお持ちの場合、FAL API キーなしで **[Tool Gateway](tool-gateway.md)** 経由で画像生成を使えます。モデルの選択はどちらの経路でも引き継がれます。

マネージドゲートウェイが特定のモデルに対して `HTTP 4xx` を返す場合、そのモデルはまだ Portal 側でプロキシされていません。エージェントがその旨と対処手順（直接アクセス用に `FAL_KEY` を設定する、または別のモデルを選ぶ）を伝えます。
:::

### FAL API キーを取得する

1. [fal.ai](https://fal.ai/) でサインアップ
2. ダッシュボードから API キーを生成

### 設定してモデルを選ぶ

tools コマンドを実行します:

```bash
hermes tools
```

**🎨 Image Generation** に移動し、バックエンド（Nous Subscription または FAL.ai）を選びます。すると、対応するすべてのモデルが列揃えのテーブルで表示されます。矢印キーで移動し、Enter で選択します:

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

選択内容は `config.yaml` に保存されます:

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # Nous Subscription を使う場合は true
```

### GPT-Image の品質

`fal-ai/gpt-image-1.5` と `fal-ai/gpt-image-2` のリクエスト品質は `medium` に固定されています（1024×1024 で約 $0.034〜$0.06/画像）。`low` / `high` のティアをユーザー向けオプションとして公開していないのは、Nous Portal の課金を全ユーザーで予測可能に保つためです。ティア間のコスト差は 3〜22 倍に及びます。より安価な選択肢が欲しい場合は Klein 9B または Z-Image Turbo を、より高品質が欲しい場合は Nano Banana Pro または Recraft V4 Pro を選んでください。

## 使い方

エージェント向けのスキーマは意図的に最小限です。モデルは設定した内容をそのまま使います:

```
桜の咲く静かな山の風景の画像を生成して
```

```
賢い老フクロウの正方形のポートレートを作って — タイポグラフィモデルを使って
```

```
未来的な都市景観を、横向きで作って
```

## アスペクト比

すべてのモデルは、エージェントの視点からは同じ 3 つのアスペクト比を受け付けます。内部では、各モデルのネイティブなサイズ指定が自動で埋められます:

| エージェント入力 | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2 が 16:9 ではなく 4:3 のプリセットにマッピングされるのは、最小ピクセル数が 655,360 だからです。`landscape_16_9` プリセット（1024×576 = 589,824）は拒否されてしまいます。

この変換は `_build_fal_payload()` で行われます。エージェントのコードは、モデルごとのスキーマの違いを知る必要がありません。

## 自動アップスケール

FAL の **Clarity Upscaler** によるアップスケールは、モデルごとにゲートされています:

| モデル | アップスケール？ | 理由 |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | 後方互換（ピッカー導入前の既定だった） |
| その他すべて | ✗ | 高速モデルは秒未満の価値が失われる。高解像度モデルは不要 |

アップスケールが実行される場合、次の設定を使います:

| 設定 | 値 |
|---|---|
| アップスケール倍率 | 2× |
| Creativity | 0.35 |
| Resemblance | 0.6 |
| Guidance scale | 4 |
| Inference steps | 18 |

アップスケールが失敗した場合（ネットワークの問題、レート制限）、元の画像が自動的に返されます。

## 内部の仕組み

1. **モデル解決** — `_resolve_fal_model()` が `config.yaml` の `image_gen.model` を読み、`FAL_IMAGE_MODEL` 環境変数、次に `fal-ai/flux-2/klein/9b` にフォールバックします。
2. **ペイロード構築** — `_build_fal_payload()` が、あなたの `aspect_ratio` をモデルのネイティブ形式（プリセット列挙、アスペクト比列挙、または GPT リテラル）に変換し、モデルの既定パラメータをマージし、呼び出し側の上書きを適用し、モデルの `supports` ホワイトリストでフィルタリングして、未対応のキーが決して送られないようにします。
3. **送信** — `_submit_fal_request()` が、ダイレクトな FAL クレデンシャルまたはマネージド Nous ゲートウェイ経由でルーティングします。
4. **アップスケール** — モデルのメタデータが `upscale: True` の場合にのみ実行されます。
5. **配信** — 最終的な画像 URL がエージェントに返され、エージェントが `MEDIA:<url>` タグを出力します。プラットフォームアダプターがこれをネイティブメディアに変換します。

## デバッグ

デバッグログを有効にします:

```bash
export IMAGE_TOOLS_DEBUG=true
```

デバッグログは `./logs/image_tools_debug_<session_id>.json` に出力され、呼び出しごとの詳細（モデル、パラメータ、タイミング、エラー）が記録されます。

## プラットフォームでの配信

| プラットフォーム | 配信 |
|---|---|
| **CLI** | 画像 URL が Markdown の `![](url)` として出力される — クリックで開く |
| **Telegram** | プロンプトをキャプションにした写真メッセージ |
| **Discord** | メッセージに埋め込み |
| **Slack** | Slack によって URL が展開される |
| **WhatsApp** | メディアメッセージ |
| **その他** | プレーンテキストの URL |

## 制限事項

- **FAL クレデンシャルが必要**（ダイレクトな `FAL_KEY` または Nous Subscription）
- **テキストから画像のみ** — このツールでのインペインティング、img2img、編集はできません
- **一時的な URL** — FAL は数時間〜数日で失効するホスト URL を返します。必要ならローカルに保存してください
- **モデルごとの制約** — 一部のモデルは `seed`、`num_inference_steps` などに対応しません。`supports` フィルタが未対応のパラメータを暗黙的に破棄します。これは想定された動作です
