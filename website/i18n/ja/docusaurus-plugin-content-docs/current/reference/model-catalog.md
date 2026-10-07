---
sidebar_position: 11
title: モデルカタログ
description: OpenRouter と Nous Portal 向けの厳選されたモデルピッカーリストを駆動する、リモートホストされたマニフェスト。
---

# モデルカタログ

Hermes は、ドキュメントサイトと並んでホストされている JSON マニフェストから、**OpenRouter** と **Nous Portal** 向けの厳選されたモデルリストを取得します。これにより、メンテナーは新しい `hermes-agent` リリースを出荷することなくピッカーリストを更新できます。

マニフェストに到達できない場合（オフライン、ネットワークブロック、ホスティング障害）、Hermes は CLI に同梱されているリポジトリ内のスナップショットに黙ってフォールバックします。マニフェストがピッカーを壊すことはありません — 最悪の場合でも、インストールされているバージョンにバンドルされていたリストが表示されます。

## ライブマニフェスト URL

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

既存の `deploy-site.yml` GitHub Pages パイプラインを介して、`main` へのマージごとに公開されます。信頼できる情報源は、リポジトリの `website/static/api/model-catalog.json` にあります。

## スキーマ

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2",         "description": "default", "default": true},
        {"id": "moonshotai/kimi-k3",   "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2", "default": true},
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k3"}
      ]
    }
  }
}
```

フィールドに関する注記:

- **`version`** — 整数のスキーマバージョン。将来のスキーマはこれをインクリメントします。Hermes は理解できないバージョンのマニフェストを拒否し、ハードコードされたスナップショットにフォールバックします。
- **`metadata`** — マニフェスト、プロバイダー、モデルの各レベルでの自由形式の辞書。任意のキーを使用できます。Hermes は不明なフィールドを無視するため、スキーマ変更を調整することなくエントリに注釈を付けられます（`"tier": "paid"`、`"tags": [...]` など）。
- **`description`** — OpenRouter のみ。ピッカーのバッジテキスト（`"recommended"`、`"free"`、`"default"`、または空）を駆動します。Nous Portal はこれを使用しません。
- **`default`** — プロバイダーごとに、ちょうど 1 つのエントリだけが `"default": true` を持てます。そのモデルが**サイレントデフォルト**です: ユーザーが一度もモデルを選択しなかった場合（GUI オンボーディングの確認カード、`model` なしで `provider` が設定されている場合、`model.default` が空の場合）に Hermes が使用するモデルです。実行時にはキャッシュのみから読み取られる（`get_default_model_from_cache`）ため、頻繁に通る解決パスがネットワークにアクセスすることはありません。キャッシュされたマニフェストが存在しない場合、Hermes はリポジトリ内の定数 `PREFERRED_SILENT_DEFAULT_MODEL` にフォールバックします。この定数はラベル付けされたエントリと一致している必要があります。これにより、メンテナーはリリースを出さずにサイレントデフォルトを入れ替えられます。意図的に高性能かつ低コストなモデルを選んでおり、最も高価なフラッグシップモデルにはしません。
- **料金とコンテキスト長** はマニフェストに含まれていません。これらは取得時にライブのプロバイダー API（`/v1/models` エンドポイント、models.dev）から取得されます。

## 取得の動作

| タイミング | 動作 |
|---|---|
| `/model` または `hermes model` | ディスクキャッシュが古い場合は取得し、そうでなければキャッシュを使用します |
| ゲートウェイの実行中 | `ttl_minutes`（デフォルト 20）ごとにバックグラウンドで更新するため、ピッカーが公開済みマニフェストから 1 ウィンドウ以上遅れることはありません |
| ディスクキャッシュが新しい（TTL 未満） | ネットワークアクセスなし |
| ネットワーク障害（キャッシュあり） | キャッシュへのサイレントフォールバック、ログ行 1 行 |
| ネットワーク障害（キャッシュなし） | リポジトリ内スナップショットへのサイレントフォールバック |
| マニフェストがスキーマ検証に失敗 | 到達不能として扱われます |

キャッシュの場所: `~/.hermes/cache/model_catalog.json`。

### GUI ピッカーでのプロバイダーごとのモデル一覧

Desktop、TUI、ダッシュボードのピッカー（`model.options`）は、各プロバイダーの行を、ディスクにキャッシュされたライブカタログ（`~/.hermes/provider_models_cache.json`）から、まだ何もキャッシュされていない場合はキュレーション済みリストから構築します。ピッカーを開く際に、プロバイダーの `/v1/models` プローブや認証プローブを待つことはありません: 古いカタログや存在しないカタログはバックグラウンドスレッドで更新され、次に開いたときに反映されます。そのため、遅い、レート制限された、または到達できないプロバイダーが 1 つあっても、ピッカー全体が読み込み状態のまま止まることはありません。**Refresh models**（または `/model --refresh`）は、キャッシュを破棄してすべてのプロバイダーをライブでプローブする明示的な操作です。

## 設定

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_minutes: 20
  providers: {}
```

`enabled: false` に設定すると、リモート取得を完全に無効化し、常にリポジトリ内のスナップショットを使用します（ゲートウェイのバックグラウンド更新も無効になります）。`ttl_minutes` はキャッシュの有効期間とゲートウェイの更新間隔の両方を設定します。旧来の `ttl_hours` キーも、明示的に設定した場合は引き続き尊重されます。

### プロバイダーごとのオーバーライド URL

サードパーティは、同じスキーマを使用して独自のキュレーションリストをセルフホストできます。プロバイダーをカスタム URL に向けます:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

オーバーライドするマニフェストは、関心のあるプロバイダーブロックのみを設定すればよいです。その他のプロバイダーは、引き続きマスター URL に対して解決されます。

### ピッカーからプロバイダーを非表示にする

`excluded_providers` を使うと、有効な認証情報が存在する場合でも、特定のプロバイダーを `/model` ピッカーから非表示にできます。通常の使用では表示されるべきでない旧来のプロバイダーやテスト用プロバイダーの認証情報が残っている場合（例: `auth.json` にまだキャッシュされている、または `gh` CLI 経由で検出された古い Copilot や OpenRouter のトークン）に便利です。

```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai
```

除外は、プロバイダーが現れうるすべてのキー — Hermes id と models.dev id（組み込みのマッピング済みプロバイダー）、オーバーレイの pid と解決済みの Hermes スラッグ（オーバーレイプロバイダー）、正規スラッグ（正規プロバイダー）— に対して大文字小文字を区別せずに照合されます。そのため、`copilot` のような 1 つのエントリで、どのセクションから出力されたかに関係なくそのプロバイダーを非表示にできます。これはすべての `/model` ピッカーで尊重されます: ゲートウェイの対話型/テキストピッカー、TUI ピッカー、対話型の `hermes model` CLI ピッカーです。空のリスト（またはキーの省略）は何の効果もありません。

## マニフェストの更新

メンテナー向け:

```bash
# リポジトリ内のハードコードされたリストから再生成します（hermes_cli/models.py の
# OPENROUTER_MODELS または _PROVIDER_MODELS["nous"] を編集した後、マニフェストを同期状態に保ちます）。
python scripts/build_model_catalog.py
```

その後、結果として生じる変更を `website/static/api/model-catalog.json` に対して `main` へ PR します。ドキュメントサイトはマージ時に自動デプロイされ、新しいマニフェストは数分以内にライブになります。

リポジトリ内のスナップショットに属さないきめ細かいメタデータの変更については、JSON を直接手動編集することもできます — ジェネレータースクリプトは便利なものであり、唯一の信頼できる情報源ではありません。
