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
        {"id": "moonshotai/kimi-k2.6", "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k2.6"}
      ]
    }
  }
}
```

フィールドに関する注記:

- **`version`** — 整数のスキーマバージョン。将来のスキーマはこれをインクリメントします。Hermes は理解できないバージョンのマニフェストを拒否し、ハードコードされたスナップショットにフォールバックします。
- **`metadata`** — マニフェスト、プロバイダー、モデルの各レベルでの自由形式の辞書。任意のキーを使用できます。Hermes は不明なフィールドを無視するため、スキーマ変更を調整することなくエントリに注釈を付けられます（`"tier": "paid"`、`"tags": [...]` など）。
- **`description`** — OpenRouter のみ。ピッカーのバッジテキスト（`"recommended"`、`"free"`、または空）を駆動します。Nous Portal はこれを使用しません — 無料ティアのゲーティングは、Portal の料金エンドポイントからライブで決定されます。
- **料金とコンテキスト長** はマニフェストに含まれていません。これらは取得時にライブのプロバイダー API（`/v1/models` エンドポイント、models.dev）から取得されます。

## 取得の動作

| タイミング | 動作 |
|---|---|
| `/model` または `hermes model` | ディスクキャッシュが古い場合は取得し、そうでなければキャッシュを使用します |
| ディスクキャッシュが新しい（TTL 未満） | ネットワークアクセスなし |
| ネットワーク障害（キャッシュあり） | キャッシュへのサイレントフォールバック、ログ行 1 行 |
| ネットワーク障害（キャッシュなし） | リポジトリ内スナップショットへのサイレントフォールバック |
| マニフェストがスキーマ検証に失敗 | 到達不能として扱われます |

キャッシュの場所: `~/.hermes/cache/model_catalog.json`。

## 設定

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 24
  providers: {}
```

`enabled: false` に設定すると、リモート取得を完全に無効化し、常にリポジトリ内のスナップショットを使用します。

### プロバイダーごとのオーバーライド URL

サードパーティは、同じスキーマを使用して独自のキュレーションリストをセルフホストできます。プロバイダーをカスタム URL に向けます:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

オーバーライドするマニフェストは、関心のあるプロバイダーブロックのみを設定すればよいです。その他のプロバイダーは、引き続きマスター URL に対して解決されます。

## マニフェストの更新

メンテナー向け:

```bash
# リポジトリ内のハードコードされたリストから再生成します（hermes_cli/models.py の
# OPENROUTER_MODELS または _PROVIDER_MODELS["nous"] を編集した後、マニフェストを同期状態に保ちます）。
python scripts/build_model_catalog.py
```

その後、結果として生じる変更を `website/static/api/model-catalog.json` に対して `main` へ PR します。ドキュメントサイトはマージ時に自動デプロイされ、新しいマニフェストは数分以内にライブになります。

リポジトリ内のスナップショットに属さないきめ細かいメタデータの変更については、JSON を直接手動編集することもできます — ジェネレータースクリプトは便利なものであり、唯一の信頼できる情報源ではありません。
