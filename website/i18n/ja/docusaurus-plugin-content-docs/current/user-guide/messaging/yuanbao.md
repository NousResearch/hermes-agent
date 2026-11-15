---
sidebar_position: 16
title: "Yuanbao"
description: "Yuanbao は Hermes コアから yuanbao Hermes Labs プラグインに移行しました"
---

# Yuanbao

:::warning Hermes Labs プラグイン
Yuanbao は現在 Hermes Labs プラグインです: Nous Research によって作られていますが、サポート対象外であり、今後はメンテナンスされない可能性が高いです。コミュニティからの修正を歓迎します。
:::

[Yuanbao](https://yuanbao.tencent.com/) ゲートウェイプラットフォームとその `yb_*` ツール（`yb_query_group_info`、`yb_query_group_members`、`yb_send_dm`、`yb_search_sticker`、`yb_send_sticker`）は、もはや Hermes コアの一部ではありません。これらは [NousResearch/hermes-labs-plugins](https://github.com/NousResearch/hermes-labs-plugins/tree/main/yuanbao) の `yuanbao` プラグインとして提供されており、完全なセットアップガイドもそちらにあります。

## インストール

```bash
hermes plugins install yuanbao
```

プラグインはプロファイルごとにインストールされます。別のプロファイルには `hermes -p <profile> plugins install yuanbao` を使ってください。その後、ゲートウェイがこのプラットフォームを提供するようにゲートウェイを再起動します。

名前はすべてそのままです: `YUANBAO_*` 環境変数（`YUANBAO_APP_ID` またはレガシーの `YUANBAO_APP_KEY`、`YUANBAO_APP_SECRET`、`YUANBAO_HOME_CHANNEL` など）、`config.yaml` 内の `platforms.yuanbao` ブロック（およびレガシーの `gateway.yuanbao` ブロック）、`yuanbao` ツールセット、ツール名。

:::info Yuanbao が同梱されていたリリースからのアップグレード
何もする必要はありません。すでに Yuanbao を使っていたすべてのプロファイル（`.env` に `YUANBAO_APP_ID` または `YUANBAO_APP_KEY` と `YUANBAO_APP_SECRET` がある、`config.yaml` で `platforms.yuanbao` が有効になっている、または別のプラットフォームに `yuanbao` / `hermes-yuanbao` ツールセットが指定されている）には、`hermes update` によって、またはそのプロファイルの初回起動時（エージェントまたはゲートウェイの起動時。`security.allow_lazy_installs` に従います）に、カタログからプラグインが自動的にインストールされます。失敗した試行はバックオフ付きで再試行され、`hermes update` は常に再試行します。これはプロファイルごとに一度だけ行われます: 後でプラグインを削除した場合（`hermes plugins remove yuanbao`）、削除されたままになります。
:::
