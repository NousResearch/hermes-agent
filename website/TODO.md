# TODO: 日本語ロケール (`ja`) の追加

`website/i18n/` 配下に日本語ファイルを追加し、ドキュメントサイトを日本語対応させる計画。

## 背景・現状

- Docusaurus サイト。`docusaurus.config.ts` の `i18n` で `en`（デフォルト）と `zh-Hans` の2ロケールを定義。
- 全 docs は **314 ファイル**。ただし `zh-Hans` は重要度の高い **3 ファイルのみ**翻訳済み（全訳ではなく厳選方式）。
  - `i18n/zh-Hans/docusaurus-plugin-content-docs/current/user-guide/windows-wsl-quickstart.md`
  - `.../user-guide/features/tool-gateway.md`
  - `.../user-guide/features/image-generation.md`
- 翻訳ファイルが無いページは **自動的に英語へフォールバック**される。よって部分翻訳でも破綻しない。
- `zh-Hans` には `code.json`（テーマUI文字列の翻訳）は存在しない → 必須ではない。

## ディレクトリ構造（追加先）

英語原本 `website/docs/<path>.md` に対応する日本語ファイルは下記に置く:

```
website/i18n/ja/docusaurus-plugin-content-docs/current/<path>.md
```

## 方針

`zh-Hans` と同じ「厳選方式」を踏襲。全314ファイルの一括翻訳はしない。
新規ユーザーが最初に触れる導線（インストール〜基本機能）を優先して日本語化する。

---

## タスク

### Phase 1: 設定（必須・最初に実施） ✅ 完了

- [x] `docusaurus.config.ts` の `i18n.locales` に `'ja'` を追加
- [x] `i18n.localeConfigs` に `ja: { label: '日本語', htmlLang: 'ja' }` を追加
- [x] 検索プラグイン `@easyops-cn/docusaurus-search-local` の `language` 配列に `'ja'` を追加
- [x] `i18n/ja/docusaurus-plugin-content-docs/current/` のディレクトリを作成

### Phase 2: 優先ドキュメントの翻訳（高優先） ✅ 完了

新規ユーザー導線の中核。まずはここから着手。

- [x] `index.md`（トップ／ランディング）
- [x] `getting-started/quickstart.md`
- [x] `getting-started/installation.md`
- [x] `user-guide/windows-wsl-quickstart.md`（zh-Hans と揃える）
- [x] `user-guide/features/overview.md`
- [x] `user-guide/features/tool-gateway.md`（zh-Hans と揃える）
- [x] `user-guide/features/image-generation.md`（zh-Hans と揃える）

### Phase 3: 基本機能ドキュメントの翻訳（中優先）

- [ ] `getting-started/learning-path.md`
- [ ] `getting-started/updating.md`
- [ ] `user-guide/cli.md`
- [ ] `user-guide/configuration.md`
- [ ] `user-guide/configuring-models.md`
- [ ] `user-guide/features/skills.md`
- [ ] `user-guide/features/memory.md`
- [ ] `reference/faq.md`

### Phase 4: 検証 ✅ 完了

- [x] `npm install`
- [x] `npm run build -- --locale ja` でビルドが通ることを確認（`[SUCCESS]`。broken link 警告は英語ソース由来の既存分のみ、翻訳ページは broken anchors に未該当）
- [x] ビルド出力で `<html lang="ja">`、各翻訳ページの日本語反映、未翻訳ページ（cli 等）が英語フォールバックされることを確認
- [ ] （任意）`npm run start -- --locale ja` でローカル起動し、ロケールドロップダウンの「日本語」表示を目視確認

---

## 翻訳ルール

- frontmatter（`title` / `description` / `sidebar_label` 等）も日本語化する。ただし `sidebar_position` 等の数値・キーは変更しない。
- コードブロック・コマンド・パス・環境変数・URL は翻訳しない。
- 固有名詞（Hermes Agent / Nous / Tool Gateway 等）は原則そのまま。必要に応じて「Tool Gateway（ツールゲートウェイ）」のように併記。
- Markdown のアンカー（`#見出し`）リンクは翻訳後の見出しに合わせて崩れないよう注意。
- 文体は「ですます調」で統一。
- 既訳 `zh-Hans` の3ファイルの構成・トーンを参考にする。

## メモ

- 部分翻訳でも英語フォールバックされるため、Phase ごとに段階的にマージ可能。
- 将来的に全訳へ拡張する場合は、`reference/`・`developer-guide/`・`guides/`・`skills/bundled`・`skills/optional` を追加対象とする。
