---
sidebar_position: 17
title: "ダッシュボードの拡張"
description: "Hermesのウェブダッシュボード向けのテーマとプラグインを構築する — パレット、タイポグラフィ、レイアウト、カスタムタブ、シェルスロット、ページスコープのスロット、バックエンドAPIルート"
---

# ダッシュボードの拡張

Hermesのウェブダッシュボード（`hermes dashboard`）は、コードベースをフォークすることなくリスキンおよび拡張できるように作られています。3つのレイヤーが公開されています:

1. **テーマ** — ダッシュボードのパレット、タイポグラフィ、レイアウト、コンポーネントごとのクロームを塗り直すYAMLファイル。`~/.hermes/dashboard-themes/` にファイルを置くと、テーマ切り替え画面に表示されます。
2. **UIプラグイン** — `manifest.json` とJavaScriptバンドルを含むディレクトリで、タブを登録したり、組み込みページを置き換えたり、ページスコープのスロットでページを拡張したり、名前付きのシェルスロットにコンポーネントを注入したりします。
3. **バックエンドプラグイン** — そのプラグインディレクトリ内のPythonファイルで、FastAPIの `router` を公開します。ルートは `/api/plugins/<name>/` の下にマウントされ、プラグインのUIから呼び出されます。

3つすべてが **実行時にドロップインで利用可能** です: リポジトリのクローンも、`npm run build` も、ダッシュボードのソースへのパッチも不要です。このページは、これら3つすべての正式なリファレンスです。

ダッシュボードを使いたいだけの場合は、[ウェブダッシュボード](./web-dashboard)を参照してください。（ウェブダッシュボードではなく）ターミナルCLIをリスキンしたい場合は、[スキンとテーマ](./skins)を参照してください — CLIのスキンシステムはダッシュボードのテーマとは無関係です。

:::note 各要素の組み合わせ方
テーマとプラグインは独立していますが、相乗効果があります。テーマは単独で成立します（YAMLファイル1つだけ）。プラグインも単独で成立します（タブ1つだけ）。両者を組み合わせると、カスタムのHUDを備えた完全なビジュアルリスキンを構築できます — 同梱の `strike-freedom-cockpit` デモはまさにそれを行います。[テーマ + プラグインの組み合わせデモ](#combined-theme--plugin-demo)を参照してください。
:::

---

## 目次

- [テーマ](#themes)
  - [クイックスタート — 最初のテーマ](#quick-start--your-first-theme)
  - [パレット、タイポグラフィ、レイアウト](#palette-typography-layout)
  - [レイアウトのバリアント](#layout-variants)
  - [テーマアセット（CSS変数としての画像）](#theme-assets-images-as-css-vars)
  - [コンポーネントクロームの上書き](#component-chrome-overrides)
  - [色の上書き](#color-overrides)
  - [生の `customCSS`](#raw-customcss)
  - [組み込みテーマ](#built-in-themes)
  - [テーマYAMLの完全リファレンス](#full-theme-yaml-reference)
- [プラグイン](#plugins)
  - [クイックスタート — 最初のプラグイン](#quick-start--your-first-plugin)
  - [ディレクトリ構成](#directory-layout)
  - [マニフェストのリファレンス](#manifest-reference)
  - [プラグインSDK](#the-plugin-sdk)
  - [シェルスロット](#shell-slots)
  - [組み込みページの置き換え（`tab.override`）](#replacing-built-in-pages-taboverride)
  - [組み込みページの拡張（ページスコープのスロット）](#augmenting-built-in-pages-page-scoped-slots)
  - [スロット専用プラグイン（`tab.hidden`）](#slot-only-plugins-tabhidden)
  - [バックエンドAPIルート](#backend-api-routes)
  - [プラグインごとのカスタムCSS](#custom-css-per-plugin)
  - [プラグインの検出とリロード](#plugin-discovery--reload)
- [テーマ + プラグインの組み合わせデモ](#combined-theme--plugin-demo)
- [APIリファレンス](#api-reference)
- [トラブルシューティング](#troubleshooting)

---

## テーマ {#themes}

テーマは `~/.hermes/dashboard-themes/` に保存されるYAMLファイルです。ファイル名は問いません（システムが使うのはテーマの `name:` フィールドです）が、慣例としては `<name>.yaml` です。すべてのフィールドは任意です — 欠けているキーは組み込みの `default` テーマにフォールバックするため、テーマは色1つだけの小ささにもできます。

### クイックスタート — 最初のテーマ {#quick-start--your-first-theme}

```bash
mkdir -p ~/.hermes/dashboard-themes
```

```yaml
# ~/.hermes/dashboard-themes/neon.yaml
name: neon
label: Neon
description: Pure magenta on black

palette:
  background: "#000000"
  midground: "#ff00ff"
```

ダッシュボードを更新します。ヘッダーのパレットアイコンをクリックして **Neon** を選びます。背景が黒になり、テキストとアクセントがマゼンタになり、派生するすべての色（カード、ボーダー、ミュート、リングなど）がその2色のトリプレットからCSSの `color-mix()` を介して再計算されます。

これがオンボーディングのすべてです: 1ファイル、2色。以下のものはすべて任意の改良です。

### パレット、タイポグラフィ、レイアウト {#palette-typography-layout}

これら3つのブロックがテーマの中心です。それぞれが独立しています — 1つを上書きし、他はそのままにできます。

#### パレット（3層）

パレットは色のレイヤーのトリプレットに加えて、暖かい輝きのビネット色とノイズグレインの乗数で構成されます。ダッシュボードのデザインシステムのカスケードは、このトリプレットからCSSの `color-mix()` を介して、shadcn互換のすべてのトークン（カード、ポップオーバー、ミュート、ボーダー、プライマリ、デストラクティブ、リングなど）を導出します。3色を上書きすると、UI全体にカスケードします。

| キー | 説明 |
|-----|-------------|
| `palette.background` | 最も深いキャンバスの色 — 通常はほぼ黒。ページの背景とカードの塗りを駆動します。 |
| `palette.midground` | 主要なテキストとアクセント。ほとんどのUIクローム（前景テキスト、ボタンのアウトライン、フォーカスリング）がこれを読みます。 |
| `palette.foreground` | 最上層のハイライト。デフォルトのテーマはこれをアルファ0の白（不可視）に設定します。最上層に明るいアクセントが欲しいテーマは、そのアルファを上げられます。 |
| `palette.warmGlow` | `<Backdrop />` がビネット色として使う `rgba(...)` 文字列。 |
| `palette.noiseOpacity` | グレインオーバーレイにかかる0〜1.2の乗数。低いほど柔らかく、高いほど粗くなります。 |

各レイヤーは `{hex: "#RRGGBB", alpha: 0.0–1.0}` か、裸のhex文字列（アルファはデフォルトで1.0）のいずれかを受け付けます。

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # 裸のhex、alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # 不可視の最上層
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### タイポグラフィ

| キー | 型 | 説明 |
|-----|------|-------------|
| `fontSans` | string | 本文用のCSS font-familyスタック（`html`、`body` に適用）。 |
| `fontMono` | string | コードブロック、`<code>`、`.font-mono` ユーティリティ用のCSS font-familyスタック。 |
| `fontDisplay` | string | 任意の見出し/ディスプレイ用スタック。`fontSans` にフォールバックします。 |
| `fontUrl` | string | 任意の外部スタイルシートURL。テーマ切り替え時に `<head>` 内へ `<link rel="stylesheet">` として注入されます。同じURLが2回注入されることはありません。Google Fonts、Bunny Fonts、セルフホストの `@font-face` シートなど、リンク可能なものなら何でも動作します。 |
| `baseSize` | string | ルートのフォントサイズ — remスケールを制御します。例: `"14px"`、`"16px"`。 |
| `lineHeight` | string | デフォルトの行の高さ。例: `"1.5"`、`"1.65"`。 |
| `letterSpacing` | string | デフォルトの字間。例: `"0"`、`"0.01em"`、`"-0.01em"`。 |

```yaml
typography:
  fontSans: '"Orbitron", "Eurostile", "Impact", sans-serif'
  fontMono: '"Share Tech Mono", ui-monospace, monospace'
  fontDisplay: '"Orbitron", "Eurostile", sans-serif'
  fontUrl: "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Share+Tech+Mono&display=swap"
  baseSize: "14px"
  lineHeight: "1.5"
  letterSpacing: "0.04em"
```

#### レイアウト

| キー | 値 | 説明 |
|-----|--------|-------------|
| `radius` | 任意のCSSの長さ（`"0"`、`"0.25rem"`、`"0.5rem"`、`"1rem"`、...） | 角丸トークン。`--radius` にマッピングされ、`--radius-sm/md/lg/xl` にカスケードします — 丸みを帯びたすべての要素が一緒に変化します。 |
| `density` | `compact` \| `comfortable` \| `spacious` | `--spacing-mul` CSS変数として適用される間隔の乗数。`compact = 0.85×`、`comfortable = 1.0×`（デフォルト）、`spacious = 1.2×`。Tailwindの基本間隔をスケールするため、パディング、ギャップ、space-betweenユーティリティがすべて比例して変化します。 |

```yaml
layout:
  radius: "0"
  density: compact
```

### レイアウトのバリアント {#layout-variants}

`layoutVariant` はシェル全体のレイアウトを選択します。指定がない場合は `"standard"` がデフォルトです。

| バリアント | 動作 |
|---------|-----------|
| `standard` | 単一カラム、最大幅1600px（デフォルト）。 |
| `cockpit` | 左サイドバーレール（260px）+ メインコンテンツ。`sidebar` スロットを介してプラグインによって埋められます — [シェルスロット](#shell-slots)を参照してください。プラグインがない場合、レールはプレースホルダーを表示します。 |
| `tiled` | 最大幅のクランプを外し、ページがビューポートの全幅を使えるようにします。 |

```yaml
layoutVariant: cockpit
```

現在のバリアントは `document.documentElement.dataset.layoutVariant` として公開されるため、`customCSS` 内の生のCSSは `:root[data-layout-variant="cockpit"] ...` でそれをターゲットにできます。

### テーマアセット（CSS変数としての画像） {#theme-assets-images-as-css-vars}

アートワークのURLをテーマに同梱します。名前付きの各スロットはCSS変数（`--theme-asset-<name>`）になり、組み込みのシェルや任意のプラグインがそれを読めます。`bg` スロットは自動的にバックドロップに接続されます。その他のスロットはプラグイン向けです。

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # <Backdrop /> に自動接続
  hero: "/my-images/strike-freedom.png"           # プラグインのサイドバー用
  crest: "/my-images/crest.svg"                   # header-left のプラグイン用
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

値は次を受け付けます:

- 裸のURL — 自動的に `url(...)` でラップされます。
- 事前にラップされた `url(...)`、`linear-gradient(...)`、`radial-gradient(...)` の式 — そのまま使われます。
- `"none"` — 明示的なオプトアウト。

各アセットは `--theme-asset-<name>-raw`（ラップされていないURL）としても出力されます。プラグインが `background-image` ではなく `<img src>` にそれを渡す必要がある場合に備えてです。

プラグインはこれらを素のCSSまたはJSで読みます:

```javascript
// プラグインのスロット内
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### コンポーネントクロームの上書き {#component-chrome-overrides}

`componentStyles` は、CSSセレクターを書かずに個々のシェルコンポーネントのスタイルを変更します。各バケットのエントリはCSS変数（`--component-<bucket>-<kebab-property>`）になり、シェルの共有コンポーネントがそれを読みます。そのため、`card:` の上書きはすべての `<Card>` に適用され、`header:` はアプリバーに適用される、といった具合です。

```yaml
componentStyles:
  card:
    clipPath: "polygon(12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%, 0 12px)"
    background: "linear-gradient(180deg, rgba(10, 22, 52, 0.85), rgba(5, 9, 26, 0.92))"
    boxShadow: "inset 0 0 0 1px rgba(64, 200, 255, 0.28)"
  header:
    background: "linear-gradient(180deg, rgba(16, 32, 72, 0.95), rgba(5, 9, 26, 0.9))"
  tab:
    clipPath: "polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)"
  sidebar: {}
  backdrop: {}
  footer: {}
  progress: {}
  badge: {}
  page: {}
```

サポートされているバケット: `card`、`header`、`footer`、`sidebar`、`tab`、`progress`、`badge`、`backdrop`、`page`。

プロパティ名はキャメルケース（`clipPath`）を使い、ケバブ（`clip-path`）として出力されます。値は素のCSS文字列です — CSSが受け付けるもの（`clip-path`、`border-image`、`background`、`box-shadow`、`animation` など）なら何でも構いません。

### 色の上書き {#color-overrides}

ほとんどのテーマではこれは不要です — 3層パレットがすべてのshadcnトークンを導出します。`colorOverrides` は、導出では生成されない特定のアクセントが欲しいときに使います（パステルテーマ向けの柔らかめのデストラクティブな赤、ブランド向けの特定のサクセスグリーンなど）。

```yaml
colorOverrides:
  primary: "#ffce3a"
  primaryForeground: "#05091a"
  accent: "#3fd3ff"
  ring: "#3fd3ff"
  destructive: "#ff3a5e"
  border: "rgba(64, 200, 255, 0.28)"
```

サポートされているキー: `card`、`cardForeground`、`popover`、`popoverForeground`、`primary`、`primaryForeground`、`secondary`、`secondaryForeground`、`muted`、`mutedForeground`、`accent`、`accentForeground`、`destructive`、`destructiveForeground`、`success`、`warning`、`border`、`input`、`ring`。

各キーは `--color-<kebab>` CSS変数に1対1でマッピングされます（例: `primaryForeground` → `--color-primary-foreground`）。ここで設定したキーは、アクティブなテーマに対してのみパレットカスケードに優先します — 別のテーマに切り替えると上書きはクリアされます。

### 生の `customCSS` {#raw-customcss}

`componentStyles` では表現できないセレクターレベルのクローム — 擬似要素、アニメーション、メディアクエリ、テーマスコープの上書き — には、生のCSSを `customCSS` に書き込みます:

```yaml
customCSS: |
  /* スキャンラインオーバーレイ — cockpitバリアントがアクティブなときだけ表示。 */
  :root[data-layout-variant="cockpit"] body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 100;
    background: repeating-linear-gradient(to bottom,
      transparent 0px, transparent 2px,
      rgba(64, 200, 255, 0.035) 3px, rgba(64, 200, 255, 0.035) 4px);
    mix-blend-mode: screen;
  }
```

CSSはテーマ適用時に単一のスコープ付き `<style data-hermes-theme-css>` タグとして注入され、テーマ切り替え時にクリーンアップされます。**テーマごとに32 KiBが上限です。**

### 組み込みテーマ {#built-in-themes}

各組み込みテーマは独自のパレット、タイポグラフィ、レイアウトを備えています — 切り替えると、色だけでなく目に見える変化が生じます。

| テーマ | パレット | タイポグラフィ | レイアウト |
|-------|---------|------------|--------|
| **Hermes Teal**（`default`） | ダークティール + クリーム | システムスタック、15px | 0.5rem radius、comfortable |
| **Hermes Teal (Large)**（`default-large`） | デフォルトと同じ | システムスタック、18px、行の高さ1.65 | 0.5rem radius、spacious |
| **Midnight**（`midnight`） | ディープブルーバイオレット | Inter + JetBrains Mono、14px | 0.75rem radius、comfortable |
| **Ember**（`ember`） | 暖かいクリムゾン + ブロンズ | Spectral（セリフ）+ IBM Plex Mono、15px | 0.25rem radius、comfortable |
| **Mono**（`mono`） | グレースケール | IBM Plex Sans + IBM Plex Mono、13px | 0 radius、compact |
| **Cyberpunk**（`cyberpunk`） | 黒地にネオングリーン | 全体にShare Tech Mono、14px | 0 radius、compact |
| **Rosé**（`rose`） | ピンク + アイボリー | Fraunces（セリフ）+ DM Mono、16px | 1rem radius、spacious |

Google Fontsを参照するテーマ（Hermes Teal以外すべて）は、スタイルシートをオンデマンドで読み込みます — 初めて切り替えたときに `<link>` タグが `<head>` に注入されます。

### テーマYAMLの完全リファレンス {#full-theme-yaml-reference}

すべてのつまみを1ファイルに — コピーして不要なものを削ってください:

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

# 3層パレット（{hex, alpha} または裸のhexを受け付ける）
palette:
  background:
    hex: "#0a1628"
    alpha: 1.0
  midground:
    hex: "#a8d0ff"
    alpha: 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0.0
  warmGlow: "rgba(255, 107, 107, 0.35)"
  noiseOpacity: 0.7

typography:
  fontSans: "Poppins, system-ui, sans-serif"
  fontMono: "Fira Code, ui-monospace, monospace"
  fontDisplay: "Poppins, system-ui, sans-serif"   # 任意
  fontUrl: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap"
  baseSize: "15px"
  lineHeight: "1.6"
  letterSpacing: "-0.003em"

layout:
  radius: "0.75rem"
  density: comfortable

layoutVariant: standard        # standard | cockpit | tiled

assets:
  bg: "https://example.com/ocean-bg.jpg"
  hero: "/my-images/kraken.png"
  crest: "/my-images/anchor.svg"
  logo: "/my-images/logo.png"
  custom:
    pattern: "/my-images/waves.svg"

componentStyles:
  card:
    boxShadow: "inset 0 0 0 1px rgba(168, 208, 255, 0.18)"
  header:
    background: "linear-gradient(180deg, rgba(10, 22, 40, 0.95), rgba(5, 9, 26, 0.9))"

colorOverrides:
  destructive: "#ff6b6b"
  ring: "#ff6b6b"

customCSS: |
  /* セレクターレベルの追加調整があればここに */
```

ファイルを作成した後、ダッシュボードを更新します。ヘッダーバーからライブでテーマを切り替えます — パレットアイコンをクリックします。選択は `config.yaml` の `dashboard.theme` に永続化され、リロード時に復元されます。

---

## プラグイン {#plugins}

ダッシュボードプラグインは、`manifest.json`、ビルド済みのJSバンドル、そして任意でCSSファイルとFastAPIルートを含むPythonファイルを持つディレクトリです。プラグインは他のHermesプラグインと並んで `~/.hermes/plugins/<name>/` に置かれます — ダッシュボード拡張はそのプラグインディレクトリ内の `dashboard/` サブフォルダなので、1つのプラグインで単一のインストールからCLI/ゲートウェイとダッシュボードの両方を拡張できます。

プラグインはReactやUIコンポーネントをバンドルしません。`window.__HERMES_PLUGIN_SDK__` に公開された **プラグインSDK** を使います。これによりプラグインバンドルは小さく保たれ（通常は数KB）、バージョンの衝突を避けられます。

### クイックスタート — 最初のプラグイン {#quick-start--your-first-plugin}

ディレクトリ構造を作成します:

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

マニフェストを書きます:

```json
// ~/.hermes/plugins/my-plugin/dashboard/manifest.json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills"
  },
  "entry": "dist/index.js"
}
```

JSバンドルを書きます（素のIIFE — ビルドステップ不要）:

```javascript
// ~/.hermes/plugins/my-plugin/dashboard/dist/index.js
(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardHeader, CardTitle, CardContent } = SDK.components;

  function MyPage() {
    return React.createElement(Card, null,
      React.createElement(CardHeader, null,
        React.createElement(CardTitle, null, "My Plugin"),
      ),
      React.createElement(CardContent, null,
        React.createElement("p", { className: "text-sm text-muted-foreground" },
          "Hello from my custom dashboard tab.",
        ),
      ),
    );
  }

  window.__HERMES_PLUGINS__.register("my-plugin", MyPage);
})();
```

ダッシュボードを更新します — あなたのタブが **Skills** の後にナビバーに表示されます。

:::tip React.createElementをスキップする
JSXを好む場合は、Reactを外部依存とし、IIFE出力にする任意のバンドラー（esbuild、Vite、rollup）を使ってください。唯一の絶対的な要件は、最終ファイルが `<script>` で読み込み可能な単一のJSファイルであることです。Reactは決してバンドルされません。`SDK.React` から来ます。
:::

### ディレクトリ構成 {#directory-layout}

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # 任意 — 既存のCLI/ゲートウェイプラグインマニフェスト
├── __init__.py              # 任意 — 既存のCLI/ゲートウェイフック
└── dashboard/               # ダッシュボード拡張
    ├── manifest.json        # 必須 — タブ設定、アイコン、エントリーポイント
    ├── dist/
    │   ├── index.js         # 必須 — ビルド済みJSバンドル（IIFE）
    │   └── style.css        # 任意 — カスタムCSS
    └── plugin_api.py        # 任意 — バックエンドAPIルート（FastAPI）
```

1つのプラグインディレクトリは、3つの直交する拡張を持てます:

- `plugin.yaml` + `__init__.py` — CLI/ゲートウェイプラグイン（[プラグインのページを参照](./plugins)）。
- `dashboard/manifest.json` + `dashboard/dist/index.js` — ダッシュボードUIプラグイン。
- `dashboard/plugin_api.py` — ダッシュボードのバックエンドルート。

いずれも必須ではありません。必要なレイヤーだけを含めてください。

### マニフェストのリファレンス {#manifest-reference}

```json
{
  "name": "my-plugin",
  "label": "My Plugin",
  "description": "What this plugin does",
  "icon": "Sparkles",
  "version": "1.0.0",
  "tab": {
    "path": "/my-plugin",
    "position": "after:skills",
    "override": "/",
    "hidden": false
  },
  "slots": ["sidebar", "header-left"],
  "entry": "dist/index.js",
  "css": "dist/style.css",
  "api": "plugin_api.py"
}
```

| フィールド | 必須 | 説明 |
|-------|----------|-------------|
| `name` | はい | 一意のプラグイン識別子。小文字、ハイフン可。URLと登録に使われます。 |
| `label` | はい | ナビタブに表示される表示名。 |
| `description` | いいえ | 短い説明（ダッシュボードの管理画面に表示されます）。 |
| `icon` | いいえ | Lucideのアイコン名。デフォルトは `Puzzle`。未知の名前は `Puzzle` にフォールバックします。 |
| `version` | いいえ | Semver文字列。デフォルトは `0.0.0`。 |
| `tab.path` | はい | タブのURLパス（例: `/my-plugin`）。 |
| `tab.position` | いいえ | タブを挿入する位置。`"end"`（デフォルト）、`"after:<path>"`、または `"before:<path>"` — コロンの後の値は対象タブの **パスセグメント**（先頭スラッシュなし）です。例: `"after:skills"`、`"before:config"`。 |
| `tab.override` | いいえ | 組み込みルートのパス（`"/"`、`"/sessions"`、`"/config"` など）に設定すると、新しいタブを追加する代わりにそのページを **置き換え** ます。[組み込みページの置き換え](#replacing-built-in-pages-taboverride)を参照してください。 |
| `tab.hidden` | いいえ | trueの場合、ナビにタブを追加せずにコンポーネントと任意のスロットを登録します。スロット専用プラグインで使われます。[スロット専用プラグイン](#slot-only-plugins-tabhidden)を参照してください。 |
| `slots` | いいえ | このプラグインが埋める名前付きシェルスロット。**ドキュメント上の補助のみ** — 実際の登録はJSバンドルから `registerSlot()` を介して行われます。ここにスロットを列挙すると、検出画面の情報がより充実します。 |
| `entry` | はい | `dashboard/` からの相対パスでのJSバンドルへのパス。デフォルトは `dist/index.js`。 |
| `css` | いいえ | `<link>` タグとして注入するCSSファイルへのパス。 |
| `api` | いいえ | FastAPIルートを含むPythonファイルへのパス。`/api/plugins/<name>/` にマウントされます。 |

#### 利用可能なアイコン

プラグインはLucideのアイコン名を使います。ダッシュボードはこれらを名前でマッピングします — 未知の名前は黙って `Puzzle` にフォールバックします。

現在マッピングされているもの: `Activity`、`BarChart3`、`Clock`、`Code`、`Database`、`Eye`、`FileText`、`Globe`、`Heart`、`KeyRound`、`MessageSquare`、`Package`、`Puzzle`、`Settings`、`Shield`、`Sparkles`、`Star`、`Terminal`、`Wrench`、`Zap`。

別のアイコンが必要ですか? `web/src/App.tsx` の `ICON_MAP` にPRを開いてください — 純粋に追加だけの変更です。

### プラグインSDK {#the-plugin-sdk}

プラグインに必要なものはすべて `window.__HERMES_PLUGIN_SDK__` にあります。プラグインは決してReactを直接インポートすべきではありません。

```javascript
const SDK = window.__HERMES_PLUGIN_SDK__;

// React + hooks
SDK.React                    // the React instance
SDK.hooks.useState
SDK.hooks.useEffect
SDK.hooks.useCallback
SDK.hooks.useMemo
SDK.hooks.useRef
SDK.hooks.useContext
SDK.hooks.createContext

// UI components (shadcn/ui primitives)
SDK.components.Card
SDK.components.CardHeader
SDK.components.CardTitle
SDK.components.CardContent
SDK.components.Badge
SDK.components.Button
SDK.components.Input
SDK.components.Label
SDK.components.Select
SDK.components.SelectOption
SDK.components.Separator
SDK.components.Tabs
SDK.components.TabsList
SDK.components.TabsTrigger
SDK.components.PluginSlot    // render a named slot (useful for nested plugin UIs)

// Hermes API client + raw fetcher
SDK.api                      // typed client — getStatus, getSessions, getConfig, ...
SDK.fetchJSON                // raw fetch for custom endpoints (plugin-registered routes)

// Utilities
SDK.utils.cn                 // Tailwind class merger (clsx + twMerge)
SDK.utils.timeAgo            // "5m ago" from unix timestamp
SDK.utils.isoTimeAgo         // "5m ago" from ISO string

// Hooks
SDK.useI18n                  // i18n hook for multi-language plugins
```

#### プラグインのバックエンドを呼び出す

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON` はセッション認証トークンを注入し、エラーをスローされる例外として表面化し、JSONを自動的にパースします。

#### 組み込みのHermesエンドポイントを呼び出す

```javascript
// エージェントのステータス
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// 最近のセッション
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

完全な一覧については [ウェブダッシュボード → REST API](./web-dashboard#rest-api) を参照してください。

### シェルスロット {#shell-slots}

スロットを使うと、プラグインはタブ全体を占有することなく、アプリシェルの名前付きの場所 — コックピットのサイドバー、ヘッダー、フッター、オーバーレイ層 — にコンポーネントを注入できます。複数のプラグインが同じスロットを埋めることができます。それらは登録順に積み重なってレンダリングされます。

プラグインバンドルの内部から登録します:

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### スロットのカタログ {#slot-catalogue}

**シェル全体のスロット**（アプリのクローム内のどこにでもレンダリングされます）:

| スロット | 場所 |
|------|----------|
| `backdrop` | `<Backdrop />` のレイヤースタック内、ノイズレイヤーの上。 |
| `header-left` | 上部バーのHermesブランドの前。 |
| `header-right` | 上部バーのテーマ/言語切り替えの前。 |
| `header-banner` | ナビの下の全幅ストリップ。 |
| `sidebar` | コックピットのサイドバーレール — **`layoutVariant === "cockpit"` のときのみレンダリングされます**。 |
| `pre-main` | ルートアウトレットの上（`<main>` 内）。 |
| `post-main` | ルートアウトレットの下（`<main>` 内）。 |
| `footer-left` | フッターのセル内容（デフォルトを置き換え）。 |
| `footer-right` | フッターのセル内容（デフォルトを置き換え）。 |
| `overlay` | すべての上の固定位置レイヤー。`customCSS` だけでは実現できないクローム（スキャンライン、ビネット）に便利です。 |

**ページスコープのスロット**（名前付きの組み込みページにのみレンダリングされます — ルート全体を上書きせずに既存のページへウィジェット、カード、ツールバーを注入するために使います）:

| スロット | レンダリングされる場所 |
|------|------------------|
| `sessions:top` / `sessions:bottom` | `/sessions` ページの上部 / 下部。 |
| `analytics:top` / `analytics:bottom` | `/analytics` ページの上部 / 下部。 |
| `logs:top` / `logs:bottom` | `/logs` の上部（フィルターツールバーの上）/ 下部（ログビューアーの下）。 |
| `cron:top` / `cron:bottom` | `/cron` ページの上部 / 下部。 |
| `skills:top` / `skills:bottom` | `/skills` ページの上部 / 下部。 |
| `config:top` / `config:bottom` | `/config` ページの上部 / 下部。 |
| `env:top` / `env:bottom` | `/env`（Keys）ページの上部 / 下部。 |
| `docs:top` / `docs:bottom` | `/docs` の上部（iframeの上）/ 下部。 |
| `chat:top` / `chat:bottom` | `/chat` の上部 / 下部（埋め込みチャットが有効なときのみアクティブ）。 |

例 — Sessionsページの上部にバナーカードを追加します:

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

プラグインが既存のページを拡張するだけで、独自のサイドバータブを必要としない場合は、ページスコープのスロットと `tab.hidden: true` を組み合わせてください。

シェルは上記のスロットに対してのみ `<PluginSlot name="..." />` をレンダリングします。それ以外の名前はネストされたプラグインUIのためにレジストリで受け付けられます — プラグインは `SDK.components.PluginSlot` を介して独自のスロットを公開できます。

#### 再登録とHMR

同じ `(plugin, slot)` のペアが2回登録された場合、後の呼び出しが先のものを置き換えます — これは、React HMRがプラグインの再マウントに期待する動作と一致します。

### 組み込みページの置き換え（`tab.override`） {#replacing-built-in-pages-taboverride}

`tab.override` を組み込みルートのパスに設定すると、プラグインのコンポーネントが新しいタブを追加する代わりにそのページを置き換えます。テーマがカスタムのホームページ（`/`）を望みつつ、ダッシュボードの残りはそのまま保ちたい場合に便利です。

```json
{
  "name": "my-home",
  "label": "Home",
  "tab": {
    "path": "/my-home",
    "override": "/",
    "position": "end"
  },
  "entry": "dist/index.js"
}
```

`override` が設定されている場合:

- `/` の元のページコンポーネントがルーターから削除されます。
- あなたのプラグインが代わりに `/` でレンダリングされます。
- `tab.path` のナビタブは追加されません（上書きこそが目的です）。

1つのパスを上書きできるプラグインは1つだけです。2つのプラグインが同じ上書きを主張した場合、最初のものが優先され、2番目は開発モードの警告とともに無視されます。

ページ全体を引き継ぐことなく、既存のページにカードやツールバーを追加したいだけの場合は、代わりに[ページスコープのスロット](#augmenting-built-in-pages-page-scoped-slots)を使ってください。

### 組み込みページの拡張（ページスコープのスロット） {#augmenting-built-in-pages-page-scoped-slots}

`tab.override` による完全な置き換えは重いです — あなたのプラグインが、今後私たちが配信する更新を含め、ページ全体を所有することになります。たいていの場合、既存のページにバナー、カード、ツールバーを追加したいだけです。そのためにあるのが **ページスコープのスロット** です。

すべての組み込みページは、コンテンツ領域の上部と下部にレンダリングされる `<page>:top` と `<page>:bottom` のスロットを公開します。プラグインは `registerSlot()` を呼び出してその1つを埋めます — 組み込みページは通常どおり動作し続け、あなたのコンポーネントがその横にレンダリングされます。

利用可能なスロット: `sessions:*`、`analytics:*`、`logs:*`、`cron:*`、`skills:*`、`config:*`、`env:*`、`docs:*`、`chat:*`（それぞれに `:top` と `:bottom`）。完全なカタログは[シェルスロット → スロットのカタログ](#slot-catalogue)を参照してください。

最小限の例 — Sessionsページの上部にバナーをピン留めします:

```json
// ~/.hermes/plugins/session-notes/dashboard/manifest.json
{
  "name": "session-notes",
  "label": "Session Notes",
  "tab": { "path": "/session-notes", "hidden": true },
  "slots": ["sessions:top"],
  "entry": "dist/index.js"
}
```

```javascript
// ~/.hermes/plugins/session-notes/dashboard/dist/index.js
(function () {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  const { React } = SDK;
  const { Card, CardContent } = SDK.components;

  function Banner() {
    return React.createElement(Card, null,
      React.createElement(CardContent, { className: "py-2 text-xs" },
        "Remember to label important sessions before archiving."),
    );
  }

  // 隠しタブ用のプレースホルダー。
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // 実際の処理。
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

要点:

- `tab.hidden: true` はプラグインをサイドバーから外します — 独立したページを持ちません。
- `slots` マニフェストフィールドはドキュメントのみです。実際のバインディングはJSバンドル内で `registerSlot()` を介して行われます。
- 複数のプラグインが同じページスコープのスロットを主張できます。それらは登録順に積み重なってレンダリングされます。
- プラグインが何も登録しなければ影響はゼロです: 組み込みページは以前とまったく同じようにレンダリングされます。

リファレンスプラグイン（[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/example-dashboard) の `example-dashboard`）は、`sessions:top` にバナーを注入するライブデモを同梱しています — インストールすると、このパターンを端から端まで確認できます。

### スロット専用プラグイン（`tab.hidden`） {#slot-only-plugins-tabhidden}

`tab.hidden: true` の場合、プラグインは（直接URLにアクセスされたときのために）そのコンポーネントと任意のスロットを登録しますが、ナビゲーションにタブを追加することはありません。スロットに注入するためだけに存在するプラグイン — ヘッダーの紋章、サイドバーのHUD、オーバーレイ — で使われます。

```json
{
  "name": "header-crest",
  "label": "Header Crest",
  "tab": {
    "path": "/header-crest",
    "position": "end",
    "hidden": true
  },
  "slots": ["header-left"],
  "entry": "dist/index.js"
}
```

バンドルは依然としてプレースホルダーコンポーネントで `register()` を呼び（誰かが直接URLにアクセスした場合に備えての良い慣行です）、その後 `registerSlot()` を呼んで実際の処理を行います。

### バックエンドAPIルート {#backend-api-routes}

プラグインはマニフェストで `api` を設定することでFastAPIルートを登録できます。ファイルを作成して `router` をエクスポートします:

```python
# ~/.hermes/plugins/my-plugin/dashboard/plugin_api.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/data")
async def get_data():
    return {"items": ["one", "two", "three"]}

@router.post("/action")
async def do_action(body: dict):
    return {"ok": True, "received": body}
```

ルートは `/api/plugins/<name>/` の下にマウントされるため、上記は次のようになります:

- `GET  /api/plugins/my-plugin/data`
- `POST /api/plugins/my-plugin/action`

プラグインのAPIルートは、ダッシュボードサーバーがデフォルトでlocalhostにバインドするため、セッショントークン認証をバイパスします。**信頼できないプラグインを実行する場合は、`--host 0.0.0.0` でダッシュボードを公開インターフェースに公開しないでください** — そのルートも到達可能になります。

#### Hermesの内部へのアクセス

バックエンドルートはダッシュボードプロセス内で実行されるため、hermes-agentのコードベースから直接インポートできます:

```python
from fastapi import APIRouter
from hermes_state import SessionDB
from hermes_cli.config import load_config

router = APIRouter()

@router.get("/session-count")
async def session_count():
    db = SessionDB()
    try:
        count = len(db.list_sessions(limit=9999))
        return {"count": count}
    finally:
        db.close()

@router.get("/config-snapshot")
async def config_snapshot():
    cfg = load_config()
    return {"model": cfg.get("model", {})}
```

### プラグインごとのカスタムCSS {#custom-css-per-plugin}

プラグインがTailwindクラスやインラインの `style=` を超えるスタイルを必要とする場合は、CSSファイルを追加してマニフェストで参照します:

```json
{
  "css": "dist/style.css"
}
```

このファイルは、プラグインの読み込み時に `<link>` タグとして注入されます。ダッシュボードのスタイルとの衝突を避けるために具体的なクラス名を使い、テーマに対応するためにダッシュボードのCSS変数を参照してください:

```css
/* dist/style.css */
.my-plugin-chart {
  border: 1px solid var(--color-border);
  background: var(--color-card);
  color: var(--color-card-foreground);
  padding: 1rem;
}
.my-plugin-chart:hover {
  border-color: var(--color-ring);
}
```

ダッシュボードは、すべてのshadcnトークンを `--color-*` として公開するほか、テーマの追加分（`--theme-asset-*`、`--component-<bucket>-*`、`--radius`、`--spacing-mul`）も公開します。それらを参照すれば、プラグインはアクティブなテーマで自動的にリスキンされます。

### プラグインの検出とリロード {#plugin-discovery--reload}

ダッシュボードは3つのディレクトリで `dashboard/manifest.json` をスキャンします:

| 優先度 | ディレクトリ | ソースラベル |
|----------|-----------|--------------|
| 1（衝突時に優先） | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project` — `HERMES_ENABLE_PROJECT_PLUGINS` が設定されている場合のみ |

検出結果はダッシュボードプロセスごとにキャッシュされます。新しいプラグインを追加した後は、次のいずれかを行います:

```bash
# 再起動なしで再スキャンを強制
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

…または `hermes dashboard` を再起動します。

#### プラグインの読み込みライフサイクル

1. ダッシュボードが読み込まれます。`main.tsx` がSDKを `window.__HERMES_PLUGIN_SDK__` に、レジストリを `window.__HERMES_PLUGINS__` に公開します。
2. `App.tsx` が `usePlugins()` を呼び出し → `GET /api/dashboard/plugins` を取得します。
3. 各マニフェストについて: CSSの `<link>` が注入され（宣言されている場合）、その後 `<script>` タグがJSバンドルを読み込みます。
4. プラグインのIIFEが実行され、`window.__HERMES_PLUGINS__.register(name, Component)` を呼び出します — そして任意で各スロットについて `.registerSlot(name, slot, Component)` を呼び出します。
5. ダッシュボードは登録されたコンポーネントをマニフェストに照らして解決し、（`hidden` でない限り）タブをナビゲーションに追加し、コンポーネントをルートとしてマウントします。

プラグインは、スクリプトが読み込まれてから `register()` を呼び出すまでに最大 **2秒** の猶予があります。それを過ぎると、ダッシュボードは待機をやめて初期レンダリングを完了します。プラグインが後で登録した場合でも、それは表示されます — ナビはリアクティブです。

プラグインのスクリプトの読み込みに失敗した場合（404、構文エラー、IIFE中の例外）、ダッシュボードはブラウザのコンソールに警告をログ出力し、それなしで続行します。

---

## テーマ + プラグインの組み合わせデモ {#combined-theme--plugin-demo}

[`strike-freedom-cockpit`](https://github.com/NousResearch/hermes-example-plugins/tree/main/strike-freedom-cockpit) プラグイン（コンパニオンリポジトリ `hermes-example-plugins`）は、完全なリスキンデモです。テーマYAMLとスロット専用プラグインを組み合わせて、ダッシュボードをフォークせずにコックピット風のHUDを生成します。

**何を示すか:**

- パレット、タイポグラフィ、`fontUrl`、`layoutVariant: cockpit`、`assets`、`componentStyles`（ノッチ付きのカード角、グラデーション背景）、`colorOverrides`、`customCSS`（スキャンラインオーバーレイ）を使ったフルテーマ。
- 3つのスロットに登録するスロット専用プラグイン（`tab.hidden: true`）:
  - `sidebar` — `SDK.api.getStatus()` によって駆動されるライブテレメトリバーを備えたMS-STATUSパネル。
  - `header-left` — アクティブなテーマから `--theme-asset-crest` を読む派閥の紋章。
  - `footer-right` — デフォルトの組織行を置き換えるカスタムのタグライン。
- プラグインはCSS変数を介してテーマが提供するアートワークを読むため、テーマを入れ替えると、プラグインのコードを変更せずにヒーロー/紋章が変わります。

**インストール:**

```bash
git clone https://github.com/NousResearch/hermes-example-plugins.git

# テーマ
cp hermes-example-plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# プラグイン
cp -r hermes-example-plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

ダッシュボードを開き、テーマ切り替えから **Strike Freedom** を選びます。コックピットのサイドバーが現れ、紋章がヘッダーに表示され、タグラインがフッターを置き換えます。**Hermes Teal** に戻すと、プラグインはインストールされたまま不可視になります（`sidebar` スロットは `cockpit` レイアウトバリアントの下でのみレンダリングされます）。

プラグインのソース（コンパニオンリポジトリの `strike-freedom-cockpit/dashboard/dist/index.js`）を読むと、CSS変数の読み方、スロットをサポートしない古いダッシュボードへの防御、1つのバンドルから3つのスロットを登録する方法がわかります。

---

## APIリファレンス {#api-reference}

### テーマのエンドポイント

| エンドポイント | メソッド | 説明 |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | 利用可能なテーマ + アクティブな名前を一覧表示。組み込みは `{name, label, description}` を返します。ユーザーテーマは正規化されたテーマオブジェクト全体を含む `definition` フィールドも含みます。 |
| `/api/dashboard/theme` | PUT | アクティブなテーマを設定。ボディ: `{"name": "midnight"}`。`config.yaml` の `dashboard.theme` に永続化します。 |

### プラグインのエンドポイント

| エンドポイント | メソッド | 説明 |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | 検出されたプラグインを一覧表示（マニフェスト付き、内部フィールドを除く）。 |
| `/api/dashboard/plugins/rescan` | GET | 再起動せずにプラグインディレクトリの再スキャンを強制。 |
| `/dashboard-plugins/<name>/<path>` | GET | プラグインの `dashboard/` ディレクトリから静的アセットを配信。パストラバーサルはブロックされます。 |
| `/api/plugins/<name>/*` | * | プラグインが登録したバックエンドルート。 |

### `window` 上のSDK

| グローバル | 型 | 提供元 |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` — React、hooks、UIコンポーネント、APIクライアント、ユーティリティ。 |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | プラグインのメインコンポーネントを登録。 |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | 名前付きのシェルスロットに登録。 |

---

## トラブルシューティング {#troubleshooting}

**テーマが選択画面に表示されません。**
ファイルが `~/.hermes/dashboard-themes/` にあり、`.yaml` または `.yml` で終わっていることを確認してください。ページを更新してください。`curl http://127.0.0.1:9119/api/dashboard/themes` を実行します — あなたのテーマがレスポンスに含まれるはずです。YAMLにパースエラーがある場合、ダッシュボードは `~/.hermes/logs/` の `errors.log` にログを出力します。

**プラグインのタブが表示されません。**
1. マニフェストが `~/.hermes/plugins/<name>/dashboard/manifest.json` にあることを確認してください（`dashboard/` サブディレクトリに注意）。
2. `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan` で再検出を強制します。
3. ブラウザの開発ツール → ネットワークを開きます — `manifest.json`、`index.js`、および任意のCSSが404なしで読み込まれたことを確認してください。
4. ブラウザの開発ツール → コンソールを開きます — IIFE中のエラーや `window.__HERMES_PLUGINS__ is undefined`（SDKが初期化されなかったことを示し、通常はそれ以前のReactレンダリングのクラッシュが原因）を探してください。
5. バンドルが `manifest.json:name` と **同じ名前** で `window.__HERMES_PLUGINS__.register(...)` を呼び出していることを確認してください。

**スロットに登録したコンポーネントがレンダリングされません。**
`sidebar` スロットは、アクティブなテーマが `layoutVariant: cockpit` を持つときのみレンダリングされます。他のスロットは常にレンダリングされます。一致のないスロットに登録している場合は、`registerSlot` 内に `console.log` を追加して、プラグインバンドルがそもそも実行されたかを確認してください。

**プラグインのバックエンドルートが404を返します。**
1. マニフェストに `"api": "plugin_api.py"` があり、`dashboard/` 内の既存のファイルを指していることを確認してください。
2. `hermes dashboard` を再起動してください — プラグインのAPIルートは起動時に一度マウントされ、再スキャン時には **マウントされません**。
3. `plugin_api.py` がモジュールレベルの `router = APIRouter()` をエクスポートしていることを確認してください。他のエクスポート名は取り込まれません。
4. `~/.hermes/logs/errors.log` で `Failed to load plugin <name> API routes` をtailしてください — インポートエラーがそこにログ出力されます。

**テーマを変更すると色の上書きが消えます。**
`colorOverrides` はアクティブなテーマにスコープされ、テーマ切り替え時にクリアされます — これは設計どおりです。永続する上書きが欲しい場合は、ライブの切り替えではなく、テーマのYAMLに入れてください。

**テーマのcustomCSSが切り詰められます。**
`customCSS` ブロックはテーマごとに32 KiBが上限です。大きなスタイルシートを複数のテーマに分割するか、`css` フィールドを介してフルのスタイルシートを注入するプラグインに切り替えてください（サイズ上限なし）。

**PyPIでプラグインを配布したいです。**
ダッシュボードプラグインは、pipのエントリーポイントではなくディレクトリ構成によってインストールされます。現状で最もクリーンな配布経路は、ユーザーが `~/.hermes/plugins/` にクローンするgitリポジトリです。ダッシュボードプラグイン用のpipベースのインストーラーは、現在は組み込まれていません。
