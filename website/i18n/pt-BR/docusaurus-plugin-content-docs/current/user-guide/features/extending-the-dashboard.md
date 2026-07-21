---
sidebar_position: 17
title: "Estendendo o Dashboard"
description: "Construa temas e plugins para o web dashboard Hermes — paletas, tipografia, layouts, abas customizadas, shell slots, slots scoped à página e rotas de API backend"
---

# Estendendo o Dashboard {#extending-the-dashboard}

O web dashboard Hermes (`hermes dashboard`) foi construído para ser reskinado e estendido sem fork do codebase. Três camadas são expostas:

1. **Themes** — arquivos YAML que repintam paleta, tipografia, layout e chrome por componente do dashboard. Solte um arquivo em `~/.hermes/dashboard-themes/`; ele aparece no theme switcher.
2. **UI plugins** — diretório com `manifest.json` + bundle JavaScript que registra uma aba, substitui página built-in, augmenta uma via page-scoped slots, ou injeta componentes em shell slots nomeados.
3. **Backend plugins** — arquivo Python dentro desse diretório de plugin que expõe um `router` FastAPI; rotas são montadas sob `/api/plugins/<name>/` e chamadas pela UI do plugin.

As três são **drop-in em runtime**: sem clone de repo, sem `npm run build`, sem patch do source do dashboard. Esta página é a referência canônica para as três.

Se você só quer usar o dashboard, veja [Web Dashboard](./web-dashboard). Se quer reskinar a CLI de terminal (não o web dashboard), veja [Skins & Themes](./skins) — o sistema de skins da CLI não tem relação com temas do dashboard.

:::note Não é o app desktop
Esta página cobre o sistema de plugins do **web dashboard** (`hermes dashboard`) — `window.__HERMES_PLUGIN_SDK__`, um `manifest.json` e bundle JS pré-construído. O **app desktop nativo** (`hermes desktop`) tem seu próprio SDK não relacionado — `@hermes/plugin-sdk`, um arquivo ESM, sem build step — documentado em [Desktop Plugin SDK](/developer-guide/desktop-plugin-sdk). Só o namespace backend `plugin_api.py` (`/api/plugins/<name>`) é compartilhado entre eles.
:::

:::note Como as peças se compõem
Themes e plugins são independentes mas sinérgicos. Um theme pode ficar sozinho (só um YAML). Um plugin pode ficar sozinho (só uma aba). Juntos permitem construir reskin visual completo com HUDs customizados — o demo `strike-freedom-cockpit` (vive no repo companion `hermes-example-plugins` — veja [Demo combinado theme + plugin](#combined-theme--plugin-demo) para passos de install) faz exatamente isso.
:::

---

## Índice {#table-of-contents}

- [Themes](#themes)
  - [Quick start — your first theme](#quick-start--your-first-theme)
  - [Palette, typography, layout](#palette-typography-layout)
  - [Layout variants](#layout-variants)
  - [Theme assets (images as CSS vars)](#theme-assets-images-as-css-vars)
  - [Component chrome overrides](#component-chrome-overrides)
  - [Color overrides](#color-overrides)
  - [Raw `customCSS`](#raw-customcss)
  - [Built-in themes](#built-in-themes)
  - [Full theme YAML reference](#full-theme-yaml-reference)
- [Plugins](#plugins)
  - [Quick start — your first plugin](#quick-start--your-first-plugin)
  - [Directory layout](#directory-layout)
  - [Manifest reference](#manifest-reference)
  - [The Plugin SDK](#the-plugin-sdk)
  - [Shell slots](#shell-slots)
  - [Replacing built-in pages (`tab.override`)](#replacing-built-in-pages-taboverride)
  - [Augmenting built-in pages (page-scoped slots)](#augmenting-built-in-pages-page-scoped-slots)
  - [Slot-only plugins (`tab.hidden`)](#slot-only-plugins-tabhidden)
  - [Backend API routes](#backend-api-routes)
  - [Custom CSS per plugin](#custom-css-per-plugin)
  - [Plugin discovery & reload](#plugin-discovery--reload)
- [Combined theme + plugin demo](#combined-theme--plugin-demo)
- [API reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Themes {#themes}

Themes são arquivos YAML armazenados em `~/.hermes/dashboard-themes/`. O nome do arquivo não importa (o campo `name:` do theme é o que o sistema usa), mas a convenção é `<name>.yaml`. Todo campo é opcional — keys faltantes caem no theme built-in `default`, então um theme pode ser tão pequeno quanto uma cor.

### Início rápido — seu primeiro theme {#quick-start--your-first-theme}

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

Atualize o dashboard. Clique o ícone de paleta no header e escolha **Neon**. O background fica preto, texto e accents ficam magenta, e toda cor derivada (card, border, muted, ring, etc.) é recomputada desse triplet de 2 cores via `color-mix()` em CSS.

Esse é o onboarding inteiro: um arquivo, duas cores. Tudo abaixo é refinamento opcional.

### Palette, typography, layout {#palette-typography-layout}

Esses três blocos são o coração de um theme. Cada um é independente — override um, deixe os outros.

#### Palette (3-layer) {#palette-3-layer}

A palette é um triplet de camadas de cor plus cor de vignette warm-glow e multiplicador noise-grain. A cascata design-system do dashboard deriva todo token compatível com shadcn (card, popover, muted, border, primary, destructive, ring, etc.) desse triplet via CSS `color-mix()`. Override de três cores cascateia na UI inteira.

| Key | Descrição |
|-----|-------------|
| `palette.background` | Cor de canvas mais profunda — tipicamente near-black. Dirige page background e card fill. |
| `palette.midground` | Texto primário e accent. A maior parte do chrome UI lê isto (foreground text, button outlines, focus rings). |
| `palette.foreground` | Highlight da camada superior. O theme default seta isto para branco com alpha 0 (invisível); themes que querem accent brilhante no topo podem subir seu alpha. |
| `palette.warmGlow` | string `rgba(...)` usada como cor de vignette por `<Backdrop />`. |
| `palette.noiseOpacity` | Multiplicador 0–1.2 no overlay grain. Menor = mais suave, maior = mais gritty. |

Cada camada aceita `{hex: "#RRGGBB", alpha: 0.0–1.0}` ou uma string hex simples (alpha padrão 1.0).

```yaml
palette:
  background:
    hex: "#05091a"
    alpha: 1.0
  midground: "#d8f0ff"          # bare hex, alpha = 1.0
  foreground:
    hex: "#ffffff"
    alpha: 0                    # invisible top layer
  warmGlow: "rgba(255, 199, 55, 0.24)"
  noiseOpacity: 0.7
```

#### Tipografia {#typography}

| Key | Tipo | Descrição |
|-----|------|-------------|
| `fontSans` | string | Stack CSS font-family para corpo de texto (aplicado a `html`, `body`). |
| `fontMono` | string | Stack CSS font-family para blocos de código, `<code>`, utilitários `.font-mono`. |
| `fontDisplay` | string | Stack opcional para headings/display. Faz fallback para `fontSans`. |
| `fontUrl` | string | URL opcional de stylesheet externo. Injetado como `<link rel="stylesheet">` em `<head>` na troca de theme. A mesma URL nunca é injetada duas vezes. Funciona com Google Fonts, Bunny Fonts, sheets `@font-face` self-hosted — qualquer coisa linkável. |
| `baseSize` | string | Tamanho de fonte raiz — controla a escala rem. Ex.: `"14px"`, `"16px"`. |
| `lineHeight` | string | Line-height padrão. Ex.: `"1.5"`, `"1.65"`. |
| `letterSpacing` | string | Letter-spacing padrão. Ex.: `"0"`, `"0.01em"`, `"-0.01em"`. |

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

##### Alterando a fonte pela UI (sem YAML) {#changing-the-font-from-the-ui-no-yaml}

O theme picker no header do dashboard tem uma seção **Font** abaixo da
lista de themes. Escolha qualquer fonte ali e ela sobrescreve a fonte do corpo
de qualquer theme ativo — a escolha é independente do theme e persiste entre
trocas de theme (armazenada em `config.yaml` sob `dashboard.font`). Escolha
**Theme default** para limpar o override e voltar ao `fontSans` do theme ativo.

O picker oferece um catálogo curado (stacks de sistema mais um conjunto de famílias
Google Fonts em sans / serif / mono). Ele deliberadamente **não** aceita
URL de fonte em texto livre — o stylesheet da fonte é injetado como `<link>`, então o
catálogo mantém as origens injetadas fixas. Para uma face totalmente customizada, defina
`fontSans` + `fontUrl` num YAML de theme como mostrado acima. O `fontMono` do theme
(blocos de código, terminal) sempre fica intocado pelo override da UI.

#### Layout {#layout}

| Key | Valores | Descrição |
|-----|--------|-------------|
| `radius` | qualquer comprimento CSS (`"0"`, `"0.25rem"`, `"0.5rem"`, `"1rem"`, ...) | Token de corner-radius. Mapeia para `--radius` e cascateia em `--radius-sm/md/lg/xl` — todo elemento arredondado muda junto. |
| `density` | `compact` \| `comfortable` \| `spacious` | Multiplicador de espaçamento aplicado como CSS var `--spacing-mul`. `compact = 0.85×`, `comfortable = 1.0×` (padrão), `spacious = 1.2×`. Escala o espaçamento base do Tailwind, então padding, gap e utilitários space-between mudam proporcionalmente. |

```yaml
layout:
  radius: "0"
  density: compact
```

### Variantes de layout {#layout-variants}

`layoutVariant` escolhe o layout geral do shell. Padrão `"standard"` quando ausente.

| Variant | Comportamento |
|---------|---------------|
| `standard` | Coluna única, max-width 1600px (padrão). |
| `cockpit` | Sidebar rail à esquerda (260px) + conteúdo principal. Populado por plugins via slot `sidebar` — veja [Shell slots](#shell-slots). Sem plugin o rail mostra um placeholder. |
| `tiled` | Remove o clamp de max-width para páginas usarem a largura total do viewport. |

```yaml
layoutVariant: cockpit
```

A variante atual é exposta como `document.documentElement.dataset.layoutVariant`, então CSS raw em `customCSS` pode mirá-la via `:root[data-layout-variant="cockpit"] ...`.

### Assets do theme (imagens como CSS vars) {#theme-assets-images-as-css-vars}

Inclua URLs de artwork com um theme. Cada slot nomeado vira uma CSS var (`--theme-asset-<name>`) que o shell built-in e qualquer plugin podem ler. O slot `bg` é conectado automaticamente ao backdrop; outros slots são voltados a plugins.

```yaml
assets:
  bg: "https://example.com/hero-bg.jpg"           # auto-wired into <Backdrop />
  hero: "/my-images/strike-freedom.png"           # for plugin sidebars
  crest: "/my-images/crest.svg"                   # for header-left plugins
  logo: "/my-images/logo.png"
  sidebar: "/my-images/rail.png"
  header: "/my-images/header-art.png"
  custom:
    scanLines: "/my-images/scanlines.png"         # → --theme-asset-custom-scanLines
```

Valores aceitos:

- URLs simples — envolvidas em `url(...)` automaticamente.
- Expressões pré-envolvidas `url(...)`, `linear-gradient(...)`, `radial-gradient(...)` — usadas como estão.
- `"none"` — opt-out explícito.

Todo asset também é emitido como `--theme-asset-<name>-raw` (a URL sem wrapper), caso um plugin precise passá-la para `<img src>` em vez de `background-image`.

Plugins leem isto com CSS ou JS simples:

```javascript
// In a plugin slot
const hero = getComputedStyle(document.documentElement)
  .getPropertyValue("--theme-asset-hero").trim();
```

### Overrides de chrome por componente {#component-chrome-overrides}

`componentStyles` reestiliza componentes individuais do shell sem escrever seletores CSS. As entradas de cada bucket viram CSS vars (`--component-<bucket>-<kebab-property>`) que os componentes compartilhados do shell leem. Assim, overrides em `card:` aplicam a todo `<Card>`, `header:` à app bar, etc.

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

Buckets suportados: `card`, `header`, `footer`, `sidebar`, `tab`, `progress`, `badge`, `backdrop`, `page`.

Nomes de propriedade usam camelCase (`clipPath`) e são emitidos em kebab (`clip-path`). Valores são strings CSS simples — qualquer coisa que CSS aceita (`clip-path`, `border-image`, `background`, `box-shadow`, `animation`, ...).

### Overrides de cor {#color-overrides}

A maioria dos themes não precisa disto — a palette de 3 camadas deriva todo token shadcn. Use `colorOverrides` quando quiser um accent específico que a derivação não produz (um vermelho destructive mais suave para theme pastel, um verde success específico para uma marca).

```yaml
colorOverrides:
  primary: "#ffce3a"
  primaryForeground: "#05091a"
  accent: "#3fd3ff"
  ring: "#3fd3ff"
  destructive: "#ff3a5e"
  border: "rgba(64, 200, 255, 0.28)"
```

Keys suportadas: `card`, `cardForeground`, `popover`, `popoverForeground`, `primary`, `primaryForeground`, `secondary`, `secondaryForeground`, `muted`, `mutedForeground`, `accent`, `accentForeground`, `destructive`, `destructiveForeground`, `success`, `warning`, `border`, `input`, `ring`.

Cada key mapeia 1:1 para a CSS var `--color-<kebab>` (ex.: `primaryForeground` → `--color-primary-foreground`). Qualquer key definida aqui vence a cascata da palette só para o theme ativo — trocar para outro theme limpa os overrides.

### `customCSS` raw {#raw-customcss}

Para chrome em nível de seletor que `componentStyles` não expressa — pseudo-elementos, animações, media queries, overrides scoped ao theme — coloque CSS raw em `customCSS`:

```yaml
customCSS: |
  /* Scanline overlay — only visible when cockpit variant is active. */
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

O CSS é injetado como uma tag `<style data-hermes-theme-css>` scoped na aplicação do theme e removido na troca de theme. **Limitado a 32 KiB por theme.**

### Temas built-in {#built-in-themes}

Cada built-in traz sua própria palette, tipografia e layout — trocar produz mudanças visíveis além da cor sozinha.

| Theme | Palette | Typography | Layout |
|-------|---------|------------|--------|
| **Hermes Teal** (`default`) | Teal escuro + creme | Stack de sistema, 15px | radius 0.5rem, comfortable |
| **Hermes Teal (Large)** (`default-large`) | Igual ao default | Stack de sistema, 18px, line-height 1.65 | radius 0.5rem, spacious |
| **Midnight** (`midnight`) | Azul-violeta profundo | Inter + JetBrains Mono, 14px | radius 0.75rem, comfortable |
| **Ember** (`ember`) | Carmesim quente + bronze | Spectral (serif) + IBM Plex Mono, 15px | radius 0.25rem, comfortable |
| **Mono** (`mono`) | Escala de cinza | IBM Plex Sans + IBM Plex Mono, 13px | radius 0, compact |
| **Cyberpunk** (`cyberpunk`) | Verde neon sobre preto | Share Tech Mono em tudo, 14px | radius 0, compact |
| **Rosé** (`rose`) | Rosa + marfim | Fraunces (serif) + DM Mono, 16px | radius 1rem, spacious |

Themes que referenciam Google Fonts (todos exceto Hermes Teal) carregam o stylesheet sob demanda — na primeira vez que você troca para eles uma tag `<link>` é injetada em `<head>`.

### Referência YAML completa do theme {#full-theme-yaml-reference}

Todo knob num arquivo — copie e remova o que não precisar:

```yaml
# ~/.hermes/dashboard-themes/ocean.yaml
name: ocean
label: Ocean Deep
description: Deep sea blues with coral accents

# 3-layer palette (accepts {hex, alpha} or bare hex)
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
  fontDisplay: "Poppins, system-ui, sans-serif"   # optional
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
  /* Any additional selector-level tweaks */
```

Atualize o dashboard após criar o arquivo. Troque themes ao vivo pela barra do header — clique o ícone de paleta. A seleção persiste em `config.yaml` sob `dashboard.theme` e é restaurada no reload.

---

## Plugins {#plugins}

Um plugin de dashboard é um diretório com `manifest.json`, bundle JS pré-construído e, opcionalmente, arquivo CSS e arquivo Python com rotas FastAPI. Plugins vivem junto aos outros plugins Hermes em `~/.hermes/plugins/<name>/` — a extensão do dashboard é uma subpasta `dashboard/` dentro desse diretório de plugin, então um plugin pode estender CLI/gateway e dashboard num único install.

Plugins não empacotam React nem componentes de UI. Eles usam o **Plugin SDK** exposto em `window.__HERMES_PLUGIN_SDK__`. Isso mantém bundles de plugin pequenos (tipicamente alguns KB) e evita conflitos de versão.

### Início rápido — seu primeiro plugin {#quick-start--your-first-plugin}

Crie a estrutura de diretórios:

```bash
mkdir -p ~/.hermes/plugins/my-plugin/dashboard/dist
```

Escreva o manifest:

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

Escreva o bundle JS (IIFE simples — sem build step):

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

Atualize o dashboard — sua aba aparece na nav bar, depois de **Skills**.

:::tip Pule React.createElement
Se preferir JSX, use qualquer bundler (esbuild, Vite, rollup) com React como external e saída IIFE. O único requisito rígido é que o arquivo final seja um único JS carregável via `<script>`. React nunca é empacotado; vem de `SDK.React`.
:::

### Layout de diretório {#directory-layout}

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml              # optional — existing CLI/gateway plugin manifest
├── __init__.py              # optional — existing CLI/gateway hooks
└── dashboard/               # dashboard extension
    ├── manifest.json        # required — tab config, icon, entry point
    ├── dist/
    │   ├── index.js         # required — pre-built JS bundle (IIFE)
    │   └── style.css        # optional — custom CSS
    └── plugin_api.py        # optional — backend API routes (FastAPI)
```

Um único diretório de plugin pode carregar três extensões ortogonais:

- `plugin.yaml` + `__init__.py` — plugin CLI/gateway ([veja página de plugins](./plugins)).
- `dashboard/manifest.json` + `dashboard/dist/index.js` — plugin UI do dashboard.
- `dashboard/plugin_api.py` — rotas backend do dashboard.

Nenhuma é obrigatória; inclua só as camadas que precisar.

### Referência do manifest {#manifest-reference}

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

| Field | Obrigatório | Descrição |
|-------|----------|-------------|
| `name` | Sim | Identificador único do plugin. Minúsculas, hífens ok. Usado em URLs e registro. |
| `label` | Sim | Nome de exibição na aba da nav. |
| `description` | Não | Descrição curta (mostrada em superfícies admin do dashboard). |
| `icon` | Não | Nome de ícone Lucide. Padrão `Puzzle`. Nomes desconhecidos fazem fallback para `Puzzle`. |
| `version` | Não | String semver. Padrão `0.0.0`. |
| `tab.path` | Sim | Caminho URL da aba (ex.: `/my-plugin`). |
| `tab.position` | Não | Onde inserir a aba. `"end"` (padrão), `"after:<path>"`, ou `"before:<path>"` — o valor após os dois pontos é o **segmento de path** da aba alvo (sem barra inicial). Exemplos: `"after:skills"`, `"before:config"`. |
| `tab.override` | Não | Defina como path de rota built-in (`"/"`, `"/sessions"`, `"/config"`, ...) para **substituir** essa página em vez de adicionar aba nova. Veja [Substituindo páginas built-in](#replacing-built-in-pages-taboverride). |
| `tab.hidden` | Não | Quando true, registra o componente e quaisquer slots sem adicionar aba na nav. Usado por plugins só de slot. Veja [Plugins só de slot](#slot-only-plugins-tabhidden). |
| `slots` | Não | Shell slots nomeados que este plugin popula. **Só ajuda de documentação** — o registro real acontece no bundle JS via `registerSlot()`. Listar slots aqui torna superfícies de discovery mais informativas. |
| `entry` | Sim | Caminho do bundle JS relativo a `dashboard/`. Padrão `dist/index.js`. |
| `css` | Não | Caminho de arquivo CSS para injetar como tag `<link>`. |
| `api` | Não | Caminho de arquivo Python com rotas FastAPI. Montado em `/api/plugins/<name>/`. |

#### Ícones disponíveis {#available-icons}

Plugins usam nomes de ícones Lucide. O dashboard mapeia por nome — nomes desconhecidos fazem fallback silencioso para `Puzzle`.

Atualmente mapeados: `Activity`, `BarChart3`, `Clock`, `Code`, `Database`, `Eye`, `FileText`, `Globe`, `Heart`, `KeyRound`, `MessageSquare`, `Package`, `Puzzle`, `Settings`, `Shield`, `Sparkles`, `Star`, `Terminal`, `Wrench`, `Zap`.

Precisa de outro ícone? Abra um PR no `ICON_MAP` de `web/src/App.tsx` — mudança puramente aditiva.

### O Plugin SDK {#the-plugin-sdk}

Tudo que um plugin precisa está em `window.__HERMES_PLUGIN_SDK__`. Plugins nunca devem importar React diretamente.

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

#### Chamando o backend do seu plugin {#calling-your-plugins-backend}

```javascript
SDK.fetchJSON("/api/plugins/my-plugin/data")
  .then((data) => console.log(data))
  .catch((err) => console.error("API call failed:", err));
```

`fetchJSON` injeta o token de auth da sessão, expõe erros como exceções lançadas e faz parse de JSON automaticamente.

#### Chamando endpoints built-in do Hermes {#calling-built-in-hermes-endpoints}

```javascript
// Agent status
SDK.api.getStatus().then((s) => console.log("Version:", s.version));

// Recent sessions
SDK.api.getSessions(10).then((resp) => console.log(resp.sessions.length));
```

Veja [Web Dashboard → REST API](./web-dashboard#rest-api) para a lista completa.

### Shell slots {#shell-slots}

Slots permitem que um plugin injete componentes em locais nomeados do app shell — sidebar cockpit, header, footer, camada overlay — sem reivindicar uma aba inteira. Vários plugins podem popular o mesmo slot; renderizam empilhados na ordem de registro.

Registre de dentro do bundle do plugin:

```javascript
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sidebar", MySidebar);
window.__HERMES_PLUGINS__.registerSlot("my-plugin", "header-left", MyCrest);
```

#### Catálogo de slots {#slot-catalogue}

**Shell-wide slots** (renderizam em qualquer lugar do chrome do app):

| Slot | Localização |
|------|-------------|
| `backdrop` | Dentro da pilha de camadas `<Backdrop />`, acima da camada noise. |
| `header-left` | Antes da marca Hermes na barra superior. |
| `header-right` | Antes dos switchers de theme/idioma na barra superior. |
| `header-banner` | Faixa full-width abaixo da nav. |
| `sidebar` | Sidebar rail cockpit — **só renderizado quando `layoutVariant === "cockpit"`**. |
| `pre-main` | Acima do route outlet (dentro de `<main>`). |
| `post-main` | Abaixo do route outlet (dentro de `<main>`). |
| `footer-left` | Conteúdo da célula do footer (substitui o padrão). |
| `footer-right` | Conteúdo da célula do footer (substitui o padrão). |
| `overlay` | Camada fixed-position acima de tudo. Útil para chrome (scanlines, vignettes) que `customCSS` sozinho não alcança. |

**Page-scoped slots** (renderizam só na página built-in nomeada — use para injetar widgets, cards ou toolbars numa página existente sem sobrescrever a rota inteira):

| Slot | Onde renderiza |
|------|----------------|
| `sessions:top` / `sessions:bottom` | Topo / base da página `/sessions`. |
| `analytics:top` / `analytics:bottom` | Topo / base da página `/analytics`. |
| `logs:top` / `logs:bottom` | Topo (acima da toolbar de filtro) / base (abaixo do log viewer) de `/logs`. |
| `cron:top` / `cron:bottom` | Topo / base da página `/cron`. |
| `skills:top` / `skills:bottom` | Topo / base da página `/skills`. |
| `config:top` / `config:bottom` | Topo / base da página `/config`. |
| `env:top` / `env:bottom` | Topo / base da página `/env` (Keys). |
| `docs:top` / `docs:bottom` | Topo (acima do iframe) / base de `/docs`. |
| `chat:top` / `chat:bottom` | Topo / base de `/chat` (só ativo quando chat embutido está habilitado). |

Exemplo — adicione um card banner no topo da página Sessions:

```javascript
function PinnedSessionsBanner() {
  return React.createElement(Card, null,
    React.createElement(CardContent, { className: "py-2 text-xs" },
      "Pinned note injected by my-plugin"),
  );
}

window.__HERMES_PLUGINS__.registerSlot("my-plugin", "sessions:top", PinnedSessionsBanner);
```

Combine page-scoped slots com `tab.hidden: true` se seu plugin só augmenta páginas existentes e não precisa de aba própria na sidebar.

O shell só renderiza `<PluginSlot name="..." />` para os slots acima. Nomes adicionais são aceitos pelo registry para UIs de plugin aninhadas — um plugin pode expor seus próprios slots via `SDK.components.PluginSlot`.

#### Re-registro e HMR {#re-registration-and-hmr}

Se o mesmo par `(plugin, slot)` for registrado duas vezes, a chamada posterior substitui a anterior — isso corresponde ao comportamento esperado pelo HMR do React em re-mounts de plugin.

### Substituindo páginas built-in (`tab.override`) {#replacing-built-in-pages-taboverride}

Definir `tab.override` como path de rota built-in faz o componente do plugin substituir essa página em vez de adicionar aba nova. Útil quando um theme quer home page customizada (`/`) mas quer manter o resto do dashboard intacto.

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

Com `override` definido:

- O componente de página original em `/` é removido do router.
- Seu plugin renderiza em `/` no lugar.
- Nenhuma aba nav é adicionada para `tab.path` (o override é o ponto).

Só um plugin pode sobrescrever um path dado. Se dois plugins reivindicarem o mesmo override, o primeiro vence e o segundo é ignorado com aviso em dev-mode.

Se você só precisa adicionar card ou toolbar a uma página existente sem tomá-la, use [page-scoped slots](#augmenting-built-in-pages-page-scoped-slots) em vez disso.

### Augmentando páginas built-in (page-scoped slots) {#augmenting-built-in-pages-page-scoped-slots}

Substituição completa via `tab.override` é pesada — seu plugin agora possui a página inteira, incluindo atualizações futuras que enviarmos. Na maioria das vezes você só quer adicionar banner, card ou toolbar a uma página existente. É para isso que servem os **page-scoped slots**.

Toda página built-in expõe slots `<page>:top` e `<page>:bottom` renderizados no topo e na base da área de conteúdo. Seu plugin popula um chamando `registerSlot()` — a página built-in continua funcionando normalmente e seu componente renderiza ao lado dela.

Slots disponíveis: `sessions:*`, `analytics:*`, `logs:*`, `cron:*`, `skills:*`, `config:*`, `env:*`, `docs:*`, `chat:*` (cada um com `:top` e `:bottom`). Veja o catálogo completo em [Shell slots → Catálogo de slots](#slot-catalogue).

Exemplo mínimo — fixe um banner no topo da página Sessions:

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

  // Placeholder for the hidden tab.
  window.__HERMES_PLUGINS__.register("session-notes", function () { return null; });

  // The real work.
  window.__HERMES_PLUGINS__.registerSlot("session-notes", "sessions:top", Banner);
})();
```

Pontos-chave:

- `tab.hidden: true` mantém o plugin fora da sidebar — não tem página standalone.
- O campo `slots` do manifest é só documentação. O binding real acontece no bundle JS via `registerSlot()`.
- Vários plugins podem reivindicar o mesmo page-scoped slot. Renderizam empilhados na ordem de registro.
- Zero footprint quando nenhum plugin registra: a página built-in renderiza exatamente como antes.

Um plugin de referência (`example-dashboard` em [`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins/tree/main/example-dashboard)) inclui demo ao vivo que injeta banner em `sessions:top` — instale para ver o padrão end-to-end.

### Plugins só de slot (`tab.hidden`) {#slot-only-plugins-tabhidden}

Quando `tab.hidden: true`, o plugin registra seu componente (para visitas diretas por URL) e quaisquer slots, mas nunca adiciona aba na navegação. Usado por plugins que só existem para injetar em slots — crest no header, HUD na sidebar, overlay.

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

O bundle ainda chama `register()` com componente placeholder (boa prática caso alguém acesse a URL diretamente) e depois `registerSlot()` para fazer o trabalho real.

### Rotas de API backend {#backend-api-routes}

Plugins podem registrar rotas FastAPI definindo `api` no manifest. Crie o arquivo e exporte um `router`:

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

Rotas são montadas sob `/api/plugins/<name>/`, então o acima vira:

- `GET  /api/plugins/my-plugin/data`
- `POST /api/plugins/my-plugin/action`

Rotas de API de plugin ignoram autenticação por session-token porque o servidor do dashboard faz bind em localhost por padrão. **Não exponha o dashboard numa interface pública com `--host 0.0.0.0` se rodar plugins não confiáveis** — as rotas deles também ficam acessíveis.

#### Acessando internals do Hermes {#accessing-hermes-internals}

Rotas backend rodam dentro do processo do dashboard, então podem importar diretamente do codebase hermes-agent:

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

### CSS customizado por plugin {#custom-css-per-plugin}

Se seu plugin precisa de estilos além de classes Tailwind e `style=` inline, adicione arquivo CSS e referencie no manifest:

```json
{
  "css": "dist/style.css"
}
```

O arquivo é injetado como tag `<link>` no carregamento do plugin. Use nomes de classe específicos para evitar conflitos com estilos do dashboard e referencie CSS vars do dashboard para permanecer theme-aware:

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

O dashboard expõe todo token shadcn como `--color-*` mais extras de theme (`--theme-asset-*`, `--component-<bucket>-*`, `--radius`, `--spacing-mul`). Referencie esses e seu plugin reskina automaticamente com o theme ativo.

### Discovery e reload de plugins {#plugin-discovery--reload}

O dashboard escaneia três diretórios por `dashboard/manifest.json`:

| Prioridade | Diretório | Source label |
|----------|-----------|--------------|
| 1 (vence em conflito) | `~/.hermes/plugins/<name>/dashboard/` | `user` |
| 2 | `<repo>/plugins/memory/<name>/dashboard/` | `bundled` |
| 2 | `<repo>/plugins/<name>/dashboard/` | `bundled` |
| 3 | `./.hermes/plugins/<name>/dashboard/` | `project` — só quando `HERMES_ENABLE_PROJECT_PLUGINS` está definido |

Resultados de discovery são cacheados por processo do dashboard. Após adicionar plugin novo, ou:

```bash
# Force a rescan without restart
curl http://127.0.0.1:9119/api/dashboard/plugins/rescan
```

…ou reinicie `hermes dashboard`.

#### Ciclo de vida de carregamento do plugin {#plugin-load-lifecycle}

1. Dashboard carrega. `main.tsx` expõe o SDK em `window.__HERMES_PLUGIN_SDK__` e o registry em `window.__HERMES_PLUGINS__`.
2. `App.tsx` chama `usePlugins()` → busca `GET /api/dashboard/plugins`.
3. Para cada manifest: `<link>` CSS é injetado (se declarado), depois tag `<script>` carrega o bundle JS.
4. O IIFE do plugin roda e chama `window.__HERMES_PLUGINS__.register(name, Component)` — e opcionalmente `.registerSlot(name, slot, Component)` para cada slot.
5. O dashboard resolve o componente registrado contra o manifest, adiciona a aba na navegação (a menos que `hidden`) e monta o componente como rota.

Plugins têm até **2 segundos** após o script carregar para chamar `register()`. Depois disso o dashboard para de esperar e finaliza o render inicial. Se um plugin registrar depois, ainda aparece — a nav é reativa.

Se o script de um plugin falhar ao carregar (404, erro de sintaxe, exceção durante IIFE), o dashboard registra aviso no console do browser e continua sem ele.

---

## Demo combinado theme + plugin {#combined-theme--plugin-demo}

O plugin [`strike-freedom-cockpit`](https://github.com/NousResearch/hermes-example-plugins/tree/main/strike-freedom-cockpit) (repo companion `hermes-example-plugins`) é um demo completo de reskin. Ele combina YAML de theme com plugin só de slot para produzir HUD estilo cockpit sem fork do dashboard.

**O que demonstra:**

- Theme completo usando palette, tipografia, `fontUrl`, `layoutVariant: cockpit`, `assets`, `componentStyles` (cantos de card entalhados, backgrounds gradiente), `colorOverrides` e `customCSS` (overlay scanline).
- Plugin só de slot (`tab.hidden: true`) que registra em três slots:
  - `sidebar` — painel MS-STATUS com barras de telemetria ao vivo via `SDK.api.getStatus()`.
  - `header-left` — crest de facção que lê `--theme-asset-crest` do theme ativo.
  - `footer-right` — tagline customizada substituindo a linha org padrão.
- O plugin lê artwork fornecido pelo theme via CSS vars, então trocar themes muda hero/crest sem mudar código do plugin.

**Instalação:**

```bash
git clone https://github.com/NousResearch/hermes-example-plugins.git

# Theme
cp hermes-example-plugins/strike-freedom-cockpit/theme/strike-freedom.yaml \
   ~/.hermes/dashboard-themes/

# Plugin
cp -r hermes-example-plugins/strike-freedom-cockpit ~/.hermes/plugins/
```

Abra o dashboard, escolha **Strike Freedom** no theme switcher. A sidebar cockpit aparece, o crest mostra no header, a tagline substitui o footer. Volte para **Hermes Teal** e o plugin permanece instalado mas invisível (o slot `sidebar` só renderiza sob a variante de layout `cockpit`).

Leia o source do plugin (`strike-freedom-cockpit/dashboard/dist/index.js` no repo companion) para ver como lê CSS vars, protege contra dashboards antigos sem suporte a slot e registra três slots num bundle.

---

## Referência de API {#api-reference}

### Endpoints de theme {#theme-endpoints}

| Endpoint | Method | Descrição |
|----------|--------|-------------|
| `/api/dashboard/themes` | GET | Lista themes disponíveis + nome ativo. Built-ins retornam `{name, label, description}`; themes de usuário também incluem campo `definition` com o objeto theme normalizado completo. |
| `/api/dashboard/theme` | PUT | Define theme ativo. Body: `{"name": "midnight"}`. Persiste em `config.yaml` sob `dashboard.theme`. |

### Endpoints de plugin {#plugin-endpoints}

| Endpoint | Method | Descrição |
|----------|--------|-------------|
| `/api/dashboard/plugins` | GET | Lista plugins descobertos (com manifests, menos campos internos). |
| `/api/dashboard/plugins/rescan` | GET | Força re-scan dos diretórios de plugin sem reiniciar. |
| `/dashboard-plugins/<name>/<path>` | GET | Serve assets estáticos do diretório `dashboard/` de um plugin. Path traversal é bloqueado. |
| `/api/plugins/<name>/*` | * | Rotas backend registradas pelo plugin. |

### SDK em `window` {#sdk-on-window}

| Global | Type | Provider |
|--------|------|----------|
| `window.__HERMES_PLUGIN_SDK__` | object | `registry.ts` — React, hooks, componentes de UI, client de API, utils. |
| `window.__HERMES_PLUGINS__.register(name, Component)` | function | Registra o componente principal de um plugin. |
| `window.__HERMES_PLUGINS__.registerSlot(name, slot, Component)` | function | Registra num shell slot nomeado. |

---

## Solução de problemas {#troubleshooting}

**Meu theme não aparece no picker.**
Verifique se o arquivo está em `~/.hermes/dashboard-themes/` e termina em `.yaml` ou `.yml`. Atualize a página. Rode `curl http://127.0.0.1:9119/api/dashboard/themes` — seu theme deve estar na resposta. Se o YAML tiver erro de parse, o dashboard registra em `errors.log` sob `~/.hermes/logs/`.

**A aba do meu plugin não aparece.**
1. Verifique se o manifest está em `~/.hermes/plugins/<name>/dashboard/manifest.json` (note a subpasta `dashboard/`).
2. `curl http://127.0.0.1:9119/api/dashboard/plugins/rescan` para forçar re-discovery.
3. Abra dev tools do browser → Network — confirme que `manifest.json`, `index.js` e qualquer CSS carregaram sem 404s.
4. Abra dev tools do browser → Console — procure erros durante o IIFE ou `window.__HERMES_PLUGINS__ is undefined` (indica que o SDK não inicializou, geralmente crash de render React anterior).
5. Verifique se seu bundle chama `window.__HERMES_PLUGINS__.register(...)` com o **mesmo name** que `manifest.json:name`.

**Componentes registrados em slot não renderizam.**
O slot `sidebar` só renderiza quando o theme ativo tem `layoutVariant: cockpit`. Outros slots sempre renderizam. Se você registra num slot sem efeito, adicione `console.log` dentro de `registerSlot` para confirmar que o bundle do plugin rodou.

**Rotas backend do plugin retornam 404.**
1. Confirme que o manifest tem `"api": "plugin_api.py"` apontando para arquivo existente dentro de `dashboard/`.
2. Reinicie `hermes dashboard` — rotas de API de plugin são montadas uma vez na inicialização, **não** no rescan.
3. Verifique se `plugin_api.py` exporta `router = APIRouter()` em nível de módulo. Outros nomes de export não são detectados.
4. Acompanhe `~/.hermes/logs/errors.log` por `Failed to load plugin <name> API routes` — erros de import são registrados lá.

**Troca de theme remove meus color overrides.**
`colorOverrides` são scoped ao theme ativo e limpos na troca de theme — isso é intencional. Se quiser overrides que persistem, coloque-os no YAML do seu theme, não no switcher ao vivo.

**customCSS do theme é truncado.**
O bloco `customCSS` é limitado a 32 KiB por theme. Divida stylesheets grandes entre vários themes, ou mude para plugin que injeta stylesheet completo via campo `css` (sem limite de tamanho).

**Quero publicar um plugin no PyPI.**
Plugins de dashboard são instalados por layout de diretório, não por pip entry point. O caminho de distribuição mais limpo hoje é um repo git que o usuário clona em `~/.hermes/plugins/`. Um instalador pip para plugins de dashboard não está conectado atualmente.
