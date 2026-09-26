---
sidebar_position: 10
title: "Skins & Themes"
description: "Customize a CLI do Hermes com skins built-in e definidas pelo usuário"
---

# Skins & Themes

Skins controlam a **apresentação visual** da CLI do Hermes: cores do banner, faces e verbos do spinner, rótulos da caixa de resposta, texto de branding e o prefixo de atividade de tools.

Estilo conversacional e estilo visual são conceitos separados:

- **Personality** muda o tom e a redação do agente.
- **Skin** muda a aparência da CLI.

## Change skins {#change-skins}

```bash
/skin                # show the current skin and list available skins
/skin ares           # switch to a built-in skin
/skin mytheme        # switch to a custom skin from ~/.hermes/skins/mytheme.yaml
```

Ou defina a skin padrão em `~/.hermes/config.yaml`:

```yaml
display:
  skin: default
```

## Built-in skins {#built-in-skins}

| Skin | Descrição | Branding do agente | Caráter visual |
|------|-------------|----------------|------------------|
| `default` | Hermes clássico — dourado e kawaii | `Hermes Agent` | Bordas douradas quentes, texto cornsilk, faces kawaii nos spinners. O banner caduceu familiar. Limpo e acolhedor. |
| `ares` | Tema de deus da guerra — carmesim e bronze | `Ares Agent` | Bordas carmesim profundas com acentos bronze. Verbos agressivos no spinner ("forging", "marching", "tempering steel"). Banner ASCII customizado de espada e escudo. |
| `mono` | Monocromático — grayscale limpo | `Hermes Agent` | Tudo em tons de cinza — sem cor. Bordas `#555555`, texto `#c9d1d9`. Ideal para setups de terminal minimalistas ou gravações de tela. |
| `slate` | Azul frio — focado em desenvolvedores | `Hermes Agent` | Bordas azul royal (`#4169e1`), texto azul suave. Calmo e profissional. Sem spinner customizado — usa faces padrão. |
| `daylight` | Tema claro para terminais brilhantes com texto escuro e acentos azul frio | `Hermes Agent` | Pensado para terminais brancos ou claros. Texto slate escuro com bordas azuis, superfícies de status pálidas e menu de completion claro legível em perfis de terminal light. |
| `warm-lightmode` | Texto marrom/dourado quente para fundos de terminal claros | `Hermes Agent` | Tons pergaminho quentes para terminais claros. Texto marrom escuro com acentos saddle-brown, superfícies de status creme. Alternativa terrosa ao tema daylight mais frio. |
| `poseidon` | Tema de deus oceânico — azul profundo e espuma do mar | `Poseidon Agent` | Gradiente de azul profundo a espuma do mar. Spinners com tema oceânico ("charting currents", "sounding the depth"). Banner ASCII de tridente. |
| `sisyphus` | Tema sísifo — grayscale austero com persistência | `Sisyphus Agent` | Cinzas claros com contraste forte. Spinners com tema de rocha ("pushing uphill", "resetting the boulder", "enduring the loop"). Banner ASCII de rocha e colina. |
| `charizard` | Tema vulcânico — laranja queimado e brasa | `Charizard Agent` | Gradiente quente de laranja queimado a brasa. Spinners com tema de fogo ("banking into the draft", "measuring burn"). Banner ASCII de silhueta de dragão. |

## Complete list of configurable keys {#complete-list-of-configurable-keys}

### Colors (`colors:`) {#colors}

Controla todos os valores de cor na CLI. Valores são strings hex de cor.

| Key | Description | Default (`default` skin) |
|-----|-------------|--------------------------|
| `banner_border` | Panel border around the startup banner | `#CD7F32` (bronze) |
| `banner_title` | Title text color in the banner | `#FFD700` (gold) |
| `banner_accent` | Section headers in the banner (Available Tools, etc.) | `#FFBF00` (amber) |
| `banner_dim` | Muted text in the banner (separators, secondary labels) | `#B8860B` (dark goldenrod) |
| `banner_text` | Body text in the banner (tool names, skill names) | `#FFF8DC` (cornsilk) |
| `ui_accent` | General UI accent color (highlights, active elements) | `#FFBF00` |
| `ui_label` | UI labels and tags | `#4dd0e1` (teal) |
| `ui_ok` | Success indicators (checkmarks, completion) | `#4caf50` (green) |
| `ui_error` | Error indicators (failures, blocked) | `#ef5350` (red) |
| `ui_warn` | Warning indicators (caution, approval prompts) | `#ffa726` (orange) |
| `prompt` | Interactive prompt text color | `#FFF8DC` |
| `input_rule` | Horizontal rule above the input area | `#CD7F32` |
| `response_border` | Border around the agent's response box (ANSI escape) | `#FFD700` |
| `session_label` | Session label color | `#DAA520` |
| `session_border` | Session ID dim border color | `#8B8682` |
| `status_bar_bg` | Background color for the TUI status / usage bar | `#1a1a2e` |
| `voice_status_bg` | Background color for the voice-mode status badge | `#1a1a2e` |
| `selection_bg` | Background color for the TUI mouse-selection highlighter. Falls back to `completion_menu_current_bg` when unset. | `#333355` |
| `completion_menu_bg` | Background color for the completion menu list | `#1a1a2e` |
| `completion_menu_current_bg` | Background color for the active completion row | `#333355` |
| `completion_menu_meta_bg` | Background color for the completion meta column | `#1a1a2e` |
| `completion_menu_meta_current_bg` | Background color for the active completion meta column | `#333355` |

### Spinner (`spinner:`) {#spinner}

Controla o spinner animado exibido enquanto aguarda respostas da API.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `waiting_faces` | list of strings | Faces cycled while waiting for API response | `["(⚔)", "(⛨)", "(▲)"]` |
| `thinking_faces` | list of strings | Faces cycled during model reasoning | `["(⚔)", "(⌁)", "(<>)"]` |
| `thinking_verbs` | list of strings | Verbs shown in spinner messages | `["forging", "plotting", "hammering plans"]` |
| `wings` | list of [left, right] pairs | Decorative brackets around the spinner | `[["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"]]` |

Quando valores do spinner estão vazios (como em `default` e `mono`), defaults hardcoded de `display.py` são usados.

### Branding (`branding:`) {#branding}

Strings de texto usadas na interface da CLI.

| Key | Description | Default |
|-----|-------------|---------|
| `agent_name` | Name shown in banner title and status display | `Hermes Agent` |
| `welcome` | Welcome message shown at CLI startup | `Welcome to Hermes Agent! Type your message or /help for commands.` |
| `goodbye` | Message shown on exit | `Goodbye! ⚕` |
| `response_label` | Label on the response box header | ` ⚕ Hermes ` |
| `prompt_symbol` | Symbol before the user input prompt (bare token, renderers add a trailing space) | `❯` |
| `help_header` | Header text for the `/help` command output | `(^_^)? Available Commands` |

### Other top-level keys {#other-top-level-keys}

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `tool_prefix` | string | Character prefixed to tool output lines in the CLI | `┊` |
| `tool_emojis` | dict | Per-tool emoji overrides for spinners and progress (`{tool_name: emoji}`) | `{}` |
| `banner_logo` | string | Rich-markup ASCII art logo (replaces the default HERMES_AGENT banner) | `""` |
| `banner_hero` | string | Rich-markup hero art (replaces the default caduceus art) | `""` |

## Custom skins {#custom-skins}

Crie arquivos YAML em `~/.hermes/skins/`. Skins do usuário herdam valores ausentes da skin built-in `default`, então você só precisa especificar as chaves que quer mudar.

### Full custom skin YAML template {#full-custom-skin-yaml-template}

```yaml
# ~/.hermes/skins/mytheme.yaml
# Complete skin template — all keys shown. Delete any you don't need;
# missing values automatically inherit from the 'default' skin.

name: mytheme
description: My custom theme

colors:
  banner_border: "#CD7F32"
  banner_title: "#FFD700"
  banner_accent: "#FFBF00"
  banner_dim: "#B8860B"
  banner_text: "#FFF8DC"
  ui_accent: "#FFBF00"
  ui_label: "#4dd0e1"
  ui_ok: "#4caf50"
  ui_error: "#ef5350"
  ui_warn: "#ffa726"
  prompt: "#FFF8DC"
  input_rule: "#CD7F32"
  response_border: "#FFD700"
  session_label: "#DAA520"
  session_border: "#8B8682"
  status_bar_bg: "#1a1a2e"
  voice_status_bg: "#1a1a2e"
  selection_bg: "#333355"
  completion_menu_bg: "#1a1a2e"
  completion_menu_current_bg: "#333355"
  completion_menu_meta_bg: "#1a1a2e"
  completion_menu_meta_current_bg: "#333355"

spinner:
  waiting_faces:
    - "(⚔)"
    - "(⛨)"
    - "(▲)"
  thinking_faces:
    - "(⚔)"
    - "(⌁)"
    - "(<>)"
  thinking_verbs:
    - "processing"
    - "analyzing"
    - "computing"
    - "evaluating"
  wings:
    - ["⟪⚡", "⚡⟫"]
    - ["⟪●", "●⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome to My Agent! Type your message or /help for commands."
  goodbye: "See you later! ⚡"
  response_label: " ⚡ My Agent "
  prompt_symbol: "⚡"
  help_header: "(⚡) Available Commands"

tool_prefix: "┊"

# Per-tool emoji overrides (optional)
tool_emojis:
  terminal: "⚔"
  web_search: "🔮"
  read_file: "📄"

# Custom ASCII art banners (optional, Rich markup supported)
# banner_logo: |
#   [bold #FFD700] MY AGENT [/]
# banner_hero: |
#   [#FFD700]  Custom art here  [/]
```

### Minimal custom skin example {#minimal-custom-skin-example}

Como tudo herda de `default`, uma skin mínima só precisa mudar o que é diferente:

```yaml
name: cyberpunk
description: Neon terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

## Hermes Mod — Visual Skin Editor {#hermes-mod--visual-skin-editor}

[Hermes Mod](https://github.com/cocktailpeanut/hermes-mod) é uma UI web da comunidade para criar e gerenciar skins visualmente. Em vez de escrever YAML à mão, você tem um editor point-and-click com preview ao vivo.

![Hermes Mod skin editor](https://raw.githubusercontent.com/cocktailpeanut/hermes-mod/master/nous.png)

**O que faz:**

- Lista todas as skins built-in e customizadas
- Abre qualquer skin em um editor visual com todos os campos de skin do Hermes (colors, spinner, branding, tool prefix, tool emojis)
- Gera `banner_logo` text art a partir de um prompt de texto
- Converte imagens enviadas (PNG, JPG, GIF, WEBP) em ASCII art `banner_hero` com vários estilos de render (braille, ASCII ramp, blocks, dots)
- Salva diretamente em `~/.hermes/skins/`
- Ativa uma skin atualizando `~/.hermes/config.yaml`
- Mostra o YAML gerado e um preview ao vivo

### Install {#install}

**Opção 1 — Pinokio (1 clique):**

Encontre em [pinokio.computer](https://pinokio.computer) e instale com um clique.

**Opção 2 — npx (mais rápido pelo terminal):**

```bash
npx -y hermes-mod
```

**Opção 3 — Manual:**

```bash
git clone https://github.com/cocktailpeanut/hermes-mod.git
cd hermes-mod/app
npm install
npm start
```

### Usage {#usage}

1. Inicie o app (via Pinokio ou terminal).
2. Abra **Skin Studio**.
3. Escolha uma skin built-in ou customizada para editar.
4. Gere um logo a partir de texto e/ou envie uma imagem para hero art. Escolha estilo de render e largura.
5. Edite colors, spinner, branding e outros campos.
6. Clique **Save** para gravar o YAML da skin em `~/.hermes/skins/`.
7. Clique **Activate** para defini-la como skin atual (atualiza `display.skin` em `config.yaml`).

O Hermes Mod respeita a variável de ambiente `HERMES_HOME`, então funciona com [profiles](/user-guide/profiles) também.

## Operational notes {#operational-notes}

- Skins built-in carregam de `hermes_cli/skin_engine.py`.
- Skins desconhecidas fazem fallback automático para `default`.
- `/skin` atualiza o tema CLI ativo imediatamente para a sessão atual.
- Skins do usuário em `~/.hermes/skins/` têm precedência sobre skins built-in com o mesmo nome.
- Mudanças de skin via `/skin` são só da sessão. Para tornar uma skin seu padrão permanente, defina em `config.yaml`.
- Os campos `banner_logo` e `banner_hero` suportam markup Rich console (por exemplo, `[bold #FF0000]text[/]`) para ASCII art colorido.
