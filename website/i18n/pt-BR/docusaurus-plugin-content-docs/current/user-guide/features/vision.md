---
title: Visão e colagem de imagem
description: Cole imagens da área de transferência no CLI do Hermes para análise de visão multimodal.
sidebar_label: Visão e colagem de imagem
sidebar_position: 7
---

# Visão e colagem de imagem

O Hermes Agent suporta **visão multimodal** — você pode colar imagens da área de transferência diretamente no CLI e pedir ao agente para analisá-las, descrevê-las ou trabalhar com elas. Imagens são enviadas ao modelo como blocos de conteúdo codificados em base64, então qualquer modelo com visão pode processá-las.

:::tip
Assinantes Portal obtêm modelos com visão (Claude, GPT-5, Gemini) no mesmo catálogo — sem credenciais extras. Veja [Nous Portal](/integrations/nous-portal).
:::

## Como funciona {#how-it-works}

1. Copie uma imagem para a área de transferência (screenshot, imagem do navegador, etc.)
2. Anexe usando um dos métodos abaixo
3. Digite sua pergunta e pressione Enter
4. A imagem aparece como badge `[📎 Image #1]` acima do input
5. Ao enviar, a imagem vai ao modelo como bloco de conteúdo de visão

Você pode anexar várias imagens antes de enviar — cada uma recebe seu próprio badge. Pressione `Ctrl+C` para limpar todas as imagens anexadas.

Imagens são salvas em `~/.hermes/images/` como arquivos PNG com nomes timestamped.

## Métodos de colagem {#paste-methods}

Como você anexa uma imagem depende do seu ambiente de terminal. Nem todo método funciona em todo lugar — aqui está o breakdown completo:

### Comando `/paste` {#paste-command}

**O fallback explícito de anexar imagem mais confiável.**

```
/paste
```

Digite `/paste` e pressione Enter. O Hermes verifica sua área de transferência por uma imagem e a anexa. É a opção mais segura quando seu terminal reescreve `Cmd+V`/`Ctrl+V`, ou quando você copiou apenas uma imagem e não há payload de texto bracketed-paste para inspecionar.

### Ctrl+V / Cmd+V {#ctrlv--cmdv}

O Hermes agora trata colagem como fluxo em camadas:
- colagem de texto normal primeiro
- fallback nativo de clipboard / OSC52 se o terminal não entregou texto limpo
- anexar imagem quando clipboard ou payload colado resolve para imagem ou caminho de imagem

Isso significa que caminhos temp de screenshot macOS e URIs de imagem `file://...` podem anexar imediatamente em vez de ficar no composer como texto cru.

:::warning
Se sua área de transferência tem **apenas uma imagem** (sem texto), terminais ainda não podem enviar bytes binários de imagem diretamente. Use `/paste` como fallback explícito de anexar imagem.
:::

### `/terminal-setup` para VS Code / Cursor / Windsurf {#terminal-setup-for-vs-code--cursor--windsurf}

Se você roda o TUI dentro de um terminal integrado local da família VS Code no macOS, o Hermes pode instalar os bindings recomendados `workbench.action.terminal.sendSequence` para melhor paridade de multiline e undo/redo:

```text
/terminal-setup
```

Isso é especialmente útil quando `Cmd+Enter`, `Cmd+Z` ou `Shift+Cmd+Z` são interceptados pelo IDE. Rode apenas na máquina local — não dentro de sessão SSH.

## Compatibilidade de plataforma {#platform-compatibility}

| Ambiente | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | Notas |
|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | n/a | Melhor experiência — clipboard nativo + recuperação de caminho de screenshot |
| **Apple Terminal** | ✅ | ✅ | n/a | Se Cmd+←/→/⌫ for reescrito, use fallbacks Ctrl+A / Ctrl+E / Ctrl+U |
| **Linux X11 desktop** | ✅ | ✅ | n/a | Requer `xclip` (`apt install xclip`) |
| **Linux Wayland desktop** | ✅ | ✅ | n/a | Requer `wl-paste` (`apt install wl-clipboard`) |
| **WSL2 (Windows Terminal)** | ✅ | ✅ | n/a | Usa `powershell.exe` — sem instalação extra |
| **VS Code / Cursor / Windsurf (local)** | ✅ | ✅ | ✅ | Recomendado para melhor paridade Cmd+Enter / undo / redo |
| **VS Code / Cursor / Windsurf (SSH)** | ❌² | ❌² | ❌³ | Rode `/terminal-setup` na máquina local |
| **Terminal SSH (qualquer)** | ❌² | ❌² | n/a | Clipboard remoto inacessível |

² Veja [SSH e sessões remotas](#ssh--remote-sessions) abaixo
³ O comando escreve keybindings locais do IDE e não deve ser rodado do host remoto

## Setup específico de plataforma {#platform-specific-setup}

### macOS {#macos}

**Nenhum setup necessário.** O Hermes usa `osascript` (built-in no macOS) para ler a área de transferência. Para performance maior, instale opcionalmente `pngpaste`:

```bash
brew install pngpaste
```

### Linux (X11) {#linux-x11}

Instale `xclip`:

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland) {#linux-wayland}

Desktops Linux modernos (Ubuntu 22.04+, Fedora 34+) frequentemente usam Wayland por padrão. Instale `wl-clipboard`:

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip Como verificar se você está no Wayland
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = sem display server
```
:::

### WSL2 {#wsl2}

**Nenhum setup extra necessário.** O Hermes detecta WSL2 automaticamente (via `/proc/version`) e usa `powershell.exe` para acessar a área de transferência Windows via `System.Windows.Forms.Clipboard` do .NET. Isso é built-in na interop Windows do WSL2 — `powershell.exe` está disponível por padrão.

Os dados do clipboard são transferidos como PNG codificado em base64 via stdout, então não são necessários conversão de caminho de arquivo ou arquivos temp.

:::info Nota WSLg
Se você roda WSLg (WSL2 com suporte GUI), o Hermes tenta o caminho PowerShell primeiro, depois fallback para `wl-paste`. A ponte de clipboard do WSLg só suporta formato BMP para imagens — o Hermes converte BMP para PNG automaticamente usando Pillow (se instalado) ou o comando `convert` do ImageMagick.
:::

#### Verificar acesso ao clipboard WSL2 {#verify-wsl2-clipboard-access}

```bash
# 1. Check WSL detection
grep -i microsoft /proc/version

# 2. Check PowerShell is accessible
which powershell.exe

# 3. Copy an image, then check
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# Should print "True"
```

## SSH e sessões remotas {#ssh--remote-sessions}

**Colagem de imagem do clipboard não funciona completamente via SSH.** Quando você faz SSH em uma máquina remota, o CLI Hermes roda no host remoto. Ferramentas de clipboard (`xclip`, `wl-paste`, `powershell.exe`, `osascript`) leem a área de transferência da máquina onde rodam — que é o servidor remoto, não sua máquina local. Sua imagem local no clipboard fica inacessível do lado remoto.

Texto às vezes ainda passa por colagem de terminal ou OSC52, mas acesso ao clipboard de imagem e caminhos temp de screenshot local permanecem ligados à máquina que roda o Hermes.

### Contornos para SSH {#workarounds-for-ssh}

1. **Envie o arquivo de imagem** — Salve a imagem localmente, envie ao servidor remoto via `scp`, explorador de arquivos do VSCode (drag-and-drop) ou qualquer método de transferência. Depois referencie por caminho. *(Um comando `/attach <filepath>` está planejado para release futura.)*

2. **Use uma URL** — Se a imagem está acessível online, cole a URL na mensagem. O agente pode usar `vision_analyze` para olhar qualquer URL de imagem diretamente.

3. **X11 forwarding** — Conecte com `ssh -X` para encaminhar X11. Isso permite que `xclip` na máquina remota acesse seu clipboard X11 local. Requer servidor X rodando localmente (XQuartz no macOS, built-in em desktops Linux X11). Lento para imagens grandes.

4. **Use uma plataforma de mensagens** — Envie imagens ao Hermes via Telegram, Discord, Slack ou WhatsApp. Essas plataformas tratam upload de imagem nativamente e não são afetadas por limitações de clipboard/terminal.

## Por que terminais não conseguem colar imagens {#why-terminals-cant-paste-images}

Isso é fonte comum de confusão, então aqui está a explicação técnica:

Terminais são interfaces **baseadas em texto**. Quando você pressiona Ctrl+V (ou Cmd+V), o emulador de terminal:

1. Lê a área de transferência por **conteúdo de texto**
2. Envolve em sequências de escape [bracketed paste](https://en.wikipedia.org/wiki/Bracketed-paste)
3. Envia à aplicação pelo fluxo de texto do terminal

Se a área de transferência contém apenas uma imagem (sem texto), o terminal não tem nada para enviar. Não há sequência de escape padrão de terminal para dados binários de imagem. O terminal simplesmente não faz nada.

Por isso o Hermes usa uma verificação separada de clipboard — em vez de receber dados de imagem pelo evento de colagem do terminal, chama ferramentas em nível de OS (`osascript`, `powershell.exe`, `xclip`, `wl-paste`) diretamente via subprocess para ler a área de transferência independentemente.

## Modelos suportados {#supported-models}

Colagem de imagem funciona com qualquer modelo com visão. A imagem é enviada como data URL codificada em base64 no formato de conteúdo de visão OpenAI:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

A maioria dos modelos modernos suporta esse formato, incluindo GPT-4 Vision, Claude (com visão), Gemini e modelos multimodais open-source servidos via OpenRouter.

## Roteamento de imagem (modelos com visão vs só texto) {#image-routing-vision-capable-vs-text-only-models}

Quando um usuário anexa uma imagem — do clipboard do CLI, gateway (foto Telegram/Discord) ou qualquer outro ponto de entrada — o Hermes roteia com base em se seu modelo atual realmente suporta visão:

| Seu modelo | O que acontece com a imagem |
|---|---|
| **Com visão** (GPT-4V, Claude com visão, Gemini, Qwen-VL, MiMo-VL, etc.) | Enviada como **pixels reais** usando o formato nativo de conteúdo de imagem do provider acima. Sem camada de resumo em texto. |
| **Só texto** (DeepSeek V3, modelos open-source menores, endpoints chat-only antigos) | Roteada pela ferramenta auxiliar `vision_analyze` — um modelo auxiliar de visão descreve a imagem e a descrição em texto é injetada na conversa. |

Você não configura isso — o Hermes consulta a capacidade do seu modelo atual nos metadados do provider e escolhe o caminho certo automaticamente. O efeito prático: você pode alternar entre modelos com e sem visão no meio da sessão e o tratamento de imagem "simplesmente funciona" sem mudar seu fluxo. Modelos só texto recebem contexto coerente sobre a imagem em vez de um payload multimodal quebrado que teriam de rejeitar.

Qual modelo auxiliar trata o caminho de descrição em texto é configurável em `auxiliary.vision` — veja [Auxiliary Models](/user-guide/configuration#auxiliary-models).

### `vision_analyze` tem o mesmo comportamento dual {#vision_analyze-has-the-same-dual-behavior}

A ferramenta `vision_analyze` em si segue o mesmo roteamento. Quando o modelo principal ativo tem visão **e** seu provider suporta conteúdo de imagem dentro de resultados de ferramenta (atualmente stacks Anthropic, OpenAI, Azure-OpenAI e Gemini 3.x), `vision_analyze` encurta o descrevedor auxiliar e retorna os pixels crus da imagem como envelope multimodal de resultado de ferramenta. O modelo principal vê a imagem nativamente no próximo turno — sem chamada aux, sem perda de informação por resumo em texto, sem latência extra.

Para modelos principais só texto (ou providers cujo canal de resultado de ferramenta não carrega imagens), `vision_analyze` faz fallback ao caminho legado: pede ao modelo auxiliar de visão configurado para descrever a imagem e retorna a descrição como texto simples. De qualquer forma a assinatura da ferramenta chamadora é a mesma — a ferramenta decide qual caminho tomar em runtime com base no modelo ativo.
