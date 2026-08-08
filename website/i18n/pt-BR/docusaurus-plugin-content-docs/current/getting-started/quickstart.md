---
sidebar_position: 1
title: "Início rápido"
description: "Sua primeira conversa com o Hermes Agent — da instalação ao chat em menos de 5 minutos"
---

# Início rápido

Este guia leva você do zero até um setup Hermes que aguenta uso real. Instale, escolha um provider, verifique um chat funcionando e saiba exatamente o que fazer quando algo quebrar.

## Prefere assistir?

O **Onchain AI Garage** montou um Masterclass de instalação, setup e comandos básicos — um bom complemento a esta página se preferir seguir em vídeo. Para mais, veja a playlist completa [Hermes Agent Tutorials & Use Cases](https://www.youtube.com/playlist?list=PLmpUb_PWAkDxewld5ZYyKifuHxgIbiq2d).

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '1.5rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube-nocookie.com/embed/R3YOGfTBcQg"
    title="Hermes Agent Masterclass: Installation, Setup, Basic Commands"
    frameBorder="0"
    allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen
  ></iframe>
</div>

## Para quem é isto

- Está começando e quer o caminho mais curto até um setup funcionando
- Está trocando de provider e não quer perder tempo com erro de config
- Está montando o Hermes para um time, bot ou fluxo always-on
- Cansou de "instalou, mas ainda não faz nada"

## O caminho mais rápido

Escolha a linha que bate com o seu objetivo:

| Objetivo | Faça isto primeiro | Depois faça isto |
|---|---|---|
| Só quero o Hermes funcionando na minha máquina | `hermes setup` | Rode um chat de verdade e confirme que responde |
| Já sei o meu provider | `hermes model` | Salve a config e comece a conversar |
| Quero um bot ou setup always-on | `hermes gateway setup` depois que o CLI funcionar | Conecte Telegram, Discord, Slack ou outra plataforma |
| Quero um modelo local ou self-hosted | `hermes model` → custom endpoint | Verifique endpoint, nome do modelo e tamanho de contexto |
| Quero fallback multi-provider | `hermes model` primeiro | Só adicione routing e fallback depois que o chat base funcionar |

**Regra de ouro:** se o Hermes não completa um chat normal, não acrescente mais features ainda. Faça uma conversa limpa funcionar primeiro; depois empilhe gateway, cron, skills, voz ou routing.

---

## 1. Instalar o Hermes Agent
### Com o instalador do Hermes Desktop no macOS ou Windows (recomendado)
Para instalar com facilidade o CLI e o app desktop, [baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) no site e execute-o.

### Sem o Hermes Desktop:
Para uma instalação só de linha de comando, sem o Hermes Desktop, execute:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (nativo)

No PowerShell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

:::tip Android / Termux
Se estiver instalando no celular, veja o [guia Termux](./termux.md) dedicado para o caminho manual testado, extras suportados e limitações atuais específicas do Android.
:::

Quando terminar, recarregue o shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

Para opções detalhadas de instalação, pré-requisitos e troubleshooting, veja o [guia de Instalação](./installation.md).

## 2. Escolher um provider

O passo de setup mais importante. Use `hermes model` para escolher de forma interativa:

```bash
hermes model
```

:::tip Caminho mais fácil: Nous Portal
Uma assinatura cobre mais de 300 modelos e o [Tool Gateway](../user-guide/features/tool-gateway.md) (busca na web, geração de imagem, TTS, browser na nuvem). Numa instalação nova:

```bash
hermes setup --portal
```

Isso faz login, define a Nous como provider e liga o Tool Gateway num único comando.
:::

:::info Modos de setup
Numa instalação nova, o `hermes setup` oferece três modos:

- **Quick Setup (Nous Portal)** — login OAuth gratuito, sem API keys; configura um modelo e as tools do Tool Gateway. O caminho rápido recomendado.
- **Full Setup** — percorre cada provider, tool e opção na mão (você traz as próprias chaves).
- **Blank Slate** — tudo começa **desligado**, exceto o mínimo para rodar um agente: **provider e modelo, o toolset File Operations e o toolset Terminal**. Sem web, browser, execução de código, visão, memória, delegação, cron, skills, plugins ou servidores MCP — e compression, checkpoints, smart routing e captura de memória ficam desabilitados. Depois que a baseline mínima é aplicada, você escolhe um de dois caminhos: **começar com tudo desligado** (terminar agora com o agente mínimo) ou **percorrer todas as configurações** (optar por tools, skills, plugins, MCP e messaging). Escolha isto quando quiser um agente mínimo, totalmente controlado, e pretendia habilitar só o que precisa.

O Blank Slate grava uma lista explícita `platform_toolsets.cli` mais `agent.disabled_toolsets`, então nada que você não escolheu carrega — nem depois de um `hermes update`. Reabilite depois com `hermes tools`, faça seed de skills com `hermes skills opt-in --sync` ou ajuste settings com `hermes setup agent`.
:::

Bons defaults:

| Provider | O que é | Como configurar |
|----------|-----------|---------------|
| **Nous Portal** | Baseado em assinatura, zero-config | Login OAuth via `hermes model` |
| **OpenAI Codex** | ChatGPT OAuth, usa modelos Codex | Device code auth via `hermes model` |
| **Anthropic** | Modelos Claude direto — plano Max + créditos extras (OAuth), ou API key pay-per-token | `hermes model` → login OAuth (exige Max + créditos extras), ou uma API key Anthropic |
| **OpenRouter** | Routing multi-provider por muitos modelos | Informe sua API key |
| **Fireworks AI** | API de modelos direta compatível com OpenAI | Defina `FIREWORKS_API_KEY` |
| **Z.AI** | Modelos GLM / Zhipu-hosted | Defina `GLM_API_KEY` / `ZAI_API_KEY` (também aceita `Z_AI_API_KEY`) |
| **Kimi / Moonshot** | Modelos de coding e chat hospedados pela Moonshot | Defina `KIMI_API_KEY` (ou o `KIMI_CODING_API_KEY` específico do Kimi-Coding) |
| **Kimi / Moonshot China** | Endpoint Moonshot na região China | Defina `KIMI_CN_API_KEY` |
| **Arcee AI** | Modelos Trinity | Defina `ARCEEAI_API_KEY` |
| **GMI Cloud** | API direta multi-modelo | Defina `GMI_API_KEY` |
| **MiniMax (OAuth)** | Modelo frontier MiniMax via OAuth no browser — sem API key (o nome do modelo em `hermes_cli/models.py` pode mudar entre releases) | `hermes model` → MiniMax (OAuth) |
| **MiniMax** | Endpoint MiniMax internacional | Defina `MINIMAX_API_KEY` |
| **MiniMax China** | Endpoint MiniMax na região China | Defina `MINIMAX_CN_API_KEY` |
| **Alibaba Cloud** | Modelos Qwen via DashScope | Defina `DASHSCOPE_API_KEY` (Qwen Coding Plan também aceita `ALIBABA_CODING_PLAN_API_KEY`) |
| **Hugging Face** | Mais de 20 modelos open via router unificado (Qwen, DeepSeek, Kimi, etc.) | Defina `HF_TOKEN` |
| **AWS Bedrock** | Claude, Nova, Llama, DeepSeek via Converse API nativa | IAM role ou `aws configure` ([guia](../guides/aws-bedrock.md)) |
| **Azure Foundry** | Modelos hospedados no Azure AI Foundry | Defina `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| **Google AI Studio** | Modelos Gemini via API direta | Defina `GOOGLE_API_KEY` / `GEMINI_API_KEY` |
| **xAI** | Modelos Grok via API direta | Defina `XAI_API_KEY` |
| **xAI Grok OAuth** | Assinatura SuperGrok / Premium+, sem API key | `hermes model` → xAI Grok OAuth |
| **NovitaAI** | Gateway de API multi-modelo | Defina `NOVITA_API_KEY` |
| **StepFun** | Modelos Step Plan | Defina `STEPFUN_API_KEY` |
| **Xiaomi MiMo** | Modelos hospedados pela Xiaomi | Defina `XIAOMI_API_KEY` |
| **Tencent TokenHub** | Modelos hospedados pela Tencent | Defina `TOKENHUB_API_KEY` |
| **Ollama Cloud** | Modelos Ollama gerenciados na nuvem | Defina `OLLAMA_API_KEY` |
| **LM Studio** | App desktop local expondo API compatível com OpenAI | Defina `LM_API_KEY` (e `LM_BASE_URL` se não for o default) |
| **Qwen OAuth** | OAuth no browser do Qwen Portal — sem API key | `hermes model` → Qwen OAuth |
| **Kilo Code** | Modelos hospedados no KiloCode | Defina `KILOCODE_API_KEY` |
| **OpenCode Zen** | Acesso pay-as-you-go a modelos curados | Defina `OPENCODE_ZEN_API_KEY` |
| **OpenCode Go** | Assinatura de US$ 10/mês para modelos open | Defina `OPENCODE_GO_API_KEY` |
| **DeepSeek** | Acesso direto à API DeepSeek | Defina `DEEPSEEK_API_KEY` |
| **NVIDIA NIM** | Modelos Nemotron via build.nvidia.com ou NIM local | Defina `NVIDIA_API_KEY` (opcional: `NVIDIA_BASE_URL`) |
| **GitHub Copilot** | Assinatura GitHub Copilot (GPT-5.x, Claude, Gemini, etc.) | OAuth via `hermes model`, ou `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| **GitHub Copilot ACP** | Backend de agente Copilot ACP (spawna o CLI local `copilot`) | `hermes model` (exige CLI `copilot` + `copilot login`) |
| **Custom Endpoint** | VLLM, SGLang, Ollama ou qualquer API compatível com OpenAI | Defina base URL + API key |

Para a maioria dos iniciantes: escolha um provider e aceite os defaults, a menos que saiba por que está mudando. O catálogo completo de providers com env vars e passos de setup fica na página [Providers](../integrations/providers.md).

:::caution Contexto mínimo: 64K tokens
O Hermes Agent exige um modelo com pelo menos **64.000 tokens** de contexto. Modelos com janela menor não mantêm memória de trabalho suficiente para fluxos multi-step com tool-calling e serão rejeitados na inicialização. A maioria dos modelos hospedados (Claude, GPT, Gemini, Qwen, DeepSeek) atende isso com folga. Se estiver rodando um modelo local, configure o tamanho de contexto para pelo menos 64K (ex.: `--ctx-size 65536` no llama.cpp ou `-c 65536` no Ollama).
:::

:::tip
Você pode trocar de provider a qualquer momento com `hermes model` — sem lock-in. Para a lista completa de providers suportados e detalhes de setup, veja [AI Providers](../integrations/providers.md).
:::

### Como as settings são guardadas

O Hermes separa segredos da config normal:

- **Segredos e tokens** → `~/.hermes/.env`
- **Settings que não são segredo** → `~/.hermes/config.yaml`

O jeito mais fácil de setar valores corretamente é pelo CLI:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

O valor certo vai para o arquivo certo automaticamente.

## 3. Rodar o primeiro chat

```bash
hermes            # classic CLI
hermes --tui      # modern TUI (recommended)
```

Você verá um banner de boas-vindas com o modelo, tools disponíveis e skills. Use um prompt específico e fácil de verificar:

:::tip Escolha a interface
O Hermes vem com duas interfaces de terminal: o CLI clássico `prompt_toolkit` e um [TUI](../user-guide/tui.md) mais novo com overlays modais, seleção por mouse e input não bloqueante. Os dois compartilham as mesmas sessões, slash commands e config — experimente cada um com `hermes` vs `hermes --tui`.
:::

```
Summarize this repo in 5 bullets and tell me what the main entrypoint is.
```

```
Check my current directory and tell me what looks like the main project file.
```

```
Help me set up a clean GitHub PR workflow for this codebase.
```

**Como parece o sucesso:**

- O banner mostra o modelo/provider escolhido
- O Hermes responde sem erro
- Ele consegue usar uma tool se precisar (terminal, leitura de arquivo, busca na web)
- A conversa continua normalmente por mais de um turno

Se isso funcionar, você passou da parte mais difícil.

## 4. Verificar que as sessões funcionam

Antes de seguir, confirme que o resume funciona:

```bash
hermes --continue    # Resume the most recent session
hermes -c            # Short form
```

Isso deve trazer de volta a sessão que você acabou de ter. Se não trouxer, confira se está no mesmo profile e se a sessão de fato foi salva. Isso importa depois, quando você estiver malabarizando vários setups ou máquinas.

## 5. Experimentar recursos principais

### Usar o terminal

```
❯ What's my disk usage? Show the top 5 largest directories.
```

O agente roda comandos de terminal por você e mostra os resultados.

### Slash commands

Digite `/` para ver um dropdown de autocomplete com todos os comandos:

| Comando | O que faz |
|---------|-------------|
| `/help` | Mostra todos os comandos disponíveis |
| `/tools` | Lista as tools disponíveis |
| `/model` | Troca de modelo de forma interativa |
| `/personality pirate` | Experimenta uma personalidade divertida |
| `/save` | Salva a conversa |

### Input multi-linha

Pressione `Alt+Enter`, `Ctrl+J` ou `Shift+Enter` para adicionar uma nova linha. `Shift+Enter` exige um terminal que envie isso como sequência distinta (Kitty / foot / WezTerm / Ghostty por padrão; iTerm2 / Alacritty / terminal do VS Code quando o protocolo de teclado Kitty está habilitado). `Alt+Enter` e `Ctrl+J` funcionam em todo terminal.

### Interromper o agente

Se o agente estiver demorando demais, digite uma nova mensagem e pressione Enter — ele interrompe a tarefa atual e passa para as novas instruções. `Ctrl+C` também funciona.

## 6. Acrescentar a próxima camada

Só depois que o chat base funcionar. Escolha o que precisa:

### Bot ou assistente compartilhado

```bash
hermes gateway setup    # Interactive platform configuration
```

Conecte [Telegram](/user-guide/messaging/telegram), [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp), [Signal](/user-guide/messaging/signal), [Email](/user-guide/messaging/email), [Home Assistant](/user-guide/messaging/homeassistant) ou [Microsoft Teams](/user-guide/messaging/teams).

### Automação e tools

- `hermes tools` — ajusta o acesso a tools por plataforma
- `hermes skills` — navega e instala workflows reutilizáveis
- Cron — só depois que o bot ou o setup CLI estiver estável

### Terminal em sandbox

Por segurança, rode o agente num container Docker ou num servidor remoto:

```bash
hermes config set terminal.backend docker    # Docker isolation
hermes config set terminal.backend ssh       # Remote server
```

### Modo de voz

```bash
# From the Hermes install directory (the curl installer placed it at
# ~/.hermes/hermes-agent on Linux/macOS or %LOCALAPPDATA%\hermes\hermes-agent on Windows):
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# Includes faster-whisper for free local speech-to-text
```

Depois, no CLI: `/voice on`. Pressione `Ctrl+B` para gravar. Veja [Modo de voz](../user-guide/features/voice-mode.md).

### Skills

Skills são documentos de instrução sob demanda que ensinam o Hermes a fazer uma tarefa específica — deploy no Kubernetes, abrir um PR no GitHub, fine-tune de um modelo, buscar GIFs. Cada uma é um arquivo `SKILL.md` com nome, descrição e um procedimento passo a passo. O agente lê as descrições curtas de graça e só carrega o conteúdo completo de uma skill quando a tarefa de fato pede, então adicionar skills não infla cada request.

O Hermes já vem com um catálogo de skills bundled instaladas em `~/.hermes/skills/`. Você pode acrescentar mais pelo Skills Hub ou escrever as suas.

**Navegar e instalar pelo hub:**

```bash
hermes skills browse                      # list everything available
hermes skills search kubernetes           # find skills by keyword
hermes skills install openai/skills/k8s   # install one (runs a security scan first)
```

O argumento do install é um slug `source/path` do hub — `openai/skills/k8s` significa a skill `k8s` do catálogo da OpenAI. O `hermes skills browse` mostra os slugs exatos a usar.

**Usar uma skill** — toda skill instalada vira um slash command automaticamente:

```bash
/k8s deploy the staging manifest          # run the skill with a request
/k8s                                       # load it and let Hermes ask what you need
```

Isso funciona no CLI e em qualquer plataforma de messaging conectada. Você não precisa instalar tudo de antemão — o agente escolhe sozinho a skill bundled certa durante a conversa normal quando a tarefa bate com uma delas.

Veja [Sistema de skills](../user-guide/features/skills.md) para escrever as suas, diretórios externos de skills e a lista completa de sources do hub.

### Servidores MCP

```yaml
# Add to ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### Integração com editor (ACP)

O suporte a ACP já vem com os extras padrão `[all]`, então o instalador curl já inclui. Basta rodar:

```bash
hermes acp
```

(Se você instalou sem `[all]`, rode antes `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"`.)

Veja [ACP Editor Integration](../user-guide/features/acp.md).

---

## Modos de falha comuns

Estes são os problemas que mais fazem perder tempo:

| Sintoma | Causa provável | Correção |
|---|---|---|
| O Hermes abre, mas as respostas vêm vazias ou quebradas | Auth do provider ou seleção de modelo está errada | Rode `hermes model` de novo e confirme provider, modelo e auth |
| Endpoint custom "funciona", mas devolve lixo | Base URL errada, nome de modelo errado ou não é de fato compatível com OpenAI | Verifique o endpoint num cliente separado primeiro |
| O gateway sobe, mas ninguém consegue mandar mensagem | Token do bot, allowlist ou setup da plataforma incompleto | Rode de novo `hermes gateway setup` e confira `hermes gateway status` |
| `hermes --continue` não acha a sessão antiga | Trocou de profile ou a sessão nunca foi salva | Confira `hermes sessions list` e se está no profile certo |
| Modelo indisponível ou fallback estranho | Routing do provider ou settings de fallback agressivos demais | Mantenha o routing desligado até o provider base estar estável |
| `hermes doctor` aponta problemas de config | Valores de config faltando ou desatualizados | Corrija a config, reteste um chat simples antes de acrescentar features |

## Kit de recuperação

Quando algo parecer estranho, use esta ordem:

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

Essa sequência te tira do "tá quebrado, sei lá" e volta a um estado conhecido rápido.

---

## Referência rápida

| Comando | Descrição |
|---------|-------------|
| `hermes` | Começar a conversar |
| `hermes model` | Escolher o provider de LLM e o modelo |
| `hermes tools` | Configurar quais tools estão habilitadas por plataforma |
| `hermes setup` | Wizard completo de setup (configura tudo de uma vez) |
| `hermes doctor` | Diagnosticar problemas |
| `hermes update` | Atualizar para a versão mais recente |
| `hermes gateway` | Iniciar o messaging gateway |
| `hermes --continue` | Retomar a última sessão |

## Próximos passos

- **[Guia do CLI](../user-guide/cli.md)** — Domine a interface de terminal
- **[Configuração](../user-guide/configuration.md)** — Personalize o setup
- **[Messaging Gateway](../user-guide/messaging/index.md)** — Conecte Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, Teams e mais
- **[Tools & Toolsets](../user-guide/features/tools.md)** — Explore as capacidades disponíveis
- **[AI Providers](../integrations/providers.md)** — Lista completa de providers e detalhes de setup
- **[Sistema de skills](../user-guide/features/skills.md)** — Workflows e conhecimento reutilizáveis
- **[Dicas e boas práticas](../guides/tips.md)** — Dicas para usuários avançados
