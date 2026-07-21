---
sidebar_position: 1
title: "Início Rápido"
description: "Sua primeira conversa com o Hermes Agent — da instalação ao chat em menos de 5 minutos"
---

# Início Rápido

Este guia leva você do zero a uma configuração funcional do Hermes que sobrevive ao uso real. Instale, escolha um provider, verifique um chat funcional e saiba exatamente o que fazer quando algo quebrar.

## Prefere assistir em vídeo?

O **Onchain AI Garage** preparou um passo a passo Masterclass sobre instalação, configuração e comandos básicos — um bom complemento a esta página se você prefere seguir em vídeo. Para mais, veja a playlist completa [Hermes Agent Tutorials & Use Cases](https://www.youtube.com/playlist?list=PLmpUb_PWAkDxewld5ZYyKifuHxgIbiq2d).

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

## Para quem é este guia

- É novo e quer o caminho mais curto para uma configuração funcional
- Está trocando de provider e não quer perder tempo com erros de configuração
- Está configurando o Hermes para uma equipe, bot ou fluxo de trabalho sempre ativo
- Está cansado de "instalou, mas ainda não faz nada"

## O caminho mais rápido

Escolha a linha que corresponde ao seu objetivo:

| Objetivo                                                         | Faça isto primeiro          | Depois faça isto                                                      |
|------------------------------------------------------------------|-----------------------------|-----------------------------------------------------------------------|
| Só quero o Hermes funcionando na minha máquina                   | `hermes setup`              | Execute um chat real e verifique se ele responde                      |
| Já conheço meu provider                                           | `hermes model`              | Salve a configuração, depois comece a conversar                       |
| Quero um bot ou configuração sempre ativa                        | `hermes gateway setup` após o CLI funcionar | Conecte Telegram, Discord, Slack ou outra plataforma                  |
| Quero um modelo local ou auto-hospedado                          | `hermes model` → endpoint customizado | Verifique o endpoint, nome do modelo e tamanho do contexto            |
| Quero fallback de múltiplos providers                            | `hermes model` primeiro     | Adicione roteamento e fallback somente após o chat base funcionar     |

**Regra prática:** se o Hermes não consegue completar um chat normal, não adicione mais recursos ainda. Primeiro tenha uma conversa limpa funcionando, depois adicione gateway, cron, skills, voz ou roteamento.

---

## 1. Instale o Hermes Agent
### Com o instalador Hermes Desktop no macOS ou Windows (recomendado)
Para instalar facilmente os aplicativos de linha de comando e desktop, [baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) do nosso site e execute-o.

### Sem o Hermes Desktop:
Para uma instalação apenas com linha de comando sem o Hermes Desktop, execute:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (nativo)

Execute no powershell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

:::tip Android / Termux
Se você estiver instalando em um telefone, veja o [guia Termux](./termux.md) dedicado para o caminho manual testado, extras suportados e limitações atuais específicas do Android.
:::

Após a conclusão, recarregue seu shell:

```bash
source ~/.bashrc   # ou source ~/.zshrc
```

Para opções detalhadas de instalação, pré-requisitos e solução de problemas, veja o [guia de Instalação](./installation.md).

## 2. Escolha um Provider

A etapa de configuração mais importante. Use `hermes model` para percorrer a escolha interativamente:

```bash
hermes model
```

:::tip Caminho mais fácil: Nous Portal
Uma assinatura cobre mais de 300 modelos mais o [Tool Gateway](../user-guide/features/tool-gateway.md) (pesquisa web, geração de imagem, TTS, navegador na nuvem). Em uma instalação nova:

```bash
hermes setup --portal
```

Isso faz login, define Nous como seu provider e ativa o Tool Gateway em um único comando.
:::

:::info Modos de configuração
Em uma instalação nova, `hermes setup` oferece três modos:

- **Quick Setup (Nous Portal)** — login OAuth gratuito, sem chaves de API; configura um modelo mais as ferramentas do Tool Gateway. O caminho rápido recomendado.
- **Full Setup** — percorra cada provider, ferramenta e opção você mesmo (traga suas próprias chaves).
- **Blank Slate** — tudo começa **desligado**, exceto o mínimo necessário para executar um agente: **provider & model, o toolset File Operations e o toolset Terminal**. Sem web, navegador, execução de código, visão, memória, delegação, cron, skills, plugins ou servidores MCP — e compressão, checkpoints, roteamento inteligente e captura de memória estão todos desabilitados. Após a aplicação da base mínima, você escolhe um de dois caminhos: **começar com tudo desabilitado** (finalizar agora com o agente mínimo), ou **percorrer todas as configurações** (optar por ferramentas, skills, plugins, MCP e mensagens). Escolha esta opção quando quiser um agente mínimo e totalmente controlado, e pretender habilitar apenas exatamente o que você precisa.

O Blank Slate escreve uma lista explícita `platform_toolsets.cli` mais `agent.disabled_toolsets`, para que nada que você não escolheu seja carregado — nem mesmo após `hermes update`. Reative qualquer coisa depois com `hermes tools`, semeie skills com `hermes skills opt-in --sync`, ou ajuste configurações com `hermes setup agent`.
:::

Bons padrões:

| Provider              | O que é                                    | Como configurar                                                    |
|-----------------------|--------------------------------------------|--------------------------------------------------------------------|
| **Nous Portal**       | Baseado em assinatura, zero-config         | Login OAuth via `hermes model`                                     |
| **OpenAI Codex**      | OAuth do ChatGPT, usa modelos Codex        | Auth via código de dispositivo via `hermes model`                   |
| **Anthropic**         | Modelos Claude diretamente — Plano Max + créditos de uso extras (OAuth), ou chave de API para pagamento por token | `hermes model` → login OAuth (requer Max + créditos extras), ou uma chave de API Anthropic |
| **OpenRouter**        | Roteamento multi-provider entre muitos modelos | Insira sua chave de API                                             |
| **Fireworks AI**      | API de modelo direta compatível com OpenAI | Defina `FIREWORKS_API_KEY`                                         |
| **Z.AI**              | Modelos GLM / Zhipu                       | Defina `GLM_API_KEY` / `ZAI_API_KEY` (também aceita `Z_AI_API_KEY`) |
| **Kimi / Moonshot**   | Modelos de coding e chat hospedados pela Moonshot | Defina `KIMI_API_KEY` (ou a específica para Kimi-Coding `KIMI_CODING_API_KEY`) |
| **Kimi / Moonshot China** | Endpoint Moonshot região China            | Defina `KIMI_CN_API_KEY`                                           |
| **Arcee AI**          | Modelos Trinity                             | Defina `ARCEEAI_API_KEY`                                           |
| **GMI Cloud**         | API multi-modelo direta                    | Defina `GMI_API_KEY`                                               |
| **MiniMax (OAuth)**   | Modelo frontier MiniMax via OAuth do navegador — sem chave de API necessária (o nome do modelo em `hermes_cli/models.py` pode mudar entre versões) | `hermes model` → MiniMax (OAuth)                                   |
| **MiniMax**           | Endpoint MiniMax internacional             | Defina `MINIMAX_API_KEY`                                           |
| **MiniMax China**     | Endpoint MiniMax região China              | Defina `MINIMAX_CN_API_KEY`                                        |
| **Alibaba Cloud**     | Modelos Qwen via DashScope                 | Defina `DASHSCOPE_API_KEY` (Qwen Coding Plan também aceita `ALIBABA_CODING_PLAN_API_KEY`) |
| **Hugging Face**      | Mais de 20 modelos abertos via roteador unificado (Qwen, DeepSeek, Kimi, etc.) | Defina `HF_TOKEN`                                                  |
| **AWS Bedrock**       | Claude, Nova, Llama, DeepSeek via Converse API nativa | Role IAM ou `aws configure` ([guia](../guides/aws-bedrock.md))      |
| **Azure Foundry**     | Modelos hospedados pelo Azure AI Foundry   | Defina `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL`          |
| **Google AI Studio**  | Modelos Gemini via API direta              | Defina `GOOGLE_API_KEY` / `GEMINI_API_KEY`                          |
| **xAI**               | Modelos Grok via API direta               | Defina `XAI_API_KEY`                                               |
| **xAI Grok OAuth**    | Assinatura SuperGrok / Premium+, sem chave de API necessária | `hermes model` → xAI Grok OAuth                                    |
| **NovitaAI**          | Gateway de API multi-modelo                | Defina `NOVITA_API_KEY`                                            |
| **StepFun**           | Modelos Step Plan                          | Defina `STEPFUN_API_KEY`                                           |
| **Xiaomi MiMo**       | Modelos hospedados pela Xiaomi             | Defina `XIAOMI_API_KEY`                                            |
| **Tencent TokenHub**  | Modelos hospedados pela Tencent            | Defina `TOKENHUB_API_KEY`                                          |
| **Ollama Cloud**      | Modelos Ollama gerenciados                 | Defina `OLLAMA_API_KEY`                                            |
| **LM Studio**         | Aplicativo desktop local expondo API compatível com OpenAI | Defina `LM_API_KEY` (e `LM_BASE_URL` se não for o padrão)          |
| **Qwen OAuth**        | OAuth do Qwen Portal no navegador — sem chave de API necessária | `hermes model` → Qwen OAuth                                        |
| **Kilo Code**         | Modelos hospedados pela KiloCode           | Defina `KILOCODE_API_KEY`                                          |
| **OpenCode Zen**      | Acesso pay-as-you-go a modelos curados     | Defina `OPENCODE_ZEN_API_KEY`                                      |
| **OpenCode Go**       | Assinatura de $10/mês para modelos abertos | Defina `OPENCODE_GO_API_KEY`                                       |
| **DeepSeek**          | Acesso direto à API DeepSeek              | Defina `DEEPSEEK_API_KEY`                                          |
| **NVIDIA NIM**        | Modelos Nemotron via build.nvidia.com ou NIM local | Defina `NVIDIA_API_KEY` (opcional: `NVIDIA_BASE_URL`)               |
| **GitHub Copilot**    | Assinatura GitHub Copilot (GPT-5.x, Claude, Gemini, etc.) | OAuth via `hermes model`, ou `COPILOT_GITHUB_TOKEN` / `GH_TOKEN`    |
| **GitHub Copilot ACP**| Backend do agente Copilot ACP (inicia CLI `copilot` local) | `hermes model` (requer CLI `copilot` + `copilot login`)            |
| **Custom Endpoint**   | VLLM, SGLang, Ollama ou qualquer API compatível com OpenAI | Defina base URL + chave de API                                     |

Para a maioria dos usuários iniciantes: escolha um provider, aceite os padrões a menos que você saiba por que está alterando-os. O catálogo completo de providers com variáveis de ambiente e etapas de configuração está na página [Providers](../integrations/providers.md).

:::caution Contexto mínimo: 64K tokens
O Hermes Agent requer um modelo com pelo menos **64.000 tokens** de contexto. Modelos com janelas menores não conseguem manter memória de trabalho suficiente para fluxos de trabalho com múltiplas chamadas de ferramentas e serão rejeitados na inicialização. A maioria dos modelos hospedados (Claude, GPT, Gemini, Qwen, DeepSeek) atende isso facilmente. Se você estiver executando um modelo local, defina seu tamanho de contexto para pelo menos 64K (ex.: `--ctx-size 65536` para llama.cpp ou `-c 65536` para Ollama).
:::

:::tip
Você pode trocar de provider a qualquer momento com `hermes model` — sem lock-in. Para uma lista completa de todos os providers suportados e detalhes de configuração, veja [AI Providers](../integrations/providers.md).
:::

### Como as configurações são armazenadas

O Hermes separa segredos da configuração normal:

- **Segredos e tokens** → `~/.hermes/.env`
- **Configurações não secretas** → `~/.hermes/config.yaml`

A maneira mais fácil de definir valores corretamente é através do CLI:

```bash
hermes config set model anthropic/claude-opus-4.6
hermes config set terminal.backend docker
hermes config set OPENROUTER_API_KEY sk-or-...
```

O valor certo vai para o arquivo certo automaticamente.

## 3. Execute Seu Primeiro Chat

```bash
hermes            # CLI clássico
hermes --tui      # TUI moderno (recomendado)
```

Você verá um banner de boas-vindas com seu modelo, ferramentas disponíveis e skills. Use um prompt específico e fácil de verificar:

:::tip Escolha sua interface
O Hermes vem com duas interfaces de terminal: o CLI clássico `prompt_toolkit` e um [TUI](../user-guide/tui.md) mais novo com sobreposições modais, seleção com mouse e entrada não bloqueante. Ambos compartilham as mesmas sessões, comandos de barra e configuração — experimente cada um com `hermes` vs `hermes --tui`.
:::

```
Resuma este repositório em 5 pontos e me diga qual é o ponto de entrada principal.
```

```
Verifique meu diretório atual e me diga qual parece ser o arquivo principal do projeto.
```

```
Me ajude a configurar um fluxo de trabalho limpo de PR no GitHub para este código.
```

**O que significa sucesso:**

- O banner mostra seu modelo/provider escolhido
- O Hermes responde sem erros
- Ele pode usar uma ferramenta se necessário (terminal, leitura de arquivo, pesquisa web)
- A conversa continua normalmente por mais de um turno

Se isso funcionar, você passou da parte mais difícil.

## 4. Verifique se as Sessões Funcionam

Antes de continuar, certifique-se de que o resume funciona:

```bash
hermes --continue    # Retomar a sessão mais recente
hermes -c            # Forma abreviada
```

Isso deve trazê-lo de volta à sessão que você acabou de ter. Se não funcionar, verifique se você está no mesmo profile e se a sessão realmente foi salva. Isso será importante mais tarde quando você estiver lidando com múltiplas configurações ou máquinas.

## 5. Teste os Principais Recursos

### Use o terminal

```
❯ Qual é o meu uso de disco? Mostre os 5 maiores diretórios.
```

O agente executa comandos de terminal em seu nome e mostra os resultados.

### Comandos de barra (/)

Digite `/` para ver um menu suspenso de autocomplete com todos os comandos:

| Comando         | O que faz                                          |
|-----------------|----------------------------------------------------|
| `/help`         | Mostra todos os comandos disponíveis               |
| `/tools`        | Lista ferramentas disponíveis                      |
| `/model`        | Troca de modelo interativamente                    |
| `/personality pirate` | Experimente uma personalidade divertida       |
| `/save`         | Salva a conversa                                   |

### Entrada multi-linha

Pressione `Alt+Enter`, `Ctrl+J` ou `Shift+Enter` para adicionar uma nova linha. `Shift+Enter` requer um terminal que o envie como uma sequência distinta (Kitty / foot / WezTerm / Ghostty por padrão; iTerm2 / Alacritty / terminal VS Code com o protocolo Kitty habilitado). `Alt+Enter` e `Ctrl+J` funcionam em todos os terminais.

### Interrompa o agente

Se o agente estiver demorando muito, digite uma nova mensagem e pressione Enter — ele interrompe a tarefa atual e muda para suas novas instruções. `Ctrl+C` também funciona.

## 6. Adicione a Próxima Camada

Somente depois que o chat base funcionar. Escolha o que você precisa:

### Bot ou assistente compartilhado

```bash
hermes gateway setup    # Configuração interativa de plataforma
```

Conecte [Telegram](/user-guide/messaging/telegram), [Discord](/user-guide/messaging/discord), [Slack](/user-guide/messaging/slack), [WhatsApp](/user-guide/messaging/whatsapp), [Signal](/user-guide/messaging/signal), [Email](/user-guide/messaging/email), ou [Home Assistant](/user-guide/messaging/homeassistant), ou [Microsoft Teams](/user-guide/messaging/teams).

### Automação e ferramentas

- `hermes tools` — ajuste o acesso a ferramentas por plataforma
- `hermes skills` — navegue e instale fluxos de trabalho reutilizáveis
- Cron — somente após seu bot ou configuração CLI estar estável

### Terminal em sandbox

Para segurança, execute o agente em um container Docker ou em um servidor remoto:

```bash
hermes config set terminal.backend docker    # Isolamento Docker
hermes config set terminal.backend ssh       # Servidor remoto
```

### Modo de voz

```bash
# Do diretório de instalação do Hermes (o instalador curl o coloca em
# ~/.hermes/hermes-agent no Linux/macOS ou %LOCALAPPDATA%\hermes\hermes-agent no Windows):
cd ~/.hermes/hermes-agent
uv pip install -e ".[voice]"
# Inclui faster-whisper para conversão de fala para texto local gratuita
```

Depois no CLI: `/voice on`. Pressione `Ctrl+B` para gravar. Veja [Modo de Voz](../user-guide/features/voice-mode.md).

### Skills

Skills são documentos de instrução sob demanda que ensinam o Hermes a fazer uma tarefa específica — implantar no Kubernetes, abrir um PR no GitHub, ajustar um modelo, pesquisar GIFs. Cada uma é um arquivo `SKILL.md` com um nome, uma descrição e um procedimento passo a passo. O agente lê as descrições curtas gratuitamente e só carrega o conteúdo completo de uma skill quando uma tarefa realmente a solicita, então adicionar skills não incha toda requisição.

O Hermes vem com um catálogo de skills inclusas já instaladas em `~/.hermes/skills/`. Você pode adicionar mais do Skills Hub, ou escrever as suas próprias.

**Navegue e instale do hub:**

```bash
hermes skills browse                      # listar tudo disponível
hermes skills search kubernetes           # encontrar skills por palavra-chave
hermes skills install openai/skills/k8s   # instalar uma (executa uma verificação de segurança primeiro)
```

O argumento de instalação é um slug `fonte/caminho` do hub — `openai/skills/k8s` significa a skill `k8s` do catálogo da OpenAI. `hermes skills browse` mostra os slugs exatos a serem usados.

**Use uma skill** — toda skill instalada se torna automaticamente um comando de barra:

```bash
/k8s implantar o manifesto de staging          # executar a skill com uma solicitação
/k8s                                           # carregá-la e deixar o Hermes perguntar o que você precisa
```

Isso funciona no CLI e em qualquer plataforma de mensagens conectada. Você não precisa instalar tudo de antemão — o agente escolhe a skill inclusa certa por conta própria durante a conversa normal quando uma tarefa corresponde a ela.

Veja [Sistema de Skills](../user-guide/features/skills.md) para escrever as suas próprias, diretórios externos de skills e a lista completa de fontes do hub.

### Servidores MCP

```yaml
# Adicione em ~/.hermes/config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_xxx"
```

### Integração com editor (ACP)

O suporte ACP vem com os extras padrão `[all]`, então o instalador curl já o inclui. Apenas execute:

```bash
hermes acp
```

(Se você instalou sem `[all]`, execute `cd ~/.hermes/hermes-agent && uv pip install -e ".[acp]"` primeiro.)

Veja [Integração com Editor ACP](../user-guide/features/acp.md).

---

## Modos Comuns de Falha

Estes são os problemas que mais desperdiçam tempo:

| Sintoma                                                       | Causa provável                                    | Correção                                                              |
|---------------------------------------------------------------|---------------------------------------------------|-----------------------------------------------------------------------|
| Hermes abre mas dá respostas vazias ou quebradas              | Auth do provider ou seleção de modelo errada      | Execute `hermes model` novamente e confirme provider, modelo e auth  |
| Endpoint customizado "funciona" mas retorna lixo              | URL base, nome do modelo errado, ou não é realmente compatível com OpenAI | Verifique o endpoint em um cliente separado primeiro                  |
| Gateway inicia mas ninguém consegue enviar mensagem           | Token do bot, lista de permissões ou configuração da plataforma incompleta | Execute `hermes gateway setup` novamente e verifique `hermes gateway status` |
| `hermes --continue` não encontra sessão antiga               | Profiles trocados ou sessão nunca foi salva       | Verifique `hermes sessions list` e confirme que você está no profile correto |
| Modelo indisponível ou comportamento de fallback estranho     | Roteamento de provider ou configurações de fallback muito agressivas | Mantenha o roteamento desligado até que o provider base esteja estável |
| `hermes doctor` sinaliza problemas de configuração            | Valores de configuração ausentes ou desatualizados | Corrija a configuração, teste um chat simples antes de adicionar recursos |

## Kit de Recuperação

Quando algo parecer errado, use esta ordem:

1. `hermes doctor`
2. `hermes model`
3. `hermes setup`
4. `hermes sessions list`
5. `hermes --continue`
6. `hermes gateway status`

Essa sequência leva você de "vibrações quebradas" de volta a um estado conhecido rapidamente.

---

## Referência Rápida

| Comando              | Descrição                                                     |
|----------------------|---------------------------------------------------------------|
| `hermes`             | Comece a conversar                                            |
| `hermes model`       | Escolha seu provider e modelo LLM                             |
| `hermes tools`       | Configure quais ferramentas estão ativadas por plataforma     |
| `hermes setup`       | Assistente de configuração completo (configura tudo de uma vez) |
| `hermes doctor`      | Diagnostique problemas                                        |
| `hermes update`      | Atualize para a versão mais recente                           |
| `hermes gateway`     | Inicie o gateway de mensagens                                 |
| `hermes --continue`  | Retome a última sessão                                        |

## Próximos Passos

- **[Guia do CLI](../user-guide/cli.md)** — Domine a interface de terminal
- **[Configuração](../user-guide/configuration.md)** — Personalize sua instalação
- **[Gateway de Mensagens](../user-guide/messaging/index.md)** — Conecte Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant, Teams e mais
- **[Ferramentas e Toolsets](../user-guide/features/tools.md)** — Explore as capacidades disponíveis
- **[AI Providers](../integrations/providers.md)** — Lista completa de providers e detalhes de configuração
- **[Sistema de Skills](../user-guide/features/skills.md)** — Fluxos de trabalho e conhecimento reutilizáveis
- **[Dicas e Melhores Práticas](../guides/tips.md)** — Dicas de usuário avançado
