---
sidebar_position: 1
title: "Referência de Comandos da CLI"
description: "Referência oficial dos comandos de terminal do Hermes e famílias de comandos"
---

# Referência de Comandos da CLI

Esta página cobre os **comandos de terminal** que você executa a partir do seu shell.

Para slash commands dentro do chat, veja a [Referência de Slash Commands](./slash-commands.md).

## Ponto de entrada global {#global-entrypoint}

```bash
hermes [global-options] <command> [subcommand/options]
```

### Opções globais {#global-options}

| Opção | Descrição |
|--------|-------------|
| `--version`, `-V` | Mostra a versão e encerra. |
| `--profile <name>`, `-p <name>` | Seleciona qual perfil do Hermes usar nesta invocação. Sobrescreve o padrão fixo definido por `hermes profile use`. |
| `--resume <session>`, `-r <session>` | Retoma uma sessão anterior por ID ou título. |
| `--continue [name]`, `-c [name]` | Retoma a sessão mais recente, ou a sessão mais recente que corresponda a um título. |
| `--worktree`, `-w` | Inicia em uma worktree do git isolada para fluxos de agentes paralelos. |
| `--yolo` | Ignora os prompts de aprovação de comandos perigosos. |
| `--pass-session-id` | Inclui o ID da sessão no system prompt do agente. |
| `--ignore-user-config` | Ignora `~/.hermes/config.yaml` e volta para os padrões embutidos. As credenciais em `.env` continuam sendo carregadas. |
| `--ignore-rules` | Ignora a injeção automática de `AGENTS.md`, `SOUL.md`, `.cursorrules`, memória e skills pré-carregadas. |
| `--tui` | Inicia a [TUI](../user-guide/tui.md) em vez da CLI clássica. Equivalente a `HERMES_TUI=1`. Sempre prevalece sobre `display.interface`. |
| `--cli` | Força o REPL clássico baseado em prompt_toolkit. Use para sobrescrever `display.interface: tui` em uma única invocação. |
| `--dev` | Com `--tui`: executa as fontes TypeScript diretamente via `tsx` em vez do bundle pré-compilado (para contribuidores da TUI). |

## Comandos de nível superior {#top-level-commands}

| Comando | Finalidade |
|---------|-------|
| `hermes chat` | Chat interativo ou de disparo único com o agente. |
| `hermes model` | Escolhe interativamente o provedor e o modelo padrão. |
| `hermes moa` | Configura presets nomeados de Mixture of Agents selecionáveis no seletor de modelos. |
| `hermes fallback` | Gerencia provedores de fallback tentados quando o modelo principal falha. |
| `hermes gateway` | Executa ou gerencia o serviço de gateway de mensagens. |
| `hermes proxy` | Proxy local compatível com OpenAI que aplica credenciais de provedor via OAuth. Veja [Subscription Proxy](../user-guide/features/subscription-proxy.md). |
| `hermes lsp` | Gerencia a integração com o Language Server Protocol (diagnósticos semânticos para write_file/patch). |
| `hermes setup` | Assistente de configuração interativo para toda ou parte da configuração. |
| `hermes whatsapp` | Configura e pareia a bridge do WhatsApp. |
| `hermes whatsapp-cloud` | Configura o adaptador oficial da Meta WhatsApp Business Cloud API (é necessário conta Business + webhook público). Diferente de `hermes whatsapp` (bridge de conta pessoal via Baileys). |
| `hermes slack` | Utilitários do Slack (atualmente: gera o manifesto do app com cada comando como slash nativo). |
| `hermes auth` | Gerencia credenciais — adiciona, lista, remove, redefine, status, logout. Lida com os fluxos OAuth de Codex/Nous/Anthropic. |
| `hermes login` / `logout` | **Descontinuado** — use `hermes auth`. |
| `hermes send` | Envia uma mensagem de disparo único a uma plataforma de mensagens configurada (Telegram, Discord, Slack, Signal, SMS, …). Útil em scripts de shell, cron jobs, hooks de CI e daemons de monitoramento — sem loop de agente, sem LLM. |
| `hermes secrets` | Gerencia fontes externas de segredos (atualmente Bitwarden Secrets Manager) para buscar chaves de API na inicialização do processo em vez de `~/.hermes/.env`. |
| `hermes migrate` | Diagnostica e (opcionalmente) reescreve o `config.yaml` para substituir referências a modelos descontinuados ou configurações obsoletas (ex.: `migrate xai`). |
| `hermes status` | Mostra o status do agente, autenticação e plataformas. |
| `hermes cron` | Inspeciona e executa um tick do agendador de cron. |
| `hermes kanban` | Quadro de colaboração multi-perfil (tarefas, links, dispatcher). |
| `hermes project` | Gerencia workspaces nomeados de múltiplas pastas (projetos). Ancora o agrupamento de sessões no desktop e, quando vinculado a um quadro kanban, dá às tarefas uma convenção determinística de worktree + branch. O estado é por perfil. |
| `hermes webhook` | Gerencia assinaturas dinâmicas de webhook para ativação orientada a eventos. |
| `hermes hooks` | Inspeciona, aprova ou remove hooks de script de shell declarados no `config.yaml`. |
| `hermes doctor` | Diagnostica problemas de configuração e dependências. |
| `hermes security audit` | Auditoria sob demanda da cadeia de suprimentos (OSV.dev) para o venv, requisitos de plugins e servidores MCP fixados. |
| `hermes dump` | Resumo de configuração pronto para copiar e colar, para suporte/depuração. |
| `hermes prompt-size` | Mostra o detalhamento em bytes do system prompt + schemas de ferramentas (índice de skills, memória, perfil). Executa offline. |
| `hermes debug` | Ferramentas de depuração — envia logs e informações do sistema para suporte. |
| `hermes backup` | Faz backup do diretório home do Hermes em um arquivo zip. |
| `hermes checkpoints` | Inspeciona / limpa / apaga `~/.hermes/checkpoints/` (o armazenamento sombra usado por `/rollback`). Execute sem argumentos para uma visão geral de status. |
| `hermes import` | Restaura um backup do Hermes a partir de um arquivo zip. |
| `hermes logs` | Visualiza, acompanha e filtra os arquivos de log do agente/gateway/erros. |
| `hermes config` | Mostra, edita, migra e consulta os arquivos de configuração. |
| `hermes pairing` | Aprova ou revoga códigos de pareamento de mensagens. |
| `hermes skills` | Navega, instala, publica, audita e configura skills. |
| `hermes bundles` | Agrupa várias skills sob um único slash command `/<name>`. Veja [Skill Bundles](../user-guide/features/skills.md#skill-bundles). |
| `hermes curator` | Manutenção de skills em segundo plano — status, execução, pausa, fixação. Veja [Curator](../user-guide/features/curator.md). |
| `hermes memory` | Configura o provedor de memória externo. Subcomandos específicos de plugin (ex.: `hermes honcho`) são registrados automaticamente quando o respectivo provedor está ativo. |
| `hermes acp` | Executa o Hermes como um servidor ACP para integração com editores. |
| `hermes mcp` | Gerencia configurações de servidores MCP e executa o Hermes como servidor MCP. |
| `hermes plugins` | Gerencia plugins do Hermes Agent (instala, ativa, desativa, remove). |
| `hermes portal` | Status do Nous Portal, link de assinatura e roteamento do Tool Gateway. Veja [Tool Gateway](../user-guide/features/tool-gateway.md). |
| `hermes tools` | Configura as ferramentas ativadas por plataforma. |
| `hermes computer-use` | Instala ou verifica o backend cua-driver (Computer Use no macOS). |
| `hermes pets` | Navega, instala e seleciona [pets do petdex](../user-guide/features/pets.md) animados exibidos na CLI, TUI e app desktop. Subcomandos: `list`, `install`, `select`, `show`, `off`, `scale`, `remove`, `doctor`. |
| `hermes sessions` | Navega, exporta, limpa, renomeia e exclui sessões. |
| `hermes insights` | Mostra análises de tokens/custo/atividade. |
| `hermes claw` | Utilitários de migração do OpenClaw. |
| `hermes dashboard` | Inicia o dashboard web para gerenciar configuração, chaves de API e sessões. |
| `hermes desktop` (alias `gui`) | Compila e inicia o app desktop nativo em Electron. |
| `hermes profile` | Gerencia perfis — múltiplas instâncias isoladas do Hermes. |
| `hermes completion` | Imprime scripts de autocompletar do shell (bash/zsh/fish). |
| `hermes version` | Mostra informações de versão. |
| `hermes update` | Baixa o código mais recente e reinstala as dependências. `--check` mostra uma prévia sem instalar; `--backup` faz um snapshot do `HERMES_HOME` antes do pull. |
| `hermes uninstall` | Remove o Hermes do sistema. |

## `hermes chat`

```bash
hermes chat [options]
```

Opções comuns:

| Opção | Descrição |
|--------|-------------|
| `-q`, `--query "..."` | Prompt não interativo de disparo único. |
| `-m`, `--model <model>` | Sobrescreve o modelo para esta execução. |
| `-t`, `--toolsets <csv>` | Ativa um conjunto de toolsets separados por vírgula. |
| `--provider <provider>` | Força um provedor: `auto`, `openrouter`, `nous`, `openai-codex`, `copilot-acp`, `copilot`, `anthropic`, `gemini`, `huggingface`, `novita` (aliases `novita-ai`, `novitaai`), `openai-api`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `kilocode`, `xiaomi`, `arcee`, `gmi`, `upstage` (alias `solar`), `alibaba`, `alibaba-coding-plan` (alias `alibaba_coding`), `deepseek`, `nvidia`, `ollama-cloud`, `xai` (alias `grok`), `xai-oauth` (alias `grok-oauth`), `qwen-oauth`, `bedrock`, `opencode-zen`, `opencode-go`, `azure-foundry`, `lmstudio`, `stepfun`, `tencent-tokenhub` (alias `tencent`, `tokenhub`). |
| `-s`, `--skills <name>` | Pré-carrega uma ou mais skills para a sessão (pode ser repetido ou separado por vírgula). |
| `-v`, `--verbose` | Saída detalhada. |
| `-Q`, `--quiet` | Modo programático: suprime banner/spinner/prévias de ferramentas. |
| `--image <path>` | Anexa uma imagem local a uma única query. |
| `--resume <session>` / `--continue [name]` | Retoma uma sessão diretamente a partir de `chat`. |
| `--worktree` | Cria uma worktree do git isolada para esta execução. |
| `--checkpoints` | Ativa checkpoints do sistema de arquivos antes de alterações destrutivas em arquivos. |
| `--yolo` | Ignora os prompts de aprovação. |
| `--pass-session-id` | Passa o ID da sessão para o system prompt. |
| `--ignore-user-config` | Ignora `~/.hermes/config.yaml` e usa os padrões embutidos. As credenciais em `.env` continuam sendo carregadas. Útil para execuções de CI isoladas, relatórios de bugs reproduzíveis e integrações de terceiros. |
| `--ignore-rules` | Ignora a injeção automática de `AGENTS.md`, `SOUL.md`, `.cursorrules`, memória persistente e skills pré-carregadas. Combine com `--ignore-user-config` para uma execução totalmente isolada. |
| `--safe-mode` | Modo de solução de problemas: desativa TODAS as personalizações — configuração do usuário, injeção de regras/memória, plugins, hooks de shell e servidores MCP (implica `--ignore-user-config` e `--ignore-rules`). Use para isolar se um problema vem da sua configuração ou do próprio Hermes. |
| `--source <tag>` | Tag de origem da sessão para filtragem (padrão: `cli`). Use `tool` para integrações de terceiros que não devem aparecer nas listas de sessões do usuário. |
| `--max-turns <N>` | Número máximo de iterações de chamada de ferramentas por turno de conversa (padrão: 90, ou `agent.max_turns` na config). |

Exemplos:

```bash
hermes
hermes chat -q "Summarize the latest PRs"
hermes chat --provider openrouter --model anthropic/claude-sonnet-4.6
hermes chat --toolsets web,terminal,skills
hermes chat --quiet -q "Return only JSON"
hermes chat --worktree -q "Review this repo and open a PR"
hermes chat --ignore-user-config --ignore-rules -q "Repro without my personal setup"
hermes chat --safe-mode -q "Is this bug mine or Hermes'?"
```

### `hermes -z <prompt>` — disparo único via script {#hermes-z-prompt-scripted-one-shot}

Para chamadores programáticos (scripts de shell, CI, cron, processos pai encadeando um prompt), `hermes -z` é o ponto de entrada de disparo único mais puro: **um prompt entra, o texto da resposta final sai, e nada mais na stdout ou stderr.** Sem banner, sem spinner, sem prévias de ferramentas, sem linha `Session:` — apenas a resposta final do agente como texto simples.

```bash
hermes -z "What's the capital of France?"
# → Paris.

# Parent scripts can cleanly capture the response:
answer=$(hermes -z "summarize this" < /path/to/file.txt)
```

Sobrescrições por execução (sem alterar `~/.hermes/config.yaml`):

| Flag | Variável de ambiente equivalente | Finalidade |
|---|---|---|
| `-m` / `--model <model>` | `HERMES_INFERENCE_MODEL` | Sobrescreve o modelo para esta execução |
| `--provider <provider>` | _(nenhuma)_ | Sobrescreve o provedor para esta execução |

```bash
hermes -z "…" --provider openrouter --model openai/gpt-5.5
# or:
HERMES_INFERENCE_MODEL=anthropic/claude-sonnet-4.6 hermes -z "…"
```

Mesmo agente, mesmas ferramentas, mesmas skills — apenas remove todas as camadas interativas / cosméticas. Se você precisa da saída das ferramentas na transcrição também, use `hermes chat -q`; `-z` é feito explicitamente para "eu só quero a resposta final".

## `hermes model`

Seletor interativo de provedor + modelo. **Este é o comando para adicionar novos provedores, configurar chaves de API e executar fluxos OAuth.** Execute-o a partir do seu terminal — não de dentro de uma sessão de chat ativa do Hermes.

```bash
hermes model
```

Use isto quando quiser:
- **adicionar um novo provedor** (OpenRouter, Anthropic, Copilot, DeepSeek, personalizado, etc.)
- fazer login em provedores baseados em OAuth (Anthropic, Copilot, Codex, Nous Portal)
- inserir ou atualizar chaves de API
- escolher entre listas de modelos específicas do provedor
- configurar um endpoint personalizado/auto-hospedado
- salvar o novo padrão na config

:::warning hermes model vs /model — conheça a diferença
**`hermes model`** (executado no seu terminal, fora de qualquer sessão do Hermes) é o **assistente completo de configuração de provedores**. Ele pode adicionar novos provedores, executar fluxos OAuth, solicitar chaves de API e configurar endpoints.

**`/model`** (digitado dentro de uma sessão de chat ativa do Hermes) só pode **alternar entre provedores e modelos que você já configurou**. Ele não pode adicionar novos provedores, executar OAuth, ou solicitar chaves de API.

**Se você precisa adicionar um novo provedor:** Saia da sua sessão do Hermes primeiro (`Ctrl+C` ou `/quit`), depois execute `hermes model` a partir do prompt do seu terminal.
:::

### Slash command `/model` (durante a sessão) {#model-slash-command-mid-session}

Alterne entre modelos já configurados sem saltar da sessão:

```
/model                              # Show current model and available options
/model claude-sonnet-4              # Switch model (auto-detects provider)
/model zai:glm-5                    # Switch provider and model
/model custom:qwen-2.5              # Use model on your custom endpoint
/model custom                       # Auto-detect model from custom endpoint
/model custom:local:qwen-2.5        # Use a named custom provider
/model openrouter:anthropic/claude-sonnet-4  # Switch back to cloud
```

Por padrão, as alterações do `/model` se aplicam **apenas à sessão atual**. Adicione `--global` para persistir a mudança no `config.yaml` (ou defina `model.persist_switch_by_default: true` para tornar toda mudança persistente):

```
/model claude-sonnet-4 --global     # Switch and save as new default
```

:::info E se eu só ver modelos da OpenRouter?
Se você só configurou a OpenRouter, o `/model` vai mostrar apenas os modelos da OpenRouter. Para adicionar outro provedor (Anthropic, DeepSeek, Copilot, etc.), saia da sua sessão e execute `hermes model` a partir do terminal.
:::

Em uma troca `--global`, as alterações de provedor e URL base são persistidas no `config.yaml` junto com o modelo. Ao trocar de um endpoint personalizado, a URL base antiga é limpa para evitar que ela vaze para outros provedores.

## `hermes gateway`

```bash
hermes gateway <subcommand>
```

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `run` | Executa o gateway em foreground. Recomendado para WSL, Docker e Termux. |
| `start` | Inicia o serviço em segundo plano instalado no systemd/launchd. |
| `stop` | Para o serviço (ou o processo em foreground). |
| `restart` | Reinicia o serviço. |
| `status` | Mostra o status do serviço. |
| `list` | Lista **todos os perfis** e se o gateway de cada perfil está atualmente em execução (com PID quando disponível). Útil quando você executa vários perfis em paralelo e quer uma visão geral única. |
| `install` | Instala como serviço em segundo plano do systemd (Linux) ou launchd (macOS). |
| `uninstall` | Remove o serviço instalado. |
| `setup` | Configuração interativa de plataformas de mensagens. |
| `migrate-legacy` | Remove unidades legadas `hermes.service` deixadas por instalações anteriores à renomeação. Unidades de perfil (`hermes-gateway-<profile>.service`) e serviços não relacionados nunca são tocados. Flags: `--dry-run`, `-y`/`--yes`. |
| `enroll` | Experimental: registra este gateway com um conector relay e salva as credenciais do relay para plataformas com backend de conector. |

Opções:

| Opção | Descrição |
|--------|-------------|
| `--all` | Em `start` / `restart` / `stop`: age sobre o gateway de **cada perfil**, não apenas o `HERMES_HOME` ativo. Útil se você executa vários perfis em paralelo e quer reiniciar todos após `hermes update`. |
| `--no-supervise` | Em `run`: dentro da imagem Docker s6-overlay, opta por não usar a auto-supervisão e usa a semântica de foreground pré-s6 — o gateway roda como processo principal do container sem auto-reinício. Sem efeito fora da imagem s6. Equivalente a definir `HERMES_GATEWAY_NO_SUPERVISE=1`. |
| `--external-supervisor` | Em `run`: declara que um gerenciador de processos fornecido por um wrapper é responsável pelo gateway em foreground. Use isso quando `sudo`, `env -i`, ou outro wrapper remover o marcador de ambiente nativo do launchd/systemd. Reinícios e atualizações dentro do chat saem de volta para esse gerenciador em vez de gerar um substituto destacado. |

`--external-supervisor` é um contrato de política de reinício: um reinício iniciado dentro do chat ou uma atualização via reinício de serviço encerra com status `75`, então o supervisor do wrapper precisa relançar o gateway após esse código de saída não zero. No systemd, use `Restart=on-failure` ou `Restart=always` e não inclua `75` em `RestartPreventExitStatus`; no launchd, configure `KeepAlive` para relançar após saídas sem sucesso. Sem essa política, um reinício solicitado deixa o gateway parado.

`hermes gateway enroll` aceita `--token`, `--connector-url`, `--gateway-id` e `--wake-url`. Ele troca o token de registro com o conector e grava os valores resultantes `GATEWAY_RELAY_ID`, `GATEWAY_RELAY_SECRET`, `GATEWAY_RELAY_DELIVERY_KEY`, opcionalmente `GATEWAY_RELAY_URL`, e (quando `--wake-url` é fornecido) `GATEWAY_RELAY_WAKE_URL` no `.env` do perfil ativo.

:::tip Usuários do WSL
Use `hermes gateway run` em vez de `hermes gateway start` — o suporte a systemd do WSL é instável. Envolva em tmux para persistência: `tmux new -s hermes 'hermes gateway run'`. Veja [FAQ do WSL](/reference/faq#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails) para detalhes.
:::

## `hermes lsp`

```bash
hermes lsp <subcommand>
```

Gerencia a integração com o Language Server Protocol. O LSP executa servidores de linguagem reais (pyright, gopls, rust-analyzer, …) em segundo plano e alimenta seus diagnósticos na verificação pós-escrita usada por `write_file` e `patch`. Condicionado à detecção de workspace git — o LSP só é executado quando o cwd ou o arquivo editado estão dentro de uma worktree do git.

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `status` | Mostra o estado do serviço, os servidores configurados, o status de instalação. |
| `list` | Imprime o registro de servidores suportados. Passe `--installed-only` para pular os ausentes. |
| `install <id>` | Instala o binário de um servidor específico imediatamente. |
| `install-all` | Instala todo servidor com uma receita de auto-instalação conhecida. |
| `restart` | Encerra os clientes em execução para que a próxima edição os reinicie. |
| `which <id>` | Imprime o caminho resolvido do binário de um servidor. |

Veja [LSP — Diagnósticos Semânticos](/user-guide/features/lsp) para o guia completo, linguagens suportadas e opções de configuração.

## `hermes setup`

```bash
hermes setup [model|tts|terminal|gateway|tools|agent] [--non-interactive] [--reset] [--quick] [--reconfigure] [--portal]
```

**Caminho mais fácil:** `hermes setup --portal` — autentica via OAuth no Nous Portal e ativa o [Tool Gateway](../user-guide/features/tool-gateway.md) em um único passo.

**Primeira execução:** inicia o assistente de primeiro uso.

**Usuário recorrente (já configurado):** entra direto no assistente completo de reconfiguração — cada prompt mostra seu valor atual como padrão, pressione Enter para manter ou digite um novo valor. Sem menu.

Vá direto a uma seção em vez do assistente completo:

| Seção | Descrição |
|---------|-------------|
| `model` | Configuração de provedor e modelo. |
| `terminal` | Configuração de backend de terminal e sandbox. |
| `gateway` | Configuração de plataformas de mensagens. |
| `tools` | Ativa/desativa ferramentas por plataforma. |
| `agent` | Configurações de comportamento do agente. |

Opções:

| Opção | Descrição |
|--------|-------------|
| `--quick` | Em execuções de usuário recorrente: só pergunta pelos itens que estão ausentes ou não configurados. Pula os itens que você já configurou. |
| `--non-interactive` | Usa padrões / valores de ambiente sem prompts. |
| `--reset` | Restaura a configuração para os padrões antes da configuração. |
| `--reconfigure` | Alias retrocompatível — `hermes setup` puro em uma instalação existente agora faz isso por padrão. |
| `--portal` | Configuração de disparo único do Nous Portal: faz login via OAuth, define o Nous como provedor de inferência e ativa o [Tool Gateway](../user-guide/features/tool-gateway.md). Pula o restante do assistente. |

## `hermes portal`

```bash
hermes portal [status|open|tools]
```

Inspeciona a autenticação do Nous Portal, o roteamento do Tool Gateway e acessa a página de assinatura. A invocação sem subcomando executa `status`.

| Subcomando | Descrição |
|------------|-------------|
| `status` (padrão) | Estado de autenticação do Portal + resumo do roteamento do Tool Gateway por ferramenta. Também exibido quando nenhum subcomando é dado. |
| `open` | Abre `portal.nousresearch.com/manage-subscription` no seu navegador padrão. |
| `tools` | Lista todo parceiro do Tool Gateway (Firecrawl, FAL, OpenAI TTS, Browser Use, Modal) e quais são roteados via Nous. |

Para configuração do próprio gateway, veja [Tool Gateway](../user-guide/features/tool-gateway.md). Para o caminho de configuração de disparo único, veja `hermes setup --portal` acima.

## `hermes whatsapp`

```bash
hermes whatsapp
```

Executa o fluxo de pareamento/configuração do WhatsApp, incluindo seleção de modo e pareamento por QR code.

## `hermes slack`

```bash
hermes slack manifest              # print manifest to stdout
hermes slack manifest --write      # write to ~/.hermes/slack-manifest.json
hermes slack manifest --slashes-only  # just the features.slash_commands array
```

Gera um manifesto de app do Slack que registra todo comando do gateway em `COMMAND_REGISTRY` (`/btw`, `/stop`, `/model`, …) como um slash command nativo de primeira classe do Slack — equivalente ao Discord e ao Telegram. Cole a saída na configuração do seu app do Slack em
[https://api.slack.com/apps](https://api.slack.com/apps) → seu app →
**Features → App Manifest → Edit**, e depois **Save**. O Slack solicita reinstalação se os escopos ou slash commands mudarem.

| Flag | Padrão | Finalidade |
|------|---------|---------|
| `--write [PATH]` | stdout | Grava em um arquivo em vez da stdout. `--write` sozinho grava em `$HERMES_HOME/slack-manifest.json`. |
| `--name NAME` | `Hermes` | Nome de exibição do bot no Slack. |
| `--description DESC` | texto padrão | Descrição do bot exibida no diretório de apps do Slack. |
| `--slashes-only` | desativado | Emite apenas `features.slash_commands`, para juntar a um manifesto mantido manualmente. |

Execute `hermes slack manifest --write` novamente após `hermes update` para incorporar novos comandos.


## `hermes send`

```bash
hermes send --to <target> "message text"
hermes send --to <target> --file <path>
echo "message" | hermes send --to <target>
hermes send --list [platform]
```

Envia uma mensagem de disparo único a uma plataforma de mensagens configurada sem iniciar um agente ou loop de gateway. Reutiliza as credenciais já configuradas do gateway (`~/.hermes/.env` + `~/.hermes/config.yaml`) para que scripts de operações, cron jobs, hooks de CI e daemons de monitoramento possam publicar atualizações de status sem reimplementar o cliente REST de cada plataforma.

Para plataformas de token de bot (Telegram, Discord, Slack, Signal, SMS, WhatsApp-CloudAPI) nenhum gateway em execução é necessário — `hermes send` fala diretamente com o endpoint REST da plataforma. Plataformas de plugin que precisam de um adaptador persistente ainda exigem um gateway ativo.

| Opção | Descrição |
|--------|-------------|
| `-t`, `--to <TARGET>` | Destino de entrega. Formatos: `platform` (usa o canal principal), `platform:chat_id`, `platform:chat_id:thread_id`, ou `platform:#channel-name`. Exemplos: `telegram`, `telegram:-1001234567890`, `discord:#ops`, `slack:C0123ABCD`, `signal:+15551234567`. |
| `-f`, `--file <PATH>` | Lê o corpo da mensagem de `PATH` (apenas arquivos de texto — logs, relatórios, markdown). Passe `-` para forçar a leitura da stdin. Para enviar uma imagem ou outro arquivo binário, use `MEDIA:<path>` (veja abaixo). |
| `-s`, `--subject <LINE>` | Adiciona uma linha de assunto/cabeçalho antes do corpo da mensagem. |
| `-l`, `--list [platform]` | Lista os destinos configurados em todas as plataformas (ou apenas na plataforma dada). |
| `-q`, `--quiet` | Suprime a saída em caso de sucesso — útil em scripts (confie apenas no código de saída). |
| `--json` | Emite o resultado como JSON puro em vez de saída legível para humanos. |

Se nem um argumento posicional `message` nem `--file` for fornecido, `hermes send` lê da stdin quando ela não é um TTY. Códigos de saída: `0` em sucesso, `1` em falha de entrega/backend, `2` em erros de uso.

### Enviando imagens e outras mídias {#sending-images-and-other-media}

`--file` serve apenas para corpos de *texto*. Para entregar uma imagem, documento, vídeo ou arquivo de áudio como um anexo nativo da plataforma, referencie-o dentro do texto da mensagem com a diretiva `MEDIA:<local_path>`:

```bash
hermes send --to telegram "MEDIA:/tmp/screenshot.png"
hermes send --to telegram "Build chart for today MEDIA:/tmp/chart.png"   # with caption
hermes send --to discord:#ops "MEDIA:/tmp/report.pdf"
```

Por padrão, arquivos de imagem são enviados como fotos (plataformas como o Telegram as recomprimem). Adicione `[[as_document]]` à mensagem para entregá-las como anexos de arquivo sem compressão:

```bash
hermes send --to telegram "[[as_document]] MEDIA:/tmp/screenshot.png"
```

Exemplos:

```bash
hermes send --to telegram "deploy finished"
echo "RAM 92%" | hermes send --to telegram:-1001234567890
hermes send --to discord:#ops --file /tmp/report.md
hermes send --to slack:#eng --subject "[CI]" --file build.log
hermes send --list                  # all platforms
hermes send --list telegram         # filter by platform
```


## `hermes secrets`

```bash
hermes secrets bitwarden <subcommand>
hermes secrets bw <subcommand>          # short alias
```

Busca chaves de API de um gerenciador de segredos externo na inicialização do processo, em vez de armazená-las em `~/.hermes/.env`. Atualmente suporta o **Bitwarden Secrets Manager**. Veja o guia completo: [Integração com Bitwarden](../user-guide/secrets/bitwarden.md).

Subcomandos de `bitwarden` (alias `bw`):

| Subcomando | Descrição |
|------------|-------------|
| `setup` | Assistente interativo: instala o binário `bws` fixado, armazena um token de acesso e escolhe um projeto. Aceita `--project-id`, `--access-token` e `--server-url` para uso não interativo. |
| `status` | Mostra a configuração atual, caminho/versão do binário e informações da última busca. |
| `token` | Rotaciona o token de acesso: valida o novo token junto ao Bitwarden antes de armazená-lo no `.env` (um token rejeitado não altera nada). Aceita `--access-token` para uso não interativo e `--no-verify` para pular a verificação. |
| `sync` | Busca os segredos agora e reporta o que mudou. Adicione `--apply` para de fato exportar os segredos para o ambiente do shell atual (o padrão é dry-run). |
| `install` | Baixa e verifica o binário `bws` fixado. `--force` baixa novamente mesmo que já exista uma cópia gerenciada. |
| `disable` | Desativa a integração com o Bitwarden. |


## `hermes migrate`

```bash
hermes migrate <type>
```

Diagnostica e (opcionalmente) reescreve o `config.yaml` ativo para substituir referências a modelos descontinuados ou configurações obsoletas. Um backup com timestamp do `config.yaml` original é feito antes de qualquer reescrita (pule com `--no-backup`).

| Subcomando | Descrição |
|------------|-------------|
| `xai` | Verifica o `config.yaml` em busca de referências a modelos xAI programados para descontinuação em 15 de maio de 2026 e (com `--apply`) os reescreve no local pelos substitutos oficiais, conforme o guia de migração da xAI. O padrão é dry-run. |

Flags comuns dos subcomandos de migração:

| Flag | Descrição |
|------|-------------|
| `--apply` | Reescreve o `config.yaml` no local (padrão: dry-run, sem gravações). |
| `--no-backup` | Pula o backup com timestamp do `config.yaml` ao aplicar. |

> Não confundir com `hermes claw migrate` (importação de disparo único da configuração do OpenClaw para o Hermes) — `hermes migrate` é o comando de reescrita de configuração de nível superior.


## `hermes proxy`

```bash
hermes proxy <subcommand>
```

Executa um servidor HTTP local compatível com OpenAI que encaminha requisições a um provedor upstream autenticado via OAuth (ex.: Nous Portal, xAI). Aplicativos externos podem apontar para o proxy com qualquer bearer token; o proxy aplica suas credenciais OAuth reais na saída. Veja [Subscription Proxy](../user-guide/features/subscription-proxy.md) para o guia completo.

| Subcomando | Descrição |
|------------|-------------|
| `start` | Executa o proxy em foreground. Flags: `--provider <nous\|xai>` (padrão `nous`), `--host <addr>` (padrão `127.0.0.1`; use `0.0.0.0` para expor na LAN), `--port <int>` (padrão `8645`). |
| `status` | Mostra quais upstreams do proxy estão prontos (credenciais presentes, OAuth válido). |
| `providers` | Lista os provedores upstream de proxy disponíveis. |


## `hermes security`

```bash
hermes security <subcommand>
```

Verificação de vulnerabilidades sob demanda contra o [OSV.dev](https://osv.dev). Cobre o venv do Hermes (distribuições PyPI instaladas), dependências Python declaradas por plugins em `~/.hermes/plugins/`, e servidores MCP fixados via `npx`/`uvx` no `config.yaml`. NÃO verifica pacotes instalados globalmente ou extensões de editor/navegador.

| Subcomando | Descrição |
|------------|-------------|
| `audit` | Executa uma auditoria única da cadeia de suprimentos. |

Flags de `audit`:

| Flag | Padrão | Descrição |
|------|---------|-------------|
| `--json` | desativado | Emite JSON legível por máquina em vez de texto legível por humanos. |
| `--fail-on <level>` | `critical` | Encerra com código diferente de zero quando algum resultado atinge esta severidade (`low`, `moderate`, `high`, `critical`). |
| `--skip-venv` | desativado | Pula a verificação do venv Python do Hermes. |
| `--skip-plugins` | desativado | Pula a verificação dos arquivos de requisitos de plugins. |
| `--skip-mcp` | desativado | Pula a verificação dos servidores MCP fixados no `config.yaml`. |


## `hermes login` / `hermes logout` *(Descontinuado)*

:::caution
`hermes login` foi removido. Use `hermes auth` para gerenciar credenciais OAuth, `hermes model` para selecionar um provedor, ou `hermes setup` para a configuração interativa completa.
:::

## `hermes auth`

Gerencia pools de credenciais para rotação de chaves do mesmo provedor. Veja [Credential Pools](/user-guide/features/credential-pools) para a documentação completa.

```bash
hermes auth                                              # Interactive wizard
hermes auth list                                         # Show all pools
hermes auth list openrouter                              # Show specific provider
hermes auth add openrouter --api-key sk-or-v1-xxx        # Add API key
hermes auth add anthropic --type oauth                   # Add OAuth credential
hermes auth remove openrouter 2                          # Remove by index
hermes auth reset openrouter                             # Clear cooldowns
hermes auth status anthropic                             # Show auth status for a provider
hermes auth logout anthropic                             # Log out and clear stored auth state
hermes auth spotify                                      # Authenticate Hermes with Spotify via PKCE
```

Subcomandos: `add`, `list`, `remove`, `reset`, `status`, `logout`, `spotify`. Quando chamado sem subcomando, inicia o assistente interativo de gerenciamento.

## `hermes status`

```bash
hermes status [--all] [--deep]
```

| Opção | Descrição |
|--------|-------------|
| `--all` | Mostra todos os detalhes em um formato redigido e compartilhável. |
| `--deep` | Executa verificações mais profundas que podem levar mais tempo. |

## `hermes cron`

```bash
hermes cron <list|create|edit|pause|resume|run|remove|status|tick>
```

| Subcomando | Descrição |
|------------|-------------|
| `list` | Mostra os jobs agendados. |
| `create` / `add` | Cria um job agendado a partir de um prompt, opcionalmente anexando uma ou mais skills via `--skill` repetido. |
| `edit` | Atualiza a agenda, o prompt, o nome, a entrega, o número de repetições ou as skills anexadas de um job. Suporta `--clear-skills`, `--add-skill` e `--remove-skill`. |
| `pause` | Pausa um job sem excluí-lo. |
| `resume` | Retoma um job pausado e calcula sua próxima execução futura. |
| `run` | Dispara um job no próximo tick do agendador. |
| `remove` | Exclui um job agendado. |
| `status` | Verifica se o agendador de cron está em execução. |
| `tick` | Executa os jobs devidos uma vez e encerra. |

O **gatilho** do cron é conectável via a chave de config `cron.provider`. Vazio
(o padrão) usa o ticker embutido no processo. Defina como `chronos` (o
provedor gerenciado pela NAS para gateways hospedados escaláveis a zero) — configurado via
as chaves `cron.chronos.*` (`portal_url`, `callback_url`, `expected_audience`,
`nas_jwks_url`) — ou nomeie um provedor personalizado em `plugins/cron/<name>/` ou
`$HERMES_HOME/plugins/<name>/`. Um provedor desconhecido ou indisponível recorre ao
embutido, então o cron nunca fica sem um gatilho. Veja o
documento de [internals do cron](../developer-guide/cron-internals.md#gateway-integration).

## `hermes kanban`

```bash
hermes kanban [--board <slug>] <action> [options]
```

Quadro de colaboração multi-perfil e multi-projeto. Cada instalação pode hospedar vários quadros (um por projeto, repositório ou domínio); cada quadro é uma fila independente com seu próprio banco SQLite e escopo de dispatcher. Novas instalações começam com um quadro chamado `default`, cujo banco é `~/.hermes/kanban.db` por retrocompatibilidade; quadros adicionais ficam em `~/.hermes/kanban/boards/<slug>/kanban.db`. O dispatcher embutido no gateway varre todos os quadros a cada tick.

**Flags globais (aplicam-se a toda ação abaixo):**

| Flag | Finalidade |
|------|---------|
| `--board <slug>` | Opera em um quadro específico. O padrão é o quadro atual (definido via `hermes kanban boards switch`, a variável de ambiente `HERMES_KANBAN_BOARD`, ou `default`). |

**Esta é a superfície humana / de scripts.** Trabalhadores agentes gerados pelo dispatcher operam o quadro através de um [toolset](/user-guide/features/kanban#how-workers-interact-with-the-board) dedicado `kanban_*` (`kanban_show`, `kanban_complete`, `kanban_block`, `kanban_create`, `kanban_link`, `kanban_comment`, `kanban_heartbeat`; perfis orquestradores também recebem `kanban_list` e `kanban_unblock`) em vez de invocar `hermes kanban` via shell. Os trabalhadores têm `HERMES_KANBAN_BOARD` fixado em seu ambiente, então fisicamente não conseguem ver outros quadros.

| Ação | Finalidade |
|--------|---------|
| `init` | Cria o `kanban.db` se estiver faltando. Idempotente. |
| `boards list` / `boards ls` | Lista todos os quadros com contagens de tarefas. `--json`, `--all` (inclui arquivados). |
| `boards create <slug>` | Cria um novo quadro. Flags: `--name`, `--description`, `--icon`, `--color`, `--switch` (torna ativo). O slug é kebab-case, minúsculo automaticamente. |
| `boards switch <slug>` / `boards use` | Persiste `<slug>` como o quadro ativo (grava em `~/.hermes/kanban/current`). |
| `boards show` / `boards current` | Imprime o nome, o caminho do banco e as contagens de tarefas do quadro ativo. |
| `boards rename <slug> "<name>"` | Altera o nome de exibição de um quadro. O slug é imutável. |
| `boards rm <slug>` | Arquiva (padrão) ou exclui definitivamente um quadro. `--delete` pula a etapa de arquivamento. Quadros arquivados vão para `boards/_archived/<slug>-<ts>/`. Recusado para `default`. |
| `create "<title>"` | Cria uma nova tarefa no quadro ativo. Flags: `--body`, `--assignee`, `--parent` (repetível), `--workspace scratch\|worktree\|dir:<path>`, `--tenant`, `--priority`, `--triage`, `--idempotency-key`, `--max-runtime`, `--max-retries`, `--skill` (repetível). |
| `list` / `ls` | Lista as tarefas do quadro ativo. Filtre com `--mine`, `--assignee`, `--status`, `--tenant`, `--archived`, `--json`. |
| `show <id>` | Mostra uma tarefa com comentários e eventos. `--json` para saída legível por máquina. |
| `assign <id> <profile>` | Atribui ou reatribui. Use `none` para desatribuir. Recusado enquanto a tarefa está em execução. |
| `link <parent> <child>` | Adiciona uma dependência. Ciclos são detectados. Ambas as tarefas devem estar no mesmo quadro. |
| `unlink <parent> <child>` | Remove uma dependência. |
| `claim <id>` | Reivindica atomicamente uma tarefa pronta. Imprime o caminho do workspace resolvido. |
| `comment <id> "<text>"` | Adiciona um comentário. O próximo trabalhador que reivindicar a tarefa o lê como parte de sua resposta a `kanban_show()`. |
| `complete <id>` | Marca a tarefa como concluída. Flags: `--result`, `--summary`, `--metadata`. |
| `block <id> "<reason>"` | Marca a tarefa como bloqueada esperando entrada humana. Também adiciona o motivo como um comentário. |
| `schedule <id> "<reason>"` | Estaciona trabalho de atraso/acompanhamento em `scheduled` para que não apareça como um bloqueio humano. |
| `unblock <id>` | Retorna uma tarefa bloqueada ou agendada para pronta (ou `todo` se as dependências ainda estiverem abertas). |
| `archive <id>` | Oculta da lista padrão. `gc` removerá os workspaces temporários. |
| `tail <id>` | Acompanha o fluxo de eventos de uma tarefa. |
| `dispatch` | Uma passagem do dispatcher no quadro ativo. Flags: `--dry-run`, `--max N`, `--failure-limit N`, `--json`. |
| `context <id>` | Imprime o contexto completo que um trabalhador veria (título + corpo + resultados dos pais + comentários). |
| `specify <id>` / `specify --all` | Desenvolve uma tarefa da coluna de triagem em uma especificação concreta (título + corpo com objetivo, abordagem, critérios de aceite) via o LLM auxiliar, depois a promove para `todo`. Flags: `--tenant` (limita `--all` a um tenant), `--author`, `--json`. Configure o modelo em `auxiliary.triage_specifier` no `config.yaml`. |
| `decompose <id>` / `decompose --all` | Divide uma tarefa da coluna de triagem em um grafo de tarefas filhas roteadas para perfis especialistas com base na descrição. Recorre à promoção de tarefa única no estilo `specify` quando o LLM decide que a tarefa não se beneficia da divisão. Mesmas flags de `specify`. Configure o modelo do decompositor em `auxiliary.kanban_decomposer` no `config.yaml`; `kanban.orchestrator_profile` só controla quem fica responsável pela tarefa raiz/orquestração após a divisão. Também é executado automaticamente a cada tick do dispatcher quando `kanban.auto_decompose: true` (o padrão). Veja [Orquestração automática vs manual](/user-guide/features/kanban#auto-vs-manual-orchestration). |
| `gc` | Remove workspaces temporários de tarefas arquivadas. |

Exemplos:

```bash
# Create a second board and put a task on it without switching away.
hermes kanban boards create atm10-server --name "ATM10 Server" --icon 🎮
hermes kanban --board atm10-server create "Restart server" --assignee ops

# Switch the active board for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban list                  # shows atm10-server tasks

# Archive a board (recoverable) or hard-delete it.
hermes kanban boards rm atm10-server
hermes kanban boards rm atm10-server --delete
```

Ordem de resolução do quadro (maior precedência primeiro): flag `--board <slug>` → variável de ambiente `HERMES_KANBAN_BOARD` → arquivo `~/.hermes/kanban/current` → `default`.

Todas as ações também estão disponíveis como slash command no gateway (`/kanban …`), com a mesma superfície de argumentos — incluindo subcomandos `boards` e a flag `--board`.

Para o design completo — comparação com Cline Kanban / Paperclip / NanoClaw / Gemini Enterprise, oito padrões de colaboração, quatro histórias de usuário, prova de correção de concorrência — veja `docs/hermes-kanban-v1-spec.pdf` no repositório ou o [guia de usuário do Kanban](/user-guide/features/kanban).

## `hermes project`

```bash
hermes project <create|list|show|add-folder|remove-folder|rename|set-primary|use|archive|restore|bind-board>
```

Projetos são workspaces nomeados por humanos que podem abranger múltiplas pastas / repositórios. Eles ancoram o agrupamento de sessões no desktop e, quando vinculados a um quadro kanban, dão às tarefas uma convenção determinística de worktree + branch. O estado é por perfil.

| Subcomando | Descrição |
|------------|-------------|
| `create` | Cria um novo projeto. |
| `list` (alias `ls`) | Lista os projetos. |
| `show` | Mostra os detalhes de um projeto. |
| `add-folder` | Adiciona uma pasta / repositório a um projeto. |
| `remove-folder` | Remove uma pasta de um projeto. |
| `rename` | Renomeia um projeto. |
| `set-primary` | Define a pasta principal. |
| `use` | Define o projeto ativo. |
| `archive` | Arquiva um projeto (recuperável). |
| `restore` | Restaura um projeto arquivado. |
| `bind-board` | Vincula um quadro kanban a este projeto. |

## `hermes webhook`

```bash
hermes webhook <subscribe|list|remove|test>
```

Gerencia assinaturas dinâmicas de webhook para ativação do agente orientada a eventos. Requer que a plataforma de webhook esteja ativada na config — se não estiver configurada, imprime instruções de configuração.

| Subcomando | Descrição |
|------------|-------------|
| `subscribe` / `add` | Cria uma rota de webhook. Retorna a URL e o segredo HMAC para configurar no seu serviço. |
| `list` / `ls` | Mostra todas as assinaturas criadas pelo agente. |
| `remove` / `rm` | Exclui uma assinatura dinâmica. Rotas estáticas do config.yaml não são afetadas. |
| `test` | Envia um POST de teste para verificar se uma assinatura está funcionando. |

### `hermes webhook subscribe`

```bash
hermes webhook subscribe <name> [options]
```

| Opção | Descrição |
|--------|-------------|
| `--prompt` | Template de prompt com referências de payload em `{dot.notation}`. |
| `--events` | Tipos de evento aceitos, separados por vírgula (ex.: `issues,pull_request`). Vazio = todos. |
| `--description` | Descrição legível por humanos. |
| `--skills` | Nomes de skills separados por vírgula a carregar para a execução do agente. |
| `--deliver` | Destino de entrega: `log` (padrão), `telegram`, `discord`, `slack`, `github_comment`. |
| `--deliver-chat-id` | ID do chat/canal de destino para entrega entre plataformas. |
| `--secret` | Segredo HMAC personalizado. Gerado automaticamente se omitido. |
| `--deliver-only` | Pula o agente — entrega o `--prompt` renderizado como a mensagem literal. Custo zero de LLM, entrega em menos de um segundo. Requer que `--deliver` seja um destino real (não `log`). |
| `--script` | Script de filtro/transformação em `~/.hermes/scripts/`. O payload do webhook é passado como JSON na stdin; a saída JSON na stdout substitui o payload, e stdout vazio, `[SILENT]`, ou um código de saída diferente de zero ignora o webhook. Veja [Filtros e Transformações via Script](../user-guide/messaging/webhooks.md#script-filters-and-transforms). |

As assinaturas persistem em `~/.hermes/webhook_subscriptions.json` e são recarregadas a quente pelo adaptador de webhook sem reiniciar o gateway.

## `hermes doctor`

```bash
hermes doctor [--fix]
```

| Opção | Descrição |
|--------|-------------|
| `--fix` | Tenta reparos automáticos quando possível. |

## `hermes dump`

```bash
hermes dump [--show-keys]
```

Produz um resumo compacto em texto simples de toda a sua configuração do Hermes. Projetado para ser colado no Discord, em issues do GitHub ou no Telegram ao pedir suporte — sem cores ANSI, sem formatação especial, apenas dados.

| Opção | Descrição |
|--------|-------------|
| `--show-keys` | Mostra prefixos redigidos das chaves de API (primeiros e últimos 4 caracteres) em vez de apenas `set`/`not set`. |

### O que inclui {#what-it-includes}

| Seção | Detalhes |
|---------|---------|
| **Cabeçalho** | Versão do Hermes, data de lançamento, hash do commit git |
| **Ambiente** | SO, versão do Python, versão do SDK da OpenAI |
| **Identidade** | Nome do perfil ativo, caminho do HERMES_HOME |
| **Modelo** | Modelo e provedor padrão configurados |
| **Terminal** | Tipo de backend (local, docker, ssh, etc.) |
| **Chaves de API** | Verificação de presença para as 22 chaves de API de provedores/ferramentas |
| **Recursos** | Toolsets ativados, contagem de servidores MCP, provedor de memória |
| **Serviços** | Status do gateway, plataformas de mensagens configuradas |
| **Carga de trabalho** | Contagens de jobs de cron, número de skills instaladas |
| **Sobrescrições de config** | Quaisquer valores de config que diferem dos padrões |

### Exemplo de saída {#example-output}

```
--- hermes dump ---
version:          0.8.0 (2026.4.8) [af4abd2f]
os:               Linux 6.14.0-37-generic x86_64
python:           3.11.14
openai_sdk:       2.24.0
profile:          default
hermes_home:      ~/.hermes
model:            anthropic/claude-opus-4.6
provider:         openrouter
terminal:         local

api_keys:
  openrouter           set
  openai               not set
  anthropic            set
  nous                 not set
  firecrawl            set
  ...

features:
  toolsets:           all
  mcp_servers:        0
  memory_provider:    built-in
  gateway:            running (systemd)
  platforms:          telegram, discord
  cron_jobs:          3 active / 5 total
  skills:             42

config_overrides:
  agent.max_turns: 250
  compression.threshold: 0.85
  display.streaming: True
--- end dump ---
```

### Quando usar {#when-to-use}

- Ao reportar um bug no GitHub — cole o dump na sua issue
- Ao pedir ajuda no Discord — compartilhe em um bloco de código
- Ao comparar sua configuração com a de outra pessoa
- Verificação rápida de sanidade quando algo não está funcionando

:::tip
`hermes dump` é feito especificamente para compartilhamento. Para diagnósticos interativos, use `hermes doctor`. Para uma visão geral visual, use `hermes status`.
:::

## `hermes debug`

```bash
hermes debug share [options]
```

Envia um relatório de depuração (informações do sistema + logs recentes) para um serviço de paste e obtém uma URL compartilhável. Útil para pedidos de suporte rápidos — inclui tudo que um ajudante precisa para diagnosticar seu problema.

| Opção | Descrição |
|--------|-------------|
| `--lines <N>` | Número de linhas de log a incluir por arquivo de log (padrão: 200). |
| `--expire <days>` | Validade do paste em dias (padrão: 7). |
| `--nous` | Envia para o armazenamento de diagnósticos interno da Nous em vez de um serviço de paste público. Use quando o suporte da Nous solicitar um pacote de diagnóstico privado. |
| `--local` | Imprime o relatório localmente em vez de enviá-lo. |
| `--no-redact` | Desativa a redação de segredos no momento do envio. Por padrão, os envios são redigidos. |

O relatório inclui informações do sistema (SO, versão do Python, versão do Hermes), logs recentes do agente, gateway, GUI/dashboard e desktop (limite de 512 KB por arquivo), e status redigido das chaves de API. Por padrão, os envios são redigidos para que segredos não sejam incluídos.

Os envios padrão usam serviços de paste públicos, tentados em ordem: paste.rs, dpaste.com. `--nous` envia o mesmo pacote de depuração para o armazenamento de diagnósticos privado da Nous; o link do visualizador retornado é para a equipe da Nous e se auto-exclui após 14 dias.

### Exemplos {#examples}

```bash
hermes debug share              # Upload debug report, print URL
hermes debug share --lines 500  # Include more log lines
hermes debug share --expire 30  # Keep paste for 30 days
hermes debug share --nous       # Upload a private diagnostics bundle for Nous support
hermes debug share --local      # Print report to terminal (no upload)
```

## `hermes backup`

```bash
hermes backup [options]
```

Cria um arquivo zip da sua configuração, skills, sessões e dados do Hermes. O backup exclui o próprio código-fonte do hermes-agent.

| Opção | Descrição |
|--------|-------------|
| `-o`, `--output <path>` | Caminho de saída para o arquivo zip (padrão: `~/hermes-backup-<timestamp>.zip`). |
| `-q`, `--quick` | Snapshot rápido: apenas arquivos de estado críticos (config.yaml, state.db, .env, auth, cron jobs). Muito mais rápido que um backup completo. |
| `-l`, `--label <name>` | Rótulo para o snapshot (usado apenas com `--quick`). |

O backup usa a API `backup()` do SQLite para cópia segura, então funciona corretamente mesmo quando o Hermes está em execução (seguro em modo WAL).

**O que é excluído do zip:**

- `*.db-wal`, `*.db-shm`, `*.db-journal` — os arquivos auxiliares de WAL / memória compartilhada / journal do SQLite. O arquivo `*.db` já obteve um snapshot consistente via `sqlite3.backup()`; incluir os auxiliares ativos junto a ele permitiria que uma restauração visse um estado parcialmente confirmado.
- `checkpoints/` — caches de trajetória por sessão. Indexados por hash e regenerados por sessão; não seriam portados corretamente para outra instalação de qualquer forma.
- O próprio código do `hermes-agent` (este é um backup de dados do usuário, não um snapshot do repositório).

### Exemplos {#examples-1}

```bash
hermes backup                           # Full backup to ~/hermes-backup-*.zip
hermes backup -o /tmp/hermes.zip        # Full backup to specific path
hermes backup --quick                   # Quick state-only snapshot
hermes backup --quick --label "pre-upgrade"  # Quick snapshot with label
```

## `hermes checkpoints`

```bash
hermes checkpoints [COMMAND]
```

Inspeciona e gerencia o armazenamento sombra do git em `~/.hermes/checkpoints/` — a camada de armazenamento por trás do comando `/rollback` dentro da sessão. Seguro para executar em qualquer momento; não requer que o agente esteja em execução.

| Subcomando | Descrição |
|------------|-------------|
| `status` (padrão) | Mostra o tamanho total, a contagem de projetos e o detalhamento por projeto. `hermes checkpoints` puro é equivalente. |
| `list` | Alias para `status`. |
| `prune` | Força uma limpeza — exclui projetos órfãos e obsoletos, faz GC no armazenamento, aplica o limite de tamanho. Ignora o marcador de idempotência de 24h. |
| `clear` | Exclui toda a base de checkpoints. Irreversível; pede confirmação salvo com `-f`. |
| `clear-legacy` | Exclui apenas os arquivos `legacy-<timestamp>/` produzidos pela migração v1→v2. |

### Opções {#options}

| Opção | Subcomando | Descrição |
|--------|------------|-------------|
| `--limit N` | `status`, `list` | Máximo de projetos a listar (padrão 20). |
| `--retention-days N` | `prune` | Descarta projetos cujo `last_touch` seja mais antigo que N dias (padrão 7). |
| `--max-size-mb N` | `prune` | Após a passagem de órfãos/obsoletos, descarta o commit mais antigo de cada projeto até o tamanho total do armazenamento ser ≤ N MB (padrão 500). |
| `--keep-orphans` | `prune` | Pula a exclusão de projetos cujo diretório de trabalho não existe mais. |
| `-f`, `--force` | `clear`, `clear-legacy` | Pula o prompt de confirmação. |

### Exemplos {#examples-2}

```bash
hermes checkpoints                                  # status overview
hermes checkpoints prune --retention-days 3         # aggressive cleanup
hermes checkpoints prune --max-size-mb 200          # tighten size cap once
hermes checkpoints clear-legacy -f                  # drop v1 archive dirs
hermes checkpoints clear -f                         # wipe everything
```

Veja [Checkpoints e `/rollback`](../user-guide/checkpoints-and-rollback.md) para a arquitetura completa e os comandos disponíveis durante a sessão.

## `hermes import`

```bash
hermes import <zipfile> [options]
```

Restaura um backup do Hermes criado anteriormente no seu diretório home do Hermes. Todos os arquivos do arquivo sobrescrevem os arquivos existentes no seu home do Hermes; `--force` apenas pula o prompt de confirmação que dispara quando o destino já tem uma instalação do Hermes.

| Opção | Descrição |
|--------|-------------|
| `-f`, `--force` | Pula o prompt de confirmação de instalação existente. |

:::warning
Pare o gateway antes de importar para evitar conflitos com processos em execução.
:::

### Exemplos {#examples-3}
```bash
hermes import ~/hermes-backup-20260423.zip           # Prompts before overwriting existing config
hermes import ~/hermes-backup-20260423.zip --force   # Overwrite without prompting
```

## `hermes logs`

```bash
hermes logs [log_name] [options]
```

Visualiza, acompanha e filtra os arquivos de log do Hermes. Todos os logs são armazenados em `~/.hermes/logs/` (ou `<profile>/logs/` para perfis não padrão).

### Arquivos de log {#log-files}

| Nome | Arquivo | O que captura |
|------|------|-----------------|
| `agent` (padrão) | `agent.log` | Toda a atividade do agente — chamadas de API, despacho de ferramentas, ciclo de vida da sessão (INFO e acima) |
| `errors` | `errors.log` | Apenas avisos e erros — um subconjunto filtrado do agent.log |
| `gateway` | `gateway.log` | Atividade do gateway de mensagens — conexões de plataforma, despacho de mensagens, eventos de webhook |
| `gui` | `gui.log` | Eventos do dashboard / TUI-gateway / PTY-bridge / websocket |
| `desktop` | `desktop.log` | App desktop Electron — inicialização, saída de disparo do backend, e tracebacks Python recentes |

### Opções {#options-1}

| Opção | Descrição |
|--------|-------------|
| `log_name` | Qual log visualizar: `agent` (padrão), `errors`, `gateway`, ou `list` para mostrar os arquivos disponíveis com seus tamanhos. |
| `-n`, `--lines <N>` | Número de linhas a mostrar (padrão: 50). |
| `-f`, `--follow` | Acompanha o log em tempo real, como `tail -f`. Pressione Ctrl+C para parar. |
| `--level <LEVEL>` | Nível mínimo de log a mostrar: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--session <ID>` | Filtra linhas contendo uma substring de ID de sessão. |
| `--since <TIME>` | Mostra linhas a partir de um tempo relativo: `30m`, `1h`, `2d`, etc. Suporta `s` (segundos), `m` (minutos), `h` (horas), `d` (dias). |
| `--component <NAME>` | Filtra por componente: `gateway`, `agent`, `tools`, `cli`, `cron`. |

### Exemplos {#examples-4}

```bash
# View the last 50 lines of agent.log (default)
hermes logs

# Follow agent.log in real time
hermes logs -f

# View the last 100 lines of gateway.log
hermes logs gateway -n 100

# Show only warnings and errors from the last hour
hermes logs --level WARNING --since 1h

# Filter by a specific session
hermes logs --session abc123

# Follow errors.log, starting from 30 minutes ago
hermes logs errors --since 30m -f

# List all log files with their sizes
hermes logs list
```

### Filtragem {#filtering}

Filtros podem ser combinados. Quando vários filtros estão ativos, uma linha de log precisa passar por **todos** eles para ser exibida:

```bash
# WARNING+ lines from the last 2 hours containing session "tg-12345"
hermes logs --level WARNING --since 2h --session tg-12345
```

Linhas sem um timestamp analisável são incluídas quando `--since` está ativo (podem ser linhas de continuação de uma entrada de log multilinha). Linhas sem um nível detectável são incluídas quando `--level` está ativo.

### Rotação de logs {#log-rotation}

O Hermes usa o `RotatingFileHandler` do Python. Logs antigos são rotacionados automaticamente — procure por `agent.log.1`, `agent.log.2`, etc. O subcomando `hermes logs list` mostra todos os arquivos de log, incluindo os rotacionados.


## `hermes prompt-size`

```bash
hermes prompt-size [--platform <name>] [--json]
```

Reporta o orçamento fixo do prompt para uma sessão nova — o que é enviado em cada
chamada de API *antes* de qualquer conteúdo da conversa. Útil quando um adaptador ou
proxy downstream tem um orçamento de prompt mais restrito que a janela de contexto do modelo, ou quando você
quer ver qual bloco (índice de skills, memória, perfil) domina.

Ele monta o mesmo system prompt que o agente montaria, depois o detalha:

- **Total do system prompt** — o prompt completo montado (identidade, orientações, índice
  de skills, arquivos de contexto, memória, perfil, timestamp).
- **Índice de skills** — o bloco `<available_skills>`. Esse costuma ser o maior
  bloco isolado quando muitas skills estão instaladas.
- **Memória** e **perfil de usuário** — seus snapshots de `MEMORY.md` / `USER.md`.
- **Camadas do prompt** — estável / contexto / volátil, correspondendo a como o Hermes organiza
  o prompt para ser amigável ao cache.
- **Schemas de ferramentas** — o JSON de todas as ferramentas ativadas (a outra metade do
  payload fixo por chamada).

Executa totalmente offline — sem chamada de API, funciona sem nenhuma credencial configurada.

```bash
# Human-readable breakdown for the CLI platform (default)
hermes prompt-size

# Simulate a messaging platform's prompt (different platform hint)
hermes prompt-size --platform telegram

# Machine-readable output for scripts
hermes prompt-size --json
```

:::tip
O índice de skills e os schemas de ferramentas crescem com a quantidade de skills e ferramentas que você tem
ativadas. Para reduzir o prompt, desative toolsets não usados (`hermes tools`) ou
desinstale skills que você não precisa (`hermes skills`). Arquivos de contexto (AGENTS.md,
.cursorrules) no seu diretório atual também contam para o total.
:::

## `hermes config`

```bash
hermes config <subcommand>
```

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `show` | Mostra os valores atuais da config. |
| `edit` | Abre o `config.yaml` no seu editor. |
| `set <key> <value>` | Define um valor de config. |
| `path` | Imprime o caminho do arquivo de config. |
| `env-path` | Imprime o caminho do arquivo `.env`. |
| `check` | Verifica config ausente ou obsoleta. |
| `migrate` | Adiciona interativamente opções recém-introduzidas. |

## `hermes pairing`

```bash
hermes pairing <list|approve|revoke|clear-pending>
```

| Subcomando | Descrição |
|------------|-------------|
| `list` | Mostra usuários pendentes e aprovados. |
| `approve <platform> <code>` | Aprova um código de pareamento. |
| `revoke <platform> <user-id>` | Revoga o acesso de um usuário. |
| `clear-pending` | Limpa os códigos de pareamento pendentes. |

## `hermes skills`

```bash
hermes skills <subcommand>
```

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `browse` | Navegador paginado para registros de skills. |
| `search` | Busca em registros de skills. |
| `install` | Instala uma skill. |
| `inspect` | Visualiza uma skill sem instalá-la. |
| `list` | Lista as skills instaladas. |
| `check` | Verifica skills instaladas do hub por atualizações upstream. |
| `update` | Reinstala skills do hub com alterações upstream disponíveis. |
| `audit` | Reescaneia as skills instaladas do hub. |
| `uninstall` | Remove uma skill instalada via hub. |
| `reset` | Desprende uma skill embutida marcada como `user_modified` limpando sua entrada no manifesto. Com `--restore`, também substitui a cópia do usuário pela versão embutida. |
| `opt-out` | Impede que skills embutidas sejam semeadas no perfil ativo. Grava um marcador `.no-bundled-skills` para que o instalador, `hermes update`, e qualquer sincronização pulem a semeadura de skills embutidas. Seguro por padrão — nada em disco é tocado. Com `--remove`, também exclui skills embutidas já presentes que estão **não modificadas** (skills editadas pelo usuário, instaladas via hub e escritas manualmente nunca são removidas; mostra prévia e confirma antes, `--yes` para pular). |
| `opt-in` | Desfaz `opt-out` removendo o marcador `.no-bundled-skills` para que skills embutidas voltem a ser semeadas no próximo `hermes update`. Com `--sync`, semeia novamente imediatamente. |
| `publish` | Publica uma skill em um registro. |
| `snapshot` | Exporta/importa configurações de skills. |
| `tap` | Gerencia fontes personalizadas de skills. |
| `config` | Configuração interativa de ativação/desativação de skills por plataforma. |

Exemplos comuns:

```bash
hermes skills browse
hermes skills browse --source official
hermes skills search react --source skills-sh
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect official/security/1password
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install official/migration/openclaw-migration
hermes skills install skills-sh/anthropics/skills/pdf --force
hermes skills install https://sharethis.chat/SKILL.md                     # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill        # Override name when frontmatter has none
hermes skills check
hermes skills update
hermes skills config
hermes skills reset google-workspace
hermes skills reset google-workspace --restore --yes
hermes skills opt-out                  # stop future bundled-skill seeding (nothing deleted)
hermes skills opt-out --remove --yes   # also delete UNMODIFIED bundled skills
hermes skills opt-in --sync            # undo: remove marker and re-seed now
```

Observações:
- `--force` pode sobrescrever bloqueios de política não perigosos para skills de terceiros/comunidade.
- `--force` não sobrescreve um veredito de varredura `dangerous`.
- `--source skills-sh` busca no diretório público do `skills.sh`.
- `--source well-known` permite apontar o Hermes para um site que exponha `/.well-known/skills/index.json`.
- `--source browse-sh` busca no catálogo do [browse.sh](https://browse.sh) com mais de 200 skills de automação de navegador específicas por site. Os identificadores parecem com `browse-sh/airbnb.com/search-listings-ddgioa`.
- Passar uma URL `http(s)://…/*.md` instala o `SKILL.md` mais os arquivos explicitamente referenciados em `references/`, `templates/`, `scripts/`, `assets/` e `examples/`. Quando o frontmatter não tem `name:` e o slug da URL não é um identificador válido, um terminal interativo pede um nome; superfícies não interativas (`/skills install` dentro da TUI, plataformas do gateway) exigem `--name <x>` em vez disso.

## `hermes bundles`

```bash
hermes bundles <subcommand>
```

Bundles de skills agrupam várias skills sob um único slash command `/<bundle-name>`. Invocar o bundle carrega toda skill referenciada em uma única mensagem de usuário combinada. Armazenamento: `~/.hermes/skill-bundles/<slug>.yaml`. Veja [Skill Bundles](../user-guide/features/skills.md#skill-bundles) para o schema YAML e o comportamento.

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `list` | Lista os bundles instalados (padrão quando nenhum subcomando é dado) |
| `show <name>` | Mostra o nome, a descrição, as skills e o caminho do arquivo de um bundle |
| `create <name>` | Cria um novo bundle. Passe `--skill <id>` (repetível) ou omita para entrada interativa. `--description`, `--instruction`, `--force` disponíveis. |
| `delete <name>` | Remove um arquivo de bundle |
| `reload` | Reescaneia `~/.hermes/skill-bundles/` e reporta bundles adicionados/removidos |

Exemplos:

```bash
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work"

hermes bundles list
hermes bundles show backend-dev
hermes bundles delete backend-dev
```

Em uma sessão de chat, `/bundles` lista os bundles instalados e `/<bundle-name>` carrega um deles.

## `hermes curator`

```bash
hermes curator <subcommand>
```

O curator é uma tarefa em segundo plano baseada em modelo auxiliar que revisa periodicamente skills criadas pelo agente, poda as obsoletas, consolida sobreposições e arquiva skills descontinuadas. Skills embutidas e instaladas via hub nunca são tocadas. Arquivos são recuperáveis; a exclusão automática nunca ocorre.

| Subcomando | Descrição |
|------------|-------------|
| `status` | Mostra o status do curator e estatísticas das skills |
| `run` | Dispara uma revisão do curator agora (bloqueia até a passagem do LLM terminar) |
| `run --background` | Inicia a passagem do LLM em uma thread em segundo plano e retorna imediatamente |
| `run --dry-run` | Apenas prévia — produz o relatório de revisão sem mutações |
| `backup` | Faz um snapshot tar.gz manual de `~/.hermes/skills/` (o curator também faz snapshot automaticamente antes de toda execução real) |
| `rollback` | Restaura `~/.hermes/skills/` a partir de um snapshot (padrão: o mais recente) |
| `rollback --list` | Lista os snapshots disponíveis |
| `rollback --id <ts>` | Restaura um snapshot específico por id |
| `rollback -y` | Pula o prompt de confirmação |
| `pause` | Pausa o curator até ser retomado |
| `resume` | Retoma um curator pausado |
| `pin <skill>` | Fixa uma skill para que o curator nunca a transicione automaticamente |
| `unpin <skill>` | Remove a fixação de uma skill |
| `restore <skill>` | Restaura uma skill arquivada |
| `archive <skill>` | Arquiva uma skill manualmente |
| `prune` | Poda manualmente skills que o curator normalmente limparia |
| `list-archived` | Lista as skills arquivadas (recuperáveis via `restore`) |

Em uma instalação nova, a primeira passagem agendada é postergada por um `interval_hours` completo (7 dias por padrão) — o gateway não fará curadoria imediatamente no primeiro tick após `hermes update`. Use `hermes curator run --dry-run` para uma prévia antes disso acontecer.

Veja [Curator](../user-guide/features/curator.md) para comportamento e configuração.

## `hermes moa`

Configura presets nomeados de Mixture of Agents. Presets aparecem como modelos selecionáveis sob um provedor `Mixture of Agents` em todo seletor de modelos; `/moa <prompt>` executa um prompt através do preset padrão.

```bash
hermes moa list
hermes moa configure [name]
hermes moa delete <name>
```

`hermes moa configure` reutiliza o seletor de provedor → modelo do Hermes para cada modelo de referência e o agregador. Um preset é uma configuração de modo de execução, não um modelo ou provedor principal.

## `hermes fallback`

```bash
hermes fallback <subcommand>
```

Gerencia a cadeia de provedores de fallback. Provedores de fallback são tentados em ordem quando o modelo principal falha com erros de limite de taxa, sobrecarga ou conexão.

| Subcomando | Descrição |
|------------|-------------|
| `list` (alias: `ls`) | Mostra a cadeia de fallback atual (padrão quando nenhum subcomando é dado) |
| `add` | Escolhe um provedor + modelo (mesmo seletor de `hermes model`) e adiciona à cadeia |
| `remove` (alias: `rm`) | Escolhe uma entrada para excluir da cadeia |
| `clear` | Remove todas as entradas de fallback |

Veja [Provedores de Fallback](../user-guide/features/fallback-providers.md).

## `hermes hooks`

```bash
hermes hooks <subcommand>
```

Inspeciona hooks de script de shell declarados em `~/.hermes/config.yaml`, testa-os contra payloads sintéticos, e gerencia a lista de consentimento de primeiro uso em `~/.hermes/shell-hooks-allowlist.json`.

| Subcomando | Descrição |
|------------|-------------|
| `list` (alias: `ls`) | Lista os hooks configurados com matcher, timeout e status de consentimento |
| `test <event>` | Dispara todo hook que corresponda a `<event>` contra um payload sintético |
| `revoke` (aliases: `remove`, `rm`) | Remove as entradas de allowlist de um comando (aplicado no próximo reinício) |
| `doctor` | Verifica cada hook configurado: bit de execução, allowlist, deriva de mtime, validade do JSON e tempo de execução sintética |

Veja [Hooks](../user-guide/features/hooks.md) para as assinaturas de eventos e formatos de payload.

## `hermes memory`

```bash
hermes memory <subcommand>
```

Configura e gerencia plugins de provedor de memória externo. Provedores disponíveis: honcho, openviking, mem0, hindsight, holographic, retaindb, byterover, supermemory. Apenas um provedor externo pode estar ativo por vez. A memória embutida (MEMORY.md/USER.md) está sempre ativa.

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `setup` | Seleção e configuração interativa de provedor. |
| `status` | Mostra a configuração do provedor de memória atual. |
| `off` | Desativa o provedor externo (apenas embutido). |

:::info Subcomandos específicos de provedor
Quando um provedor de memória externo está ativo, ele pode registrar seu próprio comando de nível superior `hermes <provider>` para gerenciamento específico do provedor (ex.: `hermes honcho` quando o Honcho está ativo). Provedores inativos não expõem seus subcomandos. Execute `hermes --help` para ver o que está atualmente conectado.
:::

## `hermes acp`

```bash
hermes acp
```

Inicia o Hermes como um servidor stdio ACP (Agent Client Protocol) para integração com editores.

Pontos de entrada relacionados:

```bash
hermes-acp
python -m acp_adapter
```

Instale o suporte primeiro:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

Veja [Integração ACP com Editores](../user-guide/features/acp.md) e [Internals do ACP](../developer-guide/acp-internals.md).

## `hermes mcp`

```bash
hermes mcp <subcommand>
```

Gerencia configurações de servidores MCP (Model Context Protocol) e executa o Hermes como servidor MCP.

| Subcomando | Descrição |
|------------|-------------|
| *(nenhum)* ou `picker` | Seletor interativo de catálogo — navega pelos MCPs aprovados pela Nous e instala/ativa/desativa. |
| `catalog` | Lista os MCPs aprovados pela Nous (texto simples, scriptável). |
| `install <name>` | Instala uma entrada do catálogo (ex.: `hermes mcp install n8n`). |
| `serve [-v\|--verbose]` | Executa o Hermes como servidor MCP — expõe conversas para outros agentes. |
| `add <name> [--url URL] [--command CMD] [--auth oauth\|header] [--args ...]` | Adiciona um servidor MCP personalizado com descoberta automática de ferramentas. `--args` passa o restante do argv para o comando stdio, então coloque-o por último. |
| `remove <name>` (alias: `rm`) | Remove um servidor MCP da config. |
| `list` (alias: `ls`) | Lista os servidores MCP configurados. |
| `test <name>` | Testa a conexão com um servidor MCP. |
| `configure <name>` (alias: `config`) | Alterna a seleção de ferramentas de um servidor. |
| `login <name>` | Força a reautenticação de um servidor MCP baseado em OAuth. |

Veja [Referência de Configuração MCP](./mcp-config-reference.md), [Usar MCP com o Hermes](../guides/use-mcp-with-hermes.md) e [Modo Servidor MCP](../user-guide/features/mcp.md#running-hermes-as-an-mcp-server).

## `hermes plugins`

```bash
hermes plugins [subcommand]
```

Gerenciamento unificado de plugins — plugins gerais, provedores de memória e mecanismos de contexto em um só lugar. Executar `hermes plugins` sem subcomando abre uma tela composta interativa com duas seções:

- **Plugins Gerais** — caixas de seleção múltipla para ativar/desativar plugins instalados
- **Plugins de Provedor** — configuração de seleção única para Provedor de Memória e Mecanismo de Contexto. Pressione ENTER em uma categoria para abrir um seletor de opção única.

| Subcomando | Descrição |
|------------|-------------|
| *(nenhum)* | UI interativa composta — alternâncias de plugins gerais + configuração de plugins de provedor. |
| `install <identifier> [--force]` | Instala um plugin de uma URL Git ou `owner/repo`. |
| `update <name>` | Baixa as últimas alterações de um plugin instalado. |
| `remove <name>` (aliases: `rm`, `uninstall`) | Remove um plugin instalado. |
| `enable <name>` | Ativa um plugin desativado. |
| `disable <name>` | Desativa um plugin sem removê-lo. |
| `list` (alias: `ls`) | Lista os plugins instalados com status ativado/desativado. |

As seleções de plugins de provedor são salvas no `config.yaml`:
- `memory.provider` — provedor de memória ativo (vazio = apenas embutido)
- `context.engine` — mecanismo de contexto ativo (`"compressor"` = padrão embutido)

A lista de plugins gerais desativados é armazenada no `config.yaml` em `plugins.disabled`.

Veja [Plugins](../user-guide/features/plugins.md) e [Construir um Plugin do Hermes](../developer-guide/plugins/index.md).

## `hermes tools`

```bash
hermes tools [--summary]
```

| Opção | Descrição |
|--------|-------------|
| `--summary` | Imprime o resumo atual de ferramentas ativadas e encerra. |

Sem `--summary`, isso inicia a UI interativa de configuração de ferramentas por plataforma.

## `hermes computer-use`

```bash
hermes computer-use <subcommand>
```

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `install` | Executa o instalador upstream do cua-driver (macOS, Windows e Linux). |
| `install --upgrade` | Reexecuta o instalador mesmo que o cua-driver já esteja no PATH. O script upstream sempre baixa o lançamento mais recente, então isso executa uma atualização no local. |
| `status` | Imprime se o `cua-driver` está no `$PATH` e qual versão está instalada. |

`hermes computer-use install` é o ponto de entrada estável para instalar o binário
[cua-driver](https://github.com/trycua/cua) usado pelo toolset
`computer_use`. Ele executa o mesmo instalador upstream que
`hermes tools` invoca quando você ativa o Computer Use pela primeira vez, então é seguro
usá-lo para reexecutar a instalação se a alternância do toolset não a disparou
(por exemplo, em configurações de usuário recorrente).

`hermes update` reexecuta automaticamente o instalador upstream ao final
da atualização se o cua-driver estiver no PATH, então a maioria dos usuários não precisará
chamar `--upgrade` manualmente. Use-o quando o upstream lançar uma correção que você quer
agora, sem esperar pela próxima atualização do Hermes.

## `hermes pets`

```bash
hermes pets <list|install|select|show|off|scale|remove|doctor>
```

[Petdex](https://github.com/crafter-station/petdex) é uma galeria pública de pets animados em sprite para agentes de codificação. Instale um e o Hermes o exibe reagindo à atividade do agente na CLI, TUI e app desktop.

| Subcomando | Descrição |
|------------|-------------|
| `list` | Navega pela galeria do petdex. |
| `install` | Instala um pet da galeria. |
| `select` | Define o pet ativo (grava `display.pet.*`). |
| `show` | Anima o pet ativo no terminal. |
| `off` | Desativa a exibição do pet. |
| `scale` | Redimensiona o pet em todos os lugares (`display.pet.scale`). |
| `remove` | Exclui um pet instalado. |
| `doctor` | Verifica a configuração do pet + o suporte gráfico do terminal. |

Você também pode gerar um pet totalmente novo a partir de uma descrição em texto com o slash command `/hatch`. Veja [Pets](../user-guide/features/pets.md).

## `hermes sessions`

```bash
hermes sessions <subcommand>
```

Subcomandos:

| Subcomando | Descrição |
|------------|-------------|
| `list` | Lista as sessões recentes. |
| `browse` | Seletor interativo de sessões com busca e retomada. |
| `export <output> [--session-id ID]` | Exporta sessões para JSONL. |
| `delete <session-id>` | Exclui uma sessão. |
| `prune` | Exclui sessões que correspondam aos filtros: limites de tempo `--older-than`/`--newer-than`/`--before`/`--after` (durações como `5h`/`2d`, dias simples, ou timestamps ISO); atributos `--source`, `--title`, `--model`, `--provider`, `--branch`, `--end-reason`, `--user`, `--chat-id`, `--chat-type`, `--cwd`; limites numéricos `--min/--max-messages`, `--min/--max-tokens`, `--min/--max-cost`, `--min/--max-tool-calls`; além de `--include-archived`, `--dry-run`, `--yes`. Padrão: mais antigas que 90 dias. |
| `archive` | Arquiva em massa (oculta sem excluir) sessões que correspondam aos mesmos filtros de `prune`. Requer pelo menos um filtro. |
| `stats` | Mostra estatísticas do armazenamento de sessões. |
| `rename <session-id> <title>` | Define ou altera o título de uma sessão. |

## `hermes insights`

```bash
hermes insights [--days N] [--source platform]
```

| Opção | Descrição |
|--------|-------------|
| `--days <n>` | Analisa os últimos `n` dias (padrão: 30). |
| `--source <platform>` | Filtra por origem, como `cli`, `telegram`, ou `discord`. |

## `hermes claw`

```bash
hermes claw migrate [options]
```

Migra sua configuração do OpenClaw para o Hermes. Lê de `~/.openclaw` (ou um caminho personalizado) e grava em `~/.hermes`. Detecta automaticamente nomes de diretório legados (`~/.clawdbot`, `~/.moltbot`) e nomes de arquivo de config (`clawdbot.json`, `moltbot.json`).

| Opção | Descrição |
|--------|-------------|
| `--dry-run` | Mostra uma prévia do que seria migrado sem gravar nada. |
| `--preset <name>` | Preset de migração: `full` (todas as configurações compatíveis) ou `user-data` (exclui configuração de infraestrutura). Nenhum dos presets importa segredos — passe `--migrate-secrets` explicitamente. |
| `--overwrite` | Sobrescreve arquivos existentes do Hermes em caso de conflito (padrão: recusa aplicar quando o plano tem conflitos). |
| `--migrate-secrets` | Inclui chaves de API na migração. Necessário mesmo com `--preset full`. |
| `--no-backup` | Pula o snapshot zip de pré-migração de `~/.hermes/` (por padrão, um único arquivo de ponto de restauração é gravado em `~/.hermes/backups/pre-migration-*.zip` antes de aplicar; restaurável com `hermes import`). |
| `--source <path>` | Diretório personalizado do OpenClaw (padrão: `~/.openclaw`). |
| `--workspace-target <path>` | Diretório de destino para instruções de workspace (AGENTS.md). |
| `--skill-conflict <mode>` | Trata colisões de nomes de skills: `skip` (padrão), `overwrite`, ou `rename`. |
| `--yes` | Pula o prompt de confirmação. |

### O que é migrado {#what-gets-migrated}

A migração cobre mais de 30 categorias entre persona, memória, skills, provedores de modelo, plataformas de mensagens, comportamento do agente, políticas de sessão, servidores MCP, TTS e mais. Os itens são **importados diretamente** para equivalentes do Hermes ou **arquivados** para revisão manual.

**Importados diretamente:** SOUL.md, MEMORY.md, USER.md, AGENTS.md, skills (4 diretórios de origem), modelo padrão, provedores personalizados, servidores MCP, tokens e listas de permissão de plataformas de mensagens (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost), padrões do agente (esforço de raciocínio, compressão, delay humano, timezone, sandbox), políticas de reinício de sessão, regras de aprovação, configuração de TTS, configurações de navegador, configurações de ferramentas, timeout de execução, lista de permissão de comandos, configuração de gateway, e chaves de API de 3 fontes.

**Arquivados para revisão manual:** Jobs de cron, plugins, hooks/webhooks, backend de memória (QMD), configuração de registro de skills, UI/identidade, logging, configuração multi-agente, vínculos de canal, IDENTITY.md, TOOLS.md, HEARTBEAT.md, BOOTSTRAP.md.

**A resolução de chaves de API** verifica três fontes em ordem de prioridade: valores de config → `~/.openclaw/.env` → `auth-profiles.json`. Todos os campos de token lidam com strings simples, templates de ambiente (`${VAR}`), e objetos SecretRef.

Para o mapeamento completo de chaves de config, detalhes do tratamento de SecretRef, e checklist pós-migração, veja o **[guia de migração completo](../guides/migrate-from-openclaw.md)**.

### Exemplos {#examples-5}

```bash
# Preview what would be migrated
hermes claw migrate --dry-run

# Full migration (all compatible settings, no secrets)
hermes claw migrate --preset full

# Full migration including API keys
hermes claw migrate --preset full --migrate-secrets

# Migrate user data only (no secrets), overwrite conflicts
hermes claw migrate --preset user-data --overwrite

# Migrate from a custom OpenClaw path
hermes claw migrate --source /home/user/old-openclaw
```

## `hermes serve`

```bash
hermes serve [options]
```

Inicia o **servidor backend** do Hermes — o gateway JSON-RPC/WebSocket ao qual o [app desktop](/user-guide/desktop) e clientes remotos se conectam. É o mesmo servidor que `hermes dashboard` executa, mas **sem interface**: nunca abre uma UI no navegador. O app desktop inicia seu próprio backend `hermes serve`; use este comando diretamente quando quiser um backend sem interface em um host remoto. Aceita as mesmas opções `--host` / `--port` / `--insecure` / `--skip-build` / `--stop` / `--status` de `hermes dashboard` abaixo (um bind não-loopback ativa o mesmo portão de autenticação). Requer o extra `[web]`; o socket de Chat embutido também precisa de `[pty]` em um host POSIX.

## `hermes dashboard`

```bash
hermes dashboard [options]
```

Inicia o dashboard web — uma UI baseada em navegador para gerenciar configuração, chaves de API, e monitorar sessões. (Para um backend sem interface e sem UI de navegador — por exemplo, o que o app desktop inicia — use [`hermes serve`](#hermes-serve) acima.) Requer `cd ~/.hermes/hermes-agent && uv pip install -e ".[web]"` (FastAPI + Uvicorn). A aba de Chat embutida no navegador está sempre disponível e adicionalmente precisa do extra `pty` (`cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"`) além de um ambiente PTY POSIX como Linux, macOS, ou WSL2. Veja [Dashboard Web](/user-guide/features/web-dashboard) para a documentação completa.

| Opção | Padrão | Descrição |
|--------|---------|-------------|
| `--port` | `9119` | Porta para executar o servidor web |
| `--host` | `127.0.0.1` | Endereço de bind |
| `--no-open` | — | Não abre o navegador automaticamente |
| `--insecure` | desativado | **Descontinuado / sem efeito.** Antes ignorava a autenticação em um bind não-loopback. Desde o hardening de junho de 2026, um bind público *sempre* requer um provedor de autenticação (senha ou OAuth). Vincule a `127.0.0.1` e use um túnel para mantê-lo local. |
| `--skip-build` | desativado | Pula a etapa de build da UI web e serve o `dist` existente diretamente. Útil para contextos não interativos (Tarefas Agendadas do Windows, CI) onde o npm não está disponível. Pré-compile com `cd web && npm run build`. |
| `--isolated` | desativado | Quando iniciado a partir de um perfil nomeado (`worker dashboard`), executa um servidor dedicado por perfil em vez de rotear para o dashboard da máquina. |
| `--stop` | — | Para os processos `hermes dashboard` em execução e encerra. |
| `--status` | — | Lista os processos `hermes dashboard` em execução e encerra. |

### `hermes dashboard register` {#hermes-dashboard-register}

Registra esta instalação como um dashboard auto-hospedado na sua conta do Nous Portal. Cria um cliente OAuth, grava `HERMES_DASHBOARD_OAUTH_CLIENT_ID` em `~/.hermes/.env`, e imprime como acionar o portão de login. Requer estar logado (`hermes setup`).

| Opção | Descrição |
|--------|-------------|
| `--name` | Rótulo legível por humanos para o dashboard (padrão: gerado automaticamente). |
| `--redirect-uri` | URI de redirecionamento OAuth HTTPS pública (ex.: `https://hermes.example.com/auth/callback`). Omita para uso apenas em localhost. |
| `--portal-url` | Sobrescreve a URL base do Nous Portal para o registro (padrão: o portal em que você fez login). Também configurável via `HERMES_DASHBOARD_PORTAL_URL`. |

```bash
# Default — opens browser to http://127.0.0.1:9119
hermes dashboard

# Custom port, no browser
hermes dashboard --port 8080 --no-open

# From a profile alias — routes to the machine dashboard with the
# profile preselected in the sidebar switcher (attach if running)
worker dashboard
```

## `hermes profile`

```bash
hermes profile <subcommand>
```

Gerencia perfis — múltiplas instâncias isoladas do Hermes, cada uma com sua própria config, sessões, skills e diretório home.

| Subcomando | Descrição |
|------------|-------------|
| `list` | Lista todos os perfis. |
| `use <name>` | Define um perfil padrão fixo. |
| `create <name> [--clone] [--clone-all] [--clone-from <source>] [--no-alias]` | Cria um novo perfil. `--clone` copia config, `.env`, `SOUL.md`, e skills do perfil ativo. `--clone-all` copia todo o estado. `--clone-from` especifica um perfil de origem e implica clonagem de config a menos que combinado com `--clone-all`. |
| `delete <name> [-y]` | Exclui um perfil. |
| `show <name>` | Mostra os detalhes de um perfil (diretório home, config, etc.). |
| `alias <name> [--remove] [--name NAME]` | Gerencia scripts wrapper para acesso rápido ao perfil. |
| `rename <old> <new>` | Renomeia um perfil. |
| `export <name> [-o FILE]` | Exporta um perfil para um arquivo `.tar.gz` (backup local). |
| `import <archive> [--name NAME]` | Importa um perfil de um arquivo `.tar.gz` (restauração local). |
| `install <source> [--name N] [--alias] [--force] [-y]` | Instala uma distribuição de perfil de uma URL git ou diretório local. |
| `update <name> [--force-config] [-y]` | Rebusca uma distribuição; preserva dados do usuário (memórias, sessões, autenticação). |
| `info <name>` | Mostra o manifesto de distribuição de um perfil (versão, requisitos, origem). |

Exemplos:

```bash
hermes profile list
hermes profile create work --clone
hermes profile use work
hermes profile alias work --name h-work
hermes profile export work -o work-backup.tar.gz
hermes profile import work-backup.tar.gz --name restored
hermes profile install github.com/user/my-distro --alias
hermes profile update work
hermes -p work chat -q "Hello from work profile"
```

## `hermes completion`

```bash
hermes completion [bash|zsh|fish]
```

Imprime um script de autocompletar do shell na stdout. Carregue a saída no perfil do seu shell para o autocompletar via tab dos comandos, subcomandos e nomes de perfis do Hermes.

Exemplos:

```bash
# Bash
hermes completion bash >> ~/.bashrc

# Zsh
hermes completion zsh >> ~/.zshrc

# Fish
hermes completion fish > ~/.config/fish/completions/hermes.fish
```

## `hermes update`

```bash
hermes update [--gateway] [--check] [--no-backup] [--backup] [--yes]
```

Baixa o código mais recente do `hermes-agent` e reinstala as dependências no venv gerenciado, depois reexecuta os hooks de pós-instalação (servidores MCP, sincronização de skills, instalação de autocompletar). Seguro para executar em uma instalação ativa. Use `--check` para ver se seu checkout está atrasado em relação a `origin/main` sem instalar.

`hermes update` baixa a branch de atualização configurada (padrão: `main`). Se seu checkout está em outra branch, o Hermes pode fazer checkout da branch de atualização antes do pull. Faça commit do trabalho na branch antes de atualizar quando quiser mantê-lo fora do fluxo de autostash da atualização.

| Opção | Descrição |
|--------|-------------|
| `--gateway` | Modo interno usado pelo comando `/update` de mensagens. Usa IPC baseado em arquivo para prompts e streaming de progresso em vez de ler da stdin do terminal. Não é uma flag de reinício do gateway. |
| `--check` | Verifica se há atualização disponível sem baixar, instalar dependências, ou reiniciar nada. |
| `--no-backup` | Pula todos os backups pré-atualização nesta execução (tanto o snapshot rápido de estado quanto o zip completo), independentemente de `updates.pre_update_backup`. |
| `--backup` | Força um backup pré-atualização **completo** nesta execução: o snapshot rápido de estado mais um zip completo do `HERMES_HOME` (config, autenticação, sessões, skills, dados de pareamento). O modo padrão é `quick` — apenas um snapshot leve de estado. Defina o modo permanente via `updates.pre_update_backup: quick | full | off` no `config.yaml`. |
| `--yes`, `-y` | Assume "sim" para prompts interativos como migração de config e restauração de stash. A entrada de chaves de API é pulada; execute `hermes config migrate` separadamente para isso. |

Comportamento adicional:

- **Reinício do gateway.** Após uma atualização bem-sucedida, o Hermes tenta reiniciar automaticamente todos os perfis de gateway em execução para que incorporem o novo código. Use `hermes gateway restart` quando quiser reiniciar um gateway sem aplicar uma atualização.
- **Alterações de código local.** Para instalações via git, arquivos rastreados sujos e arquivos não rastreados são automaticamente colocados em stash antes do checkout de branch ou pull (`git stash push --include-untracked`). Atualizações em terminal interativo perguntam antes de restaurar o stash. Atualizações não interativas o restauram por padrão; defina `updates.non_interactive_local_changes: discard` apenas em instalações gerenciadas onde edições locais de código devem ser descartadas após um pull bem-sucedido. Se a restauração do stash tiver conflitos ou o pull falhar, o stash é deixado no lugar para recuperação manual.
- **Instabilidade do lockfile do npm.** Antes de colocar em stash ou trocar de branch, o Hermes faz uma limpeza best-effort das diferenças rastreadas de `package-lock.json` produzidas por etapas de npm install/build. Faça commit ou coloque manualmente em stash edições intencionais de lockfile antes de executar `hermes update`.
- **Snapshot de dados de pareamento.** Mesmo quando `--backup` está desativado, `hermes update` faz um snapshot leve de `~/.hermes/pairing/` e das regras de comentário do Feishu antes do `git pull`. Você pode revertê-lo com `hermes backup restore --state pre-update` se um pull reescrever um arquivo que você estava editando.
- **Aviso de `hermes.service` legado.** Se o Hermes detectar uma unidade systemd `hermes.service` anterior à renomeação (em vez do atual `hermes-gateway.service`), ele imprime uma dica de migração única para você evitar problemas de flap-loop.
- **Códigos de saída.** `0` em sucesso, `1` em erros de pull/instalação/pós-instalação, `2` em alterações inesperadas na árvore de trabalho que bloqueiam o `git pull`.

## Comandos de manutenção {#maintenance-commands}

| Comando | Descrição |
|---------|-------------|
| `hermes version` | Imprime informações de versão. |
| `hermes update` | Baixa as últimas alterações e reinstala as dependências. |
| `hermes postinstall` | Bootstrap interno. Executa uma vez após o script de instalação provisionar o Hermes (ou após `hermes update`) para instalar dependências não-Python que o pip não pode fornecer — runtime Node.js, navegador headless, ripgrep, ffmpeg — e então dispara o `hermes setup` se o perfil ainda não tiver sido configurado. Seguro para reexecutar idempotentemente. |
| `hermes uninstall [--full] [--gui] [--yes]` | Remove o Hermes, opcionalmente excluindo toda a config/dados. `--gui` remove apenas a GUI de Chat desktop, deixando o agente intacto; `--full` também exclui config/dados; `--yes` pula os prompts. |

## Veja também {#see-also}

- [Referência de Slash Commands](./slash-commands.md)
- [Interface da CLI](../user-guide/cli.md)
- [Sessões](../user-guide/sessions.md)
- [Sistema de Skills](../user-guide/features/skills.md)
- [Skins e Temas](../user-guide/features/skins.md)
