---
sidebar_position: 3
title: "Configurando models"
description: "Como configurar o model principal e models auxiliares do Hermes Agent"
---

# Configurando models

O Hermes usa dois tipos de slots de model:

- **Model principal** — com o que o agente pensa. Toda mensagem user, todo loop de tool-call, toda resposta streamed passa por este model.
- **Models auxiliares** — side-jobs menores que o agente offloada. Compressão de contexto, visão (análise de imagem), resumo de página web, scoring de aprovação, roteamento de ferramentas MCP, geração de título de sessão e busca de skills. Cada um tem seu próprio slot e pode ser overridden independentemente.

Esta página cobre configurar ambos pelo dashboard. Se preferir arquivos de config ou CLI, vá para [Métodos alternativos](#alternative-methods) no final.

:::tip Caminho mais rápido: Nous Portal
O [Nous Portal](/user-guide/features/tool-gateway) oferece 300+ models em uma assinatura. Em instalação fresh, rode `hermes setup --portal` para logar e definir Nous como provider em um comando. Inspecione o que está wired com `hermes portal info`.

- Assinantes Portal também ganham **10% off em providers cobrados por token**.
:::

:::note Schema `model:` — string vazia vs. mapping
Em instalação brand-new o config default bundled tem `model: ""` (sentinel de string vazia significando "ainda não configurado"). Na primeira vez que você roda `hermes setup` ou `hermes model`, essa key é upgraded in-place para um mapping com sub-keys `provider`, `default`, `base_url` e `api_mode` — a forma mostrada nesta página e em [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md). Se você vir string vazia em `config.yaml`, rode `hermes model` (ou clique **Change** no dashboard) e o Hermes escreverá a forma dict para você.
:::

## A página Models {#the-models-page}

Abra o dashboard e clique **Models** na sidebar. Você tem duas seções:

1. **Model Settings** — painel superior, onde você atribui models aos slots.
2. **Usage analytics** — cards ranqueados mostrando todo model que rodou sessão no período selecionado, com contagens de tokens, custo e badges de capability.

![Visão geral da página Models](/img/docs/dashboard-models/overview.png)

O card superior é o painel **Model Settings**. A linha principal sempre mostra o que o agente vai subir para novas sessões. Clique **Change** para abrir o picker.

## Definindo o model principal {#setting-the-main-model}

Clique **Change** na linha Main model:

![Diálogo do model picker](/img/docs/dashboard-models/picker-dialog.png)

O picker tem duas colunas:

- **Esquerda** — providers autenticados. Só providers que você configurou (API key set, OAuth feito, ou definido como endpoint custom) aparecem. Se um provider faltar, vá em **Keys** e adicione credencial.
- **Direita** — lista curada de models do provider selecionado. São os models agentic que o Hermes recomenda para aquele provider, não o dump cru de `/models` (que no OpenRouter inclui 400+ models incluindo TTS, geradores de imagem e rerankers).

Digite na caixa de filtro para estreitar por nome de provider, slug ou model ID.

Escolha um model, clique **Switch**, e o Hermes grava em `~/.hermes/config.yaml` na seção `model`. **Isso aplica só a novas sessões** — qualquer aba de chat já aberta continua rodando o model com que começou. Para hot-swap o chat atual, use slash command `/model` dentro dele.

### Trocas mid-session e avisos de contexto {#mid-session-switches-and-context-warnings}

Quando você troca models **dentro de sessão ativa** (model picker Herm TUI, CLI `hermes` ou `/model` no Telegram/Discord), o Hermes estima se sua **próxima mensagem** vai rodar **compressão de contexto preflight** contra a janela do model novo. Se a sessão já está perto ou acima do threshold de compressão daquele model (veja [Context Compression](./configuration.md#context-compression)), a resposta da troca inclui aviso — o mesmo caminho `warning_message` usado para avisos de model caro. A troca ainda aplica imediatamente; compressão roda na **primeira mensagem user após a troca**, antes do model responder.

:::warning Trocas mid-session resetam o prompt cache
Prompt caches são keyed ao model servindo a requisição, então qualquer mudança de model mid-conversation — troca explícita `/model`, [fallback automático](./features/fallback-providers.md), ou rotação de [credential-pool](./features/credential-pools.md) para conta diferente — significa que a próxima mensagem relê a conversa inteira a preço full de input-token em vez da taxa cached (~75–90% desconto). Em sessão longa esse re-read one-time pode eclipsar a diferença per-token entre os dois models. Troque quando precisar, mas prefira cedo na conversa ou logo após iniciar sessão fresh.
:::

### Tiers de data-training desassistidos {#unattended-data-training-tiers}

Modelos como `muse-spark-1.2-contributor` são com desconto porque o vendor pode treinar nos seus prompts e completions. Seleção interativa de modelo sempre mostra um prompt de confirmação. Caminhos de startup não interativos como workers Kanban e agentes cron falham fechado porque não podem fazer aquela pergunta.

Se treinar nos dados da carga desassistida for aceitável, registre um acknowledgement persistente:

```bash
hermes config set security.allow_data_training_tiers_noninteractive true
```

O Hermes ainda imprime o aviso completo de política de dados e a chave de acknowledgement em cada startup desassistido, então logs de worker retêm uma trilha de auditoria. Esta setting não aprova avisos de modelo caro ou de roteamento de provider, e não substitui o prompt de confirmação interativo. Revoque com `hermes config unset security.allow_data_training_tiers_noninteractive`.

## Definindo models auxiliares {#setting-auxiliary-models}

Clique **Show auxiliary** para revelar os 11 task slots:

![Painel auxiliary expandido](/img/docs/dashboard-models/auxiliary-expanded.png)

Toda tarefa auxiliar default para `auto` — o Hermes tenta seu model principal para aquele job também. Se essa rota estiver indisponível ou bater falha estilo capacity, `auto` segue qualquer `auxiliary.<task>.fallback_chain` específico da tarefa, depois a cadeia principal `fallback_providers` / `fallback_model`, depois a cadeia built-in de discovery auxiliary do Hermes. Override uma tarefa específica quando quiser model mais barato ou rápido para side-job.

### Padrões comuns de override {#common-override-patterns}

| Tarefa | Quando fazer override |
|---|---|
| **Title Gen** | Quando latência ou custo do título importa mais do que casar com o model principal. Pinar um model flash conhecido como bom, ou defina `auxiliary.title_generation.prefer_fast_model: true` para o Hermes escolher o tier rápido do provider. |
| **Vision** | Quando seu model principal não tem suporte a visão. Aponte para `google/gemini-2.5-flash` ou `gpt-4o-mini`. |
| **Compression** | Quando você queima reasoning tokens em Opus/M2.7 só para resumir contexto. Um model chat rápido faz o job a 1/50 do custo. |
| **Approval** | Para `approval_mode: smart` — model rápido/barato (haiku, flash, gpt-5-mini) decide se auto-aprova comandos low-risk. Models caros aqui são desperdício. |
| **Web Extract** | Quando usa `web_extract` pesado. Mesma lógica de compression — resumo não precisa de reasoning. |
| **Skills Hub** | `hermes skills search` usa isto. Geralmente ok em `auto`. |
| **MCP** | Roteamento de ferramentas MCP. Geralmente ok em `auto`. |
| **Triage Specifier** | Roteia o triage specifier Kanban (`hermes kanban specify`) que expande one-liner rough em spec concreta. Model barato e capaz funciona bem. |
| **Kanban Decomposer** | Roteia decomposição de tarefas Kanban — divide tarefa de triage em grafo de child tasks para profiles specialist. |
| **Profile Describer** | Roteia geração de descrição de profile (`hermes profile describe --auto` / botão auto-generate do dashboard). Chamada curta e barata. |
| **Curator** | Roteia pass de review de skill-usage do curator. Pode rodar minutos em models de reasoning, então model aux barato costuma valer a pena. |

### Override por tarefa {#per-task-override}

Clique **Change** em qualquer linha auxiliary. Mesmo picker abre, mesmo comportamento — escolha provider + model, clique Switch. A linha atualiza para mostrar `provider · model` em vez de `auto (use main model)`.

### Resetar tudo para auto {#reset-all-to-auto}

Se over-tuned e quiser recomeçar, clique **Reset all to auto** no topo da seção auxiliary. Todo slot volta a usar seu model principal.

## Atalho "Use as" {#the-use-as-shortcut}

Todo card de model na página tem dropdown **Use as**. Este é o caminho rápido — escolha model que vê nos analytics, clique **Use as**, e atribua ao slot principal ou qualquer task auxiliary específica em um clique:

![Dropdown Use as](/img/docs/dashboard-models/use-as-dropdown.png)

O dropdown tem:

- **Main model** — igual a clicar Change na linha principal.
- **All auxiliary tasks** — atribui este model a todos os 11 aux slots de uma vez. Útil quando quer todo side-job num model flash barato.
- **Individual task options** — Vision, Web Extract, Compression, etc. O model atualmente atribuído a cada tarefa está marcado `current`.

Cards são badged com `main` ou `aux · <task>` quando estão atribuídos a algo — para ver de relance quais models históricos estão wired onde.

## O que é gravado em `config.yaml` {#what-gets-written-to-configyaml}

Quando você salva via dashboard, o Hermes grava em `~/.hermes/config.yaml`:

**Model principal:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # limpo na troca de provider
  api_mode: chat_completions
```

**Override auxiliary (exemplo — vision em gemini-flash):**
```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
```

**Auxiliary em auto (padrão):**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... outros campos inalterados
```

`provider: auto` com `model: ''` diz ao Hermes para usar o model principal naquela tarefa, ainda honrando fallback policy se a rota principal não puder servir a chamada auxiliary.

Fallback chains específicas de tarefa ficam sob a mesma tarefa auxiliary:

```yaml
auxiliary:
  title_generation:
    provider: auto
    model: ''
    fallback_chain:
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free
```

Quando `fallback_chain` está ausente, `auto` usa a cadeia top-level `fallback_providers` antes da cadeia built-in de discovery auxiliary.

## Opções de request por provider {#per-provider-request-options}

Entradas de provider (`providers.<name>` no dict `providers:`, ou itens na lista legacy `custom_providers`) aceitam dois knobs que moldam como o Hermes fala com o endpoint:

**`extra_headers`** — mapping de headers HTTP extras anexados a toda requisição LLM roteada para a base URL daquele provider. Aplicados por último, após defaults de URL/profile e overrides de header do usuário, então sobrevivem a swaps de credencial e rebuilds de client. Útil para Cloudflare Access service tokens, auth de proxy ou esquemas bearer custom:

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    api_key: sk-...
    extra_headers:
      CF-Access-Client-Id: "xxxx.access"
      CF-Access-Client-Secret: "yyyy"
```

Valores de header carregam credenciais rotineiramente — o Hermes nunca os loga. `extra_headers` aplica a rotas OpenAI-compatible; os modos de API `anthropic_messages` e `bedrock_converse` não o usam.

**`discover_models`** — defina `false` (padrão `true`) para pular query da listagem `/models` do endpoint e usar só os `models` que você configurou na entrada. Handy para gateways cuja listagem de models é lenta, não confiável ou ruidosa:

```yaml
providers:
  my-gateway:
    api: https://llm.internal.example.com/v1
    discover_models: false
    models:
      - my-finetune-v2
      - my-finetune-v1
```

Com discovery off, o model picker (`hermes model`, `/model`) mostra a lista configurada em vez de probe live.

Para um gateway compatível com Anthropic que resolve um alias bare de modelo só
depois de receber a requisição, opte o alias em marcadores nativos de prompt-cache
com a capability per-model `prompt_caching`:

```yaml
providers:
  anthropic-proxy:
    api: https://gateway.example.com/anthropic
    transport: anthropic_messages
    models:
      fable:
        context_length: 1000000
        prompt_caching: true
```

O Hermes casa esta declaração com a rota exata do provider e o id de modelo
em runtime, sem reescrever o alias. Defina `prompt_caching: false` para
desabilitar explicitamente marcadores de cache para um modelo; quando omitido, o Hermes
mantém sua detecção normal de capability de provider e modelo.

:::note Formato legacy
Configs antigos usavam lista top-level `custom_providers:` (com `base_url` em vez de `api`). Ainda funciona e é auto-migrado para dict `providers:` no `hermes update` (config v12).
:::

## Quando entra em vigor? {#when-does-it-take-effect}

- **CLI** (`hermes chat`): próxima invocação `hermes chat`.
- **Gateway** (Telegram, Discord, Slack, etc.): próxima sessão *nova*. Sessões existentes mantêm seu model. Reinicie o gateway (`hermes gateway restart`) se quiser forçar todas as sessões a pegar a mudança.
- **Aba chat do dashboard** (`/chat`): próximo PTY novo. O chat aberto mantém seu model — use `/model` dentro dele para hot-swap.

Mudanças nunca invalidam prompt caches em sessões rodando. Isso é deliberado: trocar model principal dentro de sessão requer reset de cache (system prompt contém conteúdo específico de model), e reservamos isso para slash command explícito `/model` dentro do chat.

## Solução de problemas {#troubleshooting}

### "No authenticated providers" no picker

O Hermes lista provider só se tem credencial funcional. Cheque **Keys** na sidebar — deve ver um de: API key, OAuth bem-sucedido, ou URL de endpoint custom. Se o provider que quer não está lá, rode `hermes setup` para configurar, ou vá em **Keys** e adicione a env var.

### Model principal não mudou no chat rodando

Esperado. O dashboard grava `config.yaml`, que novas sessões leem. O chat aberto é processo agent live — mantém o model com que foi spawnado. Use `/model <name>` dentro do chat para hot-swap aquela sessão específica.

### Override auxiliary "não entrou em vigor"

Três coisas para checar:

1. **Iniciou sessão nova?** Chats existentes não relêem config.
2. **`provider` está em algo diferente de `auto`?** Se o campo mostra `auto`, a tarefa ainda usa seu model principal. Clique **Change** e escolha provider real.
3. **Provider está autenticado?** Se atribuiu `minimax` a uma tarefa mas não tem API key MiniMax, a tarefa cai para default openrouter e loga aviso em `agent.log`.

### Escolhi model mas Hermes trocou providers

No OpenRouter (ou qualquer agregador), nomes bare de model resolvem *dentro* do agregador primeiro. Então `claude-sonnet-4` no OpenRouter vira `anthropic/claude-sonnet-4.6`, ficando na auth OpenRouter. Mas se digitou `claude-sonnet-4` numa auth Anthropic nativa, ficaria `claude-sonnet-4-6`. Se vir troca inesperada de provider, cheque se provider atual é o esperado — o picker sempre mostra o main atual no topo do diálogo.

## Métodos alternativos {#alternative-methods}

### Slash command CLI

Dentro de qualquer sessão `hermes chat`:

```
/model gpt-5.4 --provider openrouter             # só sessão
/model gpt-5.4 --provider openrouter --global    # também persiste em config.yaml
/model claude-opus-4.6 --once                    # só próximo turno, depois auto-restaura
```

`--global` faz o mesmo que **Change** do dashboard, mais troca a sessão rodando in-place.

`--once` troca por um turno e restaura o model anterior depois — em sucesso, erro ou interrupt igualmente. Nada persiste: restart de gateway mid-turn volta no model original. Útil para escalar uma pergunta difícil para model caro ("ask Opus just this once") ou cair para model barato numa query descartável.

:::note Custo de prompt-cache
Troca one-turn quebra o prefixo prompt-cache do provider duas vezes (saindo e voltando). Em sessão longa num provider cached-prefix (Anthropic, OpenAI), o próximo turno re-paga custo full de input — `--once` ganha em sessões curtas ou escalação barato→caro, mas pergunta lateral rápida dentro de sessão longa cara pode custar mais do que economiza.
:::

### Aliases custom

Defina nomes curtos para models que você alcança frequentemente, depois use `/model <alias>` no CLI ou qualquer plataforma de mensagens. Dois formatos equivalentes — escolha o que encaixa no workflow.

**Canônico (top-level `model_aliases:`)** — controle total sobre provider + base_url:

```yaml
# ~/.hermes/config.yaml
model_aliases:
  fav:
    model: claude-sonnet-4.6
    provider: anthropic
  grok:
    model: grok-4
    provider: x-ai
```

**Forma string curta (`model.aliases.<name>: provider/model`)** — conveniente do shell porque `hermes config set` grava escalares e agora também parseia literais inline de list/mapping, embora esta forma curta de alias ainda não possa carregar um `base_url` custom:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

> `hermes config set` também aceita **literais inline de list/mapping** (estilo flow JSON/YAML). Coloque-os entre aspas para o shell passá-los intactos:
>
> ```bash
> hermes config set platform_toolsets.line '["clarify", "file", "web"]'
> hermes config set display.tool_progress_overrides '{"terminal": "off"}'
> ```

Ambos os caminhos alimentam o mesmo loader (`hermes_cli/model_switch.py`). Entradas declaradas em `model_aliases:` têm precedência sobre entradas `model.aliases:` com o mesmo nome.

Depois `/model fav` ou `/model grok` no chat. Aliases do usuário sombreiam nomes curtos built-in (`sonnet`, `kimi`, `opus`, etc.). Veja [Aliases de model custom](/reference/slash-commands#custom-model-aliases) para referência completa.

### Subcomando `hermes model`

```bash
hermes model            # Picker interativo provider + model (forma canônica de trocar defaults)
```

`hermes model` guia você a escolher provider, autenticar (OAuth flows abrem browser; providers API-key pedem a key), e depois escolher model específico do catálogo curado daquele provider. A escolha é gravada em `model.provider` e `model.default` em `~/.hermes/config.yaml`.

Para listar providers/models sem lançar picker, use dashboard ou endpoints REST abaixo. Para inspecionar o que o CLI vai usar agora: `hermes config get model --json` e `hermes status`.

### Edição direta de config

Edite `~/.hermes/config.yaml` e reinicie o que o lê. Veja [Referência de configuração](./configuration.md) para schema completo.

### REST API

O dashboard usa três endpoints. Útil para scripting:

```bash
# Lista providers autenticados + listas curadas de models
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# Lê atribuições atuais main + auxiliary
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# Define model principal
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# Override de uma task auxiliary
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Atribui um model a toda task auxiliary
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Reseta todas as tasks auxiliary para auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

O session token é injetado no HTML do dashboard no startup e rotaciona a cada restart do server. Pegue em devtools do browser (`window.__HERMES_SESSION_TOKEN__`) se estiver scriptando contra dashboard rodando.
