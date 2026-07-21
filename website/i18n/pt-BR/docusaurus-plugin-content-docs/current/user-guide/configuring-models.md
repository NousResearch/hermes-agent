---
sidebar_position: 3
---

# Configurando modelos

O Hermes usa dois tipos de slots de modelo:

- **Modelo principal** — com o qual o agente pensa. Toda mensagem do usuário, todo loop de chamada de ferramenta, toda resposta em streaming passa por este modelo.
- **Modelos auxiliares** — trabalhos menores que o agente terceiriza. Compressão de contexto, visão (análise de imagem), sumarização de páginas web, pontuação de aprovação, roteamento de ferramentas MCP, geração de título de sessão e busca de skills. Cada um tem seu próprio slot e pode ser substituído independentemente.

Esta página cobre a configuração de ambos pelo dashboard. Se preferir arquivos de config ou CLI, vá para [Métodos alternativos](#alternative-methods) no final.

:::tip Caminho mais rápido: Nous Portal
O [Nous Portal](/user-guide/features/tool-gateway) oferece 300+ modelos em uma assinatura. Em uma instalação nova, execute `hermes setup --portal` para fazer login e definir Nous como seu provedor em um comando. Inspecione o que está conectado com `hermes portal info`.

- Assinantes Portal também ganham **10% de desconto em provedores faturados por token**.
:::

:::note Schema `model:` — string vazia vs. mapping
Em uma instalação nova, a config padrão incluída tem `model: ""` (sentinela de string vazia significando "ainda não configurado"). Na primeira vez que você executa `hermes setup` ou `hermes model`, essa chave é atualizada in-place para um mapping com sub-chaves `provider`, `default`, `base_url` e `api_mode` — a forma mostrada ao longo desta página e em [`profiles.md`](./profiles.md) / [`configuration.md`](./configuration.md). Se você vir uma string vazia em `config.yaml`, execute `hermes model` (ou clique **Change** no dashboard) e o Hermes gravará a forma dict para você.
:::

## A página Models

Abra o dashboard e clique em **Models** na barra lateral. Você tem duas seções:

1. **Model Settings** — o painel superior, onde você atribui modelos aos slots.
2. **Usage analytics** — cards ranqueados mostrando todo modelo que executou uma sessão no período selecionado, com contagens de tokens, custo e badges de capacidade.

![Models page overview](/img/docs/dashboard-models/overview.png)

O card superior é o painel **Model Settings**. A linha principal sempre mostra o que o agente usará para novas sessões. Clique **Change** para abrir o seletor.

## Definindo o modelo principal

Clique **Change** na linha Main model:

![Model picker dialog](/img/docs/dashboard-models/picker-dialog.png)

O seletor tem duas colunas:

- **Esquerda** — provedores autenticados. Só aparecem provedores que você configurou (chave de API definida, OAuth feito ou endpoint custom definido). Se um provedor estiver ausente, vá em **Keys** e adicione a credencial.
- **Direita** — a lista curada de modelos do provedor selecionado. São os modelos agenticos que o Hermes recomenda para aquele provedor, não o dump bruto `/models` (que no OpenRouter inclui 400+ modelos incluindo TTS, geradores de imagem e rerankers).

Digite na caixa de filtro para restringir por nome de provedor, slug ou ID de modelo.

Escolha um modelo, pressione **Switch**, e o Hermes grava em `~/.hermes/config.yaml` na seção `model`. **Isso se aplica só a novas sessões** — qualquer aba de chat já aberta continua rodando o modelo com que começou. Para trocar quente o chat atual, use o slash command `/model` dentro dele.

### Trocas no meio da sessão e avisos de contexto

Quando você troca modelos **dentro de uma sessão ativa** (seletor de modelo Herm TUI, CLI `hermes` ou `/model` no Telegram/Discord), o Hermes estima se sua **próxima mensagem** executará **compressão de contexto preflight** contra a janela do novo modelo. Se a sessão já estiver perto ou acima do limiar de compressão daquele modelo (veja [Context Compression](./configuration.md#context-compression)), a resposta da troca inclui um aviso — o mesmo caminho `warning_message` usado para avisos de modelo caro. A troca ainda se aplica imediatamente; a compressão roda na **primeira mensagem do usuário após a troca**, antes do modelo responder.

:::warning Trocas no meio da sessão resetam o cache de prompt
Caches de prompt são vinculados ao modelo que atende a requisição, então qualquer mudança de modelo no meio da conversa — uma troca explícita `/model`, um [fallback automático](./features/fallback-providers.md) ou uma rotação de [credential-pool](./features/credential-pools.md) para outra conta — significa que a próxima mensagem relê toda a conversa pelo preço integral de tokens de entrada em vez da taxa em cache (~75–90% de desconto). Em uma sessão longa, essa releitura única pode superar a diferença por token entre os dois modelos. Troque quando precisar, mas prefira fazer cedo na conversa ou logo após iniciar uma sessão nova.
:::

## Definindo modelos auxiliares

Clique **Show auxiliary** para revelar os 11 slots de tarefa:

![Auxiliary panel expanded](/img/docs/dashboard-models/auxiliary-expanded.png)

Toda tarefa auxiliar usa `auto` por padrão — significa que o Hermes tenta seu modelo principal para aquele trabalho também. Se essa rota estiver indisponível ou bater em falha estilo capacidade, `auto` segue qualquer `auxiliary.<task>.fallback_chain` específico da tarefa, depois a cadeia principal `fallback_providers` / `fallback_model`, depois a cadeia de descoberta auxiliar built-in do Hermes. Substitua uma tarefa específica quando quiser um modelo mais barato ou rápido para um side-job.

### Padrões comuns de override

| Task | When to override |
|---|---|
| **Title Gen** | Almost always. A $0.10/M flash model writes session titles as well as Opus. Default config sets this to `google/gemini-3-flash-preview` on OpenRouter. |
| **Vision** | When your main model lacks vision support. Point it at `google/gemini-2.5-flash` or `gpt-4o-mini`. |
| **Compression** | When you're burning reasoning tokens on Opus/M2.7 just to summarize context. A fast chat model does the job at 1/50th the cost. |
| **Approval** | For `approval_mode: smart` — a fast/cheap model (haiku, flash, gpt-5-mini) decides whether to auto-approve low-risk commands. Expensive models here are waste. |
| **Web Extract** | When you use `web_extract` heavily. Same logic as compression — summarization doesn't need reasoning. |
| **Skills Hub** | `hermes skills search` uses this. Usually fine at `auto`. |
| **MCP** | MCP tool routing. Usually fine at `auto`. |
| **Triage Specifier** | Routes the Kanban triage specifier (`hermes kanban specify`) that expands a rough one-liner into a concrete spec. A cheap, capable model works well. |
| **Kanban Decomposer** | Routes Kanban task decomposition — splits a triage task into a graph of child tasks for specialist profiles. |
| **Profile Describer** | Routes profile-description generation (`hermes profile describe --auto` / the dashboard auto-generate button). Short, cheap call. |
| **Curator** | Routes the curator skill-usage review pass. Can run for minutes on reasoning models, so a cheaper aux model is often worthwhile. |

### Override por tarefa

Clique **Change** em qualquer linha auxiliar. O mesmo seletor abre, mesmo comportamento — escolha provedor + modelo, pressione Switch. A linha atualiza para mostrar `provider · model` em vez de `auto (use main model)`.

### Resetar tudo para auto

Se você ajustou demais e quer recomeçar, clique **Reset all to auto** no topo da seção auxiliar. Cada slot volta a usar seu modelo principal.

## O atalho "Use as"

Todo card de modelo na página tem um dropdown **Use as**. Este é o caminho rápido — escolha um modelo que vê em suas analytics, clique **Use as**, e atribua ao slot principal ou a qualquer tarefa auxiliar específica em um clique:

![Use as dropdown](/img/docs/dashboard-models/use-as-dropdown.png)

O dropdown tem:

- **Main model** — igual a clicar Change na linha principal.
- **All auxiliary tasks** — atribui este modelo a todos os 11 slots aux de uma vez. Útil quando você quer todo side-job em um flash barato.
- **Individual task options** — Vision, Web Extract, Compression, etc. O modelo atualmente atribuído a cada tarefa está marcado `current`.

Cards recebem badge `main` ou `aux · <task>` quando estão atribuídos a algo — para ver de relance quais modelos históricos estão conectados onde.

## O que é gravado em `config.yaml`

Quando você salva pelo dashboard, o Hermes grava em `~/.hermes/config.yaml`:

**Main model:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4.7
  base_url: ''        # cleared on provider switch
  api_mode: chat_completions
```

**Auxiliary override (example — vision on gemini-flash):**
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

**Auxiliary on auto (default):**
```yaml
auxiliary:
  compression:
    provider: auto
    model: ''
    base_url: ''
    # ... other fields unchanged
```

`provider: auto` com `model: ''` diz ao Hermes para usar o modelo principal para aquela tarefa, ainda honrando a política de fallback se a rota principal não puder atender a chamada auxiliar.

Cadeias de fallback específicas por tarefa ficam sob a mesma tarefa auxiliar:

```yaml
auxiliary:
  title_generation:
    provider: auto
    model: ''
    fallback_chain:
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free
```

Quando `fallback_chain` está ausente, `auto` usa a cadeia top-level `fallback_providers` antes da cadeia de descoberta auxiliar built-in.

## Quando entra em vigor?

- **CLI** (`hermes chat`): na próxima invocação `hermes chat`.
- **Gateway** (Telegram, Discord, Slack, etc.): na próxima sessão *nova*. Sessões existentes mantêm seu modelo. Reinicie o gateway (`hermes gateway restart`) se quiser forçar todas as sessões a pegar a mudança.
- **Dashboard chat tab** (`/chat`): no próximo PTY novo. O chat aberto mantém seu modelo — use `/model` dentro dele para trocar quente.

Mudanças nunca invalidam caches de prompt em sessões rodando. Isso é deliberado: trocar o modelo principal dentro de uma sessão exige reset de cache (o system prompt contém conteúdo específico do modelo), e reservamos isso para o slash command explícito `/model` dentro do chat.

## Solução de problemas

### "No authenticated providers" no seletor

O Hermes lista um provedor só se tiver credencial funcional. Verifique **Keys** na barra lateral — você deve ver uma de: chave de API, OAuth bem-sucedido ou URL de endpoint custom. Se o provedor que você quer não estiver lá, execute `hermes setup` para conectá-lo, ou vá em **Keys** e adicione a env var.

### Modelo principal não mudou no meu chat rodando

Esperado. O dashboard grava `config.yaml`, que novas sessões leem. O chat aberto é um processo de agente ao vivo — mantém o modelo com que foi spawnado. Use `/model <name>` dentro do chat para trocar quente aquela sessão específica.

### Override auxiliar "não surtiu efeito"

Três coisas para verificar:

1. **Você iniciou uma sessão nova?** Chats existentes não relêem config.
2. **`provider` está definido para algo além de `auto`?** Se o campo mostrar `auto`, a tarefa ainda usa seu modelo principal. Clique **Change** e escolha um provedor real.
3. **O provedor está autenticado?** Se você atribuiu `minimax` a uma tarefa mas não tem chave MiniMax, a tarefa faz fallback para o padrão openrouter e registra aviso em `agent.log`.

### Escolhi um modelo mas o Hermes trocou de provedor

No OpenRouter (ou qualquer agregador), nomes de modelo bare resolvem *dentro* do agregador primeiro. Então `claude-sonnet-4` no OpenRouter vira `anthropic/claude-sonnet-4.6`, permanecendo na sua auth OpenRouter. Mas se você digitou `claude-sonnet-4` em auth Anthropic nativa, ficaria como `claude-sonnet-4-6`. Se vir troca inesperada de provedor, verifique se seu provedor atual é o esperado — o seletor sempre mostra o principal atual no topo do diálogo.

## Métodos alternativos {#alternative-methods}

### Slash command CLI

Dentro de qualquer sessão `hermes chat`:

```
/model gpt-5.4 --provider openrouter             # session-only
/model gpt-5.4 --provider openrouter --global    # also persists to config.yaml
/model claude-opus-4.6 --once                    # next turn only, then auto-restores
```

`--global` faz a mesma coisa que o botão **Change** do dashboard, além de trocar a sessão rodando in-place.

`--once` troca por um único turno e restaura o modelo anterior depois — em sucesso, erro ou interrupção. Nada é persistido: reiniciar o gateway no meio do turno volta ao modelo original. Útil para escalar uma pergunta difícil a um modelo caro ("pergunte ao Opus só desta vez") ou cair a um modelo barato para uma consulta descartável.

:::note Custo de prompt-cache
Uma troca de um turno quebra o prefixo de prompt-cache do provedor duas vezes (saindo e voltando). Em uma sessão longa em provedor com prefixo em cache (Anthropic, OpenAI), o próximo turno paga custo integral de entrada — `--once` ganha em sessões curtas ou escalação barato→caro, mas uma pergunta rápida dentro de uma sessão longa cara pode custar mais do que economiza.
:::

### Aliases customizados

Defina seus próprios nomes curtos para modelos que você usa muito, depois use `/model <alias>` no CLI ou qualquer plataforma de mensagens. Existem dois formatos equivalentes — escolha o que encaixa no seu fluxo.

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

**Forma string curta (`model.aliases.<name>: provider/model`)** — conveniente do shell porque `hermes config set` só grava valores escalares, mas não carrega `base_url` custom:

```bash
hermes config set model.aliases.fav anthropic/claude-opus-4.6
hermes config set model.aliases.grok x-ai/grok-4
```

Ambos os caminhos alimentam o mesmo loader (`hermes_cli/model_switch.py`). Entradas declaradas em `model_aliases:` prevalecem sobre entradas `model.aliases:` com o mesmo nome.

Depois `/model fav` ou `/model grok` no chat. Aliases do usuário sobrepõem nomes curtos built-in (`sonnet`, `kimi`, `opus`, etc.). Veja [Custom model aliases](/reference/slash-commands#custom-model-aliases) para a referência completa.

### Subcomando `hermes model`

```bash
hermes model            # Interactive provider + model picker (the canonical way to switch defaults)
```

`hermes model` guia você a escolher provedor, autenticar (fluxos OAuth abrem navegador; provedores de chave API pedem a chave) e escolher um modelo específico do catálogo curado daquele provedor. A escolha é gravada em `model.provider` e `model.default` em `~/.hermes/config.yaml`.

Para listar provedores/modelos sem abrir o seletor, use o dashboard ou os endpoints REST abaixo. Para inspecionar o que o CLI usará agora: `hermes config get model --json` e `hermes status`.

### Edição direta de config

Edite `~/.hermes/config.yaml` e reinicie o que o lê. Veja a [referência de Configuration](./configuration.md) para o schema completo.

### REST API

O dashboard usa três endpoints. Útil para scripting:

```bash
# List authenticated providers + curated model lists
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/options

# Read current main + auxiliary assignments
curl -H "X-Hermes-Session-Token: $TOKEN" http://localhost:PORT/api/model/auxiliary

# Set the main model
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"main","provider":"openrouter","model":"anthropic/claude-opus-4.7"}' \
  http://localhost:PORT/api/model/set

# Override a single auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"vision","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Assign one model to every auxiliary task
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"","provider":"openrouter","model":"google/gemini-2.5-flash"}' \
  http://localhost:PORT/api/model/set

# Reset all auxiliary tasks to auto
curl -X POST -H "Content-Type: application/json" -H "X-Hermes-Session-Token: $TOKEN" \
  -d '{"scope":"auxiliary","task":"__reset__","provider":"","model":""}' \
  http://localhost:PORT/api/model/set
```

O session token é injetado no HTML do dashboard na inicialização e rotaciona a cada reinício do servidor. Pegue nas devtools do navegador (`window.__HERMES_SESSION_TOKEN__`) se estiver scriptando contra um dashboard rodando.
