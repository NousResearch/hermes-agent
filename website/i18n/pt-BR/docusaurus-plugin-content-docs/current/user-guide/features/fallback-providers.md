---
title: Providers de Fallback
description: Configure failover automático para providers LLM de backup quando seu modelo primário estiver indisponível.
sidebar_label: Providers de Fallback
sidebar_position: 8
---

# Providers de Fallback

O Hermes Agent tem três camadas de resiliência que mantêm suas sessões rodando quando providers encontram problemas:

1. **[Credential pools](./credential-pools.md)** — rotaciona entre múltiplas API keys para o *mesmo* provider (tentado primeiro)
2. **Fallback do modelo primário** — troca automaticamente para um par provider:modelo *diferente* quando seu modelo principal falha
3. **Fallback de tarefas auxiliares** — resolução independente de provider para tarefas auxiliares como visão, compressão e extração web

Credential pools lidam com rotação no mesmo provider (ex.: múltiplas chaves OpenRouter). Esta página cobre fallback cross-provider. Ambos são opcionais e funcionam de forma independente.

## Fallback do modelo primário {#primary-model-fallback}

Quando seu provider LLM principal encontra erros — rate limits, sobrecarga de servidor, falhas de auth, quedas de conexão — o Hermes pode trocar automaticamente para um par provider:modelo de backup no meio da sessão sem perder sua conversa.

### Configuração {#configuration}

O caminho mais fácil é o gerenciador interativo:

```bash
hermes fallback
```

`hermes fallback` reutiliza o provider picker de `hermes model` — mesma lista de providers, mesmos prompts de credenciais, mesma validação. Use os subcomandos `add`, `list` (alias `ls`), `remove` (alias `rm`) e `clear` para gerenciar a chain. Mudanças persistem sob a lista top-level `fallback_providers:` em `config.yaml`.

Se preferir editar o YAML diretamente, adicione uma lista top-level `fallback_providers` em `~/.hermes/config.yaml`:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

Cada entrada exige `provider` e `model`. Entradas sem qualquer um dos campos são ignoradas.

:::note `fallback_model` vs `fallback_providers`
`fallback_providers` (plural, lista) é o formato de config atual e suporta múltiplos fallbacks tentados em ordem. `fallback_model` (singular) é a chave legada de fallback único — o Hermes ainda a honra por back-compat, mas `hermes fallback` grava a chave atual `fallback_providers` e migra config legada na escrita. Quando ambos estão definidos, `fallback_providers` tem prioridade.
:::

### Providers suportados {#supported-providers}

| Provider | Valor | Requisitos |
|----------|-------|-------------|
| AI Gateway | `ai-gateway` | `AI_GATEWAY_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal` (fresh) ou `hermes auth add nous` (OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` ou `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | Processo externo (integração com editor) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` ou credenciais Claude Code |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (China) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (opcional: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (opcional: `GMI_BASE_URL`) |
| Upstage Solar | `upstage` (alias `solar`) | `UPSTAGE_API_KEY` (opcional: `UPSTAGE_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (opcional: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (alias: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (alias `grok`) | `XAI_API_KEY` (opcional: `XAI_BASE_URL`) |
| xAI Grok OAuth (SuperGrok) | `xai-oauth` (alias `grok-oauth`) | `hermes model` → xAI Grok OAuth (login no browser; assinatura SuperGrok) |
| AWS Bedrock | `bedrock` | Auth boto3 padrão (`AWS_REGION` + `AWS_PROFILE` ou `AWS_ACCESS_KEY_ID`) |
| Qwen Portal (OAuth) | `qwen-oauth` | `hermes model` (Qwen Portal OAuth; opcional: `HERMES_QWEN_BASE_URL`) |
| MiniMax (OAuth) | `minimax-oauth` | `hermes model` (MiniMax portal OAuth) |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY` (fallback para `DASHSCOPE_API_KEY`) |
| Kimi / Moonshot (China) | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (local) | `lmstudio` | `LM_API_KEY` (ou nenhuma para local) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| Endpoint customizado | `custom` | `base_url` + `key_env` (veja abaixo) |

### Fallback de endpoint customizado {#custom-endpoint-fallback}

Para um endpoint OpenAI-compatible customizado, adicione `base_url` e opcionalmente `key_env`:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # env var name containing the API key
```

### Quando o fallback dispara {#when-fallback-triggers}

O fallback ativa automaticamente quando o modelo primário falha com:

- **Rate limits** (HTTP 429) — após esgotar tentativas de retry
- **Erros de servidor** (HTTP 500, 502, 503) — após esgotar tentativas de retry
- **Falhas de auth** (HTTP 401, 403) — imediatamente (não adianta retry)
- **Not found** (HTTP 404) — imediatamente
- **Respostas inválidas** — quando a API retorna respostas malformadas ou vazias repetidamente

Quando disparado, o Hermes:

1. Resolve credenciais para o provider de fallback
2. Constrói um novo client de API
3. Troca modelo, provider e client in-place
4. Reseta o contador de retry e continua a conversa

A troca é seamless — seu histórico de conversa, tool calls e contexto são preservados. O agente continua exatamente de onde parou, só usando um modelo diferente.

:::warning Fallback reseta o prompt cache
Prompt caches são keyed ao modelo (e na maioria dos providers, à conta) servindo a requisição. Quando o fallback dispara, o novo provider:modelo não tem prefixo cached para sua conversa, então a próxima requisição relê todo o histórico a preço cheio de input tokens em vez da taxa cached com ~75–90% de desconto. O mesmo vale quando o turno termina e o primário é restaurado — essa primeira requisição de volta ao primário é uma releitura completa também (a menos que o TTL de cache do primário não tenha expirado). Isso é inevitável — é o custo de permanecer vivo durante uma outage — mas é por isso que uma sessão longa que oscila entre providers pode custar perceptivelmente mais que uma que fica fixa.
:::

:::info Por turno, não por sessão
Fallback é **turn-scoped**: cada nova mensagem do usuário começa com o modelo primário restaurado. Se o primário falha no meio do turno, o fallback ativa só naquele turno. Na próxima mensagem, o Hermes tenta o primário de novo. Dentro de um único turno, o fallback ativa no máximo uma vez — se o fallback também falhar, o error handling normal assume (retries, depois mensagem de erro). Isso previne loops de failover em cascata dentro de um turno enquanto dá ao modelo primário uma chance nova a cada turno.

O retry por turno é **reset-aware**: quando as credenciais do primário reportam um horário de reset de rate limit que ainda não passou (janelas de assinatura como os blocos de 5 horas do Claude Pro/Max ou limites semanais do Codex reportam isso em horas ou dias), o Hermes pula o retry fadado e permanece no fallback até o reset passar — evitando duas trocas de provider inúteis (e duas invalidações de prompt cache) por turno. No momento em que o horário de reset passa, o próximo turno volta ao primário automaticamente. 429s transitórios sem horário de reset mantêm o comportamento existente: um cooldown curto, depois retry a cada turno.
:::

### Exemplos {#examples}

**OpenRouter como fallback para Anthropic nativo:**
```yaml
model:
  provider: anthropic
  default: claude-sonnet-4-6

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

**Nous Portal como fallback para OpenRouter:**
```yaml
model:
  provider: openrouter
  default: anthropic/claude-opus-4

fallback_providers:
  - provider: nous
    model: nous-hermes-3
```

**Modelo local como fallback para cloud:**
```yaml
fallback_providers:
  - provider: custom
    model: llama-3.1-70b
    base_url: http://localhost:8000/v1
    key_env: LOCAL_API_KEY
```

**Codex OAuth como fallback:**
```yaml
fallback_providers:
  - provider: openai-codex
    model: gpt-5.3-codex
```

### Onde o fallback funciona {#where-fallback-works}

| Contexto | Fallback suportado |
|---------|-------------------|
| Sessões CLI | ✔ |
| Gateway de messaging (Telegram, Discord, etc.) | ✔ |
| Delegação de subagentes | ✔ (subagentes herdam a chain de fallback do pai) |
| Cron jobs | ✔ (agentes cron herdam fallback providers configurados) |
| Tarefas auxiliares em `provider: auto` | ✔ (tenta fallback por tarefa, depois a chain de fallback principal antes da descoberta aux built-in) |

:::tip
Não há environment variables para a chain de fallback primária — configure exclusivamente via `config.yaml` ou `hermes fallback`. Isso é intencional: configuração de fallback é uma escolha deliberada, não algo que um export stale de shell deveria sobrescrever.
:::

---

## Fallback de tarefas auxiliares {#auxiliary-task-fallback}

O Hermes usa modelos leves separados para tarefas auxiliares. Cada tarefa tem sua própria chain de resolução de provider que age como sistema de fallback integrado.

### Tarefas com resolução independente de provider {#tasks-with-independent-provider-resolution}

| Tarefa | O que faz | Chave de config |
|------|-------------|-----------|
| Vision | Análise de imagem, screenshots de browser | `auxiliary.vision` |
| Web Extract | Sumarização de páginas web | `auxiliary.web_extract` |
| Compression | Resumos de compressão de contexto | `auxiliary.compression` |
| Skills Hub | Busca e descoberta de skills | `auxiliary.skills_hub` |
| MCP | Operações helper MCP | `auxiliary.mcp` |
| Approval | Classificação inteligente de aprovação de comandos | `auxiliary.approval` |
| Title Generation | Resumos de título de sessão | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / botão ✨ do dashboard — expande uma tarefa de triagem one-liner em spec real | `auxiliary.triage_specifier` |

### Chain de auto-detecção {#auto-detection-chain}

Quando o provider de uma tarefa está em `"auto"` (o padrão), o Hermes primeiro tenta o provider principal + modelo principal para aquela tarefa auxiliar. Se essa rota estiver indisponível ou falhar depois com erro estilo capacity, o Hermes agora honra a política de fallback configurada pelo usuário antes de usar a chain de descoberta built-in:

```text
Main provider + main model → auxiliary.<task>.fallback_chain →
fallback_providers / fallback_model → built-in auxiliary discovery chain
```

A chain específica da tarefa é mais precisa e vence quando presente. A chain top-level `fallback_providers` é a mesma política que o agente principal usa, então regras de fallback free-only ou same-provider se aplicam a tarefas auxiliares em `auto` também.

**Chain de descoberta de texto built-in (compression, web extract, title generation, etc.):**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
```

**Chain de descoberta de vision built-in:**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

Essas chains built-in são fallback de conveniência para usuários que não declararam política de fallback específica da tarefa ou principal.

### Configurando providers auxiliares {#configuring-auxiliary-providers}

Cada tarefa pode ser configurada independentemente em `config.yaml`:

```yaml
auxiliary:
  vision:
    provider: "auto"              # auto | openrouter | nous | codex | main | anthropic
    model: ""                     # e.g. "openai/gpt-4o"
    base_url: ""                  # direct endpoint (takes precedence over provider)
    api_key: ""                   # API key for base_url

  web_extract:
    provider: "auto"
    model: ""

  compression:
    provider: "auto"
    model: ""
    fallback_chain:              # optional, task-specific fallback policy
      - provider: openrouter
        model: inclusionai/ring-2.6-1t:free

  skills_hub:
    provider: "auto"
    model: ""

  mcp:
    provider: "auto"
    model: ""
```

Toda tarefa acima segue o mesmo padrão **provider / model / base_url**. Cada tarefa também pode declarar sua própria `fallback_chain`; se omitida, `provider: auto` usa a chain top-level `fallback_providers` antes da chain de descoberta auxiliar built-in do Hermes.

A compressão de contexto é configurada sob `auxiliary.compression`:

```yaml
auxiliary:
  compression:
    provider: main                                    # Same provider options as other auxiliary tasks
    model: google/gemini-3-flash-preview
    base_url: null                                    # Custom OpenAI-compatible endpoint
```

E a chain de fallback primária usa:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # Optional custom endpoint
```

Os três — auxiliary, compression, fallback — funcionam da mesma forma: defina `provider` para escolher quem trata a requisição, `model` para escolher qual modelo, e `base_url` para apontar a um endpoint customizado (sobrescreve provider).

### Opções de provider para tarefas auxiliares {#provider-options-for-auxiliary-tasks}

Essas opções se aplicam apenas a entradas `auxiliary:`, `compression:` e `fallback_providers:` — `"main"` **não** é um valor válido para seu `model.provider` top-level. Para endpoints customizados, use `provider: custom` na seção `model:` (veja [AI Providers](/integrations/providers)).

| Provider | Descrição | Requisitos |
|----------|-------------|-------------|
| `"auto"` | Tenta providers em ordem até um funcionar (padrão) | Pelo menos um provider configurado |
| `"openrouter"` | Força OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | Força Nous Portal | `hermes auth` |
| `"codex"` | Força Codex OAuth | `hermes model` → Codex |
| `"main"` | Usa qualquer provider que o agente principal usa (apenas tarefas auxiliares) | Provider principal ativo configurado |
| `"anthropic"` | Força Anthropic nativo | `ANTHROPIC_API_KEY` ou credenciais Claude Code |

### Override de endpoint direto {#direct-endpoint-override}

Para qualquer tarefa auxiliar, definir `base_url` contorna a resolução de provider totalmente e envia requisições diretamente àquele endpoint:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` tem precedência sobre `provider`. O Hermes usa a `api_key` configurada para autenticação, caindo para `OPENAI_API_KEY` se não definida. Ele **não** reutiliza `OPENROUTER_API_KEY` para endpoints customizados.

---

## Fallback de erro de capacity auxiliar {#auxiliary-capacity-error-fallback}

Quando você define um provider auxiliar explícito (ex.: `auxiliary.vision.provider: glm`), o Hermes trata isso como sua escolha preferida — mas se o provider literalmente não consegue servir a requisição por um **erro de capacity** (HTTP 402 payment required, esgotamento de quota diária HTTP 429, falha de conexão), o Hermes faz fallback por uma chain em camadas em vez de falhar silenciosamente:

1. **Provider aux primário** — o que você configurou (tentado primeiro, sempre)
2. **`auxiliary.<task>.fallback_chain`** — sua lista de override por tarefa, se escreveu uma
3. **Provider + modelo do agente principal** — rede de segurança last-resort (sempre tentado, mesmo se não escreveu uma chain)
4. **Warn + re-raise** — se toda camada falhar, o Hermes loga `Auxiliary <task>: ... all fallbacks exhausted` em nível WARNING e re-lança o erro original

HTTP 429 transitórios de rate limit (`Retry-After: ...`) são tratados como restrições de requisição, não problemas de capacity — respeitam sua escolha explícita de provider e **não** disparam a ladder de fallback. Apenas esgotamento de quota diária/mensal, erros de pagamento e falhas de conexão contornam o gate de provider explícito.

Para usuários em `provider: auto` (sem provider aux explícito), a chain de auto-detecção existente roda no lugar dos passos 2–3. Seu primeiro passo já é o modelo do agente principal, então usuários `auto` obtêm o mesmo resultado com zero config.

### Opcional: chain de fallback por tarefa {#optional-per-task-fallback-chain}

Se quiser uma ordem de fallback diferente de "modelo do agente principal primeiro", configure `fallback_chain` explicitamente. Cada entrada precisa de pelo menos `provider`; `model`, `base_url` e `api_key` são opcionais.

```yaml
auxiliary:
  vision:
    provider: glm
    model: glm-4v-flash
    fallback_chain:
      - provider: openrouter
        model: google/gemini-3-flash-preview
      - provider: nous
        model: anthropic/claude-sonnet-4

  compression:
    provider: openrouter
    fallback_chain:
      - provider: openai
        model: gpt-4o-mini
        timeout: 240            # optional — this candidate's own deadline (seconds)
```

Você **não** precisa configurar `fallback_chain` para obter fallback — a rede de segurança do agente principal roda de qualquer forma. Use apenas quando quiser uma ordem diferente da default.

Cada entrada de `fallback_chain` também pode declarar seu próprio `timeout` (segundos). Sem ele, um candidato de fallback herda o timeout no nível da tarefa — que pode estar afinado para o provider primário. Declarar um `timeout` por entrada deixa um fallback mais lento mas confiável (ex.: um summarizer de contexto grande) receber o budget que realmente precisa em vez de morrer no relógio do primário.

### Erros de quota de provider que disparam fallback {#provider-quota-errors-that-trigger-fallback}

O Hermes reconhece estes como equivalentes a capacity de esgotamento de crédito 402 (não rate limits transitórios):

- Bedrock / LiteLLM: `Too many tokens per day`, `daily limit`, `tokens per day`
- Vertex AI / GCP: `quota exceeded`, `resource exhausted`, `RESOURCE_EXHAUSTED`
- Genérico: `daily quota`, `quota_exceeded`

Se seu provider retorna uma frase diferente para esgotamento de quota diária e o Hermes não dispara fallback, isso é um bug — abra uma issue com a string de erro exata.

---

## Fallback de compressão de contexto {#context-compression-fallback}

A compressão de contexto usa o bloco de config `auxiliary.compression` para controlar qual modelo e provider trata a sumarização:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Migração legada
Configs antigas com `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` são migradas automaticamente para `auxiliary.compression.*` no primeiro load (config version 17).
:::

Se nenhum provider estiver disponível para compression, o Hermes descarta turnos do meio da conversa sem gerar resumo em vez de falhar a sessão.

---

## Override de provider de delegação {#delegation-provider-override}

Subagentes criados por `delegate_task` herdam a chain de fallback primária do agente pai. Você ainda pode rotear subagentes para um par provider:modelo primário diferente para otimização de custo:

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

Veja [Delegação de Subagentes](/user-guide/features/delegation) para detalhes completos de configuração.

---

## Providers de cron jobs {#cron-job-providers}

Cron jobs herdam sua chain `fallback_providers` configurada (ou `fallback_model` legado) quando criam um agente. Para usar um provider primário diferente para um cron job, configure overrides `provider` e `model` no próprio cron job:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

Veja [Tarefas Agendadas (Cron)](/user-guide/features/cron) para detalhes completos de configuração.

---

## Resumo {#summary}

| Recurso | Mecanismo de fallback | Local de config |
|---------|-------------------|----------------|
| Modelo do agente principal | `fallback_providers` em config.yaml — failover por turno em erros (primário restaurado a cada turno) | `fallback_providers:` (lista top-level) |
| Tarefas auxiliares (qualquer) — usuários auto | Chain de auto-detecção completa (modelo do agente principal primeiro, depois chain de providers) em erros de capacity | `auxiliary.<task>.provider: auto` |
| Tarefas auxiliares (qualquer) — provider explícito | `fallback_chain` (se definida) → modelo do agente principal → warn + raise, apenas em erros de capacity | `auxiliary.<task>.fallback_chain` |
| Vision | Em camadas (veja acima) + retry interno OpenRouter | `auxiliary.vision` |
| Web extraction | Em camadas (veja acima) + retry interno OpenRouter | `auxiliary.web_extract` |
| Compressão de contexto | Em camadas (veja acima); degrada para no-summary se todas as camadas indisponíveis | `auxiliary.compression` |
| Skills hub | Em camadas (veja acima) | `auxiliary.skills_hub` |
| Helpers MCP | Em camadas (veja acima) | `auxiliary.mcp` |
| Classificação de aprovação | Em camadas (veja acima) | `auxiliary.approval` |
| Geração de título | Em camadas (veja acima) | `auxiliary.title_generation` |
| Triage specifier | Em camadas (veja acima) | `auxiliary.triage_specifier` |
| Delegação | Herda a chain `fallback_providers` do pai; override opcional de provider/model | `delegation.provider` / `delegation.model` |
| Cron jobs | Herdam a chain `fallback_providers` configurada; override opcional de provider por job | `provider` / `model` por job |
