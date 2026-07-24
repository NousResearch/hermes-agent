---
title: Provedores de fallback
description: Configure failover automático para provedores LLM de backup quando seu modelo principal estiver indisponível.
sidebar_label: Provedores de fallback
sidebar_position: 8
---

# Provedores de fallback {#fallback-providers}

O Hermes Agent tem três camadas de resiliência que mantêm suas sessões rodando quando providers encontram problemas:

1. **[Pools de credenciais](./credential-pools.md)** — rotaciona entre várias API keys para o *mesmo* provider (tentado primeiro)
2. **Fallback do modelo principal** — troca automaticamente para um provider:model *diferente* quando seu modelo principal falha
3. **Fallback de tarefas auxiliares** — resolução de provider independente para tarefas laterais como visão, compressão e extração web

Pools de credenciais tratam rotação no mesmo provider (por exemplo, várias keys OpenRouter). Esta página cobre fallback entre providers. Ambos são opcionais e funcionam de forma independente.

## Fallback do modelo principal {#primary-model-fallback}

Quando seu provider LLM principal encontra erros — rate limits, sobrecarga do servidor, falhas de auth, quedas de conexão — o Hermes pode trocar automaticamente para um par provider:model de backup no meio da sessão sem perder sua conversa.

### Configuração {#configuration}

O caminho mais fácil é o gerenciador interativo:

```bash
hermes fallback
```

`hermes fallback` reutiliza o seletor de provider de `hermes model` — mesma lista de providers, mesmos prompts de credencial, mesma validação. Use os subcomandos `add`, `list` (alias `ls`), `remove` (alias `rm`) e `clear` para gerenciar a cadeia. As alterações persistem na lista de nível superior `fallback_providers:` em `config.yaml`.

Se preferir editar o YAML diretamente, adicione uma lista `fallback_providers` de nível superior em `~/.hermes/config.yaml`:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

Cada entrada exige `provider` e `model`. Entradas sem qualquer um dos campos são ignoradas.

:::note `fallback_model` vs `fallback_providers`
`fallback_providers` (plural, lista) é o formato de config atual e suporta vários fallbacks tentados em ordem. `fallback_model` (singular) é a chave legada de fallback único — o Hermes ainda a honra por compatibilidade retroativa, mas `hermes fallback` grava a chave atual `fallback_providers` e migra config legada na escrita. Quando ambos estão definidos, `fallback_providers` tem prioridade.
:::

### Providers suportados {#supported-providers}

| Provider | Valor | Requisitos |
|----------|-------|-------------|
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Nous Portal | `nous` | `hermes setup --portal` (fresh) or `hermes auth add nous` (OAuth) |
| OpenAI Codex | `openai-codex` | `hermes model` (ChatGPT OAuth) |
| GitHub Copilot | `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` |
| GitHub Copilot ACP | `copilot-acp` | External process (editor integration) |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` or Claude Code credentials |
| z.ai / GLM | `zai` | `GLM_API_KEY` |
| Kimi / Moonshot | `kimi-coding` | `KIMI_API_KEY` |
| MiniMax | `minimax` | `MINIMAX_API_KEY` |
| MiniMax (China) | `minimax-cn` | `MINIMAX_CN_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| NVIDIA NIM | `nvidia` | `NVIDIA_API_KEY` (optional: `NVIDIA_BASE_URL`) |
| GMI Cloud | `gmi` | `GMI_API_KEY` (optional: `GMI_BASE_URL`) |
| Upstage Solar | `upstage` (alias `solar`) | `UPSTAGE_API_KEY` (optional: `UPSTAGE_BASE_URL`) |
| StepFun | `stepfun` | `STEPFUN_API_KEY` (optional: `STEPFUN_BASE_URL`) |
| Ollama Cloud | `ollama-cloud` | `OLLAMA_API_KEY` |
| Google AI Studio | `gemini` | `GOOGLE_API_KEY` (alias: `GEMINI_API_KEY`) |
| xAI (Grok) | `xai` (alias `grok`) | `XAI_API_KEY` (optional: `XAI_BASE_URL`) |
| xAI Grok OAuth (SuperGrok) | `xai-oauth` (alias `grok-oauth`) | `hermes model` → xAI Grok OAuth (browser login; SuperGrok subscription) |
| AWS Bedrock | `bedrock` | Standard boto3 auth (`AWS_REGION` + `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`) |
| Qwen Portal (OAuth) | `qwen-oauth` | `hermes model` (Qwen Portal OAuth; optional: `HERMES_QWEN_BASE_URL`) |
| MiniMax (OAuth) | `minimax-oauth` | `hermes model` (MiniMax portal OAuth) |
| OpenCode Zen | `opencode-zen` | `OPENCODE_ZEN_API_KEY` |
| OpenCode Go | `opencode-go` | `OPENCODE_GO_API_KEY` |
| Kilo Code | `kilocode` | `KILOCODE_API_KEY` |
| Xiaomi MiMo | `xiaomi` | `XIAOMI_API_KEY` |
| Arcee AI | `arcee` | `ARCEEAI_API_KEY` |
| GMI Cloud | `gmi` | `GMI_API_KEY` |
| Alibaba / DashScope | `alibaba` | `DASHSCOPE_API_KEY` |
| Alibaba Coding Plan | `alibaba-coding-plan` | `ALIBABA_CODING_PLAN_API_KEY` (falls back to `DASHSCOPE_API_KEY`) |
| Kimi / Moonshot (China) | `kimi-coding-cn` | `KIMI_CN_API_KEY` |
| StepFun | `stepfun` | `STEPFUN_API_KEY` |
| Tencent TokenHub | `tencent-tokenhub` | `TOKENHUB_API_KEY` |
| Microsoft Foundry | `azure-foundry` | `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_BASE_URL` |
| LM Studio (local) | `lmstudio` | `LM_API_KEY` (or none for local) + `LM_BASE_URL` |
| Hugging Face | `huggingface` | `HF_TOKEN` |
| Custom endpoint | `custom` | `base_url` + `key_env` (see below) |

### Custom Endpoint Fallback {#custom-endpoint-fallback}

Para um endpoint customizado compatível com OpenAI, adicione `base_url` e opcionalmente `key_env`:

```yaml
fallback_providers:
  - provider: custom
    model: my-local-model
    base_url: http://localhost:8000/v1
    key_env: MY_LOCAL_KEY            # env var name containing the API key
```

### When Fallback Triggers {#when-fallback-triggers}

O fallback ativa automaticamente quando o modelo principal falha com:

- **Rate limits** (HTTP 429) — após esgotar tentativas de retry
- **Erros de servidor** (HTTP 500, 502, 503) — após esgotar tentativas de retry
- **Falhas de auth** (HTTP 401, 403) — imediatamente (não adianta retry)
- **Not found** (HTTP 404) — imediatamente
- **Respostas inválidas** — quando a API retorna respostas malformadas ou vazias repetidamente

Quando acionado, o Hermes:

1. Resolve credenciais para o provider de fallback
2. Constrói um novo cliente de API
3. Troca modelo, provider e cliente in-place
4. Reseta o contador de retry e continua a conversa

A troca é transparente — seu histórico de conversa, tool calls e contexto são preservados. O agente continua exatamente de onde parou, só usando um modelo diferente.

:::warning Fallback resets the prompt cache
Caches de prompt são indexados ao modelo (e na maioria dos providers, à conta) que atende a requisição. Quando o fallback dispara, o novo provider:model não tem prefixo em cache para sua conversa, então a próxima requisição relê todo o histórico a preço cheio de input tokens em vez da taxa com desconto de cache (~75–90%). O mesmo vale quando o turno termina e o primário é restaurado — essa primeira requisição de volta ao primário também é releitura completa (a menos que o TTL de cache do primário não tenha expirado). Isso é inevitável — é o custo de continuar vivo durante uma indisponibilidade — mas é por isso que uma sessão longa que alterna entre providers pode custar perceptivelmente mais que uma que permanece fixa.
:::

:::info Per-Turn, Not Per-Session
Fallback é **escopado por turno**: cada nova mensagem do usuário começa com o modelo primário restaurado. Se o primário falhar no meio do turno, o fallback ativa só naquele turno. Na próxima mensagem, o Hermes tenta o primário de novo. Dentro de um único turno, o fallback ativa no máximo uma vez — se o fallback também falhar, o tratamento de erro normal assume (retries, depois mensagem de erro). Isso evita loops de failover em cascata dentro do turno, mas dá ao modelo primário uma chance nova a cada turno.
:::

### Examples {#examples}

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
| Gateway de mensagens (Telegram, Discord, etc.) | ✔ |
| Delegação de subagentes | ✔ (subagentes herdam a cadeia de fallback do pai) |
| Jobs cron | ✔ (agentes cron herdam fallback providers configurados) |
| Tarefas auxiliares com `provider: auto` | ✔ (tenta fallback por tarefa, depois a cadeia principal antes da descoberta aux built-in) |

:::tip
Não há variáveis de ambiente para a cadeia de fallback primária — configure exclusivamente via `config.yaml` ou `hermes fallback`. Isso é intencional: configuração de fallback é uma escolha deliberada, não algo que um export stale de shell deva sobrescrever.
:::

---

## Fallback de tarefas auxiliares {#auxiliary-task-fallback}

O Hermes usa modelos leves separados para tarefas laterais. Cada tarefa tem sua própria cadeia de resolução de provider que funciona como sistema de fallback built-in.

### Tarefas com resolução independente de provider {#tasks-with-independent-provider-resolution}

| Tarefa | O que faz | Chave de config |
|------|-------------|-----------|
| Vision | Análise de imagem, screenshots de browser | `auxiliary.vision` |
| Web Extract | Sumarização de página web | `auxiliary.web_extract` |
| Compression | Resumos de compressão de contexto | `auxiliary.compression` |
| Skills Hub | Busca e descoberta de skills | `auxiliary.skills_hub` |
| MCP | Operações auxiliares MCP | `auxiliary.mcp` |
| Approval | Classificação inteligente de aprovação de comandos | `auxiliary.approval` |
| Title Generation | Resumos de título de sessão | `auxiliary.title_generation` |
| Triage Specifier | `hermes kanban specify` / botão ✨ do dashboard — expande tarefa de triagem one-liner em spec real | `auxiliary.triage_specifier` |

### Cadeia de auto-detecção {#auto-detection-chain}

Quando o provider de uma tarefa está definido como `"auto"` (o padrão), o Hermes primeiro tenta o provider principal + modelo principal para aquela tarefa auxiliar. Se essa rota estiver indisponível ou falhar depois com erro de capacidade, o Hermes agora honra a política de fallback configurada pelo usuário antes de usar a cadeia de descoberta built-in:

```text
Main provider + main model → auxiliary.<task>.fallback_chain →
fallback_providers / fallback_model → built-in auxiliary discovery chain
```

A cadeia específica da tarefa é mais precisa e vence quando presente. A cadeia de nível superior `fallback_providers` é a mesma política que o agente principal usa, então regras de fallback free-only ou same-provider se aplicam a tarefas auxiliares em `auto` também.

**Cadeia de descoberta de texto built-in (compression, web extract, title generation, etc.):**

```text
OpenRouter → Nous Portal → Custom endpoint → Codex OAuth →
API-key providers (z.ai, Kimi, MiniMax, Xiaomi MiMo, Hugging Face, Anthropic) → give up
```

**Cadeia de descoberta de visão built-in:**

```text
Main provider (if vision-capable) → OpenRouter → Nous Portal →
Codex OAuth → Anthropic → Custom endpoint → give up
```

Essas cadeias built-in são fallback de conveniência para usuários que não declararam política de fallback específica da tarefa ou principal.

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

Toda tarefa acima segue o mesmo padrão **provider / model / base_url**. Cada tarefa também pode declarar sua própria `fallback_chain`; se omitida, `provider: auto` usa a cadeia de nível superior `fallback_providers` antes da cadeia de descoberta auxiliar built-in do Hermes.

A compressão de contexto é configurada em `auxiliary.compression`:

```yaml
auxiliary:
  compression:
    provider: main                                    # Same provider options as other auxiliary tasks
    model: google/gemini-3-flash-preview
    base_url: null                                    # Custom OpenAI-compatible endpoint
```

E a cadeia de fallback primária usa:

```yaml
fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
    # base_url: http://localhost:8000/v1             # Optional custom endpoint
```

Os três — auxiliary, compression, fallback — funcionam da mesma forma: defina `provider` para escolher quem trata a requisição, `model` para escolher qual modelo, e `base_url` para apontar a um endpoint customizado (sobrescreve provider).

### Opções de provider para tarefas auxiliares {#provider-options-for-auxiliary-tasks}

Essas opções se aplicam só a entradas em `auxiliary:`, `compression:` e `fallback_providers:` — `"main"` **não** é valor válido para seu `model.provider` de nível superior. Para endpoints customizados, use `provider: custom` na seção `model:` (veja [AI Providers](/integrations/providers)).

| Provider | Descrição | Requisitos |
|----------|-------------|-------------|
| `"auto"` | Tenta providers em ordem até um funcionar (padrão) | Pelo menos um provider configurado |
| `"openrouter"` | Força OpenRouter | `OPENROUTER_API_KEY` |
| `"nous"` | Força Nous Portal | `hermes auth` |
| `"codex"` | Força Codex OAuth | `hermes model` → Codex |
| `"main"` | Usa o provider que o agente principal usa (só tarefas auxiliares) | Provider principal ativo configurado |
| `"anthropic"` | Força Anthropic nativo | `ANTHROPIC_API_KEY` or Claude Code credentials |

### Override de endpoint direto {#direct-endpoint-override}

Para qualquer tarefa auxiliar, definir `base_url` contorna a resolução de provider inteiramente e envia requisições diretamente àquele endpoint:

```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` tem precedência sobre `provider`. O Hermes usa a `api_key` configurada para autenticação, com fallback para `OPENAI_API_KEY` se não definida. Ele **não** reutiliza `OPENROUTER_API_KEY` para endpoints customizados.

---

## Fallback por erro de capacidade em tarefas auxiliares {#auxiliary-capacity-error-fallback}

Quando você define um provider auxiliar explícito (por exemplo, `auxiliary.vision.provider: glm`), o Hermes trata isso como sua escolha preferida — mas se o provider literalmente não puder atender a requisição por **erro de capacidade** (HTTP 402 payment required, HTTP 429 esgotamento de cota diária, falha de conexão), o Hermes faz fallback por uma cadeia em camadas em vez de falhar silenciosamente:

1. **Provider aux primário** — o que você configurou (tentado primeiro, sempre)
2. **`auxiliary.<task>.fallback_chain`** — sua lista de override por tarefa, se você escreveu uma
3. **Provider + model do agente principal** — rede de segurança de último recurso (sempre tentado, mesmo se você não escreveu cadeia)
4. **Aviso + re-raise** — se toda camada falhar, o Hermes registra `Auxiliary <task>: ... all fallbacks exhausted` em nível WARNING e relança o erro original

HTTP 429 transitório de rate limit (`Retry-After: ...`) é tratado como restrição de requisição, não problema de capacidade — respeita sua escolha de provider explícito e **não** dispara a escada de fallback. Só esgotamento de cota diária/mensal, erros de pagamento e falhas de conexão contornam o gate de provider explícito.

Para usuários em `provider: auto` (sem provider aux explícito), a cadeia de auto-detecção existente roda no lugar dos passos 2–3. Seu primeiro passo já é o modelo do agente principal, então usuários `auto` obtêm o mesmo resultado com zero config.

### Opcional: cadeia de fallback por tarefa {#optional-per-task-fallback-chain}

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

Você **não** precisa configurar `fallback_chain` para ter fallback — a rede de segurança do agente principal roda de qualquer forma. Use só quando quiser uma ordem específica diferente do padrão.

Cada entrada de `fallback_chain` também pode declarar seu próprio `timeout` (segundos). Sem isso, um candidato de fallback herda o timeout da tarefa — que pode estar ajustado para o provider primário. Declarar `timeout` por entrada permite que um fallback mais lento porém confiável (por exemplo, um summarizer de contexto grande) receba o orçamento que realmente precisa em vez de morrer no relógio do primário.

### Erros de cota de provider que disparam fallback {#provider-quota-errors-that-trigger-fallback}

O Hermes reconhece estes como equivalentes a esgotamento de crédito 402 (não rate limits transitórios):

- Bedrock / LiteLLM: `Too many tokens per day`, `daily limit`, `tokens per day`
- Vertex AI / GCP: `quota exceeded`, `resource exhausted`, `RESOURCE_EXHAUSTED`
- Generic: `daily quota`, `quota_exceeded`

Se seu provider retornar frase diferente para esgotamento de cota diária e o Hermes não disparar fallback, isso é bug — abra uma issue com a string de erro exata.

---

## Fallback de compressão de contexto {#context-compression-fallback}

A compressão de contexto usa o bloco de config `auxiliary.compression` para controlar qual modelo e provider trata a sumarização:

```yaml
auxiliary:
  compression:
    provider: "auto"                              # auto | openrouter | nous | main
    model: "google/gemini-3-flash-preview"
```

:::info Legacy migration
Configs antigas com `compression.summary_model` / `compression.summary_provider` / `compression.summary_base_url` são migradas automaticamente para `auxiliary.compression.*` no primeiro carregamento (config version 17).
:::

Se nenhum provider estiver disponível para compressão, o Hermes descarta turnos do meio da conversa sem gerar resumo em vez de falhar a sessão.

---

## Override de provider na delegação {#delegation-provider-override}

Subagentes criados por `delegate_task` herdam a cadeia de fallback primária do agente pai. Você ainda pode rotear subagentes para um par provider:model primário diferente para otimização de custo:

```yaml
delegation:
  provider: "openrouter"                      # override provider for all subagents
  model: "google/gemini-3-flash-preview"      # override model
  # base_url: "http://localhost:1234/v1"      # or use a direct endpoint
  # api_key: "local-key"
```

Veja [Subagent Delegation](/user-guide/features/delegation) para detalhes completos de configuração.

---

## Providers em jobs cron {#cron-job-providers}

Jobs cron herdam sua cadeia `fallback_providers` configurada (ou `fallback_model` legado) quando criam um agente. Para usar provider primário diferente em um job cron, configure overrides de `provider` e `model` no próprio job:

```python
cronjob(
    action="create",
    schedule="every 2h",
    prompt="Check server status",
    provider="openrouter",
    model="google/gemini-3-flash-preview"
)
```

Veja [Scheduled Tasks (Cron)](/user-guide/features/cron) para detalhes completos de configuração.

---

## Resumo {#summary}

| Recurso | Mecanismo de fallback | Local da config |
|---------|-------------------|----------------|
| Modelo do agente principal | `fallback_providers` em config.yaml — failover por turno em erros (primary restaurado a cada turno) | `fallback_providers:` (lista de nível superior) |
| Tarefas auxiliares (qualquer) — usuários auto | Cadeia completa de auto-detecção (modelo do agente principal primeiro, depois cadeia de providers) em erros de capacidade | `auxiliary.<task>.provider: auto` |
| Tarefas auxiliares (qualquer) — provider explícito | `fallback_chain` (se definida) → modelo do agente principal → warn + raise, só em erros de capacidade | `auxiliary.<task>.fallback_chain` |
| Vision | Em camadas (veja acima) + retry interno OpenRouter | `auxiliary.vision` |
| Extração web | Em camadas (veja acima) + retry interno OpenRouter | `auxiliary.web_extract` |
| Compressão de contexto | Em camadas (veja acima); degrada para sem-resumo se todas as camadas indisponíveis | `auxiliary.compression` |
| Skills hub | Em camadas (veja acima) | `auxiliary.skills_hub` |
| Helpers MCP | Em camadas (veja acima) | `auxiliary.mcp` |
| Classificação de aprovação | Em camadas (veja acima) | `auxiliary.approval` |
| Geração de título | Em camadas (veja acima) | `auxiliary.title_generation` |
| Triage specifier | Em camadas (veja acima) | `auxiliary.triage_specifier` |
| Delegação | Só override de provider (sem fallback automático) | `delegation.provider` / `delegation.model` |
| Jobs cron | Só override de provider por job (sem fallback automático) | `provider` / `model` por job |
