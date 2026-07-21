---
sidebar_position: 99
title: "Memória Honcho"
description: "Memória persistente nativa de IA via Honcho — raciocínio dialético, modelagem multiagente de usuário e personalização profunda"
---

# Memória Honcho {#honcho-memory}

[Honcho](https://github.com/plastic-labs/honcho) é um backend de memória nativo de IA que adiciona raciocínio dialético e modelagem profunda de usuário sobre o sistema de memória built-in do Hermes. Em vez de armazenamento simples chave-valor, o Honcho mantém um model em execução de quem o usuário é — preferências, estilo de comunicação, objetivos e padrões — raciocinando sobre conversas depois que acontecem.

:::info Honcho é um plugin Memory Provider
Honcho está integrado ao sistema [Memory Providers](./memory-providers.md). Todos os recursos abaixo estão disponíveis pela interface unificada de memory provider.
:::

## O que o Honcho adiciona {#what-honcho-adds}

| Capacidade | Memória built-in | Honcho |
|-----------|----------------|--------|
| Persistência entre sessões | ✔ File-based MEMORY.md/USER.md | ✔ Server-side com API |
| Perfil de usuário | ✔ Curadoria manual do agente | ✔ Raciocínio dialético automático |
| Resumo de sessão | — | ✔ Injeção de contexto scoped à sessão |
| Isolamento multiagente | — | ✔ Separação de perfil por peer |
| Modos de observação | — | ✔ Observação unified ou directional |
| Conclusions (insights derivados) | — | ✔ Raciocínio server-side sobre padrões |
| Busca no histórico | ✔ Busca FTS5 de sessão | ✔ Busca semântica sobre conclusions |

**Raciocínio dialético**: Após cada turn de conversa (gated por `dialecticCadence`), o Honcho analisa a troca e deriva insights sobre preferências, hábitos e objetivos do usuário. Eles se acumulam ao longo do tempo, dando ao agente um entendimento cada vez mais profundo além do que o usuário declarou explicitamente. A dialética suporta profundidade multi-pass (1–3 passes) com seleção automática de prompt cold/warm — queries cold start focam em fatos gerais do usuário enquanto queries warm priorizam contexto scoped à sessão.

**Contexto scoped à sessão**: O contexto base agora inclui o resumo de sessão junto com a representação do usuário e o peer card. Isso dá ao agente consciência do que já foi discutido na sessão atual, reduzindo repetição e habilitando continuidade.

**Perfis multiagente**: Quando múltiplas instâncias Hermes falam com o mesmo usuário (ex.: assistente de código e assistente pessoal), o Honcho mantém perfis "peer" separados. Cada peer vê apenas suas próprias observations e conclusions, evitando contaminação cruzada de contexto.

## Setup {#setup}

```bash
hermes memory setup    # select "honcho" from the provider list
```

Ou configure manualmente:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo 'HONCHO_API_KEY=***' >> ~/.hermes/.env
```

Obtenha uma API key em [honcho.dev](https://honcho.dev).

## Arquitetura {#architecture}

### Injeção de contexto em duas camadas {#two-layer-context-injection}

A cada turn (em modo `hybrid` ou `context`), o Honcho monta duas camadas de contexto injetadas no system prompt:

1. **Contexto base** — resumo de sessão, representação do usuário, user peer card, self-representation da IA e AI identity card. Atualizado em `contextCadence`. Esta é a camada "quem é este usuário".
2. **Suplemento dialético** — raciocínio sintetizado por LLM sobre o estado e necessidades atuais do usuário. Atualizado em `dialecticCadence`. Esta é a camada "o que importa agora".

Ambas as camadas são concatenadas e truncadas ao orçamento `contextTokens` (se definido).

### Seleção de prompt cold/warm {#coldwarm-prompt-selection}

A dialética seleciona automaticamente entre duas estratégias de prompt:

- **Cold start** (ainda sem contexto base): Query geral — "Who is this person? What are their preferences, goals, and working style?"
- **Warm session** (contexto base existe): Query scoped à sessão — "Given what's been discussed in this session so far, what context about this user is most relevant?"

Isso acontece automaticamente com base em se o contexto base já foi populado.

### Três knobs de config ortogonais {#three-orthogonal-config-knobs}

Custo e profundidade são controlados por três knobs independentes:

| Knob | Controla | Padrão |
|------|----------|---------|
| `contextCadence` | Turns entre chamadas API `context()` (refresh da camada base) | `1` |
| `dialecticCadence` | Turns entre chamadas LLM `peer.chat()` (refresh da camada dialética) | `2` (recomendado 1–5) |
| `dialecticDepth` | Número de passes `.chat()` por invocação dialética (1–3) | `1` |

São ortogonais — você pode ter refreshes frequentes de contexto com dialética infrequente, ou dialética multi-pass profunda em baixa frequência. Exemplo: `contextCadence: 1, dialecticCadence: 5, dialecticDepth: 2` refresca contexto base a cada turn, roda dialética a cada 5 turns, e cada run dialética faz 2 passes.

### Profundidade dialética (multi-pass) {#dialectic-depth-multi-pass}

Quando `dialecticDepth` > 1, cada invocação dialética roda múltiplos passes `.chat()`:

- **Pass 0**: Prompt cold ou warm (veja acima)
- **Pass 1**: Self-audit — identifica lacunas na avaliação inicial e sintetiza evidência de sessões recentes
- **Pass 2**: Reconciliation — checa contradições entre passes anteriores e produz síntese final

Cada pass usa um nível de reasoning proporcional (passes iniciais mais leves, nível base no pass principal). Override por pass com `dialecticDepthLevels` — ex.: `["minimal", "medium", "high"]` para run depth-3.

Passes saem cedo se o pass anterior retornou sinal forte (saída longa e estruturada), então depth 3 nem sempre significa 3 chamadas LLM.

### Prewarm no início da sessão {#session-start-prewarm}

Na init de sessão, o Honcho dispara uma chamada dialética em background na `dialecticDepth` configurada completa e entrega o resultado diretamente à montagem de contexto do turn 1. Um prewarm single-pass em peer cold frequentemente retorna saída fina — depth multi-pass roda o ciclo audit/reconcile antes do usuário falar. Se o prewarm não chegou até o turn 1, o turn 1 cai para chamada síncrona com timeout limitado.

### Nível de reasoning adaptativo à query {#query-adaptive-reasoning-level}

A dialética auto-injetada escala `dialecticReasoningLevel` pelo comprimento da query: +1 nível em ≥120 chars, +2 em ≥400, clamped em `reasoningLevelCap` (padrão `"high"`). Desabilite com `reasoningHeuristic: false` para fixar toda chamada auto em `dialecticReasoningLevel`. Níveis disponíveis: `minimal`, `low`, `medium`, `high`, `max`.

## Opções de configuração {#configuration-options}

Honcho é configurado em `~/.honcho/config.json` (global) ou `$HERMES_HOME/honcho.json` (profile-local). O setup wizard cuida disso para você.

### Honcho self-hosted com autenticação {#self-hosted-honcho-with-authentication}

Ao apontar o Hermes para um servidor Honcho self-hosted, `hermes honcho setup` (e `hermes memory setup`) pedem um **token JWT / bearer local** após a URL base. Cole um JWT assinado com o `AUTH_JWT_SECRET` do servidor (env var do compose Honcho) para habilitar acesso autenticado; deixe em branco para servidores rodando com `AUTH_USE_AUTH=false`. O token local fica no bloco host (`hosts.<host>.apiKey` em `honcho.json`), separado de qualquer `apiKey` cloud, para você poder voltar o prompt `Cloud or local?` para `cloud` depois sem perder nenhuma credencial.

### Referência completa de config {#full-config-reference}

| Key | Padrão | Descrição |
|-----|---------|-------------|
| `contextTokens` | `null` (uncapped) | Orçamento de tokens para contexto auto-injetado por turn. Defina inteiro (ex. 1200) para limitar. Trunca em limites de palavra |
| `contextCadence` | `1` | Mínimo de turns entre chamadas `context()` (refresh camada base) |
| `dialecticCadence` | `2` | Mínimo de turns entre chamadas LLM `peer.chat()` (camada dialética). Recomendado 1–5. Em modo `tools`, irrelevante — model chama explicitamente |
| `dialecticDepth` | `1` | Número de passes `.chat()` por invocação dialética. Clamped a 1–3 |
| `dialecticDepthLevels` | `null` | Array opcional de níveis de reasoning por pass, ex. `["minimal", "low", "medium"]`. Override defaults proporcionais |
| `dialecticReasoningLevel` | `'low'` | Nível base de reasoning: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | Quando `true`, model pode override nível de reasoning por chamada via param de ferramenta |
| `dialecticMaxChars` | `600` | Max chars do resultado dialético injetado no system prompt |
| `recallMode` | `'hybrid'` | `hybrid` (auto-inject + tools), `context` (inject only), `tools` (tools only) |
| `writeFrequency` | `'async'` | Quando flush mensagens: `async` (thread background), `turn` (sync), `session` (batch no fim), ou inteiro N |
| `saveMessages` | `true` | Se persiste mensagens na API Honcho |
| `observationMode` | `'directional'` | `directional` (tudo on) ou `unified` (pool compartilhado). Override com objeto `observation` para controle granular |
| `messageMaxChars` | `25000` | Max chars por mensagem enviada via `add_messages()`. Chunked se exceder |
| `dialecticMaxInputChars` | `10000` | Max chars para input dialético em `peer.chat()` |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, ou `global` |
| `pinUserPeer` | `false` | Gateway only. Quando `true`, todo usuário gateway não-agent colapsa para `peerName` |
| `userPeerAliases` | `{}` | Gateway only. Map de runtime IDs para peers (`{"7654321": "alice"}`). Many-to-one |
| `runtimePeerPrefix` | `""` | Gateway only. Namespace runtime IDs desconhecidos (`telegram_7654321`) quando nenhum alias combina |

**Session strategy** controla como sessões Honcho mapeiam para seu trabalho:
- `per-session` — cada run `hermes` recebe sessão fresh. Starts limpos, memória via tools. Recomendado para novos usuários.
- `per-directory` — uma sessão Honcho por diretório de trabalho. Contexto acumula entre runs.
- `per-repo` — uma sessão por repositório git.
- `global` — sessão única em todos os diretórios.

**Recall mode** controla como memória flui para conversas:
- `hybrid` — contexto auto-injetado no system prompt E tools disponíveis (model decide quando consultar).
- `context` — só auto-injection, tools ocultas.
- `tools` — só tools, sem auto-injection. Agent deve chamar explicitamente `honcho_reasoning`, `honcho_search`, etc.

**Settings por recall mode:**

| Setting | `hybrid` | `context` | `tools` |
|---------|----------|-----------|---------|
| `writeFrequency` | flush mensagens | flush mensagens | flush mensagens |
| `contextCadence` | gates refresh contexto base | gates refresh contexto base | irrelevante — sem injection |
| `dialecticCadence` | gates chamadas LLM auto | gates chamadas LLM auto | irrelevante — model chama explicitamente |
| `dialecticDepth` | multi-pass por invocação | multi-pass por invocação | irrelevante — model chama explicitamente |
| `contextTokens` | caps injection | caps injection | irrelevante — sem injection |
| `dialecticDynamic` | gates override do model | N/A (sem tools) | gates override do model |

Em modo `tools`, o model está totalmente no controle — chama `honcho_reasoning` quando quer, em qualquer `reasoning_level` que escolher. Settings de cadence e budget só se aplicam a modos com auto-injection (`hybrid` e `context`).

## Mapeamento de identidade no gateway {#gateway-identity-mapping}

Essas settings só importam quando você roda o [gateway Hermes](../../developer-guide/gateway-internals.md) — o único entrypoint onde usuários chegam com runtime IDs nativos de plataforma (UID Telegram, snowflake Discord, user Slack). CLI, TUI e sessões desktop não têm runtime ID e sempre resolvem para `peerName`, então off-gateway essas keys não fazem nada.

O setup wizard detecta se uma plataforma gateway está conectada e pula este passo inteiramente se não. Quando roda, faz uma pergunta — *quem fala com este gateway?* — e deriva as keys:

| Resposta | Resultado |
|--------|--------|
| **just me** | `pinUserPeer: true` — todo usuário gateway não-agent colapsa para seu peer. Pin override todos os aliases, então escolha isso só quando nenhuma identidade user-side precisa de peer próprio. Se agents separados alcançam o gateway e cada um precisa de peer distinto, **não** pin — deixe `pinUserPeer: false` e mapeie via `userPeerAliases` (editor `[e]`) |
| **me + other people** (pooled) | `pinUserPeer: false` + `userPeerAliases` mapeando seus runtime IDs para `peerName` — você fica no histórico compartilhado, outros recebem peers próprios |
| **only other people** | `pinUserPeer: false`, `runtimePeerPrefix` opcional — cada usuário recebe peer próprio |

Escolha `[e]` no prompt para definir as três keys diretamente.

O resolver tenta as keys de cima para baixo, first match wins: `pinUserPeer` → `userPeerAliases[id]` → `runtimePeerPrefix + id` → raw runtime ID → `peerName` → fallback session-key.

:::warning Un-pinning orphan pooled memory
Mudar `pinUserPeer` de `true` para `false` não migra dados — memória acumulada sob `peerName` fica lá, e usuários de plataforma resolvem para peers fresh e vazios. Para manter sua continuidade, escolha o caminho **pooled** para seus runtime IDs aliasarem de volta para `peerName`. O wizard oferece esse steer automaticamente quando detecta a transição.
:::

:::note Key deprecated
`pinPeerName` é alias legacy de `pinUserPeer` — ainda lido para back-compat (`pinUserPeer` vence onde ambos estão set), nunca escrito. Re-rodar setup migra para a key canônica.
:::

## Observation (directional vs. unified) {#observation-directional-vs-unified}

Honcho modela uma conversa como peers trocando mensagens. Cada peer tem dois toggles de observation que mapeiam 1:1 para `SessionPeerConfig` do Honcho:

| Toggle | Efeito |
|--------|--------|
| `observeMe` | Honcho constrói representação deste peer a partir de suas próprias mensagens |
| `observeOthers` | Este peer observa mensagens do outro peer (alimenta reasoning cross-peer) |

Dois peers × dois toggles = quatro flags. `observationMode` é preset shorthand:

| Preset | User flags | AI flags | Semântica |
|--------|-----------|----------|-----------|
| `"directional"` (padrão) | me: on, others: on | me: on, others: on | Observação mútua completa. Habilita dialética cross-peer — "o que a IA sabe sobre o usuário, com base no que o usuário disse e a IA respondeu." |
| `"unified"` | me: on, others: off | me: off, others: on | Semântica shared-pool — a IA observa só mensagens do usuário, user peer só self-models. Pool single-observer. |

Override o preset com bloco `observation` explícito para controle por peer:

```json
"observation": {
  "user": { "observeMe": true,  "observeOthers": true },
  "ai":   { "observeMe": true,  "observeOthers": false }
}
```

Padrões comuns:

| Intent | Config |
|--------|--------|
| Observação completa (maioria dos usuários) | `"observationMode": "directional"` |
| IA não deve re-modelar usuário a partir de suas próprias replies | `"ai": {"observeMe": true, "observeOthers": false}` |
| Persona forte que o peer IA não deve atualizar por self-observation | `"ai": {"observeMe": false, "observeOthers": true}` |

Toggles server-side set via [dashboard Honcho](https://app.honcho.dev) vencem defaults locais — Hermes sincroniza de volta na init de sessão.

## Ferramentas {#tools}

Quando Honcho está ativo como memory provider, cinco ferramentas ficam disponíveis:

| Tool | Propósito |
|------|---------|
| `honcho_profile` | Ler ou atualizar peer card — passe `card` (lista de fatos) para atualizar, omita para ler |
| `honcho_search` | Busca semântica sobre contexto — trechos raw, sem síntese LLM |
| `honcho_context` | Contexto completo de sessão — summary, representation, card, mensagens recentes |
| `honcho_reasoning` | Resposta sintetizada pelo LLM do Honcho — passe `reasoning_level` (minimal/low/medium/high/max) para controlar profundidade |
| `honcho_conclude` | Criar ou deletar conclusions — passe `conclusion` para criar, `delete_id` para remover (só PII) |

## Comandos CLI {#cli-commands}

O subcomando `hermes honcho` **só é registrado quando Honcho é o memory provider ativo** (`memory.provider: honcho` em `config.yaml`). Em install fresh, configure Honcho diretamente com `hermes memory setup honcho` (ou rode `hermes memory setup` e escolha da lista); o subcomando `hermes honcho` aparece na próxima invocação.

```bash
hermes memory setup honcho    # Configure Honcho directly (works before activation)
hermes honcho status          # Connection status, config, and key settings
hermes honcho setup           # Redirects to `hermes memory setup` (post-activation alias)
hermes honcho strategy        # Show or set session strategy (per-session/per-directory/per-repo/global)
hermes honcho peer            # Show or update peer names + dialectic reasoning level
hermes honcho mode            # Show or set recall mode (hybrid/context/tools)
hermes honcho tokens          # Show or set token budget for context and dialectic
hermes honcho identity        # Seed or show the AI peer's Honcho identity
hermes honcho sync            # Sync Honcho config to all existing profiles
hermes honcho peers           # Show peer identities across all profiles
hermes honcho sessions        # List known Honcho session mappings
hermes honcho map             # Map current directory to a Honcho session name
hermes honcho enable          # Enable Honcho for the active profile
hermes honcho disable         # Disable Honcho for the active profile
hermes honcho migrate         # Step-by-step migration guide from openclaw-honcho
```

## Migrando de `hermes honcho` {#migrating-from-hermes-honcho}

Se você usava o standalone `hermes honcho setup`:

1. Sua config existente (`honcho.json` ou `~/.honcho/config.json`) é preservada
2. Seus dados server-side (memories, conclusions, user profiles) estão intactos
3. Defina `memory.provider: honcho` em config.yaml para reativar

Sem re-login ou re-setup. Rode `hermes memory setup` e selecione "honcho" — o wizard detecta sua config existente.

## Documentação completa {#full-documentation}

Veja [Memory Providers — Honcho](./memory-providers.md#honcho) para a referência completa.
