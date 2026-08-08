---
sidebar_position: 4
title: "Provedores de memória"
description: "Plugins de provedor de memória externo — Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory"
---

# Provedores de memória

O Hermes Agent inclui 8 plugins de provedor de memória externo que dão ao agente conhecimento persistente cross-session além de MEMORY.md e USER.md built-in. Apenas **um** provedor externo pode estar ativo por vez — a memória built-in está sempre ativa junto.

## Início rápido {#quick-start}

```bash
hermes memory setup      # interactive picker + configuration
hermes memory status     # check what's active
hermes memory off        # disable external provider
```

Você também pode selecionar o provedor de memória ativo via `hermes plugins` → Provider Plugins → Memory Provider.

Ou defina manualmente em `~/.hermes/config.yaml`:

```yaml
memory:
  provider: openviking   # or honcho, mem0, hindsight, holographic, retaindb, byterover, supermemory
```

## Como funciona {#how-it-works}

Quando um provedor de memória está ativo, o Hermes automaticamente:

1. **Injeta contexto do provedor** no system prompt (o que o provedor sabe)
2. **Prefetch de memórias relevantes** antes de cada turno (background, non-blocking)
3. **Sincroniza turnos de conversa** com o provedor após cada resposta
4. **Extrai memórias no fim da sessão** (para provedores que suportam)
5. **Espelha escritas de memória built-in** no provedor externo
6. **Adiciona ferramentas específicas do provedor** para o agente buscar, armazenar e gerenciar memórias

A memória built-in (MEMORY.md / USER.md) continua funcionando exatamente como antes. O provedor externo é aditivo.

## Provedores disponíveis {#available-providers}

### Honcho

Modelagem cross-session de usuário nativa de IA com raciocínio dialético, injeção de contexto com escopo de sessão, busca semântica e conclusões persistentes. O contexto base agora inclui o resumo da sessão junto com representação do usuário e peer cards, dando ao agente consciência do que já foi discutido.

| | |
|---|---|
| **Melhor para** | Sistemas multi-agente com contexto cross-session, alinhamento usuário-agente |
| **Requer** | `pip install honcho-ai` + [API key](https://app.honcho.dev) ou instância self-hosted |
| **Armazenamento de dados** | Honcho Cloud ou self-hosted |
| **Custo** | Preços Honcho (cloud) / grátis (self-hosted) |

**Tools (5):** `honcho_profile` (ler/atualizar peer card), `honcho_search` (busca semântica), `honcho_context` (contexto de sessão — summary, representation, card, messages), `honcho_reasoning` (sintetizado por LLM), `honcho_conclude` (criar/excluir conclusions)

**Arquitetura:** Injeção de contexto em duas camadas — uma camada base (resumo de sessão + representação + peer card, atualizada em `contextCadence`) mais um suplemento dialético (raciocínio LLM, atualizado em `dialecticCadence`). A dialética seleciona automaticamente prompts cold-start (fatos gerais do usuário) vs. warm prompts (contexto com escopo de sessão) com base em se existe contexto base.

**Três knobs de config ortogonais** controlam custo e profundidade independentemente:

- `contextCadence` — com que frequência a camada base atualiza (frequência de chamadas API)
- `dialecticCadence` — com que frequência o LLM dialético dispara (frequência de chamadas LLM)
- `dialecticDepth` — quantos passes `.chat()` por invocação dialética (1–3, profundidade de raciocínio)

A dialética auto-injetada também escala seu nível de raciocínio pela extensão da query (query mais longa → raciocínio mais profundo, limitado por `reasoningLevelCap`); veja [Query-Adaptive Reasoning Level](./honcho.md#query-adaptive-reasoning-level).

**Setup Wizard:**
```bash
hermes memory setup        # select "honcho" — executa post-setup específico do Honcho
```

O comando legado `hermes honcho setup` ainda funciona (agora redireciona para `hermes memory setup`), mas só é registrado depois que Honcho é selecionado como provedor de memória ativo.

**Config:** `$HERMES_HOME/honcho.json` (local ao profile) ou `~/.honcho/config.json` (global). Ordem de resolução: `$HERMES_HOME/honcho.json` > `~/.hermes/honcho.json` > `~/.honcho/config.json`. Veja a [referência de config](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md) e o [guia de integração Honcho](https://docs.honcho.dev/v3/guides/integrations/hermes).

<details>
<summary>Referência completa de config</summary>

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `apiKey` | -- | API key de [app.honcho.dev](https://app.honcho.dev) |
| `baseUrl` | -- | Base URL para Honcho self-hosted |
| `peerName` | -- | Identidade do peer usuário |
| `aiPeer` | host key | Identidade do peer AI (uma por profile) |
| `workspace` | host key | ID de workspace compartilhado |
| `contextTokens` | `null` (sem limite) | Budget de tokens para contexto auto-injetado por turno. Trunca em limites de palavra |
| `contextCadence` | `1` | Turnos mínimos entre chamadas API `context()` (refresh da camada base) |
| `dialecticCadence` | `2` | Turnos mínimos entre chamadas LLM `peer.chat()`. Recomendado 1–5. Só em modos `hybrid`/`context` |
| `dialecticDepth` | `1` | Número de passes `.chat()` por invocação dialética. Limitado 1–3. Pass 0: prompt cold/warm, pass 1: self-audit, pass 2: reconciliation |
| `dialecticDepthLevels` | `null` | Array opcional de níveis de raciocínio por pass, ex. `["minimal", "low", "medium"]`. Sobrescreve defaults proporcionais |
| `dialecticReasoningLevel` | `'low'` | Nível base de raciocínio: `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | Quando `true`, modelo pode sobrescrever nível de raciocínio por chamada via param de ferramenta |
| `dialecticMaxChars` | `600` | Máx. chars de resultado dialético injetados no system prompt |
| `recallMode` | `'hybrid'` | `hybrid` (auto-inject + tools), `context` (só inject), `tools` (só tools) |
| `writeFrequency` | `'async'` | Quando flush de mensagens: `async` (thread background), `turn` (sync), `session` (batch no fim), ou inteiro N |
| `saveMessages` | `true` | Se persiste mensagens na API Honcho |
| `observationMode` | `'directional'` | `directional` (tudo on) ou `unified` (pool compartilhado). Sobrescreva com objeto `observation` |
| `messageMaxChars` | `25000` | Máx. chars por mensagem (chunked se exceder) |
| `dialecticMaxInputChars` | `10000` | Máx. chars de input de query dialética para `peer.chat()` |
| `sessionStrategy` | `'per-directory'` | `per-directory`, `per-repo`, `per-session`, `global` |
| `pinUserPeer` | `false` | Só gateway. Quando `true`, todo usuário gateway não-agent colapsa em `peerName`; o pin sobrescreve todos os aliases |
| `userPeerAliases` | `{}` | Só gateway. Mapeia runtime IDs para peers (`{"7654321": "alice"}`). Many-to-one |
| `runtimePeerPrefix` | `""` | Só gateway. Namespace para runtime IDs desconhecidos (`telegram_7654321`) quando nenhum alias bate |

</details>

<details>
<summary>honcho.json mínimo (cloud)</summary>

```json
{
  "apiKey": "your-key-from-app.honcho.dev",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

<details>
<summary>honcho.json mínimo (self-hosted)</summary>

```json
{
  "baseUrl": "http://localhost:8000",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "peerName": "your-name",
      "workspace": "hermes"
    }
  }
}
```

</details>

:::tip Migrando de `hermes honcho`
Se você usava `hermes honcho setup`, sua config e todos os dados server-side estão intactos. Reative pelo setup wizard ou defina manualmente `memory.provider: honcho` para reativar pelo novo sistema.
:::

**Setup multi-peer:**

Honcho modela conversas como peers trocando mensagens — um peer usuário mais um peer AI por profile Hermes, todos compartilhando workspace. O workspace é o ambiente compartilhado: o peer usuário é global entre profiles, cada peer AI tem identidade própria. Cada peer AI constrói representação/card independente das próprias observações, então profile `coder` fica orientado a código enquanto `writer` fica editorial com o mesmo usuário.

O mapeamento:

| Conceito | O que é |
|---------|-----------|
| **Workspace** | Ambiente compartilhado. Todos os profiles Hermes em um workspace veem a mesma identidade de usuário. |
| **User peer** (`peerName`) | O humano. Compartilhado entre profiles no workspace. |
| **AI peer** (`aiPeer`) | Um por profile Hermes. Host key `hermes` → padrão; `hermes.<profile>` para outros. |
| **Observation** | Toggles por peer controlando o que Honcho modela de quais mensagens. `directional` (padrão, quatro on) ou `unified` (pool single-observer). |

### Novo profile, peer Honcho fresco

```bash
hermes profile create coder --clone
```

`--clone` cria bloco host `hermes.coder` em `honcho.json` com `aiPeer: "coder"`, `workspace` compartilhado, `peerName`, `recallMode`, `writeFrequency`, `observation` herdados, etc. O peer AI é criado eagerly no Honcho antes da primeira mensagem.

### Profiles existentes, backfill de peers Honcho

```bash
hermes honcho sync
```

Escaneia todo profile Hermes, cria blocos host para profiles sem um, herda settings do bloco `hermes` padrão e cria peers AI eagerly. Idempotente — pula profiles que já têm bloco host.

### Observação por profile

Cada bloco host pode sobrescrever config de observation independentemente. Exemplo: profile focado em código onde o peer AI observa o usuário mas não faz self-model:

```json
"hermes.coder": {
  "aiPeer": "coder",
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": false, "observeOthers": true }
  }
}
```

**Toggles de observation (um conjunto por peer):**

| Toggle | Efeito |
|--------|--------|
| `observeMe` | Honcho constrói representação deste peer das próprias mensagens |
| `observeOthers` | Este peer observa mensagens do outro peer (alimenta raciocínio cross-peer) |

Predefinições via `observationMode`:

- **`"directional"`** (padrão) — quatro flags on. Observação mútua completa; habilita dialética cross-peer.
- **`"unified"`** — user `observeMe: true`, AI `observeOthers: true`, resto false. Pool single-observer; AI modela usuário mas não a si, peer usuário só self-models.

Toggles server-side definidos no [dashboard Honcho](https://app.honcho.dev) vencem defaults locais — sincronizados no init da sessão.

Veja a [página Honcho](./honcho.md#observation-directional-vs-unified) para referência completa de observation.

### Mapeamento de identidade no gateway

O modelo de peers acima cobre sessões CLI, TUI e desktop, onde toda conversa resolve para `peerName`. O [gateway](../../developer-guide/gateway-internals.md) adiciona segundo eixo: usuários chegam com runtime IDs nativos da plataforma (Telegram UID, Discord snowflake, Slack user), e três chaves decidem para qual peer cada ID resolve.

| Chave | Efeito |
|-----|--------|
| `pinUserPeer: true` | Todo usuário gateway não-agent colapsa em `peerName`. O pin é checado primeiro e sobrescreve aliases — use só quando nenhuma identidade user-side precisa de peer próprio |
| `userPeerAliases` | Mapeia runtime IDs específicos para peers (`{"7654321": "alice"}`). Lar do roteamento de identidades distintas — incluindo agents com peer próprio |
| `runtimePeerPrefix` | Namespace para runtime ID não mapeado (`telegram_7654321`) para plataformas com IDs de forma igual não colidirem |

Fora do gateway essas chaves não fazem nada. `hermes memory setup` só pergunta por elas quando detecta plataforma gateway conectada. Veja a [página Honcho](./honcho.md#gateway-identity-mapping) para a ladder de resolução e o fluxo de setup.

<details>
<summary>Exemplo completo honcho.json (multi-profile)</summary>

```json
{
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "dialecticReasoningLevel": "low",
      "dialecticDynamic": true,
      "dialecticCadence": 2,
      "dialecticDepth": 1,
      "dialecticMaxChars": 600,
      "contextCadence": 1,
      "messageMaxChars": 25000,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "tools",
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    },
    "hermes.writer": {
      "enabled": true,
      "aiPeer": "writer",
      "workspace": "hermes",
      "peerName": "eri"
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```

</details>

Veja a [referência de config](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/honcho/README.md) e o [guia de integração Honcho](https://docs.honcho.dev/v3/guides/integrations/hermes).


---

### OpenViking

Banco de contexto da Volcengine (ByteDance) com hierarquia de conhecimento estilo filesystem, recuperação em camadas e extração automática de memória em 6 categorias.

| | |
|---|---|
| **Melhor para** | Gerenciamento de conhecimento self-hosted com navegação estruturada |
| **Requer** | `pip install openviking` + servidor em execução |
| **Armazenamento de dados** | Self-hosted (local ou cloud) |
| **Custo** | Grátis (open-source, AGPL-3.0) |

**Tools:** `viking_search` (busca semântica), `viking_read` (em camadas: abstract/overview/full), `viking_browse` (navegação filesystem), `viking_remember` (armazenar fatos), `viking_add_resource` (ingest de URLs/docs)

**Setup:**
```bash
# Start the OpenViking server first
pip install openviking
openviking-server

# Then configure Hermes
hermes memory setup    # select "openviking"
# Or manually:
hermes config set memory.provider openviking
echo "OPENVIKING_ENDPOINT=http://localhost:1933" >> ~/.hermes/.env
# Authenticated servers should use a user/admin API key:
echo "OPENVIKING_API_KEY=..." >> ~/.hermes/.env
```

**Recursos principais:**
- Carregamento de contexto em camadas: L0 (~100 tokens) → L1 (~2k) → L2 (full)
- Extração automática de memória no commit de sessão (profile, preferences, entities, events, cases, patterns)
- Esquema URI `viking://` para navegação hierárquica de conhecimento

`OPENVIKING_ACCOUNT` e `OPENVIKING_USER` são usados para modo local/trusted.
`OPENVIKING_AGENT` é o peer ID do Hermes no OpenViking para memórias com escopo de peer.

---

### Mem0

Extração de fatos LLM server-side com busca semântica, reranking e deduplicação automática. Três modos de conexão: **Platform** (Mem0 Cloud), **self-hosted dashboard** (servidor Mem0 que você roda via Docker) e **OSS** (Mem0 in-process com seu próprio LLM + vector store).

| | |
|---|---|
| **Melhor para** | Gestão hands-off de memória — Mem0 trata extração automaticamente |
| **Requer** | `pip install mem0ai` + API key (platform), servidor Mem0 rodando (dashboard self-hosted), ou LLM + vector store (OSS) |
| **Armazenamento de dados** | Mem0 Cloud (platform), seu servidor Mem0 (dashboard self-hosted), ou in-process (OSS) |
| **Custo** | Preços Mem0 (platform) / grátis (self-hosted ou OSS) |

**Tools (4):** `mem0_search` (busca semântica; reranking opcional no platform mode, off por padrão), `mem0_add` (armazenar fatos verbatim), `mem0_update` (atualizar por ID), `mem0_delete` (excluir por ID)

**Setup (Platform):**
```bash
hermes memory setup    # select "mem0" → "Platform"
# Or manually:
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

**Setup (OSS):**
```bash
hermes memory setup    # select "mem0" → "Open Source (self-hosted)"
# Or via flags:
hermes memory setup mem0 --mode oss --oss-llm openai --oss-llm-key sk-... --oss-vector qdrant
```

Preview sem escrever arquivos:
```bash
hermes memory setup mem0 --mode oss --oss-llm-key sk-... --dry-run
```

**Setup (Self-Hosted Dashboard):** conecte a um servidor Mem0 que você roda via Docker (REST API do dashboard):

```bash
hermes memory setup    # select "mem0" → "Self-hosted server"
# Or via flags:
hermes memory setup mem0 --mode selfhosted --host http://localhost:8888 --api-key your-admin-api-key
```

Ou configure manualmente — como env vars:

```bash
echo "MEM0_HOST=http://localhost:8888" >> ~/.hermes/.env
echo "MEM0_API_KEY=your-admin-api-key" >> ~/.hermes/.env
```

ou em `mem0.json`:

```json
{ "host": "http://localhost:8888", "api_key": "your-admin-api-key" }
```

O plugin autentica com `X-API-Key` e usa rotas `/search` / `/memories` do servidor. `api_key` é opcional (omita só em servidores `AUTH_DISABLED`). Não defina `mode: oss` — tem precedência sobre `host`.

**Config:** `$HERMES_HOME/mem0.json` (settings comportamentais). Só o secret `MEM0_API_KEY` vai em `~/.hermes/.env`.

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `mode` | `platform` | `platform` (Mem0 Cloud) ou `oss` (self-managed, in-process) |
| `host` | — | URL do servidor Mem0 self-hosted (dashboard Docker). Roteia via HTTP com `X-API-Key`; não combine com `mode: oss` |
| `user_id` | `hermes-user` | Identificador de usuário |
| `agent_id` | `hermes` | Identificador de agent |
| `rerank` | `false` | Rerank de resultados de busca por relevância (somente platform mode) |

**Provedores suportados no OSS:**

| Componente | Provedores |
|-----------|-----------|
| LLM | openai, ollama |
| Embedder | openai, ollama |
| Vector Store | qdrant (local/server), pgvector |

**Trocar modos:** Reexecute `hermes memory setup mem0 --mode <platform|selfhosted|oss>` ou edite `mem0.json` diretamente.

---

### Hindsight

Memória de longo prazo com knowledge graph, resolução de entidades e recuperação multi-estratégia. A ferramenta `hindsight_reflect` fornece síntese cross-memory que nenhum outro provedor oferece. Retém automaticamente turnos completos de conversa (incluindo tool calls) com rastreamento de documento por sessão.

| | |
|---|---|
| **Melhor para** | Recall baseado em knowledge graph com relações de entidades |
| **Requer** | Cloud: API key de [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io). Local: API key LLM (OpenAI, Groq, OpenRouter, etc.) |
| **Armazenamento de dados** | Hindsight Cloud ou PostgreSQL embedded local |
| **Custo** | Preços Hindsight (cloud) ou grátis (local) |

**Tools:** `hindsight_retain` (armazenar com extração de entidades), `hindsight_recall` (busca multi-estratégia), `hindsight_reflect` (síntese cross-memory)

**Setup:**
```bash
hermes memory setup    # select "hindsight"
# Or manually:
hermes config set memory.provider hindsight
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
```

O setup wizard instala dependências automaticamente e só o necessário para o modo selecionado (`hindsight-client` para cloud, `hindsight-all` para local). Requer `hindsight-client >= 0.4.22` (auto-atualizado no início da sessão se desatualizado).

**UI modo local:** `hindsight-embed -p hermes ui start`

**Config:** `$HERMES_HOME/hindsight/config.json`

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `mode` | `cloud` | `cloud` ou `local` |
| `bank_id` | `hermes` | Identificador do memory bank |
| `recall_budget` | `mid` | Profundidade de recall: `low` / `mid` / `high` |
| `memory_mode` | `hybrid` | `hybrid` (context + tools), `context` (só auto-inject), `tools` (só tools) |
| `auto_retain` | `true` | Retém turnos de conversa automaticamente |
| `auto_recall` | `true` | Recall automático de memórias antes de cada turno |
| `retain_async` | `true` | Processa retain assincronamente no servidor |
| `retain_context` | `conversation between Hermes Agent and the User` | Label de contexto para memórias retidas |
| `retain_tags` | — | Tags padrão em memórias retidas; mescladas com tags de ferramenta por chamada |
| `retain_source` | — | `metadata.source` opcional anexado a memórias retidas |
| `retain_user_prefix` | `User` | Label antes de turnos de usuário em transcripts auto-retidos |
| `retain_assistant_prefix` | `Assistant` | Label antes de turnos de assistant em transcripts auto-retidos |
| `recall_tags` | — | Tags para filtrar no recall |

Veja [README do plugin](https://github.com/NousResearch/hermes-agent/blob/main/plugins/memory/hindsight/README.md) para referência completa de configuração.

---

### Holographic

Fact store SQLite local com busca full-text FTS5, trust scoring e HRR (Holographic Reduced Representations) para queries algébricas composicionais.

| | |
|---|---|
| **Melhor para** | Memória só local com retrieval avançado, sem dependências externas |
| **Requer** | Nada (SQLite sempre disponível). NumPy opcional para álgebra HRR. |
| **Armazenamento de dados** | Local SQLite |
| **Custo** | Grátis |

**Tools:** `fact_store` (9 actions: add, search, probe, related, reason, contradict, update, remove, list), `fact_feedback` (rating helpful/unhelpful que treina trust scores)

**Setup:**
```bash
hermes memory setup    # select "holographic"
# Or manually:
hermes config set memory.provider holographic
```

**Config:** `config.yaml` em `plugins.hermes-memory-store`

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | Caminho do banco SQLite |
| `auto_extract` | `false` | Auto-extração de fatos no fim da sessão |
| `default_trust` | `0.5` | Trust score padrão (0.0–1.0) |

**Capacidades únicas:**
- `probe` — recall algébrico por entidade (todos os fatos sobre pessoa/coisa)
- `reason` — queries AND composicionais entre múltiplas entidades
- `contradict` — detecção automatizada de fatos conflitantes
- Trust scoring com feedback assimétrico (+0.05 helpful / -0.10 unhelpful)

---

### RetainDB

API de memória cloud com busca híbrida (Vector + BM25 + Reranking), 7 tipos de memória e delta compression.

| | |
|---|---|
| **Melhor para** | Times que já usam infraestrutura RetainDB |
| **Requer** | Conta RetainDB + API key |
| **Armazenamento de dados** | RetainDB Cloud |
| **Custo** | $20/mês |

**Tools:** `retaindb_profile` (profile de usuário), `retaindb_search` (busca semântica), `retaindb_context` (contexto relevante à tarefa), `retaindb_remember` (armazenar com type + importance), `retaindb_forget` (excluir memórias)

**Setup:**
```bash
hermes memory setup    # select "retaindb"
# Or manually:
hermes config set memory.provider retaindb
echo "RETAINDB_API_KEY=your-key" >> ~/.hermes/.env
```

---

### ByteRover

Memória persistente via CLI `brv` — árvore hierárquica de conhecimento com retrieval em camadas (texto fuzzy → busca driven por LLM). Local-first com sync cloud opcional.

| | |
|---|---|
| **Melhor para** | Desenvolvedores que querem memória portável, local-first, com CLI |
| **Requer** | ByteRover CLI (`npm install -g byterover-cli` ou [script de instalação](https://byterover.dev)) |
| **Armazenamento de dados** | Local (padrão) ou ByteRover Cloud (sync opcional) |
| **Custo** | Grátis (local) ou preços ByteRover (cloud) |

**Tools:** `brv_query` (buscar na árvore de conhecimento), `brv_curate` (armazenar fatos/decisions/patterns), `brv_status` (versão do CLI + stats da árvore)

**Setup:**
```bash
# Install the CLI first
curl -fsSL https://byterover.dev/install.sh | sh

# Then configure Hermes
hermes memory setup    # select "byterover"
# Or manually:
hermes config set memory.provider byterover
```

**Recursos principais:**
- Extração automática pré-compressão (salva insights antes da compressão de contexto descartá-los)
- Árvore de conhecimento em `$HERMES_HOME/byterover/` (escopada ao profile)
- Sync cloud certificado SOC2 Type II (opcional)

---

### Supermemory

Memória semântica de longo prazo com recall de profile, busca semântica, ferramentas explícitas de memória e ingest de conversa no fim da sessão via graph API Supermemory.

| | |
|---|---|
| **Melhor para** | Recall semântico com profiling de usuário e construção de grafo por sessão |
| **Requer** | `pip install supermemory` + [cloud API key](http://app.supermemory.ai/integrations?connect=hermes), ou [servidor self-hosted](https://supermemory.ai/docs/self-hosting/overview) |
| **Armazenamento de dados** | Supermemory Cloud ou self-hosted |
| **Custo** | Preços Supermemory (cloud) / grátis (self-hosted) |

**Tools:** `supermemory_store` (salvar memórias explícitas), `supermemory_search` (busca por similaridade semântica), `supermemory_forget` (esquecer por ID ou query best-match), `supermemory_profile` (profile persistente + contexto recente)

**Setup:**
```bash
hermes memory setup    # select "supermemory"
# Or manually:
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

Setup self-hosted:

```bash
npx supermemory local
```

Antes de rodar `hermes memory setup`, defina `base_url` em
`$HERMES_HOME/supermemory.json`:

```json
{
  "base_url": "http://localhost:6767"
}
```

Depois rode `hermes memory setup` e insira a API key impressa pelo servidor local.
Configurar o endpoint primeiro garante que o probe de conexão do setup também
permaneça local.

**Config:** `$HERMES_HOME/supermemory.json`

| Chave | Padrão | Descrição |
|-----|---------|-------------|
| `base_url` | `https://api.supermemory.ai` | Endpoint API para Supermemory hosted ou self-hosted. Tem prioridade sobre `SUPERMEMORY_BASE_URL`. |
| `container_tag` | `hermes` | Container tag para busca e writes. Suporta template `{identity}` para tags escopadas ao profile. |
| `auto_recall` | `true` | Injeta contexto de memória relevante antes dos turnos |
| `auto_capture` | `true` | Armazena turnos user-assistant limpos após cada resposta |
| `max_recall_results` | `10` | Máx. itens recalled para formatar em contexto |
| `profile_frequency` | `50` | Inclui fatos de profile no primeiro turno e a cada N turnos |
| `capture_mode` | `all` | Pula turnos pequenos ou triviais por padrão |
| `search_mode` | `hybrid` | Modo de busca: `hybrid`, `memories` ou `documents` |
| `api_timeout` | `5.0` | Timeout para requests SDK e ingest |

**Variáveis de ambiente:** `SUPERMEMORY_API_KEY` (obrigatória), `SUPERMEMORY_BASE_URL` (fallback de compatibilidade quando `base_url` não está configurado), `SUPERMEMORY_CONTAINER_TAG` (sobrescreve config).

Precedência de base URL: `supermemory.json` → `SUPERMEMORY_BASE_URL` → `https://api.supermemory.ai`. Operações SDK, probes de setup/status e ingest de conversa usam o endpoint resolvido.

**Recursos principais:**
- Context fencing automático — remove memórias recalled de turnos capturados para evitar poluição recursiva de memória
- Ingest de sessão completa — conversa inteira enviada uma vez nos limites de sessão
- Ingest de conversa no fim da sessão (para `/v4/conversations`) para profile + grafo mais ricos no Supermemory
- Roteamento self-hosted end-to-end — SDK, probe e requests de conversation-ingest usam o mesmo endpoint configurado
- Fatos de profile injetados no primeiro turno e em intervalos configuráveis
- **Containers escopados ao profile** — use `{identity}` em `container_tag` (ex. `hermes-{identity}` → `hermes-coder`) para isolar memórias por profile Hermes
- **Modo multi-container** — habilite `enable_custom_container_tags` com lista `custom_containers` para o agente ler/escrever em containers nomeados. Operações automáticas ficam no container primário.

<details>
<summary>Exemplo multi-container</summary>

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding context."
}
```

</details>

**Suporte:** [Discord](https://supermemory.link/discord) · [support@supermemory.com](mailto:support@supermemory.com)

### Memori

Memória estruturada de longo prazo usando Memori Cloud, com captura de turnos completos em background, contexto de turno tool-aware e ferramentas explícitas de recall para fatos, resumos, quota, signup e feedback.

| | |
|---|---|
| **Melhor para** | Recall controlado pelo agent com atribuição estruturada de projeto e sessão |
| **Requer** | `pip install hermes-memori` + `hermes-memori install` + [API key Memori](https://app.memorilabs.ai/signup) |
| **Armazenamento de dados** | Memori Cloud |
| **Custo** | Preços Memori |

**Tools:** `memori_recall` (buscar memória de longo prazo), `memori_recall_summary` (contexto resumido), `memori_quota` (uso/quota), `memori_signup` (solicitar email de signup), `memori_feedback` (enviar feedback de integração)

**Setup:**
```bash
pip install hermes-memori
hermes-memori install
hermes config set memory.provider memori
hermes memory setup
```

---

## Comparação de provedores {#provider-comparison}

| Provedor | Armazenamento | Custo | Tools | Dependências | Recurso único |
|----------|---------|------|-------|-------------|----------------|
| **Honcho** | Cloud | Pago | 5 | `honcho-ai` | Modelagem dialética de usuário + contexto com escopo de sessão |
| **OpenViking** | Self-hosted | Grátis | 5 | `openviking` + server | Hierarquia filesystem + carregamento em camadas |
| **Mem0** | Cloud/Self-hosted | Grátis/Pago | 4 | `mem0ai` | Extração LLM server-side + modos self-hosted/OSS |
| **Hindsight** | Cloud/Local | Grátis/Pago | 3 | `hindsight-client` | Grafo de conhecimento + síntese reflect |
| **Holographic** | Local | Grátis | 2 | None | Álgebra HRR + trust scoring |
| **RetainDB** | Cloud | $20/mo | 5 | `requests` | Delta compression |
| **ByteRover** | Local/Cloud | Grátis/Pago | 3 | `brv` CLI | Extração pré-compressão |
| **Supermemory** | Cloud/Self-hosted | Grátis/Pago | 4 | `supermemory` | Context fencing + ingest de grafo de sessão + multi-container |
| **Memori** | Cloud | Grátis/Pago | 5 | `hermes-memori` | Memória tool-aware + recall estruturado |

## Isolamento de perfil {#profile-isolation}

Os dados de cada provedor são isolados por [perfil](/user-guide/profiles):

- **Provedores de armazenamento local** (Holographic, ByteRover) usam caminhos `$HERMES_HOME/` que diferem por perfil
- **Provedores com arquivo de config** (Honcho, Mem0, Hindsight, Supermemory) armazenam config em `$HERMES_HOME/` para cada perfil ter suas próprias credenciais
- **Provedores cloud** (RetainDB) derivam automaticamente nomes de projeto com escopo de perfil
- **Provedores via env var** (OpenViking) são configurados via `.env` de cada perfil

## Construindo um provedor de memória {#building-a-memory-provider}

Veja o [Guia do desenvolvedor: plugins de provedor de memória](/developer-guide/memory-provider-plugin) para criar o seu.
