---
title: Pools de credenciais
description: Agrupe várias API keys ou tokens OAuth por provider para rotação automática e recuperação de rate limit.
sidebar_label: Pools de credenciais
sidebar_position: 9
---

# Pools de credenciais {#credential-pools}

Pools de credenciais permitem registrar várias API keys ou tokens OAuth para o mesmo provider. Quando uma key atinge rate limit ou cota de billing, o Hermes rotaciona automaticamente para a próxima key saudável — mantendo sua sessão ativa sem trocar de provider.

Isso é diferente de [fallback providers](./fallback-providers.md), que mudam para um provider *diferente*. Pools de credenciais são rotação no mesmo provider; fallback providers são failover entre providers. Os pools são tentados primeiro — se todas as keys do pool esgotarem, *aí* o fallback provider entra em ação.

:::warning Rotação de key reinicia o prompt cache
Caches de prompt no lado do provider (Anthropic, OpenAI, OpenRouter) são escopados à conta/API key que fez a requisição. Quando o pool rotaciona para uma key diferente no meio da sessão, a nova key não tem prefixo em cache para sua conversa — a próxima requisição relê todo o histórico a preço cheio de input, e rotacionar de volta depois é outra releitura completa, a menos que o TTL de cache da key anterior ainda esteja vivo. A rotação mantém sua sessão rodando, que é o objetivo, mas em conversas longas cada rotação custa uma passagem a preço cheio sobre o contexto.
:::

:::tip
Pools de credenciais servem principalmente para providers com API key (OpenRouter, Anthropic). Um único OAuth do [Nous Portal](/integrations/nous-portal) cobre 300+ modelos, então a maioria dos usuários não precisa de pool quando usa o Portal.
:::

## Como funciona {#how-it-works}

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Plan/usage limit reached (e.g. ChatGPT/Codex "usage limit reached")?
          → Rotate to next pool key immediately (no retry — the cap won't clear on retry)
      → Generic / transient 429?
          → Retry same key once (transient blip)
          → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (24h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → Success → continue normally
```

## Início rápido {#quick-start}

Se você já tem uma API key definida em `.env`, o Hermes a descobre automaticamente como pool de 1 key. Para se beneficiar do pooling, adicione mais keys:

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (requires Claude Max plan + extra usage credits)
hermes auth add anthropic --type oauth
# Opens browser for OAuth login
```

Verifique seus pools:

```bash
hermes auth list
```

Saída:
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   hermes_pkce ←
  #2  claude_code          oauth   claude_code
  #3  ANTHROPIC_API_KEY    api_key env:ANTHROPIC_API_KEY
```

O `←` marca a credencial selecionada no momento.

## Gerenciamento interativo {#interactive-management}

Execute `hermes auth` sem subcomando para um assistente interativo:

```bash
hermes auth
```

Isso mostra o status completo do pool e oferece um menu:

```
What would you like to do?
  1. Add a credential
  2. Remove a credential
  3. Reset cooldowns for a provider
  4. Set rotation strategy for a provider
  5. Exit
```

Para providers que suportam API key e OAuth (Anthropic, Nous, Codex), o fluxo de adição pergunta qual tipo:

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

## Comandos CLI {#cli-commands}

| Comando | Descrição |
|---------|-------------|
| `hermes auth` | Assistente interativo de gerenciamento de pool |
| `hermes auth list` | Mostra todos os pools e credenciais |
| `hermes auth list <provider>` | Mostra o pool de um provider específico |
| `hermes auth add <provider>` | Adiciona credencial (pergunta tipo e key) |
| `hermes auth add <provider> --type api-key --api-key <key>` | Adiciona API key de forma não interativa |
| `hermes auth add <provider> --type oauth` | Adiciona credencial OAuth via login no browser |
| `hermes auth remove <provider> <index>` | Remove credencial pelo índice (base 1) |
| `hermes auth reset <provider>` | Limpa todos os cooldowns/status de esgotamento |

## Estratégias de rotação {#rotation-strategies}

Configure via `hermes auth` → "Set rotation strategy" ou em `config.yaml`:

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| Estratégia | Comportamento |
|----------|----------|
| `fill_first` (padrão) | Usa a primeira key saudável até esgotar, depois passa para a próxima |
| `round_robin` | Alterna entre keys de forma uniforme, rotacionando após cada seleção |
| `least_used` | Sempre escolhe a key com menor contagem de requisições |
| `random` | Seleção aleatória entre keys saudáveis |

## Recuperação de erros {#error-recovery}

O pool trata erros diferentes de formas diferentes:

| Erro | Comportamento | Cooldown |
|-------|----------|----------|
| **429 Rate Limit** | Tenta a mesma key uma vez (transiente). Segundo 429 consecutivo rotaciona para a próxima key | 1 hora |
| **402 Billing/Quota** | Rotaciona imediatamente para a próxima key | 24 horas |
| **401 Auth Expired** | Tenta renovar o token OAuth primeiro. Rotaciona só se a renovação falhar | — |
| **Todas as keys esgotadas** | Passa para `fallback_model` se configurado | — |

A flag `has_retried_429` é resetada em toda chamada de API bem-sucedida, então um único 429 transiente não dispara rotação.

## Pools de endpoint customizado {#custom-endpoint-pools}

Endpoints OpenAI-compatíveis customizados (Together.ai, RunPod, servidores locais) têm pools próprios, indexados pelo nome do endpoint em `custom_providers` no config.yaml.

Quando você configura um endpoint customizado via `hermes model`, ele gera automaticamente um nome como "Together.ai" ou "Local (localhost:8080)". Esse nome vira a chave do pool.

```bash
# After setting up a custom endpoint via hermes model:
hermes auth list
# Shows:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# Add a second key for the same endpoint:
hermes auth add Together.ai --api-key sk-together-second-key
```

Pools de endpoint customizado ficam em `auth.json` sob `credential_pool` com prefixo `custom:`:

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## Auto-descoberta {#auto-discovery}

O Hermes descobre credenciais automaticamente de várias fontes e popula o pool na inicialização:

| Fonte | Exemplo | Auto-seeded? |
|--------|---------|-------------|
| Variáveis de ambiente | `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` | Sim |
| Tokens OAuth (auth.json) | Codex device code, Nous device code | Sim |
| Credenciais Claude Code | `~/.claude/.credentials.json` | Sim (Anthropic) |
| OAuth PKCE do Hermes | `~/.hermes/auth.json` | Sim (Anthropic) |
| Config de endpoint customizado | `model.api_key` em config.yaml | Sim (endpoints customizados) |
| Entradas manuais | Adicionadas via `hermes auth add` | Persistidas em auth.json |

Entradas auto-seeded são atualizadas a cada carregamento do pool — se você remover uma env var, a entrada correspondente no pool é podada automaticamente. Entradas manuais (via `hermes auth add`) nunca são auto-podadas.

Segredos emprestados em runtime (por exemplo env vars, referências Bitwarden/Vault/keyring/systemd e valores customizados de config) são apenas referência na fronteira do `auth.json`. O Hermes pode usar o valor resolvido em memória na execução atual, mas persiste só metadados como ref de origem, label, status, contadores de requisição e fingerprint não reversível. Entradas manuais e estado OAuth/device-code do Hermes mantêm os tokens duráveis necessários para refresh.

## Delegação e compartilhamento com subagentes {#delegation-subagent-sharing}

Quando o agente cria subagentes via `delegate_task`, o pool de credenciais do pai é compartilhado automaticamente com os filhos:

- **Mesmo provider** — o filho recebe o pool completo do pai, permitindo rotação de key em rate limits
- **Provider diferente** — o filho carrega o pool próprio desse provider (se configurado)
- **Sem pool configurado** — o filho usa a API key única herdada

Subagentes se beneficiam da mesma resiliência a rate limit que o pai, sem configuração extra. Leasing de credencial por tarefa garante que filhos não conflitem ao rotacionar keys em paralelo.

## Thread safety {#thread-safety}

O pool de credenciais usa lock de threading em todas as mutações de estado (`select()`, `mark_exhausted_and_rotate()`, `try_refresh_current()`, `mark_used()`). Isso garante acesso concorrente seguro quando o gateway trata várias sessões de chat simultaneamente.

## Arquitetura {#architecture}

Para o diagrama completo de fluxo de dados, veja [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g) no repositório.

O pool de credenciais integra na camada de resolução de provider:

1. **`agent/credential_pool.py`** — Gerenciador do pool: armazenamento, seleção, rotação, cooldowns
2. **`hermes_cli/auth_commands.py`** — Comandos CLI e assistente interativo
3. **`hermes_cli/runtime_provider.py`** — Resolução de credenciais com awareness de pool
4. **`run_agent.py`** — Recuperação de erro: 429/402/401 → rotação do pool → fallback

## Armazenamento {#storage}

O estado do pool fica em `~/.hermes/auth.json` sob a chave `credential_pool`:

```json
{
  "version": 1,
  "credential_pool": {
    "openrouter": [
      {
        "id": "abc123",
        "label": "OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "priority": 0,
        "source": "env:OPENROUTER_API_KEY",
        "secret_source": "bitwarden",
        "secret_fingerprint": "sha256:12ab34cd56ef7890",
        "last_status": "ok",
        "request_count": 142
      }
    ],
    "anthropic": [
      {
        "id": "manual1",
        "label": "personal-api-key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-ant-api03-..."
      }
    ]
  }
}
```

A entrada OpenRouter acima foi emprestada de uma fonte externa, então a key bruta não fica em `auth.json`. A entrada Anthropic manual foi adicionada intencionalmente ao armazenamento de credenciais do Hermes, então o token permanece persistível.

Estratégias ficam em `config.yaml` (não em `auth.json`):

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
