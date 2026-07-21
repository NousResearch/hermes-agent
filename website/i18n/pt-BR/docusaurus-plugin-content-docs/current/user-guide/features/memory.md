---
sidebar_position: 3
title: "Memória persistente"
description: "Como o Hermes Agent lembra entre sessões — MEMORY.md, USER.md e busca de sessões"
---

# Memória persistente

O Hermes Agent tem memória limitada e curada que persiste entre sessões. Isso permite lembrar suas preferências, projetos, ambiente e o que aprendeu.

## Como funciona {#how-it-works}

Dois arquivos compõem a memória do agente:

| Arquivo | Propósito | Limite de caracteres |
|------|---------|------------|
| **MEMORY.md** | Notas pessoais do agente — fatos do ambiente, convenções, coisas aprendidas | 2.200 chars (~800 tokens) |
| **USER.md** | Perfil do usuário — suas preferências, estilo de comunicação, expectativas | 1.375 chars (~500 tokens) |

Ambos ficam em `~/.hermes/memories/` e são injetados no system prompt como snapshot congelado no início da sessão. O agente gerencia a própria memória via a ferramenta `memory` — pode adicionar, substituir ou remover entradas.

:::info
Limites de caracteres mantêm a memória focada. A memória **não** compacta automaticamente: quando uma escrita excederia o limite, a ferramenta `memory` retorna erro em vez de descartar entradas silenciosamente. O agente então abre espaço — consolidando ou removendo entradas no mesmo turno antes de tentar de novo (veja [O que acontece quando a memória enche](#what-happens-when-memory-is-full)). Note que `replace` também respeita o limite: trocar uma entrada por conteúdo mais longo ainda pode estourar, então o novo conteúdo deve ser encurtado (ou outra entrada removida) para caber.
:::

## Como a memória aparece no system prompt {#how-memory-appears-in-the-system-prompt}

No início de cada sessão, entradas de memória são carregadas do disco e renderizadas no system prompt como um bloco congelado:

```
══════════════════════════════════════════════
MEMORY (your personal notes) [67% — 1,474/2,200 chars]
══════════════════════════════════════════════
User's project is a Rust web service at ~/code/myapi using Axum + SQLx
§
This machine runs Ubuntu 22.04, has Docker and Podman installed
§
User prefers concise responses, dislikes verbose explanations
```

O formato inclui:
- Um cabeçalho mostrando qual store (MEMORY ou USER PROFILE)
- Porcentagem de uso e contagem de caracteres para o agente saber a capacidade
- Entradas individuais separadas por delimitadores `§` (section sign)
- Entradas podem ser multilinha

**Padrão de snapshot congelado:** A injeção no system prompt é capturada uma vez no início da sessão e nunca muda no meio da sessão. Isso é intencional — preserva o prefix cache do LLM para desempenho. Quando o agente adiciona/remove entradas de memória durante a sessão, as mudanças são persistidas no disco imediatamente, mas só aparecem no system prompt quando a próxima sessão começa. Respostas de ferramentas sempre mostram o estado ao vivo.

## Ações da ferramenta memory {#memory-tool-actions}

O agente usa a ferramenta `memory` com estas ações:

- **add** — Adiciona uma nova entrada de memória
- **replace** — Substitui uma entrada existente por conteúdo atualizado (usa correspondência por substring via `old_text`)
- **remove** — Remove uma entrada que não é mais relevante (usa correspondência por substring via `old_text`)

Não há ação `read` — o conteúdo da memória é injetado automaticamente no system prompt no início da sessão. O agente vê suas memórias como parte do contexto da conversa.

### Correspondência por substring {#substring-matching}

As ações `replace` e `remove` usam correspondência por substring curta e única — você não precisa do texto completo da entrada. O parâmetro `old_text` só precisa ser uma substring única que identifique exatamente uma entrada:

```python
# If memory contains "User prefers dark mode in all editors"
memory(action="replace", target="memory",
       old_text="dark mode",
       content="User prefers light mode in VS Code, dark mode in terminal")
```

Se a substring corresponder a várias entradas, um erro é retornado pedindo uma correspondência mais específica.

## Dois targets explicados {#two-targets-explained}

### `memory` — Notas pessoais do agente {#memory--agents-personal-notes}

Para informação que o agente precisa lembrar sobre ambiente, fluxos de trabalho e lições aprendidas:

- Fatos do ambiente (SO, ferramentas, estrutura do projeto)
- Convenções e configuração do projeto
- Quirks e workarounds de ferramentas descobertos
- Entradas de diário de tarefas concluídas
- Skills e técnicas que funcionaram

### `user` — Perfil do usuário {#user--user-profile}

Para informação sobre identidade, preferências e estilo de comunicação do usuário:

- Nome, papel, fuso horário
- Preferências de comunicação (conciso vs detalhado, preferências de formato)
- Pet peeves e coisas a evitar
- Hábitos de fluxo de trabalho
- Nível de habilidade técnica

## O que salvar vs ignorar {#what-to-save-vs-skip}

### Salve isto (proativamente) {#save-these-proactively}

O agente salva automaticamente — você não precisa pedir. Ele salva quando aprende:

- **Preferências do usuário:** "Prefiro TypeScript a JavaScript" → salvar em `user`
- **Fatos do ambiente:** "Este servidor roda Debian 12 com PostgreSQL 16" → salvar em `memory`
- **Correções:** "Não use `sudo` para comandos Docker, usuário está no grupo docker" → salvar em `memory`
- **Convenções:** "Projeto usa tabs, line width 120 chars, docstrings estilo Google" → salvar em `memory`
- **Trabalho concluído:** "Migrei banco de MySQL para PostgreSQL em 2026-01-15" → salvar em `memory`
- **Pedidos explícitos:** "Lembre que minha rotação de API key é mensal" → salvar em `memory`

### Ignore isto {#skip-these}

- **Info trivial/óbvia:** "Usuário perguntou sobre Python" — vago demais para ser útil
- **Fatos facilmente redescobertos:** "Python 3.12 suporta f-string nesting" — pode pesquisar na web
- **Dumps de dados brutos:** Blocos grandes de código, logs, tabelas — grandes demais para memória
- **Efêmeros da sessão:** Caminhos temporários de arquivo, contexto de debug pontual
- **Informação já em arquivos de contexto:** Conteúdo de SOUL.md e AGENTS.md

## Gerenciamento de capacidade {#capacity-management}

A memória tem limites rígidos de caracteres para manter system prompts limitados:

| Store | Limite | Entradas típicas |
|-------|-------|----------------|
| memory | 2.200 chars | 8-15 entradas |
| user | 1.375 chars | 5-10 entradas |

### O que acontece quando a memória enche {#what-happens-when-memory-is-full}

Quando você tenta adicionar uma entrada que excederia o limite, a ferramenta retorna erro:

```json
{
  "success": false,
  "error": "Memory at 2,100/2,200 chars. Adding this entry (250 chars) would exceed the limit. Consolidate now: use 'replace' to merge overlapping entries into shorter ones or 'remove' stale or less important entries (see current_entries below), then retry this add — all in this turn.",
  "current_entries": ["..."],
  "usage": "2,100/2,200"
}
```

O agente deve então:
1. Ler as entradas atuais (mostradas na resposta de erro)
2. Identificar entradas que podem ser removidas ou consolidadas
3. Usar `replace` para mesclar entradas relacionadas em versões mais curtas
4. Depois `add` a nova entrada

**Boa prática:** Quando a memória estiver acima de 80% da capacidade (visível no cabeçalho do system prompt), consolide entradas antes de adicionar novas. Por exemplo, mescle três entradas separadas "projeto usa X" em uma descrição abrangente do projeto.

### Exemplos práticos de boas entradas de memória {#practical-examples-of-good-memory-entries}

**Entradas compactas e densas em informação funcionam melhor:**

```
# Good: Packs multiple related facts
User runs macOS 14 Sonoma, uses Homebrew, has Docker Desktop and Podman. Shell: zsh with oh-my-zsh. Editor: VS Code with Vim keybindings.

# Good: Specific, actionable convention
Project ~/code/api uses Go 1.22, sqlc for DB queries, chi router. Run tests with 'make test'. CI via GitHub Actions.

# Good: Lesson learned with context
The staging server (10.0.1.50) needs SSH port 2222, not 22. Key is at ~/.ssh/staging_ed25519.

# Bad: Too vague
User has a project.

# Bad: Too verbose
On January 5th, 2026, the user asked me to look at their project which is
located at ~/code/api. I discovered it uses Go version 1.22 and...
```

## Prevenção de duplicatas {#duplicate-prevention}

O sistema de memória rejeita automaticamente entradas duplicadas exatas. Se você tentar adicionar conteúdo que já existe, retorna sucesso com mensagem "no duplicate added".

## Varredura de segurança {#security-scanning}

Entradas de memória são escaneadas por padrões de injeção e exfiltração antes de serem aceitas, já que são injetadas no system prompt. Conteúdo que corresponde a padrões de ameaça (prompt injection, exfiltração de credenciais, backdoors SSH) ou contém caracteres Unicode invisíveis é bloqueado.

## Busca de sessões {#session-search}

Além de MEMORY.md e USER.md, o agente pode buscar conversas passadas usando a ferramenta `session_search`:

- Todas as sessões CLI e de mensagens ficam em SQLite (`~/.hermes/state.db`) com busca full-text FTS5
- Consultas retornam mensagens reais do DB — sem sumarização LLM, sem truncamento
- O agente encontra coisas discutidas semanas atrás, mesmo que não estejam na memória ativa
- O agente também pode rolar para frente/trás dentro de qualquer sessão encontrada

```bash
hermes sessions list    # Browse past sessions
```

Veja [Ferramenta session search](/user-guide/sessions#session-search-tool) para as três formas de chamada (discovery / scroll / browse) e o formato de resposta.

### session_search vs memory {#session_search-vs-memory}

| Recurso | Memória persistente | Busca de sessões |
|---------|------------------|----------------|
| **Capacidade** | ~1.300 tokens no total | Ilimitada (todas as sessões) |
| **Velocidade** | Instantânea (no system prompt) | ~20ms consulta FTS5, ~1ms scroll |
| **Custo** | Custo de token em todo prompt | Grátis — sem chamadas LLM |
| **Caso de uso** | Fatos-chave sempre disponíveis | Encontrar conversas passadas específicas |
| **Gerenciamento** | Curada manualmente pelo agente | Automático — todas as sessões armazenadas |
| **Custo de token** | Fixo por sessão (~1.300 tokens) | Sob demanda (buscado quando necessário) |

**Memória** é para fatos críticos que devem estar sempre no contexto. **Busca de sessões** é para consultas do tipo "discutimos X na semana passada?" onde o agente precisa lembrar detalhes de conversas passadas.

## Configuração {#configuration}

```yaml
# In ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # false = write freely (default) | true = require approval
```

## Controlando escritas de memória (`write_approval`) {#controlling-memory-writes-write_approval}

Por padrão o agente salva memória livremente — inclusive pela revisão de autoaperfeiçoamento em background que roda após um turno. Se preferir aprovar saves primeiro, defina `memory.write_approval: true`. É um gate simples liga/desliga aplicado a **ambos** turnos em foreground e à revisão em background:

| `write_approval` | Comportamento |
|------------------|-----------|
| `false` (padrão) | Escrever livremente — o gate está desligado (comportamento pré-gate). |
| `true` | Exigir aprovação antes de qualquer save. No CLI interativo, escritas em foreground pedem inline (entradas são pequenas o suficiente para ler por completo). Em todo lugar — plataformas de mensagens, scripts e a revisão de autoaperfeiçoamento em background — escritas são **staged** para revisão com `/memory pending`. |

> Para desligar memória por completo (não só gated), defina `memory_enabled: false`.

Revise escritas staged do CLI ou de qualquer plataforma de mensagens:

```
/memory pending             # list staged memory writes (auto ones tagged [auto])
/memory approve <id>        # apply one (or 'all')
/memory reject <id>         # drop one (or 'all')
/memory approval on         # turn the gate on (or 'off') and persist it
```

Esta é a resposta para "o agente salvou uma suposição errada sobre mim": defina
`write_approval: true`, e todo save — especialmente os unprompted em background
— espera seu sim/não antes de entrar no seu perfil.

## Notificações de revisão em background (`display.memory_notifications`) {#background-review-notifications-displaymemory_notifications}

Após um turno, a revisão de autoaperfeiçoamento em background pode salvar memória
ou atualizar uma skill silenciosamente. Este é o loop de aprendizado consciente de consentimento do Hermes: correções repetidas e lições duráveis de fluxo de trabalho viram entradas compactas de memória ou skills procedurais, enquanto `write_approval` pode staged essas escritas para revisão
antes de afetar sessões futuras. Por padrão exibe uma linha curta
`💾 Memory updated` no chat para você saber que aconteceu. Controle o quão verboso
isso é:

```yaml
display:
  memory_notifications: on    # off | on (default) | verbose
```

| Valor | Comportamento |
|-------|-----------|
| `off` | Sem notificação no chat. A revisão ainda roda e ainda escreve — você só não vê uma linha. |
| `on` (padrão) | Linha genérica, ex.: `💾 Memory updated`, `💾 Skill 'foo' patched`. |
| `verbose` | Inclui preview compacto do que mudou, ex. `💾 Memory ➕ User prefers terse replies` ou snippet de diff de skill `"old" → "new"`. |

> Isso governa apenas a **notificação de chat do gateway**. A revisão em si, e
> escritas nos seus stores de memória/skill, não são afetadas por esta configuração. Defina
> por plataforma via `display.platforms.<platform>.memory_notifications`.

## Rodando a revisão em um modelo mais barato (`auxiliary.background_review`) {#running-the-review-on-a-cheaper-model-auxiliarybackground_review}

A revisão roda no seu **modelo principal de chat** por padrão, replayando a
conversa — que já está quente no prompt cache, então são leituras baratas de cache.
Em um modelo principal caro você pode rodar a revisão em um modelo mais barato
em vez disso:

```yaml
auxiliary:
  background_review:
    provider: openrouter
    model: google/gemini-3-flash-preview   # auto (default) = main chat model
```

Quando você aponta para um modelo **diferente** do principal, a revisão roda
lá por custo substancialmente menor (~3–5× em benchmarks). Como um modelo
diferente não pode reutilizar o prompt cache do seu modelo principal, o fork automaticamente
replaya um **digest** compacto da conversa (turnos recentes verbatim + um
resumo dos mais antigos) em vez da transcrição completa — minimizando o que escreve
no novo cache. Captura mantida: em testes, captura de memória foi
idêntica e captura de skill quase idêntica à revisão no modelo principal.

Deixe em `auto` (ou defina para seu modelo principal) e nada muda — a
revisão continua no modelo principal com replay completo do cache quente.

## Controlando escritas de skills (`skills.write_approval`) {#controlling-skill-writes-skillswrite_approval}

Skills usam o mesmo gate liga/desliga, mas a UX de revisão difere porque um
`SKILL.md` é grande demais para ler em uma bolha de chat:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

Quando `write_approval: true`, escritas de skill (create / edit / patch / write_file /
delete) sempre **staged** independentemente da origem. Você revisa o gist de uma linha
inline, mas o diff completo fica out-of-band:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

Em uma plataforma de mensagens, aprove uma skill pelo gist + metadata, ou abra
`/skills diff` no CLI / dashboard / o arquivo staged em
`~/.hermes/pending/skills/<id>.json` quando quiser ler a mudança inteira.
Detalhes completos em [Gating de escritas de skill pelo agente](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).


## Provedores de memória externos {#external-memory-providers}

Para memória persistente mais profunda que vai além de MEMORY.md e USER.md, o Hermes inclui 8 plugins de provedor de memória externo — incluindo Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover e Supermemory.

Provedores externos rodam **junto** com a memória built-in (nunca substituindo) e adicionam capacidades como grafos de conhecimento, busca semântica, extração automática de fatos e modelagem de usuário cross-session.

```bash
hermes memory setup      # pick a provider and configure it
hermes memory status     # check what's active
```

Veja o guia [Provedores de memória](./memory-providers.md) para detalhes completos de cada provedor, instruções de setup e comparação.
