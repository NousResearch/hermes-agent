---
sidebar_position: 3
title: "Memória Persistente"
description: "Como o Hermes Agent lembra entre sessões — MEMORY.md, USER.md e busca de sessões"
---

# Memória Persistente {#persistent-memory}

O Hermes Agent tem memória limitada e curada que persiste entre sessões. Isso permite lembrar suas preferências, seus projetos, seu ambiente e coisas que aprendeu.

## Como Funciona {#how-it-works}

Dois arquivos compõem a memória do agente:

| Arquivo | Propósito | Limite de Caracteres |
|------|---------|------------|
| **MEMORY.md** | Notas pessoais do agente — fatos do ambiente, convenções, coisas aprendidas | 2.200 chars (~800 tokens) |
| **USER.md** | Perfil do usuário — suas preferências, estilo de comunicação, expectativas | 1.375 chars (~500 tokens) |

Ambos ficam em `~/.hermes/memories/` e são injetados no system prompt como um snapshot congelado no início da sessão. O agente gerencia a própria memória via a ferramenta `memory` — pode adicionar, substituir ou remover entradas.

:::caution Um agente por Hermes home
Não aponte dois processos de agente para o mesmo diretório Hermes home. Escritas de memória são automáticas e voltam ao system prompt no início da sessão, então dois escritores compartilhando um home vão acumular entradas uma da outra em um estado que nenhum dos dois (nem você) autorizou. A memória é escopada por [profile](/user-guide/profiles) de propósito — dê ao segundo agente seu próprio profile e, se precisarem de memória compartilhada, use um [provedor de memória externo](/user-guide/features/memory-providers).
:::

:::info
Limites de caracteres mantêm a memória focada. A memória **não** compacta automaticamente: quando uma escrita excederia o limite, a ferramenta `memory` retorna erro em vez de descartar entradas silenciosamente. O agente então abre espaço — consolidando ou removendo entradas no mesmo turno antes de tentar de novo (veja [O Que Acontece Quando a Memória Enche](#what-happens-when-memory-is-full)). Note que `replace` também respeita o limite: trocar uma entrada por conteúdo maior ainda pode estourar, então o novo conteúdo precisa ser encurtado (ou outra entrada removida) para caber.
:::

## Como a Memória Aparece no System Prompt {#how-memory-appears-in-the-system-prompt}

No início de toda sessão, entradas de memória são carregadas do disco e renderizadas no system prompt como um bloco congelado:

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
- Percentual de uso e contagem de caracteres para o agente saber a capacidade
- Entradas individuais separadas por delimitadores `§` (section sign)
- Entradas podem ser multilinha

**Padrão de snapshot congelado:** A injeção no system prompt é capturada uma vez no início da sessão e nunca muda no meio da sessão. Isso é intencional — preserva o prefix cache do LLM para desempenho. Quando o agente adiciona/remove entradas de memória durante a sessão, as mudanças são persistidas no disco imediatamente, mas só aparecem no system prompt quando a próxima sessão começa. Respostas de ferramentas sempre mostram o estado ao vivo.

## Ações da Ferramenta Memory {#memory-tool-actions}

O agente usa a ferramenta `memory` com estas ações:

- **add** — Adiciona uma nova entrada de memória
- **replace** — Substitui uma entrada existente por conteúdo atualizado (usa correspondência por substring via `old_text`)
- **remove** — Remove uma entrada que não é mais relevante (usa correspondência por substring via `old_text`)

Não há ação `read` — o conteúdo de memória é injetado automaticamente no system prompt no início da sessão. O agente vê suas memórias como parte do contexto da conversa.

### Correspondência por Substring {#substring-matching}

As ações `replace` e `remove` usam correspondência por substring curta e única — você não precisa do texto completo da entrada. O parâmetro `old_text` só precisa ser uma substring única que identifique exatamente uma entrada:

```python
# Se a memória contém "User prefers dark mode in all editors"
memory(action="replace", target="memory",
       old_text="dark mode",
       content="User prefers light mode in VS Code, dark mode in terminal")
```

Se a substring corresponder a múltiplas entradas, um erro é retornado pedindo uma correspondência mais específica.

## Dois Targets Explicados {#two-targets-explained}

### `memory` — Notas Pessoais do Agente {#memory-agents-personal-notes}

Para informação que o agente precisa lembrar sobre ambiente, fluxos de trabalho e lições aprendidas:

- Fatos do ambiente (SO, ferramentas, estrutura do projeto)
- Convenções e configuração do projeto
- Quirks de ferramentas e workarounds descobertos
- Entradas de diário de tarefas concluídas
- Skills e técnicas que funcionaram

### `user` — Perfil do Usuário {#user-user-profile}

Para informação sobre identidade, preferências e estilo de comunicação do usuário:

- Nome, função, fuso horário
- Preferências de comunicação (conciso vs detalhado, preferências de formato)
- Pet peeves e coisas a evitar
- Hábitos de workflow
- Nível técnico

## O Que Salvar vs Ignorar {#what-to-save-vs-skip}

### Salve Estes (Proativamente) {#save-these-proactively}

O agente salva automaticamente — você não precisa pedir. Salva quando aprende:

- **Preferências do usuário:** "Prefiro TypeScript a JavaScript" → salvar em `user`
- **Fatos do ambiente:** "Este servidor roda Debian 12 com PostgreSQL 16" → salvar em `memory`
- **Correções:** "Não use `sudo` para comandos Docker, usuário está no grupo docker" → salvar em `memory`
- **Convenções:** "Projeto usa tabs, linha de 120 chars, docstrings estilo Google" → salvar em `memory`
- **Trabalho concluído:** "Migrei o banco de MySQL para PostgreSQL em 2026-01-15" → salvar em `memory`
- **Pedidos explícitos:** "Lembre que a rotação da minha API key é mensal" → salvar em `memory`

### Ignore Estes {#skip-these}

- **Info trivial/óbvia:** "Usuário perguntou sobre Python" — vago demais para ser útil
- **Fatos facilmente redescobríveis:** "Python 3.12 suporta f-string aninhada" — dá para buscar na web
- **Dumps de dados brutos:** Blocos grandes de código, logs, tabelas — grande demais para memória
- **Efêmeros da sessão:** Caminhos temporários de arquivo, contexto de debug pontual
- **Informação já em arquivos de contexto:** Conteúdo de SOUL.md e AGENTS.md

## Gerenciamento de Capacidade {#capacity-management}

A memória tem limites rígidos de caracteres para manter system prompts limitados:

| Store | Limite | Entradas típicas |
|-------|-------|----------------|
| memory | 2.200 chars | 8-15 entradas |
| user | 1.375 chars | 5-10 entradas |

### O Que Acontece Quando a Memória Enche {#what-happens-when-memory-is-full}

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

### Exemplos Práticos de Boas Entradas de Memória {#practical-examples-of-good-memory-entries}

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

## Prevenção de Duplicatas {#duplicate-prevention}

O sistema de memória rejeita automaticamente entradas duplicadas exatas. Se você tentar adicionar conteúdo que já existe, retorna sucesso com mensagem "no duplicate added".

## Varredura de Segurança {#security-scanning}

Entradas de memória são escaneadas por padrões de injeção e exfiltração antes de serem aceitas, já que são injetadas no system prompt. Conteúdo que corresponde a padrões de ameaça (prompt injection, exfiltração de credenciais, backdoors SSH) ou contém caracteres Unicode invisíveis é bloqueado.

## Busca de Sessões {#session-search}

Além de MEMORY.md e USER.md, o agente pode buscar conversas passadas com a ferramenta `session_search`:

- Todas as sessões CLI e de mensagens ficam em SQLite (`~/.hermes/state.db`) com busca full-text FTS5
- Consultas retornam mensagens reais do DB — sem sumarização por LLM, sem truncamento
- O agente encontra coisas discutidas semanas atrás, mesmo que não estejam na memória ativa
- O agente também pode rolar para frente/trás dentro de qualquer sessão encontrada

```bash
hermes sessions list    # Navegar sessões passadas
```

Veja [Session Search Tool](/user-guide/sessions#session-search-tool) para as três formas de chamada (discovery / scroll / browse) e o formato de resposta.

### session_search vs memory {#session_search-vs-memory}

| Recurso | Memória Persistente | Busca de Sessões |
|---------|------------------|----------------|
| **Capacidade** | ~1.300 tokens no total | Ilimitada (todas as sessões) |
| **Velocidade** | Instantânea (no system prompt) | ~20ms consulta FTS5, ~1ms scroll |
| **Custo** | Custo de token em todo prompt | Grátis — sem chamadas LLM |
| **Caso de uso** | Fatos-chave sempre disponíveis | Encontrar conversas passadas específicas |
| **Gerenciamento** | Curada manualmente pelo agente | Automático — todas as sessões armazenadas |
| **Custo de token** | Fixo por sessão (~1.300 tokens) | Sob demanda (buscado quando necessário) |

**Memória** é para fatos críticos que devem estar sempre no contexto. **Busca de sessões** é para consultas do tipo "discutimos X na semana passada?" em que o agente precisa lembrar detalhes de conversas passadas.

## Jornada de Aprendizado (`/journey`) {#learning-journey-journey}

A jornada de aprendizado é uma visão em linha do tempo de tudo que o Hermes aprendeu — skills salvas e entradas de memória plotadas ao longo do tempo (mais antigas no topo, mais novas embaixo), com um scrubber "constelação" reproduzível que replaya a construção. Os mesmos dados do grafo alimentam três superfícies:

- **CLI clássico / standalone** — `hermes journey` (aliases: `hermes learning`, `hermes memory-graph`) renderiza a linha do tempo no terminal. Flags: `--play` anima a construção (`--fps` para ajustar), `--width`/`--height` sobrescrevem o tamanho, `--no-color` desabilita cor, e `--json` despeja o payload bruto do grafo.
- **TUI** — `/journey` (aliases: `/learning`, `/memory-graph`) abre a linha do tempo como overlay.
- **App desktop** — `/journey` abre o painel Star Map / memory-graph, uma visualização interativa dos mesmos nós.

Além de visualizar, a jornada também é onde você **poda e corrige** o que o Hermes aprendeu:

| Comando | O que faz |
|---------|--------------|
| `hermes journey list` | Lista ids de nós — nomes de skills e ids `memory:<source>:<index>` para chunks de memória. |
| `hermes journey delete <node> [-y]` | Exclui um nó. Skills são **arquivadas** (restauráveis), chunks de memória são removidos. `-y` pula a confirmação. |
| `hermes journey edit <node>` | Abre o conteúdo do nó (`SKILL.md` da skill ou o chunk de memória) em `$EDITOR`. |

Os mesmos subcomandos `list` / `delete <id>` / `edit <id>` funcionam pelo comando `/journey` no chat da CLI, e o painel desktop oferece edit/delete nos nós diretamente.

## Configuração {#configuration}

```yaml
# Em ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # false = escrever livremente (padrão) | true = exigir aprovação
```

Definir **ambos** `memory_enabled` e `user_profile_enabled` como `false`
desliga os stores embutidos por completo: a tool `memory` é removida do
schema e o bloco de orientação some do system prompt, então o modelo nunca
é informado sobre uma tool que não pode usar. Um provider externo definido
via `memory.provider` (Hindsight, Mem0, Honcho, …) não é afetado e mantém
suas próprias tools — use isso quando quiser um backend de memória de
terceiros *em vez dos* arquivos embutidos. Listar `memory` em
`agent.disabled_toolsets` é o interruptor mais pesado: ele também esconde
as tools de providers externos.

Com só `memory_enabled: false` (user profile ainda ligado), a tool
permanece — ela sustenta o store de perfil — mas o system prompt troca a
orientação completa de memória por um bloco mais estreito só de perfil. O
schema da tool anuncia apenas o target `user`, e escritas diretas ou
staged em `MEMORY.md` desabilitado são rejeitadas. A configuração inversa
anuncia apenas `memory` e rejeita escritas em `USER.md`.

## Controlando escritas de memória (`write_approval`) {#controlling-memory-writes-write_approval}

Por padrão o agente salva memória livremente — inclusive pela revisão de auto-melhoria em background que roda após um turno. Se preferir aprovar saves primeiro, defina `memory.write_approval: true`. É um gate simples liga/desliga aplicado a **ambos** turnos em foreground e a revisão em background:

| `write_approval` | Comportamento |
|------------------|-----------|
| `false` (padrão) | Escrever livremente — o gate está desligado (comportamento pré-gate). |
| `true` | Exigir aprovação antes de qualquer coisa ser salva. Na CLI interativa, escritas em foreground pedem confirmação inline (entradas são pequenas o suficiente para ler por completo). Em todo o resto — plataformas de mensagens, scripts e a revisão de auto-melhoria em background — escritas ficam **staged** para revisão com `/memory pending`. |

> Para desligar a memória por completo (não só gatear), defina ambos `memory_enabled: false` e `user_profile_enabled: false`. Quando os dois stores embutidos estão desabilitados, a tool embutida `memory` é ocultada automaticamente.

Revise escritas staged pela CLI ou qualquer plataforma de mensagens:

```
/memory pending             # listar escritas de memória staged (auto ones tagged [auto])
/memory approve <id>        # aplicar uma (ou 'all')
/memory reject <id>         # descartar uma (ou 'all')
/memory approval on         # ligar o gate (ou 'off') e persistir
```

Esta é a resposta para "o agente salvou uma suposição errada sobre mim": defina
`write_approval: true`, e todo save — especialmente os em background não solicitados —
espera seu sim/não antes de entrar no seu perfil.

## Notificações de revisão em background (`display.memory_notifications`) {#background-review-notifications-displaymemory_notifications}

Após um turno, a revisão de auto-melhoria em background pode salvar memória
ou atualizar uma skill silenciosamente. Este é o loop de aprendizado consciente do consentimento do Hermes: correções repetidas e lições duráveis de workflow viram entradas compactas de memória ou skills procedurais, enquanto `write_approval` pode colocar essas escritas em staging para revisão
antes de afetarem sessões futuras. Por padrão aparece uma linha curta
`💾 Memory updated` no chat para você saber que aconteceu. Controle o quão verboso fica:

```yaml
display:
  memory_notifications: on    # off | on (padrão) | verbose
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
    model: google/gemini-3-flash-preview   # auto (padrão) = modelo principal de chat
```

Quando você aponta para um modelo **diferente** do principal, a revisão roda
lá com custo substancialmente menor (~3–5× em benchmarks). Como um modelo
diferente não reutiliza o prompt cache do principal, o fork automaticamente
replaya um **digest** compacto da conversa (turnos recentes verbatim + um
resumo dos mais antigos) em vez da transcrição completa — minimizando o que escreve
no novo cache. Captura manteve: em testes, captura de memória foi
idêntica e captura de skill quase idêntica à revisão no modelo principal.

Deixe em `auto` (ou defina para seu modelo principal) e nada muda — a
revisão continua no modelo principal com replay completo do cache quente.

### Desabilitando revisões automáticas (`enabled`) {#disabling-automatic-reviews-enabled}

O fork de revisão pode queimar uma parcela significativa dos tokens totais em hosts
ocupados. Operadores podem desabilitá-lo sem zerar intervalos de nudge:

```yaml
auxiliary:
  background_review:
    enabled: true              # false = skip automatic post-turn forks
```

Com `enabled: false`, forks automáticos pós-turno não spawnam; `/refine`
manual ainda funciona.

Uso do fork é persistido em `session_model_usage` com `task='background_review'`
e uma linha de completion é escrita em `agent.log`
(`Background review complete: thread=bg-review calls=… in=… out=… result=…`).

## Controlando escritas de skills (`skills.write_approval`) {#controlling-skill-writes-skillswrite_approval}

Skills usam o mesmo gate liga/desliga, mas a UX de revisão difere porque um
`SKILL.md` é grande demais para ler num balão de chat:

```yaml
skills:
  write_approval: false     # false = escrever livremente (padrão) | true = exigir aprovação
```

Quando `write_approval: true`, escritas de skill (create / edit / patch / write_file /
delete) sempre **staged** independente da origem. Você revisa o gist de uma linha
inline, mas o diff completo fica out-of-band:

```
/skills pending             # listar escritas de skill staged + gist de uma linha cada
/skills diff <id>           # diff unificado completo (melhor visto na CLI ou dashboard)
/skills approve <id>        # aplicar (ou 'all')
/skills reject <id>         # descartar (ou 'all')
/skills approval on         # ligar o gate (ou 'off') e persistir
```

Em plataforma de mensagens, aprove uma skill pelo gist + metadados, ou abra
`/skills diff` na CLI / dashboard / o arquivo staged em
`~/.hermes/pending/skills/<id>.json` quando quiser ler a mudança inteira.
Detalhes completos em [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).


## Provedores de Memória Externos {#external-memory-providers}

Para memória persistente mais profunda além de MEMORY.md e USER.md, o Hermes inclui 8 plugins de provedor de memória externo — incluindo Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover e Supermemory.

Provedores externos rodam **junto** com a memória built-in (nunca a substituem) e adicionam capacidades como grafos de conhecimento, busca semântica, extração automática de fatos e modelagem de usuário entre sessões.

```bash
hermes memory setup      # escolher um provedor e configurar
hermes memory status     # verificar o que está ativo
```

Veja o guia [Memory Providers](./memory-providers.md) para detalhes completos de cada provedor, instruções de setup e comparação.
