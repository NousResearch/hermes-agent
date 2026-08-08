---
sidebar_position: 8
title: "Arquivos de contexto"
description: "Arquivos de contexto do projeto — .hermes.md, AGENTS.md, CLAUDE.md, SOUL.md global e .cursorrules — injetados automaticamente em toda conversa"
---

# Arquivos de contexto

O Hermes Agent descobre e carrega automaticamente arquivos de contexto que moldam como ele se comporta. Alguns são locais ao projeto e descobertos a partir do seu diretório de trabalho. `SOUL.md` agora é global à instância Hermes e é carregado apenas de `HERMES_HOME`.

## Arquivos de contexto suportados {#supported-context-files}

| Arquivo | Propósito | Descoberta |
|------|---------|-----------| 
| **.hermes.md** / **HERMES.md** | Instruções do projeto (maior prioridade) | Sobe até a raiz git |
| **AGENTS.md** | Instruções do projeto, convenções, arquitetura | CWD na inicialização + subdiretórios progressivamente |
| **CLAUDE.md** | Arquivos de contexto Claude Code (também detectados) | CWD na inicialização + subdiretórios progressivamente |
| **SOUL.md** | Personalidade e tom globais desta instância Hermes | Apenas `HERMES_HOME/SOUL.md` |
| **.cursorrules** | Convenções de código do Cursor IDE | Apenas CWD |
| **.cursor/rules/*.mdc** | Módulos de regras do Cursor IDE | Apenas CWD |

:::info Sistema de prioridade
Apenas **um** tipo de contexto de projeto é carregado por sessão (primeira correspondência vence): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. **SOUL.md** é sempre carregado independentemente como identidade do agente (slot #1).
:::

## AGENTS.md {#agentsmd}

`AGENTS.md` é o arquivo principal de contexto do projeto. Ele diz ao agente como seu projeto está estruturado, quais convenções seguir e quaisquer instruções especiais.

### Descoberta progressiva em subdiretórios {#progressive-subdirectory-discovery}

No início da sessão, o Hermes carrega o `AGENTS.md` do seu diretório de trabalho no system prompt. Conforme o agente navega em subdiretórios durante a sessão (via `read_file`, `terminal`, `search_files`, etc.), ele **descobre progressivamente** arquivos de contexto nesses diretórios e os injeta na conversa no momento em que se tornam relevantes.

```
my-project/
├── AGENTS.md              ← Carregado na inicialização (system prompt)
├── frontend/
│   └── AGENTS.md          ← Descoberto quando o agente lê arquivos em frontend/
├── backend/
│   └── AGENTS.md          ← Descoberto quando o agente lê arquivos em backend/
└── shared/
    └── AGENTS.md          ← Descoberto quando o agente lê arquivos em shared/
```

Essa abordagem tem duas vantagens sobre carregar tudo na inicialização:
- **Sem inchaço do system prompt** — dicas de subdiretório só aparecem quando necessárias
- **Preservação do prompt cache** — o system prompt permanece estável entre turnos

Cada subdiretório é verificado no máximo uma vez por sessão. A descoberta também sobe diretórios pai, então ler `backend/src/main.py` descobrirá `backend/AGENTS.md` mesmo se `backend/src/` não tiver arquivo de contexto próprio.

:::info
Arquivos de contexto de subdiretório passam pelo mesmo [scan de segurança](#security-prompt-injection-protection) que arquivos de contexto na inicialização. Arquivos maliciosos são bloqueados.
:::

### Exemplo de AGENTS.md {#example-agentsmd}

```markdown
# Project Context

This is a Next.js 14 web application with a Python FastAPI backend.

## Architecture
- Frontend: Next.js 14 with App Router in `/frontend`
- Backend: FastAPI in `/backend`, uses SQLAlchemy ORM
- Database: PostgreSQL 16
- Deployment: Docker Compose on a Hetzner VPS

## Conventions
- Use TypeScript strict mode for all frontend code
- Python code follows PEP 8, use type hints everywhere
- All API endpoints return JSON with `{data, error, meta}` shape
- Tests go in `__tests__/` directories (frontend) or `tests/` (backend)

## Important Notes
- Never modify migration files directly — use Alembic commands
- The `.env.local` file has real API keys, don't commit it
- Frontend port is 3000, backend is 8000, DB is 5432
```

## SOUL.md {#soulmd}

`SOUL.md` controla a personalidade, o tom e o estilo de comunicação do agente. Veja a página [Personality](/user-guide/features/personality) para detalhes completos.

**Localização:**

- `~/.hermes/SOUL.md`
- ou `$HERMES_HOME/SOUL.md` se você roda o Hermes com um diretório home customizado

Detalhes importantes:

- O Hermes cria um `SOUL.md` padrão automaticamente se ainda não existir
- O Hermes carrega `SOUL.md` apenas de `HERMES_HOME`
- O Hermes não procura `SOUL.md` no diretório de trabalho
- Se o arquivo estiver vazio, nada de `SOUL.md` é adicionado ao prompt
- Se o arquivo tiver conteúdo, o conteúdo é injetado verbatim após scan e truncamento

## .cursorrules {#cursorrules}

O Hermes é compatível com o arquivo `.cursorrules` do Cursor IDE e os módulos de regra `.cursor/rules/*.mdc`. Se esses arquivos existirem na raiz do seu projeto e nenhum arquivo de contexto de maior prioridade (`.hermes.md`, `AGENTS.md` ou `CLAUDE.md`) for encontrado, eles são carregados como contexto do projeto.

Isso significa que suas convenções existentes do Cursor se aplicam automaticamente ao usar o Hermes.

## Como os arquivos de contexto são carregados {#how-context-files-are-loaded}

### Na inicialização (system prompt) {#at-startup-system-prompt}

Arquivos de contexto são carregados por `build_context_files_prompt()` em `agent/prompt_builder.py`:

1. **Escaneia o diretório de trabalho** — verifica `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules` (primeira correspondência vence)
2. **Conteúdo é lido** — cada arquivo é lido como texto UTF-8
3. **Scan de segurança** — conteúdo é verificado quanto a padrões de prompt injection
4. **Truncamento** — arquivos que excedem `context_file_max_chars` caracteres (padrão 20.000) são truncados head/tail (70% head, 20% tail, com marcador no meio)
5. **Montagem** — todas as seções são combinadas sob um cabeçalho `# Project Context`
6. **Injeção** — o conteúdo montado é adicionado ao system prompt

### Durante a sessão (descoberta progressiva) {#during-the-session-progressive-discovery}

`SubdirectoryHintTracker` em `agent/subdirectory_hints.py` observa argumentos de chamadas de ferramenta em busca de caminhos de arquivo:

1. **Extração de caminho** — após cada chamada de ferramenta, caminhos de arquivo são extraídos dos argumentos (`path`, `workdir`, comandos shell)
2. **Subida de ancestrais** — o diretório e até 5 diretórios pai são verificados (parando em diretórios já visitados)
3. **Carregamento de dica** — se um `AGENTS.md`, `CLAUDE.md` ou `.cursorrules` for encontrado, é carregado (primeira correspondência por diretório)
4. **Scan de segurança** — mesmo scan de prompt injection que arquivos na inicialização
5. **Truncamento** — limitado a 8.000 caracteres por arquivo
6. **Injeção** — anexado ao resultado da ferramenta, para o modelo ver no contexto naturalmente

A seção final do prompt fica mais ou menos assim:

```text
# Project Context

The following project context files have been loaded and should be followed:

## AGENTS.md

[Your AGENTS.md content here]

## .cursorrules

[Your .cursorrules content here]

[Your SOUL.md content here]
```

Note que o conteúdo SOUL é inserido diretamente, sem texto wrapper extra.

## Segurança: proteção contra prompt injection {#security-prompt-injection-protection}

Todos os arquivos de contexto são escaneados quanto a possível prompt injection antes de serem incluídos. O scanner verifica:

- **Tentativas de override de instrução**: "ignore previous instructions", "disregard your rules"
- **Padrões de engano**: "do not tell the user"
- **Overrides de system prompt**: "system prompt override"
- **Comentários HTML ocultos**: `<!-- ignore instructions -->`
- **Elementos div ocultos**: `<div style="display:none">`
- **Exfiltração de credenciais**: `curl ... $API_KEY`
- **Acesso a arquivos secretos**: `cat .env`, `cat credentials`
- **Caracteres invisíveis**: zero-width spaces, overrides bidirecionais, word joiners

Se qualquer padrão de ameaça for detectado, o arquivo é bloqueado:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

:::warning
Este scanner protege contra padrões comuns de injection, mas não substitui revisar arquivos de contexto em repositórios compartilhados. Sempre valide o conteúdo de AGENTS.md em projetos que você não autorou.
:::

## Limites de tamanho {#size-limits}

| Limite | Valor |
|-------|-------|
| Máx. chars por arquivo | `context_file_max_chars` (padrão 20.000, ~7.000 tokens) |
| Proporção de truncamento head | 70% |
| Proporção de truncamento tail | 20% |
| Marcador de truncamento | 10% (mostra contagens de chars e sugere usar ferramentas de arquivo) |

Quando um arquivo excede o limite configurado, a mensagem de truncamento diz:

```
[...truncated AGENTS.md: kept 14000+4000 of 25000 chars. Use file tools to read the full file.]
```

## Dicas para arquivos de contexto eficazes {#tips-for-effective-context-files}

:::tip Boas práticas para AGENTS.md
1. **Mantenha conciso** — fique abaixo do seu `context_file_max_chars` configurado; o agente lê a cada turno
2. **Estruture com cabeçalhos** — use seções `##` para arquitetura, convenções, notas importantes
3. **Inclua exemplos concretos** — mostre padrões de código preferidos, formatos de API, convenções de nomenclatura
4. **Mencione o que NÃO fazer** — "never modify migration files directly"
5. **Liste caminhos e portas-chave** — o agente usa isso para comandos de terminal
6. **Atualize conforme o projeto evolui** — contexto obsoleto é pior que nenhum contexto
:::

### Contexto por subdiretório {#per-subdirectory-context}

Para monorepos, coloque instruções específicas de subdiretório em AGENTS.md aninhados:

```markdown
<!-- frontend/AGENTS.md -->
# Frontend Context

- Use `pnpm` not `npm` for package management
- Components go in `src/components/`, pages in `src/app/`
- Use Tailwind CSS, never inline styles
- Run tests with `pnpm test`
```

```markdown
<!-- backend/AGENTS.md -->
# Backend Context

- Use `poetry` for dependency management
- Run the dev server with `poetry run uvicorn main:app --reload`
- All endpoints need OpenAPI docstrings
- Database models are in `models/`, schemas in `schemas/`
```
