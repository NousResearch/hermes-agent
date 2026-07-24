---
sidebar_position: 7
title: "Sessões"
description: "Persistência de sessões, retomada, busca, gerenciamento e rastreamento de sessões por plataforma"
---

import useBaseUrl from '@docusaurus/useBaseUrl';

# Sessões

O Hermes Agent salva automaticamente toda conversa como uma sessão. Sessões permitem retomar conversas, busca entre sessões e gerenciamento completo do histórico de conversas.

## How Sessions Work {#how-sessions-work}

Toda conversa — seja da CLI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Teams ou qualquer outra plataforma de mensagens — é armazenada como sessão com histórico completo de mensagens. Sessões são rastreadas em:

1. **Banco SQLite** (`~/.hermes/state.db`) — metadados estruturados de sessão com busca full-text FTS5, além do histórico completo de mensagens

O banco SQLite armazena:
- Session ID, plataforma de origem, user ID
- **Título da sessão** (nome único, legível por humanos)
- Nome e configuração do model
- Snapshot do system prompt
- Histórico completo de mensagens (role, content, tool calls, tool results)
- Contagens de tokens (input/output)
- Timestamps (started_at, ended_at)
- Parent session ID (para divisão de sessão disparada por compressão)

### What Counts Toward Context {#what-counts-toward-context}

O Hermes armazena histórico de sessão para poder retomar conversas, mas não
continua reenviando todo byte que já manipulou. A cada turn, o model vê
o system prompt selecionado, a janela de conversa atual e qualquer conteúdo
que o Hermes injeta explicitamente naquele turn.

Anexos de mídia são tratados como inputs com escopo de turn:

- Imagens podem ser anexadas nativamente à próxima chamada do model, ou pré-analisadas em
  uma descrição em texto quando o model ativo não suporta vision nativa.
- Áudio é transcrito em texto quando speech-to-text está configurado.
- Documentos de texto podem ter seu texto extraído incluído; outros tipos de documento
  geralmente são representados por um path local salvo e uma nota curta.
- Paths de anexo e texto extraído/derivado podem aparecer no transcript, mas
  os bytes brutos de imagem, áudio ou arquivo binário não são copiados repetidamente em
  prompts futuros.

Por exemplo, se um usuário envia uma imagem e pede ao Hermes para fazer um meme dela,
o Hermes pode inspecionar essa imagem uma vez com vision e executar um script de
processamento de imagem. Turns futuros não carregam automaticamente o JPEG original no contexto.
Eles carregam apenas o que foi escrito na conversa, como o pedido do usuário,
uma descrição curta da imagem, um path de cache local ou a resposta final do assistant.

A causa mais comum de crescimento de contexto não é o arquivo de mídia em si. É
texto verboso: transcripts colados, logs completos, outputs grandes de tools, diffs longos,
relatórios de status repetidos e dumps detalhados de prova. Prefira resumos, file
paths, trechos focados e lookups via tools em vez de copiar artefatos grandes
no chat.

:::tip
Use `/compress` quando uma sessão ficar longa, `/new` para um thread novo, e
`hermes sessions prune` apenas quando quiser excluir sessões encerradas antigas do
armazenamento. Compressão reduz o contexto ativo; não é uma exclusão por privacidade.
Passe um nome para `/new` (ex.: `/new payments-refactor`) para definir o título inicial
da nova sessão de antemão — útil para encontrá-la depois com `/resume <name>` ou
no seletor `/sessions`.
:::

### Session Sources {#session-sources}

Cada sessão é marcada com sua plataforma de origem:

| Source | Descrição |
|--------|-------------|
| `cli` | CLI interativa (`hermes` ou `hermes chat`) |
| `telegram` | Telegram messenger |
| `discord` | Discord server/DM |
| `slack` | Slack workspace |
| `whatsapp` | WhatsApp messenger |
| `signal` | Signal messenger |
| `matrix` | Matrix rooms and DMs |
| `mattermost` | Mattermost channels |
| `email` | Email (IMAP/SMTP) |
| `sms` | SMS via Twilio |
| `dingtalk` | DingTalk messenger |
| `feishu` | Feishu/Lark messenger |
| `wecom` | WeCom (WeChat Work) |
| `weixin` | Weixin (personal WeChat) |
| `bluebubbles` | Apple iMessage via BlueBubbles macOS server |
| `qqbot` | QQ Bot (Tencent QQ) via Official API v2 |
| `homeassistant` | Home Assistant conversation |
| `webhook` | Incoming webhooks |
| `api-server` | API server requests |
| `acp` | ACP editor integration |
| `cron` | Scheduled cron jobs |
| `batch` | Batch processing runs |

## CLI Session Resume {#cli-session-resume}

Retome conversas anteriores da CLI usando `--continue` ou `--resume`:

### Continue Last Session {#continue-last-session}

```bash
# Retomar a sessão CLI mais recente
hermes --continue
hermes -c

# Ou com o subcomando chat
hermes chat --continue
hermes chat -c
```

Isso busca a sessão `cli` mais recente no banco SQLite e carrega seu histórico completo de conversa.

### Resume by Name {#resume-by-name}

Se você deu um título a uma sessão (veja [Session Naming](#session-naming) abaixo), pode retomá-la por nome:

```bash
# Retomar uma sessão nomeada
hermes -c "my project"

# Se houver variantes de linhagem (my project, my project #2, my project #3),
# isso retoma automaticamente a mais recente
hermes -c "my project"   # → retoma "my project #3"
```

### Resume Specific Session {#resume-specific-session}

```bash
# Retomar uma sessão específica por ID
hermes --resume 20250305_091523_a1b2c3d4
hermes -r 20250305_091523_a1b2c3d4

# Retomar por título
hermes --resume "refactoring auth"

# Ou com o subcomando chat
hermes chat --resume 20250305_091523_a1b2c3d4
```

Session IDs são mostrados quando você sai de uma sessão CLI, e podem ser encontrados com `hermes sessions list`.

### Conversation Recap on Resume {#conversation-recap-on-resume}

Quando você retoma uma sessão, o Hermes exibe um recap compacto da conversa anterior em um painel estilizado antes do prompt de entrada:

<img className="docs-terminal-figure" src={useBaseUrl('/img/docs/session-recap.svg')} alt="Stylized preview of the Previous Conversation recap panel shown when resuming a Hermes session." />
<p className="docs-figure-caption">O modo resume mostra um painel de recap compacto com turns recentes de usuário e assistant antes de retornar você ao prompt ao vivo.</p>

O recap:
- Mostra **mensagens do usuário** (gold `●`) e **respostas do assistant** (green `◆`)
- **Trunca** mensagens longas (300 chars para usuário, 200 chars / 3 linhas para assistant)
- **Colapsa tool calls** em uma contagem com nomes de tools (ex.: `[3 tool calls: terminal, web_search]`)
- **Oculta** mensagens de system, tool results e reasoning interno
- **Limita** às últimas 10 trocas com indicador "... N earlier messages ..."
- Usa **estilo dim** para distinguir da conversa ativa

Para desabilitar o recap e manter o comportamento minimal de uma linha, defina em `~/.hermes/config.yaml`:

```yaml
display:
  resume_display: minimal   # default: full
```

:::tip
Session IDs seguem o formato `YYYYMMDD_HHMMSS_<hex>` — sessões CLI/TUI usam sufixo hex de 6 chars (ex. `20250305_091523_a1b2c3`), sessões de gateway usam sufixo de 8 chars (ex. `20250305_091523_a1b2c3d4`). Você pode retomar por ID (completo ou prefixo único) ou por título — ambos funcionam com `-c` e `-r`.
:::

## Cross-Platform Handoff {#cross-platform-handoff}

Use `/handoff <platform>` de uma sessão CLI para transferir a conversa ao vivo para o canal home de uma plataforma de mensagens. O agente continua exatamente de onde a CLI parou — mesmo session id, transcript completo com roles, tool calls e tudo.

```bash
# Dentro de uma sessão CLI
/handoff telegram
```

O que acontece:

1. A CLI valida que `<platform>` está habilitada e tem um canal home definido (execute `/sethome` do chat de destino uma vez para configurar).
2. A CLI marca a sessão como pending e **faz block-poll no gateway**. Recusa se o agente estiver no meio de um turn — aguarde a resposta atual terminar primeiro.
3. O watcher do gateway reivindica o handoff e pede ao adapter de destino um thread novo:
   - **Telegram** — abre um forum topic novo (DM topics se Bot API 9.4+ Topics mode estiver habilitado no chat, ou um forum supergroup topic).
   - **Discord** — cria um thread com auto-archive de 1440 min sob o text channel home.
   - **Slack** — posta uma seed message e usa seu `ts` como âncora do thread.
   - **WhatsApp / Signal / Matrix / SMS** — sem threads nativos, faz fallback direto para o canal home.
4. O gateway re-vincula a chave de destino ao seu session id CLI existente, depois forja um turn sintético de usuário pedindo ao agente para confirmar e resumir. A resposta cai no thread novo.
5. Quando o gateway confirma sucesso, a CLI imprime uma dica de `/resume` e sai limpo:

   ```
   ↻ Handoff complete. The session is now active on telegram.
     Resume it on this CLI later with: /resume my-session-title
   ```

6. A partir daí, a conversa vive na plataforma. Responda no thread novo — qualquer pessoa autorizada naquele canal compartilha a mesma sessão, e qualquer mensagem real posterior de usuário no thread entra sem costura porque sessões de thread são keyed sem `user_id`.

**Retomar de volta à CLI:** quando quiser voltar a um desktop, basta executar `/resume <title>` (ou `hermes -r "<title>"` do shell) e continuar de onde a plataforma parou.

**Modos de falha:**
- Nenhum canal home configurado → CLI recusa com dica de `/sethome`.
- Plataforma não habilitada / gateway não executando → CLI expira em 60s com mensagem clara e sua sessão CLI permanece intacta.
- Falha na criação de thread (permissões, topics-mode off) → faz fallback direto para o canal home e ainda completa; sem isolamento de thread mas o handoff em si funciona.
- Falha em `adapter.send` (rate limit, erro transitório de API) → handoff marcado como failed com o motivo; a linha é limpa para você tentar de novo.

**Limitação que vale saber:** para plataformas sem suporte a thread com canais home de grupo multi-usuário, o turn sintético é keyed como sessão estilo DM. Isso funciona para canais home self-DM (o setup típico) mas não é ideal para group chats genuinamente compartilhados. Threading cobre Telegram / Discord / Slack — de longe o caso comum — então a maioria dos setups nunca encontra isso.

## Session Naming {#session-naming}

Dê títulos legíveis por humanos às sessões para encontrá-las e retomá-las facilmente.

### Auto-Generated Titles {#auto-generated-titles}

O Hermes gera automaticamente um título descritivo curto (3–7 palavras) para cada sessão após a primeira troca. Isso roda em uma thread de background usando um model auxiliar rápido, então não adiciona latência. Você verá títulos auto-gerados ao navegar sessões com `hermes sessions list` ou `hermes sessions browse`.

Auto-titling dispara apenas uma vez por sessão e é ignorado se você já definiu um título manualmente.

### Setting a Title Manually {#setting-a-title-manually}

Use o slash command `/title` dentro de qualquer sessão de chat (CLI ou gateway):

```
/title my research project
```

O título é aplicado imediatamente. Se a sessão ainda não foi criada no banco (ex., você executa `/title` antes de enviar sua primeira mensagem), fica enfileirado e aplicado quando a sessão iniciar.

Você também pode renomear sessões existentes pela linha de comando:

```bash
hermes sessions rename 20250305_091523_a1b2c3d4 "refactoring auth module"
```

### Title Rules {#title-rules}

- **Único** — duas sessões não podem compartilhar o mesmo título
- **Máx. 100 caracteres** — mantém a saída de listagem limpa
- **Sanitizado** — caracteres de controle, chars zero-width e overrides RTL são removidos automaticamente
- **Unicode normal funciona** — emoji, CJK, caracteres acentuados, tudo funciona

### Auto-Lineage on Compression {#auto-lineage-on-compression}

Quando o contexto de uma sessão é comprimido (manualmente via `/compress` ou automaticamente), o Hermes cria uma nova sessão de continuação. Se a original tinha um título, a nova sessão recebe automaticamente um título numerado:

```
"my project" → "my project #2" → "my project #3"
```

Quando você retoma por nome (`hermes -c "my project"`), escolhe automaticamente a sessão mais recente na linhagem.

### /title in Messaging Platforms {#title-in-messaging-platforms}

O comando `/title` funciona em todas as plataformas de gateway (Telegram, Discord, Slack, WhatsApp):

- `/title My Research` — define o título da sessão
- `/title` — mostra o título atual

## Session Management Commands {#session-management-commands}

O Hermes fornece um conjunto completo de comandos de gerenciamento de sessão via `hermes sessions`:

### List Sessions {#list-sessions}

```bash
# Listar sessões recentes (padrão: últimas 20)
hermes sessions list

# Filtrar por plataforma
hermes sessions list --source telegram

# Mostrar mais sessões
hermes sessions list --limit 50
```

Quando sessões têm títulos, a saída mostra títulos, previews e timestamps relativos:

```
Title                  Preview                                  Last Active   ID
────────────────────────────────────────────────────────────────────────────────────────────────
refactoring auth       Help me refactor the auth module please   2h ago        20250305_091523_a
my project #3          Can you check the test failures?          yesterday     20250304_143022_e
—                      What's the weather in Las Vegas?          3d ago        20250303_101500_f
```

Quando nenhuma sessão tem títulos, um formato mais simples é usado:

```
Preview                                            Last Active   Src    ID
──────────────────────────────────────────────────────────────────────────────────────
Help me refactor the auth module please             2h ago        cli    20250305_091523_a
What's the weather in Las Vegas?                    3d ago        tele   20250303_101500_f
```

### Export Sessions {#export-sessions}

`hermes sessions export` é uma superfície para todo formato de export, selecionado com `--format`:

| Format | Output | Use it for |
|--------|--------|------------|
| `jsonl` (default) | one JSON object per session | backups, machine round-trip |
| `md` / `qmd` | one Markdown/Quarto file per session + manifest | readable archives, notes |
| `html` | single self-contained page (sidebar for multi-session) | sharing, browsing |
| `trace` | Claude Code JSONL | HF Agent Trace Viewer, `--upload` |

Mais `--only user-prompts` para uma view só de prompts (jsonl ou md).

Todos os formatos compartilham os mesmos knobs de seleção: `--session-id` para uma sessão, ou o conjunto completo de filtros de `prune`/`archive` — `--older-than` / `--newer-than` / `--before` / `--after` (durações como `5h`/`2d`/`1w`, dias bare, ou timestamps ISO), `--source`, `--title`, `--model`, `--provider`, `--cwd`, `--min/--max-messages`, `--min/--max-tokens`, `--min/--max-cost`, `--min/--max-tool-calls`, `--user`, `--chat-id`, `--chat-type`, `--branch`, `--end-reason`. `--dry-run` previewa o conjunto de match sem escrever. `--redact` remove secrets (API keys, tokens, credentials) do conteúdo exportado em qualquer formato — recomendado para qualquer coisa que você planeja compartilhar. Nota: filtros em bulk correspondem a sessões *encerradas*; `export` sem filtro despeja tudo, incluindo ativas.

#### JSONL (default) {#jsonl-default}

```bash
# Exportar todas as sessões para um arquivo JSONL
hermes sessions export backup.jsonl

# Exportar sessões de uma plataforma específica
hermes sessions export telegram-history.jsonl --source telegram

# Exportar uma sessão
hermes sessions export session.jsonl --session-id 20250305_091523_a1b2c3d4

# Redact API keys/tokens/credentials do conteúdo exportado
hermes sessions export backup.jsonl --redact
```

Arquivos exportados contêm um objeto JSON por linha com metadados completos de sessão e todas as mensagens.

#### HTML {#html}

`--format html` escreve um único arquivo HTML autocontido — sem dependências remotas — com message bubbles estilizadas, output de tool recolhível e (para exports multi-sessão) sidebar para alternar entre sessões:

```bash
# Uma sessão como página HTML standalone
hermes sessions export --format html --session-id 20250305_091523_a1b2c3d4 transcript.html

# Todas as sessões Telegram da última semana em um arquivo, secrets redacted
hermes sessions export --format html --newer-than 1w --source telegram --redact archive.html
```

#### Prompts Only {#prompts-only}

`--only user-prompts` exporta apenas os prompts que você escreveu — sem respostas do assistant, output de tool ou contexto de system. Útil para construir bibliotecas de prompts ou revisar o que você pediu:

```bash
# Um registro JSONL por prompt (session id, index, timestamp, text)
hermes sessions export prompts.jsonl --session-id 20250305_091523_a1b2c3d4 --only user-prompts

# Markdown, direto para stdout
hermes sessions export - --session-id 20250305_091523_a1b2c3d4 --only user-prompts --format md
```

Funciona com `--format jsonl` (padrão) ou `md`, honra os mesmos filtros para export em bulk, e combina com `--redact`.

#### Traces (HF Agent Trace Viewer) {#traces-hf-agent-trace-viewer}

`--format trace` emite Claude Code JSONL — o formato de transcript que o Hugging Face Hub auto-detecta para seu [Agent Trace Viewer](https://huggingface.co/docs/hub/agent-traces). Escreva localmente, ou adicione `--upload` para enviar ao seu próprio dataset privado `hermes-traces` (lê `HF_TOKEN`):

```bash
# Trace da sessão mais recente, para stdout
hermes sessions export --format trace

# Uma sessão para um arquivo trace local
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 trace.jsonl

# Upload direto para seu dataset privado HF traces
hermes sessions export --format trace --session-id 20250305_091523_a1b2c3d4 --upload
```

Exports trace são secret-redacted por padrão (foram feitos para sair da máquina); `--no-redact` opta por sair após revisão manual. `--upload` é privado a menos que `--public`. Export trace em bulk com filtros escreve um `<id>.trace.jsonl` por sessão.

#### Markdown / QMD {#markdown-qmd}

Passe `--format md` ou `--format qmd` quando quiser um archive legível baseado em arquivos antes de ocultar ou excluir sessões antigas. Exports Markdown/QMD escrevem um arquivo por sessão em um diretório (padrão: `~/.hermes/session-exports`).

```bash
# Exportar uma sessão para Markdown
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4

# Exportar uma linhagem de compressão como um documento lógico
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --lineage logical

# Preview sessões encerradas com mais de 90 dias sem escrever arquivos
hermes sessions export --format md --older-than 90 --dry-run

# Exportar sessões Telegram encerradas com mais de 2 semanas para arquivos QMD
hermes sessions export --format qmd --older-than 2w --source telegram

# Exportar sessões Claude longas, secrets redacted
hermes sessions export --format md --model sonnet --min-messages 50 --redact

# Só após verificação, exportar e excluir uma sessão explicitamente nomeada
hermes sessions export --format md --session-id 20250305_091523_a1b2c3d4 --delete-after-verified --yes
```

Export Markdown/QMD escreve um arquivo `.md` ou `.qmd` por sessão exportada mais um `manifest.jsonl` com file path, message count, lineage ids e SHA-256. Export em bulk exige pelo menos um filtro; um export bulk bare é recusado. `--delete-after-verified` é intencionalmente limitado a `--session-id` e exige `--yes`. `--redact` remove secrets (API keys, tokens, credentials) de message content e tool output antes de escrever — recomendado para qualquer export que você planeja compartilhar.

### Delete a Session {#delete-a-session}

```bash
# Excluir uma sessão específica (com confirmação)
hermes sessions delete 20250305_091523_a1b2c3d4

# Excluir sem confirmação
hermes sessions delete 20250305_091523_a1b2c3d4 --yes
```

### Rename a Session {#rename-a-session}

```bash
# Definir ou alterar o título de uma sessão
hermes sessions rename 20250305_091523_a1b2c3d4 "debugging auth flow"

# Títulos multi-palavra não precisam de aspas na CLI
hermes sessions rename 20250305_091523_a1b2c3d4 debugging auth flow
```

Se o título já estiver em uso por outra sessão, um erro é mostrado.

### Prune Old Sessions {#prune-old-sessions}

```bash
# Excluir sessões encerradas com mais de 90 dias (padrão)
hermes sessions prune

# Limiar de idade customizado — números bare são dias
hermes sessions prune --older-than 30

# Durações também funcionam: 5h, 30m, 2d, 1w
hermes sessions prune --older-than 12h

# Excluir apenas uma janela de tempo específica (ex. um lote de sessões de teste
# criadas nas últimas 5 horas)
hermes sessions prune --newer-than 5h

# Janela explícita com timestamps absolutos
hermes sessions prune --after "2026-07-05 09:00" --before "2026-07-05 14:30"

# Podar apenas sessões de uma plataforma específica (todas as idades — qualquer filtro
# desabilita o padrão implícito de 90 dias)
hermes sessions prune --source telegram
hermes sessions prune --source cron --older-than 60   # adicione flag de tempo para restringir

# Mais filtros — todos AND juntos
hermes sessions prune --newer-than 5h --title "smoke test"   # substring de título
hermes sessions prune --older-than 30 --max-messages 3        # sessões pequenas
hermes sessions prune --cwd ~/scratch --end-reason done       # por cwd / end reason
hermes sessions prune --model gpt-5 --older-than 1w           # por model (substring)
hermes sessions prune --provider openrouter --older-than 60   # por billing provider
hermes sessions prune --branch feature/old-experiment         # por git branch
hermes sessions prune --user 12345678 --chat-type group       # por origem de mensagens
hermes sessions prune --max-tokens 500 --older-than 7         # por uso de tokens
hermes sessions prune --max-cost 0.01 --max-tool-calls 0      # execuções baratas, sem tools

# Preview do que seria excluído, sem excluir nada
hermes sessions prune --newer-than 5h --dry-run

# Pular confirmação
hermes sessions prune --older-than 30 --yes
```

Valores de tempo (`--older-than`, `--newer-than`, `--before`, `--after`) aceitam uma
duração (`5h`, `30m`, `2d`, `1w`), um número bare de dias, ou um timestamp ISO
(`2026-07-05`, `2026-07-05 14:30`). `--older-than`/`--before` definem
o limite superior; `--newer-than`/`--after` definem o limite inferior. Combine ambos
para uma janela.

Filtros de atributo: `--source` (plataforma, exato), `--title` / `--model` /
`--branch` (substring case-insensitive), `--provider` (billing provider,
exato), `--end-reason`, `--user`, `--chat-id`, `--chat-type` (exato),
`--cwd` (prefixo de path), mais bounds numéricos `--min/--max-messages`,
`--min/--max-tokens` (input+output), `--min/--max-cost` (USD, actual caindo
back para estimated), e `--min/--max-tool-calls`. Usar qualquer filtro desabilita
o padrão implícito de 90 dias, então `hermes sessions prune --source cron` ou
`--model gpt-4o` corresponde a todas as idades — adicione uma flag de tempo para restringir. Apenas um
`hermes sessions prune` completamente bare mantém o cutoff de 90 dias. Toda
execução sem `--yes` mostra a contagem de match mais a sessão matching mais antiga e mais recente
antes de pedir confirmação.

Sessões arquivadas são ignoradas por padrão; passe `--include-archived` para
excluí-las também.

:::info
Pruning exclui apenas sessões **encerradas** (sessões que foram explicitamente encerradas ou auto-reset). Sessões ativas nunca são podadas.
:::

### Bulk-Archive Sessions {#bulk-archive-sessions}

Se quiser sessões fora das suas listagens sem excluir nada,
`hermes sessions archive` aceita os mesmos filtros que `prune` mas soft-hide
sessões matching em vez disso (define o mesmo flag archived que arquivar uma única
sessão pela UI Desktop/Dashboard — mensagens e busca permanecem intactas):

```bash
# Arquivar tudo das últimas 5 horas (ex. 75 sessões smoke-test de CI)
hermes sessions archive --newer-than 5h

# Arquivar por substring de título, preview primeiro
hermes sessions archive --title "dry run" --dry-run
hermes sessions archive --title "dry run" --yes
```

Pelo menos um filtro é obrigatório — um `hermes sessions archive` bare recusa
arquivar todo o seu histórico. Sessões arquivadas ficam ocultas de
`hermes sessions list` e `/resume` mas permanecem no banco e podem ser
desarquivadas pela lista de sessões Desktop/Dashboard.

### Session Statistics {#session-statistics}

```bash
hermes sessions stats
```

Saída:

```
Total sessions: 142
Total messages: 3847
  cli: 89 sessions
  telegram: 38 sessions
  discord: 15 sessions
Database size: 12.4 MB
```

Para analytics mais profundos — uso de tokens, estimativas de custo, breakdown de tools e padrões de atividade — use [`hermes insights`](/reference/cli-commands#hermes-insights).

## Session Search Tool {#session-search-tool}

O agente tem uma tool built-in `session_search` que executa busca full-text em todas as conversas passadas usando o engine FTS5 do SQLite — e deixa o agente rolar por qualquer sessão que encontrar. Sem chamadas LLM, sem summarization, sem truncation. Toda forma retorna mensagens reais do DB.

### Three calling shapes {#three-calling-shapes}

A tool infere o que você quer a partir de quais argumentos você define. Não há parâmetro `mode`.

**1. Discovery — passe `query`:**

```python
session_search(query="auth refactor", limit=3)
```

Executa FTS5, deduplica hits por linhagem de sessão, retorna as top N sessões. Cada resultado carrega:

- `session_id`, `title`, `when`, `source`
- `snippet` — trecho de match destacado por FTS5
- `bookend_start` — primeiras 3 mensagens user+assistant da sessão (goal/kickoff)
- `messages` — ±5 mensagens ao redor do match FTS5, com a mensagem âncora marcada (o hit em contexto)
- `bookend_end` — últimas 3 mensagens user+assistant da sessão (resolution/decisions)
- `match_message_id`, `messages_before`, `messages_after`

Bookends + window juntos reconstruem goal → match → resolution sem pagar pelo transcript inteiro. Wall time típico: 15–50ms em um session DB real.

**2. Scroll — passe `session_id` + `around_message_id`:**

```python
session_search(session_id="20260510_174648_805cc2", around_message_id=590803, window=10)
```

Retorna uma janela de ±`window` mensagens centradas na âncora. Sem FTS5, sem bookends — apenas o slice. Use após uma chamada discovery quando precisar de mais contexto que a janela ±5 padrão.

- Para rolar **para frente**: passe `messages[-1].id` de volta como `around_message_id`
- Para rolar **para trás**: passe `messages[0].id` de volta como `around_message_id`
- A mensagem de boundary aparece em ambas as janelas como marcador de orientação
- Quando `messages_before` ou `messages_after` for menor que `window`, você está no início ou fim da sessão

Wall time típico: 1–2ms por chamada scroll.

**3. Browse — sem args:**

```python
session_search()
```

Retorna sessões recentes cronologicamente (títulos, previews, timestamps). Útil quando o usuário pergunta "no que eu estava trabalhando" sem nomear um tópico.

### FTS5 query syntax {#fts5-query-syntax}

O modo keyword suporta sintaxe padrão de query FTS5:

- Keywords simples: `docker deployment` (FTS5 default para AND)
- Frases: `"exact phrase"`
- Boolean: `docker OR kubernetes`, `python NOT java`
- Prefix: `deploy*`

### Optional parameters {#optional-parameters}

- `sort` — `newest` ou `oldest`, sobre o ranking FTS5. Omita para ordenação só por relevância (o padrão; adequado para recall exploratório). Use `newest` para perguntas "onde paramos X", `oldest` para "como X começou".
- `role_filter` — roles separadas por vírgula para incluir. Discovery default para `user,assistant` (output de tool geralmente é ruído). Passe `user,assistant,tool` para incluir output de tool (debugging de comportamento de tool) ou `tool` para buscar só output de tool.

### When It's Used {#when-its-used}

O agente é instruído a usar session search automaticamente:

> *"When the user references something from a past conversation or you suspect relevant prior context exists, use session_search to recall it before asking them to repeat themselves."*

Gatilhos típicos: "we did this before", "remember when", "last time", "as I mentioned", ou qualquer referência a um projeto/pessoa/conceito que não está na janela atual.

## Per-Platform Session Tracking {#per-platform-session-tracking}

### Gateway Sessions {#gateway-sessions}

Em plataformas de mensagens, sessões são keyed por uma session key determinística construída a partir da origem da mensagem:

| Chat Type | Default Key Format | Behavior |
|-----------|--------------------|----------|
| Telegram DM | `agent:main:telegram:dm:<chat_id>` | One session per DM chat |
| Discord DM | `agent:main:discord:dm:<chat_id>` | One session per DM chat |
| WhatsApp DM | `agent:main:whatsapp:dm:<canonical_identifier>` | One session per DM user (LID/phone aliases collapse to one identity when mapping exists) |
| Group chat | `agent:main:<platform>:group:<chat_id>:<user_id>` | Per-user inside the group when the platform exposes a user ID |
| Group thread/topic | `agent:main:<platform>:group:<chat_id>:<thread_id>` | Shared session for all thread participants (default). Per-user with `thread_sessions_per_user: true`. |
| Channel | `agent:main:<platform>:channel:<chat_id>:<user_id>` | Per-user inside the channel when the platform exposes a user ID |

Quando o Hermes não consegue obter um identificador de participante para um chat compartilhado, faz fallback para uma sessão compartilhada para aquela sala.

### Shared vs Isolated Group Sessions {#shared-vs-isolated-group-sessions}

Por padrão, o Hermes usa `group_sessions_per_user: true` em `config.yaml`. Isso significa:

- Alice e Bob podem falar com o Hermes no mesmo canal Discord sem compartilhar histórico de transcript
- a tarefa longa e pesada em tools de um usuário não polui a janela de contexto de outro
- o tratamento de interrupt também fica per-user porque a running-agent key corresponde à session key isolada

Se quiser um "room brain" compartilhado em vez disso, defina:

```yaml
group_sessions_per_user: false
```

Isso reverte groups/channels para uma única sessão compartilhada por sala, o que preserva contexto conversacional compartilhado mas também compartilha custos de token, estado de interrupt e crescimento de contexto.

### Session Reset Policies {#session-reset-policies}

**Por padrão sessões de gateway nunca fazem auto-reset** (`mode: none`). Você pode optar
por resets automáticos via a seção `session_reset` em `config.yaml`:

- **none** — nunca auto-reset (padrão; contexto gerenciado por `/reset` e compressão)
- **idle** — reset após N minutos de inatividade
- **daily** — reset em uma hora específica a cada dia
- **both** — reset no que vier primeiro (idle ou daily)

Antes de um auto-reset de sessão, o agente recebe um turn para salvar memórias ou skills importantes da conversa.

Sessões com **processos de background ativos** nunca fazem auto-reset, independente da policy.

## Storage Locations {#storage-locations}

| What | Path | Description |
|------|------|-------------|
| SQLite database | `~/.hermes/state.db` | All session metadata + messages with FTS5 |
| Gateway messages    | `~/.hermes/state.db`   | SQLite — canonical store for all session messages |
| Gateway routing index | `~/.hermes/sessions/sessions.json` | Maps session keys to active session IDs (origin metadata, expiry flags) |

O banco SQLite usa modo WAL para readers concorrentes e um writer, o que combina bem com a arquitetura multi-plataforma do gateway.

:::warning `sessions.json` is not the session list
`~/.hermes/sessions/sessions.json` é o **gateway routing index** — mapeia
session keys de mensagens (`agent:main:<platform>:...`) para session IDs ativos.
Contém apenas entradas gateway/messaging, então se você executa uma plataforma
de mensagens verá só essas (ex. `agent:main:whatsapp:dm:...`).

Isso é **esperado** e **não** significa que suas sessões CLI estão faltando.
`hermes sessions list`, `/sessions` e o dashboard leem `state.db`,
que contém **toda** sessão (CLI, TUI e gateway). Os snapshots `/save`
em `~/.hermes/sessions/saved/*.json` são exports de conveniência, não o index.

Se sessões CLI genuinamente não aparecem em `hermes sessions list`, a causa é
`state.db` não recebê-las — execute `hermes sessions repair` e observe um
aviso `⚠ Session store unavailable` na inicialização da CLI, o que significa que a persistência SQLite
falhou naquela execução.
:::

:::note Legacy JSONL transcripts
Sessões criadas antes de state.db se tornar canônico podem ter arquivos
`*.jsonl` leftover em `~/.hermes/sessions/`. Não são mais escritos nem
lidos pelo Hermes. Seguro excluir após verificar que a sessão correspondente
existe em state.db.
:::

### Database Schema {#database-schema}

Tabelas principais em `state.db`:

- **sessions** — metadados de sessão (id, source, user_id, model, title, timestamps, token counts). Títulos têm unique index (títulos NULL permitidos, apenas non-NULL devem ser únicos).
- **messages** — histórico completo de mensagens (role, content, tool_calls, tool_name, token_count)
- **messages_fts** — tabela virtual FTS5 para busca full-text em message content

## Session Expiry and Cleanup {#session-expiry-and-cleanup}

### Automatic Cleanup {#automatic-cleanup}

- Sessões de gateway fazem auto-reset com base na reset policy configurada
- Antes do reset, o agente salva memórias e skills da sessão que expira
- Auto-pruning opt-in: quando `sessions.auto_prune` é `true`, sessões encerradas com mais de `sessions.retention_days` (padrão 90) são podadas na inicialização CLI/gateway
- Após um prune que realmente removeu linhas, `state.db` recebe `VACUUM` para recuperar espaço em disco (SQLite não encolhe o arquivo em DELETE simples)
- Pruning roda no máximo uma vez por `sessions.min_interval_hours` (padrão 24); o timestamp da última execução é rastreado dentro do próprio `state.db` então é compartilhado entre todo processo Hermes no mesmo `HERMES_HOME`

O padrão é **off** — histórico de sessão é valioso para recall de `session_search`, e excluir silenciosamente poderia surpreender usuários. Habilite em `~/.hermes/config.yaml`:

```yaml
sessions:
  auto_prune: true          # opt in — default is false
  retention_days: 90        # keep ended sessions this many days
  vacuum_after_prune: true  # reclaim disk space after a pruning sweep
  min_interval_hours: 24    # don't re-run the sweep more often than this
```

Sessões ativas nunca são auto-pruned, independente da idade.

### Manual Cleanup {#manual-cleanup}

```bash
# Podar sessões com mais de 90 dias
hermes sessions prune

# Excluir uma sessão específica
hermes sessions delete <session_id>

# Exportar antes de podar (backup)
hermes sessions export backup.jsonl
hermes sessions prune --older-than 30 --yes
```

:::tip
O banco cresce lentamente (típico: 10-15 MB para centenas de sessões) e o histórico de sessão alimenta recall de `session_search` em conversas passadas, então auto-prune vem desabilitado. Habilite se você executa uma carga pesada de gateway/cron onde `state.db` afeta significativamente a performance (modo de falha observado: state.db de 384 MB com ~1000 sessões desacelerando inserts FTS5 e listagem `/resume`). Use `hermes sessions prune` para limpeza pontual sem ligar o sweep automático.
:::
