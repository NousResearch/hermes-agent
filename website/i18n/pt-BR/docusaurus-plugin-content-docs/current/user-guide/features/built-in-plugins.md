---
sidebar_position: 12
sidebar_label: "Built-in Plugins"
title: "Built-in Plugins"
description: "Plugins incluídos com o Hermes Agent que rodam automaticamente via lifecycle hooks — disk-cleanup e outros"
---

# Built-in Plugins

O Hermes inclui um pequeno conjunto de plugins bundled com o repositório. Eles ficam em `<repo>/plugins/<name>/` e carregam automaticamente junto com plugins instalados pelo usuário em `~/.hermes/plugins/`. Usam a mesma superfície de plugin que plugins de terceiros — hooks, tools, slash commands — só que mantidos in-tree.

Veja a página [Plugins](/user-guide/features/plugins) para o sistema geral de plugins, e [Build a Hermes Plugin](/developer-guide/plugins) para escrever o seu.

## How discovery works {#how-discovery-works}

O `PluginManager` varre quatro fontes, em ordem:

1. **Bundled** — `<repo>/plugins/<name>/` (o que esta página documenta)
2. **User** — `~/.hermes/plugins/<name>/`
3. **Project** — `./.hermes/plugins/<name>/` (requires `HERMES_ENABLE_PROJECT_PLUGINS=1`)
4. **Pip entry points** — `hermes_agent.plugins`

Em colisão de nome, fontes posteriores vencem — um plugin do usuário chamado `disk-cleanup` substituiria o bundled.

`plugins/memory/` e `plugins/context_engine/` são deliberadamente excluídos da varredura bundled. Esses diretórios usam caminhos de descoberta próprios porque memory providers e context engines são providers de seleção única configurados via `hermes memory setup` / `context.engine` no config.

## Bundled plugins are opt-in {#bundled-plugins-are-opt-in}

Plugins bundled vêm desabilitados. A descoberta os encontra (aparecem em `hermes plugins list` e na UI interativa `hermes plugins`), mas nenhum carrega até você habilitá-los explicitamente:

```bash
hermes plugins enable disk-cleanup
```

Ou via `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - disk-cleanup
```

Este é o mesmo mecanismo que plugins instalados pelo usuário. Plugins bundled nunca são auto-habilitados — nem em instalação fresh, nem para usuários existentes atualizando para um Hermes mais novo. Você sempre opta explicitamente.

Para desligar um plugin bundled de novo:

```bash
hermes plugins disable disk-cleanup
# or: remove it from plugins.enabled in config.yaml
```

## Currently shipped {#currently-shipped}

O repo inclui estes plugins bundled em `plugins/`. Todos são opt-in — habilite via `hermes plugins enable <name>`.

| Plugin | Kind | Purpose |
|---|---|---|
| `disk-cleanup` | hooks + slash command | Auto-track ephemeral files and clean them on session end |
| `security-guidance` | hooks | Pattern-match dangerous code on `write_file`/`patch` and append a security warning (or block) — 25 rules (Apache-2.0 fork of Anthropic's `claude-plugins-official` patterns) |
| `observability/langfuse` | hooks | Trace turns / LLM calls / tools to [Langfuse](https://langfuse.com) |
| `observability/nemo_relay` | hooks | Relay observability events (turns / LLM calls / tools) to an NVIDIA NeMo endpoint |
| `teams_pipeline` | standalone | Microsoft Teams meeting pipeline — Graph-backed, transcript-first meeting summaries |
| `spotify` | backend (7 tools) | Native Spotify playback, queue, search, playlists, albums, library |
| `google_meet` | standalone | Join Meet calls, live-caption transcription, optional realtime duplex audio |
| `image_gen/openai` | image backend | OpenAI `gpt-image-2` image generation backend (alternative to FAL) |
| `image_gen/openai-codex` | image backend | OpenAI image generation via Codex OAuth |
| `image_gen/xai` | image backend | xAI `grok-2-image` backend |
| `hermes-achievements` | dashboard tab | Steam-style collectible badges generated from your real Hermes session history |
| `kanban/dashboard` | dashboard tab | Kanban board UI for the multi-agent dispatcher — tasks, comments, fan-out, board switching. See [Kanban Multi-Agent](./kanban.md). |

Memory providers (`plugins/memory/*`) e context engines (`plugins/context_engine/*`) são listados separadamente em [Memory Providers](./memory-providers.md) — gerenciados via `hermes memory` e `hermes plugins` respectivamente. O detalhe completo por plugin para os dois plugins longos baseados em hooks segue abaixo.

### disk-cleanup {#disk-cleanup}

Rastreia e remove automaticamente arquivos efêmeros criados durante sessões — scripts de teste, saídas temp, logs cron, perfis chrome stale — sem exigir que o agente lembre de chamar uma tool.

**Como funciona:**

| Hook | Behaviour |
|---|---|
| `post_tool_call` | When `write_file` / `terminal` / `patch` creates a file matching `test_*`, `tmp_*`, or `*.test.*` inside `HERMES_HOME` or `/tmp/hermes-*`, track it silently as `test` / `temp` / `cron-output`. |
| `on_session_end` | If any test files were auto-tracked during the turn, run the safe `quick` cleanup and log a one-line summary. Stays silent otherwise. |

**Regras de exclusão:**

| Category | Threshold | Confirmation |
|---|---|---|
| `test` | every session end | Never |
| `temp` | >7 days since tracked | Never |
| `cron-output` | >14 days since tracked | Never |
| empty dirs under HERMES_HOME | always | Never |
| `research` | >30 days, beyond 10 newest | Always (deep only) |
| `chrome-profile` | >14 days since tracked | Always (deep only) |
| files >500 MB | never auto | Always (deep only) |

**Slash command** — `/disk-cleanup` disponível em sessões CLI e gateway:

```
/disk-cleanup status                     # breakdown + top-10 largest
/disk-cleanup dry-run                    # preview without deleting
/disk-cleanup quick                      # run safe cleanup now
/disk-cleanup deep                       # quick + list items needing confirmation
/disk-cleanup track <path> <category>    # manual tracking
/disk-cleanup forget <path>              # stop tracking (does not delete)
```

**State** — tudo fica em `$HERMES_HOME/disk-cleanup/`:

| File | Contents |
|---|---|
| `tracked.json` | Tracked paths with category, size, and timestamp |
| `tracked.json.bak` | Atomic-write backup of the above |
| `cleanup.log` | Append-only audit trail of every track / skip / reject / delete |

**Safety** — cleanup só toca caminhos sob `HERMES_HOME` ou `/tmp/hermes-*`. Mounts Windows (`/mnt/c/...`) são rejeitados. Dirs de state top-level conhecidos (`logs/`, `memories/`, `sessions/`, `cron/`, `cache/`, `skills/`, `plugins/`, o próprio `disk-cleanup/`) nunca são removidos mesmo vazios — uma instalação fresh não é esvaziada no primeiro session end.

**Enabling:** `hermes plugins enable disk-cleanup` (ou marque a caixa em `hermes plugins`).

**Disabling again:** `hermes plugins disable disk-cleanup`.

### security-guidance {#security-guidance}

Avisos de segurança rápidos por pattern match em writes de arquivo. Quando chamadas `write_file` / `patch` / `skill_manage` do agente carregam conteúdo que corresponde a padrão de código conhecidamente perigoso — `pickle.load`, `yaml.load` sem `SafeLoader`, `eval(`, `os.system`, `subprocess(...,  shell=True)`, JS `child_process.exec`, React `dangerouslySetInnerHTML`, `.innerHTML =` / `.outerHTML =` / `document.write` raw, Node `crypto.createCipher`, modo AES ECB, verificação TLS desabilitada, parsers `xml.etree` / `minidom` propensos a XXE, `<script src="//..." >` sem SRI, `torch.load` sem `weights_only=True`, injeção `${{ github.event.* }}` em GitHub Actions — o plugin anexa um bloco `⚠️ Security guidance` ao resultado da tool.

O arquivo ainda é escrito. O modelo lê o aviso na tool message do próximo turno e pode corrigir o código ou documentar por que o construct é seguro neste contexto. Pattern matching tem taxa de falso positivo não trivial, por isso warn (não block) é o padrão.

**Coverage:** 25 regras no total, cobrindo desserialização insegura, command injection, XSS sinks, crypto footguns, XXE, supply-chain (SRI) e injeção em workflow CI/CD. Os dados de pattern são fork verbatim Apache-2.0 de [Anthropic's `claude-plugins-official`](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/security-guidance/hooks) — veja os arquivos `LICENSE` e `NOTICE` do plugin para atribuição.

**Modes:**

| Env var | Effect |
|---|---|
| (unset) | **warn mode** (default) — file is written, warning appended to result |
| `SECURITY_GUIDANCE_BLOCK=1` | **block mode** — write refused, warning returned as the block reason |
| `SECURITY_GUIDANCE_DISABLE=1` | kill switch — plugin loads but does nothing |

**Enabling:** `hermes plugins enable security-guidance` (ou marque a caixa em `hermes plugins`).

**Disabling again:** `hermes plugins disable security-guidance`.

**O que ainda não faz:** o plugin upstream da Anthropic tem mais duas camadas — revisão diff LLM em cada turno do agente que tocou arquivos, e revisão agentic no commit que rastreia fluxo de dados entre arquivos. Nenhuma foi portada. O agente já pode rodar essas revisões sob demanda via `delegate_task`.

### observability/langfuse {#observabilitylangfuse}

Rastreia turns, LLM calls e invocações de tool do Hermes para [Langfuse](https://langfuse.com) — plataforma open-source de observabilidade LLM. Um span por turn, uma generation por API call, uma tool observation por tool call. Totais de uso, contagens de token por tipo e estimativas de custo vêm dos números canônicos `agent.usage_pricing` do Hermes, então o dashboard Langfuse vê o mesmo breakdown (input / output / `cache_read_input_tokens` / `cache_creation_input_tokens` / `reasoning_tokens`) que aparece em `hermes logs`.

O plugin é fail-open: SDK não instalado, sem credenciais ou erro Langfuse transitório — tudo vira no-op silencioso no hook. O loop do agente nunca é impactado.

**Setup (interativo — recomendado):**

```bash
hermes tools          # → Langfuse Observability → Cloud or Self-Hosted
```

O wizard coleta suas keys, faz `pip install` do SDK `langfuse` e adiciona `observability/langfuse` a `plugins.enabled` para você. Reinicie o Hermes e o próximo turn envia um trace.

**Setup (manual):**

```bash
pip install langfuse
hermes plugins enable observability/langfuse
```

Depois coloque as credenciais em `~/.hermes/.env`:

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

**Como funciona:**

| Hook | Behaviour |
|---|---|
| `pre_api_request` / `pre_llm_call` | Open (or reuse) a per-turn root span "Hermes turn". Start a `generation` child observation for this API call with serialized recent messages as input. |
| `post_api_request` / `post_llm_call` | Close the generation, attach `usage_details`, `cost_details`, `finish_reason`, assistant output + tool calls. If no tool calls and non-empty content, close the turn. |
| `pre_tool_call` | Start a `tool` child observation with sanitized `args`. |
| `post_tool_call` | Close the tool observation with sanitized `result`. `read_file` payloads get summarized (head + tail + omitted-line count) so a huge file read stays under `HERMES_LANGFUSE_MAX_CHARS`. |

Agrupamento de sessão usa o session ID do Hermes (ou task ID para sub-agents) via `langfuse.propagate_attributes`, então tudo em uma única sessão `hermes chat` vive sob uma sessão Langfuse.

**Verify:**

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # check the Langfuse UI for a "Hermes turn" trace
```

**Optional tuning** (in `.env`):

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_LANGFUSE_ENV` | — | Environment tag on traces (`production`, `staging`, …) |
| `HERMES_LANGFUSE_RELEASE` | — | Release/version tag |
| `HERMES_LANGFUSE_SAMPLE_RATE` | `1.0` | Sampling rate passed to the SDK (0.0–1.0) |
| `HERMES_LANGFUSE_MAX_CHARS` | `12000` | Per-field truncation for message content / tool args / tool results |
| `HERMES_LANGFUSE_DEBUG` | `false` | Verbose plugin logging to `agent.log` |

Env vars com prefixo Hermes e SDK padrão (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`) são ambas aceitas — prefixo Hermes vence quando ambas estão definidas.

**Performance:** o cliente Langfuse é cacheado após a primeira chamada de hook. Se credenciais ou SDK estiverem ausentes, essa decisão também é cacheada — hooks subsequentes retornam rápido sem re-checar env vars ou recarregar config.

**Disabling:** `hermes plugins disable observability/langfuse`. O módulo do plugin ainda é descoberto, mas nenhum código do módulo roda até você re-habilitar.

### google_meet {#google_meet}

Permite que o agente **entre, transcreva e participe de chamadas Google Meet** — anotar uma reunião, resumir o vai-e-vem depois, dar follow-up em pontos específicos e (opcionalmente) falar respostas de volta na chamada via TTS.

**O que adiciona:**

- Um participante virtual headless que entra em URL Meet usando automação de browser
- Transcrição ao vivo do áudio da reunião via provider STT configurado
- Um toolset `meet_summarize` / `meet_speak` / `meet_followup` que o agente invoca para agir no que ouviu
- Artefatos pós-reunião (transcript, notas com speaker, action items) salvos em `~/.hermes/cache/google_meet/<meeting_id>/`

**Setup:**

```bash
hermes plugins enable google_meet
# Prompts you to sign in via the plugin's OAuth flow on first use —
# needs a Google account with Meet access. Host approval may be required
# if the meeting enforces "only invited participants can join".
```

Uso no chat:

> "Join meet.google.com/abc-defg-hij and take notes. After the call, send me a summary with action items."

O agente inicia o join da reunião, transmite a transcrição de volta ao contexto conforme a chamada prossegue, e produz um resumo estruturado quando a reunião termina (ou quando você mandar parar).

**Quando usar:** standups recorrentes onde você quer um bot para transcrever + resumir para participantes async; entrevistas estilo depoimento onde quer notas estruturadas; qualquer caso onde precisaria de Fireflies / Otter / Grain. Quando preferir não ter IA escutando — não habilite.

**Disabling:** `hermes plugins disable google_meet`. Transcripts e gravações em cache permanecem em `~/.hermes/cache/google_meet/` até você removê-los.

### hermes-achievements {#hermes-achievements}

Adiciona uma **aba de achievements estilo Steam ao dashboard** — 60+ badges colecionáveis em tiers gerados do seu histórico real de sessões Hermes. Feats de cadeia de tools, padrões de debugging, streaks de vibe-coding, uso de skill/memory, variedade de model/provider, quirks de lifestyle (sessões de fim de semana e noite). Originalmente autoria de [@PCinkusz](https://github.com/PCinkusz) como plugin externo; trazido in-tree para ficar em lockstep com mudanças de features do Hermes.

**Como funciona:**

- Varre todo o histórico de sessões `~/.hermes/state.db` no backend do dashboard
- Stats por sessão são cacheados por fingerprint `(started_at, last_active)`, então só sessões novas ou alteradas re-analisam em scans subsequentes
- O primeiro scan roda em thread background — o dashboard nunca bloqueia esperando, mesmo em databases com milhares de sessões
- Estado de unlock persiste em `$HERMES_HOME/plugins/hermes-achievements/state.json`

**Progressão de tier:** Copper → Silver → Gold → Diamond → Olympian. Cada card expõe seção "What counts" listando a métrica exata rastreada.

**Estados de achievement:**

| State | Meaning |
|---|---|
| Unlocked | At least one tier achieved |
| Discovered | Known achievement, progress visible, not yet earned |
| Secret | Hidden until Hermes detects the first related signal in your history |

**API** — rotas montam em `/api/plugins/hermes-achievements/`:

| Endpoint | Purpose |
|---|---|
| `GET /achievements` | Full catalog with per-badge unlock state (returns a pending placeholder while the first cold scan is running) |
| `GET /scan-status` | State of the background scanner: `idle` / `running` / `failed`, last duration, run count |
| `GET /recent-unlocks` | Twenty most recently unlocked badges, newest first |
| `GET /sessions/{id}/badges` | Badges earned primarily in one specific session |
| `POST /rescan` | Manual synchronous rescan (blocks; use when the user clicks the rescan button) |
| `POST /reset-state` | Clear unlock history and cached snapshot |

**State files** — ficam em `$HERMES_HOME/plugins/hermes-achievements/`:

| File | Contents |
|---|---|
| `state.json` | Unlock history: which badges you've earned and when. Stable across Hermes updates. |
| `scan_snapshot.json` | Last completed scan payload (served immediately on dashboard load) |
| `scan_checkpoint.json` | Per-session stats cache keyed by fingerprint (makes warm rescans fast) |

**Notas de performance:**

- Cold scan em ~8.000 sessões leva alguns minutos. Roda em thread background na primeira requisição do dashboard; a UI vê placeholder pending e faz poll em `/scan-status`.
- **Resultados incrementais durante cold scan** — o scanner publica snapshot parcial a cada ~250 sessões para cada refresh do dashboard mostrar mais badges desbloqueados conforme o scan progride. Sem minuto olhando zeros.
- Warm rescan reutiliza stats por sessão para toda sessão cujo fingerprint `started_at` + `last_active` bate com o checkpoint — completa em segundos mesmo em históricos grandes.
- TTL do snapshot in-memory é 120s; requisições stale servem o snapshot antigo imediatamente e disparam refresh em background. Você nunca espera spinner só porque o TTL expirou.

**Enabling:** Nada para habilitar — `hermes-achievements` é plugin só de dashboard (sem lifecycle hooks, sem tools visíveis ao modelo). Auto-registra como aba em `hermes dashboard` no primeiro launch. A config `plugins.enabled` só gateia plugins lifecycle/tool; plugins de dashboard são descobertos puramente via `dashboard/manifest.json`.

**Opting out:** Delete ou renomeie `plugins/hermes-achievements/dashboard/manifest.json`, ou sobrescreva com plugin do usuário com o mesmo nome em `~/.hermes/plugins/hermes-achievements/` que não inclua dashboard. Os state files do plugin em `$HERMES_HOME/plugins/hermes-achievements/` sobrevivem — reinstalar preserva seu histórico de unlock.

## Adding a bundled plugin {#adding-a-bundled-plugin}

Plugins bundled são escritos exatamente como qualquer outro plugin Hermes — veja [Build a Hermes Plugin](/developer-guide/plugins). As únicas diferenças são:

- O diretório fica em `<repo>/plugins/<name>/` em vez de `~/.hermes/plugins/<name>/`
- A fonte do manifest é reportada como `bundled` em `hermes plugins list`
- Plugins do usuário com o mesmo nome sobrescrevem a versão bundled

Um plugin é bom candidato a bundling quando:

- Não tem dependências opcionais (ou já são deps de `pip install .[all]`)
- O comportamento beneficia a maioria dos usuários e é opt-out em vez de opt-in
- A lógica se integra a lifecycle hooks que o agente precisaria lembrar de invocar
- Complementa uma capacidade core sem expandir a superfície de tool visível ao modelo

Contra-exemplos — coisas que devem ficar como plugins instaláveis pelo usuário, não bundled: integrações de terceiros com API keys, workflows de nicho, árvores de dependência grandes, qualquer coisa que mudaria materialmente o comportamento do agente por padrão.
