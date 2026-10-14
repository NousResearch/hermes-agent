---
title: Codex App-Server Runtime (opcional)
sidebar_label: Codex App-Server Runtime
---

# Codex App-Server Runtime

O Hermes pode opcionalmente entregar turns `openai/*` e `openai-codex/*` ao [Codex CLI app-server](https://github.com/openai/codex) em vez de rodar seu próprio loop de ferramentas. Quando habilitado, comandos de terminal, edições de arquivo, sandboxing e MCP tool calls executam todos dentro do runtime Codex — o Hermes vira o shell em torno (sessions DB, slash commands, gateway, revisão de memory e skill).

Isso é **somente opt-in**. O comportamento padrão do Hermes não muda a menos que você ligue a flag. O Hermes nunca roteia você automaticamente para este runtime.

:::tip
Não usa OpenAI Codex? `hermes setup --portal` configura um backend não-Codex com Claude/Gemini/etc. num passo. Veja [Nous Portal](/integrations/nous-portal).
:::

## Por quê {#why}

- Rodar turns de agente OpenAI contra sua **assinatura ChatGPT** (sem API key) usando o mesmo fluxo de auth que o Codex CLI usa.
- Usar **o toolset e sandbox do Codex** — `shell` para terminal/read/write/search, `apply_patch` para edições estruturadas, `update_plan` para planejamento, tudo rodando dentro de sandboxing seatbelt/landlock.
- **Plugins nativos Codex** — Linear, GitHub, Gmail, Calendar, Canva, etc. — instalados via `codex plugin` são auto-migrados e ativos na sua sessão Hermes.
- **Ferramentas mais ricas do Hermes vêm junto** — web_search, web_extract, automação de browser, vision, geração de imagem, skills e TTS funcionam via callback MCP. O Codex chama de volta ao Hermes por ferramentas que não tem built-in.
- **Nudges de memory e skill continuam funcionando** — eventos do Codex são projetados na forma de mensagens Hermes para o loop de autoaperfeiçoamento ver um transcript de aparência normal.

## Quais ferramentas o modelo realmente tem {#what-tools-the-model-actually-has}

É a parte que a maioria dos usuários quer saber de cara. Com este runtime ligado, o modelo rodando seu turn tem três fontes independentes de ferramentas:

### 1. Toolset built-in do Codex (sempre ligado) {#1-codexs-built-in-toolset-always-on}

Estes vêm com o próprio `codex app-server` — sem envolvimento Hermes, sem MCP, sem plugins. Todos os cinco estão disponíveis no momento em que o runtime inicia:

- **`shell`** — roda comandos shell arbitrários dentro do sandbox. É assim que o modelo lê arquivos (`cat`, `head`, `tail`), escreve (`echo > foo`, heredocs), busca (`find`, `rg`, `grep`), navega diretórios (`ls`, `cd`), roda builds, gerencia processos e qualquer coisa que faria em bash.
- **`apply_patch`** — aplica um diff multi-arquivo estruturado no formato patch do Codex. O modelo usa para edições de código não triviais (adicionar função, refatorar entre arquivos); heredocs shell ainda estão disponíveis para writes avulsos.
- **`update_plan`** — todo / plan tracker interno do codex. Equivalente da ferramenta `todo` do Hermes, mas gerenciado inteiramente no runtime codex.
- **`view_image`** — carrega arquivo de imagem local na conversa para o modelo ver.
- **`web_search`** — codex tem busca web built-in quando configurado. O Hermes também expõe `web_search` (Firecrawl-backed) via callback abaixo; o modelo escolhe o que preferir.

Então **qualquer coisa que faria via terminal — read/write/search/find/run — o codex faz nativamente**. O perfil sandbox (`:workspace` por padrão quando você habilita o runtime) controla o que é gravável.

### 2. Plugins nativos Codex (auto-migrados do seu `codex plugin` install) {#2-native-codex-plugins-auto-migrated-from-your-codex-plugin-install}

Quando você habilita o runtime, o Hermes consulta o RPC `plugin/list` do codex e grava uma entrada `[plugins."<name>@openai-curated"]` para cada plugin instalado. Os plugins em si são gerenciados pelo codex e autorizados uma vez via UI própria do codex.

Exemplos (os que o thread OpenClaw destacou como "dignos de vídeo YouTube"):

- **Linear** — find/update issues
- **GitHub** — search code, view PRs, comment
- **Gmail** — read/send mail
- **Google Calendar** — create/find events
- **Outlook calendar/email** — mesma forma via conector Microsoft
- **Canva** — design generation
- ...o que mais você instalou via `codex plugin marketplace add openai-curated` + `codex plugin install ...`

O que NÃO é migrado:
- Plugins que você ainda não instalou — instale no Codex primeiro.
- Entradas do marketplace de apps ChatGPT (`app/list`) — já habilitadas dentro do codex pela auth da conta.

### 3. Callback de ferramentas Hermes (servidor MCP, registrado em `~/.codex/config.toml`) {#3-hermes-tool-callback-mcp-server-registered-in-codexconfigtoml}

O Hermes se registra como servidor MCP para o codex chamar de volta por ferramentas que o codex não envia. Disponíveis via callback:

- **`web_search`** / **`web_extract`** — Firecrawl-backed; tende a ser mais limpo que scraping para conteúdo estruturado.
- **`browser_navigate` / `browser_click` / `browser_type` / `browser_press` / `browser_snapshot` / `browser_scroll` / `browser_back` / `browser_get_images` / `browser_console` / `browser_vision`** — automação de browser completa via Camofox ou Browserbase.
- **`vision_analyze`** — chama modelo de vision separado para inspecionar imagem (diferente de `view_image` do codex que carrega na conversa).
- **`image_generate`** — geração de imagem pela cadeia plugin image_gen do Hermes.
- **`skill_view` / `skills_list`** — leitura da biblioteca de skills do Hermes.
- **`text_to_speech`** — TTS pelo provider configurado do Hermes.

Quando o modelo quer uma delas, o codex spawna o subprocess `hermes_tools_mcp_server` via stdio MCP, a chamada é despachada por `model_tools.handle_function_call()` (mesmo code path do runtime padrão Hermes) e o resultado volta ao codex como qualquer resposta MCP.

### O que NÃO está disponível neste runtime {#whats-not-available-on-this-runtime}

Estas quatro ferramentas Hermes exigem o contexto AIAgent em execução (estado mid-loop) para despachar, e um callback MCP stateless não pode dirigí-las. Volte ao runtime padrão (`/codex-runtime auto`) quando precisar de qualquer uma:

- **`delegate_task`** — spawn subagents
- **`memory`** — memory store persistente do Hermes
- **`session_search`** — busca cross-session
- **`todo`** — todo store do Hermes (`update_plan` do codex é o equivalente in-runtime)

## Recursos de workflow (`/goal`, kanban, cron) {#workflow-features-goal-kanban-cron}

### `/goal` (o loop Ralph) {#goal-the-ralph-loop}

**Funciona neste runtime.** Goals persistem em `state_meta` keyed por session id, o continuation prompt volta como mensagem user normal por `run_conversation()` e o codex executa o próximo turn nativamente. O goal judge roda via auxiliary client (configurado via `auxiliary.goal_judge` em config.yaml), independente de qual runtime está ativo. O veredito "blocked, needs user input" do judge é escape limpo se o codex travar em aprovações.

**Uma coisa a ter em mente:** cada continuation prompt é um turn codex novo, o que significa que o codex reavalia política de aprovação de comando do zero. Se você faz um goal longo com muitos writes, espere mais prompts de aprovação do que numa tarefa in-session única. Defina `default_permissions = ":workspace"` (que o Hermes faz automaticamente quando você habilita o runtime) para writes simples no workspace não exigirem prompt.

### Kanban (dispatch worktree multi-agent) {#kanban-multi-agent-worktree-dispatch}

**Funciona neste runtime, com uma dependência sutil.** O dispatcher kanban spawna cada worker como subprocess `hermes chat -q` separado que lê o config do usuário — o que significa que se `model.openai_runtime: codex_app_server` estiver definido globalmente, workers também sobem no runtime codex.

O que funciona dentro de um worker codex-runtime:
- Toolset completo do Codex (shell, apply_patch, update_plan, view_image, web_search) — o worker faz o trabalho real nativamente
- Plugins codex migrados — Linear, GitHub, etc.
- Callback de ferramentas Hermes para browser_*, vision, image_gen, skills, TTS

O que também funciona porque o callback MCP expõe:
- **`kanban_complete` / `kanban_block` / `kanban_comment` / `kanban_heartbeat`** — ferramentas de handoff do worker. Leem `HERMES_KANBAN_TASK` do env (definido pelo dispatcher), gateiam acesso corretamente e escrevem no SQLite DB por board fixado por `HERMES_KANBAN_DB`. Sem estes no callback, um worker neste runtime poderia fazer a tarefa mas não reportar de volta, pendurando até timeout do dispatcher.
- **`kanban_show` / `kanban_list`** — consultas read-only do board para o worker checar seu contexto.
- **`kanban_create` / `kanban_unblock` / `kanban_link`** — operações só de orchestrator. Disponíveis para agentes orchestrator no runtime codex que precisam despachar tarefas novas.

Ferramentas kanban são gated pela env var `HERMES_KANBAN_TASK` que o dispatcher define — essa var é propagada ao subprocess codex (codex herda env) e daí ao subprocess MCP `hermes-tools` spawnado. Então as ferramentas veem o task id certo e gateiam corretamente. Para workers Codex app-server, o Hermes também passa overrides estreitos de sandbox app-server quando `HERMES_KANBAN_TASK` está presente: manter sandboxing `workspace-write`, adicionar o **diretório DB do board mais todo caminho Kanban que o dispatcher fixou** como roots graváveis extras (`HERMES_KANBAN_WORKSPACES_ROOT`, `HERMES_KANBAN_WORKSPACE`, legacy `HERMES_KANBAN_ROOT` — deduplicados, DB-dir primeiro) e manter network desabilitada por padrão. Isso evita o workaround frágil `:danger-no-sandbox` enquanto deixa `kanban_complete` / `kanban_block` atualizar o board DB **e** workers escreverem reports/artefatos sob mounts de workspace fora do diretório DB (ex. `/media/.../kanban-workspaces/...` num drive separado — [issue #27941](https://github.com/NousResearch/hermes-agent/issues/27941)).

### Cron jobs {#cron-jobs}

**Não testado especificamente.** Cron jobs rodam via `cronjob` → `AIAgent.run_conversation`, o mesmo code path da CLI. Se o config do cron job tiver `openai_runtime: codex_app_server` rodará no codex. As mesmas regras de disponibilidade de ferramentas se aplicam — built-ins codex + plugins + callback MCP funcionam, ferramentas agent-loop (delegate_task, memory, session_search, todo) não. Se seu cron job depende delas, escope o cron a um perfil que usa runtime padrão.

## Trade-offs {#trade-offs}

|  | Runtime padrão Hermes | Codex app-server (opt-in) |
|---|---|---|
| Subagentes `delegate_task` | sim | indisponível — precisa de contexto do loop do agente |
| `memory`, `session_search`, `todo` | sim | indisponível — precisa de contexto do loop do agente |
| `web_search`, `web_extract` | sim | sim (via callback MCP) |
| Automação de browser (Camofox/Browserbase) | sim | sim (via callback MCP) |
| `vision_analyze`, `image_generate` | sim | sim (via callback MCP) |
| `skill_view`, `skills_list` | sim | sim (via callback MCP) |
| `text_to_speech` | sim | sim (via callback MCP) |
| Codex `shell` (terminal/read/write/search/find/run) | — | sim (built-in Codex) |
| Codex `apply_patch` (edições estruturadas multi-arquivo) | — | sim (built-in Codex) |
| Codex `update_plan` (todo in-runtime) | — | sim (built-in Codex) |
| Codex `view_image` (carregar imagem na conversa) | — | sim (built-in Codex) |
| Sandbox Codex (seatbelt/landlock, perfis) | — | sim (built-in Codex) |
| Auth de assinatura ChatGPT | — | sim (via provider `openai-codex`) |
| Plugins nativos Codex (Linear, GitHub, etc.) | — | sim (auto-migrados) |
| Servidores MCP do usuário | sim | sim (auto-migrados para codex) |
| Revisão de memory + skill (background) | sim | sim (via projeção de item) |
| Conversas multi-turn | sim | sim |
| `/goal` (loop Ralph) | sim | sim |
| Dispatch de worker kanban | sim | sim (via callback) |
| Ferramentas orchestrator kanban | sim | sim (via callback) |
| Todas as plataformas gateway | sim | sim |
| Providers não-OpenAI | sim | n/a — escopo OpenAI/Codex |

### Exibição ao vivo {#live-display}

Embora o agent loop rode dentro do subprocess Codex, o runtime
ponteia o event stream do Codex no mesmo caminho de exibição que o runtime padrão
usa:

- Deltas de assistente ao vivo, reasoning (incluindo summary deltas) e eventos estáveis de
  tool start/completion aparecem na TUI, desktop e gateways de mensagens enquanto o turn roda. O projetor de histórico só na conclusão permanece
  separado, então uma sessão retomada hidrata os mesmos tool cards mostrados durante
  o turn.
- Comentary de gateway permanece visível quando streaming de token está desabilitado, e
  eventos de ferramenta ao vivo são encaminhados mesmo para notificações drenadas antes de um
  pedido de aprovação. Commentary honra `display.show_commentary`.

## Pré-requisitos {#prerequisites}

1. **Codex CLI instalado:**
   ```bash
   npm i -g @openai/codex
   codex --version   # 0.130.0 or newer
   ```
2. **Login OAuth Codex.** O subprocess codex lê `~/.codex/auth.json`. Duas formas de popular:
   ```bash
   codex login                  # writes tokens to ~/.codex/auth.json
   ```
   O próprio `hermes auth add openai-codex` do Hermes grava em `~/.hermes/auth.json` — sessão separada. **Execute `codex login` separadamente** se ainda não fez.

3. **(Opcional) Instale os plugins Codex que quiser.** Quando você habilita o runtime, o Hermes auto-migra quaisquer plugins curated que já instalou via Codex CLI:
   ```bash
   codex plugin marketplace add openai-curated
   # then via codex's TUI, install Linear / GitHub / Gmail / etc.
   ```
   O Hermes os descobre e grava entradas `[plugins."<name>@openai-curated"]` em `~/.codex/config.toml` automaticamente.

## Habilitar {#enabling}

Numa sessão Hermes:

```
/codex-runtime codex_app_server
```

Esse comando:
- Verifica se a CLI `codex` está instalada (bloqueia com dica de install se não).
- Persiste `model.openai_runtime: codex_app_server` no seu config.yaml.
- Migra servidores MCP do usuário de `~/.hermes/config.yaml` para `~/.codex/config.toml`.
- **Descobre e migra plugins nativos Codex instalados** (Linear, GitHub, Gmail, Calendar, Canva, etc.) consultando o RPC `plugin/list` do Codex.
- **Registra as próprias ferramentas do Hermes como servidor MCP** para o subprocess codex chamar de volta por ferramentas que o codex não envia.
- **Grava `default_permissions = ":workspace"`** para o sandbox permitir writes no workspace sem prompt a cada operação.
- Diz o que foi migrado. Entra em vigor na **próxima** sessão — o agent em cache atual mantém o runtime anterior para prompt caches permanecerem válidos.

Sinônimos: `/codex-runtime on`, `/codex-runtime off`, `/codex-runtime auto`.

Para checar estado atual sem mudar nada:
```
/codex-runtime
```

Você também pode definir manualmente em `~/.hermes/config.yaml`:
```yaml
model:
  openai_runtime: codex_app_server   # default is "auto" (= Hermes runtime)
```

## Loop de autoaperfeiçoamento (memory + skill nudges) {#self-improvement-loop-memory-skill-nudges}

O autoaperfeiçoamento em segundo plano do Hermes dispara em limiares de contador:

- A cada 10 prompts user → um agente de revisão forkado olha a conversa e decide se algo deve ir para memory.
- A cada 10 iterações de ferramenta num único turn → mesma ideia mas para skills (writes `skill_manage`).

**Ambos continuam funcionando no runtime codex.** O caminho codex projeta cada item `commandExecution` / `fileChange` / `mcpToolCall` / `dynamicToolCall` completado numa mensagem sintética `assistant tool_call` + `tool` result, então quando a revisão roda vê a mesma forma que no runtime Hermes padrão.

Como a fiação permanece equivalente:

| | Runtime padrão | Runtime Codex |
|---|---|---|
| `_turns_since_memory` incrementa | por prompt user, no pre-loop de run_conversation | mesmo caminho de código, antes do early-return |
| `_iters_since_skill` incrementa | por iteração de ferramenta no loop chat-completions | por `turn.tool_iterations` após o turn codex retornar |
| Gatilho de memory (`_turns_since_memory >= _memory_nudge_interval`) | calculado no pre-loop, dispara após resposta | calculado no pre-loop, repassado ao helper codex |
| Gatilho de skill (`_iters_since_skill >= _skill_nudge_interval`) | calculado após o loop | calculado após o turn codex |
| `_spawn_background_review(messages_snapshot=..., review_memory=..., review_skills=...)` | chamado quando qualquer gatilho dispara | chamado identicamente quando qualquer gatilho dispara |

Um detalhe: o fork de revisão em si precisa chamar ferramentas agent-loop do Hermes (`memory`, `skill_manage`), que exigem dispatch próprio do Hermes. Então quando o agent pai está em `codex_app_server`, o fork de revisão é **rebaixado para `codex_responses`** — mesmas credenciais OAuth, mesmo provider `openai-codex`, mas fala com a Responses API da OpenAI diretamente para o Hermes possuir o loop e as ferramentas agent-loop funcionarem. Invisível ao usuário.

Efeito líquido: habilite o runtime codex e seus nudges de memory + skill continuam disparando exatamente como de outra forma.

## Como aprovações funcionam {#how-approvals-work}

O Codex pede aprovação antes de executar comandos ou aplicar patches. Estes viram o prompt padrão "Dangerous Command" do Hermes:

```
╭───────────────────────────────────────╮
│ Dangerous Command                     │
│                                       │
│ /bin/bash -lc 'echo hello > foo.txt'  │
│                                       │
│ ❯ 1. Allow once                       │
│   2. Allow for this session           │
│   3. Deny                             │
│                                       │
│ Codex requests exec in /your/cwd      │
╰───────────────────────────────────────╯
```

- **Allow once** → aprova este comando único.
- **Allow for this session** → Codex não re-prompta comandos similares.
- **Deny** → comando rejeitado; Codex continua em modo read-only.

Para aprovações `apply_patch` (edição de arquivo), o Hermes mostra resumo do que mudou (`1 add, 1 update: /tmp/new.py, /tmp/old.py`) quando o codex fornece os dados via item `fileChange` correspondente.

## Perfis de permissão {#permission-profiles}

O Codex tem três perfis de permissão built-in:
- `:read-only` — sem writes; todo comando shell exige aprovação
- `:workspace` — writes no workspace atual permitidos sem prompts (padrão Hermes quando você habilita o runtime)
- `:danger-no-sandbox` — sem sandbox algum (não use a menos que entenda)

Você pode sobrescrever o padrão em `~/.codex/config.toml` fora do bloco gerenciado do Hermes:

```toml
default_permissions = ":read-only"
```

(O Hermes preserva seu override na re-migração enquanto estiver fora dos marcadores `# managed by hermes-agent`.)

## Tarefas auxiliares e custo de token de assinatura ChatGPT {#auxiliary-tasks-and-chatgpt-subscription-token-cost}

Com este runtime ligado e provider `openai-codex`, **tarefas auxiliares (geração de título, compressão de contexto, auto-detect vision, fork de revisão de autoaperfeiçoamento em segundo plano) também fluem pela assinatura ChatGPT por padrão**, porque o auxiliary client do Hermes usa provider/modelo principal quando não há override por tarefa.

Isso não é específico de `codex_app_server` — também vale para o caminho `codex_responses` existente — mas fica mais visível aqui porque você opta explicitamente pela cobrança de assinatura.

Para rotear tarefas aux específicas a modelo mais barato / diferente, defina overrides explícitos em `~/.hermes/config.yaml`:

```yaml
auxiliary:
  title_generation:
    provider: openrouter
    model: google/gemini-3-flash-preview
  compression:
    provider: openrouter
    model: google/gemini-3-flash-preview
  vision:
    provider: openrouter
    model: google/gemini-3-flash-preview
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

O fork de revisão de autoaperfeiçoamento herda o runtime principal via `_current_main_runtime()` e o Hermes rebaixa de `codex_app_server` para `codex_responses` automaticamente (para o fork poder chamar `memory` e `skill_manage` — ferramentas agent-loop do Hermes). Esse fork ainda usa auth de assinatura a menos que você tenha roteado tarefas aux para outro lugar.

## Editar `~/.codex/config.toml` com segurança {#editing-codexconfigtoml-safely}

O Hermes envolve tudo que gerencia entre dois comentários marcadores:

```toml
# managed by hermes-agent — `hermes codex-runtime migrate` regenerates this section
default_permissions = ":workspace"
[mcp_servers.filesystem]
...
[plugins."github@openai-curated"]
...
# end hermes-agent managed section
```

Qualquer coisa **fora** desse bloco é sua. Re-executar migração (via `/codex-runtime codex_app_server` ou sempre que alternar o runtime) substitui o bloco gerenciado no lugar mas preserva conteúdo user acima e abaixo verbatim. Isso significa que você pode:

- Adicionar seus próprios servidores MCP que o Hermes não conhece
- Sobrescrever `default_permissions` para `:read-only` se preferir ser promptado
- Configurar opções só-codex (model, providers, otel, etc.)
- Adicionar perfis de permissão definidos pelo usuário em tabelas `[permissions.<name>]`

Qualquer coisa que adicionar **dentro** do bloco gerenciado será sobrescrita na próxima migração. Se precisar de ajuste que exige editar o bloco gerenciado, abra uma issue e adicionaremos o knob.

## Multi-profile / setups multi-tenant {#multi-profile-multi-tenant-setups}

Por padrão, o Hermes aponta o subprocess codex para `~/.codex/` independente de qual perfil Hermes está ativo. Isso significa que `hermes -p work` e `hermes -p personal` compartilham a mesma auth, plugins e config Codex. Para a maioria dos usuários é o comportamento certo — combina com rodar `codex` CLI diretamente.

Se quiser isolamento Codex por perfil (auth separada, plugins instalados separados, config separada), defina `CODEX_HOME` explicitamente por perfil. A forma mais limpa é apontar a um diretório sob seu `HERMES_HOME`:

```bash
# Dentro do perfil work, você pode envolver o hermes:
CODEX_HOME=~/.hermes/profiles/work/codex hermes chat
```

Você precisará re-executar `codex login` uma vez com esse `CODEX_HOME` definido para tokens OAuth caírem no local com escopo de perfil. Depois disso, `hermes -p work` opera em estado Codex isolado.

Não auto-escopamos porque mover o `~/.codex/` existente de um usuário invalidaria silenciosamente a auth Codex CLI — quem já rodou `codex login` teria que re-autenticar. Opt-in parece mais seguro que surpreender usuários.

## Passthrough da variável de ambiente HOME {#home-environment-variable-passthrough}

O Hermes NÃO reescreve `HOME` ao spawnar o subprocess codex app-server (usamos `os.environ.copy()` e só sobrepomos `CODEX_HOME` e `RUST_LOG`). Isso significa:

- Comandos que o codex roda via ferramenta `shell` veem o `HOME` real do usuário e encontram `~/.gitconfig`, `~/.gh/`, `~/.aws/`, `~/.npmrc`, etc. corretamente.
- Estado interno do Codex permanece isolado via `CODEX_HOME` (que aponta para `~/.codex/` por padrão).

Combina com o limite que o OpenClaw chegou após experimentação inicial: isolar estado do Codex, deixar home do usuário em paz. (Cf. openclaw/openclaw#81562.)

## Migração de servidor MCP {#mcp-server-migration}

O config `mcp_servers` do Hermes é auto-traduzido para o formato TOML que o Codex espera. A migração roda toda vez que você habilita o runtime e é idempotente — re-runs substituem a seção gerenciada mas preservam config Codex editada pelo usuário.

O que traduz:

| Hermes (`config.yaml`) | Codex (`config.toml`) |
|---|---|
| `command` + `args` + `env` | stdio transport |
| `url` + `headers` | streamable_http transport |
| `timeout` | `tool_timeout_sec` |
| `connect_timeout` | `startup_timeout_sec` |
| `enabled: false` | `enabled = false` |

O que não é migrado:
- Chaves específicas Hermes como `sampling` (client MCP do Codex não tem equivalente — são descartadas com aviso por servidor).

## Migração de plugin nativo Codex {#native-codex-plugin-migration}

Plugins instalados via `codex plugin` (Linear, GitHub, Gmail, Calendar, Canva, etc.) são descobertos pelo RPC `plugin/list` do Codex. Para cada plugin com `installed: true`, o Hermes grava um bloco `[plugins."<name>@openai-curated"]` habilitando-o na sessão Hermes.

Isso significa: quando seu amigo diz "tenho Calendar e GitHub configurados no meu Codex CLI" e habilita o runtime codex do Hermes, o Hermes os ativa automaticamente. Sem reconfiguração.

O que NÃO é migrado:
- Plugins que você ainda não instalou — instale no Codex primeiro.
- Plugins onde o codex reporta `availability != AVAILABLE` (install quebrado, OAuth expirado, removido do marketplace, etc.). São pulados para evitar escrever config que falharia na ativação.
- Entradas do marketplace de apps ChatGPT (resultados `app/list` por conta — já habilitadas dentro do codex pela auth da conta).
- OAuth de plugin — você autoriza cada plugin uma vez no próprio Codex; o Hermes não toca credenciais.

## Callback de ferramentas Hermes (o novo servidor MCP) {#hermes-tool-callback-the-new-mcp-server}

O toolset built-in do Codex cobre shell/file ops/patches mas não tem web search, automação de browser, vision, geração de imagem, etc. Para manter estes usáveis num turn codex, o Hermes se registra como servidor MCP em `~/.codex/config.toml`:

```toml
[mcp_servers.hermes-tools]
command = "/path/to/python"
args = ["-m", "agent.transports.hermes_tools_mcp_server"]
env = { HERMES_HOME = "/your/.hermes", PYTHONPATH = "...", HERMES_QUIET = "1" }
startup_timeout_sec = 30.0
tool_timeout_sec = 600.0
```

Quando o modelo chama `web_search` (ou outra ferramenta Hermes exposta), o codex spawna o subprocess `hermes_tools_mcp_server` via stdio, a requisição é despachada por `model_tools.handle_function_call()` e o resultado é projetado de volta ao codex como qualquer resposta MCP.

**Ferramentas disponíveis via callback:** `web_search`, `web_extract`, `browser_navigate`, `browser_click`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_scroll`, `browser_back`, `browser_get_images`, `browser_console`, `browser_vision`, `vision_analyze`, `image_generate`, `skill_view`, `skills_list`, `text_to_speech`.

**Ferramentas NÃO disponíveis:** `delegate_task`, `memory`, `session_search`, `todo`. Precisam do contexto AIAgent em execução para despachar (estado mid-loop) e um callback MCP stateless não pode dirigí-las. Use o runtime Hermes padrão (`/codex-runtime auto`) quando precisar delas.

## Desabilitar {#disabling}

Volte a qualquer momento:

```
/codex-runtime auto
```

Efetivo na próxima sessão. O bloco gerenciado Codex permanece em `~/.codex/config.toml` para re-habilitar depois sem perder config — ou remova manualmente se preferir.

## Limitações {#limitations}

Este runtime é **beta opt-in**. Funcionando a partir de Hermes Agent 2026.5 + Codex CLI 0.130.0:

- Conversas multi-turn
- Aprovações `commandExecution` e `fileChange` (apply_patch) via UI Hermes
- MCP tool calls (verificado contra `@modelcontextprotocol/server-filesystem` e o novo callback `hermes-tools`)
- Migração de plugin nativo Codex (verificado contra inventário Linear / GitHub / Calendar)
- Caminhos deny/cancel
- Ciclo toggle on/off
- Contadores de nudge memory e skill (verificado live via testes de integração)
- Hermes web_search through codex (verificado live: "OpenAI Codex CLI – Getting Started" retornou end-to-end)

Limitações conhecidas:

- **Auth Hermes e auth codex são sessões separadas.** Você precisa de `codex login` E `hermes auth add openai-codex` para a UX mais limpa (o runtime usa sessão codex para a chamada LLM). Escolha deliberada de design em `_import_codex_cli_tokens` do Hermes — não compartilha estado OAuth com codex CLI para evitar sobrescrever um ao outro no refresh de token.
- **`delegate_task`, `memory`, `session_search`, `todo` indisponíveis neste runtime.** Precisam do contexto AIAgent em execução que um callback MCP stateless não fornece. Use `/codex-runtime auto` quando precisar delas.
- **Sem preview inline de patch nos prompts de aprovação quando codex não rastreia changeset.** Params de aprovação `fileChange` do Codex nem sempre carregam changeset. O Hermes cacheia dados da notificação `item/started` correspondente quando possível, mas se aprovação chega antes do item ter streamed, o prompt cai no que `reason` codex fornece.
- **Cancelamento sub-segundo não garantido.** Interrupts mid-stream (Ctrl+C enquanto codex responde) são enviados via `turn/interrupt`, mas se codex já flushou a mensagem final, você recebe a resposta mesmo assim.

Se achar um bug, [abra uma issue](https://github.com/NousResearch/hermes-agent/issues) com a saída de `hermes logs --since 5m`. Mencione `codex-runtime` no título para triagem fácil.

## Arquitetura {#architecture}

```
                ┌─── Hermes shell (CLI / TUI / gateway) ───┐
                │  sessions DB · slash commands · memory   │
                │  & skill review · cron · session pickers │
                └──┬──────────────────────────────────────┬┘
                   │ user_message               final     │
                   ▼                            text +    │
        ┌──────────────────────────────────┐   projected  │
        │  AIAgent.run_conversation()       │   messages   │
        │   if api_mode == codex_app_server │              │
        │     → CodexAppServerSession       │              │
        │   else: chat_completions / codex_responses (default)
        └────┬─────────────────────────────┘              │
             │ JSON-RPC over stdio                        │
             ▼                                            │
        ┌──────────────────────────────────┐              │
        │  codex app-server (subprocess)    │──────────────┘
        │   thread/start, turn/start        │
        │   item/* notifications            │
        │   shell + apply_patch + update_plan│
        │   view_image + sandbox            │
        │   ┌─────────────────────────┐     │
        │   │  MCP client             │     │
        │   │  ├─ user MCP servers    │     │
        │   │  ├─ native plugins      │     │
        │   │  │   (linear, github,   │     │
        │   │  │    gmail, calendar,  │     │
        │   │  │    canva, ...)       │     │
        │   │  └─ hermes-tools ───────┼─────────────────┐
        │   │       (callback to     │     │           │
        │   │        Hermes' richer  │     │           │
        │   │        tools)          │     │           │
        │   └─────────────────────────┘     │           │
        └──────────────────────────────────┘           │
                                                        │
                                                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │  hermes_tools_mcp_server.py (subprocess on demand)        │
        │   web_search, web_extract, browser_*, vision_analyze,    │
        │   image_generate, skill_view, skills_list, text_to_speech│
        └──────────────────────────────────────────────────────────┘
```

Para detalhes de implementação, veja [PR #24182](https://github.com/NousResearch/hermes-agent/pull/24182) e o [Codex app-server protocol README](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md).
