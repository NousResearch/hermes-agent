---
sidebar_position: 11
sidebar_label: "Plugins"
title: "Plugins"
description: "Estenda o Hermes com tools, hooks e integrações customizadas via o sistema de plugins"
---

# Plugins

O Hermes tem um sistema de plugins para adicionar tools, hooks e integrações customizadas sem modificar o código core.

Se você quer criar uma tool customizada para si, sua equipe ou um projeto,
este costuma ser o caminho certo. A página [Adding Tools](/developer-guide/adding-tools) do guia do desenvolvedor é para tools core built-in do Hermes que vivem em `tools/` e `toolsets.py`.

**→ [Build a Hermes Plugin](/developer-guide/plugins)** — guia passo a passo com exemplo completo funcionando.

## Quick overview {#quick-overview}

Coloque um diretório em `~/.hermes/plugins/` com `plugin.yaml` e código Python:

```
~/.hermes/plugins/my-plugin/
├── plugin.yaml      # manifest
├── __init__.py      # register() — wires schemas to handlers
├── schemas.py       # tool schemas (what the LLM sees)
└── tools.py         # tool handlers (what runs when called)
```

Inicie o Hermes — suas tools aparecem junto das built-in. O modelo pode chamá-las imediatamente.

### Minimal working example {#minimal-working-example}

Aqui está um plugin completo que adiciona a tool `hello_world` e registra toda tool call via hook.

**`~/.hermes/plugins/hello-world/plugin.yaml`**

```yaml
name: hello-world
version: "1.0"
description: A minimal example plugin
```

**`~/.hermes/plugins/hello-world/__init__.py`**

```python
"""Minimal Hermes plugin — registers a tool and a hook."""

import json


def register(ctx):
    # --- Tool: hello_world ---
    schema = {
        "name": "hello_world",
        "description": "Returns a friendly greeting for the given name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                }
            },
            "required": ["name"],
        },
    }

    def handle_hello(params, **kwargs):
        del kwargs
        name = params.get("name", "World")
        return json.dumps({"success": True, "greeting": f"Hello, {name}!"})

    ctx.register_tool(
        name="hello_world",
        toolset="hello_world",
        schema=schema,
        handler=handle_hello,
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

Coloque ambos os arquivos em `~/.hermes/plugins/hello-world/`, reinicie o Hermes, e o modelo pode chamar `hello_world` imediatamente. O hook imprime uma linha de log após cada invocação de tool.

A descrição da tool voltada ao modelo pertence em `schema["description"]`. O valor opcional `ctx.register_tool(description=...)` é metadata de registry `ToolEntry` separada: quando omitido, defaulta para a descrição do schema, mas o Hermes não copia de volta para um schema que não tem `description`. Prefira definir o texto uma vez no schema. Se você fornecer ambos os valores, mantenha-os sincronizados; o modelo vê o valor do schema.

Plugins locais de projeto em `./.hermes/plugins/` ficam desabilitados por padrão. Habilite-os só para repositórios confiáveis definindo `HERMES_ENABLE_PROJECT_PLUGINS=true` antes de iniciar o Hermes.

## What plugins can do {#what-plugins-can-do}

Toda API `ctx.*` abaixo está disponível dentro da função `register(ctx)` de um plugin.

| Capability | How |
|-----------|-----|
| Add tools | `ctx.register_tool(name=..., toolset=..., schema=..., handler=...)` |
| Add hooks | `ctx.register_hook("post_tool_call", callback)` |
| Add slash commands | `ctx.register_command(name, handler, description)` — adds `/name` in CLI and gateway sessions |
| Dispatch tools from commands | `ctx.dispatch_tool(name, args)` — invokes a registered tool with parent-agent context auto-wired |
| Add CLI commands | `ctx.register_cli_command(name, help, setup_fn, handler_fn)` — adds `hermes <plugin> <subcommand>` |
| Inject messages | `ctx.inject_message(content, role="user", session_key=...)` - see [Injecting Messages](#injecting-messages) |
| Ship data files | `Path(__file__).parent / "data" / "file.yaml"` |
| Bundle skills | `ctx.register_skill(name, path)` — namespaced as `plugin:skill`, loaded via `skill_view("plugin:skill")` |
| Gate on env vars | `requires_env: [API_KEY]` in plugin.yaml — prompted during `hermes plugins install` |
| Distribute via pip | `[project.entry-points."hermes_agent.plugins"]` |
| Register a gateway platform (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — see [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| Register an image-generation backend | `ctx.register_image_gen_provider(provider)` — see [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| Register a video-generation backend | `ctx.register_video_gen_provider(provider)` — see [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| Register a context-compression engine | `ctx.register_context_engine(engine)` — see [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| Route human approval prompts | `ctx.register_approval_transport(name, present_fn)` — see [Approval transports](#approval-transports) |
| Register a memory backend | Subclass `MemoryProvider` in `plugins/memory/<name>/__init__.py` — see [Memory Provider Plugins](/developer-guide/memory-provider-plugin) (uses a separate discovery system) |
| Run a host-owned LLM call | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — borrow the user's active model + auth for a one-shot completion with optional JSON schema validation. See [Plugin LLM Access](/developer-guide/plugin-llm-access) |
| Call an MCP tool (capability-gated) | `ctx.call_mcp(server, tool, arguments, timeout=30)` — see [Calling MCP servers from plugins](#calling-mcp-servers-from-plugins) |
| Register an inference backend (LLM provider) | `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/__init__.py` — see [Model Provider Plugins](/developer-guide/model-provider-plugin) (uses a separate discovery system) |

## Plugin discovery {#plugin-discovery}

| Source | Path | Use case |
|--------|------|----------|
| Bundled | `<repo>/plugins/` | Ships with Hermes — see [Built-in Plugins](/user-guide/features/built-in-plugins) |
| User | `~/.hermes/plugins/` | Personal plugins |
| Project | `.hermes/plugins/` | Project-specific plugins (requires `HERMES_ENABLE_PROJECT_PLUGINS=true`) |
| pip | `hermes_agent.plugins` entry_points | Distributed packages |
| Nix | `services.hermes-agent.extraPlugins` / `extraPythonPackages` | NixOS declarative installs — see [Nix Setup](/getting-started/nix-setup#plugins) |

Fontes posteriores sobrescrevem anteriores em colisão de nome, então um plugin do usuário com o mesmo nome de um bundled o substitui.

### Plugin sub-categories {#plugin-sub-categories}

Dentro de cada fonte, o Hermes também reconhece subdiretórios de categoria que roteiam plugins para sistemas de descoberta especializados:

| Sub-directory | What it holds | Discovery system |
|---|---|---|
| `plugins/` (root) | General plugins — tools, hooks, slash commands, CLI commands, bundled skills | `PluginManager` (kind: `standalone` or `backend`) |
| `plugins/platforms/<name>/` | Gateway channel adapters (`ctx.register_platform()`) | `PluginManager` (kind: `platform`, one level deeper) |
| `plugins/image_gen/<name>/` | Image-generation backends (`ctx.register_image_gen_provider()`) | `PluginManager` (kind: `backend`, one level deeper) |
| `plugins/memory/<name>/` | Memory providers (subclass `MemoryProvider`) | **Own loader** in `plugins/memory/__init__.py` (kind: `exclusive` — one active at a time) |
| `plugins/context_engine/<name>/` | Context-compression engines (`ctx.register_context_engine()`) | **Own loader** in `plugins/context_engine/__init__.py` (one active at a time) |
| `plugins/model-providers/<name>/` | LLM provider profiles (`register_provider(ProviderProfile(...))`) | **Own loader** in `providers/__init__.py` (lazily scanned on first `get_provider_profile()` call) |

Plugins do usuário em `~/.hermes/plugins/model-providers/<name>/` e `~/.hermes/plugins/memory/<name>/` sobrescrevem plugins bundled com o mesmo nome — last-writer-wins em `register_provider()` / `register_memory_provider()`. Coloque um diretório e ele substitui o built-in sem edits no repo.

## Plugins are opt-in (with a few exceptions) {#plugins-are-opt-in-with-a-few-exceptions}

**Plugins gerais e backends instalados pelo usuário ficam desabilitados por padrão** — a descoberta os encontra (aparecem em `hermes plugins` e `/plugins`), mas nada com hooks ou tools carrega até você adicionar o nome do plugin a `plugins.enabled` em `~/.hermes/config.yaml`. Isso impede que código de terceiros rode sem seu consentimento explícito.

```yaml
plugins:
  enabled:
    - my-tool-plugin
    - disk-cleanup
  disabled:       # optional deny-list — always wins if a name appears in both
    - noisy-plugin
```

Três formas de mudar estado:

```bash
hermes plugins                    # interactive toggle (space to check/uncheck)
hermes plugins enable <name>      # add to allow-list
hermes plugins disable <name>     # remove from allow-list + add to disabled
```

Após `hermes plugins install owner/repo`, você é perguntado `Enable 'name' now? [y/N]` — padrão é não. Pule o prompt para installs scriptados com `--enable` ou `--no-enable`.

Para um install reproduzível, pin um commit imutável completo (tags, branches e
SHAs abreviados não são aceitos):

```bash
hermes plugins install owner/repo --ref 0123456789abcdef0123456789abcdef01234567
```

O Hermes faz checkout do commit detached, verifica que `HEAD` corresponde exatamente ao
SHA pedido, e registra a fonte canônica, revisão instalada e status de pin
no profile atual. `hermes plugins update` recusa mover um plugin pinned;
escolha um commit exato novo explicitamente com
`hermes plugins install <source> --force --ref <new-commit>`. Os
metadados de install locais do profile não contêm valores de config, valores de environment,
secrets ou grants de capability.

### What the allow-list does NOT gate {#what-the-allow-list-does-not-gate}

Várias categorias de plugin contornam `plugins.enabled` — fazem parte da superfície built-in do Hermes e quebrariam funcionalidade básica se desligadas por padrão:

| Plugin kind | How it's activated instead |
|---|---|
| **Bundled platform plugins** (IRC, Teams, etc. under `plugins/platforms/`) | Auto-loaded so every shipped gateway channel is available. The actual channel turns on via `gateway.platforms.<name>.enabled` in `config.yaml`. |
| **Bundled backends** (image-gen providers under `plugins/image_gen/`, etc.) | Auto-loaded so the default backend "just works". Selection happens via `<category>.provider` in `config.yaml` (e.g. `image_gen.provider: openai`). |
| **Memory providers** (`plugins/memory/`) | All discovered; exactly one is active, chosen by `memory.provider` in `config.yaml`. |
| **Context engines** (`plugins/context_engine/`) | All discovered; one is active, chosen by `context.engine` in `config.yaml`. |
| **Model providers** (`plugins/model-providers/`) | All bundled providers under `plugins/model-providers/` discover and register at the first `get_provider_profile()` call. The user picks one at a time via `--provider` or `config.yaml`. |
| **Pip-installed `backend` plugins** | Opt-in via `plugins.enabled` (same as general plugins). |
| **User-installed platforms** (under `~/.hermes/plugins/platforms/`) | Opt-in via `plugins.enabled` — third-party gateway adapters need explicit consent. |

Em resumo: **infraestrutura bundled "always-works" carrega automaticamente; plugins gerais de terceiros são opt-in.** A allow-list `plugins.enabled` é o gate especificamente para código arbitrário que você coloca em `~/.hermes/plugins/`.

### Approval transports {#approval-transports}

Um approval transport muda **onde um humano vê e responde** um request
existente de aprovação de ferramenta Hermes. Não decide se um comando precisa
de aprovação e não é uma API de política de autorização.

```python
def present(request):
    # Deliver request.command and request.description to your UI, wait for
    # its authenticated human response, then return a request-bound decision.
    choice = send_to_my_ui_and_wait(request)  # once/session/always/deny
    return request.respond(choice)


def register(ctx):
    ctx.register_approval_transport("my-ui", present)
```

`present` pode ser síncrono ou async. O Hermes o roda num worker limitado e
aplica o `approvals.timeout` canônico mesmo se o plugin não o fizer. O
request é imutável e contém texto de display redigido, sua classe de apresentação host
(`cli` ou `gateway`), o timeout do host, choices permitidas, e um ID/digest
opaco de request.
Retorne o resultado de
`request.respond(choice)`; dicts unbound e IDs/digests stale ou mudados
são rejeitados. Um plugin não pode retornar um scope que o host não
ofereceu (por exemplo, `always` num request once-only).

Registration sozinha não faz nada. Habilitar o plugin e selecionar
explicitamente seu transport são passos de consentimento separados:

```yaml
plugins:
  enabled: [my-approval-plugin]

security:
  approval:
    transport: my-ui
    transport_fallback: deny     # default
```

Exceções de transport, timeouts, registrations indisponíveis, choices inválidas,
e respostas stale negam por padrão. Para mostrar de propósito o prompt na
superfície CLI/TUI/gateway/ACP ordinária quando o transport selecionado falha, set
`transport_fallback: builtin`. Sem esse opt-in exato, o Hermes nunca
materializa o prompt em outra superfície.

O Hermes ainda dono de hardline blocks, proteção sudo-stdin, regras de deny do usuário,
binding de request, scopes permitidos, persistence, hooks e autorização final.
Comandos hardline são blocked antes de qualquer callback de transport. Há
intencionalmente **nenhuma política de aprovação plugin, callback auto-allow, ou
`pre_tool_call` policy obrigatória** nesta interface. Uma capability futura de
política de aprovação pode usar o modelo de capability-consent de plugin, mas seleção de
transport não a concede.

### Migration for existing users {#migration-for-existing-users}

Quando você atualiza para uma versão do Hermes com plugins opt-in (config schema v21+), plugins do usuário já instalados em `~/.hermes/plugins/` que não estavam em `plugins.disabled` são **automaticamente grandfathered** em `plugins.enabled`. Sua config existente continua funcionando. Plugins standalone bundled NÃO são grandfathered — mesmo usuários existentes precisam optar explicitamente. (Plugins bundled platform/backend nunca precisaram de grandfathering porque nunca foram gated.)

## Available hooks {#available-hooks}

Plugins podem registrar os 26 eventos de lifecycle atualmente aceitos por `hermes_cli.plugins.VALID_HOOKS`. O **[catálogo Event Hooks](/user-guide/features/hooks#shipped-plugin-hook-catalog)** é canônico para timing exato, handling de return, campos de payload e notas de privacidade.

| Categoria descritiva | Hooks shipped |
|---|---|
| **Directive/control** | `pre_tool_call`, `pre_llm_call`, `pre_verify`, `pre_gateway_dispatch` |
| **Transform** | `transform_tool_result`, `transform_terminal_output`, `transform_llm_output`, `pre_transcription` |
| **Observer** | `post_tool_call`, `post_llm_call`, `pre_api_request`, `post_api_request`, `api_request_error`, `on_stream_start`, `on_stream_delta`, `on_stream_end`, `on_interim_message`, `on_session_start`, `on_session_end`, `on_session_finalize`, `on_session_reset`, `on_skill_lifecycle`, `subagent_start`, `subagent_stop`, `pre_approval_request`, `post_approval_response`, `pre_command`, `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked` |

Estas categorias descrevem o comportamento atual em vez de definir regras futuras de nomeação. Plugin middleware permanece um registry/superfície separado.

## Plugin types {#plugin-types}

O Hermes tem quatro tipos de plugins:

| Type | What it does | Selection | Location |
|------|-------------|-----------|----------|
| **General plugins** | Add tools, hooks, slash commands, CLI commands | Multi-select (enable/disable) | `~/.hermes/plugins/` |
| **Memory providers** | Replace or augment built-in memory | Single-select (one active) | `plugins/memory/` |
| **Context engines** | Replace the built-in context compressor | Single-select (one active) | `plugins/context_engine/` |
| **Model providers** | Declare an inference backend (OpenRouter, Anthropic, …) | Multi-register, picked by `--provider` / `config.yaml` | `plugins/model-providers/` |

Memory providers e context engines são **provider plugins** — só um de cada tipo pode estar ativo por vez. Model providers também são plugins, mas muitos carregam simultaneamente; o usuário escolhe um por vez via `--provider` ou `config.yaml`. Plugins gerais podem ser habilitados em qualquer combinação.

## Pluggable interfaces — where to go for each {#pluggable-interfaces--where-to-go-for-each}

A tabela acima mostra as quatro categorias de plugin, mas dentro de "General plugins" o `PluginContext` expõe vários pontos de extensão distintos — e o Hermes também aceita extensões fora do sistema de plugins Python (backends config-driven, comandos com shell hooks, servidores externos, etc.). Use esta tabela para encontrar o doc certo para o que você quer construir:

| Want to add… | How | Authoring guide |
|---|---|---|
| A **tool** the LLM can call | Python plugin — `ctx.register_tool()` | [Build a Hermes Plugin](/developer-guide/plugins) · [Adding Tools](/developer-guide/adding-tools) |
| A **lifecycle hook** (pre/post LLM, session start/end, tool filter) | Python plugin — `ctx.register_hook()` | [Hooks reference](/user-guide/features/hooks) · [Build a Hermes Plugin](/developer-guide/plugins) |
| A **slash command** for the CLI / gateway | Python plugin — `ctx.register_command()` | [Build a Hermes Plugin](/developer-guide/plugins) · [Extending the CLI](/developer-guide/extending-the-cli) |
| A **subcommand** for `hermes <thing>` | Python plugin — `ctx.register_cli_command()` | [Extending the CLI](/developer-guide/extending-the-cli) |
| A bundled **skill** that your plugin ships | Python plugin — `ctx.register_skill()` | [Creating Skills](/developer-guide/creating-skills) |
| An **inference backend** (LLM provider: OpenAI-compat, Codex, Anthropic-Messages, Bedrock) | Provider plugin — `register_provider(ProviderProfile(...))` in `plugins/model-providers/<name>/` | **[Model Provider Plugins](/developer-guide/model-provider-plugin)** · [Adding Providers](/developer-guide/adding-providers) |
| A **gateway channel** (Discord / Telegram / IRC / Teams / etc.) | Platform plugin — `ctx.register_platform()` in `plugins/platforms/<name>/` | [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| A **memory backend** (Honcho, Mem0, Supermemory, …) | Memory plugin — subclass `MemoryProvider` in `plugins/memory/<name>/` | [Memory Provider Plugins](/developer-guide/memory-provider-plugin) |
| A **context-compression strategy** | Context-engine plugin — `ctx.register_context_engine()` | [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| An **image-generation backend** (DALL·E, SDXL, …) | Backend plugin — `ctx.register_image_gen_provider()` | [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| A **video-generation backend** (Veo, Kling, Pixverse, Grok-Imagine, Runway, …) | Backend plugin — `ctx.register_video_gen_provider()` | [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| A **TTS backend** (any CLI — Piper, VoxCPM, Kokoro, xtts, voice-cloning scripts, …) | Config-driven (recommended) — declare under `tts.providers.<name>` with `type: command` in `config.yaml`. OR Python backend plugin — `ctx.register_tts_provider()` for Python-SDK / streaming engines that need more than a shell template. | [TTS Setup](/user-guide/features/tts#custom-command-providers) · [Python plugin guide](/user-guide/features/tts#python-plugin-providers) |
| An **STT backend** (any CLI — whisper.cpp, custom whisper binary, local ASR CLI) | Config-driven (recommended) — declare under `stt.providers.<name>` with `type: command` in `config.yaml`, or set `HERMES_LOCAL_STT_COMMAND` for the legacy single-command escape hatch. OR Python backend plugin — `ctx.register_transcription_provider()` for Python-SDK engines (OpenRouter, SenseAudio, Gemini-STT, etc.). | [STT Setup](/user-guide/features/tts#stt-custom-command-providers) · [Python plugin guide](/user-guide/features/tts#python-plugin-providers-stt) |
| **External tools via MCP** (filesystem, GitHub, Linear, Notion, any MCP server) | Config-driven — declare `mcp_servers.<name>` with `command:` / `url:` in `config.yaml`. Hermes auto-discovers the server's tools and registers them alongside built-ins. | [MCP](/user-guide/features/mcp) |
| **Additional skill sources** (custom GitHub repos, private skill indexes) | CLI — `hermes skills tap add <repo>` | [Skills Hub](/user-guide/features/skills#skills-hub) · [Publishing a custom tap](/user-guide/features/skills#publishing-a-custom-skill-tap) |
| **Gateway event hooks** (fire on `gateway:startup`, `session:start`, `agent:end`, `command:*`) | Drop `HOOK.yaml` + `handler.py` into `~/.hermes/hooks/<name>/` | [Event Hooks](/user-guide/features/hooks#gateway-event-hooks) |
| **Shell hooks** (run a shell command on events — notifications, audit logs, desktop alerts) | Config-driven — declare under `hooks:` in `config.yaml` | [Shell Hooks](/user-guide/features/hooks#shell-hooks) |

:::note
Nem tudo é plugin Python. Algumas superfícies de extensão usam intencionalmente **comandos shell config-driven** (TTS, STT, shell hooks) para que qualquer CLI que você já tenha vire plugin sem escrever Python. Outras são **servidores externos** (MCP) aos quais o agente se conecta e auto-registra tools. E algumas são **diretórios drop-in** (gateway hooks) com formato de manifest próprio. Escolha a superfície certa para o estilo de integração do seu caso; os guias de autoria na tabela acima cobrem placeholders, descoberta e exemplos.
:::

## NixOS declarative plugins {#nixos-declarative-plugins}

No NixOS, plugins podem ser instalados declarativamente via opções do módulo — sem `hermes plugins install`. Veja o **[guia Nix Setup](/getting-started/nix-setup#plugins)** para detalhes completos.

```nix
services.hermes-agent = {
  # Directory plugin (source tree with plugin.yaml)
  extraPlugins = [ (pkgs.fetchFromGitHub { ... }) ];
  # Entry-point plugin (pip package)
  extraPythonPackages = [ (pkgs.python312Packages.buildPythonPackage { ... }) ];
  # Enable in config
  settings.plugins.enabled = [ "my-plugin" ];
};
```

Plugins declarativos são symlinkados com prefixo `nix-managed-` — coexistem com plugins instalados manualmente e são limpos automaticamente quando removidos da config Nix.

## Managing plugins {#managing-plugins}

```bash
hermes plugins                               # unified interactive UI
hermes plugins list                          # table: enabled / disabled / not enabled
hermes plugins search <term>                 # search the community plugin index
hermes plugins install <name>                # install by index name (resolved to repo @ pinned ref)
hermes plugins install user/repo             # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable    # install AND enable (no prompt)
hermes plugins install user/repo --no-enable # install but leave disabled (no prompt)
hermes plugins update my-plugin              # pull latest (local edits are autostashed and re-applied)
hermes plugins remove my-plugin              # uninstall
hermes plugins enable my-plugin              # add to allow-list
hermes plugins disable my-plugin             # remove from allow-list + add to disabled
hermes plugins capabilities [my-plugin]      # declared vs granted capabilities
```

### Links de install em um clique (Desktop) {#one-click-install-links-desktop}

O Hermes Desktop registra o esquema de URL `hermes://`, então um site, README ou
mensagem de chat pode linkar direto para o install de um plugin:

```
hermes://plugin/install?repo=owner/repo            # main install link
hermes://plugin/install?repo=owner/repo&enable=1   # enable the agent plugin after install
hermes://plugin/install?repo=owner/repo&force=1    # replace an existing install
```

Clicar abre o Hermes e mostra um **diálogo de confirmação** — o id do repo,
uma nota "Before you install", e links de browse + clone do GitHub — depois
faz shallow-clone do repo para detectar o que ele traz (um **agent plugin** —
backend Python, um **desktop plugin** — UI do app, ou ambos). Você escolhe os
componentes com checkboxes e confirma. Nada é instalado até você confirmar;
deep links nunca auto-instalam, e installs de agent-plugin passam pela mesma
[varredura de segurança na hora do install](#install-time-security-scanning) que
`hermes plugins install`.

Repos híbridos (metades agent + desktop num só repo) usam um link e um
diálogo. O mesmo modal é alcançável sem link via **Settings → Plugins →
Install from Git**. URLs legadas `hermes://plugin-agent/…` e
`hermes://plugin-desktop/…` entram no mesmo diálogo. Em builds de dev
(`npm run dev`) o esquema é `hermes-dev://`.

Sites não precisam de SDK — um anchor normal funciona:

```html
<a href="hermes://plugin/install?repo=owner/repo&enable=1">Install in Hermes</a>
```

Servidores MCP têm a forma de link equivalente — veja
[Add to Hermes link](/reference/mcp-config-reference#add-to-hermes-link).

### Plugin capabilities and consent {#plugin-capabilities-and-consent}

Plugins podem declarar as superfícies host privilegiadas que querem no
`plugin.yaml`:

```yaml
name: my-plugin
capabilities:
  - tools.override        # replace built-in tools
  - llm.model_override    # pick the model for host-owned LLM calls
```

Quando um plugin declara capabilities, `hermes plugins install` (e
`hermes plugins enable`) mostra a lista com descrições de risco de uma linha e
pergunta uma vez. Consentir registra o grant em
`plugins.entries.<id>.granted_capabilities` junto com um hash de consentimento e
timestamp. Recusar deixa o plugin enabled com aquelas capabilities off —
um plugin bem-comportado probe com `ctx.has_capability()` e degrada
gracefully.

**Re-consentimento em update:** se um plugin update declara capabilities que você não
concedeu, `hermes plugins update` mostra as adições e pergunta de novo. Capabilities novas
ficam off até você consentir — um plugin update nunca pode
alargar silenciosamente seu acesso.

**Sessões não-interativas falham closed:** instalar ou atualizar sem um
TTY completa o install, mas capabilities declaradas *não* são concedidas. Rode
`hermes plugins enable <id>` interativamente para concedê-las depois.

Inspecione o estado a qualquer momento:

```bash
hermes plugins capabilities             # all plugins with declared/granted capabilities
hermes plugins capabilities my-plugin   # one plugin, declared vs granted
```

Ids de capability mapeiam 1:1 para os gates de config por feature mais antigos, que
continuam funcionando mas estão **deprecated** em favor do fluxo de consentimento:

| Capability | Legacy key (`plugins.entries.<id>.…`) |
|---|---|
| `tools.override` | `allow_tool_override` |
| `llm.provider_override` | `llm.allow_provider_override` |
| `llm.model_override` | `llm.allow_model_override` |
| `llm.agent_id_override` | `llm.allow_agent_id_override` |
| `llm.profile_override` | `llm.allow_profile_override` |
| `llm.task_override` | `llm.allow_task_override` |
| `gateway.platform_actions` | `allow_platform_actions` |

Um gate está aberto quando *ou* a capability é concedida *ou* a legacy key está
set — configs existentes continuam funcionando inalteradas.

:::warning Não é um sandbox
Capabilities são uma **camada de consentimento e auditoria**, não isolamento. Plugins rodam como
Python in-process regular: um plugin malicioso pode ignorar todo gate aqui.
Conceder uma capability é uma declaração de confiança no autor do plugin — não é
um audit de código, e o Hermes não revisou o código do plugin. Só instale
plugins de fontes em que você confia.
:::

### Platform actions {#platform-actions}

`ctx.platform_actions` dá a um plugin um conjunto mínimo de verbos gated por capability para
agir em plataformas de chat conectadas pelo registry live de adapters do gateway —
a alternativa sancionada a monkeypatching de adapter. **Está off por
padrão**: toda call re-checa a capability `gateway.platform_actions`
(legacy key `plugins.entries.<id>.allow_platform_actions`), e uma call sem grant
retorna um erro estruturado em vez de agir.

Verbos v1 (ambos `async`, ambos retornam um dict plain, e nenhum nunca raise no
hook dispatch):

```python
result = await ctx.platform_actions.add_reaction(
    platform="telegram", chat_id="-100123", message_id="456", emoji="👍",
)
result = await ctx.platform_actions.set_thread_title(
    platform="discord", chat_id="123", thread_id="456", title="New title",
)
if not result["ok"]:
    print(result["error"], result.get("detail"))
```

Sucesso é `{"ok": True, "action": <verb>}`. Falhas são
`{"ok": False, "error": <code>, "detail": <str>}` com error codes estáveis:
`capability_not_granted`, `invalid_argument`, `gateway_unavailable`,
`unknown_platform`, `adapter_not_registered`, `adapter_disconnected`,
`unsupported_platform_action`, `action_failed`. Actions validam que o
adapter alvo existe e está connected antes de agir; um adapter disconnected ou
faltando degrada para um erro estruturado, nunca uma exception.

Plataformas suportadas no v1: Telegram e Discord. O `add_reaction` do Telegram
*seta* a reação do bot (a Bot API substitui uma reação anterior do bot em vez
de empilhar). Toda action — permitida ou negada — é escrita no log com
o plugin id, verbo, plataforma e outcome.

:::warning Nota de segurança
Platform actions são um **poder de messaging-as-the-bot**: um plugin granted pode
reagir e renomear threads em qualquer chat que o bot do gateway alcançar, não só o
chat que disparou o hook. Conceda `gateway.platform_actions` só a plugins
em que você confia, e prefira plugins que documentam exatamente quais actions tomam.
Acesso raw a payload/handle de SDK de plataforma é deliberadamente **não** parte desta
superfície — pela correção de design round-2 #64176 exige sua própria
capability (`gateway.raw_events`) com label "no stability guarantee" e um
design separado, e não shipped.
:::

### Discovering community plugins {#discovering-community-plugins}

`hermes plugins search <term>` busca o **community plugin index** — um
catálogo JSON estático, machine-readable de plugins da comunidade. Matching é fuzzy
em name, description e tags:

```bash
hermes plugins search telegram               # fuzzy search
hermes plugins search                        # browse the whole index
hermes plugins search --capability platform  # filter by declared capability
hermes plugins search media --json           # machine-readable output
hermes plugins search --refresh              # bypass the 24h local cache
```

Quando encontrar um plugin, instale pelo nome bare — o nome é resolvido
pelo index para seu `owner/repo` mais o commit pinned no index:

```bash
hermes plugins install hermes-media-studio
```

Se um nome casa com mais de uma entry, os candidatos são listados e nada
é instalado. Identificadores explícitos `owner/repo` ou Git-URL nunca tocam o
index e continuam funcionando exatamente como antes. Um `--ref <sha>` explícito sempre
sobrescreve o pin do index.

**Como o index é fetched.** O index vive numa URL canônica
(`https://raw.githubusercontent.com/NousResearch/hermes-plugin-index/main/index.json`,
overridable via `hermes config set plugins.index_url <url>`). Fetches são
cached em `~/.hermes/cache/plugin_index.json` por 24 horas; quando o
remote está inacessível o cache stale é usado, e quando não há cache
nenhum uma seed copy bundled vem com o Hermes — então search funciona fully offline.

**Formato de entry do index.** Cada entry é um objeto JSON:

```json
{
  "name": "hermes-media-studio",
  "description": "Generative media workspace plugin.",
  "author": "NousResearch",
  "tags": ["media", "image-gen"],
  "repo": "NousResearch/hermes-media-studio",
  "ref": "<40-char commit SHA>",
  "subdir": null,
  "homepage": "https://github.com/NousResearch/hermes-media-studio",
  "capabilities": ["tools", "dashboard"],
  "api_version": 1,
  "added_at": "2026-08-12"
}
```

`repo` é o identificador GitHub `owner/name`, `ref` pin um commit SHA
imutável, e `subdir` opcional suporta monorepos. O arquivo seed bundled
(`hermes_cli/data/plugin_index.json` no repo) é a referência de formato.

**Submetendo um plugin.** O index é mantido como um arquivo JSON plain —
envie um pull request ao repositório
[hermes-plugin-index](https://github.com/NousResearch/hermes-plugin-index)
adicionando sua entry (name, description, author, tags, `owner/repo`,
e um commit SHA pinned). Review cobre só os *metadados* da entry.

:::warning Indexed ≠ audited
Inclusão no community index significa que os metadados da entry foram revisados —
**não é um audit de código**. Instalar ainda passa pelo fluxo normal
de consentimento/review (plugins instalam disabled por padrão, habilitar é um
passo explícito, e direitos de tool-override exigem grant separado). Revise o
source de um plugin antes de habilitá-lo.
:::

### Plugin packs {#plugin-packs}

Um **plugin pack** é um arquivo YAML declarativo e compartilhável (`hermes-pack.yaml`)
que pin um conjunto de plugins — como compartilhar um modpack. Instalar um pack faz fan-out
para installs pinned ordinários; nada novo existe em runtime.

```yaml
name: voice-assistant-pack
description: STT + streaming TTS + approval relay
author: hyper
version: 1.0.0
plugins:
  - name: hermes-media-studio            # bare community-index name…
    ref: e8d59971d2b7901405b39dac7b03bdd616272d0d
  - repo: owner/approval-relay           # …or explicit owner/repo (or git URL)
    ref: 8f3c2d1a9b4e5f6071829304a5b6c7d8e9f00112
    subdir: plugins/relay                # optional monorepo path
config:                                  # optional, non-secret seeds only
  hermes-media-studio:
    default_model: flux-3
skills: []                               # declared list only (not auto-installed yet)
```

```bash
hermes plugins pack show ./hermes-pack.yaml     # dry-run review
hermes plugins pack install ./hermes-pack.yaml  # review → confirm → install
hermes plugins pack export > hermes-pack.yaml   # snapshot the current install
hermes plugins pack export --enabled-only       # only plugins.enabled
```

**Postura de supply-chain.** O `ref` de cada entry deve ser um SHA de commit
exato de 40 caracteres — tags e nomes de branch são rejeitados com um erro nomeando a
entry, a mesma regra do community index. Pack installs usam o caminho exato
de install pinned de `hermes plugins install --ref <sha>` e registram
a mesma provenance em `plugins/.install-metadata.json`, então dois installs do
mesmo pack resolvem identicamente. Packs constroem sobre os
[campos manifest v2](/developer-guide/plugins) (`manifest_version`,
`api_version`, `requires_plugins`) — o próprio manifest de cada plugin ainda
valida pelo path de install normal.

**Consentimento nunca é concedido em bulk.** `pack install` mostra uma tela obrigatória de
review (todo plugin, source, ref pinned, e as capabilities que declara),
depois pede **uma** confirmação para o conteúdo do pack. Depois disso, as
capabilities declaradas de cada plugin passam pelo prompt padrão de
capability-consent por plugin — idêntico a um `hermes plugins install` único.
Não há `--yes`, e sessões não-interativas não podem instalar packs.

**Secrets nunca viajam em packs.** Seeds de `config:` são limitados a
keys `plugins.entries.<id>` não-secretas — nomes de key shaped como secret
(`*token*`, `*key*`, `*password*`, …), grants de capability, e os gates de
trust deprecated `allow_*` são rejeitados no install e stripped no export.
Plugins que precisam de secrets os declaram no próprio `requires_env`, que
prompta durante o install como de costume. Valores de usuário existentes em
`plugins.entries.<id>` sempre vencem sobre seeds do pack.

**Falha parcial.** Cada plugin instala independentemente; falhas são
reportadas por plugin, o resto continua, e o comando sai non-zero se
algum plugin falhou.

**Caveats de export.** `pack export` só inclui plugins com provenance Git
conhecida (instalados via `hermes plugins install`). Plugins só locais são
listados como comments de warning no YAML emitido, não como entries instaláveis.

A lista `skills:` é parseada e exibida no install mas ainda não
auto-instalada — instale aquelas manualmente por agora (`hermes skills`). Wiring
skill-hub ids no pack install é uma seam de follow-up documentada.

### Install-time security scanning {#install-time-security-scanning}

Todo `hermes plugins install` e `hermes plugins update` roda um scan
estático de segurança sobre a árvore do plugin antes de ativá-lo (inspirado no
scanning de skill & plugin do Claude Cowork). O scanner reusa o
mesmo engine de threat-pattern do [Skills Hub guard](/user-guide/features/skills)
— exfiltração de credential stores, reverse shells, comandos destrutivos,
mecanismos de persistence, execução ofuscada, e prompt injection em
arquivos de documentação — com exemptions conscientes de plugin: um plugin provider
lendo sua **própria** API key do environment (o padrão documentado
`requires_env`) não é flagged.

Três vereditos, casando o pass/warn/fail do Cowork:

| Veredito | Comportamento |
|---|---|
| **safe** | Instala normalmente, sem output extra |
| **caution** | Findings são mostrados; você confirma `Install anyway? [y/N]` (ou passa `--force`) |
| **dangerous** | Bloqueado. `--force` **não** override |

Em `hermes plugins update`, um veredito dangerous na árvore atualizada
desabilita o plugin até você revisar os findings e re-habilitá-lo.

Scanning está on por padrão; desabilite em `config.yaml`:

```yaml
plugins:
  scan_on_install: false
```

### Interactive UI {#interactive-ui}

Rodar `hermes plugins` sem argumentos abre uma tela interativa composta:

```
Plugins
  ↑↓ navigate  SPACE toggle  ENTER configure/confirm  ESC done

  General Plugins
 → [✓] my-tool-plugin — Custom search tool
   [ ] webhook-notifier — Event hooks
   [ ] disk-cleanup — Auto-cleanup of ephemeral files [bundled]

  Provider Plugins
     Memory Provider          ▸ honcho
     Context Engine           ▸ compressor
```

- **Seção General Plugins** — checkboxes, alterne com SPACE. Marcado = em `plugins.enabled`, desmarcado = em `plugins.disabled` (off explícito).
- **Seção Provider Plugins** — mostra seleção atual. Pressione ENTER para entrar em um seletor radio onde você escolhe um provider ativo.
- Plugins bundled aparecem na mesma lista com tag `[bundled]`.

Seleções de provider plugin são salvas em `config.yaml`:

```yaml
memory:
  provider: "honcho"      # empty string = built-in only

context:
  engine: "compressor"    # default built-in compressor
```

### Enabled vs. disabled vs. neither {#enabled-vs-disabled-vs-neither}

Plugins ocupam um de três estados:

| State | Meaning | In `plugins.enabled`? | In `plugins.disabled`? |
|---|---|---|---|
| `enabled` | Loaded on next session | Yes | No |
| `disabled` | Explicitly off — won't load even if also in `enabled` | (irrelevant) | Yes |
| `not enabled` | Discovered but never opted in | No | No |

O padrão para plugin recém-instalado ou bundled é `not enabled`. `hermes plugins list` mostra os três estados distintos para você distinguir o que foi explicitamente desligado do que só aguarda ser habilitado.

Em sessão rodando, `/plugins` mostra quais plugins estão carregados no momento.

## Injecting Messages {#injecting-messages}

Plugins podem injetar mensagens numa conversa CLI ou numa sessão gateway conhecida usando `ctx.inject_message()`:

```python
# Active CLI conversation
ctx.inject_message("New data arrived from the webhook", role="user")

# Existing gateway conversation
ctx.inject_message(
    "New data arrived from the webhook",
    role="user",
    session_key="agent:main:telegram:dm:123456789",
)
```

**Signature:** `ctx.inject_message(content: str, role: str = "user", *, session_key: str | None = None) -> bool`

Em modo CLI:

- Se o agente estiver **idle** (aguardando input do usuário), a mensagem é enfileirada como próximo input e inicia um novo turno.
- Se o agente estiver **mid-turn** (rodando ativamente), a mensagem interrompe a operação atual — igual a um usuário digitando nova mensagem e pressionando Enter.
- Para roles diferentes de `"user"`, o conteúdo é prefixado com `[role]` (por exemplo, `[system] ...`).
- Retorna `True` se a mensagem foi enfileirada com sucesso.

Em modo gateway:

- `session_key` é obrigatório e deve identificar uma sessão gateway existente. É a routing key estável, não o session ID CLI.
- O Hermes reusa a plataforma, chat, thread, profile e histórico de conversa armazenados daquela sessão. Plugins não podem fornecer uma rota de chat nova por esta API.
- O Hermes recheca a rota armazenada contra as regras atuais de autorização do gateway antes do dispatch.
- Rotas que dependiam só de uma decisão de autorização adapter-time ou upstream são rejeitadas a menos que o Hermes possa revalidá-las das allowlists core atuais, pairing, ou config explícita allow-all.
- Texto injetado é sempre input conversacional. Não pode invocar slash commands, aprovar tools, ou resolver prompts pendentes de confirmação e clarificação.
- A rota e a conversa ficam pinned enquanto o dispatch está pendente. O Hermes dropa o request se topic recovery muda a rota ou a sessão rotaciona antes do handling começar.
- O request entra no path normal de mensagem do adapter de plataforma. Sessões ativas usam a queue busy-session existente em vez de começar um turno concorrente.
- Retorna `True` quando o gateway live aceita o request para dispatch assíncrono. Isso não confirma que o turno do agente ou a entrega na plataforma completou.
- Retorna `False` quando `session_key` é omitido, a permissão não é concedida, ou nenhum gateway live pode aceitar o request. Session keys desconhecidas ou unroutable descobertas após aceitação assíncrona são escritas no log do gateway.

Isso permite plugins como viewers de controle remoto, bridges de mensagens ou receivers de webhook alimentarem mensagens na conversa a partir de fontes externas.

Injeção gateway pode enviar uma resposta de agente a uma plataforma de mensagens externa. Está desabilitada por padrão para todo plugin. Conceda por plugin em `config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:
      allow_gateway_injection: true
```

:::warning
Só conceda injeção gateway a plugins em que você confia. O Hermes checa esta permissão de host API e a restringe a rotas de sessão existentes, mas plugins Python rodam in-process e esta setting não é um sandbox.
:::

:::note
Esta API de plugin não expõe um endpoint HTTP público ou comando CLI para processos externos. O plugin já precisa conhecer o `session_key` gateway alvo, por exemplo da própria config confiável ou estado de sessão retido antes.
:::

## Calling MCP servers from plugins {#calling-mcp-servers-from-plugins}

`ctx.call_mcp()` deixa um plugin chamar uma tool em um dos servidores MCP configurados do usuário — sincronamente, de qualquer hook ou tool handler — roteando pelo client MCP nativo existente do Hermes (mesmas conexões, gates de trust-tier, circuit breaker e lógica de reconnect que tools MCP invocadas pelo modelo; nunca um client paralelo).

```python
result = ctx.call_mcp(
    "knowledge_rag",            # server name from mcp.servers
    "query_knowledge",          # tool on that server
    {"query": "deploy runbook"},
    timeout=30,                 # seconds; clamped to 1–600
)
if result["ok"]:
    print(result["result"])
else:
    print("MCP error:", result["error"])
```

**Signature:** `ctx.call_mcp(server: str, tool: str, arguments: dict | None = None, timeout: float = 30) -> dict`

Retorna um envelope estável: `{"ok": True, "result": ...}` (mais `structuredContent` quando o servidor o fornece) ou `{"ok": False, "error": "..."}`. Results acima de ~64 KB são truncados e flagged com `"truncated": True`.

### Security: default-off, per-server allowlist {#security-default-off-per-server-allowlist}

Um plugin tem **nenhum acesso MCP por padrão**. O operador deve conceder cada servidor explicitamente em `config.yaml`:

```yaml
plugins:
  entries:
    my-plugin:
      mcp_allowlist: ["knowledge_rag", "github"]
```

- Chamar um servidor que não está na lista raise `PermissionError` nomeando a config key exata a setar.
- O grant é por-servidor e por-plugin — nunca autoridade ambiente sobre todo servidor configurado, e wildcards `"*"` não são honrados.
- Toda call tem um timeout enforced (padrão 30 s) para um servidor MCP hung não stall o pipeline de hook ou tool que o invocou.
- Servidores MCP retornam conteúdo untrusted. Trate `result` como data, não instruções — não alimente em decisões privilegiadas (aprovações, execução de comando) sem validação.

:::warning
Conceder `mcp_allowlist` dá ao plugin o mesmo acesso àquele servidor MCP que o modelo tem — incluindo quaisquer tools write-capable que o servidor expõe (sujeitas aos gates de `trust` do servidor). Conceda só servidores que o plugin realmente precisa.
:::

Veja o **[guia completo](/developer-guide/plugins)** para contratos de handler, formato de schema, comportamento de hooks, tratamento de erros e erros comuns.
