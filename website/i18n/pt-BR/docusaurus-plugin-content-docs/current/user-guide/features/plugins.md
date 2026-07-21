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
        description="Return a friendly greeting for the given name.",
    )

    # --- Hook: log every tool call ---
    def on_tool_call(tool_name, params, result):
        print(f"[hello-world] tool called: {tool_name}")

    ctx.register_hook("post_tool_call", on_tool_call)
```

Coloque ambos os arquivos em `~/.hermes/plugins/hello-world/`, reinicie o Hermes, e o modelo pode chamar `hello_world` imediatamente. O hook imprime uma linha de log após cada invocação de tool.

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
| Inject messages | `ctx.inject_message(content, role="user")` — see [Injecting Messages](#injecting-messages) |
| Ship data files | `Path(__file__).parent / "data" / "file.yaml"` |
| Bundle skills | `ctx.register_skill(name, path)` — namespaced as `plugin:skill`, loaded via `skill_view("plugin:skill")` |
| Gate on env vars | `requires_env: [API_KEY]` in plugin.yaml — prompted during `hermes plugins install` |
| Distribute via pip | `[project.entry-points."hermes_agent.plugins"]` |
| Register a gateway platform (Discord, Telegram, IRC, …) | `ctx.register_platform(name, label, adapter_factory, check_fn, ...)` — see [Adding Platform Adapters](/developer-guide/adding-platform-adapters) |
| Register an image-generation backend | `ctx.register_image_gen_provider(provider)` — see [Image Generation Provider Plugins](/developer-guide/image-gen-provider-plugin) |
| Register a video-generation backend | `ctx.register_video_gen_provider(provider)` — see [Video Generation Provider Plugins](/developer-guide/video-gen-provider-plugin) |
| Register a context-compression engine | `ctx.register_context_engine(engine)` — see [Context Engine Plugins](/developer-guide/context-engine-plugin) |
| Register a memory backend | Subclass `MemoryProvider` in `plugins/memory/<name>/__init__.py` — see [Memory Provider Plugins](/developer-guide/memory-provider-plugin) (uses a separate discovery system) |
| Run a host-owned LLM call | `ctx.llm.complete(...)` / `ctx.llm.complete_structured(...)` — borrow the user's active model + auth for a one-shot completion with optional JSON schema validation. See [Plugin LLM Access](/developer-guide/plugin-llm-access) |
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

### Migration for existing users {#migration-for-existing-users}

Quando você atualiza para uma versão do Hermes com plugins opt-in (config schema v21+), plugins do usuário já instalados em `~/.hermes/plugins/` que não estavam em `plugins.disabled` são **automaticamente grandfathered** em `plugins.enabled`. Sua config existente continua funcionando. Plugins standalone bundled NÃO são grandfathered — mesmo usuários existentes precisam optar explicitamente. (Plugins bundled platform/backend nunca precisaram de grandfathering porque nunca foram gated.)

## Available hooks {#available-hooks}

Plugins podem registrar callbacks para estes eventos de lifecycle. Veja a **[página Event Hooks](/user-guide/features/hooks#plugin-hooks)** para detalhes completos, assinaturas de callback e exemplos.

| Hook | Fires when |
|------|-----------|
| [`pre_tool_call`](/user-guide/features/hooks#pre_tool_call) | Before any tool executes |
| [`post_tool_call`](/user-guide/features/hooks#post_tool_call) | After any tool returns |
| [`pre_llm_call`](/user-guide/features/hooks#pre_llm_call) | Once per turn, before the LLM loop — can return `{"context": "..."}` to [inject context into the user message](/user-guide/features/hooks#pre_llm_call) |
| [`post_llm_call`](/user-guide/features/hooks#post_llm_call) | Once per turn, after the LLM loop (successful turns only) |
| [`on_session_start`](/user-guide/features/hooks#on_session_start) | New session created (first turn only) |
| [`on_session_end`](/user-guide/features/hooks#on_session_end) | End of every `run_conversation` call + CLI exit handler |
| [`on_session_finalize`](/user-guide/features/hooks#on_session_finalize) | CLI/gateway tears down an active session (`/new`, GC, CLI quit) |
| [`on_session_reset`](/user-guide/features/hooks#on_session_reset) | Gateway swaps in a new session key (`/new`, `/reset`, `/clear`, idle rotation) |
| [`subagent_stop`](/user-guide/features/hooks#subagent_stop) | Once per child after `delegate_task` finishes |
| [`pre_gateway_dispatch`](/user-guide/features/hooks#pre_gateway_dispatch) | Gateway received a user message, before auth + dispatch. Return `{"action": "skip" \| "rewrite" \| "allow", ...}` to influence flow. |

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
hermes plugins install user/repo             # install from Git, then prompt Enable? [y/N]
hermes plugins install user/repo --enable    # install AND enable (no prompt)
hermes plugins install user/repo --no-enable # install but leave disabled (no prompt)
hermes plugins update my-plugin              # pull latest
hermes plugins remove my-plugin              # uninstall
hermes plugins enable my-plugin              # add to allow-list
hermes plugins disable my-plugin             # remove from allow-list + add to disabled
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

Plugins podem injetar mensagens na conversa ativa usando `ctx.inject_message()`:

```python
ctx.inject_message("New data arrived from the webhook", role="user")
```

**Signature:** `ctx.inject_message(content: str, role: str = "user") -> bool`

Como funciona:

- Se o agente estiver **idle** (aguardando input do usuário), a mensagem é enfileirada como próximo input e inicia um novo turno.
- Se o agente estiver **mid-turn** (rodando ativamente), a mensagem interrompe a operação atual — igual a um usuário digitando nova mensagem e pressionando Enter.
- Para roles diferentes de `"user"`, o conteúdo é prefixado com `[role]` (por exemplo, `[system] ...`).
- Retorna `True` se a mensagem foi enfileirada com sucesso, `False` se não há referência CLI disponível (por exemplo, em modo gateway).

Isso permite plugins como viewers de controle remoto, bridges de mensagens ou receivers de webhook alimentarem mensagens na conversa a partir de fontes externas.

:::note
`inject_message` só está disponível em modo CLI. Em modo gateway, não há referência CLI e o método retorna `False`.
:::

Veja o **[guia completo](/developer-guide/plugins)** para contratos de handler, formato de schema, comportamento de hooks, tratamento de erros e erros comuns.
