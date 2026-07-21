---
sidebar_position: 2
title: "Configuração"
description: "Configure o Hermes Agent — config.yaml, providers, models, API keys e mais"
---

# Configuração

Todas as configurações ficam armazenadas no diretório `~/.hermes/` para fácil acesso.

:::tip Caminho mais fácil para um `config.yaml` funcional
Execute `hermes setup --portal` — um OAuth configura um model provider e todas as quatro ferramentas do Tool Gateway sem editar YAML manualmente. Assinantes do Portal também ganham 10% de desconto em providers cobrados por token. Veja [Nous Portal](/integrations/nous-portal).
:::

## Estrutura de diretórios

```text
~/.hermes/
├── config.yaml     # Settings (model, terminal, TTS, compression, etc.)
├── .env            # API keys and secrets
├── auth.json       # OAuth provider credentials (Nous Portal, etc.)
├── SOUL.md         # Primary agent identity (slot #1 in system prompt)
├── memories/       # Persistent memory (MEMORY.md, USER.md)
├── skills/         # Agent-created skills (managed via skill_manage tool)
├── cron/           # Scheduled jobs
├── sessions/       # Gateway sessions
└── logs/           # Logs (errors.log, gateway.log — secrets auto-redacted)
```

## Gerenciando a configuração

```bash
hermes config              # View current configuration
hermes config edit         # Open config.yaml in your editor
hermes config get KEY      # Print a resolved value
hermes config set KEY VAL  # Set a specific value
hermes config unset KEY    # Remove a user-set value
hermes config check        # Check for missing options (after updates)
hermes config migrate      # Interactively add missing options

# Examples:
hermes config get model
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config unset terminal.backend
hermes config set OPENROUTER_API_KEY sk-or-...  # Saves to .env
```

:::tip
O comando `hermes config set` encaminha os valores automaticamente para o arquivo certo — API keys vão para `.env`, todo o resto para `config.yaml`.
:::

## Precedência de configuração

As configurações são resolvidas nesta ordem (maior prioridade primeiro):

1. **Argumentos da CLI** — ex.: `hermes chat --model anthropic/claude-sonnet-4` (override por invocação)
2. **`~/.hermes/config.yaml`** — o arquivo principal de configuração para todas as configurações que não são segredos
3. **`~/.hermes/.env`** — fallback para env vars; **obrigatório** para segredos (API keys, tokens, senhas)
4. **Defaults embutidos** — defaults seguros hardcoded quando nada mais está definido

:::info Regra prática
Segredos (API keys, bot tokens, senhas) vão em `.env`. Todo o resto (model, terminal backend, compression settings, memory limits, toolsets) vai em `config.yaml`. Quando ambos estão definidos, `config.yaml` prevalece para configurações que não são segredos.
:::

:::tip Deployments organizacionais
Um administrador pode fixar valores específicos de config e segredos que um usuário padrão
não pode sobrescrever, via um diretório gerenciado em nível de sistema. Veja
[Managed Scope](/user-guide/managed-scope).
:::

## Substituição de variáveis de ambiente

Você pode referenciar variáveis de ambiente em `config.yaml` usando a sintaxe `${VAR_NAME}`:

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

Múltiplas referências em um único valor funcionam: `url: "${HOST}:${PORT}"`. Se uma variável referenciada não estiver definida, o placeholder é mantido literalmente (`${UNDEFINED_VAR}` permanece como está). Apenas a sintaxe `${VAR}` é suportada — `$VAR` sem chaves não é expandido.

Para configuração de AI providers (OpenRouter, Anthropic, Copilot, endpoints customizados, LLMs self-hosted, fallback models, etc.), veja [AI Providers](/integrations/providers).

### Timeouts de provider

Você pode definir `providers.<id>.request_timeout_seconds` para um timeout de requisição em todo o provider, além de `providers.<id>.models.<model>.timeout_seconds` para um override específico por model. Aplica-se ao client principal do turno em todo transport (OpenAI-wire, native Anthropic, Anthropic-compatible), à cadeia de fallback, rebuilds após rotação de credenciais e (para OpenAI-wire) ao kwarg de timeout por requisição — assim o valor configurado prevalece sobre a env var legada `HERMES_API_TIMEOUT`.

Você também pode definir `providers.<id>.stale_timeout_seconds` para o detector de chamadas stale sem streaming, além de `providers.<id>.models.<model>.stale_timeout_seconds` para um override específico por model. Isso prevalece sobre a env var legada `HERMES_API_CALL_STALE_TIMEOUT`.

Deixar esses valores sem definir mantém os defaults legados (`HERMES_API_TIMEOUT=1800`s, `HERMES_API_CALL_STALE_TIMEOUT=90`s, native Anthropic 900s). O detector stale sem streaming é desativado automaticamente para endpoints locais quando deixado implícito e pode escalar para cima em contextos muito grandes. Ainda não está conectado para AWS Bedrock (ambos os caminhos `bedrock_converse` e AnthropicBedrock SDK usam boto3 com sua própria configuração de timeout). Veja o exemplo comentado em [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example).

## Comportamento de atualização

As configurações de `hermes update` ficam em `updates` no `config.yaml`:

```yaml
updates:
  pre_update_backup: quick       # quick (state snapshot, default) | full (snapshot + HERMES_HOME zip) | off
  backup_keep: 5                 # Keep this many full pre-update backup zips
  non_interactive_local_changes: stash  # stash | discard
```

`pre_update_backup` é o único controle de segurança pré-atualização: `quick` (default) faz snapshot de arquivos críticos de estado (pairing data, cron jobs, config, auth; arquivos acima de 1 GiB são ignorados) em `state-snapshots/`; `full` adicionalmente compacta todo o `HERMES_HOME` em `backups/` e pode levar minutos em homes grandes; `off` desativa ambos. Booleanos legados são respeitados (`true` → `full`, `false` → `off`).

Para instalações git, o Hermes faz auto-stash de arquivos rastreados sujos e arquivos não rastreados antes de fazer checkout da branch de atualização ou pull. Atualizações interativas no terminal pedem confirmação antes de restaurar esse stash. Atualizações não interativas (desktop/chat app, gateway ou `--yes`) usam `updates.non_interactive_local_changes`: `stash` restaura edições locais de source após um pull bem-sucedido, enquanto `discard` descarta o stash criado pela atualização após um pull bem-sucedido. Use `discard` apenas em instalações gerenciadas onde edições locais de source nunca devem persistir.

Antes desse passo de stash, o Hermes também restaura diffs rastreados de `package-lock.json` deixados por churn de npm install/build. Faça commit ou stash manual de edições intencionais de lockfile antes de atualizar.

## Configuração do terminal backend

O Hermes suporta seis terminal backends. Cada um determina onde os comandos shell do agente realmente executam — sua máquina local, um container Docker, um servidor remoto via SSH, um sandbox na nuvem Modal (direto ou via gateway gerenciado pela Nous), um workspace Daytona, ou um container Singularity/Apptainer.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | singularity
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  timeout: 180      # Per-command timeout in seconds
  home_mode: auto   # auto | real | profile — subprocess HOME policy
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Container image for Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Container image for Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Container image for Daytona backend
```

Para sandboxes na nuvem como Modal e Daytona, `container_persistent: true` significa que o Hermes tentará preservar o estado do filesystem entre recriações de sandbox. Isso não garante que o mesmo sandbox ativo, PID space ou processos em background ainda estarão rodando depois.

### Visão geral dos backends

| Backend | Onde os comandos rodam | Isolamento | Melhor para |
|---------|-------------------|-----------|----------|
| **local** | Sua máquina diretamente | Nenhum | Desenvolvimento, uso pessoal |
| **docker** | Container Docker persistente único (compartilhado entre sessão, `/new`, subagents) | Completo (namespaces, cap-drop) | Sandboxing seguro, CI/CD |
| **ssh** | Servidor remoto via SSH | Limite de rede | Dev remoto, hardware potente |
| **modal** | Sandbox na nuvem Modal | Completo (cloud VM) | Compute efêmero na nuvem, evals |
| **daytona** | Workspace Daytona | Completo (cloud container) | Ambientes de dev gerenciados na nuvem |
| **singularity** | Container Singularity/Apptainer | Namespaces (--containall) | Clusters HPC, máquinas compartilhadas |

### Backend local

O padrão. Comandos rodam diretamente na sua máquina sem isolamento. Nenhuma configuração especial necessária.

```yaml
terminal:
  backend: local
```

Por padrão, subprocessos locais de ferramentas mantêm o `HOME` real do usuário do SO. Isso permite
que CLIs externos como `git`, `ssh`, `gh`, `az`, `npm`, Claude Code e Codex
encontrem as credenciais e config que já usam no seu shell normal. O estado do Hermes
continua com escopo de profile via `HERMES_HOME`; `HOME` não é como profiles
selecionam config, memory, sessions ou skills.

O Hermes **não** altera o `HOME` em todo o sistema, seus arquivos de startup do shell ou
o home da conta do sistema operacional. Essa configuração controla apenas o ambiente
passado a subprocessos que o Hermes lança via ferramentas como `terminal`,
processos de terminal em background, `execute_code` e processos helper do ACP.

#### `terminal.home_mode`

| Mode | Instalações no host | Containers | Tradeoff |
|---|---|---|---|
| `auto` | Mantém o `HOME` real do usuário do SO | Usa `{HERMES_HOME}/home` | Default recomendado. CLIs no host continuam funcionando; estado do container persiste. |
| `real` | Força o `HOME` real do usuário do SO | Força o `HOME` real do usuário do SO se visível | Útil se um processo pai iniciou acidentalmente com `HOME` apontando para um profile home. |
| `profile` | Usa `{HERMES_HOME}/home` quando existir | Usa `{HERMES_HOME}/home` quando existir | Isolamento estrito de config de CLI por profile, mas `~/.ssh`, `~/.gitconfig`, `~/.azure`, `~/.config/gh`, auth Claude/Codex, estado npm, etc. normais não ficarão visíveis a menos que você inicialize ou crie links dentro do profile home. |

A desvantagem do default é que profiles no host compartilham as mesmas credenciais/config de CLI normais em nível de usuário em `~`. Se você precisa de um profile com identidade git separada, chaves SSH, login GitHub CLI, config npm ou login de cloud CLI, use `home_mode: profile` e inicialize essas ferramentas dentro desse profile home de forma deliberada.

Se você quer intencionalmente isolamento estrito de config de ferramentas por profile, defina:

```yaml
terminal:
  home_mode: profile
```

Nesse mode, subprocessos de ferramentas usam `{HERMES_HOME}/home` como `HOME`. O Hermes também
define `HERMES_REAL_HOME` para que scripts ainda possam localizar o home real do usuário quando
precisarem. Backends de container continuam usando `{HERMES_HOME}/home` em mode `auto`
porque esse diretório fica no volume persistente de dados do Hermes.

Scripts que precisam distinguir estado de profile do home real do usuário devem
preferir `HERMES_HOME` para dados do Hermes e `HERMES_REAL_HOME` para o home da conta:

```python
from pathlib import Path
import os

hermes_home = Path(os.environ["HERMES_HOME"])
real_home = Path(os.environ.get("HERMES_REAL_HOME", os.environ["HOME"]))
```

:::warning
O agente tem o mesmo acesso ao filesystem que sua conta de usuário. Use `hermes tools` para desabilitar ferramentas que você não quer, ou mude para Docker para sandboxing.
:::

### Backend Docker {#docker-backend}

Executa comandos dentro de um container Docker com hardening de segurança (todas as capabilities removidas, sem privilege escalation, limites de PID).

**Container persistente único, compartilhado entre processos Hermes.** O Hermes inicia UM container de longa duração no primeiro uso e encaminha toda chamada de terminal, file e `execute_code` via `docker exec` para esse mesmo container — entre sessões, `/new`, `/reset` e subagents de `delegate_task`. Mudanças de working directory, pacotes instalados, arquivos em `/workspace` e **processos em background** persistem de uma chamada de ferramenta para a próxima, e de um processo Hermes para o outro. Quando você fecha uma sessão TUI, executa `/quit` ou inicia uma nova invocação `hermes`, o container continua rodando e o próximo processo Hermes o reutiliza via lookup por label. Veja **Container lifecycle** abaixo para as regras exatas de teardown.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace
  docker_run_as_host_user: false   # See "Running container as host user" below
  docker_forward_env:              # Host env vars to forward into container
    - "GITHUB_TOKEN"
  docker_env:                      # Literal env vars to inject (KEY=value)
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:                  # Host directory mounts
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"   # :ro for read-only
  docker_extra_args:               # Extra flags appended verbatim to `docker run`
    - "--gpus=all"
    - "--network=host"
  docker_network: true             # false = air-gap the container (--network=none)

  # Resource limits
  container_cpu: 1                 # CPU cores (0 = unlimited)
  container_memory: 5120           # MB (0 = unlimited)
  container_disk: 51200            # MB (requires overlay2 on XFS+pquota)
  container_persistent: true       # Persist /workspace and /root bind-mount dirs

  # Cross-process container reuse (defaults match the "one long-lived
  # container shared across sessions" contract — see Container lifecycle).
  docker_persist_across_processes: true   # Reuse container across Hermes restarts
  docker_orphan_reaper: true              # Sweep abandoned Exited containers at startup

  # Cross-backend lifecycle settings (apply to docker as well)
  timeout: 180                     # Per-command timeout in seconds
  lifetime_seconds: 300            # Idle-reaper window; also feeds 2× orphan-reaper threshold
```

**`docker_env`** vs **`docker_forward_env`**: o primeiro injeta pares `KEY=value` literais que você especifica na config (os valores ficam no seu `config.yaml` ou são passados como dict JSON via `TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'`). O segundo encaminha valores do seu shell ou `~/.hermes/.env`, então o segredo real nunca aparece no arquivo de config. Use `docker_forward_env` para tokens e `docker_env` para knobs estáticos que o container precisa.

**`terminal.docker_extra_args`** (também sobrescrevível via `TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'`) permite passar flags arbitrárias de `docker run` que o Hermes não expõe como chaves de primeira classe — `--gpus`, `--network`, `--add-host`, overrides alternativos de `--security-opt`, etc. Cada entrada deve ser uma string; a lista é anexada por último à invocação `docker run` montada, para poder sobrescrever os defaults do Hermes se necessário. Use com parcimônia — flags que conflitam com o hardening do sandbox (capability drops, `--user`, o bind mount do workspace) enfraquecem o isolamento silenciosamente.

**`terminal.docker_network`** (default `true`; env: `TERMINAL_DOCKER_NETWORK`) — defina como `false` para rodar o container sandbox com `--network=none`, cortando todo egress de rede dos comandos do agente. Isso se aplica ao container de execução usado por `terminal`, `execute_code` e as ferramentas de file. Como containers persistem entre processos Hermes, mudar isso para `false` enquanto um container antigo com rede existe remove esse container e inicia um novo air-gapped (um aviso é registrado); processos em background rodando dentro dele são perdidos. Prefira essa chave em vez de passar `--network=none` via `docker_extra_args`.

**Requisitos:** Docker Desktop ou Docker Engine instalado e rodando. O Hermes verifica `$PATH` mais locais comuns de instalação no macOS (`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, app bundle do Docker Desktop). Podman é suportado out of the box: defina `HERMES_DOCKER_BINARY=podman` (ou o caminho completo) para forçá-lo quando ambos estão instalados.

#### Container lifecycle

Todo container gerenciado pelo Hermes é marcado com três labels para que processos subsequentes (e o orphan reaper) possam identificá-lo:

- `hermes-agent=1` — marca como gerenciado pelo Hermes
- `hermes-task-id=<sanitized task_id>` — chaveia a sonda de reuso por task
- `hermes-profile=<sanitized profile name>` — delimita reuso e reaping ao profile Hermes ativo

Na inicialização, o Hermes executa `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<profile>` e **anexa ao container existente** quando encontra um. Se o container está `exited` (ex.: após restart do daemon Docker), ele recebe `docker start` e é reutilizado — estado do filesystem e pacotes instalados sobrevivem, mas processos em background dentro do container não.

Quando um processo Hermes encerra — `/quit`, fechar sessão TUI, shutdown do gateway, até SIGKILL — o caminho de cleanup é **no-op para o container no mode default**. O container continua rodando. O próximo processo Hermes anexa a ele em milissegundos via a sonda de label. Esse é o comportamento que o contrato de "container long-lived único compartilhado entre sessões" exige: é a única forma de processos em background (watchers npm, dev servers, pytest longo) sobreviverem entre sessões.

**O container só é destruído (stopped e `docker rm -f`) nestes casos:**

| Trigger | Quando dispara |
|---|---|
| `docker_persist_across_processes: false` | Isolamento explícito por processo. Todo `cleanup()` faz `stop` + `rm -f`. Corresponde ao comportamento pré-issue-#20561. |
| Idle reaper (`lifetime_seconds`, default 300s) | Apenas quando o env é `persist_across_processes=false`. Envs em mode persist são no-op; container sobrevive ao idle sweep. |
| Orphan reaper na próxima inicialização | Varre containers hermes-labeled **Exited** mais antigos que `2 × lifetime_seconds` (default 600s = 10 min), com escopo do profile atual. **Containers Running nunca são tocados** — segurança entre processos irmãos. Defina `docker_orphan_reaper: false` para desabilitar. |
| Ação direta do usuário | `docker rm -f`, `docker system prune`, restart do Docker Desktop. Não definimos `--restart=always`, então reboot do host deixa o container `Exited` (sua camada CoW sobrevive e é reutilizada na próxima inicialização, mas processos bg somem). |

Edge cases que valem saber:

- **OOM kill do PID 1 dentro do container** transiciona o container para `Exited`. O próximo reuso fará `docker start`; estado do filesystem sobrevive, processos bg não.
- **Trocar profiles** isola containers entre si — um container com label `hermes-profile=work` é invisível para um processo Hermes rodando sob `hermes-profile=research`. O orphan reaper também é com escopo de profile, então containers cross-profile não são reaped acidentalmente, mas também não são limpos automaticamente até você iniciar o Hermes novamente sob o profile original.

Subagents paralelos spawnados via `delegate_task(tasks=[...])` compartilham esse container — `cd` concorrente, mutações de env e writes no mesmo path colidem. Se um subagent precisa de sandbox isolado, ele deve registrar um override de image por task via `register_task_env_overrides()`, que ambientes RL e benchmark (TerminalBench2, HermesSweEnv, etc.) fazem automaticamente para suas images Docker por task.

**Hardening de segurança:**
- `--cap-drop ALL` com apenas `DAC_OVERRIDE`, `CHOWN`, `FOWNER` readicionadas
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- tmpfs com limite de tamanho para `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)

**Encaminhamento de credenciais:** Env vars listadas em `docker_forward_env` são resolvidas do ambiente do seu shell primeiro, depois `~/.hermes/.env`. Skills também podem declarar `required_environment_variables`, que são mescladas automaticamente.

#### Overrides de variáveis de ambiente

Toda chave em `terminal:` tem um override de env var no formato `TERMINAL_<KEY_UPPERCASE>`. As mais úteis para o backend Docker:

| Env var | Mapeia para | Notas |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | Base image |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | JSON array: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | JSON dict: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | JSON array de strings `"host:container[:ro]"` |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | JSON array |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_NETWORK` | `docker_network` | `true` / `false` — default `true`; `false` = `--network=none` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — default `true` |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — default `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | CPU cores |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `true` / `false` — controla os dirs de workspace bind-mount, distinto de `docker_persist_across_processes` |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | Janela do idle reaper |
| `TERMINAL_TIMEOUT` | `timeout` | Timeout por comando |
| `HERMES_DOCKER_BINARY` | _none_ | Força um caminho específico de binary docker/podman |

### Backend SSH {#ssh-backend}

Executa comandos em um servidor remoto via SSH. Usa ControlMaster para reuso de conexão (keepalive idle de 5 minutos). Persistent shell está habilitado por padrão — estado (cwd, env vars) sobrevive entre comandos.

```yaml
terminal:
  backend: ssh
  persistent_shell: true           # Keep a long-lived bash session (default: true)
```

**Variáveis de ambiente obrigatórias:**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**Opcionais:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | Porta SSH |
| `TERMINAL_SSH_KEY` | (system default) | Caminho para chave privada SSH |
| `TERMINAL_SSH_PERSISTENT` | `true` | Habilitar persistent shell |

**Como funciona:** Conecta na inicialização com `BatchMode=yes` e `StrictHostKeyChecking=accept-new`. Persistent shell mantém um único processo `bash -l` vivo no host remoto, comunicando via arquivos temporários. Comandos que precisam de `stdin_data` ou `sudo` fazem fallback automático para mode one-shot.

### Backend Modal

Executa comandos em um sandbox na nuvem [Modal](https://modal.com). Cada task recebe uma VM isolada com CPU, memória e disco configuráveis. O filesystem pode ser snapshot/restaurado entre sessões.

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # Snapshot/restore filesystem
```

**Obrigatório:** Variáveis de ambiente `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, ou um arquivo de config `~/.modal.toml`.

**Persistência:** Quando habilitada, o filesystem do sandbox é snapshotado no cleanup e restaurado na próxima sessão. Snapshots são rastreados em `~/.hermes/modal_snapshots.json`. Isso preserva estado do filesystem, não processos vivos, PID space ou jobs em background.

**Arquivos de credenciais:** Montados automaticamente de `~/.hermes/` (tokens OAuth, etc.) e sincronizados antes de cada comando.

### Backend Daytona

Executa comandos em um workspace gerenciado [Daytona](https://daytona.io). Suporta stop/resume para persistência.

```yaml
terminal:
  backend: daytona
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB → converted to GiB
  container_disk: 10240            # MB → converted to GiB (max 10 GiB)
  container_persistent: true       # Stop/resume instead of delete
```

**Obrigatório:** Variável de ambiente `DAYTONA_API_KEY`.

**Persistência:** Quando habilitada, sandboxes são stopped (não deleted) no cleanup e resumed na próxima sessão. Nomes de sandbox seguem o padrão `hermes-{task_id}`.

**Limite de disco:** Daytona impõe máximo de 10 GiB. Requisições acima disso são limitadas com aviso.

### Backend Singularity/Apptainer

Executa comandos em um container [Singularity/Apptainer](https://apptainer.org). Projetado para clusters HPC e máquinas compartilhadas onde Docker não está disponível.

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB
  container_persistent: true       # Writable overlay persists across sessions
```

**Requisitos:** Binary `apptainer` ou `singularity` em `$PATH`.

**Tratamento de image:** URLs Docker (`docker://...`) são convertidas automaticamente para arquivos SIF e cacheadas. Arquivos `.sif` existentes são usados diretamente.

**Diretório scratch:** Resolvido nesta ordem: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent` (convenção HPC) → `~/.hermes/sandboxes/singularity`.

**Isolamento:** Usa `--containall --no-home` para isolamento completo de namespace sem montar o home do host.

### Problemas comuns de terminal backend

Se comandos de terminal falham imediatamente ou a ferramenta terminal é reportada como desabilitada:

- **Local** — Sem requisitos especiais. O default mais seguro ao começar.
- **Docker** — Execute `docker version` para verificar se o Docker está funcionando. Se falhar, corrija o Docker ou `hermes config set terminal.backend local`.
- **SSH** — Tanto `TERMINAL_SSH_HOST` quanto `TERMINAL_SSH_USER` devem estar definidos. O Hermes registra um erro claro se algum estiver faltando.
- **Modal** — Precisa de env var `MODAL_TOKEN_ID` ou `~/.modal.toml`. Execute `hermes doctor` para verificar.
- **Daytona** — Precisa de `DAYTONA_API_KEY`. O SDK Daytona cuida da configuração de server URL.
- **Singularity** — Precisa de `apptainer` ou `singularity` em `$PATH`. Comum em clusters HPC.

Na dúvida, defina `terminal.backend` de volta para `local` e verifique se os comandos rodam lá primeiro.

### Remote-to-Host File Sync no teardown

Para os backends **SSH**, **Modal** e **Daytona** (onde a working tree do agente vive em uma máquina diferente do host rodando o Hermes), o Hermes rastreia arquivos que o agente tocou dentro do sandbox remoto e, no teardown de sessão / cleanup de sandbox, **sincroniza os arquivos modificados de volta para o host** em `~/.hermes/cache/remote-syncs/<session-id>/`.

- Dispara em: fechamento de sessão, `/new`, `/reset`, timeout de mensagem do gateway, conclusão de subagent `delegate_task` quando o filho usou backend remoto.
- Cobre toda a árvore que o agente modificou, não apenas arquivos que abriu explicitamente. Adições, edições e deleções são capturadas.
- O sandbox remoto pode ter sido destruído quando você for procurar; a cópia local em `~/.hermes/cache/remote-syncs/…` é o registro autoritativo do que o agente alterou.
- Outputs binários grandes (checkpoints de model, datasets brutos) são limitados por tamanho — o sync ignora arquivos acima de `file_sync_max_mb` (default `100`). Aumente se esperar artefatos maiores de volta.

```yaml
terminal:
  file_sync_max_mb: 100     # default — sync files up to 100 MB each
  file_sync_enabled: true   # default — set false to skip the sync entirely
```

É assim que você recupera resultados de sandboxes na nuvem efêmeros destruídos após o fim da sessão, sem precisar dizer ao agente para fazer `scp` ou `modal volume put` explicitamente em cada artefato.

### Docker volume mounts

Ao usar o backend Docker, `docker_volumes` permite compartilhar diretórios do host com o container. Cada entrada usa sintaxe padrão Docker `-v`: `host_path:container_path[:options]`.

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/projects:/workspace/projects"   # Read-write (default)
    - "/home/user/datasets:/data:ro"              # Read-only
    - "/home/user/.hermes/cache/documents:/output" # Gateway-visible exports
```

Isso é útil para:
- **Fornecer arquivos** ao agente (datasets, configs, código de referência)
- **Receber arquivos** do agente (código gerado, relatórios, exports)
- **Workspaces compartilhados** onde você e o agente acessam os mesmos arquivos

Se você usa um messaging gateway e quer que o agente envie arquivos gerados via
`MEDIA:/...`, prefira um mount de export dedicado visível no host, como
`/home/user/.hermes/cache/documents:/output`.

- Escreva arquivos dentro do Docker em `/output/...`
- Emita o **caminho do host** em `MEDIA:`, por exemplo:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- **Não** emita `/workspace/...` ou `/output/...` a menos que esse caminho exato também
  exista para o processo gateway no host

:::warning
Chaves YAML duplicadas sobrescrevem silenciosamente as anteriores. Se você já tem um
bloco `docker_volumes:`, mescle novos mounts na mesma lista em vez de adicionar
outra chave `docker_volumes:` mais adiante no arquivo.
:::

Também pode ser definido via variável de ambiente: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (JSON array).

### Encaminhamento de credenciais Docker

Por padrão, sessões de terminal Docker não herdam credenciais arbitrárias do host. Se você precisa de um token específico dentro do container, adicione-o a `terminal.docker_forward_env`.

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

O Hermes resolve cada variável listada do shell atual primeiro, depois faz fallback para `~/.hermes/.env` se foi salva com `hermes config set`.

:::warning
Qualquer coisa listada em `docker_forward_env` fica visível para comandos rodados dentro do container. Encaminhe apenas credenciais com as quais você se sente confortável em expor à sessão de terminal.
:::

### Rodar o container como seu usuário do host

Por padrão, containers Docker rodam como `root` (UID 0). Arquivos criados dentro de `/workspace` ou outros bind-mounts acabam owned by root no host, então após uma sessão você precisa `sudo chown` antes de editá-los no editor do host. A flag `terminal.docker_run_as_host_user` corrige isso:

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # default: false
```

Quando habilitada, o Hermes anexa `--user $(id -u):$(id -g)` ao comando `docker run` para que arquivos escritos em diretórios bind-mounted (`/workspace`, `/root`, qualquer coisa em `docker_volumes`) sejam owned pelo seu usuário do host, não root. O trade-off: o container não pode mais `apt install` ou escrever em paths owned by root como `/root/.npm` — use uma base image cujo `HOME` é owned by um usuário non-root (ou adicione sua tooling necessária no build da image) se precisar de ambos.

Deixe como `false` (o default) para comportamento retrocompatível. Ative quando seu workflow é principalmente "editar arquivos montados do host" e você está cansado de `sudo chown -R`.

### Opcional: montar o diretório de launch em `/workspace`

Sandboxes Docker permanecem isolados por padrão. O Hermes **não** passa seu working directory atual do host para o container a menos que você opte explicitamente.

Habilite em `config.yaml`:

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

Quando habilitado:
- se você lançar o Hermes de `~/projects/my-app`, esse diretório do host é bind-mounted em `/workspace`
- o backend Docker inicia em `/workspace`
- ferramentas de file e comandos de terminal veem o mesmo projeto montado

Quando desabilitado, `/workspace` permanece owned pelo sandbox a menos que você monte algo explicitamente via `docker_volumes`.

Tradeoff de segurança:
- `false` preserva o limite do sandbox
- `true` dá ao sandbox acesso direto ao diretório de onde você lançou o Hermes

Use o opt-in apenas quando você quer intencionalmente que o container trabalhe em arquivos vivos do host.

### Persistent shell

Por padrão, cada comando de terminal roda em seu próprio subprocess — working directory, variáveis de ambiente e variáveis de shell resetam entre comandos. Quando **persistent shell** está habilitado, um único processo bash long-lived é mantido entre chamadas `execute()` para que o estado sobreviva entre comandos.

Isso é mais útil para o **backend SSH**, onde também elimina overhead de conexão por comando. Persistent shell está **habilitado por padrão para SSH** e desabilitado para o backend local.

```yaml
terminal:
  persistent_shell: true   # default — enables persistent shell for SSH
```

Para desabilitar:

```bash
hermes config set terminal.persistent_shell false
```

**O que persiste entre comandos:**
- Working directory (`cd /tmp` permanece para o próximo comando)
- Variáveis de ambiente exportadas (`export FOO=bar`)
- Variáveis de shell (`MY_VAR=hello`)

**Precedência:**

| Level | Variable | Default |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| SSH override | `TERMINAL_SSH_PERSISTENT` | segue config |
| Local override | `TERMINAL_LOCAL_PERSISTENT` | `false` |

Variáveis de ambiente por backend têm precedência máxima. Se você quer persistent shell no backend local também:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
Comandos que requerem `stdin_data` ou sudo fazem fallback automático para mode one-shot, já que o stdin do persistent shell já está ocupado pelo protocolo IPC.
:::

Veja [Code Execution](features/code-execution.md) e a [seção Terminal do README](features/tools.md) para detalhes de cada backend.

## Configurações de skills {#skill-settings}

Skills podem declarar suas próprias configurações via frontmatter do SKILL.md. Esses são valores não secretos (paths, preferências, configurações de domínio) armazenados no namespace `skills.config` em `config.yaml`.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # Example — each skill defines its own keys
```

**Como as configurações de skill funcionam:**

- `hermes config migrate` escaneia todas as skills habilitadas, encontra configurações não definidas e oferece prompt
- `hermes config show` exibe todas as configurações de skill em "Skill Settings" com a skill a que pertencem
- Quando uma skill carrega, seus valores de config resolvidos são injetados no contexto da skill automaticamente

**Definindo valores manualmente:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

Para detalhes sobre declarar configurações nas suas próprias skills, veja [Creating Skills — Config Settings](/developer-guide/creating-skills#config-settings-configyaml).

### Guard em writes de skills criadas pelo agente {#guard-on-agent-created-skill-writes}

Quando o agente usa `skill_manage` para criar, editar, patch ou deletar uma skill, o Hermes pode opcionalmente escanear o conteúdo novo/atualizado em busca de padrões de keyword perigosos (credential harvesting, prompt injection óbvio, instruções de exfil). O scanner está **desligado por padrão** — workflows reais de agente que legitimamente tocam `~/.ssh/` ou mencionam `$OPENAI_API_KEY` estavam acionando a heurística com frequência demais. Reative se quiser que o scanner peça confirmação antes dos writes de skill do agente:

```yaml
skills:
  guard_agent_created: true   # default: false
```

Quando ligado, qualquer write `skill_manage` sinalizado aparece como prompt de approval com a rationale do scanner. Writes aceitos são aplicados; writes negados retornam erro explicativo ao agente.

### Write approval para writes de skill

Independente do scanner de conteúdo acima, `skills.write_approval` exige **sua** approval explícita para **todo** write de skill do agente (create / edit / patch / delete / arquivos de suporte) — o mesmo mecanismo approve/deny de comandos perigosos:

```yaml
skills:
  write_approval: false   # false = write freely (default) | true = stage every write for review
```

Quando ligado, writes de skill ficam staged em `~/.hermes/pending/skills/` e são revisados com `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>` — da CLI ou qualquer plataforma de messaging. Alterne em runtime com `/skills approval on|off`. Memory tem o mesmo gate (`memory.write_approval`, abaixo). Walkthrough completo: [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).

## Configuração de memory

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # true = require approval before any memory write
```

Com `memory.write_approval: true`, writes de memory precisam da sua approval antes de serem aplicados: turns interativos da CLI pedem inline; sessões de messaging e a background self-improvement review fazem stage do write para revisão via `/memory pending` → `/memory approve <id>` / `/memory reject <id>`. Alterne em runtime com `/memory approval on|off`. Veja [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval).

## Truncamento de context files

Controla quanto conteúdo o Hermes carrega de cada context file automático antes de aplicar truncamento head/tail. Isso se aplica a arquivos injetados no system prompt como `SOUL.md`, `.hermes.md`, `AGENTS.md`, `CLAUDE.md` e `.cursorrules`. **Não** afeta a ferramenta `read_file`.

```yaml
context_file_max_chars: 20000  # default
```

Aumente quando você mantém intencionalmente arquivos maiores de identidade ou contexto de projeto e usa models com context window suficiente para carregá-los:

```yaml
context_file_max_chars: 25000
```

## Segurança de leitura de arquivos

Controla quanto conteúdo uma única chamada `read_file` pode retornar. Leituras que excedem o limite são rejeitadas com erro dizendo ao agente para usar `offset` e `limit` para um range menor. Isso impede que uma única leitura de um bundle JS minificado ou arquivo de dados grande inunde a context window.

```yaml
file_read_max_chars: 100000  # default — ~25-35K tokens
```

Aumente se você está em um model com context window grande e lê arquivos grandes com frequência. Diminua para models de contexto pequeno para manter leituras eficientes:

```yaml
# Large context model (200K+)
file_read_max_chars: 200000

# Small local model (16K context)
file_read_max_chars: 30000
```

O agente também deduplica leituras de arquivo automaticamente — se a mesma região de arquivo é lida duas vezes e o arquivo não mudou, um stub leve é retornado em vez de reenviar o conteúdo. Isso reseta na context compression para o agente poder reler arquivos após o conteúdo ser resumido.

## Limites de truncamento de output de ferramentas

Três caps relacionados controlam quanto output bruto uma ferramenta pode retornar antes do Hermes truncar:

```yaml
tool_output:
  max_bytes: 50000        # terminal output cap (chars)
  max_lines: 2000         # read_file pagination cap
  max_line_length: 2000   # per-line cap in read_file's line-numbered view
```

- **`max_bytes`** — Quando um comando `terminal` produz mais que esse número de caracteres de stdout/stderr combinados, o Hermes mantém os primeiros 40% e últimos 60% e insere um aviso `[OUTPUT TRUNCATED]` entre eles. Default `50000` (≈12-15K tokens em tokenisers típicos).
- **`max_lines`** — Limite superior no parâmetro `limit` de uma única chamada `read_file`. Requisições acima disso são limitadas para uma única leitura não inundar a context window. Default `2000`.
- **`max_line_length`** — Cap por linha aplicado quando `read_file` emite a view com numeração de linhas. Linhas mais longas são truncadas para esse número de chars seguido de `... [truncated]`. Default `2000`.

Aumente os limites em models com context window grande que podem absorver mais output bruto por chamada. Diminua para models de contexto pequeno para manter resultados de ferramentas compactos:

```yaml
# Large context model (200K+)
tool_output:
  max_bytes: 150000
  max_lines: 5000

# Small local model (16K context)
tool_output:
  max_bytes: 20000
  max_lines: 500
```

## Desabilitação global de toolset

Para suprimir toolsets específicos na CLI e em toda plataforma gateway de uma vez,
liste os nomes em `agent.disabled_toolsets`:

```yaml
agent:
  disabled_toolsets:
    - memory       # hide memory tools + MEMORY_GUIDANCE injection
    - web          # no web_search / web_extract anywhere
```

Isso se aplica **depois** da config de ferramentas por plataforma (`platform_toolsets` escrita por
`hermes tools`), então um toolset listado aqui é sempre removido — mesmo se a
config salva de uma plataforma ainda o listar. Use quando quer um único
switch para "desligar X em todo lugar" em vez de editar 15+ linhas de plataforma na
UI `hermes tools`.

Deixar a lista vazia, ou omitir a chave, é no-op.

## Isolamento de git worktree

Habilite git worktrees isolados para rodar múltiplos agentes em paralelo no mesmo repo:

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

Quando habilitado, cada sessão CLI cria um worktree novo em `.worktrees/` com sua própria branch. Agentes podem editar arquivos, commit, push e criar PRs sem interferir uns nos outros. Worktrees limpos são removidos ao sair; sujos são mantidos para recuperação manual.

Por padrão a nova branch do worktree parte da **ponta remota recém-fetchada** (upstream da branch atual, senão a default branch do remote) para começar atualizada com o projeto em vez do `HEAD` local possivelmente stale do clone. Isso mantém o diff de um PR limitado à mudança real em vez de herdar o quanto o clone local estava atrás. Defina `worktree_sync: false` para ramificar do `HEAD` local — útil offline, ou quando você quer deliberadamente o estado exato atual do clone como base. Se o remote não puder ser alcançado, faz fallback automático para `HEAD` local.

```yaml
worktree_sync: true    # Default — branch from the fetched remote tip
# worktree_sync: false # Branch from local HEAD (offline / pinned base)
```

Você também pode listar arquivos gitignored para copiar em worktrees via `.worktreeinclude` na raiz do repo:

```
# .worktreeinclude
.env
.venv/
node_modules/
```

## Context compression

O Hermes comprime conversas longas automaticamente para permanecer dentro da context window do seu model. O summarizer de compression é uma chamada LLM separada — você pode apontá-lo para qualquer provider ou endpoint.

Todas as configurações de compression ficam em `config.yaml` (sem variáveis de ambiente).

### Referência completa

```yaml
compression:
  enabled: true                                     # Toggle compression on/off
  threshold: 0.50                                   # Compress at this % of context limit
  target_ratio: 0.20                                # Fraction of threshold to preserve as recent tail
  protect_last_n: 20                                # Min recent messages to keep uncompressed
  protect_first_n: 3                                # Non-system head messages pinned across compactions (0 = pin nothing)
  hygiene_hard_message_limit: 5000                  # Gateway safety valve — see below

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: ""                                       # Empty = use main chat model. Override with e.g. "google/gemini-3-flash-preview" for cheaper/faster compression.
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info Migração de config legada
Configs antigas com `compression.summary_model`, `compression.summary_provider` e `compression.summary_base_url` são migradas automaticamente para `auxiliary.compression.*` no primeiro load (config version 17). Nenhuma ação manual necessária.
:::

`hygiene_hard_message_limit` é uma **válvula de segurança pré-compression** exclusiva do gateway. Existe para quebrar uma espiral de morte: quando chamadas API continuam desconectando em sessão oversized, o gateway nunca recebe dados de uso de tokens, então o threshold baseado em tokens não dispara, o transcript continua crescendo e desconexões pioram. Esse piso baseado em contagem de mensagens dispara só pela contagem (sempre conhecida, independente de falhas API) para forçar compression e recuperar a sessão. Default `5000` — bem acima de qualquer sessão normal, incluindo models de contexto grande (1M+) fazendo milhares de turns curtos, que comprimem no threshold de tokens muito antes disso. Aumente para plataformas incomuns, diminua para forçar compression mais agressiva. Editar esse valor em gateway rodando entra em vigor na próxima mensagem (veja abaixo).

`protect_first_n` controla quantas mensagens **não-system** do início são pinned em toda compaction. Default `3` — a troca user/assistant inicial sobrevive a todo pass do summarizer para o objetivo original permanecer visível. Em sessões longas de rolling-compaction onde o turn inicial não é mais relevante, defina `protect_first_n: 0` para não pinar nada além do system prompt + summary + tail. O system prompt em si é sempre preservado independente dessa config.

:::tip Hot-reload de compression e context length no gateway
Em releases recentes, editar `model.context_length` ou qualquer chave `compression.*` em `config.yaml` em gateway rodando entra em vigor na próxima mensagem — sem restart do gateway, sem `/reset`, sem rotação de sessão. A assinatura do cached-agent inclui essas chaves, então o gateway reconstrói o agent transparentemente ao ver mudança. API keys e config de tool/skill ainda exigem os caminhos de reload usuais.
:::

### Configurações comuns

**Default (auto-detect) — nenhuma configuração necessária:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
Usa seu provider principal e model principal. Override por task (ex.: `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`) se quiser compression em model mais barato que seu main chat model.

**Forçar um provider específico** (OAuth ou baseado em API key):
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
Funciona com qualquer provider: `nous`, `openrouter`, `codex`, `anthropic`, `main`, etc.

**Endpoint customizado** (self-hosted, Ollama, zai, DeepSeek, etc.):
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
Aponta para endpoint OpenAI-compatible customizado. Usa `OPENAI_API_KEY` para auth.

### Como os três knobs interagem

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | Result |
|---------------------|---------------------|--------|
| `auto` (default) | not set | Auto-detect best available provider |
| `nous` / `openrouter` / etc. | not set | Force that provider, use its auth |
| any | set | Use the custom endpoint directly (provider ignored) |

:::warning Requisito de context length do summary model
O summary model **deve** ter context window pelo menos tão grande quanto o do main agent model. O compressor envia a seção middle completa da conversa ao summary model — se a context window desse model for menor que a do main model, a chamada de summarization falhará com erro de context length. Quando isso acontece, os turns do meio são **descartados sem summary**, perdendo contexto da conversa silenciosamente. Se você override o model, verifique se sua context length atende ou excede a do main model.
:::

## Context engine

O context engine controla como conversas são gerenciadas ao se aproximar do token limit do model. O engine built-in `compressor` usa summarization lossy (veja [Context Compression](/developer-guide/context-compression-and-caching)). Plugin engines podem substituí-lo por estratégias alternativas.

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
```

Para usar um plugin engine (ex.: LCM para lossless context management):

```yaml
context:
  engine: "lcm"          # must match the plugin's name
```

Plugin engines **nunca são auto-ativados** — você deve definir explicitamente `context.engine` para o nome do plugin. Engines disponíveis podem ser navegados e selecionados via `hermes plugins` → Provider Plugins → Context Engine.

Veja [Memory Providers](/user-guide/features/memory-providers) para o sistema análogo de seleção única para plugins de memory.

## Iteration budget

Quando o agente trabalha em task complexa com muitas tool calls, pode esgotar seu iteration budget (default: 90 turns). O Hermes **não** injeta avisos de pressão mid-task — builds anteriores avisavam o model em 70%/90% do budget, o que fazia models abandonarem tasks complexas prematuramente e foi removido em abril de 2026.

Em vez disso, quando o budget é realmente esgotado (90/90), o Hermes injeta uma mensagem pedindo ao model para concluir e permite uma única **grace call** para entregar resposta final. Se essa grace call ainda não produzir texto, o agente é pedido para resumir o que realizou.

```yaml
agent:
  max_turns: 90                # Max iterations per conversation turn (default: 90)
  api_max_retries: 3           # Retries per provider before fallback engages (default: 3)
```

Quando o iteration budget é totalmente esgotado, a CLI mostra notificação ao usuário: `⚠ Iteration budget reached (90/90) — response may be incomplete`.

`agent.api_max_retries` controla quantas vezes o Hermes retenta uma chamada API de provider em erros transientes (rate limits, connection drops, 5xx) **antes** do fallback-provider switching entrar. O default é `3` — quatro tentativas no total. Se você tem [fallback providers](/user-guide/features/fallback-providers) configurados e quer fail over mais rápido, diminua para `0` para o primeiro erro transient no primary passar imediatamente ao fallback em vez de churning retries no endpoint instável.

## Standing goals (`/goal`)

Quando um standing goal está ativo, o Hermes julga se cada resposta assistant o satisfaz. Se não, alimenta um continuation prompt de volta na mesma sessão e continua trabalhando até o goal estar done, o turn budget esgotado, ou o usuário pausar/limpar. O turn budget é o backstop real — falhas do judge fazem **fail open** (continuar) para um judge instável nunca travar progresso.

```yaml
goals:
  max_turns: 20   # Max continuation turns before Hermes auto-pauses the goal (default: 20)
```

`max_turns` limita quantos continuation turns um goal pode dirigir antes do Hermes auto-pausá-lo e pedir ao usuário `/goal resume`. Protege contra false negatives do judge (goal na verdade done mas judge diz continue) e gasto ilimitado de model em goals fuzzy ou inatingíveis. Veja [Goals](/user-guide/features/goals) para o feature completo.

### API timeouts

O Hermes tem camadas de timeout separadas para streaming, mais um detector stale para chamadas non-streaming. Os detectores stale auto-ajustam para providers locais apenas quando você os deixa nos defaults implícitos.

| Timeout | Default | Local providers | Config / env |
|---------|---------|----------------|--------------|
| Socket read timeout | 120s | Auto-raised to 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Stale stream detection | 180s | Auto-disabled | `HERMES_STREAM_STALE_TIMEOUT` |
| Stale non-stream detection | 300s | Auto-disabled when left implicit | `providers.<id>.stale_timeout_seconds` or `HERMES_API_CALL_STALE_TIMEOUT` |
| API call (non-streaming) | 1800s | Unchanged | `providers.<id>.request_timeout_seconds` / `timeout_seconds` or `HERMES_API_TIMEOUT` |

O **socket read timeout** controla quanto o httpx espera pelo próximo chunk de dados do provider. LLMs locais podem levar minutos para prefill em contextos grandes antes do primeiro token, então o Hermes eleva para 30 minutos quando detecta endpoint local. Se você define explicitamente `HERMES_STREAM_READ_TIMEOUT`, esse valor é sempre usado independente da detecção de endpoint.

A **stale stream detection** mata conexões que recebem SSE keep-alive pings mas nenhum conteúdo real. Isso é desabilitado inteiramente para providers locais já que não enviam keep-alive pings durante prefill.

A **stale non-stream detection** mata chamadas non-streaming que não produzem resposta por tempo demais. Por padrão o Hermes desabilita isso em endpoints locais para evitar false positives durante prefills longos. Se você define explicitamente `providers.<id>.stale_timeout_seconds`, `providers.<id>.models.<model>.stale_timeout_seconds`, ou `HERMES_API_CALL_STALE_TIMEOUT`, esse valor explícito é respeitado mesmo em endpoints locais.

## Avisos de context pressure

Separado da pressão de iteration budget, context pressure rastreia quão perto a conversa está do **compaction threshold** — o ponto onde context compression dispara para resumir mensagens antigas. Isso ajuda você e o agente a entender quando a conversa está ficando longa.

| Progress | Level | What happens |
|----------|-------|-------------|
| **≥ 60%** to threshold | Info | CLI mostra barra de progresso ciano; gateway envia aviso informativo |
| **≥ 85%** to threshold | Warning | CLI mostra barra amarela em negrito; gateway avisa que compaction é iminente |

Na CLI, context pressure aparece como barra de progresso no feed de output de ferramentas:

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

Em plataformas de messaging, uma notificação em texto simples é enviada:

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

Se auto-compression está desabilitada, o aviso diz que o contexto pode ser truncado em vez disso.

Context pressure é automático — nenhuma configuração necessária. Dispara puramente como notificação user-facing e não modifica o message stream nem injeta nada no contexto do model.

## Estratégias de credential pool {#credential-pool-strategies}

Quando você tem múltiplas API keys ou tokens OAuth para o mesmo provider, configure a estratégia de rotação:

```yaml
credential_pool_strategies:
  openrouter: round_robin    # cycle through keys evenly
  anthropic: least_used      # always pick the least-used key
```

Opções: `fill_first` (default), `round_robin`, `least_used`, `random`. Veja [Credential Pools](/user-guide/features/credential-pools) para documentação completa.

## Prompt caching

O Hermes liga prompt caching cross-session automaticamente quando o provider ativo suporta — nenhuma config de usuário necessária.

Para Claude em **native Anthropic**, **OpenRouter** e **Nous Portal**, o Hermes anexa breakpoints `cache_control` com TTL de 1 hora (`ttl: "1h"`) no system prompt e blocos de skill. O primeiro envio dentro de uma hora fresca paga taxas de input completas; envios subsequentes em qualquer sessão dentro da mesma hora puxam do cache na taxa discounted de cached-read. Isso significa que system prompt, conteúdo de skill carregado e a porção inicial de qualquer include de long-context são reutilizados entre sessões `hermes` e entre subagents forked na primeira hora.

O upstream Qwen Cloud (Alibaba DashScope) limita cache TTL a 5 minutos, então o Hermes usa breakpoint TTL de 5 minutos lá. Outros caminhos Claude-via-third-party (AWS Bedrock, Azure Foundry) fazem fallback para os defaults de caching do provider. xAI Grok usa mecanismo separado de conversation-id pinned por sessão — veja [xAI prompt caching](/integrations/providers#xai-grok--responses-api--prompt-caching).

Nenhum knob existe para desabilitar isso — caching é always-on e economiza dinheiro mesmo em conversas single-turn porque o system prompt sozinho é fração significativa da contagem de input tokens.

O único knob explícito é o tier de cache TTL que o Hermes solicita em breakpoints estilo Anthropic:

```yaml
prompt_caching:
  cache_ttl: "5m"   # "5m" or "1h" (Anthropic-supported tiers); other values are ignored
```

`cache_ttl` seleciona o breakpoint TTL que o Hermes anexa para Claude via native Anthropic API, OpenRouter e Nous Portal. Apenas os dois tiers suportados pela Anthropic (`"5m"`, `"1h"`) são respeitados — qualquer outro valor é ignorado. Providers com seus próprios caps (ex.: Qwen Cloud, que maxima em 5 minutos) ainda limitam ao que o upstream permite.

## Auxiliary models

O Hermes usa models "auxiliary" para side tasks como análise de imagem, summarization de páginas web, análise de screenshots do browser, geração de título de sessão e context compression. Por padrão (`auxiliary.*.provider: "auto"`), o Hermes encaminha toda auxiliary task para seu **main chat model** — o mesmo provider/model que você escolheu em `hermes model`. Você não precisa configurar nada para começar, mas esteja ciente de que em reasoning models caros (Opus, MiniMax M2.7, etc.) auxiliary tasks adicionam custo significativo. Se quer side tasks baratas e rápidas independente do main model, defina `auxiliary.<task>.provider` e `auxiliary.<task>.model` explicitamente (por exemplo, Gemini Flash no OpenRouter para vision e web extraction).

:::note Por que "auto" usa seu main model
Builds anteriores separavam usuários de aggregator (OpenRouter, Nous Portal) para um default barato do lado do provider. Isso era surpreendente — usuários que pagaram assinatura de aggregator viam model diferente lidando com tráfego auxiliary. `auto` agora usa o main model para todos, e overrides por task em `config.yaml` ainda prevalecem (veja [Referência completa de config auxiliary](#full-auxiliary-config-reference) abaixo).
:::

### Configurando auxiliary models interativamente

Em vez de editar YAML manualmente, execute `hermes model` e escolha **"Configure auxiliary models"** no menu. Você recebe um picker interativo por task:

```
$ hermes model
→ Configure auxiliary models

[ ] vision               currently: auto / main model
[ ] web_extract          currently: auto / main model
[ ] title_generation     currently: openrouter / google/gemini-3-flash-preview
[ ] tts_audio_tags       currently: auto / main model
[ ] compression          currently: auto / main model
[ ] approval             currently: auto / main model
[ ] triage_specifier     currently: auto / main model
[ ] kanban_decomposer    currently: auto / main model
[ ] profile_describer    currently: auto / main model
```

Selecione uma task, escolha um provider (OAuth flows abrem browser; providers com API key pedem), escolha um model. A mudança persiste em `auxiliary.<task>.*` em `config.yaml`. Mesma maquinaria do picker de main model — nenhuma sintaxe extra para aprender.

Se você não quer que o Hermes auto-gere títulos após a primeira troca, defina
`auxiliary.title_generation.enabled: false`. Títulos manuais ainda funcionam via
`/title` e `hermes sessions rename`.

### Video tutorial

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### O padrão universal de config

Todo model slot no Hermes — auxiliary tasks, compression, fallback — usa os mesmos três knobs:

| Key | What it does | Default |
|-----|-------------|---------|
| `provider` | Qual provider usar para auth e routing | `"auto"` |
| `model` | Qual model solicitar | provider's default |
| `base_url` | Endpoint OpenAI-compatible customizado (override provider) | not set |

Blocos de auxiliary task aceitam adicionalmente o knob `reasoning_effort`:

| Key | What it does | Default |
|-----|-------------|---------|
| `reasoning_effort` | Nível de thinking para chamadas LLM dessa task: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` | not set (provider default) |

Esse é o counterpart por task do `agent.reasoning_effort` global: rode compression em `low` ou vision em `none` para cortar latência e custo de side tasks quando seu main model é reasoning model caro, sem tocar no comportamento do main chat. Funciona em todo bloco de auxiliary task (`vision`, `web_extract`, `compression`, `title_generation`, `curator`, `background_review`, ...), nos três wire formats auxiliary (chat completions, Codex Responses, Anthropic Messages). Um `extra_body.reasoning` explícito na mesma task prevalece sobre o shorthand.

MoA é a exceção: reasoning depth para Mixture-of-Agents é configurado **por slot** no preset MoA (`moa.presets.<name>.reference_models[].reasoning_effort` / `aggregator.reasoning_effort`), não nos blocos auxiliary `moa_reference`/`moa_aggregator` — veja [Mixture of Agents](/user-guide/features/mixture-of-agents).

```yaml
auxiliary:
  compression:
    reasoning_effort: "low"    # summaries don't need deep thinking
  vision:
    reasoning_effort: "none"   # disable thinking for image description
```

Quando `base_url` está definido, o Hermes ignora o provider e chama esse endpoint diretamente (usando `api_key` ou `OPENAI_API_KEY` para auth). Quando apenas `provider` está definido, o Hermes usa auth e base URL built-in desse provider.

Providers disponíveis para auxiliary tasks: `auto`, `main`, mais qualquer provider no [provider registry](/reference/environment-variables) — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `azure-foundry` — ou qualquer custom provider nomeado da sua lista `custom_providers` (ex.: `provider: "beans"`).

:::tip MiniMax OAuth
`minimax-oauth` faz login via browser OAuth (sem API key necessária). Execute `hermes model` e selecione **MiniMax (OAuth)** para autenticar. Auxiliary tasks usam `MiniMax-M2.7-highspeed` automaticamente. Veja o [guia MiniMax OAuth](../guides/minimax-oauth.md).
:::

:::tip xAI Grok OAuth
`xai-oauth` faz login via browser OAuth para assinantes SuperGrok e X Premium+ (sem API key necessária). Execute `hermes model` e selecione **xAI Grok OAuth (SuperGrok / Premium+)** para autenticar. O mesmo token OAuth é reutilizado para toda superfície direct-to-xAI (chat, auxiliary tasks, TTS, image gen, video gen, transcription). Veja o [guia xAI Grok OAuth](../guides/xai-grok-oauth.md), e se o Hermes está em host remoto veja [OAuth over SSH / Remote Hosts](../guides/oauth-over-ssh.md).
:::

:::warning `"main"` é apenas para auxiliary tasks
A opção de provider `"main"` significa "usar qualquer provider que meu main agent usa" — é válida apenas dentro de `auxiliary:`, `compression:`, e entradas de fallback primário (`fallback_providers:` ou legacy `fallback_model:`). **Não** é valor válido para sua config top-level `model.provider`. Se você usa endpoint OpenAI-compatible customizado, defina `provider: custom` na seção `model:`. Veja [AI Providers](/integrations/providers) para todas as opções de main model provider.
:::

### Referência completa de config auxiliary {#full-auxiliary-config-reference}

```yaml
auxiliary:
  # Image analysis (vision_analyze tool + browser screenshots)
  vision:
    provider: "auto"           # "auto", "openrouter", "nous", "codex", "main", etc.
    model: ""                  # e.g. "openai/gpt-4o", "google/gemini-2.5-flash"
    base_url: ""               # Custom OpenAI-compatible endpoint (overrides provider)
    api_key: ""                # API key for base_url (falls back to OPENAI_API_KEY)
    timeout: 120               # seconds — LLM API call timeout; vision payloads need generous timeout
    download_timeout: 30       # seconds — image HTTP download; increase for slow connections
    max_concurrency: 8         # max concurrent image encode/resize bursts across the process
                               # (default: host CPU core count, no ceiling) — bounds only the
                               # CPU-bound encode step so a video-frame fan-out can't saturate
                               # every core and starve the event loop; LLM calls stay fully
                               # concurrent. Minimum 1; values < 1 are ignored.

  # Web page summarization + browser page text extraction
  web_extract:
    provider: "auto"
    model: ""                  # e.g. "google/gemini-2.5-flash"
    base_url: ""
    api_key: ""
    timeout: 360               # seconds (6min) — per-attempt LLM summarization

  # Dangerous command approval classifier
  approval:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30                # seconds

  # Gemini 3.1 TTS hidden audio-tag insertion
  tts_audio_tags:
    provider: "auto"
    model: ""                  # empty = main chat model
    base_url: ""
    api_key: ""
    timeout: 30

  # Context compression timeout (separate from compression.* config)
  compression:
    timeout: 120               # seconds — compression summarizes long conversations, needs more time
    # fallback_chain:           # Optional — providers to try on rate-limit / connectivity failure
    #   - provider: nous
    #     model: deepseek/deepseek-chat
    #   - provider: openrouter
    #     model: google/gemini-2.5-flash
    #     base_url: ""
    #     api_key: ""

  # Auto-generated session titles. Empty language follows the conversation;
  # set e.g. "English" or "Japanese" to pin titles to one language.
  title_generation:
    enabled: true              # set false to disable auto-title generation
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    language: ""

  # Skills hub — skill matching and search
  skills_hub:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # MCP tool dispatch
  mcp:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30

  # Kanban triage specifier — `hermes kanban specify <id>` (or the
  # dashboard's ✨ Specify button on Triage-column cards) uses this
  # slot to expand a one-liner into a concrete spec and promote the
  # task to `todo`. Cheap fast models work well here; spec expansion
  # is short and doesn't need reasoning depth.
  triage_specifier:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 120
```

:::tip
Cada auxiliary task tem `timeout` configurável (em segundos). Defaults: vision 120s, web_extract 360s, approval 30s, compression 120s. Aumente se usa models locais lentos para auxiliary tasks. Vision também tem `download_timeout` separado (default 30s) para download HTTP de imagem — aumente para conexões lentas ou image servers self-hosted.
:::

:::info
Context compression tem seu próprio bloco `compression:` para thresholds e bloco `auxiliary.compression:` para config de model/provider — veja [Context Compression](#context-compression) acima. A cadeia de fallback primária usa lista top-level `fallback_providers:` — veja [Fallback Providers](/integrations/providers#fallback-providers). Os três seguem o mesmo padrão provider/model/base_url.
:::

### Fallback chain por task para auxiliary tasks

Cada auxiliary task pode opcionalmente definir `fallback_chain` — lista de entradas provider/model que o Hermes tenta quando o auxiliary provider primário falha por rate limits, problemas de conectividade ou restrições de pagamento:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openai/gpt-4o-mini
    fallback_chain:
      - provider: nous
        model: deepseek/deepseek-chat
      - provider: openrouter
        model: google/gemini-2.5-flash
```

Quando o auxiliary provider primário (`openrouter` / `openai/gpt-4o-mini`) retorna rate-limit, connection timeout ou payment-required error, o Hermes percorre `fallback_chain` em ordem. Pula entradas cujo provider coincide com o provider já falho, e tenta cada entrada restante até uma ter sucesso ou a cadeia esgotar. Se todos fallbacks falham, o Hermes faz fallback para o main agent model como rede de segurança final.

Cada entrada suporta os mesmos três knobs de qualquer config de auxiliary task:

| Key | Description |
|-----|-------------|
| `provider` | Nome do provider (`nous`, `openrouter`, `anthropic`, `gemini`, `main`, etc.) |
| `model` | Nome do model para esse provider |
| `base_url` | (Opcional) Endpoint OpenAI-compatible customizado |

`fallback_chain` está disponível em qualquer auxiliary task — `compression`, `vision`, `web_extract`, `approval`, `skills_hub`, `mcp`, etc.

### OpenRouter routing e Pareto Code para auxiliary tasks {#openrouter-routing--pareto-code-for-auxiliary-tasks}

Quando uma auxiliary task resolve para OpenRouter (explicitamente ou via `provider: "main"` enquanto seu main agent está no OpenRouter), as configurações `provider_routing` e `openrouter.min_coding_score` do main agent **não propagam** — por design, cada auxiliary task é independente. Para definir preferências de provider OpenRouter ou usar o [Pareto Code router](/integrations/providers#openrouter-pareto-code-router) para uma aux task específica, defina-as por task via `extra_body`:

```yaml
auxiliary:
  compression:
    provider: openrouter
    model: openrouter/pareto-code         # use the Pareto Code router for this task
    extra_body:
      provider:                            # OpenRouter provider routing prefs
        order: [anthropic, google]         # try these providers in order
        sort: throughput                   # or "price" | "latency"
        # only: [anthropic]                # restrict to a specific provider
        # ignore: [deepinfra]              # exclude specific providers
      plugins:                             # OpenRouter Pareto Code router knob
        - id: pareto-router
          min_coding_score: 0.5            # 0.0–1.0; higher = stronger coders
```

A forma espelha o que OpenRouter aceita no request body de chat completions. O Hermes encaminha o `extra_body` inteiro verbatim, então qualquer outro campo de request-body OpenRouter documentado em [openrouter.ai/docs](https://openrouter.ai/docs) funciona da mesma forma.

### Alterando o vision model

Para usar GPT-4o em vez de Gemini Flash para análise de imagem:

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

Ou via variável de ambiente (em `~/.hermes/.env`):

```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### Opções de provider

Essas opções se aplicam a **configs de auxiliary task** (`auxiliary:`, `compression:`) e entradas de fallback primário (`fallback_providers:` ou legacy `fallback_model:`), não à sua config `model.provider` principal.

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `"auto"` | Melhor disponível (default). Vision tenta OpenRouter → Nous → Codex. | — |
| `"openrouter"` | Força OpenRouter — roteia para qualquer model (Gemini, GPT-4o, Claude, etc.) | `OPENROUTER_API_KEY` |
| `"nous"` | Força Nous Portal | `hermes auth` |
| `"codex"` | Força Codex OAuth (conta ChatGPT). Suporta vision (gpt-5.3-codex). | `hermes model` → Codex |
| `"minimax-oauth"` | Força MiniMax OAuth (login browser, sem API key). Usa MiniMax-M2.7-highspeed para auxiliary tasks. | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | Força xAI Grok OAuth (login browser para assinantes SuperGrok ou X Premium+, sem API key). Mesmo token OAuth cobre chat, TTS, image, video e transcription. | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | Usa seu endpoint custom/main ativo. Pode vir de `OPENAI_BASE_URL` + `OPENAI_API_KEY` ou endpoint custom salvo via `hermes model` / `config.yaml`. Funciona com OpenAI, models locais ou qualquer API OpenAI-compatible. **Apenas auxiliary tasks — inválido para `model.provider`.** | Credenciais de endpoint custom + base URL |

Providers com API key direta do catálogo principal também funcionam aqui quando você quer side tasks bypassando seu router default. Por exemplo, `gmi` é válido quando `GMI_API_KEY` está configurada, e `fireworks` é válido quando `FIREWORKS_API_KEY` está configurada:

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

Para routing auxiliary GMI, use o model ID exato retornado pelo endpoint `/v1/models` da GMI. Model IDs Fireworks usam a forma slash nativa do provider, por exemplo `accounts/fireworks/models/glm-5p2`.

### Configurações comuns

**Usando endpoint custom direto** (mais claro que `provider: "main"` para APIs locais/self-hosted):
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` tem precedência sobre `provider`, então essa é a forma mais explícita de rotear auxiliary task para endpoint específico. Para overrides de endpoint direto, o Hermes usa `api_key` configurada ou faz fallback para `OPENAI_API_KEY`; não reutiliza `OPENROUTER_API_KEY` para esse endpoint custom.

**Usando OpenAI API key para vision:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**Usando OpenRouter para vision** (rotear para qualquer model):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Usando Codex OAuth** (conta ChatGPT Pro/Plus — sem API key necessária):
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    # model defaults to gpt-5.3-codex (supports vision)
```

**Usando MiniMax OAuth** (login browser, sem API key necessária):
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
Execute `hermes model` e selecione **MiniMax (OAuth)** para login e definir isso automaticamente. Para região China, a base URL será `https://api.minimaxi.com/anthropic`. Veja o [guia MiniMax OAuth](../guides/minimax-oauth.md) para walkthrough completo.

**Usando model local/self-hosted:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your active custom endpoint
    model: "my-local-model"
```

`provider: "main"` usa qualquer provider que o Hermes usa para chat normal — seja custom provider nomeado (ex.: `beans`), provider built-in como `openrouter`, ou endpoint legacy `OPENAI_BASE_URL`.

:::tip
Se você usa Codex OAuth como main model provider, vision funciona automaticamente — nenhuma config extra necessária. Codex está incluído na cadeia de auto-detection para vision.
:::

:::warning
**Vision requer model multimodal.** Se você define `provider: "main"`, certifique-se de que seu endpoint suporta multimodal/vision — senão análise de imagem falhará.
:::

### Variáveis de ambiente (legado)

Auxiliary models também podem ser configurados via variáveis de ambiente. Porém, `config.yaml` é o método preferido — é mais fácil de gerenciar e suporta todas as opções incluindo `base_url` e `api_key`.

| Setting | Environment Variable |
|---------|---------------------|
| Vision provider | `AUXILIARY_VISION_PROVIDER` |
| Vision model | `AUXILIARY_VISION_MODEL` |
| Vision endpoint | `AUXILIARY_VISION_BASE_URL` |
| Vision API key | `AUXILIARY_VISION_API_KEY` |
| Web extract provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web extract model | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web extract endpoint | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web extract API key | `AUXILIARY_WEB_EXTRACT_API_KEY` |

Configurações de compression e fallback model são apenas config.yaml.

:::tip
Execute `hermes config` para ver suas configurações atuais de auxiliary model. Overrides só aparecem quando diferem dos defaults.
:::

## Reasoning effort

Controle quanto "thinking" o model faz antes de responder:

```yaml
agent:
  reasoning_effort: ""   # empty = medium. Options: none, minimal, low, medium, high, xhigh, max, ultra
```

Quando não definido (default), reasoning effort default é "medium" — nível equilibrado que funciona bem para a maioria das tasks. Definir um valor faz override — reasoning effort mais alto dá melhores resultados em tasks complexas ao custo de mais tokens e latência.

:::note Models adaptive-thinking (Claude 4.6+, Fable/Mythos-class) via OpenRouter
Esses models usam thinking *adaptativo* e não aceitam o campo usual `reasoning.effort`
— OpenRouter o ignora para eles. O Hermes roteia transparentemente seu
`reasoning_effort` para o parâmetro `verbosity` do OpenRouter (que mapeia para
`output_config.effort` da Anthropic), então o mesmo knob de effort continua funcionando com
os níveis suportados pelo model selecionado. `none` (ou unset) deixa o model
no default adaptativo próprio. O
native Anthropic provider já controla effort diretamente e não é afetado.
:::

Você também pode alterar reasoning effort em runtime com o comando `/reasoning`:

```
/reasoning                # Show current effort level and display state
/reasoning high           # Set reasoning effort to high (this session only)
/reasoning high --global  # Set effort and persist to config.yaml
/reasoning none           # Disable reasoning (this session only)
/reasoning show           # Show model thinking above each response
/reasoning hide           # Hide model thinking
```

Mudanças de effort são com escopo de sessão por padrão; adicione `--global` para salvar o
novo nível como default de `agent.reasoning_effort`.

#### Overrides de reasoning por model

Você pode definir níveis diferentes de reasoning effort para models diferentes. Isso é útil quando quer reasoning alto para models complexos mas medium para os mais rápidos:

```yaml
agent:
  reasoning_effort: "medium"       # global default
  reasoning_overrides:
    "openrouter/anthropic/claude-opus-4.5": "xhigh"
    "openai/gpt-5": "low"
    "claude-sonnet-4.6": "high"    # bare model name also works
```

A correspondência de chave é **tolerante a grafia** — qualquer grafia razoável corresponderá:
- `claude-opus-4.5`, `claude-opus-4-5`, `claude-opus.4.5` (dots e dashes são intercambiáveis)
- `anthropic/claude-opus-4.5`, `openrouter/anthropic/claude-opus-4.5` (prefixo de provider opcional)
- Matches exatos têm precedência sobre variantes

:::note
Não há suporte `hermes config set` para chaves `reasoning_overrides` — edite o arquivo YAML diretamente. Isso porque nomes de model frequentemente contêm dots (ex.: `claude-opus-4.5`), que conflitam com a sintaxe dotted-key da CLI.
:::

**Prioridade de resolução:**

1. Override com escopo de sessão `/reasoning --session` (gateway only)
2. Override por model de `agent.reasoning_overrides` (tolerante a grafia)
3. `agent.reasoning_effort` global
4. Default do provider

O override se aplica automaticamente em todo lugar: startup CLI, messaging gateway, Desktop/TUI, cron jobs, switches mid-session `/model`, e ativação de fallback model.

## Tool-use enforcement

Alguns models ocasionalmente descrevem ações pretendidas como texto em vez de fazer tool calls ("I would run the tests..." em vez de chamar terminal de fato). Tool-use enforcement injeta guidance no system prompt que direciona o model de volta a chamar ferramentas de fato.

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Habilitado para models matching: `gpt`, `codex`, `gemini`, `gemma`, `grok`. Desabilitado para todos os outros (Claude, DeepSeek, Qwen, etc.). |
| `true` | Sempre habilitado, independente do model. Útil se notar seu model atual descrevendo ações em vez de executá-las. |
| `false` | Sempre desabilitado, independente do model. |
| `["gpt", "codex", "qwen", "llama"]` | Habilitado apenas quando o nome do model contém uma das substrings listadas (case-insensitive). |

### O que injeta

Quando habilitado, três camadas de guidance podem ser adicionadas ao system prompt:

1. **Tool-use enforcement geral** (todos models matched) — instrui o model a fazer tool calls imediatamente em vez de descrever intenções, continuar trabalhando até a task estar completa, e nunca terminar um turn com promessa de ação futura.

2. **OpenAI execution discipline** (apenas models GPT e Codex) — guidance adicional abordando failure modes específicos de GPT: abandonar trabalho em resultados parciais, pular lookups de prerequisite, alucinar em vez de usar ferramentas, e declarar "done" sem verificação.

3. **Google operational guidance** (apenas models Gemini e Gemma) — concisão, absolute paths, parallel tool calls, e padrões verify-before-edit.

Isso é transparente ao usuário e afeta apenas o system prompt. Models que já usam ferramentas de forma confiável (como Claude) não precisam dessa guidance, por isso `"auto"` os exclui.

### Quando ligar

Se você usa model fora da lista auto default e nota que frequentemente descreve o que *faria* em vez de fazer, defina `tool_use_enforcement: true` ou adicione a substring do model à lista:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## Tool-loop guardrails

O Hermes detecta quando o agente está preso em loop improdutivo de tool calling — a mesma tool call falhando repetidamente, a mesma ferramenta falhando sem parar, ou chamada idempotente retornando o mesmo resultado sem progresso. Por padrão injeta **aviso** no tool result para o model se autocorrigir; não para hard, já que quem observa CLI/TUI pode intervir.

Para deployments gateway / server unattended, habilite hard stops para um agente preso ser circuit-broken em vez de queimar iteration budget:

```yaml
tool_loop_guardrails:
  warnings_enabled: true       # inject warnings into tool results (default: true)
  hard_stop_enabled: false     # also BLOCK the call past the hard-stop threshold (default: false)
  warn_after:
    exact_failure: 2           # identical failing call repeated N times
    same_tool_failure: 3       # same tool failing N times (different args)
    idempotent_no_progress: 2  # same result, no progress, N times
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
```

`hard_stop_enabled` default é `false` porque sessões interativas têm humano no loop. Em deployments unattended (gateway, cron, kanban workers) defina como `true` para falhas repetidas serem bloqueadas em vez de apenas avisadas. Veja também [Docker / unattended deployments](docker.md).

## Configuração TTS

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts"
  speed: 1.0                    # Global speed multiplier (fallback for all providers)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Speed multiplier (converted to rate percentage, e.g. 1.5 → +50%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0                  # Speed multiplier (clamped to 0.25–4.0 by the API)
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
  minimax:
    speed: 1.0                  # Speech speed multiplier
    # base_url: ""              # Optional: override for OpenAI-compatible TTS endpoints
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"   # or gemini-3.1-flash-tts-preview
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, etc.
    audio_tags: false           # Hidden Gemini 3.1 TTS audio-tag insertion
    persona_prompt_file: ""      # Optional Markdown/text file with Gemini voice direction
  xai:
    voice_id: "eve"             # xAI TTS voice
    language: "en"              # ISO 639-1
    sample_rate: 24000
    bit_rate: 128000            # MP3 bitrate
    # base_url: "https://api.x.ai/v1"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

Isso controla tanto a ferramenta `text_to_speech` quanto respostas faladas em voice mode (`/voice tts` na CLI ou messaging gateway).

**Hierarquia de fallback de speed:** speed específico do provider (ex.: `tts.edge.speed`) → `tts.speed` global → default `1.0`. Defina `tts.speed` global para aplicar speed uniforme em todos providers, ou override por provider para controle fino.

## Configurações de display {#display-settings}

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # Enable /verbose slash command in messaging gateway
  platforms: {}           # Per-platform display overrides (see below)
  tool_progress_overrides: {}  # DEPRECATED — use display.platforms instead
  interim_assistant_messages: true  # Gateway: send natural mid-turn assistant updates as separate messages
  show_commentary: true   # Codex models: deliver commentary-channel progress narration as visible mid-turn updates
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: "kawaii"  # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  show_reasoning: false   # Show model reasoning/thinking above each response (toggle with /reasoning show|hide)
  streaming: false        # Stream tokens to terminal as they arrive (real-time output)
  show_cost: false        # Show estimated $ cost in the CLI status bar
  timestamps: false       # When true, prefixes user and assistant labels with [HH:MM] timestamps in the CLI / TUI transcript
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  runtime_footer:         # Gateway: append a runtime-context footer to final replies
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # Append an advisory footer when write_file/patch calls failed this turn
  credits_notices: true   # Nous credits status-bar notices (usage bands, grant-spent, depleted). false = silence them; /usage still works
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### File-mutation verifier

Quando `display.file_mutation_verifier` é `true` (default), o Hermes anexa advisory de uma linha à resposta final do assistant sempre que uma chamada `write_file` ou `patch` falhou durante o turn e nunca foi substituída por write bem-sucedido no mesmo path. Isso captura a classe "batch de patches paralelos, metade falha silenciosamente, model resume sucesso" sem exigir `git status` manual após cada edição.

Exemplo de footer:

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

Defina `file_mutation_verifier: false` (ou `HERMES_FILE_MUTATION_VERIFIER=0`) para suprimir o footer. O verifier só dispara quando falhas reais estão pendentes no fim do turn — model que retenta patch falho e tem sucesso no mesmo turn não acionará para esse arquivo.

**Confie no verifier mais que no resumo do model.** O footer significa que os arquivos listados **não** foram modificados em disco, mesmo se a mensagem final do assistant disser que a task está done. Causas comuns:

- **Write denied** — path está na denylist de credenciais ou fora de `HERMES_WRITE_SAFE_ROOT` (veja [File write safety](./security.md#file-write-safety))
- **Patch mismatch** — `old_string` não correspondeu ao arquivo em disco
- **Syntax gate** — conteúdo candidato falhou validação JSON/YAML/TOML antes do write

Exemplo de footer quando writes são bloqueados:

```
⚠️ File-mutation verifier: 2 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • ~/.hermes/cron/jobs.json — [patch] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
  • ~/.hermes/scripts/monitor.py — [write_file] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
```

Se writes em estado Hermes (cron jobs, skills, scripts em `~/.hermes/`) estão falhando, verifique se `HERMES_WRITE_SAFE_ROOT` está definido no ambiente. Para mudanças de cron, use a ferramenta `cronjob` ou `hermes cron edit` em vez de patch direto em `jobs.json`.

### Idioma da UI para mensagens estáticas

A config `display.language` traduz um pequeno conjunto de mensagens estáticas user-facing — o prompt de approval da CLI, algumas respostas de slash command do gateway (ex.: avisos de restart-drain, "approval expired", "goal cleared"). **Não** traduz respostas do agente, linhas de log, output de ferramentas, tracebacks de erro ou descrições de slash command — esses permanecem em inglês. Se quer o agente respondendo em outro idioma, diga no prompt ou system message.

Valores suportados: `en` (default), `zh` (Simplified Chinese), `zh-hant` (Traditional Chinese), `ja` (Japanese), `de` (German), `es` (Spanish), `fr` (French), `tr` (Turkish), `uk` (Ukrainian), `af` (Afrikaans), `ko` (Korean), `it` (Italian), `ga` (Irish), `pt` (Portuguese), `ru` (Russian), `hu` (Hungarian). Valores desconhecidos fazem fallback para inglês.

Você também pode definir por sessão com env var `HERMES_LANGUAGE`, que override o valor da config.

```yaml
display:
  language: zh   # CLI approval prompts appear in Chinese
```

| Mode | What you see |
|------|-------------|
| `off` | Silencioso — apenas a resposta final |
| `new` | Indicador de ferramenta apenas quando a ferramenta muda |
| `all` | Toda tool call com preview curto (default) |
| `verbose` | Args completos, resultados e logs de debug |

Na CLI, alterne entre esses modes com `/verbose`. Para usar `/verbose` em plataformas de messaging (Telegram, Discord, Slack, etc.), defina `tool_progress_command: true` na seção `display` acima. O comando então alterna o mode e salva na config.

Tool progress requer adapter gateway que possa exibir progress updates com segurança. Plataformas sem suporte a edição de mensagem, incluindo Signal, suprimem tool-progress bubbles mesmo se `/verbose` salvar mode não-`off`.

### Footer de runtime-metadata (gateway only)

Quando `display.runtime_footer.enabled: true`, o Hermes anexa footer pequeno de runtime-context à **mensagem final** de cada turn gateway. O footer atual pode mostrar model, porcentagem de context window e working directory atual. Desligado por default; opt-in por gateway se sua equipe quer toda resposta incluir essa proveniência.

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # supported fields: model, context_pct, cwd
```

O slash command `/footer` alterna isso em runtime em qualquer sessão.

Exemplo de footer anexado a resposta Telegram/Discord/Slack:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

Apenas a **mensagem final** de um turn recebe o footer; interim updates permanecem limpos.

### Overrides de progress por plataforma

Plataformas diferentes têm necessidades de verbosidade diferentes. Use `display.platforms` para definir modes por plataforma:

```yaml
display:
  tool_progress: all          # global default
  platforms:
    signal:
      tool_progress: 'off'    # Signal cannot currently display tool-progress bubbles
    telegram:
      tool_progress: verbose  # detailed progress on Telegram
    slack:
      tool_progress: 'off'    # quiet in shared Slack workspace
```

Plataformas sem override fazem fallback para o valor global `tool_progress`. Chaves de plataforma válidas: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. A chave legacy `display.tool_progress_overrides` ainda carrega para retrocompatibilidade mas está deprecated e é migrada para `display.platforms` no primeiro load.

Signal está listado como chave de plataforma válida porque a config pode ser salva por plataforma, mas o adapter Signal atual não edita mensagens enviadas e não renderiza tool-progress bubbles. Mantenha Signal `tool_progress` em `off`; use CLI ou plataforma de messaging com edição se precisa ver cada tool call ao vivo.

`interim_assistant_messages` é gateway-only. Quando habilitado, o Hermes envia assistant updates mid-turn completos como mensagens de chat separadas. Isso é independente de `tool_progress` e não requer gateway streaming.

`show_commentary` (default `true`) controla o commentary channel de Codex Responses models — a narração de progresso polida que esses models produzem junto ao reasoning privado. Quando habilitado, cada mensagem de commentary completada é entregue como update mid-turn visível (no gateway isso também requer `interim_assistant_messages`). Defina como `false` se a narração extra incomoda: commentary então faz fallback para reasoning channel e só é mostrado quando `show_reasoning` está habilitado.

## Privacidade

```yaml
privacy:
  redact_pii: false  # Strip PII from LLM context (gateway only)
```

Quando `redact_pii` é `true`, o gateway redige informações pessoalmente identificáveis do system prompt antes de enviá-lo ao LLM em plataformas suportadas:

| Field | Treatment |
|-------|-----------|
| Phone numbers (user ID on WhatsApp/Signal) | Hashed to `user_<12-char-sha256>` |
| User IDs | Hashed to `user_<12-char-sha256>` |
| Chat IDs | Numeric portion hashed, platform prefix preserved (`telegram:<hash>`) |
| Home channel IDs | Numeric portion hashed |
| User names / usernames | **Não afetados** (escolhidos pelo usuário, publicamente visíveis) |

**Suporte por plataforma:** Redaction se aplica a WhatsApp, Signal e Telegram. Discord e Slack são excluídos porque seus sistemas de mention (`<@user_id>`) exigem o ID real no contexto LLM.

Hashes são determinísticos — o mesmo usuário sempre mapeia para o mesmo hash, então o model ainda distingue usuários em group chats. Routing e delivery usam valores originais internamente.

## Speech-to-Text (STT)

```yaml
stt:
  enabled: true                # Auto-transcribe inbound voice messages (default: true)
  echo_transcripts: true       # Post raw transcripts back to the chat as 🎙️ "..." (default: true)
  provider: "local"            # "local" | "groq" | "openai" | "mistral"
  local:
    model: "base"              # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe
  # model: "whisper-1"         # Legacy fallback key still respected
```

Defina `stt.echo_transcripts: false` quando o gateway deve transcrever voice notes para o agente mas não deve postar o transcript bruto de volta no chat (por exemplo, bots WhatsApp customer-facing).

Comportamento por provider:

- `local` usa `faster-whisper` rodando na sua máquina. Instale separadamente com `pip install faster-whisper`.
- `groq` usa endpoint Whisper-compatible da Groq e lê `GROQ_API_KEY`.
- `openai` usa speech API da OpenAI e lê `VOICE_TOOLS_OPENAI_KEY`.

Se o provider solicitado não está disponível, o Hermes faz fallback automaticamente nesta ordem: `local` → `groq` → `openai`.

Overrides de model Groq e OpenAI são driven por ambiente:

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

## Voice mode (CLI)

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  beep_enabled: true            # Play record start/stop beeps in CLI voice mode
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

Use `/voice on` na CLI para habilitar microphone mode, `record_key` para iniciar/parar gravação, e `/voice tts` para alternar respostas faladas. Veja [Voice Mode](/user-guide/features/voice-mode) para setup end-to-end e comportamento por plataforma.

## Streaming

Stream tokens para terminal ou plataformas de messaging conforme chegam, em vez de esperar a resposta completa.

### CLI streaming

```yaml
display:
  streaming: true         # Stream tokens to terminal in real-time
  show_reasoning: true    # Also stream reasoning/thinking tokens (optional)
```

Quando habilitado, respostas aparecem token a token dentro de uma streaming box. Tool calls ainda são capturadas silenciosamente. Se o provider não suporta streaming, faz fallback automaticamente para display normal.

### Gateway streaming (Telegram, Discord, Slack)

```yaml
streaming:
  enabled: true           # Enable progressive message editing
  transport: edit         # "edit" (progressive message editing) or "off"
  edit_interval: 0.3      # Seconds between message edits
  buffer_threshold: 40    # Characters before forcing an edit flush
  cursor: " ▉"            # Cursor shown during streaming
  fresh_final_after_seconds: 0    # Opt in to fresh final (Telegram) when preview is this old
```

Quando habilitado, o bot envia mensagem no primeiro token, depois edita progressivamente conforme mais tokens chegam. Plataformas sem suporte a edição de mensagem (Signal, Email, Home Assistant) são auto-detectadas na primeira tentativa — streaming é desabilitado gracefully para essa sessão sem flood de mensagens.

Para assistant updates mid-turn naturais separados sem progressive token editing, defina `display.interim_assistant_messages: true`.

**Tratamento de overflow:** Se o texto streamed excede o limite de tamanho de mensagem da plataforma (~4096 chars), a mensagem atual é finalizada e uma nova começa automaticamente.

**Fresh final (Telegram):** `editMessageText` do Telegram preserva o timestamp original da mensagem, então uma resposta streamed longa manteria o timestamp do first-token mesmo após conclusão. Defina `fresh_final_after_seconds > 0` para opt-in em entregar previews antigos como mensagens finais novas com best-effort de deletar preview. O default é `0`, que sempre finaliza respostas streamed in place e evita a breve sequência duplicate-message/delete em clients que mostram ambas operações.

:::note Defaults de streaming por plataforma
O switch master `streaming.enabled` é `false` por default — nada streama até você ligar. Uma vez habilitado, streaming é decidido **por plataforma**: Telegram vem com `display.platforms.telegram.streaming: true` (streama) e Discord com `display.platforms.discord.streaming: false` (não streama). Então após habilitar streaming, Telegram streama out of the box e Discord permanece em respostas whole-message até você mudar o toggle. Você pode ajustar esses switches por plataforma nos toggles **Channels** do dashboard ou diretamente em `~/.hermes/config.yaml`.
:::

## Isolamento de sessão em group chat

Limite quantas sessões de chat podem estar ativamente abertas entre CLI, TUI/dashboard
e messaging gateway:

```yaml
max_concurrent_sessions: null  # null/0 = unlimited; positive integer = active session cap
```

Quando o cap é atingido, o Hermes retorna mensagem de limite direta para novas sessões.
Sessões ativas existentes mantêm comportamento normal.

A chave canônica é top-level `max_concurrent_sessions`. O Hermes também aceita
`gateway.max_concurrent_sessions` como fallback, mas a chave top-level prevalece quando
ambas estão definidas.

O cap é enforced com lease file de runtime local e é best-effort: o Hermes
fails open se o registry não puder ser lido ou locked para usuários não ficarem stranded.
É intended para runtime single host/profile, não `$HERMES_HOME`
montado em múltiplas máquinas.

Controle se chats compartilhados mantêm uma conversa por room ou uma conversa por participante:

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true` é o default e configuração recomendada. Em Discord channels, Telegram groups, Slack channels e contextos compartilhados similares, cada sender recebe sua própria sessão quando a plataforma fornece user ID.
- `false` reverte ao comportamento antigo de shared-room. Pode ser útil se você quer explicitamente que o Hermes trate um channel como uma conversa colaborativa, mas também significa que usuários compartilham contexto, custos de token e estado de interrupt.
- DMs não são afetados. O Hermes ainda keya DMs por chat/DM ID como usual.
- Threads permanecem isolados do parent channel de qualquer forma; com `true`, cada participante também recebe sua própria sessão dentro da thread.

Para detalhes de comportamento e exemplos, veja [Sessions](/user-guide/sessions) e o [guia Discord](/user-guide/messaging/discord).

## Comportamento de DM não autorizado

Controle o que o Hermes faz quando usuário desconhecido envia direct message:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` é o default para plataformas DM estilo chat. O Hermes nega acesso, mas responde com código de pairing one-time em DMs.
- `ignore` descarta silenciosamente DMs não autorizados.
- Email default é `ignore` a menos que `platforms.email.unauthorized_dm_behavior: pair` esteja definido, porque inboxes podem conter mail não relacionado unread.
- Seções de plataforma override o default global, então você pode manter pairing habilitado amplamente enquanto torna uma plataforma mais silenciosa.

## Quick commands

Defina comandos customizados que executam shell commands sem invocar o LLM, ou fazem alias de um slash command para outro. Exec quick commands são zero-token e úteis de plataformas de messaging (Telegram, Discord, etc.) para checagens rápidas de server ou utility scripts.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  update:
    type: exec
    command: cd ~/.hermes/hermes-agent && git pull && pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

Uso: digite `/status`, `/disk`, `/update`, `/gpu` ou `/restart` na CLI ou qualquer plataforma de messaging. Comandos `exec` rodam localmente no host e retornam output diretamente — sem LLM call, sem tokens consumidos. Comandos `alias` reescrevem para o slash command target configurado.

- **Timeout de 30 segundos** — comandos long-running são killed com mensagem de erro
- **Prioridade** — quick commands são checados antes de skill commands, então você pode override nomes de skill
- **Autocomplete** — quick commands são resolvidos no dispatch time e não aparecem nas tabelas de autocomplete de slash command built-in
- **Type** — types suportados são `exec` e `alias`; outros types mostram erro
- **Funciona em todo lugar** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant

Atalhos de prompt string-only não são quick commands válidos. Para workflows de prompt reutilizáveis, crie skill ou alias para slash command existente.

## Human delay

Simule pacing de resposta human-like em plataformas de messaging:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # Minimum delay (custom mode)
  max_ms: 2500                 # Maximum delay (custom mode)
```

## Code execution

Configure a ferramenta `execute_code`:

```yaml
code_execution:
  mode: project                # project (default) | strict
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

**`mode`** controla working directory e interpretador Python para scripts:

- **`project`** (default) — scripts rodam no working directory da sessão com python do virtualenv/conda env ativo. Deps do projeto (`pandas`, `torch`, pacotes do projeto) e paths relativos (`.env`, `./data.csv`) resolvem naturalmente, matching o que `terminal()` vê.
- **`strict`** — scripts rodam em diretório staging temp com `sys.executable` (python do próprio Hermes). Máxima reprodutibilidade, mas deps do projeto e paths relativos não resolvem.

Environment scrubbing (remove `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL`, `*_PASSWD`, `*_AUTH`) e tool whitelist aplicam identicamente em ambos modes — trocar mode não muda a postura de segurança.

## Web search backends

As ferramentas `web_search` e `web_extract` suportam cinco backend providers. Configure o backend em `config.yaml` ou via `hermes tools`:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # Or use per-capability keys to mix providers (e.g. free search + paid extract):
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| Backend | Env Var | Search | Extract |
|---------|---------|--------|---------|
| **Firecrawl** (default) | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ |

**Seleção de backend:** Se `web.backend` não está definido, o backend é auto-detectado de API keys disponíveis. Se apenas `SEARXNG_URL` está definido, SearXNG é usado. Se apenas `EXA_API_KEY` está definido, Exa é usado. Se apenas `TAVILY_API_KEY` está definido, Tavily é usado. Se apenas `PARALLEL_API_KEY` está definido, Parallel é usado. Caso contrário Firecrawl é o default.

**SearXNG** é metasearch engine free, self-hosted e privacy-respecting que consulta 70+ search engines. Sem API key necessária — apenas defina `SEARXNG_URL` para sua instância (ex.: `http://localhost:8080`). SearXNG é search-only; `web_extract` requer extract provider separado (defina `web.extract_backend`). Veja o [guia de setup Web Search](/user-guide/features/web-search) para instruções Docker.

**Firecrawl self-hosted:** Defina `FIRECRAWL_API_URL` para apontar à sua instância. Quando URL custom está definida, API key torna-se opcional (defina `USE_DB_AUTHENTICATION=*** on the server to disable auth).

**Parallel search modes:** Defina `PARALLEL_SEARCH_MODE` para controlar comportamento de search — `fast`, `one-shot`, ou `agentic` (default: `agentic`).

**Exa:** Defina `EXA_API_KEY` em `~/.hermes/.env`. Suporta filtragem `category` (`company`, `research paper`, `news`, `people`, `personal site`, `pdf`) e filtros de domain/date.

## Browser

Configure comportamento de browser automation:

```yaml
browser:
  inactivity_timeout: 120        # Seconds before auto-closing idle sessions
  command_timeout: 30             # Timeout in seconds for browser commands (screenshot, navigate, etc.)
  record_sessions: false         # Auto-record browser sessions as WebM videos to ~/.hermes/browser_recordings/
  # Optional CDP override — when set, Hermes attaches directly to your own
  # Chromium-family browser (via /browser connect) rather than starting a headless browser.
  cdp_url: ""
  # Dialog supervisor — controls how native JS dialogs (alert / confirm / prompt)
  # are handled when a CDP backend is attached (Browserbase, local Chromium-family
  # browser via /browser connect). Ignored on Camofox and default local agent-browser mode.
  dialog_policy: must_respond    # must_respond | auto_dismiss | auto_accept
  dialog_timeout_s: 300          # Safety auto-dismiss under must_respond (seconds)
  camofox:
    managed_persistence: false   # When true, Camofox sessions persist cookies/logins across restarts
    user_id: ""                  # Optional externally managed Camofox userId
    session_key: ""              # Optional session key sent when Hermes creates a tab
    adopt_existing_tab: false    # Reuse an existing tab for this identity before creating one
```

**Dialog policies:**

- `must_respond` (default) — captura o dialog, expõe em `browser_snapshot.pending_dialogs`, e espera o agente chamar `browser_dialog(action=...)`. Após `dialog_timeout_s` segundos sem resposta, o dialog é auto-dismissed para impedir que a JS thread da página trave para sempre.
- `auto_dismiss` — captura, dismiss imediatamente. O agente ainda vê o registro do dialog em `browser_snapshot.recent_dialogs` com `closed_by="auto_policy"` depois.
- `auto_accept` — captura, accept imediatamente. Útil para páginas com `beforeunload` prompts agressivos.

Veja a [página de feature browser](./features/browser.md#browser_dialog) para workflow completo de dialog.

O toolset browser suporta múltiplos providers. Veja a [página de feature Browser](/user-guide/features/browser) para detalhes de Browserbase, Browser Use e setup local Chromium-family CDP.

## Timezone

Override o timezone local do server com string IANA timezone. Afeta timestamps em logs, cron scheduling e injeção de hora no system prompt.

```yaml
timezone: "America/New_York"   # IANA timezone (default: "" = server-local time)
```

Valores suportados: qualquer identificador IANA timezone (ex. `America/New_York`, `Europe/London`, `Asia/Kolkata`, `UTC`). Deixe vazio ou omita para hora local do server.

## Discord

Configure comportamento específico do Discord para o messaging gateway:

```yaml
discord:
  require_mention: true          # Require @mention to respond in server channels
  free_response_channels: ""     # Comma-separated channel IDs where bot responds without @mention
  auto_thread: true              # Auto-create threads on @mention in channels
```

- `require_mention` — quando `true` (default), o bot só responde em server channels quando mencionado com `@BotName`. DMs sempre funcionam sem mention.
- `free_response_channels` — lista separada por vírgula de channel IDs onde o bot responde a toda mensagem sem exigir mention.
- `auto_thread` — quando `true` (default), mentions em channels criam thread automaticamente para a conversa, mantendo channels limpos (similar a threading Slack).

## Segurança

Pre-execution security scanning e secret redaction:

```yaml
security:
  redact_secrets: true           # Redact API key patterns in tool output and logs (on by default)
  tirith_enabled: true           # Enable Tirith security scanning for terminal commands
  tirith_path: "tirith"          # Path to tirith binary (default: "tirith" in $PATH)
  tirith_timeout: 5              # Seconds to wait for tirith scan before timing out
  tirith_fail_open: true         # Allow command execution if tirith is unavailable
  website_blocklist:             # See Website Blocklist section below
    enabled: false
    domains: []
    shared_files: []
```

- `redact_secrets` — quando `true`, detecta e redige automaticamente padrões que parecem API keys, tokens e senhas em tool output antes de entrar no contexto da conversa e logs. **Ligado por default**. Defina como `false` explicitamente apenas quando precisa de strings raw tipo credential para debug ou desenvolvimento do redactor.
- `tirith_enabled` — quando `true`, comandos de terminal são escaneados por [Tirith](https://github.com/sheeki03/tirith) antes da execução para detectar operações potencialmente perigosas.
- `tirith_path` — caminho para binary tirith. Defina se tirith está instalado em local não padrão.
- `tirith_timeout` — segundos máximos para esperar scan tirith. Comandos prosseguem se o scan expirar.
- `tirith_fail_open` — quando `true` (default), comandos são permitidos se tirith está indisponível ou falha. Defina como `false` para bloquear comandos quando tirith não pode verificá-los.

## Website blocklist

Bloqueie domínios específicos de serem acessados pelas ferramentas web e browser do agente:

```yaml
security:
  website_blocklist:
    enabled: false               # Enable URL blocking (default: false)
    domains:                     # List of blocked domain patterns
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:                # Load additional rules from external files
      - "/etc/hermes/blocked-sites.txt"
```

Quando habilitado, qualquer URL matching um padrão de domínio bloqueado é rejeitada antes da ferramenta web ou browser executar. Isso se aplica a `web_search`, `web_extract`, `browser_navigate`, e qualquer ferramenta que acessa URLs.

Regras de domínio suportam:
- Domínios exatos: `admin.example.com`
- Wildcard subdomains: `*.internal.company.com` (bloqueia todos subdomains)
- TLD wildcards: `*.local`

Arquivos compartilhados contêm uma regra de domínio por linha (linhas em branco e comentários `#` são ignorados). Arquivos missing ou unreadable registram aviso mas não desabilitam outras web tools.

A policy é cacheada por 30 segundos, então mudanças de config entram em vigor rapidamente sem restart.

## Smart approvals

Controle como o Hermes lida com comandos potencialmente perigosos:

```yaml
approvals:
  mode: smart   # smart | manual | off
```

| Mode | Behavior |
|------|----------|
| `smart` (default) | Usa LLM auxiliary para avaliar se comando flagged é realmente perigoso. Comandos low-risk são auto-approved apenas para esse comando. Comandos genuinamente risky são denied; decisões incertas escalam para o usuário. |
| `manual` | Pede confirmação do usuário antes de executar qualquer comando flagged. Na CLI, mostra dialog de approval interativo. Em messaging, enfileira pending approval request. |
| `off` | Pula todas approval checks. Equivalente a `HERMES_YOLO_MODE=true`. **Use com cautela.** |

Smart mode é particularmente útil para reduzir approval fatigue — deixa o agente trabalhar mais autonomamente em operações seguras enquanto ainda captura comandos genuinamente destrutivos.

:::warning
Definir `approvals.mode: off` desabilita todas safety checks para comandos de terminal. Use apenas em ambientes confiáveis e sandboxed.
:::

### Deny rules

`approvals.deny` é lista de glob patterns que bloqueiam matching terminal commands incondicionalmente — mesmo sob `--yolo`, `/yolo`, ou `mode: off`. É o counterpart editável pelo usuário da blocklist hardline built-in:

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

Patterns são fnmatch globs case-insensitive e devem ser quoted em YAML (leading `*` bare é parse error). Veja [Security — User-Defined Deny Rules](/user-guide/security#user-defined-deny-rules-approvalsdeny) para detalhes.

## Checkpoints

Filesystem snapshots automáticos antes de operações destrutivas de arquivo. Veja [Checkpoints & Rollback](/user-guide/checkpoints-and-rollback) para detalhes.

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes chat --checkpoints). Default: false (opt-in).
  max_snapshots: 20              # Max checkpoints to keep per directory (default: 20)
```


## Delegation

Configure comportamento de subagent para a ferramenta delegate:

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  # api_mode: ""                            # Wire protocol for base_url: "chat_completions", "codex_responses", or "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix → anthropic_messages). Set explicitly for non-standard endpoints the heuristic can't detect.
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
```

**Override provider:model de subagent:** Por default, subagents herdam provider e model do parent agent. Defina `delegation.provider` e `delegation.model` para rotear subagents para par provider:model diferente — ex.: usar model barato/rápido para subtasks narrow-scoped enquanto seu primary agent roda reasoning model caro.

**Override de endpoint direto:** Se quer o caminho óbvio de custom endpoint, defina `delegation.base_url`, `delegation.api_key` e `delegation.model`. Isso envia subagents diretamente para esse endpoint OpenAI-compatible e prevalece sobre `delegation.provider`. Se `delegation.api_key` for omitida, o Hermes faz fallback apenas para `OPENAI_API_KEY`.

**Wire protocol (`api_mode`):** O Hermes auto-detecta wire protocol de `delegation.base_url` (ex.: paths terminando em `/anthropic` → `anthropic_messages`; hostnames Codex / native Anthropic / Kimi-coding mantêm detecção existente). Para endpoints que a heurística não classifica — por exemplo Azure AI Foundry, MiniMax, Zhipu GLM, ou proxies LiteLLM fronting backend Anthropic-shaped — defina `delegation.api_mode` explicitamente para um de `chat_completions`, `codex_responses`, ou `anthropic_messages`. Deixe vazio (default) para manter auto-detection.

O delegation provider usa a mesma resolução de credenciais que startup CLI/gateway. Todos providers configurados são suportados: `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. Quando provider está definido, o sistema resolve automaticamente base URL, API key e API mode corretos — sem wiring manual de credenciais.

**Precedência:** `delegation.base_url` na config → `delegation.provider` na config → parent provider (herdado). `delegation.model` na config → parent model (herdado). Definir apenas `model` sem `provider` muda apenas o nome do model mantendo credenciais do parent (útil para trocar models dentro do mesmo provider como OpenRouter).

**Width e depth:** `max_concurrent_children` limita quantos subagents rodam em paralelo por batch (default `3`, floor 1, sem ceiling). Também pode ser definido via env var `DELEGATION_MAX_CONCURRENT_CHILDREN`. Quando o model submete array `tasks` maior que o cap, `delegate_task` retorna tool error explicando o limite em vez de truncar silenciosamente. `max_spawn_depth` controla profundidade da delegation tree (clamped 1-3). No default `1`, delegation é flat: children não podem spawn grandchildren, e passar `role="orchestrator"` degrada silenciosamente para `leaf`. Suba para `2` para orchestrator children spawnarem leaf grandchildren; `3` para árvores de três níveis. O agente opta orchestration por call via `role="orchestrator"`; `orchestrator_enabled: false` força todo child de volta para leaf independentemente. Custo escala multiplicativamente — em `max_spawn_depth: 3` com `max_concurrent_children: 3`, a árvore pode atingir 3×3×3 = 27 leaf agents concurrent. Veja [Subagent Delegation → Depth Limit and Nested Orchestration](features/delegation.md#depth-limit-and-nested-orchestration) para padrões de uso.

## Clarify

Configure comportamento do clarification prompt:

```yaml
clarify:
  timeout: 120                 # Seconds to wait for user clarification response
```

## Context files (SOUL.md, AGENTS.md)

O Hermes usa dois escopos de contexto diferentes:

| File | Purpose | Scope |
|------|---------|-------|
| `SOUL.md` | **Identidade primária do agente** — define quem o agente é (slot #1 no system prompt) | `~/.hermes/SOUL.md` or `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | Instruções específicas do projeto (maior prioridade) | Walks to git root |
| `AGENTS.md` | Instruções específicas do projeto, convenções de coding | Recursive directory walk |
| `CLAUDE.md` | Arquivos de contexto Claude Code (também detectados) | Working directory only |
| `.cursorrules` | Regras Cursor IDE (também detectadas) | Working directory only |
| `.cursor/rules/*.mdc` | Arquivos de regra Cursor (também detectados) | Working directory only |

- **SOUL.md** é a identidade primária do agente. Ocupa slot #1 no system prompt, substituindo completamente a identidade default built-in. Edite para customizar totalmente quem o agente é.
- Se SOUL.md está missing, vazio ou não pode ser carregado, o Hermes faz fallback para identidade default built-in.
- **Context files de projeto usam sistema de prioridade** — apenas UM type é carregado (first match wins): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. SOUL.md é sempre carregado independentemente.
- **AGENTS.md** é hierárquico: se subdiretórios também têm AGENTS.md, todos são combinados.
- O Hermes seed automaticamente um `SOUL.md` default se um ainda não existir.
- Todos context files carregados são limitados a `context_file_max_chars` caracteres (default 20.000) com truncamento inteligente.

Veja também:
- [Personality & SOUL.md](/user-guide/features/personality)
- [Context Files](/user-guide/features/context-files)

## Working directory

| Context | Default |
|---------|---------|
| **CLI (`hermes`)** | Diretório atual onde você executa o comando |
| **Messaging gateway** | `terminal.cwd` de `~/.hermes/config.yaml`; se unset, home directory `~` |
| **Docker / Singularity / Modal / SSH** | Home directory do usuário dentro do container ou máquina remota |

Override do working directory:
```yaml
# In ~/.hermes/config.yaml:
terminal:
  cwd: /home/myuser/projects
```

`MESSAGING_CWD` e entradas diretas `TERMINAL_CWD` em `~/.hermes/.env` são fallbacks de compatibilidade legacy. Novas configurações devem usar `terminal.cwd`.

## Network

Workarounds de conectividade para HTTP outbound:

```yaml
network:
  force_ipv4: false   # Force IPv4 for outbound connections (default: false)
```

`force_ipv4` — em servers com IPv6 quebrado ou unreachable, Python resolve AAAA records primeiro e pode hangar pelo TCP timeout completo antes de fallback para IPv4. Defina como `true` para pular IPv6 inteiramente e conectar via IPv4 diretamente.

## Onboarding

Dicas de onboarding first-touch e oferta estruturada de profile-build:

```yaml
onboarding:
  profile_build: "ask"   # "ask" (default) | "off"
  seen: {}               # internal latch — leave empty
```

- `profile_build` — controla o path profile-build oferecido na very first gateway message ever. `"ask"` (default) oferece construir user profile; a oferta é **opt-in e consent-gated** — o agente pergunta antes de qualquer lookup e nunca lê contas conectadas silenciosamente. `"off"` mostra intro plain apenas. A oferta dispara no máximo uma vez.
- `seen` — estado interno. O Hermes latch cada hint mostrada aqui para nunca disparar de novo; a oferta profile-build também é registrada aqui uma vez mostrada. Não edite manualmente — wipe a seção `onboarding` inteira se quer re-ver todas hints.

## Dashboard

Configuração para o [web dashboard](/user-guide/features/web-dashboard) — tema visual, URL pública e auth providers. Os auth providers (OAuth, basic password, drain) estão documentados em detalhe na página web-dashboard; esta é a forma em `config.yaml`.

```yaml
dashboard:
  theme: "default"            # "default" | "midnight" | "ember" | "mono" | "cyberpunk" | "rose"
  show_token_analytics: false # Re-enable the (local-estimate-only) token/cost analytics surfaces
  public_url: ""              # Full public authority for OAuth redirect_uri (env: HERMES_DASHBOARD_PUBLIC_URL)
  oauth:                      # Portal OAuth gate (engaged with --host and not --insecure)
    client_id: ""             # agent:{instance_id} — Portal provisions this
    portal_url: ""            # blank → plugin default (production Portal)
  basic_auth:                 # Self-hosted username/password gate (dashboard_auth/basic plugin)
    username: ""              # blank → plugin no-op
    password_hash: ""         # scrypt$... (preferred — no plaintext at rest)
    password: ""              # plaintext fallback (hashed in-memory at load)
    secret: ""                # token-signing key; blank → random per-process
    session_ttl_seconds: 0    # 0 → plugin default (12h)
  drain_auth:                 # Drain-control service-credential gate (dashboard_auth/drain plugin)
    scope: "drain"            # capability label on the verified principal
    min_secret_chars: 43      # entropy bar (url-safe-b64 chars; 43 ≈ 256 bits)
```

- `theme` — tema visual do dashboard.
- `show_token_analytics` — off por default. A página Analytics e figuras token/cost são **estimativa local lower-bound** (excluem auxiliary calls, retries, fallbacks e cache writes), então podem ler muito abaixo da fatura do provider. Defina `true` apenas se entende que não são billing.
- `public_url` — quando definida, esta é a authority completa (scheme + host + optional path prefix) de onde o OAuth `redirect_uri` é construído. Defina para deploys atrás de reverse proxies que não forwardam headers `X-Forwarded-*` de forma confiável. Deixe vazio para usar reconstrução proxy-header.
- `oauth` / `basic_auth` / `drain_auth` — config de auth provider lida pelos plugins dashboard-auth bundled. O drain secret em si **não** é definido aqui; é provisionado via env var `HERMES_DASHBOARD_DRAIN_SECRET`. Veja [Web Dashboard](/user-guide/features/web-dashboard) para setup completo de auth.
