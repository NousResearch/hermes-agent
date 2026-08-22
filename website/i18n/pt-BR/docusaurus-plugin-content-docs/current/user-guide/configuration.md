---
sidebar_position: 2
title: "Configuração"
description: "Configure o Hermes Agent — config.yaml, provedores, modelos, chaves de API e mais"
---

# Configuração

Todas as configurações ficam no diretório `~/.hermes/` para fácil acesso.

:::tip Caminho mais fácil para um `config.yaml` funcional
Execute `hermes setup --portal` — um OAuth obtém um provedor de modelo e as quatro ferramentas do Tool Gateway sem editar YAML manualmente. Assinantes do Portal também ganham 10% de desconto em provedores cobrados por token. Veja [Nous Portal](/integrations/nous-portal).
:::

## Estrutura de diretórios {#directory-structure}

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

## Gerenciando a configuração {#managing-configuration}

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
O comando `hermes config set` encaminha valores automaticamente ao arquivo certo — chaves de API vão para `.env`, todo o resto para `config.yaml`.
:::

## Precedência de configuração {#configuration-precedence}

As configurações são resolvidas nesta ordem (maior prioridade primeiro):

1. **Argumentos da CLI** — ex.: `hermes chat --model anthropic/claude-sonnet-4` (sobrescrita por invocação)
2. **`~/.hermes/config.yaml`** — arquivo principal de configuração para todas as configurações não secretas
3. **`~/.hermes/.env`** — fallback para vars de ambiente; **obrigatório** para segredos (chaves de API, tokens, senhas)
4. **Padrões embutidos** — padrões seguros hardcoded quando nada mais está definido

:::info Regra prática
Segredos (chaves de API, tokens de bot, senhas) vão em `.env`. Todo o resto (modelo, backend de terminal, configurações de compressão, limites de memória, toolsets) vai em `config.yaml`. Quando ambos estão definidos, `config.yaml` vence para configurações não secretas.
:::

:::tip Implantações organizacionais
Um administrador pode fixar valores específicos de config e segredos que um usuário padrão
não pode sobrescrever, via um diretório gerenciado em nível de sistema. Veja
[Managed Scope](/user-guide/managed-scope).
:::

## Limites de runtime {#runtime-limits}

Superfícies de servidor de longa duração do Hermes (incluindo o gateway e
`hermes serve --isolated`) aplicam o limite soft configurado de `RLIMIT_NOFILE`
durante o startup, quando o sistema operacional o suporta:

```yaml
runtime:
  nofile_soft_limit: 4096
```

O padrão é `4096`. O Hermes limita o alvo ao hard limit do sistema operacional e nunca reduz um processo que já tenha um soft limit maior. Defina o valor como `0`, `false` ou `null` para desabilitar o ajuste. No Windows e em sandboxes
onde o limite não pode ser alterado, o startup continua sem mudar o
limite.

## Substituição de variáveis de ambiente {#environment-variable-substitution}

Você pode referenciar variáveis de ambiente em `config.yaml` com a sintaxe `${VAR_NAME}`:

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

Múltiplas referências em um único valor funcionam: `url: "${HOST}:${PORT}"`. Se uma variável referenciada não estiver definida, o placeholder permanece literal (`${UNDEFINED_VAR}` fica como está) e um aviso é registrado. `$VAR` simples não é expandido.

A sintaxe SecretRef estilo Cursor também é aceita: `${env:VAR_NAME}` resolve exatamente como `${VAR_NAME}` (o prefixo `env:` é removido), então trechos MCP ou de provedor copiados de configs Cursor / Claude funcionam inalterados tanto em `config.yaml` quanto no bloco `mcp_servers`. Outras fontes SecretRef (`${file:...}`, `${vault:...}`, `${bitwarden:...}`) **não** são resolvidas inline — backends de segredos externos injetam seus valores no ambiente na inicialização via o bloco `secrets:`, então referencie-os como `${env:NAME}`; prefixos desconhecidos avisam uma vez e permanecem literais.

Para configuração de provedores de IA (OpenRouter, Anthropic, Copilot, endpoints customizados, LLMs self-hosted, modelos de fallback, etc.), veja [AI Providers](/integrations/providers).

### Timeouts de provedor {#provider-timeouts}

Você pode definir `providers.<id>.request_timeout_seconds` para um timeout de requisição em todo o provedor, mais `providers.<id>.models.<model>.timeout_seconds` para sobrescrita por modelo. Aplica-se ao cliente de turno principal em todo transporte (OpenAI-wire, Anthropic nativo, compatível com Anthropic), cadeia de fallback, reconstruções após rotação de credenciais e (para OpenAI-wire) o kwarg de timeout por requisição — então o valor configurado vence a env legada `HERMES_API_TIMEOUT`.

Você também pode definir `providers.<id>.stale_timeout_seconds` para o detector de chamadas stale não-streaming, mais `providers.<id>.models.<model>.stale_timeout_seconds` para sobrescrita por modelo. Isso vence a env legada `HERMES_API_CALL_STALE_TIMEOUT`.

Deixar esses valores sem definir mantém os padrões legados (`HERMES_API_TIMEOUT=1800`s, `HERMES_API_CALL_STALE_TIMEOUT=90`s, Anthropic nativo 900s). O detector stale não-streaming é auto-desabilitado para endpoints locais quando deixado implícito e pode escalar para cima em contextos muito grandes. Ainda não conectado para AWS Bedrock (ambos os caminhos `bedrock_converse` e SDK AnthropicBedrock usam boto3 com sua própria configuração de timeout). Veja o exemplo comentado em [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example).

## Comportamento de atualização {#update-behavior}

Configurações de `hermes update` ficam em `updates` em `config.yaml`:

```yaml
updates:
  pre_update_backup: quick       # quick (state snapshot, default) | full (snapshot + HERMES_HOME zip) | off
  backup_keep: 5                 # Keep this many full pre-update backup zips
  non_interactive_local_changes: stash  # stash | discard
  auto_switch_parked_branch: true       # auto-switch a clean, fully merged parked branch back to main
```

`pre_update_backup` é o único botão de segurança pré-atualização: `quick` (padrão) faz snapshot de arquivos críticos de estado (dados de pairing, jobs cron, config, auth; arquivos acima de 1 GiB são ignorados) em `state-snapshots/`; `full` adicionalmente compacta todo o `HERMES_HOME` em `backups/` e pode levar minutos em homes grandes; `off` desativa ambos. Booleanos legados são honrados (`true` → `full`, `false` → `off`).

Para instalações git, o Hermes auto-stasha arquivos rastreados sujos e não rastreados antes de fazer checkout da branch de atualização ou pull. Atualizações interativas no terminal pedem confirmação antes de restaurar esse stash. Atualizações não interativas (desktop/app de chat, gateway ou `--yes`) usam `updates.non_interactive_local_changes`: `stash` restaura edições locais de código-fonte após pull bem-sucedido, enquanto `discard` descarta o stash criado pela atualização após pull bem-sucedido. Use `discard` apenas em instalações gerenciadas onde edições locais de código-fonte nunca devem persistir.

Antes desse passo de stash, o Hermes também restaura diffs rastreados de `package-lock.json` deixados por churn de npm install/build. Faça commit ou stash manual de edições intencionais de lockfile antes de atualizar.

## Configuração do backend de terminal {#terminal-backend-configuration}

O Hermes suporta sete backends de terminal. Cada um determina onde os comandos shell do agente realmente executam — sua máquina local, um container Docker, um servidor remoto via SSH, um sandbox Modal na nuvem (direto ou via gateway gerenciado Nous), um workspace Daytona, um Vercel Sandbox ou um container Singularity/Apptainer.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | vercel_sandbox | singularity
  cwd: "."          # Gateway/cron working directory (CLI always uses launch dir)
  font_family: ""   # Desktop terminal font; e.g. "MesloLGS NF"
  timeout: 180      # Per-command timeout in seconds
  home_mode: auto   # auto | real | profile — subprocess HOME policy
  env_passthrough: []  # Env var names to forward to sandboxed execution (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Container image for Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Container image for Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Container image for Daytona backend
```

`terminal.font_family` controla o terminal embutido no Hermes Desktop. Aceita um nome de família instalada localmente (por exemplo, `MesloLGS NF`) ou uma pilha CSS de fontes. O Hermes anexa sua pilha JetBrains Mono incluída como fallback, e um valor vazio mantém o padrão. Você pode editar a mesma config com escopo de perfil em **Settings → Appearance → Terminal Font**; não é necessário download de Google Fonts nem permissão de fonte do sistema.

Para sandboxes na nuvem como Modal, Daytona e Vercel Sandbox, `container_persistent: true` significa que o Hermes tentará preservar o estado do filesystem entre recriações de sandbox. Não promete que o mesmo sandbox vivo, espaço de PID ou processos em segundo plano ainda estarão rodando depois.

### Visão geral dos backends {#backend-overview}

| Backend | Onde os comandos rodam | Isolamento | Melhor para |
|---------|-------------------|-----------|----------|
| **local** | Sua máquina diretamente | Nenhum | Desenvolvimento, uso pessoal |
| **docker** | Container Docker persistente único (compartilhado entre sessão, `/new`, subagentes) | Total (namespaces, cap-drop) | Sandbox seguro, CI/CD |
| **ssh** | Servidor remoto via SSH | Limite de rede | Dev remoto, hardware potente |
| **modal** | Sandbox Modal na nuvem | Total (VM na nuvem) | Compute efêmero na nuvem, evals |
| **daytona** | Workspace Daytona | Total (container na nuvem) | Ambientes de dev gerenciados na nuvem |
| **vercel_sandbox** | Vercel Sandbox | Total (microVM na nuvem) | Execução na nuvem com persistência de filesystem via snapshot |
| **singularity** | Container Singularity/Apptainer | Namespaces (--containall) | Clusters HPC, máquinas compartilhadas |

### Backend local {#local-backend}

O padrão. Comandos rodam diretamente na sua máquina sem isolamento. Nenhuma configuração especial necessária.

```yaml
terminal:
  backend: local
```

Por padrão, subprocessos de ferramentas locais mantêm o `HOME` real do usuário do SO. Isso permite que
CLIs externos como `git`, `ssh`, `gh`, `az`, `npm`, Claude Code e Codex
encontrem as credenciais e config que já usam no shell normal. O estado do Hermes
continua com escopo de perfil via `HERMES_HOME`; `HOME` não é como perfis
selecionam config, memória, sessões ou skills.

O Hermes **não** altera seu `HOME` em todo o sistema, seus arquivos de startup de shell nem
o home da conta do sistema operacional. Esta configuração controla apenas o ambiente
passado a subprocessos que o Hermes lança via ferramentas como `terminal`,
processos de terminal em segundo plano, `execute_code` e processos auxiliares ACP.

#### `terminal.home_mode` {#terminalhome_mode}

| Modo | Instalações no host | Containers | Tradeoff |
|---|---|---|---|
| `auto` | Mantém o `HOME` real do usuário do SO | Usa `{HERMES_HOME}/home` | Padrão recomendado. CLIs no host continuam funcionando; estado do container persiste. |
| `real` | Força o `HOME` real do usuário do SO | Força o `HOME` real do usuário do SO se visível | Útil se um processo pai iniciou acidentalmente com `HOME` apontando para um home de perfil. |
| `profile` | Usa `{HERMES_HOME}/home` quando existir | Usa `{HERMES_HOME}/home` quando existir | Isolamento estrito de config de CLI por perfil, mas `~/.ssh`, `~/.gitconfig`, `~/.azure`, `~/.config/gh`, auth Claude/Codex, estado npm, etc. normais não serão visíveis a menos que você inicialize ou linke dentro do home do perfil. |

A desvantagem do padrão é que perfis no host compartilham as mesmas credenciais/config de CLI em nível de usuário em `~`. Se precisar de um perfil com identidade git separada, chaves SSH, login GitHub CLI, config npm ou login de CLI na nuvem, use `home_mode: profile` e inicialize essas ferramentas dentro desse home
de perfil deliberadamente.

Se quiser isolamento estrito de config de ferramentas por perfil intencionalmente, defina:

```yaml
terminal:
  home_mode: profile
```

Nesse modo subprocessos de ferramentas usam `{HERMES_HOME}/home` como `HOME`. O Hermes também
define `HERMES_REAL_HOME` para scripts ainda localizarem o home real do usuário quando
precisarem. Backends de container continuam usando `{HERMES_HOME}/home` no modo `auto`
porque esse diretório vive no volume persistente de dados do Hermes.

Scripts que precisam distinguir estado de perfil do home real do usuário devem
preferir `HERMES_HOME` para dados Hermes e `HERMES_REAL_HOME` para o home da conta:

```python
from pathlib import Path
import os

hermes_home = Path(os.environ["HERMES_HOME"])
real_home = Path(os.environ.get("HERMES_REAL_HOME", os.environ["HOME"]))
```

:::warning
O agente tem o mesmo acesso ao filesystem que sua conta de usuário. Use `hermes tools` para desabilitar ferramentas que não quer, ou mude para Docker para sandbox.
:::

### Backend Docker {#docker-backend}

Executa comandos dentro de um container Docker com hardening de segurança (todas as capabilities removidas, sem escalação de privilégio, limites de PID).

**Container persistente único, compartilhado entre processos Hermes.** O Hermes inicia UM container de longa duração no primeiro uso e roteia toda chamada de terminal, arquivo e `execute_code` via `docker exec` nesse mesmo container — entre sessões, `/new`, `/reset` e subagentes `delegate_task`. Mudanças de diretório de trabalho, pacotes instalados, arquivos em `/workspace` e **processos em segundo plano** persistem de uma chamada de ferramenta para a próxima, e de um processo Hermes para o outro. Quando você fecha uma sessão TUI, executa `/quit` ou inicia uma nova invocação `hermes`, o container continua rodando e o próximo processo Hermes o reutiliza via lookup rotulado. Veja **Ciclo de vida do container** abaixo para as regras exatas de teardown.

**Modo de isolamento por sessão (`container_persistent: false`).** Definir `container_persistent: false` no backend Docker troca para um container **por sessão**: cada chat (sessão do app desktop, conversa do gateway, sessão TUI) recebe sua própria sandbox nova, criada na primeira chamada de terminal/arquivo e removida quando a sessão fecha ou fica idle além de `lifetime_seconds`. Nada persiste entre sessões — nenhum estado de filesystem, nenhum mount, nenhum processo em segundo plano. Com `docker_mount_cwd_to_workspace: true`, apenas o workspace **anexado a essa sessão** é montado em `/workspace`; uma sessão nova sem diretório anexado recebe um workspace vazio em vez de herdar o mount da sessão anterior. Subagentes `delegate_task` ainda compartilham o container da sessão pai. Use este modo quando a sandbox é uma fronteira de segurança entre conversas; mantenha o padrão `true` quando quiser o container compartilhado de longa duração descrito acima.

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
  container_persistent: true       # true = persist /workspace + /root, shared container; false = fresh container per session (see below)

  # Cross-process container reuse (defaults match the "one long-lived
  # container shared across sessions" contract — see Container lifecycle).
  docker_persist_across_processes: true   # Reuse container across Hermes restarts
  docker_orphan_reaper: true              # Sweep abandoned Exited containers at startup

  # Cross-backend lifecycle settings (apply to docker as well)
  timeout: 180                     # Per-command timeout in seconds
  lifetime_seconds: 300            # Idle-reaper window; also feeds 2× orphan-reaper threshold
```

**`docker_env`** vs **`docker_forward_env`**: o primeiro injeta pares `KEY=value` literais que você especifica na config (os valores ficam no seu `config.yaml` ou são passados como dict JSON via `TERMINAL_DOCKER_ENV='{"DEBUG":"1"}'`). O segundo encaminha valores do seu shell ou `~/.hermes/.env`, então o segredo real nunca aparece no arquivo de config. Use `docker_forward_env` para tokens e `docker_env` para knobs estáticos que o container precisa.

**`terminal.docker_extra_args`** (também sobrescrevível via `TERMINAL_DOCKER_EXTRA_ARGS='["--gpus=all"]'`) permite passar flags arbitrárias de `docker run` que o Hermes não expõe como chaves de primeira classe — `--gpus`, `--network`, `--add-host`, sobrescritas alternativas de `--security-opt`, etc. Cada entrada deve ser uma string; a lista é anexada por último à invocação `docker run` montada para poder sobrescrever os padrões do Hermes se necessário. Use com moderação — flags que conflitam com o hardening de sandbox (remoção de capabilities, `--user`, bind mount do workspace) enfraquecerão silenciosamente o isolamento.

**`terminal.docker_network`** (padrão `true`; env: `TERMINAL_DOCKER_NETWORK`) — defina `false` para rodar o container sandbox com `--network=none`, cortando todo egress de rede de comandos do agente. Isso se aplica ao container de execução usado por `terminal`, `execute_code` e ferramentas de arquivo. Como containers persistem entre processos Hermes, mudar para `false` enquanto um container antigo com rede existe removerá esse container e iniciará um novo air-gapped (um aviso é registrado); processos em segundo plano rodando dentro dele são perdidos. Prefira esta chave a passar `--network=none` via `docker_extra_args`.

**Requisitos:** Docker Desktop ou Docker Engine instalado e rodando. O Hermes sonda `$PATH` mais locais comuns de instalação macOS (`/usr/local/bin/docker`, `/opt/homebrew/bin/docker`, app bundle Docker Desktop). Podman é suportado out of the box: defina `HERMES_DOCKER_BINARY=podman` (ou o caminho completo) para forçá-lo quando ambos estão instalados.

#### Ciclo de vida do container {#container-lifecycle}

Todo container gerenciado pelo Hermes é rotulado com três labels para processos subsequentes (e o orphan reaper) identificá-lo:

- `hermes-agent=1` — marca como gerenciado pelo Hermes
- `hermes-task-id=<sanitized task_id>` — chaveia a sonda de reutilização por task
- `hermes-profile=<sanitized profile name>` — escopa reutilização e reaping ao perfil Hermes ativo

Na inicialização, o Hermes executa `docker ps --filter label=hermes-task-id=<id> --filter label=hermes-profile=<profile>` e **anexa ao container existente** quando encontra um. Se o container está `exited` (ex.: após reinício do daemon Docker), é `docker start`'d e reutilizado — estado do filesystem e pacotes instalados sobrevivem, mas processos em segundo plano in-container não.

Quando um processo Hermes termina — `/quit`, fechar sessão TUI, shutdown do gateway, até SIGKILL — o caminho de cleanup é **no-op para o container no modo padrão**. O container continua rodando. O próximo processo Hermes anexa em milissegundos via sonda de label. Esse é o comportamento que o contrato "um container de longa duração compartilhado entre sessões" exige: é a única forma de processos em segundo plano (watchers npm, dev servers, pytest longo) sobreviverem entre sessões.

**O container só é destruído (stopped e `docker rm -f`'d) nestes casos:**

| Gatilho | Quando dispara |
|---|---|
| `docker_persist_across_processes: false` | Isolamento explícito por processo. Todo `cleanup()` faz `stop` + `rm -f`. Corresponde ao comportamento pré-issue-#20561. |
| Idle reaper (`lifetime_seconds`, padrão 300s) | Apenas quando a env é `persist_across_processes=false`. Envs em modo persist são no-op'd; container sobrevive ao sweep idle. |
| Orphan reaper na próxima inicialização | Varre containers hermes-labeled **Exited** mais antigos que `2 × lifetime_seconds` (padrão 600s = 10 min), escopados ao perfil atual. **Containers Running nunca são tocados** — segurança entre processos irmãos. Defina `docker_orphan_reaper: false` para desabilitar. |
| Ação direta do usuário | `docker rm -f`, `docker system prune`, reinício Docker Desktop. Não definimos `--restart=always`, então reboot do host deixa o container `Exited` (sua camada CoW sobrevive e é reutilizada na próxima inicialização, mas processos bg se vão). |

Casos extremos que valem saber:

- **OOM kill do PID 1 in-container** transiciona o container para `Exited`. Próxima reutilização fará `docker start`; estado do filesystem sobrevive, processos bg não.
- **Trocar perfis** isola containers entre si — um container rotulado `hermes-profile=work` é invisível a um processo Hermes rodando sob `hermes-profile=research`. O orphan reaper também é escopado por perfil, então containers cross-profile não são reaped acidentalmente, mas também não são limpos automaticamente até você iniciar o Hermes novamente sob o perfil original.

Subagentes paralelos gerados via `delegate_task(tasks=[...])` compartilham este container — `cd` concorrente, mutações de env e escritas no mesmo caminho colidirão. Se um subagente precisa de sandbox isolado, deve registrar sobrescrita de imagem por task via `register_task_env_overrides()`, que ambientes RL e benchmark (TerminalBench2, HermesSweEnv, etc.) fazem automaticamente para suas imagens Docker por task.

**Hardening de segurança:**
- `--cap-drop ALL` com apenas `DAC_OVERRIDE`, `CHOWN`, `FOWNER` readicionados
- `--security-opt no-new-privileges`
- `--pids-limit 256`
- tmpfs com limite de tamanho para `/tmp` (512MB), `/var/tmp` (256MB), `/run` (64MB)

**Encaminhamento de credenciais:** Vars de ambiente listadas em `docker_forward_env` são resolvidas do ambiente do shell primeiro, depois `~/.hermes/.env`. Skills também podem declarar `required_environment_variables` que são mescladas automaticamente.

#### Sobrescritas de variáveis de ambiente {#environment-variable-overrides}

Toda chave sob `terminal:` tem uma sobrescrita de env var da forma `TERMINAL_<KEY_UPPERCASE>`. As mais úteis para o backend Docker:

| Env var | Mapeia para | Notas |
|---|---|---|
| `TERMINAL_DOCKER_IMAGE` | `docker_image` | Imagem base |
| `TERMINAL_DOCKER_FORWARD_ENV` | `docker_forward_env` | Array JSON: `'["GITHUB_TOKEN","OPENAI_API_KEY"]'` |
| `TERMINAL_DOCKER_ENV` | `docker_env` | Dict JSON: `'{"DEBUG":"1"}'` |
| `TERMINAL_DOCKER_VOLUMES` | `docker_volumes` | Array JSON de strings `"host:container[:ro]"` |
| `TERMINAL_DOCKER_EXTRA_ARGS` | `docker_extra_args` | Array JSON |
| `TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE` | `docker_mount_cwd_to_workspace` | `true` / `false` |
| `TERMINAL_DOCKER_RUN_AS_HOST_USER` | `docker_run_as_host_user` | `true` / `false` |
| `TERMINAL_DOCKER_NETWORK` | `docker_network` | `true` / `false` — padrão `true`; `false` = `--network=none` |
| `TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES` | `docker_persist_across_processes` | `true` / `false` — padrão `true` |
| `TERMINAL_DOCKER_ORPHAN_REAPER` | `docker_orphan_reaper` | `true` / `false` — padrão `true` |
| `TERMINAL_CONTAINER_CPU` | `container_cpu` | Núcleos CPU |
| `TERMINAL_CONTAINER_MEMORY` | `container_memory` | MB |
| `TERMINAL_CONTAINER_DISK` | `container_disk` | MB |
| `TERMINAL_CONTAINER_PERSISTENT` | `container_persistent` | `true` / `false` — controla dirs de workspace bind-mount, distinto de `docker_persist_across_processes` |
| `TERMINAL_LIFETIME_SECONDS` | `lifetime_seconds` | Janela idle reaper |
| `TERMINAL_TIMEOUT` | `timeout` | Timeout por comando |
| `HERMES_DOCKER_BINARY` | _none_ | Força caminho específico do binário docker/podman |

### Backend SSH {#ssh-backend}

Executa comandos em um servidor remoto via SSH. Usa ControlMaster para reutilização de conexão (keepalive idle de 5 minutos). Shell persistente habilitado por padrão — estado (cwd, vars de ambiente) sobrevive entre comandos.

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

| Variável | Padrão | Descrição |
|----------|---------|-------------|
| `TERMINAL_SSH_PORT` | `22` | Porta SSH |
| `TERMINAL_SSH_KEY` | (padrão do sistema) | Caminho para chave privada SSH |
| `TERMINAL_SSH_PERSISTENT` | `true` | Habilitar shell persistente |

**Como funciona:** Conecta na inicialização com `BatchMode=yes` e `StrictHostKeyChecking=accept-new`. Shell persistente mantém um único processo `bash -l` vivo no host remoto, comunicando via arquivos temporários. Comandos que precisam de `stdin_data` ou `sudo` caem automaticamente para modo one-shot.

### Backend Modal {#modal-backend}

Executa comandos em um sandbox na nuvem [Modal](https://modal.com). Cada task recebe uma VM isolada com CPU, memória e disco configuráveis. Filesystem pode ser snapshot/restaurado entre sessões.

```yaml
terminal:
  backend: modal
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB (5GB)
  container_disk: 51200            # MB (50GB)
  container_persistent: true       # Snapshot/restore filesystem
```

**Obrigatório:** Variáveis de ambiente `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, ou arquivo de config `~/.modal.toml`.

**Persistência:** Quando habilitada, o filesystem do sandbox é snapshotado no cleanup e restaurado na próxima sessão. Snapshots são rastreados em `~/.hermes/modal_snapshots.json`. Isso preserva estado do filesystem, não processos vivos, espaço de PID ou jobs em segundo plano.

**Arquivos de credencial:** Montados automaticamente de `~/.hermes/` (tokens OAuth, etc.) e sincronizados antes de cada comando.

### Backend Daytona {#daytona-backend}

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

**Persistência:** Quando habilitada, sandboxes são parados (não deletados) no cleanup e retomados na próxima sessão. Nomes de sandbox seguem o padrão `hermes-{task_id}`.

**Limite de disco:** Daytona impõe máximo de 10 GiB. Requisições acima disso são limitadas com aviso.

### Backend Vercel Sandbox {#vercel-sandbox-backend}

Executa comandos em um microVM na nuvem [Vercel Sandbox](https://vercel.com/docs/vercel-sandbox). O Hermes usa as superfícies normais de terminal e arquivo; não há ferramentas voltadas ao modelo específicas do Vercel.

```yaml
terminal:
  backend: vercel_sandbox
  vercel_runtime: node24          # node24 | node22 | python3.13
  cwd: /vercel/sandbox            # default workspace root
  container_persistent: true      # Snapshot/restore filesystem
  container_disk: 51200           # Shared default only; custom disk is unsupported
```

**Instalação obrigatória:** Instale o extra SDK opcional:

```bash
pip install 'hermes-agent[vercel]'
```

**Autenticação obrigatória:** Configure auth com access token com os três `VERCEL_TOKEN`, `VERCEL_PROJECT_ID` e `VERCEL_TEAM_ID`. Esta é a config suportada para deploys e processos Hermes de longa duração normais no Render, Railway, Docker e hosts similares.

Para desenvolvimento local one-off, o Hermes também aceita tokens OIDC Vercel de curta duração:

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

De um diretório de projeto Vercel linkado, você pode omitir o nome do projeto:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

Tokens OIDC são de curta duração e não devem ser usados como caminho de deploy documentado.

**Runtime:** `terminal.vercel_runtime` suporta `node24`, `node22` e `python3.13`. Se não definido, o Hermes usa `node24` por padrão.

**Persistência:** Quando `container_persistent: true`, o Hermes faz snapshot do filesystem do sandbox durante cleanup e restaura um sandbox posterior para a mesma task desse snapshot. Conteúdo do snapshot pode incluir credenciais, skills e arquivos de cache sincronizados pelo Hermes que foram copiados para o sandbox. Isso preserva apenas estado do filesystem; não preserva identidade viva do sandbox, espaço de PID, estado de shell ou processos em segundo plano rodando.

**Comandos em segundo plano:** `terminal(background=true)` usa o fluxo genérico de processo em segundo plano não-local do Hermes. Você pode spawnar, fazer poll, wait, ver logs e matar processos pela ferramenta de processo normal enquanto o sandbox está vivo. O Hermes não fornece recuperação nativa Vercel de processos destacados após cleanup ou reinício.

**Dimensionamento de disco:** Vercel Sandbox atualmente não suporta o knob de recurso `container_disk` do Hermes. Deixe `container_disk` sem definir ou no padrão compartilhado `51200`; valores não padrão falham diagnósticos e criação de backend em vez de serem ignorados silenciosamente.

### Backend Singularity/Apptainer {#singularityapptainer-backend}

Executa comandos em um container [Singularity/Apptainer](https://apptainer.org). Projetado para clusters HPC e máquinas compartilhadas onde Docker não está disponível.

```yaml
terminal:
  backend: singularity
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"
  container_cpu: 1                 # CPU cores
  container_memory: 5120           # MB
  container_persistent: true       # Writable overlay persists across sessions
```

**Requisitos:** Binário `apptainer` ou `singularity` em `$PATH`.

**Tratamento de imagem:** URLs Docker (`docker://...`) são convertidas automaticamente para arquivos SIF e cacheadas. Arquivos `.sif` existentes são usados diretamente.

**Diretório scratch:** Resolvido nesta ordem: `TERMINAL_SCRATCH_DIR` → `TERMINAL_SANDBOX_DIR/singularity` → `/scratch/$USER/hermes-agent` (convenção HPC) → `~/.hermes/sandboxes/singularity`.

**Isolamento:** Usa `--containall --no-home` para isolamento total de namespace sem montar o home do host.

### Problemas comuns de backend de terminal {#common-terminal-backend-issues}

Se comandos de terminal falham imediatamente ou a ferramenta terminal é reportada como desabilitada:

- **Local** — Sem requisitos especiais. O padrão mais seguro ao começar.
- **Docker** — Execute `docker version` para verificar se Docker funciona. Se falhar, corrija Docker ou `hermes config set terminal.backend local`.
- **SSH** — Tanto `TERMINAL_SSH_HOST` quanto `TERMINAL_SSH_USER` devem estar definidos. O Hermes registra erro claro se algum faltar.
- **Modal** — Precisa de env var `MODAL_TOKEN_ID` ou `~/.modal.toml`. Execute `hermes doctor` para verificar.
- **Daytona** — Precisa de `DAYTONA_API_KEY`. O SDK Daytona trata configuração de URL do servidor.
- **Singularity** — Precisa de `apptainer` ou `singularity` em `$PATH`. Comum em clusters HPC.

Na dúvida, defina `terminal.backend` de volta para `local` e verifique se comandos rodam lá primeiro.

### Sincronização remoto-para-host no teardown {#remote-to-host-state-sync-on-teardown}

Para os backends **SSH**, **Modal** e **Daytona**, o Hermes envia seu estado `~/.hermes/` (arquivos de credencial, skills, cache) para o sandbox remoto durante a sessão, e no teardown **sincroniza de volta arquivos de estado alterados** para suas localizações originais no host. Arquivos que diferem do que foi enviado originalmente (comparados por hash de conteúdo) são aplicados no lugar; arquivos remotos novos sob um diretório sincronizado (ex.: uma skill que o agente criou remotamente) são mapeados de volta ao caminho correspondente no host. Arquivos de credencial upload-only nunca são sobrescritos no host.

- O sync-back tenta até 3 vezes com backoff e recusa extrair arquivos remotos maiores que 2 GiB.
- Docker e Singularity usam bind mounts (visão live do filesystem do host) e não precisam disso.
- Isso cobre estado Hermes (`~/.hermes/`), **não** arquivos arbitrários da working tree dentro do sandbox — faça o agente copiar artefatos importantes explicitamente (ex.: `scp`, `modal volume put`) antes do sandbox ser destruído.

### Mounts de volume Docker {#docker-volume-mounts}

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

Se você usa um gateway de mensagens e quer que o agente envie arquivos gerados via
`MEDIA:/...`, prefira um mount de export visível no host como
`/home/user/.hermes/cache/documents:/output`.

- Escreva arquivos dentro do Docker em `/output/...`
- Emita o **caminho do host** em `MEDIA:`, por exemplo:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`
- **Não** emita `/workspace/...` ou `/output/...` a menos que esse caminho exato também
  exista para o processo gateway no host

:::warning
Chaves duplicadas em YAML sobrescrevem silenciosamente as anteriores. Se você já tem um
bloco `docker_volumes:`, mescle novos mounts na mesma lista em vez de adicionar
outra chave `docker_volumes:` depois no arquivo.
:::

Também pode ser definido via variável de ambiente: `TERMINAL_DOCKER_VOLUMES='["/host:/container"]'` (array JSON).

### Encaminhamento de credenciais Docker {#docker-credential-forwarding}

Por padrão, sessões de terminal Docker não herdam credenciais arbitrárias do host. Se precisar de um token específico dentro do container, adicione em `terminal.docker_forward_env`.

```yaml
terminal:
  backend: docker
  docker_forward_env:
    - "GITHUB_TOKEN"
    - "NPM_TOKEN"
```

O Hermes resolve cada variável listada do shell atual primeiro, depois cai para `~/.hermes/.env` se foi salva com `hermes config set`.

:::warning
Qualquer coisa listada em `docker_forward_env` fica visível a comandos rodados dentro do container. Encaminhe apenas credenciais que você aceita expor à sessão de terminal.
:::

### Rodar o container como seu usuário do host {#running-the-container-as-your-host-user}

Por padrão containers Docker rodam como `root` (UID 0). Arquivos criados dentro de `/workspace` ou outros bind-mounts acabam owned by root no host, então após uma sessão você precisa `sudo chown` antes de editá-los no editor do host. A flag `terminal.docker_run_as_host_user` corrige isso:

```yaml
terminal:
  backend: docker
  docker_run_as_host_user: true   # default: false
```

Quando habilitada, o Hermes anexa `--user $(id -u):$(id -g)` ao comando `docker run` para arquivos escritos em diretórios bind-mounted (`/workspace`, `/root`, qualquer coisa em `docker_volumes`) serem owned pelo seu usuário do host, não root. O trade-off: o container não pode mais `apt install` ou escrever em caminhos owned by root como `/root/.npm` — use uma imagem base cujo `HOME` é owned by um usuário non-root (ou adicione tooling necessário no build da imagem) se precisar de ambos.

Deixe `false` (padrão) para comportamento retrocompatível. Ative quando seu fluxo é sobretudo "editar arquivos montados do host" e você está cansado de `sudo chown -R`.

### Opcional: montar o diretório de lançamento em `/workspace` {#optional-mount-the-launch-directory-into-workspace}

Sandboxes Docker permanecem isolados por padrão. O Hermes **não** passa seu diretório de trabalho atual do host para o container a menos que você opte explicitamente.

Habilite em `config.yaml`:

```yaml
terminal:
  backend: docker
  docker_mount_cwd_to_workspace: true
```

Quando habilitado:
- se você lançar o Hermes de `~/projects/my-app`, esse diretório do host é bind-mounted em `/workspace`
- o backend Docker inicia em `/workspace`
- ferramentas de arquivo e comandos de terminal veem o mesmo projeto montado

Quando desabilitado, `/workspace` permanece owned by sandbox a menos que você monte algo explicitamente via `docker_volumes`.

Tradeoff de segurança:
- `false` preserva o limite do sandbox
- `true` dá ao sandbox acesso direto ao diretório de onde você lançou o Hermes

Use o opt-in apenas quando quiser intencionalmente que o container trabalhe em arquivos live do host.

### Shell persistente {#persistent-shell}

Por padrão, cada comando de terminal roda em seu próprio subprocess — diretório de trabalho, variáveis de ambiente e variáveis de shell resetam entre comandos. Quando **shell persistente** está habilitado, um único processo bash de longa duração é mantido vivo entre chamadas `execute()` para que estado sobreviva entre comandos.

Isso é mais útil para o **backend SSH**, onde também elimina overhead de conexão por comando. Shell persistente está **habilitado por padrão para SSH** e desabilitado para o backend local.

```yaml
terminal:
  persistent_shell: true   # default — enables persistent shell for SSH
```

Para desabilitar:

```bash
hermes config set terminal.persistent_shell false
```

**O que persiste entre comandos:**
- Diretório de trabalho (`cd /tmp` permanece para o próximo comando)
- Variáveis de ambiente exportadas (`export FOO=bar`)
- Variáveis de shell (`MY_VAR=hello`)

**Precedência:**

| Nível | Variável | Padrão |
|-------|----------|---------|
| Config | `terminal.persistent_shell` | `true` |
| Sobrescrita SSH | `TERMINAL_SSH_PERSISTENT` | segue config |
| Sobrescrita local | `TERMINAL_LOCAL_PERSISTENT` | `false` |

Variáveis de ambiente por backend têm maior precedência. Se quiser shell persistente no backend local também:

```bash
export TERMINAL_LOCAL_PERSISTENT=true
```

:::note
Comandos que requerem `stdin_data` ou sudo caem automaticamente para modo one-shot, já que o stdin do shell persistente já está ocupado pelo protocolo IPC.
:::

Veja [Code Execution](features/code-execution.md) e a [seção Terminal do README](features/tools.md) para detalhes de cada backend.

## Configurações de skills {#skill-settings}

Skills podem declarar suas próprias configurações via frontmatter SKILL.md. São valores não secretos (caminhos, preferências, configurações de domínio) armazenados sob o namespace `skills.config` em `config.yaml`.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data   # Example — each skill defines its own keys
```

**Como funcionam as configurações de skill:**

- `hermes config migrate` varre todas as skills habilitadas, encontra configurações não definidas e oferece prompt
- `hermes config show` exibe todas as configurações de skill em "Skill Settings" com a skill a que pertencem
- Quando uma skill carrega, seus valores de config resolvidos são injetados no contexto da skill automaticamente

**Definindo valores manualmente:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

Para detalhes sobre declarar configurações em suas próprias skills, veja [Creating Skills — Config Settings](/developer-guide/creating-skills#config-settings-configyaml).

### Guarda em escritas de skills criadas pelo agente {#guard-on-agent-created-skill-writes}

Quando o agente usa `skill_manage` para criar, editar, patch ou deletar uma skill, o Hermes pode opcionalmente escanear o conteúdo novo/atualizado por padrões de palavras-chave perigosos (coleta de credenciais, prompt injection óbvio, instruções de exfil). O scanner está **desligado por padrão** — fluxos reais de agente que legitimamente tocam `~/.ssh/` ou mencionam `$OPENAI_API_KEY` disparavam a heurística com frequência demais. Reative se quiser que o scanner peça aprovação antes das escritas de skill do agente:

```yaml
skills:
  guard_agent_created: true   # default: false
```

Quando ligado, qualquer escrita `skill_manage` sinalizada aparece como prompt de aprovação com a justificativa do scanner. Escritas aceitas entram; negadas retornam erro explicativo ao agente.

### Aprovação de escrita para skills {#write-approval-for-skill-writes}

Independente do scanner de conteúdo acima, `skills.write_approval` exige **toda** escrita de skill do agente (create / edit / patch / delete / arquivos de suporte) atrás da sua aprovação explícita — o mesmo mecanismo approve/deny de comandos perigosos:

```yaml
skills:
  write_approval: false   # false = write freely (default) | true = stage every write for review
```

Quando ligado, escritas de skill são staged em `~/.hermes/pending/skills/` e revisadas com `/skills pending`, `/skills diff <id>`, `/skills approve <id>`, `/skills reject <id>` — da CLI ou qualquer plataforma de mensagens. Alterne em runtime com `/skills approval on|off`. Memória tem o mesmo gate (`memory.write_approval`, abaixo). Walkthrough completo: [Gating agent skill writes](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval).
## Configuração de memória {#memory-configuration}

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # true = require approval before any memory write
```

Com `memory.write_approval: true`, escritas de memória precisam da sua aprovação antes de entrarem: turnos interativos da CLI pedem inline; sessões de mensagens e a revisão de auto-melhoria em segundo plano fazem stage da escrita para revisão `/memory pending` → `/memory approve <id>` / `/memory reject <id>`. Alterne em runtime com `/memory approval on|off`. Veja [Controlling memory writes](/user-guide/features/memory#controlling-memory-writes-write_approval).

## Truncamento de arquivos de contexto {#context-file-truncation}

Controla quanto conteúdo o Hermes carrega de cada arquivo de contexto automático antes de aplicar truncamento head/tail. Aplica-se a arquivos injetados no prompt de sistema como `SOUL.md`, `.hermes.md`, `AGENTS.md`, `CLAUDE.md` e `.cursorrules`. **Não** afeta a ferramenta `read_file`.

```yaml
context_file_max_chars: null  # default — dynamic cap scaled to the model's context window (floor 20K, ceiling 500K chars)
```

Defina um inteiro positivo para fixar um limite fixo em vez do comportamento dinâmico:

```yaml
context_file_max_chars: 25000
```

## Segurança de leitura de arquivo {#file-read-safety}

Controla quanto conteúdo uma única chamada `read_file` pode retornar. Leituras que excedem o limite são rejeitadas com erro dizendo ao agente para usar `offset` e `limit` para um intervalo menor. Isso evita que uma leitura de bundle JS minificado ou arquivo de dados grande inunde a janela de contexto.

```yaml
file_read_max_chars: 100000  # default — ~25-35K tokens
```

Aumente se estiver em um modelo com janela grande e lê arquivos grandes com frequência. Diminua para modelos de contexto pequeno:

```yaml
# Large context model (200K+)
file_read_max_chars: 200000

# Small local model (16K context)
file_read_max_chars: 30000
```

O agente também deduplica leituras de arquivo automaticamente — se a mesma região do arquivo é lida duas vezes e o arquivo não mudou, um stub leve é retornado em vez de reenviar o conteúdo. Isso reseta na compressão de contexto para o agente reler arquivos após o conteúdo ser resumido.

## Limites de truncamento de saída de ferramentas {#tool-output-truncation-limits}

Três caps relacionados controlam quanta saída bruta uma ferramenta pode retornar antes do Hermes truncar:

```yaml
tool_output:
  max_bytes: 50000        # terminal output cap (chars)
  max_lines: 2000         # read_file pagination cap
  max_line_length: 2000   # per-line cap in read_file's line-numbered view
```

- **`max_bytes`** — Quando um comando `terminal` produz mais que este número de caracteres de stdout/stderr combinados, o Hermes mantém os primeiros 40% e últimos 60% e insere um aviso `[OUTPUT TRUNCATED]` entre eles. Padrão `50000` (≈12-15K tokens em tokenizers típicos).
- **`max_lines`** — Limite superior no parâmetro `limit` de uma única chamada `read_file`. Requisições acima disso são limitadas para uma leitura não inundar a janela de contexto. Padrão `2000`.
- **`max_line_length`** — Cap por linha aplicado quando `read_file` emite a view numerada por linha. Linhas mais longas são truncadas a este número de chars seguido de `... [truncated]`. Padrão `2000`.

Aumente os limites em modelos com janela grande que podem pagar mais saída bruta por chamada. Diminua para modelos de contexto pequeno:

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

### Orçamento de spillover de tool-result {#tool-result-spillover-budget}

Separadamente da truncagem, *resultados* de tool oversized são derramados em disco em vez de cortados: a saída completa é salva sob `$HERMES_HOME/cache/spillover/` e o conteúdo in-context é substituído por um preview mais o path do arquivo salvo (legível com `read_file` usando `offset`/`limit`, ou processável com `execute_code`). O limiar genérico de spillover por resultado é 100.000 chars, reduzido automaticamente para modelos de contexto pequeno.

Resultados de tool MCP (tools nomeadas `mcp_*`) derramam num default mais apertado de **50.000 chars**: servidores MCP rotineiramente retornam payloads grandes sem paginação (catálogos de tool-discovery, execuções em batch) que de outra forma ficariam sob o limiar genérico e inchariam o contexto em todo turn seguinte. Nada se perde — o resultado completo é preservado em disco. Sobrescreva o limiar via:

```yaml
tool_budget:
  mcp_result_size_chars: 50000   # per-result spillover threshold for mcp_* tools
```

O limiar MCP é sempre capped no limiar genérico por resultado (possivelmente scaled pelo contexto), então aumentá-lo não pode exceder o que a janela do modelo ativo permite.

O Hermes também sinaliza **elisão do lado do provider**: quando um resultado de tool MCP ou web embute seus próprios marcadores de truncagem (`...N more items`, `"has_more": true`, notas "saved to sandbox"), um aviso de uma linha é anexado ao resultado avisando que os dados visíveis estão incompletos e devem ser paged/fetched antes de tratar qualquer enumeração como completa.

## Desabilitação global de toolset {#global-toolset-disable}

Para suprimir toolsets específicos na CLI e em toda plataforma de gateway em um
lugar, liste seus nomes em `agent.disabled_toolsets`:

```yaml
agent:
  disabled_toolsets:
    - memory       # hide memory tools + MEMORY_GUIDANCE injection
    - web          # no web_search / web_extract anywhere
```

Isso se aplica **depois** da config de ferramentas por plataforma (`platform_toolsets` escrita por
`hermes tools`), então um toolset listado aqui é sempre removido — mesmo se a
config salva da plataforma ainda o listar. Use quando quiser um único
interruptor para "desligar X em todo lugar" em vez de editar 15+ linhas de plataforma na
UI `hermes tools`.

Deixar a lista vazia, ou omitir a chave, é no-op.

## Isolamento de git worktree {#git-worktree-isolation}

Habilite git worktrees isolados para rodar vários agentes em paralelo no mesmo repo:

```yaml
worktree: true    # Always create a worktree (same as hermes -w)
# worktree: false # Default — only when -w flag is passed
```

Quando habilitado, cada sessão CLI cria um worktree novo em `.worktrees/` com sua própria branch. Agentes podem editar arquivos, commit, push e criar PRs sem interferir uns nos outros. Worktrees limpos são removidos ao sair; sujos são mantidos para recuperação manual.

Por padrão a nova branch do worktree parte da **ponta remota recém-buscada** (upstream da branch atual, senão branch padrão do remoto) para começar atual com o projeto em vez do `HEAD` local possivelmente stale do clone. Isso mantém o diff de um PR escopado à mudança real em vez de herdar o quanto o clone local estava atrás. Defina `worktree_sync: false` para ramificar do `HEAD` local — útil offline, ou quando quer deliberadamente o estado exato atual do clone como base. Se o remoto não for alcançável, cai para `HEAD` local automaticamente.

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

## Compressão de contexto {#context-compression}

O Hermes comprime automaticamente conversas longas para permanecer na janela de contexto do modelo. O summarizer de compressão é uma chamada LLM separada — você pode apontá-la a qualquer provedor ou endpoint.

Todas as configurações de compressão ficam em `config.yaml` (sem variáveis de ambiente).

### Referência completa {#full-reference}

```yaml
compression:
  enabled: true                                     # Toggle compression on/off
  progress_notices: false                           # Opt-in: deliver routine compression progress notices to chat platforms — see below
  threshold: 0.50                                   # Compress at this % of context limit
  threshold_tokens: null                            # Absolute token cap (optional) — takes lower of ratio vs absolute
  target_ratio: 0.20                                # Fraction of threshold to preserve as recent tail
  tail_mode: legacy                                 # Tail retention: "legacy" (0.20×window verbatim tail) or "lean" (clamped 2.5% tail, 10K-25K, with digests + anchor index + session_search recovery pointers in the summary — ~3x fewer retained tokens after compaction)
  protect_last_n: 20                                # Min recent messages to keep uncompressed
  protect_first_n: 3                                # Non-system head messages pinned across compactions (0 = pin nothing)
  in_place: true                                    # Compact on the same session id (no rotation) — see below
  idle_compact_after_seconds: 0                     # Opt-in idle compaction (0 = disabled) — see below
  hygiene_hard_message_limit: 5000                  # Gateway safety valve — see below
  hygiene_timeout_seconds: 30                       # Max seconds of NO summary-model output before hygiene compression is cut off
  hygiene_total_ceiling_seconds: 600                # Absolute cap on the hygiene wait even while tokens are still streaming
  hygiene_failure_cooldown_seconds: 300             # First rung of the per-session hygiene-failure backoff (x1/x3/x9, capped at 1h)
  context_timeout_seconds: 120                      # Inactivity budget for in-agent compress_context (loop /compress / preflight) — see below
  context_total_ceiling_seconds: 600                # Absolute cap on the *pre-commit* in-agent compress_context wait even while tokens are still streaming (an already-started SessionDB commit is never abandoned; overruns are logged + surfaced)
  proactive_prune_tokens: 0                         # Opt-in tokens trigger for the no-LLM tool-result prune (0 = off; see below)
  proactive_prune_min_result_chars: 8000            # Prune's summarize pass only touches tool results larger than this (clamped >= 200)
  proactive_prune_min_reclaim_tokens: 4096          # Prune only commits when it reclaims at least this many tokens (0 = commit any)

# The summarization model/provider is configured under auxiliary:
auxiliary:
  compression:
    model: ""                                       # Empty = use main chat model. Override with e.g. "google/gemini-3-flash-preview" for cheaper/faster compression.
    provider: "auto"                                # Provider: "auto", "openrouter", "nous", "codex", "main", etc.
    base_url: null                                  # Custom OpenAI-compatible endpoint (overrides provider)
```

:::info Migração de config legada
Configs antigas com `compression.summary_model`, `compression.summary_provider` e `compression.summary_base_url` são migradas automaticamente para `auxiliary.compression.*` no primeiro carregamento (config version 17). Nenhuma ação manual necessária.
:::

`progress_notices` (padrão `false`) controla se **status de progresso rotineiros** de compressão chegam a plataformas de chat (Telegram, Discord, Slack, etc.). Por design, compressão automática é silenciosa em superfícies de chat — roda em segundo plano só com logging server-side. Defina `progress_notices: true` para optar por ver o ciclo de vida rotineiro em plataformas de chat: aviso inicial "Compacting context…", gatilhos preflight/pré-API, compactação idle, progresso de retry ("Compressed 30 → 12 messages, retrying…") e aviso "Context compaction complete". O gate é escopado só a status de compressão — ruído operacional não relacionado (falhas de modelo auxiliar, chatter de rate-limit/retry do provedor) permanece suprimido de qualquer forma. Avisos de **falha** de compressão e feedback manual `/compress` são sempre visíveis independentemente desta config. Editar este valor em um gateway rodando entra em vigor na próxima mensagem.

`hygiene_hard_message_limit` é uma **válvula de segurança pré-compressão** só do gateway. Existe para quebrar uma espiral da morte: quando chamadas de API continuam desconectando em sessão oversized, o gateway nunca recebe dados de uso de tokens, então o threshold baseado em tokens não dispara, a transcrição continua crescendo e desconexões pioram. Este piso baseado em contagem de mensagens dispara só na contagem (sempre conhecida, independente de falhas de API) para forçar compressão e recuperar a sessão. Padrão `5000` — muito acima de qualquer sessão normal, incluindo modelos de contexto grande (1M+) fazendo milhares de turnos curtos, que comprimem no threshold de tokens muito antes disso. Aumente para plataformas incomuns, diminua para forçar compressão mais agressiva. Editar este valor em gateway rodando entra em vigor na próxima mensagem (veja abaixo).

`hygiene_timeout_seconds` é o **orçamento de inatividade** do gateway para este passe de compressão pré-agente — não um cap total de relógio. A chamada de resumo de compressão faz stream do modelo, e cada token chegando conta como progresso: um modelo de raciocínio lento que ainda gera continua estendendo seu próprio deadline, então modelos de resumo lentos mas saudáveis nunca são cortados no meio da geração. Só quando o modelo de resumo produz **nenhuma saída** por este número de segundos (backend down, conexão pendurada, provedor silencioso) o gateway avisa o usuário, continua a mensagem entrante sem compressão e registra cooldown temporário de falha por sessão em vez de parecer travado.

`hygiene_total_ceiling_seconds` (padrão `600`) limita a espera total mesmo enquanto tokens ainda se movem, para um stream trickle degenerado não prender um turno indefinidamente. É limitado a pelo menos `hygiene_timeout_seconds`.

`hygiene_failure_cooldown_seconds` controla esse cooldown por sessão após timeout ou abort de compressão hygiene. Durante o cooldown, o gateway pula tentativas hygiene repetidas para a mesma sessão oversized para toda mensagem entrante não bloquear no mesmo backend auxiliar quebrado. `/compress`, `/reset` ou um turno saudável posterior ainda podem recuperar a sessão.

O valor é a **primeira degrau** de uma escada escalonada, não intervalo fixo: falhas consecutivas para a mesma sessão esperam `1x`, `3x`, depois `9x` este valor, limitado a uma hora. Uma sessão cujo modelo de resumo está permanentemente quebrado faz backoff em vez de retry forever em intervalo fixo, e uma execução que realmente encolhe a transcrição reseta para o primeiro degrau. Escalonamento é por sessão e process-local — reinício do gateway reseta para o primeiro degrau enquanto o deadline do cooldown sobrevive.

`context_timeout_seconds` (padrão `120`) é o mesmo **orçamento de inatividade** para `compress_context` in-agent — loop de conversa, compactação preflight e `/compress` manual — para um modelo de resumo pendurado não travar uma sessão indefinidamente. Tokens de resumo em stream estendem a espera; só um worker silencioso é cortado. No timeout o Hermes pula compactação, mantém as mensagens existentes e avisa o usuário. Defina `0` para desabilitar. Hygiene de sessão do gateway mantém seu próprio caminho `hygiene_timeout_seconds` e não é double-wrapped.

`context_total_ceiling_seconds` (padrão `600`) limita a espera **pré-commit** in-agent (fase summary / stream) mesmo enquanto tokens ainda se movem. É limitado a pelo menos `context_timeout_seconds`. A garantia exata: **a fase de resumo é limitada por este teto; a fase de commit é logada e exposta se exceder.** Uma vez que o worker entrou na fence de commit de compressão e mutação SessionDB está em flight, o commit nunca é abandonado no meio — isso arriscaria divergência de transcrição — mas a espera deixa de ser silenciosa: se o commit passa do teto, o Hermes loga o overrun (WARNING, escalando para ERROR em repetição), envia aviso one-shot pelo canal de warning visível ao usuário e continua esperando em incrementos limitados até o commit completar.

`protect_first_n` controla quantas mensagens **não-system** de cabeça são fixadas em toda compactação. Padrão `3` — a troca user/assistant inicial sobrevive a todo passe do summarizer para o objetivo original permanecer visível. Em sessões de compactação rolling de longa duração onde o turno inicial não é mais relevante, defina `protect_first_n: 0` para não fixar nada além do prompt de sistema + resumo + tail. O prompt de sistema em si é sempre preservado independentemente desta config.

`in_place` (padrão `true`) controla o que acontece com a identidade da sessão quando compactação dispara. Quando `true`, compactação reescreve a lista de mensagens e reconstrói o prompt de sistema **sem rotacionar o id da sessão** — a conversa mantém um id durável por toda a vida (sem cadeia `parent_session_id`, sem renumeração `name #2` / `#3` em listas de sessão). Compactação é não-destrutiva: o contexto live é compactado, mas os turnos pré-compactação são soft-archived sob o mesmo id (marcados inactive/compacted) — ainda pesquisáveis via `session_search` e recuperáveis, não deletados. Hooks veem o modo via o campo `in_place` no evento `session:compress`. Defina `in_place: false` para restaurar o comportamento legado onde cada compactação rotaciona para um novo id de sessão ligado ao antigo.

`threshold_tokens` define um **cap absoluto de tokens** opcional para o gatilho de compressão. Quando definido, compressão dispara no menor entre o `threshold` baseado em ratio e esta contagem absoluta — então compressão nunca dispara depois do número de tokens preferido do usuário independentemente do modelo ativo. Isso resolve o problema onde trocar entre modelos com janelas diferentes (ex.: 1M → 400K) desloca o ponto absoluto do gatilho. O cap é limitado ao context length do modelo, então defini-lo maior que o modelo suporta é seguro — o threshold baseado em ratio é usado. Padrão `null` (desabilitado — só threshold baseado em ratio). O cap sobrevive trocas de modelo e ativações de fallback.

`idle_compact_after_seconds` é um gatilho **opt-in baseado em tempo** que complementa o `threshold` baseado em tamanho. Padrão `0` (desabilitado). Quando acima de 0, uma sessão que retoma após pelo menos tantos segundos de inatividade compacta seu histórico acumulado upfront, antes da primeira resposta — então um thread long-lived (ex.: conversa Telegram à qual você volta horas depois) não relê todo o contexto stale a cada turno subsequente. Nunca dispara quando o contexto já está no ou abaixo do alvo pós-compressão (`threshold × target_ratio`), e honra os mesmos guards de failure-cooldown, anti-thrash e lock por sessão que toda compactação automática. Exemplo: `idle_compact_after_seconds: 1800` compacta após 30 minutos idle.

`proactive_prune_tokens` habilita um prune determinístico sem LLM de payloads antigos de resultado de ferramenta que roda independentemente de `threshold`. Em modelos de janela grande o `threshold` de compressão (≈50% da janela) raramente dispara, então saídas volumosas de ferramenta (dumps de terminal, leituras de arquivo, extracts web) viajam no histórico e são reenviadas a cada turno subsequente. Quando histórico reenviado excede `proactive_prune_tokens` (padrão `0` = off; tente `48000` para habilitar), o prune deduplica resultados idênticos, resume oversized antigos e trunca argumentos grandes de tool-call — protegendo as `protect_last_n` mensagens mais recentes e nunca chamando o modelo. Saídas completas permanecem recuperáveis do store de sessão. `proactive_prune_min_result_chars` (padrão `8000`, limitado a ≥ 200) define o tamanho abaixo do qual um resultado de ferramenta fica intocado. `proactive_prune_min_reclaim_tokens` (padrão `4096`) impede que um prune faça commit a menos que recupere pelo menos tantos tokens — um prune committed reescreve histórico já enviado e invalida o prefixo de prompt-cache do provedor, então este gate mantém essas quebras de cache episódicas e amortizadas (uma quebra significativa, como um limite de compressão) em vez de disparar a cada iteração de ferramenta. Isso roda só sob o engine `compressor` embutido; outros context engines herdam no-op.

:::tip Hot-reload do gateway de compressão e context length
Em releases recentes, editar `model.context_length` ou qualquer chave `compression.*` em `config.yaml` em gateway rodando entra em vigor na próxima mensagem — sem reinício de gateway, `/reset` ou rotação de sessão. A assinatura do agente em cache inclui essas chaves, então o gateway reconstrói o agente transparentemente quando vê mudança. Chaves de API e config de tool/skill ainda exigem os caminhos de reload usuais.
:::

### Configurações comuns {#common-setups}

**Padrão (auto-detect) — nenhuma configuração necessária:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```
Usa seu provedor e modelo principal. Sobrescreva por task (ex.: `auxiliary.compression.provider: openrouter` + `model: google/gemini-2.5-flash`) se quiser compressão em modelo mais barato que seu chat principal.

**Forçar um provedor específico** (OAuth ou baseado em API key):
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```
Funciona com qualquer provedor: `nous`, `openrouter`, `codex`, `anthropic`, `main`, etc.

**Endpoint customizado** (self-hosted, Ollama, zai, DeepSeek, etc.):
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```
Aponta a um endpoint OpenAI-compatible customizado. Usa `OPENAI_API_KEY` para auth.

### Como os três knobs interagem {#how-the-three-knobs-interact}

| `auxiliary.compression.provider` | `auxiliary.compression.base_url` | Resultado |
|---------------------|---------------------|--------|
| `auto` (default) | not set | Auto-detect best available provider |
| `nous` / `openrouter` / etc. | not set | Force that provider, use its auth |
| any | set | Use the custom endpoint directly (provider ignored) |

:::warning Requisito de context length do modelo de resumo
O modelo de resumo **deve** ter janela de contexto pelo menos tão grande quanto a do modelo principal do agente. O compressor envia a seção do meio completa da conversa ao modelo de resumo — se a janela desse modelo for menor que a do principal, a chamada de summarização falhará com erro de context length. Quando isso acontece, os turnos do meio são **descartados sem resumo**, perdendo contexto de conversa silenciosamente. Se sobrescrever o modelo, verifique se seu context length atende ou excede o do principal.
:::

## Timeout de lease de turno do gateway {#gateway-turn-lease-timeout}

O gateway serializa turnos pelo id de sessão resolvido para duas routing keys
não poderem carregar e escrever a mesma transcrição concorrentemente. Configure a espera
máxima de lease independentemente do timeout de inatividade ordinário do agente:

```yaml
agent:
  gateway_turn_lease_timeout: 1800
```

Se outro turno ainda segura o lease da sessão quando este orçamento expira, o Hermes
falha fechado: não carrega a transcrição nem roda o modelo para a mensagem
esperando. O usuário recebe aviso de rejeição e deve reenviar. O Hermes não
reencadeia automaticamente a mensagem porque fazer isso sem ordenação durável e
idempotência poderia processá-la duas vezes. Valores não positivos usam o padrão
de 1800 segundos.

## Watchdog de parada de sessão {#session-stall-watchdog}

O gateway roda um watchdog de parada notify-only (`agent.session_stall_timeout`, padrão `300` segundos, `0` = desabilitado). Quando uma sessão ocupada tem um **follow-up inbound pendente** e o relógio de atividade compartilhado do agente ficou idle por pelo menos este tempo, o gateway registra WARNING e envia notificação one-shot ao usuário:

```
⚠️ Agent session appears stalled (last activity N min ago). Try /new to reset.
```

Semântica:

- **Só notifica.** O watchdog nunca mata o turno — contraste com `agent.gateway_timeout`, que cancela uma execução após inatividade prolongada. O aviso de parada só diz que o agente parece emperrado para você decidir (`/new`, `/stop` ou continuar esperando).
- **Uma notificação por episódio de parada.** O latch limpa quando o inbound pendente drena ou atividade retoma, então uma sessão que se recupera e para de novo notifica de novo.
- Progresso vem só do snapshot de atividade compartilhado (tool calls, progresso de stream de API, heartbeats de compressão). Inbound pendente é gate de notificação, não relógio de progresso.

```yaml
agent:
  session_stall_timeout: 300   # seconds; 0 disables the watchdog
```

## Escalação de atenção de reconnect {#reconnect-attention-escalation}

Quando um adapter de plataforma falha em conectar (queda de rede, token de bot revogado, sidecar quebrado), o gateway retenta indefinidamente com backoff exponencial limitado — as retries nunca param, então uma queda transitória sempre se auto-recupera sem ação do operador. O lado ruim é que uma falha *permanente* (um token Telegram revogado, intents privilegiados Discord faltando) parece idêntica a um blip: "retrying", para sempre.

Dois mecanismos tornam falhas permanentes visíveis:

- **Classificação terminal.** Falhas cujo *tipo* de exceção prova que nunca podem se auto-recuperar — tokens rejeitados/revogados (`telegram_auth_error`, `discord_auth_error`, `email_auth_error`), intents privilegiados faltando (`discord_intents_required`), um sidecar Photon cujas dependências não instalam (`SIDECAR_DEPS_MISSING`) ou cujo binário node está ausente (`SIDECAR_NODE_MISSING`) — são marcadas como fatais em vez de entrar na fila de retry. A classificação é estritamente baseada em tipo; erros ambíguos sempre continuam retentando.
- **Escalação needs-attention.** Uma plataforma continuamente na fila de retry além de `agent.reconnect_attention_after` (padrão `7200` segundos = 2 horas, `0` desabilita) recebe `needs_attention: true` e um timestamp `retrying_since` no status de runtime do gateway (`hermes status`), mais um log WARNING. As retries continuam inalteradas — isto é um sinal, não um circuit breaker. A flag limpa no reconnect bem-sucedido.

```yaml
agent:
  reconnect_attention_after: 7200   # seconds; 0 disables the escalation flag
```

## Cache de agente do gateway {#gateway-agent-cache}

O gateway mantém um agente por sessão para uma conversa reutilizar seu prefixo de prompt em cache em vez de reconstruir o prompt de sistema a cada turno. Esse agente em cache também segura a transcrição completa da sessão — saída de ferramenta incluída, que são dezenas de megabytes em sessão com centenas de tool calls. Em gateway multi-plataforma ocupado o cache é portanto o maior consumidor único de memória no processo.

```yaml
agent:
  agent_cache:
    max_size: 128            # LRU entry cap
    idle_ttl_secs: 3600      # evict an agent idle this long
    memory_high_mb: auto     # anon-RSS budget; number, "auto", or 0/off
    max_evictions_per_pass: 16
    protect_recent: 8
```

`max_size` e `idle_ttl_secs` limitam o cache por contagem e tempo. Nenhum sabe quantos bytes segura, então `memory_high_mb` adiciona um terceiro limite: uma vez que a memória residente anônima do próprio gateway cruza o orçamento, ele descarta transcrições least-recently-used, que recarregam da sessão armazenada no próximo turno. Diminua se o gateway compete por memória com outros serviços; aumente (ou defina `0` para desligar o passe) se preferir manter todo prefixo warm.

`auto` deriva o orçamento do limite de memória sob o qual o gateway realmente roda — limite cgroup para container ou unit systemd, RAM total caso contrário — então um `MemoryMax`/`MemoryHigh` na unit é respeitado sem um segundo número para manter em sync.

Sessões mid-turn, as `protect_recent` mais recentemente usadas e qualquer sessão cuja transcrição não terminou de ser escrita em disco nunca são descartadas. Eviction é logada em WARNING com RSS medido e sessões dropadas:

```
Agent cache pressure: anon RSS 6802MB over budget 6656MB — evicting 5 LRU session(s): ...
```

## Context engine {#context-engine}

O context engine controla como conversas são gerenciadas ao se aproximar do limite de tokens do modelo. O engine `compressor` embutido usa summarização lossy (veja [Context Compression](/developer-guide/context-compression-and-caching)). Engines plugin podem substituí-lo por estratégias alternativas.

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
```

Para usar um engine plugin (ex.: LCM para gerenciamento lossless de contexto):

```yaml
context:
  engine: "lcm"          # must match the plugin's name
```

Engines plugin **nunca são auto-ativados** — você deve definir explicitamente `context.engine` para o nome do plugin. Engines disponíveis podem ser navegados e selecionados via `hermes plugins` → Provider Plugins → Context Engine.

Veja [Memory Providers](/user-guide/features/memory-providers) para o sistema análogo de seleção única para plugins de memória.

## Orçamento de iterações {#iteration-budget}

Quando o agente trabalha em tarefa complexa com muitas tool calls, pode queimar o orçamento de iterações (padrão: 500 turnos). O Hermes **não** injeta avisos de pressão mid-task — builds anteriores avisavam o modelo em 70%/90% do orçamento, o que fazia modelos abandonarem tarefas complexas prematuramente e foi removido em abril de 2026.

Em vez disso, quando o orçamento esgota de fato (500/500), o Hermes injeta uma mensagem pedindo ao modelo para encerrar e permite uma **grace call** única para entregar resposta final. Se essa grace call ainda não produz texto, o agente é pedido para resumir o que realizou.

```yaml
agent:
  max_turns: none              # Iterations per conversation turn (default: none = unlimited)
                               # Set a positive integer to cap; "none"/"null"/
                               # "unlimited"/"inf"/"infinity"/"infinite"/0/-1 = no limit
  api_max_retries: 3           # Retries per provider before fallback engages (default: 3)
```

`agent.max_turns` é **ilimitado por padrão** — o cap de turns causava mais problemas do que resolvia (truncagem silenciosa mid-task), então out of the box o Hermes roda um conversation turn até completar. Para impor um cap, defina um inteiro positivo. Para ser explícito sobre "sem limite", qualquer uma destas grafias case-insensitive funciona: `"none"`, `"null"`, `"unlimited"`, `"infinite"`, `"infinity"`, `"inf"`, `0`, `-1` (resolvem para um sentinel `sys.maxsize` para o loop nunca sair por contagem de turns).

`agent.api_max_retries` controla quantas vezes o Hermes retenta uma chamada de API do provedor em erros transitórios (rate limits, quedas de conexão, 5xx) **antes** do fallback-provider switching engajar. O padrão é `3` — quatro tentativas no total. Se tem [fallback providers](/user-guide/features/fallback-providers) configurados e quer failover mais rápido, diminua para `0` para o primeiro erro transitório no primário passar imediatamente ao fallback em vez de churn de retries no endpoint instável.

## Orçamento wall-clock de run {#wall-clock-run-budget}

Separado do orçamento de iterações, você pode dar a cada conversation run um orçamento **wall-clock** opcional. Isso é pensado para invocações one-shot e eval-harness que rodam sob um teto externo duro (ex. limite de 900 segundos por task): sem ele, uma run pode dar timeout com o trabalho essencialmente feito — uma geração aquém de emitir a resposta final, ou presa numa única chamada de provider hung.

```yaml
agent:
  run_budget_seconds: null     # Optional; unset/null = feature fully off (default)
```

Ou por invocação via CLI:

```bash
hermes chat --run-budget 850 -q "..."
```

Quando um orçamento está definido, duas coisas acontecem:

1. **Aviso de wrap-up a 80%.** Quando 80% do orçamento passou, o Hermes injeta um aviso **one-time** (entregue de forma cache-safe, anexado ao tool result mais novo como mensagens `/steer`) dizendo ao modelo para parar trabalho novo de discovery/verification e produzir o deliverable final a partir do estado que já tem. Dispara no máximo uma vez por run e espelha o mecanismo existente de wrap-up do orçamento de iterações — não há avisos de pressão repetidos.
2. **Timeouts stale scaled pelo deadline.** Timeouts stale implícitos non-streaming (o default de 90s e os floors de modelos de reasoning, ex. 600s para modelos DeepSeek reasoning) são capped em `max(60, remaining_budget × 0.5)` para que uma única chamada de provider silenciosamente hung nunca consuma o resto da run. O cap só *aperta* o timeout — nunca o eleva — e um `stale_timeout_seconds` explicitamente configurado (config de provider/modelo ou `HERMES_API_CALL_STALE_TIMEOUT`) sempre vence intacto.

O orçamento é por turn `run_conversation` (reseta a cada mensagem do usuário) e a feature fica completamente dormente quando unset — sem leituras de clock, sem injeção, sem mudanças de timeout.

## Verify-on-Stop (verificação de código) {#verify-on-stop-coding-verification}

Quando habilitado, o Hermes recusa aceitar resposta final em turno onde o agente editou código em workspace mas produziu nenhuma evidência de verificação fresca (teste passando, build, lint, etc.) — injeta follow-up sintético pedindo ao agente verificar ou explicar por que não pode. Edições só doc/markdown/skill nunca disparam, e o loop é limitado para nunca prender o agente.

```yaml
agent:
  verify_on_stop: false        # true | false | "auto" (surface-aware: on for CLI/TUI/desktop, off for messaging)
  verify_guidance: true        # Append creative-UI / clean-diff guidance to the missing-evidence nudge
  max_verify_nudges: 3         # Cap on consecutive continue nudges per turn (built-in + pre_verify hooks)
  coding_instructions: ""      # Standing project-wide coding rules appended to the coding brief
```

`verify_on_stop` aceita `true` (ligado em todo lugar), `false` (desligado — o padrão) ou `"auto"` (comportamento legado ciente da superfície: ligado para superfícies interativas de coding — CLI, TUI, desktop — e chamadores programáticos; desligado para superfícies de mensagens como Telegram/Discord onde a narrativa de verificação soa como ruído de chat). Off é o padrão em todo lugar: instalações novas vêm com `false` e a migração de config desligou em instalações existentes, então habilitar é um opt-in explícito. A env var `HERMES_VERIFY_ON_STOP` sobrescreve o valor de config quando definida.

Para um gate de política user/plugin no mesmo ponto — manter o agente indo com suas próprias checagens — veja o [hook `pre_verify`](/user-guide/features/hooks#pre_verify).

## Objetivos permanentes (`/goal`) {#standing-goals-goal}

Quando um objetivo permanente está ativo, o Hermes julga se cada resposta do assistente o satisfaz. Se não, alimenta um prompt de continuação de volta na mesma sessão e continua trabalhando até o objetivo terminar, o orçamento de turnos esgotar ou o usuário pausar/limpar. O orçamento de turnos é o backstop real — falhas do judge falham **open** (continuar) para um judge instável nunca emperrar progresso.

```yaml
goals:
  max_turns: 20   # Max continuation turns before Hermes auto-pauses the goal (default: 20)
```

`max_turns` limita quantos turnos de continuação um objetivo pode conduzir antes do Hermes auto-pausá-lo e pedir `/goal resume`. Protege contra false negatives do judge (objetivo feito mas judge diz continuar) e gasto ilimitado do modelo em objetivos fuzzy ou inatingíveis. Veja [Goals](/user-guide/features/goals) para o recurso completo.

### Timeouts de API {#api-timeouts}

O Hermes tem camadas de timeout separadas para streaming, mais detector stale para chamadas não-streaming. Detectores stale auto-ajustam para provedores locais só quando deixados nos padrões implícitos.

| Timeout | Padrão | Provedores locais | Config / env |
|---------|---------|----------------|--------------|
| Socket read timeout | 120s | Auto-elevado para 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Stale stream detection | 180s | Elevado para teto 900s (`agent.local_stream_stale_timeout`) | `HERMES_STREAM_STALE_TIMEOUT` |
| Stale non-stream detection | 90s | Auto-desabilitado quando deixado implícito | `providers.<id>.stale_timeout_seconds` or `HERMES_API_CALL_STALE_TIMEOUT` |
| API call (non-streaming) | 1800s | Inalterado | `providers.<id>.request_timeout_seconds` / `timeout_seconds` or `HERMES_API_TIMEOUT` |

O **socket read timeout** controla quanto httpx espera pelo próximo chunk de dados do provedor. LLMs locais podem levar minutos para prefill em contextos grandes antes do primeiro token, então o Hermes eleva para 30 minutos quando detecta endpoint local. Se definir explicitamente `HERMES_STREAM_READ_TIMEOUT`, esse valor é sempre usado independentemente da detecção de endpoint.

A **stale stream detection** mata conexões que recebem pings SSE keep-alive mas nenhum conteúdo real. Para provedores locais (que não enviam keep-alive pings durante prefill) o padrão é elevado para teto finito de 900 segundos em vez da base 180s — configurável via `agent.local_stream_stale_timeout` ou env var `HERMES_LOCAL_STREAM_STALE_TIMEOUT`.

A **stale non-stream detection** mata chamadas não-streaming que não produzem resposta por tempo demais. Por padrão o Hermes desabilita isso em endpoints locais para evitar false positives durante prefills longos. Se definir explicitamente `providers.<id>.stale_timeout_seconds`, `providers.<id>.models.<model>.stale_timeout_seconds` ou `HERMES_API_CALL_STALE_TIMEOUT`, esse valor explícito é honrado mesmo em endpoints locais.

Este orçamento limita toda chamada não-streaming, incluindo as que jobs cron e subagentes delegados rodam inline. Um provedor que aceita requisição e depois fica silencioso — conexão aberta, sem bytes, sem erro — é abortado no stale timeout e retentado, em vez de pendurar até o socket read timeout muito mais longo (ou, para execução cron não supervisionada, até algo externo matar o processo).

## Avisos de pressão de contexto {#context-pressure-warnings}

Separado da pressão de orçamento de iterações, pressão de contexto rastreia quão perto a conversa está do **threshold de compactação** — o ponto onde compressão de contexto dispara para resumir mensagens antigas. Isso ajuda você e o agente a entender quando a conversa está ficando longa.

| Progresso | Nível | O que acontece |
|----------|-------|-------------|
| **≥ 60%** até threshold | Info | CLI mostra barra cyan; gateway envia aviso informativo |
| **≥ 85%** até threshold | Warning | CLI mostra barra amarela bold; gateway avisa que compactação é iminente |

Na CLI, pressão de contexto aparece como barra de progresso no feed de saída de ferramentas:

```
  ◐ context ████████████░░░░░░░░ 62% to compaction  48k threshold (50%) · approaching compaction
```

Em plataformas de mensagens, uma notificação em texto simples é enviada:

```
◐ Context: ████████████░░░░░░░░ 62% to compaction (threshold: 50% of window).
```

Se auto-compressão está desabilitada, o aviso diz que contexto pode ser truncado.

Pressão de contexto é automática — nenhuma config necessária. Dispara puramente como notificação ao usuário e não modifica o stream de mensagens nem injeta nada no contexto do modelo.

## Estratégias de credential pool {#credential-pool-strategies}

Quando tem múltiplas chaves de API ou tokens OAuth para o mesmo provedor, configure a estratégia de rotação:

```yaml
credential_pool_strategies:
  openrouter: round_robin    # cycle through keys evenly
  anthropic: least_used      # always pick the least-used key
```

Opções: `fill_first` (padrão), `round_robin`, `least_used`, `random`. Veja [Credential Pools](/user-guide/features/credential-pools) para documentação completa.

## Prompt caching {#prompt-caching}

O Hermes liga prompt caching cross-session automaticamente quando o provedor ativo suporta — nenhuma config de usuário necessária.

Para Claude em **Anthropic nativo**, **OpenRouter** e **Nous Portal**, o Hermes anexa breakpoints `cache_control` com TTL de 1 hora (`ttl: "1h"`) no prompt de sistema e blocos de skill. O primeiro envio dentro de uma hora fresca paga taxas de input completas; envios subsequentes em qualquer sessão dentro da mesma hora puxam do cache na taxa discounted de cached-read. Isso significa que prompt de sistema, conteúdo de skill carregado e a porção inicial de qualquer include de contexto longo são reutilizados entre sessões `hermes` e entre subagentes forked na primeira hora.

O upstream Qwen Cloud (Alibaba DashScope) limita cache TTL a 5 minutos, então o Hermes usa breakpoint TTL de 5 minutos lá. Outros caminhos Claude-via-terceiros (AWS Bedrock, Azure Foundry) caem para defaults de caching do provedor. xAI Grok usa mecanismo separado de conversation-id pinned por sessão — veja [xAI prompt caching](/integrations/providers#xai-grok--responses-api--prompt-caching).

Nenhum knob existe para desabilitar — caching é always-on e economiza dinheiro mesmo em conversas single-turn porque só o prompt de sistema já é fração significativa da contagem de input tokens.

O único knob explícito é o tier de cache TTL que o Hermes solicita em breakpoints estilo Anthropic:

```yaml
prompt_caching:
  cache_ttl: "5m"   # "5m" or "1h" (Anthropic-supported tiers); other values are ignored
```

`cache_ttl` seleciona o breakpoint TTL que o Hermes anexa para Claude via API Anthropic nativa, OpenRouter e Nous Portal. Apenas os dois tiers suportados pela Anthropic (`"5m"`, `"1h"`) são honrados — qualquer outro valor é ignorado. Provedores com seus próprios caps (ex.: Qwen Cloud, que maxima em 5 minutos) ainda limitam ao que o upstream permite.

## Modelos auxiliares {#auxiliary-models}

O Hermes usa modelos "auxiliares" para tarefas laterais como análise de imagem, summarização de página web, análise de screenshot de browser, geração de título de sessão e compressão de contexto. Por padrão (`auxiliary.*.provider: "auto"`), o Hermes roteia toda tarefa auxiliar ao **modelo de chat principal** — o mesmo provider/model que você escolheu em `hermes model`. Você não precisa configurar nada para começar, mas esteja ciente de que em modelos de raciocínio caros (Opus, MiniMax M2.7, etc.) tarefas auxiliares somam custo significativo. Se quer tarefas laterais baratas e rápidas independentemente do modelo principal, defina `auxiliary.<task>.provider` e `auxiliary.<task>.model` explicitamente (por exemplo, Gemini Flash no OpenRouter para vision e web extraction).

:::note Por que "auto" usa seu modelo principal
Builds anteriores separavam usuários de agregador (OpenRouter, Nous Portal) para um default barato do lado do provedor. Isso era surpreendente — usuários que pagaram assinatura de agregador viam modelo diferente tratando tráfego auxiliar. `auto` agora usa o modelo principal para todos, e sobrescritas por task em `config.yaml` ainda vencem (veja [Referência completa de config auxiliar](#full-auxiliary-config-reference) abaixo).
:::

### Configurando modelos auxiliares interativamente {#configuring-auxiliary-models-interactively}

Em vez de editar YAML manualmente, execute `hermes model` e escolha **"Configure auxiliary models"** no menu. Você terá um picker interativo por task:

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
[ ] delegation           currently: auto / inherit main agent
```

Selecione uma task, escolha provedor (fluxos OAuth abrem browser; provedores API-key pedem), escolha modelo. A mudança persiste em `auxiliary.<task>.*` em `config.yaml`. Mesma maquinaria do picker de modelo principal — nenhuma sintaxe extra para aprender.

A entrada **Delegation** é especial: ela roteia o modelo usado por subagentes `delegate_task` e persiste na seção top-level `delegation.*` (`delegation.provider` / `delegation.model`) em vez de `auxiliary.*`, porque subagentes são agentes filhos completos, não chamadas LLM laterais. Seu `auto` significa "herdar provider, modelo e credenciais do agente pai."

Se não quiser que o Hermes auto-gere títulos após a primeira troca, defina
`auxiliary.title_generation.enabled: false`. Títulos manuais ainda funcionam via
`/title` e `hermes sessions rename`.

### Endpoints stream-only {#stream-only-endpoints}

Alguns endpoints OpenAI-compatible rejeitam requisições de chat não-streaming outright (ex.: Tencent Copilot retorna HTTP 400 `"Non-stream chat request is currently not supported"`). Chat interativo já faz stream, mas tarefas auxiliares (title generation, compression, web extraction) usam chamadas não-streaming e falhariam a cada tentativa. O Hermes sempre trata `copilot.tencent.com` como stream-only; para qualquer outro endpoint assim, liste um substring de URL em `auxiliary.stream_only_base_urls`:

```yaml
auxiliary:
  stream_only_base_urls:
    - "my-stream-only-proxy.example.com"
```

Chamadas auxiliares correspondentes são enviadas com `stream=True` e chunks (incluindo deltas de tool-call) são agregados client-side — nenhuma mudança de comportamento para qualquer outro endpoint.

### Tutorial em vídeo {#video-tutorial}

<div style={{position: 'relative', width: '100%', aspectRatio: '16 / 9', marginBottom: '1.5rem'}}>
  <iframe
    src="https://www.youtube.com/embed/NoF-YajElIM"
    title="Hermes Agent — Auxiliary Models Tutorial"
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', border: 0}}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  />
</div>

### O padrão universal de config {#the-universal-config-pattern}

Todo slot de modelo no Hermes — tarefas auxiliares, compression, fallback — usa os mesmos três knobs:

| Chave | O que faz | Padrão |
|-----|-------------|---------|
| `provider` | Qual provedor usar para auth e roteamento | `"auto"` |
| `model` | Qual modelo solicitar | default do provedor |
| `base_url` | Endpoint OpenAI-compatible customizado (sobrescreve provider) | não definido |

Blocos de tarefa auxiliar aceitam adicionalmente um knob `reasoning_effort`:

| Chave | O que faz | Padrão |
|-----|-------------|---------|
| `reasoning_effort` | Nível de thinking para chamadas LLM dessa task: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` | não definido (default do provedor) |

Este é o counterpart por task do `agent.reasoning_effort` global: rode compression em `low` ou vision em `none` para cortar latência e custo de side-task quando seu modelo principal é um modelo de raciocínio caro, sem tocar comportamento de chat principal. Funciona em todo bloco de tarefa auxiliar (`vision`, `web_extract`, `compression`, `title_generation`, `curator`, `background_review`, ...), em todos os três formatos wire auxiliares (chat completions, Codex Responses, Anthropic Messages). Um `extra_body.reasoning` explícito na mesma task vence sobre o shorthand.

MoA é a exceção: profundidade de raciocínio para Mixture-of-Agents é configurada **por slot** no preset MoA (`moa.presets.<name>.reference_models[].reasoning_effort` / `aggregator.reasoning_effort`), não nos blocos auxiliares `moa_reference`/`moa_aggregator` — veja [Mixture of Agents](/user-guide/features/mixture-of-agents).

```yaml
auxiliary:
  compression:
    reasoning_effort: "low"    # summaries don't need deep thinking
  vision:
    reasoning_effort: "none"   # disable thinking for image description
```

Quando `base_url` está definido, o Hermes ignora o provider e chama esse endpoint diretamente (usando `api_key` ou `OPENAI_API_KEY` para auth). Quando só `provider` está definido, o Hermes usa auth e base URL embutidos desse provider.

Provedores disponíveis para tarefas auxiliares: `auto`, `main`, mais qualquer provedor no [provider registry](/reference/environment-variables) — `openrouter`, `nous`, `openai-codex`, `copilot`, `copilot-acp`, `anthropic`, `gemini`, `qwen-oauth`, `zai`, `kimi-coding`, `kimi-coding-cn`, `minimax`, `minimax-cn`, `minimax-oauth`, `deepseek`, `nvidia`, `xai`, `xai-oauth`, `ollama-cloud`, `alibaba`, `bedrock`, `huggingface`, `arcee`, `xiaomi`, `kilocode`, `opencode-zen`, `opencode-go`, `opencode-free`, `commandcode`, `commandcode-anthropic`, `ai-gateway`, `azure-foundry` — ou qualquer provedor custom nomeado do seu dict `providers:` (ex.: `provider: "beans"`).

:::tip MiniMax OAuth
`minimax-oauth` faz login via browser OAuth (sem API key). Execute `hermes model` e selecione **MiniMax (OAuth)** para autenticar. Tarefas auxiliares usam `MiniMax-M2.7-highspeed` automaticamente. Veja o [guia MiniMax OAuth](../guides/minimax-oauth.md).
:::

:::tip xAI Grok OAuth
`xai-oauth` faz login via browser OAuth para assinantes SuperGrok e X Premium+ (sem API key). Execute `hermes model` e selecione **xAI Grok OAuth (SuperGrok / Premium+)** para autenticar. O mesmo token OAuth é reutilizado para toda superfície direct-to-xAI (chat, tarefas auxiliares, TTS, image gen, video gen, transcription). Veja o [guia xAI Grok OAuth](../guides/xai-grok-oauth.md), e se o Hermes está em host remoto veja [OAuth over SSH / Remote Hosts](../guides/oauth-over-ssh.md).
:::

:::warning `"main"` é só para tarefas auxiliares
A opção de provider `"main"` significa "usar qualquer provider meu agente principal usa" — só é válida dentro de `auxiliary:`, `compression:` e entradas de fallback primário (`fallback_providers:` ou legado `fallback_model:`). **Não** é valor válido para sua config top-level `model.provider`. Se usa endpoint OpenAI-compatible customizado, defina `provider: custom` na seção `model:`. Veja [AI Providers](/integrations/providers) para todas as opções de provider de modelo principal.
:::

### Referência completa de config auxiliar {#full-auxiliary-config-reference}

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
    # max_concurrency: 2       # Optional: cap simultaneous compression LLM calls so
                               # multiple sessions don't pile retries on a degraded provider

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

  # Auto-generated short session titles after the first exchange
  title_generation:
    provider: "auto"
    model: ""
    base_url: ""
    api_key: ""
    timeout: 30
    # max_concurrency: 2       # Optional: cap simultaneous title-generation calls

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
Cada tarefa auxiliar tem `timeout` configurável (em segundos). Padrões: vision 120s, web_extract 360s, approval 30s, compression 120s. Aumente se usa modelos locais lentos para tarefas auxiliares. Vision também tem `download_timeout` separado (padrão 30s) para download HTTP de imagem — aumente para conexões lentas ou servidores de imagem self-hosted.
:::

:::info
Compressão de contexto tem seu próprio bloco `compression:` para thresholds e bloco `auxiliary.compression:` para config model/provider — veja [Context Compression](#context-compression) acima. A cadeia de fallback primária usa lista top-level `fallback_providers:` — veja [Fallback Providers](/integrations/providers#fallback-providers). Todos seguem o mesmo padrão provider/model/base_url.
:::

### Cadeia de fallback por task para tarefas auxiliares {#per-task-fallback-chain-for-auxiliary-tasks}

Cada tarefa auxiliar pode opcionalmente definir `fallback_chain` — lista de entradas provider/model que o Hermes tenta quando o provedor auxiliar primário falha por rate limits, problemas de conectividade ou restrições de pagamento:

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

Quando o provedor auxiliar primário (`openrouter` / `openai/gpt-4o-mini`) retorna rate-limit, timeout de conexão ou payment-required, o Hermes percorre `fallback_chain` em ordem. Pula entradas cujo provider corresponde ao já falho, e tenta cada entrada restante até uma ter sucesso ou a cadeia esgotar. Se todos fallbacks falham, o Hermes cai para o modelo principal do agente como rede de segurança final.

Cada entrada suporta os mesmos três knobs de qualquer config de tarefa auxiliar:

| Chave | Descrição |
|-----|-------------|
| `provider` | Nome do provider (`nous`, `openrouter`, `anthropic`, `gemini`, `main`, etc.) |
| `model` | Nome do modelo para esse provider |
| `base_url` | (Opcional) Endpoint OpenAI-compatible customizado |

`fallback_chain` está disponível em qualquer tarefa auxiliar — `compression`, `vision`, `web_extract`, `approval`, `skills_hub`, `mcp`, etc.

### Limitando concorrência auxiliar {#limiting-auxiliary-concurrency}

`max_concurrency` limita chamadas LLM in-flight para tarefas auxiliares como `compression` e `title_generation` em todo o processo. `auxiliary.vision.max_concurrency` é excluído: já controla só workers CPU-bound de encode/resize de imagem do vision, não requisições LLM. Mais útil quando:

- Muitas sessões podem spawnar trabalho em segundo plano simultaneamente (canais Discord/Telegram, vários terminais)
- Seu provider é rate-limited ou passando por incidente e retries amplificariam o burst

O padrão é ilimitado. Um cap de segurança típico é `2`:

```yaml
auxiliary:
  title_generation:
    max_concurrency: 2
  compression:
    max_concurrency: 2
```

O semáforo envolve a chamada inteira incluindo retries e fallbacks, então uma chamada lenta conta só uma vez para o limite.

### OpenRouter routing e Pareto Code para tarefas auxiliares {#openrouter-routing--pareto-code-for-auxiliary-tasks}

Quando uma tarefa auxiliar resolve para OpenRouter (explicitamente ou via `provider: "main"` enquanto seu agente principal está no OpenRouter), as config `provider_routing` e `openrouter.min_coding_score` do agente principal **não propagam** — por design, cada tarefa auxiliar é independente. Para definir preferências de provider OpenRouter ou usar o [Pareto Code router](/integrations/providers#openrouter-pareto-code-router) para uma task aux específica, defina por task via `extra_body`:

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

A forma espelha o que OpenRouter aceita no body de requisição chat completions. O Hermes encaminha o `extra_body` inteiro verbatim, então qualquer outro campo documentado em [openrouter.ai/docs](https://openrouter.ai/docs) funciona igual.

### Alterando o modelo Vision {#changing-the-vision-model}

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

### Opções de provider {#provider-options}

Estas opções aplicam-se a **configs de tarefa auxiliar** (`auxiliary:`, `compression:`) e entradas de fallback primário (`fallback_providers:` ou legado `fallback_model:`), não à sua config `model.provider` principal.

| Provider | Descrição | Requisitos |
|----------|-------------|-------------|
| `"auto"` | Melhor disponível (padrão). Vision tenta OpenRouter → Nous → Codex. | — |
| `"openrouter"` | Força OpenRouter — roteia a qualquer modelo (Gemini, GPT-4o, Claude, etc.) | `OPENROUTER_API_KEY` |
| `"nous"` | Força Nous Portal | `hermes auth` |
| `"codex"` | Força Codex OAuth (conta ChatGPT). Suporta vision (gpt-5.3-codex). | `hermes model` → ChatGPT or Codex Subscription |
| `"minimax-oauth"` | Força MiniMax OAuth (login browser, sem API key). Usa MiniMax-M2.7-highspeed para tarefas auxiliares. | `hermes model` → MiniMax (OAuth) |
| `"xai-oauth"` | Força xAI Grok OAuth (login browser para assinantes SuperGrok ou X Premium+, sem API key). Mesmo token OAuth cobre chat, TTS, image, video e transcription. | `hermes model` → xAI Grok OAuth (SuperGrok / Premium+) |
| `"main"` | Usa seu endpoint custom/main ativo. Pode vir de `OPENAI_BASE_URL` + `OPENAI_API_KEY` ou endpoint custom salvo via `hermes model` / `config.yaml`. Funciona com OpenAI, modelos locais ou qualquer API OpenAI-compatible. **Só tarefas auxiliares — não válido para `model.provider`.** | Credenciais de endpoint custom + base URL |

Provedores direct API-key do catálogo principal também funcionam aqui quando quer side tasks bypassando seu router padrão. Por exemplo, `gmi` é válido quando `GMI_API_KEY` está configurada, e `fireworks` é válido quando `FIREWORKS_API_KEY` está configurada:

```yaml
auxiliary:
  compression:
    provider: "gmi"
    model: "anthropic/claude-opus-4.6"
```

Para roteamento auxiliar GMI, use o ID exato de modelo retornado pelo endpoint `/v1/models` da GMI. IDs de modelo Fireworks usam a forma slash nativa do provider, por exemplo `accounts/fireworks/models/glm-5p2`.

### Configurações comuns {#common-setups-1}

**Usando endpoint custom direto** (mais claro que `provider: "main"` para APIs local/self-hosted):
```yaml
auxiliary:
  vision:
    base_url: "http://localhost:1234/v1"
    api_key: "local-key"
    model: "qwen2.5-vl"
```

`base_url` tem precedência sobre `provider`, então esta é a forma mais explícita de rotear uma tarefa auxiliar a um endpoint específico. Para sobrescritas de endpoint direto, o Hermes usa `api_key` configurada ou cai para `OPENAI_API_KEY`; não reutiliza `OPENROUTER_API_KEY` para esse endpoint custom.

**Usando chave OpenAI API para vision:**
```yaml
# In ~/.hermes/.env:
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_API_KEY=sk-...

auxiliary:
  vision:
    provider: "main"
    model: "gpt-4o"       # or "gpt-4o-mini" for cheaper
```

**Usando OpenRouter para vision** (roteie a qualquer modelo):
```yaml
auxiliary:
  vision:
    provider: "openrouter"
    model: "openai/gpt-4o"      # or "google/gemini-2.5-flash", etc.
```

**Usando Codex OAuth** (conta ChatGPT Pro/Plus — sem API key):
```yaml
auxiliary:
  vision:
    provider: "codex"     # uses your ChatGPT OAuth token
    # model defaults to gpt-5.3-codex (supports vision)
```

**Usando MiniMax OAuth** (login browser, sem API key):
```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```
Execute `hermes model` e selecione **MiniMax (OAuth)** para login e definir automaticamente. Para região China, a base URL será `https://api.minimaxi.com/anthropic`. Veja o [guia MiniMax OAuth](../guides/minimax-oauth.md) para o walkthrough completo.

**Usando modelo local/self-hosted:**
```yaml
auxiliary:
  vision:
    provider: "main"      # uses your active custom endpoint
    model: "my-local-model"
```

`provider: "main"` usa qualquer provider que o Hermes usa para chat normal — seja um provider custom nomeado (ex.: `beans`), um built-in como `openrouter`, ou endpoint legado `OPENAI_BASE_URL`.

:::tip
Se usa Codex OAuth como provider de modelo principal, vision funciona automaticamente — nenhuma config extra necessária. Codex está incluído na cadeia de auto-detecção para vision.
:::

:::warning
**Vision requer modelo multimodal.** Se definir `provider: "main"`, certifique-se de que seu endpoint suporta multimodal/vision — caso contrário análise de imagem falhará.
:::

### Variáveis de ambiente (legado) {#environment-variables-legacy}

Modelos auxiliares também podem ser configurados via variáveis de ambiente. Porém, `config.yaml` é o método preferido — mais fácil de gerenciar e suporta todas as opções incluindo `base_url` e `api_key`.

| Configuração | Variável de ambiente |
|---------|---------------------|
| Vision provider | `AUXILIARY_VISION_PROVIDER` |
| Vision model | `AUXILIARY_VISION_MODEL` |
| Vision endpoint | `AUXILIARY_VISION_BASE_URL` |
| Vision API key | `AUXILIARY_VISION_API_KEY` |
| Web extract provider | `AUXILIARY_WEB_EXTRACT_PROVIDER` |
| Web extract model | `AUXILIARY_WEB_EXTRACT_MODEL` |
| Web extract endpoint | `AUXILIARY_WEB_EXTRACT_BASE_URL` |
| Web extract API key | `AUXILIARY_WEB_EXTRACT_API_KEY` |

Configurações de compression e fallback model são só config.yaml.

:::tip
Execute `hermes config` para ver suas configurações atuais de modelo auxiliar. Sobrescritas só aparecem quando diferem dos padrões.
:::
## Reasoning effort {#reasoning-effort}

Controle quanto "thinking" o modelo faz antes de responder:

```yaml
agent:
  reasoning_effort: ""   # empty = medium. Options: none, minimal, low, medium, high, xhigh, max, ultra
```

Quando não definido (padrão), reasoning effort usa "medium" — nível equilibrado que funciona bem para a maioria das tarefas. Definir um valor sobrescreve — reasoning effort maior dá melhores resultados em tarefas complexas ao custo de mais tokens e latência.

:::note Modelos adaptive-thinking (Claude 4.6+, classe Fable/Mythos) via OpenRouter
Estes modelos usam thinking *adaptativo* e não aceitam o campo usual `reasoning.effort`
— OpenRouter o ignora para eles. O Hermes roteia transparentemente seu
`reasoning_effort` para o parâmetro `verbosity` do OpenRouter (que mapeia para
`output_config.effort` da Anthropic), então o mesmo knob de effort continua funcionando com
os níveis suportados pelo modelo selecionado. `none` (ou unset) deixa o modelo
no próprio default adaptativo. O
provider Anthropic nativo já controla effort diretamente e não é afetado.
:::

:::note Modelos OpenRouter e níveis de effort suportados
Para outros modelos roteados pelo OpenRouter, o Hermes lê os metadados de reasoning do catálogo live de modelos (`supported_parameters` + `reasoning.supported_efforts` por modelo) para decidir se envia controles de reasoning
e para limitar o effort pedido ao nível mais próximo que a rota realmente
suporta (sempre para baixo — ex.: `ultra` vira `high` numa rota que para
em `high`, nunca uma escalação silenciosa). Novos vendors com reasoning funcionam
automaticamente sem esperar um update do Hermes; quando o catálogo está
inacessível ou um modelo não está listado, o Hermes cai para sua lista embutida
de famílias de modelo e passa seu effort inalterado.
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

Mudanças de effort são escopadas à sessão por padrão; adicione `--global` para salvar o
novo nível como seu default `agent.reasoning_effort`.

#### Sobrescritas de reasoning por modelo {#per-model-reasoning-overrides}

Você pode definir níveis diferentes de reasoning effort para modelos diferentes. Útil quando quer reasoning alto para modelos complexos mas medium para os mais rápidos:

```yaml
agent:
  reasoning_effort: "medium"       # global default
  reasoning_overrides:
    "openrouter/anthropic/claude-opus-4.5": "xhigh"
    "openai/gpt-5": "low"
    "claude-sonnet-4.6": "high"    # bare model name also works
```

A correspondência de chave é **tolerante à grafia** — qualquer grafia razoável corresponde:
- `claude-opus-4.5`, `claude-opus-4-5`, `claude-opus.4.5` (pontos e hífens são intercambiáveis)
- `anthropic/claude-opus-4.5`, `openrouter/anthropic/claude-opus-4.5` (prefixo de provider opcional)
- Correspondências exatas têm precedência sobre variantes

:::note
Não há suporte `hermes config set` para chaves `reasoning_overrides` — edite o arquivo YAML diretamente. Isso porque nomes de modelo frequentemente contêm pontos (ex.: `claude-opus-4.5`), que conflitam com a sintaxe dotted-key da CLI.
:::

**Prioridade de resolução:**

1. Sobrescrita `/reasoning --session` com escopo de sessão (só gateway)
2. Sobrescrita por modelo de `agent.reasoning_overrides` (tolerante à grafia)
3. `agent.reasoning_effort` global
4. Default do provider

A sobrescrita aplica automaticamente em todo lugar: startup CLI, gateway de mensagens, Desktop/TUI, jobs cron, trocas mid-session `/model` e ativação de modelo fallback.

## Enforcement de uso de ferramentas {#tool-use-enforcement}

Alguns modelos ocasionalmente descrevem ações pretendidas como texto em vez de fazer tool calls ("I would run the tests..." em vez de chamar terminal de fato). Tool-use enforcement injeta orientação no prompt de sistema que direciona o modelo de volta a chamar ferramentas de fato.

```yaml
agent:
  tool_use_enforcement: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| Valor | Comportamento |
|-------|----------|
| `"auto"` (padrão) | Habilitado para modelos que correspondem: `gpt`, `codex`, `gemini`, `gemma`, `grok`, `glm`, `qwen`, `deepseek`. Desabilitado para todos os outros (ex.: Claude). |
| `true` | Sempre habilitado, independentemente do modelo. Útil se notar seu modelo atual descrevendo ações em vez de executá-las. |
| `false` | Sempre desabilitado, independentemente do modelo. |
| `["gpt", "codex", "qwen", "llama"]` | Habilitado só quando o nome do modelo contém um dos substrings listados (case-insensitive). |

### O que injeta {#what-it-injects}

Quando habilitado, duas camadas de orientação podem ser adicionadas ao prompt de sistema:

1. **Enforcement geral de uso de ferramentas** (todos modelos correspondidos) — instrui o modelo a fazer tool calls imediatamente em vez de descrever intenções, continuar trabalhando até a tarefa completar e nunca terminar um turno prometendo ação futura.

2. **Orientação operacional Google** (só modelos Gemini e Gemma) — concisão, caminhos absolutos, tool calls paralelas e padrões verify-before-edit.

São transparentes ao usuário e só afetam o prompt de sistema. Modelos que já usam ferramentas de forma confiável (como Claude) não precisam desta orientação, por isso `"auto"` os exclui.

### Quando ligar {#when-to-turn-it-on}

Se usa um modelo fora da lista auto padrão e nota que frequentemente descreve o que *faria* em vez de fazer, defina `tool_use_enforcement: true` ou adicione o substring do modelo à lista:

```yaml
agent:
  tool_use_enforcement: ["gpt", "codex", "gemini", "grok", "my-custom-model"]
```

## Orientação de execution-discipline {#execution-discipline-guidance}

Separadamente do tool-use enforcement, o Hermes injeta um bloco de **execution-discipline** para famílias de modelo que compartilham um conjunto de modos de falha agentic observados em traces de eval: fazer aritmética em prosa em vez de código, pular verificação de read-back depois de writes externos, "reparar" identificadores malformados, reivindicar completude apesar de mismatches de contagem, e declarar "done" sem verificar todo critério de aceitação.

```yaml
agent:
  execution_guidance: "auto"   # "auto" | true | false | ["model-substring", ...]
```

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Habilitado para modelos que batem: `gpt`, `codex`, `grok`, `deepseek`, `kimi`, `qwen`, `glm`, `minimax`, `mimo`, `mistral`. |
| `true` | Sempre habilitado, independentemente do modelo. |
| `false` | Sempre desabilitado, independentemente do modelo. |
| `["deepseek", "my-custom-model"]` | Habilitado só quando o nome do modelo contém um dos substrings listados (case-insensitive). |

O bloco injetado cobre:

- **Persistência de tool** — continue chamando tools até a tarefa estar completa *e* verificada; retente resultados de lookup vazios, parciais ou suspeitosamente estreitos com uma query mais ampla ou diferente antes de concluir.
- **Uso obrigatório de tool** — aritmética, hashes, datas, estado do sistema e fatos de arquivo sempre vêm de uma tool, nunca de computação mental.
- **Read-back de write externo** — depois de qualquer write que muda estado num sistema externo, leia de volta o target exato antes de reivindicar sucesso (edits internos de arquivo que uma tool já confirmou não são re-verificados).
- **Reconciliação de contagem** — totais declarados (`total`, `reply_count`, `has_more`) são asserções duras; em mismatch, re-fetch ou parse programaticamente.
- **Preservação literal** — nunca normalize ou "repare" identificadores que falham um formato declarado; um lookup bem-sucedido não valida um token de fonte malformado.
- **Completion gated por verificação** — "done" significa que todo critério de aceitação nomeado está verificado, nunca um subset plausível.

O gate é independente de `tool_use_enforcement` — qualquer um pode estar on sem o outro. A orientação é escolhida uma vez no início da sessão keyed no nome do modelo, então o system prompt permanece byte-stable (e friendly ao prompt-cache) pela vida da conversa. Gemini/Gemma são excluídos da lista auto porque recebem a orientação operacional Google mais específica; Claude é excluído porque não exibe esses modos de falha — opte qualquer modelo com `true` ou uma lista de substrings.

## Guardrails de tool-loop {#tool-loop-guardrails}

O Hermes detecta quando o agente está preso em loop improdutivo de tool-calling — mesma tool call falhando repetidamente, mesma ferramenta falhando uma atrás da outra, ou chamada idempotente retornando o mesmo resultado sem progresso. Por padrão injeta um **aviso** no resultado da ferramenta para o modelo se autocorrigir; não para hard-stop, já que alguém observando CLI/TUI pode intervir.

Para deploys gateway / servidor não supervisionados, habilite hard stops para um agente preso ser circuit-broken em vez de queimar o orçamento de iterações:

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
  loop_caps:
    max_web_searches: 50       # max web_search calls per turn (0 = unlimited)
    max_subagents: 50          # max subagents spawned per turn (0 = unlimited)
```

`hard_stop_enabled` padrão `false` porque sessões interativas têm humano no loop. Em deploys não supervisionados (gateway, cron, workers kanban) defina `true` para falhas repetidas serem bloqueadas em vez de só avisadas. Veja também [Docker / deploys não supervisionados](docker.md).

### Caps de loop runaway por turno {#per-turn-runaway-loop-caps}

Separado dos thresholds baseados em falha acima, `loop_caps` define tetos rígidos de quantas chamadas `web_search` e spawns de subagente um único loop de agente (turno) pode fazer. Contadores resetam no início de cada turno, então sessão multi-turn legítima nunca fica faminta — mas um turno que espirala em busca ou delegação ilimitada é parado. Estão sempre ligados e disparam independentemente de `hard_stop_enabled`. Um turno emitindo dezenas de buscas web ou spawnando dezenas de subagentes já é patológico, então os padrões são baixos. Quando um cap é atingido, a tool call ofensora é bloqueada com mensagem explicativa e o turno para limpo em vez de queimar o resto do orçamento. Defina qualquer valor em `0` para desabilitar aquele cap.

Um único lote `delegate_task` conta cada task em `max_subagents` (lote de 3 gasta 3), então o cap rastreia subagentes reais spawnados em vez de invocações `delegate_task`.

Isso espelha os caps por sessão WebSearch e subagent do Claude Code (v2.1.212), que também padrão 200 e resetam em `/clear`.

### Guards anti-stall em runtime {#runtime-anti-stall-guards}

Complementando os guardrails baseados em falha acima, `agent.stall_guards` (default `true`) habilita dois guards conservadores de runtime contra turns desperdiçados. Primeiro, um **identical-call loop breaker**: quando a mesma tool é chamada 3+ vezes consecutivas com argumentos idênticos *e* retorna um resultado idêntico, um aviso curto de uma linha é anexado àquele tool result dizendo ao modelo para não repetir a chamada — nunca bloqueia a chamada, e pollers legitimamente-repetíveis (`process`, `*_get_result`, `*_poll`) estão isentos. Segundo, uma **continue-intent recovery**: quando o modelo termina um turn sem tool calls mas sua reply curta termina anunciando uma ação ("Let me now update the file…"), o Hermes o re-prompta a agir via o mesmo mecanismo bounded de continuação usado para intent-ack recovery (máx. 2 re-prompts por turn). Ambos são cache-safe (avisos são adicionados na construção do result, nunca retroativamente) e podem ser desabilitados juntos:

```yaml
agent:
  stall_guards: false
```

O mesmo gate também habilita **result-reference stubbing**: quando uma tool call idêntica re-emitida retorna um resultado fresh byte-idêntico, o payload duplicado entra no contexto como um stub de referência curto apontando para o resultado anterior (nome da tool, `tool_call_id`, um resumo de args, e — se o primeiro resultado foi persistido em disco — seu path de spillover) em vez de repetir a saída completa. A tool ainda executa toda vez, então a semântica de polling é preservada: um resultado mudado sempre flui inteiro. Resultados sob 512 caracteres, resultados de erro e resultados multimodais nunca são stubbed, e pollers *são* stubbed (um poll inalterado é exatamente o caso em que o payload duplicado não carrega informação).

## Configuração TTS {#tts-configuration}

```yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper" | "deepinfra"
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

Isso controla tanto a ferramenta `text_to_speech` quanto respostas faladas no voice mode (`/voice tts` na CLI ou gateway de mensagens).

**Hierarquia de fallback de speed:** speed específico do provider (ex.: `tts.edge.speed`) → `tts.speed` global → default `1.0`. Defina `tts.speed` global para aplicar speed uniforme em todos providers, ou sobrescreva por provider para controle fino.

## Configurações de exibição {#display-settings}

```yaml
display:
  tool_progress: all      # off | new | all | verbose
  tool_progress_command: false  # Enable /verbose slash command in messaging gateway
  focus_view: false       # CLI focus view (/focus) — reduced output, display-only
  platforms: {}           # Per-platform display overrides (see below)
  interim_assistant_messages: true  # Gateway: send natural mid-turn assistant updates as separate messages
  show_commentary: true   # Codex models: deliver commentary-channel progress narration as visible mid-turn updates
  skin: default           # Built-in or custom CLI skin (see user-guide/features/skins)
  personality: ""         # Legacy cosmetic field still surfaced in some summaries
  compact: false          # Compact output mode (less whitespace)
  cli_multiline_shortcuts: true  # CLI: Ctrl+J, \ + Enter, and supported Shift+Enter insert newlines (false = legacy c-j submit fallback)
  resume_display: full    # full (show previous messages on resume) | minimal (one-liner only)
  bell_on_complete: false # Play terminal bell when agent finishes (great for long tasks)
  show_reasoning: true    # Show model reasoning/thinking above each response (default: true; toggle with /reasoning show|hide)
  streaming: false        # Stream tokens to terminal as they arrive (real-time output)
  show_cost: false        # Show estimated $ cost in the CLI status bar
  timestamps: false       # When true, prefixes user and assistant labels with timestamps in the CLI / TUI transcript
  timestamp_format: "%H:%M"  # strftime format for those timestamps (e.g. "%b-%d %H:%M" for month-day)
  tool_preview_length: 0  # Max chars for tool call previews (0 = no limit, show full paths/commands)
  turn_summary: true      # CLI only: print a one-line post-turn accounting footer after each interactive turn
  spinner_token_flow: true # CLI only: append live cumulative turn tokens to the spinner timer
  runtime_footer:         # Gateway: append a runtime-context footer to final replies
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true    # Append an advisory footer when write_file/patch calls failed this turn
  credits_notices: true   # Nous credits status-bar notices (usage bands, grant-spent, depleted). false = silence them; /usage still works
  cli_rebuild_scrollback_on_redraw: false  # Classic CLI: also wipe terminal scrollback (CSI 3J) on /redraw / Ctrl+L / width-change resize recovery. Enable when a terminal/tmux stack stamps stale prompt chrome into scrollback on maximize/restore.
  language: en            # UI language for static messages (approval prompts, some gateway replies). en | zh | zh-hant | ja | de | es | fr | tr | uk | af | ko | it | ga | pt | ru | hu
```

### Resumo por turno e fluxo de tokens no spinner {#per-turn-summary-and-spinner-token-flow}

`display.turn_summary` (padrão `true`) imprime uma linha contábil dim após cada turno **interativo da CLI**, resumindo o que aquele turno fez de fato:

```
⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands
```

A contagem é observada do feed tool-progress que a CLI já recebe, então não custa nada extra. Detalhes:

- Wall time é a duração real do turno (`2m05s` após um minuto).
- Tool calls são agrupadas por verbo (`edited`, `read`, `ran`, `searched`, …) com pluralização correta; tools plugin/MCP sem verbo curado caem em `called N tools`.
- Deltas de linha `+X -Y` aparecem só quando o resultado da ferramenta já reporta diff (atualmente `patch`). O Hermes nunca executa git para computá-los, então edição `write_file` é contada sem delta.
- **Tool calls falhas não são contadas** — escrita negada nunca renderiza como edição bem-sucedida (veja o [verificador de mutação de arquivo](#file-mutation-verifier) para o aviso complementar).
- Turnos longos limitam a quatro segmentos de verbo mais cauda `+N more` para a linha nunca quebrar.
- Turno rápido sem tool calls não imprime nada.

`display.spinner_token_flow` (padrão `true`) anexa tokens de saída cumulativos do turno em execução ao timer live do spinner da CLI:

```
  ⚡ Reading cli.py  (  2.3s · ↓ 1.2k tok)
```

A contagem é por turno (totais de sessão são baselined no início do turno) e atualiza conforme cada chamada de API no turno reporta usage. Nada renderiza antes do primeiro usage report chegar, então você nunca vê `↓ 0 tok` enganoso.

Ambas chaves são só display e só CLI: suprimidas em quiet mode, quando `display.tool_progress` é `off`, em execuções batch single-query/`-Q` e em superfícies gateway/messaging (essas usam `display.runtime_footer`). Defina qualquer chave em `false` para desligar.

### Verificador de mutação de arquivo {#file-mutation-verifier}

Quando `display.file_mutation_verifier` é `true` (padrão), o Hermes anexa aviso one-line à resposta final do assistente sempre que uma chamada `write_file` ou `patch` falhou durante o turno e nunca foi superseded por escrita bem-sucedida no mesmo caminho. Isso captura a classe "lote de patches paralelos, metade falha silenciosamente, modelo resume sucesso" sem exigir `git status` manual após cada edição.

Exemplo de rodapé:

```
⚠️ File-mutation verifier: 3 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • concepts/automatic-organization.md — [patch] Could not find match for old_string
  • concepts/lora.md — [patch] Could not find match for old_string
  • concepts/rag-pipeline.md — [patch] Could not find match for old_string
```

Defina `file_mutation_verifier: false` (ou `HERMES_FILE_MUTATION_VERIFIER=0`) para suprimir o rodapé. O verificador só dispara quando falhas reais estão pendentes no fim do turno — um modelo que retenta patch falho e tem sucesso no mesmo turno não dispara para aquele arquivo.

**Confie no verificador mais que no resumo do modelo.** O rodapé significa que os arquivos listados **não** foram modificados em disco, mesmo se a mensagem final do assistente disser que a tarefa terminou. Causas comuns:

- **Escrita negada** — caminho está na denylist de credencial ou fora de `HERMES_WRITE_SAFE_ROOT` (veja [File write safety](./security.md#file-write-safety))
- **Patch mismatch** — `old_string` não correspondeu ao arquivo em disco
- **Syntax gate** — conteúdo candidato falhou validação JSON/YAML/TOML antes da escrita

Exemplo de rodapé quando escritas são bloqueadas:

```
⚠️ File-mutation verifier: 2 file(s) were NOT modified this turn despite any wording above that may suggest otherwise. Run `git status` or `read_file` to confirm.
  • ~/.hermes/cron/jobs.json — [patch] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
  • ~/.hermes/scripts/monitor.py — [write_file] Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (/path/to/project)
```

Se escritas em estado Hermes (jobs cron, skills, scripts sob `~/.hermes/`) falham, verifique se `HERMES_WRITE_SAFE_ROOT` está definido no ambiente. Para mudanças cron, use a ferramenta `cronjob` ou `hermes cron edit` em vez de patch direto em `jobs.json`.

### Idioma da UI para mensagens estáticas {#ui-language-for-static-messages}

A config `display.language` traduz um pequeno conjunto de mensagens estáticas ao usuário — prompt de aprovação da CLI, um punhado de respostas de slash-command do gateway (ex.: avisos restart-drain, "approval expired", "goal cleared"). **Não** traduz respostas do agente, linhas de log, saída de ferramentas, tracebacks de erro ou descrições de slash-command — esses permanecem em inglês. Se quer o agente respondendo em outro idioma, diga no prompt ou system message.

Valores suportados: `en` (padrão), `zh` (Chinês simplificado), `zh-hant` (Chinês tradicional), `ja` (Japonês), `de` (Alemão), `es` (Espanhol), `fr` (Francês), `tr` (Turco), `uk` (Ucraniano), `af` (Afrikaans), `ko` (Coreano), `it` (Italiano), `ga` (Irlandês), `pt` (Português), `ru` (Russo), `hu` (Húngaro). Valores desconhecidos caem para inglês.

Você também pode definir por sessão com a env var `HERMES_LANGUAGE`, que sobrescreve o valor de config.

```yaml
display:
  language: zh   # CLI approval prompts appear in Chinese
```

| Modo | O que você vê |
|------|-------------|
| `off` | Silencioso — só a resposta final |
| `new` | Indicador de ferramenta só quando a ferramenta muda |
| `all` | Toda tool call com preview curto (padrão) |
| `verbose` | Args completos, resultados e logs de debug |

Na CLI, percorra estes modos com `/verbose`. Para usar `/verbose` em plataformas de mensagens (Telegram, Discord, Slack, etc.), defina `tool_progress_command: true` na seção `display` acima. O comando então percorre o modo e salva em config.

Tool progress requer adaptador de gateway que possa exibir atualizações de progresso com segurança. Plataformas sem suporte a edição de mensagem, incluindo Signal, suprimem bubbles tool-progress mesmo se `/verbose` salvar modo não-`off`.

### Focus view (`/focus`, CLI + TUI) {#focus-view-focus-cli--tui}

`display.focus_view: true` habilita **focus view** — modo de saída reduzida para quando quer a resposta, não o play-by-play. É uma camada fina sobre a mesma maquinaria `tool_progress` em vez de um segundo caminho de supressão:

- ligá-lo fixa `tool_progress` em `off` e guarda seu modo anterior em `display.focus_saved_tool_progress`;
- `/focus off` restaura aquele modo exatamente, então setup `/verbose verbose` sobrevive a ida e volta;
- cada turno completado termina com linha de recuperação dim — `⋯ 7 tool lines hidden · /focus off to show` — contada contra seu modo *pré-focus*, então nunca afirma ter escondido linhas que você já tinha desligado;
- badge persistente `◉ focus` fica na barra de status (tanto CLI prompt_toolkit quanto TUI Ink) para o modo reduzido nunca ser invisível;
- percorrer `/verbose` com focus ligado devolve o modo a `/verbose` e limpa o badge.

Focus view é **só display**. Nunca edita histórico de conversa, prompt de sistema, schemas de ferramenta ou payload de requisição — detalhe escondido é suprimido na tela, nunca descartado, e prompt caching não é afetado.

### Rodapé de metadados de runtime (só gateway) {#runtime-metadata-footer-gateway-only}

Quando `display.runtime_footer.enabled: true`, o Hermes anexa rodapé pequeno de contexto runtime à **mensagem final** de cada turno do gateway. O rodapé atual pode mostrar modelo, porcentagem de janela de contexto e diretório de trabalho atual. Desligado por padrão; opte in por gateway se sua equipe quer toda resposta com esta proveniência.

```yaml
display:
  runtime_footer:
    enabled: true
    fields: ["model", "context_pct", "cwd"]   # order shown; drop any to hide
```

Campos suportados:

| Campo | Renderiza | Exemplo |
| --- | --- | --- |
| `model` | Id de modelo bare, prefixo de vendor removido | `gpt-5.4` |
| `context_pct` | Ocupação de contexto da última chamada como percentual | `5%` |
| `latency` | Duração wall-clock do turno | `22s`, `1m05s` |
| `cwd` | Diretório de trabalho relativo ao home | `~` |

O conjunto padrão de campos é `["model", "context_pct", "cwd"]`. `latency` é opt-in — adicione a `fields` para usar. Campos cujos dados não estão disponíveis são pulados silenciosamente em vez de renderizar slot vazio.

O slash command `/footer` alterna isso em runtime em qualquer sessão.

Exemplo de rodapé anexado a resposta Telegram/Discord/Slack:

```
— claude-opus-4.7 · 12 tool calls · 2m 14s · $0.042
```

Só a **mensagem final** de um turno recebe o rodapé; atualizações interim permanecem limpas.

### Sobrescritas de progresso por plataforma {#per-platform-progress-overrides}

Plataformas diferentes têm necessidades de verbosidade diferentes. Use `display.platforms` para definir modos por plataforma:

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

Plataformas sem sobrescrita caem para o valor global `tool_progress`. Chaves de plataforma válidas: `telegram`, `discord`, `slack`, `signal`, `whatsapp`, `matrix`, `mattermost`, `email`, `sms`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot`. A chave legada `display.tool_progress_overrides` ainda carrega por retrocompatibilidade mas está depreciada e migrada para `display.platforms` no primeiro carregamento.

Signal está listado como chave de plataforma válida porque a config pode ser salva por plataforma, mas o adaptador Signal atual não pode editar mensagens enviadas e não renderiza bubbles tool-progress. Mantenha Signal `tool_progress` em `off`; use CLI ou plataforma de mensagens com edição se precisar ver cada tool call ao vivo.

`interim_assistant_messages` é só gateway. Quando habilitado, o Hermes envia atualizações do assistente mid-turn completadas como mensagens de chat separadas. Isso é independente de `tool_progress` e não requer streaming do gateway.

`show_commentary` (padrão `true`) controla o canal commentary de modelos Codex Responses — a narração de progresso polida que esses modelos produzem junto ao reasoning privado. Quando habilitado, cada mensagem commentary completada é entregue como atualização mid-turn visível (no gateway isso também requer `interim_assistant_messages`). Defina `false` se a narração extra irrita: commentary então cai para o canal reasoning e só é mostrada quando `show_reasoning` está habilitado.

## Privacidade {#privacy}

```yaml
privacy:
  redact_pii: false  # Strip PII from LLM context (gateway only)
```

Quando `redact_pii` é `true`, o gateway redige informação pessoal identificável do prompt de sistema antes de enviá-lo ao LLM em plataformas suportadas:

| Campo | Tratamento |
|-------|-----------|
| Números de telefone (user ID no WhatsApp/Signal) | Hasheados para `user_<12-char-sha256>` |
| User IDs | Hasheados para `user_<12-char-sha256>` |
| Chat IDs | Porção numérica hasheada, prefixo de plataforma preservado (`telegram:<hash>`) |
| Home channel IDs | Porção numérica hasheada |
| Nomes de usuário / usernames | **Não afetados** (escolhidos pelo usuário, publicamente visíveis) |

**Suporte de plataforma:** Redação aplica-se a WhatsApp, Signal e Telegram. Discord e Slack são excluídos porque seus sistemas de mention (`<@user_id>`) exigem o ID real no contexto LLM.

Hashes são determinísticos — o mesmo usuário sempre mapeia para o mesmo hash, então o modelo ainda distingue usuários em group chats. Roteamento e entrega usam valores originais internamente.

## Speech-to-Text (STT) {#speech-to-text-stt}

```yaml
stt:
  enabled: true                # Auto-transcribe inbound voice messages (default: true)
  echo_transcripts: true       # Post raw transcripts back to the chat as 🎙️ "..." (default: true)
  provider: "local"            # "local" | "groq" | "openai" | "mistral" | "xai" | "elevenlabs" | "deepinfra" | ...
  language: "en"               # GLOBAL language hint for every provider (per-provider language wins); set "" for auto-detect
  cloud_trim_silence: true     # trim long pauses with ffmpeg before uploading to a cloud provider (default: true)
  cloud_trim_threshold_db: -40 # audio quieter than this counts as silence
  cloud_trim_keep_ms: 300      # how much of each pause survives the trim (keeps natural pacing)
  # prompt: "Hermes, Teknium, Nous Research, kanban"   # Static vocabulary hint (see below)
  local:
    model: "base"              # tiny, base, small, medium, large-v3
    language: ""               # per-provider override of stt.language
    initial_prompt: ""         # optional whisper prompt to bias vocabulary/script (e.g. Simplified Chinese)
    vad: true                  # Silero VAD filter (default on) — silence never reaches whisper; false = raw behavior (music/ambient)
    vad_min_silence_ms: 500    # min silence (ms) that splits speech chunks when vad is on
    no_speech_prob_threshold: 0.6  # drop a segment only when no_speech_prob > this...
    logprob_threshold: -1.0        # ...AND avg_logprob < this (both must hit — quiet real speech survives)
    unload_after_idle_seconds: 0   # 0=never unload (default); e.g. 300 = release the model after 5min idle
  groq:
    language: ""               # per-provider override of stt.language
  openai:
    model: "whisper-1"         # whisper-1 | gpt-4o-mini-transcribe | gpt-4o-transcribe | gpt-transcribe
    language: ""               # per-provider override of stt.language
  # model: "whisper-1"         # Legacy fallback key still respected
```

Resolução de idioma é a mesma para **todo** provider STT (local, groq, openai, mistral, xai, elevenlabs, deepinfra, command providers e plugins): `stt.<provider>.language` → `stt.language` → env var `HERMES_LOCAL_STT_LANGUAGE` → auto-detect do provider. **O padrão é `stt.language: "en"`** — auto-detecção Whisper frequentemente identifica errado clips curtos ou com sotaque, o que aparece como notas de voz transcritas no idioma errado. Falantes não ingleses devem definir `stt.language` para seu código de idioma uma vez (ex.: `"es"`, `"zh"`, `"uk"`); defina `""` para restaurar auto-detecção para uso multilíngue.

Defina `stt.echo_transcripts: false` quando o gateway deve transcrever notas de voz para o agente mas não deve postar a transcrição bruta de volta ao chat (por exemplo, bots WhatsApp voltados ao cliente).

Comportamento do provider:

- `local` usa `faster-whisper` na sua máquina. Instale separadamente com `pip install faster-whisper`. Hardening contra alucinação de silêncio está ligado por padrão: filtro Silero VAD impede silêncio/ruído de chegar ao Whisper, condicionamento cross-window está desabilitado, e segmentos que o modelo marca como provavelmente-não-fala *e* baixa confiança são descartados. Defina `stt.local.vad: false` para transcrever áudio não-fala (música, ambiente) com comportamento raw. O modelo permanece carregado em memória entre mensagens de voz para transcrição de baixa latência; defina `stt.local.unload_after_idle_seconds` (ex.: `300` por 5 minutos) para liberar automaticamente o modelo quando idle. Isso libera memória GPU em hosts CUDA (o ganho principal quando LLM local compartilha GPU); em CPU a memória fica reutilizável pelo processo, embora footprint visível ao SO possa não encolher até o processo precisar do espaço para outra coisa. A próxima mensagem de voz recarrega o modelo transparentemente.
- `groq` usa endpoint Whisper-compatible da Groq e lê `GROQ_API_KEY`. Passe `stt.groq.language` (ou env var global `HERMES_LOCAL_STT_LANGUAGE`) para pular auto-detecção e reduzir latência.
- `openai` usa API speech da OpenAI e lê `VOICE_TOOLS_OPENAI_KEY`.

Providers cloud (groq, openai, mistral, xai, elevenlabs, deepinfra) recebem **trim de silêncio pré-upload** por padrão quando `ffmpeg` está instalado: pausas longas em nota de voz são colapsadas client-side antes do upload, mantendo `cloud_trim_keep_ms` de cada pausa para ritmo natural sobreviver. Áudio mais curto significa uploads mais rápidos, billing menor por minuto de áudio e menos alucinações de silêncio do modelo remoto. Clips menores que 12 segundos pulam o trim (economia não importa lá, e vários providers cobram mínimo por requisição). O trim é best-effort — se ffmpeg falta, trim falha, clip é sobretudo silêncio, ou trim economizaria menos que ~10%, o arquivo original é enviado intacto. Defina `stt.cloud_trim_silence: false` para sempre enviar o original (ex.: ao transcrever música ou áudio ambiente via provider cloud). Providers command-type e plugin nunca recebem áudio trimmed.

Um `stt.provider` explicitamente selecionado é honrado estritamente — se estiver indisponível, a transcrição erra com orientação para rodar `hermes tools` em vez de trocar providers. Só quando nenhum provider jamais foi selecionado o Hermes auto-detecta nesta ordem: `local` → `groq` → `openai`.

Sobrescritas de modelo Groq e OpenAI são driven por ambiente:

```bash
STT_GROQ_MODEL=whisper-large-v3-turbo
STT_OPENAI_MODEL=whisper-1
GROQ_BASE_URL=https://api.groq.com/openai/v1
STT_OPENAI_BASE_URL=https://api.openai.com/v1
```

### Prompt de transcrição (dicas de vocabulário) {#transcription-prompt-vocabulary-hints}

`stt.prompt` é uma dica estática opcional passada a backends STT que suportam prompt. Use para nomes próprios, nomes de produto e jargão que modelos da família Whisper ouvem errado:

```yaml
stt:
  provider: "local"
  prompt: "Hermes, Teknium, Nous Research, kanban, Ollama"
```

**Composição.** O valor de config é a base. Plugins que registram o hook [`pre_transcription`](/user-guide/features/hooks#pre_transcription) mutam por cima, last-writer-wins por campo. Dicas de múltiplos plugins se compostem de forma determinística: a descoberta de plugins carrega plugins em ordem ordenada por plugin id, e os callbacks de cada plugin rodam na própria ordem de registro, então o mesmo conjunto de plugins sempre produz o mesmo prompt final. Um hook retornando string vazia para `prompt` limpa o prompt de config daquela requisição. Hooks também podem sobrescrever `language` e `model`; `file_path` é read-only e qualquer tentativa de mudá-lo é logada e descartada. Sem hook registrado e sem `stt.prompt` definido, a requisição outgoing é idêntica a releases anteriores.

**Suporte de provider.**

| Provider | Parâmetro de prompt | Comportamento |
|----------|-----------------|----------|
| `local` (faster-whisper) | `initial_prompt` | Encaminhado inalterado ao modelo local |
| `openai` | `prompt` | Encaminhado inalterado na requisição de transcrição |
| `groq` | `prompt` | Encaminhado inalterado na requisição de transcrição |
| `mistral` | `prompt` | Encaminhado inalterado na requisição de transcrição |
| `deepinfra` | `prompt` | Caminho OpenAI-compatible, encaminhado inalterado |
| `xai` | não suportado | Logado em DEBUG, a requisição segue sem o prompt |
| `elevenlabs` | não suportado | Logado em DEBUG, a requisição segue sem o prompt |
| `local_command` | não suportado | Logado em DEBUG, a requisição segue sem o prompt |
| `stt.providers.<name>` com `type: command` | não suportado | Logado em DEBUG, a requisição segue sem o prompt |
| Providers registrados por plugin | `prompt` nos kwargs `transcribe(**extra)` | Só enviado quando um prompt está definido, então providers anteriores a esta chave veem chamadas inalteradas |

**Comprimento.** Modelos da família Whisper só condicionam nos ~224 tokens finais do prompt. Para os backends da família whisper (`local`, `openai`, `groq`, `deepinfra`) o Hermes aplica esse cap client-side: um prompt final longo demais é truncado para a cauda com um warning logado — a requisição nunca erra por comprimento de prompt. Outros backends (`mistral`, providers plugin) recebem o prompt inalterado e possuem a própria validação. Mantenha as dicas curtas e específicas de qualquer forma.

:::warning Prompts são enviados junto com o áudio
O prompt final é enviado ao provider STT configurado junto com o arquivo de áudio. Mantenha segredos e contexto derivado de sessão fora de `stt.prompt` e de qualquer coisa que um hook `pre_transcription` retorne, especialmente quando o provider é uma API hospedada em vez do `faster-whisper` local.
:::

## Voice mode (CLI) {#voice-mode-cli}

```yaml
voice:
  record_key: "ctrl+b"         # Push-to-talk key inside the CLI
  max_recording_seconds: 120    # Hard stop for long recordings
  auto_tts: false               # Enable spoken replies automatically when /voice on
  beep_enabled: true            # Play record start/stop beeps in CLI voice mode
  beep_volume: 0.3              # Beep amplitude (0.0-1.0); raise it on quiet systems / headphones
  silence_threshold: 200        # RMS threshold for speech detection
  silence_duration: 3.0         # Seconds of silence before auto-stop
```

Use `/voice on` na CLI para habilitar modo microfone, `record_key` para iniciar/parar gravação, e `/voice tts` para alternar respostas faladas. Veja [Voice Mode](/user-guide/features/voice-mode) para setup end-to-end e comportamento por plataforma.

## Streaming {#streaming}

Stream tokens para terminal ou plataformas de mensagens conforme chegam, em vez de esperar a resposta completa.

### Streaming CLI {#cli-streaming}

```yaml
display:
  streaming: true         # Stream tokens to terminal in real-time
  show_reasoning: true    # Also stream reasoning/thinking tokens (optional)
```

Quando habilitado, respostas aparecem token a token dentro de uma caixa streaming. Tool calls ainda são capturadas silenciosamente. Se o provider não suporta streaming, cai automaticamente para display normal.

### Streaming gateway (Telegram, Discord, Slack) {#gateway-streaming-telegram-discord-slack}

```yaml
streaming:
  enabled: true           # Enable progressive message editing (default: false)
  transport: auto         # "auto" (default) | "edit" (progressive message editing) | "off"
  edit_interval: 0.8      # Seconds between message edits (default: 0.8)
  buffer_threshold: 24    # Characters before forcing an edit flush (default: 24)
  cursor: " ▉"            # Cursor shown during streaming
  fresh_final_after_seconds: 0    # Opt in to fresh final (Telegram) when preview is this old
```

Quando habilitado, o bot envia mensagem no primeiro token, depois edita progressivamente conforme mais tokens chegam. Plataformas que não suportam edição de mensagem (Signal, Email, Home Assistant) são auto-detectadas na primeira tentativa — streaming é desabilitado graciosamente para aquela sessão sem flood de mensagens.

Para atualizações naturais separadas do assistente mid-turn sem edição progressiva de tokens, defina `display.interim_assistant_messages: true`.

**Tratamento de overflow:** Se o texto streamed excede o limite de comprimento de mensagem da plataforma (~4096 chars), a mensagem atual é finalizada e uma nova começa automaticamente.

**Fresh final (Telegram):** `editMessageText` do Telegram preserva o timestamp original da mensagem, então uma resposta streamed longa manteria o timestamp do primeiro token mesmo após conclusão. Defina `fresh_final_after_seconds > 0` para optar por entregar previews antigos como mensagens finais novas com best-effort delete do preview. O padrão é `0`, que sempre finaliza respostas streamed in place e evita breve sequência duplicate-message/delete em clientes que mostram ambas operações.

:::note Padrões de streaming por plataforma
O interruptor master `streaming.enabled` é `false` por padrão — nada faz stream até você ligar. Uma vez habilitado, streaming é decidido **por plataforma**: Telegram vem com `display.platforms.telegram.streaming: true` (faz stream) e Discord com `display.platforms.discord.streaming: false` (não faz). Então após habilitar streaming, Telegram faz stream out of the box e Discord permanece em respostas whole-message até mudar seu toggle. Você pode ajustar estes toggles por plataforma nos toggles **Channels** do dashboard ou diretamente em `~/.hermes/config.yaml`.
:::

## Isolamento de sessão em group chat {#group-chat-session-isolation}

Limite quantas sessões de chat podem estar ativamente abertas entre CLI, TUI/dashboard
e gateway de mensagens:

```yaml
max_concurrent_sessions: null  # null/0 = unlimited; positive integer = active session cap
```

Um slot é tomado quando uma sessão executa seu **primeiro turno**, não quando uma janela de chat
é aberta. Abrir, retomar ou reconectar a um chat não custa nada até você
enviar mensagem, então abas desktop idle (e resumes em segundo plano que um websocket instável
dispara) não podem famintar o gateway de mensagens que compartilha este cap.

Quando o cap é atingido, o Hermes retorna mensagem de limite direta nomeando quais
superfícies seguram os slots. Sessões ativas existentes mantêm comportamento normal.
Execute `hermes status` para ver uso atual de slots e todo holder.

A chave canônica é top-level `max_concurrent_sessions`. O Hermes também aceita
`gateway.max_concurrent_sessions` como fallback, mas a chave top-level vence quando
ambas estão definidas.

O cap é aplicado com arquivo de lease runtime local e é best-effort: o Hermes
falha open se o registry não puder ser lido ou locked para usuários não ficarem presos.
Destinado a runtime host/perfil único, não `$HERMES_HOME`
montado em várias máquinas.

Controle se chats compartilhados mantêm uma conversa por sala ou uma conversa por participante:

```yaml
group_sessions_per_user: true  # true = per-user isolation in groups/channels, false = one shared session per chat
```

- `true` é o padrão e config recomendada. Em canais Discord, grupos Telegram, canais Slack e contextos compartilhados similares, cada remetente recebe sua própria sessão quando a plataforma fornece user ID.
- `false` reverte ao comportamento antigo de sala compartilhada. Pode ser útil se quer explicitamente que o Hermes trate um canal como uma conversa colaborativa, mas também significa que usuários compartilham contexto, custo de tokens e estado de interrupção.
- DMs não são afetados. O Hermes ainda chaveia DMs por chat/DM ID como usual.
- Threads permanecem isolados do canal pai de qualquer forma; com `true`, cada participante também recebe sua própria sessão dentro do thread.

Para detalhes de comportamento e exemplos, veja [Sessions](/user-guide/sessions) e o [guia Discord](/user-guide/messaging/discord).

## Comportamento de DM não autorizado {#unauthorized-dm-behavior}

Controle o que o Hermes faz quando um usuário desconhecido envia mensagem direta:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` é o padrão para plataformas DM estilo chat. O Hermes nega acesso, mas responde com código de pairing one-time em DMs.
- `ignore` descarta silenciosamente DMs não autorizados.
- Email padrão `ignore` a menos que `platforms.email.unauthorized_dm_behavior: pair` esteja definido, porque inboxes podem conter mail não lido não relacionado.
- Seções de plataforma sobrescrevem o padrão global, então você pode manter pairing habilitado amplamente enquanto torna uma plataforma mais silenciosa.

## Quick commands {#quick-commands}

Defina comandos customizados que executam comandos shell sem invocar o LLM, ou fazem alias de um slash command para outro. Quick commands exec são zero-token e úteis de plataformas de mensagens (Telegram, Discord, etc.) para checagens rápidas de servidor ou scripts utilitários.

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
    command: cd ~/.hermes/hermes-agent && git pull && uv pip install -e .
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

Uso: digite `/status`, `/disk`, `/update`, `/gpu` ou `/restart` na CLI ou qualquer plataforma de mensagens. Comandos `exec` rodam localmente no host e retornam saída diretamente — sem chamada LLM, sem tokens consumidos. Comandos `alias` reescrevem para o slash command alvo configurado.

- **Timeout de 30 segundos** — comandos longos são mortos com mensagem de erro
- **Prioridade** — quick commands são checados antes de skill commands, então você pode sobrescrever nomes de skill
- **Autocomplete** — quick commands são resolvidos no dispatch time e não aparecem nas tabelas de autocomplete de slash-command embutidas
- **Tipo** — tipos suportados são `exec` e `alias`; outros tipos mostram erro
- **Funciona em todo lugar** — CLI, Telegram, Discord, Slack, WhatsApp, Signal, Email, Home Assistant

Atalhos de prompt string-only não são quick commands válidos. Para fluxos de prompt reutilizáveis, crie uma skill ou alias para slash command existente.

## Human delay {#human-delay}

Simule ritmo de resposta humano em plataformas de mensagens:

```yaml
human_delay:
  mode: "off"                  # off | natural | custom
  min_ms: 800                  # Minimum delay (custom mode)
  max_ms: 2500                 # Maximum delay (custom mode)
```

## Code execution {#code-execution}

Configure a ferramenta `execute_code`:

```yaml
code_execution:
  mode: project                # project (default) | strict
  timeout: 300                 # Max execution time in seconds
  max_tool_calls: 50           # Max tool calls within code execution
```

**`mode`** controla diretório de trabalho e interpretador Python para scripts:

- **`project`** (padrão) — scripts rodam no diretório de trabalho da sessão com python do virtualenv/conda env ativo. Deps do projeto (`pandas`, `torch`, pacotes do projeto) e caminhos relativos (`.env`, `./data.csv`) resolvem naturalmente, correspondendo ao que `terminal()` vê.
- **`strict`** — scripts rodam em diretório staging temp com `sys.executable` (python próprio do Hermes). Máxima reprodutibilidade, mas deps do projeto e caminhos relativos não resolvem.

Scrubbing de ambiente (remove `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, `*_CREDENTIAL`, `*_PASSWD`, `*_AUTH`) e whitelist de ferramentas aplicam identicamente em ambos modos — trocar mode não muda postura de segurança.

## Backends de web search {#web-search-backends}

As ferramentas `web_search` e `web_extract` suportam cinco providers backend. Configure o backend em `config.yaml` ou via `hermes tools`:

```yaml
web:
  backend: firecrawl    # firecrawl | searxng | parallel | tavily | exa

  # Or use per-capability keys to mix providers (e.g. free search + paid extract):
  search_backend: "searxng"
  extract_backend: "firecrawl"

  # Keyless free-tier fallback (default: true). With no backend configured
  # and no API keys present, web tools rotate across the Exa/Parallel/
  # Tavily/Firecrawl/Keenable free tiers. Set false to disable.
  keyless_fallback: true

  # One-shot keyless rescue (default: true). When the chosen/keyed backend
  # fails a call, that single call retries on the keyless ring; the next
  # call attempts the chosen backend again (never sticky).
  keyless_rescue: true

  # Pin Exa/Parallel to a tier (set by the hermes tools Free/Paid rows).
  # free = always the anonymous endpoint; paid = always the keyed SDK path;
  # unset = auto (key present -> paid, otherwise free).
  provider_tier:
    parallel: free
    exa: paid
```

| Backend | Env Var | Search | Extract |
|---------|---------|--------|---------|
| **Firecrawl** (padrão) | `FIRECRAWL_API_KEY` | ✔ | ✔ |
| **SearXNG** | `SEARXNG_URL` | ✔ | — |
| **Parallel** | `PARALLEL_API_KEY` (opcional — free tier keyless) | ✔ | ✔ |
| **Tavily** | `TAVILY_API_KEY` (opcional — keyless quando selecionado) | ✔ | ✔ |
| **Exa** | `EXA_API_KEY` (opcional — free tier keyless) | ✔ | ✔ |

**Seleção de backend:** O runtime sempre usa a seleção armazenada de `web.backend` (definida via `hermes tools`; `nous` roteia pelo Tool Gateway gerenciado). Só se nenhum backend web jamais foi selecionado um é auto-detectado de chaves de API disponíveis: se só `SEARXNG_URL` está definido, SearXNG é usado; se só `EXA_API_KEY` está definido, Exa; se só `TAVILY_API_KEY` está definido, Tavily; se só `PARALLEL_API_KEY` está definido, Parallel; se só `KEENABLE_API_KEY` está definido, Keenable. Com **nenhuma seleção e nenhuma credencial**, requests rotacionam round-robin pelo ring keyless de free-tier (Exa / Parallel / Tavily / Firecrawl / Keenable) com failover automático next-in-line em rate limits — veja o [guia Web Search](/user-guide/features/web-search) para detalhes. Uma vez que uma seleção existe, adicionar uma chave ao `.env` não muda a rota. Selecionar Tavily, Firecrawl ou Keenable em `hermes tools` também funciona sem chave.

**SearXNG** é metasearch engine gratuito, self-hosted e respeitoso à privacidade que consulta 70+ search engines. Sem API key — só defina `SEARXNG_URL` para sua instância (ex.: `http://localhost:8080`). SearXNG é só search; `web_extract` requer provider extract separado (defina `web.extract_backend`). Veja o [guia de setup Web Search](/user-guide/features/web-search) para instruções Docker.

**Firecrawl self-hosted:** Defina `FIRECRAWL_API_URL` para apontar à sua instância. Quando URL custom está definida, API key torna-se opcional (defina `USE_DB_AUTHENTICATION=*** no servidor para desabilitar auth).

**Modos Parallel search:** Defina `PARALLEL_SEARCH_MODE` para controlar comportamento de busca — `fast`, `one-shot` ou `agentic` (padrão: `agentic`).

**Exa:** Defina `EXA_API_KEY` em `~/.hermes/.env`. Suporta filtro `category` (`company`, `research paper`, `news`, `people`, `personal site`, `pdf`) e filtros de domínio/data.

## Browser {#browser}

Configure comportamento de automação de browser:

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

**Políticas de dialog:**

- `must_respond` (padrão) — captura o dialog, expõe em `browser_snapshot.pending_dialogs` e espera o agente chamar `browser_dialog(action=...)`. Após `dialog_timeout_s` segundos sem resposta, o dialog é auto-dismissed para o thread JS da página não travar forever.
- `auto_dismiss` — captura, dismiss imediatamente. O agente ainda vê o registro do dialog em `browser_snapshot.recent_dialogs` com `closed_by="auto_policy"` depois.
- `auto_accept` — captura, accept imediatamente. Útil para páginas com prompts `beforeunload` agressivos.

Veja a [página de recurso browser](./features/browser.md#browser_dialog) para o fluxo completo de dialog.

O toolset browser suporta vários providers. Veja a [página Browser](/user-guide/features/browser) para detalhes de Browserbase, Browser Use e setup CDP Chromium-family local.

## Timezone {#timezone}

Sobrescreva timezone local do servidor com string timezone IANA. Afeta timestamps em logs, agendamento cron e injeção de hora no prompt de sistema.

```yaml
timezone: "America/New_York"   # IANA timezone (default: "" = server-local time)
```

Valores suportados: qualquer identificador timezone IANA (ex.: `America/New_York`, `Europe/London`, `Asia/Kolkata`, `UTC`). Deixe vazio ou omita para hora local do servidor.

## Discord {#discord}

Configure comportamento específico Discord para o gateway de mensagens:

```yaml
discord:
  require_mention: true          # Require @mention to respond in server channels
  free_response_channels: ""     # Comma-separated channel IDs where bot responds without @mention
  auto_thread: true              # Auto-create threads on @mention in channels
```

- `require_mention` — quando `true` (padrão), o bot só responde em canais de servidor quando mencionado com `@BotName`. DMs sempre funcionam sem mention.
- `free_response_channels` — lista separada por vírgulas de IDs de canal onde o bot responde a toda mensagem sem exigir mention.
- `auto_thread` — quando `true` (padrão), mentions em canais criam automaticamente thread para a conversa, mantendo canais limpos (similar a threading Slack).

## Segurança {#security}

Scanning de segurança pré-execução e redação de segredos:

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

- `redact_secrets` — quando `true`, detecta e redige automaticamente padrões que parecem chaves de API, tokens e senhas em saída de ferramenta antes de entrar no contexto de conversa e logs. **Ligado por padrão**. Defina `false` explicitamente só quando precisar de strings raw tipo credencial para debug ou desenvolvimento de redactor.
- `tirith_enabled` — quando `true`, comandos de terminal são escaneados por [Tirith](https://github.com/sheeki03/tirith) antes da execução para detectar operações potencialmente perigosas.
- `tirith_path` — caminho para o binário tirith. Defina se tirith está instalado em local não padrão.
- `tirith_timeout` — segundos máximos para esperar scan tirith. Comandos prosseguem se o scan expirar.
- `tirith_fail_open` — quando `true` (padrão), comandos são permitidos se tirith não está disponível ou falha. Defina `false` para bloquear comandos quando tirith não pode verificá-los.

## Website blocklist {#website-blocklist}

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

Quando habilitado, qualquer URL correspondendo a padrão de domínio bloqueado é rejeitada antes da ferramenta web ou browser executar. Isso aplica-se a `web_search`, `web_extract`, `browser_navigate` e qualquer ferramenta que acesse URLs.

Regras de domínio suportam:
- Domínios exatos: `admin.example.com`
- Subdomínios wildcard: `*.internal.company.com` (bloqueia todos subdomínios)
- Wildcards TLD: `*.local`

Arquivos compartilhados contêm uma regra de domínio por linha (linhas em branco e comentários `#` são ignorados). Arquivos ausentes ou ilegíveis registram aviso mas não desabilitam outras ferramentas web.

A política é cacheada por 30 segundos, então mudanças de config entram em vigor rapidamente sem reinício.

## Smart approvals {#smart-approvals}

Controle como o Hermes lida com comandos potencialmente perigosos:

```yaml
approvals:
  mode: smart   # smart | manual | off
```

| Modo | Comportamento |
|------|----------|
| `smart` (padrão) | Usa LLM auxiliar para avaliar se comando sinalizado é realmente perigoso. Comandos de baixo risco são auto-aprovados só para aquele comando. Comandos genuinamente arriscados são negados; decisões incertas escalam ao usuário. |
| `manual` | Pede ao usuário antes de executar qualquer comando sinalizado. Na CLI, mostra diálogo interativo de aprovação. Em mensagens, enfileira requisição de aprovação pendente. |
| `off` | Pula todas checagens de aprovação. Equivalente a `HERMES_YOLO_MODE=true`. **Use com cautela.** |

Smart mode é particularmente útil para reduzir fadiga de aprovação — deixa o agente trabalhar mais autonomamente em operações seguras enquanto ainda captura comandos genuinamente destrutivos.

:::warning
Definir `approvals.mode: off` desabilita todas checagens de segurança para comandos de terminal. Use só em ambientes confiáveis e sandboxed.
:::

### Circuit breaker de negação {#denial-circuit-breaker}

`approvals.denial_breaker_threshold` (padrão `3`) protege contra o agente retentando variações de comando que o revisor smart-approval continua negando — cada retry queima outra chamada LLM guardian. Após tantas negações consecutivas em uma sessão, a mensagem de deny escala para instrução hard-stop dizendo ao agente parar, reportar a operação bloqueada e pedir para você executar manualmente ou `/approve`. Qualquer aprovação reseta a contagem; defina `0` para desabilitar:

```yaml
approvals:
  denial_breaker_threshold: 3   # 0 disables the breaker
```

### Regras de deny {#deny-rules}

`approvals.deny` é lista de padrões glob que bloqueiam comandos de terminal correspondentes incondicionalmente — mesmo sob `--yolo`, `/yolo` ou `mode: off`. É o counterpart editável pelo usuário da blocklist hardline embutida:

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

Padrões são globs fnmatch case-insensitive e devem ser quoted em YAML (`*` leading bare é erro de parse). Veja [Security — User-Defined Deny Rules](/user-guide/security#user-defined-deny-rules-approvalsdeny) para detalhes.

### Política custom de smart-approval {#custom-smart-approval-policy}

`approvals.smart_policy` permite anexar suas próprias regras às instruções do revisor smart-approval. Quando definido, o texto é adicionado ao system prompt do LLM guardian (canal confiável — nunca junto ao texto não confiável do comando), então você pode apertar ou relaxar julgamento para seu ambiente sem editar código:

```yaml
approvals:
  smart_policy: |
    Always ESCALATE commands that modify anything under /etc.
    APPROVE docker compose restarts in ~/deploys — they are routine here.
```


## Checkpoints {#checkpoints}

Snapshots automáticos de filesystem antes de operações de arquivo destrutivas. Veja [Checkpoints & Rollback](/user-guide/checkpoints-and-rollback) para detalhes.

```yaml
checkpoints:
  enabled: false                 # Enable automatic checkpoints (also: hermes chat --checkpoints). Default: false (opt-in).
  max_snapshots: 20              # Max checkpoints to keep per directory (default: 20)
```


## Delegation {#delegation}

Configure comportamento de subagente para a ferramenta delegate:

```yaml
delegation:
  # model: "google/gemini-3-flash-preview"  # Override model (empty = inherit parent)
  # provider: "openrouter"                  # Override provider (empty = inherit parent)
  # base_url: "http://localhost:1234/v1"    # Direct OpenAI-compatible endpoint (takes precedence over provider)
  # api_key: "local-key"                    # API key for base_url (falls back to OPENAI_API_KEY)
  # api_mode: ""                            # Wire protocol for base_url: "chat_completions", "codex_responses", or "anthropic_messages". Empty = auto-detect from URL (e.g. /anthropic suffix → anthropic_messages). Set explicitly for non-standard endpoints the heuristic can't detect.
  max_concurrent_children: 3                # Parallel children per batch (floor 1, no ceiling). Also via DELEGATION_MAX_CONCURRENT_CHILDREN env var.
  worktree_isolation: false                 # Give each child its own git worktree branched from HEAD (local backend + git repos only; inspired by Muse Code). See Subagent Delegation → Worktree Isolation.
  max_spawn_depth: 1                        # Delegation tree depth cap (1-3, clamped). 1 = flat (default): parent spawns leaves that cannot delegate. 2 = orchestrator children can spawn leaf grandchildren. 3 = three levels.
  orchestrator_enabled: true                # Global kill switch. When false, role="orchestrator" is ignored and every child is forced to leaf regardless of max_spawn_depth.
```

**Sobrescrita provider:model de subagente:** Por padrão, subagentes herdam provider e modelo do agente pai. Defina `delegation.provider` e `delegation.model` para rotear subagentes a par provider:model diferente — ex.: usar modelo barato/rápido para subtarefas estreitas enquanto agente primário roda modelo de raciocínio caro.

**Sobrescrita de endpoint direto:** Se quer o caminho óbvio de endpoint custom, defina `delegation.base_url`, `delegation.api_key` e `delegation.model`. Isso envia subagentes diretamente àquele endpoint OpenAI-compatible e tem precedência sobre `delegation.provider`. Se `delegation.api_key` for omitida, o Hermes cai para `OPENAI_API_KEY` apenas.

**Wire protocol (`api_mode`):** O Hermes auto-detecta wire protocol de `delegation.base_url` (ex.: caminhos terminando em `/anthropic` → `anthropic_messages`; hostnames Codex / Anthropic nativo / Kimi-coding mantêm detecção existente). Para endpoints que a heurística não classifica — por exemplo Azure AI Foundry, MiniMax, Zhipu GLM ou proxies LiteLLM fronteando backend shaped Anthropic — defina `delegation.api_mode` explicitamente para um de `chat_completions`, `codex_responses` ou `anthropic_messages`. Deixe vazio (padrão) para manter auto-detecção.

O provider de delegation usa a mesma resolução de credencial que startup CLI/gateway. Todos providers configurados são suportados: `openrouter`, `nous`, `copilot`, `zai`, `kimi-coding`, `minimax`, `minimax-cn`. Quando provider está definido, o sistema resolve automaticamente base URL, API key e API mode corretos — sem wiring manual de credencial.

**Precedência:** `delegation.base_url` em config → `delegation.provider` em config → provider pai (herdado). `delegation.model` em config → modelo pai (herdado). Definir só `model` sem `provider` muda apenas nome do modelo mantendo credenciais do pai (útil para trocar modelos no mesmo provider como OpenRouter).

**Largura e profundidade:** `max_concurrent_children` limita quantos subagentes rodam em paralelo por lote (padrão `3`, mínimo 1, sem teto). Também pode ser definido via env var `DELEGATION_MAX_CONCURRENT_CHILDREN`. Quando o modelo submete array `tasks` maior que o cap, `delegate_task` retorna erro de ferramenta explicando o limite em vez de truncar silenciosamente. `max_spawn_depth` controla profundidade da árvore de delegation (limitado a 1-3). No padrão `1`, delegation é plana: filhos não podem spawnar netos, e passar `role="orchestrator"` degrada silenciosamente para `leaf`. Aumente para `2` para filhos orquestradores spawnarem netos folha; `3` para árvores de três níveis. O agente opta por orquestração por chamada via `role="orchestrator"`; `orchestrator_enabled: false` força todo filho de volta a leaf. Custo escala multiplicativamente — com `max_spawn_depth: 3` e `max_concurrent_children: 3`, a árvore pode atingir 3×3×3 = 27 agentes folha concorrentes. Veja [Subagent Delegation → Depth Limit and Nested Orchestration](features/delegation.md#depth-limit-and-nested-orchestration) para padrões de uso.

## Clarify {#clarify}

Configure quanto tempo o gateway espera resposta a pergunta esclarecedora. A chave canônica é `agent.clarify_timeout` (padrão `3600` segundos); chave legada top-level `clarify.timeout` ainda é honrada se definida explicitamente:

```yaml
agent:
  clarify_timeout: 3600        # Seconds to wait for user clarification response (0 or less = unlimited)
```

## Arquivos de contexto (SOUL.md, AGENTS.md) {#context-files-soulmd-agentsmd}

O Hermes usa dois escopos de contexto diferentes:

| Arquivo | Propósito | Escopo |
|------|---------|-------|
| `SOUL.md` | **Identidade primária do agente** — define quem o agente é (slot #1 no prompt de sistema) | `~/.hermes/SOUL.md` ou `$HERMES_HOME/SOUL.md` |
| `.hermes.md` / `HERMES.md` | Instruções específicas do projeto (maior prioridade) | Walks to git root |
| `AGENTS.md` | Instruções específicas do projeto, convenções de coding | Recursive directory walk |
| `CLAUDE.md` | Arquivos de contexto Claude Code (também detectados) | Working directory only |
| `.cursorrules` | Regras Cursor IDE (também detectadas) | Working directory only |
| `.cursor/rules/*.mdc` | Arquivos de regra Cursor (também detectados) | Working directory only |

- **SOUL.md** é a identidade primária do agente. Ocupa slot #1 no prompt de sistema, substituindo completamente a identidade default embutida. Edite para customizar totalmente quem o agente é.
- Se SOUL.md está ausente, vazio ou não pode ser carregado, o Hermes cai para identidade default embutida.
- **Arquivos de contexto de projeto usam sistema de prioridade** — só UM tipo é carregado (first match wins): `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. SOUL.md é sempre carregado independentemente.
- **AGENTS.md** é hierárquico: se subdiretórios também têm AGENTS.md, todos são combinados.
- O Hermes faz seed automaticamente de `SOUL.md` default se ainda não existir.
- Todos arquivos de contexto carregados são limitados a `context_file_max_chars` caracteres (padrão 20.000) com truncamento inteligente.

Veja também:
- [Personality & SOUL.md](/user-guide/features/personality)
- [Context Files](/user-guide/features/context-files)

## Diretório de trabalho {#working-directory}

| Contexto | Padrão |
|---------|---------|
| **CLI (`hermes`)** | Diretório atual onde você executa o comando |
| **Gateway de mensagens** | `terminal.cwd` de `~/.hermes/config.yaml`; se unset, home directory `~` |
| **Docker / Singularity / Modal / SSH** | Home directory do usuário dentro do container ou máquina remota |

Sobrescreva diretório de trabalho:
```yaml
# In ~/.hermes/config.yaml:
terminal:
  cwd: /home/myuser/projects
```

`MESSAGING_CWD` e entradas diretas `TERMINAL_CWD` em `~/.hermes/.env` são fallbacks de compatibilidade legados. Novas configurações devem usar `terminal.cwd`.

## Rede {#network}

Workarounds de conectividade para HTTP outbound:

```yaml
network:
  force_ipv4: false   # Force IPv4 for outbound connections (default: false)
```

`force_ipv4` — em servidores com IPv6 quebrado ou inalcançável, Python resolve registros AAAA primeiro e pode pendurar pelo timeout TCP completo antes de cair para IPv4. Defina `true` para pular IPv6 inteiramente e conectar via IPv4 diretamente.

## Onboarding {#onboarding}

Dicas de onboarding first-touch e oferta estruturada de profile-build:

```yaml
onboarding:
  profile_build: "ask"   # "ask" (default) | "off"
  seen: {}               # internal latch — leave empty
```

- `profile_build` — controla o caminho profile-build oferecido na very first gateway message ever. `"ask"` (padrão) oferece construir perfil de usuário; a oferta é **opt-in e consent-gated** — o agente pergunta antes de qualquer lookup e nunca lê contas conectadas silenciosamente. `"off"` mostra intro simples apenas. A oferta dispara no máximo uma vez.
- `seen` — estado interno. O Hermes trava cada hint mostrada aqui para nunca disparar de novo; a oferta profile-build também é registrada aqui uma vez mostrada. Não edite manualmente — apague toda a seção `onboarding` se quiser rever todas hints.

## Dashboard {#dashboard}

Configuração para o [web dashboard](/user-guide/features/web-dashboard) — tema visual, URL pública e providers de autenticação. Os providers de auth (OAuth, senha básica, drain) estão documentados em detalhe na página web-dashboard; esta é a forma `config.yaml`.

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
- `show_token_analytics` — desligado por padrão. A página Analytics e figuras token/cost são **estimativa local lower-bound** (excluem chamadas auxiliares, retries, fallbacks e cache writes), então podem ler muito abaixo da fatura do provider. Defina `true` só se entende que não são billing.
- `public_url` — quando definido, esta é a authority completa (scheme + host + optional path prefix) de onde o OAuth `redirect_uri` é construído. Defina para deploys atrás de reverse proxies que não encaminham headers `X-Forwarded-*` de forma confiável. Deixe vazio para usar reconstrução proxy-header.
- `oauth` / `basic_auth` / `drain_auth` — config de auth provider lida pelos plugins dashboard-auth incluídos. O segredo drain em si **não** é definido aqui; é provisionado via env var `HERMES_DASHBOARD_DRAIN_SECRET`. Veja [Web Dashboard](/user-guide/features/web-dashboard) para setup auth completo.
