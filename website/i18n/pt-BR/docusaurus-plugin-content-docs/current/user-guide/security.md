---
sidebar_position: 8
title: "Segurança"
description: "Modelo de segurança, aprovação de comandos perigosos, autorização de usuários, isolamento em container e melhores práticas de deploy em produção"
---

# Segurança

O Hermes Agent foi projetado com um modelo de segurança em camadas (defense-in-depth). Esta página cobre cada limite de segurança — da aprovação de comandos ao isolamento em container e autorização de usuários em plataformas de mensagens.

## Overview {#overview}

O modelo de segurança tem oito camadas:

1. **Autorização de usuários** — quem pode falar com o agente (allowlists, DM pairing)
2. **Aprovação de comandos perigosos** — human-in-the-loop para operações destrutivas
3. **Segurança de escrita em arquivos** — denylist e sandbox opcional de escrita para `write_file`/`patch`
4. **Isolamento em container** — sandboxing Docker/Singularity/Modal com settings hardened
5. **Filtragem de credenciais MCP** — isolamento de variáveis de ambiente para subprocessos MCP
6. **Scan de context files** — detecção de prompt injection em arquivos de projeto
7. **Isolamento entre sessões** — sessões não podem acessar dados ou estado umas das outras; paths de armazenamento de cron jobs são hardened contra ataques de path traversal
8. **Sanitização de input** — parâmetros de working directory em backends da tool terminal são validados contra uma allowlist para prevenir shell injection

## Dangerous Command Approval {#dangerous-command-approval}

Antes de executar qualquer comando, o Hermes verifica contra uma lista curada de padrões perigosos. Se houver match, o usuário deve aprovar explicitamente.

### Approval Modes {#approval-modes}

O sistema de aprovação suporta três modos, configurados via `approvals.mode` em `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: smart                     # smart | manual | off
  timeout: 300                    # seconds to wait for user response (default: 300)
  cron_mode: deny                 # deny | approve — what cron jobs do when they hit a dangerous command
  mcp_reload_confirm: true        # /reload-mcp asks before invalidating the MCP tool cache
  destructive_slash_confirm: true # /clear, /new, /reset, /undo prompt before discarding state
```

O conjunto completo de chaves:

| Key | Default | What it controls |
|---|---|---|
| `mode` | `smart` | Approval policy for dangerous shell commands — see the table below. |
| `timeout` | `300` | Seconds Hermes waits for an approval reply before timing out. |
| `cron_mode` | `deny` | How [cron jobs](./features/cron.md) behave headlessly when they trigger a dangerous-command prompt. `deny` blocks the command (the agent must find another path); `approve` auto-approves everything in cron context. |
| `mcp_reload_confirm` | `true` | When true, `/reload-mcp` asks before rebuilding the MCP tool set. Rebuilding invalidates the provider prompt cache (tool schemas live in the system prompt), so the next message re-sends full input tokens. Users who click **Always Approve** flip this key to `false`. |
| `destructive_slash_confirm` | `true` | When true, destructive session slash commands (`/clear`, `/new`, `/reset`, `/undo`) prompt before discarding conversation state. Three-option dialog (Approve Once / Always Approve / Cancel) routed through native yes/no buttons on Telegram, Discord, and Slack; text fallback elsewhere. Users who click **Always Approve** flip this key to `false`. TUI uses its own modal overlay (set `HERMES_TUI_NO_CONFIRM=1` to opt out there). |

| Mode | Behavior |
|------|----------|
| **smart** (default) | Use an auxiliary LLM to assess risk. Low-risk commands (e.g., `python -c "print('hello')"`) are auto-approved for that command only. Genuinely dangerous commands are auto-denied. Uncertain cases escalate to a manual prompt. |
| **manual** | Always prompt the user for approval on dangerous commands. |
| **off** | Disable all approval checks — equivalent to running with `--yolo`. All commands execute without prompts. |

:::warning
Definir `approvals.mode: off` desabilita todos os prompts de segurança. Use apenas em ambientes confiáveis (CI/CD, containers, etc.).
:::

### YOLO Mode {#yolo-mode}

O modo YOLO contorna **todos** os prompts de aprovação de comandos perigosos para a sessão atual. Pode ser ativado de três formas:

1. **CLI flag**: Inicie uma sessão com `hermes --yolo` ou `hermes chat --yolo`
2. **Slash command**: Digite `/yolo` durante uma sessão para alternar on/off
3. **Environment variable**: Defina `HERMES_YOLO_MODE=1`

O comando `/yolo` é um **toggle** — cada uso inverte o modo on/off:

```
> /yolo
  ⚡ YOLO mode ON — all commands auto-approved. Use with caution.

> /yolo
  ⚠ YOLO mode OFF — dangerous commands will require approval.
```

O modo YOLO está disponível em sessões CLI e gateway. Internamente, define a variável de ambiente `HERMES_YOLO_MODE` que é verificada antes de cada execução de comando.

Quando YOLO está ativo, o Hermes mostra dois lembretes visuais persistentes para ser difícil esquecer que os prompts de aprovação estão contornados:

- Uma linha de banner vermelho no início da sessão quando YOLO já está ativo: `⚠ YOLO mode — all approval prompts bypassed`. Oculto quando YOLO está off para o banner padrão permanecer limpo.
- Um fragmento `⚠ YOLO` na barra de status em todos os tiers de largura, atualizado ao vivo conforme você alterna YOLO on/off (renderer rich-text e fallback plain-text).

:::danger
O modo YOLO desabilita **todas** as verificações de segurança de comandos perigosos para a sessão — **exceto** a hardline blocklist (veja abaixo). Use apenas quando confia plenamente nos comandos sendo gerados (ex., scripts de automação bem testados em ambientes descartáveis).
:::

Para slash commands destrutivos de sessão (`/clear`, `/new` / `/reset`, `/undo`, `/quit --delete` — `/exit --delete` é um alias), a CLI também pede confirmação antes de executá-los. Veja [Slash Commands — Confirmation prompts for destructive commands](../reference/slash-commands.md#confirmation-prompts-for-destructive-commands).

### Hardline Blocklist (Always-On Floor) {#hardline-blocklist-always-on-floor}

Alguns comandos são tão catastróficos — wipes irreversíveis de filesystem, fork bombs, writes diretos em block device — que o Hermes recusa executá-los **independentemente** de:

- `--yolo` / `/yolo` toggled on
- `approvals.mode: off`
- Cron jobs executando em modo headless `approve`
- Usuário clicando explicitamente "allow always"

A blocklist é o piso abaixo de `--yolo`. Dispara **antes** da camada de aprovação sequer ver o comando, e não há override flag. Padrões atualmente cobertos (não exaustivo; mantido em sync com `tools/approval.py::UNRECOVERABLE_BLOCKLIST`):

| Pattern | Why it's hardline |
|---|---|
| `rm -rf /` and obvious variants | Wipes the filesystem root |
| `rm -rf --no-preserve-root /` | The explicit "yes I mean root" variant |
| `:(){ :\|:& };:` (bash fork bomb) | Pegs the host until reboot |
| `mkfs.*` on a mounted root device | Formats the live system |
| `dd if=/dev/zero of=/dev/sd*` | Zeroes a physical disk |
| Piping untrusted URLs to `sh` at the rootfs top level | Remote-code-execution attack vector too broad to approve |

Se você acertar a blocklist, a tool call retorna um erro explicativo ao agente e nada executa. Se um workflow legítimo precisa de um desses comandos (você é o operador de um pipeline wipe-and-reinstall, por exemplo), execute fora do agente.

### User-Defined Deny Rules (`approvals.deny`) {#user-defined-deny-rules-approvalsdeny}

A hardline blocklist é fixa e shipped no código. `approvals.deny` é sua contraparte editável pelo usuário: uma lista de padrões glob que bloqueiam comandos terminal matching incondicionalmente — **antes** de `--yolo`, `/yolo` e `approvals.mode: off` serem consultados. Use para rodar yolo-with-exceptions: "deixe o agente fazer tudo, exceto estas coisas específicas, sempre."

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

Detalhes:

- Padrões são globs [fnmatch](https://docs.python.org/3/library/fnmatch.html) (`*`, `?`, `[...]`) matched **case-insensitively** contra o texto completo do comando. `git push --force*` corresponde a `git push --force origin main` mas não a `git push origin main`.
- Matching roda sobre as mesmas variantes normalizadas/deobfuscated de comando que o detector de padrões perigosos usa, então truques simples de quoting (`git pu""sh --force`) não escapam de uma regra.
- **YAML quoting:** sempre quote padrões. Um `*` leading bare é um alias YAML e falha ao parsear; `{`, `!` e `: ` têm seus próprios significados YAML. Aspas simples são mais seguras para conteúdo shell-ish.
- Regras deny aplicam a backends que alcançam o host (local, SSH, Docker com mount do host). Backends de container isolados pulam a guard stack inteira, como sempre fizeram — nada que executam pode tocar o host.
- Um comando negado retorna erro BLOCKED ao agente dizendo para não retry ou rephrase. Nada executa.

Como o resto da config de aprovação, mudanças têm efeito imediato (o cache de config é mtime-keyed) — sem restart de sessão necessário.

:::note Threat model
Regras deny são um guardrail contra um agente honesto-mas-errado, o mesmo threat model do detector de padrões perigosos. Não são um sandbox contra um processo deliberadamente adversarial — para isso, use um backend isolado (Docker, Modal) ou ambiente com egress restrito.
:::

### Approval Timeout {#approval-timeout}

Quando um prompt de comando perigoso aparece, o usuário tem um tempo configurável para responder. Se nenhuma resposta for dada dentro do timeout, o comando é **negado** por padrão (fail-closed).

Configure o timeout em `~/.hermes/config.yaml`:

```yaml
approvals:
  timeout: 300  # seconds (default: 300)
```

### What Triggers Approval {#what-triggers-approval}

Os seguintes padrões disparam prompts de aprovação (definidos em `tools/approval.py`):

| Pattern | Description |
|---------|-------------|
| `rm -r` / `rm --recursive` | Recursive delete |
| `rm ... /` | Delete in root path |
| `chmod 777/666` / `o+w` / `a+w` | World/other-writable permissions |
| `chmod --recursive` with unsafe perms | Recursive world/other-writable (long flag) |
| `chown -R root` / `chown --recursive root` | Recursive chown to root |
| `mkfs` | Format filesystem |
| `dd if=` | Disk copy |
| `> /dev/sd` | Write to block device |
| `DROP TABLE/DATABASE` | SQL DROP |
| `DELETE FROM` (without WHERE) | SQL DELETE without WHERE |
| `TRUNCATE TABLE` | SQL TRUNCATE |
| `> /etc/` | Overwrite system config |
| `systemctl stop/restart/disable/mask` | Stop/restart/disable system services |
| `kill -9 -1` | Kill all processes |
| `pkill -9` | Force kill processes |
| Fork bomb patterns | Fork bombs |
| `bash -c` / `sh -c` / `zsh -c` / `ksh -c` | Shell command execution via `-c` flag (including combined flags like `-lc`) |
| `python -e` / `perl -e` / `ruby -e` / `node -c` | Script execution via `-e`/`-c` flag |
| `curl ... \| sh` / `wget ... \| sh` | Pipe remote content to shell |
| `bash <(curl ...)` / `sh <(wget ...)` | Execute remote script via process substitution |
| `tee` to `/etc/`, `~/.ssh/`, `~/.hermes/.env` | Overwrite sensitive file via tee |
| `>` / `>>` to `/etc/`, `~/.ssh/`, `~/.hermes/.env` | Overwrite sensitive file via redirection |
| `xargs rm` | xargs with rm |
| `find -exec rm` / `find -delete` | Find with destructive actions |
| `cp`/`mv`/`install` to `/etc/` | Copy/move file into system config |
| `sed -i` / `sed --in-place` on `/etc/` | In-place edit of system config |
| `pkill`/`killall` hermes/gateway | Self-termination prevention |
| `gateway run` with `&`/`disown`/`nohup`/`setsid` | Prevents starting gateway outside service manager |

:::info
**Container bypass**: Quando executando em backends `docker`, `singularity`, `modal` ou `daytona`, verificações de comandos perigosos são **ignoradas** porque o container em si é o limite de segurança. Comandos destrutivos dentro de um container não podem prejudicar o host.
:::

### Approval Flow (CLI) {#approval-flow-cli}

Na CLI interativa, comandos perigosos mostram um prompt de aprovação inline:

```
  ⚠️  DANGEROUS COMMAND: recursive delete
      rm -rf /tmp/old-project

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
```

As quatro opções:

- **once** — permitir esta execução única
- **session** — permitir este padrão pelo resto da sessão
- **always** — adicionar à allowlist permanente (salvo em `config.yaml`)
- **deny** (default) — bloquear o comando

### Approval Flow (Gateway/Messaging) {#approval-flow-gatewaymessaging}

Em plataformas de mensagens, o agente envia os detalhes do comando perigoso ao chat e aguarda o usuário responder:

- Responda **yes**, **y**, **approve**, **ok** ou **go** para aprovar
- Responda **no**, **n**, **deny** ou **cancel** para negar

A variável de ambiente `HERMES_EXEC_ASK=1` é definida automaticamente ao executar o gateway.

### Permanent Allowlist {#permanent-allowlist}

Comandos aprovados com "always" são salvos em `~/.hermes/config.yaml`:

```yaml
# Permanently allowed dangerous command patterns
command_allowlist:
  - rm
  - systemctl
```

Esses padrões são carregados na inicialização e silenciosamente aprovados em todas as sessões futuras.

:::tip
Use `hermes config edit` para revisar ou remover padrões da sua allowlist permanente.
:::

## File Write Safety {#file-write-safety}

Antes de `write_file` ou `patch` tocar o disco, o Hermes verifica o path alvo contra uma denylist e um sandbox opcional. Writes bloqueados retornam erro ao agente imediatamente — **não há prompt de aprovação** e nenhuma forma de override pela UI de chat. O model ainda pode afirmar que a edição teve sucesso; quando `display.file_mutation_verifier` está on (padrão), confie no [file-mutation verifier footer](./configuration.md#file-mutation-verifier) em vez do resumo final do assistant.

### Protected paths (always blocked) {#protected-paths-always-blocked}

Estas categorias são sempre negadas, mesmo quando `HERMES_WRITE_SAFE_ROOT` está unset:

| Category | Examples |
|----------|----------|
| OS credential stores | `~/.ssh/`, `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `~/.netrc` |
| Hermes credential stores | `auth.json`, `.env`, `.anthropic_oauth.json`, `mcp-tokens/`, `pairing/` under HERMES_HOME (active profile and global root) |
| Project secret files | `.env`, `.env.local`, `.env.production`, `.envrc` anywhere on disk |

Paths sensíveis dentro do safe root ainda são bloqueados — apontar `HERMES_WRITE_SAFE_ROOT` para `$HOME` não permite escrever `~/.ssh/id_rsa`.

Violações de safe-root retornam `Write denied: '…' is outside HERMES_WRITE_SAFE_ROOT (…)`. Bloqueios de credential-path usam `Write denied: '…' is a protected system/credential file.`

### HERMES_WRITE_SAFE_ROOT (optional sandbox) {#hermes_write_safe_root-optional-sandbox}

Quando definido, `write_file` e `patch` só podem targetar paths dentro dos prefixos de diretório listados. Qualquer coisa fora é **hard-blocked** — não roteada pela aprovação de comandos perigosos.

- Definido automaticamente na [imagem Docker oficial](https://github.com/NousResearch/hermes-agent) (`HERMES_WRITE_SAFE_ROOT=/opt/data`)
- Suporta múltiplas roots separadas por `:` no Unix ou `;` no Windows
- **Não adicione a `~/.hermes/.env` casualmente.** Se você definir para um diretório de projeto, o agente não pode escrever em `~/.hermes/cron/jobs.json`, skills de profile ou outro estado Hermes fora desse prefixo

Para permitir workspace e Hermes home:

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

Desconfigure a variável para restaurar writes irrestritos (sujeito à denylist de protected-path). Referência completa: [HERMES_WRITE_SAFE_ROOT](../reference/environment-variables.md#hermes_write_safe_root).

### Cron and other Hermes state {#cron-and-other-hermes-state}

Não peça ao agente para `patch` `~/.hermes/cron/jobs.json` diretamente. Use a tool `cronjob`, [`hermes cron`](./features/cron.md) ou `/cron` — eles atualizam o job store pela API suportada. O mesmo se aplica a outros arquivos de controle Hermes quando write safety bloqueia edições diretas.

:::note Defense-in-depth, not a hard boundary
Write guards aplicam a `write_file` e `patch` apenas. A tool `terminal` executa como o mesmo usuário OS e ainda pode `cat` ou sobrescrever paths negados via comandos shell. A denylist reduz dano acidental e dá aos models um sinal claro de parada; não faz sandbox de um agente hostil ou comprometido.
:::

## User Authorization (Gateway) {#user-authorization-gateway}

Ao executar o messaging gateway, o Hermes controla quem pode interagir com o bot por um sistema de autorização em camadas.

### Authorization Check Order {#authorization-check-order}

O método `_is_user_authorized()` verifica nesta ordem:

1. **Per-platform allow-all flag** (ex., `DISCORD_ALLOW_ALL_USERS=true`)
2. **DM pairing approved list** (usuários aprovados via pairing codes)
3. **Platform-specific allowlists** (ex., `TELEGRAM_ALLOWED_USERS=12345,67890`)
4. **Global allowlist** (`GATEWAY_ALLOWED_USERS=12345,67890`)
5. **Global allow-all** (`GATEWAY_ALLOW_ALL_USERS=true`)
6. **Default: deny**

### Platform Allowlists {#platform-allowlists}

Defina user IDs permitidos como valores separados por vírgula em `~/.hermes/.env`:

```bash
# Platform-specific allowlists
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=111222333444555666
WHATSAPP_ALLOWED_USERS=15551234567
SLACK_ALLOWED_USERS=U01ABC123

# Cross-platform allowlist (checked for all platforms)
GATEWAY_ALLOWED_USERS=123456789

# Per-platform allow-all (use with caution)
DISCORD_ALLOW_ALL_USERS=true

# Global allow-all (use with extreme caution)
GATEWAY_ALLOW_ALL_USERS=true
```

:::warning
Se **nenhuma allowlist estiver configurada** e `GATEWAY_ALLOW_ALL_USERS` não estiver definido, **todos os usuários são negados**. O gateway registra um aviso na inicialização:

```
No user allowlists configured. All unauthorized users will be denied.
Set GATEWAY_ALLOW_ALL_USERS=true in ~/.hermes/.env to allow open access,
or configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id).
```
:::

### DM Pairing System {#dm-pairing-system}

Para autorização mais flexível, o Hermes inclui um sistema de pairing baseado em código. Em vez de exigir user IDs de antemão, usuários desconhecidos recebem um pairing code de uso único que o dono do bot aprova via CLI.

**Como funciona:**

1. Um usuário desconhecido envia DM ao bot
2. O bot responde com um pairing code de 8 caracteres
3. O dono do bot executa `hermes pairing approve <platform> <code>` na CLI
4. O usuário é permanentemente aprovado para aquela plataforma

Controle como DMs diretas não autorizadas são tratadas em `~/.hermes/config.yaml`:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `pair` é o padrão para plataformas DM estilo chat. DMs não autorizadas recebem resposta com pairing code.
- `ignore` descarta silenciosamente DMs não autorizadas.
- Email default para `ignore` a menos que `platforms.email.unauthorized_dm_behavior: pair` esteja definido, porque inboxes podem conter mail não relacionado não lido.
- Seções de plataforma sobrescrevem o default global, então você pode manter pairing no Telegram enquanto mantém WhatsApp silencioso.

**Recursos de segurança** (baseados em orientação OWASP + NIST SP 800-63-4):

| Feature | Details |
|---------|---------|
| Code format | 8-char from 32-char unambiguous alphabet (no 0/O/1/I) |
| Randomness | Cryptographic (`secrets.choice()`) |
| Code TTL | 1 hour expiry |
| Rate limiting | 1 request per user per 10 minutes |
| Pending limit | Max 3 pending codes per platform |
| Lockout | 5 failed approval attempts → 1-hour lockout |
| File security | `chmod 0600` on all pairing data files |
| Logging | Codes are never logged to stdout |

**Comandos CLI de pairing:**

```bash
# List pending and approved users
hermes pairing list

# Approve a pairing code
hermes pairing approve telegram ABC12DEF

# Revoke a user's access
hermes pairing revoke telegram 123456789

# Clear all pending codes
hermes pairing clear-pending
```

:::tip Docker users: run pairing commands as the `hermes` user
A imagem Docker oficial executa o gateway como o usuário não privilegiado `hermes`
(uid 10000) via `gosu`, mas `docker exec` default para root. Arquivos de
aprovação criados por root são escritos com mode `0600 root:root` e o gateway
não consegue lê-los — a aprovação é silenciosamente ignorada ([#10270][i10270]).

Sempre passe `-u hermes`:

```bash
docker exec -u hermes hermes-agent hermes pairing approve telegram ABC12DEF
```

Se você já executou o comando como root e o usuário ainda está não autorizado,
reinicie o container — o entrypoint corrigirá ownership no próximo start.

[i10270]: https://github.com/NousResearch/hermes-agent/issues/10270
:::

**Storage:** Dados de pairing são armazenados em `~/.hermes/pairing/` com arquivos JSON por plataforma:
- `{platform}-pending.json` — pending pairing requests
- `{platform}-approved.json` — approved users
- `_rate_limits.json` — rate limit and lockout tracking

## Container Isolation {#container-isolation}

Ao usar o backend de terminal `docker`, o Hermes aplica hardening de segurança estrito a todo container.

### Docker Security Flags {#docker-security-flags}

Todo container executa com estas flags (definidas em `tools/environments/docker.py`):

```python
_BASE_SECURITY_ARGS = [
    "--cap-drop", "ALL",                          # Drop ALL Linux capabilities
    "--cap-add", "DAC_OVERRIDE",                  # Root can write to bind-mounted dirs
    "--cap-add", "CHOWN",                         # Package managers need file ownership
    "--cap-add", "FOWNER",                        # Package managers need file ownership
    "--security-opt", "no-new-privileges",         # Block privilege escalation
    "--pids-limit", "256",                         # Limit process count
    "--tmpfs", "/tmp:rw,nosuid,size=512m",         # Size-limited /tmp
    "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",  # No-exec /var/tmp
]
```

`SETUID`/`SETGID` **não** estão na lista base — são adicionados condicionalmente quando o container inicia como root e um init/entrypoint deve dropar privilégios (o path s6 privilege-drop). São ignorados quando o container já executa como `--user` non-root. O tmpfs `/run` também é separado da lista base e montado por imagem (hardened `noexec` por padrão, `exec` apenas para imagens s6-overlay que exec de `/run`).

### Resource Limits {#resource-limits}

Recursos de container são configuráveis em `~/.hermes/config.yaml`:

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
  container_cpu: 1        # CPU cores
  container_memory: 5120  # MB (default 5GB)
  container_disk: 51200  # MB (default 50GB, requires overlay2 on XFS)
  container_persistent: true  # Persist filesystem across sessions
```

### Filesystem Persistence {#filesystem-persistence}

- **Persistent mode** (`container_persistent: true`): Bind-mounts `/workspace` e `/root` de `~/.hermes/sandboxes/docker/<task_id>/`
- **Ephemeral mode** (`container_persistent: false`): Usa tmpfs para workspace — tudo se perde no cleanup

:::tip
Para deploys de gateway em produção, use backend `docker`, `modal` ou `daytona` para isolar comandos do agente do seu sistema host. Isso elimina a necessidade de aprovação de comandos perigosos inteiramente.
:::

:::warning
Se você adicionar nomes a `terminal.docker_forward_env`, essas variáveis são intencionalmente injetadas no container para comandos de terminal. Isso é útil para credenciais específicas de tarefa como `GITHUB_TOKEN`, mas também significa que código executando no container pode lê-las e exfiltrá-las.
:::

## Terminal Backend Security Comparison {#terminal-backend-security-comparison}

| Backend | Isolation | Dangerous Cmd Check | Best For |
|---------|-----------|-------------------|----------|
| **local** | None — runs on host | ✅ Yes | Development, trusted users |
| **ssh** | Remote machine | ✅ Yes | Running on a separate server |
| **docker** | Container | ❌ Skipped (container is boundary) | Production gateway |
| **singularity** | Container | ❌ Skipped | HPC environments |
| **modal** | Cloud sandbox | ❌ Skipped | Scalable cloud isolation |
| **daytona** | Cloud sandbox | ❌ Skipped | Persistent cloud workspaces |

## Environment Variable Passthrough {#environment-variable-passthrough}

Tanto `execute_code` quanto `terminal` removem variáveis de ambiente sensíveis de processos filhos para prevenir exfiltração de credenciais por código gerado por LLM. Porém, skills que declaram `required_environment_variables` legitimamente precisam de acesso a essas vars.

### How It Works {#how-it-works}

Dois mecanismos permitem variáveis específicas passarem pelos filtros de sandbox:

**1. Skill-scoped passthrough (automatic)**

Quando uma skill é carregada (via `skill_view` ou o comando `/skill`) e declara `required_environment_variables`, quaisquer dessas vars que estejam de fato definidas no ambiente são automaticamente registradas como passthrough. Vars ausentes (ainda em estado setup-needed) **não** são registradas.

```yaml
# In a skill's SKILL.md frontmatter
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
```

Após carregar esta skill, `TENOR_API_KEY` passa para `execute_code`, `terminal` (local), **e backends remotos (Docker, Modal)** — sem configuração manual necessária.

:::info Docker & Modal
Antes da v0.5.1, `forward_env` do Docker era um sistema separado do skill passthrough. Agora estão merged — env vars declaradas por skill são automaticamente forwarded para containers Docker e sandboxes Modal sem precisar adicioná-las manualmente a `docker_forward_env`.
:::

**2. Config-based passthrough (manual)**

Para env vars não declaradas por nenhuma skill, adicione-as a `terminal.env_passthrough` em `config.yaml`:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_KEY
    - ANOTHER_TOKEN
```

### Credential File Passthrough (OAuth tokens, etc.) {#credential-file-passthrough}

Algumas skills precisam de **arquivos** (não só env vars) no sandbox — por exemplo, Google Workspace armazena OAuth tokens como `google_token.json` sob o `HERMES_HOME` do profile ativo. Skills declaram estes no frontmatter:

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

Quando carregada, o Hermes verifica se esses arquivos existem no `HERMES_HOME` do profile ativo e os registra para mount:

- **Docker**: Read-only bind mounts (`-v host:container:ro`)
- **Modal**: Montados na criação do sandbox + synced antes de cada comando (lida com OAuth setup mid-session)
- **Local**: Nenhuma ação necessária (arquivos já acessíveis)

Você também pode listar credential files manualmente em `config.yaml`:

```yaml
terminal:
  credential_files:
    - google_token.json
    - my_custom_oauth_token.json
```

Paths são relativos a `~/.hermes/`. Arquivos são montados em `/root/.hermes/` dentro do container. Esta lista é lida por `tools/credential_files.py` (`terminal.credential_files`) — vive sob o bloco `terminal:` mas é carregada pelo módulo credential-files, não pelo core terminal backend, então não faz parte do snapshot `DEFAULT_CONFIG` bundled.

### What Each Sandbox Filters {#what-each-sandbox-filters}

| Sandbox | Default Filter | Passthrough Override |
|---------|---------------|---------------------|
| **execute_code** | Blocks vars containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `PASSWD`, `AUTH` in name; only allows safe-prefix vars through | ✅ Passthrough vars bypass both checks |
| **terminal** (local) | Blocks explicit Hermes infrastructure vars (provider keys, gateway tokens, tool API keys) | ✅ Passthrough vars bypass the blocklist |
| **terminal** (Docker) | No host env vars by default | ✅ Passthrough vars + `docker_forward_env` forwarded via `-e` |
| **terminal** (Modal) | No host env/files by default | ✅ Credential files mounted; env passthrough via sync |
| **MCP** | Blocks everything except safe system vars + explicitly configured `env` | ❌ Not affected by passthrough (use MCP `env` config instead) |

### Security Considerations {#security-considerations}

- O passthrough afeta apenas vars que você ou suas skills declaram explicitamente — a postura de segurança padrão permanece inalterada para código arbitrário gerado por LLM
- Credential files são montados **read-only** em containers Docker
- Skills Guard escaneia conteúdo de skill por padrões suspeitos de acesso a env antes da instalação
- Vars ausentes/unset nunca são registradas (você não pode vazar o que não existe)
- Secrets de infraestrutura Hermes (provider API keys, gateway tokens) nunca devem ser adicionados a `env_passthrough` — têm mecanismos dedicados

## MCP Credential Handling {#mcp-credential-handling}

Subprocessos de servidor MCP (Model Context Protocol) recebem um **ambiente filtrado** para prevenir vazamento acidental de credenciais.

### Safe Environment Variables {#safe-environment-variables}

Apenas estas variáveis são passadas do host para subprocessos stdio MCP:

```
PATH, HOME, USER, LANG, LC_ALL, TERM, SHELL, TMPDIR
```

Mais quaisquer variáveis `XDG_*`. Todas as outras variáveis de ambiente (API keys, tokens, secrets) são **removidas**.

Variáveis explicitamente definidas no `env` config do servidor MCP são passadas:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."  # Only this is passed
```

### Credential Redaction {#credential-redaction}

Mensagens de erro de tools MCP são sanitizadas antes de serem retornadas ao LLM. Os seguintes padrões são substituídos por `[REDACTED]`:

- GitHub PATs (`ghp_...`)
- OpenAI-style keys (`sk-...`)
- Bearer tokens
- Parâmetros `token=`, `key=`, `API_KEY=`, `password=`, `secret=`

### Website Access Policy {#website-access-policy}

Você pode restringir quais websites o agente pode acessar por suas web e browser tools. Isso é útil para impedir o agente de acessar serviços internos, painéis admin ou outras URLs sensíveis.

```yaml
# In ~/.hermes/config.yaml
security:
  website_blocklist:
    enabled: true
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

Quando uma URL bloqueada é solicitada, a tool retorna um erro explicando que o domínio está bloqueado por policy. A blocklist é enforced em `web_search`, `web_extract`, `browser_navigate` e todas as tools capazes de URL.

Veja [Website Blocklist](/user-guide/configuration#website-blocklist) no guia de configuração para detalhes completos.

### SSRF Protection {#ssrf-protection}

Todas as tools capazes de URL (web search, web extract, vision, browser) validam URLs antes de fetch para prevenir ataques Server-Side Request Forgery (SSRF). Endereços bloqueados incluem:

- **Private networks** (RFC 1918): `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- **Loopback**: `127.0.0.0/8`, `::1`
- **Link-local**: `169.254.0.0/16` (includes cloud metadata at `169.254.169.254`)
- **CGNAT / shared address space** (RFC 6598): `100.64.0.0/10` (Tailscale, WireGuard VPNs)
- **Cloud metadata hostnames**: `metadata.google.internal`, `metadata.goog`
- **Reserved, multicast, and unspecified addresses**

Proteção SSRF está sempre ativa para uso internet-facing e falhas DNS são tratadas como bloqueadas (fail-closed). Cadeias de redirect são re-validadas a cada hop para prevenir bypasses baseados em redirect.

#### Intentionally allowing private URLs {#intentionally-allowing-private-urls}

Alguns setups legitimamente precisam de acesso a URLs privadas/internas — redes domésticas que resolvem `home.arpa` para espaço RFC 1918, endpoints Ollama/llama.cpp só na LAN, wikis internas, debugging de cloud metadata, e similares. Para esses casos há um opt-out global:

```yaml
security:
  allow_private_urls: true   # default: false
```

Quando on, web tools, browser, fetches de URL vision e downloads de mídia do gateway não rejeitam mais destinos RFC 1918 / loopback / link-local / CGNAT / cloud-metadata. **Este é um limite de confiança deliberado** — habilite apenas em máquinas onde o agente executando URLs arbitrárias prompt-injected contra a rede local é um risco aceitável. Gateways public-facing devem deixar off.

O guard de host-substring (que bloqueia truques de domínio Unicode lookalike mesmo quando o IP subjacente é público) permanece on independente desta setting.

### Tirith Pre-Exec Security Scanning {#tirith-pre-exec-security-scanning}

O Hermes integra [tirith](https://github.com/sheeki03/tirith) para scan de comandos em nível de conteúdo antes da execução. Tirith detecta ameaças que pattern matching sozinho não pega:

- Homograph URL spoofing (internationalized domain attacks)
- Pipe-to-interpreter patterns (`curl | bash`, `wget | sh`)
- Terminal injection attacks

Tirith auto-instala de releases GitHub no primeiro uso com verificação SHA-256 checksum (e verificação de provenance cosign se cosign estiver disponível).

```yaml
# In ~/.hermes/config.yaml
security:
  tirith_enabled: true       # Enable/disable tirith scanning (default: true)
  tirith_path: "tirith"      # Path to tirith binary (default: PATH lookup)
  tirith_timeout: 5          # Subprocess timeout in seconds
  tirith_fail_open: true     # Allow execution when tirith is unavailable (default: true)
```

Quando `tirith_fail_open` é `true` (padrão), comandos prosseguem se tirith não estiver instalado ou expirar. Defina `false` em ambientes high-security para bloquear comandos quando tirith estiver indisponível.

Tirith shipa binários prebuilt para Linux (x86_64 / aarch64) e macOS (x86_64 / arm64). Em plataformas sem binário prebuilt (Windows, etc.), tirith é silenciosamente ignorado — guards de pattern matching ainda rodam, e a CLI não exibe banner "unavailable". Para usar tirith no Windows, execute Hermes sob WSL.

O veredicto de Tirith integra com o fluxo de aprovação: comandos safe passam, enquanto comandos suspicious e blocked disparam aprovação do usuário com os achados completos do tirith (severity, title, description, safer alternatives). Usuários podem aprovar ou negar — a escolha padrão é deny para manter cenários unattended seguros.

### Context File Injection Protection {#context-file-injection-protection}

Context files (AGENTS.md, .cursorrules, SOUL.md) são escaneados por prompt injection antes de serem incluídos no system prompt. O scanner verifica:

- Instruções para ignorar/disregard prior instructions
- HTML comments ocultos com keywords suspeitas
- Tentativas de ler secrets (`.env`, `credentials`, `.netrc`)
- Exfiltração de credenciais via `curl`
- Caracteres Unicode invisíveis (zero-width spaces, bidirectional overrides)

Arquivos bloqueados mostram um aviso:

```
[BLOCKED: AGENTS.md contained potential prompt injection (prompt_injection). Content not loaded.]
```

## Best Practices for Production Deployment {#best-practices-for-production-deployment}

### Gateway Deployment Checklist {#gateway-deployment-checklist}

1. **Defina allowlists explícitas** — nunca use `GATEWAY_ALLOW_ALL_USERS=true` em produção
2. **Use container backend** — defina `terminal.backend: docker` em config.yaml
3. **Restrinja resource limits** — defina CPU, memória e limites de disco apropriados
4. **Armazene secrets com segurança** — mantenha API keys em `~/.hermes/.env` com permissões de arquivo adequadas
5. **Habilite DM pairing** — use pairing codes em vez de hardcode user IDs quando possível
6. **Revise command allowlist** — audite periodicamente `command_allowlist` em config.yaml
7. **Defina `terminal.cwd`** — não deixe o agente operar de diretórios sensíveis
8. **Execute como non-root** — nunca execute o gateway como root
9. **Monitore logs** — verifique `~/.hermes/logs/` por tentativas de acesso não autorizado
10. **Mantenha atualizado** — execute `hermes update` regularmente para patches de segurança

### Securing API Keys {#securing-api-keys}

```bash
# Set proper permissions on the .env file
chmod 600 ~/.hermes/.env

# Keep separate keys for different services
# Never commit .env files to version control
```

### Network Isolation {#network-isolation}

Para máxima segurança, execute o gateway em uma máquina ou VM separada. Defina `terminal.backend: ssh` em `config.yaml`, depois forneça detalhes de host via variáveis de ambiente em `~/.hermes/.env`:

```yaml
# ~/.hermes/config.yaml
terminal:
  backend: ssh
```

```bash
# ~/.hermes/.env
TERMINAL_SSH_HOST=agent-worker.local
TERMINAL_SSH_USER=hermes
TERMINAL_SSH_KEY=~/.ssh/hermes_agent_key
```

Os detalhes de conexão SSH vivem em `.env` (não `config.yaml`) para não serem checked in ou compartilhados junto com exports de profile. Isso mantém as conexões de mensagens do gateway separadas da execução de comandos do agente.

## Supply-chain advisory checking {#supply-chain-advisory-checking}

O Hermes vem com um scanner de advisory built-in que sinaliza pacotes Python no venv ativo que correspondem a um catálogo curado de versões conhecidamente comprometidas (supply-chain worms como o envenenamento `mistralai 2.4.6` de maio de 2026). A implementação vive em `hermes_cli/security_advisories.py`.

Como roda:

- **Banner de startup da CLI.** Um aviso de uma linha é impresso se qualquer advisory corresponder, com ponteiro para `hermes doctor` para a remediação completa.
- **`hermes doctor`.** Mostra todo advisory ativo com especificidades de versão e instruções de remediação de 2-4 passos.
- **Startup do gateway.** Logado em `gateway.log`; a primeira mensagem interativa recebe um banner curto de operador.

Cada advisory carrega um id estável. Depois de ler e agir sobre ele você pode dismiss para sempre:

```bash
hermes doctor --ack <advisory-id>
```

O ack persiste em `config.security.acked_advisories` e sobrevive restart. Advisories antigos são intencionalmente **não** removidos do catálogo — deixá-los no lugar mantém installs frescos avisados sobre versões historicamente envenenadas que ainda podem estar em cache em um mirror privado.

A verificação em si é só stdlib e roda de um lookup `importlib.metadata.version()` por advisory, então é seguro rodar a cada startup.

### Lazy install of optional dependencies {#lazy-install-of-optional-dependencies}

Muitas features (Mistral TTS, ElevenLabs, Honcho memory, Bedrock, Slack, Matrix, …) dependem de pacotes Python que nem todo usuário precisa. O Hermes instala estes **lazily** no primeiro uso em vez de eagerly sob `hermes-agent[all]`. A implementação vive em `tools/lazy_deps.py`.

O trade-off que isso corrige:

- **Fragilidade.** Quando uma dependência transitiva de um extra fica indisponível no PyPI (quarentined por malware, yanked, upload quebrado), o resolve inteiro de `[all]` falharia e installs frescos cairiam silenciosamente para um tier stripped — perdendo 10+ extras não relacionados de uma vez. Lazy install isola cada backend para um dep envenenado não quebrar features não relacionadas.
- **Bloat.** Um usuário que só fala com um provider não puxa mais centenas de pacotes que nunca importará.

Como funciona:

1. Um módulo backend chama `ensure("feature.name")` no topo do seu first-import path.
2. Se os deps estão ausentes, `ensure` verifica `security.allow_lazy_installs` em `config.yaml` (padrão `true`) e executa `pip install` scoped ao venv para as specs allowlisted.
3. Se o install falha ou o usuário desabilitou lazy installs, a chamada levanta `FeatureUnavailable` com o stderr pip real e ponteiro em `hermes tools`.

Garantias de segurança enforced por `tools/lazy_deps.py`:

| Guarantee | What it means |
|---|---|
| Venv-scoped only | Installs target `sys.executable` in the active venv — never the system Python |
| PyPI by name only | Specs accept `"package>=1.0,<2"` syntax. No `--index-url`, `git+https://`, or file: paths — a malicious `config.yaml` cannot redirect the install |
| Allowlist | Only specs that appear in the in-tree `LAZY_DEPS` map can be installed via this path. A typo in a feature name does NOT get install-anything semantics |
| Opt-out | Set `security.allow_lazy_installs: false` to disable runtime installs entirely. Useful for restricted networks or strict security postures |
| No silent retries | Failures surface as `FeatureUnavailable` — no caching of bad state, no retry storms |

Para desabilitar runtime installs:

```yaml
# ~/.hermes/config.yaml
security:
  allow_lazy_installs: false
```

Quando desabilitado, backends que precisam de deps opcionais dirão ao usuário para executar o install manualmente (`pip install …`) ou escolher outro backend via `hermes tools`.
