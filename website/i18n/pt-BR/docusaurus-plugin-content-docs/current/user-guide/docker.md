---
sidebar_position: 7
title: "Docker"
description: "Executando o Hermes Agent no Docker e usando Docker como backend de terminal"
---

# Hermes Agent — Docker

Há duas formas distintas em que o Docker intersecta com o Hermes Agent:

1. **Executar o Hermes NO Docker** — o agente em si roda dentro de um container (foco principal desta página)
2. **Docker como backend de terminal** — o agente roda no seu host mas executa todo comando dentro de um único container sandbox Docker persistente que sobrevive entre tool calls, `/new` e subagents pela vida do processo Hermes (veja [Configuration → Docker Backend](./configuration.md#docker-backend))

Esta página cobre a opção 1. O container armazena todos os dados do usuário (config, API keys, sessões, skills, memórias) em um único diretório montado do host em `/opt/data`. A imagem em si é stateless e pode ser atualizada puxando uma nova versão sem perder configuração.

## Quick start {#quick-start}

Se é sua primeira vez executando o Hermes Agent, crie um diretório de dados no host e inicie o container interativamente para rodar o setup wizard:

:::caution Evite consoles VPS baseados em browser para os comandos de instalação
Alguns provedores VPS (Hetzner Cloud, e vários outros) oferecem um console
baseado em browser para gerenciar hosts. Esses consoles transmitem caracteres
especiais incorretamente — `:` pode chegar como `;`, `@` pode ser mal renderizado, e layouts
de teclado não inglês saem pior — o que corrompe silenciosamente argumentos `docker run`
como `-v ~/.hermes:/opt/data`, `-e KEY=value` e API keys / tokens colados.

**Conecte via SSH** (`ssh root@<host>`) para entrada de comandos segura para copy-paste.
Se precisar usar o console browser, digite os comandos manualmente
em vez de colar, e verifique cada `:`, `@`, `=` e `/` no
resultado antes de pressionar Enter.
:::

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

Isso coloca você no setup wizard, que pedirá suas API keys e as escreverá em `~/.hermes/.env`. Você só precisa fazer isso uma vez. É altamente recomendado configurar um sistema de chat para o gateway funcionar neste ponto.

:::tip
Dentro do container, execute `hermes setup --portal` uma vez — o refresh token persiste no volume montado `~/.hermes`. Veja [Nous Portal](/integrations/nous-portal).
:::

## Running in gateway mode {#running-in-gateway-mode}

Uma vez configurado, execute o container em background como gateway persistente (Telegram, Discord, Slack, WhatsApp, etc.):

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

A porta 8642 expõe o [API server compatível com OpenAI](./features/api-server.md) do gateway e o health endpoint. É opcional se você usa apenas plataformas de chat (Telegram, Discord, etc.), mas obrigatória se quiser que o dashboard ou tools externas alcancem o gateway.

:::tip Gateway runs supervised
Dentro da imagem Docker oficial, `gateway run` é **automaticamente supervisionado pelo s6-overlay**: se o processo do gateway crashar ele é reiniciado em alguns segundos sem perder o container, e o dashboard (quando `HERMES_DASHBOARD=1` está definido) é supervisionado junto. O processo CMD `gateway run` em si é um heartbeat `sleep infinity` que mantém o container vivo enquanto o s6 gerencia o processo real do gateway — então `docker stop` ainda desliga tudo limpo, mas `docker logs` mostra a saída do gateway supervisionado.

Você verá um breadcrumb de uma linha em `docker logs` confirmando o upgrade. Para optar por sair — e obter a semântica histórica "gateway é o main process do container, exit do container = exit do gateway" — passe `--no-supervise` ou defina `HERMES_GATEWAY_NO_SUPERVISE=1`. O opt-out é útil para smoke tests CI que querem o container sair com o status code do gateway; para deploys de produção o default supervisionado é estritamente melhor.

Este comportamento aplica-se apenas à imagem baseada em s6. Imagens anteriores (baseadas em tini) ainda executam `gateway run` como foreground main process.
:::

:::note Where gateway logs go
Veja a seção [Where the logs go](#where-the-logs-go) abaixo para o mapa completo de roteamento (gateways por profile, dashboard, boot reconciler, `docker logs` container-wide).
:::

:::note Tool-loop hard stops for unattended gateways
A setting `tool_loop_guardrails.hard_stop_enabled` default para `false`, o que é razoável para sessões CLI e TUI interativas onde uma pessoa vê avisos repetidos de tool-call. Em deploys de gateway ou server unattended, avisos sozinhos podem não parar um agente preso em um loop repetido de tool-call. Operadores que querem comportamento circuit-breaker devem habilitar explicitamente hard stops no `config.yaml` do profile:

```yaml
tool_loop_guardrails:
  hard_stop_enabled: true
  hard_stop_after:
    exact_failure: 5
    idempotent_no_progress: 5
```
:::

Nota: o API server é gated em `API_SERVER_ENABLED=true`. Para expô-lo além de `127.0.0.1` dentro do container, também defina `API_SERVER_HOST=0.0.0.0` e uma `API_SERVER_KEY` (mínimo 8 caracteres — gere uma com `openssl rand -hex 32`). Exemplo:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY="$(openssl rand -hex 32)" \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

Abrir qualquer porta em uma máquina exposta à internet é risco de segurança. Você não deve fazer isso a menos que entenda os riscos.

## Running the dashboard {#running-the-dashboard}

O web dashboard built-in roda como serviço s6-rc supervisionado junto ao gateway no mesmo container. Defina `HERMES_DASHBOARD=1` para subi-lo:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -p 9119:9119 \
  -e HERMES_DASHBOARD=1 \
  nousresearch/hermes-agent gateway run
```

O dashboard é supervisionado pelo s6 — se crashar, `s6-supervise` o reinicia automaticamente após um backoff curto. stdout/stderr do dashboard é forwarded para `docker logs <container>` (sem prefix; a saída própria do gateway agora vive em um arquivo s6-log por profile — veja [Where the logs go](#where-the-logs-go) abaixo — então os dois streams não colidem).

| Environment variable | Description | Default |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | Set to `1` (or `true` / `yes`) to enable the supervised dashboard service | *(unset — service is registered but stays down)* |
| `HERMES_DASHBOARD_HOST` | Bind address for the dashboard HTTP server | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | Port for the dashboard HTTP server | `9119` |
| `HERMES_DASHBOARD_INSECURE` | **Deprecated / no-op.** Formerly bypassed the auth gate; as of the June 2026 hardening it no longer disables authentication. A non-loopback bind always requires an auth provider | *(ignored — configure a provider instead)* |

O dashboard dentro do container default para bind `0.0.0.0` — sem isso, a porta publicada `-p 9119:9119` não seria alcançável do host. Para restringir o bind ao loopback do container (para setups sidecar / reverse-proxy), defina `HERMES_DASHBOARD_HOST=127.0.0.1`.

O auth gate do dashboard engaja automaticamente quando ambos são verdadeiros:

1. O bind host é non-loopback (ex. o default `0.0.0.0` dentro do container), **e**
2. Um plugin `DashboardAuthProvider` está registrado.

Há três formas bundled de satisfazer a segunda condição:

- **Username/password** — o mais simples para um container self-hosted / on-prem / homelab em rede confiável ou atrás de VPN: defina `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` (e `HERMES_DASHBOARD_BASIC_AUTH_SECRET` para sessões estáveis entre restarts). Não adequado para exposição direta à internet pública.
- **OAuth (Nous Portal)** — para deploys hosted/public: o provider `dashboard_auth/nous` ativa sempre que `HERMES_DASHBOARD_OAUTH_CLIENT_ID` está definido.
- **OIDC self-hosted** — para autenticar contra seu próprio identity provider via OpenID Connect padrão: o provider `dashboard_auth/self_hosted` ativa quando `HERMES_DASHBOARD_OIDC_ISSUER` + `HERMES_DASHBOARD_OIDC_CLIENT_ID` estão definidos.

Qualquer que escolha, o gate redireciona callers para uma login page antes de alcançarem qualquer rota protegida. Veja [Web Dashboard → Authentication](features/web-dashboard.md#authentication-gated-mode) para os três providers.

Se nenhum provider estiver registrado e o bind for non-loopback, o dashboard **falha fechado na inicialização** com um erro específico apontando para a env var ausente. Não há mais escape hatch que serve o dashboard sem autenticação em bind público: `HERMES_DASHBOARD_INSECURE=1` agora é um no-op deprecated (loga um aviso e é ignorado). Configure um provider, ou faça bind `HERMES_DASHBOARD_HOST=127.0.0.1` e alcance o dashboard via túnel SSH / Tailscale.

:::warning Why `--insecure` was removed
Um dashboard público sem autenticação foi o ponto de entrada para a campanha de persistência MCP-config de junho de 2026: scanners de internet alcançaram dashboards expostos (e API servers OpenAI) e conduziram o agente a plantar um backdoor SSH-key. O auth gate agora é obrigatório em todo bind non-loopback. Para uma caixa homelab/LAN confiável, o provider username/password bundled (`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `_PASSWORD`) é a forma zero-infra de satisfazê-lo.
:::

Executar o dashboard como container separado **é** suportado quando esse container compartilha o PID e network namespace do host (ex. `network_mode: host`, como o `docker-compose.yml` do repo faz — veja seu serviço `dashboard`). Sua detecção de liveness do gateway exige um PID namespace compartilhado com o processo gateway, então a limitação aplica-se apenas a dashboards executados em containers bridge-network isolados sem PID namespace compartilhado.

## Running interactively (CLI chat) {#running-interactively-cli-chat}

Para abrir uma sessão de chat interativa contra um diretório de dados em execução:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

Ou se você já abriu um terminal no seu container em execução (via Docker Desktop por exemplo), basta executar:

```sh
/opt/hermes/.venv/bin/hermes
```

## Persistent volumes {#persistent-volumes}

O volume `/opt/data` é a única fonte de verdade para todo estado Hermes. Mapeia para o diretório `~/.hermes/` do host e contém:

| Path | Contents |
|------|----------|
| `.env` | API keys and secrets |
| `config.yaml` | All Hermes configuration |
| `SOUL.md` | Agent personality/identity |
| `sessions/` | Conversation history |
| `memories/` | Persistent memory store |
| `skills/` | Installed skills |
| `home/` | Per-profile HOME for Hermes tool subprocesses (`git`, `ssh`, `gh`, `npm`, and skill CLIs) |
| `cron/` | Scheduled job definitions |
| `hooks/` | Event hooks |
| `logs/` | Runtime logs |
| `skins/` | Custom CLI skins |

### Immutable install tree {#immutable-install-tree}

Em imagens Docker hosted e publicadas, `/opt/hermes` é a árvore de instalação da aplicação. É root-owned e read-only para o usuário runtime `hermes`, então agent turns, sessões de gateway, ações do dashboard e comandos normais `docker exec hermes hermes ...` não podem editar o source core, `.venv` bundled, `node_modules` ou bundle TUI in place.

Todo estado Hermes mutável pertence a `/opt/data`: config, `.env`, profiles, skills, memórias, sessões, logs, uploads do dashboard, plugins e outros arquivos gerenciados pelo usuário. A imagem também desabilita writes runtime de `.pyc` e lazy dependency installs do Hermes em `/opt/hermes`; dependências opcionais de plataforma necessárias pela imagem publicada devem ser baked na imagem ou instaladas via novo build de imagem.

Em imagens hosted/publicadas, auto-melhoria do agente é scoped a skills, memória, plugins e config sob `/opt/data`. O source core instalado sob `/opt/hermes` é imutável; mudanças core são feitas via PRs no repo e shipped atualizando a imagem, não editando live a instalação em execução.

Se um operador precisa reparar ou inspecionar arquivos fora de `/opt/data`, use um shell root intencionalmente. O shim `hermes` normalmente faz `docker exec hermes hermes ...` voltar ao usuário runtime; defina `HERMES_DOCKER_EXEC_AS_ROOT=1` para uma invocação root one-off quando precisar explicitamente de semântica root.

Skill CLIs que armazenam credenciais sob `~` devem ser inicializadas contra o HOME do subprocess, não só a raiz do data-volume. Por exemplo, a [skill xurl](./skills/bundled/social-media/social-media-xurl.md) armazena estado OAuth em `~/.xurl`; no layout Docker oficial, tool calls Hermes leem isso como `/opt/data/home/.xurl`, então execute auth xurl manual com `HOME=/opt/data/home` e verifique com `HOME=/opt/data/home xurl auth status`.

:::warning
Nunca execute dois containers **gateway** Hermes contra o mesmo diretório de dados simultaneamente — arquivos de sessão e memory stores não foram projetados para acesso de escrita concorrente.
:::

## Multi-profile support {#multi-profile-support}

O Hermes suporta [múltiplos profiles](../reference/profile-commands.md) — subdiretórios `~/.hermes/` separados que permitem executar agentes independentes (SOUL, skills, memória, sessões, credenciais diferentes) de uma única instalação. **Dentro da imagem Docker oficial, a árvore de supervisão s6 trata cada profile como serviço supervisionado de primeira classe**, então o deploy recomendado é **um container hospedando todos os profiles**.

Cada profile criado com `hermes profile create <name>` recebe:

- Um slot de serviço s6 dedicado em `/run/service/gateway-<name>/`, registrado dinamicamente pelo runtime — sem rebuild de container necessário.
- Auto-restart em crash, backoff gerenciado por `s6-supervise`.
- Logs rotacionados por profile em `${HERMES_HOME}/logs/gateways/<name>/current` (10 archives × 1 MB cada).
- Persistência de estado entre restarts de container: o reconciler de boot lê `gateway_state.json` de cada diretório de profile e sobe o slot de volta apenas para profiles cujo último estado registrado era `running`. Apenas um gateway que você parou explicitamente (`hermes gateway stop`) permanece down após restart — restart de container, upgrade de imagem ou exit inesperado deixa o estado registrado como `running`, então o gateway auto-inicia no próximo boot.

Os comandos de lifecycle que você executaria no host funcionam igual de dentro do container:

```sh
# Create a profile — registers the gateway-<name> s6 slot.
docker exec hermes hermes profile create coder

# Start / stop / restart — dispatches s6-svc; the gateway lifecycle survives docker restart.
docker exec hermes hermes -p coder gateway start
docker exec hermes hermes -p coder gateway stop
docker exec hermes hermes -p coder gateway restart

# Status — reports `Manager: s6 (container supervisor)` inside the container.
docker exec hermes hermes -p coder gateway status

# Remove a profile — tears down the s6 slot too.
docker exec hermes hermes profile delete coder
```

Por baixo dos panos, `hermes gateway start/stop/restart` dentro do container é interceptado e roteado para `s6-svc` contra o diretório de serviço certo; você não precisa aprender os comandos s6 diretamente. Para estado raw do supervisor, use `/command/s6-svstat /run/service/gateway-<name>` (note que `/command/` está no PATH apenas para processos spawned pela árvore de supervisão — ao chamar de `docker exec`, passe o path absoluto).

### Reaching more than one profile from outside the container {#reaching-more-than-one-profile-from-outside-the-container}

Duas superfícies diferentes alcançam o gateway de um profile de fora, e se comportam diferente — não as confunda:

**Hermes Desktop (e web dashboard).** A conexão **Remote Gateway** do app Desktop fala com um backend `hermes dashboard` (porta padrão **9119**, habilitada por `HERMES_DASHBOARD=1`) — *não* o API server OpenAI. Um backend dashboard serve **todo** profile co-localizado: o seletor de profile do app envia o profile alvo com cada request e o backend abre o `HERMES_HOME` daquele profile no disco. Então você **não** precisa de uma segunda porta — ou segunda conexão — por profile para Desktop; uma conexão `:9119` cobre todos via seletor.

**Clientes API compatíveis com OpenAI (Open WebUI, LobeChat, `/v1/...`).** Estes falam com o **API server** de cada profile, que faz bind na **porta 8642 para todo profile** (resolvida de `API_SERVER_PORT` / `platforms.api_server.extra.port` — não há auto-alocação e nenhuma chave `config.yaml`/`gateway.port`). Se quiser um cliente alcançando um *segundo* profile específico, dê a aquele profile um `API_SERVER_PORT` distinto no **próprio** `.env`, senão seu gateway tenta bind 8642 também e conflita com o profile default:

```sh
# Create the profile (registers its gateway-<name> s6 slot)
docker exec hermes hermes profile create work

# Point its API server at a free port (write to the profile's own .env)
cat >> /opt/data/profiles/work/.env <<'EOF'
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
EOF

docker exec hermes hermes -p work gateway restart
```

Mantenha `API_SERVER_PORT` no **próprio** `.env` de cada profile, nunca no bloco `environment:` container-wide — um valor global forçaria todo profile na mesma porta e colidiriam. Com bridge networking, publique a porta extra em `docker-compose.yml` (`- "8643:8643"`); com `network_mode: host` já é alcançável no host. A conexão 8642 do profile default permanece intacta.

### Why one container with many profiles, not many containers {#why-one-container-with-many-profiles-not-many-containers}

Antes da migração s6, "um container por profile" era o padrão recomendado porque não havia supervisor in-container para gerenciar múltiplos gateways. Com s6 como PID 1, isso não é mais necessário, e o layout single-container é mais simples em quase toda dimensão:

| | One container, many profiles | One container per profile |
|---|---|---|
| Disk overhead | One image, one bundled venv, one Playwright cache | N images / N caches |
| Memory overhead | Shared Python interpreter cache, shared node_modules | Duplicated per container |
| Profile creation | `docker exec ... hermes profile create <name>` (seconds) | New `docker run` invocation + port allocation + bind-mount config |
| Per-profile crash recovery | `s6-supervise` auto-restart | Docker's `--restart unless-stopped` (slower, kills sibling work) |
| Logs | Per-profile rotated file via `s6-log`, plus container-boot audit log | `docker logs <name>` per container — no built-in rotation |
| Backup | One `~/.hermes` directory | N directories to coordinate |

O profile default (`default`) é sempre registrado no primeiro boot, então um container fresh shipa com um gateway supervisionado out of the box. Profiles adicionais são adds puramente runtime.

### When you DO want a separate container {#when-you-do-want-a-separate-container}

Profile-in-container é o default. Execute um container separado por profile apenas quando tiver um motivo específico:

- **Isolamento de recursos por workload** — ex. uma sessão browser-tool runaway no profile A não deve conseguir OOM o profile B. Containers dão `--memory` / `--cpus` por profile.
- **Image pinning independente** — tags de imagem upstream diferentes por workload.
- **Segmentação de rede** — Docker networks distintas por profile (ex. um customer-facing, um internal).
- **Compliance / blast radius** — credenciais distintas nunca compartilham uma process tree em nível OS.

Nesses casos, declare um serviço por profile com `container_name`, `volumes` e `ports` distintos:

```yaml
services:
  hermes-work:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-work
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes-work:/opt/data

  hermes-personal:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-personal
    restart: unless-stopped
    command: gateway run
    ports:
      - "8643:8642"
    volumes:
      - ~/.hermes-personal:/opt/data
```

O aviso de [Persistent volumes](#persistent-volumes) ainda se aplica: nunca aponte dois containers ao mesmo diretório `~/.hermes` simultaneamente. O supervisor s6 dentro de cada container gerencia seu próprio conjunto de profiles; compartilhamento cross-container de um data volume corrompe arquivos de sessão e memory stores.

## Where the logs go {#where-the-logs-go}

O container s6 tem quatro superfícies de log distintas, e "por que meu gateway não aparece em `docker logs`" é uma surpresa comum. Cheatsheet:

| Source | Where it lands | How to read it |
|---|---|---|
| **Per-profile gateway** (`hermes gateway run` and per-profile gateways under s6) | Tee'd to two places: `docker logs <container>` (real time, no extra prefix) **and** `${HERMES_HOME}/logs/gateways/<profile>/current` (rotated, ISO-8601 timestamped, 10 archives × 1 MB each) | `docker logs -f hermes` or `tail -F ~/.hermes/logs/gateways/default/current` on the host |
| **Dashboard** (when `HERMES_DASHBOARD=1`) | `docker logs <container>` (no prefix) | `docker logs -f hermes` — interleaved with gateway lines |
| **Boot reconciler** (records which profile gateways were restored on each container start) | `${HERMES_HOME}/logs/container-boot.log` (append-only audit log) | `tail -F ~/.hermes/logs/container-boot.log` |
| **Generic Hermes logs** (`agent.log`, `errors.log`) | `${HERMES_HOME}/logs/` (profile-aware) | `docker exec hermes hermes logs --follow [--level WARNING] [--session <id>]` |

Duas consequências práticas que valem saber:

- A cópia em arquivo em `logs/gateways/<profile>/current` é o que sobrevive restarts de container. `docker logs` retém apenas saída do lifetime do container atual (e é apagado em `docker rm`); os arquivos rotacionados persistem no volume bind-mounted.
- O shape da linha de audit do boot reconciler é `<iso-timestamp> profile=<name> prior_state=<state> action=<registered|started>`, então um `grep profile=coder ~/.hermes/logs/container-boot.log` rápido revela quando um profile dado foi restaurado por último e se o s6 auto-iniciou.

## Environment variable forwarding {#environment-variable-forwarding}

API keys são lidas de `/opt/data/.env` dentro do container. Você também pode passar variáveis de ambiente diretamente:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

Flags `-e` diretas sobrescrevem valores de `.env`. Isso é útil para CI/CD ou integrações secrets-manager onde você não quer keys no disco.

:::note Looking for Docker as the **terminal backend**?
Esta página cobre executar o Hermes em si dentro do Docker. Se quiser que o Hermes execute as chamadas `terminal` / `execute_code` do agente dentro de um container sandbox Docker (um container long-lived compartilhado entre processos Hermes — veja issue #20561), esse é um bloco de config separado — `terminal.backend: docker` mais `terminal.docker_image`, `terminal.docker_volumes`, `terminal.docker_forward_env`, `terminal.docker_env`, `terminal.docker_run_as_host_user`, `terminal.docker_extra_args`, `terminal.docker_persist_across_processes` e `terminal.docker_orphan_reaper`. Veja [Configuration → Docker Backend](configuration.md#docker-backend) para o conjunto completo incluindo regras de lifecycle de container.
:::

## Docker Compose example {#docker-compose-example}

Para deploy persistente com gateway e dashboard, um `docker-compose.yaml` é conveniente:

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # gateway API
      - "9119:9119"   # dashboard (only reached when HERMES_DASHBOARD=1)
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # Uncomment to forward specific env vars instead of using .env file:
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

Inicie com `docker compose up -d` e veja logs com `docker compose logs -f`. stdout do gateway supervisionado também é tee'd para `${HERMES_HOME}/logs/gateways/<profile>/current` no volume — veja [Where the logs go](#where-the-logs-go) para o mapa completo de roteamento.

## Optional: Linux desktop audio bridge {#optional-linux-desktop-audio-bridge}

Voice mode no Docker precisa de duas coisas separadas para funcionar: o Hermes deve poder sondar dispositivos de áudio dentro do container, e o container deve alcançar o audio server do host. O setup abaixo cobre o plumbing de áudio do host para desktops Linux que expõem um socket compatível com PulseAudio, incluindo muitos setups PipeWire.

:::caution
Isso é um workaround de desktop Linux, não um feature geral do Docker Desktop. É útil quando você já tem áudio no host funcionando e quer CLI voice mode dentro do container Hermes. Se o Hermes ainda reportar `Running inside Docker container -- no audio devices`, use um build que inclua suporte Docker audio probing para `PULSE_SERVER` / `PIPEWIRE_REMOTE`.
:::

Primeiro, crie um config ALSA ao lado do seu Compose file:

```conf title="asound.conf"
pcm.!default {
    type pulse
    hint {
        show on
        description "Default ALSA Output (PulseAudio)"
    }
}

pcm.pulse {
    type pulse
}

ctl.!default {
    type pulse
}
```

Depois construa uma imagem derivada pequena com o plugin ALSA PulseAudio instalado:

```dockerfile title="Dockerfile.audio"
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*
```

Use essa imagem no Compose e passe o socket PulseAudio e cookie do usuário host:

```yaml
services:
  hermes:
    build:
      context: .
      dockerfile: Dockerfile.audio
    image: hermes-agent-audio
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    volumes:
      - ~/.hermes:/opt/data
      - /run/user/${HERMES_UID}/pulse:/run/user/${HERMES_UID}/pulse
      - ~/.config/pulse/cookie:/tmp/pulse-cookie:ro
      - ./asound.conf:/etc/asound.conf:ro
    environment:
      - HERMES_UID=${HERMES_UID}
      - HERMES_GID=${HERMES_GID}
      - XDG_RUNTIME_DIR=/run/user/${HERMES_UID}
      - PULSE_SERVER=unix:/run/user/${HERMES_UID}/pulse/native
      - PULSE_COOKIE=/tmp/pulse-cookie
```

Inicie com UID/GID do host para o processo do container acessar o socket de áudio per-user:

```sh
export HERMES_UID="$(id -u)"
export HERMES_GID="$(id -g)"
docker compose up -d --build
```

Para verificar o que PortAudio vê dentro do container:

```sh
docker exec hermes /opt/hermes/.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Resource limits {#resource-limits}

O container Hermes precisa de recursos moderados. Mínimos recomendados:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Memory | 1 GB | 2–4 GB |
| CPU | 1 core | 2 cores |
| Disk (data volume) | 500 MB | 2+ GB (grows with sessions/skills) |

Automação de browser (Playwright/Chromium) é o feature mais memory-hungry. Se não precisa de browser tools, 1 GB é suficiente. Com browser tools ativas, aloque pelo menos 2 GB.

Defina limites no Docker:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## What the Dockerfile does {#what-the-dockerfile-does}

A imagem oficial é baseada em `debian:13.4` e inclui:

- Python 3.13 com dependências synced do lockfile via `uv sync --frozen --no-install-project` para os extras baked (`all`, `messaging`, Anthropic/Bedrock/Azure identity, Hindsight, Matrix), seguido de install editable no-deps do Hermes em si.
- Node.js 22 + npm (para automação de browser, WhatsApp bridge, bundles TUI/Desktop e build tooling de workspace)
- Playwright com Chromium (`npx playwright install --with-deps chromium --only-shell`)
- ripgrep, ffmpeg, git e `xz-utils` como utilitários de sistema
- **`docker-cli`** — para agents executando dentro do container dirigirem o Docker daemon do host (bind-mount `/var/run/docker.sock` para opt-in) para `docker build`, `docker run`, inspeção de container, etc.
- **`openssh-client`** — habilita o [SSH terminal backend](/user-guide/configuration#ssh-backend) de dentro do container. O backend SSH faz shell out para o binário `ssh` do sistema; sem isso, falhava silenciosamente em installs containerizados.
- O WhatsApp bridge (`scripts/whatsapp-bridge/`)
- **[`s6-overlay`](https://github.com/just-containers/s6-overlay) v3** como PID 1 (substitui o `tini` mais antigo) — supervisiona dashboard e gateways por profile com auto-restart em crash, reaps subprocessos zombie e forward signals.

A imagem trata `/opt/hermes` como árvore de instalação imutável em runtime. Extras Python opcionais, workspaces Node e assets TUI que devem estar disponíveis dentro do Docker precisam ser baked durante o build da imagem; lazy installs runtime estão desabilitados para gateways supervisionados e comandos `docker exec hermes …` não tentarem escrever artefatos de dependência de volta na árvore source read-only.

O `ENTRYPOINT` do container é `/init` do s6-overlay. No boot ele:
1. Executa `/etc/cont-init.d/01-hermes-setup` (= `docker/stage2-hook.sh`) como root: remap opcional UID/GID, corrige ownership do volume, seed `.env` / `config.yaml` / `SOUL.md` no primeiro boot, executa migrações config-schema non-interactive a menos que `HERMES_SKIP_CONFIG_MIGRATION=1`, sync skills bundled.
2. Executa `/etc/cont-init.d/02-reconcile-profiles` (= `hermes_cli.container_boot`): percorre `$HERMES_HOME/profiles/<name>/`, recria o slot de serviço gateway s6 por profile sob `/run/service/gateway-<profile>/`, e auto-inicia apenas aqueles cujo último estado registrado era `running` (veja [Per-profile gateway supervision](#per-profile-gateway-supervision)).
3. Inicia os serviços s6-rc estáticos `main-hermes` e `dashboard`.
4. Exec o CMD do container como main program (`/opt/hermes/docker/main-wrapper.sh`), que roteia os argumentos que o usuário passou a `docker run`:
   - sem args → `hermes` (o default)
   - primeiro arg é executável no PATH (ex. `sleep`, `bash`) → exec direto
   - qualquer outra coisa → `hermes <args>` (subcommand passthrough)
   O container sai quando este main program sai, com seu exit code.

:::warning Breaking change vs. pre-s6 images
O ENTRYPOINT do container agora é `/init` (s6-overlay), não `/usr/bin/tini`. Todos os cinco padrões documentados de invocação `docker run` (sem args, `chat -q "…"`, `sleep infinity`, `bash`, `--tui`) se comportam identicamente à imagem baseada em tini. Se você tem um wrapper downstream que dependia de comportamento de signal específico do tini ou invocação hard-coded `/usr/bin/tini --`, fixe na tag de imagem anterior.
:::

:::warning Privilege model
Não sobrescreva o entrypoint da imagem a menos que mantenha `/init` (ou, equivalentemente, o shim legacy `docker/entrypoint.sh` que forward para o stage2 hook) na command chain. `/init` do s6-overlay executa como root para poder chown o volume no primeiro boot, depois drop para o usuário `hermes` via `s6-setuidgid` para todo serviço supervisionado E para o main program. Iniciar `hermes gateway run` como root dentro da imagem oficial é recusado por padrão porque pode deixar arquivos root-owned em `/opt/data` e quebrar starts posteriores de dashboard ou gateway. Defina `HERMES_ALLOW_ROOT_GATEWAY=1` apenas quando aceitar intencionalmente esse risco.
:::

### `docker exec` automatically drops to the `hermes` user {#docker-exec-automatically-drops-to-the-hermes-user}

`docker exec hermes <cmd>` default para executar como root dentro do container, mas a imagem shipa um shim fino em `/opt/hermes/bin/hermes` (primeiro no PATH) que detecta callers root e re-exec transparentemente via `s6-setuidgid hermes`. Então `docker exec hermes login`, `docker exec hermes profile create …`, `docker exec hermes setup`, etc. todos escrevem arquivos owned por UID 10000 — i.e. legíveis pelo gateway supervisionado — sem flag `--user` extra necessária. Callers non-root (os próprios processos supervisionados, `docker exec --user hermes`, subagents kanban dentro do container) acertam um short-circuit que exec o binário venv diretamente, então não há overhead nos hot paths.

Se precisar especificamente de um `docker exec` que retém semântica root (sessões de diagnóstico, inspecionar estado root-only, arquivos fora de `/opt/data` que root acontece de possuir), opt out por invocação:

```sh
docker exec -e HERMES_DOCKER_EXEC_AS_ROOT=1 hermes <cmd>
```

O shim aceita `1` / `true` / `yes` (case-insensitive). Qualquer outra coisa — incluindo typos como `=0` — cai no drop, então opt-outs silenciosos não são possíveis. Se `s6-setuidgid` não estiver disponível (builds custom que removeram s6-overlay), o shim recusa executar como root e sai 126, surfacing o privilege model quebrado alto em vez de regredir para o footgun histórico onde `docker exec hermes login` escreveria `auth.json` como `root:root` e quebraria auth do gateway supervisionado em toda mensagem de chat platform.

### Per-profile gateway supervision {#per-profile-gateway-supervision}

Cada profile criado com `hermes profile create <name>` recebe automaticamente um serviço gateway supervisionado por s6 registrado em `/run/service/gateway-<name>/`, com auto-restart persistente de estado entre restarts de container. Veja [Multi-profile support](#multi-profile-support) acima para o workflow user-facing e os comandos de lifecycle.

**Benefícios de supervisão sobre a imagem pre-s6:**

- Crashes de gateway são auto-reiniciados por `s6-supervise` após backoff ~1s.
- Dashboard, quando habilitado com `HERMES_DASHBOARD=1`, é supervisionado na mesma árvore de supervisão e recebe o mesmo tratamento auto-restart.
- `docker restart`, upgrades de imagem (`docker compose up -d --force-recreate`) e exits inesperados preservam gateways running: o reconciler cont-init lê `$HERMES_HOME/profiles/<name>/gateway_state.json` e sobe o slot de volta se o último estado registrado era `running`. Apenas um `hermes gateway stop` explícito registra `stopped` e mantém o gateway down através do restart; SIGTERM container/s6 enviado em restart ou upgrade é tratado como "still running" e auto-inicia.
- Logs de gateway por profile persistem sob `$HERMES_HOME/logs/gateways/<profile>/current` (rotacionados por `s6-log`), e ações do reconciler são appended a `$HERMES_HOME/logs/container-boot.log` por boot. Veja [Where the logs go](#where-the-logs-go) para o mapa completo de roteamento.

`hermes status` dentro do container reporta `Manager: s6 (container supervisor)`. Use `/command/s6-svstat /run/service/gateway-<name>` para a view raw do supervisor (note que `/command/` está no PATH apenas para processos da árvore de supervisão; passe o path absoluto ao chamar de `docker exec`).

## Upgrading {#upgrading}

Puxe a imagem mais recente e recrie o container. Seu diretório de dados é
preservado, e o container executa migrações config-schema non-interactive
contra o `$HERMES_HOME/config.yaml` montado antes de iniciar o gateway.
Quando uma migração é necessária, o Hermes escreve backups timestamped ao lado de
`config.yaml` e `.env` primeiro.

```sh
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

Ou com Docker Compose:

```sh
docker compose pull
docker compose up -d
```

Defina `HERMES_SKIP_CONFIG_MIGRATION=1` apenas se precisar inspecionar ou migrar a
config persistida manualmente antes de deixar a nova imagem reescrevê-la.

## Skills and credential files {#skills-and-credential-files}

Ao usar Docker como ambiente de execução (não os métodos acima, mas quando o agente executa comandos dentro de um sandbox Docker — veja [Configuration → Docker Backend](./configuration.md#docker-backend)), o Hermes reutiliza um único container long-lived para todas as tool calls e automaticamente bind-mounts o diretório de skills (`~/.hermes/skills/`) e quaisquer credential files declaradas por skills naquele container como volumes read-only. Scripts, templates e references de skill estão disponíveis dentro do sandbox sem configuração manual, e como o container persiste pela vida do processo Hermes, quaisquer dependências que você instala ou arquivos que escreve permanecem para a próxima tool call.

O mesmo syncing acontece para backends SSH e Modal — skills e credential files são uploaded via rsync ou Modal mount API antes de cada comando.

## Installing more tools in the container {#installing-more-tools-in-the-container}

A imagem oficial shipa com um conjunto curado de utilitários (veja [What the Dockerfile does](#what-the-dockerfile-does)), mas nem toda tool que um agente pode querer vem pré-instalada. Há cinco abordagens recomendadas, em ordem crescente de esforço e durabilidade.

### npm or Python tools — use `npx` or `uvx` {#npm-or-python-tools-use-npx-or-uvx}

Para qualquer tool publicada no npm ou PyPI, instrua o Hermes a executá-la via `npx` (npm) ou `uvx` (Python) e lembrar esse comando em sua memória persistente. Se a tool precisa de config file ou credenciais, instrua a colocá-las sob `/opt/data` (ex. `/opt/data/<tool>/config.yaml`).

Dependências são fetched on demand e cached pela vida do container. Configuração escrita sob `/opt/data` sobrevive restarts de container porque vive no diretório bind-mounted do host. O cache de pacotes em si é reconstruído após `docker rm`, mas `npx` e `uvx` re-fetch transparentemente na próxima vez que a tool rodar.

### Other tools (apt packages, binaries) — install and remember {#other-tools-apt-packages-binaries-install-and-remember}

Para qualquer coisa fora npm ou PyPI — pacotes `apt`, binários prebuilt, runtimes de linguagem não já na imagem — instrua o Hermes como instalá-la (ex. `apt-get update && apt-get install -y <package>`) e diga para lembrar o comando de install. A tool persiste pelo resto da vida do container, e o Hermes re-executará o comando de install após restart de container quando precisar da tool de novo.

Isso encaixa bem para tools rápidas de instalar e usadas ocasionalmente. Para tools usadas constantemente, prefira a próxima abordagem.

### Durable installs — build a derived image {#durable-installs-build-a-derived-image}

Quando uma tool deve estar disponível imediatamente a cada start de container sem delay de re-install, construa uma nova imagem que herda de `nousresearch/hermes-agent` e instala a tool em uma layer:

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends <your-package> \
    && rm -rf /var/lib/apt/lists/*
USER hermes
```

Build e use no lugar da imagem oficial:

```sh
docker build -t my-hermes:latest .
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  my-hermes:latest gateway run
```

O entrypoint script e semântica `/opt/data` são herdados inalterados, então o resto desta página ainda se aplica. Lembre de rebuild a imagem ao puxar um upstream `nousresearch/hermes-agent` mais novo.

### Complex tools or multi-service stacks — run a sidecar container {#complex-tools-or-multi-service-stacks-run-a-sidecar-container}

Para tools que trazem seu próprio serviço (database, web server, queue, headless browser farm) ou são pesadas demais para viver dentro do container Hermes, execute-as como container separado em uma Docker network compartilhada. O Hermes alcança o sidecar por container name, da mesma forma que alcança um inference server local (veja [Connecting to local inference servers](#connecting-to-local-inference-servers-vllm-ollama-etc)).

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

  my-tool:
    image: example/my-tool:latest
    container_name: my-tool
    restart: unless-stopped
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

De dentro do container Hermes, o sidecar é alcançável em `http://my-tool:<port>` (ou qualquer protocolo que sirva). Este padrão mantém lifecycle, resource limits e cadência de upgrade de cada serviço independentes, e evita inflar a imagem Hermes com dependências necessárias só por uma tool.

### Broadly useful tools — open an issue or pull request {#broadly-useful-tools-open-an-issue-or-pull-request}

Se uma tool provavelmente será útil para a maioria dos usuários Hermes Agent, considere contribuí-la upstream em vez de carregá-la numa imagem derivada privada. Abra uma issue ou pull request no [repositório hermes-agent](https://github.com/NousResearch/hermes-agent) descrevendo a tool e seu caso de uso. Tools bundled na imagem oficial beneficiam todo usuário e evitam overhead de manutenção de um fork downstream.

## Connecting to local inference servers (vLLM, Ollama, etc.) {#connecting-to-local-inference-servers-vllm-ollama-etc}

Ao executar Hermes no Docker e seu inference server (vLLM, Ollama, text-generation-inference, etc.) também roda no host ou em outro container, networking exige atenção extra.

### Docker Compose (recommended) {#docker-compose-recommended}

Coloque ambos serviços na mesma Docker network. Esta é a abordagem mais confiável:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --served-model-name my-model
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    networks:
      - hermes-net
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

Depois em seu `~/.hermes/config.yaml`, use o **container name** como hostname:

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip Key points
- Use o **container name** (`vllm`) como hostname — não `localhost` ou `127.0.0.1`, que referem ao próprio container Hermes.
- O valor `model` deve corresponder ao `--served-model-name` que você passou ao vLLM.
- Defina `api_key` para qualquer string não vazia (vLLM exige o header mas não valida por padrão).
- **Não** inclua trailing slash em `base_url`.
:::

### Standalone Docker run (no Compose) {#standalone-docker-run-no-compose}

Se seu inference server roda diretamente no host (não no Docker), use `host.docker.internal` no macOS/Windows, ou `--network host` no Linux:

**macOS / Windows:**

```sh
docker run -d \
  --name hermes \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://host.docker.internal:8000/v1
  api_key: "none"
```

**Linux (host networking):**

```sh
docker run -d \
  --name hermes \
  --network host \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  api_key: "none"
```

:::warning Com `--network host`, a flag `-p` é ignorada — todas as portas do container são expostas diretamente no host.
:::

### Verifying connectivity {#verifying-connectivity}

De dentro do container Hermes, confirme que o inference server é alcançável:

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

Você deve ver uma resposta JSON listando seu served model. Se falhar, verifique:

1. Ambos containers estão na mesma Docker network (`docker network inspect hermes-net`)
2. O inference server está listening em `0.0.0.0`, não `127.0.0.1`
3. O número da porta corresponde

### Ollama {#ollama}

Ollama funciona da mesma forma. Se Ollama roda no host, use `host.docker.internal:11434` (macOS/Windows) ou `127.0.0.1:11434` (Linux com `--network host`). Se Ollama roda em seu próprio container na mesma Docker network:

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## Troubleshooting {#troubleshooting}

### Container exits immediately {#container-exits-immediately}

Verifique logs: `docker logs hermes`. Causas comuns:
- Arquivo `.env` ausente ou inválido — execute interativamente primeiro para completar setup
- Conflitos de porta se executando com portas expostas

### "Permission denied" errors {#permission-denied-errors}

O stage2 hook do container drop privilégios para o usuário non-root `hermes` (UID 10000) via `s6-setuidgid` dentro de cada serviço supervisionado. Se seu `~/.hermes/` no host é owned por UID diferente, defina `HERMES_UID`/`HERMES_GID` — ou aliases `PUID`/`PGID`, para paridade com LinuxServer.io e imagens NAS — para corresponder ao seu usuário host, ou garanta que o diretório de dados seja writable:

```sh
chmod -R 755 ~/.hermes
```

Em NAS (UGOS, Synology, unRAID) o diretório de dados é tipicamente um **bind mount** owned por UID host que o container não pode `chown`. Defina `PUID`/`PGID` (ou `HERMES_UID`/`HERMES_GID`) para aquele usuário host para o runtime executar como owner do mount em vez de UID 10000:

```sh
docker run -d \
  --name hermes \
  -e PUID=1000 -e PGID=10 \
  -v /volume1/docker/hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

`docker exec hermes <cmd>` também drop automaticamente para UID 10000 — veja [`docker exec` automatically drops to the `hermes` user](#docker-exec-automatically-drops-to-the-hermes-user) para detalhes e opt-out por invocação.

### Browser tools not working {#browser-tools-not-working}

Playwright precisa de shared memory. Adicione `--shm-size=1g` ao seu comando Docker run:

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### Gateway not reconnecting after network issues {#gateway-not-reconnecting-after-network-issues}

A flag `--restart unless-stopped` lida com a maioria das falhas transitórias. Se o gateway estiver preso, reinicie o container:

```sh
docker restart hermes
```

### Checking container health {#checking-container-health}

```sh
docker logs --tail 50 hermes          # Recent logs
docker run -it --rm nousresearch/hermes-agent:latest version     # Verify version
docker stats hermes                    # Resource usage
```
