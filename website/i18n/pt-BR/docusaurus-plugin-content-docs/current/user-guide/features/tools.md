---
sidebar_position: 1
title: "Ferramentas e toolsets"
description: "Visão geral das ferramentas do Hermes Agent — o que está disponível, como os toolsets funcionam e backends de terminal"
---

# Ferramentas e toolsets

Ferramentas são funções que estendem as capacidades do agente. Elas são organizadas em **toolsets** lógicos que podem ser habilitados ou desabilitados por plataforma.

## Ferramentas disponíveis {#available-tools}

O Hermes vem com um amplo registro de ferramentas built-in cobrindo busca na web, automação de browser, execução de terminal, edição de arquivos, memória, delegação, tarefas agendadas, Home Assistant e muito mais.

:::note
A **memória cross-session Honcho** está disponível como plugin de memory provider (`plugins/memory/honcho/`), não como toolset built-in. Veja [Plugins](./plugins.md) para instalação.
:::

Categorias de alto nível:

| Categoria | Exemplos | Descrição |
|----------|----------|-------------|
| **Web** | `web_search`, `web_extract` | Buscar na web e extrair conteúdo de páginas. |
| **X Search** | `x_search` | Buscar posts e threads no X (Twitter) via ferramenta Responses `x_search` built-in da xAI — condicionada a credenciais xAI (SuperGrok OAuth ou `XAI_API_KEY`); desligada por padrão, opt-in via `hermes tools` → 🐦 X (Twitter) Search. |
| **Terminal e arquivos** | `terminal`, `process`, `read_file`, `patch` | Executar comandos e manipular arquivos. |
| **Browser** | `browser_navigate`, `browser_snapshot`, `browser_vision` | Automação interativa de browser com suporte a texto e visão. |
| **Mídia** | `vision_analyze`, `image_generate`, `text_to_speech` | Análise e geração multimodal. |
| **Orquestração do agente** | `todo`, `clarify`, `execute_code`, `delegate_task` | Planejamento, esclarecimento, execução de código e delegação a subagentes. |
| **Memória e recall** | `memory`, `session_search` | Memória persistente e busca de sessões. |
| **Automação** | `cronjob` | Tarefas agendadas com ações create/list/update/pause/resume/run/remove. A entrega outbound é feita pela própria entrega do cron, pelo CLI `hermes send` e pelo notificador do gateway — não por uma ferramenta invocável pelo agente. |
| **Integrações** | `ha_*`, ferramentas de servidores MCP | Home Assistant, MCP e outras integrações. |

Para o registro autoritativo derivado do código, veja [Referência de ferramentas built-in](/reference/tools-reference) e [Referência de toolsets](/reference/toolsets-reference).

:::tip Nous Tool Gateway
Assinantes pagos do [Nous Portal](https://portal.nousresearch.com) podem usar busca na web, geração de imagem, TTS e automação de browser pelo **[Tool Gateway](tool-gateway.md)** — sem chaves de API separadas. Execute `hermes model` para habilitar, ou configure ferramentas individuais com `hermes tools`.
:::

## Usando toolsets {#using-toolsets}

```bash
# Use toolsets específicos
hermes chat --toolsets "web,terminal"

# Veja todas as ferramentas disponíveis
hermes tools

# Configure ferramentas por plataforma (interativo)
hermes tools
```

Toolsets comuns incluem `web`, `search`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, `homeassistant`, `messaging`, `spotify`, `discord`, `discord_admin`, `debugging` e `safe`.

Veja [Referência de toolsets](/reference/toolsets-reference) para o conjunto completo, incluindo presets por plataforma como `hermes-cli`, `hermes-telegram` e toolsets MCP dinâmicos como `mcp-<server>`.

## Anotações de resultado de ferramenta {#tool-result-annotations}

Alguns comportamentos de ferramenta vale conhecer quando você lê transcripts do agente:

- **Mortes por signal são explicadas.** Quando um comando de terminal é morto por um signal, o resultado carrega uma nota legível em vez de um código numérico nu — ex.: exit `-9`/`137` vira "terminated by signal 9: SIGKILL — often the kernel OOM killer on memory exhaustion, or an explicit kill -9", e segfaults, aborts, SIGTERM, broken pipes e limites de CPU/tamanho de arquivo são rotulados da mesma forma. Códigos negativos (semântica de subprocess) são afirmados de forma definitiva; a convenção `128+signum` do shell é hedged com "usually" porque uma aplicação pode sair legitimamente com esses códigos.
- **Arquivos de texto UTF-16 são transcoded, não recusados.** `read_file` detecta UTF-16 (BOM ou heurística de padrão de bytes, qualquer endianness — comum em arquivos do Notepad do Windows e redirects `>` do PowerShell) e transcodifica para UTF-8 para display em vez de marcar o arquivo como binário. O resultado inclui uma dica divulgando a conversão; edições via `patch`/`write_file` re-encodam como UTF-8. Arquivos acima de 10 MB e arquivos genuinamente binários ainda recebem a recusa de arquivo binário.

## Backends de terminal {#terminal-backends}

A ferramenta de terminal pode executar comandos em ambientes diferentes:

| Backend | Descrição | Caso de uso |
|---------|-------------|----------|
| `local` | Executa na sua máquina (padrão) | Desenvolvimento, tarefas confiáveis |
| `docker` | Containers isolados | Segurança, reprodutibilidade |
| `ssh` | Servidor remoto | Sandbox, manter o agente longe do próprio código |
| `singularity` | Containers HPC | Computação em cluster, sem root |
| `modal` | Execução na nuvem | Serverless, escala |
| `daytona` | Workspace sandbox na nuvem | Ambientes remotos de dev persistentes |
| `vercel_sandbox` | microVM na nuvem Vercel Sandbox | Execução na nuvem com persistência de filesystem via snapshots |

### Configuração {#configuration}

```yaml
# Em ~/.hermes/config.yaml
terminal:
  backend: local    # ou: docker, ssh, singularity, modal, daytona, vercel_sandbox
  cwd: "."          # Diretório de trabalho
  timeout: 180      # Timeout de comando em segundos
```

### Arquivos de startup do shell e comandos não interativos {#shell-startup-files-and-non-interactive-commands}

Chamadas de terminal do agente rodam seu shell de forma **não interativa** — não há TTY nem humano no prompt. Inicialização pesada ou interativa do shell que você nunca percebe em um terminal normal pode quebrar ou atrasar muito todo comando que o agente executa:

- **Init lento (`nvm`, gerenciadores de versão, prompts que tocam na rede):** o sourcing clássico de `nvm.sh` adiciona latência perceptível a *cada* start de shell, e o agente inicia muitos shells. rc files de vários segundos transformam um `git status` rápido em risco de timeout.
- **Blocos que esperam TTY:** qualquer coisa em `.bashrc`/`.zshrc` que faz prompt, roda attach de `tmux`/`screen`, chama `read` ou imprime um menu trava um shell não interativo — o comando parece rodar para sempre e depois estoura o timeout.
- **Saída incondicional:** rc files que fazem `echo` de banners poluem a saída de todo comando que o agente precisa parsear.

A correção é o guard padrão que a maioria das distros já coloca no topo de `.bashrc` — retornar cedo quando o shell é não interativo, e manter qualquer coisa pesada ou interativa abaixo dele:

```bash
# ~/.bashrc — mantenha este guard perto do topo
case $- in
  *i*) ;;      # interativo: continua
  *) return;;  # não interativo: para aqui
esac

# init pesado/interativo vai ABAIXO do guard
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

Usuários de Zsh: coloque setup só de login em `.zprofile` e setup só interativo em `.zshrc`; mantenha `.zshenv` mínimo, pois ele roda para todo shell, inclusive não interativos. Se o agente realmente precisa de uma ferramenta que só seu rc file coloca no `PATH`, exporte a mudança de `PATH` *acima* do guard (exports de path são baratos) ou faça symlink do binário em `~/.local/bin`.

Se comandos de terminal do agente travam ou estouram timeout logo após funcionarem no seu próprio terminal, a init do shell é o primeiro suspeito.

### Backend Docker {#docker-backend}

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**Um container persistente, compartilhado por todo o processo.** O Hermes inicia um único container de longa duração no primeiro uso (`docker run -d ... sleep infinity`) e roteia toda chamada de terminal, arquivo e `execute_code` via `docker exec` nesse mesmo container. Mudanças de diretório de trabalho, pacotes instalados, ajustes de ambiente e arquivos escritos em `/workspace` persistem de uma chamada de ferramenta para a próxima, através de `/new`, `/reset` e subagentes de `delegate_task`, pelo tempo de vida do processo Hermes. O container é parado e removido no shutdown.

Isso significa que o backend Docker se comporta como uma VM sandbox persistente, não um container novo por comando. Se você fizer `pip install foo` uma vez, fica lá pelo resto da sessão. Se fizer `cd /workspace/project`, chamadas subsequentes de `ls` veem esse diretório. Veja [Configuração → Backend Docker](../configuration.md#docker-backend) para detalhes completos do ciclo de vida e a flag `container_persistent` que controla se `/workspace` e `/root` sobrevivem entre restarts do Hermes.

### Backend SSH {#ssh-backend}

Recomendado para segurança — o agente não pode modificar o próprio código:

```yaml
terminal:
  backend: ssh
```
```bash
# Defina credenciais em ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer {#singularityapptainer}

```bash
# Pré-construa SIF para workers paralelos
apptainer build ~/python.sif docker://python:3.11-slim

# Configure
hermes config set terminal.backend singularity
hermes config set terminal.singularity_image ~/python.sif
```

### Modal (nuvem serverless) {#modal-serverless-cloud}

```bash
uv pip install modal
modal setup
hermes config set terminal.backend modal
```

### Vercel Sandbox {#vercel-sandbox}

```bash
pip install 'hermes-agent[vercel]'
hermes config set terminal.backend vercel_sandbox
hermes config set terminal.vercel_runtime node24
```

Autentique com os três: `VERCEL_TOKEN`, `VERCEL_PROJECT_ID` e `VERCEL_TEAM_ID`. Esse setup com access token é o caminho suportado para deploys e processos Hermes de longa duração em Render, Railway, Docker e hosts similares. Runtimes suportados são `node24`, `node22` e `python3.13`; o Hermes usa `/vercel/sandbox` como raiz do workspace remoto por padrão.

Para desenvolvimento local pontual, o Hermes também aceita tokens OIDC Vercel de curta duração:

```bash
VERCEL_OIDC_TOKEN="$(vc project token <project-name>)" hermes chat
```

A partir de um diretório de projeto Vercel linkado:

```bash
VERCEL_OIDC_TOKEN="$(vc project token)" hermes chat
```

Com `container_persistent: true`, o Hermes usa snapshots Vercel para preservar o estado do filesystem entre recriações de sandbox para a mesma tarefa. Isso pode incluir credenciais sincronizadas pelo Hermes, skills e arquivos de cache dentro do sandbox. Snapshots não preservam processos vivos, espaço de PID ou a mesma identidade de sandbox ativa.

Comandos de terminal em background usam o fluxo genérico de processo não local do Hermes: spawn, poll, wait, log e kill funcionam pela ferramenta de processo normal enquanto o sandbox está vivo, mas o Hermes não oferece recuperação nativa de processo detached Vercel após cleanup ou restart.

Deixe `container_disk` unset ou no default compartilhado `51200`; dimensionamento customizado de disco não é suportado para Vercel Sandbox e falhará em diagnósticos/criação de backend.

### Recursos de container {#container-resources}

Configure CPU, memória, disco e persistência para todos os backends de container:

```yaml
terminal:
  backend: docker  # ou singularity, modal, daytona, vercel_sandbox
  container_cpu: 1              # núcleos de CPU (padrão: 1)
  container_memory: 5120        # Memória em MB (padrão: 5GB)
  container_disk: 51200         # Disco em MB (padrão: 50GB)
  container_persistent: true    # Persiste filesystem entre sessões (padrão: true)
```

Quando `container_persistent: true`, pacotes instalados, arquivos e config sobrevivem entre sessões.

### Segurança de container {#container-security}

Todos os backends de container rodam com hardening de segurança:

- Filesystem root somente leitura (Docker)
- Todas as capabilities Linux removidas
- Sem escalação de privilégio
- Limites de PID (256 processos)
- Isolamento completo de namespace
- Workspace persistente via volumes, não camada root gravável

Docker pode receber opcionalmente uma allowlist explícita de env via `terminal.docker_forward_env`, mas variáveis encaminhadas ficam visíveis a comandos dentro do container e devem ser tratadas como expostas àquela sessão.

## Gerenciamento de processos em background {#background-process-management}

Inicie processos em background e gerencie-os:

```python
terminal(command="pytest -v tests/", background=true)
# Retorna: {"session_id": "proc_abc123", "pid": 12345}

# Depois gerencie com a ferramenta process:
process(action="list")       # Mostra todos os processos em execução
process(action="poll", session_id="proc_abc123")   # Verifica status
process(action="wait", session_id="proc_abc123")   # Bloqueia até terminar
process(action="log", session_id="proc_abc123")    # Saída completa
process(action="kill", session_id="proc_abc123")   # Encerra
process(action="write", session_id="proc_abc123", data="y")  # Envia input
```

Modo PTY (`pty=true`) habilita ferramentas CLI interativas como Codex e Claude Code.

## Suporte a sudo {#sudo-support}

Se um comando precisar de sudo, você será solicitado pela senha (cacheada para a sessão). Ou defina `SUDO_PASSWORD` em `~/.hermes/.env`.

:::warning
Em plataformas de mensagens, se o sudo falhar, a saída inclui uma dica para adicionar `SUDO_PASSWORD` em `~/.hermes/.env`.
:::
