---
sidebar_position: 1
title: "Ferramentas e toolsets"
description: "Visão geral das ferramentas do Hermes Agent — o que está disponível, como os toolsets funcionam e backends de terminal"
---

# Ferramentas e toolsets

Ferramentas são funções que estendem as capacidades do agente. Elas são organizadas em **toolsets** lógicos que podem ser habilitados ou desabilitados por plataforma.

## Ferramentas disponíveis {#available-tools}

O Hermes inclui um amplo registro built-in de ferramentas cobrindo web search, automação de navegador, execução de terminal, edição de arquivos, memória, delegação, tarefas agendadas, Home Assistant e muito mais.

:::note
**Memória cross-session do Honcho** está disponível como plugin de provedor de memória (`plugins/memory/honcho/`), não como toolset built-in. Veja [Plugins](./plugins.md) para instalação.
:::

Categorias de alto nível:

| Categoria | Exemplos | Descrição |
|----------|----------|-------------|
| **Web** | `web_search`, `web_extract` | Pesquise na web e extraia conteúdo de páginas. |
| **X Search** | `x_search` | Pesquise posts e threads do X (Twitter) via ferramenta `x_search` built-in do xAI Responses — condicionado a credenciais xAI (SuperGrok OAuth ou `XAI_API_KEY`); desligado por padrão, opt-in via `hermes tools` → 🐦 X (Twitter) Search. |
| **Terminal e arquivos** | `terminal`, `process`, `read_file`, `patch` | Execute comandos e manipule arquivos. |
| **Navegador** | `browser_navigate`, `browser_snapshot`, `browser_vision` | Automação interativa de navegador com suporte a texto e visão. |
| **Mídia** | `vision_analyze`, `image_generate`, `text_to_speech` | Análise e geração multimodal. |
| **Orquestração do agente** | `todo`, `clarify`, `execute_code`, `delegate_task` | Planejamento, esclarecimento, execução de código e delegação de subagentes. |
| **Memória e recall** | `memory`, `session_search` | Memória persistente e busca de sessões. |
| **Automação** | `cronjob` | Tarefas agendadas com ações create/list/update/pause/resume/run/remove. Entrega outbound é tratada pela própria entrega do cron, pelo CLI `hermes send` e pelo notificador do gateway — não por uma ferramenta invocável pelo agente. |
| **Integrações** | `ha_*`, ferramentas de servidores MCP | Home Assistant, MCP e outras integrações. |

Para o registro autoritativo derivado do código, veja [Referência de ferramentas built-in](/reference/tools-reference) e [Referência de toolsets](/reference/toolsets-reference).

:::tip Nous Tool Gateway
Assinantes pagos do [Nous Portal](https://portal.nousresearch.com) podem usar web search, geração de imagem, TTS e automação de navegador pelo **[Tool Gateway](tool-gateway.md)** — sem chaves de API separadas. Execute `hermes model` para habilitar, ou configure ferramentas individuais com `hermes tools`.
:::

## Usando toolsets {#using-toolsets}

```bash
# Use specific toolsets
hermes chat --toolsets "web,terminal"

# See all available tools
hermes tools

# Configure tools per platform (interactive)
hermes tools
```

Toolsets comuns incluem `web`, `search`, `terminal`, `file`, `browser`, `vision`, `image_gen`, `skills`, `tts`, `todo`, `memory`, `session_search`, `cronjob`, `code_execution`, `delegation`, `clarify`, `homeassistant`, `messaging`, `spotify`, `discord`, `discord_admin`, `debugging` e `safe`.

Veja [Referência de toolsets](/reference/toolsets-reference) para o conjunto completo, incluindo presets de plataforma como `hermes-cli`, `hermes-telegram` e toolsets MCP dinâmicos como `mcp-<server>`.

## Backends de terminal {#terminal-backends}

A ferramenta terminal pode executar comandos em ambientes diferentes:

| Backend | Descrição | Caso de uso |
|---------|-------------|----------|
| `local` | Roda na sua máquina (padrão) | Desenvolvimento, tarefas confiáveis |
| `docker` | Containers isolados | Segurança, reprodutibilidade |
| `ssh` | Servidor remoto | Sandboxing, manter o agente longe do próprio código |
| `singularity` | Containers HPC | Computação em cluster, rootless |
| `modal` | Execução na nuvem | Serverless, escala |
| `daytona` | Workspace sandbox na nuvem | Ambientes remotos de dev persistentes |

### Configuração {#configuration}

```yaml
# In ~/.hermes/config.yaml
terminal:
  backend: local    # or: docker, ssh, singularity, modal, daytona
  cwd: "."          # Working directory
  timeout: 180      # Command timeout in seconds
```

### Backend Docker {#docker-backend}

```yaml
terminal:
  backend: docker
  docker_image: python:3.11-slim
```

**Um container persistente, compartilhado por todo o processo.** O Hermes inicia um único container de longa duração no primeiro uso (`docker run -d ... sleep 2h`) e roteia cada terminal, arquivo e chamada `execute_code` via `docker exec` para esse mesmo container. Mudanças de diretório de trabalho, pacotes instalados, ajustes de ambiente e arquivos escritos em `/workspace` persistem de uma chamada de ferramenta para a próxima, através de `/new`, `/reset` e subagentes `delegate_task`, durante a vida do processo Hermes. O container é parado e removido no shutdown.

Isso significa que o backend Docker se comporta como uma VM sandbox persistente, não um container novo por comando. Se você fizer `pip install foo` uma vez, ele fica lá pelo resto da sessão. Se você fizer `cd /workspace/project`, chamadas subsequentes de `ls` veem esse diretório. Veja [Configuração → Backend Docker](../configuration.md#docker-backend) para detalhes completos do ciclo de vida e a flag `container_persistent` que controla se `/workspace` e `/root` sobrevivem entre reinícios do Hermes.

### Backend SSH {#ssh-backend}

Recomendado para segurança — o agente não pode modificar o próprio código:

```yaml
terminal:
  backend: ssh
```
```bash
# Set credentials in ~/.hermes/.env
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=myuser
TERMINAL_SSH_KEY=~/.ssh/id_rsa
```

### Singularity/Apptainer {#singularityapptainer}

```bash
# Pre-build SIF for parallel workers
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

### Recursos de container {#container-resources}

Configure CPU, memória, disco e persistência para todos os backends de container:

```yaml
terminal:
  backend: docker  # or singularity, modal, daytona
  container_cpu: 1              # CPU cores (default: 1)
  container_memory: 5120        # Memory in MB (default: 5GB)
  container_disk: 51200         # Disk in MB (default: 50GB)
  container_persistent: true    # Persist filesystem across sessions (default: true)
```

Quando `container_persistent: true`, pacotes instalados, arquivos e config sobrevivem entre sessões.

### Segurança de container {#container-security}

Todos os backends de container rodam com hardening de segurança:

- Root filesystem somente leitura (Docker)
- Todas as capabilities Linux removidas
- Sem escalação de privilégio
- Limites de PID (256 processos)
- Isolamento completo de namespace
- Workspace persistente via volumes, não camada root gravável

Docker pode receber opcionalmente uma allowlist explícita de env via `terminal.docker_forward_env`, mas variáveis encaminhadas são visíveis a comandos dentro do container e devem ser tratadas como expostas àquela sessão.

## Gerenciamento de processos em background {#background-process-management}

Inicie processos em background e gerencie-os:

```python
terminal(command="pytest -v tests/", background=true)
# Returns: {"session_id": "proc_abc123", "pid": 12345}

# Then manage with the process tool:
process(action="list")       # Show all running processes
process(action="poll", session_id="proc_abc123")   # Check status
process(action="wait", session_id="proc_abc123")   # Block until done
process(action="log", session_id="proc_abc123")    # Full output
process(action="kill", session_id="proc_abc123")   # Terminate
process(action="write", session_id="proc_abc123", data="y")  # Send input
```

Modo PTY (`pty=true`) habilita ferramentas CLI interativas como Codex e Claude Code.

## Suporte a sudo {#sudo-support}

Se um comando precisar de sudo, você será solicitado a informar sua senha (cacheada para a sessão). Ou defina `SUDO_PASSWORD` em `~/.hermes/.env`.

:::warning
Em plataformas de mensagens, se o sudo falhar, a saída inclui uma dica para adicionar `SUDO_PASSWORD` em `~/.hermes/.env`.
:::
