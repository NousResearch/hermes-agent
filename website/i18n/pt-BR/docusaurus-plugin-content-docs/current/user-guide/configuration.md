---
sidebar_position: 2
title: "Configuração"
description: "Configure o Hermes Agent — config.yaml, providers, modelos, chaves de API e muito mais"
---

# Configuração

Todas as configurações são armazenadas no diretório `~/.hermes/` para fácil acesso.

:::tip Caminho mais fácil para um `config.yaml` funcional
Execute `hermes setup --portal` — um único OAuth fornece um modelo de provider e todas as quatro ferramentas do Tool Gateway sem editar YAML manualmente. Assinantes do Portal também ganham 10% de desconto em providers por token. Veja [Nous Portal](/integrations/nous-portal).
:::

## Estrutura de Diretórios

```text
~/.hermes/
├── config.yaml     # Configurações (model, terminal, TTS, compression, etc.)
├── .env            # Chaves de API e segredos
├── auth.json       # Credenciais OAuth do provider (Nous Portal, etc.)
├── SOUL.md         # Identidade primária do agente (slot #1 no system prompt)
├── memories/       # Memória persistente (MEMORY.md, USER.md)
├── skills/         # Skills criadas pelo agente (gerenciadas via skill_manage tool)
├── cron/           # Tarefas agendadas
├── sessions/       # Sessões do gateway
└── logs/           # Logs (errors.log, gateway.log — segredos auto-redactados)
```

## Gerenciando a Configuração

```bash
hermes config              # Visualizar configuração atual
hermes config edit         # Abrir config.yaml no editor
hermes config get KEY      # Imprimir um valor resolvido
hermes config set KEY VAL  # Definir um valor específico
hermes config unset KEY    # Remover um valor definido pelo usuário
hermes config check        # Verificar opções ausentes (após atualizações)
hermes config migrate      # Adicionar interativamente opções ausentes

# Exemplos:
hermes config get model
hermes config set model anthropic/claude-opus-4
hermes config set terminal.backend docker
hermes config unset terminal.backend
hermes config set OPENROUTER_API_KEY sk-or-...  # Salva em .env
```

:::tip
O comando `hermes config set` roteia automaticamente os valores para o arquivo correto — chaves de API são salvas em `.env`, todo o resto em `config.yaml`.
:::

## Precedência de Configuração

As configurações são resolvidas nesta ordem (maior prioridade primeiro):

1. **Argumentos do CLI** — ex.: `hermes chat --model anthropic/claude-sonnet-4` (substituição por invocação)
2. **`~/.hermes/config.yaml`** — o arquivo de configuração principal para todas as configurações não secretas
3. **`~/.hermes/.env`** — fallback para variáveis de ambiente; **obrigatório** para segredos (chaves de API, tokens, senhas)
4. **Padrões internos** — padrões seguros codificados quando nada mais está definido

:::info Regra Prática
Segredos (chaves de API, tokens de bot, senhas) vão em `.env`. Todo o resto (model, terminal backend, configurações de compressão, limites de memória, toolsets) vai em `config.yaml`. Quando ambos estão definidos, `config.yaml` vence para configurações não secretas.
:::

:::tip Implantações organizacionais
Um administrador pode fixar configurações específicas e valores secretos que um usuário padrão
não pode substituir, através de um diretório gerenciado a nível de sistema. Veja
[Managed Scope](/user-guide/managed-scope).
:::

## Substituição de Variáveis de Ambiente

Você pode referenciar variáveis de ambiente em `config.yaml` usando a sintaxe `${VAR_NAME}`:

```yaml
auxiliary:
  vision:
    api_key: ${GOOGLE_API_KEY}
    base_url: ${CUSTOM_VISION_URL}

delegation:
  api_key: ${DELEGATION_KEY}
```

Múltiplas referências em um único valor funcionam: `url: "${HOST}:${PORT}"`. Se uma variável referenciada não estiver definida, o placeholder é mantido literalmente (`${UNDEFINED_VAR}` permanece como está). Apenas a sintaxe `${VAR}` é suportada — `$VAR` simples não é expandido.

Para configuração de providers de IA (OpenRouter, Anthropic, Copilot, endpoints customizados, LLMs auto-hospedados, modelos de fallback, etc.), veja [AI Providers](/integrations/providers).

### Timeouts de Provider

Você pode definir `providers.<id>.request_timeout_seconds` para um timeout de requisição do provider, além de `providers.<id>.models.<model>.timeout_seconds` para uma substituição específica de modelo. Aplica-se ao cliente de turno principal em todos os transports (OpenAI-wire, native Anthropic, Anthropic-compatible), a cadeia de fallback, reconstruções após rotação de credenciais e (para OpenAI-wire) o argumento timeout por requisição — para que o valor configurado prevaleça sobre a variável de ambiente legada `HERMES_API_TIMEOUT`.

Você também pode definir `providers.<id>.stale_timeout_seconds` para o detector de chamadas obsoletas não-streaming, além de `providers.<id>.models.<model>.stale_timeout_seconds` para uma substituição específica de modelo. Isso prevalece sobre a variável de ambiente legada `HERMES_API_CALL_STALE_TIMEOUT`.

Deixar estes valores sem definição mantém os padrões legados (`HERMES_API_TIMEOUT=1800s`, `HERMES_API_CALL_STALE_TIMEOUT=90s`, Anthropic nativo 900s). O detector de chamadas obsoletas não-streaming é automaticamente desabilitado para endpoints locais quando deixado implícito e pode escalar para contextos muito grandes. Atualmente não configurado para AWS Bedrock (ambos os caminhos `bedrock_converse` e AnthropicBedrock SDK usam boto3 com sua própria configuração de timeout). Veja o exemplo comentado em [`cli-config.yaml.example`](https://github.com/NousResearch/hermes-agent/blob/main/cli-config.yaml.example).

## Comportamento de Atualização

As configurações de `hermes update` vivem em `updates` no `config.yaml`:

```yaml
updates:
  pre_update_backup: quick       # quick (snapshot de estado, padrão) | full (snapshot + zip do HERMES_HOME) | off
  backup_keep: 5                 # Manter este número de zips de backup completos pré-atualização
  non_interactive_local_changes: stash  # stash | discard
```

`pre_update_backup` é o único controle de segurança pré-atualização: `quick` (padrão) tira snapshot dos arquivos de estado críticos (dados de pareamento, cron jobs, config, auth; arquivos acima de 1 GiB são pulados) em `state-snapshots/`; `full` adicionalmente zipa todo o `HERMES_HOME` em `backups/` e pode adicionar minutos em homes grandes; `off` desabilita ambos. Booleanos legados são honrados (`true` → `full`, `false` → `off`).

Para instalações git, o Hermes faz auto-stash de arquivos rastreados sujos e arquivos não rastreados antes de fazer checkout do branch de atualização ou puxar. Atualizações interativas de terminal solicitam antes de restaurar esse stash. Atualizações não interativas (desktop/chat app, gateway ou `--yes`) usam `updates.non_interactive_local_changes`: `stash` restaura edições locais do código fonte após um pull bem-sucedido, enquanto `discard` descarta o stash criado pela atualização após um pull bem-sucedido. Use `discard` apenas em instalações gerenciadas onde edições locais do código fonte nunca devem persistir.

Antes dessa etapa de stash, o Hermes também restaura diferenças rastreadas de `package-lock.json` deixadas pelo churn de npm install/build. Faça commit ou faça stash manual de edições intencionais de lockfile antes de atualizar.

## Configuração do Backend de Terminal

O Hermes suporta seis backends de terminal. Cada um determina onde os comandos shell do agente realmente executam — sua máquina local, um container Docker, um servidor remoto via SSH, um sandbox cloud Modal (direto ou via gateway gerenciado pela Nous), um workspace Daytona ou um container Singularity/Apptainer.

```yaml
terminal:
  backend: local    # local | docker | ssh | modal | daytona | singularity
  cwd: "."          # Diretório de trabalho do gateway/cron (CLI sempre usa o diretório de lançamento)
  timeout: 180      # Timeout por comando em segundos
  home_mode: auto   # auto | real | profile — política HOME do subprocesso
  env_passthrough: []  # Nomes de variáveis de ambiente para encaminhar à execução em sandbox (terminal + execute_code)
  singularity_image: "docker://nikolaik/python-nodejs:python3.11-nodejs20"  # Imagem do container para Singularity backend
  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"                 # Imagem do container para Modal backend
  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"               # Imagem do container para Daytona backend
```

Para sandboxes cloud como Modal e Daytona, `container_persistent: true` significa que o Hermes tentará preservar o estado do sistema de arquivos entre recriações de sandbox. Não promete que o mesmo sandbox vivo, espaço PID ou processos em segundo plano ainda estarão em execução mais tarde.

### Visão Geral dos Backends

| Backend       | Onde os comandos executam              | Isolamento                    | Melhor para                           |
|---------------|----------------------------------------|-------------------------------|---------------------------------------|
| **local**     | Sua máquina diretamente                | Nenhum                        | Desenvolvimento, uso pessoal          |
| **docker**    | Container Docker persistente único     | Completo (namespaces, cap-drop)| Sandbox seguro, CI/CD                 |
| **ssh**       | Servidor remoto via SSH                | Limite de rede                | Desenvolvimento remoto, hardware potente |
| **modal**     | Sandbox cloud Modal                    | Completo (VM cloud)           | Computação cloud efêmera, avaliações  |
| **daytona**   | Workspace Daytona                      | Completo (container cloud)    | Ambientes de desenvolvimento cloud gerenciados |
| **singularity**| Container Singularity/Apptainer       | Namespaces (--containall)     | Clusters HPC, máquinas compartilhadas |

### Backend Local

O padrão. Comandos executam diretamente em sua máquina sem isolamento. Nenhuma configuração especial necessária.

```yaml
terminal:
  backend: local
```

Por padrão, subprocessos locais de ferramentas mantêm seu `HOME` real do usuário do SO. Isso permite que CLIs externos como `git`, `ssh`, `gh`, `az`, `npm`, Claude Code e Codex encontrem as credenciais e configuração que já usam em seu shell normal. O estado do Hermes ainda é escopo de profile através de `HERMES_HOME`; `HOME` não é como profiles selecionam config, memória, sessões ou skills.

O Hermes **não** altera seu `HOME` do sistema, seus arquivos de inicialização de shell ou o diretório home da conta do sistema operacional. Esta configuração controla apenas o ambiente passado para subprocessos que o Hermes inicia através de ferramentas como `terminal`, processos de terminal em segundo plano, `execute_code` e processos auxiliares ACP.

#### `terminal.home_mode`

| Modo     | Instalações host   | Containers                   | Trade-off                                                                                                |
|----------|--------------------|------------------------------|----------------------------------------------------------------------------------------------------------|
| `auto`   | Manter o `HOME` real do usuário do SO | Usar `{HERMES_HOME}/home`     | Padrão recomendado. CLIs do host continuam funcionando; estado do container persiste.                     |
| `real`   | Forçar o `HOME` real do usuário do SO | Forçar o `HOME` real do usuário do SO se visível | Útil se um processo pai acidentalmente iniciou com `HOME` apontado para um profile home.                 |
| `profile`| Usar `{HERMES_HOME}/home` quando existe  | Usar `{HERMES_HOME}/home` quando existe | Isolamento estrito de configuração CLI por profile, mas `~/.ssh`, `~/.gitconfig`, etc. não estarão visíveis. |

A desvantagem do padrão é que profiles do host compartilham as mesmas credenciais/configuração de CLI normais em nível de usuário sob `~`. Se você precisa de um profile com identidade git separada, chaves SSH, login GitHub CLI, config npm ou login CLI cloud, use `home_mode: profile` e inicialize essas ferramentas dentro desse profile home deliberadamente.

### Backend Docker

Executa comandos dentro de um container Docker com hardening de segurança (todas as capacidades removidas, sem escalonamento de privilégio, limites PID).

**Container persistente único, compartilhado entre processos Hermes.** O Hermes inicia UM container de longa duração no primeiro uso e roteia toda chamada de terminal, arquivo e `execute_code` através de `docker exec` para esse mesmo container — entre sessões, `/new`, `/reset` e subagentes `delegate_task`. Alterações de diretório de trabalho, pacotes instalados, arquivos em `/workspace` e **processos em segundo plano** todos persistem de uma chamada de ferramenta para a próxima.

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_mount_cwd_to_workspace: false  # Montar diretório de lançamento em /workspace
  docker_run_as_host_user: false
  docker_forward_env:
    - "GITHUB_TOKEN"
  docker_env:
    DEBUG: "1"
    PYTHONUNBUFFERED: "1"
  docker_volumes:
    - "/home/user/projects:/workspace/projects"
    - "/home/user/data:/data:ro"
  docker_extra_args:
    - "--gpus=all"
    - "--network=host"
  docker_network: true
  container_cpu: 1
  container_memory: 5120
  container_disk: 51200
  container_persistent: true
  docker_persist_across_processes: true
  docker_orphan_reaper: true
  timeout: 180
  lifetime_seconds: 300
```

(Pular seções puramente de referência YAML — o conteúdo técnico do restante do documento de configuração (conteúdo de 2127 linhas) permanece principalmente como blocos de código YAML que não são traduzidos.)

### Backend SSH

Executa comandos em um servidor remoto via SSH. Usa ControlMaster para reutilização de conexão (keepalive de 5 minutos ocioso). Shell persistente está ativado por padrão — estado (cwd, variáveis de ambiente) sobrevive entre comandos.

```yaml
terminal:
  backend: ssh
  persistent_shell: true
```

**Variáveis de ambiente necessárias:**

```bash
TERMINAL_SSH_HOST=my-server.example.com
TERMINAL_SSH_USER=ubuntu
```

**Opcional:**

| Variável               | Padrão         | Descrição                           |
|------------------------|----------------|-------------------------------------|
| `TERMINAL_SSH_PORT`    | `22`           | Porta SSH                           |
| `TERMINAL_SSH_KEY`     | (padrão do sistema) | Caminho para chave privada SSH    |
| `TERMINAL_SSH_PERSISTENT` | `true`      | Ativar shell persistente            |

(Os backends Modal, Daytona e Singularity seguem padrão similar com seus respectivos blocos YAML de configuração — consulte o original em inglês para os valores exatos.)

### Shell Persistente

Por padrão, cada comando de terminal executa em seu próprio subprocesso — diretório de trabalho, variáveis de ambiente e variáveis de shell são redefinidos entre comandos. Quando o **shell persistente** está ativado, um único processo bash de longa duração é mantido vivo entre chamadas `execute()` para que o estado sobreviva entre comandos.

```yaml
terminal:
  persistent_shell: true
```

## Configurações de Skills

Skills podem declarar suas próprias configurações através do frontmatter de seu SKILL.md. Estes são valores não secretos (caminhos, preferências, configurações de domínio) armazenados sob o namespace `skills.config` em `config.yaml`.

```yaml
skills:
  config:
    myplugin:
      path: ~/myplugin-data
```

**Como as configurações de skill funcionam:**

- `hermes config migrate` escaneia todas as skills ativadas, encontra configurações não configuradas e oferece solicitá-las
- `hermes config show` exibe todas as configurações de skill sob "Skill Settings"
- Quando uma skill carrega, seus valores de configuração resolvidos são injetados no contexto da skill automaticamente

**Definindo valores manualmente:**

```bash
hermes config set skills.config.myplugin.path ~/myplugin-data
```

Para detalhes sobre como declarar configurações em suas próprias skills, veja [Criando Skills — Config Settings](/developer-guide/creating-skills#config-settings-configyaml).

### Proteção em gravações de skill criadas pelo agente

Quando o agente usa `skill_manage` para criar, editar, patch ou deletar uma skill, o Hermes pode opcionalmente escanear o conteúdo novo/atualizado em busca de padrões de palavras-chave perigosas (coleta de credenciais, injeção de prompt óbvia, instruções de exfiltração). O scanner está **desligado por padrão** — fluxos de trabalho reais de agentes que legitimamente tocam em `~/.ssh/` ou mencionam `$OPENAI_API_KEY` estavam disparando a heurística com muita frequência.

```yaml
skills:
  guard_agent_created: true   # padrão: false
```

### Aprovação de gravação para skill writes

Independente do scanner de conteúdo acima, `skills.write_approval` coloca **toda** gravação de skill do agente (criar / editar / patch / deletar / arquivos de suporte) atrás de sua aprovação explícita:

```yaml
skills:
  write_approval: false   # false = escrever livremente (padrão) | true = estagiar toda gravação para revisão
```

## Configuração de Memória

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  write_approval: false
```

## Truncamento de Arquivo de Contexto

Controla quanto conteúdo o Hermes carrega de cada arquivo de contexto automático antes de aplicar truncamento cabeça/cauda.

```yaml
context_file_max_chars: 20000
```

## Segurança de Leitura de Arquivo

Controla quanto conteúdo uma única chamada `read_file` pode retornar.

```yaml
file_read_max_chars: 100000
```

## Limites de Truncamento de Saída de Ferramentas

```yaml
tool_output:
  max_bytes: 50000
  max_lines: 2000
  max_line_length: 2000
```

## Desativação Global de Toolset

Para suprimir toolsets específicos no CLI e em toda plataforma de gateway em um só lugar, liste seus nomes sob `agent.disabled_toolsets`:

```yaml
agent:
  disabled_toolsets:
    - memory
    - web
```

## Isolamento Git Worktree

Ative worktrees git isolados para executar múltiplos agentes em paralelo no mesmo repositório:

```yaml
worktree: true
worktree_sync: true
```

## Compressão de Contexto

O Hermes comprime automaticamente conversas longas para permanecer dentro da janela de contexto do seu modelo.

### Referência completa

```yaml
compression:
  enabled: true
  threshold: 0.50
  target_ratio: 0.20
  protect_last_n: 20
  protect_first_n: 3
  hygiene_hard_message_limit: 5000

auxiliary:
  compression:
    model: ""
    provider: "auto"
    base_url: null
```

### Configurações comuns

**Padrão (auto-detect) — nenhuma configuração necessária:**
```yaml
compression:
  enabled: true
  threshold: 0.50
```

**Forçar um provider específico:**
```yaml
auxiliary:
  compression:
    provider: nous
    model: gemini-3-flash
```

**Endpoint customizado:**
```yaml
auxiliary:
  compression:
    model: glm-4.7
    base_url: https://api.z.ai/api/coding/paas/v4
```

## Mecanismo de Contexto

```yaml
context:
  engine: "compressor"
```

## Orçamento de Iteração

```yaml
agent:
  max_turns: 90
  api_max_retries: 3
```

## Metas Persistentes (`/goal`)

```yaml
goals:
  max_turns: 20
```

### Timeouts de API

| Timeout                    | Padrão | Providers locais | Config / env                          |
|----------------------------|--------|------------------|---------------------------------------|
| Socket read timeout        | 120s   | Automático 1800s | `HERMES_STREAM_READ_TIMEOUT`          |
| Stale stream detection     | 180s   | Auto-desabilitado| `HERMES_STREAM_STALE_TIMEOUT`         |
| Stale non-stream detection | 300s   | Auto-desabilitado| `providers.<id>.stale_timeout_seconds` ou `HERMES_API_CALL_STALE_TIMEOUT` |
| API call (non-streaming)   | 1800s  | Inalterado       | `providers.<id>.request_timeout_seconds` / `timeout_seconds` ou `HERMES_API_TIMEOUT` |

## Avisos de Pressão de Contexto

Separado da pressão de orçamento de iteração, a pressão de contexto rastreia o quão perto a conversa está do **limiar de compactação**.

| Progresso            | Nível   | O que acontece                                                    |
|----------------------|---------|-------------------------------------------------------------------|
| **≥ 60%** do limiar  | Informação| CLI mostra barra de progresso ciano; gateway envia aviso informativo |
| **≥ 85%** do limiar  | Aviso   | CLI mostra barra amarela em negrito; gateway avisa que compactação é iminente |

## Estratégias de Pool de Credenciais

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

## Cache de Prompt

O Hermes ativa o cache de prompt entre sessões automaticamente quando o provider ativo o suporta.

```yaml
prompt_caching:
  cache_ttl: "5m"
```

## Modelos Auxiliares

O Hermes usa modelos "auxiliares" para tarefas secundárias como análise de imagem, sumarização de página web, análise de screenshot de navegador, geração de título de sessão e compressão de contexto. Por padrão (`auxiliary.*.provider: "auto"`), o Hermes roteia toda tarefa auxiliar para seu **modelo de chat principal**.

### Configurando modelos auxiliares interativamente

Em vez de editar YAML manualmente, execute `hermes model` e escolha **"Configure auxiliary models"** no menu.

### O padrão de configuração universal

| Chave      | O que faz                                                     | Padrão     |
|------------|---------------------------------------------------------------|------------|
| `provider` | Qual provider usar para auth e roteamento                     | `"auto"`   |
| `model`    | Qual modelo solicitar                                         | padrão do provider |
| `base_url` | Endpoint customizado compatível com OpenAI (sobrescreve provider)| não definido |

### Mudando o Modelo de Visão

```yaml
auxiliary:
  vision:
    model: "openai/gpt-4o"
```

Ou via variável de ambiente (em `~/.hermes/.env`):
```bash
AUXILIARY_VISION_MODEL=openai/gpt-4o
```

### Opções de Provider

| Provider          | Descrição                                                                             | Requisitos                           |
|-------------------|---------------------------------------------------------------------------------------|--------------------------------------|
| `"auto"`          | Melhor disponível (padrão). Visão tenta OpenRouter → Nous → Codex.                    | —                                    |
| `"openrouter"`    | Forçar OpenRouter                                                                     | `OPENROUTER_API_KEY`                 |
| `"nous"`          | Forçar Nous Portal                                                                    | `hermes auth`                        |
| `"codex"`         | Forçar Codex OAuth (conta ChatGPT). Suporta visão (gpt-5.3-codex).                   | `hermes model` → Codex               |
| `"minimax-oauth"` | Forçar MiniMax OAuth                                                                  | `hermes model` → MiniMax (OAuth)     |
| `"xai-oauth"`     | Forçar xAI Grok OAuth                                                                | `hermes model` → xAI Grok OAuth      |
| `"main"`          | Usar seu endpoint customizado/principal ativo                                         | Credenciais + URL base do endpoint   |

## Esforço de Raciocínio

Controla quanto "pensamento" o modelo faz antes de responder:

```yaml
agent:
  reasoning_effort: ""   # vazio = medium. Opções: none, minimal, low, medium, high, xhigh, max, ultra
```

### Substituições de Raciocínio por Modelo

```yaml
agent:
  reasoning_effort: "medium"
  reasoning_overrides:
    "openrouter/anthropic/claude-opus-4.5": "xhigh"
    "openai/gpt-5": "low"
```

## Reforço de Uso de Ferramentas

```yaml
agent:
  tool_use_enforcement: "auto"
```

| Valor        | Comportamento                                                                                             |
|--------------|-----------------------------------------------------------------------------------------------------------|
| `"auto"` (padrão) | Ativado para modelos contendo: `gpt`, `codex`, `gemini`, `gemma`, `grok`. Desativado para todos os outros. |
| `true`       | Sempre ativado                                                                                            |
| `false`      | Sempre desativado                                                                                         |
| `["gpt", "codex", "qwen", "llama"]` | Ativado apenas quando o nome do modelo contém uma das substrings listadas              |

## Proteções de Loop de Ferramentas

```yaml
tool_loop_guardrails:
  warnings_enabled: true
  hard_stop_enabled: false
  warn_after:
    exact_failure: 2
    same_tool_failure: 3
    idempotent_no_progress: 2
  hard_stop_after:
    exact_failure: 5
    same_tool_failure: 8
    idempotent_no_progress: 5
```

## Configuração de TTS

```yaml
tts:
  provider: "edge"
  speed: 1.0
  edge:
    voice: "en-US-AriaNeural"
    speed: 1.0
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"
    speed: 1.0
    base_url: "https://api.openai.com/v1"
  minimax:
    speed: 1.0
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"
  gemini:
    model: "gemini-2.5-flash-preview-tts"
    voice: "Kore"
  xai:
    voice_id: "eve"
    language: "en"
```

## Configurações de Exibição

```yaml
display:
  tool_progress: all
  tool_progress_command: false
  platforms: {}
  interim_assistant_messages: true
  show_commentary: true
  skin: default
  compact: false
  resume_display: full
  bell_on_complete: false
  show_reasoning: false
  streaming: false
  show_cost: false
  timestamps: false
  tool_preview_length: 0
  runtime_footer:
    enabled: false
    fields: ["model", "context_pct", "cwd"]
  file_mutation_verifier: true
  credits_notices: true
  language: en
```

### Verificador de mutação de arquivo

Quando `display.file_mutation_verifier` é `true` (padrão), o Hermes anexa um aviso de uma linha à resposta final do assistente sempre que uma chamada `write_file` ou `patch` falhou durante o turno e nunca foi substituída por uma gravação bem-sucedida no mesmo caminho.

### Idioma da interface para mensagens estáticas

A configuração `display.language` traduz um pequeno conjunto de mensagens estáticas voltadas ao usuário. Valores suportados: `en` (padrão), `zh`, `zh-hant`, `ja`, `de`, `es`, `fr`, `tr`, `uk`, `af`, `ko`, `it`, `ga`, `pt`, `ru`, `hu`.

## Privacidade

```yaml
privacy:
  redact_pii: false
```

## Fala para Texto (STT)

```yaml
stt:
  enabled: true
  echo_transcripts: true
  provider: "local"
  local:
    model: "base"
  openai:
    model: "whisper-1"
```

## Modo de Voz (CLI)

```yaml
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0
```

## Streaming

### CLI Streaming

```yaml
display:
  streaming: true
  show_reasoning: true
```

### Gateway Streaming (Telegram, Discord, Slack)

```yaml
streaming:
  enabled: true
  transport: edit
  edit_interval: 0.3
  buffer_threshold: 40
  cursor: " ▉"
  fresh_final_after_seconds: 0
```

## Isolamento de Sessão em Grupo

```yaml
max_concurrent_sessions: null
group_sessions_per_user: true
```

## Comportamento de DM Não Autorizada

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

## Comandos Rápidos

Defina comandos customizados que executam comandos shell sem invocar o LLM, ou criam alias de um comando de barra para outro.

```yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  disk:
    type: exec
    command: df -h /
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

## Atraso Humano

Simula ritmo de resposta humano em plataformas de mensagem:

```yaml
human_delay:
  mode: "off"
  min_ms: 800
  max_ms: 2500
```

## Execução de Código

```yaml
code_execution:
  mode: project
  timeout: 300
  max_tool_calls: 50
```

## Backends de Pesquisa Web

```yaml
web:
  backend: firecrawl
  search_backend: "searxng"
  extract_backend: "firecrawl"
```

| Backend              | Variável de Ambiente | Pesquisa | Extração |
|----------------------|---------------------|----------|----------|
| **Firecrawl** (padrão) | `FIRECRAWL_API_KEY`  | ✔        | ✔        |
| **SearXNG**          | `SEARXNG_URL`        | ✔        | —        |
| **Parallel**         | `PARALLEL_API_KEY`   | ✔        | ✔        |
| **Tavily**           | `TAVILY_API_KEY`     | ✔        | ✔        |
| **Exa**              | `EXA_API_KEY`        | ✔        | ✔        |

## Navegador

```yaml
browser:
  inactivity_timeout: 120
  command_timeout: 30
  record_sessions: false
  cdp_url: ""
  dialog_policy: must_respond
  dialog_timeout_s: 300
  camofox:
    managed_persistence: false
    user_id: ""
    session_key: ""
    adopt_existing_tab: false
```

## Fuso Horário

```yaml
timezone: "America/New_York"
```

## Discord

```yaml
discord:
  require_mention: true
  free_response_channels: ""
  auto_thread: true
```

## Segurança

```yaml
security:
  redact_secrets: true
  tirith_enabled: true
  tirith_path: "tirith"
  tirith_timeout: 5
  tirith_fail_open: true
  website_blocklist:
    enabled: false
    domains: []
    shared_files: []
```

## Lista de Bloqueio de Sites

```yaml
security:
  website_blocklist:
    enabled: false
    domains:
      - "*.internal.company.com"
      - "admin.example.com"
      - "*.local"
    shared_files:
      - "/etc/hermes/blocked-sites.txt"
```

## Aprovações Inteligentes

```yaml
approvals:
  mode: smart
```

| Modo     | Comportamento                                                                                                       |
|----------|---------------------------------------------------------------------------------------------------------------------|
| `smart` (padrão) | Usa um LLM auxiliar para avaliar se um comando sinalizado é realmente perigoso. Comandos de baixo risco são auto-aprovados. |
| `manual` | Solicitar ao usuário antes de executar qualquer comando sinalizado.                                                  |
| `off`    | Pular todas as verificações de aprovação. Equivalente a `HERMES_YOLO_MODE=true`.                                    |

### Regras de negação

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
```

## Checkpoints

```yaml
checkpoints:
  enabled: false
  max_snapshots: 20
```

## Delegação

```yaml
delegation:
  max_concurrent_children: 3
  max_spawn_depth: 1
  orchestrator_enabled: true
```

## Clarify

```yaml
clarify:
  timeout: 120
```

## Arquivos de Contexto (SOUL.md, AGENTS.md)

| Arquivo              | Propósito                                                        | Escopo                    |
|----------------------|------------------------------------------------------------------|---------------------------|
| `SOUL.md`            | **Identidade primária do agente** — define quem o agente é       | `~/.hermes/SOUL.md`       |
| `.hermes.md` / `HERMES.md` | Instruções específicas do projeto                          | Caminha até git root      |
| `AGENTS.md`          | Instruções específicas do projeto, convenções de código          | Caminhada recursiva       |
| `CLAUDE.md`          | Arquivos de contexto do Claude Code                              | Diretório de trabalho     |
| `.cursorrules`       | Regras do Cursor IDE                                             | Diretório de trabalho     |

## Diretório de Trabalho

| Contexto                               | Padrão                                                    |
|----------------------------------------|-----------------------------------------------------------|
| **CLI (`hermes`)**                     | Diretório atual onde você executa o comando               |
| **Messaging gateway**                  | `terminal.cwd` de `~/.hermes/config.yaml`                 |
| **Docker / Singularity / Modal / SSH** | Diretório home do usuário dentro do container ou máquina remota |

```yaml
terminal:
  cwd: /home/myuser/projects
```

## Rede

```yaml
network:
  force_ipv4: false
```

## Onboarding

```yaml
onboarding:
  profile_build: "ask"
```

## Dashboard

```yaml
dashboard:
  theme: "default"
  show_token_analytics: false
  public_url: ""
  oauth:
    client_id: ""
    portal_url: ""
  basic_auth:
    username: ""
    password_hash: ""
    password: ""
    secret: ""
    session_ttl_seconds: 0
  drain_auth:
    scope: "drain"
    min_secret_chars: 43
```
