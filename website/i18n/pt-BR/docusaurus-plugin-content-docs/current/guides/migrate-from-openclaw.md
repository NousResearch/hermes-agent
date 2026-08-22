---
sidebar_position: 10
title: "Migrar do OpenClaw"
description: "Guia completo para migrar sua configuração do OpenClaw / Clawdbot para o Hermes Agent — o que é migrado, como o config é mapeado e o que verificar depois."
---

# Migrar do OpenClaw

`hermes claw migrate` importa sua configuração do OpenClaw (ou do legado Clawdbot/Moldbot) para o Hermes. Este guia cobre exatamente o que é migrado, os mapeamentos de chaves de configuração e o que verificar após a migração.

:::tip
Se sua configuração do OpenClaw era multi-provedor, `hermes setup --portal` a reduz a um único OAuth — mais de 300 modelos mais o Tool Gateway em um único login. Veja [Nous Portal](/integrations/nous-portal).
:::

## Início rápido {#quick-start}

```bash
# Preview then migrate (always shows a preview first, then asks to confirm)
hermes claw migrate

# Preview only, no changes
hermes claw migrate --dry-run

# Full migration including API keys, skip confirmation
hermes claw migrate --preset full --migrate-secrets --yes
```

A migração sempre mostra um preview completo do que será importado antes de fazer qualquer alteração. Revise a lista e confirme para continuar.

Lê de `~/.openclaw/` por padrão. Diretórios legados `~/.clawdbot/` ou `~/.moltbot/` são detectados automaticamente. O mesmo vale para nomes de arquivo de configuração legados (`clawdbot.json`, `moltbot.json`).

## Opções {#options}

| Opção | Descrição |
|--------|-------------|
| `--dry-run` | Apenas preview — para depois de mostrar o que seria migrado. |
| `--preset <name>` | `full` (todas as configurações compatíveis) ou `user-data` (exclui configuração de infraestrutura). Nenhum dos presets importa segredos por padrão — passe `--migrate-secrets` explicitamente. |
| `--overwrite` | Sobrescreve arquivos existentes do Hermes em caso de conflito (padrão: recusa aplicar quando o plano tem conflitos). |
| `--migrate-secrets` | Inclui chaves de API. Necessário mesmo com `--preset full` — nenhum preset importa segredos silenciosamente. |
| `--no-backup` | Pula o snapshot zip pré-migração de `~/.hermes/` (por padrão, um único arquivo de ponto de restauração é gravado antes de aplicar, em `~/.hermes/backups/pre-migration-*.zip`; restaurável com `hermes import`). |
| `--source <path>` | Diretório personalizado do OpenClaw. |
| `--workspace-target <path>` | Onde colocar o `AGENTS.md`. |
| `--skill-conflict <mode>` | `skip` (padrão), `overwrite`, ou `rename`. |
| `--yes` | Pula o prompt de confirmação após o preview. |

## O que é migrado {#what-gets-migrated}

### Persona, memória e instruções {#persona-memory-and-instructions}

| O quê | Fonte no OpenClaw | Destino no Hermes | Notas |
|------|----------------|-------------------|-------|
| Persona | `workspace/SOUL.md` | `~/.hermes/SOUL.md` | Cópia direta |
| Instruções do workspace | `workspace/AGENTS.md` | `AGENTS.md` em `--workspace-target` | Requer a flag `--workspace-target` |
| Memória de longo prazo | `workspace/MEMORY.md` | `~/.hermes/memories/MEMORY.md` | Analisada em entradas, mesclada com as existentes, com duplicatas removidas. Usa o delimitador `§`. |
| Perfil do usuário | `workspace/USER.md` | `~/.hermes/memories/USER.md` | Mesma lógica de mesclagem de entradas que a memória. |
| Arquivos de memória diária | `workspace/memory/*.md` | `~/.hermes/memories/MEMORY.md` | Todos os arquivos diários mesclados na memória principal. |

Os arquivos do workspace também são verificados em `workspace.default/` e `workspace-main/` como caminhos alternativos (o OpenClaw renomeou `workspace/` para `workspace-main/` em versões recentes, e usa `workspace-{agentId}` para configurações multiagente).

### Skills (4 fontes) {#skills-4-sources}

| Fonte | Localização no OpenClaw | Destino no Hermes |
|--------|------------------|-------------------|
| Skills do workspace | `workspace/skills/` | `~/.hermes/skills/openclaw-imports/` |
| Skills gerenciadas/compartilhadas | `~/.openclaw/skills/` | `~/.hermes/skills/openclaw-imports/` |
| Pessoais entre projetos | `~/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |
| Compartilhadas no nível do projeto | `workspace/.agents/skills/` | `~/.hermes/skills/openclaw-imports/` |

Os conflitos de skills são tratados por `--skill-conflict`: `skip` mantém a skill existente do Hermes, `overwrite` a substitui, `rename` cria uma cópia `-imported`.

### Configuração de modelo e provedor {#model-and-provider-configuration}

| O quê | Caminho de config no OpenClaw | Destino no Hermes | Notas |
|------|---------------------|-------------------|-------|
| Modelo padrão | `agents.defaults.model` | `config.yaml` → `model` | Pode ser uma string ou um objeto `{primary, fallbacks}` |
| Provedores personalizados | `models.providers.*` | `config.yaml` → `custom_providers` | Mapeia `baseUrl`, `apiType`/`api` — trata tanto valores curtos ("openai", "anthropic") quanto hifenizados ("openai-completions", "anthropic-messages", "google-generative-ai") |
| Chaves de API dos provedores | `models.providers.*.apiKey` | `~/.hermes/.env` | Requer `--migrate-secrets`. Veja [Resolução de chaves de API](#api-key-resolution) abaixo. |

### Comportamento do agente {#agent-behavior}

| O quê | Caminho de config no OpenClaw | Caminho de config no Hermes | Mapeamento |
|------|---------------------|-------------------|---------|
| Máximo de turnos | `agents.defaults.timeoutSeconds` | `agent.max_turns` | `timeoutSeconds / 10`, limitado a 200 |
| Modo verboso | `agents.defaults.verboseDefault` | `agent.verbose` | "off" / "on" / "full" |
| Esforço de raciocínio | `agents.defaults.thinkingDefault` | `agent.reasoning_effort` | "always"/"high"/"xhigh" → "high", "auto"/"medium"/"adaptive" → "medium", "off"/"low"/"none"/"minimal" → "low" |
| Compressão | `agents.defaults.compaction.mode` | `compression.enabled` | "off" → false, qualquer outro valor → true |
| Modelo de compressão | `agents.defaults.compaction.model` | `compression.summary_model` | Cópia direta da string |
| Atraso humano | `agents.defaults.humanDelay.mode` | `human_delay.mode` | "natural" / "custom" / "off" |
| Tempo do atraso humano | `agents.defaults.humanDelay.minMs` / `.maxMs` | `human_delay.min_ms` / `.max_ms` | Cópia direta |
| Fuso horário | `agents.defaults.userTimezone` | `timezone` | Cópia direta da string |
| Tempo limite de execução | `tools.exec.timeoutSec` | `terminal.timeout` | Cópia direta (o campo é `timeoutSec`, não `timeout`) |
| Sandbox Docker | `agents.defaults.sandbox.backend` | `terminal.backend` | "docker" → "docker" |
| Imagem Docker | `agents.defaults.sandbox.docker.image` | `terminal.docker_image` | Cópia direta |

### Políticas de reinício de sessão {#session-reset-policies}

| Caminho de config no OpenClaw | Caminho de config no Hermes | Notas |
|---------------------|-------------------|-------|
| `session.reset.mode` | `session_reset.mode` | "daily", "idle", ou ambos |
| `session.reset.atHour` | `session_reset.at_hour` | Hora (0–23) para o reinício diário |
| `session.reset.idleMinutes` | `session_reset.idle_minutes` | Minutos de inatividade |

Nota: o OpenClaw também tem `session.resetTriggers` (um array de strings simples como `["daily", "idle"]`). Se o `session.reset` estruturado não estiver presente, a migração recorre a inferir a partir de `resetTriggers`.

### Servidores MCP {#mcp-servers}

| Campo no OpenClaw | Campo no Hermes | Notas |
|----------------|-------------|-------|
| `mcp.servers.*.command` | `mcp_servers.*.command` | Transporte stdio |
| `mcp.servers.*.args` | `mcp_servers.*.args` | |
| `mcp.servers.*.env` | `mcp_servers.*.env` | |
| `mcp.servers.*.cwd` | `mcp_servers.*.cwd` | |
| `mcp.servers.*.url` | `mcp_servers.*.url` | Transporte HTTP/SSE |
| `mcp.servers.*.tools.include` | `mcp_servers.*.tools.include` | Filtragem de ferramentas |
| `mcp.servers.*.tools.exclude` | `mcp_servers.*.tools.exclude` | |

### TTS (texto para fala) {#tts-text-to-speech}

As configurações de TTS são lidas de **duas** localizações de config do OpenClaw, nesta ordem de prioridade:

1. `messages.tts.providers.{provider}.*` (localização canônica)
2. `talk.providers.{provider}.*` no nível superior (alternativa)
3. Chaves planas legadas `messages.tts.{provider}.*` (formato mais antigo)

| O quê | Destino no Hermes |
|------|-------------------|
| Nome do provedor | `config.yaml` → `tts.provider` |
| ID de voz do ElevenLabs | `config.yaml` → `tts.elevenlabs.voice_id` |
| ID de modelo do ElevenLabs | `config.yaml` → `tts.elevenlabs.model_id` |
| Modelo da OpenAI | `config.yaml` → `tts.openai.model` |
| Voz da OpenAI | `config.yaml` → `tts.openai.voice` |
| Voz do Edge TTS | `config.yaml` → `tts.edge.voice` (o OpenClaw renomeou "edge" para "microsoft" — ambos são reconhecidos) |
| Arquivos de TTS | `~/.hermes/tts/` (cópia de arquivo) |

### Plataformas de mensagens {#messaging-platforms}

| Plataforma | Caminho de config no OpenClaw | Variável `.env` do Hermes | Notas |
|----------|---------------------|----------------------|-------|
| Telegram | `channels.telegram.botToken` ou `.accounts.default.botToken` | `TELEGRAM_BOT_TOKEN` | O token pode ser uma string ou [SecretRef](#secretref-handling). Ambos os layouts, plano e por contas, são suportados. |
| Telegram | `credentials/telegram-default-allowFrom.json` | `TELEGRAM_ALLOWED_USERS` | Unido por vírgulas a partir do array `allowFrom[]` |
| Discord | `channels.discord.token` ou `.accounts.default.token` | `DISCORD_BOT_TOKEN` | |
| Discord | `channels.discord.allowFrom` ou `.accounts.default.allowFrom` | `DISCORD_ALLOWED_USERS` | |
| Slack | `channels.slack.botToken` ou `.accounts.default.botToken` | `SLACK_BOT_TOKEN` | |
| Slack | `channels.slack.appToken` ou `.accounts.default.appToken` | `SLACK_APP_TOKEN` | |
| Slack | `channels.slack.allowFrom` ou `.accounts.default.allowFrom` | `SLACK_ALLOWED_USERS` | |
| WhatsApp | `channels.whatsapp.allowFrom` ou `.accounts.default.allowFrom` | `WHATSAPP_ALLOWED_USERS` | Autenticação via pareamento por QR do Baileys — requer repareamento após a migração |
| Signal | `channels.signal.account` ou `.accounts.default.account` | `SIGNAL_ACCOUNT` | |
| Signal | `channels.signal.httpUrl` ou `.accounts.default.httpUrl` | `SIGNAL_HTTP_URL` | |
| Signal | `channels.signal.allowFrom` ou `.accounts.default.allowFrom` | `SIGNAL_ALLOWED_USERS` | |
| Matrix | `channels.matrix.accessToken` ou `.accounts.default.accessToken` | `MATRIX_ACCESS_TOKEN` | Usa `accessToken` (não `botToken`) |
| Mattermost | `channels.mattermost.botToken` ou `.accounts.default.botToken` | `MATTERMOST_BOT_TOKEN` | |

### Outras configurações {#other-config}

| O quê | Caminho no OpenClaw | Caminho no Hermes | Notas |
|------|-------------|-------------|-------|
| Modo de aprovação | `approvals.exec.mode` | `config.yaml` → `approvals.mode` | "auto"→"off", "always"→"manual", "smart"→"smart" |
| Lista de permissão de comandos | `exec-approvals.json` | `config.yaml` → `command_allowlist` | Padrões mesclados e sem duplicatas |
| URL CDP do navegador | `browser.cdpUrl` | `config.yaml` → `browser.cdp_url` | |
| Navegador sem interface | `browser.headless` | `config.yaml` → `browser.headless` | |
| Chave de busca Brave | `tools.web.search.brave.apiKey` | `.env` → `BRAVE_API_KEY` | Requer `--migrate-secrets` |
| Token de autenticação do gateway | `gateway.auth.token` | `.env` → `HERMES_GATEWAY_TOKEN` | Requer `--migrate-secrets` |
| Diretório de trabalho | `agents.defaults.workspace` | `config.yaml` → `terminal.cwd` | Migrações legadas ainda podem emitir `MESSAGING_CWD` como alternativa de compatibilidade |

### Arquivados (sem equivalente direto no Hermes) {#archived-no-direct-hermes-equivalent}

Estes são salvos em `~/.hermes/migration/openclaw/<timestamp>/archive/` para revisão manual:

| O quê | Arquivo de arquivamento | Como recriar no Hermes |
|------|-------------|--------------------------|
| `IDENTITY.md` | `archive/workspace/IDENTITY.md` | Mesclar no `SOUL.md` |
| `TOOLS.md` | `archive/workspace/TOOLS.md` | O Hermes tem instruções de ferramentas integradas |
| `HEARTBEAT.md` | `archive/workspace/HEARTBEAT.md` | Use tarefas agendadas (cron jobs) para tarefas periódicas |
| `BOOTSTRAP.md` | `archive/workspace/BOOTSTRAP.md` | Use arquivos de contexto ou skills |
| Tarefas agendadas | `archive/cron-config.json` | Recrie com `hermes cron create` |
| Plugins | `archive/plugins-config.json` | Veja o [guia de plugins](/user-guide/features/hooks) |
| Hooks/webhooks | `archive/hooks-config.json` | Use `hermes webhook` ou hooks do gateway |
| Backend de memória | `archive/memory-backend-config.json` | Configure via `hermes honcho` |
| Registro de skills | `archive/skills-registry-config.json` | Use `hermes skills config` |
| Identidade/UI | `archive/ui-identity-config.json` | Use o comando `/skin` |
| Logging | `archive/logging-diagnostics-config.json` | Defina na seção logging do `config.yaml` |
| Lista multiagente | `archive/agents-list.json` | Use perfis do Hermes |
| Vínculos de canal | `archive/bindings.json` | Configuração manual por plataforma |
| Canais complexos | `archive/channels-deep-config.json` | Configuração manual de plataforma |

## Resolução de chaves de API {#api-key-resolution}

Quando `--migrate-secrets` está ativado, as chaves de API são coletadas de **quatro fontes**, em ordem de prioridade:

1. **Valores de configuração** — `models.providers.*.apiKey` e chaves de provedores de TTS em `openclaw.json`
2. **Arquivo de ambiente** — `~/.openclaw/.env` (chaves como `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
3. **Subobjeto de env na configuração** — `openclaw.json` → `"env"` ou `"env"."vars"` (algumas configurações armazenam chaves aqui em vez de um arquivo `.env` separado)
4. **Perfis de autenticação** — `~/.openclaw/agents/main/agent/auth-profiles.json` (credenciais por agente)

Os valores de configuração têm prioridade. Cada fonte subsequente preenche as lacunas restantes.

### Alvos de chaves suportados {#supported-key-targets}

`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `MINIMAX_API_KEY`, `ELEVENLABS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `VOICE_TOOLS_OPENAI_KEY`

Chaves fora dessa lista de permissão nunca são copiadas.

## Tratamento de SecretRef {#secretref-handling}

Os valores de configuração do OpenClaw para tokens e chaves de API podem estar em três formatos:

```json
// Plain string
"channels": { "telegram": { "botToken": "123456:ABC-DEF..." } }

// Environment template
"channels": { "telegram": { "botToken": "${TELEGRAM_BOT_TOKEN}" } }

// SecretRef object
"channels": { "telegram": { "botToken": { "source": "env", "id": "TELEGRAM_BOT_TOKEN" } } }
```

A migração resolve os três formatos. Para templates de ambiente e objetos SecretRef com `source: "env"`, ela procura o valor em `~/.openclaw/.env` e no subobjeto de env de `openclaw.json`. Objetos SecretRef com `source: "file"` ou `source: "exec"` não podem ser resolvidos automaticamente — a migração avisa sobre eles, e esses valores devem ser adicionados manualmente ao Hermes via `hermes config set`.

## Após a migração {#after-migration}

1. **Verifique o relatório de migração** — impresso ao final, com contagens de itens migrados, ignorados e conflitantes.

2. **Revise os arquivos arquivados** — qualquer coisa em `~/.hermes/migration/openclaw/<timestamp>/archive/` precisa de atenção manual.

3. **Inicie uma nova sessão** — skills e entradas de memória importadas têm efeito em novas sessões, não na atual.

4. **Verifique as chaves de API** — execute `hermes status` para checar a autenticação dos provedores.

5. **Teste as mensagens** — se você migrou tokens de plataforma, reinicie o gateway: `systemctl --user restart hermes-gateway`

6. **Verifique as políticas de sessão** — execute `hermes config show` e confirme que o valor de `session_reset` corresponde às suas expectativas.

7. **Repareie o WhatsApp** — o WhatsApp usa pareamento por código QR (Baileys), não migração de token. Execute `hermes whatsapp` para parear.

8. **Limpeza do arquivamento** — depois de confirmar que tudo funciona, execute `hermes claw cleanup` para renomear os diretórios remanescentes do OpenClaw para `.pre-migration/` (evita confusão de estado).

## Solução de Problemas {#troubleshooting}

### "OpenClaw directory not found" {#openclaw-directory-not-found}

A migração verifica `~/.openclaw/`, depois `~/.clawdbot/`, depois `~/.moltbot/`. Se sua instalação estiver em outro lugar, use `--source /path/to/your/openclaw`.

### "No provider API keys found" {#no-provider-api-keys-found}

As chaves podem estar armazenadas em vários lugares, dependendo da sua versão do OpenClaw: inline em `openclaw.json` sob `models.providers.*.apiKey`, em `~/.openclaw/.env`, no subobjeto `"env"` de `openclaw.json`, ou em `agents/main/agent/auth-profiles.json`. A migração verifica todos os quatro. Se as chaves usarem SecretRefs com `source: "file"` ou `source: "exec"`, elas não podem ser resolvidas automaticamente — adicione-as via `hermes config set`.

### Skills não aparecem após a migração {#skills-not-appearing-after-migration}

Skills importadas ficam em `~/.hermes/skills/openclaw-imports/`. Inicie uma nova sessão para que elas tenham efeito, ou execute `/skills` para verificar se foram carregadas.

### Voz de TTS não migrada {#tts-voice-not-migrated}

O OpenClaw armazena configurações de TTS em dois lugares: `messages.tts.providers.*` e a configuração `talk` de nível superior. A migração verifica ambos. Se o ID da sua voz foi definido pela interface do OpenClaw (armazenado em um caminho diferente), pode ser necessário defini-lo manualmente: `hermes config set tts.elevenlabs.voice_id YOUR_VOICE_ID`.
