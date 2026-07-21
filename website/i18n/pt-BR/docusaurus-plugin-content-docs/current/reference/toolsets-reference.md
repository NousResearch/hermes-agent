---
sidebar_position: 4
title: "Referência de Toolsets"
description: "Referência dos toolsets core, compostos, de plataforma e dinâmicos do Hermes"
---

# Referência de Toolsets

Toolsets são pacotes nomeados de ferramentas que controlam o que o agente pode fazer. São o principal mecanismo para configurar a disponibilidade de ferramentas por plataforma, por sessão ou por tarefa.

## Como os Toolsets Funcionam {#how-toolsets-work}

Cada ferramenta pertence a exatamente um toolset. Ao ativar um toolset, todas as ferramentas daquele pacote ficam disponíveis para o agente. Os toolsets existem em três tipos:

- **Core** — Um único grupo lógico de ferramentas relacionadas (ex.: `file` reúne `read_file`, `write_file`, `patch`, `search_files`)
- **Composto** — Combina múltiplos toolsets core para um cenário comum (ex.: `debugging` reúne ferramentas de arquivo, terminal e web)
- **Plataforma** — Uma configuração completa de ferramentas para um contexto de implantação específico (ex.: `hermes-cli` é o padrão para sessões interativas de CLI)

## Configurando Toolsets {#configuring-toolsets}

### Por sessão (CLI) {#per-session-cli}

```bash
hermes chat --toolsets web,file,terminal
hermes chat --toolsets debugging        # composto — expande para file + terminal + web
hermes chat --toolsets all              # tudo
```

### Por plataforma (config.yaml) {#per-platform-configyaml}

```yaml
toolsets:
  - hermes-cli          # padrão para CLI
  # - hermes-telegram   # sobrescreve para o gateway do Telegram
```

### Gerenciamento interativo {#interactive-management}

```bash
hermes tools                            # UI em curses para ativar/desativar por plataforma
```

Ou dentro da sessão:

```
/tools list
/tools disable browser
/tools enable homeassistant
```

## Toolsets Core {#core-toolsets}

| Toolset | Ferramentas | Finalidade |
|---------|-------|---------|
| `browser` | `browser_back`, `browser_cdp`, `browser_click`, `browser_console`, `browser_dialog`, `browser_get_images`, `browser_navigate`, `browser_press`, `browser_scroll`, `browser_snapshot`, `browser_type`, `browser_vision`, `web_search` | Automação de navegador core. Inclui `web_search` como alternativa para buscas rápidas. `browser_cdp` e `browser_dialog` são condicionados em runtime — só são registrados quando um endpoint CDP está acessível no início da sessão (via `/browser connect`, configuração `browser.cdp_url`, Browserbase ou Camofox). `browser_dialog` funciona junto com os campos `pending_dialogs` e `frame_tree` que `browser_snapshot` adiciona quando um supervisor CDP está conectado. |
| `clarify` | `clarify` | Fazer uma pergunta ao usuário quando o agente precisa de esclarecimento. |
| `code_execution` | `execute_code` | Executar scripts Python que chamam ferramentas do Hermes programaticamente. |
| `coding` | composto (`file` + `terminal` + `search` + `web` + `skills` + `browser` + `todo` + `memory` + `session_search` + `clarify` + `code_execution` + `delegation` + `vision`) | Pacote focado em programação: edição de arquivos, terminal, busca, documentação web, skills, navegador, delegação e execução de código. |
| `cronjob` | `cronjob` | Agendar e gerenciar tarefas recorrentes. |
| `debugging` | composto (`file` + `terminal` + `web`) | Pacote de depuração — arquivo, processo/terminal, extração/busca web. |
| `delegation` | `delegate_task` | Criar instâncias isoladas de subagentes para trabalho paralelo. |
| `discord` | `discord` | Ações core de texto/embed/DM do Discord (apenas gateway). Ativo no toolset `hermes-discord`. |
| `discord_admin` | `discord_admin` | Moderação do Discord (banimentos, mudanças de cargo, gestão de canais). Ativo no toolset `hermes-discord`; requer que o bot tenha as permissões relevantes do Discord. |
| `feishu_doc` | `feishu_doc_read` | Ler o conteúdo de documentos Feishu/Lark. Usado pelo handler de resposta inteligente a comentários de documentos do Feishu. |
| `feishu_drive` | `feishu_drive_add_comment`, `feishu_drive_list_comments`, `feishu_drive_list_comment_replies`, `feishu_drive_reply_comment` | Operações de comentário no drive do Feishu/Lark. Restrito ao agente de comentários; não exposto no `hermes-cli` ou em outros toolsets de mensageria. |
| `file` | `patch`, `read_file`, `search_files`, `write_file` | Leitura, escrita, busca e edição de arquivos. |
| `homeassistant` | `ha_call_service`, `ha_get_state`, `ha_list_entities`, `ha_list_services` | Controle de casa inteligente via Home Assistant. Disponível apenas quando `HASS_TOKEN` está definido. |
| `computer_use` | `computer_use` | Controle de desktop em segundo plano via cua-driver — não rouba o cursor/foco. Funciona com qualquer modelo capaz de usar ferramentas. macOS, Windows e Linux; requer `cua-driver` no `$PATH`. |
| `context_engine` | (varia) | Ferramentas de runtime expostas pelo plugin de context engine ativo (vazio até que um plugin o preencha). |
| `image_gen` | `image_generate` | Geração de texto-para-imagem via FAL.ai (com backends opcionais OpenAI / xAI). |
| `video_gen` | `video_generate` | Texto-para-vídeo e imagem-para-vídeo via backends registrados por plugins (xAI Grok-Imagine, FAL.ai Veo 3.1 / Pixverse v6 / Kling O3). Passe `image_url` para animar uma imagem; omita para texto-para-vídeo. |
| `kanban` | `kanban_block`, `kanban_comment`, `kanban_complete`, `kanban_create`, `kanban_heartbeat`, `kanban_link`, `kanban_list`, `kanban_show`, `kanban_unblock` | Ferramentas de coordenação multiagente. Registradas para workers de tarefa gerados pelo dispatcher (`HERMES_KANBAN_TASK`) e para perfis que listam explicitamente o toolset `kanban` pelo nome (o wildcard `all`/`*` **não** o ativa). Workers marcam tarefas como concluídas, bloqueiam, enviam heartbeat, comentam e criam/vinculam tarefas de acompanhamento; perfis orquestradores também recebem ferramentas de roteamento de board como list/unblock. |
| `memory` | `memory` | Gerenciamento de memória persistente entre sessões. |
| `project` | `project_create`, `project_list`, `project_switch` | Criar e alternar entre [Projects](../user-guide/cli.md) de desktop (workspaces nomeados e multi-pasta). Apenas sessões GUI/desktop. |
| `safe` | `image_generate`, `vision_analyze`, `web_extract`, `web_search` (via `includes`) | Pesquisa somente leitura + geração de mídia. Sem escrita de arquivos, sem terminal, sem execução de código. |
| `search` | `web_search` | Apenas busca web (sem extração). |
| `session_search` | `session_search` | Buscar sessões de conversa anteriores. |
| `skills` | `skill_manage`, `skill_view`, `skills_list` | CRUD e navegação de skills. |
| `spotify` | `spotify_albums`, `spotify_devices`, `spotify_library`, `spotify_playback`, `spotify_playlists`, `spotify_queue`, `spotify_search` | Controle nativo do Spotify (reprodução, fila, busca, playlists, álbuns, biblioteca). Registrado pelo plugin `spotify` incluído. |
| `terminal` | `process`, `terminal` | Execução de comandos shell e gerenciamento de processos em segundo plano. |
| `todo` | `todo` | Gerenciamento de lista de tarefas dentro de uma sessão. |
| `tts` | `text_to_speech` | Geração de áudio texto-para-fala. |
| `vision` | `vision_analyze` | Análise de imagens via modelos com capacidade de visão. |
| `video` | `video_analyze` | Ferramentas de análise e compreensão de vídeo (opt-in, não presente no toolset padrão — adicione explicitamente via `--toolsets`). |
| `web` | `web_extract`, `web_search` | Busca web e extração de conteúdo de páginas. |
| `x_search` | `x_search` | Buscar posts e threads do X (Twitter) via ferramenta Responses `x_search` embutida da xAI. Desativado por padrão; ative via `hermes tools`. O schema só é registrado quando credenciais da xAI (OAuth SuperGrok ou `XAI_API_KEY`) estão configuradas. |
| `yuanbao` | `yb_query_group_info`, `yb_query_group_members`, `yb_search_sticker`, `yb_send_dm`, `yb_send_sticker` | Ações de DM/grupo e busca de stickers do Yuanbao. Registrado apenas em `hermes-yuanbao`. |

## Toolsets de Plataforma {#platform-toolsets}

Toolsets de plataforma definem a configuração completa de ferramentas para um destino de implantação. A maioria das plataformas de mensageria usa o mesmo conjunto do `hermes-cli`:

| Toolset | Diferenças em relação ao `hermes-cli` |
|---------|-------------------------------|
| `hermes-cli` | Toolset completo — o padrão para sessões interativas de CLI. Inclui file, terminal, web, browser, memory, skills, vision, image_gen, todo, tts, delegation, code_execution, cronjob, session_search e clarify, além do pacote `safe` (somente leitura). |
| `hermes-acp` | Remove `clarify`, `cronjob`, `image_generate`, `text_to_speech` e as quatro ferramentas do Home Assistant. Focado em tarefas de programação em contexto de IDE. |
| `hermes-api-server` | Remove `clarify` e `text_to_speech`. Mantém todo o resto — adequado para acesso programático onde a interação com o usuário não é possível. |
| `hermes-cron` | Igual ao `hermes-cli`. |
| `hermes-telegram` | Igual ao `hermes-cli`. |
| `hermes-discord` | Adiciona `discord` e `discord_admin` sobre o `hermes-cli`. |
| `hermes-slack` | Igual ao `hermes-cli`. |
| `hermes-whatsapp` | Igual ao `hermes-cli`. |
| `hermes-signal` | Igual ao `hermes-cli`. |
| `hermes-matrix` | Igual ao `hermes-cli`. |
| `hermes-mattermost` | Igual ao `hermes-cli`. |
| `hermes-email` | Igual ao `hermes-cli`. |
| `hermes-sms` | Igual ao `hermes-cli`. |
| `hermes-bluebubbles` | Igual ao `hermes-cli`. |
| `hermes-dingtalk` | Igual ao `hermes-cli`. |
| `hermes-feishu` | Adiciona as cinco ferramentas `feishu_doc_*` / `feishu_drive_*` (usadas apenas pelo handler de comentários em documentos, não pelo adaptador de chat comum). |
| `hermes-qqbot` | Igual ao `hermes-cli`. |
| `hermes-wecom` | Igual ao `hermes-cli`. |
| `hermes-wecom-callback` | Igual ao `hermes-cli`. |
| `hermes-weixin` | Igual ao `hermes-cli`. |
| `hermes-yuanbao` | Adiciona as cinco ferramentas `yb_*` (DM/grupo/sticker) sobre o `hermes-cli`. |
| `hermes-homeassistant` | Igual ao `hermes-cli` (as ferramentas do Home Assistant já estão presentes por padrão e se ativam quando `HASS_TOKEN` está definido). |
| `hermes-webhook` | Igual ao `hermes-cli`. |
| `hermes-gateway` | Toolset orquestrador interno do gateway — união de todos os toolsets `hermes-<platform>`; usado quando o gateway precisa aceitar qualquer origem de mensagem. |

## Toolsets Dinâmicos {#dynamic-toolsets}

### Toolsets de servidor MCP {#mcp-server-toolsets}

Cada servidor MCP configurado gera um toolset `mcp-<server>` em runtime. Por exemplo, se você configurar um servidor MCP `github`, um toolset `mcp-github` é criado contendo todas as ferramentas que aquele servidor expõe.

```yaml
# config.yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
```

Isso cria um toolset `mcp-github` que você pode referenciar em `--toolsets` ou em configurações de plataforma.

### Toolsets de plugins {#plugin-toolsets}

Plugins podem registrar seus próprios toolsets via `ctx.register_tool()` durante a inicialização do plugin. Eles aparecem junto aos toolsets embutidos e podem ser ativados/desativados da mesma forma.

### Toolsets personalizados {#custom-toolsets}

Defina toolsets personalizados em `config.yaml` para criar pacotes específicos do projeto:

```yaml
toolsets:
  - hermes-cli
custom_toolsets:
  data-science:
    - file
    - terminal
    - code_execution
    - web
    - vision
```

### Wildcards {#wildcards}

- `all` ou `*` — expande para todos os toolsets registrados (embutidos + dinâmicos + plugins)

Algumas ferramentas têm uma verificação de disponibilidade adicional além da participação no toolset e **não** são ativadas apenas por `all`/`*`:

- Ferramentas **condicionadas por capacidade** (browser, `computer_use`, `code_execution`, Feishu, Home Assistant, cronjob) só aparecem quando o pré-requisito de backend/credencial está configurado.
- Ferramentas **condicionadas por fluxo de trabalho** — o toolset `kanban` — são deliberadamente opt-in. `all`/`*` **não** ativa o kanban; você precisa listar `kanban` explicitamente (ou ser um worker gerado pelo dispatcher com `HERMES_KANBAN_TASK` definido). As ferramentas do kanban alteram o estado compartilhado do board, então permanecem desativadas por padrão mesmo sob `all`.

## Relação com `hermes tools` {#relationship-to-hermes-tools}

O comando `hermes tools` fornece uma UI baseada em curses para ativar/desativar ferramentas individuais por plataforma. Isso opera no nível da ferramenta (mais granular que os toolsets) e persiste em `config.yaml`. Ferramentas desativadas são filtradas mesmo que seu toolset esteja ativado.

Veja também: [Referência de Ferramentas](./tools-reference.md) para a lista completa de ferramentas individuais e seus parâmetros.
