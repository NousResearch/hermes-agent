---
sidebar_position: 3
title: "Referência de Ferramentas Embutidas"
description: "Referência oficial das ferramentas embutidas do Hermes, agrupadas por toolset"
---

# Referência de Ferramentas Embutidas

Esta página documenta as ferramentas embutidas do Hermes, agrupadas por toolset. A disponibilidade varia por plataforma, credenciais e toolsets ativados.

**Contagem rápida (registro atual):** ~73 ferramentas — 10 ferramentas de browser (core) + 2 ferramentas de browser condicionadas por CDP, 4 ferramentas de arquivo, 4 ferramentas do Home Assistant, 3 ferramentas de terminal (`terminal`, `process`, `read_terminal`), 2 ferramentas web, 5 ferramentas do Feishu, 7 ferramentas do Spotify (registradas pelo plugin `spotify` incluído), 5 ferramentas do Yuanbao, 9 ferramentas de kanban (registradas quando o dispatcher do kanban cria o agente), 3 ferramentas de projeto (sessões desktop/GUI), 2 ferramentas do Discord, e um punhado de ferramentas independentes (`memory`, `clarify`, `delegate_task`, `execute_code`, `cronjob`, `session_search`, `skill_view`/`skill_manage`/`skills_list`, `text_to_speech`, `image_generate`, `video_generate`, `vision_analyze`, `video_analyze`, `todo`, `computer_use`).

:::tip Ferramentas MCP
Além das ferramentas embutidas, o Hermes pode carregar ferramentas dinamicamente de servidores MCP. As ferramentas MCP aparecem com o prefixo `mcp_<server>_` (ex.: `mcp_github_create_issue` para o servidor MCP `github`). Veja [Integração MCP](/user-guide/features/mcp) para configuração.
:::

## Toolset `browser` {#browser-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `browser_back` | Navega para a página anterior no histórico do navegador. Requer que browser_navigate tenha sido chamado antes. | — |
| `browser_click` | Clica em um elemento identificado pelo seu ref ID no snapshot (ex.: '@e5'). Os ref IDs aparecem entre colchetes na saída do snapshot. Requer que browser_navigate e browser_snapshot tenham sido chamados antes. | — |
| `browser_console` | Obtém a saída do console do navegador e erros de JavaScript da página atual. Retorna mensagens console.log/warn/error/info e exceções JS não tratadas. Use para detectar erros silenciosos de JavaScript, chamadas de API que falharam e avisos da aplicação. Requi… | — |
| `browser_get_images` | Obtém uma lista de todas as imagens na página atual com suas URLs e texto alternativo. Útil para encontrar imagens para analisar com a ferramenta de visão. Requer que browser_navigate tenha sido chamado antes. | — |
| `browser_navigate` | Navega para uma URL no navegador. Inicializa a sessão e carrega a página. Deve ser chamado antes de outras ferramentas de browser. Para buscas simples de informação, prefira web_search ou web_extract (mais rápidas, mais baratas). Use ferramentas de browser quando precisar… | — |
| `browser_press` | Pressiona uma tecla do teclado. Útil para enviar formulários (Enter), navegar (Tab) ou atalhos de teclado. Requer que browser_navigate tenha sido chamado antes. | — |
| `browser_scroll` | Rola a página em uma direção. Use para revelar mais conteúdo que pode estar abaixo ou acima da viewport atual. Requer que browser_navigate tenha sido chamado antes. | — |
| `browser_snapshot` | Obtém um snapshot em texto da árvore de acessibilidade da página atual. Retorna elementos interativos com ref IDs (como @e1, @e2) para browser_click e browser_type. full=false (padrão): visão compacta com elementos interativos. full=true: comp… | — |
| `browser_type` | Digita texto em um campo de entrada identificado pelo seu ref ID. Limpa o campo primeiro, depois digita o novo texto. Requer que browser_navigate e browser_snapshot tenham sido chamados antes. | — |
| `browser_vision` | Tira um screenshot da página atual e a analisa com IA de visão. Use quando precisar entender visualmente o que está na página — especialmente útil para CAPTCHAs, desafios de verificação visual, layouts complexos, ou quando o snaps… (de texto) não for suficiente. | — |

## Toolset `browser` (ferramentas condicionadas por CDP) {#browser-toolset-cdp-gated-tools}

Essas duas ferramentas vivem no toolset `browser`, mas só se registram quando um endpoint Chrome DevTools Protocol está acessível no início da sessão — via `/browser connect`, configuração `browser.cdp_url`, uma sessão Browserbase, ou Camofox.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `browser_cdp` | Envia um comando bruto do Chrome DevTools Protocol. Escape hatch para operações de browser não cobertas pelas ferramentas `browser_*` de alto nível. Veja https://chromedevtools.github.io/devtools-protocol/ | Endpoint CDP |
| `browser_dialog` | Responde a um diálogo nativo de JavaScript (alert / confirm / prompt / beforeunload). Chame `browser_snapshot` primeiro — diálogos pendentes aparecem no campo `pending_dialogs`. Depois chame `browser_dialog(action='accept'\|'dismiss')`. | Endpoint CDP |

## Toolset `clarify` {#clarify-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `clarify` | Faz uma pergunta ao usuário quando você precisa de esclarecimento, feedback ou uma decisão antes de continuar. Suporta dois modos: 1. **Múltipla escolha** — forneça até 4 opções. O usuário escolhe uma ou digita sua própria resposta via uma 5ª opção 'Outro'. 2.… | — |

## Toolset `code_execution` {#code_execution-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `execute_code` | Executa um script Python que pode chamar ferramentas do Hermes programaticamente. Use quando precisar de 3+ chamadas de ferramentas com lógica de processamento entre elas, precisar filtrar/reduzir saídas grandes de ferramentas antes que entrem no seu contexto, precisar de ramificação condicional (… | — |

## Toolset `cronjob` {#cronjob-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `cronjob` | Gerenciador unificado de tarefas agendadas. Use `action="create"`, `"list"`, `"update"`, `"pause"`, `"resume"`, `"run"`, ou `"remove"` para gerenciar jobs. Suporta jobs com skills anexadas (uma ou mais), e `skills=[]` em um update limpa as skills anexadas. Execuções cron acontecem em sessões novas, sem contexto da conversa atual. | — |

## Toolset `delegation` {#delegation-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `delegate_task` | Cria um ou mais subagentes para trabalhar em tarefas em contextos isolados. Cada subagente recebe sua própria conversa, sessão de terminal e toolset. Apenas o resumo final é retornado — resultados intermediários de ferramentas nunca entram na sua janela de contexto. DUAS… | — |

## Toolset `feishu_doc` {#feishu_doc-toolset}

Restrito ao handler de resposta inteligente a comentários de documentos do Feishu (`gateway/platforms/feishu_comment.py`). Não exposto no `hermes-cli` ou no adaptador de chat comum do Feishu.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `feishu_doc_read` | Lê o conteúdo de texto completo de um documento Feishu/Lark (Docx, Doc ou Sheet) dado seu file_type e token. | Credenciais do app Feishu |

## Toolset `feishu_drive` {#feishu_drive-toolset}

Restrito ao handler de comentários de documentos do Feishu. Controla operações de leitura/escrita de comentários em arquivos do drive.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `feishu_drive_add_comment` | Adiciona um comentário de nível superior em um documento ou arquivo do Feishu/Lark. | Credenciais do app Feishu |
| `feishu_drive_list_comments` | Lista comentários de um arquivo Feishu/Lark inteiro, mais recentes primeiro. | Credenciais do app Feishu |
| `feishu_drive_list_comment_replies` | Lista respostas em uma thread de comentário específica do Feishu (documento inteiro ou seleção local). | Credenciais do app Feishu |
| `feishu_drive_reply_comment` | Publica uma resposta em uma thread de comentário do Feishu, com menção `@` opcional. | Credenciais do app Feishu |

## Toolset `file` {#file-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `patch` | Edições direcionadas de busca-e-substituição em arquivos. Use em vez de sed/awk no terminal. Usa correspondência fuzzy (9 estratégias) para que pequenas diferenças de espaçamento/indentação não a quebrem. Retorna um diff unificado. Executa verificações de sintaxe automaticamente após editar… | — |
| `read_file` | Lê um arquivo de texto com números de linha e paginação. Use em vez de cat/head/tail no terminal. Formato de saída: 'LINE_NUM\|CONTENT'. Sugere nomes de arquivo similares se não encontrado. Use offset e limit para arquivos grandes. NOTA: Não pode ler imagens o… | — |
| `search_files` | Busca conteúdo de arquivos ou encontra arquivos por nome. Use em vez de grep/rg/find/ls no terminal. Baseado em ripgrep, mais rápido que os equivalentes de shell. Busca de conteúdo (target='content'): busca por regex dentro dos arquivos. Modos de saída: correspondências completas com line… | — |
| `write_file` | Escreve conteúdo em um arquivo, substituindo completamente o conteúdo existente. Use em vez de echo/cat heredoc no terminal. Cria diretórios pais automaticamente. SOBRESCREVE o arquivo inteiro — use 'patch' para edições direcionadas. | — |

## Toolset `homeassistant` {#homeassistant-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `ha_call_service` | Chama um serviço do Home Assistant para controlar um dispositivo. Use ha_list_services para descobrir os serviços disponíveis e seus parâmetros para cada domínio. | — |
| `ha_get_state` | Obtém o estado detalhado de uma única entidade do Home Assistant, incluindo todos os atributos (brilho, cor, ponto de ajuste de temperatura, leituras de sensor, etc.). | — |
| `ha_list_entities` | Lista entidades do Home Assistant. Opcionalmente filtra por domínio (light, switch, climate, sensor, binary_sensor, cover, fan, etc.) ou por nome de área (sala, cozinha, quarto, etc.). | — |
| `ha_list_services` | Lista os serviços (ações) disponíveis no Home Assistant para controle de dispositivos. Mostra quais ações podem ser executadas em cada tipo de dispositivo e quais parâmetros aceitam. Use para descobrir como controlar dispositivos encontrados via ha_list_entities. | — |

## Toolset `computer_use` {#computer_use-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `computer_use` | Controle de desktop em segundo plano via cua-driver — screenshots (SOM / vision / AX), click / drag / scroll / type / key / wait, list_apps, focus_app. NÃO rouba o cursor ou o foco de teclado do usuário. Funciona com qualquer modelo capaz de usar ferramentas. macOS, Windows e Linux. | `cua-driver` no `$PATH` (instale via `hermes tools`). |


:::note
As **ferramentas Honcho** (`honcho_profile`, `honcho_search`, `honcho_context`, `honcho_reasoning`, `honcho_conclude`) não são mais embutidas. Elas estão disponíveis via o plugin de provedor de memória Honcho em `plugins/memory/honcho/`. Veja [Provedores de Memória](../user-guide/features/memory-providers.md) para instalação e uso.
:::

## Toolset `image_gen` {#image_gen-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `image_generate` | Gera imagens a partir de prompts de texto (texto-para-imagem) ou edita/transforma uma imagem existente (imagem-para-imagem) via o backend configurado pelo usuário (FAL.ai, OpenAI, autenticação OpenAI Codex, xAI, Krea). Passe `image_url` para editar uma imagem e `reference_image_urls` para referências de estilo; omita ambos para texto-para-imagem. O modelo é configurado pelo usuário e não é selecionável pelo agente. Retorna uma única URL de imagem ou caminho local. | FAL_KEY / OPENAI_API_KEY / OAuth do Codex / OAuth da xAI / KREA_API_KEY |

## Toolset `kanban` {#kanban-toolset}

Registrado quando o agente é (a) criado pelo dispatcher do kanban (env `HERMES_KANBAN_TASK` definido) ou (b) executando em um perfil que ativa explicitamente o toolset `kanban`. Workers com escopo de tarefa usam ferramentas de ciclo de vida para sua tarefa atribuída; perfis orquestradores também recebem ferramentas de roteamento de board como `kanban_list` e `kanban_unblock`. Veja [Kanban Multiagente](/user-guide/features/kanban) para o fluxo de trabalho completo.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `kanban_show` | Mostra a tarefa kanban ativa atribuída a este worker (título, descrição, comentários, dependências). | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_list` | Lista tarefas do board com filtros. Somente orquestrador; oculto para workers de tarefa criados pelo dispatcher. | perfil com toolset `kanban` |
| `kanban_complete` | Marca a tarefa atual como concluída com um payload de handoff estruturado (resultados, artefatos, próximos passos). | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_block` | Bloqueia a tarefa atual em uma pergunta para o usuário — o dispatcher pausa, exibe a pergunta e retoma quando um humano responde. | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_heartbeat` | Envia um heartbeat de progresso durante uma operação longa para que o dispatcher saiba que o worker ainda está ativo. | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_comment` | Adiciona um comentário à thread da tarefa sem alterar seu estado — útil para expor descobertas intermediárias. | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_create` | Gera tarefas filhas a partir da tarefa atual. Usado por orquestradores e workers que criam acompanhamentos. | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_link` | Vincula tarefas com uma aresta de dependência pai → filho. | `HERMES_KANBAN_TASK` ou toolset `kanban` |
| `kanban_unblock` | Move uma tarefa bloqueada para `ready` quando todos os pais estão concluídos, ou para `todo` enquanto algum pai ainda estiver aberto. Somente orquestrador; oculto para workers de tarefa criados pelo dispatcher. | perfil com toolset `kanban` |

## Toolset `project` {#project-toolset}

Ferramentas para operar [Projects](../user-guide/cli.md) de desktop — workspaces nomeados e multi-pasta. Registrado quando o toolset `project` está ativado (principalmente nas superfícies do app desktop / dashboard).

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `project_create` | Cria um Project de desktop (um workspace nomeado) e muda este chat para ele. Passe `path` para ancorá-lo a um repositório/pasta. | — |
| `project_list` | Lista os Projects de desktop e qual está ativo. | — |
| `project_switch` | Muda este chat para um Project existente (por nome, slug ou id); move o workspace da sessão para a pasta principal do projeto. | — |

## Toolset `memory` {#memory-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `memory` | Salva informações importantes em memória persistente que sobrevive entre sessões. Sua memória aparece no seu system prompt no início da sessão — é assim que você lembra coisas sobre o usuário e seu ambiente entre conversas. QUANDO SA… | — |

## Toolset `session_search` {#session_search-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `session_search` | Busca sessões passadas armazenadas no banco de sessões local, ou navega dentro de uma. Recuperação baseada em FTS5; retorna mensagens reais do banco (sem chamadas de LLM). Três formatos: descoberta (passe `query`), navegação (passe `session_id` + `around_message_id`), navegação livre (sem args). | — |

## Toolset `skills` {#skills-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `skill_manage` | Gerencia skills (criar, atualizar, excluir). Skills são sua memória procedural — abordagens reutilizáveis para tipos de tarefa recorrentes. Novas skills vão para ~/.hermes/skills/; skills existentes podem ser modificadas onde estiverem. Ações: create (SKILL.m… completo) | — |
| `skill_view` | Skills permitem carregar informações sobre tarefas e workflows específicos, além de scripts e templates. Carrega o conteúdo completo de uma skill ou acessa seus arquivos vinculados (referências, templates, scripts). A primeira chamada retorna o conteúdo do SKILL.md mais a… | — |
| `skills_list` | Lista skills disponíveis (nome + descrição). Use skill_view(name) para carregar o conteúdo completo. | — |

## Toolset `terminal` {#terminal-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `process` | Gerencia processos em segundo plano iniciados com terminal(background=true). Ações: 'list' (mostrar todos), 'poll' (verificar status + nova saída), 'log' (saída completa com paginação), 'wait' (bloquear até concluir ou expirar), 'kill' (terminar), 'write' (env… | — |
| `terminal` | Executa comandos shell em um ambiente Linux. O sistema de arquivos persiste entre chamadas. Defina `background=true` para servidores de longa duração. Defina `notify_on_complete=true` (com `background=true`) para receber uma notificação automática quando o processo terminar — sem necessidade de polling. NÃO use cat/head/tail — use read_file. NÃO use grep/rg/find — use search_files. | — |
| `read_terminal` | Lê o que está exibido atualmente no painel de terminal embutido da GUI desktop do Hermes (o shell embutido ao lado deste chat). Apenas no app desktop. | — |

## Toolset `todo` {#todo-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `todo` | Gerencia sua lista de tarefas para a sessão atual. Use para tarefas complexas com 3+ etapas ou quando o usuário fornece várias tarefas. Chame sem parâmetros para ler a lista atual. Escrita: - Forneça o array 'todos' para criar/atualizar itens - merge=… | — |

## Toolset `vision` {#vision-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `vision_analyze` | Analisa imagens usando IA de visão. Em modelos principais com capacidade de visão, retorna os pixels brutos da imagem como um resultado de ferramenta multimodal para que o modelo os veja nativamente na próxima rodada. Em modelos principais somente-texto, recorre a um modelo de visão auxiliar que descreve a imagem e retorna a descrição como texto. A assinatura da ferramenta é idêntica em ambos os casos. | — |

## Toolset `video` {#video-toolset}

Toolset opt-in (não carregado no conjunto padrão do `hermes-cli`). Adicione via `--toolsets video` ou inclua `video` na sua configuração `toolsets:`.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `video_analyze` | Analisa conteúdo de vídeo a partir de uma URL ou caminho de arquivo — legendas, divisão de cenas, timestamps-chave e descrições visuais. | — |

## Toolset `video_gen` {#video_gen-toolset}

Toolset opt-in (não carregado no conjunto padrão do `hermes-cli`). Adicione via `--toolsets video_gen` ou ative em `hermes tools` → Video Generation, que também guia você na escolha de um backend.

Os backends são distribuídos como plugins em `plugins/video_gen/<name>/`:

- **xAI Grok-Imagine** — texto-para-vídeo e imagem-para-vídeo (OAuth SuperGrok ou `XAI_API_KEY`).
- **FAL.ai** — Veo 3.1, Pixverse v6, Kling O3 (requer `FAL_KEY`).

A ferramenta única `video_generate` cobre ambas as modalidades — passe `image_url` para animar uma imagem estática, omita para gerar apenas a partir de texto. O backend ativo roteia automaticamente para o endpoint correto. A descrição da ferramenta é reconstruída no início da sessão para refletir as capacidades reais do backend ativo (modalidades, proporções, resoluções, faixa de duração, máximo de imagens de referência, suporte a áudio). Veja [Plugins de Provedor de Geração de Vídeo](/developer-guide/video-gen-provider-plugin) para criar backends.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `video_generate` | Gera um vídeo a partir de um prompt de texto (texto-para-vídeo) ou anima uma imagem estática (imagem-para-vídeo) usando o backend de geração de vídeo configurado pelo usuário. Passe `image_url` para animar aquela imagem; omita para gerar apenas a partir de texto. O backend roteia automaticamente para o endpoint correto. Retorna uma URL HTTP ou um caminho de arquivo absoluto no campo `video`. | Plugin `video_gen` ativo + sua credencial (ex.: `XAI_API_KEY`, `FAL_KEY`) |

## Toolset `web` {#web-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `web_search` | Busca informações na web. Retorna até 5 resultados por padrão com títulos, URLs e descrições. Aceita um `limit` opcional (1-100, padrão 5). A query é passada ao backend configurado, então operadores como `site:domain`, `filetype:pdf`, `intitle:word`, `-term` e `"frase exata"` podem funcionar quando o backend os suportar. | EXA_API_KEY ou PARALLEL_API_KEY ou FIRECRAWL_API_KEY ou TAVILY_API_KEY |
| `web_extract` | Extrai conteúdo de URLs de páginas web. Retorna o conteúdo da página em formato markdown. Também funciona com URLs de PDF — passe o link do PDF diretamente e ele converte para texto markdown. Páginas com menos de 5000 caracteres retornam markdown completo; páginas maiores são resumidas por LLM. | EXA_API_KEY ou PARALLEL_API_KEY ou FIRECRAWL_API_KEY ou TAVILY_API_KEY |

## Toolset `x_search` {#x_search-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `x_search` | Busca posts, perfis e threads do X (Twitter) usando a ferramenta Responses `x_search` embutida da xAI. Use para discussões, reações ou afirmações atuais no X, em vez de páginas web gerais. Desativado por padrão — ative via `hermes tools` → 🐦 X (Twitter) Search. O schema só é registrado quando credenciais da xAI estão configuradas (condicionado por check_fn). | XAI_API_KEY **ou** login OAuth do xAI Grok (SuperGrok / Premium+) |

## Toolset `tts` {#tts-toolset}

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `text_to_speech` | Converte texto em áudio de fala. Retorna um caminho MEDIA: que a plataforma entrega como mensagem de voz. No Telegram é reproduzido como uma bolha de voz, no Discord/WhatsApp como um anexo de áudio. No modo CLI, salva em ~/voice-memos/. Voz e provedor… | — |

## Toolset `discord` {#discord-toolset}

Registrado no toolset da plataforma `hermes-discord` (apenas gateway). Usa o mesmo token de bot que o adaptador de mensageria.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `discord` | Lê e participa de um servidor Discord. Ações incluem `search_members`, `fetch_messages`, `send_message`, `react`, `fetch_channel`, `list_channels`, entre outras. | `DISCORD_BOT_TOKEN` |

## Toolset `discord_admin` {#discord_admin-toolset}

Registrado no toolset da plataforma `hermes-discord`. Ações de moderação exigem que o bot tenha as permissões correspondentes do Discord.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `discord_admin` | Gerencia um servidor Discord via API REST: lista guilds/canais/cargos, cria/edita/exclui canais, gerencia concessões de cargo, timeouts, expulsões e banimentos. | `DISCORD_BOT_TOKEN` + permissões de bot |

## Toolset `spotify` {#spotify-toolset}

Registrado pelo plugin `spotify` incluído. Requer um token OAuth — execute `hermes auth spotify` uma vez para autorizar.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `spotify_playback` | Controla a reprodução do Spotify, inspeciona o estado de reprodução ativo ou busca as faixas tocadas recentemente. | OAuth do Spotify |
| `spotify_devices` | Lista dispositivos Spotify Connect ou transfere a reprodução para outro dispositivo. | OAuth do Spotify |
| `spotify_queue` | Inspeciona a fila do usuário no Spotify ou adiciona um item a ela. | OAuth do Spotify |
| `spotify_search` | Busca no catálogo do Spotify por faixas, álbuns, artistas, playlists, programas ou episódios. | OAuth do Spotify |
| `spotify_playlists` | Lista, inspeciona, cria, atualiza e modifica playlists do Spotify. | OAuth do Spotify |
| `spotify_albums` | Busca metadados de álbuns do Spotify ou as faixas de um álbum. | OAuth do Spotify |
| `spotify_library` | Lista, salva ou remove as faixas ou álbuns salvos do usuário no Spotify. | OAuth do Spotify |

## Toolset `hermes-yuanbao` {#hermes-yuanbao-toolset}

Registrado apenas no toolset da plataforma `hermes-yuanbao`. Yuanbao é o app de chat da Tencent; essas ferramentas operam suas APIs de DM/grupo/sticker.

| Ferramenta | Descrição | Requer ambiente |
|------|-------------|----------------------|
| `yb_query_group_info` | Consulta informações básicas sobre um grupo (chamado "派/Pai" no app): nome, proprietário, número de membros. | Credenciais do Yuanbao |
| `yb_query_group_members` | Consulta membros de um grupo (para menções `@`, encontrar um usuário pelo nome, listar bots). | Credenciais do Yuanbao |
| `yb_send_dm` | Envia uma mensagem privada/direta a um usuário em um grupo, com arquivos de mídia opcionais. | Credenciais do Yuanbao |
| `yb_search_sticker` | Busca no catálogo de stickers embutido do Yuanbao (TIM face) por palavra-chave. | Credenciais do Yuanbao |
| `yb_send_sticker` | Envia um sticker embutido no chat atual do Yuanbao. | Credenciais do Yuanbao |

