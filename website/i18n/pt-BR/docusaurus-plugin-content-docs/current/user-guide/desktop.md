---
sidebar_position: 3
title: "App Desktop"
description: "O app desktop nativo do Hermes — uma experiência polida para conversar com o Hermes, com saída de ferramentas em streaming, previews lado a lado, navegador de arquivos, voz, cron, profiles, skills e configurações. macOS, Windows e Linux."
---

# App Desktop

O app desktop do Hermes é um aplicativo nativo construído em torno do **mesmo** agente que você obtém do CLI e do gateway — mesma config, mesmas API keys, mesmas sessões, mesmas skills, mesma memória. Não é um produto separado nem um clone leve; usa o mesmo core do Hermes Agent e as mesmas configurações, e o dirige por uma UI moderna e cuidadosamente projetada. Se você já usou `hermes` no terminal, tudo que configurou lá já está aqui, e qualquer coisa que fizer aqui aparece lá.

Roda em **macOS, Windows e Linux**.

:::tip Qual interface é qual?
O Hermes tem várias front ends que conversam com o mesmo agente:

- **App Desktop** (esta página) — um aplicativo nativo com UI feita sob medida para chat, configuração e gerenciamento.
- **CLI** (`hermes`) e **[TUI](./tui.md)** (`hermes --tui`) — interfaces de terminal.
- **[Web Dashboard](./features/web-dashboard.md)** (`hermes dashboard`) — um painel admin no browser; sua aba **Chat** opcional embute a TUI por pseudo-terminal.

Escolha o que couber no momento. Eles compartilham estado, então você pode iniciar uma sessão em um e retomá-la em outro.
:::

## Instalação {#install}

Siga as [instruções de instalação do Hermes Desktop](../getting-started/installation.md).

Se você já tem o Hermes instalado, basta rodar

```bash
hermes desktop
```

Isso usa sua config, chaves, sessões e skills atuais.

## O que há no app {#whats-in-the-app}

O app desktop é organizado como uma janela chat-first com sidebar esquerda para navegação. Foi feito para gerenciar múltiplas conversas de agente simultâneas, configurar providers de messaging, criar artifacts, navegar estruturas de pastas de projetos e trabalhar em vários projetos ao mesmo tempo.

### Chat {#chat}

O centro do app. Você obtém:

- **Respostas em streaming** com atividade de ferramentas ao vivo e resumos estruturados de tool-call enquanto o agente trabalha.
- **O mesmo histórico de conversa** de toda outra superfície Hermes — sessões iniciadas aqui retomam no CLI/TUI e vice-versa.
- **Arrastar e soltar arquivos** em qualquer lugar da área de chat para anexá-los à próxima mensagem.
- **Um rail de preview à direita** — renderize páginas web, arquivos e saídas de ferramentas lado a lado enquanto continua conversando.
- **Histórico do composer e edição de fila** — pressione as setas up/down em um composer vazio para recuperar e reutilizar prompts anteriores, e edite mensagens que colocou na fila antes de serem enviadas. Pressionar Stop (ou Esc) enquanto turnos estão na fila pausa a fila e a expande acima do composer; retome dali, ou envie, edite e exclua entradas individuais.
- **Um rail de timeline de conversa** — chats longos ganham um rail fino de marcadores ao longo da borda da transcrição, um por prompt. Passe o mouse para abrir a lista de prompts, clique em um para ir direto àquele ponto da conversa. (Aparece quando o chat tem alguns turnos.)
- **Buscar na página** — pressione **Cmd/Ctrl+F** para abrir uma barra de busca que pesquisa a transcrição renderizada do chat. Enter / Shift+Enter (ou Cmd/Ctrl+G / Cmd/Ctrl+Shift+G com a barra aberta) percorrem matches; Esc fecha.

#### Barra de status {#status-bar}

A barra ao longo da base do chat mostra estado ao vivo da sessão e expõe controles rápidos sem abrir Settings:

- **Toggle YOLO por sessão** — ligue ou desligue YOLO só nesta sessão (igual à TUI). YOLO contorna os prompts de aprovação de comandos perigosos, então saiba o que está desligando — veja [Security → YOLO Mode](./security.md#yolo-mode).
- **Medidor de uso de contexto** — um medidor ao vivo "% cheio" da janela de contexto da sessão. Clique para abrir o popover **Context Usage** com breakdown de tokens por categoria (system prompt, definições de ferramentas, skills, memória, rules, MCP, definições de subagent e a própria conversa) para ver exatamente o que está consumindo a janela antes da compressão entrar em ação.
- **Itens customizáveis** — clique com botão direito na barra de status (**Show in status bar**) para escolher o que aparece: medidor de contexto, workspace, modelo, aprovações, timers de turno/sessão, terminal, Command Center, versão do backend e mais — ou oculte a barra inteira (**Cmd/Ctrl+Shift+S** alterna).

Conversando contra uma instância Hermes em outra máquina em vez do backend local bundled? Veja [Conectando a um backend remoto](#connecting-to-a-remote-backend) abaixo — e para o quadro completo de como funciona a conexão do dashboard hospedado remotamente (o auth gate, o socket de chat `/api/ws` e triagem de close codes WebSocket), veja [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

#### Descoberta de repositórios {#repository-discovery}

O Hermes Desktop descobre repositórios Git locais para a sidebar Projects escaneando seu diretório home até uma profundidade limitada. Você pode mudar isso por profile em **Settings → Workspace**, ou em `config.yaml`:

```yaml
desktop:
  repo_scan_enabled: true
  repo_scan_roots: []
  repo_scan_exclude_paths: []
```

- Defina `repo_scan_enabled: false` para parar o scan de filesystem completamente. Linhas de cache de disk-discovery existentes para aquele profile são limpas; projetos explícitos e repositórios inferidos de sessões Hermes intencionais permanecem disponíveis.
- Defina `repo_scan_roots` como lista de pastas para restringir o scan. Lista vazia preserva o scan padrão do diretório home.
- Defina `repo_scan_exclude_paths` para pastas cujas subtrees completas devem ser puladas.

Mudar qualquer um desses valores invalida apenas o cache de disk-discovery daquele profile e inicia um refresh conforme a política. **Hide from sidebar** permanece uma ação de curadoria separada por item.

#### Escolhendo um modelo {#choosing-a-model}

O model picker fica no **composer**, à esquerda do microfone. Clique para trocar modelo, reasoning effort e fast mode de um dropdown.

- **O picker do composer é estado sticky de UI e nunca toca seu default.** É lembrado localmente (por dispositivo) e **segue** entre chats novos e restarts em vez de voltar ao default — escolha um modelo uma vez e o próximo `Cmd/Ctrl+N` abre nele. Com chat ao vivo, trocar modelos limita a mudança ao **chat atual**; de qualquer forma a seleção acompanha quando a sessão é criada/trocada e **nunca** é gravada no default do profile. (Trocar [profiles](#sessions--profiles) reseed para o default daquele profile.)
- **Defina o default em Settings → Model.** Esse modelo "main" é seu **default global por profile** — é de onde chats novos, crons, subagentes e tarefas auxiliares partem, e é o único lugar que o grava. Cada [profile](#sessions--profiles) mantém seu próprio default.
- **Presets effort/fast por modelo.** Cada modelo lembra sua própria escolha de reasoning effort e fast mode no app desktop, reaplicada à sessão sempre que você escolhe aquele modelo. Esses presets são conveniência desktop e não mudam crons ou subagentes.
- **Trocas no meio do chat resetam o prompt cache.** Trocar o modelo dentro de um chat ao vivo significa que a próxima mensagem relê toda a conversa a preço cheio de input (prompt caches de provider são keyed ao modelo). Ok ocasionalmente; num chat longo, um chat novo no modelo novo costuma ser mais barato que ficar alternando.

### Navegador de arquivos {#file-browser}

Explore e faça preview do working directory sem sair do app — útil para acompanhar enquanto o agente lê, grava e edita arquivos. Defina o diretório inicial do projeto com `hermes desktop --cwd <path>` (ou a environment variable `HERMES_DESKTOP_CWD`).

### Artifacts {#artifacts}

A view **Artifacts** reúne o que suas sessões geram — **imagens, arquivos e links** — numa galeria pesquisável e navegável. Abra pela sidebar, pela command palette (**Artifacts — Browse generated outputs**) ou por um atalho `nav.artifacts` que você mesmo bindar. Indexa saídas recentes de sessão automaticamente; todo artifact mostra qual sessão o produziu com salto de volta àquele chat, e imagens e arquivos abrem em preview com ações de download / open-in-browser / copy.

### Janelas, abas e painéis {#windows-tabs--panes}

O app foi feito para trabalhar em várias coisas ao mesmo tempo:

- **Abas** — **Cmd/Ctrl+T** abre uma aba de sessão nova; **Ctrl+Tab** / **Ctrl+Shift+Tab** alternam sessões, e **Ctrl+1…9** saltam para uma sessão recente por posição. **Cmd/Ctrl+W** fecha a aba focada e **Cmd/Ctrl+Shift+T** reabre a última fechada.
- **Múltiplas janelas** — **Cmd/Ctrl+Shift+N** abre uma janela nova, e qualquer sessão pode ser popped out pelo menu de contexto (**New window**) ou pela command palette. Uma janela popped out renderiza aquele chat único sem a sidebar global — útil para estacionar uma sessão longa em outro monitor. Saída ao vivo do agente faz stream em toda janela mostrando a sessão.
- **Painéis** — **Cmd/Ctrl+B** alterna a sidebar esquerda, **Cmd/Ctrl+J** a direita, e **Cmd/Ctrl+\\** troca de qual lado as sidebars ficam.

### Terminal {#terminal}

Um terminal real fica na sidebar direita, ao lado do file browser:

- **Ctrl+`** mostra o terminal (abrindo um se não existir); **Ctrl+Shift+`** cria outro. Múltiplos terminais empilham num rail de abas — **Ctrl+Shift+↓/↑** percorrem entre eles, **Ctrl+Shift+W** fecha o ativo.
- **Shells persistem enquanto ocultos.** Fechar ou ocultar o painel não mata seu shell — todo terminal aberto permanece montado com scrollback e processos rodando intactos até você fechá-lo explicitamente.
- **Add to chat** — selecione saída do terminal e envie ao composer como contexto da próxima mensagem.

### Git review e worktrees {#git-review--worktrees}

Para sessões rodando dentro de um repositório Git, o app tem uma superfície de source-control built-in:

- **Painel Review** — **Cmd/Ctrl+G** alterna o painel de review da working tree: status de branch e ahead/behind, arquivos alterados (list ou tree view) e diffs escopados a **Uncommitted**, **Branch** ou **Last turn** (só o que o agente mudou no turno mais recente). Stage/unstage arquivos, reverta mudanças, escreva mensagem de commit (ou **Generate commit message**), depois **Commit** ou **Commit & Push** — e **Create PR** via GitHub CLI (`gh`), ou entregue tudo ao agente com **Ask Hermes to open PR**. Você também pode criar e trocar branches daqui.
- **Worktrees** — **Cmd/Ctrl+Shift+B** (ou **New worktree** num projeto na sidebar) cria um Git worktree num branch novo para o agente trabalhar numa cópia paralela do repo sem tocar seu checkout. Worktrees aparecem como lanes próprias sob o projeto; remover uma oferece deletar o diretório do worktree (o branch permanece) ou só ocultar a lane e deixá-la no disco, com opção force quando há mudanças uncommitted.

### Memory Graph {#memory-graph}

O **Memory Graph** (command palette → *Memory Graph*, ou item da status bar) é um mapa interativo do que o Hermes aprendeu para você — skills e memórias dispostas como grafo de nós zoomável com timeline, filtrável por **All / Used / Learned**. Um controle de share exporta o layout do mapa como código compacto que você pode colar para alguém (só layout — nenhum texto de memória ou skill seu é incluído) e importa códigos da mesma forma.

### Quick Entry {#quick-entry}

Quick Entry é um composer pequeno sempre disponível invocado por um **hotkey global de qualquer lugar do sistema** — dispare um prompt sem trocar para (ou nem abrir) a janela principal. Habilite em **Settings → Advanced → Quick Entry**; o atalho padrão é **Ctrl/Cmd+Shift+Space** e você pode definir o seu (precisa de pelo menos um modificador). Se outro app já possui o chord, a linha de settings avisa para você escolher outro.

### Voz {#voice}

Converse com o Hermes e ouça de volta, o mesmo [voice mode](./features/voice-mode.md) disponível em outros lugares. No macOS o SO pedirá acesso ao microfone uma vez.

### Settings e onboarding {#settings--onboarding}

Gerencie providers, modelos, ferramentas e credenciais numa UI real em vez de editar YAML. O onboarding de primeira execução leva você à primeira mensagem em segundos. Os painéis de settings cobrem providers/keys, seleção de modelo, configuração de toolset, servidores MCP, gateway e gerenciamento de sessão.

- **Painel Providers settings** — lugar dedicado para gerenciar inference providers, com UX Accounts / API-keys para login e armazenamento de credenciais por provider.
- **Todo provider e modelo nos menus** — a GUI expõe a lista completa de providers e todo modelo que `hermes model` conhece, para você escolher do mesmo catálogo que o CLI vê em vez de um subconjunto curado.
- **xAI Grok OAuth** — Grok é provider OAuth de primeira classe no launcher; faça login pelo fluxo de browser como os outros providers OAuth.
- **Instalações de tool-backend pela GUI** — rode passos post-setup de install de um tool backend direto do app em vez de ir ao terminal.
- **Terminal font picker** — escolha uma fonte instalada em **Settings → Appearance**. Nerd Fonts como `MesloLGS NF` renderizam separadores e ícones Powerlevel10k em terminais interativos e de agente; a configuração é salva por profile.
- **Aviso de modelo auxiliar** — se trocar o modelo principal para um provider novo enquanto tarefas auxiliares (titling, summarization e helpers similares) ainda estão pinned a outro provider, o app avisa para você não dividir trabalho entre dois providers sem saber.
- **Temas VS Code Marketplace** — além dos presets de tema built-in, as appearance settings incluem busca ao vivo no VS Code Marketplace: escolha qualquer color theme e o app baixa, converte e instala como tema desktop. O mesmo importer está disponível na command palette (*Install theme*), e temas importados podem ser removidos de novo nas appearance settings.
- **Keep computer awake** — **Settings → Advanced → Keep computer awake** impede a máquina de dormir para execuções longas ou overnight do agente continuarem (o display ainda pode escurecer). Configuração por computador.

O onboarding de primeira execução foi redesenhado num design system de overlay unificado, e você pode escolher **Choose provider later** para pular setup de provider e entrar no app primeiro.

### Painéis de gerenciamento {#management-panes}

O app também expõe a superfície mais ampla de gerenciamento Hermes para você não precisar ir ao terminal:

- **Skills** — navegue, instale e gerencie [skills](./features/skills.md).
- **Memory graph (Star Map)** — digite `/journey` (aliases `/learning`, `/memory-graph`) no chat para abrir uma constelação interativa de skills e memórias aprendidas ao longo do tempo, com playback scrubber. Nós podem ser editados ou excluídos direto do painel (skills são arquivadas, memórias removidas). Veja [Learning Journey](./features/memory.md#learning-journey-journey).
- **Cron** — veja e gerencie [scheduled jobs](../reference/cli-commands.md#hermes-cron).
- **Profiles** — alterne entre [profiles Hermes](./profiles.md) (config/skills/sessões isoladas).
- **Messaging** — configure canais do gateway.
- **Agents** e **Command Center** — superfícies de orquestração para trabalho multi-agente.

### Teclado e navegação {#keyboard--navigation}

- **Command palette** — pressione **Cmd+K** ou **Cmd+P** (Ctrl+K / Ctrl+P no Windows/Linux) para saltar a ações e navegar o app pelo teclado: abrir qualquer página ou seção de settings, saltar a uma sessão por título ou id, trocar model/theme/color mode, spawnar terminal, reiniciar gateway, atualizar Hermes e mais.
- **Atalhos rebindáveis** — **Settings → Keyboard Shortcuts** (ou **Cmd/Ctrl+/**) abre o painel de shortcuts onde você pode remapear quase todo binding — troca de profile, navegação de sessão, toggles de view e quaisquer shortcuts de desktop plugins. Atribuições duplicadas são sinalizadas como conflitos. Alguns defaults que valem saber: **Cmd/Ctrl+N** nova sessão, **Cmd/Ctrl+.** Command Center, **Cmd/Ctrl+,** Settings, **Cmd/Ctrl+Shift+F** buscar sessões, **Cmd/Ctrl+1–9** trocar profiles, **Shift+X** alternar light/dark.
- **Atalhos de zoom customizados** — zoom da interface em incrementos de meio passo para controle mais fino do tamanho do texto.
- **Seletor de idioma da UI** — mude o idioma da interface do app in-app, incluindo Chinês Simplificado (zh-Hans).

### Sessões e profiles {#sessions--profiles}

- **Overhaul da lista de sessões** — lista de sessões refeita com archiving e higiene geral de sessão para manter a lista gerenciável conforme cresce.
- **Buscar sessões por id** — encontre uma sessão específica diretamente pelo id.
- **Sessões multi-profile concorrentes** — rode sessões em múltiplos [profiles](./profiles.md) ao mesmo tempo, e referencie uma sessão em outro profile com links cross-profile `@session`.

## Atualização {#updating}

O app verifica atualizações em background e oferece update de um clique quando uma está pronta.

O [processo de atualização manual](https://hermes-agent.nousresearch.com/docs/getting-started/updating) também funciona com a GUI.

## Desinstalação {#uninstalling}

Abra **Settings → About → Danger zone** e escolha quanto remover:

- **Uninstall Chat GUI only** — remove o app desktop e seus dados; o agente Hermes, sua config e seus chats permanecem. (Igual a `hermes uninstall --gui`.)
- **Uninstall GUI + agent, keep my data** — remove o app e o agente mas mantém config, chats e secrets para reinstalação futura. (Igual a `hermes uninstall`.)
- **Uninstall everything** — remove app, agente e todos os dados do usuário. (Igual a `hermes uninstall --full`.)

O app fecha para terminar o trabalho (a limpeza roda depois que sai para poder remover o app bundle em execução e seu próprio venv). As opções que removem o agente ficam ocultas automaticamente quando nenhum agente local está instalado (por exemplo, cliente "lite" só GUI conectado a backend remoto).

Você pode fazer o mesmo pelo terminal — `hermes uninstall --gui` só para a GUI, ou `hermes uninstall` / `hermes uninstall --full` para o agente também.

:::note
Rodar `hermes uninstall --gui` de um **source checkout** (build dev de `hermes desktop`) também remove `node_modules` do workspace e build output `apps/desktop/{dist,release}`, já que são artifacts de build da GUI. Recuperáveis com `hermes desktop` (ou `npm install` + rebuild) — mas se está hackeando ativamente no app desktop, espere reinstalar dependências depois.
:::

## Referência CLI: `hermes desktop` {#cli-reference-hermes-desktop}

Para lançar via CLI, basta rodar `hermes desktop`. Por padrão instala dependências Node do workspace, constrói o app Electron unpacked do OS atual e lança aquele artifact empacotado.

| Flag                 | Descrição                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build`       | Pula npm install/package e lança o app unpacked existente de `apps/desktop/release` |
| `--force-build`      | Força rebuild completo mesmo se o content stamp bater                                    |
| `--build-only`       | Constrói o app desktop mas não lança (usado por `hermes update`)                      |
| `--source`           | Lança via `electron .` contra `apps/desktop/dist` em vez do app empacotado           |
| `--cwd PATH`         | Diretório inicial do projeto para sessões de chat desktop (define `HERMES_DESKTOP_CWD`)           |
| `--hermes-root PATH` | Sobrescreve a raiz source Hermes que o app usa (define `HERMES_DESKTOP_HERMES_ROOT`)          |
| `--ignore-existing`  | Força o app a ignorar qualquer CLI `hermes` já no `PATH` durante resolução de backend      |
| `--fake-boot`        | Habilita delays de boot determinísticos para validar a UI de startup                            |

## Como funciona {#how-it-works}

O app empacotado inclui o shell Electron e uma superfície de chat React nativa. No primeiro launch pode instalar o runtime Hermes Agent em `HERMES_HOME` (`~/.hermes`, ou `%LOCALAPPDATA%\hermes` no Windows) — **o mesmo layout que uma instalação CLI usa**, por isso os dois são intercambiáveis. A resolução de backend primeiro honra `HERMES_DESKTOP_HERMES_ROOT`, depois uma instalação managed completa, depois um `hermes` sondado no `PATH` (a menos que `--ignore-existing` / `HERMES_DESKTOP_IGNORE_EXISTING=1` esteja definido), e por fim um override explícito de comando `HERMES_DESKTOP_HERMES` para packagers como Nix. O renderer React conversa com um backend headless que o app lança para você — um processo `hermes serve` que serve a API JSON-RPC/WebSocket do `tui_gateway` — e reutiliza o runtime do agente em vez de embutir `hermes --tui`. O app desktop é **self-contained**: roda seu próprio backend `hermes serve` e nunca abre nem exige o [web dashboard](./features/web-dashboard.md). (Runtimes mais antigos que o comando `serve` caem automaticamente para `dashboard --no-open` headless, então um update do app nunca ultrapassa seu backend.) Lógica de install, resolução de backend e self-update vivem no processo main do Electron.

## Conectando a um backend remoto {#connecting-to-a-remote-backend}

Por padrão o app inicia e gerencia seu próprio backend **local**. Você pode apontá-lo a um backend Hermes rodando em outra máquina — VPS, home server ou Mini atrás do Tailscale.

**Settings → Gateway → Connection mode** oferece as alternativas ao gateway local:

- **Remote gateway** — insira a URL de um backend `hermes serve` que você roda e faça login. Este é o modo que o resto desta seção percorre.
- **Hermes Cloud** — faça login uma vez no Hermes Cloud e escolha entre os agentes da sua conta; sem URL para colar. O app descobre seus agentes (com organization picker se sua conta abrange várias orgs), e conectar a um troca a sessão automaticamente. A status bar mostra a conexão cloud enquanto ativa.

Modos de conexão são configurados **por profile** — um override por profile pode apontar um profile a backend remoto ou cloud enquanto outros ficam locais (**Use default gateway** remove um override).

:::info O backend remoto é um processo `hermes serve` rodando
"Remote backend" significa um servidor **`hermes serve`** rodando na máquina remota — esse é o processo ao qual o app desktop conecta. Nada nesta seção funciona a menos que aquele backend esteja de fato up e alcançável. O app desktop não o inicia para você; você (ou um serviço `systemd`) mantém `hermes serve` rodando no host remoto, e o app se anexa a ele. Se também usa canais de messaging (Telegram, Discord, etc.), o **gateway** é um processo long-running *separado* que você inicia independentemente — veja a nota após os passos de setup.
:::

A conexão tem duas metades: no backend você a protege com um **auth provider**, e no app você insere a URL do backend e faz login. Bindar o backend a um endereço non-loopback engaja automaticamente seu auth gate, e o provider que você configura é o que deixa o app desktop passar.

**Escolha um provider com base em onde o backend vive:**

- **OAuth (Nous Portal) — preferido para qualquer coisa alcançável além da sua própria máquina.** Logins são verificados contra sua conta Nous, então esta é a opção adequada para VPS, host público ou qualquer backend remoto. Registre o dashboard com `hermes dashboard register` (ou a página Portal [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards)) para provisionar seu OAuth client, depois faça login do app com **Sign in with Nous Research**. Um provider OIDC self-hosted funciona igual se você roda seu próprio identity provider.
- **Username/password — uso local / trusted-network apenas.** A opção mais simples quando o backend está na mesma LAN confiável ou alcançável só por VPN (ex.: Tailscale). Protege uma credencial compartilhada única sem identity provider externo, então **não use para dashboard exposto à internet pública** — use OAuth lá.

O resto desta seção mostra o caminho username/password porque é o mais rápido de subir numa rede confiável; para o caminho OAuth veja [Web Dashboard → Default provider: Nous Research](./features/web-dashboard.md#default-provider-nous-research).

### No backend (a máquina remota) {#on-the-backend-the-remote-machine}

Defina username e password, depois inicie o backend bindado a um endereço alcançável. As credenciais vivem em `~/.hermes/.env` (arquivo de secrets, mode 0600):

```bash
# 1. Set the dashboard login credentials.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
# Without it a random key is generated per boot and you'll be logged out
# on every restart.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the backend bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes serve --host 0.0.0.0 --port 9119
```

Mantenha aquele processo `hermes serve` rodando enquanto quiser que o app desktop consiga conectar — se parar, o app não alcança mais o backend. Rode sob `systemd`, `tmux` ou seu process manager de preferência para sobreviver a logout e reboots.

Separadamente, garanta que o **gateway está rodando** no host remoto se depende de canais de messaging — o backend `hermes serve` é com o que o app desktop conversa, mas suas sessões de gateway Telegram/Discord/Slack são outro processo que você inicia e mantém por conta própria. Veja [Messaging](./messaging/index.md) para setup do gateway.

Prefere não manter password plaintext at rest? Defina `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` como hash scrypt — calcule com `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Superfície de configuração completa (chaves config.yaml, toda env var, rate limiter): [Web Dashboard → Username/password provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

Rodando o backend como serviço systemd? Dê à unit `EnvironmentFile=%h/.hermes/.env` para as credenciais estarem no environment no boot.

:::warning
O backend lê e grava seu `.env` (API keys, secrets) e pode rodar comandos de agente. O setup **username/password** mostrado acima é para rede confiável — nunca exponha um backend protegido por password diretamente à internet aberta; coloque atrás de VPN. [Tailscale](https://tailscale.com/) é a opção limpa: bind ao tailscale IP da máquina (`--host <tailscale-ip>`) e use `http://<tailscale-ip>:9119` como Remote URL para só sua tailnet alcançar. Para alcançar um backend pela internet pública, use o provider **OAuth (Nous Portal)**.
:::

### No app {#in-the-app}

**Settings → Gateway → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (prefixos de path como `/hermes` funcionam se você frontar com reverse proxy)
2. **Sign in** — o app detecta qual provider o backend anuncia e adapta o botão. Para backend username/password mostra botão **Sign in** que abre formulário de credenciais (insira as credenciais do passo 1). Para backend OAuth mostra **Sign in with `<provider>`** (ex.: *Sign in with Nous Research*), que roda o sign-in no browser do provider. De qualquer forma o app termina com sessão autenticada contra o backend.
3. **Save and reconnect** — troca o shell desktop para o backend remoto. A sessão refresca automaticamente; você permanece logado entre restarts quando `HERMES_DASHBOARD_BASIC_AUTH_SECRET` está definido.

Você também pode definir a URL do backend sem a UI via environment variable `HERMES_DESKTOP_REMOTE_URL` antes de lançar o app (sobrescreve a config in-app); ainda faz login no painel Gateway settings.

:::note Hosts remotos por profile
O host do gateway remoto é configurado por [profile](./profiles.md), então cada profile pode apontar ao seu próprio backend remoto (ou ficar no local). Trocar profiles troca a qual host remoto o app conecta.
:::

### Solução de problemas {#troubleshooting}

- **Sign-in falha com 401 / "Invalid credentials"** — username ou password não batem com `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` do backend. O backend retorna o mesmo erro genérico para usuário desconhecido e password errado (sem oracle de enumeração), então confira ambos. Confirme que o gate está on com `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` — deve reportar `true` e incluir `"basic"`.
- **Sem botão "Sign in" — pede session token** — o provider username/password do backend não está ativo. `/api/status` não listará `"basic"` em `auth_providers`. Garanta que username e password (ou password hash) estão definidos em `~/.hermes/.env` e que o processo dashboard de fato os carregou.
- **Deslogado a cada restart** — defina `HERMES_DASHBOARD_BASIC_AUTH_SECRET` com valor estável. Sem ele a chave de assinatura de token é regenerada por boot, invalidando todas as sessões.
- **Connection refused / times out** — backend bindado a `127.0.0.1` (padrão) ou firewall/VPN bloqueando a porta. Bind a `0.0.0.0` ou tailscale IP e abra a porta para sua rede confiável.

Para o mesmo setup pelo ângulo do web dashboard, veja [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend); as env vars estão catalogadas em [Environment Variables → Web Dashboard & Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop).

## Estendendo o app desktop {#extending-the-desktop-app}

O app desktop é contribution-driven — painéis, páginas, nav da sidebar,
itens da status bar, comandos da palette, keybinds e temas registram por um SDK, e
você pode adicionar os seus. Um plugin é um único arquivo ESM dropado em
`$HERMES_HOME/desktop-plugins/<id>/plugin.js`; o app carrega em segundos e
hot-reloads a cada save. Gerencie plugins instalados ao vivo em **Settings → Plugins**.

Veja [Desktop Plugin SDK](../developer-guide/desktop-plugin-sdk.md) para a referência
completa. (Isso é separado do [sistema de plugins do web dashboard](./features/extending-the-dashboard.md).)

## Solução de problemas {#troubleshooting-1}

Boot logs caem em `HERMES_HOME/logs/desktop.log` (inclui saída do backend e tracebacks Python recentes) — confira primeiro se o app reporta boot failure. Você também pode tail pelo CLI:

```bash
hermes logs gui -f
```

Resets comuns:

```bash
# Force a clean first-launch setup (macOS/Linux)
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# Rebuild a broken Python venv (macOS/Linux)
rm -rf "$HOME/.hermes/hermes-agent/venv"

# Reset a stuck macOS microphone prompt
tccutil reset Microphone com.nousresearch.hermes
```

### "Build desktop app" travado no download do Electron {#build-desktop-app-stuck-on-electron-download}

O build baixa o runtime Electron (~114&nbsp;MB) de `github.com/electron/electron/releases`. Se o instalador trava no passo **Build desktop app** com a saída ao vivo repetindo `retrying attempt=…`, GitHub está sendo bloqueado ou throttled na sua rede (firewall, proxy ou região).

O instalador se auto-cura automaticamente: num build falho (1) limpa um zip Electron cached corrupto e retenta, depois (2) se ainda falhar e você não definiu `ELECTRON_MIRROR`, retenta mais uma vez via `npmmirror.com`, o mirror comunitário de facto do Electron. `@electron/get` SHASUM-checks o download, mas os checksums vêm do mesmo mirror — isso pega download corrupto ou parcial, não mirror comprometido. Se preferir não confiar em host third-party, fixe seu próprio `ELECTRON_MIRROR` (abaixo); o build nunca sobrescreve um que você definiu.

Para **escolher seu próprio mirror** (ex.: corporativo/confiável), defina `ELECTRON_MIRROR` antes de instalar ou rebuild manualmente — o build honra e não sobrescreve:

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

Para limpar um zip cached corrupto manualmente:

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
```

## Build a partir do source {#building-from-source}

Se quer hackear no app em si, instale deps do workspace da raiz do repo uma vez, depois rode o dev server de `apps/desktop`:

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

Aponte o app a um checkout específico, ou sandbox do sua config real:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

Build installers:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

Signing e notarization macOS/Windows rodam automaticamente quando as credenciais relevantes estão presentes no environment (`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*` para macOS, `WIN_CSC_*` para Windows).

### Permissões macOS e rebuilds locais (TCC) {#macos-permissions-and-local-rebuilds-tcc}

O macOS lembra grants de permissão (Full Disk Access, Desktop/Downloads/Documents,
Accessibility, Automation, microfone) contra a *identidade de code-signing* do app,
não seu path. Apps built localmente e self-updated são assinados com assinatura ad-hoc
pinned por identificador estável, então grants persistem entre updates out of the
box.

Para a garantia mais forte — identidade ancorada em certificado, o mesmo
mecanismo em que usuários yabai/skhd confiam — crie um certificado de code-signing
self-signed uma vez e diga ao Hermes para usá-lo:

1. Keychain Access → Certificate Assistant → **Create a Certificate…**
2. Name: `Hermes Local Signing`, Identity Type: *Self-Signed Root*,
   Certificate Type: **Code Signing**.
3. `hermes config set desktop.macos_signing_identity "Hermes Local Signing"`

O próximo update re-assina o app rebuilt com aquele certificado; todo grant TCC
sobrevive. Nenhuma conta Apple Developer é necessária. Builds de release notarized são
detectados e nunca re-assinados.

Nota one-time: mudar a identidade de signing (incluindo o primeiro update após
este fix) muda a identidade do app uma vez, então o macOS re-promptará uma última
vez. Grants ficam estáveis daí em diante. Se uma permissão ficar stuck, reset com
`tccutil reset All com.nousresearch.hermes` e re-grant.

## Veja também {#see-also}

- [CLI Guide](./cli.md) — a interface de terminal
- [TUI](./tui.md) — a UI de terminal moderna usada por `hermes --tui` e a aba chat do dashboard
- [Web Dashboard](./features/web-dashboard.md) — painel admin no browser com aba chat embutida
- [Configuration](./configuration.md) — config que o app desktop lê e grava
- [Windows (Native)](./windows-native.md) — caminho de instalação nativa Windows
