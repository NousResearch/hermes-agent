---
sidebar_position: 3
title: "App Desktop"
description: "O app desktop nativo do Hermes — uma experiência refinada para conversar com o Hermes, com saída de ferramentas em streaming, pré-visualizações lado a lado, navegador de arquivos, voz, cron, perfis, skills e configurações. macOS, Windows e Linux."
---

# App Desktop

O app desktop do Hermes é um aplicativo nativo construído em torno do **mesmo** agente que você obtém no CLI e no gateway — mesma config, mesmas chaves de API, mesmas sessões, mesmas skills, mesma memória. Não é um produto separado nem um clone simplificado; ele usa o mesmo núcleo do Hermes Agent e as mesmas configurações, e o conduz por uma UI moderna e bem pensada. Se você já usou `hermes` no terminal, tudo o que configurou lá já está aqui, e qualquer coisa que fizer aqui aparece lá também.

Ele roda em **macOS, Windows e Linux**.

:::tip Qual interface é qual?
O Hermes tem várias interfaces que conversam com o mesmo agente:

- **App Desktop** (esta página) — um aplicativo nativo com UI dedicada para chat, configuração e gerenciamento.
- **CLI** (`hermes`) e **[TUI](./tui.md)** (`hermes --tui`) — interfaces de terminal.
- **[Web Dashboard](./features/web-dashboard.md)** (`hermes dashboard`) — um painel de administração no navegador; a aba **Chat** opcional incorpora o TUI por meio de um pseudo-terminal.

Escolha a que fizer sentido no momento. Elas compartilham estado, então você pode iniciar uma sessão em uma e retomá-la em outra.
:::

## Instalação

Siga as [instruções de instalação do Hermes Desktop](../getting-started/installation.md).

Se você já tem o Hermes instalado, basta executar

```bash
hermes desktop
```

Isso usa sua config, chaves, sessões e skills atuais.

## O que há no app

O app desktop é organizado como uma janela centrada no chat, com uma barra lateral esquerda para navegação. Ele foi feito para permitir gerenciar várias conversas com o agente ao mesmo tempo, configurar provedores de mensagens, criar artefatos, explorar a estrutura de pastas dos projetos e trabalhar em vários projetos de uma vez.

### Chat

O centro do app. Você tem:

- **Respostas em streaming** com atividade de ferramentas ao vivo e resumos estruturados de chamadas de ferramentas enquanto o agente trabalha.
- **O mesmo histórico de conversa** de qualquer outra superfície do Hermes — sessões iniciadas aqui retomam no CLI/TUI e vice-versa.
- **Arrastar e soltar arquivos** em qualquer lugar da área de chat para anexá-los à sua próxima mensagem.
- **Um painel de pré-visualização à direita** — renderize páginas web, arquivos e saídas de ferramentas lado a lado enquanto continua conversando.
- **Histórico do compositor e edição da fila** — pressione as setas para cima/baixo em um compositor vazio para recuperar e reutilizar prompts anteriores, e edite mensagens que você colocou na fila antes de serem enviadas. Pressionar Stop (ou Esc) enquanto há turnos na fila pausa a fila e a expande acima do compositor; retome a partir daí, ou envie, edite e exclua entradas individuais.

#### Barra de status

A barra na parte inferior do chat mostra o estado da sessão ao vivo e expõe controles rápidos sem abrir Configurações:

- **Alternância YOLO por sessão** — ligue ou desligue o YOLO apenas para esta sessão (igual ao TUI). O YOLO ignora os prompts de aprovação de comandos perigosos, então saiba o que você está desativando — veja [Segurança → Modo YOLO](./security.md#yolo-mode).

Conversando com uma instância Hermes em outra máquina em vez do backend local incluído? Veja [Conectando a um backend remoto](#connecting-to-a-remote-backend) abaixo — e para o panorama completo de como funciona a conexão do dashboard hospedado remotamente (o gate de autenticação, o socket de chat `/api/ws` e a triagem de códigos de fechamento WebSocket), veja [Web Dashboard → Conectando o Hermes Desktop a um backend remoto](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend).

#### Descoberta de repositórios

O Hermes Desktop descobre repositórios Git locais para a barra lateral Projetos escaneando seu diretório home até uma profundidade limitada. Você pode alterar isso por perfil em **Settings → Workspace**, ou em `config.yaml`:

```yaml
desktop:
  repo_scan_enabled: true
  repo_scan_roots: []
  repo_scan_exclude_paths: []
```

- Defina `repo_scan_enabled: false` para interromper completamente a varredura do filesystem. As linhas de cache de descoberta em disco existentes para esse perfil são limpas; projetos explícitos e repositórios inferidos de sessões Hermes intencionais permanecem disponíveis.
- Defina `repo_scan_roots` como uma lista de pastas para restringir a varredura. Uma lista vazia preserva a varredura padrão do diretório home.
- Defina `repo_scan_exclude_paths` para pastas cujas subárvores completas devem ser ignoradas.

Alterar qualquer um desses valores invalida apenas o cache de descoberta em disco daquele perfil e inicia uma atualização conforme a política. **Hide from sidebar** continua sendo uma ação de curadoria separada por item.

#### Escolhendo um modelo

O seletor de modelo fica no **compositor**, imediatamente à esquerda do microfone. Clique nele para alternar o modelo, o esforço de raciocínio e o modo rápido em um único dropdown.

- **O seletor do compositor é estado de UI persistente e nunca altera seu padrão.** Ele é lembrado localmente (por dispositivo) e **segue** entre novos chats e reinicializações, em vez de voltar ao padrão — escolha um modelo uma vez e o próximo `Cmd/Ctrl+N` abre nele. Com um chat ativo, trocar de modelo limita a mudança a **este chat**; de qualquer forma, a seleção acompanha quando a sessão é criada/trocada e **nunca** é gravada no padrão do perfil. (Trocar de [perfil](#sessions--profiles) reinicia com o padrão daquele perfil.)
- **Defina o padrão em Settings → Model.** Esse modelo "principal" é seu **padrão global por perfil** — é de onde novos chats, crons, subagentes e tarefas auxiliares partem, e é o único lugar que o grava. Cada [perfil](#sessions--profiles) mantém seu próprio padrão.
- **Presets de esforço/rápido por modelo.** Cada modelo lembra seu próprio esforço de raciocínio e escolha de modo rápido no app desktop, reaplicados à sessão sempre que você escolhe aquele modelo. Esses presets são uma conveniência do desktop e não alteram crons ou subagentes.
- **Trocas no meio do chat resetam o cache de prompt.** Trocar o modelo dentro de um chat ativo significa que a próxima mensagem relê toda a conversa pelo preço integral de entrada (caches de prompt do provedor são vinculados ao modelo). Tudo bem ocasionalmente; em um chat longo, um chat novo no novo modelo costuma ser mais barato do que ficar alternando.

### Navegador de arquivos

Explore e pré-visualize o diretório de trabalho sem sair do app — útil para acompanhar enquanto o agente lê, grava e edita arquivos. Defina o diretório inicial do projeto com `hermes desktop --cwd <path>` (ou a variável de ambiente `HERMES_DESKTOP_CWD`).

### Voz

Converse com o Hermes e ouça as respostas, o mesmo [modo de voz](./features/voice-mode.md) disponível em outros lugares. No macOS, o sistema pedirá acesso ao microfone uma vez.

### Configurações e onboarding

Gerencie provedores, modelos, ferramentas e credenciais por uma UI real em vez de editar YAML. O onboarding na primeira execução leva você à primeira mensagem em segundos. Os painéis de configurações cobrem provedores/chaves, seleção de modelo, configuração de toolsets, servidores MCP, o gateway e gerenciamento de sessões.

- **Painel de provedores** — um lugar dedicado para gerenciar provedores de inferência, com UX de Contas / chaves de API para login e armazenamento de credenciais por provedor.
- **Todo provedor e modelo nos menus** — a GUI expõe a lista completa de provedores e todos os modelos que `hermes model` conhece, para você escolher do mesmo catálogo que o CLI vê, em vez de um subconjunto curado.
- **xAI Grok OAuth** — Grok é um provedor OAuth de primeira classe no launcher; faça login pelo fluxo do navegador como os outros provedores OAuth.
- **Instalações de backends de ferramentas pela GUI** — execute as etapas pós-setup de instalação de um backend de ferramenta diretamente no app em vez de ir ao terminal.
- **Aviso de modelo auxiliar** — se você trocar o modelo principal para um novo provedor enquanto tarefas auxiliares (titulação, sumarização e helpers similares) ainda estão fixadas em outro provedor, o app avisa para que você não divida o trabalho entre dois provedores sem perceber.

O onboarding da primeira execução foi redesenhado em um sistema unificado de overlay, e você pode escolher **Choose provider later** para pular a configuração de provedor e entrar no app primeiro.

### Painéis de gerenciamento

O app também expõe a superfície mais ampla de gerenciamento do Hermes para que você não precise ir ao terminal:

- **Skills** — navegue, instale e gerencie [skills](./features/skills.md).
- **Cron** — visualize e gerencie [tarefas agendadas](../reference/cli-commands.md#hermes-cron).
- **Profiles** — alterne entre [perfis Hermes](./profiles.md) (config/skills/sessões isolados).
- **Messaging** — configure canais do gateway.
- **Agents** e **Command Center** — superfícies de orquestração para trabalho multiagente.

### Teclado e navegação

- **Paleta de comandos** — pressione **Cmd+K** (Ctrl+K no Windows/Linux) para ir a ações e navegar no app pelo teclado.
- **Atalhos reconfiguráveis** — um painel de atalhos em Settings permite remapear os atalhos de teclado do app para suas próprias teclas.
- **Atalhos de zoom personalizados** — amplie a interface em incrementos de meio passo para controle mais fino do tamanho do texto.
- **Seletor de idioma da UI** — altere o idioma da interface do app dentro do app, incluindo Chinês Simplificado (zh-Hans).

### Sessões e perfis {#sessions--profiles}

- **Reformulação da lista de sessões** — uma lista de sessões reformulada com arquivamento e higiene geral de sessões para manter a lista gerenciável conforme cresce.
- **Buscar sessões por id** — encontre uma sessão específica diretamente pelo id.
- **Sessões multi-perfil concorrentes** — execute sessões em vários [perfis](./profiles.md) ao mesmo tempo, e referencie uma sessão em outro perfil com links `@session` entre perfis.

## Atualização

O app verifica atualizações em segundo plano e oferece atualização com um clique quando uma estiver pronta.

O [processo de atualização manual](https://hermes-agent.nousresearch.com/docs/getting-started/updating) também funciona com a GUI.

## Desinstalação

Abra **Settings → About → Danger zone** e escolha o quanto remover:

- **Uninstall Chat GUI only** — remove o app desktop e seus dados; o agente Hermes, sua config e seus chats permanecem. (Igual a `hermes uninstall --gui`.)
- **Uninstall GUI + agent, keep my data** — remove o app e o agente, mas mantém config, chats e segredos para uma reinstalação futura. (Igual a `hermes uninstall`.)
- **Uninstall everything** — remove o app, o agente e todos os dados do usuário. (Igual a `hermes uninstall --full`.)

O app fecha para concluir o trabalho (a limpeza roda depois que ele sai, para poder remover o bundle do app em execução e seu próprio venv). As opções que removem o agente ficam ocultas automaticamente quando nenhum agente local está instalado (por exemplo, um cliente "lite" só de GUI conectado a um backend remoto).

Você pode fazer o mesmo pelo terminal — `hermes uninstall --gui` só para a GUI, ou `hermes uninstall` / `hermes uninstall --full` para o agente também.

:::note
Executar `hermes uninstall --gui` a partir de um **checkout de código-fonte** (um build de dev `hermes desktop`) também remove o `node_modules` do workspace e a saída de build `apps/desktop/{dist,release}`, já que são artefatos de build da GUI. Eles são recuperáveis com `hermes desktop` (ou `npm install` + rebuild) — mas se você estiver ativamente desenvolvendo o app desktop, espere reinstalar dependências depois.
:::

## Referência CLI: `hermes desktop`

Para iniciar via CLI, basta executar `hermes desktop`. Por padrão, instala dependências Node do workspace, compila o app Electron descompactado do SO atual e inicia esse artefato empacotado.

| Flag                 | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build`       | Skip npm install/package and launch the existing unpacked app from `apps/desktop/release` |
| `--force-build`      | Force a full rebuild even if the content stamp matches                                    |
| `--build-only`       | Build the desktop app but do not launch it (used by `hermes update`)                      |
| `--source`           | Launch via `electron .` against `apps/desktop/dist` instead of the packaged app           |
| `--cwd PATH`         | Initial project directory for desktop chat sessions (sets `HERMES_DESKTOP_CWD`)           |
| `--hermes-root PATH` | Override the Hermes source root the app uses (sets `HERMES_DESKTOP_HERMES_ROOT`)          |
| `--ignore-existing`  | Force the app to ignore any `hermes` CLI already on `PATH` during backend resolution      |
| `--fake-boot`        | Enable deterministic boot delays for validating the startup UI                            |

## Como funciona

O app empacotado inclui o shell Electron e uma superfície de chat React nativa. Na primeira inicialização, ele pode instalar o runtime Hermes Agent em `HERMES_HOME` (`~/.hermes`, ou `%LOCALAPPDATA%\hermes` no Windows) — **o mesmo layout de uma instalação CLI**, por isso os dois são intercambiáveis. A resolução de backend primeiro respeita `HERMES_DESKTOP_HERMES_ROOT`, depois uma instalação gerenciada concluída, depois um `hermes` detectado em `PATH` (a menos que `--ignore-existing` / `HERMES_DESKTOP_IGNORE_EXISTING=1` esteja definido), e por fim uma substituição explícita de comando `HERMES_DESKTOP_HERMES` para empacotadores como Nix. O renderer React conversa com um backend headless que o app inicia para você — um processo `hermes serve` que serve a API JSON-RPC/WebSocket do `tui_gateway` — e reutiliza o runtime do agente em vez de incorporar `hermes --tui`. O app desktop é **autocontido**: executa seu próprio backend `hermes serve` e nunca abre nem exige o [web dashboard](./features/web-dashboard.md). (Runtimes anteriores ao comando `serve` fazem fallback automaticamente para um `dashboard --no-open` headless, para que uma atualização do app nunca ultrapasse seu backend.) A lógica de instalação, resolução de backend e autoatualização vive no processo principal do Electron.

## Conectando a um backend remoto {#connecting-to-a-remote-backend}

Por padrão, o app inicia e gerencia seu próprio backend **local**. Você também pode apontá-lo para um backend Hermes em outra máquina — um VPS, um servidor doméstico ou um Mini atrás do Tailscale.

:::info O backend remoto é um processo `hermes serve` em execução
"Backend remoto" significa um servidor **`hermes serve`** rodando na máquina remota — esse é o processo ao qual o app desktop se conecta. Nada nesta seção funciona a menos que esse backend esteja de fato ativo e acessível. O app desktop não o inicia para você; você (ou um serviço `systemd`) mantém `hermes serve` rodando no host remoto, e o app se conecta a ele. Se você também usa canais de mensagens (Telegram, Discord, etc.), o **gateway** é um processo de longa duração *separado* que você inicia independentemente — veja a nota após as etapas de setup.
:::

A conexão tem duas metades: no backend você a protege com um **provedor de autenticação**, e no app você informa a URL do backend e faz login. Vincular o backend a um endereço que não seja loopback ativa automaticamente seu gate de autenticação, e o provedor que você configurar é o que permite a passagem do app desktop.

**Escolha um provedor com base em onde o backend está:**

- **OAuth (Nous Portal) — preferido para qualquer coisa acessível além da sua própria máquina.** Logins são verificados contra sua conta Nous, então esta é a opção adequada para um VPS, um host público ou qualquer backend remoto. Registre o dashboard com `hermes dashboard register` (ou a página Portal [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards)) para provisionar seu cliente OAuth, depois faça login no app com **Sign in with Nous Research**. Um provedor OIDC self-hosted funciona da mesma forma se você executar seu próprio provedor de identidade.
- **Username/password — apenas para uso local / rede confiável.** A opção mais simples quando o backend está na mesma LAN confiável ou acessível somente por VPN (ex.: Tailscale). Protege uma única credencial compartilhada sem provedor de identidade externo, então **não use para um dashboard exposto à internet pública** — use OAuth nesse caso.

O restante desta seção mostra o caminho username/password porque é o mais rápido de subir em uma rede confiável; para o caminho OAuth, veja [Web Dashboard → Default provider: Nous Research](./features/web-dashboard.md#default-provider-nous-research).

### No backend (a máquina remota)

Defina um username e password, depois inicie o backend vinculado a um endereço acessível. As credenciais ficam em `~/.hermes/.env` (o arquivo de segredos, mode 0600):

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

Mantenha esse processo `hermes serve` rodando enquanto quiser que o app desktop possa conectar — se parar, o app não consegue mais alcançar o backend. Execute sob `systemd`, `tmux` ou o gerenciador de processos de sua preferência para sobreviver a logout e reboots.

Separadamente, certifique-se de que o **gateway está rodando** no host remoto se você depende de canais de mensagens — o backend `hermes serve` é com o qual o app desktop conversa, mas suas sessões de gateway Telegram/Discord/Slack são um processo diferente que você inicia e mantém por conta própria. Veja [Messaging](./messaging/index.md) para setup do gateway.

Prefere não manter uma senha em texto plano em repouso? Defina `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` como um hash scrypt — calcule com `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`. Superfície completa de configuração (chaves config.yaml, cada env var, o rate limiter): [Web Dashboard → Username/password provider](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

Rodando o backend como serviço systemd? Dê à unit `EnvironmentFile=%h/.hermes/.env` para que as credenciais estejam no ambiente na inicialização.

:::warning
O backend lê e grava seu `.env` (chaves de API, segredos) e pode executar comandos do agente. O setup **username/password** mostrado acima é para uma rede confiável — nunca exponha um backend protegido por senha diretamente à internet aberta; coloque-o atrás de uma VPN. [Tailscale](https://tailscale.com/) é a opção limpa: vincule ao IP tailscale da máquina (`--host <tailscale-ip>`) e use `http://<tailscale-ip>:9119` como Remote URL para que apenas sua tailnet alcance. Para alcançar um backend pela internet pública, use o provedor **OAuth (Nous Portal)**.
:::

### No app

**Settings → Gateway → Remote gateway:**

1. **Remote URL** — `http://<backend-host>:9119` (prefixos de path como `/hermes` funcionam se você colocar um reverse proxy na frente)
2. **Sign in** — o app detecta qual provedor o backend anuncia e adapta o botão. Para um backend username/password, mostra um botão **Sign in** que abre um formulário de credenciais (informe as credenciais da etapa 1). Para um backend OAuth, mostra **Sign in with `<provider>`** (ex.: *Sign in with Nous Research*), que executa o login do provedor no navegador. De qualquer forma, o app termina com uma sessão autenticada contra o backend.
3. **Save and reconnect** — troca o shell desktop para o backend remoto. A sessão é atualizada automaticamente; você permanece logado entre reinicializações quando `HERMES_DASHBOARD_BASIC_AUTH_SECRET` está definido.

Você também pode definir a URL do backend sem a UI via a variável de ambiente `HERMES_DESKTOP_REMOTE_URL` antes de iniciar o app (ela substitui a configuração no app); você ainda faz login no painel Gateway.

:::note Hosts remotos por perfil
O host do gateway remoto é configurado por [perfil](./profiles.md), então cada perfil pode apontar para seu próprio backend remoto (ou permanecer no local). Trocar de perfil troca para qual host remoto o app se conecta.
:::

### Solução de problemas

- **Login falha com 401 / "Invalid credentials"** — o username ou password não corresponde a `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` do backend. O backend retorna o mesmo erro genérico para usuário desconhecido e senha errada (sem oracle de enumeração), então confira ambos. Confirme que o gate está ativo com `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` — deve reportar `true` e incluir `"basic"`.
- **Sem botão "Sign in" — pede um session token** — o provedor username/password do backend não está ativo. `/api/status` não listará `"basic"` em `auth_providers`. Certifique-se de que username e password (ou hash de password) estão definidos em `~/.hermes/.env` e que o processo do dashboard realmente os carregou.
- **Deslogado a cada reinicialização** — defina `HERMES_DASHBOARD_BASIC_AUTH_SECRET` com um valor estável. Sem isso, a chave de assinatura de token é regenerada a cada boot, invalidando todas as sessões.
- **Connection refused / times out** — o backend está vinculado a `127.0.0.1` (o padrão) ou um firewall/VPN está bloqueando a porta. Vincule a `0.0.0.0` ou ao IP tailscale e abra a porta para sua rede confiável.

Para o mesmo setup pela perspectiva do web dashboard, veja [Web Dashboard → Connecting Hermes Desktop to a remote backend](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend); as env vars estão catalogadas em [Environment Variables → Web Dashboard & Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop).

## Estendendo o app desktop

O app desktop é orientado a contribuições — painéis, páginas, navegação da barra lateral, itens da barra de status,
comandos da paleta, keybinds e temas registram-se por um SDK, e
você pode adicionar os seus. Um plugin é um único arquivo ESM colocado em
`$HERMES_HOME/desktop-plugins/<id>/plugin.js`; o app o carrega em segundos e
recarrega a cada save. Gerencie plugins instalados ao vivo em **Settings → Plugins**.

Veja [Desktop Plugin SDK](../developer-guide/desktop-plugin-sdk.md) para a referência
completa. (Isso é separado do [sistema de plugins do web dashboard](./features/extending-the-dashboard.md).)

## Solução de problemas

Logs de boot vão para `HERMES_HOME/logs/desktop.log` (inclui saída do backend e tracebacks Python recentes) — verifique primeiro se o app reportar falha de boot. Você também pode acompanhar pelo CLI:

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

### "Build desktop app" travado no download do Electron

O build baixa o runtime Electron (~114&nbsp;MB) de `github.com/electron/electron/releases`. Se o instalador travar na etapa **Build desktop app** com a saída ao vivo repetindo `retrying attempt=…`, o GitHub está sendo bloqueado ou limitado na sua rede (firewall, proxy ou região).

O instalador se recupera automaticamente: em um build com falha, (1) limpa um zip Electron em cache corrompido e tenta de novo, depois (2) se ainda falhar e você não definiu `ELECTRON_MIRROR`, tenta mais uma vez via `npmmirror.com`, o espelho comunitário de facto do Electron. `@electron/get` verifica SHASUM do download, mas os checksums vêm do mesmo espelho — isso detecta download corrompido ou parcial, não um espelho comprometido. Se preferir não confiar em um host de terceiros, fixe seu próprio `ELECTRON_MIRROR` (abaixo); o build nunca substitui um que você definiu.

Para **escolher seu próprio espelho** (ex.: corporativo/confiável), defina `ELECTRON_MIRROR` antes de instalar ou reconstrua manualmente — o build respeita e não substitui:

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ \
  bash -c 'cd "$HOME/.hermes/hermes-agent/apps/desktop" && CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack'
```

Para limpar um zip em cache corrompido manualmente:

```bash
rm -f "$HOME/Library/Caches/electron"/electron-*.zip   # macOS
rm -f "$HOME/.cache/electron"/electron-*.zip            # Linux
```

## Compilando a partir do código-fonte

Se quiser hackear o app em si, instale as deps do workspace a partir da raiz do repo uma vez, depois execute o servidor de dev de `apps/desktop`:

```bash
npm install          # from repo root — links apps/desktop, web, apps/shared
cd apps/desktop
npm run dev          # Vite renderer + Electron, which boots the Python backend
```

Aponte o app para um checkout específico, ou isole-o da sua config real:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # exercise the startup overlay with deterministic delays
```

Compile instaladores:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # unpacked app under release/ (no installer)
```

Assinatura e notarização macOS/Windows rodam automaticamente quando as credenciais relevantes estão presentes no ambiente (`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*` para macOS, `WIN_CSC_*` para Windows).

## Veja também

- [CLI Guide](./cli.md) — a interface de terminal
- [TUI](./tui.md) — a UI de terminal moderna usada por `hermes --tui` e a aba chat do dashboard
- [Web Dashboard](./features/web-dashboard.md) — painel de administração no navegador com aba chat incorporada
- [Configuration](./configuration.md) — config que o app desktop lê e grava
- [Windows (Native)](./windows-native.md) — caminho de instalação nativa no Windows
