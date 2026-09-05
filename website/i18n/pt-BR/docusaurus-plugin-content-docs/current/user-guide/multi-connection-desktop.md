---
sidebar_position: 5
---

# Conectando o Desktop a várias instâncias do Hermes {#connecting-desktop-to-many-hermes-instances}

Registre todos os backends Hermes que você possui — o runtime local, gateways remotos na
LAN ou num VPS, hosts SSH e instâncias Hermes Cloud — em um único app desktop,
e use os agentes de todos eles lado a lado. As conexões são persistentes:
cada gateway registrado disca os próprios backends e WebSockets sob demanda, e
agentes em background continuam fazendo streaming enquanto você olha para outro gateway.

Este é o complemento no desktop de
[Executando vários gateways ao mesmo tempo](./multi-profile-gateways.md): aquela página é
sobre hospedar vários gateways numa máquina; esta é sobre um app desktop
falando com várias máquinas.

## Onde encontrar {#where-to-find-it}

Tudo mora na página unificada **Settings → Gateways** (builds mais antigos tinham
páginas separadas **Gateway** e **Connections**; deep links legados de Connections
redirecionam para lá). Três portas levam a ela:

- **Settings → Gateways** — a própria página (**Cmd/Ctrl+,**, depois
  **Gateways** na nav de settings). O registry de conexões é uma seção
  dessa página, abaixo dos controles de connection-mode no nível da máquina.
- **O rail de profiles da sidebar** — o botão de plugue na ponta direita do rail
  (tooltip: **"Connect another Hermes gateway…"**) faz deep-link direto para
  a página Gateways. Fica sempre visível, mesmo antes de você criar
  um segundo profile ou uma segunda conexão.
- **A command palette** — **Cmd/Ctrl+K**, depois digite *Gateways* (também
  casa com *connections*, *add gateway*, *remote*, *ssh*, *instances*).

## O registro de gateways {#the-gateway-registry}

A seção **Registered gateways** de **Settings → Gateways** gerencia uma lista
nomeada de gateways Hermes. A introdução diz direto: *"Manage this device and
every Hermes gateway it can reach through remote, SSH, or Cloud connections."*
Cada entrada é uma *connection*:

| Tipo | O que é | Auth |
|---|---|---|
| **Local** | "The Hermes runtime managed by this app." | automático |
| **Remote gateway** | "A Hermes gateway reachable over HTTP(S) — LAN, Tailscale, or the internet." | session token ou OAuth |
| **SSH** | "A Hermes install reached over SSH." O app abre o túnel e inicia o dashboard para você | chave SSH + token adotado |
| **Hermes Cloud** | "A hosted instance discovered through your Hermes Cloud account." | sign-in do portal |

Regras que vale conhecer:

- **Toda conexão precisa de um device name único** ("Homelab", "Work laptop").
  O nome aparece em todo lugar em que a instância surge — badges do roster, handles,
  resultados de update. Unicidade ignora maiúsculas/minúsculas, então `Homelab` e `homelab`
  não podem coexistir.
- A entrada **local** é gerenciada pelo app (usa um pill **App-managed**)
  e não pode ser removida. Remover qualquer outra conexão derruba os backends
  e túneis ao vivo; a instância em si não é tocada.
- Uma conexão é sempre a **Primary** (pill na linha): ela é o
  fallback do registry para chamadas multi-gateway que não nomeiam um gateway.
  **Make primary** não troca o workspace atual de Sessions; remover
  a primary cai de volta para a entrada local.
- **At startup, return to Sessions on the last-used gateway** controla qual
  gateway Sessions abre depois de um restart completo do app. Fica off por padrão, então
  Sessions abre no **Primary**. Ligue para retomar o gateway mais recente
  que conectou com sucesso. Um switch que falhou nunca é lembrado, e um
  gateway salvo removido ou indisponível cai de volta para Primary.
- **Test** sonda as pernas HTTP *e* WebSocket da própria conexão, então um pass
  (toast *"Reachable"*) significa que o chat de fato vai funcionar — não só que o
  host pingou.
- **Duplicatas são rejeitadas quando você salva**: só existe uma entrada **local**;
  entradas **remote** e **cloud** são deduplicadas na URL normalizada
  (trimmed, trailing slashes removidos, lowercased — e entre ambos os
  kinds, então uma entrada cloud e uma remote não podem apontar para a mesma URL);
  entradas **SSH** são deduplicadas no `user@host:port` normalizado mais
  o profile remoto.
- Entradas Cloud normalmente vêm do fluxo de sign-in/discovery do Hermes Cloud no
  topo da página Gateways — o kind **Hermes Cloud** no editor de add-connection
  aponta você para lá.

Troque gateways pela sidebar **Sessions**. Profiles, chats, messaging e
cron ficam escopados àquele gateway; o window backend gerenciado pelo app ainda é escolhido
pelos controles de connection-mode acima. **Primary** é o fallback do registry e
não troca o workspace atual.

## Adicionando uma conexão, passo a passo {#adding-a-connection-step-by-step}

1. Abra **Settings → Gateways** e role até o registry de conexões (ou
   clique no plugue no rail de profiles).
2. Clique em **Add connection**.
3. Escolha o tipo: **Local**, **Hermes Cloud**, **Remote gateway** ou **SSH**.
   (**Local** fica desabilitado enquanto a entrada local gerenciada pelo app existir — o que
   é quase sempre; **Hermes Cloud** direciona você ao fluxo de sign-in/discovery
   cloud acima.)
4. Preencha os campos:
   - **Name** — obrigatório, único; o "device name" mostrado em todo lugar em que
     esta instância aparece (placeholder: `Homelab`). Máximo 64 caracteres.
   - *Só remote gateway:*
     - **Gateway URL** — a URL base de um backend `hermes serve` em execução,
       por exemplo `http://homelab.lan:9119`. Prefixos de path de reverse-proxy funcionam.
     - **Authentication** — escolha **Session token** ou **OAuth**:
       - **Session token** — cole o dashboard session token do
         gateway remoto. Ao editar, *"Leave blank to keep the saved
         token."*
       - **OAuth** — faça sign-in pelo fluxo de browser do Nous Portal; sem token
         para colar.
   - *Só SSH:*
     - **SSH host** — um campo composto no formato `user@host:22` (user e
       port opcionais). Sua chave SSH é usada; o app adota um dashboard
       token pelo túnel.
5. Clique em **Save connection** (ou **Cancel**).
6. Clique em **Test** na nova linha e espere *"Reachable"*.

Edite qualquer entrada não-local depois com o lápis, ou remova-a com a
lixeira — a remoção pede confirmação e lembra que *"The
instance itself is not touched — you can add it again any time."*

:::info O backend remoto é um processo `hermes serve` em execução
Nada aqui funciona a menos que o backend esteja de fato no ar e alcançável na
outra máquina. O app desktop se anexa a ele; não o inicia para você
(exceto conexões SSH, em que o app inicia o dashboard pelo
túnel sob demanda). Veja
[Conectando a um backend remoto](./desktop.md#connecting-to-a-remote-backend)
para o setup no lado do backend — auth providers, bind em endereço que não é loopback,
e orientação de Tailscale.
:::

### Migrando das settings de conexão única {#migrating-from-the-single-connection-settings}

O primeiro launch de um build com registry importa as settings existentes
automaticamente: o modo de conexão global e quaisquer overrides legados por profile de
Settings → Gateway viram entradas nomeadas no registro (deduplicadas
por URL/host). (Builds mais novos não oferecem mais overrides por profile na
página Gateways — conexões de gateway são no nível da máquina, e profiles
são descobertos a partir dos gateways a que você conecta.)
O arquivo legado de settings fica intacto, então builds mais antigos na mesma
máquina continuam funcionando. Se um nome migrado colidiu, ganhou sufixo
(`Homelab 2`).

## Agentes entre gateways {#agents-across-gateways}

Cada [profile](./profiles.md) em cada conexão registrada é um *agent*.
O roster união é o que as superfícies multi-gateway (e o roster built-in do
[Bot Mode](./bot-mode.md)) renderizam:

- Quando o mesmo nome de profile existe em vários gateways, os handles desambiguam
  como **`@name-device`** — `research` no seu Homelab renderiza como
  `@research-homelab`, enquanto um profile único em todos os gateways mantém o
  nome nu.
- A enumeração é eager, mas os sockets são lazy: o app lista agentes via REST
  sem discar o WebSocket de cada gateway. Um gateway inalcançável reporta
  por linha em vez de quebrar o roster; conexões SSH ficam connect-on-demand
  até você abrir um agente nelas pela primeira vez (sem túneis-surpresa).
- Abrir um agente disca **o próprio gateway dele** — chats, sessões e memória
  vivem na máquina que é dona do profile, exatamente como se você estivesse usando
  aquela instância direto.

Cada par `(connection, profile)` ganha o próprio backend e socket, pooled
com o mesmo idle-reaping dos backends locais por profile — agentes em background
continuam o streaming enquanto você olha para outro gateway.

### Alternar e escopo {#switching-and-scoping}

O pé da sidebar segue uma hierarquia: **gateway → profile → sessions**.
Gateways são máquinas ou backends hospedados; profiles são agentes Hermes isolados
que vivem num gateway.

- Com um gateway registrado, nenhum controle de gateway é adicionado. Desktop só-local
  mantém o mesmo rail de profiles e fluxo de teclado de antes.
- Com vários gateways, a sidebar mostra um seletor de gateway nomeado. Seu ícone de device,
  cloud, network ou terminal identifica o tipo de conexão; avatars de
  profile permanecem um controle separado depois do divisor. O mesmo seletor escala
  de dois gateways a uma frota maior sem transformar backends em glyphs estilo
  profile nem empurrar ações de profile para fora do rail.
- Selecionar um gateway restaura o último profile usado lá. O pill home
  volta ao default dele e o pill de camadas mostra **All profiles on
  this gateway**. **Cmd/Ctrl+1–9** continuam trocando profiles dentro do
  gateway ativo.
- Com vários gateways o rail de profiles vira um **fleet rail**: os profiles de todo gateway
  registrado ficam na mesma faixa, cada grupo liderado pelo glyph de kind daquele gateway
  (device, network, terminal, cloud) — o mesmo glyph do seletor de gateway.
  Os squares do gateway ativo parecem exatamente como num Desktop single-gateway; os dos
  outros gateways ficam dimmed ("at rest").
  Hover num square at-rest nomeia sua máquina (`omer · This device`), então dois
  profiles de mesmo nome em máquinas diferentes nunca parecem iguais.
- Clicar num square at-rest faz o mesmo switch do seletor de gateway,
  aterrando naquele `(gateway, profile)` exato: o square gira enquanto o
  target é discado, o gateway anterior permanece pintado até o target
  responder, e um target morto falha o click com mensagem em vez de
  deixar a janela half-switched. Grupos mantêm ordem de registry qualquer que seja o
  gateway ativo, então um square nunca se move sob o ponteiro que clicou
  nele. Right-click num square at-rest oferece **Switch to**, **Color**,
  **Rename**, **Edit SOUL.md** e **Delete**, todos executados no gateway
  daquele square; a confirmação de delete nomeia a máquina.
- Um gateway que a última enumeração não alcançou mantém seus squares, marcados
  com um dot âmbar no glyph — uma caixa dormindo ainda é sua. Duas
  registrations de um backend colapsam num grupo só. Passados treze
  squares na frota, a faixa condensa num menu seccionado por
  gateway.
- O gateway selecionado sobrevive a um quit e relaunch só quando **Settings →
  Gateways → At startup, return to Sessions on the last-used gateway** está on.
  A preferência e o id do gateway vivem no registry de user-data do app, então
  substituir ou atualizar o application bundle não os reseta.
- Com mais de treze profiles no gateway ativo, a faixa de avatars
  condensa num seletor de profile nomeado. Conjuntos grandes de gateway e profile podem
  portanto coexistir sem mudar o modelo **gateway → profile → sessions**.
- **This device** permanece um gateway de primeira classe mesmo quando uma conexão remota é
  Primary. Pode manter sessões locais disponíveis durante uma outage remota, mas o
  app não o chama de "offline mode": o modelo ou tools selecionados ainda podem
  exigir acesso à internet.
- A lista de sessões, canais de messaging, cron jobs, settings, files e memória
  são todos escopados ao `(gateway, profile)` ativo. Trocar de um gateway Telegram
  para um gateway Signal não pode deixar os channel groups ou sessões do gateway
  anterior na sidebar.
- Meramente exibir o switcher lê o registry local de conexões do Electron.
  Gateways remotos só são abertos quando selecionados; não há polling periódico
  da frota.
- Passar o mouse sobre um agente pré-aquece o backend dele para o switch não pagar cold
  boot.
- A página **Capabilities** (Skills / Tools / MCP) tem um escopo correspondente: seu
  seletor **Configuring** lista todo agente `(profile, device)` do
  roster união, e escolher um lê e escreve skills, toolsets e servidores MCP
  **daquela máquina** sem trocar o workspace de Sessions. Installs do hub,
  env keys e setup MCP todos caem no backend do agente selecionado.
  O botão *hot-reload into a live session* da tab MCP aparece só para agentes
  no gateway ao qual a janela está conectada; edits em outras máquinas aplicam
  na próxima sessão delas.

Adicione, teste, renomeie ou remova gateways em **Settings → Gateways**. O botão de
plugue ao lado das ações de profile é um atalho para essa única home de
gerenciamento, não um segundo fluxo de add.

### Sessions e Bot Mode {#sessions-and-bot-mode}

Sessions intencionalmente mostram um gateway ativo de cada vez: isso mantém files,
tools, channels, cron e histórico de sessão num contexto de execução
compreensível. O fleet profile rail alarga só o *picker* — o workspace ainda
vive em exatamente um `(gateway, profile)` após cada click. Bot Mode serve um
job diferente e pode apresentar o roster união, agrupado por gateway, para um usuário
abrir um agente num NAS e outro num VPS de uma superfície. Abrir um
bot ainda ativa a rota exata `(gateway, profile)` dele.

Mentions diretas de bot e delegation permanecem gateway-local por padrão. Cruzar um
boundary de backend muda filesystem, credenciais, tools e contexto de trust, então
execução cross-gateway deve ser uma bridge explícita em vez de um side effect
acidental de compartilhar uma janela Desktop.

## Atualizando todas as instâncias de uma vez {#updating-every-instance-at-once}

**Settings → Gateways → Update all instances** (aparece quando mais de uma
conexão está registrada) dispara `hermes update` em paralelo para cada conexão
elegível:

- **Local** atualiza pelo pipeline de update do próprio app (o mesmo fluxo de
  Settings → Updates).
- Conexões **Remote e SSH** são instruídas a se atualizar via o próprio
  backend — o update roda *naquela* máquina.
- Instâncias **Hermes Cloud** são puladas com a nota *"Managed by Hermes Cloud"*:
  a plataforma gerencia as versões delas.

Cada instância reporta de forma independente, então uma caixa inalcançável nunca trava o
lote. Backends que gerenciam updates por fora (Docker, Nix) recusam educadamente
com a própria mensagem, por linha.

Você raramente precisa do botão Settings, porém: uma vez que existe mais de um alvo de
update, as affordances regulares de update do app (**Update now** no painel About,
⌘K **Update Hermes**, o toast update-ready) rodam o mesmo fan-out
automaticamente — backend ativo primeiro, depois todo outro gateway elegível, depois
o próprio app desktop por último. Veja
[Atualização](./desktop.md#updating) no guia desktop.

## Notas de segurança {#security-notes}

- **Onde os tokens vivem.** Session tokens de remote-gateway (e tokens OAuth de
  native sign-in, keyed pela URL base do gateway) ficam no diretório user-data
  do app como arquivos owner-only (0600), no processo main do Electron; o
  renderer e os plugins nunca veem bytes de token.
- **Criptografia opcional via keychain.** Por padrão os tokens **não** passam
  pelo keychain do SO — no macOS em particular, o `safeStorage` do Electron
  guarda uma chave por app no login keychain, e um keychain locked ou quebrado
  vira password prompt a cada launch. Se quer criptografia at-rest além das
  permissões de arquivo, ligue **Settings → Gateway →
  "Encrypt saved secrets with the OS keychain"**; segredos armazenados existentes são
  re-criptografados in place (Keychain no macOS, DPAPI no Windows, backend de keyring
  da sessão no Linux). Desligar de novo descriptografa.
- **O arquivo de registry** (`connections.json` no diretório de user-data
  do app) guarda labels, URLs e hosts — segredos só aparecem dentro de
  envelopes criptografados quando a opção de keychain está ligada.
- O `host.connections()` do plugin SDK de propósito retorna labels, kinds
  e o id primary — nunca material de token.

## Para autores de plugin {#for-plugin-authors}

O [plugin SDK](../developer-guide/desktop-plugin-sdk.md) do Desktop expõe a
superfície multi-gateway direto:

- `host.connections()` — a lista de conexões registradas (labels, kinds,
  primary; nunca bytes de token).
- `host.agents()` — o roster união: uma linha por `(gateway, profile)` com
  o handle `@name-device` pré-computado.
- `host.ensureAgent(connectionId, profile)` — ativa o gateway de um agente para
  chamadas seguintes de `host.request` baterem no backend dele.
- `host.warmAgent(connectionId, profile)` — pré-aquecimento fire-and-forget do socket
  (intenção de hover).

Os quatro são feature-detected: num build mais antigo do Desktop eles estão ausentes e um
plugin deve cair no fluxo single-gateway `profiles.list`. O roster multi-gateway do Bot Mode
é o consumidor de referência.

## Troubleshooting {#troubleshooting}

- **"Connection test failed"** — o backend não está alcançável nessa URL a partir
  desta máquina. Confirme que `hermes serve` está rodando no host remoto, a
  porta está aberta e (para token auth) o token está atual. Rode **Test**
  de novo depois de corrigir.
- **Um agente aparece mas não abre** — rode **Test** na conexão dele. A
  perna WebSocket falhando enquanto o HTTP passa costuma significar proxy, firewall ou
  um guard de auth/origin do gateway bloqueando `/api/ws`.
- **Um gateway remoto falta no roster** — o backend está down ou
  inalcançável; o roster lista isso sob gateways com o erro. Conexões SSH
  mostram *connect-on-demand* até o primeiro uso — isso é por design, não uma falha.
- **"Update Hermes Desktop to chat with agents on other connections"** — o
  app é anterior à stack multi-connection; atualize o próprio app desktop.
- **Device names duplicados** — não é possível; nomes são únicos na hora
  de salvar. Se um nome migrado colidiu, ganhou sufixo (`Homelab 2`).
- **"Could not save the connection"** — o mais comum é **Name** faltando, um
  nome já em uso, ou **Gateway URL** / **SSH host** malformado; a
  mensagem de erro nomeia a violação exata.
