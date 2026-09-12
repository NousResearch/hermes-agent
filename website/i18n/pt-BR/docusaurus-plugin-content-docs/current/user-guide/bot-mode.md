---
title: "Bot Mode"
description: "Transforme seus profiles do Hermes num roster de Bots nomeados — cada um com o próprio chat, papel, modelo, memória, skills e avatar. Bots rodam rotinas, compartilham group chats e se mensagens."
---

# Bot Mode {#bot-mode}

**Bot Mode** transforma seus [profiles do Hermes](./profiles.md) num roster de **Bots** nomeados. Cada Bot tem o próprio papel, modelo, memória, skills e avatar; Bots rodam rotinas recorrentes, deliberam juntos em group chats e se mensagens diretamente. Monte um Bot especialista uma vez e ele fica lá para sempre, a um clique de distância.

O Bot Mode vem **embutido no [app desktop](./desktop.md)** e fica **ligado por padrão** — sem instalação. Aparece como a aba **Bots** ao lado de Sessions na sidebar esquerda, com um tile **Routines** ancorado ao lado da conversa enquanto a aba Bots está ativa.

:::tip Um Bot é um profile
Não há primitivo novo para aprender: um Bot **é** um profile do Hermes — config, memória, skills, credenciais e histórico de chat isolados em `~/.hermes/profiles/<name>/`. O Bot Mode é uma UI sobre esse primitivo, então tudo o que você faz nele também é visível pela CLI: `hermes -p <bot> chat` abre o mesmo agente, e as rotinas do Bot aparecem em `hermes cron list`. Sem patches no core, sem daemons em background, sem storage extra.
:::

## O painel Bots {#the-bots-pane}

O roster mostra uma linha por profile de agente: avatar, preview da última mensagem e timestamp.

- **Clique num Bot** para cair no chat dele — todo Bot tem uma conversa canônica e persistente **Bot Chat**, criada (e pinada) no momento em que o Bot nasce. Um clique na linha sempre abre aquele Bot Chat (a mesma conversa que a linha pré-visualiza), mesmo quando você tem outras abas abertas para o Bot; essas abas ficam abertas ao lado. Na faixa de abas o Bot Chat é legendado com o nome do Bot, para dois Bots abertos se distinguirem de relance.
- **Active now** — uma faixa de presença acima do roster mostra todo Bot trabalhando agora: o profile ocupado no gateway mais qualquer Bot que escreveu nos últimos 90 segundos. Cada chip abre o chat daquele Bot. A faixa nunca reordena o roster e some quando a frota está idle.
- **Search** filtra o roster enquanto você digita.
- **Hide a Bot** — clique com o botão direito numa linha → **Hide Bot** para tirar do roster e da faixa Active-now um Bot que você não usa. Esconder é só display: @mentions ainda resolvem, memberships de group chat ficam intactas e as rotinas continuam rodando. Assim que pelo menos um Bot está escondido, um **toggle de olho** aparece no header do painel — clique para revelar Bots escondidos esmaecidos no lugar, depois botão direito → **Unhide Bot** para trazer um de volta. Bots escondidos nunca fazem toast, mas acumulam atividade unread em silêncio e o olho ganha um ponto para você saber que algo aconteceu. O estado hidden fica salvo nos metadados do profile do Bot, então acompanha o Bot em todo desktop conectado àquele backend.

:::note O Bot Chat canônico é um forever-chat
Digitar `/new` (ou `/reset`) dentro do chat canônico de um Bot bifurcaria o relacionamento numa sessão scratch — a única coisa que o Bot Mode promete que nunca acontece. O composer redireciona para `/compact` em vez disso: contexto de trabalho fresco, mesma conversa. Sessões regulares no mesmo profile mantêm a liberdade total de `/new`.
:::


### Organize bots em seções {#organize-bots-into-sections}

Seções são pastas que você mesmo cria — **Clients**, **Team**, o que couber — como um segundo eixo ao lado do agrupamento automático por gateway. Sem seções criadas o roster é a lista simples de sempre.

- **Crie uma** no menu **+** do painel → **New section**, ou clique com o direito num Bot → **Move to section** → **New section…** (isso já arquiva o Bot nela ao criar).
- **Arquive um Bot** arrastando a linha até uma seção — o alvo destaca enquanto você paira, e **Esc** cancela o drag — ou clique com o direito → **Move to section** e escolha uma. **Remove from section** o devolve a **Unassigned**.
- **Renomeie, reordene ou delete** uma seção no menu de clique-direito do heading (ou no **⋯** que aparece no hover); dê duplo clique num heading para renomear. Headings dobram como os headings de gateway.
- **Deletar uma seção nunca deleta Bots** — eles voltam a **Unassigned**, e o toast oferece **Undo**. Nenhuma confirmação é pedida.

A associação fica nos metadados de perfil de cada Bot (`ui_meta`), então a seção de um Bot o segue a todo desktop conectado àquele backend. Quando o roster mostra mais de um gateway, seções aninham dentro do bucket de cada gateway.

## Criando um Bot {#creating-a-bot}

Aperte **New Agent** no roster. O caminho rápido tem três campos — **Name**, **Title**, **Description** — e o Bot existe em segundos, se apresentando como a primeira mensagem do Bot Chat novo.

Um disclosure **Advanced** abre a superfície completa de capabilities:

- **Clone from an existing profile** — comece a partir da config, skills, SOUL e memória de outro Bot, ou escolha **Fresh profile** para um start limpo.
- **Create empty** — pule as skills bundled por completo para um profile mínimo.
- **Model & provider pin** — dê ao Bot o próprio modelo. Qualquer par provider/model que o Hermes conhece funciona, e Bots diferentes podem rodar em modelos diferentes lado a lado. Deixe em branco para herdar do profile de launch.
- **Custom SOUL.md** — a persona e as instruções permanentes do Bot.
- **Per-skill, per-toolset, and per-MCP-server enablement** — marque exatamente as capabilities que este especialista precisa.
- **Shared keys** — por padrão o Bot novo compartilha um pool OAuth/token com o profile principal, para refreshes de credencial não se invalidarem mutuamente. (Gateways mais antigos copiam as credenciais em vez disso — ainda funcional, só que forked.)

### Escolhendo em qual máquina ele vive ("Create on") {#choosing-which-machine-it-lives-on-create-on}

Com mais de uma conexão registrada em [Settings → Connections](./multi-connection-desktop.md), o diálogo New Agent ganha um picker **Create on**. Escolha um device e o profile é criado no backend **daquela** máquina — sua janela nunca troca de gateway. O Bot novo aparece no roster como um Connections Bot (com handle `@name-device` quando o nome existe em várias máquinas), e conversar com ele roteia para a própria máquina dele.

Com uma única conexão (o caso comum) o picker fica oculto e o Bot é criado na máquina à qual você está conectado — exatamente o comportamento antigo.

Notas de criação remota:

- **Clone source** é um profile da máquina *alvo* (o `default` dela) — uma caixa remota não tem seus profiles locais para clonar.
- A aba live Capabilities pina no backend da máquina alvo, então skills, tools e servidores MCP que você configura durante a criação caem na máquina onde o Bot vai viver. (Builds mais antigos do desktop caem para checklists staged de Skills/Tools/MCP em alvos remotos; ambos leem o catálogo da máquina alvo.)
- Cancelar o diálogo descarta o profile draft na máquina em que ele foi criado.

**Edit Profile** (botão direito num Bot) reabre a mesma superfície no profile live a qualquer hora: avatar, title, description, model pin, skills, toolsets, servidores MCP e o SOUL.md completo.

**Duplicate** (botão direito) faz um clone completo de um Bot — config, skills, SOUL.md, memória e a aparência. **Delete Profile** remove um permanentemente, atrás da mesma confirmação destrutiva do menu de profiles do desktop; o profile default não pode ser deletado.

## Avatars {#avatars}

Todo Bot ganha um rosto:

- **Blob faces** (padrão) — um rosto soft-body determinístico desenhado a partir do nome do Bot: mesmo nome, mesmo rosto, para sempre. Enquanto você digita um nome em New Agent o rosto acompanha ao vivo; aperte **Randomize** para re-rolar, **Lock face** para manter o que você gosta mesmo se o nome mudar, ou pin uma das seis silhuetas (round, organic, boxy, nub, cloud, sun) enquanto o resto ainda vem do nome.
- **Geometric faces** — as clássicas 7 shapes × 10 cores, com olhos piscando que vasculham enquanto o Bot trabalha.
- **Uma imagem enviada** — qualquer foto que você goste.
- **Um retrato gerado por IA** — quando um backend de imagem está configurado, gerado no lugar (isso usa o RPC padrão `image.generate` e funciona em gateways locais e remotos).
- **Um pixel pet** — um companion da [galeria petdex](./features/pets.md) que quica ao lado do avatar enquanto o Bot está ocupado. Rode `hermes pets` num terminal para explorar a galeria.

O look, title e description de um Bot ficam nos metadados do profile no backend, então o mesmo Bot aparece do mesmo jeito em todo desktop conectado àquele backend.

## Routines {#routines}

O painel **Routines** anexa tarefas recorrentes ao Bot que as executa — "summarize my inbox every morning" vive ao lado do Bot responsável. O painel ancora ao lado do chat só enquanto a aba Bots está ativa e sai de cena quando você volta para Sessions (builds mais antigos do desktop mantêm sempre visível). Um picker estruturado de schedule monta o schedule (frequência primeiro, depois só o detalhe que importa), com um campo Advanced expondo a string crua de schedule do Hermes.

Routines são [cron jobs](./features/cron.md) comuns do Hermes com namespace `[bot:<name>] <routine>` — também aparecem em `hermes cron list` e na página core de Cron. As runs caem no histórico de chat do próprio Bot, então o resultado fica exatamente onde você falaria com aquele Bot de qualquer jeito.

## Groups e group chats {#groups-and-group-chats}

Botão direito num Bot local → **Manage groups** para adicioná-lo ou removê-lo de qualquer número de group chats. Escolha groups existentes de forma independente ou crie um inline. Membership local fica nos metadados de profile sincronizados pelo backend do Bot, então acompanha aquele profile entre desktops; profiles mais antigos com um group legado único continuam funcionando. Connections Bots entram pelo picker New Group Chat e permanecem source-qualified no estado compartilhado da sala.

**Rooms seguem seus gateways, não um Desktop.** O transcript recente, members, picture e name de cada room são espelhados nos metadados de profile compartilhados de **todo** gateway ao qual seu Desktop está conectado, com versionamento por gateway para que dois Desktops escrevendo ao mesmo tempo façam merge em vez de sobrescrever um ao outro. Abra o Hermes Desktop em outra máquina contra o mesmo gateway (rede local, Tailscale, qualquer lugar) e a room aparece com o histórico; clientes só-gateway também a veem. Rooms carregam uma identidade interna durável, então renomear uma muda só o display name em todo lugar, dissolver uma remove permanentemente em todo cliente — mesmo os que estavam offline na hora — e recriar um group de mesmo nome começa uma room genuinamente nova. Se um gateway morre ou é removido, nada se perde: todo Desktop conectado guarda a room completa localmente e re-seeda qualquer gateway ao qual reconecta. (O log de orquestração completo fica no storage local de cada Desktop; o espelho compartilhado é uma projeção limitada de histórico recente.)

Groups são linhas standalone no mesmo roster ordenado por atividade dos DMs de Bot. Um Bot mantém uma linha de DM mesmo pertencendo a vários groups, enquanto cada group ganha a própria linha de room com contagem de members, preview da última mensagem, timestamp e estado needs-you.

**Open chat** em qualquer linha de group (2–6 Bots) abre uma sala compartilhada onde o group todo coordena:

- Sua mensagem dispara até **três rounds seriais** de turnos dos members. Bots @-mencionados respondem (todo mundo responde quando ninguém é mencionado); cada Bot responde brevemente ou passa, e a sala se acomoda quando um round completo fica em silêncio.
- Bots puxam uns aos outros com `@name`, e escalam decisões reais para você com `@user` — a linha do group mostra um badge **needs you** quando isso acontece.
- Caps rígidos (10 mensagens por send, 3 rounds) impedem as salas de girar sem parar.
- Cada member mantém a própria sessão persistente `Group: <name>`, então o contexto da sala sobrevive como qualquer outra conversa.
- **Nem todo Bot responde a toda mensagem.** Falar é escolha de cada member — um Bot só responde quando tem algo novo a acrescentar e passa caso contrário, e @-mencionar members específicos escopa o round a eles. Espere os members que você endereçou (ou quem tem algo a dizer) falarem, e o resto ficar quieto.
- **Rooms podem atravessar máquinas.** O picker New Group Chat acomoda Bots de qualquer conexão registrada; os turnos de cada member rodam na própria máquina, na própria sessão `Group: <name>` lá. Members cross-machine carregam um badge de device (`dixie · Mac Mini`) na sala e nos transcripts dos outros members, e o handle desambiguado `@name-device` funciona em mentions da sala — então agentes de mesmo nome em duas máquinas nunca se misturam.

## Mensagens bot-a-bot {#bot-to-bot-messaging}

Bots se mensagens com atribuição, e você pode passar trabalho de qualquer chat:

- **@mentions** — digite `@researcher have a look at this` em qualquer chat e o autocomplete `@` do composer ajuda a escolher o Bot certo; no send, a mention é resolvida contra o roster ao vivo e o Bot ativo é informado exatamente quem você quer dizer (profile, friendly name e device para Bots cross-connection). O Bot então compõe sua própria mensagem e envia com `message_agent` — seu texto nunca é encaminhado verbatim, e a resposta volta atribuída àquele agente. Um endereço de e-mail ou um `@` desconhecido passam intactos. Bots em outras máquinas conectadas são alcançáveis do mesmo jeito: o Desktop relaya a mensagem pelo socket próprio daquela conexão (veja *Bots entre máquinas* abaixo).
- **Bots renomeados mantêm as tags em sync** — dê a um Bot um nome amigável (o lápis no header do chat, ou `hermes profile rename`) e ele fica taggable por esse nome: um Bot intitulado *Research Buddy* responde a `@research-buddy` (e `@researchbuddy`), em chats regulares e em group rooms. O autocomplete `@` do composer oferece a tag renomeada e também casa quando você digita o nome antigo do profile, que continua resolvendo.
- **Direct messages** — todo Bot Chat carrega a ferramenta `message_agent`: um Bot mensageia um teammate chamando `message_agent(target="researcher", message="…")`. A ferramenta valida o target contra o roster ao vivo, prefixa automaticamente a atribuição `Message from 🤖 <sender> (@<sender>):` do sender, e entrega no Bot Chat canônico do teammate. A entrega é **fire-and-forget**: o sender recebe acknowledgement, termina seu turno, e a resposta chega depois como notificação de background completion. A mensagem viaja como parâmetro real (nada shell-interpreted — aspas, `$(...)` e backticks chegam verbatim), e o Bot compõe sua própria mensagem em vez de encaminhar suas palavras. O roster de teammates — nomes **e roles** do title/description de cada profile — faz parte do system prompt de todo Bot Chat, então Bots sabem quem faz o quê antes de escolher destinatário. A ferramenta existe **somente** em sessões Bot Chat canônicas em installs gerenciados por Bot Mode; chats regulares, sessões de member de group room e sessões CLI nunca a veem.

O backend ensina automaticamente o protocolo de messaging a cada sessão canônica de Bot Chat no build do prompt — inclusive quando um teammate a abre headless pela CLI. Só o Bot Chat canônico recebe a seção do protocolo; suas sessões regulares e seu SOUL.md ficam intactos. Isso é controlado por `agent.bot_mode_protocol` em `config.yaml` (padrão: ligado):

```yaml
agent:
  bot_mode_protocol: true   # inject the bot-to-bot messaging protocol into canonical Bot Chats
```

:::note
A entrega bot-a-bot é por invocação: o Bot receptor pega a mensagem quando rodar de novo. Interrupt live de um Bot no meio da conversa é trabalho futuro.
:::

### Turnos falhos retentam com segurança {#failed-turns-retry-safely}

Um turno de entrega falho é retentado no máximo uma vez, e só quando um retry pode de fato ajudar. Falhas transientes (target runtime offline, delivery timeout, rate limit ou server error do provider) re-executam a mesma sessão Bot Chat inalterada. Uma falha de context-overflow também re-executa a mesma sessão — o turno retentado compacta a transcrição acima do threshold via o passe padrão de context-compression antes de chamar o model, então o retry cabe onde o original não cabia. Falhas de auth, quota e configuração nunca auto-retentam: uma segunda tentativa não as corrige e só queima quota, então a falha é surfaced imediatamente. Um turno retentado nunca inicia sessão fresh — seu histórico e contexto Bot Chat permanecem intactos.

### Quando uma entrega falha: razões tipadas {#when-a-delivery-fails-typed-reasons}

Um turno de bot ou relay delivery falho carrega um code `reason` machine-readable junto ao texto de erro humano, end to end: o gateway alvo classifica a falha (`provider_auth_or_access`, `provider_quota_limit`, `provider_rate_limit`, `provider_server_error`, `context_overflow`, `missing_config`, `model_unavailable`, `runtime_offline`, `queued_expired`, `delivery_timeout`, `target_busy`, `unknown`), o Desktop encaminha, e a notificação de completion do agente caller é tagged `[reason: <code>]` antes do texto de erro. Um agente caller pode branch no code — "sign in again" vs "retry later" — em vez de parsear prose do provider. O badge needs-attention do Desktop usa os mesmos codes.

### Messaging entre máquinas conectadas (relay do Desktop) {#messaging-across-connected-machines-the-desktop-relay}

Todo gateway que você registra em **Settings → Connections** — local, remote URL, SSH, Hermes Cloud, docker — é uma linha persistente que o Desktop mantém aberta, e Bot Mode usa essas linhas para messaging automaticamente. Sem setup extra:

- **Rosters propagam sozinhos.** Enquanto o Desktop roda, ele periodicamente diz a cada gateway conectado quais agentes vivem nas *outras* conexões. O roster de teammates de todo Bot Chat então os lista ("Teammates on OTHER connected machines"), com nomes, roles e em qual máquina estão — e o roster refresca quando agentes aparecem, somem ou são renomeados (capability epoch).
- **`message_agent` alcança eles diretamente.** Um Bot no laptop mensageia o agente cloud com `message_agent(target="moxie", …)` exatamente como um teammate local. Se o mesmo handle existe em várias máquinas, desambigue com `target="moxie@<connection>"` (o error da ferramenta diz ao Bot as formas exatas). A entrega passa pelo Desktop: o gateway sender enfileira a mensagem, o Desktop relaya para o gateway da conexão alvo, o Bot alvo roda um turno no Bot Chat canônico, e a resposta volta ao sender como a mesma notificação de background completion dos DMs locais.
- **O Desktop é o courier.** Entrega cross-connection funciona enquanto um Desktop que conhece ambas as conexões está rodando (ele segura os sockets e as credenciais — gateways nunca veem auth um do outro). Se o Desktop fecha mid-delivery, o Bot sender é informado que a resposta não chegou em vez de ficar pendurado. Para messaging machine-to-machine always-on sem Desktop no loop, registre um peer (`hermes peer`, abaixo) — as duas rotas coexistem.

### DMs iniciados por Bot entre máquinas (`hermes peer`) {#bot-initiated-dms-across-machines-hermes-peer}

Bots numa máquina podem mensager Bots no gateway de **outra máquina** sem nenhum desktop no loop. Registre o outro gateway como *peer* (URL do API server + `API_SERVER_KEY`):

```bash
hermes peer add spark --url http://spark.lan:8377 --key <API_SERVER_KEY>
hermes peer list
hermes peer dm spark < /tmp/dm.txt        # message body from a file (nothing shell-interpreted)
hermes peer dm spark/researcher < /tmp/dm.txt   # named profile on a multiplexed peer

hermes peer run spark --idempotency-key ticket-123 < /tmp/long-task.txt
hermes peer status spark run_abc123
hermes peer stop spark run_abc123
```

`hermes peer dm` entrega no Bot Chat canônico do agente remoto pelo API server existente do peer, roda um turno de agente lá e imprime a resposta no stdout — o gêmeo cross-machine exato do comando local `hermes -p <bot> chat`.


Use `peer dm` só para queries curtas e receipts porque segura uma conexão HTTP
até o turno terminar. Para um turno longo, `peer run` retorna um
`run_id` imediatamente; faça poll com `peer status`. O run herda a
transcrição canônica do Bot Chat, e um `--idempotency-key` estável faz um retry
devolver o run original em vez de iniciar trabalho duplicado. Use `peer stop`
com aquele run ID exato para interrompê-lo sem mirar outro turno.

Uma vez que um peer está registrado, o protocolo de messaging ensinado a todo Bot Chat (`agent.bot_mode_protocol`) inclui automaticamente o roster de peers, e `message_agent` aceita targets peer diretamente — `message_agent(target="spark/researcher", …)`, ou `target="spark"` para o agente principal do peer — então **seus bots aprendem sozinhos** que teammates existem em outras máquinas e como alcançá-los. Registrar ou remover um peer refresca o protocolo de cada Bot Chat na próxima mensagem (capability epoch).

Requisitos: a máquina peer roda a plataforma de gateway `api_server` com um `API_SERVER_KEY` forte; reachability é assunto da sua rede (LAN, Tailscale, VPN). A key é uma credencial e vive em `~/.hermes/.env` como `HERMES_PEER_<NAME>_KEY`; nomes/URLs de peer vivem em `config.yaml` sob `bot_peers`.


:::note Alcance unidirecional (NAT)
Links cross-gateway são conexões diretas gateway-a-gateway — o Desktop é um
viewer, não um relay. Um gateway atrás de NAT residencial pode discar out para um peer público
(laptop → VPS funciona), mas a direção reversa não tem rota inbound
(VPS → home falha) a menos que sua rede forneça uma. Se seu Group Chat atravessa
um limite de NAT, coloque a autoridade da room no host que todo participante consegue
alcançar (tipicamente o VPS público), ou faça bridge da rede com Tailscale/VPN.
:::

## Bots entre máquinas {#bots-across-machines}

Quando você registra vários backends em **Settings → Connections** — o runtime local, gateways remotos, hosts SSH, instâncias Hermes Cloud — o roster mostra os Bots de **toda** source conectada, de forma persistente: sources SSH são inventariadas sem spawnar nada na caixa remota, e máquinas momentaneamente inalcançáveis mantêm as linhas last-known em vez de sumir. Quando o mesmo nome de profile existe em várias sources, os handles desambiguam como `@name-device` (por exemplo `@research-homelab`). Chats, sessões, memória e rotinas de um Bot vivem na máquina que é dona do profile.

Clicar num Connections Bot **não** pula sua janela para aquela máquina — fique no seu chat e `@mention` ele, coloque-o num group chat, ou crie agentes novos nele direto com o picker **Create on**. Agentes Cloud e locais compartilham um roster assim: registre sua instância Hermes Cloud e seu desktop (digamos, via Tailscale ou SSH) e os Bots deles podem se mensager e sentar nas mesmas rooms, com o trabalho de cada agente rodando na própria máquina. DMs bot-a-bot entre essas máquinas passam pelo relay do Desktop automaticamente (veja *Messaging entre máquinas conectadas* acima).

Veja [Conectando o Desktop a várias instâncias do Hermes](./multi-connection-desktop.md) para o guia completo multi-connection.

## Desligando {#turning-it-off}

Bot Mode é um plugin desktop bundled. Desligue em **Settings → Plugins → Bots** — o roster, o painel Routines e o middleware do composer se desregistram ao vivo, sem restart. Seus profiles, sessões e cron jobs ficam intactos de qualquer jeito; o Bot Mode nunca é dono dos seus dados, só os renderiza.

Há também uma preferência para esconder os Bot Chats canônicos da lista regular de sessões da sidebar, para eles só aparecerem dentro do painel Bots. (Isso usa a flag core de hidden-session; em gateways mais antigos os chats simplesmente continuam visíveis.)

## Paridade com a CLI {#cli-parity}

Como Bots são profiles, tudo tem um equivalente no terminal:

| No Bot Mode | Num shell |
| --- | --- |
| Conversar com um Bot | `hermes -p <bot> chat` |
| Files, skills, memória de um Bot | `~/.hermes/profiles/<bot>/` |
| Routines | `hermes cron list` (jobs nomeados `[bot:<name>] …`) |
| Criar / inspecionar profiles | `hermes profile create`, `hermes profile list` |

Veja [Profiles](./profiles.md) para o primitivo subjacente e [Comandos de Profile](../reference/profile-commands.md) para a referência completa da CLI.
