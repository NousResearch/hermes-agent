# Spotify

O Hermes pode controlar o Spotify diretamente — playback, fila, busca, playlists, faixas/álbuns salvos e histórico de escuta — usando a Web API oficial do Spotify com OAuth PKCE. Tokens ficam em `~/.hermes/auth.json` e são renovados automaticamente em 401; você só faz login uma vez por máquina (refresh tokens expiram após ~6 meses; execute `hermes auth spotify` de novo quando expirarem).

Diferente das integrações OAuth built-in do Hermes (Google, GitHub Copilot, Codex), o Spotify exige que cada usuário registre seu próprio app de desenvolvedor leve. O Spotify não permite que terceiros enviem um app OAuth público que qualquer um possa usar. Leva cerca de dois minutos e `hermes auth spotify` guia você.

## Pré-requisitos {#prerequisites}

- Uma conta Spotify. **Free** funciona para busca, playlist, biblioteca e ferramentas de atividade. **Premium** é necessário para controle de playback (play, pause, skip, seek, volume, adicionar à fila, transfer).
- Hermes Agent instalado e em execução.
- Para ferramentas de playback: um **dispositivo Spotify Connect ativo** — o app Spotify deve estar aberto em pelo menos um dispositivo (celular, desktop, web player, speaker) para a Web API ter algo para controlar. Se nada estiver ativo você receberá `403 Forbidden` com mensagem "no active device"; abra o Spotify em qualquer dispositivo e tente de novo.

## Setup {#setup}

### Em uma passagem: `hermes tools` ou setup de primeira execução {#one-shot-hermes-tools-or-first-run-setup}

O caminho mais rápido. Execute:

```bash
hermes tools
```

Role até `🎵 Spotify`, pressione space para ligar, depois `s` para salvar. O mesmo toggle também está disponível durante o fluxo `hermes setup` / `hermes setup tools` de primeira execução. Spotify continua opt-in, então habilitá-lo ali roda a mesma configuração consciente de provider que `hermes tools`.

O Hermes joga você direto no fluxo OAuth — se ainda não tiver um app Spotify, guia a criação inline. Quando terminar, o toolset está habilitado E autenticado numa passagem.

Se preferir fazer os passos separadamente (ou reautenticar depois), use o fluxo em duas etapas abaixo.

### Fluxo em duas etapas {#two-step-flow}

#### 1. Habilitar o toolset

```bash
hermes tools
```

Ligue `🎵 Spotify`, salve e, quando o wizard inline abrir, feche (Ctrl+C). O toolset permanece ligado; só a etapa de auth fica adiada.

#### 2. Executar o wizard de login

```bash
hermes auth spotify
```

As 7 ferramentas Spotify só aparecem no toolset do agente após a etapa 1 — estão desligadas por padrão para usuários que não as querem não enviarem schemas extras de ferramenta a cada chamada de API.

Se `HERMES_SPOTIFY_CLIENT_ID` não estiver definido, o Hermes guia o registro do app inline:

1. Abre `https://developer.spotify.com/dashboard` no seu navegador
2. Imprime os valores exatos para colar no formulário "Create app" do Spotify
3. Pede o Client ID que você recebe de volta
4. Salva em `~/.hermes/.env` para execuções futuras pularem este passo
5. Continua direto no fluxo de consentimento OAuth

Depois que você aprovar, tokens são gravados sob `providers.spotify` em `~/.hermes/auth.json`. O provider de inferência ativo NÃO muda — auth Spotify é independente do seu provider LLM.

### Criar o app Spotify (o que o wizard pede) {#creating-the-spotify-app}

Quando o dashboard abrir, clique **Create app** e preencha:

| Campo | Valor |
|-------|-------|
| App name | qualquer coisa (ex.: `hermes-agent`) |
| App description | qualquer coisa (ex.: `personal Hermes integration`) |
| Website | deixe em branco |
| Redirect URI | `http://127.0.0.1:43827/spotify/callback` |
| Which API/SDKs? | marque **Web API** |

Aceite os termos e clique **Save**. Na página seguinte clique **Settings** → copie o **Client ID** e cole no prompt do Hermes. Esse é o único valor que o Hermes precisa — PKCE não usa client secret.

### Rodando via SSH / em ambiente headless {#running-over-ssh-in-a-headless-environment}

Se `SSH_CLIENT` ou `SSH_TTY` estiver definido, o Hermes pula a abertura automática do navegador tanto no wizard quanto na etapa OAuth. Copie a URL do dashboard e a URL de autorização que o Hermes imprime, abra no navegador da sua máquina local e prossiga normalmente — o listener HTTP local ainda roda no host remoto na porta `43827`. O navegador do laptop não alcança o loopback remoto sem forward local SSH:

```bash
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

Para jump-box / bastion e outras armadilhas (mosh, tmux, conflitos de porta), veja [OAuth over SSH / Remote Hosts](../../guides/oauth-over-ssh.md).

## Verificar {#verify}

```bash
hermes auth status spotify
```

Mostra se tokens estão presentes e quando o access token expira. Refresh é automático: quando qualquer chamada Spotify retorna 401, o client troca o refresh token e tenta uma vez. Refresh tokens persistem entre reinícios do Hermes, então você só reautentica se revogar o app nas configurações da conta Spotify ou executar `hermes auth logout spotify`.

## Usando {#using-it}

Depois de logado, o agente tem acesso a 7 ferramentas Spotify. Você fala com o agente naturalmente — ele escolhe a ferramenta e ação certas. Para o melhor comportamento, o agente carrega uma skill companion que ensina padrões canônicos de uso (single-search-then-play, quando não fazer preflight `get_state`, etc.).

```
> play some miles davis
> what am I listening to
> add this track to my Late Night Jazz playlist
> skip to the next song
> make a new playlist called "Focus 2026" and add the last three songs I played
> which of my saved albums are by Radiohead
> search for acoustic covers of Blackbird
> transfer playback to my kitchen speaker
```

### Referência de ferramentas {#tool-reference}

Todas as ações que mutam playback aceitam um `device_id` opcional para mirar um dispositivo específico. Se omitido, o Spotify usa o dispositivo atualmente ativo.

#### `spotify_playback`
Controla e inspeciona playback, além de buscar histórico recently played.

| Ação | Propósito | Premium? |
|--------|---------|----------|
| `get_state` | Estado completo de playback (faixa, dispositivo, progresso, shuffle/repeat) | Não |
| `get_currently_playing` | Só a faixa atual (retorna vazio em 204 — veja abaixo) | Não |
| `play` | Iniciar/retomar playback. Opcional: `context_uri`, `uris`, `offset`, `position_ms` | Sim |
| `pause` | Pausar playback | Sim |
| `next` / `previous` | Pular faixa | Sim |
| `seek` | Ir para `position_ms` | Sim |
| `set_repeat` | `state` = `track` / `context` / `off` | Sim |
| `set_shuffle` | `state` = `true` / `false` | Sim |
| `set_volume` | `volume_percent` = 0-100 | Sim |
| `recently_played` | Últimas faixas tocadas. Opcional: `limit`, `before`, `after` (Unix ms) | Não |

#### `spotify_devices`
| Ação | Propósito |
|--------|---------|
| `list` | Todo dispositivo Spotify Connect visível na sua conta |
| `transfer` | Move playback para `device_id`. Opcional `play: true` inicia playback na transferência |

### Alto-falantes gerenciados pelo Home Assistant {#home-assistant-managed-speakers}

Se o Home Assistant gerencia alto-falantes que já suportam Spotify Connect (por exemplo Sonos, Echo, Nest ou outros com Connect), eles aparecem em `spotify_devices list` automaticamente quando o Spotify os vê. O Hermes não precisa de ponte Home Assistant ↔ Spotify neste caminho — o Spotify roteia dispositivos nativamente.

Peça ao Hermes para transferir playback pelo nome de exibição do speaker (por exemplo, "transfer Spotify to the kitchen speaker"), ou chame `spotify_devices list` e passe o `device_id` exato para `spotify_devices transfer` ao scriptar. Se o speaker estiver ausente, abra o app Spotify ou a integração Spotify do speaker uma vez para o Spotify registrá-lo como alvo Connect ativo.

#### `spotify_queue`
| Ação | Propósito | Premium? |
|--------|---------|----------|
| `get` | Faixas atualmente na fila | Não |
| `add` | Anexa `uri` à fila | Sim |

#### `spotify_search`
Busca no catálogo. `query` é obrigatório. Opcional: `types` (array de `track` / `album` / `artist` / `playlist` / `show` / `episode`), `limit`, `offset`, `market`.

#### `spotify_playlists`
| Ação | Propósito | Args obrigatórios |
|--------|---------|---------------|
| `list` | Playlists do usuário | — |
| `get` | Uma playlist + faixas | `playlist_id` |
| `create` | Nova playlist | `name` (+ opcional `description`, `public`, `collaborative`) |
| `add_items` | Adicionar faixas | `playlist_id`, `uris` (opcional `position`) |
| `remove_items` | Remover faixas | `playlist_id`, `uris` (+ opcional `snapshot_id`) |
| `update_details` | Renomear / editar | `playlist_id` + qualquer um de `name`, `description`, `public`, `collaborative` |

#### `spotify_albums`
| Ação | Propósito | Args obrigatórios |
|--------|---------|---------------|
| `get` | Metadados do álbum | `album_id` |
| `tracks` | Lista de faixas do álbum | `album_id` |

#### `spotify_library`
Acesso unificado a faixas e álbuns salvos. Escolha a coleção com o arg `kind`.

| Ação | Propósito |
|--------|---------|
| `list` | Listagem paginada da biblioteca |
| `save` | Adiciona `ids` / `uris` à biblioteca |
| `remove` | Remove `ids` / `uris` da biblioteca |

Obrigatório: `kind` = `tracks` ou `albums`, mais `action`.

### Matriz de recursos: Free vs Premium {#feature-matrix-free-vs-premium}

Ferramentas read-only funcionam em contas Free. Qualquer coisa que muta playback ou a fila exige Premium.

| Funciona no Free | Exige Premium |
|---------------|------------------|
| `spotify_search` (tudo) | `spotify_playback` — play, pause, next, previous, seek, set_repeat, set_shuffle, set_volume |
| `spotify_playback` — get_state, get_currently_playing, recently_played | `spotify_queue` — add |
| `spotify_devices` — list | `spotify_devices` — transfer |
| `spotify_queue` — get | |
| `spotify_playlists` (tudo) | |
| `spotify_albums` (tudo) | |
| `spotify_library` (tudo) | |

## Agendamento: Spotify + cron {#scheduling-spotify-cron}

Como ferramentas Spotify são ferramentas Hermes normais, um job cron rodando numa sessão Hermes pode disparar playback em qualquer agenda. Sem código novo.

### Playlist matinal de despertar {#morning-wake-up-playlist}

```bash
hermes cron add \
  --name "morning-commute" \
  "0 7 * * 1-5" \
  "Transfer playback to my kitchen speaker and start my 'Morning Commute' playlist. Volume to 40. Shuffle on."
```

O que acontece às 7h todo dia útil:
1. Cron sobe uma sessão Hermes headless.
2. Agente lê o prompt, chama `spotify_devices list` para achar "kitchen speaker" pelo nome, depois `spotify_devices transfer` → `spotify_playback set_volume` → `spotify_playback set_shuffle` → `spotify_search` + `spotify_playback play`.
3. Música começa no speaker alvo. Custo total: uma sessão, poucas tool calls, sem input humano.

### Wind-down à noite {#wind-down-at-night}

```bash
hermes cron add \
  --name "wind-down" \
  "30 22 * * *" \
  "Pause Spotify. Then set volume to 20 so it's quiet when I start it again tomorrow."
```

### Armadilhas {#gotchas}

- **Um dispositivo ativo deve existir quando o cron dispara.** Se nenhum client Spotify estiver rodando (celular/desktop/speaker Connect), ações de playback retornam `403 no active device`. Para playlists matinais, o truque é mirar um dispositivo sempre ligado (Sonos, Echo, smart speaker) em vez do celular.
- **Premium necessário para qualquer coisa que muta playback** — play, pause, skip, volume, transfer. Jobs cron read-only (agendados "me mande minhas faixas recently played por email") funcionam bem no Free.
- **O agente cron herda seus toolsets ativos.** Spotify deve estar habilitado em `hermes tools` para a sessão cron ver as ferramentas Spotify.
- **Jobs cron rodam com `skip_memory=True`** então não escrevem na sua memory store.

Referência completa de cron: [Cron Jobs](./cron).

## Sair {#sign-out}

```bash
hermes auth logout spotify
```

Remove tokens de `~/.hermes/auth.json`. Para limpar também o config do app, delete `HERMES_SPOTIFY_CLIENT_ID` (e `HERMES_SPOTIFY_REDIRECT_URI` se definiu) de `~/.hermes/.env`, ou execute o wizard de novo.

Para revogar o app no lado Spotify, visite [Apps connected to your account](https://www.spotify.com/account/apps/) e clique **REMOVE ACCESS**.

## Solução de problemas {#troubleshooting}

**`403 Forbidden — Player command failed: No active device found`** — Você precisa do Spotify rodando em pelo menos um dispositivo. Abra o app Spotify no celular, desktop ou web player, inicie qualquer faixa por um segundo para registrá-lo e tente de novo. `spotify_devices list` mostra o que está visível agora.

**`403 Forbidden — Premium required`** — Você está numa conta Free tentando uma ação que muta playback. Veja a matriz de recursos acima.

**`204 No Content` em `get_currently_playing`** — nada está tocando em nenhum dispositivo. Resposta normal do Spotify, não é erro; o Hermes expõe como resultado vazio explicativo (`is_playing: false`).

**`INVALID_CLIENT: Invalid redirect URI`** — a redirect URI nas configurações do app Spotify não corresponde ao que o Hermes usa. O padrão é `http://127.0.0.1:43827/spotify/callback`. Adicione isso às redirect URIs permitidas do app, ou defina `HERMES_SPOTIFY_REDIRECT_URI` em `~/.hermes/.env` para o que você registrou.

**`429 Too Many Requests`** — rate limit do Spotify. O Hermes retorna erro amigável; espere um minuto e tente de novo. Se persistir, provavelmente está rodando loop apertado num script — a cota do Spotify reseta em ~30 segundos.

**`401 Unauthorized` continua voltando** — Seu refresh token foi revogado (geralmente porque removeu o app da conta ou o app foi deletado). Execute `hermes auth spotify` de novo.

**Wizard não abre o navegador** — Se estiver via SSH ou em container sem display, o Hermes detecta e pula auto-open. Copie a URL do dashboard que imprime e abra manualmente.

## Avançado: scopes customizados {#advanced-custom-scopes}

Por padrão o Hermes pede os scopes necessários para toda ferramenta enviada. Sobrescreva se quiser restringir acesso:

```bash
hermes auth spotify --scope "user-read-playback-state user-modify-playback-state playlist-read-private"
```

Referência de scopes: [Spotify Web API scopes](https://developer.spotify.com/documentation/web-api/concepts/scopes). Se pedir menos scopes do que uma ferramenta precisa, as chamadas dessa ferramenta falham com 403.

## Avançado: client ID / redirect URI customizados {#advanced-custom-client-id-redirect-uri}

```bash
hermes auth spotify --client-id <id> --redirect-uri http://localhost:3000/callback
```

Ou defina permanentemente em `~/.hermes/.env`:

```
HERMES_SPOTIFY_CLIENT_ID=<your_id>
HERMES_SPOTIFY_REDIRECT_URI=http://localhost:3000/callback
```

A redirect URI deve estar na allow-list nas configurações do app Spotify. O padrão serve para quase todos — só mude se a porta 43827 estiver ocupada.

## Onde as coisas ficam {#where-things-live}

| Arquivo | Conteúdo |
|------|----------|
| `~/.hermes/auth.json` → `providers.spotify` | access token, refresh token, expiry, scope, redirect URI |
| `~/.hermes/.env` | `HERMES_SPOTIFY_CLIENT_ID`, opcional `HERMES_SPOTIFY_REDIRECT_URI` |
| Spotify app | de sua propriedade em [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard); contém o Client ID e a allow-list de redirect URI |
