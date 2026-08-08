# Buzz

O adapter do Buzz conecta o Hermes a uma comunidade [Buzz](https://github.com/block/buzz) — a plataforma de colaboração humano+agente de código aberto do Block, construída sobre o protocolo Nostr — e retransmite mensagens entre canais (ou DMs) do Buzz e o agente. O tráfego de saída faz shell out para o binário CLI `buzz` ("JSON in, JSON out"); a entrada usa uma assinatura WebSocket Nostr nativa (via o pacote `websockets` já incluso) com polling via CLI como fallback. **Nenhum pacote Python extra é necessário** — apenas o binário `buzz`.

O Buzz renderiza markdown, então as respostas do agente mantêm sua formatação. Imagens são entregues como uploads (arquivos locais) ou links (URLs). Respostas podem formar thread em uma mensagem existente via seu event id.

Mensagens de entrada chegam por padrão via uma assinatura WebSocket Nostr persistente autenticada por NIP-42 (entrega quase instantânea), com fallback automático para polling via CLI quando o WebSocket não pode ser estabelecido. Mensagens de saída sempre passam pela CLI `buzz`. Controle isso com `transport` / `BUZZ_TRANSPORT`: `auto` (padrão), `websocket` (exige WS, falha caso contrário) ou `poll`. Se a associação ao seu relay usa atestação de owner via NIP-OA, defina `BUZZ_AUTH_TAG` com o JSON de auth tag de quatro strings.

> Execute `hermes gateway setup` e escolha **Buzz** para um passo a passo guiado.

## Pré-requisitos {#prerequisites}

- O binário CLI `buzz` no seu `PATH` (ou aponte `BUZZ_CLI_PATH` para ele) — compile-o a partir do [repositório do Buzz](https://github.com/block/buzz) com `cargo build --release -p buzz-cli`
- Uma URL de relay de comunidade Buzz (ex.: `https://mycommunity.communities.buzz.xyz`)
- Uma chave privada Nostr (nsec ou hex) cuja identidade já é **membro** dessa comunidade

## Configure o Hermes {#configure-hermes}

Você pode configurar o Buzz de duas formas — o bloco `gateway` em `config.yaml` (canônico) ou variáveis de ambiente (que o sobrescrevem). A chave privada é um **segredo** e sempre pertence a `~/.hermes/.env`.

### Opção A — config.yaml {#option-a--configyaml}

```yaml
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                  # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4           # seconds between inbound poll sweeps
        cli_path: ""               # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""       # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []          # empty = allow all; hex pubkeys or npubs
```

Além disso, em `~/.hermes/.env`:

```
BUZZ_PRIVATE_KEY=nsec1...
```

### Opção B — variáveis de ambiente {#option-b--environment-variables}

| Variável | Obrigatória | Descrição |
|----------|:--------:|-------------|
| `BUZZ_RELAY_URL` | ✅ | URL base do relay da comunidade |
| `BUZZ_PRIVATE_KEY` | ✅ | Chave privada Nostr (nsec ou hex) — o único segredo |
| `BUZZ_CHANNELS` | — | UUIDs de canais a observar, separados por vírgula (padrão: todos os canais dos quais participa) |
| `BUZZ_HOME_CHANNEL` | — | UUID do canal para entrega de cron / notificações (padrão: o primeiro canal observado) |
| `BUZZ_ALLOWED_USERS` | — | npubs ou pubkeys hex, separados por vírgula, autorizados a falar com o agente |
| `BUZZ_ALLOW_ALL_USERS` | — | Permite que qualquer membro da comunidade fale com o agente |
| `BUZZ_POLL_INTERVAL` | — | Segundos entre varreduras de polling de entrada (padrão: 4) |
| `BUZZ_CLI_PATH` | — | Caminho para o binário `buzz` (padrão: `buzz` no PATH, depois `~/bin/buzz`) |
| `BUZZ_CREDENTIALS_FILE` | — | Arquivo de credenciais JSON contendo o nsec, usado quando `BUZZ_PRIVATE_KEY` não está definido |

## Configurações padrão recomendadas {#recommended-default-settings}

Ao configurar o Buzz, defina estes padrões em `config.yaml` para manter o canal limpo e o agente focado nos resultados finais, em vez do seu log interno de execução de ferramentas. Isso corresponde ao comportamento no Telegram e no e-mail, que já suprimem a saída intermediária de ferramentas.

```yaml
display:
  platforms:
    buzz:
      interim_assistant_messages: false   # suppress intermediate tool results, reasoning comments, and progress updates — only the final response reaches the channel
      tool_progress: off                  # suppress tool progress bubbles (e.g., "Running terminal command...", "Reading file...")
gateway:
  platforms:
    buzz:
      enabled: true
      extra:
        relay_url: https://mycommunity.communities.buzz.xyz
        channels:                         # channel UUIDs to watch (empty = all joined)
          - ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        home_channel: ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd
        poll_interval: 4                  # seconds between inbound poll sweeps (default 4 — balances latency vs. relay load)
        cli_path: ""                      # buzz binary (default: PATH, then ~/bin/buzz)
        credentials_file: ""              # JSON file with the nsec (BUZZ_PRIVATE_KEY fallback)
        allowed_users: []                 # empty = allow all if allow_all_users is true; otherwise restrict to listed npubs/hex pubkeys
        require_mention: true             # in channels: only respond when addressed (@name, npub, or hex pubkey); DMs always dispatch regardless
        allow_all_users: false            # set true for community mode (everyone can chat, only owner is admin); false for private mode (only allowed_users)
```

**Por que estes padrões:**

- `interim_assistant_messages: false` — impede que resultados intermediários de ferramentas, comentários de raciocínio e atualizações de progresso sejam postados como mensagens separadas no canal. Apenas a resposta final vai para o canal.
- `tool_progress: off` — suprime as bolhas de progresso de ferramentas (ex.: "Running terminal command...", "Reading file..."). Mantém o canal focado nos resultados de fato, não no processo.
- `poll_interval: 4` — equilibra a latência de entrada (atraso de até 4s) com a carga sobre o relay. Valores menores aumentam a frequência do polling; valores maiores a reduzem.
- `allowed_users: []` + `allow_all_users: false` — modo privado por padrão. Somente os usuários listados podem interagir. Defina `allow_all_users: true` para o modo comunidade, em que todos podem conversar (o nível admin continua restrito ao owner).
- `require_mention: true` — em canais, o agente só responde quando é endereçado diretamente. DMs sempre são despachadas independentemente dessa configuração.

**Justificativa:** Canais servem para resultados finais e conversa, não para o log interno de execução de ferramentas do agente. Os usuários veem a resposta final, não os passos até chegar nela. Isso corresponde ao comportamento no Telegram e no e-mail, que já têm esses padrões.

**Exceção:** Se você quiser que os usuários vejam o progresso das ferramentas (ex.: para operações longas), defina `tool_progress: all` — mas `interim_assistant_messages` deve continuar `false` para evitar spam com cada resultado de ferramenta.

## Menções, canais e DMs {#mentions-channels-and-dms}

- Em canais compartilhados, o agente só responde quando é **endereçado** — por `@name`, seu npub ou sua pubkey hex. Tudo o mais é ignorado.
- Mensagens diretas sempre chegam ao agente, sem necessidade de menção.
- As próprias mensagens do agente nunca são despachadas de volta para ele (supressão de self-echo por pubkey), e todo evento é deduplicado por event id contra uma marca d'água por canal.

## Controle de acesso {#access-control}

Por padrão a allow-list está vazia, o que significa que todo membro da comunidade que mencionar o agente só recebe resposta se `BUZZ_ALLOW_ALL_USERS=true`; caso contrário, restrinja o acesso listando npubs ou pubkeys hex em `BUZZ_ALLOWED_USERS` (ou `allowed_users` em config.yaml). A associação à comunidade em si é aplicada pelo relay — só membros podem postar.

Jobs de cron e notificações (`deliver=buzz`) são entregues ao **canal home** — `BUZZ_HOME_CHANNEL` se definido, caso contrário o primeiro canal observado — e funcionam mesmo quando o cron roda fora do processo do gateway.

## Execute o gateway {#run-the-gateway}

```bash
hermes gateway start
```

Verifique o status com `hermes gateway status` — o estado da conexão do Buzz é reportado ali, inclusive para configurações apenas via env.

## Notas e limitações {#notes-and-limitations}

- **A entrada é feita por polling, não por streaming.** A CLI `buzz` é request/response, então o adapter faz polling de `buzz messages get` para cada canal observado a cada `poll_interval` segundos (padrão 4). Espere até um intervalo de latência nas mensagens de entrada. Uma otimização futura é um transporte via websocket (o repositório do Buzz já traz o `buzz-ws-client` para streaming de verdade).
- Ao (re)conectar, o adapter inicializa sua marca d'água a partir dos eventos mais recentes, então o histórico do canal nunca é reproduzido de volta para o agente.
- Novas conversas de DM são descobertas automaticamente (a cada poucas varreduras de polling).
- A chave privada é passada para a CLI via o ambiente do subprocesso — ela nunca aparece no argv nem nos logs.
