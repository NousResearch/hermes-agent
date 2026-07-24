---
sidebar_position: 4
---

# Executando vários gateways ao mesmo tempo

Opere vários [profiles](./profiles.md) — cada um com seus próprios tokens de bot,
sessões e memória — como serviços gerenciados em uma única máquina. Esta página
cobre as preocupações operacionais: iniciá-los todos juntos, visualizar logs
entre profiles, impedir que o host entre em suspensão e recuperar-se de
peculiaridades comuns de launchd/systemd.

Se você executa apenas um agente Hermes, não precisa desta página — veja
[Profiles](./profiles.md) para o básico.

## Quando usar

Você quer este setup quando tem dois ou mais agentes Hermes que devem ficar
online ao mesmo tempo. Motivos comuns:

- Um assistente pessoal em um bot Telegram e um agente de código em outro
- Um agente por membro da família ou um por workspace Slack
- Instâncias sandbox + produção da mesma configuração
- Um agente de pesquisa + um de escrita + um bot acionado por cron — cada um com memória
  e skills isoladas

Todo profile já recebe seu próprio LaunchAgent por plataforma
(`ai.hermes.gateway-<name>.plist`) ou serviço systemd de usuário
(`hermes-gateway-<name>.service`). Este guia adiciona os padrões para gerenciá-los
coletivamente.

## Início rápido

```bash
# Criar profiles (uma vez)
hermes profile create coder
hermes profile create personal-bot
hermes profile create research

# Configurar cada um
coder setup
personal-bot setup
research setup

# Instalar cada gateway como serviço gerenciado
coder gateway install
personal-bot gateway install
research gateway install

# Iniciar todos
coder gateway start
personal-bot gateway start
research gateway start
```

Pronto — três agentes independentes, cada um em seu próprio processo, reiniciando
automaticamente em crash e no login do usuário.

## Alternativa: um gateway para todos os profiles (multiplexação)

O modelo acima executa **um processo por profile**. Esse é o padrão e a
escolha certa para a maioria dos setups. Mas em um host com muitos profiles — ou uma
implantação em container em que um processo por profile é pesado operacionalmente — você
pode executar um **único gateway multiplexador**: o gateway do profile default
torna-se o único processo de entrada e atende mensagens para *todos* os profiles na
máquina.

Isso é **opt-in** e **desligado por padrão**. Quando está off, nada nesta página
muda — todo comportamento abaixo fica inerte.

### Quando preferir multiplexação

- Implantação container/VPS em que N units de supervisor, N portas e N PID files
  são um fardo.
- Muitos profiles de baixo tráfego que não justificam cada um um processo completo.
- Você quer uma única coisa para iniciar, monitorar e reiniciar.

Fique com um-processo-por-profile quando quiser isolamento rígido em nível de processo
entre profiles (footprints de memória separados, domínios de crash independentes, a
capacidade de reiniciar um profile sem tocar nos outros).

### Como ativar

Defina a flag no **profile default** (ele possui o multiplexador) e reinicie
seu gateway:

```bash
hermes config set gateway.multiplex_profiles true
hermes gateway restart
```

Equivalentemente, no `~/.hermes/config.yaml` do profile default:

```yaml
gateway:
  multiplex_profiles: true
```

(A flag também é aceita como `multiplex_profiles: true` no top-level por
conveniência.) Na próxima inicialização, o gateway default enumera todo profile,
sobe as plataformas habilitadas de cada profile sob as credenciais
próprias daquele profile e roteia cada mensagem de entrada para o profile a que pertence. Cada
turn resolve config, skills, memória, SOUL **e chaves de provedor** do profile roteado —
credenciais nunca são compartilhadas entre profiles.

Você **não** executa `hermes gateway start` para profiles secundários — o
gateway default os atende. Veja as mudanças de contrato abaixo.

### O que muda com multiplexação ligada

Habilitar a flag altera o comportamento de algumas coisas. Tudo isso reverte no
momento em que a flag é desligada.

#### 1. Profiles secundários não devem iniciar gateway próprio

Com um multiplexador em execução, `hermes gateway start` / `run` de um profile nomeado é um
**erro fatal**, apontando de volta ao multiplexador:

```
The default gateway is running as a profile multiplexer and already serves
profile 'coder'. ...
```

O multiplexador é o único processo de entrada; um segundo gateway de profile
duplicaria o bind das plataformas daquele profile. Passe `--force` apenas se deliberadamente
quiser um processo separado para aquele profile (não recomendado com o multiplexador
em execução). O script wrapper de lifecycle cross-profile mais adiante nesta página, portanto, **não** é
usado em modo multiplex — você só gerencia o gateway default.

#### 2. Plataformas HTTP-inbound são alcançadas via prefixo de URL `/p/<profile>/`

Tráfego webhook (e outras plataformas HTTP-inbound) para um profile secundário chega no
listener default sob um prefixo de profile, **não** em uma segunda porta:

```
# profile default
POST http://host:8644/webhooks/<route>
# profile "coder", mesmo listener
POST http://host:8644/p/coder/webhooks/<route>
```

Um profile desconhecido ou não configurado no prefixo retorna `404`. Como o
listener compartilhado já atende todo profile assim, um **profile secundário
não deve habilitar uma plataforma que faz bind de porta** — fazer isso é erro de config
que pula o profile secundário inteiro enquanto default e outros profiles saudáveis
continuam. O aviso nomeia o profile ignorado e toda plataforma conflitante:

```
Skipping secondary profile 'coder' due to port-binding config error: Profile
'coder' enables port-binding platform(s) webhook, but gateway.multiplex_profiles
is on. ... Remove these platform entries from profile 'coder's config.yaml or
configure them only on the default profile.
```

Plataformas com bind de porta cobertas por esta regra: `webhook`, `api_server`,
`msgraph_webhook`, `feishu`, `wecom_callback`, `bluebubbles`, `sms`,
`whatsapp_cloud`, `line`. Configure qualquer uma delas **apenas no profile default**;
todo profile é alcançável pelo prefixo `/p/<profile>/`.

Apenas este conflito de listener compartilhado degrada para profile ignorado. Erros de config
de segurança permanecem fatais: por exemplo, uma plataforma own-policy `open`
sem `GATEWAY_ALLOW_ALL_USERS` ou opt-in allow-all específico da plataforma
ainda aborta a inicialização do gateway em vez de silenciosamente descartar o profile inseguro.

#### 3. Plataformas por credencial ainda precisam de token próprio por profile

Plataformas de polling/conexão (Telegram, Discord, Slack, Matrix, Signal, …) funcionam
bem multiplexadas, mas cada profile que habilita uma deve fornecer seu **próprio** token de bot
— o mesmo token não pode ser polled por dois profiles ao mesmo tempo. Se dois profiles
configuram o mesmo `(platform, token)`, a inicialização falha rápido nomeando ambos os profiles
(veja [Segurança contra conflito de token](#seguranca-contra-conflito-de-token) — a regra não mudou,
só é aplicada dentro do único processo agora).

#### 4. Session keys são namespaced por profile

As sessões de cada profile ficam sob namespace `agent:<profile>:…` para que dois
profiles na mesma plataforma/chat nunca colidam no session store compartilhado.
O profile **default** mantém o namespace histórico `agent:main:…`
byte-a-byte, então sessões existentes do profile default não são afetadas — sem
migração, sem histórico órfão.

#### 5. Um PID/lock e uma superfície de status

Há um único PID e lock em nível de processo (o multiplexador, sob o home
default). `hermes status` reporta o multiplexador e os profiles que atende;
`hermes status -p <name>` fatia para um profile. Cada profile ainda grava seu
próprio `runtime_status.json` sob seu próprio home, então leitores per-profile existentes
continuam funcionando.

#### O que **não** muda

O isolamento de credenciais `.env` por profile é preservado e, se algo,
mais rígido: as chaves de um profile são resolvidas do escopo próprio e nunca unidas
em um ambiente compartilhado (isso também significa que subprocessos como servidores MCP e
workers Kanban só veem os segredos do próprio profile). Kanban,
skills/memória/SOUL com escopo de profile e roteamento de modelo se comportam por profile
exatamente como com gateways separados.

### Roteando chats de bot compartilhado para profiles (`profile_routes`)

A multiplexação seleciona um profile por **credencial** (token de bot próprio de cada profile)
ou por **prefixo de URL** (`/p/<profile>/` para plataformas HTTP). Quando várias
comunidades compartilham **um** token de bot — por exemplo um bot Discord servindo muitas
guilds — você pode adicionalmente rotear guilds/canais/threads específicos para
profiles diferentes com `gateway.profile_routes`:

```yaml
gateway:
  multiplex_profiles: true
  profile_routes:
    # Um servidor Discord inteiro → um profile
    - name: acme-server
      platform: discord
      guild_id: "1234567890"
      profile: acme

    # Um canal naquele servidor → profile diferente
    - name: acme-support
      platform: discord
      guild_id: "1234567890"
      chat_id: "9876543210"
      profile: acme-support

    # Um grupo Telegram (sem conceito de guild — só chat_id)
    - name: tg-group
      platform: telegram
      chat_id: "-1001234567890"
      profile: tg-profile
```

Rotas são casadas da mais específica para a menos (`thread_id` > `chat_id` > `guild_id`),
todos os campos declarados devem valer (AND), e uma rota keyed em um canal também
casa threads/forum posts cujo pai é aquele canal. Mensagens sem rota ficam no profile default/ativo. O profile roteado recebe o isolamento per-profile completo descrito acima (config, skills, memória, credenciais,
namespace de sessão). Roteamento funciona em todo adapter de plataforma, não só Discord.

`profile_routes` exige `gateway.multiplex_profiles: true`; com
multiplexação off as rotas são ignoradas. Se uma rota nomeia um profile que não
existe em disco, o gateway registra aviso nomeando profile e origem e
cai back para o home default.

## Iniciar, parar ou reiniciar todos os gateways de uma vez

A CLI inclui comandos de lifecycle de profile único. Para agir em todo
profile, envolva-os em um loop shell. Coloque o snippet abaixo em
`~/.local/bin/hermes-gateways` e `chmod +x`:

```sh
#!/bin/sh
set -eu

# Adicione ou remova nomes de profile aqui conforme criar / excluir profiles.
profiles="default coder personal-bot research"

usage() {
  echo "Usage: hermes-gateways {start|stop|restart|status|list}"
}

run_for_profile() {
  profile="$1"
  action="$2"
  if [ "$profile" = "default" ]; then
    hermes gateway "$action"
  else
    hermes -p "$profile" gateway "$action"
  fi
}

action="${1:-}"
case "$action" in
  start|stop|restart|status)
    for profile in $profiles; do
      echo "==> $action $profile"
      run_for_profile "$profile" "$action"
    done
    ;;
  list)
    hermes gateway list
    ;;
  *)
    usage
    exit 2
    ;;
esac
```

Então:

```bash
hermes-gateways start      # iniciar todo profile configurado
hermes-gateways stop       # parar todo profile configurado
hermes-gateways restart    # reiniciar todos
hermes-gateways status     # status em todos
hermes-gateways list       # delega para `hermes gateway list`
```

:::tip
O profile `default` é alvo com `hermes gateway <action>` (sem `-p`),
não `hermes -p default gateway <action>`. O wrapper acima trata ambas as formas.
:::

## Gerenciar um profile

Os atalhos que todo profile instala:

```bash
coder gateway run        # foreground (Ctrl-C para parar)
coder gateway start      # iniciar o serviço gerenciado
coder gateway stop       # parar o serviço gerenciado
coder gateway restart    # reiniciar
coder gateway status     # status
coder gateway install    # criar LaunchAgent / unit systemd
coder gateway uninstall  # remover o arquivo de serviço
```

São equivalentes a `hermes -p coder gateway <action>` — útil se um
alias de profile não está no `PATH` ou se você alveja profiles dinamicamente de um
script.

## Arquivos de serviço

Cada profile instala seu próprio serviço com nome único, então instalações
nunca colidem:

| Platform | Path                                                              |
| -------- | ----------------------------------------------------------------- |
| macOS    | `~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist`        |
| Linux    | `~/.config/systemd/user/hermes-gateway-<profile>.service`         |

O profile default mantém os nomes históricos: `ai.hermes.gateway.plist` /
`hermes-gateway.service`.

## Visualizando logs

Cada profile grava em seus próprios arquivos de log:

```bash
# Profile default
tail -f ~/.hermes/logs/gateway.log
tail -f ~/.hermes/logs/gateway.error.log

# Profile nomeado
tail -f ~/.hermes/profiles/<name>/logs/gateway.log
tail -f ~/.hermes/profiles/<name>/logs/gateway.error.log
```

Transmita o log de todo profile simultaneamente:

```bash
tail -f ~/.hermes/logs/gateway.log ~/.hermes/profiles/*/logs/gateway.log
```

A CLI também tem um visualizador estruturado de logs:

```bash
hermes logs -f                  # seguir profile default
hermes -p coder logs -f         # seguir um profile
hermes logs --help              # filtros, níveis, saída JSON
```

## Identificar o que está realmente em execução

```bash
hermes profile list             # profiles + model + estado do gateway
hermes-gateways status          # status completo em todo profile
launchctl list | grep hermes    # macOS — PIDs e labels
systemctl --user list-units 'hermes-gateway-*'   # Linux — units
```

## Editando configuração

Todo profile mantém sua config dentro do próprio diretório:

```
~/.hermes/profiles/<name>/
├── .env              # API keys, bot tokens (chmod 600)
├── config.yaml       # model, provider, toolsets, gateway settings
└── SOUL.md           # personalidade / system prompt
```

O profile default usa `~/.hermes/` diretamente com os mesmos três arquivos.

Edite com qualquer editor ou via CLI:

```bash
hermes config set model.model anthropic/claude-sonnet-4    # profile default
coder config set model.model openai/gpt-5                  # profile nomeado
```

Depois de editar `.env` ou `config.yaml`, reinicie o gateway afetado:

```bash
coder gateway restart
# ou, para tudo:
hermes-gateways restart
```

## Mantendo o host acordado

O processo gateway pode rodar o dia todo, mas o sistema operacional ainda tentará
dormir quando ocioso. Dois padrões:

### macOS — `caffeinate`

`caffeinate` vem no macOS e impede suspensão enquanto roda. Sem instalação.

```bash
caffeinate -dis                    # bloquear display, idle e system sleep
caffeinate -dis -t 28800           # idem, auto-exit após 8 horas
caffeinate -i -w $(cat ~/.hermes/gateway.pid) &   # acordado enquanto gateway default roda

# Persistente: rodar em background e esquecer
nohup caffeinate -dis >/dev/null 2>&1 &
disown

# Inspecionar / parar
pmset -g assertions | grep -iE 'caffeinate|prevent|user is active'
pkill caffeinate
```

| Flag   | Effect                                            |
| ------ | ------------------------------------------------- |
| `-d`   | block display sleep                               |
| `-i`   | block idle system sleep (default)                 |
| `-m`   | block disk sleep                                  |
| `-s`   | block system sleep (AC-powered Macs only)         |
| `-u`   | simulate user activity (prevents screen lock)     |
| `-t N` | auto-exit after `N` seconds                       |
| `-w P` | exit when PID `P` exits                           |

:::warning Fechar a tampa ainda suspende o Mac
`caffeinate` não pode sobrescrever a suspensão por fechar tampa acionada por hardware em MacBooks.
Para operação com tampa fechada, altere preferências de Energy Saver / Battery ou
use ferramenta de terceiros.
:::

### Linux — `systemd-inhibit` ou `loginctl`

```bash
# Inibir suspend enquanto um comando roda
systemd-inhibit --what=idle:sleep --who=hermes --why="gateways running" \
  sleep infinity &

# Permitir que serviços de usuário continuem após logout (recomendado)
sudo loginctl enable-linger "$USER"
```

Depois de habilitar lingering, suas units systemd de usuário (incluindo
`hermes-gateway-<profile>.service`) continuam rodando entre desconexões SSH
e reboots.

## Segurança contra conflito de token

Cada profile deve usar tokens de bot únicos para cada plataforma. Se dois profiles
compartilham um token Telegram, Discord, Slack, WhatsApp ou Signal, o segundo
gateway recusa iniciar com erro nomeando o profile conflitante.

Para auditar:

```bash
grep -H 'TELEGRAM_BOT_TOKEN\|DISCORD_BOT_TOKEN' \
     ~/.hermes/.env ~/.hermes/profiles/*/.env
```

## Atualizando o código

`hermes update` puxa o código mais recente uma vez e sincroniza novas skills embutidas em
todo profile:

```bash
hermes update
hermes-gateways restart
```

Skills modificadas pelo usuário nunca são sobrescritas.

## Solução de problemas

### "Could not find service in domain for user gui: 501"

Você executou `hermes gateway start` depois de um `hermes gateway stop` anterior. O
`stop` da CLI faz `launchctl unload` completo, que remove o serviço do
registro do launchd. A CLI captura este erro específico em `start` e
recarrega automaticamente o plist (`↻ launchd job was unloaded; reloading
service definition`). O serviço inicia normalmente. Nada a corrigir.

### PID obsoleto após crash

Se o gateway de um profile mostra `not running`, mas um processo ainda está vivo:

```bash
ps -ef | grep "hermes_cli.*-p <profile>"
cat ~/.hermes/profiles/<profile>/gateway.pid
kill -TERM <pid>          # graceful
kill -KILL <pid>          # se falhar após alguns segundos
<profile> gateway start
```

### Forçando reset duro de um serviço

```bash
# macOS
launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist
launchctl load   ~/Library/LaunchAgents/ai.hermes.gateway-<profile>.plist

# Linux
systemctl --user restart hermes-gateway-<profile>.service
```

### Health check

```bash
hermes doctor                  # profile default
hermes -p <profile> doctor     # um profile
```
