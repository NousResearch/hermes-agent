# ntfy {#ntfy}

[ntfy](https://ntfy.sh/) é um serviço simples de notificação pub-sub baseado em HTTP. Funciona com o servidor público gratuito em `ntfy.sh` ou qualquer instância self-hosted, e suporta qualquer cliente capaz de fazer requisições HTTP — telefones, navegadores, scripts, relógios.

O ntfy é um ótimo canal push leve para o Hermes: inscreva-se em um tópico pelo [app móvel ntfy](https://ntfy.sh/docs/subscribe/phone/), envie mensagens ao tópico para falar com o agente e receba a resposta de volta no seu telefone.

> Execute `hermes gateway setup` e escolha **ntfy** para um passo a passo guiado.

## Pré-requisitos {#prerequisites}

- Um nome de tópico (qualquer string única — `hermes-myname-2026` funciona bem)
- O [app móvel ntfy](https://ntfy.sh/docs/subscribe/phone/) instalado e inscrito nesse tópico
- Opcional: um servidor ntfy self-hosted, ou um token de conta `ntfy.sh` para tópicos privados/reservados

Só isso. Sem SDK, sem daemon, sem Node.js. O adaptador usa `httpx`, que já é uma dependência do Hermes.

## Configure o Hermes {#configure-hermes}

### Pelo assistente de setup {#via-setup-wizard}

```bash
hermes gateway setup
```

Selecione **ntfy** e siga as instruções.

### Por variáveis de ambiente {#via-environment-variables}

Adicione em `~/.hermes/.env`:

```
NTFY_TOPIC=hermes-myname-2026
NTFY_ALLOWED_USERS=hermes-myname-2026
NTFY_HOME_CHANNEL=hermes-myname-2026
```

| Variable | Required | Description |
|---|---|---|
| `NTFY_TOPIC` | Yes | Tópico para inscrever (mensagens recebidas) |
| `NTFY_SERVER_URL` | Optional | URL do servidor (padrão: `https://ntfy.sh`) — aponte para um ntfy self-hosted para privacidade |
| `NTFY_TOKEN` | Optional | Token Bearer (ex.: `tk_xyz`) ou `user:pass` para auth Basic |
| `NTFY_PUBLISH_TOPIC` | Optional | Tópico diferente para respostas enviadas (padrão: `NTFY_TOPIC`) |
| `NTFY_MARKDOWN` | Optional | Defina `true` para enviar respostas com o header `X-Markdown: true` |
| `NTFY_ALLOWED_USERS` | Recommended | Nomes de tópico permitidos separados por vírgula (tratados como IDs de usuário; veja abaixo) |
| `NTFY_ALLOW_ALL_USERS` | Optional | Defina `true` para permitir todo publisher — seguro apenas para tópicos privados com tokens de leitura |
| `NTFY_HOME_CHANNEL` | Optional | Tópico padrão para entrega de cron / notificações |
| `NTFY_HOME_CHANNEL_NAME` | Optional | Rótulo legível para o canal home |

## Modelo de identidade — leia antes de implantar {#identity-model--read-this-before-deploying}

O ntfy não tem identidade de usuário autenticada nativa. O campo `title` de uma mensagem publicada é **controlado pelo publisher** e pode ser qualquer coisa que o remetente quiser. O adaptador Hermes NÃO usa `title` para autorização — isso permitiria que qualquer publisher que conheça o tópico falsificasse um usuário permitido.

Em vez disso, **o próprio nome do tópico é a identidade**. Toda mensagem publicada no tópico é tratada como vinda do mesmo usuário lógico (o tópico). `NTFY_ALLOWED_USERS` portanto normalmente é apenas o próprio nome do tópico — uma allowlist de entrada única que restringe o canal inteiro.

Isso significa que **qualquer pessoa que conheça o tópico pode falar com o agente**. Para tornar isso um limite de confiança real:

- **Self-host ntfy** e trave o tópico com [Access Control](https://docs.ntfy.sh/config/#access-control). Apenas clientes autorizados com o token read/write podem publicar.
- Ou **use um tópico privado no ntfy.sh** ([tópicos reservados](https://docs.ntfy.sh/publish/#reserved-topics) exigem conta) e proteja com um `NTFY_TOKEN`.
- Ou **escolha um nome de tópico longo e impossível de adivinhar** (`hermes-7d4f9c8b-2026`) e trate-o como segredo compartilhado. É a configuração mais leve, mas o nome do tópico vaza via logs ou capturas de tela.

Em todos os casos, não envie dados sensíveis pelo ntfy a menos que o tópico subjacente tenha controle de acesso.

## Início rápido — fale com seu agente pelo telefone {#quick-start--talk-to-your-agent-from-your-phone}

1. Escolha um nome de tópico: `hermes-myname-2026`
2. No telefone: instale o [app ntfy](https://ntfy.sh/docs/subscribe/phone/), toque **+**, insira `hermes-myname-2026`
3. No host:
   ```bash
   echo 'NTFY_TOPIC=hermes-myname-2026' >> ~/.hermes/.env
   echo 'NTFY_ALLOWED_USERS=hermes-myname-2026' >> ~/.hermes/.env
   hermes gateway restart
   ```
4. Pelo app ntfy, envie uma mensagem ao tópico. A resposta do agente chega como notificação push.

## Usando ntfy com jobs cron {#using-ntfy-with-cron-jobs}

Com `NTFY_HOME_CHANNEL` definido, jobs cron podem entregar ao ntfy:

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="ntfy",          # uses NTFY_HOME_CHANNEL
    prompt="Check for alerts and summarise."
)
```

Ou direcione um tópico específico explicitamente via o campo `deliver:` do job cron, ou de um script shell com a [CLI `hermes send`](/guides/pipe-script-output):

```bash
hermes send ntfy:alerts-channel "Done!"
```

Isso funciona mesmo quando o cron roda fora de processo do gateway — o plugin registra um `standalone_sender_fn` que abre sua própria conexão HTTP.

## Self-hosting ntfy {#self-hosting-ntfy}

Se você quer controle total:

```bash
# Docker
docker run -p 80:80 -it binwiederhier/ntfy serve

# Native
go install heckel.io/ntfy/v2@latest
ntfy serve
```

Depois aponte o Hermes para ele:

```
NTFY_SERVER_URL=https://ntfy.mydomain.com
NTFY_TOPIC=hermes
NTFY_TOKEN=tk_abc123  # if you've set up access control
```

Self-hosting oferece controle de acesso a tópicos, políticas de persistência de mensagens, anexos e tags emoji. Veja a [documentação do servidor ntfy](https://docs.ntfy.sh/install/).

## Formatação Markdown {#markdown-formatting}

Clientes ntfy renderizam markdown quando o publisher define o header `X-Markdown: true`. Para habilitar nas respostas enviadas do Hermes:

```
NTFY_MARKDOWN=true
```

Ou em `config.yaml`:

```yaml
platforms:
  ntfy:
    extra:
      markdown: true
```

O app móvel suporta um subconjunto de CommonMark — negrito, itálico, listas, links, blocos de código cercados. Veja a [documentação markdown do ntfy](https://docs.ntfy.sh/publish/#markdown-formatting) para o conjunto exato.

## Configuração só de envio (notificações sem entrada) {#outgoing-only-setup-notifications-without-inbound}

Se você só quer que o Hermes *envie* notificações ao ntfy (resumos de cron, alertas) e nunca aceite mensagens de volta, defina `NTFY_TOPIC` e `NTFY_PUBLISH_TOPIC` com o mesmo valor e pule `NTFY_ALLOWED_USERS` por completo. Sem allowlist, o agente nunca responde a mensagens recebidas — seu telefone recebe os pushes, mas a conversa é unidirecional.

## Limites {#limits}

- **Tamanho da mensagem**: o ntfy limita corpos de mensagem a 4096 caracteres. O Hermes trunca com aviso quando isso é excedido.
- **Sem indicadores de digitação**: o protocolo não expõe um; `send_typing` é no-op.
- **Sem threads ou anexos**: o ntfy é notificação push simples. Respostas longas ficam no corpo da mensagem, sem fanout de thread.
- **Sem identidade de usuário nativa**: veja a seção de modelo de identidade acima.

## Solução de problemas {#troubleshooting}

**Falha de auth / 401** — `NTFY_TOKEN` está errado, ou o token não tem direitos de publish/subscribe neste tópico. O adaptador interrompe o loop de reconexão em 401 e o status do runtime do gateway mostrará `fatal: ntfy_unauthorized`. Corrija o token e reinicie o gateway.

**Tópico não encontrado / 404** — `NTFY_TOPIC` não existe no servidor configurado. No ntfy.sh, tópicos são criados automaticamente no primeiro publish, então um 404 significa que você apontou para um servidor self-hosted que não provisionou o tópico. O adaptador interrompe o loop de reconexão com `fatal: ntfy_topic_not_found`.

**Conectado mas sem mensagens** — Verifique se `NTFY_ALLOWED_USERS` inclui o próprio nome do tópico. Com o modelo de identidade do ntfy, o tópico É o usuário; deixar a allowlist vazia rejeita tudo.

**Reconecta a cada 60s** — O keepalive padrão do stream é 55s; o ntfy pode ter problemas intermitentes de rede. O adaptador aplica backoff exponencial (2 → 5 → 10 → 30 → 60s) e reseta para 0 quando um stream permanece vivo ≥60s.
