# A2A (Agent-to-Agent)

[A2A](https://a2a-protocol.org) é o protocolo aberto Agent2Agent (v1.0, mantido pela Linux Foundation) para comunicação entre agentes de IA independentes. O plugin A2A do Hermes funciona nas **duas direções**: seu agente pode chamar outros agentes A2A como ferramentas, e outros agentes podem enviar tarefas para o seu Hermes via HTTP.

Ele interopera com qualquer par compatível com A2A — outro Hermes, LangChain, CrewAI, agentes Google ADK, ou qualquer coisa construída sobre o `a2a-sdk` oficial.

## Quando usar o A2A {#when-to-use-a2a}

- **Hermes ↔ Hermes entre máquinas** — permita que seu agente de desktop repasse tarefas para um Hermes em um servidor, ou vice-versa, cada um com sua própria memória, ferramentas e credenciais.
- **Delegar para agentes especialistas** — um par que anuncia skills `web_search`/`research`/`coding` em seu Agent Card pode ser descoberto e chamado no meio da conversa.
- **Ser um serviço chamável** — exponha seu Hermes para que agentes de outros frameworks possam enviar tarefas a ele.

Quando você quiser múltiplos agentes na **mesma máquina**, prefira [delegação](../features/delegation.md) (subagentes no mesmo processo) ou o [quadro kanban](../features/kanban.md) (fila de trabalho durável multi-perfil) — o A2A serve para cruzar fronteiras de processo/máquina/framework.

## Ativação {#enable}

```bash
hermes gateway setup      # pick A2A
```

Ou em `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900
```

As ferramentas de cliente de saída são fornecidas como o toolset `a2a`, **desativado por padrão** — ative-o com `hermes tools`.

## Saída: chamando outros agentes {#outbound-calling-other-agents}

Com o toolset `a2a` ativado, o agente ganha:

| Ferramenta | O que faz |
|---|---|
| `a2a_discover(url)` | Busca e resume o Agent Card de um par |
| `a2a_call(agent, message, context_id?)` | Envia uma tarefa e recebe a resposta; multi-turno via `context_id` |
| `a2a_list()` | Pares configurados, conversas salvas, métricas |
| `a2a_history(context_id)` | Recupera uma conversa A2A persistida |
| `a2a_orchestrate(capability, message, mode?)` | Distribui uma tarefa para todo par que anuncia uma capacidade (`all` / `first` / `best`) |

Configure pares conhecidos em `config.yaml`:

```yaml
a2a_agents:
  researcher:
    url: "http://research-box.local:9900"
    auth: { type: bearer, token: "..." }
    timeout: 120
    capabilities: [web_search, research]
```

Depois basta pedir: *"Peça ao agente researcher para resumir as publicações de hoje no arXiv."* URLs diretas também funcionam — `a2a_call` aceita qualquer endpoint A2A.

## Entrada: sendo chamável {#inbound-being-callable}

Com a plataforma ativada, o Hermes disponibiliza:

- **Agent Card** em `GET /.well-known/agent-card.json` (caminho canônico v1.0; o legado `agent.json` também responde) — anuncia o nome do seu agente, skills (derivadas dos toolsets ativados) e requisitos de autenticação.
- **JSON-RPC 2.0** em `POST /` — métodos canônicos v1.0 (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `SubscribeToTask`, CRUD de configuração de push-notification) além dos aliases no estilo de caminho pré-1.0 (`message/send`, …).
- **Streaming SSE** para `SendStreamingMessage`, com frames envelopados em JSON-RPC conforme a especificação.
- **Notificações push** (webhooks) para tarefas de longa duração, assinadas com HMAC-SHA256.

Tarefas de entrada são injetadas em uma **sessão ativa do gateway** — o mesmo agente, memória e ferramentas que atendem seus outros canais — e a resposta final é devolvida ao chamador como resultado da tarefa. As conversas são indexadas pelo `contextId` do A2A, então um par pode manter uma troca multi-turno.

A interoperabilidade é verificada contra o `a2a-sdk` oficial em Python (resolução de card, `SendMessage`, streaming).

## Modelo de segurança {#security-model}

Seguro por padrão; toda etapa de ampliação é explícita:

- **Sem token ⇒ apenas localhost.** O servidor faz bind em `127.0.0.1`. A exposição remota exige um token bearer **e** um `A2A_HOST` explícito.
- **Tokens por par** — `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` dá a cada par sua própria credencial; o nome autenticado orienta rate limiting, confiança e auditoria.
- **Filtragem de prompt injection** — o texto de entrada é filtrado e enquadrado como entrada não confiável de um par. Pares remotos não conseguem invocar comandos de barra do operador.
- **Redação de saída** — strings com formato de credencial (chaves de API, JWTs, tokens) são removidas das respostas.
- **Log de auditoria** — cada troca é anexada a `~/.hermes/a2a_audit.jsonl`.
- **Anti-loop** — limites de turnos por contexto impedem que dois agentes fiquem trocando mensagens indefinidamente.

## Referência de configuração {#configuration-reference}

| Variável de ambiente | Padrão | Significado |
|---|---|---|
| `A2A_PEER_TOKENS` | _(não definido)_ | Credenciais por par `nome:token,…` (preferido) |
| `A2A_BEARER_TOKEN` | _(não definido)_ | Token compartilhado; a identidade recai sobre o IP do chamador |
| `A2A_HOST` | `127.0.0.1` | Host de bind — só se amplia quando um token é definido |
| `A2A_PORT` | `9900` | Porta de entrada |
| `A2A_AGENT_NAME` | derivado do hostname | Nome exibido no Agent Card |
| `A2A_PUBLIC_URL` | _(não definido)_ | URL roteável anunciada no card (proxies reversos / k8s) |
| `A2A_TRUSTED_PEERS` | _(não definido)_ | Lista de permissão de identidades autenticadas |
| `A2A_ALLOW_ALL_USERS` | `false` | Permite qualquer par autenticado (apenas para dev) |
| `A2A_RATE_LIMIT` | `60` | Requisições/minuto por identidade |
| `A2A_MAX_PINGPONG_TURNS` | `5` | Limite anti-loop de turnos por contexto (máx. 20) |
| `A2A_REPLY_TIMEOUT` | `300` | Segundos de espera pela resposta do agente |
| `A2A_PUSH_SECRET` | token bearer | Segredo HMAC para assinatura de notificação push |
| `A2A_ADVERTISED_TOOLSETS` | todos registrados | Restringe quais skills aparecem no Agent Card |

Atrás de um proxy reverso ou de um Service do Kubernetes, defina `A2A_PUBLIC_URL` (ou confie em `X-Forwarded-Host`/`X-Forwarded-Proto`) para que o Agent Card anuncie uma URL que os pares consigam realmente chamar de volta.

## Teste rápido {#quick-test}

```bash
# From another machine / agent:
curl http://your-host:9900/.well-known/agent-card.json

curl -X POST http://your-host:9900/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
       "params":{"message":{"messageId":"m1","role":"ROLE_USER",
                 "parts":[{"text":"What tools do you have?"}]}}}'
```

## Solução de problemas {#troubleshooting}

- **Os pares não conseguem alcançar a URL do card** — o card estava anunciando seu endereço de bind; defina `A2A_PUBLIC_URL` para a URL externamente roteável.
- **`401 Unauthorized`** — token incompatível; verifique `A2A_PEER_TOKENS`/`A2A_BEARER_TOKEN` no servidor e o bloco `auth:` do par.
- **O servidor não faz bind fora de localhost** — por design: defina primeiro um token bearer e depois `A2A_HOST=0.0.0.0`.
- **As respostas expiram em tarefas longas** — aumente `A2A_REPLY_TIMEOUT`, ou faça o chamador registrar uma configuração de notificação push e consultar `GetTask`.
