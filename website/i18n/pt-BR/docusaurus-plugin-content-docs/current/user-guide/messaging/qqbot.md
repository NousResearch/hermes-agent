# QQ Bot

Conecte o Hermes ao QQ pela **Official QQ Bot API (v2)** — com suporte a mensagens privadas (C2C), menções @ em grupos, guild, mensagens diretas e transcrição de voz.

## Visão geral {#overview}

O adaptador QQ Bot usa a [Official QQ Bot API](https://bot.q.qq.com/wiki/develop/api-v2/) para:

- Receber mensagens via conexão **WebSocket** persistente com o QQ Gateway
- Enviar respostas em texto e markdown pela **REST API**
- Baixar e processar imagens, mensagens de voz e anexos de arquivo
- Transcrever mensagens de voz usando o ASR integrado da Tencent ou um provedor STT configurável

## Pré-requisitos {#prerequisites}

1. **Aplicativo QQ Bot** — Registre-se em [q.qq.com](https://q.qq.com):
   - Crie um novo aplicativo e anote seu **App ID** e **App Secret**
   - Habilite os intents necessários: mensagens C2C, mensagens @ em grupos, mensagens de guild
   - Configure seu bot em modo sandbox para testes ou publique para produção

2. **Dependências** — O adaptador requer `aiohttp` e `httpx`:
   ```bash
   pip install aiohttp httpx
   ```

## Configuração {#configuration}

### Configuração interativa

```bash
hermes gateway setup
```

Selecione **QQ Bot** na lista de plataformas e siga as instruções.

### Configuração manual

Defina as variáveis de ambiente necessárias em `~/.hermes/.env`:

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## Variáveis de ambiente {#environment-variables}

| Variable | Description | Default |
|---|---|---|
| `QQ_APP_ID` | QQ Bot App ID (required) | — |
| `QQ_CLIENT_SECRET` | QQ Bot App Secret (required) | — |
| `QQBOT_HOME_CHANNEL` | OpenID para entrega de cron/notificações | — |
| `QQBOT_HOME_CHANNEL_NAME` | Nome de exibição do canal home | `Home` |
| `QQ_ALLOWED_USERS` | OpenIDs de usuário separados por vírgula para acesso a DM | open (todos os usuários) |
| `QQ_GROUP_ALLOWED_USERS` | OpenIDs de grupo separados por vírgula para acesso a grupos | — |
| `QQ_ALLOW_ALL_USERS` | Defina como `true` para permitir todas as DMs | `false` |
| `QQ_PORTAL_HOST` | Substitua o host do portal QQ (defina como `sandbox.q.qq.com` para roteamento sandbox) | `q.qq.com` |
| `QQ_STT_API_KEY` | Chave de API do provedor de voz para texto | — |
| `QQ_STT_BASE_URL` | (Não lido diretamente — defina `platforms.qqbot.extra.stt.baseUrl` em `config.yaml` em vez disso) | n/a |
| `QQ_STT_MODEL` | Nome do model STT | `glm-asr` |

## Configuração avançada {#advanced-configuration}

Para controle detalhado, adicione configurações de plataforma em `~/.hermes/config.yaml`:

```yaml
platforms:
  qqbot:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # habilita markdown QQ (msg_type 2). Apenas config; sem equivalente em env var.
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled
      group_allow_from:
        - "group_openid_1"
      stt:
        provider: "zai"          # zai (GLM-ASR), openai (Whisper), etc.
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## Mensagens de voz (STT) {#voice-messages-stt}

A transcrição de voz funciona em duas etapas:

1. **ASR integrado do QQ** (gratuito, sempre tentado primeiro) — O QQ fornece `asr_refer_text` em anexos de mensagem de voz, que usa o reconhecimento de fala da própria Tencent
2. **Provedor STT configurado** (fallback) — Se o ASR do QQ não retornar texto, o adaptador chama uma API STT compatível com OpenAI:

   - **Zhipu/GLM (zai)**: Provedor padrão, usa o model `glm-asr`
   - **OpenAI Whisper**: Defina `QQ_STT_BASE_URL` e `QQ_STT_MODEL`
   - Qualquer endpoint STT compatível com OpenAI

## Solução de problemas {#troubleshooting}

### Bot desconecta imediatamente (desconexão rápida)

Isso geralmente significa:
- **App ID / Secret inválidos** — Verifique suas credenciais em q.qq.com
- **Permissões ausentes** — Certifique-se de que o bot tem os intents necessários habilitados
- **Bot apenas sandbox** — Se o bot está em modo sandbox, só pode receber mensagens do canal de teste sandbox do QQ

### Mensagens de voz não transcritas

1. Verifique se o `asr_refer_text` integrado do QQ está presente nos dados do anexo
2. Se estiver usando um provedor STT personalizado, verifique se `QQ_STT_API_KEY` está definido corretamente
3. Verifique os logs do gateway em busca de mensagens de erro STT

### Mensagens não entregues

- Verifique se os **intents** do bot estão habilitados em q.qq.com
- Verifique `QQ_ALLOWED_USERS` se o acesso a DM estiver restrito
- Para mensagens de grupo, certifique-se de que o bot foi **@mencionado** (a política de grupo pode exigir allowlist)
- Verifique `QQBOT_HOME_CHANNEL` para entrega de cron/notificações

### Erros de conexão

- Certifique-se de que `aiohttp` e `httpx` estão instalados: `pip install aiohttp httpx`
- Verifique a conectividade de rede com `api.sgroup.qq.com` e o gateway WebSocket
- Revise os logs do gateway para mensagens de erro detalhadas e comportamento de reconexão
