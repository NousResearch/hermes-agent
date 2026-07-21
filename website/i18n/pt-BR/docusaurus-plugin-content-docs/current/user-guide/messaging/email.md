---
sidebar_position: 7
title: "Email"
description: "Configure o Hermes Agent como assistente de email via IMAP/SMTP"
---

# Configuração de Email {#email-setup}

O Hermes pode receber e responder emails usando os protocolos padrão IMAP e SMTP. Envie um email para o endereço do agente e ele responde na mesma thread — sem cliente especial ou API de bot. Funciona com Gmail, Outlook, Yahoo, Fastmail ou qualquer provedor que suporte IMAP/SMTP.

:::info Apenas adaptador do gateway: sem dependências externas
Esta página cobre o adaptador de email do gateway, que usa os módulos built-in do Python `imaplib`, `smtplib` e `email`. Nenhum pacote adicional ou serviço externo é necessário para este caminho do gateway.
:::

Isso é separado da [skill de email Himalaya](/docs/user-guide/skills/bundled/email/email-himalaya) incluída, que permite ao agente gerenciar email por comandos de terminal e exige o CLI externo `himalaya` mais um arquivo de config do Himalaya.

| Caso de uso | O que configurar | Dependência externa |
|---|---|---|
| Permitir que pessoas enviem email ao agente Hermes e recebam respostas | Adaptador de email do gateway nesta página | Nenhuma além de uma conta de email IMAP/SMTP |
| Permitir que o agente inspecione, componha, mova e gerencie mensagens da caixa de entrada via ferramentas de terminal | Skill de email Himalaya | CLI `himalaya` e `~/.config/himalaya/config.toml` |

---

## Pré-requisitos {#prerequisites}

- **Uma conta de email dedicada** para o seu agente Hermes (não use seu email pessoal)
- **IMAP habilitado** na conta de email
- **Uma senha de app** se usar Gmail ou outro provedor com 2FA

### Configuração do Gmail {#gmail-setup}

1. Habilite a autenticação de dois fatores na sua Conta Google
2. Acesse [App Passwords](https://myaccount.google.com/apppasswords)
3. Crie uma nova senha de app (selecione "Mail" ou "Other")
4. Copie a senha de 16 caracteres — você usará isso em vez da senha regular

### Outlook / Microsoft 365 {#outlook--microsoft-365}

1. Acesse [Security Settings](https://account.microsoft.com/security)
2. Habilite 2FA se ainda não estiver ativo
3. Crie uma senha de app em "Additional security options"
4. Host IMAP: `outlook.office365.com`, host SMTP: `smtp.office365.com`

### Outros provedores {#other-providers}

A maioria dos provedores de email suporta IMAP/SMTP. Consulte a documentação do seu provedor para:
- Host e porta IMAP (geralmente porta 993 com SSL)
- Host e porta SMTP (geralmente porta 587 com STARTTLS)
- Se senhas de app são necessárias

---

## Passo 1: Configure o Hermes {#step-1-configure-hermes}

A forma mais fácil:

```bash
hermes gateway setup
```

Selecione **Email** no menu de plataformas. O assistente solicita seu endereço de email, senha, hosts IMAP/SMTP e remetentes permitidos.

### Configuração manual {#manual-configuration}

Adicione em `~/.hermes/.env`:

```bash
# Required
EMAIL_ADDRESS=hermes@gmail.com
EMAIL_PASSWORD=abcd efgh ijkl mnop    # App password (not your regular password)
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com

# Security (recommended)
EMAIL_ALLOWED_USERS=your@email.com,colleague@work.com

# Optional
EMAIL_IMAP_PORT=993                    # Default: 993 (IMAP SSL)
EMAIL_SMTP_PORT=587                    # Default: 587 (SMTP STARTTLS)
EMAIL_POLL_INTERVAL=15                 # Seconds between inbox checks (default: 15)
EMAIL_HOME_ADDRESS=your@email.com      # Default delivery target for cron jobs
```

---

## Passo 2: Inicie o gateway {#step-2-start-the-gateway}

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

Na inicialização, o adaptador:
1. Testa as conexões IMAP e SMTP
2. Marca todas as mensagens existentes na caixa de entrada como "vistas" (processa apenas emails novos)
3. Inicia o polling de novas mensagens

---

## Como funciona {#how-it-works}

### Recebendo mensagens {#receiving-messages}

O adaptador faz polling da caixa de entrada IMAP em busca de mensagens UNSEEN em um intervalo configurável (padrão: 15 segundos). Para cada email novo:

- A **linha de assunto** é incluída como contexto (ex.: `[Subject: Deploy to production]`)
- **Emails de resposta** (assunto começando com `Re:`) ignoram o prefixo de assunto — o contexto da thread já está estabelecido
- **Anexos** são armazenados em cache localmente:
  - Imagens (JPEG, PNG, GIF, WebP) → disponíveis para a ferramenta de visão
  - Documentos (PDF, ZIP, etc.) → disponíveis para acesso a arquivos
- **Emails só em HTML** têm tags removidas para extração de texto simples
- **Mensagens próprias** são filtradas para evitar loops de resposta
- **Remetentes automatizados/noreply** são ignorados silenciosamente — `noreply@`, `mailer-daemon@`, `bounce@`, `no-reply@`, e emails com cabeçalhos `Auto-Submitted`, `Precedence: bulk` ou `List-Unsubscribe`

### Enviando respostas {#sending-replies}

As respostas são enviadas via SMTP com threading de email adequado:

- Cabeçalhos **In-Reply-To** e **References** mantêm a thread
- **Linha de assunto** preservada com prefixo `Re:` (sem `Re: Re:` duplicado)
- **Message-ID** gerado com o domínio do agente
- Respostas enviadas como texto simples (UTF-8)

### Anexos de arquivo {#file-attachments}

O agente pode enviar anexos de arquivo nas respostas. Inclua `MEDIA:/path/to/file` na resposta e o arquivo é anexado ao email de saída.

### Ignorando anexos {#skipping-attachments}

Para ignorar todos os anexos recebidos (proteção contra malware ou economia de banda), adicione ao seu `config.yaml`:

```yaml
platforms:
  email:
    skip_attachments: true
```

Quando habilitado, partes de anexo e inline são ignoradas antes da decodificação do payload. O corpo de texto do email ainda é processado normalmente.

---

## Controle de acesso {#access-control}

O acesso por email é mais restrito por padrão do que plataformas estilo chat:

1. **`EMAIL_ALLOWED_USERS` definido** → apenas emails desses endereços são processados
2. **Sem allowlist** → remetentes desconhecidos são ignorados silenciosamente
3. **`EMAIL_ALLOW_ALL_USERS=true`** → qualquer remetente é aceito (use com cautela)
4. **`platforms.email.unauthorized_dm_behavior: pair`** → remetentes desconhecidos recebem um código de pareamento

:::warning
**Use uma caixa de entrada dedicada e configure `EMAIL_ALLOWED_USERS` para operação normal.** O pareamento por email é opt-in porque caixas compartilhadas frequentemente contêm mensagens não lidas não relacionadas, e o Hermes não deve responder a esses contatos por padrão.
:::

---

## Solução de problemas {#troubleshooting}

| Problema | Solução |
|---------|----------|
| **"IMAP connection failed"** na inicialização | Verifique `EMAIL_IMAP_HOST` e `EMAIL_IMAP_PORT`. Confirme que IMAP está habilitado na conta. No Gmail, habilite em Settings → Forwarding and POP/IMAP. |
| **"SMTP connection failed"** na inicialização | Verifique `EMAIL_SMTP_HOST` e `EMAIL_SMTP_PORT`. Confirme que a senha está correta (use App Password no Gmail). |
| **Mensagens não recebidas** | Verifique se `EMAIL_ALLOWED_USERS` inclui o email do remetente. Verifique a pasta de spam — alguns provedores marcam respostas automatizadas. |
| **"Authentication failed"** | No Gmail, você deve usar uma App Password, não a senha regular. Habilite 2FA primeiro. |
| **Respostas duplicadas** | Garanta que apenas uma instância do gateway está rodando. Verifique `hermes gateway status`. |
| **Resposta lenta** | O intervalo de polling padrão é 15 segundos. Reduza com `EMAIL_POLL_INTERVAL=5` para resposta mais rápida (mas mais conexões IMAP). |
| **Respostas sem threading** | O adaptador usa cabeçalhos In-Reply-To. Alguns clientes de email (especialmente web) podem não fazer thread corretamente com mensagens automatizadas. |

---

## Segurança {#security}

:::warning
**Use uma conta de email dedicada.** Não use seu email pessoal — o agente armazena a senha em `.env` e tem acesso completo à caixa de entrada via IMAP.
:::

- Use **App Passwords** em vez da senha principal (obrigatório no Gmail com 2FA)
- Defina `EMAIL_ALLOWED_USERS` para restringir quem pode interagir com o agente
- A senha fica em `~/.hermes/.env` — proteja este arquivo (`chmod 600`)
- IMAP usa SSL (porta 993) e SMTP usa STARTTLS (porta 587) por padrão — conexões são criptografadas

---

## Referência de variáveis de ambiente {#environment-variables-reference}

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EMAIL_ADDRESS` | Yes | — | Agent's email address |
| `EMAIL_PASSWORD` | Yes | — | Email password or app password |
| `EMAIL_IMAP_HOST` | Yes | — | IMAP server host (e.g., `imap.gmail.com`) |
| `EMAIL_SMTP_HOST` | Yes | — | SMTP server host (e.g., `smtp.gmail.com`) |
| `EMAIL_IMAP_PORT` | No | `993` | IMAP server port |
| `EMAIL_SMTP_PORT` | No | `587` | SMTP server port |
| `EMAIL_POLL_INTERVAL` | No | `15` | Seconds between inbox checks |
| `EMAIL_ALLOWED_USERS` | No | — | Comma-separated allowed sender addresses |
| `EMAIL_HOME_ADDRESS` | No | — | Default delivery target for cron jobs |
| `EMAIL_ALLOW_ALL_USERS` | No | `false` | Allow all senders (not recommended) |
