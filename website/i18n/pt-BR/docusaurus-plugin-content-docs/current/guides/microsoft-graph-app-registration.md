---
title: "Registrar um aplicativo do Microsoft Graph"
description: "Passo a passo no portal do Azure para criar o registro de aplicativo que alimenta o pipeline de reuniões do Teams"
---

# Registrar um aplicativo do Microsoft Graph {#register-a-microsoft-graph-application}

O pipeline de reuniões do Teams lê transcrições, gravações e artefatos relacionados de reuniões a partir do Microsoft Graph usando autenticação **somente para aplicativo** (daemon) — sem login de usuário, sem consentimento interativo por reunião. Isso exige um registro de aplicativo do Azure AD com permissões de aplicativo consentidas pelo administrador.

Este guia percorre:

1. Criar o registro do aplicativo
2. Criar um segredo de cliente
3. Conceder as permissões da API do Graph que o pipeline precisa
4. Consentir essas permissões como administrador
5. (Opcional) Restringir o aplicativo a usuários específicos com uma Application Access Policy

Você precisa de **direitos de administrador do locatário** (ou de um administrador para conceder o consentimento em seu nome) para concluir isso. Guarde os valores que você coletar — eles vão para o `~/.hermes/.env` no final.

## Pré-requisitos {#prerequisites}

- Um locatário do Microsoft 365 com licenças Teams Premium ou Teams que gerem transcrições e gravações de reuniões
- Acesso de administrador ao portal do Azure em [entra.microsoft.com](https://entra.microsoft.com)
- Um endpoint HTTPS publicamente acessível para notificações de alteração do Graph (configurado depois, na etapa do listener de webhook)

## Etapa 1: crie o registro do aplicativo {#step-1-create-the-app-registration}

1. Entre em [entra.microsoft.com](https://entra.microsoft.com) como administrador do locatário.
2. Navegue até **Identity → Applications → App registrations**.
3. Clique em **New registration**.
4. Preencha:
   - **Name:** `Hermes Teams Meeting Pipeline` (ou qualquer nome que você reconheça).
   - **Supported account types:** *Accounts in this organizational directory only (Single tenant)*.
   - **Redirect URI:** deixe em branco — a autenticação somente para aplicativo não precisa de uma.
5. Clique em **Register**.

Você chegará à página de visão geral do aplicativo. Copie dois valores:

- **Application (client) ID** → `MSGRAPH_CLIENT_ID`
- **Directory (tenant) ID** → `MSGRAPH_TENANT_ID`

## Etapa 2: crie um segredo de cliente {#step-2-create-a-client-secret}

1. No menu à esquerda, abra **Certificates & secrets**.
2. Clique em **New client secret**.
3. **Description:** `hermes-graph-secret`. **Expires:** escolha um valor que corresponda à sua política de rotação (6 a 24 meses é típico).
4. Clique em **Add**.
5. Copie a coluna **Value** imediatamente — ela só é exibida uma vez. Esse valor é o `MSGRAPH_CLIENT_SECRET`.

> A coluna **Secret ID** não é o segredo. Você quer a coluna **Value**.

## Etapa 3: conceda as permissões da API do Graph {#step-3-grant-graph-api-permissions}

O pipeline usa um conjunto mínimo viável de permissões de aplicativo. Adicione apenas o que você precisa; cada uma amplia o que o aplicativo pode ler em todo o locatário.

1. No menu à esquerda, abra **API permissions**.
2. Clique em **Add a permission** → **Microsoft Graph** → **Application permissions**.
3. Adicione as permissões da tabela abaixo que correspondem ao que você quer que o pipeline faça.
4. Depois de adicionar, clique em **Grant admin consent for `<your tenant>`**. A coluna Status deve mudar para uma marca de verificação verde em cada permissão.

### Necessário para resumos baseados em transcrição {#required-for-transcript-first-summaries}

| Permissão | O que ela permite ao aplicativo fazer |
|------------|--------------------------|
| `OnlineMeetings.Read.All` | Ler metadados de reuniões online do Teams (assunto, participantes, URL de entrada). |
| `OnlineMeetingTranscript.Read.All` | Ler transcrições de reunião geradas pelo Teams. |

### Necessário para o fallback de gravação (quando a transcrição não está disponível) {#required-for-recording-fallback-when-a-transcript-is-unavailable}

| Permissão | O que ela permite ao aplicativo fazer |
|------------|--------------------------|
| `OnlineMeetingRecording.Read.All` | Baixar gravações de reuniões do Teams para processamento de STT offline. |
| `CallRecords.Read.All` | Resolver reuniões a partir de registros de chamada quando apenas a URL de entrada é conhecida. |

### Necessário para entrega de resumos de saída (apenas modo Graph) {#required-for-outbound-summary-delivery-graph-mode-only}

Se `platforms.teams.extra.delivery_mode` for `graph`, o pipeline publica resumos em um canal ou chat do Teams via a API do Graph. Pule estas permissões se você usar o modo de entrega `incoming_webhook` em vez disso.

| Permissão | O que ela permite ao aplicativo fazer |
|------------|--------------------------|
| `ChannelMessage.Send` | Publicar mensagens em canais do Teams em nome do aplicativo. |
| `Chat.ReadWrite.All` | Publicar mensagens em chats individuais e em grupo (apenas se você definir `chat_id` como destino de entrega). |

### Não recomendado {#not-recommended}

- `OnlineMeetings.ReadWrite.All` / `Chat.ReadWrite` sem `.All` — mais amplo do que o pipeline precisa.
- Permissões delegadas — o pipeline usa fluxo somente para aplicativo (client-credentials); permissões delegadas não funcionam sem login de usuário.

## Etapa 4: (recomendado) restrinja o aplicativo com uma Application Access Policy {#step-4-recommended-scope-the-app-with-an-application-access-policy}

Por padrão, permissões de aplicativo como `OnlineMeetings.Read.All` concedem ao aplicativo acesso a **todas** as reuniões do locatário. Para demonstrações de parceiro e locatários de desenvolvimento isso é aceitável; para produção você quase certamente vai querer restringir de quais usuários o aplicativo pode ler as reuniões.

A Microsoft oferece **Application Access Policies** para o Teams exatamente para isso. A política é uma superfície apenas via PowerShell; não há interface no portal para ela.

De um PowerShell de administrador com o módulo MicrosoftTeams instalado e conectado (`Connect-MicrosoftTeams`):

```powershell
# Cria uma política restrita ao aplicativo Hermes
New-CsApplicationAccessPolicy `
  -Identity "Hermes-Meeting-Pipeline-Policy" `
  -AppIds "<MSGRAPH_CLIENT_ID>" `
  -Description "Restrict Hermes meeting pipeline to allow-listed users"

# Concede a política a usuários específicos cujas reuniões o pipeline pode ler
Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "alice@example.com"

Grant-CsApplicationAccessPolicy `
  -PolicyName "Hermes-Meeting-Pipeline-Policy" `
  -Identity "bob@example.com"
```

A propagação pode levar até 30 minutos após a concessão. Verifique com:

```powershell
Test-CsApplicationAccessPolicy -Identity "alice@example.com" -AppId "<MSGRAPH_CLIENT_ID>"
```

Sem a política, as reuniões de **qualquer** usuário podem ser lidas — é isso que a permissão tecnicamente concede. Não pule essa etapa em um locatário de produção.

## Etapa 5: grave as credenciais no seu arquivo de ambiente {#step-5-write-the-credentials-to-your-env-file}

Coloque os três valores que você coletou em `~/.hermes/.env`:

```bash
MSGRAPH_TENANT_ID=<directory-tenant-id>
MSGRAPH_CLIENT_ID=<application-client-id>
MSGRAPH_CLIENT_SECRET=<client-secret-value>
```

Defina permissões de arquivo para que apenas você possa ler o segredo:

```bash
chmod 600 ~/.hermes/.env
```

## Etapa 6: verifique o fluxo de token {#step-6-verify-the-token-flow}

O Hermes fornece um teste rápido de autenticação do Graph. Na sua instalação do Hermes:

```python
python -c "
import asyncio
from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
provider = MicrosoftGraphTokenProvider.from_env()
token = asyncio.run(provider.get_access_token())
print('Token acquired, length:', len(token))
print(provider.inspect_token_health())
"
```

Uma execução bem-sucedida imprime uma string de token longa e um dicionário de saúde mostrando `cached: True` e um valor de `expires_in_seconds` próximo de 3600. Falhas produzem um `MicrosoftGraphTokenError` com o código de erro do Azure — os mais comuns são:

| Erro do Azure | Significado | Correção |
|-------------|---------|-----|
| `AADSTS7000215: Invalid client secret` | O valor do segredo está incorreto ou expirou. | Gere um novo segredo na etapa 2; atualize o `.env`. |
| `AADSTS700016: Application not found` | `MSGRAPH_CLIENT_ID` errado ou locatário errado. | Verifique novamente se os valores da etapa 1 são do mesmo aplicativo. |
| `AADSTS90002: Tenant not found` | Erro de digitação no `MSGRAPH_TENANT_ID`. | Copie novamente o Directory (tenant) ID da visão geral do aplicativo. |
| `insufficient_claims` na hora da chamada (não na hora do token) | O token é obtido, mas o Graph retorna 401/403. | Você pulou o consentimento de administrador da etapa 3, ou adicionou permissões mas não consentiu novamente. Revise as permissões da API e clique em **Grant admin consent** de novo. |

## Rotacionando o segredo do cliente {#rotating-the-client-secret}

Segredos de cliente do Azure têm uma expiração rígida. Antes que o seu expire:

1. Crie um segundo segredo de cliente na etapa 2, sem excluir o primeiro.
2. Atualize `MSGRAPH_CLIENT_SECRET` em `~/.hermes/.env` com o novo valor.
3. Reinicie o gateway para que o novo segredo seja usado: `hermes gateway restart`.
4. Verifique com o teste rápido acima.
5. Exclua o segredo antigo do portal do Azure.

## Próximos passos {#next-steps}

Depois que as credenciais forem verificadas com sucesso, continue com:

- **Configuração do listener de webhook** — configure a plataforma de gateway `msgraph_webhook`, que recebe notificações de alteração do Graph.
- **Configuração do pipeline** — configure o tempo de execução do pipeline de reuniões do Teams e a CLI de operador.
- **Entrega de saída** — conecte os resumos de volta a um canal ou chat do Teams.

Essas páginas chegam junto com os PRs que adicionam o tempo de execução correspondente. Essa configuração de credenciais é um pré-requisito independente e é seguro concluí-la com antecedência.
