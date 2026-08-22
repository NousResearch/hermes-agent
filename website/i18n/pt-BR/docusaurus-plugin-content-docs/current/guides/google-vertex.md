---
sidebar_position: 15
title: "Google Vertex AI"
description: "Use o Hermes Agent com o Gemini no Google Cloud Vertex AI — conta de serviço OAuth2 ou ADC, faturamento e cotas do GCP, sem chave de API estática"
---

# Google Vertex AI {#google-vertex-ai}

O Hermes Agent oferece suporte a **modelos Gemini no Google Cloud Vertex AI** por meio do endpoint compatível com OpenAI do Vertex. Diferente do [provedor Google AI Studio](/guides/google-gemini) (que usa uma chave de API estática contra `generativelanguage.googleapis.com`), o Vertex oferece **limites de taxa de nível empresarial e faturamento/créditos do GCP**, sendo a escolha certa quando você quer que o uso do Gemini seja debitado da sua conta do Google Cloud em vez de uma chave do AI Studio.

:::info O Vertex autentica com OAuth2, não com uma chave de API
O Vertex **não tem chave de API estática** para o endpoint padrão. Cada requisição precisa de um **token de acesso OAuth2** de curta duração (TTL ≈ 1 hora) gerado a partir de um JSON de conta de serviço ou de Application Default Credentials (ADC). O Hermes gera e **atualiza automaticamente** esses tokens para você — você nunca cola um token manualmente. É por isso que colar um token temporário no campo `api_key` de um provedor personalizado não funciona: ele expira no meio da sessão.
:::

## Pré-requisitos {#prerequisites}

- **Um projeto do Google Cloud** com a **API do Vertex AI habilitada** e faturamento ativo.
- **Credenciais**, uma destas:
  - uma chave **JSON de conta de serviço** com a função `roles/aiplatform.user`, ou
  - **Application Default Credentials** via `gcloud auth application-default login` (ou o servidor de metadados quando executando em uma VM do GCP).
- **`google-auth`** — instalado automaticamente na primeira vez que você selecionar o Vertex (instalação sob demanda), ou explicitamente com `pip install 'hermes-agent[vertex]'`.

## Início Rápido {#quick-start}

```bash
# Opção A — JSON de conta de serviço (recomendado para servidores / gateways)
echo "VERTEX_CREDENTIALS_PATH=/path/to/service-account.json" >> ~/.hermes/.env

# Opção B — Application Default Credentials (bom para desenvolvimento local)
gcloud auth application-default login

# Selecione o Vertex como seu provedor
hermes model
# → Escolha "More providers..." → "Google Vertex AI"
# → Digite o ID do seu projeto GCP (ou deixe em branco para usar o das suas credenciais)
# → Escolha uma região (padrão: global)
# → Selecione um modelo Gemini

# Comece a conversar
hermes chat
```

## Configuração {#configuration}

O Vertex divide suas configurações por sensibilidade:

- O **caminho da credencial** é um ponteiro para um segredo e fica em `~/.hermes/.env`.
- **ID do projeto e região** são configurações de roteamento não sensíveis e ficam em `~/.hermes/config.yaml`.

`~/.hermes/.env`:

```bash
# Uma destas (verificadas nesta ordem); omita ambas para usar ADC:
VERTEX_CREDENTIALS_PATH=/path/to/service-account.json
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

`~/.hermes/config.yaml`:

```yaml
model:
  default: google/gemini-3-flash-preview
  provider: vertex

vertex:
  project_id: my-gcp-project   # em branco → usa o projeto embutido nas credenciais
  region: global               # "global" é obrigatório para os previews do Gemini 3.x
```

:::tip Variáveis de ambiente têm prioridade sobre o config.yaml
`VERTEX_PROJECT_ID` e `VERTEX_REGION` sobrescrevem os valores `vertex.project_id` / `vertex.region` no `config.yaml`. Use-as para substituições pontuais no shell; mantenha as configurações duráveis no `config.yaml`.
:::

### Como a autenticação funciona {#how-authentication-works}

1. O Hermes resolve as credenciais nesta ordem: `VERTEX_CREDENTIALS_PATH` → `GOOGLE_APPLICATION_CREDENTIALS` → ADC.
2. Ele gera um token de acesso OAuth2 (escopo `cloud-platform`) e o armazena em cache, atualizando-o quando estiver a menos de 5 minutos da expiração.
3. O token é entregue a um cliente OpenAI padrão apontando para o endpoint do Vertex:
   ```text
   https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi
   ```
   Localizações regionais usam um host `{region}-aiplatform.googleapis.com` em vez disso.
4. Se uma sessão durar mais do que o tempo de vida do token e uma requisição retornar `401`, o Hermes gera o token novamente e tenta de novo automaticamente. Em um gateway de longa duração, se o refresh token do ADC também tiver expirado, o Hermes recorre ao JSON de conta de serviço, quando configurado.

## Modelos Disponíveis {#available-models}

O Vertex exige o prefixo de fornecedor `google/` nos IDs de modelo. O seletor do `hermes model` oferece:

| Modelo | ID |
|-------|----|
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` |
| Gemini 3 Pro Preview | `google/gemini-3-pro-preview` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| Gemini 3.1 Flash Lite Preview | `google/gemini-3.1-flash-lite-preview` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |

:::note Região `global` para o Gemini 3.x
Os modelos de preview do Gemini 3.x são servidos pelo endpoint `global`. Endpoints regionais (`us-central1`, etc.) podem retornar 404 para eles. Deixe `region: global` a menos que você tenha um motivo específico para fixar uma região.
:::

## Trocando de Modelo Durante a Sessão {#switching-models-mid-session}

```text
/model google/gemini-3-pro-preview
/model google/gemini-3-flash-preview
```

O `/model` alterna entre provedores e modelos já configurados; ele não coleta novas credenciais. Configure o Vertex com `hermes model` primeiro.

## Raciocínio / Thinking {#reasoning--thinking}

O Vertex expõe o orçamento de "thinking" do Gemini através da superfície compatível com OpenAI. O Hermes mapeia automaticamente sua configuração de esforço de raciocínio para `extra_body.google.thinking_config`, então `reasoning_effort` funciona da mesma forma que em outras superfícies do Gemini.

## Diagnóstico {#diagnostics}

```bash
hermes doctor
```

O doctor informa se as credenciais do Vertex podem ser resolvidas (caminho da conta de serviço ou ADC) e se o provedor está configurado.

## Solução de Problemas {#troubleshooting}

### "Vertex AI credentials could not be resolved" {#vertex-ai-credentials-could-not-be-resolved}

O Hermes não encontrou nem um JSON de conta de serviço nem um ADC funcional. Defina `VERTEX_CREDENTIALS_PATH` em `~/.hermes/.env`, ou execute `gcloud auth application-default login`. Se seu projeto não estiver embutido nas credenciais, defina `vertex.project_id` em `config.yaml`.

### `google-auth` não instalado {#google-auth-not-installed}

Instale o extra: `pip install 'hermes-agent[vertex]'`. O Hermes também o instala sob demanda na primeira vez que você selecionar o provedor Vertex.

### 404 em modelos Gemini 3.x {#404-on-gemini-3x-models}

Você provavelmente está em um endpoint regional. Defina `region: global` na seção `vertex:` do `config.yaml` (ou remova `VERTEX_REGION`).

### 403 / permissão negada {#403--permission-denied}

A conta de serviço (ou sua identidade ADC) precisa da função `roles/aiplatform.user` no projeto, e a API do Vertex AI deve estar habilitada para esse projeto.

## Relacionados {#related}

- [Google Gemini (AI Studio)](/guides/google-gemini) — Gemini com chave de API estática, sem GCP
- [AWS Bedrock](/guides/aws-bedrock) — outra integração nativa com provedor de nuvem
- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
