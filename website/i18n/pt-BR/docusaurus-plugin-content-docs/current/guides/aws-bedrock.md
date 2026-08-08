---
sidebar_position: 14
title: "AWS Bedrock"
description: "Use o Hermes Agent com o Amazon Bedrock — API Converse nativa, autenticação IAM, Guardrails e inferência entre regiões"
---

# AWS Bedrock {#aws-bedrock}

O Hermes Agent oferece suporte ao Amazon Bedrock como provedor nativo usando a **API Converse** — não o endpoint compatível com OpenAI. Isso dá acesso total ao ecossistema Bedrock: autenticação IAM, Guardrails, perfis de inferência entre regiões e todos os modelos de fundação.

## Pré-requisitos {#prerequisites}

- **Credenciais AWS** — qualquer fonte suportada pela [cadeia de credenciais do boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html):
  - função IAM de instância (EC2, ECS, Lambda — sem configuração)
  - variáveis de ambiente `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
  - `AWS_PROFILE` para SSO ou perfis nomeados
  - `aws configure` para desenvolvimento local
- **boto3** — instale com `cd ~/.hermes/hermes-agent && uv pip install -e ".[bedrock]"`
- **Permissões IAM** — no mínimo:
  - `bedrock:InvokeModel` e `bedrock:InvokeModelWithResponseStream` (para inferência)
  - `bedrock:ListFoundationModels` e `bedrock:ListInferenceProfiles` (para descoberta de modelos)

:::tip EC2 / ECS / Lambda
Em computação AWS, anexe uma função IAM com `AmazonBedrockFullAccess` e está pronto. Sem chaves de API, sem configuração no `.env` — o Hermes detecta automaticamente a função da instância.
:::

## Início Rápido {#quick-start}

```bash
# Instale com suporte ao Bedrock
cd ~/.hermes/hermes-agent && uv pip install -e ".[bedrock]"

# Selecione o Bedrock como seu provedor
hermes model
# → Escolha "More providers..." → "AWS Bedrock"
# → Selecione sua região e modelo

# Comece a conversar
hermes chat
```

## Configuração {#configuration}

Depois de executar `hermes model`, seu `~/.hermes/config.yaml` conterá:

```yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
  base_url: https://bedrock-runtime.us-east-2.amazonaws.com

bedrock:
  region: us-east-2
```

### Região {#region}

Defina a região da AWS de qualquer uma destas formas (ordem de prioridade, do maior para o menor):

1. `bedrock.region` no `config.yaml`
2. variável de ambiente `AWS_REGION`
3. variável de ambiente `AWS_DEFAULT_REGION`
4. Padrão: `us-east-1`

### Guardrails {#guardrails}

Para aplicar [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) a todas as invocações de modelo:

```yaml
bedrock:
  region: us-east-2
  guardrail:
    guardrail_identifier: "abc123def456"  # Do console do Bedrock
    guardrail_version: "1"                # Número da versão ou "DRAFT"
    stream_processing_mode: "async"       # "sync" ou "async"
    trace: "disabled"                     # "enabled", "disabled" ou "enabled_full"
```

### Descoberta de Modelos {#model-discovery}

O Hermes descobre automaticamente os modelos disponíveis por meio do plano de controle do Bedrock. Você pode personalizar a descoberta:

```yaml
bedrock:
  discovery:
    enabled: true
    provider_filter: ["anthropic", "amazon"]  # Mostra apenas estes provedores
    refresh_interval: 3600                     # Cache por 1 hora
```

## Modelos Disponíveis {#available-models}

Os modelos do Bedrock usam **IDs de perfil de inferência** para invocação sob demanda. O seletor do `hermes model` mostra esses IDs automaticamente, com os modelos recomendados no topo:

| Modelo | ID | Notas |
|-------|-----|-------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | Recomendado — melhor equilíbrio entre velocidade e capacidade |
| Claude Opus 4.6 | `us.anthropic.claude-opus-4-6-v1` | Mais capaz |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude mais rápido |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` | O carro-chefe da Amazon |
| Amazon Nova Micro | `us.amazon.nova-micro-v1:0` | Mais rápido e mais barato |
| DeepSeek V3.2 | `deepseek.v3.2` | Modelo aberto forte |
| Llama 4 Scout 17B | `us.meta.llama4-scout-17b-instruct-v1:0` | O mais recente da Meta |

:::info Inferência entre regiões
Modelos com o prefixo `us.` usam perfis de inferência entre regiões, que oferecem melhor capacidade e failover automático entre regiões da AWS. Modelos com o prefixo `global.` roteiam por todas as regiões disponíveis no mundo.
:::

## Trocando de Modelo Durante a Sessão {#switching-models-mid-session}

Use o comando `/model` durante uma conversa:

```
/model us.amazon.nova-pro-v1:0
/model deepseek.v3.2
/model us.anthropic.claude-opus-4-6-v1
```

## Diagnóstico {#diagnostics}

```bash
hermes doctor
```

O doctor verifica:
- Se as credenciais AWS estão disponíveis (variáveis de ambiente, função IAM, SSO)
- Se o `boto3` está instalado
- Se a API do Bedrock está acessível (ListFoundationModels)
- Número de modelos disponíveis na sua região

## Gateway (Plataformas de Mensagens) {#gateway-messaging-platforms}

O Bedrock funciona com todas as plataformas de gateway do Hermes (Telegram, Discord, Slack, Feishu, etc.). Configure o Bedrock como seu provedor e depois inicie o gateway normalmente:

```bash
hermes gateway setup
hermes gateway start
```

O gateway lê o `config.yaml` e usa a mesma configuração do provedor Bedrock.

## Solução de Problemas {#troubleshooting}

### "No API key found" / "No AWS credentials" {#no-api-key-found--no-aws-credentials}

O Hermes verifica as credenciais nesta ordem:
1. `AWS_BEARER_TOKEN_BEDROCK`
2. `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
3. `AWS_PROFILE`
4. Metadados da instância EC2 (IMDS)
5. Credenciais de contêiner do ECS
6. Função de execução do Lambda

Se nenhuma for encontrada, execute `aws configure` ou anexe uma função IAM à sua instância de computação.

### "Invocation of model ID ... with on-demand throughput isn't supported" {#invocation-of-model-id--with-on-demand-throughput-isnt-supported}

Use um **ID de perfil de inferência** (com prefixo `us.` ou `global.`) em vez do ID de modelo de fundação puro. Por exemplo:
- ❌ `anthropic.claude-sonnet-4-6`
- ✅ `us.anthropic.claude-sonnet-4-6`

### "ThrottlingException" {#throttlingexception}

Você atingiu o limite de taxa por modelo do Bedrock. O Hermes tenta novamente automaticamente com backoff. Para aumentar os limites, solicite um aumento de cota no [console AWS Service Quotas](https://console.aws.amazon.com/servicequotas/).

## Implantação na AWS em Um Clique {#one-click-aws-deployment}

Para uma implantação totalmente automatizada em EC2 com CloudFormation:

**[sample-hermes-agent-on-aws-with-bedrock](https://github.com/JiaDe-Wu/sample-hermes-agent-on-aws-with-bedrock)** — cria VPC, função IAM, instância EC2 e configura o Bedrock automaticamente. Implante em qualquer região com um clique.
