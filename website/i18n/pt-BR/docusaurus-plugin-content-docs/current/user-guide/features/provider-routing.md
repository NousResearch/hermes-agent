---
title: Roteamento de provider
description: Configure preferências de provider do OpenRouter ou Nous Portal para otimizar custo, velocidade ou qualidade.
sidebar_label: Roteamento de provider
sidebar_position: 7
---

# Roteamento de provider {#provider-routing}

Ao usar [OpenRouter](https://openrouter.ai) ou [Nous Portal](/integrations/nous-portal) como provider LLM, o Hermes Agent suporta **roteamento de provider** — controle fino sobre quais providers subjacentes tratam suas requisições e como são priorizados.

O OpenRouter roteia requisições para muitos providers (por exemplo, Anthropic, Google, AWS Bedrock, Together AI). O roteamento de provider permite otimizar por custo, velocidade, qualidade ou impor requisitos específicos de provider.

:::tip
Tráfego roteado pelo Nous Portal respeita as mesmas preferências de provider — e assinantes do Portal ganham 10% de desconto em providers cobrados por token.
:::

## Configuração {#configuration}

Adicione uma seção `provider_routing` ao seu `~/.hermes/config.yaml`:

```yaml
provider_routing:
  sort: "price"           # How to rank providers
  only: []                # Whitelist: only use these providers
  ignore: []              # Blacklist: never use these providers
  order: []               # Explicit provider priority order
  require_parameters: false  # Only use providers that support all parameters
  data_collection: null   # Control data collection ("allow" or "deny")
```

:::info
Roteamento de provider só se aplica ao usar OpenRouter ou Nous Portal. Não tem efeito com conexões diretas ao provider (por exemplo, conectar diretamente à API da Anthropic).
:::

## Opções {#options}

### `sort` {#sort}

Controla como o OpenRouter ranqueia providers disponíveis para sua requisição.

| Valor | Descrição |
|-------|-------------|
| `"price"` | Provider mais barato primeiro |
| `"throughput"` | Maior tokens por segundo primeiro |
| `"latency"` | Menor tempo até o primeiro token primeiro |

```yaml
provider_routing:
  sort: "price"
```

### `only` {#only}

Whitelist de slugs de provider. Quando definida, **somente** esses providers serão usados. Todos os outros são excluídos. Use o slug em minúsculas mostrado pelo OpenRouter para cada provider.

```yaml
provider_routing:
  only:
    - "anthropic"
    - "google"
```

### `ignore` {#ignore}

Blacklist de nomes de provider. Esses providers **nunca** serão usados, mesmo que ofereçam a opção mais barata ou rápida.

```yaml
provider_routing:
  ignore:
    - "together"
    - "deepinfra"
```

### `order` {#order}

Ordem de prioridade explícita. Providers listados primeiro são preferidos. Providers não listados são usados como fallback.

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
    - "amazon-bedrock"
```

### `require_parameters` {#require-parameters}

Quando `true`, o OpenRouter só roteia para providers que suportam **todos** os parâmetros da sua requisição (como `temperature`, `top_p`, `tools`, etc.). Isso evita quedas silenciosas de parâmetros.

```yaml
provider_routing:
  require_parameters: true
```

### `data_collection` {#data-collection}

Controla se providers podem usar seus prompts para treinamento. Opções: `"allow"` ou `"deny"`.

```yaml
provider_routing:
  data_collection: "deny"
```

## Exemplos práticos {#practical-examples}

### Otimizar para custo {#optimize-for-cost}

Roteie para o provider disponível mais barato. Bom para alto volume e desenvolvimento:

```yaml
provider_routing:
  sort: "price"
```

### Otimizar para velocidade {#optimize-for-speed}

Priorize providers de baixa latência para uso interativo:

```yaml
provider_routing:
  sort: "latency"
```

### Otimizar para throughput {#optimize-for-throughput}

Melhor para geração longa onde tokens por segundo importam:

```yaml
provider_routing:
  sort: "throughput"
```

### Fixar em providers específicos {#lock-to-specific-providers}

Garanta que todas as requisições passem por um provider específico para consistência:

```yaml
provider_routing:
  only:
    - "anthropic"
```

### Evitar providers específicos {#avoid-specific-providers}

Exclua providers que você não quer usar (por exemplo, por privacidade de dados):

```yaml
provider_routing:
  ignore:
    - "together"
    - "lepton"
  data_collection: "deny"
```

### Ordem preferida com fallbacks {#preferred-order-with-fallbacks}

Tente seus providers preferidos primeiro, com fallback para outros se indisponíveis:

```yaml
provider_routing:
  order:
    - "anthropic"
    - "google"
  require_parameters: true
```

## Como funciona {#how-it-works}

Preferências de roteamento de provider são passadas ao OpenRouter ou Nous Portal em requisições de chat do agente e resumos de limite de iteração via o campo `extra_body.provider`. (`extra_body` é o argumento do SDK Python OpenAI; vira o objeto `provider` de nível superior na requisição JSON.) Tarefas auxiliares como compressão e geração de título são configuradas independentemente em `auxiliary.<task>.extra_body`.

- **Modo CLI** — configurado em `~/.hermes/config.yaml`, carregado na inicialização
- **Modo gateway** — mesmo arquivo de config, carregado quando o gateway inicia

A config de roteamento é lida de `config.yaml` e passada como parâmetros ao criar o `AIAgent`:

```
providers_allowed  ← from provider_routing.only
providers_ignored  ← from provider_routing.ignore
providers_order    ← from provider_routing.order
provider_sort      ← from provider_routing.sort
provider_require_parameters ← from provider_routing.require_parameters
provider_data_collection    ← from provider_routing.data_collection
```

:::tip
Você pode combinar várias opções. Por exemplo, ordenar por preço mas excluir certos providers e exigir suporte a parâmetros:

```yaml
provider_routing:
  sort: "price"
  ignore: ["together"]
  require_parameters: true
  data_collection: "deny"
```
:::

## Comportamento padrão {#default-behavior}

Sem seção `provider_routing` configurada (o padrão), o agregador usa sua própria lógica de roteamento, que geralmente equilibra custo e disponibilidade automaticamente.

:::tip Roteamento de provider vs. modelos fallback
Roteamento de provider controla quais **sub-providers por trás do OpenRouter ou Nous Portal** tratam suas requisições. Para failover automático a um provider totalmente diferente quando seu modelo principal falha, veja [Fallback Providers](/user-guide/features/fallback-providers).
:::
