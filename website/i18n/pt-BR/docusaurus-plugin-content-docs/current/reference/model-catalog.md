---
sidebar_position: 11
title: Catálogo de Modelos
description: Manifesto hospedado remotamente que alimenta as listas curadas de seleção de modelos para OpenRouter e Nous Portal.
---

# Catálogo de Modelos

O Hermes busca listas curadas de modelos para **OpenRouter** e **Nous Portal** a partir de um manifesto JSON hospedado junto ao site de documentação. Isso permite que os mantenedores atualizem as listas de seleção sem lançar uma nova versão do `hermes-agent`.

Quando o manifesto está inacessível (offline, rede bloqueada, falha de hospedagem), o Hermes recorre silenciosamente ao snapshot embutido no repositório que acompanha o CLI. O manifesto nunca quebra o seletor — na pior das hipóteses, você vê a lista que veio com a sua versão instalada.

## URL do manifesto ativo {#live-manifest-url}

```
https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
```

Publicado em cada merge para `main` via o pipeline existente do GitHub Pages `deploy-site.yml`. A fonte da verdade está no repositório em `website/static/api/model-catalog.json`.

## Schema {#schema}

```json
{
  "version": 1,
  "updated_at": "2026-04-25T22:00:00Z",
  "metadata": {},
  "providers": {
    "openrouter": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2",         "description": "default", "default": true},
        {"id": "moonshotai/kimi-k3",   "description": "recommended", "metadata": {}},
        {"id": "openai/gpt-5.4",       "description": ""}
      ]
    },
    "nous": {
      "metadata": {},
      "models": [
        {"id": "z-ai/glm-5.2", "default": true},
        {"id": "anthropic/claude-opus-4.7"},
        {"id": "moonshotai/kimi-k3"}
      ]
    }
  }
}
```

Notas de campo:

- **`version`** — versão inteira do schema. Schemas futuros incrementam este valor; o Hermes rejeita manifestos com versões que não reconhece e recorre ao snapshot embutido.
- **`metadata`** — dicionário livre no nível do manifesto, do provedor e do modelo. Quaisquer chaves. O Hermes ignora campos desconhecidos, então você pode anotar entradas (`"tier": "paid"`, `"tags": [...]`, etc.) sem coordenar uma mudança de schema.
- **`description`** — somente OpenRouter. Alimenta o texto do badge do seletor (`"recommended"`, `"free"`, `"default"`, ou vazio). O Nous Portal não usa isso — a liberação do tier gratuito é determinada em tempo real a partir do endpoint de preços do Portal.
- **`default`** — exatamente uma entrada por provedor pode carregar `"default": true`. Esse modelo é o **padrão silencioso**: aquele em que o Hermes recai quando o usuário nunca selecionou um modelo (cartão de confirmação do onboarding na GUI, `provider` configurado sem `model`, `model.default` vazio). Lido apenas do cache em runtime (`get_default_model_from_cache`) para que caminhos de resolução críticos nunca acessem a rede; quando não existe manifesto em cache, o Hermes recorre à constante embutida `PREFERRED_SILENT_DEFAULT_MODEL`, que deve corresponder à entrada marcada. Isso permite que os mantenedores rotacionem o padrão silencioso sem lançar uma versão. É deliberadamente um modelo capaz e de baixo custo, nunca o carro-chefe mais caro.
- **Preço e tamanho de contexto** NÃO estão no manifesto. Eles vêm de APIs de provedores em tempo real (endpoints `/v1/models`, models.dev) no momento da busca.

## Comportamento de busca {#fetch-behavior}

| Quando | O que acontece |
|---|---|
| `/model` ou `hermes model` | Busca se o cache em disco está obsoleto, senão usa o cache |
| Cache em disco atualizado (< TTL) | Nenhum acesso à rede |
| Falha de rede com cache | Recai silenciosamente no cache, uma linha de log |
| Falha de rede, sem cache | Recai silenciosamente no snapshot embutido |
| Manifesto falha na validação de schema | Tratado como inacessível |

Local do cache: `~/.hermes/cache/model_catalog.json`.

## Configuração {#config}

```yaml
model_catalog:
  enabled: true
  url: https://hermes-agent.nousresearch.com/docs/api/model-catalog.json
  ttl_hours: 1
  providers: {}
```

Defina `enabled: false` para desativar totalmente a busca remota e sempre usar o snapshot embutido no repositório.

### URLs de sobrescrita por provedor {#per-provider-override-urls}

Terceiros podem hospedar sua própria lista de curadoria usando o mesmo schema. Aponte um provedor para uma URL personalizada:

```yaml
model_catalog:
  providers:
    openrouter:
      url: https://example.com/my-openrouter-curation.json
```

O manifesto de sobrescrita só precisa preencher o(s) bloco(s) de provedor com o(s) qual(is) se importa. Outros provedores continuam resolvendo contra a URL principal.

### Ocultando provedores do seletor {#hiding-providers-from-the-picker}

`excluded_providers` permite ocultar provedores específicos do seletor `/model`, mesmo quando existem credenciais válidas. Útil quando credenciais estão presentes para provedores legados ou de teste que não devem aparecer no uso normal (ex.: um token antigo do Copilot ou OpenRouter ainda em cache em `auth.json` ou descoberto via CLI `gh`).

```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai
```

A exclusão é comparada sem diferenciar maiúsculas/minúsculas contra toda chave sob a qual um provedor pode aparecer — o id do Hermes e o id do models.dev (provedores mapeados embutidos), o pid do overlay e o slug do Hermes resolvido (provedores de overlay), e o slug canônico (provedores canônicos) — então uma única entrada como `copilot` oculta o provedor independentemente de qual seção o emite. É respeitado por toda superfície de seleção `/model`: os seletores interativo/texto do gateway, o seletor da TUI e o seletor interativo do CLI `hermes model`. Uma lista vazia (ou omitir a chave) não tem efeito.

## Atualizando o manifesto {#updating-the-manifest}

Mantenedores:

```bash
# Regenera a partir das listas embutidas no repositório (mantém o manifesto sincronizado
# após editar OPENROUTER_MODELS ou _PROVIDER_MODELS["nous"] em hermes_cli/models.py).
python scripts/build_model_catalog.py
```

Depois abra um PR com a mudança resultante em `website/static/api/model-catalog.json` para `main`. O site de documentação faz deploy automático no merge e o novo manifesto fica ativo em poucos minutos.

Você também pode editar o JSON manualmente para mudanças finas de metadados que não pertencem ao snapshot embutido — o script gerador é uma conveniência, não a única fonte da verdade.
