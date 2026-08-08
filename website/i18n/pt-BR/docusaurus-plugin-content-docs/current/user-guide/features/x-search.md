---
title: Busca no X (Twitter)
description: Pesquise posts e threads no X (Twitter) a partir do agente usando a ferramenta x_search integrada da xAI na Responses API — funciona com login OAuth SuperGrok ou XAI_API_KEY.
sidebar_label: Busca no X (Twitter)
sidebar_position: 7
---

# Busca no X (Twitter)

A ferramenta `x_search` permite que o agente pesquise posts, perfis e threads no X (Twitter) diretamente. Ela é alimentada pela ferramenta `x_search` integrada da xAI na Responses API em `https://api.x.ai/v1/responses` — o próprio Grok executa a busca no servidor e retorna resultados sintetizados com citações dos posts de origem.

**Use isso em vez de `web_search`** quando você quiser especificamente discussão, reações ou afirmações **no X**. Para páginas web em geral, continue usando `web_search` / `web_extract`.

:::tip
Se você já paga o Portal por um modelo xAI, chamadas Live Search são cobradas na mesma chave xAI configurada para chat. Veja [Nous Portal](/integrations/nous-portal).
:::

## Autenticação {#authentication}

`x_search` se registra quando **qualquer um** dos caminhos de credencial xAI estiver disponível:

| Credencial | Origem | Configuração |
|------------|--------|--------------|
| **OAuth SuperGrok / X Premium+** (preferido) | Login no navegador em `accounts.x.ai`, renovado automaticamente | `hermes auth add xai-oauth` — veja [xAI Grok OAuth (SuperGrok / X Premium+)](../../guides/xai-grok-oauth.md) |
| **`XAI_API_KEY`** | Chave de API paga da xAI | Defina em `~/.hermes/.env` |

Ambos atingem o mesmo endpoint com o mesmo payload — a única diferença é o bearer token. **Quando ambos estão configurados, o OAuth SuperGrok vence**, para que o x_search rode contra a cota da sua assinatura em vez de gasto de API paga.

O `check_fn` da ferramenta executa o resolvedor de credenciais xAI sempre que a lista de ferramentas do modelo é reconstruída. Um retorno `True` significa que o bearer é obtível E não vazio E (se estiver expirado) renovado com sucesso. Tokens revogados com falha na renovação ocultam a ferramenta do schema; o modelo simplesmente não a vê.

## Habilitando a ferramenta {#enabling-the-tool}

Habilita-se automaticamente quando credenciais xAI (token OAuth ou `XAI_API_KEY`) estão presentes. Desabilite explicitamente via `hermes tools` → Search → x_search se não quiser isso.

```bash
hermes tools
# → 🐦 X (Twitter) Search   (pressione space para alternar)
```

O seletor oferece duas opções de credencial:

1. **xAI Grok OAuth (SuperGrok / Premium+)** — abre o navegador em `accounts.x.ai` se você ainda não estiver logado
2. **Chave de API xAI** — solicita `XAI_API_KEY`

Qualquer escolha satisfaz o gating. Você pode escolher as credenciais que já tiver; a ferramenta funciona igualmente com ambas. Se ambas acabarem configuradas, o OAuth é preferido no momento da chamada.

## Configuração {#configuration}

```yaml
# ~/.hermes/config.yaml
x_search:
  # Modelo xAI usado na chamada Responses.
  # grok-4.5 é o padrão recomendado; qualquer modelo Grok
  # com acesso à ferramenta x_search funciona.
  model: grok-4.5

  # Esforço de raciocínio opcional: low, medium, high ou xhigh. Quando omitido,
  # aplica-se o padrão do modelo selecionado. xhigh só é suportado por
  # modelos que o documentam, como grok-4.20-multi-agent.
  # reasoning_effort: low

  # Timeout da requisição em segundos. x_search pode levar 60–120s para
  # consultas complexas — o padrão é generoso. Mínimo: 30.
  timeout_seconds: 180

  # Número de tentativas automáticas em 5xx / ReadTimeout / ConnectionError.
  # Cada tentativa faz backoff (1.5x segundos da tentativa, limitado a 5s).
  retries: 2
```

`reasoning_effort` é enviado à xAI Responses API como
`reasoning: {effort: ...}`. Deixe sem definir para modelos que não suportam
raciocínio configurável. Valores inválidos falham antes de uma requisição HTTP à xAI.

## Parâmetros da ferramenta {#tool-parameters}

O agente chama `x_search` com estes argumentos:

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `query` | string (obrigatório) | O que pesquisar no X. |
| `allowed_x_handles` | array de strings | Lista opcional de handles a incluir **exclusivamente** (máx. 10). `@` inicial é removido. |
| `excluded_x_handles` | array de strings | Lista opcional de handles a excluir (máx. 10). Mutuamente exclusivo com `allowed_x_handles`. |
| `from_date` | string | Data inicial opcional `YYYY-MM-DD`. |
| `to_date` | string | Data final opcional `YYYY-MM-DD`. |
| `enable_image_understanding` | boolean | Pede à xAI para analisar imagens anexadas aos posts correspondentes. |
| `enable_video_understanding` | boolean | Pede à xAI para analisar vídeos anexados aos posts correspondentes. |

A ferramenta retorna JSON com:

- `answer` — resposta de texto sintetizada pelo Grok
- `citations` — citações retornadas pelo campo de nível superior da Responses API
- `inline_citations` — anotações `url_citation` extraídas do corpo da mensagem (cada uma com `url`, `title`, `start_index`, `end_index`)
- `degraded` — `true` quando qualquer filtro restritivo (`allowed_x_handles`, `excluded_x_handles`, `from_date`, `to_date`) foi definido E ambos os canais de citação voltaram vazios. Nesse caso, o `answer` foi sintetizado a partir do conhecimento próprio do modelo, não do índice X — trate como sem fonte. `false` caso contrário (incluindo o caso "sem filtros definidos" — uma resposta ampla sem fonte é apenas uma resposta, não uma falha de filtro)
- `degraded_reason` — string curta nomeando quais filtros estavam ativos, ou `null` quando `degraded` é `false`
- `credential_source` — `"xai-oauth"` se OAuth resolveu, `"xai"` se chave de API resolveu
- `model`, `query`, `provider`, `tool`, `success`

### Validação de datas {#date-validation}

`from_date` / `to_date` são validados no cliente antes da chamada HTTP:

- Ambos, se fornecidos, devem ser parseados como `YYYY-MM-DD`.
- Quando ambos estão definidos, `from_date` deve ser igual ou anterior a `to_date`.
- `from_date` não pode ser posterior a hoje em UTC — nenhum post pode existir em uma janela que ainda não começou, então a chamada retornaria zero citações com certeza.
- `to_date` no futuro é permitido (chamadores podem legitimamente pedir "de ontem até amanhã" para capturar posts conforme chegam).

Falhas de validação aparecem como resultado de ferramenta estruturado `{"error": "..."}`, nunca como chamada HTTP à xAI.

## Exemplo {#example}

Conversando com o agente:

> O que as pessoas no X estão dizendo sobre os novos recursos de imagem do Grok? Foque em respostas de @xai.

O agente irá:

1. Chamar `x_search` com `query="reactions to new Grok image features"`, `allowed_x_handles=["xai"]`
2. Receber uma resposta sintetizada mais uma lista de citações ligando a posts específicos
3. Responder com a resposta e referências

## Solução de problemas {#troubleshooting}

### "No xAI credentials available"

A ferramenta exibe isso quando ambos os caminhos de auth falham. Defina `XAI_API_KEY` em `~/.hermes/.env` ou execute `hermes auth add xai-oauth` e complete o login no navegador. Depois reinicie sua sessão para o agente reler o registro de ferramentas.

### "`x_search` is not enabled for this model"

O `x_search.model` configurado não tem acesso à ferramenta `x_search` no servidor. Mude para `grok-4.5` (o padrão) ou outro modelo Grok que a suporte. Consulte a [documentação xAI](https://docs.x.ai/) para a lista atual.

### A ferramenta não aparece no schema

Duas causas possíveis:

1. **Toolset não habilitado.** Execute `hermes tools` e confirme que `🐦 X (Twitter) Search` está marcado.
2. **Sem credenciais xAI.** O check_fn retorna False, então o schema permanece oculto. Execute `hermes auth status` para confirmar o estado de login xai-oauth, e verifique se `XAI_API_KEY` está definida (se você usa o caminho de chave de API).

### `degraded: true` — resposta sem citações

Quando você usou `allowed_x_handles`, `excluded_x_handles` ou um intervalo de datas e a resposta volta com `degraded: true`, o índice X da xAI não retornou posts correspondentes, mas o Grok ainda produziu uma resposta sintetizada a partir dos próprios dados de treinamento. A resposta não tem fonte — não a trate como resultado real do X.

Causas a verificar:

- **Erro de digitação no handle.** Remova o `@`, confira a ortografia e confirme que a conta existe.
- **Intervalo de datas estreito demais** ou deslizando além dos posts de hoje; amplie e tente de novo.
- **Lacuna no índice xAI.** Algumas contas ativas falham intermitentemente em aparecer no `x_search` mesmo postando regularmente. Tente de novo após alguns minutos, ou use a skill `xurl` para leituras diretas da API X quando precisar da timeline exata de um handle.

## Veja também {#see-also}

- [xAI Grok OAuth (SuperGrok / Premium+)](../../guides/xai-grok-oauth.md) — o guia de configuração OAuth
- [Busca web e extração](web-search.md) — para busca web geral (não X)
- [Tools Reference](../../reference/tools-reference.md) — catálogo completo de ferramentas
