---
title: Busca de Ferramentas
description: Camada opt-in de progressive disclosure que adia schemas de ferramentas MCP e de plugin até serem necessários.
sidebar_position: 95
---

# Busca de Ferramentas

Quando você tem muitos servidores MCP ou ferramentas de plugin não-core anexadas a uma
sessão, seus JSON schemas podem consumir uma fração substancial da
janela de contexto a cada turno — mesmo quando apenas algumas delas são relevantes
ao que o usuário realmente pediu.

**Tool Search** é a camada opt-in de progressive disclosure do Hermes para esse
problema. Quando ativado, ferramentas MCP e de plugin são substituídas no
array de ferramentas visível ao modelo por três bridge tools, e o modelo carrega o
schema de cada ferramenta específica sob demanda.

:::info Ferramentas built-in do Hermes nunca são adiadas
As ferramentas que compõem o core capability set do Hermes (`terminal`,
`read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`,
`browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`,
`delegate_task`, `session_search` e o resto de
`_HERMES_CORE_TOOLS`) são *sempre* carregadas diretamente. Apenas ferramentas MCP e
ferramentas de plugin não-core são elegíveis para deferral.
:::

## Como funciona {#how-it-works}

Quando Tool Search ativa para um turno, o modelo vê três ferramentas novas no
lugar das adiadas:

```
tool_search(query, limit?)     — search the deferred-tool catalog
tool_describe(name)            — load the full schema for one tool
tool_call(name, arguments)     — invoke a deferred tool
```

Uma interação típica parece com:

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

Quando o modelo invoca `tool_call`, o Hermes **desembrulha a bridge** e
despacha a ferramenta subjacente exatamente como se o modelo a tivesse chamado
diretamente. Hooks pre-tool-call, guardrails, prompts de aprovação e
hooks post-tool-call rodam contra o nome real da ferramenta — não contra
`tool_call`. O feed de atividade no CLI e gateway também desembrulha para você
ver a ferramenta subjacente, não a bridge.

## Quando ativa? {#when-does-it-activate}

Tool Search usa **tiered disclosure**: a presença de *qualquer* ferramenta
deferível (MCP/plugin) ativa a bridge; o que escala com o tamanho do catálogo é
quanto do catálogo permanece visível, não se os schemas são adiados.

| Nível | Condição | O que o modelo vê |
| --- | --- | --- |
| **0** | Sem ferramentas MCP/plugin | Toda ferramenta eager, sem bridge. Pass-through. |
| **1** | Listing do catálogo adiado cabe no budget | Bridge + um manifest estilo skills de toda ferramenta adiada (nome + descrição curta, degradando para só nomes quando acima do budget). Degradação é **por servidor**: quando um servidor oversized (Cloudflare) está anexado junto a pequenos (Linear), os servidores pequenos mantêm seus listings por ferramenta e só o servidor oversized colapsa para uma linha de resumo. |
| **2** | Listing por ferramenta excede o budget mesmo só com nomes para todo servidor (ex.: superfície flat da API do Cloudflare sozinha: ~3.300 ferramentas cujos nomes são ~32K tokens) | Bridge bare + um resumo de uma linha por servidor (nome do servidor + contagem de ferramentas), para o modelo saber quais domínios são alcançáveis; ferramentas individuais são descobríveis apenas via `tool_search`. |

O budget de listing é `min(threshold_pct% do contexto, listing_max_tokens)`.
A decisão é reavaliada toda vez que o array de ferramentas é construído, então
adicionar ou remover servidores MCP no meio da sessão move a sessão entre
tiers na próxima montagem.

## Configuração {#configuration}

```yaml
tools:
  tool_search:
    enabled: auto       # auto (default), on, or off
    threshold_pct: 5    # listing budget as a percentage of context
    search_default_limit: 5
    max_search_limit: 20
    listing: auto       # embed a grouped name+description catalog manifest
    listing_max_tokens: 4000
```

| Chave | Padrão | Significado |
| --- | --- | --- |
| `enabled` | `auto` | `auto`/`on` ativam sempre que existir pelo menos uma ferramenta deferível; `off` desabilita totalmente (tudo permanece eager). |
| `threshold_pct` | `5` | Budget de listing como porcentagem do context length do modelo ativo. Faixa 0–100. |
| `search_default_limit` | `5` | Hits retornados quando o modelo chama `tool_search` sem `limit`. |
| `max_search_limit` | `20` | Limite superior rígido que o modelo pode pedir via `limit`. Faixa 1–50. |
| `listing` | `auto` | Embute um manifest estilo skills de toda ferramenta adiada (nome + primeira frase da descrição, ≤60 chars, agrupado por servidor MCP) na descrição da bridge `tool_search`. `auto` inclui quando cabe no budget (caindo para só nomes, depois para o resumo tier-2 por servidor); `on`/`off` forçam de um jeito ou outro. |
| `listing_max_tokens` | `4000` | Cap absoluto no listing embutido, independentemente do tamanho do contexto. Faixa 200–60000. Catálogos grandes degradam para só nomes ou resumos por servidor, mantendo schemas completos disponíveis via search. |

### Por que o listing existe {#why-the-listing-exists}

Sem ele, capacidades adiadas ficam *invisíveis* — benchmarking ao vivo mostrou
modelos substituindo ferramentas core visíveis (rodando `gh` no terminal em vez
de buscar a ferramenta GitHub adiada) ou declarando uma capacidade
inexistente em vez de chamar `tool_search`. O listing aplica o padrão de skills
a ferramentas: toda capacidade permanece descobrível por nome o tempo todo,
enquanto schemas completos de parâmetros permanecem adiados. Se o modelo vê o nome exato
da ferramenta no listing, pode pular `tool_search` e ir direto para
`tool_describe`, economizando um round trip.

Você também pode usar a forma booleana legada:

```yaml
tools:
  tool_search: true   # equivalent to {enabled: auto}
```

## Quando NÃO usar {#when-not-to-use-it}

Tool Search troca um custo fixo de tokens por turno (os três schemas de bridge tool
mais o listing do catálogo) e pelo menos um round trip extra em
ferramentas frias (describe → call) pelas economias nos schemas adiados.
No tier 1 o listing mantém toda capacidade visível, então o round trip de descoberta
geralmente desaparece — o modelo vai direto para
`tool_describe`. Benchmarking ao vivo mostrou o modo listing igualando
o sucesso de tarefa do eager loading enquanto custava menos que a bridge bare.

Se quiser o comportamento always-eager antigo para um toolset pequeno, defina
`enabled: off`.

## Trade-offs que não desaparecem {#trade-offs-that-dont-go-away}

Estes vêm da invariante de integridade do prompt cache — são inerentes
a qualquer design de progressive disclosure, não específicos desta implementação:

- **Um round trip extra em ferramentas frias.** Na primeira vez que o modelo precisa
  de uma ferramenta adiada, gasta uma ou duas chamadas de modelo extras para encontrar e
  carregar o schema. As economias de tokens no lado estático são reais, mas uma
  parte é paga de volta em runtime.
- **Sem benefício de cache em schemas adiados.** Um resultado `tool_describe`
  carregado entra no histórico de conversa (então é cached em
  turnos subsequentes) mas nunca se beneficia do prefixo de cache do
  system prompt.
- **Dependência de qualidade do modelo.** Tool Search assume que o modelo consegue escrever uma
  query de busca razoável para a ferramenta que quer. Modelos menores fazem isso
  menos bem; os números publicados da Anthropic (49% → 74% no Opus 4 com
  vs. sem tool search) mostram o upside mas também que ~26 pontos de
  precisão ainda são falha de retrieval.
- **Edições de toolset invalidam cache.** Adicionar ou remover uma ferramenta no meio da
  sessão muda as descrições das bridge tools (que incluem a
  contagem de ferramentas adiadas) e o catálogo, então o prompt cache é
  invalidado. Este é o mesmo trade-off de qualquer edição de toolset.

## Detalhes de implementação {#implementation-details}

- **Retrieval:** BM25 sobre nome de ferramenta tokenizado + descrição + nomes de
  parâmetros. Cai para match literal de substring no nome da ferramenta quando
  BM25 retorna zero hits com score positivo, o que protege contra
  casos degenerados de zero-IDF (ex.: buscar `"github"` contra um
  catálogo onde todo nome de ferramenta contém "github").
- **Catálogo é stateless entre turnos.** Reconstrói da lista atual de tool-defs
  a cada montagem — sem `Map` keyed por sessão. Isso evita
  a classe de bug onde um catálogo armazenado deriva do sync com o
  registry de ferramentas vivo.
- **O catálogo é escopado aos toolsets da sessão.** `tool_search`,
  `tool_describe` e `tool_call` só veem e invocam ferramentas que a
  sessão foi de fato concedida. Um subagente, worker kanban ou sessão de gateway
  restrita a um subconjunto de toolsets não pode usar a bridge para
  descobrir ou chamar uma ferramenta fora desse subconjunto — o catálogo adiado é
  a fatia deferível dos próprios toolsets enabled/disabled da sessão,
  não o registry inteiro do processo.
- **Sem sandbox JS.** O Hermes usa o modo "structured tools" mais simples
  (search / describe / call como funções plain). O "code
  mode" de sandbox JS que algumas outras implementações oferecem é uma superfície grande; nós
  pulamos.

## Veja também {#see-also}

- `tools/tool_search.py` — a implementação
- `tests/tools/test_tool_search.py` — a suíte de regressão
- O PDF `openclaw-tool-search-report` no PR da implementação original
  para a pesquisa que moldou o design
