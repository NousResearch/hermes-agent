---
title: Busca de ferramentas
sidebar_position: 95
---

# Busca de ferramentas

Quando você tem muitos servidores MCP ou ferramentas de plugin não-core
anexadas a uma sessão, seus schemas JSON podem consumir uma fração
substancial da janela de contexto a cada turno — mesmo quando apenas
algumas delas são relevantes para o que você realmente pediu.

**Busca de ferramentas** é a camada opt-in de divulgação progressiva do Hermes para esse
problema. Quando ativado, ferramentas MCP e de plugin são substituídas no
array de ferramentas visível ao modelo por três ferramentas ponte, e o modelo
carrega o schema de cada ferramenta específica sob demanda.

:::info Ferramentas built-in do Hermes nunca são adiadas
As ferramentas que compõem o conjunto de capacidades core do Hermes (`terminal`,
`read_file`, `write_file`, `patch`, `search_files`, `todo`, `memory`,
`browser_*`, `web_search`, `web_extract`, `clarify`, `execute_code`,
`delegate_task`, `session_search` e o restante de
`_HERMES_CORE_TOOLS`) são *sempre* carregadas diretamente. Apenas ferramentas MCP
e de plugin não-core são elegíveis para adiamento.
:::

## Como funciona {#how-it-works}

Quando a Busca de ferramentas ativa para um turno, o modelo vê três novas ferramentas no
lugar das adiadas:

```
tool_search(query, limit?)     — pesquisa o catálogo de ferramentas adiadas
tool_describe(name)            — carrega o schema completo de uma ferramenta
tool_call(name, arguments)     — invoca uma ferramenta adiada
```

Uma interação típica se parece com:

```
Model: tool_search("create a github issue")
  → { matches: [{ name: "mcp_github_create_issue", ... }, ...] }
Model: tool_describe("mcp_github_create_issue")
  → { parameters: { type: "object", properties: { ... } } }
Model: tool_call("mcp_github_create_issue", { title: "...", body: "..." })
  → { ok: true, issue_number: 42 }
```

Quando o modelo invoca `tool_call`, o Hermes **desembrulha a ponte** e
despacha a ferramenta subjacente exatamente como se o modelo a tivesse chamado
diretamente. Hooks pre-tool-call, guardrails, prompts de aprovação e
hooks post-tool-call rodam contra o nome real da ferramenta — não contra
`tool_call`. O feed de atividade no CLI e no gateway também desembrulha, para você
ver a ferramenta subjacente, não a ponte.

## Quando ele ativa? {#when-does-it-activate}

Por padrão a Busca de ferramentas roda em modo `auto`: ativa apenas quando os
schemas de ferramentas adiáveis consumiriam pelo menos 10% da janela de contexto
do modelo ativo. Abaixo disso, a montagem do array de ferramentas é um
pass-through puro e você não paga overhead.

Essa decisão é reavaliada sempre que o array de ferramentas é montado, então:

- Uma sessão com poucas ferramentas MCP e um modelo de contexto longo nunca
  ativa a Busca de ferramentas.
- Uma sessão com muitos servidores MCP anexados (tipicamente 15+ ferramentas) começa
  a ativá-lo.
- Remover servidores MCP no meio da sessão retorna corretamente à exposição
  direta na próxima montagem.

## Configuração {#configuration}

```yaml
tools:
  tool_search:
    enabled: auto       # auto (padrão), on ou off
    threshold_pct: 10   # porcentagem do contexto — usado apenas em modo auto
    search_default_limit: 5
    max_search_limit: 20
```

| Chave | Padrão | Significado |
| --- | --- | --- |
| `enabled` | `auto` | `auto` ativa acima do limiar; `on` sempre ativa se houver pelo menos uma ferramenta adiável; `off` desabilita totalmente. |
| `threshold_pct` | `10` | Porcentagem do comprimento de contexto em que o modo `auto` entra em ação. Intervalo 0–100. |
| `search_default_limit` | `5` | Acertos retornados quando o modelo chama `tool_search` sem `limit`. |
| `max_search_limit` | `20` | Limite superior rígido que o modelo pode solicitar via `limit`. Intervalo 1–50. |

Você também pode alternar o formato booleano legado:

```yaml
tools:
  tool_search: true   # equivalente a {enabled: auto}
```

## Quando NÃO usar {#when-not-to-use-it}

A Busca de ferramentas troca um custo fixo de tokens por turno (os três schemas
de ferramentas ponte, ~300 tokens) e pelo menos uma ida e volta extra (search →
describe → call) pela economia nos schemas adiados. É uma vitória clara quando
você tem muitas ferramentas e usa poucas por turno; é overhead quando
você tem poucas ferramentas no total.

O padrão `auto` cuida disso para você. Se você definir `enabled: on`
incondicionalmente, espere um leve custo por turno em toolsets pequenos.

## Trade-offs que não desaparecem {#trade-offs-that-dont-go-away}

Estes vêm do invariante de integridade do prompt-cache — são inerentes
a qualquer design de divulgação progressiva, não específicos desta implementação:

- **Uma ida e volta extra em ferramentas frias.** Na primeira vez que o modelo precisa
  de uma ferramenta adiada, gasta uma ou duas chamadas extras de modelo para encontrar e
  carregar o schema. A economia de tokens no lado estático é real, mas uma
  parte é paga em runtime.
- **Sem benefício de cache nos schemas adiados.** Um resultado de `tool_describe`
  carregado entra no histórico da conversa (então é cacheado em
  turnos subsequentes), mas nunca se beneficia do cache de prefixo do
  system prompt.
- **Dependência da qualidade do modelo.** A Busca de ferramentas assume que o modelo consegue escrever uma
  consulta de busca razoável para a ferramenta que quer. Modelos menores fazem isso
  pior; os números publicados da Anthropic (49% → 74% no Opus 4 com
  vs. sem tool search) mostram o upside, mas também que ~26 pontos de
  precisão ainda são falha de recuperação.
- **Edições de toolset invalidam o cache.** Adicionar ou remover uma ferramenta no meio da
  sessão altera as descrições das ferramentas ponte (que incluem a
  contagem de ferramentas adiadas) e o catálogo, então o prompt cache é
  invalidado. É o mesmo trade-off de qualquer edição de toolset.

## Detalhes de implementação {#implementation-details}

- **Recuperação:** BM25 sobre nome da ferramenta tokenizado + descrição + nomes de
  parâmetros. Faz fallback para correspondência literal de substring no nome da ferramenta quando
  BM25 não retorna acertos com score positivo, o que protege contra
  casos degenerados de IDF zero (ex.: buscar `"github"` contra um
  catálogo onde todo nome de ferramenta contém "github").
- **Catálogo é stateless entre turnos.** Reconstrói a partir da lista atual de
  tool-defs a cada montagem — sem `Map` indexado por session-key. Isso evita
  a classe de bug em que um catálogo armazenado fica fora de sync com o
  registro vivo de ferramentas.
- **O catálogo está escopado aos toolsets da sessão.** `tool_search`,
  `tool_describe` e `tool_call` só veem e invocam ferramentas que a
  sessão realmente recebeu. Um subagente, worker kanban ou sessão de gateway
  restrita a um subconjunto de toolsets não pode usar a ponte para
  descobrir ou chamar uma ferramenta fora desse subconjunto — o catálogo adiado é
  a fatia adiável dos próprios toolsets habilitados/desabilitados da sessão,
  não o registro inteiro do processo.
- **Sem sandbox JS.** O Hermes usa o modo mais simples de "ferramentas estruturadas"
  (search / describe / call como funções simples). O "modo código" com sandbox JS
  que algumas outras implementações oferecem é uma superfície grande; nós
  pulamos isso.

## Veja também {#see-also}

- `tools/tool_search.py` — a implementação
- `tests/tools/test_tool_search.py` — a suíte de regressão
- O PDF `openclaw-tool-search-report` no PR da implementação original
  para a pesquisa que moldou o design
