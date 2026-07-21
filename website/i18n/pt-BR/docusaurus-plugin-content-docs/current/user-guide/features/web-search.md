---
title: Busca web e extração
description: Pesquise na web e extraia conteúdo de páginas com vários backends — incluindo SearXNG self-hosted gratuito.
sidebar_label: Busca web
sidebar_position: 6
---

# Busca web e extração

O Hermes Agent inclui duas ferramentas web invocáveis pelo modelo, com vários providers:

- **`web_search`** — pesquisa na web e retorna resultados ranqueados
- **`web_extract`** — busca e extrai conteúdo legível de uma ou mais URLs

Ambas são configuradas por uma única seleção de backend. Providers são escolhidos via `hermes tools` ou definidos diretamente em `config.yaml`.

## Backends {#backends}

| Provider | Variável de ambiente | Busca | Extração | Free tier |
|----------|----------------------|-------|----------|-----------|
| **Firecrawl** (padrão) | `FIRECRAWL_API_KEY` | ✔ | ✔ | 500 credits/mo |
| **SearXNG** | `SEARXNG_URL` | ✔ | — | ✔ Grátis (self-hosted) |
| **Brave Search (free tier)** | `BRAVE_SEARCH_API_KEY` | ✔ | — | 2 000 queries/mo |
| **DDGS (DuckDuckGo)** | — (sem chave) | ✔ | — | ✔ Grátis |
| **Tavily** | `TAVILY_API_KEY` | ✔ | ✔ | 1 000 searches/mo |
| **Exa** | `EXA_API_KEY` | ✔ | ✔ | 1 000 searches/mo |
| **Parallel** | `PARALLEL_API_KEY` | ✔ | ✔ | Pago |
| **xAI (Grok)** | `XAI_API_KEY` ou `hermes auth add xai-oauth` | ✔ | — | Pago (SuperGrok ou por token) |

Brave Search, DDGS e xAI são **search-only** — combine qualquer um com Firecrawl/Tavily/Exa/Parallel quando também precisar de `web_extract`. DDGS usa o pacote Python [`ddgs`](https://pypi.org/project/ddgs/) por baixo; se ainda não estiver instalado, rode `pip install ddgs` (ou deixe o Hermes lazy-install na primeira uso). xAI roda a ferramenta server-side `web_search` do Grok na Responses API — resultados são gerados por LLM em vez de index-backed, então títulos, descrições e escolha de URL são toda saída do modelo (veja a [ressalva de trust model](#xai-grok) abaixo).

**Split por capacidade:** você pode usar providers diferentes para search e extract independentemente — por exemplo SearXNG (grátis) para search e Firecrawl para extract. Veja [Configuração por capacidade](#per-capability-configuration) abaixo.

:::tip Assinantes Nous
Se você tem assinatura paga do [Nous Portal](https://portal.nousresearch.com), web search e extract estão disponíveis pelo **[Tool Gateway](tool-gateway.md)** via Firecrawl gerenciado — sem chave de API. Instalações novas podem rodar `hermes setup --portal` para login e ligar todas as ferramentas do gateway de uma vez; instalações existentes podem ligar só web via `hermes tools`.
:::

---

## Como `web_extract` trata páginas longas {#how-web_extract-handles-long-pages}

Backends retornam markdown cru da página, que pode ser enorme (threads de fórum, sites de docs, artigos de notícias com comentários embutidos). Para manter sua janela de contexto usável e custos baixos, `web_extract` passa o conteúdo retornado pelo **modelo auxiliar `web_extract`** antes de entregar ao agente. O comportamento é puramente orientado a tamanho:

| Tamanho da página (caracteres) | O que acontece |
|------------------------|--------------|
| Abaixo de 5 000 | Retornado como está — sem chamada LLM, markdown completo chega ao agente |
| 5 000 – 500 000 | Resumo single-pass via modelo auxiliar `web_extract`, limitado a ~5 000 chars de saída |
| 500 000 – 2 000 000 | Chunked: dividido em chunks de 100 k chars, cada um resumido em paralelo, depois síntese final (~5 000 chars) |
| Acima de 2 000 000 | Recusado com dica para usar URL de fonte mais focada |

O resumo mantém citações, blocos de código e fatos-chave na formatação original — é um compressor de conteúdo, não um parafraseador. Se a sumarização falhar ou der timeout, o Hermes faz fallback aos primeiros ~5 000 chars de conteúdo cru em vez de erro inútil.

### Qual modelo faz a sumarização? {#which-model-does-the-summarizing}

A tarefa auxiliar `web_extract`. Por padrão (`auxiliary.web_extract.provider: "auto"`), este é seu **modelo principal de chat** — mesmo provider, mesmo modelo que `hermes model`. Isso funciona para a maioria dos setups, mas em modelos de raciocínio caros (Opus, MiniMax M2.7, etc.) cada extract de página longa adiciona custo significativo.

Para rotear resumos de extração a um modelo barato e rápido independentemente do principal:

```yaml
# ~/.hermes/config.yaml
auxiliary:
  web_extract:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 360       # seconds; raise if you hit summarization timeouts
```

Ou escolha interativamente: `hermes model` → **Configure auxiliary models** → `web_extract`.

Veja [Auxiliary Models](/user-guide/configuration#auxiliary-models) para referência completa e padrões de override por tarefa.

### Quando a sumarização atrapalha {#when-summarization-gets-in-the-way}

Se você precisa especificamente de conteúdo cru, não sumarizado — por exemplo, está raspando uma página estruturada onde o resumo LLM droparia campos importantes — use `browser_navigate` + `browser_snapshot`. A ferramenta browser retorna a árvore de acessibilidade live sem reescrita do modelo auxiliar (sujeito ao próprio cap de snapshot de 8 000 chars em páginas enormes).

---

## Setup {#setup}

### Setup rápido via `hermes tools` {#quick-setup-via-hermes-tools}

Rode `hermes tools`, navegue até **Busca web e extração** e escolha um provider. O wizard solicita a URL ou chave de API necessária e grava na sua config.

```bash
hermes tools
```

---

### Firecrawl (padrão) {#firecrawl-default}

Search e extract completos. Recomendado para a maioria dos usuários.

```bash
# ~/.hermes/.env
FIRECRAWL_API_KEY=fc-your-key-here
```

Obtenha uma chave em [firecrawl.dev](https://firecrawl.dev). O free tier inclui 500 credits/mês.

**Firecrawl self-hosted:** Aponte para sua própria instância em vez da API cloud:

```bash
# ~/.hermes/.env
FIRECRAWL_API_URL=http://localhost:3002
```

Quando `FIRECRAWL_API_URL` está definido, a chave de API é opcional (desabilite auth do servidor com `USE_DB_AUTHENTICATION=false`).

---

### SearXNG (grátis, self-hosted) {#searxng-free-self-hosted}

SearXNG é um metasearch engine open-source que respeita privacidade e agrega resultados de 70+ motores de busca. **Nenhuma chave de API necessária** — basta apontar o Hermes para uma instância SearXNG rodando.

SearXNG é **search-only** — `web_extract` requer um provider de extract separado.

#### Opção A — Self-host com Docker (recomendado) {#option-a--self-host-with-docker-recommended}

Isso dá uma instância privada sem rate limits.

**1. Crie um diretório de trabalho:**

```bash
mkdir -p ~/searxng/searxng
cd ~/searxng
```

**2. Escreva um `docker-compose.yml`:**

```yaml
# ~/searxng/docker-compose.yml
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
    restart: unless-stopped
```

**3. Inicie o container:**

```bash
docker compose up -d
```

**4. Habilite o formato JSON da API:**

SearXNG vem com saída JSON desabilitada por padrão. Copie a config gerada e habilite:

```bash
# Copy the auto-generated config out of the container
docker cp searxng:/etc/searxng/settings.yml ~/searxng/searxng/settings.yml
```

Abra `~/searxng/searxng/settings.yml`.
Se `use_default_settings: true` estiver presente, o arquivo contém apenas seus overrides. Todas as outras configurações são herdadas dos defaults built-in.
Para habilitar respostas JSON para o Hermes, adicione o override seguinte:

```yaml
search:
  formats:
    - html
    - json
```

Seu `settings.yml` deve parecer similar a:

```yaml
# Read the documentation before extending the defaults:
# https://docs.searxng.org/admin/settings/

use_default_settings: true

server:
  secret_key: "abcdef12345678"
  image_proxy: true

search:
  formats:
    - html
    - json
```

**5. Reinicie para aplicar:**

```bash
docker cp ~/searxng/searxng/settings.yml searxng:/etc/searxng/settings.yml
docker restart searxng
```

**6. Verifique que funciona:**

```bash
curl -s "http://localhost:8888/search?q=test&format=json" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"results\"])} results')"
```

Você deve ver algo como `10 results`. Se receber `403 Forbidden`, o formato JSON ainda está desabilitado — reverifique o passo 4.

**7. Configure o Hermes:**

```bash
# ~/.hermes/.env
SEARXNG_URL=http://localhost:8888
```

Depois selecione SearXNG como backend de search em `~/.hermes/config.yaml`:

```yaml
web:
  search_backend: "searxng"
```

Ou defina via `hermes tools` → Busca web e extração → SearXNG.

---

#### Opção B — Use uma instância pública {#option-b--use-a-public-instance}

Instâncias públicas SearXNG estão listadas em [searx.space](https://searx.space/). Filtre por instâncias com **formato JSON habilitado** (mostrado na tabela).

```bash
# ~/.hermes/.env
SEARXNG_URL=https://searx.example.com
```

:::caution Instâncias públicas
Instâncias públicas têm rate limits, uptime variável e podem desabilitar formato JSON a qualquer momento. Para uso em produção, self-hosting é fortemente recomendado.
:::

---

#### Combine SearXNG com um provider de extract {#pair-searxng-with-an-extract-provider}

SearXNG cuida do search; você precisa de um provider separado para `web_extract`. Use as chaves por capacidade:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"   # or tavily, exa, parallel
```

Com essa config, o Hermes usa SearXNG para todas as consultas de search e Firecrawl para extração de URL — combinando search grátis com extração de alta qualidade.

---

### Tavily {#tavily}

Search e extract otimizados para IA com free tier generoso.

```bash
# ~/.hermes/.env
TAVILY_API_KEY=tvly-your-key-here
```

Obtenha uma chave em [app.tavily.com](https://app.tavily.com/home). O free tier inclui 1 000 searches/mês.

---

### Exa {#exa}

Search neural com entendimento semântico. Bom para pesquisa e encontrar conteúdo conceitualmente relacionado.

```bash
# ~/.hermes/.env
EXA_API_KEY=your-exa-key-here
```

Obtenha uma chave em [exa.ai](https://exa.ai). O free tier inclui 1 000 searches/mês.

---

### Parallel {#parallel}

Search e extração AI-native com capacidades de deep research.

```bash
# ~/.hermes/.env
PARALLEL_API_KEY=your-parallel-key-here
```

Obtenha acesso em [parallel.ai](https://parallel.ai).

---

### xAI (Grok) {#xai-grok}

Roteia `web_search` pela ferramenta server-side [web_search tool](https://docs.x.ai/developers/tools/web-search) do Grok na Responses API. Grok executa a busca real e retorna os melhores resultados como JSON estruturado.

Funciona com qualquer caminho de credencial — sem novas env vars, sem novo wizard de setup:

```bash
# ~/.hermes/.env (env-var path)
XAI_API_KEY=sk-xai-your-key-here
```

ou para assinantes SuperGrok:

```bash
hermes auth add xai-oauth
```

Depois selecione xAI como backend de search:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "xai"
```

**Knobs opcionais:**

```yaml
web:
  backend: "xai"
  xai:
    model: grok-build-0.1        # reasoning model required by web_search (default)
    allowed_domains:             # optional, max 5 — mutex with excluded_domains
      - arxiv.org
    excluded_domains:            # optional, max 5
      - example-spam.com
    timeout: 90                  # seconds (default)
```

**Search-only** — combine com Firecrawl / Tavily / Exa / Parallel se também precisar de `web_extract`. Em 401 o provider faz um único refresh forçado de token OAuth e retenta (cobre revogação mid-window e tokens opacos que o check proativo de expiração não decodifica); credenciais env-var pulam o retry.

:::caution Trust model
Diferente de providers index-backed (Brave, Tavily, Exa) que retornam resultados verbatim de motor de busca, xAI é um LLM escolhendo quais URLs surfacear e escrevendo títulos e descrições. O *conteúdo* da consulta influencia a saída, então uma consulta maliciosamente craftada (ex.: injetada via input upstream não confiável que o agente pegou) pode em princípio steer Grok a emitir URLs escolhidas por atacante. Trate URLs retornadas como trataria qualquer link gerado pelo modelo — valide antes de buscar, especialmente se a consulta veio de input não confiável.
:::

---

## Configuração {#configuration}

### Backend único {#single-backend}

Defina um provider para todas as capacidades web:

```yaml
# ~/.hermes/config.yaml
web:
  backend: "searxng"   # firecrawl | searxng | brave-free | ddgs | tavily | exa | parallel | xai
```

### Configuração por capacidade {#per-capability-configuration}

Use providers diferentes para search vs extract. Isso permite combinar search grátis (SearXNG) com provider de extract pago, ou vice-versa:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "searxng"     # used by web_search
  extract_backend: "firecrawl"  # used by web_extract
```

Quando chaves por capacidade estão vazias, ambas fazem fallback para `web.backend`. Quando `web.backend` também está vazio, o backend é auto-detectado de qualquer chave/URL presente.

**Ordem de prioridade (por capacidade):**
1. `web.search_backend` / `web.extract_backend` (explícito por capacidade)
2. `web.backend` (fallback compartilhado)
3. Auto-detect de variáveis de ambiente

### Auto-detecção {#auto-detection}

Se nenhum backend estiver explicitamente configurado, o Hermes escolhe o primeiro disponível com base em quais credenciais estão definidas:

| Credencial presente | Backend auto-selecionado |
|--------------------|-----------------------|
| `FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL` | firecrawl |
| `PARALLEL_API_KEY` | parallel |
| `TAVILY_API_KEY` | tavily |
| `EXA_API_KEY` | exa |
| `SEARXNG_URL` | searxng |

xAI Web Search **não** está na cadeia de auto-detecção — ter `XAI_API_KEY` definida (ou estar logado via xAI Grok OAuth) não roteia automaticamente tráfego web pela xAI, já que essas credenciais também são usadas para inferência / TTS / image gen e o usuário pode querer backend diferente para web. Opt-in explicitamente com `web.backend: "xai"`.

---

## Verifique seu setup {#verify-your-setup}

Rode `hermes setup` para ver qual backend web é detectado:

```
✅ Busca web e extração (searxng)
```

Ou confira via CLI:

```bash
# Activate the venv and run the web tools module directly
source ~/.hermes/hermes-agent/.venv/bin/activate
python -m tools.web_tools
```

Isso imprime o backend ativo e seu status:

```
✅ Web backend: searxng
   Using SearXNG (search only): http://localhost:8888
```

---

## Solução de problemas {#troubleshooting}

### `web_search` retorna `{"success": false}`

- Verifique se `SEARXNG_URL` está acessível: `curl -s "http://localhost:8888/search?q=test&format=json"`
- Se receber HTTP 403, formato JSON está desabilitado — adicione `json` à lista `formats` em `settings.yml` e reinicie
- Se receber erro de conexão, o container pode não estar rodando: `docker ps | grep searxng`

### `web_extract` diz "search-only backend"

SearXNG não pode extrair conteúdo de URL. Defina `web.extract_backend` para um provider que suporte extração:

```yaml
web:
  search_backend: "searxng"
  extract_backend: "firecrawl"  # or tavily / exa / parallel
```

### SearXNG retorna 0 resultados

Algumas instâncias públicas desabilitam certos motores ou categorias. Tente:
- Uma consulta diferente
- Outra instância pública de [searx.space](https://searx.space/)
- Self-hosting sua própria instância para resultados confiáveis

### Rate limited em instância pública

Mude para instância self-hosted (veja [Opção A](#option-a--self-host-with-docker-recommended) acima). Com Docker, sua própria instância não tem rate limits.

### `web_extract` retorna conteúdo truncado com nota "summarization timed out"

O modelo auxiliar não terminou de sumarizar dentro do timeout configurado. Ou:

- Aumente `auxiliary.web_extract.timeout` em `config.yaml` (padrão 360s em instalações novas, 30s se a chave estiver ausente)
- Troque a tarefa auxiliar `web_extract` para um modelo mais rápido (ex.: `google/gemini-3-flash-preview`) — veja [Como `web_extract` trata páginas longas](#how-web_extract-handles-long-pages)
- Para páginas onde sumarização é a ferramenta errada, use `browser_navigate` em vez disso

---

## Skill opcional: `searxng-search` {#optional-skill-searxng-search}

Para agentes que precisam usar SearXNG via `curl` diretamente (ex.: como fallback quando o toolset web não está disponível), instale a skill opcional `searxng-search`:

```bash
hermes skills install official/research/searxng-search
```

Isso adiciona uma skill que ensina o agente a:
- Chamar a API JSON SearXNG via `curl` ou Python
- Filtrar por categoria (`general`, `news`, `science`, etc.)
- Tratar paginação e casos de erro
- Fazer fallback graciosamente quando SearXNG está inacessível
