---
title: Automação de navegador
description: Controle navegadores com vários provedores, navegadores locais da família Chromium via CDP ou navegadores na nuvem para interação web, preenchimento de formulários, scraping e muito mais.
sidebar_label: Navegador
sidebar_position: 5
---

# Automação de navegador

O Hermes Agent inclui um conjunto completo de ferramentas de automação de navegador com várias opções de backend:

- **Modo na nuvem Browserbase** via [Browserbase](https://browserbase.com) para navegadores gerenciados na nuvem e ferramentas anti-bot
- **Modo na nuvem Browser Use** via [Browser Use](https://browser-use.com) como provedor alternativo de navegador na nuvem
- **Modo na nuvem Firecrawl** via [Firecrawl](https://firecrawl.dev) para navegadores na nuvem com scraping integrado
- **Modo local Camofox** via [Camofox](https://github.com/jo-inc/camofox-browser) para navegação local anti-detecção (spoofing de fingerprint baseado em Firefox)
- **CDP local da família Chromium** — conecte as ferramentas de navegador à sua própria instância de Chrome, Brave, Chromium ou Edge usando `/browser connect`
- **Modo de navegador local** via a CLI `agent-browser` e uma instalação local do Chromium

Em todos os modos, o agente pode navegar em sites, interagir com elementos da página, preencher formulários e extrair informações.

## Visão geral {#overview}

As páginas são representadas como **árvores de acessibilidade** (snapshots baseados em texto), o que as torna ideais para agentes LLM. Elementos interativos recebem IDs de referência (como `@e1`, `@e2`) que o agente usa para clicar e digitar.

Principais capacidades:

- **Execução na nuvem com vários provedores** — Browserbase, Browser Use ou Firecrawl — sem necessidade de navegador local
- **Integração local da família Chromium** — conecte-se ao seu Chrome, Brave, Chromium ou Edge em execução via CDP para navegação prática
- **Stealth integrado** — fingerprints aleatórios, resolução de CAPTCHA, proxies residenciais (Browserbase)
- **Isolamento de sessão** — cada tarefa recebe sua própria sessão de navegador
- **Limpeza automática** — sessões inativas são fechadas após um timeout
- **Análise de visão** — screenshot + análise por IA para compreensão visual

## Configuração {#setup}

:::tip Assinantes Nous
Se você tem uma assinatura paga do [Nous Portal](https://portal.nousresearch.com), pode usar a automação de navegador pelo **[Tool Gateway](tool-gateway.md)** sem chaves de API separadas. Instalações novas podem executar `hermes setup --portal` para fazer login e ativar todas as ferramentas do gateway de uma vez; instalações existentes podem escolher **Nous Subscription** como provedor de navegador via `hermes model` ou `hermes tools`.
:::

### Modo na nuvem Browserbase

Para usar navegadores na nuvem gerenciados pelo Browserbase, adicione:

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

Obtenha suas credenciais em [browserbase.com](https://browserbase.com).

### Modo na nuvem Browser Use

Para usar o Browser Use como seu provedor de navegador na nuvem, adicione:

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

Obtenha sua chave de API em [browser-use.com](https://browser-use.com). O Browser Use fornece um navegador na nuvem via sua REST API. Se credenciais do Browserbase e do Browser Use estiverem definidas, o Browserbase tem prioridade.

### Modo na nuvem Firecrawl

Para usar o Firecrawl como seu provedor de navegador na nuvem, adicione:

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

Obtenha sua chave de API em [firecrawl.dev](https://firecrawl.dev). Em seguida, selecione o Firecrawl como seu provedor de navegador:

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

Configurações opcionais:

```bash
# Self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

### Roteamento híbrido: nuvem para URLs públicas, local para LAN/localhost

Quando um provedor na nuvem está configurado, o Hermes cria automaticamente um **sidecar Chromium local**
para URLs que resolvem para endereços privados/loopback/LAN (`localhost`, `127.0.0.1`,
`192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`, `*.local`, `*.lan`, `*.internal`,
loopback IPv6 `::1`, link-local `169.254.x.x`). URLs públicas continuam usando o
provedor na nuvem na mesma conversa.

Isso resolve o fluxo comum de "estou desenvolvendo localmente, mas usando Browserbase" —
o agente pode capturar screenshot do seu dashboard em `http://localhost:3000` E fazer scraping de
`https://github.com` sem você trocar de provedor ou desabilitar a proteção SSRF.
O provedor na nuvem nunca vê a URL privada.

O recurso está **ativado por padrão**. Para desabilitá-lo (todas as URLs vão para o provedor na nuvem
configurado, como antes):

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

Com o roteamento automático desabilitado, URLs privadas são rejeitadas com
`"Blocked: URL targets a private or internal address"`, a menos que você também defina
`browser.allow_private_urls: true` (o que permite que o provedor na nuvem tente acessá-las —
geralmente não funciona, já que Browserbase etc. não conseguem alcançar sua LAN).

Requisitos: o sidecar local usa a mesma CLI `agent-browser` do modo local puro,
portanto você precisa tê-la instalada (`hermes setup tools → Browser Automation`
instala automaticamente). Redirecionamentos pós-navegação de uma URL pública para um endereço
privado ainda são bloqueados (você não pode usar um truque de redirect-to-internal para alcançar
sua LAN pelo caminho público).

### Modo local Camofox

[Camofox](https://github.com/jo-inc/camofox-browser) é um servidor Node.js self-hosted que encapsula o Camoufox (um fork do Firefox com spoofing de fingerprint em C++). Ele oferece navegação local anti-detecção sem dependências de nuvem.

```bash
# Clone the Camofox browser server first
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# Build and start with Docker using the default container settings
# (auto-detects arch: aarch64 on M1/M2, x86_64 on Intel)
make up

# Stop and remove the default container
make down

# Force a clean rebuild (for example, after upgrading VERSION/RELEASE)
make reset

# Just download binaries without building
make fetch

# Override arch or version explicitly
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up` inicia o container padrão imediatamente. Se você quiser configurações de runtime personalizadas, como um heap Node maior, VNC ou um diretório de perfil persistente, construa a imagem primeiro e depois execute você mesmo:

```bash
# Build the image without starting the default container
make build

# Start with persistence, VNC live view, and a larger Node heap
mkdir -p ~/.camofox-docker
docker run -d \
  --name camofox-browser \
  --restart unless-stopped \
  -p 9377:9377 \
  -p 6080:6080 \
  -p 5901:5900 \
  -e CAMOFOX_PORT=9377 \
  -e ENABLE_VNC=1 \
  -e VNC_BIND=0.0.0.0 \
  -e VNC_RESOLUTION=1920x1080 \
  -e MAX_OLD_SPACE_SIZE=2048 \
  -v ~/.camofox-docker:/root/.camofox \
  camofox-browser:135.0.1-aarch64
```

Com o VNC habilitado, o navegador roda em modo headed e pode ser acompanhado ao vivo no seu navegador em `http://localhost:6080` (noVNC). Você também pode conectar um cliente VNC nativo em `localhost:5901`.

Se você já executou `make up`, pare e remova esse container padrão antes de iniciar o personalizado:

```bash
make down
# then run the custom docker run command above
```

Em seguida, defina em `~/.hermes/.env`:

```bash
CAMOFOX_URL=http://localhost:9377
```

Se o Camofox estiver rodando no Docker e você quiser que ele abra apps web servidos pela máquina host, habilite a reescrita de loopback. `CAMOFOX_URL` ainda deve apontar para a API de controle publicada pelo host, mas URLs de página como `http://127.0.0.1:3000` devem ser abertas de dentro do container como `http://host.docker.internal:3000`:

```yaml
# ~/.hermes/config.yaml
browser:
  camofox:
    rewrite_loopback_urls: true
    loopback_host_alias: host.docker.internal  # default; use a LAN IP if needed
```

Variáveis de ambiente equivalentes:

```bash
CAMOFOX_REWRITE_LOOPBACK_URLS=true
CAMOFOX_LOOPBACK_HOST_ALIAS=host.docker.internal
```

A reescrita se aplica apenas a URLs de navegação de página com hosts loopback (`localhost`, `127.0.0.1`, `::1`). Ela não altera `CAMOFOX_URL`. Deixe desabilitado para instalações Camofox fora do Docker, onde o navegador já roda no host e URLs loopback estão corretas.

Ou configure via `hermes tools` → Browser Automation → Camofox.

Quando `CAMOFOX_URL` está definido, todas as ferramentas de navegador roteiam automaticamente pelo Camofox em vez de Browserbase ou agent-browser.

#### Sessões persistentes de navegador

Por padrão, cada sessão Camofox recebe uma identidade aleatória — cookies e logins não sobrevivem entre reinicializações do agente. Para habilitar sessões persistentes de navegador, adicione o seguinte em `~/.hermes/config.yaml`:

```yaml
browser:
  camofox:
    managed_persistence: true
```

Em seguida, reinicie o Hermes completamente para que a nova configuração seja carregada.

:::warning O caminho aninhado importa
O Hermes lê `browser.camofox.managed_persistence`, **não** um `managed_persistence` de nível superior. Um erro comum é escrever:

```yaml
# ❌ Wrong — Hermes ignores this
managed_persistence: true
```

Se a flag estiver no caminho errado, o Hermes silenciosamente volta a um `userId` efêmero aleatório e seu estado de login será perdido a cada sessão.
:::

##### O que o Hermes faz
- Envia um `userId` determinístico com escopo de perfil ao Camofox para que o servidor possa reutilizar o mesmo perfil Firefox entre sessões.
- Pula a destruição de contexto no servidor durante a limpeza, para que cookies e logins sobrevivam entre tarefas do agente.
- Escopa o `userId` ao perfil Hermes ativo, para que perfis Hermes diferentes recebam perfis de navegador diferentes (isolamento de perfil).

##### O que o Hermes não faz
- Não força persistência no servidor Camofox. O Hermes apenas envia um `userId` estável; o servidor deve honrá-lo mapeando esse `userId` para um diretório persistente de perfil Firefox.
- Se sua build do servidor Camofox trata cada requisição como efêmera (por exemplo, sempre chama `browser.newContext()` sem carregar um perfil armazenado), o Hermes não consegue tornar essas sessões persistentes. Certifique-se de estar rodando uma build Camofox que implemente persistência de perfil baseada em userId.

##### Verifique se está funcionando

1. Inicie o Hermes e seu servidor Camofox.
2. Abra o Google (ou qualquer site de login) em uma tarefa de navegador e faça login manualmente.
3. Encerre a tarefa de navegador normalmente.
4. Inicie uma nova tarefa de navegador.
5. Abra o mesmo site novamente — você ainda deve estar autenticado.

Se a etapa 5 fizer logout, o servidor Camofox não está honrando o `userId` estável. Confira o caminho da configuração, confirme que reiniciou o Hermes completamente após editar `config.yaml` e verifique se sua versão do servidor Camofox suporta perfis persistentes por usuário.

##### Onde o estado fica

O Hermes deriva o `userId` estável do diretório com escopo de perfil `~/.hermes/browser_auth/camofox/` (ou o equivalente em `$HERMES_HOME` para perfis não padrão). Os dados reais do perfil de navegador ficam no lado do servidor Camofox, indexados por esse `userId`. Para resetar completamente um perfil persistente, limpe-o no servidor Camofox e remova o diretório de estado correspondente do perfil Hermes.

#### Sessões Camofox gerenciadas externamente

Quando outro app controla o navegador Camofox visível (um assistente desktop, uma integração personalizada, outro agente), configure o Hermes para operar dentro dessa mesma identidade em vez de criar seu próprio perfil isolado.

Três controles definem o comportamento:

| Configuração | Env var | Efeito |
|---------|---------|--------|
| `browser.camofox.user_id` | `CAMOFOX_USER_ID` | `userId` Camofox que o Hermes usa ao criar abas. Definir isso coloca a sessão em modo "externally managed". |
| `browser.camofox.session_key` | `CAMOFOX_SESSION_KEY` | `sessionKey` (também conhecido como `listItemId`) enviado na criação de aba. Usado para corresponder a uma aba existente durante a adoção. Padrão: um valor por tarefa se não definido. |
| `browser.camofox.adopt_existing_tab` | `CAMOFOX_ADOPT_EXISTING_TAB` | Quando true, o Hermes chama `GET /tabs?userId=<user_id>` no primeiro uso e reutiliza uma aba existente antes de criar uma nova. |

Variáveis de ambiente têm precedência sobre `config.yaml`. Qualquer forma funciona:

```yaml
browser:
  camofox:
    user_id: shared-camofox
    session_key: visible-tab
    adopt_existing_tab: true
```

```bash
CAMOFOX_USER_ID=shared-camofox
CAMOFOX_SESSION_KEY=visible-tab
CAMOFOX_ADOPT_EXISTING_TAB=true
```

**O que muda quando `user_id` está definido:**

- O Hermes pula a limpeza destrutiva no fim da tarefa (igual a `managed_persistence: true`). A aba/cookies/perfil do outro app sobrevivem.
- O Hermes **não** chama `DELETE /sessions/<user_id>` — esse endpoint apaga todos os dados do usuário, então dispararia e destruiria a sessão do app externo.

**Como funciona a adoção de aba (quando `adopt_existing_tab: true`):**

1. Na primeira chamada de ferramenta de navegador após iniciar o processo, o Hermes emite `GET /tabs?userId=<user_id>` (timeout de 5 segundos).
2. Se alguma aba na resposta tiver `listItemId == session_key`, o Hermes adota a mais recentemente criada nesse grupo.
3. Caso contrário, o Hermes adota a aba mais recentemente criada do usuário (qualquer `listItemId`).
4. Se não existirem abas ou a requisição falhar, o Hermes volta a criar uma nova aba na próxima operação.

A adoção só dispara até `tab_id` ser preenchido para a sessão. Se o app externo fechar a aba adotada no meio da execução, a próxima chamada de ferramenta de navegador exibirá um erro Camofox — o Hermes não faz re-poll de uma aba nova a cada chamada.

**Escolhendo `session_key`:** se você quer que o Hermes se conecte de forma confiável a uma aba existente *específica*, defina `session_key` como o `listItemId` que o app externo usou ao criá-la. Se você deixar `session_key` indefinido e definir apenas `user_id`, o Hermes gera um `session_key` por tarefa (`task_<id>`) — o Hermes compartilhará cookies e o perfil com o app externo, mas abrirá sua própria aba ao lado em vez de reutilizar uma.

**Nota sobre concorrência:** o app externo e o Hermes podem controlar o mesmo `userId` Camofox simultaneamente, mas o Camofox não coordena foco por aba entre clientes. Coordene a propriedade na camada de aplicação (por exemplo, o app externo pausa enquanto o Hermes executa).

#### Visualização ao vivo via VNC

Quando o Camofox roda em modo headed (com janela de navegador visível), ele expõe uma porta VNC na resposta do health check. O Hermes descobre isso automaticamente e inclui a URL VNC nas respostas de navegação, para que o agente possa compartilhar um link para você acompanhar o navegador ao vivo.

### Navegador local da família Chromium via CDP (`/browser connect`)

Em vez de um provedor na nuvem, você pode conectar as ferramentas de navegador do Hermes à sua própria instância de Chrome, Brave, Chromium ou Edge em execução via Chrome DevTools Protocol (CDP). Isso é útil quando você quer ver o que o agente está fazendo em tempo real, interagir com páginas que exigem seus próprios cookies/sessões ou evitar custos de navegador na nuvem.

:::note
`/browser connect` é um **comando slash interativo da CLI** — não é despachado pelo gateway. Se você tentar executá-lo dentro de uma WebUI, Telegram, Discord ou outro chat do gateway, a mensagem será enviada ao agente como texto simples e o comando não será executado. Inicie o Hermes pelo terminal (`hermes` ou `hermes chat`) e emita `/browser connect` lá.
:::

Na CLI, use:

```
/browser connect                 # Auto-launch/connect to a local Chromium-family browser at http://127.0.0.1:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status                  # Check current connection
/browser disconnect              # Detach and return to cloud/local mode
```

Se um navegador ainda não estiver rodando com remote debugging, o Hermes tentará iniciar automaticamente um navegador suportado da família Chromium com `--remote-debugging-port=9222`. A detecção inclui Brave, Google Chrome, Chromium e Microsoft Edge, com caminhos comuns de instalação no Linux como `/opt/brave-bin/brave` e `/snap/bin/brave`.

:::tip
Para iniciar manualmente um navegador da família Chromium com CDP habilitado, use um user-data-dir dedicado para que a porta de debug realmente suba mesmo se o navegador já estiver rodando com seu perfil normal:

```bash
# Linux — Brave
brave-browser \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# Linux — Google Chrome
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS — Brave
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &

# macOS — Google Chrome
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

Em seguida, inicie a CLI do Hermes e execute `/browser connect`.

**Por que `--user-data-dir`?** Sem ele, iniciar um navegador da família Chromium enquanto uma instância regular já está rodando normalmente abre uma nova janela no processo existente — e esse processo existente não foi iniciado com `--remote-debugging-port`, então a porta 9222 nunca abre. Um user-data-dir dedicado força um processo de navegador novo onde a porta de debug realmente escuta. `--no-first-run --no-default-browser-check` pula o assistente de primeira execução para o perfil novo.
:::

Quando conectado via CDP, todas as ferramentas de navegador (`browser_navigate`, `browser_click`, etc.) operam na sua instância de navegador ao vivo em vez de criar uma sessão na nuvem.

### WSL2 + Chrome no Windows: prefira MCP em vez de `/browser connect`

Se o Hermes roda dentro do WSL2, mas a janela do Chrome que você quer controlar roda no host Windows, `/browser connect` muitas vezes não é o melhor caminho.

Por quê:

- `/browser connect` espera que o próprio Hermes alcance um endpoint CDP utilizável
- sessões modernas de live-debugging do Chrome frequentemente expõem um endpoint local do host que não é diretamente acessível do WSL da mesma forma que uma porta clássica `9222`
- mesmo quando o Chrome no Windows é depurável, a integração mais limpa costuma ser deixar um servidor MCP de navegador no lado Windows conectar ao Chrome e deixar o Hermes falar com esse servidor MCP

Para essa configuração, prefira `chrome-devtools-mcp` pelo suporte MCP do Hermes.

Veja o guia MCP para a configuração prática:

- [Use MCP with Hermes](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### Modo de navegador local

Se você **não** definir credenciais na nuvem e não usar `/browser connect`, o Hermes ainda pode usar as ferramentas de navegador por meio de uma instalação local do Chromium controlada por `agent-browser`.

### Variáveis de ambiente opcionais

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in seconds (max 21600 = 6 hours) (default: project default)
# Examples: 600 (10min), 1800 (30min), 21600 (6h max)
BROWSERBASE_SESSION_TIMEOUT=1800

# Inactivity timeout before auto-cleanup in seconds (default: 120)
BROWSER_INACTIVITY_TIMEOUT=120

# Extra Chromium launch flags (comma- or newline-separated). Hermes auto-injects
# `--no-sandbox,--disable-dev-shm-usage` when it detects root or AppArmor-restricted
# unprivileged user namespaces (Ubuntu 23.10+, DGX Spark, many container images),
# so most users don't need to set this. Set it manually only if you need a flag
# Hermes doesn't add automatically; setting it disables the auto-injection.
AGENT_BROWSER_ARGS=--no-sandbox
```

### Instalar a CLI agent-browser

```bash
npm install -g agent-browser
# Or install locally in the repo:
npm install
```

:::info
O toolset `browser` deve estar incluído na lista `toolsets` da sua config ou habilitado via `hermes config set toolsets '["hermes-cli", "browser"]'`.
:::

## Ferramentas disponíveis {#available-tools}

### `browser_navigate`

Navega até uma URL. Deve ser chamado antes de qualquer outra ferramenta de navegador. Inicializa a sessão Browserbase.

```
Navigate to https://github.com/NousResearch
```

:::tip
Para recuperação simples de informação, prefira `web_search` ou `web_extract` — são mais rápidos e baratos. Use ferramentas de navegador quando precisar **interagir** com uma página (clicar em botões, preencher formulários, lidar com conteúdo dinâmico).
:::

### `browser_snapshot`

Obtém um snapshot baseado em texto da árvore de acessibilidade da página atual. Retorna elementos interativos com IDs de referência como `@e1`, `@e2` para uso com `browser_click` e `browser_type`.

- **`full=false`** (padrão): visualização compacta mostrando apenas elementos interativos
- **`full=true`**: conteúdo completo da página

Snapshots com mais de 15.000 caracteres são automaticamente truncados ou resumidos por um LLM (o mesmo orçamento por página de `web_extract`). Quando isso acontece, o snapshot completo é salvo em `~/.hermes/cache/web/` e a saída da ferramenta inclui o caminho do arquivo mais uma chamada pronta para uso de `read_file`, para que o agente possa percorrer a árvore de acessibilidade completa — incluindo refs de elementos além do corte — sem refazer snapshot.

### `browser_click`

Clica em um elemento identificado pelo ID de referência do snapshot.

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

Digita texto em um campo de entrada. Limpa o campo primeiro e depois digita o novo texto.

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

Rola a página para cima ou para baixo para revelar mais conteúdo.

```
Scroll down to see more results
```

### `browser_press`

Pressiona uma tecla do teclado. Útil para enviar formulários ou navegação.

```
Press Enter to submit the form
```

Teclas suportadas: `Enter`, `Tab`, `Escape`, `ArrowDown`, `ArrowUp` e mais.

### `browser_back`

Navega de volta à página anterior no histórico do navegador.

### `browser_get_images`

Lista todas as imagens na página atual com suas URLs e texto alt. Útil para encontrar imagens para analisar.

### `browser_vision`

Captura um screenshot e analisa com IA de visão. Use quando snapshots de texto não capturam informação visual importante — especialmente útil para CAPTCHAs, layouts complexos ou desafios de verificação visual.

O screenshot é salvo de forma persistente e o caminho do arquivo é retornado junto com a análise por IA. Em plataformas de mensagens (Telegram, Discord, Slack, WhatsApp), você pode pedir ao agente para compartilhar o screenshot — ele será enviado como anexo de foto nativo via o mecanismo `MEDIA:`.

```
What does the chart on this page show?
```

Screenshots ficam armazenados em `~/.hermes/cache/screenshots/` e são limpos automaticamente após 24 horas.

### `browser_console`

Obtém a saída do console do navegador (mensagens log/warn/error) e exceções JavaScript não capturadas da página atual. Essencial para detectar erros JS silenciosos que não aparecem na árvore de acessibilidade.

```
Check the browser console for any JavaScript errors
```

Use `clear=True` para limpar o console após a leitura, para que chamadas subsequentes mostrem apenas mensagens novas.

`browser_console` também avalia JavaScript quando chamado com um argumento `expression` — mesma forma do console DevTools, o resultado volta parseado (objetos serializados em JSON viram dicts; valores primitivos permanecem primitivos).

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

Quando um supervisor CDP está ativo para a sessão atual (típico para qualquer sessão que executou `browser_navigate` contra um backend compatível com CDP), a avaliação roda pelo WebSocket persistente do supervisor — sem custo de startup de subprocesso. Caso contrário, cai no caminho padrão da CLI agent-browser. O comportamento é idêntico de qualquer forma; apenas a latência muda.

A avaliação é irrestrita por padrão — o agente pode usar `fetch`, ler storage, consultar valores de formulário e executar qualquer extração DOM. Requisições direcionadas a endereços privados/internos ainda são bloqueadas em backends não locais (a proteção SSRF é independente dessa configuração). Se você navega em páginas hostis com um perfil logado e quer uma denylist estrita sobre primitivas JS sensíveis (cookies, storage, clipboard, chamadas de rede, valores de formulário), opte por `browser.restrict_evaluate: true` em `config.yaml`. Note que a denylist corresponde a *nomes* de primitivas, então também bloqueia expressões legítimas que apenas contenham palavras como `fetch` ou `cookie`.

### `browser_cdp`

Passagem direta do Chrome DevTools Protocol — a saída de emergência para operações de navegador não cobertas pelas outras ferramentas. Use para tratamento de diálogos nativos, avaliação com escopo de iframe, controle de cookies/rede ou qualquer verbo CDP que o agente precise.

**Disponível apenas quando um endpoint CDP é acessível no início da sessão** — ou seja, `/browser connect` conectou a um Chrome, Brave, Chromium ou Edge em execução, ou `browser.cdp_url` está definido em `config.yaml`. O modo local padrão agent-browser, Camofox e provedores na nuvem (Browserbase, Browser Use, Firecrawl) atualmente não expõem CDP para esta ferramenta — provedores na nuvem têm URLs CDP por sessão, mas roteamento de sessão ao vivo é um follow-up.

**Referência de métodos CDP:** https://chromedevtools.github.io/devtools-protocol/ — o agente pode usar `web_extract` na página de um método específico para consultar parâmetros e forma de retorno.

Padrões comuns:

```
# List tabs (browser-level, no target_id)
browser_cdp(method="Target.getTargets")

# Handle a native JS dialog on a tab
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# Evaluate JS in a specific tab
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# Get all cookies
browser_cdp(method="Network.getAllCookies")
```

Métodos de nível de navegador (`Target.*`, `Browser.*`, `Storage.*`) omitem `target_id`. Métodos de nível de página (`Page.*`, `Runtime.*`, `DOM.*`, `Emulation.*`) exigem um `target_id` de `Target.getTargets`. Cada chamada stateless é independente — sessões não persistem entre chamadas.

**Iframes cross-origin:** passe `frame_id` (de `browser_snapshot.frame_tree.children[]` onde `is_oopif=true`) para rotear a chamada CDP pela sessão ao vivo do supervisor para esse iframe. É assim que `Runtime.evaluate` dentro de um iframe cross-origin funciona no Browserbase, onde conexões CDP stateless atingiriam expiração de signed-URL. Exemplo:

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

Iframes same-origin não precisam de `frame_id` — use `document.querySelector('iframe').contentDocument` a partir de um `Runtime.evaluate` de nível superior.

### `browser_dialog`

Responde a um diálogo JS nativo (`alert` / `confirm` / `prompt` / `beforeunload`). Antes desta ferramenta existir, diálogos bloqueavam silenciosamente a thread JavaScript da página e chamadas subsequentes de `browser_*` travavam ou lançavam erro; agora o agente vê diálogos pendentes na saída de `browser_snapshot` e responde explicitamente.

**Fluxo de trabalho:**
1. Chame `browser_snapshot`. Se um diálogo estiver bloqueando a página, ele aparece como `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]`.
2. Chame `browser_dialog(action="accept")` ou `browser_dialog(action="dismiss")`. Para diálogos `prompt()`, passe `prompt_text="..."` para fornecer a resposta.
3. Refaça snapshot — `pending_dialogs` está vazio; a thread JS da página retomou.

**A detecção acontece automaticamente** via um supervisor CDP persistente — um WebSocket por tarefa que assina eventos Page/Runtime/Target. O supervisor também preenche um campo `frame_tree` no snapshot para que o agente veja a estrutura de iframe da página atual, incluindo iframes cross-origin (OOPIF).

**Matriz de disponibilidade:**

| Backend | Detecção via `pending_dialogs` | Resposta (ferramenta `browser_dialog`) |
|---|---|---|
| Chrome local via `/browser connect` ou `browser.cdp_url` | ✓ | ✓ fluxo completo |
| Browserbase | ✓ | ✓ fluxo completo (via ponte XHR injetada) |
| Camofox / agent-browser local padrão | ✗ | ✗ (sem endpoint CDP) |

**Como funciona no Browserbase.** O proxy CDP do Browserbase descarta automaticamente diálogos nativos reais no lado do servidor em ~10ms, então não podemos usar `Page.handleJavaScriptDialog`. O supervisor injeta um script pequeno via `Page.addScriptToEvaluateOnNewDocument` que substitui `window.alert`/`confirm`/`prompt` por um XHR síncrono. Interceptamos esses XHRs via `Fetch.enable` — a thread JS da página permanece bloqueada no XHR até chamarmos `Fetch.fulfillRequest` com a resposta do agente. Valores de retorno de `prompt()` voltam para o JS da página inalterados.

**Política de diálogo** é configurada em `config.yaml` em `browser.dialog_policy`:

| Política | Comportamento |
|--------|----------|
| `must_respond` (padrão) | Captura, exibe no snapshot, aguarda chamada explícita de `browser_dialog()`. Auto-dismiss de segurança após `browser.dialog_timeout_s` (padrão 300s) para que um agente com bug não trave para sempre. |
| `auto_dismiss` | Captura, descarta imediatamente. O agente ainda vê o diálogo no histórico de `browser_state`, mas não precisa agir. |
| `auto_accept` | Captura, aceita imediatamente. Útil ao navegar em páginas com prompts `beforeunload` agressivos. |

**Árvore de frames** dentro de `browser_snapshot.frame_tree` é limitada a 30 frames e profundidade OOPIF 2 para manter payloads limitados em páginas cheias de anúncios. Uma flag `truncated: true` aparece quando os limites foram atingidos; agentes que precisam da árvore completa podem usar `browser_cdp` com `Page.getFrameTree`.

## Exemplos práticos {#practical-examples}

### Preenchendo um formulário web

```
User: Sign up for an account on example.com with my email john@example.com

Agent workflow:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → sees form fields with refs
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → clicks "Create Account"
6. browser_snapshot()  → confirms success
```

### Pesquisando conteúdo dinâmico

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
```

## Gravação de sessão {#session-recording}

Grave automaticamente sessões de navegador como arquivos de vídeo WebM:

```yaml
browser:
  record_sessions: true  # default: false
```

Quando habilitado, a gravação começa automaticamente no primeiro `browser_navigate` e salva em `~/.hermes/browser_recordings/` quando a sessão fecha. Funciona nos modos local e na nuvem (Browserbase). Gravações com mais de 72 horas são limpas automaticamente.

## Modo headed (janela visível do navegador) {#headed-mode-visible-browser-window}

Por padrão, o navegador local roda headless. Habilite o modo headed para obter uma janela Chromium visível que você pode acompanhar e com a qual pode interagir:

```yaml
browser:
  headed: true  # default: false
```

Ou via variável de ambiente: `AGENT_BROWSER_HEADED=1`.

O modo headed faz duas coisas:

1. **Inicia o Chromium com uma janela visível** (passa `--headed` ao agent-browser no modo local).
2. **Mantém a janela aberta entre turnos.** Normalmente a sessão de navegador é limpa após cada resposta do agente; no modo headed a limpeza por turno é ignorada para que você possa acompanhar o agente trabalhando, intervir manualmente (desafios de login, CAPTCHAs) e manter o estado de login aquecido ao longo da conversa.

Sessões ociosas ainda são encerradas após `browser.inactivity_timeout` (padrão 120s sem atividade de navegador), e todas as sessões são fechadas no shutdown. O modo headed afeta apenas o navegador local — sessões na nuvem (Browserbase) não são afetadas.

## Recursos de stealth {#stealth-features}

O Browserbase fornece capacidades de stealth automáticas:

| Recurso | Padrão | Notas |
|---------|---------|-------|
| Basic Stealth | Sempre ativo | Fingerprints aleatórios, randomização de viewport, resolução de CAPTCHA |
| Residential Proxies | Ativo | Roteia por IPs residenciais para melhor acesso |
| Advanced Stealth | Desligado | Build Chromium personalizado, requer Scale Plan |
| Keep Alive | Ativo | Reconexão de sessão após instabilidades de rede |

:::note
Se recursos pagos não estiverem disponíveis no seu plano, o Hermes faz fallback automaticamente — primeiro desabilitando `keepAlive`, depois proxies — para que a navegação continue funcionando em planos gratuitos.
:::

## Gerenciamento de sessão {#session-management}

- Cada tarefa recebe uma sessão de navegador isolada via Browserbase
- Sessões são limpas automaticamente após inatividade (padrão: 2 minutos)
- Uma thread em background verifica a cada 30 segundos por sessões obsoletas
- Limpeza de emergência roda na saída do processo para evitar sessões órfãs
- Sessões são liberadas via API Browserbase (status `REQUEST_RELEASE`)

## Limitações {#limitations}

- **Interação baseada em texto** — depende da árvore de acessibilidade, não de coordenadas de pixel
- **Tamanho do snapshot** — páginas grandes podem ser truncadas ou resumidas por LLM em 15.000 caracteres (igual a `web_extract`); o snapshot completo é salvo em `~/.hermes/cache/web/` e a saída aponta para ele para paginação com `read_file`
- **Timeout de sessão** — sessões na nuvem expiram conforme as configurações do plano do seu provedor
- **Custo** — sessões na nuvem consomem créditos do provedor; sessões são limpas automaticamente quando a conversa termina ou após inatividade. Use `/browser connect` para navegação local gratuita.
- **Sem downloads de arquivo** — não é possível baixar arquivos pelo navegador
