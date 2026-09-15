# Inteligência Centralizada — Hermes Workstation (Hermes Work)

Este documento atua como a **base de conhecimento canônica e fonte única da verdade (inteligência centralizada)** sobre o funcionamento, a arquitetura de baixo nível, os contratos de persistência e a integração do **Hermes Workstation (Hermes Work)** nesta branch/fork downstream do repositório Hermes Agent.

Qualquer desenvolvedor ou agente de IA que for trabalhar neste domínio **DEVE** ler este documento para se situar sobre os conceitos, invariantes arquiteturais, armadilhas conhecidas e restrições estabelecidas antes de modificar qualquer código em `apps/desktop/`, `workstation/`, `tools/browser_workstation.py` ou superfícies de integração associadas.

> **Atualização operacional:** este documento foi enriquecido com as validações
> de runtime, persistência, Gateway, superfícies de cliente e qualificação de
> release realizadas na sessão de 2026-09-12/14. Os contratos de código e os
> testes em `main` continuam sendo a autoridade final quando houver divergência.

---

## 1. O que é o Hermes Workstation?

O **Hermes Workstation** é uma camada de produto de primeira classe que adiciona uma interface rica de Desktop (via Electron) e um motor de navegação isolado para automação de tarefas web e desktop com controle híbrido (Humano + Agente).

Em vez de simplesmente rodar no terminal ou usar navegadores remotos em nuvem de forma volátil, o Hermes Work integra uma aplicação Desktop nativa com um **perfil de Chromium próprio e dedicado** (totalmente isolado do navegador pessoal Chrome/Edge/Brave do usuário). Isso permite navegar, raspar dados, autenticar em serviços e agir na web em nome do usuário com persistência de sessões, sem vazamento de dados privados.

A arquitetura trata o Desktop e o Core Agent como entidades desacopladas que podem rodar em processos separados, máquinas distintas ou através de pontes de rede (loopback, LAN autenticada, Tailscale), mantendo invariantes rigorosos de segurança e persistência de sessão.

### Princípios Fundamentais:
1. **Controle Híbrido Sem Fricção:** O usuário e o agente podem alternar a qualquer momento a posse da aba ativa ("Take Control" / "Release Control") sem quebrar a sessão ou perder o estado do DOM.
2. **Economia Radical de Tokens:** O sistema prioriza percepção estruturada/semântica do DOM em vez de queimar milhares de tokens de visão com capturas de tela contínuas a cada clique.
3. **Isolamento e Segurança Fail-Closed:** O controlador de browser é acessível apenas em loopback (`127.0.0.1`), protegido por Bearer Token único. Caso a infraestrutura do browser caia durante uma tarefa ativa, o agente falha de forma fechada (*fail-closed*), impedindo vazamento de dados para instâncias genéricas ou sem cookies.
4. **Cintura Estreita (Narrow Waist):** A capacidade do navegador de desktop é atribuída dinamicamente à sessão ativa no Gateway, nunca poluindo o schema central permanente do LLM nem quebrando o cache de prompt (*prompt caching*).

---

## 2. Componentes Principais

### 2.1. Electron Desktop App (`apps/desktop/`)
Aplicação local do usuário construída com Electron + Vite + React.
- **Superfície de Conversação:** Chat construído com `@assistant-ui/react`, gerenciando streaming de tokens, blocos de ferramentas e visualização de artefatos.
- **Superfícies de Exibição do Navegador:** O navegador interno NÃO é um `<iframe>` nem um webview HTML comum; ele utiliza **`WebContentsView`** nativo do Chromium acoplado diretamente ao compositor da janela principal (`BrowserWindow`).
- **Hosts de Exibição:** Engloba o painel lateral contextual do Chat (`WorkstationBrowserPane`), o painel global de tarefas (`BrowserHub`) e o painel Kanban.

### 2.2. Chromium / BrowserRuntime (`WorkstationBrowserRuntime`)
O motor interno do browser localizado em `apps/desktop/electron/workstation-browser-runtime.ts`. O Hermes não usa os perfis pessoais do usuário nem contamina o navegador do sistema operacional.
- **Isolamento de Dados:** Gerencia seu próprio diretório de perfil (`HermesWorkstation/Browser/User Data`), com banco SQLite de cookies, armazenamento LocalStorage e cache isolados.
- **Controle CDP de Baixo Nível:** Emite cliques e digitação usando Chrome DevTools Protocol (`wc.debugger`) diretamente no compositor do Chromium, gerando eventos de mouse com coordenadas reais (`cdpClick`) em vez de scripts sintéticos JS que falham em SPAs modernos.
- **Desacoplamento Arquitetural:** A responsabilidade do `BrowserRuntime` é abstrata. A implementação canônica apoia-se no Chromium do Electron, mas a arquitetura é projetada para não acoplar irreversivelmente regras de negócio ao Electron (permitindo adapters alternativos como headless Lightpanda ou Chromium remoto).

### 2.3. O Conceito Central de `BrowserTask`
Este é o conceito **MAIS IMPORTANTE** da abstração web. Uma `BrowserTask` representa a **posse semântica** e o **ciclo de vida estrutural** de uma tarefa de automação:
- **Relacionamento 1-para-1 estrito:** No processo Electron, uma `BrowserTask` possui no máximo uma aba ao vivo (`live page` / `BrowserEntry`) vinculada pelo identificador `ownerTaskId`. Uma tarefa restaurada pode permanecer lazy, sem `WebContents`, até ser materializada; nunca existem duas páginas independentes para o mesmo `taskId` nem uma página compartilhada por tarefas.
- **Ciclo de vida flexível e estados:** O contrato persistido usa `visible`, `hidden` e `parked`, com `fresh`, `restored` e `recreated` como estados de recuperação. Rótulos como `active`, `waiting-for-human`, `background`, `recent` e `stalled` são projeções operacionais da Task Rail/recursos, não novos estados duráveis. A tarefa pode ser escondida (`hide`), estacionada (`park`) e exibida (`show`).
- **Ocultar/Estacionar não destrói a página:** Invocar `hideTask` ou `parkTask` **NÃO encerra** o processo ou o objeto da página (o DOM, listeners, variáveis e contexto JS continuam vivos na memória do processo). Apenas a visualização no host (janela Electron) é destacada/removida (`window.contentView.removeChildView`).
- **Destruição explícita:** Uma task só é descartada e liberada da memória quando há invocação explícita de `destroyTask(taskId)` ou ação direta de fechamento pelo usuário.

### 2.4. Persistência Estrutural (`BrowserSessionState`)
O `BrowserSessionState` é o subsistema responsável por capturar o estado estrutural das abas e tarefas, salvando-o atomicamente em disco:
- **Onde reside:** O arquivo composto canônico é `workstationBasePath() / Runtime / browser-session.json`. Ele contém as abas lógicas, a aba ativa e o snapshot de `BrowserTask`. O antigo `browser-tasks.json` continua sendo aceito apenas como origem de migração quando o arquivo composto ainda não existe; o fluxo normal grava a projeção composta.
- **Resolução do diretório:** `HERMES_WORKSTATION_HOME` pode apontar para uma raiz isolada (obrigatório em validações de candidato); sem override, Windows usa `%LOCALAPPDATA%\HermesWorkstation`, macOS usa `~/Library/Application Support/HermesWorkstation` e Linux usa `$XDG_CONFIG_HOME/HermesWorkstation` ou `~/.config/HermesWorkstation`.
- **Separação de responsabilidades:** O estado de sessão persistido é estritamente **estrutural** (IDs de abas, ordem no array, apontador de aba ativa, políticas de recuperação e URLs seguras).
- **Invariante de Segurança:** Segredos digitados, senhas, tokens de autenticação extraídos e estado volátil da heap Javascript **nunca** são persistidos nesses arquivos JSON em texto claro.
- **Sanitização:** A URL persistida aceita somente `about:blank` ou HTTP(S), rejeita userinfo, barras invertidas, caracteres de controle, percent-encoding malformado, query/hash e padrões reconhecíveis de credencial, JWT, OTP ou token opaco. O limite é 2.048 caracteres; títulos de página não atravessam o boundary durável (`safeTitleMetadata` retorna `null`). Há no máximo 128 abas e uma aba por `BrowserTask`; duplicatas inválidas são descartadas.
- **Escrita atômica e tolerância a Windows:** a projeção é normalizada antes da escrita, gravada em temporário privado `0600` e promovida por rename. Em `EPERM`/`EBUSY`, o runtime tenta `copyFileSync` e remove o temporário. Se a troca falhar, a última intenção normalizada permanece em memória para a próxima tentativa, enquanto o disco conserva o snapshot completo anterior. Versões futuras não são sobrescritas nem rebaixadas por uma versão antiga.
- **Restart Recovery (Restauração Preguiçosa / Lazy):** Ao reiniciar a aplicação, as tarefas prévias são restauradas logicamente em estado `parked`, com `recoveryState: 'restored'`. Abas comuns podem ser recriadas a partir de URL segura; abas de tarefa ficam pendentes e a página Chromium real só é instanciada ao usar/mostrar a tarefa. Se a página recuperada já estiver `stale/page-gone`, um novo ID de view é criado sem duplicar a identidade lógica da tarefa. Isso economiza memória e deixa explícita a diferença entre recuperar metadados e recuperar o objeto `WebContents`.

---

## 3. Topologia de Comunicação e Protocolos

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HERMES CORE AGENT                              │
│                      (CLI / Gateway / Tool Calling Loop)                    │
└───────────────────────┬─────────────────────────────────────▲───────────────┘
                        │ HTTP POST (JSON)                    │
                        │ Loopback 127.0.0.1:<port>           │ Resposta Sanitizada
                        │ Authorization: Bearer <token>       │ (Force-Redacted)
                        ▼                                     │
┌─────────────────────────────────────────────────────────────┴───────────────┐
│                       WORKSTATION CONTROLLER SERVICE                        │
│               (Loopback HTTP Server em browser-control.json)                │
└───────────────────────┬─────────────────────────────────────────────────────┘
                        │ Chamadas Internas / Métodos de Runtime
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WorkstationBrowserRuntime                           │
│                      (Electron Main Process Layer)                          │
│                                                                             │
│  ┌───────────────────────────────┐     IPC (preload.ts bridge)              │
│  │   Chromium WebContentsView    │◄─────────────────────────────┐           │
│  │   (Aba viva / CDP Input)      │                              │           │
│  └───────────────────────────────┘                              │           │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
                                   ┌──────────────────────────────┴───────────┐
                                   │         Desktop UI (React Renderer)      │
                                   │  - Chat (WorkstationBrowserPane)         │
                                   │  - Browser Hub (Central de Tarefas)      │
                                   └──────────────────────────────────────────┘
```

### 3.1. Agente Core (Python) ↔ Desktop Controller (Electron)
A comunicação não depende de pipes instáveis de terminal. Ela ocorre via **Loopback TCP Local com Bearer Auth**:

1. **Arquivo Descritor de Controle (`browser-control.json`):**
   Ao inicializar o controlador, o Electron abre um servidor HTTP em `127.0.0.1` numa porta dinâmica (passando `0` para alocação automática pelo sistema operacional) e gera um token aleatório seguro (`crypto.randomBytes(32)`).
   Ele grava o descritor atômico em:
   - **Windows:** `%LOCALAPPDATA%\HermesWorkstation\Runtime\browser-control.json`
   - **macOS:** `~/Library/Application Support/HermesWorkstation/Runtime/browser-control.json`
   - **Linux:** `~/.config/HermesWorkstation/Runtime/browser-control.json`

   Exemplo de payload do descritor:
   ```json
   {
     "version": 1,
     "pid": 12345,
     "url": "http://127.0.0.1:54321",
     "token": "4f8a92...c29b",
     "runtime": "electron-chromium",
     "profile_path": ".../HermesWorkstation/Browser/User Data",
     "created_at": "2026-09-14T12:00:00.000Z"
   }
   ```
   O descritor real também inclui `runtime: "electron-chromium"`,
   `profile_path` e `created_at`. O arquivo é escrito atomicamente com modo
   privado quando suportado e é removido no shutdown somente se ainda pertence
   ao mesmo token/processo. Ele nunca deve ser impresso, retornado ao modelo ou
   publicado pela rota LAN.
2. **Roteamento em `tools/browser_workstation.py`:**
   Toda ferramenta `browser_*` (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`, `browser_vision`, `browser_console`, etc.) é interceptada antes do backend legado:
   - Lê o descritor local e envia um POST JSON com cabeçalho `Authorization: Bearer <token>`.
   - **Fast Health Probe:** O status do controlador possui cache de disponibilidade de 0.75s (`_AVAILABILITY_CACHE_SECONDS`) com timeout de probe de 200ms (`_HEALTH_TIMEOUT_SECONDS`), garantindo que o agente não sofra atrasos se o Desktop estiver fechado.
   - **Contrato HTTP:** `GET /health` retorna prontidão; `GET /resources` retorna a projeção de recursos; `GET /events?task_id=...&limit=...` retorna eventos limitados; `POST /v1/action` executa uma ação `browser_*`. Todas as rotas exigem o Bearer exato e respostas são envelopes JSON `{success, ...}`.
   - **Payload de ação:** o envelope leva `action`, `arguments`, `task_id`, `session_id` e, quando disponível, `kanban_card_id`/`card_id` e `run_id`. Identidades são strings limitadas a 256 caracteres e sem caracteres de controle; ações não iniciadas por `browser_` são rejeitadas.
   - **Proteção de Carga:** O servidor loopback do Electron impõe `MAX_CONTROL_BODY_BYTES = 512 * 1024` para repelir requisições malformadas ou payload bombs. A leitura de eventos é limitada a 200 itens e o cliente Python também normaliza esse limite.
3. **Fail-Closed Rigoroso:**
   Assim que o agente executa a primeira ação no Workstation Browser em uma tarefa, essa tarefa é registrada em `_BOUND_TASKS`. Se o controlador Desktop for encerrado ou cair no meio da tarefa, a execução **falha de forma fechada (fail-closed)** com erro explicativo, em vez de realizar um fallback silencioso para outro navegador em nuvem ou sem cookies. Isso protege credenciais e evita ações em ambientes desautenticados.
4. **Redação no Boundary:**
   Toda resposta de ferramentas do Workstation passa obrigatoriamente por `redact_sensitive_text` antes de cruzar o limite para o modelo LLM. Segredos do descritor, portas, tokens e cookies brutos nunca chegam ao prompt do modelo.

### 3.1.1. Resolução por sessão e fallback

O health check não decide sozinho se a sessão deve conhecer a superfície. A
capacidade de schema é resolvida pela sessão do Gateway:

- `HERMES_SESSION_SOURCE=desktop` tem precedência sobre
  `HERMES_SESSION_PLATFORM`; somente essa superfície recebe os schemas
  `browser_*` quando `browser.workstation.enabled` (ou o default habilitado)
  permite o Workstation Browser;
- o probe de disponibilidade (`200 ms`, cache de `0,75 s`) é apenas uma decisão
  de execução/recovery, nunca um gate process-global que remove ferramentas do
  prompt;
- `HERMES_WORKSTATION_BROWSER_ROUTING` ou
  `browser.workstation.routing_enabled` controla fallback. O default é
  interno-only: se o controller não estiver disponível, a chamada falha
  fechada mesmo para tarefa ainda não vinculada;
- se routing estiver explicitamente habilitado, somente uma tarefa ainda não
  vinculada pode cair na lane legada. Depois de qualquer ação interna bem-
  sucedida, a chave `task_id`/`session_id` entra em `_BOUND_TASKS` e nunca é
  silenciosamente movida para outro browser.

Essa separação é necessária porque um processo Hermes pode atender várias
sessões e topologias. `HERMES_DESKTOP=1` identifica quem iniciou um backend,
mas não prova que uma GUI está conectada e, portanto, não pode ser usado para
decidir a presença do schema de browser.

### 3.2. Frontend UI (React) ↔ Processo Principal (Electron IPC)
A interface de usuário comunica-se com o runtime via ponte segura definida em `apps/desktop/electron/preload.ts`:
- **Canal de Estado Reativo:** `hermes:workstation-browser:state` (envia atualizações de abas ativas, URLs, títulos, status de carregamento, erros e proprietário do controle).
- **Ações IPC Expostas:**
  - `attach(bounds, host)`: acopla a view nativa nas coordenadas do painel.
  - `detach()`: desanexa a view da janela mantendo a aba viva.
  - `setVisible(visible)`: oculta/restaura o `WebContentsView` instantaneamente no compositor nativo.
  - `clearError()`: limpa o último erro registrado da tela.
  - `transferViewport(targetHost, bounds)`: transfere a exibição entre o painel lateral de chat (`chat`) e a central global (`hub`).
  - `takeControl()` / `releaseControl()`: gerencia a alternância de posse entre humano e IA.

O preload expõe somente métodos tipados da ponte; o renderer não recebe o
objeto `BrowserWindow`, `WebContentsView`, token do controller ou acesso Node
genérico. Os handlers usam `senderWindow(event)` e rejeitam invocações cujo
sender não seja um `BrowserWindow`. O canal de estado envia a projeção inteira
(`ready`, `attached`, `viewportHost`, `paused`, `controlOwner`, tabs, tasks,
downloads e `lastError`) para todas as janelas Desktop vivas.

### 3.3. Projeção comum de recursos e eventos

O runtime é o dono de página/`BrowserTask`; o `ExecutionJournal` é o dono da
história durável. `workstation-browser-resources.ts` apenas deriva uma
projeção UI-neutral versionada, sem criar um terceiro armazenamento:

- `schema_version: 1`, `runtime: "electron-chromium"` e
  `generated_at` formam o envelope;
- há um recurso `browser:electron-chromium`, um
  `browser-task:<taskId>` por tarefa e um
  `execution-journal:<taskId>` por diário;
- cada recurso preserva `task_id`, `session_id`, permissões e estado. A
  linhagem de tarefa inclui `kanban_card_id` e `run_id`; evidências apontam para
  `browser://controller`, `browser://tab/<id>` e
  `workstation://task/<id>` quando apropriado;
- `agent-control` só aparece quando o controller está pronto, o browser não
  está pausado e `controlOwner === 'agent'`; caso contrário, a permissão cai
  para `read`;
- `execution_status` é derivado de evidência real: `hold` em pausa,
  `waiting-for-human` para lease/controle humano, `stalled` sem controller ou
  tab viva, e `running` apenas quando há ambos;
- o recurso mostra no máximo os 200 eventos mais recentes. A inspeção profunda
  continua no diário da tarefa; a projeção não deve copiar a linha do tempo
  inteira.

O cliente Python em `workstation/client.py` é somente um adaptador de
transporte: valida schema/runtime, normaliza recursos/eventos e retorna um
envelope degradado (`available: false`, arrays vazios e erro limitado a 500
caracteres) quando o controller cai. Dashboard e TUI usam o mesmo cliente,
respectivamente em `/api/workstation/resources` e
`/api/workstation/events`, e nos métodos JSON-RPC `workstation.resources` e
`workstation.events`. Nenhum desses clientes cria tarefa, grava estado ou
substitui o controller.

### 3.4. Relação com Gateway, backend e SessionDB

O Desktop conversa com um backend `hermes serve` headless por WebSocket/JSON-RPC
para chat, sessões, streaming e comandos; esse processo não precisa servir a
SPA do Dashboard. `dashboard` e `serve` compartilham o servidor oficial, mas
são superfícies independentes. Para runtimes antigos, o launcher pode usar
`dashboard --no-open` somente como fallback de compatibilidade quando `serve`
não existe.

O Browser Controller local continua sendo a ponte específica para as ações
`browser_*` e para o Chromium que vive no processo Electron. O Gateway mantém
a identidade de sessão e expõe apenas projeções autenticadas/read-only para
clientes. SessionDB continua sendo a autoridade do histórico de chat e dos
IDs de sessão; Workstation só referencia essa identidade em `sessionHost` e
nos envelopes de ação/evento.

---

## 4. Surfaces e Hosts de Exibição

- **Chat Browser View (`WorkstationBrowserPane`):** View do browser atrelada a uma janela de conversa (contextual). Fica fixada à sessão onde foi gerada.
- **Browser Hub:** View do browser principal que concentra todas as tarefas do Workstation (global).
- **Single-Host Contract:** Tanto o Chat Browser View quanto o Browser Hub são apenas **hosts de exibição**. O objeto vivo (a aba real do Chromium) é único e muda de Host conforme o uso, garantindo que não existam abas fantasmas operando a mesma automação de forma assíncrona/duplicada.
- **Supressão Mútua:** Quando a tela cheia do Browser Hub é aberta, o painel lateral do chat é automaticamente suprimido para impedir instâncias concorrentes disputando limites e foco na mesma janela.

### 4.1. Geometria nativa e transferência de viewport

Chat e Hub normalmente compartilham a mesma `BrowserWindow`; o host não é
identificado pelo sender IPC sozinho. Por isso, o renderer envia um `host`
explícito em `attach`/`transferViewport` e um `expectedHost` em atualizações de
limite. O runtime:

- desanexa a view antiga antes de anexar a nova, mantendo uma única view viva;
- converte DIP do renderer para pixels nativos usando `webContents.zoomFactor`;
- valida números finitos, garante largura/altura mínimas de 1 px e limita
  `x/y/width/height` aos `contentBounds` da janela;
- ignora uma atualização de resize se `expectedHost` não corresponder ao
  `viewportHost` vigente. Isso evita que um pane desmontado ou em background
  mova a view para o próprio retângulo;
- reconcilia bounds em `resize`, `maximize` e `unmaximize`, e remove os
  listeners no teardown;
- deixa `WebContentsView` estacionado em background com frame rate reduzido
  (por default 6 FPS) e privilegia a view visível (até 60 FPS), sem fechar a
  página.

O teste integrado confirma a sequência de bounds stale → bounds aceito,
transferência Hub → Chat, maximize/restore e limpeza de hide/park/destroy.
Uma alteração de bounds jamais deve criar uma segunda aba, trocar o
`ownerTaskId` ou ser aceita apenas porque veio de um renderer válido.

---

## 5. Eficiência Radical de Tokens e Estratégia de Percepção

Agentes convencionais que automatizam navegadores costumam enviar capturas de tela contínuas em alta resolução para o LLM a cada clique ou rolagem. Isso consome entre **1.500 e 2.500 tokens de visão por ação**, levando a custos proibitivos e esgotamento rápido de rate-limits.

O Hermes Workstation emprega uma abordagem de **Percepção Semântica Estruturada**:

### 5.1. Árvore Semântica Compacta (`formatInventory`)
Ao solicitar um snapshot da página (`browser_snapshot` ou como resultado automático de `browser_click` / `browser_type`), o runtime injeta o script `inventoryScript` e formata os dados em texto estruturado:
```text
URL: https://exemplo.com/login
Title: Entrar no Sistema

Interactive elements:
- [c1] input "E-mail ou usuário"
- [c2] input "Senha"
- [c3] button "Entrar" disabled
- [c4] a "Esqueci minha senha"

Page text:
Bem-vindo ao sistema. Digite suas credenciais para continuar.
```
* **Orçamento Rígido de Tokens (Compact Budget):**
  - **Modo Compacto (padrão):** Limite de **8.000 caracteres de texto** (`COMPACT_TEXT_CHARS`) e máximo de **120 elementos interativos** (`COMPACT_ELEMENTS`).
  - Consumo médio por turno: Apenas **~1.000 a 1.800 tokens de texto**, uma fração mínima do custo de visão.
  - **Modo Completo (`full=True`):** 24.000 caracteres e 400 elementos, usado apenas se o modelo solicitar explicitamente inspecionar a página inteira.

### 5.2. Visão como Fallback Cirúrgico (`browser_vision`)
A ferramenta visual existe e funciona perfeitamente, mas é classificada como ferramenta de **exceção**. O agente só a utiliza quando o inventário semântico é insuficiente (ex: desafios visuais tipo CAPTCHA, mapas canvas, diagramas interativos ou layouts puramente pictóricos).

### 5.3. Preservação do Prompt Caching Sagrado
Conforme definido em `AGENTS.md`, o cache de prompt de cada conversa é sagrado:
- O Hermes nunca altera o schema de ferramentas no meio da conversa.
- Em sessões onde o Workstation Browser não está habilitado (como chats puros ou gateways de mensagens), as ferramentas `browser_*` sequer entram no schema inicial, economizando milhares de tokens fixos de declaração de API.

O snapshot compacto é produzido no processo Electron por `executeJavaScript`,
mas cliques e digitação usam `wc.debugger`/CDP (`Input.dispatchMouseEvent`,
`Input.dispatchKeyEvent` e `Input.insertText`) sobre a mesma `WebContents`. O
runtime devolve o snapshot após um pequeno atraso de estabilização da página
(aproximadamente 220 ms para click/back, 160 ms para type, 140 ms para scroll e
120 ms para key press), não uma captura visual automática. `browser_console`
executa expressão somente quando explicitamente solicitado; a captura visual
grava PNG local em `Browser/Screenshots` e continua sendo uma exceção.

---

## 6. Controle Híbrido (Human vs Agent)

O Workstation Browser suporta alternância de controle em tempo real através do mecanismo "Take Control" / "Release Control":
1. **Agente Atuando (`controlOwner: 'agent'`):** O modelo LLM envia comandos de automação via protocolo loopback CDP.
2. **Usuário Assume o Controle (`takeControl`):** O usuário clica no botão "Take Control" na interface do Desktop. O `controlOwner` muda para `'human'`. Enquanto o humano estiver no controle, comandos automatizados de entrada do agente são temporariamente bloqueados para evitar colisões de digitação ou cliques erráticos.
3. **Usuário Devolve o Controle (`releaseControl`):** O humano conclui o login, resolve um CAPTCHA ou confere a compra e clica em "Release Control". O agente recebe sinal verde e retoma sua execução exatamente de onde parou, preservando o estado do DOM e todos os cookies da sessão na **MESMA** `BrowserTask`.

O gate de controle é aplicado no controller, não apenas visualmente no
renderer: ações mutáveis (`navigate`, `click`, `type`, `scroll`, `back` e
`press`) são rejeitadas enquanto `paused` ou enquanto o dono é `human`.
Leituras como snapshot, imagens e eventos continuam sendo uma projeção de
observação quando disponíveis, mas a permissão do recurso cai para `read` sem
evidência/controle. Pausar é uma barreira de execução e não logout, limpeza de
cookies ou destruição de tab.

---

## 7. Decisões de UI, Overlays e Descobertas Críticas

Durante a construção e validação da interface Desktop, importantes desafios de integração entre o Chromium nativo e o React foram identificados e solucionados:

### 7.1. Oclusão do `WebContentsView` sobre Menus HTML/React
* **O Problema:** O `WebContentsView` é uma superfície nativa do sistema operacional (janela HWND no Windows). Menus de contexto HTML gerados pelo React/Radix UI (`[data-radix-menu-content]`) são renderizados no DOM da janela principal (`document.body`). Pelo modelo do Electron, **qualquer superfície nativa do Chromium se sobrepõe permanentemente a nós DOM HTML**, independentemente de `z-index: 999999` ou de camadas CSS. Quando o usuário clicava com o botão direito no ícone do browser, o menu abria por baixo do browser, ficando invisível e inacessível.
* **A Solução Arquitetural:**
  1. Criação do método `setVisible(visible: boolean)` no runtime do Electron, que desanexa temporariamente o `WebContentsView` do compositor da janela (`window.contentView.removeChildView`) sem destruir a página, sem perder o estado JS e sem recarregar.
  2. Implementação de um `MutationObserver` no `WorkstationBrowserPane` e no `BrowserHub` que monitora `document.body` em busca de portais de overlay do Radix (`[data-radix-menu-content]`, `[role="menu"]`, `[data-radix-popper-content-wrapper]`, dropdowns e diálogos).
  3. No milissegundo em que o menu do botão direito se abre, o browser cede visibilidade; assim que o menu é fechado ou uma opção é clicada, o browser é restaurado instantaneamente sem flicker.

### 7.2. Ciclo de Vida de Erros Transitórios de DOM (`stale_or_unknown_ref`)
* **O Problema:** Em Single Page Applications (SPAs) modernas como o Instagram ou X (Twitter), nós do DOM são constantemente virtualizados e remontados pelo React durante rolagens ou atualizações. Quando o agente tentava clicar em um identificador `ref` cujo nó havia acabado de ser desanexado (`!el.isConnected`), o Chromium retornava `{ error: 'stale_or_unknown_ref' }`. O runtime gravava esse erro em `this.lastError`, mas como não havia rotina de expiração, a barra vermelha de aviso ficava presa na tela indefinidamente, mesmo após o agente continuar seu trabalho com sucesso.
* **A Solução:**
  1. Erros transitórios de elementos (`stale_or_unknown_ref`, `element_not_visible`, `element_unavailable`, `ref_required`) agora possuem um **timeout automático de expiração de 7 segundos**.
  2. O runtime limpa `lastError` imediatamente após qualquer requisição subsequente bem-sucedida ou ao navegar para uma nova URL.
  3. Adicionado botão de dispensar manual (`✕`) na interface chamando `bridge.clearError()`.
  4. Mensagem humanizada e amigável formatada para o usuário final.

### 7.3. Isolamento de Abas por Sessão de Chat
* **Fixação por Sessão:** O painel lateral do browser pertence à sessão onde foi invocado. Quando o usuário cria ou troca de sessão de chat, o browser anterior não "acompanha" para a nova conversa como uma assombração visual; a view se desanexa até que seja explicitamente acionada no novo chat.
* **Nomeação Amigável de Tarefas:** Tarefas criadas no Browser Hub recebem nomes baseados no contexto da sessão de chat associada e no título da aba, evitando UUIDs técnicos impenetráveis para o usuário.

### 7.4. Inicialização preguiçosa e erro transitório

O controller HTTP é iniciado no `app.whenReady()` para que o agente possa
encontrar a superfície desde o boot, mas a criação de uma `WebContentsView`
desanexada é adiada até o browser ou uma ação `browser_*` realmente precisar
dela. Isso evita deixar um target Electron sem janela proprietária e reduz o
custo de inicialização. A sessão persistente e a restauração estrutural podem
ocorrer antes da primeira view visível.

Erros de `stale_or_unknown_ref`, `element_not_visible`,
`element_unavailable` e `ref_required` são transitórios: entram em `lastError`,
expiram após 7 segundos se não forem substituídos, e uma requisição bem-
sucedida limpa o erro imediatamente. O renderer também pode chamar
`clearError`. Não transforme esse alerta em estado de falha permanente e não
faça retry cego de um `ref`: gere novo snapshot e use uma referência atual.

---

## 8. Isolamento de Perfis, User-Agent e Persistência de Dados

### 8.1. Onde os Dados Residem
Para garantir segurança e facilidade de manutenção, o Hermes separa rigorosamente o **código-fonte** dos **dados do usuário**:

| Tipo de Dado | Local de Armazenamento | Descrição |
|---|---|---|
| **Perfil Chromium (Cookies / Logins)** | `%LOCALAPPDATA%\HermesWorkstation\Browser\User Data` | Dados do navegador (WhatsApp Web, Instagram, Google, abas salvas). **Nunca** é apagado ao atualizar código. |
| **Metadados estruturais de sessão/tarefas** | `%LOCALAPPDATA%\HermesWorkstation\Runtime\browser-session.json` | Snapshot composto versionado de abas, ordem, aba ativa, URLs sanitizadas e `BrowserTask`; `browser-tasks.json` é somente origem de migração legada. |
| **Memória do Agente / Histórico Chat** | `~/.hermes` (`C:\Users\<User>\.hermes`) | Banco de dados SQLite (`state.db`), `config.yaml`, `.env`, memórias de longo prazo. |
| **Preferências do Electron** | `%APPDATA%\hermes` | Configurações de tema, janelas e layout do Desktop. |

### 8.2. User-Agent Neutro e Compatibilidade Web
O Workstation Browser implementa a função `getStandardChromeUserAgent()`:
```text
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36
```
Ele remove qualquer menção a strings como `Electron` ou `Hermes`. Isso previne bloqueios automatizados por serviços web sensíveis (como o aviso *"Atualize seu navegador Chrome"* ao tentar abrir o WhatsApp Web).

### 8.3. Resolução Flexível de Backend (`resolveHermesBackend`)
O aplicativo Desktop localiza o backend Python através de uma hierarquia clara:
1. **`HERMES_DESKTOP_HERMES_ROOT` (Prioridade Máxima):** Se o desenvolvedor definir essa variável de ambiente apontando para o seu repositório local (ex: `C:\Github\hermes-agent`), o executável Desktop executa diretamente esse código-fonte, aproveitando imediatamente todas as modificações e novidades locais sem precisar recompilar o instalador.
2. **`SOURCE_REPO_ROOT`:** Quando executado via `npm run dev` diretamente de um checkout git.
3. **`ACTIVE_HERMES_ROOT`:** Runtime canônico em `%LOCALAPPDATA%\hermes\hermes-agent`.
4. **Bootstrap Installer:** Disparado apenas se nenhum runtime funcional for encontrado.

### 8.4. Cache, extensões e downloads

O perfil é aberto com `session.fromPath(profilePath, { cache: true })`, com
`contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` e
`backgroundThrottling: false` nas views. O User-Agent Chrome neutro é aplicado
na sessão, em `onBeforeSendHeaders` e novamente em cada `WebContentsView`.

`cleanupCache` pode ser executado por limite ou à força, mas limpa somente
cache: cookies, LocalStorage, IndexedDB e estado de login permanecem. A
manutenção roda em timer de 30 minutos e em uma verificação inicial atrasada;
falha de métrica/cache não pode derrubar o browser. Extensões são carregadas
pela `ChromeExtensionManager` na sessão dedicada, após download/extração
qualificados. Downloads são eventos de estado em memória, limitados aos 20
mais recentes, e não devem ser confundidos com BrowserSessionState.

---

## 9. Fluxo de Decisão de Footprint (The Footprint Ladder)

O Hermes Workstation respeita rigorosamente o **Footprint Ladder** (`AGENTS.md`). Não adicionamos ferramentas Core a torto e a direito para resolver problemas do Workstation, pois cada ferramenta adicionada é cobrada em tokens em **todas** as chamadas de API:

1. **Estender código existente:** Zero novo surface.
2. **CLI command + skill:** Gerencia configurações expressíveis via terminal (`hermes cron`, `hermes tools`). Zero footprint no schema do modelo.
3. **Service-gated tool (`check_fn`):** Só aparece quando o serviço pré-requisito está configurado.
4. **Plugin:** Para capacidades de terceiros ou nichos específicos.
5. **Servidor MCP (no catálogo):** Se a capacidade precisa de I/O estruturado mas não é fundamental ao core.
6. **Nova Core Tool (Último recurso):** Somente quando a capacidade for fundamental, útil para quase todos os usuários e inalcançável via terminal + arquivo.

---

## 10. Invariantes Críticos e Edge-Cases (Downstream vs Upstream)

Ao atuar em camadas de testes ou SO hospedeiro, considere armadilhas inerentes à ponte POSIX para Windows:
- **File Modes e Ownership (`438 !== 384/493`):** Testes de `hardening` baseados em File Modes estritos (`0o700`) tendem a gerar divergências lógicas no Windows devido ao NTFS não suportar nativamente permissões octais do Unix da mesma forma, exigindo abstrações corretas ao invés de expectações hardcoded. O mesmo vale para formatações nativas de `Intl` que trocam o ponto flutuante, podendo falhar testes de formatação gráfica estrita de UI.
- **Symlinks e EPERM em Worktrees:** Construções de socket (`ssh-connection`) e diretórios de cache ou paths temporários na LocalAppData geram falhas de `EPERM` se manipulados indevidamente no Windows (ex: bloqueios ao rodar git-worktrees simultâneos em caminhos de rede/UNC ou namespaces de WSL `\\?\`).
- **PowerShell Hand-off e Timings:** O ciclo de atualizações da workstation e transferências de arquivos com PowerShell retém os metadados corretos de `acquisition time`, porém é propenso a timeouts lentos nos runners CI de integração (gerando Timeouts fixos de 5000ms e 15000ms no Electron), que não devem ser tratados trivialmente com `continue-on-error`.

### 10.1. Matriz de escopo de estado

Antes de editar qualquer integração, classifique o estado no escopo correto:

| Escopo | Dono/Exemplo | Regra |
|---|---|---|
| Processo Electron | singleton `WorkstationBrowserRuntime`, listeners IPC, token do controller | não usar como identidade de sessão; encerrar limpa o processo, não o perfil |
| Perfil Browser | `session.fromPath`, cookies, LocalStorage, IndexedDB, cache, extensões | fica fora do repositório; não compartilhar com Chrome/Edge pessoal |
| Sessão Hermes/Gateway | `session_id`, `HERMES_SESSION_SOURCE`, `HERMES_SESSION_PLATFORM` | decide surface/schema e linhagem; não é substituída pelo BrowserTask |
| BrowserTask | `taskId`, `ownerTaskId`, `sessionHost`, `kanbanCardId`, `runId`, lease/estado | uma identidade lógica e no máximo uma página viva |
| Host/renderer | `viewportHost`, bounds, `attached`, pane Chat ou Hub | view de apresentação; nunca fonte de verdade da task |
| Journal/Kanban | SessionDB, card/run canônicos, `ExecutionJournal` JSONL | histórico e lineage persistentes; não duplicar no runtime |

Dois testes de segurança são especialmente importantes: uma sessão Desktop
deve receber a superfície mesmo se o probe de controller estiver indisponível,
e uma sessão não-Desktop deve permanecer sem o schema; e uma tarefa já ligada
deve falhar fechada quando o descritor/controller desaparecer, mesmo que a
lane legada esteja funcional.

### 10.2. Limites que fazem parte do contrato

Os seguintes números são limites de segurança/recursos, não metas que testes
devam congelar como catálogos mutáveis: controller body 512 KiB; identidades
de controller 256 caracteres; URL restaurável 2.048 caracteres; 128 abas no
snapshot; eventos públicos 200; inventário compacto 8.000 caracteres/120
elementos e inventário completo 24.000 caracteres/400 elementos. Alterar um
limite exige revisar o boundary, o cliente normalizador e a evidência E2E em
conjunto.

### 10.3. Persistência não é identidade viva

O `browser-session.json` pode carregar uma intenção lógica recente mesmo se o
disco ainda contiver o snapshot anterior após uma falha de rename. Durante
shutdown, o runtime persiste antes de fechar `WebContents`; eventos tardios de
destruição não devem apagar a recuperação. Já `WebContents`, debugger CDP,
heap JavaScript e handles de janela são sempre process-local. Qualquer texto
que diga que uma página foi "preservada" através de restart deve significar
metadados + perfil Chromium recuperáveis, nunca o mesmo objeto de memória.

---

## 11. Mecânica de Testes e Anti-Patterns de Validação

A Workspace provê suítes automatizadas completas para garantir não-regressão:

### 11.1. Suíte Electron / Desktop (Vitest)
Executada em `apps/desktop/`:
```powershell
npx vitest run electron/workstation-browser src/store/preview.test.ts
```
Cobre:
- Ciclo de vida e isolamento de `BrowserTask` (`workstation-browser-task.test.ts`);
- Durabilidade e tolerância a falhas do `SessionState` (`workstation-browser-session-state-resilience.test.ts`);
- Restauração de viewport e acoplamento de janelas (`workstation-browser-runtime-viewport.test.ts`);
- Recuperação de crash de abas e resiliência de processo (`workstation-browser-runtime-recovery.test.ts`);
- Comportamento de `setVisible` e auto-limpeza de `clearError` (`workstation-browser-runtime-task.test.ts`).

### 11.2. Suíte de Rota e Contratos Python (Pytest)
Executada na raiz do repositório:
```powershell
$env:PYTHONPATH="."; python -m pytest workstation/tests
```
Cobre atualmente 175 testes de rotas, contratos de host, pipelines de eventos,
drift, políticas, memória, workers, isolamento, release qualification e
evidência de carga. O comando CI-parity do repositório continua sendo
`scripts/run_tests.sh workstation/tests/`; a contagem deve ser reportada como
evidência da execução, nunca como um assert de número fixo.

### 11.3. Regras Estritas de Validação
- **Causalidade de Regressão Rigorosa (Não Conte Vermelhos):** O Windows downstream já possui problemas endêmicos e falhas de runtime fixadas num conhecido **KI-006** (EPERMs, timeouts no WSL Bridge e file masks). Comparar a qualidade de uma PR pela mera contagem de testes quebrados é um antipattern grave. Qualquer validação obriga executar uma matriz de **Test Identity (1:1)** exata rodando os arquivos do Baseline lado-a-lado com o Candidato, ignorando diferenças voláteis geradas pelo harness, como strings literais de temporários randômicos (ex: `ssh-test-XYZ`).
- **NUNCA enfraqueça testes (No Skips):** Não tente pular testes falhos, "corrigir" temporariamente asserts de POSIX ou utilizar `.env` mutáveis apenas para apagar ruídos. Comporte-se restritamente sob as regras de baseline.
- **Tipagem e Linter Estritos:** `npm run typecheck --workspace apps/desktop` deve retornar 0 erros. O ESLint deve ser mantido limpo com zero warnings e conformidade à ordenação de imports (`perfectionist/sort-imports`).

---

## 12. Anti-Patterns — O que NUNCA fazer no Hermes Workstation

- ❌ **NÃO mutar o histórico de mensagens nem injetar mensagens sintéticas de usuário:** Isso quebra o cache de prompt (`prompt caching`) no provedor de IA e multiplica em até 10x o custo por mensagem.
- ❌ **NÃO adicionar ferramentas nativas de browser ao `_HERMES_CORE_TOOLS`:** Ferramentas enviadas no core oneram todas as chamadas de API em todas as plataformas (CLI, WhatsApp, Telegram). Ferramentas de browser devem permanecer vinculadas à sessão ativa (`session-scoped`).
- ❌ **NÃO tirar capturas de tela (screenshots) a cada clique:** Utilize primariamente o inventário semântico estruturado (`formatInventory`). Reserve visão (`browser_vision`) exclusivamente para tarefas onde o texto e botões da página forem insuficientes.
- ❌ **NÃO destruir instâncias de WebContents ao ocultar ou parkear abas:** Ocultar (`hide`/`park`) deve apenas desanexar a view da janela (`removeChildView`), preservando a aba viva em memória.
- ❌ **NÃO salvar segredos, tokens ou dados brutos de formulários em arquivos JSON de estado:** Arquivos como `browser-session.json` e `browser-tasks.json` devem conter apenas identificadores e URLs sanitizadas.
- ❌ **NÃO usar variáveis de ambiente de processo para determinar capacidade de interface:** Uma variável como `HERMES_DESKTOP=1` indica apenas quem disparou o processo; a disponibilidade de ferramentas GUI deve ser consultada a partir da sessão ativa que se comunica com o Gateway.
- ❌ **NÃO deixar erros operacionais transitórios (ex: `stale_or_unknown_ref`) bloqueados permanentemente na UI:** Erros de elementos em SPAs devem ter auto-expiração e limpeza reativa em ações subsequentes.
- ❌ **NÃO reintroduzir normalização heurística de títulos como garantia de segurança:** títulos controlados pela página são sempre `null` no estado durável; qualquer título visto na UI é uma leitura viva de `WebContents`.
- ❌ **NÃO afirmar que snapshots compostos intermediários são impossíveis:** uma operação pode substituir primeiro `browserTasks` e depois a projeção de abas (ou vice-versa). Cada combinação observável precisa ser normalizada, segura e recuperável.
- ❌ **NÃO transformar `BrowserSessionState` em um segundo banco:** a persistência composta é projeção estrutural; SessionDB, Kanban, ExecutionJournal e o perfil Chromium continuam com seus próprios donos.

---

## 13. Gaps reproduzidos e contratos confirmados

Esta seção registra os fatos reproduzidos durante a validação da Implementation 4
que não devem ser perdidos em uma refatoração. Os números de commit/workflow são
evidência histórica do candidato da PR #9; repita a matriz baseline/candidato
antes de projetá-los sobre outro SHA.

No código atual, `BrowserTaskLifecycle` possui um único mapa por `taskId` e
`createTask` é idempotente. O registro V1 contém `taskId`, `createdAt`,
`updatedAt`, `panelHost`, `controlHost`, `sessionHost`, `kanbanCardId`, `runId`,
`localConnection`, `status`, `leaseState`, `parked` e `recoveryState`
(`fresh | restored | recreated | null`). O enum persistido de status é
`visible | hidden | parked`; a página viva é mantida separadamente pelo
`taskTabs` do runtime, sempre com no máximo uma relação `taskId -> tabId`.

### 13.1. Três lacunas de BrowserTask reproduzidas

1. **Snapshot que parkava Take/Release Control:** `entryForTask()` estacionava
   incondicionalmente depois de operações de leitura/controle. Uma tarefa visível
   e anexada perdia a view. A causa era ignorar `active && attached`; parqueie
   somente entrada não visível/anexada e mantenha `visible`.
2. **Crash reutilizado como página viva:** `render-process-gone` marcava `crashed`,
   mas a verificação consultava apenas `isDestroyed()`. A mesma `WebContents`
   inválida era reutilizada. `rawEntryForTask` deve descartar `crashed ||
   isDestroyed()`, recriar lazy e manter uma única relação `taskId -> page`.
3. **Active tab perdido ao destruir:** `closeTab()` fechava primeiro e só então
   decidia se era ativa; o evento síncrono `destroyed` limpava `activeTabId`.
   Capturar `wasActive`, centralizar `discardEntry` e ativar o sobrevivente (ou
   `about:blank`) torna a operação determinística.

Os testes dessas regressões cobrem também isolamento de duas tarefas, limpeza
explícita de uma página crashada e a asserção negativa de que tokens/senhas de
URL ou formulário não aparecem no estado persistido. A suíte focada chegou a
16/16 e os typechecks Electron passaram no candidato validado.

### 13.2. Regras de leitura sem ambiguidade

- O enum de código continua `visible | hidden | parked`; *Waiting for Human*,
  *Background* e *Recent* são agrupamentos derivados da UI.
- `browser-session.json` é o snapshot composto atual; `browser-tasks.json` apenas
  migra estado legado e não é um segundo page store.
- O descriptor `browser-control.json` não contém `session_id`. Seus campos são
  `version`, `pid`, `url`, `token`, `runtime`, `profile_path` e `created_at`.
  `session_id` pertence ao envelope de ação/evento.
- A porta do controller vem de `listen(0, '127.0.0.1')`; clientes leem o descriptor
  e usam `Authorization: Bearer <token>`, sem presumir porta fixa ou imprimir token.
- `safeTitleMetadata` retorna `null` por design. Título visível é leitura efêmera
  de `WebContents`, não identidade durável.
- Restart conserva IDs, ordem, active tab, políticas e URLs seguras e reabre o
  perfil persistente; não conserva `WebContentsView`, debugger CDP ou heap.
- `ownerTaskId` é a barreira entre tarefas. Controller, IPC, resources, Journal e
  Kanban carregam a mesma identidade; conflito de vínculo falha fechado.

### 13.3. Outcomes históricos de CI

Na matriz 1:1 do candidato da PR #9, BrowserTask focused passou 15/15 (16/16
após o teste de segredo) e o Workstation Pytest passou. O vermelho restante foi
classificado, não contado como regressão de core:

| Step | Outcome real | Classificação |
| --- | --- | --- |
| Desktop UI | mock sem `BROWSER_ROUTE` | baseline/harness |
| Desktop platform | sete arquivos conhecidos (EPERM, WSL/POSIX, timeouts) | baseline Windows |
| Docker | classificador; build/test pulados | `NON-EXECUTABLE` |
| Nix/CI sem runner | queued/cancelled, `steps=[]` | `NON-EXECUTABLE` |
| Desktop E2E | guardado por `false &&` | `NON-EXECUTABLE` |

Não relaxar asserts, adicionar skips ou promover um workflow que não executou o
teste. Para qualquer candidato novo, repetir os mesmos arquivos contra baseline
do mesmo SHA, registrar o primeiro erro real e classificar `PASS`, `FAIL
regression`, `FAIL baseline` ou `NON-EXECUTABLE`.

### 13.4. Fronteira de manutenção

Não criar segundo `BrowserRuntime`, page store, SessionDB, Kanban DB ou Journal;
não persistir secrets; não decidir a superfície GUI pelo ambiente global do
processo; não alterar ownership de OpenCode/Antigravity; e não declarar a
Implementation 4 completa apenas por testes focados verdes. A aprovação final
continua pertencendo ao Conductor, depois de evidência nativa/clean-machine.

---

## 14. Delta factual verificado em código (sessão de 2026-09-14)

Esta seção fecha os detalhes de baixo nível observados após a consolidação das
seções anteriores. Ela não cria uma nova arquitetura; serve como um mapa de
leitura rápida para quem precisa rastrear uma chamada desde o tool até o
Chromium e de volta ao cliente.

### 14.1. Boot, descriptor e endpoints reais

O boot é intencionalmente assimétrico: `app.whenReady()` abre a sessão dedicada,
restaura o snapshot e inicia o servidor loopback; a primeira
`WebContentsView` só é criada quando Hub, Chat ou uma ação `browser_*` a exige.
O servidor escuta em `127.0.0.1` com porta `0`, portanto nunca há porta fixa
para descobrir ou publicar.

O descriptor escrito em `Runtime/browser-control.json` tem apenas:

```text
version, pid, url, token, runtime, profile_path, created_at
```

`token` é `crypto.randomBytes(32).toString('base64url')`. A escrita usa arquivo
temporário privado + rename; remoção no shutdown compara o token atual, para
que um processo antigo não apague o descriptor de um processo novo. O Python
valida versão, prefixo literal `http://127.0.0.1:` e token antes de abrir
qualquer socket. Não há `session_id` no descriptor.

O contrato HTTP atual é:

```text
GET  /health                       -> state completo do runtime
GET  /resources                    -> resource snapshot schema v1
GET  /events?task_id=&limit=       -> event snapshot schema v1, limit <= 200
POST /v1/action                    -> {action, arguments, task_id, ...}
```

Todos exigem Bearer exato. Corpo acima de 512 KiB é rejeitado enquanto é lido;
respostas são `application/json`, `no-store` e não incluem token/descriptor.
`/v1/action` só aceita nomes iniciados por `browser_`; erros de parsing,
política, controle humano ou identidade retornam `success: false`/HTTP 400.

### 14.2. Do tool ao `WebContentsView`

`tools.browser_workstation` calcula a chave de binding como
`task_id || session_id || "default"`, faz o health probe com cache curto e
monta o payload. Card/run podem vir de argumentos explícitos ou das variáveis
de contexto do turno (`HERMES_KANBAN_TASK`, `HERMES_KANBAN_RUN_ID`). O retorno
é redigido recursivamente antes de ser entregue ao modelo.

No controller, identidades são normalizadas e vinculadas antes da mutação. Em
`browser_navigate`, `entryForTask()` garante/reusa a única entrada da task,
normaliza o alvo e carrega a URL. Para ações de leitura/interação sem tab
previamente associada, a resolução tenta, nesta ordem: hint lazy restaurado,
uma tab ativa não proprietária (que passa a ser owned), ou criação de uma nova
tab para a task. Se nada puder ser ligado, retorna
`no_bound_browser_tab` em vez de operar uma página aleatória.

Navegação, click, type, scroll, back e press exigem `assertAgentControl()`;
snapshot, imagens, console e vision são leituras. Click/type/scroll/back/press
aguardam, respectivamente, aproximadamente 220/160/140/220/120 ms antes do
snapshot de retorno. A entrada é manipulada por CDP (`Input.dispatch*`/
`Input.insertText`) e não por JS sintético de alto nível.

`setWindowOpenHandler` nunca abre uma segunda janela: popup top-level seguro de
uma task é redirecionado para a própria entrada (idempotência de `ownerTaskId`)
e popup inseguro é negado. `will-navigate` e `will-redirect` repetem a guarda
para navegações top-level. Eventos `render-process-gone` marcam
`crashed/stale/page-gone`; a próxima resolução descarta o objeto inválido e
cria exatamente uma replacement, preservando a relação lógica.

### 14.3. Navegação e metadata segura

`normalizeWorkstationBrowserTarget` usa as seguintes heurísticas, antes da
policy de website: `about:blank` permanece; `http(s)` é normalizado; hosts
locais (`localhost`, `127.0.0.1`, `[::1]`) recebem `http`; host-like sem
espaços recebe `https`; texto restante vira query DuckDuckGo. A policy rejeita
IMDS/metadata (incluindo variantes IPv4-mapped/IPv6) e o roteador rejeita
prefixos de segredo antes de construir a busca.

O runtime mantém dois valores de URL: a URL viva de `WebContents` para a
interface e `safeUrl` para restart. O segundo é reavaliado em cada
`did-navigate`, `did-navigate-in-page`, `page-title-updated` e crash. Query e
fragmento nunca sobrevivem; userinfo, `\\`, controles, encoding ambíguo,
atribuições `token/password`, JWTs, tokens opacos e rotas de login/OTP com
credenciais falham fechados. Mesmo uma URL assinada só pode deixar sobreviver a
parte estrutural sem o material secreto.

O título é uma armadilha recorrente: `wc.getTitle()` aparece no estado de UI
para o usuário, mas `safeTitleMetadata()` retorna sempre `null`. Isto vale para
“título inocente” e título com token; não existe allow-list parcial que permita
inferir que texto controlado por uma página é seguro.

### 14.4. Reconciliação, ordem e atomicidade

O snapshot composto é uma projeção de `entries`, `taskTabs`, `activeTabId` e
`BrowserTaskLifecycle`. No caminho de lifecycle, uma falha pode deixar no disco
**nova metadata de task + projeção de tabs anterior**; gravações independentes
de abas também podem conservar metadata de task anterior. Ambas as formas são
tratadas como snapshots estruturais, normalizadas na leitura, e os testes
C1/C2/C3 reiniciam a partir da combinação exata em vez de assumir uma transação
inexistente.

A ordem persistida é restaurada em duas fases: primeiro materializar todas as
tabs possíveis (ordinárias agora, tasks como pending/lazy), depois reconciliar
`restoredTabOrder`. Limpar a fila durante a primeira fase reordena uma sequência
como `ordinary-A, ordinary-B, task-T` e quebra `activeTabId` lógico. Um task
ativo logicamente pode permanecer pending até `show/navigate`; o fallback
`about:blank` físico não deve roubar essa seleção.

No shutdown, `persistBrowserSessionState()` roda antes de `WebContents.close()`
e eventos `destroyed` são suprimidos. Em operação normal, `destroyed` lembra a
tab como stale para a próxima reconciliação; `destroyTask` remove explicitamente
a metadata. Assim, crash e destroy têm semânticas distintas e observáveis.

### 14.5. Preview/Chat/Hub não são page stores

`$previewTabs` tem persistência própria apenas para o rail React: arquivos,
URLs e artifacts exibíveis. A superfície URL do Workstation usa o id singleton
`url:browser`; abrir uma segunda URL troca o alvo do mesmo browser e não cria
uma segunda instância Chromium. `$sessionPreviewTabs` apenas pinna a seleção do
rail por conversa e promove a chave runtime para a chave persistida quando o
primeiro turno é salvo.

`WorkstationBrowserPane` chama `attach(bounds, "chat", taskId)`, atualiza
geometria com `ResizeObserver` + `expectedHost="chat"` e chama apenas `detach()`
no unmount. O Hub faz o mesmo com host `hub`; ambos transferem a única view.
`use-preview-routing` abre a superfície somente para uma sessão visível e não
fecha o preview de uma sessão em background. `MutationObserver` remove/recoloca
a view enquanto menus/dialogs Radix ocupam o DOM, pois `z-index` não atravessa
uma superfície nativa.

### 14.6. Resources, eventos e clientes não mutantes

O resource snapshot é calculado a cada consulta a partir do runtime e do
`ExecutionJournal`; não é salvo pelo controller. O cliente Python aceita apenas
`schema_version=1`, `runtime="electron-chromium"`, recursos/eventos com tipos
válidos e retorna `available:false` em degradação. O Dashboard
(`/api/workstation/resources`, `/api/workstation/events`) e o TUI
(`workstation.resources`, `workstation.events`) atravessam esse mesmo cliente e
nunca leem o descriptor diretamente.

`execution_journal:<taskId>` expõe somente os últimos 200 eventos na projeção;
o diário JSONL completo continua sendo a fonte de histórico. `running` é uma
afirmação conquistada por controller + tab viva; sem evidência o recurso é
`stalled`, mesmo que uma task lógica ainda exista.

### 14.7. Evidência e limites desta atualização

Os documentos de contexto/journal registram H010 (restart/metadata), H011
(soak nativo), H012 (backend headless/reconnect) e H013 (janela oculta,
geometria e parity de resources/events), além de 175/175 contratos Workstation
Python, UI/platform/typecheck e gates de packaging. Esses números são o estado
qualificado do snapshot documentado; para uma alteração nova, repetir a matriz
com SHA, runner e ambiente, e classificar `PASS`, `FAIL regression`, `FAIL
baseline` ou `NON-EXECUTABLE`.

O que continua explicitamente fora deste núcleo: corrigir KI-007/“Session not
found”, alterar SessionDB/Gateway sem reprodução, Preview unification, Browser
Memory, LAN/Tailscale, novo Kanban/control plane, ou qualquer fallback que
quebre uma task já bound. A regra de manutenção segue sendo: uma identidade
lógica, uma página viva por task, um runtime, um snapshot composto, um Journal e
owners já existentes.

---

## 15. Provas adversariais e crash consistency do snapshot composto

### 15.1. Títulos e URLs: garantia que o código realmente prova

O boundary de restart não tenta classificar texto livre. `safeTitleMetadata()`
retorna sempre `null`, inclusive para `Example Domain — Dashboard` ou
`Customer 482913`; títulos vivos continuam sendo fornecidos por
`WebContents.getTitle()` para a UI enquanto o processo existe. Assim, conteúdo
arbitrário de `<title>` nunca pode aparecer em `browser-session.json`.

`safeRestorableUrlMetadata()` preserva somente estrutura suficiente para uma
recuperação segura. A inspeção faz até oito camadas de percent-decoding e falha
fechada para encoding malformado/instável; barras invertidas são rejeitadas
antes do parser. Query e fragment são removidos, e pathname/resultado final
passam por marcadores de atribuição, rotas de autenticação, JWT e token opaco.
As regressões mínimas são:

| Entrada adversarial | Resultado durável exigido |
|---|---|
| query `access_token` | somente origem/path; valor ausente |
| OAuth `code` em query | somente origem/path; código ausente |
| token em fragment | somente origem/path; token ausente |
| URL com `username:password@host` | rejeitada (`null`) |
| JWT em pathname | rejeitada (`null`) |
| URL assinada/presigned | credenciais removidas; somente estrutura segura, ou `null` |
| `/recovery/code/482913` | rejeitada |
| `/verification/code/482913` | rejeitada |
| `/otp/482913` | rejeitada |
| `/temporary/pin/482913` | rejeitada |
| `/magic/login/code/482913` | rejeitada |
| `/customers/482913` | permitida como identificador estrutural |

Os cinco títulos de credencial (`Recovery code 482913`, `Verification code
482913`, `OTP 482913`, `Temporary PIN 482913`, `Magic login code 482913`) são
verificados no retorno, no JSON escrito, no snapshot recarregado e no JSON
re-serializado. Essa prova é deliberadamente diferente de “o sanitizer parece
seguro”: ela verifica que o material proibido não reaparece depois do restart.

### 15.2. Intermediários de persistência são possíveis e seguros

Cada replacement do arquivo é atômico, mas uma operação lógica pode executar
mais de um replacement. Portanto, o estado abaixo é **possível e aceito**:

```text
browserTasks novos + projeção anterior de tabs/safeUrl/safeTitle
```

Ele não é descrito como impossível. A normalização elimina referências a task
inexistente, deduplica IDs, rebaixa tarefas restauradas para `parked`/lazy e
reaplica a política de metadata antes de qualquer view ser criada. O próximo
save bem-sucedido grava um único snapshot composto canônico.

### 15.3. C1 — `createTask` interrompido entre projeções

O seam de fault-injection falha um `renameSync` escolhido após a metadata da
task ter sido durabilizada. Um runtime novo deve observar `T` no máximo uma vez,
com `status=parked` e `recoveryState=restored`; nenhuma página de `T` pode ser
ressuscitada durante `ensure()`. `showTask(T)` deve então criar uma única
`WebContentsView`, manter um único `taskTabs`/`ownerTaskId` e produzir o
snapshot canônico sem título ou URL insegura.

### 15.4. C2 — recriação/show interrompidos

Partindo de task lazy com URL sanitizada, a falha depois de
`visible/recreated` mas antes da projeção final deixa a projeção anterior de
tabs/active no disco. No restart, a task volta a parked/lazy; a ordem e a aba
ativa lógica válidas são reconciliadas, uma única materialização ocorre mesmo
com dois `showTask(T)` repetidos e nenhum query, fragment, título ou segredo
antigo pode introduzir conteúdo não sanitizado. O estado converge para uma task,
uma relação task→tab e um snapshot composto.

### 15.5. C3 — `destroyTask` interrompido

Depois da remoção durável de `T`, qualquer relação de tab órfã é eliminada antes
da próxima recuperação. Ao reiniciar, `listTasks()` não contém `T`,
`showTask(T)` falha com “BrowserTask not found”, nenhuma página pode ser criada
para o ID destruído e a aba/ordem ativa restante é escolhida
deterministicamente. A view crashada também é fechada explicitamente; crash e
destroy não são tratados como a mesma transição.

Esses testes usam apenas `BrowserSessionStateFilePersistence` com IO injetado e
o `WorkstationBrowserRuntime`; não introduzem transaction manager, SessionDB,
page store ou control plane adicional. A prova de integração deve permanecer
junto dos testes C1/C2/C3, para que uma futura alteração do ordering de saves
não volte a ser mascarada por mocks isolados.

