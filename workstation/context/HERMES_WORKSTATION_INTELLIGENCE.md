# Inteligência Centralizada — Hermes Workstation (Hermes Work)

## Execução durável e economia de contexto — 2026-09-16

O Task Compiler classifica planos estruturados e `work_execute`, capability de
sessão em `desktop_ui`, conecta o agente ao DurableBatchRunner existente. Pedidos
repetitivos quantificados exigem compilação antes de mutações avulsas; uma segunda
tentativa sem plano resulta em halt. O modelo decide uma vez; o runtime processa
WorkItems, valida, persiste checkpoints/evidências e retorna somente refs, ledger
e exceções limitadas. Um teste de conversa com provider fake prova 100 operações
com duas chamadas ao provider, sem LLM entre itens.

O Context State Ledger vem da Kanban DB canônica, não do transcript ou reasoning.
Read Cache usa hash após leitura autorizada e detecta alterações com mtime restaurado;
Schema Cache preserva HW-011 e projeta schemas repetidos por ref em sessões Desktop.
Browser Transactions preservam uma BrowserTask; Prompt Queue espera conclusão com
deadline/polls limitados, captura evidência e só então avança. Constraints estruturadas
do usuário não podem ser ampliadas pelo plano. O Circuit Breaker detecta ciclos 2–4.
Blobs duráveis ficam por hash/ref; mensagens multimodais legadas permanecem intactas.
Token_count reportado pelo provider chega ao flush da SessionDB; ausência fica null.

Contratos, limites de recuperação de mutações incertas, métricas e detalhes do
Reference-First Boundary estão em `../ARCHITECTURE.md`, seção Durable execution
routing. Não se reivindica geração automática de digest visual, eliminação de todo
I/O de leitura, cobertura de aplicações reais de browser ou economia de tokens
medida com provider pago. A arquitetura não adiciona stores paralelas.

## Correção arquitetural: execução adaptativa e compilação progressiva — 2026-09-18

Dogfood real do Browser nativo revelou um excesso na fronteira criada para
economizar tokens. O objetivo continua correto: o LLM deve planejar, interpretar e
resolver exceções; runtime determinístico deve carregar IDs, checkpoints, polling,
retries, reconciliação e repetição mecânica. O erro foi deixar o sinal de
repetibilidade atuar como permissão global: uma solicitação reconhecida como
batch podia fazer qualquer mutação posterior exigir work_execute, inclusive
interações stateful necessárias para descobrir o próprio procedimento.

A regra canônica passa a ser: **determinismo é destino do aprendizado, não
pré-requisito da exploração**.

A execução evolui por níveis:

~~~text
ADAPTIVE/DISCOVERY
  -> COMPILED SEGMENT
  -> COMPILED WORK (work_execute)
  -> PROMOTED ROUTINE
  -> drift => NEEDS_REASONING => adaptação localizada
~~~

ALLOW_ADAPTIVE | SUGGEST_COMPILE | REQUIRE_COMPILE | REQUIRE_HUMAN substitui
conceitualmente o latch binário. REQUIRE_COMPILE continua obrigatório para
fan-out mutável homogêneo, mas deve pertencer a uma assinatura/operação concreta
e não contaminar toda a sessão.

No Browser nativo, observar/raciocinar/agir/observar é legítimo durante descoberta
limitada. BrowserTask, TaskRun, lease de mutação, approvals e uncertain-effect
protocol continuam valendo. Procedimentos conhecidos passam a distinguir
PREPARE/INTERACT/COMMIT/VERIFY e escolher evidência proporcional ao efeito:
E0 tool ACK, E1 observação semântica na sessão, E2 readback semântico persistente,
E3 readback persistente independente. E0 nunca prova efeito externo durável.

tools.effects é a taxonomia canônica; browser_console permanece potencialmente
mutante. Tools do Workstation Browser devem normalizar para a rota native_browser
antes de comparar constraints.

O aprendizado deve capturar a experiência adaptativa no
ExecutionJournal/ArtifactStore, gerar candidatos no RecipeStore/ProceduralMemory,
validar/replay, promover via RoutinePromotionService e voltar ao Hermes apenas
quando houver novidade/drift. O runtime deve auto-reusar receitas/rotinas
verificadas quando scope/preconditions/fingerprint coincidirem.

Nova métrica de harness: Guardrail Obstruction Rate — proporção de tarefas com um
caminho seguro/autorizado que foram impedidas pela política do harness. O alvo é
zero sem relaxar canary, uncertainty, fencing, acceptance ou circuit breaker.

Especificação detalhada:
workstation/context/ADAPTIVE_EXECUTION_COMPILATION.md.


## Intake de hardening upstream — 2026-09-18

A varredura das branches/PRs do upstream identificou evidência de campo útil para
endurecer as bordas do Work sem substituir seus owners canônicos. O plano de
implementação e a disposição por prioridade estão em
`workstation/context/UPSTREAM_RELIABILITY_HARDENING_2026-09-18.md`.

Regras para qualquer agente que use essa evidência:

- upstream PR é **referência**, não autorização para cherry-pick cego;
- primeiro verificar se o Work já implementa invariante equivalente;
- quando já existe owner mais forte no Work, importar o cenário de regressão e
  adaptar somente o patch mínimo;
- nenhuma correção pode criar segundo SessionDB/Kanban/BrowserTask/ArtifactStore,
  segundo delivery ledger ou browser paralelo;
- o cluster imediato é: BrowserTask real hide/park/show, recovery/resync de
  transcript, writer único de sessão, proveniência/heartbeat/worker truth do
  Kanban e recovery de browser com orçamento finito.

Estado confirmado no código no momento deste intake:

- BrowserTask já unit-testa `hide -> show`, `park -> show`, página única e
  destruição somente explícita; além disso, H004 já provou em Electron real
  `BrowserWindow` + `WebContentsView` a mesma identidade de página/WebContents
  e renderer sentinel através de hide/show e park/show, enquanto H013 cobre o
  caminho integrado Desktop/Browser. O gap atual não é “não existe E2E nativo”:
  é revalidar/estender esses probes no `main` corrente com timer/input/scroll,
  ação real via controller e assertiva explícita de ausência de fallback;
- o browser nativo já produz seu inventário principal por uma única chamada
  `webContents.executeJavaScript(inventoryScript(...))`; #115056 é otimização
  de qualidade/benchmark, não nova autoridade de snapshot;
- `tools/browser_supervisor.py` ainda possui reconnect pós-attach sem orçamento
  terminal;
- o journal Desktop ainda contém a seleção antiga do primeiro projection row;
- background sync ainda não preserva a mensagem otimista local;
- auto-heartbeat Kanban ainda expressa tentativa, não sucesso persistido;
- a classificação de exit de worker ainda depende de evidência process-local
  quando o exit code não é transportado por um artefato durável.

Esses itens são **planejados até seus testes de aceitação passarem**. Só então
devem migrar desta seção/roadmap para a descrição de arquitetura implementada.

### Mapa de execução confirmado para o próximo agente

- **P0.0:** reaproveitar H004/H013; não criar terceiro harness nativo.
- **P0.1:** portar seletivamente #115068 em
  `apps/desktop/src/lib/inflight-turn-journal.ts` + teste existente e #115085
  em `use-background-sync.ts` + teste existente.
- **P0.2:** estender `hermes_cli/active_sessions.py` para um único writer por
  `session_id`, transferência fenced e resume estrangeiro read-only/observer;
  falha de leitura do registry é fail-closed.
- **P0.3:** validar provenance de Kanban via SessionDB/request-scoped context;
  heartbeat só é verdadeiro se os writes persistirem; delegated child não
  prolonga liveness do worker; exit status ganha evidência durável entre
  processos sem substituir o breaker/protocol-violation policy existente.
- **P0.4:** `tools/browser_supervisor.py` recebe orçamento finito de reconnect
  pós-attach e eviction do supervisor morto; isso é para a lane legado/fallback,
  não uma nova política de WebSocket para o browser Electron interno.
- **P1:** consolidar entrega durável em um único rail; #115056 serve apenas para
  qualidade/freshness/occlusion/benchmark porque o snapshot nativo já é one-call.

Diferença estrutural relevante para #114904: no fork atual, o one-shot exit está
em `cli.py`; não existe o mesmo `hermes_cli/quiet_single_query.py` do upstream.
Portar o contrato de evidência, não a topologia de arquivos.



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

O Hermes Workstation respeita rigorosamente o **Footprint Ladder** (`AGENTS.md`). Não adicionamos ferramentas Core a torto e direito para resolver problemas do Workstation, pois cada ferramenta adicionada é cobrada em tokens em **todas** as chamadas de API:

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

### 15.6. H-053: Kanban híbrido em tempo real

O Hybrid Kanban continua no mesmo domínio e banco `hermes_cli.kanban_db`; não é
uma store do Electron nem um segundo Execution Journal. O endpoint autenticado
`/events` transmite `hybrid_events` e `hybrid_cursor` junto de `task_events`.
Ao receber o frame, o Desktop invalida imediatamente as queries `['kanban',
'hybrid']`, eliminando a latência do polling. A UI permite reordenação horizontal
de colunas (`moveHybridColumn`), drawer de atividade com proveniência `human`/
`agent` e exclusão em cascata de board, coluna e cartão. O domínio aplica
`expected_revision`, locking/transação e reindexação densa; posição visual não
altera `tasks.status` nem conclui automaticamente uma tarefa agêntica.

---

## 16. Resultados grandes, spillover e referências operacionais

A análise de workload desta sessão confirmou que **resultado de ferramenta é
um boundary de armazenamento**, não apenas texto transitório do prompt. O owner
canônico continua em `tools/tool_result_storage.py`; Workstation não deve criar
um segundo ArtifactStore para resolver o mesmo problema.

A defesa contra estouro de contexto possui três camadas complementares:

1. **Cap por tool:** cada ferramenta pode reduzir seu próprio retorno antes de
   entregá-lo ao loop do agente.
2. **Persistência por resultado:** `maybe_persist_tool_result()` retira outputs
   acima do limite do contexto e grava o conteúdo completo em
   `$HERMES_HOME/cache/spillover`.
3. **Budget agregado por turno:** `enforce_turn_budget()` calcula o total de
   resultados do turno e externaliza primeiro os maiores resultados ainda
   inline até ficar abaixo do orçamento agregado.

### 16.1. Content addressing é escopado, não um cache global de execução

Quando há `result_scope`, o nome durável deriva de dois hashes:

```text
scope_hash   = sha256(result_scope)[:16]
content_hash = sha256(content)
result_ref   = result://<scope_hash>/<content_hash>
```

O bloco de substituição entregue ao modelo carrega, quando disponível:

```text
status
result_ref
artifact_ref
content_hash
bytes
cache_status
inline_truncated
```

Conteúdo idêntico **no mesmo escopo** reutiliza o blob imutável e retorna
`cache_status: hit`/`status: unchanged`; o mesmo conteúdo em outro escopo recebe
outra referência. Essa separação é deliberada para impedir que uma referência
se torne reutilizável por acidente através de fronteiras de task/sessão/tenant.

Este mecanismo é um **cache de representação/persistência**, não um cache de
execução. Nunca transforme `(tool_name, args)` em chave global que possa pular
uma ação real, nem reutilize resultado através de escopos só porque os argumentos
parecem iguais. Side effects e dados externos mutáveis continuam exigindo a
execução e a policy próprias.

### 16.2. Host, sandbox e limpeza

O home canônico do spillover é host-side, ao lado dos demais caches Hermes. Em
backends remotos o caminho é traduzido/sincronizado e **testado quanto à
legibilidade** antes de ser devolvido; containers antigos sem o mount correto
caem para escrita no temp da sandbox. A escrita remota grande usa stdin em vez
de embutir o conteúdo no argv/heredoc do comando local, evitando o limite
Linux `MAX_ARG_STRLEN` (~128 KiB por argumento) exatamente no cenário em que a
persistência mais importa.

O Gateway limpa spillover no housekeeping periódico e processos CLI puros fazem
um prune best-effort no primeiro spill. A política operacional é: se o resultado
já foi persistido, **leia o artifact por offset/limit em vez de repetir a mesma
consulta remota**.

### 16.3. Invariantes e anti-patterns

- Um preview truncado nunca substitui o artifact como fonte do resultado
  completo.
- `result_ref` precisa sobreviver a journal, worker handoff, compaction e UI sem
  que o payload gigante volte a ser copiado inline.
- Persistência de resultado não deve depender de uma sandbox já ter sido criada;
  sessões MCP/Gateway sem terminal também precisam do spillover.
- Não introduza um segundo cache, uma segunda store ou um novo banco apenas para
  “otimizar” resultados que o owner atual já externaliza.
- Métricas de regressão devem observar bytes inline, bytes externalizados,
  `cache_status`, refs criadas e refs reutilizadas; wall-clock sozinho não prova
  economia estrutural.

---

## 17. Persistent Workers: claim durável, resultado até ACK e limites de exactly-once

`workstation/workers.py` diferencia claramente **mensagem a executar** de
**resultado entregue**. Essa distinção nasceu da evidência de workload em que
workers long-lived precisavam sobreviver a restart e retomada sem inventar
“completed” depois de uma queda.

### 17.1. Identidades e envelopes

`WorkerMessage` possui `message_id` e um `work_item_id` estável. O último é a
chave que um executor com side effects pode usar para idempotência através de
reconstruções. O registry **não promete exactly-once para um executor externo
arbitrário**.

`WorkerResultEnvelope` preserva pelo menos:

```text
worker_id
parent_task_id
session_id
sequence
status / semantic_status
result_id
work_item_id
deliverables
model / provider
usage / cost_usd
evidence
error
created_at / duration_seconds
```

O record persistente mantém `pending_messages`, `pending_results` e
`in_flight_message` no owner já existente; não existe necessidade de criar um
segundo task/result database.

### 17.2. Ordem de durabilidade

O protocolo correto é:

```text
claim da mensagem
    ↓
persistir in_flight_message
    ↓
executar executor arbitrário
    ↓
construir WorkerResultEnvelope
    ↓
persistir pending_results + limpar in-flight
    ↓
publicar Journal/EventBus
    ↓
consumidor incorpora/persiste o resultado
    ↓
ACK explícito remove pending_result
```

A publicação de `worker.result` ou de um evento de conclusão acontece **somente
depois** de o envelope durável existir. Se a persistência falhar depois do
executor, o código reverte a falsa conclusão, restaura o item em
`in_flight_message` e deixa o worker em estado de falha/reconciliação. Não há
“completed” otimista sem prova durável.

`wait()` é deliberadamente não destrutivo. O resultado continua acessível após
reconstrução até `acknowledge_result(result_id)`. Se o próprio ACK falhar ao
persistir, a remoção em memória é revertida.

### 17.3. Recovery não é replay automático

`recovery_work_item()` expõe um item interrompido, mas não o executa de novo
silenciosamente. Um processo pode morrer **depois** de um efeito externo e
**antes** do resultado durável; replay cego poderia duplicar uma compra, envio,
commit ou outra mutação. O responsável pela retomada deve reconciliar o efeito
ou refazer a operação usando `work_item_id` como chave de idempotência quando o
sistema externo permitir.

Invariante crítico:

> durabilidade do Hermes pode garantir que o trabalho interrompido permaneça
> visível; não pode fabricar exactly-once de um sistema externo que não oferece
> idempotência/reconciliação.

O arquivo atual de registry permanece sob o home do Hermes
(`$HERMES_HOME/workstation/workers.json`) e usa temp + replace atômico, com
retry curto para sharing locks transitórios no Windows. Não crie outro worker
store para resolver o mesmo lifecycle.

---

## 18. Browser dogfooding: sinais operacionais promovidos a contrato

As trajetórias reais analisadas nesta sessão mostraram que autonomia de browser
falha com frequência não por falta de “mais visão”, mas por quatro classes de
fricção: semântica ambígua de input, páginas SPA/canvas ainda hidratando,
auth walls que exigem humano e loops de extração item-a-item. O `main` atual já
incorpora hardening específico para essas classes.

### 18.1. Digitação: clear e append são semânticas explícitas

`browser_type` calcula `clear` de forma compatível com `append`: quando `clear`
não é fornecido, `append=true` impede a limpeza prévia. Não reintroduza um
handler que sempre limpe o campo ou que tente inferir a intenção pelo texto.
Essa diferença é especialmente importante em editores ricos, filtros e inputs
que recebem composição incremental.

### 18.2. SPA/canvas: settlement adaptativo antes de concluir “página vazia”

Quando o snapshot retorna dois ou menos elementos, o runtime verifica sinais de
canvas/WebGL/Maps/feed e pode repetir o inventário em até quatro janelas de
aproximadamente 300 ms. Se o DOM continuar esparso em uma página canvas, o
snapshot anota que a cena é renderizada fora do DOM e sugere percepção/texto ou
`browser_extract_items`.

Isso é um **settlement bounded**, não licença para sleeps longos. Prefira sinais
observáveis e retries curtos condicionados; não volte a `sleep(10/30/60)` como
estratégia genérica para páginas dinâmicas.

### 18.3. Extração estruturada em lote

`browser_extract_items` existe para reduzir o padrão caro
`snapshot → click/read → voltar → repetir` quando a página contém uma coleção
estruturável. Use batch extraction quando a pergunta é “quais itens estão aqui?”
e preserve browser actions individuais para mutações, navegação ou inspeções
pontuais que realmente dependam de estado por item.

### 18.4. Auth wall e handoff humano

`snapshotForEntry()` executa `detectAuthWall(url, title, text)`. Quando detecta
login/verificação/challenge, retorna `wall_detected`/`wall_reason`, atualiza
`lastError` para um handoff humano e deixa a mesma task/perfil disponíveis para
Take Control. A resposta correta não é abrir um browser diferente nem reiniciar
a sessão; é manter a identidade e permitir intervenção humana na mesma página.

### 18.5. Erros de recovery devem ser classificáveis

O hardening do controller preserva a string de erro compatível, mas também
passou a distinguir classes de recuperação como referência stale, falta de tab
vinculada, controle humano, timeout, argumentos inválidos e controller
indisponível. Consumidores novos devem tomar decisões pela classificação/
metadata estável quando disponível, e não por comparação frágil de texto
humano. Para `stale_or_unknown_ref`, a ação correta continua sendo novo snapshot
+ nova ref, nunca retry cego do identificador antigo.

---

## 19. Controle humano: estado real e lease escopado

The historical implementation projected `controlOwner` globally in
`WorkstationBrowserRuntime`; that was the proven gap at the audit base. The
current working tree replaces that authority with `BrowserHumanControlLease`
owned by the bound `BrowserTask`, while retaining `controlOwner` only as a
derived compatibility projection.

Consequências práticas:

- preservar a mesma tab/perfil durante handoff está implementado;
- impedir colisão humano/agente está implementado;
- duas BrowserTasks independentes podem manter scopes diferentes; a lease de
  uma só bloqueia ações do agente dentro daquela task;
- a lease carrega owner, task/session/tab/page/profile scope, aquisição,
  renovação, expiração e release, e é coberta pelos testes Electron focados;
- restore limpa a autoridade humana process-local, portanto crash/restart não
  deixa um lock stale bloqueando o sistema indefinidamente.

The implementation reuses `BrowserTask`, `sessionHost`, recovery state and the
existing owners. An active human lease produces a structured control error for
agent mutation in that scope; an unrelated task remains runnable.

Anti-pattern: criar um `browser_locks.db` ou outro control plane somente para
leases. A identidade da task e o lifecycle já existem; a evolução deve encaixar
nessa fronteira.

---

## 20. Hybrid Kanban e AgentTask são domínios relacionados, não a mesma entidade

H-053 prova que o Hybrid Kanban é Trello-like e humano-first dentro do mesmo
`hermes_cli.kanban_db`, com revisions, activity, realtime e reordenação, mas
**mover um Human Card não altera `tasks.status`**. Essa separação é intencional.

A direção de produto “Entregar isto ao Hermes” deve ser lida como uma **ponte de
delegação**, não como unificação dos ciclos de vida:

```text
Human Card
   │ delegação explícita
   ▼
Agent Task canônica
   │ execução / BrowserTask / workers / approvals
   ▼
result_ref + evidência + status projetado
   │
   └──────────────► Human Card original
```

Na revisão de `main` realizada em 2026-09-15, a ponte ainda era um seam/gap em
`b6ac2d…`. O working tree atual fecha esse gap com a tabela de vínculo
`hybrid_card_delegations` no `hermes_cli.kanban_db`, a ação “Entregar isto ao
Hermes”, idempotência por attempt e projeção compacta de resultado/evidência.

Os invariantes implementados e testados são:

- Human Card continua com colunas, descrição, archive/activity e lifecycle
  humano;
- Agent Task continua usando o task store/lifecycle agêntico existente;
- o vínculo é explícito, persistente, restart-safe e idempotente;
- double-click/retry não cria tarefas duplicadas sem intenção;
- resultado grande volta por `result_ref`/evidência, não por cópia integral;
- concluir/mover uma entidade não conclui/move a outra por efeito colateral;
- retry do mesmo attempt e nova delegação são eventos diferentes;
- não criar segundo AgentTaskStore, segundo Kanban DB ou uma store do Electron
  para mediar a ponte.

Essa fronteira permite que o board humano permaneça simples como Trello e, ao
mesmo tempo, seja uma superfície real de delegação para o Hermes.

---

## 21. Paths cross-platform: a gramática pertence ao dado, não ao runner

A sessão de 2026-09-15 expôs uma classe de bug diferente dos antigos EPERMs de
Windows: código rodando em Linux pode receber **paths Windows válidos** como
dados de evidência/policy. Nesse caso, usar `Path(...)` ou `os.path.abspath()`
do host antes de identificar a gramática destrói a semântica original.

Exemplos concretos:

```text
C:/clean
C:\Windows\System32\calc.exe
/etc/passwd
\\server\share\path
```

`C:/clean` é absoluto segundo gramática Windows mesmo num runner Ubuntu.
`C:\Windows\System32\calc.exe` continua sendo um alvo crítico/protegido mesmo
se a policy estiver sendo testada em POSIX. Reciprocamente, `/etc/...` não deve
ser reinterpretado como Windows só porque o produto principal roda em Windows.

Regra durável:

1. classifique a sintaxe/origem do path;
2. use semântica independente do host (`PureWindowsPath`/`ntpath`,
   `PurePosixPath` ou equivalente explícito);
3. só então faça absolute/containment/sensitive-path checks;
4. comparação de containment entre gramáticas incompatíveis deve falhar de
   forma segura, não ser “normalizada” pelo SO do CI.

### 21.1. Evidência de regressão observada

Em `main@b6ac2d273a43e287122db377bcfa702af6e7553c`, o Workstation CI executou
194 testes Python e terminou com **191 pass / 3 failures**. Duas falhas de
`ReleaseQualificationRunner._check_clean_install()` rejeitaram `C:/clean` como
relativo no Ubuntu; a terceira deixou uma tentativa de remover
`C:\Windows\System32\calc.exe` cair em `REQUIRE_APPROVAL` em vez de
`DENY/CRITICAL`.

Isso é evidência histórica do bug naquele SHA, não autorização para eternizar o
status. Antes de uma nova mudança, reexecute o CI atual. A lição permanente é:
**não corrija esses testes enfraquecendo o assert; corrija a interpretação
host-independent do path**.

Na base atual `main@220a684f465b063411626028b3bdd7444083e3f2`, com
`origin/main` sincronizado e working tree limpo, a mesma suíte passou **247/247**. A correção está
em `workstation.path_utils`, compartilhada por ReleaseQualification e
ScopedPolicyEngine, com containment incompatível tratado de forma fail-closed.

---

## 22. Context compaction, SessionDB e a diferença entre contexto e estado

As trajetórias analisadas mostraram compactions extensas e resumos recursivos.
Isso reforça uma fronteira fundamental: **o contexto do LLM pode ser resumido;
o estado operacional não pode depender exclusivamente do resumo**.

A fonte de verdade operacional deve permanecer nos owners apropriados:

- SessionDB → sessão/histórico de chat;
- Kanban DB → cards/tasks/revisions;
- BrowserSessionState/BrowserTask → estrutura e ownership de browser;
- WorkerRegistry → mensagens, in-flight e resultados pendentes;
- spillover/artifacts → outputs grandes referenciados;
- ExecutionJournal → história/evidência operacional.

Uma compaction pode reduzir prosa, explicações e histórico repetido, mas a
continuação precisa preservar ou poder recuperar identificadores como:

```text
task_id
session_id
worker_id / work_item_id / result_id
BrowserTask id
kanban_card_id / run_id
result_ref / artifact_ref
approval/recovery state
```

Anti-patterns:

- serializar snapshots completos anteriores dentro de todo novo resumo;
- usar compaction como “banco informal” de tasks;
- copiar artifacts gigantes de volta para o resumo;
- perder IDs/refs e compensar consultando novamente APIs remotas;
- inferir que SessionDB está corrompido apenas porque um export histórico
  mostrou estado ausente.

### 22.1. KI-007 / `session:null`: disciplina forense

O caso histórico “Session not found” / export com `session: null` permanece
**observado, mas com causalidade não provada**. O caminho correto é reproduzir
no `main` atual uma sequência determinística de criação → execução → persistência
→ restart/reconnect → export e localizar endpoint, caller, session id e a linha
ou race responsável.

Até essa prova existir:

- não reescreva SessionDB;
- não altere Gateway por hipótese;
- não trate compaction/rotation como causa confirmada;
- mantenha observabilidade e regression coverage capazes de provar ou refutar o
  problema.

Esse padrão vale para qualquer issue persistente: um artefato histórico é uma
pista, não uma licença para alterar o owner canônico sem reprodução atual.

O teste determinístico atual em `tests/hermes_state/test_ki007_session_export.py`
também cobre criação/bind, persistência, finalização, reconnect, rotação-like
writer activity e close/export concorrente; não reproduziu `session: null`.

---

## 23. Evidência de workload real e benchmark de regressão

Os oito SQLite anexados à sessão de 2026-09-15 são logs de trajetórias do
trabalho de engenharia, não stores de produto do Hermes. Juntos preservam
**3.285 steps** e concentram operações sobre browser/workstation, sessão,
Kanban, workers, artifacts, retries, leases, compaction, Gateway e IPC. Eles
servem como evidência de **pressão operacional real**, mas não devem ser
commitados no repositório nem promovidos a fonte canônica de estado.

Uma análise de corpus preservada nessas trajetórias examinou 50 conversas e
registrou aproximadamente:

- 21.037 mensagens;
- 10.463 resultados de ferramentas;
- ~39% de payloads de tool result exatamente repetidos;
- 340 mensagens de context compaction, somando cerca de 3,92 milhões de
  caracteres;
- 578 sleeps explícitos, totalizando pelo menos 19.930 segundos (~5,54 h);
- sequências determinísticas longas de terminal→terminal,
  `browser_console`→`browser_console` e `read_file`→`read_file`.

Esses números explicam por que result refs, batch extraction, workers duráveis,
event-driven waits e compaction disciplinada têm alto valor. Eles **não provam
que cada repetição era removível** nem que cada sleep era bug.

### 23.1. Como transformar o corpus em teste útil

O benchmark de regressão deve usar fixtures sintéticas/provider-free que
reproduzam a forma do workload sem incluir bancos privados. Prefira métricas
estruturais determinísticas:

- quantidade de tool calls;
- bytes de tool results mantidos inline;
- bytes substituídos por refs;
- `cache_status` hit/miss dentro do escopo;
- refs/artifacts criados;
- resultado de worker antes/depois de restart e ACK;
- work item in-flight após crash;
- footprint de compaction;
- classificações de erro/recovery;
- número de polls/sleeps necessários para uma condição simulada;
- capacidade de continuar uma tarefa usando apenas IDs/refs preservados.

Evite thresholds frágeis de wall-clock no CI quando counters/ratios/budgets
provam melhor a regressão. O baseline determinístico versionado agora está em
`workstation/benchmarks/workload_baseline.json`, exercitado por
`workstation/tests/test_workload_benchmark.py`; ele cobre refs/dedup, worker
restart/ACK, compaction, policy errors e event invalidation sem commitar bancos
privados. O objetivo do benchmark é impedir que refactors reintroduzam payloads
gigantes, polling mecânico ou estado implícito, não criar mais um runtime
paralelo.

---

## 24. Extensões do browser: capability governada e atualização last-known-good

O vertical slice agêntico de extensões está implementado como capacidade de
sessão Desktop, não como core tool universal. O fluxo qualificado cobre
instalação/listagem/remoção/options, inspeção de manifesto/permissões,
classificação de risco, policy/approval, load no Electron, verificação e
restauração verificada no startup. Paths inseguros de CRX/ZIP são rejeitados e
uma instalação que não chega a estado carregado/verificado não deve ser tratada
como sucesso.

Permanecem deliberadamente distintos desse slice:

- busca semântica/catálogo de marketplace;
- UI humana dedicada de gerenciamento de extensões;
- integrações de marketplace e outras superfícies ainda futuras.

Para update `v1 → v2`, o invariante **last-known-good** agora é implementado por
staging, journal de promoção, `commit_update`/`rollback_update` e recuperação no
startup. Testes cobrem staging/replace/load/verification, interrupção do
registry e restart; a versão funcional anterior permanece intacta até a
verificação Electron. A busca de marketplace e a UI humana dedicada continuam
deliberadamente futuras. Reutilize `ChromeExtensionManager`, policy, controller
e Journal existentes; não crie outro extension registry para resolver atualização.

---

## 25. Hierarquia de evidência e regra de adjudicação

Este arquivo guarda conhecimento **duradouro**, mas não substitui a evidência do
checkout atual. Quando fontes divergem, use esta ordem:

1. código + testes executáveis do `main` atual;
2. contratos arquiteturais/decisões canônicas vigentes;
3. este arquivo de inteligência para conhecimento já consolidado;
4. `CURRENT_STATE.md` como snapshot datado;
5. `KNOWN_ISSUES.md` para sintomas/reproduções ainda abertas;
6. engineering journal para chronology, experimentos e evidência temporal;
7. `ROADMAP.md` para intenção futura, não implementação presente;
8. corpora/DBs/logs de sessões como evidência histórica de workload.

Consequências:

- uma entrada antiga dizendo `Planned` não vence código + testes que já provam a
  feature;
- uma entrada `Done` não vence um regression test atual reproduzível;
- uma hipótese do corpus não autoriza criar infraestrutura;
- um item de roadmap não deve ser descrito como implementado só porque seu
  design está detalhado;
- antes de criar `*_store`, `*_manager`, `*_runner`, `*_db` ou
  `*_controller`, procure o owner existente e estenda-o;
- resultados de CI sempre devem ser associados a SHA, runner/OS e comando; uma
  contagem histórica nunca é um assert permanente.

A regra de manutenção consolidada após a auditoria desta sessão é:

> **prove o gap antes de implementar; preserve o owner existente; prove a
> correção depois; e promova ao intelligence apenas o que sobrevive à mudança de
> sessão como contrato arquitetural, invariant ou lição de engenharia.**

---

## 26. Adendo técnico da sessão — contratos que devem sobreviver à próxima mudança

Esta seção reúne detalhes que foram confirmados no código e nos contratos de
regressão durante a rodada atual. Ela complementa as seções anteriores; não
introduz outro runtime, store ou autoridade. Se um detalhe abaixo divergir do
checkout futuro, a regra de adjudicação da seção 25 continua valendo: código e
teste executável do `main` vencem documentação histórica.

### 26.1. Compaction: narrativa limitada, handles preservados

O caminho real de compaction em `agent/context_compressor.py` tem duas camadas
determinísticas antes/depois da chamada ao modelo auxiliar:

1. `_build_operational_reference_envelope()` percorre conteúdo e envelope das
   mensagens, aplica a mesma redação de boundary e extrai somente pares
   rotulados (`task_id`, `session_id`, `worker_id`, `parent_task_id`,
   `child_task_id`, `browser_task_id`, `result_ref`, `artifact_ref`,
   `evidence_refs`, `approval_state` e `recovery_id`). Duplicatas são removidas
   e o bloco fica limitado a 6.000 caracteres. O envelope é LLM-free: o modelo
   não pode parafrasear ou omitir um identificador de controle.
2. `_build_anchor_index()` colhe PRs/issues, SHAs, branches, caminhos, erros,
   handles e URLs por regex, com limite de 7.000 caracteres e caps por
   categoria. Esses valores exatos funcionam também como âncoras para
   `session_search`; não são uma cópia da conversa inteira.

O resultado continua sendo uma mensagem de handoff compatível com a alternância
de papéis. Marcadores internos (`_compressed_summary`, delimitador de merge e
marcador de fim) permitem que `split_user_originated_turn()` separe o scaffold
oculto do pedido humano vivo. Assim, um carrier `role=user` não é colhido por
engano como uma nova fala do usuário, e o conteúdo de mídia/API obsoleto é
descartado antes de chegar ao próximo request. A compaction em lote também
zera o resumo rolling de micro-compaction, porque o marcador de lote já contém
uma janela maior; manter os dois seria uma fonte de recursão/duplicação.

O contrato provider-free em
`tests/agent/test_compaction_operational_refs.py` força uma compaction quando o
LLM auxiliar falha e verifica que os handles permanecem byte-identificáveis no
resumo. O teste não prova qualidade semântica do modelo; prova que o estado
operacional não depende da memória narrativa do modelo.

### 26.2. Erros do controller: compatibilidade textual + ação recomendada

`POST /v1/action` mantém o campo textual `error` para clientes antigos, mas o
Electron também devolve um envelope normalizado por
`normalizeWorkstationControllerError()`:

```json
{
  "error_code": "STALE_REF",
  "message": "...",
  "retryable": true,
  "retry_after_ms": 0,
  "state_changed": true,
  "recommended_action": "RESNAPSHOT",
  "resource_ref": "...",
  "details": {}
}
```

As classificações hoje cobertas no runtime são `USER_CONTROL_ACTIVE`,
`NO_BOUND_TAB`, `STALE_REF`, `TIMEOUT`, `CONTROLLER_DOWN`,
`INVALID_ARGUMENT` e `CAPABILITY_MISSING`. Conflitos de controle, tab ausente
e ref stale usam HTTP 409; erros de argumento/capacidade usam 400. A resposta
não deve induzir um retry idêntico: ref stale exige snapshot novo, controle
humano exige esperar release e controller indisponível exige reconciliação.

O adaptador Python (`tools/browser_workstation.py`) ainda preserva
`WorkstationBrowserUnavailable` para descriptor ausente, URL não-loopback,
timeout ou falha de conexão, e `WorkstationBrowserError` para HTTP inválido.
Clientes antigos recebem uma mensagem limitada (até 1.000 caracteres) e não
devem depender de fazer parsing de texto como contrato futuro; quando o
boundary tipado for ampliado, a classificação deve ser transportada sem
perder a compatibilidade de `error`.

### 26.3. Política e paths: gramática do alvo antes do SO do runner

`workstation.path_utils` classifica o texto original como
`WINDOWS_ABSOLUTE`, `POSIX_ABSOLUTE` ou `RELATIVE` antes de usar qualquer
`abspath`. `path_is_within()` retorna `None` para sintaxe incompatível ou
drives diferentes, distinguindo “fora” de “não é comparável”; a
`ScopedPolicyEngine` trata esse caso como boundary não provado e exige
aprovação. `is_sensitive_path()` reconhece Windows (`Windows`, `Program Files`,
shares administrativos e `.ssh/.gnupg/.aws`) e raízes POSIX (`/etc`, `/usr/bin`,
`/bin`, `/sbin`, `/boot`) independentemente do host do teste.

Esse contrato evita que `C:/clean` vire relativo em Ubuntu ou que
`C:\\Windows\\System32\\calc.exe` caia em `REQUIRE_APPROVAL` por ser avaliado
num runner POSIX. A policy ainda separa `DENY` para padrões destrutivos/caminhos
protegidos, `REQUIRE_APPROVAL` para writes fora do workspace e ações de alto
impacto, `SANDBOX` para execução não confiável e `ALLOW` para operações de baixo
risco. Cada decisão mantém `task_id`, `session_id`, capability, alvo, razão,
risk e constraints no audit trail em memória do engine; esse log não é um novo
telemetry plane.

### 26.4. BrowserTask e lease humano: campos, TTL e projeção

`BrowserHumanControlLease` é parte do registro da task e contém
`owner`, `taskId`, `sessionId`, `tabId`, `pageId`, `profileScope`,
`acquiredAt`, `expiresAt` e `renewedAt`. O controller expõe
`takeControl(taskId, sessionId)`, `releaseControl(taskId)`,
`renewControl(taskId)` e `expireHumanControl(taskId)`; o TTL padrão é cinco
minutos. Toda leitura de estado chama `hasActiveHumanControl()`, que expira o
lease vencido antes de projetar permissões.

O campo legado `controlOwner` é derivado da task visível (ou de um lease
temporário da tab `default`), não é mais um lock global que possa bloquear
tasks independentes. `workstation-browser-resources.ts` remove `agent-control`
somente do recurso que possui lease humano; outra task continua executável.
Restore após crash remove a autoridade humana process-local e nunca restaura
um lock sem uma nova aquisição contra a página viva. O teste Electron cobre
escopo por task, renew/release/expiry, seleção da task visível e limpeza no
restart.

### 26.5. Delegação Hybrid Kanban: uma ponte idempotente, não uma fusão

As rotas autenticadas `/hybrid/cards/{card_id}/delegate`,
`/retry-delegation` e `/cancel-delegation` chamam o domínio
`hermes_cli.hybrid_kanban`. A tabela `hybrid_card_delegations` vive no mesmo
SQLite por-board do Kanban; cada linha tem `human_card_id`, `agent_task_id`,
`attempt`, `state`, refs de evidência, resumo e timestamps.

Uma chamada repetida para uma tentativa ativa devolve a mesma task. Retry de
uma delegação bloqueada reutiliza a task e registra atividade própria; falha ou
cancelamento terminal não é ressuscitado silenciosamente. `new_attempt=true`
cria uma nova Agent Task e preserva a tentativa anterior. A sincronização lê o
estado canônico da Agent Task e projeta apenas `state`, resumo, `result_ref` e
`evidence_refs` no cartão humano, com no máximo 2.000 caracteres de resumo.
Mover/concluir o cartão não muda `tasks.status`, e uma transição agêntica não
muda coluna humana sem ação/policy explícita. Os testes cobrem restart,
double-click, retry, redelegação, terminal writeback e ausência de cópia do
payload grande.

### 26.6. Extensões: transação last-known-good no registry existente

`ChromeExtensionManager` usa `extensions.json` como seu pequeno journal de
transação. Durante `prepare_install_from_bytes()`, o registro grava `_update`
com `state=promoting`, `previous_entry`, `backup_path` e `candidate_path`;
somente depois move a instalação anterior para o backup e promove o candidato.
O consumidor que carregar/verificar a extensão chama `commit_update()` após
sucesso ou `rollback_update()` em falha. Um restart encontra `_update` e:

- restaura o backup quando a promoção estava em andamento;
- mantém o candidato se o estado já era `committed`, terminando apenas a
  limpeza;
- remove paths de staging somente depois de verificar que pertencem ao root de
  extensões.

Isso mantém a versão anterior funcional até o load/verification no Electron e
evita que uma interrupção de replace, registry ou cleanup deixe uma extensão
meia-promovida. Não é um segundo registry: o mesmo manager continua sendo dono
de descoberta, instalação, load, verificação e remoção.

### 26.7. Benchmark provider-free: medir estrutura, não maquiar resultado

`python -m workstation.workload_benchmark` executa um workload sintético sem
provider e usa diretamente `WorkerRegistry`, `RuntimeEventBus`,
`ScopedPolicyEngine`, `EvidenceState` e o compactor. O fixture versionado em
`workstation/benchmarks/workload_baseline.json` registra, entre outros:

- cinco tool calls, quatro grandes resultados substituídos por refs e dois
  dedup hits;
- 11 referências operacionais preservadas em 553 caracteres;
- resultado do worker recuperado após restart e removido somente após ACK;
- três decisões de policy (`deny`, `require_approval`, `sandbox`), com o
  caminho sensível negado;
- três invalidações de recurso/evento, zero eventos descartados e zero checks de
  polling.

O benchmark não mede qualidade de provider, não grava bancos privados e não
usa wall-clock como critério de aprovação. Ele detecta regressão estrutural:
payload novamente inline, perda de refs após restart/ACK, compaction sem handles,
policy que deixa atravessar um path sensível ou retorno ao polling mecânico.

---

## 34. Revisão de coerência e provas adicionais — 2026-09-15

Esta rodada posterior auditou o `main` já consolidado em
`220a684f465b063411626028b3bdd7444083e3f2`, depois de verificar o código, os
testes e o histórico recente. O snapshot do `main` auditado e
`origin/main` apontavam para o mesmo commit. As correções abaixo são pequenas
extensões dos owners existentes; não houve criação de um segundo controller,
store, DB, lifecycle ou stack realtime.

### 34.1. Resolução implícita de controle humano deve seguir a aba visível

O Browser Hub chama `takeControl()` sem `taskId` porque a intenção humana é
assumir o conteúdo atualmente visível. O runtime ainda mantém
`preferredTaskId` para que o attach de um painel de Chat selecione a task
correspondente, mas essa preferência é contexto de navegação, não autoridade
de controle.

A ordem segura de resolução passou a ser:

```text
taskId explícito
    > ownerTaskId da aba ativa
    > preferredTaskId do último attach
    > lease não vinculado à aba
```

Antes da correção, a ordem `preferredTaskId > aba ativa` permitia que o usuário
visualizasse a task B, clicasse em Take Control e adquirisse silenciosamente a
lease da task A que havia sido o último contexto de Chat. Nesse intervalo, o
agente de B continuaria autorizado enquanto a UI indicava controle humano. Isso
violava diretamente o isolamento por `BrowserTask`.

O teste Electron de regressão cria duas tasks, anexa o host com A como
preferência, ativa a aba de B, chama `takeControl()` sem argumento e prova que
A continua executável e B retorna o erro estruturado de controle humano. A
regra para novos callers é simples: se não houver `taskId` explícito, derive a
identidade da página efetivamente visível, nunca de uma preferência histórica.

Há uma segunda sutileza no mesmo boundary: `hasActiveHumanControl()` expira a
lease e persiste a remoção. A projeção de `state()` precisa executar essa
expiração antes de capturar `listTasks()`, caso contrário uma única resposta
poderia conter `controlOwner: agent` junto de um snapshot de task ainda exibindo
a lease vencida.

### 34.2. Progressão de Agent Task usa o event boundary canônico do Kanban

O WebSocket `/api/plugins/kanban/events` continua sendo o único canal de
atualização. Ele mantém a conexão SQLite no seu
`ThreadPoolExecutor(max_workers=1)` para respeitar afinidade de thread, lê
`task_events` em lotes limitados e, para cada `task_id` alterado, chama
`hybrid_kanban.sync_delegations_for_agent_task()` na mesma conexão. Essa função
localiza somente os `human_card_id` vinculados e executa a projeção canônica em
transação.

Quando uma transição não terminal muda `queued`, `running` ou `waiting`, o
domínio grava `delegation_progressed` em `hybrid_activity`, incluindo
`previous_state`, `state`, `agent_task_id` e `attempt`. Transições terminais
continuam usando `delegation_completed`, `delegation_failed` ou
`delegation_cancelled`, com `result_ref`/`evidence_refs` compactos. O frame
retorna o evento da task e o evento do card; o renderer invalida somente as
queries Hybrid afetadas. O polling de 4/8 segundos permanece como fallback de
reconciliação, não como fonte primária de progresso.

Esse desenho evita dois erros comuns: esperar que o GET eventual do card seja
o mecanismo de realtime, ou criar um WebSocket específico para o board humano.
Também evita fanout global: uma task sem delegação não produz atividade Hybrid,
e uma task vinculada atualiza apenas seus cards. A activity continua durável,
portanto um cliente que reconecta pode recuperar o evento pelo cursor
`hybrid_since`.

### 34.3. Contratos de teste executados nesta revisão

As provas focadas que fecharam essa rodada foram:

```text
npm exec --workspace apps/desktop -- vitest run --project electron electron/workstation-browser-runtime-task.test.ts
  1 file, 22 tests passed

python -m pytest -q -p no:cacheprovider \
  tests/hermes_cli/test_hybrid_kanban.py \
  tests/plugins/test_kanban_dashboard_plugin.py
  50 passed, 1 warning externo do TestClient/httpx

python -m pytest -q -p no:cacheprovider workstation/tests
  247 passed

npm run --workspace apps/desktop typecheck
  tsc renderer, Electron e E2E concluídos sem erros
```

`git diff --check` também passou. O warning do TestClient/httpx não altera o
contrato funcional testado e não foi mascarado por `skip`, `xfail` ou asserts
relaxados. Uma tentativa inicial de Pytest dentro do sandbox falhou antes da
coleta por `WinError 5` no diretório temporário global do Windows; a execução
aprovada fora dessa limitação confirmou os resultados acima.

### 34.4. Ledger final desta revisão

| Item | Classificação | Evidência | Decisão |
| --- | --- | --- | --- |
| Lease sem `taskId` podia apontar para preferência de Chat stale | `PROVEN_GAP` | teste com A/B, attach em A e aba visível B | corrigida precedência para a aba ativa; teste Electron adicionado |
| Progresso da Agent Task podia depender do polling para aparecer no Human Card | `PARTIAL_HARDENING` | `task_events` já existia, mas a projeção só ocorria no GET do card | reutilizado o WebSocket/cursor existente com projeção seletiva e `delegation_progressed` |
| Paths host-independent, delegação, LKG de extensões, compaction operacional, workers duráveis e benchmark | `ALREADY_IMPLEMENTED` | suites Workstation, Hybrid, extension, compaction e benchmark verdes | preservados os owners e contratos existentes |
| KI-007 / `session: null` | `NOT_REPRODUCED` | teste determinístico de restart/reconnect/export e concorrência | não reescrever SessionDB; manter observabilidade/regression coverage |
| Marketplace semântico e UI dedicada de extensões | `DELIBERATELY_DEFERRED` | roadmap e seção de extensões | não ampliar escopo |
| Proteção obrigatória de branch e checks críticos | `REPO_CONFIGURATION` | estado do repositório não é configurado por código | recomendação permanece para administradores do GitHub |

### 34.5. Adjudicação do WIP encontrado durante a consolidação

Depois do checkout do `main`, havia alterações não commitadas de uma frente de
WIP de Workstation no mesmo diretório. O histórico da branch
`feat/workstation-hybrid-kanban` já era ancestral do `main`; portanto o que
restava não era um merge Git pendente, mas uma camada adicional ainda não
validada. Ela inclui, entre outros, `workstation/artifacts.py`,
`workstation/durable_tasks.py`, `workstation/browser_session.py`,
`workstation/browser_supervisor.py`, `workstation/daemon/` e alterações em
`tools/browser_tool.py`, `tools/close_preview_tool.py`,
`apps/desktop/src/plugins/kanban/api.ts` e `hermes_cli/kanban_db.py`.

Esse WIP não foi promovido para `main` por razões verificáveis:

- `ArtifactStore` duplica o owner de resultados content-addressed já existente
  em `tools/tool_result_storage.py` e não implementa a mesma deduplicação/
  referência escopada;
- `BrowserControlLeaseManager` duplica a lease canônica já pertencente ao
  `BrowserTask` Electron, não compartilha sua persistência nem sua recuperação;
- `workstation/daemon/browser_daemon.py` cria um segundo Browser Controller,
  contrariando o owner Electron/loopback existente;
- a mudança de assinatura de `delegateHybridCard` não é compatível com a UI
  atual, que ainda passa `newAttempt` booleano, e adiciona uma rota de leitura
  sem correspondente backend comprovado;
- o teste de pós-processamento de `browser_extract_items` não concluiu em
  execução focalizada, impedindo chamar essa integração de provider-free e
  operacionalmente comprovada.

Esses itens ficam classificados como `DELIBERATELY_DEFERRED`/WIP preservado,
não como código integrado. A decisão evita “resolver” o conflito apagando
trabalho de outra frente e evita comprometer os invariantes do produto. Para
uma futura promoção, a frente precisa primeiro reutilizar os owners canônicos,
remover as duplicações, corrigir o contrato da API/UI, adicionar testes com
terminação bounded e passar por uma revisão independente de arquitetura. O WIP
foi preservado de forma recuperável em um stash Git criado antes da
sincronização;
os commits de documentação desta rodada contêm somente a inteligência
revisada.
---

## 27. Backend Desktop, Gateway e readiness: separar boot, transporte, sessão e provider

O Desktop não deve inferir que “Hermes está pronto” a partir de uma linha de
stdout/stderr, de um PID vivo ou da simples existência de uma porta. O contrato
atual é um backend **headless** separado da UI: o comando canônico é
`hermes serve --host 127.0.0.1 --port 0`; apenas runtimes antigos que ainda não
registram `serve` recebem o fallback compatível `dashboard --no-open`. A UI
Electron e o browser interno continuam sendo processos/superfícies distintas do
Gateway.

### 27.1. Readiness autenticada é um protocolo, não um sleep

`apps/desktop/electron/backend-health.ts` trata readiness como polling de
`/api/health`, com orçamento default de 45 s, intervalo de 500 ms e timeout de
5 s por health probe. O fallback para `/api/status` existe somente para runtimes
legados em que `/api/health` está realmente ausente ou escondido por um gate
anônimo compatível.

A classificação de falhas faz parte do contrato:

- `401/403` em **probe credenciado** significa sessão rejeitada e deve falhar
  rápido para reautenticação; não é “backend ainda subindo”;
- `404` genuíno de `/api/health` pode significar runtime antigo e habilitar o
  fallback compatível;
- `429` e timeouts permanecem transitórios dentro do budget;
- `502/503/504` são falhas server-side e não devem ser reescritas como erro de
  configuração local;
- sucesso de health significa que o backend respondeu ao protocolo esperado,
  não que um provider remoto específico terá baixa latência.

Uma regressão histórica do corpus fazia o boot depender da stream errada:
readiness era emitida em stderr enquanto o launcher observava stdout. A lição
durável é **não fazer a autoridade de readiness depender de uma única stream de
logs**. No `main` atual, a autoridade é o endpoint de health, não a antiga linha
de console.

### 27.2. Tentativas concorrentes usam geração para evitar teardown stale

`createBackendConnectionState()` mantém uma `generation`. Um attempt ou child
process só pode instalar/limpar promise/processo se ainda pertencer à geração
vigente. `invalidate()` incrementa a geração antes de devolver o processo antigo
para teardown.

Isso impede uma race importante:

```text
attempt A inicia
attempt B invalida A e inicia
A termina atrasado
A NÃO pode limpar/substituir o processo ou promise de B
```

Não troque esse mecanismo por um booleano global `connecting` nem permita que
callbacks de um child antigo mutem o owner novo.

### 27.3. Teardown precisa matar a árvore certa

O child gerenciado pelo Desktop tem semântica diferente por SO:

- Windows usa tree-kill (`taskkill /T /F`) para encerrar o root e seus
  descendentes;
- POSIX sinaliza primeiro o process group (`process.kill(-pid, SIGTERM)`), pois
  o backend é lançado em grupo/sessão própria; se isso falhar, cai para o child
  direto.

Encerrar apenas o PID pai pode deixar gateway/workers órfãos; matar por nome de
processo, por outro lado, pode atingir instâncias que pertencem a outro usuário,
perfil ou teste.

### 27.4. O reader do Gateway é infraestrutura de controle

O loop que lê RPC não pode ficar bloqueado por operações de segundos/minutos.
Handlers lentos — inclusive `workstation.resources` e `workstation.events` —
são enviados para um pool, deixando `approval.respond`,
`session.interrupt` e o fast path legíveis enquanto o trabalho continua.

Sessões WebSocket desconectadas são estacionadas por um curto grace period para
reconexão, mas não ficam órfãs indefinidamente: o Gateway interrompe e reapa uma
sessão abandonada após o budget configurado. Reconexão temporária e abandono
definitivo são estados diferentes.

### 27.5. Taxonomia de diagnóstico

Antes de “corrigir a lentidão”, localize o domínio:

1. **boot do backend local** — processo não iniciou/health não responde;
2. **auth/transporte do Gateway** — credencial rejeitada, WebSocket/RPC,
   reconnect;
3. **controller do Workstation Browser** — descriptor/token/loopback da
   automação web;
4. **provider LLM** — TTFT/rate limit/latência depois que o Gateway já está
   saudável;
5. **site externo** — navegação, auth wall, Cloudflare, rede do destino.

Misturar essas camadas produz falsos fixes — por exemplo, aumentar timeout do
provider não corrige health local, e reiniciar o Browser Controller não corrige
TTFT de uma API remota.

---

## 28. Isolamento multi-sessão e execução de browser em background

As conversas de dogfooding mostraram uma falha de produto particularmente
perigosa: duas conversas podiam “brigar” pela aba visualmente ativa. O contrato
correto é o oposto: **a aba ativa é uma escolha de apresentação; a identidade da
execução é a BrowserTask**.

No runtime atual, cada entrada task-owned carrega `ownerTaskId`. A resolução de
uma ação do controller recebe a task específica e opera naquela `BrowserEntry`,
inclusive quando ela está em background. Uma ação de uma task em background não
deve ativar/focar a aba só para poder clicar, digitar, extrair ou navegar.

A ativação visual ocorre quando é semanticamente necessária, por exemplo:

- não existe aba ativa ainda;
- a própria task já é a ativa;
- a UI anexou o viewport com aquela task como `preferredTaskId` porque o usuário
  mudou deliberadamente para a conversa/host correspondente.

`attach(window, bounds, host, preferredTaskId)` é, portanto, uma operação de
apresentação/seleção, não uma transferência de ownership operacional.

### 28.1. `hide`, `park` e `setVisible(false)` não são sinônimos

Uma nuance importante descoberta no corpus e confirmada no runtime atual:

- **hide** desanexa a view da superfície visível sem destruir a página;
- **setVisible(false)** também pode remover a view nativa temporariamente, por
  exemplo para permitir overlays React/Radix sobre o compositor;
- **park** preserva uma task viva em background e pode manter uma view
  task-owned acoplada no compositor na borda da janela, com apenas um sliver
  mínimo visível e frame rate reduzido. Isso existe porque páginas Chromium
  completamente retiradas do compositor podem deixar de renderizar/avançar
  como uma automação background espera.

As views são criadas com `backgroundThrottling: false`; o runtime ainda reduz
explicitamente o FPS das entradas não visíveis. Não “otimize” removendo
indiscriminadamente toda view estacionada nem mantenha todas a 60 FPS.

### 28.2. Invariantes de isolamento

- uma `BrowserTask` possui no máximo uma live page;
- uma live page não pertence a duas tasks;
- `sessionHost`, `kanbanCardId` e `runId` são linhagem, não chaves alternativas
  para compartilhar a mesma page;
- uma task em background pode continuar, mas não pode roubar foco/viewport da
  sessão que o usuário está olhando;
- mudar de chat não migra implicitamente a task anterior;
- `activeTabId` não determina qual task um worker pode mutar;
- crash/restart restaura identidade estrutural e perfil, nunca o mesmo objeto
  `WebContents`;
- qualquer fallback que mova uma task bound para outro browser viola isolamento.

Esse contrato deve ser testado com **duas sessões simultâneas**, não apenas uma
task isolada: navegação/ação em A enquanto B está visível, troca para A, volta
para B, background progress e restart.

---

## 29. Promoção automática para Kanban e follow-ups descobertos em execução

O Agentic Kanban não serve apenas como UI. `WorkstationKanbanBridge` é a ponte
que transforma trabalho suficientemente longo/multistep em uma task canônica do
Kanban existente.

### 29.1. Promoção do pedido raiz

`promote_request_if_multistep()`:

- respeita `tasks.create_kanban_for_multistep`;
- usa `is_multistep_request()` para reconhecer workflows, automação,
  pesquisa/web/browser, batch e pedidos compostos;
- cria a task via `hermes_cli.kanban_db` com `created_by="workstation"`,
  `session_id` e estado inicial `running`;
- inicia/continua o `ExecutionJournal` da mesma task e grava `TASK_CREATED`.

A heurística decide **visibilidade operacional**, não cria um novo executor.
Desabilitar a UI do Kanban não deve mudar o owner da execução.

### 29.2. Dependências de follow-up têm direção semântica

Quando uma execução descobre uma nova `DiscoveredTask`, a relação depende de ela
ser necessária para terminar o pai:

- **required_for_parent:** o child precisa ser imediatamente executável; por
  isso ele nasce sem depender do pai, depois é criado o vínculo
  `child -> parent` (child é pré-requisito do pai) e o pai fica `blocked`;
- **opcional/depois do pai:** o child pode nascer com o pai como pré-requisito.

Inverter esse vínculo cria deadlock lógico: um child obrigatório não pode
depender da task que justamente está bloqueada esperando por ele.

O journal do pai guarda `child_task_id`, razão, descobridor, origem,
`required_for_parent` e evidências. A hierarquia de produto e a direção do
grafo de dependência são relacionadas, mas **não são a mesma coisa**.

### 29.3. Conclusão exige relatório estruturado

`complete_task_with_report()` usa `BrowserTaskReport` para levar objetivo,
resultado, sites e evidências ao Kanban canônico e grava `TASK_COMPLETED` no
Journal. Não marque uma task como concluída apenas porque a UI fechou a aba ou
porque o modelo produziu uma resposta textual.

Esse fluxo é distinto do Hybrid Kanban humano da seção 20: promoção automática
cria/acompanha **Agent Tasks**; delegação de Human Card cria um vínculo explícito
entre os dois lifecycles.

---

## 30. Control plane agêntico V1–V3: owners complementares, não um “super store”

As conversas mais longas do corpus registram a evolução de um conjunto de
subsistemas que permanece presente no `main`. Eles devem ser entendidos como
**camadas complementares sobre a task canônica**, e não como bancos concorrentes.

### 30.1. Perception Engine: o modelo vê uma projeção, não o DOM bruto

`PerceptionEngine` transforma captura DOM/accessibility em `PerceptionView`
compacta, com refs e provenance path. Dois hardenings são especialmente
duradouros:

- nós explicitamente ocultos (`hidden`, `aria-hidden`, `type=hidden`,
  `display:none`, `visibility:hidden`) são excluídos antes de virar evidência;
- quando o token budget aperta, actions/forms/inputs são Tier 1, links Tier 2 e
  conteúdo estático Tier 3. A truncation preserva primeiro o que permite agir.

Não volte a truncar “do topo para baixo”: CTAs e formulários no footer podem ser
semanticamente mais importantes que centenas de blocos de texto anteriores.

### 30.2. Procedural Memory: memória de como fazer, não PKM

`ProceduralMemory` persiste `WebProcedure` e `MemoryRecord`. Um `ProcedureStep`
pode resolver target por fallback semântico na ordem:

```text
data-testid -> role/name -> texto -> selector original
```

Isso reduz fragilidade de selectors Tailwind/CSS e permite revalidar uma
procedure quando o DOM muda. Procedures têm lifecycle
`discovered -> validated -> promoted -> retired`, evidência de validação,
confidence e contadores de sucesso/falha.

Não confunda essa store com SessionDB, Vault ou BrowserSessionState.

### 30.3. Drift Governor: adaptar só dentro do risco permitido

`DriftGovernor` compara expectativa procedural com a percepção atual e classifica
`none`, `benign`, `structural` ou `breaking`. Ele pode sugerir proceed,
adapted-target retry, dismiss overlay, re-explore, approval ou halt.

Guardas duráveis:

- CAPTCHA/security check/session expired são breaking e pedem intervenção;
- overlays de consent/cookies podem receber `DISMISS_OVERLAY` quando há alvo
  seguro;
- drift estrutural com candidato semântico forte pode adaptar;
- se a ação for high/critical ou já exigir approval, **drift nunca autoriza
  auto-adaptação de alto impacto**.

### 30.4. Scheduler: lease e heartbeat evitam ownership órfão

`MultiTaskScheduler` mantém estados `queued`, `active`,
`waiting_for_human`, `parked`, `completed`, `failed`; dispatch é prioridade
descendente + FIFO. A task ativa recebe lease com expiry, renovável por
`heartbeat()`. `reap_expired_leases()` estaciona owner abandonado e libera a
fila.

Esse scheduler é uma camada de coordenação em memória e não substitui o Kanban
durável nem o `BrowserHumanControlLease`. O estado `ACTIVE` do scheduler deve
ser lido como ownership do slot que ele governa, **não** como prova de que toda
BrowserTask `parked` deixou de progredir: o runtime de browser atual suporta
pages task-owned em background. Scheduler state e BrowserTask visibility são
projeções diferentes e não devem ser colapsadas.

### 30.5. EvidenceState: `running` precisa ser conquistado

`EvidenceState` não aceita `RUNNING` como mero label. Sem
`OperationalEvidence` viva, uma transição para running degrada para `STALLED`
com recovery strategy. Evidência expira e `reconcile()` rebaixa uma task cuja
prova operacional ficou stale.

Isso impede o anti-pattern “task marcada running para sempre porque um JSON
antigo dizia running”.

### 30.6. EventBus, Supervisor e Recovery Plane

`RuntimeEventBus` faz fan-out não bloqueante com filas limitadas. Quando uma fila
estoura, o drop é contado; erro de journal fica observável e não trava todos os
subscribers.

`RuntimeSupervisor` vive fora do runtime supervisionado, faz health/restart com
limite de crash loop e pode checkpointar/restaurar artifact last-known-good.
`RecoveryPlane` registra quarantine/restore/diagnostics fora da task normal.

A recuperação não deve depender do mesmo componente que acabou de morrer.

### 30.7. Host capabilities, events e policy

`HostCapabilityProvider` expõe filesystem/process/clipboard/git/notification/
diagnostics atrás de adapters de SO. Resultado de capability é estruturado; o
caller não deve inferir sucesso apenas de stdout.

`SystemEventPipeline` recebe crash, build failure/success, download, mudança de
repo/controller, completion e necessidade de atenção. Se já há task alvo, ele
enriquece o `ExecutionJournal`; evento crítico sem task pode ser promovido via
`WorkstationKanbanBridge`. Não existe um “event-task DB” paralelo.

`ScopedPolicyEngine` decide `ALLOW`, `SANDBOX`, `REQUIRE_APPROVAL` ou `DENY`
com task/session/capability/target/workspace. Comandos destrutivos, paths
sensíveis, boundary de workspace e extensões de risco são avaliados antes da
execução. Policy deve usar a gramática cross-platform da seção 21.

### 30.8. Budget e roteamento de modelo continuam fora do provider call

`BudgetTracker` aplica tetos explícitos de actions, tokens e custo e lança
`BudgetExceeded` antes de aceitar consumo que ultrapasse o limite. `ModelRouter`
escolhe deterministicamente entre candidatos disponíveis ponderando qualidade,
custo e latência; a chamada real ao provider permanece em outra camada.

Isso evita dois anti-patterns: esconder budget dentro de prompts (“tente gastar
pouco”) e acoplar a decisão de modelo ao transporte/provider. Budget é contrato
de execução; routing é decisão observável; provider call é efeito externo.

### 30.9. Lightpanda é rota opt-in e fail-closed

O adapter Lightpanda só pode aceitar tarefas:

```text
enabled
AND public_read_only
AND headless_ok
AND NOT bound_to_internal
AND NOT requires_auth
AND NOT requires_visible_state
```

Redirect para login/auth falha fechado; gzip/deflate são tratados antes da
extração. Uma tarefa autenticada ou já bound ao Chromium interno nunca deve ser
“otimizada” silenciosamente para um browser stateless.

---

## 31. Hermes Vault: PKM local-first separado da memória operacional

`workstation/vault.py` implementa uma base de conhecimento Markdown sob
`$HERMES_HOME/vault` por default. É deliberadamente interoperável com a
gramática de PKM estilo Obsidian:

- `.md` como formato primário;
- YAML frontmatter;
- tags inline/frontmatter;
- aliases;
- `[[wikilinks]]`, inclusive heading e alias;
- índice bidirecional de forward links/backlinks;
- pesquisa por título/tag/conteúdo;
- grafo `nodes/edges` derivado dos wikilinks;
- sugestões de wikilink/autocomplete.

O índice é reconstruível a partir dos arquivos; **os Markdown são a fonte
durável**, não o grafo em memória.

### 31.1. Fronteiras com outros stores

| Store | Pergunta que responde |
|---|---|
| SessionDB | “o que aconteceu nesta conversa/sessão?” |
| ExecutionJournal | “o que esta task executou e que evidência produziu?” |
| ProceduralMemory | “como executar novamente este tipo de procedimento?” |
| Vault | “que conhecimento/notas o usuário quer conservar e conectar?” |
| BrowserSessionState | “que estrutura de tabs/tasks pode ser recuperada?” |

Não copie automaticamente todo chat, tool result ou Journal para o Vault.
Promoção de conhecimento deve ser explícita/criteriosa; do contrário PKM vira
um segundo log operacional impossível de manter.

### 31.2. Segurança de path é parte da API

`VaultManager.write_note()` aceita `subfolder`; qualquer evolução de escrita,
importação, sync ou plugin deve garantir containment real dentro de
`vault_dir`, inclusive `..`, symlink/reparse point e diferenças de gramática de
path. Desde o hardening de 2026-09-15, o owner valida gramática host + Windows
antes de qualquer side effect, exclui symlinks de arquivo/diretório, revalida o
parent e usa replace atômico. Tools e plugin RPC compartilham a instância
process-wide e o watcher polling bounded/debounced. Novas operações devem
passar pelas mesmas primitivas; não basta fazer `resolve()` depois da escrita.

---

## 32. Validação nativa: processo real, baseline exato e evidência negativa

O corpus contém várias sessões em que testes “verdes” seriam insuficientes para
provar o comportamento do Workstation. A metodologia consolidada é:

### 32.1. Compare identidades de teste, não quantidade de vermelhos

Para causalidade baseline×candidate:

1. derive a lista exata de arquivos/testes do baseline;
2. rode **a mesma lista** no candidate;
3. registre ambiente (SO/build, Node/npm/Python/Git), SHA e comando;
4. compare nomes/primeiro erro de cada falha;
5. classifique `PASS`, `FAIL regression`, `FAIL baseline` ou
   `NON-EXECUTABLE`.

Testes novos do candidate são úteis para cobertura, mas não podem alterar a
população usada para provar causalidade contra o baseline.

### 32.2. BrowserSessionState precisa atravessar um processo Electron real

A validação nativa do corpus provou, em um candidato histórico, sequência com
tabs ordinárias + task tab, ordem restaurada, active tab lógica, lazy restore,
primeiro/segundo `showTask`, ownership único, separação de perfil, ausência de
process identity no snapshot e recuperação após saída abrupta.

A lição é o formato do teste, não o SHA histórico:

- teste unitário de persistence não substitui restart entre processos;
- graceful shutdown e abrupt exit são cenários separados;
- cookies/login pertencem ao perfil Chromium, não ao JSON estrutural;
- restore lazy deve provar **zero pages eager antes do show**;
- repetir show não pode criar segunda page;
- segredo deve ser procurado no retorno, no JSON em disco, no reload e na
  reserialização — evidência negativa em todas as fronteiras.

### 32.3. Harness também pode causar falso negativo

No Windows, um subprocess Electron pode ter terminado logicamente e ainda
parecer “pendurado” ao harness por handles/pipes stdout/stderr herdados. Testes
de exit precisam administrar streams e árvore de processos explicitamente.

Da mesma forma, compositor nativo `WebContentsView`, z-order/oclusão e alguns
fluxos visuais não devem ser declarados PASS só porque uma simulação headless
passou. Use smoke nativo/artefato visual/manual quando a propriedade que está
sendo testada existe no compositor real.

### 32.4. Fault injection deve atingir o seam certo

Para crash consistency, injete falha entre os replacements reais e reinicie a
partir do snapshot intermediário exato. “Mockar save como false” sem reproduzir
o ordering de efeitos pode esconder o bug que se queria testar.

---

## 33. Corpus ampliado desta auditoria: como usar conversas históricas sem criar doc drift

O arquivo `conversations.zip` analisado nesta sessão continha **23 bancos SQLite
de trajetória**. A leitura percorreu **14.509 steps** e recuperou cerca de
**73,35 milhões de caracteres** de strings presentes em prompts, respostas,
tool calls/results, erros e metadados.

Esse corpus inclui:

- sessões profundas do Hermes Workstation;
- subagents interrompidos/rate-limited;
- pesquisas e implementação em branches históricas;
- conversas de outros projetos, como SudoExpo Match e K-Tools.

Tudo foi inventariado, mas somente conteúdo pertencente ao Hermes Workstation
deve entrar nesta inteligência.

### 33.1. Regra de promoção do histórico

Uma afirmação encontrada em conversa histórica recebe uma destas classes:

- **CURRENT-CONFIRMED:** o `main` atual ainda possui código/contrato que a prova;
- **DURABLE-LESSON:** o bug antigo pode já estar corrigido, mas o invariant/teste
  que ele revelou continua válido;
- **HISTORICAL-ONLY:** descreve SHA/branch/launcher que já não existe;
- **UNPROVEN:** plano, hipótese ou claim de agente sem confirmação no checkout;
- **OUT-OF-DOMAIN:** pertence a outro projeto.

Somente as duas primeiras devem ser promovidas como conhecimento arquitetural
normal. `HISTORICAL-ONLY` só entra quando explica um anti-pattern importante;
`UNPROVEN` permanece em journal/roadmap/known issue; `OUT-OF-DOMAIN` é excluído.

### 33.2. Conversa não vence código

Exemplos desta própria auditoria:

- o corpus descrevia `HERMES_DESKTOP_READY_FILE`; o `main` atual não o possui,
  portanto ele não é documentado como protocolo vigente;
- claims antigos de lease humano global foram superados pelo lease task-scoped
  atual;
- a delegação Human Card -> Agent Task era roadmap em uma sessão e virou
  implementação depois; o intelligence deve refletir o estado confirmado mais
  recente;
- bugs de BrowserSessionState em SHAs históricos entram como regressions que os
  testes devem impedir, não como bugs atuais automaticamente.

Esse procedimento é a defesa principal contra transformar
`HERMES_WORKSTATION_INTELLIGENCE.md` em arqueologia contraditória.

---

## 35. Execução repetitiva durável: compilação única, dispatch existente e retorno por referência

Esta seção descreve o contrato atual da capability `work_execute`, integrada em
2026-09-16. Ela trata trabalho homogêneo e quantificado sem criar um executor,
um browser ou uma store paralela. Seu objetivo é manter o modelo na decisão e
o runtime na repetição verificável.

### 35.1. Entrada no tool loop e surface de sessão

`tools/workstation_work.py` registra `work_execute` no toolset `desktop_ui`.
Ele não pertence a `_HERMES_CORE_TOOLS`: clientes sem a superfície Desktop não
pagam seu schema em cada chamada. O `check_fn` consulta apenas se Workstation
está habilitado; ele não usa `HERMES_DESKTOP` para inferir que uma GUI está
observando a sessão. A inclusão do toolset continua sendo resolvida pela origem
da sessão no Gateway, como os demais recursos de Desktop.

No loop de conversa, `batch_intent()` reconhece pedidos repetitivos
quantificados e `requires_compilation()` bloqueia mutações isoladas que seriam
parte daquele lote. O primeiro bloqueio devolve um resultado compacto
`durable_compile_required`: nenhuma ferramenta subjacente executa. Uma segunda
tentativa sem plano produz `durable_compile_failed` e aciona o halt do
guardrail. Isso é uma defesa de custo e de segurança: não deixe o modelo
alternar “planejar um item, mutar um item” cem vezes e chamar isso de batch.

O plano enviado a `work_execute` usa `operation_key`, `items` **ou**
`items_ref`, e passos `{tool, args, expect, wait}`. Valores em `args` podem
usar `$item.campo`; cada mutação requer `expect` que confira um caminho JSON do
resultado. `items_ref` é preferível quando o conjunto já está no ArtifactStore:
não rematerialize um dataset grande no transcript para iniciar ou retomar uma
execução.

### 35.2. Ownership, identidade e retomada

`TaskCompiler` deriva a identidade do trabalho da conversa raiz estável, do
task id original e de `operation_key`. `run_agent.py` passa
`_conversation_root_id()` ao contexto de execução; uma compaction que cria uma
sessão filha não pode assumir a execução da conversa pai. O request
fingerprint persistido impede reutilizar a mesma `operation_key` para um
objetivo diferente.

O resultado é materializado nos `WorkPlan`/`WorkItem` canônicos e no
ArtifactStore existente. Antes do dispatch de uma mutação, o runner persiste a
intenção; depois de verificar o resultado, persiste evidência/checkpoint e só
então avança. Em restart, passos concluídos são pulados. Uma mutação cuja
intenção foi persistida mas não tem checkpoint verificável vira bloqueio para
raciocínio/handoff: ela não é repetida no escuro apenas para recuperar
progresso.

`plan_id` permite `status` ou `resume` sem depender do transcript. A retomada
recarrega objective, constraints e o `browser_task_id` original dos metadados
duráveis e confirma que a conversa solicitante é a dona. Para browser work,
isso conserva a `BrowserTask` já bound; controller ausente ou perda dessa
capacidade falha fechado, sem migrar cookies/estado para um fallback genérico.

### 35.3. Dispatcher, BrowserTask e prompt queues

O compiler não chama handlers arbitrariamente. `execution_context()` recebe do
agente o dispatcher scoped já existente. Ferramentas deferred passam pela
resolução e middleware normal; portanto política, validação de schema,
aprovação, captura de resultado e guardrails não são contornados pelo batch.
O executor sequencial é chamado com `finalize=False`, para que mensagens
internas de cem itens não sejam gravadas como cem rodadas de conversa; o
resultado consolidado externo é o boundary de transcript.

O contexto também carrega callback de progresso, resultados brutos capturados
antes do spillover e refs operacionais confiáveis. Ao fim, `run_agent.py`
anexa os refs `KanbanRun` à mensagem externa de tool. O ledger é uma projeção
derivada da DB canônica: plano, fase, objective ref, constraints, itens
concluídos/pendentes/falhos, artifact handles, blockers e próxima ação. Ele não
é transcript, cadeia de raciocínio, nem mais uma fonte de verdade de Kanban.

Browser transactions mantêm uma única BrowserTask. Prompt queues só aceitam
espera de ferramenta read-only, com deadline, interval e número máximo de
polls; cada conclusão observada é persistida como evidência antes do próximo
item. Falha de browser ou queue interrompe o lote no primeiro desvio, porque
continuar assumindo que o estado visual/assíncrono permaneceu igual é um erro
de ownership.

### 35.4. Constraints e roteamento sem sondagem lateral

`user_constraints()` aceita apenas constraints estruturadas ou a gramática
estreita dos pedidos explícitos de rota. `merge_constraints()` pode reduzir o
conjunto permitido, nunca ampliá-lo: allowed routes são intersectadas e
forbidden routes são unidas. `require_allowed_route()` valida listas limitadas
(até 64 strings de até 256 caracteres) e bloqueia por prefixo antes de dispatch,
descoberta, probe de credencial ou chamada de provider.

`ModelRouter.choose()` aplica o mesmo filtro antes de pontuar candidatos; o
provider `openai` é comparado à rota `openai_api`. Não tente contornar uma
constraint consultando primeiro a disponibilidade do provider. A ausência de
rota válida é um resultado explícito, não justificativa para fallback silencioso.

### 35.5. Reference-first boundary, caches e telemetria honesta

`content_reference()` armazena conteúdo por hash dentro do escopo da task.
`blob_references()` troca payloads `data:` por descritores MIME/hash/ref e
preserva bytes recuperáveis; um digest visual só recebe referência quando foi
fornecido por uma fonte real. Não se inventa digest nem se reintroduz pixels no
contexto apenas para parecer multimodal.

`ReadCache` reduz contexto repetido, não autorização nem I/O: `read_file`
ainda faz uma leitura fresca autorizada e calcula o hash do conteúdo. Isso
detecta a edição de mesmo tamanho com mtime restaurado, caso que uma deduplicação
por mtime isolado perderia. `schema_projection()` mantém a primeira descrição
completa de ferramenta e, nas repetições, retorna somente tool/hash/ref/capacidades
quando o fingerprint do registry e da capability não mudou.

As métricas contam dispatches, transições, bytes de entrada/saída de ferramentas,
cache hits/misses, checkpoints já concluídos e tamanho físico de artifacts. Uso
de tokens só é preenchido quando o provider o reporta no request de compilação;
nesse caso o escopo é literalmente `compile_request`. Sem relatório, o valor é
desconhecido, não zero e não uma estimativa de economia. O `token_count` da
mensagem assistant e `provider_usage` chegam ao flush da SessionDB quando
existem; ausência permanece `null`.

### 35.6. Guardrails e classificação de falhas

O controller de guardrails mantém histórico limitado de chamadas e detecta
ciclos idênticos de duas a quatro iterações, avisando depois de três voltas e
parando no limiar configurado. Pollers conhecidos continuam isentos; falhas
distintas de terminal/processo/browser não recebem halt meramente por terem o
mesmo nome de ferramenta. Progresso só é resetado por mudança observável ou
por `mark_verified_progress()`, nunca pelo texto otimista do resultado.

Itens read-only podem usar retry limitado. Falha de mutação, verifier inválido,
interrupção, estado `waiting`, `blocked`, `cancelled` ou `suspect` não recebe
replay automático. A saída compacta separa itens concluídos, exceções que
pedem raciocínio e refs para evidência completa; `verbosity=full` ainda retorna
registros por referência, não um novo transcript gigante.

### 35.7. Provas de regressão e anti-patterns

`workstation/tests/test_durable_agent_integration.py` executa o dispatcher
sequencial real do `AIAgent`: 100 operações produzem um único resultado externo
e o provider fake recebe duas chamadas. `test_task_compiler.py` cobre restart
após 37 itens, retomada por `plan_id`, isolamento de owner, checkpoint de
mutação, constraints, queue bounded, cache e métricas de uso conhecido ou
desconhecido. `test_reference_plane.py` cobre hash de conteúdo, recuperação de
blob, projeção de schema e invalidação com mtime restaurado.

O benchmark provider-free em `workstation/benchmarks/durable_execution.py`
modela 100 limites de planner contra dois, mede transporte inline e injeta
falha transitória, crash e exceção cognitiva. Ele prova propriedades
estruturais — inclusive zero replay dos itens concluídos — e não mede preço de
tokens nem comportamento de browser real.

Anti-patterns proibidos nesta fronteira:

- criar `TaskCompiler` como segundo dono de Kanban, BrowserTask, SessionDB ou
  artifacts;
- executar itens por handlers diretos e escapar de middleware/policy;
- tratar sucesso textual como verifier, ou repetir mutação incerta;
- guardar outputs integrais no histórico de chat para “facilitar resume”;
- habilitar tool de GUI por env de processo, ou sondar rota proibida;
- alegar token saving pago a partir de um benchmark sintético.

### 35.8. Hardening final: taxonomy, graph e constraints antes do planner

`READ_TOOLS` foi removido. A fonte central de classificação é
`tools.effects.ToolEffect`: PURE_READ, DISCOVERY, IDEMPOTENT_WRITE, MUTATION,
INTERACTIVE e COGNITIVE. O registry aceita metadata de efeito, contrato de chave
idempotente e rotas; unknown é MUTATION. MCP captura readOnlyHint em discovery
e cache, sem dispensar sua autorização/trust gate. Discovery de board/lista/cards
pode ocorrer antes de `work_execute`, inclusive junto a writes rejeitados na
mesma resposta. Clarificação humana é permitida; delegação cognitiva não substitui
compilação. IDEMPOTENT_WRITE exige contrato e chave presente; não autoriza replay
de mutação incerta.

Setup e finalize são WorkItems compartilhados do mesmo WorkPlan. IDs, depends_on
e dependências de bindings são validados antes da execução; faltas, ciclos e
dependências de uma fase futura falham antes do dispatch. A ordenação de nós
prontos é determinística. `$setup.node.campo` carrega resultado confirmado do
setup; `$steps.node.campo` resolve passos locais; `$items_ref` entrega ao finalize
um manifest, sem despejar outputs no transcript. O finalize só avança quando
setup e todos os itens necessários estão completos. Checkpoints de passo e
proteção de mutação incerta valem também para fases compartilhadas.

O ledger reconstrói setup/items/finalize, progresso de passos, fase, failed,
pending, uncertain, blockers e artifact refs a partir da DB e metadata do plano.
A conexão do DurableTaskStore criado pela tool é fechada após a chamada externa;
isso também permite apagar fixtures temporários no Windows sem handle SQLite
pendurado. O formato anterior `items + steps` conserva sua execução linear.

Detecção inicial usa sinais gramaticais/coleção/contagem e records homogêneos;
detecção runtime compara nome e shape dos argumentos, preservando discriminadores
de operação. A terceira mutação distinta equivalente é bloqueada antes de
executar. Resultados já executados ficam em artifacts e são adotados pelo plano
apenas quando seu verifier confirma o output anterior. O controle é por turno;
não é deduplicação global de execução nem outro banco de tarefas.

TurnConstraintContext é criado antes da primeira chamada do planner. Seleção
principal, fallback, boundaries streaming/non-streaming, resolução/cache auxiliary
e ModelRouter usam constraints; fallback proibido é eliminado antes de sondagem
de credencial/client. ContextVar é restaurada ao encerrar o turno, para não
contaminar outras sessões. Rotas auxiliary auto/composite sob constraints exigem
provider explícito permitido: destinos internos desconhecidos falham fechado.
Tool/compiler/browser routing e metadata do WorkPlan recebem a mesma restrição.

Guardrails foram comparados com upstream `4716ec0ba4` e adaptados pontualmente:
failure-tolerant names, progress-reset candidates e opção de hard stop para
plataformas não interativas. Sucesso textual de terminal/browser não prova
progresso; write landed, verifier confiável ou checkpoint confirmado inicia novo
experimento. O campo textual `actual_delta=true` não possui autoridade.

`test_durable_hardening.py` cobre os contratos A–M. O replay ACIRV/Trello-like
usa fake provider e adapter remoto, mantendo dispatch AIAgent e estado durável
reais: 12 criações, 2 provider calls, 3 setup calls, zero replay de mutações e
21 cache hits. Os bytes inline medidos são aproximadamente 2,7 KB contra
2,56 MB no baseline de transporte modelado. Isso não mede tokens pagos. O CI
instala a configuração dev do projeto via uv.lock, incluindo as dependências
normais e pytest-asyncio; a evidência de GitHub deve ser registrada no journal.

### 35.9 HW-022 — Canary, recipes verificadas e continuidade durável

O primeiro WorkItem real de um batch com mutações por item funciona como canary,
sem criar entidade extra. Cada mutação precisa de read/discovery posterior com
`verifies=[mutation_id]` e `expect` sobre estado persistido. `success=true`, clique
concluído ou texto afirmando progresso não substituem esse vínculo. `browser_console`
executa JavaScript arbitrário e permanece MUTATION, portanto não serve como verifier
read. Falha no canary interrompe fan-out: o restante fica pending, não failed.
Checkpoint confirmado não se repete; dispatch sem confirmação continua exigindo
`uncertain_mutation_requires_review`. Setup/finalize mantêm os checkpoints e barreiras
do graph existente. A sintaxe linear permanece, com prova mais forte para batches
mutáveis. Uma leitura bem-sucedida sem `verifies` não reseta o guardrail.

RecipeStore não é banco de tarefas: corpos versionados ficam no ArtifactStore e um
índice JSON contém refs, SHA, fingerprint e VERIFIED/STALE/QUARANTINED. Lock de SO
(msvcrt/fcntl), fsync e replace atômico protegem escrita entre processos. Promoção
depende do graph completo externamente verificado; status fornecido pelo planner
não concede autoridade. Sanitização recursiva remove credenciais, cookies, estado
de sessão/storage e raciocínio privado, inclusive argumentos JSON codificados.
Fingerprint considera graph, schemas/effects/routes, família runtime, scope e
preflight; IDs de cards/timestamps não caracterizam a receita. Reuso exige preflight
read-only limitado a oito probes com expectativas. Schema/scope incompatível ou
preflight falho bloqueia mutações e marca STALE. Falha inesperada de verificação
quarentena o procedimento. Native browser exige host/path_family e evidência da URL
real; binding para outro host não pode escapar ao scope.

`continuation.py` reconstrói handoff do ledger persistido, sem chain-of-thought:
phase, progresso, constraints, canary/recipe, blockers, refs e next_action. O caminho
normal cabe em 4 KB; constraints grandes referenciam sua fonte persistida para não
truncar política. Ownership e referência operacional autenticada pelo runtime são
obrigatórios. Não confiar em labels `plan_id`/`artifact://` escritos no transcript.
Projeção ao provider opera sobre cópia, preserva system byte-estável e último turno
humano real, remove histórico operacional já coberto e mantém tool pairs ativos
não relacionados. SessionDB continua contendo histórico completo. Um plano concluído
antes do último turno humano não reduz uma nova conversa comum. Compaction automática
usa handoff com dedupe por hash; force/manual, commit fence ou ausência de estado
confiável seguem o compressor geral. Não rotacionar sessão nem injetar user sintético.

`work_execute(action="contract")` explica bindings, graph, canary, recipes e resume
sem dispatch; erros do compiler incluem code/fix acionáveis. ExperimentKey ignora
valores específicos das entidades, mas retém hipótese operacional: tool, seletor,
comando, operação, rota, ordem/multiplicidade de steps e predicate do verifier.
Geração de progresso muda apenas com evidência confiável. Registry/effects é a
taxonomia canônica também para reads instrucionais; desconhecidas ficam conservadoras.
TurnConstraintContext já tinha lifecycle correto e testes de dois turnos provam
reset e propagação auxiliar, sem reimplementar sua seleção.

Replay controlado usa AIAgent/dispatcher/persistência reais, provider e adapter
Trello-like fake, sem API real: sete cenários invalid/corrected/restart-hit/stale/
invalid-after-stale/corrected-v2/known-v2. Correto: 12 entidades, 53 tools (inclui
um novo preflight), 3 setup, 2 provider calls, zero LLM no executor e zero replay.
Inválido expõe uma mutação; stale expõe zero. Handoff correto 1439 bytes e wire
final ao provider 1711 bytes, mantendo 1289737 bytes de artifacts. Tokens cached/
uncached e custo são null quando não reportados; não inferir economia paga dos
bytes ou do baseline modelado. Evidência GitHub/SHAs está no journal H-062.

## Canonical Work Loop current implementation (2026-09-17)

[Canonical Work Loop implementation evidence](CANONICAL_WORK_LOOP.md) records
verified completion, intent authority, live handles vs durable proof, lineage,
additive migrations and exact pending integrations. Prior descriptions of
unconditional report completion and automatic critical-event promotion are
superseded by these fail-closed contracts. The complete program remains active.

## Canonical Execution Reliability Gate — Implementação e Invariantes (2026-09-17)

O milestone canônico de confiabilidade de execução fecha o ciclo causal entre intenção, execução e evidência com as seguintes garantias provadas em código e testes:

1. **Linhagem Causal Completa:**
   - Todo trabalho mutável transita com quádrupla canônica: `task_id`, `run_id`, `execution_key` e `operation_id`.
   - `WorkPlan` e `WorkItem` persistem `run_id`, `execution_key` e `operation_id` com migrações SQLite aditivas e compatíveis.
   - `ExecutionEvent`, `BrowserTaskReport` e `TaskOutcome` propagam a linhagem completa.

2. **Commit Canônico com Fencing de Run:**
   - `workstation/kanban.py::complete_task_with_report` exige e valida `expected_run_id` via CAS em `tasks.current_run_id`.
   - Runs obsoletos/abandonados que tentam relatar ou completar pós-takeover/reclaim são rejeitados imediatamente antes de avaliar aceitação ou mutar o estado da tarefa.
   - O evento `TASK_COMPLETED` só é gravado no `ExecutionJournal` APÓS o commit canônico bem-sucedido na base de dados. Falha de CAS emite `acceptance_commit_failed` e retorna `False`.

3. **Reconciliação de Árvore Terminal:**
   - Transição de planos pais para estados terminais (`interrupted`, `failed`, `cancelled`, `blocked`) bloqueia em cascata todos os itens filhos ativos (`running`, `pending`, `ready`, `claimed`, `retrying`).
   - `DurableTaskStore.reconcile_terminal_plans()` executa reconciliação de inicialização e varreduras periódicas.

4. **Fencing em Human Takeover:**
   - `BrowserControlLeaseManager` gera gerações monotônicas e `fence_token`s únicos.
   - O takeover humano revoga imediatamente a autoridade residual de mutação do agente (`browser_click`, `browser_type`, `navigate`, `submit`, etc.) e invalida tokens de fence de runs anteriores.
   - Retomada do agente emite nova geração e novo token de autorização.

5. **Journal Hash Chaining Streaming $O(1)$:**
   - `ExecutionJournal.append()` utiliza `_get_last_record()` para inspecionar o último registro sem reler o arquivo completo, garantindo tempo constante $O(1)$ mesmo em tarefas longas com mais de 100 eventos.
   - Teste de stress de 120 eventos comprova ausência de degradação e integridade criptográfica da cadeia.

6. **Auditabilidade Total no Cockpit:**
   - `task_cockpit()` projeta linhagem canônica ponta a ponta: `task_id`, `run_id`, `execution_key`, `operation_id`, `workplan_id`, `human_card_id`, `acceptance_status` e `acceptance_approved`.
# Read-only durable bootstrap (2026-09-17)

`work_execute action=discover` prepares owner-scoped evidence before a graph is
known; `browser_extract_items mode=inspect` provides structured isolated-world
DOM reads. Arbitrary terminal/JS remain write-capable. Compile refusals preserve
the read dispatcher. See [the caller protocol](READONLY_DURABLE_PREFLIGHT.md).
Successful read capability does not establish write capability. Fan-out is
authorized by verified persistence of a representative canary, not by successful
execution of a preparation step. No configuration switch or threshold change.
