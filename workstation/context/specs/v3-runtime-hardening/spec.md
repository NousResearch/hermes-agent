# SPEC: V3 Runtime Hardening Foundation

Status: VALIDATED

## 1. Objetivo

Implementar a primeira fundação executável para V3.1 e V3.2: estado de execução
baseado em evidência, eventos com cancelamento/backpressure, supervisão e
recovery independentes, promoção/replay determinístico de procedimentos e
workers persistentes com lineage canônico.

## 2. Estado atual relevante

- `ExecutionJournal` já é o registro append-only de ações e evidências.
- `ProceduralMemory` já é o owner de procedimentos web persistentes.
- `WorkerRegistry` já preserva a linhagem Hermes de delegações descartáveis.
- `MultiTaskScheduler` já protege o limite de um host nativo ativo.
- Não existe ainda um estado operacional que impeça `running` zumbi, nem
  supervisor/recovery fora do runtime.

## 3. Escopo

Incluído: contratos Python puros e persistência atomicamente recuperável para
EvidenceState, RuntimeEventBus, typed resources, human handoff, budgets/model
routing, supervisor/recovery, rotina promovida e worker persistente.

Fora de escopo nesta fatia: substituir o loop de agente Hermes, criar uma nova
SessionDB/Kanban/Memory, alterar o schema de ferramentas do modelo, ou alegar
validação nativa Windows/Electron sem executar essa fronteira.

## 4. Requisitos e critérios

### REQ-001 — Running exige evidência viva

`running` deve degradar para `stalled` quando não houver evidência válida ou
quando a evidência expirar; estados de bloqueio, aprovação e falha não podem
ser reportados como conclusão normal.

### REQ-002 — Eventos não bloqueiam consumidores

Um consumidor lento não pode impedir publicação para consumidores independentes;
cancelamento e deadline devem produzir resultados observáveis.

### REQ-003 — Recovery é independente

Um processo supervisor consegue iniciar, health-checkar, reiniciar e restaurar
um checkpoint sem depender do objeto do runtime que supervisiona.

### REQ-004 — Rotinas só executam após promoção

Uma rotina descoberta deve passar por validação explícita, replay deve falhar
closed em precondição/anchor/postcondição inválida e devolver controle ao agente.

### REQ-005 — Workers persistentes têm ciclo controlável

Um worker pode receber múltiplas mensagens, ser steerado, aguardado, pausado ou
cancelado, mantendo parent/task/session lineage e envelope de resultado.

### REQ-006 — Estado compartilhável é tipado

Recursos expostos a clientes carregam identidade, lineage, permissões e estado
operacional sem expor scratch privado ou secrets.

## 5. Invariantes

- Hermes continua owner de sessões, Kanban e Memory.
- BrowserTask continua owner de página viva; nenhum módulo novo cria page store.
- `running` sem evidência viva nunca é um estado terminal de sucesso.
- Bound tasks continuam fail-closed.
- Persistência usa escrita atômica e não grava credenciais/payloads sensíveis.

## 6. Evidência mínima

Testes Python de contrato para cada estado/transição, fault injection de
persistência, consumidores independentes, rotina em drift e ciclo persistente
de worker. Validação nativa e soak prolongado permanecem gates posteriores até
serem executados no ambiente correspondente.
