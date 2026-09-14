# PLAN / DESIGN: V3 Runtime Hardening Foundation

## 1. Spec de origem

`spec.md`, V3.1 e V3.2 em `workstation/ROADMAP.md`.

## 2. Ownership

| Estado | Owner | Lifetime | Recovery |
|---|---|---|---|
| EvidenceState | `runtime.py` projection | task/run | stale -> stalled |
| execution events | `RuntimeEventBus` + `ExecutionJournal` | process/task | bounded queue + journal |
| procedures/routines | `ProceduralMemory` | durable memory | versioned validation |
| worker lineage | `WorkerRegistry` | parent/task | persistent record + stop |
| runtime liveness | `RuntimeSupervisor` | supervisor process | restart/checkpoint |
| optional component health | `RecoveryPlane` | control plane | quarantine/restore |

## 3. Sequência

1. Adicionar contratos e testes RED para runtime/evidence/events/resources.
2. Implementar supervisor/recovery e rotina determinística sobre os owners
   existentes.
3. Estender `WorkerRegistry` para queue/message/steer/wait/stop.
4. Adicionar memória temporal, replay portátil e budget/model routing.
5. Rodar regressões Workstation e atualizar estado/journal com evidência real.

## 4. Segurança e rollback

Todos os artefatos usam paths fornecidos pelo chamador ou `HERMES_HOME`,
escrita temporária + replace e payloads estruturais. O rollback é remover a
projeção nova e manter o journal/memory anterior intacto; não há migração
destrutiva de dados existentes.

## 5. Limites de evidência

Testes unit/contract provam os contratos Python. Eles não provam Electron,
Windows, multi-processo real ou resistência a soak; esses claims só serão
promovidos após probes próprios.
