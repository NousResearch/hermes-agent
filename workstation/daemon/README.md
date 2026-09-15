# V4.0 — Headless Browser Daemon Architecture & Decoupling Specification

> **Status:** Protótipo / Especificação Técnica (Fase 1 da V4.0)
> **Objetivo:** Desacoplar o controller e o runtime do navegador do ciclo de vida da janela do Electron Main Process, garantindo que o agente e suas tarefas web continuem operando mesmo se o Desktop for fechado, minimizado, recarregado ou travar.

---

## 1. Problema e Motivação

Na V3.5 (arquitetura atual):
- O runtime do Chromium é hospedado dentro do processo Electron Main (`WebContentsView`).
- Se o usuário fechar a janela do Electron, se houver um crash de GPU/renderer, ou se o processo da UI for reiniciado, o runtime de automação do browser morre instantaneamente.
- Qualquer tarefa longa em andamento sofre desconexão de socket (`WinError 10061`).

## 2. Visão da Arquitetura V4.0

```
┌────────────────────────────────────────────────────────┐
│             Hermes Agent / Python Runtime              │
│       (AIAgent Loop + DurableBatchRunner + Tools)       │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP / WebSocket Loopback
                           ▼
┌────────────────────────────────────────────────────────┐
│           Hermes Headless Browser Daemon               │
│       (Processo Sidecar Independente de Background)    │
│  - Loopback API: /health, /v1/action, /v1/resources   │
│  - Session & Tab State Authority                       │
│  - Chromium Automation (CDP / Headless Engine)         │
│  - Viewport Stream / Attachment Token Provider         │
└──────────────────────────▲─────────────────────────────┘
                           │ Viewport Attach / IPC
┌──────────────────────────┴─────────────────────────────┐
│             Hermes Desktop (Electron UI)               │
│         (Cliente de Apresentação / Visualizador)       │
│  - Não hospeda o cérebro da automação                  │
│  - Conecta na visualização das abas existentes         │
│  - Pode ser aberto/fechado sem interromper automações  │
└────────────────────────────────────────────────────────┘
```

## 3. Contrato do Daemon (`browser_daemon.py`)

1. **Ciclo de Vida Independente:**
   O daemon é iniciado como um processo de segundo plano no sistema operacional.
2. **Descritor de Controle Local:**
   Grava `%HERMES_HOME%/workstation/daemon-control.json` contendo `url`, `token` (bearer secreto), `pid` e `version`.
3. **Persistência de Abas (Tab Retention):**
   Abas alocadas para tarefas duráveis (`taskId`) sobrevivem a desconexões e reconexões de clientes.
4. **Fixação e Desanexação de Viewport (Viewport Attachment):**
   Quando o Electron Desktop é aberto, ele solicita um token de viewport para a aba ativa e passa a projetar os frames; ao ser fechado, o daemon continua executando a aba em modo headless.
