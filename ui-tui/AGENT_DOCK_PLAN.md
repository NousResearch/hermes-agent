# Plan: TUI Agent Dock — klickbare Live-Subagent-Leiste (Claude-Code-Style)

**Branch:** `feat/tui-agent-dock` (getrennt von PR #72610/#65009/#65003)
**Autor:** Simeon (RFI-IRFOS) + Hermes
**Ziel:** Immer-sichtbare, klickbare Agent-Liste direkt unterm Composer.
Kein `/agents` tippen mehr nötig. Bis zu 4 Top-Level-Subagenten als eigene
Spalten/Listen, mit Laufzeit + Token-Tiefe + Tool-Tiefe. Klick → Detailansicht
des Agents. ESC → zurück. Auto-Close 15 s nachdem der letzte Agent fertig ist.

---

## 1. Kontext — was schon da ist (nicht neu erfinden)

- `src/lib/subagentTree.ts` — `buildSubagentTree(items)` → `SubagentNode[]`.
  Jeder Node hat `item` (`SubagentProgress`) + `aggregate` (subtree rollup).
  `flattenTree()`, `topLevelSubagents()`, `treeTotals()` existieren schon.
- `src/types.ts` — `SubagentProgress` hat: `id, goal, status, depth, index,
  parentId, toolCount, durationSeconds, startedAt, inputTokens, outputTokens,
  reasoningTokens, model, iteration, apiCalls, children?` + `aggregate`:
  `totalTools, totalDuration, descendantCount, activeCount, maxDepthFromHere,
  inputTokens, outputTokens`.
- `src/app/delegationStore.ts` — `$delegationState` (live, fließt alle paar s).
- `src/app/turnStore.ts` — `subagents: SubagentProgress[]` (live Liste).
- `src/components/agentsOverlay.tsx` — voller spawn-tree Dashboard.
  `Detail({node})` zeigt Budget/Files/Tool-calls/Output. `ListRow`,
  `OverlaySection` (schon mit `onClick`), `patchOverlayState({agents:true})`.
- `src/app/overlayStore.ts` — `patchOverlayState()` API (schon da).
- `src/app/useMainApp.ts` — zentraler Layout-Renderer, Composer sitzt hier.

**Erkenntnis:** Alle Daten + die Detail-Render-Logik existieren. Wir bauen
KEINEN neuen Tree, KEINE neue Datenquelle — wir rendern die gleichen
`SubagentNode`s nur in einer neuen, nicht-modalen Leiste unterm Composer.

---

## 2. Neue Komponente: `src/components/agentDock.tsx`

Immer-sichtbar (nicht-modal, blockt Input NICHT). Rendert nur wenn
`subagents.length > 0` ODER ein Agent in den letzten 15 s fertig wurde.

### Layout (bis zu 4 Spalten)
```
┌─ Agent 1 ───────┐ ┌─ Agent 2 ───────┐ ┌─ Agent 3 ───────┐ ┌─ Agent 4 ───────┐
│ ● goal preview  │ │ ○ goal preview  │ │ ✓ goal preview  │ │ ⚠ goal preview  │
│ 12s · 3.2k tok  │ │ 4s  · 1.1k tok  │ │ 47s · 8.9k tok  │ │ 9s  · 0.4k tok  │
│ depth 2 · 5t    │ │ depth 1 · 2t    │ │ depth 3 · 12t   │ │ depth 1 · 1t    │
└────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
```
- **Spalten = `topLevelSubagents(tree)`** (max 4, wie vom User gewünscht —
  `delegation.maxConcurrentChildren` cap ist 4, passt).
- Pro Spalte: Status-Glyph (running ○ / completed ✓ / error ⚠), Goal-Preview
  (`compactPreview`, schon in agentsOverlay), Laufzeit (`fmtDuration`),
  **Token-Tiefe** (`fmtTokens(inputTokens+outputTokens)` aus `aggregate`),
  **Tool-Tiefe** (`aggregate.totalTools` + `maxDepthFromHere`).
- Aktiver Agent (der gerade geklickte) wird `inverse`/accent highlighted.

### Interaktion
- **Klick auf Spalte** → `patchOverlayState({ agents: true, agentsInitialHistoryIndex: <index> })`
  → öffnet den existierenden `agentsOverlay` gefiltert auf diesen Agent.
  (Reuse, kein eigener Detail-Renderer nötig — spart Code + hält Tests grün.)
- **ESC** (im Dock-Kontext) → `patchOverlayState({ agents: false })` bzw.
  zurück zur Dock-Ansicht (kein Full-Overlay mehr).
- **Auto-Close:** Sobald `activeCount === 0` (alle fertig), startet ein 15-s-
  Timer. Nach 15 s → Dock ausblenden (`dockVisible = false`). Jeder neue
  Agent (status wechselt zu running) resetet den Timer + zeigt Dock wieder.

### State-Management (lokal in Komponente)
```ts
const subagents = useStore($turnStore).subagents   // live
const tree = useMemo(() => buildSubagentTree(subagents), [subagents])
const tops = useMemo(() => topLevelSubagents(tree).slice(0, 4), [tree])
const [dockVisible, setDockVisible] = useState(false)
const [autoCloseAt, setAutoCloseAt] = useState<number | null>(null)

// Effekt: wenn activeCount>0 → sichtbar + timer nullen
// wenn activeCount===0 && dockVisible → autoCloseAt = now+15000
// Intervall (1s) prüft: now >= autoCloseAt → setDockVisible(false)
```
Timer via `setInterval` in `useEffect` (cleanup bei unmount). Kein neuer
Store nötig — lokal reicht, da der Dock nur UI-State spiegelt.

---

## 3. Mount in `useMainApp.ts`

Direkt UNTER dem Composer rendern (gleiche Box wie der Composer, `marginTop`):
```tsx
<Composer ... />          {/* existing */}
<AgentDock />            {/* NEW — nicht-modal, kein useInput-Block */}
```
Wichtig: `AgentDock` darf `useInput` NICHT global blockieren (im Gegensatz
zum modalen Overlay). Klick-Handling via Ink `onClick` (wie `OverlaySection`
schon macht) — das funktioniert ohne Input-Lock.

---

## 4. Änderungen im Detail

| File | Änderung |
|------|----------|
| `src/components/agentDock.tsx` | NEU — die Leiste (siehe §2) |
| `src/app/useMainApp.ts` | `<AgentDock />` unter Composer mounten (~Zeile 497, wo Composer gerendert wird) |
| `src/lib/subagentTree.ts` | KEINE (topLevelSubagents/flattenTree schon da) |
| `src/app/overlayStore.ts` | KEINE (patchOverlayState schon da) |
| `src/components/agentsOverlay.tsx` | KEINE Änderung — wir rufen ihn nur auf |

**Prompt-Cache:** Nur lesend (kein System-Prompt-Touch) → sicher.
**Tests:** `tests/tui/` schon da (vitest + Ink Testing Library). Neuer Test:
`agentDock.test.tsx` — rendert 4 Spalten bei 4 Top-Level-Agents, Klick öffnet
Overlay, Auto-Close nach 15 s (mit fake timers).

---

## 5. Risiken / Hermes-Rubric-Check

- ✅ **Narrow waist:** Kein neues Core-Tool, nur UI-Komponente (Edges).
- ✅ **Cache-safe:** Kein System-Prompt-Mutate, keine Toolset-Änderung.
- ✅ **Extend not duplicate:** Reuse `buildSubagentTree` + `agentsOverlay.Detail`.
- ⚠️ **Input-Block:** Muss sicherstellen dass `AgentDock` nicht den Composer
  blockiert (kein `useInput` capture außer bei Klick). Test deckt das ab.
- ⚠️ **Auto-Close-Timing:** 15 s via `Date.now()` + `setInterval`, nicht via
  setTimeout-Stack (sonst reset-Probleme bei neuen Agents).

---

## 6. Commit- + PR-Strategie

Branch `feat/tui-agent-dock` (von `main` abgezweigt, NICHT von dei 3 PRs):
1. `feat(tui): add non-modal agentDock under composer` — Komponente + Mount
2. `test(tui): agentDock renders up to 4 agents, click opens overlay, auto-closes 15s`

Danach PR zum offiziellen Nous-Repo (wie vom User gewünscht).
Deine 3 bestehenden PRs (#72610/#65009/#65003) bleiben unangetastet.

---

## 7. Offene Fragen (vom User zu klären)

- [ ] **4 Spalten hart-capen** oder scrollbar bei >4? (User sagt "bis zu 4" →
      wir capen auf 4, Rest im `/agents` Overlay sichtbar)
- [ ] **ESC-Verhalten:** ESC schließt Dock komplett, oder nur den
      Detail-Overlay zurück zur Dock-Ansicht? (Plan: ESC = zurück zur Dock)
- [ ] **Auto-Close bei aktiven Agents pausiert?** (Plan: pausierte Agents
      zählen als nicht-running → Timer läuft)
