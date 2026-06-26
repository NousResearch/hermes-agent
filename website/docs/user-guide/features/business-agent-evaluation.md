# Business agent evaluation

Use this pattern when Hermes is deployed as a multi-agent business operating system: route each class of work to the smallest capable agent, verify outputs before handoff, and keep durable knowledge in skills/docs rather than hot memory.

## Reference architecture

```mermaid
flowchart LR
    Operator["Business operator"] --> Gateway["Hermes Gateway\nTelegram / CLI / TUI"]
    Gateway --> Router["Task router\nprofiles + skills + toolsets"]

    Router --> Web["web-scraper\nsource discovery + extraction"]
    Router --> Code["code-review\nquality + security"]
    Router --> Design["design-expert\nUX + brand"]
    Router --> Finance["financial-guru\nmarket data + earnings"]
    Router --> Risk["risk-reward\nEV + sizing"]
    Router --> Devil["devils-advocate\nthesis stress test"]

    Web --> Verify["Verification gate\nprimary sources + tests"]
    Code --> Verify
    Design --> Verify
    Finance --> Verify
    Risk --> Verify
    Devil --> Verify

    Verify --> Docs["Docs / skills / indexes"]
    Verify --> Delivery["Decision memo / PR / dashboard"]
```

## Agent-by-application matrix

| Business application | Primary agent | Input artifacts | Acceptance criteria | KPIs |
|---|---|---|---|---|
| Market map or source gathering | `web-scraper` | URLs, search terms, company names, filings | Primary sources captured; fallbacks labeled; extraction reproducible | source coverage, citation quality, freshness |
| Financial screen or earnings analysis | `financial-guru` | local DuckDB, filings, price data, earnings calendars | Data freshness verified; calculations tested; assumptions labeled | hit-rate, data latency, false positives |
| Position sizing / trade structuring | `risk-reward` | thesis, downside bounds, volatility/liquidity, constraints | downside bounded; upside/downside ratio explicit; capacity addressed | expected value, max loss, Kelly fraction, drawdown |
| Investment thesis red-team | `devils-advocate` | memo, model, key assumptions, catalysts | strongest counter-case written; kill criteria identified | assumption fragility, base-rate fit, missing evidence count |
| Codebase or automation change | `code-review` | branch diff, tests, logs, threat model | tests pass; security regressions absent; dead code removed | test coverage, defect density, review findings |
| Dashboard / product experience | `design-expert` | screenshots, flows, target users, brand constraints | visual hierarchy clear; mobile path tested; accessibility issues listed | task completion, readability, Core Web Vitals |
| Cross-functional production launch | default Hermes orchestrator | project docs, repo, crons, deploy targets | specialists run in parallel; outputs verified; docs/indexes rebuilt | cycle time, escaped defects, manual interventions |

## Production evaluation loop

```mermaid
sequenceDiagram
    participant O as Operator
    participant H as Hermes orchestrator
    participant S as Specialist agents
    participant V as Verification
    participant G as GitHub / Docs

    O->>H: Ask for business outcome
    H->>H: Classify task + choose smallest capable agent
    H->>S: Dispatch isolated subagents with acceptance criteria
    S-->>H: Return findings + evidence handles
    H->>V: Verify files, tests, sources, dashboards, crons
    alt verified
        V-->>H: Pass
        H->>G: Commit docs/code to branch and rebuild indexes
        H-->>O: Deliver decision-ready summary
    else failed
        V-->>H: Fail with evidence
        H->>S: Retry narrower task or escalate to operator
    end
```

## Production readiness checklist

- [ ] Route work by domain instead of asking one agent to do everything.
- [ ] Give subagents full context and explicit acceptance criteria.
- [ ] Verify subagent outputs with real file/source/test handles before reporting success.
- [ ] Use hooks for silent maintenance and `hermes hooks doctor` for health checks.
- [ ] Rebuild project registry and indexes after structural changes.
- [ ] Commit code/docs to a named branch; push to an accessible remote or report the permission blocker.
- [ ] Move recurring procedures into skills; move durable long-form knowledge into docs.

## Example routing prompt

```text
Context: Evaluate whether this earnings-screening pipeline is production ready.
Agents:
- financial-guru: validate data freshness and factor math.
- code-review: audit pipeline code and tests.
- devils-advocate: identify false-positive and stale-data failure modes.
- design-expert: review dashboard UX.
- risk-reward: evaluate whether outputs support asymmetric trade selection.
Done when: every agent returns evidence, Hermes verifies it, docs/indexes are rebuilt, and the branch is pushed.
```
