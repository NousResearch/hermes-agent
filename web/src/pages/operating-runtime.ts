import {
  businessScorecards,
  decisionLedger,
  liveSignalIntegrations,
  modelRoutingPolicies,
  operatingLoops,
  permissionPolicies,
  routedTasks,
  type OperatingSystemStage,
} from "./operating-system-data";
import { fetchJSON } from "@/lib/api";

export type RuntimeEvidenceKind =
  | "snapshot"
  | "memory"
  | "permission"
  | "model"
  | "loop"
  | "business"
  | "workbench"
  | "quality"
  | "autonomy"
  | "registry"
  | "telemetry"
  | "incident"
  | "deployment"
  | "secrets"
  | "catalog"
  | "finance"
  | "learning"
  | "eval"
  | "executive";
export type RuntimeEvidenceState = "ready" | "stored" | "allowed" | "blocked" | "gated" | "warning" | "failed";

export interface RuntimeEvidenceRecord {
  id: string;
  kind: RuntimeEvidenceKind;
  subject: string;
  state: RuntimeEvidenceState;
  owner: string;
  detail: string;
  updatedAt: string;
}

export interface RuntimeAuditRecord {
  id: string;
  action: string;
  actor: string;
  allowed: boolean;
  approval: "none" | "confirm" | "explicit";
  reason: string;
  createdAt: string;
}

export interface OperatingRuntimeState {
  evidence: RuntimeEvidenceRecord[];
  audit: RuntimeAuditRecord[];
}

interface ServerEvidenceRecord {
  id: string;
  kind: RuntimeEvidenceKind;
  subject: string;
  state: RuntimeEvidenceState;
  owner: string;
  detail: string;
  updated_at?: string;
  updatedAt?: string;
}

interface ServerAuditRecord {
  id: string;
  action: string;
  actor: string;
  allowed: boolean;
  approval: "none" | "confirm" | "explicit";
  reason: string;
  created_at?: string;
  createdAt?: string;
}

interface ServerReadinessResponse {
  decision: {
    allowed: boolean;
    approval: "none" | "confirm" | "explicit";
    reason: string;
  };
  audit: ServerAuditRecord;
  evidence: ServerEvidenceRecord;
}

const STORAGE_KEY = "hermes.operatingRuntime.v1";

export function loadOperatingRuntimeState(): OperatingRuntimeState {
  const seeded = seedRuntimeState();
  if (typeof window === "undefined") return seeded;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return seeded;
    const parsed = JSON.parse(raw) as OperatingRuntimeState;
    return {
      evidence: mergeById(seeded.evidence, parsed.evidence ?? []),
      audit: mergeById(seeded.audit, parsed.audit ?? []),
    };
  } catch {
    return seeded;
  }
}

export function saveOperatingRuntimeState(state: OperatingRuntimeState) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export async function loadOperatingRuntimeStateFromServer(): Promise<OperatingRuntimeState> {
  const [evidenceResponse, auditResponse] = await Promise.all([
    fetchJSON<{ evidence: ServerEvidenceRecord[] }>("/api/operating-runtime/evidence"),
    fetchJSON<{ audit: ServerAuditRecord[] }>("/api/operating-runtime/audit?limit=50"),
  ]);
  const seeded = seedRuntimeState();
  const next = {
    evidence: mergeById(seeded.evidence, evidenceResponse.evidence.map(normalizeEvidence)),
    audit: mergeById(seeded.audit, auditResponse.audit.map(normalizeAudit)),
  };
  saveOperatingRuntimeState(next);
  return next;
}

export function recordsForStage(stage: OperatingSystemStage, state: OperatingRuntimeState) {
  return state.evidence.filter((record) => kindForStage(stage.version).includes(record.kind));
}

export function runRuntimeReadinessCheck(stage: OperatingSystemStage, state: OperatingRuntimeState): OperatingRuntimeState {
  const now = new Date().toISOString();
  const action = actionForStage(stage.version);
  const permission = permissionForAction(action);
  const allowed = permission.approval !== "explicit" && stage.risk !== "high";
  const audit: RuntimeAuditRecord = {
    id: `audit-${stage.version.toLowerCase()}-${Date.now()}`,
    action,
    actor: "Hermes operator",
    allowed,
    approval: permission.approval,
    reason: allowed
      ? "Readiness check allowed because it is read-only or confirm-level."
      : "Readiness check recorded, but live execution remains gated by explicit approval, audit persistence, and kill-switch controls.",
    createdAt: now,
  };
  const evidence: RuntimeEvidenceRecord = {
    id: `evidence-${stage.version.toLowerCase()}-${Date.now()}`,
    kind: kindForStage(stage.version)[0],
    subject: `${stage.version} readiness check`,
    state: allowed ? "ready" : "gated",
    owner: stage.owner,
    detail: audit.reason,
    updatedAt: now,
  };
  const next = {
    evidence: mergeById(state.evidence, [evidence]),
    audit: [audit, ...state.audit].slice(0, 30),
  };
  saveOperatingRuntimeState(next);
  return next;
}

export async function runServerRuntimeReadinessCheck(
  stage: OperatingSystemStage,
  state: OperatingRuntimeState,
): Promise<OperatingRuntimeState> {
  const response = await fetchJSON<ServerReadinessResponse>("/api/operating-runtime/readiness-check", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      stage: stage.version,
      action: actionForStage(stage.version),
      actor: "Hermes operator",
      actor_role: "operator",
      explicit_approval: false,
    }),
  });
  const audit = normalizeAudit(response.audit);
  const evidence = normalizeEvidence(response.evidence);
  const next = {
    evidence: mergeById(state.evidence, [evidence]),
    audit: [audit, ...state.audit.filter((record) => record.id !== audit.id)].slice(0, 50),
  };
  saveOperatingRuntimeState(next);
  return next;
}

export function runtimeSummaryForStage(stage: OperatingSystemStage, state: OperatingRuntimeState) {
  const records = recordsForStage(stage, state);
  const ready = records.filter((record) => ["ready", "stored", "allowed"].includes(record.state)).length;
  const gated = records.filter((record) => ["gated", "blocked", "warning", "failed"].includes(record.state)).length;
  const audited = state.audit.filter((record) => actionForStage(stage.version) === record.action).length;
  return { total: records.length, ready, gated, audited };
}

function seedRuntimeState(): OperatingRuntimeState {
  const now = "2026-07-17T00:00:00.000Z";
  const evidence: RuntimeEvidenceRecord[] = [
    ...liveSignalIntegrations.map((item) => ({
      id: `snapshot-${item.id}`,
      kind: "snapshot" as const,
      subject: item.project,
      state: item.status === "ready" ? "ready" as const : "gated" as const,
      owner: item.project,
      detail: `${item.signals.join(", ")} via ${item.endpoint}`,
      updatedAt: now,
    })),
    ...decisionLedger.map((item) => ({
      id: `memory-${item.id}`,
      kind: "memory" as const,
      subject: item.decision,
      state: "stored" as const,
      owner: item.owner,
      detail: item.reason,
      updatedAt: item.reviewedAt,
    })),
    ...routedTasks.map((item) => ({
      id: `memory-${item.id}`,
      kind: "memory" as const,
      subject: item.title,
      state: item.status === "blocked" ? "blocked" as const : "stored" as const,
      owner: item.owner,
      detail: item.nextStep,
      updatedAt: now,
    })),
    ...permissionPolicies.map((item) => ({
      id: `permission-${item.id}`,
      kind: "permission" as const,
      subject: item.action,
      state: item.approval === "explicit" ? "gated" as const : "allowed" as const,
      owner: item.level,
      detail: `${item.approval} approval; audit ${item.audit ? "required" : "not required"}`,
      updatedAt: now,
    })),
    ...modelRoutingPolicies.map((item) => ({
      id: `model-${item.id}`,
      kind: "model" as const,
      subject: item.taskType,
      state: item.approvalRequired ? "gated" as const : "ready" as const,
      owner: item.preferred,
      detail: `${item.costMode}; fallback ${item.fallback}`,
      updatedAt: now,
    })),
    ...operatingLoops.map((item) => ({
      id: `loop-${item.id}`,
      kind: "loop" as const,
      subject: item.name,
      state: item.status === "ready" ? "ready" as const : "gated" as const,
      owner: item.owner,
      detail: `${item.cadence}; ${item.output}`,
      updatedAt: now,
    })),
    ...businessScorecards.map((item) => ({
      id: `business-${item.id}`,
      kind: "business" as const,
      subject: item.business,
      state: item.health >= 70 ? "ready" as const : "warning" as const,
      owner: "TLC Capital Group OS",
      detail: `${item.health}/100; ${item.operatingFocus}`,
      updatedAt: now,
    })),
    { id: "workbench-plan", kind: "workbench", subject: "Plan to approval workflow", state: "ready", owner: "Hermes", detail: "Plan, approval, evidence, and report slots are available.", updatedAt: now },
    { id: "workbench-execute", kind: "workbench", subject: "Live execution bridge", state: "gated", owner: "Operations", detail: "Requires permission runtime, audit store, and explicit approval.", updatedAt: now },
    { id: "quality-visual", kind: "quality", subject: "Dashboard visual checks", state: "ready", owner: "Hermes", detail: "Playwright desktop/mobile dashboard suite covers governed routes.", updatedAt: now },
    { id: "quality-production", kind: "quality", subject: "Production URL checks", state: "gated", owner: "Operations", detail: "Needs deployed production URL screenshot and health checks.", updatedAt: now },
    { id: "autonomy-kill-switch", kind: "autonomy", subject: "Kill switch", state: "gated", owner: "Operations", detail: "Must be wired before scheduled autonomous execution.", updatedAt: now },
    { id: "autonomy-budget-breaker", kind: "autonomy", subject: "Budget breaker", state: "gated", owner: "Operations", detail: "Must stop runaway provider or loop spend.", updatedAt: now },
    { id: "registry-root", kind: "registry", subject: "Root dashboard registry", state: "ready", owner: "Hermes", detail: "Merged root and sibling manifests define the production dashboard catalog.", updatedAt: now },
    { id: "registry-production-check", kind: "registry", subject: "Production route verification", state: "gated", owner: "Operations", detail: "Requires live DNS, Caddy, health, and screenshot checks.", updatedAt: now },
    { id: "telemetry-contract", kind: "telemetry", subject: "Telemetry signal families", state: "ready", owner: "Hermes", detail: "Health, logs, cost, capacity, storage, and queue signals are defined.", updatedAt: now },
    { id: "telemetry-project-adapters", kind: "telemetry", subject: "Project telemetry adapters", state: "warning", owner: "Project teams", detail: "Each production app still needs full usage/storage/queue instrumentation.", updatedAt: now },
    { id: "incident-schema", kind: "incident", subject: "Incident response schema", state: "ready", owner: "Operations", detail: "Severity, owner, next step, and rollback path are required.", updatedAt: now },
    { id: "incident-remediation", kind: "incident", subject: "Automated remediation", state: "gated", owner: "Operations", detail: "Requires explicit permission gates and audit persistence.", updatedAt: now },
    { id: "deployment-promotion-rail", kind: "deployment", subject: "Shared deployment promotion rail", state: "ready", owner: "Operations", detail: "Validate, build, test, migrate, deploy, health-check, screenshot, rollback.", updatedAt: now },
    { id: "deployment-auto-promote", kind: "deployment", subject: "Automatic production promotion", state: "gated", owner: "Operations", detail: "Blocked until permission, quality, and autonomy gates pass.", updatedAt: now },
    { id: "secrets-presence", kind: "secrets", subject: "Secret presence checks", state: "ready", owner: "Operations", detail: "Presence and scope can be tracked without exposing values.", updatedAt: now },
    { id: "secrets-rotation", kind: "secrets", subject: "Secret rotation policy", state: "gated", owner: "Operations", detail: "Needs vault or managed secret backend before automation.", updatedAt: now },
    { id: "catalog-source-schema", kind: "catalog", subject: "Data source schema", state: "ready", owner: "Hermes", detail: "Owner, cadence, freshness, cost, retention, and consumers are required.", updatedAt: now },
    { id: "catalog-lineage", kind: "catalog", subject: "Automated source lineage", state: "gated", owner: "Hermes", detail: "Requires project adapter reports and code scanning.", updatedAt: now },
    { id: "finance-cost-schema", kind: "finance", subject: "Cost attribution schema", state: "ready", owner: "TLC Capital Group OS", detail: "Costs can map to project, business unit, and cost bucket.", updatedAt: now },
    { id: "finance-invoices", kind: "finance", subject: "Invoice reconciliation", state: "gated", owner: "Finance", detail: "Needs actual billing exports or manual monthly imports.", updatedAt: now },
    { id: "learning-evidence-loop", kind: "learning", subject: "Learning evidence lifecycle", state: "ready", owner: "Hermes", detail: "Evidence can move from candidate to finding to policy proposal.", updatedAt: now },
    { id: "learning-auto-policy", kind: "learning", subject: "Automated policy updates", state: "gated", owner: "Operations", detail: "Requires approval, regression checks, and counterevidence review.", updatedAt: now },
    { id: "eval-provider-matrix", kind: "eval", subject: "Provider evaluation matrix", state: "ready", owner: "Hermes", detail: "Correctness, tests, design, cost, and latency are the first eval dimensions.", updatedAt: now },
    { id: "eval-auto-routing", kind: "eval", subject: "Automatic provider routing", state: "gated", owner: "Hermes", detail: "Requires scored outcome history before automatic replacement.", updatedAt: now },
    { id: "executive-cockpit-rollup", kind: "executive", subject: "Executive operating cockpit", state: "ready", owner: "TLC Capital Group OS", detail: "Rolls up health, cost, incidents, deploys, learning, revenue, and autonomy.", updatedAt: now },
    { id: "executive-autonomy-controls", kind: "executive", subject: "Executive autonomy controls", state: "gated", owner: "Operations", detail: "CEO-level controls remain gated by project limits and explicit approval.", updatedAt: now },
    { id: "registry-v41-runner", kind: "registry", subject: "Live production verification runner", state: "gated", owner: "Operations", detail: "DNS, Caddy, health, snapshot, and screenshot checks require operator-triggered execution.", updatedAt: now },
    { id: "permission-v42-command-gates", kind: "permission", subject: "Command gate runtime", state: "ready", owner: "Operations", detail: "High-risk commands can be evaluated and audited before execution.", updatedAt: now },
    { id: "telemetry-v43-adapter-kit", kind: "telemetry", subject: "Project telemetry adapter kit", state: "ready", owner: "Hermes", detail: "Projects can emit standard health, cost, storage, API, queue, deployment, and action signals.", updatedAt: now },
    { id: "incident-v44-ingestion", kind: "incident", subject: "Incident ingestion rules", state: "gated", owner: "Operations", detail: "Automatic ingestion waits on reliable production verification and telemetry adapters.", updatedAt: now },
    { id: "deployment-v45-runner", kind: "deployment", subject: "Shared promotion runner", state: "gated", owner: "Operations", detail: "Live deploy execution must go through command gates and promotion evidence.", updatedAt: now },
    { id: "secrets-v46-scanner", kind: "secrets", subject: "Secrets posture scanner", state: "gated", owner: "Operations", detail: "Live GitHub and Hetzner checks require approved credentials and audit records.", updatedAt: now },
    { id: "finance-v47-engine", kind: "finance", subject: "Cost attribution engine", state: "ready", owner: "TLC Capital Group OS", detail: "Project cost records can be attributed to buckets and business units.", updatedAt: now },
    { id: "learning-v48-ingestion", kind: "learning", subject: "Learning ingestion pipeline", state: "ready", owner: "Hermes", detail: "Runtime learning records can capture outcome evidence and recommendations.", updatedAt: now },
    { id: "eval-v49-harness", kind: "eval", subject: "Agent and model eval harness", state: "warning", owner: "Hermes", detail: "Eval records are available; golden task execution history still needs depth.", updatedAt: now },
    { id: "autonomy-v50-circuit-breakers", kind: "autonomy", subject: "Runtime circuit breakers", state: "gated", owner: "Operations", detail: "Hard kill-switch and budget-breaker enforcement must wrap live execution paths.", updatedAt: now },
    { id: "registry-v51-production-sweep", kind: "registry", subject: "Production DNS and health sweep", state: "gated", owner: "Operations", detail: "Live production sweep can record DNS, TLS, health, snapshot, and screenshot evidence.", updatedAt: now },
    { id: "deployment-v52-hetzner-execution", kind: "deployment", subject: "Hetzner promotion execution", state: "gated", owner: "Operations", detail: "Real promotion execution remains behind explicit approval, breaker checks, and post-deploy verification.", updatedAt: now },
    { id: "permission-v53-gate-coverage", kind: "permission", subject: "Command gate coverage auditor", state: "warning", owner: "Operations", detail: "Command handlers can be inventoried and missing gates can create incidents.", updatedAt: now },
    { id: "telemetry-v54-adapter-rollout", kind: "telemetry", subject: "Project adapter rollout", state: "warning", owner: "Hermes", detail: "Registered dashboard manifests can be checked for standard telemetry adapter adoption.", updatedAt: now },
    { id: "incident-v55-automation", kind: "incident", subject: "Incident automation engine", state: "gated", owner: "Operations", detail: "Failures can create incidents automatically while remediation stays approval-gated.", updatedAt: now },
    { id: "secrets-v56-live-scan", kind: "secrets", subject: "Live secret presence scan", state: "gated", owner: "Operations", detail: "Secret scans record presence and scope only; values never enter payloads.", updatedAt: now },
    { id: "finance-v57-reconciliation", kind: "finance", subject: "Cost reconciliation import", state: "warning", owner: "Finance", detail: "Manual rates and invoice totals can reconcile estimates to actuals.", updatedAt: now },
    { id: "learning-v58-outcome-feeds", kind: "learning", subject: "Outcome learning feeds", state: "warning", owner: "Hermes", detail: "Project outcomes can be batch-ingested into learning records with source evidence.", updatedAt: now },
    { id: "eval-v59-golden-execution", kind: "eval", subject: "Golden eval execution", state: "warning", owner: "Hermes", detail: "Golden task batches can record provider scores before model routing changes.", updatedAt: now },
    { id: "autonomy-v60-hard-breakers", kind: "autonomy", subject: "Hard circuit breaker enforcement", state: "gated", owner: "Operations", detail: "Live execution paths can be blocked by kill, budget, provider, scheduler, and project breakers.", updatedAt: now },
    { id: "registry-v61-network-runner", kind: "registry", subject: "Network runner adapter", state: "gated", owner: "Operations", detail: "Approved network probe adapters can produce production sweep evidence.", updatedAt: now },
    { id: "deployment-v62-hetzner-ssh", kind: "deployment", subject: "Hetzner SSH adapter", state: "gated", owner: "Operations", detail: "Remote command plans can be recorded before approved SSH execution.", updatedAt: now },
    { id: "secrets-v63-provider-adapter", kind: "secrets", subject: "Secret provider adapter", state: "gated", owner: "Operations", detail: "Provider scans record names, scope, and posture without raw values.", updatedAt: now },
    { id: "finance-v64-billing-adapter", kind: "finance", subject: "Billing provider adapter", state: "warning", owner: "Finance", detail: "Billing provider imports can reconcile usage and invoices after approval.", updatedAt: now },
    { id: "learning-v65-outcome-emitter", kind: "learning", subject: "Project outcome emitter", state: "warning", owner: "Hermes", detail: "Projects can emit standard outcome events into Hermes learning feeds.", updatedAt: now },
    { id: "eval-v66-provider-runner", kind: "eval", subject: "Provider eval runner", state: "gated", owner: "Hermes", detail: "Golden task execution can be run through breaker-aware provider adapters.", updatedAt: now },
    { id: "autonomy-v67-breaker-middleware", kind: "autonomy", subject: "Breaker middleware SDK", state: "gated", owner: "Operations", detail: "Projects need a reusable middleware call before live execution.", updatedAt: now },
    { id: "incident-v68-subscriptions", kind: "incident", subject: "Incident subscription bus", state: "warning", owner: "Operations", detail: "Operational failure sources can subscribe into incident automation with dedupe.", updatedAt: now },
    { id: "catalog-v69-artifact-store", kind: "catalog", subject: "Evidence artifact store", state: "warning", owner: "Hermes", detail: "Large evidence should be indexed by pointer, hash, source, and retention policy.", updatedAt: now },
    { id: "deployment-v70-release-train", kind: "deployment", subject: "Release train orchestrator", state: "gated", owner: "Operations", detail: "Multi-project release trains remain manual-first until adapters prove reliable.", updatedAt: now },
    { id: "quality-v71-screenshot-runner", kind: "quality", subject: "Production screenshot runner", state: "gated", owner: "Operations", detail: "Approved sweeps can attach screenshot evidence once the browser runner and artifact backend are configured.", updatedAt: now },
    { id: "deployment-v72-hetzner-transport", kind: "deployment", subject: "Hetzner promotion transport", state: "gated", owner: "Operations", detail: "Shared SSH promotion transport is modeled; live execution requires configured credentials.", updatedAt: now },
    { id: "secrets-v73-server-posture", kind: "secrets", subject: "Server secret posture scanner", state: "gated", owner: "Operations", detail: "Server env-name checks can verify presence without exposing values.", updatedAt: now },
    { id: "incident-v74-notification-fanout", kind: "incident", subject: "Incident notification fanout", state: "warning", owner: "Operations", detail: "Incident records need channel fanout, dedupe, acknowledgement, and resolution.", updatedAt: now },
    { id: "catalog-v75-artifact-backend", kind: "catalog", subject: "Durable artifact backend", state: "warning", owner: "Hermes", detail: "Artifact backend records can choose local, object-store, or hybrid storage.", updatedAt: now },
    { id: "telemetry-v76-outcome-adapters", kind: "telemetry", subject: "Remaining project outcome adapters", state: "warning", owner: "Project teams", detail: "All production dashboards should emit /api/hermes/outcomes.", updatedAt: now },
    { id: "autonomy-v77-breaker-rollout", kind: "autonomy", subject: "Breaker middleware rollout", state: "gated", owner: "Operations", detail: "Scheduler, provider, deploy, and autopilot paths need breaker enforcement.", updatedAt: now },
    { id: "eval-v78-provider-execution", kind: "eval", subject: "Provider eval execution", state: "gated", owner: "Hermes", detail: "Paid provider evals must run behind budget breakers and approval.", updatedAt: now },
    { id: "finance-v79-billing-integrations", kind: "finance", subject: "Billing provider integrations", state: "warning", owner: "Finance", detail: "Provider usage imports should reconcile against manual billing actuals.", updatedAt: now },
    { id: "deployment-v80-release-execution", kind: "deployment", subject: "Release train execution", state: "gated", owner: "Operations", detail: "Release trains require green sweeps, secrets, breakers, artifacts, incidents, rollback, and summaries.", updatedAt: now },
  ];

  return {
    evidence,
    audit: [
      {
        id: "audit-seed-readonly",
        action: "run-readiness-check",
        actor: "Hermes",
        allowed: true,
        approval: "none",
        reason: "Read-only readiness checks can run without explicit approval.",
        createdAt: now,
      },
    ],
  };
}

function kindForStage(version: OperatingSystemStage["version"]): RuntimeEvidenceKind[] {
  const map: Record<OperatingSystemStage["version"], RuntimeEvidenceKind[]> = {
    V22: ["snapshot"],
    V23: ["memory"],
    V24: ["permission"],
    V25: ["model"],
    V26: ["loop"],
    V27: ["business", "snapshot"],
    V28: ["workbench", "permission", "memory"],
    V29: ["quality", "model"],
    V30: ["autonomy", "permission", "quality"],
    V31: ["registry", "snapshot"],
    V32: ["telemetry", "snapshot"],
    V33: ["incident", "deployment"],
    V34: ["deployment", "quality", "permission"],
    V35: ["secrets", "permission"],
    V36: ["catalog", "telemetry"],
    V37: ["finance", "business"],
    V38: ["learning", "quality"],
    V39: ["eval", "model"],
    V40: ["executive", "business", "autonomy"],
    V41: ["registry", "snapshot", "quality"],
    V42: ["permission"],
    V43: ["telemetry", "snapshot"],
    V44: ["incident", "telemetry"],
    V45: ["deployment", "permission", "quality"],
    V46: ["secrets", "permission"],
    V47: ["finance", "business"],
    V48: ["learning", "incident", "deployment"],
    V49: ["eval", "model", "quality"],
    V50: ["autonomy", "permission", "finance"],
    V51: ["registry", "snapshot", "quality"],
    V52: ["deployment", "permission", "quality"],
    V53: ["permission", "incident"],
    V54: ["telemetry", "snapshot", "registry"],
    V55: ["incident", "registry", "telemetry"],
    V56: ["secrets", "permission"],
    V57: ["finance", "catalog"],
    V58: ["learning", "incident", "deployment", "eval"],
    V59: ["eval", "model", "finance"],
    V60: ["autonomy", "permission", "finance"],
    V61: ["registry", "quality", "incident"],
    V62: ["deployment", "permission", "quality"],
    V63: ["secrets", "permission"],
    V64: ["finance", "catalog"],
    V65: ["learning", "telemetry"],
    V66: ["eval", "model", "finance"],
    V67: ["autonomy", "permission"],
    V68: ["incident", "telemetry"],
    V69: ["catalog", "quality"],
    V70: ["deployment", "registry", "incident", "executive"],
    V71: ["quality", "registry", "catalog"],
    V72: ["deployment", "permission", "quality"],
    V73: ["secrets", "permission", "incident"],
    V74: ["incident", "telemetry"],
    V75: ["catalog", "quality"],
    V76: ["telemetry", "learning"],
    V77: ["autonomy", "permission"],
    V78: ["eval", "model", "finance"],
    V79: ["finance", "catalog"],
    V80: ["deployment", "registry", "incident", "executive"],
  };
  return map[version];
}

function actionForStage(version: OperatingSystemStage["version"]) {
  const map: Record<OperatingSystemStage["version"], string> = {
    V22: "refresh-snapshots",
    V23: "review-memory-store",
    V24: "evaluate-permission-runtime",
    V25: "evaluate-model-routing",
    V26: "run-loop-dry-run",
    V27: "refresh-business-command",
    V28: "open-agent-workbench",
    V29: "run-quality-gates",
    V30: "evaluate-autonomy-readiness",
    V31: "refresh-project-registry",
    V32: "evaluate-telemetry-fabric",
    V33: "review-incident-command",
    V34: "evaluate-deployment-promotion",
    V35: "review-secrets-posture",
    V36: "refresh-data-source-catalog",
    V37: "evaluate-finance-attribution",
    V38: "review-learning-engine",
    V39: "run-agent-eval-lab",
    V40: "evaluate-executive-cockpit",
    V41: "run-production-verification",
    V42: "evaluate-command-gates",
    V43: "refresh-telemetry-adapters",
    V44: "ingest-incidents",
    V45: "run-deployment-promotion",
    V46: "scan-secrets-posture",
    V47: "evaluate-cost-attribution",
    V48: "ingest-learning-evidence",
    V49: "run-model-eval-harness",
    V50: "evaluate-circuit-breakers",
    V51: "execute-production-sweep",
    V52: "execute-hetzner-promotion",
    V53: "audit-command-gate-coverage",
    V54: "rollout-project-adapters",
    V55: "run-incident-automation",
    V56: "scan-live-secret-presence",
    V57: "import-cost-reconciliation",
    V58: "ingest-outcome-learning-feeds",
    V59: "execute-golden-eval-batch",
    V60: "enforce-hard-circuit-breakers",
    V61: "execute-network-runner-adapter",
    V62: "execute-hetzner-ssh-adapter",
    V63: "scan-secret-provider-adapter",
    V64: "import-billing-provider-adapter",
    V65: "emit-project-outcome",
    V66: "execute-provider-eval-runner",
    V67: "apply-breaker-middleware",
    V68: "subscribe-incident-source",
    V69: "index-evidence-artifact",
    V70: "orchestrate-release-train",
    V71: "execute-production-screenshot-runner",
    V72: "execute-hetzner-promotion-transport",
    V73: "scan-server-secret-posture",
    V74: "dispatch-incident-notification-fanout",
    V75: "configure-durable-artifact-backend",
    V76: "rollout-project-outcome-adapters",
    V77: "apply-breaker-middleware-rollout",
    V78: "execute-provider-eval-run",
    V79: "import-billing-provider-usage",
    V80: "execute-release-train",
  };
  return map[version];
}

function permissionForAction(action: string) {
  if (action.includes("loop") || action.includes("autonomy") || action.includes("workbench") || action.includes("promotion") || action.includes("secrets") || action.includes("secret") || action.includes("circuit") || action.includes("execute") || action.includes("import")) {
    return { approval: "explicit" as const };
  }
  if (action.includes("refresh") || action.includes("evaluate")) {
    return { approval: "confirm" as const };
  }
  return { approval: "none" as const };
}

function normalizeEvidence(record: ServerEvidenceRecord): RuntimeEvidenceRecord {
  return {
    id: record.id,
    kind: record.kind,
    subject: record.subject,
    state: record.state,
    owner: record.owner,
    detail: record.detail,
    updatedAt: record.updatedAt ?? record.updated_at ?? new Date().toISOString(),
  };
}

function normalizeAudit(record: ServerAuditRecord): RuntimeAuditRecord {
  return {
    id: record.id,
    action: record.action,
    actor: record.actor,
    allowed: Boolean(record.allowed),
    approval: record.approval,
    reason: record.reason,
    createdAt: record.createdAt ?? record.created_at ?? new Date().toISOString(),
  };
}

function mergeById<T extends { id: string }>(base: T[], incoming: T[]) {
  const map = new Map<string, T>();
  for (const item of base) map.set(item.id, item);
  for (const item of incoming) map.set(item.id, item);
  return Array.from(map.values());
}
