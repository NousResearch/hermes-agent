import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { AlertTriangle, ArrowRight, Bot, CheckCircle2, ClipboardCheck, GitBranch, HeartPulse, Inbox, PlayCircle, ShieldCheck, Sparkles } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { KanbanBoardResponse, KanbanTask, TeamConstitutionContract, TeamProposal, TeamProposalPlanPreviewResponse, TeamProposalSuggestedProfile, TeamProposalsResponse, TeamPulseSummary } from "@/lib/api";
import OldTeamProposalsPage from "@/pages/TeamProposalsPage";

type TeamCommandMode = "evolution" | "operative" | "proposals";
type DecisionAction = "Approva" | "Modifica" | "Indirizza" | "Scarta";
type MatureProposalAction = "accept" | "develop" | "reject";

interface PendingPlanConfirmation {
  proposal: TeamProposal;
  preview: TeamProposalPlanPreviewResponse;
}

interface TeamCommandConfig {
  mode: TeamCommandMode;
  title: string;
  description: string;
  board: string;
  siblingHref: string;
  siblingLabel: string;
  archiveHref: string;
  domainLabel: string;
  emptyDecision: string;
  pulseFallback: string;
  pulseMode: "evolution" | "operative";
}

const CONFIG: Record<TeamCommandMode, TeamCommandConfig> = {
  evolution: {
    mode: "evolution",
    title: "Sviluppo Hermes",
    description: "Centro di comando per evolvere Hermes: poche decisioni, guardrail chiari, dettagli fuori dalla prima vista.",
    board: "sviluppo-hermes",
    siblingHref: "/team-work",
    siblingLabel: "Apri Team Operativo",
    archiveHref: "/team-evolution/archive",
    domainLabel: "sviluppo interno Hermes",
    emptyDecision: "Nessuna decisione di sviluppo pronta per Daniele.",
    pulseFallback: "Nessun segnale tecnico rilevante oggi.",
    pulseMode: "evolution",
  },
  operative: {
    mode: "operative",
    title: "Team Operativo",
    description: "Centro di comando per i progetti reali: decisioni operative, polso leggero, evidenze e handoff separati.",
    board: "team-operativo",
    siblingHref: "/team-evolution",
    siblingLabel: "Apri Sviluppo Hermes",
    archiveHref: "/team-work/archive",
    domainLabel: "lavoro operativo",
    emptyDecision: "Nessuna decisione operativa pronta per Daniele.",
    pulseFallback: "Nessun segnale operativo rilevante oggi.",
    pulseMode: "operative",
  },
  proposals: {
    mode: "proposals",
    title: "Team & Proposte",
    description: "Centro di comando speculare a Sviluppo Hermes per il team operativo: poche decisioni, proposte mature, lavoro partito e dettagli fuori dalla prima vista.",
    board: "team-operativo",
    siblingHref: "/team-evolution",
    siblingLabel: "Apri Sviluppo Hermes",
    archiveHref: "/team-proposals/archive",
    domainLabel: "team operativo e proposte",
    emptyDecision: "Nessuna proposta operativa pronta per Daniele.",
    pulseFallback: "Nessun segnale operativo rilevante oggi.",
    pulseMode: "operative",
  },
};

function allTasks(board?: KanbanBoardResponse | null): KanbanTask[] {
  return board?.columns.flatMap((column) => column.tasks) ?? [];
}

function taskText(task: KanbanTask): string {
  return `${task.title}\n${task.body ?? ""}\n${task.latest_summary ?? ""}`;
}

function isTestTask(task: KanbanTask): boolean {
  const title = task.title.trim();
  // Do not hide real development cards just because their brief mentions
  // tests/test coverage. Only synthetic/probe cards should be filtered out
  // from the command-center first view.
  return (
    /^(\[[^\]]+\]\s*)?TEST\b/i.test(title)
    || /should invalidate hash/i.test(title)
    || /test\/escluse|card test/i.test(taskText(task))
  );
}

function isProntaPerDaniele(task: KanbanTask): boolean {
  const text = taskText(task).toLowerCase();
  return task.status === "blocked" && text.includes("pipeline:pronta-per-daniele") && !isTestTask(task);
}

const ACTIVE_KANBAN_STATUSES = new Set(["triage", "todo", "scheduled", "ready", "running", "blocked", "review"]);

function isActiveTask(task: KanbanTask): boolean {
  return ACTIVE_KANBAN_STATUSES.has(task.status) && !isTestTask(task);
}

function isHandoff(task: KanbanTask): boolean {
  return /\bhandoff\b|cross-team/i.test(taskText(task));
}

function isActiveHandoff(task: KanbanTask): boolean {
  return isActiveTask(task) && isHandoff(task);
}

function isLaunchedKanbanWork(task: KanbanTask): boolean {
  return ["running", "ready"].includes(task.status) && !isTestTask(task);
}

function dedupeTasksById(tasks: KanbanTask[]): KanbanTask[] {
  const seen = new Set<string>();
  return tasks.filter((task) => {
    if (seen.has(task.id)) return false;
    seen.add(task.id);
    return true;
  });
}

function currentUnixSeconds(): number {
  return Math.floor(Date.now() / 1000);
}

function statusTone(status: string): "success" | "warning" | "secondary" | "destructive" {
  if (status === "running") return "success";
  if (status === "ready") return "warning";
  if (status === "blocked") return "destructive";
  return "secondary";
}

function kanbanBoardHref(board: string, suffix = ""): string {
  const path = suffix === "/v2" ? "/kanban-mission-v2" : "/kanban";
  return `${path}?board=${encodeURIComponent(board)}`;
}

function kanbanTaskHref(board: string, taskId: string): string {
  return `/kanban?board=${encodeURIComponent(board)}&task=${encodeURIComponent(taskId)}`;
}

function rememberKanbanBoard(board: string): void {
  try {
    window.localStorage.setItem("hermes.kanban.selectedBoard", board);
  } catch {
    /* ignore private/quota mode */
  }
}

function openKanbanBoard(board: string, suffix = ""): void {
  rememberKanbanBoard(board);
  window.location.href = kanbanBoardHref(board, suffix);
}

function openKanbanTask(board: string, taskId: string): void {
  rememberKanbanBoard(board);
  window.location.href = kanbanTaskHref(board, taskId);
}

function riskGate(task: KanbanTask): "🟢" | "🟡" | "🔴" {
  const text = taskText(task).toLowerCase();
  if (/rosso|🔴|external|gateway|deploy|restart|spesa|invio esterno|claim|legal|contratt/.test(text)) return "🔴";
  if (/giallo|🟡|reversibile|interno|cron|dashboard|kanban|documento/.test(text)) return "🟡";
  return "🟢";
}

function extractLine(task: KanbanTask, labels: string[], fallback: string): string {
  const lines = taskText(task).split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  for (const label of labels) {
    const rx = new RegExp(`^[-*#\\s]*${label}\\s*[:：-]\\s*(.+)$`, "i");
    const match = lines.map((line) => line.match(rx)).find(Boolean);
    if (match?.[1]) return match[1].trim();
  }
  return fallback;
}

function formatDate(seconds?: number | null): string {
  if (!seconds) return "n/d";
  return new Intl.DateTimeFormat("it-IT", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(seconds * 1000));
}

function formatIsoDate(value?: string | null): string {
  if (!value) return "n/d";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("it-IT", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatProposalFormulatedDate(proposal: TeamProposal): string {
  return formatIsoDate(proposal.formulated_at ?? proposal.created_at ?? proposal.status_updated_at ?? proposal.updated_at ?? proposal.last_signal_at);
}

function priorityLabel(value?: number | null): string {
  if (value === null || value === undefined) return "n/d";
  if (value >= 90) return "P0";
  if (value >= 70) return "P1";
  if (value >= 40) return "P2";
  return "P3";
}

function formatProposalContractValue(value: unknown, fallback = "n/d"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((item) => formatProposalContractValue(item, "")).filter(Boolean).join("\n") || fallback;
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return String(obj.summary ?? obj.hypothesis ?? obj.synthesis ?? obj.acceptance ?? obj.rationale ?? JSON.stringify(obj));
  }
  return String(value);
}

function formatProposalViewpoint(
  viewpoint: TeamProposal["supporter_view"] | TeamProposal["critic_view"],
  fallbackActor?: string | null,
  fallbackRationale?: string | null,
): string {
  const actor = viewpoint?.actor ?? fallbackActor ?? "non indicato";
  const rationale = viewpoint?.rationale ?? fallbackRationale ?? "razionale non disponibile";
  return `${actor}: ${rationale}`;
}

function formatProposalGate(proposal: TeamProposal): string {
  return [
    proposal.gate_state ?? "review_required",
    proposal.autonomy_gate ?? "approval_required",
    proposal.no_auto_dispatch ? "no auto-dispatch" : "solo dopo conferma",
  ].join(" · ");
}

function verifiedSuggestedProfiles(proposal: TeamProposal): TeamProposalSuggestedProfile[] {
  return (proposal.suggested_profiles ?? []).filter((profile) => profile.exists_verified && Boolean(profile.profile));
}

function unverifiedSuggestedProfilesCount(proposal: TeamProposal): number {
  return (proposal.suggested_profiles ?? []).filter((profile) => !profile.exists_verified || !profile.profile).length;
}

function formatProposalEvidence(proposal: TeamProposal): string {
  const refs = [
    ...(proposal.evidence_refs ?? []),
    ...(proposal.evidence_contract?.refs ?? []),
  ];
  return [
    proposal.evidence,
    formatProposalContractValue(proposal.signal ?? proposal.source_signal, ""),
    refs.length ? `Fonti: ${Array.from(new Set(refs)).join(", ")}` : "",
    proposal.source_agent ? `Source agent: ${proposal.source_agent}${proposal.source_agent_status ? ` · ${proposal.source_agent_status}` : ""}` : "",
  ].filter(Boolean).join("\n") || "Evidenza non disponibile nel payload: tenere la proposta in review, senza dedurre consenso.";
}

function formatProposalDependencies(proposal: TeamProposal): string {
  const gate = proposal.gate;
  const gateStatus = proposal.gate_status;
  const sourceDetails = proposal.source_agent_details;
  const dependencies = [
    "Conferma esplicita Daniele prima di creare o preparare task Kanban.",
    "Preview piano/task visibile e hash confermato prima della conversione.",
    verifiedSuggestedProfiles(proposal).length > 0 ? "Owner suggeriti verificati contro registry profili Hermes." : "Owner non sufficientemente verificati: scegliere/validare profili reali prima della conversione.",
    sourceDetails ? `Registry/source mapping: ${sourceDetails.mapping_status}${sourceDetails.profile ? ` · ${sourceDetails.profile}` : ""}` : "Source mapping non disponibile nel payload corrente.",
    gateStatus?.status ? `Gate status: ${gateStatus.status}` : null,
    gate?.decision_needed ? `Decisione richiesta: ${gate.decision_needed}` : null,
    gate?.forbidden_without_approval?.length ? `Bloccato senza conferma: ${gate.forbidden_without_approval.join(", ")}` : "Bloccato senza conferma: dispatch, cron e invii esterni.",
  ];
  return dependencies.filter(Boolean).join("\n");
}

const CLOSED_PROPOSAL_STATUSES = new Set(["scartata", "trasformata_in_task", "rejected", "converted_to_kanban", "archived"]);

function isVisibleRegistryProposal(proposal: TeamProposal, mode: "evolution" | "operative"): boolean {
  return proposal.kind === mode && !CLOSED_PROPOSAL_STATUSES.has(proposal.status);
}

function mergeUniqueProposals(primary: TeamProposal[], fallback: TeamProposal[]): TeamProposal[] {
  const seen = new Set<string>();
  return [...primary, ...fallback].filter((proposal) => {
    if (seen.has(proposal.id)) return false;
    seen.add(proposal.id);
    return true;
  });
}

function AutonomousProposalEvidenceGrid({ proposal }: { proposal: TeamProposal }) {
  return (
    <div className="grid gap-3 border-t border-border/60 pt-3 md:grid-cols-3">
      <Field label="Segnale" value={formatProposalContractValue(proposal.signal ?? proposal.source_signal ?? proposal.whyNow)} />
      <Field label="Interpretazione" value={formatProposalContractValue(proposal.interpretation)} />
      <Field label="Supporter" value={formatProposalViewpoint(proposal.supporter ?? proposal.supporter_view, proposal.challenge?.supporter, proposal.challenge?.support)} />
      <Field label="Critic" value={formatProposalViewpoint(proposal.critic ?? proposal.critic_view, proposal.challenge?.critic, proposal.challenge?.challenge)} />
      <Field label="Chief synthesis" value={formatProposalContractValue(proposal.chief_synthesis ?? proposal.challenge?.chief_synthesis)} />
      <Field label="Gate / dispatch" value={formatProposalGate(proposal)} />
      <Field label="Source agent" value={proposal.source_agent_status && proposal.source_agent_legacy ? `${proposal.source_agent ?? "n/d"} · ${proposal.source_agent_status} da ${proposal.source_agent_legacy}` : proposal.source_agent ?? "n/d"} />
    </div>
  );
}

function useCommandCenterData(mode: TeamCommandMode) {
  const config = CONFIG[mode];
  const otherConfig = mode === "evolution" ? CONFIG.proposals : CONFIG.evolution;
  const [board, setBoard] = useState<KanbanBoardResponse | null>(null);
  const [otherBoard, setOtherBoard] = useState<KanbanBoardResponse | null>(null);
  const [registry, setRegistry] = useState<TeamProposalsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = async (options?: { silent?: boolean }) => {
    if (!options?.silent) setLoading(true);
    setError(null);
    try {
      const [teamBoardResult, siblingBoardResult, proposalRegistry] = await Promise.all([
        api.getKanbanBoard(config.board).then((value) => ({ ok: true as const, value })).catch((reason) => ({ ok: false as const, reason })),
        api.getKanbanBoard(otherConfig.board).then((value) => ({ ok: true as const, value })).catch((reason) => ({ ok: false as const, reason })),
        api.getTeamProposals().catch(() => null),
      ]);
      setBoard(teamBoardResult.ok ? teamBoardResult.value : null);
      setOtherBoard(siblingBoardResult.ok ? siblingBoardResult.value : null);
      setRegistry(proposalRegistry);
      if (!teamBoardResult.ok || !siblingBoardResult.ok) {
        setError("Kanban non raggiungibile in questa sessione: il centro di comando resta in fallback read-only, senza inventare decisioni o metriche.");
      }
    } finally {
      if (!options?.silent) setLoading(false);
    }
  };

  useEffect(() => {
    void Promise.resolve().then(() => refresh());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  return { board, error, loading, otherBoard, refresh, registry };
}

export function TeamOperationsPage() {
  return <TeamCommandCenterPage mode="operative" />;
}

export function TeamEvolutionPage() {
  return <TeamCommandCenterPage mode="evolution" />;
}

export function TeamProposalsCommandPage() {
  return <TeamCommandCenterPage mode="proposals" />;
}

export function TeamOperationsArchivePage() {
  return <ArchiveShell mode="operative" />;
}

export function TeamProposalsArchivePage() {
  return <ArchiveShell mode="proposals" />;
}

export function TeamEvolutionArchivePage() {
  return <ArchiveShell mode="evolution" />;
}

function ArchiveShell({ mode }: { mode: TeamCommandMode }) {
  const config = CONFIG[mode];
  return (
    <div className="flex flex-col gap-5 pb-10">
      <section className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <H2>{config.title} — archivio contenuto precedente</H2>
          <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
            Vista legacy reversibile: registry proposte, Radar, Chief queue, PM workspace, review duplicate e audit storico. Fuori dalla vista principale.
          </p>
        </div>
        <Button ghost size="sm" onClick={() => { window.location.href = mode === "operative" ? "/team-work" : "/team-evolution"; }}>
          Torna al centro di comando
        </Button>
      </section>
      <OldTeamProposalsPage mode={mode === "proposals" ? "operative" : mode} />
    </div>
  );
}

export default function TeamCommandCenterPage({ mode }: { mode: TeamCommandMode }) {
  const config = CONFIG[mode];
  const { board, error, loading, otherBoard, refresh, registry } = useCommandCenterData(mode);
  const [notice, setNotice] = useState<string | null>(null);
  const [updating, setUpdating] = useState<string | null>(null);
  const [editingProposal, setEditingProposal] = useState<TeamProposal | null>(null);
  const [pendingPlanConfirmation, setPendingPlanConfirmation] = useState<PendingPlanConfirmation | null>(null);
  const [editForm, setEditForm] = useState({ acceptance: "", title: "", whyNow: "" });
  const [optimisticLaunchedTasks, setOptimisticLaunchedTasks] = useState<KanbanTask[]>([]);

  const tasks = useMemo(() => allTasks(board), [board]);
  const decisions = useMemo(() => tasks.filter(isProntaPerDaniele), [tasks]);
  const visibleDecisions = decisions.slice(0, 3);
  const hiddenDecisions = decisions.slice(3);
  const launchedWork = useMemo(
    () => {
      const liveIds = new Set(tasks.map((task) => task.id));
      const pendingOptimisticTasks = optimisticLaunchedTasks.filter((task) => !liveIds.has(task.id));
      return dedupeTasksById([...pendingOptimisticTasks, ...tasks])
        .filter(isLaunchedKanbanWork)
        .sort((a, b) => (b.started_at ?? b.created_at ?? 0) - (a.started_at ?? a.created_at ?? 0))
        .slice(0, 6);
    },
    [optimisticLaunchedTasks, tasks],
  );
  const incomingHandoffs = useMemo(() => tasks.filter(isActiveHandoff), [tasks]);
  const outgoingHandoffs = useMemo(() => allTasks(otherBoard).filter((task) => isActiveHandoff(task) && taskText(task).toLowerCase().includes(config.board.replace("-", " ").split(" ")[0])), [otherBoard, config.board]);
  const pulse = registry?.team_pulses?.[config.pulseMode] ?? registry?.team_pulse;
  const constitution = registry?.team_constitutions?.[config.pulseMode];
  const registryActiveProposals = useMemo(
    () => registry?.proposals.filter((proposal) => isVisibleRegistryProposal(proposal, config.pulseMode)) ?? [],
    [registry, config.pulseMode],
  );

  const recordDecision = async (task: KanbanTask, action: DecisionAction) => {
    setUpdating(`${task.id}:${action}`);
    setNotice(null);
    try {
      const body = [
        `Decisione Daniele registrata dal Centro di Comando: ${action}.`,
        "Proposal Mode puro: nessun worker, dispatch, cron o invio esterno avviato da questa azione.",
        "La card resta sotto controllo umano finché Daniele non autorizza esplicitamente l'esecuzione separata.",
      ].join("\n");
      await api.commentKanbanTask(task.id, body, config.board, "command-center");
      if (task.status !== "blocked") {
        await api.updateKanbanTask(task.id, { status: "blocked", block_reason: `Decisione Daniele registrata: ${action}; execution gate separato.` }, config.board);
      }
      setNotice(`${action} registrato su ${config.board}/${task.id}. Dispatcher attivabile secondo priorità.`);
      await refresh();
    } catch (e) {
      setNotice(`Errore registrazione decisione: ${String(e)}`);
    } finally {
      setUpdating(null);
    }
  };

  const runMatureProposalAction = async (proposal: TeamProposal, action: MatureProposalAction) => {
    if (action === "develop") {
      setEditingProposal(proposal);
      setEditForm({
        acceptance: proposal.acceptance ?? "",
        title: proposal.title,
        whyNow: proposal.whyNow ?? "",
      });
      setNotice(`Modifica ${proposal.title}, poi conferma lo sviluppo.`);
      return;
    }
    setUpdating(`${proposal.id}:${action}`);
    setNotice(null);
    try {
      if (action === "accept") {
        await api.reviewTeamProposalAsChief(
          proposal.id,
          "shortlist",
          "Accettata da Daniele dalla pagina team: standby, interessa ma non è da sviluppare ora.",
        );
        setNotice(`In standby: ${proposal.title}. Libera spazio per nuove proposte; nessun dispatch automatico avviato.`);
      } else if (action === "reject") {
        await api.reviewTeamProposalAsChief(
          proposal.id,
          "reject",
          "Scartata da Daniele dalla pagina team.",
        );
        setNotice(`Scartata: ${proposal.title}.`);
      }
      await refresh();
    } catch (e) {
      setNotice(`Errore azione proposta: ${String(e)}`);
    } finally {
      setUpdating(null);
    }
  };

  const saveAndDevelopProposal = async () => {
    if (!editingProposal) return;
    setUpdating(`${editingProposal.id}:preview-plan`);
    setNotice(null);
    try {
      const updated = await api.updateTeamProposal(editingProposal.id, {
        acceptance: editForm.acceptance,
        title: editForm.title,
        whyNow: editForm.whyNow,
      });
      const preview = await api.getTeamProposalPlanPreview(editingProposal.id);
      setPendingPlanConfirmation({ proposal: updated.proposal ?? editingProposal, preview });
      setEditingProposal(null);
      setNotice(`Preview piano pronta per ${editForm.title || editingProposal.title}: nessun task Kanban creato finché Daniele non conferma nel dialog.`);
    } catch (e) {
      setNotice(`Errore preparazione preview piano: ${String(e)}`);
    } finally {
      setUpdating(null);
    }
  };

  const confirmPlanConversion = async () => {
    if (!pendingPlanConfirmation) return;
    const { proposal, preview } = pendingPlanConfirmation;
    setUpdating(`${proposal.id}:confirm-plan`);
    setNotice(null);
    try {
      await api.approveTeamProposalMinStep(proposal.id, {
        action_type: "plan",
        board: config.board,
        confirmed_preview_hash: preview.preview_hash,
        note: "Conferma esplicita Daniele dal dialog blocker-debate del Centro di comando: autorizza conversione piano Kanban.",
      });
      const response = await api.convertTeamProposalToPlan(proposal.id, preview.preview_hash, config.board);
      const now = currentUnixSeconds();
      setOptimisticLaunchedTasks([
        {
          id: response.plan.parent_task_id,
          title: response.plan.title,
          body: `Piano appena lanciato da ${config.title}. In attesa del prossimo refresh Kanban live.`,
          status: "ready",
          assignee: response.plan.parent_assignee,
          created_at: now,
          priority: 100,
        },
        ...response.plan.child_task_ids.map((id, index) => ({
          id,
          title: preview.plan.tasks[index]?.title ?? `Task figlio ${index + 1}`,
          body: `Task figlio appena lanciato da ${config.title}. In attesa del prossimo refresh Kanban live.`,
          status: "ready",
          assignee: response.plan.child_assignees?.[index] ?? preview.plan.tasks[index]?.assignee ?? null,
          created_at: now - index - 1,
          priority: preview.plan.tasks[index]?.priority ?? 72,
        })),
      ]);
      setNotice(`Piano lanciato in Kanban ready su ${config.board}: parent ${response.plan.parent_task_id} + ${response.plan.child_task_ids.length} task figli. Li mostro subito qui sotto; conferma live Kanban in background.`);
      setPendingPlanConfirmation(null);
      void refresh({ silent: true });
      for (const delay of [800, 1800, 3500]) {
        window.setTimeout(() => { void refresh({ silent: true }); }, delay);
      }
    } catch (e) {
      setNotice(`Errore conversione piano dopo conferma: ${String(e)}`);
    } finally {
      setUpdating(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground">
        <Spinner className="mr-2 text-xl text-primary" />
        Carico centro di comando…
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 pb-10">
      <CommandHeader config={config} onRefresh={refresh} />
      <TeamConstitutionPanel constitution={constitution} mode={mode} />
      {error && <Card><CardContent className="py-3 text-sm text-destructive">{error}</CardContent></Card>}
      {notice && <Card><CardContent className="py-3 text-sm text-success" role="status">{notice}</CardContent></Card>}

      <DecisionBlock
        config={config}
        hidden={hiddenDecisions}
        onDecision={recordDecision}
        tasks={visibleDecisions}
        updating={updating}
      />

      <PulseBlock fallback={config.pulseFallback} pulseSummary={pulse} updatedAt={pulse?.last_run_at ?? registry?.updated_at} />
      <MatureProposalsBlock
        editingProposal={editingProposal}
        editForm={editForm}
        onAction={runMatureProposalAction}
        onCancelEdit={() => setEditingProposal(null)}
        onChangeEdit={setEditForm}
        onConfirmEdit={saveAndDevelopProposal}
        pulseSummary={pulse}
        registryProposals={registryActiveProposals}
        updating={updating}
      />
      <LaunchedKanbanWorkBlock board={config.board} tasks={launchedWork} />
      <StandbyProposalsBlock
        editingProposal={editingProposal}
        editForm={editForm}
        onAction={runMatureProposalAction}
        onCancelEdit={() => setEditingProposal(null)}
        onChangeEdit={setEditForm}
        onConfirmEdit={saveAndDevelopProposal}
        pulseSummary={pulse}
        updating={updating}
      />
      {pendingPlanConfirmation && (
        <CommandCenterPlanConfirmationDialog
          busy={updating === `${pendingPlanConfirmation.proposal.id}:confirm-plan`}
          onCancel={() => setPendingPlanConfirmation(null)}
          onConfirm={confirmPlanConversion}
          pending={pendingPlanConfirmation}
        />
      )}
      <HealthBlock decisions={decisions.length} tasks={tasks} />
      <HandoffBlock currentBoard={config.board} incoming={incomingHandoffs} outgoing={outgoingHandoffs} />
      <DeepLinksBlock config={config} />
    </div>
  );
}

function CommandHeader({ config, onRefresh }: { config: TeamCommandConfig; onRefresh: () => void }) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <H2>{config.title}</H2>
          <p className="mt-2 max-w-3xl text-sm text-muted-foreground">{config.description}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button ghost size="sm" onClick={onRefresh}>Aggiorna</Button>
          <Button ghost size="sm" onClick={() => { window.location.href = config.siblingHref; }}>
            {config.siblingLabel}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 rounded-sm border bg-muted/30 p-2 text-xs text-muted-foreground">
        <Badge tone="secondary">Centro di comando</Badge>
        <span>Linea: {config.domainLabel}</span>
        <span>Fonte decisioni: Kanban `{config.board}` · Proposal Mode puro</span>
      </div>
    </section>
  );
}

function TeamConstitutionPanel({ constitution, mode }: { constitution?: TeamConstitutionContract; mode: TeamCommandMode }) {
  const missing = !constitution;
  const modeLabel = mode === "evolution" ? "Team Evoluzione Hermes" : "Team Operativo";
  const reads = constitution?.cycle_start_reads ?? [];
  const promptSources = constitution?.prompt_sources ?? [];
  const mustNot = constitution?.must_not ?? [];
  return (
    <section className="rounded-xl border border-primary/20 bg-card/70 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Costituzione applicata</h3>
        <Badge tone="secondary">{modeLabel}</Badge>
        <Badge tone="warning">Proposal → task spec → Kanban controllata</Badge>
      </div>
      {missing ? (
        <p className="mt-3 text-sm text-muted-foreground">Costituzione non disponibile dal registry: mantieni comunque Proposal Mode puro e nessun dispatch automatico.</p>
      ) : (
        <div className="mt-3 grid gap-3 lg:grid-cols-3">
          <Field label="Missione / stella polare" value={`${constitution.mission}\n${constitution.north_star}`} />
          <Field label="Prompt attivo: Costituzione comune + identità specifica" value={promptSources.slice(0, 3).join("\n") || "n/d"} />
          <Field label="Lettura a inizio ciclo" value={reads.join("\n") || "n/d"} />
          <Field label="Cosa non fare" value={mustNot.slice(0, 3).join("\n") || "n/d"} />
          <Field label="Handoff se emerge dominio dell'altro team" value={constitution.handoff} />
          <Field label="Gate autonomia" value={`${constitution.mode} · no invii esterni · no auto-spawn · no cron · review Daniele prima di Kanban/dispatch`} />
        </div>
      )}
    </section>
  );
}

function DecisionBlock({
  config,
  hidden,
  onDecision,
  tasks,
  updating,
}: {
  config: TeamCommandConfig;
  hidden: KanbanTask[];
  onDecision: (task: KanbanTask, action: DecisionAction) => Promise<void>;
  tasks: KanbanTask[];
  updating: string | null;
}) {
  return (
    <section className="flex flex-col gap-3 rounded-xl border border-primary/25 bg-card/80 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <ClipboardCheck className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Da decidere ora</h3>
        <Badge tone="warning">max 3</Badge>
      </div>
      {tasks.length === 0 ? (
        <Card>
          <CardContent className="py-4 text-sm text-muted-foreground">{config.emptyDecision}</CardContent>
        </Card>
      ) : (
        <div className="grid gap-3">
          {tasks.map((task) => (
            <DecisionCard key={task.id} onDecision={onDecision} task={task} updating={updating} />
          ))}
        </div>
      )}
      {hidden.length > 0 && (
        <details className="rounded-sm border border-border/60 bg-muted/20 p-3">
          <summary className="cursor-pointer text-sm font-medium">Altre proposte ({hidden.length})</summary>
          <div className="mt-3 grid gap-3">
            {hidden.map((task) => (
              <DecisionCard key={task.id} onDecision={onDecision} task={task} updating={updating} />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

function DecisionCard({ task, onDecision, updating }: { task: KanbanTask; onDecision: (task: KanbanTask, action: DecisionAction) => Promise<void>; updating: string | null }) {
  const recommendation = extractLine(task, ["raccomandazione", "recommendation", "proposta"], task.title);
  const whyNow = extractLine(task, ["perché ora", "why now", "why_now", "contesto"], task.latest_summary ?? "Da valutare ora perché è marcata Pronta per Daniele.");
  const risk = extractLine(task, ["rischio", "risk", "veto risk"], "n/d");
  const gate = riskGate(task);
  const actions: DecisionAction[] = ["Approva", "Modifica", "Indirizza", "Scarta"];
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 py-4">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge tone="secondary">{task.id}</Badge>
          <Badge tone="warning">{priorityLabel(task.priority)}</Badge>
          <span>Creata: {formatDate(task.created_at)}</span>
          <span>Gate: {gate}</span>
        </div>
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr_0.7fr_auto] lg:items-start">
          <Field label="Raccomandazione" value={recommendation} />
          <Field label="Perché ora" value={whyNow} />
          <Field label="Rischio" value={risk} />
          <div className="flex flex-wrap gap-2 lg:justify-end">
            {actions.map((action) => (
              <Button
                key={action}
                size="sm"
                ghost={action !== "Approva"}
                disabled={Boolean(updating)}
                onClick={() => void onDecision(task, action)}
              >
                {updating === `${task.id}:${action}` ? <Spinner className="mr-1 h-3 w-3" /> : null}
                {action}
              </Button>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PulseBlock({ fallback, pulseSummary, updatedAt }: { fallback: string; pulseSummary?: TeamPulseSummary | null; updatedAt?: string | null }) {
  const cycle = pulseSummary?.last_cycle;
  const mature = cycle?.mature_count ?? pulseSummary?.top_autonomous?.length ?? 0;
  const generated = (cycle?.created_count ?? 0) + (cycle?.updated_count ?? 0);
  return (
    <section className="rounded-xl border border-border/70 bg-card/70 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <HeartPulse className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Polso di oggi</h3>
        <Badge tone="secondary">sola lettura</Badge>
      </div>
      <p className="mt-3 text-sm text-muted-foreground">{cycle?.summary || pulseSummary?.summary || fallback}</p>
      <div className="mt-3 grid gap-2 md:grid-cols-4">
        <Metric label="Ultimo ciclo" value={formatIsoDate(cycle?.ran_at ?? pulseSummary?.last_run_at ?? updatedAt)} />
        <Metric label="Proposte mature" value={String(mature)} />
        <Metric label="Registry aggiornato" value={String(generated)} />
        <Metric label="Dispatch" value="solo dopo sblocco" />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Fonte: {cycle?.source ?? "registry"} · active={cycle?.active_count ?? pulseSummary?.active_count ?? 0} · parked={cycle?.parked_count ?? 0} · intake senza dispatch; conversione esplicita crea task ready.
      </p>
    </section>
  );
}

function LaunchedKanbanWorkBlock({ board, tasks }: { board: string; tasks: KanbanTask[] }) {
  return (
    <section className="rounded-xl border border-success/40 bg-card/70 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <PlayCircle className="h-5 w-5 text-success" />
          <h3 className="font-semibold">Lavoro partito in Kanban</h3>
          <Badge tone="success">{board}</Badge>
        </div>
        <Button ghost size="sm" onClick={() => { openKanbanBoard(board); }}>
          Apri Kanban completo
        </Button>
      </div>
      {tasks.length === 0 ? (
        <p className="mt-3 text-sm text-muted-foreground">Nessuna task ready/running visibile su questa board.</p>
      ) : (
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {tasks.map((task) => (
            <Card key={task.id}>
              <CardContent className="grid gap-2 py-3">
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge tone="secondary">{task.id}</Badge>
                  <Badge tone={statusTone(task.status)}>{task.status}</Badge>
                  {task.assignee ? <span>Assignee: {task.assignee}</span> : null}
                </div>
                <div className="font-medium">{task.title}</div>
                <div className="flex flex-wrap gap-2">
                  <Button ghost size="sm" onClick={() => { openKanbanTask(board, task.id); }}>
                    Apri card
                  </Button>
                  {task.status === "running" ? <Badge tone="success">worker attivo</Badge> : null}
                  {task.status === "ready" ? <Badge tone="warning">in coda dispatcher</Badge> : null}
                  {task.status === "todo" ? <Badge tone="secondary">in attesa dipendenze/promozione</Badge> : null}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </section>
  );
}

function MatureProposalsBlock({
  editingProposal,
  editForm,
  onAction,
  onCancelEdit,
  onChangeEdit,
  onConfirmEdit,
  pulseSummary,
  registryProposals,
  updating,
}: {
  editingProposal: TeamProposal | null;
  editForm: { acceptance: string; title: string; whyNow: string };
  onAction: (proposal: TeamProposal, action: MatureProposalAction) => Promise<void>;
  onCancelEdit: () => void;
  onChangeEdit: (form: { acceptance: string; title: string; whyNow: string }) => void;
  onConfirmEdit: () => Promise<void>;
  pulseSummary?: TeamPulseSummary | null;
  registryProposals: TeamProposal[];
  updating: string | null;
}) {
  const pulseProposals = pulseSummary?.mature_proposals?.length ? pulseSummary.mature_proposals : pulseSummary?.top_autonomous ?? [];
  const proposals = mergeUniqueProposals(pulseProposals, registryProposals);
  return (
    <section className="rounded-xl border border-border/70 bg-card/70 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <ClipboardCheck className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Proposte attive / mature</h3>
        <Badge tone="secondary">azioni dirette · review-gated</Badge>
        <Badge tone="warning">registry visibile</Badge>
      </div>
      {proposals.length === 0 ? (
        <p className="mt-3 text-sm text-muted-foreground">Nessuna proposta attiva visibile nel registry per questo team.</p>
      ) : (
        <div className="mt-3 grid gap-3">
          {proposals.map((proposal) => (
            <MatureProposalCard
              editingProposal={editingProposal}
              editForm={editForm}
              key={proposal.id}
              onAction={onAction}
              onCancelEdit={onCancelEdit}
              onChangeEdit={onChangeEdit}
              onConfirmEdit={onConfirmEdit}
              proposal={proposal}
              updating={updating}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function MatureProposalCard({
  editingProposal,
  editForm,
  onAction,
  onCancelEdit,
  onChangeEdit,
  onConfirmEdit,
  proposal,
  updating,
}: {
  editingProposal: TeamProposal | null;
  editForm: { acceptance: string; title: string; whyNow: string };
  onAction: (proposal: TeamProposal, action: MatureProposalAction) => Promise<void>;
  onCancelEdit: () => void;
  onChangeEdit: (form: { acceptance: string; title: string; whyNow: string }) => void;
  onConfirmEdit: () => Promise<void>;
  proposal: TeamProposal;
  updating: string | null;
}) {
  const actions: Array<{ action: MatureProposalAction; label: string; ghost?: boolean }> = [
    { action: "accept", label: "Accetta" },
    { action: "develop", label: "Sviluppa" },
    { action: "reject", label: "Scarta", ghost: true },
  ];
  return (
    <Card>
      <CardContent className="grid gap-3 py-4">
        <div className="grid gap-3 lg:grid-cols-[1fr_1fr_0.65fr_0.8fr_auto] lg:items-start">
          <Field label="Proposta" value={proposal.title} />
          <Field label="Perché ora" value={proposal.whyNow ?? proposal.evidence ?? "n/d"} />
          <Field label="Formulata" value={formatProposalFormulatedDate(proposal)} />
          <Field label="Gate" value={`${proposal.status} · ${proposal.recommendation ?? "n/d"} · ${formatProposalGate(proposal)}`} />
          <div className="flex flex-wrap gap-2 lg:justify-end">
            {actions.map(({ action, ghost, label }) => (
              <Button
                key={action}
                size="sm"
                ghost={ghost}
                disabled={Boolean(updating)}
                onClick={() => void onAction(proposal, action)}
              >
                {updating === `${proposal.id}:${action}` ? <Spinner className="mr-1 h-3 w-3" /> : null}
                {label}
              </Button>
            ))}
          </div>
        </div>
        <AutonomousProposalEvidenceGrid proposal={proposal} />
        <BlockerDebatePreview proposal={proposal} />
      </CardContent>
      {editingProposal?.id === proposal.id && (
        <CardContent className="border-t border-border/60 pt-4">
          <EditBeforeDevelopBlock
            busy={updating === `${proposal.id}:preview-plan`}
            form={editForm}
            onCancel={onCancelEdit}
            onChange={onChangeEdit}
            onConfirm={onConfirmEdit}
            proposal={proposal}
          />
        </CardContent>
      )}
    </Card>
  );
}

function StandbyProposalsBlock({
  editingProposal,
  editForm,
  onAction,
  onCancelEdit,
  onChangeEdit,
  onConfirmEdit,
  pulseSummary,
  updating,
}: {
  editingProposal: TeamProposal | null;
  editForm: { acceptance: string; title: string; whyNow: string };
  onAction: (proposal: TeamProposal, action: MatureProposalAction) => Promise<void>;
  onCancelEdit: () => void;
  onChangeEdit: (form: { acceptance: string; title: string; whyNow: string }) => void;
  onConfirmEdit: () => Promise<void>;
  pulseSummary?: TeamPulseSummary | null;
  updating: string | null;
}) {
  const proposals = pulseSummary?.standby_proposals ?? [];
  if (proposals.length === 0) return null;
  return (
    <section className="rounded-xl border border-border/70 bg-card/70 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Inbox className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Standby — interessano, non ora</h3>
        <Badge tone="secondary">non bloccano nuove proposte</Badge>
      </div>
      <div className="mt-3 grid gap-3">
        {proposals.map((proposal) => (
          <Card key={proposal.id}>
            <CardContent className="grid gap-3 py-4">
              <div className="grid gap-3 lg:grid-cols-[1fr_1fr_0.65fr_auto] lg:items-start">
                <Field label="Proposta" value={proposal.title} />
                <Field label="Motivo standby" value={proposal.chief_review_reason ?? "Accettata: interessante, non da sviluppare ora."} />
                <Field label="Formulata" value={formatProposalFormulatedDate(proposal)} />
                <div className="flex flex-wrap gap-2 lg:justify-end">
                  <Button disabled={Boolean(updating)} size="sm" onClick={() => void onAction(proposal, "develop")}>Sviluppa</Button>
                  <Button disabled={Boolean(updating)} ghost size="sm" onClick={() => void onAction(proposal, "reject")}>Scarta</Button>
                </div>
              </div>
              <AutonomousProposalEvidenceGrid proposal={proposal} />
              <BlockerDebatePreview proposal={proposal} />
            </CardContent>
            {editingProposal?.id === proposal.id && (
              <CardContent className="border-t border-border/60 pt-4">
                <EditBeforeDevelopBlock
                  busy={updating === `${proposal.id}:preview-plan`}
                  form={editForm}
                  onCancel={onCancelEdit}
                  onChange={onChangeEdit}
                  onConfirm={onConfirmEdit}
                  proposal={proposal}
                />
              </CardContent>
            )}
          </Card>
        ))}
      </div>
    </section>
  );
}

function BlockerDebatePreview({ proposal }: { proposal: TeamProposal }) {
  const verifiedOwners = verifiedSuggestedProfiles(proposal);
  const hiddenInvalidCount = unverifiedSuggestedProfilesCount(proposal);
  return (
    <section className="rounded-xl border border-warning/40 bg-warning/5 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <AlertTriangle className="h-5 w-5 text-warning" />
        <h4 className="font-semibold">Blocker debate prima della conversione Kanban</h4>
        <Badge tone="warning">preview read-only</Badge>
        <Badge tone="secondary">owner registry-safe</Badge>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Daniele vede supporto, critica, dipendenze e owner prima della conversione. Questa sezione non crea task, cron o invii esterni.
      </p>
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <Field label="Titolo proposta" value={proposal.title} />
        <Field label="Evidenza / origine" value={formatProposalEvidence(proposal)} />
        <Field label="Supporter · perché convertirla" value={formatProposalViewpoint(proposal.supporter ?? proposal.supporter_view, proposal.challenge?.supporter, proposal.challenge?.support)} />
        <Field label="Critic / blocker · cosa può fermarla" value={formatProposalViewpoint(proposal.critic ?? proposal.critic_view, proposal.challenge?.critic, proposal.challenge?.challenge)} />
        <Field label="Chief synthesis · raccomandazione" value={formatProposalContractValue(proposal.chief_synthesis ?? proposal.challenge?.chief_synthesis)} />
        <Field label="Dipendenze, rischi e gate" value={formatProposalDependencies(proposal)} />
      </div>
      <div className="mt-3 rounded-sm border border-border/70 bg-background/35 p-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Owner raccomandati verificati</div>
        {verifiedOwners.length === 0 ? (
          <p className="mt-2 text-sm text-warning">Nessun owner verificato nel payload: conversione da tenere bloccata finché il registry non valida i profili.</p>
        ) : (
          <div className="mt-2 flex flex-wrap gap-2">
            {verifiedOwners.map((owner) => (
              <Badge key={`${proposal.id}-${owner.profile}-${owner.role}`} tone="success">
                {owner.profile} · {owner.role} · {owner.confidence}
              </Badge>
            ))}
          </div>
        )}
        {hiddenInvalidCount > 0 && (
          <p className="mt-2 text-xs text-warning">{hiddenInvalidCount} suggerimento/i non verificato/i nascosto/i: non verranno proposti come assignee.</p>
        )}
        {verifiedOwners.length > 0 && (
          <div className="mt-2 grid gap-1 text-xs text-muted-foreground">
            {verifiedOwners.map((owner) => (
              <div key={`${proposal.id}-${owner.profile}-reason`}>{owner.profile}: {owner.reason}</div>
            ))}
          </div>
        )}
      </div>
      <div className="mt-3 rounded-sm border border-primary/30 bg-primary/5 p-3 text-sm">
        Gate esplicito: il bottone “Sviluppa” salva solo la copia e apre la preview del piano. La creazione Kanban richiede poi conferma nel dialog con preview hash.
      </div>
    </section>
  );
}

function CommandCenterPlanConfirmationDialog({
  busy,
  onCancel,
  onConfirm,
  pending,
}: {
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
  pending: PendingPlanConfirmation;
}) {
  const { proposal, preview } = pending;
  const tasks = preview.plan.tasks ?? [];
  const allAssignees = [preview.plan.assignee, ...tasks.map((task) => task.assignee)].filter(Boolean).join(", ") || "n/d";
  return (
    <section className="rounded-xl border border-primary/50 bg-card p-4 shadow-lg">
      <div className="flex flex-wrap items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Conferma esplicita prima di creare il piano Kanban</h3>
        <Badge tone="warning">Daniele confirmation gate</Badge>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        Preview read-only generata per “{proposal.title}”. Finché non premi conferma qui sotto, nessun task Kanban viene creato e non parte alcun cron/invio esterno.
      </p>
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <Field label="Preview hash" value={preview.preview_hash} />
        <Field label="Owner proposti dal piano" value={allAssignees} />
        <Field label="Parent task preview" value={`${preview.plan.title}\nAssignee: ${preview.plan.assignee ?? "n/d"}\nInitial status: ${preview.plan.initial_status}\nPriority: ${preview.plan.priority}`} />
        <Field label="Task figli preview" value={tasks.map((task, index) => `${index + 1}. ${task.title}\nAssignee: ${task.assignee ?? "n/d"} · ${task.assignment_reason ?? "reason n/d"}`).join("\n\n") || "Nessun task figlio nel preview."} />
      </div>
      <div className="mt-3 rounded-sm border border-warning/50 bg-warning/10 p-3 text-sm text-warning">
        Confermando autorizzi solo la conversione di questo preview hash su board corrente. Il backend rivalida hash e gate prima di creare task ready.
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button disabled={busy} onClick={() => void onConfirm()} size="sm">
          {busy ? <Spinner className="mr-1 h-3 w-3" /> : null}
          Conferma e prepara piano Kanban
        </Button>
        <Button disabled={busy} ghost onClick={onCancel} size="sm">Annulla: resta preview</Button>
      </div>
    </section>
  );
}

function EditBeforeDevelopBlock({
  busy,
  form,
  onCancel,
  onChange,
  onConfirm,
  proposal,
}: {
  busy: boolean;
  form: { acceptance: string; title: string; whyNow: string };
  onCancel: () => void;
  onChange: (form: { acceptance: string; title: string; whyNow: string }) => void;
  onConfirm: () => Promise<void>;
  proposal: TeamProposal;
}) {
  return (
    <section className="rounded-xl border border-primary/40 bg-card/80 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Sparkles className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Modifica prima di sviluppare</h3>
        <Badge tone="warning">{proposal.id}</Badge>
      </div>
      <div className="mt-3 grid gap-3">
        <label className="grid gap-1 text-sm">
          <span className="font-medium">Titolo</span>
          <input className="rounded-md border border-border bg-background px-3 py-2" value={form.title} onChange={(event) => onChange({ ...form, title: event.target.value })} />
        </label>
        <label className="grid gap-1 text-sm">
          <span className="font-medium">Perché ora</span>
          <textarea className="min-h-24 rounded-md border border-border bg-background px-3 py-2" value={form.whyNow} onChange={(event) => onChange({ ...form, whyNow: event.target.value })} />
        </label>
        <label className="grid gap-1 text-sm">
          <span className="font-medium">Criterio di accettazione / brief sviluppo</span>
          <textarea className="min-h-24 rounded-md border border-border bg-background px-3 py-2" value={form.acceptance} onChange={(event) => onChange({ ...form, acceptance: event.target.value })} />
        </label>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button disabled={busy} onClick={() => void onConfirm()} size="sm">
          {busy ? <Spinner className="mr-1 h-3 w-3" /> : null}
          Salva e mostra preview piano
        </Button>
        <Button disabled={busy} ghost onClick={onCancel} size="sm">Annulla</Button>
      </div>
    </section>
  );
}

function HealthBlock({ decisions, tasks }: { decisions: number; tasks: KanbanTask[] }) {
  const realTasks = tasks.filter((task) => !isTestTask(task));
  return (
    <section className="rounded-xl border border-border/70 bg-card/70 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <CheckCircle2 className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Salute del team</h3>
        <Badge tone="warning">in rodaggio</Badge>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-4">
        <Metric label="Tasso approvazione" value="n/d" />
        <Metric label="Proposte pronte" value={String(decisions)} />
        <Metric label="Tempo medio" value="n/d" />
        <Metric label="Trend" value={realTasks.length > 0 ? "in rodaggio" : "n/d"} />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">Nessuna metrica inventata: i valori restano n/d finché Kanban non produce dati reali.</p>
    </section>
  );
}

function HandoffBlock({ currentBoard, incoming, outgoing }: { currentBoard: string; incoming: KanbanTask[]; outgoing: KanbanTask[] }) {
  const siblingBoard = currentBoard === "sviluppo-hermes" ? "team-operativo" : "sviluppo-hermes";
  const items = [
    ...incoming.map((task) => ({ task, board: currentBoard, from: siblingBoard, to: currentBoard, direction: "In arrivo" })),
    ...outgoing.map((task) => ({ task, board: siblingBoard, from: currentBoard, to: siblingBoard, direction: "In uscita" })),
  ];
  return (
    <details className="rounded-xl border border-border/70 bg-card/70 p-4">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2 font-semibold">
        <GitBranch className="h-5 w-5 text-primary" />
        Handoff
        <Badge tone="secondary">{items.length} aperti</Badge>
      </summary>
      <div className="mt-2 text-xs text-muted-foreground">
        Mostro solo handoff ancora attivi: done/archived restano nell'audit Kanban, non nella coda operativa.
      </div>
      <div className="mt-3 grid gap-2">
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nessun handoff aperto.</p>
        ) : (
          items.map(({ board, task, direction, from, to }) => (
            <Card key={`${direction}-${task.id}`}>
              <CardContent className="py-3 text-sm">
                <div className="grid gap-2 md:grid-cols-[0.8fr_0.8fr_1.4fr_0.7fr_auto] md:items-center">
                  <Field label="Da chi" value={`${direction} · ${from}`} />
                  <Field label="Verso chi" value={to} />
                  <Field label="Perché" value={task.latest_summary ?? extractLine(task, ["problema", "perché conta", "proposta minima"], task.title)} />
                  <Field label="Stato" value={task.status} />
                  <Button ghost size="sm" onClick={() => { openKanbanTask(board, task.id); }}>
                    Link card {task.id}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </details>
  );
}

function DeepLinksBlock({ config }: { config: TeamCommandConfig }) {
  return (
    <details className="rounded-xl border border-border/70 bg-card/70 p-4">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2 font-semibold">
        <Inbox className="h-5 w-5 text-primary" />
        Approfondimenti
      </summary>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <DeepLink icon={<ShieldCheck className="h-4 w-4" />} title="Kanban completo" body={`Apri il plugin Kanban e filtra la board ${config.board}.`} href={kanbanBoardHref(config.board)} />
        <DeepLink icon={<Sparkles className="h-4 w-4" />} title="Archivio contenuto precedente" body="Registry legacy, Radar, Chief queue, PM workspace e review duplicate." href={config.archiveHref} />
        <DeepLink icon={<Bot className="h-4 w-4" />} title="Mission Control V2" body="Esecuzione, blocker, output review e dispatch separati dal cockpit decisionale." href={kanbanBoardHref(config.board, "/v2")} />
        <DeepLink icon={<ArrowRight className="h-4 w-4" />} title="File stato/log" body="Stato e log restano nel Document Vault; la UI non inventa metriche." href="/files" />
      </div>
    </details>
  );
}

function openDeepLink(href: string): void {
  const match = href.match(/[?&]board=([^&]+)/);
  if (match) {
    rememberKanbanBoard(decodeURIComponent(match[1]));
  }
  window.location.href = href;
}

function DeepLink({ body, href, icon, title }: { body: string; href: string; icon: ReactNode; title: string }) {
  return (
    <button
      type="button"
      className="rounded-sm border border-border/70 bg-background/35 p-3 text-left transition hover:border-primary/50"
      onClick={() => { openDeepLink(href); }}
    >
      <div className="flex items-center gap-2 text-sm font-semibold">{icon}{title} <ArrowRight className="ml-auto h-4 w-4" /></div>
      <p className="mt-1 text-xs text-muted-foreground">{body}</p>
    </button>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 whitespace-pre-line text-sm text-foreground">{value}</div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/60 bg-background/40 p-3">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-base font-semibold">{value}</div>
    </div>
  );
}
