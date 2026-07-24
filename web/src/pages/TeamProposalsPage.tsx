import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Brain,
  CheckCircle2,
  CircleDot,
  Gauge,
  Lightbulb,
  PauseCircle,
  Radar,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  TeamProposal,
  TeamProposalStatus,
  TeamProposalTaskPreview,
  TeamProposalPlanPreview,
  TeamProposalUpsertRequest,
  TeamProposalsResponse,
  TeamConstitutionContract,
  TeamPulseSummary,
  TeamPulseControversialProposal,
  TeamPulseControversyLane,
  TeamPulseControversyRiskLevel,
  TeamSpecialist,
  AgentGrowthField,
  AgentGrowthProfile,
  KanbanTaskDetailResponse,
  RadarHermesProposal,
  RadarHermesSnapshot,
} from "@/lib/api";

type Priority = TeamProposal["priority"];
type Recommendation = TeamProposal["recommendation"];
type TeamProposalsMode = "autonomous" | "operative" | "evolution" | "pm";

interface NowAction {
  title: string;
  body: string;
  gate: string;
  proposal?: TeamProposal;
  primaryAction?: "plan-preview" | "task-preview" | "approve" | "pulse";
}

const MODE_COPY: Record<TeamProposalsMode, {
  title: string;
  description: string;
  primaryKindLabel: string;
  chiefTitle: string;
  chiefRecommendation: string;
  chiefBody: string;
}> = {
  autonomous: {
    title: "Team & Proposte",
    description:
      "Pagina autonoma del consiglio operativo di Hermes: raccoglie segnali, proposte operative ed evolutive, challenge supporter/critic e raccomandazione Chief prima di qualunque task Kanban.",
    primaryKindLabel: "Proposte operative ed evolutive",
    chiefTitle: "Chief recommendation",
    chiefRecommendation: "Mostrare solo 1–3 decisioni ad alto valore, con evidenza e controargomento visibili.",
    chiefBody:
      "Status, approve, park e discard restano registry-only. Preview piano/task è read-only. Solo conferma esplicita crea task Kanban ready con assignee reali e motivazione visibile.",
  },
  operative: {
    title: "Team Operativo",
    description:
      "Pagina in cui il team di subagenti lavora sulle task operative del tuo lavoro: segnali, blocker, evidenze, proposte operative e piani da portare in Mission Control.",
    primaryKindLabel: "Proposte operative",
    chiefTitle: "Chief operativo",
    chiefRecommendation: "Ridurre rumore e trasformare segnali operativi in poche decisioni utili.",
    chiefBody:
      "Qui entrano solo proposte legate al lavoro reale: CO2Farm, documenti, evidenze, task bloccate, follow-up e priorità operative. La conversione in task resta approval-gated.",
  },
  evolution: {
    title: "Sviluppo Hermes",
    description:
      "Spazio unico per il team sviluppo Hermes: Hermes PM governa roadmap, subagenti tecnici, radar evolutivo, challenge interna, dashboard, automazioni, affidabilità e nuove capacità del sistema.",
    primaryKindLabel: "Sviluppo Hermes",
    chiefTitle: "Hermes PM — direzione sviluppo",
    chiefRecommendation: "Governare una roadmap corta e verificabile: poche proposte, challenge interna e lancio solo quando il piano è chiaro.",
    chiefBody:
      "Qui entrano solo proposte sul sistema Hermes: crescita subagenti, nuove funzionalità, radar evolutivo, reliability, UX e governance. È separato dal lavoro operativo di Daniele, ma condivide la stessa vision low-noise e quality-first.",
  },
  pm: {
    title: "Hermes PM",
    description:
      "Pagina in cui lavori con HermesPM: il project manager dedicato all'evoluzione interna di Hermes, Mission Control, Kanban, subagenti e governance dell'autonomia.",
    primaryKindLabel: "Proposte Hermes PM",
    chiefTitle: "HermesPM",
    chiefRecommendation: "Governare l'evoluzione interna con una roadmap corta, challenge interna e gate espliciti.",
    chiefBody:
      "Qui non partono worker automaticamente: HermesPM raccoglie segnali, sfida le priorità, prepara specifiche e porta a Daniele solo decisioni ad alto valore.",
  },
};

const EMPTY_PROPOSAL_FORM: TeamProposalUpsertRequest = {
  title: "",
  kind: "evolution",
  origin: "chief/manual",
  category: "Dashboard/UI",
  whyNow: "",
  evidence: "",
  benefit: "medium",
  effort: "medium",
  risk: "low",
  priority: "P2",
  confidence: "medium",
  recommendation: "prepare",
  acceptance: "",
  source_key: "",
};

const EMPTY_PROPOSALS: TeamProposal[] = [];
const EMPTY_SPECIALISTS: TeamSpecialist[] = [];

const RADAR_LEVEL_LABEL: Record<string, string> = {
  high: "alto",
  medium: "medio",
  low: "basso",
};

const RADAR_APPROVAL_LABEL: Record<string, string> = {
  candidate: "Candidate",
  needs_review: "Da rivedere",
  approved_for_spec: "Spec approvata",
  approved_for_kanban: "Kanban autorizzato",
  parked: "Parcheggiata",
  rejected: "Scartata",
  done: "Completata",
};

function formatRadarDate(value?: string | null): string {
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
  return formatRadarDate(proposal.formulated_at ?? proposal.created_at ?? proposal.status_updated_at ?? proposal.updated_at ?? proposal.last_signal_at);
}

function toneForPriority(priority: Priority): "destructive" | "warning" | "success" | "secondary" {
  if (priority === "P0") return "destructive";
  if (priority === "P1") return "warning";
  if (priority === "P2") return "success";
  return "secondary";
}

function recommendationLabel(r: Recommendation): string {
  switch (r) {
    case "do_now":
      return "Fare ora";
    case "prepare":
      return "Preparare";
    case "park":
      return "Parcheggiare";
    case "reject":
      return "Scartare";
  }
}

function kindLabel(kind: TeamProposal["kind"]): string {
  return kind === "evolution" ? "Sviluppo Hermes" : "Operativa";
}

function statusTone(status: TeamSpecialist["status"]): "success" | "warning" | "secondary" {
  if (status === "active") return "success";
  if (status === "watching") return "warning";
  return "secondary";
}

function proposalStatusTone(status: TeamProposalStatus): "success" | "warning" | "destructive" | "secondary" {
  if (status === "approvata" || status === "trasformata_in_task") return "success";
  if (status === "raccomandata") return "warning";
  if (status === "scartata") return "destructive";
  return "secondary";
}

function statusLabel(status: TeamProposalStatus): string {
  switch (status) {
    case "proposta":
      return "Proposta";
    case "raccomandata":
      return "Raccomandata";
    case "approvata":
      return "Approvata";
    case "standby":
      return "Standby";
    case "parcheggiata":
      return "Parcheggiata";
    case "scartata":
      return "Scartata";
    case "trasformata_in_task":
      return "Trasformata in task";
    case "signal_detected":
      return "Segnale";
    case "interpreting":
      return "Interpretazione";
    case "challenging":
      return "Challenge";
    case "synthesized":
      return "Sintetizzata";
    case "needs_reliability_check":
      return "Reliability check";
    case "ready_for_gate":
      return "Pronta per gate";
    case "approved_min_step":
      return "Prossimo passo approvato";
    case "blocked_by_daniele":
      return "Bloccata da Daniele";
    case "parked":
      return "Parcheggiata";
    case "rejected":
      return "Scartata";
    case "converted_to_kanban":
      return "Convertita in Kanban";
    case "archived":
      return "Archiviata";
  }
}

function isActiveProposal(proposal: TeamProposal): boolean {
  return !["standby", "scartata", "trasformata_in_task", "converted_to_kanban", "archived", "rejected", "parked"].includes(proposal.status);
}

function hasOperationalChallengeContract(proposal: TeamProposal): boolean {
  return Boolean(
    proposal.kind === "operative" &&
      isActiveProposal(proposal) &&
      (proposal.source_agent || proposal.suggested_next_action) &&
      (proposal.source_signal || proposal.whyNow) &&
      proposal.evidence &&
      proposal.challenge,
  );
}

function reviewLabel(proposal: TeamProposal): string {
  if (proposal.chief_review_status === "pending") return "In review Chief";
  if (proposal.chief_review_status === "shortlisted") return "Shortlist Chief";
  if (proposal.chief_review_status === "deferred") return "Differita";
  if (proposal.chief_review_status === "rejected") return "Respinta";
  if (proposal.status === "parcheggiata") return "Parcheggiata";
  return "Review mancante";
}

function riskGateLabel(proposal: TeamProposal): string {
  const text = `${proposal.evidence ?? ""} ${proposal.whyNow ?? ""} ${proposal.challenge?.challenge ?? ""} ${proposal.challenge?.veto_risk ?? ""}`.toLowerCase();
  if (/soc|boundary|claim|legal|contratt|mrv/.test(text)) return "Gate MRV/Legal";
  if (/cron|gateway|dashboard|tecnic|dipenden|stabilit/.test(text)) return "Dipendenza tecnica";
  if (!proposal.evidence) return "Evidenza debole";
  return "Low-noise";
}

const CONTROVERSY_STATUS_COPY: Record<TeamPulseControversyLane["status"], { label: string; tone: "success" | "warning" | "destructive" | "secondary"; message: string }> = {
  has_controversy: {
    label: "Criticità aperta",
    tone: "warning",
    message: "Challenge utile prima del consenso: una proposta resta potenzialmente utile, ma contiene un punto da decidere o documentare.",
  },
  no_meaningful_controversy: {
    label: "Nessuna criticità materiale",
    tone: "success",
    message: "Le proposte revisionate non mostrano obiezioni materiali, rischi veto o gap evidenza aperti.",
  },
  insufficient_review_data: {
    label: "Revisione incompleta",
    tone: "warning",
    message: "Non trattare l'assenza di criticità come consenso: mancano critic/review data completi.",
  },
};

const CONTROVERSY_DECISION_COPY: Record<TeamPulseControversialProposal["chief_synthesis"]["decision_state"], string> = {
  unresolved: "Chief: da decidere",
  accepted_with_caveats: "Chief: caveat accettati",
  rejected: "Chief: respinta",
  needs_more_evidence: "Chief: serve evidenza",
  deferred: "Chief: differita",
};

const CONTROVERSY_ACTION_COPY: Record<TeamPulseControversialProposal["recommended_action"]["action"], string> = {
  include_in_shortlist: "Includere in shortlist",
  include_with_caveat: "Includere con caveat",
  revise_before_shortlist: "Revisionare prima della shortlist",
  request_evidence: "Richiedere evidenza",
  route_to_specialist: "Instradare a specialista",
  defer: "Differire",
  reject: "Scartare",
};

function controversyRiskTone(level: TeamPulseControversyRiskLevel): "success" | "warning" | "destructive" | "secondary" {
  if (level === "L3_blocking_veto" || level === "L4_stop_escalate") return "destructive";
  if (level === "L2_material_risk") return "warning";
  if (level === "L1_objection") return "secondary";
  return "success";
}

function controversyRiskShortLabel(level: TeamPulseControversyRiskLevel): string {
  return level.replace("L0_none", "L0").replace("L1_objection", "L1").replace("L2_material_risk", "L2").replace("L3_blocking_veto", "L3").replace("L4_stop_escalate", "L4");
}

function sourceRefLabel(ref: { kind: string; id: string }): string {
  return `${ref.kind}:${ref.id}`;
}

function formatControversyDate(value?: string | null): string {
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

export function TeamOperationsPage() {
  return <TeamProposalsPage mode="operative" />;
}

export function TeamEvolutionPage() {
  return <TeamProposalsPage mode="evolution" />;
}

export function TeamPmPage() {
  return <TeamProposalsPage mode="pm" />;
}

function validTeamProposalsMode(value: string | null): TeamProposalsMode | null {
  if (value === "autonomous") return value;
  if (value === "operative" || value === "evolution") return value;
  if (value === "pm") return "evolution";
  return null;
}

export default function TeamProposalsPage({ mode: initialMode = "autonomous" }: { mode?: TeamProposalsMode }) {
  const [searchParams] = useSearchParams();
  const queryMode = validTeamProposalsMode(searchParams.get("tab"));
  const mode = queryMode ?? initialMode;
  const isAutonomousMode = mode === "autonomous";
  const proposalKindMode: TeamProposal["kind"] = mode === "operative" ? "operative" : "evolution";
  const siblingHref = mode === "operative" ? "/team-evolution" : "/team-work";
  const siblingLabel = mode === "operative" ? "Apri Sviluppo Hermes" : "Apri Team Operativo";
  const [data, setData] = useState<TeamProposalsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updating, setUpdating] = useState<string | null>(null);
  const [previewFor, setPreviewFor] = useState<TeamProposal | null>(null);
  const [taskPreview, setTaskPreview] = useState<TeamProposalTaskPreview | null>(null);
  const [taskPreviewHash, setTaskPreviewHash] = useState<string | null>(null);
  const [planPreviewFor, setPlanPreviewFor] = useState<TeamProposal | null>(null);
  const [planPreview, setPlanPreview] = useState<TeamProposalPlanPreview | null>(null);
  const [planPreviewHash, setPlanPreviewHash] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [showNewProposal, setShowNewProposal] = useState(false);
  const [proposalForm, setProposalForm] = useState<TeamProposalUpsertRequest>({ ...EMPTY_PROPOSAL_FORM, kind: proposalKindMode });
  const [collectorStatus, setCollectorStatus] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [radarSnapshot, setRadarSnapshot] = useState<RadarHermesSnapshot | null>(null);
  const [radarLoading, setRadarLoading] = useState(true);
  const [radarError, setRadarError] = useState<string | null>(null);

  const refreshRadarHermes = async () => {
    setRadarLoading(true);
    setRadarError(null);
    try {
      setRadarSnapshot(await api.getRadarHermes());
    } catch (e) {
      setRadarError(String(e));
    } finally {
      setRadarLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    api
      .getTeamProposals()
      .then((response) => {
        if (!cancelled) setData(response);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    api.getRadarHermes()
      .then((response) => {
        if (!cancelled) setRadarSnapshot(response);
      })
      .catch((e) => {
        if (!cancelled) setRadarError(String(e));
      })
      .finally(() => {
        if (!cancelled) setRadarLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const allProposals = data?.proposals ?? EMPTY_PROPOSALS;
  const proposals = useMemo(
    () => allProposals.filter((p) => (isAutonomousMode || p.kind === proposalKindMode) && isActiveProposal(p)),
    [allProposals, isAutonomousMode, proposalKindMode],
  );
  const archivedProposals = useMemo(
    () => allProposals.filter((p) => (isAutonomousMode || p.kind === proposalKindMode) && !isActiveProposal(p)),
    [allProposals, isAutonomousMode, proposalKindMode],
  );
  const specialists = data?.specialists ?? EMPTY_SPECIALISTS;
  const recommended = useMemo(
    () => proposals.filter((p) => p.recommendation === "do_now" || p.status === "raccomandata"),
    [proposals],
  );
  const evolution = useMemo(() => proposals.filter((p) => p.kind === "evolution"), [proposals]);
  const operative = useMemo(() => proposals.filter((p) => p.kind === "operative"), [proposals]);
  const chiefQueue = (data?.chief_review?.queue ?? recommended.slice(0, 5)).filter((p) => isAutonomousMode || p.kind === proposalKindMode);
  const strategicReview = data?.strategic_review;
  const copy = MODE_COPY[mode];
  const constitution = data?.team_constitutions?.[proposalKindMode];
  const isEvolutionMode = mode === "evolution" || isAutonomousMode;
  const isPmMode = mode === "pm";
  const modePulse = !isPmMode
    ? data?.team_pulses?.[proposalKindMode] ?? (data?.team_pulse?.kind === proposalKindMode ? data.team_pulse : undefined)
    : undefined;
  const operationalActive = useMemo(
    () => allProposals.filter((p) => p.kind === "operative" && isActiveProposal(p)),
    [allProposals],
  );
  const operationalChallengeated = useMemo(
    () => operationalActive.filter(hasOperationalChallengeContract),
    [operationalActive],
  );
  const operationalLegacy = useMemo(
    () => operationalActive.filter((p) => !hasOperationalChallengeContract(p)),
    [operationalActive],
  );
  const pendingOrDefaultedReviews = useMemo(
    () => allProposals.filter((p) => p.chief_review_status === "pending"),
    [allProposals],
  );
  const nowActions = useMemo(
    () => buildNowActions({ mode, chiefQueue, modePulse, proposals }),
    [chiefQueue, mode, modePulse, proposals],
  );

  const replaceProposal = (proposal: TeamProposal) => {
    const replaceInList = (items: TeamProposal[]) =>
      items.map((p) => (p.id === proposal.id ? proposal : p));
    setData((current) => {
      if (!current) return current;
      return {
        ...current,
        updated_at: new Date().toISOString(),
        proposals: replaceInList(current.proposals),
        chief_review: current.chief_review
          ? { ...current.chief_review, queue: replaceInList(current.chief_review.queue) }
          : current.chief_review,
        strategic_review: current.strategic_review
          ? {
              ...current.strategic_review,
              top_operative: replaceInList(current.strategic_review.top_operative),
              top_evolution: replaceInList(current.strategic_review.top_evolution),
              parked: replaceInList(current.strategic_review.parked),
            }
          : current.strategic_review,
      };
    });
  };

  const setProposalStatus = async (proposal: TeamProposal, status: TeamProposalStatus) => {
    setUpdating(`${proposal.id}:${status}`);
    setError(null);
    setActionNotice(null);
    try {
      const response = await api.setTeamProposalStatus(proposal.id, status);
      replaceProposal(response.proposal);
      setActionNotice(
        `Proposta “${response.proposal.title}” aggiornata a ${statusLabel(response.proposal.status)}. ` +
        "Non è partito nessun worker: per creare task usa “Trasforma in task…” o “Trasforma in piano…”.",
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  const reviewAsChief = async (proposal: TeamProposal, action: "shortlist" | "defer" | "reject") => {
    setUpdating(`${proposal.id}:chief-${action}`);
    setError(null);
    try {
      const response = await api.reviewTeamProposalAsChief(proposal.id, action);
      replaceProposal(response.proposal);
      setData((current) => current ? { ...current, chief_review: { queue: response.chief_review.queue, summary: current.chief_review?.summary ?? "Shortlist Chief" } } : current);
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  const openTaskPreview = async (proposal: TeamProposal) => {
    setPreviewFor(proposal);
    setTaskPreview(null);
    setTaskPreviewHash(null);
    setPreviewLoading(true);
    setError(null);
    try {
      const response = await api.getTeamProposalTaskPreview(proposal.id);
      setTaskPreview(response.task);
      setTaskPreviewHash(response.preview_hash);
    } catch (e) {
      setError(String(e));
      setPreviewFor(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const confirmConvertToTask = async () => {
    if (!previewFor || !taskPreviewHash) return;
    setUpdating(`${previewFor.id}:convert`);
    setError(null);
    try {
      await api.approveTeamProposalMinStep(previewFor.id, {
        action_type: "task",
        confirmed_preview_hash: taskPreviewHash,
        note: "Approvato da Daniele in Team & Proposte: crea 1 task Kanban ready; dispatcher avviabile secondo priorità.",
      });
      const response = await api.convertTeamProposalToTask(previewFor.id, taskPreviewHash);
      replaceProposal(response.proposal);
      setActionNotice(`Task “${response.task.title}” creata in Kanban con stato ${response.task.status ?? "ready"}. Assignee scelto automaticamente: ${taskPreview?.assignee ?? "default"}; il dispatcher può avviarla secondo priorità e concorrenza.`);
      setPreviewFor(null);
      setTaskPreview(null);
      setTaskPreviewHash(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  const openPlanPreview = async (proposal: TeamProposal) => {
    setPlanPreviewFor(proposal);
    setPlanPreview(null);
    setPlanPreviewHash(null);
    setPreviewLoading(true);
    setError(null);
    try {
      const response = await api.getTeamProposalPlanPreview(proposal.id);
      setPlanPreview(response.plan);
      setPlanPreviewHash(response.preview_hash);
    } catch (e) {
      setError(String(e));
      setPlanPreviewFor(null);
    } finally {
      setPreviewLoading(false);
    }
  };

  const confirmConvertToPlan = async () => {
    if (!planPreviewFor || !planPreviewHash) return;
    setUpdating(`${planPreviewFor.id}:convert-plan`);
    setError(null);
    try {
      await api.approveTeamProposalMinStep(planPreviewFor.id, {
        action_type: "plan",
        confirmed_preview_hash: planPreviewHash,
        note: "Approvato da Daniele in Team & Proposte: crea piano Kanban ready; dispatcher avviabile secondo priorità.",
      });
      const response = await api.convertTeamProposalToPlan(planPreviewFor.id, planPreviewHash);
      replaceProposal(response.proposal);
      setActionNotice(`Piano creato in Kanban ready: parent ${response.plan.parent_task_id}, ${response.plan.child_task_ids.length} task figli auto-assegnati a ${(response.plan.child_assignees ?? []).join(", ") || "profili disponibili"}. Il dispatcher può avviarli secondo priorità e concorrenza.`);
      setPlanPreviewFor(null);
      setPlanPreview(null);
      setPlanPreviewHash(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  const submitProposal = async () => {
    if (!proposalForm.title.trim()) {
      setError("Il titolo della proposta è obbligatorio");
      return;
    }
    setUpdating("proposal:create");
    setError(null);
    try {
      const response = await api.upsertTeamProposal({
        ...proposalForm,
        kind: proposalKindMode,
        source_key: proposalForm.source_key?.trim() || undefined,
      });
      setData((current) => {
        if (!current) return current;
        const exists = current.proposals.some((p) => p.id === response.proposal.id);
        return {
          ...current,
          proposals: exists
            ? current.proposals.map((p) => p.id === response.proposal.id ? response.proposal : p)
            : [response.proposal, ...current.proposals],
        };
      });
      setProposalForm({ ...EMPTY_PROPOSAL_FORM, kind: proposalKindMode });
      setShowNewProposal(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  const runAutonomousTeamPulse = async () => {
    setUpdating("team-pulse:generate");
    setError(null);
    setCollectorStatus(null);
    try {
      const response = await api.generateAutonomousTeamProposals(5, proposalKindMode);
      setData((current) => {
        if (!current) return current;
        const incoming = new Map([...response.created, ...response.updated].map((p) => [p.id, p]));
        const merged = current.proposals.map((p) => incoming.get(p.id) ?? p);
        for (const proposal of response.created) {
          if (!current.proposals.some((p) => p.id === proposal.id)) merged.unshift(proposal);
        }
        return {
          ...current,
          proposals: merged,
          team_pulse: response.team_pulse,
          team_pulses: { ...(current.team_pulses ?? {}), [proposalKindMode]: response.team_pulse } as TeamProposalsResponse["team_pulses"],
          updated_at: new Date().toISOString(),
        };
      });
      setCollectorStatus(
        `${proposalKindMode === "operative" ? "Team Pulse operativo" : "Team Pulse evolutivo"}: ${response.created_count} nuove proposte autonome, ${response.updated_count} aggiornate, ${response.team_pulse.challenged_count} con challenge interna.`,
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setUpdating(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-muted-foreground">
        <Spinner className="mr-2 text-xl text-primary" />
        Carico spazio team…
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 pb-10">
      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <H2>{copy.title}</H2>
            <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
              {copy.description}
            </p>
            {data?.updated_at && (
              <p className="mt-1 text-xs text-muted-foreground">Registro aggiornato: {data.updated_at}</p>
            )}
          </div>
          <Button
            ghost
            size="sm"
            onClick={() => { window.location.href = siblingHref; }}
            aria-label={siblingLabel}
          >
            {siblingLabel}
          </Button>
        </div>
        <div className="flex flex-wrap gap-2 rounded-sm border bg-muted/30 p-2">
          <Badge tone="secondary">Linea separata: {isAutonomousMode ? "Team & Proposte autonomo" : mode === "operative" ? "lavoro operativo" : "sviluppo interno Hermes"}</Badge>
          {isAutonomousMode && <Badge tone="success">Status/preview registry-only · conferma finale ready</Badge>}
          <span className="self-center text-xs text-muted-foreground">
            Apri, scegli una mossa, oppure vai agli approfondimenti: status/approve/park/discard non creano task; preview piano/task è no-side-effect; conferma finale crea Kanban ready senza cron, invii esterni o dispatch diretto.
          </span>
        </div>
        {collectorStatus && (
          <Card>
            <CardContent className="py-3 text-sm text-muted-foreground">{collectorStatus}</CardContent>
          </Card>
        )}
        {actionNotice && (
          <Card>
            <CardContent className="py-3 text-sm text-success" role="status" aria-live="polite">
              {actionNotice}
            </CardContent>
          </Card>
        )}
        {error && (
          <Card>
            <CardContent className="py-3 text-sm text-destructive">{error}</CardContent>
          </Card>
        )}
      </section>

      {!isPmMode && (
        <NowActionsPanel
          actions={nowActions}
          mode={mode}
          onApprove={(proposal) => setProposalStatus(proposal, "approvata")}
          onPulse={runAutonomousTeamPulse}
          onTaskPreview={openTaskPreview}
          onPlanPreview={openPlanPreview}
          updating={updating}
        />
      )}

      {!isPmMode && (
        <details className="rounded-lg border border-border/60 bg-muted/15 p-3">
          <summary className="cursor-pointer text-sm font-semibold text-muted-foreground">
            Approfondimenti e audit ({proposals.length} attive · {chiefQueue.length} in review · {archivedProposals.length} archiviate)
          </summary>
          <div className="mt-4 flex flex-col gap-6">
            <TeamConstitutionPanel constitution={constitution} mode={mode} />
            {isEvolutionMode && (
              <section className="flex flex-col gap-3 rounded-sm border border-border/60 bg-background/30 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold">Proposta manuale</h3>
                    <p className="text-xs text-muted-foreground">Opzione avanzata: resta fuori dalla vista principale.</p>
                  </div>
                  <Button
                    ghost
                    size="sm"
                    onClick={() => setShowNewProposal((v) => !v)}
                    disabled={updating === "proposal:create"}
                  >
                    {showNewProposal ? "Chiudi" : "Nuova proposta manuale"}
                  </Button>
                </div>
                {showNewProposal && (
                  <NewProposalForm
                    busy={updating === "proposal:create"}
                    form={proposalForm}
                    onChange={setProposalForm}
                    onSubmit={submitProposal}
                  />
                )}
              </section>
            )}
            {isEvolutionMode ? (
              <>
                <HermesPmWorkspacePanel
                  proposals={allProposals}
                  specialists={specialists}
                  strategicReview={strategicReview}
                  updatedAt={data?.updated_at}
                  onTaskPreview={openTaskPreview}
                  onPlanPreview={openPlanPreview}
                  updating={updating}
                />
                <RadarHermesSection
                  error={radarError}
                  loading={radarLoading}
                  onRefresh={refreshRadarHermes}
                  snapshot={radarSnapshot}
                />
              </>
            ) : (
              <OperationalReviewPanel
                active={operationalActive}
                challenged={operationalChallengeated}
                legacy={operationalLegacy}
                onPlanPreview={openPlanPreview}
                onStatus={setProposalStatus}
                pendingReviews={pendingOrDefaultedReviews.length}
                source={data?.source}
                safety={data?.safety}
                updatedAt={data?.updated_at}
                updating={updating}
              />
            )}

            {modePulse && (
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold">{isEvolutionMode ? "Team Pulse — evoluzione viva" : "Pulse operativo — team al lavoro"}</h3>
                  <Badge tone="warning">proposal-only</Badge>
                </div>
                <Card>
                  <CardContent className="flex flex-col gap-4 py-4">
                    <p className="text-sm text-muted-foreground">{modePulse.summary}</p>
                    <div className="grid gap-2 sm:grid-cols-4">
                      <Metric label="Autonome" value={String(modePulse.autonomous_count)} />
                      <Metric label="Challenge" value={String(modePulse.challenged_count)} />
                      <Metric label="Controverse" value={String(modePulse.controversial_count)} />
                      <Metric label="Ultimo pulse" value={modePulse.last_run_at ?? "mai"} />
                    </div>
                    <div className="rounded-sm border border-success/50 bg-success/10 p-3 text-sm text-success">
                      Guardrail: intake proposal-only; no cron, no webhook, no invii esterni. Alla conversione confermata, Kanban entra ready: il dispatcher parte secondo priorità e concorrenza.
                    </div>
                    <TeamPulseControversyLaneSection lane={modePulse.controversy_lane} />
                    <div className="grid gap-3 lg:grid-cols-2">
                      <StrategicList title="Top proposte autonome" proposals={modePulse.top_autonomous} />
                      <StrategicList title="Proposta controversa" proposals={modePulse.controversial} />
                    </div>
                  </CardContent>
                </Card>
              </section>
            )}

            <section className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
              <Card>
                <CardContent className="flex flex-col gap-4 py-5">
                  <div className="flex items-center gap-2">
                    <Brain className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">{copy.chiefTitle}</h3>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">Prossima evoluzione consigliata</p>
                    <p className="text-lg font-semibold">{copy.chiefRecommendation}</p>
                    <p className="text-sm text-muted-foreground">{copy.chiefBody}</p>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-4">
                    <Metric label="Priorità" value="P0" />
                    <Metric label="Beneficio" value="Alto" />
                    <Metric label="Effort" value="Medio" />
                    <Metric label="Rischio" value="Basso" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="flex flex-col gap-4 py-5">
                  <div className="flex items-center gap-2">
                    <Gauge className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">Proposal gates</h3>
                  </div>
                  {[
                    "Il Chief filtra e raccomanda: i subagenti non decidono da soli.",
                    "Le proposte evolutive valgono quanto quelle operative.",
                    "La conversione in task richiede preview e conferma; dopo la conferma entra ready e il dispatcher la avvia secondo priorità e concorrenza.",
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                      <span>{item}</span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </section>

            <section className="grid gap-4 xl:grid-cols-5">
              {specialists.map((role) => (
                <Card key={role.id}>
                  <CardContent className="flex h-full flex-col gap-3 py-4">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Bot className="h-4 w-4 text-primary" />
                        <h3 className="text-sm font-semibold">{role.name}</h3>
                      </div>
                      <Badge tone={statusTone(role.status)}>{role.status}</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">{role.mission}</p>
                    <RoleFact label="Osserva" value={role.observes} />
                    <RoleFact label="Segnale" value={role.currentSignal} />
                    <RoleFact label="Proposta" value={role.nextProposal} />
                    {role.metrics && (
                      <div className="grid gap-2 text-xs sm:grid-cols-2">
                        <Metric label="Trust" value={`${role.metrics.trust_score}/100`} />
                        <Metric label="Proposte" value={String(role.metrics.proposal_count)} />
                        <Metric label="Accettate" value={String(role.metrics.approved_count)} />
                        <Metric label="Scartate" value={String(role.metrics.rejected_count)} />
                      </div>
                    )}
                    {role.growth_profile && <AgentGrowthCard growth={role.growth_profile} />}
                    <div className="mt-auto pt-1 text-xs text-muted-foreground">
                      Confidenza: <span className="text-foreground">{role.confidence}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </section>

            {strategicReview && (
              <section className="flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold">Strategic Chief Review</h3>
                  <Badge tone="success">Low-noise</Badge>
                </div>
                <Card>
                  <CardContent className="flex flex-col gap-4 py-4">
                    <p className="text-sm text-muted-foreground">{strategicReview.summary}</p>
                    <div className="grid gap-2 sm:grid-cols-5">
                      <Metric label="Attive" value={String(strategicReview.counts.active)} />
                      <Metric label="Operative" value={String(strategicReview.counts.operative)} />
                      <Metric label="Evolutive" value={String(strategicReview.counts.evolution)} />
                      <Metric label="Parcheggiate" value={String(strategicReview.counts.parked)} />
                      <Metric label="Scartate" value={String(strategicReview.counts.rejected)} />
                    </div>
                    <div className="grid gap-3 lg:grid-cols-2">
                      <StrategicList
                        title={isEvolutionMode ? "Top evoluzione Hermes" : "Top operative"}
                        proposals={isEvolutionMode ? strategicReview.top_evolution : strategicReview.top_operative}
                      />
                      <StrategicList title="Parcheggiate / da rivalutare" proposals={strategicReview.parked.filter((p) => p.kind === proposalKindMode)} />
                    </div>
                  </CardContent>
                </Card>
              </section>
            )}

            <section className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Chief review queue</h3>
                <Badge tone="warning">Top {chiefQueue.length}</Badge>
              </div>
              <div className="grid gap-3">
                {chiefQueue.map((proposal) => (
                  <ProposalCard
                    key={`chief-${proposal.id}`}
                    proposal={proposal}
                    compact={false}
                    updating={updating}
                    onStatus={setProposalStatus}
                    onTaskPreview={openTaskPreview}
                    onPlanPreview={openPlanPreview}
                    onChiefReview={reviewAsChief}
                    readOnly={!isEvolutionMode}
                  />
                ))}
              </div>
            </section>

            <section className="grid gap-4">
              {isAutonomousMode ? (
                <div className="grid gap-4 lg:grid-cols-2">
                  <ProposalColumn
                    title="Proposte operative"
                    icon={<ShieldCheck className="h-5 w-5" />}
                    proposals={operative}
                    updating={updating}
                    onStatus={setProposalStatus}
                    onTaskPreview={openTaskPreview}
                    onPlanPreview={openPlanPreview}
                    onChiefReview={reviewAsChief}
                    readOnly={false}
                  />
                  <ProposalColumn
                    title="Proposte evolutive"
                    icon={<Wrench className="h-5 w-5" />}
                    proposals={evolution}
                    updating={updating}
                    onStatus={setProposalStatus}
                    onTaskPreview={openTaskPreview}
                    onPlanPreview={openPlanPreview}
                    onChiefReview={reviewAsChief}
                    readOnly={false}
                  />
                </div>
              ) : (
                <ProposalColumn
                  title={copy.primaryKindLabel}
                  icon={isEvolutionMode ? <Wrench className="h-5 w-5" /> : <ShieldCheck className="h-5 w-5" />}
                  proposals={isEvolutionMode ? evolution : operative}
                  updating={updating}
                  onStatus={setProposalStatus}
                  onTaskPreview={openTaskPreview}
                  onPlanPreview={openPlanPreview}
                  onChiefReview={reviewAsChief}
                  readOnly={!isEvolutionMode}
                />
              )}
            </section>

            <section className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Raccomandate dal Chief</h3>
              </div>
              <div className="grid gap-3">
                {recommended.map((proposal) => (
                  <ProposalCard
                    key={proposal.id}
                    proposal={proposal}
                    compact={false}
                    updating={updating}
                    onStatus={setProposalStatus}
                    onTaskPreview={openTaskPreview}
                    onPlanPreview={openPlanPreview}
                    onChiefReview={reviewAsChief}
                    readOnly={!isEvolutionMode}
                  />
                ))}
              </div>
            </section>

            {archivedProposals.length > 0 && (
              <section className="flex flex-col gap-2 rounded-lg border border-border/60 bg-muted/20 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-success" />
                  <h3 className="text-sm font-semibold">Archivio proposte chiuse</h3>
                  <Badge tone="secondary">{archivedProposals.length}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  Le proposte trasformate in piano/task o scartate sono fuori dalla coda attiva: restano qui solo come audit storico e fanno spazio a nuove proposte Team Pulse.
                </p>
              </section>
            )}
          </div>
        </details>
      )}

      {previewFor && (
        <TaskPreviewDialog
          busy={updating === `${previewFor.id}:convert`}
          loading={previewLoading}
          onCancel={() => {
            setPreviewFor(null);
            setTaskPreview(null);
          }}
          onConfirm={confirmConvertToTask}
          preview={taskPreview}
          proposal={previewFor}
        />
      )}

      {planPreviewFor && (
        <PlanPreviewDialog
          busy={updating === `${planPreviewFor.id}:convert-plan`}
          loading={previewLoading}
          onCancel={() => {
            setPlanPreviewFor(null);
            setPlanPreview(null);
          }}
          onConfirm={confirmConvertToPlan}
          preview={planPreview}
          proposal={planPreviewFor}
        />
      )}
    </div>
  );
}


function TeamPulseControversyLaneSection({ lane }: { lane?: TeamPulseControversyLane }) {
  if (!lane) {
    return (
      <section className="rounded-lg border border-dashed border-border/70 bg-background/25 p-4">
        <div className="flex flex-wrap items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          <h4 className="font-semibold">Prima della shortlist: criticità utili</h4>
          <Badge tone="secondary">contratto legacy assente</Badge>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Controversy lane non disponibile nel payload legacy. Usa la shortlist esistente, ma non dedurre consenso da un campo mancante.
        </p>
      </section>
    );
  }

  const statusCopy = CONTROVERSY_STATUS_COPY[lane.status];
  const isHighRisk = lane.items.some((item) => item.risk.level === "L3_blocking_veto" || item.risk.level === "L4_stop_escalate");

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-warning/30 bg-warning/5 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-warning" />
            <h4 className="font-semibold">Prima della shortlist: criticità utili</h4>
            <Badge tone={statusCopy.tone}>{statusCopy.label}</Badge>
            {isHighRisk && <Badge tone="destructive">High veto risk</Badge>}
          </div>
          <p className="text-sm text-muted-foreground">Challenge utile prima del consenso. Non è un registro di colpe: serve a decidere meglio.</p>
          <p className="text-xs text-muted-foreground">
            Policy: {lane.selection_policy_version} · Aggiornato {formatControversyDate(lane.generated_at)} · Cycle {lane.cycle_id}
          </p>
        </div>
        <Badge tone="secondary">{lane.items.length} item</Badge>
      </div>

      {lane.items.length === 0 ? (
        <div className="rounded-sm border border-border/70 bg-background/40 p-3 text-sm">
          <div className="font-medium">{lane.empty_state?.title ?? statusCopy.label}</div>
          <p className="mt-1 text-muted-foreground">{lane.empty_state?.message ?? statusCopy.message}</p>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span>reviewed_proposal_count: {lane.empty_state?.reviewed_proposal_count ?? "n/d"}</span>
            <span>review_completeness: {lane.empty_state?.review_completeness ?? "unknown"}</span>
          </div>
        </div>
      ) : (
        <div className="grid gap-3">
          {lane.items.map((item) => (
            <ControversyItemCard key={item.proposal_id} item={item} />
          ))}
        </div>
      )}
    </section>
  );
}

function ControversyItemCard({ item }: { item: TeamPulseControversialProposal }) {
  const sourceRefs = [
    ...(item.contested.source_ref ? [item.contested.source_ref] : []),
    ...item.critic_rationale.source_refs,
    ...(item.evidence_gap.current_evidence_refs ?? []),
    ...item.provenance.created_from,
  ];
  const uniqueSourceRefs = Array.from(new Map(sourceRefs.map((ref) => [sourceRefLabel(ref), ref])).values()).slice(0, 5);
  const criticNames = [item.critic.subagent_label, ...(item.additional_critics ?? []).map((critic) => critic.subagent_label)];
  const highRisk = item.risk.level === "L3_blocking_veto" || item.risk.level === "L4_stop_escalate";

  return (
    <Card>
      <CardContent className="flex flex-col gap-4 py-4">
        <div className="flex flex-wrap gap-2">
          <Badge tone={controversyRiskTone(item.risk.level)}>
            {item.risk.label} · {controversyRiskShortLabel(item.risk.level)}
          </Badge>
          <Badge tone={item.chief_synthesis.decision_state === "unresolved" || item.chief_synthesis.decision_state === "needs_more_evidence" ? "warning" : "secondary"}>
            {CONTROVERSY_DECISION_COPY[item.chief_synthesis.decision_state]}
          </Badge>
          {highRisk && <Badge tone="destructive">non shortlist senza risoluzione</Badge>}
        </div>

        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Proposta contestata</div>
          <h5 className="mt-1 text-base font-semibold">{item.proposal_title || `Proposta senza titolo · ${item.proposal_id}`}</h5>
          {item.proposal_summary && <p className="mt-1 text-sm text-muted-foreground">{item.proposal_summary}</p>}
          <p className="mt-1 text-xs text-muted-foreground">Proposer: {item.proposer.subagent_label}</p>
        </div>

        <div className="grid gap-3 lg:grid-cols-2">
          <FieldBlock label="Critic · Obiezione sollevata da" value={criticNames.join(" + ")} />
          <FieldBlock label="Why contested · Perché è contestata" value={`Punto contestato: ${item.contested.text}\nTipo: ${item.contested.type}\nRazionale critic: ${item.critic_rationale.summary}\nGap evidenza: ${item.evidence_gap.summary}`} />
          <FieldBlock label="Veto risk · Rischio veto" value={`${item.risk.label} — ${item.risk.rationale}\nDominio: ${item.risk.domains.join(", ")}${item.risk.veto_owner ? `\nVeto owner: ${item.risk.veto_owner}` : ""}`} />
          <FieldBlock label="Chief synthesis · Sintesi Chief" value={`${item.chief_synthesis.summary}${item.chief_synthesis.unresolved_decision ? `\nDecisione aperta: ${item.chief_synthesis.unresolved_decision}` : ""}${item.chief_synthesis.rationale ? `\nRazionale: ${item.chief_synthesis.rationale}` : ""}`} />
        </div>

        {item.additional_critics && item.additional_critics.length > 0 && (
          <div className="rounded-sm border border-border/70 bg-background/35 p-3 text-sm">
            <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Critic multipli</div>
            <ul className="space-y-1 text-muted-foreground">
              {item.additional_critics.map((critic) => (
                <li key={`${critic.subagent_id}-${critic.source_task_id ?? critic.subagent_label}`}>
                  {critic.subagent_label}{critic.rationale_summary ? ` — ${critic.rationale_summary}` : ""}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid gap-3 lg:grid-cols-2">
          <FieldBlock label="Evidenza e provenienza" value={`Fonti: ${uniqueSourceRefs.length ? uniqueSourceRefs.map(sourceRefLabel).join(", ") : "n/d"}\nConfidence: ${item.provenance.confidence} · Ultimo aggiornamento: ${formatControversyDate(item.provenance.last_updated_at)}${item.provenance.notes ? `\nNote: ${item.provenance.notes}` : ""}`} />
          <FieldBlock label="Prossima azione" value={`${CONTROVERSY_ACTION_COPY[item.recommended_action.action]} — ${item.recommended_action.next_step}\nOwner: ${item.recommended_action.owner ?? "n/d"}${item.recommended_action.due_by_cycle ? ` · Entro: ${item.recommended_action.due_by_cycle}` : ""}`} />
        </div>
      </CardContent>
    </Card>
  );
}

function RadarHermesSection({
  error,
  loading,
  onRefresh,
  snapshot,
}: {
  error: string | null;
  loading: boolean;
  onRefresh: () => void;
  snapshot: RadarHermesSnapshot | null;
}) {
  const top = snapshot?.blocks.top ?? [];
  const controversial = snapshot?.blocks.controversial ?? null;
  const controversyState = snapshot?.controversy_state ?? null;
  const parkable = snapshot?.blocks.parkable ?? null;
  const sideEffects = snapshot?.source_summary.side_effects;
  const sideEffectCount = sideEffects
    ? Number(sideEffects.kanban_mutated) + Number(sideEffects.cron_created) + Number(sideEffects.external_send) + Number(sideEffects.subagent_spawned)
    : 0;

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-primary/20 bg-card/70 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Radar className="h-5 w-5 text-primary" />
            <h3 className="font-semibold">Radar Hermes</h3>
            <Badge tone="warning">READ-ONLY · APPROVAL-GATED</Badge>
            <Badge tone="secondary">Mission Control gate</Badge>
          </div>
          <p className="max-w-4xl text-sm text-muted-foreground">
            Proposte di sviluppo emerse dal lavoro del team, ordinate per impatto e pronte solo per review/approvazione.
          </p>
          <p className="text-xs text-muted-foreground">
            Preview read-only: nessun task, cron, dispatch o invio esterno parte senza conferma di Daniele.
          </p>
          <p className="text-xs text-muted-foreground">
            Aggiornato: {formatRadarDate(snapshot?.generated_at)} · Fonti: {snapshot?.source_summary.sources_read.join(", ") ?? "in caricamento"}
          </p>
        </div>
        <Button type="button" size="sm" ghost onClick={onRefresh} disabled={loading}>
          {loading ? <Spinner className="mr-2 h-4 w-4" /> : null}
          Aggiorna Radar
        </Button>
      </div>

      <div className="grid gap-2 sm:grid-cols-4">
        <Metric label="Top" value={loading && !snapshot ? "…" : String(top.length)} />
        <Metric label="Controversa" value={controversial ? "1" : "0"} />
        <Metric label="Parcheggiabile" value={parkable ? "1" : "0"} />
        <Metric label="Side effects" value={String(sideEffectCount)} />
      </div>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="py-3 text-sm text-destructive">Errore Radar Hermes: {error}</CardContent>
        </Card>
      ) : null}

      {snapshot?.empty_state ? (
        <Card>
          <CardContent className="py-3 text-sm text-muted-foreground">
            <span className="font-medium text-foreground">{snapshot.empty_state.title}</span> — {snapshot.empty_state.message}
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-semibold">Top proposte evolutive</h4>
            <p className="text-xs text-muted-foreground">Le candidate più utili ora, filtrate per evidenza, impatto e fattibilità.</p>
          </div>
          {top.length > 0 ? (
            top.map((proposal, index) => (
              <RadarProposalCard key={proposal.id} label={`#${index + 1}`} proposal={proposal} />
            ))
          ) : (
            <RadarEmptyCard loading={loading} text="Nessuna top proposta evolutiva pronta." />
          )}
        </div>

        <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-warning" />
              <h4 className="text-sm font-semibold">Proposta controversa</h4>
            </div>
            <p className="text-xs text-muted-foreground">Utile se vera, ma richiede challenge esplicita prima di diventare lavoro.</p>
            {controversial ? (
              <RadarProposalCard label="Trade-off" proposal={controversial} tone="warning" />
            ) : (
              <RadarEmptyCard
                loading={loading}
                text={controversyState ? `${controversyState.title}: ${controversyState.message}` : "Nessuna proposta controversa pronta."}
              />
            )}
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <PauseCircle className="h-4 w-4 text-muted-foreground" />
              <h4 className="text-sm font-semibold">Proposta parcheggiabile</h4>
            </div>
            <p className="text-xs text-muted-foreground">Idea valida, ma non abbastanza urgente per entrare nelle top proposte.</p>
            {parkable ? (
              <RadarProposalCard label="Park" proposal={parkable} tone="muted" />
            ) : (
              <RadarEmptyCard loading={loading} text="Nessuna proposta parcheggiabile pronta." />
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function RadarProposalCard({
  label,
  proposal,
  tone = "default",
}: {
  label: string;
  proposal: RadarHermesProposal;
  tone?: "default" | "warning" | "muted";
}) {
  const cardClass =
    tone === "warning"
      ? "border-warning/40 bg-warning/5"
      : tone === "muted"
        ? "border-muted-foreground/20 bg-muted/20"
        : "border-border bg-background/70";
  const evidence = proposal.evidence[0];

  return (
    <Card className={cardClass}>
      <CardContent className="space-y-3 py-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="secondary">{label}</Badge>
          <Badge tone="outline">Score {proposal.ranking.score}</Badge>
          <Badge tone="outline">{RADAR_APPROVAL_LABEL[proposal.approval_state] ?? proposal.approval_state}</Badge>
          <Badge tone="warning">Gate Daniele</Badge>
        </div>
        <div>
          <h5 className="font-semibold leading-snug">{proposal.title}</h5>
          <p className="mt-1 text-sm text-muted-foreground">Perché ora: {proposal.rationale || proposal.source.excerpt || "evidenza sintetica non disponibile"}</p>
        </div>
        <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
          <span>Impatto: {RADAR_LEVEL_LABEL[proposal.priority.impact] ?? proposal.priority.impact}</span>
          <span>Effort: {RADAR_LEVEL_LABEL[proposal.priority.effort] ?? proposal.priority.effort}</span>
          <span>Rischio: {RADAR_LEVEL_LABEL[proposal.priority.risk] ?? proposal.priority.risk}</span>
          <span>Confidence: {RADAR_LEVEL_LABEL[proposal.priority.confidence] ?? proposal.priority.confidence}</span>
        </div>
        <div className="rounded-sm border border-dashed border-border p-3 text-xs text-muted-foreground">
          <div><span className="font-medium text-foreground">Evidenza:</span> {evidence?.summary || proposal.source.excerpt || "Fonte non disponibile"}</div>
          <div><span className="font-medium text-foreground">Provenance:</span> {proposal.source.kind} · {proposal.source.source_id}</div>
          <div><span className="font-medium text-foreground">Assignee suggerito:</span> {proposal.suggested_assignee}</div>
          <div><span className="font-medium text-foreground">Motivo ranking:</span> {proposal.ranking.block} #{proposal.ranking.rank}, score {proposal.ranking.score}/100, execution-gated.</div>
        </div>
        <div className="rounded-sm border border-success/40 bg-success/10 p-3 text-xs text-success">
          Gate Daniele: Mission Control deve approvare ogni passaggio. La card non crea task, non avvia worker, non programma cron e non invia messaggi.
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" ghost>Vedi evidenze</Button>
          <Button type="button" size="sm" ghost disabled>Crea task — gated</Button>
          <Button type="button" size="sm" ghost disabled>Dispatch separato</Button>
        </div>
      </CardContent>
    </Card>
  );
}

interface HermesPmKickoffHandoff {
  summary?: string;
  recommendation?: string;
  decision_needed_from_daniele?: string;
  visible_in_team_pm?: {
    proposal_id?: string;
    task_id?: string;
    pm_signals_count_after_update?: number;
  };
  proposal_contract?: {
    segnale?: string;
    interpretazione?: string;
    supporter?: string;
    critic?: string;
    sintesi_pm?: string;
    gate?: string;
    no_auto_dispatch?: boolean;
  };
  guardrails_respected?: Record<string, boolean | number | string>;
  artifacts?: Record<string, string>;
}

function parseHermesPmKickoffHandoff(body?: string | null): HermesPmKickoffHandoff | null {
  if (!body) return null;
  const match = body.match(/```json\s*([\s\S]*?)\s*```/i);
  if (!match) return null;
  try {
    return JSON.parse(match[1]) as HermesPmKickoffHandoff;
  } catch {
    return null;
  }
}

function humanizeGuardrailKey(key: string): string {
  const labels: Record<string, string> = {
    kanban_tasks_created: "Task creati",
    dispatch_run: "Dispatch automatico",
    cron_created: "Cron creati",
    external_send: "Invii esterni",
    code_edits: "Modifiche codice",
  };
  return labels[key] ?? key.replace(/_/g, " ");
}

function guardrailValue(value: boolean | number | string): string {
  if (value === false || value === 0 || value === "false" || value === "0") return "no";
  if (value === true || value === "true") return "sì";
  return String(value);
}

function basename(path?: string): string {
  if (!path) return "";
  return path.split("/").filter(Boolean).pop() ?? path;
}

function HermesPmKickoffReadableResult({
  handoff,
  summary,
  rawAvailable,
}: {
  handoff: HermesPmKickoffHandoff | null;
  summary: string;
  rawAvailable: boolean;
}) {
  const contract = handoff?.proposal_contract;
  const guardrails = handoff?.guardrails_respected ? Object.entries(handoff.guardrails_respected) : [];
  const proposalId = handoff?.visible_in_team_pm?.proposal_id;
  const briefPath = handoff?.artifacts?.brief;
  const nextSlice = handoff?.recommendation ?? contract?.sintesi_pm ?? summary;

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-sm border border-primary/40 bg-primary/10 p-4">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <Badge tone="success">lettura PM</Badge>
          {proposalId && <Badge tone="secondary">{proposalId}</Badge>}
          {handoff?.visible_in_team_pm?.pm_signals_count_after_update !== undefined && (
            <Badge tone="secondary">{handoff.visible_in_team_pm.pm_signals_count_after_update} segnale PM</Badge>
          )}
        </div>
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Sintesi semplice</div>
        <p className="mt-1 text-sm leading-relaxed text-foreground/90">{handoff?.summary ?? summary}</p>
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="rounded-sm border border-warning/40 bg-warning/10 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Decisione richiesta a Daniele</div>
          <p className="mt-1 text-sm leading-relaxed text-foreground/90">
            {handoff?.decision_needed_from_daniele ?? "Serve una decisione di Daniele prima di creare task o dispatchare lavoro."}
          </p>
        </div>
        <div className="rounded-sm border border-border/60 bg-background/30 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Prossimo slice consigliato</div>
          <p className="mt-1 text-sm leading-relaxed text-foreground/90">
            {nextSlice}
          </p>
        </div>
      </div>

      {contract && (
        <div className="grid gap-3 lg:grid-cols-2">
          <FieldBlock label="Segnale" value={contract.segnale ?? "Segnale non disponibile."} />
          <FieldBlock label="Interpretazione" value={contract.interpretazione ?? "Interpretazione non disponibile."} />
          <FieldBlock label="Supporter" value={contract.supporter ?? "Supporter non disponibile."} />
          <FieldBlock label="Critic" value={contract.critic ?? "Critic non disponibile."} />
          <div className="lg:col-span-2">
            <FieldBlock label="Sintesi PM" value={contract.sintesi_pm ?? "Sintesi PM non disponibile."} />
          </div>
        </div>
      )}

      <div className="rounded-sm border border-success/40 bg-success/10 p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Guardrail rispettati</div>
        <div className="flex flex-wrap gap-2">
          {guardrails.length > 0 ? (
            guardrails.map(([key, value]) => (
              <Badge key={key} tone={value === false || value === 0 ? "success" : "secondary"}>
                {humanizeGuardrailKey(key)}: {guardrailValue(value)}
              </Badge>
            ))
          ) : (
            <Badge tone="success">no task · no cron · no dispatch automatico</Badge>
          )}
          {contract?.gate && <Badge tone="warning">Gate: {contract.gate}</Badge>}
          {contract?.no_auto_dispatch !== undefined && <Badge tone="success">execution gate: preview → conferma → ready</Badge>}
        </div>
      </div>

      {(briefPath || rawAvailable) && (
        <details className="rounded-sm border border-border/60 bg-background/30 p-3 text-sm">
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Dettagli tecnici / artefatti
          </summary>
          <div className="mt-2 space-y-2 text-xs text-muted-foreground">
            {briefPath && <div>Brief: {basename(briefPath)}</div>}
            <div>Output raw disponibile nel task Kanban; resta nascosto qui per mantenere la pagina semplice.</div>
          </div>
        </details>
      )}
    </div>
  );
}

function RadarEmptyCard({ loading, text }: { loading: boolean; text: string }) {
  return (
    <Card>
      <CardContent className="py-4 text-sm text-muted-foreground">
        {loading ? "Caricamento Radar Hermes…" : text}
      </CardContent>
    </Card>
  );
}

function HermesPmWorkspacePanel({
  proposals,
  specialists,
  strategicReview,
  updatedAt,
  onTaskPreview,
  onPlanPreview,
  updating,
}: {
  proposals: TeamProposal[];
  specialists: TeamSpecialist[];
  strategicReview?: TeamProposalsResponse["strategic_review"];
  updatedAt?: string;
  onTaskPreview: (proposal: TeamProposal) => void;
  onPlanPreview: (proposal: TeamProposal) => void;
  updating: string | null;
}) {
  const kickoffTaskId = "t_648a4e26";
  const [kickoffTask, setKickoffTask] = useState<KanbanTaskDetailResponse | null>(null);
  const [kickoffError, setKickoffError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getKanbanTask(kickoffTaskId, "co2farm-chief")
      .then((response) => {
        if (!cancelled) setKickoffTask(response);
      })
      .catch((e) => {
        if (!cancelled) setKickoffError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const latestPmComment = kickoffTask?.comments?.slice().reverse().find((comment) => comment.author === "hermespm") ?? kickoffTask?.comments?.slice().reverse()[0];
  const latestRun = kickoffTask?.runs?.slice().reverse()[0];
  const kickoffHandoff = parseHermesPmKickoffHandoff(latestPmComment?.body);

  const pmSignals = proposals.filter((proposal) => {
    const text = `${proposal.title} ${proposal.origin} ${proposal.category ?? ""} ${proposal.source_agent ?? ""} ${proposal.whyNow} ${proposal.evidence ?? ""}`.toLowerCase();
    return text.includes("hermespm") || text.includes("hermes pm") || text.includes("project manager") || text.includes("roadmap");
  });
  const pmSpecialist = specialists.find((role) => role.id === "hermespm" || role.name.toLowerCase().includes("hermespm"));
  const chatUrl = "/chat?profile=hermespm";

  return (
    <section className="flex flex-col gap-4">
      <Card>
        <CardContent className="flex flex-col gap-4 py-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Hermes PM — cockpit di lavoro</h3>
              </div>
              <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
                Questa è la stanza dedicata a HermesPM: qui Daniele lavora con il project manager dello sviluppo interno, non con il registro generico delle proposte.
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Profilo: hermespm · alias CLI: hermespm · ultimo registro Sviluppo Hermes: {updatedAt ?? "unknown"}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge tone="success">profilo dedicato</Badge>
              <Badge tone="warning">review gate</Badge>
              <Badge tone="secondary">dispatch gated</Badge>
            </div>
          </div>

          <div className="grid gap-2 sm:grid-cols-4">
            <Metric label="PM signals" value={String(pmSignals.length)} />
            <Metric label="Roadmap visibile" value="max 3" />
            <Metric label="Gate" value="Daniele" />
            <Metric label="Scope" value="Hermes only" />
          </div>

          <div className="rounded-sm border border-primary/40 bg-primary/10 p-3 text-sm">
            <div className="font-semibold text-foreground">Team sviluppo Hermes</div>
            <p className="mt-1 text-muted-foreground">
              Per le proposte HermesPM la pagina userà un team separato dal team operativo CO2Farm: coordinamento a <strong>ops</strong>, implementazione a <strong>default</strong>, QA/runtime a <strong>reliability</strong>. HermesPM sceglie automaticamente tra profili reali disponibili e mostra l'assegnazione in preview.
            </p>
          </div>

          <div className="rounded-sm border border-success/50 bg-success/10 p-3 text-sm text-success">
            Regola: HermesPM propone; Daniele legge e conferma. Dopo la conversione in Kanban, la card entra ready e il dispatcher la avvia secondo priorità e concorrenza. Niente cron o invii esterni.
          </div>

          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => { window.location.href = chatUrl; }}>
              Apri chat con HermesPM
            </Button>
            <Button ghost size="sm" onClick={() => { window.location.href = "/team-evolution"; }}>
              Resta in Sviluppo Hermes
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <Card>
          <CardContent className="flex flex-col gap-3 py-5">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Come lavora con te</h3>
            </div>
            {[
              "Raccoglie segnali da Mission Control, Team Pulse, review gate e feedback di Daniele.",
              "Trasforma segnali in una roadmap corta: massimo tre priorità visibili.",
              "Per ogni proposta mostra supporter, critic, sintesi PM e gate prima di creare lavoro.",
              "Coordina profili reali solo dopo approvazione: niente backlog automatico, conversione ready e dispatch secondo priorità/concorrenza.",
            ].map((item) => (
              <div key={item} className="flex items-start gap-2 text-sm text-muted-foreground">
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-success" />
                <span>{item}</span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-3 py-5">
            <div className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Contratto operativo</h3>
            </div>
            <div className="grid gap-2 text-xs sm:grid-cols-2">
              <Metric label="Segnale" value="obbligatorio" />
              <Metric label="Interpretazione" value="obbligatoria" />
              <Metric label="Supporter/Critic" value="obbligatori" />
              <Metric label="Gate" value="preview → conferma → ready" />
            </div>
            <p className="text-sm text-muted-foreground">
              Hermes PM non è più una terza pagina: è la regia interna dello spazio Sviluppo Hermes. Il Team Operativo resta separato e lavora solo sulle task di Daniele.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-3 py-5">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            <h3 className="font-semibold">Cosa porta ora al tavolo</h3>
            <Badge tone="secondary">proposal-only</Badge>
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            <FieldBlock label="Prossimo passo consigliato" value="Usare HermesPM come front door per roadmap e priorità interne; il primo output utile è un brief con max 3 priorità e un solo prossimo slice." />
            <FieldBlock label="Segnale corrente" value="Daniele ha chiesto una pagina dedicata per lavorare direttamente con HermesPM, distinta da operative ed evoluzione generale." />
            <FieldBlock label="Rischio da evitare" value="Trasformare il PM in un'altra sorgente di rumore. La UI deve restare un cockpit decisionale, non un nuovo backlog automatico." />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-3 py-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <CircleDot className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Risultati kickoff HermesPM</h3>
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Badge tone="secondary">{kickoffTaskId}</Badge>
              <Badge tone={kickoffTask?.task.status === "blocked" ? "warning" : kickoffTask?.task.status === "running" ? "success" : "secondary"}>
                {kickoffTask?.task.status ?? "loading"}
              </Badge>
            </div>
          </div>
          {kickoffError && <p className="text-sm text-destructive">{kickoffError}</p>}
          {!kickoffTask && !kickoffError && <p className="text-sm text-muted-foreground">Carico output del task kickoff…</p>}
          {kickoffTask && (
            <>
              <div className="grid gap-2 sm:grid-cols-4">
                <Metric label="Assignee" value={kickoffTask.task.assignee ?? "unknown"} />
                <Metric label="Run" value={latestRun ? `#${latestRun.id}` : "nessuna"} />
                <Metric label="Outcome" value={latestRun?.outcome ?? latestRun?.status ?? "unknown"} />
                <Metric label="Commenti" value={String(kickoffTask.comments.length)} />
              </div>
              <HermesPmKickoffReadableResult
                handoff={kickoffHandoff}
                summary={kickoffTask.task.latest_summary ?? latestRun?.summary ?? "Nessuna sintesi ancora disponibile."}
                rawAvailable={Boolean(latestPmComment?.body)}
              />
              {kickoffTask.task.workspace_path && <p className="text-xs text-muted-foreground">Workspace tecnico disponibile nel task Kanban.</p>}
            </>
          )}
        </CardContent>
      </Card>

      {(pmSpecialist || pmSignals.length > 0 || strategicReview) && (
        <Card>
          <CardContent className="flex flex-col gap-3 py-5">
            <h3 className="font-semibold">Stato collegato dal registro</h3>
            {pmSpecialist && <FieldBlock label="Profilo PM" value={`${pmSpecialist.name}: ${pmSpecialist.mission}`} />}
            {pmSignals.length > 0 ? (
              <div className="grid gap-3">
                {pmSignals.slice(0, 5).map((proposal) => (
                  <HermesPmLaunchCard
                    key={proposal.id}
                    proposal={proposal}
                    onTaskPreview={onTaskPreview}
                    onPlanPreview={onPlanPreview}
                    updating={updating}
                  />
                ))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Nessuna proposta HermesPM dedicata nel registro corrente: la pagina è pronta come workspace, non forza la creazione di task.</p>
            )}
            {strategicReview && (
              <p className="text-xs text-muted-foreground">Strategic Review attiva: {strategicReview.summary}</p>
            )}
          </CardContent>
        </Card>
      )}
    </section>
  );
}

function OperationalReviewPanel({
  active,
  challenged,
  legacy,
  onPlanPreview,
  onStatus,
  pendingReviews,
  source,
  safety,
  updatedAt,
  updating,
}: {
  active: TeamProposal[];
  challenged: TeamProposal[];
  legacy: TeamProposal[];
  onPlanPreview: (proposal: TeamProposal) => void;
  onStatus: (proposal: TeamProposal, status: TeamProposalStatus) => void;
  pendingReviews: number;
  source?: TeamProposalsResponse["source"];
  safety?: TeamProposalsResponse["safety"];
  updatedAt?: string;
  updating: string | null;
}) {
  return (
    <section className="flex flex-col gap-4">
      <Card>
        <CardContent className="flex flex-col gap-4 py-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="font-semibold">Registro verificato</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Stato corrente letto dal registro dashboard, non da frasi storiche embedded nelle proposte.
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Fonte primaria: {source?.label ?? "team_proposals.json"} · freshness {source?.freshness ?? "fresh"} · aggiornato {updatedAt ?? "unknown"}
              </p>
            </div>
            <Badge tone="warning">approval-gated</Badge>
          </div>
          <div className="grid gap-2 sm:grid-cols-4">
            <Metric label="Proposte operative attive" value={String(active.length)} />
            <Metric label="Autonome challengeate" value={String(challenged.length)} />
            <Metric label="Review Chief pending / default" value={String(pendingReviews)} />
            <Metric label="Record da arricchire" value={String(legacy.length)} />
          </div>
          <div className="rounded-sm border border-warning/50 bg-warning/10 p-3 text-sm text-warning">
            Nota evidenza: “2 proposte attive e 8 review pending” è un segnale storico/autogenerato, non il conteggio corrente verificato.
          </div>
          <div className="rounded-sm border border-success/50 bg-success/10 p-3 text-sm text-success">
            Invarianti: no auto-task durante intake, no cron, no webhook, no invii esterni. Safety API: conversion_initial_status={safety?.conversion_initial_status ?? "unknown"}; preview read-only: {safety?.preview_read_only ? "true" : "unknown"}.
          </div>
        </CardContent>
      </Card>

      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-primary" />
          <h3 className="font-semibold">Proposte operative challengeate</h3>
          <Badge tone="success">{challenged.length}</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          Poche prossime mosse operative generate dal team, filtrate perché hanno segnale osservabile, evidenza minima, challenge e sintesi Chief.
        </p>
        <div className="grid gap-3">
          {challenged.length === 0 && (
            <Card><CardContent className="py-4 text-sm text-muted-foreground">Nessuna proposta operativa challengeata pronta. Non generare backlog: serve un segnale verificabile prima del gate.</CardContent></Card>
          )}
          {challenged.map((proposal) => (
            <OperationalProposalCard
              key={`operational-${proposal.id}`}
              proposal={proposal}
              updating={updating}
              onPlanPreview={onPlanPreview}
              onStatus={onStatus}
            />
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <CircleDot className="h-5 w-5 text-warning" />
          <h3 className="font-semibold">Da arricchire prima di decidere</h3>
          <Badge tone="secondary">{legacy.length}</Badge>
        </div>
        <div className="grid gap-2">
          {legacy.length === 0 && <p className="text-sm text-muted-foreground">Nessun record legacy operativo attivo.</p>}
          {legacy.map((proposal) => (
            <div key={`legacy-${proposal.id}`} className="rounded-sm border border-border/70 bg-background/35 p-3 text-sm">
              <div className="font-medium">{proposal.title}</div>
              <div className="mt-1 text-muted-foreground">{reviewLabel(proposal)} · {proposal.evidence ? "evidenza presente" : "evidenza non disponibile"} · challenge {proposal.challenge ? "presente" : "mancante"}</div>
            </div>
          ))}
        </div>
      </section>

      <Card>
        <CardContent className="flex flex-col gap-2 py-4 text-sm">
          <div className="font-semibold">Radar evolutivo separato</div>
          <p className="text-muted-foreground">
            Le idee di sviluppo Hermes restano nello spazio Sviluppo Hermes: questa vista non mescola lavoro operativo e roadmap del sistema.
          </p>
        </CardContent>
      </Card>
    </section>
  );
}

function formatContractValue(value: unknown, fallback = "non disponibile"): string {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.map((item) => formatContractValue(item, "")).filter(Boolean).join("\n") || fallback;
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return String(obj.summary ?? obj.hypothesis ?? obj.synthesis ?? obj.acceptance ?? obj.rationale ?? JSON.stringify(obj));
  }
  return String(value);
}

function formatViewpoint(
  viewpoint: TeamProposal["supporter_view"] | TeamProposal["critic_view"],
  fallbackActor?: string | null,
  fallbackRationale?: string | null,
): string {
  const actor = viewpoint?.actor ?? fallbackActor ?? "non indicato";
  const rationale = viewpoint?.rationale ?? fallbackRationale ?? "razionale non disponibile";
  const method = viewpoint?.method ? ` · metodo ${viewpoint.method}` : "";
  return `${actor}: ${rationale}${method}`;
}

function formatGateState(proposal: TeamProposal): string {
  return [
    proposal.gate_state ?? "review_required",
    proposal.autonomy_gate ?? "approval_required",
    proposal.no_auto_dispatch ? "preview-only; ready-on-convert" : "ready-on-convert",
  ].join(" · ");
}

function formatSourceAgent(proposal: TeamProposal): string {
  return proposal.source_agent ?? "source_agent mancante: record legacy normalizzato, audit trail preservato";
}

function formatEvidenceRefs(proposal: TeamProposal): string {
  return proposal.evidence_refs?.length ? proposal.evidence_refs.join("\n") : (proposal.evidence ?? "riferimenti evidenza non disponibili");
}

function AutonomousProposalContractGrid({ proposal }: { proposal: TeamProposal }) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <FieldBlock label="Segnale" value={formatContractValue(proposal.signal ?? proposal.source_signal ?? proposal.whyNow)} />
      <FieldBlock label="Interpretazione" value={formatContractValue(proposal.interpretation, "interpretazione non disponibile")} />
      <FieldBlock label="Supporter" value={formatViewpoint(proposal.supporter_view, proposal.challenge?.supporter, proposal.challenge?.support)} />
      <FieldBlock label="Critic" value={formatViewpoint(proposal.critic_view, proposal.challenge?.critic, proposal.challenge?.challenge)} />
      <FieldBlock label="Chief synthesis" value={formatContractValue(proposal.chief_synthesis ?? proposal.challenge?.chief_synthesis, "sintesi Chief non disponibile")} />
      <FieldBlock label="Stato gate" value={formatGateState(proposal)} />
      <FieldBlock label="source_agent" value={formatSourceAgent(proposal)} />
      <FieldBlock label="Riferimenti evidenza" value={formatEvidenceRefs(proposal)} />
    </div>
  );
}

function OperationalProposalCard({
  proposal,
  updating,
  onPlanPreview,
  onStatus,
}: {
  proposal: TeamProposal;
  updating: string | null;
  onPlanPreview: (proposal: TeamProposal) => void;
  onStatus: (proposal: TeamProposal, status: TeamProposalStatus) => void;
}) {
  const isBusy = updating?.startsWith(`${proposal.id}:`) ?? false;
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 py-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h4 className="font-semibold">{proposal.title}</h4>
            <p className="mt-1 text-xs text-muted-foreground">Owner suggerito: {proposal.source_agent ?? "non indicato"} · Formulata: {formatProposalFormulatedDate(proposal)}</p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Badge tone="secondary">{reviewLabel(proposal)}</Badge>
            <Badge tone="warning">{riskGateLabel(proposal)}</Badge>
            <Badge tone="success">Gate approval_required</Badge>
          </div>
        </div>
        <AutonomousProposalContractGrid proposal={proposal} />
        <FieldBlock label="Prossima mossa suggerita" value={proposal.suggested_next_action ?? proposal.acceptance} />
        <div className="rounded-sm border border-success/40 bg-success/10 p-3 text-sm text-success">
          Gate operativo umano — Nessuna azione qui avvia worker, cron, messaggi esterni o dispatch automatico.
          <div className="mt-1 text-xs text-muted-foreground">
            Preview piano Kanban, approva, respingi o richiedi revisione cambiano solo stato/visibilità della proposta; dopo eventuale conversione il task entra ready.
          </div>
        </div>
        <div className="flex flex-wrap gap-2 border-t border-border/60 pt-3">
          <Button type="button" size="sm" disabled={isBusy || proposal.status === "trasformata_in_task"} onClick={() => onPlanPreview(proposal)}>
            Preview piano Kanban
          </Button>
          <Button type="button" size="sm" ghost disabled={isBusy || proposal.status === "approvata"} onClick={() => onStatus(proposal, "approvata")}>
            Approva
          </Button>
          <Button type="button" size="sm" ghost disabled={isBusy || proposal.status === "scartata"} onClick={() => onStatus(proposal, "scartata")}>
            Respingi
          </Button>
          <Button type="button" size="sm" ghost disabled={isBusy || proposal.status === "parcheggiata"} onClick={() => onStatus(proposal, "parcheggiata")}>
            Richiedi revisione
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function FieldBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/60 bg-background/30 p-3 text-sm">
      <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</div>
      <p className="whitespace-pre-wrap text-foreground/90">{value}</p>
    </div>
  );
}

function buildNowActions({
  mode,
  chiefQueue,
  modePulse,
  proposals,
}: {
  mode: TeamProposalsMode;
  chiefQueue: TeamProposal[];
  modePulse?: TeamPulseSummary;
  proposals: TeamProposal[];
}): NowAction[] {
  const actions: NowAction[] = [];
  const actionableProposals = proposals.filter((p) => p.status !== "trasformata_in_task");
  const topChief = chiefQueue.find((p) => p.status !== "trasformata_in_task");
  const topApproved = actionableProposals.find((p) => p.status === "approvata");
  const topRecommended = actionableProposals.find(
    (p) => p.status === "raccomandata" || p.recommendation === "do_now",
  );
  const topProposal = topApproved ?? topChief ?? topRecommended;

  if (topApproved) {
    actions.push({
      title: "1. Apri preview piano prima di creare lavoro",
      body: `Proposta già approvata: “${topApproved.title}”. Il prossimo passo sicuro è vedere il piano Kanban prima di creare card.`,
      gate: "Preview only → conferma → task ready; dispatcher secondo priorità/concorrenza.",
      proposal: topApproved,
      primaryAction: "plan-preview",
    });
  } else if (topProposal) {
    actions.push({
      title: "1. Decidi se promuovere una sola proposta",
      body: `Candidata: “${topProposal.title}”. Evita backlog: approva solo se vuoi trasformarla in piano/task ispezionabile.`,
      gate: "Approve cambia solo stato della proposta; non crea task e non avvia worker.",
      proposal: topProposal,
      primaryAction: "approve",
    });
  } else {
    actions.push({
      title: "1. Genera un pulse controllato",
      body: mode === "evolution"
        ? "Non ci sono proposte evolutive pronte: fai generare al team idee di sviluppo Hermes con challenge interna."
        : "Non ci sono proposte operative pronte: fai raccogliere segnali operativi e blocker in modalità proposal-only.",
      gate: "Genera solo proposte/challenge; nessun task, cron, invio o dispatch.",
      primaryAction: "pulse",
    });
  }

  if (modePulse && modePulse.controversial_count > 0) {
    const controversial = modePulse.controversial[0];
    actions.push({
      title: "2. Leggi la challenge prima di decidere",
      body: controversial
        ? `La proposta più contestata è “${controversial.title}”: verifica supporter, critic e Chief synthesis prima di promuoverla.`
        : "Ci sono proposte con challenge interna: usa la critica per non trasformare rumore in lavoro.",
      gate: "Challenge prima della shortlist; se c'è rischio veto, resta proposta parcheggiata.",
      proposal: controversial,
    });
  } else {
    actions.push({
      title: "2. Mantieni separati proposta, preview e lancio",
      body: "Una proposta approvata non è ancora lavoro eseguibile: prima preview, poi conferma. Dopo la conversione la task entra ready e il dispatcher la avvia secondo priorità/concorrenza.",
      gate: "Sequenza: proposta → preview piano/task → conferma → task ready → dispatch secondo priorità/concorrenza.",
    });
  }

  actions.push({
    title: "3. Non avviare subagenti finché il piano non è leggibile",
    body: mode === "evolution"
      ? "Per lo sviluppo dei subagenti, il prossimo valore è chiarezza del cockpit: cosa fare, cosa non fare, e quale gate serve."
      : "Per lavoro operativo, i subagenti devono produrre output su task chiari, non partire da segnali grezzi o ambigui.",
    gate: "No external send, no cron; le task confermate entrano ready e seguono il dispatcher Kanban secondo priorità/concorrenza.",
  });

  return actions.slice(0, 3);
}

function NowActionsPanel({
  actions,
  mode,
  onApprove,
  onPulse,
  onTaskPreview,
  onPlanPreview,
  updating,
}: {
  actions: NowAction[];
  mode: TeamProposalsMode;
  onApprove: (proposal: TeamProposal) => void;
  onPulse: () => void;
  onTaskPreview: (proposal: TeamProposal) => void;
  onPlanPreview: (proposal: TeamProposal) => void;
  updating: string | null;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <ShieldCheck className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">Cosa fare ora</h3>
        <Badge tone="success">max 3</Badge>
      </div>
      <Card>
        <CardContent className="flex flex-col gap-4 py-4">
          <p className="text-sm text-muted-foreground">
            Questa è la corsia a bassa frizione: ti dice cosa fare adesso senza confondere proposta, task e dispatch.
          </p>
          <div className="grid gap-3 lg:grid-cols-3">
            {actions.map((action) => {
              const proposal = action.proposal;
              const busy = proposal ? (updating?.startsWith(`${proposal.id}:`) ?? false) : updating === "team-pulse:generate";
              return (
                <div key={`${action.title}:${proposal?.id ?? action.primaryAction ?? "guardrail"}`} className="flex flex-col gap-3 rounded-sm border border-border/70 bg-background/35 p-3">
                  <div>
                    <h4 className="text-sm font-semibold">{action.title}</h4>
                    <p className="mt-1 text-sm text-muted-foreground">{action.body}</p>
                  </div>
                  <div className="rounded-sm border border-success/40 bg-success/10 p-2 text-xs text-success">
                    {action.gate}
                  </div>
                  {action.primaryAction && (
                    <div className="mt-auto flex flex-wrap gap-2 pt-1">
                      {action.primaryAction === "approve" && proposal && (
                        <Button size="sm" disabled={busy} onClick={() => onApprove(proposal)}>
                          Approva proposta
                        </Button>
                      )}
                      {action.primaryAction === "plan-preview" && proposal && (
                        <Button size="sm" disabled={busy} onClick={() => onPlanPreview(proposal)}>
                          Preview piano…
                        </Button>
                      )}
                      {action.primaryAction === "task-preview" && proposal && (
                        <Button size="sm" disabled={busy} onClick={() => onTaskPreview(proposal)}>
                          Preview task…
                        </Button>
                      )}
                      {action.primaryAction === "pulse" && (
                        <Button size="sm" disabled={busy} onClick={onPulse} prefix={busy ? <Spinner /> : undefined}>
                          {mode === "evolution" ? "Attiva Team Pulse" : "Attiva Pulse operativo"}
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}

function growthBandLabel(band: AgentGrowthProfile["scoring"]["readiness_band"]): string {
  switch (band) {
    case "trusted_for_preview": return "Pronto per preview";
    case "operational": return "Operativo";
    case "emerging": return "In emersione";
    case "insufficient_data": return "Segnali insufficienti";
  }
}

function provenanceLabel(field: AgentGrowthField): string {
  const refs = field.provenance ?? [];
  if (!refs.length) return "Fonte: dato mancante/unknown";
  const ref = refs[0];
  return `Fonte: ${ref.source_id ?? ref.kind}${ref.proposal_id ? ` · ${ref.proposal_id}` : ""}`;
}

function AgentGrowthFieldRow({ label, field }: { label: string; field: AgentGrowthField }) {
  const tone = field.state === "present" ? "success" : field.state === "redacted" ? "destructive" : "secondary";
  const missingCopy = field.state === "present" ? null : `Dato mancante: ${field.missing_reason ?? "not_recorded"}`;
  return (
    <div className="rounded-sm border border-border/60 bg-background/30 p-2">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
        <Badge tone={tone}>{field.state} · {field.confidence}</Badge>
      </div>
      <p className="text-xs text-foreground/90">{field.display}</p>
      {missingCopy && <p className="mt-1 text-[0.68rem] text-muted-foreground">{missingCopy}</p>}
      <p className="mt-1 text-[0.68rem] text-muted-foreground">{provenanceLabel(field)}</p>
    </div>
  );
}

function AgentGrowthCard({ growth }: { growth: AgentGrowthProfile }) {
  const score = growth.scoring.state === "computed" && growth.scoring.growth_score != null
    ? `${growth.scoring.growth_score}/100`
    : "non calcolato";
  const band = growthBandLabel(growth.scoring.readiness_band);
  const badgeTone = growth.scoring.readiness_band === "trusted_for_preview"
    ? "success"
    : growth.scoring.readiness_band === "operational"
      ? "warning"
      : "secondary";
  return (
    <div className="rounded-sm border border-primary/30 bg-primary/5 p-3 text-xs">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="font-semibold text-foreground">Crescita osservabile</span>
        <Badge tone={badgeTone}>{score} · {band}</Badge>
      </div>
      <p className="mb-2 text-[0.7rem] text-muted-foreground">
        Mappa dati: {growth.agent.mapping_confidence} · profilo {growth.agent.agent_id}{growth.agent.logical_role ? ` · ruolo logico ${growth.agent.logical_role}` : ""} · schema {growth.schema_version}. Non è una classifica tra agenti.
      </p>
      <div className="grid gap-2">
        <AgentGrowthFieldRow label="Ultimo segnale osservato" field={growth.last_observed_signal} />
        <AgentGrowthFieldRow label="Proposta propria" field={growth.own_proposal} />
        <AgentGrowthFieldRow label="Challenge ricevute" field={growth.challenges_received} />
        <AgentGrowthFieldRow label="Learning note" field={growth.learning_notes} />
        <AgentGrowthFieldRow label="Prossimo sviluppo ruolo" field={growth.next_role_development} />
      </div>
      {growth.scoring.explainers.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {growth.scoring.explainers.slice(0, 3).map((item) => (
            <Badge key={item} tone="secondary">{item}</Badge>
          ))}
        </div>
      )}
      <p className="mt-2 text-[0.68rem] text-muted-foreground">
        Gate: read-only, review Daniele richiesta, nessun cron/invio/dispatch automatico.
      </p>
    </div>
  );
}

function TeamConstitutionPanel({
  constitution,
  mode,
}: {
  constitution?: TeamConstitutionContract;
  mode: TeamProposalsMode;
}) {
  if (!constitution) {
    return (
      <section className="rounded-sm border border-warning/50 bg-warning/10 p-3 text-sm text-warning">
        Costituzione team non caricata: la pagina resta in modalità proposal-only finché il contratto non è disponibile.
      </section>
    );
  }

  return (
    <section className="flex flex-col gap-3 rounded-sm border border-border/60 bg-background/30 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold">Costituzione applicata — {constitution.team}</h3>
          <p className="text-xs text-muted-foreground">
            Prompt attivo: Costituzione comune + identità specifica · Lead {constitution.lead} · {mode === "operative" ? "Team Operativo" : "Sviluppo Hermes"}
          </p>
        </div>
        <Badge tone="secondary">Proposal Mode default</Badge>
      </div>
      <div className="grid gap-2 lg:grid-cols-2">
        <FieldBlock label="Missione" value={constitution.mission} />
        <FieldBlock label="North Star" value={constitution.north_star} />
      </div>
      <div className="grid gap-2 lg:grid-cols-2">
        <FieldBlock label="Cosa NON deve fare" value={constitution.must_not.map((item) => `• ${item}`).join("\n")} />
        <FieldBlock label="Lettura a inizio ciclo" value={constitution.cycle_start_reads.map((item) => `• ${item}`).join("\n")} />
      </div>
      <FieldBlock label="Handoff se emerge dominio dell'altro team" value={constitution.handoff} />
      <details className="rounded-sm border border-border/50 bg-muted/20 p-2">
        <summary className="cursor-pointer text-xs font-semibold text-muted-foreground">Fonti prompt e file stato/log</summary>
        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
          {constitution.prompt_sources.map((source) => (
            <li key={source} className="break-all">{source}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/70 bg-background/35 p-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 font-semibold">{value}</div>
    </div>
  );
}

function RoleFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/60 bg-background/30 p-2">
      <div className="text-[0.65rem] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-xs text-foreground/90">{value}</div>
    </div>
  );
}

function HermesPmLaunchCard({
  proposal,
  onTaskPreview,
  onPlanPreview,
  updating,
}: {
  proposal: TeamProposal;
  onTaskPreview: (proposal: TeamProposal) => void;
  onPlanPreview: (proposal: TeamProposal) => void;
  updating: string | null;
}) {
  const isConverted = proposal.status === "trasformata_in_task";
  const busy = updating?.startsWith(`${proposal.id}:`) ?? false;
  return (
    <div className="rounded-sm border border-border/70 bg-background/35 p-3 text-sm">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-semibold text-foreground">{proposal.title}</div>
          <p className="mt-1 text-xs text-muted-foreground">{proposal.whyNow}</p>
          {proposal.suggested_next_action && <p className="mt-1 text-xs text-muted-foreground">Prossima mossa: {proposal.suggested_next_action}</p>}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge tone={toneForPriority(proposal.priority)}>{proposal.priority}</Badge>
          <Badge tone={proposalStatusTone(proposal.status)}>{statusLabel(proposal.status)}</Badge>
          <Badge tone="secondary">Hermes dev team</Badge>
        </div>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <Metric label="Coordinamento" value="ops" />
        <Metric label="Implementazione" value="default" />
        <Metric label="QA/runtime" value="reliability" />
      </div>
      {proposal.challenge && (
        <div className="mt-3 grid gap-2 lg:grid-cols-2">
          <FieldBlock label="Supporter" value={`${proposal.challenge.supporter}: ${proposal.challenge.support}`} />
          <FieldBlock label="Critic" value={`${proposal.challenge.critic}: ${proposal.challenge.challenge}`} />
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2 border-t border-border/60 pt-3">
        <Button
          size="sm"
          disabled={busy || isConverted}
          onClick={() => onPlanPreview(proposal)}
          prefix={updating === `${proposal.id}:convert-plan` ? <Spinner /> : undefined}
        >
          Preview piano auto-assegnato
        </Button>
        <Button
          ghost
          size="sm"
          disabled={busy || isConverted}
          onClick={() => onTaskPreview(proposal)}
          prefix={updating === `${proposal.id}:convert` ? <Spinner /> : undefined}
        >
          Task singola ready…
        </Button>
        {isConverted && <Badge tone="success">già lanciata in Kanban</Badge>}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        La preview non crea nulla. La conferma crea task ready: il dispatcher parte secondo priorità/concorrenza.
      </p>
    </div>
  );
}

function StrategicList({ title, proposals }: { title: string; proposals: TeamProposal[] }) {
  return (
    <div className="rounded-sm border border-border/60 bg-background/30 p-3">
      <div className="mb-2 text-sm font-semibold">{title}</div>
      <div className="flex flex-col gap-2">
        {proposals.length === 0 && <p className="text-xs text-muted-foreground">Nessuna proposta.</p>}
        {proposals.map((proposal) => (
          <div key={proposal.id} className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">{proposal.title}</span> · score {proposal.chief_review_score ?? 0}/100 · {proposal.priority}
          </div>
        ))}
      </div>
    </div>
  );
}

function ProposalColumn({
  title,
  icon,
  proposals,
  updating,
  onStatus,
  onTaskPreview,
  onPlanPreview,
  onChiefReview,
  readOnly = false,
}: {
  title: string;
  icon: ReactNode;
  proposals: TeamProposal[];
  updating: string | null;
  onStatus: (proposal: TeamProposal, status: TeamProposalStatus) => void;
  onTaskPreview: (proposal: TeamProposal) => void;
  onPlanPreview: (proposal: TeamProposal) => void;
  onChiefReview: (proposal: TeamProposal, action: "shortlist" | "defer" | "reject") => void;
  readOnly?: boolean;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center gap-2 text-primary">
        {icon}
        <h3 className="font-semibold text-foreground">{title}</h3>
      </div>
      <div className="grid gap-3">
        {proposals.map((proposal) => (
          <ProposalCard
            key={proposal.id}
            proposal={proposal}
            compact
            updating={updating}
            onStatus={onStatus}
            onTaskPreview={onTaskPreview}
            onPlanPreview={onPlanPreview}
            onChiefReview={onChiefReview}
            readOnly={readOnly}
          />
        ))}
      </div>
    </section>
  );
}

function ProposalCard({
  proposal,
  compact,
  updating,
  onStatus,
  onTaskPreview,
  onPlanPreview,
  onChiefReview,
  readOnly = false,
}: {
  proposal: TeamProposal;
  compact: boolean;
  updating: string | null;
  onStatus: (proposal: TeamProposal, status: TeamProposalStatus) => void;
  onTaskPreview: (proposal: TeamProposal) => void;
  onPlanPreview: (proposal: TeamProposal) => void;
  onChiefReview: (proposal: TeamProposal, action: "shortlist" | "defer" | "reject") => void;
  readOnly?: boolean;
}) {
  const action = (status: TeamProposalStatus) => `${proposal.id}:${status}`;
  const isBusy = updating?.startsWith(`${proposal.id}:`) ?? false;
  return (
    <Card>
      <CardContent className="flex flex-col gap-3 py-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-primary" />
              <h4 className="font-semibold">{proposal.title}</h4>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">Origine: {proposal.origin}</p>
            {proposal.category && (
              <p className="mt-1 text-xs text-muted-foreground">Categoria: {proposal.category}</p>
            )}
            {proposal.signals_count && proposal.signals_count > 0 && (
              <p className="mt-1 text-xs text-muted-foreground">
                Segnali raccolti: {proposal.signals_count}
                {proposal.last_signal_at ? ` · ultimo: ${proposal.last_signal_at}` : ""}
              </p>
            )}
            <p className="mt-1 text-xs text-muted-foreground">Formulata: {formatProposalFormulatedDate(proposal)}</p>
            {proposal.source_agent && (
              <p className="mt-1 text-xs text-muted-foreground">Subagente: {proposal.source_agent}</p>
            )}
            {proposal.chief_review_score !== undefined && (
              <p className="mt-1 text-xs text-muted-foreground">
                Chief score: {proposal.chief_review_score}/100 · {proposal.chief_review_status ?? "pending"}
              </p>
            )}
            {proposal.task_id && (
              <p className="mt-1 text-xs text-muted-foreground">Task Kanban: {proposal.task_id}</p>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Badge tone={toneForPriority(proposal.priority)}>{proposal.priority}</Badge>
            <Badge tone={proposal.kind === "evolution" ? "success" : "secondary"}>{kindLabel(proposal.kind)}</Badge>
            <Badge tone={proposalStatusTone(proposal.status)}>{statusLabel(proposal.status)}</Badge>
            <Badge tone="secondary">{reviewLabel(proposal)}</Badge>
            <Badge tone="warning">{riskGateLabel(proposal)}</Badge>
            {proposal.no_auto_dispatch && <Badge tone="success">proposal-only</Badge>}
          </div>
        </div>
        <p className="text-sm text-muted-foreground">{proposal.whyNow}</p>
        {proposal.evidence && (
          <div className="rounded-sm border border-border/60 bg-background/30 p-3 text-sm">
            <div className="mb-1 flex items-center gap-2 font-medium">
              <CircleDot className="h-3.5 w-3.5 text-primary" />
              Evidenza / segnale
            </div>
            <p className="whitespace-pre-wrap text-muted-foreground">{proposal.evidence}</p>
          </div>
        )}
        {(proposal.source_signal || proposal.interpretation) && (
          <div className="rounded-sm border border-primary/40 bg-primary/10 p-3 text-sm">
            <div className="mb-1 font-medium">Segnale autonomo</div>
            {proposal.source_signal && <p className="text-muted-foreground">{proposal.source_signal}</p>}
            {proposal.interpretation && <p className="mt-1 text-muted-foreground">Interpretazione: {formatContractValue(proposal.interpretation)}</p>}
          </div>
        )}
        {proposal.challenge && (
          <div className="rounded-sm border border-warning/50 bg-warning/10 p-3 text-sm">
            <div className="mb-2 font-medium text-warning">Challenge interna</div>
            <p><strong>Supporter:</strong> {proposal.challenge.supporter} — {proposal.challenge.support}</p>
            <p className="mt-1"><strong>Critic:</strong> {proposal.challenge.critic} — {proposal.challenge.challenge}</p>
            <p className="mt-1"><strong>Chief synthesis:</strong> {proposal.challenge.chief_synthesis}</p>
            {proposal.challenge.veto_risk && proposal.challenge.veto_risk !== "none" && (
              <p className="mt-1 text-xs text-muted-foreground">Rischio veto: {proposal.challenge.veto_risk}</p>
            )}
          </div>
        )}
        {(proposal.record_type === "autonomous_proposal_candidate" || proposal.autonomy_level || proposal.supporter_view || proposal.critic_view || proposal.evidence_refs?.length) && (
          <AutonomousProposalContractGrid proposal={proposal} />
        )}
        {!compact && (
          <div className="rounded-sm border border-border/60 bg-background/30 p-3 text-sm">
            <div className="mb-1 flex items-center gap-2 font-medium">
              <CircleDot className="h-3.5 w-3.5 text-primary" />
              Criteri di accettazione
            </div>
            <p className="text-muted-foreground">{proposal.acceptance}</p>
          </div>
        )}
        <div className="grid gap-2 text-xs sm:grid-cols-5">
          <Metric label="Benefit" value={proposal.benefit} />
          <Metric label="Effort" value={proposal.effort} />
          <Metric label="Risk" value={proposal.risk} />
          <Metric label="Confidence" value={proposal.confidence} />
          <div className="rounded-sm border border-primary/40 bg-primary/10 p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Chief</div>
            <div className="mt-1 flex items-center gap-1 font-semibold">
              {recommendationLabel(proposal.recommendation)}
              <ArrowRight className="h-3 w-3" />
            </div>
          </div>
        </div>
        {readOnly ? (
          <div className="flex flex-col gap-2 border-t border-border/60 pt-3 text-sm">
            <div className="rounded-sm border border-success/40 bg-success/10 p-3 text-success">
              Vedi preview per Mission Control
              <div className="mt-1 text-xs text-muted-foreground">
                La preview non crea task. Approvazione e conversione sono separate; dopo la conversione la task entra ready e segue il dispatcher secondo priorità/concorrenza.
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2 border-t border-border/60 pt-3">
          <Button
            size="sm"
            disabled={isBusy || proposal.status === "approvata"}
            onClick={() => onStatus(proposal, "approvata")}
            prefix={updating === action("approvata") ? <Spinner /> : undefined}
          >
            {proposal.status === "approvata" ? "Approvata" : "Approva"}
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.status === "raccomandata"}
            onClick={() => onStatus(proposal, "raccomandata")}
            prefix={updating === action("raccomandata") ? <Spinner /> : undefined}
          >
            Raccomanda
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.status === "parcheggiata"}
            onClick={() => onStatus(proposal, "parcheggiata")}
            prefix={updating === action("parcheggiata") ? <Spinner /> : undefined}
          >
            Parcheggia
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.status === "scartata"}
            onClick={() => onStatus(proposal, "scartata")}
            prefix={updating === action("scartata") ? <Spinner /> : undefined}
          >
            Scarta
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.chief_review_status === "shortlisted"}
            onClick={() => onChiefReview(proposal, "shortlist")}
            prefix={updating === `${proposal.id}:chief-shortlist` ? <Spinner /> : undefined}
          >
            Shortlist Chief
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.chief_review_status === "deferred"}
            onClick={() => onChiefReview(proposal, "defer")}
            prefix={updating === `${proposal.id}:chief-defer` ? <Spinner /> : undefined}
          >
            Defer Chief
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.status === "trasformata_in_task"}
            onClick={() => onTaskPreview(proposal)}
            prefix={updating === `${proposal.id}:convert` ? <Spinner /> : undefined}
          >
            Trasforma in task…
          </Button>
          <Button
            ghost
            size="sm"
            disabled={isBusy || proposal.status === "trasformata_in_task"}
            onClick={() => onPlanPreview(proposal)}
            prefix={updating === `${proposal.id}:convert-plan` ? <Spinner /> : undefined}
          >
            Trasforma in piano…
          </Button>
        </div>
        )}
      </CardContent>
    </Card>
  );
}

function NewProposalForm({
  busy,
  form,
  onChange,
  onSubmit,
}: {
  busy: boolean;
  form: TeamProposalUpsertRequest;
  onChange: (form: TeamProposalUpsertRequest) => void;
  onSubmit: () => void;
}) {
  const update = (patch: Partial<TeamProposalUpsertRequest>) => onChange({ ...form, ...patch });
  const inputClass = "rounded-sm border border-border/70 bg-background/60 px-3 py-2 text-sm text-foreground outline-none focus:border-primary";
  return (
    <Card>
      <CardContent className="flex flex-col gap-4 py-4">
        <div>
          <h3 className="font-semibold">Nuova proposta / segnale Chief</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Inserisce o aggiorna una proposta tramite dedupe. È solo intake: non crea task né dispatch.
          </p>
        </div>
        <div className="grid gap-3 lg:grid-cols-3">
          <label className="flex flex-col gap-1 lg:col-span-2">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Titolo</span>
            <input className={inputClass} value={form.title} onChange={(e) => update({ title: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Tipo</span>
            <select className={inputClass} value={form.kind} onChange={(e) => update({ kind: e.target.value as TeamProposal["kind"] })}>
              <option value="evolution">Sviluppo Hermes</option>
              <option value="operative">Operativa</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Categoria</span>
            <input className={inputClass} value={form.category ?? ""} onChange={(e) => update({ category: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Origine</span>
            <input className={inputClass} value={form.origin ?? ""} onChange={(e) => update({ origin: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Source key / dedupe</span>
            <input className={inputClass} value={form.source_key ?? ""} onChange={(e) => update({ source_key: e.target.value })} placeholder="opzionale" />
          </label>
          <label className="flex flex-col gap-1 lg:col-span-3">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Perché ora</span>
            <textarea className={inputClass} rows={3} value={form.whyNow ?? ""} onChange={(e) => update({ whyNow: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1 lg:col-span-3">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Evidenza / segnale</span>
            <textarea className={inputClass} rows={3} value={form.evidence ?? ""} onChange={(e) => update({ evidence: e.target.value })} />
          </label>
          <label className="flex flex-col gap-1 lg:col-span-3">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">Criteri di accettazione</span>
            <textarea className={inputClass} rows={3} value={form.acceptance ?? ""} onChange={(e) => update({ acceptance: e.target.value })} />
          </label>
        </div>
        <div className="grid gap-3 sm:grid-cols-5">
          <SelectField label="Benefit" value={form.benefit ?? "medium"} values={["high", "medium", "low"]} onChange={(benefit) => update({ benefit: benefit as TeamProposal["benefit"] })} />
          <SelectField label="Effort" value={form.effort ?? "medium"} values={["low", "medium", "high"]} onChange={(effort) => update({ effort: effort as TeamProposal["effort"] })} />
          <SelectField label="Risk" value={form.risk ?? "low"} values={["low", "medium", "high"]} onChange={(risk) => update({ risk: risk as TeamProposal["risk"] })} />
          <SelectField label="Priority" value={form.priority ?? "P2"} values={["P0", "P1", "P2", "P3"]} onChange={(priority) => update({ priority: priority as TeamProposal["priority"] })} />
          <SelectField label="Confidence" value={form.confidence ?? "medium"} values={["high", "medium", "low"]} onChange={(confidence) => update({ confidence: confidence as TeamProposal["confidence"] })} />
        </div>
        <div className="flex justify-end">
          <Button size="sm" onClick={onSubmit} disabled={busy} prefix={busy ? <Spinner /> : undefined}>
            Salva proposta
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function SelectField({
  label,
  onChange,
  value,
  values,
}: {
  label: string;
  onChange: (value: string) => void;
  value: string;
  values: string[];
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-muted-foreground">{label}</span>
      <select
        className="rounded-sm border border-border/70 bg-background/60 px-3 py-2 text-sm text-foreground outline-none focus:border-primary"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {values.map((v) => <option key={v} value={v}>{v}</option>)}
      </select>
    </label>
  );
}

function PlanPreviewDialog({
  busy,
  loading,
  onCancel,
  onConfirm,
  preview,
  proposal,
}: {
  busy: boolean;
  loading: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  preview: TeamProposalPlanPreview | null;
  proposal: TeamProposal;
}) {
  const bodyBlockClass =
    "max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-sm border border-border/60 bg-black/30 p-3 text-sm leading-relaxed text-muted-foreground";

  return (
    <div className="fixed inset-0 z-[100] flex items-stretch justify-center bg-black/70 p-3 backdrop-blur-sm sm:items-center sm:p-4">
      <Card className="flex min-h-0 w-full max-w-6xl flex-col">
        <CardContent className="flex max-h-[calc(100vh-1.5rem)] min-h-0 w-full flex-col gap-4 overflow-hidden py-5 sm:max-h-[90vh]">
          <div className="flex shrink-0 flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <h3 className="text-lg font-semibold">Preview trasformazione in piano</h3>
              <p className="mt-1 max-w-3xl break-words text-sm text-muted-foreground">{proposal.title}</p>
            </div>
            <Badge tone="warning">Preview + confirm</Badge>
          </div>
          {loading && <div className="py-8 text-sm text-muted-foreground"><Spinner className="mr-2" /> Carico preview piano…</div>}
          {preview && (
            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1 text-sm">
              <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_10rem]">
                <Metric label="Parent card" value={preview.title} />
                <Metric label="Task figli" value={String(preview.tasks.length)} />
              </div>
              <div className="rounded-sm border border-success/50 bg-success/10 p-3 text-success">
                Hermes assegnerà automaticamente i task agli agenti più adatti tra i profili disponibili
                {preview.available_profiles?.length ? ` (${preview.available_profiles.join(", ")})` : ""}.
                Coordinatore piano: <strong>{preview.assignee ?? "default"}</strong>. La conferma crea parent e task figli in <strong>ready</strong>: il dispatcher li avvia secondo priorità e concorrenza, senza cron o invii esterni.
              </div>
              <div className="rounded-sm border border-border/70 bg-background/40 p-3">
                <div className="mb-2 font-medium">Parent card</div>
                <pre className={bodyBlockClass}>{preview.body}</pre>
              </div>
              <div className="rounded-sm border border-border/70 bg-background/40 p-3">
                <div className="mb-2 font-medium">Task creati e auto-assegnati</div>
                <div className="grid gap-3">
                  {preview.tasks.map((task, index) => (
                    <div key={task.idempotency_key} className="rounded-sm border border-border/60 bg-background/30 p-3">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div className="min-w-0 font-medium leading-snug">
                          {index + 1}. {task.title}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Badge tone="success">{task.assignee ?? "default"}</Badge>
                          <Badge tone="secondary">{task.initial_status}</Badge>
                        </div>
                      </div>
                      {task.assignment_reason && (
                        <div className="mt-2 text-xs text-muted-foreground">Motivo assegnazione: {task.assignment_reason}</div>
                      )}
                      <div className="mt-1 break-all text-[0.68rem] text-muted-foreground">{task.idempotency_key}</div>
                      <pre className={`mt-3 ${bodyBlockClass}`}>{task.body}</pre>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-sm border border-warning/50 bg-warning/10 p-3 text-warning">
                La conferma crea una parent card e {preview.tasks.length} task figli già assegnati in stato ready. Non crea cron e non invia all’esterno: il dispatch Kanban segue priorità e concorrenza.
              </div>
            </div>
          )}
          <div className="flex shrink-0 justify-end gap-2 border-t border-border/60 pt-3">
            <Button ghost size="sm" onClick={onCancel} disabled={busy}>Annulla</Button>
            <Button size="sm" onClick={onConfirm} disabled={busy || loading || !preview} prefix={busy ? <Spinner /> : undefined}>
              Conferma e lancia piano ready in Kanban
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function TaskPreviewDialog({
  busy,
  loading,
  onCancel,
  onConfirm,
  preview,
  proposal,
}: {
  busy: boolean;
  loading: boolean;
  onCancel: () => void;
  onConfirm: () => void;
  preview: TeamProposalTaskPreview | null;
  proposal: TeamProposal;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <Card>
        <CardContent className="flex max-h-[85vh] w-[min(900px,calc(100vw-2rem))] flex-col gap-4 overflow-hidden py-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold">Preview trasformazione in task</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Confermando verrà creata una sola card Kanban idempotente per “{proposal.title}”.
                Entra ready e segue il dispatcher Kanban secondo priorità/concorrenza; nessun cron o invio esterno.
              </p>
            </div>
            <Badge tone="warning">conferma richiesta</Badge>
          </div>
          {loading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <Spinner className="mr-2 text-primary" />
              Preparo preview…
            </div>
          ) : preview ? (
            <div className="grid min-h-0 gap-3">
              <div className="rounded-sm border border-border/60 bg-background/35 p-3">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Titolo task</div>
                <div className="mt-1 font-semibold">{preview.title}</div>
              </div>
              <div className="grid gap-2 text-xs sm:grid-cols-4">
                <Metric label="Priority" value={String(preview.priority)} />
                <Metric label="Tenant" value={preview.tenant} />
                <Metric label="Workspace" value={preview.workspace_kind} />
                <Metric label="Assignee" value={preview.assignee ?? "non assegnato"} />
                <Metric label="Stato" value={preview.initial_status} />
              </div>
              <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-sm border border-border/60 bg-black/30 p-3 text-xs text-muted-foreground">
                {preview.body}
              </pre>
            </div>
          ) : null}
          <div className="flex flex-wrap justify-end gap-2 border-t border-border/60 pt-3">
            <Button ghost size="sm" onClick={onCancel} disabled={busy}>Annulla</Button>
            <Button size="sm" onClick={onConfirm} disabled={!preview || busy} prefix={busy ? <Spinner /> : undefined}>
              Conferma e lancia task ready in Kanban
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
