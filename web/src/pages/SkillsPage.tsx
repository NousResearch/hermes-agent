import { useEffect, useState, useMemo } from "react";
import {
  Package,
  Search,
  Wrench,
  ChevronDown,
  ChevronRight,
  Filter,
  X,
  History,
  FileText,
  CheckCircle2,
  GitPullRequest,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SkillChangeDetail, SkillChangeEvent, SkillGovernanceProposal, SkillInfo, ToolsetInfo } from "@/lib/api";
import {
  formatSkillChangeTime,
  latestChangeBySkill,
  loadSkillChangesBestEffort,
  reasonKindLabel,
  reviewStatusLabel,
} from "@/lib/skillChanges";
import {
  allowedProposalDecisionStatuses,
  extractFetchErrorMessage,
  proposalDecisionActionLabel,
  proposalDecisionActionStatuses,
  proposalDecisionTransitionHint,
  proposalDecisionNote,
  proposalPreviewText,
  proposalStatusLabel,
  unavailableProposalDecisionStatuses,
} from "@/lib/skillGovernance";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/i18n";

/* ------------------------------------------------------------------ */
/*  Types & helpers                                                    */
/* ------------------------------------------------------------------ */

interface CategoryGroup {
  name: string;        // display name
  key: string;         // raw key (or "__none__")
  skills: SkillInfo[];
  enabledCount: number;
}

const CATEGORY_LABELS: Record<string, string> = {
  mlops: "MLOps",
  "mlops/cloud": "MLOps / Cloud",
  "mlops/evaluation": "MLOps / Evaluation",
  "mlops/inference": "MLOps / Inference",
  "mlops/models": "MLOps / Models",
  "mlops/training": "MLOps / Training",
  "mlops/vector-databases": "MLOps / Vector DBs",
  mcp: "MCP",
  "red-teaming": "Red Teaming",
  ocr: "OCR",
  p5js: "p5.js",
  ai: "AI",
  ux: "UX",
  ui: "UI",
};

function prettyCategory(raw: string | null | undefined, generalLabel: string): string {
  if (!raw) return generalLabel;
  if (CATEGORY_LABELS[raw]) return CATEGORY_LABELS[raw];
  return raw
    .split(/[-_/]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function proposalRiskVariant(risk: string): "default" | "secondary" | "destructive" | "outline" {
  if (risk === "high") return "destructive";
  if (risk === "medium") return "outline";
  if (risk === "low") return "secondary";
  return "secondary";
}

function proposalStatusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "approved") return "default";
  if (status === "rejected") return "destructive";
  if (status === "pending" || status === "needs_changes") return "outline";
  return "secondary";
}

function proposalActionVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  if (status === "approved") return "default";
  if (status === "rejected") return "destructive";
  if (status === "deferred") return "secondary";
  return "outline";
}


/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [skillChanges, setSkillChanges] = useState<SkillChangeEvent[]>([]);
  const [governanceProposals, setGovernanceProposals] = useState<SkillGovernanceProposal[]>([]);
  const [selectedProposal, setSelectedProposal] = useState<SkillGovernanceProposal | null>(null);
  const [decidingProposal, setDecidingProposal] = useState<string | null>(null);
  const [selectedChange, setSelectedChange] = useState<SkillChangeDetail | null>(null);
  const [loadingChangeDetail, setLoadingChangeDetail] = useState(false);
  const [reviewingChange, setReviewingChange] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  // Start collapsed by default
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string> | "all">("all");
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  useEffect(() => {
    Promise.all([
      api.getSkills(),
      api.getToolsets(),
      loadSkillChangesBestEffort(
        () => api.getSkillChanges({ limit: 20 }),
        () => showToast("Skill change history unavailable", "error"),
      ),
      api.getSkillGovernanceProposals({ limit: 20 }).catch(() => {
        showToast("Curator proposals unavailable", "error");
        return [] as SkillGovernanceProposal[];
      }),
    ])
      .then(([s, tsets, changes, proposals]) => {
        setSkills(s);
        setToolsets(tsets);
        setSkillChanges(changes);
        setGovernanceProposals(proposals);
        setSelectedProposal(proposals[0] ?? null);
        if (proposals[0]) {
          api.getSkillGovernanceProposal(proposals[0].proposal_id)
            .then(setSelectedProposal)
            .catch(() => undefined);
        }
      })
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
  }, []);

  /* ---- Toggle skill ---- */
  const handleToggleSkill = async (skill: SkillInfo) => {
    setTogglingSkills((prev) => new Set(prev).add(skill.name));
    try {
      await api.toggleSkill(skill.name, !skill.enabled);
      setSkills((prev) =>
        prev.map((s) =>
          s.name === skill.name ? { ...s, enabled: !s.enabled } : s
        )
      );
      showToast(
        `${skill.name} ${skill.enabled ? t.common.disabled : t.common.enabled}`,
        "success"
      );
    } catch {
      showToast(`${t.common.failedToToggle} ${skill.name}`, "error");
    } finally {
      setTogglingSkills((prev) => {
        const next = new Set(prev);
        next.delete(skill.name);
        return next;
      });
    }
  };

  const handleSelectChange = async (change: SkillChangeEvent) => {
    setLoadingChangeDetail(true);
    try {
      const detail = await api.getSkillChange(change.event_id);
      setSelectedChange(detail);
    } catch {
      showToast(`Failed to load change for ${change.skill}`, "error");
    } finally {
      setLoadingChangeDetail(false);
    }
  };

  const handleMarkReviewed = async (eventId: string) => {
    setReviewingChange(eventId);
    try {
      const updated = await api.reviewSkillChange(eventId, "reviewed");
      setSkillChanges((prev) =>
        prev.map((event) => (event.event_id === eventId ? { ...event, ...updated } : event))
      );
      setSelectedChange((prev) =>
        prev?.event_id === eventId ? { ...prev, ...updated, diff_text: prev.diff_text } : prev
      );
      showToast("Skill change marked reviewed", "success");
    } catch {
      showToast("Failed to mark skill change reviewed", "error");
    } finally {
      setReviewingChange(null);
    }
  };

  const handleSelectProposal = async (proposal: SkillGovernanceProposal) => {
    try {
      const detail = await api.getSkillGovernanceProposal(proposal.proposal_id);
      setSelectedProposal(detail);
    } catch {
      showToast(`Failed to load proposal ${proposal.proposal_id}`, "error");
    }
  };

  const handleProposalDecision = async (proposalId: string, status: string) => {
    setDecidingProposal(proposalId);
    try {
      const updated = await api.decideSkillGovernanceProposal(
        proposalId,
        status,
        proposalDecisionNote(status),
      );
      setGovernanceProposals((prev) =>
        prev.map((proposal) => (proposal.proposal_id === proposalId ? { ...proposal, ...updated } : proposal))
      );
      setSelectedProposal((prev) =>
        prev?.proposal_id === proposalId ? { ...prev, ...updated } : prev
      );
      showToast(`Proposal marked ${proposalStatusLabel(status).toLowerCase()}`, "success");
    } catch (error) {
      showToast(extractFetchErrorMessage(error), "error");
    } finally {
      setDecidingProposal(null);
    }
  };

  /* ---- Derived data ---- */
  const lowerSearch = search.toLowerCase();

  const filteredSkills = useMemo(() => {
    return skills.filter((s) => {
      const matchesSearch =
        !search ||
        s.name.toLowerCase().includes(lowerSearch) ||
        s.description.toLowerCase().includes(lowerSearch) ||
        (s.category ?? "").toLowerCase().includes(lowerSearch);
      const matchesCategory =
        !activeCategory ||
        (activeCategory === "__none__" ? !s.category : s.category === activeCategory);
      return matchesSearch && matchesCategory;
    });
  }, [skills, search, lowerSearch, activeCategory]);

  const categoryGroups: CategoryGroup[] = useMemo(() => {
    const map = new Map<string, SkillInfo[]>();
    for (const s of filteredSkills) {
      const key = s.category || "__none__";
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(s);
    }
    // Sort: General first, then alphabetical
    const entries = [...map.entries()].sort((a, b) => {
      if (a[0] === "__none__") return -1;
      if (b[0] === "__none__") return 1;
      return a[0].localeCompare(b[0]);
    });
    return entries.map(([key, list]) => ({
      key,
      name: prettyCategory(key === "__none__" ? null : key, t.common.general),
      skills: list.sort((a, b) => a.name.localeCompare(b.name)),
      enabledCount: list.filter((s) => s.enabled).length,
    }));
  }, [filteredSkills]);

  const allCategories = useMemo(() => {
    const cats = new Map<string, number>();
    for (const s of skills) {
      const key = s.category || "__none__";
      cats.set(key, (cats.get(key) || 0) + 1);
    }
    return [...cats.entries()]
      .sort((a, b) => {
        if (a[0] === "__none__") return -1;
        if (b[0] === "__none__") return 1;
        return a[0].localeCompare(b[0]);
      })
      .map(([key, count]) => ({ key, name: prettyCategory(key === "__none__" ? null : key, t.common.general), count }));
  }, [skills]);

  const enabledCount = skills.filter((s) => s.enabled).length;
  const latestChanges = useMemo(() => latestChangeBySkill(skillChanges), [skillChanges]);
  const unreviewedChangeCount = skillChanges.filter((event) => event.review_status === "unreviewed").length;
  const pendingProposalCount = governanceProposals.filter((proposal) => proposal.decision_status === "pending").length;
  const selectedProposalPreview = selectedProposal ? proposalPreviewText(selectedProposal) : null;
  const selectedAllowedProposalDecisions = selectedProposal ? allowedProposalDecisionStatuses(selectedProposal) : [];
  const selectedProposalDecisionActions = selectedProposal ? proposalDecisionActionStatuses(selectedProposal) : [];
  const selectedUnavailableProposalDecisions = selectedProposal ? unavailableProposalDecisionStatuses(selectedProposal) : [];
  const selectedProposalTransitionHint = selectedProposal ? proposalDecisionTransitionHint(selectedProposal) : null;

  const filteredToolsets = useMemo(() => {
    return toolsets.filter(
      (ts) =>
        !search ||
        ts.name.toLowerCase().includes(lowerSearch) ||
        ts.label.toLowerCase().includes(lowerSearch) ||
        ts.description.toLowerCase().includes(lowerSearch)
    );
  }, [toolsets, search, lowerSearch]);

  const isCollapsed = (key: string): boolean => {
    if (collapsedCategories === "all") return true;
    return collapsedCategories.has(key);
  };

  const toggleCollapse = (key: string) => {
    setCollapsedCategories((prev) => {
      if (prev === "all") {
        // Switching from "all collapsed" → expand just this one
        const allKeys = new Set(categoryGroups.map((g) => g.key));
        allKeys.delete(key);
        return allKeys;
      }
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  /* ---- Loading ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      {/* ═══════════════ Header + Search ═══════════════ */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Package className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-base font-semibold">{t.skills.title}</h1>
          <span className="text-xs text-muted-foreground">
            {t.skills.enabledOf.replace("{enabled}", String(enabledCount)).replace("{total}", String(skills.length))}
          </span>
        </div>
      </div>

      {/* ═══════════════ Search + Category Filter ═══════════════ */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            className="pl-9"
            placeholder={t.skills.searchPlaceholder}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && (
            <button
              type="button"
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              onClick={() => setSearch("")}
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Category pills */}
      {allCategories.length > 1 && (
        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <button
            type="button"
            className={`inline-flex items-center px-3 py-1 text-xs font-medium transition-colors cursor-pointer ${
              !activeCategory
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
            }`}
            onClick={() => setActiveCategory(null)}
          >
            {t.skills.all} ({skills.length})
          </button>
          {allCategories.map(({ key, name, count }) => (
            <button
              key={key}
              type="button"
              className={`inline-flex items-center px-3 py-1 text-xs font-medium transition-colors cursor-pointer ${
                activeCategory === key
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
              }`}
              onClick={() =>
                setActiveCategory(activeCategory === key ? null : key)
              }
            >
              {name}
              <span className="ml-1 opacity-60">{count}</span>
            </button>
          ))}
        </div>
      )}

      {/* Curator proposal inbox */}
      <Card>
        <CardHeader className="py-3 px-4">
          <div className="flex items-center justify-between gap-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <GitPullRequest className="h-4 w-4 text-muted-foreground" />
              Curator proposals
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={pendingProposalCount > 0 ? "outline" : "secondary"} className="text-[10px]">
                {pendingProposalCount} pending
              </Badge>
              <Badge variant="secondary" className="text-[10px]">
                decision-only MVP
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="px-4 pb-4 pt-0">
          {governanceProposals.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No Curator proposals imported yet. Run a dry-run importer to turn Curator reports into PM-reviewable cards.
            </p>
          ) : (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(340px,0.95fr)]">
              <div className="grid gap-2">
                {governanceProposals.slice(0, 8).map((proposal) => (
                  <button
                    key={proposal.proposal_id}
                    type="button"
                    className={`text-left rounded-md border px-3 py-2 transition-colors hover:bg-muted/40 ${
                      selectedProposal?.proposal_id === proposal.proposal_id ? "border-primary/70 bg-primary/5" : "border-border"
                    }`}
                    onClick={() => handleSelectProposal(proposal)}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono-ui text-xs text-foreground truncate">{proposal.title}</span>
                          <Badge variant={proposalRiskVariant(proposal.risk_level)} className="text-[10px]">
                            {proposal.risk_level} risk
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                          {proposal.pm_summary || proposal.rationale}
                        </p>
                        <p className="mt-1 text-[10px] text-muted-foreground">
                          {proposal.target_skills.length} affected · {proposal.action} · {proposal.source_run_id || proposal.source}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <Badge variant={proposalStatusVariant(proposal.decision_status)} className="text-[10px]">
                          {proposalStatusLabel(proposal.decision_status)}
                        </Badge>
                        <p className="mt-1 text-[10px] text-muted-foreground">
                          {formatSkillChangeTime(proposal.updated_at)}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              <div className="rounded-md border border-border bg-muted/20 p-3 min-h-[220px]">
                {selectedProposal ? (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h3 className="font-mono-ui text-sm">{selectedProposal.title}</h3>
                        <p className="text-xs text-muted-foreground">
                          {selectedProposal.action} · {selectedProposal.source} · schema v{selectedProposal.schema_version}
                        </p>
                      </div>
                      <Badge variant={proposalStatusVariant(selectedProposal.decision_status)} className="text-[10px]">
                        {proposalStatusLabel(selectedProposal.decision_status)}
                      </Badge>
                    </div>

                    <div className="grid gap-2 text-xs text-muted-foreground">
                      <p>{selectedProposal.pm_summary || selectedProposal.rationale}</p>
                      {selectedProposal.impact_summary && <p>Impact: {selectedProposal.impact_summary}</p>}
                      <div className="flex flex-wrap gap-2">
                        <Badge variant={proposalRiskVariant(selectedProposal.risk_level)} className="text-[10px]">
                          {selectedProposal.risk_level} risk
                        </Badge>
                        <Badge variant="secondary" className="text-[10px]">
                          Codex: {selectedProposal.codex_review_status}
                        </Badge>
                        <Badge variant="secondary" className="text-[10px]">
                          Pins: {selectedProposal.pin_policy_status}
                        </Badge>
                      </div>
                    </div>

                    {selectedProposal.target_skills.length > 0 && (
                      <div>
                        <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">Affected skills</p>
                        <div className="flex flex-wrap gap-1">
                          {selectedProposal.target_skills.map((skill) => (
                            <Badge key={skill} variant="secondary" className="text-[10px] font-mono">
                              {skill}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedProposal.evidence.length > 0 && (
                      <div>
                        <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">Evidence</p>
                        <ul className="list-disc pl-4 text-xs text-muted-foreground">
                          {selectedProposal.evidence.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {selectedProposalPreview && (
                      <div>
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            {selectedProposalPreview.title}
                          </p>
                          {selectedProposalPreview.kind !== "diff" && (
                            <Badge variant="secondary" className="text-[10px]">
                              not an apply diff
                            </Badge>
                          )}
                        </div>
                        <pre className="max-h-72 overflow-auto rounded border border-border bg-background/80 p-3 text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap">
                          {selectedProposalPreview.body}
                        </pre>
                      </div>
                    )}

                    <div className="rounded border border-border bg-background/70 p-2 text-[11px] text-muted-foreground">
                      <div className="flex items-start gap-2">
                        <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        <p>
                          MVP records the PM decision only. It does not apply, archive, delete, unpin, repin, or mutate any skill.
                        </p>
                      </div>
                    </div>

                    {selectedProposal.decision_note && (
                      <div className="rounded border border-border bg-background/70 p-2 text-[11px] text-muted-foreground">
                        Decision note: {selectedProposal.decision_note}
                      </div>
                    )}

                    <div className="flex flex-wrap gap-2">
                      <p className="basis-full text-[10px] text-muted-foreground">
                        Allowed now: {selectedAllowedProposalDecisions.map(proposalStatusLabel).join(", ")}
                      </p>
                      {selectedProposalTransitionHint && (
                        <p className="basis-full text-[11px] text-muted-foreground">
                          {selectedProposalTransitionHint}
                        </p>
                      )}
                      {selectedProposalDecisionActions.map((status) => (
                        <Button
                          key={status}
                          size="sm"
                          variant={proposalActionVariant(status)}
                          disabled={decidingProposal === selectedProposal.proposal_id}
                          onClick={() => handleProposalDecision(selectedProposal.proposal_id, status)}
                        >
                          {status === "bad_test_target" && <AlertTriangle className="h-3.5 w-3.5" />}
                          {proposalDecisionActionLabel(status)}
                        </Button>
                      ))}
                      {selectedUnavailableProposalDecisions.length > 0 && (
                        <p className="basis-full text-[10px] text-muted-foreground">
                          Unavailable now: {selectedUnavailableProposalDecisions.map(proposalStatusLabel).join(", ")}
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Select a Curator proposal to inspect rationale, risk, evidence, affected skills, and decision state.
                  </p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Skill change ledger */}
      <Card>
        <CardHeader className="py-3 px-4">
          <div className="flex items-center justify-between gap-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <History className="h-4 w-4 text-muted-foreground" />
              Skill change history
            </CardTitle>
            <Badge variant={unreviewedChangeCount > 0 ? "outline" : "secondary"} className="text-[10px]">
              {unreviewedChangeCount} unreviewed
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="px-4 pb-4 pt-0">
          {skillChanges.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No skill changes recorded yet. New ledger events will appear here with reasons and diffs.
            </p>
          ) : (
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.9fr)]">
              <div className="grid gap-2">
                {skillChanges.slice(0, 8).map((change) => (
                  <button
                    key={change.event_id}
                    type="button"
                    className={`text-left rounded-md border px-3 py-2 transition-colors hover:bg-muted/40 ${
                      selectedChange?.event_id === change.event_id ? "border-primary/70 bg-primary/5" : "border-border"
                    }`}
                    onClick={() => handleSelectChange(change)}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-mono-ui text-xs text-foreground truncate">{change.skill}</span>
                          <Badge variant="secondary" className="text-[10px] font-normal">
                            {change.action}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                          {change.reason || reasonKindLabel(change.reason_kind)}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <Badge
                          variant={change.review_status === "unreviewed" ? "outline" : "secondary"}
                          className="text-[10px]"
                        >
                          {reviewStatusLabel(change.review_status)}
                        </Badge>
                        <p className="mt-1 text-[10px] text-muted-foreground">
                          {formatSkillChangeTime(change.timestamp)}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>

              <div className="rounded-md border border-border bg-muted/20 p-3 min-h-[160px]">
                {loadingChangeDetail ? (
                  <div className="flex items-center justify-center py-10">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  </div>
                ) : selectedChange ? (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h3 className="font-mono-ui text-sm">{selectedChange.skill}</h3>
                        <p className="text-xs text-muted-foreground">
                          {selectedChange.action} · {reasonKindLabel(selectedChange.reason_kind)}
                        </p>
                      </div>
                      {selectedChange.review_status !== "reviewed" && (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={reviewingChange === selectedChange.event_id}
                          onClick={() => handleMarkReviewed(selectedChange.event_id)}
                        >
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          Mark reviewed
                        </Button>
                      )}
                    </div>

                    <div className="text-xs text-muted-foreground">
                      <p>{selectedChange.reason || reasonKindLabel(selectedChange.reason_kind)}</p>
                      <p className="mt-1">
                        Source: {selectedChange.source} · Actor: {selectedChange.actor}
                      </p>
                    </div>

                    {selectedChange.changed_files.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {selectedChange.changed_files.map((file) => (
                          <Badge key={file} variant="secondary" className="text-[10px] font-mono">
                            <FileText className="h-3 w-3" />
                            {file}
                          </Badge>
                        ))}
                      </div>
                    )}

                    <pre className="max-h-72 overflow-auto rounded bg-background/80 p-3 text-[11px] leading-relaxed text-muted-foreground">
                      {selectedChange.diff_text || "No diff artifact recorded for this event."}
                    </pre>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Select a change to inspect provenance, reason, files, and raw diff.
                  </p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ═══════════════ Skills by Category ═══════════════ */}
      <section className="flex flex-col gap-3">

        {filteredSkills.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center text-sm text-muted-foreground">
              {skills.length === 0
                ? t.skills.noSkills
                : t.skills.noSkillsMatch}
            </CardContent>
          </Card>
        ) : (
          categoryGroups.map(({ key, name, skills: catSkills, enabledCount: catEnabled }) => {
            const collapsed = isCollapsed(key);
            return (
              <Card key={key}>
                <CardHeader
                  className="cursor-pointer select-none py-3 px-4"
                  onClick={() => toggleCollapse(key)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {collapsed ? (
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                      )}
                      <CardTitle className="text-sm font-medium">{name}</CardTitle>
                      <Badge variant="secondary" className="text-[10px] font-normal">
                        {t.skills.skillCount.replace("{count}", String(catSkills.length)).replace("{s}", catSkills.length !== 1 ? "s" : "")}
                      </Badge>
                    </div>
                    <Badge
                      variant={catEnabled === catSkills.length ? "success" : "outline"}
                      className="text-[10px]"
                    >
                      {t.skills.enabledOf.replace("{enabled}", String(catEnabled)).replace("{total}", String(catSkills.length))}
                    </Badge>
                  </div>
                </CardHeader>

                {collapsed ? (
                  /* Peek: show first few skill names so collapsed isn't blank */
                  <div className="px-4 pb-3 flex items-center min-h-[28px]">
                    <p className="text-xs text-muted-foreground/60 truncate leading-normal">
                      {catSkills.slice(0, 4).map((s) => s.name).join(", ")}
                      {catSkills.length > 4 && `, ${t.skills.more.replace("{count}", String(catSkills.length - 4))}`}
                    </p>
                  </div>
                ) : (
                  <CardContent className="pt-0 px-4 pb-3">
                    <div className="grid gap-1">
                      {catSkills.map((skill) => (
                        <div
                          key={skill.name}
                          className="group flex items-start gap-3 rounded-md px-3 py-2.5 transition-colors hover:bg-muted/40"
                        >
                          <div className="pt-0.5 shrink-0">
                            <Switch
                              checked={skill.enabled}
                              onCheckedChange={() => handleToggleSkill(skill)}
                              disabled={togglingSkills.has(skill.name)}
                            />
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span
                                className={`font-mono-ui text-sm ${
                                  skill.enabled
                                    ? "text-foreground"
                                    : "text-muted-foreground"
                                }`}
                              >
                                {skill.name}
                              </span>
                              {latestChanges.get(skill.name) && (
                                <Badge
                                  variant={latestChanges.get(skill.name)?.review_status === "unreviewed" ? "outline" : "secondary"}
                                  className="text-[10px]"
                                >
                                  changed
                                </Badge>
                              )}
                            </div>
                            <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                              {skill.description || t.skills.noDescription}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                )}
              </Card>
            );
          })
        )}
      </section>

      {/* ═══════════════ Toolsets ═══════════════ */}
      <section className="flex flex-col gap-4">
        <h2 className="text-sm font-medium text-muted-foreground flex items-center gap-2">
          <Wrench className="h-4 w-4" />
          {t.skills.toolsets} ({filteredToolsets.length})
        </h2>

        {filteredToolsets.length === 0 ? (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {t.skills.noToolsetsMatch}
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {filteredToolsets.map((ts) => {
              // Strip emoji prefix from label for cleaner display
              const labelText = ts.label.replace(/^[\p{Emoji}\s]+/u, "").trim() || ts.name;
              const emoji = ts.label.match(/^[\p{Emoji}]+/u)?.[0] || "🔧";

              return (
                <Card key={ts.name} className="relative overflow-hidden">
                  <CardContent className="py-4">
                    <div className="flex items-start gap-3">
                      <div className="text-2xl shrink-0 leading-none mt-0.5">{emoji}</div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium text-sm">{labelText}</span>
                          <Badge
                            variant={ts.enabled ? "success" : "outline"}
                            className="text-[10px]"
                          >
                            {ts.enabled ? t.common.active : t.common.inactive}
                          </Badge>
                        </div>
                        <p className="text-xs text-muted-foreground mb-2">
                          {ts.description}
                        </p>
                        {ts.enabled && !ts.configured && (
                          <p className="text-[10px] text-amber-300/80 mb-2">
                            {t.skills.setupNeeded}
                          </p>
                        )}
                        {ts.tools.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {ts.tools.map((tool) => (
                              <Badge
                                key={tool}
                                variant="secondary"
                                className="text-[10px] font-mono"
                              >
                                {tool}
                              </Badge>
                            ))}
                          </div>
                        )}
                        {ts.tools.length === 0 && (
                          <span className="text-[10px] text-muted-foreground/60">
                            {ts.enabled ? `${ts.name} toolset` : t.skills.disabledForCli}
                          </span>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
