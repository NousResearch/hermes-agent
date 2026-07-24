import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, CircleDot, Eye, PauseCircle, Radar, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { RadarHermesProposal, RadarHermesSnapshot } from "@/lib/api";

const LEVEL_LABEL: Record<string, string> = {
  high: "Alto",
  medium: "Medio",
  low: "Basso",
};

const APPROVAL_LABEL: Record<string, string> = {
  candidate: "Candidata",
  needs_review: "Richiede review",
  approved_for_spec: "Approvata per spec",
  approved_for_kanban: "Approvata per Kanban",
  parked: "Parcheggiata",
  rejected: "Scartata",
  done: "Già fatta",
};

function formatDate(value?: string | null): string {
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

function guardrailText(snapshot: RadarHermesSnapshot | null): string {
  const sideEffects = snapshot?.source_summary.side_effects;
  const zeroSideEffects = sideEffects
    ? !sideEffects.kanban_mutated && !sideEffects.cron_created && !sideEffects.external_send && !sideEffects.subagent_spawned
    : true;
  return zeroSideEffects
    ? "Side effects: 0 — nessuna card, cron, invio esterno o automazione creata."
    : "Attenzione: lo snapshot segnala side effect inattesi. Verificare prima di procedere.";
}

function ProposalCard({ proposal, label, tone = "default" }: { proposal: RadarHermesProposal; label?: string; tone?: "default" | "warn" | "muted" }) {
  const toneClass =
    tone === "warn"
      ? "border-amber-500/30 bg-amber-500/5"
      : tone === "muted"
        ? "border-muted-foreground/20 bg-muted/20"
        : "border-primary/20 bg-card";
  return (
    <Card className={toneClass}>
      <CardContent className="space-y-4 p-4">
        <div className="flex flex-wrap items-center gap-2">
          {label ? <Badge tone="secondary">{label}</Badge> : null}
          <Badge tone="outline">{APPROVAL_LABEL[proposal.approval_state] ?? proposal.approval_state}</Badge>
          <Badge tone="outline">Gate: approvazione richiesta</Badge>
        </div>
        <div className="space-y-2">
          <div className="flex items-start gap-2">
            <CircleDot className="mt-1 h-4 w-4 shrink-0 text-primary" />
            <div>
              <h3 className="text-base font-semibold leading-tight text-foreground">{proposal.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">Perché ora: {proposal.rationale || proposal.source.excerpt || "evidenza sintetica non disponibile"}</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span>Impatto: {LEVEL_LABEL[proposal.priority.impact] ?? proposal.priority.impact}</span>
            <span>· Effort: {LEVEL_LABEL[proposal.priority.effort] ?? proposal.priority.effort}</span>
            <span>· Rischio: {LEVEL_LABEL[proposal.priority.risk] ?? proposal.priority.risk}</span>
            <span>· Profilo: {proposal.suggested_assignee}</span>
          </div>
        </div>
        <div className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">
          <div className="font-medium text-foreground">Gate read-only</div>
          Preview/spec/Kanban richiedono approvazione esplicita separata. In v1 questa card non avvia task, dispatch, cron o invii.
        </div>
        <details className="rounded-lg border border-border p-3 text-sm">
          <summary className="cursor-pointer font-medium">Vedi dettaglio</summary>
          <div className="mt-3 space-y-2 text-muted-foreground">
            <p><span className="font-medium text-foreground">Fonte:</span> {proposal.source.kind} · {proposal.source.source_id}</p>
            <p><span className="font-medium text-foreground">Evidenza:</span> {proposal.evidence[0]?.summary || proposal.source.excerpt || "n/d"}</p>
            <p><span className="font-medium text-foreground">Score:</span> {proposal.ranking.score}/100 · ranking {proposal.ranking.block} #{proposal.ranking.rank}</p>
            <p><span className="font-medium text-foreground">Stato approval:</span> {APPROVAL_LABEL[proposal.approval_state] ?? proposal.approval_state}</p>
          </div>
        </details>
        <div className="flex flex-wrap gap-2">
          <Button type="button" ghost size="sm">
            <Eye className="mr-2 h-4 w-4" /> Vedi dettaglio
          </Button>
          <Button type="button" ghost size="sm" disabled>
            Prepara spec — non attivo v1
          </Button>
          <Button type="button" ghost size="sm" disabled>
            Approva Kanban — non attivo v1
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function SnapshotPill({ icon: Icon, label, value }: { icon: typeof CheckCircle2; label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-border bg-card/70 px-3 py-2 text-sm">
      <Icon className="h-4 w-4 text-primary" />
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold text-foreground">{value}</span>
    </div>
  );
}

export default function RadarHermesPage() {
  const [snapshot, setSnapshot] = useState<RadarHermesSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async (showSpinner = true) => {
    if (showSpinner) {
      setLoading(true);
      setError(null);
    }
    try {
      setSnapshot(await api.getRadarHermes());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void load(false);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const top = snapshot?.blocks.top ?? [];
  const controversial = snapshot?.blocks.controversial ?? null;
  const parkable = snapshot?.blocks.parkable ?? null;
  const snapshotLine = useMemo(() => {
    if (!snapshot) return "Caricamento Radar Hermes…";
    return `${top.length} proposte candidate · ${controversial ? 1 : 0} trade-off da decidere · ${parkable ? 1 : 0} idea parcheggiata`;
  }, [controversial, parkable, snapshot, top.length]);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-4 sm:p-6">
      <section className="rounded-3xl border border-border bg-card/80 p-5 shadow-sm sm:p-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Radar className="h-5 w-5 text-primary" />
              <H2 className="m-0">Radar Hermes</H2>
              <Badge tone="secondary">Read-only v1</Badge>
              <Badge tone="outline">Approval required</Badge>
              <Badge tone="outline">No auto-dispatch</Badge>
            </div>
            <p className="max-w-3xl text-sm text-muted-foreground">
              Top idee di sviluppo Hermes dal team, pronte per decisione — non per esecuzione automatica.
              Radar Hermes suggerisce micro-slice di sviluppo: preview e Kanban richiedono approvazione esplicita separata.
            </p>
          </div>
          <Button type="button" ghost onClick={() => void load()} disabled={loading}>
            {loading ? <Spinner className="mr-2 h-4 w-4" /> : null}
            Aggiorna
          </Button>
        </div>
        <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <SnapshotPill icon={CheckCircle2} label="Top" value={top.length} />
          <SnapshotPill icon={AlertTriangle} label="Controversa" value={controversial ? 1 : 0} />
          <SnapshotPill icon={PauseCircle} label="Parcheggiabile" value={parkable ? 1 : 0} />
          <SnapshotPill icon={ShieldCheck} label="Side effects" value="0" />
        </div>
        <p className="mt-4 text-sm text-muted-foreground">{snapshotLine}</p>
        <p className="text-sm text-muted-foreground">{guardrailText(snapshot)}</p>
        {snapshot ? (
          <p className="text-xs text-muted-foreground">
            Aggiornato: {formatDate(snapshot.generated_at)} · Fonti: {snapshot.source_summary.sources_read.join(", ")}
          </p>
        ) : null}
      </section>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="p-4 text-sm text-destructive">Errore Radar Hermes: {error}</CardContent>
        </Card>
      ) : null}

      {loading && !snapshot ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner className="h-4 w-4" /> Caricamento snapshot…</div>
      ) : null}

      {snapshot?.empty_state ? (
        <Card>
          <CardContent className="p-4">
            <h3 className="font-semibold">{snapshot.empty_state.title}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{snapshot.empty_state.message}</p>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <section className="space-y-3">
          <div>
            <h2 className="text-lg font-semibold">Top proposte evolutive</h2>
            <p className="text-sm text-muted-foreground">Micro-slice candidate, ordinate per impatto e qualità evidenza. Max 5.</p>
          </div>
          {top.map((proposal, index) => (
            <ProposalCard key={proposal.id} proposal={proposal} label={`${index + 1}. Top proposta`} />
          ))}
        </section>

        <aside className="space-y-6">
          <section className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Proposta controversa</h2>
              <p className="text-sm text-muted-foreground">Un trade-off da decidere prima di qualsiasi azione.</p>
            </div>
            {controversial ? <ProposalCard proposal={controversial} label="Trade-off" tone="warn" /> : <EmptyBlock text="Nessuna controversia source-grounded pronta." />}
          </section>

          <section className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Proposta parcheggiabile</h2>
              <p className="text-sm text-muted-foreground">Idea utile ma non urgente, tenuta visibile senza rumore.</p>
            </div>
            {parkable ? <ProposalCard proposal={parkable} label="Parcheggiabile" tone="muted" /> : <EmptyBlock text="Nessuna proposta parcheggiabile emersa." />}
          </section>
        </aside>
      </div>
    </div>
  );
}

function EmptyBlock({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-sm text-muted-foreground">{text}</CardContent>
    </Card>
  );
}
