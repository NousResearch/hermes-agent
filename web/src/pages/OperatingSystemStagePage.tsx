import { useEffect, useMemo, useState } from "react";
import { Activity, PlayCircle, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  ProgressMetric,
  StatusPill,
  type DataTableColumn,
} from "@hermes/dashboard-kit";
import { operatingSystemStages, type OperatingSystemStage } from "./operating-system-data";
import {
  loadOperatingRuntimeState,
  loadOperatingRuntimeStateFromServer,
  recordsForStage,
  runRuntimeReadinessCheck,
  runServerRuntimeReadinessCheck,
  runtimeSummaryForStage,
  type RuntimeEvidenceRecord,
} from "./operating-runtime";

type StageRow = OperatingSystemStage["rows"][number];

const columns: DataTableColumn<StageRow>[] = [
  { id: "capability", header: "Capability", accessor: (row) => row.capability, sortValue: (row) => row.capability },
  {
    id: "state",
    header: "State",
    accessor: (row) => <StatusPill tone={toneForState(row.state)}>{row.state}</StatusPill>,
    sortValue: (row) => row.state,
  },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "next", header: "Next Step", accessor: (row) => row.nextStep, sortValue: (row) => row.nextStep },
];

const evidenceColumns: DataTableColumn<RuntimeEvidenceRecord>[] = [
  { id: "subject", header: "Subject", accessor: (row) => row.subject, sortValue: (row) => row.subject },
  { id: "kind", header: "Kind", accessor: (row) => row.kind, sortValue: (row) => row.kind },
  {
    id: "state",
    header: "State",
    accessor: (row) => <StatusPill tone={toneForRuntimeState(row.state)}>{row.state}</StatusPill>,
    sortValue: (row) => row.state,
  },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "detail", header: "Detail", accessor: (row) => row.detail, sortValue: (row) => row.detail },
];

interface OperatingSystemStagePageProps {
  version: OperatingSystemStage["version"];
}

export function OperatingSystemStagePage({ version }: OperatingSystemStagePageProps) {
  const stage = operatingSystemStages.find((item) => item.version === version);
  const [runtimeState, setRuntimeState] = useState(loadOperatingRuntimeState);
  const [runtimeMode, setRuntimeMode] = useState<"server" | "local">("local");
  const [runtimeError, setRuntimeError] = useState("");
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    refreshRuntimeState().catch(() => undefined);
    return () => {
      cancelled = true;
    };

    async function refreshRuntimeState() {
      try {
        const next = await loadOperatingRuntimeStateFromServer();
        if (cancelled) return;
        setRuntimeState(next);
        setRuntimeMode("server");
        setRuntimeError("");
      } catch (error) {
        if (cancelled) return;
        setRuntimeState(loadOperatingRuntimeState());
        setRuntimeMode("local");
        setRuntimeError(error instanceof Error ? error.message : "Server runtime unavailable.");
      }
    }
  }, [version]);

  const evidence = useMemo(() => (stage ? recordsForStage(stage, runtimeState) : []), [runtimeState, stage]);
  const runtimeSummary = useMemo(() => (stage ? runtimeSummaryForStage(stage, runtimeState) : null), [runtimeState, stage]);

  if (!stage) {
    return (
      <DashboardShell
        header={<DashboardHeader title="Operating System Stage Missing" eyebrow={version} description="No operating-system stage contract was found." />}
      >
        <DashboardSection title="Missing Stage" description="Add the stage to operatingSystemStages before this route can render.">
          <p className="text-sm text-muted-foreground">No stage data is available for {version}.</p>
        </DashboardSection>
      </DashboardShell>
    );
  }

  return (
    <DashboardShell
      sidebar={
        <DashboardSidebar
          title={stage.version}
          description={stage.primaryMetric}
          items={[{ id: "stage", label: stage.sectionTitle, href: "#stage", active: true, icon: Activity }]}
        />
      }
      header={
        <DashboardHeader
          title={stage.title}
          eyebrow={stage.eyebrow}
          description={stage.description}
          actions={
            <Button onClick={() => refreshRuntimeState(setRuntimeState, setRuntimeMode, setRuntimeError)}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          }
          meta={
            <>
              <StatusPill tone={runtimeMode === "server" ? "success" : "warning"}>{runtimeMode} runtime</StatusPill>
              <StatusPill tone={stage.status === "ready" ? "success" : stage.status === "gated" ? "warning" : "info"}>{stage.status}</StatusPill>
              <StatusPill tone={stage.risk === "high" ? "critical" : stage.risk === "medium" ? "warning" : "info"}>{stage.risk} risk</StatusPill>
            </>
          }
        />
      }
    >
      <MetricGrid columns={4}>
        <KpiCard label="Version" value={stage.version} detail={stage.owner} tone="info" />
        <KpiCard label="Primary Metric" value={stage.primaryMetric} detail="operating focus" tone="success" />
        <KpiCard label="Runtime Status" value={stage.status} detail="infrastructure readiness" tone={stage.status === "gated" ? "warning" : "success"} />
        <ProgressMetric label="Trackable Build" value={stage.progress} tone={stage.progress >= 100 ? "success" : "warning"} />
      </MetricGrid>
      {runtimeSummary ? (
        <MetricGrid columns={4}>
          <KpiCard label="Runtime Evidence" value={runtimeSummary.total} detail="records tied to this stage" tone="info" />
          <KpiCard label="Ready Evidence" value={runtimeSummary.ready} detail="ready, stored, or allowed" tone="success" />
          <KpiCard label="Gated Evidence" value={runtimeSummary.gated} detail="blocked, gated, warning, or failed" tone={runtimeSummary.gated ? "warning" : "success"} />
          <KpiCard label="Audit Events" value={runtimeSummary.audited} detail="readiness checks for this stage" tone="info" />
        </MetricGrid>
      ) : null}
      <MetricGrid columns={3}>
        {stage.cards.map((card) => (
          <KpiCard key={card.label} label={card.label} value={card.value} detail={card.detail} tone={card.tone} />
        ))}
      </MetricGrid>
      <DashboardSection id="stage" title={stage.sectionTitle} description={stage.sectionDescription}>
        <DataTable columns={columns} rows={stage.rows} getRowKey={(row) => row.id} />
      </DashboardSection>
      <DashboardSection
        id="runtime-evidence"
        title="Runtime Evidence"
        description={
          runtimeMode === "server"
            ? "Server-backed SQLite evidence, audit, and readiness checks are active for this stage."
            : `Local fallback is active while the server runtime is unavailable${runtimeError ? `: ${runtimeError}` : "."}`
        }
        action={
          <Button
            disabled={checking}
            onClick={async () => {
              setChecking(true);
              try {
                const next = await runServerRuntimeReadinessCheck(stage, runtimeState);
                setRuntimeState(next);
                setRuntimeMode("server");
                setRuntimeError("");
              } catch (error) {
                setRuntimeState((current) => runRuntimeReadinessCheck(stage, current));
                setRuntimeMode("local");
                setRuntimeError(error instanceof Error ? error.message : "Server readiness check unavailable.");
              } finally {
                setChecking(false);
              }
            }}
          >
            <PlayCircle className="h-4 w-4" />
            {checking ? "Checking..." : "Run readiness check"}
          </Button>
        }
      >
        <DataTable columns={evidenceColumns} rows={evidence} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}

async function refreshRuntimeState(
  setRuntimeState: (state: ReturnType<typeof loadOperatingRuntimeState>) => void,
  setRuntimeMode: (mode: "server" | "local") => void,
  setRuntimeError: (message: string) => void,
) {
  try {
    const next = await loadOperatingRuntimeStateFromServer();
    setRuntimeState(next);
    setRuntimeMode("server");
    setRuntimeError("");
  } catch (error) {
    setRuntimeState(loadOperatingRuntimeState());
    setRuntimeMode("local");
    setRuntimeError(error instanceof Error ? error.message : "Server runtime unavailable.");
  }
}

function toneForState(state: StageRow["state"]) {
  if (state === "built") return "success";
  if (state === "gated") return "warning";
  if (state === "blocked") return "critical";
  return "info";
}

function toneForRuntimeState(state: RuntimeEvidenceRecord["state"]) {
  if (["ready", "stored", "allowed"].includes(state)) return "success";
  if (["blocked", "failed"].includes(state)) return "critical";
  if (["gated", "warning"].includes(state)) return "warning";
  return "info";
}
