import { FileWarning, LockKeyhole, Shield } from "lucide-react";
import { useEffect, useState } from "react";

import { api, type MissionControlActiveEnvelopeResponse } from "@/lib/api";

const envelopeStatus = [
  "Display only",
  "No active authorization",
  "Operational actions locked",
  "Discussion/status only",
  "trusted_for_execution=false",
  "inert_context_only=true",
  "non-authorizing",
];

const emptyEnvelopeBoundaryFields = [
  "Active lane: Unset",
  "Active mode: Unset",
  "Execution boundary: No active authorization",
  "Allowed actions: None declared",
  "Forbidden actions: Unknown",
  "Checkpoint: None",
];

const emptyEnvelopeEvidenceFields = [
  "Repo state: Unknown / not probed",
  "Evidence: No envelope evidence attached",
  "Data source: No persisted envelope",
  "exists=false",
];

function valueOrUnset(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Unset";
  if (typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None declared";
  return String(value);
}

function DetailRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-text-secondary">
      <span className="text-text-primary">{label}:</span> {valueOrUnset(value)}
    </div>
  );
}

function PanelShell({
  badge,
  title,
  copy,
  children,
}: {
  badge: string;
  title: string;
  copy: string;
  children: React.ReactNode;
}) {
  return (
    <section className="font-readable-ui rounded-2xl border border-amber-400/25 bg-[linear-gradient(135deg,rgba(34,23,8,0.78),rgba(7,15,15,0.94))] p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.04)] lg:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-xs uppercase tracking-wide text-amber-100">
            <LockKeyhole className="h-3.5 w-3.5" />
            {badge}
          </div>
          <h2 className="text-xl font-semibold text-text-primary">{title}</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-text-secondary">{copy}</p>
        </div>

        <div className="grid gap-2 text-sm sm:grid-cols-2 lg:min-w-96">
          {envelopeStatus.map((item) => (
            <div key={item} className="rounded-xl border border-[#3d3a24] bg-black/25 px-3 py-2 text-amber-100">
              {item}
            </div>
          ))}
        </div>
      </div>

      {children}
    </section>
  );
}

function EmptyState() {
  return (
    <PanelShell
      badge="No persisted envelope"
      title="No Active Task Envelope"
      copy="No persisted envelope is attached, so there is no active authorization for this Mission Control surface."
    >
      <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="rounded-xl border border-[#284848] bg-black/25 p-3">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-midground">
            <Shield className="h-4 w-4" />
            Authorization boundary
          </div>
          <div className="grid gap-2 text-sm">
            {emptyEnvelopeBoundaryFields.map((line) => (
              <div key={line} className="rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-text-secondary">
                {line}
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-[#284848] bg-black/25 p-3">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-midground">
            <FileWarning className="h-4 w-4" />
            Evidence posture
          </div>
          <div className="grid gap-2 text-sm">
            {emptyEnvelopeEvidenceFields.map((line) => (
              <div key={line} className="rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-text-secondary">
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    </PanelShell>
  );
}

function ActiveState({ data }: { data: MissionControlActiveEnvelopeResponse }) {
  const envelope = data.task_control_envelope;
  const selection = data.selection;

  return (
    <PanelShell
      badge="Persisted envelope"
      title="Active Task Envelope"
      copy="A persisted task envelope is shown as inert context only. It does not authorize actions from this surface."
    >
      <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="rounded-xl border border-[#284848] bg-black/25 p-3">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-midground">
            <Shield className="h-4 w-4" />
            Envelope metadata
          </div>
          <div className="grid gap-2 text-sm">
            <DetailRow label="exists=true" value={data.exists} />
            <DetailRow label="Active lane" value={data.active_lane} />
            <DetailRow label="Active mode" value={data.active_mode} />
            <DetailRow label="Execution boundary" value={data.execution_boundary} />
            <DetailRow label="Allowed actions" value={data.allowed_actions} />
            <DetailRow label="Forbidden actions" value={data.forbidden_actions} />
            <DetailRow label="Checkpoint" value={data.checkpoint} />
            <DetailRow label="Data source" value={data.data_source} />
          </div>
        </div>

        <div className="rounded-xl border border-[#284848] bg-black/25 p-3">
          <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-midground">
            <FileWarning className="h-4 w-4" />
            Selection posture
          </div>
          <div className="grid gap-2 text-sm">
            <DetailRow label="Envelope ID" value={envelope?.id} />
            <DetailRow label="Schema" value={envelope?.schema} />
            <DetailRow label="Status" value={envelope?.status} />
            <DetailRow label="Title" value={envelope?.title} />
            <DetailRow label="Mode label" value={envelope?.mode_label} />
            <DetailRow label="Created at" value={envelope?.created_at} />
            <DetailRow label="Updated at" value={envelope?.updated_at} />
            <DetailRow label="selection_reason" value={selection?.selection_reason} />
            <DetailRow label="selected_from_count" value={selection?.selected_from_count} />
            <DetailRow label="ambiguous=true" value={selection?.ambiguous === true ? "true" : "false"} />
          </div>
          {selection?.ambiguous === true && (
            <div className="mt-3 rounded-xl border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
              Multiple persisted envelopes matched; newest active envelope is being shown as inert context.
            </div>
          )}
        </div>
      </div>
    </PanelShell>
  );
}

function InertStatusPanel({ title, copy }: { title: string; copy: string }) {
  return (
    <PanelShell badge="Read-only state" title={title} copy={copy}>
      <div className="mt-4 rounded-xl border border-[#284848] bg-black/25 p-3 text-sm text-text-secondary">
        No actions are available from this panel.
      </div>
    </PanelShell>
  );
}

export function ActiveEnvelopePanel() {
  const [data, setData] = useState<MissionControlActiveEnvelopeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    api.getMissionControlActiveEnvelope()
      .then((response) => {
        if (!mounted) return;
        setData(response);
        setError(null);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Task envelope could not be loaded.");
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  if (loading) {
    return <InertStatusPanel title="Loading task envelope." copy="Envelope context is being read without enabling actions." />;
  }

  if (error) {
    return <InertStatusPanel title="Task envelope could not be loaded." copy={error} />;
  }

  if (!data || data.exists === false) {
    return <EmptyState />;
  }

  return <ActiveState data={data} />;
}
