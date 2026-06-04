import { CheckCircle2, ClipboardList, Gauge, LockKeyhole, ShieldAlert } from "lucide-react";
import { useEffect, useState } from "react";

import { Badge } from "@nous-research/ui/ui/components/badge";
import { H2 } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api, type MissionControlLaneDashboardResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const panel = "rounded-xl border border-[#284848] bg-black/30 p-4";
const labelClass = "text-xs font-semibold uppercase tracking-[0.08em] text-text-secondary";

function listText(values?: string[] | null): string {
  if (!values?.length) return "None declared";
  return values.join(", ");
}

function textValue(value?: string | number | boolean | null): string {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-black/25 px-3 py-2">
      <div className={labelClass}>{label}</div>
      <div className="mt-1 break-words text-sm font-semibold text-text-primary">{value}</div>
    </div>
  );
}

function Section({
  title,
  children,
  tone,
}: {
  title: string;
  children: React.ReactNode;
  tone?: string;
}) {
  return (
    <section className={cn(panel, tone)}>
      <div className="mb-3 text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">{title}</div>
      {children}
    </section>
  );
}

function InertState({ title, copy }: { title: string; copy: string }) {
  return (
    <Card className="font-readable-ui border-[#264545] bg-[#071717]/90">
      <CardContent className="p-5">
        <div className="text-base font-semibold text-text-primary">{title}</div>
        <div className="mt-1 text-sm leading-6 text-text-secondary">{copy}</div>
      </CardContent>
    </Card>
  );
}

function DashboardContent({ data }: { data: MissionControlLaneDashboardResponse }) {
  return (
    <Card className="font-readable-ui border-[#264545] bg-[#071717]/90 shadow-[0_0_0_1px_rgba(47,214,161,0.04),0_18px_60px_rgba(0,0,0,0.28)]">
      <CardContent className="space-y-5 p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <ClipboardList className="h-5 w-5" />
              <H2 className="text-xl">Mission Control Lane Dashboard</H2>
            </div>
            <div className="mt-1 text-sm leading-6 text-text-secondary">
              Read-only API/UI only. Compact lane status without runner integration, tool interception, transcript loading, execution controls, parent scans, or quarantined path access.
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="outline" className="border-amber-400/35 bg-amber-500/10 text-amber-100">
              {data.mode}
            </Badge>
            <Badge tone="outline" className="border-red-400/35 bg-red-500/10 text-red-100">
              trusted_for_execution=false
            </Badge>
            <Badge tone="outline" className="border-emerald-400/35 bg-emerald-500/10 text-emerald-100">
              inert_context_only=true
            </Badge>
          </div>
        </div>

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <Metric label="Active lane" value={data.active_lane.label} />
          <Metric label="Approval tier" value={data.approval_tier.label} />
          <Metric label="Start Gate" value={`${data.start_gate.status} · ${data.start_gate.repo_state}`} />
          <Metric label="Evidence summaries" value={`${data.evidence.count} recorded`} />
        </section>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <Section title="Task Control Envelope">
            <div className="grid gap-2 sm:grid-cols-2">
              <Metric label="Envelope" value={data.task_control_envelope.exists ? textValue(data.task_control_envelope.summary) : "No active envelope"} />
              <Metric label="Mode" value={textValue(data.active_lane.mode)} />
              <Metric label="Checkpoint" value={textValue(data.task_control_envelope.checkpoint)} />
              <Metric label="Selected records" value={textValue(data.task_control_envelope.selected_from_count)} />
            </div>
          </Section>

          <Section title="Quarantine / parent-scan warning" tone="border-amber-400/25 bg-amber-500/10">
            <div className="flex gap-3 text-sm leading-6 text-amber-100">
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <div>{data.safety.quarantine_parent_scan_warning}</div>
            </div>
          </Section>
        </div>

        <div className="grid gap-3 lg:grid-cols-2">
          <Section title="Allowed actions">
            <div className="break-words text-sm leading-6 text-text-secondary">{listText(data.allowed_actions)}</div>
          </Section>
          <Section title="Forbidden actions">
            <div className="break-words text-sm leading-6 text-text-secondary">{listText(data.forbidden_actions)}</div>
          </Section>
        </div>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Section title="Evidence summaries">
            <div className="space-y-2">
              {data.evidence.summaries.length === 0 && (
                <div className="rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-sm text-text-secondary">
                  No evidence summaries attached.
                </div>
              )}
              {data.evidence.summaries.map((item) => (
                <details key={item.id || item.title} className="rounded-lg border border-white/10 bg-black/25 px-3 py-2 text-sm text-text-secondary">
                  <summary className="cursor-default list-none font-semibold text-text-primary">
                    {textValue(item.title)} <span className="font-normal text-text-secondary">· {textValue(item.kind)}</span>
                  </summary>
                  <div className="mt-2 leading-6">
                    <div>{textValue(item.summary)}</div>
                    <div className="mt-1 text-xs text-text-tertiary">Evidence details are collapsed until requested.</div>
                  </div>
                </details>
              ))}
              {data.evidence.details_on_demand && (
                <div className="flex items-center gap-2 text-xs text-text-tertiary">
                  <LockKeyhole className="h-3.5 w-3.5" />
                  Evidence details only on demand.
                </div>
              )}
            </div>
          </Section>

          <Section title="Token/context budget">
            <div className="grid gap-2 sm:grid-cols-2">
              <Metric label="Input estimate" value={`${data.token_context_budget.estimated_input_tokens.toLocaleString()} tokens`} />
              <Metric label="Output estimate" value={`${data.token_context_budget.estimated_output_tokens.toLocaleString()} tokens`} />
              <Metric label="Context" value={data.token_context_budget.remaining_context_window} />
              <Metric label="Token estimates" value={data.token_context_budget.show_token_estimates ? "Shown" : "Hidden"} />
            </div>
            <div className="mt-3 flex items-start gap-2 rounded-lg border border-emerald-400/20 bg-emerald-500/10 px-3 py-2 text-sm leading-6 text-emerald-100">
              <Gauge className="mt-1 h-4 w-4 shrink-0" />
              <div>{data.token_context_budget.conservation_behavior}</div>
            </div>
          </Section>
        </div>

        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.8fr)]">
          <Section title="Next recommended action">
            <div className="flex gap-3 text-sm leading-6 text-text-secondary">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-200" />
              <div>{data.next_recommended_action}</div>
            </div>
          </Section>
          <Section title="Safety posture">
            <div className="grid gap-2 text-sm text-text-secondary">
              <Metric label="Transcript loaded" value={String(data.safety.transcript_loaded)} />
              <Metric label="Execution controls" value={String(data.safety.execution_controls)} />
              <Metric label="Runner integration" value={String(data.safety.runner_integration)} />
              <Metric label="Tool interception" value={String(data.safety.tool_interception)} />
            </div>
          </Section>
        </div>
      </CardContent>
    </Card>
  );
}

export function MissionControlLaneDashboard() {
  const [data, setData] = useState<MissionControlLaneDashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    api.getMissionControlLaneDashboard()
      .then((response) => {
        if (cancelled) return;
        setData(response);
        setError(null);
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Lane dashboard could not be loaded.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <InertState title="Loading lane dashboard." copy="Mission Control lane state is being read without enabling actions." />;
  }

  if (error) {
    return (
      <InertState
        title="Lane dashboard could not be loaded."
        copy={`Read-only surface remains inert. ${error}`}
      />
    );
  }

  if (!data) {
    return (
      <InertState
        title="No lane dashboard state."
        copy="No Mission Control lane dashboard payload was returned."
      />
    );
  }

  if (data.safety.parent_scan_performed || data.safety.quarantined_path_accessed) {
    return (
      <InertState
        title="Lane dashboard safety warning."
        copy="Read model reported forbidden parent-scan or quarantined-path access."
      />
    );
  }

  return <DashboardContent data={data} />;
}
