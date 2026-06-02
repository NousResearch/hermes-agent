import { ClipboardList, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";

import { Badge } from "@nous-research/ui/ui/components/badge";
import { H2 } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api, type ApprovalSliceSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

const statusOrder = ["active", "revoked", "expired", "completed"] as const;
const panel = "rounded-xl border border-[#284848] bg-black/30 p-4";
const labelClass = "text-xs font-semibold uppercase tracking-[0.08em] text-text-secondary";

function listText(values?: string[] | null): string {
  if (!values?.length) return "None declared";
  return values.join(", ");
}

function textValue(value?: string | null): string {
  return value && value.trim() ? value : "-";
}

function formatTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function SliceField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className={labelClass}>{label}</div>
      <div className="mt-1 break-words rounded-lg border border-[#284848] bg-black/35 px-3 py-2 text-sm text-text-secondary">
        {value}
      </div>
    </div>
  );
}

function StatusSummary({ items }: { items: ApprovalSliceSummary[] }) {
  return (
    <div className="grid gap-2 sm:grid-cols-4">
      {statusOrder.map((status) => (
        <div key={status} className="rounded-xl border border-white/10 bg-black/25 px-3 py-2">
          <div className={labelClass}>{status}</div>
          <div className="mt-1 text-xl font-semibold text-text-primary">
            {items.filter((item) => item.status === status).length}
          </div>
        </div>
      ))}
    </div>
  );
}

function SliceCard({ item }: { item: ApprovalSliceSummary }) {
  return (
    <article className={panel}>
      <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="break-words text-base font-semibold text-text-primary">{item.title || item.id}</div>
          <div className="mt-1 flex flex-wrap gap-2 text-xs text-text-secondary">
            <span>{item.id}</span>
            <span>{item.status}</span>
            <span>{formatTime(item.updated_at || item.created_at)}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone="outline" className="border-amber-400/35 bg-amber-500/10 text-amber-100">
            Decision record only — does not authorize execution
          </Badge>
          <Badge tone="outline" className="border-red-400/30 text-red-100">
            trusted_for_execution={String(item.trusted_for_execution)}
          </Badge>
          <Badge tone="outline" className="border-emerald-400/30 text-emerald-100">
            inert_context_only={String(item.inert_context_only)}
          </Badge>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        <SliceField label="allowed_actions" value={listText(item.allowed_actions)} />
        <SliceField label="forbidden_actions" value={listText(item.forbidden_actions)} />
        <SliceField label="stop_condition" value={textValue(item.stop_condition)} />
        <SliceField label="checkpoint" value={textValue(item.checkpoint)} />
        <SliceField label="Goal Contract ID" value={textValue(item.linked_goal_contract_id)} />
        <SliceField label="Repo path" value={textValue(item.repo_path)} />
      </div>
    </article>
  );
}

function InertState({ title, copy }: { title: string; copy: string }) {
  return (
    <div className={cn(panel, "text-sm text-text-secondary")}>
      <div className="font-semibold text-text-primary">{title}</div>
      <div className="mt-1">{copy}</div>
    </div>
  );
}

export function ApprovalSlicesPanel() {
  const [items, setItems] = useState<ApprovalSliceSummary[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    api.listApprovalSlices()
      .then((response) => {
        if (cancelled) return;
        setItems(response.items);
        setWarnings(response.warnings);
        setError(null);
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Approval Slice records could not be loaded.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Card className="font-readable-ui border-[#264545] bg-[#071717]/90 shadow-[0_0_0_1px_rgba(47,214,161,0.04),0_18px_60px_rgba(0,0,0,0.28)]">
      <CardContent className="space-y-5 p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <ClipboardList className="h-5 w-5" />
              <H2 className="text-xl">Approval Slices</H2>
            </div>
            <div className="mt-1 text-sm leading-6 text-text-secondary">
              G3 decision records shown as read-only visibility for Mission Control.
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="outline" className="border-amber-400/35 bg-amber-500/10 text-amber-100">
              G3 decision records
            </Badge>
            <Badge tone="outline" className="border-red-400/35 bg-red-500/10 text-red-100">
              <ShieldCheck className="mr-1 h-3.5 w-3.5" />
              trusted_for_execution=false
            </Badge>
            <Badge tone="outline" className="border-emerald-400/35 bg-emerald-500/10 text-emerald-100">
              inert_context_only=true
            </Badge>
          </div>
        </div>

        {loading && <InertState title="Loading approval slices." copy="Existing records are being read without enabling actions." />}

        {!loading && error && (
          <InertState title="Approval Slice records could not be loaded." copy={error} />
        )}

        {!loading && !error && warnings.length > 0 && (
          <div className="space-y-2 rounded-xl border border-amber-400/25 bg-amber-500/10 p-3 text-sm text-amber-100">
            {warnings.map((warning) => (
              <div key={warning}>{warning}</div>
            ))}
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <InertState title="No Approval Slice records found." copy="Mission Control has no existing Approval Slice records to display." />
        )}

        {!loading && !error && items.length > 0 && (
          <>
            <StatusSummary items={items} />
            <div className="grid gap-3">
              {items.map((item) => (
                <SliceCard key={item.id} item={item} />
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
