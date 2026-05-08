import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Database, MessageSquareText, RefreshCw, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  api,
  type MorningBriefCanaryResponse,
  type MorningBriefV0CanaryResponse,
  type MorningBriefTelegramPreviewResponse,
  type MorningBriefSafetyPreviewResponse,
} from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";

function JsonList({ value }: { value: unknown[] | undefined }) {
  if (!value || value.length === 0) {
    return <p className="text-sm text-muted-foreground">No rows.</p>;
  }
  return (
    <div className="space-y-2">
      {value.map((item, index) => (
        <pre
          key={index}
          className="overflow-auto rounded-md border border-current/15 bg-black/20 p-3 font-mono-ui text-[11px] leading-4 text-muted-foreground normal-case"
        >
          {JSON.stringify(item, null, 2)}
        </pre>
      ))}
    </div>
  );
}

function BoundaryBadge({ label, value }: { label: string; value?: boolean }) {
  return (
    <Badge tone={value ? "destructive" : "success"} className="text-[10px]">
      {label}: {value ? "enabled" : "off"}
    </Badge>
  );
}

export default function ControlPlanePage() {
  const [data, setData] = useState<MorningBriefCanaryResponse | null>(null);
  const [v0Canary, setV0Canary] = useState<MorningBriefV0CanaryResponse | null>(null);
  const [telegramPreview, setTelegramPreview] =
    useState<MorningBriefTelegramPreviewResponse | null>(null);
  const [safetyPreview, setSafetyPreview] =
    useState<MorningBriefSafetyPreviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd } = usePageHeader();

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    setTelegramPreview(null);
    setSafetyPreview(null);
    setV0Canary(null);

    try {
      const canary = await api.getMorningBriefCanary();
      setData(canary);
    } catch {
      setData(null);
      setError("Control-plane read failed; check server logs or Supabase CLI link state.");
      setLoading(false);
      return;
    }

    try {
      const v0 = await api.getMorningBriefV0Canary();
      setV0Canary(v0);
    } catch {
      setV0Canary({
        enabled: false,
        reason: "morning_brief_v0_canary_unavailable",
        safe_next_action: "verify Gate B artifact status or keep hold/observe",
      });
    }

    try {
      const preview = await api.getMorningBriefTelegramPreview();
      setTelegramPreview(preview);
    } catch {
      setTelegramPreview({
        enabled: false,
        reason: "telegram_preview_unavailable",
        message: "Telegram preview unavailable; control-plane readback can still be shown.",
      });
    }

    try {
      const safety = await api.getMorningBriefSafetyPreview();
      setSafetyPreview(safety);
    } catch {
      setSafetyPreview({
        enabled: false,
        reason: "safety_preview_unavailable",
        message: "Safety preview unavailable; keep publish/send blocked until verified.",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useLayoutEffect(() => {
    setAfterTitle(
      <span className="flex items-center gap-2">
        {loading && <Spinner className="shrink-0 text-base text-primary" />}
        <Badge tone={loading && !data ? "secondary" : data?.enabled ? "success" : "secondary"} className="text-[10px]">
          {loading && !data ? "loading canary" : data?.enabled ? "read-only canary" : "not configured"}
        </Badge>
      </span>,
    );
    setEnd(
      <Button
        type="button"
        size="sm"
        outlined
        onClick={refresh}
        disabled={loading}
        prefix={loading ? <Spinner /> : <RefreshCw />}
      >
        Refresh
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [data?.enabled, loading, refresh, setAfterTitle, setEnd]);

  const run = data?.run;
  const boundaries = data?.boundaries ?? {};

  return (
    <div className="flex flex-col gap-4 normal-case">
      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="flex items-center gap-2 text-sm uppercase">
            <Database className="h-4 w-4" />
            HERA-198 Control Plane Canary
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 p-4">
          {error && <p className="text-sm text-destructive">{error}</p>}
          {data && !data.enabled && (
            <div className="rounded-md border border-warning/30 bg-warning/10 p-3">
              <p className="text-sm font-semibold uppercase text-warning">Disabled</p>
              <p className="text-sm text-muted-foreground">{data.reason}: {data.message}</p>
            </div>
          )}
          <div className="flex flex-wrap gap-2 uppercase">
            <Badge tone="secondary" className="text-[10px]">project: {loading && !data ? "loading" : data?.project ?? "unknown"}</Badge>
            <Badge tone="secondary" className="text-[10px]">mode: {data?.mode ?? "read-only"}</Badge>
            <Badge tone="secondary" className="text-[10px]">updated: {data?.updated_at ?? "-"}</Badge>
          </div>
          <div className="flex flex-wrap gap-2 uppercase">
            <BoundaryBadge label="writes" value={boundaries.writes_enabled} />
            <BoundaryBadge label="telegram" value={boundaries.telegram_send_enabled} />
            <BoundaryBadge label="obsidian authority" value={boundaries.obsidian_authority_edit_enabled} />
            <BoundaryBadge label="cron" value={boundaries.cron_change_enabled} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="flex items-center gap-2 text-sm uppercase">
            <ShieldCheck className="h-4 w-4" />
            Morning Brief v0.2 Gate B/C Verification
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 p-4 text-sm">
          <div className="flex flex-wrap gap-2 uppercase">
            <Badge tone={v0Canary?.enabled ? "success" : "secondary"} className="text-[10px]">
              {v0Canary?.enabled ? "local readback pass" : "safe disabled"}
            </Badge>
            <Badge tone="secondary" className="text-[10px]">
              Hermes DB readback: {v0Canary?.hermes_direct_db_verification ? "yes" : "no"}
            </Badge>
            <Badge tone={v0Canary?.boundaries?.webui_mutation_enabled ? "destructive" : "success"} className="text-[10px]">
              webui mutation: {v0Canary?.boundaries?.webui_mutation_enabled ? "enabled" : "off"}
            </Badge>
          </div>
          {v0Canary?.enabled ? (
            <dl className="grid gap-2 md:grid-cols-2">
              <div><dt className="text-xs uppercase text-muted-foreground">Canary key</dt><dd className="break-all font-mono-ui text-xs">{v0Canary.canary_key}</dd></div>
              <div><dt className="text-xs uppercase text-muted-foreground">Result</dt><dd>{v0Canary.gateb_verification_result}</dd></div>
              <div><dt className="text-xs uppercase text-muted-foreground">Verification</dt><dd>{v0Canary.verification_status}</dd></div>
              <div><dt className="text-xs uppercase text-muted-foreground">Rollback</dt><dd>{v0Canary.rollback_status}</dd></div>
              <div className="md:col-span-2"><dt className="text-xs uppercase text-muted-foreground">Evidence source</dt><dd>{v0Canary.verification_source}</dd></div>
              <div className="md:col-span-2"><dt className="text-xs uppercase text-muted-foreground">Snapshot ref</dt><dd className="break-all font-mono-ui text-xs">{v0Canary.report_snapshot_ref}</dd></div>
            </dl>
          ) : (
            <div className="rounded-md border border-warning/30 bg-warning/10 p-3">
              <p className="text-sm font-semibold uppercase text-warning">Safe disabled</p>
              <p className="text-sm text-muted-foreground">{v0Canary?.reason ?? "morning_brief_v0_canary_not_configured"}</p>
            </div>
          )}
          <p className="text-xs text-muted-foreground normal-case">
            Read-only only: no browser-side Supabase secret, no DB mutation, no Telegram send, no cron/routine integration.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm uppercase">Latest Morning Brief Run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 p-4 text-sm">
            {run ? (
              <>
                <p className="font-semibold text-foreground">{run.title}</p>
                <p className="text-muted-foreground">{run.summary_kr}</p>
                <dl className="grid gap-2 md:grid-cols-2">
                  <div><dt className="text-xs uppercase text-muted-foreground">Status</dt><dd>{run.status}</dd></div>
                  <div><dt className="text-xs uppercase text-muted-foreground">Date</dt><dd>{run.report_date}</dd></div>
                  <div className="md:col-span-2"><dt className="text-xs uppercase text-muted-foreground">Run ref</dt><dd className="break-all font-mono-ui text-xs">{run.run_ref}</dd></div>
                  <div className="md:col-span-2"><dt className="text-xs uppercase text-muted-foreground">Obsidian ref</dt><dd className="break-all font-mono-ui text-xs">{run.obsidian_ref}</dd></div>
                  <div className="md:col-span-2"><dt className="text-xs uppercase text-muted-foreground">Paperclip ref</dt><dd className="break-all font-mono-ui text-xs">{run.paperclip_parent_ref}</dd></div>
                </dl>
              </>
            ) : loading ? (
              <p className="text-muted-foreground">Loading Morning Brief run…</p>
            ) : (
              <p className="text-muted-foreground">No Morning Brief run found.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="flex items-center gap-2 text-sm uppercase">
              <ShieldCheck className="h-4 w-4" />
              Counts / Verification
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4">
            <div className="grid gap-2 md:grid-cols-2">
              {Object.entries(data?.counts ?? {}).map(([key, value]) => (
                <div key={key} className="rounded-md border border-current/15 p-3">
                  <p className="text-xs uppercase text-muted-foreground">{key}</p>
                  <p className="text-lg font-semibold text-foreground">{value}</p>
                </div>
              ))}
              {loading && !data && (
                <p className="text-sm text-muted-foreground">Loading verification counts…</p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="flex items-center gap-2 text-sm uppercase">
            <ShieldCheck className="h-4 w-4" />
            Source Safety Preview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 p-4 text-sm">
          {safetyPreview && !safetyPreview.enabled && (
            <div className="rounded-md border border-warning/30 bg-warning/10 p-3">
              <p className="text-sm font-semibold uppercase text-warning">Safety preview disabled</p>
              <p className="text-sm text-muted-foreground">
                {safetyPreview.reason}: {safetyPreview.message}
              </p>
            </div>
          )}
          <div className="flex flex-wrap gap-2 uppercase">
            <Badge tone={safetyPreview?.source_safety?.state === "has_quarantine" ? "warning" : "success"} className="text-[10px]">
              source safety: {safetyPreview?.source_safety?.state ?? "loading"}
            </Badge>
            <Badge tone={safetyPreview?.rollback_readiness?.publish_blocked ? "warning" : "success"} className="text-[10px]">
              publish_blocked: {safetyPreview?.rollback_readiness?.publish_blocked ? "yes" : "no"}
            </Badge>
            <Badge tone={safetyPreview?.boundaries?.webui_mutation_enabled ? "destructive" : "success"} className="text-[10px]">
              webui mutation: {safetyPreview?.boundaries?.webui_mutation_enabled ? "enabled" : "off"}
            </Badge>
          </div>
          <dl className="grid gap-2 md:grid-cols-2">
            <div>
              <dt className="text-xs uppercase text-muted-foreground">quarantined_source_count</dt>
              <dd className="text-lg font-semibold text-foreground">
                {safetyPreview?.source_safety?.quarantined_source_count ?? 0}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">notification action</dt>
              <dd>{safetyPreview?.notification_readiness?.action_state ?? "preview_only"}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">authority state</dt>
              <dd>{safetyPreview?.authority_boundary?.state ?? "no_obsidian_ref"}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">rollback available</dt>
              <dd>{safetyPreview?.rollback_readiness?.available ? "yes" : "no"}</dd>
            </div>
            <div className="md:col-span-2">
              <dt className="text-xs uppercase text-muted-foreground">publish block reason</dt>
              <dd className="normal-case text-muted-foreground">
                {safetyPreview?.rollback_readiness?.publish_block_reason_kr ?? "No safety preview loaded yet."}
              </dd>
            </div>
          </dl>
          <p className="text-xs text-muted-foreground normal-case">
            Read-only safety preview: no send controls, no browser-side Supabase access, no cron change, no Obsidian authority edit.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="py-3 px-4">
          <CardTitle className="flex items-center gap-2 text-sm uppercase">
            <MessageSquareText className="h-4 w-4" />
            Telegram Dry-run Payload Preview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 p-4">
          {telegramPreview && !telegramPreview.enabled && (
            <div className="rounded-md border border-warning/30 bg-warning/10 p-3">
              <p className="text-sm font-semibold uppercase text-warning">Preview disabled</p>
              <p className="text-sm text-muted-foreground">
                {telegramPreview.reason}: {telegramPreview.message}
              </p>
            </div>
          )}
          <div className="flex flex-wrap gap-2 uppercase">
            <Badge tone="secondary" className="text-[10px]">
              channel: {telegramPreview?.channel ?? "telegram"}
            </Badge>
            <Badge tone="secondary" className="text-[10px]">
              mode: {telegramPreview?.mode ?? "dry-run preview"}
            </Badge>
            <Badge tone={telegramPreview?.send_enabled ? "destructive" : "success"} className="text-[10px]">
              send: {telegramPreview?.send_enabled ? "enabled" : "off"}
            </Badge>
            <Badge tone={telegramPreview?.write_enabled ? "destructive" : "success"} className="text-[10px]">
              supabase write: {telegramPreview?.write_enabled ? "enabled" : "off"}
            </Badge>
            <Badge tone={telegramPreview?.validation?.length_ok ? "success" : "secondary"} className="text-[10px]">
              length: {telegramPreview?.message_length ?? 0}/{telegramPreview?.validation?.max_length ?? 600}
            </Badge>
          </div>
          <div>
            <p className="text-xs uppercase text-muted-foreground">Target ref</p>
            <p className="break-all font-mono-ui text-xs text-foreground">
              {telegramPreview?.target_ref ?? "telegram://dry-run/hera-198"}
            </p>
          </div>
          <pre className="whitespace-pre-wrap rounded-md border border-current/15 bg-black/20 p-3 text-sm leading-6 text-foreground normal-case">
            {telegramPreview?.message_text ?? "No preview generated."}
          </pre>
          <p className="text-xs text-muted-foreground normal-case">
            Dry-run only: no Telegram send, no Supabase write, no cron change, no Obsidian authority edit.
          </p>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader className="py-3 px-4"><CardTitle className="text-sm uppercase">Sections</CardTitle></CardHeader>
          <CardContent className="p-4"><JsonList value={data?.sections} /></CardContent>
        </Card>
        <Card>
          <CardHeader className="py-3 px-4"><CardTitle className="text-sm uppercase">Source Anchors</CardTitle></CardHeader>
          <CardContent className="p-4"><JsonList value={data?.sources} /></CardContent>
        </Card>
        <Card>
          <CardHeader className="py-3 px-4"><CardTitle className="text-sm uppercase">Delivery Events</CardTitle></CardHeader>
          <CardContent className="p-4"><JsonList value={data?.delivery_events} /></CardContent>
        </Card>
        <Card>
          <CardHeader className="py-3 px-4"><CardTitle className="text-sm uppercase">Audit Events</CardTitle></CardHeader>
          <CardContent className="p-4"><JsonList value={data?.audit_events} /></CardContent>
        </Card>
      </div>
    </div>
  );
}
