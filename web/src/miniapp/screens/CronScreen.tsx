import { useEffect, useState } from "react";
import type { CronJob } from "@/lib/api";
import { del, get, post } from "../api";
import { useMiniApp } from "../context";

function relativeToNow(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const target = new Date(iso).getTime();
  if (Number.isNaN(target)) return null;
  const seconds = (target - Date.now()) / 1000;
  if (seconds < 0) return null;
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m`;
  if (seconds < 86400) return `${(Math.round(seconds / 360) / 10).toFixed(1)}h`;
  return `${(Math.round(seconds / 8640) / 10).toFixed(1)}d`;
}

export function CronScreen({ onShowLog }: { onShowLog: (title: string, text: string) => void }) {
  const { isAdmin, showToast, askConfirm } = useMiniApp();
  const [jobs, setJobs] = useState<CronJob[] | null>(null);
  const [ranJobs, setRanJobs] = useState<Record<string, boolean>>({});
  const [pending, setPending] = useState<Record<string, boolean>>({});

  const load = () => get<CronJob[]>("/api/cron/jobs").then(setJobs).catch(() => setJobs([]));
  useEffect(() => {
    load();
  }, []);

  if (!jobs) return null;

  const toggle = async (job: CronJob) => {
    const next = !(pending[job.id] ?? job.enabled);
    setPending((p) => ({ ...p, [job.id]: next }));
    try {
      await post(`/api/cron/jobs/${job.id}/${next ? "resume" : "pause"}`);
      showToast(next ? `${job.name || job.id} enabled` : `${job.name || job.id} disabled`);
      load();
    } catch {
      setPending((p) => ({ ...p, [job.id]: job.enabled }));
      showToast("Couldn't update job");
    }
  };

  const run = async (job: CronJob) => {
    setRanJobs((r) => ({ ...r, [job.id]: true }));
    try {
      await post(`/api/cron/jobs/${job.id}/trigger`);
      showToast(`Dispatched ${job.name || job.id}`);
    } catch {
      showToast("Couldn't dispatch job");
    } finally {
      setTimeout(() => setRanJobs((r) => ({ ...r, [job.id]: false })), 2600);
    }
  };

  const showLog = (job: CronJob) => {
    const text = [job.last_error, job.last_delivery_error].filter(Boolean).join("\n\n") || "no log available";
    onShowLog(`${job.name || job.id} — last run`, text);
  };

  const removeJob = (job: CronJob) =>
    askConfirm({
      title: "Delete this cron job?",
      body: `${job.name || job.id} will stop running and its schedule is removed. This cannot be undone.`,
      label: "Delete job",
      destructive: true,
      run: async () => {
        try {
          await del(`/api/cron/jobs/${job.id}`);
          showToast(`${job.name || job.id} deleted`);
          load();
        } catch {
          showToast("Couldn't delete job");
        }
      },
    });

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", padding: "0 4px" }}>
        <div style={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
          Cron jobs
        </div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)", whiteSpace: "nowrap" }}>{jobs.length} jobs</div>
      </div>
      {jobs.map((job) => (
        <CronJobCard
          key={job.id}
          job={job}
          enabled={pending[job.id] ?? job.enabled}
          ran={!!ranJobs[job.id]}
          isAdmin={isAdmin}
          onToggle={() => toggle(job)}
          onRun={() => run(job)}
          onRemove={() => removeJob(job)}
          onShowLog={() => showLog(job)}
        />
      ))}
      <div style={{ fontSize: 11, color: "var(--t3)", padding: "2px 4px", lineHeight: 1.5 }}>
        Run now returns after dispatch — long jobs keep running in the background.
      </div>
    </div>
  );
}


const LAST_STATUS: Record<string, { label: string; color: string }> = {
  ok: { label: "last run ok", color: "var(--success)" },
  error: { label: "last run failed", color: "var(--destr)" },
};
const NEVER_RAN = { label: "never ran", color: "var(--t3)" };

interface CronJobCardProps {
  job: CronJob;
  enabled: boolean;
  ran: boolean;
  isAdmin: boolean;
  onToggle: () => void;
  onRun: () => void;
  onRemove: () => void;
  onShowLog: () => void;
}

function CronJobCard({ job, enabled, ran, isAdmin, onToggle, onRun, onRemove, onShowLog }: CronJobCardProps) {
  const hasError = !!job.last_error;
  const nextRun = enabled ? relativeToNow(job.next_run_at) : null;
  return (
    <div
      style={{
        background: "var(--card)",
        border: "1px solid var(--line)",
        borderRadius: 14,
        padding: "13px 14px",
        opacity: enabled ? 1 : 0.55,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontSize: 14,
            fontWeight: 600,
            color: "var(--mid)",
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {job.name || job.id}
        </span>
        <CronToggle enabled={enabled} onClick={onToggle} />
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--t2)", marginTop: 5 }}>
        {job.schedule_display || job.schedule?.display || ""}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 9 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--t3)", whiteSpace: "nowrap" }}>
          {enabled ? (nextRun ? `next in ${nextRun}` : "") : "paused"}
        </span>
        <CronStatusBadge status={job.last_status} hasError={hasError} onShowLog={onShowLog} />
        <CronRunButton ran={ran} onClick={onRun} />
        {isAdmin && <CronDeleteButton onClick={onRemove} />}
      </div>
      {hasError && (
        <div
          onClick={onShowLog}
          style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--destr)", marginTop: 7, lineHeight: 1.4, cursor: "pointer" }}
        >
          {job.last_error} — tap for log
        </div>
      )}
    </div>
  );
}

function CronToggle({ enabled, onClick }: { enabled: boolean; onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      role="switch"
      aria-checked={enabled}
      style={{
        width: 42,
        height: 25,
        borderRadius: 99,
        position: "relative",
        cursor: "pointer",
        flexShrink: 0,
        transition: "background 180ms ease",
        background: enabled ? "var(--success)" : "var(--line2)",
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 2.5,
          width: 20,
          height: 20,
          borderRadius: 99,
          background: "var(--bg)",
          transition: "left 180ms ease",
          boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
          left: enabled ? 19.5 : 2.5,
        }}
      />
    </div>
  );
}

function CronStatusBadge({ status, hasError, onShowLog }: { status: CronJob["last_status"]; hasError: boolean; onShowLog: () => void }) {
  const { label, color } = (status && LAST_STATUS[status]) || NEVER_RAN;
  return (
    <span
      onClick={hasError ? onShowLog : undefined}
      style={{
        fontFamily: "var(--mono)",
        fontSize: 10,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        padding: "2px 7px",
        borderRadius: 6,
        border: "1px solid var(--line)",
        color,
        whiteSpace: "nowrap",
        flexShrink: 0,
        cursor: hasError ? "pointer" : "default",
        textDecoration: hasError ? "underline dotted" : "none",
        textUnderlineOffset: 2,
      }}
    >
      {label}
    </span>
  );
}

function CronRunButton({ ran, onClick }: { ran: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      disabled={ran}
      style={{
        marginLeft: "auto",
        fontFamily: "var(--mono)",
        fontSize: 11,
        letterSpacing: "0.05em",
        padding: "5px 11px",
        borderRadius: 8,
        border: `1px solid ${ran ? "color-mix(in srgb, var(--success) 45%, transparent)" : "var(--line2)"}`,
        background: "transparent",
        cursor: ran ? "default" : "pointer",
        color: ran ? "var(--success)" : "var(--t2)",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      {ran ? "Dispatched" : "Run now"}
    </button>
  );
}

function CronDeleteButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "var(--mono)",
        fontSize: 11,
        letterSpacing: "0.05em",
        padding: "5px 11px",
        borderRadius: 8,
        border: "1px solid var(--line2)",
        background: "transparent",
        cursor: "pointer",
        color: "var(--destr)",
        whiteSpace: "nowrap",
        flexShrink: 0,
      }}
    >
      Delete
    </button>
  );
}
