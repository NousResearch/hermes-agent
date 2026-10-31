import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import { MonitorDot, RefreshCw } from "lucide-react";
import { Segmented } from "@nous-research/ui/ui/components/segmented";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { authedFetch, fetchJSON } from "@/lib/api";
import type { PaginatedSessions, SessionInfo } from "@/lib/api";
import { getManagementProfile } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { errorMessage } from "@/lib/api-error";
import { usePageHeader } from "@/contexts/usePageHeader";
import { sourceLabel } from "@/pages/SessionsPage_sources";

interface ComputerStatus {
  enabled: boolean;
  last_capture_at: number | null;
  count: number;
}

const AUTO_INTERVAL_MS = 10_000;

function dotClass(s: SessionInfo): string {
  return s.is_active ? "bg-success" : "bg-muted-foreground/50";
}

function SessionCell({ s, onClick }: { s: SessionInfo; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex min-w-0 flex-col gap-0.5 rounded-sm border border-border bg-card px-2 py-1.5 text-left hover:border-foreground/40"
      title={s.title ?? s.id}
    >
      <span className="flex min-w-0 items-center gap-1.5">
        <span className={`h-2 w-2 shrink-0 rounded-full ${dotClass(s)}`} aria-hidden="true" />
        <span className="flex min-w-0 flex-1 gap-1 text-sm text-foreground">
          <span className="min-w-0 truncate">{s.title || s.preview || s.id.slice(0, 12)}</span>
        </span>
      </span>
      <span className="flex min-w-0 items-center gap-1.5 text-[0.6875rem] text-text-tertiary">
        <span className="truncate">{sourceLabel(s.source ?? "unknown")}</span>
        <span aria-hidden="true">·</span>
        <span className="shrink-0 truncate">{timeAgo(s.last_active)}</span>
        {typeof s.message_count === "number" && (
          <>
            <span aria-hidden="true">·</span>
            <span className="shrink-0">{s.message_count} msgs</span>
          </>
        )}
      </span>
    </button>
  );
}

function getSessionsGrid(profile: string): Promise<PaginatedSessions> {
  return fetchJSON<PaginatedSessions>(
    `/api/sessions?limit=24&offset=0&order=recent&profile=${encodeURIComponent(profile)}`);
}

export default function ComputerPage() {
  const { setAfterTitle, setEnd } = usePageHeader();
  const navigate = useNavigate();
  const [status, setStatus] = useState<ComputerStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<PaginatedSessions | null>(null);
  const [shots, setShots] = useState<Blob | null>(null);
  const [shotAt, setShotAt] = useState<number | null>(null);
  const [shotLoading, setShotLoading] = useState(false);
  const [shotError, setShotError] = useState<string | null>(null);
  const [autoMode, setAutoMode] = useState<"off" | "10s">("off");
  const shotUrlRef = useRef<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const s = await fetchJSON<ComputerStatus>("/api/computer/status");
      setStatus(s);
      setStatusError(null);
    } catch (e) {
      setStatusError(errorMessage(e));
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      setSessions(
        await getSessionsGrid(getManagementProfile()),
      );
    } catch {
      // non-destructive: the ticker sections still render when the list fails
    }
  }, []);

  const loadShot = useCallback(async () => {
    setShotLoading(true);
    try {
      const res = await authedFetch("/api/computer/screenshot");
      if (!res.ok) throw new Error(`${res.status}`);
      const blob = await res.blob();
      setShots(blob);
      setShotAt(Date.now());
      setShotError(null);
      void loadStatus();
    } catch (e) {
      const msg = errorMessage(e);
      setShotError(msg.includes("409") ? "disabled" : msg);
    } finally {
      setShotLoading(false);
    }
  }, [loadStatus]);

  // The page header shows the live capture stamp on the right.
  useEffect(() => {
    setAfterTitle(
      shotAt ? (
        <span className="text-xs text-text-tertiary">captured {timeAgo(shotAt)}</span>
      ) : null,
    );
    setEnd(null);
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [shotAt, setAfterTitle, setEnd]);

  useEffect(() => {
    void loadStatus();
    void loadSessions();
  }, [loadStatus, loadSessions]);

  useEffect(() => {
    if (status && status.enabled && shotUrlRef.current === null && !shotError) {
      void loadShot();
    }
  }, [status, loadShot, shotError]);

  useEffect(() => {
    if (autoMode === "off") return;
    const id = window.setInterval(() => void loadShot(), AUTO_INTERVAL_MS);
    return () => window.clearInterval(id);
  }, [autoMode, loadShot]);

  // Keep exactly one object URL alive; revoke the previous on every refresh.
  const objectUrl = shots ? URL.createObjectURL(shots) : null;
  useEffect(() => {
    const prev = shotUrlRef.current;
    if (prev && prev !== objectUrl) URL.revokeObjectURL(prev);
    shotUrlRef.current = objectUrl;
    return () => {
      if (objectUrl && shotUrlRef.current === objectUrl) {
        URL.revokeObjectURL(objectUrl);
        shotUrlRef.current = null;
      }
    };
  }, [objectUrl]);

  const rows = sessions?.sessions ?? [];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 pb-4 pt-2 sm:px-6">
      <section aria-label="Live session grid">
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-sm uppercase tracking-wider text-text-tertiary">
              Live sessions {sessions?.total !== undefined && `(${sessions.total})`}
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {rows.map((s: SessionInfo) => (
              <SessionCell
                key={s.id}
                s={s}
                onClick={() => navigate(`/chat?resume=${encodeURIComponent(s.id)}`)}
              />
            ))}
            {rows.length === 0 && (
              <p className="col-span-full py-4 text-center text-xs text-text-secondary">
                No sessions to show
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      <section aria-label="Computer view">
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="flex items-center gap-2 text-sm uppercase tracking-wider text-text-tertiary">
              <MonitorDot className="h-4 w-4" /> Computer view
              {status && (
                <Badge tone={status.enabled ? "success" : "secondary"}>
                  {status.enabled ? "opted in" : "off"}
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {statusError && (
              <p className="text-xs text-destructive">{statusError}</p>
            )}
            {status && !status.enabled && (
              <p className="text-xs text-text-secondary">
                Screen capture is opt-in because it sends physical-screen pixels through the
                dashboard. Enable it once with:
                <code className="ml-1 rounded bg-card px-1 py-0.5 text-[0.6875rem]">
                  dashboard.computer_view: true
                </code>
                in /home/min/.hermes/config.yaml (or the Config page).
              </p>
            )}
            {status?.enabled && (
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" onClick={() => void loadShot()} disabled={shotLoading}>
                  {shotLoading ? <Spinner className="h-3.5 w-3.5" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  Capture now
                </Button>
                <Segmented
                  value={autoMode}
                  onChange={(v: string) => setAutoMode(v === "10s" ? "10s" : "off")}
                  options={[
                    { value: "off", label: "Manual" },
                    { value: "10s", label: "Auto 10s" },
                  ]}
                />
                {shotError && shotError !== "disabled" && (
                  <span className="text-xs text-destructive">{shotError}</span>
                )}
              </div>
            )}
            {objectUrl && (
              <img
                src={objectUrl}
                alt="Latest desktop screenshot"
                className="max-h-[60vh] w-auto max-w-full self-start rounded-sm border border-border"
              />
            )}
          </CardContent>
        </Card>
      </section>
    </div>
  );
}