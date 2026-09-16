import { useCallback, useLayoutEffect, useState } from "react";
import { useNavigate } from "react-router";
import { AlertTriangle, ExternalLink, RefreshCw, ShieldCheck } from "lucide-react";

import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useRemoteApprovals } from "@/contexts/RemoteApprovals";
import { buildChatResumePath } from "@/lib/session-navigation";
import { isRemoteApprovalExpired, type RemoteApproval } from "@/lib/remote-approvals";

function formatTimestamp(value: number | undefined): string {
  if (value === undefined) return "Unknown";
  const date = new Date(value * 1000);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString();
}

function formatAge(value: number | undefined): string {
  if (value === undefined) return "Age unavailable";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - value));
  if (seconds < 60) return "Received less than a minute ago";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `Received ${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Received ${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `Received ${days} day${days === 1 ? "" : "s"} ago`;
}

function sourceLabel(source: string): string {
  return source
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function choiceLabel(choice: RemoteApproval["choices"][number]): string {
  switch (choice) {
    case "once":
      return "Approve once";
    case "session":
      return "Approve for session";
    case "always":
      return "Approve always";
    case "deny":
      return "Deny";
    default:
      return choice;
  }
}

function RemoteApprovalCard({ approval }: { approval: RemoteApproval }) {
  const navigate = useNavigate();
  const { respond } = useRemoteApprovals();
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const expired = isRemoteApprovalExpired(approval);

  const handleRespond = async (choice: RemoteApproval["choices"][number]) => {
    if (submitting || expired) return;
    setSubmitting(choice);
    setActionError(null);
    try {
      await respond(approval, choice);
    } catch (reason: unknown) {
      setActionError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <Card data-slot="remote-approval-card" className="border-amber-500/40">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" aria-hidden="true" />
            <div className="min-w-0">
              <CardTitle className="truncate text-base">
                {approval.title || "Untitled session"}
              </CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">
                {sourceLabel(approval.source)} · profile {approval.profile} · {formatAge(approval.created_at)}
              </p>
            </div>
          </div>
          <span className={expired ? "text-xs text-destructive" : "text-xs text-amber-600"}>
            {expired ? "Expired" : "Waiting for approval"}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-md border border-current/15 bg-current/5 p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Risk explanation</p>
          <p className="mt-1 whitespace-pre-wrap break-words text-sm">
            {approval.description || "The agent is waiting for permission to continue."}
          </p>
          <p className="mt-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">Command</p>
          <code className="mt-1 block whitespace-pre-wrap break-words text-sm text-secondary-foreground">
            {approval.command || "Command details unavailable"}
          </code>
        </div>

        <dl className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
          <div>
            <dt className="font-medium text-foreground">Session</dt>
            <dd className="break-all font-mono">{approval.stored_session_id}</dd>
          </div>
          <div>
            <dt className="font-medium text-foreground">Expires</dt>
            <dd>{formatTimestamp(approval.expires_at)}</dd>
          </div>
        </dl>

        {actionError && <p role="alert" className="text-sm text-destructive">{actionError}</p>}
        {expired && (
          <p role="status" className="text-sm text-muted-foreground">
            This request is no longer actionable. Refresh to reconcile the server state.
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2" aria-label="Approval actions">
          {approval.choices.map((choice) => (
            <Button
              key={choice}
              type="button"
              data-choice={choice}
              disabled={expired || submitting !== null}
              onClick={() => void handleRespond(choice)}
              outlined={choice === "deny"}
            >
              {submitting === choice ? "Submitting…" : choiceLabel(choice)}
            </Button>
          ))}
          <Button
            type="button"
            data-action="open-session"
            ghost
            onClick={() => navigate(buildChatResumePath(approval.stored_session_id, approval.profile))}
          >
            <ExternalLink className="mr-2 h-4 w-4" aria-hidden="true" />
            Open session
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function PendingApprovalsPage() {
  const { profile, currentProfile } = useProfileScope();
  const { approvals, error, lastUpdatedAt, loading, refresh, refreshing } = useRemoteApprovals();
  const { setAfterTitle, setEnd } = usePageHeader();
  const scope = profile || currentProfile || "default";

  const refreshPage = useCallback(() => refresh(), [refresh]);

  useLayoutEffect(() => {
    setAfterTitle(null);
    setEnd(
      <Button
        type="button"
        ghost
        size="icon"
        aria-label="Refresh pending approvals"
        title="Refresh pending approvals"
        disabled={refreshing}
        onClick={() => void refreshPage()}
      >
        {refreshing ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [refreshPage, refreshing, setAfterTitle, setEnd]);

  return (
    <div data-slot="pending-approvals-page" className="flex min-w-0 flex-col gap-6">
      <div>
        <p className="text-sm text-muted-foreground">
          Review dangerous actions waiting on the selected profile ({scope}) for Native Chat/TUI sessions attached to this Dashboard backend. Requests are read from the server-owned gateway queue; this page never starts a new agent turn. General messaging/API gateway runs and approvals owned by another process are not listed here.
        </p>
        {lastUpdatedAt !== null && (
          <p className="mt-1 text-xs text-muted-foreground">Last checked {new Date(lastUpdatedAt).toLocaleTimeString()}</p>
        )}
      </div>

      {loading && approvals.length === 0 && (
        <div className="flex justify-center py-24" role="status" aria-live="polite">
          <Spinner className="text-2xl text-primary" />
        </div>
      )}
      {error && (
        <Card role="alert">
          <CardContent className="flex items-start gap-3 py-6">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" aria-hidden="true" />
            <div>
              <p className="text-sm font-medium">Could not load pending approvals</p>
              <p className="mt-1 break-words text-sm text-destructive">{error}</p>
            </div>
          </CardContent>
        </Card>
      )}
      {!loading && !error && approvals.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-16 text-center">
            <ShieldCheck className="h-8 w-8 text-emerald-500" aria-hidden="true" />
            <p className="text-sm font-medium">No pending approvals</p>
            <p className="max-w-md text-sm text-muted-foreground">
              There are no server-owned approval requests waiting in this profile.
            </p>
          </CardContent>
        </Card>
      )}
      {approvals.length > 0 && (
        <div className="grid min-w-0 gap-4" aria-live="polite">
          {approvals.map((approval) => (
            <RemoteApprovalCard key={`${approval.session_key}:${approval.request_id}`} approval={approval} />
          ))}
        </div>
      )}
    </div>
  );
}
