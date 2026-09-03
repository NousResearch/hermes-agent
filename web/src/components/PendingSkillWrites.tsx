import { useCallback, useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, FileDiff, RefreshCw, X } from "lucide-react";
import { api } from "@/lib/api";
import type { SkillPendingWrite } from "@/lib/api";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";

interface PendingSkillWritesProps {
  onApproved: () => void;
  profile?: string;
  showToast: (message: string, kind: "success" | "error") => void;
}

type PendingAction = {
  kind: "approve" | "reject";
  write: SkillPendingWrite;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed.";
}

export function PendingSkillWrites({
  onApproved,
  profile,
  showToast,
}: PendingSkillWritesProps) {
  const [pending, setPending] = useState<SkillPendingWrite[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [diff, setDiff] = useState<{ id: string; text: string } | null>(null);
  const [loadingDiffId, setLoadingDiffId] = useState<string | null>(null);
  const [action, setAction] = useState<PendingAction | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .getPendingSkillWrites(profile)
      .then((writes) => {
        if (!cancelled) setPending(writes);
      })
      .catch((error) => {
        if (!cancelled) showToast(`Pending skill writes: ${errorMessage(error)}`, "error");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [profile, showToast]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      setPending(await api.getPendingSkillWrites(profile));
      setDiff(null);
    } catch (error) {
      showToast(`Pending skill writes: ${errorMessage(error)}`, "error");
    } finally {
      setRefreshing(false);
    }
  }, [profile, showToast]);

  const toggleDiff = useCallback(
    async (write: SkillPendingWrite) => {
      if (diff?.id === write.id) {
        setDiff(null);
        return;
      }
      setLoadingDiffId(write.id);
      try {
        const result = await api.getPendingSkillDiff(write.id, profile);
        setDiff({ id: write.id, text: result.diff });
      } catch (error) {
        showToast(`Skill diff: ${errorMessage(error)}`, "error");
      } finally {
        setLoadingDiffId(null);
      }
    },
    [diff?.id, profile, showToast],
  );

  const submitAction = useCallback(async () => {
    if (!action) return;
    setApplying(true);
    try {
      if (action.kind === "approve") {
        await api.approvePendingSkillWrite(action.write.id, profile);
        onApproved();
      } else {
        await api.rejectPendingSkillWrite(action.write.id, profile);
      }
      setPending((writes) => writes.filter((write) => write.id !== action.write.id));
      setDiff((current) => (current?.id === action.write.id ? null : current));
      showToast(
        `${action.kind === "approve" ? "Approved" : "Rejected"} ${action.write.id}`,
        "success",
      );
      setAction(null);
    } catch (error) {
      showToast(`Skill write: ${errorMessage(error)}`, "error");
    } finally {
      setApplying(false);
    }
  }, [action, onApproved, profile, showToast]);

  if (loading || pending.length === 0) return null;

  return (
    <>
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-3 py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            Pending skill writes
            <Badge>{pending.length}</Badge>
          </CardTitle>
          <Button
            ghost
            size="xs"
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label="Refresh pending skill writes"
            title="Refresh"
          >
            <RefreshCw className={refreshing ? "animate-spin" : ""} />
          </Button>
        </CardHeader>
        <CardContent className="pt-0">
          <ul className="divide-y divide-border border-y border-border">
            {pending.map((write) => {
              const expanded = diff?.id === write.id;
              const loadingDiff = loadingDiffId === write.id;
              return (
                <li key={write.id} className="py-3">
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                    <div className="min-w-0">
                      <p className="truncate text-sm text-foreground">{write.summary}</p>
                      <p className="mt-0.5 font-mono-ui text-xs text-text-secondary">
                        {write.id} · {write.action}
                        {write.origin === "background_review" ? " · auto" : ""}
                      </p>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        ghost
                        size="sm"
                        prefix={<FileDiff />}
                        suffix={expanded ? <ChevronUp /> : <ChevronDown />}
                        onClick={() => void toggleDiff(write)}
                        disabled={loadingDiff}
                        aria-expanded={expanded}
                        aria-controls={`skill-diff-${write.id}`}
                      >
                        {expanded ? "Hide" : "Diff"}
                      </Button>
                      <Button
                        outlined
                        size="sm"
                        prefix={<Check />}
                        onClick={() => setAction({ kind: "approve", write })}
                      >
                        Approve
                      </Button>
                      <Button
                        destructive
                        size="sm"
                        prefix={<X />}
                        onClick={() => setAction({ kind: "reject", write })}
                      >
                        Reject
                      </Button>
                    </div>
                  </div>
                  {expanded && (
                    <pre
                      id={`skill-diff-${write.id}`}
                      className="mt-3 max-h-80 overflow-auto border border-border bg-muted/30 p-3 font-mono-ui text-xs leading-relaxed whitespace-pre-wrap break-all"
                    >
                      {diff.text}
                    </pre>
                  )}
                </li>
              );
            })}
          </ul>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={action !== null}
        onCancel={() => !applying && setAction(null)}
        onConfirm={() => void submitAction()}
        title={action?.kind === "approve" ? "Approve skill write" : "Reject skill write"}
        description={
          action
            ? `${action.kind === "approve" ? "Apply" : "Discard"} ${action.write.id}: ${action.write.summary}`
            : undefined
        }
        confirmLabel={action?.kind === "approve" ? "Approve" : "Reject"}
        destructive={action?.kind === "reject"}
        loading={applying}
      />
    </>
  );
}
