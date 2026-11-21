import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { Check, ShieldCheck, Trash2, Users, X } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { PairingResponse, PairingUser } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";

function getUserKey(user: PairingUser): string {
  return `${user.platform}:${user.user_id}`;
}

function splitUserKey(key: string): { platform: string; user_id: string } {
  const idx = key.indexOf(":");
  if (idx === -1) return { platform: "", user_id: key };
  return { platform: key.slice(0, idx), user_id: key.slice(idx + 1) };
}

function getUserLabel(user: PairingUser): string {
  return user.user_name || user.user_id;
}

export default function PairingPage() {
  const [pending, setPending] = useState<PairingUser[]>([]);
  const [approved, setApproved] = useState<PairingUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [approving, setApproving] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const { t } = useI18n();
  const pairingUi = t.pairingUi ?? en.pairingUi!;

  const loadPairing = useCallback(() => {
    api
      .getPairing()
      .then((res: PairingResponse) => {
        setPending(res.pending);
        setApproved(res.approved);
      })
      .catch(() => showToast(pairingUi.failedToLoad, "error"))
      .finally(() => setLoading(false));
  }, [pairingUi.failedToLoad, showToast]);

  useEffect(() => {
    loadPairing();
  }, [loadPairing]);

  const handleApprove = async (user: PairingUser) => {
    if (!user.request_id) {
      showToast(pairingUi.missingRequest, "error");
      return;
    }
    const key = getUserKey(user);
    setApproving(key);
    try {
      await api.approvePairing(user.platform, user.request_id);
      showToast(pairingUi.approved.replace("{user}", getUserLabel(user)), "success");
      loadPairing();
    } catch (e) {
      showToast(pairingUi.failedToApprove.replace("{error}", errorMessage(e)), "error");
    } finally {
      setApproving(null);
    }
  };

  const handleClearPending = useCallback(async () => {
    if (!window.confirm(pairingUi.confirmClearPending)) return;
    setClearing(true);
    try {
      const res = await api.clearPendingPairing();
      showToast(pairingUi.clearedPending.replace("{count}", String(res.cleared)), "success");
      loadPairing();
    } catch (e) {
      showToast(pairingUi.failedToClear.replace("{error}", errorMessage(e)), "error");
    } finally {
      setClearing(false);
    }
  }, [loadPairing, pairingUi.clearedPending, pairingUi.confirmClearPending, pairingUi.failedToClear, showToast]);

  const userRevoke = useConfirmDelete({
    onDelete: useCallback(
      async (key: string) => {
        const { platform, user_id } = splitUserKey(key);
        const user = approved.find((u) => getUserKey(u) === key);
        try {
          await api.revokePairing(platform, user_id);
          showToast(
            pairingUi.revoked.replace("{user}", user ? getUserLabel(user) : user_id),
            "success",
          );
          loadPairing();
        } catch (e) {
          showToast(pairingUi.failedToRevoke.replace("{error}", errorMessage(e)), "error");
          throw e;
        }
      },
      [approved, loadPairing, pairingUi.failedToRevoke, pairingUi.revoked, showToast],
    ),
  });

  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        onClick={handleClearPending}
        disabled={clearing}
        prefix={clearing ? <Spinner /> : <Trash2 className="h-4 w-4" />}
      >
        {pairingUi.clearPending}
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [clearing, handleClearPending, pairingUi.clearPending, setEnd]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const pendingRevokeUser = userRevoke.pendingId
    ? approved.find((u) => getUserKey(u) === userRevoke.pendingId)
    : null;

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={userRevoke.isOpen}
        onCancel={userRevoke.cancel}
        onConfirm={userRevoke.confirm}
        title={pairingUi.revokeAccess}
        description={
          pendingRevokeUser
            ? pairingUi.revokeUserDescription.replace("{user}", getUserLabel(pendingRevokeUser))
            : pairingUi.revokeDescription
        }
        confirmLabel={pairingUi.revoke}
        loading={userRevoke.isDeleting}
      />

      {/* Pending requests */}
      <div className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Users className="h-4 w-4" />
          {pairingUi.pendingRequests.replace("{count}", String(pending.length))}
        </H2>

        {pending.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {pairingUi.noPendingRequests}
            </CardContent>
          </Card>
        )}

        {pending.map((user) => {
          const key = getUserKey(user);
          return (
            <Card key={key}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge tone="outline">{user.platform}</Badge>
                    <span className="font-medium text-sm truncate">
                      {getUserLabel(user)}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span className="truncate">{user.user_id}</span>
                    {typeof user.age_minutes === "number" && (
                      <span>{pairingUi.minutesAgo.replace("{count}", String(user.age_minutes))}</span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    size="sm"
                    className="uppercase"
                    onClick={() => handleApprove(user)}
                    disabled={approving === key || !user.request_id}
                    prefix={
                      approving === key ? (
                        <Spinner />
                      ) : (
                        <Check className="h-4 w-4" />
                      )
                    }
                  >
                    {pairingUi.approve}
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Approved users */}
      <div className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <ShieldCheck className="h-4 w-4" />
          {pairingUi.approvedUsers.replace("{count}", String(approved.length))}
        </H2>

        {approved.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              {pairingUi.noApprovedUsers}
            </CardContent>
          </Card>
        )}

        {approved.map((user) => {
          const key = getUserKey(user);
          return (
            <Card key={key}>
              <CardContent className="flex items-start gap-4 py-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge tone="outline">{user.platform}</Badge>
                    <span className="font-medium text-sm truncate">
                      {user.user_id}
                    </span>
                  </div>
                  {user.user_name && (
                    <div className="text-xs text-muted-foreground truncate">
                      {user.user_name}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-1 shrink-0">
                  <Button
                    ghost
                    size="icon"
                    title={pairingUi.revoke}
                    aria-label={pairingUi.revoke}
                    className="text-destructive"
                    onClick={() => userRevoke.requestDelete(key)}
                  >
                    <X />
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
