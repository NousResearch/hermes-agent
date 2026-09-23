import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Check,
  Copy,
  KeyRound,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@nous-research/ui/ui/components/dialog";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { ApiServerInfo, ApiServerKey } from "@/lib/api";
import { copyTextToClipboard } from "@/lib/clipboard";
import { errorMessage } from "@/lib/api-error";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return DATE_FORMAT.format(new Date(iso));
  } catch {
    return iso;
  }
}

export default function ApiKeysPage() {
  const [info, setInfo] = useState<ApiServerInfo | null>(null);
  const [keys, setKeys] = useState<ApiServerKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [plaintextShown, setPlaintextShown] = useState<{
    name: string;
    plaintext: string;
  } | null>(null);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const createModalRef = useModalBehavior({
    open: createOpen,
    onClose: () => setCreateOpen(false),
  });

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [infoResp, keysResp] = await Promise.all([
        api.getApiServerInfo(),
        api.listApiServerKeys(false),
      ]);
      setInfo(infoResp);
      setKeys(keysResp.keys);
    } catch (e) {
      showToast(`Could not load API keys: ${errorMessage(e)}`, "error");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    setEnd(null);
    return () => setEnd(null);
  }, [setEnd]);

  const onCreated = useCallback(
    async (name: string, description: string) => {
      try {
        const resp = await api.createApiServerKey({ name, description });
        // Show the plaintext exactly once. The dialog stays open until the
        // operator explicitly closes it — they need to copy the secret now.
        setPlaintextShown({ name: resp.name, plaintext: resp.plaintext });
        setCreateOpen(false);
        showToast(`API key "${resp.name}" created.`, "success");
        await refresh();
      } catch (e) {
        showToast(`Could not create key: ${errorMessage(e)}`, "error");
        throw e;
      }
    },
    [refresh, showToast],
  );

  const keyDelete = useConfirmDelete<string>({
    onDelete: useCallback(
      async (id: string) => {
        try {
          await api.revokeApiServerKey(id);
          const k = keys.find((kk) => kk.id === id);
          showToast(`Revoked "${k?.name ?? id}".`, "success");
          await refresh();
        } catch (e) {
          showToast(`Could not revoke: ${errorMessage(e)}`, "error");
          throw e;
        }
      },
      [keys, refresh, showToast],
    ),
  });
  const pendingKey = keyDelete.pendingId
    ? keys.find((k) => k.id === keyDelete.pendingId) ?? null
    : null;

  return (
    <div className="flex flex-col gap-6 p-6">
      <Card>
        <CardContent className="flex flex-col gap-3 py-5">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-muted-foreground" />
            <span className="font-semibold text-sm tracking-wide">
              Hermes API
            </span>
          </div>
          {info ? (
            <BaseUrlRow baseUrl={info.base_url} />
          ) : (
            <div className="text-sm text-muted-foreground">
              Loading endpoint…
            </div>
          )}
          {info?.legacy_configured ? (
            <div className="text-xs text-muted-foreground">
              The legacy <span className="font-mono">API_SERVER_KEY</span> is
              also still active; both it and managed keys authenticate
              requests to this endpoint.
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-4 py-5">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="font-semibold text-sm tracking-wide">
                API Keys
              </div>
              <div className="text-xs text-muted-foreground">
                Manage credentials that external applications send as{" "}
                <span className="font-mono">Authorization: Bearer …</span> to
                authenticate against the Hermes OpenAI-compatible API.
              </div>
            </div>
            <Button onClick={() => setCreateOpen(true)}>
              <Plus className="mr-1 h-3.5 w-3.5" /> Create API Key
            </Button>
          </div>

          {loading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="text-base" /> Loading keys…
            </div>
          ) : keys.length === 0 ? (
            <div className="rounded border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
              No managed keys yet. Create one to give an external app access
              to the Hermes API.
            </div>
          ) : (
            <div className="overflow-hidden rounded border border-border">
              <table className="w-full text-sm">
                <thead className="bg-muted/40 text-xs text-muted-foreground">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Name</th>
                    <th className="px-3 py-2 text-left font-medium">Key</th>
                    <th className="px-3 py-2 text-left font-medium">
                      Created
                    </th>
                    <th className="px-3 py-2 text-left font-medium">
                      Last used
                    </th>
                    <th className="px-3 py-2 text-left font-medium">
                      Status
                    </th>
                    <th className="px-3 py-2 text-right font-medium">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k.id} className="border-t border-border">
                      <td className="px-3 py-2">
                        <div className="font-medium">{k.name}</div>
                        {k.description ? (
                          <div className="text-xs text-muted-foreground">
                            {k.description}
                          </div>
                        ) : null}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs">
                        {k.prefix}…
                      </td>
                      <td className="px-3 py-2 text-xs">
                        {formatDate(k.created_at)}
                      </td>
                      <td className="px-3 py-2 text-xs">
                        {formatDate(k.last_used_at)}
                      </td>
                      <td className="px-3 py-2">
                        {k.active ? (
                          <Badge tone="success">Active</Badge>
                        ) : (
                          <Badge tone="secondary">Revoked</Badge>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {k.active ? (
                          <Button
                            ghost
                            size="sm"
                            disabled={keyDelete.isDeleting}
                            onClick={() => keyDelete.requestDelete(k.id)}
                          >
                            {keyDelete.isDeleting ? (
                              <Spinner className="text-base" />
                            ) : (
                              <>
                                <Trash2 className="mr-1 h-3.5 w-3.5" /> Revoke
                              </>
                            )}
                          </Button>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            —
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      <CreateKeyDialog
        open={createOpen}
        onOpenChange={(open) => {
          if (!open) setCreateOpen(false);
        }}
        modalRef={createModalRef}
        onSubmit={onCreated}
      />

      <PlaintextDialog
        shown={plaintextShown}
        onClose={() => setPlaintextShown(null)}
      />

      <ConfirmDialog
        open={keyDelete.isOpen}
        onCancel={keyDelete.cancel}
        onConfirm={() => {
          void keyDelete.confirm();
        }}
        title="Revoke API key?"
        description={
          pendingKey
            ? `The key "${pendingKey.name}" will be rejected by the API server immediately. External applications using it will receive 401 errors until they are reconfigured with a new key.`
            : "This API key will be revoked immediately."
        }
        confirmLabel="Revoke"
      />
      <Toast toast={toast} />
    </div>
  );
}

function BaseUrlRow({ baseUrl }: { baseUrl: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(async () => {
    const ok = await copyTextToClipboard(baseUrl);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  }, [baseUrl]);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Base URL:</span>
      <code className="flex-1 truncate rounded border border-border bg-muted/30 px-2 py-1 font-mono text-xs">
        {baseUrl}
      </code>
      <Button ghost size="sm" onClick={onCopy}>
        {copied ? (
          <>
            <Check className="mr-1 h-3.5 w-3.5" /> Copied
          </>
        ) : (
          <>
            <Copy className="mr-1 h-3.5 w-3.5" /> Copy
          </>
        )}
      </Button>
    </div>
  );
}

function CreateKeyDialog({
  open,
  onOpenChange,
  modalRef,
  onSubmit,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  modalRef: React.RefObject<HTMLDivElement | null>;
  onSubmit: (name: string, description: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) {
      setName("");
      setDescription("");
      setSubmitting(false);
    }
  }, [open]);

  const nameError = useMemo(() => {
    if (!name.trim()) return null;
    if (name.trim().length > 128) return "Name must be 128 characters or fewer";
    return null;
  }, [name]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent ref={modalRef} className="max-w-md">
        <DialogHeader>
          <DialogTitle>Create API key</DialogTitle>
          <DialogDescription>
            External applications use this key to call the Hermes API as{" "}
            <span className="font-mono">Authorization: Bearer …</span>. The
            full secret is shown{" "}
            <span className="font-medium">exactly once</span> after creation
            and cannot be retrieved later.
          </DialogDescription>
        </DialogHeader>

        <form
          onSubmit={async (e) => {
            e.preventDefault();
            if (!name.trim()) return;
            setSubmitting(true);
            try {
              await onSubmit(name.trim(), description.trim());
            } catch {
              // The page-level toast already showed the failure.
            } finally {
              setSubmitting(false);
            }
          }}
          className="flex flex-col gap-4"
        >
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="api-key-name">Name</Label>
            <Input
              id="api-key-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Mac Studio — Codex"
              maxLength={128}
              required
              autoFocus
            />
            {nameError ? (
              <span className="text-xs text-destructive">{nameError}</span>
            ) : null}
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="api-key-description">
              Description{" "}
              <span className="text-xs text-muted-foreground">(optional)</span>
            </Label>
            <Input
              id="api-key-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What is this key for?"
              maxLength={512}
            />
          </div>

          <DialogFooter className="gap-2">
            <Button
              type="button"
              ghost
              onClick={() => onOpenChange(false)}
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || submitting}>
              {submitting ? <Spinner className="text-base" /> : "Create"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function PlaintextDialog({
  shown,
  onClose,
}: {
  shown: { name: string; plaintext: string } | null;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    if (shown) {
      setCopied(false);
      setAcknowledged(false);
    }
  }, [shown]);

  const onCopy = useCallback(async () => {
    if (!shown) return;
    const ok = await copyTextToClipboard(shown.plaintext);
    if (ok) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    }
  }, [shown]);

  if (!shown) return null;

  return (
    <Dialog open={true} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Save your new API key</DialogTitle>
          <DialogDescription>
            This is the only time the full secret will be shown. Copy it now
            and store it somewhere safe — for example, paste it into your
            external app's Hermes API credentials.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <div className="rounded border border-border bg-muted/30 p-3">
            <div className="text-xs text-muted-foreground">
              Key: <span className="font-medium">{shown.name}</span>
            </div>
            <code className="mt-2 block break-all font-mono text-xs">
              {shown.plaintext}
            </code>
          </div>

          <Button onClick={onCopy}>
            {copied ? (
              <>
                <Check className="mr-1 h-3.5 w-3.5" /> Copied
              </>
            ) : (
              <>
                <Copy className="mr-1 h-3.5 w-3.5" /> Copy key
              </>
            )}
          </Button>

          <label className="flex items-start gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
              className="mt-0.5"
            />
            <span>
              I have saved this key. I understand it cannot be retrieved from
              the dashboard later.
            </span>
          </label>
        </div>

        <DialogFooter>
          <Button onClick={onClose} disabled={!acknowledged}>
            <X className="mr-1 h-3.5 w-3.5" /> Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}