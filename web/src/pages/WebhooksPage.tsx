import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  AlertTriangle,
  Check,
  Copy,
  Plus,
  RotateCw,
  Trash2,
  Webhook,
  X,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type { WebhookRoute, WebhooksResponse } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn, themedBody } from "@/lib/utils";
import { useI18n } from "@/i18n";

interface CreatedWebhook {
  url: string;
  secret: string;
}

function CopyButton({ value }: { value: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard
      .writeText(value)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  }, [value]);
  return (
    <Button
      ghost
      size="icon"
      title={t.webhooks.copySecret}
      aria-label={t.webhooks.copySecret}
      onClick={handleCopy}
      className="text-muted-foreground hover:text-foreground"
    >
      {copied ? <Check /> : <Copy />}
    </Button>
  );
}

export default function WebhooksPage() {
  const { t } = useI18n();
  const [data, setData] = useState<WebhooksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [enabling, setEnabling] = useState(false);
  const [restartNeeded, setRestartNeeded] = useState(false);
  const [restartMessage, setRestartMessage] = useState<string | null>(null);
  const [restartError, setRestartError] = useState<string | null>(null);
  const [restarting, setRestarting] = useState(false);
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();

  // New subscription modal state
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [events, setEvents] = useState("");
  const [deliver, setDeliver] = useState("log");
  const [deliverOnly, setDeliverOnly] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CreatedWebhook | null>(null);

  const closeCreateModal = useCallback(() => {
    setCreateModalOpen(false);
    setCreated(null);
  }, []);
  const createModalRef = useModalBehavior({
    open: createModalOpen,
    onClose: closeCreateModal,
  });

  const enabled = data?.enabled ?? false;
  const subscriptions = data?.subscriptions ?? [];

  const loadWebhooks = useCallback(() => {
    return api
      .getWebhooks()
      .then(setData)
      .catch(() => showToast("Failed to load webhooks", "error"))
      .finally(() => setLoading(false));
  }, [showToast]);

  useEffect(() => {
    loadWebhooks();
  }, [loadWebhooks]);

  const watchRestartOutcome = useCallback(async () => {
    for (let i = 0; i < 20; i++) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      try {
        const st = await api.getActionStatus("gateway-restart", 5);
        if (st.running) continue;
        if (st.exit_code !== 0 && st.exit_code !== null) {
          setRestartMessage(null);
          setRestartNeeded(true);
          setRestartError(`Gateway restart failed with exit ${st.exit_code}.`);
          showToast(
            `Gateway restart failed (exit ${st.exit_code}) — restart manually`,
            "error",
          );
        } else {
          setRestartMessage(null);
          setRestartNeeded(false);
          setRestartError(null);
        }
        return;
      } catch {
        // The dashboard may briefly lose its connection while the gateway restarts.
      }
    }
    setRestartMessage(null);
  }, [showToast]);

  const handleRestart = useCallback(async () => {
    setRestarting(true);
    try {
      await api.restartGateway();
      setRestartNeeded(false);
      setRestartError(null);
      setRestartMessage("Gateway restarting…");
      showToast("Gateway restarting…", "success");
      setTimeout(() => void loadWebhooks(), 4000);
      void watchRestartOutcome();
    } catch (e) {
      setRestartNeeded(true);
      setRestartError(String(e));
      showToast(`Failed to restart: ${e}`, "error");
    } finally {
      setRestarting(false);
    }
  }, [loadWebhooks, showToast, watchRestartOutcome]);

  const handleEnableWebhooks = useCallback(async () => {
    setEnabling(true);
    setRestartNeeded(false);
    setRestartError(null);
    try {
      const result = await api.enableWebhooks();
      await loadWebhooks();
      if (result.restart_started) {
        setRestartMessage("Webhooks enabled; gateway restarting…");
        showToast("Webhooks enabled; gateway restarting…", "success");
        setTimeout(() => void loadWebhooks(), 4000);
        void watchRestartOutcome();
      } else {
        const detail = result.restart_error ? `: ${result.restart_error}` : ".";
        setRestartMessage(null);
        setRestartNeeded(true);
        setRestartError(`Gateway restart failed${detail}`);
        showToast(`Webhooks enabled; gateway restart failed${detail}`, "error");
      }
    } catch (e) {
      showToast(`Failed to enable webhooks: ${e}`, "error");
    } finally {
      setEnabling(false);
    }
  }, [loadWebhooks, showToast, watchRestartOutcome]);

  const resetForm = useCallback(() => {
    setName("");
    setDescription("");
    setEvents("");
    setDeliver("log");
    setDeliverOnly(false);
    setPrompt("");
  }, []);

  const handleCreate = async () => {
    if (!name.trim()) {
      showToast("Name required", "error");
      return;
    }
    setCreating(true);
    try {
      const eventsList = events
        .split(",")
        .map((e) => e.trim())
        .filter(Boolean);
      const res = await api.createWebhook({
        name: name.trim(),
        description: description.trim() || undefined,
        events: eventsList.length ? eventsList : undefined,
        deliver,
        deliver_only: deliverOnly,
        prompt: prompt.trim() || undefined,
      });
      showToast("Created ✓", "success");
      setCreated({ url: res.url, secret: res.secret });
      resetForm();
      loadWebhooks();
    } catch (e) {
      showToast(`Failed to create: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const [togglingName, setTogglingName] = useState<string | null>(null);

  const handleToggleEnabled = useCallback(
    async (subName: string, nextEnabled: boolean) => {
      setTogglingName(subName);
      try {
        await api.setWebhookEnabled(subName, nextEnabled);
        showToast(
          nextEnabled ? `Enabled: "${subName}"` : `Disabled: "${subName}"`,
          "success",
        );
        loadWebhooks();
      } catch (e) {
        showToast(`Error: ${e}`, "error");
      } finally {
        setTogglingName(null);
      }
    },
    [loadWebhooks, showToast],
  );

  const webhookDelete = useConfirmDelete({
    onDelete: useCallback(
      async (name: string) => {
        try {
          await api.deleteWebhook(name);
          showToast(`Deleted: "${name}"`, "success");
          loadWebhooks();
        } catch (e) {
          showToast(`Error: ${e}`, "error");
          throw e;
        }
      },
      [loadWebhooks, showToast],
    ),
  });

  // Put "New subscription" button in page header
  useLayoutEffect(() => {
    setEnd(
      <Button
        className="uppercase"
        size="sm"
        disabled={!enabled || enabling}
        prefix={<Plus />}
        onClick={() => {
          setCreated(null);
          setCreateModalOpen(true);
        }}
      >
        New subscription
      </Button>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, enabled, enabling, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const pendingName = webhookDelete.pendingId ?? "";

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={webhookDelete.isOpen}
        onCancel={webhookDelete.cancel}
        onConfirm={webhookDelete.confirm}
        title="Delete webhook"
        description={
          pendingName
            ? `"${pendingName}" — this will permanently remove this webhook subscription.`
            : "This will permanently remove this webhook subscription."
        }
        loading={webhookDelete.isDeleting}
      />

      {/* Create subscription modal */}
      {createModalOpen && (
        <div
          ref={createModalRef}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
          onClick={(e) => e.target === e.currentTarget && closeCreateModal()}
          role="dialog"
          aria-modal="true"
          aria-labelledby="create-webhook-title"
        >
          <div className={cn(themedBody, "relative w-full max-w-lg border border-border bg-card shadow-2xl flex flex-col max-h-[90vh] overflow-y-auto")}>
            <Button
              ghost
              size="icon"
              onClick={closeCreateModal}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label="Close"
            >
              <X />
            </Button>

            <header className="p-5 pb-3 border-b border-border">
              <h2
                id="create-webhook-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                New subscription
              </h2>
            </header>

            {created ? (
              <div className="p-5 grid gap-4">
                <p className="text-sm text-muted-foreground">
                  Subscription created. Copy the secret now — it is only shown
                  once.
                </p>

                <div className="grid gap-2">
                  <Label>Webhook URL</Label>
                  <div className="flex items-center gap-2 border border-border bg-background/40 px-3 py-2">
                    <span className="flex-1 min-w-0 truncate font-mono text-xs">
                      {created.url}
                    </span>
                    <CopyButton value={created.url} />
                  </div>
                </div>

                <div className="grid gap-2">
                  <Label>Secret (shown once)</Label>
                  <div className="flex items-center gap-2 border border-warning/40 bg-warning/10 px-3 py-2">
                    <span className="flex-1 min-w-0 truncate font-mono text-xs">
                      {created.secret}
                    </span>
                    <CopyButton value={created.secret} />
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button
                    className="uppercase"
                    size="sm"
                    onClick={closeCreateModal}
                  >
                    Done
                  </Button>
                </div>
              </div>
            ) : (
              <div className="p-5 grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="webhook-name">Name</Label>
                  <Input
                    id="webhook-name"
                    autoFocus
                    placeholder="e.g. github-push"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="webhook-description">Description</Label>
                  <Input
                    id="webhook-description"
                    placeholder="What this webhook does (optional)"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="webhook-events">Events</Label>
                  <Input
                    id="webhook-events"
                    placeholder="comma-separated, leave empty for all"
                    value={events}
                    onChange={(e) => setEvents(e.target.value)}
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="webhook-deliver">Deliver to</Label>
                    <Select
                      id="webhook-deliver"
                      value={deliver}
                      onValueChange={(v) => setDeliver(v)}
                    >
                      <SelectOption value="log">Log</SelectOption>
                      <SelectOption value="telegram">Telegram</SelectOption>
                      <SelectOption value="discord">Discord</SelectOption>
                      <SelectOption value="slack">Slack</SelectOption>
                      <SelectOption value="email">Email</SelectOption>
                      <SelectOption value="github_comment">
                        GitHub comment
                      </SelectOption>
                    </Select>
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="webhook-deliver-only">Deliver only</Label>
                    <label className="flex items-center gap-2 text-sm text-muted-foreground h-9">
                      <input
                        id="webhook-deliver-only"
                        type="checkbox"
                        checked={deliverOnly}
                        onChange={(e) => setDeliverOnly(e.target.checked)}
                      />
                      Skip the agent, deliver payload directly
                    </label>
                  </div>
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="webhook-prompt">Prompt</Label>
                  <textarea
                    id="webhook-prompt"
                    className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 focus-visible:border-foreground/25"
                    placeholder="Instructions for the agent when this webhook fires (optional)"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>

                <div className="flex justify-end">
                  <Button
                    className="uppercase"
                    size="sm"
                    onClick={handleCreate}
                    disabled={creating}
                    prefix={creating ? <Spinner /> : undefined}
                  >
                    {creating ? t.webhooks.creating : t.webhooks.create}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {!enabled && (
        <Card className="border-warning/50">
          <CardContent className="flex flex-col gap-4 py-6 text-sm sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <Webhook className="h-5 w-5 shrink-0 text-warning" />
              <div className="flex flex-col gap-1">
                <span className="font-medium">Webhook receiver disabled</span>
                <span className="text-muted-foreground">
                  Webhooks are their own gateway platform. Enable them here to
                  accept incoming HTTP events; chat channels are only needed
                  when a subscription delivers to Telegram, Discord, Slack, or
                  another channel.
                </span>
              </div>
            </div>
            <Button
              size="sm"
              className="uppercase shrink-0"
              onClick={handleEnableWebhooks}
              disabled={enabling}
              prefix={enabling ? <Spinner /> : <Webhook className="h-4 w-4" />}
            >
              {enabling ? "Enabling…" : "Enable webhooks"}
            </Button>
          </CardContent>
        </Card>
      )}

      {restartMessage && !restartNeeded && (
        <Card className="border-border">
          <CardContent className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
            <RotateCw className="h-4 w-4 shrink-0 text-warning" />
            <span>{restartMessage}</span>
          </CardContent>
        </Card>
      )}

      {restartNeeded && (
        <Card className="border-warning/50">
          <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-2 text-sm">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
              <span>
                {restartError ??
                  "Webhooks are enabled, but the gateway still needs a restart before the receiver can come online."}
              </span>
            </div>
            <Button
              size="sm"
              className="uppercase shrink-0"
              onClick={handleRestart}
              disabled={restarting}
              prefix={restarting ? <Spinner /> : <RotateCw className="h-4 w-4" />}
            >
              {restarting ? "Restarting…" : "Restart gateway"}
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="flex flex-col gap-3">
        <H2
          variant="sm"
          className="flex items-center gap-2 text-muted-foreground"
        >
          <Webhook className="h-4 w-4" />
          Subscriptions ({subscriptions.length})
        </H2>

        <p className="text-xs text-muted-foreground -mt-1">
          Subscription changes hot-reload once the webhook receiver is running.
          Disabled subscriptions reject incoming events.
        </p>

        {subscriptions.length === 0 && (
          <Card>
            <CardContent className="py-8 text-center text-sm text-muted-foreground">
              No webhook subscriptions yet.
            </CardContent>
          </Card>
        )}

        {subscriptions.map((sub: WebhookRoute) => (
          <Card key={sub.name}>
            <CardContent className="flex items-start gap-4 py-4">
              <div className={cn("flex-1 min-w-0", !sub.enabled && "opacity-60")}>
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="font-medium text-sm truncate">
                    {sub.name}
                  </span>
                  <Badge tone="outline">{sub.deliver}</Badge>
                  {sub.deliver_only && (
                    <Badge tone="secondary">deliver only</Badge>
                  )}
                  {!sub.enabled && <Badge tone="warning">disabled</Badge>}
                </div>

                {sub.description && (
                  <p className="text-xs text-muted-foreground mb-2">
                    {sub.description}
                  </p>
                )}

                <div className="flex items-center gap-1 flex-wrap mb-2">
                  {sub.events.length === 0 ? (
                    <Badge tone="secondary">(all)</Badge>
                  ) : (
                    sub.events.map((evt) => (
                      <Badge key={evt} tone="secondary">
                        {evt}
                      </Badge>
                    ))
                  )}
                </div>

                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="flex-1 min-w-0 truncate font-mono">
                    {sub.url}
                  </span>
                  <CopyButton value={sub.url} />
                </div>
              </div>

              <div className="flex items-center gap-1 shrink-0">
                <Button
                  ghost
                  size="sm"
                  className="uppercase"
                  disabled={togglingName === sub.name}
                  onClick={() => handleToggleEnabled(sub.name, !sub.enabled)}
                >
                  {sub.enabled ? "Disable" : "Enable"}
                </Button>
                <Button
                  ghost
                  destructive
                  size="icon"
                  title={t.webhooks.delete}
                  aria-label={t.webhooks.delete}
                  onClick={() => webhookDelete.requestDelete(sub.name)}
                >
                  <Trash2 />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
