import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { webhookTestCurl } from "@hermes/shared";
import {
  AlertTriangle,
  Check,
  Copy,
  Plus,
  RotateCw,
  Send,
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
import type { WebhookRoute, WebhookSignature, WebhooksResponse } from "@/lib/api";
import { copyTextToClipboard } from "@/lib/clipboard";
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
import { errorMessage } from "@/lib/api-error";

interface CreatedWebhook {
  url: string;
  secret: string;
  signature: WebhookSignature | null;
}

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    void copyTextToClipboard(value).then((copied) => {
      if (!copied) return;
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  }, [value]);
  return (
    <Button
      ghost
      size="icon"
      title="Copy"
      aria-label="Copy"
      onClick={handleCopy}
      className="text-muted-foreground hover:text-foreground"
    >
      {copied ? <Check /> : <Copy />}
    </Button>
  );
}

// A copy-paste signed curl. Without the secret (shown once, at creation) it carries a placeholder.
function TestCommand({
  url,
  signature,
  secret,
}: {
  url: string;
  signature: WebhookSignature | null;
  secret?: string;
}) {
  const command = webhookTestCurl({ url, signature, secret });
  return (
    <div className="grid gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-muted-foreground">Test command</span>
        <CopyButton value={command} />
      </div>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all border border-border bg-background/40 px-3 py-2 font-mono text-[0.7rem] leading-relaxed">
        {command}
      </pre>
      <p className="text-xs text-muted-foreground">
        Paste into a terminal to send one signed event; each run gets its own delivery id.
        {secret
          ? null
          : " Replace the placeholder with the secret shown at creation, or use Send test."}
      </p>
    </div>
  );
}

const DELIVER_OPTIONS: ReadonlyArray<readonly [string, string]> = [
  ["log", "Log"],
  ["telegram", "Telegram"],
  ["discord", "Discord"],
  ["slack", "Slack"],
  ["email", "Email"],
  ["github_comment", "GitHub comment"],
];

const SIGNATURE_OPTIONS: ReadonlyArray<readonly [WebhookSignature, string]> = [
  ["generic", "Generic (timestamped)"],
  ["github", "GitHub"],
  ["gitlab", "GitLab"],
  ["linear", "Linear"],
];

function SignatureField({
  value,
  onChange,
}: {
  value: WebhookSignature;
  onChange: (value: WebhookSignature) => void;
}) {
  return (
    <div className="grid gap-2">
      <Label htmlFor="webhook-signature">Signature</Label>
      <Select
        id="webhook-signature"
        value={value}
        onValueChange={(v) => onChange(v as WebhookSignature)}
      >
        {SIGNATURE_OPTIONS.map(([scheme, label]) => (
          <SelectOption key={scheme} value={scheme}>{label}</SelectOption>
        ))}
      </Select>
      <p className="text-xs text-muted-foreground">
        The only signature scheme this route accepts. GitHub, GitLab and Linear
        deliveries need their own pin.
      </p>
    </div>
  );
}

interface SubscriptionCardProps {
  sub: WebhookRoute;
  toggling: boolean;
  showToast: (message: string, kind: "success" | "error") => void;
  onToggle: () => void;
  onDelete: () => void;
}

function SubscriptionCard({
  sub,
  toggling,
  showToast,
  onToggle,
  onDelete,
}: SubscriptionCardProps) {
  const { testing, sendTest } = useSendTest(sub.name, showToast);
  return (
    <Card>
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

          <details className="mt-2">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              Test this webhook
            </summary>
            <div className="mt-2">
              <TestCommand url={sub.url} signature={sub.signature} />
            </div>
          </details>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <Button
            ghost
            size="sm"
            className="uppercase"
            disabled={!sub.enabled || testing}
            prefix={testing ? <Spinner /> : <Send className="h-4 w-4" />}
            onClick={() => void sendTest()}
          >
            {testing ? "Sending…" : "Send test"}
          </Button>
          <Button
            ghost
            size="sm"
            className="uppercase"
            disabled={toggling}
            onClick={onToggle}
          >
            {sub.enabled ? "Disable" : "Enable"}
          </Button>
          <Button
            ghost
            destructive
            size="icon"
            title="Delete"
            aria-label="Delete"
            onClick={onDelete}
          >
            <Trash2 />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// The create result: URL, one-time secret and a test curl that already carries that secret.
function CreatedResult({ created, onDone }: { created: CreatedWebhook; onDone: () => void }) {
  return (
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

      <TestCommand
        url={created.url}
        signature={created.signature}
        secret={created.secret}
      />

      <div className="flex justify-end">
        <Button
          className="uppercase"
          size="sm"
          onClick={onDone}
        >
          Done
        </Button>
      </div>
    </div>
  );
}

// Send test, owned by one card: the backend signs with the stored secret and reports the
// receiver's answer. Each card keeps its own pending state, so two tests never share a spinner.
function useSendTest(subName: string, showToast: (message: string, kind: "success" | "error") => void) {
  const [testing, setTesting] = useState(false);
  const sendTest = useCallback(async () => {
    setTesting(true);
    try {
      const result = await api.sendWebhookTest(subName);
      showToast(
        result.ok
          ? `Test delivered (${result.status}) · ${result.delivery_id}`
          : result.outcome === "ignored"
            ? "Delivered, but the subscription ignored it (its event or filter rules drop a test event)"
            : `Receiver answered ${result.status}: ${result.response}`,
        result.ok ? "success" : "error",
      );
    } catch (e) {
      showToast(`Test could not be sent: ${errorMessage(e)}`, "error");
    } finally {
      setTesting(false);
    }
  }, [subName, showToast]);
  return { testing, sendTest };
}

export default function WebhooksPage() {
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
  const [signature, setSignature] = useState<WebhookSignature>("generic");
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
      setRestartError(errorMessage(e));
      showToast(`Failed to restart: ${errorMessage(e)}`, "error");
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
      showToast(`Failed to enable webhooks: ${errorMessage(e)}`, "error");
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
    setSignature("generic");
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
        signature,
        prompt: prompt.trim() || undefined,
      });
      showToast("Created ✓", "success");
      setCreated({ url: res.url, secret: res.secret, signature: res.signature });
      resetForm();
      loadWebhooks();
    } catch (e) {
      showToast(`Failed to create: ${errorMessage(e)}`, "error");
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
        showToast(`Error: ${errorMessage(e)}`, "error");
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
          showToast(`Error: ${errorMessage(e)}`, "error");
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
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
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
              <CreatedResult created={created} onDone={closeCreateModal} />
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
                      {DELIVER_OPTIONS.map(([value, label]) => (
                        <SelectOption key={value} value={value}>{label}</SelectOption>
                      ))}
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

                <SignatureField value={signature} onChange={setSignature} />

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
                    {creating ? "Creating…" : "Create"}
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
          <SubscriptionCard
            key={sub.name}
            sub={sub}
            toggling={togglingName === sub.name}
            showToast={showToast}
            onToggle={() => handleToggleEnabled(sub.name, !sub.enabled)}
            onDelete={() => webhookDelete.requestDelete(sub.name)}
          />
        ))}
      </div>
    </div>
  );
}
