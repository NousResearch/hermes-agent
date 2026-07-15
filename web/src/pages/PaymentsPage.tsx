import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  ExternalLink,
  Mail,
  RefreshCw,
  Wallet,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import {
  api,
  type PaymentRequest,
  type PaymentsResponse,
  type PaymentShadowReportResponse,
} from "@/lib/api";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
});

const DATETIME_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const BOARD_COLUMNS: Array<{
  id: "needs_review" | "ready_to_pay" | "paid" | "ignored";
  label: string;
  empty: string;
}> = [
  { id: "needs_review", label: "Needs review", empty: "Nothing waiting for review." },
  { id: "ready_to_pay", label: "Ready to pay", empty: "Nothing queued for payment." },
  { id: "paid", label: "Paid", empty: "No recently settled items." },
  { id: "ignored", label: "Ignored", empty: "No parked false positives." },
];

const STATUS_OPTIONS: Array<{ id: PaymentRequest["status"] | "all"; label: string }> = [
  { id: "all", label: "All" },
  { id: "needs_review", label: "Needs review" },
  { id: "ready_to_pay", label: "Ready to pay" },
  { id: "paid", label: "Paid" },
  { id: "ignored", label: "Ignored" },
];

function statusBadgeTone(status: PaymentRequest["status"]) {
  if (status === "paid") return "success";
  if (status === "ready_to_pay") return "warning";
  if (status === "ignored") return "secondary";
  return "outline";
}

function confidenceBadgeTone(confidence: string) {
  if (confidence === "high") return "success";
  if (confidence === "medium") return "warning";
  return "outline";
}

function amountLabel(payment: PaymentRequest): string {
  return payment.amount.display || "Amount missing";
}

function formatDate(value: string | null): string {
  if (!value) return "Missing";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return DATE_FORMAT.format(date);
}

function formatDateTime(value: string | null): string {
  if (!value) return "Missing";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return DATETIME_FORMAT.format(date);
}

function paymentTitle(payment: PaymentRequest): string {
  return payment.vendor || payment.title || "Untitled payment request";
}

function searchText(payment: PaymentRequest): string {
  return [
    payment.vendor,
    payment.title,
    payment.preview_text,
    payment.invoice_number,
    payment.payment_reference,
    payment.original.label,
  ]
    .join(" ")
    .toLowerCase();
}

function operatorStatus(
  payment: PaymentRequest,
): "needs_review" | "ready_to_pay" | "paid" | "ignored" {
  return payment.operator_status || (payment.status === "new" ? "needs_review" : payment.status);
}

function fileDownloadUrl(path: string | undefined): string {
  return path ? `/api/files/download?path=${encodeURIComponent(path)}` : "";
}

function dueSoon(payment: PaymentRequest): boolean {
  if (!payment.due_date) return false;
  const due = new Date(payment.due_date);
  if (Number.isNaN(due.getTime())) return false;
  const now = new Date();
  const diffDays = (due.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);
  return diffDays <= 7;
}

function sortPayments(payments: PaymentRequest[]): PaymentRequest[] {
  return [...payments].sort((a, b) => {
    const aDue = a.due_date ? new Date(a.due_date).getTime() : Number.POSITIVE_INFINITY;
    const bDue = b.due_date ? new Date(b.due_date).getTime() : Number.POSITIVE_INFINITY;
    if (aDue !== bDue) return aDue - bDue;
    const aReceived = a.received_at ? new Date(a.received_at).getTime() : 0;
    const bReceived = b.received_at ? new Date(b.received_at).getTime() : 0;
    return bReceived - aReceived;
  });
}

function PaymentField({
  label,
  value,
  onCopy,
}: {
  label: string;
  value: string;
  onCopy?: () => void;
}) {
  const isMissing = !value.trim() || value === "Missing";
  return (
    <div className="space-y-1 border-b border-border/60 pb-3 last:border-b-0 last:pb-0">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          {label}
        </span>
        {onCopy ? (
          <Button
            ghost
            size="icon"
            type="button"
            onClick={() => void onCopy()}
            aria-label={`Copy ${label}`}
            disabled={isMissing}
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
        ) : null}
      </div>
      <p className={cn("text-sm text-foreground", isMissing && "italic text-muted-foreground")}>
        {isMissing ? "Missing" : value}
      </p>
    </div>
  );
}

function PaymentCard({
  payment,
  selected,
  onSelect,
  onStatusChange,
  savingStatus,
}: {
  payment: PaymentRequest;
  selected: boolean;
  onSelect: () => void;
  onStatusChange: (status: PaymentRequest["status"]) => void;
  savingStatus: string | null;
}) {
  const activeStatus = operatorStatus(payment);
  const busy = savingStatus !== null;
  return (
    <div
      className={cn(
        "rounded-2xl border p-4 transition-colors",
        selected ? "border-foreground/20 bg-foreground/5" : "border-border/70 bg-background/90",
      )}
    >
      <button type="button" onClick={onSelect} className="w-full text-left">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate font-medium text-foreground">{paymentTitle(payment)}</p>
            <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
              {payment.title || payment.preview_text || "No captured summary"}
            </p>
          </div>
          <Badge tone={statusBadgeTone(activeStatus)}>
            {activeStatus.replaceAll("_", " ")}
          </Badge>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge tone={confidenceBadgeTone(payment.confidence)}>
            {payment.confidence} confidence
          </Badge>
          {payment.looks_paid ? <Badge tone="secondary">Receipt-like</Badge> : null}
          {dueSoon(payment) ? <Badge tone="warning">Due soon</Badge> : null}
          <span>{amountLabel(payment)}</span>
          <span>Due {formatDate(payment.due_date)}</span>
        </div>
      </button>
      <div className="mt-4 grid grid-cols-2 gap-2">
        <Button
          type="button"
          size="sm"
          outlined={activeStatus !== "ready_to_pay"}
          disabled={busy || activeStatus === "ready_to_pay"}
          onClick={() => void onStatusChange("ready_to_pay")}
        >
          {savingStatus === "ready_to_pay" ? <Spinner /> : "Ready"}
        </Button>
        <Button
          type="button"
          size="sm"
          outlined={activeStatus !== "paid"}
          disabled={busy || activeStatus === "paid"}
          onClick={() => void onStatusChange("paid")}
        >
          {savingStatus === "paid" ? <Spinner /> : "Paid"}
        </Button>
        <Button
          type="button"
          size="sm"
          outlined={activeStatus !== "needs_review"}
          disabled={busy || activeStatus === "needs_review"}
          onClick={() => void onStatusChange("needs_review")}
        >
          {savingStatus === "needs_review" ? <Spinner /> : "Review"}
        </Button>
        <Button
          type="button"
          size="sm"
          outlined={activeStatus !== "ignored"}
          disabled={busy || activeStatus === "ignored"}
          onClick={() => void onStatusChange("ignored")}
        >
          {savingStatus === "ignored" ? <Spinner /> : "Ignore"}
        </Button>
      </div>
    </div>
  );
}

export default function PaymentsPage() {
  const { toast, showToast } = useToast();
  const { setEnd } = usePageHeader();
  const [data, setData] = useState<PaymentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState<PaymentRequest["status"] | "all">("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [savingStatus, setSavingStatus] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [shadowReport, setShadowReport] = useState<PaymentShadowReportResponse | null>(null);
  const [shadowSyncing, setShadowSyncing] = useState(false);
  const [shadowScheduling, setShadowScheduling] = useState(false);
  const [dueSoonOnly, setDueSoonOnly] = useState(false);
  const [likelyPaidOnly, setLikelyPaidOnly] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [paymentsResult, shadowResult] = await Promise.all([
        api.getPayments(),
        api.getPaymentsShadowReport().catch(() => null),
      ]);
      setData(paymentsResult);
      setShadowReport(shadowResult);
    } catch (error) {
      showToast(`Failed to load payments: ${error}`, "error");
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    setEnd(
      <Button
        ghost
        size="icon"
        type="button"
        onClick={() => void load()}
        disabled={loading}
        aria-label="Refresh payments"
      >
        {loading ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    return () => setEnd(null);
  }, [load, loading, setEnd]);

  const filteredRequests = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return (data?.requests || []).filter((payment) => {
      if (sourceFilter !== "all" && payment.source !== sourceFilter) return false;
      if (statusFilter !== "all" && operatorStatus(payment) !== statusFilter) return false;
      if (dueSoonOnly && !dueSoon(payment)) return false;
      if (likelyPaidOnly && !payment.looks_paid) return false;
      if (needle && !searchText(payment).includes(needle)) return false;
      return true;
    });
  }, [data?.requests, dueSoonOnly, likelyPaidOnly, query, sourceFilter, statusFilter]);

  useEffect(() => {
    if (!filteredRequests.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !filteredRequests.some((payment) => payment.id === selectedId)) {
      setSelectedId(filteredRequests[0].id);
    }
  }, [filteredRequests, selectedId]);

  const selectedPayment =
    filteredRequests.find((payment) => payment.id === selectedId) || filteredRequests[0] || null;

  const grouped = useMemo(() => {
    const buckets = new Map(BOARD_COLUMNS.map((column) => [column.id, [] as PaymentRequest[]]));
    for (const payment of filteredRequests) {
      const lane = operatorStatus(payment);
      if (buckets.has(lane)) buckets.get(lane)!.push(payment);
    }
    for (const [key, items] of buckets) {
      buckets.set(key, sortPayments(items));
    }
    return buckets;
  }, [filteredRequests]);

  const copyValue = useCallback(
    async (label: string, value: string) => {
      if (!value.trim()) return;
      try {
        await navigator.clipboard.writeText(value);
        showToast(`${label} copied`, "success");
      } catch (error) {
        showToast(`Failed to copy ${label}: ${error}`, "error");
      }
    },
    [showToast],
  );

  const updateStatus = useCallback(
    async (paymentId: string, status: PaymentRequest["status"]) => {
      setSavingStatus(`${paymentId}:${status}`);
      try {
        const updated = await api.updatePaymentStatus(paymentId, status);
        setData((prev) =>
          prev
            ? {
                ...prev,
                requests: prev.requests.map((item) =>
                  item.id === updated.id ? updated : item,
                ),
              }
            : prev,
        );
        showToast(`Payment marked ${status.replaceAll("_", " ")}`, "success");
      } catch (error) {
        showToast(`Failed to update payment: ${error}`, "error");
      } finally {
        setSavingStatus(null);
      }
    },
    [showToast],
  );

  const sourceFilters = data?.sources || [];
  const gmailSource = sourceFilters.find((source) => source.id === "gmail");

  const syncGmail = useCallback(async () => {
    setSyncing(true);
    try {
      const result = await api.syncPayments({ source: "gmail" });
      showToast(
        `Gmail sync complete: ${result.imported} imported, ${result.updated} updated`,
        "success",
      );
      await load();
    } catch (error) {
      showToast(`Failed to sync Gmail: ${error}`, "error");
    } finally {
      setSyncing(false);
    }
  }, [load, showToast]);

  const shadowSyncGmail = useCallback(async () => {
    setShadowSyncing(true);
    try {
      const result = await api.shadowSyncPayments({ source: "gmail" });
      showToast(
        `Shadow sync complete: ${result.shadow.mirrored} mirrored, parity ${result.shadow.parity.parity_ok ? "ok" : "needs review"}`,
        "success",
      );
      await load();
    } catch (error) {
      showToast(`Failed to run shadow sync: ${error}`, "error");
    } finally {
      setShadowSyncing(false);
    }
  }, [load, showToast]);

  const scheduleShadowSync = useCallback(async () => {
    setShadowScheduling(true);
    try {
      const result = await api.schedulePaymentsShadowSync({
        schedule: "every 6h",
        run_now: false,
      });
      showToast(
        `Shadow sync schedule updated: ${result.job.schedule_display || result.job.schedule || "configured"}`,
        "success",
      );
      await load();
    } catch (error) {
      showToast(`Failed to schedule shadow sync: ${error}`, "error");
    } finally {
      setShadowScheduling(false);
    }
  }, [load, showToast]);

  const shadowParity = shadowReport?.parity || null;
  const shadowSnapshot = shadowReport?.snapshot || null;
  const shadowIssueCount = shadowParity
    ? shadowParity.status_mismatches.length +
      shadowParity.amount_mismatches.length +
      shadowParity.due_date_mismatches.length +
      shadowParity.reference_mismatches.length +
      shadowParity.missing_in_shadow.length +
      shadowParity.shadow_only.length
    : 0;

  return (
    <>
      <div className="space-y-6">
        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {sourceFilters.map((source) => (
            <Card key={source.id} className="border-border/70 bg-background/70">
              <CardContent className="flex items-start justify-between gap-4 p-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-foreground">{source.label}</span>
                    <Badge tone={source.connected ? "success" : "outline"}>
                      {source.connected ? "Connected" : "Not connected"}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{source.detail}</p>
                </div>
                <Mail className="mt-0.5 h-4 w-4 text-muted-foreground" />
              </CardContent>
            </Card>
          ))}
        </section>

        <Card className="border-border/70 bg-background/80">
          <CardHeader className="space-y-4 pb-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle>Shadow parity</CardTitle>
                <p className="mt-2 text-sm text-muted-foreground">
                  Mirror the legacy payments queue into <code>inbox_items</code> and compare parity before cutover.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge tone={shadowParity?.parity_ok ? "success" : shadowParity ? "warning" : "outline"}>
                  {shadowParity ? (shadowParity.parity_ok ? "Parity ok" : "Mismatches detected") : "Unavailable"}
                </Badge>
                <Button
                  type="button"
                  size="sm"
                  outlined
                  onClick={() => void shadowSyncGmail()}
                  disabled={shadowSyncing || gmailSource?.connected === false}
                >
                  {shadowSyncing ? <Spinner /> : "Shadow sync"}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  outlined
                  onClick={() => void scheduleShadowSync()}
                  disabled={shadowScheduling}
                >
                  {shadowScheduling ? <Spinner /> : "Schedule"}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-border/70 bg-background p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Counts</p>
              <p className="mt-3 text-sm text-foreground">
                {shadowParity
                  ? `${shadowParity.payments_count} payments · ${shadowParity.shadow_count} shadow · ${shadowParity.compared_count} compared`
                  : "No parity data available"}
              </p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Mismatch count</p>
              <p className="mt-3 text-sm text-foreground">{shadowParity ? shadowIssueCount : "Unavailable"}</p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Last snapshot</p>
              <p className="mt-3 text-sm text-foreground">
                {shadowSnapshot?.updated_at ? formatDateTime(shadowSnapshot.updated_at) : "No snapshot yet"}
              </p>
            </div>
            <div className="rounded-xl border border-border/70 bg-background p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Storage</p>
              <p className="mt-3 break-all font-mono-ui text-xs text-muted-foreground">
                {shadowSnapshot?.path || shadowReport?.storage_path || "Unavailable"}
              </p>
            </div>
            {shadowParity && !shadowParity.parity_ok ? (
              <div className="md:col-span-2 xl:col-span-4 rounded-xl border border-warning/40 bg-warning/5 p-4 text-sm text-foreground">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                  <div>
                    <p className="font-medium">Parity issues need review before cutover.</p>
                    <p className="mt-2 text-muted-foreground">
                      Status mismatches: {shadowParity.status_mismatches.length} · Amount mismatches: {shadowParity.amount_mismatches.length} · Due date mismatches: {shadowParity.due_date_mismatches.length} · Reference mismatches: {shadowParity.reference_mismatches.length} · Missing in shadow: {shadowParity.missing_in_shadow.length}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <section className="grid gap-6 2xl:grid-cols-[minmax(0,1fr)_25rem]">
          <div className="space-y-6">
            <Card className="border-border/70 bg-background/80">
              <CardHeader className="space-y-4 pb-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <CardTitle>Payments board</CardTitle>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Manual review queue backed by the canonical payments review store.
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge tone="outline">{filteredRequests.length}</Badge>
                    <Button
                      type="button"
                      size="sm"
                      outlined
                      onClick={() => void syncGmail()}
                      disabled={syncing || gmailSource?.connected === false}
                    >
                      {syncing ? <Spinner /> : "Sync Gmail"}
                    </Button>
                  </div>
                </div>
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search vendor, invoice, reference..."
                />
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    outlined={sourceFilter !== "all"}
                    onClick={() => setSourceFilter("all")}
                  >
                    All sources
                  </Button>
                  {sourceFilters.map((source) => (
                    <Button
                      key={source.id}
                      type="button"
                      size="sm"
                      outlined={sourceFilter !== source.id}
                      onClick={() => setSourceFilter(source.id)}
                    >
                      {source.label}
                    </Button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {STATUS_OPTIONS.map((status) => (
                    <Button
                      key={status.id}
                      type="button"
                      size="sm"
                      outlined={statusFilter !== status.id}
                      onClick={() => setStatusFilter(status.id)}
                    >
                      {status.label}
                    </Button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    outlined={!dueSoonOnly}
                    onClick={() => setDueSoonOnly((value) => !value)}
                  >
                    Due soon
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    outlined={!likelyPaidOnly}
                    onClick={() => setLikelyPaidOnly((value) => !value)}
                  >
                    Receipt-like
                  </Button>
                </div>
              </CardHeader>
            </Card>

            {loading ? (
              <Card className="grid min-h-72 place-items-center border-border/70 bg-background/80">
                <Spinner />
              </Card>
            ) : filteredRequests.length === 0 ? (
              <Card className="border-border/70 bg-background/80">
                <CardContent className="space-y-3 p-5 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2 text-foreground">
                    <Wallet className="h-4 w-4" />
                    <span className="font-medium">No captured payment requests match this view</span>
                  </div>
                  <p>
                    Adjust filters or sync Gmail to refresh the review queue.
                  </p>
                  {data?.storage_path ? (
                    <p className="font-mono-ui text-xs text-muted-foreground/80">
                      Storage: {data.storage_path}
                    </p>
                  ) : null}
                </CardContent>
              </Card>
            ) : (
              <section className="grid gap-4 lg:grid-cols-2 2xl:grid-cols-4">
                {BOARD_COLUMNS.map((column) => {
                  const items = grouped.get(column.id) || [];
                  return (
                    <Card key={column.id} className="border-border/70 bg-background/80">
                      <CardHeader className="pb-4">
                        <div className="flex items-center justify-between gap-3">
                          <CardTitle className="text-base">{column.label}</CardTitle>
                          <Badge tone="outline">{items.length}</Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {items.length === 0 ? (
                          <p className="rounded-xl border border-dashed border-border/70 bg-muted/20 p-4 text-sm text-muted-foreground">
                            {column.empty}
                          </p>
                        ) : (
                          items.map((payment) => (
                            <PaymentCard
                              key={payment.id}
                              payment={payment}
                              selected={selectedPayment?.id === payment.id}
                              onSelect={() => setSelectedId(payment.id)}
                              onStatusChange={(status) => updateStatus(payment.id, status)}
                              savingStatus={
                                savingStatus?.startsWith(`${payment.id}:`)
                                  ? savingStatus.split(":", 2)[1]
                                  : null
                              }
                            />
                          ))
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </section>
            )}
          </div>

          <Card className="h-fit border-border/70 bg-background/80 2xl:sticky 2xl:top-6">
            <CardHeader className="space-y-3 pb-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <CardTitle>
                    {selectedPayment ? paymentTitle(selectedPayment) : "Payment details"}
                  </CardTitle>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {selectedPayment
                      ? selectedPayment.title || "No subject captured yet."
                      : "Select a payment card to inspect the source and extracted fields."}
                  </p>
                </div>
                {selectedPayment ? (
                  <Badge tone={statusBadgeTone(operatorStatus(selectedPayment))}>
                    {operatorStatus(selectedPayment).replaceAll("_", " ")}
                  </Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-5">
              {selectedPayment ? (
                <>
                  <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-1">
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Source</p>
                      <p className="text-sm text-foreground">{selectedPayment.source_label}</p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">From</p>
                      <p className="text-sm text-foreground">
                        {selectedPayment.original.label || selectedPayment.vendor || "Missing"}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Received</p>
                      <p className="text-sm text-foreground">{formatDateTime(selectedPayment.received_at)}</p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Amount</p>
                      <p className="text-sm text-foreground">{amountLabel(selectedPayment)}</p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Due date</p>
                      <p className="text-sm text-foreground">{formatDate(selectedPayment.due_date)}</p>
                    </div>
                  </div>

                  <div className="rounded-xl border border-border/70 bg-muted/20 p-4">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-warning" />
                      <p className="font-medium text-foreground">Review summary</p>
                    </div>
                    <p className="mt-3 whitespace-pre-wrap text-sm text-muted-foreground">
                      {selectedPayment.preview_text || "No preview text has been captured yet."}
                    </p>
                  </div>

                  <div className="grid gap-4">
                    <div className="rounded-xl border border-border/70 bg-background p-4">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Attachments
                      </p>
                      {selectedPayment.attachments.length ? (
                        <ul className="mt-3 space-y-2 text-sm text-foreground">
                          {selectedPayment.attachments.map((attachment) => (
                            <li key={attachment}>{attachment}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="mt-3 text-sm italic text-muted-foreground">No attachments recorded</p>
                      )}
                    </div>
                    <div className="rounded-xl border border-border/70 bg-background p-4">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Review note
                      </p>
                      <p className="mt-3 text-sm text-foreground">
                        {selectedPayment.review_note || "No manual note recorded"}
                      </p>
                    </div>
                    <div className="rounded-xl border border-border/70 bg-background p-4">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
                        Warnings
                      </p>
                      {selectedPayment.warnings.length ? (
                        <ul className="mt-3 space-y-2 text-sm text-foreground">
                          {selectedPayment.warnings.map((warning) => (
                            <li key={warning} className="flex gap-2">
                              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
                              <span>{warning}</span>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="mt-3 text-sm italic text-muted-foreground">
                          No extraction warnings
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="grid gap-3">
                    <Button
                      type="button"
                      outlined
                      onClick={() => window.open(selectedPayment.original.url, "_blank", "noopener,noreferrer")}
                      disabled={!selectedPayment.original.url}
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      Open original
                    </Button>
                    <Button
                      type="button"
                      outlined
                      onClick={() =>
                        window.open(fileDownloadUrl(selectedPayment.raw_text_path), "_blank", "noopener,noreferrer")
                      }
                      disabled={!selectedPayment.raw_text_path}
                    >
                      Download raw artifact
                    </Button>
                    <Button
                      type="button"
                      outlined
                      onClick={() =>
                        window.open(
                          fileDownloadUrl(selectedPayment.materialized_path),
                          "_blank",
                          "noopener,noreferrer",
                        )
                      }
                      disabled={!selectedPayment.materialized_path}
                    >
                      Download materialized artifact
                    </Button>
                    <Button
                      type="button"
                      outlined
                      onClick={() => void updateStatus(selectedPayment.id, "ready_to_pay")}
                      disabled={savingStatus !== null || operatorStatus(selectedPayment) === "ready_to_pay"}
                    >
                      {savingStatus === `${selectedPayment.id}:ready_to_pay` ? <Spinner /> : "Mark ready to pay"}
                    </Button>
                    <Button
                      type="button"
                      className="w-full"
                      onClick={() => void updateStatus(selectedPayment.id, "paid")}
                      disabled={savingStatus !== null || operatorStatus(selectedPayment) === "paid"}
                    >
                      {savingStatus === `${selectedPayment.id}:paid` ? <Spinner /> : "Mark paid"}
                    </Button>
                    <Button
                      type="button"
                      outlined
                      onClick={() => void updateStatus(selectedPayment.id, "needs_review")}
                      disabled={savingStatus !== null || operatorStatus(selectedPayment) === "needs_review"}
                    >
                      {savingStatus === `${selectedPayment.id}:needs_review` ? <Spinner /> : "Needs review"}
                    </Button>
                    <Button
                      type="button"
                      outlined
                      onClick={() => void updateStatus(selectedPayment.id, "ignored")}
                      disabled={savingStatus !== null || operatorStatus(selectedPayment) === "ignored"}
                    >
                      {savingStatus === `${selectedPayment.id}:ignored` ? <Spinner /> : "Ignore / not a payment"}
                    </Button>
                  </div>

                  <div className="space-y-4 rounded-xl border border-border/70 bg-background p-4">
                    <p className="text-sm text-muted-foreground">
                      Copy these fields into your banking app. No payments are sent automatically.
                    </p>
                    <PaymentField
                      label="Payee"
                      value={selectedPayment.payee_name}
                      onCopy={() => copyValue("payee", selectedPayment.payee_name)}
                    />
                    <PaymentField
                      label="Account holder"
                      value={selectedPayment.account_holder}
                      onCopy={() => copyValue("account holder", selectedPayment.account_holder)}
                    />
                    <PaymentField
                      label="Account number"
                      value={selectedPayment.account_number}
                      onCopy={() => copyValue("account number", selectedPayment.account_number)}
                    />
                    <PaymentField
                      label="Sort code"
                      value={selectedPayment.sort_code}
                      onCopy={() => copyValue("sort code", selectedPayment.sort_code)}
                    />
                    <PaymentField
                      label="IBAN"
                      value={selectedPayment.iban}
                      onCopy={() => copyValue("IBAN", selectedPayment.iban)}
                    />
                    <PaymentField
                      label="SWIFT"
                      value={selectedPayment.swift}
                      onCopy={() => copyValue("SWIFT", selectedPayment.swift)}
                    />
                    <PaymentField
                      label="Routing number"
                      value={selectedPayment.routing_number}
                      onCopy={() => copyValue("routing number", selectedPayment.routing_number)}
                    />
                    <PaymentField
                      label="Amount"
                      value={amountLabel(selectedPayment)}
                      onCopy={() => copyValue("amount", amountLabel(selectedPayment))}
                    />
                    <PaymentField
                      label="Reference"
                      value={selectedPayment.payment_reference}
                      onCopy={() => copyValue("reference", selectedPayment.payment_reference)}
                    />
                    <PaymentField
                      label="Invoice number"
                      value={selectedPayment.invoice_number}
                      onCopy={() => copyValue("invoice number", selectedPayment.invoice_number)}
                    />
                    <PaymentField
                      label="Raw artifact path"
                      value={selectedPayment.raw_text_path || ""}
                      onCopy={() => copyValue("raw artifact path", selectedPayment.raw_text_path || "")}
                    />
                    <PaymentField
                      label="Materialized artifact path"
                      value={selectedPayment.materialized_path || ""}
                      onCopy={() =>
                        copyValue("materialized artifact path", selectedPayment.materialized_path || "")
                      }
                    />
                    <PaymentField
                      label="Due date"
                      value={formatDate(selectedPayment.due_date)}
                      onCopy={() => copyValue("due date", formatDate(selectedPayment.due_date))}
                    />
                    <PaymentField label="Billing address" value={selectedPayment.billing_address} />
                    <PaymentField label="Tax details" value={selectedPayment.tax_details} />
                  </div>
                </>
              ) : (
                <div className="grid min-h-64 place-items-center text-sm text-muted-foreground">
                  Select a payment card from the board.
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>

      <PluginSlot name="dashboard.payments.below" />

      <Toast toast={toast} />
    </>
  );
}
