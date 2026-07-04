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
import { api, type PaymentRequest, type PaymentsResponse } from "@/lib/api";
import { PluginSlot } from "@/plugins";
import { cn } from "@/lib/utils";

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
});

const DATETIME_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

const STATUS_OPTIONS: Array<{ id: PaymentRequest["status"] | "all"; label: string }> = [
  { id: "all", label: "All" },
  { id: "new", label: "New" },
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

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const result = await api.getPayments();
      setData(result);
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
      if (statusFilter !== "all" && payment.status !== statusFilter) return false;
      if (needle && !searchText(payment).includes(needle)) return false;
      return true;
    });
  }, [data?.requests, query, sourceFilter, statusFilter]);

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
    async (status: PaymentRequest["status"]) => {
      if (!selectedPayment) return;
      setSavingStatus(status);
      try {
        const updated = await api.updatePaymentStatus(selectedPayment.id, status);
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
    [selectedPayment, showToast],
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

        <section className="grid gap-6 xl:grid-cols-[21rem_minmax(0,1fr)_22rem]">
          <Card className="min-h-[36rem] border-border/70 bg-background/80">
            <CardHeader className="space-y-4 pb-4">
              <div className="flex items-center justify-between gap-3">
                <CardTitle>Queue</CardTitle>
                <div className="flex items-center gap-2">
                  <Badge tone="outline">{filteredRequests.length}</Badge>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
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
                  variant={sourceFilter === "all" ? "default" : "secondary"}
                  onClick={() => setSourceFilter("all")}
                >
                  All sources
                </Button>
                {sourceFilters.map((source) => (
                  <Button
                    key={source.id}
                    type="button"
                    size="sm"
                    variant={sourceFilter === source.id ? "default" : "secondary"}
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
                    variant={statusFilter === status.id ? "default" : "secondary"}
                    onClick={() => setStatusFilter(status.id)}
                  >
                    {status.label}
                  </Button>
                ))}
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {loading ? (
                <div className="grid min-h-48 place-items-center">
                  <Spinner />
                </div>
              ) : filteredRequests.length === 0 ? (
                <div className="space-y-3 rounded-xl border border-dashed border-border/80 bg-muted/20 p-5 text-sm text-muted-foreground">
                  <div className="flex items-center gap-2 text-foreground">
                    <Wallet className="h-4 w-4" />
                    <span className="font-medium">No captured payment requests yet</span>
                  </div>
                  <p>
                    Connect Gmail or Email, or stage invoice files for extraction. This queue
                    will keep the manual-payment details once capture is wired in.
                  </p>
                  {data?.storage_path ? (
                    <p className="font-mono-ui text-xs text-muted-foreground/80">
                      Storage: {data.storage_path}
                    </p>
                  ) : null}
                </div>
              ) : (
                filteredRequests.map((payment) => (
                  <button
                    key={payment.id}
                    type="button"
                    onClick={() => setSelectedId(payment.id)}
                    className={cn(
                      "w-full rounded-xl border px-4 py-3 text-left transition-colors",
                      selectedPayment?.id === payment.id
                        ? "border-foreground/20 bg-foreground/5"
                        : "border-border/70 bg-background hover:bg-muted/25",
                    )}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate font-medium text-foreground">
                          {paymentTitle(payment)}
                        </p>
                        <p className="truncate text-sm text-muted-foreground">
                          {payment.title || payment.preview_text || "No source summary"}
                        </p>
                      </div>
                      <Badge tone={statusBadgeTone(payment.status)}>
                        {payment.status.replaceAll("_", " ")}
                      </Badge>
                    </div>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge tone="outline">{payment.source_label}</Badge>
                      <Badge tone={confidenceBadgeTone(payment.confidence)}>
                        {payment.confidence} confidence
                      </Badge>
                      <span>{amountLabel(payment)}</span>
                      <span>Due {formatDate(payment.due_date)}</span>
                    </div>
                  </button>
                ))
              )}
            </CardContent>
          </Card>

          <Card className="min-h-[36rem] border-border/70 bg-background/80">
            <CardHeader className="space-y-3 pb-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <CardTitle>{selectedPayment ? paymentTitle(selectedPayment) : "Invoice / Request"}</CardTitle>
                  <p className="mt-2 text-sm text-muted-foreground">
                    {selectedPayment
                      ? selectedPayment.title || "No subject captured yet."
                      : "Select a captured item to inspect the source and extracted fields."}
                  </p>
                </div>
                {selectedPayment ? (
                  <Badge tone={statusBadgeTone(selectedPayment.status)}>
                    {selectedPayment.status.replaceAll("_", " ")}
                  </Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-5">
              {selectedPayment ? (
                <>
                  <div className="grid gap-4 md:grid-cols-2">
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

                  <div className="grid gap-4 md:grid-cols-2">
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

                  <div className="flex flex-wrap gap-3">
                    <Button
                      type="button"
                      variant="secondary"
                      onClick={() => window.open(selectedPayment.original.url, "_blank", "noopener,noreferrer")}
                      disabled={!selectedPayment.original.url}
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      Open original
                    </Button>
                  </div>
                </>
              ) : (
                <div className="grid min-h-64 place-items-center text-sm text-muted-foreground">
                  Select a payment request from the queue.
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="min-h-[36rem] border-border/70 bg-background/80">
            <CardHeader className="space-y-3 pb-4">
              <CardTitle>Payment Details</CardTitle>
              <p className="text-sm text-muted-foreground">
                Copy these fields into your banking app. No payments are sent automatically.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedPayment ? (
                <>
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
                    label="Due date"
                    value={formatDate(selectedPayment.due_date)}
                    onCopy={() => copyValue("due date", formatDate(selectedPayment.due_date))}
                  />
                  <PaymentField label="Billing address" value={selectedPayment.billing_address} />
                  <PaymentField label="Tax details" value={selectedPayment.tax_details} />

                  <div className="space-y-2 pt-2">
                    <Button
                      type="button"
                      className="w-full"
                      onClick={() => void updateStatus("ready_to_pay")}
                      disabled={savingStatus !== null || selectedPayment.status === "ready_to_pay"}
                    >
                      {savingStatus === "ready_to_pay" ? <Spinner /> : "Mark ready to pay"}
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full"
                      onClick={() => void updateStatus("paid")}
                      disabled={savingStatus !== null || selectedPayment.status === "paid"}
                    >
                      {savingStatus === "paid" ? <Spinner /> : "Mark paid"}
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full"
                      onClick={() => void updateStatus("needs_review")}
                      disabled={savingStatus !== null || selectedPayment.status === "needs_review"}
                    >
                      {savingStatus === "needs_review" ? <Spinner /> : "Needs review"}
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      className="w-full"
                      onClick={() => void updateStatus("ignored")}
                      disabled={savingStatus !== null || selectedPayment.status === "ignored"}
                    >
                      {savingStatus === "ignored" ? <Spinner /> : "Not a payment"}
                    </Button>
                  </div>
                </>
              ) : (
                <div className="grid min-h-64 place-items-center text-sm text-muted-foreground">
                  Payment details will appear here.
                </div>
              )}
            </CardContent>
          </Card>
        </section>
      </div>

      <PluginSlot name="dashboard.payments.below" />

      {toast && <Toast message={toast.message} type={toast.type} onClose={toast.hide} />}
    </>
  );
}
