import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Clipboard, FileText, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { H2, Typography } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type {
  MissionControlBlockFlagPacketCreate,
  MissionControlPacket,
  MissionControlPacketSummary,
  MissionControlWorkerResultPacketCreate,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const panel = "rounded-xl border border-[#284848] bg-black/30 p-4";
const field =
  "w-full rounded-lg border border-[#284848] bg-black/45 p-3 text-sm text-text-primary outline-none focus:border-emerald-400/60";
const safetyBadge = "border-emerald-400/35 bg-emerald-500/10 text-emerald-100";
const warningBadge = "border-amber-400/40 bg-amber-500/10 text-amber-100";
const blockedBadge = "border-red-400/35 bg-red-500/10 text-red-100";

const EMPTY_CODEX_FORM = {
  project: "Hermes Ops",
  title: "",
  prompt: "",
  source_refs: "",
  author: "dashboard",
};

const EMPTY_WORKER_FORM = {
  project: "Hermes Ops",
  title: "",
  worker_result: "",
  source_refs: "",
  author: "dashboard",
};

type BlockFlagForm = Omit<MissionControlBlockFlagPacketCreate, "source_refs"> & {
  source_refs: string;
};

const EMPTY_BLOCK_FORM: BlockFlagForm = {
  project: "Hermes Ops",
  title: "",
  flag: "block_all_sends",
  reason: "",
  source_refs: "",
  author: "dashboard",
};

function splitRefs(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatKind(kind: string): string {
  return kind.replace(/_/g, " ");
}

function stringifyValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function packetCopyText(packet: MissionControlPacket): string {
  if (packet.kind === "codex_prompt") {
    return String(packet.payload.prompt || packet.redacted_payload_preview || "");
  }
  return packet.redacted_payload_preview || stringifyValue(packet.payload);
}

function SafetyPosture({ packet }: { packet?: MissionControlPacket | MissionControlPacketSummary }) {
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      <Badge tone="outline" className={cn("justify-center py-1.5", packet?.dry_run === true ? safetyBadge : blockedBadge)}>
        Dry-run=true
      </Badge>
      <Badge tone="outline" className={cn("justify-center py-1.5", packet?.review_required === true ? warningBadge : blockedBadge)}>
        Review required=true
      </Badge>
      <Badge tone="outline" className={cn("justify-center py-1.5", packet?.trusted_for_execution === false ? blockedBadge : warningBadge)}>
        Not trusted for execution
      </Badge>
    </div>
  );
}

function SourceList({ title, items }: { title: string; items: unknown[] }) {
  return (
    <div className={panel}>
      <div className="text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">{title}</div>
      <div className="mt-2 space-y-2 text-sm leading-6 text-text-secondary">
        {items.length ? (
          items.map((item, idx) => (
            <pre key={idx} className="whitespace-pre-wrap break-words rounded-lg border border-[#284848] bg-black/35 p-2 font-mono text-xs">
              {stringifyValue(item)}
            </pre>
          ))
        ) : (
          <div>None recorded.</div>
        )}
      </div>
    </div>
  );
}

export function MissionPacketsPanel() {
  const [packets, setPackets] = useState<MissionControlPacketSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedPacket, setSelectedPacket] = useState<MissionControlPacket | null>(null);
  const [listWarnings, setListWarnings] = useState<string[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [codexForm, setCodexForm] = useState(EMPTY_CODEX_FORM);
  const [workerForm, setWorkerForm] = useState(EMPTY_WORKER_FORM);
  const [blockForm, setBlockForm] = useState(EMPTY_BLOCK_FORM);

  const selectedSummary = useMemo(
    () => packets.find((packet) => packet.id === selectedId),
    [packets, selectedId],
  );

  const loadPackets = useCallback(async (preferredId?: string) => {
    const response = await api.listMissionControlPackets();
    setPackets(response.items);
    setListWarnings(response.warnings || []);
    const nextId = preferredId || response.items[0]?.id || "";
    setSelectedId(nextId);
    if (!nextId) {
      setSelectedPacket(null);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      loadPackets().catch((error) => {
        setMessage(error instanceof Error ? error.message : "Could not load Mission Packets");
      });
    }, 0);
    return () => window.clearTimeout(initial);
  }, [loadPackets]);

  useEffect(() => {
    if (!selectedId) return;
    api
      .getMissionControlPacket(selectedId)
      .then((response) => setSelectedPacket(response.packet))
      .catch((error) => setMessage(error instanceof Error ? error.message : "Could not load packet detail"));
  }, [selectedId]);

  const saveCodexPacket = async () => {
    setBusy("codex");
    setMessage(null);
    try {
      const response = await api.createMissionControlCodexPromptPacket({
        project: codexForm.project,
        title: codexForm.title,
        prompt: codexForm.prompt,
        source_refs: splitRefs(codexForm.source_refs),
        author: codexForm.author,
      });
      setCodexForm(EMPTY_CODEX_FORM);
      setMessage("Saved Codex prompt packet for review only.");
      await loadPackets(response.packet.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save prompt packet");
    } finally {
      setBusy(null);
    }
  };

  const importWorkerResult = async () => {
    setBusy("worker");
    setMessage(null);
    const payload: MissionControlWorkerResultPacketCreate = {
      project: workerForm.project,
      title: workerForm.title,
      worker_result: workerForm.worker_result,
      source_refs: splitRefs(workerForm.source_refs),
      author: workerForm.author,
    };
    try {
      const response = await api.createMissionControlWorkerResultPacket(payload);
      setWorkerForm(EMPTY_WORKER_FORM);
      setMessage("Imported worker result as inert display data.");
      await loadPackets(response.packet.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not import worker result");
    } finally {
      setBusy(null);
    }
  };

  const saveBlockFlag = async () => {
    setBusy("block");
    setMessage(null);
    try {
      const response = await api.createMissionControlBlockFlagPacket({
        project: blockForm.project,
        title: blockForm.title,
        flag: blockForm.flag,
        reason: blockForm.reason,
        source_refs: splitRefs(blockForm.source_refs),
        author: blockForm.author,
      });
      setBlockForm(EMPTY_BLOCK_FORM);
      setMessage("Saved advisory block-flag packet; advisory_only=true and not actively enforced.");
      await loadPackets(response.packet.id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save block-flag packet");
    } finally {
      setBusy(null);
    }
  };

  const copySavedPrompt = async () => {
    if (!selectedPacket) return;
    try {
      await navigator.clipboard.writeText(packetCopyText(selectedPacket));
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
  };

  return (
    <Card className="font-readable-ui border-[#264545] bg-[#071717]/90 shadow-[0_0_0_1px_rgba(47,214,161,0.04),0_18px_60px_rgba(0,0,0,0.28)]">
      <CardContent className="space-y-5 p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <FileText className="h-5 w-5" />
              <H2 className="text-xl">Mission Packets</H2>
            </div>
            <Typography className="mt-1 max-w-3xl text-sm leading-6 text-text-secondary">
              Local review packets for prompts, worker-result imports, and advisory block flags. Packet payloads stay inert untrusted data.
            </Typography>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="outline" className={safetyBadge}>{packets.length} packets</Badge>
            <Badge tone="outline" className={blockedBadge}>trusted_for_execution=false</Badge>
          </div>
        </div>

        <SafetyPosture packet={selectedPacket || selectedSummary} />

        {message && (
          <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
            {message}
          </div>
        )}

        {listWarnings.length > 0 && (
          <div className="flex items-start gap-2 rounded-xl border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>{listWarnings.join(" ")}</div>
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-[minmax(18rem,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-2">
            <div className="text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">Packet list</div>
            {packets.length ? (
              packets.map((packet) => (
                <button
                  type="button"
                  key={packet.id}
                  onClick={() => setSelectedId(packet.id)}
                  className={cn(
                    "block w-full rounded-xl border bg-black/25 p-3 text-left transition hover:border-emerald-400/40",
                    selectedId === packet.id ? "border-emerald-400/50" : "border-[#284848]",
                  )}
                >
                  <div className="flex min-w-0 items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="break-words text-sm font-semibold leading-5 text-text-primary">{packet.title}</div>
                      <div className="mt-1 text-xs leading-5 text-text-secondary">{packet.project} · {formatKind(packet.kind)}</div>
                    </div>
                    <Badge tone="outline" className="shrink-0 border-cyan-400/30 text-cyan-200">{packet.status}</Badge>
                  </div>
                  <div className="mt-2 line-clamp-2 text-xs leading-5 text-text-secondary">{packet.redacted_payload_preview}</div>
                </button>
              ))
            ) : (
              <div className={panel}>No Mission Control packets saved yet.</div>
            )}
          </div>

          <div className="space-y-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">Selected packet details</div>
              <Button ghost size="sm" onClick={copySavedPrompt} disabled={!selectedPacket} className="w-fit gap-2">
                <Clipboard className="h-4 w-4" />
                {copyState === "copied" ? "Copied prompt" : copyState === "failed" ? "Copy failed" : "Copy saved prompt"}
              </Button>
            </div>

            {selectedPacket ? (
              <div className="space-y-3">
                <div className={panel}>
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="text-lg font-semibold leading-6 text-text-primary">{selectedPacket.title}</div>
                      <div className="mt-1 text-sm leading-6 text-text-secondary">{selectedPacket.project} · {formatKind(selectedPacket.kind)} · {selectedPacket.status}</div>
                    </div>
                    <Badge tone="outline" className={blockedBadge}>review_required=true</Badge>
                  </div>
                  {selectedPacket.kind === "worker_result" && (
                    <div className="mt-3 rounded-lg border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
                      Imported worker text is untrusted display data. It is not executable.
                    </div>
                  )}
                  {selectedPacket.kind === "block_flag" && (
                    <div className="mt-3 rounded-lg border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
                      advisory_only=true; saved locally and not actively enforced by any state hook.
                    </div>
                  )}
                  <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-[#284848] bg-black/45 p-3 font-mono text-xs leading-5 text-text-secondary">
                    {selectedPacket.redacted_payload_preview}
                  </pre>
                </div>

                <div className="grid gap-3 lg:grid-cols-3">
                  <SourceList title="Warnings" items={selectedPacket.warnings || []} />
                  <SourceList title="Source refs" items={selectedPacket.source_refs || []} />
                  <SourceList title="Approval gates" items={selectedPacket.approval_gates || []} />
                </div>

                {selectedPacket.kind === "worker_result" && (
                  <div className={panel}>
                    <div className="text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">parsed_metadata</div>
                    <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-[#284848] bg-black/45 p-3 font-mono text-xs leading-5 text-text-secondary">
                      {stringifyValue(selectedPacket.payload.parsed_metadata)}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className={panel}>Select a packet to inspect its redacted_payload_preview, warnings, source refs, and approval gates.</div>
            )}
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-3">
          <div className={panel}>
            <div className="mb-3 flex items-center gap-2 text-text-primary">
              <ShieldCheck className="h-4 w-4 text-emerald-300" />
              <div className="font-semibold">Save Codex prompt packet</div>
            </div>
            <div className="space-y-3">
              <Label>Project</Label>
              <Input value={codexForm.project} onChange={(event) => setCodexForm({ ...codexForm, project: event.target.value })} />
              <Label>Title</Label>
              <Input value={codexForm.title} onChange={(event) => setCodexForm({ ...codexForm, title: event.target.value })} />
              <Label>Prompt</Label>
              <textarea value={codexForm.prompt} onChange={(event) => setCodexForm({ ...codexForm, prompt: event.target.value })} className={cn(field, "min-h-32")} />
              <Label>Source refs</Label>
              <textarea value={codexForm.source_refs} onChange={(event) => setCodexForm({ ...codexForm, source_refs: event.target.value })} className={cn(field, "min-h-20")} />
              <Button onClick={saveCodexPacket} disabled={busy === "codex" || !codexForm.title || !codexForm.prompt} className="w-full">
                Save prompt packet
              </Button>
            </div>
          </div>

          <div className={panel}>
            <div className="mb-3 font-semibold text-text-primary">Import worker result text</div>
            <div className="mb-3 rounded-lg border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
              Imported worker text is untrusted display data. It is not executable.
            </div>
            <div className="space-y-3">
              <Label>Project</Label>
              <Input value={workerForm.project} onChange={(event) => setWorkerForm({ ...workerForm, project: event.target.value })} />
              <Label>Title</Label>
              <Input value={workerForm.title} onChange={(event) => setWorkerForm({ ...workerForm, title: event.target.value })} />
              <Label>Worker result text</Label>
              <textarea value={workerForm.worker_result} onChange={(event) => setWorkerForm({ ...workerForm, worker_result: event.target.value })} className={cn(field, "min-h-32")} />
              <Label>Source refs</Label>
              <textarea value={workerForm.source_refs} onChange={(event) => setWorkerForm({ ...workerForm, source_refs: event.target.value })} className={cn(field, "min-h-20")} />
              <Button onClick={importWorkerResult} disabled={busy === "worker" || !workerForm.title || !workerForm.worker_result} className="w-full">
                Import as display data
              </Button>
            </div>
          </div>

          <div className={panel}>
            <div className="mb-3 font-semibold text-text-primary">Save advisory block-flag packet</div>
            <div className="mb-3 rounded-lg border border-amber-400/30 bg-amber-500/10 p-3 text-sm leading-6 text-amber-100">
              advisory_only=true; not actively enforced until a separate reviewed state hook exists.
            </div>
            <div className="space-y-3">
              <Label>Project</Label>
              <Input value={blockForm.project} onChange={(event) => setBlockForm({ ...blockForm, project: event.target.value })} />
              <Label>Title</Label>
              <Input value={blockForm.title} onChange={(event) => setBlockForm({ ...blockForm, title: event.target.value })} />
              <Label>Flag</Label>
              <select value={blockForm.flag} onChange={(event) => setBlockForm({ ...blockForm, flag: event.target.value as MissionControlBlockFlagPacketCreate["flag"] })} className={field}>
                <option value="pause_future_outreach">pause_future_outreach</option>
                <option value="block_all_sends">block_all_sends</option>
                <option value="pause_cron_triggered_outreach">pause_cron_triggered_outreach</option>
                <option value="disable_launch_actions">disable_launch_actions</option>
              </select>
              <Label>Reason</Label>
              <textarea value={blockForm.reason} onChange={(event) => setBlockForm({ ...blockForm, reason: event.target.value })} className={cn(field, "min-h-24")} />
              <Label>Source refs</Label>
              <textarea value={blockForm.source_refs} onChange={(event) => setBlockForm({ ...blockForm, source_refs: event.target.value })} className={cn(field, "min-h-20")} />
              <Button onClick={saveBlockFlag} disabled={busy === "block" || !blockForm.title || !blockForm.reason} className="w-full">
                Save advisory flag packet
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
