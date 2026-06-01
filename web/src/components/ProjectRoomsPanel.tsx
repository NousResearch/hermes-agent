import { useCallback, useEffect, useMemo, useState } from "react";
import { Clipboard, FileUp, MessageSquare, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { H2, Typography } from "@/components/NouiTypography";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { ProjectRoom, ProjectRoomMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

const panel = "rounded-xl border border-[#284848] bg-black/30 p-4";
const field =
  "w-full rounded-lg border border-[#284848] bg-black/45 p-3 text-sm text-text-primary outline-none focus:border-emerald-400/60";
const safetyBadge = "border-red-400/35 bg-red-500/10 text-red-100";

function splitLines(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function roomCopyText(room: ProjectRoom | null, messages: ProjectRoomMessage[]): string {
  if (!room) return "";
  const lines = [
    `Project Room: ${room.title}`,
    `Project key: ${room.project_key}`,
    "trusted_for_execution=false",
    "inert_context_only=true",
    "",
    ...messages.map((message) => `[${message.role}] ${message.content_text}`),
  ];
  return lines.join("\n");
}

export function ProjectRoomsPanel() {
  const [rooms, setRooms] = useState<ProjectRoom[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [messages, setMessages] = useState<ProjectRoomMessage[]>([]);
  const [noteText, setNoteText] = useState("");
  const [sourceRefs, setSourceRefs] = useState("");
  const [packetIds, setPacketIds] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const selectedRoom = useMemo(
    () => rooms.find((room) => room.id === selectedId) || rooms.find((room) => room.slug === selectedId) || null,
    [rooms, selectedId],
  );

  const loadRooms = useCallback(async (preferredId?: string) => {
    const response = await api.listProjectRooms();
    setRooms(response.rooms);
    const nextId = preferredId || response.rooms[0]?.id || "";
    setSelectedId(nextId);
    if (!nextId) setMessages([]);
  }, []);

  const loadMessages = useCallback(async (roomId: string) => {
    const response = await api.listProjectRoomMessages(roomId);
    setMessages(response.messages);
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      loadRooms().catch((error) => {
        setMessage(error instanceof Error ? error.message : "Could not load Project Rooms");
      });
    }, 0);
    return () => window.clearTimeout(initial);
  }, [loadRooms]);

  useEffect(() => {
    if (!selectedId) return;
    loadMessages(selectedId).catch((error) => {
      setMessage(error instanceof Error ? error.message : "Could not load room messages");
    });
  }, [loadMessages, selectedId]);

  const addNote = async () => {
    if (!selectedRoom || !noteText.trim()) return;
    setBusy("note");
    setMessage(null);
    try {
      await api.addProjectRoomMessage(selectedRoom.id, {
        author: "dashboard",
        role: "note",
        content_type: "text",
        content_text: noteText,
        source_refs: splitLines(sourceRefs),
        linked_packet_ids: splitLines(packetIds),
        trusted_for_execution: false,
      });
      setNoteText("");
      setSourceRefs("");
      setPacketIds("");
      await loadRooms(selectedRoom.id);
      await loadMessages(selectedRoom.id);
      setMessage("Saved as local inert project context.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save note");
    } finally {
      setBusy(null);
    }
  };

  const attachFiles = async (files: FileList | null) => {
    if (!selectedRoom || !files?.length) return;
    setBusy("file");
    setMessage(null);
    try {
      for (const file of Array.from(files)) {
        await api.uploadProjectRoomAttachment(selectedRoom.id, file);
      }
      await loadRooms(selectedRoom.id);
      setMessage("Attachment metadata saved; files remain inert project context.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not attach file");
    } finally {
      setBusy(null);
    }
  };

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(roomCopyText(selectedRoom, messages));
      setMessage("Copied Project Room text.");
    } catch {
      setMessage("Copy failed.");
    }
  };

  return (
    <Card className="font-readable-ui overflow-hidden rounded-2xl border border-emerald-400/20 bg-black/30">
      <CardContent className="space-y-4 p-5">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-midground">
              <MessageSquare className="h-5 w-5" />
              <H2 className="text-xl">Project Rooms</H2>
            </div>
            <Typography className="mt-1 text-sm leading-6 text-text-secondary">
              Paste text, logs, and notes into local inert project context. No execution path is attached.
            </Typography>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge tone="outline" className={cn("py-1.5", safetyBadge)}>
              trusted_for_execution=false
            </Badge>
            <Badge tone="outline" className="border-emerald-400/35 bg-emerald-500/10 py-1.5 text-emerald-100">
              review context only
            </Badge>
          </div>
        </div>

        {message && (
          <div className="rounded-xl border border-[#284848] bg-black/35 px-3 py-2 text-sm text-text-secondary">
            {message}
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-[260px_minmax(0,1fr)]">
          <div className={panel}>
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">
              <ShieldCheck className="h-4 w-4" />
              Rooms
            </div>
            <div className="space-y-2">
              {rooms.map((room) => (
                <button
                  key={room.id}
                  type="button"
                  onClick={() => setSelectedId(room.id)}
                  className={cn(
                    "w-full rounded-lg border px-3 py-2 text-left text-sm transition",
                    selectedRoom?.id === room.id
                      ? "border-emerald-400/50 bg-emerald-500/10 text-emerald-100"
                      : "border-[#284848] bg-black/25 text-text-secondary hover:border-emerald-400/30",
                  )}
                >
                  <div className="font-semibold text-text-primary">{room.title}</div>
                  <div className="mt-1 text-xs text-text-secondary">
                    {room.message_count} notes · {room.attachment_count} files
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <div className={panel}>
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="text-lg font-semibold text-text-primary">{selectedRoom?.title || "No Project Room selected"}</div>
                  <div className="mt-1 text-sm leading-6 text-text-secondary">{selectedRoom?.description || "Rooms are stored locally under Mission Control state."}</div>
                </div>
                <Button ghost onClick={copyText} className="gap-2" disabled={!selectedRoom}>
                  <Clipboard className="h-4 w-4" />
                  Copy text
                </Button>
              </div>
            </div>

            <div className={panel}>
              <div className="mb-3 text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">Paste text, logs, and notes</div>
              <textarea
                value={noteText}
                onChange={(event) => setNoteText(event.target.value)}
                className={cn(field, "min-h-28")}
                placeholder="Local room note"
              />
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <label className="space-y-2">
                  <span className="text-xs uppercase tracking-[0.08em] text-text-secondary">Source refs</span>
                  <textarea value={sourceRefs} onChange={(event) => setSourceRefs(event.target.value)} className={cn(field, "min-h-20")} />
                </label>
                <label className="space-y-2">
                  <span className="text-xs uppercase tracking-[0.08em] text-text-secondary">Mission Packet IDs</span>
                  <textarea value={packetIds} onChange={(event) => setPacketIds(event.target.value)} className={cn(field, "min-h-20")} />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button onClick={addNote} disabled={!selectedRoom || !noteText.trim() || busy === "note"}>
                  {busy === "note" ? "Saving" : "Add note"}
                </Button>
                <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[#284848] bg-black/35 px-3 py-2 text-sm text-text-primary hover:border-emerald-400/40">
                  <FileUp className="h-4 w-4" />
                  Attach files
                  <input className="hidden" type="file" multiple onChange={(event) => attachFiles(event.target.files)} />
                </label>
              </div>
            </div>

            <div className={panel}>
              <div className="mb-3 text-sm font-semibold uppercase tracking-[0.08em] text-emerald-100/90">Room notes</div>
              <div className="space-y-2">
                {messages.length ? (
                  messages.map((item) => (
                    <div key={item.id} className="rounded-lg border border-[#284848] bg-black/25 p-3">
                      <div className="mb-1 flex flex-wrap items-center gap-2 text-xs text-text-secondary">
                        <span>{item.author}</span>
                        <span>{item.role}</span>
                        <span>{item.created_at}</span>
                      </div>
                      <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-6 text-text-primary">{item.content_text}</pre>
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-text-secondary">No room notes yet.</div>
                )}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
