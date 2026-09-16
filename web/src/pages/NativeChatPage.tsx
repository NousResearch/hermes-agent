import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { GatewayClient, type ConnectionState, type GatewayEvent } from "@/lib/gatewayClient";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";
import { api } from "@/lib/api";
import {
  ARTIFACT_STORAGE_CHANGE_EVENT,
  MAX_PERSISTED_ARTIFACT_STORAGE_BYTES,
  artifactStorageBytes,
  getArtifactStorage,
  getArtifactStorageKey,
  makeArtifactId,
  readStoredArtifacts,
} from "@/lib/artifact-storage";
import { cn } from "@/lib/utils";
import { ChatSessionList, type SessionActivityStatus } from "@/components/ChatSessionList";
import { SlashPopover, type SlashPopoverHandle } from "@/components/SlashPopover";
import { MarkdownMessage } from "@/components/chat/MarkdownMessage";
import {
  chooseCompletedText,
  mergeCompletedAssistantMessage,
  mergeInflightTranscript,
  mergeLiveTimelineTranscript,
  mergeSnapshotTranscript,
  snapshotHasField,
  snapshotMatchesSession,
} from "@/lib/native-chat-reconcile";
import {
  applyEditedTranscript,
  buildEditSubmitParams,
  parseDurableRowId,
  type EditSubmitResponse,
} from "@/lib/native-chat-edit";
import {
  initialNativeChatTimeline,
  projectTimelineEntries,
  reduceNativeChatTimeline,
  type TimelineAction,
  type TimelineEventInput,
} from "@/lib/native-chat-timeline";
import { getVirtualRange } from "@/lib/native-chat-virtualization";
import { filterTranscriptMessages } from "@/lib/native-chat-search";
import { appendVoiceTranscript, canRecordVoice, chooseRecordingMimeType } from "@/lib/voice";
import { ToolActivity, type ToolActivityItem } from "@/components/chat/ToolActivity";
import { ApprovalCard, type ApprovalRequest } from "@/components/chat/ApprovalCard";
import { ClarificationCard, type ClarificationRequest } from "@/components/chat/ClarificationCard";
import { MessageActions } from "@/components/chat/MessageActions";
import { CommandPalette } from "@/components/chat/CommandPalette";
import { EditMessageDialog } from "@/components/chat/EditMessageDialog";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { ArrowDown, Menu, MessageSquare, Mic, Paperclip, RotateCcw, Send, Square, X } from "lucide-react";
import { useSearchParams } from "react-router";
import {
  nativeChatModelChoices,
  nativeChatSessionCreateParams,
  NATIVE_REASONING_OPTIONS,
  selectionFromSearchParams,
  type ModelOptionsCatalog,
  type NativeReasoningLevel,
} from "@/lib/native-chat-routing";

type TranscriptMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  rowId?: number;
  messageId?: string;
  turnId?: string;
  streaming?: boolean;
  error?: string;
  interim?: boolean;
};

type ResumeMessage = { role?: unknown; text?: unknown; content?: unknown; row_id?: unknown; id?: unknown; message_id?: unknown; messageId?: unknown; turn_id?: unknown; turnId?: unknown; error?: unknown; interim?: unknown; streaming?: unknown };
type ApprovalSnapshot = { request_id?: unknown; command?: unknown; description?: unknown; choices?: unknown; allow_session?: unknown; allow_permanent?: unknown; smart_denied?: unknown; expires_at?: unknown };
type ClarifySnapshot = { answers?: Record<string, string>; request_id?: unknown; question?: unknown; choices?: unknown; multi_select?: unknown; questions?: unknown };
type InflightSnapshot = {
  user?: unknown;
  assistant?: unknown;
  streaming?: unknown;
  error?: unknown;
  status?: unknown;
  recoverable?: unknown;
  error_surface?: unknown;
  turn_scoped?: unknown;
  turn_id?: unknown;
  turnId?: unknown;
  message_id?: unknown;
  messageId?: unknown;
  started_at?: unknown;
  corrections?: unknown;
  correction_offsets?: unknown;
};
type ResumeResponse = {
  session_id?: string;
  stored_session_id?: string;
  messages?: ResumeMessage[];
  messages_omitted?: boolean;
  running?: boolean;
  turn_started_at?: number | null;
  status?: string;
  error?: string;
  info?: { running?: boolean; turn_started_at?: number | null; status?: string; error?: string; stored_session_id?: string; session_key?: string };
  inflight?: InflightSnapshot | null;
  session_key?: string;
  pending_approval?: ApprovalSnapshot;
  pending_clarify?: ClarifySnapshot;
};

type StopTarget = {
  sessionId: string | null;
  durableSessionId: string | null;
  sessionKey: string | null;
  sessionGeneration: number;
  requestGeneration: number;
  stoppedAt: number;
  promptText: string | null;
  assistantId: string | null;
  assistantText: string | null;
  messageId: string | null;
  turnId: string | null;
  turnGeneration: number;
};

type ResyncBarrier = {
  sessionGeneration: number;
  operationToken: number;
  stopRequestGeneration: number;
  turnGeneration: number;
  connectionEpoch: number;
};

type BranchResponse = {
  session_id?: string;
  stored_session_id?: string;
  title?: string;
};

function compatibleLiveText(liveText: string, snapshotText: string): string {
  if (!liveText) return snapshotText;
  if (!snapshotText) return liveText;
  if (liveText.startsWith(snapshotText)) return liveText;
  if (snapshotText.startsWith(liveText)) return snapshotText;
  return liveText;
}

function snapshotText(message: ResumeMessage): string {
  if (typeof message.text === "string") return message.text;
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content.map((part) => typeof part === "string" ? part : typeof part === "object" && part !== null && "text" in part ? String((part as { text?: unknown }).text ?? "") : "").join("");
  }
  return "";
}

function snapshotTranscript(messages: ResumeMessage[] | undefined): TranscriptMessage[] {
  return (messages ?? []).flatMap((message, index) => {
    const role = message.role === "user" ? "user" : message.role === "assistant" ? "assistant" : null;
    if (!role) return [];
    const rowId = parseDurableRowId(message.row_id);
    const messageId = typeof message.message_id === "string" && message.message_id
      ? message.message_id
      : typeof message.messageId === "string" && message.messageId ? message.messageId : undefined;
    const turnId = typeof message.turn_id === "string" && message.turn_id
      ? message.turn_id
      : typeof message.turnId === "string" && message.turnId ? message.turnId : undefined;
    const metadata = {
      ...(typeof message.streaming === "boolean" ? { streaming: message.streaming } : {}),
      ...(typeof message.error === "string" && message.error ? { error: message.error } : {}),
      ...(typeof message.interim === "boolean" ? { interim: message.interim } : {}),
    };
    return [{
      id: String(rowId ?? message.row_id ?? messageId ?? message.id ?? `snapshot-${index}`),
      role,
      text: snapshotText(message),
      ...(rowId !== undefined ? { rowId } : {}),
      ...(messageId ? { messageId } : {}),
      ...(turnId ? { turnId } : {}),
      ...metadata,
    }];
  });
}

function inflightText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function inflightIdentity(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function inflightLatestPrompt(snapshot: InflightSnapshot): string {
  const corrections = Array.isArray(snapshot.corrections)
    ? snapshot.corrections.filter((value): value is string => typeof value === "string" && value.length > 0)
    : [];
  return corrections.at(-1) ?? inflightText(snapshot.user);
}

function inflightTranscript(snapshot: InflightSnapshot, sessionId: string, fallbackIdentity = "active"): TranscriptMessage[] {
  const messageId = inflightIdentity(snapshot.message_id ?? snapshot.messageId);
  const turnId = inflightIdentity(snapshot.turn_id ?? snapshot.turnId);
  const identityMetadata = {
    ...(messageId ? { messageId } : {}),
    ...(turnId ? { turnId } : {}),
  };
  const identity = messageId
    ?? turnId
    ?? (typeof snapshot.started_at === "number" ? String(snapshot.started_at) : fallbackIdentity);
  const rowIdentity = identity.startsWith("snapshot-") ? identity : `${sessionId}:${identity}`;
  const user = inflightText(snapshot.user);
  const assistant = inflightText(snapshot.assistant);
  const corrections = Array.isArray(snapshot.corrections)
    ? snapshot.corrections.filter((value): value is string => typeof value === "string" && value.length > 0)
    : [];
  const offsets = Array.isArray(snapshot.correction_offsets)
    ? snapshot.correction_offsets.map((value) => typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null)
    : [];
  const error = typeof snapshot.error === "string" && snapshot.error
    ? snapshot.error
    : snapshot.status === "error" ? "Turn failed" : undefined;
  const rows: TranscriptMessage[] = [];
  if (user) rows.push({ id: `inflight-user:${rowIdentity}`, role: "user", text: user });
  if (corrections.length > 0) {
    let cursor = 0;
    corrections.forEach((correction, index) => {
      const requestedOffset = offsets[index];
      const offset = requestedOffset === null || requestedOffset === undefined
        ? (index === 0 ? assistant.length : cursor)
        : Math.min(assistant.length, Math.max(cursor, requestedOffset));
      rows.push({
        id: `inflight-assistant:${rowIdentity}:${index}`,
        role: "assistant",
        text: assistant.slice(cursor, offset),
        streaming: false,
      });
      rows.push({ id: `inflight-correction-user:${rowIdentity}:${index}`, role: "user", text: correction });
      cursor = offset;
    });
    rows.push({
      id: `inflight-assistant:${rowIdentity}:${corrections.length}`,
      role: "assistant",
      text: assistant.slice(cursor),
      streaming: snapshot.streaming === true,
      ...identityMetadata,
      ...(error ? { error } : {}),
    });
  } else if (assistant || snapshot.streaming === true || error) {
    rows.push({ id: `inflight-assistant:${rowIdentity}`, role: "assistant", text: assistant, streaming: snapshot.streaming === true, ...identityMetadata });
    if (error) rows[rows.length - 1] = { ...rows[rows.length - 1], error };
  }
  return rows;
}

const APPROVAL_CHOICES = new Set(["once", "session", "always", "deny"]);

export function normalizeApprovalRequestPayload(payload: Record<string, unknown>): ApprovalRequest | null {
  const requestId = payload.request_id;
  if (typeof requestId !== "string" || !requestId.trim()) return null;
  const hasChoices = Object.prototype.hasOwnProperty.call(payload, "choices");
  let choices: string[] | undefined;
  if (hasChoices) {
    const raw = payload.choices;
    const valid = Array.isArray(raw)
      ? raw.filter((choice): choice is string => typeof choice === "string" && APPROVAL_CHOICES.has(choice))
      : [];
    choices = valid.filter((choice, index) => valid.indexOf(choice) === index);
    if (payload.allow_session !== true) choices = choices.filter((choice) => choice !== "session");
    if (payload.allow_permanent !== true) choices = choices.filter((choice) => choice !== "always");
    if (!choices.includes("deny")) choices.push("deny");
    if (choices.length === 0) choices = ["deny"];
  }
  const allowSession = payload.allow_session === undefined ? true : payload.allow_session === true;
  const allowPermanent = payload.allow_permanent === undefined ? true : payload.allow_permanent === true;
  return {
    request_id: requestId,
    ...(typeof payload.command === "string" ? { command: payload.command } : {}),
    ...(typeof payload.description === "string" ? { description: payload.description } : {}),
    ...(choices ? { choices } : {}),
    allow_session: allowSession,
    allow_permanent: allowPermanent,
    smart_denied: payload.smart_denied === true,
    ...(typeof payload.expires_at === "number" && Number.isFinite(payload.expires_at) ? { expires_at: payload.expires_at } : {}),
  };
}

function approvalFromSnapshot(snapshot?: ApprovalSnapshot): ApprovalRequest | null {
  return snapshot ? normalizeApprovalRequestPayload(snapshot as Record<string, unknown>) : null;
}

export function appendApprovalRequest(queue: ApprovalRequest[], next: ApprovalRequest): ApprovalRequest[] {
  return queue.some((item) => item.request_id === next.request_id) ? queue : [...queue, next];
}

export function removeApprovalRequest(queue: ApprovalRequest[], requestId: string): { head: ApprovalRequest | null; queue: ApprovalRequest[] } {
  const remaining = queue.filter((item) => item.request_id !== requestId);
  return { head: remaining[0] ?? null, queue: remaining };
}

export function reconnectActivateParams(sessionId: string, profile: string): Record<string, unknown> {
  return {
    session_id: sessionId,
    omit_messages: false,
    continue_on_disconnect: true,
    ...(profile ? { profile } : {}),
  };
}

export function isNativeChatWorking(input: { streaming: boolean; runningTools: number; turnStartedAt: number | null; hasPendingInteraction: boolean }): boolean {
  return input.streaming || input.runningTools > 0 || input.turnStartedAt !== null || input.hasPendingInteraction;
}

export function shouldClearClarificationResponse(response: unknown, questionId?: string): boolean {
  if (!questionId) return true;
  if (typeof response !== "object" || response === null) return false;
  const remaining = (response as { remaining?: unknown }).remaining;
  return remaining === 0 || (Array.isArray(remaining) && remaining.length === 0);
}

function clarifyFromSnapshot(snapshot?: ClarifySnapshot): ClarificationRequest | null {
  if (!snapshot || typeof snapshot.request_id !== "string" || !snapshot.request_id.trim()) return null;
  return { request_id: snapshot.request_id, question: typeof snapshot.question === "string" ? snapshot.question : undefined, choices: Array.isArray(snapshot.choices) ? snapshot.choices.filter((x): x is string => typeof x === "string") : null, multi_select: snapshot.multi_select === true, questions: Array.isArray(snapshot.questions) ? snapshot.questions as ClarificationRequest["questions"] : undefined, answers: snapshot.answers };
}
type TextPayload = { text?: unknown; message?: unknown; kind?: unknown; running?: unknown; turn_started_at?: unknown; status?: unknown; request_id?: unknown; answer?: unknown; question?: unknown; choices?: unknown; command?: unknown; description?: unknown; allow_session?: unknown; expires_at?: unknown; tool_id?: unknown; name?: unknown; context?: unknown; args?: unknown; result?: unknown; summary?: unknown; progress?: unknown; questions?: unknown; multi_select?: unknown; allow_permanent?: unknown; smart_denied?: unknown; seq?: unknown; event_id?: unknown; eventId?: unknown; message_id?: unknown; messageId?: unknown; assistant_id?: unknown; assistantId?: unknown; turn_id?: unknown; turnId?: unknown; elapsed_ms?: unknown; elapsedMs?: unknown; error?: unknown; error_surface?: unknown; turn_scoped?: unknown };

type ResyncState = "idle" | "syncing" | "synced" | "partial" | "error";
type VoiceState = "idle" | "starting" | "recording" | "transcribing";

type PendingAttachment = {
  id: string;
  file: File;
  state: "pending" | "uploading" | "attached" | "error";
  error?: string;
  refText?: string;
  refPath?: string;
  previewUrl?: string;
};
type PendingPrompt = { id: string; text: string; mode: "retry" | "queued" };

const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

// Keep the few quick prompts local until every locale has a translated
// command-palette/chat namespace.
const QUICK_PROMPTS = [
  "Summarize this text",
  "Explain a concept simply",
  "Draft an email",
  "Plan my next steps",
] as const;

function fileDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read attachment"));
    reader.onload = () => typeof reader.result === "string" ? resolve(reader.result) : reject(new Error("Could not read attachment"));
    reader.readAsDataURL(file);
  });
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function stopMediaStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop());
}

export function attachmentPromptText(text: string, attachments: PendingAttachment[]): string {
  const refs = attachments.filter((item) => item.state === "attached").flatMap((item) => [item.refText, item.refPath]).filter(Boolean);
  return refs.length ? [text, refs.join("\n")].filter(Boolean).join("\n\n") : text;
}

export function shouldSubmitComposerKey(key: string, shiftKey: boolean, isComposing: boolean): boolean {
  return key === "Enter" && !shiftKey && !isComposing;
}

function eventText(event: GatewayEvent): string {
  const payload = event.payload as TextPayload | undefined;
  if (typeof payload?.text === "string") return payload.text;
  if (typeof payload?.message === "string") return payload.message;
  return "";
}

function eventTurnId(event: GatewayEvent): string | null {
  const payload = event.payload as TextPayload | undefined;
  const value = payload?.turn_id ?? payload?.turnId;
  return typeof value === "string" && value ? value : null;
}

function eventMessageId(event: GatewayEvent): string | null {
  const payload = event.payload as TextPayload | undefined;
  const value = payload?.message_id ?? payload?.messageId ?? payload?.assistant_id ?? payload?.assistantId;
  return typeof value === "string" && value ? value : null;
}

export function shouldReleaseQueueDrain(capturedGeneration: number, currentGeneration: number): boolean {
  return capturedGeneration === currentGeneration;
}

export function shouldRestoreStopTarget(
  captured: Pick<StopTarget, "sessionGeneration" | "requestGeneration" | "assistantId" | "turnId" | "turnGeneration">,
  current: Pick<StopTarget, "sessionGeneration" | "requestGeneration" | "assistantId" | "turnId" | "turnGeneration">,
): boolean {
  return captured.sessionGeneration === current.sessionGeneration
    && captured.requestGeneration === current.requestGeneration
    && captured.assistantId === current.assistantId
    && captured.turnId === current.turnId
    && captured.turnGeneration === current.turnGeneration;
}

export type DurableIdentityValidation = {
  accepted: boolean;
  canonicalId: string | null;
  validatedIds: string[];
};

export function validateDurableIdentityResponse(
  returnedCanonical: string | undefined,
  returnedAliases: readonly (string | undefined)[],
  currentDurableIds: readonly string[],
): DurableIdentityValidation {
  const returned = [...new Set([returnedCanonical, ...returnedAliases].filter((value): value is string => typeof value === "string" && value.length > 0))];
  if (returned.length === 0) return { accepted: false, canonicalId: null, validatedIds: [] };
  if (currentDurableIds.length === 0) {
    return { accepted: true, canonicalId: returnedCanonical ?? returned[0] ?? null, validatedIds: returned };
  }
  const known = returned.filter((value) => currentDurableIds.includes(value));
  if (known.length === 0) return { accepted: false, canonicalId: null, validatedIds: [] };
  const canonicalIsKnown = returnedCanonical !== undefined && currentDurableIds.includes(returnedCanonical);
  const unknownAliases = returnedAliases
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .filter((value) => !currentDurableIds.includes(value) && value !== returnedCanonical);
  const validatedIds = [...new Set([
    ...known,
    ...(returnedCanonical && !currentDurableIds.includes(returnedCanonical) ? [returnedCanonical] : []),
    ...(canonicalIsKnown && new Set(unknownAliases).size === 1 ? [unknownAliases[0] as string] : []),
  ])];
  return {
    accepted: true,
    canonicalId: returnedCanonical ?? known[0] ?? null,
    validatedIds,
  };
}

export function shouldMergeIdentityLessInflight(
  activeAssistantText: string,
  incomingAssistantText: string,
  activePromptText: string | null,
  incomingPromptText: string,
): boolean {
  return activeAssistantText.length > 0
    && incomingAssistantText.length > 0
    && activeAssistantText === incomingAssistantText
    && activePromptText !== null
    && activePromptText === incomingPromptText;
}

export function buildInflightFallbackKey(
  scope: string,
  user: string,
  status: string,
  error: string,
  revision?: number,
): string {
  return JSON.stringify([scope, user, status, error, revision ?? null]);
}

type InflightFallbackKeyParts = {
  scope: string;
  user: string;
  status: string;
  error: string;
  revision: number | null;
};

function parseInflightFallbackKey(value: string): InflightFallbackKeyParts | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed) || parsed.length !== 5
      || typeof parsed[0] !== "string"
      || typeof parsed[1] !== "string"
      || typeof parsed[2] !== "string"
      || typeof parsed[3] !== "string"
      || (parsed[4] !== null && typeof parsed[4] !== "number")) return null;
    return { scope: parsed[0], user: parsed[1], status: parsed[2], error: parsed[3], revision: parsed[4] };
  } catch {
    return null;
  }
}

function sameInflightFallbackBase(left: string, right: string): boolean {
  const leftParts = parseInflightFallbackKey(left);
  const rightParts = parseInflightFallbackKey(right);
  return leftParts !== null
    && rightParts !== null
    && leftParts.scope === rightParts.scope
    && leftParts.user === rightParts.user
    && leftParts.status === rightParts.status
    && leftParts.error === rightParts.error;
}

function rekeyInflightFallbackKey(value: string, fromScope: string, toScope: string): string | null {
  const parts = parseInflightFallbackKey(value);
  if (!parts || parts.scope !== fromScope) return null;
  return buildInflightFallbackKey(parts.scope === fromScope ? toScope : parts.scope, parts.user, parts.status, parts.error, parts.revision ?? undefined);
}

export function buildScopedIdentityKey(scope: string, identifier: string): string {
  return JSON.stringify([scope, identifier]);
}

function parseScopedIdentityKey(value: string): [string, string] | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      && parsed.length === 2
      && typeof parsed[0] === "string"
      && typeof parsed[1] === "string"
      ? [parsed[0], parsed[1]]
      : null;
  } catch {
    return null;
  }
}

function rekeyScopedSet(values: Set<string>, fromScope: string | null | undefined, toScope: string | null | undefined): void {
  if (!fromScope || !toScope || fromScope === toScope) return;
  const additions: Array<{ oldKey: string; newKey: string }> = [];
  for (const value of values) {
    const parsed = parseScopedIdentityKey(value);
    if (parsed?.[0] === fromScope) additions.push({ oldKey: value, newKey: buildScopedIdentityKey(toScope, parsed[1]) });
  }
  for (const { oldKey, newKey } of additions) {
    values.delete(oldKey);
    values.add(newKey);
  }
}

function messageIdentityKey(
  runtimeSessionId: string | null | undefined,
  messageId: string,
  durableSessionId?: string | null,
  sessionKey?: string | null,
): string {
  return buildScopedIdentityKey(sessionKey ?? durableSessionId ?? runtimeSessionId ?? "global", messageId);
}

function rekeyStopTargetScope(target: StopTarget | null, fromScope: string, toScope: string): StopTarget | null {
  if (!target || fromScope === toScope) return target;
  return {
    ...target,
    durableSessionId: target.durableSessionId === fromScope ? toScope : target.durableSessionId,
    sessionKey: target.sessionKey === fromScope ? toScope : target.sessionKey,
  };
}

function stopTargetSessionMatches(
  target: StopTarget,
  runtimeSessionId: string | null,
  durableSessionId: string | null,
  sessionKey: string | null,
): boolean {
  return (target.sessionKey !== null && sessionKey !== null && target.sessionKey === sessionKey)
    || (target.durableSessionId !== null && durableSessionId !== null && target.durableSessionId === durableSessionId)
    || target.sessionId === runtimeSessionId;
}

const TURN_SCOPED_EVENT_TYPES = new Set([
  "message.start",
  "message.delta",
  "message.interim",
  "message.complete",
  "error",
  "thinking.delta",
  "reasoning.delta",
  "tool.generating",
  "tool.start",
  "tool.progress",
  "tool.complete",
  "approval.request",
  "clarify.request",
]);

function eventElapsedMs(payload: TextPayload, startedAt?: number): number | undefined {
  const explicit = payload.elapsed_ms ?? payload.elapsedMs;
  if (typeof explicit === "number" && Number.isFinite(explicit) && explicit >= 0) return explicit;
  if (startedAt !== undefined) return Math.max(0, Date.now() - startedAt);
  return undefined;
}

function toTimelineEvent(event: GatewayEvent, sessionKey?: string | null): TimelineEventInput {
  const payload = (event.payload ?? {}) as TextPayload;
  const rawEventId = (event as GatewayEvent & { event_id?: unknown }).event_id ?? payload.event_id ?? payload.eventId;
  const rawSeq = (event as GatewayEvent & { seq?: unknown }).seq ?? payload.seq;
  return {
    type: event.type,
    session_id: event.session_id,
    session_key: sessionKey ?? undefined,
    payload: event.payload,
    event_id: typeof rawEventId === "string" ? rawEventId : undefined,
    seq: typeof rawSeq === "number" ? rawSeq : undefined,
  };
}

export function shouldFollowTranscript(distanceFromBottom: number): boolean {
  return distanceFromBottom <= 96;
}

function connectionLabel(state: ConnectionState): string {
  return state === "open" ? "Connected" : state[0].toUpperCase() + state.slice(1);
}

function rekeyPinnedArtifacts(
  profile: string | undefined,
  fromSessionId: string | null | undefined,
  toSessionId: string | null | undefined,
): void {
  if (!fromSessionId || !toSessionId || fromSessionId === toSessionId) return;
  const storage = getArtifactStorage();
  if (!storage) return;
  const profileScope = profile || "default";
  const current = readStoredArtifacts(storage, profileScope);
  const rekeyed = current.map((artifact) => artifact.sessionId === fromSessionId
    ? {
      ...artifact,
      sessionId: toSessionId,
      id: makeArtifactId(toSessionId, artifact.kind, artifact.language, artifact.title, artifact.code),
    }
    : artifact);
  if (rekeyed.every((artifact, index) => artifact === current[index])) return;
  if (artifactStorageBytes(rekeyed) > MAX_PERSISTED_ARTIFACT_STORAGE_BYTES) return;
  try {
    const key = getArtifactStorageKey(profileScope, storage);
    storage.setItem(key, JSON.stringify(rekeyed));
    if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
      window.dispatchEvent(new CustomEvent(ARTIFACT_STORAGE_CHANGE_EVENT, { detail: { key } }));
    }
  } catch {
    // Keep the old records if the browser refuses the migration write.
  }
}

interface NativeChatPageProps {
  /** Open the shell navigation drawer on compact/mobile layouts. */
  onOpenNavigation?: () => void;
}

const TranscriptBubble = memo(function TranscriptBubble({
  message,
  sessionId,
  onUseAsPrompt,
  onSpeak,
  onEdit,
  onRegenerate,
}: {
  message: TranscriptMessage;
  sessionId?: string;
  onUseAsPrompt: (message: string) => void;
  onSpeak: (message: string) => Promise<void>;
  onEdit: (message: TranscriptMessage) => void;
  onRegenerate?: () => void;
}) {
  return (
    <article
      data-slot="transcript-message"
      data-message-id={message.id}
      data-message-role={message.role}
      data-message-streaming={message.streaming ? "true" : "false"}
      data-message-error={message.error ? "true" : "false"}
      data-message-interim={message.interim ? "true" : "false"}
      aria-label={message.role === "user" ? "Your message" : "Hermes message"}
      className={cn(
        "w-fit max-w-[85%] whitespace-pre-wrap rounded-md px-3 py-2 text-sm",
        message.role === "user" ? "ml-auto bg-primary text-primary-foreground" : "bg-muted text-foreground",
      )}
    >
      {message.role === "assistant"
        ? <MarkdownMessage content={message.text || (message.streaming ? "…" : "")} sessionId={sessionId} streaming={message.streaming} />
        : message.text}
      {message.text && (
        <MessageActions
          message={message.text}
          messageRole={message.role}
          onUseAsPrompt={onUseAsPrompt}
          editLabel={message.rowId !== undefined ? "Edit" : "Edit draft"}
          onEdit={message.role === "user" ? () => onEdit(message) : undefined}
          onRegenerate={message.role === "assistant" ? onRegenerate : undefined}
          onSpeak={message.role === "assistant" ? onSpeak : undefined}
        />
      )}
      {message.error && <span role="alert" className="mt-2 block text-xs text-destructive">{message.error}</span>}
    </article>
  );
});

function ToolTimeline({
  tools,
  className,
}: {
  tools: readonly ToolActivityItem[];
  className: string;
}) {
  const runningTools = tools.filter((tool) => tool.state === "running");
  const completedTools = tools.filter((tool) => tool.state === "complete");
  if (runningTools.length === 0 && completedTools.length === 0) return null;
  return (
    <div
      data-testid="tool-timeline"
      data-slot="tool-timeline"
      className={className}
      aria-label="Tool activity timeline"
    >
      {runningTools.map((tool) => <ToolActivity key={tool.id} item={tool} />)}
      {completedTools.length > 0 && (
        <details
          data-slot="completed-tool-activity"
          className="rounded-md border border-border bg-background px-3 py-2 text-xs"
        >
          <summary className="flex cursor-pointer list-none items-center gap-2 text-muted-foreground [&::-webkit-details-marker]:hidden">
            <span aria-hidden>✓</span>
            <span>Background work complete · {completedTools.length} {completedTools.length === 1 ? "step" : "steps"}</span>
            <span className="ml-auto">Show details</span>
          </summary>
          <div data-slot="tool-activity-details" className="mt-2 space-y-2">
            {completedTools.map((tool) => <ToolActivity key={tool.id} item={tool} />)}
          </div>
        </details>
      )}
    </div>
  );
}

export default function NativeChatPage({ onOpenNavigation }: NativeChatPageProps) {
  const { profile } = useProfileScope();
  const { t } = useI18n();
  const chat = {
    title: t.chat?.title ?? "Chat",
    subtitle: t.chat?.subtitle ?? "Native gateway chat",
    startConversation: t.chat?.startConversation ?? "Start a conversation.",
    tryPrompt: t.chat?.tryPrompt ?? "Try a prompt",
    messagePlaceholder: t.chat?.messagePlaceholder ?? "Message Hermes… (drop or paste files)",
    dropPasteAttach: t.chat?.dropPasteAttach ?? "Drop files or paste to attach",
    send: t.chat?.send ?? "Send",
    queue: t.chat?.queue ?? "Queue",
    stop: t.chat?.stop ?? "Stop",
    recordVoice: t.chat?.recordVoice ?? "Record voice",
    stopVoiceRecording: t.chat?.stopVoiceRecording ?? "Stop voice recording",
    requestingMicrophone: t.chat?.requestingMicrophone ?? "Requesting microphone…",
    transcribing: t.chat?.transcribing ?? "Transcribing…",
    syncing: t.chat?.syncing ?? "Syncing",
    ready: t.chat?.ready ?? "Ready",
    working: t.chat?.working ?? "Working",
  };
  const [searchParams, setSearchParams] = useSearchParams();
  const resumeParam = searchParams.get("resume");
  const routeModel = searchParams.get("model");
  const routeProvider = searchParams.get("provider");
  const routeReasoning = searchParams.get("reasoning");
  const [modelCatalog, setModelCatalog] = useState<ModelOptionsCatalog>({});
  const modelChoices = useMemo(() => nativeChatModelChoices(modelCatalog), [modelCatalog]);
  const routingSelection = useMemo(() => selectionFromSearchParams(
    routeModel, routeProvider, routeReasoning, modelChoices,
  ), [modelChoices, routeModel, routeProvider, routeReasoning]);
  const routingSelectionRef = useRef(routingSelection);
  routingSelectionRef.current = routingSelection;
  const gateway = useMemo(() => new GatewayClient(), []);
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const connectionStateRef = useRef<ConnectionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [durableSessionId, setDurableSessionId] = useState<string | null>(resumeParam);
  const sessionIdRef = useRef<string | null>(null);
  const durableSessionIdRef = useRef<string | null>(resumeParam);
  const sessionKeyRef = useRef<string | null>(null);
  const validatedDurableAliasesRef = useRef(new Set<string>());
  const [freshGeneration, setFreshGeneration] = useState(0);
  const [draft, setDraft] = useState("");
  const [transcriptQuery, setTranscriptQuery] = useState("");
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const transcriptStateRef = useRef(transcript);
  transcriptStateRef.current = transcript;
  const activePromptTextRef = useRef<string | null>(null);
  const [editTarget, setEditTarget] = useState<TranscriptMessage | null>(null);
  const [editSubmitting, setEditSubmitting] = useState(false);
  const submitOwnerTokenRef = useRef<number | null>(null);
  const submitOwnerKindRef = useRef<"submit" | "edit" | null>(null);
  const [timelineState, dispatchTimelineState] = useReducer(reduceNativeChatTimeline, initialNativeChatTimeline);
  const timelineStateRef = useRef(timelineState);
  timelineStateRef.current = timelineState;
  const dispatchTimeline = useCallback((action: TimelineAction) => {
    const next = reduceNativeChatTimeline(timelineStateRef.current, action);
    timelineStateRef.current = next;
    dispatchTimelineState(action);
  }, [dispatchTimelineState]);
  const liveTimelineMessages = useMemo(() => projectTimelineEntries(timelineState.entries), [timelineState.entries]);
  const activePromptText = activePromptTextRef.current;
  const displayTranscript = useMemo(() => mergeLiveTimelineTranscript(
    transcript,
    liveTimelineMessages,
    activePromptText,
  ), [activePromptText, liveTimelineMessages, transcript]);
  const filteredTranscript = useMemo(() => filterTranscriptMessages(displayTranscript, transcriptQuery), [displayTranscript, transcriptQuery]);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resyncState, setResyncState] = useState<ResyncState>("idle");
  const [errorAction, setErrorAction] = useState<"reconnect" | "resend" | null>(null);
  const [failedPrompt, setFailedPrompt] = useState<PendingPrompt | null>(null);
  const [queuedPrompts, setQueuedPrompts] = useState<PendingPrompt[]>([]);
  const queueDrainInFlightRef = useRef(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const submitInFlightRef = useRef(false);
  const stopInFlightRef = useRef(false);
  const stopTargetRef = useRef<StopTarget | null>(null);
  const retiredStopTargetRef = useRef<StopTarget | null>(null);
  const unboundStopFenceRef = useRef<StopTarget | null>(null);
  const stopRequestGenerationRef = useRef(0);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const attachmentsRef = useRef<PendingAttachment[]>([]);
  const stagingRef = useRef(new Set<string>());
  const uploadControllersRef = useRef(new Map<string, AbortController>());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const messageSequenceRef = useRef(0);
  const inflightSnapshotSequenceRef = useRef(0);
  const inflightFallbackIdentityMapRef = useRef(new Map<string, string>());
  const activeInflightFallbackKeyRef = useRef<string | null>(null);
  const sessionGenerationRef = useRef(0);
  const operationTokenRef = useRef(0);
  const turnGenerationRef = useRef(0);
  const activeTurnIdRef = useRef<string | null>(null);
  const activeMessageIdRef = useRef<string | null>(null);
  const ignoredTurnIdsRef = useRef(new Set<string>());
  const retiredTurnIdsRef = useRef(new Set<string>());
  const retiredTurnTextsRef = useRef(new Map<string, string>());
  const replacementTurnMessageIdsRef = useRef(new Map<string, string>());
  const replacementTurnProvenByDeltaRef = useRef(new Set<string>());
  const retiredMessageIdsRef = useRef(new Set<string>());
  const blockedTurnGenerationRef = useRef<number | null>(null);
  const unboundStreamEstablishedRef = useRef(false);
  const unboundStreamPendingRef = useRef(false);
  const unboundStartAcceptedRef = useRef(false);
  const postRetirementUnboundStartPendingRef = useRef(false);
  const allowUnboundStartAfterRetirementRef = useRef(true);
  const composingRef = useRef(false);
  const assistantIdRef = useRef<string | null>(null);
  const [tools, setTools] = useState<ToolActivityItem[]>([]);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const approvalQueueRef = useRef<ApprovalRequest[]>([]);
  const [clarify, setClarify] = useState<ClarificationRequest | null>(null);
  const approvalRef = useRef<ApprovalRequest | null>(null);
  const clarifyRef = useRef<ClarificationRequest | null>(null);
  approvalRef.current = approval;
  clarifyRef.current = clarify;
  const setApprovalState = useCallback((next: ApprovalRequest | null) => {
    if (next === null) {
      approvalQueueRef.current = [];
      approvalRef.current = null;
      setApproval(null);
      return;
    }
    const queue = appendApprovalRequest(approvalQueueRef.current, next);
    approvalQueueRef.current = queue;
    const head = queue[0] ?? null;
    approvalRef.current = head;
    setApproval(head);
  }, []);
  const dismissApprovalState = useCallback((requestId: string) => {
    const next = removeApprovalRequest(approvalQueueRef.current, requestId);
    approvalQueueRef.current = next.queue;
    approvalRef.current = next.head;
    setApproval(next.head);
  }, []);
  const setClarifyState = useCallback((next: ClarificationRequest | null) => {
    clarifyRef.current = next;
    setClarify(next);
  }, []);
  const [streaming, setStreaming] = useState(false);
  const [turnStartedAt, setTurnStartedAt] = useState<number | null>(null);
  const [clockNow, setClockNow] = useState(() => Date.now());
  const transcriptRef = useRef<HTMLDivElement>(null);
  const virtualRowHeightsRef = useRef(new Map<string, number>());
  const [virtualMeasureRevision, setVirtualMeasureRevision] = useState(0);
  const [virtualViewport, setVirtualViewport] = useState({ scrollTop: 0, viewportHeight: 600 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const slashPopoverRef = useRef<SlashPopoverHandle>(null);
  const followTranscriptRef = useRef(true);
  const [showScrollToBottom, setShowScrollToBottom] = useState(false);
  const [mobileSessionNavigatorOpen, setMobileSessionNavigatorOpen] = useState(false);
  const wasOpenRef = useRef(false);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectInFlightRef = useRef(false);
  const reconnectRequestTokenRef = useRef(0);
  const seenSeqRef = useRef(new Map<string, number>());
  const seenEventIdsRef = useRef(new Set<string>());
  const reconnectingRef = useRef(false);
  const connectionEpochRef = useRef(0);
  const startResyncRef = useRef<(() => void) | null>(null);
  const resyncAfterStopRef = useRef(false);
  const resyncBarrierRef = useRef<ResyncBarrier | null>(null);
  const resyncBufferedEventsRef = useRef<Array<{ event: GatewayEvent; handler: (event: GatewayEvent) => void }>>([]);
  const resyncEventHandlersRef = useRef(new Map<string, (event: GatewayEvent) => void>());
  const replayingResyncEventsRef = useRef(false);
  const replayResyncEventsRef = useRef<(() => void) | null>(null);
  const resyncBufferOverflowRef = useRef(false);
  const initialAttachPendingRef = useRef(false);
  const initialAttachBufferedEventsRef = useRef<Array<{ event: GatewayEvent; handler: (event: GatewayEvent) => void }>>([]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!followTranscriptRef.current) return;
    const element = transcriptRef.current;
    if (!element) return;
    const timer = window.setTimeout(() => {
      element.scrollTop = element.scrollHeight;
    }, 0);
    return () => window.clearTimeout(timer);
  }, [approval, clarify, displayTranscript, error, status, tools]);

  const handleTranscriptScroll = useCallback(() => {
    const element = transcriptRef.current;
    if (!element) return;
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    const shouldFollow = shouldFollowTranscript(distanceFromBottom);
    followTranscriptRef.current = shouldFollow;
    setVirtualViewport((current) => current.scrollTop === element.scrollTop ? current : { ...current, scrollTop: element.scrollTop });
    setShowScrollToBottom(!shouldFollow);
  }, []);

  const scrollToBottom = useCallback(() => {
    const element = transcriptRef.current;
    if (!element) return;
    followTranscriptRef.current = true;
    element.scrollTop = element.scrollHeight;
    setVirtualViewport((current) => ({ ...current, scrollTop: element.scrollHeight }));
    setShowScrollToBottom(false);
  }, []);

  const measureTranscriptRow = useCallback((id: string, node: HTMLElement | null) => {
    if (!node) return;
    const height = Math.ceil(node.getBoundingClientRect().height);
    if (height <= 0) return;
    const previous = virtualRowHeightsRef.current.get(id);
    if (previous !== undefined && Math.abs(previous - height) < 1) return;
    virtualRowHeightsRef.current.set(id, height);
    setVirtualMeasureRevision((revision) => revision + 1);
  }, []);

  const liveAssistantText = useCallback((id: string) => (
    timelineStateRef.current.entries.find((entry) => entry.id === id)?.text ?? ""
  ), []);
  const retireActiveTurnIdentity = useCallback(() => {
    const turnId = activeTurnIdRef.current;
    if (turnId) ignoredTurnIdsRef.current.add(turnId);
    const messageId = activeMessageIdRef.current;
    if (messageId) retiredMessageIdsRef.current.add(messageIdentityKey(sessionIdRef.current, messageId, durableSessionIdRef.current, sessionKeyRef.current));
  }, []);
  const commitLiveAssistantMessage = useCallback((id: string, text: string, activePromptText = activePromptTextRef.current, error?: string) => {
    setTranscript((current) => mergeCompletedAssistantMessage(
      current,
      id,
      text,
      activePromptText,
      error,
    ) as TranscriptMessage[]);
    dispatchTimeline({ type: "reset" });
  }, [dispatchTimeline]);

  const retireInflightFallbackIdentity = useCallback(() => {
    const key = activeInflightFallbackKeyRef.current;
    if (key) inflightFallbackIdentityMapRef.current.delete(key);
    activeInflightFallbackKeyRef.current = null;
  }, []);

  const clearLocalTurnState = useCallback((finalText?: string, finalError?: string, allowNextUnboundStart = false) => {
    const id = assistantIdRef.current;
    const activePromptText = activePromptTextRef.current;
    retireInflightFallbackIdentity();
    unboundStopFenceRef.current = null;
    retireActiveTurnIdentity();
    if (id) commitLiveAssistantMessage(id, finalText ?? liveAssistantText(id), activePromptText, finalError);
    activeTurnIdRef.current = null;
    activeMessageIdRef.current = null;
    activePromptTextRef.current = null;
    postRetirementUnboundStartPendingRef.current = false;
    allowUnboundStartAfterRetirementRef.current = allowNextUnboundStart;
    unboundStreamEstablishedRef.current = false;
    unboundStreamPendingRef.current = false;
    unboundStartAcceptedRef.current = false;
    turnGenerationRef.current += 1;
    blockedTurnGenerationRef.current = turnGenerationRef.current;
    assistantIdRef.current = null;
    setStreaming(false);
    setTurnStartedAt(null);
    setApprovalState(null);
    setClarifyState(null);
    setTools((items) => items.map((item) => item.state === "running"
      ? { ...item, state: "complete", summary: item.summary ?? "Stopped by user" }
      : item));
  }, [commitLiveAssistantMessage, liveAssistantText, retireActiveTurnIdentity, retireInflightFallbackIdentity, setApprovalState, setClarifyState]);

  useLayoutEffect(() => {
    const updateViewport = () => {
      const element = transcriptRef.current;
      if (!element) return;
      const viewportHeight = element.clientHeight || 600;
      setVirtualViewport((current) => current.viewportHeight === viewportHeight ? current : { ...current, viewportHeight });
    };
    updateViewport();
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  const closeCommandPalette = useCallback(() => setCommandPaletteOpen(false), []);
  const focusComposer = useCallback(() => {
    textareaRef.current?.focus();
    setCommandPaletteOpen(false);
  }, []);
  const toggleSessionNavigator = useCallback(() => setMobileSessionNavigatorOpen((open) => !open), []);

  useEffect(() => {
    const onShortcut = (event: globalThis.KeyboardEvent) => {
      if (event.isComposing) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onShortcut);
    return () => window.removeEventListener("keydown", onShortcut);
  }, []);

  const updateAttachments = useCallback((next: PendingAttachment[] | ((current: PendingAttachment[]) => PendingAttachment[])) => {
    setAttachments((current) => {
      const updated = typeof next === "function" ? next(current) : next;
      attachmentsRef.current = updated;
      return updated;
    });
  }, []);

  const removeAttachment = useCallback((id: string) => {
    uploadControllersRef.current.get(id)?.abort();
    uploadControllersRef.current.delete(id);
    const item = attachmentsRef.current.find((entry) => entry.id === id);
    if (item?.previewUrl && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(item.previewUrl);
    updateAttachments((current) => current.filter((entry) => entry.id !== id));
  }, [updateAttachments]);

  const clearAttachments = useCallback(() => {
    for (const controller of uploadControllersRef.current.values()) controller.abort();
    uploadControllersRef.current.clear();
    for (const item of attachmentsRef.current) {
      if (item.previewUrl && typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(item.previewUrl);
    }
    updateAttachments([]);
  }, [updateAttachments]);

  const stageAttachment = useCallback(async (item: PendingAttachment) => {
    const attachmentSessionId = sessionIdRef.current;
    const attachmentSessionGeneration = sessionGenerationRef.current;
    if (!attachmentSessionId) return;
    if (stagingRef.current.has(item.id)) return;
    stagingRef.current.add(item.id);
    const controller = new AbortController();
    uploadControllersRef.current.set(item.id, controller);
    updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "uploading", error: undefined } : entry));
    try {
      const dataUrl = await fileDataUrl(item.file);
      if (sessionGenerationRef.current !== attachmentSessionGeneration || sessionIdRef.current !== attachmentSessionId) return;
      const result = item.file.type.startsWith("image/")
        ? await gateway.request<{ attached?: boolean; path?: string; ref_path?: string; ref_text?: string }>("image.attach_bytes", {
          session_id: attachmentSessionId, content_base64: dataUrl.slice(dataUrl.indexOf(",") + 1), filename: item.file.name,
        }, 120_000, controller.signal)
        : await gateway.request<{ attached?: boolean; path?: string; ref_path?: string; ref_text?: string }>("file.attach", {
          session_id: attachmentSessionId, name: item.file.name, path: "", data_url: dataUrl,
        }, 120_000, controller.signal);
      if (result.attached === false) throw new Error("Attachment was rejected");
      if (sessionGenerationRef.current !== attachmentSessionGeneration || sessionIdRef.current !== attachmentSessionId) return;
      updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "attached", refText: result.ref_text, refPath: result.ref_path ?? result.path } : entry));
    } catch (reason: unknown) {
      if (controller.signal.aborted || sessionGenerationRef.current !== attachmentSessionGeneration || sessionIdRef.current !== attachmentSessionId) return;
      updateAttachments((current) => current.map((entry) => entry.id === item.id ? { ...entry, state: "error", error: reason instanceof Error ? reason.message : String(reason) } : entry));
    } finally {
      stagingRef.current.delete(item.id);
      uploadControllersRef.current.delete(item.id);
    }
  }, [gateway, updateAttachments]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files);
    const accepted: PendingAttachment[] = [];
    for (const file of incoming) {
      if (file.size > MAX_ATTACHMENT_BYTES) {
        setError(`${file.name} is too large (max 25 MB)`);
        continue;
      }
      accepted.push({
        id: `${Date.now()}-${Math.random()}`,
        file,
        state: "pending",
        previewUrl: file.type.startsWith("image/") && typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : undefined,
      });
    }
    if (!accepted.length) return;
    updateAttachments((current) => [...current, ...accepted]);
  }, [updateAttachments]);

  useEffect(() => {
    if (!sessionId) return;
    for (const item of attachments) {
      if (item.state === "pending") void stageAttachment(item);
    }
  }, [attachments, sessionId, stageAttachment]);

  useEffect(() => {
    const listenerSessionGeneration = ++sessionGenerationRef.current;
    queueDrainInFlightRef.current = false;
    reconnectingRef.current = false;
    resyncBarrierRef.current = null;
    resyncBufferedEventsRef.current = [];
    resyncEventHandlersRef.current.clear();
    replayResyncEventsRef.current = null;
    replayingResyncEventsRef.current = false;
    resyncBufferOverflowRef.current = false;
    initialAttachPendingRef.current = false;
    initialAttachBufferedEventsRef.current = [];
    const isPendingUnboundReplacementDelta = (event: GatewayEvent): boolean => {
      return event.type === "message.delta"
        && eventTurnId(event) === null
        && eventMessageId(event) === null
        && stopInFlightRef.current
        && (stopTargetRef.current !== null || retiredStopTargetRef.current !== null)
        && (eventText(event).length > 0 || assistantIdRef.current === null || activeTurnIdRef.current !== null || activeMessageIdRef.current !== null);
    };
    const acceptTurnEvent = (event: GatewayEvent): boolean => {
      if (sessionGenerationRef.current !== listenerSessionGeneration) return false;
      const activeResyncBarrier = resyncBarrierRef.current;
      if (activeResyncBarrier
        && !replayingResyncEventsRef.current
        && connectionStateRef.current === "open"
        && activeResyncBarrier.connectionEpoch === connectionEpochRef.current
        && activeResyncBarrier.sessionGeneration === sessionGenerationRef.current) {
        if (activeResyncBarrier.operationToken !== operationTokenRef.current
          || activeResyncBarrier.stopRequestGeneration !== stopRequestGenerationRef.current
          || activeResyncBarrier.turnGeneration !== turnGenerationRef.current) {
          // A submit/stop/new turn superseded this snapshot. Handle newer
          // events live instead of buffering them behind the stale barrier.
        } else {
        const handler = resyncEventHandlersRef.current.get(event.type);
        if (handler) {

          if (resyncBufferedEventsRef.current.length < 2048) resyncBufferedEventsRef.current.push({ event, handler });
          else resyncBufferOverflowRef.current = true;
        }
        return false;
        }
      }
      const turnId = eventTurnId(event);
      const messageId = eventMessageId(event);
      const eventPayload = (event.payload ?? {}) as TextPayload;
      const turnScopedError = event.type === "error"
        && Boolean(turnId || messageId || eventPayload.kind === "turn" || eventPayload.turn_scoped === true || eventPayload.error_surface === "turn");
      const terminalEvent = event.type === "message.complete" || turnScopedError;
      const hasExplicitIdentityChange = Boolean(
        (messageId !== null && activeMessageIdRef.current !== null && activeMessageIdRef.current !== messageId)
        || (turnId !== null && activeTurnIdRef.current !== null && activeTurnIdRef.current !== turnId)
        || (assistantIdRef.current !== null
          && activeTurnIdRef.current === null
          && activeMessageIdRef.current === null
          && (turnId !== null || messageId !== null)),
      );
      const replacingExplicitIdentity = TURN_SCOPED_EVENT_TYPES.has(event.type)
        && assistantIdRef.current !== null
        && hasExplicitIdentityChange;
      const ambiguousUnboundTurnError = turnScopedError
        && turnId === null
        && messageId === null
        && activePromptTextRef.current === null
        && assistantIdRef.current !== null
        && (event.type === "error" || !unboundStartAcceptedRef.current);
      const durableMessageTarget = Boolean(messageId && transcriptStateRef.current.some((message) => message.role === "assistant" && (message.id === messageId || message.messageId === messageId)));
      const durableErrorTerminal = event.type === "error"
        || eventPayload.status === "error"
        || (typeof eventPayload.error === "string" && eventPayload.error.length > 0);
      if (messageId && retiredMessageIdsRef.current.has(messageIdentityKey(event.session_id, messageId, durableSessionIdRef.current, sessionKeyRef.current))
        && !(terminalEvent && durableMessageTarget && durableErrorTerminal)) return false;
      if (turnId && !messageId && retiredTurnIdsRef.current.has(turnId)) {
        const priorText = retiredTurnTextsRef.current.get(turnId);
        const currentText = assistantIdRef.current ? liveAssistantText(assistantIdRef.current) : "";
        const incomingText = eventText(event);
        const replacementMessageId = replacementTurnMessageIdsRef.current.get(turnId);
        const currentMessageIsReplacement = replacementMessageId === undefined
          ? activeMessageIdRef.current !== null
          : activeMessageIdRef.current === replacementMessageId;
        const incomingMatchesRetiredText = Boolean(
          priorText
          && incomingText.length > 0
          && (priorText.startsWith(incomingText)),
        );
        const contentProvesReplacement = event.type === "message.delta"
          ? activeTurnIdRef.current === turnId
            && currentMessageIsReplacement
            && incomingText.length > 0
            && !incomingMatchesRetiredText
          : event.type === "message.complete"
            && activeTurnIdRef.current === turnId
            && currentMessageIsReplacement
            && currentText.length > 0
            && (incomingText.length === 0
              ? replacementTurnProvenByDeltaRef.current.has(turnId)
              : !incomingMatchesRetiredText
                && incomingText !== priorText
                && (currentText === incomingText || currentText.startsWith(incomingText) || incomingText.startsWith(currentText)));
        if (!contentProvesReplacement) return false;
      }
      const stopTarget = stopTargetRef.current;
      const stopTargetMatches = stopInFlightRef.current
        && stopTarget !== null
        && stopTargetSessionMatches(stopTarget, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
        && stopTarget.sessionGeneration === sessionGenerationRef.current
        && stopTarget.requestGeneration === stopRequestGenerationRef.current
        && stopTarget.assistantId === assistantIdRef.current
        && stopTarget.turnGeneration === turnGenerationRef.current
        && stopTarget.turnId === activeTurnIdRef.current
        && (!turnId || stopTarget.turnId === turnId)
        && (!messageId || stopTarget.messageId === messageId);
      const pendingStopTerminal = terminalEvent && stopTargetMatches;
      const staleUnboundStopTerminal = terminalEvent
        && stopInFlightRef.current
        && stopTarget !== null
        && !stopTargetMatches
        && !turnId
        && !replacingExplicitIdentity
        && !(messageId && activeMessageIdRef.current === messageId)
        && !unboundStreamEstablishedRef.current
        && !unboundStartAcceptedRef.current;
      const postRetirementUnboundStartEvent = blockedTurnGenerationRef.current === turnGenerationRef.current
        && !turnId
        && !messageId
        && !activePromptTextRef.current
        && !allowUnboundStartAfterRetirementRef.current
        && !postRetirementUnboundStartPendingRef.current
        && !stopInFlightRef.current
        && retiredStopTargetRef.current === null
        && !isPendingUnboundReplacementDelta(event)
        && event.type === "message.start";
      const postRetirementUnboundPreStartEvent = blockedTurnGenerationRef.current === turnGenerationRef.current
        && !turnId
        && !messageId
        && !activePromptTextRef.current
        && !unboundStartAcceptedRef.current
        && !(event.type === "message.delta"
          && postRetirementUnboundStartPendingRef.current
          && unboundStopFenceRef.current !== null)
        && !isPendingUnboundReplacementDelta(event)
        && (event.type === "message.delta"
          || event.type === "message.interim"
          || event.type === "thinking.delta"
          || event.type === "reasoning.delta"
          || event.type === "tool.generating"
          || event.type === "tool.start"
          || event.type === "tool.progress"
          || event.type === "tool.complete"
          || event.type === "approval.request"
          || event.type === "clarify.request");
      const staleUnboundReplacementTerminal = terminalEvent
        && turnId === null
        && messageId === null
        && assistantIdRef.current !== null
        && unboundStopFenceRef.current !== null
        && (() => {
          const terminalText = eventText(event);
          const stoppedText = unboundStopFenceRef.current?.assistantText ?? "";
          return (!terminalText.length && !activePromptTextRef.current)
            || (terminalText.length > 0
              && stoppedText.length > 0
              && stoppedText.startsWith(terminalText));
        })();
      const staleUnboundReplacementDelta = event.type === "message.delta"
        && turnId === null
        && messageId === null
        && postRetirementUnboundStartPendingRef.current
        && unboundStopFenceRef.current !== null
        && (() => {
          const deltaText = eventText(event);
          const stoppedText = unboundStopFenceRef.current?.assistantText ?? "";
          return deltaText.length > 0
            && stoppedText.length > 0
            && stoppedText.startsWith(deltaText);
        })();
      if (staleUnboundReplacementDelta) return false;
      if (staleUnboundReplacementTerminal) return false;
      if (staleUnboundStopTerminal) return false;
      if (ambiguousUnboundTurnError) return false;
      if (postRetirementUnboundStartEvent || postRetirementUnboundPreStartEvent) return false;
      const retiredStopTarget = retiredStopTargetRef.current;
      const retiredStopTargetMatchesCurrentMessage = Boolean(
        terminalEvent
        && messageId
        && activeMessageIdRef.current
        && messageId === activeMessageIdRef.current
        && messageId !== retiredStopTarget?.messageId,
      );
      if (
        terminalEvent
        && !turnId
        && retiredStopTarget !== null
        && stopTargetSessionMatches(retiredStopTarget, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
        && retiredStopTarget.sessionGeneration === sessionGenerationRef.current
        && retiredStopTarget.requestGeneration <= stopRequestGenerationRef.current
        && !unboundStartAcceptedRef.current
        && !retiredStopTargetMatchesCurrentMessage
      ) return false;
      if (!TURN_SCOPED_EVENT_TYPES.has(event.type) && !pendingStopTerminal) {
        if (turnId && ignoredTurnIdsRef.current.has(turnId)) return false;
        if (turnId && activeTurnIdRef.current && activeTurnIdRef.current !== turnId) return false;
        return true;
      }
      if (turnId && ignoredTurnIdsRef.current.has(turnId) && !pendingStopTerminal
        && !(messageId && durableMessageTarget && durableErrorTerminal)) return false;
      if (!pendingStopTerminal && event.type !== "message.start" && activeTurnIdRef.current && turnId && activeTurnIdRef.current !== turnId && !replacingExplicitIdentity) return false;
      if (!pendingStopTerminal && event.type !== "message.start" && messageId && activeMessageIdRef.current && activeMessageIdRef.current !== messageId && !replacingExplicitIdentity) return false;
      if (!pendingStopTerminal && terminalEvent && !turnId && !messageId && blockedTurnGenerationRef.current === turnGenerationRef.current && !unboundStreamEstablishedRef.current && !unboundStartAcceptedRef.current) return false;
      return true;
    };
    const rememberRejectedEvent = (event: GatewayEvent) => {
      const payload = (event.payload ?? {}) as TextPayload;
      const rawEventId = (event as GatewayEvent & { event_id?: unknown }).event_id ?? payload.event_id ?? payload.eventId;
      const scope = sessionKeyRef.current ?? durableSessionIdRef.current ?? event.session_id ?? sessionIdRef.current ?? "global";
      const eventKey = typeof rawEventId === "string" && rawEventId ? buildScopedIdentityKey(scope, rawEventId) : undefined;
      if (eventKey && seenEventIdsRef.current.has(eventKey)) return;
      if (eventKey) seenEventIdsRef.current.add(eventKey);
      const rawSeq = (event as GatewayEvent & { seq?: unknown }).seq ?? payload.seq;
      if (event.session_id && typeof rawSeq === "number") {
        const previousSeq = seenSeqRef.current.get(event.session_id);
        if (previousSeq === undefined || rawSeq > previousSeq) seenSeqRef.current.set(event.session_id, rawSeq);
      }
    };
    const transcriptIdForMessageId = (messageId: string): string => transcriptStateRef.current.find((message) => message.role === "assistant" && (message.id === messageId || message.messageId === messageId))?.id ?? messageId;
    const accept = (event: GatewayEvent, sessionBound = false): TextPayload | null => {
      if (sessionBound && !sessionIdRef.current) {
        if (!initialAttachPendingRef.current || !event.session_id) return null;
        const handler = resyncEventHandlersRef.current.get(event.type);
        if (handler && initialAttachBufferedEventsRef.current.length < 2048) {
          initialAttachBufferedEventsRef.current.push({ event, handler });
        }
        return null;
      }
      if (sessionBound && event.session_id !== sessionIdRef.current) return null;
      if (!sessionBound && sessionIdRef.current && event.session_id && event.session_id !== sessionIdRef.current) return null;
      const payload = (event.payload ?? {}) as TextPayload;
      const seq = typeof (event as GatewayEvent & { seq?: unknown }).seq === "number"
        ? (event as GatewayEvent & { seq?: number }).seq
        : payload.seq;
      const eventId = (event as GatewayEvent & { event_id?: unknown }).event_id
        ?? payload.event_id
        ?? payload.eventId;
      const eventScope = sessionKeyRef.current ?? durableSessionIdRef.current ?? event.session_id ?? sessionIdRef.current ?? "global";
      const eventKey = typeof eventId === "string" && eventId ? buildScopedIdentityKey(eventScope, eventId) : undefined;
      if (eventKey && seenEventIdsRef.current.has(eventKey)) return null;
      const bufferedLengthBeforeAdmission = resyncBufferedEventsRef.current.length;
      if (!acceptTurnEvent(event)) {
        const wasBuffered = resyncBufferedEventsRef.current.length > bufferedLengthBeforeAdmission
          && resyncBufferedEventsRef.current[bufferedLengthBeforeAdmission]?.event === event;
        if (!wasBuffered) rememberRejectedEvent(event);
        return null;
      }
      const sequenceScope = event.session_id ?? sessionIdRef.current ?? "global";
      if (event.session_id && typeof seq === "number") {
        const previous = seenSeqRef.current.get(sequenceScope);
        const allowReplaySequenceReset = replayingResyncEventsRef.current && eventKey !== undefined;
        if (previous !== undefined && seq <= previous && !allowReplaySequenceReset) {
          if (eventKey) seenEventIdsRef.current.add(eventKey);
          return null;
        }
        if (previous === undefined || seq > previous) seenSeqRef.current.set(sequenceScope, seq);
      }
      if (eventKey) seenEventIdsRef.current.add(eventKey);
      const acceptedTurnId = eventTurnId(event);
      if (event.type === "message.delta"
        && acceptedTurnId !== null
        && replacementTurnMessageIdsRef.current.get(acceptedTurnId) === activeMessageIdRef.current
        && eventText(event).length > 0) {
        replacementTurnProvenByDeltaRef.current.add(acceptedTurnId);
      }
      return payload;
    };
    const applySessionSnapshot = (snapshot: ResumeResponse, expectedBarrier?: ResyncBarrier): boolean => {
      if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration || !snapshotMatchesSession(snapshot, sessionIdRef.current)) return false;
      if (expectedBarrier && (
        resyncBarrierRef.current !== expectedBarrier
        || sessionGenerationRef.current !== expectedBarrier.sessionGeneration
        || operationTokenRef.current !== expectedBarrier.operationToken
        || stopRequestGenerationRef.current !== expectedBarrier.stopRequestGeneration
        || turnGenerationRef.current !== expectedBarrier.turnGeneration
        || connectionEpochRef.current !== expectedBarrier.connectionEpoch
      )) return false;
      const info = snapshot.info ?? snapshot;
      const snapshotSessionKey = snapshot.session_key ?? info.session_key;
      const currentDurableIds = [durableSessionIdRef.current, sessionKeyRef.current, ...validatedDurableAliasesRef.current].filter((value): value is string => typeof value === "string" && value.length > 0);
      const identityValidation = validateDurableIdentityResponse(
        snapshotSessionKey,
        [snapshot.stored_session_id, info.stored_session_id],
        currentDurableIds,
      );
      if (!identityValidation.accepted) return false;
      const returnedDurableId = identityValidation.canonicalId;
      for (const alias of identityValidation.validatedIds) validatedDurableAliasesRef.current.add(alias);
      const previousScope = sessionKeyRef.current ?? durableSessionIdRef.current ?? sessionIdRef.current;
      const canonicalScope = sessionKeyRef.current ?? durableSessionIdRef.current;
      const nextScope = snapshotSessionKey ?? canonicalScope ?? returnedDurableId ?? sessionIdRef.current;
      if (sessionIdRef.current && previousScope && nextScope && previousScope !== nextScope) {
        rekeyScopedSet(seenEventIdsRef.current, previousScope, nextScope);
        rekeyScopedSet(retiredMessageIdsRef.current, previousScope, nextScope);
        const fallbackEntries = [...inflightFallbackIdentityMapRef.current.entries()]
          .map(([key, identity]) => ({ oldKey: key, newKey: rekeyInflightFallbackKey(key, previousScope, nextScope), identity }))
          .filter((entry): entry is { oldKey: string; newKey: string; identity: string } => entry.newKey !== null);
        for (const { oldKey, newKey, identity } of fallbackEntries) {
          inflightFallbackIdentityMapRef.current.delete(oldKey);
          inflightFallbackIdentityMapRef.current.set(newKey, identity);
        }
        const activeFallbackKey = activeInflightFallbackKeyRef.current;
        const rekeyedActiveFallbackKey = activeFallbackKey
          ? rekeyInflightFallbackKey(activeFallbackKey, previousScope, nextScope)
          : null;
        if (rekeyedActiveFallbackKey) activeInflightFallbackKeyRef.current = rekeyedActiveFallbackKey;
        stopTargetRef.current = rekeyStopTargetScope(stopTargetRef.current, previousScope, nextScope);
        retiredStopTargetRef.current = rekeyStopTargetScope(retiredStopTargetRef.current, previousScope, nextScope);
        unboundStopFenceRef.current = rekeyStopTargetScope(unboundStopFenceRef.current, previousScope, nextScope);
        rekeyPinnedArtifacts(profile, previousScope, nextScope);
        dispatchTimeline({ type: "rebind-session", fromSession: sessionIdRef.current, toSession: sessionIdRef.current, fromScope: previousScope ?? undefined, toScope: nextScope ?? undefined });
      }
      if (typeof snapshotSessionKey === "string" && snapshotSessionKey) sessionKeyRef.current = snapshotSessionKey;
      const storedId = snapshotSessionKey ?? sessionKeyRef.current ?? durableSessionIdRef.current ?? snapshot.stored_session_id ?? info.stored_session_id;
      if (typeof storedId === "string" && storedId) {
        durableSessionIdRef.current = storedId;
        setDurableSessionId(storedId);
      }
      const running = typeof info.running === "boolean"
        ? info.running
        : typeof snapshot.running === "boolean" ? snapshot.running : undefined;
      const hasRunning = snapshotHasField(info, "running") || snapshotHasField(snapshot, "running");
      const inflight = snapshot.inflight && typeof snapshot.inflight === "object" ? snapshot.inflight : null;
      const inflightTurnId = inflightIdentity(inflight?.turn_id ?? inflight?.turnId);
      const inflightMessageId = inflightIdentity(inflight?.message_id ?? inflight?.messageId);
      const inflightIsRunning = Boolean(inflight && (running === true || (running === undefined && inflight.streaming === true)));
      const runtimeSessionId = sessionIdRef.current;
      if (runtimeSessionId && Array.isArray(snapshot.messages)) {
        for (const message of snapshot.messages) {
          if (message.role !== "assistant" || message.streaming === true || message.interim === true) continue;
          const historicalMessageId = inflightIdentity(message.message_id ?? message.messageId ?? message.id);
          const historicalTurnId = inflightIdentity(message.turn_id ?? message.turnId);
          const matchesRunningInflight = inflightIsRunning
            && (historicalMessageId !== null || historicalTurnId !== null)
            && (historicalMessageId === null || historicalMessageId === inflightMessageId)
            && (historicalTurnId === null || historicalTurnId === inflightTurnId);
          if (!matchesRunningInflight) {
            if (historicalMessageId) retiredMessageIdsRef.current.add(messageIdentityKey(runtimeSessionId, historicalMessageId, durableSessionIdRef.current, sessionKeyRef.current));
            if (historicalTurnId) ignoredTurnIdsRef.current.add(historicalTurnId);
          }
        }
      }
      const hasInflightIdentity = Boolean(inflight && (
        inflightIdentity(inflight.message_id ?? inflight.messageId) !== null
        || inflightIdentity(inflight.turn_id ?? inflight.turnId) !== null
        || typeof inflight.started_at === "number"
      ));
      let inflightFallbackKey: string | null = null;
      if (inflight && !hasInflightIdentity) {
        const fallbackScope = sessionKeyRef.current ?? durableSessionIdRef.current ?? runtimeSessionId ?? "session";
        const fallbackUser = inflightText(inflight.user);
        const fallbackStatus = typeof inflight.status === "string" ? inflight.status : "";
        const fallbackError = typeof inflight.error === "string" ? inflight.error : "";
        const fallbackBaseKey = buildInflightFallbackKey(fallbackScope, fallbackUser, fallbackStatus, fallbackError);
        const activeFallbackKey = activeInflightFallbackKeyRef.current;
        const activeAssistantText = assistantIdRef.current ? liveAssistantText(assistantIdRef.current) : "";
        const incomingAssistantText = inflightText(inflight.assistant);
        const activeKeyMatchesBase = activeFallbackKey !== null && sameInflightFallbackBase(activeFallbackKey, fallbackBaseKey);
        const textContinuesActiveTurn = shouldMergeIdentityLessInflight(
          activeAssistantText,
          incomingAssistantText,
          activePromptTextRef.current,
          inflightLatestPrompt(inflight),
        );
        inflightFallbackKey = activeKeyMatchesBase && textContinuesActiveTurn
          ? activeFallbackKey ?? fallbackBaseKey
          : activeKeyMatchesBase
            ? buildInflightFallbackKey(fallbackScope, fallbackUser, fallbackStatus, fallbackError, ++inflightSnapshotSequenceRef.current)
            : fallbackBaseKey;
      }
      let inflightFallbackIdentity = "active";
      if (inflightFallbackKey) {
        const existing = inflightFallbackIdentityMapRef.current.get(inflightFallbackKey);
        if (existing) inflightFallbackIdentity = existing;
        else {
          inflightFallbackIdentity = `snapshot-${++inflightSnapshotSequenceRef.current}`;
          inflightFallbackIdentityMapRef.current.set(inflightFallbackKey, inflightFallbackIdentity);
          while (inflightFallbackIdentityMapRef.current.size > 128) {
            const oldest = inflightFallbackIdentityMapRef.current.keys().next().value;
            if (typeof oldest !== "string") break;
            inflightFallbackIdentityMapRef.current.delete(oldest);
          }
        }
      }
      const inflightMatchesActiveIdentity = Boolean(
        (inflightTurnId !== null && activeTurnIdRef.current === inflightTurnId)
        || (inflightMessageId !== null && activeMessageIdRef.current === inflightMessageId)
      );
      const inflightRows = inflight && runtimeSessionId
        ? inflightTranscript(inflight, sessionKeyRef.current ?? durableSessionIdRef.current ?? runtimeSessionId, inflightFallbackIdentity)
        : [];
      const inflightRunning = inflight && typeof inflight.streaming === "boolean"
        ? inflight.streaming
        : undefined;
      const effectiveRunning = running ?? (inflight ? inflightRunning : undefined);
      const inflightError = inflight && typeof inflight.error === "string" && inflight.error
        ? inflight.error
        : inflight?.status === "error" ? "Turn failed" : null;
      const hasSnapshotMessages = Array.isArray(snapshot.messages)
        && (snapshot.messages_omitted !== true || snapshot.messages.length > 0);
      const inflightPromptCandidate = inflightText(inflight?.user);
      const candidateInflightPrompt = inflightPromptCandidate && Array.isArray(snapshot.messages) && snapshot.messages.some((m) => m.role === "user" && snapshotText(m) === inflightPromptCandidate)
        ? (inflightPromptCandidate || inflightLatestPrompt(inflight!) || null)
        : null;
      const activeUserTextForInflightMerge = activePromptTextRef.current ?? candidateInflightPrompt;
      if (effectiveRunning !== undefined) setStreaming(effectiveRunning);
      if (snapshotHasField(info, "turn_started_at") || snapshotHasField(snapshot, "turn_started_at")) {
        const startedAt = info.turn_started_at ?? snapshot.turn_started_at;
        setTurnStartedAt(typeof startedAt === "number" ? startedAt * 1000 : null);
      } else if (typeof inflight?.started_at === "number") {
        setTurnStartedAt(inflight.started_at * 1000);
      } else if (effectiveRunning === false) {
        setTurnStartedAt(null);
      }
      const snapshotStatus = typeof info.status === "string" && info.status ? info.status : snapshot.status;
      const idleSnapshot = !hasRunning
        && !inflight
        && (snapshotStatus === "idle" || snapshotStatus === "ready")
        && Array.isArray(snapshot.messages)
        && snapshot.messages_omitted !== true;
      const errorStatusSnapshot = !hasRunning && !inflight && snapshotStatus === "error";
      if (typeof snapshotStatus === "string" && snapshotStatus) setStatus(snapshotStatus);
      else if (hasRunning && effectiveRunning === false) setStatus("Ready");
      if (hasSnapshotMessages) {
        const next = snapshotTranscript(snapshot.messages);
        const coverage = snapshot.messages_omitted === true ? "tail" : "prefix";
        setTranscript((current) => {
          const merged = mergeSnapshotTranscript(next, current, coverage);
          return inflightRows.length > 0 && !inflightMatchesActiveIdentity
            ? mergeInflightTranscript(merged, inflightRows, activeUserTextForInflightMerge)
            : merged;
        });
      } else if (inflightRows.length > 0 && !inflightMatchesActiveIdentity) {
        setTranscript((current) => mergeInflightTranscript(current, inflightRows, activeUserTextForInflightMerge));
      }
      if (inflightRows.length && effectiveRunning && inflight && !inflightError) {
        const assistantRow = [...inflightRows].reverse().find((row) => row.role === "assistant");
        const inflightUserPrompt = inflightText(inflight.user);
        const inflightPrompt = inflightLatestPrompt(inflight);
        const retiredStopTarget = retiredStopTargetRef.current;
        const isDistinctReplacement = retiredStopTarget !== null && (
          (inflightTurnId !== null && retiredStopTarget.turnId !== null && inflightTurnId !== retiredStopTarget.turnId)
          || (inflightMessageId !== null && retiredStopTarget.messageId !== null && inflightMessageId !== retiredStopTarget.messageId)
          || (typeof inflight.started_at === "number" && inflight.started_at * 1000 >= retiredStopTarget.stoppedAt)
          || (inflightUserPrompt !== "" && retiredStopTarget.promptText !== null && inflightUserPrompt !== retiredStopTarget.promptText)
        );
        const previousAssistantId = assistantIdRef.current;
        const previousTurnId = activeTurnIdRef.current;
        const previousMessageId = activeMessageIdRef.current;
        const previousPromptText = activePromptTextRef.current;
        const hasExplicitInflightIdentity = inflightTurnId !== null || inflightMessageId !== null;
        const sameTurnMessageReplacement = previousAssistantId !== null
          && inflightTurnId !== null
          && previousTurnId === inflightTurnId
          && inflightMessageId !== null
          && previousMessageId !== inflightMessageId;
        const identityConflict = (
          inflightTurnId !== null
          && previousTurnId !== null
          && inflightTurnId !== previousTurnId
        ) || (
          inflightMessageId !== null
          && previousMessageId !== null
          && inflightMessageId !== previousMessageId
        );
        const identityMatch = (
          (inflightTurnId !== null && previousTurnId === inflightTurnId)
          || (inflightMessageId !== null && previousMessageId === inflightMessageId)
        );
        const sameActiveTurn = previousAssistantId !== null && !identityConflict && !sameTurnMessageReplacement && (
          identityMatch
          || (!hasExplicitInflightIdentity && inflightUserPrompt !== "" && previousPromptText === inflightUserPrompt)
          || (!hasExplicitInflightIdentity && assistantRow?.id === previousAssistantId)
        );
        if (previousAssistantId && !sameActiveTurn) {
          retireInflightFallbackIdentity();
          retireActiveTurnIdentity();
          if (sameTurnMessageReplacement && inflightTurnId !== null && inflightMessageId !== null) {
            ignoredTurnIdsRef.current.delete(inflightTurnId);
            retiredTurnIdsRef.current.add(inflightTurnId);
            replacementTurnMessageIdsRef.current.set(inflightTurnId, inflightMessageId);
            replacementTurnProvenByDeltaRef.current.delete(inflightTurnId);
          }
          const previousText = liveAssistantText(previousAssistantId);
          if (sameTurnMessageReplacement && inflightTurnId !== null && previousText) retiredTurnTextsRef.current.set(inflightTurnId, previousText);
          if (previousText) setTranscript((current) => mergeCompletedAssistantMessage(current, previousAssistantId, previousText, previousPromptText) as TranscriptMessage[]);
          else dispatchTimeline({ type: "reset" });
          activePromptTextRef.current = null;
          turnGenerationRef.current += 1;
        }
        if (sameActiveTurn && previousAssistantId) {
          const liveText = liveAssistantText(previousAssistantId);
          const reconciledText = compatibleLiveText(liveText, assistantRow?.text ?? "");
          if (runtimeSessionId && reconciledText.length > liveText.length && reconciledText.startsWith(liveText)) {
            dispatchTimeline({
              type: "update",
              event: {
                type: "message.delta",
                session_id: runtimeSessionId,
                session_key: sessionKeyRef.current ?? undefined,
                payload: {
                  text: reconciledText.slice(liveText.length),
                  ...(inflightTurnId ? { turn_id: inflightTurnId } : {}),
                  ...(inflightMessageId ? { message_id: inflightMessageId } : {}),
                },
              },
              entryId: previousAssistantId,
            });
          }
          assistantIdRef.current = previousAssistantId;
          activeTurnIdRef.current = inflightTurnId ?? previousTurnId;
          activeMessageIdRef.current = inflightMessageId ?? previousMessageId;
          blockedTurnGenerationRef.current = null;
          if (inflightPrompt) activePromptTextRef.current = inflightPrompt;
          activeInflightFallbackKeyRef.current = inflightFallbackKey;
        } else {
          if (isDistinctReplacement) retiredStopTargetRef.current = null;
          dispatchTimeline({ type: "reset" });
          assistantIdRef.current = assistantRow?.id ?? null;
          activeTurnIdRef.current = inflightTurnId;
          activeMessageIdRef.current = inflightMessageId;
          blockedTurnGenerationRef.current = null;
          if (assistantRow && runtimeSessionId) {
            dispatchTimeline({
              type: "append",
              event: {
                type: "message.start",
                session_id: runtimeSessionId,
                session_key: sessionKeyRef.current ?? undefined,
                payload: {
                  text: assistantRow.text,
                  ...(inflightTurnId ? { turn_id: inflightTurnId } : {}),
                  ...(inflightMessageId ? { message_id: inflightMessageId } : {}),
                },
              },
              entryId: assistantRow.id,
            });
          }
          if (inflightPrompt) activePromptTextRef.current = inflightPrompt;
          activeInflightFallbackKeyRef.current = inflightFallbackKey;
        }
      } else if ((hasRunning && effectiveRunning === false) || Boolean(inflightError) || idleSnapshot || errorStatusSnapshot) {
        const preserveLive = !idleSnapshot && (snapshot.messages_omitted === true
          || !Array.isArray(snapshot.messages)
          || (snapshot.messages.length === 0 && !snapshotHasField(snapshot, "messages_omitted")));
        const oldAssistantId = assistantIdRef.current;
        const activePromptText = activePromptTextRef.current;
        const inflightErrorMatchesActiveTurn = Boolean(inflight && inflightError && (
          (inflightTurnId !== null && activeTurnIdRef.current === inflightTurnId)
          || (inflightMessageId !== null && activeMessageIdRef.current === inflightMessageId)
          || (inflightTurnId === null && inflightMessageId === null
            && activePromptText !== null
            && inflightLatestPrompt(inflight) === activePromptText)
        ));
        retireInflightFallbackIdentity();
        setStreaming(false);
        setTurnStartedAt(null);
        retireActiveTurnIdentity();
        if (preserveLive && oldAssistantId) {
          const oldText = liveAssistantText(oldAssistantId);
          if (oldText) setTranscript((current) => mergeCompletedAssistantMessage(
            current,
            oldAssistantId,
            oldText,
            activePromptText,
            inflightErrorMatchesActiveTurn ? inflightError ?? undefined : undefined,
          ) as TranscriptMessage[]);
        }
        assistantIdRef.current = null;
        activeTurnIdRef.current = null;
        activeMessageIdRef.current = null;
        activePromptTextRef.current = null;
        turnGenerationRef.current += 1;
        blockedTurnGenerationRef.current = turnGenerationRef.current;
        dispatchTimeline({ type: "reset" });
        setTools((items) => items.map((item) => item.state === "running"
          ? { ...item, state: "complete", summary: item.summary ?? "Session is idle" }
          : item));
      }
      if (snapshotStatus === "error") setResyncState("error");
      else if (snapshot.messages_omitted === true) setResyncState("partial");
      else setResyncState("synced");
      // Only replace pending registries when the backend explicitly sends the
      // field. An older snapshot without these fields must not erase a live
      // approval/clarification that arrived after the snapshot was requested.
      if (snapshotHasField(snapshot, "pending_approval")) {
        const pendingApproval = approvalFromSnapshot(snapshot.pending_approval);
        if (pendingApproval) setApprovalState(pendingApproval);
        else if (snapshot.pending_approval === null) setApprovalState(null);
      }
      if (snapshotHasField(snapshot, "pending_clarify")) {
        setClarifyState(clarifyFromSnapshot(snapshot.pending_clarify));
      }
      if (hasRunning && effectiveRunning === false
        && !snapshotHasField(snapshot, "pending_approval")
        && !snapshotHasField(snapshot, "pending_clarify")) {
        setApprovalState(null);
        setClarifyState(null);
      }
      if (inflight && inflightError) {
        const promptText = inflightLatestPrompt(inflight);
        setError(inflightError);
        setErrorAction(promptText ? "resend" : null);
        if (promptText) setFailedPrompt({ id: `retry-${runtimeSessionId ?? "session"}-${inflightIdentity(inflight?.turn_id ?? inflight?.turnId) ?? "inflight"}`, text: promptText, mode: "retry" });
        setStatus("Error");
      } else if (snapshotStatus === "error") {
        const snapshotError = typeof info.error === "string" && info.error
          ? info.error
          : typeof snapshot.error === "string" && snapshot.error ? snapshot.error : "Session reported an error";
        setError(snapshotError);
        setErrorAction(null);
      } else if (effectiveRunning === false || idleSnapshot) {
        setError(null);
        setErrorAction(null);
      } else if (effectiveRunning === true || Boolean(inflight)) {
        setError(null);
        setErrorAction(null);
        setFailedPrompt(null);
      }
      return true;
    };
    let cancelled = false;
    const scheduleReconnect = () => {
      if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration || !wasOpenRef.current || reconnectTimerRef.current !== null || reconnectInFlightRef.current) return;
      const attempt = Math.min(reconnectAttemptRef.current + 1, 5);
      reconnectAttemptRef.current = attempt;
      const delayMs = Math.min(250 * 2 ** (attempt - 1), 3000);
      setStatus("Reconnecting…");
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration) return;
        const reconnectRequestToken = ++reconnectRequestTokenRef.current;
        reconnectInFlightRef.current = true;
        let connected = false;
        void gateway.connect()
          .then(() => { connected = true; })
          .catch((reason: unknown) => {
            if (!cancelled && sessionGenerationRef.current === listenerSessionGeneration) {
              setError(reason instanceof Error ? reason.message : String(reason));
              setErrorAction("reconnect");
            }
          })
          .finally(() => {
            if (sessionGenerationRef.current === listenerSessionGeneration
              && reconnectRequestTokenRef.current === reconnectRequestToken) reconnectInFlightRef.current = false;
            if (!cancelled && sessionGenerationRef.current === listenerSessionGeneration && !connected) scheduleReconnect();
          });
      }, delayMs);
    };
    const startResync = () => {
      if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration || reconnectingRef.current || connectionStateRef.current !== "open") return;
      if (stopInFlightRef.current) {
        resyncAfterStopRef.current = true;
        setResyncState("idle");
        return;
      }
      resyncAfterStopRef.current = false;
      const sid = sessionIdRef.current;
      if (!sid) return;
      reconnectingRef.current = true;
      const resyncBarrier: ResyncBarrier = {
        sessionGeneration: listenerSessionGeneration,
        operationToken: operationTokenRef.current,
        stopRequestGeneration: stopRequestGenerationRef.current,
        turnGeneration: turnGenerationRef.current,
        connectionEpoch: connectionEpochRef.current,
      };
      resyncBarrierRef.current = resyncBarrier;
      resyncBufferedEventsRef.current = [];
      resyncBufferOverflowRef.current = false;
      setResyncState("syncing");
      setStatus("Syncing…");
      void (async () => {
        let snapshotApplied = false;
        let resyncFailed = false;
        const barrierIsCurrent = () => !cancelled
          && sessionGenerationRef.current === resyncBarrier.sessionGeneration
          && resyncBarrierRef.current === resyncBarrier
          && operationTokenRef.current === resyncBarrier.operationToken
          && stopRequestGenerationRef.current === resyncBarrier.stopRequestGeneration
          && turnGenerationRef.current === resyncBarrier.turnGeneration
          && connectionEpochRef.current === resyncBarrier.connectionEpoch;
        const barrierOwnsSnapshotResult = () => !cancelled
          && resyncBarrierRef.current === resyncBarrier
          && sessionGenerationRef.current === resyncBarrier.sessionGeneration
          && operationTokenRef.current === resyncBarrier.operationToken
          && stopRequestGenerationRef.current === resyncBarrier.stopRequestGeneration
          && turnGenerationRef.current === resyncBarrier.turnGeneration
          && connectionEpochRef.current === resyncBarrier.connectionEpoch;
        try {
          if (!barrierIsCurrent()) return;
          let snapshot: ResumeResponse;
          try {
            snapshot = await gateway.request<ResumeResponse>("session.activate", reconnectActivateParams(sid, profile));

          } catch {
            if (!barrierIsCurrent()) return;
            snapshot = await gateway.request<ResumeResponse>("session.resume", { session_id: durableSessionIdRef.current ?? sid, omit_messages: false, continue_on_disconnect: true, ...(profile ? { profile } : {}) });
          }
          if (!barrierIsCurrent()) return;
          const returnedSessionKey = snapshot.session_key ?? snapshot.info?.session_key;
          const returnedStoredSessionId = snapshot.stored_session_id ?? snapshot.info?.stored_session_id;
          const returnedDurableIds = [returnedSessionKey, returnedStoredSessionId].filter((value): value is string => typeof value === "string" && value.length > 0);
          const currentDurableIds = [durableSessionIdRef.current, sessionKeyRef.current, ...validatedDurableAliasesRef.current].filter((value): value is string => typeof value === "string" && value.length > 0);
          const identityValidation = validateDurableIdentityResponse(returnedSessionKey, [returnedStoredSessionId], currentDurableIds);
          if (!identityValidation.accepted) {
            throw new Error(returnedDurableIds.length === 0
              ? "Gateway returned no durable session identity during reconnect"
              : "Gateway returned a different durable session during reconnect");
          }
          for (const alias of identityValidation.validatedIds) validatedDurableAliasesRef.current.add(alias);
          const sequenceSessionId = snapshot.session_id ?? sid;
          if (snapshot.session_id && snapshot.session_id !== sessionIdRef.current) {
            const previousRuntimeId = sessionIdRef.current;
            const previousScope = sessionKeyRef.current ?? durableSessionIdRef.current ?? previousRuntimeId;
            const nextScope = snapshot.session_key ?? snapshot.info?.session_key ?? durableSessionIdRef.current ?? snapshot.session_id;
            rekeyScopedSet(seenEventIdsRef.current, previousScope, nextScope);
            rekeyScopedSet(retiredMessageIdsRef.current, previousScope, nextScope);
            rekeyPinnedArtifacts(profile, previousScope, nextScope);
            if (previousRuntimeId) dispatchTimeline({ type: "rebind-session", fromSession: previousRuntimeId, toSession: snapshot.session_id, fromScope: previousScope ?? undefined, toScope: nextScope ?? undefined });
            sessionIdRef.current = snapshot.session_id;
            setSessionId(snapshot.session_id);
          }
          snapshotApplied = applySessionSnapshot(snapshot, resyncBarrier);
          if (!snapshotApplied && barrierIsCurrent()) throw new Error("Gateway returned a conflicting durable session");
          if (snapshotApplied && barrierOwnsSnapshotResult()) {
            if (resyncBufferOverflowRef.current) throw new Error("Resync event buffer overflow");
            replayResyncEventsRef.current?.();
            seenSeqRef.current.delete(sequenceSessionId);
            dispatchTimeline({ type: "reset-sequence", session: sequenceSessionId });
          }
        } catch (reason: unknown) {
          resyncFailed = true;
          if (barrierIsCurrent()) {
            setResyncState("error");
            setError(reason instanceof Error ? reason.message : String(reason));
            setErrorAction("reconnect");
            setStatus("Resync failed");
          }
        } finally {
          const barrierStillOwned = resyncBarrierRef.current === resyncBarrier;
          if (resyncFailed || !snapshotApplied || !barrierStillOwned) {
            resyncBufferedEventsRef.current = [];
            resyncBufferOverflowRef.current = false;
          }
          const invalidated = !snapshotApplied && !barrierIsCurrent();
          const operationInvalidated = operationTokenRef.current !== resyncBarrier.operationToken
            || stopRequestGenerationRef.current !== resyncBarrier.stopRequestGeneration
            || connectionEpochRef.current !== resyncBarrier.connectionEpoch;
          const shouldRetry = invalidated && operationInvalidated && !cancelled
            && sessionGenerationRef.current === listenerSessionGeneration
            && connectionStateRef.current === "open";
          if (barrierStillOwned) {
            resyncBarrierRef.current = null;
            if (!snapshotApplied && !resyncFailed && !invalidated) setResyncState("idle");
            else if (!resyncFailed && !shouldRetry && !cancelled && sessionGenerationRef.current === listenerSessionGeneration) setResyncState("synced");
          }
          if (sessionGenerationRef.current === listenerSessionGeneration) reconnectingRef.current = false;
          if (shouldRetry) {
            queueMicrotask(startResync);
          }
        }
      })();
    };
    startResyncRef.current = startResync;
    wasOpenRef.current = false;
    reconnectAttemptRef.current = 0;
    reconnectInFlightRef.current = false;
    reconnectRequestTokenRef.current += 1;
    clearReconnectTimer();
    const offState = gateway.onState((state) => {
      if (sessionGenerationRef.current !== listenerSessionGeneration) return;
      setConnectionState(state);
      connectionStateRef.current = state;
      if (state !== "open") {
        connectionEpochRef.current += 1;
        if (state === "connecting") setStatus("Reconnecting…");
        else if (state === "closed") {
          setStatus("Disconnected");
          scheduleReconnect();
        } else if (state === "error") {
          setStatus("Reconnecting…");
          scheduleReconnect();
        }
        return;
      }
      if (!wasOpenRef.current) { wasOpenRef.current = true; return; }
      clearReconnectTimer();
      reconnectAttemptRef.current = 0;
      reconnectInFlightRef.current = false;
      startResync();
    });
    const adoptUnboundTaggedTurn = (turnId: string | null, messageId: string | null): boolean => {
      if (!assistantIdRef.current || turnId === null || messageId !== null || activeTurnIdRef.current !== null || activeMessageIdRef.current !== null) return false;
      activeTurnIdRef.current = turnId;
      blockedTurnGenerationRef.current = null;
      unboundStartAcceptedRef.current = false;
      return true;
    };
    const handleStart = (event: GatewayEvent) => {
      const accepted = accept(event, true);

      if (!accepted) return;
      const turnId = eventTurnId(event);
      const messageId = eventMessageId(event);
      const currentAssistantId = assistantIdRef.current;
      const replacingMessageIdentity = Boolean(
        currentAssistantId
        && messageId
        && (
          (activeMessageIdRef.current !== null && activeMessageIdRef.current !== messageId)
          || (activeMessageIdRef.current === null && turnId !== null && activeTurnIdRef.current === turnId)
        ),
      );
      const replacingTurnIdentity = Boolean(
        currentAssistantId
        && turnId
        && activeTurnIdRef.current
        && activeTurnIdRef.current !== turnId,
      );
      const replacingExplicitStart = replacingMessageIdentity || replacingTurnIdentity;
      const preservePromptForSameTurnReplacement = replacingMessageIdentity
        && turnId === activeTurnIdRef.current;
      const sameTurn = Boolean(currentAssistantId && turnId && activeTurnIdRef.current === turnId && !replacingExplicitStart);
      const sameMessage = Boolean(currentAssistantId && messageId && activeMessageIdRef.current === messageId);
      const sameUnboundStream = Boolean(
        currentAssistantId
        && !turnId
        && activeTurnIdRef.current === null,
      );
      const sameUnboundIdentity = sameUnboundStream
        && !messageId
        && activeMessageIdRef.current === null;
      const taggedStartWithoutIdentity = Boolean(
        currentAssistantId
        && turnId
        && activeTurnIdRef.current === null
        && !messageId,
      );
      if ((sameTurn || sameMessage) && !replacingExplicitStart) return;
      if (sameUnboundIdentity || (currentAssistantId && !turnId && (activeTurnIdRef.current !== null || activeMessageIdRef.current !== null))) return;
      if (taggedStartWithoutIdentity) {
        adoptUnboundTaggedTurn(turnId, messageId);
        return;
      }
      if (!currentAssistantId) {
        setTools((items) => items.some((item) => item.state === "complete") ? [] : items);
      }
      if (replacingMessageIdentity && turnId !== null && activeTurnIdRef.current === turnId) {
        if (activeMessageIdRef.current) retiredMessageIdsRef.current.add(messageIdentityKey(sessionIdRef.current, activeMessageIdRef.current, durableSessionIdRef.current, sessionKeyRef.current));
        if (messageId) replacementTurnMessageIdsRef.current.set(turnId, messageId);
        replacementTurnProvenByDeltaRef.current.delete(turnId);
        retiredTurnIdsRef.current.add(turnId);
        const previousText = currentAssistantId ? liveAssistantText(currentAssistantId) : "";
        if (previousText) retiredTurnTextsRef.current.set(turnId, previousText);
      } else {
        retireActiveTurnIdentity();
      }
      if (!currentAssistantId && !turnId && !messageId && stopInFlightRef.current) {
        unboundStreamPendingRef.current = true;
      }
      if (!stopInFlightRef.current && retiredStopTargetRef.current !== null && !currentAssistantId && !turnId && !messageId) {
        const retiredTarget = retiredStopTargetRef.current;
        unboundStopFenceRef.current = stopTargetSessionMatches(retiredTarget, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
          && retiredTarget.sessionGeneration === sessionGenerationRef.current
          ? retiredTarget
          : null;
        retiredStopTargetRef.current = null;
      }
      const provisionalUnboundReplacement = !stopInFlightRef.current
        && !turnId
        && !messageId
        && !activePromptTextRef.current
        && unboundStopFenceRef.current !== null;
      const previousAssistantId = assistantIdRef.current;
      if (previousAssistantId && (activeTurnIdRef.current !== turnId || replacingExplicitStart)) {
        const previousText = liveAssistantText(previousAssistantId);
        if (previousText) commitLiveAssistantMessage(previousAssistantId, previousText);
        else dispatchTimeline({ type: "reset" });
        if (!preservePromptForSameTurnReplacement) activePromptTextRef.current = null;
      }
      if (replacingExplicitStart) {
        setTools([]);
        setApprovalState(null);
        setClarifyState(null);
      }
      if (!preservePromptForSameTurnReplacement && activePromptTextRef.current === null) turnGenerationRef.current += 1;
      if (!turnId && !messageId) {
        if (provisionalUnboundReplacement) {
          postRetirementUnboundStartPendingRef.current = true;
          unboundStartAcceptedRef.current = false;
          blockedTurnGenerationRef.current = turnGenerationRef.current;
        } else {
          unboundStartAcceptedRef.current = true;
        }
      }
      if ((turnId !== null || messageId !== null) && activePromptTextRef.current === null) {
        blockedTurnGenerationRef.current = turnGenerationRef.current;
      }
      const id = `assistant-${++messageSequenceRef.current}`;
      assistantIdRef.current = id;
      activeTurnIdRef.current = turnId;
      activeMessageIdRef.current = messageId;
      unboundStreamEstablishedRef.current = false;
      dispatchTimeline({ type: "append", event: toTimelineEvent(event, sessionKeyRef.current), entryId: id });
      setStreaming(true);
      setTurnStartedAt((started) => started ?? Date.now());
      setStatus("Thinking…");
    };
    const offStart = gateway.on("message.start", handleStart);
    const handleDelta = (event: GatewayEvent) => {
      const accepted = accept(event, true);

      if (!accepted) return;
      const text = eventText(event);
      const pendingUnboundReplacementDelta = isPendingUnboundReplacementDelta(event);
      if (!text && !pendingUnboundReplacementDelta) return;
      const turnId = eventTurnId(event);
      const messageId = eventMessageId(event);
      const wasUnboundTaggedContinuation = unboundStreamEstablishedRef.current
        && activeMessageIdRef.current === null
        && turnId !== null
        && activeTurnIdRef.current === turnId
        && messageId === null;
      const adoptedUnboundTaggedTurn = adoptUnboundTaggedTurn(turnId, messageId);
      const replacingExplicitStream = Boolean(
        assistantIdRef.current
        && (
          (
            (messageId !== null
              && activeMessageIdRef.current !== null
              && activeMessageIdRef.current !== messageId)
            || (!adoptedUnboundTaggedTurn
              && !wasUnboundTaggedContinuation
              && messageId !== null
              && activeMessageIdRef.current === null
              && turnId !== null
              && activeTurnIdRef.current === turnId
              && liveAssistantText(assistantIdRef.current).length > 0)
          )
          || (turnId && activeTurnIdRef.current && activeTurnIdRef.current !== turnId)
          || (activeTurnIdRef.current === null
            && activeMessageIdRef.current === null
            && (turnId !== null || messageId !== null))
        ),
      );
      if (replacingExplicitStream) {
        const replacementId = replaceActiveAssistantForExplicitIdentity(turnId, messageId);
        dispatchTimeline({ type: "append", event: toTimelineEvent(event, sessionKeyRef.current), entryId: replacementId });
        setStreaming(true);
        setTurnStartedAt((started) => started ?? Date.now());
        setStatus("Thinking…");
        return;
      }
      if (isPendingUnboundReplacementDelta(event)
        && assistantIdRef.current !== null
        && (activeTurnIdRef.current !== null || activeMessageIdRef.current !== null)) {
        const replacementPromptText = activePromptTextRef.current;
        const replacementId = replaceActiveAssistantForExplicitIdentity(null, null);
        if (replacementPromptText !== null) activePromptTextRef.current = replacementPromptText;
        const stopFence = retiredStopTargetRef.current ?? stopTargetRef.current;
        unboundStopFenceRef.current = stopFence && stopTargetSessionMatches(stopFence, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
          ? stopFence
          : null;
        stopTargetRef.current = null;
        retiredStopTargetRef.current = null;
        unboundStreamPendingRef.current = false;
        unboundStreamEstablishedRef.current = true;
        unboundStartAcceptedRef.current = true;
        blockedTurnGenerationRef.current = null;
        dispatchTimeline({ type: "append", event: toTimelineEvent(event, sessionKeyRef.current), entryId: replacementId });
        setStreaming(true);
        setTurnStartedAt((started) => started ?? Date.now());
        setStatus("Thinking…");
        return;
      }
      if (!turnId && !messageId) {
        if (unboundStreamPendingRef.current) {
          const stopFence = retiredStopTargetRef.current ?? stopTargetRef.current;
          unboundStopFenceRef.current = stopFence && stopTargetSessionMatches(stopFence, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
          ? stopFence
          : null;
          stopTargetRef.current = null;
          retiredStopTargetRef.current = null;
          unboundStreamPendingRef.current = false;
          unboundStartAcceptedRef.current = false;
        }
        if (activePromptTextRef.current !== null && (stopInFlightRef.current || stopTargetRef.current !== null || retiredStopTargetRef.current !== null)) {
          const stopFence = retiredStopTargetRef.current ?? stopTargetRef.current;
          unboundStopFenceRef.current = stopFence && stopTargetSessionMatches(stopFence, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
          ? stopFence
          : null;
          stopTargetRef.current = null;
          retiredStopTargetRef.current = null;
        }
        if (postRetirementUnboundStartPendingRef.current && text.length > 0) {
          postRetirementUnboundStartPendingRef.current = false;
          unboundStreamEstablishedRef.current = true;
          unboundStartAcceptedRef.current = true;
        }
        unboundStreamEstablishedRef.current = true;
        unboundStartAcceptedRef.current = true;
        blockedTurnGenerationRef.current = null;
      }
      if (turnId && !activeTurnIdRef.current) {
        activeTurnIdRef.current = turnId;
        blockedTurnGenerationRef.current = null;
      }
      if (messageId && !activeMessageIdRef.current) activeMessageIdRef.current = messageId;
      if (!assistantIdRef.current && activePromptTextRef.current === null) turnGenerationRef.current += 1;
      if (!assistantIdRef.current && activePromptTextRef.current !== null) blockedTurnGenerationRef.current = null;
      const id = assistantIdRef.current ?? `assistant-${++messageSequenceRef.current}`;
      if (!assistantIdRef.current) {
        assistantIdRef.current = id;
        dispatchTimeline({
          type: "append",
          event: {
            type: "message.start",
            session_id: event.session_id,
            payload: {
              ...(turnId ? { turn_id: turnId } : {}),
              ...(messageId ? { message_id: messageId } : {}),
            },
          },
          entryId: id,
        });
      }
      dispatchTimeline({ type: "update", event: toTimelineEvent(event, sessionKeyRef.current), entryId: id });
    };
    const offDelta = gateway.on("message.delta", handleDelta);
    const handleThinking = (event: GatewayEvent) => { const p = accept(event, true); if (p && eventText(event)) setStatus(`Thinking: ${eventText(event)}`); };
    const offThinking = gateway.on("thinking.delta", handleThinking);
    const handleReasoning = (event: GatewayEvent) => { const p = accept(event, true); if (p && eventText(event)) setStatus(`Reasoning: ${eventText(event)}`); };
    const offReasoning = gateway.on("reasoning.delta", handleReasoning);
    const handleInterim = (event: GatewayEvent) => { const p = accept(event, true); const text = eventText(event); if (!p || !text) return; setTranscript((messages) => [...messages, { id: `interim-${++messageSequenceRef.current}`, role: "assistant", text, interim: true }]); };
    const offInterim = gateway.on("message.interim", handleInterim);
    const handleToolGenerating = (event: GatewayEvent) => { const p = accept(event, true); if (p) setStatus(`Preparing tool: ${String(p.name ?? "tool")}`); };
    const offToolGenerating = gateway.on("tool.generating", handleToolGenerating);
    const replaceActiveAssistantForExplicitIdentity = (turnId: string | null, messageId: string | null): string => {
      const previousAssistantId = assistantIdRef.current;
      const previousTurnId = activeTurnIdRef.current;
      const previousMessageId = activeMessageIdRef.current;
      const previousPromptText = activePromptTextRef.current;
      const sameLogicalTurn = previousTurnId === turnId;
      if (previousAssistantId) {
        if (turnId !== null && previousTurnId === turnId) {
          if (previousMessageId) retiredMessageIdsRef.current.add(messageIdentityKey(sessionIdRef.current, previousMessageId, durableSessionIdRef.current, sessionKeyRef.current));
          if (messageId) replacementTurnMessageIdsRef.current.set(turnId, messageId);
          replacementTurnProvenByDeltaRef.current.delete(turnId);
          retiredTurnIdsRef.current.add(turnId);
          const previousText = liveAssistantText(previousAssistantId);
          if (previousText) retiredTurnTextsRef.current.set(turnId, previousText);
        } else {
          retireActiveTurnIdentity();
        }
        const previousText = liveAssistantText(previousAssistantId);
        if (previousText) commitLiveAssistantMessage(previousAssistantId, previousText, previousPromptText);
        else dispatchTimeline({ type: "reset" });
      }
      setTools([]);
      setApprovalState(null);
      setClarifyState(null);
      if (sameLogicalTurn) activePromptTextRef.current = previousPromptText;
      else activePromptTextRef.current = null;
      unboundStreamEstablishedRef.current = false;
      unboundStreamPendingRef.current = false;
      unboundStartAcceptedRef.current = false;
      if (!sameLogicalTurn) turnGenerationRef.current += 1;
      const replacementId = `assistant-${++messageSequenceRef.current}`;
      assistantIdRef.current = replacementId;
      activeTurnIdRef.current = turnId;
      activeMessageIdRef.current = messageId;
      blockedTurnGenerationRef.current = null;
      return replacementId;
    };
    const handleComplete = (event: GatewayEvent) => {
      const acceptedComplete = accept(event, true);
      if (!acceptedComplete) return;
      let id = assistantIdRef.current;
      const eventTurnIdValue = eventTurnId(event);
      const messageId = eventMessageId(event);
      const completionPayload = (event.payload ?? {}) as TextPayload;
      const completionHasError = completionPayload.status === "error"
        || typeof completionPayload.error === "string"
        || completionPayload.kind === "error";
      const completionErrorText = typeof completionPayload.error === "string" && completionPayload.error
        ? completionPayload.error
        : eventText(event) || "Gateway error";
      const durableCompletionErrorTarget = Boolean(messageId && transcriptStateRef.current.some((entry) => entry.role === "assistant" && (entry.id === messageId || entry.messageId === messageId)));
      if (messageId && durableCompletionErrorTarget && completionHasError && activeMessageIdRef.current !== messageId) {
        const completionId = transcriptIdForMessageId(messageId);
        setTranscript((current) => mergeCompletedAssistantMessage(current, completionId, "", null, completionErrorText) as TranscriptMessage[]);
        retiredMessageIdsRef.current.add(messageIdentityKey(event.session_id, messageId, durableSessionIdRef.current, sessionKeyRef.current));
        return;
      }
      adoptUnboundTaggedTurn(eventTurnIdValue, messageId);
      const turnId = eventTurnIdValue ?? activeTurnIdRef.current;
      const replacingExplicitCompletion = Boolean(
        id
        && (
          (messageId && activeMessageIdRef.current && activeMessageIdRef.current !== messageId)
          || (eventTurnIdValue && activeTurnIdRef.current && activeTurnIdRef.current !== eventTurnIdValue)
          || (activeTurnIdRef.current === null
            && activeMessageIdRef.current === null
            && (eventTurnIdValue !== null || messageId !== null))
        ),
      );
      if (replacingExplicitCompletion) {
        id = replaceActiveAssistantForExplicitIdentity(eventTurnIdValue, messageId);
      }
      const stopTarget = stopTargetRef.current;
      const stopTerminal = stopInFlightRef.current
        && stopTarget !== null
        && stopTargetSessionMatches(stopTarget, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
        && stopTarget.sessionGeneration === sessionGenerationRef.current
        && stopTarget.requestGeneration === stopRequestGenerationRef.current
        && stopTarget.turnGeneration === turnGenerationRef.current
        && (!stopTarget.turnId || stopTarget.turnId === turnId)
        && (!stopTarget.messageId || stopTarget.messageId === messageId)
        && (!stopTarget.assistantId || stopTarget.assistantId === id);
      if (stopTerminal) retiredStopTargetRef.current = stopTarget;
      const liveText = id ? liveAssistantText(id) : "";
      const finalText = chooseCompletedText(liveText, eventText(event));
      const payload = (event.payload ?? {}) as TextPayload;
      const terminalError = payload.status === "error" || typeof payload.error === "string" || payload.kind === "error";
      const terminalErrorText = typeof payload.error === "string" && payload.error
        ? payload.error
        : eventText(event) || "Gateway error";
      const promptText = activePromptTextRef.current;
      if (retiredStopTargetRef.current && (turnId || messageId) && !stopTerminal) retiredStopTargetRef.current = null;
      const allowNextUnboundStart = !terminalError && !stopTerminal;
      const durableCompletionTarget = Boolean(messageId && transcriptStateRef.current.some((message) => message.role === "assistant" && (message.id === messageId || message.messageId === messageId)));
      if (id) clearLocalTurnState(finalText, terminalError ? terminalErrorText : undefined, allowNextUnboundStart);
      else clearLocalTurnState(undefined, undefined, allowNextUnboundStart);
      if (stopTerminal && stopTarget) retiredStopTargetRef.current = stopTarget;
      if (messageId) retiredMessageIdsRef.current.add(messageIdentityKey(event.session_id, messageId, durableSessionIdRef.current, sessionKeyRef.current));
      if (!id && (durableCompletionTarget || finalText || terminalError)) {
        const completionId = messageId ? transcriptIdForMessageId(messageId) : `assistant-${++messageSequenceRef.current}`;
        setTranscript((current) => mergeCompletedAssistantMessage(
          current,
          completionId,
          finalText,
          promptText,
          terminalError ? terminalErrorText : undefined,
        ) as TranscriptMessage[]);
      }
      if (turnId) ignoredTurnIdsRef.current.add(turnId);
      if (terminalError) {
        setError(terminalErrorText);
        setErrorAction(promptText ? "resend" : null);
        if (promptText) setFailedPrompt({ id: `retry-${turnId ?? messageId ?? messageSequenceRef.current}`, text: promptText, mode: "retry" });
        setStatus("Error");
      } else {
        setError(null);
        setErrorAction(null);
        setFailedPrompt(null);
        setStatus("Ready");
      }
    };
    const offComplete = gateway.on("message.complete", handleComplete);
    const handleToolStart = (event: GatewayEvent) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? `${p.name ?? "tool"}-${Date.now()}`); const startedAt = Date.now(); setStatus(`Running tool: ${String(p.name ?? "tool")}`); setTools((items) => items.some((item) => item.id === id) ? items : [...items, { id, name: String(p.name ?? "tool"), state: "running", context: typeof p.context === "string" ? p.context : undefined, args: p.args, startedAt }]); };
    const offToolStart = gateway.on("tool.start", handleToolStart);
    const handleToolProgress = (event: GatewayEvent) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? ""); if (!id) return; const progress = typeof p.progress === "string" ? p.progress : typeof p.text === "string" ? p.text : ""; if (progress) setStatus(`Working: ${progress}`); setTools((items) => items.map((item) => item.id === id ? { ...item, progress: progress || item.progress, elapsedMs: eventElapsedMs(p, item.startedAt) } : item)); };
    const offToolProgress = gateway.on("tool.progress", handleToolProgress);
    const handleToolComplete = (event: GatewayEvent) => { const p = accept(event, true); if (!p) return; const id = String(p.tool_id ?? `${p.name ?? "tool"}-${Date.now()}`); setTools((items) => { const existing = items.find((item) => item.id === id); const elapsed = eventElapsedMs(p, existing?.startedAt); return existing ? items.map((item) => item.id === id ? { ...item, state: "complete", args: p.args ?? item.args, result: p.result, summary: typeof p.summary === "string" ? p.summary : item.summary, elapsedMs: elapsed ?? item.elapsedMs } : item) : [...items, { id, name: String(p.name ?? "tool"), state: "complete", args: p.args, result: p.result, summary: typeof p.summary === "string" ? p.summary : undefined, elapsedMs: elapsed }]; }); };
    const offToolComplete = gateway.on("tool.complete", handleToolComplete);
    const handleApproval = (event: GatewayEvent) => { const p = accept(event, true); if (!p) return; const request = normalizeApprovalRequestPayload(p as Record<string, unknown>); if (request) setApprovalState(request); };
    const offApproval = gateway.on("approval.request", handleApproval);
    const handleClarify = (event: GatewayEvent) => { const p = accept(event, true); if (!p || typeof p.request_id !== "string" || !p.request_id.trim()) return; setClarifyState({ request_id: p.request_id, question: typeof p.question === "string" ? p.question : undefined, choices: Array.isArray(p.choices) ? p.choices.filter((x): x is string => typeof x === "string") : null, multi_select: p.multi_select === true, questions: Array.isArray(p.questions) ? p.questions as ClarificationRequest["questions"] : undefined }); };
    const offClarify = gateway.on("clarify.request", handleClarify);
    const handleError = (event: GatewayEvent) => {
      if (!accept(event, true)) return;
      const payload = (event.payload ?? {}) as TextPayload;
      const turnId = eventTurnId(event);
      const messageId = eventMessageId(event);
      const turnError = Boolean(turnId || messageId || payload.kind === "turn" || payload.turn_scoped === true || payload.error_surface === "turn");
      const message = typeof payload.error === "string" && payload.error
        ? payload.error
        : eventText(event) || "Gateway error";
      if (!turnError) {
        setError(message);
        setErrorAction(null);
        setStatus("Error");
        return;
      }
      const durableMessageTarget = Boolean(messageId && transcriptStateRef.current.some((entry) => entry.role === "assistant" && (entry.id === messageId || entry.messageId === messageId)));
      if (messageId && durableMessageTarget && activeMessageIdRef.current !== messageId) {
        const completionId = transcriptIdForMessageId(messageId);
        setTranscript((current) => mergeCompletedAssistantMessage(current, completionId, "", null, message) as TranscriptMessage[]);
        retiredMessageIdsRef.current.add(messageIdentityKey(event.session_id, messageId, durableSessionIdRef.current, sessionKeyRef.current));
        return;
      }
      let id = assistantIdRef.current;
      adoptUnboundTaggedTurn(turnId, messageId);
      const replacingExplicitError = Boolean(
        id
        && (
          (messageId && activeMessageIdRef.current && activeMessageIdRef.current !== messageId)
          || (turnId && activeTurnIdRef.current && activeTurnIdRef.current !== turnId)
          || (activeTurnIdRef.current === null
            && activeMessageIdRef.current === null
            && (turnId !== null || messageId !== null))
        ),
      );
      if (replacingExplicitError) {
        id = replaceActiveAssistantForExplicitIdentity(turnId, messageId);
      }
      const activeTurnId = turnId ?? activeTurnIdRef.current;
      const stopTarget = stopTargetRef.current;
      const stopTerminal = stopInFlightRef.current
        && stopTarget !== null
        && stopTargetSessionMatches(stopTarget, sessionIdRef.current, durableSessionIdRef.current, sessionKeyRef.current)
        && stopTarget.sessionGeneration === sessionGenerationRef.current
        && stopTarget.requestGeneration === stopRequestGenerationRef.current
        && stopTarget.turnGeneration === turnGenerationRef.current
        && (!stopTarget.turnId || stopTarget.turnId === activeTurnId)
        && (!stopTarget.messageId || stopTarget.messageId === messageId)
        && (!stopTarget.assistantId || stopTarget.assistantId === id);
      if (stopTerminal) retiredStopTargetRef.current = stopTarget;
      const promptText = activePromptTextRef.current;
      if (turnError && retiredStopTargetRef.current && (activeTurnId || messageId) && !stopTerminal) retiredStopTargetRef.current = null;
      clearLocalTurnState(id ? liveAssistantText(id) : undefined, message);
      if (stopTerminal && stopTarget) retiredStopTargetRef.current = stopTarget;
      if (messageId) retiredMessageIdsRef.current.add(messageIdentityKey(event.session_id, messageId, durableSessionIdRef.current, sessionKeyRef.current));
      if (!id) {
        const completionId = messageId ? transcriptIdForMessageId(messageId) : `assistant-${++messageSequenceRef.current}`;
        setTranscript((current) => mergeCompletedAssistantMessage(
          current,
          completionId,
          eventText(event),
          promptText,
          message,
        ) as TranscriptMessage[]);
      }
      if (activeTurnId) ignoredTurnIdsRef.current.add(activeTurnId);
      setError(message);
      setErrorAction(promptText ? "resend" : null);
      if (promptText) setFailedPrompt({ id: `retry-${activeTurnId ?? messageId ?? messageSequenceRef.current}`, text: promptText, mode: "retry" });
      setStatus("Error");
    };
    const offError = gateway.on("error", handleError);
    const handleStatus = (event: GatewayEvent) => { if (accept(event, true)) setStatus(eventText(event) || "Working…"); };
    const offStatus = gateway.on("status.update", handleStatus);
    const handleInfo = (event: GatewayEvent) => {
      const p = accept(event, true);
      if (!p) return;
      const resyncBarrier = resyncBarrierRef.current;
      if (resyncBarrier && (
        sessionGenerationRef.current !== resyncBarrier.sessionGeneration
        || operationTokenRef.current !== resyncBarrier.operationToken
        || stopRequestGenerationRef.current !== resyncBarrier.stopRequestGeneration
        || turnGenerationRef.current !== resyncBarrier.turnGeneration
        || connectionEpochRef.current !== resyncBarrier.connectionEpoch
      )) return;
      const hasRunning = typeof p.running === "boolean";
      const running = p.running === true;
      const infoTurnId = eventTurnId(event);
      const infoMessageId = eventMessageId(event);
      const hasActiveTurn = Boolean(assistantIdRef.current || activeTurnIdRef.current || activePromptTextRef.current);
      const identityMatchesActiveTurn = Boolean(
        (infoTurnId !== null || infoMessageId !== null)
        && (infoTurnId === null || activeTurnIdRef.current === infoTurnId)
        && (infoMessageId === null || activeMessageIdRef.current === infoMessageId),
      );
      if (hasRunning && p.running === false && hasActiveTurn
        && !identityMatchesActiveTurn) return;
      if (!hasRunning && (p.status === "idle" || p.status === "ready") && hasActiveTurn) {
        if (!identityMatchesActiveTurn) return;
        clearLocalTurnState();
        setError(null);
        setErrorAction(null);
        setStatus("Ready");
        return;
      }
      if (hasRunning) {
        setStreaming(running);
        setTurnStartedAt(typeof p.turn_started_at === "number" ? p.turn_started_at * 1000 : (running ? (started) => started ?? Date.now() : null));
      }
      if (typeof p.status === "string" && p.status) {
        setStatus(p.status);
        if (p.status === "error") {
          const statusError = typeof p.error === "string" && p.error ? p.error : "Session reported an error";
          setError(statusError);
          setErrorAction(null);
        }
      }
      else if (hasRunning && running) setStatus("Working…");
      else if (hasRunning) setStatus("Ready");
      if (hasRunning && p.running === false) {
        const oldAssistantId = assistantIdRef.current;
        const activePromptText = activePromptTextRef.current;
        retireActiveTurnIdentity();
        if (oldAssistantId) {
          const oldText = liveAssistantText(oldAssistantId);
          if (oldText) setTranscript((current) => mergeCompletedAssistantMessage(current, oldAssistantId, oldText, activePromptText) as TranscriptMessage[]);
        }
        assistantIdRef.current = null;
        activeTurnIdRef.current = null;
        activeMessageIdRef.current = null;
        activePromptTextRef.current = null;
        postRetirementUnboundStartPendingRef.current = false;
        allowUnboundStartAfterRetirementRef.current = false;
        turnGenerationRef.current += 1;
        blockedTurnGenerationRef.current = turnGenerationRef.current;
        dispatchTimeline({ type: "reset" });
        setTools((items) => items.map((item) => item.state === "running"
          ? { ...item, state: "complete", summary: item.summary ?? "Session is idle" }
          : item));
        setApprovalState(null);
        setClarifyState(null);
      }
    };
    const offInfo = gateway.on("session.info", handleInfo);
    resyncEventHandlersRef.current = new Map([
      ["message.start", handleStart],
      ["message.delta", handleDelta],
      ["thinking.delta", handleThinking],
      ["reasoning.delta", handleReasoning],
      ["message.interim", handleInterim],
      ["tool.generating", handleToolGenerating],
      ["message.complete", handleComplete],
      ["tool.start", handleToolStart],
      ["tool.progress", handleToolProgress],
      ["tool.complete", handleToolComplete],
      ["approval.request", handleApproval],
      ["clarify.request", handleClarify],
      ["error", handleError],
      ["status.update", handleStatus],
      ["session.info", handleInfo],
    ]);
    replayResyncEventsRef.current = () => {
      const buffered = resyncBufferedEventsRef.current.splice(0);

      if (buffered.length === 0) return;
      replayingResyncEventsRef.current = true;
      try {
        for (const { event, handler } of buffered) {

          handler(event);
        }
      } finally {
        replayingResyncEventsRef.current = false;
      }
    };
    queueMicrotask(() => {
      if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration) return;
      sessionIdRef.current = null;
      durableSessionIdRef.current = resumeParam;
      sessionKeyRef.current = null;
      validatedDurableAliasesRef.current.clear();
      if (resumeParam) validatedDurableAliasesRef.current.add(resumeParam);
      assistantIdRef.current = null;
      activeMessageIdRef.current = null;
      unboundStreamEstablishedRef.current = false;
      unboundStreamPendingRef.current = false;
      unboundStartAcceptedRef.current = false;
      postRetirementUnboundStartPendingRef.current = false;
      allowUnboundStartAfterRetirementRef.current = true;
      setSessionId(null);
      setDurableSessionId(resumeParam);
      setTranscript([]);
      activePromptTextRef.current = null;
      setTranscriptQuery("");
      setEditTarget(null);
      setEditSubmitting(false);
      dispatchTimeline({ type: "reset" });
      virtualRowHeightsRef.current.clear();
      setVirtualMeasureRevision((revision) => revision + 1);
      setVirtualViewport((current) => ({ ...current, scrollTop: 0 }));
      setTools([]);
      setApprovalState(null);
      setClarifyState(null);

      setTurnStartedAt(null);
      setStreaming(false);
      followTranscriptRef.current = true;
      setShowScrollToBottom(false);
      clearAttachments();
      seenSeqRef.current.clear();
      seenEventIdsRef.current.clear();
      messageSequenceRef.current = 0;
      inflightSnapshotSequenceRef.current = 0;
      inflightFallbackIdentityMapRef.current.clear();
      activeInflightFallbackKeyRef.current = null;
      turnGenerationRef.current += 1;
      stopRequestGenerationRef.current += 1;
      stopTargetRef.current = null;
      retiredStopTargetRef.current = null;
      unboundStopFenceRef.current = null;
      activeTurnIdRef.current = null;
      ignoredTurnIdsRef.current.clear();
      retiredTurnIdsRef.current.clear();
      retiredTurnTextsRef.current.clear();
      replacementTurnMessageIdsRef.current.clear();
      replacementTurnProvenByDeltaRef.current.clear();
      retiredMessageIdsRef.current.clear();
      blockedTurnGenerationRef.current = null;
      setStatus(null);
      setError(null);
      setErrorAction(null);
      setResyncState("idle");
      setFailedPrompt(null);
      setQueuedPrompts([]);
      queueDrainInFlightRef.current = false;
      setSubmitting(false);
      setStopping(false);
      submitInFlightRef.current = false;
      submitOwnerTokenRef.current = null;
      submitOwnerKindRef.current = null;
      stopInFlightRef.current = false;
    });
    void gateway.connect()
      .then(async () => {
        if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration) return;
        initialAttachPendingRef.current = true;
        initialAttachBufferedEventsRef.current = [];
        let response: ResumeResponse;
        if (resumeParam) {
          try {
            response = await gateway.request<ResumeResponse>("session.activate", reconnectActivateParams(resumeParam, profile));
          } catch {
            if (cancelled || sessionGenerationRef.current !== listenerSessionGeneration) return;
            response = await gateway.request<ResumeResponse>("session.resume", { session_id: resumeParam, continue_on_disconnect: true, ...(profile ? { profile } : {}) });
          }
        } else {
          response = await gateway.request<ResumeResponse>("session.create", {
            ...nativeChatSessionCreateParams(profile, routingSelectionRef.current),
          });
        }
        if (!cancelled && sessionGenerationRef.current === listenerSessionGeneration) {
          const runtimeId = response.session_id;
          if (!runtimeId) throw new Error("Gateway returned no session id");
          const returnedSessionKey = response.session_key ?? response.info?.session_key;
          const returnedStoredSessionId = response.stored_session_id ?? response.info?.stored_session_id;
          const returnedDurableIds = [returnedSessionKey, returnedStoredSessionId].filter((value): value is string => typeof value === "string" && value.length > 0);
          const identityValidation = validateDurableIdentityResponse(
            returnedSessionKey,
            [returnedStoredSessionId],
            resumeParam ? [resumeParam] : [],
          );
          if (!identityValidation.accepted) {
            throw new Error(returnedDurableIds.length === 0
              ? "Gateway returned no durable session identity"
              : "Gateway returned a different durable session");
          }
          for (const alias of identityValidation.validatedIds) validatedDurableAliasesRef.current.add(alias);
          const storedId = identityValidation.canonicalId;
          durableSessionIdRef.current = storedId;
          sessionIdRef.current = runtimeId;
          setSessionId(runtimeId);
          setDurableSessionId(storedId);
          const initialAttachEvents = initialAttachBufferedEventsRef.current
            .filter(({ event }) => event.session_id === runtimeId);
          initialAttachBufferedEventsRef.current = [];
          initialAttachPendingRef.current = false;
          if (!applySessionSnapshot(response)) throw new Error("Gateway returned a conflicting durable session");
          if (initialAttachEvents.length) {
            replayingResyncEventsRef.current = true;
            try {
              for (const { event, handler } of initialAttachEvents) handler(event);
            } finally {
              replayingResyncEventsRef.current = false;
            }
          }
          void gateway.request<ModelOptionsCatalog>("model.options", {
            include_unconfigured: true,
            ...(profile ? { profile } : {}),
            ...(runtimeId ? { session_id: runtimeId } : {}),
          })
            .then((catalog) => { if (!cancelled && sessionGenerationRef.current === listenerSessionGeneration) setModelCatalog(catalog); })
            .catch(() => { /* catalog is best-effort; Adaptive remains available */ });
        }
      })
      .catch((reason: unknown) => {
        initialAttachPendingRef.current = false;
        initialAttachBufferedEventsRef.current = [];
        if (!cancelled && sessionGenerationRef.current === listenerSessionGeneration) setError(reason instanceof Error ? reason.message : String(reason));
      });

    return () => {
      queueDrainInFlightRef.current = false;
      sessionGenerationRef.current += 1;
      operationTokenRef.current += 1;
      cancelled = true;
      reconnectingRef.current = false;
      resyncBarrierRef.current = null;
      resyncBufferedEventsRef.current = [];
      resyncEventHandlersRef.current.clear();
      replayResyncEventsRef.current = null;
      replayingResyncEventsRef.current = false;
      resyncBufferOverflowRef.current = false;
      initialAttachPendingRef.current = false;
      initialAttachBufferedEventsRef.current = [];
      reconnectRequestTokenRef.current += 1;
      clearReconnectTimer();
      reconnectInFlightRef.current = false;
      offState();
      offStart();
      offDelta();
      offThinking();
      offReasoning();
      offInterim();
      offToolGenerating();
      offComplete();
      offToolStart();
      offToolProgress();
      offToolComplete();
      offApproval();
      offClarify();
      offError();
      offStatus();
      offInfo();
      clearAttachments();
      startResyncRef.current = null;
      resyncAfterStopRef.current = false;
      gateway.close();
    };
  }, [clearAttachments, clearLocalTurnState, clearReconnectTimer, commitLiveAssistantMessage, dispatchTimeline, freshGeneration, gateway, liveAssistantText, profile, resumeParam, retireActiveTurnIdentity, retireInflightFallbackIdentity, routeModel, routeProvider, routeReasoning, setApprovalState, setClarifyState]);

  const changeRouting = useCallback((nextModel: string, nextProvider: string, nextReasoning: NativeReasoningLevel) => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      if (nextModel && nextProvider) {
        next.set("model", nextModel);
        next.set("provider", nextProvider);
      } else {
        next.delete("model");
        next.delete("provider");
      }
      if (nextReasoning === "auto") next.delete("reasoning");
      else next.set("reasoning", nextReasoning);
      next.delete("resume");
      return next;
    }, { replace: false });
  }, [setSearchParams]);

  const startNewChat = useCallback(() => {
    sessionGenerationRef.current += 1;
    operationTokenRef.current += 1;
    stopRequestGenerationRef.current += 1;
    stopTargetRef.current = null;
    retiredStopTargetRef.current = null;
    replacementTurnMessageIdsRef.current.clear();
    replacementTurnProvenByDeltaRef.current.clear();
    stopInFlightRef.current = false;
    submitInFlightRef.current = false;
    submitOwnerTokenRef.current = null;
    submitOwnerKindRef.current = null;
    queueDrainInFlightRef.current = false;
    activePromptTextRef.current = null;
    postRetirementUnboundStartPendingRef.current = false;
    allowUnboundStartAfterRetirementRef.current = true;
    resyncBarrierRef.current = null;
    setStopping(false);
    setSubmitting(false);
    setEditSubmitting(false);
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      next.delete("resume");
      return next;
    }, { replace: false });
    setFreshGeneration((generation) => generation + 1);
  }, [setSearchParams]);

  const branchSession = useCallback(async () => {
    const requestSessionId = sessionIdRef.current;
    if (!sessionId || !requestSessionId || connectionState !== "open") return;
    const turnActive = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null || approval !== null || clarify !== null;
    if (turnActive) {
      setError("Wait for the active turn to finish before branching");
      setErrorAction(null);
      return;
    }
    setError(null);
    setErrorAction(null);
    setStatus("Branching…");
    const requestSessionGeneration = sessionGenerationRef.current;
    const requestStopGeneration = stopRequestGenerationRef.current;
    const requestToken = ++operationTokenRef.current;
    const operationIsCurrent = () => sessionGenerationRef.current === requestSessionGeneration
      && sessionIdRef.current === requestSessionId
      && stopRequestGenerationRef.current === requestStopGeneration
      && operationTokenRef.current === requestToken;
    try {
      const response = await gateway.request<BranchResponse>("session.branch", { session_id: requestSessionId });
      if (!operationIsCurrent()) return;
      const target = response.stored_session_id ?? response.session_id;
      if (!target) throw new Error("Gateway returned no branch session id");
      setSearchParams((previous) => {
        const next = new URLSearchParams(previous);
        next.set("resume", target);
        return next;
      }, { replace: false });
    } catch (reason: unknown) {
      if (!operationIsCurrent()) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction("reconnect");
      setStatus("Branch failed");
    }
  }, [approval, clarify, connectionState, gateway, sessionId, setSearchParams, streaming, tools, turnStartedAt]);

  const submit = useCallback(async (event?: FormEvent, pendingPrompt?: PendingPrompt) => {
    event?.preventDefault();
    if (submitInFlightRef.current) return;
    const text = pendingPrompt?.text ?? draft.trim();
    const hasAttachment = attachmentsRef.current.some((item) => item.state === "attached");
    if ((!text && !hasAttachment) || !sessionId || connectionState !== "open") return;
    if (pendingPrompt?.mode === "retry" && failedPrompt?.id !== pendingPrompt.id) return;
    if (attachmentsRef.current.some((item) => item.state === "uploading" || item.state === "pending")) { setError("Please wait for attachments to finish uploading"); return; }
    if (attachmentsRef.current.some((item) => item.state === "error")) { setError("Retry or remove failed attachments before sending"); return; }
    const promptText = pendingPrompt ? pendingPrompt.text : attachmentPromptText(text, attachmentsRef.current);
    const messageId = pendingPrompt?.id ?? `user-${Date.now()}`;
    const turnActive = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null || approval !== null || clarify !== null;
    const stopBarrierActive = stopping || stopInFlightRef.current;

    if (pendingPrompt && stopBarrierActive) {
      setQueuedPrompts((current) => current.some((item) => item.id === pendingPrompt.id)
        ? current
        : [pendingPrompt, ...current]);
      return;
    }
    if (pendingPrompt && turnActive) {
      setQueuedPrompts((current) => current.some((item) => item.id === pendingPrompt.id)
        ? current
        : [pendingPrompt, ...current]);
      setStatus("Queued");
      return;
    }
    if (!pendingPrompt && (turnActive || stopBarrierActive)) {
      setQueuedPrompts((current) => [...current, { id: `queued-${Date.now()}-${current.length}`, text: promptText, mode: "queued" }]);
      setDraft("");
      clearAttachments();
      setStatus("Queued");
      return;
    }

    const requestSessionId = sessionIdRef.current;
    if (!requestSessionId) return;
    const requestSessionGeneration = sessionGenerationRef.current;
    const requestStopGeneration = stopRequestGenerationRef.current;
    const requestToken = ++operationTokenRef.current;
    const operationIsCurrent = () => (
      sessionGenerationRef.current === requestSessionGeneration
      && sessionIdRef.current === requestSessionId
      && stopRequestGenerationRef.current === requestStopGeneration
      && operationTokenRef.current === requestToken
    );
    submitInFlightRef.current = true;
    submitOwnerTokenRef.current = requestToken;
    submitOwnerKindRef.current = "submit";
    activePromptTextRef.current = promptText;
    turnGenerationRef.current += 1;
    blockedTurnGenerationRef.current = turnGenerationRef.current;
    setSubmitting(true);
    setError(null);
    setErrorAction(null);
    setStatus("Sending…");
    if (!pendingPrompt || pendingPrompt.mode === "queued") {
      setDraft("");
      setTranscript((messages) => [...messages, { id: messageId, role: "user", text: promptText }]);
    }
    try {
      await gateway.request("prompt.submit", { session_id: requestSessionId, text: promptText });
      if (!operationIsCurrent()) return;
      clearAttachments();
      setFailedPrompt(null);
      setStatus("Working…");
    } catch (reason: unknown) {
      if (!operationIsCurrent()) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction("resend");
      setFailedPrompt({ id: messageId, text: promptText, mode: "retry" });
      setStatus("Error");
    } finally {
      if (submitOwnerTokenRef.current === requestToken && submitOwnerKindRef.current === "submit") {
        submitInFlightRef.current = false;
        submitOwnerTokenRef.current = null;
        submitOwnerKindRef.current = null;
        setSubmitting(false);
      }
    }
  }, [approval, clarify, clearAttachments, connectionState, draft, failedPrompt?.id, gateway, sessionId, stopping, streaming, tools, turnStartedAt]);

  useEffect(() => {
    const turnActive = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null || approval !== null || clarify !== null;
    if (turnActive || submitting || stopping || stopInFlightRef.current || queueDrainInFlightRef.current || !sessionId || connectionState !== "open" || queuedPrompts.length === 0) return;
    const next = queuedPrompts[0];
    const queueDrainGeneration = sessionGenerationRef.current;
    queueDrainInFlightRef.current = true;
    setQueuedPrompts((current) => current[0]?.id === next.id ? current.slice(1) : current);
    void submit(undefined, next).finally(() => {
      if (shouldReleaseQueueDrain(queueDrainGeneration, sessionGenerationRef.current)) queueDrainInFlightRef.current = false;
    });
  }, [approval, clarify, connectionState, queuedPrompts, sessionId, stopping, streaming, submitting, submit, tools, turnStartedAt]);

  const runLastPromptAgain = useCallback(() => {
    const lastUser = [...transcript].reverse().find((message) => message.role === "user");
    if (!lastUser || !sessionId || connectionState !== "open") return;
    const pending: PendingPrompt = { id: `rerun-${Date.now()}`, text: lastUser.text, mode: "queued" };
    const turnActive = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null || approval !== null || clarify !== null;
    if (turnActive) {
      setQueuedPrompts((current) => [...current, pending]);
      setStatus("Queued");
      return;
    }
    void submit(undefined, pending);
  }, [approval, clarify, connectionState, sessionId, streaming, submit, tools, transcript, turnStartedAt]);

  const applyMessageAsPrompt = useCallback((message: string) => {
    setDraft(message);
    const textarea = textareaRef.current;
    if (textarea && !textarea.disabled) {
      textarea.focus();
      textarea.setSelectionRange(message.length, message.length);
    }
  }, []);

  const beginMessageEdit = useCallback((message: TranscriptMessage) => {
    if (message.role !== "user") return;
    if (parseDurableRowId(message.rowId) === undefined) {
      applyMessageAsPrompt(message.text);
      setStatus("Message is not durable yet; edit loaded as draft");
      return;
    }
    if (!sessionId || connectionState !== "open") return;
    setError(null);
    setErrorAction(null);
    setEditTarget(message);
  }, [applyMessageAsPrompt, connectionState, sessionId]);

  const submitEditedMessage = useCallback(async (editedText: string) => {
    const target = editTarget;
    const requestSessionId = sessionIdRef.current;
    if (!target || !sessionId || !requestSessionId || connectionState !== "open") return;
    if (editSubmitting || submitInFlightRef.current) return;
    const stopBarrierActive = stopping || stopInFlightRef.current;
    if (stopBarrierActive) {
      setError("Wait for Stop to finish before editing");
      setErrorAction(null);
      return;
    }
    const turnActive = streaming || tools.some((tool) => tool.state === "running") || turnStartedAt !== null || approval !== null || clarify !== null;
    if (turnActive) {
      setError("Wait for the active turn to finish before editing");
      setErrorAction(null);
      return;
    }

    let params;
    try {
      params = buildEditSubmitParams(requestSessionId, target, editedText, transcript);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction(null);
      return;
    }

    submitInFlightRef.current = true;
    const requestSessionGeneration = sessionGenerationRef.current;
    const requestStopGeneration = stopRequestGenerationRef.current;
    const requestToken = ++operationTokenRef.current;
    const operationIsCurrent = () => (
      sessionGenerationRef.current === requestSessionGeneration
      && sessionIdRef.current === requestSessionId
      && stopRequestGenerationRef.current === requestStopGeneration
      && operationTokenRef.current === requestToken
    );
    submitOwnerTokenRef.current = requestToken;
    submitOwnerKindRef.current = "edit";
    turnGenerationRef.current += 1;
    blockedTurnGenerationRef.current = turnGenerationRef.current;
    setEditSubmitting(true);
    setSubmitting(true);
    setError(null);
    setErrorAction(null);
    activePromptTextRef.current = editedText;
    setStatus("Editing…");
    try {
      const response = await gateway.request<EditSubmitResponse>("prompt.submit", params);
      if (!operationIsCurrent()) return;
      setTranscript((current) => applyEditedTranscript(
        current,
        target.id,
        editedText,
        response,
        `edited-${Date.now()}`,
      ) as TranscriptMessage[]);
      dispatchTimeline({ type: "reset" });
      setTools([]);
      setApprovalState(null);
      setClarifyState(null);
      setFailedPrompt(null);
      setEditTarget(null);
      setStreaming(true);
      setTurnStartedAt(Date.now());
      setStatus("Working…");
    } catch (reason: unknown) {
      if (!operationIsCurrent()) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction(null);
      setStatus("Edit failed");
    } finally {
      if (submitOwnerTokenRef.current === requestToken && submitOwnerKindRef.current === "edit") {
        submitInFlightRef.current = false;
        submitOwnerTokenRef.current = null;
        submitOwnerKindRef.current = null;
        setEditSubmitting(false);
        setSubmitting(false);
      }
    }
  }, [approval, clarify, connectionState, dispatchTimeline, editSubmitting, editTarget, gateway, sessionId, stopping, streaming, tools, transcript, turnStartedAt]);

  const cancelMessageEdit = useCallback(() => {
    if (!editSubmitting) setEditTarget(null);
  }, [editSubmitting]);

  const applyQuickPrompt = useCallback((prompt: string) => {
    setDraft(prompt);
    textareaRef.current?.focus();
  }, []);

  const stopVoiceRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const startVoiceRecording = useCallback(async () => {
    if (voiceState !== "idle") return;
    const recorderConstructor = typeof MediaRecorder === "undefined" ? undefined : MediaRecorder;
    if (typeof navigator === "undefined" || !canRecordVoice(navigator.mediaDevices, recorderConstructor)) {
      setVoiceError("Voice input is not available in this browser");
      return;
    }
    setVoiceError(null);
    setVoiceState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const supported = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]
        .filter((type) => recorderConstructor?.isTypeSupported(type));
      const preferredMime = chooseRecordingMimeType(supported);
      const recorder = preferredMime ? new MediaRecorder(stream, { mimeType: preferredMime }) : new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        stream.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
        const blob = new Blob(chunks, { type: recorder.mimeType || preferredMime || "audio/webm" });
        if (!blob.size) {
          setVoiceState("idle");
          setVoiceError("No audio was captured");
          return;
        }
        setVoiceState("transcribing");
        void fileDataUrl(blob)
          .then((dataUrl) => api.transcribeAudio(dataUrl, blob.type || "audio/webm", profile || undefined))
          .then((result) => {
            const transcript = result.transcript?.trim() ?? "";
            if (transcript) {
              setDraft((current) => appendVoiceTranscript(current, transcript));
              textareaRef.current?.focus();
            }
            setVoiceError(transcript ? null : "No speech was detected");
          })
          .catch((reason: unknown) => setVoiceError(reason instanceof Error ? reason.message : String(reason)))
          .finally(() => setVoiceState("idle"));
      }, { once: true });
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.start();
      setVoiceState("recording");
    } catch (reason: unknown) {
      stopMediaStream(mediaStreamRef.current);
      mediaStreamRef.current = null;
      mediaRecorderRef.current = null;
      setVoiceError(reason instanceof Error ? reason.message : "Microphone permission was denied");
      setVoiceState("idle");
    }
  }, [profile, voiceState]);

  const speakMessage = useCallback(async (message: string) => {
    const result = await api.speakText(message, profile || undefined);
    if (!result.data_url) throw new Error("Speech audio was not returned");
    audioRef.current?.pause();
    const audio = new Audio(result.data_url);
    audioRef.current = audio;
    await audio.play();
  }, [profile]);

  useEffect(() => () => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    stopMediaStream(mediaStreamRef.current);
    mediaStreamRef.current = null;
    mediaRecorderRef.current = null;
    audioRef.current?.pause();
    audioRef.current = null;
  }, []);

  const onComposerKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (slashPopoverRef.current?.handleKey(event)) return;
    if (shouldSubmitComposerKey(event.key, event.shiftKey, composingRef.current || event.nativeEvent.isComposing)) {
      event.preventDefault();
      void submit();
    }
  }, [submit]);

  const stop = useCallback(async () => {
    if (!sessionId || !streaming || connectionState !== "open" || stopInFlightRef.current) return;
    const stopSessionId = sessionIdRef.current;
    const stopAssistantId = assistantIdRef.current;
    const stopMessageId = activeMessageIdRef.current;
    const stopTurnGeneration = turnGenerationRef.current;
    const stopTurnId = activeTurnIdRef.current;
    const stopRequestToken = ++stopRequestGenerationRef.current;
    const stopSessionGeneration = sessionGenerationRef.current;
    const stopPromptText = activePromptTextRef.current;
    const stopTarget: StopTarget = {
      sessionId: stopSessionId,
      durableSessionId: durableSessionIdRef.current,
      sessionKey: sessionKeyRef.current,
      sessionGeneration: stopSessionGeneration,
      requestGeneration: stopRequestToken,
      stoppedAt: Date.now(),
      promptText: stopPromptText,
      assistantId: stopAssistantId,
      assistantText: stopAssistantId ? liveAssistantText(stopAssistantId) : null,
      messageId: stopMessageId,
      turnId: stopTurnId,
      turnGeneration: stopTurnGeneration,
    };
    stopTargetRef.current = stopTarget;
    if (stopTurnId) ignoredTurnIdsRef.current.add(stopTurnId);
    blockedTurnGenerationRef.current = stopTurnGeneration;
    stopInFlightRef.current = true;
    setStopping(true);
    setError(null);
    setErrorAction(null);
    setStatus("Stopping…");
    let interruptSucceeded = false;
    let stopTargetWasOwned = false;
    const stopTargetIsCurrent = () => shouldRestoreStopTarget(stopTarget, {
      sessionGeneration: sessionGenerationRef.current,
      requestGeneration: stopRequestGenerationRef.current,
      assistantId: assistantIdRef.current,
      turnId: activeTurnIdRef.current,
      turnGeneration: turnGenerationRef.current,
    });
    try {
      await gateway.request("session.interrupt", { session_id: stopSessionId });
      interruptSucceeded = true;
      if (stopTargetIsCurrent() && sessionIdRef.current === stopSessionId) {
        stopTargetWasOwned = true;
        retiredStopTargetRef.current = stopTarget;
        clearLocalTurnState();
        setStatus("Stopped");
      }
    }
    catch (reason: unknown) {
      if (stopTargetIsCurrent() && sessionIdRef.current === stopSessionId) {
        if (stopTurnId) ignoredTurnIdsRef.current.delete(stopTurnId);
        if (blockedTurnGenerationRef.current === stopTurnGeneration) blockedTurnGenerationRef.current = null;
        setError(reason instanceof Error ? reason.message : String(reason));
        setErrorAction("reconnect");
        setStatus("Error");
      }
    }
    finally {
      if (sessionGenerationRef.current === stopSessionGeneration && stopRequestGenerationRef.current === stopRequestToken) {
        const shouldResyncAfterStop = resyncAfterStopRef.current && connectionStateRef.current === "open";
        if (shouldResyncAfterStop) resyncAfterStopRef.current = false;
        if (interruptSucceeded && stopTargetWasOwned) retiredStopTargetRef.current = stopTarget;
        stopInFlightRef.current = false;
        stopTargetRef.current = null;
        setStopping(false);
        if (shouldResyncAfterStop) {
          queueMicrotask(() => {
            if (sessionGenerationRef.current === stopSessionGeneration && connectionStateRef.current === "open") startResyncRef.current?.();
          });
        }
      }
    }
  }, [clearLocalTurnState, connectionState, gateway, liveAssistantText, sessionId, streaming]);
  const respondApproval = useCallback(async (choice: string) => {
    const request = approval;
    const requestSessionId = sessionIdRef.current;
    if (!request || !sessionId || !requestSessionId) return;
    const requestSessionGeneration = sessionGenerationRef.current;
    const requestStopGeneration = stopRequestGenerationRef.current;
    const requestToken = ++operationTokenRef.current;
    const operationIsCurrent = () => sessionGenerationRef.current === requestSessionGeneration
      && sessionIdRef.current === requestSessionId
      && stopRequestGenerationRef.current === requestStopGeneration
      && operationTokenRef.current === requestToken;
    try {
      const result = await gateway.request<{ resolved?: unknown }>("approval.respond", { choice, request_id: request.request_id, session_id: requestSessionId, ...(profile ? { profile } : {}) });
      if (result?.resolved !== 1) {
        if (operationIsCurrent() && approvalRef.current?.request_id === request.request_id) {
          setError("Approval was stale, expired, or already resolved");
        }
        throw new Error("Approval was stale, expired, or already resolved");
      }
      if (operationIsCurrent() && approvalRef.current?.request_id === request.request_id) dismissApprovalState(request.request_id);
    } catch (reason: unknown) {
      if (!operationIsCurrent() || approvalRef.current?.request_id !== request.request_id) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      throw reason;
    }
  }, [approval, dismissApprovalState, gateway, profile, sessionId]);
  const respondClarify = useCallback(async (answer: string, questionId?: string) => {
    const request = clarify;
    const requestSessionId = sessionIdRef.current;
    if (!request || !sessionId || !requestSessionId || !answer.trim()) return;
    const params = questionId
      ? { answer, question_id: questionId, request_id: request.request_id, session_id: requestSessionId, ...(profile ? { profile } : {}) }
      : { answer, request_id: request.request_id, session_id: requestSessionId, ...(profile ? { profile } : {}) };
    const requestSessionGeneration = sessionGenerationRef.current;
    const requestStopGeneration = stopRequestGenerationRef.current;
    const requestToken = ++operationTokenRef.current;
    const operationIsCurrent = () => sessionGenerationRef.current === requestSessionGeneration
      && sessionIdRef.current === requestSessionId
      && stopRequestGenerationRef.current === requestStopGeneration
      && operationTokenRef.current === requestToken;
    try {
      const response = await gateway.request<{ remaining?: unknown }>("clarify.respond", params);
      if (operationIsCurrent() && clarifyRef.current?.request_id === request.request_id
        && shouldClearClarificationResponse(response, questionId)) setClarifyState(null);
    } catch (reason: unknown) {
      if (!operationIsCurrent() || clarifyRef.current?.request_id !== request.request_id) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      throw reason;
    }
  }, [clarify, gateway, profile, sessionId]);
  const retry = useCallback(() => {
    const retrySessionGeneration = sessionGenerationRef.current;
    clearReconnectTimer();
    reconnectAttemptRef.current = 0;
    setError(null);
    setErrorAction(null);
    void gateway.connect().catch((reason: unknown) => {
      if (sessionGenerationRef.current !== retrySessionGeneration) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setErrorAction("reconnect");
    });
  }, [clearReconnectTimer, gateway]);
  const resendFailedPrompt = useCallback(() => {
    if (failedPrompt) void submit(undefined, failedPrompt);
  }, [failedPrompt, submit]);
  const isWorking = isNativeChatWorking({
    streaming,
    runningTools: tools.filter((tool) => tool.state === "running").length,
    turnStartedAt,
    hasPendingInteraction: approval !== null || clarify !== null,
  });
  const activityStatus = isWorking
    ? (approval ? "Waiting for approval" : clarify ? "Waiting for clarification" : (status || "Thinking…"))
    : null;
  const pageStatus = error
    ? "Error"
    : connectionState !== "open"
    ? connectionLabel(connectionState)
    : resyncState === "syncing"
      ? chat.syncing
      : isWorking
        ? chat.working
        : chat.ready;
  const sessionActivityStatus: SessionActivityStatus = error
    ? "error"
    : connectionState !== "open"
      ? "offline"
      : approval || clarify
        ? "waiting"
        : isWorking
          ? "working"
          : "ready";
  const sessionStatuses = useMemo<Readonly<Record<string, SessionActivityStatus>>>(() => (
    resumeParam ? { [resumeParam]: sessionActivityStatus } : {}
  ), [resumeParam, sessionActivityStatus]);
  const elapsedSeconds = turnStartedAt == null ? 0 : Math.max(0, Math.floor((clockNow - turnStartedAt) / 1000));
  const measuredHeights = useMemo(
    () => {
      void virtualMeasureRevision;
      return filteredTranscript.map((message) => virtualRowHeightsRef.current.get(message.id) ?? 0);
    },
    [filteredTranscript, virtualMeasureRevision],
  );
  const virtualRange = useMemo(
    () => getVirtualRange(filteredTranscript.length, virtualViewport.scrollTop, virtualViewport.viewportHeight, measuredHeights, 144, 6),
    [filteredTranscript.length, measuredHeights, virtualViewport],
  );
  const visibleTranscript = filteredTranscript.slice(virtualRange.start, virtualRange.end);
  const lastAssistantId = [...displayTranscript].reverse().find((message) => message.role === "assistant")?.id;
  const connectionTone = connectionState === "open"
    ? "success"
    : connectionState === "error"
      ? "destructive"
      : connectionState === "connecting"
        ? "warning"
        : "secondary";
  const statusDotClass = error
    ? "bg-destructive"
    : connectionState !== "open"
    ? "bg-muted-foreground"
    : isWorking
      ? "bg-primary"
      : "bg-success";
  useEffect(() => {
    if (!isWorking) return;
    setClockNow(Date.now());
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [isWorking]);

  useLayoutEffect(() => {
    const container = transcriptRef.current;
    if (!container) return;
    for (const row of Array.from(container.querySelectorAll<HTMLElement>("[data-slot='transcript-row']"))) {
      const id = row.dataset.messageId;
      if (id) measureTranscriptRow(id, row);
    }
  }, [approval, clarify, error, filteredTranscript, measureTranscriptRow, status, tools, virtualRange.end, virtualRange.start]);

  return (
    <section
      data-slot="native-chat-shell"
      data-layout="desktop-like"
      className="flex min-h-0 min-w-0 flex-1 flex-col pb-4"
      aria-label="Native chat"
    >
      {commandPaletteOpen && (
        <CommandPalette
          onClose={closeCommandPalette}
          onFocusComposer={focusComposer}
          onNewChat={startNewChat}
          onToggleSessions={toggleSessionNavigator}
          onInsertPrompt={applyQuickPrompt}
          onBranchSession={sessionId ? branchSession : undefined}
          queuedCount={queuedPrompts.length}
          onClearQueue={() => setQueuedPrompts([])}
        />
      )}
      <header
        data-slot="chat-header"
        className="flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-3 border-b border-current/15 py-3"
      >
        <div className="flex min-w-0 items-center gap-1">
          {onOpenNavigation && (
            <Button
              ghost
              size="icon"
              type="button"
              className="shrink-0 lg:hidden"
              aria-label={t.app.openNavigation}
              onClick={onOpenNavigation}
            >
              <Menu />
            </Button>
          )}
          <Button
            ghost
            size="icon"
            type="button"
            className="shrink-0 lg:hidden"
            aria-label={t.sessions.title}
            aria-expanded={mobileSessionNavigatorOpen}
            aria-controls="native-chat-session-navigator"
            data-session-navigator-toggle
            onClick={() => setMobileSessionNavigatorOpen((open) => !open)}
          >
            <MessageSquare />
          </Button>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold">{chat.title}</h1>
            <p className="truncate text-sm text-text-secondary">{chat.subtitle}</p>
          </div>
        </div>
        <div
          data-slot="chat-routing-controls"
          className="flex min-w-0 flex-wrap items-center justify-end gap-2"
        >
          <label className="sr-only" htmlFor="native-chat-model">Chat model</label>
          <select
            id="native-chat-model"
            aria-label="Chat model"
            className="h-9 max-w-44 min-w-0 border border-midground/15 bg-background/40 px-2 py-1 font-courier text-[16px] text-midground focus-visible:border-midground/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30 sm:text-xs"
            value={routingSelection.model ? `${routingSelection.model.provider}:${routingSelection.model.model}` : "adaptive"}
            onChange={(event) => {
              if (event.target.value === "adaptive") return void changeRouting("", "", routingSelection.reasoning);
              const choice = modelChoices.find((item) => `${item.provider}:${item.model}` === event.target.value);
              if (choice) changeRouting(choice.model, choice.provider, routingSelection.reasoning);
            }}
          >
            <option value="adaptive">Adaptive</option>
            {modelChoices.map((choice) => <option key={`${choice.provider}:${choice.model}`} value={`${choice.provider}:${choice.model}`}>{choice.label}</option>)}
          </select>
          <label className="sr-only" htmlFor="native-chat-reasoning">Reasoning level</label>
          <select
            id="native-chat-reasoning"
            aria-label="Reasoning level"
            className="h-9 min-w-0 border border-midground/15 bg-background/40 px-2 py-1 font-courier text-[16px] text-midground focus-visible:border-midground/30 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30 sm:text-xs"
            value={routingSelection.reasoning}
            onChange={(event) => changeRouting(routingSelection.model?.model ?? "", routingSelection.model?.provider ?? "", event.target.value as NativeReasoningLevel)}
          >
            {NATIVE_REASONING_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
          <Badge
            data-slot="connection-badge"
            tone={connectionTone}
            className="shrink-0"
          >
            {connectionLabel(connectionState)}
          </Badge>
        </div>
      </header>

      <div
        data-slot="chat-notices"
        data-resync-state={resyncState}
        className={cn("flex shrink-0 flex-col gap-2", error && "pt-3")}
      >
        {error && (
          <div
            data-slot="chat-error"
            role="alert"
            className="flex flex-wrap items-center gap-2 border-l-2 border-destructive px-3 py-2 text-sm text-destructive"
          >
            <span className="min-w-0 flex-1 wrap-break-word">{error}</span>
            {errorAction === "resend" ? (
              <Button
                ghost
                size="sm"
                type="button"
                aria-label="Retry send"
                prefix={<RotateCcw />}
                className="shrink-0"
                onClick={resendFailedPrompt}
              >
                Retry send
              </Button>
            ) : (
              <Button
                ghost
                size="sm"
                type="button"
                prefix={<RotateCcw />}
                className="shrink-0"
                onClick={retry}
              >
                Retry
              </Button>
            )}
          </div>
        )}
        {resyncState === "syncing" && (
          <div data-slot="chat-resync" role="status" className="border-l-2 border-primary px-3 py-2 text-sm text-primary">
            Syncing conversation…
          </div>
        )}
        {resyncState === "partial" && (
          <div data-slot="chat-resync" role="status" className="border-l-2 border-warning px-3 py-2 text-sm text-text-secondary">
            Conversation synced; some older history is unavailable.
          </div>
        )}
        {resyncState === "error" && !error && (
          <div data-slot="chat-resync" role="alert" className="border-l-2 border-destructive px-3 py-2 text-sm text-destructive">
            Conversation sync failed. Retry the connection to continue.
          </div>
        )}
        {voiceError && (
          <div data-slot="voice-error" role="alert" className="flex items-center gap-2 border-l-2 border-warning px-3 py-2 text-sm text-warning">
            <span className="min-w-0 flex-1 wrap-break-word">{voiceError}</span>
            <Button ghost size="sm" type="button" onClick={() => setVoiceError(null)}>Dismiss</Button>
          </div>
        )}
      </div>

      <div
        data-slot="chat-body"
        className={cn(
          "grid min-h-0 flex-1",
          mobileSessionNavigatorOpen
            ? "grid-rows-[minmax(9rem,12rem)_minmax(0,1fr)]"
            : "grid-rows-[minmax(0,1fr)]",
          "lg:grid-cols-[16rem_minmax(0,1fr)] lg:grid-rows-1",
        )}
      >
        <aside
          id="native-chat-session-navigator"
          data-slot="session-navigator"
          data-mobile-open={mobileSessionNavigatorOpen ? "true" : "false"}
          role="complementary"
          aria-label={t.sessions.title}
          className={cn(
            "min-h-0 min-w-0 overflow-hidden border-b border-current/15 pt-3 pb-3 lg:border-r lg:border-b-0 lg:pt-4 lg:pr-4 lg:pb-0",
            !mobileSessionNavigatorOpen && "hidden lg:block",
          )}
        >
          <ChatSessionList
            activeSessionId={resumeParam}
            profile={profile ?? undefined}
            onPicked={() => setMobileSessionNavigatorOpen(false)}
            onNewChat={startNewChat}
            sessionStatuses={sessionStatuses}
          />
        </aside>

        <div
          data-slot="transcript-pane"
          role="region"
          aria-label="Conversation transcript"
          className="relative flex min-h-0 min-w-0 flex-col lg:pl-4"
        >
          <div
            data-slot="transcript-search"
            className="flex shrink-0 items-center gap-2 border-b border-current/10 py-2"
          >
            <label className="sr-only" htmlFor="native-chat-message-search">Search message content</label>
            <input
              id="native-chat-message-search"
              aria-label="Search message content"
              className="min-w-0 flex-1 rounded border border-border bg-background/40 px-2 py-1 text-sm text-foreground outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/40"
              placeholder="Search conversation"
              value={transcriptQuery}
              onChange={(event) => setTranscriptQuery(event.target.value)}
            />
            <span className="shrink-0 text-xs text-muted-foreground" aria-live="polite">
              {transcriptQuery.trim() ? `${filteredTranscript.length} match${filteredTranscript.length === 1 ? "" : "es"}` : "Search"}
            </span>
            {transcriptQuery.trim() && (
              <Button ghost size="icon" type="button" aria-label="Clear message search" onClick={() => setTranscriptQuery("")}>
                <X aria-hidden />
              </Button>
            )}
          </div>
          <div
            ref={transcriptRef}
            onScroll={handleTranscriptScroll}
            data-testid="native-chat-transcript"
            data-slot="transcript"
            className="min-h-0 flex-1 space-y-3 overflow-y-auto py-4 pr-1"
            role="log"
            aria-live="polite"
            aria-relevant="additions text"
          >
            {displayTranscript.length > 0 && filteredTranscript.length === 0 && transcriptQuery.trim() && (
              <div data-slot="transcript-search-empty" className="pb-3 py-8 text-sm text-muted-foreground" role="status">
                No messages match “{transcriptQuery.trim()}”.
              </div>
            )}
            {displayTranscript.length === 0 && !approval && !clarify && (
              <div data-slot="chat-empty-state" className="max-w-2xl space-y-3 pb-3 py-8 text-sm text-text-secondary">
                <p>{chat.startConversation}</p>
                <div className="space-y-2" role="group" aria-label={chat.tryPrompt}>
                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{chat.tryPrompt}</p>
                  <div className="flex flex-wrap gap-2">
                    {QUICK_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        data-testid="quick-prompt"
                        type="button"
                        className="rounded-md border border-border bg-card px-3 py-2 text-left text-xs text-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring"
                        onClick={() => applyQuickPrompt(prompt)}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {displayTranscript.length > 0 && (
              <>
                <div
                  data-slot="transcript-virtual-spacer"
                  data-total-height={virtualRange.totalHeight}
                  data-range-start={virtualRange.start}
                  data-range-end={virtualRange.end}
                  style={{ height: virtualRange.offsetTop }}
                  aria-hidden="true"
                />
                {visibleTranscript.map((message) => (
                  <div
                    key={message.id}
                    ref={(node) => measureTranscriptRow(message.id, node)}
                    data-slot="transcript-row"
                    data-message-id={message.id}
                    className="pb-3"
                  >
                    {activityStatus && message.id === assistantIdRef.current && (
                      <div
                        data-slot="turn-activity"
                        className="mb-1 flex w-fit max-w-[85%] items-center gap-2 border-l-2 border-primary px-2 py-1 text-sm text-primary"
                        aria-label="Agent activity"
                      >
                        <span aria-hidden className="inline-block h-2 w-2 shrink-0 rounded-full bg-primary motion-safe:animate-pulse" />
                        <span>{activityStatus}</span>
                        <span className="font-mono text-xs text-text-secondary">{elapsedSeconds}s</span>
                      </div>
                    )}
                    <TranscriptBubble
                      message={message}
                      sessionId={durableSessionId ?? undefined}
                      onUseAsPrompt={applyMessageAsPrompt}
                      onEdit={beginMessageEdit}
                      onRegenerate={message.id === lastAssistantId ? runLastPromptAgain : undefined}
                      onSpeak={speakMessage}
                    />
                    {message.id === lastAssistantId && tools.length > 0 && (
                      <ToolTimeline tools={tools} className="mt-3 space-y-2" />
                    )}
                  </div>
                ))}
                <div
                  data-slot="transcript-virtual-spacer"
                  data-total-height={virtualRange.totalHeight}
                  data-range-start={virtualRange.start}
                  data-range-end={virtualRange.end}
                  style={{ height: virtualRange.bottomSpacer }}
                  aria-hidden="true"
                />
              </>
            )}
            {!lastAssistantId && tools.length > 0 && (
              <ToolTimeline tools={tools} className="pb-3 space-y-2" />
            )}
            {approval && <div className="pb-3"><ApprovalCard request={approval} onRespond={respondApproval} /></div>}
            {clarify && <div className="pb-3"><ClarificationCard request={clarify} onRespond={respondClarify} /></div>}
          </div>

          {showScrollToBottom && (
            <Button
              ghost
              size="sm"
              type="button"
              prefix={<ArrowDown />}
              data-testid="scroll-to-bottom"
              aria-label="Scroll to latest message"
              className="absolute right-2 bottom-14 z-10 border border-border bg-card shadow-md"
              onClick={scrollToBottom}
            >
              Jump to latest
            </Button>
          )}

          <div
            data-slot="chat-status"
            role="status"
            aria-label="Chat status"
            className="flex shrink-0 items-center gap-2 border-t border-current/15 py-2 text-xs text-text-secondary"
          >
            <span aria-hidden className={cn("inline-block h-1.5 w-1.5 shrink-0 rounded-full", statusDotClass)} />
            <span>{connectionState === "open" ? "Connected" : connectionLabel(connectionState)}</span>
            <span aria-hidden>·</span>
            <span>{pageStatus}</span>
          </div>
        </div>
      </div>

      <form
        data-slot="chat-composer"
        aria-label="Message composer"
        aria-busy={submitting}
        className="flex shrink-0 flex-col gap-2 border-t border-current/15 bg-background/30 pt-3"
        onSubmit={submit}
        onDragOver={(event) => { event.preventDefault(); }}
        onDrop={(event) => { event.preventDefault(); addFiles(event.dataTransfer.files); }}
      >
        {queuedPrompts.length > 0 && (
          <div
            data-slot="prompt-queue"
            className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/20 p-2 text-xs"
            aria-label="Queued prompts"
            aria-live="polite"
          >
            <span className="font-medium">Queued ({queuedPrompts.length})</span>
            {queuedPrompts.map((item, index) => (
              <div key={item.id} className="flex min-w-0 items-center gap-1 rounded border border-border bg-background/50 px-2 py-1">
                <span className="max-w-56 truncate" title={item.text}>{index + 1}. {item.text}</span>
                <Button
                  ghost
                  size="icon"
                  type="button"
                  aria-label={`Remove queued prompt ${index + 1}`}
                  className="shrink-0"
                  onClick={() => setQueuedPrompts((current) => current.filter((entry) => entry.id !== item.id))}
                >
                  <X />
                </Button>
              </div>
            ))}
            <Button ghost size="sm" type="button" aria-label="Clear prompt queue" onClick={() => setQueuedPrompts([])}>Clear</Button>
          </div>
        )}
        {attachments.length > 0 && (
          <div
            data-slot="attachment-list"
            className="flex flex-wrap gap-2 rounded-md border border-border bg-muted/20 p-2"
            aria-label="Pending attachments"
            aria-live="polite"
          >
            {attachments.map((item) => (
              <div
                key={item.id}
                data-slot="attachment"
                className="flex min-w-0 items-center gap-1 border border-midground/15 bg-background/40 px-2 py-1 text-xs"
              >
                {item.previewUrl && (
                  <img
                    data-slot="attachment-preview"
                    src={item.previewUrl}
                    alt={`Preview of ${item.file.name}`}
                    className="h-8 w-8 shrink-0 rounded object-cover"
                  />
                )}
                <span className="min-w-0 max-w-48 truncate" title={item.file.name}>{item.file.name}</span>
                <span className="shrink-0 text-text-secondary">{formatFileSize(item.file.size)}</span>
                <span className="text-text-secondary">
                  {item.state === "uploading" ? "Uploading…" : item.state === "error" ? item.error : item.state === "attached" ? "Ready" : "Queued"}
                </span>
                {item.state === "error" && (
                  <Button
                    ghost
                    size="sm"
                    type="button"
                    aria-label={`Retry ${item.file.name}`}
                    prefix={<RotateCcw />}
                    onClick={() => void stageAttachment(item)}
                  >
                    Retry
                  </Button>
                )}
                <Button
                  ghost
                  size="icon"
                  type="button"
                  aria-label={`Remove ${item.file.name}`}
                  className="shrink-0 text-text-secondary hover:text-destructive"
                  onClick={() => removeAttachment(item.id)}
                >
                  <X />
                </Button>
              </div>
            ))}
          </div>
        )}
        <div data-slot="composer-controls" className="flex min-w-0 flex-wrap items-end gap-2 rounded-md border border-border bg-card/50 p-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf,.txt,.md,.csv,.json,.yaml,.yml,.js,.jsx,.ts,.tsx,.py,.html,.css"
            multiple
            className="hidden"
            onChange={(event) => { if (event.target.files) addFiles(event.target.files); event.currentTarget.value = ""; }}
          />
          <div className="flex min-w-0 basis-full items-end gap-2 sm:basis-0 sm:flex-1">
            <Button
              ghost
              size="icon"
              type="button"
              aria-label="Add attachment"
              className="shrink-0"
              disabled={connectionState !== "open" || !sessionId}
              onClick={() => fileInputRef.current?.click()}
            >
              <Paperclip />
            </Button>
            <Button
              ghost
              size="icon"
              type="button"
              aria-label={voiceState === "recording" ? chat.stopVoiceRecording : chat.recordVoice}
              className={cn("shrink-0", voiceState === "recording" && "text-destructive")}
              disabled={voiceState === "starting" || voiceState === "transcribing"}
              onClick={() => voiceState === "recording" ? stopVoiceRecording() : void startVoiceRecording()}
            >
              <Mic aria-hidden />
            </Button>
            <div className="relative min-w-0 flex-1">
              <SlashPopover
                ref={slashPopoverRef}
                input={draft}
                gw={gateway}
                onApply={setDraft}
              />
              <textarea
                ref={textareaRef}
                aria-label="Message"
                aria-describedby="native-chat-composer-hint"
                className="min-h-20 w-full resize-y border border-border bg-background/40 px-3 py-2 font-courier text-[16px] text-foreground outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-1 focus-visible:ring-ring/40 sm:text-sm"
                value={draft}
                disabled={connectionState !== "open" || !sessionId}
                placeholder={chat.messagePlaceholder}
                onPaste={(event) => { if (event.clipboardData.files.length) { event.preventDefault(); addFiles(event.clipboardData.files); } }}
                onChange={(event) => setDraft(event.target.value)}
                onCompositionStart={() => { composingRef.current = true; }}
                onCompositionEnd={() => { composingRef.current = false; }}
                onKeyDown={onComposerKeyDown}
              />
            </div>
          </div>
          <div data-slot="composer-actions" className="ml-auto flex shrink-0 items-center gap-2">
            <Button
              type="submit"
              size="sm"
              prefix={<Send />}
              aria-label={isWorking ? "Queue message" : "Send message"}
              className="shrink-0"
              disabled={submitting || (!draft.trim() && !attachments.some((item) => item.state === "attached")) || connectionState !== "open" || !sessionId}
            >
              {submitting ? "Sending…" : isWorking ? chat.queue : chat.send}
            </Button>
            {isWorking && (
              <Button
                destructive
                outlined
                type="button"
                size="sm"
                prefix={<Square />}
                aria-label="Stop"
                className="shrink-0"
                disabled={stopping || connectionState !== "open"}
                onClick={() => void stop()}
              >
                {stopping ? "Stopping…" : chat.stop}
              </Button>
            )}
          </div>
        </div>
        <div data-slot="composer-meta" className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1 text-xs text-muted-foreground">
          <span id="native-chat-composer-hint" data-slot="composer-attachment-hint" className="inline-flex items-center gap-1.5">
            <Paperclip aria-hidden className="h-3.5 w-3.5" />
            {chat.dropPasteAttach}
          </span>
          <span data-slot="composer-status" role="status" aria-live="polite">
            {voiceState === "starting"
              ? chat.requestingMicrophone
              : voiceState === "recording"
                ? "Recording… tap the microphone to stop"
                : voiceState === "transcribing"
                  ? chat.transcribing
                  : submitting
                    ? "Sending…"
                    : connectionState !== "open"
                      ? "Waiting for connection…"
                      : status ?? (streaming ? chat.working : chat.ready)}
          </span>
        </div>
      </form>
      {editTarget && (
        <EditMessageDialog
          key={editTarget.id}
          initialText={editTarget.text}
          loading={editSubmitting}
          onCancel={cancelMessageEdit}
          onConfirm={(text) => { void submitEditedMessage(text); }}
          open
        />
      )}
    </section>
  );
}
