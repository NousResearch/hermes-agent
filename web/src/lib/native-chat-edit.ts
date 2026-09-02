export type EditableTranscriptMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  rowId?: number;
};

export type EditSubmitResponse = {
  survivor_user_row_ids?: Array<number | null>;
  survivor_row_id_map?: Record<string, number | null>;
};

export type EditSubmitParams = {
  session_id: string;
  text: string;
  truncate_before_row_id: number;
  confirm_truncate: true;
  confirm_empty_truncate?: true;
  rebind_survivor_row_ids: number[];
};

export function parseDurableRowId(value: unknown): number | undefined {
  if (typeof value === "boolean" || value === null || value === undefined) return undefined;
  const candidate = typeof value === "string" && value.trim() !== "" ? Number(value) : value;
  return typeof candidate === "number" && Number.isSafeInteger(candidate) && candidate >= 0
    ? candidate
    : undefined;
}

export function buildEditSubmitParams(
  sessionId: string,
  target: EditableTranscriptMessage,
  editedText: string,
  messages: readonly EditableTranscriptMessage[],
): EditSubmitParams {
  const rowId = parseDurableRowId(target.rowId);
  const text = editedText.trim();
  if (target.role !== "user" || rowId === undefined) throw new Error("This message is not a durable user turn");
  if (!text) throw new Error("Edited message cannot be empty");

  const users = messages.filter((message) => message.role === "user");
  const survivorIds = messages
    .filter((message) => message.role === "user")
    .map((message) => parseDurableRowId(message.rowId))
    .filter((candidate): candidate is number => candidate !== undefined);
  return {
    session_id: sessionId,
    text,
    truncate_before_row_id: rowId,
    confirm_truncate: true,
    ...(users[0]?.id === target.id ? { confirm_empty_truncate: true as const } : {}),
    rebind_survivor_row_ids: survivorIds,
  };
}

function mappedRowId(response: EditSubmitResponse, oldRowId: number | undefined, userOrdinal: number): number | undefined {
  const map = response.survivor_row_id_map;
  const hasRebindFields = map !== undefined || response.survivor_user_row_ids !== undefined;
  if (!hasRebindFields) return oldRowId;
  if (oldRowId !== undefined && map && Object.prototype.hasOwnProperty.call(map, String(oldRowId))) {
    return parseDurableRowId(map[String(oldRowId)]);
  }
  return parseDurableRowId(response.survivor_user_row_ids?.[userOrdinal]);
}

function rebindSurvivors(
  messages: readonly EditableTranscriptMessage[],
  response: EditSubmitResponse,
): EditableTranscriptMessage[] {
  let userOrdinal = 0;
  return messages.map((message) => {
    if (message.role !== "user") return { ...message };
    const nextRowId = mappedRowId(response, parseDurableRowId(message.rowId), userOrdinal);
    userOrdinal += 1;
    if (nextRowId === undefined) return { ...message, rowId: undefined };
    return { ...message, id: String(nextRowId), rowId: nextRowId };
  });
}

export function applyEditedTranscript(
  messages: readonly EditableTranscriptMessage[],
  targetId: string,
  editedText: string,
  response: EditSubmitResponse,
  newMessageId: string,
): EditableTranscriptMessage[] {
  const targetIndex = messages.findIndex((message) => message.id === targetId);
  if (targetIndex < 0) return messages.map((message) => ({ ...message }));
  return [
    ...rebindSurvivors(messages.slice(0, targetIndex), response),
    { id: newMessageId, role: "user", text: editedText.trim() },
  ];
}
