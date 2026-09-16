import { describe, expect, it } from "vitest";

import {
  mergeCompletedAssistantMessage,
  mergeInflightTranscript,
  mergeLiveTimelineTranscript,
  mergeSnapshotTranscript,
  snapshotHasField,
  snapshotMatchesSession,
  type ReconcileMessage,
} from "./native-chat-reconcile";

const user = (id: string, text: string): ReconcileMessage => ({ id, role: "user", text });
const assistant = (id: string, text: string, streaming = false): ReconcileMessage => ({ id, role: "assistant", text, streaming });

describe("native chat snapshot reconciliation", () => {
  it("does not use turnId as identity when only one assistant messageId is present", () => {
    const current: ReconcileMessage[] = [
      user("user-1", "prompt"),
      { id: "assistant-old", turnId: "turn-1", role: "assistant", text: "old answer" },
    ];
    const live: ReconcileMessage[] = [
      { id: "assistant-new", turnId: "turn-1", messageId: "message-new", role: "assistant", text: "new answer" },
    ];
    expect(mergeLiveTimelineTranscript(current, live, "prompt")).toEqual([...current, live[0]]);
  });

  it("does not content-merge assistants whose messageIds conflict", () => {
    const current: ReconcileMessage[] = [
      user("user-1", "prompt"),
      { id: "assistant-old", turnId: "turn-1", messageId: "message-old", role: "assistant", text: "same answer" },
    ];
    const live: ReconcileMessage[] = [
      { id: "assistant-new", turnId: "turn-1", messageId: "message-new", role: "assistant", text: "same answer with suffix" },
    ];
    expect(mergeLiveTimelineTranscript(current, live, "prompt")).toEqual([...current, live[0]]);
  });

  it("does not merge assistant messages with conflicting message IDs in one turn", () => {
    const current: ReconcileMessage[] = [
      { id: "user-1", role: "user", text: "prompt" },
      { id: "assistant-old", turnId: "turn-1", messageId: "message-old", role: "assistant", text: "old answer" },
    ];
    const live: ReconcileMessage[] = [
      { id: "assistant-new", turnId: "turn-1", messageId: "message-new", role: "assistant", text: "new answer" },
    ];
    expect(mergeLiveTimelineTranscript(current, live, "prompt")).toEqual([...current, live[0]]);
  });

  it("does not attach an unmatched completion error to an assistant-first row", () => {
    const current: ReconcileMessage[] = [assistant("existing", "existing answer")];
    expect(mergeCompletedAssistantMessage(
      current,
      "new-completion",
      "new error text",
      "missing prompt",
      "turn failed",
    )).toEqual([
      ...current,
      { id: "new-completion", role: "assistant", text: "new error text", error: "turn failed" },
    ]);
  });

  it("matches durable replay assistants by messageId without rowId", () => {
    const current: ReconcileMessage[] = [
      user("live-user", "replay prompt"),
      { id: "durable-assistant", messageId: "durable-message", role: "assistant", text: "replay answer" },
    ];
    const live: ReconcileMessage[] = [
      { id: "live-assistant", role: "assistant", text: "replay answer", streaming: false },
    ];
    expect(mergeLiveTimelineTranscript(current, live)).toEqual(current);
  });

  it("keeps a live streamed suffix when the durable snapshot is behind", () => {
    const snapshot = [user("u1", "question"), assistant("a1", "hello")];
    const current = [user("u1", "question"), assistant("a1", "hello world", true)];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual([
      user("u1", "question"),
      assistant("a1", "hello world", true),
    ]);
  });

  it("uses the durable snapshot when it contains the newer completed text", () => {
    const snapshot = [assistant("a1", "complete answer")];
    const current = [assistant("a1", "complete", true)];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual(snapshot);
  });

  it("keeps a longer completed current answer when a reconnect snapshot is stale", () => {
    const snapshot = [user("snapshot-u", "prompt"), assistant("snapshot-a", "complete answer")];
    const current = [user("live-u", "prompt"), assistant("live-a", "complete answer with more detail")];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual([
      user("snapshot-u", "prompt"),
      assistant("live-a", "complete answer with more detail"),
    ]);
  });

  it("does not duplicate equivalent assistant messages without stable ids", () => {
    const snapshot = [user("snapshot-u", "prompt"), assistant("snapshot-a", "same answer")];
    const current = [user("live-u", "prompt"), assistant("live-a", "same answer")];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual(snapshot);
  });

  it("reconciles synthetic repeated turns without duplicating durable history", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "1", rowId: 1, role: "user", text: "repeat" },
      { id: "2", rowId: 2, role: "assistant", text: "same answer" },
      { id: "3", rowId: 3, role: "user", text: "repeat" },
      { id: "4", rowId: 4, role: "assistant", text: "same answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "repeat"),
      assistant("local-assistant-1", "same answer"),
      user("local-user-2", "repeat"),
      assistant("local-assistant-2", "same answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual(snapshot);
  });

  it("keeps new repeated turns when the snapshot is behind", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "1", rowId: 1, role: "user", text: "repeat" },
      { id: "2", rowId: 2, role: "assistant", text: "same answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "repeat"),
      assistant("local-assistant-1", "same answer"),
      user("local-user-2", "repeat"),
      assistant("local-assistant-2", "same answer"),
      user("local-user-3", "repeat"),
      assistant("local-assistant-3", "same answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual([
      ...snapshot,
      user("local-user-2", "repeat"),
      assistant("local-assistant-2", "same answer"),
      user("local-user-3", "repeat"),
      assistant("local-assistant-3", "same answer"),
    ]);
  });

  it("inserts an unmatched internal live segment before a later prefix anchor", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "u1", rowId: 1, role: "user", text: "first" },
      { id: "a1", rowId: 2, role: "assistant", text: "first answer" },
      { id: "u3", rowId: 5, role: "user", text: "third" },
      { id: "a3", rowId: 6, role: "assistant", text: "third answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-u1", "first"),
      assistant("local-a1", "first answer"),
      user("local-u2", "second"),
      assistant("local-a2", "second answer"),
      user("local-u3", "third"),
      assistant("local-a3", "third answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual([
      snapshot[0],
      snapshot[1],
      current[2],
      current[3],
      snapshot[2],
      snapshot[3],
    ]);
  });

  it("keeps a streaming answer after the latest user when an earlier answer has the same prefix", () => {
    const current: ReconcileMessage[] = [
      user("u1", "first prompt"),
      assistant("a1", "OK"),
      user("u2", "second prompt"),
    ];
    const live = [assistant("live-a2", "OK", true)];

    expect(mergeLiveTimelineTranscript(current, live)).toEqual([
      ...current,
      ...live,
    ]);
  });

  it("does not absorb an identical untagged backend turn without an active-user anchor", () => {
    const current: ReconcileMessage[] = [
      user("u1", "prompt"),
      assistant("a1", "same answer"),
    ];
    const live = [assistant("live-a2", "same answer", true)];

    expect(mergeLiveTimelineTranscript(current, live)).toEqual([
      ...current,
      ...live,
    ]);
  });

  it("updates a durable current-turn answer instead of appending a duplicate", () => {
    const current: ReconcileMessage[] = [
      user("u1", "first prompt"),
      assistant("a1", "OK"),
      user("u2", "second prompt"),
      { id: "durable-a2", rowId: 4, role: "assistant", text: "OK" },
    ];

    const merged = mergeCompletedAssistantMessage(current, "live-a2", "OK", "second prompt");
    expect(merged).toHaveLength(4);
    expect(merged[3]).toMatchObject({
      id: "durable-a2",
      rowId: 4,
      role: "assistant",
      text: "OK",
      streaming: false,
    });
  });

  it("retains a longer durable current-turn answer when completion is a shorter prefix", () => {
    const current: ReconcileMessage[] = [
      user("u1", "first prompt"),
      assistant("a1", "previous answer"),
      user("u2", "second prompt"),
      { id: "durable-a2", rowId: 4, role: "assistant", text: "OK, here is more" },
    ];

    const merged = mergeCompletedAssistantMessage(current, "live-a2", "OK", "second prompt");
    expect(merged).toHaveLength(4);
    expect(merged[3]?.text).toBe("OK, here is more");
  });

  it("clears stale error metadata when an assistant completes successfully", () => {
    const merged = mergeCompletedAssistantMessage(
      [{ id: "assistant-1", role: "assistant", text: "old answer", error: "previous failure" }],
      "assistant-1",
      "new answer",
    );

    expect(merged[0]).toMatchObject({ id: "assistant-1", text: "new answer", streaming: false });
    expect(merged[0]).not.toHaveProperty("error");
  });

  it("clears stale live error when a successful durable snapshot wins", () => {
    const merged = mergeSnapshotTranscript(
      [{ id: "durable-u", role: "user", text: "prompt" }, { id: "durable-a", role: "assistant", text: "answer" }],
      [{ id: "live-u", role: "user", text: "prompt" }, { id: "live-a", role: "assistant", text: "answer", error: "old failure" }],
    );

    expect(merged).toEqual([
      { id: "durable-u", role: "user", text: "prompt" },
      { id: "durable-a", role: "assistant", text: "answer" },
    ]);
  });

  it("decorates the current-turn durable assistant when an error has no identity or text", () => {
    const current: ReconcileMessage[] = [
      user("u1", "current prompt"),
      { id: "durable-a", rowId: 2, role: "assistant", text: "partial answer" },
    ];

    const merged = mergeCompletedAssistantMessage(current, "synthetic", "", "current prompt", "provider failed");
    expect(merged).toHaveLength(2);
    expect(merged[1]).toMatchObject({ id: "durable-a", error: "provider failed", streaming: false });
  });

  it("clears live error when a longer durable answer wins", () => {
    const merged = mergeSnapshotTranscript(
      [{ id: "u1", role: "user", text: "prompt" }, { id: "a1", role: "assistant", text: "long durable answer" }],
      [{ id: "u2", role: "user", text: "prompt" }, { id: "a2", role: "assistant", text: "long", error: "stale" }],
    );
    expect(merged.find((message) => message.role === "assistant")?.error).toBeUndefined();
  });
  it("does not create an empty assistant row for an unanchored terminal error", () => {
    const current: ReconcileMessage[] = [{ id: "old-a", role: "assistant", text: "old answer" }];
    expect(mergeCompletedAssistantMessage(current, "synthetic", "", null, "provider failed")).toEqual(current);
  });

  it("reconciles backend-originated replay with durable assistant when no active prompt exists", () => {
    const merged = mergeLiveTimelineTranscript(
      [{ id: "durable-u", role: "user", text: "backend prompt", rowId: 4 }, { id: "durable-a", role: "assistant", text: "backend answer", rowId: 5 }],
      [{ id: "live-a", role: "assistant", text: "backend answer", streaming: false }],
      null,
    );
    expect(merged).toHaveLength(2);
    expect(merged.filter((message) => message.role === "assistant")).toHaveLength(1);
  });

  it("does not match live assistant content across the next user turn", () => {
    const current: ReconcileMessage[] = [
      user("u1", "active prompt"),
      user("u2", "later prompt"),
      assistant("a2", "same answer"),
    ];
    expect(mergeLiveTimelineTranscript(current, [assistant("live", "same answer")], "active prompt")).toEqual([
      ...current,
      assistant("live", "same answer"),
    ]);
  });

  it("does not attach a terminal error across the next user turn", () => {
    const current: ReconcileMessage[] = [
      user("u1", "active prompt"),
      user("u2", "later prompt"),
      assistant("a2", "later answer"),
    ];
    expect(mergeCompletedAssistantMessage(current, "live", "", "active prompt", "active failure")).toEqual(current);
  });

  it("does not attach a text-bearing error across an earlier user turn", () => {
    const current: ReconcileMessage[] = [
      user("u1", "old prompt"),
      { ...assistant("a1", "same answer"), error: "old failure" },
      user("u2", "current prompt"),
    ];
    expect(mergeCompletedAssistantMessage(current, "live", "same answer", null, "current failure")).toEqual([
      ...current,
      { id: "live", role: "assistant", text: "same answer", error: "current failure" },
    ]);
  });

  it("reconciles a local unbound completion with an equivalent durable assistant", () => {
    const merged = mergeCompletedAssistantMessage(
      [
        { id: "durable-user", role: "user", text: "prompt" },
        { id: "durable-answer", role: "assistant", text: "same answer" },
      ],
      "assistant-9",
      "same answer",
      null,
    );
    expect(merged).toHaveLength(2);
    expect(merged.filter((message) => message.role === "assistant")).toHaveLength(1);
  });

  it("reconciles a same-prompt inflight pair with the current durable assistant", () => {
    const merged = mergeInflightTranscript(
      [
        { id: "durable-user", rowId: 10, role: "user", text: "same prompt" },
        { id: "durable-assistant", rowId: 11, role: "assistant", text: "partial", streaming: true },
      ],
      [
        { id: "inflight-user", role: "user", text: "same prompt" },
        { id: "inflight-assistant", role: "assistant", text: "partial answer", streaming: true },
      ],
      "same prompt",
    );
    expect(merged.filter((message) => message.role === "user")).toHaveLength(1);
    expect(merged.filter((message) => message.role === "assistant")).toHaveLength(1);
    expect(merged.find((message) => message.role === "assistant")?.text).toBe("partial answer");
  });

  it("reconciles a resume-like snapshot and same-prompt inflight pair to one user row when activeUserText is null or omitted", () => {
    const current: ReconcileMessage[] = [
      { id: "durable-user", rowId: 10, role: "user", text: "resume prompt" },
      { id: "durable-assistant", rowId: 11, role: "assistant", text: "partial", streaming: true },
    ];
    const inflight: ReconcileMessage[] = [
      { id: "inflight-user", role: "user", text: "resume prompt" },
      { id: "inflight-assistant", role: "assistant", text: "partial answer", streaming: true },
    ];

    const mergedOmitted = mergeInflightTranscript(current, inflight);
    expect(mergedOmitted.filter((message) => message.role === "user")).toHaveLength(1);
    expect(mergedOmitted.filter((message) => message.role === "assistant")).toHaveLength(1);
    expect(mergedOmitted.find((message) => message.role === "assistant")?.text).toBe("partial answer");

    const mergedNull = mergeInflightTranscript(current, inflight, null);
    expect(mergedNull.filter((message) => message.role === "user")).toHaveLength(1);
    expect(mergedNull.filter((message) => message.role === "assistant")).toHaveLength(1);
    expect(mergedNull.find((message) => message.role === "assistant")?.text).toBe("partial answer");
  });

  it("does not merge an identical untagged completion without an active-user anchor", () => {
    const current: ReconcileMessage[] = [
      user("u1", "prompt"),
      assistant("a1", "same answer"),
    ];

    expect(mergeCompletedAssistantMessage(current, "live-a2", "same answer")).toEqual([
      ...current,
      { id: "live-a2", role: "assistant", text: "same answer" },
    ]);
  });

  it("does not let an empty assistant prefix consume a completed answer", () => {
    const current: ReconcileMessage[] = [
      user("u1", "prompt"),
      assistant("a1", "completed answer"),
    ];

    expect(mergeLiveTimelineTranscript(current, [assistant("live-a2", "", true)])).toEqual([
      ...current,
      assistant("live-a2", "", true),
    ]);
  });

  it("does not let an unmatched user consume a later durable assistant", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "old-user", rowId: 1, role: "user", text: "old prompt" },
      { id: "old-assistant", rowId: 2, role: "assistant", text: "same answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user", "new prompt"),
      assistant("local-assistant", "same answer", true),
    ];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual([
      ...snapshot,
      ...current,
    ]);
  });

  it("preserves current order when reconnect provides an overlapping tail snapshot", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "2", rowId: 2, role: "user", text: "second prompt" },
      { id: "3", rowId: 3, role: "assistant", text: "same answer" },
      { id: "4", rowId: 4, role: "user", text: "third prompt" },
      { id: "5", rowId: 5, role: "assistant", text: "same answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "first prompt"),
      assistant("local-assistant-1", "first answer"),
      user("local-user-2", "second prompt"),
      assistant("local-assistant-2", "same answer"),
      user("local-user-3", "third prompt"),
      assistant("local-assistant-3", "same answer"),
      user("local-user-4", "new prompt"),
      assistant("local-assistant-4", "new answer", true),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      current[0],
      current[1],
      snapshot[0],
      snapshot[1],
      snapshot[2],
      snapshot[3],
      current[6],
      current[7],
    ]);
  });

  it("inserts missing tail-snapshot rows before a later matched anchor", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "2", rowId: 2, role: "user", text: "second prompt" },
      { id: "3", rowId: 3, role: "assistant", text: "second answer" },
      { id: "4", rowId: 4, role: "user", text: "third prompt" },
      { id: "5", rowId: 5, role: "assistant", text: "third answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "first prompt"),
      assistant("local-assistant-1", "first answer"),
      user("local-user-3", "third prompt"),
      assistant("local-assistant-3", "third answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      current[0],
      current[1],
      snapshot[0],
      snapshot[1],
      snapshot[2],
      snapshot[3],
    ]);
  });

  it("inserts missing tail-snapshot rows before newer current rows after the last anchor", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "2", rowId: 2, role: "user", text: "second prompt" },
      { id: "3", rowId: 3, role: "assistant", text: "second answer" },
      { id: "4", rowId: 4, role: "user", text: "third prompt" },
      { id: "5", rowId: 5, role: "assistant", text: "third answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "first prompt"),
      assistant("local-assistant-1", "first answer"),
      user("local-user-2", "second prompt"),
      assistant("local-assistant-2", "second answer"),
      user("local-user-4", "fourth prompt"),
      assistant("local-assistant-4", "fourth answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      current[0],
      current[1],
      snapshot[0],
      snapshot[1],
      snapshot[2],
      snapshot[3],
      current[4],
      current[5],
    ]);
  });

  it("does not match an assistant suffix across an unmatched snapshot user", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "old-user", rowId: 1, role: "user", text: "old prompt" },
      { id: "old-assistant", rowId: 2, role: "assistant", text: "same answer" },
    ];
    const current: ReconcileMessage[] = [
      user("new-user", "new prompt"),
      assistant("new-assistant", "same answer with more detail"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      ...current,
      ...snapshot,
    ]);
  });

  it("matches a tail assistant anchor when its missing internal user is absent locally", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "u2", rowId: 2, role: "user", text: "second" },
      { id: "a2", rowId: 3, role: "assistant", text: "second answer" },
      { id: "u3", rowId: 4, role: "user", text: "third" },
      { id: "a3", rowId: 5, role: "assistant", text: "third answer" },
    ];
    const current: ReconcileMessage[] = [
      user("u1", "first"),
      assistant("a1", "first answer"),
      assistant("local-a2", "second answer"),
      user("local-u3", "third"),
      assistant("local-a3", "third answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      current[0],
      current[1],
      snapshot[0],
      snapshot[1],
      snapshot[2],
      snapshot[3],
    ]);
  });

  it("keeps an empty inflight answer separate after a repeated completed turn", () => {
    const current: ReconcileMessage[] = [
      user("u1", "repeat prompt"),
      assistant("a1", "same answer"),
    ];
    const inflight: ReconcileMessage[] = [
      user("inflight-u", "repeat prompt"),
      assistant("inflight-a", "", true),
    ];

    expect(mergeInflightTranscript(current, inflight)).toEqual([
      ...current,
      ...inflight,
    ]);
  });

  it("keeps a repeated inflight user-only prompt separate from an earlier user-only turn", () => {
    const current: ReconcileMessage[] = [
      user("previous-user", "repeat prompt"),
    ];
    const inflight: ReconcileMessage[] = [
      user("inflight-user", "repeat prompt"),
    ];

    expect(mergeInflightTranscript(current, inflight)).toEqual([
      ...current,
      ...inflight,
    ]);
  });

  it("does not merge assistant-only identical tail turns without identity", () => {
    const snapshot = [assistant("snapshot-a", "same answer")];
    const current = [assistant("live-a", "same answer")];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      ...current,
      ...snapshot,
    ]);
  });

  it("aligns a tail snapshot to the latest repeated prompt occurrence", () => {
    const snapshot: ReconcileMessage[] = [
      { id: "3", rowId: 3, role: "user", text: "repeat prompt" },
      { id: "4", rowId: 4, role: "assistant", text: "repeat answer" },
    ];
    const current: ReconcileMessage[] = [
      user("local-user-1", "repeat prompt"),
      assistant("local-assistant-1", "first answer"),
      user("local-user-2", "repeat prompt"),
      assistant("local-assistant-2", "repeat answer"),
    ];

    expect(mergeSnapshotTranscript(snapshot, current, "tail")).toEqual([
      current[0],
      current[1],
      snapshot[0],
      snapshot[1],
    ]);
  });

  it("distinguishes an omitted field from an explicitly empty field", () => {
    expect(snapshotHasField({ pending_approval: null }, "pending_approval")).toBe(true);
    expect(snapshotHasField({}, "pending_approval")).toBe(false);
  });

  it("rejects snapshots for a different active runtime session", () => {
    expect(snapshotMatchesSession({ session_id: "runtime-2" }, "runtime-1")).toBe(false);
    expect(snapshotMatchesSession({ session_id: "runtime-1" }, "runtime-1")).toBe(true);
    expect(snapshotMatchesSession({}, "runtime-1")).toBe(true);
  });
});
