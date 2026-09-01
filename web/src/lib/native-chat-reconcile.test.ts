import { describe, expect, it } from "vitest";

import {
  mergeSnapshotTranscript,
  snapshotHasField,
  snapshotMatchesSession,
  type ReconcileMessage,
} from "./native-chat-reconcile";

const user = (id: string, text: string): ReconcileMessage => ({ id, role: "user", text });
const assistant = (id: string, text: string, streaming = false): ReconcileMessage => ({ id, role: "assistant", text, streaming });

describe("native chat snapshot reconciliation", () => {
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

  it("does not duplicate equivalent assistant messages without stable ids", () => {
    const snapshot = [assistant("snapshot-a", "same answer")];
    const current = [assistant("live-a", "same answer")];

    expect(mergeSnapshotTranscript(snapshot, current)).toEqual(snapshot);
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
