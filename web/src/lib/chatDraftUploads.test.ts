import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DRAFT_CONTROL_PREFIX, type DraftAttachRequest, type DraftIdentity } from "@hermes/shared";
import { ChatDraftUploads } from "./chatDraftUploads";

const { authedFetch } = vi.hoisted(() => ({ authedFetch: vi.fn() }));
vi.mock("./api", () => ({ authedFetch }));
const identity: DraftIdentity = { pty_instance: "pty-1", connection_generation: 3, session_id: "session-1", draft_id: "draft-1" };
const context = { profile: "work & notes", scope: "chat-one", active: true };
const file = new File([new Uint8Array([0, 255, 13, 10])], "data.bin", { type: "application/octet-stream" });
function socket() { return { readyState: WebSocket.OPEN, send: vi.fn() } as unknown as WebSocket; }
function response(path = "/scoped/attachments/data.bin") { return { ok: true, json: async () => ({ path, name: file.name, bytes: file.size, mime_type: file.type }) }; }
function ready(controller: ChatDraftUploads, ws: WebSocket, current = identity, available = true) {
  controller.receive(ws, JSON.stringify({ type: "draft.state", identity: current, available }));
}
function requests(ws: WebSocket): DraftAttachRequest[] {
  return vi.mocked(ws.send).mock.calls.map(([frame]) => {
    expect(typeof frame).toBe("string");
    expect((frame as string).startsWith(DRAFT_CONTROL_PREFIX)).toBe(true);
    return JSON.parse((frame as string).slice(DRAFT_CONTROL_PREFIX.length));
  });
}
function result(controller: ChatDraftUploads, ws: WebSocket, request: DraftAttachRequest, status = "attached", current = identity) {
  controller.receive(ws, JSON.stringify({ type: "draft.result", request_id: request.request_id, identity: current, status }));
}
async function flush() { await new Promise<void>(resolve => setImmediate(resolve)); }
let controller: ChatDraftUploads;
let ws: WebSocket;
beforeEach(() => {
  authedFetch.mockReset();
  authedFetch.mockResolvedValue(response());
  controller = new ChatDraftUploads();
  ws = socket();
  controller.setContext(context);
  controller.bind(ws);
  ready(controller, ws);
});
afterEach(() => { controller.dispose(); vi.useRealTimers(); });

describe("draft upload transaction", () => {
  it("does not assume randomUUID is available on a remote HTTP dashboard", async () => {
    vi.stubGlobal("crypto", { getRandomValues: (bytes: Uint8Array) => { bytes.fill(7); return bytes; } });
    try {
      expect(() => controller.select([file, file])).not.toThrow();
      await flush();
      const original = requests(ws)[0];
      result(controller, ws, original, "failed");
      controller.retry(original.request_id);
      expect(requests(ws)).toEqual([original, original]);
      expect(new Set(controller.getSnapshot().uploads.map(entry => entry.id)).size).toBe(2);
    } finally { vi.unstubAllGlobals(); }
  });

  it("retires all files in a rejected stale identity until the next authoritative handshake", async () => {
    controller.select([file, file]);
    await flush();
    result(controller, ws, requests(ws)[0], "stale");
    expect(controller.getSnapshot().uploads.map(entry => entry.state)).toEqual(["stale", "stale"]);
    expect(controller.getSnapshot().available).toBe(false);
    controller.select([file]);
    await flush();
    expect(requests(ws)).toHaveLength(1);
    ready(controller, ws, { ...identity, draft_id: "replacement" });
    controller.select([file]);
    await flush();
    expect(requests(ws).at(-1)!.expected.draft_id).toBe("replacement");
  });

  it("requires a valid available handshake and consumes malformed control without accepting it", () => {
    controller.bind(ws);
    const malformed = [
      DRAFT_CONTROL_PREFIX + "not JSON",
      '{"type":"draft.state",broken',
      JSON.stringify({ type: "draft.state", available: true, identity: { ...identity, connection_generation: "3" } }),
      JSON.stringify({ type: "draft.state", available: "true", identity }),
      JSON.stringify({ type: "draft.result", status: "attached" }),
      JSON.stringify({ type: "draft.unknown" }),
    ];
    for (const frame of malformed) {
      expect(controller.receive(ws, frame)).toBe(true);
      expect(controller.getSnapshot().available).toBe(false);
      controller.select([file]);
    }
    expect(authedFetch).not.toHaveBeenCalled();
    expect(controller.receive(ws, "ordinary ANSI text")).toBe(false);
    expect(controller.receive(ws, JSON.stringify({ type: "resume", session_id: "a" }))).toBe(false);
    ready(controller, ws, identity, false);
    controller.select([file]);
    expect(authedFetch).not.toHaveBeenCalled();
    ready(controller, ws);
    expect(controller.getSnapshot().available).toBe(true);
  });

  it.each(["stale", "unavailable"])("surfaces native %s without success or automatic retry", async status => {
    controller.select([file]);
    await flush();
    const request = requests(ws)[0];
    result(controller, ws, request, status);
    expect(controller.getSnapshot().uploads[0]).toMatchObject({ state: status === "stale" ? "stale" : "failed" });
    expect(requests(ws)).toEqual([request]);
    if (status === "unavailable") {
      ready(controller, ws, identity, false);
      expect(controller.getSnapshot().available).toBe(false);
      controller.retry(request.request_id);
      expect(requests(ws)).toEqual([request]);
      ready(controller, ws);
      expect(requests(ws)).toEqual([request]);
      controller.retry(request.request_id);
      expect(requests(ws)).toEqual([request, request]);
    } else {
      controller.retry(request.request_id);
      expect(requests(ws)).toEqual([request]);
    }
  });

  it.each(["dismiss", "dispose", "unavailable"])("cancels unfinished bytes on %s and ignores late completion", async change => {
    let complete!: (value: ReturnType<typeof response>) => void;
    authedFetch.mockImplementationOnce(() => new Promise(resolve => { complete = resolve; }));
    controller.select([file]);
    const id = controller.getSnapshot().uploads[0].id;
    const signal = authedFetch.mock.calls[0][1].signal as AbortSignal;
    if (change === "dismiss") controller.remove(id);
    if (change === "dispose") controller.dispose();
    if (change === "unavailable") ready(controller, ws, identity, false);
    expect(signal.aborted).toBe(true);
    complete(response());
    await flush();
    expect(requests(ws)).toEqual([]);
    expect(controller.getSnapshot().uploads.length).toBe(change === "unavailable" ? 1 : 0);
  });

  it("reports a lost ACK as unknown and retries only the original request, accepting a late exact ACK", async () => {
    vi.useFakeTimers();
    controller.select([file]);
    await vi.advanceTimersByTimeAsync(0);
    const original = requests(ws)[0];
    await vi.advanceTimersByTimeAsync(60_000);
    expect(controller.getSnapshot().uploads[0]).toMatchObject({ state: "unknown", error: "Attachment not confirmed." });
    controller.retry(original.request_id);
    expect(requests(ws)).toEqual([original, original]);
    expect(authedFetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(60_000);
    result(controller, ws, original);
    expect(controller.getSnapshot().uploads).toEqual([]);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(controller.getSnapshot().uploads).toEqual([]);
  });

  it.each(["upload", "socket"])("retries the failed %s action only and does not expose raw server errors", async failure => {
    if (failure === "upload") authedFetch.mockResolvedValueOnce({ ok: false, status: 413, text: async () => "/private/server/path very long internal error" });
    else vi.mocked(ws.send).mockImplementationOnce(() => { throw new Error("/private/socket/path"); });
    controller.select([file]);
    await flush();
    const failed = controller.getSnapshot().uploads[0];
    expect(failed.state).toBe("failed");
    expect(failed.error).not.toContain("/private/");
    const prior = requests(ws);
    controller.retry(failed.id);
    controller.retry(failed.id);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(failure === "upload" ? 2 : 1);
    if (failure === "socket") expect(requests(ws)).toEqual([prior[0], prior[0]]);
    expect(authedFetch.mock.calls.map(([url]) => url)).toEqual(Array(failure === "upload" ? 2 : 1).fill("/api/chat/file-upload?profile=work%20%26%20notes"));
    result(controller, ws, requests(ws).at(-1)!);
    expect(controller.getSnapshot().uploads).toEqual([]);
  });

  it.each(["profile", "scope", "socket", "pty", "generation", "session", "draft", "hidden", "closed socket"])("never retargets a pending upload after %s, even if the old state returns", async change => {
    let complete!: (value: ReturnType<typeof response>) => void;
    authedFetch.mockImplementationOnce(() => new Promise(resolve => { complete = resolve; }));
    controller.select([file]);
    const id = controller.getSnapshot().uploads[0].id;
    const replacement = socket();
    if (change === "profile") controller.setContext({ ...context, profile: "other" });
    if (change === "scope") controller.setContext({ ...context, scope: "other" });
    if (change === "socket") { controller.bind(replacement); ready(controller, replacement); }
    if (change === "hidden") controller.setContext({ ...context, active: false });
    if (change === "closed socket") Object.assign(ws, { readyState: 3 });
    const changes: Record<string, Partial<DraftIdentity>> = {
      pty: { pty_instance: "pty-2" }, generation: { connection_generation: 4 },
      session: { session_id: "session-2" }, draft: { draft_id: "draft-2" },
    };
    if (changes[change]) ready(controller, ws, { ...identity, ...changes[change] });
    complete(response());
    await flush();
    expect(requests(ws)).toEqual([]);
    expect(requests(replacement)).toEqual([]);
    expect(controller.getSnapshot().uploads[0].state).toBe("stale");
    controller.setContext(context);
    controller.bind(ws);
    Object.assign(ws, { readyState: WebSocket.OPEN });
    ready(controller, ws);
    controller.retry(id);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(1);
    expect(requests(ws)).toEqual([]);
  });

  it("ignores mismatched ACK identity and retires outstanding handoffs on draft change", async () => {
    controller.select([file]);
    await flush();
    const request = requests(ws)[0];
    const other = { ...identity, draft_id: "new-draft" };
    result(controller, ws, request, "attached", other);
    expect(controller.getSnapshot().uploads[0].state).toBe("attaching");
    ready(controller, ws, other);
    expect(controller.getSnapshot().uploads[0].state).toBe("stale");
    result(controller, ws, request);
    controller.retry(request.request_id);
    expect(controller.getSnapshot().uploads[0].state).toBe("stale");
    expect(requests(ws)).toEqual([request]);
  });

  it("retries native failure with the saved bytes and same request ID, without overtaking", async () => {
    controller.select([file, new File(["later"], "later.txt")]);
    await flush();
    const original = requests(ws)[0];
    result(controller, ws, original, "failed");
    expect(controller.getSnapshot().uploads[0].state).toBe("failed");
    controller.retry(original.request_id);
    controller.retry(original.request_id);
    await flush();
    expect(requests(ws)).toEqual([original, original]);
    expect(authedFetch).toHaveBeenCalledTimes(1);
    result(controller, ws, original);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(2);
    expect(controller.getSnapshot().uploads.map(entry => entry.name)).toEqual(["later.txt"]);
  });

  it("serializes whole file transactions through native ACK, including later selections", async () => {
    controller.select([file, new File(["two"], "two.txt")]);
    controller.select([new File(["three"], "three.pdf")]);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(1);
    expect(controller.getSnapshot().uploads.map(entry => entry.state)).toEqual(["attaching", "queued", "queued"]);
    const first = requests(ws)[0];
    result(controller, ws, first);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(2);
    result(controller, ws, requests(ws)[1]);
    await flush();
    expect(authedFetch).toHaveBeenCalledTimes(3);
    result(controller, ws, requests(ws)[2]);
    expect(controller.getSnapshot().uploads).toEqual([]);
    expect(new Set(requests(ws).map(request => request.request_id)).size).toBe(3);
  });
});
