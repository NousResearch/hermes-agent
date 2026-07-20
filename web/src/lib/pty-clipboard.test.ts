import { describe, expect, it } from "vitest";

import {
  DASHBOARD_COPY_LAST_SEQUENCE,
  MAX_OSC52_CLIPBOARD_BYTES,
  buildNativeDraftSubmissionPayload,
  consumeNativeSubmitAck,
  consumePendingOsc52Write,
  sendNativeDraftSubmission,
} from "./pty-clipboard";

describe("DASHBOARD_COPY_LAST_SEQUENCE", () => {
  it("uses a control sequence instead of injecting slash-command text", () => {
    expect(DASHBOARD_COPY_LAST_SEQUENCE).toBe("\x1b[99;13u");
    expect(DASHBOARD_COPY_LAST_SEQUENCE).not.toContain("/copy");
    expect(DASHBOARD_COPY_LAST_SEQUENCE).not.toContain("\r");
  });
});

describe("buildNativeDraftSubmissionPayload", () => {
  it("encodes one UTF-8 dashboard submission in an APC frame", () => {
    expect(buildNativeDraftSubmissionPayload("한글\nmessage", "request-1")).toMatch(
      /^\x1b_HERMES_SUBMIT;request-1;[A-Za-z0-9+/]+=*\x1b\\$/,
    );
  });

  it("does not encode a synthetic clear, paste, or Return key", () => {
    const payload = buildNativeDraftSubmissionPayload("draft", "request-1");
    expect(payload).not.toContain("\x15");
    expect(payload).not.toContain("\x1b[200~");
    expect(payload).not.toContain("\r");
  });
});

describe("sendNativeDraftSubmission", () => {
  it("returns success only after the atomic frame is sent", () => {
    const send = (payload: string) => {
      expect(payload).toBe("\x1b_HERMES_SUBMIT;request-1;ZHJhZnQ=\x1b\\");
    };
    expect(sendNativeDraftSubmission("draft", "request-1", send)).toBe(true);
  });

  it("reports send failure so the caller retains nativeDraft", () => {
    expect(
      sendNativeDraftSubmission("keep me", "request-1", () => {
        throw new Error("socket closed");
      }),
    ).toBe(false);
  });
});

describe("consumeNativeSubmitAck", () => {
  it("accepts only the matching pending request", () => {
    expect(consumeNativeSubmitAck("HERMES_SUBMIT_ACK;request-1", "request-1")).toBe(true);
    expect(consumeNativeSubmitAck("HERMES_SUBMIT_ACK;request-2", "request-1")).toBe(false);
    expect(consumeNativeSubmitAck("HERMES_SUBMIT_ACK;request-1", null)).toBe(false);
  });
});

describe("consumePendingOsc52Write", () => {
  const encoded = (text: string, target = "c") =>
    `${target};${btoa(unescape(encodeURIComponent(text)))}`;

  it("authorizes exactly one valid clipboard-target write while pending", () => {
    expect(consumePendingOsc52Write(encoded("latest answer"), 1_000, 1_100)).toEqual({
      text: "latest answer",
      pendingAt: null,
    });
    expect(consumePendingOsc52Write(encoded("latest answer"), null, 1_100)).toEqual({
      text: null,
      pendingAt: null,
    });
  });

  it("rejects invalid targets, expired requests, and oversized payloads", () => {
    expect(consumePendingOsc52Write(encoded("x", "p"), 1_000, 1_100).text).toBeNull();
    expect(consumePendingOsc52Write(encoded("x"), 1_000, 10_000).text).toBeNull();
    const oversized = "a".repeat(MAX_OSC52_CLIPBOARD_BYTES + 1);
    expect(consumePendingOsc52Write(encoded(oversized), 1_000, 1_100).text).toBeNull();
  });

  it("clears pending authorization even when validation fails", () => {
    expect(consumePendingOsc52Write("c;not base64!", 1_000, 1_100)).toEqual({
      text: null,
      pendingAt: null,
    });
  });
});
