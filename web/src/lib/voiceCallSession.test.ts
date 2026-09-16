/**
 * The dashboard voice card must ask the gateway for a real session —
 * invented session ids 404 at prompt.submit — and that session must stay
 * visible: the gateway hides `source: "tool"` / `"kanban"` rows from every
 * human listing, which would strand the consult answers. (Card behavior lives
 * in components/VoiceCallCard.test.tsx; the consult/steer machine in
 * apps/shared voice-supervisor tests.)
 */
import { describe, expect, it } from "vitest";

import { voiceCallSessionCreateParams } from "@/lib/voiceCallSession";

// Mirrors tui_gateway/methods_session.py `_LISTING_DENY_SOURCES`.
const HIDDEN_SESSION_SOURCES = ["tool", "kanban"];

describe("voiceCallSessionCreateParams", () => {
  it("requests a reaped session and forwards profile", () => {
    expect(voiceCallSessionCreateParams()).toEqual({
      close_on_disconnect: true,
    });
    expect(voiceCallSessionCreateParams("coder")).toEqual({
      close_on_disconnect: true,
      profile: "coder",
    });
  });

  it("never files the call under a source hidden from the session list", () => {
    const source = voiceCallSessionCreateParams("coder").source;
    expect(source === undefined || !HIDDEN_SESSION_SOURCES.includes(String(source))).toBe(true);
  });
});
