import { describe, expect, it, vi } from "vitest";

import {
  activeResumeTranscript,
  loadResumeTranscript,
  resumeTranscriptPhase,
  type ResumeTranscriptState,
} from "@/lib/resume-transcript";

const msg = (id: string) =>
  ({
    id,
    role: "user",
    content: `message ${id}`,
    created_at: 0,
  }) as never; // SessionMessage shape used only for pass-through

describe("loadResumeTranscript", () => {
  it("hydrates the transcript from the profile-scoped messages endpoint", async () => {
    const loadMessages = vi.fn(async () => ({
      messages: [msg("m1"), msg("m2")],
    }));

    const state = await loadResumeTranscript(loadMessages, "session/one", "worker");

    expect(loadMessages).toHaveBeenCalledWith("session/one", "worker");
    expect(state).toEqual({
      sessionId: "session/one",
      profile: "worker",
      messages: [msg("m1"), msg("m2")],
      error: null,
    });
  });

  it("returns an error state instead of throwing when the fetch fails", async () => {
    const loadMessages = vi.fn(async () => {
      throw new Error("Session not found");
    });

    const state = await loadResumeTranscript(loadMessages, "sid", "");

    expect(state.messages).toBeNull();
    expect(state.error).toBe("Session not found");
    expect(state.sessionId).toBe("sid");
    expect(state.profile).toBe("");
  });

  it("falls back to a generic message for non-Error rejections", async () => {
    const loadMessages = vi.fn(async () => {
      throw "boom";
    });

    const state = await loadResumeTranscript(loadMessages, "sid", "");

    expect(state.error).toBe("failed to load messages");
  });
});

describe("activeResumeTranscript", () => {
  const ready: ResumeTranscriptState = {
    sessionId: "sid",
    profile: "worker",
    messages: [msg("m1")],
    error: null,
  };

  it("returns the state when session id and profile match the target", () => {
    expect(activeResumeTranscript(ready, "sid", "worker")).toBe(ready);
  });

  it("returns null for a stale session id (session switched)", () => {
    expect(activeResumeTranscript(ready, "other", "worker")).toBeNull();
  });

  it("returns null for a stale profile (management profile switched)", () => {
    expect(activeResumeTranscript(ready, "sid", "default")).toBeNull();
  });

  it("returns null when no resume target is set", () => {
    expect(activeResumeTranscript(ready, null, "worker")).toBeNull();
    expect(activeResumeTranscript(null, null, "worker")).toBeNull();
  });
});

describe("resumeTranscriptPhase", () => {
  it("is idle without a resume target", () => {
    expect(resumeTranscriptPhase(null, null, "worker")).toBe("idle");
  });

  it("is loading while the target has no matching transcript yet", () => {
    expect(resumeTranscriptPhase(null, "sid", "worker")).toBe("loading");
    // stale transcript from a previous session must not count as loaded
    const stale: ResumeTranscriptState = {
      sessionId: "old",
      profile: "worker",
      messages: [],
      error: null,
    };
    expect(resumeTranscriptPhase(stale, "sid", "worker")).toBe("loading");
  });

  it("is ready when hydrated messages exist for the target", () => {
    const ready: ResumeTranscriptState = {
      sessionId: "sid",
      profile: "worker",
      messages: [msg("m1")],
      error: null,
    };
    expect(resumeTranscriptPhase(ready, "sid", "worker")).toBe("ready");
  });

  it("is empty when the target hydrated with zero messages", () => {
    const empty: ResumeTranscriptState = {
      sessionId: "sid",
      profile: "worker",
      messages: [],
      error: null,
    };
    expect(resumeTranscriptPhase(empty, "sid", "worker")).toBe("empty");
  });

  it("is error when the target transcript failed to load", () => {
    const errored: ResumeTranscriptState = {
      sessionId: "sid",
      profile: "worker",
      messages: null,
      error: "Session not found",
    };
    expect(resumeTranscriptPhase(errored, "sid", "worker")).toBe("error");
  });
});
