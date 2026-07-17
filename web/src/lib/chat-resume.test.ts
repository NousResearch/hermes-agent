import { describe, expect, it } from "vitest";

import {
  buildChatAttachScope,
  buildResumeInChatUrl,
  RESUME_NONCE_PARAM,
} from "./chat-resume";

function parseResumeUrl(url: string) {
  const parsed = new URL(url, "https://example.test");
  return {
    pathname: parsed.pathname,
    resume: parsed.searchParams.get("resume"),
    resumeNonce: parsed.searchParams.get(RESUME_NONCE_PARAM),
  };
}

describe("buildResumeInChatUrl", () => {
  it("keeps the selected session id and stamps a reopen nonce", () => {
    const parsed = parseResumeUrl(buildResumeInChatUrl("sess-123"));

    expect(parsed.pathname).toBe("/chat");
    expect(parsed.resume).toBe("sess-123");
    expect(parsed.resumeNonce).toMatch(/^[a-z0-9]+-[a-z0-9]+$/);
  });

  it("generates a fresh nonce for repeated reopen clicks", () => {
    const first = parseResumeUrl(buildResumeInChatUrl("sess-123"));
    const second = parseResumeUrl(buildResumeInChatUrl("sess-123"));

    expect(first.resume).toBe("sess-123");
    expect(second.resume).toBe("sess-123");
    expect(first.resumeNonce).not.toBe(second.resumeNonce);
  });
});

describe("buildChatAttachScope", () => {
  it("scopes keep-alive attachment tokens to the selected session", () => {
    expect(buildChatAttachScope("sess-123", null)).toBe("default::sess-123");
    expect(buildChatAttachScope("sess-456", null)).toBe("default::sess-456");
  });

  it("keeps fresh chat and profile-scoped chat separate", () => {
    expect(buildChatAttachScope(null, null)).toBe("default::fresh");
    expect(buildChatAttachScope("sess-123", "ops")).toBe("ops::sess-123");
  });
});
