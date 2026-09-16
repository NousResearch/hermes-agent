/**
 * Tests for the sidecar ``session.create`` params built by ChatSidebar.
 *
 * The sidecar must opt its session into close_on_disconnect so the gateway
 * reaps the slash_worker on WS disconnect (the #21370/#21467 leak), and it
 * must pass the dashboard's selected profile so model/credential info
 * matches the PTY child under profile-scoped chat.
 *
 */

import { describe, expect, it } from "vitest";

import { sidecarSessionCreateParams } from "@/components/ChatSidebar";

describe("sidecarSessionCreateParams", () => {
  it("opts into close_on_disconnect", () => {
    const params = sidecarSessionCreateParams();
    expect(params.close_on_disconnect).toBe(true);
  });

  it("sets source to 'tool'", () => {
    const params = sidecarSessionCreateParams();
    expect(params.source).toBe("tool");
  });

  it("forwards the profile when present", () => {
    const params = sidecarSessionCreateParams("work");
    expect(params.profile).toBe("work");
  });

  it("omits profile when undefined", () => {
    const params = sidecarSessionCreateParams();
    expect(params).not.toHaveProperty("profile");
  });
});
