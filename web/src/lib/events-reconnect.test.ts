import { describe, expect, it } from "vitest";

import {
  EVENTS_MAX_RECONNECT_ATTEMPTS,
  EVENTS_RECONNECT_BASE_MS,
  EVENTS_RECONNECT_MAX_MS,
  eventsGaveUpMessage,
  eventsReconnectDelayMs,
  eventsReconnectingMessage,
  eventsRejectedMessage,
  isEventsAuthRejection,
  isEventsFeedMessage,
  shouldRetryEventsClose,
  EVENTS_DISCONNECTED_MESSAGE,
} from "./events-reconnect";

describe("eventsReconnectDelayMs", () => {
  it("doubles from the base delay", () => {
    expect(eventsReconnectDelayMs(0)).toBe(EVENTS_RECONNECT_BASE_MS);
    expect(eventsReconnectDelayMs(1)).toBe(EVENTS_RECONNECT_BASE_MS * 2);
    expect(eventsReconnectDelayMs(2)).toBe(EVENTS_RECONNECT_BASE_MS * 4);
    expect(eventsReconnectDelayMs(3)).toBe(EVENTS_RECONNECT_BASE_MS * 8);
  });

  it("clamps at the cap and never exceeds it", () => {
    for (
      let attempt = 0;
      attempt <= EVENTS_MAX_RECONNECT_ATTEMPTS + 5;
      attempt++
    ) {
      expect(eventsReconnectDelayMs(attempt)).toBeLessThanOrEqual(
        EVENTS_RECONNECT_MAX_MS,
      );
    }
    expect(eventsReconnectDelayMs(99)).toBe(EVENTS_RECONNECT_MAX_MS);
  });

  it("is monotonically non-decreasing", () => {
    let previous = 0;
    for (let attempt = 0; attempt < 20; attempt++) {
      const delay = eventsReconnectDelayMs(attempt);
      expect(delay).toBeGreaterThanOrEqual(previous);
      previous = delay;
    }
  });

  it("stays finite for absurd attempt counts", () => {
    expect(Number.isFinite(eventsReconnectDelayMs(10_000))).toBe(true);
  });
});

describe("shouldRetryEventsClose", () => {
  it("retries transient drops", () => {
    // 1005 (no status) and 1006 (abnormal) are what a killed gateway and a
    // dropped network produce respectively.
    for (const code of [1001, 1005, 1006, 1011, 1012, 1013]) {
      expect(shouldRetryEventsClose(code)).toBe(true);
    }
  });

  it("does not retry a normal closure", () => {
    expect(shouldRetryEventsClose(1000)).toBe(false);
  });

  it("does not retry auth rejections", () => {
    expect(shouldRetryEventsClose(4401)).toBe(false);
    expect(shouldRetryEventsClose(4403)).toBe(false);
  });

  it("treats the whole 4400-4499 application range as terminal (#88607)", () => {
    // A bad-channel/keyed rejection (4400) and a disabled feed (4404) never
    // get better by reconnecting — and each retry burns a ticket.
    expect(shouldRetryEventsClose(4400)).toBe(false);
    expect(shouldRetryEventsClose(4404)).toBe(false);
    expect(shouldRetryEventsClose(4449)).toBe(false);
    expect(isEventsAuthRejection(4400)).toBe(true);
    expect(isEventsAuthRejection(4404)).toBe(true);
    // 4500 is outside the application range: still retryable.
    expect(shouldRetryEventsClose(4500)).toBe(true);
  });

  it("retries when the code is missing", () => {
    expect(shouldRetryEventsClose(undefined)).toBe(true);
  });

  it("never both retries and reports an auth rejection", () => {
    for (const code of [1000, 1005, 1006, 4401, 4403, 4500]) {
      expect(shouldRetryEventsClose(code) && isEventsAuthRejection(code)).toBe(
        false,
      );
    }
  });
});

describe("isEventsFeedMessage", () => {
  it("recognizes every message this module can surface", () => {
    expect(isEventsFeedMessage(EVENTS_DISCONNECTED_MESSAGE)).toBe(true);
    expect(isEventsFeedMessage(eventsReconnectingMessage(4_000))).toBe(true);
    expect(isEventsFeedMessage(eventsRejectedMessage(4401))).toBe(true);
    expect(isEventsFeedMessage(eventsGaveUpMessage())).toBe(true);
  });

  it("does not claim messages owned by other banner sources", () => {
    // The banner is shared with info.credential_warning and the JSON-RPC
    // sidecar — clearing those on an events reconnect would hide a real
    // problem from the user.
    expect(isEventsFeedMessage("ANTHROPIC_API_KEY is not set")).toBe(false);
    expect(isEventsFeedMessage("WebSocket connection failed")).toBe(false);
    expect(isEventsFeedMessage("gateway not connected")).toBe(false);
    expect(isEventsFeedMessage(null)).toBe(false);
    expect(isEventsFeedMessage("")).toBe(false);
  });
});

describe("reconnect message copy", () => {
  it("renders the delay in whole seconds", () => {
    expect(eventsReconnectingMessage(1_000)).toContain("1s");
    expect(eventsReconnectingMessage(30_000)).toContain("30s");
  });

  it("names the close code in the rejection message", () => {
    expect(eventsRejectedMessage(4403)).toContain("4403");
  });

  it("does not reference the tools box removed in #51737", () => {
    const messages = [
      EVENTS_DISCONNECTED_MESSAGE,
      eventsReconnectingMessage(1_000),
      eventsRejectedMessage(4401),
      eventsGaveUpMessage(),
    ];
    for (const message of messages) {
      expect(message).not.toContain("tool calls");
    }
  });
});
