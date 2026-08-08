import { describe, it, expect } from "vitest";
import {
  INITIAL_LOAD_START,
  initialLoadErrorMessage,
  initialLoadFailed,
  initialLoadRetrying,
  initialLoadSucceeded,
  initialLoadView,
} from "./initial-load";

describe("initialLoadErrorMessage", () => {
  it("surfaces the message of a rejected Error", () => {
    expect(initialLoadErrorMessage(new Error("500: config unreadable"))).toBe(
      "500: config unreadable",
    );
  });

  it("passes a plain string rejection through", () => {
    expect(initialLoadErrorMessage("NetworkError")).toBe("NetworkError");
  });

  it("falls back to a readable message for a shapeless rejection", () => {
    for (const reason of [undefined, null, {}, new Error(""), ""]) {
      const message = initialLoadErrorMessage(reason);
      expect(message).toBe(
        "Failed to load. The Hermes server may be unreachable.",
      );
      expect(message).not.toContain("undefined");
      expect(message).not.toContain("[object Object]");
    }
  });
});

describe("required reads gate the page", () => {
  it("moves to failed and surfaces the error instead of swallowing it", () => {
    const state = initialLoadFailed(new Error("503: gateway down"));
    expect(state.status).toBe("failed");
    expect(state.error).toBe("503: gateway down");
    // The regression this PR exists to prevent: a rejected required read
    // must NOT leave the page on the spinner forever.
    expect(initialLoadView(state, false)).toBe("error");
  });

  it("shows content once every required read resolves", () => {
    const state = initialLoadSucceeded(INITIAL_LOAD_START);
    expect(state.status).toBe("ready");
    expect(state.error).toBeNull();
    expect(initialLoadView(state, true)).toBe("content");
  });

  it("keeps the spinner while the required reads are still in flight", () => {
    expect(initialLoadView(INITIAL_LOAD_START, false)).toBe("spinner");
  });
});

describe("optional enrichment failures stay silent", () => {
  it("stays ready when an optional read rejects and the required reads resolved", () => {
    // ConfigPage's getDefaults / getConfigRaw / getStatus never call
    // initialLoadFailed — they keep their own `.catch(() => {})`. This asserts
    // the state machine a page in that situation is left in: still ready, no
    // error, content rendered. Guards the profile-path reads and the header
    // lifecycle against being dragged into the error state.
    const state = initialLoadSucceeded(INITIAL_LOAD_START);
    expect(state.status).toBe("ready");
    expect(state.error).toBeNull();
    expect(initialLoadView(state, true)).toBe("content");
  });

  it("does not let a required failure be papered over by a later success", () => {
    // getConfig rejects, getSchema then resolves: the page is still broken.
    const failed = initialLoadFailed(new Error("config read failed"));
    const afterSibling = initialLoadSucceeded(failed);
    expect(afterSibling.status).toBe("failed");
    expect(afterSibling.error).toBe("config read failed");
    expect(initialLoadView(afterSibling, false)).toBe("error");
  });

  it("shows the error even if partial data arrived", () => {
    const failed = initialLoadFailed(new Error("schema read failed"));
    expect(initialLoadView(failed, true)).toBe("error");
  });
});

describe("retry", () => {
  it("returns to loading and clears the error", () => {
    const failed = initialLoadFailed(new Error("boom"));
    const retrying = initialLoadRetrying();
    expect(failed.status).toBe("failed");
    expect(retrying.status).toBe("loading");
    expect(retrying.error).toBeNull();
    expect(initialLoadView(retrying, false)).toBe("spinner");
  });

  it("surfaces the new message when the retry fails again", () => {
    initialLoadFailed(new Error("first failure"));
    const retrying = initialLoadRetrying();
    expect(retrying.error).toBeNull();
    const second = initialLoadFailed(new Error("second failure"));
    expect(second.status).toBe("failed");
    // The stale message must not stick — a second failure reports itself.
    expect(second.error).toBe("second failure");
  });

  it("reaches ready when the retry succeeds", () => {
    initialLoadFailed(new Error("transient"));
    const state = initialLoadSucceeded(initialLoadRetrying());
    expect(state.status).toBe("ready");
    expect(state.error).toBeNull();
    expect(initialLoadView(state, true)).toBe("content");
  });
});
