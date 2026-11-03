import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CONSOLE_TICKET_TIMEOUT_MS, raceTicket } from "./console-connect";

describe("raceTicket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns the ticket when it arrives in time", async () => {
    const promise = raceTicket(async () => "ws://localhost/api/console");
    await vi.advanceTimersByTimeAsync(0);

    await expect(promise).resolves.toEqual({
      status: "ok",
      value: "ws://localhost/api/console",
    });
  });

  it("reports a rejection rather than waiting out the deadline", async () => {
    const boom = new Error("ticket endpoint unavailable");
    const promise = raceTicket(() => Promise.reject(boom));
    await vi.advanceTimersByTimeAsync(0);

    await expect(promise).resolves.toEqual({ error: boom, status: "failed" });
  });

  it("times out a stalled request", async () => {
    const promise = raceTicket(() => new Promise<string>(() => {}));

    await vi.advanceTimersByTimeAsync(CONSOLE_TICKET_TIMEOUT_MS);

    await expect(promise).resolves.toEqual({ status: "timeout" });
  });

  it("keeps the timeout verdict when the ticket lands late", async () => {
    let land!: (url: string) => void;
    const promise = raceTicket(
      () =>
        new Promise<string>((resolve) => {
          land = resolve;
        }),
    );

    await vi.advanceTimersByTimeAsync(CONSOLE_TICKET_TIMEOUT_MS);
    land("ws://localhost/api/console?ticket=stale");
    await vi.advanceTimersByTimeAsync(0);

    // A late ticket must not become a socket behind the reported failure.
    await expect(promise).resolves.toEqual({ status: "timeout" });
  });

  it("does not leave its timer armed after settling", async () => {
    const promise = raceTicket(async () => "ws://localhost/api/console");
    await vi.advanceTimersByTimeAsync(0);
    await promise;

    expect(vi.getTimerCount()).toBe(0);
  });
});
