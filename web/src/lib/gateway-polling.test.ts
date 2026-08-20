import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { pollGatewayStatus } from "./gateway-polling";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("pollGatewayStatus", () => {
  it("resolves true when the desired state is reached immediately", async () => {
    const getStatus = vi.fn().mockResolvedValue(true);

    const promise = pollGatewayStatus(getStatus, {
      name: "test",
      desiredState: true,
      timeoutMs: 10_000,
      intervalMs: 500,
    });

    // Fast-forward past the initial delay
    await vi.advanceTimersByTimeAsync(500);

    await expect(promise).resolves.toBe(true);
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it("polls until the desired state is reached", async () => {
    let callCount = 0;
    const getStatus = vi.fn().mockImplementation(async () => {
      callCount++;
      return callCount >= 3;
    });

    const promise = pollGatewayStatus(getStatus, {
      name: "test",
      desiredState: true,
      timeoutMs: 10_000,
      intervalMs: 50,
    });

    // Advance past first poll (50ms) -> false, second (100ms) -> false, third (150ms) -> true
    await vi.advanceTimersByTimeAsync(200);

    await expect(promise).resolves.toBe(true);
    expect(getStatus).toHaveBeenCalledTimes(3);
  });

  it("resolves false on timeout", async () => {
    const getStatus = vi.fn().mockResolvedValue(false);

    const promise = pollGatewayStatus(getStatus, {
      name: "test",
      desiredState: true,
      timeoutMs: 100,
      intervalMs: 50,
    });

    // Advance past timeout
    await vi.advanceTimersByTimeAsync(200);

    await expect(promise).resolves.toBe(false);
  });

  it("resolves false when aborted via signal", async () => {
    const ac = new AbortController();
    const getStatus = vi.fn().mockImplementation(async () => {
      // Abort on the first poll
      ac.abort();
      return false;
    });

    const promise = pollGatewayStatus(
      getStatus,
      {
        name: "test",
        desiredState: true,
        timeoutMs: 10_000,
        intervalMs: 500,
      },
      ac.signal,
    );

    await vi.advanceTimersByTimeAsync(500);

    await expect(promise).resolves.toBe(false);
  });

  it("continues polling after a transient fetch error", async () => {
    let callCount = 0;
    const getStatus = vi.fn().mockImplementation(async () => {
      callCount++;
      if (callCount === 1) throw new Error("network error");
      return true;
    });

    const promise = pollGatewayStatus(getStatus, {
      name: "test",
      desiredState: true,
      timeoutMs: 10_000,
      intervalMs: 50,
    });

    // First poll (50ms) -> throws, second (100ms) -> success
    await vi.advanceTimersByTimeAsync(150);

    await expect(promise).resolves.toBe(true);
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it("stops polling when desired state changes to false (stop gateway)", async () => {
    let callCount = 0;
    const getStatus = vi.fn().mockImplementation(async () => {
      callCount++;
      return callCount < 3;
    });

    const promise = pollGatewayStatus(getStatus, {
      name: "test",
      desiredState: false,
      timeoutMs: 10_000,
      intervalMs: 50,
    });

    // First poll (50ms) -> true (not stopped yet), second (100ms) -> true, third (150ms) -> false
    await vi.advanceTimersByTimeAsync(200);

    await expect(promise).resolves.toBe(true);
    // Call 3 returned false === desiredState (false) so we stop
    expect(getStatus).toHaveBeenCalledTimes(3);
  });
});
