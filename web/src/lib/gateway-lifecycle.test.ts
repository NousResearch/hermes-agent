import { afterEach, describe, expect, it, vi } from "vitest";

import {
  waitForActionCompletion,
  waitForGatewayState,
} from "./gateway-lifecycle";

const never = new Promise<never>(() => {});

afterEach(() => {
  vi.useRealTimers();
});

describe("waitForGatewayState", () => {
  it("keeps polling until the selected gateway reaches the expected state", async () => {
    const getStatus = vi
      .fn()
      .mockResolvedValueOnce({ gateway_running: false })
      .mockResolvedValueOnce({ gateway_running: true });
    const sleep = vi.fn().mockResolvedValue(undefined);

    const status = await waitForGatewayState(getStatus, true, {
      intervalMs: 1,
      timeoutMs: 1_000,
      sleep,
    });

    expect(status.gateway_running).toBe(true);
    expect(getStatus).toHaveBeenCalledTimes(2);
    expect(sleep).toHaveBeenCalledTimes(1);
  });

  it("times out when a status request never settles", async () => {
    vi.useFakeTimers();
    const result = waitForGatewayState(() => never, true, { timeoutMs: 10 });
    const rejection = expect(result).rejects.toThrow(
      "Timed out waiting for gateway state transition",
    );

    await vi.advanceTimersByTimeAsync(10);

    await rejection;
  });

  it("rejects a matching status that resolves after the deadline", async () => {
    vi.useFakeTimers();
    let resolveStatus!: (status: { gateway_running: boolean }) => void;
    const delayedStatus = new Promise<{ gateway_running: boolean }>((resolve) => {
      resolveStatus = resolve;
    });
    const result = waitForGatewayState(() => delayedStatus, true, {
      timeoutMs: 10,
    });
    const rejection = expect(result).rejects.toThrow(
      "Timed out waiting for gateway state transition",
    );

    await vi.advanceTimersByTimeAsync(11);
    resolveStatus({ gateway_running: true });
    await vi.runAllTimersAsync();

    await rejection;
  });
});

describe("waitForActionCompletion", () => {
  it("accepts a restart already complete before polling the running gateway", async () => {
    const getActionStatus = vi.fn().mockResolvedValue({
      running: false,
      exit_code: 0,
      pid: 41,
    });
    const getStatus = vi.fn().mockResolvedValue({ gateway_running: true });

    await expect(
      waitForActionCompletion(getActionStatus, 41, { timeoutMs: 1_000 }),
    ).resolves.toMatchObject({ running: false, exit_code: 0 });
    await expect(
      waitForGatewayState(getStatus, true, { timeoutMs: 1_000 }),
    ).resolves.toMatchObject({ gateway_running: true });
    expect(getActionStatus).toHaveBeenCalledTimes(1);
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it("rejects a completed action with an unsuccessful exit", async () => {
    const getActionStatus = vi.fn().mockResolvedValue({
      running: false,
      exit_code: 1,
      pid: 41,
    });

    await expect(waitForActionCompletion(getActionStatus, 41)).rejects.toThrow(
      "Gateway action failed with exit code 1",
    );
  });

  it("rejects a shared action slot replaced by a different process", async () => {
    const getActionStatus = vi.fn().mockResolvedValue({
      running: false,
      exit_code: 0,
      pid: 99,
    });

    await expect(waitForActionCompletion(getActionStatus, 41)).rejects.toThrow(
      "Gateway action was replaced by process 99",
    );
  });

  it("times out when an action status request never settles", async () => {
    vi.useFakeTimers();
    const result = waitForActionCompletion(() => never, 41, { timeoutMs: 10 });
    const rejection = expect(result).rejects.toThrow(
      "Timed out waiting for gateway action completion",
    );

    await vi.advanceTimersByTimeAsync(10);

    await rejection;
  });

  it("rejects a completed action that resolves after the deadline", async () => {
    vi.useFakeTimers();
    let resolveStatus!: (status: {
      running: boolean;
      exit_code: number | null;
      pid: number | null;
    }) => void;
    const delayedStatus = new Promise<{
      running: boolean;
      exit_code: number | null;
      pid: number | null;
    }>((resolve) => {
      resolveStatus = resolve;
    });
    const result = waitForActionCompletion(() => delayedStatus, 41, {
      timeoutMs: 10,
    });
    const rejection = expect(result).rejects.toThrow(
      "Timed out waiting for gateway action completion",
    );

    await vi.advanceTimersByTimeAsync(11);
    resolveStatus({ running: false, exit_code: 0, pid: 41 });
    await vi.runAllTimersAsync();

    await rejection;
  });
});
