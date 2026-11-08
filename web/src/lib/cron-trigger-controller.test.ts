import { describe, expect, it, vi } from "vitest";

import { createCronTriggerController } from "@hermes/shared";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, reject, resolve };
}

describe("createCronTriggerController", () => {
  it("announces immediately and coalesces the same job while it is running", async () => {
    const request = deferred<string>();
    const order: string[] = [];
    const action = vi.fn(() => {
      order.push("action");
      return request.promise;
    });
    const onStarted = vi.fn(() => order.push("started"));
    const onRunningChange = vi.fn();
    const controller = createCronTriggerController(onRunningChange);

    const first = controller.run("profile-a:job-1", action, onStarted);
    const duplicate = await controller.run("profile-a:job-1", action, onStarted);

    expect(action).toHaveBeenCalledTimes(1);
    expect(onStarted).toHaveBeenCalledTimes(1);
    expect(order).toEqual(["started", "action"]);
    expect(onRunningChange).toHaveBeenNthCalledWith(1, "profile-a:job-1", true);
    expect(duplicate).toEqual({ started: false, value: null });

    request.resolve("done");
    await expect(first).resolves.toEqual({ started: true, value: "done" });
    expect(onRunningChange).toHaveBeenLastCalledWith("profile-a:job-1", false);
  });

  it("releases the job after failure so a retry can start", async () => {
    const request = deferred<void>();
    const controller = createCronTriggerController();

    const failed = controller.run("job-1", () => request.promise);
    request.reject(new Error("failed"));

    await expect(failed).rejects.toThrow("failed");
    await expect(controller.run("job-1", async () => "retried")).resolves.toEqual({
      started: true,
      value: "retried",
    });
  });

  it("releases the job when the immediate feedback callback fails", async () => {
    const controller = createCronTriggerController();

    await expect(
      controller.run("job-1", async () => "not-called", () => {
        throw new Error("toast failed");
      }),
    ).rejects.toThrow("toast failed");

    await expect(controller.run("job-1", async () => "retried")).resolves.toEqual({
      started: true,
      value: "retried",
    });
  });

  it("allows the same job id in different profiles to run concurrently", async () => {
    const defaultRequest = deferred<string>();
    const workRequest = deferred<string>();
    const defaultAction = vi.fn(() => defaultRequest.promise);
    const workAction = vi.fn(() => workRequest.promise);
    const controller = createCronTriggerController();

    const defaultRun = controller.run("default:job-1", defaultAction);
    const workRun = controller.run("work:job-1", workAction);

    expect(defaultAction).toHaveBeenCalledTimes(1);
    expect(workAction).toHaveBeenCalledTimes(1);

    defaultRequest.resolve("default");
    workRequest.resolve("work");

    await expect(defaultRun).resolves.toEqual({ started: true, value: "default" });
    await expect(workRun).resolves.toEqual({ started: true, value: "work" });
  });

  it("releases the job when the running-state callback fails", async () => {
    const controller = createCronTriggerController((_key, running) => {
      if (running) throw new Error("state callback failed");
    });

    await expect(controller.run("job-1", async () => "not-called")).rejects.toThrow(
      "state callback failed",
    );
    expect(controller.isRunning("job-1")).toBe(false);
  });

  it("releases the job before the stopped-state callback runs", async () => {
    const action = vi.fn(async () => "done");
    const controller = createCronTriggerController((_key, running) => {
      if (!running) throw new Error("stopped callback failed");
    });

    await expect(controller.run("job-1", action)).rejects.toThrow("stopped callback failed");
    expect(controller.isRunning("job-1")).toBe(false);

    await expect(controller.run("job-1", action)).rejects.toThrow("stopped callback failed");
    expect(action).toHaveBeenCalledTimes(2);
  });
});
