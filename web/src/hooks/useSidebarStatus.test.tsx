// @vitest-environment jsdom
// The dashboard's ProfileSwitcher mirrors its selection into a module-level
// global (`setManagementProfile`) that fetchJSON reads at call time, so a
// poll fired for the OLD profile can still resolve after the user switches.
// Regression coverage: a stale response must never overwrite the new
// profile's status (see MemoryPressureBanner's "profile:" diagnostic).

import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useSidebarStatus } from "./useSidebarStatus";
import type { StatusResponse } from "@/lib/api";

let managementProfile = "";
let resolvers: Array<(value: StatusResponse) => void> = [];

vi.mock("@/lib/api", () => ({
  api: {
    getStatus: vi.fn(
      () =>
        new Promise<StatusResponse>((resolve) => {
          resolvers.push(resolve);
        }),
    ),
  },
  getManagementProfile: () => managementProfile,
}));

let container: HTMLDivElement;
let root: Root;
let latest: StatusResponse | null = null;

function Probe() {
  const status = useSidebarStatus();
  useEffect(() => {
    latest = status;
  }, [status]);
  return null;
}

beforeEach(() => {
  managementProfile = "";
  resolvers = [];
  latest = null;
  vi.useFakeTimers();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.useRealTimers();
});

function statusFor(profile: string): StatusResponse {
  return { hermes_home: profile } as unknown as StatusResponse;
}

describe("useSidebarStatus", () => {
  it("drops a stale-profile response that resolves after the profile switched", async () => {
    await act(async () => root.render(<Probe />));
    expect(resolvers).toHaveLength(1); // the mount-time poll, fired for "".

    // User switches the management profile before the in-flight poll answers.
    managementProfile = "alpha";
    await act(async () => resolvers[0](statusFor("default")));
    expect(latest).toBeNull(); // stale "default" data must not render as "alpha".

    // The next scheduled poll is fired (and answered) under the new profile.
    await act(async () => {
      vi.advanceTimersByTime(10_000);
    });
    expect(resolvers).toHaveLength(2);
    await act(async () => resolvers[1](statusFor("alpha")));
    expect(latest).toEqual(statusFor("alpha"));
  });
});
