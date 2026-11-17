import { describe, expect, it } from "vitest";
import { pollActivity, pollInterval, type PollSurface } from "./poll-interval";

const LEGACY_FIXED_MS: Record<PollSurface, number> = {
  chatSessions: 20_000,
  logs: 5_000,
  sessions: 5_000,
  sidebarStatus: 10_000,
};
const SURFACES = Object.keys(LEGACY_FIXED_MS) as PollSurface[];

describe("pollInterval", () => {
  it("polls a settled dashboard at the old fixed cadence and faster only mid-change", () => {
    const idle = { gateway_running: true, gateway_state: "running", active_agents: 0 };
    for (const surface of SURFACES) {
      expect(pollInterval(surface, pollActivity(idle))).toBe(LEGACY_FIXED_MS[surface]);
      expect(pollInterval(surface, pollActivity(null))).toBe(LEGACY_FIXED_MS[surface]);
      const working = pollInterval(surface, pollActivity({ ...idle, active_agents: 2 }));
      expect(working).toBeLessThan(LEGACY_FIXED_MS[surface]);
    }
  });

  it("treats a booting or draining gateway as mid-change, but not a dead one's agent count", () => {
    expect(pollActivity({ gateway_running: true, gateway_state: "starting" })).toBe("transitioning");
    expect(pollActivity({ gateway_running: true, gateway_state: "draining" })).toBe("transitioning");
    expect(pollActivity({ gateway_running: false, gateway_state: "stopped", active_agents: 3 })).toBe(
      "stable",
    );
    expect(pollActivity({ gateway_running: false, gateway_state: "startup_failed" })).toBe("stable");
  });
});
