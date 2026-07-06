import { describe, expect, it } from "vitest";

import {
  buildControlRoomSnapshot,
  formatControlRoomRelativeTime,
  selectPriorityCronJobs,
} from "./control-room";
import type { CronJob, SessionInfo, StatusResponse, ToolsetInfo } from "./api";

const baseStatus: StatusResponse = {
  active_sessions: 2,
  config_path: "/Users/mouxy/.hermes/config.yaml",
  config_version: 3,
  env_path: "/Users/mouxy/.hermes/.env",
  gateway_exit_reason: null,
  gateway_health_url: "http://127.0.0.1:8765/health",
  gateway_pid: 1234,
  gateway_platforms: {
    photon: { state: "connected", updated_at: "2026-07-04T11:59:00Z" },
    telegram: {
      state: "error",
      error_message: "offline",
      updated_at: "2026-07-04T11:58:00Z",
    },
    discord: { state: "disabled", updated_at: "2026-07-04T11:57:00Z" },
  },
  gateway_running: true,
  gateway_state: "running",
  gateway_updated_at: "2026-07-04T08:45:00Z",
  hermes_home: "/Users/mouxy/.hermes",
  latest_config_version: 3,
  release_date: "2026-07-01",
  version: "0.5.0",
};

const sessions: SessionInfo[] = [
  {
    id: "active-business",
    source: "photon",
    model: "gpt-5.5",
    title: "Business project",
    started_at: 1783150000,
    ended_at: null,
    last_active: 1783150500,
    is_active: true,
    message_count: 32,
    tool_call_count: 8,
    input_tokens: 1000,
    output_tokens: 400,
    preview: "Working on Hermes control room",
  },
  {
    id: "quiet",
    source: "telegram",
    model: "gpt-5.5",
    title: "Older work",
    started_at: 1783140000,
    ended_at: 1783140300,
    last_active: 1783140300,
    is_active: false,
    message_count: 4,
    tool_call_count: 0,
    input_tokens: 100,
    output_tokens: 40,
    preview: null,
  },
];

const jobs: CronJob[] = [
  {
    id: "healthy",
    name: "Daily backup",
    enabled: true,
    state: "idle",
    schedule_display: "daily at 03:00",
    next_run_at: "2026-07-05T03:00:00Z",
    last_status: "success",
  },
  {
    id: "paused",
    name: "Paused research",
    enabled: false,
    state: "paused",
    schedule_display: "every 2h",
    last_status: "success",
  },
  {
    id: "broken",
    name: "Broken canary",
    enabled: true,
    state: "idle",
    schedule_display: "every 15m",
    last_status: "error",
    last_error: "HEY CLI failed",
  },
];

describe("formatControlRoomRelativeTime", () => {
  it("formats seconds, minutes, hours and empty timestamps compactly", () => {
    const now = new Date("2026-07-04T12:00:00Z");

    expect(formatControlRoomRelativeTime("2026-07-04T11:59:35Z", now)).toBe("25s ago");
    expect(formatControlRoomRelativeTime("2026-07-04T11:17:00Z", now)).toBe("43m ago");
    expect(formatControlRoomRelativeTime("2026-07-04T08:30:00Z", now)).toBe("3h ago");
    expect(formatControlRoomRelativeTime(null, now)).toBe("—");
  });
});

describe("selectPriorityCronJobs", () => {
  it("prioritises failed jobs, then paused jobs, then upcoming enabled jobs", () => {
    expect(selectPriorityCronJobs(jobs).map((job) => job.id)).toEqual([
      "broken",
      "paused",
      "healthy",
    ]);
  });
});

describe("buildControlRoomSnapshot", () => {
  it("combines status, sessions, cron and toolsets into cockpit metrics", () => {
    const snapshot = buildControlRoomSnapshot({
      status: baseStatus,
      sessions,
      jobs,
      toolsets: [
        { name: "web", label: "Web", description: "", enabled: true, configured: true, tools: [] },
        { name: "terminal", label: "Terminal", description: "", enabled: true, configured: true, tools: [] },
        { name: "spotify", label: "Spotify", description: "", enabled: false, configured: false, tools: [] },
      ] satisfies ToolsetInfo[],
      now: new Date("2026-07-04T12:00:00Z"),
    });

    expect(snapshot.gateway.health).toBe("warning");
    expect(snapshot.gateway.connectedPlatforms).toBe(1);
    expect(snapshot.gateway.enabledPlatforms).toBe(2);
    expect(snapshot.gateway.primaryLine).toContain("1/2 connected");
    expect(snapshot.sessions.active).toBe(1);
    expect(snapshot.sessions.recent[0].title).toBe("Business project");
    expect(snapshot.cron.failed).toBe(1);
    expect(snapshot.cron.paused).toBe(1);
    expect(snapshot.cron.priorityJobs.map((job) => job.id)).toEqual([
      "broken",
      "paused",
      "healthy",
    ]);
    expect(snapshot.tools.enabled).toBe(2);
    expect(snapshot.tools.configured).toBe(2);
  });
}
);
