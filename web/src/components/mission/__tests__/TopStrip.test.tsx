import { describe, it } from "vitest";
import { render } from "@testing-library/react";
import { TopStrip } from "../TopStrip";
import type { MissionSnapshot } from "../types";

const base: MissionSnapshot = {
  model: "claude-sonnet-4-6",
  spendToday: 0.42,
  spendWeek: 3.15,
  streakDays: 7,
  runtimes: [],
  recentSessions: [],
};

describe("TopStrip", () => {
  it("renders without crashing", () => {
    render(<TopStrip snapshot={base} />);
  });

  it("renders with streak of 0", () => {
    render(<TopStrip snapshot={{ ...base, streakDays: 0 }} />);
  });

  it("renders with no spend data", () => {
    render(<TopStrip snapshot={{ ...base, spendToday: 0, spendWeek: 0 }} />);
  });
});
