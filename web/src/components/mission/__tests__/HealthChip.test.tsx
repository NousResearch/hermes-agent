import { describe, it } from "vitest";
import { render } from "@testing-library/react";
import { HealthChipCard } from "../HealthChip";
import type { HealthChip } from "../types";

const base: HealthChip = {
  name: "ruflo",
  label: "Ruflo",
  status: "online",
  latencyMs: 12,
  lastChecked: new Date().toISOString(),
};

describe("HealthChipCard", () => {
  it("renders without crashing", () => {
    render(<HealthChipCard chip={base} />);
  });

  it.each(["online", "degraded", "offline", "unknown"] as const)(
    "renders status=%s",
    (status) => {
      render(<HealthChipCard chip={{ ...base, status }} />);
    }
  );

  it("renders with optional detail", () => {
    render(<HealthChipCard chip={{ ...base, detail: "3 active tasks" }} />);
  });
});
