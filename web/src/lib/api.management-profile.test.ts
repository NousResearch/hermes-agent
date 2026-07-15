import { describe, expect, it, beforeEach } from "vitest";
import {
  getManagementProfile,
  managementScopedRequestUrl,
  setManagementProfile,
} from "@/lib/api";

describe("managementScopedRequestUrl", () => {
  beforeEach(() => {
    setManagementProfile("");
  });

  it("appends profile to /api/status when management profile is set", () => {
    setManagementProfile("faustina");
    expect(managementScopedRequestUrl("/api/status")).toBe(
      "/api/status?profile=faustina",
    );
  });

  it("leaves /api/status unchanged when management profile is empty", () => {
    expect(managementScopedRequestUrl("/api/status")).toBe("/api/status");
  });

  it("does not double-append when url already has profile=", () => {
    setManagementProfile("faustina");
    expect(
      managementScopedRequestUrl("/api/status?profile=other"),
    ).toBe("/api/status?profile=other");
  });

  it("does not scope machine-global endpoints", () => {
    setManagementProfile("faustina");
    expect(managementScopedRequestUrl("/api/cron/jobs")).toBe("/api/cron/jobs");
  });

  it("setManagementProfile trims whitespace", () => {
    setManagementProfile("  worker  ");
    expect(getManagementProfile()).toBe("worker");
    expect(managementScopedRequestUrl("/api/gateway")).toBe(
      "/api/gateway?profile=worker",
    );
  });
});