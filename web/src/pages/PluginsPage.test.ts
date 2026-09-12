import { describe, expect, it } from "vitest";
import type { MemoryProviderSetupInfo } from "@/lib/api";
import { setupHasDetails, setupHasInstallableSteps } from "@/lib/memory-provider-setup";

describe("memory provider preparation", (): void => {
  it("offers preparation for a PM member without legacy pip declarations", (): void => {
    const setup: MemoryProviderSetupInfo = {
      pip_dependencies: [],
      python_dependencies_declared: true,
      external_dependencies: [],
      required_env: [],
      dependencies_installed: false,
    };
    expect(setupHasDetails(setup)).toBe(true);
    expect(setupHasInstallableSteps(setup)).toBe(true);
    expect(setupHasInstallableSteps({ ...setup, python_dependencies_declared: false })).toBe(false);
  });
});