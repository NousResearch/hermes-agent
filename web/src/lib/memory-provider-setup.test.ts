import { describe, expect, it } from "vitest";

import { memorySetupHasInstallableSteps } from "./memory-provider-setup";

describe("memorySetupHasInstallableSteps", () => {
  it("uses the public installable flag without requiring command bodies", () => {
    expect(
      memorySetupHasInstallableSteps({
        dependencies_installed: false,
        external_dependencies: [{ name: "brv", installable: true }],
        pip_dependencies: [],
        required_env: [],
      }),
    ).toBe(true);
  });
});
