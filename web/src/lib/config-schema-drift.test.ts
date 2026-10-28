import { describe, expect, it } from "vitest";

import { configSchemaBehind } from "./config-schema-drift";

describe("configSchemaBehind", () => {
  it("counts the unapplied migrations", () => {
    // The shape this was written for: a container pinned weeks behind its config schema.
    expect(configSchemaBehind({ config_version: 33, latest_config_version: 44 })).toBe(11);
  });

  it("is silent when the schema is current", () => {
    expect(configSchemaBehind({ config_version: 45, latest_config_version: 45 })).toBe(0);
  });

  it("clamps a config written by a newer build than the one serving it", () => {
    // A downgrade or a shared HERMES_HOME. Real state, but "-3 behind" is not a thing.
    expect(configSchemaBehind({ config_version: 48, latest_config_version: 45 })).toBe(0);
  });

  it("stays silent when either number is absent", () => {
    // An older backend that does not report them must not render a bogus drift.
    expect(configSchemaBehind({ config_version: 33 })).toBe(0);
    expect(configSchemaBehind({ latest_config_version: 44 })).toBe(0);
    expect(configSchemaBehind(null)).toBe(0);
    expect(configSchemaBehind(undefined)).toBe(0);
  });
});
