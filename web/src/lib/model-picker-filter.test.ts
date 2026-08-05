import { describe, it, expect } from "vitest";
import { queryMatchesProviderOnly } from "./model-picker-filter";

describe("queryMatchesProviderOnly", () => {
  it("returns true when the query finds the provider but no model id (issue #65374)", () => {
    // Reproduces the exact case from the issue: typing "aws" locates the
    // "AWS Build" provider, but none of its Claude model ids contain "aws".
    const provider = { name: "AWS Build", slug: "aws-build" };
    const models = ["claude-sonnet-4.5", "claude-sonnet-4", "claude-haiku-4.5"];

    expect(queryMatchesProviderOnly(provider, models, "aws")).toBe(true);
  });

  it("returns false when the query also matches a model id — keeps normal filtering", () => {
    const provider = { name: "AWS Build", slug: "aws-build" };
    const models = ["claude-sonnet-4.5", "claude-sonnet-4", "claude-haiku-4.5"];

    expect(queryMatchesProviderOnly(provider, models, "sonnet")).toBe(false);
  });

  it("returns false when the query does not match the provider at all", () => {
    const provider = { name: "AWS Build", slug: "aws-build" };
    const models = ["claude-sonnet-4.5"];

    expect(queryMatchesProviderOnly(provider, models, "openrouter")).toBe(false);
  });

  it("returns false for an empty query", () => {
    const provider = { name: "AWS Build", slug: "aws-build" };
    const models = ["claude-sonnet-4.5"];

    expect(queryMatchesProviderOnly(provider, models, "")).toBe(false);
  });

  it("returns false when there is no selected provider", () => {
    expect(queryMatchesProviderOnly(null, ["claude-sonnet-4.5"], "aws")).toBe(false);
  });
});
