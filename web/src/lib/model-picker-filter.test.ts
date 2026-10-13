import { describe, it, expect } from "vitest";
import { queryMatchesProviderOnly, verbatimModelId } from "./model-picker-filter";

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

describe("verbatimModelId", () => {
  const openrouter = {
    slug: "openrouter",
    models: ["z-ai/glm-5.3-flash", "z-ai/glm-4.7-flash", "deepseek/deepseek-v4-flash"],
  };

  it("offers a provider-pinned id that is not in the catalog", () => {
    expect(verbatimModelId(openrouter, "z-ai/glm-5.3-flash:wafer")).toBe(
      "z-ai/glm-5.3-flash:wafer",
    );
  });

  it("offers an id with a quantization-tag suffix", () => {
    expect(verbatimModelId(openrouter, "z-ai/glm-5.3-flash:deepinfra/fp4")).toBe(
      "z-ai/glm-5.3-flash:deepinfra/fp4",
    );
  });

  it("returns null when the query exactly matches a listed model", () => {
    expect(verbatimModelId(openrouter, "z-ai/glm-5.3-flash")).toBeNull();
  });

  it("returns null for a non-model-id query like 'fast cheap'", () => {
    expect(verbatimModelId(openrouter, "fast cheap")).toBeNull();
    expect(verbatimModelId(openrouter, "glm")).toBeNull();
  });

  it("returns null for non-openrouter providers", () => {
    expect(
      verbatimModelId(
        { slug: "deepseek", models: ["deepseek-chat"] },
        "deepseek/deepseek-chat:wafer",
      ),
    ).toBeNull();
  });

  it("returns null without a selected provider", () => {
    expect(verbatimModelId(null, "z-ai/glm-5.3-flash:wafer")).toBeNull();
  });
});
