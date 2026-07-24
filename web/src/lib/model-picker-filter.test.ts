import { describe, it, expect } from "vitest";
import { fuzzyRank } from "./fuzzy";
import {
  filterModelPickerProviders,
  modelQueryForSelectedProvider,
  queryMatchesProviderOnly,
} from "./model-picker-filter";

const providers = [
  {
    name: "Custom GLM Gateway",
    slug: "custom-glm",
    models: [
      "glm-4-flash",
      "glm-4-plus",
      "glm-5.2",
      "glm-5.2-thinking",
      "qwen-2.5-72b",
    ],
  },
  {
    name: "NVIDIA NIM",
    slug: "nvidia-nim",
    models: [
      "llama-3.1-8b-instruct",
      "mistral-7b-instruct",
      "glm-5.2",
      "deepseek-r1-distill-llama-70b",
    ],
  },
  {
    name: "OpenRouter",
    slug: "openrouter",
    models: ["claude-sonnet-4", "gpt-4o", "glm-5.2"],
  },
];

describe("filterModelPickerProviders", () => {
  it("keeps model-only provider matches in backend order", () => {
    expect(
      filterModelPickerProviders(providers, "glm-5.2").map((p) => p.slug),
    ).toEqual(["custom-glm", "nvidia-nim", "openrouter"]);
  });

  it("still ranks provider identity matches", () => {
    expect(
      filterModelPickerProviders(providers, "nvidia").map((p) => p.slug),
    ).toEqual(["nvidia-nim"]);
  });

  it("does not include providers without provider or model matches", () => {
    expect(
      filterModelPickerProviders(providers, "claude").map((p) => p.slug),
    ).toEqual(["openrouter"]);
  });
});

describe("modelQueryForSelectedProvider", () => {
  it("keeps normal model-only filtering", () => {
    expect(
      modelQueryForSelectedProvider(providers[0], providers[0].models, "glm-5.2"),
    ).toBe("glm-5.2");
  });

  it("returns an empty query when the query only locates the selected provider", () => {
    const provider = { name: "AWS Build", slug: "aws-build" };
    const models = ["claude-sonnet-4.5", "claude-sonnet-4", "claude-haiku-4.5"];

    expect(modelQueryForSelectedProvider(provider, models, "aws")).toBe("");
  });

  it("supports mixed provider and model queries in the provider and model columns", () => {
    const [provider] = filterModelPickerProviders(providers, "nvidia glm-5.2");
    const modelQuery = modelQueryForSelectedProvider(
      provider,
      provider.models,
      "nvidia glm-5.2",
    );

    expect(provider.slug).toBe("nvidia-nim");
    expect(modelQuery).toBe("glm-5.2");
    expect(
      fuzzyRank(provider.models, modelQuery, (model) => model).map(
        (r) => r.item,
      ),
    ).toEqual(["glm-5.2"]);
  });
});

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
