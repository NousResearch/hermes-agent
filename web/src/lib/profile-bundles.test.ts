import { describe, expect, it } from "vitest";

import {
  buildProfileBundlePlan,
  defaultProfileBundle,
  getProfileBundle,
  listProfileBundles,
  sanitizeBundlePrefix,
} from "./profile-bundles";

describe("profile bundle catalog", () => {
  it("exposes starter bundles with role definitions", () => {
    const bundles = listProfileBundles();
    const ids = new Set(bundles.map((bundle) => bundle.id));

    expect(ids.has("development")).toBe(true);
    expect(ids.has("finance")).toBe(true);
    expect(defaultProfileBundle().id).toBe("development");

    for (const bundle of bundles) {
      expect(bundle.roles.length).toBeGreaterThan(0);
      for (const role of bundle.roles) {
        expect(role.slug).toMatch(/^[a-z0-9][a-z0-9_-]*$/);
        expect(role.title.length).toBeGreaterThan(0);
        expect(role.description.length).toBeGreaterThan(0);
        expect(role.focus.length).toBeGreaterThan(0);
      }
    }
  });

  it("keeps role slugs unique inside each bundle", () => {
    for (const bundle of listProfileBundles()) {
      const slugs = bundle.roles.map((role) => role.slug);
      expect(new Set(slugs).size).toBe(slugs.length);
    }
  });

  it("looks up bundles by id", () => {
    expect(getProfileBundle("finance")?.title).toBe("Finance Ops");
    expect(getProfileBundle("missing")).toBeUndefined();
  });
});

describe("sanitizeBundlePrefix", () => {
  it("normalizes user-entered prefixes", () => {
    expect(sanitizeBundlePrefix(" Dev Team! ", "dev")).toBe("dev-team");
    expect(sanitizeBundlePrefix("QA_suite", "qa")).toBe("qa_suite");
  });

  it("falls back when the prefix has no usable characters", () => {
    expect(sanitizeBundlePrefix("!!!", "dev")).toBe("dev");
    expect(sanitizeBundlePrefix("   ", "fin")).toBe("fin");
  });
});

describe("buildProfileBundlePlan", () => {
  it("derives profile names and conflict markers from bundle data", () => {
    const development = getProfileBundle("development");
    expect(development).toBeDefined();
    if (!development) return;

    const plan = buildProfileBundlePlan(
      development,
      "dev",
      new Set(["dev-developer"]),
    );

    expect(plan.map((item) => item.name)).toContain("dev-developer");
    expect(plan.find((item) => item.name === "dev-developer")?.exists).toBe(
      true,
    );
    expect(plan.find((item) => item.name === "dev-qa")?.exists).toBe(false);
  });
});
