import { describe, expect, it } from "vitest";

import { serviceWorkerScope, serviceWorkerUrl } from "./pwa";

describe("PWA path helpers", () => {
  it("keeps root deployments rooted", () => {
    expect(serviceWorkerUrl("")).toBe("/sw.js");
    expect(serviceWorkerScope("")).toBe("/");
  });

  it("keeps reverse-proxy deployments inside their base path", () => {
    expect(serviceWorkerUrl("/hermes/" )).toBe("/hermes/sw.js");
    expect(serviceWorkerScope("/hermes/" )).toBe("/hermes/");
  });
});
