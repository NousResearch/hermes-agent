import { describe, expect, it } from "vitest";

import config from "../vite.config";

describe("dashboard production build", () => {
  it("uses relative chunk URLs so lazy pages stay under a proxy prefix", () => {
    expect(config.base).toBe("./");
  });
});
