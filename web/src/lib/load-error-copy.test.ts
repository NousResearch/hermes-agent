import { describe, expect, it } from "vitest";

import { en } from "@/i18n/en";
import { loadErrorCopy } from "./load-error-copy";

describe("loadErrorCopy", () => {
  it("fills the translated template with what failed and the detail line", () => {
    const copy = loadErrorCopy(en.common, en.cron.loadWhat!, "The Hermes service hit an internal error.");
    expect(copy.title).toContain("cron jobs");
    expect(copy.title).not.toContain("{what}");
    expect(copy.title).toContain(en.common.retry);
    expect(copy.details).toContain("internal error");
    expect(copy.details).not.toContain("{detail}");
  });

  it("omits the details line when there is no detail", () => {
    expect(loadErrorCopy(en.common, en.skills.loadWhat!, null).details).toBeNull();
    expect(loadErrorCopy(en.common, en.skills.loadWhat!, "").details).toBeNull();
  });

  it("keeps the page strings in the locale table rather than the component", () => {
    for (const key of [en.skills.browseHub, en.skills.createSkill, en.cron.scriptRequired]) {
      expect(typeof key).toBe("string");
      expect(key!.trim()).not.toBe("");
    }
    // The script-only message names the field, not the `no_agent` config key.
    expect(en.cron.scriptRequired).toContain("Script");
    expect(en.cron.scriptRequired).not.toContain("no_agent");
  });
});
