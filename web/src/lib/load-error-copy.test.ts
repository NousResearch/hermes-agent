import { describe, expect, it } from "vitest";

import { en } from "@/i18n/en";
import { loadErrorCopy } from "./load-error-copy";

describe("loadErrorCopy", () => {
  it("fills the translated template with what failed and the detail line", () => {
    const copy = loadErrorCopy(en.common, en.cron.loadWhat!, "The Hermes service hit an internal error.");
    expect(copy.title).toContain(en.cron.loadWhat!);
    expect(copy.title).not.toContain("{what}");
    expect(copy.title).toContain(en.common.retry);
    expect(copy.details).toContain("internal error");
    expect(copy.details).not.toContain("{detail}");
  });

  it("omits the details line when there is no detail", () => {
    expect(loadErrorCopy(en.common, en.skills.loadWhat!, null).details).toBeNull();
    expect(loadErrorCopy(en.common, en.skills.loadWhat!, "").details).toBeNull();
  });
});
