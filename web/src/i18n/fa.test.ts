import { describe, expect, it } from "vitest";

import { fa } from "./fa";

// The dashboard pluralizes English count labels by substituting the `{s}`
// token with a literal "s" (see EnvPage/SkillsPage: `.replace("{s}", n !== 1 ?
// "s" : "")`). Persian nouns stay singular after a numeral — "۵ مهارت", never
// "۵ مهارت‌s" — so the Persian catalog must not carry the token at all. When it
// is absent the replace is a harmless no-op, which keeps the shared component
// code untouched.
const renderCount = (template: string, count: number) =>
  template.replace("{count}", String(count)).replace("{s}", count !== 1 ? "s" : "");

describe("Persian locale pluralization", () => {
  const countStrings = {
    "skills.skillCount": fa.skills.skillCount,
    "skills.resultCount": fa.skills.resultCount,
    "config.fields": fa.config.fields,
    "env.keysCount": fa.env.keysCount,
    "env.customConfigured": fa.env.customConfigured,
  };

  it.each(Object.entries(countStrings))(
    "%s carries no English {s} plural token",
    (_key, template) => {
      expect(template).not.toContain("{s}");
    },
  );

  it.each(Object.entries(countStrings))(
    "%s renders identically for singular and plural counts",
    (_key, template) => {
      expect(renderCount(template, 1)).not.toMatch(/s$/);
      expect(renderCount(template, 5)).not.toMatch(/s$/);
    },
  );

  it("keeps the count placeholder so the number still interpolates", () => {
    expect(renderCount(fa.skills.skillCount, 5)).toBe("5 مهارت");
    expect(renderCount(fa.skills.resultCount, 1)).toBe("1 نتیجه");
    expect(renderCount(fa.env.keysCount, 3)).toBe("3 کلید");
    expect(renderCount(fa.env.customConfigured, 2)).toBe("2 کلید سفارشی تنظیم شد");
  });
});
