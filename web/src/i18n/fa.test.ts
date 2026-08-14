import { describe, expect, it } from "vitest";

import { fa } from "./fa";

// The dashboard pluralizes count labels by substituting the `{s}` token with a
// literal "s" (EnvPage/SkillsPage/ConfigPage: `.replace("{s}", n !== 1 ? "s" :
// "")`). Persian nouns stay singular after a numeral — "۵ مهارت", never
// "۵ مهارتs" — so the Persian catalog must not carry the token at all. When it
// is absent the replace is a no-op, which keeps the shared component code
// untouched.
const renderCount = (template: string, count: number) =>
  template
    .replace("{count}", String(count))
    .replace("{s}", count !== 1 ? "s" : "");

// Labels that interpolate the number themselves.
const countTemplates = {
  "skills.skillCount": fa.skills.skillCount,
  "skills.resultCount": fa.skills.resultCount,
  "env.keysCount": fa.env.keysCount,
  "env.customConfigured": fa.env.customConfigured,
};

// `config.fields` is a bare noun: ConfigPage renders the number in JSX next to
// it (`{fields.length}{" "}{t.config.fields...}`), so the string itself carries
// no `{count}`.
const allLabels = { ...countTemplates, "config.fields": fa.config.fields };

describe("Persian locale pluralization", () => {
  it.each(Object.entries(allLabels))(
    "%s carries no English {s} plural token",
    (_key, template) => {
      expect(template).not.toContain("{s}");
    },
  );

  it.each(Object.entries(countTemplates))(
    "%s differs only by the interpolated number across counts",
    (_key, template) => {
      expect(template).toContain("{count}");
      // Rendering must equal the template with *only* {count} substituted —
      // proving the plural branch contributes nothing for either count.
      expect(renderCount(template, 1)).toBe(template.replace("{count}", "1"));
      expect(renderCount(template, 5)).toBe(template.replace("{count}", "5"));
    },
  );

  it("config.fields is count-invariant", () => {
    expect(fa.config.fields).not.toContain("{count}");
    expect(renderCount(fa.config.fields, 1)).toBe(
      renderCount(fa.config.fields, 5),
    );
  });

  it("renders the expected Persian text for singular and plural counts", () => {
    expect(renderCount(fa.skills.skillCount, 1)).toBe("1 مهارت");
    expect(renderCount(fa.skills.skillCount, 5)).toBe("5 مهارت");
    expect(renderCount(fa.skills.resultCount, 1)).toBe("1 نتیجه");
    expect(renderCount(fa.skills.resultCount, 5)).toBe("5 نتیجه");
    expect(renderCount(fa.env.keysCount, 1)).toBe("1 کلید");
    expect(renderCount(fa.env.keysCount, 3)).toBe("3 کلید");
    expect(renderCount(fa.env.customConfigured, 1)).toBe(
      "1 کلید سفارشی تنظیم شد",
    );
    expect(renderCount(fa.env.customConfigured, 2)).toBe(
      "2 کلید سفارشی تنظیم شد",
    );
    expect(renderCount(fa.config.fields, 1)).toBe("فیلد");
    expect(renderCount(fa.config.fields, 5)).toBe("فیلد");
  });
});
