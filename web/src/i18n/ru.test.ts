import { expect, it } from "vitest";

import { en } from "./en";
import { countLabel } from "./count-label";
import { ru } from "./ru";

function leaves(value: unknown, prefix = ""): Map<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return new Map([[prefix, value]]);
  return new Map(
    Object.entries(value).flatMap(([key, child]) =>
      [...leaves(child, prefix ? `${prefix}.${key}` : key)],
    ),
  );
}

it("keeps the Russian dashboard catalog aligned with English keys and placeholders", () => {
  const source = leaves(en);
  const translated = leaves(ru);
  expect([...translated.keys()].sort()).toEqual([...source.keys()].sort());
  const slots = (value: string) => [...value.matchAll(/\{[A-Za-z_]\w*\}/g)].map(match => match[0]).sort();
  for (const [path, value] of source) {
    if (typeof value === "string") {
      // {s} is an English suffix, not data: Russian uses an invariable count label.
      const expected = slots(value).filter(slot => slot !== "{s}");
      expect(slots(translated.get(path) as string), path).toEqual(path === "config.fields" ? ["{count}"] : expected);
    }
  }
});

it("keeps Russian session deletion confirmations grammatical at plural boundaries", () => {
  const templates = [
    ru.sessions.deleteEmptyConfirmMessage,
    ru.sessions.deleteSelectedConfirmTitle,
    ru.sessions.deleteSelectedConfirmMessage,
  ];

  for (const count of [1, 2, 5, 11, 21]) {
    for (const template of templates) {
      const rendered = template.replace("{count}", String(count));
      expect(rendered).toContain(String(count));
      expect(rendered).not.toMatch(/\b1 сессий\b|\b2 выбранных сессий\b|\b21 сессий\b/);
    }
  }
});

it("formats count labels through locale copy without adding English suffixes to Russian", () => {
  for (const count of [1, 2, 5, 11, 21]) {
    expect(countLabel(ru.skills.skillCount, count)).toBe(`Навыков: ${count}`);
    expect(countLabel(ru.skills.resultCount, count)).toBe(`Результатов: ${count}`);
    expect(countLabel(ru.env.keysCount, count)).toBe(`Ключей: ${count}`);
    expect(countLabel(ru.config.fields, count)).toBe(`Полей: ${count}`);
  }
  expect(countLabel(en.skills.skillCount, 1)).toBe("1 skill");
  expect(countLabel(en.skills.skillCount, 2)).toBe("2 skills");
  expect(countLabel(en.config.fields, 1)).toBe("1 field");
  expect(countLabel(en.config.fields, 2)).toBe("2 fields");
});

it("keeps valid profile names and nested-plugin install syntax in Russian hints", () => {
  expect(ru.profiles.namePlaceholder).toContain("coder");
  expect(ru.profiles.nameRule).toContain("латинские");
  expect(ru.pluginsPage.installHint).toContain("owner/repo/path/to/plugin");
  expect(ru.pluginsPage.installHint).toContain("<url>#path/to/plugin");
});
