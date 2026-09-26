import { expect, it } from "vitest";

import { en } from "./en";
import { countLabel } from "./count-label";
import { ru } from "./ru";

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
