import { expect, it } from "vitest";

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
