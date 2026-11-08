import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { flattenMessageKeys } from "@hermes/shared/i18n";
import { expect, it } from "vitest";

import { en } from "./en";

const repoRoot = resolve(__dirname, "../../..");

it("the committed locales/_keys.web.json matches the English catalog (run `pnpm run i18n:keys`)", () => {
  const committed = JSON.parse(readFileSync(join(repoRoot, "locales", "_keys.web.json"), "utf8"));

  expect(committed.surface).toBe("web");
  expect(committed.keys).toEqual(flattenMessageKeys(en));
  expect(committed.keys).toContain("common.save");
  expect(committed.keys).toContain("pluginsPage.updateAllDone"); // function-valued entries are keys too
});
