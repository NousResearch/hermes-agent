import { describe, expect, it } from "vitest";

import { en } from "./en";
import { fr } from "./fr";
import { tr } from "./tr";
import type { Translations } from "./types";
import { resolvePageTitle } from "../lib/resolve-page-title";

const LOCALIZED_BUILTIN_ROUTES = [
  ["/chat", "chat"],
  ["/sessions", "sessions"],
  ["/analytics", "analytics"],
  ["/models", "models"],
  ["/logs", "logs"],
  ["/cron", "cron"],
  ["/skills", "skills"],
  ["/plugins", "plugins"],
  ["/mcp", "mcp"],
  ["/channels", "channels"],
  ["/webhooks", "webhooks"],
  ["/pairing", "pairing"],
  ["/profiles", "profiles"],
  ["/config", "config"],
  ["/env", "keys"],
  ["/system", "system"],
  ["/docs", "documentation"],
] as const;

function navEntry(t: Translations, key: string): string | undefined {
  return (t.app.nav as Record<string, string | undefined>)[key];
}

describe("localized built-in navigation", () => {
  it.each(LOCALIZED_BUILTIN_ROUTES)(
    "resolves %s through the active locale entry",
    (path, key) => {
      expect(resolvePageTitle(path, fr, [])).toBe(navEntry(fr, key));
      expect(resolvePageTitle(path, en, [])).toBe(navEntry(en, key));
    },
  );

  it("resolves the files route through the active locale entry", () => {
    expect(navEntry(fr, "files")).toBeDefined();
    expect(navEntry(en, "files")).toBeDefined();
    expect(resolvePageTitle("/files", fr, [])).toBe(navEntry(fr, "files"));
    expect(resolvePageTitle("/files", en, [])).toBe(navEntry(en, "files"));
  });

  it("falls back to the English files label when a locale omits the entry", () => {
    expect(navEntry(tr, "files")).toBeUndefined();
    expect(resolvePageTitle("/files", tr, [])).toBe(navEntry(en, "files"));
  });
});
