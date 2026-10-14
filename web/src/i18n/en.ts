import type { Translations } from "./types";
import { shellEn } from "./en/shell";
import { workspaceEn } from "./en/workspace";
import { settingsEn } from "./en/settings";
import { extensionsEn } from "./en/extensions";

/** Dashboard source catalog, composed from topical catalogs. */
export const en: Translations = {
  ...shellEn,
  ...workspaceEn,
  ...settingsEn,
  ...extensionsEn,
};
