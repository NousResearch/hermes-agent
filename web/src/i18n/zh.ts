import type { TranslationOverlay } from "./types";
import { shellZh } from "./zh/shell";
import { workspaceZh } from "./zh/workspace";
import { settingsZh } from "./zh/settings";
import { extensionsZh } from "./zh/extensions";

/** Dashboard Simplified Chinese pack, composed from topical catalogs. */
export const zh: TranslationOverlay = {
  ...shellZh,
  ...workspaceZh,
  ...settingsZh,
  ...extensionsZh,
};
