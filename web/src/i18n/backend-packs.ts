/**
 * Backend-delivered language packs for the web dashboard. A Hermes
 * language-pack plugin (or a user overlay under `$HERMES_HOME/locales/`)
 * ships `<lang>.web.yaml`; the dashboard serves the flattened strings over
 * `GET /api/i18n/catalog?lang=&surface=web` and the selectable languages over
 * `GET /api/i18n/languages` — the HTTP twin of the `i18n.catalog` /
 * `i18n.languages` JSON-RPC pair Desktop uses (`apps/desktop/src/i18n/backend-packs.ts`).
 *
 * Tolerant by design: an older backend without the routes (404), a network
 * hiccup or a malformed reply leaves the bundled catalog in place — never
 * blank the UI over i18n.
 */

import {
  adaptStringOverrides,
  isRecord,
  mergeTranslations,
  unflattenMessages,
  type FlatMessages,
  type TranslationOverride,
} from "@hermes/shared/i18n";

import { fetchJSON } from "@/lib/api";

import type { Translations } from "./types";

export const WEB_SURFACE = "web";

/** One row of `/api/i18n/languages` (`agent.i18n.language_options()`). */
export interface BackendLanguageOption {
  id: string;
  endonym?: string;
  rtl?: boolean;
  source?: string;
  /** Surfaces some backend layer has strings for. */
  surfaces?: string[];
}

function languageRows(value: unknown): BackendLanguageOption[] {
  const rows = isRecord(value) && Array.isArray(value.languages) ? value.languages : [];
  return rows.filter(
    (row): row is BackendLanguageOption =>
      isRecord(row) && typeof row.id === "string" && row.id.trim() !== "",
  );
}

function catalogMessages(value: unknown): FlatMessages {
  if (!isRecord(value) || !isRecord(value.messages)) return {};
  const messages: FlatMessages = {};
  for (const [key, text] of Object.entries(value.messages)) {
    if (typeof text === "string") messages[key] = text;
  }
  return messages;
}

async function tolerant<T>(call: () => Promise<T>): Promise<T | null> {
  try {
    return await call();
  } catch (error) {
    console.debug("[i18n] backend language pack unavailable", error);
    return null;
  }
}

/** Languages the backend can show on the web surface. A language the
 *  dashboard does not bundle renders in English unless some backend layer has
 *  web strings for it, so only those are offered. `null` when the backend did
 *  not answer (older dashboard, offline). */
export async function fetchWebLanguages(): Promise<BackendLanguageOption[] | null> {
  const reply = await tolerant(() => fetchJSON<unknown>("/api/i18n/languages"));
  if (reply === null) return null;
  return languageRows(reply).filter(
    (row) => Array.isArray(row.surfaces) && row.surfaces.includes(WEB_SURFACE),
  );
}

/** The pack + overlay web layer for `lang` (flat dotted keys); `{}` when the
 *  backend has none or did not answer. */
export async function fetchWebCatalog(lang: string): Promise<FlatMessages> {
  const params = new URLSearchParams({ lang, surface: WEB_SURFACE });
  const reply = await tolerant(() => fetchJSON<unknown>(`/api/i18n/catalog?${params}`));
  return catalogMessages(reply);
}

/** Merge a backend layer over `base`: flat keys are re-nested along the base
 *  catalog's own shape, and a plain string over a function-valued entry
 *  becomes a positional (`{0}`, `{1}`) formatter, so the result keeps every
 *  call signature the components rely on. */
export function applyWebPack(base: Translations, messages: FlatMessages): Translations {
  if (!Object.keys(messages).length) return base;
  const overrides = adaptStringOverrides(base, unflattenMessages(messages, base));
  return mergeTranslations<Translations>(base, overrides as TranslationOverride<Translations>);
}
