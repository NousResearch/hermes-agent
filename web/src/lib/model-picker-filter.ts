import { fuzzyScoreMulti } from "@hermes/shared";

/**
 * True when `trimmedQuery` located the selected provider by name/slug but
 * matches none of its models by id — the case where a single search box
 * filtering both the provider and model columns would otherwise leave the
 * model pane empty even though the user just successfully found the
 * provider they were looking for.
 */
export function queryMatchesProviderOnly(
  selectedProvider: { name: string; slug: string } | null,
  models: readonly string[],
  trimmedQuery: string,
): boolean {
  if (!trimmedQuery || !selectedProvider) return false;

  const matchesProvider =
    fuzzyScoreMulti(`${selectedProvider.name} ${selectedProvider.slug}`, trimmedQuery) !=
    null;
  const matchesAnyModel = models.some((m) => fuzzyScoreMulti(m, trimmedQuery) != null);

  return matchesProvider && !matchesAnyModel;
}

/** `vendor/model` or `vendor/model:suffix` where `suffix` may carry a slash
 *  (`z-ai/glm-5.3-flash:wafer`, `...:deepinfra/fp4`). */
const MODEL_ID_SHAPE_RE = /^\w[\w.-]*\/\w[\w.-]*(?::[\w./-]+)?$/;

/**
 * The model id to offer as a verbatim "not in catalog" entry, or null.
 * OpenRouter provider-pin ids (`z-ai/glm-5.3-flash:wafer`) and ids the cached
 * catalog has simply not caught up with are wire-valid but never listed in
 * `/models` — a typed id that LOOKS like a model id and matches nothing gets a
 * verbatim row instead of a dead end. OpenRouter-only: other providers have no
 * suffix routing, and offering unlisted ids there would just 400 at the wire.
 *
 * `trimmedQuery` arrives pre-trimmed from the picker. This regex is the
 * stricter TS twin of the backend's permissive
 * `hermes_constants.openrouter_slug_parts`: the picker only offers ids a human
 * plausibly typed as a model id, while the Python validator stays permissive
 * about what a suffix may contain.
 */
export function verbatimModelId(
  selectedProvider: { slug: string; models?: readonly string[] } | null,
  trimmedQuery: string,
): string | null {
  const q = trimmedQuery;
  if (!q || !MODEL_ID_SHAPE_RE.test(q)) return null;
  if (!selectedProvider || selectedProvider.slug !== "openrouter") return null;
  const listed = selectedProvider.models ?? [];
  if (listed.some((m) => m.toLowerCase() === q.toLowerCase())) return null;
  return q;
}
