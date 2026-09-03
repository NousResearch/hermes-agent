import { fuzzyScoreMulti } from "@/lib/fuzzy";

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
