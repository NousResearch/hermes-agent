import { fuzzyRank, fuzzyScoreMulti } from "@/lib/fuzzy";

interface ModelPickerSearchProvider {
  name: string;
  slug: string;
  models?: string[];
}

function providerSearchText(provider: ModelPickerSearchProvider): string {
  return `${provider.name} ${provider.slug}`;
}

function queryTokens(query: string): string[] {
  return query.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

function tokenMatchesAnyModel(models: readonly string[], token: string): boolean {
  return models.some((model) => fuzzyScoreMulti(model, token) != null);
}

/**
 * Filter providers for the model picker without letting model-id matches
 * reshuffle provider identity.
 *
 * Provider name/slug matches are still ranked normally. Model-only matches are
 * included so typing a model id still surfaces providers that offer it, but
 * those rows keep the backend order. This avoids shared model ids (for example
 * `glm-5.2`) jumping to a different provider just because one concatenated
 * search string scored better than another.
 */
export function filterModelPickerProviders<T extends ModelPickerSearchProvider>(
  providers: readonly T[],
  query: string,
): T[] {
  const trimmed = query.trim();

  if (!trimmed) {
    return [...providers];
  }

  const identityMatches = fuzzyRank(providers, trimmed, providerSearchText);
  const identitySlugs = new Set(identityMatches.map((r) => r.item.slug));
  const tokens = queryTokens(trimmed);
  const modelOrMixedMatches = providers.filter((provider) => {
    if (identitySlugs.has(provider.slug)) {
      return false;
    }

    const identityText = providerSearchText(provider);
    const models = provider.models ?? [];

    return tokens.every(
      (token) =>
        fuzzyScoreMulti(identityText, token) ||
        tokenMatchesAnyModel(models, token),
    );
  });

  return [...identityMatches.map((r) => r.item), ...modelOrMixedMatches];
}

/**
 * Derive the query used by the model column for the selected provider.
 *
 * If the query matches only the selected provider's identity, return an empty
 * query so that provider's model list stays visible. For mixed provider+model
 * searches like `nvidia glm-5.2`, strip the selected provider's identity terms
 * before filtering models so the model column still shows `glm-5.2` instead of
 * going empty on the provider token.
 */
export function modelQueryForSelectedProvider(
  selectedProvider: ModelPickerSearchProvider | null,
  models: readonly string[],
  query: string,
): string {
  const trimmed = query.trim();

  if (!trimmed || !selectedProvider) {
    return trimmed;
  }

  const tokens = queryTokens(trimmed);

  if (!tokens.length || tokens.every((token) => tokenMatchesAnyModel(models, token))) {
    return trimmed;
  }

  const identityText = providerSearchText(selectedProvider);
  const modelTokens = tokens.filter((token) => {
    if (tokenMatchesAnyModel(models, token)) {
      return true;
    }

    return fuzzyScoreMulti(identityText, token) == null;
  });

  return modelTokens.join(" ");
}

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
  if (!trimmedQuery) return false;

  return modelQueryForSelectedProvider(selectedProvider, models, trimmedQuery) === "";
}
