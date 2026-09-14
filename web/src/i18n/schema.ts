import type { SchemaTranslations } from "./types";

function lookup(
  table: Record<string, string>,
  key: string,
): string | undefined {
  return Object.hasOwn(table, key) ? table[key] : undefined;
}

function joinTranslatedTerms(terms: string[]): string {
  return terms.reduce((label, term) => {
    if (!label) return term;

    // Keep a boundary around technical identifiers while allowing natural
    // compounds for writing systems that do not normally use spaces.
    const previousIsTechnical = /[A-Za-z0-9]$/.test(label);
    const currentIsTechnical = /^[A-Za-z0-9]/.test(term);
    return `${label}${previousIsTechnical || currentIsTechnical ? " " : ""}${term}`;
  }, "");
}

function generatedSchemaLabel(
  translations: SchemaTranslations,
  schemaKey: string,
): string {
  return schemaKey
    .split(".")
    .map(
      (segment) =>
        lookup(translations.segments, segment) ??
        joinTranslatedTerms(
          segment
            .split("_")
            .map((term) => lookup(translations.terms, term) ?? term),
        ),
    )
    .join(translations.pathSeparator);
}

/** Resolve schema wording from the active locale pack with an English fallback. */
export function resolveSchemaLabel(
  translations: SchemaTranslations,
  schemaKey: string,
  fallback: string,
): string {
  return (
    lookup(translations.labels, schemaKey) ??
    (translations.generateLabels
      ? generatedSchemaLabel(translations, schemaKey)
      : fallback)
  );
}

/** Resolve the final path segment for nested object and array editors. */
export function resolveSchemaLeafLabel(
  translations: SchemaTranslations,
  schemaKey: string,
  fallback: string,
): string {
  return (
    resolveSchemaLabel(translations, schemaKey, fallback)
      .split(translations.pathSeparator)
      .at(-1) ?? fallback
  );
}

/** Prefer locale-owned descriptions and fall back to backend English copy. */
export function resolveSchemaDescription(
  translations: SchemaTranslations,
  schemaKey: string,
  fallback: string,
): string {
  return lookup(translations.descriptions, schemaKey) ?? fallback;
}
