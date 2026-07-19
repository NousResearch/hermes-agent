import type { Translations } from "@/i18n/types";

type ConfigTranslations = Translations["config"];

function titleCase(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function translateWords(value: string, terms?: Record<string, string>): string {
  if (!terms) return titleCase(value);

  return value
    .replace(/_/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => {
      const match = word.match(
        /^([^A-Za-z0-9]*)([A-Za-z0-9]+)([^A-Za-z0-9]*)$/,
      );
      if (!match) return word;
      const [, prefix, core, suffix] = match;
      return `${prefix}${terms[core.toLowerCase()] ?? core}${suffix}`;
    })
    .join(" ");
}

// Shape-based fallbacks for schema keys with no explicit label. The wording
// lives in the locale catalogs — a locale that omits `fieldPatterns` (English
// does) keeps upstream's plain title-cased identifier.
const IDENTIFIER_SHAPES: Array<[RegExp, string]> = [
  [/^max_(.+)$/, "max"],
  [/^min_(.+)$/, "min"],
  [/^(.+)_enabled$/, "enabled"],
  [/^(.+)_disabled$/, "disabled"],
  [/^(.+)_timeout$/, "timeout"],
  [/^(.+)_count$/, "count"],
  [/^(.+)_mode$/, "mode"],
  [/^(.+)_path$/, "path"],
  [/^(.+)_url$/, "url"],
  [/^(.+)_interval$/, "interval"],
  [/^(.+)_limit$/, "limit"],
];

function translateIdentifier(
  identifier: string,
  config: ConfigTranslations,
): string {
  const patterns = config.fieldPatterns;

  if (patterns) {
    for (const [shape, key] of IDENTIFIER_SHAPES) {
      const match = identifier.match(shape);
      const template = match && patterns[key];
      if (match && template) {
        return template.replace(
          "{name}",
          translateWords(match[1], config.fieldTerms),
        );
      }
    }
  }

  return translateWords(identifier, config.fieldTerms);
}

export function configCategoryLabel(
  category: string,
  config: ConfigTranslations,
): string {
  const categoryKey = category as keyof typeof config.categories;
  return (
    config.categoryLabels?.[category] ??
    config.categories[categoryKey] ??
    translateWords(category, config.fieldTerms)
  );
}

export function configSectionLabel(
  section: string,
  config: ConfigTranslations,
): string {
  return (
    config.sectionLabels?.[section] ??
    translateWords(section, config.fieldTerms)
  );
}

export function configFieldLabel(
  schemaKey: string,
  config: ConfigTranslations,
): string {
  const rawLabel = schemaKey.split(".").pop() ?? schemaKey;
  return (
    config.fieldLabels?.[schemaKey] ??
    config.fieldLeafLabels?.[rawLabel] ??
    translateIdentifier(rawLabel, config)
  );
}

export function configDescription(
  schemaKey: string,
  description: string,
  config: ConfigTranslations,
): string {
  const override = config.descriptionOverrides?.[schemaKey];
  if (override) return override;

  // No term vocabulary means this locale has nothing to say about the schema's
  // own prose — hand back the server's English description untouched, exactly
  // as upstream rendered it (including the "Section → Field" arrow direction).
  if (!config.fieldTerms) return description;

  if (description.includes("→")) {
    const parts = description.split("→").map((part) => part.trim());
    return [
      ...parts
        .slice(0, -1)
        .map((part) =>
          configSectionLabel(
            part.toLowerCase().replace(/[^a-z0-9]+/g, "_"),
            config,
          ),
        ),
      configFieldLabel(schemaKey, config),
    ].join(" ← ");
  }

  const rawLabel = schemaKey.split(".").pop() ?? schemaKey;
  const normalizedDescription = description
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");
  if (normalizedDescription === rawLabel.toLowerCase()) {
    return configFieldLabel(schemaKey, config);
  }

  return translateWords(description, config.fieldTerms);
}
