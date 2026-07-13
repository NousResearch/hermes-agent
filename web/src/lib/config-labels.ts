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

function translateIdentifier(
  identifier: string,
  terms?: Record<string, string>,
): string {
  const patterns: Array<[RegExp, (match: RegExpMatchArray) => string]> = [
    [/^max_(.+)$/, (match) => `الحد الأقصى لـ ${translateWords(match[1], terms)}`],
    [/^min_(.+)$/, (match) => `الحد الأدنى لـ ${translateWords(match[1], terms)}`],
    [/^(.+)_enabled$/, (match) => `تفعيل ${translateWords(match[1], terms)}`],
    [/^(.+)_disabled$/, (match) => `تعطيل ${translateWords(match[1], terms)}`],
    [/^(.+)_timeout$/, (match) => `مهلة ${translateWords(match[1], terms)}`],
    [/^(.+)_count$/, (match) => `عدد ${translateWords(match[1], terms)}`],
    [/^(.+)_mode$/, (match) => `وضع ${translateWords(match[1], terms)}`],
    [/^(.+)_path$/, (match) => `مسار ${translateWords(match[1], terms)}`],
    [/^(.+)_url$/, (match) => `رابط ${translateWords(match[1], terms)}`],
    [/^(.+)_interval$/, (match) => `فاصل ${translateWords(match[1], terms)}`],
    [/^(.+)_limit$/, (match) => `حد ${translateWords(match[1], terms)}`],
  ];

  for (const [pattern, render] of patterns) {
    const match = identifier.match(pattern);
    if (match) return render(match);
  }

  return translateWords(identifier, terms);
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
    translateIdentifier(rawLabel, config.fieldTerms)
  );
}

export function configDescription(
  schemaKey: string,
  description: string,
  config: ConfigTranslations,
): string {
  const override = config.descriptionOverrides?.[schemaKey];
  if (override) return override;

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
