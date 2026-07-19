import type { Translations } from "@/i18n/types";

type ConfigTranslations = Translations["config"];

function titleCase(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

/** Arabic definite article. */
const DEFINITE_ARTICLE = "ال";

/**
 * Attach the Arabic preposition lām to a noun phrase, assimilating a following
 * definite article: `الرموز` → `للرموز`, `إعادات` → `لإعادات`. Written attached,
 * never as a detached `لـ ` with a space after it.
 */
function attachLam(name: string): string {
  if (name.startsWith(DEFINITE_ARTICLE)) {
    const stem = name.slice(DEFINITE_ARTICLE.length);
    // لـ + الـ assimilates to للـ (الرموز → للرموز). When the stem's own first
    // letter is lām, one of the three is dropped in writing
    // (اللقطات → للقطات, never لللقطات).
    return `لل${stem.startsWith("ل") ? stem.slice(1) : stem}`;
  }
  // A foreign word takes the prefix on a tatweel rather than glued to Latin
  // glyphs: لـmb, not لmb.
  if (/^[A-Za-z0-9]/.test(name)) return `لـ${name}`;
  return `ل${name}`;
}

/** The annexed noun of a genitive construct is indefinite. */
function stripArticle(word: string): string {
  return word.startsWith(DEFINITE_ARTICLE) && word.length > DEFINITE_ARTICLE.length
    ? word.slice(DEFINITE_ARTICLE.length)
    : word;
}

function translateWords(
  value: string,
  config?: Pick<
    ConfigTranslations,
    "fieldTerms" | "fieldAdjectives" | "fieldTermOrder"
  >,
  options?: { reorder?: boolean },
): string {
  const terms = config?.fieldTerms;
  // No vocabulary for this locale (English included) — upstream's plain
  // title-cased identifier, unchanged.
  if (!terms) return titleCase(value);

  const spaced = value.replace(/_/g, " ").trim();

  // A catalog may spell a whole compound out idiomatically ("api_call" →
  // "استدعاءات الواجهة البرمجية"); that always beats word-by-word assembly.
  const phrase = terms[spaced.toLowerCase().replace(/\s+/g, "_")];
  if (phrase) return phrase;

  const sourceWords = spaced.split(/\s+/).filter(Boolean);
  const translated = sourceWords.map((word) => {
    const match = word.match(/^([^A-Za-z0-9]*)([A-Za-z0-9]+)([^A-Za-z0-9]*)$/);
    if (!match) return { core: "", text: word };
    const [, prefix, core, suffix] = match;
    return {
      core: core.toLowerCase(),
      text: `${prefix}${terms[core.toLowerCase()] ?? core}${suffix}`,
    };
  });

  // Reordering is only safe where the remainder is reliably a noun phrase —
  // i.e. the operand of a recognised shape (`max_*`, `*_count`, …). A bare
  // identifier can start with a verb ("allow_private_urls") or contain a
  // preposition ("bell_on_complete"), which no word-order rule can rescue, so
  // those keep the source order they had before.
  if (
    !options?.reorder ||
    config?.fieldTermOrder !== "head-initial" ||
    translated.length < 2
  ) {
    return translated.map((w) => w.text).join(" ");
  }

  // English compounds are head-final ("output tokens" = tokens of output);
  // a head-initial language builds them the other way round ("رموز المخرجات").
  const reordered = translated.slice().reverse();
  const adjectives = new Set(config.fieldAdjectives ?? []);
  // A trailing adjective agrees with the noun it now follows and keeps its
  // article ("الجلسات المتزامنة"); a trailing noun makes this a genitive
  // construct, so everything before it drops the article ("رموز المخرجات").
  const isAgreement = adjectives.has(reordered[reordered.length - 1].core);
  return reordered
    .map((word, i) =>
      isAgreement || i === reordered.length - 1
        ? word.text
        : stripArticle(word.text),
    )
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
        const name = translateWords(match[1], config, { reorder: true });
        return template.includes("{lname}")
          ? template.replace("{lname}", attachLam(name))
          : template.replace("{name}", name);
      }
    }
  }

  return translateWords(identifier, config);
}

export function configCategoryLabel(
  category: string,
  config: ConfigTranslations,
): string {
  const categoryKey = category as keyof typeof config.categories;
  return (
    config.categoryLabels?.[category] ??
    config.categories[categoryKey] ??
    translateWords(category, config)
  );
}

export function configSectionLabel(
  section: string,
  config: ConfigTranslations,
): string {
  const explicit = config.sectionLabels?.[section];
  if (explicit) return explicit;
  // A locale with no section vocabulary keeps upstream's raw identifier —
  // the config page rendered `auxiliary`, not `Auxiliary`.
  if (!config.fieldTerms) return section.replace(/_/g, " ");
  return translateWords(section, config);
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

  return translateWords(description, config);
}
