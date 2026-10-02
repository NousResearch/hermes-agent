import type { Translations } from "@/i18n/types";

/**
 * Frontend copy lookup for the generic Settings/config form.
 *
 * Today the form's field LABELS are synthesized client-side from the config key
 * (AutoField) and DESCRIPTIONS come from the backend schema — so a field with no
 * server-side translation renders English even in a non-English UI. This module
 * inserts ONE authorable layer in front of both:
 *
 *   label       → `config.fieldCopy[<flatKey>]`
 *   description → `config.fieldCopy[<flatKey> + "_desc"]`
 *
 * where `<flatKey>` is the config schema key verbatim — `model`,
 * `model_context_length`, `delegation.provider`, … (dotted for nested paths).
 * A hit always WINS; a miss falls back to today's behaviour (synthesized label /
 * backend schema prose), so partially translated locales and backend-only keys
 * keep working unchanged.
 *
 * Pure and React-free so the precedence rule is unit-testable on its own.
 */

/** Anything that carries the optional `config.fieldCopy` map (i.e. `t`). */
export interface ConfigCopyCarrier {
  config?: { fieldCopy?: Record<string, string> };
}

/** The authored field-copy map, never null. */
export function configFieldCopy(t: ConfigCopyCarrier | undefined): Record<string, string> {
  const copy = t?.config?.fieldCopy;
  return copy && typeof copy === "object" ? copy : {};
}

function authored(t: ConfigCopyCarrier | undefined, key: string): string | undefined {
  const value = configFieldCopy(t)[key];
  return typeof value === "string" && value.trim().length > 0 ? value : undefined;
}

/** Authored label for a schema key, or undefined when none is translated. */
export function lookupConfigFieldLabel(
  t: ConfigCopyCarrier | undefined,
  schemaKey: string,
): string | undefined {
  return authored(t, schemaKey);
}

/** Authored description for a schema key (the `_desc` entry), or undefined. */
export function lookupConfigFieldDescription(
  t: ConfigCopyCarrier | undefined,
  schemaKey: string,
): string | undefined {
  return authored(t, `${schemaKey}_desc`);
}

/**
 * Today's synthesized English label: last path segment, snake_case → Title Case.
 * Kept byte-identical to the original AutoField expression so untranslated keys
 * render exactly what they rendered before this layer existed.
 */
export function synthesizedConfigFieldLabel(schemaKey: string): string {
  const raw = schemaKey.split(".").pop() ?? schemaKey;
  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Label precedence: authored i18n copy first, synthesized English otherwise. */
export function configFieldLabel(
  t: ConfigCopyCarrier | undefined,
  schemaKey: string,
): string {
  return lookupConfigFieldLabel(t, schemaKey) ?? synthesizedConfigFieldLabel(schemaKey);
}

/** Description precedence: authored i18n copy first, backend schema prose otherwise. */
export function configFieldDescription(
  t: ConfigCopyCarrier | undefined,
  schemaKey: string,
  schemaDescription: unknown,
): string {
  return (
    lookupConfigFieldDescription(t, schemaKey) ??
    (typeof schemaDescription === "string" ? schemaDescription : "")
  );
}

/**
 * Lower-cased haystack the config search filters on: the raw key, the
 * synthesized AND authored labels, the category, and the authored AND backend
 * descriptions. Built here (not in ConfigPage) so the search stays unit-testable
 * and an authored label makes a field findable in the UI language too.
 */
export function configFieldSearchHaystack(
  t: ConfigCopyCarrier | undefined,
  schemaKey: string,
  schema: Record<string, unknown> | null | undefined,
): string {
  const category = typeof schema?.category === "string" ? schema.category : "";
  const backendDescription = typeof schema?.description === "string" ? schema.description : "";
  return [
    schemaKey,
    synthesizedConfigFieldLabel(schemaKey),
    lookupConfigFieldLabel(t, schemaKey) ?? "",
    category,
    lookupConfigFieldDescription(t, schemaKey) ?? "",
    backendDescription,
  ]
    .join(" ")
    .toLowerCase();
}

/** Type guard keeping `Translations` assignable where a carrier is expected. */
export type ConfigCopySource = Pick<Translations, "config">;
