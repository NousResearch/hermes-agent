/** Localized metadata for config-schema fields.
 *
 * Schema keys are supplied by the backend and can grow independently of the
 * Dashboard. Locale packs therefore overlay keyed wording and may opt into a
 * generic glossary-based label generator without adding locale branches to
 * form components.
 */
export interface SchemaTranslations {
  descriptions: Record<string, string>
  generateLabels: boolean
  labels: Record<string, string>
  pathSeparator: string
  segments: Record<string, string>
  terms: Record<string, string>
}
