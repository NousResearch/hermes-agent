// Shapes of /api/cron/blueprints catalog entries (cron/blueprint_catalog.py blueprint_catalog_entry).

export interface AutomationBlueprintField {
  name: string
  type: 'time' | 'enum' | 'text' | 'weekdays'
  label: string
  default: string | null
  options: string[]
  optional: boolean
  /** When false, options are suggestions — any value is accepted. */
  strict?: boolean
  help: string
}

export interface AutomationBlueprint {
  key: string
  title: string
  description: string
  category: string
  tags: string[]
  fields: AutomationBlueprintField[]
  command: string
  appUrl: string
  /** Absent on backends that predate plugin blueprints. */
  source?: 'builtin' | 'plugin'
  /** Registering plugin's name when source is "plugin" (key is `<plugin>:<key>`). */
  plugin?: string | null
}
