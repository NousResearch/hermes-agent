// Language identity and normalization are shared with Dashboard and Python
// through locales/registry.json. Each UI still owns its presentation pack.
export { type Locale, LOCALES } from '@hermes/shared/locale-registry'

export interface LangPack {
  toolVerbs: Record<string, string>
  verbs: string[]
  status: Record<string, string>
  /** UI string catalog — key set is authoritative from the EN pack. */
  catalog: Record<string, string>
  trail: { draftPrefix: string; analyzeLabel: string }
  /** How the status bar renders thinking verbs.
   *  'pad' — pad to a fixed width (Latin languages)
   *  'ellipsis' — append '…' (CJK languages) */
  verbStyle: 'ellipsis' | 'pad'
}

/** A locale may translate any subset; omitted values inherit from English. */
export interface LangPackOverlay<CatalogKey extends string = string> {
  catalog?: Partial<Record<CatalogKey, string>>
  status?: Record<string, string>
  toolVerbs?: Record<string, string>
  trail?: Partial<LangPack['trail']>
  verbs?: string[]
  verbStyle?: LangPack['verbStyle']
}
