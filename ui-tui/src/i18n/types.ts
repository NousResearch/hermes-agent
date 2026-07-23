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

// ── Shared glossary ────────────────────────────────────────────
// Terms that appear in multiple modules and MUST use the same
// translation everywhere.  When a translator encounters one of
// these in any catalog, the canonical rendering is authoritative.
//
// This is a **type-level** register — no runtime code depends on
// it, but it flags the terms so contributors and CI can verify
// consistency across TUI, Dashboard, CLI, and future modules.
//
//   Canonical translations by locale (add rows as languages mature):
//     zh: provider→服务商  clarify→询问  secret→敏感信息  steer→引导
//         persist→保存    spawn→生成    in-place→直接    commit→个提交
export type GlossaryTerm = 'provider' | 'clarify' | 'secret' | 'steer' | 'persist' | 'spawn' | 'in-place' | 'commit'
