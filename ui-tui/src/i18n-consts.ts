export type Locale = 'en' | 'zh'

/** Locale-specific transient trail patterns shared by TS-only modules. */
export const TRAIL_PATTERNS: Record<Locale, { draftPrefix: string; analyzeLabel: string }> = {
  en: { draftPrefix: 'drafting ', analyzeLabel: 'analyzing tool output…' },
  zh: { draftPrefix: '正在生成 ', analyzeLabel: '正在分析工具输出…' }
}
