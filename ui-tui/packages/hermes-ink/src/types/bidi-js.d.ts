/**
 * Ambient declaration for bidi-js (ships untyped JS under dist/).
 *
 * Must be a script-style .d.ts (no top-level imports/exports): an ambient
 * `declare module` inside a module file is treated as an augmentation, which
 * cannot create types for a module that has none.
 *
 * Only the two members hermes-ink actually calls are declared; everything
 * else stays unexported rather than guessed.
 */
declare module 'bidi-js' {
  export interface BidiEmbeddingLevels {
    readonly levels: readonly number[]
    readonly paragraphs: readonly {
      readonly level: number
      readonly start: number
      readonly end: number
    }[]
  }

  export interface BidiInstance {
    getEmbeddingLevels(
      text: string,
      direction: 'auto' | 'ltr' | 'rtl',
    ): BidiEmbeddingLevels
    getReorderSegments(
      text: string,
      embeddingLevels: BidiEmbeddingLevels,
      start?: number,
      end?: number,
    ): readonly { readonly start: number; readonly end: number }[]
  }

  export default function bidiFactory(): BidiInstance
}
