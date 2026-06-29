import type { DeepPartial, TuiTranslations } from './types.js'

// Japanese overrides, merged on top of `en`. Keys omitted here fall back to the
// English string, so this can stay partial as the catalog grows.
export const ja: DeepPartial<TuiTranslations> = {
  branding: {
    gateway: {
      disabled: '無効',
      connecting: '接続中',
      configured: '設定済み',
      failed: '失敗'
    },
    noSystemPrompt: 'システムプロンプトは読み込まれていません。',
    sessionLabel: 'セッション: '
  }
}
