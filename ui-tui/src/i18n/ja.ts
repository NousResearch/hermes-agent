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
  },
  skills: {
    loading: 'スキルを読み込み中…',
    none: '利用可能なスキルがありません',
    selectCategory: 'カテゴリーを選択',
    noneInCategory: 'このカテゴリーにスキルはありません',
    loadingShort: '読み込み中…',
    installing: 'インストール中…'
  },
  plugins: {
    loading: 'プラグインを読み込み中…',
    none: 'プラグインがインストールされていません',
    installHint: 'インストール: hermes plugins install owner/repo',
    updating: '更新中…'
  },
  sessions: {
    loading: 'セッションを読み込み中…',
    noOther: '他のセッションはありません — +new で新規開始'
  },
  pets: {
    loading: 'ペットを読み込み中…',
    adopting: '迎え入れ中…'
  },
  models: {
    loading: 'モデルを読み込み中…',
    noProviders: '利用可能なプロバイダーがありません'
  },
  agents: {
    compareHint: 'ベースライン vs 候補 · esc/q で閉じる',
    noSubagents: 'このターンにサブエージェントはありません。delegate_task を実行するとツリーに表示されます。'
  }
}
