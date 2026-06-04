import type { PanelSection } from '../types.js'

export const SETUP_REQUIRED_TITLE = '需要设置'

export const buildSetupRequiredSections = (): PanelSection[] => [
  {
    text: 'Hermes 需要一个模型提供商，TUI 才能启动会话。'
  },
  {
    rows: [
      ['/model', '就地配置提供商和模型'],
      ['/setup', '就地运行完整首次设置向导'],
      ['Ctrl+C', '退出并手动运行 `hermes setup`']
    ],
    title: '操作'
  }
]
