import { messages } from '../i18n/runtime.js'
import type { PanelSection } from '../types.js'

export const setupRequiredTitle = (): string => messages().content.setup.title

export const buildSetupRequiredSections = (): PanelSection[] => {
  const s = messages().content.setup

  return [
    {
      text: s.intro
    },
    {
      rows: [
        ['/setup', s.setupRow],
        ['/model', s.modelRow],
        ['Ctrl+C', s.exitRow]
      ],
      title: s.actions
    },
    {
      text: s.footer
    }
  ]
}
