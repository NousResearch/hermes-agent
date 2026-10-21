import { readFileSync } from 'node:fs'

import { parse } from 'yaml'

import { applyLocale } from '../i18n/runtime.js'
export const zhMessages = parse(
  readFileSync(new URL('../../../locales/zh.tui.yaml', import.meta.url), 'utf8')
) as Record<string, string>

export function activateZh() {
  applyLocale('zh', { lang: 'zh', surface: 'tui', messages: zhMessages })
}
