import { pick } from '../lib/text.js'

export const PLACEHOLDERS = [
  '请随意提问…',
  '试试"解释这份代码"',
  '试试"编写测试…"',
  '试试"重构认证模块"',
  '试试"/help"查看命令',
  '试试"修复 lint 错误"',
  '试试"配置加载器是如何工作的？"'
]

export const PLACEHOLDER = pick(PLACEHOLDERS)
