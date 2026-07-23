import { describe, expect, it } from 'vitest'

import { resolveSchemaDescription, resolveSchemaLabel, resolveSchemaLeafLabel } from './schema'
import { resolveTranslationOverlay, resolveTranslations } from './runtime'

const zhSchema = resolveTranslations('zh').schema

describe('schema localization', () => {
  it('keeps curated wording for known configuration fields', () => {
    expect(resolveSchemaLabel(zhSchema, 'display.language', 'Language')).toBe('显示 → 语言')
  })

  it('builds a readable label for fields added after the curated catalog', () => {
    expect(resolveSchemaLabel(zhSchema, 'gateway.restart_loop_guard.max_restarts', 'Max Restarts')).toBe(
      '网关 → 重启循环保护 → 最大重启次数'
    )
  })

  it('preserves unknown technical terms instead of dropping path context', () => {
    expect(resolveSchemaLabel(zhSchema, 'vertex.project_id', 'Project ID')).toBe('Vertex → 项目 ID')
  })

  it('keeps technical identifiers separated from surrounding Chinese words', () => {
    expect(resolveSchemaLabel(zhSchema, 'gateway.api_server.max_concurrent_runs', 'Max Concurrent Runs')).toBe(
      '网关 → API 服务器 → 最大并发运行次数'
    )
  })

  it('provides contextual descriptions only for informative schema overrides', () => {
    expect(resolveSchemaDescription(zhSchema, 'display.busy_input_mode', 'English fallback')).toBe(
      'Agent 运行时收到新输入后的处理方式。'
    )
    expect(resolveSchemaDescription(zhSchema, 'agent.max_turns', 'Maximum turns')).toBe('Maximum turns')
  })

  it('localizes the Codex commentary setting added by the runtime', () => {
    expect(resolveSchemaLabel(zhSchema, 'display.show_commentary', 'Show Commentary')).toBe('显示 → 显示进度解说')
    expect(
      resolveSchemaDescription(zhSchema, 'display.show_commentary', 'Show commentary emitted by Codex models.')
    ).toContain('Codex 模型')
  })

  it('covers the latest gateway delivery and xAI TTS schema fields', () => {
    const fields = {
      'gateway.delivery_ledger': '网关 → 投递账本',
      'tts.xai.auto_speech_tags': '语音合成 → xAI → 自动语音标签',
      'tts.xai.optimize_streaming_latency': '语音合成 → xAI → 流式延迟优化',
      'tts.xai.speed': '语音合成 → xAI → 播放速度'
    }

    for (const [key, expected] of Object.entries(fields)) {
      expect(resolveSchemaLabel(zhSchema, key, 'English fallback')).toBe(expected)
    }
  })

  it('covers configuration fields introduced by the latest upstream release', () => {
    const fields = {
      'mcp.auto_reload_on_config_change': 'MCP → 配置变更时自动重载',
      'tts.minimax.model': '语音合成 → MiniMax → 默认模型',
      'tts.minimax.voice_id': '语音合成 → MiniMax → 语音 ID',
      'tts.kittentts.model': '语音合成 → KittenTTS → 默认模型',
      'tts.kittentts.voice': '语音合成 → KittenTTS → 语音',
      'x_search.reasoning_effort': 'X 搜索 → 推理强度'
    }

    for (const [key, expected] of Object.entries(fields)) {
      expect(resolveSchemaLabel(zhSchema, key, 'English fallback')).toBe(expected)
    }
  })

  it('covers schema fields added by browser and Discord reconnect recovery', () => {
    const fields = {
      'browser.headed': '浏览器 → 显示浏览器窗口',
      'discord.missed_message_backfill.enabled': 'Discord → 漏消息补拉 → 启用',
      'discord.missed_message_backfill.channels': 'Discord → 漏消息补拉 → 频道',
      'discord.missed_message_backfill.window_seconds': 'Discord → 漏消息补拉 → 回溯时间窗口(秒)',
      'discord.missed_message_backfill.limit': 'Discord → 漏消息补拉 → 扫描上限',
      'discord.missed_message_backfill.max_dispatches': 'Discord → 漏消息补拉 → 最大派发数'
    }

    for (const [key, expected] of Object.entries(fields)) {
      expect(resolveSchemaLabel(zhSchema, key, 'English fallback')).toBe(expected)
      expect(resolveSchemaDescription(zhSchema, key, 'English fallback')).not.toBe('English fallback')
    }
  })

  it('provides concise labels for nested editors', () => {
    expect(resolveSchemaLeafLabel(zhSchema, 'gateway.api_server.max_concurrent_runs', 'max_concurrent_runs')).toBe(
      '最大并发运行次数'
    )
  })

  it('keeps English as the fallback for any locale pack without schema wording', () => {
    const schema = resolveTranslationOverlay({}).schema

    expect(resolveSchemaLabel(schema, 'display.language', 'Language')).toBe('Language')
    expect(resolveSchemaDescription(schema, 'display.language', 'Display language')).toBe('Display language')
  })
})
