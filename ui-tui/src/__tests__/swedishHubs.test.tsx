import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

const input = vi.hoisted(() => ({
  handler: undefined as undefined | ((text: string, key: Record<string, boolean>) => void)
}))

vi.mock('@hermes/ink', async importOriginal => ({
  ...(await importOriginal()),
  useInput: (handler: (text: string, key: Record<string, boolean>) => void) => {
    input.handler = handler
  }
}))

import { PetPicker } from '../components/petPicker.js'
import { PluginsHub } from '../components/pluginsHub.js'
import { SkillsHub } from '../components/skillsHub.js'
import { TodoPanel } from '../components/todoPanel.js'
import type { GatewayClient } from '../gatewayClient.js'
import { setTuiLanguage } from '../i18n/index.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

function mount(element: React.ReactNode) {
  const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 40 })
  const stdin = Object.assign(new PassThrough(), { isTTY: false })
  const stderr = Object.assign(new PassThrough(), { isTTY: false })
  let output = ''
  stdout.on('data', chunk => {
    output += chunk.toString()
  })
  setTuiLanguage('sv')

  const instance = renderSync(element, {
    patchConsole: false,
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: stderr as unknown as NodeJS.WriteStream
  })

  return {
    text: () => stripAnsi(output),
    press: (text = '', key: Record<string, boolean> = {}) => {
      input.handler?.(text, key)
      instance.rerender(element)
    },
    close: () => {
      instance.unmount()
      instance.cleanup()
      setTuiLanguage('en')
    }
  }
}

it('adopts the original pet slug from a Swedish gallery', async () => {
  const request = vi.fn(async () => ({
    enabled: true,
    active: '',
    pets: [{ slug: 'fixture-pet', displayName: 'Original Pet', installed: false, curated: true }]
  }))

  const view = mount(<PetPicker gw={{ request } as unknown as GatewayClient} onClose={() => {}} t={DEFAULT_THEME} />)

  try {
    await vi.waitFor(() => expect(view.text()).toContain('1 husdjur'))
    expect(view.text()).toContain('Original Pet')
    expect(view.text()).toContain('officiellt')
    view.press('', { return: true })
    expect(request).toHaveBeenLastCalledWith('pet.select', { slug: 'fixture-pet' })
  } finally {
    view.close()
  }
})

it('toggles a plugin without translating its name or boolean action', async () => {
  const plugin = { name: 'fixture-plugin', status: 'disabled', source: 'user' }
  const request = vi.fn(async () => ({ plugins: [plugin], user_count: 1, bundled_count: 0, plugin }))
  const view = mount(<PluginsHub gw={{ request } as unknown as GatewayClient} onClose={() => {}} t={DEFAULT_THEME} />)

  try {
    await vi.waitFor(() => expect(view.text()).toContain('1 egen plugin'))
    expect(view.text()).toContain('inaktiverad')
    view.press('', { return: true })
    expect(request).toHaveBeenLastCalledWith('plugins.manage', {
      action: 'toggle',
      enable: true,
      name: 'fixture-plugin'
    })
  } finally {
    view.close()
  }
})

it('inspects and reinstalls the original skill from Swedish action hints', async () => {
  const request = vi.fn(async () => ({
    skills: { 'Original Category': ['fixture-skill'] },
    info: { name: 'fixture-skill', description: 'Original description', path: '/fixture/SKILL.md' }
  }))
  const view = mount(<SkillsHub gw={{ request } as unknown as GatewayClient} onClose={() => {}} t={DEFAULT_THEME} />)

  try {
    await vi.waitFor(() => expect(view.text()).toContain('välj en kategori'))
    expect(view.text()).toContain('Original Category')
    view.press('', { return: true })
    await vi.waitFor(() => expect(view.text()).toContain('1 färdighet'))
    view.press('', { return: true })
    await vi.waitFor(() => expect(view.text()).toContain('Original description'))
    expect(view.text()).toContain('sökväg: /fixture/SKILL.md')
    expect(request).toHaveBeenLastCalledWith('skills.manage', { action: 'inspect', query: 'fixture-skill' })
    view.press('x')
    expect(request).toHaveBeenLastCalledWith('skills.manage', { action: 'install', query: 'fixture-skill' })
  } finally {
    view.close()
  }
})

it('renders Swedish remaining-work copy without translating user todos', async () => {
  const view = mount(
    <TodoPanel incomplete t={DEFAULT_THEME} todos={[{ id: 'fixture', content: 'Original todo', status: 'pending' }]} />
  )

  try {
    await vi.waitFor(() => expect(view.text()).toContain('Att göra'))
    expect(view.text()).toContain('1 återstår')
    expect(view.text()).toContain('Original todo')
  } finally {
    view.close()
  }
})
