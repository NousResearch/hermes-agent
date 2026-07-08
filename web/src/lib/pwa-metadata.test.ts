import { existsSync, readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

const webRoot = new URL('../..', import.meta.url)
const indexHtml = () => readFileSync(new URL('index.html', webRoot), 'utf8')
const manifestPath = new URL('public/manifest.webmanifest', webRoot)

describe('dashboard web-app metadata', () => {
  it('exposes installable web-app metadata from index.html', () => {
    const html = indexHtml()

    expect(html).toContain('rel="manifest"')
    expect(html).toContain('href="/manifest.webmanifest"')
    expect(html).toContain('name="theme-color"')
    expect(html).toContain('name="mobile-web-app-capable"')
    expect(html).toContain('name="apple-mobile-web-app-capable"')
    expect(html).toContain('name="apple-mobile-web-app-title"')
  })

  it('ships a manifest with phone and desktop standalone app settings', () => {
    expect(existsSync(manifestPath)).toBe(true)

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
      name?: string
      short_name?: string
      start_url?: string
      scope?: string
      display?: string
      theme_color?: string
      background_color?: string
      icons?: Array<{ src?: string; sizes?: string; type?: string; purpose?: string }>
    }

    expect(manifest.name).toBe('Hermes Agent')
    expect(manifest.short_name).toBe('Hermes')
    expect(manifest.start_url).toBe('.')
    expect(manifest.scope).toBe('.')
    expect(manifest.display).toBe('standalone')
    expect(manifest.theme_color).toMatch(/^#[0-9a-f]{6}$/i)
    expect(manifest.background_color).toMatch(/^#[0-9a-f]{6}$/i)

    const icons = manifest.icons ?? []
    for (const size of ['192x192', '512x512']) {
      const icon = icons.find(item => item.sizes === size && item.type === 'image/png')

      expect(icon).toBeTruthy()
      expect(icon?.purpose).toContain('any')
      expect(icon?.purpose).toContain('maskable')
      expect(existsSync(new URL(`public/${icon?.src?.replace(/^\//, '')}`, webRoot))).toBe(true)
    }
  })
})
