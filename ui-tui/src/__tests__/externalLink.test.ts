import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  __resetLinkTitleCache,
  fetchLinkTitle,
  hostPathLabel,
  isTitleFetchable,
  normalizeExternalUrl,
  urlSlugTitleLabel
} from '../lib/externalLink.js'

afterEach(() => {
  __resetLinkTitleCache()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('external link helpers', () => {
  it('formats URL fallbacks as host + path', () => {
    expect(
      hostPathLabel(
        'https://www.getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894/'
      )
    ).toBe('getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894')
  })

  it('derives readable title fallbacks from URL slugs', () => {
    expect(
      urlSlugTitleLabel('https://www.getyourguide.com/fajardo-l882/from-fajardo-icacos-island-full-day-catamaran-trip-t19891/')
    ).toBe('From Fajardo Icacos Island Full Day Catamaran Trip')
  })

  it('normalizes scheme-less links', () => {
    expect(normalizeExternalUrl(' expedia.com/things-to-do/puerto-rico-el-yunque ')).toBe(
      'https://expedia.com/things-to-do/puerto-rico-el-yunque'
    )
  })

  it('filters out local/non-http targets for title fetches', () => {
    expect(isTitleFetchable('https://www.expedia.com/things-to-do/foo')).toBe(true)
    expect(isTitleFetchable('http://localhost:5174')).toBe(false)
    expect(isTitleFetchable('file:///tmp/demo.html')).toBe(false)
    expect(isTitleFetchable('mailto:hello@example.com')).toBe(false)
  })

  it('deduplicates in-flight title fetches and caches results', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('<html><head><title>El Yunque Tour Water Slide, Rope Swing & Pickup</title></head></html>', {
        headers: { 'content-type': 'text/html; charset=utf-8' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    const url = 'https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure.a46272756.activity-details'
    const [first, second] = await Promise.all([fetchLinkTitle(url), fetchLinkTitle(url)])

    expect(first).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(second).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(fetchMock).toHaveBeenCalledTimes(1)

    const third = await fetchLinkTitle(url)

    expect(third).toBe('El Yunque Tour Water Slide, Rope Swing & Pickup')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('shares cache across protocol/www URL variants', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('<html><head><title>Shared Canonical Title</title></head></html>', {
        headers: { 'content-type': 'text/html' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    const first = 'https://www.getyourguide.com/san-juan-puerto-rico-l355/sunset-tours-tc306/'
    const second = 'http://getyourguide.com/san-juan-puerto-rico-l355/sunset-tours-tc306/'

    const [a, b] = await Promise.all([fetchLinkTitle(first), fetchLinkTitle(second)])

    expect(a).toBe('Shared Canonical Title')
    expect(b).toBe('Shared Canonical Title')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('ignores error-like fetched titles', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response('<html><head><title>Just a moment...</title></head></html>', {
        headers: { 'content-type': 'text/html' },
        status: 200
      })
    )

    vi.stubGlobal('fetch', fetchMock)

    const url = 'https://www.getyourguide.com/culebra-island-l145468/from-fajardo-full-day-cordillera-islands-catamaran-tour-t19894/'

    await expect(fetchLinkTitle(url)).resolves.toBe('')
  })

  it('skips network fetch for non-fetchable targets', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchLinkTitle('http://localhost:3000/path')).resolves.toBe('')
    await expect(fetchLinkTitle('mailto:hello@example.com')).resolves.toBe('')
    await expect(fetchLinkTitle('file:///tmp/demo.html')).resolves.toBe('')
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
