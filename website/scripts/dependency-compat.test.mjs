import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { test } from 'node:test'

const fromWebsite = createRequire(import.meta.url)

test('serve-handler can check a nonmatching redirect before reading a file', async () => {
  const handler = fromWebsite('serve-handler')
  let lstatCalls = 0
  let statusCode

  await handler(
    { url: '/docs/page', headers: { accept: 'application/json' } },
    {},
    { redirects: [{ source: '/other/**', destination: '/new' }] },
    {
      lstat: async () => {
        lstatCalls += 1
        throw Object.assign(new Error('missing'), { code: 'ENOENT' })
      },
      sendError: (_path, _response, _json, _current, _handlers, _config, spec) => {
        statusCode = spec.statusCode
      },
    },
  )

  assert.ok(lstatCalls > 0)
  assert.equal(statusCode, 404)
})

test('Docusaurus image loader can use its patched fromFile API', () => {
  const fromMdxLoader = createRequire(fromWebsite.resolve('@docusaurus/mdx-loader'))
  assert.equal(typeof fromMdxLoader('image-size/fromFile').imageSizeFromFile, 'function')
})
