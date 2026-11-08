import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import http from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import vm from 'node:vm'

const fromWebsite = createRequire(import.meta.url)

function fromConsumer(name) {
  return createRequire(fromWebsite.resolve(name))
}

test('serve-handler matches redirects with its callable minimatch', async () => {
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

test('Docusaurus image loader reads a PNG through image-size/fromFile', async () => {
  const fromMdxLoader = fromConsumer('@docusaurus/mdx-loader')
  const { imageSizeFromFile } = fromMdxLoader('image-size/fromFile')
  const dir = await mkdtemp(join(tmpdir(), 'hermes-image-size-'))
  try {
    const file = join(dir, 'pixel.png')
    await writeFile(file, Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/RZkAAAAASUVORK5CYII=', 'base64'))
    const dimensions = await imageSizeFromFile(file)
    assert.equal(dimensions.width, 1)
    assert.equal(dimensions.height, 1)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})

test('Docusaurus Joi consumer validates multiple email addresses', () => {
  const joi = fromConsumer('@docusaurus/utils-validation')('joi')
  const schema = joi.string().email({ multiple: true })
  assert.equal(schema.validate('one@example.com, two@example.com').error, undefined)
  assert.ok(schema.validate('one@example.com, invalid').error)
})

test('http-proxy follow-redirects strips cross-realm custom sensitive headers', async () => {
  const redirects = fromConsumer('http-proxy')('follow-redirects')
  const sensitiveHeaders = vm.runInNewContext('(["x-private-test"])')
  const destination = http.createServer((request, response) => {
    response.end(JSON.stringify(request.headers))
  })
  const origin = http.createServer((_request, response) => {
    response.writeHead(302, { Location: `http://localhost:${destination.address().port}/final` })
    response.end()
  })
  await Promise.all([
    new Promise(resolve => destination.listen(0, resolve)),
    new Promise(resolve => origin.listen(0, '127.0.0.1', resolve)),
  ])
  try {
    const headers = await new Promise((resolve, reject) => {
      redirects.http.get({
        hostname: '127.0.0.1', port: origin.address().port, path: '/',
        headers: { 'X-Private-Test': 'synthetic-test-value' }, sensitiveHeaders,
      }, response => {
        let body = ''
        response.setEncoding('utf8')
        response.on('data', chunk => { body += chunk })
        response.on('end', () => resolve(JSON.parse(body)))
      }).on('error', reject)
    })
    assert.equal(headers['x-private-test'], undefined)
  } finally {
    await Promise.all([
      new Promise(resolve => origin.close(resolve)),
      new Promise(resolve => destination.close(resolve)),
    ])
  }
})
