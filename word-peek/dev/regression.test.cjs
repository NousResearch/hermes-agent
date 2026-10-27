const fs = require('node:fs')
const path = require('node:path')
const assert = require('node:assert/strict')
const { pathToFileURL } = require('node:url')
const { dependency, nodePaths } = require('./runtime.cjs')
const { build } = dependency('esbuild')
const { chromium } = dependency('playwright')
const fixture = path.join(__dirname, 'regression-fixture.html').replaceAll('\\', '/')
const fixtureUrl = pathToFileURL(fixture).href
const original = fs.readFileSync(fixture, 'utf8')
const createdBackups = new Set()

;(async () => {
  const bundle = await build({
    stdin: { contents: "export { capturePassage } from './html-edit.js'; export { configureReadBuild, openReadBuild } from './panel.js';", resolveDir: __dirname },
    bundle: true, write: false, format: 'iife', globalName: 'regression', nodePaths
  })
  const source = bundle.outputFiles[0].text
  const browser = await chromium.launch({ headless: true, channel: 'msedge' })
  let context
  let page
  let guest
  const boot = async () => {
    if (context) await context.close()
    fs.writeFileSync(fixture, original, 'utf8')
    context = await browser.newContext()
    page = await context.newPage()
    guest = await context.newPage()
    await page.goto(fixtureUrl)
    await page.setContent('<body><webview></webview></body>')
    await page.exposeFunction('fixtureRead', file => {
      if (file !== fixture && !createdBackups.has(file)) throw new Error('Unexpected test read: ' + file)
      return { path: file, text: fs.readFileSync(file, 'utf8'), truncated: false }
    })
    await page.exposeFunction('fixtureWrite', (file, text) => {
      if (file !== fixture) {
        if (!file.startsWith(fixture + '.word-peek-') || !file.endsWith('.bak')) throw new Error('Unexpected test write: ' + file)
        if (fs.existsSync(file)) throw new Error('Refusing to overwrite an existing backup in this test')
        createdBackups.add(file)
      }
      fs.writeFileSync(file, text, 'utf8')
      return { path: file }
    })
    await page.addScriptTag({ content: source })
    await page.evaluate(({ fixtureUrl }) => {
      window.saved = {}
      window.notices = []
      window.reloadCount = 0
      window.storageBlocked = false
      const storage = {
        get: (key, fallback) => window.saved[key] ?? fallback,
        set: (key, value) => { if (!window.storageBlocked) window.saved[key] = structuredClone(value) },
        remove: key => { delete window.saved[key] }
      }
      window.hermesDesktop = { readPluginSource: file => window.fixtureRead(file), writeTextFile: (file, text) => window.fixtureWrite(file, text) }
      const guest = document.querySelector('webview')
      guest.setAttribute('src', fixtureUrl)
      guest.getURL = () => fixtureUrl
      guest.reload = () => { window.reloadCount++ }
      window.dispose = regression.configureReadBuild({ notify: notice => window.notices.push(notice) }, { storage })
    }, { fixtureUrl })
  }
  const reopen = async selector => {
    await guest.goto(fixtureUrl)
    await guest.addScriptTag({ content: source })
    const passage = await guest.evaluate(selector => {
      const text = document.querySelector(selector).firstChild
      const range = document.createRange()
      range.setStart(text, 0)
      range.setEnd(text, 7)
      getSelection().removeAllRanges()
      getSelection().addRange(range)
      return { ...regression.capturePassage(window), sessionId: 'regression-session' }
    }, selector)
    await page.evaluate(passage => regression.openReadBuild(passage, document.querySelector('webview')), passage)
    return passage
  }
  const previewNote = async text => {
    await page.getByLabel('Question or note').fill(text)
    await page.getByRole('button', { name: 'Use as my note', exact: true }).click()
    await page.getByRole('button', { name: 'Preview in HTML', exact: true }).last().click()
    await page.waitForSelector('.wb-preview details')
  }
  const apply = async expectedReloads => {
    await page.getByRole('button', { name: 'Apply to HTML', exact: true }).click()
    await page.waitForFunction(expected => window.reloadCount === expected, expectedReloads)
  }
  try {
    // Real SDK storage writes can silently fail: a failed pending record must block the original write.
    await boot()
    await reopen('#list-passage')
    await previewNote('This note must not be saved when Undo storage is unavailable.')
    await page.evaluate(() => { window.storageBlocked = true })
    await page.getByRole('button', { name: 'Apply to HTML', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'no room to save Undo' }).waitFor()
    assert.equal(fs.readFileSync(fixture, 'utf8'), original)
    assert.equal(await page.evaluate(() => window.reloadCount), 0)
    assert.equal(await page.evaluate(() => window.saved['html.pending.v1']), undefined)
    console.log('PASS: silent Undo storage failure leaves original file unchanged')

    // The selection key stays the same after an inside-LI note, but its captured blockText changes.
    await boot()
    const firstList = await reopen('#list-passage')
    await previewNote('First list explanation.')
    await apply(1)
    const secondList = await reopen('#list-passage')
    assert.equal(firstList.selectedText, secondList.selectedText)
    assert.deepEqual(firstList.path, secondList.path)
    assert.notEqual(firstList.blockText, secondList.blockText)
    await previewNote('Second list explanation.')
    await apply(2)
    const listResult = await page.evaluate(html => {
      const doc = new DOMParser().parseFromString(html, 'text/html')
      return { notes: doc.querySelectorAll('#list-passage > details').length, items: doc.querySelectorAll('ul > li').length }
    }, fs.readFileSync(fixture, 'utf8'))
    assert.deepEqual(listResult, { notes: 2, items: 2 })
    console.log('PASS: reselecting the same LI passage permits a second insertion')

    await boot()
    await reopen('#cell-passage')
    await previewNote('First table explanation.')
    await apply(1)
    await reopen('#cell-passage')
    await previewNote('Second table explanation.')
    await apply(2)
    const cellResult = await page.evaluate(html => {
      const doc = new DOMParser().parseFromString(html, 'text/html')
      return { notes: doc.querySelectorAll('#cell-passage > details').length, cells: doc.querySelectorAll('tr > td').length }
    }, fs.readFileSync(fixture, 'utf8'))
    assert.deepEqual(cellResult, { notes: 2, cells: 2 })
    console.log('PASS: reselecting the same TD passage permits a second insertion')

    // A failed next save must leave the previous successful save undoable.
    await boot()
    await reopen('#list-passage')
    await previewNote('Successfully saved first note.')
    await apply(1)
    const savedOnce = fs.readFileSync(fixture, 'utf8')
    const priorUndo = await page.evaluate(() => window.saved['html.undo.v1'])
    assert(priorUndo.beforeHash && priorUndo.afterHash && priorUndo.backup)
    assert.equal(priorUndo.before, undefined)
    assert.equal(priorUndo.after, undefined)
    assert(JSON.stringify(priorUndo).length < 1000)
    await reopen('#list-passage')
    await previewNote('Uncommitted second note.')
    await page.evaluate(() => { window.storageBlocked = true })
    await page.getByRole('button', { name: 'Apply to HTML', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'no room to save Undo' }).waitFor()
    assert.equal(fs.readFileSync(fixture, 'utf8'), savedOnce)
    assert.deepEqual(await page.evaluate(() => window.saved['html.undo.v1']), priorUndo)
    assert.equal(await page.evaluate(() => window.saved['html.pending.v1']), undefined)
    await page.evaluate(() => { window.storageBlocked = false })
    await page.getByRole('button', { name: 'Undo last HTML change', exact: true }).click()
    await page.waitForFunction(() => window.reloadCount === 2)
    assert.equal(fs.readFileSync(fixture, 'utf8'), original)
    console.log('PASS: failed next save preserves a small verified undo record and its usable backup')
  } finally {
    fs.writeFileSync(fixture, original, 'utf8')
    await browser.close()
    for (const file of createdBackups) {
      if (file.startsWith(fixture + '.word-peek-') && path.dirname(file) === path.dirname(fixture)) fs.unlinkSync(file)
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1 })

