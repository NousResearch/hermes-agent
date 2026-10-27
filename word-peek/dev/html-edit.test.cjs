const assert = require('node:assert/strict')
const path = require('node:path')
const { dependency, nodePaths } = require('./runtime.cjs')
const { build } = dependency('esbuild')
const { chromium } = dependency('playwright')

;(async () => {
  const bundle = await build({ entryPoints: [path.join(__dirname, 'html-edit.js')], bundle: true, write: false, format: 'iife', globalName: 'htmlEdit', nodePaths })
  const browser = await chromium.launch({ headless: true, channel: 'msedge' })
  try {
    const page = await browser.newPage()
    await page.addScriptTag({ content: bundle.outputFiles[0].text })
    const select = async (source, selector, start, end) => {
      await page.setContent(source)
      return page.evaluate(({ selector, start, end }) => {
        const block = document.querySelector(selector)
        const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT)
        const nodes = []
        while (walker.nextNode()) nodes.push(walker.currentNode)
        const point = offset => {
          for (const node of nodes) {
            if (offset <= node.length) return [node, offset]
            offset -= node.length
          }
          throw new Error('Test offset out of range')
        }
        const range = document.createRange()
        range.setStart(...point(start))
        range.setEnd(...point(end))
        getSelection().removeAllRanges()
        getSelection().addRange(range)
        return htmlEdit.capturePassage(window)
      }, { selector, start, end })
    }
    const plan = (source, passage, mode, answer = { title: 'Details', body: 'New explanation', sources: [] }) => page.evaluate(args => htmlEdit.planHtmlEdit(...args), [source, passage, mode, answer])
    let count = 0

    const source = '<!DOCTYPE html>\r\n<html><head><meta charset="utf-8"><title>Guide</title><style>p {color:red}</style></head>\r\n<body><p id="topic" class=lead>Alpha &amp; <strong>Beta</strong> gamma.</p><script>window.example=1</script></body></html>'
    const passage = await select(source, '#topic', 8, 12)
    assert.equal(passage.selectedText, 'Beta')
    assert.equal(passage.blockText, 'Alpha & Beta gamma.')
    assert.equal(passage.id, 'topic')
    assert.equal(passage.title, 'Guide')
    count++
    const note = await plan(source, passage, 'note', { title: '<img onerror=alert(1)>', body: '<script>alert(1)</script>\n\nSecond paragraph.', sources: [{ title: 'Reference', url: 'https://example.org/page' }, { title: 'Bad', url: 'javascript:alert(1)' }] })
    assert.equal(note.after.slice(0, note.editStart), source.slice(0, note.editStart))
    assert.equal(note.after.slice(note.editStart + note.replacement.length), source.slice(note.editEnd))
    assert(note.after.includes('</p>\r\n<details'))
    assert(note.previewHtml.includes('<details data-word-peek="note" open>'))
    assert(note.previewHtml.includes('&lt;script&gt;'))
    assert(!note.previewHtml.includes('javascript:'))
    assert.equal(note.answer.sources.length, 1)
    count++
    const replacement = await plan(source, passage, 'replace')
    assert(replacement.after.includes('<strong><span data-word-peek="addition">New explanation</span></strong>'))
    assert(replacement.after.endsWith(source.slice(replacement.editEnd)))
    count++
    const crossInline = await select(source, '#topic', 4, 15)
    const crossPlan = await plan(source, crossInline, 'replace', { body: 'new', sources: [] })
    const plainResult = await page.evaluate(html => new DOMParser().parseFromString(html, 'text/html').querySelector('#topic').textContent, crossPlan.after)
    assert.equal(plainResult, crossInline.blockText.slice(0, 4) + 'new' + crossInline.blockText.slice(15))
    count++

    const list = '<!doctype html><ul><li id="first">First &amp; item</li><li>Second</li></ul>'
    const listPassage = await select(list, '#first', 0, 5)
    const listPlan = await plan(list, listPassage, 'note')
    assert(listPlan.after.indexOf('<details') < listPlan.after.indexOf('</li>'))
    assert.equal(await page.evaluate(html => new DOMParser().parseFromString(html, 'text/html').querySelectorAll('ul > li').length, listPlan.after), 2)
    count++
    const cell = '<!doctype html><table><tr><td>Cell content</td><td>Other</td></tr></table>'
    const cellPassage = await select(cell, 'td', 0, 4)
    const cellPlan = await plan(cell, cellPassage, 'below')
    assert(cellPlan.after.indexOf('<section') < cellPlan.after.indexOf('</td>'))
    count++
    const optional = '<!doctype html><ul><li>First<li>Second</ul>'
    const optionalPassage = await select(optional, 'li', 0, 5)
    const optionalPlan = await plan(optional, optionalPassage, 'note')
    assert.equal(await page.evaluate(html => new DOMParser().parseFromString(html, 'text/html').querySelectorAll('ul > li').length, optionalPlan.after), 2)
    assert(optionalPlan.after.indexOf('<details') < optionalPlan.after.indexOf('<li>Second'))
    count++
    await assert.rejects(plan(source.replace(' gamma.', ' changed.'), passage, 'note'), /passage changed/)
    await assert.rejects(plan(source.replace('charset="utf-8"', 'charset="windows-1252"'), passage, 'note'), /UTF-8/)
    await assert.rejects(plan(source.replace('<body>', '<body><p id="topic">Another</p>'), passage, 'note'), /ambiguous/)
    count += 3
    const duplicate = '<!doctype html><p id="same">One</p><p id="same">Two</p>'
    const duplicatePassage = await select(duplicate, 'p:last-child', 0, 3)
    assert.equal(duplicatePassage.id, '')
    const duplicatePlan = await plan(duplicate, duplicatePassage, 'note')
    assert(duplicatePlan.after.startsWith(duplicate))
    count++
    await page.setContent('<p>First</p><p>Second</p>')
    const multiResult = await page.evaluate(() => {
      const range = document.createRange()
      range.setStart(document.querySelector('p').firstChild, 0)
      range.setEnd(document.querySelector('p:last-child').firstChild, 3)
      getSelection().removeAllRanges(); getSelection().addRange(range)
      try { htmlEdit.capturePassage(window); return 'accepted' } catch (error) { return error.message }
    })
    assert.match(multiResult, /one paragraph/)
    count++
    await assert.rejects(select('<p>First <button>press</button></p>', 'p', 0, 5), /interactive/)
    count++
    const movedSource = source.replace('<body>', '<body><p>New previous paragraph</p>')
    const movedPlan = await plan(movedSource, passage, 'note')
    assert(movedPlan.after.includes('<p>New previous paragraph</p>'))
    count++
    console.log('PASS: ' + count + ' HTML anchoring, source preservation, insertion, encoding, stale-content, and escaping scenarios')
  } finally {
    await browser.close()
  }
})().catch(error => { console.error(error); process.exitCode = 1 })

