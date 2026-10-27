import { parse } from 'parse5'

const BLOCK_TAGS = new Set(['P', 'LI', 'TD', 'TH', 'BLOCKQUOTE', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'])
const UNSAFE_CONTENT = 'script,style,template,noscript,iframe,object,embed,input,textarea,select,button'

// Keep this function self-contained: the host serializes it into an isolated webview.
export function capturePassage(view) {
  const doc = view.document
  const selection = view.getSelection()
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) throw new Error('Select a passage first.')
  const range = selection.getRangeAt(0)
  const unsafe = 'script,style,template,noscript,iframe,object,embed,input,textarea,select,button,[contenteditable]:not([contenteditable="false"])'
  const closestBlock = node => {
    const element = node.nodeType === 1 ? node : node.parentElement
    if (!element || element.closest(unsafe)) return null
    return element.closest('p,li,td,th,blockquote,h1,h2,h3,h4,h5,h6')
  }
  const block = closestBlock(range.startContainer)
  if (!block || block !== closestBlock(range.endContainer)) throw new Error('Select text inside one paragraph, heading, list item, or table cell.')
  if (block.querySelector(unsafe)) throw new Error('This passage contains interactive content. Select a simpler paragraph.')
  const start = doc.createRange()
  start.selectNodeContents(block)
  start.setEnd(range.startContainer, range.startOffset)
  const end = doc.createRange()
  end.selectNodeContents(block)
  end.setEnd(range.endContainer, range.endOffset)
  const startOffset = start.toString().length
  const endOffset = end.toString().length
  const selectedText = range.toString()
  const blockText = block.textContent || ''
  if (!selectedText.trim() || blockText.slice(startOffset, endOffset) !== selectedText) throw new Error('The selection could not be anchored to this passage.')
  const path = []
  let node = block
  while (node !== doc.documentElement) {
    const parent = node.parentElement
    if (!parent) throw new Error('The selected passage is no longer in the document.')
    path.unshift(Array.prototype.indexOf.call(parent.children, node))
    node = parent
  }
  const id = block.id && Array.from(doc.querySelectorAll('[id]')).filter(el => el.id === block.id).length === 1 ? block.id : ''
  return {
    url: doc.location?.href || view.location?.href || '',
    title: doc.title || '',
    selectedText,
    blockText,
    tagName: block.tagName,
    path,
    id,
    startOffset,
    endOffset,
    context: blockText.slice(Math.max(0, startOffset - 400), Math.max(0, startOffset - 400) + 2400),
    sources: Array.from((block.closest('section,article') || doc.body).querySelectorAll('a[href]')).filter(a => /^https?:\/\//i.test(a.href)).slice(0, 24).map(a => ({title:(a.textContent || a.href).trim().slice(0,180),url:a.href}))
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char])
}

function checkedAnswer(answer) {
  const title = String(answer?.title || 'Learn more').trim().slice(0, 200)
  const body = String(answer?.body || '').trim()
  if (!body) throw new Error('There is no answer to add yet.')
  if (body.length > 100_000) throw new Error('This answer is too long to add in one step.')
  const sources = []
  for (const source of Array.isArray(answer?.sources) ? answer.sources : []) {
    try {
      const url = new URL(String(source?.url || ''))
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) continue
      if (sources.some(item => item.url === url.href)) continue
      sources.push({ title: String(source.title || url.hostname).slice(0, 300), url: url.href })
    } catch { /* Only absolute web citations are accepted. */ }
  }
  return { title: title || 'Learn more', body, sources: sources.slice(0, 30), sourceNote: String(answer?.sourceNote || '').slice(0, 400) }
}

function answerHtml(answer, mode, preview = false) {
  const links = answer.sources.map((source, index) => '<a href="' + escapeHtml(source.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(source.title || '[' + (index + 1) + ']') + '</a>')
  if (mode === 'replace') {
    return '<span data-word-peek="addition">' + escapeHtml(answer.body).replace(/\r?\n/g, '<br>') + (links.length ? ' <span class="word-peek-sources">(' + links.join('; ') + ')</span>' : '') + '</span>'
  }
  const content = answer.body.split(/\r?\n\s*\r?\n/).map(paragraph => '<p>' + escapeHtml(paragraph).replace(/\r?\n/g, '<br>') + '</p>').join('\n')
  const citations = (links.length ? '\n<p class="word-peek-sources"><small>References: ' + links.join(' · ') + '</small></p>' : '') + (answer.sourceNote ? '\n<p><small>' + escapeHtml(answer.sourceNote) + '</small></p>' : '')
  return mode === 'note'
    ? '<details data-word-peek="note"' + (preview ? ' open' : '') + '><summary>' + escapeHtml(answer.title) + '</summary>\n' + content + citations + '\n</details>'
    : '<section data-word-peek="addition"><h3>' + escapeHtml(answer.title) + '</h3>\n' + content + citations + '\n</section>'
}

function allParseElements(node, result = []) {
  if (node.tagName) result.push(node)
  for (const child of node.childNodes || []) allParseElements(child, result)
  return result
}

function parseText(node) {
  return node.nodeName === '#text' ? node.value : (node.childNodes || []).map(parseText).join('')
}

function resolvePath(root, path, children) {
  let node = root
  for (const index of path) {
    if (!Number.isInteger(index) || index < 0) return null
    node = node && children(node)[index]
  }
  return node || null
}

function assertEncoding(doc, source) {
  for (const meta of doc.querySelectorAll('meta')) {
    const encoding = meta.getAttribute('charset') || (/^content-type$/i.test(meta.getAttribute('http-equiv') || '') ? (meta.getAttribute('content') || '').match(/charset\s*=\s*["']?([^\s;"']+)/i)?.[1] : '')
    if (encoding && !/^utf-?8$/i.test(encoding.trim())) throw new Error('Saving is available only for HTML declared as UTF-8.')
  }
  const xmlEncoding = source.match(/^\s*<\?xml\s[^>]*encoding\s*=\s*["']([^"']+)["']/i)?.[1]
  if (xmlEncoding && !/^utf-?8$/i.test(xmlEncoding)) throw new Error('Saving is available only for UTF-8 HTML.')
  if (source.includes('\u0000')) throw new Error('The source does not appear to be UTF-8 HTML.')
}

function textPoint(doc, block, offset, preferPrevious = false) {
  const walker = doc.createTreeWalker(block, 4)
  let node
  let seen = 0
  while ((node = walker.nextNode())) {
    if (seen + node.nodeValue.length > offset || (preferPrevious && seen + node.nodeValue.length === offset)) return { node, offset: offset - seen }
    seen += node.nodeValue.length
  }
  throw new Error('The selected text is no longer available in this passage.')
}

export function planHtmlEdit(source, passage, mode = 'note', answer) {
  if (typeof source !== 'string' || !source.trim()) throw new Error('The HTML source is empty.')
  if (!['note', 'below', 'replace'].includes(mode)) throw new Error('Choose an insertion style.')
  if (!passage || !Array.isArray(passage.path) || !BLOCK_TAGS.has(passage.tagName)) throw new Error('Select a passage in an HTML preview first.')
  const doc = new DOMParser().parseFromString(source, 'text/html')
  assertEncoding(doc, source)
  const parsed = parse(source, { sourceCodeLocationInfo: true, scriptingEnabled: false })
  const parseElements = allParseElements(parsed)
  const root = parseElements.find(node => node.tagName === 'html')
  let block
  let sourceBlock
  if (passage.id) {
    const matches = Array.from(doc.querySelectorAll('[id]')).filter(node => node.id === passage.id)
    const sourceMatches = parseElements.filter(node => node.attrs?.some(attr => attr.name === 'id' && attr.value === passage.id))
    if (matches.length !== 1 || sourceMatches.length !== 1) throw new Error('The passage identifier is missing or ambiguous. Select the passage again.')
    block = matches[0]
    sourceBlock = sourceMatches[0]
  } else {
    block = resolvePath(doc.documentElement, passage.path, node => Array.from(node.children))
    sourceBlock = resolvePath(root, passage.path, node => (node.childNodes || []).filter(child => child.tagName))
  }
  if (!block || !sourceBlock || block.tagName !== passage.tagName || sourceBlock.tagName !== passage.tagName.toLowerCase() || block.namespaceURI !== 'http://www.w3.org/1999/xhtml') throw new Error('The original passage could not be found. Select it again.')
  if (block.closest('script,style,template,noscript,[contenteditable]:not([contenteditable="false"])') || block.querySelector(UNSAFE_CONTENT + ',[contenteditable]:not([contenteditable="false"])')) throw new Error('This passage contains unsupported interactive content.')
  const currentText = block.textContent || ''
  const { startOffset, endOffset, selectedText, blockText } = passage
  if (currentText !== blockText || parseText(sourceBlock) !== currentText || !Number.isInteger(startOffset) || !Number.isInteger(endOffset) || startOffset < 0 || endOffset <= startOffset || endOffset > currentText.length || currentText.slice(startOffset, endOffset) !== selectedText) throw new Error('The passage changed after you selected it. Select it again before adding this answer.')
  const location = sourceBlock.sourceCodeLocation
  if (!location?.startTag || !Number.isInteger(location.endOffset) || location.endOffset <= location.startOffset) throw new Error('This passage has no reliable position in the original HTML.')
  const originalBlock = source.slice(location.startOffset, location.endOffset)
  const prepared = checkedAnswer(answer)
  const fragment = answerHtml(prepared, mode)
  let changedBlock
  let editStart
  let editEnd
  let replacement
  if (mode === 'replace') {
    const copy = block.cloneNode(true)
    const start = textPoint(doc, copy, startOffset)
    const end = textPoint(doc, copy, endOffset, true)
    const range = doc.createRange()
    range.setStart(start.node, start.offset)
    range.setEnd(end.node, end.offset)
    const template = doc.createElement('template')
    template.innerHTML = fragment
    range.deleteContents()
    range.insertNode(template.content)
    changedBlock = copy.outerHTML
    editStart = location.startOffset
    editEnd = location.endOffset
    replacement = changedBlock
  } else {
    const inside = ['LI', 'TD', 'TH'].includes(block.tagName)
    editStart = inside ? (location.endTag?.startOffset ?? location.endOffset) : location.endOffset
    editEnd = editStart
    const newline = source.includes('\r\n') ? '\r\n' : '\n'
    replacement = newline + fragment.replace(/\n/g, newline) + newline
    const relative = editStart - location.startOffset
    changedBlock = inside ? originalBlock.slice(0, relative) + replacement + originalBlock.slice(relative) : originalBlock + replacement
  }
  const after = source.slice(0, editStart) + replacement + source.slice(editEnd)
  if (after.length > 1_000_000) throw new Error('The updated HTML exceeds the desktop save limit.')
  return {
    before: source,
    after,
    mode,
    answer: prepared,
    previewHtml: answerHtml(prepared, mode, true),
    originalBlock,
    changedBlock,
    selectedText,
    editStart,
    editEnd,
    replacement
  }
}

