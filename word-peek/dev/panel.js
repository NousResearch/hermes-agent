import { planHtmlEdit } from './html-edit.js'
import { generateAnswer } from './inline-ai.js'

let host, ctx, panel, active
const sessions = new Map()
let saving = false
const el = (tag, cls, text) => {
  const n = document.createElement(tag)
  if (cls) n.className = cls
  if (text != null) n.textContent = text
  return n
}
function button(label, action, cls) {
  const b = el('button', cls, label); b.type = 'button'; b.onclick = action; return b
}
export function localHtmlPath(url) {
  try {
    const u = new URL(url)
    if (u.protocol !== 'file:' || (u.hostname && u.hostname !== 'localhost')) return null
    const path = decodeURIComponent(u.pathname)
    if (!/\.html?$/i.test(path)) return null
    return /^\/[a-z]:\//i.test(path) ? path.slice(1) : path
  } catch { return null }
}
async function readHtml(path) {
  const hd = window.hermesDesktop
  if (!hd?.readPluginSource || !hd?.writeTextFile) throw new Error('This Hermes build does not expose local HTML editing.')
  const result = await hd.readPluginSource(path)
  if (result.truncated || result.binary || typeof result.text !== 'string') throw new Error('The HTML file could not be read in full.')
  if (result.text.length > 900000) throw new Error('This HTML is too large for the desktop file-writing API.')
  return result.text
}
function note(message, kind = 'info') { host.notify({kind, message}) }
function style() {
  if (document.getElementById('word-peek-build-style')) return
  const s = el('style'); s.id = 'word-peek-build-style'
  s.textContent = [
    '#word-peek-build{position:fixed;z-index:2147483647;right:16px;top:16px;bottom:16px;width:460px;max-width:calc(100vw - 24px);display:flex;flex-direction:column;background:var(--ui-bg-elevated,#20242c);color:var(--ui-text-primary,#edf0f6);border:1px solid var(--ui-stroke-secondary,#39404b);border-radius:16px;box-shadow:0 18px 70px #0007;font:13px/1.65 system-ui,sans-serif;overflow:hidden}',
    '#word-peek-build *{box-sizing:border-box}',
    '#word-peek-build header{padding:16px 20px;border-bottom:1px solid var(--ui-stroke-secondary,#39404b);display:flex;justify-content:space-between;gap:10px;align-items:center;flex-shrink:0}',
    '#word-peek-build h2{font-size:17px;margin:0;font-weight:600}',
    '#word-peek-build .wb-kicker{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--ui-text-tertiary,#9ca7b8)}',
    '#word-peek-build .wb-scroll{padding:18px 20px;overflow:auto;min-height:0;flex:1;overscroll-behavior:contain}',
    '#word-peek-build blockquote{border-left:2px solid #93b8f5;margin:0 0 16px;padding:2px 0 2px 12px;color:var(--ui-text-secondary,#b6bfcd);max-height:110px;overflow:auto;white-space:pre-wrap}',
    '#word-peek-build .wb-quote{margin:0 0 16px}',
    '#word-peek-build .wb-quote summary{cursor:pointer;list-style:none;display:flex;gap:7px;align-items:center;font-size:11px;color:var(--ui-text-tertiary,#9ca7b8);user-select:none}',
    '#word-peek-build .wb-quote summary::-webkit-details-marker{display:none}',
    '#word-peek-build .wb-quote summary::before{content:"▸";font-size:9px;line-height:1}',
    '#word-peek-build .wb-quote[open] summary::before{content:"▾"}',
    '#word-peek-build .wb-quote[open] blockquote{margin-top:8px}',
    '#word-peek-build .wb-turn{padding:16px 0;border-top:1px solid var(--ui-stroke-secondary,#39404b)}',
    '#word-peek-build .wb-question{font-weight:650;margin-bottom:8px;color:#bed5fc}',
    '#word-peek-build .wb-body{white-space:pre-wrap;overflow-wrap:anywhere}',
    '#word-peek-build .wb-meta{font-size:11px;color:var(--ui-text-secondary,#a5b0c0);margin:8px 0;overflow-wrap:anywhere}',
    '#word-peek-build .wb-controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}',
    '#word-peek-build button,#word-peek-build select{font:500 12px/1.4 system-ui;border:1px solid var(--ui-stroke-secondary,#495261);border-radius:7px;background:var(--ui-bg-base,#262c36);color:inherit;padding:8px 10px;cursor:pointer}',
    '#word-peek-build button:hover{filter:brightness(1.14)}',
    '#word-peek-build button:disabled{opacity:.5;cursor:default}',
    '#word-peek-build .wb-primary{background:#b9d2ff;color:#14233e;border-color:transparent}',
    '#word-peek-build a{color:#b9d2ff}',
    '#word-peek-build .wb-composer{padding:14px 18px;border-top:1px solid var(--ui-stroke-secondary,#39404b);flex-shrink:0}',
    '#word-peek-build textarea{display:block;width:100%;background:var(--ui-bg-base,#171c24);color:inherit;border:1px solid #495261;border-radius:8px;padding:10px;font:13px/1.5 system-ui;resize:vertical;min-height:62px;max-height:180px}',
    '#word-peek-build button:focus-visible,#word-peek-build select:focus-visible,#word-peek-build textarea:focus-visible{outline:2px solid #92baff;outline-offset:2px}',
    '#word-peek-build .wb-error{color:#ffb6af;margin:8px 0;white-space:pre-wrap}',
    '#word-peek-build .wb-preview{border:1px solid #495261;border-radius:10px;padding:12px;margin:12px 0;overflow-wrap:anywhere}',
    '#word-peek-build .wb-preview p{margin:8px 0}',
    '#word-peek-build .wb-preview summary{cursor:pointer;font-weight:600}',
    '@media(max-width:500px){#word-peek-build{right:12px;top:12px;bottom:12px;width:calc(100vw - 24px)}}'
  ].join('\n')
  document.head.append(s)
}
export function configureReadBuild(pluginHost, pluginCtx) {
  host = pluginHost; ctx = pluginCtx
  return () => {
    for (const session of sessions.values()) session.controller?.abort()
    sessions.clear(); active = null
    panel?.remove(); panel = null
    document.getElementById('word-peek-build-style')?.remove()
  }
}
export function openReadBuild(passage, guest, question = '') {
  style()
  const p = passage || {selectedText:'', context:'', url:''}
  const key = JSON.stringify([p.url, p.id || p.path, p.startOffset, p.endOffset, p.selectedText, p.sessionId, p.sessionOwner])
  let session = sessions.get(key)
  if (!session) {
    session = {key, passage:p, guest, turns:[], busy:false, error:'', draft:'', filePath:localHtmlPath(p.url), prepared:null}
    sessions.set(key, session)
    if (sessions.size > 20) {
      const oldest = [...sessions.values()].find(s => !s.busy && s !== active)
      if (oldest) sessions.delete(oldest.key)
    }
  } else {
    if (session.passage.blockText !== p.blockText) session.prepared = null
    session.passage = p
    session.filePath = localHtmlPath(p.url)
    session.guest = guest || session.guest
  }
  active = session
  if (!panel) {
    panel = el('section'); panel.id = 'word-peek-build'
    panel.setAttribute('role','dialog'); panel.setAttribute('aria-label','Explore and build')
    document.body.append(panel)
  }
  render()
  if (question.trim() && !session.busy) ask(session, question)
}
function close() { panel?.remove(); panel = null; active = null }
function render() {
  if (!panel || !active) return
  const session = active
  panel.replaceChildren()
  const header = el('header')
  const title = el('div'); title.append(el('div','wb-kicker','Word Peek'), el('h2','','Explore & build'))
  const closeButton = button('×',close); closeButton.setAttribute('aria-label','Close explore panel')
  header.append(title,closeButton)
  const scroll = el('div','wb-scroll')
  const quote = el('details','wb-quote'); if (session.quoteOpen !== false) quote.open = true
  quote.ontoggle = () => { session.quoteOpen = quote.open }
  quote.append(el('summary','',`Selected text · ${session.passage.selectedText.length} characters`))
  quote.append(el('blockquote','',session.passage.selectedText))
  scroll.append(quote)
  scroll.append(el('div','wb-meta',session.filePath ? session.filePath.split(/[\\/]/).pop() + ' · Local HTML' : 'Answers stay here. Open a local HTML preview to add them to a file.'))
  if (!session.turns.length) scroll.append(el('div','wb-meta','Ask about this passage, explore an example, or add your own note.'))
  for (const [index, turn] of session.turns.entries()) {
    const section = el('article','wb-turn')
    section.append(el('div','wb-question',turn.question),el('div','wb-body',turn.answer.body))
    section.append(el('div','wb-meta',turn.answer.sourceNote || 'Your note'))
    for (const source of turn.answer.sources) {
      const link = el('a','',source.title || source.url); link.href = source.url; link.target='_blank'; link.rel='noopener noreferrer'
      const line=el('div','wb-meta');line.append(link);section.append(line)
    }
    const controls = el('div','wb-controls')
    const mode = el('select'); mode.setAttribute('aria-label','How to add answer '+(index+1))
    for (const [value,label] of [['note','Expandable note'],['below','Insert below'],['replace','Replace selection']]) {
      const option=el('option','',label);option.value=value;mode.append(option)
    }
    const preview = button('Preview in HTML',()=>prepare(session,turn.answer,mode.value,index),'wb-primary')
    preview.disabled = !session.filePath || session.busy || saving
    controls.append(mode,preview)
    section.append(controls)
    if (session.prepared?.index===index) section.append(previewElement(session))
    scroll.append(section)
  }
  if (session.busy) scroll.append(el('div','wb-meta','Thinking about this passage…'))
  if (session.error) { const e=el('div','wb-error',session.error); e.setAttribute('role','alert');scroll.append(e) }
  const undo = lastUndo(session.filePath)
  if (undo?.path === session.filePath) {
    const row=el('div','wb-controls')
    const b=button('Undo last HTML change',()=>undoLast(session));b.disabled=saving;row.append(b);scroll.append(row)
  }
  const form=el('form','wb-composer')
  const input=el('textarea');input.rows=2;input.placeholder='Ask a follow-up or write a note…';input.setAttribute('aria-label','Question or note')
  input.value=session.draft;input.oninput=()=>{session.draft=input.value}
  const row=el('div','wb-controls')
  const submit=button(session.busy?'Working…':'Ask',()=>{},'wb-primary');submit.type='submit';submit.disabled=session.busy
  const ownNote=button('Use as my note',()=>{
    if (!input.value.trim()) return
    session.turns.push({question:'My note',answer:{title:'My note',body:input.value.trim(),sources:[]}})
    session.draft='';session.error='';render()
  });ownNote.disabled=session.busy
  row.append(submit,ownNote)
  if (session.busy) row.append(button('Cancel',()=>{session.controller?.abort();session.busy=false;session.error='Request cancelled.';render()}))
  form.append(input,row)
  form.onsubmit=e=>{e.preventDefault();if(input.value.trim()&&!session.busy) ask(session,input.value)}
  panel.append(header,scroll,form)
  scroll.scrollTop=scroll.scrollHeight
}
async function ask(session, question) {
  session.controller?.abort()
  const controller = new AbortController()
  session.controller=controller;session.busy=true;session.error='';session.prepared=null;session.draft=''
  if (active===session) render()
  try {
    const answer=await generateAnswer(host,{passage:session.passage,question,history:session.turns.slice(-6),signal:controller.signal})
    if(controller.signal.aborted) return
    session.turns.push({question,answer})
  } catch (err) {
    if(!controller.signal.aborted) {session.error=err.message || 'Could not get an answer.';session.draft=question}
  } finally {
    if(session.controller===controller) {session.busy=false;if(active===session) render()}
  }
}
async function prepare(session, answer, mode, index) {
  session.error='';session.prepared=null
  try {
    const source=await readHtml(session.filePath)
    const plan=planHtmlEdit(source,session.passage,mode,answer)
    session.prepared={...plan,index,mode,answer}
  } catch(err) {session.error=err.message}
  if(active===session) render()
}
function previewElement(session) {
  const planned=session.prepared
  const wrapper=el('div')
  wrapper.append(el('div','wb-meta',planned.mode==='replace'?'Replace the highlighted text:':'Add beside the selected passage:'))
  // Only our escaped generated fragment enters the preview.
  const preview=el('div','wb-preview'); preview.innerHTML=planned.previewHtml
  wrapper.append(preview)
  const row=el('div','wb-controls')
  const apply=button('Apply to HTML',()=>applyPlan(session,planned),'wb-primary');apply.disabled=saving
  row.append(apply,button('Back',()=>{session.prepared=null;render()}))
  wrapper.append(row)
  return wrapper
}
async function refreshGuest(session) {
  try {
    const guest=session.guest
    const current=typeof guest?.getURL==='function'?guest.getURL():guest?.getAttribute?.('src')
    if (!guest?.isConnected || localHtmlPath(current)!==session.filePath) return
    if (typeof guest.reload==='function') guest.reload()
    else if (guest.tagName==='IFRAME') guest.contentWindow?.location.reload()
  } catch { note('Saved. Reload the HTML preview to see the change.') }
}
async function digest(text) {
  if (!globalThis.crypto?.subtle) throw new Error('This desktop context cannot verify local file edits.')
  const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))
  return Array.from(new Uint8Array(hash),byte=>byte.toString(16).padStart(2,'0')).join('')
}
function lastUndo(path) {
  const pending=ctx.storage?.get('html.pending.v1',null)
  if(pending?.path===path) return pending
  const undo=ctx.storage?.get('html.undo.v1',null)
  return undo?.path===path?undo:null
}
function persist(key, record) {
  ctx.storage.set(key,record)
  if(JSON.stringify(ctx.storage.get(key,null))!==JSON.stringify(record)) throw new Error('There is no room to save Undo information. Free some application storage before editing this file.')
}
async function applyPlan(session, plan) {
  if(saving) return
  saving=true;session.error=''
  if(active===session) render()
  let touched=false, backup
  try {
    const before=await readHtml(session.filePath)
    if(before!==plan.before) throw new Error('The HTML changed since this preview. Preview the change again before applying.')
    if(plan.after.length>900000) throw new Error('The updated HTML exceeds the desktop write limit.')
    const beforeHash=await digest(plan.before), afterHash=await digest(plan.after)
    backup=session.filePath+'.word-peek-'+Date.now()+'.bak'
    await window.hermesDesktop.writeTextFile(backup,plan.before)
    if(await digest(await readHtml(backup))!==beforeHash) throw new Error('The backup could not be verified. The original has not been edited.')
    const record={path:session.filePath,backup,beforeHash,afterHash}
    persist('html.pending.v1',record)
    if(await readHtml(session.filePath)!==plan.before) throw new Error('The file changed while preparing the backup. Nothing was applied.')
    touched=true
    await window.hermesDesktop.writeTextFile(session.filePath,plan.after)
    if(await readHtml(session.filePath)!==plan.after) throw new Error('Could not verify the saved file. The original is backed up at '+backup)
    persist('html.undo.v1',record)
    ctx.storage.remove('html.pending.v1')
    session.prepared=null
    note('Added to HTML. Undo is available here.')
    await refreshGuest(session)
  } catch(err) {
    session.error=err.message
    if(touched&&backup) session.error+='\nOriginal backup: '+backup
  } finally {
    if(!touched) ctx.storage?.remove('html.pending.v1')
    saving=false;if(active===session)render()
  }
}
async function undoLast(session) {
  if(saving) return
  saving=true;session.error=''
  try {
    const last=lastUndo(session.filePath)
    if(!last) throw new Error('No saved change to undo for this file.')
    const current=await readHtml(last.path)
    if(await digest(current)!==last.afterHash) throw new Error('The HTML changed after that save. Undo would overwrite newer work; the original backup is '+last.backup)
    const original=await readHtml(last.backup)
    if(await digest(original)!==last.beforeHash) throw new Error('The backup changed. Undo has been stopped.')
    if(await readHtml(last.path)!==current) throw new Error('The HTML changed while preparing Undo. No file was overwritten.')
    await window.hermesDesktop.writeTextFile(last.path,original)
    if(await readHtml(last.path)!==original) throw new Error('Could not verify undo. Your backup is '+last.backup)
    ctx.storage.remove('html.undo.v1');ctx.storage.remove('html.pending.v1')
    session.prepared=null
    note('Last HTML change undone.')
    await refreshGuest(session)
  } catch(err) {session.error=err.message}
  finally {saving=false;if(active===session)render()}
}
