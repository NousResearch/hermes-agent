const fs = require('node:fs')
const path = require('node:path')
const assert = require('node:assert/strict')
const { dependency } = require('./runtime.cjs')
const { chromium } = dependency('playwright')
const dir=__dirname
const fixture=path.join(dir,'study-fixture.html').replaceAll('\\','/')
const pluginRoot=path.join(__dirname,'test-plugins/word-peek').replaceAll('\\','/')
const original=fs.readFileSync(fixture,'utf8')
;(async()=>{
  const browser=await chromium.launch({headless:true,channel:'msedge'})
  const page=await browser.newPage({viewport:{width:1200,height:800}})
  const guest=await browser.newPage()
  try {
    await guest.goto('file:///'+fixture)
    await page.goto('file:///'+fixture)
    await page.setContent('<body style="margin:0;background:#13171f;color:#eef;font:16px system-ui"><main style="padding:60px;width:550px"><h1>Study guide</h1><p>Read, ask, and add what you learn.</p><webview></webview><div data-composer-target="main" data-composer-surface-id="test">Chat</div></main></body>')
    await page.exposeFunction('guestEval', script=>guest.evaluate(script))
    await page.exposeFunction('readFile', p=>{
      if(p.startsWith(pluginRoot+'/wn-'))return {text:'{}',path:p,truncated:false}
      if(p===fixture || p.startsWith(fixture+'.word-peek-'))return {text:fs.readFileSync(p,'utf8'),path:p,truncated:false}
      throw new Error('Unexpected read '+p)
    })
    await page.exposeFunction('writeFile',(p,text)=>{
      if(p!==fixture&&!p.startsWith(fixture+'.word-peek-'))throw new Error('Unexpected write '+p)
      fs.writeFileSync(p,text,'utf8');return {path:p}
    })
    await page.evaluate(({fixture,pluginRoot})=>{
      window.fixture=fixture;window.rpcCalls=[];window.store={};window.notices=[];window.sentChat=[]
      window.testContext={register:()=>{},onDispose:f=>window.dispose=f,storage:{get:(k,f)=>window.store[k]??f,set:(k,v)=>window.store[k]=v,remove:k=>delete window.store[k]}}
      window.mockHost={notify:n=>window.notices.push(n),state:{focusedSessionId:{get:()=> 'session-under-test'}},request:async(method,params)=>{
        window.rpcCalls.push({method,params})
        await new Promise(r=>setTimeout(r,50))
        return {text:JSON.stringify({title:'Learn more about the lattice',body:'A lattice organizes connected elements.\n\nIn this guide, the surrounding sentence explains the relationship.',sources:['https://example.org/lattice']})}
      }}
      window.hermesDesktop={desktopPluginsRoot:async()=>pluginRoot.replace('/word-peek',''),readPluginSource:p=>window.readFile(p),writeTextFile:(p,t)=>window.writeFile(p,t)}
      const w=document.querySelector('webview');w.setAttribute('src','file:///'+fixture);w.executeJavaScript=s=>window.guestEval(s);w.getURL=()=>w.getAttribute('src');w.reload=()=>{window.reloadCount=(window.reloadCount||0)+1}
      window.addEventListener('hermes:composer-submit',e=>window.sentChat.push(e.detail))
    },{fixture,pluginRoot})
    let bundle=fs.readFileSync(path.join(dir,'dist/plugin.js'),'utf8')
      .replace(/^import .* from "@hermes\/plugin-sdk";/m,'const host2=window.mockHost, PALETTE_AREA="palette";')
      .replace(/export \{\s*plugin_default as default\s*\};?/, 'window.plugin=plugin_default;')
    await page.addScriptTag({content:bundle})
    await page.evaluate(()=>plugin.register(window.testContext))
    const select=async()=>{
      await guest.evaluate(()=>{
        const text=document.querySelector('#subject strong').firstChild,r=document.createRange()
        r.selectNodeContents(text);getSelection().removeAllRanges();getSelection().addRange(r)
      })
      await page.evaluate(()=>{
        const e=new Event('context-menu',{bubbles:false})
        e.params={selectionText:'lattice',x:350,y:180,frameURL:'file:///'+window.fixture}
        document.querySelector('webview').dispatchEvent(e)
      })
    }
    await select()
    await page.getByLabel('Ask about this passage').fill('What does this mean?')
    await page.getByRole('button',{name:'Ask',exact:true}).click()
    await page.waitForSelector('.wb-turn')
    assert.equal(await page.evaluate(()=>rpcCalls[0].method),'llm.oneshot')
    assert.equal(await page.evaluate(()=>sentChat.length),0)
    const input=await page.evaluate(()=>JSON.parse(rpcCalls[0].params.input))
    assert.equal(input.document.selectedText,'lattice')
    assert.match(input.document.surroundingText,/Anduril/)
    assert.equal(input.sources[0].url,'https://example.org/lattice')
    await page.getByRole('button',{name:'Preview in HTML'}).click()
    await page.waitForSelector('.wb-preview details')
    assert.equal(fs.readFileSync(fixture,'utf8'),original)
    await page.screenshot({path:path.join(dir,'read-build-preview.png')})
    await page.getByRole('button',{name:'Apply to HTML'}).click()
    await page.waitForFunction(()=>window.reloadCount===1)
    let updated=fs.readFileSync(fixture,'utf8')
    assert(updated.includes('<details data-word-peek="note">'))
    assert(updated.includes('https://example.org/lattice'))
    assert(updated.includes('<script>window.fixtureSentinel = 123</script>'))
    assert(updated.startsWith(original.slice(0,original.indexOf('</p>')+4)))
    await page.getByRole('button',{name:'Undo last HTML change'}).click()
    await page.waitForFunction(()=>window.reloadCount===2)
    assert.equal(fs.readFileSync(fixture,'utf8'),original)
    await page.getByLabel('Question or note').fill('Give me another example.')
    await page.getByRole('button',{name:'Ask',exact:true}).click()
    await page.waitForFunction(()=>document.querySelectorAll('.wb-turn').length===2)
    assert.equal(await page.evaluate(()=>JSON.parse(rpcCalls[1].params.input).previousQuestions.length),1)
    await page.getByRole('button',{name:'Preview in HTML'}).last().click()
    await page.waitForSelector('.wb-preview')
    fs.appendFileSync(fixture,'\n<!-- concurrent edit -->')
    await page.getByRole('button',{name:'Apply to HTML'}).click()
    await page.getByRole('alert').filter({hasText:'changed since this preview'}).waitFor()
    assert(fs.readFileSync(fixture,'utf8').endsWith('<!-- concurrent edit -->'))
    fs.writeFileSync(fixture,original)
    await page.getByRole('button',{name:'Close explore panel'}).click()
    await select()
    await page.getByRole('button',{name:'Explore',exact:true}).click()
    await page.waitForFunction(()=>document.querySelectorAll('.wb-turn').length>=2)
    await page.getByLabel('Question or note').fill('My own useful note.')
    await page.waitForFunction(()=>!document.querySelector('.wb-composer button').disabled)
    await page.getByRole('button',{name:'Use as my note',exact:true}).click()
    await page.getByLabel('How to add answer 4').selectOption('replace')
    await page.getByRole('button',{name:'Preview in HTML'}).last().click()
    await page.waitForSelector('.wb-preview')
    await page.getByRole('button',{name:'Apply to HTML'}).click()
    await page.waitForFunction(()=>window.reloadCount===3)
    assert(fs.readFileSync(fixture,'utf8').includes('<strong><span data-word-peek="addition">My own useful note.</span></strong>'))
    await page.getByRole('button',{name:'Undo last HTML change'}).click()
    await page.waitForFunction(()=>window.reloadCount===4)
    assert.equal(fs.readFileSync(fixture,'utf8'),original)
    await page.setViewportSize({width:360,height:640})
    await page.screenshot({path:path.join(dir,'read-build-narrow.png')})
    const r=await page.locator('#word-peek-build').boundingBox()
    assert(r.x>=0&&r.x+r.width<=360&&r.y+r.height<=640)
    await page.evaluate(()=>dispose())
    assert.equal(await page.locator('#word-peek-build').count(),0)
    console.log('PASS: bundled guest capture, inline Q&A and followups, no chat writes, preview, actual fixture save/backup/undo, concurrent-change refusal, replacement, narrow UI, cleanup')
  } finally {fs.writeFileSync(fixture,original);await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1})

