const fs = require('node:fs')
const path = require('node:path')
const assert = require('node:assert/strict')
const { checkout: hermes, dependency } = require('./runtime.cjs')
const { build } = dependency('esbuild')
const { chromium } = dependency('playwright')

;(async () => {
  if (!hermes) throw new Error('Set HERMES_CHECKOUT to run the actual runtime loader regression.')
  // Execute the real loader; mock only its surrounding application services.
  const result = await build({
    entryPoints: [path.join(hermes, 'apps/desktop/src/contrib/runtime-loader.ts')],
    bundle: true, write: false, format: 'iife', globalName: 'loader',
    plugins: [{ name: 'app-services', setup(b) {
      b.onResolve({ filter: /.*/ }, args => args.kind === 'entry-point' ? undefined : { path: args.path, namespace: 'mock' })
      b.onLoad({ filter: /.*/, namespace: 'mock' }, () => ({ contents: `
        export const installPluginSdk=()=>{};
        export const sdkImportMap=()=>({'@hermes/plugin-sdk':window.sdkUrl});
        export const notifyError=(...args)=>window.errors.push(args);
        export const createPluginContext=()=>window.context;
        export const $pluginRecords={get:()=>({})};
        export const dropPlugin=()=>{};
        export const pluginActive=()=>true;
        export const publishPlugin=record=>window.records.push(record);
      ` }))
    } }]
  })
  const browser = await chromium.launch({ headless: true, channel: 'msedge' })
  try {
    const page = await browser.newPage()
    await page.evaluate(() => {
      window.errors=[]; window.records=[]; window.disposers=[]
      window.context={storage:{get:(_k,f)=>f,set:()=>{}},register:()=>{},onDispose:f=>window.disposers.push(f)}
      window.sdkUrl=URL.createObjectURL(new Blob(['export const host={state:{},notify:()=>{}}; export const PALETTE_AREA="palette";'],{type:'text/javascript'}))
    })
    await page.addScriptTag({ content: result.outputFiles[0].text })
    const source=fs.readFileSync(path.join(__dirname,'dist/plugin.js'),'utf8')
    const loaded=await page.evaluate(source=>loader.loadRuntimePlugin(source,'word-peek-test'),source)
    assert.equal(loaded,'word-peek',JSON.stringify(await page.evaluate(()=>errors)))
    assert.equal(await page.evaluate(()=>records.at(-1).status),'loaded')
    await page.evaluate(()=>disposers.forEach(f=>f()))
    console.log('PASS actual Hermes runtime loader imports and registers bundled Word Peek')
  } finally { await browser.close() }
})().catch(e=>{console.error(e);process.exitCode=1})

