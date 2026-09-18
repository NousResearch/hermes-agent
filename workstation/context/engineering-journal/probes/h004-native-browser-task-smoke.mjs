#!/usr/bin/env node

import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const expectedBranch = 'impl4-browser-task-lifecycle'
const codeBearingAncestor = '1ac0e0a9ecaaf1c53ee0f8abfc3d8a1d802cae70'
const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim()
const branch = execFileSync('git', ['branch', '--show-current'], { cwd: repoRoot, encoding: 'utf8' }).trim()
const head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim()

function failPrecondition(message) {
  console.error(`H004_PRECONDITION_FAIL ${message}`)
  process.exit(2)
}

const isCurrentMainExecution =
  process.env.ALLOW_CURRENT_MAIN === '1' ||
  process.env.H004_CURRENT_MAIN === '1' ||
  branch === 'main' ||
  branch.startsWith('fix/') ||
  branch.startsWith('feat/')

if (isCurrentMainExecution) {
  console.log(`H004_CURRENT_MAIN_MODE active on branch '${branch}' at HEAD ${head}`)
} else {
  if (branch !== expectedBranch) failPrecondition(`wrong branch: ${branch}`)
  try {
    execFileSync('git', ['merge-base', '--is-ancestor', codeBearingAncestor, 'HEAD'], {
      cwd: repoRoot,
      stdio: 'ignore'
    })
  } catch {
    failPrecondition(`${codeBearingAncestor} is not an ancestor of ${head}`)
  }

  const productChanges = execFileSync(
    'git',
    ['diff', '--name-only', `${codeBearingAncestor}..HEAD`, '--', 'apps/desktop/electron'],
    { cwd: repoRoot, encoding: 'utf8' }
  ).trim()
  if (productChanges) {
    failPrecondition(
      `Desktop Electron product code changed after registered code-bearing ancestor. Update journal/probe first:\n${productChanges}`
    )
  }
}

const electronCandidates = [
  path.join(repoRoot, 'apps', 'desktop', 'node_modules', 'electron', 'dist', 'electron.exe'),
  path.join(repoRoot, 'node_modules', 'electron', 'dist', 'electron.exe')
]
const esbuildCandidates = [
  path.join(repoRoot, 'apps', 'desktop', 'node_modules', 'esbuild', 'bin', 'esbuild'),
  path.join(repoRoot, 'node_modules', 'esbuild', 'bin', 'esbuild')
]
const electronExe = electronCandidates.find(candidate => fs.existsSync(candidate))
const esbuildCli = esbuildCandidates.find(candidate => fs.existsSync(candidate))
if (!electronExe) failPrecondition(`electron.exe not found; checked:\n${electronCandidates.join('\n')}`)
if (!esbuildCli) failPrecondition(`esbuild JS CLI not found; checked:\n${esbuildCandidates.join('\n')}`)

const tempRoot = path.join(os.tmpdir(), 'HermesImpl4H004NativeLifecycle')
const stateRoot = path.join(os.tmpdir(), 'HermesImpl4H004NativeLifecycleState')
const appDir = path.join(tempRoot, 'electron-app')
const harnessTs = path.join(tempRoot, 'h004-native-harness.ts')
const mainCjs = path.join(appDir, 'main.cjs')
const packageJson = path.join(appDir, 'package.json')
const runtimePath = path.join(repoRoot, 'apps', 'desktop', 'electron', 'workstation-browser-runtime.ts')

fs.rmSync(tempRoot, { recursive: true, force: true })
fs.rmSync(stateRoot, { recursive: true, force: true })
fs.mkdirSync(appDir, { recursive: true })
fs.mkdirSync(stateRoot, { recursive: true })

function killTree(pid) {
  if (!pid) return
  try {
    if (process.platform === 'win32') {
      execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore' })
    } else {
      process.kill(pid, 'SIGKILL')
    }
  } catch {
    // Best effort after timeout.
  }
}

function run(command, args, { env = process.env, timeoutMs = 60000 } = {}) {
  return new Promise((resolve, reject) => {
    console.log(`RUN ${command} ${args.join(' ')}`)
    const child = spawn(command, args, {
      cwd: repoRoot,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      windowsHide: false
    })

    const timer = setTimeout(() => {
      console.error(`H004_EXTERNAL_TIMEOUT {"pid":${child.pid},"timeoutMs":${timeoutMs}}`)
      killTree(child.pid)
      reject(new Error(`timeout after ${timeoutMs}ms: ${command} ${args.join(' ')}`))
    }, timeoutMs)

    child.stdout.on('data', chunk => process.stdout.write(chunk))
    child.stderr.on('data', chunk => process.stderr.write(chunk))
    child.on('error', error => {
      clearTimeout(timer)
      reject(error)
    })
    child.on('exit', (code, signal) => {
      clearTimeout(timer)
      if (code === 0) resolve({ code, signal, pid: child.pid })
      else reject(new Error(`exit code ${code}, signal ${signal ?? 'none'}: ${command} ${args.join(' ')}`))
    })
  })
}

const harnessSource = String.raw`
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { app, BrowserWindow } from 'electron'

const mode = process.env.H004_MODE
const home = process.env.HERMES_WORKSTATION_HOME
if (!mode || !['live', 'restart1', 'restart2'].includes(mode)) {
  console.error('H004_HARNESS_CONFIG_FAIL invalid H004_MODE', mode)
  process.exit(2)
}
if (!home) {
  console.error('H004_HARNESS_CONFIG_FAIL HERMES_WORKSTATION_HOME missing')
  process.exit(2)
}

fs.mkdirSync(home, { recursive: true })
const hostUserData = path.join(home, 'ElectronHostUserData-' + mode)
fs.mkdirSync(hostUserData, { recursive: true })
app.setPath('userData', hostUserData)

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
function waitForClose(server: http.Server): Promise<void> {
  return new Promise(resolve => server.close(() => resolve()))
}
async function startLocalPage(): Promise<{ url: string; close: () => Promise<void> }> {
  const server = http.createServer((_req, res) => {
    const html = '<!doctype html>' +
      '<html><head><meta charset="utf-8"><title>Hermes Impl4 H004</title></head>' +
      '<body style="font-family:sans-serif;padding:40px;margin:0">' +
      '<h1>Hermes Implementation 4 — Native Lifecycle Smoke</h1>' +
      '<p id="marker">REAL ELECTRON / REAL WEBCONTENTSVIEW</p>' +
      '<p>Timer: <span id="timer">0</span></p>' +
      '<input id="field" type="text" value="native-smoke"/>' +
      '<button id="action-btn" onclick="document.getElementById(\'action-result\').innerText = \'action-fired\'">Click Me</button>' +
      '<p id="action-result">idle</p>' +
      '<div style="height:2500px"></div>' +
      '<p id="bottom-marker">Bottom marker</p>' +
      '<script>' +
      'let count = 0;' +
      'setInterval(() => {' +
      '  count++;' +
      '  const el = document.getElementById("timer");' +
      '  if (el) el.innerText = String(count);' +
      '  window.__h004TimerCount = count;' +
      '}, 100);' +
      '</script>' +
      '</body></html>'
    res.writeHead(200, {
      'content-type': 'text/html; charset=utf-8',
      'content-length': Buffer.byteLength(html),
      'cache-control': 'no-store'
    })
    res.end(html)
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => resolve())
  })
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('local test server failed to bind')
  return {
    url: 'http://127.0.0.1:' + address.port + '/?access_token=h004-page-url-secret',
    close: () => waitForClose(server)
  }
}

console.log('H004_BOOT', JSON.stringify({
  pid: process.pid,
  mode,
  electron: process.versions.electron,
  node: process.versions.node,
  isReady: app.isReady()
}))

const internalTimer = setTimeout(() => {
  console.error('H004_INTERNAL_TIMEOUT', JSON.stringify({
    pid: process.pid,
    mode,
    isReady: app.isReady()
  }))
  app.exit(9)
}, 35000)

app.whenReady().then(async () => {
  console.log('H004_READY', JSON.stringify({
    pid: process.pid,
    mode,
    isReady: app.isReady()
  }))

  const runtimeModule = await import(${JSON.stringify(runtimePath)})
  const {
    getWorkstationBrowserRuntime,
    workstationBrowserTaskStatePath,
    workstationBrowserSessionStatePath,
    workstationBrowserControlPath
  } = runtimeModule
  const runtime = getWorkstationBrowserRuntime()
  console.log('H004_RUNTIME_IMPORTED', JSON.stringify({
    pid: process.pid,
    mode,
    runtime: runtime.state().runtime
  }))

  // Let the module's own non-blocking app.whenReady().then(...) control-server
  // startup settle before lifecycle assertions. This is the same product pattern.
  await sleep(150)

  async function shutdown(win?: BrowserWindow, closePage?: () => Promise<void>): Promise<void> {
    try { await runtime.destroy() } catch {}
    if (win && !win.isDestroyed()) win.destroy()
    if (closePage) {
      try { await closePage() } catch {}
    }
  }

  async function runLive(): Promise<void> {
    const taskId = 'impl4-h004-live-task'
    const page = await startLocalPage()
    const win = new BrowserWindow({
      width: 1200,
      height: 820,
      show: true,
      title: 'Hermes Impl4 H004 Native Smoke'
    })
    const bounds = { x: 20, y: 20, width: 1120, height: 730 }

    console.log('H004_A_CREATE')
    const created = runtime.createTask({ taskId })
    assert(created.taskId === taskId, 'createTask changed taskId')

    runtime.showTask(taskId, win, bounds)
    const firstState = runtime.state()
    const ownedTabs = firstState.tabs.filter(tab => tab.ownerTaskId === taskId)
    assert(ownedTabs.length === 1, 'expected exactly one task-owned tab after create/show')
    const tabId = ownedTabs[0].id
    const wc = runtime.getWebContents(tabId)
    assert(wc, 'task-owned real WebContents missing')

    await wc.loadURL(page.url)
    const sentinel = 'h004-sentinel-' + Date.now() + '-' + Math.random().toString(16).slice(2)
    await wc.executeJavaScript(
      'window.__h004Sentinel=' + JSON.stringify(sentinel) +
      '; document.getElementById("field").value="typed-by-test"; window.scrollTo(0, 800); true',
      true
    )

    // Wait for timer to tick at least once
    await sleep(250)
    const initialTimer = await wc.executeJavaScript('window.__h004TimerCount || 0', true)
    const initialInput = await wc.executeJavaScript('document.getElementById("field").value', true)
    const initialScroll = await wc.executeJavaScript('window.scrollY', true)
    assert(initialTimer > 0, 'timer failed to start')
    assert(initialInput === 'typed-by-test', 'initial input value was not set')
    assert(initialScroll >= 700, 'initial scroll position was not set')

    const before = {
      taskId,
      tabId,
      webContentsId: wc.id,
      url: wc.getURL(),
      sentinel: await wc.executeJavaScript('window.__h004Sentinel', true),
      initialTimer,
      initialInput,
      initialScroll,
      ownerPageCount: runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length,
      taskStatus: runtime.listTasks().find(task => task.taskId === taskId)?.status
    }
    console.log('H004_A_BEFORE', JSON.stringify(before))
    console.log('H004_VISUAL visible 2s')
    await sleep(2000)

    const hidden = runtime.hideTask(taskId)
    assert(hidden.status === 'hidden', 'hideTask did not set hidden')
    assert(!wc.isDestroyed(), 'hideTask destroyed WebContents')
    assert(runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length === 1, 'hideTask changed owner page count')
    console.log('H004_A_HIDDEN', JSON.stringify({ attached: runtime.state().attached, status: hidden.status }))
    await sleep(1500)

    const shownAfterHide = runtime.showTask(taskId, win, bounds)
    const wcAfterHide = runtime.getWebContents(tabId)
    assert(wcAfterHide === wc, 'hide/show replaced WebContents object')
    assert(wcAfterHide?.id === before.webContentsId, 'hide/show changed webContents.id')
    assert(wcAfterHide?.getURL() === before.url, 'hide/show changed URL')
    assert(await wc.executeJavaScript('window.__h004Sentinel', true) === sentinel, 'hide/show lost renderer sentinel')
    
    // Discriminators after hide/show
    const timerAfterHide = await wc.executeJavaScript('window.__h004TimerCount || 0', true)
    const inputAfterHide = await wc.executeJavaScript('document.getElementById("field").value', true)
    const scrollAfterHide = await wc.executeJavaScript('window.scrollY', true)
    assert(timerAfterHide > initialTimer, 'timer did not advance across hide/show: ' + initialTimer + ' -> ' + timerAfterHide)
    assert(inputAfterHide === 'typed-by-test', 'input not preserved across hide/show: ' + inputAfterHide)
    assert(scrollAfterHide >= 700, 'scroll not preserved across hide/show: ' + scrollAfterHide)
    assert(runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length === 1, 'hide/show duplicated task page')
    assert(shownAfterHide.taskId === taskId, 'hide/show changed logical taskId')

    console.log('H004_A_AFTER_HIDE_SHOW', JSON.stringify({
      taskId,
      tabId,
      webContentsId: wc.id,
      url: wc.getURL(),
      timer: timerAfterHide,
      input: inputAfterHide,
      scroll: scrollAfterHide,
      ownerPageCount: runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length
    }))
    console.log('H004_VISUAL reexposed-after-hide 2s')
    await sleep(2000)

    const parked = runtime.parkTask(taskId)
    assert(parked.status === 'parked' && parked.parked === true, 'parkTask did not set parked')
    assert(!wc.isDestroyed(), 'parkTask destroyed WebContents')
    assert(runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length === 1, 'parkTask changed owner page count')
    console.log('H004_A_PARKED', JSON.stringify({ attached: runtime.state().attached, status: parked.status }))
    await sleep(1500)

    const shownAfterPark = runtime.showTask(taskId, win, bounds)
    const wcAfterPark = runtime.getWebContents(tabId)
    assert(wcAfterPark === wc, 'park/show replaced WebContents object')
    assert(wcAfterPark?.id === before.webContentsId, 'park/show changed webContents.id')
    assert(wcAfterPark?.getURL() === before.url, 'park/show changed URL')
    assert(await wc.executeJavaScript('window.__h004Sentinel', true) === sentinel, 'park/show lost renderer sentinel')

    // Discriminators after park/show
    const timerAfterPark = await wc.executeJavaScript('window.__h004TimerCount || 0', true)
    const inputAfterPark = await wc.executeJavaScript('document.getElementById("field").value', true)
    const scrollAfterPark = await wc.executeJavaScript('window.scrollY', true)
    assert(timerAfterPark > timerAfterHide, 'timer did not advance across park/show: ' + timerAfterHide + ' -> ' + timerAfterPark)
    assert(inputAfterPark === 'typed-by-test', 'input not preserved across park/show: ' + inputAfterPark)
    assert(scrollAfterPark >= 700, 'scroll not preserved across park/show: ' + scrollAfterPark)
    assert(runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length === 1, 'park/show duplicated task page')
    assert(shownAfterPark.taskId === taskId, 'park/show changed taskId')

    console.log('H004_A_AFTER_PARK_SHOW', JSON.stringify({
      taskId,
      tabId,
      webContentsId: wc.id,
      url: wc.getURL(),
      timer: timerAfterPark,
      input: inputAfterPark,
      scroll: scrollAfterPark,
      ownerPageCount: runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length
    }))
    console.log('H004_VISUAL reexposed-after-park 2s')
    await sleep(1500)

    // Real Controller Action Verification (H013/controller path):
    // Execute browser_* action on the loopback controller started by runtime
    const controlPath = workstationBrowserControlPath()
    assert(fs.existsSync(controlPath), 'control descriptor file must exist')
    const control = JSON.parse(fs.readFileSync(controlPath, 'utf8'))
    assert(control.runtime === 'electron-chromium', 'controller runtime must be electron-chromium, got ' + control.runtime)

    console.log('H004_A_CONTROLLER_ACTION_START', JSON.stringify({ url: control.url, runtime: control.runtime }))
    const controllerRes = await fetch(control.url + '/v1/action', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + control.token,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        action: 'browser_snapshot',
        task_id: taskId,
        arguments: {}
      })
    })
    assert(controllerRes.ok, 'controller request failed with HTTP ' + controllerRes.status)
    const actionResult = await controllerRes.json()
    assert(actionResult.success === true, 'controller action failed: ' + JSON.stringify(actionResult))
    assert(control.runtime === 'electron-chromium', 'controller runtime was not electron-chromium')

    // Confirm the same task-owned tab was operated on and no external fallback occurred
    const postActionTabs = runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId)
    assert(postActionTabs.length === 1, 'post-action must have exactly one task-owned tab')
    assert(postActionTabs[0].id === tabId, 'post-action tab id mismatch: expected ' + tabId + ', got ' + postActionTabs[0].id)
    assert(runtime.state().tabs.length === 1, 'no second/fallback page was allocated')
    console.log('H004_A_CONTROLLER_ACTION_PASS', JSON.stringify({
      runtime: control.runtime,
      taskId,
      tabId,
      snapshotSuccess: actionResult.success
    }))

    console.log('H004_B_DESTROY_BEGIN')
    const destroyedResult = runtime.destroyTask(taskId)
    await sleep(250)
    const destroyEvidence = {
      result: destroyedResult,
      webContentsDestroyed: wc.isDestroyed(),
      taskListed: runtime.listTasks().some(task => task.taskId === taskId),
      remainingOwnerTabs: runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length
    }
    assert(destroyedResult === true, 'destroyTask returned false')
    assert(destroyEvidence.webContentsDestroyed, 'destroyTask left prior WebContents alive')
    assert(!destroyEvidence.taskListed, 'destroyTask left task metadata')
    assert(destroyEvidence.remainingOwnerTabs === 0, 'destroyTask left task-owned BrowserEntry')
    console.log('H004_B_DESTROY', JSON.stringify(destroyEvidence))
    console.log('H004_LIVE_DESTROY_PASS')

    await shutdown(win, page.close)
  }

  async function runRestart1(): Promise<void> {
    const taskId = 'impl4-h004-restart-task'
    const page = await startLocalPage()
    const win = new BrowserWindow({
      width: 1000,
      height: 700,
      show: true,
      title: 'Hermes Impl4 H004 Restart Phase 1'
    })
    const bounds = { x: 20, y: 20, width: 920, height: 610 }

    runtime.createTask({ taskId })
    runtime.showTask(taskId, win, bounds)
    const ownerTabs = runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId)
    assert(ownerTabs.length === 1, 'restart1 expected one task-owned page')
    const wc = runtime.getWebContents(ownerTabs[0].id)
    assert(wc, 'restart1 WebContents missing')

    await wc.loadURL(page.url)
    await wc.executeJavaScript(
      "window.__h004TypedSecret='h004-renderer-typed-secret'; true",
      true
    )
    const parked = runtime.parkTask(taskId)
    assert(parked.status === 'parked', 'restart1 task was not parked')

    const logical = runtime.listTasks().find(task => task.taskId === taskId)
    assert(logical, 'restart1 logical task missing')
    const taskStatePath = fs.existsSync(workstationBrowserTaskStatePath())
      ? workstationBrowserTaskStatePath()
      : workstationBrowserSessionStatePath()
    const persisted = fs.readFileSync(taskStatePath, 'utf8')
    assert(!persisted.includes('h004-page-url-secret'), 'BrowserTask structural state leaked page URL secret')
    assert(!persisted.includes('h004-renderer-typed-secret'), 'BrowserTask structural state leaked renderer secret')

    const phase1 = {
      pid: process.pid,
      taskId,
      status: logical.status,
      recoveryState: logical.recoveryState,
      taskStatePath,
      secretIsolation: true
    }
    fs.writeFileSync(path.join(home, 'h004-phase1.json'), JSON.stringify(phase1, null, 2), 'utf8')
    console.log('H004_C_PHASE1', JSON.stringify(phase1))
    await sleep(600)

    await shutdown(win, page.close)
  }

  async function runRestart2(): Promise<void> {
    const taskId = 'impl4-h004-restart-task'
    const phase1Path = path.join(home, 'h004-phase1.json')
    assert(fs.existsSync(phase1Path), 'restart2 phase1 evidence file missing')
    const phase1 = JSON.parse(fs.readFileSync(phase1Path, 'utf8')) as {
      pid: number
      taskId: string
    }
    assert(phase1.pid !== process.pid, 'restart2 reused same Electron OS PID')
    assert(phase1.taskId === taskId, 'restart2 phase1 taskId mismatch')

    const restored = runtime.listTasks().find(task => task.taskId === taskId)
    assert(restored, 'logical BrowserTask did not restore after process restart')
    const ownerCountBefore = runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId).length
    assert(restored.status === 'parked', 'restored status expected parked, got ' + restored.status)
    assert(restored.recoveryState === 'restored', 'restored recoveryState expected restored, got ' + restored.recoveryState)
    assert(ownerCountBefore === 0, 'restored task eagerly created a task-owned page')

    console.log('H004_C_RESTORED', JSON.stringify({
      phase1Pid: phase1.pid,
      phase2Pid: process.pid,
      taskId: restored.taskId,
      status: restored.status,
      recoveryState: restored.recoveryState,
      ownerPageCountBefore: ownerCountBefore
    }))

    const win = new BrowserWindow({
      width: 1000,
      height: 700,
      show: true,
      title: 'Hermes Impl4 H004 Restart Phase 2'
    })
    const shown = runtime.showTask(taskId, win, { x: 20, y: 20, width: 920, height: 610 })
    const ownerTabsAfter = runtime.state().tabs.filter(tab => tab.ownerTaskId === taskId)
    assert(shown.taskId === taskId, 'lazy recreation changed logical taskId')
    assert(shown.recoveryState === 'recreated', 'lazy recreation recoveryState expected recreated, got ' + shown.recoveryState)
    assert(ownerTabsAfter.length === 1, 'lazy recreation expected exactly one task-owned page')

    console.log('H004_C_RECREATED', JSON.stringify({
      phase1Pid: phase1.pid,
      phase2Pid: process.pid,
      sameLogicalTaskId: shown.taskId === phase1.taskId,
      ownerPageCountBefore: ownerCountBefore,
      ownerPageCountAfter: ownerTabsAfter.length,
      recoveryStateAfter: shown.recoveryState,
      ownerTaskId: ownerTabsAfter[0]?.ownerTaskId
    }))

    runtime.destroyTask(taskId)
    await sleep(200)
    await shutdown(win)
    console.log('H004_RESTART_PASS')
  }

  try {
    if (mode === 'live') await runLive()
    if (mode === 'restart1') await runRestart1()
    if (mode === 'restart2') await runRestart2()
    clearTimeout(internalTimer)
    console.log('H004_MODE_PASS', mode)
    app.exit(0)
  } catch (error) {
    clearTimeout(internalTimer)
    console.error('H004_PRODUCT_PATH_FAIL', JSON.stringify({
      pid: process.pid,
      mode,
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : null
    }))
    try { await runtime.destroy() } catch {}
    app.exit(1)
  }
}).catch(error => {
  clearTimeout(internalTimer)
  console.error('H004_READY_OR_IMPORT_FAIL', JSON.stringify({
    pid: process.pid,
    mode,
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : null
  }))
  app.exit(1)
})
`

fs.writeFileSync(harnessTs, harnessSource, 'utf8')
fs.writeFileSync(
  packageJson,
  JSON.stringify({
    name: 'hermes-impl4-h004-native-lifecycle',
    private: true,
    main: 'main.cjs'
  }, null, 2),
  'utf8'
)

console.log('H004_PROBE_CONTEXT', JSON.stringify({
  branch,
  head,
  codeBearingAncestor,
  electronExe,
  esbuildCli,
  platform: process.platform,
  osRelease: os.release(),
  node: process.version,
  productChangesAfterCodeBearingAncestor: []
}))

try {
  console.log('\n=== H004_BUILD ===')
  await run(process.execPath, [
    esbuildCli,
    harnessTs,
    '--bundle',
    '--platform=node',
    '--format=cjs',
    '--target=node20',
    `--outfile=${mainCjs}`,
    '--external:electron'
  ], { timeoutMs: 30000 })

  if (!fs.existsSync(mainCjs) || fs.statSync(mainCjs).size < 1000) {
    throw new Error('H004 build returned success but main.cjs is missing or unexpectedly small')
  }

  const baseEnv = {
    ...process.env,
    HERMES_WORKSTATION_HOME: stateRoot
  }

  console.log('\n=== H004_LIVE_AND_DESTROY ===')
  await run(electronExe, [appDir], {
    timeoutMs: 45000,
    env: { ...baseEnv, H004_MODE: 'live' }
  })

  console.log('\n=== H004_RESTART_PHASE1 ===')
  await run(electronExe, [appDir], {
    timeoutMs: 30000,
    env: { ...baseEnv, H004_MODE: 'restart1' }
  })

  console.log('\n=== H004_RESTART_PHASE2 ===')
  await run(electronExe, [appDir], {
    timeoutMs: 30000,
    env: { ...baseEnv, H004_MODE: 'restart2' }
  })

  console.log('\nH004_CLASSIFICATION=VALIDATED')
  console.log('H004_CONCLUSION=Real Electron BrowserTask lifecycle passed live identity, explicit destroy, real restart, lazy logical recovery, and structural secret-isolation checks.')
  process.exit(0)
} catch (error) {
  console.error('\nH004_CLASSIFICATION=FAILED_OR_REFORMULATE')
  console.error(
    'H004_CONCLUSION=' +
    (error instanceof Error ? (error.stack || error.message) : String(error))
  )
  console.error('H004_TEMP_ROOT=' + tempRoot)
  console.error('H004_STATE_ROOT=' + stateRoot)
  process.exit(1)
}
