#!/usr/bin/env node

/*
 * Native, hidden-window soak for the Electron Workstation Browser runtime.
 *
 * H010 proves a finite restart/fault scenario. H011 keeps the real
 * WebContentsView + BrowserTask runtime under repeated task selection,
 * navigation, parking/hiding and process reconnect episodes. It intentionally
 * uses hidden BrowserWindow instances and windowsHide so this probe never
 * reveals a Desktop window during CI or local validation.
 */

import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'

const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], { encoding: 'utf8' }).trim()
const head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim()

function failPrecondition(message) {
  console.error('H011_PRECONDITION_FAIL ' + message)
  process.exit(2)
}

if (process.platform !== 'win32') failPrecondition('native Browser runtime soak requires Windows')

const durationMs = Number(process.env.H011_DURATION_MS || process.argv[2] || 60_000)
const taskCount = Number(process.env.H011_TASK_COUNT || 4)
const childIterations = Number(process.env.H011_CHILD_ITERATIONS || 24)

if (!Number.isFinite(durationMs) || durationMs <= 0) failPrecondition('duration must be positive')
if (!Number.isInteger(taskCount) || taskCount < 2 || taskCount > 8) failPrecondition('task count must be between 2 and 8')
if (!Number.isInteger(childIterations) || childIterations < 1 || childIterations > 500) failPrecondition('child iterations must be between 1 and 500')

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
if (!electronExe) failPrecondition('electron.exe not found; checked: ' + electronCandidates.join(', '))
if (!esbuildCli) failPrecondition('esbuild JS CLI not found; checked: ' + esbuildCandidates.join(', '))

const configuredRoot = process.env.H011_WORK_ROOT?.trim()
const tempRoot = configuredRoot
  ? path.resolve(configuredRoot)
  : path.join(os.tmpdir(), 'HermesH011BrowserRuntimeSoak-' + head.slice(0, 12))
const appDir = path.join(tempRoot, 'electron-app')
const harnessTs = path.join(tempRoot, 'h011-native-harness.ts')
const mainCjs = path.join(appDir, 'main.cjs')
const packageJson = path.join(appDir, 'package.json')
const workstationHome = path.join(tempRoot, 'workstation')
const browserProfile = path.join(tempRoot, 'browser-profile')
const runtimePath = path.join(repoRoot, 'apps', 'desktop', 'electron', 'workstation-browser-runtime.ts')
const reportPath = process.env.H011_REPORT?.trim() ? path.resolve(process.env.H011_REPORT.trim()) : null

fs.rmSync(tempRoot, { recursive: true, force: true })
fs.mkdirSync(appDir, { recursive: true })

function killTree(pid) {
  if (!pid) return
  try {
    execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore' })
  } catch {
    // Best effort after timeout.
  }
}

function run(command, args, { env = process.env, timeoutMs = 60_000, allowedCodes = [0] } = {}) {
  return new Promise((resolve, reject) => {
    console.log('RUN ' + command + ' ' + args.join(' '))
    const child = spawn(command, args, {
      cwd: repoRoot,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      windowsHide: true
    })

    const timer = setTimeout(() => {
      console.error('H011_EXTERNAL_TIMEOUT ' + JSON.stringify({ pid: child.pid, timeoutMs }))
      killTree(child.pid)
      reject(new Error('timeout after ' + timeoutMs + 'ms'))
    }, timeoutMs)

    child.stdout.on('data', chunk => process.stdout.write(chunk))
    child.stderr.on('data', chunk => process.stderr.write(chunk))
    child.on('error', error => {
      clearTimeout(timer)
      reject(error)
    })
    child.on('exit', (code, signal) => {
      clearTimeout(timer)
      if (allowedCodes.includes(code)) resolve({ code, signal, pid: child.pid })
      else reject(new Error('exit code ' + code + ', signal ' + (signal ?? 'none')))
    })
  })
}

function startLocalPageServer() {
  const server = http.createServer((req, res) => {
    const requestUrl = new URL(req.url || '/', 'http://127.0.0.1')
    const task = requestUrl.searchParams.get('task') || 'unknown'
    const cycle = requestUrl.searchParams.get('cycle') || '0'
    const title = `Hermes H011 ${task} cycle ${cycle}`
    const html =
      '<!doctype html><html><head><meta charset="utf-8"><title>' +
      title +
      '</title></head><body><main><h1>Hermes H011 Browser runtime soak</h1><p data-task="' +
      task +
      '">' +
      task +
      '</p><p data-cycle="' +
      cycle +
      '">' +
      cycle +
      '</p></main></body></html>'
    res.writeHead(200, {
      'content-type': 'text/html; charset=utf-8',
      'content-length': Buffer.byteLength(html),
      'cache-control': 'no-store'
    })
    res.end(html)
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        reject(new Error('native H011 page failed to bind'))
        return
      }
      resolve({
        baseUrl: 'http://127.0.0.1:' + address.port,
        close: () => new Promise(done => server.close(() => done()))
      })
    })
  })
}

const harnessSource = String.raw`
import fs from 'node:fs'
import path from 'node:path'
import { app, BrowserWindow } from 'electron'

const episode = Number(process.env.H011_EPISODE || 0)
const home = process.env.HERMES_WORKSTATION_HOME
const baseUrl = process.env.H011_BASE_URL
const taskCount = Number(process.env.H011_TASK_COUNT || 4)
const iterations = Number(process.env.H011_CHILD_ITERATIONS || 24)
if (!home || !baseUrl || !Number.isInteger(taskCount) || !Number.isInteger(iterations)) {
  console.error('H011_HARNESS_CONFIG_FAIL')
  process.exit(2)
}

fs.mkdirSync(home, { recursive: true })
app.setPath('userData', path.join(home, 'ElectronHostUserData-' + episode))

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
async function shutdown(runtime: any, win: BrowserWindow): Promise<never> {
  try { await runtime.destroy() } catch {}
  if (!win.isDestroyed()) win.destroy()
  app.exit(0)
  await new Promise(() => {})
  throw new Error('unreachable')
}

const timer = setTimeout(() => {
  console.error('H011_INTERNAL_TIMEOUT', JSON.stringify({ pid: process.pid, episode }))
  app.exit(9)
}, 50_000)

app.whenReady().then(async () => {
  const runtimeModule = await import(${JSON.stringify(runtimePath)})
  const { getWorkstationBrowserRuntime } = runtimeModule
  const runtime = getWorkstationBrowserRuntime()
  await runtime.startControlServer()
  const win = new BrowserWindow({ width: 1100, height: 760, show: false, title: 'H011 hidden host' })
  const taskIds = Array.from({ length: taskCount }, (_, index) => 'h011-task-' + (index + 1))

  try {
    runtime.ensure()
    const restored = runtime.listTasks()
    if (episode > 0) {
      assert(restored.length === taskCount, 'episode ' + episode + ' restored ' + restored.length + ' tasks')
      assert(restored.every((task: any) => task.status === 'parked' && task.recoveryState === 'restored'), 'episode ' + episode + ' did not restore all tasks parked')
    }

    if (episode === 0) {
      for (const taskId of taskIds) {
        runtime.createTask({
          taskId,
          sessionHost: 'h011-session-' + taskId,
          kanbanCardId: 'h011-card-' + taskId,
          runId: 'h011-run-' + taskId
        })
      }
    } else {
      for (const taskId of taskIds) {
        assert(runtime.listTasks().some((task: any) => task.taskId === taskId), 'restored task id missing: ' + taskId)
      }
      // Restart recovery is intentionally lazy: the assertions above prove
      // that metadata came back parked before any task page is recreated.
      // Warm every task only after that boundary so the rest of this episode
      // exercises the one-live-page-per-task multi-task pool.
      for (const taskId of taskIds) runtime.createTask({ taskId })
    }

    for (let cycle = 0; cycle < iterations; cycle += 1) {
      const taskId = taskIds[cycle % taskIds.length]
      const host = cycle % 2 === 0 ? 'hub' : 'chat'
      const bounds = { x: 0, y: 0, width: 900, height: 620 }
      runtime.showTask(taskId, win, bounds, host)
      const taskTab = runtime.state().tabs.find((tab: any) => tab.ownerTaskId === taskId)
      assert(taskTab, 'task tab missing for ' + taskId)
      const taskContents = runtime.getWebContents(taskTab.id)
      assert(taskContents && !taskContents.isDestroyed(), 'task WebContents missing for ' + taskId)
      const target = baseUrl + '/task?task=' + encodeURIComponent(taskId) + '&cycle=' + cycle
      await taskContents.loadURL(target)
      const page = await taskContents.executeJavaScript('({ title: document.title, task: document.querySelector("[data-task]")?.textContent, cycle: document.querySelector("[data-cycle]")?.textContent })', true)
      assert(page.title === 'Hermes H011 ' + taskId + ' cycle ' + cycle, 'page title mismatch for ' + taskId)
      assert(page.task === taskId && page.cycle === String(cycle), 'page sentinel mismatch for ' + taskId)

      const state = runtime.state()
      assert(state.controlReady, 'controller was not ready during cycle ' + cycle)
      assert(state.viewportHost === host, 'viewport host mismatch during cycle ' + cycle)
      assert(state.tabs.filter((tab: any) => tab.ownerTaskId === taskId).length === 1, 'duplicate task page during cycle ' + cycle)
      assert(state.tabs.filter((tab: any) => tab.ownerTaskId).length === taskCount, 'task page ownership count changed during cycle ' + cycle)
      const resources = runtime.resources()
      const resource = resources.resources.find((candidate: any) => candidate.resource_id === 'browser-task:' + taskId)
      assert(resource && resource.task_id === taskId, 'resource identity mismatch during cycle ' + cycle)
      assert(resource.session_id === 'h011-session-' + taskId, 'resource session lineage mismatch during cycle ' + cycle)
      assert(resource.state.tab_id === taskTab.id, 'resource tab identity mismatch during cycle ' + cycle)

      if (cycle % 3 === 0) runtime.hideTask(taskId)
      else runtime.parkTask(taskId)
      await sleep(20)
    }

    const finalState = runtime.state()
    assert(finalState.tasks.length === taskCount, 'final task count changed')
    assert(finalState.tabs.filter((tab: any) => tab.ownerTaskId).length === taskCount, 'final owned page count changed')
    console.log('H011_EPISODE_PASS', JSON.stringify({ episode, pid: process.pid, iterations, taskCount }))
    clearTimeout(timer)
    await shutdown(runtime, win)
  } catch (error) {
    console.error('H011_EPISODE_FAIL', JSON.stringify({ episode, pid: process.pid, error: String(error) }))
    clearTimeout(timer)
    try { await runtime.destroy() } catch {}
    if (!win.isDestroyed()) win.destroy()
    app.exit(1)
  }
}).catch(error => {
  console.error('H011_HARNESS_FAIL', String(error))
  clearTimeout(timer)
  app.exit(1)
})
`

fs.writeFileSync(harnessTs, harnessSource, 'utf8')
fs.writeFileSync(packageJson, JSON.stringify({ name: 'hermes-h011-native-harness', version: '1.0.0', main: 'main.cjs' }, null, 2), 'utf8')

console.log('H011_PROBE_CONTEXT', JSON.stringify({
  head,
  durationMs,
  taskCount,
  childIterations,
  electronExe,
  esbuildCli,
  platform: process.platform,
  osRelease: os.release(),
  node: process.version
}))

await run(process.execPath, [esbuildCli, harnessTs, '--bundle', '--platform=node', '--format=cjs', '--target=node20', '--outfile=' + mainCjs, '--external:electron'])

const pageServer = await startLocalPageServer()
const deadline = Date.now() + durationMs
let episode = 0
let totalIterations = 0
try {
  while (Date.now() < deadline || episode < 2) {
    const childTimeout = Math.max(60_000, Math.min(120_000, durationMs + 30_000))
    await run(electronExe, [appDir], {
      timeoutMs: childTimeout,
      env: {
        ...process.env,
        H011_EPISODE: String(episode),
        H011_BASE_URL: pageServer.baseUrl,
        H011_TASK_COUNT: String(taskCount),
        H011_CHILD_ITERATIONS: String(childIterations),
        HERMES_HOME: workstationHome,
        HERMES_WORKSTATION_HOME: workstationHome,
        HERMES_WORKSTATION_BROWSER_PROFILE: browserProfile,
        HERMES_WORKSTATION_BROWSER_CONTROL_FILE: path.join(workstationHome, 'Runtime', 'browser-control.json')
      }
    })

    const sessionStatePath = path.join(workstationHome, 'Runtime', 'browser-session.json')
    if (!fs.existsSync(sessionStatePath)) throw new Error('durable composite BrowserSessionState missing after episode ' + episode)
    const snapshot = JSON.parse(fs.readFileSync(sessionStatePath, 'utf8'))
    if (!snapshot.browserTasks || !Array.isArray(snapshot.browserTasks.tasks) || snapshot.browserTasks.tasks.length !== taskCount) {
      throw new Error('durable BrowserTask count mismatch after episode ' + episode)
    }
    if (new Set(snapshot.browserTasks.tasks.map(task => task.taskId)).size !== taskCount) {
      throw new Error('duplicate durable BrowserTask ids after episode ' + episode)
    }

    totalIterations += childIterations
    episode += 1
  }
} finally {
  await pageServer.close()
}

const summary = {
  head,
  durationMs,
  episodes: episode,
  taskCount,
  totalIterations,
  durableState: path.join(workstationHome, 'Runtime', 'browser-session.json'),
  workRoot: tempRoot
}
if (reportPath) {
  fs.mkdirSync(path.dirname(reportPath), { recursive: true })
  fs.writeFileSync(reportPath, JSON.stringify({ accepted: true, ...summary }, null, 2) + '\n', 'utf8')
}
console.log('H011_SUMMARY', JSON.stringify(summary))
console.log('H011_NATIVE_BROWSER_RUNTIME_CLASSIFICATION=VALIDATED')
