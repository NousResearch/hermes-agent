import { Stream } from 'stream'

import type { ReactNode } from 'react'

import { logForDebugging } from '../utils/debug.js'

import type { FrameEvent } from './frame.js'
import Ink, { type Options as InkOptions } from './ink.js'
import instances from './instances.js'

export type RenderOptions = {
  /**
   * Output stream where app will be rendered.
   *
   * @default process.stdout
   */
  stdout?: NodeJS.WriteStream
  /**
   * Input stream where app will listen for input.
   *
   * @default process.stdin
   */
  stdin?: NodeJS.ReadStream
  /**
   * Error stream.
   * @default process.stderr
   */
  stderr?: NodeJS.WriteStream
  /**
   * Configure whether Ink should listen to Ctrl+C keyboard input and exit the app. This is needed in case `process.stdin` is in raw mode, because then Ctrl+C is ignored by default and process is expected to handle it manually.
   *
   * @default true
   */
  exitOnCtrlC?: boolean

  /**
   * Patch console methods to ensure console output doesn't mix with Ink output.
   *
   * @default true
   */
  patchConsole?: boolean

  /**
   * Called after each frame render with timing and flicker information.
   */
  onFrame?: (event: FrameEvent) => void
}

export type Instance = {
  /**
   * Replace previous root node with a new one or update props of the current root node.
   */
  rerender: Ink['render']
  /**
   * Manually unmount the whole Ink app.
   */
  unmount: Ink['unmount']
  /**
   * Returns a promise, which resolves when app is unmounted.
   */
  waitUntilExit: Ink['waitUntilExit']
  cleanup: () => void
}

/**
 * A managed Ink root, similar to react-dom's createRoot API.
 * Separates instance creation from rendering so the same root
 * can be reused for multiple sequential screens.
 */
export type Root = {
  render: (node: ReactNode) => void
  unmount: () => void
  waitUntilExit: () => Promise<void>
}

export const forceRedraw = (stdout: NodeJS.WriteStream = process.stdout): boolean => {
  const instance = instances.get(stdout)

  if (!instance) {
    return false
  }

  instance.forceRedraw()

  return true
}

/**
 * Mount a component and render the output.
 */
export const renderSync = (node: ReactNode, options?: NodeJS.WriteStream | RenderOptions): Instance => {
  const opts = getOptions(options)

  const inkOptions: InkOptions = {
    stdout: process.stdout,
    stdin: process.stdin,
    stderr: process.stderr,
    exitOnCtrlC: true,
    patchConsole: true,
    ...opts
  }

  const instance: Ink = getInstance(inkOptions.stdout, () => new Ink(inkOptions))

  instance.render(node)

  return {
    rerender: instance.render,
    unmount() {
      instance.unmount()
    },
    waitUntilExit: instance.waitUntilExit,
    cleanup: () => instances.delete(inkOptions.stdout)
  }
}

const wrappedRender = async (node: ReactNode, options?: NodeJS.WriteStream | RenderOptions): Promise<Instance> => {
  // Preserve the microtask boundary that `await loadYoga()` used to provide.
  // Without it, the first render fires synchronously before async startup work
  // (e.g. useReplBridge notification state) settles, and the subsequent Static
  // write overwrites scrollback instead of appending below the logo.
  await Promise.resolve()
  const instance = renderSync(node, options)
  logForDebugging(`[render] first ink render: ${Math.round(process.uptime() * 1000)}ms since process start`)

  return instance
}

export default wrappedRender

/**
 * Create an Ink root without rendering anything yet.
 * Like react-dom's createRoot — call root.render() to mount a tree.
 */
export async function createRoot({
  stdout = process.stdout,
  stdin = process.stdin,
  stderr = process.stderr,
  exitOnCtrlC = true,
  patchConsole = true,
  onFrame
}: RenderOptions = {}): Promise<Root> {
  // See wrappedRender — preserve microtask boundary from the old WASM await.
  await Promise.resolve()

  const instance = new Ink({
    stdout,
    stdin,
    stderr,
    exitOnCtrlC,
    patchConsole,
    onFrame
  })

  // Register in the instances map so that code that looks up the Ink
  // instance by stdout (e.g. external editor pause/resume) can find it.
  instances.set(stdout, instance)

  return {
    render: node => instance.render(node),
    unmount: () => instance.unmount(),
    waitUntilExit: () => instance.waitUntilExit()
  }
}

const getOptions = (stdout: NodeJS.WriteStream | RenderOptions | undefined = {}): RenderOptions => {
  if (stdout instanceof Stream) {
    return {
      stdout,
      stdin: process.stdin
    }
  }

  return stdout
}

const getInstance = (stdout: NodeJS.WriteStream, createInstance: () => Ink): Ink => {
  let instance = instances.get(stdout)

  if (!instance) {
    instance = createInstance()
    instances.set(stdout, instance)
  }

  return instance
}
