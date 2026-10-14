import createReconciler from 'react-reconciler'

import {
  appendChildNode,
  clearYogaNodeReferences,
  createNode,
  createTextNode,
  type DOMElement,
  type DOMNodeAttribute,
  type ElementNames,
  insertBeforeNode,
  markDirty,
  removeChildNode,
  setAttribute,
  setStyle,
  setTextNodeValue,
  setTextStyles,
  type TextNode
} from './dom.js'
import { Dispatcher } from './events/dispatcher.js'
import { EVENT_HANDLER_PROPS } from './events/event-handlers.js'
import { getFocusManager, getRootNode } from './focus.js'
import { LayoutDisplay } from './layout/node.js'
import applyStyles, { type Styles, type TextStyles } from './styles.js'

// We need to conditionally perform devtools connection to avoid
// accidentally breaking other third-party code.
// See https://github.com/vadimdemedes/ink/issues/384
if (process.env.NODE_ENV === 'development') {
  try {
    void import('./devtools.js')
  } catch (error: any) {
    if (error.code === 'ERR_MODULE_NOT_FOUND') {
      // biome-ignore lint/suspicious/noConsole: intentional warning
      console.warn(
        `
The environment variable DEV is set to true, so Ink tried to import \`react-devtools-core\`,
but this failed as it was not installed. Debugging with React Devtools requires it.

To install use this command:

$ npm install --save-dev react-devtools-core
				`.trim() + '\n'
      )
    } else {
      throw error
    }
  }
}

// --

type AnyObject = Record<string, unknown>

const diff = (before: AnyObject, after: AnyObject): AnyObject | undefined => {
  if (before === after) {
    return
  }

  if (!before) {
    return after
  }

  const changed: AnyObject = {}
  let isChanged = false

  for (const key of Object.keys(before)) {
    const isDeleted = after ? !Object.hasOwn(after, key) : true

    if (isDeleted) {
      changed[key] = undefined
      isChanged = true
    }
  }

  if (after) {
    for (const key of Object.keys(after)) {
      if (after[key] !== before[key]) {
        changed[key] = after[key]
        isChanged = true
      }
    }
  }

  return isChanged ? changed : undefined
}

const cleanupYogaNode = (node: DOMElement | TextNode): void => {
  const yogaNode = node.yogaNode

  if (yogaNode) {
    yogaNode.unsetMeasureFunc()
    // Clear all references BEFORE freeing to prevent other code from
    // accessing freed WASM memory during concurrent operations
    clearYogaNodeReferences(node)
    yogaNode.freeRecursive()
  }
}

// --

type Props = Record<string, unknown>

type HostContext = {
  isInsideText: boolean
}

function setEventHandler(node: DOMElement, key: string, value: unknown): void {
  if (!node._eventHandlers) {
    node._eventHandlers = {}
  }

  node._eventHandlers[key] = value
}

function applyProp(node: DOMElement, key: string, value: unknown): void {
  if (key === 'children') {
    return
  }

  if (key === 'style') {
    setStyle(node, value as Styles)

    if (node.yogaNode) {
      applyStyles(node.yogaNode, value as Styles)
    }

    return
  }

  if (key === 'textStyles') {
    node.textStyles = value as TextStyles

    return
  }

  if (EVENT_HANDLER_PROPS.has(key)) {
    setEventHandler(node, key, value)

    return
  }

  setAttribute(node, key, value as DOMNodeAttribute)
}

// --

export const dispatcher = new Dispatcher()

// --- SCROLL PROFILING (bench/scroll-e2e.sh reads via getLastYogaMs) ---
// Set by onComputeLayout wrapper in ink.tsx; read by onRender for phases.
let _lastYogaMs = 0
let _lastCommitMs = 0
let _commitStart = 0

export function recordYogaMs(ms: number): void {
  _lastYogaMs = ms
}

export function getLastYogaMs(): number {
  return _lastYogaMs
}

export function markCommitStart(): void {
  _commitStart = performance.now()
}

export function getLastCommitMs(): number {
  return _lastCommitMs
}

export function resetProfileCounters(): void {
  _lastYogaMs = 0
  _lastCommitMs = 0
  _commitStart = 0
}
// --- END ---

const reconciler = createReconciler({
  getRootHostContext: () => ({ isInsideText: false }),
  prepareForCommit: () => null,
  preparePortalMount: () => null,
  clearContainer: () => false,
  resetAfterCommit(rootNode: DOMElement) {
    _lastCommitMs = _commitStart > 0 ? performance.now() - _commitStart : 0
    _commitStart = 0

    if (typeof rootNode.onComputeLayout === 'function') {
      rootNode.onComputeLayout()
    }

    if (process.env.NODE_ENV === 'test') {
      if (rootNode.childNodes.length === 0 && rootNode.hasRenderedContent) {
        return
      }

      if (rootNode.childNodes.length > 0) {
        rootNode.hasRenderedContent = true
      }

      rootNode.onImmediateRender?.()

      return
    }

    rootNode.onRender?.()
  },
  getChildHostContext(parentHostContext: HostContext, type: ElementNames): HostContext {
    const previousIsInsideText = parentHostContext.isInsideText

    const isInsideText = type === 'ink-text' || type === 'ink-virtual-text' || type === 'ink-link'

    if (previousIsInsideText === isInsideText) {
      return parentHostContext
    }

    return { isInsideText }
  },
  shouldSetTextContent: () => false,
  createInstance(
    originalType: ElementNames,
    newProps: Props,
    _root: DOMElement,
    hostContext: HostContext,
    _internalHandle?: unknown
  ): DOMElement {
    if (hostContext.isInsideText && originalType === 'ink-box') {
      throw new Error(`<Box> can't be nested inside <Text> component`)
    }

    const type = originalType === 'ink-text' && hostContext.isInsideText ? 'ink-virtual-text' : originalType

    const node = createNode(type)

    for (const [key, value] of Object.entries(newProps)) {
      applyProp(node, key, value)
    }

    return node
  },
  createTextInstance(text: string, _root: DOMElement, hostContext: HostContext): TextNode {
    if (!hostContext.isInsideText) {
      throw new Error(`Text string "${text}" must be rendered inside <Text> component`)
    }

    return createTextNode(text)
  },
  resetTextContent() {},
  hideTextInstance(node: TextNode) {
    setTextNodeValue(node, '')
  },
  unhideTextInstance(node: TextNode, text: string) {
    setTextNodeValue(node, text)
  },
  getPublicInstance: (instance: DOMElement): DOMElement => instance,
  hideInstance(node: DOMElement) {
    node.isHidden = true
    node.yogaNode?.setDisplay(LayoutDisplay.None)
    markDirty(node)
  },
  unhideInstance(node: DOMElement) {
    node.isHidden = false
    node.yogaNode?.setDisplay(LayoutDisplay.Flex)
    markDirty(node)
  },
  appendInitialChild: appendChildNode,
  appendChild: appendChildNode,
  insertBefore: insertBeforeNode,
  finalizeInitialChildren(_node: DOMElement, _type: ElementNames, props: Props): boolean {
    return props['autoFocus'] === true
  },
  commitMount(node: DOMElement): void {
    getFocusManager(node).handleAutoFocus(node)
  },
  isPrimaryRenderer: true,
  supportsMutation: true,
  supportsPersistence: false,
  supportsHydration: false,
  scheduleTimeout: setTimeout,
  cancelTimeout: clearTimeout,
  noTimeout: -1,
  getCurrentUpdatePriority: () => dispatcher.currentUpdatePriority,
  beforeActiveInstanceBlur() {},
  afterActiveInstanceBlur() {},
  detachDeletedInstance() {},
  getInstanceFromNode: () => null,
  prepareScopeUpdate() {},
  getInstanceFromScope: () => null,
  appendChildToContainer: appendChildNode,
  insertInContainerBefore: insertBeforeNode,
  removeChildFromContainer(node: DOMElement, removeNode: DOMElement): void {
    removeChildNode(node, removeNode)
    cleanupYogaNode(removeNode)
    getFocusManager(node).handleNodeRemoved(removeNode, node)
  },
  // React 19 commitUpdate receives old and new props directly instead of an updatePayload
  commitUpdate(node: DOMElement, _type: ElementNames, oldProps: Props, newProps: Props): void {
    const props = diff(oldProps, newProps)
    const style = diff(oldProps['style'] as Styles, newProps['style'] as Styles)

    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (key === 'style') {
          setStyle(node, value as Styles)

          continue
        }

        if (key === 'textStyles') {
          setTextStyles(node, value as TextStyles)

          continue
        }

        if (EVENT_HANDLER_PROPS.has(key)) {
          setEventHandler(node, key, value)

          continue
        }

        setAttribute(node, key, value as DOMNodeAttribute)
      }
    }

    if (style && node.yogaNode) {
      applyStyles(node.yogaNode, style, newProps['style'] as Styles)
    }
  },
  commitTextUpdate(node: TextNode, _oldText: string, newText: string): void {
    setTextNodeValue(node, newText)
  },
  removeChild(node: DOMElement, removeNode: DOMElement | TextNode) {
    removeChildNode(node, removeNode)
    cleanupYogaNode(removeNode)

    if (removeNode.nodeName !== '#text') {
      const root = getRootNode(node)
      root.focusManager!.handleNodeRemoved(removeNode, root)
    }
  },
  // React 19 required methods
  maySuspendCommit(): boolean {
    return false
  },
  preloadInstance(): boolean {
    return true
  },
  startSuspendingCommit(): void {},
  suspendInstance(): void {},
  waitForCommitToBeReady(): null {
    return null
  },
  NotPendingTransition: null,
  HostTransitionContext: {
    $$typeof: Symbol.for('react.context'),
    _currentValue: null
  } as never,
  setCurrentUpdatePriority(newPriority: number): void {
    dispatcher.currentUpdatePriority = newPriority
  },
  resolveUpdatePriority(): number {
    return dispatcher.resolveEventPriority()
  },
  resetFormInstance(): void {},
  requestPostPaintCallback(): void {},
  shouldAttemptEagerTransition(): boolean {
    return false
  },
  trackSchedulerEvent(): void {},
  resolveEventType(): string | null {
    return dispatcher.currentEvent?.type ?? null
  },
  resolveEventTimeStamp(): number {
    return dispatcher.currentEvent?.timeStamp ?? -1.1
  }
})

// Wire the reconciler's discreteUpdates into the dispatcher.
// This breaks the import cycle: dispatcher.ts doesn't import reconciler.ts.
dispatcher.discreteUpdates = reconciler.discreteUpdates.bind(reconciler)

export default reconciler
