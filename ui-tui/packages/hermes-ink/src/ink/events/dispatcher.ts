import {
  ContinuousEventPriority,
  DefaultEventPriority,
  DiscreteEventPriority,
  NoEventPriority
} from 'react-reconciler/constants.js'

import { logError } from '../../utils/log.js'

import { HANDLER_FOR_EVENT } from './event-handlers.js'
import type { EventTarget, TerminalEvent } from './terminal-event.js'

// --

type DispatchListener = {
  node: EventTarget
  handler: (event: TerminalEvent) => void
  phase: 'capturing' | 'at_target' | 'bubbling'
}

function getHandler(
  node: EventTarget,
  eventType: string,
  capture: boolean
): ((event: TerminalEvent) => void) | undefined {
  const handlers = node._eventHandlers

  if (!handlers) {
    return undefined
  }

  const mapping = HANDLER_FOR_EVENT[eventType]

  if (!mapping) {
    return undefined
  }

  const propName = capture ? mapping.capture : mapping.bubble

  if (!propName) {
    return undefined
  }

  return handlers[propName] as ((event: TerminalEvent) => void) | undefined
}

/**
 * Collect all listeners for an event in dispatch order.
 *
 * Uses react-dom's two-phase accumulation pattern:
 * - Walk from target to root
 * - Capture handlers are prepended (unshift) → root-first
 * - Bubble handlers are appended (push) → target-first
 *
 * Result: [root-cap, ..., parent-cap, target-cap, target-bub, parent-bub, ..., root-bub]
 */
function collectListeners(target: EventTarget, event: TerminalEvent): DispatchListener[] {
  const listeners: DispatchListener[] = []

  let node: EventTarget | undefined = target

  while (node) {
    const isTarget = node === target

    const captureHandler = getHandler(node, event.type, true)
    const bubbleHandler = getHandler(node, event.type, false)

    if (captureHandler) {
      listeners.unshift({
        node,
        handler: captureHandler,
        phase: isTarget ? 'at_target' : 'capturing'
      })
    }

    if (bubbleHandler && (event.bubbles || isTarget)) {
      listeners.push({
        node,
        handler: bubbleHandler,
        phase: isTarget ? 'at_target' : 'bubbling'
      })
    }

    node = node.parentNode
  }

  return listeners
}

/**
 * Execute collected listeners with propagation control.
 *
 * Before each handler, calls event._prepareForTarget(node) so event
 * subclasses can do per-node setup.
 */
function processDispatchQueue(listeners: DispatchListener[], event: TerminalEvent): void {
  let previousNode: EventTarget | undefined

  for (const { node, handler, phase } of listeners) {
    if (event._isImmediatePropagationStopped()) {
      break
    }

    if (event._isPropagationStopped() && node !== previousNode) {
      break
    }

    event._setEventPhase(phase)
    event._setCurrentTarget(node)
    event._prepareForTarget(node)

    try {
      handler(event)
    } catch (error) {
      logError(error)
    }

    previousNode = node
  }
}

// --

/**
 * Map terminal event types to React scheduling priorities.
 * Mirrors react-dom's getEventPriority() switch.
 */
function getEventPriority(eventType: string): number {
  switch (eventType) {
    case 'keydown':

    case 'keyup':

    case 'click':

    case 'focus':

    case 'blur':

    case 'paste':
      return DiscreteEventPriority as number

    case 'resize':

    case 'scroll':

    case 'mousemove':
      return ContinuousEventPriority as number

    default:
      return DefaultEventPriority as number
  }
}

// --

type DiscreteUpdates = <A, B>(fn: (a: A, b: B) => boolean, a: A, b: B, c: undefined, d: undefined) => boolean

/**
 * Owns event dispatch state and the capture/bubble dispatch loop.
 *
 * The reconciler host config reads currentEvent and currentUpdatePriority
 * to implement resolveUpdatePriority, resolveEventType, and
 * resolveEventTimeStamp — mirroring how react-dom's host config reads
 * ReactDOMSharedInternals and window.event.
 *
 * discreteUpdates is injected after construction (by InkReconciler)
 * to break the import cycle.
 */
export class Dispatcher {
  currentEvent: TerminalEvent | null = null
  currentUpdatePriority: number = DefaultEventPriority as number
  discreteUpdates: DiscreteUpdates | null = null

  /**
   * Infer event priority from the currently-dispatching event.
   * Called by the reconciler host config's resolveUpdatePriority
   * when no explicit priority has been set.
   */
  resolveEventPriority(): number {
    if (this.currentUpdatePriority !== (NoEventPriority as number)) {
      return this.currentUpdatePriority
    }

    if (this.currentEvent) {
      return getEventPriority(this.currentEvent.type)
    }

    return DefaultEventPriority as number
  }

  /**
   * Dispatch an event through capture and bubble phases.
   * Returns true if preventDefault() was NOT called.
   */
  dispatch(target: EventTarget, event: TerminalEvent): boolean {
    const previousEvent = this.currentEvent
    this.currentEvent = event

    try {
      event._setTarget(target)

      const listeners = collectListeners(target, event)
      processDispatchQueue(listeners, event)

      event._setEventPhase('none')
      event._setCurrentTarget(null)

      return !event.defaultPrevented
    } finally {
      this.currentEvent = previousEvent
    }
  }

  /**
   * Dispatch with discrete (sync) priority.
   * For user-initiated events: keyboard, click, focus, paste.
   */
  dispatchDiscrete(target: EventTarget, event: TerminalEvent): boolean {
    if (!this.discreteUpdates) {
      return this.dispatch(target, event)
    }

    return this.discreteUpdates((t, e) => this.dispatch(t, e), target, event, undefined, undefined)
  }

  /**
   * Dispatch with continuous priority.
   * For high-frequency events: resize, scroll, mouse move.
   */
  dispatchContinuous(target: EventTarget, event: TerminalEvent): boolean {
    const previousPriority = this.currentUpdatePriority

    try {
      this.currentUpdatePriority = ContinuousEventPriority as number

      return this.dispatch(target, event)
    } finally {
      this.currentUpdatePriority = previousPriority
    }
  }
}
