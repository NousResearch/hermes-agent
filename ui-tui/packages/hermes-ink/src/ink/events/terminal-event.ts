import { Event } from './event.js'

type EventPhase = 'none' | 'capturing' | 'at_target' | 'bubbling'

type TerminalEventInit = {
  bubbles?: boolean
  cancelable?: boolean
}

/**
 * Base class for all terminal events with DOM-style propagation.
 *
 * Extends Event so existing event types (ClickEvent, InputEvent,
 * TerminalFocusEvent) share a common ancestor and can migrate later.
 *
 * Mirrors the browser's Event API: target, currentTarget, eventPhase,
 * stopPropagation(), preventDefault(), timeStamp.
 */
export class TerminalEvent extends Event {
  readonly type: string
  readonly timeStamp: number
  readonly bubbles: boolean
  readonly cancelable: boolean

  private _target: EventTarget | null = null
  private _currentTarget: EventTarget | null = null
  private _eventPhase: EventPhase = 'none'
  private _propagationStopped = false
  private _defaultPrevented = false

  constructor(type: string, init?: TerminalEventInit) {
    super()
    this.type = type
    this.timeStamp = performance.now()
    this.bubbles = init?.bubbles ?? true
    this.cancelable = init?.cancelable ?? true
  }

  get target(): EventTarget | null {
    return this._target
  }

  get currentTarget(): EventTarget | null {
    return this._currentTarget
  }

  get eventPhase(): EventPhase {
    return this._eventPhase
  }

  get defaultPrevented(): boolean {
    return this._defaultPrevented
  }

  stopPropagation(): void {
    this._propagationStopped = true
  }

  override stopImmediatePropagation(): void {
    super.stopImmediatePropagation()
    this._propagationStopped = true
  }

  preventDefault(): void {
    if (this.cancelable) {
      this._defaultPrevented = true
    }
  }

  // -- Internal setters used by the Dispatcher

  /** @internal */
  _setTarget(target: EventTarget): void {
    this._target = target
  }

  /** @internal */
  _setCurrentTarget(target: EventTarget | null): void {
    this._currentTarget = target
  }

  /** @internal */
  _setEventPhase(phase: EventPhase): void {
    this._eventPhase = phase
  }

  /** @internal */
  _isPropagationStopped(): boolean {
    return this._propagationStopped
  }

  /** @internal */
  _isImmediatePropagationStopped(): boolean {
    return this.didStopImmediatePropagation()
  }

  /**
   * Hook for subclasses to do per-node setup before each handler fires.
   * Default is a no-op.
   */
  _prepareForTarget(_target: EventTarget): void {}
}

export type EventTarget = {
  parentNode: EventTarget | undefined
  _eventHandlers?: Record<string, unknown>
}
