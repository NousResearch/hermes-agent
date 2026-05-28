import { type EventTarget, TerminalEvent } from './terminal-event.js'

/**
 * Focus event for component focus changes.
 *
 * Dispatched when focus moves between elements. 'focus' fires on the
 * newly focused element, 'blur' fires on the previously focused one.
 * Both bubble, matching react-dom's use of focusin/focusout semantics
 * so parent components can observe descendant focus changes.
 */
export class FocusEvent extends TerminalEvent {
  readonly relatedTarget: EventTarget | null

  constructor(type: 'focus' | 'blur', relatedTarget: EventTarget | null = null) {
    super(type, { bubbles: true, cancelable: false })
    this.relatedTarget = relatedTarget
  }
}
