import type { ClickEvent } from './click-event.js'
import type { FocusEvent } from './focus-event.js'
import type { KeyboardEvent } from './keyboard-event.js'
import type { MouseEvent } from './mouse-event.js'
import type { PasteEvent } from './paste-event.js'
import type { ResizeEvent } from './resize-event.js'

type KeyboardEventHandler = (event: KeyboardEvent) => void
type FocusEventHandler = (event: FocusEvent) => void
type PasteEventHandler = (event: PasteEvent) => void
type ResizeEventHandler = (event: ResizeEvent) => void
type ClickEventHandler = (event: ClickEvent) => void
type MouseEventHandler = (event: MouseEvent) => void
type HoverEventHandler = () => void

/**
 * Props for event handlers on Box and other host components.
 *
 * Follows the React/DOM naming convention:
 * - onEventName: handler for bubble phase
 * - onEventNameCapture: handler for capture phase
 */
export type EventHandlerProps = {
  onKeyDown?: KeyboardEventHandler
  onKeyDownCapture?: KeyboardEventHandler

  onFocus?: FocusEventHandler
  onFocusCapture?: FocusEventHandler
  onBlur?: FocusEventHandler
  onBlurCapture?: FocusEventHandler

  onPaste?: PasteEventHandler
  onPasteCapture?: PasteEventHandler

  onResize?: ResizeEventHandler

  onClick?: ClickEventHandler
  onMouseDown?: MouseEventHandler
  onMouseUp?: MouseEventHandler
  onMouseDrag?: MouseEventHandler
  onMouseEnter?: HoverEventHandler
  onMouseLeave?: HoverEventHandler
}

/**
 * Reverse lookup: event type string → handler prop names.
 * Used by the dispatcher for O(1) handler lookup per node.
 */
export const HANDLER_FOR_EVENT: Record<
  string,
  { bubble?: keyof EventHandlerProps; capture?: keyof EventHandlerProps }
> = {
  keydown: { bubble: 'onKeyDown', capture: 'onKeyDownCapture' },
  focus: { bubble: 'onFocus', capture: 'onFocusCapture' },
  blur: { bubble: 'onBlur', capture: 'onBlurCapture' },
  paste: { bubble: 'onPaste', capture: 'onPasteCapture' },
  resize: { bubble: 'onResize' },
  click: { bubble: 'onClick' },
  mousedown: { bubble: 'onMouseDown' },
  mouseup: { bubble: 'onMouseUp' },
  mousedrag: { bubble: 'onMouseDrag' }
}

/**
 * Set of all event handler prop names, for the reconciler to detect
 * event props and store them in _eventHandlers instead of attributes.
 */
export const EVENT_HANDLER_PROPS = new Set<string>([
  'onKeyDown',
  'onKeyDownCapture',
  'onFocus',
  'onFocusCapture',
  'onBlur',
  'onBlurCapture',
  'onPaste',
  'onPasteCapture',
  'onResize',
  'onClick',
  'onMouseDown',
  'onMouseUp',
  'onMouseDrag',
  'onMouseEnter',
  'onMouseLeave'
])
