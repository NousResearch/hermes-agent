import type { DOMElement } from './dom.js'
import { FocusEvent } from './events/focus-event.js'

const MAX_FOCUS_STACK = 32

/**
 * DOM-like focus manager for the Ink terminal UI.
 *
 * Pure state — tracks activeElement and a focus stack. Has no reference
 * to the tree; callers pass the root when tree walks are needed.
 *
 * Stored on the root DOMElement so any node can reach it by walking
 * parentNode (like browser's `node.ownerDocument`).
 */
export class FocusManager {
  activeElement: DOMElement | null = null
  private dispatchFocusEvent: (target: DOMElement, event: FocusEvent) => boolean
  private enabled = true
  private focusStack: DOMElement[] = []

  constructor(dispatchFocusEvent: (target: DOMElement, event: FocusEvent) => boolean) {
    this.dispatchFocusEvent = dispatchFocusEvent
  }

  focus(node: DOMElement): void {
    if (node === this.activeElement) {
      return
    }

    if (!this.enabled) {
      return
    }

    const previous = this.activeElement

    if (previous) {
      // Deduplicate before pushing to prevent unbounded growth from Tab cycling
      const idx = this.focusStack.indexOf(previous)

      if (idx !== -1) {
        this.focusStack.splice(idx, 1)
      }

      this.focusStack.push(previous)

      if (this.focusStack.length > MAX_FOCUS_STACK) {
        this.focusStack.shift()
      }

      this.dispatchFocusEvent(previous, new FocusEvent('blur', node))
    }

    this.activeElement = node
    this.dispatchFocusEvent(node, new FocusEvent('focus', previous))
  }

  blur(): void {
    if (!this.activeElement) {
      return
    }

    const previous = this.activeElement
    this.activeElement = null
    this.dispatchFocusEvent(previous, new FocusEvent('blur', null))
  }

  /**
   * Called by the reconciler when a node is removed from the tree.
   * Handles both the exact node and any focused descendant within
   * the removed subtree. Dispatches blur and restores focus from stack.
   */
  handleNodeRemoved(node: DOMElement, root: DOMElement): void {
    // Remove the node and any descendants from the stack
    this.focusStack = this.focusStack.filter(n => n !== node && isInTree(n, root))

    // Check if activeElement is the removed node OR a descendant
    if (!this.activeElement) {
      return
    }

    if (this.activeElement !== node && isInTree(this.activeElement, root)) {
      return
    }

    const removed = this.activeElement
    this.activeElement = null
    this.dispatchFocusEvent(removed, new FocusEvent('blur', null))

    // Restore focus to the most recent still-mounted element
    while (this.focusStack.length > 0) {
      const candidate = this.focusStack.pop()!

      if (isInTree(candidate, root)) {
        this.activeElement = candidate
        this.dispatchFocusEvent(candidate, new FocusEvent('focus', removed))

        return
      }
    }
  }

  handleAutoFocus(node: DOMElement): void {
    this.focus(node)
  }

  handleClickFocus(node: DOMElement): void {
    const tabIndex = node.attributes['tabIndex']

    if (typeof tabIndex !== 'number') {
      return
    }

    this.focus(node)
  }

  enable(): void {
    this.enabled = true
  }

  disable(): void {
    this.enabled = false
  }

  focusNext(root: DOMElement): void {
    this.moveFocus(1, root)
  }

  focusPrevious(root: DOMElement): void {
    this.moveFocus(-1, root)
  }

  private moveFocus(direction: 1 | -1, root: DOMElement): void {
    if (!this.enabled) {
      return
    }

    const tabbable = collectTabbable(root)

    if (tabbable.length === 0) {
      return
    }

    const currentIndex = this.activeElement ? tabbable.indexOf(this.activeElement) : -1

    const nextIndex =
      currentIndex === -1
        ? direction === 1
          ? 0
          : tabbable.length - 1
        : (currentIndex + direction + tabbable.length) % tabbable.length

    const next = tabbable[nextIndex]

    if (next) {
      this.focus(next)
    }
  }
}

function collectTabbable(root: DOMElement): DOMElement[] {
  const result: DOMElement[] = []
  walkTree(root, result)

  return result
}

function walkTree(node: DOMElement, result: DOMElement[]): void {
  const tabIndex = node.attributes['tabIndex']

  if (typeof tabIndex === 'number' && tabIndex >= 0) {
    result.push(node)
  }

  for (const child of node.childNodes) {
    if (child.nodeName !== '#text') {
      walkTree(child, result)
    }
  }
}

function isInTree(node: DOMElement, root: DOMElement): boolean {
  let current: DOMElement | undefined = node

  while (current) {
    if (current === root) {
      return true
    }

    current = current.parentNode
  }

  return false
}

/**
 * Walk up to root and return it. The root is the node that holds
 * the FocusManager — like browser's `node.getRootNode()`.
 */
export function getRootNode(node: DOMElement): DOMElement {
  let current: DOMElement | undefined = node

  while (current) {
    if (current.focusManager) {
      return current
    }

    current = current.parentNode
  }

  throw new Error('Node is not in a tree with a FocusManager')
}

/**
 * Walk up to root and return its FocusManager.
 * Like browser's `node.ownerDocument` — focus belongs to the root.
 */
export function getFocusManager(node: DOMElement): FocusManager {
  return getRootNode(node).focusManager!
}
