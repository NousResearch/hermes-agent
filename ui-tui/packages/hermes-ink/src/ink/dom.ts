import type { FocusManager } from './focus.js'
import { createLayoutNode } from './layout/engine.js'
import type { LayoutNode } from './layout/node.js'
import { LayoutMeasureMode } from './layout/node.js'
import measureText from './measure-text.js'
import { addPendingClear, nodeCache } from './node-cache.js'
import squashTextNodes from './squash-text-nodes.js'
import type { Styles, TextStyles } from './styles.js'
import { expandTabs } from './tabstops.js'
import wrapText from './wrap-text.js'

type InkNode = {
  parentNode: DOMElement | undefined
  yogaNode?: LayoutNode
  style: Styles
}

export type TextName = '#text'
export type ElementNames =
  | 'ink-root'
  | 'ink-box'
  | 'ink-text'
  | 'ink-virtual-text'
  | 'ink-link'
  | 'ink-progress'
  | 'ink-raw-ansi'

export type NodeNames = ElementNames | TextName

export type DOMElement = {
  nodeName: ElementNames
  attributes: Record<string, DOMNodeAttribute>
  childNodes: DOMNode[]
  textStyles?: TextStyles

  // Internal properties
  onComputeLayout?: () => void
  onRender?: () => void
  onImmediateRender?: () => void
  // Used to skip empty renders during React 19's effect double-invoke in test mode
  hasRenderedContent?: boolean

  // When true, this node needs re-rendering
  dirty: boolean
  // Set by the reconciler's hideInstance/unhideInstance; survives style updates.
  isHidden?: boolean
  // Event handlers set by the reconciler for the capture/bubble dispatcher.
  // Stored separately from attributes so handler identity changes don't
  // mark dirty and defeat the blit optimization.
  _eventHandlers?: Record<string, unknown>

  // Scroll state for overflow: 'scroll' boxes. scrollTop is the number of
  // rows the content is scrolled down by. scrollHeight/scrollViewportHeight
  // are computed at render time and stored for imperative access. stickyScroll
  // auto-pins scrollTop to the bottom when content grows.
  scrollTop?: number
  // Accumulated scroll delta not yet applied to scrollTop. The renderer
  // drains this at SCROLL_MAX_PER_FRAME rows/frame so fast flicks show
  // intermediate frames instead of one big jump. Direction reversal
  // naturally cancels (pure accumulator, no target tracking).
  pendingScrollDelta?: number
  // Render-time clamp bounds for virtual scroll. useVirtualScroll writes
  // the currently-mounted children's coverage span; render-node-to-output
  // clamps scrollTop to stay within it. Prevents blank screen when
  // scrollTo's direct write races past React's async re-render — instead
  // of painting spacer (blank), the renderer holds at the edge of mounted
  // content until React catches up (next commit updates these bounds and
  // the clamp releases). Undefined = no clamp (sticky-scroll, cold start).
  scrollClampMin?: number
  scrollClampMax?: number
  scrollHeight?: number
  scrollViewportHeight?: number
  scrollViewportTop?: number
  stickyScroll?: boolean
  notifyScrollChange?: () => void
  // Set by ScrollBox.scrollToElement; render-node-to-output reads
  // el.yogaNode.getComputedTop() (FRESH — same Yoga pass as scrollHeight)
  // and sets scrollTop = top + offset, then clears this. Unlike an
  // imperative scrollTo(N) which bakes in a number that's stale by the
  // time the throttled render fires, the element ref defers the position
  // read to paint time. One-shot.
  scrollAnchor?: { el: DOMElement; offset: number }
  // Only set on ink-root. The document owns focus — any node can
  // reach it by walking parentNode, like browser getRootNode().
  focusManager?: FocusManager
  // Measurement cache for ink-text nodes: avoids re-squashing and re-wrapping
  // text when yoga calls measureFunc multiple times per frame with different
  // widths during flex re-pass. Keyed by `${width}|${widthMode}`.
  _textMeasureCache?: { gen: number; entries: Map<string, { _gen: number; result: { width: number; height: number } }> }
} & InkNode

export type TextNode = {
  nodeName: TextName
  nodeValue: string
} & InkNode

export type DOMNode<T = { nodeName: NodeNames }> = T extends {
  nodeName: infer U
}
  ? U extends '#text'
    ? TextNode
    : DOMElement
  : never

export type DOMNodeAttribute = boolean | string | number

export const createNode = (nodeName: ElementNames): DOMElement => {
  const needsYogaNode = nodeName !== 'ink-virtual-text' && nodeName !== 'ink-link' && nodeName !== 'ink-progress'

  const node: DOMElement = {
    nodeName,
    style: {},
    attributes: {},
    childNodes: [],
    parentNode: undefined,
    yogaNode: needsYogaNode ? createLayoutNode() : undefined,
    dirty: false
  }

  if (nodeName === 'ink-text') {
    node.yogaNode?.setMeasureFunc(measureTextNode.bind(null, node))
  } else if (nodeName === 'ink-raw-ansi') {
    node.yogaNode?.setMeasureFunc(measureRawAnsiNode.bind(null, node))
  }

  return node
}

export const appendChildNode = (node: DOMElement, childNode: DOMElement): void => {
  if (childNode.parentNode) {
    removeChildNode(childNode.parentNode, childNode)
  }

  childNode.parentNode = node
  node.childNodes.push(childNode)

  if (childNode.yogaNode) {
    node.yogaNode?.insertChild(childNode.yogaNode, node.yogaNode.getChildCount())
  }

  markDirty(node)
}

export const insertBeforeNode = (node: DOMElement, newChildNode: DOMNode, beforeChildNode: DOMNode): void => {
  if (newChildNode.parentNode) {
    removeChildNode(newChildNode.parentNode, newChildNode)
  }

  newChildNode.parentNode = node

  const index = node.childNodes.indexOf(beforeChildNode)

  if (index >= 0) {
    // Calculate yoga index BEFORE modifying childNodes.
    // We can't use DOM index directly because some children (like ink-progress,
    // ink-link, ink-virtual-text) don't have yogaNodes, so DOM indices don't
    // match yoga indices.
    let yogaIndex = 0

    if (newChildNode.yogaNode && node.yogaNode) {
      for (let i = 0; i < index; i++) {
        if (node.childNodes[i]?.yogaNode) {
          yogaIndex++
        }
      }
    }

    node.childNodes.splice(index, 0, newChildNode)

    if (newChildNode.yogaNode && node.yogaNode) {
      node.yogaNode.insertChild(newChildNode.yogaNode, yogaIndex)
    }

    markDirty(node)

    return
  }

  node.childNodes.push(newChildNode)

  if (newChildNode.yogaNode) {
    node.yogaNode?.insertChild(newChildNode.yogaNode, node.yogaNode.getChildCount())
  }

  markDirty(node)
}

export const removeChildNode = (node: DOMElement, removeNode: DOMNode): void => {
  if (removeNode.yogaNode) {
    removeNode.parentNode?.yogaNode?.removeChild(removeNode.yogaNode)
  }

  // Collect cached rects from the removed subtree so they can be cleared
  collectRemovedRects(node, removeNode)

  removeNode.parentNode = undefined

  const index = node.childNodes.indexOf(removeNode)

  if (index >= 0) {
    node.childNodes.splice(index, 1)
  }

  markDirty(node)
}

function collectRemovedRects(parent: DOMElement, removed: DOMNode, underAbsolute = false): void {
  if (removed.nodeName === '#text') {
    return
  }

  const elem = removed as DOMElement
  // If this node or any ancestor in the removed subtree was absolute,
  // its painted pixels may overlap non-siblings — flag for global blit
  // disable. Normal-flow removals only affect direct siblings, which
  // hasRemovedChild already handles.
  const isAbsolute = underAbsolute || elem.style.position === 'absolute'
  const cached = nodeCache.get(elem)

  if (cached) {
    addPendingClear(parent, cached, isAbsolute)
    nodeCache.delete(elem)
  }

  for (const child of elem.childNodes) {
    collectRemovedRects(parent, child, isAbsolute)
  }
}

export const setAttribute = (node: DOMElement, key: string, value: DOMNodeAttribute): void => {
  // Skip 'children' - React handles children via appendChild/removeChild,
  // not attributes. React always passes a new children reference, so
  // tracking it as an attribute would mark everything dirty every render.
  if (key === 'children') {
    return
  }

  // Skip if unchanged
  if (node.attributes[key] === value) {
    return
  }

  node.attributes[key] = value
  markDirty(node)
}

export const setStyle = (node: DOMNode, style: Styles): void => {
  // Compare style properties to avoid marking dirty unnecessarily.
  // React creates new style objects on every render even when unchanged.
  if (stylesEqual(node.style, style)) {
    return
  }

  node.style = style
  markDirty(node)
}

export const setTextStyles = (node: DOMElement, textStyles: TextStyles): void => {
  // Same dirty-check guard as setStyle: React (and buildTextStyles in Text.tsx)
  // allocate a new textStyles object on every render even when values are
  // unchanged, so compare by value to avoid markDirty -> yoga re-measurement
  // on every Text re-render.
  if (shallowEqual(node.textStyles, textStyles)) {
    return
  }

  node.textStyles = textStyles
  markDirty(node)
}

function stylesEqual(a: Styles, b: Styles): boolean {
  return shallowEqual(a, b)
}

function shallowEqual<T extends object>(a: T | undefined, b: T | undefined): boolean {
  // Fast path: same object reference (or both undefined)
  if (a === b) {
    return true
  }

  if (a === undefined || b === undefined) {
    return false
  }

  // Get all keys from both objects
  const aKeys = Object.keys(a) as (keyof T)[]
  const bKeys = Object.keys(b) as (keyof T)[]

  // Different number of properties
  if (aKeys.length !== bKeys.length) {
    return false
  }

  // Compare each property
  for (const key of aKeys) {
    if (a[key] !== b[key]) {
      return false
    }
  }

  return true
}

export const createTextNode = (text: string): TextNode => {
  const node: TextNode = {
    nodeName: '#text',
    nodeValue: text,
    yogaNode: undefined,
    parentNode: undefined,
    style: {}
  }

  setTextNodeValue(node, text)

  return node
}

const MEASURE_CACHE_CAP = 16

const measureTextNode = function (
  node: DOMNode,
  width: number,
  widthMode: LayoutMeasureMode
): { width: number; height: number } {
  const elem = node.nodeName !== '#text' ? (node as DOMElement) : node.parentNode

  if (elem && elem.nodeName === 'ink-text') {
    let cache = elem._textMeasureCache

    if (!cache) {
      cache = { gen: 0, entries: new Map() }
      elem._textMeasureCache = cache
    }

    const key = `${width}|${widthMode}`
    const hit = cache.entries.get(key)

    if (hit && hit._gen === cache.gen) {
      return hit.result
    }

    const result = computeTextMeasure(node, width, widthMode)

    // Enforce cap with FIFO eviction to avoid unbounded growth during
    // pathological frames where yoga probes many widths.
    if (cache.entries.size >= MEASURE_CACHE_CAP) {
      const firstKey = cache.entries.keys().next().value

      if (firstKey !== undefined) {
        cache.entries.delete(firstKey)
      }
    }

    cache.entries.set(key, { _gen: cache.gen, result })

    return result
  }

  return computeTextMeasure(node, width, widthMode)
}

const computeTextMeasure = function (
  node: DOMNode,
  width: number,
  widthMode: LayoutMeasureMode
): { width: number; height: number } {
  const rawText = node.nodeName === '#text' ? node.nodeValue : squashTextNodes(node)

  // Expand tabs for measurement (worst case: 8 spaces each).
  // Actual tab expansion happens in output.ts based on screen position.
  const text = expandTabs(rawText)

  const dimensions = measureText(text, width)

  // Text fits into container, no need to wrap
  if (dimensions.width <= width) {
    return dimensions
  }

  // This is happening when <Box> is shrinking child nodes and layout asks
  // if we can fit this text node in a <1px space, so we just say "no"
  if (dimensions.width >= 1 && width > 0 && width < 1) {
    return dimensions
  }

  // For text with embedded newlines (pre-wrapped content), avoid re-wrapping
  // at measurement width when layout is asking for intrinsic size (Undefined mode).
  // This prevents height inflation during min/max size checks.
  //
  // However, when layout provides an actual constraint (Exactly or AtMost mode),
  // we must respect it and measure at that width. Otherwise, if the actual
  // rendering width is smaller than the natural width, the text will wrap to
  // more lines than layout expects, causing content to be truncated.
  if (text.includes('\n') && widthMode === LayoutMeasureMode.Undefined) {
    const effectiveWidth = Math.max(width, dimensions.width)

    return measureText(text, effectiveWidth)
  }

  const textWrap = node.style?.textWrap ?? 'wrap'
  const wrappedText = wrapText(text, width, textWrap)

  return measureText(wrappedText, width)
}

// ink-raw-ansi nodes hold pre-rendered ANSI strings with known dimensions.
// No stringWidth, no wrapping, no tab expansion — the producer (e.g. ColorDiff)
// already wrapped to the target width and each line is exactly one terminal row.
const measureRawAnsiNode = function (node: DOMElement): {
  width: number
  height: number
} {
  return {
    width: node.attributes['rawWidth'] as number,
    height: node.attributes['rawHeight'] as number
  }
}

/**
 * Mark a node and all its ancestors as dirty for re-rendering.
 * Also marks yoga dirty for text remeasurement if this is a text node.
 */
export const markDirty = (node?: DOMNode): void => {
  let current: DOMNode | undefined = node
  let markedYoga = false

  while (current) {
    if (current.nodeName !== '#text') {
      const elem = current as DOMElement
      elem.dirty = true

      // Only mark yoga dirty on leaf nodes that have measure functions
      if (!markedYoga && (elem.nodeName === 'ink-text' || elem.nodeName === 'ink-raw-ansi') && elem.yogaNode) {
        elem.yogaNode.markDirty()
        markedYoga = true
      }

      // Invalidate text measurement cache — child text or style changed.
      if (elem._textMeasureCache) {
        elem._textMeasureCache.gen++
      }
    }

    current = current.parentNode
  }
}

// Walk to root and call its onRender (the throttled scheduleRender). Use for
// DOM-level mutations (scrollTop changes) that should trigger an Ink frame
// without going through React's reconciler. Pair with markDirty() so the
// renderer knows which subtree to re-evaluate.
export const scheduleRenderFrom = (node?: DOMNode): void => {
  let cur: DOMNode | undefined = node

  while (cur?.parentNode) {
    cur = cur.parentNode
  }

  if (cur && cur.nodeName !== '#text') {
    ;(cur as DOMElement).onRender?.()
  }
}

export const setTextNodeValue = (node: TextNode, text: string): void => {
  if (typeof text !== 'string') {
    text = String(text)
  }

  // Skip if unchanged
  if (node.nodeValue === text) {
    return
  }

  node.nodeValue = text
  markDirty(node)
}

function isDOMElement(node: DOMElement | TextNode): node is DOMElement {
  return node.nodeName !== '#text'
}

// Clear yogaNode references recursively before freeing.
// freeRecursive() frees the node and ALL its children, so we must clear
// all yogaNode references to prevent dangling pointers.
export const clearYogaNodeReferences = (node: DOMElement | TextNode): void => {
  if ('childNodes' in node) {
    for (const child of node.childNodes) {
      clearYogaNodeReferences(child)
    }

    node._textMeasureCache = undefined
  }

  node.yogaNode = undefined
}
