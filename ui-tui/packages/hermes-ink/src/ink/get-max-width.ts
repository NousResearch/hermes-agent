import { LayoutEdge, type LayoutNode } from './layout/node.js'

/**
 * Returns the yoga node's content width (computed width minus padding and
 * border).
 *
 * Warning: can return a value WIDER than the parent container. In a
 * column-direction flex parent, width is the cross axis — align-items:
 * stretch never shrinks children below their intrinsic size, so the text
 * node overflows (standard CSS behavior). Yoga measures leaf nodes in two
 * passes: the AtMost pass determines width, the Exactly pass determines
 * height. getComputedWidth() reflects the wider AtMost result while
 * getComputedHeight() reflects the narrower Exactly result. Callers that
 * use this for wrapping should clamp to actual available screen space so
 * the rendered line count stays consistent with the layout height.
 */
const getMaxWidth = (yogaNode: LayoutNode): number => {
  return (
    yogaNode.getComputedWidth() -
    yogaNode.getComputedPadding(LayoutEdge.Left) -
    yogaNode.getComputedPadding(LayoutEdge.Right) -
    yogaNode.getComputedBorder(LayoutEdge.Left) -
    yogaNode.getComputedBorder(LayoutEdge.Right)
  )
}

export default getMaxWidth
