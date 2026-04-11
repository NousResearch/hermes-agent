export type Point = {
  x: number
  y: number
}

export type Size = {
  width: number
  height: number
}

export type Rectangle = Point & Size

/** Edge insets (padding, margin, border) */
export type Edges = {
  top: number
  right: number
  bottom: number
  left: number
}

/** Create uniform edges */
export function edges(all: number): Edges
export function edges(vertical: number, horizontal: number): Edges
export function edges(top: number, right: number, bottom: number, left: number): Edges

export function edges(a: number, b?: number, c?: number, d?: number): Edges {
  if (b === undefined) {
    return { top: a, right: a, bottom: a, left: a }
  }

  if (c === undefined) {
    return { top: a, right: b, bottom: a, left: b }
  }

  return { top: a, right: b, bottom: c, left: d! }
}

/** Add two edge values */
export function addEdges(a: Edges, b: Edges): Edges {
  return {
    top: a.top + b.top,
    right: a.right + b.right,
    bottom: a.bottom + b.bottom,
    left: a.left + b.left
  }
}

/** Zero edges constant */
export const ZERO_EDGES: Edges = { top: 0, right: 0, bottom: 0, left: 0 }

/** Convert partial edges to full edges with defaults */
export function resolveEdges(partial?: Partial<Edges>): Edges {
  return {
    top: partial?.top ?? 0,
    right: partial?.right ?? 0,
    bottom: partial?.bottom ?? 0,
    left: partial?.left ?? 0
  }
}

export function unionRect(a: Rectangle, b: Rectangle): Rectangle {
  const minX = Math.min(a.x, b.x)
  const minY = Math.min(a.y, b.y)
  const maxX = Math.max(a.x + a.width, b.x + b.width)
  const maxY = Math.max(a.y + a.height, b.y + b.height)

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}

export function clampRect(rect: Rectangle, size: Size): Rectangle {
  const minX = Math.max(0, rect.x)
  const minY = Math.max(0, rect.y)
  const maxX = Math.min(size.width - 1, rect.x + rect.width - 1)
  const maxY = Math.min(size.height - 1, rect.y + rect.height - 1)

  return {
    x: minX,
    y: minY,
    width: Math.max(0, maxX - minX + 1),
    height: Math.max(0, maxY - minY + 1)
  }
}

export function withinBounds(size: Size, point: Point): boolean {
  return point.x >= 0 && point.y >= 0 && point.x < size.width && point.y < size.height
}

export function clamp(value: number, min?: number, max?: number): number {
  if (min !== undefined && value < min) {
    return min
  }

  if (max !== undefined && value > max) {
    return max
  }

  return value
}
