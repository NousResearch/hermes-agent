import { Align, Direction, Display, Edge, FlexDirection, Justify, Overflow, PositionType, Unit, Wrap } from './enums.js'

export type Value = {
  unit: Unit
  value: number
}
const UNDEFINED_VALUE: Value = { unit: Unit.Undefined, value: NaN }
const AUTO_VALUE: Value = { unit: Unit.Auto, value: NaN }

function pointValue(v: number): Value {
  return { unit: Unit.Point, value: v }
}

function percentValue(v: number): Value {
  return { unit: Unit.Percent, value: v }
}

function resolveValue(v: Value, ownerSize: number): number {
  switch (v.unit) {
    case Unit.Point:
      return v.value

    case Unit.Percent:
      return isNaN(ownerSize) ? NaN : (v.value * ownerSize) / 100

    default:
      return NaN
  }
}

function isDefined(n: number): boolean {
  return !isNaN(n)
}

function sameFloat(a: number, b: number): boolean {
  return a === b || (a !== a && b !== b)
}

type Layout = {
  left: number
  top: number
  width: number
  height: number
  border: [number, number, number, number]
  padding: [number, number, number, number]
  margin: [number, number, number, number]
}
type Style = {
  direction: Direction
  flexDirection: FlexDirection
  justifyContent: Justify
  alignItems: Align
  alignSelf: Align
  alignContent: Align
  flexWrap: Wrap
  overflow: Overflow
  display: Display
  positionType: PositionType
  flexGrow: number
  flexShrink: number
  flexBasis: Value
  margin: Value[]
  padding: Value[]
  border: Value[]
  position: Value[]
  gap: Value[]
  width: Value
  height: Value
  minWidth: Value
  minHeight: Value
  maxWidth: Value
  maxHeight: Value
}

function defaultStyle(): Style {
  return {
    direction: Direction.Inherit,
    flexDirection: FlexDirection.Column,
    justifyContent: Justify.FlexStart,
    alignItems: Align.Stretch,
    alignSelf: Align.Auto,
    alignContent: Align.FlexStart,
    flexWrap: Wrap.NoWrap,
    overflow: Overflow.Visible,
    display: Display.Flex,
    positionType: PositionType.Relative,
    flexGrow: 0,
    flexShrink: 0,
    flexBasis: AUTO_VALUE,
    margin: new Array(9).fill(UNDEFINED_VALUE),
    padding: new Array(9).fill(UNDEFINED_VALUE),
    border: new Array(9).fill(UNDEFINED_VALUE),
    position: new Array(9).fill(UNDEFINED_VALUE),
    gap: new Array(3).fill(UNDEFINED_VALUE),
    width: AUTO_VALUE,
    height: AUTO_VALUE,
    minWidth: UNDEFINED_VALUE,
    minHeight: UNDEFINED_VALUE,
    maxWidth: UNDEFINED_VALUE,
    maxHeight: UNDEFINED_VALUE
  }
}

const EDGE_LEFT = 0
const EDGE_TOP = 1
const EDGE_RIGHT = 2
const EDGE_BOTTOM = 3

function resolveEdge(edges: Value[], physicalEdge: number, ownerSize: number, allowAuto = false): number {
  let v = edges[physicalEdge]!

  if (v.unit === Unit.Undefined) {
    if (physicalEdge === EDGE_LEFT || physicalEdge === EDGE_RIGHT) {
      v = edges[Edge.Horizontal]!
    } else {
      v = edges[Edge.Vertical]!
    }
  }

  if (v.unit === Unit.Undefined) {
    v = edges[Edge.All]!
  }

  if (v.unit === Unit.Undefined) {
    if (physicalEdge === EDGE_LEFT) {
      v = edges[Edge.Start]!
    }

    if (physicalEdge === EDGE_RIGHT) {
      v = edges[Edge.End]!
    }
  }

  if (v.unit === Unit.Undefined) {
    return 0
  }

  if (v.unit === Unit.Auto) {
    return allowAuto ? NaN : 0
  }

  return resolveValue(v, ownerSize)
}

function resolveEdgeRaw(edges: Value[], physicalEdge: number): Value {
  let v = edges[physicalEdge]!

  if (v.unit === Unit.Undefined) {
    if (physicalEdge === EDGE_LEFT || physicalEdge === EDGE_RIGHT) {
      v = edges[Edge.Horizontal]!
    } else {
      v = edges[Edge.Vertical]!
    }
  }

  if (v.unit === Unit.Undefined) {
    v = edges[Edge.All]!
  }

  if (v.unit === Unit.Undefined) {
    if (physicalEdge === EDGE_LEFT) {
      v = edges[Edge.Start]!
    }

    if (physicalEdge === EDGE_RIGHT) {
      v = edges[Edge.End]!
    }
  }

  return v
}

function isMarginAuto(edges: Value[], physicalEdge: number): boolean {
  return resolveEdgeRaw(edges, physicalEdge).unit === Unit.Auto
}

function hasAnyAutoEdge(edges: Value[]): boolean {
  for (let i = 0; i < 9; i++) {
    if (edges[i]!.unit === 3) {
      return true
    }
  }

  return false
}

function hasAnyDefinedEdge(edges: Value[]): boolean {
  for (let i = 0; i < 9; i++) {
    if (edges[i]!.unit !== 0) {
      return true
    }
  }

  return false
}

function resolveEdges4Into(edges: Value[], ownerSize: number, out: [number, number, number, number]): void {
  const eH = edges[6]!
  const eV = edges[7]!
  const eA = edges[8]!
  const eS = edges[4]!
  const eE = edges[5]!
  const pctDenom = isNaN(ownerSize) ? NaN : ownerSize / 100
  let v = edges[0]!

  if (v.unit === 0) {
    v = eH
  }

  if (v.unit === 0) {
    v = eA
  }

  if (v.unit === 0) {
    v = eS
  }

  out[0] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0
  v = edges[1]!

  if (v.unit === 0) {
    v = eV
  }

  if (v.unit === 0) {
    v = eA
  }

  out[1] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0
  v = edges[2]!

  if (v.unit === 0) {
    v = eH
  }

  if (v.unit === 0) {
    v = eA
  }

  if (v.unit === 0) {
    v = eE
  }

  out[2] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0
  v = edges[3]!

  if (v.unit === 0) {
    v = eV
  }

  if (v.unit === 0) {
    v = eA
  }

  out[3] = v.unit === 1 ? v.value : v.unit === 2 ? v.value * pctDenom : 0
}

function isRow(dir: FlexDirection): boolean {
  return dir === FlexDirection.Row || dir === FlexDirection.RowReverse
}

function isReverse(dir: FlexDirection): boolean {
  return dir === FlexDirection.RowReverse || dir === FlexDirection.ColumnReverse
}

function crossAxis(dir: FlexDirection): FlexDirection {
  return isRow(dir) ? FlexDirection.Column : FlexDirection.Row
}

function leadingEdge(dir: FlexDirection): number {
  switch (dir) {
    case FlexDirection.Row:
      return EDGE_LEFT

    case FlexDirection.RowReverse:
      return EDGE_RIGHT

    case FlexDirection.Column:
      return EDGE_TOP

    case FlexDirection.ColumnReverse:
      return EDGE_BOTTOM
  }
}

function trailingEdge(dir: FlexDirection): number {
  switch (dir) {
    case FlexDirection.Row:
      return EDGE_RIGHT

    case FlexDirection.RowReverse:
      return EDGE_LEFT

    case FlexDirection.Column:
      return EDGE_BOTTOM

    case FlexDirection.ColumnReverse:
      return EDGE_TOP
  }
}

export {
  AUTO_VALUE,
  crossAxis,
  defaultStyle,
  EDGE_BOTTOM,
  EDGE_LEFT,
  EDGE_RIGHT,
  EDGE_TOP,
  hasAnyAutoEdge,
  hasAnyDefinedEdge,
  isDefined,
  isMarginAuto,
  isReverse,
  isRow,
  type Layout,
  leadingEdge,
  percentValue,
  pointValue,
  resolveEdge,
  resolveEdgeRaw,
  resolveEdges4Into,
  resolveValue,
  sameFloat,
  type Style,
  trailingEdge,
  UNDEFINED_VALUE
}
