export const Align = {
  Auto: 0,
  FlexStart: 1,
  Center: 2,
  FlexEnd: 3,
  Stretch: 4,
  Baseline: 5,
  SpaceBetween: 6,
  SpaceAround: 7,
  SpaceEvenly: 8
} as const
export type Align = (typeof Align)[keyof typeof Align]
export const BoxSizing = {
  BorderBox: 0,
  ContentBox: 1
} as const
export type BoxSizing = (typeof BoxSizing)[keyof typeof BoxSizing]
export const Dimension = {
  Width: 0,
  Height: 1
} as const
export type Dimension = (typeof Dimension)[keyof typeof Dimension]
export const Direction = {
  Inherit: 0,
  LTR: 1,
  RTL: 2
} as const
export type Direction = (typeof Direction)[keyof typeof Direction]
export const Display = {
  Flex: 0,
  None: 1,
  Contents: 2
} as const
export type Display = (typeof Display)[keyof typeof Display]
export const Edge = {
  Left: 0,
  Top: 1,
  Right: 2,
  Bottom: 3,
  Start: 4,
  End: 5,
  Horizontal: 6,
  Vertical: 7,
  All: 8
} as const
export type Edge = (typeof Edge)[keyof typeof Edge]
export const Errata = {
  None: 0,
  StretchFlexBasis: 1,
  AbsolutePositionWithoutInsetsExcludesPadding: 2,
  AbsolutePercentAgainstInnerSize: 4,
  All: 2147483647,
  Classic: 2147483646
} as const
export type Errata = (typeof Errata)[keyof typeof Errata]
export const ExperimentalFeature = {
  WebFlexBasis: 0
} as const
export type ExperimentalFeature = (typeof ExperimentalFeature)[keyof typeof ExperimentalFeature]
export const FlexDirection = {
  Column: 0,
  ColumnReverse: 1,
  Row: 2,
  RowReverse: 3
} as const
export type FlexDirection = (typeof FlexDirection)[keyof typeof FlexDirection]
export const Gutter = {
  Column: 0,
  Row: 1,
  All: 2
} as const
export type Gutter = (typeof Gutter)[keyof typeof Gutter]
export const Justify = {
  FlexStart: 0,
  Center: 1,
  FlexEnd: 2,
  SpaceBetween: 3,
  SpaceAround: 4,
  SpaceEvenly: 5
} as const
export type Justify = (typeof Justify)[keyof typeof Justify]
export const MeasureMode = {
  Undefined: 0,
  Exactly: 1,
  AtMost: 2
} as const
export type MeasureMode = (typeof MeasureMode)[keyof typeof MeasureMode]
export const Overflow = {
  Visible: 0,
  Hidden: 1,
  Scroll: 2
} as const
export type Overflow = (typeof Overflow)[keyof typeof Overflow]
export const PositionType = {
  Static: 0,
  Relative: 1,
  Absolute: 2
} as const
export type PositionType = (typeof PositionType)[keyof typeof PositionType]
export const Unit = {
  Undefined: 0,
  Point: 1,
  Percent: 2,
  Auto: 3
} as const
export type Unit = (typeof Unit)[keyof typeof Unit]
export const Wrap = {
  NoWrap: 0,
  Wrap: 1,
  WrapReverse: 2
} as const
export type Wrap = (typeof Wrap)[keyof typeof Wrap]
