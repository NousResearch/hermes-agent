import {
  LayoutAlign,
  LayoutDisplay,
  LayoutEdge,
  LayoutFlexDirection,
  LayoutGutter,
  LayoutJustify,
  type LayoutNode,
  LayoutOverflow,
  LayoutPositionType,
  LayoutWrap
} from './layout/node.js'
import type { BorderStyle, BorderTextOptions } from './render-border.js'

export type RGBColor = `rgb(${number},${number},${number})`
export type HexColor = `#${string}`
export type Ansi256Color = `ansi256(${number})`
export type AnsiColor =
  | 'ansi:black'
  | 'ansi:red'
  | 'ansi:green'
  | 'ansi:yellow'
  | 'ansi:blue'
  | 'ansi:magenta'
  | 'ansi:cyan'
  | 'ansi:white'
  | 'ansi:blackBright'
  | 'ansi:redBright'
  | 'ansi:greenBright'
  | 'ansi:yellowBright'
  | 'ansi:blueBright'
  | 'ansi:magentaBright'
  | 'ansi:cyanBright'
  | 'ansi:whiteBright'

/** Raw color value - not a theme key */
export type Color = RGBColor | HexColor | Ansi256Color | AnsiColor

/**
 * Structured text styling properties.
 * Used to style text without relying on ANSI string transforms.
 * Colors are raw values - theme resolution happens at the component layer.
 */
export type TextStyles = {
  readonly color?: Color
  readonly backgroundColor?: Color
  readonly dim?: boolean
  readonly bold?: boolean
  readonly italic?: boolean
  readonly underline?: boolean
  readonly strikethrough?: boolean
  readonly inverse?: boolean
}

export type Styles = {
  readonly textWrap?:
    | 'wrap'
    | 'wrap-char'
    | 'wrap-trim'
    | 'end'
    | 'middle'
    | 'truncate-end'
    | 'truncate'
    | 'truncate-middle'
    | 'truncate-start'

  readonly position?: 'absolute' | 'relative'
  readonly top?: number | `${number}%`
  readonly bottom?: number | `${number}%`
  readonly left?: number | `${number}%`
  readonly right?: number | `${number}%`

  /**
   * Size of the gap between an element's columns.
   */
  readonly columnGap?: number

  /**
   * Size of the gap between element's rows.
   */
  readonly rowGap?: number

  /**
   * Size of the gap between an element's columns and rows. Shorthand for `columnGap` and `rowGap`.
   */
  readonly gap?: number

  /**
   * Margin on all sides. Equivalent to setting `marginTop`, `marginBottom`, `marginLeft` and `marginRight`.
   */
  readonly margin?: number

  /**
   * Horizontal margin. Equivalent to setting `marginLeft` and `marginRight`.
   */
  readonly marginX?: number

  /**
   * Vertical margin. Equivalent to setting `marginTop` and `marginBottom`.
   */
  readonly marginY?: number

  /**
   * Top margin.
   */
  readonly marginTop?: number

  /**
   * Bottom margin.
   */
  readonly marginBottom?: number

  /**
   * Left margin.
   */
  readonly marginLeft?: number

  /**
   * Right margin.
   */
  readonly marginRight?: number

  /**
   * Padding on all sides. Equivalent to setting `paddingTop`, `paddingBottom`, `paddingLeft` and `paddingRight`.
   */
  readonly padding?: number

  /**
   * Horizontal padding. Equivalent to setting `paddingLeft` and `paddingRight`.
   */
  readonly paddingX?: number

  /**
   * Vertical padding. Equivalent to setting `paddingTop` and `paddingBottom`.
   */
  readonly paddingY?: number

  /**
   * Top padding.
   */
  readonly paddingTop?: number

  /**
   * Bottom padding.
   */
  readonly paddingBottom?: number

  /**
   * Left padding.
   */
  readonly paddingLeft?: number

  /**
   * Right padding.
   */
  readonly paddingRight?: number

  /**
   * This property defines the ability for a flex item to grow if necessary.
   * See [flex-grow](https://css-tricks.com/almanac/properties/f/flex-grow/).
   */
  readonly flexGrow?: number

  /**
   * It specifies the “flex shrink factor”, which determines how much the flex item will shrink relative to the rest of the flex items in the flex container when there isn’t enough space on the row.
   * See [flex-shrink](https://css-tricks.com/almanac/properties/f/flex-shrink/).
   */
  readonly flexShrink?: number

  /**
   * It establishes the main-axis, thus defining the direction flex items are placed in the flex container.
   * See [flex-direction](https://css-tricks.com/almanac/properties/f/flex-direction/).
   */
  readonly flexDirection?: 'row' | 'column' | 'row-reverse' | 'column-reverse'

  /**
   * It specifies the initial size of the flex item, before any available space is distributed according to the flex factors.
   * See [flex-basis](https://css-tricks.com/almanac/properties/f/flex-basis/).
   */
  readonly flexBasis?: number | string

  /**
   * It defines whether the flex items are forced in a single line or can be flowed into multiple lines. If set to multiple lines, it also defines the cross-axis which determines the direction new lines are stacked in.
   * See [flex-wrap](https://css-tricks.com/almanac/properties/f/flex-wrap/).
   */
  readonly flexWrap?: 'nowrap' | 'wrap' | 'wrap-reverse'

  /**
   * The align-items property defines the default behavior for how items are laid out along the cross axis (perpendicular to the main axis).
   * See [align-items](https://css-tricks.com/almanac/properties/a/align-items/).
   */
  readonly alignItems?: 'flex-start' | 'center' | 'flex-end' | 'stretch'

  /**
   * It makes possible to override the align-items value for specific flex items.
   * See [align-self](https://css-tricks.com/almanac/properties/a/align-self/).
   */
  readonly alignSelf?: 'flex-start' | 'center' | 'flex-end' | 'auto'

  /**
   * It defines the alignment along the main axis.
   * See [justify-content](https://css-tricks.com/almanac/properties/j/justify-content/).
   */
  readonly justifyContent?: 'flex-start' | 'flex-end' | 'space-between' | 'space-around' | 'space-evenly' | 'center'

  /**
   * Width of the element in spaces.
   * You can also set it in percent, which will calculate the width based on the width of parent element.
   */
  readonly width?: number | string

  /**
   * Height of the element in lines (rows).
   * You can also set it in percent, which will calculate the height based on the height of parent element.
   */
  readonly height?: number | string

  /**
   * Sets a minimum width of the element.
   */
  readonly minWidth?: number | string

  /**
   * Sets a minimum height of the element.
   */
  readonly minHeight?: number | string

  /**
   * Sets a maximum width of the element.
   */
  readonly maxWidth?: number | string

  /**
   * Sets a maximum height of the element.
   */
  readonly maxHeight?: number | string

  /**
   * Set this property to `none` to hide the element.
   */
  readonly display?: 'flex' | 'none'

  /**
   * Add a border with a specified style.
   * If `borderStyle` is `undefined` (which it is by default), no border will be added.
   */
  readonly borderStyle?: BorderStyle

  /**
   * Determines whether top border is visible.
   *
   * @default true
   */
  readonly borderTop?: boolean

  /**
   * Determines whether bottom border is visible.
   *
   * @default true
   */
  readonly borderBottom?: boolean

  /**
   * Determines whether left border is visible.
   *
   * @default true
   */
  readonly borderLeft?: boolean

  /**
   * Determines whether right border is visible.
   *
   * @default true
   */
  readonly borderRight?: boolean

  /**
   * Change border color.
   * Shorthand for setting `borderTopColor`, `borderRightColor`, `borderBottomColor` and `borderLeftColor`.
   */
  readonly borderColor?: Color

  /**
   * Change top border color.
   * Accepts raw color values (rgb, hex, ansi).
   */
  readonly borderTopColor?: Color

  /**
   * Change bottom border color.
   * Accepts raw color values (rgb, hex, ansi).
   */
  readonly borderBottomColor?: Color

  /**
   * Change left border color.
   * Accepts raw color values (rgb, hex, ansi).
   */
  readonly borderLeftColor?: Color

  /**
   * Change right border color.
   * Accepts raw color values (rgb, hex, ansi).
   */
  readonly borderRightColor?: Color

  /**
   * Dim the border color.
   * Shorthand for setting `borderTopDimColor`, `borderBottomDimColor`, `borderLeftDimColor` and `borderRightDimColor`.
   *
   * @default false
   */
  readonly borderDimColor?: boolean

  /**
   * Dim the top border color.
   *
   * @default false
   */
  readonly borderTopDimColor?: boolean

  /**
   * Dim the bottom border color.
   *
   * @default false
   */
  readonly borderBottomDimColor?: boolean

  /**
   * Dim the left border color.
   *
   * @default false
   */
  readonly borderLeftDimColor?: boolean

  /**
   * Dim the right border color.
   *
   * @default false
   */
  readonly borderRightDimColor?: boolean

  /**
   * Add text within the border. Only applies to top or bottom borders.
   */
  readonly borderText?: BorderTextOptions

  /**
   * Background color for the box. Fills the interior with background-colored
   * spaces and is inherited by child text nodes as their default background.
   */
  readonly backgroundColor?: Color

  /**
   * Fill the box's interior (padding included) with spaces before
   * rendering children, so nothing behind it shows through. Like
   * `backgroundColor` but without emitting any SGR — the terminal's
   * default background is used. Useful for absolute-positioned overlays
   * where Box padding/gaps would otherwise be transparent.
   */
  readonly opaque?: boolean

  /**
   * Behavior for an element's overflow in both directions.
   * 'scroll' constrains the container's size (children do not expand it)
   * and enables scrollTop-based virtualized scrolling at render time.
   *
   * @default 'visible'
   */
  readonly overflow?: 'visible' | 'hidden' | 'scroll'

  /**
   * Behavior for an element's overflow in horizontal direction.
   *
   * @default 'visible'
   */
  readonly overflowX?: 'visible' | 'hidden' | 'scroll'

  /**
   * Behavior for an element's overflow in vertical direction.
   *
   * @default 'visible'
   */
  readonly overflowY?: 'visible' | 'hidden' | 'scroll'

  /**
   * Exclude this box's cells from text selection in fullscreen mode.
   * Cells inside this region are skipped by both the selection highlight
   * and the copied text — useful for fencing off gutters (line numbers,
   * diff sigils) so click-drag over a diff yields clean copyable code.
   * Only affects alt-screen text selection; no-op otherwise.
   *
   * `'from-left-edge'` extends the exclusion from column 0 to the box's
   * right edge for every row it occupies — this covers any upstream
   * indentation (tool message prefix, tree lines) so a multi-row drag
   * doesn't pick up leading whitespace from middle rows.
   */
  readonly noSelect?: boolean | 'from-left-edge'
}

const applyPositionStyles = (node: LayoutNode, style: Styles): void => {
  if ('position' in style) {
    node.setPositionType(style.position === 'absolute' ? LayoutPositionType.Absolute : LayoutPositionType.Relative)
  }

  if ('top' in style) {
    applyPositionEdge(node, 'top', style.top)
  }

  if ('bottom' in style) {
    applyPositionEdge(node, 'bottom', style.bottom)
  }

  if ('left' in style) {
    applyPositionEdge(node, 'left', style.left)
  }

  if ('right' in style) {
    applyPositionEdge(node, 'right', style.right)
  }
}

function applyPositionEdge(
  node: LayoutNode,
  edge: 'top' | 'bottom' | 'left' | 'right',
  v: number | `${number}%` | undefined
): void {
  if (typeof v === 'string') {
    node.setPositionPercent(edge, Number.parseInt(v, 10))
  } else if (typeof v === 'number') {
    node.setPosition(edge, v)
  } else {
    node.setPosition(edge, Number.NaN)
  }
}

const applyOverflowStyles = (node: LayoutNode, style: Styles): void => {
  // Yoga's Overflow controls whether children expand the container.
  // 'hidden' and 'scroll' both prevent expansion; 'scroll' additionally
  // signals that the renderer should apply scrollTop translation.
  // overflowX/Y are render-time concerns; for layout we use the union.
  const y = style.overflowY ?? style.overflow
  const x = style.overflowX ?? style.overflow

  if (y === 'scroll' || x === 'scroll') {
    node.setOverflow(LayoutOverflow.Scroll)
  } else if (y === 'hidden' || x === 'hidden') {
    node.setOverflow(LayoutOverflow.Hidden)
  } else if ('overflow' in style || 'overflowX' in style || 'overflowY' in style) {
    node.setOverflow(LayoutOverflow.Visible)
  }
}

const applyMarginStyles = (node: LayoutNode, style: Styles): void => {
  if ('margin' in style) {
    node.setMargin(LayoutEdge.All, style.margin ?? 0)
  }

  if ('marginX' in style) {
    node.setMargin(LayoutEdge.Horizontal, style.marginX ?? 0)
  }

  if ('marginY' in style) {
    node.setMargin(LayoutEdge.Vertical, style.marginY ?? 0)
  }

  if ('marginLeft' in style) {
    node.setMargin(LayoutEdge.Start, style.marginLeft || 0)
  }

  if ('marginRight' in style) {
    node.setMargin(LayoutEdge.End, style.marginRight || 0)
  }

  if ('marginTop' in style) {
    node.setMargin(LayoutEdge.Top, style.marginTop || 0)
  }

  if ('marginBottom' in style) {
    node.setMargin(LayoutEdge.Bottom, style.marginBottom || 0)
  }
}

const applyPaddingStyles = (node: LayoutNode, style: Styles): void => {
  if ('padding' in style) {
    node.setPadding(LayoutEdge.All, style.padding ?? 0)
  }

  if ('paddingX' in style) {
    node.setPadding(LayoutEdge.Horizontal, style.paddingX ?? 0)
  }

  if ('paddingY' in style) {
    node.setPadding(LayoutEdge.Vertical, style.paddingY ?? 0)
  }

  if ('paddingLeft' in style) {
    node.setPadding(LayoutEdge.Left, style.paddingLeft || 0)
  }

  if ('paddingRight' in style) {
    node.setPadding(LayoutEdge.Right, style.paddingRight || 0)
  }

  if ('paddingTop' in style) {
    node.setPadding(LayoutEdge.Top, style.paddingTop || 0)
  }

  if ('paddingBottom' in style) {
    node.setPadding(LayoutEdge.Bottom, style.paddingBottom || 0)
  }
}

const applyFlexStyles = (node: LayoutNode, style: Styles): void => {
  if ('flexGrow' in style) {
    node.setFlexGrow(style.flexGrow ?? 0)
  }

  if ('flexShrink' in style) {
    node.setFlexShrink(typeof style.flexShrink === 'number' ? style.flexShrink : 1)
  }

  if ('flexWrap' in style) {
    if (style.flexWrap === 'nowrap') {
      node.setFlexWrap(LayoutWrap.NoWrap)
    }

    if (style.flexWrap === 'wrap') {
      node.setFlexWrap(LayoutWrap.Wrap)
    }

    if (style.flexWrap === 'wrap-reverse') {
      node.setFlexWrap(LayoutWrap.WrapReverse)
    }
  }

  if ('flexDirection' in style) {
    if (style.flexDirection === 'row') {
      node.setFlexDirection(LayoutFlexDirection.Row)
    }

    if (style.flexDirection === 'row-reverse') {
      node.setFlexDirection(LayoutFlexDirection.RowReverse)
    }

    if (style.flexDirection === 'column') {
      node.setFlexDirection(LayoutFlexDirection.Column)
    }

    if (style.flexDirection === 'column-reverse') {
      node.setFlexDirection(LayoutFlexDirection.ColumnReverse)
    }
  }

  if ('flexBasis' in style) {
    if (typeof style.flexBasis === 'number') {
      node.setFlexBasis(style.flexBasis)
    } else if (typeof style.flexBasis === 'string') {
      node.setFlexBasisPercent(Number.parseInt(style.flexBasis, 10))
    } else {
      node.setFlexBasis(Number.NaN)
    }
  }

  if ('alignItems' in style) {
    if (style.alignItems === 'stretch' || !style.alignItems) {
      node.setAlignItems(LayoutAlign.Stretch)
    }

    if (style.alignItems === 'flex-start') {
      node.setAlignItems(LayoutAlign.FlexStart)
    }

    if (style.alignItems === 'center') {
      node.setAlignItems(LayoutAlign.Center)
    }

    if (style.alignItems === 'flex-end') {
      node.setAlignItems(LayoutAlign.FlexEnd)
    }
  }

  if ('alignSelf' in style) {
    if (style.alignSelf === 'auto' || !style.alignSelf) {
      node.setAlignSelf(LayoutAlign.Auto)
    }

    if (style.alignSelf === 'flex-start') {
      node.setAlignSelf(LayoutAlign.FlexStart)
    }

    if (style.alignSelf === 'center') {
      node.setAlignSelf(LayoutAlign.Center)
    }

    if (style.alignSelf === 'flex-end') {
      node.setAlignSelf(LayoutAlign.FlexEnd)
    }
  }

  if ('justifyContent' in style) {
    if (style.justifyContent === 'flex-start' || !style.justifyContent) {
      node.setJustifyContent(LayoutJustify.FlexStart)
    }

    if (style.justifyContent === 'center') {
      node.setJustifyContent(LayoutJustify.Center)
    }

    if (style.justifyContent === 'flex-end') {
      node.setJustifyContent(LayoutJustify.FlexEnd)
    }

    if (style.justifyContent === 'space-between') {
      node.setJustifyContent(LayoutJustify.SpaceBetween)
    }

    if (style.justifyContent === 'space-around') {
      node.setJustifyContent(LayoutJustify.SpaceAround)
    }

    if (style.justifyContent === 'space-evenly') {
      node.setJustifyContent(LayoutJustify.SpaceEvenly)
    }
  }
}

const applyDimensionStyles = (node: LayoutNode, style: Styles): void => {
  if ('width' in style) {
    if (typeof style.width === 'number') {
      node.setWidth(style.width)
    } else if (typeof style.width === 'string') {
      node.setWidthPercent(Number.parseInt(style.width, 10))
    } else {
      node.setWidthAuto()
    }
  }

  if ('height' in style) {
    if (typeof style.height === 'number') {
      node.setHeight(style.height)
    } else if (typeof style.height === 'string') {
      node.setHeightPercent(Number.parseInt(style.height, 10))
    } else {
      node.setHeightAuto()
    }
  }

  if ('minWidth' in style) {
    if (typeof style.minWidth === 'string') {
      node.setMinWidthPercent(Number.parseInt(style.minWidth, 10))
    } else {
      node.setMinWidth(style.minWidth ?? 0)
    }
  }

  if ('minHeight' in style) {
    if (typeof style.minHeight === 'string') {
      node.setMinHeightPercent(Number.parseInt(style.minHeight, 10))
    } else {
      node.setMinHeight(style.minHeight ?? 0)
    }
  }

  if ('maxWidth' in style) {
    if (typeof style.maxWidth === 'string') {
      node.setMaxWidthPercent(Number.parseInt(style.maxWidth, 10))
    } else {
      node.setMaxWidth(style.maxWidth ?? 0)
    }
  }

  if ('maxHeight' in style) {
    if (typeof style.maxHeight === 'string') {
      node.setMaxHeightPercent(Number.parseInt(style.maxHeight, 10))
    } else {
      node.setMaxHeight(style.maxHeight ?? 0)
    }
  }
}

const applyDisplayStyles = (node: LayoutNode, style: Styles): void => {
  if ('display' in style) {
    node.setDisplay(style.display === 'flex' ? LayoutDisplay.Flex : LayoutDisplay.None)
  }
}

const applyBorderStyles = (node: LayoutNode, style: Styles, resolvedStyle?: Styles): void => {
  // resolvedStyle is the full current style (already set on the DOM node).
  // style may be a diff with only changed properties. For border side props,
  // we need the resolved value because `borderStyle` in a diff may not include
  // unchanged border side values (e.g. borderTop stays false but isn't in the diff).
  const resolved = resolvedStyle ?? style

  if ('borderStyle' in style) {
    const borderWidth = style.borderStyle ? 1 : 0

    node.setBorder(LayoutEdge.Top, resolved.borderTop !== false ? borderWidth : 0)
    node.setBorder(LayoutEdge.Bottom, resolved.borderBottom !== false ? borderWidth : 0)
    node.setBorder(LayoutEdge.Left, resolved.borderLeft !== false ? borderWidth : 0)
    node.setBorder(LayoutEdge.Right, resolved.borderRight !== false ? borderWidth : 0)
  } else {
    // Handle individual border property changes (when only borderX changes without borderStyle).
    // Skip undefined values — they mean the prop was removed or never set,
    // not that a border should be enabled.
    if ('borderTop' in style && style.borderTop !== undefined) {
      node.setBorder(LayoutEdge.Top, style.borderTop === false ? 0 : 1)
    }

    if ('borderBottom' in style && style.borderBottom !== undefined) {
      node.setBorder(LayoutEdge.Bottom, style.borderBottom === false ? 0 : 1)
    }

    if ('borderLeft' in style && style.borderLeft !== undefined) {
      node.setBorder(LayoutEdge.Left, style.borderLeft === false ? 0 : 1)
    }

    if ('borderRight' in style && style.borderRight !== undefined) {
      node.setBorder(LayoutEdge.Right, style.borderRight === false ? 0 : 1)
    }
  }
}

const applyGapStyles = (node: LayoutNode, style: Styles): void => {
  if ('gap' in style) {
    node.setGap(LayoutGutter.All, style.gap ?? 0)
  }

  if ('columnGap' in style) {
    node.setGap(LayoutGutter.Column, style.columnGap ?? 0)
  }

  if ('rowGap' in style) {
    node.setGap(LayoutGutter.Row, style.rowGap ?? 0)
  }
}

const styles = (node: LayoutNode, style: Styles = {}, resolvedStyle?: Styles): void => {
  applyPositionStyles(node, style)
  applyOverflowStyles(node, style)
  applyMarginStyles(node, style)
  applyPaddingStyles(node, style)
  applyFlexStyles(node, style)
  applyDimensionStyles(node, style)
  applyDisplayStyles(node, style)
  applyBorderStyles(node, style, resolvedStyle)
  applyGapStyles(node, style)
}

export default styles
