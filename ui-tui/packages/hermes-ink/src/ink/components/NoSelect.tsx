import React from 'react'
import { c as _c } from 'react/compiler-runtime'

import Box, { type Props as BoxProps } from './Box.js'
type Props = Omit<BoxProps, 'noSelect'> & {
  /**
   * Extend the exclusion zone from column 0 to this box's right edge,
   * for every row this box occupies. Use for gutters rendered inside a
   * wider indented container (e.g. a diff inside a tool message row):
   * without this, a multi-row drag picks up the container's leading
   * indent on rows below the prefix.
   *
   * @default false
   */
  fromLeftEdge?: boolean
}

/**
 * Marks its contents as non-selectable in fullscreen text selection.
 * Cells inside this box are skipped by both the selection highlight and
 * the copied text — the gutter stays visually unchanged while the user
 * drags, making it clear what will be copied.
 *
 * Use to fence off gutters (line numbers, diff +/- sigils, list bullets)
 * so click-drag over rendered code yields clean pasteable content:
 *
 *   <Box flexDirection="row">
 *     <NoSelect fromLeftEdge><Text dimColor> 42 +</Text></NoSelect>
 *     <Text>const x = 1</Text>
 *   </Box>
 *
 * Only affects alt-screen text selection (<AlternateScreen> with mouse
 * tracking). No-op in the main-screen scrollback render where the
 * terminal's native selection is used instead.
 */
export function NoSelect(t0: Props) {
  const $ = _c(8)
  let boxProps
  let children
  let fromLeftEdge

  if ($[0] !== t0) {
    ;({ children, fromLeftEdge, ...boxProps } = t0)
    $[0] = t0
    $[1] = boxProps
    $[2] = children
    $[3] = fromLeftEdge
  } else {
    boxProps = $[1]
    children = $[2]
    fromLeftEdge = $[3]
  }

  const t1 = fromLeftEdge ? 'from-left-edge' : true
  let t2

  if ($[4] !== boxProps || $[5] !== children || $[6] !== t1) {
    t2 = (
      <Box {...boxProps} noSelect={t1}>
        {children}
      </Box>
    )
    $[4] = boxProps
    $[5] = children
    $[6] = t1
    $[7] = t2
  } else {
    t2 = $[7]
  }

  return t2
}
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJSZWFjdCIsIlByb3BzV2l0aENoaWxkcmVuIiwiQm94IiwiUHJvcHMiLCJCb3hQcm9wcyIsIk9taXQiLCJmcm9tTGVmdEVkZ2UiLCJOb1NlbGVjdCIsInQwIiwiJCIsIl9jIiwiYm94UHJvcHMiLCJjaGlsZHJlbiIsInQxIiwidDIiXSwic291cmNlcyI6WyJOb1NlbGVjdC50c3giXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IHR5cGUgUHJvcHNXaXRoQ2hpbGRyZW4gfSBmcm9tICdyZWFjdCdcbmltcG9ydCBCb3gsIHsgdHlwZSBQcm9wcyBhcyBCb3hQcm9wcyB9IGZyb20gJy4vQm94LmpzJ1xuXG50eXBlIFByb3BzID0gT21pdDxCb3hQcm9wcywgJ25vU2VsZWN0Jz4gJiB7XG4gIC8qKlxuICAgKiBFeHRlbmQgdGhlIGV4Y2x1c2lvbiB6b25lIGZyb20gY29sdW1uIDAgdG8gdGhpcyBib3gncyByaWdodCBlZGdlLFxuICAgKiBmb3IgZXZlcnkgcm93IHRoaXMgYm94IG9jY3VwaWVzLiBVc2UgZm9yIGd1dHRlcnMgcmVuZGVyZWQgaW5zaWRlIGFcbiAgICogd2lkZXIgaW5kZW50ZWQgY29udGFpbmVyIChlLmcuIGEgZGlmZiBpbnNpZGUgYSB0b29sIG1lc3NhZ2Ugcm93KTpcbiAgICogd2l0aG91dCB0aGlzLCBhIG11bHRpLXJvdyBkcmFnIHBpY2tzIHVwIHRoZSBjb250YWluZXIncyBsZWFkaW5nXG4gICAqIGluZGVudCBvbiByb3dzIGJlbG93IHRoZSBwcmVmaXguXG4gICAqXG4gICAqIEBkZWZhdWx0IGZhbHNlXG4gICAqL1xuICBmcm9tTGVmdEVkZ2U/OiBib29sZWFuXG59XG5cbi8qKlxuICogTWFya3MgaXRzIGNvbnRlbnRzIGFzIG5vbi1zZWxlY3RhYmxlIGluIGZ1bGxzY3JlZW4gdGV4dCBzZWxlY3Rpb24uXG4gKiBDZWxscyBpbnNpZGUgdGhpcyBib3ggYXJlIHNraXBwZWQgYnkgYm90aCB0aGUgc2VsZWN0aW9uIGhpZ2hsaWdodCBhbmRcbiAqIHRoZSBjb3BpZWQgdGV4dCDigJQgdGhlIGd1dHRlciBzdGF5cyB2aXN1YWxseSB1bmNoYW5nZWQgd2hpbGUgdGhlIHVzZXJcbiAqIGRyYWdzLCBtYWtpbmcgaXQgY2xlYXIgd2hhdCB3aWxsIGJlIGNvcGllZC5cbiAqXG4gKiBVc2UgdG8gZmVuY2Ugb2ZmIGd1dHRlcnMgKGxpbmUgbnVtYmVycywgZGlmZiArLy0gc2lnaWxzLCBsaXN0IGJ1bGxldHMpXG4gKiBzbyBjbGljay1kcmFnIG92ZXIgcmVuZGVyZWQgY29kZSB5aWVsZHMgY2xlYW4gcGFzdGVhYmxlIGNvbnRlbnQ6XG4gKlxuICogICA8Qm94IGZsZXhEaXJlY3Rpb249XCJyb3dcIj5cbiAqICAgICA8Tm9TZWxlY3QgZnJvbUxlZnRFZGdlPjxUZXh0IGRpbUNvbG9yPiA0MiArPC9UZXh0PjwvTm9TZWxlY3Q+XG4gKiAgICAgPFRleHQ+Y29uc3QgeCA9IDE8L1RleHQ+XG4gKiAgIDwvQm94PlxuICpcbiAqIE9ubHkgYWZmZWN0cyBhbHQtc2NyZWVuIHRleHQgc2VsZWN0aW9uICg8QWx0ZXJuYXRlU2NyZWVuPiB3aXRoIG1vdXNlXG4gKiB0cmFja2luZykuIE5vLW9wIGluIHRoZSBtYWluLXNjcmVlbiBzY3JvbGxiYWNrIHJlbmRlciB3aGVyZSB0aGVcbiAqIHRlcm1pbmFsJ3MgbmF0aXZlIHNlbGVjdGlvbiBpcyB1c2VkIGluc3RlYWQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBOb1NlbGVjdCh7XG4gIGNoaWxkcmVuLFxuICBmcm9tTGVmdEVkZ2UsXG4gIC4uLmJveFByb3BzXG59OiBQcm9wc1dpdGhDaGlsZHJlbjxQcm9wcz4pOiBSZWFjdC5SZWFjdE5vZGUge1xuICByZXR1cm4gKFxuICAgIDxCb3ggey4uLmJveFByb3BzfSBub1NlbGVjdD17ZnJvbUxlZnRFZGdlID8gJ2Zyb20tbGVmdC1lZGdlJyA6IHRydWV9PlxuICAgICAge2NoaWxkcmVufVxuICAgIDwvQm94PlxuICApXG59XG4iXSwibWFwcGluZ3MiOiI7QUFBQSxPQUFPQSxLQUFLLElBQUksS0FBS0MsaUJBQWlCLFFBQVEsT0FBTztBQUNyRCxPQUFPQyxHQUFHLElBQUksS0FBS0MsS0FBSyxJQUFJQyxRQUFRLFFBQVEsVUFBVTtBQUV0RCxLQUFLRCxLQUFLLEdBQUdFLElBQUksQ0FBQ0QsUUFBUSxFQUFFLFVBQVUsQ0FBQyxHQUFHO0VBQ3hDO0FBQ0Y7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtFQUNFRSxZQUFZLENBQUMsRUFBRSxPQUFPO0FBQ3hCLENBQUM7O0FBRUQ7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsT0FBTyxTQUFBQyxTQUFBQyxFQUFBO0VBQUEsTUFBQUMsQ0FBQSxHQUFBQyxFQUFBO0VBQUEsSUFBQUMsUUFBQTtFQUFBLElBQUFDLFFBQUE7RUFBQSxJQUFBTixZQUFBO0VBQUEsSUFBQUcsQ0FBQSxRQUFBRCxFQUFBO0lBQWtCO01BQUFJLFFBQUE7TUFBQU4sWUFBQTtNQUFBLEdBQUFLO0lBQUEsSUFBQUgsRUFJRTtJQUFBQyxDQUFBLE1BQUFELEVBQUE7SUFBQUMsQ0FBQSxNQUFBRSxRQUFBO0lBQUFGLENBQUEsTUFBQUcsUUFBQTtJQUFBSCxDQUFBLE1BQUFILFlBQUE7RUFBQTtJQUFBSyxRQUFBLEdBQUFGLENBQUE7SUFBQUcsUUFBQSxHQUFBSCxDQUFBO0lBQUFILFlBQUEsR0FBQUcsQ0FBQTtFQUFBO0VBRU0sTUFBQUksRUFBQSxHQUFBUCxZQUFZLEdBQVosZ0JBQXNDLEdBQXRDLElBQXNDO0VBQUEsSUFBQVEsRUFBQTtFQUFBLElBQUFMLENBQUEsUUFBQUUsUUFBQSxJQUFBRixDQUFBLFFBQUFHLFFBQUEsSUFBQUgsQ0FBQSxRQUFBSSxFQUFBO0lBQW5FQyxFQUFBLElBQUMsR0FBRyxLQUFLSCxRQUFRLEVBQVksUUFBc0MsQ0FBdEMsQ0FBQUUsRUFBcUMsQ0FBQyxDQUNoRUQsU0FBTyxDQUNWLEVBRkMsR0FBRyxDQUVFO0lBQUFILENBQUEsTUFBQUUsUUFBQTtJQUFBRixDQUFBLE1BQUFHLFFBQUE7SUFBQUgsQ0FBQSxNQUFBSSxFQUFBO0lBQUFKLENBQUEsTUFBQUssRUFBQTtFQUFBO0lBQUFBLEVBQUEsR0FBQUwsQ0FBQTtFQUFBO0VBQUEsT0FGTkssRUFFTTtBQUFBIiwiaWdub3JlTGlzdCI6W119
