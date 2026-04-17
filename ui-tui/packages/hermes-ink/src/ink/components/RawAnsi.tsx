import React from 'react'
import { c as _c } from 'react/compiler-runtime'
type Props = {
  /**
   * Pre-rendered ANSI lines. Each element must be exactly one terminal row
   * (already wrapped to `width` by the producer) with ANSI escape codes inline.
   */
  lines: string[]
  /** Column width the producer wrapped to. Sent to Yoga as the fixed leaf width. */
  width: number
}

/**
 * Bypass the <Ansi> → React tree → Yoga → squash → re-serialize roundtrip for
 * content that is already terminal-ready.
 *
 * Use this when an external renderer (e.g. the ColorDiff NAPI module) has
 * already produced ANSI-escaped, width-wrapped output. A normal <Ansi> mount
 * reparses that output into one React <Text> per style span, lays out each
 * span as a Yoga flex child, then walks the tree to re-emit the same escape
 * codes it was given. For a long transcript full of syntax-highlighted diffs
 * that roundtrip is the dominant cost of the render.
 *
 * This component emits a single Yoga leaf with a constant-time measure func
 * (width × lines.length) and hands the joined string straight to output.write(),
 * which already splits on '\n' and parses ANSI into the screen buffer.
 */
export function RawAnsi(t0: Props) {
  const $ = _c(6)

  const { lines, width } = t0

  if (lines.length === 0) {
    return null
  }

  let t1

  if ($[0] !== lines) {
    t1 = lines.join('\n')
    $[0] = lines
    $[1] = t1
  } else {
    t1 = $[1]
  }

  let t2

  if ($[2] !== lines.length || $[3] !== t1 || $[4] !== width) {
    t2 = <ink-raw-ansi rawHeight={lines.length} rawText={t1} rawWidth={width} />
    $[2] = lines.length
    $[3] = t1
    $[4] = width
    $[5] = t2
  } else {
    t2 = $[5]
  }

  return t2
}
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJSZWFjdCIsIlByb3BzIiwibGluZXMiLCJ3aWR0aCIsIlJhd0Fuc2kiLCJ0MCIsIiQiLCJfYyIsImxlbmd0aCIsInQxIiwiam9pbiIsInQyIl0sInNvdXJjZXMiOlsiUmF3QW5zaS50c3giXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0J1xuXG50eXBlIFByb3BzID0ge1xuICAvKipcbiAgICogUHJlLXJlbmRlcmVkIEFOU0kgbGluZXMuIEVhY2ggZWxlbWVudCBtdXN0IGJlIGV4YWN0bHkgb25lIHRlcm1pbmFsIHJvd1xuICAgKiAoYWxyZWFkeSB3cmFwcGVkIHRvIGB3aWR0aGAgYnkgdGhlIHByb2R1Y2VyKSB3aXRoIEFOU0kgZXNjYXBlIGNvZGVzIGlubGluZS5cbiAgICovXG4gIGxpbmVzOiBzdHJpbmdbXVxuICAvKiogQ29sdW1uIHdpZHRoIHRoZSBwcm9kdWNlciB3cmFwcGVkIHRvLiBTZW50IHRvIFlvZ2EgYXMgdGhlIGZpeGVkIGxlYWYgd2lkdGguICovXG4gIHdpZHRoOiBudW1iZXJcbn1cblxuLyoqXG4gKiBCeXBhc3MgdGhlIDxBbnNpPiDihpIgUmVhY3QgdHJlZSDihpIgWW9nYSDihpIgc3F1YXNoIOKGkiByZS1zZXJpYWxpemUgcm91bmR0cmlwIGZvclxuICogY29udGVudCB0aGF0IGlzIGFscmVhZHkgdGVybWluYWwtcmVhZHkuXG4gKlxuICogVXNlIHRoaXMgd2hlbiBhbiBleHRlcm5hbCByZW5kZXJlciAoZS5nLiB0aGUgQ29sb3JEaWZmIE5BUEkgbW9kdWxlKSBoYXNcbiAqIGFscmVhZHkgcHJvZHVjZWQgQU5TSS1lc2NhcGVkLCB3aWR0aC13cmFwcGVkIG91dHB1dC4gQSBub3JtYWwgPEFuc2k+IG1vdW50XG4gKiByZXBhcnNlcyB0aGF0IG91dHB1dCBpbnRvIG9uZSBSZWFjdCA8VGV4dD4gcGVyIHN0eWxlIHNwYW4sIGxheXMgb3V0IGVhY2hcbiAqIHNwYW4gYXMgYSBZb2dhIGZsZXggY2hpbGQsIHRoZW4gd2Fsa3MgdGhlIHRyZWUgdG8gcmUtZW1pdCB0aGUgc2FtZSBlc2NhcGVcbiAqIGNvZGVzIGl0IHdhcyBnaXZlbi4gRm9yIGEgbG9uZyB0cmFuc2NyaXB0IGZ1bGwgb2Ygc3ludGF4LWhpZ2hsaWdodGVkIGRpZmZzXG4gKiB0aGF0IHJvdW5kdHJpcCBpcyB0aGUgZG9taW5hbnQgY29zdCBvZiB0aGUgcmVuZGVyLlxuICpcbiAqIFRoaXMgY29tcG9uZW50IGVtaXRzIGEgc2luZ2xlIFlvZ2EgbGVhZiB3aXRoIGEgY29uc3RhbnQtdGltZSBtZWFzdXJlIGZ1bmNcbiAqICh3aWR0aCDDlyBsaW5lcy5sZW5ndGgpIGFuZCBoYW5kcyB0aGUgam9pbmVkIHN0cmluZyBzdHJhaWdodCB0byBvdXRwdXQud3JpdGUoKSxcbiAqIHdoaWNoIGFscmVhZHkgc3BsaXRzIG9uICdcXG4nIGFuZCBwYXJzZXMgQU5TSSBpbnRvIHRoZSBzY3JlZW4gYnVmZmVyLlxuICovXG5leHBvcnQgZnVuY3Rpb24gUmF3QW5zaSh7IGxpbmVzLCB3aWR0aCB9OiBQcm9wcyk6IFJlYWN0LlJlYWN0Tm9kZSB7XG4gIGlmIChsaW5lcy5sZW5ndGggPT09IDApIHtcbiAgICByZXR1cm4gbnVsbFxuICB9XG4gIHJldHVybiAoXG4gICAgPGluay1yYXctYW5zaVxuICAgICAgcmF3VGV4dD17bGluZXMuam9pbignXFxuJyl9XG4gICAgICByYXdXaWR0aD17d2lkdGh9XG4gICAgICByYXdIZWlnaHQ9e2xpbmVzLmxlbmd0aH1cbiAgICAvPlxuICApXG59XG4iXSwibWFwcGluZ3MiOiI7QUFBQSxPQUFPQSxLQUFLLE1BQU0sT0FBTztBQUV6QixLQUFLQyxLQUFLLEdBQUc7RUFDWDtBQUNGO0FBQ0E7QUFDQTtFQUNFQyxLQUFLLEVBQUUsTUFBTSxFQUFFO0VBQ2Y7RUFDQUMsS0FBSyxFQUFFLE1BQU07QUFDZixDQUFDOztBQUVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLE9BQU8sU0FBQUMsUUFBQUMsRUFBQTtFQUFBLE1BQUFDLENBQUEsR0FBQUMsRUFBQTtFQUFpQjtJQUFBTCxLQUFBO0lBQUFDO0VBQUEsSUFBQUUsRUFBdUI7RUFDN0MsSUFBSUgsS0FBSyxDQUFBTSxNQUFPLEtBQUssQ0FBQztJQUFBLE9BQ2IsSUFBSTtFQUFBO0VBQ1osSUFBQUMsRUFBQTtFQUFBLElBQUFILENBQUEsUUFBQUosS0FBQTtJQUdZTyxFQUFBLEdBQUFQLEtBQUssQ0FBQVEsSUFBSyxDQUFDLElBQUksQ0FBQztJQUFBSixDQUFBLE1BQUFKLEtBQUE7SUFBQUksQ0FBQSxNQUFBRyxFQUFBO0VBQUE7SUFBQUEsRUFBQSxHQUFBSCxDQUFBO0VBQUE7RUFBQSxJQUFBSyxFQUFBO0VBQUEsSUFBQUwsQ0FBQSxRQUFBSixLQUFBLENBQUFNLE1BQUEsSUFBQUYsQ0FBQSxRQUFBRyxFQUFBLElBQUFILENBQUEsUUFBQUgsS0FBQTtJQUQzQlEsRUFBQSxnQkFJRSxDQUhTLE9BQWdCLENBQWhCLENBQUFGLEVBQWUsQ0FBQyxDQUNmTixRQUFLLENBQUxBLE1BQUksQ0FBQyxDQUNKLFNBQVksQ0FBWixDQUFBRCxLQUFLLENBQUFNLE1BQU0sQ0FBQyxHQUN2QjtJQUFBRixDQUFBLE1BQUFKLEtBQUEsQ0FBQU0sTUFBQTtJQUFBRixDQUFBLE1BQUFHLEVBQUE7SUFBQUgsQ0FBQSxNQUFBSCxLQUFBO0lBQUFHLENBQUEsTUFBQUssRUFBQTtFQUFBO0lBQUFBLEVBQUEsR0FBQUwsQ0FBQTtFQUFBO0VBQUEsT0FKRkssRUFJRTtBQUFBIiwiaWdub3JlTGlzdCI6W119
