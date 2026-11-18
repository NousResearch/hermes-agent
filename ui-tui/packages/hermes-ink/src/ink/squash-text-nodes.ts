import type { DOMElement } from './dom.js'
import type { TextStyles } from './styles.js'

// A raw OSC 8 hyperlink inside a text node (model reasoning, tool output, any untrusted
// string handed to <Text>) would be tokenized as a real link, so visible text such as
// "docs.python.org" could open another URL. Links come only from <Link> (ink-link), so
// every OSC 8 sequence in plain text is dropped, its label kept. The renderer joins every
// text node under one ink-text into one string, so the strip runs on that joined run:
// stripping node by node would let sibling fragments ("\x1b" + "]8;;url\x07…") rebuild
// a link after the filter.
const OSC8_OPEN = '\u001b]8;'

const endsWithOpener = (value: string, out: number[]) => {
  const start = out.length - OSC8_OPEN.length

  if (start < 0) {
    return false
  }

  for (let j = 0; j < OSC8_OPEN.length; j++) {
    if (value[out[start + j]!] !== OSC8_OPEN[j]) {
      return false
    }
  }

  return true
}

// Rule: the output never contains the opener. Chars are appended one at a time; the moment
// the output ends with the opener — whether the sequence was in the input or spliced together
// by an earlier removal — it is popped and the payload up to its terminator (BEL, ST, or the
// ESC that aborts it) is skipped. The skip never crosses a newline or the end of the segment
// holding the opener's last char: untrusted text must not be able to hide the lines, nodes or
// trusted <Link> labels after it. Without a terminator in reach only the opener is dropped;
// the rest has no opener left, so it shows as plain text and cannot form a link.
// `ends` are the segment end offsets, ascending, the last one value.length. Linear in the
// input: an unterminated scan is remembered, so openers spliced inside its range skip it.
// Returns the surviving source indices, ascending, so a run can be split back into segments.
const keptIndices = (value: string, ends: readonly number[]) => {
  const out: number[] = []
  let i = 0
  let segment = 0
  let unterminatedUntil = -1

  while (i < value.length) {
    out.push(i++)

    if (!endsWithOpener(value, out)) {
      continue
    }

    out.length -= OSC8_OPEN.length

    while (ends[segment]! < i) {
      segment++
    }

    if (i < unterminatedUntil) {
      continue
    }

    const limit = ends[segment]!
    let j = i

    while (j < limit && value[j] !== '\u0007' && value[j] !== '\u001b' && value[j] !== '\n') {
      j++
    }

    if (j === limit || value[j] === '\n') {
      unterminatedUntil = j
    } else if (value[j] === '\u0007') {
      i = j + 1
    } else {
      i = j + 1 < limit && value[j + 1] === '\\' ? j + 2 : j
    }
  }

  return out
}

export const stripOsc8 = (value: string) =>
  value.includes(OSC8_OPEN)
    ? keptIndices(value, [value.length])
        .map(i => value[i])
        .join('')
    : value

/**
 * A segment of text with its associated styles.
 * Used for structured rendering without ANSI string transforms.
 */
export type StyledSegment = {
  text: string
  styles: TextStyles
  hyperlink?: string
}

function collectSegments(
  node: DOMElement,
  inheritedStyles: TextStyles,
  inheritedHyperlink: string | undefined,
  out: StyledSegment[]
): StyledSegment[] {
  const mergedStyles = node.textStyles ? { ...inheritedStyles, ...node.textStyles } : inheritedStyles

  for (const childNode of node.childNodes) {
    if (childNode === undefined) {
      continue
    }

    if (childNode.nodeName === '#text') {
      if (childNode.nodeValue.length > 0) {
        out.push({
          text: childNode.nodeValue,
          styles: mergedStyles,
          hyperlink: inheritedHyperlink
        })
      }
    } else if (childNode.nodeName === 'ink-text' || childNode.nodeName === 'ink-virtual-text') {
      collectSegments(childNode, mergedStyles, inheritedHyperlink, out)
    } else if (childNode.nodeName === 'ink-link') {
      const href = childNode.attributes['href'] as string | undefined
      collectSegments(childNode, mergedStyles, href || inheritedHyperlink, out)
    }
  }

  return out
}

// Strips the joined text of all segments, then hands each segment back the chars of its own
// range that survived. A trusted <Link> keeps its href: that is segment metadata, not text.
const stripSegments = (segments: StyledSegment[]) => {
  const joined = segments.map(s => s.text).join('')

  if (!joined.includes(OSC8_OPEN)) {
    return segments
  }

  let offset = 0

  const kept = keptIndices(
    joined,
    segments.map(s => (offset += s.text.length))
  )

  const out: StyledSegment[] = []
  let k = 0
  let end = 0

  for (const segment of segments) {
    end += segment.text.length
    const from = k

    while (k < kept.length && kept[k]! < end) {
      k++
    }

    if (k > from) {
      out.push({
        ...segment,
        text: kept
          .slice(from, k)
          .map(i => joined[i])
          .join('')
      })
    }
  }

  return out
}

/**
 * Squash text nodes into styled segments, propagating styles down through the tree.
 * This allows structured styling without relying on ANSI string transforms.
 */
export const squashTextNodesToSegments = (node: DOMElement, inheritedStyles: TextStyles = {}): StyledSegment[] =>
  stripSegments(collectSegments(node, inheritedStyles, undefined, []))

/**
 * Squash text nodes into a plain string (without styles).
 * Used for text measurement in layout calculations: the same stripped segments as rendering,
 * so the measured width is the drawn width.
 */
const squashTextNodes = (node: DOMElement): string =>
  squashTextNodesToSegments(node)
    .map(s => s.text)
    .join('')

export default squashTextNodes
