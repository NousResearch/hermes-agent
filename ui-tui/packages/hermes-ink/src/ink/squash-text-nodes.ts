import type { DOMElement } from './dom.js'
import type { TextStyles } from './styles.js'

// A raw OSC 8 hyperlink inside a text node (model reasoning, tool output, any untrusted
// string handed to <Text>) would be tokenized as a real link, so visible text such as
// "docs.python.org" could open another URL. Links come only from <Link> (ink-link), so
// every OSC 8 sequence in plain text is dropped, its label kept.
const OSC8_OPEN = '\u001b]8;'

const endsWithOpener = (out: string[]) => {
  const start = out.length - OSC8_OPEN.length

  if (start < 0) {
    return false
  }

  for (let j = 0; j < OSC8_OPEN.length; j++) {
    if (out[start + j] !== OSC8_OPEN[j]) {
      return false
    }
  }

  return true
}

// Rule: the output never contains the opener. Chars are appended one at a time; the moment
// the output ends with the opener — whether the sequence was in the input or spliced together
// by an earlier removal — it is popped and the payload up to its terminator is skipped.
// Linear in the input, so nested inputs cannot stall a frame.
export const stripOsc8 = (value: string) => {
  if (!value.includes(OSC8_OPEN)) {
    return value
  }

  const out: string[] = []
  let i = 0

  while (i < value.length) {
    out.push(value[i++]!)

    if (!endsWithOpener(out)) {
      continue
    }

    out.length -= OSC8_OPEN.length

    while (i < value.length && value[i] !== '\u0007' && value[i] !== '\u001b') {
      i++
    }

    if (value[i] === '\u0007') {
      i++
    } else if (value[i] === '\u001b' && value[i + 1] === '\\') {
      i += 2
    }
  }

  return out.join('')
}

const textOf = stripOsc8

/**
 * A segment of text with its associated styles.
 * Used for structured rendering without ANSI string transforms.
 */
export type StyledSegment = {
  text: string
  styles: TextStyles
  hyperlink?: string
}

/**
 * Squash text nodes into styled segments, propagating styles down through the tree.
 * This allows structured styling without relying on ANSI string transforms.
 */
export function squashTextNodesToSegments(
  node: DOMElement,
  inheritedStyles: TextStyles = {},
  inheritedHyperlink?: string,
  out: StyledSegment[] = []
): StyledSegment[] {
  const mergedStyles = node.textStyles ? { ...inheritedStyles, ...node.textStyles } : inheritedStyles

  for (const childNode of node.childNodes) {
    if (childNode === undefined) {
      continue
    }

    if (childNode.nodeName === '#text') {
      const text = textOf(childNode.nodeValue)

      if (text.length > 0) {
        out.push({
          text,
          styles: mergedStyles,
          hyperlink: inheritedHyperlink
        })
      }
    } else if (childNode.nodeName === 'ink-text' || childNode.nodeName === 'ink-virtual-text') {
      squashTextNodesToSegments(childNode, mergedStyles, inheritedHyperlink, out)
    } else if (childNode.nodeName === 'ink-link') {
      const href = childNode.attributes['href'] as string | undefined
      squashTextNodesToSegments(childNode, mergedStyles, href || inheritedHyperlink, out)
    }
  }

  return out
}

/**
 * Squash text nodes into a plain string (without styles).
 * Used for text measurement in layout calculations.
 */
function squashTextNodes(node: DOMElement): string {
  let text = ''

  for (const childNode of node.childNodes) {
    if (childNode === undefined) {
      continue
    }

    if (childNode.nodeName === '#text') {
      text += textOf(childNode.nodeValue)
    } else if (childNode.nodeName === 'ink-text' || childNode.nodeName === 'ink-virtual-text') {
      text += squashTextNodes(childNode)
    } else if (childNode.nodeName === 'ink-link') {
      text += squashTextNodes(childNode)
    }
  }

  return text
}

export default squashTextNodes
