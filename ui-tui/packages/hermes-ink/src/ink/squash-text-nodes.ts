import type { DOMElement } from './dom.js'
import type { TextStyles } from './styles.js'

// A raw OSC 8 hyperlink inside a text node (model reasoning, tool output, any untrusted
// string handed to <Text>) would be tokenized as a real link, so visible text such as
// "docs.python.org" could open another URL. Links come only from <Link> (ink-link), so
// every OSC 8 sequence in plain text is dropped, its label kept.
// eslint-disable-next-line no-control-regex -- matching the escape sequence is the point
const OSC8_IN_TEXT_RE = /\u001b\]8;[^\u0007\u001b]*(?:\u0007|\u001b\\)?/g

const textOf = (value: string) => (value.includes('\u001b]8;') ? value.replace(OSC8_IN_TEXT_RE, '') : value)

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
