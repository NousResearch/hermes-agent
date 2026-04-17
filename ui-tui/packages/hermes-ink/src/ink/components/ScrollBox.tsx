import '../global.d.ts'

import React, { type PropsWithChildren, type Ref, useImperativeHandle, useRef, useState } from 'react'
import type { Except } from 'type-fest'

import { markScrollActivity } from '../../bootstrap/state.js'
import type { DOMElement } from '../dom.js'
import { markDirty, scheduleRenderFrom } from '../dom.js'
import { markCommitStart } from '../reconciler.js'
import type { Styles } from '../styles.js'

import Box from './Box.js'
export type ScrollBoxHandle = {
  scrollTo: (y: number) => void
  scrollBy: (dy: number) => void
  /**
   * Scroll so `el`'s top is at the viewport top (plus `offset`). Unlike
   * scrollTo which bakes a number that's stale by the time the throttled
   * render fires, this defers the position read to render time —
   * render-node-to-output reads `el.yogaNode.getComputedTop()` in the
   * SAME Yoga pass that computes scrollHeight. Deterministic. One-shot.
   */
  scrollToElement: (el: DOMElement, offset?: number) => void
  scrollToBottom: () => void
  getScrollTop: () => number
  getPendingDelta: () => number
  getScrollHeight: () => number
  /**
   * Like getScrollHeight, but reads Yoga directly instead of the cached
   * value written by render-node-to-output (throttled, up to 16ms stale).
   * Use when you need a fresh value in useLayoutEffect after a React commit
   * that grew content. Slightly more expensive (native Yoga call).
   */
  getFreshScrollHeight: () => number
  getViewportHeight: () => number
  /**
   * Absolute screen-buffer row of the first visible content line (inside
   * padding). Used for drag-to-scroll edge detection.
   */
  getViewportTop: () => number
  /**
   * True when scroll is pinned to the bottom. Set by scrollToBottom, the
   * initial stickyScroll attribute, and by the renderer when positional
   * follow fires (scrollTop at prevMax, content grows). Cleared by
   * scrollTo/scrollBy. Stable signal for "at bottom" that doesn't depend on
   * layout values (unlike scrollTop+viewportH >= scrollHeight).
   */
  isSticky: () => boolean
  /**
   * Subscribe to imperative scroll changes (scrollTo/scrollBy/scrollToBottom).
   * Does NOT fire for stickyScroll updates done by the Ink renderer — those
   * happen during Ink's render phase after React has committed. Callers that
   * care about the sticky case should treat "at bottom" as a fallback.
   */
  subscribe: (listener: () => void) => () => void
  /**
   * Set the render-time scrollTop clamp to the currently-mounted children's
   * coverage span. Called by useVirtualScroll after computing its range;
   * render-node-to-output clamps scrollTop to [min, max] so burst scrollTo
   * calls that race past React's async re-render show the edge of mounted
   * content instead of blank spacer. Pass undefined to disable (sticky,
   * cold start).
   */
  setClampBounds: (min: number | undefined, max: number | undefined) => void
}
export type ScrollBoxProps = Except<Styles, 'textWrap' | 'overflow' | 'overflowX' | 'overflowY'> & {
  ref?: Ref<ScrollBoxHandle>
  /**
   * When true, automatically pins scroll position to the bottom when content
   * grows. Unset manually via scrollTo/scrollBy to break the stickiness.
   */
  stickyScroll?: boolean
}

/**
 * A Box with `overflow: scroll` and an imperative scroll API.
 *
 * Children are laid out at their full Yoga-computed height inside a
 * constrained container. At render time, only children intersecting the
 * visible window (scrollTop..scrollTop+height) are rendered (viewport
 * culling). Content is translated by -scrollTop and clipped to the box bounds.
 *
 * Works best inside a fullscreen (constrained-height root) Ink tree.
 */
function ScrollBox({ children, ref, stickyScroll, ...style }: PropsWithChildren<ScrollBoxProps>): React.ReactNode {
  const domRef = useRef<DOMElement>(null)
  // scrollTo/scrollBy bypass React: they mutate scrollTop on the DOM node,
  // mark it dirty, and call the root's throttled scheduleRender directly.
  // The Ink renderer reads scrollTop from the node — no React state needed,
  // no reconciler overhead per wheel event. The microtask defer coalesces
  // multiple scrollBy calls in one input batch (discreteUpdates) into one
  // render — otherwise scheduleRender's leading edge fires on the FIRST
  // event before subsequent events mutate scrollTop. scrollToBottom still
  // forces a React render: sticky is attribute-observed, no DOM-only path.
  const [, forceRender] = useState(0)
  const listenersRef = useRef(new Set<() => void>())
  const renderQueuedRef = useRef(false)

  const notify = () => {
    for (const l of listenersRef.current) {
      l()
    }
  }

  function scrollMutated(el: DOMElement): void {
    // Signal background intervals (IDE poll, LSP poll, GCS fetch, orphan
    // check) to skip their next tick — they compete for the event loop and
    // contributed to 1402ms max frame gaps during scroll drain.
    markScrollActivity()
    markDirty(el)
    markCommitStart()
    notify()

    if (renderQueuedRef.current) {
      return
    }

    renderQueuedRef.current = true
    queueMicrotask(() => {
      renderQueuedRef.current = false
      scheduleRenderFrom(el)
    })
  }

  useImperativeHandle(
    ref,
    (): ScrollBoxHandle => ({
      scrollTo(y: number) {
        const el = domRef.current

        if (!el) {
          return
        }

        // Explicit false overrides the DOM attribute so manual scroll
        // breaks stickiness. Render code checks ?? precedence.
        el.stickyScroll = false
        el.pendingScrollDelta = undefined
        el.scrollAnchor = undefined
        el.scrollTop = Math.max(0, Math.floor(y))
        scrollMutated(el)
      },
      scrollToElement(el: DOMElement, offset = 0) {
        const box = domRef.current

        if (!box) {
          return
        }

        box.stickyScroll = false
        box.pendingScrollDelta = undefined
        box.scrollAnchor = {
          el,
          offset
        }
        scrollMutated(box)
      },
      scrollBy(dy: number) {
        const el = domRef.current

        if (!el) {
          return
        }

        el.stickyScroll = false
        // Wheel input cancels any in-flight anchor seek — user override.
        el.scrollAnchor = undefined
        // Accumulate in pendingScrollDelta; renderer drains it at a capped
        // rate so fast flicks show intermediate frames. Pure accumulator:
        // scroll-up followed by scroll-down naturally cancels.
        el.pendingScrollDelta = (el.pendingScrollDelta ?? 0) + Math.floor(dy)
        scrollMutated(el)
      },
      scrollToBottom() {
        const el = domRef.current

        if (!el) {
          return
        }

        el.pendingScrollDelta = undefined
        el.stickyScroll = true
        markDirty(el)
        notify()
        forceRender(n => n + 1)
      },
      getScrollTop() {
        return domRef.current?.scrollTop ?? 0
      },
      getPendingDelta() {
        // Accumulated-but-not-yet-drained delta. useVirtualScroll needs
        // this to mount the union [committed, committed+pending] range —
        // otherwise intermediate drain frames find no children (blank).
        return domRef.current?.pendingScrollDelta ?? 0
      },
      getScrollHeight() {
        return domRef.current?.scrollHeight ?? 0
      },
      getFreshScrollHeight() {
        const content = domRef.current?.childNodes[0] as DOMElement | undefined

        return content?.yogaNode?.getComputedHeight() ?? domRef.current?.scrollHeight ?? 0
      },
      getViewportHeight() {
        return domRef.current?.scrollViewportHeight ?? 0
      },
      getViewportTop() {
        return domRef.current?.scrollViewportTop ?? 0
      },
      isSticky() {
        const el = domRef.current

        if (!el) {
          return false
        }

        return el.stickyScroll ?? Boolean(el.attributes['stickyScroll'])
      },
      subscribe(listener: () => void) {
        listenersRef.current.add(listener)

        return () => listenersRef.current.delete(listener)
      },
      setClampBounds(min, max) {
        const el = domRef.current

        if (!el) {
          return
        }

        el.scrollClampMin = min
        el.scrollClampMax = max
      }
    }),
    // notify/scrollMutated are inline (no useCallback) but only close over
    // refs + imports — stable. Empty deps avoids rebuilding the handle on
    // every render (which re-registers the ref = churn).

    []
  )

  // Structure: outer viewport (overflow:scroll, constrained height) >
  // inner content (flexGrow:1, flexShrink:0 — fills at least the viewport
  // but grows beyond it for tall content). flexGrow:1 lets children use
  // spacers to pin elements to the bottom of the scroll area. Yoga's
  // Overflow.Scroll prevents the viewport from growing to fit the content.
  // The renderer computes scrollHeight from the content box and culls
  // content's children based on scrollTop.
  //
  // stickyScroll is passed as a DOM attribute (via ink-box directly) so it's
  // available on the first render — ref callbacks fire after the initial
  // commit, which is too late for the first frame.
  return (
    <ink-box
      ref={(el: DOMElement | null) => {
        domRef.current = el

        if (el) {
          el.scrollTop ??= 0
        }
      }}
      style={{
        flexWrap: 'nowrap',
        flexDirection: style.flexDirection ?? 'row',
        flexGrow: style.flexGrow ?? 0,
        flexShrink: style.flexShrink ?? 1,
        ...style,
        overflowX: 'scroll',
        overflowY: 'scroll'
      }}
      {...(stickyScroll
        ? {
            stickyScroll: true
          }
        : {})}
    >
      <Box flexDirection="column" flexGrow={1} flexShrink={0} width="100%">
        {children}
      </Box>
    </ink-box>
  )
}

export default ScrollBox
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJuYW1lcyI6WyJSZWFjdCIsIlByb3BzV2l0aENoaWxkcmVuIiwiUmVmIiwidXNlSW1wZXJhdGl2ZUhhbmRsZSIsInVzZVJlZiIsInVzZVN0YXRlIiwiRXhjZXB0IiwibWFya1Njcm9sbEFjdGl2aXR5IiwiRE9NRWxlbWVudCIsIm1hcmtEaXJ0eSIsInNjaGVkdWxlUmVuZGVyRnJvbSIsIm1hcmtDb21taXRTdGFydCIsIlN0eWxlcyIsIkJveCIsIlNjcm9sbEJveEhhbmRsZSIsInNjcm9sbFRvIiwieSIsInNjcm9sbEJ5IiwiZHkiLCJzY3JvbGxUb0VsZW1lbnQiLCJlbCIsIm9mZnNldCIsInNjcm9sbFRvQm90dG9tIiwiZ2V0U2Nyb2xsVG9wIiwiZ2V0UGVuZGluZ0RlbHRhIiwiZ2V0U2Nyb2xsSGVpZ2h0IiwiZ2V0RnJlc2hTY3JvbGxIZWlnaHQiLCJnZXRWaWV3cG9ydEhlaWdodCIsImdldFZpZXdwb3J0VG9wIiwiaXNTdGlja3kiLCJzdWJzY3JpYmUiLCJsaXN0ZW5lciIsInNldENsYW1wQm91bmRzIiwibWluIiwibWF4IiwiU2Nyb2xsQm94UHJvcHMiLCJyZWYiLCJzdGlja3lTY3JvbGwiLCJTY3JvbGxCb3giLCJjaGlsZHJlbiIsInN0eWxlIiwiUmVhY3ROb2RlIiwiZG9tUmVmIiwiZm9yY2VSZW5kZXIiLCJsaXN0ZW5lcnNSZWYiLCJTZXQiLCJyZW5kZXJRdWV1ZWRSZWYiLCJub3RpZnkiLCJsIiwiY3VycmVudCIsInNjcm9sbE11dGF0ZWQiLCJxdWV1ZU1pY3JvdGFzayIsInBlbmRpbmdTY3JvbGxEZWx0YSIsInVuZGVmaW5lZCIsInNjcm9sbEFuY2hvciIsInNjcm9sbFRvcCIsIk1hdGgiLCJmbG9vciIsImJveCIsIm4iLCJzY3JvbGxIZWlnaHQiLCJjb250ZW50IiwiY2hpbGROb2RlcyIsInlvZ2FOb2RlIiwiZ2V0Q29tcHV0ZWRIZWlnaHQiLCJzY3JvbGxWaWV3cG9ydEhlaWdodCIsInNjcm9sbFZpZXdwb3J0VG9wIiwiQm9vbGVhbiIsImF0dHJpYnV0ZXMiLCJhZGQiLCJkZWxldGUiLCJzY3JvbGxDbGFtcE1pbiIsInNjcm9sbENsYW1wTWF4IiwiZmxleFdyYXAiLCJmbGV4RGlyZWN0aW9uIiwiZmxleEdyb3ciLCJmbGV4U2hyaW5rIiwib3ZlcmZsb3dYIiwib3ZlcmZsb3dZIl0sInNvdXJjZXMiOlsiU2Nyb2xsQm94LnRzeCJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHtcbiAgdHlwZSBQcm9wc1dpdGhDaGlsZHJlbixcbiAgdHlwZSBSZWYsXG4gIHVzZUltcGVyYXRpdmVIYW5kbGUsXG4gIHVzZVJlZixcbiAgdXNlU3RhdGUsXG59IGZyb20gJ3JlYWN0J1xuaW1wb3J0IHR5cGUgeyBFeGNlcHQgfSBmcm9tICd0eXBlLWZlc3QnXG5pbXBvcnQgeyBtYXJrU2Nyb2xsQWN0aXZpdHkgfSBmcm9tICcuLi8uLi9ib290c3RyYXAvc3RhdGUuanMnXG5pbXBvcnQgdHlwZSB7IERPTUVsZW1lbnQgfSBmcm9tICcuLi9kb20uanMnXG5pbXBvcnQgeyBtYXJrRGlydHksIHNjaGVkdWxlUmVuZGVyRnJvbSB9IGZyb20gJy4uL2RvbS5qcydcbmltcG9ydCB7IG1hcmtDb21taXRTdGFydCB9IGZyb20gJy4uL3JlY29uY2lsZXIuanMnXG5pbXBvcnQgdHlwZSB7IFN0eWxlcyB9IGZyb20gJy4uL3N0eWxlcy5qcydcbmltcG9ydCAnLi4vZ2xvYmFsLmQudHMnXG5pbXBvcnQgQm94IGZyb20gJy4vQm94LmpzJ1xuXG5leHBvcnQgdHlwZSBTY3JvbGxCb3hIYW5kbGUgPSB7XG4gIHNjcm9sbFRvOiAoeTogbnVtYmVyKSA9PiB2b2lkXG4gIHNjcm9sbEJ5OiAoZHk6IG51bWJlcikgPT4gdm9pZFxuICAvKipcbiAgICogU2Nyb2xsIHNvIGBlbGAncyB0b3AgaXMgYXQgdGhlIHZpZXdwb3J0IHRvcCAocGx1cyBgb2Zmc2V0YCkuIFVubGlrZVxuICAgKiBzY3JvbGxUbyB3aGljaCBiYWtlcyBhIG51bWJlciB0aGF0J3Mgc3RhbGUgYnkgdGhlIHRpbWUgdGhlIHRocm90dGxlZFxuICAgKiByZW5kZXIgZmlyZXMsIHRoaXMgZGVmZXJzIHRoZSBwb3NpdGlvbiByZWFkIHRvIHJlbmRlciB0aW1lIOKAlFxuICAgKiByZW5kZXItbm9kZS10by1vdXRwdXQgcmVhZHMgYGVsLnlvZ2FOb2RlLmdldENvbXB1dGVkVG9wKClgIGluIHRoZVxuICAgKiBTQU1FIFlvZ2EgcGFzcyB0aGF0IGNvbXB1dGVzIHNjcm9sbEhlaWdodC4gRGV0ZXJtaW5pc3RpYy4gT25lLXNob3QuXG4gICAqL1xuICBzY3JvbGxUb0VsZW1lbnQ6IChlbDogRE9NRWxlbWVudCwgb2Zmc2V0PzogbnVtYmVyKSA9PiB2b2lkXG4gIHNjcm9sbFRvQm90dG9tOiAoKSA9PiB2b2lkXG4gIGdldFNjcm9sbFRvcDogKCkgPT4gbnVtYmVyXG4gIGdldFBlbmRpbmdEZWx0YTogKCkgPT4gbnVtYmVyXG4gIGdldFNjcm9sbEhlaWdodDogKCkgPT4gbnVtYmVyXG4gIC8qKlxuICAgKiBMaWtlIGdldFNjcm9sbEhlaWdodCwgYnV0IHJlYWRzIFlvZ2EgZGlyZWN0bHkgaW5zdGVhZCBvZiB0aGUgY2FjaGVkXG4gICAqIHZhbHVlIHdyaXR0ZW4gYnkgcmVuZGVyLW5vZGUtdG8tb3V0cHV0ICh0aHJvdHRsZWQsIHVwIHRvIDE2bXMgc3RhbGUpLlxuICAgKiBVc2Ugd2hlbiB5b3UgbmVlZCBhIGZyZXNoIHZhbHVlIGluIHVzZUxheW91dEVmZmVjdCBhZnRlciBhIFJlYWN0IGNvbW1pdFxuICAgKiB0aGF0IGdyZXcgY29udGVudC4gU2xpZ2h0bHkgbW9yZSBleHBlbnNpdmUgKG5hdGl2ZSBZb2dhIGNhbGwpLlxuICAgKi9cbiAgZ2V0RnJlc2hTY3JvbGxIZWlnaHQ6ICgpID0+IG51bWJlclxuICBnZXRWaWV3cG9ydEhlaWdodDogKCkgPT4gbnVtYmVyXG4gIC8qKlxuICAgKiBBYnNvbHV0ZSBzY3JlZW4tYnVmZmVyIHJvdyBvZiB0aGUgZmlyc3QgdmlzaWJsZSBjb250ZW50IGxpbmUgKGluc2lkZVxuICAgKiBwYWRkaW5nKS4gVXNlZCBmb3IgZHJhZy10by1zY3JvbGwgZWRnZSBkZXRlY3Rpb24uXG4gICAqL1xuICBnZXRWaWV3cG9ydFRvcDogKCkgPT4gbnVtYmVyXG4gIC8qKlxuICAgKiBUcnVlIHdoZW4gc2Nyb2xsIGlzIHBpbm5lZCB0byB0aGUgYm90dG9tLiBTZXQgYnkgc2Nyb2xsVG9Cb3R0b20sIHRoZVxuICAgKiBpbml0aWFsIHN0aWNreVNjcm9sbCBhdHRyaWJ1dGUsIGFuZCBieSB0aGUgcmVuZGVyZXIgd2hlbiBwb3NpdGlvbmFsXG4gICAqIGZvbGxvdyBmaXJlcyAoc2Nyb2xsVG9wIGF0IHByZXZNYXgsIGNvbnRlbnQgZ3Jvd3MpLiBDbGVhcmVkIGJ5XG4gICAqIHNjcm9sbFRvL3Njcm9sbEJ5LiBTdGFibGUgc2lnbmFsIGZvciBcImF0IGJvdHRvbVwiIHRoYXQgZG9lc24ndCBkZXBlbmQgb25cbiAgICogbGF5b3V0IHZhbHVlcyAodW5saWtlIHNjcm9sbFRvcCt2aWV3cG9ydEggPj0gc2Nyb2xsSGVpZ2h0KS5cbiAgICovXG4gIGlzU3RpY2t5OiAoKSA9PiBib29sZWFuXG4gIC8qKlxuICAgKiBTdWJzY3JpYmUgdG8gaW1wZXJhdGl2ZSBzY3JvbGwgY2hhbmdlcyAoc2Nyb2xsVG8vc2Nyb2xsQnkvc2Nyb2xsVG9Cb3R0b20pLlxuICAgKiBEb2VzIE5PVCBmaXJlIGZvciBzdGlja3lTY3JvbGwgdXBkYXRlcyBkb25lIGJ5IHRoZSBJbmsgcmVuZGVyZXIg4oCUIHRob3NlXG4gICAqIGhhcHBlbiBkdXJpbmcgSW5rJ3MgcmVuZGVyIHBoYXNlIGFmdGVyIFJlYWN0IGhhcyBjb21taXR0ZWQuIENhbGxlcnMgdGhhdFxuICAgKiBjYXJlIGFib3V0IHRoZSBzdGlja3kgY2FzZSBzaG91bGQgdHJlYXQgXCJhdCBib3R0b21cIiBhcyBhIGZhbGxiYWNrLlxuICAgKi9cbiAgc3Vic2NyaWJlOiAobGlzdGVuZXI6ICgpID0+IHZvaWQpID0+ICgpID0+IHZvaWRcbiAgLyoqXG4gICAqIFNldCB0aGUgcmVuZGVyLXRpbWUgc2Nyb2xsVG9wIGNsYW1wIHRvIHRoZSBjdXJyZW50bHktbW91bnRlZCBjaGlsZHJlbidzXG4gICAqIGNvdmVyYWdlIHNwYW4uIENhbGxlZCBieSB1c2VWaXJ0dWFsU2Nyb2xsIGFmdGVyIGNvbXB1dGluZyBpdHMgcmFuZ2U7XG4gICAqIHJlbmRlci1ub2RlLXRvLW91dHB1dCBjbGFtcHMgc2Nyb2xsVG9wIHRvIFttaW4sIG1heF0gc28gYnVyc3Qgc2Nyb2xsVG9cbiAgICogY2FsbHMgdGhhdCByYWNlIHBhc3QgUmVhY3QncyBhc3luYyByZS1yZW5kZXIgc2hvdyB0aGUgZWRnZSBvZiBtb3VudGVkXG4gICAqIGNvbnRlbnQgaW5zdGVhZCBvZiBibGFuayBzcGFjZXIuIFBhc3MgdW5kZWZpbmVkIHRvIGRpc2FibGUgKHN0aWNreSxcbiAgICogY29sZCBzdGFydCkuXG4gICAqL1xuICBzZXRDbGFtcEJvdW5kczogKG1pbjogbnVtYmVyIHwgdW5kZWZpbmVkLCBtYXg6IG51bWJlciB8IHVuZGVmaW5lZCkgPT4gdm9pZFxufVxuXG5leHBvcnQgdHlwZSBTY3JvbGxCb3hQcm9wcyA9IEV4Y2VwdDxcbiAgU3R5bGVzLFxuICAndGV4dFdyYXAnIHwgJ292ZXJmbG93JyB8ICdvdmVyZmxvd1gnIHwgJ292ZXJmbG93WSdcbj4gJiB7XG4gIHJlZj86IFJlZjxTY3JvbGxCb3hIYW5kbGU+XG4gIC8qKlxuICAgKiBXaGVuIHRydWUsIGF1dG9tYXRpY2FsbHkgcGlucyBzY3JvbGwgcG9zaXRpb24gdG8gdGhlIGJvdHRvbSB3aGVuIGNvbnRlbnRcbiAgICogZ3Jvd3MuIFVuc2V0IG1hbnVhbGx5IHZpYSBzY3JvbGxUby9zY3JvbGxCeSB0byBicmVhayB0aGUgc3RpY2tpbmVzcy5cbiAgICovXG4gIHN0aWNreVNjcm9sbD86IGJvb2xlYW5cbn1cblxuLyoqXG4gKiBBIEJveCB3aXRoIGBvdmVyZmxvdzogc2Nyb2xsYCBhbmQgYW4gaW1wZXJhdGl2ZSBzY3JvbGwgQVBJLlxuICpcbiAqIENoaWxkcmVuIGFyZSBsYWlkIG91dCBhdCB0aGVpciBmdWxsIFlvZ2EtY29tcHV0ZWQgaGVpZ2h0IGluc2lkZSBhXG4gKiBjb25zdHJhaW5lZCBjb250YWluZXIuIEF0IHJlbmRlciB0aW1lLCBvbmx5IGNoaWxkcmVuIGludGVyc2VjdGluZyB0aGVcbiAqIHZpc2libGUgd2luZG93IChzY3JvbGxUb3AuLnNjcm9sbFRvcCtoZWlnaHQpIGFyZSByZW5kZXJlZCAodmlld3BvcnRcbiAqIGN1bGxpbmcpLiBDb250ZW50IGlzIHRyYW5zbGF0ZWQgYnkgLXNjcm9sbFRvcCBhbmQgY2xpcHBlZCB0byB0aGUgYm94IGJvdW5kcy5cbiAqXG4gKiBXb3JrcyBiZXN0IGluc2lkZSBhIGZ1bGxzY3JlZW4gKGNvbnN0cmFpbmVkLWhlaWdodCByb290KSBJbmsgdHJlZS5cbiAqL1xuZnVuY3Rpb24gU2Nyb2xsQm94KHtcbiAgY2hpbGRyZW4sXG4gIHJlZixcbiAgc3RpY2t5U2Nyb2xsLFxuICAuLi5zdHlsZVxufTogUHJvcHNXaXRoQ2hpbGRyZW48U2Nyb2xsQm94UHJvcHM+KTogUmVhY3QuUmVhY3ROb2RlIHtcbiAgY29uc3QgZG9tUmVmID0gdXNlUmVmPERPTUVsZW1lbnQ+KG51bGwpXG4gIC8vIHNjcm9sbFRvL3Njcm9sbEJ5IGJ5cGFzcyBSZWFjdDogdGhleSBtdXRhdGUgc2Nyb2xsVG9wIG9uIHRoZSBET00gbm9kZSxcbiAgLy8gbWFyayBpdCBkaXJ0eSwgYW5kIGNhbGwgdGhlIHJvb3QncyB0aHJvdHRsZWQgc2NoZWR1bGVSZW5kZXIgZGlyZWN0bHkuXG4gIC8vIFRoZSBJbmsgcmVuZGVyZXIgcmVhZHMgc2Nyb2xsVG9wIGZyb20gdGhlIG5vZGUg4oCUIG5vIFJlYWN0IHN0YXRlIG5lZWRlZCxcbiAgLy8gbm8gcmVjb25jaWxlciBvdmVyaGVhZCBwZXIgd2hlZWwgZXZlbnQuIFRoZSBtaWNyb3Rhc2sgZGVmZXIgY29hbGVzY2VzXG4gIC8vIG11bHRpcGxlIHNjcm9sbEJ5IGNhbGxzIGluIG9uZSBpbnB1dCBiYXRjaCAoZGlzY3JldGVVcGRhdGVzKSBpbnRvIG9uZVxuICAvLyByZW5kZXIg4oCUIG90aGVyd2lzZSBzY2hlZHVsZVJlbmRlcidzIGxlYWRpbmcgZWRnZSBmaXJlcyBvbiB0aGUgRklSU1RcbiAgLy8gZXZlbnQgYmVmb3JlIHN1YnNlcXVlbnQgZXZlbnRzIG11dGF0ZSBzY3JvbGxUb3AuIHNjcm9sbFRvQm90dG9tIHN0aWxsXG4gIC8vIGZvcmNlcyBhIFJlYWN0IHJlbmRlcjogc3RpY2t5IGlzIGF0dHJpYnV0ZS1vYnNlcnZlZCwgbm8gRE9NLW9ubHkgcGF0aC5cbiAgY29uc3QgWywgZm9yY2VSZW5kZXJdID0gdXNlU3RhdGUoMClcbiAgY29uc3QgbGlzdGVuZXJzUmVmID0gdXNlUmVmKG5ldyBTZXQ8KCkgPT4gdm9pZD4oKSlcbiAgY29uc3QgcmVuZGVyUXVldWVkUmVmID0gdXNlUmVmKGZhbHNlKVxuXG4gIGNvbnN0IG5vdGlmeSA9ICgpID0+IHtcbiAgICBmb3IgKGNvbnN0IGwgb2YgbGlzdGVuZXJzUmVmLmN1cnJlbnQpIGwoKVxuICB9XG5cbiAgZnVuY3Rpb24gc2Nyb2xsTXV0YXRlZChlbDogRE9NRWxlbWVudCk6IHZvaWQge1xuICAgIC8vIFNpZ25hbCBiYWNrZ3JvdW5kIGludGVydmFscyAoSURFIHBvbGwsIExTUCBwb2xsLCBHQ1MgZmV0Y2gsIG9ycGhhblxuICAgIC8vIGNoZWNrKSB0byBza2lwIHRoZWlyIG5leHQgdGljayDigJQgdGhleSBjb21wZXRlIGZvciB0aGUgZXZlbnQgbG9vcCBhbmRcbiAgICAvLyBjb250cmlidXRlZCB0byAxNDAybXMgbWF4IGZyYW1lIGdhcHMgZHVyaW5nIHNjcm9sbCBkcmFpbi5cbiAgICBtYXJrU2Nyb2xsQWN0aXZpdHkoKVxuICAgIG1hcmtEaXJ0eShlbClcbiAgICBtYXJrQ29tbWl0U3RhcnQoKVxuICAgIG5vdGlmeSgpXG4gICAgaWYgKHJlbmRlclF1ZXVlZFJlZi5jdXJyZW50KSByZXR1cm5cbiAgICByZW5kZXJRdWV1ZWRSZWYuY3VycmVudCA9IHRydWVcbiAgICBxdWV1ZU1pY3JvdGFzaygoKSA9PiB7XG4gICAgICByZW5kZXJRdWV1ZWRSZWYuY3VycmVudCA9IGZhbHNlXG4gICAgICBzY2hlZHVsZVJlbmRlckZyb20oZWwpXG4gICAgfSlcbiAgfVxuXG4gIHVzZUltcGVyYXRpdmVIYW5kbGUoXG4gICAgcmVmLFxuICAgICgpOiBTY3JvbGxCb3hIYW5kbGUgPT4gKHtcbiAgICAgIHNjcm9sbFRvKHk6IG51bWJlcikge1xuICAgICAgICBjb25zdCBlbCA9IGRvbVJlZi5jdXJyZW50XG4gICAgICAgIGlmICghZWwpIHJldHVyblxuICAgICAgICAvLyBFeHBsaWNpdCBmYWxzZSBvdmVycmlkZXMgdGhlIERPTSBhdHRyaWJ1dGUgc28gbWFudWFsIHNjcm9sbFxuICAgICAgICAvLyBicmVha3Mgc3RpY2tpbmVzcy4gUmVuZGVyIGNvZGUgY2hlY2tzID8/IHByZWNlZGVuY2UuXG4gICAgICAgIGVsLnN0aWNreVNjcm9sbCA9IGZhbHNlXG4gICAgICAgIGVsLnBlbmRpbmdTY3JvbGxEZWx0YSA9IHVuZGVmaW5lZFxuICAgICAgICBlbC5zY3JvbGxBbmNob3IgPSB1bmRlZmluZWRcbiAgICAgICAgZWwuc2Nyb2xsVG9wID0gTWF0aC5tYXgoMCwgTWF0aC5mbG9vcih5KSlcbiAgICAgICAgc2Nyb2xsTXV0YXRlZChlbClcbiAgICAgIH0sXG4gICAgICBzY3JvbGxUb0VsZW1lbnQoZWw6IERPTUVsZW1lbnQsIG9mZnNldCA9IDApIHtcbiAgICAgICAgY29uc3QgYm94ID0gZG9tUmVmLmN1cnJlbnRcbiAgICAgICAgaWYgKCFib3gpIHJldHVyblxuICAgICAgICBib3guc3RpY2t5U2Nyb2xsID0gZmFsc2VcbiAgICAgICAgYm94LnBlbmRpbmdTY3JvbGxEZWx0YSA9IHVuZGVmaW5lZFxuICAgICAgICBib3guc2Nyb2xsQW5jaG9yID0geyBlbCwgb2Zmc2V0IH1cbiAgICAgICAgc2Nyb2xsTXV0YXRlZChib3gpXG4gICAgICB9LFxuICAgICAgc2Nyb2xsQnkoZHk6IG51bWJlcikge1xuICAgICAgICBjb25zdCBlbCA9IGRvbVJlZi5jdXJyZW50XG4gICAgICAgIGlmICghZWwpIHJldHVyblxuICAgICAgICBlbC5zdGlja3lTY3JvbGwgPSBmYWxzZVxuICAgICAgICAvLyBXaGVlbCBpbnB1dCBjYW5jZWxzIGFueSBpbi1mbGlnaHQgYW5jaG9yIHNlZWsg4oCUIHVzZXIgb3ZlcnJpZGUuXG4gICAgICAgIGVsLnNjcm9sbEFuY2hvciA9IHVuZGVmaW5lZFxuICAgICAgICAvLyBBY2N1bXVsYXRlIGluIHBlbmRpbmdTY3JvbGxEZWx0YTsgcmVuZGVyZXIgZHJhaW5zIGl0IGF0IGEgY2FwcGVkXG4gICAgICAgIC8vIHJhdGUgc28gZmFzdCBmbGlja3Mgc2hvdyBpbnRlcm1lZGlhdGUgZnJhbWVzLiBQdXJlIGFjY3VtdWxhdG9yOlxuICAgICAgICAvLyBzY3JvbGwtdXAgZm9sbG93ZWQgYnkgc2Nyb2xsLWRvd24gbmF0dXJhbGx5IGNhbmNlbHMuXG4gICAgICAgIGVsLnBlbmRpbmdTY3JvbGxEZWx0YSA9IChlbC5wZW5kaW5nU2Nyb2xsRGVsdGEgPz8gMCkgKyBNYXRoLmZsb29yKGR5KVxuICAgICAgICBzY3JvbGxNdXRhdGVkKGVsKVxuICAgICAgfSxcbiAgICAgIHNjcm9sbFRvQm90dG9tKCkge1xuICAgICAgICBjb25zdCBlbCA9IGRvbVJlZi5jdXJyZW50XG4gICAgICAgIGlmICghZWwpIHJldHVyblxuICAgICAgICBlbC5wZW5kaW5nU2Nyb2xsRGVsdGEgPSB1bmRlZmluZWRcbiAgICAgICAgZWwuc3RpY2t5U2Nyb2xsID0gdHJ1ZVxuICAgICAgICBtYXJrRGlydHkoZWwpXG4gICAgICAgIG5vdGlmeSgpXG4gICAgICAgIGZvcmNlUmVuZGVyKG4gPT4gbiArIDEpXG4gICAgICB9LFxuICAgICAgZ2V0U2Nyb2xsVG9wKCkge1xuICAgICAgICByZXR1cm4gZG9tUmVmLmN1cnJlbnQ/LnNjcm9sbFRvcCA/PyAwXG4gICAgICB9LFxuICAgICAgZ2V0UGVuZGluZ0RlbHRhKCkge1xuICAgICAgICAvLyBBY2N1bXVsYXRlZC1idXQtbm90LXlldC1kcmFpbmVkIGRlbHRhLiB1c2VWaXJ0dWFsU2Nyb2xsIG5lZWRzXG4gICAgICAgIC8vIHRoaXMgdG8gbW91bnQgdGhlIHVuaW9uIFtjb21taXR0ZWQsIGNvbW1pdHRlZCtwZW5kaW5nXSByYW5nZSDigJRcbiAgICAgICAgLy8gb3RoZXJ3aXNlIGludGVybWVkaWF0ZSBkcmFpbiBmcmFtZXMgZmluZCBubyBjaGlsZHJlbiAoYmxhbmspLlxuICAgICAgICByZXR1cm4gZG9tUmVmLmN1cnJlbnQ/LnBlbmRpbmdTY3JvbGxEZWx0YSA/PyAwXG4gICAgICB9LFxuICAgICAgZ2V0U2Nyb2xsSGVpZ2h0KCkge1xuICAgICAgICByZXR1cm4gZG9tUmVmLmN1cnJlbnQ/LnNjcm9sbEhlaWdodCA/PyAwXG4gICAgICB9LFxuICAgICAgZ2V0RnJlc2hTY3JvbGxIZWlnaHQoKSB7XG4gICAgICAgIGNvbnN0IGNvbnRlbnQgPSBkb21SZWYuY3VycmVudD8uY2hpbGROb2Rlc1swXSBhcyBET01FbGVtZW50IHwgdW5kZWZpbmVkXG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgY29udGVudD8ueW9nYU5vZGU/LmdldENvbXB1dGVkSGVpZ2h0KCkgPz9cbiAgICAgICAgICBkb21SZWYuY3VycmVudD8uc2Nyb2xsSGVpZ2h0ID8/XG4gICAgICAgICAgMFxuICAgICAgICApXG4gICAgICB9LFxuICAgICAgZ2V0Vmlld3BvcnRIZWlnaHQoKSB7XG4gICAgICAgIHJldHVybiBkb21SZWYuY3VycmVudD8uc2Nyb2xsVmlld3BvcnRIZWlnaHQgPz8gMFxuICAgICAgfSxcbiAgICAgIGdldFZpZXdwb3J0VG9wKCkge1xuICAgICAgICByZXR1cm4gZG9tUmVmLmN1cnJlbnQ/LnNjcm9sbFZpZXdwb3J0VG9wID8/IDBcbiAgICAgIH0sXG4gICAgICBpc1N0aWNreSgpIHtcbiAgICAgICAgY29uc3QgZWwgPSBkb21SZWYuY3VycmVudFxuICAgICAgICBpZiAoIWVsKSByZXR1cm4gZmFsc2VcbiAgICAgICAgcmV0dXJuIGVsLnN0aWNreVNjcm9sbCA/PyBCb29sZWFuKGVsLmF0dHJpYnV0ZXNbJ3N0aWNreVNjcm9sbCddKVxuICAgICAgfSxcbiAgICAgIHN1YnNjcmliZShsaXN0ZW5lcjogKCkgPT4gdm9pZCkge1xuICAgICAgICBsaXN0ZW5lcnNSZWYuY3VycmVudC5hZGQobGlzdGVuZXIpXG4gICAgICAgIHJldHVybiAoKSA9PiBsaXN0ZW5lcnNSZWYuY3VycmVudC5kZWxldGUobGlzdGVuZXIpXG4gICAgICB9LFxuICAgICAgc2V0Q2xhbXBCb3VuZHMobWluLCBtYXgpIHtcbiAgICAgICAgY29uc3QgZWwgPSBkb21SZWYuY3VycmVudFxuICAgICAgICBpZiAoIWVsKSByZXR1cm5cbiAgICAgICAgZWwuc2Nyb2xsQ2xhbXBNaW4gPSBtaW5cbiAgICAgICAgZWwuc2Nyb2xsQ2xhbXBNYXggPSBtYXhcbiAgICAgIH0sXG4gICAgfSksXG4gICAgLy8gbm90aWZ5L3Njcm9sbE11dGF0ZWQgYXJlIGlubGluZSAobm8gdXNlQ2FsbGJhY2spIGJ1dCBvbmx5IGNsb3NlIG92ZXJcbiAgICAvLyByZWZzICsgaW1wb3J0cyDigJQgc3RhYmxlLiBFbXB0eSBkZXBzIGF2b2lkcyByZWJ1aWxkaW5nIHRoZSBoYW5kbGUgb25cbiAgICAvLyBldmVyeSByZW5kZXIgKHdoaWNoIHJlLXJlZ2lzdGVycyB0aGUgcmVmID0gY2h1cm4pLlxuICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSByZWFjdC1ob29rcy9leGhhdXN0aXZlLWRlcHNcbiAgICBbXSxcbiAgKVxuXG4gIC8vIFN0cnVjdHVyZTogb3V0ZXIgdmlld3BvcnQgKG92ZXJmbG93OnNjcm9sbCwgY29uc3RyYWluZWQgaGVpZ2h0KSA+XG4gIC8vIGlubmVyIGNvbnRlbnQgKGZsZXhHcm93OjEsIGZsZXhTaHJpbms6MCDigJQgZmlsbHMgYXQgbGVhc3QgdGhlIHZpZXdwb3J0XG4gIC8vIGJ1dCBncm93cyBiZXlvbmQgaXQgZm9yIHRhbGwgY29udGVudCkuIGZsZXhHcm93OjEgbGV0cyBjaGlsZHJlbiB1c2VcbiAgLy8gc3BhY2VycyB0byBwaW4gZWxlbWVudHMgdG8gdGhlIGJvdHRvbSBvZiB0aGUgc2Nyb2xsIGFyZWEuIFlvZ2Enc1xuICAvLyBPdmVyZmxvdy5TY3JvbGwgcHJldmVudHMgdGhlIHZpZXdwb3J0IGZyb20gZ3Jvd2luZyB0byBmaXQgdGhlIGNvbnRlbnQuXG4gIC8vIFRoZSByZW5kZXJlciBjb21wdXRlcyBzY3JvbGxIZWlnaHQgZnJvbSB0aGUgY29udGVudCBib3ggYW5kIGN1bGxzXG4gIC8vIGNvbnRlbnQncyBjaGlsZHJlbiBiYXNlZCBvbiBzY3JvbGxUb3AuXG4gIC8vXG4gIC8vIHN0aWNreVNjcm9sbCBpcyBwYXNzZWQgYXMgYSBET00gYXR0cmlidXRlICh2aWEgaW5rLWJveCBkaXJlY3RseSkgc28gaXQnc1xuICAvLyBhdmFpbGFibGUgb24gdGhlIGZpcnN0IHJlbmRlciDigJQgcmVmIGNhbGxiYWNrcyBmaXJlIGFmdGVyIHRoZSBpbml0aWFsXG4gIC8vIGNvbW1pdCwgd2hpY2ggaXMgdG9vIGxhdGUgZm9yIHRoZSBmaXJzdCBmcmFtZS5cbiAgcmV0dXJuIChcbiAgICA8aW5rLWJveFxuICAgICAgcmVmPXtlbCA9PiB7XG4gICAgICAgIGRvbVJlZi5jdXJyZW50ID0gZWxcbiAgICAgICAgaWYgKGVsKSBlbC5zY3JvbGxUb3AgPz89IDBcbiAgICAgIH19XG4gICAgICBzdHlsZT17e1xuICAgICAgICBmbGV4V3JhcDogJ25vd3JhcCcsXG4gICAgICAgIGZsZXhEaXJlY3Rpb246IHN0eWxlLmZsZXhEaXJlY3Rpb24gPz8gJ3JvdycsXG4gICAgICAgIGZsZXhHcm93OiBzdHlsZS5mbGV4R3JvdyA/PyAwLFxuICAgICAgICBmbGV4U2hyaW5rOiBzdHlsZS5mbGV4U2hyaW5rID8/IDEsXG4gICAgICAgIC4uLnN0eWxlLFxuICAgICAgICBvdmVyZmxvd1g6ICdzY3JvbGwnLFxuICAgICAgICBvdmVyZmxvd1k6ICdzY3JvbGwnLFxuICAgICAgfX1cbiAgICAgIHsuLi4oc3RpY2t5U2Nyb2xsID8geyBzdGlja3lTY3JvbGw6IHRydWUgfSA6IHt9KX1cbiAgICA+XG4gICAgICA8Qm94IGZsZXhEaXJlY3Rpb249XCJjb2x1bW5cIiBmbGV4R3Jvdz17MX0gZmxleFNocmluaz17MH0gd2lkdGg9XCIxMDAlXCI+XG4gICAgICAgIHtjaGlsZHJlbn1cbiAgICAgIDwvQm94PlxuICAgIDwvaW5rLWJveD5cbiAgKVxufVxuXG5leHBvcnQgZGVmYXVsdCBTY3JvbGxCb3hcbiJdLCJtYXBwaW5ncyI6IkFBQUEsT0FBT0EsS0FBSyxJQUNWLEtBQUtDLGlCQUFpQixFQUN0QixLQUFLQyxHQUFHLEVBQ1JDLG1CQUFtQixFQUNuQkMsTUFBTSxFQUNOQyxRQUFRLFFBQ0gsT0FBTztBQUNkLGNBQWNDLE1BQU0sUUFBUSxXQUFXO0FBQ3ZDLFNBQVNDLGtCQUFrQixRQUFRLDBCQUEwQjtBQUM3RCxjQUFjQyxVQUFVLFFBQVEsV0FBVztBQUMzQyxTQUFTQyxTQUFTLEVBQUVDLGtCQUFrQixRQUFRLFdBQVc7QUFDekQsU0FBU0MsZUFBZSxRQUFRLGtCQUFrQjtBQUNsRCxjQUFjQyxNQUFNLFFBQVEsY0FBYztBQUMxQyxPQUFPLGdCQUFnQjtBQUN2QixPQUFPQyxHQUFHLE1BQU0sVUFBVTtBQUUxQixPQUFPLEtBQUtDLGVBQWUsR0FBRztFQUM1QkMsUUFBUSxFQUFFLENBQUNDLENBQUMsRUFBRSxNQUFNLEVBQUUsR0FBRyxJQUFJO0VBQzdCQyxRQUFRLEVBQUUsQ0FBQ0MsRUFBRSxFQUFFLE1BQU0sRUFBRSxHQUFHLElBQUk7RUFDOUI7QUFDRjtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7RUFDRUMsZUFBZSxFQUFFLENBQUNDLEVBQUUsRUFBRVosVUFBVSxFQUFFYSxNQUFlLENBQVIsRUFBRSxNQUFNLEVBQUUsR0FBRyxJQUFJO0VBQzFEQyxjQUFjLEVBQUUsR0FBRyxHQUFHLElBQUk7RUFDMUJDLFlBQVksRUFBRSxHQUFHLEdBQUcsTUFBTTtFQUMxQkMsZUFBZSxFQUFFLEdB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
