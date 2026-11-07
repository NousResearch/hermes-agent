/**
 * Dashboard scroll-reset helpers.
 *
 * The dashboard's route outlet `<main>` (PageHeaderProvider, non-chat routes)
 * is one persistent scrollable container shared by every page and every Config
 * category. When ConfigPage swaps `activeCategory` the field list changes but
 * that ancestor's `scrollTop` stays where the previous (possibly much longer)
 * category left it, so a shorter category renders its header off-screen — on
 * tall screens the panel looks empty (#149966).
 *
 * The fix keeps this decision in a pure walk so a component test can drive a
 * real DOM and a page can re-run it on every dependant change without
 * duplicating the ancestor-scan.
 */

/** A scroller the reset should target: scrollable in the block axis, taller than the viewport. */
function isScrollable(el: HTMLElement): boolean {
	return el.scrollHeight > el.clientHeight;
}

/**
 * The nearest scrollable ancestor of `el`, `el` itself included; `null` when
 * none is scrollable. Resembles the DOM's `closest()` but keyed on scroll
 * overflow, which `closest()` cannot express.
 */
export function closestScroller(el: HTMLElement | null): HTMLElement | null {
	let probe = el;
	while (probe) {
		if (isScrollable(probe)) return probe;
		probe = probe.parentElement;
	}
	return null;
}

/**
 * Reset the nearest scrollable ancestor of `el` (inclusive) to its top. Returns
 * the element that was reset, or `null` when no scroller matched — callers
 * that need one more nudge (window.scroll) can branch on it.
 */
export function resetNearestScroller(el: HTMLElement | null): HTMLElement | null {
	const scroller = closestScroller(el);
	if (!scroller) return null;
	scroller.scrollTop = 0;
	return scroller;
}
