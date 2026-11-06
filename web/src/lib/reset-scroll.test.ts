// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import { closestScroller, resetNearestScroller } from "./reset-scroll";

function makeChain(): {
	scroller: HTMLElement;
	child: HTMLElement;
	mount(): void;
	cleanup(): void;
} {
	// outer (not scrollable) > scroller (scrollable, offset 1341) > child — the shape
	// ConfigPage's root sits in under the layout scroller.
	const outer = document.createElement("div");
	const scroller = document.createElement("div");
	Object.defineProperty(scroller, "scrollHeight", { value: 2000, configurable: true });
	Object.defineProperty(scroller, "clientHeight", { value: 500, configurable: true });
	scroller.scrollTop = 1341;
	const child = document.createElement("div");
	scroller.appendChild(child);
	outer.appendChild(scroller);
	document.body.append(outer);
	return { scroller, child, mount: () => {}, cleanup: () => scroller.remove() };
}

describe("closestScroller", () => {
	it("walks past non-scrolling wrappers to the nearest scrollable ancestor", () => {
		const { child, scroller, cleanup } = makeChain();
		try {
			expect(closestScroller(child)).toBe(scroller);
		} finally {
			cleanup();
		}
	});

	it("returns null when no ancestor is scrollable", () => {
		const p = document.createElement("p");
		document.body.appendChild(p);
		expect(closestScroller(p)).toBeNull();
	});

	it("treats the element itself as a candidate scroller", () => {
		const scroller = document.createElement("div");
		Object.defineProperty(scroller, "scrollHeight", { value: 900, configurable: true });
		Object.defineProperty(scroller, "clientHeight", { value: 300, configurable: true });
		scroller.scrollTop = 50;
		document.body.appendChild(scroller);
		expect(closestScroller(scroller)).toBe(scroller);
	});
});

describe("resetNearestScroller", () => {
	it("zeroes the nearest scroller's offset and reports it", () => {
		const { scroller, child, cleanup } = makeChain();
		try {
			expect(scroller.scrollTop).toBe(1341);
			expect(resetNearestScroller(child)).toBe(scroller);
			expect(scroller.scrollTop).toBe(0);
		} finally {
			cleanup();
		}
	});

	it("is a no-op returning null when nothing is scrollable", () => {
		const p = document.createElement("p");
		document.body.appendChild(p);
		expect(resetNearestScroller(p)).toBeNull();
	});
});
