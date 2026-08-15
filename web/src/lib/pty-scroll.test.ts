import { describe, expect, it } from "vitest";

import { isViewportPinnedToBottom, shouldFollowPtyOutput } from "./pty-scroll";

describe("isViewportPinnedToBottom", () => {
	it("is pinned when the viewport sits on the bottom row", () => {
		// xterm reports viewportY === baseY when the newest output is on screen.
		expect(isViewportPinnedToBottom({ viewportY: 120, baseY: 120 })).toBe(true);
	});

	it("releases the pin once the user scrolls up into the backlog", () => {
		// Scrolling up drops viewportY below baseY — the user is reading history,
		// so the resume replay must not yank them back down (#59591 follow-up).
		expect(isViewportPinnedToBottom({ viewportY: 40, baseY: 120 })).toBe(false);
	});

	it("stays pinned if viewportY overshoots baseY while rows are trimmed", () => {
		// scrollback eviction can momentarily push viewportY past baseY.
		expect(isViewportPinnedToBottom({ viewportY: 121, baseY: 120 })).toBe(true);
	});

	it("treats a fresh 0x0 buffer as pinned", () => {
		expect(isViewportPinnedToBottom({ viewportY: 0, baseY: 0 })).toBe(true);
	});
});

describe("shouldFollowPtyOutput", () => {
	it("follows replayed output while resuming and stuck to the bottom", () => {
		// The core #59591 fix: scroll to bottom as each replay chunk commits.
		expect(shouldFollowPtyOutput("sess-123", true)).toBe(true);
	});

	it("stops following once the user has scrolled up mid-replay", () => {
		expect(shouldFollowPtyOutput("sess-123", false)).toBe(false);
	});

	it("does not follow a fresh (non-resume) session", () => {
		// Fresh chats start empty; forcing scroll would fight normal cursor output.
		expect(shouldFollowPtyOutput(null, true)).toBe(false);
	});

	it("does not follow a fresh session even when stickToBottom is true", () => {
		expect(shouldFollowPtyOutput(null, false)).toBe(false);
	});

	it("treats an empty resume param as non-resume", () => {
		expect(shouldFollowPtyOutput("", true)).toBe(false);
	});
});
