// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  ACCESSORY_BAR_HEIGHT_PX,
  PTY_ETX,
  mountPtyMobileAccessory,
  shouldShowMobileAccessory,
  terminalBottomReservePx,
} from "./pty-mobile-accessory";

describe("mobile PTY accessory", () => {
  it("shows Paste and Ctrl+C while the phone keyboard is open or the terminal is focused", () => {
    expect(shouldShowMobileAccessory(320, true, false)).toBe(true);
    expect(shouldShowMobileAccessory(0, true, true)).toBe(true);
    expect(shouldShowMobileAccessory(0, true, false)).toBe(false);
    expect(shouldShowMobileAccessory(320, false, false)).toBe(false);
  });

  it("reserves the accessory bar above the keyboard so the composer is not covered", () => {
    expect(terminalBottomReservePx(320, true)).toBe(320 + ACCESSORY_BAR_HEIGHT_PX);
    expect(terminalBottomReservePx(0, true)).toBe(ACCESSORY_BAR_HEIGHT_PX);
    expect(terminalBottomReservePx(0, false)).toBe(0);
    expect(terminalBottomReservePx(320, false)).toBe(320);
  });

  it("sends a real interrupt byte for the Ctrl+C button", () => {
    expect(PTY_ETX).toBe("\x03");
  });

  it("fires Paste, Ctrl+C, Left, and Right without stealing textarea focus", () => {
    const host = document.createElement("div");
    document.body.append(host);
    const pastes: string[] = [];
    const interrupts: string[] = [];
    const lefts: string[] = [];
    const rights: string[] = [];
    const bar = mountPtyMobileAccessory(host, {
      paste: () => pastes.push("ok"),
      interrupt: () => interrupts.push(PTY_ETX),
      caretLeft: () => lefts.push("L"),
      caretRight: () => rights.push("R"),
    });
    bar.setInset(320, true, true);
    const buttons = [...document.querySelectorAll(".pty-mobile-accessory button")].map((b) => b.textContent);
    expect(buttons).toEqual(["Left", "Right", "Paste", "Ctrl+C"]);
    expect((document.querySelector(".pty-mobile-accessory") as HTMLElement).style.position).toBe("fixed");
    const btn = (label: string) =>
      [...document.querySelectorAll(".pty-mobile-accessory button")].find((b) => b.textContent === label)!;
    btn("Paste").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    btn("Ctrl+C").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    btn("Left").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    btn("Right").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    expect(pastes).toEqual(["ok"]);
    expect(interrupts).toEqual(["\x03"]);
    expect(lefts).toEqual(["L"]);
    expect(rights).toEqual(["R"]);
    bar.dispose();
    host.remove();
  });
});
