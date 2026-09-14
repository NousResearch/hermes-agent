// @vitest-environment jsdom
import { describe, expect, it } from "vitest";

import {
  ACCESSORY_BAR_HEIGHT_PX,
  PTY_ETX,
  dispatchPtyArrowKey,
  mountPtyMobileAccessory,
  shouldShowMobileAccessory,
  terminalBottomReservePx,
} from "./pty-mobile-accessory";

describe("mobile PTY accessory", () => {
  it("shows the terminal key bar while the phone keyboard is open or the terminal is focused", () => {
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

  it("dispatches Up and Down through xterm's helper textarea", () => {
    const host = document.createElement("div");
    const xterm = document.createElement("div");
    const textarea = document.createElement("textarea");
    xterm.className = "xterm";
    textarea.className = "xterm-helper-textarea";
    xterm.append(textarea);
    host.append(xterm);
    document.body.append(host);

    const keys: string[] = [];
    textarea.addEventListener("keydown", (event) => keys.push(event.key));

    expect(dispatchPtyArrowKey(host, "ArrowUp")).toBe(true);
    expect(dispatchPtyArrowKey(host, "ArrowDown")).toBe(true);
    expect(keys).toEqual(["ArrowUp", "ArrowDown"]);

    host.remove();
  });

  it("fires Up, Down, Left, Right, Paste, and Ctrl+C without stealing textarea focus", () => {
    const host = document.createElement("div");
    const xterm = document.createElement("div");
    const textarea = document.createElement("textarea");
    xterm.className = "xterm";
    textarea.className = "xterm-helper-textarea";
    xterm.append(textarea);
    host.append(xterm);
    document.body.append(host);

    const pastes: string[] = [];
    const interrupts: string[] = [];
    const lefts: string[] = [];
    const rights: string[] = [];
    const arrows: string[] = [];
    textarea.addEventListener("keydown", (event) => arrows.push(event.key));

    const bar = mountPtyMobileAccessory(host, {
      paste: () => pastes.push("ok"),
      interrupt: () => interrupts.push(PTY_ETX),
      caretLeft: () => lefts.push("L"),
      caretRight: () => rights.push("R"),
    });
    bar.setInset(320, true, true);

    const buttons = [...document.querySelectorAll(".pty-mobile-accessory button")].map((b) => b.textContent);
    expect(buttons).toEqual(["↑", "↓", "←", "→", "Paste", "Ctrl+C"]);
    expect((document.querySelector(".pty-mobile-accessory") as HTMLElement).style.position).toBe("fixed");

    const btn = (label: string) =>
      [...document.querySelectorAll(".pty-mobile-accessory button")].find((b) => b.textContent === label)!;
    btn("↑").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    btn("↓").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    btn("←").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    btn("→").dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    btn("Paste").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    btn("Ctrl+C").dispatchEvent(new MouseEvent("click", { bubbles: true }));

    expect(arrows).toEqual(["ArrowUp", "ArrowDown"]);
    expect(pastes).toEqual(["ok"]);
    expect(interrupts).toEqual(["\x03"]);
    expect(lefts).toEqual(["L"]);
    expect(rights).toEqual(["R"]);
    expect(document.activeElement).toBe(textarea);

    bar.dispose();
    host.remove();
  });
});
