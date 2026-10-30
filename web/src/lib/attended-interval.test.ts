// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ATTENDED_IDLE_MS, setAttendedInterval } from "./attended-interval";

let visibility: DocumentVisibilityState = "visible";

function setVisibility(next: DocumentVisibilityState) {
  visibility = next;
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("setAttendedInterval", () => {
  let stop: () => void = () => {};

  beforeEach(() => {
    vi.useFakeTimers();
    visibility = "visible";
    vi.spyOn(document, "visibilityState", "get").mockImplementation(
      () => visibility,
    );
  });

  afterEach(() => {
    stop();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("polls while the tab is visible and in use", () => {
    const fn = vi.fn();
    stop = setAttendedInterval(fn, 1_000);

    vi.advanceTimersByTime(3_000);

    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("stops while the tab is hidden and catches up once it shows again", () => {
    const fn = vi.fn();
    stop = setAttendedInterval(fn, 1_000);

    setVisibility("hidden");
    vi.advanceTimersByTime(10_000);
    expect(fn).not.toHaveBeenCalled();

    setVisibility("visible");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("stops on a visible tab nobody has touched, and resumes on input", () => {
    const fn = vi.fn();
    stop = setAttendedInterval(fn, 60_000);

    vi.advanceTimersByTime(ATTENDED_IDLE_MS + 5 * 60_000);
    const whileIdle = fn.mock.calls.length;
    vi.advanceTimersByTime(10 * 60_000);
    expect(fn).toHaveBeenCalledTimes(whileIdle);

    window.dispatchEvent(new Event("pointerdown"));
    expect(fn).toHaveBeenCalledTimes(whileIdle + 1);
  });

  it("does not fire when an idle tab is hidden", () => {
    const fn = vi.fn();
    stop = setAttendedInterval(fn, 60_000);
    vi.advanceTimersByTime(ATTENDED_IDLE_MS + 60_000);
    const whileIdle = fn.mock.calls.length;

    setVisibility("hidden");

    expect(fn).toHaveBeenCalledTimes(whileIdle);
  });
});
