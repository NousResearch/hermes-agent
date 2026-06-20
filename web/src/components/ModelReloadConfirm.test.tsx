// @vitest-environment jsdom
//
// Component tests for ModelReloadConfirm. The copy here is the user-visible
// contract described in the issue: by the time the dialog opens, the model
// is *already saved*, and the dialog is asking only about reloading the
// page. The previous copy ("Switching to X starts a fresh chat…") read as
// if Cancel could prevent the save; the new copy must not.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { ModelReloadConfirm } from "./ModelReloadConfirm";

// testing-library doesn't auto-clean in vitest unless globals are enabled.
// Keep the test file self-contained by cleaning the DOM after every test.
afterEach(cleanup);

describe("ModelReloadConfirm copy", () => {
  it("renders the title 'Switch model?' when model is set", () => {
    render(<ModelReloadConfirm model="sonnet" onCancel={() => {}} />);
    expect(screen.getByText("Switch model?")).toBeTruthy();
  });

  it("states that the model is already saved (the dialog is about reloading, not saving)", () => {
    render(<ModelReloadConfirm model="sonnet" onCancel={() => {}} />);
    // The previous copy was "Switching to sonnet starts a fresh chat…",
    // which misled users into thinking Cancel could prevent the save.
    // The new copy must say the model is already saved.
    expect(screen.getByText(/Model saved\./i)).toBeTruthy();
    expect(
      screen.queryByText(/Switching to .* starts a fresh chat/i),
    ).toBeNull();
  });

  it("includes the model name in the default description", () => {
    render(
      <ModelReloadConfirm model="kilo-auto/efficient" onCancel={() => {}} />,
    );
    // The default description should mention the model the user just picked
    // so they know which model the reload will use.
    expect(screen.getByText(/kilo-auto\/efficient/)).toBeTruthy();
  });

  it("honors a caller-provided description override (Models page path)", () => {
    render(
      <ModelReloadConfirm
        model="sonnet"
        onCancel={() => {}}
        description="Custom copy for the Models management page."
      />,
    );
    expect(
      screen.getByText("Custom copy for the Models management page."),
    ).toBeTruthy();
    // The default copy should be replaced, not duplicated.
    expect(screen.queryByText(/Model saved\./i)).toBeNull();
  });
});

describe("ModelReloadConfirm wiring", () => {
  let originalLocation: Location;

  beforeEach(() => {
    originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { ...originalLocation, reload: vi.fn() } as unknown as Location,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
  });

  it("renders nothing when model is null (dialog closed)", () => {
    const { container } = render(
      <ModelReloadConfirm model={null} onCancel={() => {}} />,
    );
    expect(container.textContent).toBe("");
  });

  it("calls window.location.reload on confirm", () => {
    const reloadSpy = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { ...originalLocation, reload: reloadSpy } as unknown as Location,
    });
    render(<ModelReloadConfirm model="sonnet" onCancel={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Reload" }));
    expect(reloadSpy).toHaveBeenCalledTimes(1);
  });

  it("calls onCancel when Cancel is clicked", () => {
    const onCancel = vi.fn();
    render(<ModelReloadConfirm model="sonnet" onCancel={onCancel} />);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
