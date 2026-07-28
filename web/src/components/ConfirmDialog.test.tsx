// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ConfirmDialog } from "./ConfirmDialog";

afterEach(cleanup);

describe("ConfirmDialog typed confirmation", () => {
  it("requires an exact phrase and clears it after reopening", () => {
    const onConfirm = vi.fn();
    const props = {
      onCancel: vi.fn(),
      onConfirm,
      title: "Restart gateway?",
      typedConfirmation: "RESTART",
    };
    const { rerender } = render(<ConfirmDialog {...props} open />);

    const confirm = screen.getByRole("button", { name: "Confirm" });
    const input = screen.getByLabelText(/Type RESTART to confirm/i);
    expect((confirm as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(input, { target: { value: "restart" } });
    expect((confirm as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(input, { target: { value: "RESTART" } });
    expect((confirm as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalledOnce();

    rerender(<ConfirmDialog {...props} open={false} />);
    rerender(<ConfirmDialog {...props} open />);
    expect(
      (screen.getByLabelText(/Type RESTART to confirm/i) as HTMLInputElement)
        .value,
    ).toBe("");
  });
});
