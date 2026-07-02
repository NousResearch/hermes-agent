import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { UnauthorizedState } from "@/components/UnauthorizedState";

describe("UnauthorizedState", () => {
  it("renders a human-readable unauthorized message with retry", async () => {
    const retry = vi.fn();

    render(<UnauthorizedState service="Kanban" onRetry={retry} />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Authentication required. Sign in again or retry the request.",
    );
    expect(screen.queryByText("t_001")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("renders a login affordance when loginUrl is provided", () => {
    render(<UnauthorizedState service="Profiles" loginUrl="/login" />);

    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute(
      "href",
      "/login",
    );
  });
});
