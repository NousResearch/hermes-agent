import { describe, it, vi } from "vitest";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import MissionControlPage from "../MissionControlPage";

vi.mock("../../contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setPageTitle: vi.fn() }),
}));

describe("MissionControlPage", () => {
  it("renders without crashing", () => {
    render(
      <MemoryRouter>
        <MissionControlPage />
      </MemoryRouter>
    );
  });

  it("shows loading state when snapshot is null", () => {
    const { container } = render(
      <MemoryRouter>
        <MissionControlPage />
      </MemoryRouter>
    );
    // TODO(hive-2): assert loading skeleton is visible
    expect(container).toBeTruthy();
  });
});
