import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MemoryPage from "@/pages/MemoryPage";
import { renderWithAppProviders } from "@/test/render";
import type { MemoryResponse } from "@/lib/api";

const mockApi = vi.hoisted(() => ({
  getMemory: vi.fn(),
  addMemoryEntry: vi.fn(),
  updateMemoryEntry: vi.fn(),
  removeMemoryEntry: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: mockApi,
}));

const baseResponse: MemoryResponse = {
  builtin_active: true,
  provider: "",
  provider_label: "built-in only",
  directory: "/tmp/memories",
  note: "Saved immediately. Changes apply to future sessions; current sessions keep their existing snapshot.",
  stores: {
    user: {
      path: "/tmp/memories/USER.md",
      entry_count: 1,
      char_count: 25,
      char_limit: 1375,
      updated_at: null,
      entries: [{ id: "user:abc123def4567890", index: 0, content: "User prefers concise replies" }],
    },
    memory: {
      path: "/tmp/memories/MEMORY.md",
      entry_count: 1,
      char_count: 24,
      char_limit: 2200,
      updated_at: null,
      entries: [{ id: "memory:fedcba0987654321", index: 0, content: "Project uses Python 3.12" }],
    },
  },
};

describe("MemoryPage", () => {
  beforeEach(() => {
    mockApi.getMemory.mockResolvedValue(baseResponse);
    mockApi.addMemoryEntry.mockResolvedValue(baseResponse);
    mockApi.updateMemoryEntry.mockResolvedValue(baseResponse);
    mockApi.removeMemoryEntry.mockResolvedValue(baseResponse);
  });

  it("renders User Profile and Memory Notes sections", async () => {
    renderWithAppProviders(<MemoryPage />);

    expect((await screen.findAllByRole("heading", { name: /memory/i })).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/user profile/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/memory notes/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/built-in only/i).length).toBeGreaterThan(0);
  });

  it("expands a row to show full content", async () => {
    const user = userEvent.setup();
    renderWithAppProviders(<MemoryPage />);

    const rowButton = await screen.findByRole("button", { name: /user prefers concise replies/i });
    await user.click(rowButton);

    expect(screen.getAllByText(/user prefers concise replies/i).length).toBeGreaterThan(1);
    expect(screen.getByRole("button", { name: /edit/i })).toBeInTheDocument();
  });

  it("adds a new entry and clears the composer", async () => {
    const user = userEvent.setup();
    const addResponse: MemoryResponse = {
      ...baseResponse,
      stores: {
        ...baseResponse.stores,
        user: {
          ...baseResponse.stores.user,
          entry_count: 2,
          char_count: 40,
          entries: [
            ...baseResponse.stores.user.entries,
            { id: "user:1", index: 1, content: "New user fact" },
          ],
        },
      },
    };
    mockApi.addMemoryEntry.mockResolvedValueOnce(addResponse);

    renderWithAppProviders(<MemoryPage />);

    await screen.findAllByRole("heading", { name: /memory/i });
    const textareas = screen.getAllByRole("textbox");
    await user.type(textareas[0], "New user fact");
    await user.click(screen.getAllByRole("button", { name: /add entry/i })[0]);

    await waitFor(() => expect(mockApi.addMemoryEntry).toHaveBeenCalledWith("user", "New user fact"));
    expect(screen.getAllByDisplayValue("").length).toBeGreaterThan(0);
  });
});
