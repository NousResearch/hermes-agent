import { screen, waitFor, within } from "@testing-library/react";
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

  it("renders User Profile and Memory Notes sections without redundant top metadata cards", async () => {
    renderWithAppProviders(<MemoryPage />);

    expect((await screen.findAllByRole("heading", { name: /memory/i })).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/user profile/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/memory notes/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/built-in only/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^provider$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^directory$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^built-in$/i)).not.toBeInTheDocument();
  });

  it("filters each section independently via the header search inputs", async () => {
    const user = userEvent.setup();
    const response: MemoryResponse = {
      ...baseResponse,
      stores: {
        user: {
          ...baseResponse.stores.user,
          entry_count: 2,
          char_count: 45,
          entries: [
            { id: "user:1", index: 0, content: "User prefers concise replies" },
            { id: "user:2", index: 1, content: "User likes dark mode" },
          ],
        },
        memory: {
          ...baseResponse.stores.memory,
          entry_count: 2,
          char_count: 48,
          entries: [
            { id: "memory:1", index: 0, content: "Project uses Python 3.12" },
            { id: "memory:2", index: 1, content: "Deploy target is staging" },
          ],
        },
      },
    };
    mockApi.getMemory.mockResolvedValue(response);

    renderWithAppProviders(<MemoryPage />);

    await screen.findByRole("button", { name: /user prefers concise replies/i });
    const userStores = screen.getAllByTestId("memory-store-user");
    const memoryStores = screen.getAllByTestId("memory-store-memory");
    const userStore = userStores[userStores.length - 1];
    const memoryStore = memoryStores[memoryStores.length - 1];
    const userSearch = within(userStore).getByRole("textbox", { name: /user profile search/i });
    const memorySearch = within(memoryStore).getByRole("textbox", { name: /memory notes search/i });

    await user.type(userSearch, "dark");
    expect(within(userStore).queryByRole("button", { name: /user prefers concise replies/i })).not.toBeInTheDocument();
    expect(within(userStore).getByRole("button", { name: /user likes dark mode/i })).toBeInTheDocument();
    expect(within(memoryStore).getByRole("button", { name: /project uses python 3.12/i })).toBeInTheDocument();

    await user.type(memorySearch, "staging");
    expect(within(memoryStore).queryByRole("button", { name: /project uses python 3.12/i })).not.toBeInTheDocument();
    expect(within(memoryStore).getByRole("button", { name: /deploy target is staging/i })).toBeInTheDocument();
    expect(within(userStore).getByRole("button", { name: /user likes dark mode/i })).toBeInTheDocument();
  });

  it("expands a row to show full content", async () => {
    const user = userEvent.setup();
    renderWithAppProviders(<MemoryPage />);

    const rowButtons = await screen.findAllByRole("button", { name: /user prefers concise replies/i });
    await user.click(rowButtons[0]);

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
    const userStores = screen.getAllByTestId("memory-store-user");
    const userStore = userStores[userStores.length - 1];
    const composer = within(userStore).getByRole("textbox", { name: /user profile add entry/i });
    await user.type(composer, "New user fact");
    await user.click(within(userStore).getByRole("button", { name: /add entry/i }));

    await waitFor(() => expect(mockApi.addMemoryEntry).toHaveBeenCalledWith("user", "New user fact"));
    expect(screen.getAllByDisplayValue("").length).toBeGreaterThan(0);
  });
});
