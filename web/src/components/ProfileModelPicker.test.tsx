// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ProfileModelPicker } from "./ProfileModelPicker";
import {
  filterProfileModelChoices,
  type ProfileModelChoice,
} from "@/lib/profile-model-picker";

const choices: ProfileModelChoice[] = [
  {
    provider: "nous",
    model: "deepseek-v4-pro",
    label: "Nous Portal · DeepSeek V4 Pro",
  },
  {
    provider: "openai-codex",
    model: "gpt-5.6-codex",
    label: "OpenAI Codex · GPT-5.6 Codex",
  },
];
const scrollIntoView = vi.fn();

beforeEach(() => {
  scrollIntoView.mockReset();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    value: scrollIntoView,
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ProfileModelPicker", () => {
  it.each([
    ["NOUS", "deepseek-v4-pro"],
    ["V4-PRO", "deepseek-v4-pro"],
    ["deepseek v4 pro", "deepseek-v4-pro"],
    ["OPENAI-CODEX", "gpt-5.6-codex"],
  ])(
    "matches provider, model id, and rendered label case-insensitively",
    (query, model) => {
      expect(
        filterProfileModelChoices(choices, query).map((choice) => choice.model),
      ).toEqual([model]);
    },
  );

  it("filters choices and selects a result", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <ProfileModelPicker
        choices={choices}
        emptyLabel="No models"
        inheritLabel="Use default"
        loadingLabel="Loading models"
        selected={"nous\u0000deepseek-v4-pro"}
        onSelect={onSelect}
      />,
    );

    await user.type(screen.getByRole("textbox", { name: "Search models" }), "codex");

    expect(screen.queryByRole("option", { name: /DeepSeek/ })).toBeNull();
    const result = screen.getByRole("option", { name: /GPT-5.6 Codex/ });
    await user.click(result);

    expect(onSelect).toHaveBeenCalledWith("openai-codex\u0000gpt-5.6-codex");
  });

  it("uses single-select semantics and keyboard navigation", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <ProfileModelPicker
        choices={choices}
        emptyLabel="No models"
        inheritLabel="Use default"
        loadingLabel="Loading models"
        selected={"nous\u0000deepseek-v4-pro"}
        onSelect={onSelect}
      />,
    );

    const search = screen.getByRole("textbox", { name: "Search models" });
    const selected = screen.getByRole("option", { name: /DeepSeek/ });
    const options = screen.getAllByRole("option");
    expect(screen.getByRole("listbox", { name: "Models" })).toBeTruthy();
    expect(options.every((option) => option.tagName === "BUTTON")).toBe(true);
    expect(options.every((option) => option.getAttribute("type") === "button")).toBe(
      true,
    );
    expect(selected.getAttribute("aria-selected")).toBe("true");
    expect(selected.getAttribute("aria-pressed")).toBeNull();

    await user.click(search);
    await user.keyboard("{ArrowDown}");
    expect(document.activeElement).toBe(selected);

    await user.click(search);
    await user.clear(search);
    await user.type(search, "codex");
    await user.keyboard("{Enter}");
    expect(onSelect).toHaveBeenLastCalledWith(
      "openai-codex\u0000gpt-5.6-codex",
    );
  });

  it("scrolls the selected option into view", () => {
    render(
      <ProfileModelPicker
        choices={choices}
        emptyLabel="No models"
        loadingLabel="Loading models"
        selected={"openai-codex\u0000gpt-5.6-codex"}
        onSelect={() => undefined}
      />,
    );

    expect(scrollIntoView).toHaveBeenCalledWith({ block: "nearest" });
  });
});
