// @vitest-environment jsdom
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";

const state = vi.hoisted(() => ({
  profile: "eligible",
  probe: vi.fn(),
  setAfterTitle: vi.fn(),
  setEnd: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: {
  getSkills: async () => [],
  getToolsets: async () => [],
  getWisdomEntitlement: state.probe,
} }));
vi.mock("@/contexts/useProfileScope", () => ({ useProfileScope: () => ({ profile: state.profile }) }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => state }));
vi.mock("@/components/ToolsetConfigDrawer", () => ({ ToolsetConfigDrawer: () => null }));
vi.mock("@/components/SkillEditorDialog", () => ({ SkillEditorDialog: () => null }));
vi.mock("@/components/CollectiveWisdomPanel", () => ({ CollectiveWisdomPanel: () => null }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

import SkillsPage from "./SkillsPage";
import { en } from "@/i18n/en";

const pageView = () => <MemoryRouter><SkillsPage /></MemoryRouter>;

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  state.profile = "eligible";
  state.probe.mockReset();
});

it("does not reuse entitlement after switching away and back with pending probes", async () => {
  state.probe.mockResolvedValue({ entitled: true, expires_at: Date.now() / 1000 + 60 });
  const page = render(pageView());
  expect(await screen.findByRole("button", { name: en.skills.wisdom.tab })).toBeTruthy();

  state.probe.mockImplementation(() => new Promise(() => {}));
  state.profile = "other";
  page.rerender(pageView());
  expect(screen.queryByRole("button", { name: en.skills.wisdom.tab })).toBeNull();
  state.profile = "eligible";
  page.rerender(pageView());
  expect(screen.queryByRole("button", { name: en.skills.wisdom.tab })).toBeNull();
});

it("hides entitlement at expiry even while the next probe is pending", async () => {
  vi.useFakeTimers();
  state.probe.mockResolvedValueOnce({ entitled: true, expires_at: Date.now() / 1000 + 16 });
  state.probe.mockImplementation(() => new Promise(() => {}));
  await act(async () => { render(pageView()); });
  expect(screen.queryByRole("button", { name: en.skills.wisdom.tab })).not.toBeNull();
  await act(async () => { await vi.advanceTimersByTimeAsync(16_001); });
  expect(state.probe).toHaveBeenCalledTimes(2);
  expect(screen.queryByRole("button", { name: en.skills.wisdom.tab })).toBeNull();
});
