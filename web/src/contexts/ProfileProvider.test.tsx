// @vitest-environment jsdom
import { act, useContext } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getActiveProfile: vi.fn(),
  getProfiles: vi.fn(),
}));
const setManagementProfile = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", () => ({ api: apiMocks, setManagementProfile }));

import { ProfileContext } from "@/contexts/profile-context";
import { ProfileProvider } from "@/contexts/ProfileProvider";

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function Probe() {
  const { profile, setProfile } = useContext(ProfileContext);
  const { search } = useLocation();
  return <button onClick={() => setProfile("later")}>{profile}|{search}</button>;
}

async function render(entry: string) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <MemoryRouter initialEntries={[entry]}>
      <ProfileProvider><Probe /></ProfileProvider>
    </MemoryRouter>,
  ));
}

beforeEach(() => {
  apiMocks.getProfiles.mockResolvedValue({ profiles: [] });
  apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
  setManagementProfile.mockClear();
  Object.defineProperty(window, "__HERMES_INITIAL_PROFILE__", {
    configurable: true,
    value: "",
  });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ProfileProvider launch bootstrap", () => {
  it("uses the server profile on the first bare-chat render", async () => {
    Object.defineProperty(window, "__HERMES_INITIAL_PROFILE__", {
      configurable: true,
      value: "worker_x",
    });

    await render("/chat");

    expect(container.textContent).toBe("worker_x|?profile=worker_x");
    expect(setManagementProfile).toHaveBeenCalledWith("worker_x");
  });

  it("lets an explicit URL profile override the launch bootstrap", async () => {
    Object.defineProperty(window, "__HERMES_INITIAL_PROFILE__", {
      configurable: true,
      value: "worker_x",
    });

    await render("/chat?profile=other");

    expect(container.textContent).toBe("other|?profile=other");
    expect(setManagementProfile).toHaveBeenCalledWith("other");
  });

  it("keeps the unscoped default and allows later profile changes", async () => {
    await render("/chat");

    expect(container.textContent).toBe("|");
    await act(async () => container.querySelector("button")?.click());
    expect(container.textContent).toBe("later|?profile=later");
  });
});
