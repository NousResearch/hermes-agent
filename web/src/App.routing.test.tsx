// @vitest-environment jsdom

import { act, type ComponentType } from "react";
import { createRoot, type Root } from "react-dom/client";
import {
  MemoryRouter,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true;

vi.mock("@/plugins", () => ({
  PluginPage: ({ name }: { name: string }) => <div>plugin:{name}</div>,
  PluginSlot: () => null,
  usePlugins: vi.fn(() => ({ manifests: [], plugins: [], loading: false })),
}));

import { buildRoutes } from "./lib/dashboard-routes";
import type { PluginManifest } from "./plugins";

function RootRedirect() {
  return <Navigate to="/sessions" replace />;
}

const builtinRoutes: Record<string, ComponentType> = {
  "/": RootRedirect,
  "/sessions": () => <div>sessions-page</div>,
};

const rootOverride: PluginManifest = {
  name: "test-dashboard",
  label: "Test Dashboard",
  description: "Router regression fixture",
  icon: "LayoutDashboard",
  version: "1.0.0",
  tab: { path: "/test-dashboard", override: "/", hidden: true },
  entry: "index.js",
  has_api: false,
  source: "test",
};

function LocationProbe() {
  const { pathname } = useLocation();
  return <output data-testid="location">{pathname}</output>;
}

function AppRoutes({
  manifests,
  loading,
}: {
  manifests: PluginManifest[];
  loading: boolean;
}) {
  const routes = buildRoutes(builtinRoutes, manifests, loading);
  return (
    <>
      <LocationProbe />
      <Routes>
        {routes.map((route) => (
          <Route key={route.key} path={route.path} element={route.element} />
        ))}
      </Routes>
    </>
  );
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function renderApp(manifests: PluginManifest[], loading: boolean) {
  await act(async () => {
    root.render(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes manifests={manifests} loading={loading} />
      </MemoryRouter>,
    );
  });
}

function locationPath() {
  return container.querySelector<HTMLOutputElement>("[data-testid='location']")
    ?.textContent;
}

describe("App root plugin routing", () => {
  it("holds the root location while plugins load, then renders a resolved root override", async () => {
    await renderApp([], true);

    expect(locationPath()).toBe("/");
    expect(container.textContent).not.toContain("sessions-page");

    await renderApp([rootOverride], false);

    expect(locationPath()).toBe("/");
    expect(container.textContent).toContain("plugin:test-dashboard");
  });

  it("redirects to sessions only after loading resolves without a root override", async () => {
    await renderApp([], true);

    expect(locationPath()).toBe("/");

    await renderApp([], false);

    expect(locationPath()).toBe("/sessions");
    expect(container.textContent).toContain("sessions-page");
  });
});
