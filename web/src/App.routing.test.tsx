// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import type { ComponentType } from "react";
import {
  MemoryRouter,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

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

afterEach(cleanup);

describe("App root plugin routing", () => {
  it("holds the root location while plugins load, then renders a resolved root override", () => {
    const { rerender } = render(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes manifests={[]} loading />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.queryByText("sessions-page")).toBeNull();

    rerender(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes manifests={[rootOverride]} loading={false} />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("location").textContent).toBe("/");
    expect(screen.getByText("plugin:test-dashboard")).toBeTruthy();
  });

  it("redirects to sessions only after loading resolves without a root override", async () => {
    const { rerender } = render(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes manifests={[]} loading />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("location").textContent).toBe("/");

    rerender(
      <MemoryRouter initialEntries={["/"]}>
        <AppRoutes manifests={[]} loading={false} />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("location").textContent).toBe("/sessions");
    });
    expect(screen.getByText("sessions-page")).toBeTruthy();
  });
});
