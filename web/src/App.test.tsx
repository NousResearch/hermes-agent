// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, describe, expect, it } from "vitest";

import { UnknownRouteFallback } from "./App";

function UnknownRoute({ pluginsLoading }: { pluginsLoading: boolean }) {
  const location = useLocation();

  return (
    <>
      <output data-testid="location">{location.pathname}</output>
      <UnknownRouteFallback pluginsLoading={pluginsLoading} />
    </>
  );
}

function RouteHarness({ pluginsLoading }: { pluginsLoading: boolean }) {
  return (
    <MemoryRouter initialEntries={["/life-os"]}>
      <Routes>
        <Route path="/sessions" element={<output data-testid="location">/sessions</output>} />
        <Route path="*" element={<UnknownRoute pluginsLoading={pluginsLoading} />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("plugin deep-link fallback", () => {
  let container: HTMLDivElement | undefined;
  let root: Root | undefined;

  afterEach(() => {
    act(() => root?.unmount());
    container?.remove();
    container = undefined;
    root = undefined;
  });

  it("keeps an unregistered plugin route visible while manifests load", () => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    act(() => root?.render(<RouteHarness pluginsLoading />));

    expect(container.querySelector('[data-testid="location"]')?.textContent).toBe("/life-os");
    expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
    expect(container.textContent).toContain("Loading");
  });

  it("navigates an unknown route to sessions after manifests finish loading", () => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    act(() => root?.render(<RouteHarness pluginsLoading />));
    act(() => root?.render(<RouteHarness pluginsLoading={false} />));

    expect(container.querySelector('[data-testid="location"]')?.textContent).toBe("/sessions");
    expect(container.querySelector('[aria-busy="true"]')).toBeNull();
  });
});
