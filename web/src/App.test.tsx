import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { UnknownRouteFallback } from "./App";

describe("plugin deep-link fallback", () => {
  it("keeps an unregistered plugin route visible while manifests load", () => {
    const html = renderToStaticMarkup(
      <MemoryRouter initialEntries={["/life-os"]}>
        <Routes>
          <Route
            path="*"
            element={<UnknownRouteFallback pluginsLoading />}
          />
        </Routes>
      </MemoryRouter>,
    );

    expect(html).toContain("aria-busy=\"true\"");
    expect(html).toContain("Loading");
  });

  it("keeps the normal redirect path after manifests finish loading", () => {
    const html = renderToStaticMarkup(
      <MemoryRouter initialEntries={["/life-os"]}>
        <Routes>
          <Route
            path="*"
            element={<UnknownRouteFallback pluginsLoading={false} />}
          />
        </Routes>
      </MemoryRouter>,
    );

    expect(html).not.toContain("Loading");
  });
});
