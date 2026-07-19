import "@testing-library/jest-dom/vitest";

// jsdom does not implement these browser APIs that ChatPage relies on.
// Provide minimal mocks so the component can mount under test.

if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList;
}

// jsdom/happy-dom report clientWidth/clientHeight = 0 by default, which makes
// ChatPage's `syncTerminalMetrics` early-return (hidden-host guard). Give every
// element a non-zero measured size so the fit path is exercisable under test.
Object.defineProperty(HTMLElement.prototype, "clientWidth", {
  configurable: true,
  get() {
    return 800;
  },
});
Object.defineProperty(HTMLElement.prototype, "clientHeight", {
  configurable: true,
  get() {
    return 600;
  },
});
// happy-dom may report isConnected=false for portal/ref elements under test;
// force it true so ChatPage's hidden-host guard does not early-return.
Object.defineProperty(HTMLElement.prototype, "isConnected", {
  configurable: true,
  get() {
    return true;
  },
});

// ChatPage's xterm/websocket effect early-returns unless the session token
// is present (loopback mode) or auth is gated. Provide a test token so the
// component actually wires up xterm and runs its fit path under test.
(window as unknown as Record<string, unknown>).__HERMES_SESSION_TOKEN__ =
  "test-token";

// ResizeObserver is used by ChatPage to debounce host resizes.
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
