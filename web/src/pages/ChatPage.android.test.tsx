// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";


class FakeFitAddon {
  activate() {}
  dispose() {}
  fit() {}
}

const webglInstances: FakeWebglAddon[] = [];

class FakeWebglAddon {
  disposed = false;
  onContextLoss() {
    return { dispose() {} };
  }
  dispose() {
    this.disposed = true;
  }
  constructor() {
    webglInstances.push(this);
  }
}

const maybeReloadForLoopbackWsAuthFailure = vi.fn(() => false);
const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => "ws://localhost/api/pty?channel=chat-1"),
  fetchJSON: vi.fn(async () => ({ valid: true })),
}));
const uploadChatImage = vi.hoisted(() =>
  vi.fn(async () => ({ path: "/tmp/pasted.png" })),
);

vi.mock("@/lib/chatImagePaste", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chatImagePaste")>()),
  uploadChatImage,
}));

vi.mock("@xterm/addon-fit", () => ({ FitAddon: FakeFitAddon }));


vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: FakeWebglAddon }));

vi.mock("@/components/ChatSidebar", () => ({
  ChatSidebar: () => null,
}));
vi.mock("@/components/ChatSessionList", () => ({
  ChatSessionList: () => null,
}));
vi.mock("@/components/Backdrop", () => ({ Backdrop: () => null }));
vi.mock("@/plugins", () => ({
  PluginSlot: () => null,
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn(), setTitle: vi.fn() }),
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "" }),
}));
vi.mock("@/themes", () => ({
  useTheme: () => ({ theme: { terminalBackground: "#000000" } }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      app: {
        closeModelTools: "Close model tools",
        modelToolsSheetSubtitle: "Tools",
        modelToolsSheetTitle: "Model",
      },
    },
  }),
}));
vi.mock("@/lib/dashboard-auth-reload", () => ({
  maybeReloadForLoopbackWsAuthFailure,
}));
vi.mock("@/lib/api", () => ({
  api: apiMocks,
  buildWsUrl: apiMocks.buildWsUrl,
  fetchJSON: apiMocks.fetchJSON,
}));

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static OPEN = 1;

  binaryType = "blob";
  onclose: ((event: CloseEventLike) => void) | null = null;
  onmessage: ((event: { data: ArrayBuffer | string }) => void) | null = null;
  onopen: (() => void) | null = null;
  readyState = FakeWebSocket.OPEN;
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.readyState = 3;
  }

  send = vi.fn();
}

type CloseEventLike = {
  code: number;
  reason: string;
  wasClean: boolean;
};

let container: HTMLDivElement;
let root: Root;

// jsdom runs without an origin here (per-file @vitest-environment jsdom on a
// node-default config), so localStorage is undefined. Stub it so components
// that persist UI state (side panel collapse) can be exercised.
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

// React only routes updates through act() when this flag is set; without it
// the isActive re-renders in the keyboard-inset gate test warn.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  webglInstances.length = 0;
  maybeReloadForLoopbackWsAuthFailure.mockClear();
  apiMocks.buildWsUrl.mockReset();
  apiMocks.buildWsUrl.mockResolvedValue("ws://localhost/api/pty?channel=chat-1");
  apiMocks.fetchJSON.mockReset();
  apiMocks.fetchJSON.mockResolvedValue({ valid: true });
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      disconnect() {}
      observe() {}
      unobserve() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    void cb;
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal("matchMedia", () => ({
    addListener() {},
    removeListener() {},
    addEventListener() {},
    matches: false,
    media: "",
    removeEventListener() {},
  }));
  vi.stubGlobal("crypto", {
    getRandomValues: (values: Uint8Array) => {
      values.fill(7);
      return values;
    },
    randomUUID: () => "chat-test-id",
  });

  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: { addEventListener() {}, removeEventListener() {}, width: 1280 },
  });
  Object.defineProperty(window, "__HERMES_SESSION_TOKEN__", {
    configurable: true,
    value: "stale-token",
    writable: true,
  });
  Object.defineProperty(window, "__HERMES_AUTH_REQUIRED__", {
    configurable: true,
    value: false,
    writable: true,
  });
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: {
      readText: vi.fn(async () => ""),
      writeText: vi.fn(async () => {}),
    },
  });
  Object.defineProperty(navigator, "userAgent", { configurable: true, value: "Mozilla/5.0 (Linux; Android 15) Chrome/140.0 Mobile" });
  sessionStorage.clear();
  vi.stubGlobal("localStorage", localStorageMock);
  localStorageMock.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});


const settle = () => new Promise((resolve) => setTimeout(resolve, 35));
async function mountedTerminal() {
 const { default: ChatPage } = await import("./ChatPage");
 await render(<MemoryRouter initialEntries={["/chat"]}><ChatPage isActive /></MemoryRouter>);
 const ws = FakeWebSocket.instances[0];
 await act(async () => { ws.onopen?.(); });
 const ta = container.querySelector<HTMLTextAreaElement>(".xterm-helper-textarea")!;
 expect(ta).not.toBeNull();
 ws.send.mockClear();
 return {ta, ws};
}
async function softInput(ta: HTMLTextAreaElement, data: string, inputType = "insertText", value = ta.value + data) {
 ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Unidentified", keyCode: 229, bubbles: true}));
 ta.dispatchEvent(new InputEvent("beforeinput", {inputType, data, bubbles: true}));
 ta.value = value;
 ta.dispatchEvent(new InputEvent("input", {inputType, data, bubbles: true, composed: true}));
 await settle();
 ta.dispatchEvent(new KeyboardEvent("keyup", {key: "Unidentified", keyCode: 229, bubbles: true}));
}
function plainEdit(ta: HTMLTextAreaElement, data: string, options: InputEventInit = {}) {
 const init = {inputType: "insertText", data, bubbles: true, composed: true, ...options};
 ta.dispatchEvent(new InputEvent("beforeinput", init));
 ta.value += data;
 ta.dispatchEvent(new InputEvent("input", init));
}
function sent(ws: FakeWebSocket) { return ws.send.mock.calls.map(c => c[0]).join(""); }
describe("actual ChatPage with real xterm Android keyboard", () => {
 it.each([false, true])("does not lose no-keydown text following paste during an active composition (late end=%s)", async (lateEnd) => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  const paste = new Event("paste", {bubbles: true, cancelable: true});
  Object.defineProperty(paste, "clipboardData", {value: {
   types: ["text/plain"], items: [], files: [], getData: () => "P",
  }});
  ta.dispatchEvent(paste);
  plainEdit(ta, "x");
  if (lateEnd) await settle();
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("éPx");
 });
 it.each([1, 7, 19, 73])("preserves seeded mixed-input transcripts (seed=%s)", async (seed) => {
  const {ta, ws} = await mountedTerminal();
  let state = seed;
  let expected = "";
  for (let i = 0; i < 48; i++) {
   state = (state * 1664525 + 1013904223) >>> 0;
   const kind = state % 4;
   const text = ["a", "a", "é", "🙂"][kind];
   if (kind === 0) {
    ta.dispatchEvent(new KeyboardEvent("keydown", {key: "a", keyCode: 65, bubbles: true}));
    plainEdit(ta, text);
    ta.dispatchEvent(new KeyboardEvent("keyup", {key: "a", keyCode: 65, bubbles: true}));
   } else if (kind === 1) {
    ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Process", keyCode: 229, bubbles: true}));
    plainEdit(ta, text);
   } else if (kind === 2) {
    ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
    ta.value += text;
    ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: text, bubbles: true}));
    ta.dispatchEvent(new CompositionEvent("compositionend", {data: text, bubbles: true}));
   } else {
    plainEdit(ta, text);
   }
   expected += text;
   if ((state >>> 8) % 7 === 0) {
    await settle();
    expect(sent(ws)).toBe(expected);
   }
  }
  await settle();
  expect(sent(ws)).toBe(expected);
  expect(ta.value).toBe(expected);
 });
 it("retains the commit before native DOM paste during the completion fence", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  const paste = new Event("paste", {bubbles: true, cancelable: true});
  Object.defineProperty(paste, "clipboardData", {value: {
   types: ["text/plain"], items: [], files: [], getData: () => "P",
  }});
  ta.dispatchEvent(paste);
  await settle();
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("éPx");
 });
 it("does not forward cancelled preedit or duplicate compositionend", async () => {
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "a");
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value += "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  ta.value = "a";
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "", bubbles: true}));
  await settle();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value += "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("aé");
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("aéx");
 });
 it("uses actual hardware keypress delivery instead of a modifier heuristic", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "A", keyCode: 65, bubbles: true}));
  ta.dispatchEvent(new KeyboardEvent("keypress", {key: "A", keyCode: 65, charCode: 65, bubbles: true}));
  plainEdit(ta, "A");
  ta.dispatchEvent(new KeyboardEvent("keyup", {key: "A", keyCode: 65, bubbles: true}));
  plainEdit(ta, "A");
  await settle();
  expect(sent(ws)).toBe("AA");
 });
 it("owns input-only edits, including repeated Unicode characters", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const data of ["a", "a", "🙂", "🙂", "ع"]) {
   ta.value += data;
   ta.dispatchEvent(new InputEvent("input", {inputType: "insertText", data, bubbles: true}));
  }
  await settle();
  expect(sent(ws)).toBe("aa🙂🙂ع");
  expect(ta.value).toBe("aa🙂🙂ع");
 });
 it("does not reinterpret exact DOM deltas as mobile whole-word replacements", async () => {
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "hello");
  plainEdit(ta, "hello");
  await settle();
  expect(sent(ws)).toBe("hellohello");
 });
 it("applies no-keydown deletion and replacement using Unicode scalar suffixes", async () => {
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "a🙂");
  const edit = (value: string, inputType: string, data: string | null) => {
   ta.dispatchEvent(new InputEvent("beforeinput", {inputType, data, bubbles: true}));
   ta.value = value;
   ta.dispatchEvent(new InputEvent("input", {inputType, data, bubbles: true}));
  };
  edit("a", "deleteContentBackward", null);
  edit("é", "insertReplacementText", "é");
  await settle();
  expect(sent(ws)).toBe("a🙂\x7f\x7fé");
 });
 it("does not duplicate a hardware backspace followed by a DOM deletion", async () => {
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "a");
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Backspace", keyCode: 8, bubbles: true}));
  ta.dispatchEvent(new InputEvent("beforeinput", {inputType: "deleteContentBackward", bubbles: true}));
  ta.value = "";
  ta.dispatchEvent(new InputEvent("input", {inputType: "deleteContentBackward", bubbles: true}));
  ta.dispatchEvent(new KeyboardEvent("keyup", {key: "Backspace", keyCode: 8, bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("a\x7f");
 });
 it("preserves consecutive equal composition transactions in the same task", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const ch of ["é", "é", "ع"]) {
   ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
   ta.value += ch;
   ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: ch, bubbles: true}));
   ta.dispatchEvent(new CompositionEvent("compositionend", {data: ch, bubbles: true}));
  }
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("ééعx");
 });
 it("uses the final post-composition DOM mutation, not stale event data", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value = "e";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "e", bubbles: true}));
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "e", bubbles: true}));
  ta.dispatchEvent(new InputEvent("beforeinput", {inputType: "insertFromComposition", data: "é", bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new InputEvent("input", {inputType: "insertFromComposition", data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("é");
 });
 it("commits composition before hardware Enter and resumes no-keydown input", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", keyCode: 13, bubbles: true}));
  ta.dispatchEvent(new KeyboardEvent("keyup", {key: "Enter", keyCode: 13, bubbles: true}));
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("é\rx");
 });
 it("keeps native composition UI, accessibility tree and Tab escape", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  expect(container.querySelector(".composition-view.active")?.textContent).toBe("é");
  expect(container.querySelector(".xterm-accessibility")).not.toBeNull();
  expect(ta.getAttribute("aria-label")).toBeTruthy();
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  const tab = new KeyboardEvent("keydown", {key: "Tab", keyCode: 9, bubbles: true, cancelable: true});
  ta.dispatchEvent(tab);
  await settle();
  expect(tab.defaultPrevented).toBe(false);
  expect(sent(ws)).toBe("é");
  expect(container.querySelector(".composition-view.active")).toBeNull();
 });
 it("forwards Android plain insertText without any keydown once", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const ch of "hello") {
   ta.dispatchEvent(new InputEvent("beforeinput", {inputType: "insertText", data: ch, bubbles: true}));
   ta.value += ch;
   ta.dispatchEvent(new InputEvent("input", {inputType: "insertText", data: ch, bubbles: true, composed: true}));
   await settle();
  }
  expect(sent(ws)).toBe("hello");
  expect(ta.value).toBe("hello");
  expect(container.querySelector(".xterm-accessibility")).not.toBeNull();
 });
 it("coalesces no-keydown edits within one task without duplicated suffixes", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const ch of "hello") plainEdit(ta, ch);
  await settle();
  expect(sent(ws)).toBe("hello");
 });
 it.each([false, true])("serializes composition then no-keydown text (same task=%s)", async (sameTask) => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {data: "", bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  await settle();
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  if (!sameTask) await settle();
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("éx");
 });
 it("does not forward composing insertText prematurely", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {data: "", bubbles: true}));
  plainEdit(ta, "é", {isComposing: true});
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("");
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("é");
 });
 it("does not duplicate hardware keydown with beforeinput, then resumes no-keydown edits", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "a", keyCode: 65, bubbles: true}));
  plainEdit(ta, "a");
  ta.dispatchEvent(new KeyboardEvent("keyup", {key: "a", keyCode: 65, bubbles: true}));
  await settle();
  plainEdit(ta, "b");
  await settle();
  expect(sent(ws)).toBe("ab");
 });
 it.each([false, true])("does not duplicate mixed real-229/no-keydown edits (synthetic first=%s)", async (syntheticFirst) => {
  const {ta, ws} = await mountedTerminal();
  const nativeEdit = (ch: string) => {
   ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Unidentified", keyCode: 229, bubbles: true}));
   plainEdit(ta, ch);
  };
  if (syntheticFirst) { plainEdit(ta, "a"); nativeEdit("b"); }
  else { nativeEdit("a"); plainEdit(ta, "b"); }
  await settle();
  expect(sent(ws)).toBe("ab");
 });
 it("retains plain text when a composition starts before the diff timer", async () => {
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "a");
  ta.dispatchEvent(new CompositionEvent("compositionstart", {data: "", bubbles: true}));
  ta.value += "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  await settle();
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  await settle();
  expect(sent(ws)).toBe("aé");
 });
 it("does not mistake a held modifier for a text-input owner", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Shift", keyCode: 16, bubbles: true}));
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("x");
 });
 it("limits the no-keydown bridge to Android", async () => {
  Object.defineProperty(navigator, "userAgent", {configurable: true, value: "Mozilla/5.0 (X11; Linux x86_64) Chrome/140.0"});
  const {ta, ws} = await mountedTerminal();
  plainEdit(ta, "x");
  await settle();
  expect(sent(ws)).toBe("");
 });
 it("forwards FUTO 229 insertText hello once, including repeated l", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const ch of "hello") await softInput(ta, ch);
  expect(sent(ws)).toBe("hello");
  expect(ta.value).toBe("hello"); // retained for screen readers, not cleared by fallback
  expect(container.querySelector(".xterm-accessibility")).not.toBeNull();
 });
 it("keeps the 229 event at the textarea and away from React bubbling", async () => {
  const {ta} = await mountedTerminal();
  const target = vi.fn(), rootBubble = vi.fn();
  ta.addEventListener("keydown", target);
  container.addEventListener("keydown", rootBubble);
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "Process", keyCode: 229, bubbles: true}));
  expect(target).toHaveBeenCalledOnce();
  expect(rootBubble).not.toHaveBeenCalled();
  await settle();
 });
 it("forwards an Android backspace only once", async () => {
  const {ta, ws} = await mountedTerminal();
  await softInput(ta, "a");
  await softInput(ta, "", "deleteContentBackward", "");
  expect(sent(ws)).toBe("a\x7f");
 });
 it("retains full-word mobile replacement normalization", async () => {
  const {ta, ws} = await mountedTerminal();
  for (const ch of "hell") await softInput(ta, ch);
  await softInput(ta, "hello", "insertReplacementText", "hello");
  // xterm forwards only the changed suffix; no second full-word append.
  expect(sent(ws)).toBe("hello");
 });
 it("does not duplicate a hardware key followed by an accessibility input event", async () => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new KeyboardEvent("keydown", {key: "a", keyCode: 65, bubbles: true}));
  ta.value = "a";
  ta.dispatchEvent(new InputEvent("input", {inputType: "insertText", data: "a", bubbles: true, composed: true}));
  await settle();
  expect(sent(ws)).toBe("a");
 });
 it.each([false, true])("preserves composition and its trailing plain character (same task=%s)", async (sameTask) => {
  const {ta, ws} = await mountedTerminal();
  ta.dispatchEvent(new CompositionEvent("compositionstart", {data: "", bubbles: true}));
  ta.value = "é";
  ta.dispatchEvent(new CompositionEvent("compositionupdate", {data: "é", bubbles: true}));
  await settle();
  ta.dispatchEvent(new CompositionEvent("compositionend", {data: "é", bubbles: true}));
  if (!sameTask) await settle();
  await softInput(ta, "x");
  expect(sent(ws)).toBe("éx");
 });
});
