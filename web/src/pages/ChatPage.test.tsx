// @vitest-environment jsdom
import { createRequire } from "node:module";
import type { Terminal } from "@xterm/xterm";
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Link, MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PTY_TICKET_TIMEOUT_MS } from "@/lib/pty-reconnect";

class FakeFitAddon {
  fit() {}
}

class FakeWebglAddon {
  onContextLoss() {
    return { dispose() {} };
  }
}

class FakeTerminal {
  options: Record<string, unknown>;
  rows = 24;
  cols = 80;
  parser = {
    registerOscHandler: vi.fn(),
  };
  unicode = { activeVersion: "" };

  constructor(options: Record<string, unknown>) {
    this.options = options;
    FakeTerminal.instances.push(this);
  }

  static instances: FakeTerminal[] = [];
  static keyHandler: (event: KeyboardEvent) => boolean;
  modes = { bracketedPasteMode: true };
  dataHandler: ((data: string) => void) | null = null;

  attachCustomKeyEventHandler(handler: (event: KeyboardEvent) => boolean) {
    FakeTerminal.keyHandler = handler;
  }

  attachCustomWheelEventHandler() {
    return true;
  }

  clearSelection() {}

  dispose() {}

  focus() {}

  getSelection() {
    return "";
  }

  loadAddon() {}

  onData(handler: (data: string) => void) {
    this.dataHandler = handler;
    return { dispose() {} };
  }

  onResize() {
    return { dispose() {} };
  }

  onScroll() {
    return { dispose() {} };
  }

  get buffer() {
    // Minimal active-buffer surface for the resume follow-scroll pin
    // (isViewportPinnedToBottom reads viewportY/baseY).
    return { active: { baseY: 0, viewportY: 0 } };
  }

  scrollToBottom() {}

  open() {}

  paste(text: string) {
    this.dataHandler?.(this.modes.bracketedPasteMode ? `\x1b[200~${text}\x1b[201~` : text);
  }

  refresh() {}

  write() {}
}

const maybeReloadForLoopbackWsAuthFailure = vi.fn(() => false);
const apiMocks = vi.hoisted(() => ({
  authedFetch: vi.fn(),
  buildWsUrl: vi.fn(async () => "ws://localhost/api/pty?channel=chat-1"),
  getSessionDetail: vi.fn(async () => ({ title: null })),
  getSessionLatestDescendant: vi.fn(async () => ({ session_id: null })),
}));
const scopeMocks = vi.hoisted(() => ({ profile: "", currentProfile: "default" }));
const terminalMocks = vi.hoisted(() => ({ real: null as Terminal | null }));

vi.mock("@xterm/addon-fit", () => ({ FitAddon: FakeFitAddon }));
vi.mock("@xterm/addon-unicode11", () => ({ Unicode11Addon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: FakeWebglAddon }));
vi.mock("@xterm/xterm", () => ({
  Terminal: class {
    constructor(options: Record<string, unknown>) {
      return terminalMocks.real ?? new FakeTerminal(options);
    }
  },
}));
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
  useProfileScope: () => scopeMocks,
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
  authedFetch: apiMocks.authedFetch,
  buildWsUrl: apiMocks.buildWsUrl,
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
  terminalMocks.real = null;
  scopeMocks.profile = "";
  scopeMocks.currentProfile = "default";
  let channelCount = 0;
  FakeTerminal.instances = [];
  apiMocks.authedFetch.mockReset();
  maybeReloadForLoopbackWsAuthFailure.mockClear();
  apiMocks.buildWsUrl.mockReset();
  apiMocks.buildWsUrl.mockResolvedValue("ws://localhost/api/pty?channel=chat-1");
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
    cb(0);
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal("matchMedia", () => ({
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
    randomUUID: () => `chat-test-id-${channelCount++}`,
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
  sessionStorage.clear();
  vi.stubGlobal("localStorage", localStorageMock);
  localStorageMock.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const draftIdentity = { pty_instance: "pty-one", connection_generation: 1, session_id: "runtime-session", draft_id: "draft-one" };
const draftPrefix = "\u0000hermes-draft-v1:";
async function handshake(socket: FakeWebSocket, identity = draftIdentity, available = true) {
  await act(async () => socket.onmessage?.({ data: JSON.stringify({ type: "draft.state", identity, available }) }));
}
async function choose(files: File[]) {
  const input = container.querySelector<HTMLInputElement>("input[type=file]")!;
  Object.defineProperty(input, "files", { configurable: true, value: files });
  await act(async () => input.dispatchEvent(new Event("change", { bubbles: true })));
  expect(input.value).toBe("");
}
function uploaded(path = "/isolated/attachments/notes.txt") {
  return { ok: true, json: async () => ({ path, name: "notes.txt", bytes: 4, mime_type: "text/plain" }) };
}
function requestFrom(socket: FakeWebSocket) {
  const frame = socket.send.mock.calls.at(-1)![0] as string;
  expect(frame.startsWith(draftPrefix)).toBe(true);
  return JSON.parse(frame.slice(draftPrefix.length));
}
async function acknowledge(socket: FakeWebSocket, status = "attached") {
  const request = requestFrom(socket);
  await act(async () => socket.onmessage?.({ data: JSON.stringify({ type: "draft.result", request_id: request.request_id, identity: request.expected, status }) }));
}
function button(text: string) {
  return Array.from(container.querySelectorAll("button")).find(item => item.textContent === text);
}

describe("ChatPage", () => {
  it("keeps the idle upload control as quiet as Copy without instructional chrome", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const upload = container.querySelector<HTMLButtonElement>("button[aria-label='Upload files']")!;
    const copy = container.querySelector<HTMLButtonElement>("button[aria-label='Copy last assistant response']")!;
    expect(upload.textContent?.trim()).toBe("Upload");
    expect(copy.textContent?.trim()).toBe("Copy");
    const uploadLabel = upload.querySelector("span")!;
    const copyLabel = copy.querySelector("span")!;
    expect(uploadLabel.className).toBe(copyLabel.className);
    const uploadIcon = upload.querySelector("svg")!;
    const copyIcon = copy.querySelector("svg")!;
    for (const attribute of ["width", "height", "stroke-width", "viewBox", "aria-hidden"]) {
      expect(uploadIcon.getAttribute(attribute)).toBe(copyIcon.getAttribute(attribute));
    }
    expect(uploadIcon.classList.contains("lucide-upload")).toBe(true);
    expect(copyIcon.classList.contains("lucide-copy")).toBe(true);
    expect(container.textContent).not.toMatch(/Upload, then add|never submits/);
    expect(upload.style.color).toBe(copy.style.color);
    expect(getComputedStyle(upload).borderRadius).toBe(getComputedStyle(copy).borderRadius);
    expect(upload.className.split(" ").filter(value => !value.startsWith("absolute"))).toEqual(copy.className.split(" ").filter(value => !value.startsWith("absolute")));
  });

  it.each(["MacIntel", "Win32"])("lets the native paste event own clipboard bytes on %s without requesting permission", async platform => {
    vi.spyOn(navigator, "platform", "get").mockReturnValue(platform);
    const { default: ChatPage } = await import("./ChatPage");
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const event = new KeyboardEvent("keydown", { key: "v", metaKey: platform === "MacIntel", ctrlKey: platform !== "MacIntel", cancelable: true });
    expect(FakeTerminal.keyHandler(event)).toBe(false);
    expect(event.defaultPrevented).toBe(false);
    const textPaste = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(textPaste, "clipboardData", { value: { items: [], files: [], getData: () => "ordinary\ntext" } });
    container.querySelector(".hermes-chat-xterm-host")!.dispatchEvent(textPaste);
    expect(textPaste.defaultPrevented).toBe(false);
    await Promise.resolve();
    expect(navigator.clipboard.readText).not.toHaveBeenCalled();
    expect(apiMocks.authedFetch).not.toHaveBeenCalled();
  });

  it.each(["paste", "drop", "picker"])("automatically hands generic %s bytes to the captured draft and retires only on ACK", async ingress => {
    const { default: ChatPage } = await import("./ChatPage");
    let finishUpload!: (response: unknown) => void;
    apiMocks.authedFetch.mockImplementationOnce(() => new Promise(resolve => { finishUpload = resolve; }));
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const socket = FakeWebSocket.instances[0];
    const identity = { pty_instance: "pty-one", connection_generation: 1, session_id: "runtime-session", draft_id: "draft-one" };
    await act(async () => {
      socket.onopen?.();
      socket.onmessage?.({ data: JSON.stringify({ type: "draft.state", identity, available: true }) });
    });
    const term = FakeTerminal.instances[0];
    const pasteToTerminal = vi.spyOn(term, "paste");
    term.dataHandler?.("Please inspect this: ");
    socket.send.mockClear();
    const host = container.querySelector(".hermes-chat-xterm-host")!;
    const file = new File([new Uint8Array([0, 255, 19, 128])], "archive.bin", { type: "application/octet-stream" });
    const event = new Event(ingress === "picker" ? "change" : ingress, { bubbles: true, cancelable: true });
    const input = container.querySelector<HTMLInputElement>("input[type=file]")!;
    if (ingress === "picker") {
      expect(input.hasAttribute("accept")).toBe(false);
      Object.defineProperty(input, "files", { value: [file] });
    } else {
      Object.defineProperty(event, ingress === "paste" ? "clipboardData" : "dataTransfer", { value: { items: [{ kind: "file", type: file.type, getAsFile: () => file }], files: [file] } });
    }
    await act(async () => (ingress === "picker" ? input : host).dispatchEvent(event));
    expect(event.defaultPrevented).toBe(ingress !== "picker");
    expect(apiMocks.authedFetch).toHaveBeenCalledTimes(1);
    const [url, options] = apiMocks.authedFetch.mock.calls[0];
    expect(url).toBe("/api/chat/file-upload");
    expect(options.body).toBeInstanceOf(FormData);
    const sentFile = options.body.get("file") as File;
    expect(sentFile.name).toBe(file.name);
    const read = (blob: Blob) => new Promise(resolve => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsArrayBuffer(blob); });
    expect(new Uint8Array(await read(sentFile) as ArrayBuffer)).toEqual(new Uint8Array([0, 255, 19, 128]));
    expect(socket.send).not.toHaveBeenCalled();
    await act(async () => finishUpload({ ok: true, json: async () => ({ path: "/isolated/attachments/archive.bin", name: file.name, bytes: file.size, mime_type: file.type }) }));
    expect(socket.send).toHaveBeenCalledTimes(1);
    const frame = socket.send.mock.calls[0][0] as string;
    expect(frame.startsWith("\u0000hermes-draft-v1:")).toBe(true);
    const request = JSON.parse(frame.slice("\u0000hermes-draft-v1:".length));
    expect(request).toEqual({ type: "draft.attach", request_id: expect.any(String), expected: identity, path: "/isolated/attachments/archive.bin" });
    expect(container.textContent).toContain("Attaching…");
    expect(container.querySelector("button[aria-label^='Dismiss upload']")).toBeNull();
    expect(container.textContent).not.toMatch(/Add to draft|Add again|Profile:|\/isolated\/|not attached|check for image/);
    expect(pasteToTerminal).not.toHaveBeenCalled();
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeTerminal.instances).toHaveLength(1);
    await act(async () => socket.onmessage?.({ data: JSON.stringify({ type: "draft.result", request_id: request.request_id, identity, status: "attached" }) }));
    expect(container.textContent).not.toContain(file.name);
    expect(socket.send).toHaveBeenCalledTimes(1);
    expect(apiMocks.authedFetch).toHaveBeenCalledTimes(1);
  });

  it("keeps plain text on real xterm and retries attachment transport outside its exception-catching input emitter", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const { Terminal: RealTerminal } = createRequire(import.meta.url)("@xterm/xterm") as typeof import("@xterm/xterm");
    const term = new RealTerminal({ allowProposedApi: true });
    terminalMocks.real = term;
    vi.spyOn(term, "open").mockImplementation(host => {
      const textarea = document.createElement("textarea");
      (term as unknown as { _core: { textarea: HTMLTextAreaElement } })._core.textarea = textarea;
      host.append(textarea);
    });
    vi.spyOn(term, "loadAddon").mockImplementation(() => {});
    vi.spyOn(term, "unicode", "get").mockReturnValue({ activeVersion: "11", versions: ["11"], register() {} });
    const paste = vi.spyOn(term, "paste");
    const { default: ChatPage } = await import("./ChatPage");
    apiMocks.authedFetch.mockResolvedValue(uploaded());
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const socket = FakeWebSocket.instances[0];
    await act(async () => socket.onopen?.());
    await handshake(socket);
    await new Promise<void>(resolve => term.write("\x1b[?2004h", resolve));
    socket.send.mockClear();
    term.paste("plain\ntext");
    expect(socket.send.mock.calls).toEqual([["\x1b[200~plain\rtext\x1b[201~"]]);
    paste.mockClear();
    socket.send.mockClear();
    socket.send.mockImplementationOnce(() => { throw new Error("/private/socket/internal error"); });
    await choose([new File(["text"], "notes.txt", { type: "text/plain" })]);
    expect(container.textContent).toContain("Connection failed.");
    expect(container.textContent).not.toContain("/private/");
    expect(paste).not.toHaveBeenCalled();
    const request = requestFrom(socket);
    await act(async () => { button("Retry")!.click(); button("Retry")?.click(); });
    expect(requestFrom(socket)).toEqual(request);
    expect(socket.send).toHaveBeenCalledTimes(2);
    expect(apiMocks.authedFetch).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("Attaching…");
    await acknowledge(socket);
    expect(container.textContent).not.toContain("notes.txt");
    expect(paste).not.toHaveBeenCalled();
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it.each(["profile", "resume", "reconnect", "new session", "hidden", "closed socket", "native draft", "native session"])("does not retarget a pending upload after %s", async change => {
    const { default: ChatPage } = await import("./ChatPage");
    let finishUpload!: (response: unknown) => void;
    apiMocks.authedFetch.mockImplementationOnce(() => new Promise(resolve => { finishUpload = resolve; }));
    scopeMocks.profile = "original";
    const ui = (isActive = true) => <MemoryRouter><ChatPage isActive={isActive} /><Link to="/chat?resume=other">Resume another</Link></MemoryRouter>;
    await render(ui());
    const original = FakeWebSocket.instances[0];
    await act(async () => original.onopen?.());
    await handshake(original);
    await choose([new File(["text"], "notes.txt", { type: "text/plain" })]);
    expect(apiMocks.authedFetch.mock.calls[0][0]).toBe("/api/chat/file-upload?profile=original");
    if (change === "profile") {
      scopeMocks.profile = "other";
      await act(async () => root.render(ui()));
    } else if (change === "resume") {
      await act(async () => container.querySelector<HTMLAnchorElement>("a")!.click());
    } else if (change === "reconnect" || change === "new session") {
      original.readyState = 3;
      await act(async () => original.onclose?.({ code: change === "reconnect" ? 1006 : 4410, reason: "", wasClean: false }));
      const label = change === "reconnect" ? "Reconnect chat" : "Start a new chat session";
      await act(async () => container.querySelector<HTMLButtonElement>(`button[aria-label='${label}']`)!.click());
    } else if (change === "hidden") {
      await act(async () => root.render(ui(false)));
      expect(FakeWebSocket.instances).toHaveLength(1);
    } else if (change === "closed socket") original.readyState = 3;
    else await handshake(original, { ...draftIdentity, ...(change === "native draft" ? { draft_id: "new-draft" } : { session_id: "new-session" }) });
    if (FakeWebSocket.instances.length > 1) {
      const replacement = FakeWebSocket.instances.at(-1)!;
      await act(async () => replacement.onopen?.());
      await handshake(replacement);
    }
    for (const socket of FakeWebSocket.instances) socket.send.mockClear();
    await act(async () => finishUpload(uploaded()));
    for (const socket of FakeWebSocket.instances) expect(socket.send).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Draft changed.");
    expect(container.textContent).not.toMatch(/Add to draft|Add again|\/isolated\/|Profile:/);
    expect(button("Retry")).toBeUndefined();
    await act(async () => container.querySelector<HTMLButtonElement>("button[aria-label='Dismiss upload notes.txt']")!.click());
    expect(container.textContent).not.toContain("notes.txt");
  });

  it("uses a compact upload failure with a real scoped retry, then dismisses pending work without resurrection", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    scopeMocks.currentProfile = "studio"; // Empty profile means this dashboard, not default.
    apiMocks.authedFetch.mockRejectedValueOnce(new Error("offline internal detail"));
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const socket = FakeWebSocket.instances[0];
    await act(async () => socket.onopen?.());
    await handshake(socket);
    const input = container.querySelector<HTMLInputElement>("input[type=file]")!;
    const click = vi.spyOn(input, "click");
    container.querySelector<HTMLButtonElement>("button[aria-label='Upload files']")!.click();
    expect(click).toHaveBeenCalledOnce();
    await choose([]);
    expect(apiMocks.authedFetch).not.toHaveBeenCalled();
    const file = new File(["text"], "notes.txt", { type: "text/plain" });
    await choose([file]);
    expect(container.textContent).toContain("Upload failed.");
    expect(container.textContent).not.toMatch(/offline internal detail|Dashboard profile|Profile:/);
    let finishUpload!: (response: unknown) => void;
    apiMocks.authedFetch.mockImplementationOnce(() => new Promise(resolve => { finishUpload = resolve; }));
    await act(async () => { button("Retry")!.click(); button("Retry")?.click(); });
    expect(apiMocks.authedFetch.mock.calls.map(([url]) => url)).toEqual(["/api/chat/file-upload", "/api/chat/file-upload"]);
    socket.send.mockClear();
    await act(async () => container.querySelector<HTMLButtonElement>("button[aria-label='Dismiss upload notes.txt']")!.click());
    await act(async () => finishUpload(uploaded()));
    expect(container.textContent).not.toContain("notes.txt");
    expect(socket.send).not.toHaveBeenCalled();
  });

  it("disables unknown TUI upload, contains file paste/drop, and consumes malformed controls before terminal output", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    await render(<MemoryRouter><ChatPage isActive /></MemoryRouter>);
    const socket = FakeWebSocket.instances[0];
    await act(async () => socket.onopen?.());
    const write = vi.spyOn(FakeTerminal.instances[0], "write");
    const upload = container.querySelector<HTMLButtonElement>("button[aria-label='Upload files']")!;
    expect(upload.disabled).toBe(true);
    for (const frame of [draftPrefix + "broken", '{"type":"draft.state",broken', JSON.stringify({ type: "draft.state", identity: {}, available: true }), JSON.stringify({ type: "draft.unknown" })]) {
      await act(async () => socket.onmessage?.({ data: frame }));
    }
    expect(write).not.toHaveBeenCalled();
    expect(upload.disabled).toBe(true);
    const host = container.querySelector(".hermes-chat-xterm-host")!;
    const file = new File(["pdf"], "notes.pdf", { type: "application/pdf" });
    for (const ingress of ["paste", "drop"]) {
      for (const data of [
        { files: [file], items: [] },
        { files: [], items: [{ kind: "file", getAsFile: () => null }] },
        { files: [], items: [], types: ["Files"] },
      ]) {
        const event = new Event(ingress, { bubbles: true, cancelable: true });
        Object.defineProperty(event, ingress === "paste" ? "clipboardData" : "dataTransfer", { value: { ...data, getData: () => "file:///browser/private/path" } });
        host.dispatchEvent(event);
        expect(event.defaultPrevented).toBe(true);
      }
    }
    expect(apiMocks.authedFetch).not.toHaveBeenCalled();
    await handshake(socket);
    expect(upload.disabled).toBe(false);
    await handshake(socket, draftIdentity, false);
    expect(upload.disabled).toBe(true);
    expect(write).not.toHaveBeenCalled();
    await act(async () => socket.onmessage?.({ data: "ordinary terminal text" }));
    expect(write).toHaveBeenCalledWith("ordinary terminal text", undefined);
  });

  it("treats loopback 4401 closes as stale-token reload candidates", async () => {
    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    FakeWebSocket.instances[0].onclose?.({
      code: 4401,
      reason: "auth: token_mismatch",
      wasClean: true,
    });

    expect(maybeReloadForLoopbackWsAuthFailure).toHaveBeenCalledWith(4401);
  });

  it("attaches visualViewport keyboard-inset listeners only while the chat tab is active", async () => {
    // NS-434 follow-up: ChatPage stays mounted (hidden) on every dashboard
    // route. The keyboard-inset/scroll-pin listeners must only be live while
    // /chat is the active tab, or the scroll pin fires when a soft keyboard
    // opens on Settings etc.
    const addEventListener = vi.fn();
    const removeEventListener = vi.fn();
    Object.defineProperty(window, "visualViewport", {
      configurable: true,
      value: { addEventListener, removeEventListener, width: 1280 },
    });

    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive={false} />
      </MemoryRouter>,
    );
    expect(addEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive />
        </MemoryRouter>,
      ),
    );
    expect(addEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);
    expect(removeEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive={false} />
        </MemoryRouter>,
      ),
    );
    expect(removeEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);
  });
});

describe("ChatPage side panel collapse", () => {
  async function renderChat() {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
  }

  it("collapses the desktop side panel and persists the choice", async () => {
    localStorage.clear();
    await renderChat();
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    const collapseButton = container.querySelector(
      '[aria-label="Collapse chat side panel"]',
    );
    expect(collapseButton).not.toBeNull();

    await act(async () => {
      collapseButton!.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(localStorage.getItem("hermes-chat-panel-collapsed")).toBe("1");
    expect(
      container.querySelector('[aria-label="Collapse chat side panel"]'),
    ).toBeNull();
    expect(
      container.querySelector('[aria-label="Show chat side panel"]'),
    ).not.toBeNull();

    // Reopening restores the panel and clears the persisted flag.
    await act(async () => {
      container
        .querySelector('[aria-label="Show chat side panel"]')!
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(localStorage.getItem("hermes-chat-panel-collapsed")).toBe("0");
    expect(
      container.querySelector('[aria-label="Collapse chat side panel"]'),
    ).not.toBeNull();
  });
});

// The gated-mode ticket request runs before any socket exists, so a rejection
// or a hang emits no `close` event and never arms PTY_CONNECTING_TIMEOUT_MS
// (that timer is set after `new WebSocket`). Without its own deadline the tab
// strands on "connecting" with no retry. Mirrors the ChatSidebar events-feed
// coverage in src/components/ChatSidebar.test.tsx.
describe("ChatPage PTY ticket connect deadline", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function renderChat() {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
  }

  /** Advance timers and flush the async connect that fires on the tick. */
  async function advance(ms: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  it("retries when the ticket request rejects", async () => {
    apiMocks.buildWsUrl.mockRejectedValueOnce(
      new Error("ticket endpoint unavailable"),
    );

    await renderChat();
    await advance(0);
    expect(FakeWebSocket.instances).toHaveLength(0);

    // First backoff step is 250ms; the retry must mint a fresh ticket.
    await advance(250);
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(2);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("times out a stalled ticket request and retries", async () => {
    let resolveStalledRequest!: (url: string) => void;
    apiMocks.buildWsUrl.mockImplementationOnce(
      () =>
        new Promise<string>((resolve) => {
          resolveStalledRequest = resolve;
        }),
    );

    await renderChat();
    await advance(0);
    expect(FakeWebSocket.instances).toHaveLength(0);

    await advance(PTY_TICKET_TIMEOUT_MS);
    expect(FakeWebSocket.instances).toHaveLength(0);

    // A late ticket from the timed-out attempt must not open a socket behind
    // the replacement the deadline scheduled.
    await act(async () => {
      resolveStalledRequest("ws://localhost/api/pty?channel=stale");
      await Promise.resolve();
    });
    expect(FakeWebSocket.instances).toHaveLength(0);

    await advance(250);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).not.toContain("channel=stale");
  });

  it("leaves a settled ticket's socket to the CONNECTING timer", async () => {
    await renderChat();
    await advance(0);
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    // NS-591 regression: once the socket exists the ticket deadline is
    // disarmed, so PTY_CONNECTING_TIMEOUT_MS stays the only thing that may
    // force-close a wedged handshake — the two must not both fire.
    await advance(PTY_TICKET_TIMEOUT_MS);
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(1);
  });
});
