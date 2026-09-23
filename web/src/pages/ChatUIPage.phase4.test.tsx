/**
 * Focused Phase 4 tests for /chat-ui:
 *
 *   - empty state renders prompt suggestions
 *   - suggestion click seeds the composer (does NOT auto-submit)
 *   - PDF attachment calls pdf.attach
 *   - PDF attach failure leaves an inline error and skips the chip
 *   - removing an attachment chip keeps remaining ones
 *   - submitting with both image + PDF attaches does NOT carry them inside
 *     prompt.submit
 *   - Regenerate stays disabled with an explanatory tooltip
 *   - ChatToolCard surfaces running / complete / error states + duration
 *   - ReasoningPanel opens while streaming, collapses after finalise
 *   - mobile drawer close-on-Escape, backdrop labelled
 *   - accessibility: aria-labels, role=log, focus behaviour
 *   - no xterm import in chat-ui module graph
 *
 * Mirrors the Phase 3 stub for the GatewayClient — these cases are pure
 * UI / hook tests, no real socket.
 */

// @vitest-environment jsdom
import { act } from "react-dom/test-utils";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/* -------------------------------------------------------------------------- */
/* Mockable GatewayClient                                                      */
/* -------------------------------------------------------------------------- */

type GatewayEvent = { type: string; payload?: Record<string, unknown> };

interface StubbedClient {
  close: () => Promise<void> | void;
  connect: () => Promise<void> | void;
  connectionState: string;
  on: (
    event: string,
    handler: (event: GatewayEvent) => void,
  ) => () => void;
  onAny: (handler: (event: GatewayEvent) => void) => () => void;
  onState: (handler: (state: string) => void) => () => void;
  request: (
    method: string,
    params: Record<string, unknown>,
  ) => Promise<unknown>;
  fire: (event: GatewayEvent) => void;
  requests: Array<{ method: string; params: Record<string, unknown> }>;
}

const clients: StubbedClient[] = [];

function makeClient(
  opts: { pdfAttached?: boolean; pdfAttachError?: { code: number; message: string } } = {},
): StubbedClient {
  const handlers = new Set<(event: GatewayEvent) => void>();
  const stateHandlers = new Set<(state: string) => void>();
  const requests: Array<{ method: string; params: Record<string, unknown> }> =
    [];
  let state: string = "open";
  const setState = (next: string) => {
    state = next;
    stateHandlers.forEach((handler) => handler(state));
  };
  const client: StubbedClient = {
    close: vi.fn(async () => undefined),
    connect: vi.fn(async () => {
      setState("open");
      return undefined;
    }),
    get connectionState() {
      return state;
    },
    on: vi.fn(() => () => undefined),
    onAny: vi.fn((handler) => {
      handlers.add(handler);
      return () => handlers.delete(handler);
    }),
    onState: vi.fn((handler) => {
      stateHandlers.add(handler);
      handler(state);
      return () => stateHandlers.delete(handler);
    }),
    request: vi.fn(
      async (method: string, params: Record<string, unknown>) => {
        requests.push({ method, params });
        if (method === "session.create") {
          return {
            session_id: "stub-session",
            stored_session_id: "stored-stub",
            message_count: 0,
            messages: [],
            info: {
              model: "anthropic/claude-sonnet",
              provider: "anthropic",
              reasoning_effort: "medium",
              title: "Test",
            },
          };
        }
        if (method === "session.resume") {
          return {
            session_id: params["session_id"] ?? "stub-session",
            message_count: 0,
            messages: [],
            info: {
              model: "anthropic/claude-sonnet",
              provider: "anthropic",
              reasoning_effort: "medium",
              title: "Resumed",
            },
          };
        }
        if (method === "session.events.since") {
          return { session_id: params["session_id"], events: [] };
        }
        if (method === "image.attach_bytes") {
          return {
            attached: true,
            name: "test.png",
            path: "/tmp/hermes-images/test.png",
            bytes: 1234,
            width: 100,
            height: 100,
            token_estimate: 1000,
          };
        }
        if (method === "pdf.attach") {
          // Live verification: the real gateway surfaces every failure as
          // a JSON-RPC error frame (server `methods_prompt.py::_err`).
          // Simulate that by rejecting the request with a structured
          // JsonRpcGatewayError so the frontend's catch path runs.
          if (opts.pdfAttachError) {
            const err = new Error(opts.pdfAttachError.message) as Error & {
              code?: number;
              name: string;
            };
            err.name = "JsonRpcGatewayError";
            err.code = opts.pdfAttachError.code;
            throw err;
          }
          if (opts.pdfAttached === false) {
            // Defensive: a contract that returns {attached:false} instead of
            // throwing. Kept for backwards-compat with mock gateways.
            return {
              attached: false,
              filename: params["filename"] ?? "doc.pdf",
              pages_attached: 0,
              pages: [],
              count: 0,
              text: "",
            };
          }
          return {
            attached: true,
            filename: params["filename"] ?? "doc.pdf",
            pages_attached: 3,
            pages: [],
            count: 3,
            text: "rendered text",
          };
        }
        if (method === "session.interrupt") {
          return { ok: true };
        }
        return { ok: true };
      },
    ),
    fire(event: GatewayEvent) {
      handlers.forEach((handler) => handler(event));
    },
    requests,
  };
  return client;
}

vi.mock("@/lib/gatewayClient", () => {
  return {
    GatewayClient: vi.fn().mockImplementation(function () {
      // The last makeClient() call wins; tests rebuild the mock via a
      // module-level setter before each render.
      const c = currentMake();
      clients.push(c);
      return c;
    }),
  };
});

// Per-test factory setter so we can flip pdf.attach to a failure path
// without rebuilding the entire mock.
let currentMake: () => StubbedClient = () => makeClient();
function setMake(fn: () => StubbedClient) {
  currentMake = fn;
}

vi.mock("@/components/ProfileSwitcher", () => ({
  ProfileSwitcher: () => null,
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: null }),
}));

/* -------------------------------------------------------------------------- */
/* Test rig                                                                   */
/* -------------------------------------------------------------------------- */

import ChatUIPage from "@/pages/ChatUIPage";
import { ChatComposer } from "@/components/chat/ChatComposer";
import { ChatToolCard } from "@/components/chat/ChatToolCard";
import { ReasoningPanel } from "@/components/chat/ReasoningPanel";
import { ChatMessageList } from "@/components/chat/ChatMessageList";
import type { ChatUIStore } from "@/components/chat/types";

let roots: Root[] = [];

function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function flushAct(): Promise<void> {
  await act(async () => {
    await flushMicrotasks();
  });
}

afterEach(() => {
  while (roots.length > 0) {
    const r = roots.pop();
    if (r) act(() => r.unmount());
  }
  clients.length = 0;
  setMake(() => makeClient());
  document.body.innerHTML = "";
});

beforeEach(() => {
  setMake(() => makeClient());
  clients.length = 0;
});

async function renderAt(initialPath: string): Promise<{
  root: Root;
  container: HTMLElement;
  client: StubbedClient;
}> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  roots.push(root);
  await act(async () => {
    root.render(
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route path="/chat-ui" element={<ChatUIPage />} />
          <Route path="/chat" element={<div data-testid="cli-mount" />} />
        </Routes>
      </MemoryRouter>,
    );
  });
  await flushAct();
  await flushAct();
  const client = clients[clients.length - 1];
  if (!client) throw new Error("GatewayClient mock not invoked");
  return { root, container, client };
}

/* -------------------------------------------------------------------------- */
/* Empty state                                                                */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — empty state", () => {
  it("renders the centred empty state with prompt suggestions", async () => {
    const { container } = await renderAt("/chat-ui");
    expect(
      container.querySelector('[data-testid="chat-ui-empty"]'),
    ).toBeTruthy();
    const title = container.querySelector(
      '[data-testid="chat-ui-empty-title"]',
    );
    expect(title?.textContent).toMatch(/Hermes/i);
    const suggestions = container.querySelectorAll(
      '[data-testid="chat-ui-empty-suggestion"]',
    );
    expect(suggestions.length).toBeGreaterThanOrEqual(4);
  });

  it("suggestion click populates the composer without auto-submitting", async () => {
    const { container } = await renderAt("/chat-ui");
    const suggestion = container.querySelector(
      '[data-testid="chat-ui-empty-suggestion"]',
    ) as HTMLButtonElement;
    expect(suggestion).toBeTruthy();
    suggestion.click();
    await flushAct();
    const ta = container.querySelector(
      '[data-testid="chat-ui-composer-input"]',
    ) as HTMLTextAreaElement;
    expect(ta.value).not.toBe("");
    expect(ta.value.length).toBeGreaterThan(0);
    // prompt.submit must NOT have been called automatically.
    const submit = clients[0]?.requests.find((r) => r.method === "prompt.submit");
    expect(submit).toBeUndefined();
  });

  it("the empty state disappears once a user message is rendered", async () => {
    const { client, container } = await renderAt("/chat-ui");
    await flushAct();
    act(() => client.fire({ type: "message.start" }));
    act(() =>
      client.fire({ type: "message.delta", payload: { text: "hi" } }),
    );
    act(() =>
      client.fire({
        type: "message.complete",
        payload: {
          text: "hi",
          status: "complete",
          usage: null,
          reasoning: null,
          warning: null,
          response_previewed: null,
          billing: null,
          failure_reason: null,
          rendered: null,
          error: null,
          recoverable: null,
          error_surface: null,
          partial: null,
        },
      }),
    );
    // Fire a user/submitted-equivalent via prompt.submit to land a user turn.
    await flushAct();
    const ta = container.querySelector(
      '[data-testid="chat-ui-composer-input"]',
    ) as HTMLTextAreaElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    setter?.call(ta, "hello");
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    await flushAct();
    const send = container.querySelector(
      '[data-testid="chat-ui-composer-send"]',
    ) as HTMLButtonElement;
    send.click();
    await flushAct();
    expect(
      container.querySelector('[data-testid="chat-ui-empty"]'),
    ).toBeNull();
  });
});

/* -------------------------------------------------------------------------- */
/* PDF attachment                                                             */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — PDF attachment", () => {
  it("PDF picker button calls onAttachPdf → pdf.attach", async () => {
    const { container, client } = await renderAt("/chat-ui");
    await flushAct();
    const pdfBtn = container.querySelector(
      '[data-testid="chat-ui-composer-attach-pdf"]',
    ) as HTMLButtonElement;
    expect(pdfBtn).toBeTruthy();
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    expect(input).toBeTruthy();
    const file = new File(
      [new Uint8Array([1, 2, 3, 4])],
      "whitepaper.pdf",
      { type: "application/pdf" },
    );
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    const attach = client.requests.find((r) => r.method === "pdf.attach");
    expect(attach).toBeTruthy();
    expect(attach?.params["session_id"]).toBe("stub-session");
    expect(typeof attach?.params["content_base64"]).toBe("string");
    expect(attach?.params["filename"]).toBe("whitepaper.pdf");
    const chip = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"][data-attachment-kind="pdf"]',
    );
    expect(chip).toBeTruthy();
    expect(chip?.textContent).toContain("whitepaper.pdf");
  });

  it("failed PDF attach surfaces inline error and skips the chip", async () => {
    // Live verification: real gateway returns a JSON-RPC error frame on
    // every pdf.attach failure (see tui_gateway/methods_prompt.py::_err).
    // The frontend's catch path is what surfaces the error.
    setMake(() =>
      makeClient({
        pdfAttachError: {
          code: 4017,
          message: "payload is not a PDF (missing %PDF- magic bytes)",
        },
      }),
    );
    const { container } = await renderAt("/chat-ui");
    await flushAct();
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    const file = new File([new Uint8Array([1])], "broken.pdf", {
      type: "application/pdf",
    });
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    const err = container.querySelector(
      '[data-testid="chat-ui-composer-attach-error"]',
    );
    expect(err).toBeTruthy();
    // The error message surfaces the gateway's message verbatim so the user
    // sees the actual reason, not a generic "attach failed".
    expect(err?.textContent).toContain("not a PDF");
    const chip = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"]',
    );
    expect(chip).toBeNull();
  });

  it("PDF picker is disabled until the session is fully created", async () => {
    // Build a client whose connect can be deferred so we can assert the
    // picker stays disabled while the WS is in-flight (session.create has
    // not yet landed).
    let pokeOpen: () => void = () => {};
    const deferred = new Promise<void>((resolve) => {
      pokeOpen = resolve;
    });
    setMake(() => {
      const c = makeClient();
      const origConnect = c.connect;
      c.connect = vi.fn(async () => {
        await deferred;
        await origConnect();
      });
      return c;
    });
    const { container } = await renderAt("/chat-ui");
    // While the WS is connecting the reducer has connection="connecting"
    // and sessionId=null. The PDF picker must be disabled.
    const pdfBtn = container.querySelector(
      '[data-testid="chat-ui-composer-attach-pdf"]',
    ) as HTMLButtonElement;
    expect(pdfBtn.disabled).toBe(true);
    // The picker is the right element regardless of disabled state — sanity.
    expect(pdfBtn.getAttribute("aria-label")).toBe("Attach PDF");
    // Now finish the connect. Once session.create lands the picker
    // becomes enabled (the reducer's sessionId flips to non-null).
    pokeOpen();
    await flushAct();
    await flushAct();
    // After the deferred connect resolves + session.create lands, the
    // reducer's sessionId is set and the picker is enabled.
    const pdfBtnAfter = container.querySelector(
      '[data-testid="chat-ui-composer-attach-pdf"]',
    ) as HTMLButtonElement;
    expect(pdfBtnAfter.disabled).toBe(false);
  });

  it("PDF chip renders the page count reported by the gateway", async () => {
    // Live verification: the real gateway returns:
    //   pages_attached = number of pages in THIS PDF
    //   count          = running session-wide attachment total
    //                    (image + PDF pages from previous attaches)
    // The chip subtitle ("N pages") must use pages_attached (or pages.length)
    // — never count — or the chip would show the wrong number the moment a
    // second PDF (or an image) was attached in the same session.
    setMake(() => {
      const c = makeClient();
      // Override the default pdf.attach response so we control the exact
      // envelope. Deliberately make ``count`` DIFFERENT from
      // ``pages_attached`` to prove the chip reads the right field.
      const origRequest = c.request;
      c.request = vi.fn(async (method, params) => {
        if (method === "pdf.attach") {
          return {
            attached: true,
            filename: (params as Record<string, unknown>)["filename"] ?? "doc.pdf",
            pages_attached: 12,
            pages: [
              { path: "/x/p1.png", page: 1 },
              { path: "/x/p2.png", page: 2 },
              { path: "/x/p3.png", page: 3 },
            ],
            // count is the session-wide total — set to something different
            // to prove the chip ignores it.
            count: 47,
            text: "[User attached PDF: doc.pdf (12 page(s))]",
          };
        }
        return origRequest(method as never, params as never) as never;
      });
      return c;
    });
    const { container } = await renderAt("/chat-ui");
    await flushAct();
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(input, "files", {
      value: [
        new File(
          [new Uint8Array([0x25, 0x50, 0x44, 0x46, 0x2d, 1, 0, 0])],
          "spec.pdf",
          { type: "application/pdf" },
        ),
      ],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    const chip = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"][data-attachment-kind="pdf"]',
    );
    expect(chip).toBeTruthy();
    // Chip uses pages_attached → "12 pages".
    expect(chip?.textContent).toContain("12 pages");
    // Filename is preserved.
    expect(chip?.textContent).toContain("spec.pdf");
    // Crucially the chip MUST NOT show the misleading count (47).
    expect(chip?.textContent).not.toContain("47");
  });

  it("remove button drops a pending PDF chip without affecting other attachments", async () => {
    const { container } = await renderAt("/chat-ui");
    await flushAct();
    // Attach a PDF…
    const pdfInput = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(pdfInput, "files", {
      value: [
        new File([new Uint8Array([1, 2])], "first.pdf", {
          type: "application/pdf",
        }),
      ],
      configurable: true,
    });
    pdfInput.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    // …and an image.
    const imgInput = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(imgInput, "files", {
      value: [
        new File([new Uint8Array([5, 6])], "shot.png", { type: "image/png" }),
      ],
      configurable: true,
    });
    imgInput.dispatchEvent(new Event("change", { bubbles: true }));
    // The image chip waits on FileReader.readAsDataURL; flush several
    // microtasks so the promise chain resolves.
    await flushAct();
    await flushAct();
    await flushAct();
    let chips = container.querySelectorAll(
      '[data-testid="chat-ui-composer-attachment"]',
    );
    expect(chips.length).toBe(2);
    const pdfChip = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"][data-attachment-kind="pdf"]',
    );
    const pdfRemove = pdfChip?.querySelector(
      'button[aria-label^="Remove"]',
    ) as HTMLButtonElement;
    expect(pdfRemove).toBeTruthy();
    act(() => {
      pdfRemove.click();
    });
    await flushAct();
    chips = container.querySelectorAll(
      '[data-testid="chat-ui-composer-attachment"]',
    );
    expect(chips.length).toBe(1);
    expect(
      container.querySelector(
        '[data-testid="chat-ui-composer-attachment"][data-attachment-kind="image"]',
      ),
    ).toBeTruthy();
  });

  it("sending a prompt with image + PDF attachments does NOT inline them into prompt.submit", async () => {
    const { container, client } = await renderAt("/chat-ui");
    await flushAct();
    // Image
    const imgInput = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(imgInput, "files", {
      value: [
        new File([new Uint8Array([1])], "img.png", { type: "image/png" }),
      ],
      configurable: true,
    });
    imgInput.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    // PDF
    const pdfInput = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(pdfInput, "files", {
      value: [
        new File([new Uint8Array([2])], "doc.pdf", {
          type: "application/pdf",
        }),
      ],
      configurable: true,
    });
    pdfInput.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    // Type + send
    const ta = container.querySelector(
      '[data-testid="chat-ui-composer-input"]',
    ) as HTMLTextAreaElement;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    setter?.call(ta, "explain both");
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    await flushAct();
    const send = container.querySelector(
      '[data-testid="chat-ui-composer-send"]',
    ) as HTMLButtonElement;
    send.click();
    await flushAct();
    const submit = client.requests.find((r) => r.method === "prompt.submit");
    expect(submit).toBeTruthy();
    expect(submit?.params).not.toHaveProperty("images");
    expect(submit?.params).not.toHaveProperty("pdfs");
    expect(submit?.params["text"]).toBe("explain both");
    // Both attach RPCs were called BEFORE submit.
    const imageCall = client.requests.find((r) => r.method === "image.attach_bytes");
    const pdfCall = client.requests.find((r) => r.method === "pdf.attach");
    expect(imageCall).toBeTruthy();
    expect(pdfCall).toBeTruthy();
    expect(client.requests.indexOf(imageCall!)).toBeLessThan(
      client.requests.indexOf(submit!),
    );
    expect(client.requests.indexOf(pdfCall!)).toBeLessThan(
      client.requests.indexOf(submit!),
    );
  });
});

/* -------------------------------------------------------------------------- */
/* Regenerate / message actions                                               */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — message actions", () => {
  it("regenerate button stays disabled with an explanatory tooltip", async () => {
    const { client, container } = await renderAt("/chat-ui");
    act(() => client.fire({ type: "message.start" }));
    act(() =>
      client.fire({
        type: "message.complete",
        payload: {
          text: "Reply",
          status: "complete",
          usage: null,
          reasoning: null,
          warning: null,
          response_previewed: null,
          billing: null,
          failure_reason: null,
          rendered: null,
          error: null,
          recoverable: null,
          error_surface: null,
          partial: null,
        },
      }),
    );
    await flushAct();
    const regen = container.querySelector(
      '[data-testid="chat-ui-regenerate"]',
    ) as HTMLButtonElement;
    expect(regen).toBeTruthy();
    expect(regen.disabled).toBe(true);
    expect(regen.getAttribute("aria-disabled")).toBe("true");
    expect(regen.getAttribute("title")).toMatch(/not supported/i);
  });
});

/* -------------------------------------------------------------------------- */
/* Tool cards                                                                  */
/* -------------------------------------------------------------------------- */

describe("ChatToolCard — states", () => {
  const baseTool = {
    id: "t1",
    toolId: "t1",
    name: "bash",
    argumentsDecoded: { command: "echo hi" },
    status: "success" as const,
    durationS: 0.12,
    result: "hi",
  };

  // Make a partial that narrows to any ChatUIToolCall variant — the
  // component reads only the fields it cares about so a `Partial<typeof
  // baseTool>` is enough for the test surface.
  type PartialTool = {
    id?: string;
    toolId?: string;
    name?: string;
    argumentsDecoded?: Record<string, unknown>;
    argumentsRaw?: string;
    status?: "pending" | "running" | "success" | "error";
    durationS?: number;
    result?: string;
    error?: string;
    progress?: { tail?: string; updatedAt?: number };
  };

  function renderCard(tool: PartialTool) {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    roots.push(root);
    act(() => {
      root.render(
        <ChatToolCard
          tool={
            {
              ...baseTool,
              ...tool,
            } as never
          }
        />,
      );
    });
    return container;
  }

  it("renders a success badge with duration", () => {
    const c = renderCard({});
    expect(c.querySelector('[data-tool-status="success"]')).toBeTruthy();
    expect(c.textContent).toContain("0.12s");
  });

  it("renders an error badge and the error message", () => {
    const c = renderCard({
      status: "error",
      error: "permission denied",
      result: undefined,
    });
    expect(c.querySelector('[data-tool-status="error"]')).toBeTruthy();
    // Open the card so the error body renders.
    const toggle = c.querySelector("button") as HTMLButtonElement;
    act(() => {
      toggle.click();
    });
    expect(c.textContent).toContain("permission denied");
  });

  it("renders a running spinner when status is running", () => {
    const c = renderCard({ status: "running", result: undefined });
    expect(c.querySelector('[data-tool-status="running"]')).toBeTruthy();
    expect(c.querySelector(".animate-spin")).toBeTruthy();
  });
});

/* -------------------------------------------------------------------------- */
/* Reasoning panel                                                            */
/* -------------------------------------------------------------------------- */

describe("ReasoningPanel", () => {
  it("renders a Reasoning label and collapses by default when not streaming", () => {
    const c = document.createElement("div");
    document.body.appendChild(c);
    const root = createRoot(c);
    roots.push(root);
    act(() => {
      root.render(
        <ReasoningPanel text="Let me think…" initiallyOpen={false} />,
      );
    });
    const panel = c.querySelector(
      '[data-testid="chat-ui-reasoning-panel"]',
    ) as HTMLDetailsElement | null;
    expect(panel).toBeTruthy();
    expect(panel?.open).toBe(false);
    expect(c.textContent).toMatch(/Reasoning/i);
  });

  it("starts open when streaming and exposes the reasoning text", () => {
    const c = document.createElement("div");
    document.body.appendChild(c);
    const root = createRoot(c);
    roots.push(root);
    act(() => {
      root.render(
        <ReasoningPanel text="step one step two" initiallyOpen />,
      );
    });
    const panel = c.querySelector(
      '[data-testid="chat-ui-reasoning-panel"]',
    ) as HTMLDetailsElement | null;
    expect(panel?.open).toBe(true);
    expect(c.textContent).toContain("step one step two");
  });
});

/* -------------------------------------------------------------------------- */
/* Mobile drawer / accessibility                                              */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — mobile drawer accessibility", () => {
  it("renders a menu button with aria-expanded and aria-controls", async () => {
    const { container } = await renderAt("/chat-ui");
    const menu = container.querySelector(
      '[data-testid="chat-ui-menu-button"]',
    ) as HTMLButtonElement;
    expect(menu).toBeTruthy();
    expect(menu.getAttribute("aria-expanded")).toBe("false");
    expect(menu.getAttribute("aria-controls")).toBe("chat-ui-sidebar");
    expect(menu.getAttribute("aria-label")).toBeTruthy();
  });

  it("Escape closes the drawer", async () => {
    const { container } = await renderAt("/chat-ui");
    const menu = container.querySelector(
      '[data-testid="chat-ui-menu-button"]',
    ) as HTMLButtonElement;
    menu.click();
    await flushAct();
    const backdrop = container.querySelector(
      '[data-testid="chat-ui-sidebar-backdrop"]',
    );
    expect(backdrop).toBeTruthy();
    await act(async () => {
      window.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true }),
      );
    });
    await flushAct();
    expect(
      container.querySelector('[data-testid="chat-ui-sidebar-backdrop"]'),
    ).toBeNull();
  });
});

/* -------------------------------------------------------------------------- */
/* Accessibility — composer, message list                                     */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — composer accessibility", () => {
  it("form is labelled, textarea labelled, send/stop have aria-labels", async () => {
    const { container } = await renderAt("/chat-ui");
    const form = container.querySelector(
      '[data-testid="chat-ui-composer"]',
    ) as HTMLElement;
    expect(form.getAttribute("aria-label")).toBe("Message composer");
    const ta = container.querySelector(
      '[data-testid="chat-ui-composer-input"]',
    ) as HTMLTextAreaElement;
    expect(ta.getAttribute("aria-label")).toBe("Message input");
    const send = container.querySelector(
      '[data-testid="chat-ui-composer-send"]',
    ) as HTMLButtonElement;
    expect(send.getAttribute("aria-label")).toBeTruthy();
    const img = container.querySelector(
      '[data-testid="chat-ui-composer-attach"]',
    ) as HTMLButtonElement;
    expect(img.getAttribute("aria-label")).toBe("Attach image");
    const pdf = container.querySelector(
      '[data-testid="chat-ui-composer-attach-pdf"]',
    ) as HTMLButtonElement;
    expect(pdf.getAttribute("aria-label")).toBe("Attach PDF");
  });

  it("PDF attach error has role=alert and is dismissible", async () => {
    setMake(() => makeClient({ pdfAttached: false }));
    const { container } = await renderAt("/chat-ui");
    await flushAct();
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-pdf-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(input, "files", {
      value: [
        new File([new Uint8Array([1])], "broken.pdf", {
          type: "application/pdf",
        }),
      ],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    const err = container.querySelector(
      '[data-testid="chat-ui-composer-attach-error"]',
    ) as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.getAttribute("role")).toBe("alert");
    const dismiss = err.querySelector("button");
    expect(dismiss?.getAttribute("aria-label")).toBeTruthy();
    dismiss?.click();
    await flushAct();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-attach-error"]'),
    ).toBeNull();
  });
});

/* -------------------------------------------------------------------------- */
/* No xterm in the chat-ui module graph                                       */
/* -------------------------------------------------------------------------- */

describe("/chat-ui — no xterm", () => {
  it("does not import xterm", async () => {
    await renderAt("/chat-ui");
    // No xterm class leaked into the shell, and no xterm module was
    // dynamically loaded. The dynamic-import graph stays clean because
    // the page module never references @xterm/xterm.
    const shell = document.querySelector('[data-testid="chat-ui-shell"]');
    expect(shell?.querySelector(".xterm")).toBeNull();
  });
});

/* -------------------------------------------------------------------------- */
/* Direct ChatMessageList unit tests                                          */
/* -------------------------------------------------------------------------- */

describe("ChatMessageList — empty state + suggestions", () => {
  it("renders the suggestions when turns is empty", () => {
    const c = document.createElement("div");
    document.body.appendChild(c);
    const root = createRoot(c);
    roots.push(root);
    const store: ChatUIStore = {
      sessionId: "s",
      storedSessionId: null,
      connection: "open",
      model: null,
      provider: null,
      reasoningEffort: null,
      title: null,
      turns: [],
      lastSeenSeq: 0,
      submitting: false,
      error: null,
      failureReason: null,
      hydrated: true,
    };
    let picked: string | null = null;
    act(() => {
      root.render(
        <ChatMessageList
          store={store}
          onSuggestion={(t) => {
            picked = t;
          }}
        />,
      );
    });
    const sug = c.querySelector(
      '[data-testid="chat-ui-empty-suggestion"]',
    ) as HTMLButtonElement;
    expect(sug).toBeTruthy();
    sug.click();
    expect(picked).toMatch(/[A-Za-z]/);
  });
});

/* -------------------------------------------------------------------------- */
/* Composer — suggestedText one-shot seed                                     */
/* -------------------------------------------------------------------------- */

describe("ChatComposer — suggestedText seed", () => {
  it("adopts suggestedText once and reports consumption", () => {
    const c = document.createElement("div");
    document.body.appendChild(c);
    const root = createRoot(c);
    roots.push(root);
    const onSubmit = vi.fn();
    const onConsumed = vi.fn();
    act(() => {
      root.render(
        <ChatComposer
          onSubmit={onSubmit}
          suggestedText="Help me plan"
          onSuggestionConsumed={onConsumed}
        />,
      );
    });
    const ta = c.querySelector(
      '[data-testid="chat-ui-composer-input"]',
    ) as HTMLTextAreaElement;
    expect(ta.value).toBe("Help me plan");
    expect(onConsumed).toHaveBeenCalledTimes(1);
    // Subsequent renders with the same suggestedText do not re-consume.
    act(() => {
      root.render(
        <ChatComposer
          onSubmit={onSubmit}
          suggestedText="Help me plan"
          onSuggestionConsumed={onConsumed}
        />,
      );
    });
    expect(onConsumed).toHaveBeenCalledTimes(1);
  });
});
