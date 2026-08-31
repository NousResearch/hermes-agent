// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MarkdownMessage } from "./MarkdownMessage";

let container: HTMLDivElement;
let root: Root;

async function renderMessage(content: string) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(<MarkdownMessage content={content} />));
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.restoreAllMocks();
});

describe("MarkdownMessage", () => {
  it("preserves plain text whitespace and newlines", async () => {
    await renderMessage("first line\n  second line\n\nthird\n");
    expect(container.querySelector("p")?.textContent).toBe(
      "first line\n  second line\n\nthird\n",
    );
    expect(container.querySelector("p")?.className).toContain("whitespace-pre-wrap");
  });

  it("renders raw HTML as text rather than executable markup", async () => {
    await renderMessage("<img src=x onerror=alert(1)> <script>alert(1)</script>");
    expect(container.querySelector("script, img")).toBeNull();
    expect(container.textContent).toContain("<script>alert(1)</script>");
  });

  it("renders common markdown blocks and inline formatting", async () => {
    await renderMessage(
      "# Title\n\n- **bold**\n- *italic*\n\n> quoted\n\n[docs](https://example.com)",
    );
    expect(container.querySelector("h1")?.textContent).toBe("Title");
    expect(container.querySelector("ul")).toBeTruthy();
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("em")?.textContent).toBe("italic");
    expect(container.querySelector("blockquote")?.textContent).toBe("quoted");
    expect(container.querySelector('a[href="https://example.com"]')?.textContent).toBe("docs");
  });

  it("does not create unsafe links", async () => {
    await renderMessage("[bad](javascript:alert(1)) [data](data:text/html,evil)");
    expect(container.querySelectorAll("a")).toHaveLength(0);
    expect(container.textContent).toContain("bad");
    expect(container.textContent).toContain("data");
  });

  it("renders fenced code with a language label and copy button", async () => {
    await renderMessage("```typescript\nconst x = 1;\n```");
    expect(container.querySelector("pre")?.textContent).toContain("const x = 1;");
    expect(container.querySelector("[data-code-language]")?.textContent).toBe("typescript");
    expect(container.querySelector('button[aria-label="Copy code"]')).toBeTruthy();
    expect(container.querySelector("pre")?.className).toContain("overflow-x-auto");
  });

  it("copies code and reports copied state", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    await renderMessage("```js\nhello\n```");
    const button = container.querySelector<HTMLButtonElement>('button[aria-label="Copy code"]')!;
    await act(async () => button.click());
    expect(writeText).toHaveBeenCalledWith("hello");
    expect(button.textContent).toContain("Copied");
  });
});
