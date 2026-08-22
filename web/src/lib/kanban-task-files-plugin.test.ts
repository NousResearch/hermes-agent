import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { describe, expect, it, vi } from "vitest";

type ElementNode = {
  type?: unknown;
  props?: Record<string, unknown> | null;
  children?: unknown[];
};

type RegisteredKanbanPage = {
  TaskFilesSection: (props: Record<string, unknown>) => unknown;
};

function loadKanbanPage(): RegisteredKanbanPage {
  const registration: { page?: RegisteredKanbanPage } = {};
  const createElement = (
    type: unknown,
    props: Record<string, unknown> | null,
    ...children: unknown[]
  ): ElementNode => ({
    type,
    props,
    children: children.flat(Number.POSITIVE_INFINITY),
  });

  const sandbox = {
    console,
    window: {
      confirm: () => true,
      localStorage: { getItem: () => null, setItem: () => undefined },
      __HERMES_PLUGINS__: {
        register: (_name: string, component: RegisteredKanbanPage) => {
          registration.page = component;
        },
      },
      __HERMES_PLUGIN_SDK__: {
        React: {
          createElement,
          Fragment: "fragment",
          Component: class {
            props: unknown;
            state = {};
            constructor(props: unknown) {
              this.props = props;
            }
          },
        },
        components: {
          Card: "Card",
          CardContent: "CardContent",
          Badge: "Badge",
          Button: "Button",
          Input: "Input",
          Label: "Label",
          Select: "Select",
          SelectOption: "SelectOption",
        },
        hooks: {
          useState: (initial: unknown) => [
            typeof initial === "function" ? (initial as () => unknown)() : initial,
            () => undefined,
          ],
          useEffect: () => undefined,
          useCallback: (fn: unknown) => fn,
          useMemo: (fn: () => unknown) => fn(),
          useRef: (value: unknown) => ({ current: value }),
        },
        utils: {
          cn: (...parts: unknown[]) => parts.filter(Boolean).join(" "),
          timeAgo: () => "",
        },
        fetchJSON: () => Promise.resolve({}),
        authedFetch: () => Promise.resolve({ ok: true }),
      },
    },
  };

  const bundle = path.resolve(
    import.meta.dirname,
    "../../../plugins/kanban/dashboard/dist/index.js",
  );
  vm.runInNewContext(fs.readFileSync(bundle, "utf8"), sandbox, { filename: bundle });

  const page = registration.page;
  if (!page) throw new Error("Kanban plugin did not register its page");
  expect(typeof page.TaskFilesSection).toBe("function");
  return page;
}

function collectText(node: unknown, texts: string[] = []): string[] {
  if (node == null || typeof node === "boolean") return texts;
  if (typeof node === "string" || typeof node === "number") {
    texts.push(String(node));
    return texts;
  }
  if (Array.isArray(node)) {
    node.forEach((child) => collectText(child, texts));
    return texts;
  }
  collectText((node as ElementNode).children, texts);
  return texts;
}

function findByTitle(node: unknown, title: string): ElementNode | undefined {
  if (node == null || typeof node !== "object") return undefined;
  if (Array.isArray(node)) {
    for (const child of node) {
      const match = findByTitle(child, title);
      if (match) return match;
    }
    return undefined;
  }
  const element = node as ElementNode;
  if (element.props?.title === title) return element;
  return findByTitle(element.children, title);
}

describe("Kanban task files plugin", () => {
  it("renders input attachments and generated artifacts as distinct groups", () => {
    const page = loadKanbanPage();
    const tree = page.TaskFilesSection({
      i18n: null,
      attachments: [
        { id: 1, filename: "brief.txt", size: 5, attachment_type: "attachment" },
        { id: 2, filename: "report.pdf", size: 9, attachment_type: "artifact" },
        { id: 3, filename: "legacy.txt", size: 1, attachment_type: null },
      ],
      onUpload: () => undefined,
      onDelete: () => undefined,
      uploadBusy: false,
    });

    const text = collectText(tree).join(" ");
    expect(text).toContain("Attachments (2)");
    expect(text).toContain("Artifacts (1)");
    expect(text.indexOf("brief.txt")).toBeLessThan(text.indexOf("Artifacts (1)"));
    expect(text.indexOf("Artifacts (1)")).toBeLessThan(text.indexOf("report.pdf"));
  });

  it("uses the in-app destructive dialog and warns that artifact deletion is irreversible", async () => {
    const page = loadKanbanPage();
    const requestDialog = vi.fn().mockResolvedValue({ confirmed: true });
    const onDelete = vi.fn();
    const tree = page.TaskFilesSection({
      i18n: null,
      attachments: [
        { id: 7, filename: "result.json", size: 12, attachment_type: "artifact" },
      ],
      onUpload: () => undefined,
      onDelete,
      requestDialog,
      uploadBusy: false,
    });

    const removeButton = findByTitle(tree, "Remove artifact");
    expect(removeButton).toBeDefined();
    const onClick = removeButton?.props?.onClick;
    expect(typeof onClick).toBe("function");
    (onClick as () => void)();
    await Promise.resolve();
    await Promise.resolve();

    expect(requestDialog).toHaveBeenCalledWith({
      kind: "confirm",
      title: "Remove artifact",
      description:
        "Permanently remove this generated artifact? This cannot be undone, and later workers may rely on it.",
      confirmLabel: "Delete",
      destructive: true,
    });
    expect(onDelete).toHaveBeenCalledWith(7);
  });
});
