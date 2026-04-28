import type { GatewayClient } from "@/lib/gatewayClient";
import { ListItem } from "@nous-research/ui";
import { ChevronRight } from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

/**
 * Slash-command autocomplete popover, rendered above the composer in ChatPage.
 * Mirrors the completion UX of the Ink TUI — type `/`, see matching commands,
 * arrow keys or click to select, Tab to apply, Enter to submit.
 *
 * The parent owns all keyboard handling via `ref.handleKey`, which returns
 * true when the popover consumed the event, so the composer's Enter/arrow
 * logic stays in one place.
 */

export interface CompletionItem {
  display: string;
  text: string;
  meta?: string;
}

export interface SlashPopoverHandle {
  /** Returns true if the key was consumed by the popover. */
  handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>): boolean;
}

interface Props {
  input: string;
  gw: GatewayClient | null;
  onApply(nextInput: string): void;
}

interface CompletionResponse {
  items?: CompletionItem[];
  replace_from?: number;
}

const DEBOUNCE_MS = 60;

export const SlashPopover = forwardRef<SlashPopoverHandle, Props>(
  function SlashPopover({ input, gw, onApply }, ref) {
    const [items, setItems] = useState<CompletionItem[]>([]);
    const [selected, setSelected] = useState(0);
    const [replaceFrom, setReplaceFrom] = useState(1);
    const lastInputRef = useRef<string>("");

    // Debounced completion fetch. We never clear `items` in the effect body
    // (doing so would flag react-hooks/set-state-in-effect); instead the
    // render guard below hides stale items once the input stops matching.
    useEffect(() => {
      const trimmed = input ?? "";

      if (!gw || !trimmed.startsWith("/") || trimmed === lastInputRef.current) {
        if (!trimmed.startsWith("/")) lastInputRef.current = "";
        return;
      }
      lastInputRef.current = trimmed;

      const timer = window.setTimeout(async () => {
        if (lastInputRef.current !== trimmed) return;
        try {
          const r = await gw.request<CompletionResponse>("complete.slash", {
            text: trimmed,
          });
          if (lastInputRef.current !== trimmed) return;
          setItems(r?.items ?? []);
          setReplaceFrom(r?.replace_from ?? 1);
          setSelected(0);
        } catch {
          if (lastInputRef.current === trimmed) setItems([]);
        }
      }, DEBOUNCE_MS);

      return () => window.clearTimeout(timer);
    }, [input, gw]);

    const apply = useCallback(
      (item: CompletionItem) => {
        onApply(input.slice(0, replaceFrom) + item.text);
      },
      [input, replaceFrom, onApply],
    );

    // Only consume keys when the popover is actually visible. Stale items from
    // a previous slash prefix are ignored once the user deletes the "/".
    const visible = items.length > 0 && input.startsWith("/");

    useImperativeHandle(
      ref,
      () => ({
        handleKey: (e) => {
          if (!visible) return false;

          switch (e.key) {
            case "ArrowDown":
              e.preventDefault();
              setSelected((s) => (s + 1) % items.length);
              return true;

            case "ArrowUp":
              e.preventDefault();
              setSelected((s) => (s - 1 + items.length) % items.length);
              return true;

            case "Tab": {
              e.preventDefault();
              const item = items[selected];
              if (item) apply(item);
              return true;
            }

            case "Escape":
              e.preventDefault();
              setItems([]);
              return true;

            default:
              return false;
          }
        },
      }),
      [visible, items, selected, apply],
    );

    if (!visible) return null;

    return (
      <div
        className="absolute bottom-full left-0 right-0 mb-2 max-h-64 overflow-y-auto rounded-md border border-border bg-popover shadow-xl text-sm"
        role="listbox"
      >
        {items.map((it, i) => {
          const active = i === selected;

          return (
            <ListItem
              key={`${it.text}-${i}`}
              active={active}
              role="option"
              aria-selected={active}
              onMouseEnter={() => setSelected(i)}
              onClick={() => apply(it)}
              className="px-3 py-1.5"
            >
              <ChevronRight
                className={`h-3 w-3 shrink-0 ${active ? "text-primary" : "text-transparent"}`}
              />

              <span className="font-mono text-xs shrink-0 truncate">
                {it.display}
              </span>

              {it.meta && (
                <span className="text-[0.7rem] text-muted-foreground/70 truncate ml-auto">
                  {it.meta}
                </span>
              )}
            </ListItem>
          );
        })}
      </div>
    );
  },
);
