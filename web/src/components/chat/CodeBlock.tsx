import { useState } from "react";
import { copyTextToClipboard } from "@/lib/clipboard";

type CodeBlockProps = {
  code: string;
  language?: string;
};

export function CodeBlock({ code, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    try {
      if (await copyTextToClipboard(code, true)) {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      } else {
        setCopied(false);
      }
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="overflow-hidden rounded-md border border-border bg-secondary/60">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 text-xs text-muted-foreground">
        <span data-code-language>{language || "Code"}</span>
        <button
          type="button"
          aria-label="Copy code"
          className="rounded px-2 py-1 hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-ring"
          onClick={copyCode}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="max-w-full overflow-x-auto p-3 text-xs leading-relaxed whitespace-pre-wrap break-words sm:whitespace-pre sm:break-normal">
        <code>{code}</code>
      </pre>
    </div>
  );
}
