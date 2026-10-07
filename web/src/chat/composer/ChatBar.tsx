import { useCallback, useEffect, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { SendHorizonal } from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ChatBarProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  profile?: string;
}

export function ChatBar({ onSend, disabled, profile }: ChatBarProps) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getModelInfo(profile)
      .then((r) => {
        if (!cancelled && r?.model) setModel(String(r.model));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [profile]);

  const send = useCallback(() => {
    const value = text.trim();
    if (!value || disabled) return;
    setText("");
    onSend(value);
  }, [text, disabled, onSend]);

  return (
    <div className="flex shrink-0 flex-col gap-2 border-t border-current/10 pt-2">
      {model && (
        <span className="w-fit rounded-full border border-current/20 px-2 py-0.5 text-[0.6875rem] text-text-secondary">
          {model}
        </span>
      )}
      <div className="flex items-end gap-2">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          disabled={disabled}
          placeholder="Message…"
          aria-label="Message"
          className={cn(
            "min-h-10 max-h-40 flex-1 resize-y rounded border border-current/20",
            "bg-background-base px-3 py-2 text-sm text-foreground",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground",
          )}
        />
        <Button
          outlined
          size="sm"
          onClick={send}
          disabled={disabled || !text.trim()}
          prefix={<SendHorizonal />}
          aria-label="Send"
        >
          Send
        </Button>
      </div>
    </div>
  );
}
