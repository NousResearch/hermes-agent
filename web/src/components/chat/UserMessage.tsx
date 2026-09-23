/**
 * User turn rendered by the new structured Chat UI.
 *
 * Mirrors the assistant row's chrome (role label, timestamp, copy action)
 * but in the user's accent color and right-aligned on wide screens.
 *
 * Attachments render as inline previews alongside the user text — the
 * gateway stores them under HERMES_HOME/images (and PDFs under
 * HERMES_HOME/pdfs) and the chat session references them on the next turn.
 *
 * Phase 4 polish:
 *   - Always-visible Copy button (not hover-gated) for keyboard users.
 *   - Attachment chip mirrors the composer chip: image thumbnail / PDF
 *     icon + filename + subtitle (dimensions or page count).
 *   - Same `kind` discriminator as `ChatAttachment` so we render image vs
 *     PDF consistently with the composer chip.
 */

import { Check, Copy, FileText, Image as ImageIcon } from "lucide-react";
import { useState } from "react";

import { Markdown } from "@/components/Markdown";
import type { ChatAttachment, ChatUIUserTurn } from "@/components/chat/types";
import { copyTextToClipboard } from "@/lib/clipboard";
import { timeAgo } from "@/lib/utils";

interface UserMessageProps {
  turn: ChatUIUserTurn;
}

export function UserMessage({ turn }: UserMessageProps) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    const ok = await copyTextToClipboard(turn.text);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <article
      className="group/turn ml-auto flex max-w-3xl flex-col gap-2 rounded-lg border border-current/10 bg-card px-4 py-3"
      data-role="user"
      data-testid="chat-ui-user-message"
    >
      <header className="flex items-center gap-2 text-xs">
        <span className="font-mono-ui text-xs uppercase tracking-wider text-primary">
          You
        </span>
        <span
          className="text-text-tertiary"
          title={new Date(turn.timestamp * 1000).toLocaleString()}
        >
          {timeAgo(turn.timestamp)}
        </span>
        <button
          type="button"
          onClick={onCopy}
          aria-label="Copy message"
          aria-pressed={copied}
          className="ml-auto flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-text-tertiary opacity-80 transition-colors hover:bg-midground/10 hover:text-foreground focus:bg-midground/10 focus:text-foreground focus:opacity-100 focus:outline-none"
          data-testid="chat-ui-copy"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-success" aria-hidden />
              <span className="hidden sm:inline">Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" aria-hidden />
              <span className="hidden sm:inline">Copy</span>
            </>
          )}
        </button>
      </header>
      <div className="text-[14px] leading-6 text-foreground">
        <Markdown content={turn.text} />
      </div>
      {turn.attachments && turn.attachments.length > 0 && (
        <ul
          className="flex flex-wrap gap-1.5"
          data-testid="chat-ui-user-attachments"
          aria-label="Attachments"
        >
          {turn.attachments.map((a) => (
            <UserAttachmentChip key={a.id} attachment={a} />
          ))}
        </ul>
      )}
    </article>
  );
}

function UserAttachmentChip({ attachment }: { attachment: ChatAttachment }) {
  const displayName = attachment.name ?? "attachment";
  const subtitle =
    attachment.kind === "pdf" && attachment.pageCount
      ? `${attachment.pageCount} page${attachment.pageCount === 1 ? "" : "s"}`
      : attachment.width && attachment.height
        ? `${attachment.width}×${attachment.height}`
        : null;
  const isImage = attachment.kind === "image";
  return (
    <li
      className="flex items-center gap-1.5 rounded-full border border-current/15 bg-background-base/60 py-0.5 pl-1 pr-2 text-xs"
      data-testid="chat-ui-user-attachment"
      data-attachment-kind={attachment.kind}
      data-attachment-id={attachment.id}
    >
      {isImage && attachment.dataUri ? (
        // eslint-disable-next-line jsx-a11y/img-redundant-alt
        <img
          src={attachment.dataUri}
          alt=""
          aria-hidden
          className="h-5 w-5 rounded-full border border-current/10 object-cover"
        />
      ) : isImage ? (
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-midground/30">
          <ImageIcon className="h-3 w-3 text-text-secondary" aria-hidden />
        </span>
      ) : (
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500/10 text-red-500">
          <FileText className="h-3 w-3" aria-hidden />
        </span>
      )}
      <span className="max-w-[12rem] truncate font-medium" title={displayName}>
        {displayName}
      </span>
      {subtitle && (
        <span className="text-text-tertiary" aria-hidden>
          {" "}
          · {subtitle}
        </span>
      )}
    </li>
  );
}
