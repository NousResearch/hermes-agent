import type { ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function ExternalLink({ href, children }: ComponentProps<"a">) {
  const external = /^(https?:|mailto:)/i.test(href ?? "");
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noreferrer" : undefined}
      className="font-medium text-primary underline decoration-primary/35 underline-offset-2 hover:decoration-primary"
    >
      {children}
    </a>
  );
}

export function FileMarkdown({ content }: { content: string }) {
  return (
    <article className="min-w-0 max-w-none space-y-4 break-words text-sm leading-7 text-foreground">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ExternalLink,
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-border pl-4 text-text-secondary">
              {children}
            </blockquote>
          ),
          code: ({ children, className }) => (
            <code
              className={`${className ?? ""} rounded-sm bg-secondary/70 px-1.5 py-0.5 font-mono text-xs text-foreground`}
            >
              {children}
            </code>
          ),
          h1: ({ children }) => (
            <h1 className="border-b border-border pb-2 text-3xl font-semibold">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="border-b border-border pb-2 text-2xl font-semibold">{children}</h2>
          ),
          h3: ({ children }) => <h3 className="text-xl font-semibold">{children}</h3>,
          h4: ({ children }) => <h4 className="text-lg font-semibold">{children}</h4>,
          hr: () => <hr className="border-border" />,
          img: ({ alt, src }) => (
            <img
              src={src}
              alt={alt ?? ""}
              className="max-w-full border border-border bg-background object-contain"
            />
          ),
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-7">{children}</ol>,
          p: ({ children }) => <p className="whitespace-pre-wrap">{children}</p>,
          pre: ({ children }) => (
            <pre className="overflow-x-auto border border-border bg-secondary/50 p-4 font-mono text-xs leading-6 text-foreground [&>code]:bg-transparent [&>code]:p-0">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <table className="w-full border-collapse text-left text-sm">{children}</table>
          ),
          td: ({ children }) => <td className="border border-border px-3 py-2">{children}</td>,
          th: ({ children }) => (
            <th className="border border-border bg-secondary/60 px-3 py-2 font-semibold">
              {children}
            </th>
          ),
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-7">{children}</ul>,
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}
