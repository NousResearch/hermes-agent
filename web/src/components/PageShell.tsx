import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface PageShellProps {
  children: ReactNode;
  title?: string;
  actions?: ReactNode;
  className?: string;
  /** Override max-width (default: max-w-6xl) */
  maxWidth?: string;
}

/**
 * Shared page wrapper — consistent layout for all content pages.
 * Provides max-width constraint, page title, action buttons, and scrollable area.
 */
export function PageShell({
  children,
  title,
  actions,
  className,
  maxWidth = 'max-w-6xl',
}: PageShellProps) {
  return (
    <div className="h-full overflow-y-auto">
      <div
        className={cn('mx-auto w-full px-2 py-6 sm:px-4 sm:py-8', maxWidth, className)}
        style={{ animation: 'page-enter 200ms ease-out' }}
      >
        {(title || actions) && (
          <div className="mb-6 flex items-center justify-between gap-4 flex-wrap">
            {title && (
              <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                {title}
              </h1>
            )}
            {actions && (
              <div className="flex items-center gap-2">{actions}</div>
            )}
          </div>
        )}
        {children}
      </div>
    </div>
  );
}
