import { Button } from "@nous-research/ui/ui/components/button";
import { AUTH_UNAUTHORIZED_MESSAGE } from "@/lib/api";

interface UnauthorizedStateProps {
  service: string;
  message?: string | null;
  loginUrl?: string;
  onRetry?: () => void;
  compact?: boolean;
}

export function UnauthorizedState({
  service,
  message = AUTH_UNAUTHORIZED_MESSAGE,
  loginUrl,
  onRetry,
  compact = false,
}: UnauthorizedStateProps) {
  const detail = message || AUTH_UNAUTHORIZED_MESSAGE;

  return (
    <div
      role="alert"
      className={
        compact
          ? "rounded-md border border-destructive/40 bg-destructive/10 p-2 text-xs text-destructive"
          : "flex flex-col gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-5 text-destructive"
      }
    >
      <div>
        <p className="font-mondwest uppercase tracking-wide">
          {service} access unauthorized
        </p>
        <p className="mt-1 text-sm text-destructive/90">{detail}</p>
      </div>
      {(loginUrl || onRetry) && (
        <div className="flex flex-wrap gap-2">
          {loginUrl && (
            <a
              href={loginUrl}
              className="inline-flex h-9 w-fit items-center justify-center rounded-md bg-primary px-3 text-xs font-medium uppercase text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              Sign in
            </a>
          )}
          {onRetry && (
            <Button
              size="sm"
              className="w-fit uppercase"
              onClick={onRetry}
            >
              Retry
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
