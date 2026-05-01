export function Spinner({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-spin rounded-full border-2 border-current border-t-transparent w-4 h-4 ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}
