export function filesPathFromSearchParams(searchParams: URLSearchParams): string | undefined {
  const requestedPath = searchParams.get("path")?.trim();
  return requestedPath || undefined;
}