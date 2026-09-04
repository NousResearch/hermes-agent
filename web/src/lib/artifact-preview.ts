const PREVIEW_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; media-src data: blob:";

/**
 * Wrap model-controlled document text for an inert iframe preview. The iframe
 * itself has an empty sandbox, while this CSP also blocks network-capable
 * resource types if a browser relaxes one sandbox capability in the future.
 */
export function sandboxedArtifactDocument(source: string): string {
  const cspMeta = `<meta http-equiv="Content-Security-Policy" content="${PREVIEW_CSP}">`;
  const headPattern = /<head(?:\s[^>]*)?>/i;
  if (headPattern.test(source)) {
    return source.replace(headPattern, (head) => `${head}${cspMeta}`);
  }
  return `<!doctype html><html><head>${cspMeta}</head><body>${source}</body></html>`;
}
