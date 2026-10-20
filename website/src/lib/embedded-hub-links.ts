/** Opt-in link handoff for sandboxed Desktop pickers; standalone pages keep normal navigation. */
export function installEmbeddedHubLinks(host: Window): () => void {
  if (host.parent === host) return () => {};

  let enabled = false;
  const onMessage = (event: MessageEvent) => {
    if (event.source === host.parent && event.data?.type === "hermes-hub-links-enable") {
      enabled = true;
    }
  };
  const onClick = (event: MouseEvent) => {
    if (!enabled || !event.isTrusted || event.button !== 0) return;
    const target = event.target as Element | null;
    const anchor = target?.closest<HTMLAnchorElement>("a[href]");
    if (!anchor) return;
    const url = new URL(anchor.href);
    const external = ["http:", "https:"].includes(url.protocol) &&
      (anchor.target === "_blank" || url.origin !== host.location.origin);
    const catalogInstall = url.protocol === "hermes:" && url.hostname === "plugin" &&
      url.pathname === "/install" && url.searchParams.has("catalog");
    if (!external && !catalogInstall) return;

    event.preventDefault();
    host.parent.postMessage({ type: "hermes-hub-open-link", url: url.href }, "*");
  };

  host.addEventListener("message", onMessage);
  host.document.addEventListener("click", onClick, true);
  host.parent.postMessage({ type: "hermes-hub-links-ready" }, "*");
  return () => {
    host.removeEventListener("message", onMessage);
    host.document.removeEventListener("click", onClick, true);
  };
}
