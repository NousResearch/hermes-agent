export interface DashboardKeyEvent {
  altKey: boolean;
  ctrlKey: boolean;
  key: string;
  metaKey: boolean;
}

export function isDashboardPasteShortcut(ev: DashboardKeyEvent, isMac: boolean): boolean {
  if (ev.key.toLowerCase() !== "v") return false;
  if (ev.altKey) return false;
  return isMac ? ev.metaKey || ev.ctrlKey : ev.ctrlKey;
}
