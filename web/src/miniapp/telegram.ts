// Thin wrapper over Telegram's WebApp JS SDK (injected as
// window.Telegram.WebApp by the Telegram client itself when this page loads
// inside a Mini App WebView). Every accessor degrades gracefully when
// window.Telegram is absent (e.g. this page opened directly in a desktop
// browser, or during local dev) — nothing here should ever throw just
// because it isn't actually running inside Telegram.

interface TelegramWebApp {
  initData: string;
  // Documented values: "android", "ios", "web", "tdesktop", "macos",
  // "weba" (web app), "unknown". Used ONLY to decide whether to render our
  // own close affordance (see shouldShowCloseButton) -- Telegram's own docs
  // don't specify per-platform close-chrome behavior, so this is a
  // best-effort signal, not a confirmed source of truth for every value.
  platform?: string;
  ready: () => void;
  expand: () => void;
  close: () => void;
  BackButton: {
    show: () => void;
    hide: () => void;
    onClick: (cb: () => void) => void;
    offClick: (cb: () => void) => void;
  };
  MainButton: {
    show: () => void;
    hide: () => void;
    setText: (text: string) => void;
    setParams: (params: { color?: string; text_color?: string }) => void;
    onClick: (cb: () => void) => void;
    offClick: (cb: () => void) => void;
  };
  HapticFeedback?: {
    notificationOccurred: (kind: "error" | "success" | "warning") => void;
  };
  CloudStorage?: {
    getItem: (key: string, cb: (err: unknown, value: string | null) => void) => void;
    setItem: (key: string, value: string, cb: (err: unknown) => void) => void;
  };
  onEvent: (event: string, cb: () => void) => void;
  offEvent: (event: string, cb: () => void) => void;
}

declare global {
  interface Window {
    Telegram?: { WebApp?: TelegramWebApp };
  }
}

function tg(): TelegramWebApp | undefined {
  return typeof window !== "undefined" ? window.Telegram?.WebApp : undefined;
}

export function isInsideTelegram(): boolean {
  return !!tg();
}

// Platforms confirmed (by direct testing against this deployment) to
// already show their OWN native close affordance around the Mini App, so
// this app's in-page close button would just be a redundant duplicate.
// "ios" deliberately NOT listed: Telegram's docs don't document
// per-platform close-chrome behavior one way or the other, and it hasn't
// been confirmed on a real iOS client -- unlisted/unknown platforms default
// to SHOWING the close button (fail toward "an extra working control"
// rather than "silently no way to leave the app").
const PLATFORMS_WITH_CONFIRMED_NATIVE_CLOSE = new Set(["android", "tdesktop"]);

export function shouldShowCloseButton(): boolean {
  const app = tg();
  if (!app) return true; // outside Telegram (browser preview / local dev): always show
  const platform = app.platform ?? "";
  return !PLATFORMS_WITH_CONFIRMED_NATIVE_CLOSE.has(platform);
}

/** Closes the Mini App via Telegram's own documented WebApp.close(). Used by
 * the in-page close button on platforms where a native one isn't confirmed
 * (see shouldShowCloseButton) -- this is the correct, official way for a
 * Mini App to dismiss itself, regardless of platform. */
export function closeMiniApp(): void {
  const app = tg();
  if (!app) return;
  try {
    app.close();
  } catch {
    /* no-op outside Telegram / unsupported on this client version */
  }
}

/** Raw initData string to send as `Authorization: Bearer <initData>` on
 * every Mini App API call — this IS the credential, verified server-side by
 * initdata.py's HMAC check. Empty outside Telegram (e.g. local dev preview),
 * which every backend endpoint here treats as an anonymous/no-token caller. */
export function getInitData(): string {
  return tg()?.initData ?? "";
}

export function readyAndExpand(): void {
  const app = tg();
  if (!app) return;
  try {
    app.ready();
    app.expand();
  } catch {
    /* not actually inside Telegram despite window.Telegram existing */
  }
}

export function showBackButton(onClick: () => void): void {
  const app = tg();
  if (!app) return;
  // onClick registered BEFORE show(), each individually guarded: on an
  // older client one of these can throw WebAppMethodUnsupported (see the
  // identical fix in showMainButton). Registering the click handler first
  // means taps still work even if show() itself throws and the native
  // button never visually renders -- the reverse order risked a visible
  // button that silently ate taps.
  try {
    app.BackButton.onClick(onClick);
  } catch {
    /* no-op outside Telegram / unsupported on this client version */
  }
  try {
    app.BackButton.show();
  } catch {
    /* no-op outside Telegram / unsupported on this client version */
  }
}

export function hideBackButton(onClick: () => void): void {
  const app = tg();
  if (!app) return;
  try {
    app.BackButton.offClick(onClick);
  } catch {
    /* no-op outside Telegram / unsupported on this client version */
  }
  try {
    app.BackButton.hide();
  } catch {
    /* no-op outside Telegram / unsupported on this client version */
  }
}

export interface MainButtonOptions {
  text: string;
  destructive?: boolean;
  onClick: () => void;
}

/** Shows Telegram's native bottom button as the primary action on a confirm
 * sheet. Callers MUST call hideMainButton with the SAME onClick when the
 * sheet closes, or the native button leaks into whatever screen is shown
 * next with a stale label/handler. */
export function showMainButton({ text, destructive, onClick }: MainButtonOptions): void {
  const app = tg();
  if (!app) return;
  // Each call is guarded INDIVIDUALLY, and the load-bearing ones (onClick +
  // show) run BEFORE the cosmetic setParams. On older Telegram clients some
  // WebApp methods throw `WebAppMethodUnsupported` when called (setParams
  // with a color is a common one). The original order — setText, setParams,
  // onClick, show, all in one try — meant a setParams throw skipped BOTH
  // onClick and show, so the native button either never appeared or appeared
  // with no handler (a dead confirm button on older clients). Registering
  // the click + showing first means the button always works; the color is
  // best-effort on top.
  const attempt = (fn: () => void) => {
    try {
      fn();
    } catch {
      /* method unsupported on this client version — skip just this one */
    }
  };
  attempt(() => app.MainButton.onClick(onClick));
  attempt(() => app.MainButton.setText(text));
  attempt(() => app.MainButton.show());
  attempt(() =>
    app.MainButton.setParams(destructive ? { color: "#c0392b", text_color: "#ffffff" } : {}),
  );
}

export function hideMainButton(onClick: () => void): void {
  const app = tg();
  if (!app) return;
  try {
    app.MainButton.offClick(onClick);
    app.MainButton.hide();
  } catch {
    /* no-op outside Telegram */
  }
}

export function haptic(kind: "error" | "success" | "warning"): void {
  const app = tg();
  if (!app?.HapticFeedback) return;
  try {
    app.HapticFeedback.notificationOccurred(kind);
  } catch {
    /* no-op outside Telegram */
  }
}

const PALETTE_STORAGE_KEY = "hermes-mini-palette";

function localGet(): string | null {
  try {
    return localStorage.getItem(PALETTE_STORAGE_KEY);
  } catch {
    return null;
  }
}

/** Reads the persisted palette choice: Telegram's CloudStorage when it's
 * actually usable (syncs across the user's devices), else localStorage.
 *
 * NEVER rejects. CloudStorage.getItem can EXIST as a method yet THROW
 * `WebAppMethodUnsupported` when called on an older Telegram client (the
 * SDK reports e.g. "version 6.0") -- checking for the method's presence
 * isn't enough, the call itself has to be in a try/catch, and any failure
 * falls back to localStorage. The original version let that throw reject
 * the promise (an unhandled rejection at mount, since the caller only
 * `.then`s it). */
export function loadPersistedPalette(): Promise<string | null> {
  const app = tg();
  if (app?.CloudStorage?.getItem) {
    try {
      return new Promise((resolve) => {
        try {
          app.CloudStorage!.getItem(PALETTE_STORAGE_KEY, (err, value) => {
            resolve(!err && value ? value : localGet());
          });
        } catch {
          resolve(localGet()); // method unsupported on this client version
        }
      });
    } catch {
      return Promise.resolve(localGet());
    }
  }
  return Promise.resolve(localGet());
}

export function persistPalette(key: string): void {
  const app = tg();
  if (app?.CloudStorage?.setItem) {
    try {
      app.CloudStorage.setItem(PALETTE_STORAGE_KEY, key, () => {});
      return;
    } catch {
      /* method unsupported on this client version — fall through to local */
    }
  }
  try {
    localStorage.setItem(PALETTE_STORAGE_KEY, key);
  } catch {
    /* privacy mode / unavailable — palette just won't persist */
  }
}
