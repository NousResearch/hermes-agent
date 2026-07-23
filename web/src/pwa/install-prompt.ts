export interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{
    outcome: "accepted" | "dismissed";
    platform: string;
  }>;
}

export interface InstallPromptState {
  available: boolean;
  prompting: boolean;
  standalone: boolean;
}

export type InstallPromptResult = "accepted" | "dismissed" | "unavailable";

interface DisplayModeQuery {
  readonly matches: boolean;
  addEventListener(
    type: "change",
    listener: (event: { matches: boolean }) => void,
  ): void;
  removeEventListener(
    type: "change",
    listener: (event: { matches: boolean }) => void,
  ): void;
}

export interface InstallPromptHost {
  addEventListener(type: string, listener: (event: Event) => void): void;
  removeEventListener(type: string, listener: (event: Event) => void): void;
  matchMedia(query: string): DisplayModeQuery;
}

export interface InstallPromptStore {
  getSnapshot(): InstallPromptState;
  prompt(): Promise<InstallPromptResult>;
  subscribe(listener: () => void): () => void;
  dispose(): void;
}

const UNAVAILABLE_STATE: InstallPromptState = Object.freeze({
  available: false,
  prompting: false,
  standalone: false,
});

export function createInstallPromptStore(
  host?: InstallPromptHost,
): InstallPromptStore {
  if (!host) {
    return {
      getSnapshot: () => UNAVAILABLE_STATE,
      prompt: async () => "unavailable",
      subscribe: () => () => undefined,
      dispose: () => undefined,
    };
  }

  const listeners = new Set<() => void>();
  const displayMode = host.matchMedia("(display-mode: standalone)");
  let standalone = displayMode.matches;
  let deferredPrompt: BeforeInstallPromptEvent | null = null;
  let prompting = false;
  let consumed = false;
  let state: InstallPromptState = {
    available: false,
    prompting: false,
    standalone,
  };

  const publish = () => {
    const next: InstallPromptState = {
      available:
        deferredPrompt !== null && !standalone && !prompting && !consumed,
      prompting,
      standalone,
    };
    if (
      next.available === state.available &&
      next.prompting === state.prompting &&
      next.standalone === state.standalone
    ) {
      return;
    }
    state = next;
    listeners.forEach((listener) => listener());
  };

  const onBeforeInstallPrompt = (rawEvent: Event) => {
    const event = rawEvent as BeforeInstallPromptEvent;
    if (
      consumed ||
      standalone ||
      typeof event.prompt !== "function" ||
      !event.userChoice
    ) {
      return;
    }
    event.preventDefault();
    deferredPrompt = event;
    publish();
  };

  const onAppInstalled = () => {
    consumed = true;
    deferredPrompt = null;
    publish();
  };

  const onDisplayModeChange = (event: { matches: boolean }) => {
    standalone = event.matches;
    if (standalone) deferredPrompt = null;
    publish();
  };

  host.addEventListener("beforeinstallprompt", onBeforeInstallPrompt);
  host.addEventListener("appinstalled", onAppInstalled);
  displayMode.addEventListener("change", onDisplayModeChange);

  return {
    getSnapshot: () => state,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    prompt: async () => {
      if (!deferredPrompt || standalone || consumed || prompting) {
        return "unavailable";
      }

      const promptEvent = deferredPrompt;
      deferredPrompt = null;
      prompting = true;
      consumed = true;
      publish();

      try {
        await promptEvent.prompt();
        const choice = await promptEvent.userChoice;
        return choice.outcome;
      } catch {
        return "unavailable";
      } finally {
        prompting = false;
        publish();
      }
    },
    dispose: () => {
      host.removeEventListener("beforeinstallprompt", onBeforeInstallPrompt);
      host.removeEventListener("appinstalled", onAppInstalled);
      displayMode.removeEventListener("change", onDisplayModeChange);
      listeners.clear();
    },
  };
}

export const installPromptStore = createInstallPromptStore(
  typeof window === "undefined"
    ? undefined
    : (window as unknown as InstallPromptHost),
);
