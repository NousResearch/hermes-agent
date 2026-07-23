// A PTY attachment identifies a browser tab's live TUI process. sessionStorage
// survives reloads in that tab, but Chromium clones it along with history when
// duplicating a tab. A live tab confirms its ownership before another page
// reuses a stored token, so clone detection does not depend on navigation type.
const PTY_ATTACH_TOKEN_KEY = "hermes.pty.token.chat";
const PTY_ATTACH_CHANNEL = "hermes.pty.attach";
const PTY_ATTACH_CLAIM_TIMEOUT_MS = 75;
let pageToken = "";
let pageIsLeaving = false;
let ownerChannel: BroadcastChannel | null = null;

interface AttachMessage {
  type: "claim" | "claimed";
  token: string;
}

function isAttachMessage(value: unknown): value is AttachMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<AttachMessage>;
  return (
    (message.type === "claim" || message.type === "claimed") &&
    typeof message.token === "string"
  );
}

function attachmentChannel(): BroadcastChannel | null {
  if (ownerChannel) return ownerChannel;

  try {
    const channel = new BroadcastChannel(PTY_ATTACH_CHANNEL);
    channel.addEventListener("message", (event) => {
      if (!isAttachMessage(event.data)) return;
      if (
        event.data.type === "claim" &&
        event.data.token === pageToken &&
        !pageIsLeaving
      ) {
        channel.postMessage({ type: "claimed", token: pageToken });
      }
    });
    window.addEventListener("pagehide", () => {
      pageIsLeaving = true;
    });
    window.addEventListener("pageshow", () => {
      pageIsLeaving = false;
    });
    ownerChannel = channel;
    return channel;
  } catch {
    return null;
  }
}

function tokenIsOwnedByLiveTab(token: string): Promise<boolean> {
  const channel = attachmentChannel();
  if (!channel) return Promise.resolve(true);

  return new Promise((resolve) => {
    let settled = false;
    const finish = (owned: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      channel.removeEventListener("message", onMessage);
      resolve(owned);
    };
    const onMessage = (event: MessageEvent<unknown>) => {
      if (
        isAttachMessage(event.data) &&
        event.data.type === "claimed" &&
        event.data.token === token
      ) {
        finish(true);
      }
    };
    const timeout = setTimeout(
      () => finish(false),
      PTY_ATTACH_CLAIM_TIMEOUT_MS,
    );
    channel.addEventListener("message", onMessage);
    channel.postMessage({ type: "claim", token });
  });
}

export async function ptyAttachToken(rotate = false): Promise<string> {
  if (!rotate && pageToken) return pageToken;

  let token = "";
  if (!rotate) {
    try {
      token = window.sessionStorage.getItem(PTY_ATTACH_TOKEN_KEY) ?? "";
    } catch {
      /* private mode / storage blocked */
    }
  }
  if (token && (await tokenIsOwnedByLiveTab(token))) token = "";
  if (!token) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    token = Array.from(bytes, (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    try {
      window.sessionStorage.setItem(PTY_ATTACH_TOKEN_KEY, token);
    } catch {
      /* ignore */
    }
  }
  pageToken = token;
  attachmentChannel();
  return token;
}
