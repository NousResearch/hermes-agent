import type { ManagedPlatformInfo, MessagingPlatform } from "./api";

/** Which controls a Channels card renders once a host declares ownership. */
export interface ManagedChannelControls {
  managed: ManagedPlatformInfo | null;
  showToggle: boolean;
  showConfigure: boolean;
  showOnboarding: boolean;
  showTest: boolean;
  /** False when the native adapter is off by design (relay), so its
   * Disabled / Not configured badge would contradict a working channel. */
  showNativeState: boolean;
}

const UNMANAGED: ManagedChannelControls = {
  managed: null,
  showToggle: true,
  showConfigure: true,
  showOnboarding: true,
  showTest: true,
  showNativeState: true,
};

export function managedChannelControls(
  platform: Pick<MessagingPlatform, "managed" | "configured">,
): ManagedChannelControls {
  const managed = platform.managed ?? null;
  if (!managed) return UNMANAGED;
  if (managed.kind === "relay") {
    return {
      managed,
      showToggle: false,
      showConfigure: false,
      showOnboarding: false,
      showTest: false,
      showNativeState: false,
    };
  }
  return {
    managed,
    showToggle: false,
    showConfigure: false,
    showOnboarding: false,
    showTest: platform.configured,
    showNativeState: true,
  };
}
