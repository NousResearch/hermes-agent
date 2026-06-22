export type NativeTerminalKeydown = {
  type: string;
  key: string;
  ctrlKey: boolean;
  altKey: boolean;
  metaKey: boolean;
  isComposing?: boolean;
  keyCode?: number;
  which?: number;
  altGraph?: boolean;
};

export function shouldDeferToNativeTerminalInput(
  event: NativeTerminalKeydown,
): boolean {
  if (event.type !== "keydown") return false;

  const altGraph = event.altGraph === true;
  const hasCommandModifier = event.metaKey || (event.ctrlKey && !altGraph);
  if (hasCommandModifier) return false;

  if (event.isComposing || altGraph) return true;

  if (
    event.key === "Dead" ||
    event.key === "Process" ||
    event.key === "Unidentified"
  ) {
    return true;
  }

  return event.keyCode === 229 || event.which === 229;
}

export function resolveNativeTerminalInputText(args: {
  armed: boolean;
  isComposing: boolean;
  data?: string | null;
  value?: string | null;
}): string | null {
  if (!args.armed || args.isComposing) return null;

  const value = args.value ?? "";
  if (value) return value;

  const data = args.data ?? "";
  return data || null;
}
