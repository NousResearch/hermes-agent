export function flushInteractionTime(): void {}

export function updateLastInteractionTime(): void {}

export function markScrollActivity(): void {}

export function getIsInteractive(): boolean {
  return !!process.stdin.isTTY && !!process.stdout.isTTY
}
