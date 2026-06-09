export function nextHasEverActivated(prev: boolean, isActive: boolean): boolean {
  return prev || isActive;
}
