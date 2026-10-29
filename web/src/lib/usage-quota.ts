export interface UsageAccountStatus {
  available: boolean;
  windows: unknown[];
  unavailable_reason?: string;
}

export function getUsageAccountUnavailableMessage(account: UsageAccountStatus): string | null {
  return account.available && account.windows.length > 0 ? null : account.unavailable_reason ?? "unavailable";
}
