import type { AuthMeResponse } from "@/lib/api-types"

export function shouldHideAuthWidget(identity: AuthMeResponse): boolean {
  return identity.provider === 'loopback'
}
