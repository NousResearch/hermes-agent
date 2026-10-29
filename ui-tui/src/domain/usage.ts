import type { Usage } from '@hermes/shared/gateway-events'

export { usageFromTick } from '@hermes/shared/usage-tick'

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }
