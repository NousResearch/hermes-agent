import type { JsonRpcRequestChannel } from '@hermes/shared/json-rpc-channel'
import { describe, expect, it } from 'vitest'

import { GatewayClient } from '../gatewayClient.js'

// The TUI channel counts streamed deltas as liveness (#115251): a turn that
// streams a frame per second for minutes must never trip the 45s deadline,
// while a socket gone fully silent still fails. This pins the wiring option
// on the only construction site outside @hermes/shared.
describe('GatewayClient heartbeat liveness wiring', () => {
  it('constructs the channel with heartbeatLiveness any-inbound', () => {
    const client = new GatewayClient()

    try {
      const channel = (client as unknown as { channel: JsonRpcRequestChannel }).channel
      const liveness = (channel as unknown as { options: { heartbeatLiveness?: string } }).options.heartbeatLiveness

      expect(liveness).toBe('any-inbound')
    } finally {
      client.kill()
    }
  })
})
