import { createContext, useContext } from 'react'

import type { GatewayProviderProps, GatewayServices } from './interfaces.js'

const GatewayContext = createContext<GatewayServices | null>(null)

export function GatewayProvider({ children, value }: GatewayProviderProps) {
  return <GatewayContext.Provider value={value}>{children}</GatewayContext.Provider>
}

export function useGateway() {
  const value = useContext(GatewayContext)

  if (!value) {
    throw new Error('GatewayContext missing')
  }

  return value
}
