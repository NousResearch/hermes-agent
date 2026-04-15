import { createContext, type ReactNode, useContext } from 'react'

import type { GatewayServices } from './interfaces.js'

const GatewayContext = createContext<GatewayServices | null>(null)

export interface GatewayProviderProps {
  children: ReactNode
  value: GatewayServices
}

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
