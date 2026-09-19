import type { PluginCardWire } from '@hermes/shared/gateway-events'
import { atom } from 'nanostores'

export interface PluginCardEntry {
  card: PluginCardWire
  key: string
}

export interface PluginCardState {
  activeKey: null | string
  cards: PluginCardEntry[]
  sessionId: null | string
}

const emptyState = (): PluginCardState => ({ activeKey: null, cards: [], sessionId: null })

export const $pluginCards = atom<PluginCardState>(emptyState())

const cardKey = (card: PluginCardWire) => `${card.plugin_id}:${card.id}`

const put = (sessionId: string, card: PluginCardWire, activate: boolean) => {
  const previous = $pluginCards.get()
  const state = previous.sessionId === sessionId ? previous : emptyState()
  const key = cardKey(card)
  const cards = [...state.cards.filter(entry => entry.key !== key), { card, key }]
  $pluginCards.set({ activeKey: activate ? key : state.activeKey, cards, sessionId })
}

export const publishPluginCard = (sessionId: string, card: PluginCardWire) => put(sessionId, card, false)

export const activatePluginCard = (sessionId: string, card: PluginCardWire) => put(sessionId, card, true)

export const openPluginCard = (key?: string) => {
  const state = $pluginCards.get()
  const target = key ?? state.cards[0]?.key ?? null

  if (target && state.cards.some(entry => entry.key === target)) {
    $pluginCards.set({ ...state, activeKey: target })
  }
}

export const closePluginCard = () => $pluginCards.set({ ...$pluginCards.get(), activeKey: null })

export const dismissPluginCard = (key: string) => {
  const state = $pluginCards.get()
  $pluginCards.set({
    ...state,
    activeKey: state.activeKey === key ? null : state.activeKey,
    cards: state.cards.filter(entry => entry.key !== key)
  })
}

export const syncPluginCardSession = (sessionId: null | string) => {
  if ($pluginCards.get().sessionId !== sessionId) {
    $pluginCards.set({ activeKey: null, cards: [], sessionId })
  }
}

export const getPluginCardState = () => $pluginCards.get()
export const resetPluginCards = () => $pluginCards.set(emptyState())
