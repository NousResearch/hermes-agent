// Simple event emitter for cross-component communication in Node.js/Ink
// Usage:
//   import { eventBus } from './eventBus.js'
//   eventBus.on('hermes:open-file', (path: string) => { ... })
//   eventBus.emit('hermes:open-file', '/path/to/file')

type Listener<T> = (payload: T) => void

class EventBus {
  private listeners = new Map<string, Listener<any>[]>()

  on<T>(event: string, listener: Listener<T>): () => void {
    const list = this.listeners.get(event) ?? []
    list.push(listener)
    this.listeners.set(event, list)
    return () => this.off(event, listener)
  }

  off<T>(event: string, listener: Listener<T>): void {
    const list = this.listeners.get(event)
    if (!list) return
    const idx = list.indexOf(listener)
    if (idx >= 0) list.splice(idx, 1)
  }

  emit<T>(event: string, payload: T): void {
    const list = this.listeners.get(event)
    if (!list) return
    for (const listener of list) {
      try {
        listener(payload)
      } catch (e) {
        console.error(`Error in event listener for ${event}:`, e)
      }
    }
  }
}

export const eventBus = new EventBus()