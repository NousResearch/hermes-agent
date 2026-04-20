export class CircularBuffer<T> {
  private buf: T[]
  private head = 0
  private len = 0

  constructor(private capacity: number) {
    this.buf = new Array<T>(capacity)
  }

  push(item: T) {
    this.buf[this.head] = item
    this.head = (this.head + 1) % this.capacity

    if (this.len < this.capacity) {
      this.len++
    }
  }

  tail(n = this.len): T[] {
    const take = Math.min(Math.max(0, n), this.len)
    const start = this.len < this.capacity ? 0 : this.head
    const out: T[] = new Array<T>(take)

    for (let i = 0; i < take; i++) {
      out[i] = this.buf[(start + this.len - take + i) % this.capacity]!
    }

    return out
  }

  drain(): T[] {
    const out = this.tail()

    this.clear()

    return out
  }

  clear() {
    this.buf = new Array<T>(this.capacity)
    this.head = 0
    this.len = 0
  }
}
