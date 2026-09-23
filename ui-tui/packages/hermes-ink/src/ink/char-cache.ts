const UTF16_BYTES_PER_CODE_UNIT = 2
const ENTRY_OVERHEAD_BYTES = 64
const CLUSTER_OVERHEAD_BYTES = 64

export const DEFAULT_CHAR_CACHE_MAX_BYTES = 8 * 1024 * 1024
export const DEFAULT_CHAR_CACHE_MAX_ENTRIES = 16384

type WeightedCluster = {
  value: string
}

type CacheEntry<T> = {
  value: T
  weight: number
}

/**
 * LRU cache for clustered terminal lines, bounded by estimated retained bytes.
 *
 * The estimate intentionally models the dominant retained data rather than
 * claiming exact V8 heap accounting: UTF-16 storage for the source key and
 * cluster values, plus fixed overhead for the Map entry and each cluster
 * object. It does not attempt to price every engine detail or shared metadata
 * such as hyperlink strings. The defaults approximate the measured retained
 * slope while keeping a count backstop for unusually small entries.
 */
export class CharCache<T extends readonly WeightedCluster[]> {
  private readonly entries = new Map<string, CacheEntry<T>>()
  private weight = 0

  constructor(
    private readonly maxBytes = DEFAULT_CHAR_CACHE_MAX_BYTES,
    private readonly maxEntries = DEFAULT_CHAR_CACHE_MAX_ENTRIES
  ) {}

  get estimatedBytes(): number {
    return this.weight
  }

  get size(): number {
    return this.entries.size
  }

  get(key: string): T | undefined {
    const entry = this.entries.get(key)

    if (!entry) {
      return undefined
    }

    this.entries.delete(key)
    this.entries.set(key, entry)

    return entry.value
  }

  set(key: string, value: T): void {
    const weight = estimateEntryBytes(key, value)

    if (weight > this.maxBytes || this.maxEntries <= 0) {
      return
    }

    const existing = this.entries.get(key)

    if (existing) {
      this.entries.delete(key)
      this.weight -= existing.weight
    }

    while (this.entries.size >= this.maxEntries || this.weight + weight > this.maxBytes) {
      const oldestKey = this.entries.keys().next().value

      if (oldestKey === undefined) {
        break
      }

      const oldest = this.entries.get(oldestKey)!
      this.entries.delete(oldestKey)
      this.weight -= oldest.weight
    }

    this.entries.set(key, { value, weight })
    this.weight += weight
  }

  clear(): void {
    this.entries.clear()
    this.weight = 0
  }
}

function estimateEntryBytes(key: string, clusters: readonly WeightedCluster[]): number {
  let bytes = ENTRY_OVERHEAD_BYTES + key.length * UTF16_BYTES_PER_CODE_UNIT

  for (const cluster of clusters) {
    bytes += CLUSTER_OVERHEAD_BYTES + cluster.value.length * UTF16_BYTES_PER_CODE_UNIT
  }

  return bytes
}
