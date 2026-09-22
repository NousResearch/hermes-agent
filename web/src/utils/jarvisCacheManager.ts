/**
 * JARVIS Executive OS — Client Cache Engine
 * Manages offline persistence and AI agent response caching.
 */

export interface CachedAIItem {
  key: string;
  agentId: string;
  query: string;
  replyText: string;
  timestamp: string;
  expiresAt: string;
  hitsCount: number;
}

export interface CacheStats {
  totalCachedQueries: number;
  totalHits: number;
  totalMisses: number;
  hitRatePercentage: number;
}

const STORAGE_KEY_AI_CACHE = 'jarvis_ai_answers_cache';
const STORAGE_KEY_CACHE_STATS = 'jarvis_cache_stats';
const DEFAULT_TTL_MS = 30 * 60 * 1000; // 30 minutes

class CacheEngine {
  private memoryCache: Map<string, CachedAIItem> = new Map();
  private hits: number = 0;
  private misses: number = 0;

  constructor() {
    this.loadFromStorage();
  }

  public generateKey(agentId: string, prompt: string): string {
    const sanitized = prompt.trim().toLowerCase();
    return `${agentId}:${sanitized}`;
  }

  private loadFromStorage() {
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') return;
    try {
      const savedCache = localStorage.getItem(STORAGE_KEY_AI_CACHE);
      if (savedCache) {
        const parsed: CachedAIItem[] = JSON.parse(savedCache);
        const now = Date.now();
        parsed.forEach((item) => {
          if (new Date(item.expiresAt).getTime() > now) {
            this.memoryCache.set(item.key, item);
          }
        });
      }

      const savedStats = localStorage.getItem(STORAGE_KEY_CACHE_STATS);
      if (savedStats) {
        const stats = JSON.parse(savedStats);
        this.hits = stats.hits || 0;
        this.misses = stats.misses || 0;
      }
    } catch (e) {
      console.warn('[CacheEngine] Error loading cache:', e);
    }
  }

  private saveToStorage() {
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') return;
    try {
      const array = Array.from(this.memoryCache.values());
      localStorage.setItem(STORAGE_KEY_AI_CACHE, JSON.stringify(array));
      localStorage.setItem(
        STORAGE_KEY_CACHE_STATS,
        JSON.stringify({ hits: this.hits, misses: this.misses })
      );
    } catch (e) {
      console.warn('[CacheEngine] Error saving cache:', e);
    }
  }

  public getCachedAnswer(agentId: string, prompt: string): CachedAIItem | null {
    const key = this.generateKey(agentId, prompt);
    const item = this.memoryCache.get(key);
    if (!item) {
      this.misses++;
      this.saveToStorage();
      return null;
    }

    if (new Date(item.expiresAt).getTime() < Date.now()) {
      this.memoryCache.delete(key);
      this.misses++;
      this.saveToStorage();
      return null;
    }

    item.hitsCount++;
    this.hits++;
    this.saveToStorage();
    return item;
  }

  public setCachedAnswer(agentId: string, prompt: string, replyText: string): CachedAIItem {
    const key = this.generateKey(agentId, prompt);
    const now = new Date();
    const item: CachedAIItem = {
      key,
      agentId,
      query: prompt,
      replyText,
      timestamp: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      expiresAt: new Date(now.getTime() + DEFAULT_TTL_MS).toISOString(),
      hitsCount: 0,
    };
    this.memoryCache.set(key, item);
    this.saveToStorage();
    return item;
  }

  public getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      totalCachedQueries: this.memoryCache.size,
      totalHits: this.hits,
      totalMisses: this.misses,
      hitRatePercentage: total > 0 ? Math.round((this.hits / total) * 100) : 0,
    };
  }
}

export const cacheEngine = new CacheEngine();
