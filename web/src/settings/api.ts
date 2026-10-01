import { fetchJSON } from '@/lib/api';
import type { PairingUser } from '@/lib/api';

export interface Profile { instructions: string; timezone: string }
export interface Topic { id: string; name?: string; mode: 'inherit' | 'all' | 'silent' }
export interface Group { id: string; name: string; mode: 'mention' | 'all'; instructions: string; topics: Topic[] }
export interface Model { id: string; efforts: string[] }
export interface Settings {
  profile: Profile;
  chat: { model: string; effort: string };
  memory: { llm_model: string; llm_reasoning_effort: string; reflect_llm_reasoning_effort: string };
  memory_status: { state: string };
  models: Model[];
  access: { notice?: string; groups: Group[]; people: PairingUser[]; pending: PairingUser[] };
  keys: Record<string, boolean>;
  admin: { username: string; available: boolean };
}
export const readSettings = () => fetchJSON<Settings>('/api/settings');
export function writeSettings<T = { ok: boolean }>(path: string, body: unknown, method = 'PUT') {
  return fetchJSON<T>(`/api/settings/${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
