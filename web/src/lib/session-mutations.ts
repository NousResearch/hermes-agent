import { createSessionMutationClient, type SessionMutationSnapshot } from '../../../apps/shared/src/session-http-mutations';

export function createDashboardSessionMutations(fetchJSON: <T>(path: string, init?: RequestInit) => Promise<T>) {
  const mutate = createSessionMutationClient();
  return <T>(id: string, method: 'PATCH' | 'DELETE' | 'POST', payload: Record<string, unknown>, profile: string): Promise<T> => {
    const path = `/api/sessions/${encodeURIComponent(id)}`;
    const query = new URLSearchParams(profile ? { profile } : {});
    const suffix = query.size ? `?${query}` : '';
    // Clone before awaiting the snapshot: a file-selection/UI change cannot
    // alter the import whose stable request identity is already being prepared.
    const frozen = JSON.parse(JSON.stringify(payload)) as Record<string, unknown>;
    const key = JSON.stringify([profile, id, method, frozen]);
    return mutate(key,
      () => fetchJSON<SessionMutationSnapshot>(`${path}/mutation-snapshot${suffix}`),
      identity => {
        if (method === 'DELETE') {
          const params = new URLSearchParams(query);
          for (const [name, value] of Object.entries(identity)) { params.set(name, String(value)); }
          return fetchJSON<T>(`${path}?${params}`, { method });
        }
        const { expected_generation: generation, ...baseIdentity } = identity;
        return fetchJSON<T>(method === 'POST' ? '/api/sessions/import' : path, { method,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...frozen, ...baseIdentity,
            ...(method === 'PATCH' ? { expected_generation: generation } : {}), profile: profile || undefined }) });
      }, method === 'POST');
  };
}
