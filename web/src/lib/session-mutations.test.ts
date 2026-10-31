import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api, setManagementProfile } from './api';

beforeEach(() => {
  vi.stubGlobal('window', { __HERMES_SESSION_TOKEN__: 'fixture' });
  setManagementProfile('work');
});
afterEach(() => { vi.unstubAllGlobals(); setManagementProfile(''); });

it('prepares import and delete from snapshots and retries unchanged after a lost reply', async () => {
  const writes: Array<{ path: string; init: RequestInit }> = [];
  let lost = true;
  vi.stubGlobal('fetch', vi.fn(async (path: string, init: RequestInit) => {
    if (!init.method) {
      const absent = path.includes('new-import');
      return Response.json({ exists: !absent, runtime_revision: absent ? 0 : 27, runtime_generation: absent ? null : 6 });
    }
    writes.push({ path, init });
    if (lost) { lost = false; throw new TypeError('Failed to fetch'); }
    return Response.json({ ok: true });
  }));
  await expect(api.deleteSession('existing')).rejects.toThrow('cannot reach');
  await api.deleteSession('existing');
  expect(writes[1]).toEqual(writes[0]);
  const query = new URL(writes[0].path, 'http://localhost').searchParams;
  expect(query.get('expected_revision')).toBe('27');
  expect(query.get('expected_generation')).toBe('6');
  expect(query.get('request_id')).toBeTruthy();
  await api.importSessions([{ id: 'new-import', messages: [] }]);
  expect(JSON.parse(writes[2].init.body as string)).toMatchObject({ expected_revision: 0,
    request_id: expect.any(String), profile: 'work' });
});

it('surfaces revision conflicts and never substitutes missing counters', async () => {
  const fetcher = vi.fn(async (_path: string, init: RequestInit) => {
    if (!init.method) { return Response.json({ exists: true, runtime_revision: 11, runtime_generation: 3 }); }
    expect(JSON.parse(init.body as string)).toMatchObject({ expected_revision: 11, expected_generation: 3 });
    return Response.json({ detail: 'revision_conflict' }, { status: 409 });
  });
  vi.stubGlobal('fetch', fetcher);
  await expect(api.renameSession('rename', 'title')).rejects.toThrow('revision_conflict');
  expect(fetcher).toHaveBeenCalledTimes(2);
  fetcher.mockReset().mockResolvedValue(Response.json({ exists: true }));
  await expect(api.deleteSession('no-snapshot')).rejects.toThrow(/snapshot/);
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it('sends mutations when crypto.randomUUID is unavailable on an insecure origin', async () => {
  const fetcher = vi.fn(async (_path: string, init: RequestInit) => {
    if (!init.method) { return Response.json({ exists: true, runtime_revision: 12, runtime_generation: 4 }); }
    return Response.json({ ok: true });
  });
  vi.stubGlobal('fetch', fetcher);
  vi.stubGlobal('crypto', {
    getRandomValues: (bytes: Uint8Array) => bytes.fill(0xab),
  });

  await api.deleteSession('insecure-origin');

  const request = new URL(fetcher.mock.calls[1][0] as string, 'http://localhost');
  expect(request.searchParams.get('request_id')).toBe('abababab-abab-4bab-abab-abababababab');
});
