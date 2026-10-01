// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Settings } from './api';
import SettingsApp from './SettingsApp';
import { getManagementProfile } from '@/lib/api';

const mocks = vi.hoisted(() => ({ fetchJSON: vi.fn(), messaging: vi.fn(), oauth: vi.fn(), restart: vi.fn(), model: vi.fn(), config: vi.fn() }));
vi.mock('@/lib/api', async importOriginal => ({
  ...await importOriginal<typeof import('@/lib/api')>(),
  HERMES_BASE_PATH: '', fetchJSON: mocks.fetchJSON,
  api: { getMessagingPlatforms: mocks.messaging, getOAuthProviders: mocks.oauth,
    getProfiles: vi.fn().mockResolvedValue({ profiles: [{ name: 'default' }, { name: 'work' }] }),
    getActiveProfile: vi.fn().mockResolvedValue({ current: 'default', active: 'default' }),
    restartGateway: mocks.restart, setModelAssignment: mocks.model, saveConfig: mocks.config },
}));
let root: Root;
let container: HTMLDivElement;
let data: Settings;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.useFakeTimers(); vi.clearAllMocks();
  data = {
    profile: { instructions: 'Brief replies', timezone: 'UTC' },
    chat: { model: 'gpt-5.6-luna', effort: 'medium' },
    memory: { llm_model: 'gpt-5.6-luna', llm_reasoning_effort: 'low', reflect_llm_reasoning_effort: 'medium' },
    memory_status: { state: 'waiting' },
    models: [{ id: 'gpt-5.6-luna', efforts: ['low', 'medium', 'high'] }],
    access: { people: [], pending: [], groups: [{ id: '-100', name: 'Team', mode: 'mention', instructions: '', topics: [{ id: '5', name: 'Ops', mode: 'inherit' }] }] },
    keys: {}, admin: { username: 'admin', available: true },
  };
  mocks.fetchJSON.mockImplementation(async (url: string) => url.startsWith('/api/settings/models') ? data.models : data);
  mocks.messaging.mockResolvedValue({ platforms: [{ id: 'telegram', state: 'disconnected' }] });
  mocks.oauth.mockResolvedValue({ providers: [] });
  mocks.restart.mockResolvedValue({ ok: true });
  mocks.model.mockResolvedValue({ ok: true });
  mocks.config.mockResolvedValue({ ok: true });
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.useRealTimers(); });
async function render(path: string) { await act(async () => root.render(<MemoryRouter initialEntries={[path]}><SettingsApp /></MemoryRouter>)); }
async function click(element: Element | null) { if (!element) throw new Error('Missing control'); await act(async () => element.dispatchEvent(new MouseEvent('click', { bubbles: true }))); }
async function fill(element: HTMLInputElement | HTMLTextAreaElement, value: string) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype, 'value')!.set!.call(element, value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

it('saves native instructions without writing the timezone', async () => {
  await render('/settings');
  await fill(container.querySelector('textarea')!, 'New instructions');
  await click([...container.querySelectorAll('button')].find(button => button.textContent === 'Save')!);
  const call = mocks.fetchJSON.mock.calls.find(([url]) => url === '/api/settings/profile');
  expect(JSON.parse(call![1].body)).toEqual({ instructions: 'New instructions' });
});

it('changes just one topic, then requests the native gateway restart', async () => {
  await render('/settings/access');
  await click(container.querySelector('[aria-label="Expand Team"]'));
  expect(container.textContent).not.toContain('Add topic');
  await click([...container.querySelectorAll('[aria-label="Replies in topic 5"] button')].find(button => button.textContent === 'Silent')!);
  const call = mocks.fetchJSON.mock.calls.find(([url]) => url === '/api/settings/groups/-100');
  expect(JSON.parse(call![1].body)).toMatchObject({ mode: 'mention', topics: [{ id: '5', mode: 'silent' }] });
  expect(mocks.restart).toHaveBeenCalledOnce();
});

it('retains the draft key and the failure message after a background refresh', async () => {
  mocks.fetchJSON.mockImplementation(async (url: string) => {
    if (url.startsWith('/api/settings/keys/')) throw new Error('Previous key was kept');
    return data;
  });
  await render('/settings/keys');
  await fill(container.querySelector('#OPENROUTER_API_KEY')!, 'test-key');
  await click([...container.querySelectorAll('button')].find(button => button.textContent === 'Check and save' && !button.disabled)!);
  await act(async () => vi.advanceTimersByTime(10000));
  expect(container.querySelector('[role="alert"]')?.textContent).toContain('Previous key was kept');
  expect((container.querySelector('#OPENROUTER_API_KEY') as HTMLInputElement).value).toBe('test-key');
  expect(mocks.restart).not.toHaveBeenCalled();
});

it('persists chat effort using the native config endpoint', async () => {
  await render('/settings/models');
  await click([...container.querySelectorAll('[aria-label="Chat reasoning"] button')].find(button => button.textContent === 'High')!);
  expect(mocks.model).toHaveBeenCalledWith(expect.objectContaining({ scope: 'main', provider: 'openai-codex', model: 'gpt-5.6-luna' }));
  expect(mocks.config).toHaveBeenCalledWith({ agent: { reasoning_overrides: { 'gpt-5.6-luna': 'high' } } });
});


it('requests the native gateway restart when a timezone change requires it', async () => {
  mocks.fetchJSON.mockImplementation(async (url: string) => url === '/api/settings/profile' ? { restart: true } : data);
  await render('/settings');
  const select = container.querySelector('select')!;
  await act(async () => { select.value = 'Europe/Prague'; select.dispatchEvent(new Event('change', { bubbles: true })); });
  expect(mocks.restart).toHaveBeenCalledOnce();
});


it('refreshes untouched instructions but preserves local edits', async () => {
  await render('/settings');
  data = { ...data, profile: { ...data.profile, instructions: 'Updated elsewhere' } };
  await act(async () => vi.advanceTimersByTime(10000));
  expect(container.querySelector('textarea')!.value).toBe('Updated elsewhere');
  expect([...container.querySelectorAll('button')].find(button => button.textContent === 'Save')!.disabled).toBe(true);
  await fill(container.querySelector('textarea')!, 'Unsaved local edit');
  data = { ...data, profile: { ...data.profile, instructions: 'Another server update' } };
  await act(async () => vi.advanceTimersByTime(10000));
  expect(container.querySelector('textarea')!.value).toBe('Unsaved local edit');
});


it.each(['/settings?profile=work', '/?profile=work'])('preserves profile scope from %s through navigation and writes', async path => {
  const scopes: string[] = [];
  mocks.fetchJSON.mockImplementation(async () => { scopes.push(getManagementProfile()); return data; });
  await render(path);
  await click(container.querySelector('a[href="/settings/access"]'));
  await click([...container.querySelectorAll('[aria-label="Replies in Team"] button')].find(button => button.textContent === 'Every message')!);
  expect(mocks.fetchJSON.mock.calls.some(([url]) => url === '/api/settings/groups/-100')).toBe(true);
  expect(scopes.length).toBeGreaterThan(1);
  expect(new Set(scopes)).toEqual(new Set(['work']));
});


it('disables memory edits for a profile without the supervised service', async () => {
  data.memory_status = { state: 'unmanaged' };
  await render('/settings/models');
  expect((container.querySelector('[aria-label="Memory model"]') as HTMLSelectElement).disabled).toBe(true);
  expect([...container.querySelectorAll<HTMLButtonElement>('[aria-label="Learning effort"] button')].every(button => button.disabled)).toBe(true);
  expect(container.textContent).toContain('Memory settings are managed from the profile');
});


it('keeps the server default timezone when saving instructions', async () => {
  data.profile.timezone = '';
  await render('/settings');
  expect(container.querySelector('select')!.value).toBe('');
  await fill(container.querySelector('textarea')!, 'New instructions');
  await click([...container.querySelectorAll('button')].find(button => button.textContent === 'Save')!);
  const call = mocks.fetchJSON.mock.calls.find(([url]) => url === '/api/settings/profile');
  expect(JSON.parse(call![1].body)).toEqual({ instructions: 'New instructions' });
});


it('restarts the native gateway after rotating a browser key', async () => {
  await render('/settings/keys');
  const input = container.querySelector<HTMLInputElement>('#BROWSER_USE_API_KEY')!;
  await fill(input, 'replacement');
  await click(input.closest('.sec')!.querySelector('button'));
  expect(mocks.restart).toHaveBeenCalledOnce();
  expect(input.value).toBe('');
});
