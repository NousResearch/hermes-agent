// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router';
import { expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  createProfile: vi.fn(),
  showToast: vi.fn(),
}));
vi.mock('@/lib/api', () => ({ api: {
  createProfile: mocks.createProfile,
  getProfiles: async () => ({ profiles: [] }),
  getActiveProfile: async () => null,
  getModelOptions: async () => ({ providers: [{ slug: 'test-provider', name: 'Test provider', models: ['test-model'] }] }),
} }));
vi.mock('@/contexts/useProfileScope', () => ({ useProfileScope: () => ({ setProfile: vi.fn() }) }));
import { PageHeaderProvider } from '@/contexts/PageHeaderProvider';
vi.mock('@nous-research/ui/hooks/use-toast', () => ({ useToast: () => ({ toast: null, showToast: mocks.showToast }) }));
import ProfilesPage from './ProfilesPage';

it.each([{ providers: [], modelFailed: false }, { providers: ['honcho', 'openviking'], modelFailed: false }, { providers: ['honcho'], modelFailed: true }])('reports clone outcome without hiding successful publication ($providers, model failure: $modelFailed)', async ({ providers, modelFailed }) => {
  (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.matchMedia = vi.fn().mockReturnValue({ matches: true });
  mocks.showToast.mockClear();
  mocks.createProfile.mockResolvedValue({ name: 'copy', ok: true, clone_needs_auth: providers, model_set: !modelFailed });
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  const click = (text: string) => {
    const button = [...(document.querySelector('[role="dialog"]') || document).querySelectorAll('button')].find(el => el.textContent?.trim() === text);
    expect(button).toBeTruthy();
    button!.click();
  };
  try {
    await act(async () => { root.render(<MemoryRouter initialEntries={['/profiles']}><PageHeaderProvider pluginTabs={[]}><ProfilesPage /></PageHeaderProvider></MemoryRouter>); });
    await act(async () => { click('Create'); });
    const input = document.querySelector<HTMLInputElement>('#profile-name')!;
    expect(input).toBeTruthy();
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, 'copy');
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    if (modelFailed) {
      await act(async () => { document.querySelector<HTMLButtonElement>('#profile-model button')!.click(); });
      const option = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(el => el.textContent === 'Test provider · test-model');
      expect(option).toBeTruthy();
      await act(async () => { option!.click(); });
    }
    await act(async () => { click('Create'); });
    expect(mocks.createProfile).toHaveBeenCalledWith(expect.objectContaining({ name: 'copy', clone_from: 'default' }));
    const message = mocks.showToast.mock.calls.at(-1)?.[0];
    expect(message).toContain('copy');
    if (modelFailed) {
      expect(mocks.createProfile).toHaveBeenLastCalledWith(expect.objectContaining({ provider: 'test-provider', model: 'test-model' }));
      expect(message).toMatch(/model could not be saved/);
    }
    if (providers.length) expect(message).toContain(providers.join(', '));
    else expect(message).not.toMatch(/sign in again/i);
  } finally {
    await act(async () => root.unmount());
    container.remove();
  }
});
