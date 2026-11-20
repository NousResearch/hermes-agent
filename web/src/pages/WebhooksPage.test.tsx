// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { I18nProvider } from '@/i18n';
import { ProfileContext } from '@/contexts/profile-context';
import { PageHeaderProvider } from '@/contexts/PageHeaderProvider';
import WebhooksPage from './WebhooksPage';

const mocks = vi.hoisted(() => ({ getWebhooks: vi.fn(), createWebhook: vi.fn() }));
vi.mock('@/lib/api', () => ({ api: mocks }));

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  Element.prototype.scrollIntoView = vi.fn();
  Element.prototype.hasPointerCapture = vi.fn(() => false);
  Element.prototype.releasePointerCapture = vi.fn();
  mocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [] });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); });

it.each(['bot-chat', 'bot-chat:recipient'])('creates and reads back %s without changing the execution profile', async target => {
  mocks.createWebhook.mockImplementation(async body => {
    const route = { ...body, profile: 'producer', enabled: true, skills: [], events: [],
      url: 'http://localhost:8644/p/producer/webhooks/fixture' };
    mocks.getWebhooks.mockResolvedValue({ enabled: true, subscriptions: [route] });
    return { ...route, secret: 'one-time-secret' };
  });
  render(<I18nProvider><MemoryRouter><ProfileContext.Provider value={{ profile: 'producer',
    currentProfile: 'producer', profiles: ['default', 'producer', 'recipient'], setProfile: () => {} }}>
    <PageHeaderProvider pluginTabs={[]}><WebhooksPage /></PageHeaderProvider>
  </ProfileContext.Provider></MemoryRouter></I18nProvider>);
  fireEvent.click(await screen.findByRole('button', { name: /new subscription/i }));
  fireEvent.change(screen.getByLabelText(/^name$/i), { target: { value: 'fixture' } });
  fireEvent.click(screen.getByRole('combobox'));
  fireEvent.click(await screen.findByRole('option', { name: 'Bot Chat' }));
  if (target.includes(':')) {
    fireEvent.click(screen.getAllByRole('combobox')[1]);
    fireEvent.click(await screen.findByRole('option', { name: 'recipient' }));
  }
  fireEvent.click(screen.getByLabelText('Deliver only'));
  expect(screen.getByText(/receiving Bot Chat may run its model/i)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: /^create$/i }));
  await waitFor(() => expect(mocks.createWebhook).toHaveBeenCalledWith(expect.objectContaining({
    deliver: target, deliver_only: true
  })));
  fireEvent.click(await screen.findByRole('button', { name: 'Done' }));
  expect(await screen.findByText(target)).toBeTruthy();
  expect(screen.getByText(/Execution profile: producer/)).toBeTruthy();
  expect(screen.getByText('http://localhost:8644/p/producer/webhooks/fixture')).toBeTruthy();
});
