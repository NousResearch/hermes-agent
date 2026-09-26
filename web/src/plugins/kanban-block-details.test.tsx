// @vitest-environment jsdom
import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';

const task = {
  id: 'blocked-example', title: 'Investigate failure', status: 'blocked',
  assignee: 'worker', workspace_kind: 'scratch', priority: 0,
  block_kind: 'needs_input', block_recurrences: 3, consecutive_failures: 2,
  last_failure_error: '<script>fixture failure</script>',
};

async function openBoard(value: Record<string, unknown> = task) {
  let Page: React.ComponentType;
  const fetchJSON = vi.fn(async (url: string) => {
    const path = new URL(url, 'http://localhost').pathname;
    if (path.endsWith('/board')) return { columns: [{ name: 'blocked', tasks: [value] }], assignees: ['worker'], tenants: [] };
    if (path.endsWith('/boards')) return { boards: [] };
    if (path.endsWith('/tasks/blocked-example')) return { task: value, comments: [], events: [], runs: [] };
    if (path.endsWith('/profiles')) return { profiles: [] };
    if (path.endsWith('/tasks')) return { tasks: [value] };
    return {};
  });
  Object.assign(window, {
    __HERMES_PLUGIN_SDK__: {
      React, hooks: React,
      components: { Card: 'div', CardContent: 'div', Badge: 'span', Button: 'button', Input: 'input', Label: 'label', Select: 'select', SelectOption: 'option' },
      utils: { cn: (...parts: string[]) => parts.filter(Boolean).join(' '), timeAgo: () => '' },
      fetchJSON, buildWsUrl: () => Promise.resolve('ws://localhost/fixture'),
    },
    __HERMES_PLUGINS__: { register: (_name: string, component: React.ComponentType) => { Page = component; } },
  });
  vi.stubGlobal('WebSocket', class { close() {} });
  vi.resetModules();
  // Execute the shipped IIFE through its real SDK registration, without source extraction.
  await import('../../../plugins/kanban/dashboard/dist/index.js');
  render(React.createElement(Page!));
  return screen.findByRole('button', { name: `Investigate failure — blocked-example — ${value.status}` });
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear(); });

it('does not label a retained kind as a current block after recovery', async () => {
  const card = await openBoard({ ...task, status: 'ready', block_recurrences: 0, consecutive_failures: 0 });
  expect(card.textContent).not.toContain('needs_input');
  fireEvent.click(card);
  expect(await screen.findByText('Status')).toBeTruthy();
  expect(screen.queryByText('Block kind')).toBeNull();
  expect(screen.queryByText('Block recurrences')).toBeNull();
  expect(screen.queryByText('Consecutive failures')).toBeNull();
  // The backend can retain historical failures; label them as historical, not a current block.
  expect(screen.getByText('Last failure')).toBeTruthy();
});

it('shows the current block kind on the card and existing failure fields in its drawer', async () => {
  const card = await openBoard();
  expect(card.textContent).toContain('needs_input');
  fireEvent.click(card);
  expect((await screen.findByText('Block recurrences')).parentElement?.textContent).toBe('Block recurrences3');
  expect(screen.getByText('Consecutive failures').parentElement?.textContent).toBe('Consecutive failures2');
  expect(screen.getByText(task.last_failure_error)).toBeTruthy();
  expect(document.querySelector('script')).toBeNull();
});

it('keeps legacy tasks without block metadata usable', async () => {
  const { block_kind, block_recurrences, consecutive_failures, last_failure_error, ...legacy } = task;
  const card = await openBoard(legacy);
  fireEvent.click(card);
  expect(await screen.findByText('Status')).toBeTruthy();
  expect(screen.queryByText('Block recurrences')).toBeNull();
  expect(screen.queryByText('Last failure')).toBeNull();
});
