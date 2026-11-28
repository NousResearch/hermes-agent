import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { describe, expect, it, vi } from 'vitest';

interface Event {
  target?: { value?: string; checked?: boolean } | object;
  currentTarget?: object;
  button?: number;
  key?: string;
  preventDefault?: () => void;
}
interface ElementProps {
  value: string;
  placeholder: string;
  type: string;
  role: string;
  disabled: boolean;
  onChange: (event: Event) => void;
  onValueChange: (value: string) => void;
  onSubmit: (event: Event) => Promise<void>;
  onClick: (event: Event) => void;
  onKeyDown: (event: Event) => void;
  onPointerDownCapture?: (event: Event) => void;
  onPointerUpCapture?: (event: Event) => void;
}
interface Element {
  type: unknown;
  props: ElementProps;
  children: unknown[];
}

// Execute the shipped no-build plugin. Only expose the component at the IIFE
// boundary; assertions exercise its event handlers and state, not source text.
function host(storage = new Map<string, string>(), blockedStorage = false) {
  let states: unknown[] = [];
  let cursor = 0;
  let effects: (() => void)[] = [];
  const useState = <T,>(initial: T | (() => T)) => {
    const i = cursor++;
    if (!(i in states)) states[i] = typeof initial === 'function' ? (initial as () => T)() : initial;
    return [states[i] as T, (value: T | ((previous: T) => T)) => {
      states[i] = typeof value === 'function' ? (value as (previous: T) => T)(states[i] as T) : value;
    }] as const;
  };
  const context = vm.createContext({
    URLSearchParams,
    window: {
      location: { pathname: '/kanban', search: '?profile=default' },
      sessionStorage: {
        getItem(key: string) { if (blockedStorage) throw Error('blocked'); return storage.get(key) ?? null; },
        setItem(key: string, value: string) { if (blockedStorage) throw Error('quota'); storage.set(key, value); },
        removeItem(key: string) { if (blockedStorage) throw Error('blocked'); storage.delete(key); },
      },
      __HERMES_PLUGIN_SDK__: {
        React: {
          Component: class {},
          createElement: (type: unknown, props: ElementProps, ...children: unknown[]) => ({ type, props: props || {}, children }),
        },
        components: Object.fromEntries(['Card', 'CardContent', 'Badge', 'Button', 'Input', 'Label', 'Select', 'SelectOption'].map(name => [name, name])),
        hooks: {
          useState,
          useRef: <T,>(initial: T) => useState(() => ({ current: initial }))[0],
          useEffect: (effect: () => void) => effects.push(effect),
        },
        utils: { cn: (...args: string[]) => args.join(' ') },
      },
    },
  });
  const source = readFileSync(new URL('../../../plugins/kanban/dashboard/dist/index.js', import.meta.url), 'utf8');
  vm.runInContext(source.replace(/\}\)\(\);\s*$/, 'globalThis.Create = InlineCreate; })();'), context);
  const defaults = { columnName: 'triage', boardSlug: 'board-a', onCancel: vi.fn(), onSubmit: vi.fn().mockResolvedValue({ id: 'created' }) };
  return {
    storage, context, defaults,
    mount() { states = []; },
    render(overrides = {}): Element {
      cursor = 0; effects = [];
      const tree = context.Create({ ...defaults, ...overrides });
      effects.forEach(effect => effect());
      return tree;
    },
  };
}
function find(tree: Element, predicate: (el: Element) => boolean): Element {
  if (predicate(tree)) return tree;
  for (const child of tree.children.flat(Infinity) as Element[]) {
    if (!child || typeof child !== 'object') continue;
    try { return find(child, predicate); } catch { /* continue searching */ }
  }
  throw Error('Element not found');
}
const title = (tree: Element) => find(tree, el => el.type === 'textarea');
const form = (tree: Element) => find(tree, el => el.type === 'form');
const submit = (tree: Element) => form(tree).props.onSubmit({ preventDefault() {} });

// React effects and handlers run through a minimal deterministic host; live
// compositor-level pointer behavior is additionally checked in Chrome.
describe('Kanban new-task draft', () => {
  it('dismisses only when both press and release are on the backdrop', () => {
    const h = host();
    const tree = h.render();
    const outside = {};
    const inside = {};
    for (const [down, up, closes] of [[inside, outside, false], [outside, inside, false], [outside, outside, true]] as const) {
      const event = (target: object) => ({ target, currentTarget: outside, button: 0 });
      tree.props.onPointerDownCapture?.(event(down));
      tree.props.onPointerUpCapture?.(event(up));
      tree.props.onClick(event(outside));
      expect(h.defaults.onCancel).toHaveBeenCalledTimes(closes ? 1 : 0);
    }
  });

  it('restores all fields after cancel/remount and a fresh page context, isolated by board and profile', () => {
    const h = host();
    let tree = h.render();
    title(tree).props.onChange({ target: { value: 'Long draft\nwith a second line' } });
    find(tree, el => el.props.placeholder === 'specifier').props.onChange({ target: { value: 'worker' } });
    find(tree, el => el.props.placeholder === 'pri').props.onChange({ target: { value: '7' } });
    find(tree, el => el.type === 'Input' && String(el.props.placeholder).startsWith('skills (')).props.onChange({ target: { value: 'github, research' } });
    find(tree, el => el.type === 'Select' && el.props.value === 'scratch').props.onValueChange('dir');
    find(tree, el => el.type === 'input' && el.props.type === 'checkbox').props.onChange({ target: { checked: true } });
    tree = h.render();
    find(tree, el => el.type === 'Input' && String(el.props.placeholder).startsWith('workspace path')).props.onChange({ target: { value: '/project' } });
    find(tree, el => el.props.placeholder === 'max turns (default 20)').props.onChange({ target: { value: '12' } });
    tree = h.render();
    tree.props.onKeyDown({ key: 'Escape' });
    h.mount();
    expect(title(h.render()).props.value).toBe('Long draft\nwith a second line');
    const fresh = host(h.storage);
    tree = fresh.render();
    expect(title(tree).props.value).toBe('Long draft\nwith a second line');
    expect(find(tree, el => el.props.placeholder === 'specifier').props.value).toBe('worker');
    expect(find(tree, el => el.props.placeholder === 'pri').props.value).toBe('7');
    expect(find(tree, el => el.props.placeholder === 'max turns (default 20)').props.value).toBe('12');
    expect(find(tree, el => el.type === 'Input' && el.props.value === '/project')).toBeTruthy();
    fresh.mount();
    expect(title(fresh.render({ boardSlug: 'board-b' })).props.value).toBe('');
    fresh.context.window.location.search = '?profile=other';
    fresh.mount();
    expect(title(fresh.render()).props.value).toBe('');
  });

  it('keeps the draft on failed submission and clears it only after success', async () => {
    const h = host();
    title(h.render()).props.onChange({ target: { value: 'Keep me' } });
    const fail = vi.fn().mockRejectedValue(new Error('Service unavailable'));
    await submit(h.render({ onSubmit: fail }));
    expect(find(h.render(), el => el.props.role === 'alert' && el.children.includes('Service unavailable'))).toBeTruthy();
    h.mount();
    let tree = h.render();
    expect(title(tree).props.value).toBe('Keep me');
    await submit(tree);
    h.mount();
    tree = h.render();
    expect(title(tree).props.value).toBe('');
    expect(h.defaults.onSubmit).toHaveBeenCalledWith(expect.objectContaining({ title: 'Keep me' }));
  });

  it('blocks duplicate submissions and dismissal while a create is pending', async () => {
    const h = host();
    title(h.render()).props.onChange({ target: { value: 'One task only' } });
    let resolve!: (value: object) => void;
    const pending = vi.fn(() => new Promise<object>(done => { resolve = done; }));
    let tree = h.render({ onSubmit: pending });
    const first = submit(tree);
    await submit(tree);
    tree = h.render({ onSubmit: pending });
    expect(find(tree, el => el.type === 'fieldset').props.disabled).toBe(true);
    tree.props.onKeyDown({ key: 'Escape' });
    expect(h.defaults.onCancel).not.toHaveBeenCalled();
    expect(pending).toHaveBeenCalledTimes(1);
    resolve({ id: 'created' });
    await first;
    expect(h.defaults.onCancel).toHaveBeenCalledTimes(1);
  });

  it('discards explicitly and tolerates malformed stored drafts', () => {
    const h = host();
    title(h.render()).props.onChange({ target: { value: 'Discard me' } });
    const tree = h.render();
    find(tree, el => el.type === 'Button' && el.children.includes('Discard')).props.onClick({});
    h.mount();
    expect(title(h.render()).props.value).toBe('');
    const invalid = new Map(h.storage);
    for (const key of invalid.keys()) invalid.set(key, '{invalid JSON');
    expect(title(host(invalid).render()).props.value).toBe('');
    for (const key of invalid.keys()) invalid.set(key, JSON.stringify({ title: {}, skills: 3, workspaceKind: 'invalid' }));
    expect(title(host(invalid).render()).props.value).toBe('');
  });

  it('preserves a memory fallback when session storage is blocked', () => {
    const h = host(new Map(), true);
    title(h.render()).props.onChange({ target: { value: 'Private mode draft' } });
    h.render(); h.mount();
    expect(title(h.render()).props.value).toBe('Private mode draft');
  });
});
