import { expect, test } from '@playwright/test';

test('native trackpad caret is visible at the selected cell without sending input', async ({ page }) => {
  await page.goto('/browser-tests/terminal.html');
  await page.waitForFunction(() => 'terminal' in window);
  await page.keyboard.type('hello');
  await page.evaluate(async () => {
    const state = window as unknown as { terminal: { write(data: string, cb: () => void): void } };
    // Real PTY output is asynchronous; the terminal fixture does not echo input.
    await new Promise<void>(resolve => state.terminal.write('> hello', resolve));
    document.querySelector('textarea')!.setSelectionRange(2, 2);
    document.dispatchEvent(new Event('selectionchange'));
  });
  const caret = page.locator('.pty-native-caret');
  await expect(caret).toBeVisible();
  const position = await caret.evaluate(el => {
    const screen = document.querySelector('.xterm-screen')!.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    return (rect.left - screen.left) / (screen.width / 80);
  });
  expect(position).toBeCloseTo(4, 1); // prompt + native offset, not the terminal tail (7)
  expect(await page.evaluate(() => (window as unknown as { frames: string[] }).frames.join(''))).toBe('hello');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'X' }));
    t.value = 'heXllo';
    t.setSelectionRange(3, 3);
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'X' }));
  });
  expect(await page.evaluate(() => (window as unknown as { frames: string[] }).frames.join(''))).toBe('hello\x7f\x7f\x7fXllo');
  // Stale rendered output must not keep a cursor at an unverified position.
  await expect(caret).not.toBeVisible();
});

test('wide cells and wrapped rows retain a visual-only caret and lifecycle restores the terminal cursor', async ({ page }) => {
  await page.goto('/browser-tests/terminal.html');
  await page.waitForFunction(() => 'terminal' in window);
  for (const boundary of ['tail', 'blur', 'reset', 'paste', 'composition', 'dispose']) {
    await page.evaluate(async () => {
      const state = window as unknown as { input: { reset(): void }; terminal: { reset(): void; resize(cols: number, rows: number): void; focus(): void; write(data: string, cb: () => void): void } };
      state.input.reset();
      state.terminal.reset();
      state.terminal.resize(10, 5);
      state.terminal.focus();
      const t = document.querySelector('textarea')!;
      const data = 'abcde界fghij';
      t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data }));
      t.value = data;
      t.setSelectionRange(data.length, data.length);
      t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data }));
      await new Promise<void>(resolve => state.terminal.write('> ' + data, resolve));
      t.setSelectionRange(6, 6);
      document.dispatchEvent(new Event('selectionchange'));
    });
    const caret = page.locator('.pty-native-caret');
    await expect(caret).toBeVisible();
    const position = await caret.evaluate(el => {
      const screen = document.querySelector('.xterm-screen')!.getBoundingClientRect();
      const rect = el.getBoundingClientRect();
      return [(rect.left - screen.left) / (screen.width / 10), (rect.top - screen.top) / (screen.height / 5)];
    });
    expect(position[0]).toBeCloseTo(9, 1);
    expect(position[1]).toBeCloseTo(0, 1);
    await page.evaluate(boundary => {
      const t = document.querySelector('textarea')!;
      const state = window as unknown as { input: { reset(): void; paste(data: string): void; dispose(): void } };
      if (boundary === 'tail') { t.setSelectionRange(t.value.length, t.value.length); document.dispatchEvent(new Event('selectionchange')); }
      if (boundary === 'blur') t.blur();
      if (boundary === 'reset') state.input.reset();
      if (boundary === 'paste') state.input.paste('P');
      if (boundary === 'composition') t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
      if (boundary === 'dispose') state.input.dispose();
    }, boundary);
    await expect(caret).not.toBeVisible();
    expect(await page.evaluate(() => (window as unknown as { terminal: { options: { theme: { cursor?: string } } } }).terminal.options.theme.cursor)).not.toBe('#00000000');
  }
});
