import { expect, test, type Page } from '@playwright/test';

async function edit(page: Page, value: string, start: number, end: number, data: string | null, inputType = 'insertText') {
  await page.evaluate(({ value, start, end, data, inputType }) => {
    const textarea = document.querySelector('textarea')!;
    textarea.setSelectionRange(start, end);
    textarea.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType, data }));
    textarea.value = value;
    textarea.setSelectionRange(value.length, value.length);
    textarea.dispatchEvent(new InputEvent('input', { bubbles: true, inputType, data }));
  }, { value, start, end, data, inputType });
}

async function bytes(page: Page) {
  return page.evaluate(() => (window as unknown as { frames: string[] }).frames.join(''));
}

test.beforeEach(async ({ page }) => {
  await page.goto('/browser-tests/terminal.html');
  await page.waitForFunction(() => 'terminal' in window);
});

test('programmatic paste settles active composition before xterm emits', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: 'hello' }));
    t.value = 'hello';
    (window as unknown as { input: { paste(data: string): void } }).input.paste('P');
  });
  expect(await bytes(page)).toBe('helloP');
});

test('native copy shortcut preserves selection and unselected Ctrl+C interrupts', async ({ page }) => {
  await page.keyboard.type('hello');
  await page.evaluate(() => document.querySelector('textarea')!.setSelectionRange(1, 4));
  await page.keyboard.press('Control+c');
  expect(await bytes(page)).toBe('hello');
  expect(await page.locator('textarea').evaluate((t: HTMLTextAreaElement) => [t.selectionStart, t.selectionEnd])).toEqual([1, 4]);
  await page.keyboard.press('Meta+c');
  expect(await bytes(page)).toBe('hello');
  expect(await page.locator('textarea').inputValue()).toBe('hello');
  await page.evaluate(() => document.querySelector('textarea')!.setSelectionRange(5, 5));
  await page.keyboard.press('Control+c');
  expect(await bytes(page)).toBe('hello\x03');
});

test('keyless line break flushes the older printable FIFO', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertLineBreak' }));
    t.value += '\n';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertLineBreak' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
  });
  expect(await bytes(page)).toBe('a\r');
});

for (const inputType of ['insertLineBreak', 'insertParagraph']) {
  test(`paste settles pending ${inputType} before its delayed native observation`, async ({ page }) => {
    for (const composing of [false, true]) {
      const result = await page.evaluate(({ inputType, composing }) => {
        const state = window as unknown as { input: { reset(): void; paste(data: string): void }; frames: string[] };
        state.input.reset();
        state.frames.length = 0;
        const t = document.querySelector('textarea')!;
        if (composing) {
          t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
          t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '好' }));
          t.value = '好';
        }
        t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType, data: null }));
        state.input.paste('P');
        const beforeNative = state.frames.join('');
        t.value += '\n';
        t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType, data: null }));
        const afterNative = state.frames.join('');
        // The claim is one-shot: an independent input-only Return still emits.
        t.value = '\n';
        t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType, data: null }));
        return { beforeNative, afterNative, afterNext: state.frames.join('') };
      }, { inputType, composing });
      const expected = composing ? '好\rP' : '\rP';
      expect(result).toEqual({ beforeNative: expected, afterNative: expected, afterNext: expected + '\r' });
    }
  });
}

for (const trailing of [false, true]) {
  test(`programmatic paste claims only its overtaken input (trailing=${trailing})`, async ({ page }) => {
    await page.evaluate(trailing => {
      const t = document.querySelector('textarea')!;
      t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'a' }));
      (window as unknown as { input: { paste(data: string): void } }).input.paste('P');
      if (trailing) {
        t.value = 'a';
        t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'a' }));
      }
      t.value = 'b';
      t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'b' }));
    }, trailing);
    expect(await bytes(page)).toBe('aPb');
  });
}

test('input-only insertion cannot erase acknowledged text without pre-edit evidence', async ({ page }) => {
  await page.keyboard.type('hello');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.value = 'x';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'x' }));
  });
  expect(await bytes(page)).toBe('hellox');
});

test('synthetic composition removal and reinsertion share one transaction (not a recorded Safari trace)', async ({ page }) => {
  await page.keyboard.type('abc');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    const input = (type: string, inputType: string, data: string | null) => {
      const event = new InputEvent(type, { bubbles: true, cancelable: type === 'beforeinput', data });
      Object.defineProperty(event, 'inputType', { value: inputType });
      t.dispatchEvent(event);
    };
    t.setSelectionRange(1, 2);
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: 'x' }));
    t.value = 'axc';
    t.setSelectionRange(1, 2);
    input('beforeinput', 'deleteCompositionText', null);
    t.value = 'ac';
    t.setSelectionRange(1, 1);
    t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: 'x' }));
    input('beforeinput', 'insertFromComposition', 'x');
    t.value = 'axc';
    t.setSelectionRange(2, 2);
    input('input', 'insertFromComposition', 'x');
  });
  expect(await bytes(page)).toBe('abc\x7f\x7fxc');
  expect(await page.locator('textarea').inputValue()).toBe('axc');
});

test('matching release expires an abandoned overtaken key transaction', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'a' }));
    (window as unknown as { input: { paste(data: string): void } }).input.paste('P');
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.value = 'a';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'a' }));
  });
  expect(await bytes(page)).toBe('aPa');
});

test('hardware text followed by a native middle edit has one owner', async ({ page }) => {
  await page.keyboard.type('hello');
  expect(await bytes(page)).toBe('hello');
  expect(await page.locator('textarea').inputValue()).toBe('hello');
  await edit(page, 'hallo', 1, 2, 'a', 'insertReplacementText');
  expect(await bytes(page)).toBe('hello' + '\x7f'.repeat(4) + 'allo');
  await page.keyboard.press('Enter');
  expect(await bytes(page)).toBe('hello' + '\x7f'.repeat(4) + 'allo\r');
});

test('composition is committed once before keyless Return, without legacy 229 deletion after paste', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Process', keyCode: 229 }));
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    for (const data of ['h', 'hello']) {
      t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data }));
      t.value = data;
      t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertCompositionText', data, isComposing: true }));
    }
  });
  expect(await bytes(page)).toBe('');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertLineBreak' }));
    t.value += '\n';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertLineBreak' }));
    t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: 'hello' }));
  });
  expect(await bytes(page)).toBe('hello\r');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Process', keyCode: 229 }));
    t.value = 'old helper\nrun';
    (window as unknown as { terminal: { paste(data: string): void } }).terminal.paste('safe');
  });
  // A browser task boundary exposes xterm's legacy timer if it was armed.
  await page.evaluate(() => new Promise(resolve => setTimeout(resolve, 30)));
  expect(await bytes(page)).toBe('hello\rsafe');
});

test('copied terminal helper text is never replayed as an editable line', async ({ page }) => {
  await edit(page, 'hello', 0, 0, 'hello');
  await page.evaluate(async () => {
    const term = (window as unknown as { terminal: { write(data: string, cb: () => void): void; selectAll(): void } }).terminal;
    await new Promise<void>(resolve => term.write('a\r\nrun', resolve));
    term.selectAll();
    document.querySelector('.xterm-screen')!.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, clientX: 10, clientY: 10 }));
  });
  expect(await page.locator('textarea').inputValue()).toContain('run');
  await edit(page, 'ab', 0, 5, 'ab', 'insertReplacementText');
  expect(await bytes(page)).toBe('helloab');
  // Selection ownership has returned to editable text, not clipboard helper.
  await edit(page, 'ac', 1, 2, 'c', 'insertReplacementText');
  expect(await bytes(page)).toBe('helloab\x7fc');
});

test('a control boundary settles a browser transaction before its delayed input', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'insertText', data: 'ä' }));
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter', keyCode: 13 }));
    t.value = 'ä';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'ä' }));
  });
  expect(await bytes(page)).toBe('ä\r');
  await edit(page, 'ä', 0, 0, 'ä');
  expect(await bytes(page)).toBe('ä\rä');
});

test('native backspace preserves the editable suffix for subsequent dictation', async ({ page }) => {
  await page.keyboard.type('hello');
  await page.keyboard.press('Backspace');
  expect(await page.locator('textarea').inputValue()).toBe('hell');
  await edit(page, 'help', 3, 4, 'p', 'insertReplacementText');
  expect(await bytes(page)).toBe('hello\x7f\x7fp');
});

test('replacement uses the embedded Ink editor grapheme erase unit', async ({ page }) => {
  await edit(page, 'a👩🏽‍💻', 0, 0, 'a👩🏽‍💻');
  await edit(page, 'ab', 1, 8, 'b', 'insertReplacementText');
  expect(await bytes(page)).toBe('a👩🏽‍💻\x7fb');
});

test('terminal navigation and deletion outside the mirror still reach the PTY', async ({ page }) => {
  await page.keyboard.type('abc');
  await page.keyboard.press('ArrowLeft');
  await page.keyboard.press('Backspace');
  expect(await bytes(page)).toBe('abc\x1b[D\x7f');
});

test('composition preedit remains visible in the native terminal', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '日本' }));
  });
  await expect(page.locator('.composition-view')).toHaveText('日本');
  await expect(page.locator('.composition-view')).toBeVisible();
  expect(await bytes(page)).toBe('');
});

test('finalized composition claims one input-only insertText observation, not a new equal edit', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '好' }));
    t.value = '好';
    t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '好' }));
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: '好' }));
  });
  expect(await bytes(page)).toBe('好');
  await edit(page, '好好', 1, 1, '好');
  expect(await bytes(page)).toBe('好好');
});

test('composition claim expires on distinct beforeinput, payload/value mismatch, release, or consumption', async ({ page }) => {
  for (const boundary of ['beforeinput', 'payload', 'value', 'release', 'consumed', 'next-task']) {
    const result = await page.evaluate(async boundary => {
      const state = window as unknown as { input: { reset(): void }; frames: string[] };
      state.input.reset();
      state.frames.length = 0;
      const t = document.querySelector('textarea')!;
      t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Process', code: 'KeyA', keyCode: 229 }));
      t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
      t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '好' }));
      t.value = '好';
      t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '好' }));
      const input = (data: string, value = data) => {
        t.value = value;
        t.setSelectionRange(value.length, value.length);
        t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data }));
      };
      if (boundary === 'beforeinput') t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: '好' }));
      if (boundary === 'payload') input('x');
      if (boundary === 'value') input('好', '好好');
      if (boundary === 'release') t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'Process', code: 'KeyA', keyCode: 229 }));
      if (boundary === 'consumed') input('好');
      if (boundary === 'next-task') await new Promise(resolve => setTimeout(resolve, 0));
      input('好', boundary === 'beforeinput' ? '好好' : '好');
      return state.frames.join('');
    }, boundary);
    expect(result, boundary).toBe(boundary === 'payload' ? '好x好' : boundary === 'value' ? '好好好' : '好好');
  }
});

test('a finalized composition cannot replay its trailing input after Return', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '好' }));
    t.value = '好';
    t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '好' }));
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter', keyCode: 13 }));
    t.value = '好';
    const input = new InputEvent('input', { bubbles: true, data: '好' });
    // Chromium normalizes WebKit-only inputType strings to ''. Preserve the
    // replayed type so this trace cannot pass by ignoring an unsupported event.
    Object.defineProperty(input, 'inputType', { value: 'insertFromComposition' });
    t.dispatchEvent(input);
  });
  expect(await bytes(page)).toBe('好\r');
  await edit(page, '好', 0, 0, '好');
  expect(await bytes(page)).toBe('好\r好');
});

test('keyless Return retains one trailing composition claim after a redundant compositionend', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }));
    t.dispatchEvent(new CompositionEvent('compositionupdate', { bubbles: true, data: '好' }));
    t.value = '好';
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertLineBreak' }));
    t.value = '好\n';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertLineBreak' }));
    t.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: '好' }));
    t.value = '好';
    const trailing = new InputEvent('input', { bubbles: true, data: '好' });
    Object.defineProperty(trailing, 'inputType', { value: 'insertFromComposition' });
    t.dispatchEvent(trailing);
  });
  expect(await bytes(page)).toBe('好\r');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.value = '好';
    const later = new InputEvent('input', { bubbles: true, data: '好' });
    Object.defineProperty(later, 'inputType', { value: 'insertFromComposition' });
    t.dispatchEvent(later);
  });
  expect(await bytes(page)).toBe('好\r好');
});

test('keyless replacement preserves an older printable key', async ({ page }) => {
  await page.keyboard.type('teh');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.setSelectionRange(0, 3);
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertReplacementText', data: 'the' }));
    t.value = 'the';
    t.setSelectionRange(3, 3);
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: 'the' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
  });
  expect(await bytes(page)).toBe('teh\x7f\x7fhea');
  expect(await page.locator('textarea').inputValue()).toBe('thea');
});

test('input-only replacement fails closed without consuming an unrelated printable key', async ({ page }) => {
  await page.keyboard.type('teh');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.value = 'the';
    t.setSelectionRange(3, 3);
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: 'the' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
  });
  expect(await bytes(page)).toBe('teha');
  expect(await page.locator('textarea').inputValue()).toBe('a');
});

test('keyless multi-character insertion preserves an older printable key', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'hello' }));
    // Browser default mutation runs after beforeinput; the adapter may have
    // settled the older key into the editable mirror in the meantime.
    t.value += 'hello';
    t.setSelectionRange(t.value.length, t.value.length);
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'hello' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
  });
  expect(await bytes(page)).toBe('ahello');
});

test('keyless single-character insertion preserves an unrelated printable key', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'x' }));
    // Browser default mutation runs after beforeinput; the adapter may have
    // settled the older key into the editable mirror in the meantime.
    t.value += 'x';
    t.setSelectionRange(t.value.length, t.value.length);
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'x' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
  });
  expect(await bytes(page)).toBe('ax');
});

test('released rollover keys never overtake the older beforeinput transaction', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    for (const key of ['a', 'b']) t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key, code: `Key${key.toUpperCase()}` }));
    t.dispatchEvent(new InputEvent('beforeinput', { bubbles: true, inputType: 'insertText', data: 'ä' }));
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'b', code: 'KeyB' }));
  });
  expect(await bytes(page)).toBe('');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.value = 'ä';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'ä' }));
  });
  expect(await bytes(page)).toBe('äb');
});

test('clipboard paste is forwarded once and cannot cause a second native mutation', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    const clipboardData = new DataTransfer();
    clipboardData.setData('text/plain', 'hello hello');
    const paste = new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData });
    t.dispatchEvent(paste);
    if (!paste.defaultPrevented) throw new Error('handled paste must prevent default');
  });
  expect(await bytes(page)).toBe('hello hello');
  expect(await page.locator('textarea').inputValue()).toBe('');
});

test('Alt terminal shortcuts are controls, not printable fallbacks', async ({ page }) => {
  await page.keyboard.press('Alt+b');
  expect(await bytes(page)).toBe('\x1bb');
});

test('native textarea selection keeps its own copy menu', async ({ page }) => {
  await edit(page, 'hello', 0, 0, 'hello');
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.setSelectionRange(1, 4);
    t.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true }));
    const copy = new ClipboardEvent('copy', { bubbles: true, cancelable: true, clipboardData: new DataTransfer() });
    t.dispatchEvent(copy);
    if (copy.defaultPrevented) throw new Error('xterm replaced native selection copy');
  });
  expect(await page.locator('textarea').inputValue()).toBe('hello');
  expect(await bytes(page)).toBe('hello');
});

test('a canceled beforeinput releases its raw key without swallowing the next keyless edit', async ({ page }) => {
  await page.evaluate(() => {
    const t = document.querySelector('textarea')!;
    t.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'a', code: 'KeyA' }));
    const before = new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'insertText', data: 'a' });
    before.preventDefault();
    t.dispatchEvent(before);
    t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key: 'a', code: 'KeyA' }));
    t.value = 'aa';
    t.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: 'a' }));
  });
  expect(await bytes(page)).toBe('aa');
});

test('native dictation replacements rewrite the acknowledged terminal tail, retaining repeated words', async ({ page }) => {
  await edit(page, 'hello', 0, 0, 'hello');
  await edit(page, 'hello hello', 5, 5, ' hello');
  await edit(page, 'Hello hello!', 0, 11, 'Hello hello!', 'insertReplacementText');
  expect(await bytes(page)).toBe('hello hello' + '\x7f'.repeat(11) + 'Hello hello!');
  // iPhone long-press-space moves only the native caret. The terminal stays at
  // the tail, so replacing a middle range must restore its unchanged suffix.
  await edit(page, 'Hello there!', 6, 11, 'there', 'insertReplacementText');
  expect(await bytes(page)).toBe('hello hello' + '\x7f'.repeat(11) + 'Hello hello!' + '\x7f'.repeat(6) + 'there!');
});
