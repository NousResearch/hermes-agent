// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ConfirmDialog } from "./ConfirmDialog";

afterEach(cleanup);


it.each(['wrong', 'correct', 'cancel', 'reopen', 'space', 'duplicate', 'retry'])(
  'requires the exact typed intent before confirming: %s', async (mode) => {
    const onConfirm = vi.fn();
    const onClose = vi.fn();
    const props = { open: true, title: 'Restart', typedConfirmation: 'RESTART', onConfirm, onCancel: onClose };
    const view = render(<ConfirmDialog {...props} />);
    const input = await screen.findByRole('textbox');
    const confirm = await screen.findByRole('button', { name: 'Confirm' });
    await waitFor(() => expect(input.matches(':focus')).toBe(true));
    fireEvent.click(confirm);
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onConfirm).not.toHaveBeenCalled();
    if (mode === 'cancel') {
      fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
      expect(onClose).toHaveBeenCalledTimes(1);
      return;
    }
    fireEvent.change(input, { target: { value: mode === 'wrong' ? 'restart' : 'RESTART' } });
    if (mode === 'space') {
      const event = new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true });
      input.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(false);
      expect(onConfirm).not.toHaveBeenCalled();
    } else if (mode === 'retry') {
      let rejectAttempt!: (error: Error) => void;
      onConfirm.mockImplementationOnce(() => new Promise<void>((_resolve, reject) => {
        rejectAttempt = reject;
      }).catch(() => {})); // The consumer reports the error and keeps the prompt open.
      fireEvent.click(confirm);
      fireEvent.click(confirm);
      expect(onConfirm).toHaveBeenCalledTimes(1);
      await act(async () => { rejectAttempt(new Error('fixture failure')); });
      fireEvent.click(confirm);
      await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(2));
    } else if (mode === 'duplicate') {
      fireEvent.click(confirm);
      fireEvent.click(confirm);
      fireEvent.keyDown(input, { key: 'Enter' });
      await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(1));
    } else if (mode === 'reopen') {
      view.rerender(<ConfirmDialog {...props} open={false} />);
      view.rerender(<ConfirmDialog {...props} />);
      await waitFor(() => expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe(''));
      fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' });
      expect(onConfirm).not.toHaveBeenCalled();
    } else {
      fireEvent.keyDown(input, { key: 'Enter' });
      await waitFor(() => expect(onConfirm).toHaveBeenCalledTimes(mode === 'correct' ? 1 : 0));
    }
  }
);
