import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";
import { KanbanInlineChat } from "./KanbanInlineChat";
import { parseSseChunk } from "./KanbanInlineChat.utils";

function renderInlineChat(onCardCreated = vi.fn()) {
  return {
    onCardCreated,
    ...render(
      <I18nProvider>
        <KanbanInlineChat boardId="default" onCardCreated={onCardCreated} />
      </I18nProvider>,
    ),
  };
}

function streamResponse(body: string): Response {
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body));
        controller.close();
      },
    }),
    {
      headers: { "Content-Type": "text/event-stream" },
      status: 200,
    },
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("parseSseChunk", () => {
  it("parses complete SSE events and preserves partial data", () => {
    const parsed = parseSseChunk(
      'data: {"type":"card_created","card_id":"t_1"}\n\n'
        + 'data: {"type":"result","content":"ok"}\n',
    );

    expect(parsed.events).toEqual([{ type: "card_created", card_id: "t_1" }]);
    expect(parsed.rest).toBe('data: {"type":"result","content":"ok"}\n');
  });
});

describe("KanbanInlineChat", () => {
  it("posts text to inline-dispatch and renders streamed card/result events", async () => {
    const fetchMock = vi.spyOn(window, "fetch").mockResolvedValue(
      streamResponse(
        'data: {"type":"card_created","card_id":"t_123"}\n\n'
          + 'data: {"type":"result","content":"Created one card."}\n\n',
      ),
    );
    const onCardCreated = vi.fn();
    renderInlineChat(onCardCreated);

    fireEvent.change(screen.getByLabelText("Tell the board what to do..."), {
      target: { value: "make a release checklist" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findByText("Created one card.");
    expect(onCardCreated).toHaveBeenCalledWith("t_123");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/plugins/kanban/boards/default/inline-dispatch?stream=true",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ text: "make a release checklist" }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("keeps TTS muted by default", async () => {
    const fetchMock = vi.spyOn(window, "fetch").mockResolvedValue(
      streamResponse('data: {"type":"result","content":"Done."}\n\n'),
    );
    renderInlineChat();

    fireEvent.change(screen.getByLabelText("Tell the board what to do..."), {
      target: { value: "summarize" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    await screen.findByText("Done.");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).not.toContain("/tts");
  });

  it("shows text-input guidance when browser voice capture is unavailable", async () => {
    renderInlineChat();

    await waitFor(() => {
      expect(screen.getByText("Voice input is unavailable in this browser. Please use text input.")).toBeTruthy();
    });
    expect(screen.getByRole("button", { name: "Start voice input" }).hasAttribute("disabled")).toBe(true);
  });
});
