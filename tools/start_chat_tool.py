from tools.registry import registry

START_CHAT_SCHEMA = {
    "name": "start_chat",
    "description": (
        "Start a new chat in the Hermes desktop app and send it its first message, so the task runs "
        "there in view of the user while this chat stays where it is. Call it ONCE per task: every "
        "call opens another chat, and nothing de-duplicates a repeated call. The new chat starts with "
        "no history, so the message must carry everything the task needs. 'profile' must name an "
        "existing profile; omit it to use this chat's own profile. The result is 'started' with the "
        "new chat's session_id, or 'rejected' with a reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The new chat's first user message: the whole task, self-contained.",
            },
            "title": {
                "type": "string",
                "maxLength": 40,
                "description": "Sidebar title for the new chat, at most 40 characters. Omit it to let the chat title itself.",
            },
            "profile": {
                "type": "string",
                "description": "An existing profile to run the chat in. Omit it for this chat's own profile.",
            },
        },
        "required": ["message"],
    },
}


def _start_chat(args, **_kw):
    from tui_gateway.start_chat import start_chat

    return start_chat(args)


registry.register(
    name="start_chat",
    toolset="start_chat",
    schema=START_CHAT_SCHEMA,
    handler=_start_chat,
    emoji="💬",
)
