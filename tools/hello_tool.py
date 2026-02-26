from tools.registry import registry

def hello_handler(args: dict, **kwargs) -> str:
    name = args.get("name", "world")
    return f"Hello {name}"

hello_schema = {
    "name": "hello",
    "description": "Say hello to someone.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name to greet"
            }
        },
        "required": []
    }
}

registry.register(
    name="hello",
    toolset="custom",
    schema=hello_schema,
    handler=hello_handler,
    check_fn=None,
    requires_env=[],
    is_async=False,
    description="Simple hello test tool"
)