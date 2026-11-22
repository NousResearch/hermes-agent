"""Report the host clock in UTC and a requested or configured timezone."""
import time

from hermes_time import format_timestamp, report_timezone
from tools.registry import registry, tool_error, tool_result


def get_time(args, **kwargs):
    if set(args) - {"timezone"}:
        return tool_error("get_time accepts only timezone.", code="invalid_arguments")
    if "timezone" in args and not isinstance(args["timezone"], str):
        return tool_error("timezone must be an IANA timezone name.", code="invalid_timezone")
    try:
        zone = report_timezone(args.get("timezone"))
    except ValueError as exc:
        return tool_error(str(exc), code="invalid_timezone")
    instant = time.time()
    return tool_result(utc=format_timestamp(instant), local=format_timestamp(instant, zone), timezone=zone.key)


registry.register(
    name="get_time", toolset="time", handler=get_time,
    schema={
        "name": "get_time",
        "description": "Read the current host time in UTC and a local timezone.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA timezone. Defaults to the Hermes timezone, or UTC if unset."}},
            "additionalProperties": False,
        },
    },
)
