"""Schemas for the browser tool surface that exists only on the extension lane.

A session whose requests can bind a browser extension controller (``browser.extension_control.enabled`` plus a
server-bound principal and transport) is offered this surface: the tools below and the extension form of the
seven long-standing browser tools. Every other session keeps the browser tools exactly as they were, with no
``tab_id``, no approval or background-tab wording and none of these tools. The choice is made per session in
``model_tools._rewrite_for_extension_lane``, from something fixed for the session, so the tool block stays
byte-identical for the whole conversation whatever is attached right now (a changing block would lose the
provider's prompt cache).
"""

import copy
from typing import Any

# Routed tools that can act on a chosen tab instead of the current one. The extension validates the id against
# the tabs attached to Hermes and refuses any other with lease_not_owned. browser_tabs, browser_tab_create and
# the two tab tools that already take a tab_id are not in this list.
TAB_TARGETING_TOOLS = frozenset({
    "browser_navigate", "browser_snapshot", "browser_click", "browser_type", "browser_scroll", "browser_back",
    "browser_press", "browser_batch", "browser_fill", "browser_select", "browser_hover", "browser_drag",
    "browser_scroll_to", "browser_screenshot",
})
_TAB_ID_PROPERTY = {
    "type": "integer", "minimum": 1,
    "description": "Tab to act on, from browser_tabs or browser_tab_create (default: the current tab). "
                   "Refs belong to the tab they came from.",
}

EXTENSION_ONLY_SCHEMAS = [
    {
        "name": "browser_batch",
        "description": "Run several browser actions in order as one call and get the page back as it is after the last step. Prefer browser_batch when you can predict two or more steps ahead (for example click a field, type, press Enter): each step saved is a model turn saved. A step takes the same arguments as the single tool of the same name. Refs come from the latest snapshot or action result; a step that changes the page (a navigation, a form that re-renders) can invalidate the refs later steps use, so end the batch there. Steps run one at a time. With stop_on_error (default true) the batch stops at the first step that fails; with false it carries on, but it always stops when the person declines an approval or the page moves away. Where the person approves actions (a coordinate click, Enter, a drag, a click on a submit or delete button), each such step asks separately and the batch waits, so keep batches that need approvals short. Returns status (completed, completed_with_errors or stopped), one entry per step with its outcome or error, and the counts completed and remaining.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "description": "The steps to run in order (1-20).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["back", "click", "drag", "fill", "hover", "navigate", "press", "scroll", "scroll_to", "select", "type"],
                                "description": "What this step does"
                            },
                            "ref": {"type": "string", "description": "Element ref (e.g. '@e5') for click, type, fill, select, hover, scroll_to; the dragged element for drag"},
                            "to_ref": {"type": "string", "description": "drag: the element to drop on"},
                            "text": {"type": "string", "description": "type: the text to type"},
                            "value": {"type": "string", "description": "fill: the text to enter; select: the option's text or value"},
                            "clear": {"type": "boolean", "description": "type: clear the field first (default true)"},
                            "url": {"type": "string", "description": "navigate: the URL to open"},
                            "key": {"type": "string", "description": "press: the key (e.g. 'Enter', 'Tab', or a combo such as 'Control+a')"},
                            "modifiers": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["Alt", "Control", "Meta", "Shift"]},
                                "description": "press: modifier keys held with the key"
                            },
                            "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "scroll: which way"},
                            "amount": {"type": "integer", "minimum": 100, "maximum": 5000, "description": "scroll: pixels (default about 85% of the viewport)"},
                            "x": {"type": "number", "description": "click: horizontal position in CSS pixels, used with y instead of ref"},
                            "y": {"type": "number", "description": "click: vertical position in CSS pixels, used with x instead of ref"}
                        },
                        "required": ["action"]
                    }
                },
                "stop_on_error": {
                    "type": "boolean",
                    "default": True,
                    "description": "Stop at the first step that fails (default true). When false, later steps still run."
                }
            },
            "required": ["actions"]
        }
    },
    {
        "name": "browser_fill",
        "description": "Set a text field's value in one step: the field is focused, its contents are replaced with value, and the page is told it changed. Prefer it over browser_type for plain text inputs and textareas; use browser_type when the page needs real keystrokes. With the browser extension, password, payment-card, one-time-code and other credential fields, and text that looks like a secret, are refused: ask the person to enter those themselves. Works on a background tab. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The field's element reference from the snapshot (e.g., '@e3')"},
                "value": {"type": "string", "description": "The text to put in the field"}
            },
            "required": ["ref", "value"]
        }
    },
    {
        "name": "browser_select",
        "description": "Choose an option in a dropdown (a <select> element) identified by its ref ID. For a custom dropdown that is not a <select>, click it and click the option instead. With the browser extension, payment and credential fields are refused. Works on a background tab. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The dropdown's element reference from the snapshot (e.g., '@e7')"},
                "value": {"type": "string", "description": "The option's visible text or its value"}
            },
            "required": ["ref", "value"]
        }
    },
    {
        "name": "browser_hover",
        "description": "Move the pointer over an element identified by its ref ID, for menus and tooltips that open on hover. With the browser extension this fails with tab_hidden on a background tab: bring the tab forward with browser_tab_activate first. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e5')"}
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_drag",
        "description": "Drag one element and drop it on another, both identified by ref IDs. With the browser extension the person is asked to approve every drag and the call waits for the answer, so do not repeat it; it fails with tab_hidden on a background tab. The page is not returned after a drag: take a snapshot to see the result. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The element to pick up (e.g., '@e5')"},
                "to_ref": {"type": "string", "description": "The element to drop it on (e.g., '@e9')"}
            },
            "required": ["ref", "to_ref"]
        }
    },
    {
        "name": "browser_scroll_to",
        "description": "Scroll the page until an element identified by its ref ID is in view, without moving the pointer. Works on a background tab. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "The element reference from the snapshot (e.g., '@e5')"}
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_tabs",
        "description": "List the browser tabs this connection controls: each tab's id, title, URL and whether it is the active one. Only tabs the person attached to Hermes, or that you opened, are listed. Use the ids as tab_id on any action, or with browser_tab_activate and browser_tab_close.",
        "parameters": {
            "type": "object", "properties": {}, "required": []
        }
    },
    {
        "name": "browser_tab_activate",
        "description": "Bring one of your tabs to the front. This changes what the person is looking at, so the browser extension asks them to approve first (the request names the page) and the call waits for their answer: do not repeat it while it waits, and treat a refusal as final. You rarely need it: pass tab_id to any action to act on a tab without bringing it forward. click, type, press, scroll and scroll_to work in a background tab; hover and drag need the tab in front.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "integer", "minimum": 1, "description": "The tab's id from browser_tabs"}
            },
            "required": ["tab_id"]
        }
    },
    {
        "name": "browser_tab_create",
        "description": "Open a URL in a new tab. With the browser extension the tab always opens in the background, behind the one the person is looking at, so it never takes their view; browser_tab_activate brings it forward after the person approves. Returns the new tab's id: pass it as tab_id to act on that tab. A URL is required.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The page to load in the new tab (e.g., 'https://example.com')"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_tab_close",
        "description": "Close a tab this connection owns, such as one opened with browser_tab_create. The browser extension asks the person to approve and the call waits for their answer; any other tab id is refused with lease_not_owned.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_id": {"type": "integer", "minimum": 1, "description": "The tab's id from browser_tabs"}
            },
            "required": ["tab_id"]
        }
    },
    {
        "name": "browser_tab_group",
        "description": "Put tabs into one native browser tab group so the person can see them together. Pass the ids from browser_tabs. It groups only tabs this connection owns that belong to the same task set, such as tabs the person attached together as one task; a tab you open with browser_tab_create is a task set of its own, so tabs from separate calls cannot be grouped with each other. It never switches the tab the person is looking at and needs no approval. Returns the group's id. lease_not_owned means an id is not one of your tabs or the tabs are not all in one task set; any other error means the browser could not group them. Chromium-based browsers only.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_ids": {
                    "type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "uniqueItems": True,
                    "description": "The ids of the tabs to group, from browser_tabs"
                }
            },
            "required": ["tab_ids"]
        }
    },
    {
        "name": "browser_tab_ungroup",
        "description": "Take tabs out of their native browser tab group and leave them open, the reverse of browser_tab_group. Pass the ids from browser_tabs. The same rules apply: only tabs this connection owns that belong to the same task set, with lease_not_owned when they do not. It never switches the tab the person is looking at and needs no approval. Chromium-based browsers only.",
        "parameters": {
            "type": "object",
            "properties": {
                "tab_ids": {
                    "type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "uniqueItems": True,
                    "description": "The ids of the tabs to take out of their group, from browser_tabs"
                }
            },
            "required": ["tab_ids"]
        }
    },
    {
        "name": "browser_screenshot",
        "description": "Take a screenshot of the page, or of one element when ref is given, and see it directly. Use it when the snapshot cannot show what you need (layout, charts, images). It takes no question, and on a model without image input the result gives the saved file's path instead. The browser extension does not ask the person before a screenshot of a page they attached; on a background tab it can fail with tab_hidden: bring the tab forward with browser_tab_activate (the person approves) and retry. Refs come from the latest snapshot or action result.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Optional element reference to capture instead of the whole page (e.g., '@e5')"}
            },
            "required": []
        }
    },
]

_REFS_SENTENCE = ("Refs come from the page snapshot attached to the turn (or the most recent one in this "
                  "conversation) or from the latest action result.")

# The long-standing tools in their extension-lane form. A description that is a plain string replaces the main
# one; a (legacy sentence, extension sentence) pair rewrites only that sentence, so the long shared text of
# browser_navigate and browser_snapshot stays single-sourced. ``properties`` are added to (or replace) the main
# schema's, ``required`` replaces its list. tests/tools/test_browser_extension_lane_surface.py asserts every
# pair still applies, so a reworded main schema cannot leave stale advice behind.
_LANE_FORMS: dict[str, dict[str, Any]] = {
    "browser_navigate": {
        "description": (
            "Must be called before other browser tools.",
            "The page you are attached to is already open: call this only to move to a different URL."),
    },
    "browser_snapshot": {
        "description": (
            "Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot "
            "— use this to refresh after interactions that change the page, or with full=true for "
            "complete content.",
            "Browser actions that change the page already return it with fresh refs: use this for "
            "full=true (complete page content), to re-read a page that changed on its own, or when "
            "the turn carries no snapshot."),
    },
    "browser_click": {
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'), or on a point given by x and y in CSS pixels from the top-left of the viewport (only when no ref reaches the target). The ref IDs are shown in square brackets in the snapshot output. With the browser extension, a link that opens a new tab opens it in the background, clicking works on a background tab, and the person is asked to approve a coordinate click and a click on a button or link labelled like submit, send, post, publish, delete, remove, pay, buy, checkout, download or upload; the call waits for the answer, so do not repeat it. Refs come from the page snapshot attached to the turn (or the most recent one in this conversation) or from the latest action result.",
        "properties": {
            "x": {"type": "number", "description": "Horizontal position in CSS pixels; use together with y instead of ref"},
            "y": {"type": "number", "description": "Vertical position in CSS pixels; use together with x instead of ref"},
        },
        "required": [],  # a ref or both coordinates: checked in code, not by the schema
    },
    "browser_type": {
        "description": "Type text into an input field identified by its ref ID. Clears the field first (unless clear is false), then types the new text. With the browser extension, password, payment-card, one-time-code and other credential fields, and text that looks like a secret, are refused: ask the person to enter those themselves. Works on a background tab. Refs come from the page snapshot attached to the turn (or the most recent one in this conversation) or from the latest action result.",
        "properties": {
            "clear": {
                "type": "boolean", "default": True,
                "description": "Clear the field before typing (default true). Pass false to add to what is already there.",
            },
        },
    },
    "browser_scroll": {
        "description": "Scroll the page up, down, left or right. Use this to reveal more content that may be outside the current viewport. Without amount a scroll moves about a screen; amount sets the distance in pixels. Works on a background tab.",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Direction to scroll"},
            "amount": {
                "type": "integer", "minimum": 100, "maximum": 5000,
                "description": "Distance in pixels (100-5000). Omit for about a screen.",
            },
        },
    },
    "browser_back": {"description": "Navigate back to the previous page in browser history."},
    "browser_press": {
        "description": "Press a keyboard key, optionally with modifier keys held. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts (Control+a). With the browser extension, Enter or Return and paste shortcuts (Control+v, Meta+v, Shift+Insert) need the person's approval, and the call waits for the answer, so do not repeat it. Works on a background tab.",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown', or a combo such as 'Control+a')",
            },
            "modifiers": {
                "type": "array", "items": {"type": "string", "enum": ["Alt", "Control", "Meta", "Shift"]},
                "description": "Modifier keys held while the key is pressed",
            },
        },
    },
}
LANE_TOOLS = frozenset(_LANE_FORMS)
#: Every tool name the extension lane adds; a session without the lane is never offered one.
EXTENSION_ONLY_TOOLS = frozenset(schema["name"] for schema in EXTENSION_ONLY_SCHEMAS)

for _schema in EXTENSION_ONLY_SCHEMAS:
    if _schema["name"] in TAB_TARGETING_TOOLS:
        _schema["parameters"]["properties"]["tab_id"] = copy.deepcopy(_TAB_ID_PROPERTY)


def lane_description(name: str, description: str) -> str:
    """The extension-lane description of a long-standing browser tool, given its main description."""
    form = _LANE_FORMS[name]["description"]
    if isinstance(form, str):
        return form
    legacy, extension = form
    if legacy not in description:
        raise ValueError(f"{name}: the extension lane rewrites a sentence its description no longer contains")
    return description.replace(legacy, extension)


def lane_schema(base: dict[str, Any]) -> dict[str, Any]:
    """Extension-lane form of the long-standing browser tool schema ``base`` (not modified)."""
    name = base["name"]
    form = _LANE_FORMS[name]
    schema = copy.deepcopy(base)
    schema["description"] = lane_description(name, base.get("description", ""))
    parameters = schema["parameters"]
    parameters["properties"].update(copy.deepcopy(form.get("properties", {})))
    if name in TAB_TARGETING_TOOLS:
        parameters["properties"]["tab_id"] = copy.deepcopy(_TAB_ID_PROPERTY)
    if "required" in form:
        parameters["required"] = list(form["required"])
    return schema
