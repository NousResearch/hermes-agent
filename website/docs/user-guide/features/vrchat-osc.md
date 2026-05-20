# VRChat OSC

Hermes can send messages to VRChat through VRChat's official local OSC
interface. The feature is opt-in and stays local: it does not use VRChat
credentials, modify the client, or load avatar assets.

## Install

Install the optional dependency:

```bash
uv pip install "hermes-agent[vrchat]"
```

Then enable the `vrchat` toolset in your Hermes tool settings or agent config.

## VRChat Setup

In VRChat, enable OSC in the radial menu:

```text
Options > OSC > Enabled
```

Hermes uses these defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VRCHAT_OSC_HOST` | `127.0.0.1` | VRChat OSC host |
| `VRCHAT_OSC_SEND_PORT` | `9000` | Port Hermes sends to |
| `VRCHAT_OSC_RECV_PORT` | `9001` | Port VRChat sends from |

## Tools

| Tool | Purpose |
| --- | --- |
| `vrchat_chatbox` | Sends text to `/chatbox/input` |
| `vrchat_typing` | Shows or hides `/chatbox/typing` |
| `vrchat_avatar_param` | Sets `/avatar/parameters/{name}` |
| `vrchat_send_osc` | Sends a raw OSC message to a local address |
| `vrchat_status` | Shows endpoint and dependency status |

## Guardrails

- Chatbox text is capped at VRChat's 144-character limit.
- Avatar parameter writes accept only booleans, integers, and floats.
- Raw OSC addresses must start with `/` and cannot contain whitespace.
- `vrchat_avatar_param` accepts a parameter name, not a full OSC path.
