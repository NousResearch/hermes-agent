# Composer behavior contributions

Desktop plugins can contribute inline ghost text and busy-submit key preferences without editing the DOM or capturing global keyboard events. `host.composer.behaviorVersion === 2` advertises this interface. Register `composer.behavior.v2` with `data.protocol: 2`; other versions are ignored. With no provider, or more than one valid provider, native composer behavior remains in control.

The concrete consumer is [Neo Composer](https://github.com/medking82/hermes-neo-composer). Prediction generation, cost selection, persistence and settings live in that standalone plugin, not in core. Core does not start model requests or choose models.

## Contract

`create(environment)` returns a controller with `update({enabled, empty})`, `decide('consumed' | 'dismissed')`, `refresh()` and `dispose()`.

- `environment.context` identifies `connectionId`, `profile`, runtime `sessionId`, durable `storedId` and `busy` state. Never substitute a runtime handle for a database session ID.
- `publish(text | null)` paints decorative ghost text. It does not write or submit a draft.
- `request(path, body)` performs a profile-routed request only under the contributing plugin's `/api/plugins/<id>` namespace. Paths are single named endpoints, not arbitrary URLs.
- `onSettled(listener)` subscribes to the native settled bookend. A final message event can precede the gateway clearing its running flag.
- `update` informs the provider when the native composer is eligible and empty. Providers must stop publishing after disposal and preserve draft ownership across profile or connection changes.
- `decide(status)` returns the handled suggestion text for **both** `consumed` and `dismissed`; `null` declines the gesture. The controller publishes `null` when it withdraws a handled suggestion. For `consumed`, the native editor inserts the returned text as an undoable draft without sending; for `dismissed`, it consumes Escape without inserting text and records the explicit decision.
- Optional `keyAction(event)` can return only `queue`, `steer` or `undefined`; it cannot supply arbitrary actions. The native composer retains IME, modifier, slash-menu and other completion arbitration. `steer` uses the upstream visible `redirectPrompt(text, 'steer')` path to the existing `session.steer` action, not `session.redirect`; it retains the visible user message without sealing the current stream. The separate upstream `onSteerHidden` path remains hidden, and the default interrupt mode remains available.
- Optional `readOnlyKeys` supplies display hints for existing read-only composer keybinds. Queue/steer tooltips follow the active contribution; without one, the native busy-input policy remains authoritative.
- Optional `lifecycle` receives profile rename/drop notifications. Treat these as cleanup signals, not permission to move data between owners.

## User control and compatibility

Ghost text is decorative until Tab inserts it. The result remains editable and undoable, and only a separate send gesture submits it. Escape dismisses; typing, non-empty drafts, busy turns and competing native completion UI suppress it. Shift+Enter remains newline.

This is a trusted-plugin extension, not a sandbox. Plugins must enforce their own profile-scoped storage and backend authorization. Request errors and plugin exceptions must not prevent normal composing. Do not use document-level event handlers, synthetic send actions or CSS/DOM overlays to emulate this contract.

## Verification

Colocated tests cover version/source admission, conflicting contributions, namespace isolation, lifecycle handling, conditional keyboard arbitration, editable Tab acceptance, Escape and genuine steer-mode dispatch. The standalone consumer additionally exercises the native Composer and prompt-action hook against the real backend steering handler with deterministic inference/tool substitutes; those fixtures are not evidence of live model output or of a user's current window being upgraded.
