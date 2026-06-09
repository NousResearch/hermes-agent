import assert from "node:assert/strict";
import test from "node:test";

import { nextHasEverActivated } from "./chatActivation";

test("stays inactive before the chat tab has ever been opened", () => {
  assert.equal(nextHasEverActivated(false, false), false);
});

test("activates the PTY the first time the chat tab becomes visible", () => {
  assert.equal(nextHasEverActivated(false, true), true);
});

test("keeps the PTY alive after the user leaves the chat tab", () => {
  assert.equal(nextHasEverActivated(true, false), true);
});
