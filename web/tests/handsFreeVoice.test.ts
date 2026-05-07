import assert from "node:assert/strict";
import test from "node:test";

import {
  HANDS_FREE_CHUNK_MAX_CHARS,
  HANDS_FREE_MAX_SPOKEN_CHARS,
  HANDS_FREE_STREAM_CHUNK_MIN_CHARS,
  HANDS_FREE_TRUNCATION_NOTICE,
  handsFreeMeterPercent,
  handsFreeSilenceThreshold,
  handsFreeSpeechThreshold,
  isHandsFreeDisableCommand,
  isHandsFreeStopCommand,
  normalizeTranscriptCommand,
  prepareHandsFreeSpeech,
  rmsFromTimeDomain,
  sanitizeHandsFreeSpeechText,
  splitHandsFreeSpeech,
  takeReadyHandsFreeSpeechChunks,
} from "../src/lib/handsFreeVoice.ts";

test("normalizes and classifies hands-free voice commands", () => {
  assert.equal(normalizeTranscriptCommand("  Arrête,   s'il te plaît! "), "arrete sil te plait");
  assert.equal(isHandsFreeStopCommand("Arrête"), true);
  assert.equal(isHandsFreeStopCommand("désactive le mode vocal"), false);
  assert.equal(isHandsFreeDisableCommand("Désactive le mode vocal."), true);
  assert.equal(isHandsFreeDisableCommand("stop"), false);
});

test("sanitizes dashboard media and markdown before speaking", () => {
  const clean = sanitizeHandsFreeSpeechText(
    "Intro **forte** [lien](https://example.com)\n```ts\nconst x = 1\n```\n[[audio_as_voice]]\nMEDIA:/tmp/voice.mp3",
  );

  assert.equal(clean, "Intro forte lien");
});

test("prepares spoken text with a global cap and a chat-detail notice", () => {
  const longText = `${"phrase ".repeat(130)}fin.`;
  const spoken = prepareHandsFreeSpeech(longText);

  assert.ok(spoken.length <= HANDS_FREE_MAX_SPOKEN_CHARS + HANDS_FREE_TRUNCATION_NOTICE.length + 1);
  assert.ok(spoken.endsWith(HANDS_FREE_TRUNCATION_NOTICE));
});

test("does not stream tiny TTS chunks and flushes at the end", () => {
  const shortText = "Une phrase courte.";
  const pending = takeReadyHandsFreeSpeechChunks(shortText, false);
  assert.deepEqual(pending.chunks, []);
  assert.equal(pending.rest, shortText);

  const flushed = takeReadyHandsFreeSpeechChunks(shortText, true);
  assert.deepEqual(flushed.chunks, [shortText]);
  assert.equal(flushed.rest, "");
});

test("streams larger TTS chunks instead of one call per sentence", () => {
  const first = "Premiere phrase. ";
  const text = first.repeat(Math.ceil((HANDS_FREE_CHUNK_MAX_CHARS + 160) / first.length));
  const { chunks, rest } = takeReadyHandsFreeSpeechChunks(text, false);

  assert.equal(chunks.length, 1);
  assert.ok(chunks[0].length >= HANDS_FREE_STREAM_CHUNK_MIN_CHARS);
  assert.ok(chunks[0].length <= HANDS_FREE_CHUNK_MAX_CHARS);
  assert.ok(rest.length > 0);
});

test("splits fallback speech into capped chunks", () => {
  const text = `${"Segment assez long pour tester le decoupage. ".repeat(60)}fin.`;
  const chunks = splitHandsFreeSpeech(text);
  const spoken = chunks.join(" ");

  assert.ok(chunks.length >= 1);
  assert.ok(chunks.every((chunk) => chunk.length <= HANDS_FREE_CHUNK_MAX_CHARS * 1.5));
  assert.ok(spoken.includes(HANDS_FREE_TRUNCATION_NOTICE));
});

test("computes adaptive VAD thresholds and RMS levels", () => {
  assert.equal(rmsFromTimeDomain(new Uint8Array([128, 128, 128])), 0);
  assert.ok(rmsFromTimeDomain(new Uint8Array([0, 255])) > 0.99);

  assert.equal(handsFreeSpeechThreshold(0), 0.022);
  assert.equal(handsFreeSilenceThreshold(0), 0.012);
  assert.ok(handsFreeSpeechThreshold(0.05) > handsFreeSpeechThreshold(0));
});

test("keeps the mic meter bounded for mobile UI", () => {
  assert.equal(handsFreeMeterPercent({ level: 0, noise: 0.01, speaking: false, threshold: 0.02 }), 3);
  assert.equal(handsFreeMeterPercent({ level: 1, noise: 0.01, speaking: true, threshold: 0.02 }), 100);
});
