#!/usr/bin/env node

/**
 * Persistent bridge from Hermes browser_exec to Stagehand's production
 * Playwright facade. Protocol messages stay on stdout; diagnostics use stderr.
 */

import { createInterface } from "node:readline";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const PROTOCOL = "hermes-stagehand-facade-v1";
let resourcesPromise;
let closing = false;

function write(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function requiredEnv(name) {
  const value = String(process.env[name] ?? "").trim();
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function sanitize(message) {
  return String(message)
    .replace(/([?&](?:signingKey|apiKey|api_key|token|key)=)[^&\s"']+/gi, "$1[redacted]")
    .replace(/\b(sk-[A-Za-z0-9_-]{6})[A-Za-z0-9_-]+/g, "$1[redacted]")
    .replace(/\b(bb_(?:live|test)_[A-Za-z0-9]{4})[A-Za-z0-9_-]+/g, "$1[redacted]")
    .replace(/\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}/gi, "$1[redacted]")
    .slice(0, 1000);
}

function stringifyResult(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

async function createResources() {
  const root = requiredEnv("STAGEHAND_FACADE_ROOT");
  const sdkUrl = pathToFileURL(join(root, "packages/sdk-ts/dist/index.mjs")).href;
  const facadeUrl = pathToFileURL(
    join(root, "packages/integrations/core/dist/facade/index.mjs"),
  ).href;
  const [{ browserbase, Stagehand }, facade] = await Promise.all([
    import(sdkUrl),
    import(facadeUrl),
  ]);
  const browser = await browserbase.launch({
    apiKey: requiredEnv("BROWSERBASE_API_KEY"),
    ...(String(process.env.BROWSERBASE_PROJECT_ID ?? "").trim()
      ? { projectId: String(process.env.BROWSERBASE_PROJECT_ID).trim() }
      : {}),
    keepAlive: false,
  });
  try {
    const stagehand = await Stagehand.create({ browser, logging: { level: "off" } });
    return {
      browser,
      stagehand,
      tools: new facade.StagehandFacadeTools(stagehand),
      runSchema: facade.CodeModeRunInputSchema,
    };
  } catch (error) {
    await browser.close().catch(() => undefined);
    throw error;
  }
}

async function resources() {
  resourcesPromise ??= createResources().catch((error) => {
    resourcesPromise = undefined;
    throw error;
  });
  return await resourcesPromise;
}

async function call(code) {
  const active = await resources();
  const input = active.runSchema.parse({ code });
  return stringifyResult(await active.tools.run(input.code));
}

async function shutdown() {
  if (closing) return;
  closing = true;
  const active = await Promise.race([
    resourcesPromise?.catch(() => undefined),
    new Promise((resolve) => setTimeout(() => resolve(undefined), 5000)),
  ]);
  if (active) {
    await active.stagehand.close().catch(() => undefined);
    await active.browser.close().catch(() => undefined);
  }
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
write({ protocol: PROTOCOL, type: "ready" });

for await (const line of lines) {
  let request;
  try {
    request = JSON.parse(line);
    if (!request || request.protocol !== PROTOCOL) {
      throw new Error("invalid facade request");
    }
    if (request.type === "shutdown") {
      await shutdown();
      write({ protocol: PROTOCOL, type: "shutdown", request_id: request.request_id });
      process.exit(0);
    }
    if (request.type !== "call") throw new Error("unknown facade request type");
    const output = await call(String(request.code ?? ""));
    write({
      protocol: PROTOCOL,
      type: "response",
      request_id: request.request_id,
      success: true,
      output,
    });
  } catch (error) {
    write({
      protocol: PROTOCOL,
      type: "response",
      request_id: request?.request_id,
      success: false,
      error: sanitize(error instanceof Error ? error.message : String(error)),
      error_type: error instanceof Error ? error.name : "Error",
    });
  }
}

await shutdown();
