#!/usr/bin/env node
import { Agent, Cursor } from "@cursor/sdk";

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", chunk => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

function emit(payload) {
  process.stdout.write(JSON.stringify(payload));
}

function modelSelection(input) {
  const model = { id: input.model || "composer-2.5" };
  if (input.thinking) {
    model.params = [{ id: "thinking", value: input.thinking }];
  }
  return model;
}

function extractAssistantText(event) {
  if (!event || event.type !== "assistant") return "";
  const content = event.message?.content ?? [];
  let text = "";
  for (const block of content) {
    if (block?.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }
  return text;
}

function summarizeToolEvent(event) {
  if (!event || event.type !== "tool_call") return undefined;
  return {
    name: event.name,
    status: event.status,
    truncated: event.truncated,
  };
}

async function listModels() {
  const models = await Cursor.models.list({ apiKey: process.env.CURSOR_API_KEY });
  emit({ success: true, models });
}

async function runAgent(input) {
  const runtime = input.runtime || "local";
  const model = modelSelection(input);
  const createOptions = {
    apiKey: process.env.CURSOR_API_KEY,
    model,
  };

  if (runtime === "cloud") {
    createOptions.cloud = {
      repos: [
        {
          url: input.cloud_repo_url,
          ...(input.cloud_starting_ref ? { startingRef: input.cloud_starting_ref } : {}),
        },
      ],
      autoCreatePR: !!input.auto_create_pr,
    };
  } else {
    createOptions.local = { cwd: input.cwd || process.cwd() };
  }

  const agent = input.resume_agent_id
    ? await Agent.resume(input.resume_agent_id, createOptions)
    : await Agent.create(createOptions);

  try {
    const run = await agent.send(input.prompt);
    let assistantText = "";
    let eventCount = 0;
    const toolEvents = [];
    const statuses = [];

    if (run.supports?.("stream") !== false) {
      for await (const event of run.stream()) {
        eventCount += 1;
        assistantText += extractAssistantText(event);
        const toolEvent = summarizeToolEvent(event);
        if (toolEvent) toolEvents.push(toolEvent);
        if (event?.type === "status") {
          statuses.push({ status: event.status, message: event.message });
        }
      }
    }

    const result = await run.wait();
    emit({
      success: result.status === "finished",
      agent_id: agent.agentId,
      run_id: run.id,
      status: result.status,
      result: result.result || assistantText,
      model: result.model || run.model || model,
      duration_ms: result.durationMs || run.durationMs,
      git: result.git || run.git,
      event_count: eventCount,
      tool_events: toolEvents.slice(-50),
      statuses: statuses.slice(-20),
    });
  } finally {
    agent.close?.();
  }
}

async function main() {
  const raw = await readStdin();
  const input = raw.trim() ? JSON.parse(raw) : {};

  if (!process.env.CURSOR_API_KEY) {
    throw new Error("CURSOR_API_KEY is required");
  }

  if (input.action === "list_models") {
    await listModels();
    return;
  }

  await runAgent(input);
}

main().catch(error => {
  emit({
    success: false,
    error: error?.message || String(error),
    name: error?.name,
    stack: process.env.DEBUG ? error?.stack : undefined,
  });
  process.exitCode = 1;
});
