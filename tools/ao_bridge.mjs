#!/usr/bin/env node
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";

const AO_ROOT =
  process.env.AO_NODE_ROOT ||
  "/opt/homebrew/lib/node_modules/@composio/agent-orchestrator/node_modules";
const execFileAsync = promisify(execFile);

async function readStdinJSON() {
  let input = "";
  for await (const chunk of process.stdin) {
    input += chunk;
  }
  return input.trim() ? JSON.parse(input) : {};
}

async function loadAO() {
  const core = await import(
    pathToFileURL(`${AO_ROOT}/@composio/ao-core/dist/config.js`).href
  );
  const cli = await import(
    pathToFileURL(`${AO_ROOT}/@composio/ao-cli/dist/lib/create-session-manager.js`).href
  );
  return { loadConfig: core.loadConfig, getSessionManager: cli.getSessionManager };
}

function projectDefaults(config, projectId) {
  const project = config.projects?.[projectId] || {};
  const agentConfig = project.agentConfig || {};
  return {
    agent: project.agent || config.defaults?.agent || null,
    model: agentConfig.model || null,
    reasoning_effort:
      agentConfig.reasoningEffort ||
      agentConfig.reasoning_effort ||
      agentConfig.modelReasoningEffort ||
      agentConfig.model_reasoning_effort ||
      null,
  };
}

function normalizeSession(session, config) {
  if (!session) return null;
  const runtimeHandle = session.runtimeHandle || null;
  const metadata = session.metadata || {};
  const defaults = projectDefaults(config, session.projectId);
  return {
    id: session.id,
    project_id: session.projectId,
    status: session.status,
    activity: session.activity,
    branch: session.branch,
    issue_id: session.issueId,
    workspace_path: session.workspacePath,
    tmux_name: metadata.tmuxName || runtimeHandle?.id || null,
    agent: metadata.agent || defaults.agent,
    model: metadata.model || defaults.model,
    reasoning_effort:
      metadata.reasoningEffort ||
      metadata.reasoning_effort ||
      defaults.reasoning_effort,
    pr: session.pr?.url || session.pr || null,
    summary: session.agentInfo?.summary || metadata.summary || null,
    created_at: session.createdAt,
    last_activity_at: session.lastActivityAt,
    runtime_handle: runtimeHandle,
    open_command: metadata.tmuxName || runtimeHandle?.id
      ? `tmux attach -t ${metadata.tmuxName || runtimeHandle.id}`
      : null,
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function captureTmuxOutput(sessionName) {
  if (!sessionName) return "";
  try {
    const { stdout } = await execFileAsync("tmux", [
      "capture-pane",
      "-t",
      sessionName,
      "-p",
      "-S",
      "-120",
    ], { timeout: 5000 });
    return stdout || "";
  } catch {
    return "";
  }
}

async function sendTmuxEnter(sessionName) {
  if (!sessionName) return;
  try {
    await execFileAsync("tmux", ["send-keys", "-t", sessionName, "Enter"], { timeout: 5000 });
  } catch {
    // Best effort. The benchmark harness will classify delivery failure if
    // Codex never produces a worker result.
  }
}

function tmuxSessionName(session) {
  return session?.metadata?.tmuxName || session?.runtimeHandle?.id || session?.id || null;
}

async function waitForAgentReady(session) {
  const sessionName = tmuxSessionName(session);
  if (!sessionName) return;
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    const output = await captureTmuxOutput(sessionName);
    if (
      output.includes("OpenAI Codex") ||
      /\n›\s/.test(output) ||
      output.includes("model:")
    ) {
      return;
    }
    await sleep(500);
  }
}

async function main() {
  const command = process.argv[2];
  const input = await readStdinJSON();
  const { loadConfig, getSessionManager } = await loadAO();
  const config = loadConfig(input.config_path || process.env.AO_CONFIG_PATH);
  if (command === "spawn" && input.minimal_worker_prompt && input.project_id) {
    const project = config.projects?.[input.project_id];
    if (project) {
      delete project.agentRules;
      delete project.agentRulesFile;
    }
  }
  const sm = await getSessionManager(config);

  if (command === "spawn") {
    const minimalWorkerPrompt = Boolean(input.minimal_worker_prompt);
    const prompt = input.prompt || undefined;
    const session = await sm.spawn({
      projectId: input.project_id,
      issueId: input.issue_id || undefined,
      prompt: minimalWorkerPrompt ? undefined : prompt,
      branch: input.branch || undefined,
      agent: input.agent || undefined,
    });
    if (minimalWorkerPrompt && prompt) {
      await waitForAgentReady(session);
      await sm.send(session.id, prompt);
      await sleep(1000);
      await sendTmuxEnter(tmuxSessionName(session));
      const updated = await sm.get(session.id);
      console.log(JSON.stringify({ ok: true, session: normalizeSession(updated || session, config) }));
      return;
    }
    console.log(JSON.stringify({ ok: true, session: normalizeSession(session, config) }));
    return;
  }

  if (command === "status") {
    const session = await sm.get(input.session_id);
    console.log(JSON.stringify({ ok: Boolean(session), session: normalizeSession(session, config) }));
    return;
  }

  if (command === "kill") {
    await sm.kill(input.session_id);
    console.log(JSON.stringify({ ok: true, session_id: input.session_id }));
    return;
  }

  if (command === "send") {
    await sm.send(input.session_id, input.message || "");
    const session = await sm.get(input.session_id);
    console.log(JSON.stringify({ ok: true, session: normalizeSession(session, config) }));
    return;
  }

  if (command === "list") {
    const sessions = await sm.list(input.project_id || undefined);
    console.log(JSON.stringify({ ok: true, sessions: sessions.map((session) => normalizeSession(session, config)) }));
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, error: String(error?.message || error) }));
  process.exit(1);
});
