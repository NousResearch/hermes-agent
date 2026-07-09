import * as vscode from 'vscode';

type ManifestPayload = {
  input_path?: string;
  engine?: string;
  output_path?: string;
  action?: string;
  command?: string[];
  surface?: string;
  member_token?: string;
  governance?: string;
  local_only?: boolean;
  runtime?: string;
  cloud?: boolean;
  iterations?: number;
  max_iterations?: number;
};

declare const require: (module: string) => any;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getHermesRepoPath(): string {
  const repo = vscode.workspace.getConfiguration('remoteUse').get<string>('hermesRepoPath', '');
  if (typeof repo === 'string' && repo.trim()) {
    return repo.trim();
  }
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (typeof folder === 'string' && folder.trim()) {
    return folder;
  }
  return '';
}

function toPowerShellSingleQuoted(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

function launchTerminalCommand(name: string, command: string): void {
  const terminal = vscode.window.createTerminal(name);
  terminal.sendText(command);
  terminal.show();
}

function getVlcPath(): string {
  try {
    const configured = vscode.workspace.getConfiguration('remoteUse').get<string>('vlcPath', '');
    if (typeof configured === 'string' && configured.trim() && require('fs').existsSync(configured.trim())) {
      return configured.trim();
    }
  } catch {
    // ignore config read errors
  }
  const candidates = [
    'C:\\Program Files\\VideoLAN\\VLC\\vlc.exe',
    '/Applications/VLC.app/Contents/MacOS/VLC',
    '/usr/bin/vlc',
  ];
  return candidates.find((path) => require('fs').existsSync(path)) || '';
}

function openInVlc(inputPath: string): void {
  const vlcPath = getVlcPath();
  if (!vlcPath) {
    vscode.window.showWarningMessage('Remote Use: VLC not found');
    return;
  }
  const command = [
    `$vlc = ${toPowerShellSingleQuoted(vlcPath)}`,
    `if (-not (Test-Path $vlc)) { Write-Error 'VLC not found at configured path'; exit 1 }`,
    `& $vlc ${toPowerShellSingleQuoted(inputPath)}`,
  ].join('; ');
  launchTerminalCommand('Hermes Media VLC', command);
}

function transcodeToMp3(inputPath: string, outputPath: string): void {
  const command = [
    `ffmpeg -y -i ${toPowerShellSingleQuoted(inputPath)} ${toPowerShellSingleQuoted(outputPath)}`,
    `Write-Output ${toPowerShellSingleQuoted(`ffmpeg_output=${outputPath}`)}`,
  ].join('; ');
  launchTerminalCommand('Hermes Media FFmpeg', command);
}

function runHermesMediaAction(action: 'run' | 'audit', manifestPath: string): void {
  const repo = getHermesRepoPath();
  const command = [
    `Set-Location ${toPowerShellSingleQuoted(repo)}`,
    `hermes media ${action} --manifest ${toPowerShellSingleQuoted(manifestPath)}`,
  ].join('; ');
  launchTerminalCommand(`Hermes Media ${action}`, command);
}

function runHermesMediaCoevolve(manifest: ManifestPayload, goal: string): void {
  const inputPath = String(manifest.input_path || '');
  const outputPath = String(manifest.output_path || '');
  if (!inputPath || !outputPath) {
    vscode.window.showErrorMessage('Manifest requires input_path and output_path for coevolve.');
    return;
  }

  const memberToken = String(manifest.member_token || '+æ');
  const governance = String(manifest.governance || 'none');
  const surface = String(manifest.surface || 'ae://HERMES-AGENT^media');
  const repo = getHermesRepoPath();
  const command = [
    `Set-Location ${toPowerShellSingleQuoted(repo)}`,
    `hermes media coevolve --goal ${toPowerShellSingleQuoted(goal)} --input ${toPowerShellSingleQuoted(inputPath)} --output ${toPowerShellSingleQuoted(outputPath)} --codec h264_nvenc --surface ${toPowerShellSingleQuoted(surface)} --member-token ${toPowerShellSingleQuoted(memberToken)} --governance ${toPowerShellSingleQuoted(governance)}`,
  ].join('; ');

  launchTerminalCommand('Hermes Media Coevolve', command);
}

function inferMediaKind(filePath: string): 'audio' | 'video' | undefined {
  const ext = filePath.toLowerCase().split('.').pop() || '';
  const audioExt = new Set(['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac']);
  const videoExt = new Set(['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v']);
  if (audioExt.has(ext)) {
    return 'audio';
  }
  if (videoExt.has(ext)) {
    return 'video';
  }
  return undefined;
}

function collectPreviewEntries(panel: vscode.WebviewPanel, manifest: ManifestPayload): Array<{ label: string; kind: 'audio' | 'video'; uri: string }> {
  const fs = require('fs');
  const entries: Array<{ label: string; kind: 'audio' | 'video'; uri: string }> = [];
  const candidates: Array<{ label: string; path?: string }> = [
    { label: 'Input Preview', path: manifest.input_path },
    { label: 'Output Preview', path: manifest.output_path },
  ];

  for (const candidate of candidates) {
    const p = String(candidate.path || '').trim();
    if (!p || !fs.existsSync(p)) {
      continue;
    }
    const kind = inferMediaKind(p);
    if (!kind) {
      continue;
    }
    const webviewUri = panel.webview.asWebviewUri(vscode.Uri.file(p)).toString();
    entries.push({ label: candidate.label, kind, uri: webviewUri });
  }

  return entries;
}

function encodeUtf8Bytes(text: string): Uint8Array {
  const { Buffer } = require('buffer');
  return Uint8Array.from(Buffer.from(text, 'utf8'));
}

function decodeUtf8Bytes(raw: Uint8Array): string {
  const { Buffer } = require('buffer');
  return Buffer.from(raw).toString('utf8');
}

async function readManifest(uri: vscode.Uri): Promise<any> {
  const raw = await vscode.workspace.fs.readFile(uri);
  const text = decodeUtf8Bytes(raw);
  return JSON.parse(text);
}

async function openMediaWindowFromUri(uri: vscode.Uri): Promise<void> {
  const manifest = await readManifest(uri);
  await openMediaWindowFromManifest(manifest, uri.fsPath);
}

async function openMediaWindowFromManifest(manifest: ManifestPayload, manifestPath: string): Promise<void> {
  const fs = require('fs');
  const localRoots: vscode.Uri[] = [];
  const candidatePaths = [manifest.input_path, manifest.output_path, manifestPath];
  for (const candidate of candidatePaths) {
    const p = String(candidate || '').trim();
    if (!p) {
      continue;
    }
    const dir = fs.existsSync(p) && fs.statSync(p).isDirectory() ? p : require('path').dirname(p);
    if (!dir) {
      continue;
    }
    localRoots.push(vscode.Uri.file(dir));
  }

  const panel = vscode.window.createWebviewPanel(
    'remoteUseMediaWindow',
    'Remote Use: Media Window',
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      localResourceRoots: localRoots,
    }
  );
  const previews = collectPreviewEntries(panel, manifest);
  panel.webview.html = renderMediaViewportHtml(manifest, manifestPath, previews);

  panel.webview.onDidReceiveMessage((message: { type?: string; manifestPath?: string; goal?: string }) => {
    const effectiveManifestPath = message?.manifestPath || manifestPath;

    if (message?.type === 'openVlc') {
      const inputPath = String((manifest as ManifestPayload)?.input_path || '');
      if (!inputPath) {
        vscode.window.showErrorMessage('Manifest has no input_path for VLC playback.');
        return;
      }
      openInVlc(inputPath);
      return;
    }

    if (message?.type === 'runManifest') {
      runHermesMediaAction('run', effectiveManifestPath);
      return;
    }

    if (message?.type === 'transcodeMp3') {
      const inputPath = String((manifest as ManifestPayload)?.input_path || '');
      if (!inputPath) {
        vscode.window.showErrorMessage('Manifest has no input_path for FFmpeg transcode.');
        return;
      }
      const outputPath = String((manifest as ManifestPayload)?.output_path || `${inputPath}.transcoded.mp3`);
      transcodeToMp3(inputPath, outputPath);
      return;
    }

    if (message?.type === 'coevolve') {
      const goal = String(message?.goal || 'optimize for mobile shortform');
      runHermesMediaCoevolve(manifest, goal);
      return;
    }

    if (message?.type === 'auditManifest') {
      runHermesMediaAction('audit', effectiveManifestPath);
      return;
    }

    if (message?.type === 'refreshManifest') {
      void openMediaWindowFromUri(vscode.Uri.file(effectiveManifestPath));
    }
  });
}

async function findDefaultManifestUri(): Promise<vscode.Uri | undefined> {
  const env = (globalThis as any)?.process?.env;
  const localAppData = typeof env?.LOCALAPPDATA === 'string' ? env.LOCALAPPDATA : '';
  if (!localAppData) {
    return undefined;
  }

  const candidates = [
    `${localAppData}\\hermes\\media\\manifests\\private-client-glocal-smoke.json`,
    `${localAppData}\\hermes\\media\\manifests\\mp3-window-test.json`,
    `${localAppData}\\hermes\\media\\manifests\\media-window-sample.json`,
    `${localAppData}\\hermes\\media\\manifests\\smoke-media.json`,
  ];

  for (const filePath of candidates) {
    const uri = vscode.Uri.file(filePath);
    try {
      await vscode.workspace.fs.stat(uri);
      return uri;
    } catch {
      // Skip missing files and continue checking defaults.
    }
  }

  return undefined;
}

async function launchDefaultMediaWindow(): Promise<boolean> {
  const defaultUri = await findDefaultManifestUri();
  if (!defaultUri) {
    return false;
  }

  try {
    await openMediaWindowFromUri(defaultUri);
    return true;
  } catch {
    return false;
  }
}

function buildPlayerManifestFromInput(inputPath: string, localAppData: string): { path: string; payload: ManifestPayload } {
  const now = new Date();
  const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
  const manifestPath = `${localAppData}\\hermes\\media\\manifests\\vscode-player-${stamp}.json`;
  const outputPath = `${localAppData}\\hermes\\media\\outputs\\vscode-player-${stamp}.mp3`;
  const payload: ManifestPayload = {
    action: 'vscode-vlc-ffmpeg-player',
    engine: 'ffmpeg',
    surface: 'æ://private client^glocal',
    runtime: 'ae://local^ollama',
    local_only: true,
    cloud: false,
    input_path: inputPath,
    output_path: outputPath,
    command: ['ffmpeg', '-y', '-i', inputPath, outputPath],
  };
  return { path: manifestPath, payload };
}

function renderMediaViewportHtml(manifest: any, manifestPath: string, previews: Array<{ label: string; kind: 'audio' | 'video'; uri: string }> = []): string {
  const rawManifest = JSON.stringify(manifest, null, 2);
  const embeddedPreviewHtml = previews.length > 0
    ? previews.map((entry) => {
        const tag = entry.kind === 'audio'
          ? `<audio controls style="width:100%;"><source src="${escapeHtml(entry.uri)}" /></audio>`
          : `<video controls style="width:100%;max-height:360px;background:#000;"><source src="${escapeHtml(entry.uri)}" /></video>`;
        return `
          <div class="row" style="margin-top: 10px;">
            <div class="k">${escapeHtml(entry.label)}</div>
            ${tag}
          </div>
        `;
      }).join('')
    : '<div class="row"><span class="k">No local previewable media found for this manifest.</span></div>';

  const title = String(manifest?.action || 'Media Manifest');
  const engine = String(manifest?.engine || 'unknown');
  const memberToken = String(manifest?.member_token || '');
  const governance = String(manifest?.governance || 'none');
  const command = Array.isArray(manifest?.command) ? manifest.command.join(' ') : '';
  const manifestHash = String(manifest?.manifest_sha256 || '');
  const inputSha = String(manifest?.input_sha256 || '');

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Hermes Media Viewport</title>
    <style>
      :root { --bg:#050505; --panel:#111; --text:#D4AF37; --muted:#8b8b8b; }
      body { font-family: "Orbitron", "JetBrains Mono", monospace; margin: 0; padding: 16px; background: var(--bg); color: var(--text); }
      .card { background: var(--panel); border: 1px solid #222; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
      h1 { margin: 0 0 10px; font-size: 18px; }
      h2 { margin: 0 0 8px; font-size: 14px; color: var(--muted); }
      .row { margin: 5px 0; }
      .k { color: var(--muted); }
      .v { color: var(--text); word-break: break-all; }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; color: #f0f0f0; }
      button { background: transparent; color: var(--text); border: 1px solid var(--text); border-radius: 6px; padding: 8px 10px; cursor: pointer; }
      button:hover { background: rgba(212,175,55,0.1); }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Hermes Media Viewport</h1>
      <div class="row"><span class="k">Action:</span> <span class="v">${escapeHtml(title)}</span></div>
      <div class="row"><span class="k">Engine:</span> <span class="v">${escapeHtml(engine)}</span></div>
      <div class="row"><span class="k">Member Token:</span> <span class="v">${escapeHtml(memberToken)}</span></div>
      <div class="row"><span class="k">Governance:</span> <span class="v">${escapeHtml(governance)}</span></div>
    </div>

    <div class="card">
      <h2>Execution</h2>
      <pre>${escapeHtml(command)}</pre>
    </div>

    <div class="card">
      <h2>Manifest</h2>
      <pre>${escapeHtml(rawManifest)}</pre>
    </div>

    <div class="card">
      <h2>Integrity</h2>
      <div class="row"><span class="k">Manifest SHA256:</span> <span class="v">${escapeHtml(manifestHash)}</span></div>
      <div class="row"><span class="k">Input SHA256:</span> <span class="v">${escapeHtml(inputSha)}</span></div>
    </div>
  </body>
</html>`;
}

function renderReachyPanelHtml(repo: string): string {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Remote Use: Reachy Primitive</title>
    <style>
      body { font-family: Segoe UI, sans-serif; margin: 0; padding: 16px; background: #0e1116; color: #e6edf3; }
      .card { background: #161b22; border: 1px solid #2d333b; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
      h1 { margin: 0 0 10px; font-size: 18px; }
      h2 { margin: 0 0 8px; font-size: 14px; color: #9da7b3; }
      .row { margin: 5px 0; }
      .k { color: #8b949e; }
      .v { color: #e6edf3; word-break: break-all; }
      button { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 8px 10px; cursor: pointer; }
      button:hover { background: #30363d; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Remote Use: Reachy Primitive</h1>
      <div class="row"><span class="k">Repo:</span> <span class="v">${escapeHtml(repo)}</span></div>
      <div class="row"><span class="k">Primitive:</span> <span class="v">reachy</span></div>
      <div class="row"><span class="k">Status:</span> <span class="v">onboarded</span></div>
    </div>

    <div class="card">
      <h2>Actions</h2>
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <button id="requestProbe">Reachy Probe</button>
        <button id="requestMediaWindow">Open Media Window</button>
      </div>
    </div>

    <script>
      const vscode = acquireVsCodeApi();
      document.getElementById('requestProbe')?.addEventListener('click', () => {
        vscode.postMessage({ type: 'requestProbe' });
      });
      document.getElementById('requestMediaWindow')?.addEventListener('click', () => {
        vscode.postMessage({ type: 'requestMediaWindow' });
      });
    </script>
  </body>
</html>`;
}

function renderAgenticHtmlViewportHtml(templatePath: string): string {
  try {
    return require('fs').readFileSync(templatePath, 'utf8');
  } catch {
    return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Remote Use: Agentic HTML Viewport</title>
    <style>
      body { font-family: Segoe UI, sans-serif; margin: 16px; background: #0e1116; color: #e6edf3; }
      .card { background: #161b22; border: 1px solid #2d333b; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Agentic HTML Viewport</h1>
      <p>Template not found at: ${escapeHtml(templatePath)}</p>
    </div>
  </body>
</html>`;
  }
}

function renderDaoBlueprintHtml(repo: string): string {
  const path = require('path');
  const blueprintPath = path.join(repo, 'blueprints', 'laptop-as-dao.md');
  let markdown = '';
  try {
    markdown = require('fs').readFileSync(blueprintPath, 'utf8');
  } catch {
    markdown = `# DAO Blueprint Not Found\n\nExpected path: ${blueprintPath}`;
  }

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Remote Use: DAO Blueprint</title>
    <style>
      body { font-family: Segoe UI, sans-serif; margin: 0; padding: 16px; background: #0e1116; color: #e6edf3; }
      .card { background: #161b22; border: 1px solid #2d333b; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
      h1 { margin: 0 0 10px; font-size: 18px; }
      .k { color: #8b949e; }
      pre { white-space: pre-wrap; word-break: break-word; margin: 0; color: #c9d1d9; line-height: 1.4; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>Remote Use: DAO Blueprint</h1>
      <div class="k">Source: ${escapeHtml(blueprintPath)}</div>
    </div>
    <div class="card">
      <pre>${escapeHtml(markdown)}</pre>
    </div>
  </body>
</html>`;
}

function renderDaoSurfaceHtml(): string {
  let daoPath = 'C:\\æ\\agentic-entrepreneurship\\dao\\index.html';
  try {
    const configured = vscode.workspace.getConfiguration('remoteUse').get<string>('daoSurfacePath', '');
    if (typeof configured === 'string' && configured.trim()) {
      daoPath = configured.trim();
    }
  } catch {
    // ignore config read errors and use default
  }
  let html = '';
  try {
    html = require('fs').readFileSync(daoPath, 'utf8');
  } catch {
    html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Remote Use: DAO Surface</title>
    <style>
      body { font-family: Segoe UI, sans-serif; margin: 16px; background: #0e1116; color: #e6edf3; }
      .card { background: #161b22; border: 1px solid #2d333b; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>DAO Surface</h1>
      <p>Could not load DAO surface at: ${escapeHtml(daoPath)}</p>
    </div>
  </body>
</html>`;
  }

  return html;
}

function renderHomeOS(): string {
  const repo = getHermesRepoPath();
  const surfaces = [
    { kind: 'terminal', command: 'remoteUse.commandPrompt', detail: 'Hermes terminal profile', surface: 'terminal' },
    { kind: 'editor', command: 'remoteUse.editor', detail: 'VS Code editor control', surface: 'editor' },
    { kind: 'files', command: 'remoteUse.files', detail: 'Bounded filesystem primitive', surface: 'files' },
    { kind: 'victus', command: 'remoteUse.victus', detail: 'Victus machine vitals', surface: 'victus' },
    { kind: 'nvidia', command: 'remoteUse.nvidia', detail: 'NVIDIA GPU C2', surface: 'nvidia' },
    { kind: 'vlc', command: 'remoteUse.vlc', detail: 'VLC fleet command', surface: 'vlc' },
    { kind: 'ffmpeg', command: 'remoteUse.ffmpeg', detail: 'FFmpeg media pipeline', surface: 'ffmpeg' },
    { kind: 'qr', command: 'remoteUse.qr', detail: 'QR-tagged HTML execution', surface: 'qr' },
    { kind: 'mesh', command: 'remoteUse.mesh', detail: 'Bounded sovereign mesh', surface: 'mesh' },
  ];

  const buttons = surfaces
    .map((s) => `<button data-command="${escapeHtml(s.command)}">${escapeHtml(s.kind)} · ${escapeHtml(s.detail)}</button>`)
    .join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>home://</title>
<style>
:root { --bg:#050505; --bg-panel:#0a0a0a; --gold:#D4AF37; --gold-60:rgba(212,175,55,0.6); --text:#e6e6e6; --muted:#8a8a8a; --border:#1a1a1a; --ff:'Orbitron','JetBrains Mono','Segoe UI',monospace; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text); font-family: var(--ff); }
body { padding: 24px; }
.header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
.title { font-size: 18px; font-weight: 700; color: var(--gold); }
.subtitle { font-size: 12px; color: var(--muted); text-transform: lowercase; letter-spacing: 0.08em; }
.surfaces { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
button { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: left; cursor: pointer; display: flex; flex-direction: column; gap: 6px; transition: border-color 0.2s; color: var(--text); font-family: var(--ff); font-size: 12px; }
button:hover { border-color: var(--gold-60); }
</style>
</head>
<body>
<div class="header"><div class="title">home://</div><div class="subtitle">agentic OS surface</div></div>
<div class="surfaces">${buttons}</div>
<script>
const vscode = acquireVsCodeApi();
document.querySelectorAll('button').forEach((btn) => {
  btn.addEventListener('click', () => vscode.postMessage({ command: btn.getAttribute('data-command') }));
});
</script>
</body>
</html>`;
}

export function activate(context: vscode.ExtensionContext) {
  const captureCmd = vscode.commands.registerCommand('remoteUse.capture', async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showInformationMessage('Remote Use: no active text editor to capture');
      return;
    }
    const text = editor.document.getText();
    const selection = editor.selection;
    const path = editor.document.uri.toString();
    const surface = {
      kind: 'remote_user_capture',
      path,
      selection: `${selection.start.line}:${selection.start.character}-${selection.end.line}:${selection.end.character}`,
      length: text.length,
      preview: text.slice(0, 2000),
    };
    await vscode.env.clipboard.writeText(JSON.stringify(surface, null, 2));
    vscode.window.showInformationMessage(`Remote Use: captured ${surface.length} chars`);
  });

  const runTaskCmd = vscode.commands.registerCommand('remoteUse.runTask', async () => {
    const repo = getHermesRepoPath();
    const options = { cwd: repo } as const;
    const definition: vscode.TaskDefinition = { type: 'remoteUse', task: 'run' };
    const task = new vscode.Task(definition, vscode.TaskScope.Workspace, 'Hermes Task', 'RemoteUse', new vscode.ShellExecution('hermes', ['hello'], options));
    await vscode.tasks.executeTask(task);
    vscode.window.showInformationMessage('Remote Use: dispatched Hermes task');
  });

  const chatCmd = vscode.commands.registerCommand('remoteUse.chat', async () => {
    await vscode.window.showInputBox({ placeHolder: 'Say something to Hermes...' }).then(async (value) => {
      if (!value) {
        return;
      }
      const terminal = vscode.window.createTerminal('Hermes Chat');
      terminal.sendText(`hermes chat "${value.replace(/"/g, '\\"')}"`);
      terminal.show();
    });
  });

  const mediaWindowCmd = vscode.commands.registerCommand('remoteUse.mediaWindow', async () => {
    const repo = getHermesRepoPath();
    const knownManifestCandidates = [
      `${repo}/media/vlc_manifest.json`,
      `${repo}/media/ffmpeg_manifest.json`,
    ];
    const knownMediaCandidates = [
      `${repo}/media/videos/private_client_super_agent_computer_use/out/private_client_super_agent_computer_use_draft.mp4`,
      `${repo}/media/videos/private_client_super_agent_computer_use/480p15/Scene0_Origin.mp4`,
      `${repo}/media/videos/private_client_super_agent_computer_use/480p15/Scene5_EndCard.mp4`,
    ];

    const foundManifest = knownManifestCandidates.find((p) => require('fs').existsSync(p));
    const foundMedia = knownMediaCandidates.find((p) => require('fs').existsSync(p));

    if (foundManifest) {
      try {
        const manifest = require(foundManifest);
        const manifestUri = vscode.Uri.file(foundManifest);
        await openMediaWindowFromManifest(manifest, manifestUri.fsPath);
        return;
      } catch {
        // fall through to media/webview fallback
      }
    }

    if (foundMedia) {
      const fallback: ManifestPayload = {
        action: 'vscode-player-fallback',
        input_path: foundMedia,
        output_path: foundMedia,
        engine: 'vscode-webview',
      };
      await openMediaWindowFromManifest(fallback, foundMedia);
      return;
    }

    const defaultUri = await findDefaultManifestUri();
    if (defaultUri) {
      await openMediaWindowFromUri(defaultUri);
      return;
    }

    vscode.window.showWarningMessage('Remote Use: media window unavailable; no manifest or media file found.');
  });

  const mediaOpenVlcCmd = vscode.commands.registerCommand('remoteUse.mediaOpenVlc', async () => {
    const picked = await vscode.window.showOpenDialog({
      canSelectMany: false,
      canSelectFiles: true,
      canSelectFolders: false,
      filters: { 'JSON': ['json'] },
      openLabel: 'Select Manifest For VLC',
    });
    if (!picked || picked.length === 0) {
      return;
    }

    try {
      const manifest = await readManifest(picked[0]);
      const inputPath = String((manifest as ManifestPayload)?.input_path || '');
      if (!inputPath) {
        vscode.window.showErrorMessage('Manifest has no input_path for VLC playback.');
        return;
      }
      openInVlc(inputPath);
    } catch {
      vscode.window.showErrorMessage('Selected file is not a valid media manifest.');
    }
  });

  const mediaRunManifestCmd = vscode.commands.registerCommand('remoteUse.mediaRunManifest', async () => {
    const repo = getHermesRepoPath();
    const candidates = [`${repo}/media/vlc_manifest.json`, `${repo}/media/ffmpeg_manifest.json`];
    for (const candidate of candidates) {
      if (require('fs').existsSync(candidate)) {
        const manifest = require(candidate);
        vscode.window.showInformationMessage(`Remote Use: manifest loaded: ${manifest.manifest || candidate}`);
        return;
      }
    }
    vscode.window.showWarningMessage('Remote Use: no manifest found');
  });

  const mediaPlayerCmd = vscode.commands.registerCommand('remoteUse.mediaPlayer', async () => {
    await vscode.window.showQuickPick(['vlc', 'ffmpeg', 'omxplayer'], { placeHolder: 'Select media runtime' }).then((choice) => {
      if (!choice) {
        return;
      }
      const terminal = vscode.window.createTerminal(`Remote Use: ${choice}`);
      terminal.sendText(`${choice} --version`);
      terminal.show();
    });
  });

  const mediaCoevolveCmd = vscode.commands.registerCommand('remoteUse.mediaCoevolve', async () => {
    const repo = getHermesRepoPath();
    const goal = await vscode.window.showInputBox({ placeHolder: 'optimize for mobile shortform' });
    if (!goal) {
      return;
    }
    const terminal = vscode.window.createTerminal('Remote Use: Coevolve');
    terminal.sendText(`Set-Location ${toPowerShellSingleQuoted(repo)}; hermes coevolve --target=${goal}`);
    terminal.show();
  });

  const reachyProbeCmd = vscode.commands.registerCommand('remoteUse.reachyProbe', async () => {
    const repo = getHermesRepoPath();
    const candidates = [`${repo}/tools/reachy/reachy_probe.py`, `${repo}/tools/reachy_operator_bridge.py`];
    let cmd = '';
    for (const script of candidates) {
      if (require('fs').existsSync(script)) {
        cmd = `python "${script}" probe`;
        break;
      }
    }
    if (!cmd) {
      cmd = 'echo "Reachy primitive onboarded; driver/runtime contact not yet configured."';
    }
    const terminal = vscode.window.createTerminal('Remote Use: Reachy Probe');
    terminal.sendText(`Set-Location ${toPowerShellSingleQuoted(repo)}; ${cmd}`);
    terminal.show();
  });

  const reachyPanelCmd = vscode.commands.registerCommand('remoteUse.reachyPanel', async () => {
    const repo = getHermesRepoPath();
    const panel = vscode.window.createWebviewPanel(
      'remoteUseReachyPanel',
      'Remote Use: Reachy Primitive',
      vscode.ViewColumn.Beside,
      { enableScripts: true }
    );
    panel.webview.html = renderReachyPanelHtml(repo);
    panel.webview.onDidReceiveMessage(async (message: any) => {
      if (message?.type === 'requestProbe') {
        await vscode.commands.executeCommand('remoteUse.reachyProbe');
      }
      if (message?.type === 'requestMediaWindow') {
        await vscode.commands.executeCommand('remoteUse.mediaWindow');
      }
    });
  });

  const agenticHtmlViewportCmd = vscode.commands.registerCommand('remoteUse.agenticHtmlViewport', async () => {
    const repo = getHermesRepoPath();
    const templatePath = `${repo}/templates/agentic.html`;
    const panel = vscode.window.createWebviewPanel(
      'remoteUseAgenticHtmlViewport',
      'Remote Use: Agentic HTML Viewport',
      vscode.ViewColumn.Beside,
      { enableScripts: true, localResourceRoots: [vscode.Uri.file(`${repo}/templates`)] }
    );
    panel.webview.html = renderAgenticHtmlViewportHtml(templatePath);
  });

  const daoBlueprintCmd = vscode.commands.registerCommand('remoteUse.daoBlueprint', async () => {
    const repo = getHermesRepoPath();
    const panel = vscode.window.createWebviewPanel(
      'remoteUseDaoBlueprint',
      'Remote Use: DAO Blueprint',
      vscode.ViewColumn.Beside,
      { enableScripts: false }
    );
    panel.webview.html = renderDaoBlueprintHtml(repo);
  });

  const daoSurfaceCmd = vscode.commands.registerCommand('remoteUse.daoSurface', async () => {
    const panel = vscode.window.createWebviewPanel(
      'remoteUseDaoSurface',
      'Remote Use: DAO Surface',
      vscode.ViewColumn.One,
      { enableScripts: true }
    );
    panel.webview.html = renderDaoSurfaceHtml();
  });

  const commandPromptCmd = vscode.commands.registerCommand('remoteUse.commandPrompt', async () => {
    const repo = getHermesRepoPath();
    const psProfilePath = require('path').join(repo, 'shell', '_commandPrompt.ps1');
    const terminal = vscode.window.createTerminal('commandprompt.ai');
    terminal.sendText(`Set-Location ${toPowerShellSingleQuoted(repo)}; if (Test-Path ${toPowerShellSingleQuoted(psProfilePath)}) { . ${toPowerShellSingleQuoted(psProfilePath)} }; $env:HERMES_REPO=${toPowerShellSingleQuoted(repo)}`);
    terminal.show();
  });

  const homeOSCmd = vscode.commands.registerCommand('remoteUse.homeOS', async () => {
    const panel = vscode.window.createWebviewPanel(
      'remoteUseHomeOS',
      'Remote Use: home://',
      vscode.ViewColumn.One,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    panel.webview.html = renderHomeOS();
    panel.webview.onDidReceiveMessage(async (message) => {
      if (!message?.command) {
        return;
      }
      switch (message.command) {
        case 'remoteUse.commandPrompt':
          await vscode.commands.executeCommand('remoteUse.commandPrompt');
          break;
        case 'remoteUse.editor': {
          const repo = getHermesRepoPath();
          await vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(repo), true);
          break;
        }
        case 'remoteUse.files': {
          const repo = getHermesRepoPath();
          await vscode.env.openExternal(vscode.Uri.file(repo));
          break;
        }
        case 'remoteUse.victus': {
          const terminal = vscode.window.createTerminal('Victus');
          terminal.sendText(`Set-Location ${toPowerShellSingleQuoted(getHermesRepoPath())}; hermes conductor +æ://victus`);
          terminal.show();
          break;
        }
        case 'remoteUse.nvidia': {
          const terminal = vscode.window.createTerminal('NVIDIA');
          terminal.sendText(`Set-Location ${toPowerShellSingleQuoted(getHermesRepoPath())}; hermes conductor NVIDIA://status`);
          terminal.show();
          break;
        }
        case 'remoteUse.vlc':
          await vscode.commands.executeCommand('remoteUse.mediaOpenVlc');
          break;
        case 'remoteUse.ffmpeg': {
          const terminal = vscode.window.createTerminal('FFmpeg');
          terminal.sendText('ffmpeg -version');
          terminal.show();
          break;
        }
        case 'remoteUse.qr': {
          const input = await vscode.window.showInputBox({ placeHolder: '<html>...</html>' });
          if (input) {
            await vscode.commands.executeCommand('remoteUse.commandPrompt');
            const terminal = vscode.window.activeTerminal;
            terminal?.sendText(`hermes conductor +æ://qrcode payload ${toPowerShellSingleQuoted(input)}`);
          }
          break;
        }
        case 'remoteUse.mesh':
          await vscode.window.showInformationMessage('Mesh surface: pc://mesh/victus/local');
          break;
        default:
          if (typeof message.command === 'string' && message.command.startsWith('remoteUse.')) {
            await vscode.commands.executeCommand(message.command);
          }
          break;
      }
    });
  });

  context.subscriptions.push(
    captureCmd,
    runTaskCmd,
    chatCmd,
    mediaWindowCmd,
    mediaOpenVlcCmd,
    mediaRunManifestCmd,
    mediaPlayerCmd,
    mediaCoevolveCmd,
    reachyProbeCmd,
    reachyPanelCmd,
    agenticHtmlViewportCmd,
    daoBlueprintCmd,
    daoSurfaceCmd,
    commandPromptCmd,
    homeOSCmd
  );

  void launchDefaultMediaWindow();
}

export function deactivate() {
  // Cleanup/noop boundary for future extension state.
}
