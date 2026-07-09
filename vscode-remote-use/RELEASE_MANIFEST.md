# Release Manifest: vscode-remote-use

Boundary file: `OPENSOURCEWARE_BOUNDARY.md`

## Summary
- Publisher: local
- License/purpose: local-first open-sourceware VS Code surface for Hermes media/primitives
- Target: VS Code ^1.85.0

## Config Surface

| Setting | Default | Purpose |
|---------|---------|---------|
| `remoteUse.hermesRepoPath` | `""` | Hermes repo root; blank disables Hermes-dependent commands |
| `remoteUse.vlcPath` | `""` | VLC binary; blank enables cross-platform auto-detect |
| `remoteUse.daoSurfacePath` | `""` | DAO surface HTML path; blank uses bundled default |

## Commands

| ID | Action |
|----|--------|
| `remoteUse.capture` | Copy active editor selection + metadata |
| `remoteUse.runTask` | Dispatch Hermes task via VS Code task API |
| `remoteUse.chat` | Open Hermes chat terminal |
| `remoteUse.mediaWindow` | Media viewport webview |
| `remoteUse.mediaCoevolve` | Ollama + FFmpeg coevolution |
| `remoteUse.reachyPanel` | Reachy primitive panel |
| `remoteUse.exportViewport` | Export media viewport HTML |
| `remoteUse.agenticHtmlViewport` | Open agentic HTML viewport |
| `remoteUse.daoBlueprint` | Open DAO blueprint |
| `remoteUse.daoSurface` | Open DAO surface HTML |
| `remoteUse.commandPrompt` | Open commandprompt.ai terminal |
| `remoteUse.homeOS` | Open home:// surface |

## QA

| Check | Status |
|-------|--------|
| Extension TS compile | pass |
| Targeted conductor/vscode pytest | 16 passed |
| Secrets scan | boundary excludes tokens/paths |
| Repo-specific paths | externalized to settings |
| VSIX package | pass |

## Artifact

| Field | Value |
|-------|-------|
| File | `vscode-remote-use-0.1.0.vsix` |
| Size | `199632` bytes |
| SHA256 | `280c7f08edaae8f9b7bd049ae2f986ffd77ba02ab61f2d52be4ae53f33ee97a4` |

## Public-Inclusion Gate Check

| Item | Included |
|------|----------|
| Secrets/tokens/keys | No |
| Repo-specific absolute paths | No; configurable |
| Internal runtime assumptions | No; graceful fallbacks |
| `OPENSOURCEWARE_BOUNDARY.md` | Yes |
| No Shebang/Per-file license headers | Yes |
| Synthetic credentials absent | Yes |

## Notes
- Default values are generic; operator-specific paths are not committed.
- Hermes-dependent commands are inert when `remoteUse.hermesRepoPath` is blank.
