# OpenSourceWare Boundary

This file marks the opensourceware release boundary for `vscode-remote-use/`.

## Included
- VS Code extension source under `src/`
- Extension manifest (`package.json`)
- TypeScript configuration (`tsconfig.json`)
- Public documentation (`README.md`)

## Excluded
- Secrets, tokens, API keys, auth files
- Internal runtime assumptions not safely generalizable
- Repo-specific absolute paths (`C:\\æ\\hermes-fork`, OneDrive paths, local media output paths)
- Internal-only commands or surfaces that depend on private Hermes runtime
- Scalar-supremacy or mesh governance configs that are private by design

## Notes
- Settings prefixed with `remoteUse.` are intended as user-configurable.
- Defaults should be generic; avoid embedding operator-specific paths.
- If a primitive depends on local Hermes conductor, it must degrade gracefully when unavailable.
