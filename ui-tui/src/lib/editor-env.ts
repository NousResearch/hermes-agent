import { delimiter } from 'node:path'

const HERMES_PYTHON_BIN = /\/\.hermes\/(?:tools\/python-[^/]+|installs\/[^/]+(?:\/environments\/[^/]+)?\/venv)\/bin\/?$/

/** Drop Hermes' bundled Python from an editor child so vim's embedded interpreter stays on the system one. */
export const editorChildEnv = (env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv => {
  const next: NodeJS.ProcessEnv = { ...env }

  delete next.PYTHONHOME
  delete next.PYTHONPATH
  delete next.VIRTUAL_ENV

  if (next.PATH) {
    next.PATH = next.PATH.split(delimiter)
      .filter(dir => !HERMES_PYTHON_BIN.test(dir.replace(/\\/g, '/')))
      .join(delimiter)
  }

  return next
}
