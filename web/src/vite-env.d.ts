/// <reference types="vite/client" />
//
// Fallback augmentation so TypeScript resolves `import.meta.env` even when
// `web/node_modules` has not been installed yet (e.g. CI lint without `npm
// install`, or IDE language server started before the first install).
//
// When `vite/client` IS installed the triple-slash reference above takes
// precedence and the declarations below merge harmlessly.

interface ImportMetaEnv {
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
  readonly MODE: string;
  readonly BASE_URL: string;
  [key: string]: string | boolean | undefined;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
