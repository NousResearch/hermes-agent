// Managed-files API types for the dashboard Files page. Moved out of api.ts so
// that facade stays under its code-health line cap (the interfaces keep their
// own cap here, per AGENTS.md § code health ratchet).

export interface ManagedFileEntry {
  name: string;
  path: string;
  is_directory: boolean;
  broken_link?: boolean;
  size: number | null;
  mtime: number | null;
  mime_type: string | null;
}

export interface ManagedFilesResponse {
  root: string | null;
  path: string;
  parent: string | null;
  locked_root: string | null;
  can_change_path: boolean;
  entries: ManagedFileEntry[];
}

export interface ManagedFileReadResponse {
  name: string;
  path: string;
  size: number;
  mime_type: string;
  data_url: string;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}

export interface ManagedFileWriteResponse {
  ok: boolean;
  path: string;
  entry: ManagedFileEntry;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}
