/** Wire types for GET /api/model/auxiliary (built-in + plugin-registered aux slots). */

export interface AuxiliaryTaskAssignment {
  task: string;
  provider: string;
  model: string;
  base_url: string;
  /** Set only on plugin-registered tasks (PluginContext.register_auxiliary_task):
   *  the plugin's display name / description / owning plugin id. Built-in tasks
   *  are labelled client-side. Absent on older backends. */
  label?: string;
  hint?: string;
  plugin?: string;
  /** Plugin tasks only: the slot this one follows until it is pinned itself. */
  inherit_from?: string | null;
  /** Inheriting plugin tasks only: the route the task resolves to right now. */
  effective?: { provider: string; model: string; base_url: string };
}

export interface AuxiliaryModelsResponse {
  tasks: AuxiliaryTaskAssignment[];
  main: { provider: string; model: string };
}
