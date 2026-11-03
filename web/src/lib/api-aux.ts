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

/** An auxiliary task still pinned to a provider that differs from the
 *  newly-selected main provider after a main-model switch. */
export interface StaleAuxAssignment {
  task: string;
  provider: string;
  model: string;
}

export interface ModelAssignmentResponse {
  confirm_message?: string;
  confirm_required?: boolean;
  ok: boolean;
  scope?: string;
  provider?: string;
  model?: string;
  tasks?: string[];
  reset?: boolean;
  /** Auxiliary slots still pinned to a different provider than the new main.
   *  Switching main never clears aux pins; this lets the UI warn the user
   *  their helper tasks aren't following the switch. Only set on scope:'main'. */
  stale_aux?: StaleAuxAssignment[];
  /** Nous window plans, Large model: a one-line warning (never a confirm). Absent for every other
   *  plan, provider and model. Only set on scope:'main'. */
  notice?: string;
}
