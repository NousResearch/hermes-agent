/** Hide product-excluded controls without changing native dispatch. */
const hidden = new Set([
  'skills', 'reload-skills', 'reload_skills', 'learn', 'bundles', 'curator',
  'kanban', 'cron', 'blueprint', 'suggestions', 'personality'
])

export function visibleEmployeeCommand(command: string): boolean {
  return !hidden.has(command.replace(/^\//, '').split(/\s/)[0].toLowerCase())
}
