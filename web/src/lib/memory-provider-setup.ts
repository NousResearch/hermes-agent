import type { MemoryProviderSetupInfo } from "@/lib/api";

export function setupHasDetails(setup?: MemoryProviderSetupInfo): boolean {
  return Boolean(
    setup && (
      setup.external_dependencies?.length ||
      setup.python_dependencies_declared ||
      setup.pip_dependencies?.length ||
      setup.required_env?.length
    ),
  );
}

export function setupHasInstallableSteps(setup?: MemoryProviderSetupInfo): boolean {
  return Boolean(
    setup && (
      setup.external_dependencies?.some((dep) => dep.install) ||
      setup.python_dependencies_declared ||
      setup.pip_dependencies?.length
    ),
  );
}
