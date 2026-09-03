import type { MemoryProviderSetupInfo } from "./api";

export function memorySetupHasInstallableSteps(setup?: MemoryProviderSetupInfo) {
  if (!setup) return false;
  return Boolean(
    setup.external_dependencies?.some((dependency) => dependency.installable) ||
      setup.pip_dependencies?.length,
  );
}
