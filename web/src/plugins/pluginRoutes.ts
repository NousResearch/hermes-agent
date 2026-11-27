export function pluginRoutePaths(path: string): [string, string] {
  return [path, `${path.replace(/\/$/, "")}/*`];
}
