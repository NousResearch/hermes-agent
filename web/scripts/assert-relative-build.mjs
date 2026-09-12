import { readdir, readFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

// Keep generated dashboard asset URLs relative so prefixed proxies can serve
// lazy chunks, styles, and workers (#90068).
const dist = fileURLToPath(new URL("../../hermes_cli/web_dist/", import.meta.url));

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await filesUnder(path)));
    else files.push(path);
  }
  return files;
}

const files = [join(dist, "index.html"), ...(await filesUnder(dist))];
const generated = files.filter((path) => /\.(html|js|css)$/.test(path));
const rootAsset = /(?:["'`]|url\(\s*["']?)\/assets\//;
const violations = [];
for (const path of generated) {
  const contents = await readFile(path, "utf8");
  if (rootAsset.test(contents)) violations.push(relative(dist, path));
}

if (violations.length > 0) {
  console.error(
    `Root-absolute generated asset URLs found in: ${violations.join(", ")}`,
  );
  process.exit(1);
}

const jsFiles = files.filter((path) => path.endsWith(".js"));
if (jsFiles.length < 2) {
  console.error("Expected split JavaScript chunks in the dashboard build");
  process.exit(1);
}

console.log(
  `Relative dashboard build verified: ${generated.length} generated files, ` +
    `${jsFiles.length} JavaScript chunks`,
);
