const fs = require("fs");
const path = require("path");

const target = path.join(
  __dirname,
  "..",
  "node_modules",
  "gray-matter",
  "lib",
  "engines.js",
);

if (!fs.existsSync(target)) {
  console.log("[postinstall] gray-matter not installed; skipping js-yaml patch.");
  process.exit(0);
}

const original = fs.readFileSync(target, "utf8");
const patched = original
  .replace("parse: yaml.safeLoad.bind(yaml),", "parse: yaml.load.bind(yaml),")
  .replace("stringify: yaml.safeDump.bind(yaml)", "stringify: yaml.dump.bind(yaml)");

if (patched === original) {
  if (original.includes("yaml.load.bind(yaml)") && original.includes("yaml.dump.bind(yaml)")) {
    console.log("[postinstall] gray-matter already supports js-yaml 4.");
    process.exit(0);
  }
  throw new Error("gray-matter YAML engine shape changed; review js-yaml 4 patch.");
}

fs.writeFileSync(target, patched);
console.log("[postinstall] patched gray-matter for js-yaml 4.");
