// Remark plugin: make relative Markdown file links resolve across the zh-Hans
// fallback boundary.
//
// Docs are authored with relative file links (`../user-guide/profiles.md`) so
// they also open on GitHub (#114428). Docusaurus resolves `./x.md` and
// `../x.md` only against the *source file's own directory*. In a localized
// build, an untranslated EN page (served as fallback) linking to a translated
// page — or the reverse — then fails to resolve, because the target's source
// lives in the other content root. A content-root-absolute file link
// (`/user-guide/profiles.md`) is resolved against every content path
// (i18n/<locale>/.../current first, then docs/), so this plugin rewrites the
// relative form to that form before Docusaurus's own link resolution runs.
//
// Only links with a Markdown extension are touched; anything else (routes,
// assets, external URLs) passes through untouched.

const path = require('path');

const MARKDOWN_LINK = /^(\.\.?\/[^?#]*\.mdx?)([?#].*)?$/i;
const LOCALIZED_ROOT = /^(.*\/i18n\/[^/]+\/docusaurus-plugin-content-docs\/[^/]+)\//;

function contentRoot(sourceFilePath, siteDir) {
  const posixPath = sourceFilePath.split(path.sep).join('/');
  const localized = LOCALIZED_ROOT.exec(posixPath);
  if (localized) {
    return localized[1];
  }
  return path.posix.join(siteDir.split(path.sep).join('/'), 'docs');
}

function walk(node, visitor) {
  if (!node || typeof node !== 'object') {
    return;
  }
  if (node.type === 'link' || node.type === 'definition') {
    visitor(node);
  }
  if (Array.isArray(node.children)) {
    node.children.forEach((child) => walk(child, visitor));
  }
}

function plugin(options = {}) {
  const siteDir = options.siteDir || process.cwd();
  return (root, file) => {
    if (!file.path) {
      return;
    }
    const sourceDir = path.posix.dirname(file.path.split(path.sep).join('/'));
    const root_ = contentRoot(file.path, siteDir);
    walk(root, (node) => {
      const match = MARKDOWN_LINK.exec(node.url || '');
      if (!match) {
        return;
      }
      const resolved = path.posix.normalize(path.posix.join(sourceDir, match[1]));
      const relToRoot = path.posix.relative(root_, resolved);
      if (relToRoot.startsWith('../')) {
        return; // points outside the docs tree; leave it to Docusaurus
      }
      node.url = '/' + relToRoot + (match[2] || '');
    });
  };
}

module.exports = plugin;
