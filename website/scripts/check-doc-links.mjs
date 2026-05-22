import fs from 'node:fs';
import path from 'node:path';

const docsDir = path.resolve('docs');
const badLinkPattern = /\]\(\/docs\//g;
const offenders = [];

function walk(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const filePath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(filePath);
      continue;
    }
    if (!entry.isFile() || !entry.name.match(/\.mdx?$/)) {
      continue;
    }

    const text = fs.readFileSync(filePath, 'utf8');
    const lines = text.split(/\r?\n/);
    for (const [index, line] of lines.entries()) {
      if (badLinkPattern.test(line)) {
        offenders.push(`${path.relative(process.cwd(), filePath)}:${index + 1}: ${line.trim()}`);
      }
      badLinkPattern.lastIndex = 0;
    }
  }
}

walk(docsDir);

if (offenders.length > 0) {
  console.error('Found docs Markdown links that include the site baseUrl (/docs/).');
  console.error('Use root-relative doc paths such as /user-guide/cli so Docusaurus can localize them without /docs/<locale>/docs double-prefixes.');
  console.error(offenders.join('\n'));
  process.exit(1);
}

console.log('Docs Markdown links do not include /docs/ baseUrl prefixes.');
