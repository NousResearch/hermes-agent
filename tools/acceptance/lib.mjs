import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export const PATH_RE = /^(UT|IT|UC)-([A-Z0-9]+(?:-[A-Z0-9]+)*)-(\d{3})(?:-([A-Z]))?$/;

export function sha256(s) { return crypto.createHash('sha256').update(s, 'utf8').digest('hex'); }
export function readJson(p) { return JSON.parse(fs.readFileSync(p, 'utf8')); }
export function writeJson(p, v) { fs.mkdirSync(path.dirname(p), {recursive:true}); fs.writeFileSync(p, JSON.stringify(v, null, 2)+'\n'); }
export function walkMarkdown(input) {
  const st = fs.statSync(input);
  if (st.isFile()) return input.endsWith('.md') ? [input] : [];
  const out=[];
  for (const ent of fs.readdirSync(input, {withFileTypes:true})) {
    if (ent.name === 'node_modules' || ent.name === '.git') continue;
    const p=path.join(input, ent.name);
    if (ent.isDirectory()) out.push(...walkMarkdown(p));
    else if (ent.isFile() && ent.name.endsWith('.md')) out.push(p);
  }
  return out.sort();
}
export function parseArgs(argv) {
  const args={_:[]};
  for (let i=0;i<argv.length;i++) {
    const a=argv[i];
    if (!a.startsWith('--')) { args._.push(a); continue; }
    const k=a.slice(2);
    if (['strict','dry-run','all','no-generate','no-run'].includes(k)) { args[k]=true; continue; }
    args[k]=argv[++i];
  }
  return args;
}
export function kindOf(code) { return code.startsWith('UT-')?'unit':code.startsWith('IT-')?'integration':'use-case'; }
export function domainOf(code) { const m=code.match(PATH_RE); return m?m[2]:null; }
export function esc(s) { return JSON.stringify(String(s)); }
export function sanitizeIdent(s) { return String(s).replace(/[^A-Za-z0-9_]/g,'_').replace(/^([0-9])/,'_$1'); }
export function markerStart(code, lang='js') { return `${commentPrefix(lang)} [acceptance:body:${code}]`; }
export function markerEnd(code, lang='js') { return `${commentPrefix(lang)} [/acceptance:body:${code}]`; }
export function commentPrefix(lang) { return lang === 'python' ? '#' : '//'; }
export function extFor(framework) {
  const f=(framework||'node-test').toLowerCase();
  if (f.includes('nunit') || f.includes('c#') || f.includes('csharp')) return 'cs';
  if (f.includes('go')) return 'go';
  if (f.includes('pytest') || f.includes('python')) return 'py';
  if (f.includes('vitest') || f.includes('typescript') || f.includes('ts')) return 'ts';
  return 'mjs';
}
export function langForExt(ext) { return ext==='py'?'python':'js'; }
export function relFrom(root,p) { return path.relative(root,p).split(path.sep).join('/'); }
