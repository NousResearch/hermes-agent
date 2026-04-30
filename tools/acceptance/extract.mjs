#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import {parseArgs, walkMarkdown, PATH_RE, sha256, writeJson} from './lib.mjs';

function parseFile(file) {
  const lines=fs.readFileSync(file,'utf8').split(/\r?\n/);
  const scenarios=[]; const warnings=[];
  let req=null, reqLine=0, cur=null;
  function finish(endLine) {
    if (!cur) return;
    const pcLine=cur.lines.find(x=>/^\s*-\s*Path Code\s*:/i.test(x.text));
    let code=null;
    if (pcLine) {
      code=pcLine.text.replace(/^\s*-\s*Path Code\s*:\s*/i,'').trim();
      if (!PATH_RE.test(code)) warnings.push({type:'invalid_path_code', file, line:pcLine.n, message:`Invalid Path Code ${code}`});
    } else warnings.push({type:'missing_path_code', file, line:cur.line, message:`Scenario lacks Path Code: ${cur.scenario}`});
    const steps=cur.lines.filter(x=>/^\s*-\s*(GIVEN|WHEN|THEN|AND)\b/i.test(x.text)).map(x=>x.text.replace(/^\s*-\s*/,'').trim());
    if (code && PATH_RE.test(code)) scenarios.push({
      pathCode:code, requirement:req?.text||'', scenario:cur.scenario, steps,
      source:{file,line:cur.line,endLine},
      fingerprint:sha256([req?.text||'', cur.scenario, ...steps].join('\n'))
    });
  }
  for (let i=0;i<lines.length;i++) {
    const n=i+1, line=lines[i];
    const rm=line.match(/^###\s+Requirement:\s*(.+?)\s*$/);
    if (rm) { finish(n-1); cur=null; req={text:rm[1], line:n}; reqLine=n; continue; }
    const sm=line.match(/^####\s+Scenario:\s*(.+?)\s*$/);
    if (sm) { finish(n-1); cur={scenario:sm[1], line:n, lines:[]}; continue; }
    if (cur) cur.lines.push({n,text:line});
  }
  finish(lines.length);
  return {scenarios,warnings};
}

const args=parseArgs(process.argv.slice(2));
const input=args._[0];
if (!input) { console.error('usage: node extract.mjs <spec.md|dir> [--out path] [--strict]'); process.exit(64); }
let scenarios=[]; let warnings=[];
for (const f of walkMarkdown(input)) { const r=parseFile(f); scenarios.push(...r.scenarios); warnings.push(...r.warnings); }
const byCode={}; const dup=[];
for (const s of scenarios) { if (byCode[s.pathCode]) dup.push(s.pathCode); byCode[s.pathCode]=s; }
const ir={version:1, generatedAt:new Date().toISOString(), input:path.resolve(input), scenarios:scenarios.sort((a,b)=>a.pathCode.localeCompare(b.pathCode)), warnings};
if (args.out) writeJson(args.out, ir); else console.log(JSON.stringify(ir,null,2));
if (warnings.length) {
  for (const w of warnings) console.error(`${w.type}: ${w.file}:${w.line}: ${w.message}`);
}
if (dup.length) console.error(`duplicate_path_code: ${[...new Set(dup)].join(', ')}`);
if (args.strict && warnings.some(w=>w.type==='missing_path_code')) process.exit(2);
if (warnings.some(w=>w.type==='invalid_path_code') || dup.length) process.exit(1);
