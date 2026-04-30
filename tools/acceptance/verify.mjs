#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {parseArgs, readJson, writeJson, kindOf, domainOf, extFor, walkMarkdown} from './lib.mjs';

function scanMetadata(root) {
  const out=[];
  function walk(d) {
    if (!fs.existsSync(d)) return;
    for (const ent of fs.readdirSync(d,{withFileTypes:true})) {
      if (ent.name==='node_modules'||ent.name==='.git'||ent.name==='.work') continue;
      const p=path.join(d,ent.name);
      if (ent.isDirectory()) walk(p);
      else if (/AcceptanceTests\.(mjs|ts|cs|go|py)$/.test(ent.name)) {
        const text=fs.readFileSync(p,'utf8'); const re=/\[acceptance:metadata\]\s*(\{[^\n]+\})/g; let m;
        while ((m=re.exec(text))) { try { out.push({...JSON.parse(m[1]), file:p}); } catch {} }
      }
    }
  }
  walk(root); return out;
}
function configuredSpecRoot(root,map,args) {
  if (args.change) return path.join(root,'openspec','changes',args.change,'specs');
  return path.join(root,map.specDirectory||'openspec/specs');
}
function generatedFiles(project, ext) {
  const dir=path.join(project,'Generated');
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f=>f.endsWith(`.${ext}`)).map(f=>path.join(dir,f));
}
function runnerFor(framework, project, codes, root, filterSyntax) {
  const f=(framework||'node-test').toLowerCase(); const pattern=codes.map(c=>c.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|') || 'a^';
  if (f.includes('node')) return {cmd:'node', args:['--test','--test-name-pattern',pattern,...generatedFiles(project,'mjs')], shell:false};
  if (f.includes('vitest')) return {cmd:'npx', args:['vitest','run',project,'--testNamePattern',pattern], shell:false};
  if (f.includes('pytest')||f.includes('python')) return {cmd:'python', args:['-m','pytest',project,'-q','-m',codes.map(c=>`path_code('${c}')`).join(' or ')], shell:false};
  if (f.includes('go')) return {cmd:'go', args:['test',project,'-run',pattern], shell:false};
  if (f.includes('nunit')||f.includes('c#')) return {cmd:'dotnet', args:['test',project,'--filter',codes.map(c=>`PathCode=${c}`).join('|')], shell:false};
  if (filterSyntax) return {cmd:filterSyntax, args:[pattern], shell:true};
  return null;
}
const args=parseArgs(process.argv.slice(2));
if (!args.all && !args.change && !args.capability) { console.error('usage: node verify.mjs (--change <id> | --capability <name> | --all) [--no-generate] [--no-run]'); process.exit(64); }
const root=process.cwd(); const mapPath=path.join(root,'openspec/.acceptance.json'); const map=readJson(mapPath);
const work=path.join(root,'tools/acceptance/.work'); fs.mkdirSync(work,{recursive:true});
const specRoot=configuredSpecRoot(root,map,args); const irPath=path.join(work,'ir.json');
let extract=spawnSync('node',[path.join(root,'tools/acceptance/extract.mjs'),specRoot,'--out',irPath,'--strict'],{encoding:'utf8'});
process.stdout.write(extract.stdout); process.stderr.write(extract.stderr); if (extract.status!==0) process.exit(extract.status);
if (!args['no-generate']) { const gen=spawnSync('node',[path.join(root,'tools/acceptance/generate.mjs'),'--ir',irPath,'--map','openspec/.acceptance.json','--root',root],{encoding:'utf8'}); process.stdout.write(gen.stdout); process.stderr.write(gen.stderr); if (gen.status!==0) process.exit(gen.status); }
const ir=readJson(irPath); let scenarios=ir.scenarios;
if (args.capability) scenarios=scenarios.filter(s=>map.domains?.[domainOf(s.pathCode)]?.capability===args.capability);
const uc=scenarios.filter(s=>kindOf(s.pathCode)==='use-case');
for (const s of uc) console.error(`handler_not_configured: ${s.pathCode}: kind=use-case skipped`);
const activeScenarios=scenarios.filter(s=>kindOf(s.pathCode)!=='use-case'); const irCodes=new Set(activeScenarios.map(s=>s.pathCode));
const metas=scanMetadata(root).filter(m=>!args.capability || m.Capability===args.capability);
const active=metas.filter(m=>!m.Ignored); const counts=new Map(); for (const m of active) counts.set(m.PathCode,(counts.get(m.PathCode)||0)+1);
const failures=[];
for (const s of activeScenarios) if (!active.some(m=>m.PathCode===s.pathCode)) failures.push({type:'missing_test', pathCode:s.pathCode});
for (const [c,n] of counts) if (n>1) failures.push({type:'duplicate_tests', pathCode:c, count:n});
for (const m of active) if (m.PathCode && !irCodes.has(m.PathCode)) failures.push({type:'orphan_test', pathCode:m.PathCode, file:path.relative(root,m.file)});
if (failures.length) { for (const f of failures) console.error(`${f.type}: ${f.pathCode}${f.file?': '+f.file:''}`); process.exit(1); }
if (!args['no-run'] && activeScenarios.length) {
  const byProject=new Map();
  for (const s of activeScenarios) {
    const d=domainOf(s.pathCode), k=kindOf(s.pathCode), cfg=map.domains[d][k], key=`${cfg.framework}|${cfg.project}`;
    if (!byProject.has(key)) byProject.set(key,{cfg,codes:[]}); byProject.get(key).codes.push(s.pathCode);
  }
  for (const g of byProject.values()) {
    const r=runnerFor(g.cfg.framework,path.resolve(root,g.cfg.project,g.cfg.subfolder||'.'),g.codes,root,map.filterSyntax);
    if (!r) { console.error(`test_failed: no runner for ${g.cfg.framework}`); process.exit(1); }
    const res=spawnSync(r.cmd,r.args,{cwd:root,encoding:'utf8',shell:r.shell}); process.stdout.write(res.stdout); process.stderr.write(res.stderr);
    if (res.status!==0) { console.error(`test_failed: ${g.codes.join(',')}`); process.exit(1); }
  }
}
console.log('acceptance gate passed');
