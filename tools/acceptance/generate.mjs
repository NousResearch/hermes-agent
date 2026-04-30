#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import {parseArgs, readJson, kindOf, domainOf, extFor, langForExt, esc, sanitizeIdent, markerStart, markerEnd, commentPrefix} from './lib.mjs';

function preserveBodies(text) {
  const out=new Map();
  const re=/^\s*(?:\/\/|#) \[acceptance:body:([^\]]+)\]\s*$([\s\S]*?)^\s*(?:\/\/|#) \[\/acceptance:body:\1\]\s*$/gm;
  let m; while ((m=re.exec(text))) out.set(m[1], m[2].replace(/^\n/,'').replace(/\n$/,''));
  return out;
}
function discoverExisting(file) {
  if (!fs.existsSync(file)) return [];
  const text=fs.readFileSync(file,'utf8');
  const re=/\[acceptance:metadata\]\s*(\{[^\n]+\})/g; const rows=[]; let m;
  while ((m=re.exec(text))) { try { rows.push(JSON.parse(m[1])); } catch {} }
  return rows;
}
function ownedBody(body, fallback, pad) {
  return body === undefined ? indent(fallback, pad) : body;
}
function jsMethod(s, body, ignored=false) {
  const reason = ignored ? `, { skip: 'orphan_test: scenario removed from IR' }` : '';
  const meta={PathCode:s.pathCode, Capability:s.capability, Fingerprint:s.fingerprint, Ignored:ignored};
  const defaultBody = `throw new Error('acceptance stub not implemented for ${s.pathCode}');`;
  return `// [acceptance:metadata] ${JSON.stringify(meta)}\ntest(${esc(`${s.pathCode} ${s.scenario}`)}${reason}, async () => {\n  ${markerStart(s.pathCode)}\n${ownedBody(body, defaultBody, '  ')}\n  ${markerEnd(s.pathCode)}\n});\n`;
}
function pyMethod(s, body, ignored=false) {
  const meta={PathCode:s.pathCode, Capability:s.capability, Fingerprint:s.fingerprint, Ignored:ignored};
  const dec=ignored?`@pytest.mark.skip(reason="orphan_test: scenario removed from IR")\n`:'';
  return `# [acceptance:metadata] ${JSON.stringify(meta)}\n@pytest.mark.path_code(${esc(s.pathCode)})\n@pytest.mark.capability(${esc(s.capability)})\n@pytest.mark.fingerprint(${esc(s.fingerprint)})\n${dec}def test_${sanitizeIdent(s.pathCode).toLowerCase()}():\n    ${markerStart(s.pathCode,'python')}\n${ownedBody(body, `assert False, "acceptance stub not implemented for ${s.pathCode}"`, '    ')}\n    ${markerEnd(s.pathCode,'python')}\n`;
}
function csMethod(s, body, ignored=false) {
  const meta={PathCode:s.pathCode, Capability:s.capability, Fingerprint:s.fingerprint, Ignored:ignored};
  const ignore=ignored?'[Ignore("orphan_test: scenario removed from IR")]\n    ':'';
  return `    // [acceptance:metadata] ${JSON.stringify(meta)}\n    [Test]\n    [Property("PathCode", ${esc(s.pathCode)})]\n    [Property("Capability", ${esc(s.capability)})]\n    [Property("Fingerprint", ${esc(s.fingerprint)})]\n    ${ignore}public void ${sanitizeIdent(s.pathCode)}()\n    {\n        ${markerStart(s.pathCode)}\n${ownedBody(body, `Assert.Fail("acceptance stub not implemented for ${s.pathCode}");`, '        ')}\n        ${markerEnd(s.pathCode)}\n    }\n`;
}
function goMethod(s, body, ignored=false) {
  const meta={PathCode:s.pathCode, Capability:s.capability, Fingerprint:s.fingerprint, Ignored:ignored};
  return `// [acceptance:metadata] ${JSON.stringify(meta)}\nfunc Test${sanitizeIdent(s.pathCode)}(t *testing.T) {\n\tt.Setenv("PathCode", ${esc(s.pathCode)})\n${ignored?'\tt.Skip("orphan_test: scenario removed from IR")\n':''}\t${markerStart(s.pathCode)}\n${ownedBody(body, `t.Fatalf("acceptance stub not implemented for ${s.pathCode}")`, '\t')}\n\t${markerEnd(s.pathCode)}\n}\n`;
}
function indent(s,p) { return String(s).split('\n').map(x=>x?p+x:x).join('\n'); }
function render(framework, domain, capability, scenarios, oldBodies, orphans) {
  const ext=extFor(framework); const lang=langForExt(ext); const cp=commentPrefix(lang);
  let head=''; let foot='';
  if (ext==='mjs' || ext==='ts') head=`import test from 'node:test';\n\n`;
  else if (ext==='py') head=`import pytest\n\n`;
  else if (ext==='cs') { head=`using NUnit.Framework;\n\nnamespace Acceptance.Generated;\n\n[TestFixture]\npublic class ${sanitizeIdent(domain)}AcceptanceTests\n{\n`; foot='}\n'; }
  else if (ext==='go') head=`package generated\n\nimport "testing"\n\n`;
  const rows=[];
  for (const s of scenarios) {
    const body=oldBodies.get(s.pathCode);
    rows.push(methodFor(ext,s,body,false));
  }
  for (const o of orphans) rows.push(methodFor(ext,o,oldBodies.get(o.pathCode),true));
  return head + rows.join('\n') + foot;
}
function methodFor(ext,s,body,ignored) {
  if (ext==='py') return pyMethod(s,body,ignored);
  if (ext==='cs') return csMethod(s,body,ignored);
  if (ext==='go') return goMethod(s,body,ignored);
  return jsMethod(s,body,ignored);
}

const args=parseArgs(process.argv.slice(2));
if (!args.ir) { console.error('usage: node generate.mjs --ir <path> [--map openspec/.acceptance.json] [--root .] [--dry-run]'); process.exit(64); }
const root=path.resolve(args.root||'.'); const map=readJson(path.resolve(root,args.map||'openspec/.acceptance.json')); const ir=readJson(path.resolve(args.ir));
const warnings=[]; const grouped=new Map();
for (const s0 of ir.scenarios) {
  const k=kindOf(s0.pathCode), d=domainOf(s0.pathCode); const dom=map.domains?.[d];
  if (!dom) { warnings.push({type:'domain_not_configured', pathCode:s0.pathCode}); continue; }
  if (k==='use-case') { warnings.push({type:'handler_not_configured', pathCode:s0.pathCode, message:'kind=use-case skipped'}); continue; }
  const cfg=dom[k]; if (!cfg) { warnings.push({type:'handler_not_configured', pathCode:s0.pathCode, message:`kind=${k} skipped`}); continue; }
  const key=`${d}|${k}`; if (!grouped.has(key)) grouped.set(key,{domain:d,kind:k,cfg,capability:dom.capability, scenarios:[]});
  grouped.get(key).scenarios.push({...s0, capability:dom.capability});
}
const generated=[];
for (const g of grouped.values()) {
  const framework=g.cfg.framework || map.framework || 'node-test'; const ext=extFor(framework);
  const dir=path.resolve(root,g.cfg.project||'.',g.cfg.subfolder||'.','Generated');
  const file=path.join(dir,`${g.domain}AcceptanceTests.${ext}`);
  const old=fs.existsSync(file)?fs.readFileSync(file,'utf8'):''; const bodies=preserveBodies(old);
  const existing=discoverExisting(file);
  const activeCodes=new Set(g.scenarios.map(s=>s.pathCode));
  const orphans=existing.filter(m=>m.PathCode && !activeCodes.has(m.PathCode)).map(m=>({pathCode:m.PathCode, scenario:`Orphan ${m.PathCode}`, capability:m.Capability||g.capability, fingerprint:m.Fingerprint||''}));
  const text=render(framework,g.domain,g.capability,g.scenarios.sort((a,b)=>a.pathCode.localeCompare(b.pathCode)),bodies,orphans);
  generated.push({file:path.relative(root,file), scenarios:g.scenarios.length, orphans:orphans.length});
  if (!args['dry-run']) { fs.mkdirSync(dir,{recursive:true}); fs.writeFileSync(file,text); }
}
for (const w of warnings) console.error(`${w.type}: ${w.pathCode}${w.message?': '+w.message:''}`);
console.log(JSON.stringify({generated,warnings},null,2));
