// @vitest-environment jsdom
import {act} from 'react';
import {createRoot} from 'react-dom/client';
import {it,expect,vi,afterEach,beforeEach} from 'vitest';
const {getPlugins}=vi.hoisted(()=>({getPlugins:vi.fn()}));
vi.mock('@/lib/api',()=>({api:{getPlugins},HERMES_BASE_PATH:'',fetchJSON:vi.fn(),authedFetch:vi.fn(),buildWsUrl:vi.fn(),buildWsAuthParam:vi.fn()}));
import {usePlugins} from './usePlugins';
import {exposePluginSDK,getPluginLoadError,unregisterPlugin} from './registry';
import {PluginSlot,unregisterPluginSlots} from './slots';
let root:ReturnType<typeof createRoot>, box:HTMLDivElement, mounted:boolean;
const manifest={name:'lifecycle-probe',label:'Probe',version:'1',source:'bundled',tab:{path:'/probe',hidden:true},slots:['config:section:probe'],entry:'dist/index.js',css:'dist/style.css'};
function Harness(){const {loading}=usePlugins();return <><output>{loading?'loading':'ready'}</output><PluginSlot name="config:section:probe"/></>}
async function flush(){await act(async()=>{await Promise.resolve();await Promise.resolve()})}
function script(){return document.querySelector<HTMLScriptElement>('script[data-hermes-plugin="lifecycle-probe"]')!}
async function register(s:HTMLScriptElement,text:string){await act(async()=>{const spy=vi.spyOn(document,'currentScript','get').mockReturnValue(s);try{window.__HERMES_PLUGINS__!.registerSlot('lifecycle-probe','config:section:probe',()=> <div>{text}</div>)}finally{spy.mockRestore()} s.onload?.(new Event('load'))});await flush()}
async function refresh(list:unknown[]){getPlugins.mockResolvedValue(list);await act(async()=>{window.dispatchEvent(new Event('hermes:dashboard-plugins-changed'))});await flush()}
beforeEach(async()=>{(globalThis as any).IS_REACT_ACT_ENVIRONMENT=true;vi.useFakeTimers();sessionStorage.clear();getPlugins.mockReset();getPlugins.mockResolvedValue([manifest]);exposePluginSDK();box=document.createElement('div');document.body.append(box);root=createRoot(box);mounted=true;await act(async()=>root.render(<Harness/>));await flush()});
afterEach(async()=>{if(mounted)await act(async()=>root.unmount());unregisterPluginSlots('lifecycle-probe');if(typeof unregisterPlugin==='function')unregisterPlugin('lifecycle-probe');box.remove();document.querySelectorAll('script[data-hermes-plugin="lifecycle-probe"],link[href*="lifecycle-probe"]').forEach(e=>e.remove());vi.useRealTimers();delete (globalThis as any).IS_REACT_ACT_ENVIRONMENT});
it('slot-only actual SDK registration is loaded without NO_REGISTER',async()=>{await register(script(),'current');expect(box.textContent).toContain('current');expect(getPluginLoadError('lifecycle-probe')).toBeUndefined();expect(box.querySelector('output')!.textContent).toBe('ready')});
it('manifest disable removes registered Config slot then reenable registers fresh content',async()=>{await register(script(),'old');await refresh([]);expect(box.textContent).not.toContain('old');expect(script()).toBeNull();await refresh([manifest]);await register(script(),'new');expect(box.textContent).toContain('new')});
it('same-path version update replaces script and clears registered slot',async()=>{const old=script();await register(old,'old');await refresh([{...manifest,version:'2'}]);expect(script()).not.toBe(old);expect(box.textContent).not.toContain('old');await register(script(),'new');await register(old,'stale');expect(box.textContent).toContain('new');expect(box.textContent).not.toContain('stale')});
it('late actual SDK registration cannot restore removed Config slot',async()=>{const old=script();await refresh([]);await register(old,'stale');expect(box.textContent).not.toContain('stale');expect(getPluginLoadError('lifecycle-probe')).toBeUndefined()});
