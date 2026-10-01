import { useEffect, useState } from 'react';
import { api, fetchJSON } from '@/lib/api';
import type { Model } from './api';
import type { OAuthProviderStatus, OAuthStartResponse } from '@/lib/api';
import { Choices, Field } from './components';
import type { PageProps } from './components';
import { writeSettings } from './api';

export default function ModelsPage({ data, run, busy }: PageProps) {
  const [account, setAccount] = useState<OAuthProviderStatus>();
  const [flow, setFlow] = useState<OAuthStartResponse>();
  const [loginError, setLoginError] = useState('');
  const [catalog, setCatalog] = useState(data.models);
  const refreshAccount = async () => { const result = await api.getOAuthProviders(); setAccount(result.providers.find(p => p.id === 'openai-codex')?.status); };
  useEffect(() => { void api.getOAuthProviders().then(result => setAccount(result.providers.find(p => p.id === 'openai-codex')?.status)).catch(e => setLoginError(String(e))); }, []);
  useEffect(() => {
    if (!flow) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api.pollOAuthSession('openai-codex', flow.session_id);
        if (stopped) return;
        if (result.status === 'approved') { setFlow(undefined); await refreshAccount(); return; }
        if (result.status !== 'pending') { setLoginError(result.error_message || `Sign-in ${result.status}. Please try again.`); setFlow(undefined); return; }
        timer = setTimeout(() => void poll(), 5000);
      } catch (e) { if (!stopped) { setLoginError(String(e)); setFlow(undefined); } }
    };
    timer = setTimeout(() => void poll(), 2000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [flow]);
  useEffect(() => { void fetchJSON<Model[]>('/api/settings/models').then(setCatalog).catch(e => setLoginError(String(e))); }, [account?.logged_in]);
  const refreshModels = () => run(async () => {
    setCatalog(await fetchJSON<Model[]>('/api/settings/models?refresh=true'));
  }, 'Model list refreshed. Availability depends on your account.');
  const efforts = (model: string) => (catalog.find(m => m.id === model)?.efforts || data.models.find(m => m.id === model)?.efforts || ['low', 'medium', 'high']).map(value => ({ value, label: ({ none: 'Off', xhigh: 'Extra high' } as Record<string, string>)[value] || value[0].toUpperCase() + value.slice(1) }));
  const chat = (model: string, effort: string) => run(async () => {
    const result = await api.setModelAssignment({ scope: 'main', provider: 'openai-codex', model });
    if (result.confirm_required) {
      if (!window.confirm(result.confirm_message || 'Confirm this model change?')) throw new Error('Model change cancelled.');
      const confirmed = await api.setModelAssignment({ scope: 'main', provider: 'openai-codex', model, confirm_expensive_model: true });
      if (!confirmed.ok) throw new Error('Model change was not accepted.');
    } else if (!result.ok) throw new Error('Model change was not accepted.');
    await api.saveConfig({ agent: { reasoning_overrides: { [model]: effort } } });
  }, 'Chat model saved. Applies on the next message unless the chat has its own model selection.');
  const memory = (model: string, learning: string, recall: string) => run(() => writeSettings('memory', { model, learning, recall }), 'Saved. Memory will restart automatically.');
  // An unset model shows an explicit placeholder instead of silently displaying the first option.
  const modelOptions = (selected: string, memory = false) => [
    ...(selected ? [] : [<option key="" value="" disabled>Choose a model</option>]),
    ...Array.from(new Set([selected, ...catalog.map(m => m.id)])).filter(id => id && (!memory || !id.endsWith('-900k'))).map(id => <option key={id}>{id}</option>),
  ];
  const unmanagedMemory = data.memory_status.state === 'unmanaged';
  const compatible = (model: string, current: string) => efforts(model).some(e => e.value === current) ? current : 'medium';
  return <><div className="ph"><h1>Models</h1><p>Hermes runs on your Codex subscription.</p></div>
    <div className="sec"><h2>Codex account</h2><p className="desc">{unmanagedMemory ? 'Used for chat and images.' : 'Used for chat, memory and images.'}</p><div className="rows"><div className="r"><div className="av brand">C</div><div className="t"><b>Codex</b><span className="line2">{account?.logged_in ? 'Signed in on this server' : 'Not connected'}</span></div><div className="btns">
      {account?.logged_in && <button className="btn quiet danger" disabled={busy} onClick={() => void run(async () => { await api.disconnectOAuthProvider('openai-codex'); await refreshAccount(); }, 'Signed out of Codex.')}>Sign out</button>}
      <button className="btn" disabled={busy || !!flow} onClick={() => void run(async () => { setLoginError(''); setFlow(await api.startOAuthLogin('openai-codex')); }, '')}>{account?.logged_in ? 'Sign in again' : 'Connect Codex'}</button></div></div></div>
      {flow && <div className="status">{flow.flow === 'device_code' ? <><p>Enter <strong className="mono">{flow.user_code}</strong> on the sign-in page.</p><a href={flow.verification_url} target="_blank" rel="noreferrer">Open Codex sign-in</a></> : <a href={flow.auth_url} target="_blank" rel="noreferrer">Open sign-in</a>} <button className="btn quiet" onClick={() => void run(async () => { await api.cancelOAuthSession(flow.session_id); setFlow(undefined); }, '')}>Cancel</button></div>}
      {loginError && <p role="alert" className="error">{loginError}</p>}
    </div>
    <div className="sec"><h2>Chat</h2><p className="desc">Changes apply on the next message. Chats with their own model selection keep it. <button type="button" className="link" disabled={busy} onClick={() => void refreshModels()}>Refresh list</button></p>
      <Field label="Model"><select aria-label="Chat model" value={data.chat.model} disabled={busy} onChange={e => void chat(e.target.value, compatible(e.target.value, data.chat.effort))}>{modelOptions(data.chat.model)}</select></Field>
      {data.chat.model && <Field label="Reasoning"><Choices label="Chat reasoning" options={efforts(data.chat.model)} value={data.chat.effort} disabled={busy} onChange={value => void chat(data.chat.model, value)} /></Field>}</div>
    <div className="sec"><h2>Memory</h2><p className="desc">{unmanagedMemory ? 'Memory settings are managed from the profile that owns this server’s Hindsight service.' : `The model Hindsight uses to learn and reflect. Changes restart memory. Status: ${data.memory_status.state}.`}</p>
      <Field label="Model"><select aria-label="Memory model" value={data.memory.llm_model} disabled={busy || unmanagedMemory} onChange={e => void memory(e.target.value, compatible(e.target.value, data.memory.llm_reasoning_effort), compatible(e.target.value, data.memory.reflect_llm_reasoning_effort))}>{modelOptions(data.memory.llm_model, true)}</select></Field>
      <Field label="Learning effort"><Choices label="Learning effort" options={efforts(data.memory.llm_model)} value={data.memory.llm_reasoning_effort} disabled={busy || unmanagedMemory} onChange={v => void memory(data.memory.llm_model, v, data.memory.reflect_llm_reasoning_effort)} /></Field>
      <Field label="Recall effort"><Choices label="Recall effort" options={efforts(data.memory.llm_model)} value={data.memory.reflect_llm_reasoning_effort} disabled={busy || unmanagedMemory} onChange={v => void memory(data.memory.llm_model, data.memory.llm_reasoning_effort, v)} /></Field></div>
  </>;
}
