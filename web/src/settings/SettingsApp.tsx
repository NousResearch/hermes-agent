import { useCallback, useEffect, useRef, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router';
import { UserRound, Cpu, Send, KeyRound, LockKeyhole } from 'lucide-react';
import { ProfileProvider } from '@/contexts/ProfileProvider';
import { useProfileScope } from '@/contexts/useProfileScope';
import { api, HERMES_BASE_PATH } from '@/lib/api';
import { readSettings } from './api';
import type { Settings } from './api';
import ProfilePage from './ProfilePage';
import ModelsPage from './ModelsPage';
import AccessPage from './AccessPage';
import KeysPage from './KeysPage';
import AdminPage from './AdminPage';
import './settings.css';

const PAGES = [
  { path: '', label: 'Profile', icon: UserRound, page: ProfilePage },
  { path: 'models', label: 'Models', icon: Cpu, page: ModelsPage },
  { path: 'access', label: 'Access', icon: Send, page: AccessPage },
  { path: 'keys', label: 'Service keys', icon: KeyRound, page: KeysPage },
  { path: 'admin', label: 'Admin login', icon: LockKeyhole, page: AdminPage },
];
export default function SettingsApp() {
  return <ProfileProvider><ProfileSettings /></ProfileProvider>;
}
function ProfileSettings() {
  const { profile } = useProfileScope();
  return <SettingsContent key={profile || '__own__'} />;
}
function SettingsContent() {
  const [data, setData] = useState<Settings>();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const [telegram, setTelegram] = useState('Checking');
  const saving = useRef(false);
  const version = useRef(0);
  const refresh = useCallback(async () => {
    const result = await readSettings();
    setData(result);
  }, []);
  useEffect(() => {
    let alive = true;
    const load = async () => {
      if (saving.current) return;
      const started = version.current;
      try { const result = await readSettings(); if (alive && started === version.current) setData(result); }
      catch (e) { if (alive) setError(e instanceof Error ? e.message : String(e)); }
      try { const result = await api.getMessagingPlatforms(); if (alive) setTelegram(result.platforms.find(p => p.id === 'telegram')?.state || 'Not configured'); }
      catch { if (alive) setTelegram('Status unavailable'); }
    };
    void load();
    const timer = setInterval(() => void load(), 10000);
    return () => { alive = false; clearInterval(timer); };
  }, []);
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(''), 6000); return () => clearTimeout(timer); }, [notice]);
  const run = async (action: () => Promise<unknown>, success = 'Saved.') => {
    if (saving.current) return;
    saving.current = true; version.current++; setBusy(true); setError('');
    try { await action(); await refresh(); setNotice(success); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { saving.current = false; setBusy(false); }
  };
  const logout = async () => {
    const response = await fetch(`${HERMES_BASE_PATH}/auth/logout`, { method: 'POST', credentials: 'same-origin', headers: { 'X-Hermes-Session-Token': window.__HERMES_SESSION_TOKEN__ || '' } });
    if (!response.ok) throw new Error('Could not sign out. Try again.');
    window.location.assign(`${HERMES_BASE_PATH}/settings`);
  };
  // Native platform states are internal words; the sidebar shows what they mean for the person.
  const status = ({ connected: 'Online', disabled: 'Telegram not set up', 'Not configured': 'Telegram not set up', Checking: 'Checking…' } as Record<string, string>)[telegram] || `Telegram ${telegram.toLowerCase()}`;
  return <div className="settings-app"><div className="app"><aside><div className="ident"><div className="face">H</div><div><b>Hermes</b><span><i className={`dot ${telegram === 'connected' ? '' : 'warn'}`} />{status}</span></div></div>
    <nav aria-label="Settings">{PAGES.map(({ path, label, icon: Icon }) => <NavLink key={path} end={!path} to={`/settings${path ? '/' + path : ''}`}><Icon />{label}{path === 'access' && !!data?.access.pending.length && <span className="count">{data.access.pending.length}</span>}</NavLink>)}</nav>
    <div className="me"><div className="av">{data?.admin.username.slice(0, 1) || 'a'}</div>{data?.admin.username || 'admin'}<button className="btn quiet" onClick={() => void run(logout, '')}>Sign out</button></div></aside>
    <main><div className="inner">{error && <div role="alert" className="error">{error}<button className="btn quiet" onClick={() => void run(refresh, '')}>Retry</button></div>}{data ? <Routes>{PAGES.map(({ path, page: Page }) => <Route key={path} path={path ? `/settings/${path}` : '/settings'} element={<Page data={data} run={run} busy={busy} />} />)}<Route path="*" element={<ProfilePage data={data} run={run} busy={busy} />} /></Routes> : <p aria-busy="true">Loading settings…</p>}</div></main>
    {notice && <div className="toast show" role="status">{notice}</div>}
  </div></div>;
}
