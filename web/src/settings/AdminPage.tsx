import { useState } from 'react';
import { HERMES_BASE_PATH } from '@/lib/api';
import { writeSettings } from './api';
import type { PageProps } from './components';
export default function AdminPage({ data, run, busy }: PageProps) {
  const [username, setUsername] = useState(data.admin.username);
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  return <><div className="ph"><h1>Admin login</h1><p>One shared login for everyone who manages Hermes.</p></div>{!data.admin.available ? <p>Password login is not configured on this server.</p> : <form className="sec" onSubmit={e => { e.preventDefault(); void run(async () => { await writeSettings('admin', { username, current_password: current, new_password: next }, 'POST'); window.location.assign(`${HERMES_BASE_PATH}/settings`); }, ''); }}>
    <div className="field"><label htmlFor="admin-username">Username</label><input id="admin-username" value={username} autoComplete="username" onChange={e => setUsername(e.target.value)} /></div>
    <div className="field"><label htmlFor="admin-current">Current password</label><input id="admin-current" type="password" autoComplete="current-password" value={current} onChange={e => setCurrent(e.target.value)} /></div>
    <div className="field"><label htmlFor="admin-new">New password</label><input id="admin-new" type="password" autoComplete="new-password" minLength={12} value={next} onChange={e => setNext(e.target.value)} /></div>
    <div className="save"><span>At least 12 characters. Changing the password signs everyone out.</span><button className="btn primary" disabled={busy || !username.trim() || !current || next.length < 12}>Change password</button></div>
  </form>}</>;
}
