import { useState } from 'react';
import { api } from '@/lib/api';
import { writeSettings } from './api';
import type { PageProps } from './components';
const KEYS = [
  ['TELEGRAM_BOT_TOKEN', 'Telegram bot', 'From @BotFather'],
  ['OPENROUTER_API_KEY', 'OpenRouter', 'Memory search and video'],
  ['BROWSER_USE_API_KEY', 'Browser Use', 'Web browser'],
  ['PARALLEL_API_KEY', 'Parallel', 'Web search'],
];
function KeyRow({ id, label, hint, saved, run, busy }: { id: string; label: string; hint: string; saved: boolean } & Pick<PageProps, 'run' | 'busy'>) {
  const [value, setValue] = useState('');
  const [checked, setChecked] = useState('');
  const save = () => run(async () => {
    const result = await writeSettings<{ username: string }>(`keys/${id}`, { value });
    await api.restartGateway();
    setValue('');
    setChecked(result.username ? `Verified @${result.username}.` : 'Key verified.');
  }, 'Key checked and saved. The gateway is restarting to apply it.');
  return <div className="sec"><div className="field top"><label htmlFor={id}>{label}<small>{hint}</small></label><input id={id} type="password" autoComplete="new-password" className="mono" value={value} placeholder={saved ? 'Key saved — paste to replace' : 'Paste your key'} onChange={e => setValue(e.target.value)} /></div><div className="save indent"><span><i className={`dot ${saved ? '' : 'warn'}`} />{checked || (saved ? 'Key saved. Replace to check again.' : 'Not connected.')}{id === 'PARALLEL_API_KEY' && ' Checking runs one search.'}</span><button className="btn primary" disabled={busy || !value.trim()} onClick={() => void save()}>Check and save</button></div></div>;
}
export default function KeysPage({ data, run, busy }: PageProps) {
  return <><div className="ph"><h1>Service keys</h1><p>Stored on this server and never shown again after saving.</p></div>{KEYS.map(([id, label, hint]) => <KeyRow key={id} id={id} label={label} hint={hint} saved={data.keys[id]} run={run} busy={busy} />)}</>;
}
