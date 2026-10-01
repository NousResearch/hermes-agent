import { useState } from 'react';
import { api } from '@/lib/api';
import type { PairingUser } from '@/lib/api';
import { Plus, ChevronRight, UserRound, Users } from 'lucide-react';
import { Choices, RowMenu, SaveField } from './components';
import type { PageProps } from './components';
import { writeSettings } from './api';
import type { Group, Topic } from './api';

const GROUP_MODES = [{ value: 'mention', label: 'When mentioned' }, { value: 'all', label: 'Every message' }];
async function changePerson(body: unknown) {
  const result = await writeSettings<{ restart: boolean }>('people', body, 'POST');
  if (result.restart) await api.restartGateway();
}
const TOPIC_MODES = [{ value: 'inherit', label: 'Same as group' }, { value: 'all', label: 'Every message' }, { value: 'silent', label: 'Silent' }];
/** Up to two initials from the words of a name; empty when the name is only an ID. */
const initials = (name: string) => (name.match(/\p{L}+/gu) || []).slice(0, 2).map(word => word[0].toUpperCase()).join('');
/** Telegram's General topic is thread 1; other unnamed topics are shown by number. */
const topicLabel = (topic: Topic) => topic.name && topic.name !== topic.id ? `# ${topic.name}` : topic.id === '1' ? '# General' : `Topic ${topic.id}`;

function GroupRow({ group, run, busy }: { group: Group } & Pick<PageProps, 'run' | 'busy'>) {
  const [open, setOpen] = useState(false);
  const named = group.name && group.name !== group.id;
  const update = (patch: Partial<Group> & { remove?: boolean }) => run(async () => {
    await writeSettings(`groups/${encodeURIComponent(group.id)}`, { ...group, ...patch });
    await api.restartGateway();
  }, 'Saved. Telegram is restarting to apply this change.');
  return <div className={`grp ${open ? 'open' : ''}`}><div className="r">
    <button type="button" className="btn quiet icon chev-btn" aria-label={`Expand ${group.name}`} aria-expanded={open} onClick={() => setOpen(!open)}><ChevronRight className="chev" /></button>
    <div className="av sq">{(named && initials(group.name)) || <Users />}</div>
    <div className="t"><b className={named ? '' : 'mono'}>{named ? group.name : group.id}</b>{named && <span className="line2">{group.id}</span>}</div>
    <Choices label={`Replies in ${group.name}`} value={group.mode} options={GROUP_MODES} disabled={busy} onChange={mode => void update({ mode: mode as Group['mode'] })} />
    <RowMenu label={`Manage ${group.name}`}><button className="danger" disabled={busy} onClick={() => void update({ remove: true })}>Remove group</button></RowMenu>
  </div><div className="gbody">
    <SaveField id={`group-${group.id}-instructions`} label="Instructions for this group" value={group.instructions} multiline hint="Saving restarts Telegram." onSave={instructions => update({ instructions })} />
    <div className="topics"><p className="lbl">Topics</p><p className="hint">Topics appear automatically after Hermes receives messages in them.</p>{group.topics.map(topic => <div className="topic" key={topic.id}><span>{topicLabel(topic)}</span><Choices label={`Replies in topic ${topic.id}`} value={topic.mode} options={TOPIC_MODES} disabled={busy} onChange={mode => void update({ topics: group.topics.map(t => t.id === topic.id ? { ...t, mode: mode as typeof t.mode } : t) })} /></div>)}
    </div>
  </div></div>;
}

function PersonTitle({ person }: { person: PairingUser }) {
  return <><div className="av">{initials(person.user_name || '') || <UserRound />}</div>
    <div className="t"><b className={person.user_name ? '' : 'mono'}>{person.user_name || person.user_id}</b>{person.user_name && <span className="line2">{person.user_id}</span>}</div></>;
}

export default function AccessPage({ data, run, busy }: PageProps) {
  const [adding, setAdding] = useState<'group' | 'person'>();
  const [id, setId] = useState('');
  const [mode, setMode] = useState('mention');
  const open = (kind: 'group' | 'person') => { setId(''); setAdding(kind); };
  const add = () => run(async () => {
    if (adding === 'group') { await writeSettings(`groups/${encodeURIComponent(id)}`, { mode }); await api.restartGateway(); }
    else await changePerson({ action: 'add', user_id: id });
    setAdding(undefined);
  }, adding === 'group' ? 'Group added. Telegram is restarting.' : 'Person added.');
  return <><div className="ph"><h1>Access</h1><p>Who can talk to Hermes on Telegram, and when it replies in groups.</p></div>
    {data.access.notice && <p role="status" className="notice">{data.access.notice}</p>}
    <div className="sec"><div className="sh"><div><h2>Groups</h2><p>Everyone in these groups can talk to Hermes.</p></div><button className="btn" onClick={() => open('group')}><Plus />Add group</button></div>
      <div className="rows">{data.access.groups.map(group => <GroupRow key={group.id} group={group} run={run} busy={busy} />)}{!data.access.groups.length && <div className="r empty">No groups allowed yet.</div>}</div></div>
    <div className="sec"><div className="sh"><div><h2>People</h2><p>Can message Hermes directly and in any group it’s in.</p></div><button className="btn" onClick={() => open('person')}><Plus />Add person</button></div>
      <div className="rows">{data.access.pending.map(person => <div className="r req" key={person.request_id}><PersonTitle person={person} /><div className="btns"><button className="btn quiet" disabled={busy} onClick={() => void run(() => writeSettings('people', { action: 'decline', user_id: person.user_id, request_id: person.request_id }, 'POST'), 'Request declined.')}>Decline</button><button className="btn primary" disabled={busy} onClick={() => void run(() => changePerson({ action: 'add', user_id: person.user_id, request_id: person.request_id }), 'Access approved.')}>Approve</button></div></div>)}
      {data.access.people.map(person => <div className="r" key={person.user_id}><PersonTitle person={person} /><RowMenu label={`Manage ${person.user_name || person.user_id}`}><button className="danger" disabled={busy} onClick={() => void run(() => changePerson({ action: 'remove', user_id: person.user_id }), 'Access removed.')}>Remove access</button></RowMenu></div>)}
      {!data.access.people.length && !data.access.pending.length && <div className="r empty">No approved people. Ask someone to message the bot to request access.</div>}</div></div>
    {adding && <div className="dialog-backdrop" onKeyDown={e => { if (e.key === 'Escape') setAdding(undefined); }}><form className="dialog-card" role="dialog" aria-modal="true" aria-label={`Add ${adding}`} onSubmit={e => { e.preventDefault(); void add(); }}>
      <h3>Add {adding}</h3>
      <p>{adding === 'group' ? 'Add the bot to the group in Telegram first.' : 'They can message the bot directly and in any group it’s in.'}</p>
      <label htmlFor="new-access-id">Telegram {adding === 'group' ? 'group' : 'user'} ID</label>
      <input id="new-access-id" type="text" className="mono" value={id} onChange={e => setId(e.target.value.trim())} placeholder={adding === 'group' ? '-100…' : '70231145'} autoFocus />
      {adding === 'group' && <><label>Replies</label><Choices label="New group replies" options={GROUP_MODES} value={mode} onChange={setMode} /></>}
      <div className="dialog-actions"><button className="btn" type="button" onClick={() => setAdding(undefined)}>Cancel</button><button className="btn primary" disabled={busy || !(adding === 'group' ? /^-\d+$/ : /^\d+$/).test(id)}>Add {adding}</button></div>
    </form></div>}
  </>;
}
