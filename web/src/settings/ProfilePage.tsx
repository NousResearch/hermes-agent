import { api } from '@/lib/api';
import { Field, SaveField } from './components';
import type { PageProps } from './components';
import { writeSettings } from './api';
export default function ProfilePage({ data, run, busy }: PageProps) {
  const save = (patch: Partial<typeof data.profile>) => run(async () => {
    const result = await writeSettings<{ restart: boolean }>('profile', patch);
    if (result.restart) await api.restartGateway();
  }, 'timezone' in patch ? 'Saved. Telegram is restarting to apply the timezone.' : 'Saved. Applies from the next conversation.');
  const zones = Array.from(new Set(['', 'UTC', data.profile.timezone, Intl.DateTimeFormat().resolvedOptions().timeZone, ...Intl.supportedValuesOf('timeZone')]));
  return <><div className="ph"><h1>Profile</h1><p>How Hermes should work for you.</p></div><div className="sec">
    <Field label="Timezone" hint="Used for schedules. Saving restarts Telegram."><select aria-label="Timezone" value={data.profile.timezone} disabled={busy} onChange={e => void save({ timezone: e.target.value })}>{zones.map(zone => <option key={zone} value={zone}>{zone || 'Server default'}</option>)}</select></Field>
    <SaveField label="Instructions" value={data.profile.instructions} multiline onSave={instructions => save({ instructions })} hint="Applies from the next conversation." />
  </div></>;
}
