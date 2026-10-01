import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { MoreHorizontal } from 'lucide-react';

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <div className="field"><label>{label}{hint && <small>{hint}</small>}</label><div>{children}</div></div>;
}
export function Choices({ value, options, onChange, disabled = false, label }: {
  value: string; options: { value: string; label: string }[]; onChange: (value: string) => void; disabled?: boolean; label: string;
}) {
  return <div className="seg" role="group" aria-label={label}>{options.map(option => <button key={option.value} type="button" className={option.value === value ? 'on' : ''} aria-pressed={option.value === value} disabled={disabled} onClick={() => onChange(option.value)}>{option.label}</button>)}</div>;
}
export function SaveField({ label, value, onSave, multiline = false, hint, id = `field-${label}` }: {
  label: string; value: string; onSave: (value: string) => Promise<void>; multiline?: boolean; hint: string; id?: string;
}) {
  const [draft, setDraft] = useState(value);
  const previousValue = useRef(value);
  useEffect(() => {
    const previous = previousValue.current;
    previousValue.current = value;
    setDraft(current => current === previous ? value : current);
  }, [value]);
  const [busy, setBusy] = useState(false);
  const save = async () => { setBusy(true); try { await onSave(draft); } finally { setBusy(false); } };
  const button = <button className="btn primary" disabled={busy || draft === value} onClick={() => void save()}>Save</button>;
  return <div className="field top"><label htmlFor={id}>{label}</label><div>
    {multiline
      ? <textarea id={id} value={draft} onChange={e => setDraft(e.target.value)} />
      : <div className="inl"><input id={id} type="text" value={draft} onChange={e => setDraft(e.target.value)} />{button}</div>}
    <div className="save tight"><span>{hint}</span>{multiline && button}</div>
  </div></div>;
}

/** Row "…" menu; closes on outside click or Escape. */
export function RowMenu({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return <div ref={ref} className={`more ${open ? 'open' : ''}`}>
    <button type="button" className="btn quiet icon" aria-label={label} aria-expanded={open} onClick={() => setOpen(!open)}><MoreHorizontal /></button>
    <div className="menu" onClick={() => setOpen(false)}>{children}</div>
  </div>;
}
export interface PageProps { data: import('./api').Settings; run: (action: () => Promise<unknown>, success?: string) => Promise<void>; busy: boolean }
