import { describe, expect, it } from 'vitest';
import { matchWakeWord } from './wakeWordMatcher';

describe('wakeWordMatcher', () => {
  it('detects simple English wake words', () => {
    const r1 = matchWakeWord('Wake up Jarvis');
    expect(r1.isWake).toBe(true);
    expect(r1.command).toBe('');

    const r2 = matchWakeWord('Hey Jarvis');
    expect(r2.isWake).toBe(true);

    const r3 = matchWakeWord('Hello Jarvis');
    expect(r3.isWake).toBe(true);

    const r4 = matchWakeWord('Jarvis are you there');
    expect(r4.isWake).toBe(true);
  });

  it('extracts English trailing commands in compound sentences', () => {
    const res = matchWakeWord('Wake up Jarvis, play Iron Man music from YouTube');
    expect(res.isWake).toBe(true);
    expect(res.command).toBe('play Iron Man music from YouTube');

    const res2 = matchWakeWord('Hey Jarvis search the web for latest news');
    expect(res2.isWake).toBe(true);
    expect(res2.command).toBe('search the web for latest news');
  });

  it('detects Egyptian and Standard Arabic wake words', () => {
    const r1 = matchWakeWord('اصحى يا جارفيس');
    expect(r1.isWake).toBe(true);
    expect(r1.command).toBe('');

    const r2 = matchWakeWord('يا جارفيس');
    expect(r2.isWake).toBe(true);

    const r3 = matchWakeWord('سامعني يا جارفيس');
    expect(r3.isWake).toBe(true);

    const r4 = matchWakeWord('اصحي يا جوين');
    expect(r4.isWake).toBe(true);
  });

  it('extracts Arabic trailing commands in compound sentences', () => {
    const res = matchWakeWord('يا جارفيس شغل موسيقى أيرون مان من يوتيوب');
    expect(res.isWake).toBe(true);
    expect(res.command).toBe('شغل موسيقى أيرون مان من يوتيوب');

    const res2 = matchWakeWord('اصحى يا جارفيس اكتب لي كود بايثون');
    expect(res2.isWake).toBe(true);
    expect(res2.command).toBe('اكتب لي كود بايثون');
  });

  it('detects standby and sleep commands in English and Arabic', () => {
    const r1 = matchWakeWord('Standby Jarvis');
    expect(r1.isStandby).toBe(true);
    expect(r1.isWake).toBe(false);

    const r2 = matchWakeWord('go to sleep');
    expect(r2.isStandby).toBe(true);

    const r3 = matchWakeWord('خليك على الـ standby');
    expect(r3.isStandby).toBe(true);

    const r4 = matchWakeWord('خلاص يا جارفيس');
    expect(r4.isStandby).toBe(true);
  });

  it('ignores non-wake utterances', () => {
    const r1 = matchWakeWord('What is the weather outside?');
    expect(r1.isWake).toBe(false);
    expect(r1.isStandby).toBe(false);

    const r2 = matchWakeWord('كم الساعة الآن؟');
    expect(r2.isWake).toBe(false);
    expect(r2.isStandby).toBe(false);
  });
});
