import { describe, expect, it } from 'vitest';
import { parseMusicCommand } from './musicCommander';

describe('parseMusicCommand', () => {
  it('detects English YouTube playback commands', () => {
    const cmd1 = parseMusicCommand('play Iron Man music from YouTube');
    expect(cmd1).toBeDefined();
    expect(cmd1?.action).toBe('play');
    expect(cmd1?.query?.toLowerCase()).toContain('iron man');
    expect(cmd1?.isYouTube).toBe(true);

    const cmd2 = parseMusicCommand('can you please play AC/DC on youtube');
    expect(cmd2).toBeDefined();
    expect(cmd2?.action).toBe('play');
    expect(cmd2?.query).toBe('AC/DC');
    expect(cmd2?.isYouTube).toBe(true);
  });

  it('detects Arabic YouTube playback commands', () => {
    const cmd1 = parseMusicCommand('يا جارفيس شغل موسيقى ايرون مان من يوتيوب');
    expect(cmd1).toBeDefined();
    expect(cmd1?.action).toBe('play');
    expect(cmd1?.query).toContain('ايرون مان');
    expect(cmd1?.isYouTube).toBe(true);

    const cmd2 = parseMusicCommand('افتح يوتيوب وشغل حمزة نمرة');
    expect(cmd2).toBeDefined();
    expect(cmd2?.action).toBe('play');
    expect(cmd2?.query).toContain('حمزة نمرة');
    expect(cmd2?.isYouTube).toBe(true);
  });

  it('handles pause and navigation commands', () => {
    expect(parseMusicCommand('وقف الموسيقى')?.action).toBe('pause');
    expect(parseMusicCommand('stop music')?.action).toBe('pause');
    expect(parseMusicCommand('next song')?.action).toBe('next');
    expect(parseMusicCommand('اللي بعدها')?.action).toBe('next');
  });
});
