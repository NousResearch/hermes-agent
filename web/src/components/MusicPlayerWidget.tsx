import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Music2,
  Shuffle,
  Repeat,
  Volume2,
  VolumeX,
  ListMusic,
  Disc3,
  Upload,
  Radio,
  Search,
  Sparkles,
  ShieldAlert,
  ExternalLink,
  Youtube,
} from 'lucide-react';
import { apiFetch } from '@/utils/jarvisApiClient';
import type { MusicTrack, MusicCommand } from '@/types/jarvis';
import { findBestTrackIndex, type MusicCommandAction } from '@/utils/musicCommander';

export interface MusicPlayerWidgetProps {
  command?: MusicCommand | null;
}

export type MusicCategory = 'all' | 'youtube' | 'hamza' | 'ironman' | 'billie' | 'cyber';

export interface EnrichedMusicTrack extends MusicTrack {
  category?: 'youtube' | 'hamza' | 'ironman' | 'billie' | 'cyber';
  badge?: string;
  youtubeId?: string;
  thumbnail?: string;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds) || seconds <= 0) {
    return '--:--';
  }
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Built-in Demo Cyberpunk Ambient Synthesizer using Web Audio API
class CyberSynthEngine {
  private ctx: AudioContext | null = null;
  private osc1: OscillatorNode | null = null;
  private osc2: OscillatorNode | null = null;
  private gainNode: GainNode | null = null;
  private isPlaying = false;

  public start(volume = 0.8) {
    if (this.isPlaying) return;
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioCtx();
      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.setValueAtTime(volume * 0.15, this.ctx.currentTime);
      this.gainNode.connect(this.ctx.destination);

      // Deep drone
      this.osc1 = this.ctx.createOscillator();
      this.osc1.type = 'sawtooth';
      this.osc1.frequency.setValueAtTime(55, this.ctx.currentTime); // A1 note

      // Shimmer detune
      this.osc2 = this.ctx.createOscillator();
      this.osc2.type = 'sine';
      this.osc2.frequency.setValueAtTime(110.5, this.ctx.currentTime); // A2 slight detune

      this.osc1.connect(this.gainNode);
      this.osc2.connect(this.gainNode);

      this.osc1.start();
      this.osc2.start();
      this.isPlaying = true;
    } catch (e) {
      console.warn('Web Audio synth failed:', e);
    }
  }

  public stop() {
    if (!this.isPlaying) return;
    try {
      this.osc1?.stop();
      this.osc2?.stop();
      this.osc1?.disconnect();
      this.osc2?.disconnect();
      this.ctx?.close();
    } catch {
      // ignore
    }
    this.isPlaying = false;
  }

  public setVolume(volume: number) {
    if (this.gainNode && this.ctx) {
      this.gainNode.gain.setValueAtTime(volume * 0.15, this.ctx.currentTime);
    }
  }

  public active() {
    return this.isPlaying;
  }
}

export const DEFAULT_DEMO_TRACKS: EnrichedMusicTrack[] = [
  // === YouTube Live Streaming (تشغيل يوتيوب المباشر) ===
  {
    id: 'yt-ironman-back-in-black',
    title: 'Back In Black (Iron Man Theme - AC/DC)',
    artist: 'AC/DC - Marvel Studios Iron Man',
    duration: 254,
    path: 'https://www.youtube.com/watch?v=IyR25B-IGyg',
    sizeBytes: 0,
    category: 'youtube',
    badge: 'YOUTUBE LIVE',
    youtubeId: 'IyR25B-IGyg',
    thumbnail: 'https://i.ytimg.com/vi/IyR25B-IGyg/hqdefault.jpg',
  },
  {
    id: 'yt-ironman-driving-with-the-top-down',
    title: 'Driving With The Top Down (Iron Man Theme)',
    artist: 'Ramin Djawadi (Iron Man Original Score)',
    duration: 190,
    path: 'https://www.youtube.com/watch?v=jNo3zmhXE9Y',
    sizeBytes: 0,
    category: 'youtube',
    badge: 'YOUTUBE LIVE',
    youtubeId: 'jNo3zmhXE9Y',
    thumbnail: 'https://i.ytimg.com/vi/jNo3zmhXE9Y/hqdefault.jpg',
  },

  // === حمزة نمرة (Hamza Namira) ===
  {
    id: 'hamza-fady-shewaya',
    title: 'فاضي شوية (Fady Shewaya)',
    artist: 'حمزة نمرة (Hamza Namira)',
    duration: 248,
    path: 'https://archive.org/download/kadero-hamza-namira-fady-shewaya/kadero-Hamza%20Namira%20-%20Fady%20Shewaya%20-.mp3',
    sizeBytes: 3960000,
    category: 'hamza',
    badge: 'EXCLUSIVE',
  },
  {
    id: 'hamza-reyah-el-hayah',
    title: 'رياح الحياة (Reyah El Hayah)',
    artist: 'حمزة نمرة (Hamza Namira)',
    duration: 280,
    path: 'https://archive.org/download/y-2mate.com-hamza-namira-reyah-el-hayah/y2mate.com%20-%20Hamza%20Namira%20%20Reyah%20El%20Hayah%20%20%D8%AD%D9%85%D8%B2%D8%A9%20%D9%86%D9%85%D8%B1%D8%A9%20%20%D8%B1%D9%8A%D8%A7%D8%AD%20%D8%A7%D9%84%D8%AD%D9%8A%D8%A7%D8%A9.mp3',
    sizeBytes: 4480000,
    category: 'hamza',
    badge: 'POPULAR',
  },
  {
    id: 'hamza-dari-ya-alby',
    title: 'داري يا قلبي (Dari Ya Alby)',
    artist: 'حمزة نمرة (Hamza Namira)',
    duration: 240,
    path: 'https://archive.org/download/dary_ya_qalby/dary_ya_qalby.mp3',
    sizeBytes: 3840000,
    category: 'hamza',
    badge: 'LEGENDARY',
  },
  {
    id: 'hamza-wa-ollak-eh',
    title: 'واقولك إيه (Wa Ollak Eh)',
    artist: 'حمزة نمرة (Hamza Namira)',
    duration: 235,
    path: 'https://archive.org/download/HamzaNamira/Hamza%20Namira%20-%20Wa%20Ollak%20Eh%20%C2%A6%20%D8%AD%D9%85%D8%B2%D8%A9%20%D9%86%D9%85%D8%B1%D8%A9%20-%20%D9%88%D8%A7%D9%82%D9%88%D9%84%D9%83%20%D8%A5%D9%8A%D9%87%20%C2%A6%20Official%20Video.mp3',
    sizeBytes: 3760000,
    category: 'hamza',
  },
  {
    id: 'hamza-ala-bab-allah',
    title: 'على باب الله - إنسان (Ala Bab Allah)',
    artist: 'حمزة نمرة (Hamza Namira)',
    duration: 255,
    path: 'https://archive.org/download/Hamza.ensan/Ala.Bab.Allah%20%D8%B9%D9%84%D9%89%20%D8%A8%D8%A7%D8%A8%20%D8%A7%D9%84%D9%84%D9%87.mp3',
    sizeBytes: 4080000,
    category: 'hamza',
  },
  {
    id: 'hamza-ya-nes-jaratli',
    title: 'يا ناس جرت لي (Ya Nes Jaratli)',
    artist: 'حمزة نمرة & زاب ثروت',
    duration: 250,
    path: 'https://archive.org/download/Hamza.Namira.Ft.Zap.TharwatYa.Nes.Jaratli_201605/Hamza.Namira.Ft.Zap.Tharwat_Ya.Nes.Jaratli.mp3',
    sizeBytes: 4000000,
    category: 'hamza',
  },

  // === Iron Man & JARVIS Audio Core (أغاني أيرون مان وجارفيس) ===
  {
    id: 'ironman-first-flight',
    title: 'First Flight (Mark II Flight Test with JARVIS)',
    artist: 'Ramin Djawadi (Iron Man Original Score)',
    duration: 145,
    path: 'https://archive.org/download/TestITestIITestDay11IronManCompleteScoreNoSFXRaminDjawadi/1-34%20First%20Flight%20%28Iron%20Man%20Complete%20Score%20No%20SFX%29%20Ramin%20Djawadi.mp3',
    sizeBytes: 2320000,
    category: 'ironman',
    badge: 'JARVIS THEME',
  },
  {
    id: 'ironman-test-day-11',
    title: 'Stark Lab Diagnostics & Test Day 11',
    artist: 'Ramin Djawadi (Iron Man Soundtrack)',
    duration: 180,
    path: 'https://archive.org/download/TestITestIITestDay11IronManCompleteScoreNoSFXRaminDjawadi/Test%20ITest%20IITest%20Day%2011%20%28Iron%20Man%20Complete%20Score%20No%20SFX%29%20Ramin%20Djawadi.mp3',
    sizeBytes: 2880000,
    category: 'ironman',
    badge: 'STARK LAB',
  },
  {
    id: 'ironman-back-in-black',
    title: 'Back In Black (Iron Man Intro Anthem)',
    artist: 'AC/DC (Stark Industries Edition)',
    duration: 255,
    path: 'https://archive.org/download/covered-in-black-an-industrial-tribute-to-the-kings-of-high-voltage-ac-dc-06-who/Covered%20In%20Black%20-%20An%20Industrial%20Tribute%20To%20The%20Kings%20Of%20High%20Voltage%20AC_DC-07-Back%20In%20Black%20%28Pigface%20Vs.%20Sheep%20On%20Drugs%29.mp3',
    sizeBytes: 4080000,
    category: 'ironman',
    badge: 'HIGH VOLTAGE',
  },
  {
    id: 'ironman-shoot-to-thrill',
    title: 'Shoot To Thrill (Stark Expo Mark IV Entrance)',
    artist: 'AC/DC (Iron Man 2 Edition)',
    duration: 200,
    path: 'https://archive.org/download/covered-in-black-an-industrial-tribute-to-the-kings-of-high-voltage-ac-dc-06-who/Covered%20In%20Black%20-%20An%20Industrial%20Tribute%20To%20The%20Kings%20Of%20High%20Voltage%20AC_DC-01-Highway%20To%20Hell%20%28The%20Electric%20Hellfire%20Club%29.mp3',
    sizeBytes: 3200000,
    category: 'ironman',
    badge: 'EXPO 2026',
  },

  // === Billie Eilish (بيلي إيليش) ===
  {
    id: 'billie-bad-guy',
    title: 'Bad Guy',
    artist: 'Billie Eilish',
    duration: 194,
    path: 'https://archive.org/download/interrupters2021-07-24.aud.flac16/2021-07-24%20Globe%20Life%20Field%2C%20Arlington%2C%20Texas/interrupters2021-07-24t-04.mp3',
    sizeBytes: 3100000,
    category: 'billie',
    badge: 'HIT SINGLE',
  },
  {
    id: 'billie-lovely',
    title: 'Lovely (ft. Khalid)',
    artist: 'Billie Eilish & Khalid',
    duration: 200,
    path: 'https://archive.org/download/Tiktok-Mixtapes/TikTok%20Chillout%20Mix.mp3',
    sizeBytes: 3200000,
    category: 'billie',
    badge: 'CHILLOUT',
  },
  {
    id: 'billie-birds-feather',
    title: 'Birds of a Feather',
    artist: 'Billie Eilish',
    duration: 198,
    path: 'https://archive.org/download/bloodline-2024-08-24ojibway-park-woodbury-mn-aud/04.%20BIRDS%20OF%20A%20FEATHER.mp3',
    sizeBytes: 3160000,
    category: 'billie',
  },
  {
    id: 'billie-ocean-eyes',
    title: 'Ocean Eyes (Acoustic Fusion)',
    artist: 'Billie Eilish',
    duration: 226,
    path: 'https://archive.org/download/ocean-eyes-remix-wvr6j8/dominic%20pierce%20-%20ocean%20eyes%20-remix-.mp3',
    sizeBytes: 5633928,
    category: 'billie',
  },

  // === Stark Cyber Synth & Laboratory ===
  {
    id: 'track-cyber-1',
    title: 'Mark 85 Cyber Sentinel (Ambient Drone)',
    artist: 'JARVIS Audio Core',
    duration: 180,
    path: 'synthetic-ambient',
    sizeBytes: 1024,
    category: 'cyber',
    badge: 'NEON CORE',
  },
  {
    id: 'track-cyber-2',
    title: 'Neon Arc Reactor Pulse',
    artist: 'Stark Holographic Acoustics',
    duration: 240,
    path: 'synthetic-ambient',
    sizeBytes: 1024,
    category: 'cyber',
    badge: 'SYNTHESIS',
  },
];

export const MusicPlayerWidget: React.FC<MusicPlayerWidgetProps> = ({ command }) => {
  const [tracks, setTracks] = useState<EnrichedMusicTrack[]>(DEFAULT_DEMO_TRACKS);
  const [selectedCategory, setSelectedCategory] = useState<MusicCategory>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState<number | null>(248);
  const [volume, setVolume] = useState(0.85);
  const [isMuted, setIsMuted] = useState(false);
  const [isShuffle, setIsShuffle] = useState(false);
  const [isRepeat, setIsRepeat] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSearchingYoutube, setIsSearchingYoutube] = useState(false);
  const [youtubeSearchInput, setYoutubeSearchInput] = useState('');

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const synthRef = useRef<CyberSynthEngine>(new CyberSynthEngine());
  const timerRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Scan API music library if available
  useEffect(() => {
    let cancelled = false;
    apiFetch('/api/music/library')
      .then((res) => {
        if (!res.ok) throw new Error(`API ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const list: EnrichedMusicTrack[] =
          Array.isArray(data?.tracks) && data.tracks.length > 0 ? data.tracks : [];
        if (list.length > 0) {
          // Merge API tracks with our curated collection without duplicates
          const existingIds = new Set(list.map((t) => t.id));
          const merged = [...list, ...DEFAULT_DEMO_TRACKS.filter((t) => !existingIds.has(t.id))];
          setTracks(merged);
        }
      })
      .catch(() => {
        // Keep default curated tracks
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const currentTrack = tracks[currentIndex] ?? null;

  // Handle Play/Pause
  const togglePlay = useCallback(() => {
    if (!currentTrack) return;

    if (isPlaying) {
      // Pause
      if (currentTrack.path === 'synthetic-ambient') {
        synthRef.current.stop();
      } else if (audioRef.current) {
        audioRef.current.pause();
      }
      clearInterval(timerRef.current);
      setIsPlaying(false);
    } else {
      // Play
      if (currentTrack.category === 'youtube' || currentTrack.youtubeId) {
        setIsPlaying(true);
        timerRef.current = setInterval(() => {
          setCurrentTime((prev) => {
            const next = prev + 1;
            if (next >= (currentTrack.duration || 240)) {
              if (isRepeat) return 0;
              playNext();
              return 0;
            }
            return next;
          });
        }, 1000);
      } else if (currentTrack.path === 'synthetic-ambient') {
        synthRef.current.start(isMuted ? 0 : volume);
        timerRef.current = setInterval(() => {
          setCurrentTime((prev) => {
            const next = prev + 1;
            if (next >= (currentTrack.duration || 180)) {
              if (isRepeat) return 0;
              playNext();
              return 0;
            }
            return next;
          });
        }, 1000);
        setIsPlaying(true);
      } else if (audioRef.current) {
        try {
          const playPromise = audioRef.current.play();
          if (playPromise && typeof playPromise.then === 'function') {
            playPromise
              .then(() => setIsPlaying(true))
              .catch((err) => {
                console.warn('Audio play error, falling back to synth:', err);
                synthRef.current.start(isMuted ? 0 : volume);
                setIsPlaying(true);
              });
          } else {
            setIsPlaying(true);
          }
        } catch {
          synthRef.current.start(isMuted ? 0 : volume);
          setIsPlaying(true);
        }
      }
    }
  }, [currentTrack, isPlaying, isMuted, volume, isRepeat]);

  const playTrackAt = useCallback(
    (index: number) => {
      synthRef.current.stop();
      clearInterval(timerRef.current);
      setCurrentIndex(index);
      setCurrentTime(0);
      const track = tracks[index];
      if (!track) return;
      setDuration(track.duration || 180);
      setError(null);

      if (track.category === 'youtube' || track.youtubeId) {
        if (audioRef.current) {
          audioRef.current.pause();
        }
        setIsPlaying(true);
        timerRef.current = setInterval(() => {
          setCurrentTime((prev) => {
            const next = prev + 1;
            if (next >= (track.duration || 240)) {
              if (isRepeat) return 0;
              playNext();
              return 0;
            }
            return next;
          });
        }, 1000);
      } else if (track.path === 'synthetic-ambient') {
        synthRef.current.start(isMuted ? 0 : volume);
        timerRef.current = setInterval(() => {
          setCurrentTime((prev) => {
            const next = prev + 1;
            if (next >= (track.duration || 180)) {
              if (isRepeat) return 0;
              playNext();
              return 0;
            }
            return next;
          });
        }, 1000);
        setIsPlaying(true);
      } else if (audioRef.current) {
        audioRef.current.src = track.path;
        audioRef.current
          .play()
          .then(() => setIsPlaying(true))
          .catch(() => {
            // Audio streaming fallback
            synthRef.current.start(isMuted ? 0 : volume);
            setIsPlaying(true);
          });
      }
    },
    [tracks, isMuted, volume, isRepeat]
  );

  const playNext = useCallback(() => {
    if (tracks.length === 0) return;
    const next = isShuffle
      ? Math.floor(Math.random() * tracks.length)
      : (currentIndex + 1) % tracks.length;
    playTrackAt(next);
  }, [tracks.length, isShuffle, currentIndex, playTrackAt]);

  const playPrev = useCallback(() => {
    if (tracks.length === 0) return;
    const prev = (currentIndex - 1 + tracks.length) % tracks.length;
    playTrackAt(prev);
  }, [tracks.length, currentIndex, playTrackAt]);

  const searchAndPlayYoutube = useCallback(
    async (query: string, openBrowser = false) => {
      const q = (query || '').trim();
      if (!q) return;
      setIsSearchingYoutube(true);
      setError(null);
      try {
        const res = await apiFetch(`/api/youtube/search?q=${encodeURIComponent(q)}&limit=5`);
        if (!res.ok) throw new Error(`Search failed: HTTP ${res.status}`);
        const data = await res.json();
        const results = Array.isArray(data?.results) ? data.results : [];
        if (results.length === 0 || results[0].error) {
          throw new Error(results[0]?.error || 'No YouTube results found');
        }

        const top = results[0];
        const newTrack: EnrichedMusicTrack = {
          id: `youtube-${top.id}-${Date.now()}`,
          title: top.title || q,
          artist: top.channel || 'YouTube',
          duration: 240,
          path: top.url || `https://www.youtube.com/watch?v=${top.id}`,
          sizeBytes: 0,
          category: 'youtube',
          badge: 'YOUTUBE LIVE',
          youtubeId: top.id,
          thumbnail: top.thumbnail,
        };

        setTracks((prev) => [newTrack, ...prev.filter((t) => t.youtubeId !== top.id)]);
        setCurrentIndex(0);
        setCurrentTime(0);
        setDuration(240);
        setIsPlaying(true);
        setSelectedCategory('youtube');

        if (openBrowser) {
          void apiFetch('/api/youtube/play', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_id: top.id, open_browser: true }),
          });
        }
      } catch (err: any) {
        console.warn('[jarvis:youtube] Playback error:', err);
        setError(`YouTube Error: ${err.message || 'Playback failed'}`);
        const fallbackIdx = findBestTrackIndex(q, tracks);
        playTrackAt(fallbackIdx);
      } finally {
        setIsSearchingYoutube(false);
      }
    },
    [tracks, playTrackAt]
  );

  useEffect(() => {
    synthRef.current.setVolume(isMuted ? 0 : volume);
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  // Clean up audio on unmount
  useEffect(() => {
    return () => {
      synthRef.current.stop();
      clearInterval(timerRef.current);
    };
  }, []);

  // Broadcast state changes for external controls / mini-players
  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent('jarvis:music:state', {
        detail: {
          isPlaying,
          currentTrack,
          currentIndex,
          currentTime,
          duration,
        },
      })
    );
  }, [isPlaying, currentTrack, currentIndex, currentTime, duration]);

  // Listen for global Jarvis music commands (voice or text dispatch)
  useEffect(() => {
    const handleMusicCommand = (e: Event) => {
      const customEvent = e as CustomEvent<MusicCommandAction>;
      const detail = customEvent.detail;
      if (!detail) return;

      if (detail.action === 'pause') {
        if (isPlaying) togglePlay();
      } else if (detail.action === 'resume') {
        if (!isPlaying) togglePlay();
      } else if (detail.action === 'next') {
        playNext();
      } else if (detail.action === 'prev') {
        playPrev();
      } else if (detail.action === 'play') {
        if (detail.isYouTube && detail.query) {
          void searchAndPlayYoutube(detail.query);
        } else {
          const q = (detail.query || '').trim();
          if (/youtube|يوتيوب/i.test(q)) {
            void searchAndPlayYoutube(q);
          } else {
            const targetIdx = findBestTrackIndex(q, tracks);
            playTrackAt(targetIdx);
          }
        }
      }
    };

    window.addEventListener('jarvis:music:command', handleMusicCommand);
    return () => {
      window.removeEventListener('jarvis:music:command', handleMusicCommand);
    };
  }, [tracks, isPlaying, togglePlay, playNext, playPrev, playTrackAt, searchAndPlayYoutube]);

  // React to prop command if passed
  useEffect(() => {
    if (!command?.request) return;
    const req = command.request.trim();
    if (/youtube|يوتيوب/i.test(req)) {
      void searchAndPlayYoutube(req);
    } else {
      const targetIdx = findBestTrackIndex(req, tracks);
      playTrackAt(targetIdx);
    }
  }, [command, tracks, playTrackAt, searchAndPlayYoutube]);

  // Filtered tracks by category and search
  const filteredTracks = useMemo(() => {
    return tracks.filter((t) => {
      const matchCategory =
        selectedCategory === 'all' || (t.category && t.category === selectedCategory);
      const matchSearch =
        !searchQuery ||
        t.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.artist.toLowerCase().includes(searchQuery.toLowerCase());
      return matchCategory && matchSearch;
    });
  }, [tracks, selectedCategory, searchQuery]);

  // Handle local file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const newTracks: EnrichedMusicTrack[] = Array.from(files).map((f, i) => ({
      id: `local-${Date.now()}-${i}`,
      title: f.name.replace(/\.[^/.]+$/, ''),
      artist: 'Local Audio File',
      duration: 180,
      path: URL.createObjectURL(f),
      sizeBytes: f.size,
      category: 'cyber',
      badge: 'LOCAL',
    }));

    setTracks((prev) => [...newTracks, ...prev]);
    playTrackAt(0);
  };

  const progress = duration && duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="w-full h-full flex flex-col rounded-xl bg-[#040d1a]/95 border border-[#00f0ff]/30 shadow-[0_0_25px_rgba(0,240,255,0.08)] overflow-hidden font-mono text-xs">
      {/* Header Deck */}
      <div className="px-4 py-3 border-b border-[#00f0ff]/20 bg-[#071526]/80 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="size-8 rounded-lg bg-[#00f0ff]/10 border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff] shadow-[0_0_10px_rgba(0,240,255,0.3)]">
            <Music2 className={`size-4 ${isPlaying ? 'animate-pulse text-[#00f0ff]' : 'text-slate-400'}`} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-[#00f0ff] tracking-wide text-sm">JARVIS AUDIO DECK</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-amber-400/10 text-amber-300 border border-amber-400/30 flex items-center gap-1 font-bold">
                <Sparkles className="size-2.5" /> HOLOGRAPHIC & VOICE-ENABLED
              </span>
            </div>
            <p className="text-[11px] text-[#80f7ff]/60">
              Hamza Namira • Iron Man Theme • Billie Eilish • Stark Lab Synthesizer
            </p>
          </div>
        </div>

        {/* Search, YouTube & Upload Actions */}
        <div className="flex items-center gap-2 flex-1 max-w-2xl justify-end flex-wrap">
          {/* Direct YouTube Search Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (youtubeSearchInput.trim()) {
                void searchAndPlayYoutube(youtubeSearchInput.trim());
                setYoutubeSearchInput('');
              }
            }}
            className="flex items-center gap-1.5"
          >
            <div className="relative">
              <Youtube className="size-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-red-500" />
              <input
                type="text"
                placeholder="Search & play YouTube..."
                value={youtubeSearchInput}
                onChange={(e) => setYoutubeSearchInput(e.target.value)}
                className="w-[130px] sm:w-[180px] pl-7 pr-2 py-1 bg-[#020b14] border border-red-500/30 rounded-lg text-slate-100 placeholder:text-red-400/40 focus:outline-none focus:border-red-500 text-[11px]"
              />
            </div>
            <button
              type="submit"
              disabled={isSearchingYoutube || !youtubeSearchInput.trim()}
              className="px-2.5 py-1 rounded bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-500/40 flex items-center gap-1 text-[11px] font-bold transition-all shrink-0 disabled:opacity-40"
              title="Search and play YouTube video immediately"
            >
              {isSearchingYoutube ? (
                <Sparkles className="size-3 animate-spin text-amber-400" />
              ) : (
                <Play className="size-3" />
              )}
              <span className="hidden sm:inline">Play YT</span>
            </button>
          </form>

          <div className="relative flex-1 max-w-[140px]">
            <Search className="size-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-cyan-400/50" />
            <input
              type="text"
              placeholder="Filter library..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-2.5 py-1 bg-[#020b14] border border-[#00f0ff]/25 rounded-lg text-slate-100 placeholder:text-cyan-400/40 focus:outline-none focus:border-[#00f0ff] text-[11px]"
            />
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            multiple
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-2.5 py-1 rounded bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/30 flex items-center gap-1.5 transition-all text-xs shrink-0"
            title="Upload audio tracks"
          >
            <Upload className="size-3" /> Load Audio
          </button>
        </div>
      </div>

      {/* Category Filter Pills */}
      <div className="px-4 py-2 bg-[#05111f]/90 border-b border-[#00f0ff]/15 flex items-center gap-1.5 overflow-x-auto">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 ${
            selectedCategory === 'all'
              ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/40 font-bold shadow-[0_0_10px_rgba(0,240,255,0.2)]'
              : 'text-slate-400 hover:text-cyan-300 border border-transparent'
          }`}
        >
          All Tracks ({tracks.length})
        </button>
        <button
          onClick={() => setSelectedCategory('youtube')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 flex items-center gap-1.5 ${
            selectedCategory === 'youtube'
              ? 'bg-red-600/25 text-red-300 border border-red-500/50 font-bold shadow-[0_0_12px_rgba(239,68,68,0.3)]'
              : 'text-slate-400 hover:text-red-300 border border-transparent'
          }`}
        >
          <Youtube className="size-3 text-red-500" />
          YouTube ({tracks.filter((t) => t.category === 'youtube' || t.youtubeId).length})
        </button>
        <button
          onClick={() => setSelectedCategory('ironman')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 ${
            selectedCategory === 'ironman'
              ? 'bg-red-500/20 text-red-300 border border-red-500/40 font-bold shadow-[0_0_10px_rgba(239,68,68,0.2)]'
              : 'text-slate-400 hover:text-red-300 border border-transparent'
          }`}
        >
          Iron Man & JARVIS
        </button>
        <button
          onClick={() => setSelectedCategory('hamza')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 ${
            selectedCategory === 'hamza'
              ? 'bg-amber-400/20 text-amber-300 border border-amber-400/40 font-bold shadow-[0_0_10px_rgba(251,191,36,0.2)]'
              : 'text-slate-400 hover:text-amber-300 border border-transparent'
          }`}
        >
          حمزة نمرة (Hamza Namira)
        </button>
        <button
          onClick={() => setSelectedCategory('billie')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 ${
            selectedCategory === 'billie'
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold shadow-[0_0_10px_rgba(168,85,247,0.2)]'
              : 'text-slate-400 hover:text-emerald-300 border border-transparent'
          }`}
        >
          Billie Eilish
        </button>
        <button
          onClick={() => setSelectedCategory('cyber')}
          className={`px-2.5 py-1 rounded-md text-[11px] transition-all shrink-0 ${
            selectedCategory === 'cyber'
              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 font-bold shadow-[0_0_10px_rgba(168,85,247,0.2)]'
              : 'text-slate-400 hover:text-purple-300 border border-transparent'
          }`}
        >
          Stark Cyber Synth
        </button>
      </div>

      {/* Now Playing Banner */}
      <div className="p-4 bg-[#051424]/90 border-b border-[#00f0ff]/15">
        {currentTrack ? (
          <div>
            <div className="flex items-center gap-3">
              <div className="size-14 rounded-xl bg-[#071d33] border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff] shrink-0 shadow-[0_0_15px_rgba(0,240,255,0.25)] relative overflow-hidden">
                {currentTrack.thumbnail ? (
                  <img
                    src={currentTrack.thumbnail}
                    alt={currentTrack.title}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <Disc3
                    className={`size-8 ${isPlaying ? 'animate-spin' : ''}`}
                    style={{ animationDuration: '4s' }}
                  />
                )}
                {isPlaying && (
                  <div className="absolute inset-0 rounded-xl border border-cyan-400/50 animate-ping pointer-events-none opacity-40" />
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-sm font-bold text-cyan-100 truncate">{currentTrack.title}</p>
                  {currentTrack.badge && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-400/10 text-amber-300 border border-amber-400/30 font-bold shrink-0">
                      {currentTrack.badge}
                    </span>
                  )}
                  {currentTrack.youtubeId && (
                    <a
                      href={currentTrack.path}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[10px] px-2 py-0.5 rounded bg-red-600/20 hover:bg-red-600/35 text-red-300 border border-red-500/40 font-bold shrink-0 flex items-center gap-1 transition-all"
                      title="Open video in YouTube"
                    >
                      <ExternalLink className="size-2.5" /> Watch on YouTube
                    </a>
                  )}
                </div>
                <p className="text-[11px] text-cyan-400/60 truncate flex items-center gap-1 mt-0.5">
                  <Radio className="size-3 text-amber-400" /> {currentTrack.artist}
                </p>
              </div>
              <div className="text-right shrink-0">
                <p className="text-xs text-amber-400 font-bold tabular-nums">
                  {formatDuration(currentTime)}
                </p>
                <p className="text-[10px] text-cyan-400/50 tabular-nums">
                  {formatDuration(duration ?? currentTrack.duration)}
                </p>
              </div>
            </div>

            {/* Embedded YouTube Holographic Viewport when YouTube track is selected */}
            {currentTrack.youtubeId && (
              <div className="mt-3 relative w-full aspect-video max-h-[260px] bg-black rounded-lg overflow-hidden border border-red-500/40 shadow-[0_0_25px_rgba(239,68,68,0.25)]">
                <iframe
                  src={`https://www.youtube-nocookie.com/embed/${currentTrack.youtubeId}?autoplay=1&enablejsapi=1`}
                  title={currentTrack.title}
                  className="w-full h-full border-0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            )}
          </div>
        ) : (
          <p className="text-xs text-cyan-400/50 py-2 text-center tracking-widest">NO TRACK SELECTED</p>
        )}

        {/* Seek Progress Bar */}
        <div className="mt-4">
          <input
            type="range"
            min={0}
            max={duration && duration > 0 ? duration : 100}
            step={0.5}
            value={currentTime}
            onChange={(e) => {
              const val = Number(e.target.value);
              setCurrentTime(val);
              if (audioRef.current && currentTrack?.path !== 'synthetic-ambient') {
                audioRef.current.currentTime = val;
              }
            }}
            className="w-full h-1.5 appearance-none rounded-full bg-cyan-950 outline-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #00f0ff ${progress}%, rgba(0,240,255,0.15) ${progress}%)`,
            }}
          />
        </div>

        {/* Transport Controls */}
        <div className="mt-3 flex items-center justify-center gap-4">
          <button
            onClick={() => setIsShuffle((prev) => !prev)}
            className={`p-2 rounded-lg transition-all ${
              isShuffle
                ? 'bg-amber-400/20 text-[#ffb700] border border-amber-400/40'
                : 'text-cyan-400/50 hover:text-cyan-300'
            }`}
            title="Toggle Shuffle"
          >
            <Shuffle className="size-4" />
          </button>

          <button
            onClick={playPrev}
            className="p-2 text-cyan-300 hover:text-amber-400 transition-colors"
            title="Previous Track"
          >
            <SkipBack className="size-5" />
          </button>

          <button
            onClick={togglePlay}
            className="size-12 rounded-full bg-gradient-to-br from-[#00f0ff] to-[#0088ff] text-[#040d1a] flex items-center justify-center shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:shadow-[0_0_30px_rgba(0,240,255,0.6)] transition-all font-bold"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="size-5" /> : <Play className="size-5 ml-0.5" />}
          </button>

          <button
            onClick={playNext}
            className="p-2 text-cyan-300 hover:text-amber-400 transition-colors"
            title="Next Track"
          >
            <SkipForward className="size-5" />
          </button>

          <button
            onClick={() => setIsRepeat((prev) => !prev)}
            className={`p-2 rounded-lg transition-all ${
              isRepeat
                ? 'bg-amber-400/20 text-[#ffb700] border border-amber-400/40'
                : 'text-cyan-400/50 hover:text-cyan-300'
            }`}
            title="Toggle Repeat"
          >
            <Repeat className="size-4" />
          </button>
        </div>

        {/* Volume Bar */}
        <div className="mt-3 flex items-center gap-2 max-w-xs mx-auto">
          <button
            onClick={() => setIsMuted(!isMuted)}
            className="text-cyan-400 hover:text-cyan-200 transition-colors"
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted || volume === 0 ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.02}
            value={isMuted ? 0 : volume}
            onChange={(e) => {
              const val = Number(e.target.value);
              setVolume(val);
              setIsMuted(val === 0);
            }}
            className="w-full h-1 appearance-none rounded-full bg-cyan-950 outline-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, #00f0ff ${
                (isMuted ? 0 : volume) * 100
              }%, rgba(0,240,255,0.15) ${(isMuted ? 0 : volume) * 100}%)`,
            }}
          />
        </div>
      </div>

      {/* Track List Queue */}
      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
        <div className="flex items-center justify-between px-2 pb-1 text-[10px] tracking-widest text-cyan-400/60 font-bold uppercase">
          <div className="flex items-center gap-2">
            <ListMusic className="size-3.5" />
            AUDIO LIBRARY QUEUE ({filteredTracks.length} of {tracks.length})
          </div>
          <span className="text-[10px] text-amber-400/70 font-normal">
            قول لجارفيس: "شغل حمزة نمرة" أو "شغل أيرون مان"
          </span>
        </div>

        {error && (
          <div className="p-2 rounded bg-red-950/40 border border-red-500/30 text-red-300 text-[11px] flex items-center gap-2">
            <ShieldAlert className="size-4 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {filteredTracks.map((t) => {
          const originalIdx = tracks.findIndex((orig) => orig.id === t.id);
          const isSelected = originalIdx === currentIndex;
          return (
            <button
              key={t.id}
              onClick={() => playTrackAt(originalIdx !== -1 ? originalIdx : 0)}
              className={`w-full flex items-center justify-between p-2.5 rounded-lg text-left transition-all border ${
                isSelected
                  ? 'bg-[#00f0ff]/15 border-[#00f0ff]/50 text-[#00f0ff] shadow-[0_0_12px_rgba(0,240,255,0.15)]'
                  : 'bg-[#06182c]/40 border-transparent hover:border-[#00f0ff]/20 text-slate-300'
              }`}
            >
              <div className="flex items-center gap-2.5 min-w-0 flex-1">
                <span className="size-5 rounded flex items-center justify-center text-[10px] bg-[#00f0ff]/10 text-[#00f0ff] shrink-0 font-bold">
                  {isSelected && isPlaying ? '▶' : originalIdx + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="truncate font-semibold text-xs text-slate-200">{t.title}</p>
                    {t.badge && (
                      <span className="text-[8px] px-1.5 py-0.2 rounded bg-amber-400/10 text-amber-300 border border-amber-400/30 font-bold shrink-0">
                        {t.badge}
                      </span>
                    )}
                  </div>
                  <p className="truncate text-[10px] text-cyan-400/60">{t.artist}</p>
                </div>
              </div>
              <span className="text-[10px] text-cyan-400/50 tabular-nums ml-2 shrink-0">
                {formatDuration(t.duration)}
              </span>
            </button>
          );
        })}
      </div>

      <audio
        ref={audioRef}
        preload="metadata"
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onEnded={playNext}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onError={(e) => {
          console.warn('HTMLAudioElement stream note:', e);
        }}
      />
    </div>
  );
};

export default MusicPlayerWidget;
