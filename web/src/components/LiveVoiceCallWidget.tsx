import React, { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Bot,
  Brain,
  Calendar,
  Clock,
  History,
  LoaderCircle,
  MessageSquare,
  Mic,
  MicOff,
  Phone,
  PhoneOff,
  Play,
  Plus,
  Power,
  Radio,
  Search,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Timer,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Zap,
} from 'lucide-react';
import { authedFetch, type SessionInfo } from '@/lib/api';
import { JarvisUltronVoiceOrb, type OrbTheme } from './JarvisUltronVoiceOrb';
import {
  detectLanguageContent,
  getVoicesSafely,
  pickArabicVoice,
  pickEnglishVoice,
  sanitizeTextForSpeech,
  splitTextIntoSentences,
  cleanSpokenText,
  extractNextSpokenSentence,
  PipelinedAudioQueue,
} from '../lib/speechUtils';
import { parseMusicCommand, findBestTrackIndex } from '../utils/musicCommander';
import {
  playKnockSound,
  playArcReactorBootSound,
  playWakeChime,
  playStandbyChime,
} from '@/lib/ironManAudioFX';
import { matchWakeWord } from '@/utils/wakeWordMatcher';
import { DEFAULT_DEMO_TRACKS } from './MusicPlayerWidget';
import {
  createCleanAudioPipeline,
  AudioSpeechRecorder,
  type CleanAudioPipeline,
} from '../lib/audioCleaningUtils';

type CallStatus = 'idle' | 'starting' | 'active';
type Speaker = 'user' | 'assistant' | 'system';
export type VoicePersona = 'jarvis' | 'gwen';
export type CallLanguage = 'Arabic' | 'English' | 'Auto';
export type TurnMode = 'hands_free' | 'push_to_talk';
export type PauseTolerance = 'relaxed' | 'balanced' | 'fast';
export type AudioLatencyMode = 'instant' | 'studio';

export interface CallMessage {

  id: string;
  sender: Speaker;
  text: string;
  persona?: VoicePersona;
  timestamp: string;
}

export interface LiveVoiceCallWidgetProps {
  initialPersona?: VoicePersona;
  onSendMessage?: (
    text: string,
    persona: VoicePersona,
    language?: CallLanguage,
    onDelta?: (partial: string) => void,
  ) => Promise<string>;
  onClose?: () => void;
  activeSessionId?: string | null;
  activeSessionTitle?: string;
  callSessions?: SessionInfo[];
  isHistoryLoading?: boolean;
  onSelectSession?: (sessionId: string) => void;
  onNewSession?: () => void;
  onDeleteSession?: (sessionId: string) => void;
  onRefreshSessions?: () => void;
  initialMessages?: CallMessage[];
}

const messageId = () => Math.random().toString(36).substring(2, 9);

export const LiveVoiceCallWidget: React.FC<LiveVoiceCallWidgetProps> = ({
  initialPersona = 'jarvis',
  onSendMessage,
  onClose,
  activeSessionId,
  activeSessionTitle,
  callSessions = [],
  isHistoryLoading = false,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  onRefreshSessions,
  initialMessages,
}) => {
  const [status, setStatus] = useState<CallStatus>('idle');
  const [selectedPersona, setSelectedPersona] = useState<VoicePersona>(initialPersona);
  const [selectedLanguage, setSelectedLanguage] = useState<CallLanguage>('Arabic');
  const [showHistory, setShowHistory] = useState(false);
  const [historySearch, setHistorySearch] = useState('');
  const [isMuted, setIsMuted] = useState(false);
  const [speakerOn, setSpeakerOn] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [typedInput, setTypedInput] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [micStatus, setMicStatus] = useState('Microphone standby');
  const [recognitionStatus, setRecognitionStatus] = useState('Speech engine ready');
  const [detectedLanguage, setDetectedLanguage] = useState<'English' | 'Arabic'>('Arabic');
  const [durationSeconds, setDurationSeconds] = useState(0);
  const [messages, setMessages] = useState<CallMessage[]>(
    initialMessages && initialMessages.length > 0
      ? initialMessages
      : [
          {
            id: messageId(),
            sender: 'system',
            text: 'Jarvis & Gwen Live Voice Sentinel ready. Press "Start Call" to initiate hands-free voice conversation with persistent memory.',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]
  );

  const [turnMode, setTurnMode] = useState<TurnMode>(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      return (localStorage.getItem('JARVIS_VOICE_TURN_MODE') as TurnMode) || 'hands_free';
    }
    return 'hands_free';
  });
  const [pauseTolerance, setPauseTolerance] = useState<PauseTolerance>(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      return (localStorage.getItem('JARVIS_VOICE_PAUSE_TOLERANCE') as PauseTolerance) || 'fast';
    }
    return 'fast';
  });
  const [antiInterruptionShield, setAntiInterruptionShield] = useState<boolean>(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('JARVIS_VOICE_ANTI_INTERRUPTION');
      return saved !== null ? saved === 'true' : true;
    }
    return true;
  });
  const [pauseCountdown, setPauseCountdown] = useState<number | null>(null);
  const [liveVolume, setLiveVolume] = useState<number>(0);

  // Audio latency mode: 'instant' (streaming sentence-level browser voice) vs 'studio' (server neural)
  const [audioLatencyMode, setAudioLatencyMode] = useState<AudioLatencyMode>(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      return (localStorage.getItem('JARVIS_AUDIO_LATENCY_MODE') as AudioLatencyMode) || 'instant';
    }
    return 'instant';
  });
  const audioLatencyModeRef = useRef<AudioLatencyMode>(audioLatencyMode);

  useEffect(() => {
    audioLatencyModeRef.current = audioLatencyMode;
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('JARVIS_AUDIO_LATENCY_MODE', audioLatencyMode);
    }
  }, [audioLatencyMode]);

  const activeAudioQueueRef = useRef<PipelinedAudioQueue | null>(null);

  // 24/7 Always-On Iron Man Ambient Mode
  const [alwaysOnMode, setAlwaysOnMode] = useState<boolean>(() => {

    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem('JARVIS_ALWAYS_ON_MODE');
      return saved !== null ? saved === 'true' : true;
    }
    return true;
  });
  const [isAmbientStandby, setIsAmbientStandby] = useState<boolean>(false);
  const [hasGreeted, setHasGreeted] = useState<boolean>(false);

  const alwaysOnModeRef = useRef<boolean>(alwaysOnMode);
  const isAmbientStandbyRef = useRef<boolean>(isAmbientStandby);
  const hasGreetedRef = useRef<boolean>(hasGreeted);
  const idleTimeoutRef = useRef<any>(null);

  useEffect(() => {
    alwaysOnModeRef.current = alwaysOnMode;
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('JARVIS_ALWAYS_ON_MODE', String(alwaysOnMode));
    }
  }, [alwaysOnMode]);

  useEffect(() => {
    isAmbientStandbyRef.current = isAmbientStandby;
  }, [isAmbientStandby]);

  useEffect(() => {
    hasGreetedRef.current = hasGreeted;
  }, [hasGreeted]);

  const messagesRef = useRef<CallMessage[]>(messages);
  const statusRef = useRef<CallStatus>('idle');
  const mutedRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const recognitionRunningRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const sessionVersionRef = useRef(0);
  const conversationEndRef = useRef<HTMLDivElement | null>(null);

  const languageRef = useRef<CallLanguage>('Arabic');
  const personaRef = useRef<VoicePersona>(selectedPersona);
  const isSpeakingRef = useRef(false);
  const isProcessingRef = useRef(false);
  const speakerOnRef = useRef(speakerOn);

  useEffect(() => {
    personaRef.current = selectedPersona;
  }, [selectedPersona]);

  useEffect(() => {
    languageRef.current = selectedLanguage;
  }, [selectedLanguage]);
  const silenceTimerRef = useRef<any>(null);
  const countdownIntervalRef = useRef<any>(null);
  const pauseDeadlineRef = useRef<number | null>(null);
  const speechAccumulatorRef = useRef<string>('');
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const cleanPipelineRef = useRef<CleanAudioPipeline | null>(null);
  const audioRecorderRef = useRef<AudioSpeechRecorder | null>(null);
  const [isWhisperFallback, setIsWhisperFallback] = useState<boolean>(false);
  const isWhisperFallbackRef = useRef<boolean>(false);
  const speechAudioCtxRef = useRef<AudioContext | null>(null);
  const speechAnalyserRef = useRef<AnalyserNode | null>(null);
  const [orbTheme, setOrbTheme] = useState<OrbTheme>(selectedPersona === 'gwen' ? 'gwen' : 'jarvis');

  useEffect(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('JARVIS_VOICE_TURN_MODE', turnMode);
    }
  }, [turnMode]);

  useEffect(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('JARVIS_VOICE_PAUSE_TOLERANCE', pauseTolerance);
    }
  }, [pauseTolerance]);

  useEffect(() => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('JARVIS_VOICE_ANTI_INTERRUPTION', String(antiInterruptionShield));
    }
  }, [antiInterruptionShield]);

  useEffect(() => {
    void getVoicesSafely();
  }, []);

  useEffect(() => {
    if (status !== 'active' || isMuted) {
      setLiveVolume(0);
      return;
    }
    let animId: number;
    const sample = () => {
      if (analyserRef.current) {
        const buffer = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i++) sum += buffer[i];
        const avg = sum / buffer.length;
        setLiveVolume(Math.min(100, Math.round((avg / 128) * 100)));
      }
      animId = requestAnimationFrame(sample);
    };
    sample();
    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [status, isMuted]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (initialMessages && initialMessages.length > 0) {
      setMessages(initialMessages);
      messagesRef.current = initialMessages;
    }
  }, [initialMessages]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    mutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    languageRef.current = selectedLanguage;
  }, [selectedLanguage]);

  useEffect(() => {
    speakerOnRef.current = speakerOn;
  }, [speakerOn]);

  useEffect(() => {
    isSpeakingRef.current = isSpeaking;
  }, [isSpeaking]);

  useEffect(() => {
    isProcessingRef.current = isProcessing;
  }, [isProcessing]);

  const replaceMessages = useCallback((next: CallMessage[]) => {
    messagesRef.current = next;
    setMessages(next);
  }, []);

  const appendMessage = useCallback(
    (sender: Speaker, text: string, persona?: VoicePersona) => {
      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const next = [...messagesRef.current, { id: messageId(), sender, text, persona, timestamp }];
      replaceMessages(next);
    },
    [replaceMessages]
  );

  const rearmTimerRef = useRef<any>(null);

  const stopSpeaking = useCallback(() => {
    if (activeAudioQueueRef.current) {
      activeAudioQueueRef.current.abort();
      activeAudioQueueRef.current = null;
    }
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.src = '';
      } catch {
        // ignore
      }
      audioRef.current = null;
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }
    setIsSpeaking(false);
    isSpeakingRef.current = false;
  }, []);


  const stopRecognition = useCallback(() => {
    if (rearmTimerRef.current) {
      clearTimeout(rearmTimerRef.current);
      rearmTimerRef.current = null;
    }
    const rec = recognitionRef.current;
    if (rec) {
      try {
        rec.onstart = null;
        rec.onresult = null;
        rec.onerror = null;
        rec.onend = null;
        rec.stop();
      } catch {
        // ignore
      }
      try {
        rec.abort();
      } catch {
        // ignore
      }
    }
    recognitionRef.current = null;
    recognitionRunningRef.current = false;
  }, []);

  const speakWithBrowser = useCallback(
    async (text: string): Promise<void> => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      const langResult = detectLanguageContent(text);
      const effectivePersona: VoicePersona = personaRef.current || selectedPersona;
      const voices = await getVoicesSafely();
      const isArabicSpeech =
        languageRef.current === 'Arabic'
          ? true
          : languageRef.current === 'English'
          ? false
          : langResult.isArabicPredominant;
      const voice = isArabicSpeech
        ? pickArabicVoice(voices, effectivePersona)
        : pickEnglishVoice(voices, effectivePersona);

      if (isArabicSpeech && !voice) {
        const anyArabic = voices.find((v) => v.lang.toLowerCase().includes('ar'));
        if (anyArabic) {
          const sentences = splitTextIntoSentences(text);
          for (const sentence of sentences) {
            if (statusRef.current !== 'active' || !speakerOnRef.current || !isSpeakingRef.current) break;
            await new Promise<void>((resolve) => {
              const utterance = new SpeechSynthesisUtterance(sentence);
              utterance.lang = anyArabic.lang || 'ar-EG';
              utterance.voice = anyArabic;
              utterance.rate = 1.05;
              utterance.pitch = effectivePersona === 'gwen' ? 1.05 : 0.95;
              utterance.onend = () => resolve();
              utterance.onerror = () => resolve();
              window.speechSynthesis.speak(utterance);
            });
          }
          return;
        }
      }

      const sentences = splitTextIntoSentences(text);
      if (sentences.length === 0) return;

      for (const sentence of sentences) {
        if (statusRef.current !== 'active' || !speakerOnRef.current || !isSpeakingRef.current) break;

        await new Promise<void>((resolve) => {
          const utterance = new SpeechSynthesisUtterance(sentence);
          utterance.lang = isArabicSpeech ? 'ar-EG' : 'en-US';
          utterance.rate = 1.05;
          utterance.pitch = effectivePersona === 'gwen' ? 1.05 : 0.95;
          if (voice) utterance.voice = voice;

          let keepAliveTimer: any = null;
          const cleanup = () => {
            if (keepAliveTimer) clearInterval(keepAliveTimer);
            resolve();
          };

          utterance.onend = cleanup;
          utterance.onerror = cleanup;

          keepAliveTimer = setInterval(() => {
            if (!window.speechSynthesis.speaking) {
              clearInterval(keepAliveTimer);
            } else {
              window.speechSynthesis.resume();
            }
          }, 1500);

          window.speechSynthesis.speak(utterance);
        });
      }
    },
    [selectedPersona]
  );

  const speakSingleSentenceBrowser = useCallback(
    async (rawSentence: string, persona: VoicePersona = 'jarvis', signal?: AbortSignal): Promise<void> => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window) || signal?.aborted) {
        return;
      }
      const text = sanitizeTextForSpeech(rawSentence);
      if (!text) return;

      const langResult = detectLanguageContent(text);
      const isArabicSpeech =
        languageRef.current === 'Arabic'
          ? true
          : languageRef.current === 'English'
          ? false
          : langResult.isArabicPredominant;

      const voices = await getVoicesSafely();
      const voice = isArabicSpeech
        ? pickArabicVoice(voices, persona)
        : pickEnglishVoice(voices, persona);

      await new Promise<void>((resolve) => {
        if (signal?.aborted) {
          resolve();
          return;
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = isArabicSpeech ? 'ar-EG' : 'en-US';
        utterance.rate = 1.06;
        utterance.pitch = persona === 'gwen' ? 1.05 : 0.95;
        if (voice) utterance.voice = voice;

        let keepAliveTimer: any = null;
        let safetyTimer: any = null;
        let settled = false;

        const cleanup = () => {
          if (settled) return;
          settled = true;
          if (keepAliveTimer) clearInterval(keepAliveTimer);
          if (safetyTimer) clearTimeout(safetyTimer);
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };

        const onAbort = () => {
          try {
            window.speechSynthesis.cancel();
          } catch {}
          cleanup();
        };

        signal?.addEventListener('abort', onAbort, { once: true });

        utterance.onstart = () => {
          try {
            if (window.speechSynthesis.paused) {
              window.speechSynthesis.resume();
            }
          } catch {}
        };

        utterance.onend = cleanup;
        utterance.onerror = cleanup;

        // Safety timeout in case browser drops onend
        const maxDurationMs = Math.max(3000, Math.min(25000, text.length * 120));
        safetyTimer = setTimeout(cleanup, maxDurationMs);

        keepAliveTimer = setInterval(() => {
          if (!window.speechSynthesis.speaking) {
            clearInterval(keepAliveTimer);
          } else {
            window.speechSynthesis.resume();
          }
        }, 800);

        try {
          if (window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
          }
        } catch {}

        window.speechSynthesis.speak(utterance);
      });
    },
    []
  );

  const speakSingleSentenceServer = useCallback(
    async (rawSentence: string, persona: VoicePersona, signal?: AbortSignal): Promise<void> => {
      if (signal?.aborted) return;
      const text = sanitizeTextForSpeech(rawSentence);
      if (!text) return;

      try {
        const timeoutController = new AbortController();
        const timeoutId = setTimeout(() => timeoutController.abort(), 800);

        const onSignalAbort = () => {
          timeoutController.abort();
        };
        signal?.addEventListener('abort', onSignalAbort, { once: true });

        const response = await authedFetch('/api/audio/speak', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            persona,
            language: languageRef.current,
          }),
          signal: timeoutController.signal,
        }).catch(() => null);

        clearTimeout(timeoutId);
        signal?.removeEventListener('abort', onSignalAbort);

        if (response && response.ok && !signal?.aborted) {
          const data = (await response.json().catch(() => null)) as {
            ok?: boolean;
            data_url?: string;
          } | null;

          if (data?.ok && data.data_url && !signal?.aborted) {
            const audio = new Audio(data.data_url);
            audioRef.current = audio;

            try {
              const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
              if (AudioContextClass) {
                let sCtx = speechAudioCtxRef.current;
                if (!sCtx || sCtx.state === 'closed') {
                  sCtx = new AudioContextClass();
                  speechAudioCtxRef.current = sCtx;
                }
                if (sCtx.state === 'suspended') {
                  void sCtx.resume();
                }
                const source = sCtx.createMediaElementSource(audio);
                const sAnalyser = sCtx.createAnalyser();
                sAnalyser.fftSize = 256;
                source.connect(sAnalyser);
                sAnalyser.connect(sCtx.destination);
                speechAnalyserRef.current = sAnalyser;
              }
            } catch (e) {
              console.warn('[Orb WebAudio] connect notice:', e);
            }

            await new Promise<void>((resolve) => {
              if (signal?.aborted) {
                try {
                  audio.pause();
                  audio.src = '';
                } catch {}
                resolve();
                return;
              }
              const done = () => {
                audioRef.current = null;
                speechAnalyserRef.current = null;
                resolve();
              };
              const onAbort = () => {
                try {
                  audio.pause();
                  audio.src = '';
                } catch {}
                done();
              };
              signal?.addEventListener('abort', onAbort, { once: true });
              audio.onended = () => {
                signal?.removeEventListener('abort', onAbort);
                done();
              };
              audio.onerror = () => {
                signal?.removeEventListener('abort', onAbort);
                done();
              };
              audio.play().catch(done);
            });
            return;
          }
        }

        // Fast zero-latency fallback to browser speech synthesis
        if (!signal?.aborted) {
          await speakSingleSentenceBrowser(text, persona, signal);
        }
      } catch {
        if (!signal?.aborted) {
          await speakSingleSentenceBrowser(text, persona, signal);
        }
      }
    },
    [speakSingleSentenceBrowser]
  );

  const speakSentence = useCallback(
    async (
      sentence: string,
      persona: VoicePersona,
      latencyMode: AudioLatencyMode,
      signal?: AbortSignal
    ): Promise<void> => {
      if (!sentence || !speakerOnRef.current || statusRef.current !== 'active' || signal?.aborted) {
        return;
      }

      if (latencyMode === 'instant') {
        await speakSingleSentenceBrowser(sentence, persona, signal);
        return;
      }

      // Studio / Server Neural mode
      await speakSingleSentenceServer(sentence, persona, signal);
    },
    [speakSingleSentenceBrowser, speakSingleSentenceServer]
  );

  const scheduleRearm = useCallback((delayMs = 150) => {
    if (rearmTimerRef.current) {
      clearTimeout(rearmTimerRef.current);
    }
    rearmTimerRef.current = setTimeout(() => {
      if (
        statusRef.current === 'active' &&
        !mutedRef.current &&
        !isSpeakingRef.current &&
        !isProcessingRef.current
      ) {
        startRecognition();
      }
    }, delayMs);
  }, []);

  const speak = useCallback(
    async (rawText: string, forcedPersona?: VoicePersona): Promise<void> => {
      const text = sanitizeTextForSpeech(rawText);
      if (!text || !speakerOnRef.current || statusRef.current !== 'active') return;

      stopSpeaking();
      stopRecognition();
      setIsSpeaking(true);
      isSpeakingRef.current = true;

      // Strictly obey the persona chosen by the user
      const effectivePersona: VoicePersona = forcedPersona || personaRef.current || selectedPersona;

      setOrbTheme(effectivePersona === 'gwen' ? 'gwen' : 'jarvis');
      setMicStatus(
        effectivePersona === 'gwen'
          ? 'Gwen (ElevenLabs Female AI) is speaking...'
          : 'Jarvis (Male AI) is speaking...'
      );

      // Hermes-style latency: in instant mode the browser voice starts in
      // ~100ms. The old path always paid a server TTS round-trip (up to 15s)
      // before falling back to the browser — greetings and error notices felt
      // frozen. Server neural audio is a studio-mode choice, not the default.
      if (audioLatencyModeRef.current === 'instant') {
        try {
          await speakWithBrowser(text);
        } finally {
          setIsSpeaking(false);
          isSpeakingRef.current = false;
          if (statusRef.current === 'active' && !mutedRef.current && !isProcessingRef.current) {
            setMicStatus('Listening...');
            scheduleRearm(150);
            if (alwaysOnModeRef.current && !isAmbientStandbyRef.current) {
              resetIdleStandbyTimer();
            }
          }
        }
        return;
      }

      try {
        const controller = new AbortController();
        const abortTimeout = setTimeout(() => controller.abort(), 8000);
        const response = await authedFetch('/api/audio/speak', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text,
            persona: effectivePersona,
            language: languageRef.current,
          }),
          signal: controller.signal,
        }).catch(() => null);
        clearTimeout(abortTimeout);

        if (response && response.ok) {
          const data = (await response.json().catch(() => null)) as {
            ok?: boolean;
            data_url?: string;
          } | null;

          if (data?.ok && data.data_url && isSpeakingRef.current) {
            const audio = new Audio(data.data_url);
            audioRef.current = audio;

            // Route audio through Web Audio API to drive the Ultron/Jarvis Orb with real frequencies
            try {
              const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
              if (AudioContextClass) {
                let sCtx = speechAudioCtxRef.current;
                if (!sCtx || sCtx.state === 'closed') {
                  sCtx = new AudioContextClass();
                  speechAudioCtxRef.current = sCtx;
                }
                if (sCtx.state === 'suspended') {
                  void sCtx.resume();
                }
                const source = sCtx.createMediaElementSource(audio);
                const sAnalyser = sCtx.createAnalyser();
                sAnalyser.fftSize = 256;
                source.connect(sAnalyser);
                sAnalyser.connect(sCtx.destination);
                speechAnalyserRef.current = sAnalyser;
              }
            } catch (webAudioErr) {
              console.warn('[Orb WebAudio] media source connect notice:', webAudioErr);
            }

            await new Promise<void>((resolve) => {
              const done = () => {
                audioRef.current = null;
                speechAnalyserRef.current = null;
                resolve();
              };
              audio.onended = done;
              audio.onerror = done;
              audio.play().catch(done);
            });
            return;
          }
        }
        // If server synthesis fails, only fallback to browser synthesis if persona is Jarvis
        await speakWithBrowser(text);
      } catch {
        await speakWithBrowser(text);
      } finally {
        setIsSpeaking(false);
        isSpeakingRef.current = false;
        if (statusRef.current === 'active' && !mutedRef.current && !isProcessingRef.current) {
          setMicStatus('Listening...');
          scheduleRearm(150);
          if (alwaysOnModeRef.current && !isAmbientStandbyRef.current) {
            resetIdleStandbyTimer();
          }
        }
      }
    },
    [scheduleRearm, selectedPersona, speakWithBrowser, stopRecognition, stopSpeaking]
  );

  const enterAmbientStandby = useCallback(() => {
    if (!alwaysOnModeRef.current || statusRef.current !== 'active' || isAmbientStandbyRef.current) return;
    playStandbyChime();
    setIsAmbientStandby(true);
    isAmbientStandbyRef.current = true;
    setMicStatus('Standby engaged: 24/7 ambient listening for "Wake up Jarvis" / "اصحى يا جارفيس"...');
  }, []);

  const resetIdleStandbyTimer = useCallback(() => {
    if (idleTimeoutRef.current) {
      clearTimeout(idleTimeoutRef.current);
      idleTimeoutRef.current = null;
    }
    if (!alwaysOnModeRef.current || statusRef.current !== 'active' || isAmbientStandbyRef.current) return;
    idleTimeoutRef.current = setTimeout(() => {
      if (
        statusRef.current === 'active' &&
        !isSpeakingRef.current &&
        !isProcessingRef.current &&
        !interimTranscript
      ) {
        enterAmbientStandby();
      }
    }, 45000);
  }, [enterAmbientStandby, interimTranscript]);

  const getGreetingText = useCallback((isArabicLang: boolean) => {
    const hour = new Date().getHours();
    if (isArabicLang) {
      if (hour >= 5 && hour < 12) {
        return 'صباح الخير يا فندم. كافة الأنظمة تعمل بكفاءة، وجارفيس مستعد للأوامر.';
      } else if (hour >= 12 && hour < 18) {
        return 'مساء الخير يا فندم. كافة الأنظمة تعمل بكفاءة، وجارفيس في الخدمة ومستعد.';
      } else {
        return 'أهلاً يا فندم. الأنظمة تعمل بكفاءة عالية، وجارفيس في الخدمة ومستعد لأي مهمة.';
      }
    } else {
      if (hour >= 5 && hour < 12) {
        return 'Good morning, sir. All systems operational. Jarvis is standing by.';
      } else if (hour >= 12 && hour < 18) {
        return 'Good afternoon, sir. All systems operational. Jarvis is standing by.';
      } else {
        return 'Good evening, sir. All systems operational. Jarvis is standing by.';
      }
    }
  }, []);

  const triggerWakeGreeting = useCallback(
    async (isWakeFromStandby = false) => {
      playArcReactorBootSound();
      setIsAmbientStandby(false);
      isAmbientStandbyRef.current = false;

      const isAr =
        languageRef.current === 'Arabic' ||
        (languageRef.current === 'Auto' &&
          typeof navigator !== 'undefined' &&
          !(navigator.language || '').startsWith('en'));

      const greetingText = isWakeFromStandby
        ? (isAr ? 'تحت أمرك يا فندم، أنا سامعك وشغال معاك.' : 'At your service, sir. How may I assist you?')
        : getGreetingText(isAr);

      appendMessage('assistant', greetingText, selectedPersona);
      await speak(greetingText, selectedPersona);
    },
    [appendMessage, getGreetingText, selectedPersona, speak]
  );

  const sendChatTurn = useCallback(
    async (text: string, sessionVersion: number, activePersona?: VoicePersona) => {
      let reply = '';
      const personaToUse: VoicePersona = activePersona || personaRef.current || selectedPersona;
      const langResult = detectLanguageContent(text);
      const isAr =
        languageRef.current === 'Arabic'
          ? true
          : languageRef.current === 'English'
          ? false
          : langResult.isArabicPredominant;

      const musicCmd = parseMusicCommand(text);
      if (musicCmd) {
        window.dispatchEvent(
          new CustomEvent('jarvis:music:command', {
            detail: musicCmd,
          })
        );

        if (musicCmd.action === 'pause') {
          reply = isAr
            ? (personaToUse === 'gwen' ? 'وقفتلك الموسيقى يا باشا تماماً.' : 'تم إيقاف تشغيل الموسيقى يا فندم فوراً.')
            : (personaToUse === 'gwen' ? 'Music paused, boss!' : 'Music playback suspended, sir.');
        } else if (musicCmd.action === 'resume') {
          reply = isAr
            ? (personaToUse === 'gwen' ? 'رجعت شغلتلك التراك يا باشا.' : 'تم استئناف تشغيل الموسيقى يا فندم.')
            : (personaToUse === 'gwen' ? 'Resumed audio playback, boss!' : 'Resuming music playback, sir.');
        } else if (musicCmd.action === 'next') {
          reply = isAr
            ? (personaToUse === 'gwen' ? 'شغلتلك التراك اللي بعده يا باشا.' : 'جاري الانتقال إلى التراك التالي يا فندم.')
            : (personaToUse === 'gwen' ? 'Advancing to next track, boss!' : 'Advancing to the next track, sir.');
        } else if (musicCmd.action === 'prev') {
          reply = isAr
            ? (personaToUse === 'gwen' ? 'رجعتلك للتراك اللي قبله يا باشا.' : 'جاري الرجوع إلى التراك السابق يا فندم.')
            : (personaToUse === 'gwen' ? 'Previous track coming up, boss!' : 'Returning to previous track, sir.');
        } else {
          const targetIdx = findBestTrackIndex(musicCmd.query || '', DEFAULT_DEMO_TRACKS);
          const matched = DEFAULT_DEMO_TRACKS[targetIdx];
          const trackTitle = matched ? matched.title : (musicCmd.query || 'التراك المختار');
          reply = isAr
            ? (personaToUse === 'gwen' ? `عيوني يا باشا! بشغلك ${trackTitle} حالاً.` : `تحت أمرك يا فندم، جاري تشغيل ${trackTitle} فوراً.`)
            : (personaToUse === 'gwen' ? `Playing ${trackTitle} for you, boss!` : `Right away, sir. Commencing playback of ${trackTitle}.`);
        }

        // Notify backend session asynchronously so memory reflects the command
        if (onSendMessage) {
          onSendMessage(text, personaToUse, languageRef.current).catch(() => {});
        }
      } else if (onSendMessage) {
        const audioQueue = new PipelinedAudioQueue((speaking) => {
          setIsSpeaking(speaking);
          isSpeakingRef.current = speaking;
          if (speaking) {
            setOrbTheme(personaToUse === 'gwen' ? 'gwen' : 'jarvis');
            setMicStatus(
              personaToUse === 'gwen'
                ? 'Gwen is speaking...'
                : 'Jarvis is speaking...'
            );
          }
        });
        activeAudioQueueRef.current = audioQueue;

        let spokenCharIndex = 0;

        try {
          const streamingId = messageId();
          let streamingShown = false;
          reply = await onSendMessage(text, personaToUse, languageRef.current, (partial) => {
            if (sessionVersion !== sessionVersionRef.current || statusRef.current !== 'active') return;
            const effective: VoicePersona = personaRef.current || personaToUse || selectedPersona;
            const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const clean = cleanSpokenText(partial);
            const displayText =
              clean.trim() ||
              (partial.includes('<think')
                ? (isAr ? '⚡ جاري التفكير...' : '⚡ Jarvis is thinking...')
                : partial);
            const next: CallMessage = { id: streamingId, sender: 'assistant', text: displayText, persona: effective, timestamp };
            if (!streamingShown) {
              streamingShown = true;
              replaceMessages([...messagesRef.current, next]);
            } else {
              replaceMessages(messagesRef.current.map((m) => (m.id === streamingId ? next : m)));
            }

            // Extract complete sentences or fast clauses from the partial stream
            while (true) {
              const res = extractNextSpokenSentence(clean, spokenCharIndex);
              if (!res) break;
              spokenCharIndex = res.nextIndex;
              const sentence = sanitizeTextForSpeech(res.sentence);
              if (sentence.length >= 2) {
                audioQueue.enqueue((signal) =>
                  speakSentence(sentence, effective, audioLatencyModeRef.current, signal)
                );
              }
            }
          });

          // Drop the streaming placeholder; the final append below renders
          // the authoritative reply (and avoids a duplicate bubble).
          if (streamingShown) {
            replaceMessages(messagesRef.current.filter((m) => m.id !== streamingId));
          }

          const effectivePersona: VoicePersona = personaRef.current || personaToUse || selectedPersona;
          const finalClean = cleanSpokenText(reply);
          const remainder = sanitizeTextForSpeech(finalClean.slice(spokenCharIndex));
          if (remainder.length >= 2) {
            audioQueue.enqueue((signal) =>
              speakSentence(remainder, effectivePersona, audioLatencyModeRef.current, signal)
            );
          }

          if (sessionVersion === sessionVersionRef.current && statusRef.current === 'active') {
            appendMessage('assistant', reply, effectivePersona);
          }

          // Wait until all sentence audio finishes playing
          await audioQueue.waitUntilDone();
        } catch (err) {
          console.warn('[voice] onSendMessage failed:', err);
          const errMsg = err instanceof Error ? err.message : String(err || 'Communication error');
          if (personaToUse === 'gwen') {
            reply = isAr
              ? `عذراً يا باشا، واجهت مشكلة في الاتصال بالنظام (${errMsg}). تقدر تكرر كلامك؟`
              : `Pardon me, boss! I encountered a connection issue (${errMsg}). Could you repeat your question?`;
          } else {
            reply = isAr
              ? `عذراً يا فندم، تعذر إتمام المعالجة عبر النواة المركزية (${errMsg}). يرجى تكرار الأمر.`
              : `Apologies, sir. Unable to communicate with the core intelligence (${errMsg}). Please repeat your request.`;
          }
          if (sessionVersion === sessionVersionRef.current && statusRef.current === 'active') {
            const effectivePersona: VoicePersona = personaRef.current || personaToUse || selectedPersona;
            appendMessage('assistant', reply, effectivePersona);
            await speak(reply, effectivePersona);
          }
        } finally {
          activeAudioQueueRef.current = null;
        }
      } else {
        if (!reply) {
          if (personaToUse === 'gwen') {
            reply = isAr
              ? `أهلاً يا باشا! أنا جوين مع حضرتك وسامعاك كويس جداً.`
              : `Hello boss! Gwen here, standing by for your instructions.`;
          } else {
            reply = isAr
              ? `تحت أمرك يا فندم، جارفيس في الخدمة وبانتظار توجيهاتك.`
              : `At your service, sir. Systems operational and standing by for your command.`;
          }
        }

        if (sessionVersion === sessionVersionRef.current && statusRef.current === 'active') {
          const effectivePersona: VoicePersona = personaRef.current || personaToUse || selectedPersona;
          appendMessage('assistant', reply, effectivePersona);
          await speak(reply, effectivePersona);
        }
      }
    },
    [appendMessage, onSendMessage, replaceMessages, selectedPersona, speak, speakSentence]
  );

  const submitTurn = useCallback(
    async (rawText: string) => {
      const text = rawText.trim();
      if (!text || isProcessingRef.current) return;
      if (statusRef.current !== 'active') {
        appendMessage('system', 'Please start the call session first by clicking "Start Call".');
        return;
      }

      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      setPauseCountdown(null);
      pauseDeadlineRef.current = null;
      speechAccumulatorRef.current = '';

      stopSpeaking();
      stopRecognition();
      setInterimTranscript('');

      // Language detection for display
      const langResult = detectLanguageContent(text);
      setDetectedLanguage(langResult.detectedLanguage);

      const activePersona: VoicePersona = personaRef.current || selectedPersona;

      appendMessage('user', text);

      setIsProcessing(true);
      isProcessingRef.current = true;
      setMicStatus(
        activePersona === 'gwen'
          ? 'Gwen is analyzing your request...'
          : 'Jarvis is processing your request...'
      );

      const sessionVersion = sessionVersionRef.current;
      try {
        await sendChatTurn(text, sessionVersion, activePersona);
      } catch (error) {
        if (sessionVersion === sessionVersionRef.current && statusRef.current === 'active') {
          const detail = error instanceof Error && error.message ? ` ${error.message}` : '';
          appendMessage('system', `Could not reach AI voice engine.${detail}`);
        }
      } finally {
        if (sessionVersion === sessionVersionRef.current) {
          setIsProcessing(false);
          isProcessingRef.current = false;
          if (statusRef.current === 'active' && !mutedRef.current && !isSpeakingRef.current) {
            setMicStatus('Listening...');
            scheduleRearm(150);
          }
        }
      }
    },
    [appendMessage, scheduleRearm, sendChatTurn, stopRecognition, stopSpeaking]
  );

  const getPauseTimeoutMs = useCallback(
    (textDraft: string): number | null => {
      if (turnMode === 'push_to_talk') {
        return null; // Push-to-talk never times out automatically
      }

      let baseMs = 250; // Ultra snappy 250ms default
      if (pauseTolerance === 'relaxed') {
        baseMs = 900;
      } else if (pauseTolerance === 'balanced') {
        baseMs = 500;
      } else if (pauseTolerance === 'fast') {
        baseMs = 250;
      }

      if (!antiInterruptionShield) {
        return baseMs;
      }

      let dynamicBonus = 0;
      const trimmed = textDraft.trim();
      const words = trimmed.split(/\s+/).filter(Boolean);

      // Detect mid-sentence continuation conjunctions and connectors
      const lastWord =
        words.length > 0
          ? words[words.length - 1].toLowerCase().replace(/[،,.:;!؟?]/g, '')
          : '';

      const continuationWordsAr = [
        'و', 'او', 'أو', 'ثم', 'ف', 'علشان', 'عشان', 'بس', 'لكن', 'يعني',
        'مع', 'في', 'من', 'عن', 'على', 'الي', 'إلى', 'إن', 'ان', 'انك', 'لو',
        'لما', 'حتى', 'قبل', 'بعد', 'بدل', 'زي', 'معلش', 'طيب', 'يا', 'ما'
      ];
      const continuationWordsEn = [
        'and', 'or', 'but', 'because', 'cause', 'so', 'then', 'if', 'when',
        'while', 'where', 'like', 'with', 'that', 'which', 'who', 'also',
        'actually', 'well', 'um', 'uh', 'er', 'ah', 'the', 'a', 'an', 'to'
      ];

      const isContinuation =
        continuationWordsAr.includes(lastWord) ||
        continuationWordsEn.includes(lastWord) ||
        lastWord.startsWith('و') ||
        lastWord.startsWith('ف');

      if (isContinuation) {
        dynamicBonus += 150;
      }

      // Hard cap at 750ms max so turns are always submitted promptly
      return Math.min(750, baseMs + dynamicBonus);
    },
    [antiInterruptionShield, pauseTolerance, turnMode]
  );


  const handleSendNow = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    setPauseCountdown(null);
    pauseDeadlineRef.current = null;

    const draft = (speechAccumulatorRef.current || interimTranscript).trim();
    if (draft && statusRef.current === 'active' && !isProcessingRef.current && !isSpeakingRef.current) {
      speechAccumulatorRef.current = '';
      setInterimTranscript('');
      setMicStatus('Processing speech turn...');
      void submitTurn(draft);
    }
  }, [interimTranscript, submitTurn]);

  const handleExtendPause = useCallback((extraMs = 2000) => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    const currentDraft = (speechAccumulatorRef.current || interimTranscript).trim();
    if (!currentDraft) return;

    const baseRemaining = pauseDeadlineRef.current
      ? Math.max(0, pauseDeadlineRef.current - Date.now())
      : 2000;
    const newTimeoutMs = baseRemaining + extraMs;
    const newDeadline = Date.now() + newTimeoutMs;
    pauseDeadlineRef.current = newDeadline;
    setPauseCountdown(Math.ceil(newTimeoutMs / 1000));

    silenceTimerRef.current = setTimeout(() => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
      setPauseCountdown(null);
      pauseDeadlineRef.current = null;

      const finalSpokenText = (speechAccumulatorRef.current || currentDraft).trim();
      if (
        finalSpokenText &&
        statusRef.current === 'active' &&
        !isProcessingRef.current &&
        !isSpeakingRef.current
      ) {
        speechAccumulatorRef.current = '';
        setInterimTranscript('');
        setMicStatus('Processing speech turn...');
        void submitTurn(finalSpokenText);
      }
    }, newTimeoutMs);
  }, [interimTranscript, submitTurn]);

  const handleCancelDraft = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    speechAccumulatorRef.current = '';
    setInterimTranscript('');
    setPauseCountdown(null);
    pauseDeadlineRef.current = null;
    setMicStatus('Listening...');
  }, []);

  const handleInterruptJarvis = useCallback(() => {
    stopSpeaking();
    if (statusRef.current === 'active' && !mutedRef.current && !isProcessingRef.current) {
      setMicStatus('Jarvis interrupted. Listening...');
      scheduleRearm(150);
    }
  }, [scheduleRearm, stopSpeaking]);

  const createRecognition = useCallback(() => {
    if (typeof window === 'undefined') return null;
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setRecognitionStatus('Speech recognition unavailable — use typed input fallback');
      return null;
    }

    // Clean up old instance if present
    if (recognitionRef.current) {
      try {
        recognitionRef.current.onstart = null;
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.abort();
      } catch {
        // ignore
      }
      recognitionRef.current = null;
    }

    const recognition = new SpeechRec();
    recognition.continuous = true;
    recognition.interimResults = true;
    const currentLang = languageRef.current;
    if (currentLang === 'English') {
      recognition.lang = 'en-US';
    } else if (currentLang === 'Arabic') {
      recognition.lang = 'ar-EG';
    } else {
      const navLang = typeof navigator !== 'undefined' ? (navigator.language || '') : '';
      recognition.lang = navLang.startsWith('en') ? 'en-US' : 'ar-EG';
    }

    recognition.onstart = () => {
      recognitionRunningRef.current = true;
      const lang = languageRef.current;
      setRecognitionStatus(
        `Listening actively (${lang === 'English' ? 'English' : lang === 'Arabic' ? 'Arabic - مصرية' : 'Auto - Adaptive'})`
      );
      if (!isSpeakingRef.current && !isProcessingRef.current) {
        setMicStatus('Listening...');
      }
    };

    recognition.onresult = (event: any) => {
      if (isSpeakingRef.current || isProcessingRef.current) {
        return;
      }

      let capturedChunks = '';
      let hasFinal = false;
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const chunk = String(event.results[index][0]?.transcript || '').trim();
        if (event.results[index].isFinal) {
          hasFinal = true;
          speechAccumulatorRef.current = speechAccumulatorRef.current
            ? `${speechAccumulatorRef.current} ${chunk}`
            : chunk;
        } else {
          capturedChunks = capturedChunks ? `${capturedChunks} ${chunk}` : chunk;
        }
      }

      const currentDraft = (
        speechAccumulatorRef.current + (capturedChunks ? ` ${capturedChunks}` : '')
      ).trim();

      if (currentDraft) {
        if (isAmbientStandbyRef.current) {
          const wakeCheck = matchWakeWord(currentDraft);
          if (wakeCheck.isWake) {
            speechAccumulatorRef.current = '';
            setInterimTranscript('');
            playWakeChime();

            if (wakeCheck.command) {
              setIsAmbientStandby(false);
              isAmbientStandbyRef.current = false;
              setMicStatus(`Wake command received: "${wakeCheck.command}"`);
              void submitTurn(wakeCheck.command);
            } else {
              void triggerWakeGreeting(true);
            }
            return;
          }
          setMicStatus('Ambient Standby: Listening for "Wake up Jarvis" / "اصحى يا جارفيس"...');
          return;
        }

        // Check for Standby command while in active conversation mode
        const standbyCheck = matchWakeWord(currentDraft);
        if (standbyCheck.isStandby) {
          speechAccumulatorRef.current = '';
          setInterimTranscript('');
          enterAmbientStandby();
          appendMessage(
            'system',
            'Jarvis entered ambient standby mode. Say "Wake up Jarvis" or tap Arc Reactor to resume.'
          );
          return;
        }

        setInterimTranscript(currentDraft);
        setMicStatus(`Listening: "${currentDraft.slice(-45)}"`);

        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }

        const calculatedTimeout = getPauseTimeoutMs(currentDraft);
        const timeoutMs =
          calculatedTimeout !== null
            ? hasFinal && !capturedChunks
              ? Math.min(calculatedTimeout, 200)
              : calculatedTimeout
            : null;

        if (timeoutMs !== null) {
          const deadline = Date.now() + timeoutMs;
          pauseDeadlineRef.current = deadline;
          setPauseCountdown(Math.ceil(timeoutMs / 1000));

          countdownIntervalRef.current = setInterval(() => {
            if (!pauseDeadlineRef.current) {
              setPauseCountdown(null);
              return;
            }
            const remainingMs = Math.max(0, pauseDeadlineRef.current - Date.now());
            const remSecs = Math.ceil(remainingMs / 1000);
            setPauseCountdown(remSecs);
            if (remainingMs <= 0) {
              if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
                countdownIntervalRef.current = null;
              }
            }
          }, 200);

          silenceTimerRef.current = setTimeout(() => {
            if (countdownIntervalRef.current) {
              clearInterval(countdownIntervalRef.current);
              countdownIntervalRef.current = null;
            }
            setPauseCountdown(null);
            pauseDeadlineRef.current = null;

            const finalSpokenText = (
              speechAccumulatorRef.current || currentDraft
            ).trim();

            if (
              finalSpokenText &&
              statusRef.current === 'active' &&
              !isProcessingRef.current &&
              !isSpeakingRef.current
            ) {
              speechAccumulatorRef.current = '';
              setInterimTranscript('');
              setMicStatus('Processing speech turn...');
              void submitTurn(finalSpokenText);
            }
          }, timeoutMs);
        } else {
          setPauseCountdown(null);
        }
      }
    };

    recognition.onerror = (event: any) => {
      const errType = event.error || 'unknown';
      if (errType === 'not-allowed' || errType === 'service-not-allowed') {
        setRecognitionStatus('Microphone permission denied — typed input active');
      } else if (errType === 'network') {
        setRecognitionStatus('Web Speech API network offline — Hermes Whisper STT standby');
        setIsWhisperFallback(true);
        isWhisperFallbackRef.current = true;
      } else if (errType !== 'aborted' && errType !== 'no-speech') {
        setRecognitionStatus(`Speech recognition note: ${errType}`);
      }
    };

    recognition.onend = () => {
      recognitionRunningRef.current = false;
      if (
        statusRef.current === 'active' &&
        !mutedRef.current &&
        !isSpeakingRef.current &&
        !isProcessingRef.current
      ) {
        scheduleRearm(200);
      }
    };

    recognitionRef.current = recognition;
    return recognition;
  }, [
    appendMessage,
    enterAmbientStandby,
    getPauseTimeoutMs,
    scheduleRearm,
    submitTurn,
    triggerWakeGreeting,
  ]);

  const startRecognition = useCallback(() => {
    if (
      statusRef.current !== 'active' ||
      mutedRef.current ||
      isSpeakingRef.current ||
      isProcessingRef.current
    ) {
      return;
    }

    try {
      let recognition = recognitionRef.current;
      if (!recognition || !recognitionRunningRef.current) {
        recognition = createRecognition();
      }
      if (!recognition) return;

      const currentLang = languageRef.current;
      if (currentLang === 'English') {
        recognition.lang = 'en-US';
      } else if (currentLang === 'Arabic') {
        recognition.lang = 'ar-EG';
      } else {
        const navLang = typeof navigator !== 'undefined' ? (navigator.language || '') : '';
        recognition.lang = navLang.startsWith('en') ? 'en-US' : 'ar-EG';
      }
      recognition.start();
      recognitionRunningRef.current = true;
    } catch (err: any) {
      // Fix deadlock: Never set recognitionRunningRef.current = true on error!
      recognitionRunningRef.current = false;
      recognitionRef.current = null;
      scheduleRearm(400);
    }
  }, [createRecognition, scheduleRearm]);

  const handleLanguageChange = useCallback(
    (lang: CallLanguage) => {
      setSelectedLanguage(lang);
      languageRef.current = lang;
      stopSpeaking();
      stopRecognition();
      const langLabel =
        lang === 'Arabic' ? 'Arabic - مصرية' : lang === 'English' ? 'English (US)' : 'Auto Adaptive';
      setMicStatus(`Language switched to ${langLabel}`);
      setRecognitionStatus(`Listening mode updated: ${langLabel}`);
      if (statusRef.current === 'active' && !isProcessingRef.current) {
        scheduleRearm(350);
      }
    },
    [scheduleRearm, stopRecognition, stopSpeaking]
  );

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (targetTag === 'input' || targetTag === 'textarea' || (e.target as HTMLElement)?.isContentEditable) {
        return;
      }

      if (e.code === 'Space') {
        if (isSpeaking) {
          e.preventDefault();
          handleInterruptJarvis();
        } else if (interimTranscript) {
          e.preventDefault();
          handleSendNow();
        }
      } else if (e.code === 'Escape') {
        if (isSpeaking) {
          e.preventDefault();
          handleInterruptJarvis();
        } else if (interimTranscript) {
          e.preventDefault();
          handleCancelDraft();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleCancelDraft, handleInterruptJarvis, handleSendNow, interimTranscript, isSpeaking]);

  // Recognition Watchdog: Automatically re-arm if active and idling
  useEffect(() => {
    if (status !== 'active') return;

    const watchdog = setInterval(() => {
      if (
        statusRef.current === 'active' &&
        !mutedRef.current &&
        !isSpeakingRef.current &&
        !isProcessingRef.current &&
        !recognitionRunningRef.current
      ) {
        startRecognition();
      }
    }, 1500);

    return () => clearInterval(watchdog);
  }, [status, startRecognition]);

  useEffect(() => {
    if (status !== 'active') return;
    const timer = window.setInterval(() => setDurationSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(timer);
  }, [status]);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
  }, [messages, interimTranscript]);

  useEffect(() => {
    return () => {
      sessionVersionRef.current += 1;
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (cleanPipelineRef.current) {
        cleanPipelineRef.current.cleanup();
        cleanPipelineRef.current = null;
      }
      if (audioRecorderRef.current) {
        audioRecorderRef.current.abort();
        audioRecorderRef.current = null;
      }
      if (audioCtxRef.current) {
        try {
          audioCtxRef.current.close();
        } catch {}
        audioCtxRef.current = null;
        analyserRef.current = null;
      }
      if (speechAudioCtxRef.current) {
        try {
          speechAudioCtxRef.current.close();
        } catch {}
        speechAudioCtxRef.current = null;
        speechAnalyserRef.current = null;
      }
      stopSpeaking();
    };
  }, [stopSpeaking]);

  const startCall = useCallback(
    async (isKnockOrAuto = false) => {
      if (statusRef.current !== 'idle') return;
      setStatus('starting');
      statusRef.current = 'starting';
      setMicStatus('Requesting microphone permission...');
      setDurationSeconds(0);
      sessionVersionRef.current += 1;
      // Warm-up in parallel with the mic prompt so the first reply pays no
      // cold-start tax: browser voices resolve now (not mid-sentence), and the
      // TTS engine pre-loads its provider via a lease (see /api/audio/tts-lease).
      void getVoicesSafely().catch(() => []);
      void authedFetch('/api/audio/tts-lease', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lease: 'jarvis-voice-call', active: true }),
      }).catch(() => null);
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
          });
          streamRef.current = stream;

          // Initialize Web Audio DSP Clean Pipeline
          const pipeline = createCleanAudioPipeline(stream);
          if (pipeline) {
            cleanPipelineRef.current = pipeline;
            audioCtxRef.current = pipeline.audioContext;
            analyserRef.current = pipeline.analyser;
          } else {
            try {
              const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
              if (AudioContextClass) {
                const ctx = new AudioContextClass();
                if (ctx.state === 'suspended') {
                  await ctx.resume();
                }
                const srcNode = ctx.createMediaStreamSource(stream);
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 256;
                srcNode.connect(analyser);
                audioCtxRef.current = ctx;
                analyserRef.current = analyser;
              }
            } catch (audioErr) {
              console.warn('[voice] AudioContext init error:', audioErr);
            }
          }
          setMicStatus('Microphone active & DSP Voice Filter engaged');
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : 'permission unavailable';
        setMicStatus(`Microphone unavailable: ${detail}. Typed input active.`);
      }
      statusRef.current = 'active';
      setStatus('active');

      if (!hasGreetedRef.current || isKnockOrAuto) {
        setHasGreeted(true);
        hasGreetedRef.current = true;
        // Greet first; startRecognition will be called safely in speak()'s finally block
        window.setTimeout(() => {
          void triggerWakeGreeting(false);
        }, 150);
      } else {
        appendMessage(
          'assistant',
          selectedPersona === 'gwen'
            ? 'أهلاً بك يا باشا! أنا جوين، الخط المباشر شغال مع جارفيس وهيرميس إيجينت.'
            : 'Jarvis online. Live voice call session initialized with Hermes Agent.',
          selectedPersona
        );
        window.setTimeout(startRecognition, 200);
      }
    },
    [appendMessage, selectedPersona, startRecognition, triggerWakeGreeting]
  );

  const handleKnockToWake = useCallback(async () => {
    playKnockSound();
    if (statusRef.current === 'idle') {
      await startCall(true);
    } else {
      window.setTimeout(() => {
        void triggerWakeGreeting(isAmbientStandbyRef.current);
      }, 150);
    }
  }, [isAmbientStandby, startCall, triggerWakeGreeting]);

  useEffect(() => {
    if (!alwaysOnMode) return;
    const tryAutoBoot = async () => {
      try {
        if (typeof navigator !== 'undefined' && navigator.permissions && navigator.permissions.query) {
          const perm = await navigator.permissions.query({ name: 'microphone' as any }).catch(() => null);
          if (perm && perm.state === 'granted') {
            if (statusRef.current === 'idle') {
              void startCall(true);
            }
          }
        }
      } catch {
        // ignore
      }
    };
    void tryAutoBoot();
  }, [alwaysOnMode, startCall]);

  const endCall = useCallback(() => {
    sessionVersionRef.current += 1;
    statusRef.current = 'idle';
    setStatus('idle');
    setIsAmbientStandby(false);
    isAmbientStandbyRef.current = false;
    setIsProcessing(false);
    setDurationSeconds(0);
    setInterimTranscript('');
    setPauseCountdown(null);
    pauseDeadlineRef.current = null;
    if (idleTimeoutRef.current) {
      clearTimeout(idleTimeoutRef.current);
      idleTimeoutRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    stopSpeaking();
    stopRecognition();
    if (cleanPipelineRef.current) {
      cleanPipelineRef.current.cleanup();
      cleanPipelineRef.current = null;
    }
    if (audioRecorderRef.current) {
      audioRecorderRef.current.abort();
      audioRecorderRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioCtxRef.current) {
      try {
        audioCtxRef.current.close();
      } catch {}
      audioCtxRef.current = null;
      analyserRef.current = null;
    }
    // Release the TTS warm-up lease so resident local models can unload.
    void authedFetch('/api/audio/tts-lease', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lease: 'jarvis-voice-call', active: false }),
    }).catch(() => null);
    setMicStatus('Microphone released');
    appendMessage('system', 'Voice call session terminated.');
  }, [appendMessage, stopRecognition, stopSpeaking]);

  const toggleMicMute = useCallback(() => {
    setIsMuted((prev) => {
      const next = !prev;
      mutedRef.current = next;
      if (streamRef.current) {
        streamRef.current.getAudioTracks().forEach((track) => {
          track.enabled = !next;
        });
      }
      if (next) {
        stopSpeaking();
        stopRecognition();
      } else if (statusRef.current === 'active') {
        scheduleRearm(150);
      }
      return next;
    });
  }, [scheduleRearm, stopRecognition, stopSpeaking]);

  const handleTypedSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!typedInput.trim()) return;
    const text = typedInput.trim();
    setTypedInput('');
    void submitTurn(text);
  };

  const filteredSessions = useMemo(() => {
    if (!historySearch.trim()) return callSessions;
    const q = historySearch.toLowerCase();
    return callSessions.filter(
      (s) =>
        (s.title && s.title.toLowerCase().includes(q)) ||
        (s.preview && s.preview.toLowerCase().includes(q)) ||
        s.id.toLowerCase().includes(q)
    );
  }, [callSessions, historySearch]);

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${remainder.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-[#030712] border border-[#00f0ff]/30 rounded-xl shadow-[0_0_30px_rgba(0,240,255,0.15)] text-[#e5e2e1] overflow-hidden flex flex-col font-sans relative">
      {/* Header Bar */}
      <div className="bg-[#071526] border-b border-[#00f0ff]/20 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className={`w-3 h-3 rounded-full ${status === 'active' ? (isAmbientStandby ? 'bg-amber-400 animate-pulse' : 'bg-[#00f0ff] animate-ping') : 'bg-slate-600'}`} />
            <div className={`w-3 h-3 rounded-full absolute top-0 left-0 ${status === 'active' ? (isAmbientStandby ? 'bg-amber-400' : 'bg-[#00f0ff]') : 'bg-slate-500'}`} />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-wider uppercase text-[#00f0ff] flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[#ffb700]" /> JARVIS LIVE VOICE SENTINEL
            </h2>
            <p className="text-[11px] font-mono text-[#80f7ff]/70">
              Hermes Agent • Status:{' '}
              {status === 'active' && isAmbientStandby ? (
                <span className="text-amber-400 uppercase font-bold">STANDBY (24/7 AMBIENT LISTENING)</span>
              ) : (
                <span className="text-[#00f0ff] uppercase font-bold">{status}</span>
              )}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* 24/7 Always-On Live Mode Pill */}
          <button
            type="button"
            onClick={() => {
              const next = !alwaysOnMode;
              setAlwaysOnMode(next);
              if (!next && isAmbientStandby) {
                setIsAmbientStandby(false);
              }
            }}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold font-mono flex items-center gap-1.5 transition-all ${
              alwaysOnMode
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.25)]'
                : 'bg-[#020b14] border border-slate-700 text-slate-400'
            }`}
            title="24/7 Always-On Mode: Jarvis stays continuously listening in background for 'Wake up Jarvis' / 'اصحى يا جارفيس'"
          >
            <span className={`w-2 h-2 rounded-full ${alwaysOnMode ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
            <span>24/7 Live Mode</span>
          </button>

          {/* Call History Drawer Toggle */}
          <button
            onClick={() => setShowHistory((prev) => !prev)}
            className={`px-2.5 py-1 rounded-lg text-xs font-semibold font-mono flex items-center gap-1.5 transition-all ${
              showHistory
                ? 'bg-[#00f0ff] text-slate-950 shadow-[0_0_12px_rgba(0,240,255,0.4)]'
                : 'bg-[#020b14] border border-[#00f0ff]/40 text-[#80f7ff] hover:bg-[#00f0ff]/10 hover:border-[#00f0ff]'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>History ({callSessions.length})</span>
          </button>

          {/* New Call Button */}
          {onNewSession && (
            <button
              onClick={() => {
                onNewSession();
                setShowHistory(false);
              }}
              title="Start a fresh voice call session"
              className="px-2.5 py-1 rounded-lg text-xs font-semibold font-mono bg-[#00f0ff]/10 border border-[#00f0ff]/30 text-[#00f0ff] hover:bg-[#00f0ff]/20 transition-all flex items-center gap-1"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New</span>
            </button>
          )}

          {/* Language selector tabs */}
          <div className="flex bg-[#030712] border border-[#00f0ff]/30 rounded-lg p-1 text-xs">
            <button
              onClick={() => handleLanguageChange('Arabic')}
              className={`px-2.5 py-1 rounded-md font-semibold transition-all ${
                selectedLanguage === 'Arabic'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                  : 'text-[#80f7ff]/50 hover:text-[#e5e2e1]'
              }`}
            >
              عربي (EG)
            </button>
            <button
              onClick={() => handleLanguageChange('English')}
              className={`px-2.5 py-1 rounded-md font-semibold transition-all ${
                selectedLanguage === 'English'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                  : 'text-[#80f7ff]/50 hover:text-[#e5e2e1]'
              }`}
            >
              English
            </button>
            <button
              onClick={() => handleLanguageChange('Auto')}
              className={`px-2.5 py-1 rounded-md font-semibold transition-all ${
                selectedLanguage === 'Auto'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/50 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                  : 'text-[#80f7ff]/50 hover:text-[#e5e2e1]'
              }`}
            >
              Auto
            </button>
          </div>

          {/* Persona selector tabs */}
          <div className="flex bg-[#030712] border border-[#00f0ff]/30 rounded-lg p-1 text-xs">
            <button
              onClick={() => {
                stopSpeaking();
                setSelectedPersona('jarvis');
                personaRef.current = 'jarvis';
                setOrbTheme('jarvis');
                setMicStatus('Switched to Jarvis (Male AI)');
                if (statusRef.current === 'active' && !isProcessingRef.current) {
                  scheduleRearm(200);
                }
              }}
              className={`px-3 py-1 rounded-md font-semibold transition-all ${
                selectedPersona === 'jarvis'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/50'
                  : 'text-[#80f7ff]/50 hover:text-[#e5e2e1]'
              }`}
            >
              Jarvis (Male AI)
            </button>
            <button
              onClick={() => {
                stopSpeaking();
                setSelectedPersona('gwen');
                personaRef.current = 'gwen';
                setOrbTheme('gwen');
                setMicStatus('Switched to Gwen (ElevenLabs Female AI)');
                if (statusRef.current === 'active' && !isProcessingRef.current) {
                  scheduleRearm(200);
                }
              }}
              className={`px-3 py-1 rounded-md font-semibold transition-all ${
                selectedPersona === 'gwen'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                  : 'text-[#80f7ff]/50 hover:text-[#e5e2e1]'
              }`}
            >
              Gwen (ElevenLabs Female AI)
            </button>
          </div>

          {onClose && (
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Body Grid */}
      <div className="p-4 space-y-4 flex-1 overflow-y-auto">
        {/* Cold-Load Iron Man HUD Audio Unlock Prompt */}
        {status === 'idle' && alwaysOnMode && (
          <div
            onClick={handleKnockToWake}
            className="cursor-pointer bg-gradient-to-r from-amber-500/15 via-[#071d33] to-cyan-500/15 border border-amber-400/50 hover:border-amber-400 rounded-xl p-3 flex items-center justify-between gap-3 shadow-[0_0_25px_rgba(245,158,11,0.2)] transition-all group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-amber-400/20 border border-amber-400/40 flex items-center justify-center text-amber-300 group-hover:scale-110 transition-transform">
                <Zap className="w-4 h-4 text-amber-300 animate-pulse" />
              </div>
              <div>
                <p className="text-xs font-bold text-amber-200">
                  ⚡ 24/7 IRON MAN LIVE MODE READY
                </p>
                <p className="text-[11px] text-slate-300 font-mono">
                  Click or tap anywhere here to initialize JARVIS HUD audio & wake up
                </p>
              </div>
            </div>
            <button
              type="button"
              className="px-3.5 py-1.5 bg-amber-400 hover:bg-amber-300 text-slate-950 font-bold rounded-lg text-xs flex items-center gap-1.5 shadow-[0_0_12px_rgba(245,158,11,0.4)] transition-all"
            >
              <Power className="w-3.5 h-3.5" />
              Initialize & Wake Up
            </button>
          </div>
        )}

        {/* Ambient Standby 24/7 Listening Banner */}
        {status === 'active' && isAmbientStandby && (
          <div className="bg-gradient-to-r from-[#00f0ff]/15 via-[#071d33] to-[#00f0ff]/15 border border-[#00f0ff]/50 rounded-xl p-3 flex items-center justify-between gap-3 shadow-[0_0_20px_rgba(0,240,255,0.2)] animate-in fade-in duration-200">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#00f0ff] animate-ping" />
              <div>
                <p className="text-xs font-bold text-[#00f0ff]">
                  JARVIS AMBIENT STANDBY — 24/7 ALWAYS LISTENING
                </p>
                <p className="text-[11px] text-[#80f7ff]/70 font-mono">
                  Say <span className="text-white font-semibold">"Wake up Jarvis"</span> or <span className="text-white font-semibold">"اصحى يا جارفيس"</span>, or click Arc Reactor below
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleKnockToWake}
              className="px-3.5 py-1.5 bg-[#00f0ff]/20 hover:bg-[#00f0ff]/30 border border-[#00f0ff] text-[#00f0ff] font-bold rounded-lg text-xs flex items-center gap-1.5 transition-all cursor-pointer shadow-[0_0_10px_rgba(0,240,255,0.3)]"
            >
              <Zap className="w-3.5 h-3.5 text-amber-300" />
              Wake Up (Knock)
            </button>
          </div>
        )}

        {/* Active Session & Memory Bar */}
        <div className="bg-[#020b14] border border-[#00f0ff]/20 px-3 py-2 rounded-lg flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-[#00f0ff] animate-pulse" />
            <span className="text-[#80f7ff]/70">Session:</span>
            <span className="text-[#00f0ff] font-bold truncate max-w-xs md:max-w-md">
              {activeSessionTitle || (activeSessionId ? `Session ${activeSessionId.slice(0, 14)}` : 'Live Continuous Session')}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-[#80f7ff]/60">
            <span className="flex items-center gap-1 text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Persistent Memory Active
            </span>
            <span>•</span>
            <span>{messages.filter((m) => m.sender !== 'system').length} turns</span>
          </div>
        </div>

        {/* Voice Cadence & Anti-Interruption Control Bar */}
        <div className="bg-[#040f1d] border border-[#00f0ff]/25 px-3.5 py-2.5 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-[#00f0ff]" />
            <span className="font-semibold text-slate-200">Voice Cadence & Shield:</span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Mode selection: Hands-Free vs Push-to-Talk */}
            <div className="flex bg-[#020b14] border border-[#00f0ff]/30 rounded-lg p-0.5 text-[11px] font-mono">
              <button
                type="button"
                onClick={() => setTurnMode('hands_free')}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  turnMode === 'hands_free'
                    ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_8px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Jarvis waits patiently for pauses before responding"
              >
                Hands-Free (Smart Pauses)
              </button>
              <button
                type="button"
                onClick={() => setTurnMode('push_to_talk')}
                className={`px-2.5 py-1 rounded-md transition-all ${
                  turnMode === 'push_to_talk'
                    ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_8px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Zero auto-interruptions: Send via Spacebar or button when finished"
              >
                Push-to-Talk (Zero Cutoffs)
              </button>
            </div>

            {/* Pause tolerance pills (shown when in hands_free mode) */}
            {turnMode === 'hands_free' && (
              <div className="flex items-center gap-1 bg-[#020b14] border border-[#00f0ff]/30 rounded-lg p-0.5 text-[11px] font-mono">
                <span className="text-[10px] text-[#80f7ff]/60 px-1.5 flex items-center gap-1">
                  <Clock className="w-3 h-3 text-[#00f0ff]" /> Pause:
                </span>
                <button
                  type="button"
                  onClick={() => setPauseTolerance('relaxed')}
                  className={`px-2 py-0.5 rounded transition-all ${
                    pauseTolerance === 'relaxed'
                      ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="3.0s pause tolerance — Very patient, perfect for thinking while speaking"
                >
                  Relaxed (3.0s)
                </button>
                <button
                  type="button"
                  onClick={() => setPauseTolerance('balanced')}
                  className={`px-2 py-0.5 rounded transition-all ${
                    pauseTolerance === 'balanced'
                      ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold border border-[#00f0ff]/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="2.2s pause tolerance — Balanced conversational pacing"
                >
                  Balanced (2.2s)
                </button>
                <button
                  type="button"
                  onClick={() => setPauseTolerance('fast')}
                  className={`px-2 py-0.5 rounded transition-all ${
                    pauseTolerance === 'fast'
                      ? 'bg-amber-500/20 text-amber-300 font-bold border border-amber-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="1.3s pause tolerance — Fast command responsiveness"
                >
                  Fast (1.3s)
                </button>
              </div>
            )}

            {/* Anti-Interruption Shield Toggle */}
            <button
              type="button"
              onClick={() => setAntiInterruptionShield((prev) => !prev)}
              className={`px-2.5 py-1 rounded-lg border text-[11px] font-mono flex items-center gap-1.5 transition-all ${
                antiInterruptionShield
                  ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.2)]'
                  : 'bg-slate-900 border-slate-700 text-slate-500'
              }`}
              title="Detects mid-sentence pauses, short intros, and continuation conjunctions (and, but, because, و, علشان) so Jarvis never interrupts"
            >
              <ShieldCheck className={`w-3.5 h-3.5 ${antiInterruptionShield ? 'text-emerald-400' : 'text-slate-500'}`} />
              Anti-Interruption Shield: {antiInterruptionShield ? 'Active' : 'Off'}
            </button>

            {/* Audio Latency / Stream TTS Engine Mode */}
            <div className="flex bg-[#020b14] border border-[#00f0ff]/30 rounded-lg p-0.5 text-[11px] font-mono">
              <button
                type="button"
                onClick={() => setAudioLatencyMode('instant')}
                className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1 ${
                  audioLatencyMode === 'instant'
                    ? 'bg-amber-400/20 text-amber-300 font-bold border border-amber-400/40 shadow-[0_0_8px_rgba(245,158,11,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="⚡ Instant Zero-Lag: Speaks sentence-by-sentence in <1s as words are generated"
              >
                <Zap className="w-3 h-3 text-amber-300" />
                <span>Instant Speech</span>
              </button>
              <button
                type="button"
                onClick={() => setAudioLatencyMode('studio')}
                className={`px-2.5 py-1 rounded-md transition-all flex items-center gap-1 ${
                  audioLatencyMode === 'studio'
                    ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold border border-[#00f0ff]/40 shadow-[0_0_8px_rgba(0,240,255,0.2)]'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="✨ Studio Neural: Pipelined server neural voice"
              >
                <Sparkles className="w-3 h-3 text-[#00f0ff]" />
                <span>Studio Voice</span>
              </button>
            </div>
          </div>
        </div>


        {/* Call History Panel (Drawer) */}
        {showHistory && (
          <div className="bg-[#051120] border border-[#00f0ff]/40 rounded-xl p-4 space-y-3 font-mono shadow-[0_0_25px_rgba(0,240,255,0.15)] animate-in fade-in-50 duration-200">
            <div className="flex items-center justify-between border-b border-[#00f0ff]/20 pb-2">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-[#00f0ff]" />
                <span className="text-xs font-bold uppercase tracking-wider text-[#00f0ff]">
                  Jarvis Voice Sessions & Call History
                </span>
                <span className="text-[10px] bg-[#00f0ff]/20 text-[#00f0ff] px-1.5 py-0.5 rounded-full">
                  {callSessions.length} saved
                </span>
              </div>
              <div className="flex items-center gap-2">
                {onNewSession && (
                  <button
                    onClick={() => {
                      onNewSession();
                      setShowHistory(false);
                    }}
                    className="px-2.5 py-1 bg-[#00f0ff] hover:bg-[#00f0ff]/80 text-slate-950 font-bold rounded text-[11px] flex items-center gap-1 transition-all"
                  >
                    <Plus className="w-3 h-3" /> Start New Call Session
                  </button>
                )}
                {onRefreshSessions && (
                  <button
                    onClick={onRefreshSessions}
                    className="p-1 text-slate-400 hover:text-[#00f0ff] transition-colors"
                    title="Refresh list"
                  >
                    <Radio className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={() => setShowHistory(false)}
                  className="p-1 text-slate-400 hover:text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Filter Search Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-500" />
              <input
                type="text"
                value={historySearch}
                onChange={(e) => setHistorySearch(e.target.value)}
                placeholder="Search past conversations and projects..."
                className="w-full bg-[#020b14] border border-[#00f0ff]/25 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#00f0ff]"
              />
            </div>

            {/* Session Cards List */}
            <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
              {isHistoryLoading ? (
                <div className="flex items-center justify-center py-6 text-xs text-[#00f0ff] gap-2">
                  <LoaderCircle className="w-4 h-4 animate-spin" /> Loading call history from memory...
                </div>
              ) : filteredSessions.length === 0 ? (
                <div className="text-center py-6 text-xs text-slate-500 font-sans">
                  {callSessions.length === 0
                    ? 'No past voice call sessions found yet. Start talking to begin recording persistent history!'
                    : 'No sessions match your search query.'}
                </div>
              ) : (
                filteredSessions.map((s: SessionInfo) => {
                  const isCurrent = s.id === activeSessionId;
                  const timeFormatted = s.last_active
                    ? new Date(s.last_active * 1000).toLocaleString([], {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })
                    : 'Recent';

                  return (
                    <div
                      key={s.id}
                      className={`p-2.5 rounded-lg border transition-all flex items-center justify-between gap-3 text-xs ${
                        isCurrent
                          ? 'bg-[#00f0ff]/15 border-[#00f0ff] shadow-[0_0_12px_rgba(0,240,255,0.25)]'
                          : 'bg-[#020b14]/80 border-[#00f0ff]/20 hover:border-[#00f0ff]/50 hover:bg-[#020b14]'
                      }`}
                    >
                      <div className="flex-1 min-w-0 space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-slate-200 truncate">
                            {s.title || `Session ${s.id.slice(0, 12)}`}
                          </span>
                          {isCurrent && (
                            <span className="px-1.5 py-0.2 bg-[#00f0ff]/30 text-[#00f0ff] rounded text-[10px] font-bold uppercase">
                              Active
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-slate-400">
                          <span className="flex items-center gap-1">
                            <Calendar className="w-3 h-3 text-[#80f7ff]/60" />
                            {timeFormatted}
                          </span>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <MessageSquare className="w-3 h-3 text-[#80f7ff]/60" />
                            {s.message_count} turns
                          </span>
                        </div>
                        {s.preview && (
                          <p className="text-[11px] text-slate-400/80 truncate font-sans">
                            {s.preview}
                          </p>
                        )}
                      </div>

                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        <button
                          onClick={() => {
                            onSelectSession?.(s.id);
                            setShowHistory(false);
                          }}
                          className={`px-2.5 py-1 rounded text-[11px] font-bold flex items-center gap-1 transition-all ${
                            isCurrent
                              ? 'bg-[#00f0ff] text-slate-950'
                              : 'bg-[#00f0ff]/20 text-[#00f0ff] hover:bg-[#00f0ff]/30'
                          }`}
                        >
                          <Play className="w-3 h-3 fill-current" />
                          <span>{isCurrent ? 'Current' : 'Resume'}</span>
                        </button>
                        {onDeleteSession && (
                          <button
                            onClick={() => {
                              if (window.confirm('Delete this conversation history from memory?')) {
                                onDeleteSession(s.id);
                              }
                            }}
                            className="p-1 text-slate-500 hover:text-rose-400 transition-colors rounded hover:bg-rose-950/40"
                            title="Delete session"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* Telemetry Strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
          <div className="bg-[#071526]/80 border border-[#00f0ff]/20 p-2.5 rounded-lg flex flex-col justify-center">
            <span className="text-[10px] text-[#80f7ff]/60 uppercase">Call Duration</span>
            <span className="text-base font-bold text-[#00f0ff]">{formatTime(durationSeconds)}</span>
          </div>

          <div className="bg-[#071526]/80 border border-[#00f0ff]/20 p-2.5 rounded-lg flex flex-col justify-center">
            <span className="text-[10px] text-[#80f7ff]/60 uppercase">Detected Speech</span>
            <span className="text-sm font-semibold text-amber-300">{detectedLanguage}</span>
          </div>

          <div className="bg-[#071526]/80 border border-[#00f0ff]/20 p-2.5 rounded-lg flex flex-col justify-center col-span-2 md:col-span-2">
            <div className="flex items-center justify-between text-[10px] text-[#80f7ff]/60 uppercase">
              <span>Sentinel Engine & Mic</span>
              <div className="flex items-center gap-1.5">
                {isWhisperFallback && (
                  <span className="px-1.5 py-0.5 rounded text-[8px] bg-amber-500/20 border border-amber-500/40 text-amber-300 font-mono">
                    Whisper Mode
                  </span>
                )}
                <span className="inline-flex items-center gap-1 text-[9px] text-emerald-300 font-mono">
                  <Sparkles className="w-2.5 h-2.5 text-emerald-400 animate-pulse" />
                  DSP Voice Filter: Active
                </span>
                <span className="inline-flex items-center gap-1 text-[9px] text-amber-300 font-mono">
                  <Zap className="w-2.5 h-2.5 text-amber-400 animate-pulse" />
                  {audioLatencyMode === 'instant' ? 'TTS: Instant (<1s)' : 'TTS: Pipelined'}
                </span>
              </div>
            </div>

            <span className="text-xs truncate text-[#80f7ff]">{micStatus} • {recognitionStatus}</span>
          </div>
        </div>

        {/* Barge-In Banner when Jarvis is Speaking */}
        {isSpeaking && (
          <div className="bg-gradient-to-r from-amber-950/70 via-amber-900/60 to-amber-950/70 border border-amber-500/60 rounded-xl p-3 flex items-center justify-between gap-3 shadow-[0_0_20px_rgba(245,158,11,0.25)] animate-in fade-in duration-150">
            <div className="flex items-center gap-2.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping" />
              <div>
                <p className="text-xs font-bold text-amber-200">
                  {selectedPersona === 'gwen' ? 'Gwen AI is speaking...' : 'Jarvis is speaking...'}
                </p>
                <p className="text-[11px] text-amber-300/70 font-mono">
                  Press <kbd className="px-1.5 py-0.5 bg-black/40 border border-amber-400/40 rounded text-amber-200 text-[10px]">Space</kbd> or click to interrupt immediately
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleInterruptJarvis}
              className="px-3.5 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold rounded-lg text-xs flex items-center gap-1.5 shadow-[0_0_10px_rgba(245,158,11,0.4)] transition-all cursor-pointer"
            >
              <Radio className="w-3.5 h-3.5 text-slate-950 animate-pulse" />
              Interrupt Jarvis
            </button>
          </div>
        )}

        {/* J.A.R.V.I.S. & ULTRON Neural Voice Orb Visualizer with Arc Reactor Tap-To-Wake */}
        <div
          onClick={handleKnockToWake}
          className="cursor-pointer group relative rounded-xl transition-all"
          title="Tap Arc Reactor / Knock to wake Jarvis (Iron Man style)"
        >
          <JarvisUltronVoiceOrb
            analyser={analyserRef.current}
            outputAnalyser={speechAnalyserRef.current}
            isActive={status === 'active' && !isAmbientStandby}
            isSpeaking={isSpeaking}
            isUserSpeaking={liveVolume > 10}
            isMuted={isMuted}
            selectedPersona={selectedPersona}
            themeMode={orbTheme}
            onThemeChange={setOrbTheme}
            sampleRate={audioCtxRef.current?.sampleRate || 48000}
          />
          {status === 'active' && isAmbientStandby && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <span className="px-3.5 py-1.5 bg-[#020b14]/80 border border-[#00f0ff]/50 rounded-full text-[11px] font-mono font-bold text-[#00f0ff] backdrop-blur-md shadow-[0_0_20px_rgba(0,240,255,0.4)] animate-pulse flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-300" />
                TAP OR SAY "WAKE UP JARVIS"
              </span>
            </div>
          )}
        </div>

        {/* Transcript Conversation Feed */}
        <div className="bg-[#071526]/50 border border-[#00f0ff]/20 rounded-xl p-4 h-64 overflow-y-auto space-y-3 font-mono text-xs shadow-inner">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${
                msg.sender === 'user'
                  ? 'items-end'
                  : msg.sender === 'assistant'
                  ? 'items-start'
                  : 'items-center text-center my-2'
              }`}
            >
              {msg.sender === 'system' ? (
                <span className="px-3 py-1 bg-slate-900/80 border border-slate-700/60 rounded-full text-[10px] text-slate-400">
                  {msg.text}
                </span>
              ) : (
                <div
                  className={`max-w-[80%] rounded-xl p-3 space-y-1 ${
                    msg.sender === 'user'
                      ? 'bg-[#00f0ff]/10 border border-[#00f0ff]/40 text-[#e5e2e1]'
                      : msg.persona === 'gwen'
                      ? 'bg-amber-950/40 border border-amber-500/40 text-amber-100'
                      : 'bg-cyan-950/40 border border-cyan-500/40 text-cyan-100'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3 text-[10px] opacity-70 font-semibold border-b border-white/10 pb-1">
                    <span className="flex items-center gap-1">
                      {msg.sender === 'user' ? (
                        'USER (YOU)'
                      ) : (
                        <>
                          <Bot className="w-3 h-3 text-[#00f0ff]" />
                          {msg.persona === 'gwen' ? 'GWEN AI' : 'JARVIS AI'}
                        </>
                      )}
                    </span>
                    <span>{msg.timestamp}</span>
                  </div>
                  <p className="text-xs leading-relaxed font-sans font-normal whitespace-pre-wrap">{msg.text}</p>
                </div>
              )}
            </div>
          ))}

          {interimTranscript && (
            <div className="flex flex-col items-end my-1">
              <div className="max-w-[90%] bg-gradient-to-br from-[#00f0ff]/15 via-[#071d33] to-[#020b14] border-2 border-[#00f0ff]/60 text-slate-100 rounded-xl p-3 text-xs shadow-[0_0_20px_rgba(0,240,255,0.25)] space-y-2">
                <div className="flex items-center justify-between gap-3 border-b border-[#00f0ff]/20 pb-1.5 text-[11px] font-mono">
                  <span className="flex items-center gap-1.5 text-[#00f0ff] font-bold">
                    <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-pulse" />
                    Speaking In Progress
                  </span>
                  {turnMode === 'hands_free' ? (
                    pauseCountdown !== null ? (
                      <span className="flex items-center gap-1 px-2 py-0.5 bg-amber-500/20 border border-amber-500/40 rounded-full text-amber-300 font-semibold text-[10px]">
                        <Timer className="w-3 h-3 text-amber-400 animate-spin" />
                        Sending in {pauseCountdown}s...
                      </span>
                    ) : (
                      <span className="text-[#80f7ff]/70 text-[10px]">Listening to your thoughts...</span>
                    )
                  ) : (
                    <span className="px-2 py-0.5 bg-[#00f0ff]/20 border border-[#00f0ff]/40 rounded-full text-[#00f0ff] font-semibold text-[10px]">
                      Push-to-Talk (No Timeout)
                    </span>
                  )}
                </div>

                <p className="text-xs text-white leading-relaxed font-sans font-normal italic">
                  "{interimTranscript}"
                </p>

                <div className="flex items-center justify-end gap-2 pt-1">
                  <button
                    type="button"
                    onClick={handleCancelDraft}
                    className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] transition-colors"
                  >
                    Cancel
                  </button>
                  {turnMode === 'hands_free' && (
                    <button
                      type="button"
                      onClick={() => handleExtendPause(2000)}
                      className="px-2.5 py-1 bg-[#00f0ff]/15 hover:bg-[#00f0ff]/25 border border-[#00f0ff]/40 text-[#00f0ff] rounded text-[11px] font-medium transition-all"
                      title="Grant 2 extra seconds of silence before sending"
                    >
                      +2s Still Talking
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleSendNow}
                    className="px-3 py-1 bg-[#00f0ff] hover:bg-[#00f0ff]/90 text-slate-950 font-bold rounded text-[11px] flex items-center gap-1 shadow-[0_0_10px_rgba(0,240,255,0.3)] transition-all"
                  >
                    <Send className="w-3 h-3" />
                    Send Now (Space)
                  </button>
                </div>
              </div>
            </div>
          )}

          {isProcessing && (
            <div className="flex items-center gap-2 text-xs text-[#00f0ff] animate-pulse">
              <LoaderCircle className="w-4 h-4 animate-spin" /> Processing speech turn...
            </div>
          )}

          <div ref={conversationEndRef} />
        </div>

        {/* Typed Input Fallback Form */}
        <form onSubmit={handleTypedSubmit} className="flex gap-2">
          <input
            type="text"
            value={typedInput}
            onChange={(e) => setTypedInput(e.target.value)}
            placeholder={status === 'active' ? 'Type message fallback or command...' : 'Start call to begin typing...'}
            disabled={status !== 'active'}
            className="flex-1 bg-[#071526] border border-[#00f0ff]/30 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-[#00f0ff] disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={status !== 'active' || !typedInput.trim()}
            className="px-4 py-2 bg-[#00f0ff]/20 border border-[#00f0ff] hover:bg-[#00f0ff]/30 text-[#00f0ff] rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 disabled:opacity-40"
          >
            <Send className="w-3.5 h-3.5" /> Send
          </button>
        </form>

        {/* Main Action Controls */}
        <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-[#00f0ff]/20">
          <div className="flex items-center gap-2">
            {status === 'idle' ? (
              <button
                onClick={() => startCall(false)}
                className="px-5 py-2.5 bg-[#00f0ff] hover:bg-[#00f0ff]/90 text-slate-950 font-bold rounded-lg text-xs tracking-wider uppercase flex items-center gap-2 shadow-[0_0_15px_rgba(0,240,255,0.4)] transition-all"
              >
                <Phone className="w-4 h-4" /> Start Call
              </button>
            ) : (
              <button
                onClick={endCall}
                className="px-5 py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-lg text-xs tracking-wider uppercase flex items-center gap-2 shadow-[0_0_15px_rgba(225,29,72,0.4)] transition-all"
              >
                <PhoneOff className="w-4 h-4" /> End Call
              </button>
            )}

            {/* Tony Stark Knock / Wake Up Button */}
            <button
              type="button"
              onClick={handleKnockToWake}
              className="px-4 py-2.5 bg-gradient-to-r from-amber-500/20 via-amber-400/25 to-amber-500/20 hover:from-amber-500/30 hover:to-amber-400/35 border border-amber-400/60 text-amber-200 font-bold rounded-lg text-xs tracking-wider uppercase flex items-center gap-1.5 shadow-[0_0_15px_rgba(245,158,11,0.25)] hover:shadow-[0_0_20px_rgba(245,158,11,0.4)] transition-all cursor-pointer"
              title="Tony Stark style: Tap or knock to wake Jarvis instantly"
            >
              <Zap className="w-4 h-4 text-amber-300 animate-pulse" />
              <span>Knock / Wake Up</span>
            </button>

            <button
              onClick={toggleMicMute}
              disabled={status !== 'active'}
              className={`px-4 py-2.5 border rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all disabled:opacity-40 ${
                isMuted
                  ? 'bg-rose-950/60 border-rose-500 text-rose-300'
                  : 'bg-[#071526] border-[#00f0ff]/30 text-slate-200 hover:border-[#00f0ff]'
              }`}
            >
              {isMuted ? <MicOff className="w-4 h-4 text-rose-400" /> : <Mic className="w-4 h-4 text-[#00f0ff]" />}
              {isMuted ? 'Muted' : 'Mic Active'}
            </button>

            <button
              onClick={() => setSpeakerOn(!speakerOn)}
              className={`px-4 py-2.5 border rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                !speakerOn
                  ? 'bg-amber-950/60 border-amber-500 text-amber-300'
                  : 'bg-[#071526] border-[#00f0ff]/30 text-slate-200 hover:border-[#00f0ff]'
              }`}
            >
              {!speakerOn ? <VolumeX className="w-4 h-4 text-amber-400" /> : <Volume2 className="w-4 h-4 text-[#00f0ff]" />}
              {!speakerOn ? 'Speaker Off' : 'Speaker On'}
            </button>

            {isSpeaking && (
              <button
                onClick={handleInterruptJarvis}
                className="px-4 py-2.5 bg-amber-500/20 border border-amber-500 text-amber-300 rounded-lg text-xs font-bold flex items-center gap-1.5 animate-pulse shadow-[0_0_12px_rgba(245,158,11,0.3)] hover:bg-amber-500/30 transition-all"
              >
                <Radio className="w-4 h-4 text-amber-400" /> Interrupt Speaking (Space)
              </button>
            )}
          </div>

          <div className="text-[11px] font-mono text-[#80f7ff]/60 flex items-center gap-2">
            <Radio className="w-3.5 h-3.5 text-[#00f0ff] animate-pulse" /> Full Duplex Sentinel
          </div>
        </div>
      </div>
    </div>
  );
};
