import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Bot,
  Send,
  Volume2,
  VolumeX,
  Sparkles,
  Copy,
  Check,
  Timer,
  Play,
  Pause,
  RotateCcw,
  Zap,
  Network,
  MessageSquare,
  Columns,
  Target,
  CheckCircle2,
  FolderGit2,
  Cloud,
  Terminal,
  Code2,
  ExternalLink,
  Search,
  Database,
  ChevronRight,
  Layers,
  Activity,
} from 'lucide-react';
import type { JarvisMessage, BiometricTelemetry } from '@/types/jarvis';
import {
  formatDisplayContentWithPunctuation,
  speakWithNabra,
  stopNabraAudio,
} from '@/utils/jarvisSpeechUtils';
import {
  cleanSpokenText,
  extractNextSpokenSentence,
  PipelinedAudioQueue,
  sanitizeTextForSpeech,
} from '@/lib/speechUtils';
import { copyTextToClipboard } from '@/lib/clipboard';
import { JarvisNetworkGraph, type GraphNode } from '@/components/JarvisNetworkGraph';
import {
  IBRAHIM_PROFILE,
  IBRAHIM_GITHUB_PROJECTS,
  IBRAHIM_HERMES_TASKS,
  IBRAHIM_GCP_SERVICES,
} from '@/data/ibrahimProfileData';

interface JarvisCoreWidgetProps {
  biometrics?: BiometricTelemetry;
  onSendMessage?: (text: string, onDelta?: (partial: string) => void) => Promise<string>;
}

const DEFAULT_BIOMETRICS: BiometricTelemetry = {
  heartRate: 72,
  hrv: 64,
  energyLevel: 92,
  stressIndex: 14,
  focusScore: 96,
  sleepQuality: 88,
  circadianPhase: 'Peak Focus',
  cameraPulseActive: true,
  lastSyncTimestamp: 'Just now',
};

export const JarvisCoreWidget: React.FC<JarvisCoreWidgetProps> = ({
  biometrics = DEFAULT_BIOMETRICS,
  onSendMessage,
}) => {
  const [messages, setMessages] = useState<JarvisMessage[]>([
    {
      id: 'm-1',
      sender: 'jarvis',
      content:
        'مرحباً بك يا باشمهندس إبراهيم! أنا J.A.R.V.I.S. (Chief of Staff & AI Systems Architect). تم فحص ومزامنة كامل ملفك الهندسي: 41 مشروع على GitHub (بما فيها hermes-agent و MR-NLP-Robust-RAG-Chatbot و dual-site-clerk-auth)، بالإضافة إلى ربط خدمات Google Cloud (BigQuery, Cloud Run, GCS, Dataproc) ومهام Hermes Agent المستقلة. جميع الأنظمة تحت السيطرة وفي أعلى مستويات الأداء. كيف يمكنني دعمك في كتابة الكود أو التحليل السحابي اليوم؟',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      technicalKeywords: [
        'Eng. Ibrahim Abdelsattar',
        'hermes-agent',
        'MR-NLP RAG',
        'dual-site-clerk-auth',
        'Google Cloud',
        'BigQuery',
        'Cloud Run',
        'GCS',
        'Dataproc',
      ],
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'chat' | 'projects' | 'cloud' | 'graph' | 'split'>('chat');

  // Projects Explorer State
  const [selectedProjectId, setSelectedProjectId] = useState<string>(IBRAHIM_GITHUB_PROJECTS[0].id);
  const [selectedFileIndex, setSelectedFileIndex] = useState<number>(0);
  const [projectSearchQuery, setProjectSearchQuery] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');

  // Pomodoro Focus Timer & Deep Work Mode State
  const [isPomodoroActive, setIsPomodoroActive] = useState<boolean>(false);
  const [pomodoroMode, setPomodoroMode] = useState<'work' | 'break'>('work');
  const [timeLeft, setTimeLeft] = useState<number>(25 * 60);
  const [completedSessions, setCompletedSessions] = useState<number>(0);
  const [isDeepWorkMode, setIsDeepWorkMode] = useState<boolean>(false);

  // Daily Focus Goal Tracker State
  const [dailyFocusGoal, setDailyFocusGoal] = useState<string>(
    'Deploy MR-NLP RAG & Hermes Agent to Google Cloud Run with BigQuery Vector Index'
  );
  const focusTargetMinutes = 60;
  const [focusElapsedSeconds, setFocusElapsedSeconds] = useState<number>(2100);
  const [isFocusGoalActive, setIsFocusGoalActive] = useState<boolean>(false);
  const [isFocusCompleted, setIsFocusCompleted] = useState<boolean>(false);
  const [isEditingFocusGoal, setIsEditingFocusGoal] = useState<boolean>(false);
  const [customGoalInput, setCustomGoalInput] = useState<string>(dailyFocusGoal);

  // Daily Focus Timer Effect
  useEffect(() => {
    let timerId: ReturnType<typeof setInterval> | null = null;
    if (isFocusGoalActive && !isFocusCompleted) {
      timerId = setInterval(() => {
        setFocusElapsedSeconds((prev) => {
          const next = prev + 1;
          if (next >= focusTargetMinutes * 60) {
            setIsFocusCompleted(true);
            setIsFocusGoalActive(false);
            if (isSpeechEnabled) {
              void speakWithNabra('عاش يا باشمهندس إبراهيم! لقد حققت 100% من هدف التركيز الهندسي بنجاح.');
            }
          }
          return next;
        });
      }, 1000);
    }
    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, [isFocusGoalActive, isFocusCompleted, focusTargetMinutes, isSpeechEnabled]);

  const toggleFocusGoalTimer = () => {
    if (isFocusCompleted) return;
    setIsFocusGoalActive((prev) => !prev);
  };

  const handleMarkGoalCompleted = () => {
    const nextCompleted = !isFocusCompleted;
    setIsFocusCompleted(nextCompleted);
    if (nextCompleted) {
      setIsFocusGoalActive(false);
      setFocusElapsedSeconds(focusTargetMinutes * 60);
      if (isSpeechEnabled) {
        void speakWithNabra('ألف مبروك يا باشمهندس إبراهيم! تم تأكيد إنجاز الهدف اليومي بنجاح!');
      }
    } else {
      setFocusElapsedSeconds(0);
    }
  };

  const resetFocusGoalTimer = () => {
    setIsFocusGoalActive(false);
    setIsFocusCompleted(false);
    setFocusElapsedSeconds(0);
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (typeof messagesEndRef.current?.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const activeAudioQueueRef = useRef<PipelinedAudioQueue | null>(null);

  useEffect(() => {
    if (!isSpeechEnabled) {
      activeAudioQueueRef.current?.abort();
      stopNabraAudio();
    }
  }, [isSpeechEnabled]);

  useEffect(() => {
    return () => {
      activeAudioQueueRef.current?.abort();
      stopNabraAudio();
    };
  }, []);

  const speakText = async (text: string) => {
    if (!isSpeechEnabled) return;
    activeAudioQueueRef.current?.abort();
    stopNabraAudio();
    await speakWithNabra(text, 'jarvis');
  };

  // Pomodoro Timer Countdown
  useEffect(() => {
    let timerId: ReturnType<typeof setInterval> | null = null;
    if (isPomodoroActive && timeLeft > 0) {
      timerId = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
      }, 1000);
    } else if (isPomodoroActive && timeLeft === 0) {
      if (pomodoroMode === 'work') {
        setCompletedSessions((c) => c + 1);
        setPomodoroMode('break');
        setTimeLeft(5 * 60);
        void speakText(
          'عاش يا باشمهندس إبراهيم! انتهت جلسة التركيز العميق (25 دقيقة). استرح قليلاً لإعادة شحن الطاقة.'
        );
      } else {
        setPomodoroMode('work');
        setTimeLeft(25 * 60);
        setIsPomodoroActive(false);
        void speakText('يا باشمهندس إبراهيم، انتهت الاستراحة، مستعدون لجلسة الهندسة القادمة؟');
      }
    }

    return () => {
      if (timerId) clearInterval(timerId);
    };
  }, [isPomodoroActive, timeLeft, pomodoroMode]);

  const togglePomodoro = () => {
    if (!isPomodoroActive) {
      setIsPomodoroActive(true);
      if (pomodoroMode === 'work') {
        void speakText('بدأت جلسة العمل العميق. 25 دقيقة من التركيز الهندسي تبدأ الآن.');
      } else {
        void speakText('بدأت فترة الاستراحة.');
      }
    } else {
      setIsPomodoroActive(false);
      void speakText('تم إيقاف مؤقت التركيز مؤقتاً.');
    }
  };

  const resetPomodoro = () => {
    setIsPomodoroActive(false);
    setPomodoroMode('work');
    setTimeLeft(25 * 60);
    void speakText('تمت إعادة تعيين مؤقت التركيز إلى 25 دقيقة.');
  };

  const toggleDeepWorkMode = () => {
    const nextState = !isDeepWorkMode;
    setIsDeepWorkMode(nextState);
    if (nextState) {
      if (!isPomodoroActive) {
        setIsPomodoroActive(true);
      }
      void speakText('تم تفعيل درع التركيز العميق! جميع إشعارات وموارد النظام في وضع الأداء الأقصى.');
    } else {
      void speakText('تم إلغاء وضع التركيز العميق.');
    }
  };

  const formatTimer = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const extractTechnicalKeywords = (content: string): string[] => {
    const pool = [
      'Ibrahim Abdelsattar',
      'hermes-agent',
      'MR-NLP RAG',
      'dual-site-clerk-auth',
      'Arabic Sentiment',
      'Fraud Detection',
      'BigQuery',
      'Cloud Run',
      'Google Cloud',
      'GCS',
      'Dataproc',
      'Cloud SQL',
      'Prompt Caching',
      'Voice Sentinel',
      'FastAPI',
      'PyTorch',
    ];
    return pool.filter((kw) => content.toLowerCase().includes(kw.toLowerCase()));
  };

  const handleSendMessage = async (customPrompt?: string) => {
    const textToSend = customPrompt || inputValue;
    if (!textToSend.trim() || isLoading) return;

    const userMsg: JarvisMessage = {
      id: `u-${Date.now()}`,
      sender: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!customPrompt) setInputValue('');
    setIsLoading(true);

    try {
      let replyContent = '';

      if (onSendMessage) {
        // Send via Hermes Gateway (same mechanism as hermes chat: raw text +
        // surface voice-live; backend injects the concise voice note). Stream
        // deltas into a live bubble so the first token paints immediately.
        const streamingId = `j-stream-${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          {
            id: streamingId,
            sender: 'jarvis',
            content: '',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);

        let spokenCharIndex = 0;
        const audioQueue = isSpeechEnabled ? new PipelinedAudioQueue() : null;
        if (audioQueue) {
          activeAudioQueueRef.current?.abort();
          activeAudioQueueRef.current = audioQueue;
        }

        replyContent = await onSendMessage(textToSend, (partial) => {
          setMessages((prev) => prev.map((m) => (m.id === streamingId ? { ...m, content: partial } : m)));

          if (audioQueue && isSpeechEnabled) {
            const clean = cleanSpokenText(partial);
            while (true) {
              const res = extractNextSpokenSentence(clean, spokenCharIndex);
              if (!res) break;
              spokenCharIndex = res.nextIndex;
              const sentence = sanitizeTextForSpeech(res.sentence);
              if (sentence.length >= 2) {
                audioQueue.enqueue(async (signal) => {
                  if (signal.aborted) return;
                  await speakWithNabra(sentence, 'jarvis', false);
                });
              }
            }
          }
        });
        // Replace the streaming placeholder with the authoritative reply.
        setMessages((prev) => prev.filter((m) => m.id !== streamingId));

        // Enqueue remaining un-spoken text
        if (audioQueue && isSpeechEnabled) {
          const finalClean = cleanSpokenText(replyContent);
          const remainder = sanitizeTextForSpeech(finalClean.slice(spokenCharIndex));
          if (remainder.length >= 2) {
            audioQueue.enqueue(async (signal) => {
              if (signal.aborted) return;
              await speakWithNabra(remainder, 'jarvis', false);
            });
          }
        }
      } else {
        // Fallback realistic response grounded in Ibrahim's stack
        await new Promise((r) => setTimeout(r, 900));
        replyContent = `يا باشمهندس إبراهيم، تلقيت أمرك: "${textToSend}". تم فحص مشاريعك على GitHub وخوادم Google Cloud المرتبطة (${biometrics.energyLevel}% طاقة، ${biometrics.focusScore}% تركيز). يتم تنفيذ الخطوات المطلوبة ومزامنة كود المشروع وتحديث السجلات بنجاح.`;
      }

      const jarvisReply: JarvisMessage = {
        id: `j-${Date.now()}`,
        sender: 'jarvis',
        content: replyContent,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        technicalKeywords: extractTechnicalKeywords(replyContent),
      };

      setMessages((prev) => [...prev, jarvisReply]);
      // Gateway replies are spoken from their streaming queue; the local
      // fallback has no stream, so speak it after completion.
      if (isSpeechEnabled && !onSendMessage) {
        void speakText(replyContent);
      }
    } catch (err: unknown) {
      console.warn('Chat error:', err);
      const message = err instanceof Error ? err.message : 'Unknown error';
      const errorReply: JarvisMessage = {
        id: `j-${Date.now()}`,
        sender: 'jarvis',
        content: `يا باشمهندس، حدث خطأ أثناء الاتصال بالخادم: ${message}. يرجى التحقق من اتصال بوابة Hermes.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorReply]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = (id: string, text: string) => {
    void copyTextToClipboard(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleGraphNodeSelect = (node: GraphNode) => {
    void handleSendMessage(`Give me a diagnostic status and telemetry update for ${node.label} (${node.category}).`);
  };

  const focusPercent = Math.min(100, Math.round((focusElapsedSeconds / (focusTargetMinutes * 60)) * 100));

  // Filtered GitHub projects
  const filteredProjects = useMemo(() => {
    return IBRAHIM_GITHUB_PROJECTS.filter((p) => {
      const matchCat = selectedCategory === 'ALL' || p.category === selectedCategory;
      const matchQuery =
        p.name.toLowerCase().includes(projectSearchQuery.toLowerCase()) ||
        p.description.toLowerCase().includes(projectSearchQuery.toLowerCase()) ||
        p.language.toLowerCase().includes(projectSearchQuery.toLowerCase());
      return matchCat && matchQuery;
    });
  }, [projectSearchQuery, selectedCategory]);

  const activeProject = useMemo(() => {
    return (
      IBRAHIM_GITHUB_PROJECTS.find((p) => p.id === selectedProjectId) ||
      IBRAHIM_GITHUB_PROJECTS[0]
    );
  }, [selectedProjectId]);

  const activeFile = activeProject.highlightFiles[selectedFileIndex] || activeProject.highlightFiles[0];

  return (
    <div
      className={`w-full h-full flex flex-col rounded-xl bg-[#040d1a]/95 border transition-all shadow-[0_0_25px_rgba(0,240,255,0.08)] ${
        isDeepWorkMode ? 'border-amber-400/80 shadow-[0_0_35px_rgba(255,183,0,0.25)]' : 'border-[#00f0ff]/30'
      }`}
    >
      {/* Telemetry Header Bar */}
      <div className="px-4 py-3 border-b border-[#00f0ff]/20 bg-[#071526]/80 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center gap-3">
          {/* Avatar & Bot Indicator */}
          <div className="relative">
            <img
              src={IBRAHIM_PROFILE.avatarUrl}
              alt="Eng. Ibrahim Abdelsattar"
              className="size-10 rounded-lg border-2 border-[#00f0ff]/60 object-cover shadow-[0_0_10px_rgba(0,240,255,0.4)]"
            />
            <div className="absolute -bottom-1 -right-1 size-4 rounded-full bg-[#00f0ff] border border-[#040d1a] flex items-center justify-center text-[#040d1a]">
              <Bot className="size-2.5" />
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-bold text-[#00f0ff] tracking-wide text-sm">JARVIS CHIEF OF STAFF</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-amber-400/10 text-amber-300 border border-amber-400/30">
                MARK 85
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-[#00f0ff]/10 text-[#80f7ff] border border-[#00f0ff]/30">
                Eng. {IBRAHIM_PROFILE.name}
              </span>
            </div>
            <div className="flex items-center gap-2 mt-0.5 text-[11px] text-[#80f7ff]/70 flex-wrap">
              <span>{IBRAHIM_PROFILE.title}</span>
              <span>•</span>
              <a
                href={IBRAHIM_PROFILE.githubUrl}
                target="_blank"
                rel="noreferrer"
                className="text-[#ffb700] hover:underline flex items-center gap-1"
              >
                <FolderGit2 className="size-3" />
                <span>@{IBRAHIM_PROFILE.githubUsername} (41 Repos)</span>
              </a>
              <span>•</span>
              <span className="text-emerald-400 flex items-center gap-1">
                <Cloud className="size-3" />
                <span>Google Cloud Linked</span>
              </span>
            </div>
          </div>
        </div>

        {/* View Switcher & Audio Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center bg-[#020b14] p-1 rounded-lg border border-[#00f0ff]/20">
            <button
              onClick={() => setViewMode('chat')}
              className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-all ${
                viewMode === 'chat'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                  : 'text-[#80f7ff]/60 hover:text-[#00f0ff]'
              }`}
            >
              <MessageSquare className="size-3.5" /> Chat
            </button>
            <button
              onClick={() => setViewMode('projects')}
              className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-all ${
                viewMode === 'projects'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                  : 'text-[#80f7ff]/60 hover:text-[#00f0ff]'
              }`}
            >
              <FolderGit2 className="size-3.5" /> Projects & Code
            </button>
            <button
              onClick={() => setViewMode('cloud')}
              className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-all ${
                viewMode === 'cloud'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                  : 'text-[#80f7ff]/60 hover:text-[#00f0ff]'
              }`}
            >
              <Cloud className="size-3.5" /> Cloud & Tasks
            </button>
            <button
              onClick={() => setViewMode('graph')}
              className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-all ${
                viewMode === 'graph'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                  : 'text-[#80f7ff]/60 hover:text-[#00f0ff]'
              }`}
            >
              <Network className="size-3.5" /> Topology
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-all ${
                viewMode === 'split'
                  ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold shadow-[0_0_10px_rgba(0,240,255,0.3)]'
                  : 'text-[#80f7ff]/60 hover:text-[#00f0ff]'
              }`}
            >
              <Columns className="size-3.5" /> Split
            </button>
          </div>

          <button
            onClick={() => setIsSpeechEnabled(!isSpeechEnabled)}
            className={`p-2 rounded-lg border transition-all ${
              isSpeechEnabled
                ? 'bg-[#00f0ff]/10 border-[#00f0ff]/40 text-[#00f0ff]'
                : 'bg-slate-800/40 border-slate-700 text-slate-400'
            }`}
            title={isSpeechEnabled ? 'Mute Speech Output' : 'Enable Speech Output'}
          >
            {isSpeechEnabled ? <Volume2 className="size-4" /> : <VolumeX className="size-4" />}
          </button>
        </div>
      </div>

      {/* Pomodoro Focus Timer & Deep Work Bar */}
      <div
        className={`px-4 py-2 border-b transition-all flex flex-wrap items-center justify-between gap-2.5 text-xs font-mono ${
          isDeepWorkMode
            ? 'bg-amber-950/30 border-amber-400/40 text-amber-200'
            : 'bg-[#031120]/70 border-[#00f0ff]/15 text-[#80f7ff]'
        }`}
      >
        {/* Daily Focus Goal Bar */}
        <div className="flex items-center gap-2 flex-1 min-w-[260px]">
          <Target className="size-4 text-[#ffb700] shrink-0" />
          <span className="text-[11px] text-[#ffb700] font-bold uppercase shrink-0">FOCUS GOAL:</span>
          {isEditingFocusGoal ? (
            <div className="flex items-center gap-1 flex-1">
              <input
                type="text"
                value={customGoalInput}
                onChange={(e) => setCustomGoalInput(e.target.value)}
                className="bg-[#040e1b] border border-[#00f0ff]/40 text-xs px-2 py-0.5 rounded text-cyan-200 flex-1 outline-none"
              />
              <button
                onClick={() => {
                  setDailyFocusGoal(customGoalInput);
                  setIsEditingFocusGoal(false);
                }}
                className="px-2 py-0.5 bg-[#00f0ff]/20 text-[#00f0ff] rounded text-[10px]"
              >
                Save
              </button>
            </div>
          ) : (
            <span
              onClick={() => setIsEditingFocusGoal(true)}
              className="text-[11px] text-cyan-100 truncate cursor-pointer hover:underline"
              title="Click to edit goal"
            >
              {dailyFocusGoal}
            </span>
          )}
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#00f0ff]/10 text-[#00f0ff] shrink-0">
            {focusPercent}% ({Math.floor(focusElapsedSeconds / 60)}/{focusTargetMinutes}m)
          </span>
          <button
            onClick={toggleFocusGoalTimer}
            className="p-1 text-cyan-300 hover:text-cyan-100"
            title={isFocusGoalActive ? 'Pause Goal Timer' : 'Start Goal Timer'}
          >
            {isFocusGoalActive ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </button>
          <button
            onClick={resetFocusGoalTimer}
            className="p-1 text-slate-400 hover:text-cyan-300"
            title="Reset Goal Timer"
          >
            <RotateCcw className="size-3" />
          </button>
          <button
            onClick={handleMarkGoalCompleted}
            className={`p-1 transition-colors ${
              isFocusCompleted ? 'text-emerald-400' : 'text-slate-400 hover:text-emerald-300'
            }`}
            title="Mark Goal as Completed"
          >
            <CheckCircle2 className="size-3.5" />
          </button>
        </div>

        {/* 25-Min Pomodoro Sprint */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#07172b] border border-[#00f0ff]/30">
            <Timer className="size-3.5 text-amber-400" />
            <span className="font-bold tabular-nums text-amber-300">{formatTimer(timeLeft)}</span>
            <span className="text-[10px] text-cyan-400 uppercase">({pomodoroMode})</span>
            {completedSessions > 0 && (
              <span className="text-[9px] px-1 rounded bg-amber-400/20 text-amber-300">
                #{completedSessions}
              </span>
            )}
          </div>

          <button
            onClick={togglePomodoro}
            className="p-1.5 rounded bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/30 transition-all"
            title={isPomodoroActive ? 'Pause Pomodoro' : 'Start Pomodoro'}
          >
            {isPomodoroActive ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </button>

          <button
            onClick={resetPomodoro}
            className="p-1.5 rounded bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/30 transition-all"
            title="Reset Pomodoro"
          >
            <RotateCcw className="size-3.5" />
          </button>

          <button
            onClick={toggleDeepWorkMode}
            className={`px-2.5 py-1 rounded text-xs flex items-center gap-1 font-bold transition-all ${
              isDeepWorkMode
                ? 'bg-amber-400 text-slate-950 shadow-[0_0_15px_rgba(255,183,0,0.5)]'
                : 'bg-amber-400/10 text-amber-300 border border-amber-400/40 hover:bg-amber-400/20'
            }`}
          >
            <Zap className="size-3" /> DEEP WORK
          </button>
        </div>
      </div>

      {/* Quick Command Prompt Chips */}
      <div className="px-4 py-2 bg-[#051120]/60 border-b border-[#00f0ff]/10 flex items-center gap-2 overflow-x-auto scrollbar-none text-[11px] font-mono">
        <span className="text-[#ffb700] shrink-0 flex items-center gap-1 font-bold">
          <Sparkles className="size-3" /> COMMANDS:
        </span>
        <button
          onClick={() =>
            handleSendMessage(
              'Inspect MR-NLP Robust RAG Chatbot architecture, Whisper ASR ingestion pipeline, and Qwen 1.5 4-bit quantization.'
            )
          }
          className="px-2.5 py-1 bg-[#07172b] border border-[#00f0ff]/30 hover:border-[#00f0ff] text-[#c8c6c5] hover:text-[#00f0ff] rounded whitespace-nowrap transition-colors"
        >
          🔬 MR-NLP RAG Pipeline
        </button>
        <button
          onClick={() =>
            handleSendMessage(
              'Review Hermes Agent turn loop prompt-caching stability, tools registry discovery, and JSON-RPC gateway performance.'
            )
          }
          className="px-2.5 py-1 bg-[#07172b] border border-[#00f0ff]/30 hover:border-[#00f0ff] text-[#c8c6c5] hover:text-[#00f0ff] rounded whitespace-nowrap transition-colors"
        >
          ⚡ Hermes Core Caching
        </button>
        <button
          onClick={() =>
            handleSendMessage(
              'Analyze Google Cloud Platform integration: BigQuery vector embeddings, Cloud Run autoscaling, and GCS model storage.'
            )
          }
          className="px-2.5 py-1 bg-[#07172b] border border-[#00f0ff]/30 hover:border-[#00f0ff] text-[#c8c6c5] hover:text-[#00f0ff] rounded whitespace-nowrap transition-colors"
        >
          ☁️ Google Cloud & BigQuery
        </button>
        <button
          onClick={() =>
            handleSendMessage(
              'Review dual-site-clerk-auth cross-domain session routing and Next.js edge middleware security rules.'
            )
          }
          className="px-2.5 py-1 bg-[#07172b] border border-[#00f0ff]/30 hover:border-[#00f0ff] text-[#c8c6c5] hover:text-[#00f0ff] rounded whitespace-nowrap transition-colors"
        >
          🔐 Clerk Dual-Site Auth
        </button>
        <button
          onClick={() =>
            handleSendMessage(
              'Assess Arabic-Sentiment-Analysis text normalization, Tashkeel removal, and transformer dialect embeddings.'
            )
          }
          className="px-2.5 py-1 bg-[#07172b] border border-[#00f0ff]/30 hover:border-[#00f0ff] text-[#c8c6c5] hover:text-[#00f0ff] rounded whitespace-nowrap transition-colors"
        >
          🗣️ Arabic Dialect NLP
        </button>
      </div>

      {/* Main Content Area: Chat or Projects or Cloud or Graph or Split */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {viewMode === 'projects' ? (
          /* Projects Explorer & Real Code Viewer */
          <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden p-3 gap-3">
            {/* Left: Repositories List */}
            <div className="w-full lg:w-80 flex flex-col bg-[#020914]/90 rounded-lg border border-[#00f0ff]/20 overflow-hidden">
              <div className="p-2.5 border-b border-[#00f0ff]/15 bg-[#07172b]/60 flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[#00f0ff] font-bold flex items-center gap-1.5">
                    <FolderGit2 className="size-3.5 text-[#ffb700]" />
                    <span>GITHUB PROJECTS (41)</span>
                  </span>
                  <span className="text-[10px] text-[#80f7ff]/60">Eng. Ibrahim</span>
                </div>
                <div className="relative">
                  <Search className="size-3.5 text-[#80f7ff]/50 absolute left-2.5 top-2" />
                  <input
                    type="text"
                    value={projectSearchQuery}
                    onChange={(e) => setProjectSearchQuery(e.target.value)}
                    placeholder="Search projects & code..."
                    className="w-full bg-[#040e1b] border border-[#00f0ff]/30 text-xs text-[#00f0ff] pl-8 pr-2.5 py-1 rounded outline-none placeholder-[#80f7ff]/40 font-mono"
                  />
                </div>
                {/* Category Filter Chips */}
                <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none text-[10px]">
                  {['ALL', 'Autonomous Agents', 'RAG & NLP', 'Full Stack & Auth', 'Machine Learning'].map((cat) => (
                    <button
                      key={cat}
                      onClick={() => setSelectedCategory(cat)}
                      className={`px-2 py-0.5 rounded whitespace-nowrap transition-colors ${
                        selectedCategory === cat
                          ? 'bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/40 font-bold'
                          : 'bg-[#041224] text-slate-400 border border-transparent hover:text-cyan-200'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>

              {/* Projects Scroll Area */}
              <div className="flex-1 overflow-y-auto p-2 space-y-1.5 font-mono text-xs">
                {filteredProjects.map((p) => {
                  const isSelected = p.id === selectedProjectId;
                  return (
                    <div
                      key={p.id}
                      onClick={() => {
                        setSelectedProjectId(p.id);
                        setSelectedFileIndex(0);
                      }}
                      className={`p-2.5 rounded-lg cursor-pointer border transition-all ${
                        isSelected
                          ? 'bg-[#00f0ff]/15 border-[#00f0ff] text-cyan-100 shadow-[0_0_12px_rgba(0,240,255,0.2)]'
                          : 'bg-[#041224]/60 border-[#00f0ff]/15 hover:border-[#00f0ff]/40 text-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1 mb-1">
                        <span className="font-bold text-xs truncate text-[#00f0ff]">{p.name}</span>
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-400/10 text-amber-300 border border-amber-400/20 shrink-0">
                          {p.language}
                        </span>
                      </div>
                      <p className="text-[10px] text-[#80f7ff]/70 line-clamp-2 leading-relaxed">
                        {p.description}
                      </p>
                      <div className="flex items-center justify-between mt-2 pt-1.5 border-t border-[#00f0ff]/10 text-[10px] text-[#80f7ff]/50">
                        <span>⭐ {p.stars} stars</span>
                        <span>🍴 {p.forks} forks</span>
                        <span className="text-[#ffb700]">{p.category}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right: Code Inspector & Project Architecture */}
            <div className="flex-1 flex flex-col bg-[#020914]/90 rounded-lg border border-[#00f0ff]/20 overflow-hidden">
              {/* Code Inspector Header */}
              <div className="p-3 border-b border-[#00f0ff]/20 bg-[#07172b]/80 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-[#00f0ff] text-sm font-mono">{activeProject.fullName}</span>
                    <a
                      href={activeProject.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs px-2 py-0.5 rounded bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 text-[#80f7ff] border border-[#00f0ff]/30 flex items-center gap-1 font-mono transition-colors"
                    >
                      <span>View on GitHub</span>
                      <ExternalLink className="size-3" />
                    </a>
                  </div>
                  <p className="text-xs text-slate-300 mt-1 max-w-2xl">{activeProject.description}</p>
                </div>

                {/* File Tabs */}
                <div className="flex items-center gap-1 bg-[#040e1b] p-1 rounded-lg border border-[#00f0ff]/30 overflow-x-auto max-w-full">
                  {activeProject.highlightFiles.map((f, idx) => (
                    <button
                      key={f.filename}
                      onClick={() => setSelectedFileIndex(idx)}
                      className={`px-2.5 py-1 rounded text-xs font-mono flex items-center gap-1.5 transition-all whitespace-nowrap ${
                        selectedFileIndex === idx
                          ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold border border-[#00f0ff]/40 shadow-[0_0_8px_rgba(0,240,255,0.3)]'
                          : 'text-slate-400 hover:text-cyan-200'
                      }`}
                    >
                      <Code2 className="size-3" />
                      <span>{f.filename}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Code Display Area */}
              <div className="flex-1 overflow-y-auto p-3 font-mono text-xs bg-[#01050c]">
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#00f0ff]/15 text-[11px] text-[#80f7ff]/60">
                  <span className="flex items-center gap-1.5">
                    <Terminal className="size-3.5 text-[#ffb700]" />
                    <span>Source Code Inspector • {activeFile.filename}</span>
                  </span>
                  <button
                    onClick={() => handleCopy(`code-${activeFile.filename}`, activeFile.code)}
                    className="px-2.5 py-1 rounded bg-[#07172b] hover:bg-[#00f0ff]/20 border border-[#00f0ff]/30 text-[#80f7ff] hover:text-[#00f0ff] flex items-center gap-1 transition-colors"
                  >
                    {copiedId === `code-${activeFile.filename}` ? (
                      <>
                        <Check className="size-3 text-emerald-400" />
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="size-3" />
                        <span>Copy Code</span>
                      </>
                    )}
                  </button>
                </div>

                <pre className="text-cyan-100/90 leading-relaxed overflow-x-auto whitespace-pre font-mono p-2 bg-[#030914] rounded border border-[#00f0ff]/10">
                  <code>{activeFile.code}</code>
                </pre>

                {/* Project File Tree List */}
                <div className="mt-4 pt-3 border-t border-[#00f0ff]/20">
                  <span className="text-[11px] font-bold text-[#ffb700] uppercase tracking-wide flex items-center gap-1 mb-2">
                    <Layers className="size-3.5" /> Project Repository Structure
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                    {activeProject.filesList.map((fName) => (
                      <div
                        key={fName}
                        className="px-2 py-1 rounded bg-[#07172b]/70 border border-[#00f0ff]/20 text-[10px] text-cyan-200 flex items-center gap-1.5 truncate"
                      >
                        <ChevronRight className="size-2.5 text-[#ffb700] shrink-0" />
                        <span className="truncate">{fName}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : viewMode === 'cloud' ? (
          /* Cloud & Hermes Tasks View */
          <div className="flex-1 overflow-y-auto p-4 space-y-4 font-mono text-xs">
            {/* Hermes Agent Autonomous Tasks */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-[#00f0ff] font-bold text-sm">
                  <Bot className="size-4 text-[#ffb700]" />
                  <span>HERMES AGENT ORCHESTRATION & TASKS</span>
                </div>
                <span className="text-[11px] text-[#80f7ff]/70">Profile Home: Eng. Ibrahim Abdelsattar</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {IBRAHIM_HERMES_TASKS.map((t) => (
                  <div
                    key={t.id}
                    className="p-3 rounded-lg bg-[#020b14]/90 border border-[#00f0ff]/25 hover:border-[#00f0ff]/50 transition-all shadow-[0_0_15px_rgba(0,240,255,0.05)]"
                  >
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <span className="font-bold text-cyan-200 text-xs">{t.title}</span>
                      <span
                        className={`text-[9px] px-2 py-0.5 rounded border ${
                          t.status === 'Operational'
                            ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                            : t.status === 'In Progress'
                            ? 'bg-amber-400/10 text-amber-300 border-amber-400/30'
                            : 'bg-[#00f0ff]/10 text-[#00f0ff] border-[#00f0ff]/30'
                        }`}
                      >
                        {t.status}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed font-sans">{t.description}</p>
                    <div className="flex items-center justify-between mt-2.5 pt-2 border-t border-[#00f0ff]/10 text-[10px] text-[#80f7ff]/60">
                      <span>Engine: {t.model}</span>
                      <span>Latency: {t.latency}</span>
                      <button
                        onClick={() => handleSendMessage(`Run diagnostic review on ${t.title}`)}
                        className="text-[#ffb700] hover:underline"
                      >
                        Inspect Task →
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Google Cloud Platform (GCP) Services */}
            <div className="pt-2">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 text-[#00f0ff] font-bold text-sm">
                  <Cloud className="size-4 text-[#ffb700]" />
                  <span>GOOGLE CLOUD PLATFORM (GCP) CONNECTED INFRASTRUCTURE</span>
                </div>
                <span className="text-[11px] text-emerald-400 flex items-center gap-1">
                  <Activity className="size-3" />
                  <span>6 Services Synchronized</span>
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {IBRAHIM_GCP_SERVICES.map((g) => (
                  <div
                    key={g.id}
                    className="p-3 rounded-lg bg-[#020b14]/90 border border-[#00f0ff]/25 hover:border-[#00f0ff]/50 transition-all flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between gap-2 mb-1">
                        <span className="font-bold text-[#00f0ff] text-xs flex items-center gap-1.5">
                          <Database className="size-3 text-[#ffb700]" />
                          <span>{g.name}</span>
                        </span>
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                          {g.status}
                        </span>
                      </div>
                      <span className="text-[10px] text-[#80f7ff]/60">{g.region} • {g.category}</span>
                      <p className="text-[11px] text-slate-300 mt-1.5 leading-relaxed font-sans">
                        {g.description}
                      </p>
                    </div>

                    <div className="mt-3 pt-2 border-t border-[#00f0ff]/10">
                      <div className="text-[10px] text-amber-300 mb-1.5">⚡ {g.metrics}</div>
                      <div className="flex flex-wrap gap-1">
                        {g.tools.map((t) => (
                          <span
                            key={t}
                            className="text-[9px] px-1.5 py-0.5 rounded bg-[#00f0ff]/10 text-[#80f7ff] border border-[#00f0ff]/20"
                          >
                            #{t}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : viewMode === 'graph' ? (
          <div className="flex-1 p-3 overflow-hidden flex flex-col justify-center">
            <JarvisNetworkGraph onNodeSelect={handleGraphNodeSelect} height={420} />
          </div>
        ) : viewMode === 'split' ? (
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-3 p-3 overflow-y-auto">
            {/* Left: Chat Feed */}
            <div className="flex flex-col min-h-[380px] bg-[#020914]/80 rounded-lg border border-[#00f0ff]/20 p-3 overflow-hidden">
              <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {messages.map((m) => (
                  <MessageCard key={m.id} message={m} copiedId={copiedId} onCopy={handleCopy} onSpeak={speakText} />
                ))}
                {isLoading && <LoadingMessage />}
                <div ref={messagesEndRef} />
              </div>
            </div>
            {/* Right: Topology */}
            <div className="flex flex-col justify-center">
              <JarvisNetworkGraph onNodeSelect={handleGraphNodeSelect} height={380} />
            </div>
          </div>
        ) : (
          /* Pure Chat View */
          <div className="flex-1 p-4 overflow-y-auto space-y-4">
            {messages.map((m) => (
              <MessageCard key={m.id} message={m} copiedId={copiedId} onCopy={handleCopy} onSpeak={speakText} />
            ))}
            {isLoading && <LoadingMessage />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Chat Input Bar */}
      <div className="p-3 border-t border-[#00f0ff]/20 bg-[#071526]/90">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSendMessage();
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Command J.A.R.V.I.S. regarding Ibrahim's GitHub repos, Hermes Agent tasks, Google Cloud..."
            className="flex-1 bg-[#040e1b] border border-[#00f0ff]/30 focus:border-[#00f0ff] text-xs text-[#00f0ff] px-3.5 py-2.5 rounded-lg outline-none placeholder-[#80f7ff]/40 font-mono transition-all shadow-[inset_0_0_8px_rgba(0,240,255,0.05)]"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-2.5 rounded-lg bg-gradient-to-r from-[#00f0ff] to-[#0088ff] text-[#040d1a] font-bold text-xs flex items-center gap-1.5 hover:shadow-[0_0_15px_rgba(0,240,255,0.5)] transition-all disabled:opacity-40 disabled:cursor-not-allowed font-mono"
          >
            <Send className="size-3.5" />
            <span>EXECUTE</span>
          </button>
        </form>
      </div>
    </div>
  );
};

function MessageCard({
  message,
  copiedId,
  onCopy,
  onSpeak,
}: {
  message: JarvisMessage;
  copiedId: string | null;
  onCopy: (id: string, text: string) => void;
  onSpeak: (text: string) => void;
}) {
  const isUser = message.sender === 'user';
  return (
    <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} group`}>
      <div
        className={`max-w-[85%] rounded-xl p-3.5 text-xs leading-relaxed font-sans shadow-md border ${
          isUser
            ? 'bg-gradient-to-br from-[#062445] to-[#0a3560] border-[#00f0ff]/40 text-cyan-100 rounded-tr-none'
            : 'bg-[#06182c]/90 border-[#00f0ff]/25 text-slate-200 rounded-tl-none'
        }`}
      >
        <div className="flex items-center justify-between gap-4 mb-1 text-[10px] font-mono text-[#80f7ff]/60 border-b border-[#00f0ff]/10 pb-1">
          <span className="font-bold flex items-center gap-1">
            {isUser ? 'ENG. IBRAHIM (COMMANDER)' : 'JARVIS MARK 85'}
            {message.isCached && (
              <span className="px-1 rounded bg-[#00f0ff]/10 text-[#00f0ff] text-[9px]">CACHED</span>
            )}
          </span>
          <span>{message.timestamp}</span>
        </div>

        <p className="whitespace-pre-wrap">{formatDisplayContentWithPunctuation(message.content)}</p>

        {message.technicalKeywords && message.technicalKeywords.length > 0 && (
          <div className="mt-2.5 pt-2 border-t border-[#00f0ff]/10 flex flex-wrap gap-1.5">
            {message.technicalKeywords.map((kw, i) => (
              <span
                key={i}
                className="px-2 py-0.5 rounded text-[10px] font-mono bg-[#00f0ff]/10 border border-[#00f0ff]/30 text-[#00f0ff]"
              >
                #{kw}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 mt-1 px-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => onCopy(message.id, message.content)}
          className="p-1 rounded text-[#80f7ff]/50 hover:text-[#00f0ff]"
          title="Copy message"
        >
          {copiedId === message.id ? <Check className="size-3 text-emerald-400" /> : <Copy className="size-3" />}
        </button>
        {!isUser && (
          <button
            onClick={() => onSpeak(message.content)}
            className="p-1 rounded text-[#80f7ff]/50 hover:text-[#00f0ff]"
            title="Read Aloud"
          >
            <Volume2 className="size-3" />
          </button>
        )}
      </div>
    </div>
  );
}

function LoadingMessage() {
  return (
    <div className="flex items-start gap-2 text-xs font-mono text-[#00f0ff] p-2 animate-pulse">
      <Bot className="size-4 animate-spin" />
      <span>JARVIS computing neural response...</span>
    </div>
  );
}

export default JarvisCoreWidget;
