export type LanguageMode = 'arabic_egyptian' | 'english';

export type DashboardPreset = 
  | 'executive_overview'
  | 'ai_architect'
  | 'saas_founder'
  | 'instructor'
  | 'biometric_health';

export interface IbrahimProfile {
  name: string;
  roles: string[];
  teachingSubjects: string[];
  saasProjects: {
    supplyMind: {
      name: string;
      description: string;
    };
    csatPlatform: {
      name: string;
      description: string;
    };
    dawrly: {
      name: string;
      description: string;
    };
  };
}

export interface BiometricTelemetry {
  heartRate: number; // bpm
  hrv: number; // ms
  energyLevel: number; // 0-100%
  stressIndex: number; // 0-100%
  focusScore: number; // 0-100%
  sleepQuality: number; // 0-100%
  circadianPhase: 'Awaiting Sensor' | 'Peak Focus' | 'Optimal Learning' | 'Recovery Window' | 'Rest Period';
  cameraPulseActive: boolean;
  lastSyncTimestamp: string;
}

export interface JarvisMessage {
  id: string;
  sender: 'user' | 'jarvis';
  content: string;
  timestamp: string;
  audioUrl?: string;
  technicalKeywords?: string[];
  actionTaken?: string;
  groundingSources?: { title: string; url: string }[];
  isCached?: boolean;
}

export interface ExecutiveTask {
  id: string;
  title: string;
  project: 'SupplyMind AI' | 'C-SAT Platform' | 'Dawrly' | 'Teaching/Mentorship' | 'University (Computer Science)' | 'Personal/Health';
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'BACKLOG' | 'IN_PROGRESS' | 'IN_REVIEW' | 'COMPLETED';
  dueDate: string;
  estimatedHours: number;
  tags: string[];
  subtasks: { id: string; title: string; completed: boolean }[];
  addedByVoice?: boolean;
}

export interface CalendarEventItem {
  id: string;
  title: string;
  timeStart: string;
  timeEnd: string;
  category: 'LECTURE' | 'RAG_DEV' | 'EXECUTIVE' | 'RECOVERY' | 'MENTORING';
  locationOrUrl?: string;
  isSyncedWithGoogle: boolean;
  notes?: string;
}

export interface VaultSecretItem {
  id: string;
  serviceName: string;
  keyName: string;
  category: 'Supabase' | 'OmniRoute' | 'Hostinger VPS' | 'Cloudflare' | 'General';
  encryptedData: string;
  lastModified: string;
}

export interface ExecutiveReportData {
  title: string;
  projectTarget: string;
  executiveSummary: string;
  swot: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  riskMatrix: {
    risk: string;
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
    likelihood: 'HIGH' | 'MEDIUM' | 'LOW';
    mitigation: string;
  }[];
  recommendations: string[];
  generatedAt: string;
}

export interface WidgetConfig {
  id: string;
  name: string;
  visible: boolean;
  order: number;
  columnSpan: 1 | 2 | 3 | 4;
}

export type AgentId = 
  | 'jarvis_chief'
  | 'supplymind_architect'
  | 'pytorch_instructor'
  | 'opensource_evaluator';

export interface AgentConfig {
  id: AgentId;
  name: string;
  roleTitle: string;
  avatarIcon: string;
  accentColor: string;
  systemPrompt: string;
  greetingMessage: string;
  capabilities: string[];
}

export interface CallSessionState {
  isActive: boolean;
  isRinging: boolean;
  isMuted: boolean;
  isSpeakerOn: boolean;
  durationSeconds: number;
  activeVoiceModelId: string;
  callLog: {
    id: string;
    speaker: 'Ibrahim' | 'AI Agent';
    text: string;
    timestamp: string;
    audioLatencyMs?: number;
  }[];
}

// Live Public API Data Types
export interface LiveWeatherData {
  temperature: number;
  humidity: number;
  windSpeed: number;
  weatherCode: number;
  location: string;
  isDay: boolean;
  feelsLike: number;
  uvIndex: number;
  timestamp: string;
}

export interface LiveNewsItem {
  title: string;
  extract: string;
  thumbnail?: string;
  url: string;
  type: 'featured' | 'mostread' | 'onthisday';
}

export interface LiveQuote {
  content: string;
  author: string;
}

export interface LiveHoliday {
  date: string;
  name: string;
  localName: string;
  countryCode: string;
  fixed: boolean;
  global: boolean;
}

export interface LiveExchangeRate {
  base: string;
  target: string;
  rate: number;
  date: string;
}

export interface LiveGitHubRepo {
  name: string;
  fullName: string;
  description: string;
  stars: number;
  forks: number;
  language: string;
  url: string;
  updatedAt: string;
}

export interface LiveNumberFact {
  number: number;
  text: string;
  type: string;
}

export interface LiveMarketCoin {
  name: string;
  symbol: string;
  priceUsd: number;
  change24h: number;
  marketCapUsd: number;
  icon: string;
}

export interface LiveGoldPrice {
  price: number;
  symbol: string;
  currency: string;
  timestamp: string;
}

export interface LiveISSPosition {
  latitude: number;
  longitude: number;
  altitudeKm: number;
  velocityKmh: number;
  visibility: string;
  timestamp: string;
}

export interface LivePublicApiEntry {
  name: string;
  description: string;
  url: string;
  auth: string;
  https: string;
  cors: string;
}

export interface LivePublicApiCatalog {
  source: string;
  sourceUrl: string;
  updatedAt: string;
  entries: LivePublicApiEntry[];
}

export interface MusicTrack {
  id: string;
  title: string;
  artist: string;
  duration: number | null;
  path: string;
  sizeBytes: number;
}

export interface MusicCommand {
  id: number;
  request: string;
}
