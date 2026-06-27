export type HealthTone = "good" | "warn" | "alert" | "neutral";
export type HealthSeverity = "critical" | "warning" | "info" | "good";
export type SourceStatus = "connected" | "pending" | "planned" | "offline";

export type HealthScore = {
  key: "readiness" | "sleep" | "recovery" | "stress" | "activity";
  label: string;
  value: number;
  delta: number;
  detail: string;
  tone: HealthTone;
};

export type HealthMetric = {
  label: string;
  value: string;
  unit?: string;
  detail: string;
  tone: HealthTone;
};

export type TrendSample = {
  label: string;
  value: number;
};

export type TrendSeries = {
  title: string;
  window: "7d" | "30d" | "90d";
  valueLabel: string;
  summary: string;
  detail: string;
  accent: string;
  samples: TrendSample[];
};

export type HealthAlert = {
  severity: HealthSeverity;
  title: string;
  detail: string;
  impact: string;
};

export type WorkoutEntry = {
  name: string;
  type: string;
  duration: string;
  load: string;
  time: string;
  note: string;
};

export type RecoveryLine = {
  label: string;
  value: string;
  detail: string;
  tone: HealthTone;
};

export type HealthNote = {
  title: string;
  body: string;
  tag: string;
  updatedAt: string;
};

export type HealthSource = {
  name: string;
  status: SourceStatus;
  description: string;
  readiness: string;
};

export type HealthDashboardSnapshot = {
  updatedAt: string;
  summary: string;
  scores: HealthScore[];
  metrics: HealthMetric[];
  trends: TrendSeries[];
  alerts: HealthAlert[];
  workouts: WorkoutEntry[];
  recovery: RecoveryLine[];
  notes: HealthNote[];
  sources: HealthSource[];
  recommendation: {
    title: string;
    detail: string;
    action: string;
  };
};

export const healthDashboardSnapshot: HealthDashboardSnapshot = {
  updatedAt: "2026-06-04 06:42 GST",
  summary: "Load is controlled, but sleep debt and reduced HRV are dragging readiness below target.",
  scores: [
    {
      key: "readiness",
      label: "Readiness",
      value: 78,
      delta: 4,
      detail: "Stable, but not fully recovered",
      tone: "warn",
    },
    {
      key: "sleep",
      label: "Sleep",
      value: 84,
      delta: -3,
      detail: "7h 18m over the last night",
      tone: "good",
    },
    {
      key: "recovery",
      label: "Recovery",
      value: 73,
      delta: -2,
      detail: "Sleep debt remains elevated",
      tone: "warn",
    },
    {
      key: "stress",
      label: "Stress",
      value: 61,
      delta: 6,
      detail: "Managed, but trending upwards",
      tone: "alert",
    },
    {
      key: "activity",
      label: "Activity",
      value: 88,
      delta: 2,
      detail: "Good movement and training adherence",
      tone: "good",
    },
  ],
  metrics: [
    { label: "Sleep duration", value: "7h 18m", detail: "+22m vs 7d avg", tone: "good" },
    { label: "Sleep quality", value: "82%", detail: "Light sleep up, deep sleep flat", tone: "good" },
    { label: "Resting HR", value: "54", unit: "bpm", detail: "+2 bpm vs baseline", tone: "warn" },
    { label: "HRV", value: "42", unit: "ms", detail: "-6 ms over 7 days", tone: "alert" },
    { label: "Body weight", value: "74.9", unit: "kg", detail: "-0.2 kg this week", tone: "neutral" },
    { label: "Body fat", value: "15.8", unit: "%", detail: "Holding steady", tone: "neutral" },
    { label: "Steps", value: "11.8k", detail: "+14% vs target", tone: "good" },
    { label: "Calories burned", value: "2,480", detail: "Active expenditure tracked", tone: "neutral" },
    { label: "Active minutes", value: "94", detail: "Above weekly baseline", tone: "good" },
    { label: "Workout count", value: "5", detail: "2 strength / 2 zone 2 / 1 mobility", tone: "good" },
    { label: "Hydration", value: "2.4L", detail: "600ml short of target", tone: "warn" },
    { label: "Energy", value: "Medium", detail: "Fatigue present mid-afternoon", tone: "warn" },
  ],
  trends: [
    {
      title: "Sleep consistency",
      window: "7d",
      valueLabel: "81%",
      summary: "Solid baseline with one late-night dip.",
      detail: "A smoother bedtime would lift the whole stack.",
      accent: "from-[#7cf0d8] via-[#45d0ff] to-[#2f6bff]",
      samples: [
        { label: "Mon", value: 78 },
        { label: "Tue", value: 80 },
        { label: "Wed", value: 85 },
        { label: "Thu", value: 83 },
        { label: "Fri", value: 79 },
        { label: "Sat", value: 76 },
        { label: "Sun", value: 84 },
      ],
    },
    {
      title: "Recovery trend",
      window: "30d",
      valueLabel: "73",
      summary: "Recovery is softer than the training block requires.",
      detail: "HRV dipped after two higher-load sessions and late sleep.",
      accent: "from-[#a78bfa] via-[#7c3aed] to-[#4f46e5]",
      samples: [
        { label: "W1", value: 71 },
        { label: "W2", value: 74 },
        { label: "W3", value: 70 },
        { label: "W4", value: 72 },
        { label: "W5", value: 75 },
        { label: "W6", value: 73 },
        { label: "W7", value: 76 },
        { label: "W8", value: 74 },
      ],
    },
    {
      title: "Load balance",
      window: "90d",
      valueLabel: "+12",
      summary: "Training load is controlled, with room to push if sleep improves.",
      detail: "Consistency is good; fatigue spikes are mostly sleep-related.",
      accent: "from-[#fbbf24] via-[#fb7185] to-[#f97316]",
      samples: [
        { label: "D1", value: 44 },
        { label: "D2", value: 46 },
        { label: "D3", value: 49 },
        { label: "D4", value: 51 },
        { label: "D5", value: 53 },
        { label: "D6", value: 56 },
        { label: "D7", value: 54 },
        { label: "D8", value: 57 },
        { label: "D9", value: 58 },
        { label: "D10", value: 60 },
      ],
    },
  ],
  alerts: [
    {
      severity: "warning",
      title: "Sleep debt creeping up",
      detail: "Two shorter nights in the last four days are suppressing recovery.",
      impact: "Target an earlier bedtime tonight.",
    },
    {
      severity: "warning",
      title: "Elevated resting heart rate",
      detail: "Resting HR is running 2 bpm above baseline.",
      impact: "Avoid stacking intensity until it normalises.",
    },
    {
      severity: "critical",
      title: "HRV reduced",
      detail: "HRV is down 6 ms week-on-week.",
      impact: "Prioritise sleep and hydration before the next load spike.",
    },
    {
      severity: "info",
      title: "Workouts still consistent",
      detail: "Training frequency remains strong with no missed sessions.",
      impact: "Maintain cadence; avoid turning it into a recovery problem.",
    },
  ],
  workouts: [
    {
      name: "Zone 2 ride",
      type: "Cardio",
      duration: "52m",
      load: "Moderate",
      time: "Yesterday 18:20",
      note: "Aerobic base without excessive strain.",
    },
    {
      name: "Upper strength",
      type: "Strength",
      duration: "46m",
      load: "High",
      time: "Tue 19:10",
      note: "Bench volume up slightly; good control.",
    },
    {
      name: "Mobility + walk",
      type: "Recovery",
      duration: "34m",
      load: "Low",
      time: "Mon 07:45",
      note: "Useful reset after travel and desk load.",
    },
  ],
  recovery: [
    { label: "Sleep debt", value: "1h 42m", detail: "Three nights to clear", tone: "warn" },
    { label: "Readiness trend", value: "Down 3", detail: "7-day slope softened", tone: "warn" },
    { label: "Strain vs recovery", value: "Balanced", detail: "Close to neutral, but on the edge", tone: "neutral" },
    { label: "Today’s call", value: "Hold", detail: "Keep the day light and move early", tone: "good" },
  ],
  notes: [
    {
      title: "Late sleep after ops call",
      body: "Bedtime slipped after a late call; that still shows up in HRV the next morning.",
      tag: "observation",
      updatedAt: "Today · 06:30",
    },
    {
      title: "Hydration lag",
      body: "Fluid intake missed target yesterday afternoon — likely contributing to fatigue.",
      tag: "flag",
      updatedAt: "Today · 06:30",
    },
    {
      title: "Good training consistency",
      body: "No missed sessions this week; the risk is overreaching rather than undertraining.",
      tag: "strength",
      updatedAt: "Yesterday · 22:15",
    },
  ],
  sources: [
    {
      name: "Apple Health",
      status: "planned",
      description: "Primary iPhone health store for sleep, steps, weight and HRV.",
      readiness: "Adapter stub only",
    },
    {
      name: "Oura",
      status: "planned",
      description: "Wearable sleep and recovery feed.",
      readiness: "Schema ready",
    },
    {
      name: "Garmin",
      status: "planned",
      description: "Training load, activity and workout metadata.",
      readiness: "Integration pending",
    },
    {
      name: "Strava",
      status: "planned",
      description: "Workout history and effort consistency.",
      readiness: "OAuth-ready later",
    },
    {
      name: "Fitbit",
      status: "planned",
      description: "Optional fallback for weight, sleep and activity.",
      readiness: "Not yet wired",
    },
    {
      name: "Manual entry",
      status: "connected",
      description: "Fast override for notes, weight, hydration and ad hoc flags.",
      readiness: "Available now",
    },
    {
      name: "CSV import",
      status: "pending",
      description: "Bulk import path for historical health data.",
      readiness: "Parser to be added",
    },
  ],
  recommendation: {
    title: "Today’s recommendation",
    detail: "Keep training light, push hydration, and move bedtime earlier by 30–45 minutes.",
    action: "Aim for a low-strain day and re-check recovery tomorrow morning.",
  },
};

export function formatSignedDelta(delta: number): string {
  const sign = delta > 0 ? "+" : "";
  return `${sign}${delta}`;
}

export function buildSparklinePath(samples: TrendSample[], width = 240, height = 96): string {
  if (!samples.length) return "";

  const values = samples.map((sample) => sample.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const paddingX = 6;
  const paddingY = 8;
  const usableWidth = width - paddingX * 2;
  const usableHeight = height - paddingY * 2;

  return samples
    .map((sample, index) => {
      const x = paddingX + (index / Math.max(samples.length - 1, 1)) * usableWidth;
      const y = paddingY + (1 - (sample.value - min) / range) * usableHeight;
      const command = index === 0 ? "M" : "L";
      return `${command}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

export function buildSparklineFillPath(samples: TrendSample[], width = 240, height = 96): string {
  const line = buildSparklinePath(samples, width, height);
  if (!line) return "";

  const paddingX = 6;
  const bottomY = height - 8;
  const firstX = paddingX;
  const lastX = width - paddingX;
  return `${line} L${lastX.toFixed(1)} ${bottomY.toFixed(1)} L${firstX.toFixed(1)} ${bottomY.toFixed(1)} Z`;
}
