import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import {
  Radio,
  Cpu,
  Zap,
  Activity,
  Maximize2,
  Minimize2,
  Shield,
  Layers,
  Sparkles,
  Flame,
} from 'lucide-react';

export type OrbTheme = 'hermes' | 'jarvis' | 'ultron' | 'gwen';
export type VisualizerMode = 'neural_orb' | 'arc_reactor' | 'waveform_matrix';

export interface JarvisUltronVoiceOrbProps {
  analyser?: AnalyserNode | null;
  outputAnalyser?: AnalyserNode | null;
  isActive: boolean;
  isSpeaking: boolean;
  isUserSpeaking?: boolean;
  isMuted?: boolean;
  selectedPersona?: 'jarvis' | 'gwen';
  accentColor?: string;
  themeMode?: OrbTheme;
  onThemeChange?: (theme: OrbTheme) => void;
  className?: string;
  sampleRate?: number;
  isMini?: boolean;
}

interface Particle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  size: number;
  alpha: number;
  maxAlpha: number;
  life: number;
  maxLife: number;
  color: string;
}

interface Shockwave {
  radius: number;
  maxRadius: number;
  alpha: number;
  color: string;
  width: number;
}

interface LightningBranch {
  points: { x: number; y: number }[];
  alpha: number;
  color: string;
  width: number;
}

// 3D Point on Sphere
interface SpherePoint {
  lat: number;
  lon: number;
  baseRadius: number;
  freqIndex: number;
}

export const JarvisUltronVoiceOrb: React.FC<JarvisUltronVoiceOrbProps> = ({
  analyser,
  outputAnalyser,
  isActive,
  isSpeaking,
  isUserSpeaking = false,
  isMuted = false,
  selectedPersona = 'jarvis',
  accentColor: _accentColor,
  themeMode: controlledTheme,
  onThemeChange,
  className = '',
  sampleRate = 48000,
  isMini = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Theme state
  const [internalTheme, setInternalTheme] = useState<OrbTheme>(
    selectedPersona === 'gwen' ? 'gwen' : 'jarvis'
  );
  const currentTheme = controlledTheme || internalTheme;

  // Visualizer Display Mode
  const [visMode, setVisMode] = useState<VisualizerMode>('neural_orb');

  // Expanded Stage Mode
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  // Audio Sensitivity Multiplier
  const [sensitivity, setSensitivity] = useState<number>(1.25);

  const handleToggleTheme = useCallback(
    (theme: OrbTheme) => {
      setInternalTheme(theme);
      onThemeChange?.(theme);
    },
    [onThemeChange]
  );

  // Sync with persona if persona changes and theme not overridden
  useEffect(() => {
    if (!controlledTheme) {
      setInternalTheme(selectedPersona === 'gwen' ? 'gwen' : 'jarvis');
    }
  }, [selectedPersona, controlledTheme]);

  // Audio telemetry states for HUD overlay
  const [telemetry, setTelemetry] = useState({
    peakFreq: 0,
    bassEnergy: 0,
    midEnergy: 0,
    trebleEnergy: 0,
    sampleRateKhz: (sampleRate / 1000).toFixed(1),
    decibels: -60,
    pitchNote: 'C4',
  });

  // Color palette configuration based on active theme
  const themePalette = useMemo(() => {
    switch (currentTheme) {
      case 'hermes':
        return {
          name: 'HERMES QUANTUM ORB',
          subtitle: 'NEURAL TELECOMM PROTOCOL',
          primary: '#00d2c4',
          secondary: '#14b8a6',
          tertiary: '#2dd4bf',
          glow: 'rgba(0, 210, 196, 0.45)',
          glowSubtle: 'rgba(0, 210, 196, 0.15)',
          coreInner: '#ffffff',
          coreMid: '#00d2c4',
          coreOuter: '#0f766e',
          lightningColor: '#99f6e4',
          particleColors: ['#00d2c4', '#14b8a6', '#2dd4bf', '#99f6e4', '#ffffff'],
          border: 'border-[#00d2c4]/50',
          bgGlow: 'from-[#00d2c4]/15 via-slate-950 to-teal-950/30',
          badgeText: 'text-[#00d2c4]',
          badgeBg: 'bg-teal-950/80 border-[#00d2c4]/50',
          accent: '#00d2c4',
        };
      case 'ultron':
        return {
          name: 'ULTRON PROTOCOL',
          subtitle: 'NEURAL CONSCIOUSNESS MATRIX',
          primary: '#ff1744',
          secondary: '#ff5252',
          tertiary: '#ff8a80',
          glow: 'rgba(255, 23, 68, 0.45)',
          glowSubtle: 'rgba(255, 23, 68, 0.15)',
          coreInner: '#ffffff',
          coreMid: '#ff1744',
          coreOuter: '#990000',
          lightningColor: '#ff8a80',
          particleColors: ['#ff1744', '#ff5252', '#ff8a80', '#ffd600', '#ffffff'],
          border: 'border-red-500/50',
          bgGlow: 'from-red-950/50 via-slate-950 to-red-950/30',
          badgeText: 'text-red-400',
          badgeBg: 'bg-red-950/80 border-red-500/50',
          accent: '#ff1744',
        };
      case 'gwen':
        return {
          name: 'GWEN NEURAL SENTINEL',
          subtitle: 'SOLAR RESONANCE MATRIX',
          primary: '#f59e0b',
          secondary: '#fbbf24',
          tertiary: '#f43f5e',
          glow: 'rgba(245, 158, 11, 0.45)',
          glowSubtle: 'rgba(245, 158, 11, 0.15)',
          coreInner: '#ffffff',
          coreMid: '#f59e0b',
          coreOuter: '#b45309',
          lightningColor: '#fde68a',
          particleColors: ['#f59e0b', '#fbbf24', '#f43f5e', '#fde68a', '#ffffff'],
          border: 'border-amber-500/50',
          bgGlow: 'from-amber-950/50 via-slate-950 to-amber-950/30',
          badgeText: 'text-amber-300',
          badgeBg: 'bg-amber-950/80 border-amber-500/50',
          accent: '#f59e0b',
        };
      case 'jarvis':
      default:
        return {
          name: 'J.A.R.V.I.S. ARC REACTOR',
          subtitle: 'STARK MK-85 QUANTUM ORB',
          primary: '#00f0ff',
          secondary: '#38bdf8',
          tertiary: '#818cf8',
          glow: 'rgba(0, 240, 255, 0.45)',
          glowSubtle: 'rgba(0, 240, 255, 0.15)',
          coreInner: '#ffffff',
          coreMid: '#00f0ff',
          coreOuter: '#0284c7',
          lightningColor: '#cffafe',
          particleColors: ['#00f0ff', '#80f7ff', '#38bdf8', '#cffafe', '#ffffff'],
          border: 'border-[#00f0ff]/50',
          bgGlow: 'from-[#00f0ff]/15 via-slate-950 to-cyan-950/30',
          badgeText: 'text-[#00f0ff]',
          badgeBg: 'bg-cyan-950/80 border-[#00f0ff]/50',
          accent: '#00f0ff',
        };
    }
  }, [currentTheme]);

  // Pre-generate 3D Sphere Lattice Nodes
  const sphereNodes = useMemo(() => {
    const nodes: SpherePoint[] = [];
    const latBands = 10;
    const lonBands = 18;

    for (let i = 0; i <= latBands; i++) {
      const lat = (i / latBands) * Math.PI - Math.PI / 2;
      for (let j = 0; j < lonBands; j++) {
        const lon = (j / lonBands) * Math.PI * 2;
        const freqIndex = (i * lonBands + j) % 64;
        nodes.push({ lat, lon, baseRadius: 1, freqIndex });
      }
    }
    return nodes;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let phase = 0;
    let rotX = 0.25;
    let rotY = 0;
    let rotZ = 0.1;

    // Smoothed energy variables for buttery physics
    let smoothBass = 0;
    let smoothMid = 0;
    let smoothTreble = 0;
    let smoothScale = 1;
    let lastVolumeSpike = 0;

    // Particles and FX
    const particles: Particle[] = [];
    const shockwaves: Shockwave[] = [];
    let lightnings: LightningBranch[] = [];

    const MAX_PARTICLES = isExpanded ? 70 : 45;

    const createParticle = (cx: number, cy: number, currentR: number) => {
      const angle = Math.random() * Math.PI * 2;
      const elevation = (Math.random() - 0.5) * Math.PI;
      const dist = currentR * (0.8 + Math.random() * 0.4);
      const speed = 0.8 + Math.random() * 2.2;
      const colors = themePalette.particleColors;

      const vx = Math.cos(angle) * Math.cos(elevation) * speed;
      const vy = Math.sin(elevation) * speed;
      const vz = Math.sin(angle) * Math.cos(elevation) * speed;

      return {
        x: cx + Math.cos(angle) * dist,
        y: cy + Math.sin(elevation) * dist * 0.5,
        z: Math.sin(angle) * dist,
        vx,
        vy,
        vz,
        size: 1.2 + Math.random() * 2.8,
        alpha: 0.1,
        maxAlpha: 0.5 + Math.random() * 0.5,
        life: 0,
        maxLife: 35 + Math.random() * 55,
        color: colors[Math.floor(Math.random() * colors.length)],
      };
    };

    const triggerShockwave = (baseR: number) => {
      shockwaves.push({
        radius: baseR * 0.8,
        maxRadius: baseR * (2.4 + Math.random() * 0.6),
        alpha: 0.9,
        color: themePalette.primary,
        width: 2.5,
      });
    };

    const triggerLightning = (cx: number, cy: number, innerR: number, outerR: number) => {
      const branches: LightningBranch[] = [];
      const numBolts = 2 + Math.floor(Math.random() * 3);

      for (let b = 0; b < numBolts; b++) {
        const startAngle = Math.random() * Math.PI * 2;
        const targetAngle = startAngle + (Math.random() - 0.5) * 0.8;
        const pts: { x: number; y: number }[] = [];

        let curX = cx + Math.cos(startAngle) * innerR;
        let curY = cy + Math.sin(startAngle) * innerR;
        pts.push({ x: curX, y: curY });

        const steps = 6 + Math.floor(Math.random() * 4);
        for (let s = 1; s <= steps; s++) {
          const t = s / steps;
          const r = innerR + (outerR - innerR) * t;
          const ang = startAngle + (targetAngle - startAngle) * t;
          const jitterX = (Math.random() - 0.5) * 16;
          const jitterY = (Math.random() - 0.5) * 16;
          curX = cx + Math.cos(ang) * r + jitterX;
          curY = cy + Math.sin(ang) * r + jitterY;
          pts.push({ x: curX, y: curY });
        }

        branches.push({
          points: pts,
          alpha: 1,
          color: themePalette.lightningColor,
          width: 1.5 + Math.random() * 1.5,
        });
      }
      return branches;
    };

    let lastTelemetryUpdate = 0;

    const render = (time: number) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = rect.width;
      const height = rect.height;

      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const maxDim = Math.min(width, height);
      const baseOrbRadius = isMini
        ? Math.max(12, maxDim * 0.4)
        : Math.max(35, maxDim * (isExpanded ? 0.28 : 0.23));

      // Frequency Audio Analysis
      const targetAnalyser = isSpeaking ? outputAnalyser || analyser : analyser;
      let rawFreq = new Uint8Array(64);
      let rawTime = new Uint8Array(64);
      let realAudioDetected = false;

      if (targetAnalyser) {
        try {
          const bins = targetAnalyser.frequencyBinCount;
          const tempBuf = new Uint8Array(bins);
          targetAnalyser.getByteFrequencyData(tempBuf);
          if (tempBuf.some((v) => v > 0)) {
            realAudioDetected = true;
            for (let i = 0; i < 64; i++) {
              rawFreq[i] = tempBuf[i] || 0;
            }
          }
          const tempTime = new Uint8Array(bins);
          targetAnalyser.getByteTimeDomainData(tempTime);
          for (let i = 0; i < 64; i++) {
            rawTime[i] = tempTime[i] || 128;
          }
        } catch {
          // ignore
        }
      }

      // Energy calculation
      let targetBass = 0;
      let targetMid = 0;
      let targetTreble = 0;

      if (realAudioDetected) {
        let bSum = 0;
        let mSum = 0;
        let tSum = 0;
        for (let i = 0; i < 8; i++) bSum += rawFreq[i];
        for (let i = 8; i < 32; i++) mSum += rawFreq[i];
        for (let i = 32; i < 64; i++) tSum += rawFreq[i];
        targetBass = (bSum / 8 / 255) * sensitivity;
        targetMid = (mSum / 24 / 255) * sensitivity;
        targetTreble = (tSum / 32 / 255) * sensitivity;
      } else if (isSpeaking) {
        // High fidelity speech formant synthesis simulation
        const cadence = Math.sin(phase * 4.5) * 0.5 + 0.5;
        const formant1 = Math.sin(phase * 8.5) * 0.35 + 0.65;
        const formant2 = Math.cos(phase * 13) * 0.3 + 0.7;
        targetBass = (0.4 + cadence * 0.45) * formant1 * sensitivity;
        targetMid = (0.35 + Math.sin(phase * 10) * 0.3) * formant2 * sensitivity;
        targetTreble = (0.25 + Math.cos(phase * 15) * 0.25) * sensitivity;
      } else if (isUserSpeaking && !isMuted) {
        targetBass = (0.35 + Math.sin(phase * 3.5) * 0.25) * sensitivity;
        targetMid = 0.3 * sensitivity;
        targetTreble = 0.2 * sensitivity;
      } else if (isActive) {
        targetBass = 0.08 + Math.sin(phase * 1.5) * 0.04;
        targetMid = 0.05 + Math.cos(phase * 1.8) * 0.03;
        targetTreble = 0.03;
      } else {
        targetBass = 0.02;
        targetMid = 0.01;
        targetTreble = 0.01;
      }

      // Smooth interpolation (spring-lerp)
      smoothBass += (targetBass - smoothBass) * 0.22;
      smoothMid += (targetMid - smoothMid) * 0.22;
      smoothTreble += (targetTreble - smoothTreble) * 0.22;

      // Detection of transient speech volume spikes for shockwaves and lightning
      if (smoothBass > 0.45 && time - lastVolumeSpike > 350) {
        lastVolumeSpike = time;
        triggerShockwave(baseOrbRadius);
        if (Math.random() > 0.4) {
          lightnings = triggerLightning(cx, cy, baseOrbRadius * 0.6, baseOrbRadius * 1.6);
        }
      }

      // Dynamic scale computation
      const targetScale = 1 + smoothBass * 0.5 + (isSpeaking ? 0.15 : 0);
      smoothScale += (targetScale - smoothScale) * 0.18;
      const currentRadius = baseOrbRadius * smoothScale;

      // Update telemetry
      if (time - lastTelemetryUpdate > 90) {
        lastTelemetryUpdate = time;
        let peakIndex = 0;
        let peakVal = 0;
        for (let i = 0; i < 64; i++) {
          if (rawFreq[i] > peakVal) {
            peakVal = rawFreq[i];
            peakIndex = i;
          }
        }
        const calculatedHz = Math.round(
          peakIndex * ((sampleRate || 48000) / (targetAnalyser?.fftSize || 256))
        );
        const overallEnergy = (smoothBass * 0.5 + smoothMid * 0.3 + smoothTreble * 0.2) * 100;
        const dbApprox = Math.round(-60 + overallEnergy * 0.62);

        // Musical note estimation
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        let pitchStr = 'A3';
        if (calculatedHz > 50) {
          const semitones = 12 * (Math.log2(calculatedHz / 440));
          const midi = Math.round(69 + semitones);
          const noteIndex = ((midi % 12) + 12) % 12;
          const octave = Math.floor(midi / 12) - 1;
          pitchStr = `${notes[noteIndex]}${octave}`;
        }

        setTelemetry({
          peakFreq: calculatedHz > 0 ? calculatedHz : isSpeaking ? 385 : 0,
          bassEnergy: Math.min(100, Math.round(smoothBass * 100)),
          midEnergy: Math.min(100, Math.round(smoothMid * 100)),
          trebleEnergy: Math.min(100, Math.round(smoothTreble * 100)),
          sampleRateKhz: ((sampleRate || 48000) / 1000).toFixed(1),
          decibels: Math.max(-60, Math.min(0, dbApprox)),
          pitchNote: pitchStr,
        });
      }

      // ─── 1. VOLUMETRIC BACKGROUND AURA & CHROMATIC RADIANCE ───
      const glowR = currentRadius * (2.1 + smoothMid * 0.7);
      const bgGlow = ctx.createRadialGradient(cx, cy, currentRadius * 0.2, cx, cy, glowR);
      bgGlow.addColorStop(0, themePalette.glow);
      bgGlow.addColorStop(0.4, themePalette.glowSubtle);
      bgGlow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = bgGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
      ctx.fill();

      // ─── 2. EXPANDING SONIC SHOCKWAVES ───
      for (let i = shockwaves.length - 1; i >= 0; i--) {
        const sw = shockwaves[i];
        sw.radius += 3.5;
        sw.alpha -= 0.022;

        if (sw.alpha <= 0 || sw.radius >= sw.maxRadius) {
          shockwaves.splice(i, 1);
          continue;
        }

        ctx.save();
        ctx.strokeStyle = sw.color;
        ctx.lineWidth = sw.width;
        ctx.globalAlpha = sw.alpha;
        ctx.shadowColor = sw.color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(cx, cy, sw.radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // ─── 3. ELECTRIC PLASMA LIGHTNING ARCS ───
      for (let i = lightnings.length - 1; i >= 0; i--) {
        const l = lightnings[i];
        l.alpha -= 0.08;
        if (l.alpha <= 0) {
          lightnings.splice(i, 1);
          continue;
        }

        if (l.points.length > 1) {
          ctx.save();
          ctx.strokeStyle = l.color;
          ctx.lineWidth = l.width;
          ctx.globalAlpha = l.alpha;
          ctx.shadowColor = l.color;
          ctx.shadowBlur = 12;
          ctx.beginPath();
          ctx.moveTo(l.points[0].x, l.points[0].y);
          for (let p = 1; p < l.points.length; p++) {
            ctx.lineTo(l.points[p].x, l.points[p].y);
          }
          ctx.stroke();
          ctx.restore();
        }
      }

      // ─── 4. HOLOGRAPHIC 3D TILTED EQUATORIAL SPECTRUM RING ───
      const eqTilt = 0.35; // perspective flattening
      const eqRadius = currentRadius * 1.48;
      const numEqBars = 48;

      ctx.save();
      ctx.translate(cx, cy);

      // Rotating dashed guide equator
      ctx.beginPath();
      ctx.ellipse(0, 0, eqRadius, eqRadius * eqTilt, phase * 0.5, 0, Math.PI * 2);
      ctx.strokeStyle = themePalette.secondary;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 6]);
      ctx.globalAlpha = 0.45 + smoothBass * 0.4;
      ctx.stroke();
      ctx.setLineDash([]);

      // Vertical 3D Equalizer pins extending out from the tilted ring
      for (let i = 0; i < numEqBars; i++) {
        const angle = (i / numEqBars) * Math.PI * 2 + phase * 0.4;
        const binVal = (rawFreq[i % 32] || 0) / 255;
        const barHeight = 4 + binVal * (36 * sensitivity) * (isSpeaking ? 1.4 : 0.8);

        const xBase = Math.cos(angle) * eqRadius;
        const yBase = Math.sin(angle) * eqRadius * eqTilt;

        // Depth cue: back side of ellipse is dimmer
        const depth = Math.sin(angle);
        const barAlpha = 0.25 + (depth + 1) * 0.35;

        ctx.strokeStyle = i % 2 === 0 ? themePalette.primary : themePalette.tertiary;
        ctx.lineWidth = 1.6;
        ctx.globalAlpha = Math.min(1, barAlpha + smoothBass * 0.3);
        ctx.beginPath();
        ctx.moveTo(xBase, yBase);
        ctx.lineTo(xBase, yBase - barHeight);
        ctx.stroke();
      }
      ctx.restore();

      // ─── 5. TRUE 3D NEURAL SPHERICAL LATTICE (Rotated in 3D Space) ───
      if (visMode === 'neural_orb' || visMode === 'waveform_matrix') {
        const fov = 340;
        ctx.save();
        ctx.translate(cx, cy);

        // Sort nodes by projected Z depth for correct painter's order
        const projectedNodes = sphereNodes.map((node) => {
          const binVal = (rawFreq[node.freqIndex] || 0) / 255;
          const radialMod = currentRadius * (1 + binVal * 0.35 * sensitivity);

          // 3D Cartesian coordinates
          const x0 = radialMod * Math.cos(node.lat) * Math.cos(node.lon);
          const y0 = radialMod * Math.sin(node.lat);
          const z0 = radialMod * Math.cos(node.lat) * Math.sin(node.lon);

          // 3D Euler Rotations: RotX -> RotY -> RotZ
          // Rotate around X
          const y1 = y0 * Math.cos(rotX) - z0 * Math.sin(rotX);
          const z1 = y0 * Math.sin(rotX) + z0 * Math.cos(rotX);

          // Rotate around Y
          const x2 = x0 * Math.cos(rotY) + z1 * Math.sin(rotY);
          const z2 = -x0 * Math.sin(rotY) + z1 * Math.cos(rotY);

          // Rotate around Z
          const x3 = x2 * Math.cos(rotZ) - y1 * Math.sin(rotZ);
          const y3 = x2 * Math.sin(rotZ) + y1 * Math.cos(rotZ);

          // Perspective Projection
          const scale = fov / (fov + z2);
          const projX = x3 * scale;
          const projY = y3 * scale;
          const alpha = Math.max(0.1, Math.min(1, 0.4 + (z2 / currentRadius) * 0.6));

          return { projX, projY, z: z2, scale, alpha, binVal };
        });

        projectedNodes.sort((a, b) => a.z - b.z);

        // Draw connections / filaments between neighboring projected nodes
        ctx.strokeStyle = themePalette.primary;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < projectedNodes.length - 1; i += 2) {
          const p1 = projectedNodes[i];
          const p2 = projectedNodes[i + 1];
          if (p1 && p2 && p1.z > -currentRadius * 0.5) {
            ctx.globalAlpha = p1.alpha * 0.4;
            ctx.moveTo(p1.projX, p1.projY);
            ctx.lineTo(p2.projX, p2.projY);
          }
        }
        ctx.stroke();

        // Draw glowing 3D nodes
        for (let i = 0; i < projectedNodes.length; i++) {
          const p = projectedNodes[i];
          const nodeRadius = Math.max(1, (1.8 + p.binVal * 3) * p.scale);

          ctx.fillStyle = p.z > 0 ? '#ffffff' : themePalette.secondary;
          ctx.globalAlpha = p.alpha;
          ctx.beginPath();
          ctx.arc(p.projX, p.projY, nodeRadius, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }

      // ─── 6. DUAL POLAR FILAMENT WAVE RINGS (Marvel Ultron/Jarvis Silhouette) ───
      const numWaves = 64;
      ctx.save();
      ctx.translate(cx, cy);

      // Primary Outer Wave Ring
      ctx.beginPath();
      for (let i = 0; i <= numWaves; i++) {
        const ang = (i / numWaves) * Math.PI * 2;
        const bin = (rawFreq[i % 32] || 0) / 255;
        const noise = Math.sin(ang * 6 + phase * 3) * 0.12 * (smoothBass + 0.3);
        const r = currentRadius * (1 + bin * 0.55 * sensitivity + noise);
        const x = Math.cos(ang) * r;
        const y = Math.sin(ang) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = themePalette.primary;
      ctx.lineWidth = 2.2;
      ctx.shadowColor = themePalette.primary;
      ctx.shadowBlur = 14 + smoothBass * 20;
      ctx.globalAlpha = 0.9;
      ctx.stroke();

      // Secondary Inner Wave Ring (Harmonic counter-wobble)
      ctx.beginPath();
      for (let i = 0; i <= numWaves; i++) {
        const ang = (i / numWaves) * Math.PI * 2;
        const bin = (rawFreq[(i + 16) % 32] || 0) / 255;
        const noise = Math.cos(ang * 8 - phase * 4) * 0.08 * (smoothMid + 0.2);
        const r = currentRadius * 0.82 * (1 + bin * 0.35 * sensitivity + noise);
        const x = Math.cos(ang) * r;
        const y = Math.sin(ang) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = themePalette.secondary;
      ctx.lineWidth = 1.5;
      ctx.shadowBlur = 8;
      ctx.globalAlpha = 0.75;
      ctx.stroke();
      ctx.restore();

      // ─── 7. THE GLOWING QUANTUM PLASMA CORE (Arc Reactor / Singularity) ───
      const coreR = currentRadius * 0.74;
      const coreGrad = ctx.createRadialGradient(
        cx - coreR * 0.18,
        cy - coreR * 0.18,
        coreR * 0.05,
        cx,
        cy,
        coreR
      );
      coreGrad.addColorStop(0, themePalette.coreInner);
      coreGrad.addColorStop(0.3, themePalette.coreMid);
      coreGrad.addColorStop(0.75, themePalette.coreOuter);
      coreGrad.addColorStop(1, 'rgba(0,0,0,0.5)');

      ctx.save();
      ctx.fillStyle = coreGrad;
      ctx.shadowColor = themePalette.primary;
      ctx.shadowBlur = 24 + smoothBass * 32;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
      ctx.fill();

      // White-hot photon center singularity
      const photonR = coreR * (0.28 + smoothBass * 0.22);
      const photonGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, photonR);
      photonGrad.addColorStop(0, '#ffffff');
      photonGrad.addColorStop(0.4, 'rgba(255,255,255,0.95)');
      photonGrad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = photonGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, photonR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // ─── 8. OSCILLOSCOPE TIME-DOMAIN WAVE RIBBON (Across Core) ───
      ctx.save();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.4;
      ctx.shadowColor = themePalette.primary;
      ctx.shadowBlur = 8;
      ctx.globalAlpha = 0.65 + smoothTreble * 0.35;
      ctx.beginPath();
      const waveSpan = currentRadius * 1.3;
      const waveStartX = cx - waveSpan;
      const steps = 40;

      for (let i = 0; i <= steps; i++) {
        const x = waveStartX + (i / steps) * (waveSpan * 2);
        const tVal = (rawTime[i % 64] - 128) / 128;
        const y =
          cy +
          (isSpeaking
            ? tVal * 28 * sensitivity + Math.sin(x * 0.05 + phase * 6) * 6
            : Math.sin(x * 0.04 + phase * 2) * 4);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // ─── 9. FLOATING HIGH-ENERGY SPARK PARTICLES ───
      if (particles.length < MAX_PARTICLES && (isSpeaking || smoothBass > 0.12)) {
        particles.push(createParticle(cx, cy, currentRadius));
      }

      ctx.save();
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        p.z += p.vz;
        p.life++;
        const prog = p.life / p.maxLife;
        p.alpha = (1 - prog) * p.maxAlpha;

        if (p.life >= p.maxLife) {
          particles.splice(i, 1);
          continue;
        }

        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 7;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // 3D Rotation Speed Modulated by Speech
      const speedFactor = isSpeaking ? 2.5 : 1.0;
      phase += 0.025 * speedFactor;
      rotX += 0.005 * speedFactor;
      rotY += 0.012 * speedFactor;
      rotZ += 0.003 * speedFactor;

      ctx.restore();
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);

    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [
    analyser,
    outputAnalyser,
    isActive,
    isSpeaking,
    isUserSpeaking,
    isMuted,
    themePalette,
    sampleRate,
    visMode,
    isExpanded,
    sensitivity,
    sphereNodes,
  ]);

  if (isMini) {
    return (
      <div
        ref={containerRef}
        className={`relative flex items-center justify-center overflow-hidden select-none ${className}`}
      >
        <canvas
          ref={canvasRef}
          className="h-full w-full rounded-full cursor-pointer filter drop-shadow-[0_0_12px_rgba(0,210,196,0.4)]"
          onClick={() => {
            if (currentTheme === 'hermes') handleToggleTheme('jarvis');
            else if (currentTheme === 'jarvis') handleToggleTheme('ultron');
            else if (currentTheme === 'ultron') handleToggleTheme('gwen');
            else handleToggleTheme('hermes');
          }}
          title="Hermes 3D Neural Voice Orb — Click to cycle theme"
        />
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full rounded-2xl bg-gradient-to-b ${themePalette.bgGlow} border ${
        themePalette.border
      } p-4 flex flex-col justify-between shadow-[0_0_50px_rgba(0,0,0,0.85)] overflow-hidden font-mono select-none transition-all duration-300 ${
        isExpanded ? 'min-h-[380px]' : 'min-h-[270px]'
      } ${className}`}
    >
      {/* Background Holographic Hex Grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-15 bg-[radial-gradient(#ffffff_1px,transparent_1px)]"
        style={{ backgroundSize: '18px 18px' }}
      />

      {/* Top HUD Banner with Full Control Center */}
      <div className="w-full flex flex-wrap items-center justify-between gap-2 z-10 text-[11px] font-mono border-b border-white/10 pb-2.5">
        <div className="flex items-center gap-2.5">
          <div
            className="size-2.5 rounded-full animate-ping"
            style={{ backgroundColor: themePalette.primary }}
          />
          <div>
            <div className="flex items-center gap-2">
              <span className={`font-black tracking-widest uppercase text-xs ${themePalette.badgeText}`}>
                {themePalette.name}
              </span>
              <span className="px-1.5 py-0.2 bg-white/10 rounded text-[9px] text-white/70 uppercase">
                {telemetry.pitchNote}
              </span>
            </div>
            <p className="text-[10px] text-white/40 hidden sm:block">{themePalette.subtitle}</p>
          </div>
        </div>

        {/* Action Controls & Mode Switchers */}
        <div className="flex items-center gap-2">
          {/* Visualizer Display Modes */}
          <div className="flex bg-black/60 p-0.5 rounded-lg border border-white/10 text-[10px]">
            <button
              type="button"
              onClick={() => setVisMode('neural_orb')}
              className={`px-2 py-0.5 rounded transition-all flex items-center gap-1 ${
                visMode === 'neural_orb'
                  ? 'bg-white/20 text-white font-bold'
                  : 'text-white/40 hover:text-white/80'
              }`}
              title="3D Neural Orb Lattice"
            >
              <Sparkles className="size-3" />
              <span className="hidden md:inline">3D Orb</span>
            </button>
            <button
              type="button"
              onClick={() => setVisMode('arc_reactor')}
              className={`px-2 py-0.5 rounded transition-all flex items-center gap-1 ${
                visMode === 'arc_reactor'
                  ? 'bg-white/20 text-white font-bold'
                  : 'text-white/40 hover:text-white/80'
              }`}
              title="Arc Core & Shockwaves"
            >
              <Zap className="size-3" />
              <span className="hidden md:inline">Reactor</span>
            </button>
            <button
              type="button"
              onClick={() => setVisMode('waveform_matrix')}
              className={`px-2 py-0.5 rounded transition-all flex items-center gap-1 ${
                visMode === 'waveform_matrix'
                  ? 'bg-white/20 text-white font-bold'
                  : 'text-white/40 hover:text-white/80'
              }`}
              title="Full Matrix Scope"
            >
              <Layers className="size-3" />
              <span className="hidden md:inline">Matrix</span>
            </button>
          </div>

          {/* Theme Persona Selector Badges */}
          <div className="flex bg-black/60 p-0.5 rounded-lg border border-white/10 text-[10px]">
            <button
              type="button"
              onClick={() => handleToggleTheme('jarvis')}
              className={`px-2 py-0.5 rounded font-bold transition-all flex items-center gap-1 ${
                currentTheme === 'jarvis'
                  ? 'bg-[#00f0ff] text-slate-950 shadow-[0_0_12px_rgba(0,240,255,0.5)]'
                  : 'text-cyan-400/60 hover:text-cyan-200'
              }`}
            >
              <Shield className="size-3" />
              <span>JARVIS</span>
            </button>
            <button
              type="button"
              onClick={() => handleToggleTheme('ultron')}
              className={`px-2 py-0.5 rounded font-bold transition-all flex items-center gap-1 ${
                currentTheme === 'ultron'
                  ? 'bg-red-600 text-white shadow-[0_0_12px_rgba(255,23,68,0.6)]'
                  : 'text-red-400/60 hover:text-red-200'
              }`}
            >
              <Flame className="size-3" />
              <span>ULTRON</span>
            </button>
            <button
              type="button"
              onClick={() => handleToggleTheme('gwen')}
              className={`px-2 py-0.5 rounded font-bold transition-all flex items-center gap-1 ${
                currentTheme === 'gwen'
                  ? 'bg-amber-500 text-slate-950 shadow-[0_0_12px_rgba(245,158,11,0.5)]'
                  : 'text-amber-400/60 hover:text-amber-200'
              }`}
            >
              <Sparkles className="size-3" />
              <span>GWEN</span>
            </button>
          </div>

          {/* Expand / Minimize Stage Toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 rounded-lg bg-black/40 hover:bg-white/10 text-white/70 hover:text-white border border-white/10 transition-colors"
            title={isExpanded ? 'Minimize Stage' : 'Expand Cinematic Stage'}
          >
            {isExpanded ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Interactive Canvas Stage Viewport */}
      <div className="relative w-full flex-1 flex items-center justify-center my-1.5">
        <canvas
          ref={canvasRef}
          className={`w-full max-w-[560px] rounded-full filter drop-shadow-[0_0_25px_rgba(0,0,0,0.7)] cursor-pointer transition-all duration-300 ${
            isExpanded ? 'h-[240px] sm:h-[280px]' : 'h-[170px] sm:h-[195px]'
          }`}
          onClick={() => {
            // Click to cycle theme or toggle expansion
            if (currentTheme === 'hermes') handleToggleTheme('jarvis');
            else if (currentTheme === 'jarvis') handleToggleTheme('ultron');
            else if (currentTheme === 'ultron') handleToggleTheme('gwen');
            else handleToggleTheme('hermes');
          }}
          title="Click the Orb to cycle protocols (HERMES → JARVIS → ULTRON → GWEN)"
        />

        {/* Floating Futuristic HUD Telemetry Panels */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-between px-2 sm:px-4 text-[10px] font-mono">
          {/* Left HUD Panel */}
          <div className="space-y-1.5 backdrop-blur-md bg-black/50 p-2.5 rounded-xl border border-white/10 shadow-lg">
            <div className="flex items-center gap-1.5">
              <Activity className="size-3" style={{ color: themePalette.primary }} />
              <span className="text-white/40">FREQ:</span>
              <span className="font-bold text-white tracking-wider">{telemetry.peakFreq} Hz</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Cpu className="size-3" style={{ color: themePalette.secondary }} />
              <span className="text-white/40">RATE:</span>
              <span className="font-bold text-white tracking-wider">{telemetry.sampleRateKhz} kHz</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-white/40 text-[9px]">PITCH:</span>
              <span className="text-emerald-400 font-bold">{telemetry.pitchNote}</span>
            </div>
          </div>

          {/* Right HUD Panel */}
          <div className="space-y-1.5 backdrop-blur-md bg-black/50 p-2.5 rounded-xl border border-white/10 text-right shadow-lg">
            <div className="flex items-center justify-end gap-1.5">
              <span className="font-bold text-white tracking-wider">{telemetry.decibels} dB</span>
              <span className="text-white/40">LEVEL:</span>
              <Zap className="size-3" style={{ color: themePalette.secondary }} />
            </div>
            <div className="flex items-center justify-end gap-1.5">
              <span className="font-bold text-emerald-400">
                {isSpeaking ? 'TRANSMITTING' : isActive ? 'LISTENING' : 'STANDBY'}
              </span>
              <Radio
                className={`size-3 ${
                  isSpeaking ? 'text-amber-400 animate-bounce' : 'text-emerald-400 animate-pulse'
                }`}
              />
            </div>
            <div className="flex items-center justify-end gap-1.5">
              <span className="text-[9px] text-white/40">RESONANCE:</span>
              <span className="font-bold" style={{ color: themePalette.primary }}>
                {Math.round(telemetry.bassEnergy)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Spectrum Equalizer & Sensitivity Control Bar */}
      <div className="w-full z-10 flex flex-wrap items-center justify-between gap-3 pt-2.5 border-t border-white/10 text-[10px]">
        {/* Tri-Band Equalizer */}
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="flex items-center gap-1.5">
            <span className="text-white/40 font-bold">BASS</span>
            <div className="w-12 sm:w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-75 rounded-full"
                style={{
                  width: `${telemetry.bassEnergy}%`,
                  backgroundColor: themePalette.primary,
                  boxShadow: `0 0 8px ${themePalette.primary}`,
                }}
              />
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-white/40 font-bold">MID</span>
            <div className="w-12 sm:w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-75 rounded-full"
                style={{
                  width: `${telemetry.midEnergy}%`,
                  backgroundColor: themePalette.secondary,
                  boxShadow: `0 0 8px ${themePalette.secondary}`,
                }}
              />
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-white/40 font-bold">TREBLE</span>
            <div className="w-12 sm:w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-75 rounded-full"
                style={{
                  width: `${telemetry.trebleEnergy}%`,
                  backgroundColor: '#ffffff',
                  boxShadow: '0 0 8px #ffffff',
                }}
              />
            </div>
          </div>
        </div>

        {/* Audio Sensitivity & Status */}
        <div className="flex items-center gap-3 text-white/60">
          <div className="flex items-center gap-1.5">
            <span className="text-white/40">GAIN:</span>
            <button
              type="button"
              onClick={() =>
                setSensitivity((prev) => (prev >= 2.0 ? 0.75 : parseFloat((prev + 0.25).toFixed(2))))
              }
              className="px-1.5 py-0.5 rounded bg-white/10 hover:bg-white/20 text-white font-bold transition-all"
              title="Cycle Reactivity Gain Multiplier"
            >
              {sensitivity}x
            </button>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold text-emerald-400">HOLOGRAPHIC QUANTUM LIVE</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default JarvisUltronVoiceOrb;
