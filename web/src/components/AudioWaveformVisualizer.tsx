import React, { useEffect, useRef } from 'react';
import { Activity, Mic, Volume2 } from 'lucide-react';

interface AudioWaveformVisualizerProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  isAgentSpeaking: boolean;
  isMuted: boolean;
  accentColor?: string;
  realVolumeMeter?: number;
}

export const AudioWaveformVisualizer: React.FC<AudioWaveformVisualizerProps> = ({
  analyser,
  isActive,
  isAgentSpeaking,
  isMuted,
  accentColor = '#00f0ff',
  realVolumeMeter = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      ctx.fillStyle = '#030712';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = '#071526';
      ctx.lineWidth = 1;
      const centerY = height / 2;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      if (!isActive) {
        ctx.strokeStyle = '#00f0ff40';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x = 0; x < width; x += 4) {
          const y = centerY + Math.sin(x * 0.03 + phase) * 2;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        phase += 0.05;
        animId = requestAnimationFrame(render);
        return;
      }

      let freqData = new Uint8Array(32);
      let hasRealAudio = false;

      if (analyser && !isMuted) {
        const bufferLength = analyser.frequencyBinCount;
        const data = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(data);
        if (data.some((val) => val > 0)) {
          freqData = data.slice(0, 32);
          hasRealAudio = true;
        }
      }

      const numBars = 32;
      const barGap = 3;
      const barWidth = Math.max(2, (width - barGap * (numBars + 1)) / numBars);

      const gradient = ctx.createLinearGradient(0, height, 0, 0);
      gradient.addColorStop(0, '#071526');
      gradient.addColorStop(0.5, accentColor);
      gradient.addColorStop(1, isAgentSpeaking ? '#ffb700' : '#00f0ff');

      for (let i = 0; i < numBars; i++) {
        let barHeight = 4;

        if (hasRealAudio && !isMuted) {
          const rawVal = freqData[i % freqData.length] || 0;
          barHeight = Math.max(4, (rawVal / 255) * (height * 0.85));
        } else if (isAgentSpeaking) {
          const wave = Math.sin(i * 0.4 + phase * 2.5) * 0.5 + 0.5;
          const noise = Math.cos(i * 0.8 + phase * 1.5) * 0.3 + 0.3;
          barHeight = Math.max(6, (wave + noise) * (height * 0.75));
        } else if (realVolumeMeter > 0 && !isMuted) {
          const mod = Math.sin(i * 0.5 + phase * 2) * 0.3 + 0.7;
          barHeight = Math.max(4, (realVolumeMeter / 100) * (height * 0.8) * mod);
        } else {
          const idleWave = Math.sin(i * 0.3 + phase) * 0.5 + 0.5;
          barHeight = 4 + idleWave * 8;
        }

        const x = barGap + i * (barWidth + barGap);
        const y = centerY - barHeight / 2;

        ctx.fillStyle = gradient;
        ctx.beginPath();
        if (typeof (ctx as any).roundRect === 'function') {
          (ctx as any).roundRect(x, y, barWidth, barHeight, 2);
        } else {
          ctx.rect(x, y, barWidth, barHeight);
        }
        ctx.fill();
      }

      phase += 0.08;
      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [analyser, isActive, isAgentSpeaking, isMuted, accentColor, realVolumeMeter]);

  return (
    <div className="bg-[#030712] border border-[#00f0ff]/30 p-3 rounded-md space-y-2 relative overflow-hidden">
      <div className="flex items-center justify-between text-[11px] font-mono text-[#80f7ff]/70">
        <div className="flex items-center gap-1.5">
          <Activity className={`w-3.5 h-3.5 ${isActive ? 'text-[#00f0ff] animate-pulse' : 'text-[#80f7ff]/50'}`} />
          <span className="text-[#e5e2e1] font-semibold">WebAudio API Frequency Stream</span>
        </div>
        <div className="flex items-center gap-2">
          {isAgentSpeaking ? (
            <span className="px-2 py-0.5 bg-[#00f0ff]/20 border border-[#00f0ff] text-[#00f0ff] text-[10px] font-bold rounded flex items-center gap-1 shadow-[0_0_8px_rgba(0,240,255,0.4)]">
              <Volume2 className="w-3 h-3 text-[#00f0ff] animate-bounce" /> SPEAKING
            </span>
          ) : isActive && !isMuted ? (
            <span className="px-2 py-0.5 bg-emerald-950/80 border border-emerald-500 text-emerald-300 text-[10px] font-bold rounded flex items-center gap-1">
              <Mic className="w-3 h-3 text-emerald-400 animate-pulse" /> LISTENING (MIC)
            </span>
          ) : (
            <span className="text-[#80f7ff]/50">STANDBY</span>
          )}
        </div>
      </div>

      <div className="w-full h-16 bg-[#07172b] border border-[#00f0ff]/20 rounded overflow-hidden flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={380}
          height={64}
          className="w-full h-full block"
        />
      </div>

      <div className="flex items-center justify-between text-[10px] font-mono text-[#80f7ff]/70">
        <span>FFT Size: 64 • 32 Band Spectrum</span>
        <span>Peak Level: <strong className="text-[#00f0ff]">{isActive && !isMuted ? `${realVolumeMeter}%` : '0%'}</strong></span>
      </div>
    </div>
  );
};
