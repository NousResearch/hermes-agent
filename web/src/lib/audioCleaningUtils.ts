/**
 * Audio Cleaning & Enhancement DSP Utilities
 * 
 * Provides real-time browser-side audio cleaning for microphone input:
 * 1. Highpass BiquadFilter (85 Hz cutoff): eliminates HVAC rumble, desk vibrations,
 *    and low-frequency proximity pops.
 * 2. Twin Notch Filters (50 Hz & 60 Hz): removes AC electrical mains hum.
 * 3. DynamicsCompressorNode: normalizes vocal dynamic range so quiet whispering
 *    is boosted and loud speech is protected from distortion.
 * 4. Voice Energy Gate: calculates RMS energy to prevent background noise
 *    from triggering false speech recognition events.
 * 5. AudioSpeechRecorder: records audio chunks and converts to base64 DataURL
 *    for backend transcription fallback (/api/audio/transcribe).
 */

import { authedFetch } from "./api";

export interface CleanAudioPipeline {
  audioContext: AudioContext;
  cleanStream: MediaStream;
  analyser: AnalyserNode;
  destinationNode: MediaStreamAudioDestinationNode;
  cleanup: () => void;
}

/**
 * Creates a real-time DSP audio cleaning graph for a MediaStream.
 */
export function createCleanAudioPipeline(sourceStream: MediaStream): CleanAudioPipeline | null {
  if (typeof window === "undefined") return null;

  const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext;
  if (!AudioContextCtor) return null;

  try {
    const audioContext = new AudioContextCtor();
    if (audioContext.state === "suspended") {
      void audioContext.resume();
    }

    const sourceNode = audioContext.createMediaStreamSource(sourceStream);

    // 1. Highpass filter: cuts low-end rumble below 85Hz
    const highpassFilter = audioContext.createBiquadFilter();
    highpassFilter.type = "highpass";
    highpassFilter.frequency.setValueAtTime(85, audioContext.currentTime);
    highpassFilter.Q.setValueAtTime(0.707, audioContext.currentTime);

    // 2. Notch filter at 50Hz (European / Asian mains hum)
    const notch50 = audioContext.createBiquadFilter();
    notch50.type = "notch";
    notch50.frequency.setValueAtTime(50, audioContext.currentTime);
    notch50.Q.setValueAtTime(4.0, audioContext.currentTime);

    // 3. Notch filter at 60Hz (American mains hum)
    const notch60 = audioContext.createBiquadFilter();
    notch60.type = "notch";
    notch60.frequency.setValueAtTime(60, audioContext.currentTime);
    notch60.Q.setValueAtTime(4.0, audioContext.currentTime);

    // 4. Dynamics compressor: levels speech volume and suppresses clipping
    const compressor = audioContext.createDynamicsCompressor();
    compressor.threshold.setValueAtTime(-24, audioContext.currentTime);
    compressor.knee.setValueAtTime(30, audioContext.currentTime);
    compressor.ratio.setValueAtTime(12, audioContext.currentTime);
    compressor.attack.setValueAtTime(0.003, audioContext.currentTime);
    compressor.release.setValueAtTime(0.25, audioContext.currentTime);

    // 5. Analyser node for real-time waveform and energy metrics
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;

    // 6. MediaStreamDestination for downstream capture / MediaRecorder
    const destinationNode = audioContext.createMediaStreamDestination();

    // Connect audio graph
    sourceNode.connect(highpassFilter);
    highpassFilter.connect(notch50);
    notch50.connect(notch60);
    notch60.connect(compressor);
    compressor.connect(analyser);
    compressor.connect(destinationNode);

    const cleanup = () => {
      try {
        sourceNode.disconnect();
        highpassFilter.disconnect();
        notch50.disconnect();
        notch60.disconnect();
        compressor.disconnect();
        analyser.disconnect();
        destinationNode.disconnect();
        if (audioContext.state !== "closed") {
          void audioContext.close();
        }
      } catch (err) {
        console.warn("[cleanAudio] Cleanup notice:", err);
      }
    };

    return {
      audioContext,
      cleanStream: destinationNode.stream,
      analyser,
      destinationNode,
      cleanup,
    };
  } catch (err) {
    console.warn("[cleanAudio] Failed initializing DSP audio graph:", err);
    return null;
  }
}

/**
 * Calculates current voice RMS volume level (0 to 100) from an AnalyserNode.
 */
export function calculateVoiceVolume(analyser: AnalyserNode | null): number {
  if (!analyser) return 0;
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(buffer);
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) {
    sum += buffer[i];
  }
  const avg = sum / buffer.length;
  return Math.min(100, Math.round((avg / 128) * 100));
}

/**
 * AudioSpeechRecorder handles microphone recording and transcribing via Hermes backend.
 */
export class AudioSpeechRecorder {
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private isRecording = false;
  private stream: MediaStream;

  constructor(stream: MediaStream) {
    this.stream = stream;
  }

  public start(): boolean {
    if (this.isRecording) return true;
    this.recordedChunks = [];

    const mimeType = this.getSupportedMimeType();
    try {
      this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          this.recordedChunks.push(e.data);
        }
      };
      this.mediaRecorder.start(250);
      this.isRecording = true;
      return true;
    } catch (err) {
      console.warn("[AudioSpeechRecorder] Failed starting MediaRecorder:", err);
      return false;
    }
  }

  public async stopAndTranscribe(): Promise<string> {
    if (!this.mediaRecorder || !this.isRecording) {
      return "";
    }

    return new Promise<string>((resolve) => {
      if (!this.mediaRecorder) {
        resolve("");
        return;
      }

      this.mediaRecorder.onstop = async () => {
        this.isRecording = false;
        if (this.recordedChunks.length === 0) {
          resolve("");
          return;
        }

        const mime = this.mediaRecorder?.mimeType || "audio/webm";
        const blob = new Blob(this.recordedChunks, { type: mime });
        this.recordedChunks = [];

        try {
          const dataUrl = await this.blobToDataUrl(blob);
          const response = await authedFetch("/api/audio/transcribe", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              data_url: dataUrl,
              mime_type: mime,
            }),
          });

          if (response.ok) {
            const data = (await response.json()) as { ok?: boolean; transcript?: string };
            resolve(String(data.transcript || "").trim());
          } else {
            resolve("");
          }
        } catch (err) {
          console.warn("[AudioSpeechRecorder] Transcribe request failed:", err);
          resolve("");
        }
      };

      try {
        this.mediaRecorder.stop();
      } catch {
        this.isRecording = false;
        resolve("");
      }
    });
  }

  public abort(): void {
    if (this.mediaRecorder && this.isRecording) {
      try {
        this.mediaRecorder.ondataavailable = null;
        this.mediaRecorder.onstop = null;
        this.mediaRecorder.stop();
      } catch {
        // ignore
      }
    }
    this.isRecording = false;
    this.recordedChunks = [];
  }

  public get recording(): boolean {
    return this.isRecording;
  }

  private getSupportedMimeType(): string {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const mime of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(mime)) {
        return mime;
      }
    }
    return "";
  }

  private blobToDataUrl(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
}
