export function chooseRecordingMimeType(
  supportedTypes: readonly string[],
): string | undefined {
  return supportedTypes.find((type) => type.toLowerCase().includes("audio/webm"))
    ?? supportedTypes.find((type) => type.toLowerCase().includes("audio/ogg"))
    ?? supportedTypes.find((type) => type.toLowerCase().startsWith("audio/"));
}

export function appendVoiceTranscript(draft: string, transcript: string): string {
  const next = transcript.trim();
  if (!next) return draft;
  const current = draft.trimEnd();
  return current ? `${current} ${next}` : next;
}

export function canRecordVoice(
  mediaDevices: Pick<MediaDevices, "getUserMedia"> | undefined,
  mediaRecorder: Pick<typeof MediaRecorder, "isTypeSupported"> | undefined,
): boolean {
  return Boolean(mediaDevices?.getUserMedia && mediaRecorder);
}
