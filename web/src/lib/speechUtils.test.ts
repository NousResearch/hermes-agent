import { describe, expect, it, vi } from 'vitest';
import {
  cleanSpokenText,
  extractNextSpokenSentence,
  pickArabicVoice,
  pickEnglishVoice,
  PipelinedAudioQueue,
  splitTextIntoSentences,
} from './speechUtils';

describe('speechUtils - Streaming Sentence Pipeline', () => {
  it('cleanSpokenText removes completed and active reasoning think tags', () => {
    const textWithCompletedThink = '<think>Checking system telemetry...</think>All systems are online.';
    expect(cleanSpokenText(textWithCompletedThink)).toBe('All systems are online.');

    const textWithActiveThink = '<think>Model is currently contemplating';
    expect(cleanSpokenText(textWithActiveThink)).toBe('');

    const textWithTailThink = 'First answer.<think>Evaluating follow-up';
    expect(cleanSpokenText(textWithTailThink)).toBe('First answer.');
  });

  it('extractNextSpokenSentence extracts sentences sequentially with English and Arabic terminators', () => {
    const textEn = 'Hello sir! All systems operational. How can I assist you?';
    const first = extractNextSpokenSentence(textEn, 0);
    expect(first?.sentence).toBe('Hello sir!');
    expect(first?.nextIndex).toBe(10);


    const second = extractNextSpokenSentence(textEn, first!.nextIndex);
    expect(second?.sentence).toBe('All systems operational.');

    const third = extractNextSpokenSentence(textEn, second!.nextIndex);
    expect(third?.sentence).toBe('How can I assist you?');

    const fourth = extractNextSpokenSentence(textEn, third!.nextIndex);
    expect(fourth).toBeNull();

    // Arabic punctuation: . ! ؟ ؛
    const textAr = 'أهلاً بك يا فندم! كيف أساعدك اليوم؟ جارفيس مستعد.';
    const arFirst = extractNextSpokenSentence(textAr, 0);
    expect(arFirst?.sentence).toBe('أهلاً بك يا فندم!');

    const arSecond = extractNextSpokenSentence(textAr, arFirst!.nextIndex);
    expect(arSecond?.sentence).toBe('كيف أساعدك اليوم؟');

    const arThird = extractNextSpokenSentence(textAr, arSecond!.nextIndex);
    expect(arThird?.sentence).toBe('جارفيس مستعد.');

    // Long unpunctuated stream should split at word boundary for zero latency
    const unpunctuated = 'This is a long continuous stream of words without any punctuation to test the latency split';
    const split = extractNextSpokenSentence(unpunctuated, 0);
    expect(split).not.toBeNull();
    expect(split?.sentence.length).toBeGreaterThanOrEqual(18);
    expect(unpunctuated.startsWith(split!.sentence)).toBe(true);
  });

  it('splitTextIntoSentences preserves existing splitting contract', () => {
    const sentences = splitTextIntoSentences('Sentence one. Sentence two! Sentence three?');
    expect(sentences).toEqual(['Sentence one.', 'Sentence two!', 'Sentence three?']);
  });

  it('PipelinedAudioQueue executes tasks in order and notifies speaking change', async () => {
    const events: string[] = [];
    const onSpeaking = vi.fn();
    const queue = new PipelinedAudioQueue(onSpeaking);

    queue.enqueue(async () => {
      await new Promise((r) => setTimeout(r, 20));
      events.push('sentence_1');
    });

    queue.enqueue(async () => {
      await new Promise((r) => setTimeout(r, 10));
      events.push('sentence_2');
    });

    await queue.waitUntilDone();

    expect(events).toEqual(['sentence_1', 'sentence_2']);
    expect(onSpeaking).toHaveBeenCalledWith(true);
    expect(onSpeaking).toHaveBeenCalledWith(false);
  });

  it('PipelinedAudioQueue supports clean abort on barge-in / interruption', async () => {
    const events: string[] = [];
    const queue = new PipelinedAudioQueue();

    queue.enqueue(async (signal) => {
      await new Promise((r) => setTimeout(r, 50));
      if (!signal.aborted) events.push('task_1');
    });

    queue.enqueue(async (signal) => {
      await new Promise((r) => setTimeout(r, 50));
      if (!signal.aborted) events.push('task_2');
    });

    // Abort after small delay
    setTimeout(() => {
      queue.abort();
    }, 10);

    await queue.waitUntilDone();

    // task_2 should never have executed
    expect(events).not.toContain('task_2');
  });

  it('extracts early conversational clauses on commas in English and Arabic for instant speech start', () => {
    const clauseEn = 'Understood sir, I will check that for you right now.';
    const firstEn = extractNextSpokenSentence(clauseEn, 0);
    expect(firstEn?.sentence).toBe('Understood sir,');
    expect(firstEn?.nextIndex).toBe(15);

    const secondEn = extractNextSpokenSentence(clauseEn, firstEn!.nextIndex);
    expect(secondEn?.sentence).toBe('I will check that for you right now.');

    const clauseAr = 'أهلاً يا فندم، جاري تجهيز كافة البيانات المطلوبة فوراً.';
    const firstAr = extractNextSpokenSentence(clauseAr, 0);
    expect(firstAr?.sentence).toBe('أهلاً يا فندم،');
    expect(firstAr?.nextIndex).toBe(14);

    const secondAr = extractNextSpokenSentence(clauseAr, firstAr!.nextIndex);
    expect(secondAr?.sentence).toBe('جاري تجهيز كافة البيانات المطلوبة فوراً.');

    // 1-word greeting with comma emits immediately on turn start (TTFA <100ms)
    const oneWordEn = 'Yes, all parameters are nominal.';
    const oneWordEnRes = extractNextSpokenSentence(oneWordEn, 0);
    expect(oneWordEnRes?.sentence).toBe('Yes,');
    expect(oneWordEnRes?.nextIndex).toBe(4);

    const oneWordAr = 'تمام، جاري الفحص فوراً.';
    const oneWordArRes = extractNextSpokenSentence(oneWordAr, 0);
    expect(oneWordArRes?.sentence).toBe('تمام،');
    expect(oneWordArRes?.nextIndex).toBe(5);

    // Initial 3-4 word phrase emits before punctuation to start speech immediately
    const earlyWords = 'I am standing by and ready to assist you with everything.';
    const earlyRes = extractNextSpokenSentence(earlyWords, 0);
    expect(earlyRes).not.toBeNull();
    expect(earlyRes?.sentence).toBe('I am standing by and ready to');
    expect(earlyRes?.nextIndex).toBe(30);
  });

  it('selects appropriate male voices for Jarvis and female voices for Gwen', () => {
    const mockVoices = [
      { name: 'Microsoft George - English (United Kingdom)', lang: 'en-GB' } as SpeechSynthesisVoice,
      { name: 'Microsoft Zira - English (United States)', lang: 'en-US' } as SpeechSynthesisVoice,
      { name: 'Microsoft Shakir - Arabic (Egypt)', lang: 'ar-EG' } as SpeechSynthesisVoice,
      { name: 'Microsoft Salma - Arabic (Egypt)', lang: 'ar-EG' } as SpeechSynthesisVoice,
    ];

    const jarvisEn = pickEnglishVoice(mockVoices, 'jarvis');
    expect(jarvisEn?.name).toContain('George');

    const gwenEn = pickEnglishVoice(mockVoices, 'gwen');
    expect(gwenEn?.name).toContain('Zira');

    const jarvisAr = pickArabicVoice(mockVoices, 'jarvis');
    expect(jarvisAr?.name).toContain('Shakir');

    const gwenAr = pickArabicVoice(mockVoices, 'gwen');
    expect(gwenAr?.name).toContain('Salma');
  });
});
