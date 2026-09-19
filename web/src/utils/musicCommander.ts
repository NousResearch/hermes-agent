import type { MusicTrack } from '@/types/jarvis';

export interface MusicCommandAction {
  action: 'play' | 'pause' | 'resume' | 'next' | 'prev';
  query?: string;
  isYouTube?: boolean;
}

export function parseMusicCommand(raw: string): MusicCommandAction | null {
  if (!raw) return null;
  let t = raw.trim();

  // Strip leading polite prefixes, wake words, and conversational openers iteratively
  let prev = '';
  while (prev !== t) {
    prev = t;
    t = t.replace(/^(?:hey|hi|hello|ok|okay|please|can you|could you|would you|i want you to)\s+/i, '');
    t = t.replace(/^(?:يا\s*)?(?:جارفيس|جوين|jarvis|gwen|bot|sentinel)\s*[,،:\s-]*/i, '');
    t = t.replace(/^(?:من فضلك|لو سمحت|بالله عليك|ممكن|ياريت|بقولك|عاوزك|عايزك|اسمع|يلا)\s+/i, '');
    t = t.trim();
  }

  // Pause / Stop
  if (
    /^(?:وقف|وقفي|وقفلي|اقفل|اقفلي|اسكت|كفاية|stop|pause)\s*(?:الموسيقى|الميوزك|الميوزيك|الاغنية|الأغنية|الأغاني|الاغاني|music|song|the music)?$/i.test(t) ||
    /^(?:وقف|وقفي|اقفل|اقفلي|stop|pause)\s+(?:الموسيقى|الميوزك|الاغنية|الأغنية|music|song)/i.test(t)
  ) {
    return { action: 'pause' };
  }

  // Resume
  if (
    /^(?:كمل|كمّل|شغل تاني|استمر|resume|unpause)\s*(?:الموسيقى|الميوزك|الاغنية|الأغنية|music|song)?$/i.test(t)
  ) {
    return { action: 'resume' };
  }

  // Next
  if (
    /^(?:اللي بعدها|الأغنية اللي بعدها|الاغنية اللي بعدها|هات اللي بعدها|next|skip|next song|next track)$/i.test(t)
  ) {
    return { action: 'next' };
  }

  // Prev
  if (
    /^(?:اللي قبلها|الأغنية اللي قبلها|الاغنية اللي قبلها|هات اللي قبلها|prev|previous|previous song)$/i.test(t)
  ) {
    return { action: 'prev' };
  }

  // Check for YouTube intent
  const hasYouTubeKeyword =
    /(?:from|on|in|via)\s+youtube/i.test(t) ||
    /youtube\s+(?:video|music|song|track)?/i.test(t) ||
    /(?:من|على|في|عبر)\s+(?:اليوتيوب|يوتيوب)/i.test(t) ||
    /(?:اليوتيوب|يوتيوب)/i.test(t);

  // Play: شغل كذا / play ... / سمعني كذا / عايز اسمع كذا / افتح يوتيوب وشغل كذا
  const playMatch = t.match(/^(?:شغل|شغلي|شغللي|شغللنا|عايز اسمع|عايز أسمع|عاوز اسمع|عاوز أسمع|نفسي اسمع|سمعني|افتح|play|start|open)\s+(.+)$/i);
  if (playMatch && playMatch[1]) {
    let q = playMatch[1].trim();

    // Strip "يوتيوب وشغل" / "youtube and play"
    q = q.replace(/^(?:اليوتيوب|يوتيوب|youtube)\s+(?:و\s*)?(?:شغل|شغلي|play|start)\s+/i, '').trim();

    q = q.replace(/^(?:اغنية|أغنية|تراك|موسيقى|ميوزك|ميوزيك|أغاني|اغاني|فيديو|video|song|music|the song|the track|some music|some songs)\s+/i, '').trim();
    q = q.replace(/\s+(?:اغنية|أغنية|تراك|موسيقى|ميوزك|ميوزيك|أغاني|اغاني|فيديو|video|song|music|the song|the track|soundtrack|theme)$/i, '').trim();

    // Strip "from youtube" / "من يوتيوب" from query body
    q = q.replace(/\s*(?:from|on|via)\s+youtube\s*/gi, ' ').trim();
    q = q.replace(/\s*(?:من|على|في|عبر)\s+(?:اليوتيوب|يوتيوب)\s*/gi, ' ').trim();
    q = q.replace(/\s*(?:اليوتيوب|يوتيوب|youtube)\s*/gi, ' ').trim();

    return { action: 'play', query: q, isYouTube: hasYouTubeKeyword };
  }

  // General play intent if just "play" or "شغل"
  if (/^(?:شغل|play|play music|شغل موسيقى|شغل ميوزك)$/i.test(t)) {
    return { action: 'play', query: '', isYouTube: false };
  }

  return null;
}

export function findBestTrackIndex(query: string, trackList: MusicTrack[]): number {
  if (!trackList || trackList.length === 0) return 0;
  const q = query.toLowerCase().trim();
  if (!q) return 0;

  // 1. Direct substring match on title or artist
  for (let i = 0; i < trackList.length; i++) {
    const t = trackList[i];
    const title = t.title.toLowerCase();
    const artist = t.artist.toLowerCase();
    if (title.includes(q) || artist.includes(q)) {
      return i;
    }
  }

  // 2. Keyword-based matching
  const hamzaKeywords = ['حمزة', 'حمزه', 'نمرة', 'نمره', 'namira', 'hamza'];
  const fadyKeywords = ['فاضي', 'شوية', 'شويه', 'fady', 'shewaya'];
  const dariKeywords = ['داري', 'دالي', 'قلبي', 'dari', 'qalby', 'alby'];
  const reyahKeywords = ['رياح', 'الحياة', 'الحياه', 'reyah', 'hayah'];
  const waollakKeywords = ['واقولك', 'واقول لك', 'إيه', 'ايه', 'ollak'];
  const insanKeywords = ['انسان', 'إنسان', 'باب الله', 'insan'];
  
  const ironManKeywords = ['ايرون', 'أيرون', 'مان', 'iron', 'man', 'جارفيس', 'jarvis', 'ستارك', 'stark', 'طوني'];
  const backInBlackKeywords = ['بلاك', 'black', 'acdc', 'ac/dc', 'باك'];
  const shootThrillKeywords = ['shoot', 'thrill', 'شوت'];
  const flightKeywords = ['flight', 'طيران', 'بدلة', 'بدله', 'suit'];

  const billieKeywords = ['بيلي', 'ايليش', 'إيليش', 'بيللي', 'billie', 'eilish'];
  const badGuyKeywords = ['bad', 'guy', 'باد', 'جاي'];
  const lovelyKeywords = ['lovely', 'لافلي', 'لوفلي'];
  const birdsKeywords = ['birds', 'feather', 'بيردز'];
  const oceanKeywords = ['ocean', 'eyes', 'اوشن'];

  // Specific song checks
  if (fadyKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.includes('فاضي') || t.title.toLowerCase().includes('fady'));
    if (idx !== -1) return idx;
  }
  if (dariKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.includes('داري') || t.title.toLowerCase().includes('dari'));
    if (idx !== -1) return idx;
  }
  if (reyahKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.includes('رياح') || t.title.toLowerCase().includes('reyah'));
    if (idx !== -1) return idx;
  }
  if (waollakKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.includes('واقولك') || t.title.toLowerCase().includes('ollak'));
    if (idx !== -1) return idx;
  }
  if (insanKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.includes('إنسان') || t.title.includes('باب الله'));
    if (idx !== -1) return idx;
  }
  if (hamzaKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.artist.includes('حمزة') || t.artist.toLowerCase().includes('namira'));
    if (idx !== -1) return idx;
  }

  if (backInBlackKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('black'));
    if (idx !== -1) return idx;
  }
  if (shootThrillKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('shoot'));
    if (idx !== -1) return idx;
  }
  if (flightKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('flight'));
    if (idx !== -1) return idx;
  }
  if (ironManKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('flight') || t.title.toLowerCase().includes('iron man') || t.title.toLowerCase().includes('black'));
    if (idx !== -1) return idx;
  }

  if (badGuyKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('bad guy'));
    if (idx !== -1) return idx;
  }
  if (lovelyKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('lovely'));
    if (idx !== -1) return idx;
  }
  if (birdsKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('birds'));
    if (idx !== -1) return idx;
  }
  if (oceanKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.title.toLowerCase().includes('ocean'));
    if (idx !== -1) return idx;
  }
  if (billieKeywords.some(k => q.includes(k))) {
    const idx = trackList.findIndex(t => t.artist.toLowerCase().includes('billie'));
    if (idx !== -1) return idx;
  }

  return 0;
}
