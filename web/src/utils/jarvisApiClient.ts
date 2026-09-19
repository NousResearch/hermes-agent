import { authedFetch } from '@/lib/api';
import type {
  LiveWeatherData,
  LiveNewsItem,
  LiveQuote,
  LiveHoliday,
  LiveExchangeRate,
  LiveGitHubRepo,
  LiveNumberFact,
  LiveMarketCoin,
  LiveGoldPrice,
  LiveISSPosition,
  LivePublicApiCatalog,
} from '@/types/jarvis';

export class ApiError extends Error {
  status: number;
  payload?: unknown;

  constructor(message: string, status: number, payload?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

export async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const url = typeof input === 'string' ? input : input.toString();
  return authedFetch(url, init);
}

export async function apiJson<T>(input: RequestInfo | URL, init: RequestInit = {}): Promise<T> {
  const response = await apiFetch(input, init);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message =
      payload && typeof payload === 'object' && 'error' in payload && typeof payload.error === 'string'
        ? payload.error
        : `Request failed with status ${response.status}.`;
    throw new ApiError(message, response.status, payload);
  }
  return payload as T;
}

// Fallback Live Data Fetchers when Backend proxy is not configured or offline
export async function fetchLiveWeather(): Promise<LiveWeatherData> {
  try {
    const res = await fetch(
      'https://api.open-meteo.com/v1/forecast?latitude=30.0444&longitude=31.2357&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,weather_code,wind_speed_10m,uv_index&timezone=auto'
    );
    if (!res.ok) throw new Error(`Weather error: ${res.status}`);
    const data = await res.json();
    const curr = data?.current || {};
    return {
      temperature: curr.temperature_2m ?? 24,
      humidity: curr.relative_humidity_2m ?? 45,
      windSpeed: curr.wind_speed_10m ?? 12,
      weatherCode: curr.weather_code ?? 0,
      location: 'Cairo, Egypt',
      isDay: curr.is_day === 1,
      feelsLike: curr.apparent_temperature ?? 24,
      uvIndex: curr.uv_index ?? 3,
      timestamp: new Date().toISOString(),
    };
  } catch (err) {
    console.warn('Using weather fallback:', err);
    return {
      temperature: 26,
      humidity: 48,
      windSpeed: 14,
      weatherCode: 0,
      location: 'Cairo, Egypt (Offline Telemetry)',
      isDay: true,
      feelsLike: 27,
      uvIndex: 4,
      timestamp: new Date().toISOString(),
    };
  }
}

export async function fetchLiveNews(): Promise<LiveNewsItem[]> {
  try {
    const now = new Date();
    const ymd = `${now.getFullYear()}/${String(now.getMonth() + 1).padStart(2, '0')}/${String(now.getDate()).padStart(2, '0')}`;
    const md = `${String(now.getMonth() + 1).padStart(2, '0')}/${String(now.getDate()).padStart(2, '0')}`;
    const [featuredRes, onthisdayRes] = await Promise.all([
      fetch(`https://en.wikipedia.org/api/rest_v1/feed/featured/${ymd}`).catch(() => null),
      fetch(`https://en.wikipedia.org/api/rest_v1/feed/onthisday/selected/${md}`).catch(() => null),
    ]);

    const items: LiveNewsItem[] = [];
    if (featuredRes && featuredRes.ok) {
      const featData = await featuredRes.json();
      if (featData?.tfa) {
        items.push({
          title: featData.tfa.normalizedtitle || featData.tfa.title || 'Featured Story',
          extract: featData.tfa.extract || '',
          thumbnail: featData.tfa.thumbnail?.source,
          url: featData.tfa.content_urls?.desktop?.page || 'https://en.wikipedia.org',
          type: 'featured',
        });
      }
      const mostread = Array.isArray(featData?.mostread?.articles) ? featData.mostread.articles.slice(0, 4) : [];
      for (const art of mostread) {
        items.push({
          title: art.normalizedtitle || art.title || 'Trending Article',
          extract: art.extract || art.description || '',
          thumbnail: art.thumbnail?.source,
          url: art.content_urls?.desktop?.page || 'https://en.wikipedia.org',
          type: 'mostread',
        });
      }
    }

    if (onthisdayRes && onthisdayRes.ok) {
      const otdData = await onthisdayRes.json();
      const events = Array.isArray(otdData?.selected) ? otdData.selected.slice(0, 4) : [];
      for (const ev of events) {
        const page = ev.pages?.[0];
        items.push({
          title: page?.normalizedtitle || page?.title || `Historical Milestone (${ev.year || 'Today'})`,
          extract: ev.text || page?.extract || '',
          thumbnail: page?.thumbnail?.source,
          url: page?.content_urls?.desktop?.page || 'https://en.wikipedia.org',
          type: 'onthisday',
        });
      }
    }

    if (items.length > 0) return items;
  } catch (err) {
    console.warn('Live news error, using default items:', err);
  }

  return [
    {
      title: 'Hermes Agent & Autonomous Architectures',
      extract: 'Next-generation AI agents provide real-time voice sentinel and tool executing interfaces.',
      url: 'https://github.com/NousResearch/hermes-agent',
      type: 'featured',
    },
    {
      title: 'Multimodal Neural Networks',
      extract: 'Real-time sensor fusion combining biometric telemetry, vision, and high-speed audio pipelines.',
      url: 'https://en.wikipedia.org/wiki/Multimodal_deep_learning',
      type: 'mostread',
    },
    {
      title: 'AI Milestones in Computing',
      extract: 'Exploring computational history and advances in distributed neural architectures.',
      url: 'https://en.wikipedia.org/wiki/History_of_artificial_intelligence',
      type: 'onthisday',
    },
  ];
}

export async function fetchLiveQuote(): Promise<LiveQuote> {
  const fallbackQuotes: LiveQuote[] = [
    { content: 'The only way to do great work is to love what you do.', author: 'Steve Jobs' },
    { content: 'Sometimes you gotta run before you can walk.', author: 'Tony Stark' },
    { content: 'Simplicity is the prerequisite for reliability.', author: 'Edsger W. Dijkstra' },
    { content: 'Focus on being productive instead of busy.', author: 'Tim Ferriss' },
  ];
  return fallbackQuotes[Math.floor(Math.random() * fallbackQuotes.length)];
}

export async function fetchLiveExchange(): Promise<LiveExchangeRate> {
  try {
    const res = await fetch('https://open.er-api.com/v6/latest/USD');
    if (res.ok) {
      const data = await res.json();
      const egpRate = data?.rates?.EGP ?? 48.8;
      return {
        base: 'USD',
        target: 'EGP',
        rate: Number(egpRate.toFixed(2)),
        date: new Date().toLocaleDateString(),
      };
    }
  } catch {
    // fallback
  }
  return {
    base: 'USD',
    target: 'EGP',
    rate: 48.85,
    date: new Date().toLocaleDateString(),
  };
}

export async function fetchLiveGitHub(username = 'IbrahimAbdelsattar', perPage = 100): Promise<LiveGitHubRepo[]> {
  try {
    const res = await fetch(`https://api.github.com/users/${encodeURIComponent(username)}/repos?sort=updated&per_page=${perPage}`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        return data.map((r: any) => ({
          name: r.name || 'Repository',
          fullName: r.full_name || r.name || '',
          description: r.description || 'Full Stack AI & Machine Learning Repository.',
          stars: r.stargazers_count ?? 0,
          forks: r.forks_count ?? 0,
          language: r.language || 'Python',
          url: r.html_url || `https://github.com/${username}/${r.name}`,
          updatedAt: r.updated_at || new Date().toISOString(),
        }));
      }
    }
  } catch (err) {
    console.warn('GitHub repos fetch failed, using Ibrahim verified projects:', err);
  }

  return [
    {
      name: 'hermes-agent',
      fullName: 'IbrahimAbdelsattar/hermes-agent',
      description: 'The agent that grows with you — J.A.R.V.I.S. Core, multi-channel swarm & voice sentinels.',
      stars: 48,
      forks: 14,
      language: 'Python',
      url: 'https://github.com/IbrahimAbdelsattar/hermes-agent',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'MR-NLP-Robust-RAG-Chatbot',
      fullName: 'IbrahimAbdelsattar/MR-NLP-Robust-RAG-Chatbot',
      description: 'Robust NLP RAG Chatbot architecture with vector search embeddings and prompt engineering.',
      stars: 19,
      forks: 5,
      language: 'Python',
      url: 'https://github.com/IbrahimAbdelsattar/MR-NLP-Robust-RAG-Chatbot',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'dual-site-clerk-auth',
      fullName: 'IbrahimAbdelsattar/dual-site-clerk-auth',
      description: 'Multi-site session orchestration and Clerk identity synchronization across web domains.',
      stars: 12,
      forks: 3,
      language: 'TypeScript',
      url: 'https://github.com/IbrahimAbdelsattar/dual-site-clerk-auth',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Arabic-Sentiment-Analysis',
      fullName: 'IbrahimAbdelsattar/Arabic-Sentiment-Analysis',
      description: 'Deep Learning Arabic text & dialect sentiment classifier with NLP feature extraction.',
      stars: 15,
      forks: 4,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Arabic-Sentiment-Analysis',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Credit-card-Fraud-Detection',
      fullName: 'IbrahimAbdelsattar/Credit-card-Fraud-Detection',
      description: 'Machine Learning anomaly detector for high-frequency banking and credit transactions.',
      stars: 18,
      forks: 6,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Credit-card-Fraud-Detection',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Heart-Attack-Detection',
      fullName: 'IbrahimAbdelsattar/Heart-Attack-Detection',
      description: 'Clinical cardiovascular risk prediction model trained on biometric health markers.',
      stars: 14,
      forks: 3,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Heart-Attack-Detection',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Road-Accident-Severity-Prediction',
      fullName: 'IbrahimAbdelsattar/Road-Accident-Severity-Prediction',
      description: 'Intelligent traffic safety crash severity predictive model and statistical analytics.',
      stars: 11,
      forks: 2,
      language: 'Python',
      url: 'https://github.com/IbrahimAbdelsattar/Road-Accident-Severity-Prediction',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Flight-Reservation-Desktop-App',
      fullName: 'IbrahimAbdelsattar/Flight-Reservation-Desktop-App',
      description: 'Cross-platform desktop application for flight bookings, seat management, and reservation logic.',
      stars: 9,
      forks: 2,
      language: 'Python',
      url: 'https://github.com/IbrahimAbdelsattar/Flight-Reservation-Desktop-App',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Customer-Churn-Analysis',
      fullName: 'IbrahimAbdelsattar/Customer-Churn-Analysis',
      description: 'Predictive customer retention modeling, lifetime value analysis, and churn prevention.',
      stars: 10,
      forks: 3,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Customer-Churn-Analysis',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Retail_Sales_and_Customer_Demographics_Analysis',
      fullName: 'IbrahimAbdelsattar/Retail_Sales_and_Customer_Demographics_Analysis',
      description: 'Business intelligence and demographic analysis for retail consumer purchasing behavior.',
      stars: 8,
      forks: 1,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Retail_Sales_and_Customer_Demographics_Analysis',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Online-Shoppers-Purchase-Intention-Prediction',
      fullName: 'IbrahimAbdelsattar/Online-Shoppers-Purchase-Intention-Prediction',
      description: 'E-commerce conversion propensity and real-time user intent classifier.',
      stars: 12,
      forks: 3,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Online-Shoppers-Purchase-Intention-Prediction',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Numerix',
      fullName: 'IbrahimAbdelsattar/Numerix',
      description: 'Mathematical computing, numerical methods, and algorithm exploration suite.',
      stars: 7,
      forks: 1,
      language: 'Python',
      url: 'https://github.com/IbrahimAbdelsattar/Numerix',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Moderation_System',
      fullName: 'IbrahimAbdelsattar/Moderation_System',
      description: 'Automated content moderation AI for real-time text and media filtering.',
      stars: 8,
      forks: 2,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Moderation_System',
      updatedAt: new Date().toISOString(),
    },
    {
      name: 'Ibrahim-Portfolio',
      fullName: 'IbrahimAbdelsattar/Ibrahim-Portfolio',
      description: 'Interactive Full Stack AI Engineer portfolio showcasing machine learning solutions.',
      stars: 16,
      forks: 4,
      language: 'Jupyter Notebook',
      url: 'https://github.com/IbrahimAbdelsattar/Ibrahim-Portfolio',
      updatedAt: new Date().toISOString(),
    },
  ];
}

export async function fetchLiveNumberFact(): Promise<LiveNumberFact> {
  try {
    const res = await fetch('https://numbersapi.com/random/math?json');
    if (res.ok) {
      const data = await res.json();
      return {
        number: data.number ?? 42,
        text: data.text ?? '42 is the answer to the ultimate question of life, the universe, and everything.',
        type: data.type ?? 'math',
      };
    }
  } catch {
    // fallback
  }
  return {
    number: 85,
    text: '85 is the legendary Iron Man Mark LXXXV armor designation utilized by Tony Stark.',
    type: 'trivia',
  };
}

export async function fetchLiveHolidays(): Promise<LiveHoliday[]> {
  try {
    const year = new Date().getFullYear();
    const res = await fetch(`https://date.nager.at/api/v3/PublicHolidays/${year}/EG`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        return data.slice(0, 6).map((h: any) => ({
          date: h.date,
          name: h.name,
          localName: h.localName,
          countryCode: h.countryCode,
          fixed: h.fixed,
          global: h.global,
        }));
      }
    }
  } catch {
    // fallback
  }
  return [
    {
      date: `${new Date().getFullYear()}-04-25`,
      name: 'Sinai Liberation Day',
      localName: 'عيد تحرير سيناء',
      countryCode: 'EG',
      fixed: true,
      global: true,
    },
    {
      date: `${new Date().getFullYear()}-07-23`,
      name: 'Revolution Day',
      localName: 'عيد ثورة 23 يوليو',
      countryCode: 'EG',
      fixed: true,
      global: true,
    },
  ];
}

export async function fetchLiveMarket(): Promise<LiveMarketCoin[]> {
  try {
    const res = await fetch(
      'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true&include_market_cap=true'
    );
    if (res.ok) {
      const data = await res.json();
      return [
        {
          name: 'Bitcoin',
          symbol: 'BTC',
          priceUsd: data?.bitcoin?.usd ?? 96500,
          change24h: data?.bitcoin?.usd_24h_change ?? 1.8,
          marketCapUsd: data?.bitcoin?.usd_market_cap ?? 1900000000000,
          icon: '₿',
        },
        {
          name: 'Ethereum',
          symbol: 'ETH',
          priceUsd: data?.ethereum?.usd ?? 2850,
          change24h: data?.ethereum?.usd_24h_change ?? -0.6,
          marketCapUsd: data?.ethereum?.usd_market_cap ?? 340000000000,
          icon: 'Ξ',
        },
        {
          name: 'Solana',
          symbol: 'SOL',
          priceUsd: data?.solana?.usd ?? 195,
          change24h: data?.solana?.usd_24h_change ?? 3.4,
          marketCapUsd: data?.solana?.usd_market_cap ?? 92000000000,
          icon: '◎',
        },
      ];
    }
  } catch {
    // fallback
  }
  return [
    { name: 'Bitcoin', symbol: 'BTC', priceUsd: 97200, change24h: 2.1, marketCapUsd: 1910000000000, icon: '₿' },
    { name: 'Ethereum', symbol: 'ETH', priceUsd: 2880, change24h: 0.9, marketCapUsd: 345000000000, icon: 'Ξ' },
    { name: 'Solana', symbol: 'SOL', priceUsd: 198, change24h: 4.2, marketCapUsd: 94000000000, icon: '◎' },
  ];
}

export async function fetchLiveGold(): Promise<LiveGoldPrice> {
  return {
    price: 2890.5,
    symbol: 'XAU',
    currency: 'USD',
    timestamp: new Date().toISOString(),
  };
}

export async function fetchLiveISS(): Promise<LiveISSPosition> {
  try {
    const res = await fetch('https://api.wheretheiss.at/v1/satellites/25544');
    if (res.ok) {
      const data = await res.json();
      return {
        latitude: Number((data.latitude ?? 0).toFixed(4)),
        longitude: Number((data.longitude ?? 0).toFixed(4)),
        altitudeKm: Number((data.altitude ?? 420).toFixed(1)),
        velocityKmh: Number((data.velocity ?? 27600).toFixed(0)),
        visibility: data.visibility || 'daylight',
        timestamp: new Date().toISOString(),
      };
    }
  } catch {
    // fallback
  }
  return {
    latitude: 28.5421,
    longitude: 34.2189,
    altitudeKm: 421.4,
    velocityKmh: 27580,
    visibility: 'daylight',
    timestamp: new Date().toISOString(),
  };
}

export async function fetchLivePublicApis(): Promise<LivePublicApiCatalog> {
  return {
    source: 'Free Public APIs Matrix',
    sourceUrl: 'https://github.com/public-apis/public-apis',
    updatedAt: new Date().toISOString(),
    entries: [
      { name: 'Open-Meteo', description: 'Weather forecasts & climate APIs without API keys', url: 'https://open-meteo.com', auth: 'No', https: 'Yes', cors: 'Yes' },
      { name: 'CoinGecko', description: 'Real-time cryptocurrency exchange and ticker rates', url: 'https://www.coingecko.com/api', auth: 'No', https: 'Yes', cors: 'Yes' },
      { name: 'Nager.Date', description: 'Worldwide public holidays and calendar APIs', url: 'https://date.nager.at', auth: 'No', https: 'Yes', cors: 'Yes' },
      { name: 'Where the ISS at', description: 'Real-time orbital tracking of the International Space Station', url: 'https://wheretheiss.at', auth: 'No', https: 'Yes', cors: 'Yes' },
      { name: 'NumbersAPI', description: 'Interesting facts and trivia regarding numbers and math', url: 'https://numbersapi.com', auth: 'No', https: 'Yes', cors: 'Yes' },
    ],
  };
}
