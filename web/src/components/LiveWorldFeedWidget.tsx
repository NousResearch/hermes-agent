import React, { useState, useEffect, useCallback } from 'react';
import {
  CloudSun,
  Cloud,
  CloudRain,
  CloudSnow,
  CloudLightning,
  Sun,
  Moon,
  Wind,
  Thermometer,
  Droplets,
  Globe,
  Newspaper,
  Star,
  GitBranch,
  RefreshCw,
  Quote,
  Calendar,
  ExternalLink,
  ArrowRightLeft,
  Sparkles,
  Coins,
  Satellite,
  Gem,
} from 'lucide-react';

import type {
  LiveWeatherData,
  LiveNewsItem,
  LiveQuote,
  LiveExchangeRate,
  LiveGitHubRepo,
  LiveNumberFact,
  LiveHoliday,
  LiveMarketCoin,
  LiveGoldPrice,
  LiveISSPosition,
  LivePublicApiCatalog,
} from '@/types/jarvis';
import { getFeed, setFeed, isFeedFresh } from '@/utils/jarvisLiveFeedCache';
import {
  fetchLiveWeather,
  fetchLiveNews,
  fetchLiveQuote,
  fetchLiveExchange,
  fetchLiveGitHub,
  fetchLiveNumberFact,
  fetchLiveHolidays,
  fetchLiveMarket,
  fetchLiveGold,
  fetchLiveISS,
  fetchLivePublicApis,
} from '@/utils/jarvisApiClient';

function getWeatherIcon(code: number, isDay: boolean) {
  if (code === 0) {
    return isDay ? (
      <Sun className="size-8 text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.6)]" />
    ) : (
      <Moon className="size-8 text-blue-300 drop-shadow-[0_0_8px_rgba(147,197,253,0.6)]" />
    );
  }
  if (code <= 3) return <CloudSun className="size-8 text-yellow-300 drop-shadow-[0_0_8px_rgba(253,224,71,0.6)]" />;
  if (code <= 48) return <Cloud className="size-8 text-gray-400 drop-shadow-[0_0_8px_rgba(156,163,175,0.4)]" />;
  if (code <= 67) return <CloudRain className="size-8 text-blue-400 drop-shadow-[0_0_8px_rgba(96,165,250,0.6)]" />;
  if (code <= 77) return <CloudSnow className="size-8 text-blue-200 drop-shadow-[0_0_8px_rgba(191,219,254,0.6)]" />;
  if (code <= 82) return <CloudRain className="size-8 text-blue-500 drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" />;
  if (code <= 99) return <CloudLightning className="size-8 text-yellow-500 drop-shadow-[0_0_8px_rgba(234,179,8,0.6)]" />;
  return <Cloud className="size-8 text-gray-400" />;
}

function getWeatherLabel(code: number): string {
  if (code === 0) return 'Clear Sky';
  if (code <= 3) return 'Partly Cloudy';
  if (code <= 48) return 'Foggy';
  if (code <= 55) return 'Light Drizzle';
  if (code <= 67) return 'Rain';
  if (code <= 77) return 'Snow';
  if (code <= 82) return 'Rain Showers';
  if (code <= 99) return 'Thunderstorm';
  return 'Clear Sky';
}

export const LiveWorldFeedWidget: React.FC = () => {
  const [weather, setWeather] = useState<LiveWeatherData | null>(() => getFeed<LiveWeatherData>('weather'));
  const [news, setNews] = useState<LiveNewsItem[]>(() => getFeed<LiveNewsItem[]>('news') ?? []);
  const [quote, setQuote] = useState<LiveQuote | null>(() => getFeed<LiveQuote>('quote'));
  const [exchange, setExchange] = useState<LiveExchangeRate | null>(() => getFeed<LiveExchangeRate>('exchange'));
  const [repos, setRepos] = useState<LiveGitHubRepo[]>(() => getFeed<LiveGitHubRepo[]>('repos') ?? []);
  const [fact, setFact] = useState<LiveNumberFact | null>(() => getFeed<LiveNumberFact>('fact'));
  const [holidays, setHolidays] = useState<LiveHoliday[]>(() => getFeed<LiveHoliday[]>('holidays') ?? []);
  const [market, setMarket] = useState<LiveMarketCoin[]>(() => getFeed<LiveMarketCoin[]>('market') ?? []);
  const [gold, setGold] = useState<LiveGoldPrice | null>(() => getFeed<LiveGoldPrice>('gold'));
  const [iss, setIss] = useState<LiveISSPosition | null>(() => getFeed<LiveISSPosition>('iss'));
  const [publicApis, setPublicApis] = useState<LivePublicApiCatalog | null>(() => getFeed<LivePublicApiCatalog>('public-apis'));

  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeNewsTab, setActiveNewsTab] = useState<'featured' | 'mostread' | 'onthisday'>('featured');
  const [lastRefresh, setLastRefresh] = useState<string>('Just now');

  const fetchAllData = useCallback(async (force = false) => {
    setIsRefreshing(true);

    const reval = async <T,>(key: string, ttlMs: number, fetcher: () => Promise<T>, apply: (d: T) => void) => {
      if (!force && isFeedFresh(key)) {
        const cached = getFeed<T>(key);
        if (cached) {
          apply(cached);
          return;
        }
      }
      try {
        const data = await fetcher();
        if (data) {
          apply(data);
          setFeed(key, data, ttlMs);
        }
      } catch (err) {
        console.warn(`Feed reval ${key} failed:`, err);
      }
    };

    await Promise.all([
      reval('weather', 5 * 60 * 1000, fetchLiveWeather, setWeather),
      reval('news', 15 * 60 * 1000, fetchLiveNews, setNews),
      reval('quote', 60 * 60 * 1000, fetchLiveQuote, setQuote),
      reval('exchange', 30 * 60 * 1000, fetchLiveExchange, setExchange),
      reval('repos', 30 * 60 * 1000, () => fetchLiveGitHub('IbrahimAbdelsattar'), setRepos),
      reval('fact', 60 * 60 * 1000, fetchLiveNumberFact, setFact),
      reval('holidays', 24 * 60 * 60 * 1000, fetchLiveHolidays, setHolidays),
      reval('market', 5 * 60 * 1000, fetchLiveMarket, setMarket),
      reval('gold', 10 * 60 * 1000, fetchLiveGold, setGold),
      reval('iss', 2 * 60 * 1000, fetchLiveISS, setIss),
      reval('public-apis', 60 * 60 * 1000, fetchLivePublicApis, setPublicApis),
    ]);

    setLastRefresh(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    setIsRefreshing(false);
  }, []);

  useEffect(() => {
    void fetchAllData();
    const interval = setInterval(() => void fetchAllData(), 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchAllData]);

  const filteredNews = news.filter((n) => n.type === activeNewsTab);

  return (
    <div className="w-full h-full flex flex-col rounded-xl bg-[#040d1a]/95 border border-[#00f0ff]/30 shadow-[0_0_25px_rgba(0,240,255,0.08)] overflow-hidden font-mono text-xs">
      {/* Header Deck */}
      <div className="px-4 py-3 border-b border-[#00f0ff]/20 bg-[#071526]/80 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="size-8 rounded-lg bg-[#00f0ff]/10 border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff]">
            <Globe className="size-4 animate-spin" style={{ animationDuration: '20s' }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-[#00f0ff] tracking-wide">LIVE WORLD FEED</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30">
                REAL-TIME TELEMETRY
              </span>
            </div>
            <p className="text-[11px] text-[#80f7ff]/60">Global Intelligence, Weather, Markets & Orbits</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] text-[#80f7ff]/60">Updated: {lastRefresh}</span>
          <button
            onClick={() => void fetchAllData(true)}
            disabled={isRefreshing}
            className="px-3 py-1.5 rounded-lg bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 border border-[#00f0ff]/30 text-[#00f0ff] flex items-center gap-1.5 transition-all disabled:opacity-50"
            title="Refresh All Feeds"
          >
            <RefreshCw className={`size-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>SYNC NOW</span>
          </button>
        </div>
      </div>

      {/* Main Grid View */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
        {/* Top Tickers: Weather, Crypto Market, Exchange & Gold */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Weather Card */}
          <div className="p-3.5 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 flex items-center justify-between">
            <div>
              <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1">
                <Thermometer className="size-3 text-amber-400" /> {weather?.location || 'Cairo, Egypt'}
              </span>
              <div className="text-xl font-bold text-cyan-100 mt-1">
                {weather?.temperature ?? 24}°C
              </div>
              <p className="text-[11px] text-amber-300 font-semibold mt-0.5">
                {getWeatherLabel(weather?.weatherCode ?? 0)}
              </p>
              <div className="flex items-center gap-2 mt-1.5 text-[10px] text-cyan-400/60">
                <span className="flex items-center gap-0.5"><Droplets className="size-3" /> {weather?.humidity ?? 45}%</span>
                <span className="flex items-center gap-0.5"><Wind className="size-3" /> {weather?.windSpeed ?? 12} km/h</span>
              </div>
            </div>
            <div>{getWeatherIcon(weather?.weatherCode ?? 0, weather?.isDay ?? true)}</div>
          </div>

          {/* Crypto Ticker */}
          <div className="p-3.5 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1">
                <Coins className="size-3 text-amber-400" /> CRYPTO TICKER
              </span>
              <span className="text-[9px] text-emerald-400 font-bold">24H LIVE</span>
            </div>
            <div className="space-y-1 mt-2">
              {market.slice(0, 3).map((coin) => (
                <div key={coin.symbol} className="flex items-center justify-between text-xs">
                  <span className="font-bold text-cyan-200">{coin.symbol}</span>
                  <span className="text-amber-300 tabular-nums">${coin.priceUsd.toLocaleString()}</span>
                  <span className={`text-[10px] tabular-nums ${coin.change24h >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {coin.change24h >= 0 ? '+' : ''}{coin.change24h}%
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Currency Exchange & Gold Card */}
          <div className="p-3.5 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1">
                <ArrowRightLeft className="size-3 text-[#00f0ff]" /> CURRENCY & GOLD
              </span>
              <span className="text-[9px] text-amber-400 font-bold flex items-center gap-0.5">
                <Gem className="size-3" /> XAU
              </span>
            </div>
            <div className="my-1.5 space-y-1">
              <div className="text-sm font-bold text-cyan-100">
                1 USD = <span className="text-amber-300">{exchange?.rate ?? 48.85} EGP</span>
              </div>
              <div className="text-xs text-amber-200 font-semibold flex items-center justify-between">
                <span>Gold (oz):</span>
                <span className="tabular-nums">${gold?.price?.toLocaleString() ?? '2,890.50'}</span>
              </div>
            </div>
            <span className="text-[9px] text-cyan-400/40">Base: USD • Live Market Rates</span>
          </div>

          {/* ISS Orbit Telemetry */}
          <div className="p-3.5 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1">
                <Satellite className="size-3 text-amber-400 animate-pulse" /> ISS SATELLITE
              </span>
              <span className="text-[9px] text-cyan-300 uppercase">{iss?.visibility || 'In Orbit'}</span>
            </div>
            <div className="my-1 text-xs text-cyan-200">
              <p>LAT: <strong className="text-amber-300">{iss?.latitude ?? 28.5}°</strong></p>
              <p>LNG: <strong className="text-amber-300">{iss?.longitude ?? 34.2}°</strong></p>
              <p className="text-[10px] text-cyan-400/60 mt-1">
                VEL: {iss?.velocityKmh?.toLocaleString() ?? '27,600'} km/h • ALT: {iss?.altitudeKm ?? 420} km
              </p>
            </div>
          </div>
        </div>

        {/* Middle Section: News Matrix & GitHub Repos */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* News Matrix (2 cols) */}
          <div className="lg:col-span-2 p-4 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 flex flex-col">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#00f0ff]/15">
              <div className="flex items-center gap-2 text-cyan-300 font-bold">
                <Newspaper className="size-4 text-[#00f0ff]" />
                <span>GLOBAL NEWS WIRE</span>
              </div>
              <div className="flex items-center gap-1 bg-[#030d1a] p-1 rounded-lg border border-[#00f0ff]/20 text-[11px]">
                {(['featured', 'mostread', 'onthisday'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveNewsTab(tab)}
                    className={`px-2.5 py-0.5 rounded capitalize transition-all ${
                      activeNewsTab === tab
                        ? 'bg-[#00f0ff]/20 text-[#00f0ff] font-bold'
                        : 'text-cyan-400/60 hover:text-cyan-200'
                    }`}
                  >
                    {tab === 'onthisday' ? 'On This Day' : tab}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3 mt-3">
              {filteredNews.map((item, idx) => (
                <a
                  key={idx}
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="p-3 rounded-lg bg-[#041120]/70 border border-[#00f0ff]/15 hover:border-[#00f0ff]/40 flex gap-3 group transition-all"
                >
                  {item.thumbnail && (
                    <img
                      src={item.thumbnail}
                      alt={item.title}
                      className="size-16 rounded-md object-cover border border-[#00f0ff]/20 shrink-0"
                    />
                  )}
                  <div className="min-w-0 flex-1">
                    <h4 className="text-xs font-bold text-cyan-200 group-hover:text-[#00f0ff] transition-colors flex items-center gap-1">
                      {item.title}
                      <ExternalLink className="size-3 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                    </h4>
                    <p className="text-[11px] text-cyan-400/70 line-clamp-2 mt-1 font-sans">
                      {item.extract}
                    </p>
                  </div>
                </a>
              ))}
              {filteredNews.length === 0 && (
                <p className="text-center text-cyan-400/50 py-6">Loading live stories...</p>
              )}
            </div>
          </div>

          {/* GitHub Repos & Facts */}
          <div className="space-y-4">
            {/* GitHub Card */}
            <div className="p-4 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25">
              <div className="flex items-center justify-between pb-2 border-b border-[#00f0ff]/15 mb-3">
                <span className="font-bold text-cyan-300 flex items-center gap-1.5">
                  <GitBranch className="size-4 text-[#00f0ff]" />
                  <span>GITHUB REPOSITORIES</span>
                </span>
                <span className="text-[10px] text-cyan-400/60">IbrahimAbdelsattar</span>
              </div>
              <div className="space-y-2">
                {repos.slice(0, 4).map((r) => (
                  <a
                    key={r.name}
                    href={r.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block p-2 rounded-lg bg-[#041120]/70 border border-[#00f0ff]/15 hover:border-[#00f0ff]/40 transition-all group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-cyan-200 group-hover:text-[#00f0ff] truncate">
                        {r.name}
                      </span>
                      <div className="flex items-center gap-2 text-[10px] text-amber-300 shrink-0">
                        <span className="flex items-center gap-0.5"><Star className="size-3" /> {r.stars}</span>
                        <span className="flex items-center gap-0.5"><GitBranch className="size-3" /> {r.forks}</span>
                      </div>
                    </div>
                    <p className="text-[10px] text-cyan-400/60 line-clamp-1 mt-0.5 font-sans">
                      {r.description}
                    </p>
                  </a>
                ))}
              </div>
            </div>

            {/* Quote of the Day & Math Trivia */}
            <div className="p-4 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 space-y-3">
              {quote && (
                <div>
                  <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1 mb-1">
                    <Quote className="size-3 text-amber-400" /> DAILY INSPIRATION
                  </span>
                  <blockquote className="text-xs italic text-cyan-100 font-sans leading-relaxed">
                    "{quote.content}"
                  </blockquote>
                  <p className="text-right text-[10px] text-amber-400 mt-1 font-mono">— {quote.author}</p>
                </div>
              )}
              {fact && (
                <div className="pt-3 border-t border-[#00f0ff]/15">
                  <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1 mb-1">
                    <Sparkles className="size-3 text-[#00f0ff]" /> NUMBER TRIVIA ({fact.number})
                  </span>
                  <p className="text-xs text-cyan-200 font-sans leading-relaxed">{fact.text}</p>
                </div>
              )}
            </div>

            {/* Holidays & Public APIs Matrix */}
            <div className="p-4 rounded-xl bg-[#06182c]/80 border border-[#00f0ff]/25 space-y-3">
              <div>
                <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1 mb-1.5">
                  <Calendar className="size-3 text-[#00f0ff]" /> UPCOMING PUBLIC HOLIDAYS
                </span>
                <div className="space-y-1 text-xs">
                  {holidays.slice(0, 2).map((h) => (
                    <div key={h.date} className="flex items-center justify-between text-cyan-200">
                      <span className="truncate">{h.localName || h.name}</span>
                      <span className="text-amber-300 text-[10px] shrink-0 ml-2 font-mono">{h.date}</span>
                    </div>
                  ))}
                  {holidays.length === 0 && (
                    <p className="text-[11px] text-cyan-400/50">No holiday data</p>
                  )}
                </div>
              </div>

              {publicApis && publicApis.entries && publicApis.entries.length > 0 && (
                <div className="pt-2 border-t border-[#00f0ff]/15">
                  <span className="text-[10px] text-cyan-400/60 uppercase font-bold flex items-center gap-1 mb-1.5">
                    <Sparkles className="size-3 text-amber-400" /> PUBLIC APIS CATALOG
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {publicApis.entries.slice(0, 5).map((entry) => (
                      <span
                        key={entry.name}
                        className="px-1.5 py-0.5 rounded text-[9px] bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/20"
                      >
                        {entry.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveWorldFeedWidget;
