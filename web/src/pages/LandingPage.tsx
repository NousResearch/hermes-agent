import { useEffect, useState } from "react";
import { Check, TrendingUp, Globe, Zap, CreditCard } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { fetchJSON } from "@/lib/api";

interface PlansResponse {
  plans: Record<string, { name: string; price_usd: number; url_limit: number; interval: string }>;
  stripe_starter_link: string;
  stripe_pro_link: string;
  stripe_payment_link: string;
  configured: boolean;
}

// url_limit values at or above this sentinel are rendered as "Unlimited".
const UNLIMITED_URLS = 1_000_000;

const urlLimitLabel = (limit: number | undefined, fallback: number): string => {
  const n = limit ?? fallback;
  return n >= UNLIMITED_URLS ? "Unlimited" : `${n}`;
};

const FEATURES = [
  { icon: Globe, text: "Add any URL — product pages, pricing tables, competitor sites" },
  { icon: Zap, text: "AI extracts prices, numbers, and metrics automatically" },
  { icon: TrendingUp, text: "Track changes over time with CSV and PDF export" },
];

export default function LandingPage() {
  const [plans, setPlans] = useState<PlansResponse | null>(null);

  useEffect(() => {
    fetchJSON<PlansResponse>("/api/subscribe/plans").then(setPlans).catch(() => {});
  }, []);

  const starterLink = plans?.stripe_starter_link ?? plans?.stripe_payment_link ?? "";
  const proLink = plans?.stripe_pro_link ?? "";
  const starter = plans?.plans?.starter;
  const pro = plans?.plans?.pro;

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Nav */}
      <nav className="border-b border-border/50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
          <span className="font-mondwest text-display tracking-wider text-foreground">
            Price Tracker
          </span>
        </div>
        <a href="/portal">
          <Button type="button" size="sm" outlined>
            Subscriber Portal
          </Button>
        </a>
      </nav>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-start px-6 pt-20 pb-16">
        <div className="max-w-2xl w-full text-center flex flex-col gap-6">
          <Badge tone="secondary" className="mx-auto text-xs">
            Powered by AI · No code required
          </Badge>

          <h1 className="font-mondwest text-display text-4xl sm:text-5xl tracking-wider text-foreground leading-tight">
            Track Any Website's<br />
            <span className="text-emerald-400">Prices & Numbers</span>
          </h1>

          <p className="text-muted-foreground text-lg max-w-lg mx-auto">
            Add a URL and our AI extracts every price, metric, and number —
            automatically, on a schedule, with export to CSV and PDF.
          </p>

          {/* Features */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mt-2">
            {FEATURES.map(({ icon: Icon, text }) => (
              <div key={text} className="flex items-start gap-2 text-sm text-muted-foreground text-left max-w-[180px]">
                <Icon className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                <span>{text}</span>
              </div>
            ))}
          </div>

          {/* Pricing */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center mt-4">
            {/* Starter */}
            <Card className="border-emerald-500/50 bg-emerald-500/5 flex-1 max-w-xs">
              <CardContent className="py-6 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <span className="font-mondwest tracking-wider text-foreground">
                    {starter?.name ?? "Starter"}
                  </span>
                  <Badge tone="success" className="text-xs">Most Popular</Badge>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold text-foreground">
                    ${starter?.price_usd ?? 19}
                  </span>
                  <span className="text-muted-foreground text-sm">/month</span>
                </div>
                <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    {urlLimitLabel(starter?.url_limit, 3)} tracked URLs
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    AI price & number extraction
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    CSV + PDF export
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    Historical change tracking
                  </li>
                </ul>
                {starterLink ? (
                  <a href={starterLink} target="_blank" rel="noopener noreferrer">
                    <Button type="button" className="w-full">
                      <CreditCard className="h-3.5 w-3.5 mr-1.5" />
                      Start Tracking
                    </Button>
                  </a>
                ) : (
                  <Button type="button" className="w-full" disabled>
                    Coming Soon
                  </Button>
                )}
              </CardContent>
            </Card>

            {/* Pro */}
            <Card className="flex-1 max-w-xs">
              <CardContent className="py-6 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <span className="font-mondwest tracking-wider text-foreground">
                    {pro?.name ?? "Pro"}
                  </span>
                  <Badge tone="secondary" className="text-xs">More URLs</Badge>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl font-bold text-foreground">
                    ${pro?.price_usd ?? 49}
                  </span>
                  <span className="text-muted-foreground text-sm">/month</span>
                </div>
                <ul className="flex flex-col gap-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-sky-400 shrink-0" />
                    {urlLimitLabel(pro?.url_limit, UNLIMITED_URLS)} tracked URLs
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-sky-400 shrink-0" />
                    Everything in Starter
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-sky-400 shrink-0" />
                    Priority scraping
                  </li>
                  <li className="flex items-center gap-2">
                    <Check className="h-3.5 w-3.5 text-sky-400 shrink-0" />
                    Email change alerts
                  </li>
                </ul>
                {proLink ? (
                  <a href={proLink} target="_blank" rel="noopener noreferrer">
                    <Button type="button" outlined className="w-full">
                      <CreditCard className="h-3.5 w-3.5 mr-1.5" />
                      Go Unlimited
                    </Button>
                  </a>
                ) : (
                  <Button type="button" outlined className="w-full" disabled>
                    Coming Soon
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Already subscribed */}
          <p className="text-sm text-muted-foreground mt-2">
            Already subscribed?{" "}
            <a href="/portal" className="underline hover:text-foreground transition-colors">
              Go to your portal →
            </a>
          </p>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 px-6 py-4 text-center text-xs text-text-tertiary">
        Powered by Hermes Agent · Secure payments via Stripe
      </footer>
    </div>
  );
}
