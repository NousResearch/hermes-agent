import { useEffect, useMemo, useState } from "react";
import { Package, RefreshCw, Search, Filter } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SkillEntry {
  name: string;
  description: string;
  family: string;
  origin: "hermes" | "openclaw";
  version: string | null;
  author: string | null;
  license: string | null;
  allowed_tools: string[];
  tags: string[];
  skill_dir: string;
}

interface SkillsSummary {
  total: number;
  by_origin: Record<string, number>;
  by_family: Record<string, number>;
  with_allowed_tools: number;
  with_tags: number;
  with_version: number;
}

interface SkillsResponse {
  summary: SkillsSummary;
  skills: SkillEntry[];
  error?: string;
}

export default function SkillsCatalogPage() {
  const [data, setData] = useState<SkillsResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [originFilter, setOriginFilter] = useState<string>("all");
  const [familyFilter, setFamilyFilter] = useState<string>("all");

  const load = async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/dual-agent/skills");
      const j = (await r.json()) as SkillsResponse;
      if (j.error) {
        setErr(j.error);
      } else {
        setData(j);
        setErr(null);
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const families = useMemo(() => {
    if (!data) return [] as string[];
    return Array.from(
      new Set(data.skills.map((s) => `${s.origin}/${s.family}`))
    ).sort();
  }, [data]);

  const filtered = useMemo(() => {
    if (!data) return [] as SkillEntry[];
    const q = query.trim().toLowerCase();
    return data.skills.filter((s) => {
      if (originFilter !== "all" && s.origin !== originFilter) return false;
      if (
        familyFilter !== "all" &&
        `${s.origin}/${s.family}` !== familyFilter
      )
        return false;
      if (!q) return true;
      return (
        s.name.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.tags.some((t) => t.toLowerCase().includes(q)) ||
        s.family.toLowerCase().includes(q)
      );
    });
  }, [data, query, originFilter, familyFilter]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <RefreshCw className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (err) {
    return (
      <div className="mx-auto max-w-4xl p-6">
        <Card>
          <CardContent className="py-6">
            <div className="font-medium">載入 skills 失敗</div>
            <div className="mt-1 text-sm text-muted-foreground">{err}</div>
            <Button className="mt-3" onClick={load}>重試</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <Package className="h-6 w-6" /> Skills catalog
          </h1>
          <div className="mt-1 text-sm text-muted-foreground">
            SKILL.md-driven unified catalog for Hermes + OpenClaw · 來自 se-023
          </div>
        </div>
        <Button variant="outline" onClick={load} className="gap-2">
          <RefreshCw className="h-4 w-4" /> 刷新
        </Button>
      </div>

      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">總覽</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Total</div>
              <div className="text-2xl font-bold">{data.summary.total}</div>
            </div>
            {Object.entries(data.summary.by_origin).map(([k, v]) => (
              <div key={k}>
                <div className="text-muted-foreground">{k}</div>
                <div className="text-2xl font-bold">{v}</div>
              </div>
            ))}
            <div>
              <div className="text-muted-foreground">with tags</div>
              <div className="text-2xl font-bold">{data.summary.with_tags}</div>
            </div>
            <div>
              <div className="text-muted-foreground">with version</div>
              <div className="text-2xl font-bold">{data.summary.with_version}</div>
            </div>
            <div>
              <div className="text-muted-foreground">with allowed-tools</div>
              <div className="text-2xl font-bold">
                {data.summary.with_allowed_tools}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Filter className="h-4 w-4" /> 篩選
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <div className="min-w-[200px] flex-1">
              <label className="mb-1 block text-xs text-muted-foreground">
                搜尋（名稱 / 描述 / tag）
              </label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="例：BWStudio、research、calendar..."
                  className="pl-9"
                />
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">
                Origin
              </label>
              <select
                value={originFilter}
                onChange={(e) => setOriginFilter(e.target.value)}
                className="rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="all">all</option>
                <option value="hermes">hermes</option>
                <option value="openclaw">openclaw</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-muted-foreground">
                Family
              </label>
              <select
                value={familyFilter}
                onChange={(e) => setFamilyFilter(e.target.value)}
                className="rounded-md border border-input bg-background px-3 py-2 text-sm max-w-[280px]"
              >
                <option value="all">all ({families.length})</option>
                {families.map((f) => (
                  <option key={f} value={f}>
                    {f} ({data.summary.by_family[f] || 0})
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-3 text-xs text-muted-foreground">
            顯示 {filtered.length} / {data.skills.length}
          </div>
        </CardContent>
      </Card>

      {/* Results list */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Skills</CardTitle>
        </CardHeader>
        <CardContent>
          {filtered.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              無結果，試著放寬篩選條件
            </div>
          ) : (
            <div className="space-y-2">
              {filtered.map((s, i) => (
                <div
                  key={s.origin + ":" + s.family + ":" + s.name + ":" + i}
                  className="rounded-md border border-border/60 bg-card/50 p-3"
                >
                  <div className="flex flex-wrap items-start gap-2">
                    <Badge
                      variant={s.origin === "hermes" ? "success" : "warning"}
                    >
                      {s.origin}
                    </Badge>
                    <Badge variant="outline">{s.family}</Badge>
                    <span className="font-mono text-sm font-medium">{s.name}</span>
                    {s.version && (
                      <span className="text-xs text-muted-foreground">
                        v{s.version}
                      </span>
                    )}
                    {s.license && (
                      <span className="text-xs text-muted-foreground">
                        · {s.license}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-sm">{s.description}</div>
                  {(s.tags.length > 0 || s.allowed_tools.length > 0) && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {s.tags.slice(0, 6).map((t) => (
                        <Badge key={t} variant="outline" className="text-xs">
                          {t}
                        </Badge>
                      ))}
                      {s.allowed_tools.slice(0, 4).map((t) => (
                        <Badge key={"at-" + t} variant="outline" className="text-xs">
                          🔧 {t}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
