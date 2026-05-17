import { type FC, useEffect } from "react";
import { usePageHeader } from "../contexts/usePageHeader";
import {
  ArchDiagram,
  LeftRail,
  RightRail,
  SpendSparkline,
  QueueSparkline,
  TopStrip,
} from "../components/mission";
import {
  useMissionStream,
  useMissionSnapshot,
  useSpendHistory,
  useQueueHistory,
} from "../components/mission/useMissionStream";

const MissionControlPage: FC = () => {
  const { setTitle } = usePageHeader();
  const chips = useMissionStream();
  const snapshot = useMissionSnapshot(30_000);
  const spendHistory = useSpendHistory("7d");
  const queueHistory = useQueueHistory("7d");

  useEffect(() => {
    setTitle("Cockpit");
  }, [setTitle]);

  // Merge live SSE chips into snapshot runtimes
  const mergedSnapshot = snapshot
    ? {
        ...snapshot,
        runtimes:
          chips.length > 0
            ? snapshot.runtimes.map(
                (r) => chips.find((c) => c.name === r.name) ?? r
              )
            : snapshot.runtimes,
      }
    : null;

  if (!mergedSnapshot) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
        Loading…
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-2 p-3 bg-background">
      <TopStrip snapshot={mergedSnapshot} />
      {/* Cockpit layout: 3-col on desktop, single column on mobile (≤768px) */}
      <div className="flex flex-col md:flex-row flex-1 gap-2 min-h-0">
        <div className="md:w-[200px] md:shrink-0 order-2 md:order-1">
          <LeftRail runtimes={mergedSnapshot.runtimes} />
        </div>
        <ArchDiagram className="flex-1 min-w-0 min-h-[200px] order-1 md:order-2" />
        <div className="md:w-[280px] md:shrink-0 order-3">
          <RightRail snapshot={mergedSnapshot} />
        </div>
      </div>
      <div className="flex gap-4 shrink-0 h-[80px]">
        {spendHistory && (
          <SpendSparkline
            points={spendHistory.points}
            label="Spend (USD)"
            className="flex-1"
          />
        )}
        {queueHistory && (
          <QueueSparkline
            points={queueHistory.points}
            openNow={queueHistory.openNow}
            label="Queue (tasks/day)"
            className="flex-1"
          />
        )}
      </div>
    </div>
  );
};

export default MissionControlPage;
