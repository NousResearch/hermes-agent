import {
  Play,
  Pause,
  RotateCcw,
  RefreshCw,
  Clock,
  Cpu,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { CodeSession } from "@/types/code";

interface CodeSessionHeaderProps {
  session: CodeSession;
  onCancel?: () => void;
  onResume?: () => void;
  onRefresh?: () => void;
}

export function CodeSessionHeader({
  session,
  onCancel,
  onResume,
  onRefresh,
}: CodeSessionHeaderProps) {
  const getStatusIcon = () => {
    switch (session.status) {
      case "running":
        return <Play className="h-4 w-4 text-green-500" />;
      case "waiting_approval":
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-success" />;
      case "failed":
      case "cancelled":
        return <XCircle className="h-4 w-4 text-destructive" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = () => {
    switch (session.status) {
      case "running":
        return (
          <Badge variant="success" className="text-[10px]">
            <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            Running
          </Badge>
        );
      case "waiting_approval":
        return (
          <Badge variant="warning" className="text-[10px]">
            Awaiting Approval
          </Badge>
        );
      case "completed":
        return (
          <Badge variant="outline" className="text-[10px]">
            Completed
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive" className="text-[10px]">
            Failed
          </Badge>
        );
      case "cancelled":
        return (
          <Badge variant="outline" className="text-[10px]">
            Cancelled
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-[10px]">
            {session.status}
          </Badge>
        );
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "—";
    try {
      return new Date(dateStr).toLocaleString();
    } catch {
      return dateStr;
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <CardTitle className="text-sm font-medium">
              {session.title || "Untitled Session"}
            </CardTitle>
          </div>
          {getStatusBadge()}
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {(session.provider || session.model) && (
            <div className="flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              <span className="font-mono">
                {[session.provider, session.model].filter(Boolean).join("/")}
              </span>
            </div>
          )}
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{formatDate(session.created_at)}</span>
          </div>
        </div>

        {session.error && (
          <div className="text-xs text-destructive p-2 border border-destructive/30 bg-destructive/5 rounded">
            {session.error}
          </div>
        )}

        <div className="flex gap-2 pt-1">
          {session.status === "running" && onCancel && (
            <Button onClick={onCancel} variant="outline" size="sm" className="h-8 text-xs">
              <Pause className="h-3 w-3 mr-1" />
              Cancel
            </Button>
          )}
          {(session.status === "cancelled" || session.status === "failed") && onResume && (
            <Button onClick={onResume} variant="outline" size="sm" className="h-8 text-xs">
              <RotateCcw className="h-3 w-3 mr-1" />
              Resume
            </Button>
          )}
          {onRefresh && (
            <Button onClick={onRefresh} variant="ghost" size="sm" className="h-8 text-xs">
              <RefreshCw className="h-3 w-3 mr-1" />
              Refresh
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
