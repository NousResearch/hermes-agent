import { useEffect } from "react";
import { AlertTriangle, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useApprovalStore } from "@/stores/codeStore";
import type { Approval } from "@/types/code";

interface CodeApprovalsPanelProps {
  codeSessionId?: string;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function CodeApprovalsPanel({ codeSessionId: _codeSessionId }: CodeApprovalsPanelProps) {
  const { approvals, fetchApprovals, approve, reject } = useApprovalStore();

  useEffect(() => {
    fetchApprovals("pending");
  }, [fetchApprovals]);

  const pendingApprovals = approvals.filter((a) => a.status === "pending");
  const recentApprovals = approvals.filter((a) => a.status !== "pending").slice(0, 5);

  const handleApprove = async (id: string) => {
    await approve(id);
    fetchApprovals("pending");
  };

  const handleReject = async (id: string) => {
    await reject(id);
    fetchApprovals("pending");
  };

  const getKindBadge = (kind: Approval["kind"]) => {
    switch (kind) {
      case "command":
        return (
          <Badge variant="outline" className="text-[10px]">
            Command
          </Badge>
        );
      case "code_review":
        return (
          <Badge variant="outline" className="text-[10px]">
            Code Review
          </Badge>
        );
      case "destructive_action":
        return (
          <Badge variant="destructive" className="text-[10px]">
            Destructive
          </Badge>
        );
      case "skill":
        return (
          <Badge variant="outline" className="text-[10px]">
            Skill
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-[10px]">
            {kind}
          </Badge>
        );
    }
  };

  const renderApproval = (approval: Approval, showActions = false) => (
    <div
      key={approval.id}
      className={`border rounded-lg p-3 space-y-2 ${
        approval.status === "pending" ? "border-yellow-500/30 bg-yellow-500/5" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <AlertTriangle className="h-4 w-4 text-yellow-500 shrink-0" />
          <span className="text-sm font-medium truncate">{approval.title}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {getKindBadge(approval.kind)}
          <Badge
            variant={approval.status === "pending" ? "warning" : approval.status === "approved" ? "success" : "destructive"}
            className="text-[10px]"
          >
            {approval.status}
          </Badge>
        </div>
      </div>

      {approval.command && (
        <code className="text-xs font-mono block p-2 bg-muted rounded truncate">
          {approval.command}
        </code>
      )}

      {approval.details && (
        <p className="text-xs text-muted-foreground">{approval.details}</p>
      )}

      {showActions && approval.status === "pending" && (
        <div className="flex gap-2 pt-1">
          <Button
            onClick={() => handleApprove(approval.id)}
            variant="outline"
            size="sm"
            className="h-8 text-xs flex-1"
          >
            <Check className="h-3 w-3 mr-1" />
            Approve
          </Button>
          <Button
            onClick={() => handleReject(approval.id)}
            variant="outline"
            size="sm"
            className="h-8 text-xs flex-1"
          >
            <X className="h-3 w-3 mr-1" />
            Reject
          </Button>
        </div>
      )}

      <div className="text-[10px] text-muted-foreground">
        {new Date(approval.created_at).toLocaleString()}
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-yellow-500" />
          Approvals
          {pendingApprovals.length > 0 && (
            <Badge variant="warning" className="text-[10px]">
              {pendingApprovals.length} pending
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {pendingApprovals.length === 0 && recentApprovals.length === 0 ? (
          <div className="text-center py-6 text-muted-foreground">
            <Check className="h-8 w-8 mx-auto mb-2 opacity-50 text-success" />
            <p className="text-sm">No pending approvals</p>
            <p className="text-xs">All clear</p>
          </div>
        ) : (
          <>
            {pendingApprovals.length > 0 && (
              <div className="space-y-2">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Pending ({pendingApprovals.length})
                </span>
                {pendingApprovals.map((a) => renderApproval(a, true))}
              </div>
            )}

            {recentApprovals.length > 0 && (
              <div className="space-y-2">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  Recent
                </span>
                {recentApprovals.map((a) => renderApproval(a, false))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
