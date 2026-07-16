/**
 * ErrorBoundary — catches render errors in child components
 * and shows a fallback UI instead of a blank page.
 */

import { Component, type ReactNode } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ChatErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("ChatErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
          <AlertTriangle className="h-10 w-10 text-destructive" />
          <div className="text-center">
            <Typography className="text-sm font-medium text-destructive">
              Something went wrong
            </Typography>
            <Typography className="text-xs text-text-tertiary mt-1">
              {this.state.error?.message || "An unexpected render error occurred."}
            </Typography>
          </div>
          <Button
            onClick={() => {
              this.setState({ hasError: false, error: null });
              window.location.reload();
            }}
          >
            Reload page
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}
