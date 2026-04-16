import { Loader2, Sparkles } from "lucide-react";

interface AnalyzeButtonProps {
  onClick: () => void;
  loading: boolean;
  disabled: boolean;
  progress?: string;
}

export default function AnalyzeButton({ onClick, loading, disabled, progress }: AnalyzeButtonProps) {
  return (
    <div className="space-y-2">
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className={`w-full py-3 px-6 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2
          ${disabled || loading
            ? "bg-bg-surface-2 text-text-muted cursor-not-allowed"
            : "bg-gradient-to-r from-accent-purple to-accent-blue text-white hover:opacity-90 active:scale-[0.98]"
          }`}
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Analyzing...
          </>
        ) : (
          <>
            <Sparkles className="w-4 h-4" />
            Analyze & Generate
          </>
        )}
      </button>

      {progress && (
        <p className="text-xs text-center text-text-muted">{progress}</p>
      )}
    </div>
  );
}
