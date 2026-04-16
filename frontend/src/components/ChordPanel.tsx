import { useState } from "react";
import { Plus, X, Play, ChevronDown, ChevronUp } from "lucide-react";
import type { ChordSuggestion, ChordResponse } from "../types/music";

interface ChordPanelProps {
  chordResponse: ChordResponse;
  onAccept: (suggestion: ChordSuggestion) => void;
  onReject: (suggestion: ChordSuggestion) => void;
  onPlay: (suggestion: ChordSuggestion) => void;
}

const DIFFICULTY_CONFIG = {
  beginner: { label: "Beginner", color: "bg-accent-green", glow: "glow-green", border: "border-accent-green/30" },
  intermediate: { label: "Intermediate", color: "bg-accent-orange", glow: "glow-orange", border: "border-accent-orange/30" },
  advanced: { label: "Advanced", color: "bg-accent-purple", glow: "glow-purple", border: "border-accent-purple/30" },
};

const CHORD_COLORS = [
  "bg-purple-600", "bg-orange-500", "bg-blue-500", "bg-green-500",
  "bg-yellow-500", "bg-red-500", "bg-pink-500", "bg-teal-500",
];

export default function ChordPanel({ chordResponse, onAccept, onReject, onPlay }: ChordPanelProps) {
  const [activeTab, setActiveTab] = useState<"chords" | "solo">("chords");
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <div className="space-y-4">
      {/* Header with tabs */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
          AI Suggestions
        </h3>
        <div className="flex gap-1 bg-bg-surface rounded-lg p-1">
          <button
            onClick={() => setActiveTab("chords")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              activeTab === "chords" ? "bg-bg-surface-2 text-text-primary" : "text-text-secondary"
            }`}
          >
            Chords
          </button>
          <button
            onClick={() => setActiveTab("solo")}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
              activeTab === "solo" ? "bg-bg-surface-2 text-text-primary" : "text-text-secondary"
            }`}
          >
            Solo
          </button>
        </div>
      </div>

      {/* Suggestion cards */}
      {activeTab === "chords" && (
        <div className="space-y-3">
          {chordResponse.suggestions.map((suggestion, idx) => {
            const config = DIFFICULTY_CONFIG[suggestion.difficulty] || DIFFICULTY_CONFIG.beginner;
            const isExpanded = expanded === idx;

            return (
              <div
                key={idx}
                className={`rounded-xl border ${config.border} bg-bg-surface p-4 transition-all animate-slide-up ${config.glow}`}
                style={{ animationDelay: `${idx * 100}ms` }}
              >
                {/* Title + badge */}
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-text-primary">{suggestion.label}</h4>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${config.color} text-white`}>
                    {config.label}
                  </span>
                </div>

                {/* Chord progression pills */}
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {suggestion.progression.map((chord, ci) => (
                    <span
                      key={ci}
                      className={`px-2.5 py-1 rounded-md text-xs font-semibold text-white ${CHORD_COLORS[ci % CHORD_COLORS.length]}`}
                    >
                      {chord}
                    </span>
                  ))}
                </div>

                {/* Description */}
                <p className="text-xs text-text-secondary mb-3 leading-relaxed">
                  {suggestion.overall_effect}
                </p>

                {/* Expanded: show changes */}
                {isExpanded && (
                  <div className="mb-3 space-y-1.5 p-2 bg-bg-primary/50 rounded-lg">
                    {suggestion.changes.map((change, ci) => (
                      <p key={ci} className="text-xs text-text-muted">
                        <span className="text-accent-red line-through">{change.original}</span>
                        {" → "}
                        <span className="text-accent-green font-medium">{change.replacement}</span>
                        {" — "}{change.reason}
                      </p>
                    ))}
                  </div>
                )}

                {/* Actions */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => onAccept(suggestion)}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-accent-green/10 text-accent-green text-xs font-medium
                             hover:bg-accent-green/20 transition-colors"
                  >
                    <Plus className="w-3 h-3" />
                    Dung cai nay
                  </button>
                  <button
                    onClick={() => onReject(suggestion)}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-bg-surface-2 text-text-secondary text-xs font-medium
                             hover:bg-accent-red/10 hover:text-accent-red transition-colors"
                  >
                    Bo
                  </button>
                  <button
                    onClick={() => onPlay(suggestion)}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-accent-purple/10 text-accent-purple text-xs font-medium
                             hover:bg-accent-purple/20 transition-colors"
                  >
                    <Play className="w-3 h-3" />
                    Play
                  </button>
                  <button
                    onClick={() => setExpanded(isExpanded ? null : idx)}
                    className="ml-auto p-1 text-text-muted hover:text-text-primary transition-colors"
                  >
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            );
          })}

          {/* Theory note */}
          {chordResponse.theory_note && (
            <div className="p-3 rounded-lg bg-accent-purple/5 border border-accent-purple/20">
              <p className="text-xs text-text-secondary italic">{chordResponse.theory_note}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
