import type { AnalysisResult } from "../types/music";
import { Clock, Music, Hash, Timer } from "lucide-react";

interface ChordDisplayProps {
  analysis: AnalysisResult;
}

const CHORD_COLORS: Record<string, string> = {
  A: "bg-purple-600", Am: "bg-purple-600", "Am7": "bg-purple-600", "Am9": "bg-purple-600",
  B: "bg-pink-600", Bm: "bg-pink-600",
  C: "bg-blue-600", Cmaj7: "bg-blue-600", "C6": "bg-blue-600",
  D: "bg-yellow-600", Dm: "bg-yellow-600", "Dm7": "bg-yellow-600",
  E: "bg-red-600", Em: "bg-red-600", "E7": "bg-red-600",
  F: "bg-orange-600", Fmaj7: "bg-orange-600", "F9": "bg-orange-600", "Fmaj9": "bg-orange-600",
  G: "bg-green-600", "G7": "bg-green-600",
};

function getChordColor(chord: string): string {
  // Try exact match first, then root note
  if (CHORD_COLORS[chord]) return CHORD_COLORS[chord];
  const root = chord.match(/^[A-G][#b]?/)?.[0];
  if (root && CHORD_COLORS[root]) return CHORD_COLORS[root];
  return "bg-gray-600";
}

export default function ChordDisplay({ analysis }: ChordDisplayProps) {
  return (
    <div className="space-y-4">
      {/* Detected chords label */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Detected chords
        </h3>
        {analysis.chords.length > 2 && (
          <span className="text-xs text-text-muted italic">repeating</span>
        )}
      </div>

      {/* Chord pills */}
      <div className="flex flex-wrap gap-2">
        {analysis.chords.map((chord, i) => (
          <span
            key={i}
            className={`px-3 py-1.5 rounded-lg text-sm font-semibold text-white ${getChordColor(chord)}`}
          >
            {chord}
          </span>
        ))}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3">
        <div className="flex items-center gap-2 bg-bg-surface rounded-lg px-3 py-2">
          <Clock className="w-3.5 h-3.5 text-accent-purple" />
          <div>
            <p className="text-xs text-text-muted">BPM</p>
            <p className="text-sm font-semibold">{analysis.bpm}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-bg-surface rounded-lg px-3 py-2">
          <Music className="w-3.5 h-3.5 text-accent-orange" />
          <div>
            <p className="text-xs text-text-muted">Key</p>
            <p className="text-sm font-semibold">{analysis.key}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-bg-surface rounded-lg px-3 py-2">
          <Hash className="w-3.5 h-3.5 text-accent-green" />
          <div>
            <p className="text-xs text-text-muted">Time</p>
            <p className="text-sm font-semibold">{analysis.time_sig}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 bg-bg-surface rounded-lg px-3 py-2">
          <Timer className="w-3.5 h-3.5 text-accent-blue" />
          <div>
            <p className="text-xs text-text-muted">Duration</p>
            <p className="text-sm font-semibold">{formatDuration(analysis.duration)}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
