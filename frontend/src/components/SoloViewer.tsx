import { useRef, useEffect } from "react";
import { Play, Square, Download } from "lucide-react";
import type { SoloResponse } from "../types/music";

interface SoloViewerProps {
  solo: SoloResponse;
  midiUrl?: string;
  playing: boolean;
  currentTime: number;
  onPlay: () => void;
  onStop: () => void;
  onExport: () => void;
}

// Piano range for display
const PITCH_MIN = 36;  // C2 (left hand low)
const PITCH_MAX = 96;  // C7 (right hand high)
const PITCH_RANGE = PITCH_MAX - PITCH_MIN;

// Right hand = purple/blue, Left hand = orange/warm
function noteColor(hand: string, velocity: number): string {
  const t = velocity / 127;
  if (hand === "left") {
    const r = Math.round(200 + 55 * t);
    const g = Math.round(100 + 50 * t);
    const b = Math.round(30 + 40 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
  // right hand
  const r = Math.round(100 + 39 * t);
  const g = Math.round(60 + 32 * t);
  const b = Math.round(200 + 46 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function SoloViewer({
  solo, midiUrl, playing, currentTime, onPlay, onStop, onExport,
}: SoloViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !solo.notes.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const bars = solo.bars || 8;
    const beatsPerBar = 4;
    const totalBeats = bars * beatsPerBar;

    // Background
    ctx.fillStyle = "#0d0d15";
    ctx.fillRect(0, 0, W, H);

    // C4 divider line (split between left and right hand)
    const c4Y = H - ((60 - PITCH_MIN) / PITCH_RANGE) * H;
    ctx.strokeStyle = "rgba(139, 92, 246, 0.2)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, c4Y);
    ctx.lineTo(W, c4Y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label C4 divider
    ctx.fillStyle = "rgba(139, 92, 246, 0.3)";
    ctx.font = "9px Inter";
    ctx.fillText("C4 — RH", 4, c4Y - 4);
    ctx.fillText("LH", 4, c4Y + 12);

    // Grid lines — bars
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let bar = 0; bar <= bars; bar++) {
      const x = (bar / bars) * W;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();

      if (bar < bars) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
        ctx.font = "10px Inter";
        ctx.fillText(`${bar + 1}`, x + 4, H - 4);
      }
    }

    // Grid lines — beats (lighter)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    for (let beat = 0; beat < totalBeats; beat++) {
      if (beat % beatsPerBar === 0) continue;
      const x = (beat / totalBeats) * W;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    // Pitch lanes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.015)";
    for (let p = PITCH_MIN; p <= PITCH_MAX; p += 2) {
      const y = H - ((p - PITCH_MIN) / PITCH_RANGE) * H;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Draw notes
    const noteHeight = Math.max(3, H / PITCH_RANGE * 0.7);

    for (const note of solo.notes) {
      const beatPos = (note.bar - 1) * beatsPerBar + (note.beat - 1);
      const x = (beatPos / totalBeats) * W;
      const w = (note.duration / totalBeats) * W * beatsPerBar;
      const y = H - ((note.pitch - PITCH_MIN) / PITCH_RANGE) * H - noteHeight / 2;

      if (note.pitch < PITCH_MIN || note.pitch > PITCH_MAX) continue;

      const hand = (note as any).hand || (note.pitch < 60 ? "left" : "right");
      const color = noteColor(hand, note.velocity);

      // Note body with rounded corners
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(x, y, Math.max(w, 3), noteHeight, 2);
      ctx.fill();

      // Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = hand === "right" ? 8 : 4;
      ctx.fillRect(x, y, Math.max(w, 3), noteHeight);
      ctx.shadowBlur = 0;

      // Grace note indicator (very short notes)
      if (note.duration <= 0.06) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.beginPath();
        ctx.arc(x + 1, y + noteHeight / 2, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Playback cursor
    if (playing && currentTime > 0) {
      const secondsPerBeat = 60 / solo.tempo;
      const totalSeconds = totalBeats * secondsPerBeat;
      const cursorX = (currentTime / totalSeconds) * W;

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, H);
      ctx.stroke();

      ctx.strokeStyle = "rgba(139, 92, 246, 0.5)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(cursorX, 0);
      ctx.lineTo(cursorX, H);
      ctx.stroke();
    }
  }, [solo, playing, currentTime]);

  // Count notes by hand
  const rhCount = solo.notes.filter((n: any) => (n.hand || (n.pitch >= 60 ? "right" : "left")) === "right").length;
  const lhCount = solo.notes.length - rhCount;

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
          AI Piano Solo — {solo.bars} Bars
        </h3>
        <div className="flex gap-3 text-xs text-text-muted">
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-purple-500 mr-1" />
            RH: {rhCount}
          </span>
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-orange-500 mr-1" />
            LH: {lhCount}
          </span>
        </div>
      </div>

      {/* Piano roll canvas */}
      <div className="relative rounded-xl overflow-hidden border border-border-subtle bg-[#0d0d15]">
        <canvas
          ref={canvasRef}
          className="w-full"
          style={{ height: 200 }}
        />
      </div>

      {/* Phrase notes */}
      {solo.phrase_notes.length > 0 && (
        <div className="space-y-1">
          {solo.phrase_notes.map((note, i) => (
            <p key={i} className="text-[10px] text-text-muted">{note}</p>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={playing ? onStop : onPlay}
          disabled={!midiUrl}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-medium transition-all
            ${playing
              ? "bg-accent-red/10 text-accent-red hover:bg-accent-red/20"
              : "bg-accent-purple/10 text-accent-purple hover:bg-accent-purple/20"
            } ${!midiUrl ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {playing ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          {playing ? "Stop" : "Play"}
        </button>

        <button
          onClick={onExport}
          disabled={!midiUrl}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-lg bg-bg-surface-2 text-text-secondary text-xs font-medium
            hover:text-text-primary transition-colors ${!midiUrl ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <Download className="w-3.5 h-3.5" />
          Export MIDI
        </button>
      </div>
    </div>
  );
}
