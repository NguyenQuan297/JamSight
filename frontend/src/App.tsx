import { useState, useCallback } from "react";
import type { Genre, TabView, ChordSuggestion } from "./types/music";
import { useVideoAnalysis } from "./hooks/useVideoAnalysis";
import { useMidiPlayback } from "./hooks/useMidiPlayback";
import Header from "./components/Header";
import VideoCapture from "./components/VideoCapture";
import ChordDisplay from "./components/ChordDisplay";
import ChordPanel from "./components/ChordPanel";
import SoloViewer from "./components/SoloViewer";
import AnalyzeButton from "./components/AnalyzeButton";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabView>("analyze");
  const [genre, setGenre] = useState<Genre>("blues");
  const [file, setFile] = useState<File | null>(null);

  const { result, loading, error, progress, analyze, submitFeedback, reset } = useVideoAnalysis();
  const { playing, currentTime, play, stop } = useMidiPlayback();

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    try {
      await analyze(file, genre, "piano");
    } catch {
      // error state handled by hook
    }
  }, [file, genre, analyze]);

  const handleAccept = useCallback((suggestion: ChordSuggestion) => {
    if (!result) return;
    submitFeedback({
      session_id: result.session_id,
      input_progression: result.analysis.chords,
      genre: result.analysis.genre,
      suggestion_shown: suggestion as unknown as Record<string, unknown>,
      suggestion_rank: suggestion.rank,
      user_action: "accepted",
    });
  }, [result, submitFeedback]);

  const handleReject = useCallback((suggestion: ChordSuggestion) => {
    if (!result) return;
    submitFeedback({
      session_id: result.session_id,
      input_progression: result.analysis.chords,
      genre: result.analysis.genre,
      suggestion_shown: suggestion as unknown as Record<string, unknown>,
      suggestion_rank: suggestion.rank,
      user_action: "rejected",
    });
  }, [result, submitFeedback]);

  const handlePlaySuggestion = useCallback((_suggestion: ChordSuggestion) => {
    // Could generate MIDI for this specific suggestion
    // For now, play the combined MIDI
    if (result?.midi_urls?.combined) {
      if (playing) stop();
      else play(result.midi_urls.combined);
    }
  }, [result, playing, play, stop]);

  const handlePlaySolo = useCallback(() => {
    if (result?.midi_urls?.solo) {
      play(result.midi_urls.solo);
    }
  }, [result, play]);

  const handleExport = useCallback(() => {
    if (result?.midi_urls?.solo) {
      const a = document.createElement("a");
      a.href = result.midi_urls.solo;
      a.download = `jamsight_solo_${result.session_id}.mid`;
      a.click();
    }
  }, [result]);

  return (
    <div className="min-h-screen flex flex-col bg-bg-primary">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        genre={genre}
        onGenreChange={setGenre}
      />

      <main className="flex-1 p-6">
        {activeTab === "analyze" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto">
            {/* LEFT COLUMN — Input */}
            <div className="space-y-6">
              <VideoCapture onFileSelect={setFile} disabled={loading} />

              {result && <ChordDisplay analysis={result.analysis} />}

              <AnalyzeButton
                onClick={handleAnalyze}
                loading={loading}
                disabled={!file}
                progress={progress}
              />

              {error && (
                <div className="p-3 rounded-lg bg-accent-red/10 border border-accent-red/30">
                  <p className="text-xs text-accent-red">{error}</p>
                </div>
              )}

              {/* Session info */}
              {result && (
                <div className="flex items-center justify-between text-xs text-text-muted px-1">
                  <span>Genre: <span className="text-text-secondary capitalize">{result.analysis.genre}</span></span>
                  <span>Model: <span className="text-text-secondary">Sonnet 4.5</span></span>
                  <span>Session: <span className="text-text-secondary">#{result.session_id}</span></span>
                </div>
              )}
            </div>

            {/* RIGHT COLUMN — AI Output */}
            <div className="space-y-6">
              {result ? (
                <>
                  <ChordPanel
                    chordResponse={result.chord_suggestions}
                    onAccept={handleAccept}
                    onReject={handleReject}
                    onPlay={handlePlaySuggestion}
                  />

                  <SoloViewer
                    solo={result.solo}
                    midiUrl={result.midi_urls.solo}
                    playing={playing}
                    currentTime={currentTime}
                    onPlay={handlePlaySolo}
                    onStop={stop}
                    onExport={handleExport}
                  />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-text-muted">
                  <div className="text-center space-y-3">
                    <div className="w-16 h-16 mx-auto rounded-full bg-bg-surface-2 flex items-center justify-center">
                      <svg className="w-8 h-8 opacity-30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M9 18V5l12-2v13" />
                        <circle cx="6" cy="18" r="3" />
                        <circle cx="18" cy="16" r="3" />
                      </svg>
                    </div>
                    <p className="text-sm">Upload a piano video to get started</p>
                    <p className="text-xs text-text-muted">AI will analyze your piano playing and suggest chord substitutions + generate solos</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === "history" && (
          <div className="max-w-4xl mx-auto text-center py-20">
            <p className="text-text-muted">Session history coming soon...</p>
          </div>
        )}

        {activeTab === "journal" && (
          <div className="max-w-4xl mx-auto text-center py-20">
            <p className="text-text-muted">Practice journal coming soon...</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border-subtle px-6 py-3 flex items-center justify-between text-xs text-text-muted">
        <span>JamSight AI v1.0 — Hackathon 2026</span>
        <span>Powered by Claude Sonnet</span>
      </footer>
    </div>
  );
}
