export interface AnalysisResult {
  bpm: number;
  key: string;
  chords: string[];
  time_sig: string;
  genre: string;
  instrument: string;
  duration: number;
  midi_notes: number[];
}

export interface ChordChange {
  position: number;
  original: string;
  replacement: string;
  reason: string;
}

export interface ChordSuggestion {
  rank: number;
  label: string;
  progression: string[];
  changes: ChordChange[];
  overall_effect: string;
  difficulty: "beginner" | "intermediate" | "advanced";
}

export interface ChordResponse {
  original_progression: string[];
  key: string;
  suggestions: ChordSuggestion[];
  theory_note: string;
}

export interface SoloNote {
  bar: number;
  beat: number;
  pitch: number;
  duration: number;
  velocity: number;
  note_name: string;
  function: string;
  hand: "right" | "left";
  technique: "normal" | "sustain" | "grace" | "staccato" | "legato";
}

export interface SoloResponse {
  title: string;
  tempo: number;
  time_signature: string;
  bars: number;
  notes: SoloNote[];
  phrase_notes: string[];
}

export interface MidiUrls {
  solo?: string;
  chords?: string;
  combined?: string;
}

export interface AnalyzeResponse {
  session_id: string;
  analysis: AnalysisResult;
  chord_suggestions: ChordResponse;
  solo: SoloResponse;
  midi_urls: MidiUrls;
}

export interface FeedbackEntry {
  session_id: string;
  input_progression: string[];
  genre: string;
  suggestion_shown: Record<string, unknown>;
  suggestion_rank: number;
  user_action: "accepted" | "rejected" | "ignored";
  rating?: number;
}

export type Genre = "blues" | "jazz" | "pop" | "rock" | "funk";
export type TabView = "analyze" | "history" | "journal";
