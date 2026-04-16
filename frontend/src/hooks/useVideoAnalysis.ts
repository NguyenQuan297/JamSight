import { useState, useCallback } from "react";
import type { AnalyzeResponse, FeedbackEntry, Genre } from "../types/music";

export function useVideoAnalysis() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  const analyze = useCallback(async (file: File, genre: Genre, instrument: string = "guitar") => {
    setLoading(true);
    setError(null);
    setProgress("Uploading...");

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("genre", genre);
      form.append("instrument", instrument);

      setProgress("Analyzing audio...");

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Server error: ${res.status}`);
      }

      setProgress("Complete!");
      const data: AnalyzeResponse = await res.json();
      setResult(data);
      return data;
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Analysis failed";
      setError(msg);
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const submitFeedback = useCallback(async (feedback: FeedbackEntry) => {
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedback),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress("");
  }, []);

  return { result, loading, error, progress, analyze, submitFeedback, reset };
}
