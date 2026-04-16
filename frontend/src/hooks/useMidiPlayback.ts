import { useState, useRef, useCallback } from "react";

export function useMidiPlayback() {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const synthRef = useRef<any>(null);
  const partRef = useRef<any>(null);

  const play = useCallback(async (midiUrl: string) => {
    try {
      const Tone = await import("tone");
      const { Midi } = await import("@tonejs/midi");

      await Tone.start();

      // Fetch and parse MIDI
      const response = await fetch(midiUrl);
      const arrayBuffer = await response.arrayBuffer();
      const midi = new Midi(arrayBuffer);

      // Create synth
      if (synthRef.current) synthRef.current.dispose();
      const synth = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: "triangle" },
        envelope: { attack: 0.02, decay: 0.3, sustain: 0.4, release: 0.8 },
      }).toDestination();
      synth.volume.value = -6;
      synthRef.current = synth;

      // Schedule notes
      const now = Tone.now();
      for (const track of midi.tracks) {
        for (const note of track.notes) {
          synth.triggerAttackRelease(
            note.name,
            note.duration,
            now + note.time,
            note.velocity
          );
        }
      }

      setPlaying(true);

      // Track time
      const totalDuration = midi.duration;
      const interval = setInterval(() => {
        const elapsed = Tone.now() - now;
        setCurrentTime(elapsed);
        if (elapsed >= totalDuration) {
          clearInterval(interval);
          setPlaying(false);
          setCurrentTime(0);
        }
      }, 100);

      // Store interval for cleanup
      partRef.current = interval;
    } catch (err) {
      console.error("MIDI playback error:", err);
      setPlaying(false);
    }
  }, []);

  const stop = useCallback(async () => {
    const Tone = await import("tone");
    Tone.getTransport().stop();
    Tone.getTransport().cancel();
    if (synthRef.current) {
      synthRef.current.releaseAll();
    }
    if (partRef.current) {
      clearInterval(partRef.current);
    }
    setPlaying(false);
    setCurrentTime(0);
  }, []);

  return { playing, currentTime, play, stop };
}
