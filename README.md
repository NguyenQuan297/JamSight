# JamSight

Upload a video of yourself playing piano — JamSight detects your chord progression, suggests smarter reharmonizations, and generates a custom 8-bar AI solo in your style.

**Stack:** Python · FastAPI · React · TypeScript · Claude Sonnet · Tone.js · PyTorch · ONNX

---

## How it works

<img width="1440" height="1250" alt="image" src="https://github.com/user-attachments/assets/05d31f83-09ab-40a5-b9b5-4be0fda09ab6" />


Genre context is optionally enriched by the **ai-jam-sessions MCP server** — a library of 120 annotated songs used as musical reference for Claude's prompts.

---

## Architecture

<img width="1440" height="1232" alt="image" src="https://github.com/user-attachments/assets/ab85a4b0-8c0a-4c89-9756-c53edae6eea7" />



---

## Data Flow

<img width="1440" height="1020" alt="image" src="https://github.com/user-attachments/assets/f166c830-f750-407e-813e-d3220d5e23d6" />




---

## Code structure

```
JamSight/
├── .env.example
├── docker-compose.yml
├── README.md
│
├── backend/
│   ├── main.py                      FastAPI app
│   │                                  POST /api/analyze
│   │                                  POST /api/feedback
│   │                                  GET  /api/download/{filename}
│   │                                  GET  /api/genres
│   │                                  GET  /api/health
│   │
│   ├── models/
│   │   └── schemas.py               Pydantic: AnalysisResult · ChordResponse
│   │                                          SoloResponse · FeedbackRequest
│   │
│   ├── services/
│   │   ├── audio_analyzer.py        FFmpeg → librosa chroma → chord/key/BPM
│   │   │                            PianoChordPredictor (ONNX, 36-dim → 96 classes)
│   │   │                            graceful fallback when model not present
│   │   │
│   │   ├── ai_orchestrator.py       Claude Sonnet (prompt-cached)
│   │   │                            build_chord_prompt() → 3 piano reharmonizations
│   │   │                              LH shell voicings + RH extensions + voice leading
│   │   │                            build_solo_prompt() → 8-bar two-hand solo
│   │   │                              RH melody MIDI 60–96 · LH bass MIDI 36–60
│   │   │
│   │   ├── midi_builder.py          pretty_midi rendering
│   │   │                            solo_json_to_midi()  → 2 tracks (RH + LH)
│   │   │                            chords_to_midi()     → backing voicings
│   │   │                            combined_midi()      → 4 tracks (solo + backing)
│   │   │
│   │   ├── context_assembler.py     Piano genre exemplars
│   │   │                            voicings · techniques · style notes
│   │   │                            for: blues · jazz · pop · rock · funk
│   │   │
│   │   └── feedback_trainer.py      SQLite feedback log
│   │                                adaptive few-shot injection into prompts
│   │
│   ├── mcp_client/
│   │   └── jam_sessions.py          stdio bridge → ai-jam-sessions
│   │                                fallback to local context_assembler
│   │
│   └── train/
│       ├── download_data.py         Auto-download MAESTRO v3 (57MB · 1,276 MIDIs)
│       ├── prepare_data.py          MIDI → 2s windows → 36-dim feature vectors
│       │                              [0:12]  chroma histogram
│       │                              [12:24] velocity-weighted chroma
│       │                              [24:30] piano-specific (hand balance, range, sustain)
│       │                              [30:36] temporal (intervals, runs, density)
│       ├── chord_classifier.py      PianoMLP: 36→128→256→128→96
│       │                              residual · LayerNorm · GELU
│       │                              weighted sampler · cosine LR · label smoothing
│       │                              4-config ablation · ONNX export
│       ├── evaluate_model.py        per-class accuracy · confusion matrix · SVM baseline
│       └── feedback_to_training.py  feedback DB → DPO pairs + few-shot curation
│
└── frontend/
    ├── vite.config.ts               proxy /api → localhost:8000
    └── src/
        ├── App.tsx                  2-column layout: input | AI output
        ├── index.css                dark theme · piano roll grid
        ├── types/
        │   └── music.ts             AnalysisResult · SoloNote (hand · technique)
        ├── hooks/
        │   ├── useVideoAnalysis.ts  POST /api/analyze · feedback submission
        │   └── useMidiPlayback.ts   Tone.js browser playback
        └── components/
            ├── Header.tsx           logo · Analyze/History/Journal tabs
            ├── GenreSelector.tsx    Blues · Jazz · Pop · Rock · Funk pills
            ├── VideoCapture.tsx     drag-drop MP4 upload + video preview
            ├── ChordDisplay.tsx     chord pills · BPM · Key · Time · Duration
            ├── AnalyzeButton.tsx    "Analyze & Generate" CTA
            ├── ChordPanel.tsx       3 suggestion cards (Beginner/Intermediate/Advanced)
            │                        LH/RH voicing details · accept/reject/play
            └── SoloViewer.tsx       canvas piano roll (RH=purple · LH=orange · C4 divider)
                                     Tone.js playback · Export MIDI
```

---

## Quick start

**Requirements:** Docker · Node.js 18+ · Python 3.11+

```bash
# 1. Clone
git clone https://github.com/your-org/jamsight.git
cd jamsight

# 2. Add API key
cp .env.example .env
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
docker-compose up --build
```

Open **http://localhost:5173**

> Without a trained ONNX model, the app runs automatically with librosa heuristic chord detection — all features work, accuracy is lower.

---

## Manual setup (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py
# → http://localhost:8000
# → API docs: http://localhost:8000/docs

# Frontend  (new terminal)
cd frontend
npm install
npm run dev
# → http://localhost:5173

# MCP server  (optional — richer genre context for Claude)
npx -y -p @mcptoolshop/ai-jam-sessions ai-jam-sessions-mcp
```

---

## License

MIT
