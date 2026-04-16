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
│   ├── main.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── audio_analyzer.py
│   │   ├── ai_orchestrator.py
│   │   ├── midi_builder.py
│   │   ├── context_assembler.py
│   │   └── feedback_trainer.py
│   ├── mcp_client/
│   │   └── jam_sessions.py
│   └── train/
│       ├── download_data.py
│       ├── prepare_data.py
│       ├── augment.py
│       ├── chord_classifier.py
│       ├── evaluate_model.py
│       └── feedback_to_training.py
│
└── frontend/
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── index.css
        ├── types/
        │   └── music.ts
        ├── hooks/
        │   ├── useVideoAnalysis.ts
        │   └── useMidiPlayback.ts
        └── components/
            ├── Header.tsx
            ├── GenreSelector.tsx
            ├── VideoCapture.tsx
            ├── ChordDisplay.tsx
            ├── AnalyzeButton.tsx
            ├── ChordPanel.tsx
            └── SoloViewer.tsx
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
