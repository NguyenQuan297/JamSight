"""JamSight AI — Piano-focused Backend

Video/Audio → Piano Analysis → Claude AI → Chord Suggestions + Two-Hand Solo MIDI
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from models.schemas import (
    AnalyzeRequest, FeedbackEntry, HealthResponse,
)
from services.audio_analyzer import extract_audio, analyze_audio
from services.ai_orchestrator import get_chord_suggestions, get_solo
from services.midi_builder import solo_json_to_midi, chords_to_midi, combined_midi
from services.context_assembler import get_exemplar, get_available_genres, assemble_context
from services.feedback_trainer import log_feedback, get_feedback_stats, build_adaptive_prompt
from mcp_client.jam_sessions import fetch_exemplar_from_mcp

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jamsight")

# Directories
UPLOAD_DIR = Path(tempfile.gettempdir()) / "jamsight_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MIDI_DIR = Path(tempfile.gettempdir()) / "jamsight_midi"
MIDI_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="JamSight AI",
    description="Video → AI chord suggestions + solo generation",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cleanup_files(*paths: str):
    """Background task to clean up temporary files after response."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.get("/api/genres")
async def list_genres():
    return {"genres": get_available_genres()}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    genre: str = Form("blues"),
    instrument: str = Form("guitar"),
):
    """Main endpoint: upload video → get analysis + suggestions + solo."""
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[{session_id}] Analyze request: genre={genre}, instrument={instrument}, file={file.filename}")

    # Validate file type
    if file.content_type and not file.content_type.startswith(("video/", "audio/")):
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use MP4, MOV, or audio files.")

    # Save upload
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    video_path = str(UPLOAD_DIR / f"{session_id}{suffix}")
    try:
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(413, "File too large. Maximum 100MB.")
        with open(video_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save upload: {e}")

    # Step 1: Extract audio
    try:
        wav_path = extract_audio(video_path)
    except RuntimeError as e:
        raise HTTPException(422, str(e))

    # Step 2: Analyze audio
    try:
        analysis = analyze_audio(wav_path)
        analysis["genre"] = genre
        analysis["instrument"] = instrument
    except Exception as e:
        logger.error(f"[{session_id}] Audio analysis failed: {e}")
        raise HTTPException(422, f"Audio analysis failed: {e}")

    # Step 3: Get exemplar context
    exemplar = fetch_exemplar_from_mcp(genre)

    # Step 4: AI generation (chord suggestions + solo)
    try:
        chord_response = get_chord_suggestions(analysis, exemplar)
    except Exception as e:
        logger.error(f"[{session_id}] Chord suggestion failed: {e}")
        chord_response = {
            "original_progression": analysis["chords"],
            "key": analysis["key"],
            "suggestions": [],
            "theory_note": f"AI suggestion temporarily unavailable: {e}",
        }

    try:
        solo_response = get_solo(analysis, exemplar)
    except Exception as e:
        logger.error(f"[{session_id}] Solo generation failed: {e}")
        solo_response = {
            "title": "Solo generation failed",
            "tempo": analysis["bpm"],
            "time_signature": analysis.get("time_sig", "4/4"),
            "bars": 0,
            "notes": [],
            "phrase_notes": [f"Error: {e}"],
        }

    # Step 5: Build MIDI files
    midi_urls = {}
    try:
        solo_midi_path = str(MIDI_DIR / f"{session_id}_solo.mid")
        solo_json_to_midi(solo_response, solo_midi_path)
        midi_urls["solo"] = f"/api/download/{session_id}_solo.mid"
    except Exception as e:
        logger.error(f"[{session_id}] Solo MIDI build failed: {e}")

    try:
        chords_midi_path = str(MIDI_DIR / f"{session_id}_chords.mid")
        chords_to_midi(analysis["chords"], analysis["bpm"], chords_midi_path)
        midi_urls["chords"] = f"/api/download/{session_id}_chords.mid"
    except Exception as e:
        logger.error(f"[{session_id}] Chord MIDI build failed: {e}")

    try:
        combined_path = str(MIDI_DIR / f"{session_id}_combined.mid")
        combined_midi(solo_response, analysis["chords"], combined_path)
        midi_urls["combined"] = f"/api/download/{session_id}_combined.mid"
    except Exception as e:
        logger.error(f"[{session_id}] Combined MIDI build failed: {e}")

    logger.info(f"[{session_id}] Analysis complete: {len(analysis['chords'])} chords, "
                f"{len(solo_response.get('notes', []))} solo notes")

    return {
        "session_id": session_id,
        "analysis": analysis,
        "chord_suggestions": chord_response,
        "solo": solo_response,
        "midi_urls": midi_urls,
    }


@app.get("/api/download/{filename}")
async def download_midi(filename: str):
    """Download a generated MIDI file."""
    # Sanitize filename
    safe_name = Path(filename).name
    file_path = MIDI_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "MIDI file not found")
    return FileResponse(
        str(file_path),
        media_type="audio/midi",
        filename=safe_name,
    )


@app.post("/api/feedback")
async def submit_feedback(entry: FeedbackEntry):
    """Log user feedback on a chord suggestion."""
    try:
        feedback_id = log_feedback(
            session_id=entry.session_id,
            progression=entry.input_progression,
            genre=entry.genre,
            suggestion=entry.suggestion_shown,
            rank=entry.suggestion_rank,
            action=entry.user_action,
            rating=entry.rating,
        )
        return {"status": "ok", "feedback_id": feedback_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to log feedback: {e}")


@app.get("/api/feedback/stats")
async def feedback_stats():
    """Get feedback analytics."""
    try:
        return get_feedback_stats()
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
