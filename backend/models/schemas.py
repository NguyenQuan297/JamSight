from pydantic import BaseModel, Field
from typing import Optional


class AnalysisResult(BaseModel):
    bpm: int = Field(description="Detected tempo in BPM")
    key: str = Field(description="Detected musical key, e.g. 'A minor'")
    chords: list[str] = Field(description="Detected chord progression")
    time_sig: str = Field(default="4/4", description="Time signature")
    genre: str = Field(default="blues", description="Selected genre")
    instrument: str = Field(default="guitar", description="Target instrument")
    duration: float = Field(description="Audio duration in seconds")
    midi_notes: list[float] = Field(default_factory=list, description="Detected MIDI note numbers")


class ChordChange(BaseModel):
    position: int
    original: str
    replacement: str
    reason: str


class ChordSuggestion(BaseModel):
    rank: int
    label: str
    progression: list[str]
    changes: list[ChordChange]
    overall_effect: str
    difficulty: str = Field(description="beginner | intermediate | advanced")


class ChordResponse(BaseModel):
    original_progression: list[str]
    key: str
    suggestions: list[ChordSuggestion]
    theory_note: str


class SoloNote(BaseModel):
    bar: int
    beat: float
    pitch: int
    duration: float
    velocity: int
    note_name: str
    function: str = ""


class SoloResponse(BaseModel):
    title: str
    tempo: int
    time_signature: str = "4/4"
    bars: int = 8
    notes: list[SoloNote]
    phrase_notes: list[str] = Field(default_factory=list)


class FeedbackEntry(BaseModel):
    session_id: str
    input_progression: list[str]
    genre: str
    suggestion_shown: dict
    suggestion_rank: int
    user_action: str = Field(description="accepted | rejected | ignored")
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class AnalyzeRequest(BaseModel):
    genre: str = "blues"
    instrument: str = "guitar"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
