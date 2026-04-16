"""Piano-focused genre exemplar database and context assembly."""

EXEMPLAR_DB: dict[str, list[dict]] = {
    "blues": [
        {
            "title": "Blues Piano in A — Slow Burn",
            "progression": ["A7", "A7", "A7", "A7", "D7", "D7", "A7", "A7", "E7", "D7", "A7", "E7"],
            "key": "A major (blues)",
            "description": "12-bar blues with left-hand boogie-woogie bass pattern. Right hand plays call-and-response licks.",
            "style_notes": "Left hand: walking bass or boogie pattern (root-5th-6th-5th). "
                          "Right hand: blue notes (b3, b5, b7), grace notes, tremolo. "
                          "Think Ray Charles, Otis Spann, Pinetop Perkins.",
            "techniques": ["boogie-woogie bass", "grace notes", "tremolo", "blue notes", "crushed notes"],
            "voicings": "LH: root + 5th shell. RH: 3rd + b7 + 9th. Blues = dominant 7th everywhere.",
        },
        {
            "title": "Minor Blues Piano — The Thrill Is Gone",
            "progression": ["Am7", "Am7", "Am7", "Am7", "Dm7", "Dm7", "Am7", "Am7", "Fmaj7", "E7b9", "Am7", "E7b9"],
            "key": "A minor",
            "description": "Minor blues with jazz extensions. Sustained, emotional phrasing.",
            "style_notes": "Left hand: whole note shells, gentle. Right hand: sustained melodies, "
                          "slow bends simulated with grace notes. Think B.B. King piano arrangement.",
            "techniques": ["shell voicings", "sustained melody", "minor pentatonic", "dorian color"],
            "voicings": "LH: A2+G3 (root+b7). RH: C4+E4+G4 (min triad). Add B4 for Am9.",
        },
    ],
    "jazz": [
        {
            "title": "Jazz Piano ii-V-I — Bill Evans Style",
            "progression": ["Dm7", "G7", "Cmaj7", "Cmaj7"],
            "key": "C major",
            "description": "The foundation of jazz piano. Rootless voicings in right hand, bass in left.",
            "style_notes": "Left hand: root only or root+5th. Right hand: rootless voicings "
                          "(3-7-9-5 or 7-3-5-9). Voice lead by half steps between chords. "
                          "Think Bill Evans, Ahmad Jamal, Keith Jarrett.",
            "techniques": ["rootless voicings", "voice leading", "chord-tone targeting", "block chords"],
            "voicings": "Dm7: LH D2, RH F4+A4+C5+E5. G7: LH G2, RH F4+A4+B4+D5. "
                       "Cmaj7: LH C2, RH E4+G4+B4+D5.",
        },
        {
            "title": "Jazz Ballad — Autumn Leaves",
            "progression": ["Am7", "D7", "Gmaj7", "Cmaj7", "F#m7b5", "B7b9", "Em", "Em"],
            "key": "E minor / G major",
            "description": "Circle of fifths descending. Each chord flows into the next.",
            "style_notes": "Rubato feel. Left hand: gentle arpeggios. Right hand: melody with extensions. "
                          "Build dynamics gradually. Sustain pedal essential.",
            "techniques": ["arpeggiation", "rubato", "pedal technique", "inner voice movement"],
            "voicings": "Use drop-2 voicings for warmth. Am7: RH C4+E4+G4+A4 (drop 2).",
        },
    ],
    "pop": [
        {
            "title": "Pop Piano Ballad — Axis Progression",
            "progression": ["C", "G", "Am", "F"],
            "key": "C major",
            "description": "The four-chord song on piano. Arpeggiated right hand, simple left hand bass.",
            "style_notes": "Left hand: octave bass notes (C2+C3). Right hand: arpeggiated pattern "
                          "(8th notes: root-3rd-5th-8va-5th-3rd). Add sus2 or add9 for color. "
                          "Think Adele, Sam Smith, Coldplay piano parts.",
            "techniques": ["arpeggiation", "octave bass", "add9 voicings", "dynamic build"],
            "voicings": "C: LH C2+C3, RH E4+G4+C5 arpeggio. Am: LH A2+A3, RH C4+E4+A4.",
        },
        {
            "title": "Emotional Pop — Minor Start",
            "progression": ["Am", "F", "C", "G"],
            "key": "A minor / C major",
            "description": "Minor-first rotation for melancholic feel.",
            "style_notes": "Broken chord pattern. Build from sparse verse (quarter notes) to "
                          "full chorus (8th note arpeggios + octave bass). Pedal on every bar.",
            "techniques": ["dynamic build", "broken chords", "pedal tones", "inversions"],
            "voicings": "Use G/B between Am and C for smooth bass: A-F-G/B-C.",
        },
    ],
    "rock": [
        {
            "title": "Rock Piano — Power Chords on Keys",
            "progression": ["Am", "G", "F", "C"],
            "key": "A minor",
            "description": "Driving rock piano. Octaves in left hand, block chords in right.",
            "style_notes": "Left hand: driving octaves on every beat. Right hand: block chords, "
                          "staccato on 2 and 4. Think Ben Folds, Billy Joel rock numbers.",
            "techniques": ["octave bass", "block chords", "staccato accents", "driving rhythm"],
            "voicings": "Keep voicings simple — triads. Power comes from rhythm, not extensions.",
        },
    ],
    "funk": [
        {
            "title": "Funk Clavinet Piano — One Chord Vamp",
            "progression": ["E9", "E9", "E9", "E9"],
            "key": "E mixolydian",
            "description": "One-chord funk groove. Rhythmic staccato pattern is everything.",
            "style_notes": "Clavinet-style playing on piano: staccato, percussive, 16th note patterns. "
                          "Ghost notes (very soft staccato hits). Emphasize 'and' of beats. "
                          "Think Stevie Wonder, Herbie Hancock Headhunters.",
            "techniques": ["16th note comping", "ghost notes", "staccato", "syncopation"],
            "voicings": "E9: LH E2, RH G#4+B4+D5+F#5. Mute between hits for percussive feel.",
        },
    ],
}


def get_exemplar(genre: str) -> str:
    """Get formatted exemplar string for a genre — piano focused."""
    genre_lower = genre.lower()
    exemplars = EXEMPLAR_DB.get(genre_lower, EXEMPLAR_DB["blues"])

    parts = []
    for ex in exemplars:
        parts.append(
            f"### {ex['title']}\n"
            f"Progression: {' - '.join(ex['progression'])}\n"
            f"Key: {ex['key']}\n"
            f"Description: {ex['description']}\n"
            f"Style: {ex['style_notes']}\n"
            f"Voicings: {ex.get('voicings', 'N/A')}\n"
            f"Techniques: {', '.join(ex['techniques'])}\n"
        )

    return "\n".join(parts)


def assemble_context(analysis: dict, genre: str) -> str:
    """Assemble full context string for AI prompts."""
    exemplar = get_exemplar(genre)

    return (
        f"=== DETECTED PIANO PERFORMANCE ===\n"
        f"Chords: {' - '.join(analysis['chords'])}\n"
        f"Key: {analysis['key']}\n"
        f"Tempo: {analysis['bpm']} BPM\n"
        f"Time Signature: {analysis.get('time_sig', '4/4')}\n"
        f"Duration: {analysis['duration']}s\n"
        f"Genre: {genre}\n"
        f"Instrument: Piano\n\n"
        f"=== PIANO GENRE EXEMPLARS ===\n"
        f"{exemplar}"
    )


def get_available_genres() -> list[dict]:
    """Return list of available genres with metadata."""
    return [
        {"id": genre, "name": genre.capitalize(), "exemplar_count": len(exs)}
        for genre, exs in EXEMPLAR_DB.items()
    ]
