"""Download datasets for JamSight piano chord detection and solo generation.

Two different problems need different data:
  1. Chord detection  → needs REAL AUDIO + chord labels (MAPS, ChoCo, JAAH)
  2. Solo generation  → needs high-quality MIDI patterns (MAESTRO)

The key insight: training on clean MIDI creates a domain gap with real user
video recordings (noise, reverb, pedal, room acoustics). MAPS solves this
by providing studio AND ambient piano recordings with aligned MIDI.
"""

import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"


# ══════════════════════════════════════════════════════════════════
# Chord detection datasets (real audio → solve domain gap)
# ══════════════════════════════════════════════════════════════════

def download_maps():
    """MAPS: MIDI Aligned Piano Sounds — real audio + aligned MIDI.

    Solves domain gap: includes both studio and ambient recordings
    from different pianos (Bösendorfer, Steinway, Yamaha, etc.)

    Structure:
      AkPnBcht/  ← Bösendorfer Imperial (studio)
      AkPnBsdf/  ← Bösendorfer Imperial (ambient)
      AkPnCGdD/  ← Steinway D (studio)
      AkPnStgb/  ← Steinway B (ambient)
      ENSTDkAm/  ← Yamaha Disklavier (studio)   ← start with this
      ENSTDkCl/  ← Yamaha Disklavier (ambient)  ← and this
      SptkBGAm/  ← Yamaha Upright (studio)
      SptkBGCl/  ← Yamaha Upright (ambient)
      StbgTGd2/  ← Steinway D (studio)

    Total: ~8GB (full) or ~1.2GB (ENSTDkAm + ENSTDkCl only)
    """
    dest = DATA / "maps"
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already present
    if any(dest.iterdir()):
        subdirs = [d.name for d in dest.iterdir() if d.is_dir()]
        wavs = list(dest.rglob("*.wav"))
        print(f"[OK] MAPS found: {len(subdirs)} piano dirs, {len(wavs)} WAV files")
        return

    print("""
╔══════════════════════════════════════════════════════════════╗
║  MAPS Dataset — Manual Download Required                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Request access at:                                       ║
║     https://adasp.telecom-paris.fr/resources/                ║
║            2010-07-08-maps-database/                         ║
║                                                              ║
║  2. Fill form → receive download link (~48h)                 ║
║                                                              ║
║  3. Quick start: download only ENSTDkAm + ENSTDkCl (~1.2GB) ║
║     → Studio + ambient from same Yamaha Disklavier           ║
║     → Enough for initial training                            ║
║                                                              ║
║  4. Extract to: backend/train/data/maps/                     ║
║                                                              ║
║  While waiting, use ChoCo + MAESTRO to start training.       ║
╚══════════════════════════════════════════════════════════════╝
""")


def download_choco():
    """ChoCo: 20,000+ chord annotations from 18 sources via HuggingFace.

    Used for: few-shot examples in Claude prompts (NOT for training classifier).
    Piano-relevant sources: billboard, wikifonia, ireal, weimar, schubert, rwc.
    """
    dest = DATA / "choco"
    dest.mkdir(parents=True, exist_ok=True)

    output = dest / "piano_choco.json"
    if output.exists():
        with open(output) as f:
            data = json.load(f)
        print(f"[OK] ChoCo already processed: {len(data)} entries")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"],
                       check=True, capture_output=True)
        from datasets import load_dataset

    print("Loading ChoCo from HuggingFace...")
    ds = load_dataset("smashub/choco", split="train")

    # Filter piano-relevant sources
    piano_keywords = {
        "billboard", "wikifonia", "ireal", "weimar",
        "schubert", "rwc-classical", "rwc-jazz",
        "jazz", "piano", "rock",
    }

    piano_entries = []
    for entry in ds:
        source = str(entry.get("source", "")).lower()
        if any(kw in source for kw in piano_keywords):
            piano_entries.append({
                "source": entry.get("source", ""),
                "title": entry.get("title", ""),
                "chords": entry.get("chords", []),
                "key": entry.get("key", ""),
                "genre": entry.get("genre", ""),
            })

    # Keep diverse subset
    piano_entries = piano_entries[:5000]

    with open(output, "w") as f:
        json.dump(piano_entries, f)

    print(f"ChoCo total:      {len(ds)} entries")
    print(f"Piano-relevant:   {len(piano_entries)} entries")
    print(f"Saved:            {output}")


def download_jaah():
    """JAAH: Jazz Audio-Aligned Harmony — jazz chord annotations + audio.

    Git repo with JAMS annotation files for jazz recordings.
    Complements MAPS with jazz-specific harmony.
    """
    dest = DATA / "jaah"

    if dest.exists() and (dest / "annotations").exists():
        jams_count = len(list((dest / "annotations").glob("*.jams")))
        print(f"[OK] JAAH already downloaded: {jams_count} annotation files")
        return

    print("Cloning JAAH repository...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/MTG/JAAH", str(dest)],
            check=True, capture_output=True, timeout=120,
        )
        jams_count = len(list((dest / "annotations").glob("*.jams")))
        print(f"[OK] JAAH ready: {jams_count} annotation files")
    except subprocess.CalledProcessError as e:
        print(f"[!!] JAAH clone failed: {e.stderr.decode()[:200]}")
    except FileNotFoundError:
        print("[!!] git not found — install git to download JAAH")


# ══════════════════════════════════════════════════════════════════
# Solo generation dataset (clean MIDI → pattern learning)
# ══════════════════════════════════════════════════════════════════

def download_maestro():
    """MAESTRO v3 MIDI-only (57MB) — for solo generation patterns.

    NOT for chord classifier training (no audio → domain gap).
    Used to learn melodic patterns, phrasing, and piano idioms
    that get injected into Claude's solo generation prompts.
    """
    dest = DATA / "maestro"
    dest.mkdir(parents=True, exist_ok=True)

    maestro_dir = dest / "maestro-v3.0.0"
    if maestro_dir.exists():
        midis = list(maestro_dir.rglob("*.midi"))
        print(f"[OK] MAESTRO already downloaded: {len(midis)} files")
        return maestro_dir

    url = ("https://storage.googleapis.com/magentadata/datasets/"
           "maestro/v3.0.0/maestro-v3.0.0-midi.zip")

    zip_path = dest / "maestro.zip"
    print("Downloading MAESTRO MIDI (~57MB)...")

    def progress(count, block_size, total_size):
        downloaded = count * block_size
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1_000_000
        total_mb = total_size / 1_000_000
        print(f"\r  {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(zip_path), reporthook=progress)
    print()

    print("Extracting...")
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(dest))
    zip_path.unlink()

    midis = list(maestro_dir.rglob("*.midi"))
    print(f"[OK] MAESTRO ready: {len(midis)} MIDI files")
    return maestro_dir


# ══════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════

def verify():
    """Check what data is available and what's missing."""
    print(f"Data directory: {DATA}\n")

    checks = [
        ("MAPS (audio+MIDI)", DATA / "maps",
         lambda p: f"{len(list(p.rglob('*.wav')))} WAV files" if any(p.iterdir()) else None),
        ("ChoCo (chord annotations)", DATA / "choco" / "piano_choco.json",
         lambda p: f"{len(json.load(open(p)))} entries" if p.exists() else None),
        ("JAAH (jazz annotations)", DATA / "jaah" / "annotations",
         lambda p: f"{len(list(p.glob('*.jams')))} files" if p.exists() else None),
        ("MAESTRO (solo MIDI)", DATA / "maestro" / "maestro-v3.0.0",
         lambda p: f"{len(list(p.rglob('*.midi')))} files" if p.exists() else None),
    ]

    for name, path, check_fn in checks:
        try:
            info = check_fn(path)
            if info:
                print(f"  [OK] {name}: {info}")
            else:
                print(f"  [--] {name}: not available")
        except Exception:
            print(f"  [--] {name}: not available")

    # Check prepared datasets
    print()
    for split in ["train", "val", "test"]:
        p = DATA / f"piano_{split}.json"
        if p.exists():
            size = p.stat().st_size / 1_000_000
            print(f"  [OK] piano_{split}.json: {size:.1f} MB")
        else:
            print(f"  [--] piano_{split}.json: not built (run prepare_data.py)")

    examples = DATA / "chord_examples.json"
    if examples.exists():
        with open(examples) as f:
            n = len(json.load(f))
        print(f"  [OK] chord_examples.json: {n} few-shot examples")
    else:
        print(f"  [--] chord_examples.json: not built")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["choco", "jaah", "maestro"]

    dispatch = {
        "maps": download_maps,
        "choco": download_choco,
        "jaah": download_jaah,
        "maestro": download_maestro,
        "verify": verify,
        "all": lambda: (download_choco(), download_jaah(), download_maestro(), download_maps()),
    }

    for target in targets:
        if target in dispatch:
            dispatch[target]()
            print()
        else:
            print(f"Unknown target: {target}")
            print(f"Available: {', '.join(dispatch.keys())}")
