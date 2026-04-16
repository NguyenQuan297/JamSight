"""Download piano MIDI datasets for chord classifier training.

Primary: MAESTRO v3.0 (57MB MIDI-only, public, auto-download)
Optional: GiantMIDI-Piano (requires manual request)
"""

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"


def download_maestro():
    """Download MAESTRO v3 MIDI-only dataset (57MB, no login needed).

    MAESTRO = MIDI and Audio Edited for Synchronous TRacks and Organization
    - 1,276 piano performances from international piano competitions
    - High-quality, aligned MIDI from Yamaha Disklavier
    - ~200 hours of music
    """
    dest = DATA / "maestro"
    dest.mkdir(parents=True, exist_ok=True)

    maestro_dir = dest / "maestro-v3.0.0"
    if maestro_dir.exists():
        midis = list(maestro_dir.rglob("*.midi"))
        print(f"MAESTRO already downloaded: {len(midis)} files")
        return maestro_dir

    url = ("https://storage.googleapis.com/magentadata/datasets/"
           "maestro/v3.0.0/maestro-v3.0.0-midi.zip")

    zip_path = dest / "maestro.zip"
    print(f"Downloading MAESTRO MIDI (~57MB)...")
    print(f"  URL: {url}")

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

    # Cleanup zip
    zip_path.unlink()

    # Load and print metadata
    meta_path = maestro_dir / "maestro-v3.0.0.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        n_files = len(meta.get("midi_filename", {}))
        durations = meta.get("duration", {})
        total_hours = sum(durations.values()) / 3600 if durations else 0
        composers = set(meta.get("canonical_composer", {}).values()) if "canonical_composer" in meta else set()
        print(f"\nMAESTRO ready:")
        print(f"  Files:     {n_files}")
        print(f"  Hours:     {total_hours:.1f}h")
        print(f"  Composers: {len(composers)}")
        if composers:
            for c in sorted(composers)[:10]:
                print(f"    - {c}")
            if len(composers) > 10:
                print(f"    ... and {len(composers) - 10} more")
    else:
        midis = list(maestro_dir.rglob("*.midi"))
        print(f"\nMAESTRO ready: {len(midis)} MIDI files")

    return maestro_dir


def print_giantmidi_instructions():
    """GiantMIDI-Piano requires manual request — print instructions."""
    print("""
GiantMIDI-Piano Dataset
=======================
10,854 piano pieces transcribed from YouTube recordings.
More diverse than MAESTRO (pop, film, contemporary).

How to get it:
1. Visit: https://github.com/bytedance/GiantMIDI-Piano
2. Fill the Google Form to request access
3. You'll receive a download link within ~24 hours
4. Download and extract to: backend/train/data/giantmidi/

While waiting, use MAESTRO to start training — it's sufficient
for a strong chord classifier.
""")


def verify_data():
    """Check what data is available."""
    print("Data directory:", DATA)
    print()

    # MAESTRO
    maestro_dir = DATA / "maestro" / "maestro-v3.0.0"
    if maestro_dir.exists():
        midis = list(maestro_dir.rglob("*.midi"))
        print(f"[OK] MAESTRO: {len(midis)} MIDI files")
    else:
        print("[--] MAESTRO: not downloaded (run: python download_data.py maestro)")

    # GiantMIDI
    giant_dir = DATA / "giantmidi"
    if giant_dir.exists():
        midis = list(giant_dir.glob("*.mid")) + list(giant_dir.glob("*.midi"))
        print(f"[OK] GiantMIDI: {len(midis)} MIDI files")
    else:
        print("[--] GiantMIDI: not available (optional)")

    # Prepared dataset
    dataset_path = DATA / "piano_dataset.json"
    if dataset_path.exists():
        size_mb = dataset_path.stat().st_size / 1_000_000
        print(f"[OK] Prepared dataset: {size_mb:.1f} MB")
    else:
        print("[--] Prepared dataset: not built (run: python prepare_data.py)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "maestro"

    if cmd == "maestro":
        download_maestro()
    elif cmd == "giantmidi":
        print_giantmidi_instructions()
    elif cmd == "verify":
        verify_data()
    elif cmd == "all":
        download_maestro()
        print()
        print_giantmidi_instructions()
    else:
        print(f"Usage: python download_data.py [maestro|giantmidi|verify|all]")
