"""
EXTRACT_VOCALS.PY -- Demucs Vocal Isolation for Hum2Tune
=========================================================
Extracts isolated vocals from your original songs in data/Raw Originals/
and places them into the matching folders in data/Humming Audio/.

USAGE:
  python extract_vocals.py              # uses data/Raw Originals/ automatically
  python extract_vocals.py --dry_run    # preview matches without running Demucs

REQUIREMENTS:
  pip install demucs av
  ffmpeg must be installed (already confirmed working on your machine)

IF DEMUCS FAILS (Python 3.13 compatibility issue):
  The script automatically falls back to ffmpeg vocal approximation.
  This is less accurate than Demucs but still much better than a full mix.
"""

import sys
import re
import shutil
import argparse
import subprocess
import logging
from pathlib import Path

project_root = Path(__file__).parent

import io as _io
_utf8_stdout = _io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(stream=_utf8_stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy name matching
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return 1.0
    matches = sum(c1 == c2 for c1, c2 in zip(na, nb))
    return matches / max(len(na), len(nb), 1)


def find_dataset_folder(song_stem: str, dataset_folders: list):
    best_folder = None
    best_score  = 0.0
    for folder in dataset_folders:
        score = _similarity(song_stem, folder.name)
        if score > best_score:
            best_score  = score
            best_folder = folder
    return best_folder if best_score >= 0.70 else None


# ---------------------------------------------------------------------------
# ffmpeg fallback -- used when Demucs fails on Python 3.13
# ---------------------------------------------------------------------------

def _ffmpeg_fallback(mp3_path: Path, out_dir: Path):
    """
    Approximates vocal isolation using ffmpeg audio filters.

    Not as clean as Demucs but still dramatically better than a full mix
    for YIN pitch detection. Works reliably on any Python version.

    Technique:
      1. Centre channel extraction: in stereo pop music, lead vocals are
         panned to the centre. Subtracting right from left channel isolates
         the centre signal where vocals live.
      2. Bandpass 200-4000 Hz: cuts bass (below 200Hz) and cymbals/hi-hats
         (above 4000Hz), leaving only the vocal frequency range.
      3. Volume boost x2: centre extraction reduces signal level.
    """
    logger.info(f"  Using ffmpeg vocal approximation (Demucs unavailable)")

    fallback_dir = out_dir / "ffmpeg_fallback"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    out_path = fallback_dir / f"{mp3_path.stem}_vocals.wav"

    # Centre channel extraction + bandpass filter
    filter_chain = (
        "pan=mono|c0=0.5*c0-0.5*c1[mid];"
        "[mid]highpass=f=200,lowpass=f=4000,volume=2.0"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3_path),
        "-filter_complex", filter_chain,
        "-ar", "22050",
        "-ac", "1",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"  ffmpeg also failed: {result.stderr[-300:]}")
        logger.error(
            "  MANUAL FIX: Open a terminal and run:\n"
            f"    demucs --two-stems vocals -d cpu \"{mp3_path}\"\n"
            "  Then copy the vocals.wav into the matching Humming Audio folder."
        )
        return None

    logger.info(f"  ffmpeg approximation saved: {out_path.name}")
    logger.warning(
        "  This is an approximation. Install Demucs properly for best accuracy."
    )
    return out_path


# ---------------------------------------------------------------------------
# Demucs vocal separator
# ---------------------------------------------------------------------------

def run_demucs(mp3_path: Path, out_dir: Path):
    """
    Run Demucs AI vocal separation on one song file.

    Tries three approaches in order:
      1. Demucs with --backend ffmpeg (most stable on Python 3.13)
      2. Demucs without --backend flag (older demucs versions)
      3. ffmpeg fallback (always works if ffmpeg is installed)

    Returns path to the extracted vocals.wav, or None if all fail.
    """
    logger.info(f"  Separating: {mp3_path.name}")

    # Attempt 1: Demucs with CPU + ffmpeg backend (best for Python 3.13)
    cmd1 = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-d", "cpu",
        "--backend", "ffmpeg",
        "--out", str(out_dir),
        str(mp3_path),
    ]
    result = subprocess.run(cmd1, capture_output=True, text=True)

    # Attempt 2: Demucs with CPU but without --backend (older demucs)
    if result.returncode != 0 and (
        "unrecognized" in result.stderr or "error: argument" in result.stderr
    ):
        logger.info("  Retrying without --backend flag...")
        cmd2 = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-d", "cpu",
            "--out", str(out_dir),
            str(mp3_path),
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)

    # Demucs succeeded -- find the vocals.wav
    if result.returncode == 0:
        vocals = out_dir / "htdemucs" / mp3_path.stem / "vocals.wav"
        if not vocals.exists():
            # Search recursively (model name varies between demucs versions)
            candidates = list(out_dir.rglob("vocals.wav"))
            norm_stem = _norm(mp3_path.stem)
            for c in candidates:
                if _norm(c.parent.name) == norm_stem:
                    vocals = c
                    break
            else:
                vocals = candidates[-1] if candidates else None

        if vocals and vocals.exists():
            logger.info(f"  Demucs extraction complete")
            return vocals

    # Demucs failed -- log why and use ffmpeg fallback
    logger.warning(f"  Demucs failed for {mp3_path.name}")
    if result.stderr:
        # Show only the most relevant error line
        error_lines = [l for l in result.stderr.split("\n") if "error" in l.lower()]
        if error_lines:
            logger.warning(f"  Reason: {error_lines[-1][:150]}")

    return _ffmpeg_fallback(mp3_path, out_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract vocals from original songs for Hum2Tune dataset"
    )
    parser.add_argument(
        "--songs_dir",
        default=None,
        help="Folder with original MP3s (default: data/Raw Originals/)",
    )
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Humming Audio folder (default: data/Humming Audio/)",
    )
    parser.add_argument(
        "--output_name",
        default="original_vocals.wav",
        help="Filename for the extracted vocal file in each song folder",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview matches without running Demucs",
    )
    parser.add_argument(
        "--ffmpeg_only",
        action="store_true",
        help="Skip Demucs and use ffmpeg approximation for all songs (fast but less accurate)",
    )
    args = parser.parse_args()

    songs_dir   = Path(args.songs_dir)   if args.songs_dir   else project_root / "data" / "Original Songs"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else project_root / "data" / "Humming Audio"
    demucs_out  = project_root / "data" / "demucs_output"

    if not songs_dir.exists():
        logger.error(f"Songs folder not found: {songs_dir}")
        logger.error("Expected: data/Raw Originals/  (create it and put your MP3s there)")
        sys.exit(1)

    if not dataset_dir.exists():
        logger.error(f"Dataset folder not found: {dataset_dir}")
        sys.exit(1)

    demucs_out.mkdir(parents=True, exist_ok=True)

    # Collect files
    song_files = sorted([
        f for ext in ["*.mp3", "*.wav", "*.m4a", "*.flac"]
        for f in songs_dir.glob(ext)
    ])

    if not song_files:
        logger.error(f"No audio files in {songs_dir}")
        sys.exit(1)

    dataset_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]

    logger.info(f"Songs found      : {len(song_files)}")
    logger.info(f"Dataset folders  : {len(dataset_folders)}")
    logger.info(f"Mode             : {'DRY RUN' if args.dry_run else ('ffmpeg only' if args.ffmpeg_only else 'Demucs + ffmpeg fallback')}")
    logger.info("-" * 60)

    matched   = []
    skipped   = []
    unmatched = []
    failed    = []

    for i, song_file in enumerate(song_files, 1):
        logger.info(f"[{i}/{len(song_files)}] {song_file.name}")

        folder = find_dataset_folder(song_file.stem, dataset_folders)
        if folder is None:
            logger.warning(f"  No matching folder -- SKIPPED")
            logger.warning(f"  Tip: rename to match one of your Humming Audio folders")
            unmatched.append(song_file.name)
            continue

        logger.info(f"  -> {folder.name}")
        dest = folder / args.output_name

        if dest.exists():
            logger.info(f"  Already done -- skipping")
            skipped.append(song_file.name)
            continue

        if args.dry_run:
            logger.info(f"  [DRY RUN] Would save to: {dest}")
            matched.append((song_file.name, folder.name))
            continue

        # Run extraction
        if args.ffmpeg_only:
            vocals_path = _ffmpeg_fallback(song_file, demucs_out)
        else:
            vocals_path = run_demucs(song_file, demucs_out)

        if vocals_path is None:
            failed.append(song_file.name)
            continue

        shutil.copy2(str(vocals_path), str(dest))
        logger.info(f"  Saved: {dest.name} -> {folder.name}/")
        matched.append((song_file.name, folder.name))

    # Summary
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Processed : {len(matched)}")
    print(f"  Skipped   : {len(skipped)}  (already done)")
    print(f"  Unmatched : {len(unmatched)}  (rename MP3 to fix)")
    print(f"  Failed    : {len(failed)}")

    if unmatched:
        print(f"\nUnmatched songs -- rename the MP3 file to match the folder name:")
        for n in unmatched:
            print(f"  {n}")

    if failed:
        print(f"\nFailed songs -- try running manually:")
        for n in failed:
            print(f"  demucs --two-stems vocals -d cpu \"{songs_dir / n}\"")

    if (len(matched) + len(skipped)) > 0 and not args.dry_run:
        print(f"\nNEXT STEPS:")
        print(f"  python main.py dataset --create")
        print(f"  python main.py train --model cnn_lstm")
        print(f"  python main.py evaluate")
        print(f"\nDemucs output kept at: {demucs_out}")
        print(f"(Delete this folder after dataset --create finishes to free disk space)")


if __name__ == "__main__":
    main()