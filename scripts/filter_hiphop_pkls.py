#!/usr/bin/env python
"""Copy only Hip-Hop PKL files to data/hiphop/pkl/ by inspecting a 'genre' field."""
import argparse
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _find_genre(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == "genre":
                try:
                    return str(v).strip()
                except Exception:
                    return None
        for v in obj.values():
            g = _find_genre(v)
            if g is not None:
                return g
    if isinstance(obj, (list, tuple)):
        for v in obj:
            g = _find_genre(v)
            if g is not None:
                return g
    return None


def _genre_from_filename(path: Path) -> str | None:
    name = path.stem
    parts = name.split("_")
    if not parts:
        return None
    token = parts[0]
    if not token.startswith("g") or len(token) < 3:
        return None
    code = token[1:3].upper()
    code_to_genre = {
        "BR": "break",
        "PO": "pop",
        "LO": "lock",
        "MH": "middle_hiphop",
        "LH": "la_hiphop",
        "HO": "house",
        "WA": "waack",
        "KR": "krump",
        "JS": "street_jazz",
        "JB": "ballet_jazz",
    }
    return code_to_genre.get(code)


def parse_aist_basename(name: str) -> dict:
    parts = name.split("_")
    meta = {
        "genre_token": None,
        "genre_code": None,
        "situation": None,
        "camera": None,
        "dancer": None,
        "music": None,
        "choreography": None,
    }
    if len(parts) >= 1:
        meta["genre_token"] = parts[0]
        if parts[0].startswith("g") and len(parts[0]) >= 3:
            meta["genre_code"] = parts[0][1:3].upper()
    if len(parts) >= 2:
        meta["situation"] = parts[1]
    if len(parts) >= 3:
        meta["camera"] = parts[2]
    if len(parts) >= 4:
        meta["dancer"] = parts[3]
    if len(parts) >= 5:
        meta["music"] = parts[4]
    if len(parts) >= 6:
        meta["choreography"] = parts[5]
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Hip-Hop PKL files by genre field")
    parser.add_argument("src_dir", type=Path, help="Directory containing .pkl files")
    parser.add_argument("--dst", type=Path, default=Path("data/hiphop/pkl"), help="Destination directory")
    parser.add_argument("--genre", type=str, default="hiphop", help="Genre name to keep")
    args = parser.parse_args()

    src_dir: Path = args.src_dir
    dst_dir: Path = args.dst
    genre_keep = args.genre.lower()

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    picked, total = 0, 0
    genre_counts: Counter[str] = Counter()
    for pkl_path in sorted(src_dir.glob("*.pkl")):
        total += 1
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            genre = _find_genre(data)
            if genre is None:
                genre = _genre_from_filename(pkl_path)
            if genre is not None:
                genre_counts[genre.strip().lower()] += 1
            # Treat hip-hop umbrella when keeping 'hiphop'
            normalized = genre.strip().lower() if genre is not None else None
            is_hiphop = normalized in {"hiphop", "hip-hop", "middle_hiphop", "la_hiphop"}
            keep = (
                (genre_keep in {"hiphop", "hip-hop"} and is_hiphop)
                or (normalized == genre_keep)
            )
            if normalized is not None and keep:
                shutil.copy2(pkl_path, dst_dir / pkl_path.name)
                picked += 1
        except Exception as e:
            print(f"Skip {pkl_path.name}: {e}")

    if genre_counts:
        print("Detected genres and counts:")
        for g, c in sorted(genre_counts.items()):
            print(f"  {g}: {c}")
    else:
        print("No genre field found in any file.")

    print(f"Kept {picked}/{total} files in {dst_dir}")


if __name__ == "__main__":
    main()


