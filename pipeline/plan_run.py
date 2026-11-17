#!/usr/bin/env python3
"""Resolve episode.json.edit + words.json into timeline.json (30 fps, frame-exact cues).

    python3 pipeline/plan_run.py <slug> [--project DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

FPS = 30
CUE_RE = re.compile(r"^(?P<base>[^+\-#]+?)(?:#(?P<nth>\d+))?(?P<off>[+-]\d+)?$")


def norm(w: str) -> str:
    return re.sub(r"[^\w']", "", str(w).lower().replace("\u2019", "'"))


def bare_ascii(w: str) -> str:
    s = unicodedata.normalize("NFKD", norm(w))
    return "".join(c for c in s if not unicodedata.combining(c)).strip("'")


def probe_audio_seconds(wav_path: Path) -> float:
    if not wav_path.is_file():
        return 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return float(out or 0.0)
    except Exception:
        return 0.0


def resolve_cue(
    cue: Any,
    words: list[dict],
    wi_prev: int,
    placed_segs: list[dict],
    total_frames: int,
    fps: int = FPS,
) -> tuple[int | None, int, str | None]:
    """Return (frame, next_wi_prev, error_msg)."""
    s = str(cue or "").strip()
    if not s:
        return None, wi_prev, "empty cue"
    if s.startswith("w:"):
        return None, wi_prev, f'forbidden "w:" prefix in cue "{s}" — pass the bare word (e.g. "{s[2:]}")'
    if s.startswith("f:"):
        try:
            return max(0, int(round(float(s[2:])))), wi_prev, None
        except ValueError:
            return None, wi_prev, f'invalid frame cue "{s}"'
    if s.startswith("s:"):
        try:
            return max(0, int(round(float(s[2:]) * fps))), wi_prev, None
        except ValueError:
            return None, wi_prev, f'invalid seconds cue "{s}"'

    m = CUE_RE.match(s)
    if not m:
        return None, wi_prev, f'unrecognized cue "{s}"'

    base, nth, off = m["base"].strip(), int(m["nth"] or 1), int(m["off"] or 0)
    if base.lower() == "end":
        return max(0, total_frames + off), wi_prev, None

    if re.fullmatch(r"S\d+", base, re.IGNORECASE):
        sid = base.upper()
        matches = [seg for seg in placed_segs if str(seg.get("shot", "")).upper() == sid]
        if not matches:
            return None, wi_prev, f'shot reference "{s}" points to {sid} which has not started yet'
        seg = matches[min(len(matches) - 1, max(0, nth - 1))]
        return max(0, int(seg["a"]) + off), wi_prev, None

    target_norm = norm(base)
    target_ascii = bare_ascii(base)

    # First pass from wi_prev using exact norm() (mirrors shotlist.py occurrence counting)
    seen = 0
    for idx in range(max(0, wi_prev), len(words)):
        if norm(str(words[idx].get("w", ""))) == target_norm:
            seen += 1
            if seen == nth:
                frame = max(0, int(round(float(words[idx].get("s", 0)) * fps)) + off)
                return frame, idx + 1, None

    # Fallback pass from wi_prev using accent-stripped bare_ascii()
    seen = 0
    for idx in range(max(0, wi_prev), len(words)):
        if bare_ascii(str(words[idx].get("w", ""))) == target_ascii and target_ascii:
            seen += 1
            if seen == nth:
                frame = max(0, int(round(float(words[idx].get("s", 0)) * fps)) + off)
                return frame, idx + 1, None

    return None, wi_prev, None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--project", default=".")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    ep_dir = root / "src" / "episodes" / args.slug
    ep_path = ep_dir / "episode.json"
    words_path = ep_dir / "words.json"
    tl_path = ep_dir / "timeline.json"

    if not ep_path.is_file():
        print(f"No episode.json at {ep_path}", file=sys.stderr)
        return 2
    ep = json.loads(ep_path.read_text(encoding="utf-8"))
    words = json.loads(words_path.read_text(encoding="utf-8")) if words_path.is_file() else []
    words = [w for w in words if isinstance(w, dict) and norm(str(w.get("w", "")))]

    words_end_sec = float(words[-1].get("e", 0.0)) if words else 10.0
    audio_sec = probe_audio_seconds(root / "public" / args.slug / "voice.wav")
    dur_sec = max(1.0, words_end_sec, audio_sec)
    total = max(30, int(round(dur_sec * FPS)))

    aspect = str(ep.get("aspect") or "16:9").strip()
    if aspect not in ("9:16", "16:9"):
        aspect = "16:9"
    width, height = (1080, 1920) if aspect == "9:16" else (1920, 1080)

    ep_shots = {str(s.get("id")): s for s in (ep.get("shots") or []) if isinstance(s, dict) and s.get("id")}
    edit_list = [e for e in (ep.get("edit") or []) if isinstance(e, dict) and e.get("shot")]
    if not edit_list:
        edit_list = [{"shot": sid, "at": "f:0"} for sid in list(ep_shots.keys())[:1]]

    resolved: list[dict] = []
    wi_prev = 0
    prev_frame = -1
    for i, entry in enumerate(edit_list):
        sid = str(entry.get("shot"))
        cue = entry.get("at", "f:0")
        frame, next_wi, fatal_err = resolve_cue(cue, words, wi_prev, resolved, total, FPS)
        if fatal_err:
            print(f"[plan] ERROR on {sid}: {fatal_err}", file=sys.stderr)
            print("RESULT_JSON:" + json.dumps({"ok": False, "error": fatal_err}))
            return 3
        if frame is None:
            print(f"[plan] warning: unresolved cue '{cue}' for {sid} — merged into previous shot", file=sys.stderr)
            continue
        if i == 0:
            frame = max(0, frame)
        if frame <= prev_frame:
            # Clamp slightly forward if words overlap by 1 frame, else skip collapsed duplicate
            if frame == prev_frame and prev_frame + 1 < total - 1:
                frame = prev_frame + 1
            else:
                print(f"[plan] warning: cue '{cue}' for {sid} (f:{frame}) <= previous cut (f:{prev_frame}) — skipping", file=sys.stderr)
                continue
        if frame >= total:
            frame = max(prev_frame + 1, total - 15)
        wi_prev = next_wi
        prev_frame = frame
        item: dict[str, Any] = {"shot": sid, "a": frame}
        if entry.get("look"):
            item["look"] = str(entry["look"])
        resolved.append(item)

    if not resolved:
        print("[plan] ERROR: no segments resolved", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": "no segments resolved"}))
        return 4

    # Ensure first segment starts at frame 0
    resolved[0]["a"] = 0

    segments: list[dict] = []
    for idx, seg in enumerate(resolved):
        a = int(seg["a"])
        b = int(resolved[idx + 1]["a"]) if idx + 1 < len(resolved) else total
        if b <= a:
            continue
        seg_obj: dict[str, Any] = {
            "shot": seg["shot"],
            "a": a,
            "b": b,
            "t0": 0,
            "t1": b - a,
        }
        if seg.get("look"):
            seg_obj["look"] = seg["look"]
        segments.append(seg_obj)

    if segments:
        segments[-1]["b"] = total
        segments[-1]["t1"] = total - int(segments[-1]["a"])

    shot_acc: dict[str, dict] = {}
    for seg in segments:
        sid = seg["shot"]
        span = int(seg["b"]) - int(seg["a"])
        if sid not in shot_acc:
            shot_acc[sid] = {"first": int(seg["a"]), "len": span}
        else:
            shot_acc[sid]["len"] = max(shot_acc[sid]["len"], span)

    tl_shots: list[dict] = []
    for sid, info in shot_acc.items():
        meta = ep_shots.get(sid, {})
        length = max(1, int(info["len"]))
        key = meta.get("key")
        if not isinstance(key, int) or key < 0 or key >= length:
            key = max(1, min(length - 1, int(round(length * 0.5))))
        sh_obj: dict[str, Any] = {
            "id": sid,
            "size": meta.get("size") or "medium",
            "set": meta.get("set") or "living",
            "view": meta.get("view") or "front",
            "move": meta.get("move") or "push",
            "key": key,
            "len": length,
            "durationSec": round(length / FPS, 2),
            "first": int(info["first"]),
        }
        if meta.get("accent"):
            sh_obj["accent"] = meta["accent"]
        tl_shots.append(sh_obj)

    timeline = {
        "slug": args.slug,
        "title": ep.get("topic") or ( " ".join(ep["title"]) if isinstance(ep.get("title"), list) else str(ep.get("title") or args.slug)),
        "aspect": aspect,
        "width": width,
        "height": height,
        "fps": FPS,
        "total": total,
        "segments": segments,
        "shots": tl_shots,
        "fx": ep.get("fx") or [],
        "audio": ep.get("sound") or {"provider": "omni-flash", "music": False, "dialogue": False},
    }
    tl_path.write_text(json.dumps(timeline, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    seconds = round(total / FPS, 2)
    print(f"[plan] {len(segments)} segment(s), {len(tl_shots)} shot(s), {total} frames ({seconds:.2f}s @ {FPS}fps, {width}x{height})")
    print("RESULT_JSON:" + json.dumps({"ok": True, "segments": len(segments), "shots": len(tl_shots), "total": total, "seconds": seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
