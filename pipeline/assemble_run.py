#!/usr/bin/env python3
"""Assemble Gemini Omni Flash 2.1 I2V clips frame-accurately onto `timeline.json`,
extract & duck Omni Flash native SFX under `:8005` `voice.wav` (no music), and produce `out/<slug>/<slug>-raw.mp4`.

    python3 pipeline/assemble_run.py <slug> [--project DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

import video_gen as vg

SR = 44100


def load_json(p: Path, default: Any = None) -> Any:
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return max(0.05, float(out or 0.0))


def decode_audio_f64(path: Path) -> np.ndarray:
    """Decode any media file's audio stream to mono float64 @ 44.1 kHz; returns empty array if silent/no audio."""
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
    )
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, dtype=np.float64)
    return np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64)


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)


def duck_under_words(sfx: np.ndarray, words: list[dict], duck_db: float = -6.0) -> np.ndarray:
    """Smoothly duck Omni Flash SFX under spoken words while letting natural room/foley breathe between phrases."""
    n = len(sfx)
    if n == 0 or not words:
        return sfx
    gain = np.ones(n, dtype=np.float64)
    target = 10.0 ** (duck_db / 20.0)
    ramp = int(0.08 * SR)
    pad = int(0.05 * SR)

    # Merge nearby spoken words (< 0.25s gap) into continuous speech regions
    regions: list[tuple[int, int]] = []
    for w in words:
        s_idx = max(0, int(round(float(w.get("s", 0.0)) * SR)) - pad)
        e_idx = min(n, int(round(float(w.get("e", 0.0)) * SR)) + pad)
        if e_idx <= s_idx:
            continue
        if regions and s_idx - regions[-1][1] <= int(0.25 * SR):
            regions[-1] = (regions[-1][0], max(regions[-1][1], e_idx))
        else:
            regions.append((s_idx, e_idx))

    for s_idx, e_idx in regions:
        gain[s_idx:e_idx] = np.minimum(gain[s_idx:e_idx], target)
        # Smooth attack before s_idx
        a0 = max(0, s_idx - ramp)
        if s_idx > a0:
            curve = np.linspace(1.0, target, s_idx - a0)
            gain[a0:s_idx] = np.minimum(gain[a0:s_idx], curve)
        # Smooth release after e_idx
        r1 = min(n, e_idx + ramp)
        if r1 > e_idx:
            curve = np.linspace(target, 1.0, r1 - e_idx)
            gain[e_idx:r1] = np.minimum(gain[e_idx:r1], curve)

    return sfx * gain


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--project", default=".")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    slug = args.slug
    ep_dir = root / "src" / "episodes" / slug
    pub_dir = root / "public" / slug
    out_dir = root / "out" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    ep = load_json(ep_dir / "episode.json", {}) or {}
    tl = load_json(ep_dir / "timeline.json", {}) or {}
    words = load_json(ep_dir / "words.json", []) or []
    voice_wav = pub_dir / "voice.wav"
    if not voice_wav.is_file():
        print(f"Missing {voice_wav}: run voice first", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"missing {voice_wav}"}))
        return 2

    segments = [s for s in (tl.get("segments") or []) if isinstance(s, dict) and s.get("shot")]
    if not segments:
        print("timeline.json has no segments", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": "timeline.json has no segments"}))
        return 2

    # 1. Ensure all I2V clips exist and are fresh relative to current stills & shot specs
    clip_code, clip_res = vg.run_clips(root, slug, set(), force=args.force)
    if clip_code != 0:
        print("RESULT_JSON:" + json.dumps(clip_res))
        return clip_code

    fps = float(tl.get("fps") or 30.0)
    aspect = str(ep.get("aspect") or tl.get("aspect") or "16:9").strip()
    vid_res = str(vg.mc.load_media_config().get("video", {}).get("resolution") or os.environ.get("WHATIF_VIDEO_RESOLUTION") or "1080p")
    width, height = vg.mc.target_dimensions(aspect, vid_res)
    clips_dir = out_dir / "clips"

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"whatif_asm_{slug}_"))
    seg_files: list[Path] = []
    sfx_chunks: list[np.ndarray] = []
    total_frames_out = 0
    fade_samples = int(0.035 * SR)

    try:
        vg.mc.emit_progress(stage="assemble", done=0, total=len(segments), detail=f"Normalizing {len(segments)} I2V segment(s)")
        for idx, seg in enumerate(segments):
            sid = str(seg["shot"]).upper()
            seg_frames = max(1, int(seg["b"]) - int(seg["a"]))
            target_dur = seg_frames / fps
            total_frames_out += seg_frames

            clip_path = clips_dir / f"{sid}.mp4"
            if not clip_path.is_file():
                raise RuntimeError(f"Missing video clip {clip_path}")

            clip_dur = probe_duration(clip_path)
            vf_parts = [
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                "setsar=1",
            ]
            if clip_dur < target_dur - 0.02:
                stretch = min(2.0, target_dur / max(0.2, clip_dur))
                pad_sec = max(1.0, target_dur - clip_dur * stretch + 0.5)
                vf_parts.append(f"setpts=PTS*{stretch:.6f}")
                vf_parts.append(f"fps={int(round(fps))}")
                vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad_sec:.2f}")
            else:
                vf_parts.append(f"fps={int(round(fps))}")
            look = str(seg.get("look") or "").strip().lower()
            if look == "memory":
                vf_parts.append("eq=saturation=0.72:contrast=0.95,colorbalance=rs=0.04:bs=-0.04")
            elif look == "tension":
                vf_parts.append("eq=contrast=1.12:saturation=0.88:brightness=-0.03")
            vf_parts.append("format=yuv420p")

            seg_v = tmp_dir / f"seg_{idx:04d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(clip_path),
                    "-an",
                    "-vf", ",".join(vf_parts),
                    "-frames:v", str(seg_frames),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(seg_v),
                ],
                check=True,
            )
            seg_files.append(seg_v)

            # Extract native SFX audio from Gemini Omni Flash clip & crossfade-safe shape
            target_samples = int(round(target_dur * SR))
            raw_sfx = decode_audio_f64(clip_path)
            if len(raw_sfx) >= target_samples:
                chunk = raw_sfx[:target_samples].copy()
            else:
                chunk = np.zeros(target_samples, dtype=np.float64)
                if len(raw_sfx) > 0:
                    chunk[: len(raw_sfx)] = raw_sfx

            f_len = min(fade_samples, len(chunk) // 2)
            if f_len > 1:
                ramp = np.linspace(0.0, 1.0, f_len)
                chunk[:f_len] *= ramp
                chunk[-f_len:] *= ramp[::-1]
            sfx_chunks.append(chunk)
            vg.mc.emit_progress(stage="assemble", done=idx + 1, total=len(segments), detail=f"Segment {sid} ({idx + 1}/{len(segments)})")

        # 2. Concatenate normalized video segments
        concat_list = tmp_dir / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files) + "\n", encoding="utf-8")
        silent_mp4 = tmp_dir / "silent.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c", "copy",
                str(silent_mp4),
            ],
            check=True,
        )

        # 3. Mix voice.wav (:8005) + ducked Omni Flash native SFX (NO music)
        total_samples = int(round((total_frames_out / fps) * SR))
        voice_raw = decode_audio_f64(voice_wav)
        voice = np.zeros(total_samples, dtype=np.float64)
        v_copy = min(len(voice_raw), total_samples)
        if v_copy > 0:
            voice[:v_copy] = voice_raw[:v_copy]

        sfx_concat = np.concatenate(sfx_chunks) if sfx_chunks else np.zeros(total_samples, dtype=np.float64)
        sfx = np.zeros(total_samples, dtype=np.float64)
        s_copy = min(len(sfx_concat), total_samples)
        if s_copy > 0:
            sfx[:s_copy] = sfx_concat[:s_copy]

        # Normalize Omni SFX to safe headroom (-10 dB relative to voice scale) and duck under speech
        sfx_peak = float(np.max(np.abs(sfx))) if len(sfx) else 0.0
        if sfx_peak > 0.35:
            sfx = sfx * (0.35 / sfx_peak)
        sfx = sfx * (10.0 ** (-10.0 / 20.0))
        sfx = duck_under_words(sfx, words, duck_db=-6.0)

        write_wav(pub_dir / "sfx.wav", sfx)
        mix = np.clip(voice + sfx, -0.99, 0.99)
        mix_wav = tmp_dir / "mix.wav"
        write_wav(mix_wav, mix)

        # 4. Mux video + mixed audio into out/<slug>/<slug>-raw.mp4
        raw_mp4 = out_dir / f"{slug}-raw.mp4"
        tmp_raw = out_dir / f".{slug}-raw.tmp.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(silent_mp4),
                "-i", str(mix_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
                "-shortest", "-movflags", "+faststart",
                str(tmp_raw),
            ],
            check=True,
        )
        tmp_raw.replace(raw_mp4)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    seconds = round(total_frames_out / fps, 2)
    print(f"[assemble] wrote out/{slug}/{slug}-raw.mp4 ({len(segments)} segments, {seconds:.2f}s @ {width}x{height})")
    print("RESULT_JSON:" + json.dumps({
        "ok": True,
        "output": f"out/{slug}/{slug}-raw.mp4",
        "segments": len(segments),
        "seconds": seconds,
        "credits": clip_res.get("credits", 0),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
