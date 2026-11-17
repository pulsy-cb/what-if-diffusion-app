#!/usr/bin/env python3
"""Generate per-shot Image-to-Video clips (with native diegetic SFX only — NO music, NO dialogue)
via Gemini Omni Flash 2.1 (or any configured video diffusion provider).

    python3 pipeline/video_gen.py <slug> [S3 S7 ...] [--project DIR] [--force] [--concurrency 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import media_client as mc


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    if not p.is_file():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_json(p: Path, default: Any = None) -> Any:
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build_video_prompt(spec: dict[str, Any], s_meta: dict[str, Any]) -> str:
    cam_move = str(spec.get("cameraMove") or s_meta.get("move") or "push").strip()
    motion = str(spec.get("videoPrompt") or "").strip()
    sfx = str(spec.get("sfxHint") or "quiet natural room ambience").strip()
    return (
        "Minimalist 2D 'What If' cartoon animation starting directly from Frame 0 (exact starting image): "
        "keep oversized round cartoon heads, bold dark charcoal outlines, flat cel-shaded outfit colors, "
        "warm golden edge rim-lighting, steady room light levels, and grounded floor/prop contact shadows "
        "(no floating props, no sudden ceiling-light switches, no character cloning, single continuous shot with NO cuts).\n"
        f"CAMERA MOTION — Smooth {cam_move} camera move.\n"
        f"ACTION — {motion}\n"
        f"AUDIO — Physical sound effects and environment ambience ONLY: {sfx}. "
        "STRICTLY NO spoken dialogue, NO mouth lip-sync talking, NO voice, NO vocals, NO narration, NO singing, and NO background music."
    )


def run_clips(root: Path, slug: str, shot_filter: set[str], force: bool = False, conc: int = 2) -> tuple[int, dict[str, Any]]:
    ep_dir = root / "src" / "episodes" / slug
    out_dir = root / "out" / slug
    board_dir = out_dir / "board"
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    ep = load_json(ep_dir / "episode.json", {}) or {}
    tl = load_json(ep_dir / "timeline.json", {}) or {}
    aspect = str(ep.get("aspect") or tl.get("aspect") or "16:9").strip()
    if aspect not in ("9:16", "16:9"):
        aspect = "16:9"
    fps = float(tl.get("fps") or 30.0)

    tl_shots = [s for s in (tl.get("shots") or []) if isinstance(s, dict) and s.get("id")]
    if not tl_shots:
        return 2, {"ok": False, "error": "timeline.json has no shots"}

    cfg = mc.load_media_config()["video"]
    driver = str(cfg.get("driver") or "mock")
    model = str(cfg.get("model") or ("omni-1.2-flash" if driver == "gflow" else "gemini-omni-flash-2.1"))
    resolution = str(cfg.get("resolution") or ("360p" if driver == "gflow" else "1080p")).strip().lower()

    if driver == "gflow" and not cfg.get("project"):
        gflow_proj = mc.ensure_gflow_project(root, slug, cfg)
        if gflow_proj and gflow_proj.get("project_id"):
            cfg = {**cfg, "project": gflow_proj["project_id"]}
    active_proj = str(cfg.get("project") or "").strip()

    board_manifest = load_json(board_dir / "manifest.json", {}) or {}
    manifest_path = clips_dir / "manifest.json"
    manifest = load_json(manifest_path, {}) or {}
    tasks: list[dict[str, Any]] = []

    for s in tl_shots:
        sid = str(s["id"]).upper()
        if shot_filter and sid not in shot_filter:
            continue
        still_path = board_dir / f"{sid}.png"
        if not still_path.is_file():
            return 2, {"ok": False, "error": f"Missing storyboard still {still_path} (run stills first)"}
        spec_path = ep_dir / "shots" / f"{sid}.json"
        spec = load_json(spec_path)
        if not isinstance(spec, dict):
            return 2, {"ok": False, "error": f"Missing or invalid {spec_path}"}

        len_frames = max(15, int(s.get("len") or 60))
        dur_sec = max(2.0, round(len_frames / fps, 2))
        prompt = build_video_prompt(spec, s)
        still_hash = sha256_file(still_path)
        sig = sha256_str(json.dumps({
            "driver": driver,
            "model": model,
            "resolution": resolution,
            "aspect": aspect,
            "dur": dur_sec,
            "prompt": prompt,
            "stillHash": still_hash,
            "seed": spec.get("seed"),
        }, sort_keys=True))

        bmeta = board_manifest.get(sid)
        start_media_id = (
            str(bmeta.get("mediaName") or "").strip()
            if isinstance(bmeta, dict) and (not active_proj or bmeta.get("projectId") == active_proj)
            else ""
        )
        tasks.append({
            "id": sid,
            "still": still_path,
            "file": clips_dir / f"{sid}.mp4",
            "prompt": prompt,
            "dur": dur_sec,
            "seed": spec.get("seed"),
            "start_media_id": start_media_id or None,
            "sig": sig,
        })

    generated: list[str] = []
    cached: list[str] = []
    to_run: list[dict[str, Any]] = []
    for t in tasks:
        prev = manifest.get(t["id"]) if isinstance(manifest.get(t["id"]), dict) else None
        if not force and t["file"].is_file() and t["file"].stat().st_size > 1024 and prev and prev.get("hash") == t["sig"]:
            cached.append(t["id"])
        else:
            to_run.append(t)

    credits_total = 0
    if to_run:
        workers = min(conc, 2) if driver == "gflow" else conc
        mc.emit_progress(stage="clips", done=0, total=len(to_run), detail=f"Animating {len(to_run)} clip(s) with {model}@{resolution}")
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_run)))) as pool:
            futs = {
                pool.submit(
                    mc.generate_video,
                    t["still"],
                    t["prompt"],
                    t["dur"],
                    aspect,
                    t["file"],
                    t["seed"],
                    cfg,
                    t["start_media_id"],
                ): t
                for t in to_run
            }
            done_n = 0
            for fut in as_completed(futs):
                t = futs[fut]
                res = fut.result()
                done_n += 1
                credits_total += int(res.get("credits") or 0)
                manifest[t["id"]] = {
                    "hash": t["sig"],
                    "file": str(t["file"].relative_to(root).as_posix()),
                    "driver": res.get("driver"),
                    "model": res.get("model"),
                    "dur": t["dur"],
                    **({"mediaName": res.get("mediaName")} if res.get("mediaName") else {}),
                    **({"projectId": res.get("projectId") or active_proj} if (res.get("projectId") or active_proj) else {}),
                }
                generated.append(t["id"])
                mc.append_cost(out_dir, {"step": "clips", "shot": t["id"], **res})
                mc.emit_progress(stage="clips", done=done_n, total=len(to_run), detail=f"{t['id']}.mp4 ready ({res.get('seconds')}s)")

    save_json(manifest_path, manifest)
    clips_out = [str(t["file"].relative_to(root).as_posix()) for t in tasks]
    print(f"[clips] {len(generated)} generated, {len(cached)} cached ({model}@{resolution} via {driver})")
    return 0, {
        "ok": True,
        "generated": generated,
        "cached": cached,
        "clips": clips_out,
        **({"gflowProjectId": cfg.get("project")} if cfg.get("project") else {}),
        "credits": credits_total,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("shots", nargs="*", help="Optional shot IDs (e.g. S3 S7)")
    ap.add_argument("--project", default=".")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_intermixed_args(argv)

    root = Path(args.project).resolve()
    shot_filter = {x.strip().upper() for x in args.shots if x.strip() and not x.startswith("-")}
    code, payload = run_clips(root, args.slug, shot_filter, force=args.force, conc=args.concurrency)
    print("RESULT_JSON:" + json.dumps(payload, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
