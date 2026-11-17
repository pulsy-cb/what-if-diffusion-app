#!/usr/bin/env python3
"""Optional Anime 2D Video Upscaler pass (Modal Real-CUGAN / Real-ESRGAN) on top of mastered audio.

Elevates 360p/720p/1080p rendered episodes to 1080p, 2K (1440p), or 4K UHD (2160p) while
preserving crisp 2D animation linework and the exact -14 LUFS audio stream untouched.

    python3 pipeline/upscale_run.py <slug> [--project DIR] [--target auto-2k|auto-uhd|auto-1080p|2x|3x|4x]
                                    [--model real-cugan|realesr-animevideov3|realesrgan-x2plus|realesrgan-x4plus]
                                    [--denoise 0|1|2] [--gpu L40S|A100-40GB|L4] [--url URL] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import media_client as mc

ALLOWED_MODELS = ("real-cugan", "realesr-animevideov3", "realesrgan-x2plus", "realesrgan-x4plus")
ALLOWED_GPUS = ("L40S", "A100-40GB", "L4")


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


def probe_video_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    parts = [p.strip() for p in out.split("x") if p.strip()]
    if len(parts) < 2:
        raise RuntimeError(f"Could not probe video resolution of {path}: {out!r}")
    return int(parts[0]), int(parts[1])


def compute_scale_plan(in_w: int, in_h: int, target: str) -> dict[str, Any]:
    """Compute Modal integer scale (2|3|4) and optional exact Lanczos finish dimensions."""
    t = str(target or "auto-2k").strip().lower()
    if t in ("2x", "2"):
        scale = 2
        return {"output_scale": scale, "final_dims": None, "skip": False, "label": "2x"}
    if t in ("3x", "3"):
        scale = 3
        return {"output_scale": scale, "final_dims": None, "skip": False, "label": "3x"}
    if t in ("4x", "4"):
        scale = 4
        return {"output_scale": scale, "final_dims": None, "skip": False, "label": "4x"}

    is_vertical = in_h > in_w
    short_side = min(in_w, in_h)

    if t in ("auto-1080p", "1080p", "fhd"):
        want_short = 1080
        label = "1080p"
    elif t in ("auto-uhd", "uhd", "4k", "2160p"):
        want_short = 2160
        label = "4K UHD (2160p)"
    else:
        want_short = 1440
        label = "2K QHD (1440p)"

    if short_side >= want_short:
        return {"output_scale": 1, "final_dims": (in_w, in_h), "skip": True, "label": label}

    want_long = int(round(want_short * 16 / 9))
    target_w, target_h = (want_short, want_long) if is_vertical else (want_long, want_short)

    raw_scale = max(2, math.ceil(want_short / max(1, short_side)))
    output_scale = min(4, raw_scale)

    up_w = in_w * output_scale
    up_h = in_h * output_scale
    # Avoid expensive over-upscaling if aspect/scale differs from target rectangle, apply exact Lanczos finish when needed
    final_dims = None if (up_w == target_w and up_h == target_h) else (target_w, target_h)
    return {
        "output_scale": output_scale,
        "final_dims": final_dims,
        "skip": False,
        "label": label,
    }


def _multipart_encode(fields: dict[str, str], file_field: str, file_path: Path, mime: str = "video/mp4") -> tuple[bytes, str]:
    boundary = f"----WhatIfUpscale{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
                f"{v}\r\n"
            ).encode("utf-8")
        )
    filename = file_path.name
    chunks.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _http_req(
    method: str,
    url: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> bytes:
    hdrs = {"User-Agent": "whatif-diffusion-app/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Upscaler HTTP {e.code} {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Upscaler request failed for {url}: {e.reason}") from e


def _upscale_modal(
    in_video_mp4: Path,
    out_video_mp4: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    output_scale: int,
    denoise_level: int,
    gpu_type: str,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body, content_type = _multipart_encode(
        {
            "model_name": model_name,
            "output_scale": str(output_scale),
            "denoise_level": str(denoise_level),
            "gpu_type": gpu_type,
        },
        file_field="file",
        file_path=in_video_mp4,
    )

    submit_hdrs = {**headers, "Content-Type": content_type, "Accept": "application/json"}
    mc.emit_progress(stage="upscale", done=0, total=100, detail=f"Uploading video to {model_name} ({output_scale}x on {gpu_type})")
    raw_resp = _http_req("POST", f"{base_url.rstrip('/')}/upscale", data=body, headers=submit_hdrs, timeout=600.0)
    resp = json.loads(raw_resp.decode("utf-8"))
    task_id = str(resp.get("task_id") or resp.get("id") or "").strip()
    if not task_id:
        raise RuntimeError(f"Upscaler /upscale did not return task_id: {resp}")

    t0 = time.monotonic()
    deadline = 1800.0
    last_status = "submitted"
    try:
        while time.monotonic() - t0 < deadline:
            st_raw = _http_req("GET", f"{base_url.rstrip('/')}/status/{task_id}", headers={**headers, "Accept": "application/json"}, timeout=60.0)
            st = json.loads(st_raw.decode("utf-8"))
            status = str(st.get("status") or st.get("state") or "").strip().lower()
            pct = st.get("progress")
            last_status = status or last_status
            if status in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(f"Modal upscaler task {task_id} failed: {st.get('error') or st.get('detail') or st}")
            if status in ("completed", "succeeded", "done", "finished") or st.get("completed") is True:
                break
            mc.emit_progress(
                stage="upscale",
                done=int(pct) if isinstance(pct, (int, float)) else 0,
                total=100,
                detail=f"Upscaling ({last_status}{f' {pct}%' if pct is not None else ''})",
            )
            time.sleep(2.0)
        else:
            raise RuntimeError(f"Modal upscaler task {task_id} timed out after {int(deadline)}s (last status: {last_status})")

        mc.emit_progress(stage="upscale", done=95, total=100, detail="Downloading upscaled MP4")
        dl_url = f"{base_url.rstrip('/')}/download/{task_id}"
        req = urllib.request.Request(dl_url, headers={"User-Agent": "whatif-diffusion-app/1.0", **headers}, method="GET")
        with urllib.request.urlopen(req, timeout=600.0) as resp_stream, open(out_video_mp4, "wb") as out_f:
            shutil.copyfileobj(resp_stream, out_f, length=1024 * 1024)
    finally:
        try:
            _http_req("DELETE", f"{base_url.rstrip('/')}/task/{task_id}", headers=headers, timeout=15.0)
        except Exception:
            pass

    return {"task_id": task_id, "seconds": round(time.monotonic() - t0, 2)}


def _upscale_mock(in_video_mp4: Path, out_video_mp4: Path, output_scale: int) -> dict[str, Any]:
    t0 = time.monotonic()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(in_video_mp4),
            "-vf",
            f"scale=iw*{output_scale}:ih*{output_scale}:flags=lanczos,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_video_mp4),
        ],
        check=True,
    )
    return {"task_id": "mock", "seconds": round(time.monotonic() - t0, 2)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--project", default=".")
    ap.add_argument("--target", default="", help="auto-uhd | auto-2k | auto-1080p | 2x | 3x | 4x")
    ap.add_argument("--model", default="", help=" | ".join(ALLOWED_MODELS))
    ap.add_argument("--denoise", type=int, default=-1, help="0 | 1 | 2")
    ap.add_argument("--gpu", default="", help=" | ".join(ALLOWED_GPUS))
    ap.add_argument("--url", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_intermixed_args(argv)

    root = Path(args.project).resolve()
    slug = args.slug
    od = root / "out" / slug
    raw = od / f"{slug}-raw.mp4"
    final = od / f"{slug}-final.mp4"
    base_copy = od / f"{slug}-base.mp4"
    upscaled = od / f"{slug}-upscaled.mp4"
    manifest_path = od / "upscale.json"
    rep_path = od / "report.json"

    if not final.is_file():
        print(f"Missing {final}: run render/master first", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"missing {final}"}))
        return 2

    # Snapshot clean base master so re-upscaling never double-processes an already-upscaled final.mp4
    if not base_copy.is_file() or (raw.is_file() and raw.stat().st_mtime > base_copy.stat().st_mtime):
        shutil.copy2(final, base_copy)

    up_cfg = mc.load_media_config().get("upscaler") or {}
    driver = str(up_cfg.get("driver") or ("mock" if os.environ.get("NODE_TEST_CONTEXT") else "modal-anime")).strip().lower()
    base_url = str(args.url or up_cfg.get("baseURL") or "https://bkrpulsy--anime-upscaler-t4-fastapi-app.modal.run").strip().rstrip("/")
    api_key = str(up_cfg.get("apiKey") or "").strip()
    target = str(args.target or up_cfg.get("target") or "auto-2k").strip().lower()
    model_name = str(args.model or up_cfg.get("model") or "real-cugan").strip().lower()
    if model_name not in ALLOWED_MODELS:
        model_name = "real-cugan"
    denoise = args.denoise if args.denoise in (0, 1, 2) else int(up_cfg.get("denoiseLevel", 1))
    if denoise not in (0, 1, 2):
        denoise = 1
    gpu_type = str(args.gpu or up_cfg.get("gpuType") or "L40S").strip()
    if gpu_type not in ALLOWED_GPUS:
        gpu_type = "L40S"

    in_w, in_h = probe_video_dims(base_copy)
    plan = compute_scale_plan(in_w, in_h, target)

    if plan["skip"] and not args.force:
        print(f"[upscale] {base_copy.name} is already {in_w}x{in_h} (>= target {plan['label']}); skipping.")
        print("RESULT_JSON:" + json.dumps({"ok": True, "skipped": True, "width": in_w, "height": in_h, "file": str(final)}))
        return 0

    sig_src = f"{base_copy.stat().st_size}:{int(base_copy.stat().st_mtime)}|{driver}|{model_name}|{plan['output_scale']}|{denoise}|{plan['final_dims']}|{base_url}"
    sig = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()

    prev = load_json(manifest_path)
    if (
        not args.force
        and isinstance(prev, dict)
        and prev.get("hash") == sig
        and upscaled.is_file()
        and upscaled.stat().st_size > 1024
        and upscaled.stat().st_mtime >= base_copy.stat().st_mtime
    ):
        shutil.copy2(upscaled, final)
        print(f"[upscale] cached ({prev.get('width')}x{prev.get('height')} via {model_name})")
        print("RESULT_JSON:" + json.dumps({"ok": True, "cached": True, **prev}))
        return 0

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"whatif_up_{slug}_"))
    try:
        tmp_in = tmp_dir / f"{slug}-video.mp4"
        tmp_up = tmp_dir / f"{slug}-up-raw.mp4"

        # Strip audio before upload: faster transfer & keeps -14 LUFS audio untouched locally
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(base_copy), "-an", "-c:v", "copy", str(tmp_in)],
            check=True,
        )

        if driver == "mock":
            up_meta = _upscale_mock(tmp_in, tmp_up, plan["output_scale"])
        else:
            up_meta = _upscale_modal(
                tmp_in,
                tmp_up,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                output_scale=plan["output_scale"],
                denoise_level=denoise,
                gpu_type=gpu_type,
            )

        # Remux neural-upscaled video with original -14 LUFS mastered audio track cleanly
        tmp_mux = upscaled.with_name(f".{slug}-upscaled.tmp.mp4")
        try:
            final_dims = plan["final_dims"]
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(tmp_up), "-i", str(base_copy), "-map", "0:v:0", "-map", "1:a:0"]
            if final_dims is not None:
                fw, fh = final_dims
                cmd += ["-vf", f"scale={fw}:{fh}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p"]
            else:
                cmd += ["-c:v", "copy"]
            cmd += ["-c:a", "copy", "-movflags", "+faststart", str(tmp_mux)]
            subprocess.run(cmd, check=True)
            os.replace(tmp_mux, upscaled)
        finally:
            if tmp_mux.exists():
                tmp_mux.unlink(missing_ok=True)

        out_w, out_h = probe_video_dims(upscaled)
        shutil.copy2(upscaled, final)

        record = {
            "ok": True,
            "hash": sig,
            "driver": driver,
            "model": model_name,
            "scale": plan["output_scale"],
            "denoise": denoise,
            "gpu": gpu_type,
            "target": target,
            "inputWidth": in_w,
            "inputHeight": in_h,
            "width": out_w,
            "height": out_h,
            "file": str(final),
            "upscaledFile": str(upscaled),
            "seconds": up_meta.get("seconds"),
        }
        save_json(manifest_path, record)

        rep = load_json(rep_path)
        if isinstance(rep, dict):
            rep["width"] = out_w
            rep["height"] = out_h
            rep["upscaled"] = {"model": model_name, "scale": plan["output_scale"], "target": target, "from": f"{in_w}x{in_h}", "to": f"{out_w}x{out_h}"}
            save_json(rep_path, rep)

        print(f"[upscale] {in_w}x{in_h} -> {out_w}x{out_h} ({model_name} x{plan['output_scale']}, {up_meta.get('seconds')}s)")
        print("RESULT_JSON:" + json.dumps(record))
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
