#!/usr/bin/env python3
"""Unified multi-provider client for 100% Diffusion Image (Nano Banana 2 Lite)
and Image-to-Video + Native SFX (Gemini Omni Flash 2.1).

Supports drivers:
  - gflow  : Google Flow / REST proxy (composite or multi ref support, sync or async jobs)
  - openai : OpenAI-compatible / Agnes / REST gateways
  - gemini : Official Google Gemini / Vertex REST (:generateContent / :predictLongRunning)
  - http   : Generic configurable JSON HTTP endpoint + dotted responsePaths + job polling
  - mock   : Deterministic offline Pillow + FFmpeg generator for tests & dry runs
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def emit_progress(**payload: Any) -> None:
    print("PROGRESS_JSON:" + json.dumps(payload, ensure_ascii=False), flush=True)


def _upscaler_from_env(is_test: bool) -> dict[str, Any]:
    enabled = (os.environ.get("WHATIF_UPSCALER_ENABLED") or "").strip().lower() in ("1", "true", "yes")
    drv = (os.environ.get("WHATIF_UPSCALER_DRIVER") or ("mock" if is_test else "modal-anime")).strip().lower()
    try:
        denoise = int(os.environ.get("WHATIF_UPSCALER_DENOISE") or "1")
    except ValueError:
        denoise = 1
    if denoise not in (0, 1, 2):
        denoise = 1
    gpu = (os.environ.get("WHATIF_UPSCALER_GPU") or "L40S").strip()
    if gpu not in ("L40S", "A100-40GB", "L4"):
        gpu = "L40S"
    return {
        "enabled": enabled,
        "driver": drv,
        "baseURL": (os.environ.get("WHATIF_UPSCALER_BASE_URL") or "https://bkrpulsy--anime-upscaler-t4-fastapi-app.modal.run").strip().rstrip("/"),
        "apiKey": (os.environ.get("WHATIF_UPSCALER_API_KEY") or "").strip(),
        "target": (os.environ.get("WHATIF_UPSCALER_TARGET") or "auto-2k").strip().lower(),
        "model": (os.environ.get("WHATIF_UPSCALER_MODEL") or "real-cugan").strip().lower(),
        "denoiseLevel": denoise,
        "gpuType": gpu,
    }


def load_media_config() -> dict[str, dict[str, Any]]:
    is_test = bool(os.environ.get("NODE_TEST_CONTEXT") or os.environ.get("NODE_ENV") == "test")
    raw_json = (os.environ.get("WHATIF_MEDIA_CONFIG_JSON") or "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "image" in parsed and "video" in parsed:
                vid = parsed["video"]
                if isinstance(vid, dict) and "resolution" not in vid:
                    def_res = "360p" if str(vid.get("driver") or "").lower() == "gflow" else "1080p"
                    res_env = (os.environ.get("WHATIF_VIDEO_RESOLUTION") or def_res).strip().lower()
                    vid["resolution"] = res_env if res_env in ("360p", "720p", "1080p") else def_res
                if "upscaler" not in parsed or not isinstance(parsed.get("upscaler"), dict):
                    parsed["upscaler"] = _upscaler_from_env(is_test)
                return parsed
        except json.JSONDecodeError:
            pass

    default_driver = "mock" if is_test else "gflow"

    def role_from_env(prefix: str, def_model: str, def_ep: str) -> dict[str, Any]:
        drv = (os.environ.get(f"WHATIF_{prefix}_PROVIDER") or default_driver).strip().lower()
        def_base = "http://127.0.0.1:8006" if drv == "gflow" else "http://127.0.0.1:8080/v1"
        def_ref_mode = "multi" if drv == "gflow" else "multi"
        def_max_refs = "10" if drv == "gflow" else "4"
        cfg: dict[str, Any] = {
            "driver": drv,
            "baseURL": (os.environ.get(f"WHATIF_{prefix}_BASE_URL") or def_base).strip(),
            "apiKey": (os.environ.get(f"WHATIF_{prefix}_API_KEY") or "").strip(),
            "model": (os.environ.get(f"WHATIF_{prefix}_MODEL") or def_model).strip(),
            "refMode": (os.environ.get(f"WHATIF_{prefix}_REF_MODE") or def_ref_mode).strip().lower(),
            "maxRefs": int(os.environ.get(f"WHATIF_{prefix}_MAX_REFS") or def_max_refs),
            "endpointPath": (os.environ.get(f"WHATIF_{prefix}_ENDPOINT") or def_ep).strip(),
            "pollPath": (os.environ.get(f"WHATIF_{prefix}_POLL_PATH") or "").strip(),
        }
        profile = (os.environ.get("WHATIF_GFLOW_PROFILE") or "").strip()
        if profile:
            cfg["profile"] = profile
        if prefix == "IMAGE":
            fb = (os.environ.get("WHATIF_IMAGE_FALLBACK_MODEL") or ("nano2.1" if drv == "gflow" else "")).strip()
            if fb:
                cfg["fallbackModel"] = fb
        if prefix == "VIDEO":
            def_res = "360p" if drv == "gflow" else "1080p"
            res_env = (os.environ.get("WHATIF_VIDEO_RESOLUTION") or def_res).strip().lower()
            cfg["resolution"] = res_env if res_env in ("360p", "720p", "1080p") else def_res
        return cfg

    vid_drv = (os.environ.get("WHATIF_VIDEO_PROVIDER") or default_driver).strip().lower()
    img_drv = (os.environ.get("WHATIF_IMAGE_PROVIDER") or default_driver).strip().lower()
    return {
        "image": role_from_env("IMAGE", "nano2-lite", "/v1/images/generations" if img_drv == "gflow" else "/images/generations"),
        "video": role_from_env(
            "VIDEO",
            "omni-1.2-flash" if vid_drv == "gflow" else "gemini-omni-flash-2.1",
            "/v1/videos/generations" if vid_drv == "gflow" else "/videos/generations",
        ),
        "upscaler": _upscaler_from_env(is_test),
    }


def target_dimensions(aspect: str, resolution: str = "1080p") -> tuple[int, int]:
    a = str(aspect or "16:9").strip()
    r = str(resolution or "1080p").strip().lower()
    if a == "1:1":
        if r == "360p":
            return 360, 360
        if r == "720p":
            return 720, 720
        return 1024, 1024
    if a == "9:16":
        if r == "360p":
            return 360, 640
        if r == "720p":
            return 720, 1280
        return 1080, 1920
    if r == "360p":
        return 640, 360
    if r == "720p":
        return 1280, 720
    return 1920, 1080


def _encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/webp" if ext == "webp" else "image/png"
    return f"data:{mime};base64,{_encode_b64(path)}"


def _get_by_path(obj: Any, dotted: str | None) -> Any:
    if not dotted:
        return None
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    hdrs = {"Accept": "application/json", "User-Agent": "whatif-diffusion-app/1.0"}
    if headers:
        hdrs.update(headers)
    data: bytes | None = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} on {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error on {url}: {exc.reason}") from exc


def _download_bytes(url: str, headers: dict[str, str] | None = None, timeout: float = 180.0) -> bytes:
    hdrs = {"User-Agent": "whatif-diffusion-app/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"HTTP {exc.code} downloading {url}: {body}") from exc


def _auth_headers(cfg: dict[str, Any]) -> dict[str, str]:
    hdrs: dict[str, str] = {}
    if isinstance(cfg.get("headers"), dict):
        for k, v in cfg["headers"].items():
            if isinstance(v, str) and v.strip():
                hdrs[str(k)] = v
    api_key = str(cfg.get("apiKey") or "").strip()
    driver = str(cfg.get("driver") or "").lower()
    if api_key:
        if driver == "gemini":
            hdrs.setdefault("x-goog-api-key", api_key)
        else:
            hdrs.setdefault("Authorization", f"Bearer {api_key}")
    return hdrs


def _join_url(base: str, ep: str) -> str:
    b = base.rstrip("/")
    if not ep:
        return b
    if ep.startswith("http://") or ep.startswith("https://"):
        return ep
    return f"{b}/{ep.lstrip('/')}"


def _gflow_urls(base_url: str) -> tuple[str, str]:
    b = str(base_url or "").strip().rstrip("/")
    origin = b[:-3] if b.endswith("/v1") else b
    return origin, f"{origin}/v1"


def _gflow_endpoint(base_url: str, ep_path: str, default_subpath: str) -> str:
    _origin, v1_root = _gflow_urls(base_url)
    ep = str(ep_path or "").strip()
    if ep.startswith(("http://", "https://")):
        return ep
    sub = ep or default_subpath
    if sub.startswith("/v1/"):
        sub = sub[3:]
    elif sub == "/v1":
        sub = default_subpath
    return f"{v1_root}/{sub.lstrip('/')}"


def _resolve_url(raw_url: str, base_url: str) -> str:
    u = str(raw_url or "").strip()
    if not u:
        return u
    if u.startswith(("http://", "https://")):
        return u
    if u.startswith("/"):
        sp = urllib.parse.urlsplit(base_url)
        if sp.scheme and sp.netloc:
            return f"{sp.scheme}://{sp.netloc}{u}"
    return _join_url(base_url, u)


def _normalize_gflow_image_model(m: str) -> str:
    s = str(m or "").strip().lower()
    if s in ("lite", "nano-lite", "nano2-lite", "nano-banana-2-lite", "harbor_seal"):
        return "nano2-lite"
    if s in ("2.1", "nano2.1", "nano-2.1", "nano21", "nano2", "nano-banana-2.1", "narwhal"):
        return "nano2.1"
    if s in ("nano-pro", "pro", "gem_pix_2"):
        return "nano-pro"
    if s in ("image4", "imagen4", "imagen_3_5"):
        return "image4"
    return str(m or "nano2-lite").strip() or "nano2-lite"


def _normalize_gflow_video_model(m: str) -> str:
    s = str(m or "").strip().lower()
    if s in (
        "omni-1.2-flash",
        "omni-1.2",
        "omni_1_2_flash",
        "omni-flash-1.2",
        "omni flash 1.2",
        "gemini-omni-flash-1.2",
        "1.2",
    ):
        return "omni-1.2-flash"
    if s in ("omni-flash", "omni_flash", "omni-flash-2.1", "gemini-omni-flash-2.1"):
        return "omni-flash"
    if s in ("veo_3_1_lite", "veo-lite"):
        return "veo-lite"
    if s in ("veo_3_1_fast", "veo-fast"):
        return "veo-fast"
    if s in ("veo_3_1_quality", "veo-quality"):
        return "veo-quality"
    if s in ("veo_3_1_lite_lower_priority", "veo-lite-lp"):
        return "veo-lite-lp"
    return str(m or "omni-1.2-flash").strip() or "omni-1.2-flash"


def ensure_gflow_project(project_root: Path, slug: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Create or reuse a dedicated Google Flow project via :8006 POST /v1/projects for this Studio episode."""
    if str(cfg.get("driver") or "").lower() != "gflow":
        return None
    base_url = str(cfg.get("baseURL") or "").strip()
    if not base_url:
        return None

    origin_root, v1_root = _gflow_urls(base_url)
    profile = str(cfg.get("profile") or "").strip()
    out_dir = project_root / "out" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = out_dir / "gflow-project.json"
    lock_file = out_dir / ".gflow-project.lock"

    with open(lock_file, "w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            if manifest_file.is_file():
                try:
                    existing = json.loads(manifest_file.read_text(encoding="utf-8"))
                    if (
                        isinstance(existing, dict)
                        and str(existing.get("project_id") or "").strip()
                        and str(existing.get("baseURL") or "").rstrip("/") == origin_root
                        and (not profile or str(existing.get("profile") or "").strip() == profile)
                    ):
                        return existing
                except Exception:
                    pass

            proj_name = project_root.name.strip()
            if proj_name and proj_name != slug and not proj_name.startswith("whatif-"):
                title = f"whatif-{proj_name}-{slug}"[:80]
            else:
                title = f"whatif-{slug}"[:80]

            headers = _auth_headers(cfg)
            payload: dict[str, Any] = {"title": title, "set_default": False}
            if profile:
                payload["profile"] = profile
            resp = _http_json("POST", f"{v1_root}/projects", payload=payload, headers=headers, timeout=120.0)
            pid = str(resp.get("project_id") or "").strip()
            if not pid:
                raise RuntimeError(f"POST {v1_root}/projects returned no project_id: {json.dumps(resp)[:300]}")
            record = {
                "project_id": pid,
                "title": str(resp.get("title") or title),
                "profile": resp.get("profile") or profile or None,
                "url": resp.get("url"),
                "baseURL": origin_root,
                "createdAt": resp.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            tmp_file = manifest_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp_file.replace(manifest_file)
            emit_progress(stage="gflow", status="project_created", projectId=pid, title=record["title"])
            return record
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def build_composite_ref_sheet(
    ref_paths: list[Path],
    labels: list[str],
    out_path: Path,
    aspect: str = "16:9",
) -> Path:
    """Assemble multiple reference PNGs (place + active characters) into ONE clean labeled sheet."""
    existing = [(p, labels[i] if i < len(labels) else p.stem) for i, p in enumerate(ref_paths) if p.is_file()]
    if not existing:
        raise ValueError("No existing reference files to assemble")
    if len(existing) == 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(existing[0][0].read_bytes())
        return out_path

    n = len(existing)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    cell_w = 520
    cell_h = 520 if aspect == "1:1" else (640 if aspect == "9:16" else 360)
    cap_h = 38
    pad = 12
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * pad, rows * (cell_h + cap_h) + (rows + 1) * pad), (245, 242, 236))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(20)

    for idx, (p, lbl) in enumerate(existing):
        x = pad + (idx % cols) * (cell_w + pad)
        y = pad + (idx // cols) * (cell_h + cap_h + pad)
        with Image.open(p) as im:
            im_rgb = im.convert("RGB")
            contained = ImageOps.contain(im_rgb, (cell_w, cell_h), Image.LANCZOS)
            ox = x + (cell_w - contained.width) // 2
            oy = y + cap_h + (cell_h - contained.height) // 2
            draw.rectangle([x, y, x + cell_w, y + cap_h + cell_h], outline=(40, 36, 48), width=2, fill=(255, 255, 255))
            draw.rectangle([x, y, x + cell_w, y + cap_h], fill=(32, 28, 44))
            draw.text((x + 10, y + 8), str(lbl)[:42], fill=(255, 230, 170), font=font)
            sheet.paste(contained, (ox, oy))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")
    return out_path


def _load_font(size: int):
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _normalize_image_bytes(raw_bytes: bytes, out_path: Path, aspect: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = target_dimensions(aspect)
    with Image.open(io.BytesIO(raw_bytes)) as im:
        im_rgb = im.convert("RGB")
        if im_rgb.size != (w, h):
            im_rgb = ImageOps.fit(im_rgb, (w, h), method=Image.LANCZOS)
        im_rgb.save(out_path, format="PNG")
    return out_path


def _extract_media_name(resp: dict[str, Any]) -> str | None:
    if not isinstance(resp, dict):
        return None
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        mn = str(data[0].get("media_name") or "").strip()
        if mn:
            return mn
    mn = str(resp.get("media_name") or "").strip()
    return mn or None


def _extract_image_from_payload(
    resp: dict[str, Any], cfg: dict[str, Any], headers: dict[str, str], base_url: str = ""
) -> bytes | None:
    rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
    b64_custom = _get_by_path(resp, rp.get("b64")) if rp else None
    url_custom = _get_by_path(resp, rp.get("url")) if rp else None
    if isinstance(b64_custom, str) and b64_custom.strip():
        return base64.b64decode(b64_custom.split(",", 1)[-1])
    if isinstance(url_custom, str) and url_custom.strip():
        return _download_bytes(_resolve_url(url_custom, base_url) if base_url else url_custom, headers=headers)

    for key in ("b64_json", "image_base64", "base64", "image"):
        val = resp.get(key)
        if isinstance(val, str) and len(val) > 64:
            return base64.b64decode(val.split(",", 1)[-1])
    for key in ("image_url", "url"):
        val = resp.get(key)
        if isinstance(val, str) and (val.startswith(("http://", "https://")) or (base_url and val.startswith("/"))):
            return _download_bytes(_resolve_url(val, base_url) if base_url else val, headers=headers)

    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        for key in ("b64_json", "image_base64", "base64"):
            if isinstance(first.get(key), str) and first[key].strip():
                return base64.b64decode(first[key].split(",", 1)[-1])
        if isinstance(first.get("url"), str) and first["url"].strip():
            return _download_bytes(_resolve_url(first["url"], base_url) if base_url else first["url"], headers=headers)

    # Gemini generateContent format
    cands = resp.get("candidates")
    if isinstance(cands, list):
        for cand in cands:
            parts = ((cand or {}).get("content") or {}).get("parts") or []
            for p in parts:
                inl = (p or {}).get("inlineData") or (p or {}).get("inline_data")
                if isinstance(inl, dict) and isinstance(inl.get("data"), str):
                    return base64.b64decode(inl["data"])
    return None


def _poll_job(
    base_url: str,
    poll_tpl: str,
    job_id: str,
    headers: dict[str, str],
    cfg: dict[str, Any],
    kind: str,
    timeout_sec: float = 360.0,
    check_url: str | None = None,
) -> dict[str, Any]:
    poll_url = _resolve_url(check_url, base_url) if (isinstance(check_url, str) and check_url.strip()) else _join_url(base_url, poll_tpl.replace("{id}", job_id))
    t0 = time.monotonic()
    rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
    poll_interval = max(0.05, float(cfg.get("pollInterval") or 2.0))
    while time.monotonic() - t0 < timeout_sec:
        res = _http_json("GET", poll_url, headers=headers, timeout=60.0)
        status_val = _get_by_path(res, rp.get("status")) if rp.get("status") else None
        status = str(status_val or res.get("status") or res.get("state") or "").strip().lower()
        done_flag = res.get("done")
        if status in ("failed", "error", "cancelled", "canceled"):
            err = res.get("error") or res.get("message") or status
            raise RuntimeError(f"{kind} job {job_id} failed: {err}")
        if done_flag is True or status in ("succeeded", "completed", "done", "finished", "ready") or (
            not status and (res.get("url") or res.get("video_url") or res.get("b64_json") or res.get("data") or res.get("response"))
        ):
            return res
        emit_progress(stage=kind, status=status or "polling", jobId=job_id, elapsed=round(time.monotonic() - t0, 1))
        time.sleep(poll_interval)
    raise RuntimeError(f"{kind} job {job_id} timed out after {int(timeout_sec)}s")


def _mock_generate_image(
    prompt: str,
    aspect: str,
    out_path: Path,
    ref_images: list[Path] | None = None,
    ref_labels: list[str] | None = None,
    seed: int | None = None,
) -> Path:
    """Render a deterministic, richly-contrasted flat 2D illustration PNG with Pillow for offline testing."""
    w, h = target_dimensions(aspect)
    digest = hashlib.sha256(f"{prompt}|{aspect}|{seed or 0}".encode("utf-8")).digest()
    bg_palette = [
        ((214, 228, 212), (168, 190, 166), (112, 86, 62)),
        ((236, 224, 204), (206, 188, 164), (124, 92, 68)),
        ((204, 220, 238), (166, 188, 214), (82, 96, 118)),
        ((232, 212, 220), (196, 168, 182), (96, 78, 92)),
    ]
    wall_c, mid_c, floor_c = bg_palette[digest[0] % len(bg_palette)]
    im = Image.new("RGB", (w, h), wall_c)
    dr = ImageDraw.Draw(im)
    ink = (30, 24, 38)

    # Layer 1: Architectural wall panels, window + light shaft (ensures balanced contrast across all quadrants)
    horizon_y = int(h * 0.68)
    dr.rectangle([0, horizon_y, w, h], fill=floor_c)
    dr.line([(0, horizon_y), (w, horizon_y)], fill=ink, width=5)

    # Left & right wall framing / window / board so left, right, top, bottom all carry drawn features
    dr.rectangle([int(w * 0.06), int(h * 0.10), int(w * 0.34), int(h * 0.44)], fill=(248, 250, 255), outline=ink, width=5)
    dr.line([(int(w * 0.20), int(h * 0.10)), (int(w * 0.20), int(h * 0.44))], fill=ink, width=4)
    dr.line([(int(w * 0.06), int(h * 0.27)), (int(w * 0.34), int(h * 0.27))], fill=ink, width=4)

    dr.rectangle([int(w * 0.64), int(h * 0.12), int(w * 0.94), int(h * 0.42)], fill=mid_c, outline=ink, width=5)
    dr.rectangle([int(w * 0.68), int(h * 0.16), int(w * 0.90), int(h * 0.38)], fill=(244, 214, 152), outline=ink, width=3)

    # Layer 2: Midground furniture / desk / prop
    dr.rectangle([int(w * 0.16), int(h * 0.60), int(w * 0.84), int(h * 0.72)], fill=(156, 108, 66), outline=ink, width=5)
    dr.rectangle([int(w * 0.20), int(h * 0.72), int(w * 0.24), int(h * 0.88)], fill=(96, 66, 42), outline=ink, width=4)
    dr.rectangle([int(w * 0.76), int(h * 0.72), int(w * 0.80), int(h * 0.88)], fill=(96, 66, 42), outline=ink, width=4)

    # Characters (from labels / prompt hash)
    char_labels = [lbl for lbl in (ref_labels or []) if lbl.startswith("CHAR:")]
    n_chars = len(char_labels) if char_labels else (1 if "NO people" not in prompt and "NO characters" not in prompt else 0)
    outfit_colors = [(58, 134, 255), (224, 87, 128), (42, 157, 143), (244, 162, 97)]
    skin_colors = [(242, 210, 182), (216, 155, 114), (198, 136, 88)]
    for i in range(n_chars):
        cx = int(w * (0.5 if n_chars == 1 else (0.30 + i * (0.40 / max(1, n_chars - 1)))))
        cy_head = int(h * 0.40)
        head_r = int(min(w, h) * 0.085)
        body_w = int(head_r * 1.65)
        body_h = int(head_r * 2.4)
        c_out = outfit_colors[(digest[1 + i] + i) % len(outfit_colors)]
        c_skin = skin_colors[(digest[2 + i] + i) % len(skin_colors)]
        # Torso + arms
        dr.rounded_rectangle(
            [cx - body_w // 2, cy_head + head_r - 6, cx + body_w // 2, cy_head + head_r + body_h],
            radius=18,
            fill=c_out,
            outline=ink,
            width=5,
        )
        # Head + dot eyes + line mouth
        dr.ellipse([cx - head_r, cy_head - head_r, cx + head_r, cy_head + head_r], fill=c_skin, outline=ink, width=5)
        dr.ellipse([cx - head_r // 3 - 5, cy_head - 8, cx - head_r // 3 + 5, cy_head + 2], fill=ink)
        dr.ellipse([cx + head_r // 3 - 5, cy_head - 8, cx + head_r // 3 + 5, cy_head + 2], fill=ink)
        dr.arc([cx - 14, cy_head + 12, cx + 14, cy_head + 28], start=10, end=170, fill=ink, width=4)

    # Layer 3: Subtle floor highlight & optional corner prop
    dr.ellipse([int(w * 0.08), int(h * 0.78), int(w * 0.28), int(h * 0.92)], fill=(78, 146, 102), outline=ink, width=4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")
    meta = {
        "driver": "mock",
        "aspect": aspect,
        "size": [w, h],
        "refCount": len(ref_images or []),
        "refLabels": ref_labels or [],
        "promptPreview": prompt[:240],
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def generate_image(
    prompt: str,
    negative_prompt: str,
    aspect: str,
    out_path: Path,
    ref_images: list[Path] | None = None,
    ref_labels: list[str] | None = None,
    seed: int | None = None,
    cfg: dict[str, Any] | None = None,
    ref_media_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a single 2D PNG image with the configured diffusion image provider."""
    cfg = cfg or load_media_config()["image"]
    driver = str(cfg.get("driver") or "mock").lower()
    model = str(cfg.get("model") or "nano2-lite")
    ref_mode = str(cfg.get("refMode") or "composite").lower()
    max_refs = int(cfg.get("maxRefs") or 4)
    refs = [p for p in (ref_images or []) if p.is_file()][:max_refs]
    labels = (ref_labels or [])[: len(refs)]

    t0 = time.monotonic()
    if driver == "mock":
        _mock_generate_image(prompt, aspect, out_path, refs, labels, seed)
        return {"ok": True, "driver": "mock", "model": model, "seconds": round(time.monotonic() - t0, 2), "credits": 1}

    base_url = str(cfg.get("baseURL") or "").strip()
    if not base_url:
        raise RuntimeError(f"Image provider '{driver}' has no baseURL configured")
    headers = _auth_headers(cfg)
    w, h = target_dimensions(aspect)
    extra = cfg.get("extraBody") if isinstance(cfg.get("extraBody"), dict) else {}

    if driver == "gemini":
        ep = _join_url(base_url, f"/models/{urllib.parse.quote(model, safe='')}:generateContent")
        parts: list[dict[str, Any]] = []
        if ref_mode != "none":
            for rp in refs:
                parts.append({"inlineData": {"mimeType": "image/png", "data": _encode_b64(rp)}})
        full_text = prompt if not negative_prompt else f"{prompt}\nNegative instructions (avoid): {negative_prompt}"
        parts.append({"text": full_text})
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect if aspect in ("16:9", "9:16", "1:1") else "16:9"},
            },
            **extra,
        }
        resp = _http_json("POST", ep, payload=body, headers=headers, timeout=240.0)
        raw_bytes = _extract_image_from_payload(resp, cfg, headers)
        if not raw_bytes:
            raise RuntimeError(f"Gemini image API returned no image bytes: {json.dumps(resp)[:400]}")
        _normalize_image_bytes(raw_bytes, out_path, aspect)
        return {"ok": True, "driver": driver, "model": model, "seconds": round(time.monotonic() - t0, 2), "credits": 5}

    if driver == "gflow":
        origin_root, _v1_root = _gflow_urls(base_url)
        ep_url = _gflow_endpoint(base_url, str(cfg.get("endpointPath") or ""), "/images/generations")
        gflow_aspect = aspect if aspect in ("9:16", "16:9", "1:1", "4:3", "3:4") else "16:9"
        gflow_prompt = prompt if not negative_prompt else f"{prompt}\nAvoid: {negative_prompt}"
        primary_model = _normalize_gflow_image_model(model)
        fb_raw = str(cfg.get("fallbackModel") or "").strip()
        fallback_model = _normalize_gflow_image_model(fb_raw) if fb_raw else ""
        clean_media_ids = [str(m).strip() for m in (ref_media_ids or []) if isinstance(m, str) and str(m).strip()]

        if ref_mode == "multi" and max(len(refs), len(clean_media_ids)) > 3 and primary_model in ("nano2-lite", "image4"):
            primary_model = fallback_model or "nano2.1"

        def _run_gflow_img(use_model: str) -> tuple[bytes, str | None]:
            body_g: dict[str, Any] = {
                "prompt": gflow_prompt,
                "model": use_model,
                "aspect": gflow_aspect,
                "n": 1,
                "wait": False,
                **({"project": str(cfg["project"]).strip()} if cfg.get("project") else {}),
                **({"profile": str(cfg["profile"]).strip()} if cfg.get("profile") else {}),
                **extra,
            }
            tmp_comp: Path | None = None
            try:
                ref_cap = 3 if use_model in ("nano2-lite", "image4") else 10
                if ref_mode == "multi" and clean_media_ids and (not refs or len(clean_media_ids) == len(refs)):
                    body_g["ref_media_ids"] = clean_media_ids[:ref_cap]
                elif refs and ref_mode == "composite":
                    tmp_comp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
                    build_composite_ref_sheet(refs, labels, tmp_comp, aspect=aspect)
                    body_g["image_base64"] = _encode_b64(tmp_comp)
                elif refs and ref_mode == "multi":
                    if len(refs) == 1:
                        body_g["image_base64"] = _encode_b64(refs[0])
                    else:
                        body_g["images_base64"] = [_encode_b64(p) for p in refs[:ref_cap]]
                resp = _http_json("POST", ep_url, payload=body_g, headers=headers, timeout=240.0)
            finally:
                if tmp_comp and tmp_comp.exists():
                    tmp_comp.unlink(missing_ok=True)

            rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
            job_id = _get_by_path(resp, rp.get("jobId")) if rp.get("jobId") else (resp.get("job_id") or resp.get("id") or resp.get("task_id"))
            status_now = str(resp.get("status") or "").strip().lower()
            if status_now in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(f"gflow image job {job_id or ''} failed: {resp.get('error') or status_now}")
            final_payload = resp
            img_bytes = _extract_image_from_payload(resp, cfg, headers, origin_root)
            if not img_bytes and isinstance(job_id, str) and job_id.strip():
                poll_tpl = str(cfg.get("pollPath") or "/v1/jobs/{id}")
                final_payload = _poll_job(
                    origin_root,
                    poll_tpl,
                    job_id.strip(),
                    headers,
                    cfg,
                    kind="image",
                    timeout_sec=420.0,
                    check_url=resp.get("check_url"),
                )
                img_bytes = _extract_image_from_payload(final_payload, cfg, headers, origin_root)
            if not img_bytes:
                raise RuntimeError(f"gflow image returned no image bytes: {json.dumps(resp)[:400]}")
            return img_bytes, _extract_media_name(final_payload)

        used_model = primary_model
        try:
            raw_bytes, media_name = _run_gflow_img(primary_model)
        except RuntimeError as first_err:
            if fallback_model and fallback_model != primary_model:
                emit_progress(stage="image", status="fallback", fromModel=primary_model, toModel=fallback_model, reason=str(first_err)[:200])
                used_model = fallback_model
                raw_bytes, media_name = _run_gflow_img(fallback_model)
            else:
                raise

        _normalize_image_bytes(raw_bytes, out_path, aspect)
        return {
            "ok": True,
            "driver": driver,
            "model": used_model,
            **({"mediaName": media_name} if media_name else {}),
            **({"projectId": cfg.get("project")} if cfg.get("project") else {}),
            "seconds": round(time.monotonic() - t0, 2),
            "credits": 5,
        }

    # openai / http
    ep_path = str(cfg.get("endpointPath") or "/images/generations")
    ep_url = _join_url(base_url, ep_path)
    body = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect,
        "ratio": aspect,
        "size": f"{w}x{h}",
        "response_format": "b64_json",
        **({"negative_prompt": negative_prompt} if negative_prompt else {}),
        **({"seed": seed} if seed is not None else {}),
        **extra,
    }

    tmp_comp: Path | None = None
    try:
        if refs and ref_mode == "composite":
            tmp_comp = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            build_composite_ref_sheet(refs, labels, tmp_comp, aspect=aspect)
            body["image_base64"] = _encode_b64(tmp_comp)
            body["reference_image"] = _data_uri(tmp_comp)
        elif refs and ref_mode == "multi":
            uris = [_data_uri(p) for p in refs]
            body["images"] = uris
            body["ref_images_base64"] = [_encode_b64(p) for p in refs]
            if len(uris) == 1:
                body["image_base64"] = _encode_b64(refs[0])

        resp = _http_json("POST", ep_url, payload=body, headers=headers, timeout=240.0)
    finally:
        if tmp_comp and tmp_comp.exists():
            tmp_comp.unlink(missing_ok=True)

    rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
    job_id = _get_by_path(resp, rp.get("jobId")) if rp.get("jobId") else (resp.get("job_id") or resp.get("id") or resp.get("task_id"))
    raw_bytes = _extract_image_from_payload(resp, cfg, headers)
    if not raw_bytes and isinstance(job_id, str) and job_id.strip():
        poll_tpl = str(cfg.get("pollPath") or "/jobs/{id}")
        polled = _poll_job(base_url, poll_tpl, job_id.strip(), headers, cfg, kind="image")
        raw_bytes = _extract_image_from_payload(polled, cfg, headers)

    if not raw_bytes:
        raise RuntimeError(f"Image provider '{driver}' returned neither image bytes nor URL: {json.dumps(resp)[:400]}")
    _normalize_image_bytes(raw_bytes, out_path, aspect)
    return {"ok": True, "driver": driver, "model": model, "seconds": round(time.monotonic() - t0, 2), "credits": 5}


def _mock_generate_video(
    image_path: Path,
    prompt: str,
    duration_sec: float,
    aspect: str,
    out_path: Path,
    seed: int | None = None,
    resolution: str = "1080p",
) -> Path:
    """Create a deterministic H.264 MP4 from image_path with subtle zoom/pan + synthesized physical SFX (no voice/music)."""
    w, h = target_dimensions(aspect, resolution)
    dur = max(1.0, float(duration_sec))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic motion direction from prompt hash
    h_val = hashlib.sha256(f"{prompt}|{seed or 0}".encode("utf-8")).digest()[0]
    zoom_dir = "0.0012" if (h_val % 2 == 0) else "-0.0010"
    start_z = "1.0" if (h_val % 2 == 0) else "1.06"
    # Synthesize subtle room tone + a physical foley thud/click at t=0.35s (strictly SFX, no music/speech)
    sfx_expr = (
        "0.03*sin(2*PI*95*t)*exp(-1.5*mod(t,2.0))"
        "+0.12*if(between(t,0.30,0.48),sin(2*PI*160*(t-0.30))*exp(-18*(t-0.30)),0)"
        "+0.015*cos(2*PI*52*t)"
    )
    vf = (
        f"scale={w*2}:{h*2},zoompan=z='if(eq(on,1),{start_z},max(1.0,min(1.12,zoom+{zoom_dir})))'"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps=30,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-framerate", "30", "-t", f"{dur:.3f}", "-i", str(image_path),
        "-f", "lavfi", "-i", f"aevalsrc='{sfx_expr}':s=48000:d={dur:.3f}",
        "-vf", vf,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
        "-shortest", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _extract_video_from_payload(
    resp: dict[str, Any], cfg: dict[str, Any], headers: dict[str, str], base_url: str = ""
) -> bytes | None:
    rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
    b64_custom = _get_by_path(resp, rp.get("b64")) if rp else None
    url_custom = _get_by_path(resp, rp.get("url")) if rp else None
    if isinstance(b64_custom, str) and b64_custom.strip():
        return base64.b64decode(b64_custom.split(",", 1)[-1])
    if isinstance(url_custom, str) and url_custom.strip():
        return _download_bytes(_resolve_url(url_custom, base_url) if base_url else url_custom, headers=headers, timeout=300.0)

    for key in ("b64_json", "video_base64", "base64"):
        val = resp.get(key)
        if isinstance(val, str) and len(val) > 64:
            return base64.b64decode(val.split(",", 1)[-1])
    for key in ("video_url", "output_url", "url"):
        val = resp.get(key)
        if isinstance(val, str) and (val.startswith(("http://", "https://")) or (base_url and val.startswith("/"))):
            return _download_bytes(_resolve_url(val, base_url) if base_url else val, headers=headers, timeout=300.0)

    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        if isinstance(first.get("b64_json"), str) and first["b64_json"].strip():
            return base64.b64decode(first["b64_json"].split(",", 1)[-1])
        for k in ("video_url", "url"):
            if isinstance(first.get(k), str) and first[k].strip():
                return _download_bytes(_resolve_url(first[k], base_url) if base_url else first[k], headers=headers, timeout=300.0)

    # Gemini operation / generateContent video payloads
    op_resp = resp.get("response") if isinstance(resp.get("response"), dict) else None
    if op_resp:
        samples = (op_resp.get("generateVideoResponse") or {}).get("generatedSamples") or []
        if isinstance(samples, list) and samples:
            vid = (samples[0] or {}).get("video") or {}
            if isinstance(vid.get("uri"), str) and vid["uri"].strip():
                return _download_bytes(vid["uri"], headers=headers, timeout=300.0)
            if isinstance(vid.get("bytesBase64Encoded"), str):
                return base64.b64decode(vid["bytesBase64Encoded"])
        nested = _extract_video_from_payload(op_resp, cfg, headers, base_url)
        if nested:
            return nested

    cands = resp.get("candidates")
    if isinstance(cands, list):
        for cand in cands:
            parts = ((cand or {}).get("content") or {}).get("parts") or []
            for p in parts:
                inl = (p or {}).get("inlineData") or (p or {}).get("inline_data")
                if isinstance(inl, dict) and str(inl.get("mimeType", "")).startswith("video/") and isinstance(inl.get("data"), str):
                    return base64.b64decode(inl["data"])
                fdata = (p or {}).get("fileData") or (p or {}).get("file_data")
                if isinstance(fdata, dict) and isinstance(fdata.get("fileUri"), str):
                    return _download_bytes(fdata["fileUri"], headers=headers, timeout=300.0)
    return None


def generate_video(
    image_path: Path,
    prompt: str,
    duration_sec: float,
    aspect: str,
    out_path: Path,
    seed: int | None = None,
    cfg: dict[str, Any] | None = None,
    start_media_id: str | None = None,
) -> dict[str, Any]:
    """Animate a storyboard still into an MP4 clip (with native SFX only) via Gemini Omni Flash 2.1 or configured provider."""
    cfg = cfg or load_media_config()["video"]
    driver = str(cfg.get("driver") or "mock").lower()
    model = str(cfg.get("model") or "gemini-omni-flash-2.1")
    resolution = str(cfg.get("resolution") or os.environ.get("WHATIF_VIDEO_RESOLUTION") or "1080p").strip().lower()
    if resolution not in ("360p", "720p", "1080p"):
        resolution = "1080p"
    w, h = target_dimensions(aspect, resolution)

    t0 = time.monotonic()
    if driver == "mock":
        _mock_generate_video(image_path, prompt, duration_sec, aspect, out_path, seed, resolution=resolution)
        return {"ok": True, "driver": "mock", "model": model, "resolution": resolution, "seconds": round(time.monotonic() - t0, 2), "credits": 2}

    base_url = str(cfg.get("baseURL") or "").strip()
    if not base_url:
        raise RuntimeError(f"Video provider '{driver}' has no baseURL configured")
    headers = _auth_headers(cfg)
    extra = cfg.get("extraBody") if isinstance(cfg.get("extraBody"), dict) else {}

    if driver == "gemini":
        ep_tpl = str(cfg.get("endpointPath") or "").strip()
        ep_url = _join_url(
            base_url,
            ep_tpl or f"/models/{urllib.parse.quote(model, safe='')}:predictLongRunning",
        )
        b64_img = _encode_b64(image_path)
        body: dict[str, Any] = {
            "instances": [
                {
                    "prompt": prompt,
                    "image": {"bytesBase64Encoded": b64_img, "mimeType": "image/png"},
                }
            ],
            "parameters": {
                "aspectRatio": aspect if aspect in ("16:9", "9:16") else "16:9",
                "durationSeconds": max(2, min(15, int(math.ceil(duration_sec)))),
                "generateAudio": True,
                **({"resolution": resolution} if resolution in ("720p", "1080p") else {}),
                **({"seed": seed} if seed is not None else {}),
            },
            **extra,
        }
        resp = _http_json("POST", ep_url, payload=body, headers=headers, timeout=240.0)
        op_name = resp.get("name")
        if isinstance(op_name, str) and op_name.strip() and not resp.get("done"):
            poll_tpl = str(cfg.get("pollPath") or "/{id}")
            resp = _poll_job(base_url, poll_tpl, op_name.lstrip("/"), headers, cfg, kind="video", timeout_sec=480.0)
        raw_bytes = _extract_video_from_payload(resp, cfg, headers)
        if not raw_bytes:
            raise RuntimeError(f"Gemini video API returned no video payload: {json.dumps(resp)[:400]}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw_bytes)
        return {"ok": True, "driver": driver, "model": model, "resolution": resolution, "seconds": round(time.monotonic() - t0, 2), "credits": 20}

    if driver == "gflow":
        origin_root, _v1_root = _gflow_urls(base_url)
        ep_url = _gflow_endpoint(base_url, str(cfg.get("endpointPath") or ""), "/videos/generations")
        use_model = _normalize_gflow_video_model(model)
        if "omni-1.2" in use_model:
            dur_int = max(3, min(15, int(math.ceil(float(duration_sec)))))
        elif "omni" in use_model:
            dur_int = next((d for d in (4, 6, 8, 10) if d >= float(duration_sec)), 10)
        else:
            dur_int = next((d for d in (4, 6, 8) if d >= float(duration_sec)), 8)

        gflow_res: str | None = None
        if "omni" in use_model:
            gflow_res = "360p" if resolution == "360p" else "720p"

        clean_start_id = str(start_media_id or "").strip()
        body_g: dict[str, Any] = {
            "prompt": prompt,
            "model": use_model,
            "mode": "i2v",
            **({"initial_frame": clean_start_id} if clean_start_id else {"image_base64": _encode_b64(image_path)}),
            "aspect": aspect if aspect in ("9:16", "16:9", "1:1") else "16:9",
            "duration": dur_int,
            "n": 1,
            "wait": False,
            **({"resolution": gflow_res} if gflow_res else {}),
            **({"project": str(cfg["project"]).strip()} if cfg.get("project") else {}),
            **({"profile": str(cfg["profile"]).strip()} if cfg.get("profile") else {}),
            **extra,
        }
        resp = _http_json("POST", ep_url, payload=body_g, headers=headers, timeout=240.0)
        rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
        job_id = _get_by_path(resp, rp.get("jobId")) if rp.get("jobId") else (resp.get("job_id") or resp.get("id") or resp.get("task_id"))
        status_now = str(resp.get("status") or "").strip().lower()
        if status_now in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"gflow video job {job_id or ''} failed: {resp.get('error') or status_now}")
        final_payload = resp
        raw_bytes = _extract_video_from_payload(resp, cfg, headers, origin_root)
        if not raw_bytes and isinstance(job_id, str) and job_id.strip():
            poll_tpl = str(cfg.get("pollPath") or "/v1/jobs/{id}")
            final_payload = _poll_job(
                origin_root,
                poll_tpl,
                job_id.strip(),
                headers,
                cfg,
                kind="video",
                timeout_sec=600.0,
                check_url=resp.get("check_url"),
            )
            raw_bytes = _extract_video_from_payload(final_payload, cfg, headers, origin_root)
        if not raw_bytes:
            raise RuntimeError(f"gflow video returned neither video bytes nor URL: {json.dumps(resp)[:400]}")
        media_name = _extract_media_name(final_payload)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw_bytes)
        return {
            "ok": True,
            "driver": driver,
            "model": use_model,
            "resolution": gflow_res or resolution,
            "durationRequested": dur_int,
            **({"mediaName": media_name} if media_name else {}),
            **({"projectId": cfg.get("project")} if cfg.get("project") else {}),
            "seconds": round(time.monotonic() - t0, 2),
            "credits": 20,
        }

    # openai / http
    ep_path = str(cfg.get("endpointPath") or "/videos/generations")
    ep_url = _join_url(base_url, ep_path)
    body = {
        "model": model,
        "prompt": prompt,
        "image_base64": _encode_b64(image_path),
        "image_url": _data_uri(image_path),
        "aspect_ratio": aspect,
        "resolution": resolution,
        "size": f"{w}x{h}",
        "duration": round(max(1.5, float(duration_sec)), 2),
        "generate_audio": True,
        **({"seed": seed} if seed is not None else {}),
        **extra,
    }
    resp = _http_json("POST", ep_url, payload=body, headers=headers, timeout=240.0)
    rp = cfg.get("responsePaths") if isinstance(cfg.get("responsePaths"), dict) else {}
    job_id = _get_by_path(resp, rp.get("jobId")) if rp.get("jobId") else (resp.get("job_id") or resp.get("id") or resp.get("task_id"))
    raw_bytes = _extract_video_from_payload(resp, cfg, headers)
    if not raw_bytes and isinstance(job_id, str) and job_id.strip():
        poll_tpl = str(cfg.get("pollPath") or "/videos/{id}")
        polled = _poll_job(base_url, poll_tpl, job_id.strip(), headers, cfg, kind="video", timeout_sec=480.0)
        raw_bytes = _extract_video_from_payload(polled, cfg, headers)

    if not raw_bytes:
        raise RuntimeError(f"Video provider '{driver}' returned neither video bytes nor URL: {json.dumps(resp)[:400]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)
    return {"ok": True, "driver": driver, "model": model, "seconds": round(time.monotonic() - t0, 2), "credits": 20}


def append_cost(out_dir: Path, entry: dict[str, Any]) -> None:
    costs_path = out_dir / "costs.json"
    try:
        cur = json.loads(costs_path.read_text(encoding="utf-8")) if costs_path.is_file() else []
        if not isinstance(cur, list):
            cur = []
    except Exception:
        cur = []
    cur.append({**entry, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    costs_path.parent.mkdir(parents=True, exist_ok=True)
    costs_path.write_text(json.dumps(cur, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
