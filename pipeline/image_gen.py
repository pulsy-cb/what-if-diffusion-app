#!/usr/bin/env python3
"""Generate neutral reference sheets (`--mode refs`) and per-shot storyboard stills (`--mode stills`)
using Nano Banana 2 Lite (or any configured image diffusion provider).

    python3 pipeline/image_gen.py <slug> --mode refs [--only CHAR1,PLACE1] [--force]
    python3 pipeline/image_gen.py <slug> --mode stills [S3 S7 ...] [--force]
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

DEFAULT_NEGATIVE = (
    "realistic human proportions, adult comic book anatomy, Western superhero comic, Marvel DC graphic novel, "
    "anime, manga, semi-realistic face, chiseled jawline, facial wrinkles, detailed nostrils, muscular realism, "
    "photorealistic, 3D CGI, Pixar, Unreal Engine, blurry muddy shading, "
    "title card, opening title, subtitles, caption bar, watermark, signature, "
    "random background text, poster words, wall signs, gibberish letters, floating objects, "
    "pale oval disc under feet, pedestal base, white circle under shoes, "
    "extra people, duplicate characters, clones, deformed hands"
)

SIGNATURE_STYLE_LOCK = (
    "Minimalist 2D 'What If' explainer cartoon animation style (NOT realistic comic-book, NOT anime): "
    "characters have simplified stylized proportions with oversized smooth round/oval heads (~1:3.5 head-to-body ratio), "
    "slender simplified limbs and rounded cartoon hands, bold uniform dark charcoal/ink outlines, "
    "simple crisp 2D facial features (clean oval or dot eyes with white catchlights, thick expressive floating eyebrows, "
    "minimalist line/D/O mouth, tiny simple C-curve nose), flat cel-shaded solid clothing blocks (one crisp shadow tone), "
    "and a signature warm golden-amber outer rim-light edge glow tracing every character silhouette against darker moody sets. "
    "Atmospheric matte 2D editorial backgrounds with mid-dark muted walls, visible practical light sources casting soft "
    "volumetric pools/cones of light, subtle floating dust motes, subtle dark semi-transparent cast shadows on the floor "
    "(strictly NO pale grey/beige oval disc or pedestal under feet), "
    "and clear 3-layer depth (softly blurred foreground edge or bokeh background when appropriate)."
)


def combine_desc(primary: Any, extra: Any) -> str:
    p = str(primary or "").strip()
    e = str(extra or "").strip()
    if p and e and e.lower() not in p.lower():
        return f"{p.rstrip('. ')}. {e}"
    return p or e


def framing_directive(size_lbl: str, view_lbl: str, angle_lbl: str, cam_lbl: str, aspect: str) -> str:
    sz = size_lbl.lower().strip()
    vw = view_lbl.lower().strip()
    ang = angle_lbl.lower().strip()
    cam = cam_lbl.lower().strip()

    parts: list[str] = []
    if sz in ("insert", "detail", "macro"):
        parts.append(
            "MACRO / TIGHT OBJECT INSERT — Focal object or hands fill 70%+ of the frame; shallow depth of field with heavily blurred background; "
            "DO NOT draw full-body figures or wide exterior architecture"
        )
    elif sz in ("close", "closeup"):
        parts.append(
            "CLOSE-UP PORTRAIT — Tight head-and-shoulders framing; character's round cartoon head and expressive face dominate the frame with soft background bokeh"
        )
    elif sz == "ots" or cam == "ots":
        parts.append(
            "OVER-THE-SHOULDER SHOT — Dark softly blurred foreground shoulder/side-of-head framing one edge looking across toward the subject"
        )
    elif sz in ("wide", "establishing"):
        parts.append(
            f"{sz.upper()} SHOT — Show full room architecture, floor contact shadows, and spatial relationship across frame"
        )
    else:
        parts.append(f"{sz.upper()} SHOT — Knee-up or waist-up clear cartoon character staging inside the room")

    if vw == "reverse":
        parts.append("REVERSE / CONTRECHAMP CAMERA — Camera sits 180 degrees opposite, looking toward the opposite wall/exit while preserving left-right continuity")
    elif vw and vw != "front":
        parts.append(f"camera view={vw}")

    if ang == "high":
        parts.append("HIGH ANGLE looking diagonally downward")
    elif ang == "low":
        parts.append("LOW ANGLE looking upward for dramatic scale")
    else:
        parts.append("eye-level horizon")

    if aspect == "9:16":
        parts.append("9:16 vertical Shorts framing (keep faces and focal hands centered horizontally inside the safe middle 65% band away from top/bottom overlays)")
    else:
        parts.append("16:9 widescreen framing (balanced headroom, clean rule-of-thirds left/right blocking)")
    return "; ".join(parts) + "."


def load_stage_cards(ep_dir: Path, ep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for s in ep.get("shots") or []:
        if isinstance(s, dict) and s.get("id") and isinstance(s.get("stage"), dict):
            by_id[str(s["id"]).strip().upper()] = s["stage"]
    stage_file = load_json(ep_dir / "shots.stage.json")
    if isinstance(stage_file, dict):
        raw_cards = stage_file.get("cards") if isinstance(stage_file.get("cards"), dict) else stage_file
        if isinstance(raw_cards, dict):
            for k, v in raw_cards.items():
                if isinstance(v, dict):
                    by_id.setdefault(str(k).strip().upper(), v)
    return by_id


def stage_blocking_line(card: dict[str, Any] | None) -> str:
    if not isinstance(card, dict):
        return ""
    bits: list[str] = []
    if card.get("zone"):
        bits.append(f"room zone={card['zone']}")
    if card.get("foreground") and str(card.get("foreground")).lower() not in ("none", ""):
        bits.append(f"blurred foreground framing={card['foreground']}")
    if card.get("subject"):
        bits.append(f"focal subject={card['subject']}")
    ppl = card.get("people") if isinstance(card.get("people"), list) else []
    if ppl:
        pos = []
        for p in ppl:
            if isinstance(p, dict):
                who = str(p.get("who") or "").strip()
                where = str(p.get("where") or "").strip()
                does = str(p.get("does") or "").strip()
                pos.append(f"{who} at [{where}] doing: {does}")
        if pos:
            bits.append("character screen positions: " + " | ".join(pos))
    objs = [str(o).strip() for o in (card.get("objects") or []) if str(o).strip()]
    if objs:
        bits.append("firmly held/placed props: " + ", ".join(objs))
    if not bits:
        return ""
    return (
        "SPATIAL PLACEMENT & FRAME-0 POSE — " + "; ".join(bits) +
        ". Ground all feet, furniture and resting props with subtle dark semi-transparent cast shadows on the floor (strictly NO pale grey/beige oval disc or pedestal under feet); depict Frame 0 of the upcoming motion (no floating objects)."
    )


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


def run_refs(root: Path, slug: str, only_ids: set[str], force: bool, conc: int) -> int:
    ep_dir = root / "src" / "episodes" / slug
    out_dir = root / "out" / slug
    refs_dir = out_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    ep = load_json(ep_dir / "episode.json", {}) or {}
    bible = load_json(ep_dir / "bible.json")
    if not isinstance(bible, dict):
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"No bible.json at {ep_dir / 'bible.json'}"}))
        return 2

    aspect = str(ep.get("aspect") or "16:9").strip()
    if aspect not in ("9:16", "16:9"):
        aspect = "16:9"

    cfg = mc.load_media_config()["image"]
    driver = str(cfg.get("driver") or "mock")
    model = str(cfg.get("model") or "nano2-lite")
    style = str(bible.get("stylePrompt") or "").strip()
    bible_neg = str(bible.get("negativePrompt") or "").strip()
    base_neg = ", ".join(x for x in [DEFAULT_NEGATIVE, bible_neg] if x)

    manifest_path = refs_dir / "manifest.json"
    manifest = load_json(manifest_path, {}) or {}
    tasks: list[dict[str, Any]] = []

    for pl in bible.get("places") or []:
        pid = str(pl.get("id") or "").strip()
        if not pid or (only_ids and pid.upper() not in only_ids):
            continue
        desc = combine_desc(pl.get("description"), pl.get("refPrompt"))
        prompt = (
            f"STYLE LOCK — {style}. {SIGNATURE_STYLE_LOCK}\n"
            f"REFERENCE ENVIRONMENT — Empty 2D background plate for \"{pl.get('name', pid)}\" "
            f"({aspect} frame). Strictly NO people, NO characters, NO silhouettes, NO random poster/wall text, NO title cards, NO captions.\n"
            f"LAYOUT & LIGHTING — {desc}"
        )
        out_file = refs_dir / f"place_{pid}.png"
        sig = sha256_str(f"{driver}|{model}|{aspect}|{prompt}|{base_neg}")
        tasks.append({
            "key": f"place_{pid}",
            "label": f"place {pid}",
            "file": out_file,
            "prompt": prompt,
            "negative": f"{base_neg}, people, person, character, crowd, text",
            "aspect": aspect,
            "sig": sig,
        })

    for ch in bible.get("characters") or []:
        cid = str(ch.get("id") or "").strip()
        if not cid or (only_ids and cid.upper() not in only_ids):
            continue
        desc = combine_desc(ch.get("appearance"), ch.get("refPrompt"))
        prompt = (
            f"STYLE LOCK — {style}. {SIGNATURE_STYLE_LOCK}\n"
            f"CHARACTER MODEL SHEET — Single minimalist round-head 2D cartoon character ONLY (\"{ch.get('name', cid)}\"), "
            f"oversized smooth round head (~1:3.5 head-to-body ratio), bold uniform dark charcoal outlines, simple clean dot/oval eyes + thick eyebrows + clean mouth, "
            f"full-body standing front-3/4 neutral pose centered on a clean plain light neutral cream/grey background, "
            f"NO floor shadow disk or pedestal under shoes (clean floating feet on plain light background). "
            f"Strictly NO extra characters, NO background scenery clutter, NO labels, NO text boxes.\n"
            f"CHARACTER DESIGN & EXACT WARDROBE — {desc}"
        )
        out_file = refs_dir / f"char_{cid}.png"
        sig = sha256_str(f"{driver}|{model}|1:1|{prompt}|{base_neg}")
        tasks.append({
            "key": f"char_{cid}",
            "label": f"char {cid}",
            "file": out_file,
            "prompt": prompt,
            "negative": f"{base_neg}, multiple people, crowd, text, labels, character sheet grid",
            "aspect": "1:1",
            "sig": sig,
        })

    if driver == "gflow" and not cfg.get("project"):
        gflow_proj = mc.ensure_gflow_project(root, slug, cfg)
        if gflow_proj and gflow_proj.get("project_id"):
            cfg = {**cfg, "project": gflow_proj["project_id"]}
    active_proj = str(cfg.get("project") or "").strip()

    generated: list[str] = []
    cached: list[str] = []
    to_run: list[dict[str, Any]] = []
    for t in tasks:
        prev = manifest.get(t["key"]) if isinstance(manifest.get(t["key"]), dict) else None
        gflow_ok = driver != "gflow" or (bool(prev and prev.get("mediaName")) and (not active_proj or (prev and prev.get("projectId") == active_proj)))
        if not force and t["file"].is_file() and prev and prev.get("hash") == t["sig"] and gflow_ok:
            cached.append(t["key"])
        else:
            to_run.append(t)

    credits_total = 0
    if to_run:
        workers = min(conc, 2) if driver == "gflow" else conc
        mc.emit_progress(stage="refs", done=0, total=len(to_run), detail=f"Generating {len(to_run)} reference(s) with {model}")
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_run)))) as pool:
            futs = {
                pool.submit(
                    mc.generate_image,
                    t["prompt"],
                    t["negative"],
                    t["aspect"],
                    t["file"],
                    None,
                    [t["label"]],
                    None,
                    cfg,
                ): t
                for t in to_run
            }
            done_n = 0
            for fut in as_completed(futs):
                t = futs[fut]
                res = fut.result()
                done_n += 1
                credits_total += int(res.get("credits") or 0)
                manifest[t["key"]] = {
                    "hash": t["sig"],
                    "file": str(t["file"].relative_to(root).as_posix()),
                    "driver": res.get("driver"),
                    "model": res.get("model"),
                    **({"mediaName": res.get("mediaName")} if res.get("mediaName") else {}),
                    **({"projectId": res.get("projectId") or active_proj} if (res.get("projectId") or active_proj) else {}),
                }
                generated.append(t["key"])
                mc.append_cost(out_dir, {"step": "refs", "item": t["key"], **res})
                mc.emit_progress(stage="refs", done=done_n, total=len(to_run), detail=f"{t['key']} ready ({res.get('seconds')}s)")

    save_json(manifest_path, manifest)
    files = [str((refs_dir / f"{t['key']}.png").relative_to(root).as_posix()) for t in tasks]
    print(f"[refs] {len(generated)} generated, {len(cached)} cached ({model} via {driver})")
    print("RESULT_JSON:" + json.dumps({
        "ok": True,
        "mode": "refs",
        "generated": generated,
        "cached": cached,
        "files": files,
        **({"gflowProjectId": cfg.get("project")} if cfg.get("project") else {}),
        "credits": credits_total,
    }))
    return 0


def run_stills(root: Path, slug: str, shot_filter: set[str], force: bool, conc: int) -> int:
    ep_dir = root / "src" / "episodes" / slug
    out_dir = root / "out" / slug
    refs_dir = out_dir / "refs"
    board_dir = out_dir / "board"
    board_dir.mkdir(parents=True, exist_ok=True)

    ep = load_json(ep_dir / "episode.json", {}) or {}
    tl = load_json(ep_dir / "timeline.json", {}) or {}
    bible = load_json(ep_dir / "bible.json")
    if not isinstance(bible, dict):
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"No bible.json at {ep_dir / 'bible.json'}"}))
        return 2

    aspect = str(ep.get("aspect") or tl.get("aspect") or "16:9").strip()
    if aspect not in ("9:16", "16:9"):
        aspect = "16:9"
    lang = "English" if str(ep.get("lang") or "fr").strip().lower() == "en" else "French"

    places = {str(p.get("id")): p for p in (bible.get("places") or []) if isinstance(p, dict) and p.get("id")}
    chars = {str(c.get("id")): c for c in (bible.get("characters") or []) if isinstance(c, dict) and c.get("id")}
    style = str(bible.get("stylePrompt") or "").strip()
    bible_neg = str(bible.get("negativePrompt") or "").strip()
    base_neg = ", ".join(x for x in [DEFAULT_NEGATIVE, bible_neg] if x)
    stage_cards = load_stage_cards(ep_dir, ep)

    cfg = mc.load_media_config()["image"]
    driver = str(cfg.get("driver") or "mock")
    model = str(cfg.get("model") or "nano2-lite")
    ref_mode = str(cfg.get("refMode") or "composite")

    tl_shots = [s for s in (tl.get("shots") or []) if isinstance(s, dict) and s.get("id")]
    if not tl_shots:
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": "timeline.json has no shots"}))
        return 2

    if driver == "gflow" and not cfg.get("project"):
        gflow_proj = mc.ensure_gflow_project(root, slug, cfg)
        if gflow_proj and gflow_proj.get("project_id"):
            cfg = {**cfg, "project": gflow_proj["project_id"]}
    active_proj = str(cfg.get("project") or "").strip()

    refs_manifest = load_json(refs_dir / "manifest.json", {}) or {}
    manifest_path = board_dir / "manifest.json"
    manifest = load_json(manifest_path, {}) or {}
    tasks: list[dict[str, Any]] = []

    for s in tl_shots:
        sid = str(s["id"]).upper()
        if shot_filter and sid not in shot_filter:
            continue
        spec_path = ep_dir / "shots" / f"{sid}.json"
        spec = load_json(spec_path)
        if not isinstance(spec, dict):
            print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"Missing or invalid {spec_path}"}))
            return 2

        pid = str(spec.get("place") or "").strip()
        place_obj = places.get(pid)
        if not place_obj:
            print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"{sid}: unknown place '{pid}' in bible.json"}))
            return 2

        char_ids = [str(c).strip() for c in (spec.get("characters") or []) if str(c).strip()]
        ref_paths: list[Path] = []
        ref_labels: list[str] = []
        ref_media_ids: list[str] = []

        place_ref = refs_dir / f"place_{pid}.png"
        if place_ref.is_file():
            ref_paths.append(place_ref)
            ref_labels.append(f"PLACE:{pid}")
            pm = refs_manifest.get(f"place_{pid}")
            if isinstance(pm, dict) and str(pm.get("mediaName") or "").strip() and (not active_proj or pm.get("projectId") == active_proj):
                ref_media_ids.append(str(pm["mediaName"]).strip())

        char_descs: list[str] = []
        for cid in char_ids:
            cobj = chars.get(cid)
            if not cobj:
                print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"{sid}: unknown character '{cid}' in bible.json"}))
                return 2
            char_descs.append(f"{cid} ({cobj.get('name', cid)}): {cobj.get('appearance', '')}")
            cref = refs_dir / f"char_{cid}.png"
            if cref.is_file():
                ref_paths.append(cref)
                ref_labels.append(f"CHAR:{cid}")
                cm = refs_manifest.get(f"char_{cid}")
                if isinstance(cm, dict) and str(cm.get("mediaName") or "").strip() and (not active_proj or cm.get("projectId") == active_proj):
                    ref_media_ids.append(str(cm["mediaName"]).strip())

        size_lbl = str(spec.get("size") or s.get("size") or "medium")
        view_lbl = str(spec.get("view") or s.get("view") or "front")
        angle_lbl = str(spec.get("angle") or "level")
        stage_card = stage_cards.get(sid)
        cam_lbl = str((stage_card or {}).get("cam") or size_lbl)
        ost = spec.get("onScreenText")

        if char_descs:
            cast_rule = (
                f"CAST COUNT — EXACTLY {len(char_descs)} character(s) visible in frame, matching their attached char_* model sheets (round cartoon heads, bold ink outlines, warm golden rim-light), "
                f"strictly NO extra background people and NO clones: " + " | ".join(char_descs)
            )
        else:
            cast_rule = "CAST COUNT — Strictly NO full characters visible in frame (hands/object focus or empty room only)."

        if isinstance(ost, dict) and str(ost.get("text") or "").strip():
            ost_lang = "English" if str(ost.get("lang") or "").lower() == "en" else ("French" if str(ost.get("lang") or "").lower() == "fr" else lang)
            text_rule = (
                f'IN-WORLD TEXT — Render clean legible {ost_lang} text ONLY inside the {ost.get("kind", "screen")}: '
                f'"{str(ost["text"]).strip()}". Strictly NO extra poster words, NO wall signs, NO episode title card, NO subtitles, NO watermarks.'
            )
        else:
            text_rule = (
                "TEXT RULE — Strictly ZERO text anywhere in the image: NO wall poster words, NO band names, NO background signs, "
                "NO random screen letters, NO episode title cards, NO subtitles, NO logos."
            )

        is_insert_no_cast = size_lbl.lower().strip() in ("insert", "detail", "macro") and not char_ids
        env_line = (
            f"ENVIRONMENT ({pid}) — Softly blurred out-of-focus surface/lighting match for {place_obj.get('name', pid)} ONLY (do NOT pull out to a wide architectural view)."
            if is_insert_no_cast
            else f"ENVIRONMENT ({pid}) — {place_obj.get('name', pid)} (match left/right architecture & consistent lighting from place_{pid} ref): {place_obj.get('description', '')}"
        )

        blocking_line = stage_blocking_line(stage_card)
        prompt_lines = [
            f"STYLE LOCK — {style}. {SIGNATURE_STYLE_LOCK}",
            f"CAMERA & FRAMING (STRICT) — {framing_directive(size_lbl, view_lbl, angle_lbl, cam_lbl, aspect)}",
            cast_rule,
        ]
        if blocking_line:
            prompt_lines.append(blocking_line)
        prompt_lines.extend([
            f"SHOT SCENE & ACTION — {str(spec.get('imagePrompt') or '').strip()}",
            env_line,
            text_rule,
        ])
        full_prompt = "\n".join(prompt_lines)
        shot_model = str(spec.get("imageModel") or "").strip() or model
        shot_neg = ", ".join(x for x in [base_neg, str(spec.get("negativePrompt") or "").strip()] if x)
        ref_hashes = [sha256_file(p) for p in ref_paths]
        sig = sha256_str(json.dumps({
            "driver": driver,
            "model": shot_model,
            "refMode": ref_mode,
            "aspect": aspect,
            "prompt": full_prompt,
            "neg": shot_neg,
            "refs": ref_hashes,
            "seed": spec.get("seed"),
        }, sort_keys=True))

        tasks.append({
            "id": sid,
            "file": board_dir / f"{sid}.png",
            "prompt": full_prompt,
            "negative": shot_neg,
            "model": shot_model,
            "aspect": aspect,
            "ref_paths": ref_paths,
            "ref_labels": ref_labels,
            "ref_media_ids": ref_media_ids if len(ref_media_ids) == len(ref_paths) else None,
            "seed": spec.get("seed"),
            "sig": sig,
        })

    generated: list[str] = []
    cached: list[str] = []
    to_run: list[dict[str, Any]] = []
    for t in tasks:
        prev = manifest.get(t["id"]) if isinstance(manifest.get(t["id"]), dict) else None
        gflow_ok = driver != "gflow" or (bool(prev and prev.get("mediaName")) and (not active_proj or (prev and prev.get("projectId") == active_proj)))
        if not force and t["file"].is_file() and prev and prev.get("hash") == t["sig"] and gflow_ok:
            cached.append(t["id"])
        else:
            to_run.append(t)

    credits_total = 0
    if to_run:
        workers = min(conc, 2) if driver == "gflow" else conc
        mc.emit_progress(stage="stills", done=0, total=len(to_run), detail=f"Generating {len(to_run)} still(s) with {model}")
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_run)))) as pool:
            futs = {
                pool.submit(
                    mc.generate_image,
                    t["prompt"],
                    t["negative"],
                    t["aspect"],
                    t["file"],
                    t["ref_paths"],
                    t["ref_labels"],
                    t["seed"],
                    {**cfg, "model": t["model"]},
                    t["ref_media_ids"],
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
                    **({"mediaName": res.get("mediaName")} if res.get("mediaName") else {}),
                    **({"projectId": res.get("projectId") or active_proj} if (res.get("projectId") or active_proj) else {}),
                }
                generated.append(t["id"])
                mc.append_cost(out_dir, {"step": "stills", "shot": t["id"], **res})
                mc.emit_progress(stage="stills", done=done_n, total=len(to_run), detail=f"{t['id']} ready ({res.get('seconds')}s)")

    save_json(manifest_path, manifest)
    stills_out = [str(t["file"].relative_to(root).as_posix()) for t in tasks]
    print(f"[stills] {len(generated)} generated, {len(cached)} cached ({model} via {driver})")
    print("RESULT_JSON:" + json.dumps({
        "ok": True,
        "mode": "stills",
        "generated": generated,
        "cached": cached,
        "stills": stills_out,
        **({"gflowProjectId": cfg.get("project")} if cfg.get("project") else {}),
        "credits": credits_total,
    }))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("shots", nargs="*", help="Optional shot IDs for --mode stills (e.g. S3 S7)")
    ap.add_argument("--mode", choices=["refs", "stills"], default="stills")
    ap.add_argument("--only", default="", help="Comma-separated IDs for --mode refs")
    ap.add_argument("--project", default=".")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_intermixed_args(argv)

    root = Path(args.project).resolve()
    if args.mode == "refs":
        only_ids = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        return run_refs(root, args.slug, only_ids, args.force, args.concurrency)
    shot_filter = {x.strip().upper() for x in args.shots if x.strip() and not x.startswith("-")}
    return run_stills(root, args.slug, shot_filter, args.force, args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
