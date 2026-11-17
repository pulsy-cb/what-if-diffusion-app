#!/usr/bin/env python3
"""Build the full storyboard contact image (`out/<slug>/storyboard.jpg`) and chunked readable sheets
(`out/<slug>/sheets/sheet-NN.jpg`) from `out/<slug>/board/S*.png`.

    python3 pipeline/board_run.py <slug> [--project DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import board_sheets as bs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--project", default=".")
    args = ap.parse_args(argv)

    root = Path(args.project).resolve()
    slug = args.slug
    ep_dir = root / "src" / "episodes" / slug
    out_dir = root / "out" / slug
    board_dir = out_dir / "board"

    tl_path = ep_dir / "timeline.json"
    tl = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.is_file() else {}
    ep_path = ep_dir / "episode.json"
    ep = json.loads(ep_path.read_text(encoding="utf-8")) if ep_path.is_file() else {}

    expected_ids = [str(s["id"]).upper() for s in (tl.get("shots") or []) if isinstance(s, dict) and s.get("id")]
    stills = sorted(board_dir.glob("S*.png"), key=bs.natural_key) if board_dir.is_dir() else []
    by_id = {p.stem.upper(): p for p in stills}
    missing = [sid for sid in expected_ids if sid not in by_id]
    if missing:
        print(f"Missing board still(s) in {board_dir}: {', '.join(missing)}", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": f"missing stills: {', '.join(missing)}"}))
        return 2
    ordered = [by_id[sid] for sid in expected_ids] if expected_ids else stills
    if not ordered:
        print(f"No stills found in {board_dir}", file=sys.stderr)
        print("RESULT_JSON:" + json.dumps({"ok": False, "error": "no stills found"}))
        return 2

    # 1. Build chunked high-readability contact sheets (`out/<slug>/sheets/sheet-NN.jpg`)
    sheets = bs.build(root, slug, None)

    # 2. Build compact multi-column global `out/<slug>/storyboard.jpg`
    with Image.open(ordered[0]) as first:
        portrait = first.height > first.width
        cell_w = 240 if portrait else 360
        cell_h = int(cell_w * first.height / first.width)

    cols = 4 if portrait else 3
    rows = (len(ordered) + cols - 1) // cols
    cap_h = 56
    gap = 10
    w = cols * cell_w + (cols + 1) * gap
    h = rows * (cell_h + cap_h) + (rows + 1) * gap
    board = Image.new("RGB", (w, h), bs.BG)
    draw = ImageDraw.Draw(board)
    f_id, f_txt = bs.font(20), bs.font(15)
    caps = {str(s.get("id")).upper(): s for s in (ep.get("shots") or []) if isinstance(s, dict)}
    tl_lens = {str(s.get("id")).upper(): float(s.get("len") or 60) / float(tl.get("fps") or 30) for s in (tl.get("shots") or []) if isinstance(s, dict)}

    for idx, p in enumerate(ordered):
        sid = p.stem.upper()
        x = gap + (idx % cols) * (cell_w + gap)
        y = gap + (idx // cols) * (cell_h + cap_h + gap)
        with Image.open(p) as im:
            board.paste(im.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS), (x, y))
        info = caps.get(sid, {})
        dur_s = tl_lens.get(sid)
        size_txt = str(info.get("size", "")).upper()
        header = f"{size_txt} · {dur_s:.1f}s" if dur_s else size_txt
        draw.text((x, y + cell_h + 4), sid, font=f_id, fill=bs.ACCENT)
        draw.text((x + 56, y + cell_h + 7), bs.ellipsize(draw, bs.printable(header), f_txt, cell_w - 56), font=f_txt, fill=bs.INK)
        draw.text((x, y + cell_h + 30), bs.ellipsize(draw, bs.printable(str(info.get("note", ""))), f_txt, cell_w), font=f_txt, fill=(170, 164, 180))

    board_path = out_dir / "storyboard.jpg"
    board.save(board_path, quality=88)

    rel_board = str(board_path.relative_to(root).as_posix())
    rel_sheets = [str(s.relative_to(root).as_posix()) for s in sheets]
    print(f"[board] wrote {rel_board} ({len(ordered)} shots) + {len(rel_sheets)} sheet(s)")
    print("RESULT_JSON:" + json.dumps({"ok": True, "board": rel_board, "sheets": rel_sheets, "shots": len(ordered)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
