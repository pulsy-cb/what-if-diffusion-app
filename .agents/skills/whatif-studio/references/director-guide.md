# What If Studio — 2D AI Diffusion Director & Prompting Guide

Reference manual for authoring `DIRECTOR.md`, `bible.json`, `shots.stage.json`, and `shots/S<N>.json` in the 100% AI Diffusion What If pipeline (Nano Banana 2 Lite + Gemini Omni Flash 2.1).

---

## 1. Non-Negotiable Episode Invariants

1. **100% Diffusion, Zero Code**: Never write `.tsx`, Remotion components, or SVGs. All visuals come from `bible.json` -> `refs/*.png` -> `shots/S<N>.json` -> Nano Banana 2 Lite (`board/S*.png`) -> Gemini Omni Flash 2.1 (`clips/S*.mp4`).
2. **Preserve `:8005` Narration Verbatim**: Keep every word of `source.fr.md` in `episode.json.script`. Never shorten or translate an EN/FR proposal.
3. **Language Purity (`fr` vs `en`)**: Read `episode.json.lang`. Any diegetic in-world text (`onScreenText`) MUST match that exact language (`fr` -> French, `en` -> English).
4. **ZERO Title Cards ("No Title")**: Never author a title shot, "What If..." opening card, chapter banner, lower-third, or burned-in subtitles in any ref or shot prompt. Shot `S1` opens directly inside the scene.
5. **Gemini Omni Flash 2.1 Audio = Physical SFX ONLY**:
   - Allowed: concrete foley and room tone (`"heavy wooden door creak, quiet rain on window glass, smartphone vibration buzz on wood table"`).
   - **STRICTLY FORBIDDEN**: background music, soundtrack, piano/synth score, singing, spoken dialogue, whispering words, or voice-over. Narration comes 100% from `:8005` `voice.wav`, under which `assemble_run.py` automatically ducks the extracted clip SFX.

---

## 2. Canonical JSON Schemas

### `bible.json` (via `whatif_write_bible`)
Use **UPPERCASE** IDs (`[A-Z][A-Z0-9_]*`) for every character and place:
```json
{
  "stylePrompt": "Minimalist 2D What If explainer cartoon animation, oversized smooth round heads (~1:3.5 ratio), bold uniform dark charcoal outlines, simple oval/dot eyes, thick expressive eyebrows, flat cel-shaded outfit blocks, warm golden outer rim-light on characters, atmospheric matte 2D editorial backgrounds with visible practical light pools",
  "negativePrompt": "realistic comic book, Marvel DC, anime, manga, chiseled jawline, 3D CGI, photorealistic, title card, subtitles, watermark, random poster text",
  "characters": [
    {
      "id": "HERO",
      "name": "Maya",
      "role": "protagonist",
      "appearance": "Late 20s woman in minimalist 2D round-head cartoon proportions, oversized smooth oval head, warm olive skin, short dark wavy bob haircut, simple oval dark eyes, wearing an oversized slate-blue denim jacket over a cream knit sweater",
      "refPrompt": "Single full-body standing character model sheet on a flat light neutral cream background, neutral front-3/4 pose, bold dark charcoal outlines"
    }
  ],
  "places": [
    {
      "id": "LIVING_ROOM",
      "name": "Darkened Apartment Living Room",
      "recipe": "living",
      "description": "Fixed layout: worn mustard sofa screen-center, low walnut coffee table front-center, tall rainy window screen-left casting cool blue moonlight, warm amber floor lamp screen-right ALWAYS ON",
      "refPrompt": "Empty architectural 2D matte editorial plate matching the exact left-center-right layout, zero people, soft volumetric lamp cone and grounded shadows"
    }
  ]
}
```

### `shots/S<N>.json` (via `whatif_edit_shot`)
```json
{
  "id": "S1",
  "place": "LIVING_ROOM",
  "characters": ["HERO"],
  "size": "medium",
  "angle": "level",
  "view": "3/4-left",
  "cameraMove": "push",
  "imagePrompt": "Medium 3/4 view at eye level inside the dimly lit living room at night. Maya (slate-blue denim jacket over cream sweater) sits hunched on the edge of the mustard sofa staring down at a glowing smartphone in her hands, illuminated by cold cyan screen light against warm background lamp glow.",
  "negativePrompt": "extra person, duplicate character, title text, subtitles",
  "onScreenText": null,
  "videoPrompt": "Maya slowly lifts the glowing smartphone closer to her face while her eyes widen slightly and her thumb hesitates over the screen, smooth gentle push-in camera move",
  "sfxHint": "soft fabric rustle, quiet rainfall pattering outside window, subtle indoor room tone"
}
```

### Diegetic In-World Text (`onScreenText`)
Use `onScreenText` **ONLY** on shots physically depicting readable writing in the scene:
```json
"onScreenText": {
  "kind": "phone",
  "text": "Tu rentres ce soir ?",
  "lang": "fr"
}
```
- Valid `kind`: `"phone" | "document" | "sign" | "board" | "screen"`.
- Always include the exact quoted `"Tu rentres ce soir ?"` string inside `imagePrompt` so Nano Banana 2 Lite renders legible words in the right place!
- Leave `onScreenText: null` on normal character/action shots.

---

## 3. Prompt Engineering Best Practices

### Nano Banana 2 Lite / 2.1 (`refs` & `stills`)
- `image_gen.py` combines `description` + `refPrompt` (for places) and `appearance` + `refPrompt` (for characters) with the signature What-If minimalist round-head 2D cartoon lock (bold dark charcoal outlines, warm golden outer rim-light on characters, atmospheric visible practical lighting). Never use `refPrompt` without keeping exact wardrobe/room colors in `appearance`/`description`.
- `image_gen.py` attaches `place_<place>.png` and `char_<ID>.png` listed in `characters[]` and tells the model `"EXACTLY N character(s)"`. Never mention plural people (`"two officers"`) when `characters` has only 1 ID (causes face cloning!).
- Enforce Spatial & 180° Axis Continuity: define a fixed Left-Center-Right layout map + invariant practical lights in `DIRECTOR.md`, state exact framing (`size`, `view`, `angle`) + screen position (`screen-left`, `centered`, `screen-right`) + facing direction + Frame-0 pose matching `videoPrompt`, and forbid random wall/poster text unless `onScreenText` is set.
- Keep `9:16` vertical center-safe framing (focal action inside the middle 65% band away from Shorts UI overlays).

### Gemini Omni Flash 2.1 (`clips`)
- `videoPrompt` animates `out/<slug>/board/S<N>.png` directly:
  - Specify **one continuous physical action** + **one clean camera motion** (`still | push | pull | pan | track | whip`).
  - Match complexity to timeline duration (`shotWordFrames`): longer shots (`> 3s`) need sustained continuous motion (e.g., `"slowly walks across the dim hall and reaches for the brass handle"`), never a quick snap that stalls halfway.
  - Avoid impossible 180° turns or sudden scene cuts inside a single shot.
- Keep `sfxHint` strictly concrete, physical, and dry — zero music, zero vocals.
