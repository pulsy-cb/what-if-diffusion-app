---
name: whatif-studio
description: >
  End-to-end 100% AI Diffusion 2D animated video production (9:16 Shorts or 16:9 episodes, in French or English) using Nano Banana 2 Lite (refs + storyboard stills), Gemini Omni Flash 2.1 (Image-to-Video + physical foley SFX), :8005 TTS/ASR, FFmpeg/-14 LUFS mastering, and a mandatory Independent Reviewer subagent via What If Studio MCP tools. Use this skill whenever the user asks to create, generate, edit, review, or render a What If diffusion video or :8005 video ID episode directly from Antigravity CLI.
---

# What If Studio — 100% AI Diffusion Production Skill

Produce complete, cohesive **2D animated AI diffusion videos** (`1080x1920` 9:16 Shorts or `1920x1080` 16:9 @ 30fps) directly from Antigravity CLI using the **`whatif-studio` MCP server**.

This pipeline runs **100% on diffusion + FFmpeg** (Nano Banana 2 Lite / `nano2.1` for neutral references and per-shot stills; `omni-1.2-flash` at `360p` default via `gflow :8006` with a dedicated Flow project per episode for Image-to-Video clips + physical foley SFX) paired with `:8005` narration & Deepgram word timings. Zero React/SVG code generation, zero Remotion/Modal, **zero title cards**, and **zero AI music or AI dialogue**.

---

## Core Production Workflow (Mandatory Sequence)

Read `references/director-guide.md` first for detailed 2D visual style, prompting rules, and anti-monotony patterns.

### 1. Project & Audio Foundation (`:8005` Integration)
1. **Create / reuse project**:
   ```json
   whatif_create_project({ "episode": "<slug>", "aspect": "9:16" | "16:9", "video": "<optional_8005_video_id>" })
   ```
2. **Import `:8005` script** *(if a `:8005` video ID was provided)*:
   ```json
   whatif_run({ "episode": "<slug>", "step": "source", "args": ["--pick", "hook", "--apply"] })
   ```
   - Supports French (`fr`) and English (`en`) seamlessly. **NEVER translate, shorten, or truncate** `episode.json.script`.
3. **Generate / Verify Voice & Deepgram Word Timings (`:8005`)**:
   ```json
   whatif_run({ "episode": "<slug>", "step": "voice" })
   ```
   - Dynamically matches narrator voice (`Léa` or `Hugo`, ElevenLabs `eleven_v4`) to episode language & POV, outputs `public/<slug>/voice.wav` and `src/episodes/<slug>/words.json`.
4. **Shotlist & 30fps Timeline**:
   ```json
   whatif_run({ "episode": "<slug>", "step": "shotlist" })
   whatif_run({ "episode": "<slug>", "step": "plan" })
   ```
   - `shotlist.py` cuts beats without any title card (`no title shots`) and locks `sound: { provider: "omni-flash", music: false, dialogue: false }`.
   - `plan_run.py` produces `src/episodes/<slug>/timeline.json`.

### 2. Visual Bible & Neutral Reference Plates (`refs`)
5. **Read Pin & Prompt Guide**:
   ```json
   whatif_episode_pin({ "episode": "<slug>", "includePromptGuide": true })
   whatif_shot_words({ "episode": "<slug>" })
   ```
6. **Write `DIRECTOR.md`** via `whatif_write_file`: lock visual style, hero/cast outfits & traits, locations, color palette moods (`warm`, `memory`, `tension`), language lock (`fr`/`en`), and strict prohibitions (no titles, no music, no Omni dialogue).
7. **Write & Validate `bible.json`** via **`whatif_write_bible`**:
   - Define `stylePrompt`, `negativePrompt`, `characters[]` (`id` UPPERCASE like `HERO`, `name`, `role`, `appearance`, `refPrompt?`), and `places[]` (`id` UPPERCASE like `LIVING_ROOM`, `name`, `recipe`, `description`, `refPrompt?`).
8. **Generate Reference Sheets**:
   ```json
   whatif_run({ "episode": "<slug>", "step": "refs" })
   ```
   - Produces **empty** places (`out/<slug>/refs/place_<ID>.png`) and **neutral plain-background** characters (`out/<slug>/refs/char_<ID>.png`). Visually inspect key refs with `read_file` before writing shots.

### 3. Staging & Diffusion Shot Specs (`shots/S<N>.json`)
9. **Save Staging Cards FIRST**: call **`whatif_stage_shots`** for every shot ID (`place`, `people`, `see`, `subject`, `angle`, `view`, `moves`, `change`) so visual rhythm & props are audited before prompt authoring.
10. **Author Per-Shot JSON Specs** via **`whatif_edit_shot`** (supports single `{ episode, id, ... }` or batch `{ episode, shots: [...] }`):
    - Canonical fields: `id`, `place`, `characters[]`, `size`, `angle`, `view`, `cameraMove`, `imagePrompt`, `imageModel?` (optional `"nano2.1"` override per shot), `negativePrompt?`, `onScreenText?` (`null` on normal shots; `{ kind, text, lang }` strictly in episode language for diegetic phone/document/sign text only), `videoPrompt` (ONE continuous physical action + camera move), `sfxHint` (physical foley/ambience ONLY).
    - **Clone & Identity Guard**: `characters[]` must list ONLY the characters visible in that shot — `image_gen.py` attaches ONLY the matching `place` + `characters` refs!

### 4. Storyboard Stills, Contact Sheets & **MANDATORY Independent Reviewer Subagent**
11. **Generate Stills & Build Contact Sheets**:
    ```json
    whatif_run({ "episode": "<slug>", "step": "stills" })
    whatif_run({ "episode": "<slug>", "step": "board" })
    ```
12. **Mandatory Subagent Reviewer Gate**:
    `whatif_run` **BLOCKS `clips` and `render`** until the Independent Reviewer subagent audits the current board:
    - Call `define_subagent` + `invoke_subagent` with a read-only Reviewer role.
    - Inside the Reviewer subagent:
      1. Call `whatif_review_pack({ "episode": "<slug>" })` (returns Round 1 Global `refs` + `sheets`, or Round 2+ Targeted Delta stills + full JSON specs).
      2. Visually inspect EVERY path in `hostImagePaths` with `read_file`! Check character identity (no clones/outfit swaps), 2D style consistency, ZERO title cards/subtitles, diegetic `onScreenText` language purity (`fr`/`en`), staging prop accuracy, and Omni `videoPrompt` / `sfxHint` compliance (NO music, NO speech).
      3. Save verdict via `whatif_save_review({ "episode": "<slug>", "pass", "summary", "issues", "round", "mode" })`.
    - If `blocking: true`: fix ONLY flagged shots via `whatif_edit_shot`, re-run `whatif_run({ "episode": "<slug>", "step": "stills", "args": ["S..."] })` + `board`, and launch a Round 2 Delta subagent review.

### 5. Image-to-Video Clips, SFX Extraction & `-14 LUFS` Master
13. **Generate I2V Clips (Gemini Omni Flash 2.1)**:
    ```json
    whatif_run({ "episode": "<slug>", "step": "clips" })
    ```
14. **Assemble & Master Final MP4 (with optional Anime 2D Upscaling)**:
    ```json
    whatif_run({ "episode": "<slug>", "step": "render" })
    ```
    - `assemble_run.py` aligns every clip onto exact `timeline.json` frames @ 30fps, extracts Omni Flash SFX, crossfades (`35 ms`) & ducks SFX under `:8005` `voice.wav` into `public/<slug>/sfx.wav`, muxes `out/<slug>/<slug>-raw.mp4`, and `master_run.py` masters `out/<slug>/<slug>-final.mp4` at **-14 LUFS**. When the Anime 2D Upscaler is enabled (or if running `whatif_run({ "episode": "<slug>", "step": "upscale" })` manually after a `360p`/`720p` render), `upscale_run.py` elevates `-final.mp4` to 1080p / 2K / 4K UHD while keeping the exact `-14 LUFS` audio untouched.
15. Confirm with `whatif_status({ "episode": "<slug>" })` and report only observed file paths, durations, resolution, and LUFS metrics.
