# What If Diffusion Studio

Application autonome de production vidéo **100 % animation 2D par diffusion IA**, pilotée par un agent orchestrateur, un **Reviewer indépendant**, et l’intégration native du serveur `:8005` (« nouveau »).

- **Images & feuilles de référence** : **Nano Banana 2 Lite** (`nano2-lite`, avec fallback / option **`nano2.1`** par plan via `imageModel`) via l’API Google Flow locale `http://127.0.0.1:8006/` (avec création automatique d’un **projet Flow dédié par épisode**), ou Gemini natif / OpenAI / REST / `mock`.
- **Vidéo Image-to-Video & bruitages physiques (SFX)** : **Omni 1.2 Flash** (`omni-1.2-flash` en **`360p`** par défaut sur `:8006`, ou Omni 2.1 / Veo) — génère le mouvement de chaque plan à partir du still validé + des **bruitages et ambiances physiques uniquement** (strictement **aucune musique**, **aucun dialogue/voix**).
- **Voix off & sous-titrage temporel (`:8005`)** : récupération verbatim des scripts FR/EN (`source_import.py`), synthèse vocale ElevenLabs v4 Hugo/Léa + alignement mot par mot Deepgram (`voice_fish.py` -> `words.json`).
- **Zéro carte-titre (« no title cards »)** : l’épisode démarre directement dans l’histoire dès `S1`. Seul du texte diégétique intégré à l’univers (écran de téléphone, document, enseigne) est autorisé, strictement dans la langue de l’épisode (`fr` ou `en`).
- **Montage & mastering local FFmpeg** : assemblage calé à la frame près sur `timeline.json` (30 fps), extraction des SFX des clips avec fondu croisé `35 ms` et ducking automatique sous `voice.wav`, puis normalisation à **-14 LUFS** (`master_run.py`).

Aucune dépendance externe à `WHATIF_SKILL_DIR`, Remotion, Chromium ou Modal n’est nécessaire.

---

## Architecture du pipeline

```text
whatif_new ─▶ source (:8005) ─▶ voice (:8005 + Deepgram) ─▶ shotlist ─▶ plan (timeline.json)
   │
   ▼
DIRECTOR.md + write_bible (bible.json) ─▶ refs (Nano Banana 2 Lite : lieux vides + persos neutres)
   │
   ▼
stage_shots ─▶ edit_shot (shots/S*.json) ─▶ stills (Nano Banana 2 Lite) ─▶ board (storyboard.jpg + sheets)
   │
   ▼
Reviewer Indépendant (Global 100% puis Delta haute résolution sur refs + planches + specs JSON)
   │
   ▼
clips (Gemini Omni Flash 2.1 I2V + SFX physiques seuls) ─▶ render (FFmpeg assemble_run + master -14 LUFS)
```

---

## Configuration des API (Multi-Provider)

Deux couches indépendantes peuvent être configurées via `.env`, `providers.json` (avec résolution `$ENV_VAR`), ou directement dans l’interface web (**bouton `⚙`**) :

1. **LLM Agent & Reviewer** (`WHATIF_PROVIDER`, `WHATIF_MODEL`, `WHATIF_API_KEY`, `WHATIF_REVIEW_*` — Anthropic, OpenAI/compatibles, Codex OAuth, Antigravity `agy` CLI).
2. **Médias Diffusion Image & Vidéo** (`mediaProviders` dans `providers.json`, `.env`, ou UI `⚙`) :

| Rôle | Variable `.env` par défaut | Drivers supportés | Défaut |
| --- | --- | --- | --- |
| **Image** | `WHATIF_IMAGE_PROVIDER`, `WHATIF_IMAGE_MODEL`, `WHATIF_IMAGE_FALLBACK_MODEL`, `WHATIF_IMAGE_BASE_URL`, `WHATIF_IMAGE_API_KEY`, `WHATIF_IMAGE_REF_MODE` | `gflow` (`:8006`, projet Flow dédié par épisode), `gemini`, `openai`, `http` (`responsePaths` + `pollPath`), `mock` | `gflow` / `nano2-lite` -> `nano2.1` (`refMode=multi`) |
| **Vidéo** | `WHATIF_VIDEO_PROVIDER`, `WHATIF_VIDEO_MODEL`, `WHATIF_VIDEO_RESOLUTION` (`360p`/`720p`/`1080p`), `WHATIF_VIDEO_BASE_URL`, `WHATIF_VIDEO_API_KEY`, `WHATIF_VIDEO_POLL_PATH` | `gflow` (`:8006`), `gemini`, `openai`, `http` (`responsePaths` + `pollPath`), `mock` | `gflow` / `omni-1.2-flash` (`360p`) |
| **Upscaler Anime 2D (optionnel)** | `WHATIF_UPSCALER_ENABLED`, `WHATIF_UPSCALER_TARGET` (`auto-1080p`/`auto-2k`/`auto-uhd`/`2x`/`3x`/`4x`), `WHATIF_UPSCALER_MODEL`, `WHATIF_UPSCALER_DENOISE`, `WHATIF_UPSCALER_GPU`, `WHATIF_UPSCALER_BASE_URL` | `modal-anime` (Modal Real-CUGAN / Real-ESRGAN), `mock` | désactivé (`real-cugan`, `L40S`) |

Exemples dans [`providers.example.json`](providers.example.json) et [`.env.example`](.env.example). Les clés secrètes (`.env` et `providers.json`) ne sont jamais exposées à l’agent ni renvoyées en clair par `/api/settings`.

---

## Démarrage rapide

### Local (Node.js 22+ · Python 3.11+ · FFmpeg)

```bash
npm install
npm run build
cp .env.example .env
# Renseigner vos clés LLM et diffusion dans .env ou providers.json (ou utiliser l'UI ⚙)

# Vérification pré-vol complète
node pipeline/preflight.mjs

# Lancer le studio web (http://127.0.0.1:3020)
npm start
```

### Tests automatisés

```bash
npm run build && npm test
```

### Docker Compose

```bash
docker compose up -d --build
docker compose exec studio node pipeline/preflight.mjs
```

---

## Intégration MCP (`whatif-studio`)

Le serveur MCP ([`src/mcp/stdio.ts`](src/mcp/stdio.ts)) expose directement aux sessions Antigravity (`agy`) tous les outils du studio :
- `whatif_create_project`, `whatif_run` (`source`, `voice`, `shotlist`, `plan`, `refs`, `stills`, `board`, `clips`, `render`, `master`, `upscale`),
- `whatif_write_bible`, `whatif_stage_shots`, `whatif_edit_shot`, `whatif_episode_pin`, `whatif_shot_words`,
- `whatif_review_pack` + `whatif_save_review` (porte obligatoire avant `clips` et `render`).
