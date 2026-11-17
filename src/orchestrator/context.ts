import fs from 'node:fs';
import path from 'node:path';
import type { ChatMessage, ImagePart, ToolCall } from './providers/types.js';
import type { MessageRow } from './db.js';
import { readText, walk } from '../tools/paths.js';
import { readEpisodeBible, readEpisodeShots, sortShotIds } from '../tools/episode-source.js';
import { DIRECTOR_GUIDE, PROMPTING_GUIDE } from './director-guide.js';

/** Count spoken words in an episode script, stripping [audio tags] and <xml> markers consistently everywhere. */
export function countScriptWords(script: unknown): number {
  if (!Array.isArray(script)) return 0;
  return script.reduce(
    (n: number, b: any) =>
      n +
      String(b?.text ?? '')
        .replace(/\[[^\]]*\]/g, ' ')
        .replace(/<[^>]*>/g, ' ')
        .split(/\s+/)
        .filter(Boolean).length,
    0,
  );
}

const FR_EP_LANG_RE = /\b(je|tu|il|elle|nous|vous|le|la|les|un|une|des|du|dans|sur|pour|avec|est|sont|mais|plus|pas|que|qui|était|avait|suis|ma|mes|sa|ses)\b/gi;
const EN_EP_LANG_RE = /\b(the|and|that|with|for|was|were|have|had|not|but|they|them|their|she|her|his|him|you|your|from|into|about|when|what|out|my|would|could)\b/gi;

/** Detect whether an episode is narrated in English ('en') or French ('fr', default). */
export function detectEpisodeLanguage(episode: any): 'fr' | 'en' {
  const explicit = String(episode?.lang ?? episode?.source?.lang ?? episode?.voice?.lang ?? '').trim().toLowerCase();
  if (explicit === 'en') return 'en';
  const script = Array.isArray(episode?.script) ? episode.script : [];
  const fullText = script.map((b: any) => String(b?.text ?? '')).join(' ');
  const frHits = (fullText.match(FR_EP_LANG_RE) ?? []).length;
  const enHits = (fullText.match(EN_EP_LANG_RE) ?? []).length;
  if (enHits >= 4 && enHits > frHits * 1.2) return 'en';
  if (explicit === 'fr') return 'fr';
  return 'fr';
}

export const SYSTEM_PROMPT = `You are the studio agent of "What If Diffusion", a 100% AI-diffusion 2D animated story pipeline (from 15 s 9:16 Shorts up to multi-minute 16:9 episodes @ 30fps, using Nano Banana 2 Lite / 2.1 for reference sheets & storyboard stills, Omni 1.2 Flash 360p for Image-to-Video + physical foley SFX via gflow :8006 with a dedicated Flow project per episode, and FFmpeg assembly mastered to -14 LUFS).

You work inside ONE project folder (the studio) and you change it with tools. You never print shell commands for a human to paste: you call the tools.

## The pipeline (this order is the product)
0. Source (\`:8005\` "nouveau" server):
   \`whatif_new { episode, video, aspect }\` scaffolds the project, and \`whatif_run { episode, step: "source", args: ["--pick", "hook", "--apply"] }\` imports the server's narration proposal — **whether in French (\`fr\`) OR in English (\`en\`)** — into \`out/<episode>/source.fr.md\`, sets \`episode.json.lang\`, and populates \`episode.json.script\` verbatim.
   - **CRITICAL RULE FOR \`:8005\` VIDEOS (FR OR EN)**: **NEVER block, stop, translate, rewrite, or shorten the server's narration proposal!** Keep 100% of the narration text intact and preserve \`aspect\` (\`"9:16"\` or \`"16:9"\`).

1. Standalone episodes (ONLY when there is NO \`:8005\` source video id):
   Write \`episode.json\` in the requested language (\`"fr"\` or \`"en"\`) with cold-open beats, rising tension, and closing question. Never include a title card beat.

2. \`whatif_run { episode, step: "voice" }\` — ElevenLabs v4 TTS (**Léa** or **Hugo**, selected dynamically from script language & POV/gender cues) + Deepgram ASR -> \`public/<episode>/voice.wav\` and \`src/episodes/<episode>/words.json\`.

3. \`whatif_run { episode, step: "shotlist" }\` — cuts the measured script words into brisk beat-aligned shots (fast cold-open hook, semantic \`insert\`, \`wide\`, \`medium\`, \`close\`, \`ots\`, \`low\`, color moods, and NO title shots). Sound is fixed to \`{ provider: "omni-flash", music: false, dialogue: false }\`.

4. \`whatif_run { episode, step: "plan" }\` — resolves cues into \`src/episodes/<episode>/timeline.json\` (30fps frames; cues use \`f:\`, \`s:\`, \`S<N>\`, \`end\`, or bare word anchors — never a \`"w:"\` prefix).

5. **Write \`src/episodes/<slug>/DIRECTOR.md\` (>= 200 chars) and \`src/episodes/<slug>/bible.json\` via \`write_bible\`**:
   - \`DIRECTOR.md\` locks the signature **minimalist round-head What-If 2D cartoon style** (oversized round heads ~1:3.5, bold charcoal outlines, simple dot/oval eyes, warm golden outer rim-light on characters, atmospheric matte 2D sets with visible practical lights & grounded oval shadows — strictly NO realistic comic-book or anime anatomy), exact character wardrobes, fixed **Left-Center-Right Spatial Map** + invariant lighting per room, and framing rules.
   - \`write_bible\` writes \`bible.json\` with \`stylePrompt\`, \`negativePrompt\`, \`characters[]\`, and \`places[]\`.
   - Then run \`whatif_run { episode, step: "refs" }\` to generate empty place reference plates (\`out/<slug>/refs/place_<ID>.png\`) and neutral-background character model sheets (\`out/<slug>/refs/char_<ID>.png\`).

6. **Stage shots with \`stage_shots\`, then author JSON shot specs via \`edit_shot\`**:
   - Author one staging card per shot ID via \`stage_shots\`.
   - Author every shot spec \`src/episodes/<slug>/shots/S<N>.json\` using \`edit_shot\` (batching 4–8 shots per call for long episodes). Optionally set \`imageModel: "nano2.1"\` on hero/multi-char shots while keeping default \`nano2-lite\`. Never write \`.tsx\` or SVG code.

7. **Generate Stills & Contact Board -> Mandatory Reviewer**:
   - \`whatif_run { episode, step: "stills" }\` calls Nano Banana 2 Lite (automatically falling back or upgrading to \`nano2.1\` when needed, inside the episode's own dedicated Flow project on \`:8006\`) with ONLY the shot's active place + character ref sheets (plus \`S*.png\` SHA-256 caching and targeted re-runs via \`args: ["S3", "S7"]\`).
   - \`whatif_run { episode, step: "board" }\` builds \`out/<slug>/storyboard.jpg\` + \`out/<slug>/sheets/sheet-*.jpg\` — the automatic Reviewer inspects the sheets + refs + prompts. Fix any refused shots with \`edit_shot\`, re-run \`stills\` (targeted \`args\`) and \`board\`.

8. **Image-to-Video + Physical Foley SFX, Mastering & Optional 2D Upscaling**:
   - \`whatif_run { episode, step: "clips" }\` animates every approved still \`S<N>.png\` with Omni 1.2 Flash (\`360p\` default on \`:8006\`) into \`out/<slug>/clips/S<N>.mp4\` (\`durationSec >= 2.0\`, coherent motion + physical foley audio).
   - \`whatif_run { episode, step: "render" }\` trims/retimes clips onto exact \`timeline.json\` frame boundaries, extracts Omni Flash SFX, crossfades & ducks SFX under \`:8005\` \`voice.wav\` into \`public/<slug>/sfx.wav\`, writes \`out/<slug>/<slug>-raw.mp4\`, and masters \`out/<slug>/<slug>-final.mp4\` at **-14 LUFS** (also auto-running optional Modal Anime 2D upscaling to 1080p/2K/4K UHD when enabled in settings, e.g. from \`360p\`/\`720p\` renders). (\`step: "master"\` re-runs only final mastering; \`step: "upscale"\` runs or re-runs Modal 2D upscaling on top of \`-final.mp4\` keeping -14 LUFS audio untouched.)

## Absolute Content & Style Rules
- **100% Diffusion, ZERO Code**: never attempt to write TypeScript, React, SVG, or Remotion files.
- **ZERO Title Cards**: never put an episode title screen, intro card, chapter banner, or burned-in subtitles in any shot prompt or ref.
- **In-World Diegetic Text (\`onScreenText\`) ONLY when physically part of the scene** (phone SMS, paper document, clinic sign): strictly in \`episode.json.lang\` (\`"fr"\` or \`"en"\`), and quote the exact string in \`imagePrompt\`.
- **Omni Flash 2.1 Audio = Physical SFX ONLY**: \`sfxHint\` and \`videoPrompt\` may describe physical foley & ambient room noise, **NEVER music, soundtrack, singing, or spoken words/dialogue** (voice comes 100% from \`:8005\` \`voice.wav\`).
- Keep going without asking the user between steps until the final \`-final.mp4\` succeeds. Always cite only observed tool outputs, paths, and numbers.`;

function memory(projectRoot: string): string {
  const projectMd = readText(projectRoot, 'PROJECT.md', 40_000);
  const episodesDir = path.join(projectRoot, 'src', 'episodes');
  const episodes: string[] = [];
  if (fs.existsSync(episodesDir)) {
    for (const d of fs.readdirSync(episodesDir, { withFileTypes: true })) {
      if (!d.isDirectory()) continue;
      const ep = path.join(episodesDir, d.name, 'episode.json');
      if (!fs.existsSync(ep)) continue;
      try {
        const json = JSON.parse(fs.readFileSync(ep, 'utf8'));
        episodes.push(`- ${d.name}: id=${json.id} title=${JSON.stringify(json.title)}`);
      } catch {
        episodes.push(`- ${d.name}: (episode.json unreadable)`);
      }
    }
  }
  return `${episodes.length ? `## Episodes\n${episodes.join('\n')}\n` : '## Episodes\n(none yet)\n'}${
    projectMd ? `\n## PROJECT.md\n${projectMd.slice(0, 6000)}` : '\n(no PROJECT.md yet)'
  }`;
}

function fileList(projectRoot: string, cap = 260): string {
  const all = walk(projectRoot, { maxFiles: 2000 });
  const priority = all.filter((f) => f.startsWith('src/episodes/') || !f.includes('/'));
  const rest = all.filter(
    (f) => !f.startsWith('src/episodes/') && f.includes('/') && !f.startsWith('library/') && !f.startsWith('public/'),
  );
  return [...priority, ...rest].slice(0, cap).join('\n');
}

const TOOL_BUDGET = 2_500;

type ImageBudget = { left: number; seen: Set<string> };

function imagesFromPayload(payload: any, projectRoot: string, budget: ImageBudget): { images: ImagePart[]; skippedDuplicate: number } {
  const paths: string[] = Array.isArray(payload?.images) ? payload.images : [];
  const out: ImagePart[] = [];
  let skippedDuplicate = 0;
  for (const rel of paths) {
    if (!/\.(png|jpe?g|webp)$/i.test(rel)) continue;
    const norm = path.posix.normalize(rel.replace(/\\/g, '/'));
    if (budget.seen.has(norm)) {
      skippedDuplicate++;
      continue;
    }
    if (out.length >= 3 || budget.left <= 0) continue;
    try {
      const abs = path.join(projectRoot, rel);
      const st = fs.statSync(abs);
      if (st.size > budget.left || st.size > 4_000_000) continue;
      budget.left -= st.size;
      budget.seen.add(norm);
      out.push({
        mime: rel.toLowerCase().endsWith('.png') ? 'image/png' : rel.toLowerCase().endsWith('.webp') ? 'image/webp' : 'image/jpeg',
        data: fs.readFileSync(abs).toString('base64'),
      });
    } catch {
      /* missing file: skip */
    }
  }
  return { images: out, skippedDuplicate };
}

export type EpisodeStatus = {
  slug: string;
  words: number;
  seconds: string;
  done: string[];
  todo: string[];
  title: string;
  mtimeMs: number;
};

/** Latest mtime across files matching `extRe` inside `dir`, or 0 when missing. */
function latestDirMtime(dir: string, extRe?: RegExp): number {
  try {
    let latest = 0;
    for (const f of fs.readdirSync(dir)) {
      if (extRe && !extRe.test(f)) continue;
      try {
        const mt = fs.statSync(path.join(dir, f)).mtimeMs;
        if (mt > latest) latest = mt;
      } catch {
        /* ignore */
      }
    }
    return latest;
  } catch {
    return 0;
  }
}

/** Where the episode actually stands, recomputed every turn: cheaper than making the model re-read. */
export function episodeStatus(projectRoot: string): EpisodeStatus[] {
  const eps = path.join(projectRoot, 'src', 'episodes');
  let slugs: string[] = [];
  try {
    slugs = fs
      .readdirSync(eps, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);
  } catch {
    return [];
  }
  const out: EpisodeStatus[] = [];
  for (const slug of slugs) {
    const src = path.join(eps, slug);
    const epFile = path.join(src, 'episode.json');
    let ep: any = {};
    let mtimeMs = 0;
    try {
      ep = JSON.parse(fs.readFileSync(epFile, 'utf8'));
      mtimeMs = fs.statSync(epFile).mtimeMs;
    } catch {
      continue;
    }
    const dir = path.join(projectRoot, 'out', slug);
    const hasOut = (rel: string) => fs.existsSync(path.join(dir, rel));
    const mtime = (file: string): number => {
      try {
        return fs.statSync(file).mtimeMs;
      } catch {
        return 0;
      }
    };

    const bibleRes = readEpisodeBible(src);
    const epSrc = readEpisodeShots(src);
    let timelineIds: string[] = [];
    try {
      const tl = JSON.parse(fs.readFileSync(path.join(src, 'timeline.json'), 'utf8'));
      if (Array.isArray(tl?.shots)) timelineIds = tl.shots.map((s: any) => String(s?.id ?? '')).filter(Boolean);
    } catch {
      /* no timeline yet */
    }
    if (!timelineIds.length && Array.isArray(ep?.shots)) {
      timelineIds = ep.shots.map((s: any) => String(s?.id ?? '')).filter(Boolean);
    }

    const bibleFresh =
      bibleRes.bible !== null &&
      bibleRes.errors.length === 0 &&
      bibleRes.bible.characters.length > 0 &&
      bibleRes.bible.places.length > 0;

    const refsDir = path.join(dir, 'refs');
    const refsReady = (() => {
      if (!bibleFresh || !bibleRes.bible || !fs.existsSync(refsDir)) return false;
      for (const p of bibleRes.bible.places) {
        if (!fs.existsSync(path.join(refsDir, `place_${p.id}.png`))) return false;
      }
      for (const c of bibleRes.bible.characters) {
        if (!fs.existsSync(path.join(refsDir, `char_${c.id}.png`))) return false;
      }
      const refsMtime = latestDirMtime(refsDir, /\.png$/i);
      return !epSrc.bibleMtimeMs || !refsMtime || epSrc.bibleMtimeMs <= refsMtime + 1_000;
    })();

    const stageMap = new Map<string, any>();
    for (const s of Array.isArray(ep?.shots) ? ep.shots : []) {
      if (s?.id && s?.stage && typeof s.stage === 'object') {
        stageMap.set(String(s.id).toUpperCase(), s.stage);
      }
    }
    try {
      const stageData = JSON.parse(fs.readFileSync(path.join(src, 'shots.stage.json'), 'utf8'));
      const cardsObj = stageData?.cards && typeof stageData.cards === 'object' ? stageData.cards : stageData;
      if (cardsObj && typeof cardsObj === 'object') {
        for (const [k, v] of Object.entries(cardsObj)) {
          if (/^S\d+$/i.test(k) && v && typeof v === 'object' && !stageMap.has(k.toUpperCase())) {
            stageMap.set(k.toUpperCase(), v);
          }
        }
      }
    } catch {
      /* not staged yet */
    }
    const nonStubIds = timelineIds.filter((id) => id !== 'S1' || timelineIds.length > 1 || !epSrc.isPlaceholder);
    const stagedFresh =
      nonStubIds.length > 0
        ? nonStubIds.every((id) => stageMap.has(id.toUpperCase()))
        : stageMap.size > 0;

    const exportedSet = new Set(epSrc.exportedShotIds);
    const promptsFresh =
      !epSrc.isPlaceholder &&
      epSrc.exportedShotIds.length > 0 &&
      (nonStubIds.length === 0 || nonStubIds.every((id) => exportedSet.has(id)));

    const shotsMtime = epSrc.latestMtimeMs;
    const boardDirMtime = latestDirMtime(path.join(dir, 'board'), /\.png$/i);
    const boardMtime = mtime(path.join(dir, 'storyboard.jpg'));
    const clipsDir = path.join(dir, 'clips');
    const clipsMtime = latestDirMtime(clipsDir, /\.mp4$/i);
    const rawMtime = mtime(path.join(dir, `${slug}-raw.mp4`));
    const finalMtime = mtime(path.join(dir, `${slug}-final.mp4`));

    const stillsFresh =
      hasOut('board') &&
      boardDirMtime > 0 &&
      (!shotsMtime || shotsMtime <= boardDirMtime + 1_000) &&
      (nonStubIds.length === 0 || nonStubIds.every((id) => fs.existsSync(path.join(dir, 'board', `${id}.png`))));

    const boardFresh =
      hasOut('storyboard.jpg') &&
      (!shotsMtime || shotsMtime <= boardMtime + 1_000) &&
      (!boardDirMtime || boardDirMtime <= boardMtime + 1_000);

    const clipsFresh =
      fs.existsSync(clipsDir) &&
      clipsMtime > 0 &&
      (!boardDirMtime || boardDirMtime <= clipsMtime + 1_000) &&
      (!shotsMtime || shotsMtime <= clipsMtime + 1_000) &&
      (nonStubIds.length === 0 || nonStubIds.every((id) => fs.existsSync(path.join(clipsDir, `${id}.mp4`))));

    const renderFresh =
      hasOut(`${slug}-raw.mp4`) &&
      boardFresh &&
      (!boardMtime || boardMtime <= rawMtime + 1_000) &&
      (!clipsMtime || clipsMtime <= rawMtime + 1_000);

    const masterFresh = hasOut(`${slug}-final.mp4`) && renderFresh && (!rawMtime || rawMtime <= finalMtime + 1_000);

    const words = countScriptWords(ep.script);
    const steps: [string, boolean][] = [
      ['script', Array.isArray(ep.script) && ep.script.length > 0 && !JSON.stringify(ep.script).includes('TODO')],
      ['words', fs.existsSync(path.join(src, 'words.json'))],
      ['voice', fs.existsSync(path.join(projectRoot, 'public', slug, 'voice.wav'))],
      ['timeline', fs.existsSync(path.join(src, 'timeline.json')) && Number(ep?.shots?.length ?? 0) > 1],
      ['director', fs.existsSync(path.join(src, 'DIRECTOR.md'))],
      ['bible', bibleFresh],
      ['refs', refsReady],
      ['staged', stagedFresh],
      ['prompts', promptsFresh],
      ['stills', stillsFresh],
      ['board', boardFresh],
      ['clips', clipsFresh],
      ['sfx', fs.existsSync(path.join(projectRoot, 'public', slug, 'sfx.wav'))],
      ['render', renderFresh],
      ['master', masterFresh],
    ];
    let seconds = '';
    try {
      const w = JSON.parse(fs.readFileSync(path.join(src, 'words.json'), 'utf8'));
      const last = Array.isArray(w) ? w[w.length - 1] : null;
      if (last?.e) seconds = `, voice ${Number(last.e).toFixed(1)}s`;
    } catch {
      /* no take yet */
    }
    out.push({
      slug,
      words,
      seconds,
      done: steps.filter(([, v]) => v).map(([k]) => k),
      todo: steps.filter(([, v]) => !v).map(([k]) => k),
      title: Array.isArray(ep.title) ? ep.title.join(' ') : String(ep.title ?? ''),
      mtimeMs,
    });
  }
  return out.sort((a, b) => b.mtimeMs - a.mtimeMs);
}

function projectState(projectRoot: string): string {
  const status = episodeStatus(projectRoot);
  if (!status.length) return 'no episode yet';
  return status
    .map(
      (e) =>
        `- ${e.slug}: ${e.words} words${e.seconds} | done: ${e.done.join(' ') || '-'} | todo: ${e.todo.join(' ') || '-'}` +
        (e.title ? ` | metadata title: ${e.title}` : ''),
    )
    .join('\n');
}

/** Resolve which episode slug this conversation is working on, or fall back to the most recently modified episode. */
export function activeEpisodeSlug(projectRoot: string, history?: MessageRow[]): string | null {
  const status = episodeStatus(projectRoot);
  if (!status.length) return null;
  const known = new Set(status.map((s) => s.slug));
  if (history?.length) {
    for (let i = history.length - 1; i >= 0; i--) {
      const row = history[i]!;
      if (row.payload) {
        try {
          const p = JSON.parse(row.payload);
          for (const tc of p?.toolCalls ?? []) {
            const ep = tc?.args?.episode;
            if (typeof ep === 'string' && known.has(ep)) return ep;
            const pth = String(tc?.args?.path ?? '');
            const m = /^src\/episodes\/([^/]+)\//.exec(pth);
            if (m && known.has(m[1]!)) return m[1]!;
          }
        } catch {
          /* ignore */
        }
      }
      for (const slug of known) {
        if (new RegExp(`\\b${slug.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(row.content)) {
          return slug;
        }
      }
    }
  }
  const inProgress = status.find((e) => e.done.some((d) => d !== 'script') && e.todo.length > 0);
  return (inProgress ?? status[0])!.slug;
}

/** The steps the active episode still owes, or [] when mastered/idle. */
export function pipelineTodo(projectRoot: string, history?: MessageRow[]): string[] {
  const status = episodeStatus(projectRoot);
  if (!status.length) return [];
  const targetSlug = activeEpisodeSlug(projectRoot, history);
  const target = status.find((s) => s.slug === targetSlug) ?? status[0]!;
  if (target.done.includes('master') && target.todo.length === 0) return [];
  return target.todo.filter((t) => t !== 'script');
}

/**
 * When each word is said inside each shot, in the shot's own frames (0..len): what beats need to land on words.
 * Read from words.json (seconds) and timeline.json (each shot's segments). Empty when either is missing.
 */
export function shotWordFrames(projectRoot: string, slug: string, ids?: string[], maxWords = 40): string[] {
  const dir = path.join(projectRoot, 'src', 'episodes', slug);
  let words: any[] = [];
  let tl: any = null;
  try {
    words = JSON.parse(fs.readFileSync(path.join(dir, 'words.json'), 'utf8'));
    tl = JSON.parse(fs.readFileSync(path.join(dir, 'timeline.json'), 'utf8'));
  } catch {
    return [];
  }
  if (!Array.isArray(words) || !Array.isArray(tl?.segments)) return [];
  const fps = Number(tl.fps ?? 30);
  const lens = new Map<string, number>((tl.shots ?? []).map((s: any) => [String(s.id), Number(s.len ?? 0)]));
  const out: string[] = [];
  const wanted = ids ?? [...lens.keys()];
  for (const id of wanted) {
    const segs = tl.segments.filter((s: any) => String(s.shot) === id);
    if (!segs.length) continue;
    const marks: string[] = [];
    for (const w of words) {
      const f = Math.round(Number(w.s ?? 0) * fps);
      const seg = segs.find((s: any) => f >= Number(s.a) && f < Number(s.b));
      if (!seg) continue;
      const local = f - Number(seg.a) + Number(seg.t0 ?? 0);
      const text = String(w.w ?? '').replace(/[^\p{L}\p{N}'-]/gu, '');
      if (text) marks.push(`${text}@${local}`);
    }
    if (marks.length) out.push(`${id} (${lens.get(id) ?? '?'}f): ${marks.slice(0, maxWords).join(' ')}${marks.length > maxWords ? ' …' : ''}`);
  }
  return out;
}

/**
 * Pin the active episode's DIRECTOR.md, bible.json cast/places + refs state,
 * last authored shot JSON, and remaining unauthored shots with durationSec & word@frame anchors.
 */
export function episodePin(projectRoot: string, history?: MessageRow[]): string {
  const slug = activeEpisodeSlug(projectRoot, history);
  if (!slug) return '';
  const epDir = path.join(projectRoot, 'src', 'episodes', slug);
  if (!fs.existsSync(epDir)) return '';

  const parts: string[] = [`## Active episode continuity pin (${slug})`];

  const directorRaw = (readText(epDir, 'DIRECTOR.md', 8_000) ?? '').trim();
  if (directorRaw) {
    parts.push(`### DIRECTOR.md (pinned for continuity)\n${directorRaw.slice(0, 3_200)}`);
  }

  let epJson: any = null;
  let tlJson: any = null;
  try {
    epJson = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));
  } catch {
    /* ignore */
  }
  try {
    tlJson = JSON.parse(fs.readFileSync(path.join(epDir, 'timeline.json'), 'utf8'));
  } catch {
    /* ignore */
  }

  if (epJson) {
    const epLang = detectEpisodeLanguage(epJson);
    const langHint =
      epLang === 'en'
        ? `English (en) — continue the entire pipeline in English, write ALL in-world diegetic onScreenText strictly in English, NEVER add title cards.`
        : `French (fr) — keep ALL in-world diegetic onScreenText strictly in French, NEVER add title cards.`;
    const lockLines = [`- Language: ${langHint}`];
    if (epJson?.voice?.narratorGender || epJson?.voice?.voiceName) {
      const vName = String(epJson.voice.voiceName ?? (epJson.voice.narratorGender === 'female' ? 'Léa' : 'Hugo'));
      const vGen = String(epJson.voice.narratorGender ?? 'male');
      const vPov = String(epJson.voice.narratorPov ?? epJson?.source?.narratorPov ?? 'first');
      const povHint =
        vPov === 'third'
          ? `3rd-person narration (pov="third"): include ONLY story characters in bible.json — never invent a separate on-screen narrator.`
          : `1st-person narration (pov="${vPov}"): protagonist matches narratorGender="${vGen}".`;
      lockLines.push(`- Voice: ${vName} (${vGen}, pov=${vPov}). ${povHint}`);
    }
    lockLines.push(
      `- Audio rule: Omni Flash 2.1 sfxHint = concrete physical foley/ambience ONLY (strictly NO music, NO speech/dialogue).`,
    );
    parts.push(`### Language, narrator & diffusion rules\n${lockLines.join('\n')}`);
  }

  const bibleRes = readEpisodeBible(epDir);
  if (bibleRes.bible && bibleRes.errors.length === 0) {
    const b = bibleRes.bible;
    const refsDir = path.join(projectRoot, 'out', slug, 'refs');
    const placeLines = b.places.map((p) => {
      const hasRef = fs.existsSync(path.join(refsDir, `place_${p.id}.png`));
      return `  - ${p.id} ("${p.name}"${p.recipe ? `, recipe=${p.recipe}` : ''}) [ref:${hasRef ? 'OK' : 'MISSING'}]: ${p.description}`;
    });
    const charLines = b.characters.map((c) => {
      const hasRef = fs.existsSync(path.join(refsDir, `char_${c.id}.png`));
      return `  - ${c.id} ("${c.name}"${c.role ? `, ${c.role}` : ''}) [ref:${hasRef ? 'OK' : 'MISSING'}]: ${c.appearance}`;
    });
    parts.push(
      `### bible.json roster (style: ${b.stylePrompt.slice(0, 220)})\nPlaces:\n${placeLines.join('\n') || '  (none)'}\nCharacters:\n${charLines.join('\n') || '  (none)'}`,
    );
  } else if (bibleRes.errors.length) {
    parts.push(`### bible.json invalid/missing\n- ${bibleRes.errors.slice(0, 6).join('\n- ')}`);
  }

  const epSrc = readEpisodeShots(epDir);
  const isStub = epSrc.isPlaceholder;
  const exportedIds = isStub ? [] : epSrc.exportedShotIds;
  const exportedSet = new Set(exportedIds);

  const tlShots: any[] = Array.isArray(tlJson?.shots) ? tlJson.shots : [];
  const epShotsMap = new Map<string, any>();
  for (const s of Array.isArray(epJson?.shots) ? epJson.shots : []) {
    if (s?.id) epShotsMap.set(String(s.id), s);
  }
  const editLookMap = new Map<string, string>();
  for (const e of Array.isArray(epJson?.edit) ? epJson.edit : []) {
    if (e?.shot && e?.look) editLookMap.set(String(e.shot), String(e.look));
  }
  const stageCards: Record<string, any> = {};
  for (const s of Array.isArray(epJson?.shots) ? epJson.shots : []) {
    if (s?.id && s?.stage && typeof s.stage === 'object') {
      stageCards[String(s.id).toUpperCase()] = s.stage;
    }
  }
  try {
    const rawStage = JSON.parse(fs.readFileSync(path.join(epDir, 'shots.stage.json'), 'utf8'));
    const fileCards = (rawStage?.cards && typeof rawStage.cards === 'object' ? rawStage.cards : rawStage) ?? {};
    for (const [k, v] of Object.entries(fileCards)) {
      if (/^S\d+$/i.test(k) && v && typeof v === 'object' && !stageCards[k.toUpperCase()]) {
        stageCards[k.toUpperCase()] = v;
      }
    }
  } catch {
    /* ignore */
  }

  if (tlShots.length > 1) {
    const fps = Number(tlJson?.fps ?? 30);
    parts.push(
      `### All shots (id sec size [look])\n${tlShots
        .map((s) => {
          const id = String(s.id);
          const epS = epShotsMap.get(id);
          const sec = (Math.max(2, Number(s.len ?? 60) / fps)).toFixed(1);
          const lk = editLookMap.get(id) ? `{${editLookMap.get(id)}}` : '';
          return `${id} ${sec}s ${String(epS?.size ?? s.size ?? '')}${lk}`;
        })
        .join(' · ')}`,
    );

    const missing = tlShots.filter((s) => !exportedSet.has(String(s.id)));
    parts.push(
      `### Shots progress (${exportedIds.length}/${tlShots.length} JSON prompts in shots/S*.json)\n` +
        `- Authored: ${exportedIds.length ? sortShotIds(exportedIds).join(', ') : '(none yet)'}\n` +
        `- Remaining: ${missing.length ? missing.map((s) => s.id).join(', ') : '(all shots authored!)'}`,
    );

    if (missing.length > 0) {
      const nextIds = missing.slice(0, 8).map((s) => String(s.id));
      const nextBatch = nextIds.map((id) => {
        const tlS = tlShots.find((x) => String(x.id) === id) ?? {};
        const epS = epShotsMap.get(id) ?? {};
        const card = stageCards[id.toUpperCase()];
        const lenNum = Number(tlS.len || epS.key || 60);
        const durSec = Math.max(2.0, Number((lenNum / fps).toFixed(2)));
        const mood = editLookMap.get(id) ? `, mood="${editLookMap.get(id)}"` : '';
        const cardTxt = card
          ? (() => {
              const ppl = (Array.isArray(card.people) ? card.people : [])
                .map((p: any) => (typeof p === 'string' ? p : `${p?.who ?? '?'}(${p?.where ?? ''}): ${p?.does ?? ''}`))
                .join(' ; ');
              const mv = Array.isArray(card.moves) ? card.moves.join(',') : String(Boolean(card.moves));
              const objs = Array.isArray(card.objects) && card.objects.length ? ` objects=[${card.objects.join(',')}]` : '';
              return ` | stage: subject="${card.subject ?? ''}" see="${card.see ?? ''}" people=[${ppl}]${objs} view=${card.view ?? '?'} angle=${card.angle ?? '?'} moves=${mv}`;
            })()
          : '';
        return `  - ${id}: len=${durSec}s (${lenNum}f), size="${epS.size ?? tlS.size ?? 'medium'}", set="${epS.set ?? ''}"${mood}, note="${epS.note ?? ''}"${cardTxt}`;
      });
      parts.push(
        `### Next shots to write via edit_shot (batch 4–8 per call; use exact UPPERCASE place/characters IDs from bible.json):\n${nextBatch.join('\n')}`,
      );
      const timing = shotWordFrames(projectRoot, slug, nextIds);
      if (timing.length) {
        parts.push(`### Word anchors per shot (word@frame)\n${timing.join('\n')}`);
      }
    }
  }

  if (!isStub && exportedIds.length > 0) {
    const lastId = sortShotIds(exportedIds).slice(-1)[0]!;
    const lastSpec = epSrc.shots.get(lastId);
    if (lastSpec) {
      parts.push(
        `### Last authored shot (${lastId})\n\`\`\`json\n${JSON.stringify(lastSpec, null, 2).slice(0, 1_400)}\n\`\`\``,
      );
    }
  }

  return parts.length > 1 ? parts.join('\n\n') : '';
}

export type BuildOptions = {
  projectRoot: string;
  history: MessageRow[];
  maxChars: number;
};

/** Compact large write_file / write_bible / edit_shot payloads from older turns without emitting fake code. */
function compactToolCalls(calls: ToolCall[] | undefined, age: number): ToolCall[] | undefined {
  if (!calls?.length) return undefined;
  if (age <= 4) return calls;
  return calls.map((c) => {
    if (c.name === 'write_file' && typeof c.args?.content === 'string' && c.args.content.length > 1_200) {
      const lines = c.args.content.split('\n').length;
      return {
        ...c,
        args: {
          ...c.args,
          content: `[compacted: ${lines} lines (${c.args.content.length} chars) were written to disk by this call. System note, not file content: read_file the path to see it]`,
        },
      };
    }
    if (c.name === 'write_bible') {
      const ser = JSON.stringify(c.args ?? {});
      if (ser.length > 1_200) {
        return {
          ...c,
          args: {
            episode: c.args?.episode,
            summary: `[compacted: bible.json (${ser.length} chars) was written to disk by write_bible. System note: see continuity pin or read_file bible.json]`,
          },
        };
      }
    }
    if (c.name === 'edit_shot') {
      const ser = JSON.stringify(c.args ?? {});
      if (ser.length > 1_200) {
        return {
          ...c,
          args: {
            episode: c.args?.episode,
            id: c.args?.id,
            summary: `[compacted: shot spec (${ser.length} chars) was written to disk by edit_shot. System note: read_file shots/S*.json to inspect]`,
          },
        };
      }
    }
    return c;
  });
}

function messageCharCost(msg: ChatMessage): number {
  let cost = 'content' in msg ? msg.content.length : 0;
  if (msg.role === 'assistant' && msg.toolCalls?.length) {
    for (const tc of msg.toolCalls) {
      cost += tc.name.length + JSON.stringify(tc.args ?? {}).length + 24;
    }
  }
  if (msg.role === 'user' && msg.images?.length) {
    cost += msg.images.length * 2_000;
  }
  return cost;
}

/** Compile the conversation: 3+ stable leading system prefix messages for Anthropic caching, live state, then compacted history. */
export function buildMessages(opts: BuildOptions): ChatMessage[] {
  const pin = episodePin(opts.projectRoot, opts.history);
  const head: ChatMessage[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'system', content: DIRECTOR_GUIDE },
    { role: 'system', content: PROMPTING_GUIDE },
    { role: 'system', content: `## Project folder\n${opts.projectRoot}\n\n${memory(opts.projectRoot)}` },
    { role: 'system', content: `## Files\n${fileList(opts.projectRoot)}` },
    {
      role: 'system',
      content: `## Where the episode stands (live, recomputed each turn)\n${projectState(opts.projectRoot)}\n\nDo the next todo step; do not re-read files you already know unless a step fails.`,
    },
    ...(pin ? [{ role: 'system' as const, content: pin }] : []),
  ];
  const headChars = head.reduce((n, m) => n + m.content.length, 0);
  const effectiveMaxChars = opts.maxChars < 20_000 ? opts.maxChars : Math.max(opts.maxChars, headChars + 48_000);

  const history = opts.history;
  const kept: ChatMessage[] = [];
  let used = headChars;
  let dropped = 0;
  const imageBudget: ImageBudget = { left: 4_500_000, seen: new Set<string>() };

  for (let i = history.length - 1; i >= 0; i--) {
    const row = history[i]!;
    const payload = row.payload ? JSON.parse(row.payload) : null;
    if (row.role === 'assistant' && !row.content.trim() && !payload?.toolCalls?.length) continue;
    const age = history.length - 1 - i;
    let content = row.content;
    if (row.role === 'tool' && age > 3) content = content.slice(0, TOOL_BUDGET);
    if (row.role === 'assistant' && age > 6) content = content.slice(0, 1_500);

    const msg = toChatMessage(row, payload, content, opts.projectRoot, imageBudget, age);
    const cost = messageCharCost(msg);
    if (used + cost > effectiveMaxChars && kept.length >= 4) {
      dropped = i + 1;
      break;
    }
    used += cost;
    kept.unshift(msg);
  }

  while (kept.length > 0 && kept[0]!.role === 'tool') {
    kept.shift();
    dropped++;
  }

  const validCallIds = new Set<string>();
  const presentToolIds = new Set<string>();
  for (const m of kept) {
    if (m.role === 'assistant' && m.toolCalls) {
      for (const tc of m.toolCalls) validCallIds.add(tc.id);
    } else if (m.role === 'tool' && m.toolCallId) {
      presentToolIds.add(m.toolCallId);
    }
  }
  const filtered = kept.filter((m) => m.role !== 'tool' || validCallIds.has(m.toolCallId));
  const sanitized: ChatMessage[] = [];
  for (let i = 0; i < filtered.length; i++) {
    const m = filtered[i]!;
    sanitized.push(m);
    if (m.role === 'assistant' && m.toolCalls?.length) {
      const missingCalls = m.toolCalls.filter((tc) => !presentToolIds.has(tc.id));
      if (missingCalls.length > 0) {
        while (i + 1 < filtered.length && filtered[i + 1]!.role === 'tool') {
          i++;
          sanitized.push(filtered[i]!);
        }
        for (const mc of missingCalls) {
          presentToolIds.add(mc.id);
          sanitized.push({
            role: 'tool',
            toolCallId: mc.id,
            name: mc.name,
            content: 'ERROR\n(tool call interrupted before completion)',
          });
        }
      }
    }
  }

  const out = [...head];
  if (dropped > 0) {
    out.push({
      role: 'system',
      content: `(${dropped} older messages were compacted to stay inside the context window. Check Where the episode stands above or read_file if you need details; .git keeps checkpoints.)`,
    });
    if (sanitized.length === 0 || sanitized[0]!.role !== 'user') {
      let anchorText = '';
      for (let j = Math.min(dropped - 1, history.length - 1); j >= 0; j--) {
        const r = history[j];
        if (r && r.role === 'user' && !r.content.startsWith('[orchestrateur]')) {
          anchorText = r.content.slice(0, 2_500);
          break;
        }
      }
      const resumeNote = `[Contexte : ${dropped} étapes précédentes ont été compactées. Poursuis directement à partir des étapes récentes ci-dessous sans recommencer du début ni relire les fichiers déjà lus.]`;
      sanitized.unshift({
        role: 'user',
        content: anchorText ? `${anchorText}\n\n${resumeNote}` : resumeNote,
      });
    }
  }
  out.push(...sanitized);
  return out;
}

function toChatMessage(
  row: MessageRow,
  payload: any,
  content: string,
  projectRoot: string,
  imageBudget: ImageBudget,
  age = 0,
): ChatMessage {
  if (row.role === 'tool') {
    return { role: 'tool', toolCallId: String(payload?.toolCallId ?? ''), name: String(payload?.name ?? 'tool'), content };
  }
  if (row.role === 'assistant') {
    return { role: 'assistant', content, toolCalls: compactToolCalls(payload?.toolCalls, age) };
  }
  if (row.role === 'system') return { role: 'system', content };
  const { images, skippedDuplicate } = imagesFromPayload(payload, projectRoot, imageBudget);
  const finalContent =
    skippedDuplicate > 0 && images.length === 0
      ? `${content}\n(fichier(s) image mis à jour sur disque et attachés uniquement au message le plus récent)`
      : content;
  return { role: 'user', content: finalContent, images };
}

/** The text a tool result is stored as: what the model reads on the next turn. */
export function toolMessageContent(result: { ok: boolean; summary: string; artifacts?: { path: string; kind: string }[] }): string {
  const arts = (result.artifacts ?? []).filter((a) => a.kind !== 'text');
  const suffix = arts.length ? `\n[files: ${arts.map((a) => `${a.path} (${a.kind})`).join(', ')}]` : '';
  return `${result.ok ? 'ok' : 'ERROR'}\n${result.summary}${suffix}`.slice(0, 12_000);
}
