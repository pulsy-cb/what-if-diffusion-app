import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { layoutReport } from '../orchestrator/layout.js';
import { countScriptWords, detectEpisodeLanguage } from '../orchestrator/context.js';
export { detectEpisodeLanguage };
import { exec } from './exec.js';
import type { Artifact, Tool, ToolResult } from './types.js';
import { kindOf, mimeOf, relOf, safeJoin } from './paths.js';
import {
  readEpisodeBible,
  readEpisodeShots,
  sortShotIds,
  validateShotSpec,
  type DiffusionShotSpec,
} from './episode-source.js';
import { missingCards, stagingPromptProblems, usesStaging } from './staging.js';

const PY = process.platform === 'win32' ? 'python' : 'python3';
const PIPELINE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'pipeline');

const EPISODE_STUB = (slug: string, aspect: '16:9' | '9:16' = '16:9') => ({
  id: slug.replace(/(^|-)([a-z])/g, (_, __, c: string) => c.toUpperCase()),
  slug,
  aspect,
  topic: '',
  script: [
    { beat: '1 cold open, demo 1', text: 'TODO' },
    { beat: '2 premise & rising tension', text: 'TODO' },
    { beat: '3 the turn: the cost, one real fact', text: 'TODO — with one real, sourced number.' },
    { beat: '4 closing question', text: 'TODO?' },
  ],
  science: [{ claim: 'TODO', working: 'TODO — the arithmetic.', source: 'TODO' }],
  // one placeholder shot so plan_run can verify stub schema before voice + shotlist exist; stills refuses to render it
  shots: [{ id: 'S1', size: 'wide', set: 'living', key: 15, note: 'placeholder: replace with real shot' }],
  edit: [{ shot: 'S1', at: 'f:0' }],
  fx: [],
  sound: { provider: 'omni-flash', music: false, dialogue: false },
});

const TIMELINE_STUB = (slug: string, aspect: '16:9' | '9:16' = '16:9') => ({
  slug,
  title: 'PLACEHOLDER',
  aspect,
  width: aspect === '9:16' ? 1080 : 1920,
  height: aspect === '9:16' ? 1920 : 1080,
  fps: 30,
  total: 300,
  segments: [{ shot: 'S1', a: 0, b: 300, t0: 0, t1: 300 }],
  shots: [{ id: 'S1', size: 'wide', key: 15, len: 300, first: 0 }],
  fx: [],
  audio: { provider: 'omni-flash', music: false, dialogue: false },
});

const SHOT_JSON_PLACEHOLDER = {
  id: 'S1',
  place: 'LIVING',
  characters: [],
  size: 'wide',
  imagePrompt: 'PLACEHOLDER — replace with real Nano Banana 2 Lite scene prompt via edit_shot.',
  videoPrompt: 'PLACEHOLDER — replace with real Gemini Omni Flash 2.1 motion prompt.',
  sfxHint: 'quiet room tone',
};

export function normalizeAspect(raw: unknown): '16:9' | '9:16' | null {
  const s = String(raw ?? '').trim().toLowerCase();
  if (!s) return null;
  if (/\b(9\s*[:/]\s*16|1080\s*x\s*1920|portrait|vertical|shorts?|tiktok|reels)\b/.test(s)) {
    return '9:16';
  }
  if (/\b(16\s*[:/]\s*9|1920\s*x\s*1080|landscape|horizontal|paysage)\b/.test(s)) {
    return '16:9';
  }
  return null;
}

/**
 * Keep episode.json.aspect and PROJECT.md aligned so 16:9 (1920x1080) and 9:16 (1080x1920)
 * stay consistent across refs, stills, board, I2V clips, and final assembly.
 */
export function syncStudioAspect(projectRoot: string, slug: string, explicit?: unknown): '16:9' | '9:16' {
  const epPath = path.join(projectRoot, 'src', 'episodes', slug, 'episode.json');
  const projMdPath = path.join(projectRoot, 'PROJECT.md');

  let ep: Record<string, any> | null = null;
  try {
    ep = JSON.parse(fs.readFileSync(epPath, 'utf8'));
  } catch {
    ep = null;
  }

  let fromMd: '16:9' | '9:16' | null = null;
  try {
    const md = fs.readFileSync(projMdPath, 'utf8');
    const m = /^-\s*(?:aspect|orientation|format)\s*:\s*([^\r\n]+)/im.exec(md);
    if (m) fromMd = normalizeAspect(m[1]);
  } catch {
    /* ignore */
  }

  const fromEp = normalizeAspect(ep?.aspect ?? ep?.format ?? ep?.orientation);
  const aspect: '16:9' | '9:16' = normalizeAspect(explicit) ?? fromMd ?? fromEp ?? '16:9';

  if (ep) {
    let epChanged = false;
    if (ep.aspect !== aspect) {
      ep.aspect = aspect;
      epChanged = true;
    }
    if (ep.source?.videoId && Array.isArray(ep.science) && ep.science.some((s: any) => /\bTODO\b/i.test(JSON.stringify(s)))) {
      ep.science = ep.science.filter((s: any) => !/\bTODO\b/i.test(JSON.stringify(s)));
      epChanged = true;
    }
    if (epChanged) {
      try {
        fs.writeFileSync(epPath, JSON.stringify(ep, null, 1) + '\n');
      } catch {
        /* ignore */
      }
    }
  }
  return aspect;
}

/** Snapshot mtimes so we can report what a step produced. */
function snapshot(root: string): Map<string, number> {
  const out = new Map<string, number>();
  for (const sub of ['out', 'public']) {
    const dir = path.join(root, sub);
    if (!fs.existsSync(dir)) continue;
    const stack = [dir];
    while (stack.length) {
      const d = stack.pop()!;
      let entries: fs.Dirent[] = [];
      try {
        entries = fs.readdirSync(d, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const e of entries) {
        const abs = path.join(d, e.name);
        if (e.isDirectory()) stack.push(abs);
        else {
          try {
            out.set(relOf(root, abs), fs.statSync(abs).mtimeMs);
          } catch {
            /* ignore */
          }
        }
      }
    }
  }
  return out;
}

function diffArtifacts(root: string, before: Map<string, number>): Artifact[] {
  const after = snapshot(root);
  const changed: Artifact[] = [];
  for (const [rel, mtime] of after) {
    if (before.get(rel) === mtime) continue;
    const kind = kindOf(rel);
    if (kind === 'file') continue;
    changed.push({ kind, path: rel, mime: mimeOf(rel) });
  }
  return changed.sort((a, b) => a.path.localeCompare(b.path)).slice(0, 40);
}

const FR_LANG_RE = /\b(je|tu|il|elle|nous|vous|le|la|les|un|une|des|du|dans|sur|pour|avec|est|sont|mais|plus|pas|que|qui|était|avait|suis|ma|mes|sa|ses)\b/gi;

/**
 * Validate the script, source translation fidelity (when imported from :8005), science backing (when standalone),
 * and language consistency in episode.json.
 */
export function episodeAudit(episode: any, slug: string): string | null {
  const script = Array.isArray(episode?.script) ? episode.script : [];
  const hasSourceVideo = Boolean(episode?.source?.videoId);
  const minBeats = hasSourceVideo ? 1 : 3;
  if (script.length < minBeats) {
    return `episode.json.script must contain at least ${minBeats} beat(s); found ${script.length}.`;
  }
  const fullText = script.map((b: any) => String(b?.text ?? '')).join(' ');
  if (/\bTODO\b/i.test(fullText)) {
    return `episode.json.script still contains "TODO": write the complete narration (or run whatif_run { step: "source", args: ["--pick", "hook", "--apply"] }) before running the pipeline.`;
  }

  const scriptWords = countScriptWords(script);
  const sourceWords = Number(episode?.source?.words ?? 0);
  if (hasSourceVideo && sourceWords >= 40 && scriptWords < Math.floor(sourceWords * 0.75)) {
    return (
      `episode.json.script (${scriptWords} words) shortened or rewrote the :8005 translation proposal (${sourceWords} words). ` +
      `Do NOT shorten or rewrite the server's translation proposal (projects can last several minutes): keep the full narration from out/${slug}/source.fr.md ` +
      `by running whatif_run { episode: "${slug}", step: "source", args: ["--pick", "hook", "--apply"] }.`
    );
  }

  const science = Array.isArray(episode?.science) ? episode.science : [];
  const nonTodoScience = science.filter((s: any) => !/\bTODO\b/i.test(`${s?.claim ?? ''} ${s?.working ?? ''} ${s?.source ?? ''}`));
  if (!hasSourceVideo) {
    if (
      !nonTodoScience.length ||
      science.some(
        (s: any) =>
          !String(s?.claim ?? '').trim() ||
          !String(s?.working ?? '').trim() ||
          !String(s?.source ?? '').trim() ||
          /\bTODO\b/i.test(`${s?.claim ?? ''} ${s?.working ?? ''} ${s?.source ?? ''}`),
      )
    ) {
      return `episode.json.science must contain at least one real, worked-out fact with non-TODO "claim", "working" and "source".`;
    }
  }

  const epLang = detectEpisodeLanguage(episode);
  const frHits = (fullText.match(FR_LANG_RE) ?? []).length;
  const isFrench = epLang === 'fr' && frHits >= 4;
  const titleStr = Array.isArray(episode?.title) ? episode.title.join(' ') : String(episode?.title ?? '');
  if (isFrench && /\bwhat\s+if\s+you\s+could\b/i.test(titleStr)) {
    return `episode.json.title mixes English ("WHAT IF YOU COULD") with a French episode ("${titleStr}").`;
  }
  if (isFrench && /\bwhat\s+if\s+you\s+could\b/i.test(fullText)) {
    return `episode.json.script contains the English phrase "What if you could" inside a French narration (${slug}): write it in French so the French TTS voice does not pronounce English.`;
  }
  if (epLang === 'en' && /\bet\s+si\s+tu\s+pouvais\b/i.test(titleStr)) {
    return `episode.json.title still has the French placeholder ("${titleStr}") on an English episode (${slug}).`;
  }
  return null;
}

/** The pace the shot list must reach: shorts cut fast, long videos hold 3 to 6 s. */
export function paceTarget(durSec: number, aspect = '16:9'): { minCutsPerMin: number; maxShotSec: number; label: string } {
  if (durSec <= 30) return { minCutsPerMin: 24, maxShotSec: 3.8, label: 'short clip' };
  if (durSec <= 150) return { minCutsPerMin: 20, maxShotSec: 5, label: 'short (under 2.5 min)' };
  if (aspect === '9:16') return { minCutsPerMin: 20, maxShotSec: 5, label: 'vertical short (9:16, any length)' };
  return { minCutsPerMin: 10, maxShotSec: 9, label: 'long video (3-6 s per shot)' };
}

/** The orientation the project renders at: episode.json first, else 16:9. */
export function projectAspect(_projectRoot: string, ep?: any): '16:9' | '9:16' {
  const a = String(ep?.aspect ?? '');
  if (a === '9:16' || a === '16:9') return a;
  return '16:9';
}

/** Null when timeline.json keeps the pace, else what is wrong and how to fix it. */
export function paceProblem(projectRoot: string, slug: string): string | null {
  const tl = readJsonFile(path.join(projectRoot, 'src', 'episodes', slug, 'timeline.json'));
  const segs = (tl?.segments ?? []) as any[];
  const fps = Math.max(1, Number(tl?.fps ?? 30));
  const durSec = Number(tl?.total ?? 0) / fps;
  if (durSec < 12 || !segs.length) return null;
  const cuts = Math.max(0, segs.length - 1);
  const cutsPerMin = (cuts / durSec) * 60;
  const longestSec = segs.reduce((m, s) => Math.max(m, (Number(s.b ?? 0) - Number(s.a ?? 0)) / fps), 0);
  const ep = readJsonFile(path.join(projectRoot, 'src', 'episodes', slug, 'episode.json'));
  const want = paceTarget(durSec, projectAspect(projectRoot, ep));
  if (cutsPerMin >= want.minCutsPerMin && longestSec <= want.maxShotSec) return null;
  return (
    `PACE TOO SLOW for a ${want.label}: ${segs.length} shots over ${durSec.toFixed(1)}s (${cutsPerMin.toFixed(1)} cuts/min, longest ${longestSec.toFixed(1)}s; ` +
    `wanted >= ${want.minCutsPerMin} cuts/min and no shot over ${want.maxShotSec}s). Run whatif_run { step: "shotlist" } then { step: "plan" }.`
  );
}

function canonicalShotSize(raw: string): string | null {
  const s = raw
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim();
  if (!s) return null;
  if (/\b(insert|macro|detail|objet)\b/.test(s)) return 'insert';
  if (/\b(close|closeup|serre|gros\s+plan|portrait)\b/.test(s)) return 'close';
  if (/\b(medium|moyen|americain|two-shot)\b/.test(s)) return 'medium';
  if (/\b(wide|large|ensemble|establishing)\b/.test(s)) return 'wide';
  return null;
}

/**
 * Compare shot sizes declared in DIRECTOR.md tables/bullets ("| S6 | close | ...") against episode.json.shots
 * ("S6": size "insert"), catching silent reclassifications made to dodge character checks.
 */
export function directorSizeMismatches(directorMd: string, episode: any): string[] {
  if (!directorMd || !Array.isArray(episode?.shots)) return [];
  const epById = new Map<string, string>();
  for (const s of episode.shots) {
    if (s?.id && s?.size) epById.set(String(s.id).trim().toUpperCase(), String(s.size).trim());
  }
  if (!epById.size) return [];
  const dirSizes = new Map<string, string>();
  for (const m of directorMd.matchAll(/^\s*\|\s*\*?\*?(S\d+)\*?\*?\s*\|\s*([^|\n]+)\|/gim)) {
    dirSizes.set(m[1]!.toUpperCase(), m[2]!.trim());
  }
  for (const m of directorMd.matchAll(/^\s*(?:[-*]|#{2,4})\s*\*?\*?(S\d+)\*?\*?\s*(?:[\[(—–:-]\s*)([^|\n:—–)\]]+)/gim)) {
    const id = m[1]!.toUpperCase();
    if (!dirSizes.has(id)) dirSizes.set(id, m[2]!.trim());
  }
  const out: string[] = [];
  for (const [id, dirRaw] of dirSizes) {
    const epRaw = epById.get(id);
    if (!epRaw) continue;
    const dBucket = canonicalShotSize(dirRaw);
    const eBucket = canonicalShotSize(epRaw);
    if (dBucket && eBucket && dBucket !== eBucket) {
      out.push(`${id} (DIRECTOR.md="${dirRaw}" vs episode.json="${epRaw}")`);
    }
  }
  return out;
}

/**
 * Audit the diffusion Bible (`bible.json`) and per-shot prompt specs (`shots/S<N>.json`)
 * against `timeline.json`, `episode.json` staging cards, language lock, and anti-title / SFX rules.
 */
export function shotsAudit(
  projectRoot: string,
  slug: string,
  episode: any,
  timeline: any,
): string | null {
  const epDir = path.join(projectRoot, 'src', 'episodes', slug);
  const tlShots: { id: string; len: number; size?: string }[] = Array.isArray(timeline?.shots) ? timeline.shots : [];
  const wanted = sortShotIds(tlShots.map((s) => String(s.id ?? '').toUpperCase()).filter(Boolean));
  if (!wanted.length) return `timeline.json has no shots: run whatif_run { step: "plan" } first`;

  const bibleRes = readEpisodeBible(epDir);
  if (!bibleRes.exists) {
    return `src/episodes/${slug}/bible.json is missing: call write_bible first to define stylePrompt, characters, and places.`;
  }
  if (!bibleRes.bible || bibleRes.errors.length) {
    return `bible.json failed validation:\n- ${bibleRes.errors.join('\n- ')}`;
  }
  const bible = bibleRes.bible;
  const placeIds = new Set(bible.places.map((p) => p.id));
  const charIds = new Set(bible.characters.map((c) => c.id));

  const bundle = readEpisodeShots(epDir);
  if (bundle.isPlaceholder) {
    return `src/episodes/${slug}/shots/ still only has the placeholder: write all shot prompts (${wanted.join(', ')}) via edit_shot.`;
  }

  const missing = wanted.filter((id) => !bundle.shotFiles.has(id));
  if (missing.length) {
    return `Missing shot prompt file(s) in shots/: ${missing.join(', ')} (write them via edit_shot).`;
  }

  const errors: string[] = [];
  const epShots: Record<string, any> = {};
  for (const s of Array.isArray(episode?.shots) ? episode.shots : []) {
    if (s?.id) epShots[String(s.id).toUpperCase()] = s;
  }

  const epLang = detectEpisodeLanguage(episode);
  const validatedSpecs = new Map<string, DiffusionShotSpec>();
  let charShotCount = 0;

  for (const id of wanted) {
    const rawText = bundle.shotFiles.get(id) ?? '';
    let parsed: unknown;
    try {
      parsed = JSON.parse(rawText);
    } catch (e) {
      errors.push(`${id}.json is invalid JSON: ${(e as Error).message}`);
      continue;
    }
    const { shot, errors: specErrs } = validateShotSpec(parsed, id);
    if (!shot || specErrs.length) {
      errors.push(...specErrs);
      continue;
    }
    validatedSpecs.set(id, shot);

    if (!placeIds.has(shot.place)) {
      errors.push(`${id}.place "${shot.place}" is not defined in bible.json.places (${[...placeIds].join(', ')})`);
    }
    for (const cid of shot.characters) {
      if (!charIds.has(cid)) {
        errors.push(`${id}.characters includes "${cid}" which is not defined in bible.json.characters`);
      }
    }
    if (shot.characters.length > 0) charShotCount++;

    if (shot.onScreenText && shot.onScreenText.text) {
      if (shot.onScreenText.lang && shot.onScreenText.lang !== epLang) {
        errors.push(`${id}.onScreenText.lang is "${shot.onScreenText.lang}" on a "${epLang}" episode: match episode language`);
      }
      if (epLang === 'fr' && /\b(YOU PUT ITCHING|WHAT IF YOU|MY BEST FRIEND|THE LAST GUY)\b/i.test(shot.onScreenText.text)) {
        errors.push(`${id}.onScreenText contains untranslated English text in a French episode`);
      }
    }
  }

  // Ensure at least 30% of shots show characters when characters exist and not all shots are inserts/establishing
  const nonInsertCount = wanted.filter((id) => !/^(insert|establishing)$/i.test(String(epShots[id]?.size ?? ''))).length;
  const minCharShots = bible.characters.length > 0 && nonInsertCount >= 3 ? Math.max(1, Math.ceil(wanted.length * 0.3)) : 0;
  if (minCharShots > 0 && charShotCount < minCharShots) {
    errors.push(
      `Only ${charShotCount}/${wanted.length} shots show any character (need at least ${minCharShots}): attach character IDs from bible.json in shots/S*.json`,
    );
  }

  const epShotList: any[] = Array.isArray(episode?.shots) ? episode.shots : [];
  const staged = usesStaging(epShotList);
  const stagingErrs = [
    ...(staged ? missingCards(epShotList) : []),
    ...stagingPromptProblems(epShotList, validatedSpecs, bible, staged, projectAspect(projectRoot, episode)),
  ];
  errors.unshift(...stagingErrs.slice(0, 8));

  if (errors.length) {
    return `Diffusion prompt audit failed:\n- ${errors.slice(0, 12).join('\n- ')}`;
  }
  return null;
}

/**
 * Verify that `bible.json` and `shots/S*.json` were not edited AFTER the last `board` (and `board/S*.png`),
 * so `clips` / `render` never ship outdated images that the Reviewer never saw.
 */
export function boardStale(projectRoot: string, slug: string): string | null {
  const mtime = (file: string): number => {
    try {
      return fs.statSync(file).mtimeMs;
    } catch {
      return 0;
    }
  };
  const board = mtime(path.join(projectRoot, 'out', slug, 'storyboard.jpg'));
  const bundle = readEpisodeShots(path.join(projectRoot, 'src', 'episodes', slug));
  if (!board || !bundle.latestMtimeMs || bundle.latestMtimeMs <= board + 1_000) return null;
  return `bible.json or shots/ changed after the last board, so nobody has inspected the updated images yet: re-run whatif_run { step: "stills", args: [<changed shots>] } then { step: "board" } before clips/render`;
}

/** Why the render / clips must wait for the independent reviewer, or null. */
export function reviewRefusal(projectRoot: string, slug: string): string | null {
  const review = readJsonFile(path.join(projectRoot, 'out', slug, 'review.json'));
  if (!review?.blocking) return null;
  const todo = (review.issues ?? []).filter((i: any) => i.severity === 'error').map((i: any) => `[${i.shot}] ${i.problem}`);
  return `the independent reviewer refused this board. Fix these shots with edit_shot, then re-run stills for them and board (the reviewer looks again):\n- ${todo.slice(0, 8).join('\n- ')}`;
}

function missingRefImages(projectRoot: string, slug: string, epDir: string, wantedIds: string[]): string[] {
  const bibleRes = readEpisodeBible(epDir);
  if (!bibleRes.bible) return [];
  const refsDir = path.join(projectRoot, 'out', slug, 'refs');
  const bundle = readEpisodeShots(epDir);
  const missing: string[] = [];

  for (const id of wantedIds) {
    const spec = bundle.shots.get(id);
    if (!spec) continue;
    const placePng = path.join(refsDir, `place_${spec.place}.png`);
    if (!fs.existsSync(placePng)) missing.push(`out/${slug}/refs/place_${spec.place}.png`);
    for (const cid of spec.characters) {
      const charPng = path.join(refsDir, `char_${cid}.png`);
      if (!fs.existsSync(charPng)) missing.push(`out/${slug}/refs/char_${cid}.png`);
    }
  }
  return [...new Set(missing)];
}

/**
 * Enforce strict legal pipeline order so an unattended agent never skips prerequisites.
 */
function prerequisite(projectRoot: string, slug: string, step: string): string | null {
  const src = path.join(projectRoot, 'src', 'episodes', slug);
  let broken = '';
  const readJson = (file: string): any => {
    try {
      return JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (err) {
      broken = (err as Error).message;
      return null;
    }
  };
  const episodePath = path.join(src, 'episode.json');
  const episode = readJson(episodePath);
  if (!fs.existsSync(episodePath)) return `no src/episodes/${slug}/episode.json: scaffold the episode with whatif_new first`;
  if (!episode) return `src/episodes/${slug}/episode.json is not valid JSON (${broken}): repair it before running ${step}`;

  const scriptWords = countScriptWords(episode.script);
  const words = readJson(path.join(src, 'words.json'));
  const timeline = readJson(path.join(src, 'timeline.json'));
  const voiced = fs.existsSync(path.join(projectRoot, 'public', slug, 'voice.wav'));
  const staleTimeline =
    !timeline ||
    String(timeline.title ?? '') === 'PLACEHOLDER' ||
    !Array.isArray(timeline.segments) ||
    timeline.segments.length === 0 ||
    (Number(timeline.total ?? 0) === 300 && timeline.segments.length === 1 && (timeline.shots?.[0]?.len ?? 0) === 300 && !voiced);

  if (['voice', 'shotlist', 'plan', 'stills', 'clips', 'render'].includes(step)) {
    if (!scriptWords) return `episode.json has no script yet: write the script first (${slug})`;
    const epErr = episodeAudit(episode, slug);
    if (epErr) return epErr;
  }
  if (step === 'shotlist' || step === 'plan') {
    if (!words?.length) return `src/episodes/${slug}/words.json is missing: run whatif_run { step: "voice" } first`;
    if (!voiced) return `public/${slug}/voice.wav is missing: run whatif_run { step: "voice" } first`;
  }
  if (step === 'refs') {
    const bibleRes = readEpisodeBible(src);
    if (!bibleRes.exists) return `src/episodes/${slug}/bible.json is missing: call write_bible first`;
    if (!bibleRes.bible || bibleRes.errors.length) {
      return `bible.json failed validation:\n- ${bibleRes.errors.join('\n- ')}`;
    }
  }
  if (['stills', 'board', 'clips', 'render'].includes(step)) {
    const shots = (episode?.shots ?? []) as any[];
    if (shots.length && shots.every((sh) => /placeholder/i.test(String(sh.note ?? '')))) {
      return `${shots.length} placeholder shot(s) only: run whatif_run { step: "shotlist" } then { step: "plan" }, write DIRECTOR.md, stage_shots, write_bible, and edit_shot first`;
    }
    if (staleTimeline) {
      return `timeline.json is still the placeholder: run whatif_run { step: "voice" }, { step: "shotlist" }, and { step: "plan" } first`;
    }
    const directorPath = path.join(src, 'DIRECTOR.md');
    const directorText = fs.existsSync(directorPath) ? fs.readFileSync(directorPath, 'utf8').trim() : '';
    if (directorText.length < 200) {
      return (
        `src/episodes/${slug}/DIRECTOR.md is missing or incomplete (${directorText.length} chars): write a real Réalisation & Mise en Scène document in src/episodes/${slug}/DIRECTOR.md (>= 200 chars) covering:\n` +
        `1. Style 2D & lieux (3 couches de profondeur, palette, éclairage),\n` +
        `2. Casting & apparence invariante des personnages,\n` +
        `3. Découpage plan par plan sans titre (cadrage, angles, mouvements I2V & bruitages Omni Flash sans musique ni dialogue).`
      );
    }
    const pace = paceProblem(projectRoot, slug);
    if (pace) return pace;
    const auditErr = shotsAudit(projectRoot, slug, episode, timeline);
    if (auditErr) return auditErr;

    const wantedIds = sortShotIds((timeline?.shots ?? []).map((s: any) => String(s?.id ?? '').toUpperCase()).filter(Boolean));
    const missingRefs = missingRefImages(projectRoot, slug, src, wantedIds);
    if (missingRefs.length) {
      return `Missing reference image(s) (${missingRefs.slice(0, 6).join(', ')}): run whatif_run { episode: "${slug}", step: "refs" } first so Nano Banana 2 Lite has neutral character & place sheets.`;
    }
  }
  if (step === 'clips' || step === 'render') {
    if (!words?.length || !voiced) return `voice.wav / words.json missing: run whatif_run { step: "voice" } first`;
    if (!fs.existsSync(path.join(projectRoot, 'out', slug, 'storyboard.jpg'))) {
      return `out/${slug}/storyboard.jpg does not exist yet: run whatif_run { step: "stills" } and { step: "board" } and inspect the storyboard before ${step}`;
    }
    const stale = boardStale(projectRoot, slug);
    if (stale) return stale;
    const refusal = reviewRefusal(projectRoot, slug);
    if (refusal) return refusal;
  }
  if (step === 'master' && !fs.existsSync(path.join(projectRoot, 'out', slug, `${slug}-raw.mp4`))) {
    return `out/${slug}/${slug}-raw.mp4 does not exist: run whatif_run { step: "render" } first`;
  }
  if (step === 'upscale' && !fs.existsSync(path.join(projectRoot, 'out', slug, `${slug}-final.mp4`))) {
    return `out/${slug}/${slug}-final.mp4 does not exist: run whatif_run { step: "render" } (or { step: "master" }) first`;
  }
  return null;
}

const forwardLineEarly = (ctx: { log: (line: string) => void }) => (line: string) => ctx.log(line);

const readJsonFile = (file: string): any => {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
};

function episodeEdit(projectRoot: string, slug: string): any[] | null {
  return readJsonFile(path.join(projectRoot, 'src', 'episodes', slug, 'episode.json'))?.edit ?? null;
}

function anchorHints(projectRoot: string, slug: string): string {
  const words = readJsonFile(path.join(projectRoot, 'src', 'episodes', slug, 'words.json'));
  if (!Array.isArray(words) || !words.length) return '(no words.json: run voice first)';
  const step = Math.max(1, Math.floor(words.length / 14));
  const out: string[] = [];
  for (let i = 0; i < words.length; i += step) {
    const w = String(words[i]?.w ?? '').replace(/[^\p{L}\p{N}'-]/gu, '');
    if (w) out.push(`"at": "${w}" (${Number(words[i]?.s ?? 0).toFixed(2)}s)`);
  }
  return out.join(', ');
}

const STEPS: Record<string, { py: string; fixedArgs?: string[]; timeoutMs?: number; desc: string }> = {
  source: {
    py: 'source_import.py',
    timeoutMs: 1_200_000,
    desc: 'nouveau video id -> English transcript -> FR/EN narration proposals',
  },
  voice: {
    py: 'voice_fish.py',
    timeoutMs: 900_000,
    desc: 'ElevenLabs v4 TTS (Hugo/Léa) or Fish Audio through :8005 + Deepgram word timings',
  },
  shotlist: {
    py: 'shotlist.py',
    desc: 'cut measured words into brisk clause-aligned shots (no title shots)',
  },
  plan: {
    py: 'plan_run.py',
    desc: 'resolve cues into timeline.json (30 fps)',
  },
  refs: {
    py: 'image_gen.py',
    fixedArgs: ['--mode', 'refs'],
    timeoutMs: 900_000,
    desc: 'generate neutral character & place reference sheets with Nano Banana 2 Lite',
  },
  stills: {
    py: 'image_gen.py',
    fixedArgs: ['--mode', 'stills'],
    timeoutMs: 1_800_000,
    desc: 'generate per-shot storyboard stills with Nano Banana 2 Lite',
  },
  board: {
    py: 'board_run.py',
    timeoutMs: 180_000,
    desc: 'build storyboard.jpg + chunked sheets/sheet-*.jpg',
  },
  clips: {
    py: 'video_gen.py',
    timeoutMs: 3_600_000,
    desc: 'animate approved storyboard stills into MP4 + native SFX via Gemini Omni Flash 2.1',
  },
  render: {
    py: 'assemble_run.py',
    timeoutMs: 3_600_000,
    desc: 'generate missing I2V clips (Omni Flash 2.1), frame-align onto timeline.json + ducked SFX + voice.wav, and auto-master -14 LUFS',
  },
  motiongrid: {
    py: 'motiongrid_run.py',
    timeoutMs: 180_000,
    desc: 'fast-seek frame grids per shot from raw render',
  },
  master: {
    py: 'master_run.py',
    timeoutMs: 900_000,
    desc: 'fast -vn EBU R128 loudness mastering to -14 LUFS',
  },
  upscale: {
    py: 'upscale_run.py',
    timeoutMs: 1_800_000,
    desc: 'optional Anime 2D upscaler (Modal Real-CUGAN / Real-ESRGAN) to 1080p / 2K / 4K UHD while keeping -14 LUFS audio untouched',
  },
};

const newProject: Tool = {
  name: 'whatif_new',
  description:
    'Scaffold a 100% Diffusion episode workspace inside the project (`src/episodes/<slug>/episode.json`, `timeline.json`, `shots/S1.json`, `PROJECT.md`, `.gitignore`).',
  policy: 'ask',
  mutating: true,
  schema: {
    type: 'object',
    properties: {
      episode: { type: 'string', description: 'Slug, e.g. "pause-everyone".' },
      video: { type: 'string', description: 'nouveau (:8005) video id to use as the source of the episode.' },
      aspect: {
        type: 'string',
        enum: ['16:9', '9:16'],
        description: 'Video orientation: "16:9" (1920x1080 landscape) or "9:16" (1080x1920 vertical Shorts/Reels/TikTok).',
      },
      force: { type: 'boolean', description: 'Scaffold into a non-empty folder.' },
    },
    required: ['episode'],
    additionalProperties: false,
  },
  validate: (input) => (/^[a-z0-9][a-z0-9-]*$/.test(String(input.episode ?? '')) ? null : 'episode must be a kebab-case slug'),
  async run(input: { episode: string; video?: string; aspect?: string; force?: boolean }, ctx): Promise<ToolResult> {
    const slug = input.episode;
    const epJsonCheck = path.join(ctx.projectRoot, 'src', 'episodes', slug, 'episode.json');
    if (fs.existsSync(epJsonCheck) && !input.force) {
      throw new Error(`${ctx.projectRoot}/src/episodes/${slug} already exists (pass force: true to overwrite)`);
    }

    fs.mkdirSync(ctx.projectRoot, { recursive: true });
    fs.mkdirSync(path.join(ctx.projectRoot, 'src', 'episodes'), { recursive: true });
    fs.mkdirSync(path.join(ctx.projectRoot, 'public', slug), { recursive: true });
    fs.mkdirSync(path.join(ctx.projectRoot, 'out', slug), { recursive: true });

    fs.writeFileSync(
      path.join(ctx.projectRoot, '.gitignore'),
      'out/\npublic/*/*.wav\npublic/*/*.mp3\n*.mp4\n',
      'utf8',
    );
    const pkgPath = path.join(ctx.projectRoot, 'package.json');
    if (!fs.existsSync(pkgPath)) {
      fs.writeFileSync(pkgPath, JSON.stringify({ name: `whatif-${slug}`, private: true, type: 'module' }, null, 2) + '\n', 'utf8');
    }

    const aspect = normalizeAspect(input.aspect) ?? '16:9';
    const dir = safeJoin(ctx.projectRoot, `src/episodes/${slug}`);
    fs.mkdirSync(path.join(dir, 'shots'), { recursive: true });
    const ep: Record<string, any> = EPISODE_STUB(slug, aspect);
    if (input.video) {
      ep.source = { provider: 'nouveau', videoId: input.video };
      ep.voice = { provider: 'elevenlabs', voiceId: '3Tp5mYhT9wrfnry9pzvV', modelId: 'eleven_v4', speakingRate: 1.15 };
    }
    fs.writeFileSync(path.join(dir, 'episode.json'), JSON.stringify(ep, null, 1) + '\n');
    fs.writeFileSync(path.join(dir, 'timeline.json'), JSON.stringify(TIMELINE_STUB(slug, aspect), null, 1) + '\n');
    fs.writeFileSync(path.join(dir, 'shots', 'S1.json'), JSON.stringify(SHOT_JSON_PLACEHOLDER, null, 2) + '\n');
    syncStudioAspect(ctx.projectRoot, slug, aspect);

    const memory = path.join(ctx.projectRoot, 'PROJECT.md');
    if (!fs.existsSync(memory)) {
      fs.writeFileSync(
        memory,
        `# Project\n\n- mode: 100% diffusion (Nano Banana 2 Lite + Gemini Omni Flash 2.1)\n- episode: ${slug}\n- aspect: ${aspect} (${aspect === '9:16' ? '1080x1920 vertical' : '1920x1080 landscape'})\n${input.video ? `- source video (nouveau :8005): ${input.video}\n` : ''}- pipeline: source -> voice -> shotlist -> plan -> DIRECTOR.md -> stage_shots -> write_bible -> refs -> edit_shot -> stills -> board -> reviewer -> render (clips + assemble + master -14 LUFS)\n- voice: ElevenLabs TTS (default: eleven_v4, Hugo/Léa) through :8005\n- title cards: disabled\n- sfx: native Omni Flash 2.1 physical SFX + ambience (strictly no music, no dialogue)\n\n## Decisions\n\n- (the agent appends decisions here)\n`,
      );
    }

    return {
      ok: true,
      summary: `scaffolded 100% diffusion project at ${ctx.projectRoot} with episode "${slug}" (id ${ep.id}, aspect ${aspect}). Next: ${input.video ? 'run whatif_run { step: "source", args: ["--pick","hook","--apply"] }' : 'fill episode.json.script'} then voice -> shotlist -> plan.`,
      data: { projectRoot: ctx.projectRoot, episode: slug, id: ep.id, aspect },
    };
  },
};

const runStep: Tool = {
  name: 'whatif_run',
  description:
    'Run one diffusion pipeline step on an episode: source (:8005), voice (:8005 TTS/ASR), shotlist, plan, refs (Nano Banana 2 Lite reference sheets), stills (Nano Banana 2 Lite per-shot images), board, clips (Gemini Omni Flash 2.1 I2V + SFX), render (assemble I2V clips + ducked SFX + voice.wav + auto-master -14 LUFS + optional upscale), motiongrid, master, or upscale (Modal Anime 2D 1080p/2K/4K UHD).',
  policy: 'ask',
  schema: {
    type: 'object',
    properties: {
      episode: { type: 'string' },
      step: {
        type: 'string',
        enum: ['source', 'voice', 'shotlist', 'plan', 'refs', 'stills', 'board', 'clips', 'render', 'motiongrid', 'master', 'upscale'],
      },
      args: { type: 'array', items: { type: 'string' }, description: 'Extra argv, e.g. ["--pick","hook","--apply"], ["S2","S5"], or ["--target","auto-uhd"].' },
    },
    required: ['episode', 'step'],
    additionalProperties: false,
  },
  validate: (input) => {
    const slug = String(input.episode ?? '').trim();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(slug)) return 'episode must be a clean slug';
    const step = String(input.step ?? '');
    if (!STEPS[step]) return `unknown step ${step}`;
    return null;
  },
  async run(input: { episode: string; step: string; args?: string[] }, ctx): Promise<ToolResult> {
    const slug = String(input.episode ?? '').trim();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(slug)) throw new Error(`invalid episode slug "${slug}"`);
    const extra = (input.args ?? []).map(String);
    syncStudioAspect(ctx.projectRoot, slug);

    const blocked = prerequisite(ctx.projectRoot, slug, input.step);
    if (blocked) throw new Error(`cannot run ${input.step} yet — ${blocked}`);

    const before = snapshot(ctx.projectRoot);
    const rawMp4 = `out/${slug}/${slug}-raw.mp4`;
    const extraNotes: string[] = [];
    const childEnv = ctx.mediaEnv ?? process.env;

    const forwardLine = (line: string) => {
      if (line.startsWith('RESULT_JSON:')) return;
      if (line.startsWith('PROGRESS_JSON:')) {
        try {
          const p = JSON.parse(line.slice('PROGRESS_JSON:'.length));
          const msg = p.stage ? `[${p.stage}] ${p.detail ?? p.status ?? JSON.stringify(p)}` : JSON.stringify(p);
          ctx.log(msg);
        } catch {
          ctx.log(line.slice('PROGRESS_JSON:'.length));
        }
        return;
      }
      ctx.log(line);
    };

    const stepOnce = async (name: string): Promise<{ code: number; text: string; result?: Record<string, unknown> }> => {
      const def = STEPS[name]!;
      const argv2 = [PY, path.join(PIPELINE, def.py), slug, ...(def.fixedArgs ?? [])];
      ctx.log(`→ sub-step ${name} (${slug})…`);
      const r = await exec(argv2, { cwd: ctx.projectRoot, env: childEnv, timeoutMs: def.timeoutMs ?? 600_000, signal: ctx.signal, onLine: forwardLine });
      const raw = r.stdout.split('\n').filter(Boolean);
      const line = raw.filter((l) => l.startsWith('RESULT_JSON:')).pop();
      let result: Record<string, unknown> | undefined;
      try {
        result = line ? (JSON.parse(line.slice('RESULT_JSON:'.length)) as Record<string, unknown>) : undefined;
      } catch {
        result = undefined;
      }
      return { code: r.code, text: raw.filter((l) => !l.startsWith('RESULT_JSON:')).slice(-18).join('\n'), result };
    };

    // Before planning, ensure shotlist has been generated and keys are valid integers
    if (input.step === 'plan') {
      const epFile = path.join(ctx.projectRoot, 'src', 'episodes', slug, 'episode.json');
      const epData = readJsonFile(epFile);
      const shots = (epData?.shots ?? []) as any[];
      const coarse = shots.length < 2 || shots.every((sh) => /placeholder/i.test(String(sh.note ?? '')));
      if (coarse && fs.existsSync(path.join(ctx.projectRoot, 'src', 'episodes', slug, 'words.json'))) {
        const cut = await stepOnce('shotlist');
        extraNotes.push(`shot list built before planning (${shots.length || 1} -> ${String(cut.result?.shots ?? '?')} shots):\n${cut.text}`);
      } else if (epData && Array.isArray(epData.shots)) {
        let fixedKeys = false;
        for (const sh of epData.shots) {
          if (typeof sh.key !== 'number' || !Number.isInteger(sh.key) || sh.key < 6) {
            sh.key = 15;
            fixedKeys = true;
          }
        }
        if (fixedKeys) fs.writeFileSync(epFile, JSON.stringify(epData, null, 1) + '\n');
      }
    }

    if (input.step === 'render' || input.step === 'clips') {
      const preLayout = await layoutReport(path.join(ctx.projectRoot, 'out', slug, 'board')).catch(() => '');
      if (preLayout.includes('\nEMPTY:')) {
        throw new Error(`cannot run ${input.step} while storyboard stills fail layout checks:\n${preLayout}\nFix those shots with edit_shot and re-run stills + board first.`);
      }
    }

    const def = STEPS[input.step]!;
    const pyExtra =
      input.step === 'render' ? extra.filter((a) => a !== '--no-master' && a !== '--no-upscale') : extra;
    const argv = [PY, path.join(PIPELINE, def.py), slug, ...(def.fixedArgs ?? []), ...pyExtra];
    if (input.step === 'master' && !extra.includes('--skip-render')) {
      argv.push('--skip-render');
    }
    const timeoutMs = def.timeoutMs ?? 600_000;

    ctx.log(`whatif_run ${input.step} ${slug}: ${argv.join(' ')}`);
    const res = await exec(argv, { cwd: ctx.projectRoot, env: childEnv, timeoutMs, signal: ctx.signal, onLine: forwardLine });
    const lines = res.stdout.split('\n').filter(Boolean);
    let payload: Record<string, unknown> | undefined;
    const resultLine = lines.filter((l) => l.startsWith('RESULT_JSON:')).pop();
    if (resultLine) {
      try {
        payload = JSON.parse(resultLine.slice('RESULT_JSON:'.length)) as Record<string, unknown>;
      } catch {
        payload = undefined;
      }
    }
    const visible = lines.filter((l) => !l.startsWith('RESULT_JSON:') && !l.startsWith('PROGRESS_JSON:'));
    const tail = (s: string[]) => s.slice(-25).join('\n');
    const credits =
      typeof payload?.credits === 'number'
        ? payload.credits
        : typeof payload?.character_count === 'number'
          ? payload.character_count
          : undefined;
    let failed = payload ? payload.ok === false : res.code !== 0;

    // Auto-master right after assemble_run.py writes `<slug>-raw.mp4`
    if (
      input.step === 'render' &&
      !failed &&
      !extra.includes('--no-master') &&
      fs.existsSync(safeJoin(ctx.projectRoot, rawMp4))
    ) {
      const autoMaster = await exec([PY, path.join(PIPELINE, 'master_run.py'), slug, '--skip-render', '--force'], {
        cwd: ctx.projectRoot,
        env: childEnv,
        timeoutMs: 900_000,
        signal: ctx.signal,
        onLine: forwardLine,
      }).catch(() => null);
      if (autoMaster && autoMaster.code === 0) {
        const mVisible = autoMaster.stdout
          .split('\n')
          .filter((l) => l && !l.startsWith('RESULT_JSON:') && !l.startsWith('PROGRESS_JSON:'));
        extraNotes.push(`auto-mastered out/${slug}/${slug}-final.mp4 (-14 LUFS) & out/${slug}/report.json:\n${mVisible.slice(-14).join('\n')}`);
      } else {
        failed = true;
        extraNotes.push(`master_run.py failed after assembly${autoMaster ? `: ${autoMaster.stderr}` : ''}`);
      }
    }

    // Optional Anime 2D upscaler (Modal Real-CUGAN / Real-ESRGAN) right after mastering when enabled
    const upscalerEnabled = ['1', 'true', 'yes'].includes(String(childEnv.WHATIF_UPSCALER_ENABLED ?? '').trim().toLowerCase());
    const finalMp4 = `out/${slug}/${slug}-final.mp4`;
    if (
      input.step === 'render' &&
      !failed &&
      upscalerEnabled &&
      !extra.includes('--no-upscale') &&
      fs.existsSync(safeJoin(ctx.projectRoot, finalMp4))
    ) {
      const autoUp = await exec([PY, path.join(PIPELINE, 'upscale_run.py'), slug], {
        cwd: ctx.projectRoot,
        env: childEnv,
        timeoutMs: 1_800_000,
        signal: ctx.signal,
        onLine: forwardLine,
      }).catch(() => null);
      if (autoUp && autoUp.code === 0) {
        const uVisible = autoUp.stdout
          .split('\n')
          .filter((l) => l && !l.startsWith('RESULT_JSON:') && !l.startsWith('PROGRESS_JSON:'));
        extraNotes.push(`auto-upscaled out/${slug}/${slug}-final.mp4 (Anime 2D Upscaler, -14 LUFS audio preserved):\n${uVisible.slice(-10).join('\n')}`);
      } else {
        failed = true;
        extraNotes.push(
          `upscale_run.py failed after mastering${autoUp ? `: ${autoUp.stderr || autoUp.stdout.slice(-300)}` : ''}`,
        );
      }
    }

    if (input.step === 'board' && !failed) {
      const sheets = Array.isArray(payload?.sheets) ? (payload!.sheets as string[]) : [];
      if (sheets.length) {
        extraNotes.push(
          `contact sheets (8-9 shots each, readable): ${sheets.join(', ')}. Inspect them visually; open any single still (out/${slug}/board/S<n>.png) if a shot looks wrong.`,
        );
      }
    }

    const artifacts = diffArtifacts(ctx.projectRoot, before);
    if (input.step === 'stills' || input.step === 'board') {
      try {
        const epDir = path.join(ctx.projectRoot, 'src', 'episodes', slug);
        const dirMd = fs.existsSync(path.join(epDir, 'DIRECTOR.md')) ? fs.readFileSync(path.join(epDir, 'DIRECTOR.md'), 'utf8') : '';
        const epNow = readJsonFile(path.join(epDir, 'episode.json'));
        const mismatches = directorSizeMismatches(dirMd, epNow);
        if (mismatches.length) {
          extraNotes.push(
            `DIRECTOR.md / episode.json shot size mismatch (${mismatches.length}): ${mismatches.slice(0, 8).join(', ')}. Keep DIRECTOR.md and episode.json in sync.`,
          );
        }
      } catch {
        /* ignore */
      }
    }

    const requestedShots = input.step === 'stills' ? extra.map((a) => a.trim()).filter((a) => /^S\d+$/i.test(a)) : [];
    const layout =
      input.step === 'stills' || input.step === 'board'
        ? await layoutReport(
            path.join(ctx.projectRoot, 'out', slug, 'board'),
            requestedShots.length ? requestedShots : undefined,
          ).catch(() => '')
        : '';
    let layoutFailure = '';
    if (layout.includes('\nEMPTY:')) {
      failed = true;
      const failedMatch = /FAILED_SHOTS:\s*([^\n]+)/.exec(layout);
      const failedArgs = failedMatch
        ? failedMatch[1]!.split(',').map((s) => s.trim()).filter(Boolean)
        : [];
      const rerunHint = failedArgs.length
        ? `whatif_run { episode: "${slug}", step: "stills", args: ${JSON.stringify(failedArgs)} } then { episode: "${slug}", step: "board" }`
        : `whatif_run { episode: "${slug}", step: "stills" }`;
      layoutFailure = `One or more generated stills failed the layout/ink check (see EMPTY above): adjust their prompts via edit_shot and regenerate with ${rerunHint}.`;
    }

    let planWarning = '';
    if (input.step === 'plan' && !failed) {
      const tl = readJsonFile(path.join(ctx.projectRoot, 'src', 'episodes', slug, 'timeline.json'));
      const wanted = new Set<string>(((episodeEdit(ctx.projectRoot, slug) ?? []) as any[]).map((e) => String(e.shot ?? '')));
      const segs = (tl?.segments ?? []) as any[];
      const got = segs.length;
      if (wanted.size > 1 && got < wanted.size) {
        failed = true;
        planWarning =
          `PLAN COLLAPSED: ${got} segment(s) for ${wanted.size} shots. plan_run.py times each shot from an anchor in ` +
          `episode.json.edit matched against words.json.\nAnchor candidates:\n${anchorHints(ctx.projectRoot, slug)}`;
      } else {
        const pace = paceProblem(ctx.projectRoot, slug);
        if (pace) {
          failed = true;
          planWarning = pace;
        }
      }
    }

    const payloadWithoutLog = payload ? Object.fromEntries(Object.entries(payload).filter(([k]) => k !== 'log')) : undefined;
    return {
      ok: !failed,
      summary: `$ ${argv.join(' ')}\nexit ${res.code}${res.timedOut ? ' (timeout)' : ''}\n${tail(visible)}${res.stderr ? `\n[stderr]\n${tail(res.stderr.split('\n').filter(Boolean))}` : ''}${
        payloadWithoutLog ? `\n\nresult: ${JSON.stringify(payloadWithoutLog).slice(0, 1200)}` : ''
      }${artifacts.length ? `\n\n${artifacts.length} new file(s): ${artifacts.map((a) => a.path).join(', ')}` : ''}${
        layout ? `\n\n${layout}` : ''
      }${layoutFailure ? `\n\n${layoutFailure}` : ''}${extraNotes.length ? `\n\n${extraNotes.join('\n\n')}` : ''}${planWarning ? `\n\n${planWarning}` : ''}`,
      artifacts,
      credits,
      data: { step: input.step, code: res.code, result: payload },
    };
  },
};

const status: Tool = {
  name: 'whatif_status',
  description: 'Where the diffusion episode stands: script, voice, words, timeline, director, bible, refs, prompts, stills, board, clips, render, final, report, upscale.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: { episode: { type: 'string' } },
    required: ['episode'],
    additionalProperties: false,
  },
  async run(input: { episode: string }, ctx) {
    const slug = String(input.episode ?? '').trim();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(slug)) throw new Error(`invalid episode slug "${slug}"`);
    const dir = safeJoin(ctx.projectRoot, `src/episodes/${slug}`);
    const out = safeJoin(ctx.projectRoot, `out/${slug}`);
    const pub = safeJoin(ctx.projectRoot, `public/${slug}`);
    const has = (p: string) => fs.existsSync(p);
    const read = (p: string) => (has(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null);
    const ep = read(path.join(dir, 'episode.json'));
    const tl = read(path.join(dir, 'timeline.json'));
    const report = read(path.join(out, 'report.json'));
    const upscaleInfo = read(path.join(out, 'upscale.json'));
    const gflowProj = read(path.join(out, 'gflow-project.json'));
    const refsDir = path.join(out, 'refs');
    const boardDir = path.join(out, 'board');
    const clipsDir = path.join(out, 'clips');
    const refs = has(refsDir) ? fs.readdirSync(refsDir).filter((f) => f.endsWith('.png')) : [];
    const stills = has(boardDir) ? fs.readdirSync(boardDir).filter((f) => /^S\d+\.png$/i.test(f)) : [];
    const clips = has(clipsDir) ? fs.readdirSync(clipsDir).filter((f) => /^S\d+\.mp4$/i.test(f)) : [];
    const bundle = readEpisodeShots(dir);

    const steps = {
      script: Boolean(ep && !JSON.stringify(ep.script ?? '').includes('TODO')),
      words: has(path.join(dir, 'words.json')),
      timeline: Boolean(tl && (tl.segments?.length ?? 0) > 0 && (tl.shots?.length ?? 0) > 0 && String(tl.title ?? '') !== 'PLACEHOLDER'),
      voice: has(path.join(pub, 'voice.wav')) || has(path.join(pub, 'voice.mp3')),
      director: has(path.join(dir, 'DIRECTOR.md')),
      bible: Boolean(bundle.bible),
      refs: refs.length,
      prompts: bundle.isPlaceholder ? 0 : bundle.exportedShotIds.length,
      stills: stills.length,
      board: has(path.join(out, 'storyboard.jpg')),
      clips: clips.length,
      sfx: has(path.join(pub, 'sfx.wav')),
      render: has(path.join(out, `${input.episode}-raw.mp4`)),
      final: has(path.join(out, `${input.episode}-final.mp4`)),
      report: report ? { seconds: report.seconds ?? report.duration, lufs: report.lufs, credits: report.credits } : null,
      upscale: upscaleInfo
        ? {
            width: upscaleInfo.width,
            height: upscaleInfo.height,
            scale: upscaleInfo.output_scale,
            model: upscaleInfo.model,
            target: upscaleInfo.target,
          }
        : null,
      gflowProject: gflowProj
        ? {
            projectId: gflowProj.project_id,
            title: gflowProj.title,
            url: gflowProj.url,
          }
        : null,
    };
    return {
      ok: true,
      summary: `${input.episode}: ${JSON.stringify(steps, null, 1)}`,
      data: steps,
      artifacts: [],
    };
  },
};

export const whatifTools: Tool[] = [newProject, runStep, status];
