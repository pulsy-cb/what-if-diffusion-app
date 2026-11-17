import fs from 'node:fs';
import path from 'node:path';

/** Sort shot identifiers numerically (`S1`, `S2`, ..., `S10`, `S66`). */
export function sortShotIds(ids: Iterable<string>): string[] {
  return [...new Set(ids)].sort((a, b) => {
    const na = Number(/^S(\d+)$/i.exec(a)?.[1] ?? Number.MAX_SAFE_INTEGER);
    const nb = Number(/^S(\d+)$/i.exec(b)?.[1] ?? Number.MAX_SAFE_INTEGER);
    if (na !== nb) return na - nb;
    return a.localeCompare(b, 'en', { numeric: true });
  });
}

/** Extract unique canonical `S<N>` shot IDs from any string (e.g. `"S1, S7, s42"` -> `['S1', 'S7', 'S42']`). */
export function extractShotIds(text: string): string[] {
  return sortShotIds((String(text ?? '').match(/\bS\d+\b/gi) ?? []).map((s) => s.toUpperCase()));
}

export type BibleCharacter = {
  id: string;
  name: string;
  role?: string;
  appearance: string;
  refPrompt?: string;
};

export type BiblePlace = {
  id: string;
  name: string;
  recipe?: string;
  description: string;
  refPrompt?: string;
};

export type EpisodeBible = {
  stylePrompt: string;
  negativePrompt?: string;
  characters: BibleCharacter[];
  places: BiblePlace[];
};

export type OnScreenTextSpec = {
  kind: 'phone' | 'document' | 'sign' | 'board' | 'screen';
  text: string;
  lang?: 'fr' | 'en';
} | null;

export const VALID_SHOT_ANGLES = ['level', 'low', 'high'] as const;
export const VALID_SHOT_VIEWS = ['front', 'reverse', 'left', 'right', '3/4-left', '3/4-right'] as const;
export const VALID_CAMERA_MOVES = ['still', 'push', 'pull', 'pan', 'track', 'whip'] as const;
export const VALID_ONSCREEN_KINDS = ['phone', 'document', 'sign', 'board', 'screen'] as const;

export type DiffusionShotSpec = {
  id: string;
  place: string;
  characters: string[];
  size?: string;
  angle?: (typeof VALID_SHOT_ANGLES)[number];
  view?: (typeof VALID_SHOT_VIEWS)[number];
  cameraMove?: (typeof VALID_CAMERA_MOVES)[number];
  imagePrompt: string;
  imageModel?: string;
  negativePrompt?: string;
  onScreenText?: OnScreenTextSpec;
  videoPrompt: string;
  sfxHint: string;
  seed?: number;
};

const UPPER_ID_RE = /^[A-Z][A-Z0-9_]*$/;
const SHOT_ID_RE = /^S\d+$/;

const NEGATIVE_CLAUSE_RE =
  /\b(?:strictly\s+no|absolutely\s+no|no|without|avoid|forbid(?:den)?|never(?:\s+add)?|zero|exclude|excluding|free\s+of)\b(?:(?!\b(?:but|with|only|except|featuring|showing|including|use|keep|render|maintain)\b)[^.;!?\n—])*/gi;

/** Strip explicit negative clauses ("no music", "without dialogue", "avoid title cards") before checking positive requests. */
export function stripNegativeClauses(text: string): string {
  return String(text ?? '').replace(NEGATIVE_CLAUSE_RE, ' ');
}

const TITLE_REQUEST_RE =
  /\b(title\s*card|opening\s+title|episode\s+title|hook[\s-]+title|title\s+screen|subtitles?|caption\s+bar)\b/i;
const MUSIC_REQUEST_RE =
  /\b(background\s+music|soundtrack|bgm|musical\s+score|music\s+bed|cinematic\s+score|orchestral\s+music|lo-?fi\s+beat|pop\s+song|music\s+track|piano\s+melody)\b/i;
const DIALOGUE_REQUEST_RE =
  /\b(dialogue|voiceover|voice[\s-]over|spoken\s+lines?|speaking\s+aloud|narrator\s+(?:says|speaks)|characters?\s+(?:speaking|talking)|spoken\s+words?|lip[\s-]*sync(?:ed)?(?:\s+speech)?|questioning\s+\w+|interrogating\s+\w+|talking\s+to\s+\w+|shouting\s+words|mouth\s+moving\s+in\s+speech)\b/i;

export function hasPositiveTitleRequest(text: string): boolean {
  return TITLE_REQUEST_RE.test(stripNegativeClauses(text));
}

export function hasPositiveMusicRequest(text: string): boolean {
  return MUSIC_REQUEST_RE.test(stripNegativeClauses(text));
}

export function hasPositiveDialogueRequest(text: string): boolean {
  return DIALOGUE_REQUEST_RE.test(stripNegativeClauses(text));
}

export function validateBible(raw: unknown): { bible: EpisodeBible | null; errors: string[] } {
  const errors: string[] = [];
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { bible: null, errors: ['bible.json must be a JSON object'] };
  }
  const obj = raw as Record<string, unknown>;
  const stylePrompt = String(obj.stylePrompt ?? '').trim();
  if (stylePrompt.length < 20) {
    errors.push('bible.stylePrompt is too short (at least 20 chars describing the 2D visual style)');
  } else if (hasPositiveTitleRequest(stylePrompt)) {
    errors.push('bible.stylePrompt must NOT request title cards or subtitles (the episode uses no titles/subtitles)');
  }

  const negativePrompt = typeof obj.negativePrompt === 'string' ? obj.negativePrompt.trim() : undefined;

  const rawPlaces = Array.isArray(obj.places) ? obj.places : [];
  if (!rawPlaces.length) {
    errors.push('bible.places must contain at least 1 place object ({ id, name, description })');
  }
  const places: BiblePlace[] = [];
  const placeIds = new Set<string>();
  for (let i = 0; i < rawPlaces.length; i++) {
    const item = rawPlaces[i];
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      errors.push(`bible.places[${i}] must be an object`);
      continue;
    }
    const p = item as Record<string, unknown>;
    const id = String(p.id ?? '').trim();
    const name = String(p.name ?? id).trim();
    const description = String(p.description ?? '').trim();
    const recipe = typeof p.recipe === 'string' && p.recipe.trim() ? p.recipe.trim() : undefined;
    const refPrompt = typeof p.refPrompt === 'string' && p.refPrompt.trim() ? p.refPrompt.trim() : undefined;
    if (!UPPER_ID_RE.test(id)) {
      errors.push(`bible.places[${i}].id must be UPPERCASE_SLUG (got "${id}")`);
    } else if (placeIds.has(id)) {
      errors.push(`bible.places[${i}].id "${id}" is duplicated`);
    } else {
      placeIds.add(id);
    }
    if (description.length < 20) {
      errors.push(`bible.places[${i}] (${id || i}).description is too short (at least 20 chars)`);
    }
    places.push({ id, name: name || id, ...(recipe ? { recipe } : {}), description, ...(refPrompt ? { refPrompt } : {}) });
  }

  const rawChars = Array.isArray(obj.characters) ? obj.characters : [];
  const characters: BibleCharacter[] = [];
  const charIds = new Set<string>();
  for (let i = 0; i < rawChars.length; i++) {
    const item = rawChars[i];
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      errors.push(`bible.characters[${i}] must be an object`);
      continue;
    }
    const c = item as Record<string, unknown>;
    const id = String(c.id ?? '').trim();
    const name = String(c.name ?? id).trim();
    const role = typeof c.role === 'string' && c.role.trim() ? c.role.trim() : undefined;
    const appearance = String(c.appearance ?? '').trim();
    const refPrompt = typeof c.refPrompt === 'string' && c.refPrompt.trim() ? c.refPrompt.trim() : undefined;
    if (!UPPER_ID_RE.test(id)) {
      errors.push(`bible.characters[${i}].id must be UPPERCASE_SLUG (got "${id}")`);
    } else if (charIds.has(id)) {
      errors.push(`bible.characters[${i}].id "${id}" is duplicated`);
    } else {
      charIds.add(id);
    }
    if (appearance.length < 20) {
      errors.push(`bible.characters[${i}] (${id || i}).appearance is too short (at least 20 chars describing hair, face, clothes, build)`);
    }
    characters.push({ id, name: name || id, ...(role ? { role } : {}), appearance, ...(refPrompt ? { refPrompt } : {}) });
  }

  if (errors.length) return { bible: null, errors };
  return {
    bible: {
      stylePrompt,
      ...(negativePrompt ? { negativePrompt } : {}),
      characters,
      places,
    },
    errors: [],
  };
}

export function validateShotSpec(raw: unknown, expectedId?: string): { shot: DiffusionShotSpec | null; errors: string[] } {
  const errors: string[] = [];
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { shot: null, errors: ['shot spec must be a JSON object'] };
  }
  const obj = raw as Record<string, unknown>;
  const id = String(obj.id ?? expectedId ?? '').trim().toUpperCase();
  if (!SHOT_ID_RE.test(id)) {
    errors.push(`shot.id must be "S<N>" (got "${String(obj.id ?? '')}")`);
  } else if (expectedId && id !== expectedId.toUpperCase()) {
    errors.push(`shot.id "${id}" does not match expected "${expectedId.toUpperCase()}"`);
  }

  const place = String(obj.place ?? '').trim();
  if (!UPPER_ID_RE.test(place)) {
    errors.push(`${id || 'shot'}.place must be an UPPERCASE place id from bible.json (got "${place}")`);
  }

  const rawChars = obj.characters;
  const characters: string[] = [];
  if (!Array.isArray(rawChars)) {
    errors.push(`${id || 'shot'}.characters must be an array of character IDs from bible.json (use [] when no character is visible)`);
  } else {
    const seen = new Set<string>();
    for (const c of rawChars) {
      const cid = String(c ?? '').trim();
      if (!UPPER_ID_RE.test(cid)) {
        errors.push(`${id || 'shot'}.characters entry "${cid}" must be an UPPERCASE character ID`);
      } else if (seen.has(cid)) {
        errors.push(`${id || 'shot'}.characters contains duplicate ID "${cid}"`);
      } else {
        seen.add(cid);
        characters.push(cid);
      }
    }
  }

  const size = typeof obj.size === 'string' && obj.size.trim() ? obj.size.trim() : undefined;
  if (size && /\b(title|hook-title|title-card)\b/i.test(size)) {
    errors.push(`${id || 'shot'}.size "${size}" is forbidden: title shots are removed in this studio`);
  }

  let angle: DiffusionShotSpec['angle'];
  if (obj.angle !== undefined && obj.angle !== null && String(obj.angle).trim() !== '') {
    const a = String(obj.angle).trim();
    if (!VALID_SHOT_ANGLES.includes(a as any)) {
      errors.push(`${id || 'shot'}.angle must be one of ${VALID_SHOT_ANGLES.join(', ')}`);
    } else {
      angle = a as DiffusionShotSpec['angle'];
    }
  }

  let view: DiffusionShotSpec['view'];
  if (obj.view !== undefined && obj.view !== null && String(obj.view).trim() !== '') {
    const v = String(obj.view).trim();
    if (!VALID_SHOT_VIEWS.includes(v as any)) {
      errors.push(`${id || 'shot'}.view must be one of ${VALID_SHOT_VIEWS.join(', ')}`);
    } else {
      view = v as DiffusionShotSpec['view'];
    }
  }

  let cameraMove: DiffusionShotSpec['cameraMove'];
  if (obj.cameraMove !== undefined && obj.cameraMove !== null && String(obj.cameraMove).trim() !== '') {
    const m = String(obj.cameraMove).trim();
    if (!VALID_CAMERA_MOVES.includes(m as any)) {
      errors.push(`${id || 'shot'}.cameraMove must be one of ${VALID_CAMERA_MOVES.join(', ')}`);
    } else {
      cameraMove = m as DiffusionShotSpec['cameraMove'];
    }
  }

  const imagePrompt = String(obj.imagePrompt ?? '').trim();
  if (imagePrompt.length < 30) {
    errors.push(`${id || 'shot'}.imagePrompt is too short (${imagePrompt.length} chars, need at least 30 chars in English for Nano Banana 2 Lite)`);
  } else if (hasPositiveTitleRequest(imagePrompt)) {
    errors.push(`${id || 'shot'}.imagePrompt requests a title card or subtitles, which are forbidden (only in-world text via onScreenText is allowed)`);
  }

  const negativePrompt = typeof obj.negativePrompt === 'string' && obj.negativePrompt.trim() ? obj.negativePrompt.trim() : undefined;
  let imageModel: string | undefined;
  if (obj.imageModel !== undefined && obj.imageModel !== null && String(obj.imageModel).trim() !== '') {
    const im = String(obj.imageModel).trim();
    if (/[\s\x00-\x1f]/.test(im)) {
      errors.push(`${id || 'shot'}.imageModel must not contain spaces or control characters (got "${im}")`);
    } else {
      imageModel = im;
    }
  }

  let onScreenText: OnScreenTextSpec | undefined;
  if (obj.onScreenText !== undefined) {
    if (obj.onScreenText === null) {
      onScreenText = null;
    } else if (typeof obj.onScreenText !== 'object' || Array.isArray(obj.onScreenText)) {
      errors.push(`${id || 'shot'}.onScreenText must be null or { kind, text, lang }`);
    } else {
      const ost = obj.onScreenText as Record<string, unknown>;
      const kind = String(ost.kind ?? '').trim();
      const text = String(ost.text ?? '').trim();
      const lang = ost.lang !== undefined && ost.lang !== null ? String(ost.lang).trim().toLowerCase() : undefined;
      if (!text || /^(?:none|null|n\/a|nil)$/i.test(text) || /^none$/i.test(kind)) {
        onScreenText = null;
      } else {
        if (!VALID_ONSCREEN_KINDS.includes(kind as any)) {
          errors.push(`${id || 'shot'}.onScreenText.kind must be one of ${VALID_ONSCREEN_KINDS.join(', ')}`);
        }
        if (hasPositiveTitleRequest(text)) {
          errors.push(`${id || 'shot'}.onScreenText.text must not be a title card or subtitle`);
        }
        if (lang !== undefined && lang !== '' && lang !== 'fr' && lang !== 'en') {
          errors.push(`${id || 'shot'}.onScreenText.lang must be "fr" or "en"`);
        }
        if (!errors.length) {
          onScreenText = {
            kind: kind as NonNullable<OnScreenTextSpec>['kind'],
            text,
            ...(lang === 'fr' || lang === 'en' ? { lang } : {}),
          };
        }
      }
    }
  }

  const videoPrompt = String(obj.videoPrompt ?? '').trim();
  if (videoPrompt.length < 20) {
    errors.push(`${id || 'shot'}.videoPrompt is too short (${videoPrompt.length} chars, need at least 20 chars describing motion for Gemini Omni Flash 2.1)`);
  } else {
    if (hasPositiveMusicRequest(videoPrompt)) {
      errors.push(`${id || 'shot'}.videoPrompt must NOT request music or a soundtrack (Omni Flash generates SFX + ambience only)`);
    }
    if (hasPositiveDialogueRequest(videoPrompt)) {
      errors.push(`${id || 'shot'}.videoPrompt must NOT request spoken dialogue or voiceover (narration comes from :8005)`);
    }
  }

  const sfxHint = String(obj.sfxHint ?? '').trim();
  if (sfxHint.length < 5) {
    errors.push(`${id || 'shot'}.sfxHint is required (at least 5 chars describing physical foley & ambient SFX for Omni Flash 2.1, e.g. "wooden chair scrape, quiet room tone")`);
  } else {
    if (hasPositiveMusicRequest(sfxHint)) {
      errors.push(`${id || 'shot'}.sfxHint must NOT request music or a musical score (SFX and ambience only)`);
    }
    if (hasPositiveDialogueRequest(sfxHint)) {
      errors.push(`${id || 'shot'}.sfxHint must NOT request dialogue or spoken voice`);
    }
  }

  let seed: number | undefined;
  if (obj.seed !== undefined && obj.seed !== null && obj.seed !== '') {
    const n = Number(obj.seed);
    if (!Number.isInteger(n) || n < 0) {
      errors.push(`${id || 'shot'}.seed must be a non-negative integer`);
    } else if (n > 0) {
      seed = n;
    }
  }

  if (errors.length) return { shot: null, errors };
  return {
    shot: {
      id,
      place,
      characters,
      ...(size ? { size } : {}),
      ...(angle ? { angle } : {}),
      ...(view ? { view } : {}),
      ...(cameraMove ? { cameraMove } : {}),
      imagePrompt,
      ...(imageModel ? { imageModel } : {}),
      ...(negativePrompt ? { negativePrompt } : {}),
      ...(onScreenText !== undefined ? { onScreenText } : {}),
      videoPrompt,
      sfxHint,
      ...(seed !== undefined ? { seed } : {}),
    },
    errors: [],
  };
}

function safeMtimeMs(file: string): number {
  try {
    return fs.statSync(file).mtimeMs;
  } catch {
    return 0;
  }
}

export function readEpisodeBible(epDir: string): {
  exists: boolean;
  bible: EpisodeBible | null;
  raw: string;
  mtimeMs: number;
  errors: string[];
} {
  const biblePath = path.join(epDir, 'bible.json');
  if (!fs.existsSync(biblePath)) {
    return { exists: false, bible: null, raw: '', mtimeMs: 0, errors: ['bible.json is missing'] };
  }
  const mtimeMs = safeMtimeMs(biblePath);
  const raw = fs.readFileSync(biblePath, 'utf8');
  try {
    const parsed = JSON.parse(raw);
    const { bible, errors } = validateBible(parsed);
    return { exists: true, bible, raw, mtimeMs, errors };
  } catch (err) {
    return {
      exists: true,
      bible: null,
      raw,
      mtimeMs,
      errors: [`bible.json is invalid JSON: ${(err as Error).message}`],
    };
  }
}

export type EpisodeShotsBundle = {
  exists: boolean;
  isPlaceholder: boolean;
  bible: EpisodeBible | null;
  bibleRaw: string;
  bibleMtimeMs: number;
  shots: Map<string, DiffusionShotSpec>;
  shotFiles: Map<string, string>;
  shotBodies: Record<string, string>;
  exportedShotIds: string[];
  combinedSource: string;
  latestMtimeMs: number;
  shotMtimesMs: Map<string, number>;
};

/**
 * Read an episode's diffusion Bible (`bible.json`) and modular per-shot prompt specs (`shots/S<N>.json`).
 */
export function readEpisodeShots(epDir: string): EpisodeShotsBundle {
  const shotsDir = path.join(epDir, 'shots');
  const bibleRes = readEpisodeBible(epDir);
  const shots = new Map<string, DiffusionShotSpec>();
  const shotFiles = new Map<string, string>();
  const shotBodies: Record<string, string> = {};
  const shotMtimesMs = new Map<string, number>();
  let latestMtimeMs = bibleRes.mtimeMs;

  if (fs.existsSync(shotsDir) && fs.statSync(shotsDir).isDirectory()) {
    const entries = fs
      .readdirSync(shotsDir)
      .filter((f) => /^S\d+\.json$/i.test(f))
      .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
    for (const fname of entries) {
      const id = fname.replace(/\.json$/i, '').toUpperCase();
      const fpath = path.join(shotsDir, fname);
      const raw = fs.readFileSync(fpath, 'utf8');
      const mtime = safeMtimeMs(fpath);
      if (mtime > latestMtimeMs) latestMtimeMs = mtime;
      shotMtimesMs.set(id, mtime);
      shotFiles.set(id, raw);
      shotBodies[id] = raw.trim();
      try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          shots.set(id, parsed as DiffusionShotSpec);
        }
      } catch {
        /* invalid JSON handled by audits */
      }
    }
  }

  const exportedShotIds = sortShotIds([...shotFiles.keys()]);
  const isPlaceholder =
    exportedShotIds.length === 0 ||
    (exportedShotIds.length <= 1 && /PLACEHOLDER/i.test(shotFiles.get('S1') ?? ''));

  const combinedParts: string[] = [];
  if (bibleRes.raw.trim()) combinedParts.push(bibleRes.raw.trim());
  for (const id of exportedShotIds) {
    const body = shotBodies[id];
    if (body) combinedParts.push(body);
  }

  return {
    exists: bibleRes.exists || shotFiles.size > 0,
    isPlaceholder,
    bible: bibleRes.bible,
    bibleRaw: bibleRes.raw,
    bibleMtimeMs: bibleRes.mtimeMs,
    shots,
    shotFiles,
    shotBodies,
    exportedShotIds,
    combinedSource: combinedParts.join('\n\n'),
    latestMtimeMs,
    shotMtimesMs,
  };
}
