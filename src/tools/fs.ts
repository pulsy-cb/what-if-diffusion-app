import fs from 'node:fs';
import path from 'node:path';
import type { Tool, ToolResult } from './types.js';
import {
  exportedSymbols,
  globToRegExp,
  isBinaryPath,
  mimeOf,
  readText,
  relOf,
  safeJoin,
  walk,
} from './paths.js';
import {
  VALID_CAMERA_MOVES,
  VALID_ONSCREEN_KINDS,
  VALID_SHOT_ANGLES,
  VALID_SHOT_VIEWS,
  readEpisodeBible,
  readEpisodeShots,
  sortShotIds,
  validateBible,
  validateShotSpec,
  type DiffusionShotSpec,
} from './episode-source.js';

/**
 * Where the agent may write inside a project: its episodes, notes at the root (PROJECT.md…) and scratch files in out/.
 * Everything else (package manifests, scripts, git internals, .env, providers.json) is refused.
 */
const WRITABLE_PATHS = [/^src\/episodes\//, /^[^/]+\.md$/, /^out\//];

function assertWritableProjectPath(projectRoot: string, relPath: string): { abs: string; rel: string } {
  const raw = relPath.replace(/\\/g, '/').replace(/^\/+/, '');
  if (/\(exports:|\[compacted/i.test(raw)) {
    throw new Error(`writing to "${raw}" is refused: pass the clean file path without annotations`);
  }
  const abs = safeJoin(projectRoot, relPath);
  const norm = relOf(projectRoot, abs);
  if (
    !norm ||
    norm === 'package.json' ||
    norm === 'package-lock.json' ||
    norm.startsWith('scripts/') ||
    norm.startsWith('pipeline/') ||
    norm.startsWith('.git/') ||
    /(^|\/)\.env(\.|$)/i.test(norm) ||
    /(^|\/)providers\.json$/i.test(norm)
  ) {
    throw new Error(`writing to "${norm || raw}" is refused by security policy`);
  }
  const shotMatch = /^src\/episodes\/[^/]+\/shots\/([^/]+)$/i.exec(norm);
  if (shotMatch) {
    const file = shotMatch[1]!;
    if (!/^S\d+\.json$/i.test(file)) {
      throw new Error(`shot file "${norm}" must be a JSON prompt spec named "S<N>.json" (not TSX or arbitrary files)`);
    }
    const stem = file.replace(/\.json$/i, '');
    if (stem !== stem.toUpperCase() || !file.endsWith('.json')) {
      throw new Error(`shot file "${norm}" must use uppercase "${stem.toUpperCase()}.json"`);
    }
  }
  if (WRITABLE_PATHS.some((re) => re.test(norm))) return { abs, rel: norm };
  throw new Error(
    `writing to "${norm}" is refused: write inside src/episodes/<slug>/ (episode.json, DIRECTOR.md, bible.json, shots/S<N>.json…), a root note like PROJECT.md, or out/.`,
  );
}

function assertNotCompactedPlaceholder(text: string, toolName: string): void {
  if (/\[compacted:\s*\d+\s*lines/i.test(text)) {
    throw new Error(
      `${toolName} refused: content contains a "[compacted: ...]" placeholder instead of real data. Write the complete JSON or Markdown content.`,
    );
  }
}

/** Validate bible.json or shots/S<N>.json when written or edited directly via write_file / edit_file. */
function validateStructuredEpisodeJson(projectRoot: string, normPath: string, content: string): string {
  if (/^src\/episodes\/[^/]+\/bible\.json$/.test(normPath)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(content);
    } catch (e) {
      throw new Error(`invalid JSON in ${normPath}: ${(e as Error).message}`);
    }
    const { bible, errors } = validateBible(parsed);
    if (!bible || errors.length) {
      throw new Error(`${normPath} failed validation:\n- ${errors.join('\n- ')}`);
    }
    return JSON.stringify(bible, null, 2) + '\n';
  }
  const shotMatch = /^src\/episodes\/([^/]+)\/shots\/(S\d+)\.json$/.exec(normPath);
  if (shotMatch) {
    const slug = shotMatch[1]!;
    const id = shotMatch[2]!;
    let parsed: unknown;
    try {
      parsed = JSON.parse(content);
    } catch (e) {
      throw new Error(`invalid JSON in ${normPath}: ${(e as Error).message}`);
    }
    const { shot, errors } = validateShotSpec(parsed, id);
    if (!shot || errors.length) {
      throw new Error(`${normPath} failed validation:\n- ${errors.join('\n- ')}`);
    }
    const bibleErrs = checkShotAgainstBible(path.join(projectRoot, 'src', 'episodes', slug), shot);
    if (bibleErrs.length) {
      throw new Error(`${normPath} does not match bible.json:\n- ${bibleErrs.join('\n- ')}`);
    }
    return JSON.stringify(shot, null, 2) + '\n';
  }
  return content;
}

function checkShotAgainstBible(epDir: string, shot: DiffusionShotSpec): string[] {
  const { exists, bible, errors } = readEpisodeBible(epDir);
  if (!exists) return [];
  if (!bible || errors.length) {
    return [`bible.json is invalid (${errors[0] ?? 'parse error'}) — fix it with write_bible first`];
  }
  const errs: string[] = [];
  const placeIds = new Set(bible.places.map((p) => p.id));
  const charIds = new Set(bible.characters.map((c) => c.id));
  if (!placeIds.has(shot.place)) {
    errs.push(`${shot.id}.place "${shot.place}" is not in bible.json.places (available: ${[...placeIds].join(', ')})`);
  }
  for (const cid of shot.characters) {
    if (!charIds.has(cid)) {
      errs.push(`${shot.id}.characters "${cid}" is not in bible.json.characters (available: ${[...charIds].join(', ') || 'none'})`);
    }
  }
  return errs;
}

const repoMap: Tool = {
  name: 'repo_map',
  description:
    'Overview of the project: files (excluding node_modules/out/.git), sizes and exported symbols. Call this first, and again after big changes.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: {
      maxFiles: { type: 'integer', description: 'Cap on listed files (default 300).' },
      filter: { type: 'string', description: 'Optional substring the path must contain.' },
    },
  },
  async run(input: { maxFiles?: number; filter?: string }, ctx) {
    const files = walk(ctx.projectRoot, { maxFiles: 5000 });
    const selected = (input.filter ? files.filter((f) => f.includes(input.filter!)) : files).slice(
      0,
      Math.max(1, input.maxFiles ?? 300),
    );
    const lines: string[] = [];
    let chars = 0;
    for (const rel of selected) {
      const abs = path.join(ctx.projectRoot, rel);
      let size = 0;
      try {
        size = fs.statSync(abs).size;
      } catch {
        continue;
      }
      let symbols: string[] = [];
      if (/\.(ts|tsx|js|mjs|py)$/.test(rel) && size < 120_000) {
        const src = readText(ctx.projectRoot, rel);
        if (src) symbols = exportedSymbols(src);
      }
      const line = `${rel}${symbols.length ? ` — ${symbols.join(', ')}` : ''} (${size}b)`;
      if (chars + line.length > 60_000) {
        lines.push('… (repo map truncated)');
        break;
      }
      chars += line.length + 1;
      lines.push(line);
    }
    const episodes = files.filter((f) => /^src\/episodes\/[^/]+\/episode\.json$/.test(f)).map((f) => f.split('/')[2]);
    return {
      ok: true,
      summary: `${selected.length} files. Episodes: ${episodes.length ? episodes.join(', ') : 'none yet'}.\n${lines.join('\n')}`,
      data: { files: selected, episodes },
    };
  },
};

const readFile: Tool = {
  name: 'read_file',
  description:
    'Read a text file with line numbers. It returns the WHOLE file (up to 6000 lines): do not paginate it ' +
    'into windows, call it once. start/end exist only for files above that limit.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Project-relative path.' },
      start: { type: 'integer', description: 'First line (1-based).' },
      end: { type: 'integer', description: 'Last line, inclusive.' },
      limit: { type: 'integer', description: 'Optional max number of lines to read.' },
    },
    required: ['path'],
    additionalProperties: false,
  },
  async run(input: { path: string; start?: number; end?: number; limit?: number }, ctx) {
    const normRel = relOf(ctx.projectRoot, safeJoin(ctx.projectRoot, input.path));
    if (/(^|\/)\.env(\.|$)/i.test(normRel) || /(^|\/)providers\.json$/i.test(normRel)) {
      throw new Error(`reading "${normRel}" is refused by security policy`);
    }
    const src = readText(ctx.projectRoot, input.path, 2_000_000);
    if (src === null) throw new Error(`cannot read ${input.path} (missing, binary or > 2MB)`);
    const all = src.split('\n');
    const start = Math.max(1, input.start ?? 1);
    const end = Math.min(all.length, input.end ?? (input.limit ? start + input.limit - 1 : start + 5_999));
    const slice = all.slice(start - 1, end);
    let body = '';
    for (let i = 0; i < slice.length; i++) body += `${String(start + i).padStart(5)}| ${slice[i]}\n`;
    const truncated = start > 1 || end < all.length;
    return {
      ok: true,
      summary: `${input.path} lines ${start}-${end} of ${all.length}${truncated ? ' (partial)' : ''}\n${body}`,
      data: { path: input.path, lines: all.length },
    };
  },
};

const writeFile: Tool = {
  name: 'write_file',
  description: 'Create a file, replace it entirely, or append to it (when append: true).',
  policy: 'auto',
  mutating: true,
  schema: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      content: { type: 'string' },
      append: { type: 'boolean', description: 'If true, append content to the end of the file instead of overwriting.' },
    },
    required: ['path', 'content'],
    additionalProperties: false,
  },
  async run(input: { path: string; content: string; append?: boolean }, ctx): Promise<ToolResult> {
    if (isBinaryPath(input.path)) throw new Error('write_file refuses binary extensions');
    const { abs, rel: normPath } = assertWritableProjectPath(ctx.projectRoot, input.path);
    assertNotCompactedPlaceholder(input.content, 'write_file');
    const existed = fs.existsSync(abs);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    if (input.append && existed) {
      const prev = fs.readFileSync(abs, 'utf8');
      const sep = prev.length > 0 && !prev.endsWith('\n') ? '\n' : '';
      const combined = validateStructuredEpisodeJson(ctx.projectRoot, normPath, prev + sep + input.content);
      fs.writeFileSync(abs, combined, 'utf8');
      const totalLines = combined.split('\n').length;
      return {
        ok: true,
        summary: `appended to ${normPath} (+${input.content.split('\n').length} lines, total ${totalLines} lines)`,
        data: { path: normPath, created: false, appended: true, lines: totalLines },
      };
    }
    const content = validateStructuredEpisodeJson(ctx.projectRoot, normPath, input.content);
    fs.writeFileSync(abs, content, 'utf8');
    const lines = content.split('\n').length;
    return {
      ok: true,
      summary: `${existed ? 'replaced' : 'created'} ${normPath} (${lines} lines, ${Buffer.byteLength(content)} bytes)`,
      data: { path: normPath, created: !existed, lines },
    };
  },
};

/** A whitespace-tolerant match: same text once leading indentation and blank lines are ignored. */
function looseMatch(
  haystack: string,
  needle: string,
): { start: number; end: number; replace: (text: string) => string } | null {
  const want = needle
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (!want.length) return null;
  const lines = haystack.split('\n');
  const offsets: number[] = [];
  let pos = 0;
  for (const line of lines) {
    offsets.push(pos);
    pos += line.length + 1;
  }

  if (want.length === 1) {
    const hits: { at: number; end: number }[] = [];
    lines.forEach((line, i) => {
      const at = line.indexOf(want[0]!);
      if (at >= 0) hits.push({ at: offsets[i]! + at, end: offsets[i]! + at + want[0]!.length });
    });
    if (hits.length !== 1) return null;
    return {
      start: hits[0]!.at,
      end: hits[0]!.end,
      replace: (text: string) => text.split('\n').map((l) => l.trim()).join('\n'),
    };
  }

  let found: { start: number; end: number; indent: string } | null = null;
  let hits = 0;
  for (let i = 0; i + want.length <= lines.length; i++) {
    let ok = true;
    for (let k = 0; k < want.length; k++) {
      if (lines[i + k]!.trim() !== want[k]) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    hits++;
    const indent = /^\s*/.exec(lines[i]!)?.[0] ?? '';
    const last = lines[i + want.length - 1]!;
    found = { start: offsets[i]!, end: offsets[i + want.length - 1]! + last.length, indent };
  }
  if (hits !== 1 || !found) return null;
  const { start, end, indent } = found;
  return {
    start,
    end,
    replace: (text: string) =>
      text
        .split('\n')
        .map((l, idx) => (idx === 0 || l.trim() === '' ? l : indent + l.replace(/^\s+/, '')))
        .join('\n'),
  };
}

/** The block of the file that most resembles the requested oldText, to hand back as a hint. */
function nearestBlock(haystack: string, needle: string): { line: number; endLine: number; text: string } | null {
  const want = needle.trim().split('\n')[0]!.trim();
  if (want.length < 4) return null;
  const lines = haystack.split('\n');
  let best = { index: -1, score: 0 };
  lines.forEach((line, i) => {
    const t = line.trim();
    if (!t) return;
    const score = commonRun(t, want);
    if (score > best.score) best = { index: i, score };
  });
  if (best.index < 0 || best.score < Math.min(6, Math.ceil(want.length / 2))) return null;
  const count = Math.max(1, needle.split('\n').length);
  const from = Math.max(0, best.index - 2);
  const to = Math.min(lines.length, best.index + count + 3);
  const slice = lines.slice(from, to);
  return { line: from + 1, endLine: from + slice.length, text: slice.join('\n') };
}

/** Length of the longest common substring. */
function commonRun(a: string, b: string): number {
  let best = 0;
  const prev = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    let diag = 0;
    for (let j = 1; j <= b.length; j++) {
      const keep = prev[j]!;
      if (a[i - 1] === b[j - 1]) {
        prev[j] = diag + 1;
        if (prev[j]! > best) best = prev[j]!;
      } else prev[j] = 0;
      diag = keep;
    }
  }
  return best;
}

const editFile: Tool = {
  name: 'edit_file',
  description:
    'Targeted text replacement: each edit replaces one unique occurrence of oldText with newText. Fails loudly if oldText is missing or ambiguous. Batch nearby changes in one call.',
  policy: 'auto',
  mutating: true,
  schema: {
    type: 'object',
    properties: {
      path: { type: 'string' },
      edits: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            oldText: { type: 'string', description: 'Exact text to replace, unique in the file.' },
            newText: { type: 'string' },
          },
          required: ['oldText', 'newText'],
          additionalProperties: false,
        },
      },
    },
    required: ['path', 'edits'],
    additionalProperties: false,
  },
  async run(input: { path: string; edits: { oldText: string; newText: string }[] }, ctx) {
    const { abs, rel: normPath } = assertWritableProjectPath(ctx.projectRoot, input.path);
    for (const e of input.edits) assertNotCompactedPlaceholder(e.newText, 'edit_file');
    const src = readText(ctx.projectRoot, normPath, 2_000_000);
    if (src === null) throw new Error(`cannot read ${input.path}`);
    let out = src;
    const applied: number[] = [];
    for (let i = 0; i < input.edits.length; i++) {
      const { oldText, newText } = input.edits[i]!;
      const first = out.indexOf(oldText);
      if (first >= 0) {
        if (out.indexOf(oldText, first + 1) >= 0) {
          const hitLines: number[] = [];
          let cursor = 0;
          while (hitLines.length < 6) {
            const idx = out.indexOf(oldText, cursor);
            if (idx < 0) break;
            hitLines.push(out.slice(0, idx).split('\n').length);
            cursor = idx + Math.max(1, oldText.length);
          }
          throw new Error(
            `edit ${i}: oldText is not unique in ${input.path} (matches at lines ${hitLines.join(', ')} — include surrounding lines to disambiguate).`,
          );
        }
        out = out.slice(0, first) + newText + out.slice(first + oldText.length);
        applied.push(first);
        continue;
      }
      const loose = looseMatch(out, oldText);
      if (loose) {
        out = out.slice(0, loose.start) + loose.replace(newText) + out.slice(loose.end);
        applied.push(loose.start);
        continue;
      }
      const near = nearestBlock(out, oldText);
      throw new Error(
        `edit ${i}: oldText not found in ${input.path}.` +
          (near
            ? ` The closest text is lines ${near.line}-${near.endLine}:\n---\n${near.text}\n---\nCopy the exact lines you want to replace (indentation included).`
            : ' Read the file and copy the text exactly.'),
      );
    }
    const validated = validateStructuredEpisodeJson(ctx.projectRoot, normPath, out);
    fs.writeFileSync(abs, validated, 'utf8');
    return {
      ok: true,
      summary: `${applied.length} edit(s) applied to ${normPath}`,
      data: { path: normPath, edits: applied.length },
    };
  },
};

const glob: Tool = {
  name: 'glob',
  description: 'List project files matching a glob (** and * supported), e.g. "src/episodes/**/shots/*.json".',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: { pattern: { type: 'string' }, limit: { type: 'integer' } },
    required: ['pattern'],
    additionalProperties: false,
  },
  async run(input: { pattern: string; limit?: number }, ctx) {
    const re = globToRegExp(input.pattern);
    const files = walk(ctx.projectRoot, { maxFiles: 6000 }).filter((f) => re.test(f));
    const limit = input.limit ?? 200;
    return {
      ok: true,
      summary: `${files.length} match(es)${files.length > limit ? `, first ${limit}` : ''}\n${files.slice(0, limit).join('\n')}`,
      data: { matches: files },
    };
  },
};

const grep: Tool = {
  name: 'grep',
  description: 'Regex search across project text files, with line numbers.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: {
      pattern: { type: 'string' },
      glob: { type: 'string', description: 'Optional path glob filter.' },
      path: { type: 'string', description: 'Optional file path or glob filter (alias for glob).' },
      max: { type: 'integer' },
    },
    required: ['pattern'],
    additionalProperties: false,
  },
  async run(input: { pattern: string; glob?: string; path?: string; max?: number }, ctx) {
    if (input.pattern.length > 300) throw new Error('grep pattern is too long (max 300 chars)');
    let re: RegExp;
    try {
      re = new RegExp(input.pattern, 'i');
    } catch (e) {
      throw new Error(`invalid regex "${input.pattern}": ${(e as Error).message}`);
    }
    const filterPat = input.glob ?? input.path;
    const globRe = filterPat ? globToRegExp(filterPat) : null;
    const max = Math.max(1, Math.min(500, input.max ?? 120));
    const hits: string[] = [];
    const deadline = Date.now() + 3_000;
    for (const rel of walk(ctx.projectRoot, { maxFiles: 6000 })) {
      if (Date.now() > deadline) break;
      if (globRe && !globRe.test(rel)) continue;
      const src = readText(ctx.projectRoot, rel);
      if (!src) continue;
      const lines = src.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i]!;
        if (re.test(line.length > 2000 ? line.slice(0, 2000) : line)) {
          hits.push(`${rel}:${i + 1}: ${line.trim().slice(0, 240)}`);
          if (hits.length >= max) break;
        }
      }
      if (hits.length >= max) break;
    }
    return {
      ok: true,
      summary: hits.length ? `${hits.length} hit(s)\n${hits.join('\n')}` : 'no match',
      data: { hits },
    };
  },
};

const viewImage: Tool = {
  name: 'view_image',
  description: 'Open an image or short video from the project (a ref sheet, a still, a storyboard sheet, a motion grid) so you can look at it.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: { path: { type: 'string' } },
    required: ['path'],
    additionalProperties: false,
  },
  async run(input: { path: string }, ctx) {
    const abs = safeJoin(ctx.projectRoot, input.path);
    if (!fs.existsSync(abs)) throw new Error(`no such file: ${input.path}`);
    const rel = relOf(ctx.projectRoot, abs);
    return {
      ok: true,
      summary: `attached ${rel}`,
      artifacts: [{ kind: mimeOf(abs).startsWith('image/') ? 'image' : 'video', path: rel, mime: mimeOf(abs) }],
    };
  },
};

const writeBible: Tool = {
  name: 'write_bible',
  description:
    'Write or update the episode visual continuity Bible (`src/episodes/<episode>/bible.json`): invariant 2D stylePrompt, negativePrompt, recurring characters (id, name, role, appearance, refPrompt), and places (id, name, recipe, description, refPrompt). Call this BEFORE `whatif_run { step: "refs" }` and BEFORE writing shot prompts with `edit_shot`.',
  policy: 'auto',
  mutating: true,
  schema: {
    type: 'object',
    properties: {
      episode: { type: 'string' },
      stylePrompt: {
        type: 'string',
        description: 'Invariant English 2D visual style prompt matching the channel aesthetic (flat 2D cartoon, crisp outlines, 3 depth layers, NO 3D, NO photorealism, NO title cards/subtitles).',
      },
      negativePrompt: {
        type: 'string',
        description: 'Optional negative prompt applied to every generated reference and still.',
      },
      characters: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'UPPERCASE_ID (e.g. MAYA, PROF)' },
            name: { type: 'string' },
            role: { type: 'string' },
            appearance: { type: 'string', description: 'Detailed invariant English physical description: age, build, skin tone, hair style/color, exact outfit colors, distinguishing accessories.' },
            refPrompt: { type: 'string', description: 'Optional extra guidance for generating the neutral full-body character reference sheet.' },
          },
          required: ['id', 'name', 'appearance'],
          additionalProperties: false,
        },
      },
      places: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'UPPERCASE_ID (e.g. CLASSROOM, LIVING)' },
            name: { type: 'string' },
            recipe: { type: 'string', description: 'Optional semantic set slug (e.g. classroom, living, corridor).' },
            description: { type: 'string', description: 'Detailed invariant English description of walls, furniture layout, colors, and lighting.' },
            refPrompt: { type: 'string', description: 'Optional extra guidance for generating the empty room reference image.' },
          },
          required: ['id', 'name', 'description'],
          additionalProperties: false,
        },
      },
    },
    required: ['episode', 'stylePrompt', 'characters', 'places'],
    additionalProperties: false,
  },
  validate: (input) => {
    const slug = String(input.episode ?? '').trim();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(slug)) return 'episode must be a clean slug';
    return null;
  },
  async run(
    input: {
      episode: string;
      stylePrompt: string;
      negativePrompt?: string;
      characters: unknown[];
      places: unknown[];
    },
    ctx,
  ): Promise<ToolResult> {
    const slug = String(input.episode ?? '').trim();
    const epRel = `src/episodes/${slug}`;
    assertWritableProjectPath(ctx.projectRoot, `${epRel}/bible.json`);
    const epDir = safeJoin(ctx.projectRoot, epRel);
    if (!fs.existsSync(epDir)) {
      throw new Error(`${epRel} does not exist yet: scaffold the episode first`);
    }
    const candidate = {
      stylePrompt: input.stylePrompt,
      ...(input.negativePrompt !== undefined ? { negativePrompt: input.negativePrompt } : {}),
      characters: input.characters,
      places: input.places,
    };
    assertNotCompactedPlaceholder(JSON.stringify(candidate), 'write_bible');
    const { bible, errors } = validateBible(candidate);
    if (!bible || errors.length) {
      return {
        ok: false,
        summary: `bible.json rejected:\n- ${errors.join('\n- ')}`,
      };
    }
    const biblePath = safeJoin(ctx.projectRoot, `${epRel}/bible.json`);
    fs.writeFileSync(biblePath, JSON.stringify(bible, null, 2) + '\n', 'utf8');
    return {
      ok: true,
      summary:
        `Saved ${epRel}/bible.json (${bible.places.length} place(s): ${bible.places.map((p) => p.id).join(', ')}; ` +
        `${bible.characters.length} character(s): ${bible.characters.map((c) => c.id).join(', ') || 'none'}). ` +
        `Next: run whatif_run { episode: "${slug}", step: "refs" } to generate neutral reference sheets with Nano Banana 2 Lite.`,
      data: {
        path: `${epRel}/bible.json`,
        places: bible.places.map((p) => p.id),
        characters: bible.characters.map((c) => c.id),
      },
    };
  },
};

const shotItemSchema = {
  type: 'object' as const,
  properties: {
    id: { type: 'string' as const, description: 'Shot id, e.g. "S1".' },
    place: { type: 'string' as const, description: 'Place ID from bible.json.places (e.g. "LIVING").' },
    characters: {
      type: 'array' as const,
      items: { type: 'string' as const },
      description: 'ONLY the character IDs from bible.json.characters actually visible in this shot ([] for empty room or macro object insert).',
    },
    size: { type: 'string' as const, description: '"wide" | "medium" | "close" | "insert" | "establishing"' },
    angle: { type: 'string' as const, enum: [...VALID_SHOT_ANGLES] },
    view: { type: 'string' as const, enum: [...VALID_SHOT_VIEWS] },
    cameraMove: { type: 'string' as const, enum: [...VALID_CAMERA_MOVES] },
    imagePrompt: {
      type: 'string' as const,
      description: 'Detailed English scene prompt for Nano Banana 2 Lite (framing, exact character poses, expressions, held/tabletop objects, NO title card, NO subtitles).',
    },
    imageModel: {
      type: 'string' as const,
      description: 'Optional per-shot image model override (e.g. "nano2.1" when a shot needs higher detail or >3 refs while keeping "nano2-lite" as default).',
    },
    negativePrompt: { type: 'string' as const },
    onScreenText: {
      type: ['object', 'null'] as ('object' | 'null')[],
      description: 'Optional in-world diegetic text ONLY (phone SMS, document, sign, board) in the episode language. Pass null or omit if none. NEVER an episode title.',
      properties: {
        kind: { type: 'string' as const, enum: [...VALID_ONSCREEN_KINDS] },
        text: { type: 'string' as const },
        lang: { type: 'string' as const, enum: ['fr', 'en'] },
      },
      required: ['kind', 'text'],
    },
    videoPrompt: {
      type: 'string' as const,
      description: 'English Image-to-Video motion & camera prompt for Gemini Omni Flash 2.1 (single clean action matching shot duration, no cuts).',
    },
    sfxHint: {
      type: 'string' as const,
      description: 'Physical sound effects & room ambience for Gemini Omni Flash 2.1 (STRICTLY NO music, NO dialogue/voice).',
    },
    seed: { type: 'integer' as const },
  },
};

const editShot: Tool = {
  name: 'edit_shot',
  description:
    'Write or update diffusion prompt specs (`src/episodes/<episode>/shots/S<N>.json`) for Nano Banana 2 Lite (image) + Gemini Omni Flash 2.1 (I2V + SFX). Supports either a single shot (`{ episode, id, place, characters, imagePrompt, videoPrompt, sfxHint, ... }` or `{ episode, id, shot: {...} }`) OR a batch (`{ episode, shots: [...] }`).',
  policy: 'auto',
  mutating: true,
  schema: {
    type: 'object',
    properties: {
      episode: { type: 'string' },
      id: { type: 'string', description: 'Shot id such as "S7" (required when updating a single shot without `shots`).' },
      shot: { ...shotItemSchema, description: 'Single shot object (alternative to flat fields).' },
      shots: {
        type: 'array',
        description: 'Batch of 1..15 shot spec objects to write at once.',
        items: shotItemSchema,
      },
      place: shotItemSchema.properties.place,
      characters: shotItemSchema.properties.characters,
      size: shotItemSchema.properties.size,
      angle: shotItemSchema.properties.angle,
      view: shotItemSchema.properties.view,
      cameraMove: shotItemSchema.properties.cameraMove,
      imagePrompt: shotItemSchema.properties.imagePrompt,
      imageModel: shotItemSchema.properties.imageModel,
      negativePrompt: shotItemSchema.properties.negativePrompt,
      onScreenText: shotItemSchema.properties.onScreenText,
      videoPrompt: shotItemSchema.properties.videoPrompt,
      sfxHint: shotItemSchema.properties.sfxHint,
      seed: shotItemSchema.properties.seed,
    },
    required: ['episode'],
    additionalProperties: false,
  },
  validate: (input) => {
    const slug = String(input.episode ?? '').trim();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9_-]*$/.test(slug)) {
      return 'episode must be a clean slug (letters, digits, hyphens, underscores)';
    }
    const hasBatch = Array.isArray(input.shots) && input.shots.length > 0;
    if (hasBatch) return null;
    const id = String(input.id ?? (input.shot as any)?.id ?? '').trim().toUpperCase();
    if (Array.isArray(input.shots) && !id) return 'shots array cannot be empty';
    if (!/^S\d+$/.test(id)) return 'pass either `shots: [...]` or a single shot `id` like "S7"';
    return null;
  },
  async run(input: Record<string, unknown>, ctx): Promise<ToolResult> {
    const slug = String(input.episode ?? '').trim();
    const epRel = `src/episodes/${slug}`;
    assertWritableProjectPath(ctx.projectRoot, `${epRel}/shots/S1.json`);
    const epDir = safeJoin(ctx.projectRoot, epRel);
    if (!fs.existsSync(epDir)) {
      throw new Error(`${epRel} does not exist yet: scaffold the episode first`);
    }

    const candidates: { expectedId?: string; raw: Record<string, unknown> }[] = [];
    if (Array.isArray(input.shots) && input.shots.length > 0) {
      for (const item of input.shots) {
        if (!item || typeof item !== 'object' || Array.isArray(item)) {
          return { ok: false, summary: 'every entry in shots[] must be a JSON object' };
        }
        const obj = item as Record<string, unknown>;
        const eid = typeof obj.id === 'string' ? obj.id.trim().toUpperCase() : undefined;
        candidates.push({ expectedId: eid, raw: obj });
      }
    } else {
      const baseShot =
        input.shot && typeof input.shot === 'object' && !Array.isArray(input.shot)
          ? { ...(input.shot as Record<string, unknown>) }
          : { ...input };
      delete baseShot.episode;
      delete baseShot.shot;
      delete baseShot.shots;
      const eid = String(input.id ?? baseShot.id ?? '').trim().toUpperCase();
      baseShot.id = eid;
      candidates.push({ expectedId: eid, raw: baseShot });
    }

    assertNotCompactedPlaceholder(JSON.stringify(candidates), 'edit_shot');

    const validated: DiffusionShotSpec[] = [];
    const errors: string[] = [];
    for (const c of candidates) {
      const res = validateShotSpec(c.raw, c.expectedId);
      if (!res.shot || res.errors.length) {
        errors.push(...res.errors);
        continue;
      }
      errors.push(...checkShotAgainstBible(epDir, res.shot));
      validated.push(res.shot);
    }

    if (errors.length) {
      return {
        ok: false,
        summary: `edit_shot rejected:\n- ${errors.slice(0, 15).join('\n- ')}`,
      };
    }

    const shotsDir = path.join(epDir, 'shots');
    fs.mkdirSync(shotsDir, { recursive: true });
    // Remove scaffold S1 placeholder if another shot is written first
    if (!validated.some((s) => s.id === 'S1')) {
      const stubS1 = path.join(shotsDir, 'S1.json');
      if (fs.existsSync(stubS1) && fs.readFileSync(stubS1, 'utf8').includes('PLACEHOLDER')) {
        fs.unlinkSync(stubS1);
      }
    }

    const writtenIds: string[] = [];
    let replacedCount = 0;
    for (const spec of validated) {
      const shotRel = `${epRel}/shots/${spec.id}.json`;
      const shotAbs = safeJoin(ctx.projectRoot, shotRel);
      const existed = fs.existsSync(shotAbs) && !fs.readFileSync(shotAbs, 'utf8').includes('PLACEHOLDER');
      if (existed) replacedCount++;
      fs.writeFileSync(shotAbs, JSON.stringify(spec, null, 2) + '\n', 'utf8');
      writtenIds.push(spec.id);
    }

    const bundle = readEpisodeShots(epDir);
    return {
      ok: true,
      summary: `Saved ${writtenIds.length} shot prompt(s) (${sortShotIds(writtenIds).join(', ')}; ${replacedCount} replaced). Total exported in shots/: ${bundle.exportedShotIds.length}.`,
      data: {
        ids: sortShotIds(writtenIds),
        totalExported: bundle.exportedShotIds.length,
        replaced: replacedCount,
      },
    };
  },
};

export const fsTools: Tool[] = [repoMap, readFile, writeFile, editFile, writeBible, editShot, glob, grep, viewImage];
