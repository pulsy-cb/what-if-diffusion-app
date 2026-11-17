import fs from 'node:fs';
import path from 'node:path';
import type { Tool } from './types.js';
import { stripNegativeClauses, type DiffusionShotSpec, type EpisodeBible } from './episode-source.js';

/**
 * One staging card per shot, decided BEFORE writing diffusion prompts.
 * Guarantees visual variety, character/object continuity, and clear framing
 * before Nano Banana 2 Lite and Gemini Omni Flash 2.1 are invoked.
 */
export const VALID_STAGE_VIEWS = ['front', 'reverse', 'left', 'right', '3/4-left', '3/4-right'] as const;
export type StageView = (typeof VALID_STAGE_VIEWS)[number];
export const VALID_STAGE_CAMS = ['establishing', 'wide', 'medium', 'close', 'deep', 'ots', 'pov'] as const;
export type StageCam = (typeof VALID_STAGE_CAMS)[number];
export const VALID_STAGE_FOREGROUNDS = ['auto', 'plant', 'doorframe', 'lamp', 'chair', 'curtain', 'none'] as const;
export type StageForeground = (typeof VALID_STAGE_FOREGROUNDS)[number];

export type StageCard = {
  see: string;
  /** the one thing the eye goes to: a person, an object, a hand on a desk */
  subject?: string;
  people: { who: string; where: string; does: string }[];
  objects: string[];
  angle?: 'level' | 'low' | 'high';
  view?: StageView;
  cam?: StageCam;
  zone?: string;
  foreground?: StageForeground;
  moves?: boolean;
  change?: string;
};

const norm = (s: unknown): string =>
  String(s ?? '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

const VAGUE =
  /^(reagit|reagissent|regarde|ecoute|attend|reste|parle|discute|observe|change d expression|se tient|est la|est debout|est assise?|reacts?|looks?|listens?|stands?|waits?|talks?|watches|is there|idle)\b/;

export const vagueAction = (does: string): boolean => {
  const n = norm(does);
  return VAGUE.test(n) && n.split(' ').length <= 4;
};

/** A new-format episode: shotlist.py writes a `stage` key (null until filled) on every shot. */
export const usesStaging = (shots: any[]): boolean =>
  Array.isArray(shots) && shots.some((s) => s && Object.prototype.hasOwnProperty.call(s, 'stage'));

const NARRATOR = /^(NARR|NARRATOR|NARRATEUR|NARRATRICE|VOIX|VOICE|VOIX_OFF)$/;

/** What is wrong with one card; empty when it is usable. */
export function stageCardProblems(card: any, shot: { id: string; size?: string; note?: string }, first: boolean): string[] {
  const out: string[] = [];
  const id = shot.id;
  if (!card || typeof card !== 'object') return [`${id}: no staging card`];
  const see = String(card.see ?? '').trim();
  if (see.length < 50) {
    out.push(
      `${id}.see is too short (${see.length} chars, 50 at least): one full sentence on what the viewer SEES (who, doing what, with what, where in the frame)`,
    );
  } else if (shot.note && norm(see) === norm(shot.note)) {
    out.push(`${id}.see repeats the voice line: describe the picture, not what is said`);
  }
  const people = Array.isArray(card.people) ? card.people : null;
  if (!people) out.push(`${id}.people must be an array (empty for an insert or an empty room)`);
  for (const [i, p] of (people ?? []).entries()) {
    if (NARRATOR.test(String(p?.who ?? '').toUpperCase())) {
      out.push(
        `${id}.people[${i}]: the narrator is a voice, not someone in the picture: remove "${p.who}" from people (a first-person narrator is on screen under their own character name)`,
      );
    } else if (!/^[A-Z][A-Z0-9_]*$/.test(String(p?.who ?? ''))) {
      out.push(`${id}.people[${i}].who must be the UPPERCASE character ID used in bible.json (PROF, MAYA…), got "${p?.who ?? ''}"`);
    }
    if (String(p?.where ?? '').trim().length < 3) {
      out.push(`${id}.people[${i}].where is missing: where in the room ("left, near the board", "front desk right")`);
    }
    if (vagueAction(String(p?.does ?? ''))) {
      out.push(`${id}.people[${i}].does "${p.does}" says nothing visual: say HOW (which hand, toward what, holding what, leaning on what)`);
    } else if (String(p?.does ?? '').trim().length < 8) {
      out.push(`${id}.people[${i}].does is too vague: a concrete action ("turns the bag upside down over the desk"), not "idle" or "talks"`);
    }
  }
  if (String(card.subject ?? '').trim().length < 3) {
    out.push(`${id}.subject is missing: the ONE thing the eye goes to in this picture (a face, a bag being emptied, a hand on a desk)`);
  }
  if (!Array.isArray(card.objects) || card.objects.some((o: unknown) => typeof o !== 'string' || !o.trim())) {
    out.push(`${id}.objects must be an array of names (empty when nothing is held or shown)`);
  }
  if (card.angle !== undefined && !['level', 'low', 'high'].includes(card.angle)) {
    out.push(`${id}.angle must be "level", "low" or "high"`);
  }
  if (card.view !== undefined && !VALID_STAGE_VIEWS.includes(card.view)) {
    out.push(`${id}.view must be one of ${VALID_STAGE_VIEWS.join(', ')}`);
  }
  if (card.cam !== undefined && !VALID_STAGE_CAMS.includes(card.cam)) {
    out.push(`${id}.cam must be one of ${VALID_STAGE_CAMS.join(', ')}`);
  }
  if (card.foreground !== undefined && !VALID_STAGE_FOREGROUNDS.includes(card.foreground)) {
    out.push(`${id}.foreground must be one of ${VALID_STAGE_FOREGROUNDS.join(', ')}`);
  }
  if (card.zone !== undefined && (typeof card.zone !== 'string' || !card.zone.trim())) {
    out.push(`${id}.zone must be a non-empty room zone ('window', 'door', 'sofa', 'left', 'right')`);
  }
  if (card.moves !== undefined && typeof card.moves !== 'boolean') {
    out.push(`${id}.moves must be true or false`);
  }
  const size = String(shot.size ?? '').toLowerCase();
  const empty = !(people ?? []).length;
  if (empty && !/insert|establish|detail|macro/.test(size) && !(Array.isArray(card.objects) && card.objects.length)) {
    out.push(`${id} has no person and no object on a "${size || 'medium'}" shot: say who is there, or what object fills the frame`);
  }
  if (!first && String(card.change ?? '').trim().length < 10) {
    out.push(`${id}.change is missing: what differs from the previous shot (who moved, new framing, new object)`);
  }
  return out;
}

const signature = (c: StageCard): string =>
  (c.people ?? [])
    .map((p) => `${p.who}:${norm(p.does)}`)
    .sort()
    .join('|');

/** The shots that still have no card, as one line (empty when every shot has one). */
export function missingCards(shots: any[]): string[] {
  const missing = shots.filter((s) => !s?.stage).map((s) => String(s.id));
  return missing.length
    ? [`${missing.length}/${shots.length} shots have no staging card (${missing.slice(0, 12).join(', ')}): call stage_shots { episode, cards: [...] } BEFORE writing prompts`]
    : [];
}

/** Is this shot prompt spec written? The scaffold's placeholder file does not count. */
export function shotCoded(episodeDir: string, id: string): boolean {
  try {
    const raw = fs.readFileSync(path.join(episodeDir, 'shots', `${id}.json`), 'utf8');
    if (/PLACEHOLDER/i.test(raw)) return false;
    const parsed = JSON.parse(raw);
    return Boolean(parsed && typeof parsed === 'object' && typeof parsed.imagePrompt === 'string' && parsed.imagePrompt.length >= 20);
  } catch {
    return false;
  }
}

/** Cards as a whole: none missing, no two neighbours alike, nobody doing the same thing all along. */
export function stagingProblems(shots: any[]): string[] {
  const out: string[] = [];
  const missing = missingCards(shots);
  if (missing.length) return missing;
  shots.forEach((s, i) => out.push(...stageCardProblems(s.stage, { id: String(s.id), size: s.size, note: s.note }, i === 0)));
  for (let i = 1; i < shots.length; i++) {
    const a = shots[i - 1].stage as StageCard;
    const b = shots[i].stage as StageCard;
    const sameSee = norm(a.see) === norm(b.see);
    const samePeople = signature(a) !== '' && signature(a) === signature(b) && String(shots[i - 1].size) === String(shots[i].size);
    if (sameSee || samePeople) {
      out.push(`${shots[i - 1].id} and ${shots[i].id} have the same card: two neighbouring shots must differ in action or in framing`);
    }
  }
  const perWho = new Map<string, Map<string, string[]>>();
  for (const s of shots) {
    for (const p of (s.stage as StageCard).people ?? []) {
      const m = perWho.get(p.who) ?? new Map<string, string[]>();
      m.set(norm(p.does), [...(m.get(norm(p.does)) ?? []), String(s.id)]);
      perWho.set(p.who, m);
    }
  }
  for (const [who, m] of perWho) {
    const total = [...m.values()].reduce((n, v) => n + v.length, 0);
    for (const [does, ids] of m) {
      if (ids.length >= 4 && ids.length > total / 3) {
        out.push(`${who} "${does}" in ${ids.length}/${total} shots (${ids.slice(0, 6).join(', ')}): vary what this person does`);
      }
    }
  }
  return out;
}

const HANDLED =
  /\b(phone|telephone|portable|smartphone|handset|combine|bag|sac|cartable|backpack|paper|papier|feuille|lettre|letter|note|copie|crumpled|cup|tasse|mug|verre|glass|keys?|cles?|book|livre|cahier|pen|stylo|crayon|pencil|craie|chalk|folder|dossier|box|boite|carton|wallet|portefeuille|manette|controller|remote|telecommande|camera|billet|ticket|badge|card|carte|bottle|bouteille|tray|plateau|plate|assiette|knife|couteau|ring|bague|gift|cadeau|photo|pill|pilule|comprime|envelope|enveloppe|fiole|vial|flask)s?\b/;

/** The objects of a card that a person holds or handles (the rest is set dressing). */
export const handledObjects = (objects: unknown): string[] =>
  Array.isArray(objects) ? objects.filter((o): o is string => typeof o === 'string' && HANDLED.test(norm(o))) : [];

/** Bilingual French/English equivalence groups so French staging cards match English diffusion prompts cleanly. */
const OBJECT_SYNONYM_GROUPS: string[][] = [
  ['phone', 'smartphone', 'telephone', 'portable', 'mobile', 'cellphone', 'handset', 'combine', 'screen', 'sms', 'text'],
  ['bag', 'backpack', 'sac', 'cartable', 'purse', 'handbag', 'satchel', 'totebag'],
  ['paper', 'papier', 'feuille', 'sheet', 'note', 'lettre', 'letter', 'copie', 'crumpled', 'document', 'bill', 'facture', 'receipt', 'dossier', 'folder', 'contract', 'contrat'],
  ['cup', 'tasse', 'mug', 'verre', 'glass', 'coffee', 'cafe'],
  ['key', 'keys', 'cle', 'cles', 'keyring'],
  ['book', 'livre', 'cahier', 'notebook', 'diary', 'journal'],
  ['pen', 'stylo', 'crayon', 'pencil', 'craie', 'chalk', 'marker'],
  ['box', 'boite', 'carton', 'package', 'parcel'],
  ['wallet', 'portefeuille', 'billet', 'cash', 'money', 'monnaie'],
  ['controller', 'manette', 'gamepad', 'remote', 'telecommande'],
  ['bottle', 'bouteille', 'fiole', 'vial', 'flask'],
  ['tray', 'plateau', 'plate', 'assiette'],
  ['knife', 'couteau', 'blade'],
  ['ring', 'bague', 'alliance'],
  ['gift', 'cadeau', 'present'],
  ['photo', 'picture', 'photograph', 'polaroid'],
  ['pill', 'pills', 'pilule', 'comprime', 'tablet', 'medicine'],
  ['envelope', 'enveloppe'],
  ['badge', 'card', 'carte', 'id'],
  ['camera', 'appareil'],
];

function objectMentionedInPrompt(objLabel: string, promptHaystackNorm: string): boolean {
  const objNorm = norm(objLabel);
  if (!objNorm) return true;
  if (promptHaystackNorm.includes(objNorm)) return true;
  const objWords = objNorm.split(' ').filter((w) => w.length >= 2);
  for (const group of OBJECT_SYNONYM_GROUPS) {
    if (objWords.some((w) => group.includes(w))) {
      if (group.some((syn) => new RegExp(`\\b${syn}\\b`).test(promptHaystackNorm))) {
        return true;
      }
    }
  }
  return objWords.some((w) => w.length >= 4 && promptHaystackNorm.includes(w));
}

const MOTION_VERBS_RE =
  /\b(walk|walks|walking|cross|crosses|crossing|enter|enters|entering|exit|exits|exiting|leave|leaves|leaving|step|steps|stepping|stride|striding|run|runs|running|rush|rushes|approach|approaches|approaching|move|moves|moving|paces?|pacing)\b/i;

const PLURAL_PEOPLE_RE =
  /\b(two|three|four|both|several|multiple|a\s+pair\s+of|couple\s+of)\s+(?:[a-z]+\s+){0,2}(officers?|cops?|police(?:\s+officers?)?|swat|agents?|guards?|soldiers?|operators?|generals?|men|women|people|persons?|teens?|teenagers?|boys?|girls?|players?|friends?|family\s+members?)\b/i;

/**
 * Verify each shot's diffusion prompt spec (`shots/S<N>.json`) against its staging card and `bible.json`:
 * - `characters` matches `card.people[].who` (every card person included, no undeclared extra people),
 * - handled objects appear in `imagePrompt` / `videoPrompt`,
 * - `angle` and `view` match the card,
 * - neighboring shots have distinct `imagePrompt`s,
 * - episode-wide anti-monotony across POV (`place + size + view + angle`) and `cameraMove`.
 */
export function stagingPromptProblems(
  shots: any[],
  shotSpecs: Map<string, DiffusionShotSpec>,
  _bible: EpisodeBible | null,
  withCards: boolean,
  _aspect = '16:9',
): string[] {
  const out: string[] = [];

  for (let i = 0; i < shots.length; i++) {
    const s = shots[i];
    const id = String(s?.id ?? '').toUpperCase();
    const spec = shotSpecs.get(id);
    if (!spec) continue;

    if (i > 0) {
      const prevId = String(shots[i - 1]?.id ?? '').toUpperCase();
      const prevSpec = shotSpecs.get(prevId);
      if (prevSpec && norm(prevSpec.imagePrompt) === norm(spec.imagePrompt)) {
        out.push(`${prevId} and ${id} have identical imagePrompt text: each neighboring shot must describe a fresh image`);
      }
    }

    const card = withCards ? (s?.stage as StageCard | null) : null;
    if (!card) continue;

    const cardPeople = (card.people ?? []).map((p) => String(p.who ?? '').trim().toUpperCase()).filter(Boolean);
    const specChars = (spec.characters ?? []).map((c) => c.trim().toUpperCase()).filter(Boolean);

    for (const who of cardPeople) {
      if (!specChars.includes(who)) {
        out.push(`${id}: the staging card puts ${who} in the shot ("${card.people.find((p) => p.who.toUpperCase() === who)?.does ?? ''}") but ${id}.json.characters omits ${who}`);
      }
    }
    for (const cid of specChars) {
      if (!cardPeople.includes(cid)) {
        out.push(`${id}: ${id}.json.characters attaches ${cid}, who is NOT in the staging card's people[] (remove extra character refs to avoid diffusion clones)`);
      }
    }

    const handled = handledObjects(card.objects);
    if (handled.length) {
      const haystack = norm(`${spec.imagePrompt} ${spec.videoPrompt} ${spec.onScreenText?.text ?? ''}`);
      for (const obj of handled) {
        if (!objectMentionedInPrompt(obj, haystack)) {
          out.push(`${id}: the staging card lists handled object "${obj}" but ${id}.json.imagePrompt does not describe it`);
        }
      }
    }

    if ((card.angle === 'low' || card.angle === 'high') && spec.angle !== card.angle) {
      out.push(`${id}: the staging card asks for angle "${card.angle}" but ${id}.json.angle is "${spec.angle ?? 'unset'}"`);
    }
    if (card.view && card.view !== 'front' && spec.view !== card.view) {
      out.push(`${id}: the staging card asks for view "${card.view}" but ${id}.json.view is "${spec.view ?? 'unset'}"`);
    }
    if (card.moves && !MOTION_VERBS_RE.test(`${spec.videoPrompt} ${spec.imagePrompt}`)) {
      out.push(`${id}: the staging card sets moves=true (someone crosses/walks) but videoPrompt describes no physical movement`);
    }

    if (specChars.length === 1) {
      const positiveScene = stripNegativeClauses(`${spec.imagePrompt} ${spec.videoPrompt}`);
      const pluralHit = PLURAL_PEOPLE_RE.exec(positiveScene);
      if (pluralHit) {
        out.push(
          `${id}: characters has only 1 ID (${specChars[0]}) yet prompt mentions "${pluralHit[0]}" (causes diffusion face cloning). Depict only ${specChars[0]} (or silhouette) OR define a distinct second character ID in bible.json + stage_shots`,
        );
      }
    }

    if ((card.people ?? []).length === 1 && specChars.length === 1) {
      const wNorm = norm(card.people[0]?.where ?? '');
      const pNorm = norm(spec.imagePrompt);
      const cardLeft = /\b(left|gauche)\b/.test(wNorm) && !/\b(right|droite)\b/.test(wNorm);
      const cardRight = /\b(right|droite)\b/.test(wNorm) && !/\b(left|gauche)\b/.test(wNorm);
      const promptLeft = /\b(screen left|left side|on the left)\b/.test(pNorm);
      const promptRight = /\b(screen right|right side|on the right)\b/.test(pNorm);
      if ((cardLeft && promptRight && !promptLeft) || (cardRight && promptLeft && !promptRight)) {
        out.push(
          `${id}: staging card puts ${specChars[0]} at "${card.people[0]?.where}" while imagePrompt places them on the opposite side of frame (align left/right placement)`,
        );
      }
    }
  }

  // Episode-wide anti-monotony checks on >= 8 shots
  if (shots.length >= 8) {
    type PovEntry = { id: string; isCharShot: boolean; place: string; size: string; view: string; angle: string; move: string };
    const povs: PovEntry[] = [];
    for (const s of shots) {
      const id = String(s?.id ?? '').toUpperCase();
      const spec = shotSpecs.get(id);
      if (!spec) continue;
      const size = String(spec.size ?? s?.size ?? 'medium').toLowerCase();
      const isCharShot = spec.characters.length > 0 && !/insert|establishing/.test(size);
      povs.push({
        id,
        isCharShot,
        place: spec.place,
        size,
        view: spec.view ?? String(s?.view ?? 'front'),
        angle: spec.angle ?? 'level',
        move: spec.cameraMove ?? String(s?.move ?? 'push'),
      });
    }

    let triadWarnings = 0;
    for (let i = 2; i < povs.length && triadWarnings < 2; i++) {
      const a = povs[i - 2]!;
      const b = povs[i - 1]!;
      const c = povs[i]!;
      if (
        a.isCharShot &&
        b.isCharShot &&
        c.isCharShot &&
        a.place === b.place &&
        b.place === c.place &&
        a.size === b.size &&
        b.size === c.size &&
        a.view === b.view &&
        b.view === c.view &&
        a.angle === b.angle &&
        b.angle === c.angle
      ) {
        out.push(
          `${a.id}, ${b.id}, ${c.id} repeat the exact same POV (${a.place}, size '${a.size}', view '${a.view}', angle '${a.angle}') three times in a row: vary view ('3/4-left' | '3/4-right' | 'reverse'), framing size, angle, or insert shots`,
        );
        triadWarnings++;
        i += 2;
      }
    }

    const charEntries = povs.filter((p) => p.isCharShot);
    if (charEntries.length >= 8) {
      const moveCounts = new Map<string, number>();
      for (const p of charEntries) moveCounts.set(p.move, (moveCounts.get(p.move) ?? 0) + 1);
      const topMove = [...moveCounts.entries()].sort((x, y) => y[1] - x[1])[0];
      if (topMove && topMove[1] > Math.ceil(charEntries.length * 0.7)) {
        out.push(
          `cameraMove '${topMove[0]}' is used in ${topMove[1]}/${charEntries.length} character shots (> 70%): vary across 'push', 'pull', 'pan', 'track', 'whip', and 'still'`,
        );
      }
    }
  }

  return out;
}

/** The cards as the Reviewer reads them: one block per shot. */
export function formatStageCards(shots: any[], ids?: string[]): string {
  return shots
    .filter((s) => s?.stage && (!ids || ids.includes(String(s.id))))
    .map((s) => {
      const c = s.stage as StageCard;
      const who = (c.people ?? []).map((p) => `${p.who} (${p.where}): ${p.does}`).join(' ; ') || 'nobody';
      const vw = c.view ?? s.view;
      const rev = vw && vw !== 'front' ? `view: ${vw}` : '';
      const extra = [
        c.objects?.length ? `objects: ${c.objects.join(', ')}` : '',
        c.cam ? `cam: ${c.cam}` : '',
        c.zone ? `zone: ${c.zone}` : '',
        c.foreground && c.foreground !== 'none' ? `foreground: ${c.foreground}` : '',
        c.angle && c.angle !== 'level' ? `angle: ${c.angle}` : '',
        rev,
        c.moves ? 'someone crosses the room' : '',
      ]
        .filter(Boolean)
        .join(' · ');
      return `${s.id} [${s.size ?? ''}] ${c.see}${c.subject ? ` — sujet : ${c.subject}` : ''}\n   ${who}${extra ? `\n   ${extra}` : ''}`;
    })
    .join('\n');
}

export const stageShots: Tool = {
  name: 'stage_shots',
  description:
    'Write the staging card of each shot BEFORE writing diffusion prompts: what the viewer sees, who is where doing what, which objects, the angle, what changes from the previous shot. Saves into episode.json and checks continuity & anti-monotony. Call it in batches of ~10, then build bible.json + shots/S*.json to match.',
  policy: 'auto',
  schema: {
    type: 'object',
    properties: {
      episode: { type: 'string' },
      cards: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'shot id, e.g. S14' },
            see: { type: 'string', description: 'one sentence: the picture, not the voice line' },
            subject: { type: 'string', description: 'the one thing the eye goes to' },
            people: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  who: { type: 'string', description: 'UPPERCASE character ID from bible.json: PROF, MAYA…' },
                  where: { type: 'string' },
                  does: { type: 'string' },
                },
                required: ['who', 'where', 'does'],
              },
            },
            objects: {
              type: 'array',
              items: { type: 'string' },
              description: 'ONLY what a person holds or handles in this shot (phone, bag, letter). Furniture and set go in `see`, not here',
            },
            angle: { type: 'string', enum: ['level', 'low', 'high'] },
            view: {
              type: 'string',
              enum: [...VALID_STAGE_VIEWS],
              description: "room perspective: 'front', 'reverse' (contrechamp), '3/4-left', '3/4-right', 'left', or 'right'",
            },
            cam: {
              type: 'string',
              enum: [...VALID_STAGE_CAMS],
              description: "optional shot framing override such as 'ots' (over-the-shoulder), 'pov' (first-person), or 'deep'",
            },
            zone: {
              type: 'string',
              description: "optional room sub-zone ('window', 'door', 'sofa', 'counter', 'left', 'right')",
            },
            foreground: {
              type: 'string',
              enum: [...VALID_STAGE_FOREGROUNDS],
              description: "optional foreground framing element ('auto', 'plant', 'doorframe', 'lamp', 'chair', 'curtain')",
            },
            moves: { type: 'boolean', description: 'true when someone walks across the room in this shot' },
            change: { type: 'string', description: 'what differs from the previous shot' },
          },
          required: ['id', 'see', 'subject', 'people', 'objects'],
        },
      },
      userAsked: {
        type: 'boolean',
        description: "true ONLY when the user's own message asks to change these shots after the board exists. Never to make a check pass.",
      },
    },
    required: ['episode', 'cards'],
    additionalProperties: false,
  },
  async run(input: { episode: string; cards: any[]; userAsked?: boolean }, ctx) {
    const slug = String(input.episode ?? '').trim();
    if (!/^[A-Za-z0-9_-]+$/.test(slug)) throw new Error('episode must be a slug such as "my-episode"');
    const file = path.join(ctx.projectRoot, 'src', 'episodes', slug, 'episode.json');
    if (!fs.existsSync(file)) throw new Error(`src/episodes/${slug}/episode.json does not exist: run the shotlist step first`);
    const ep = JSON.parse(fs.readFileSync(file, 'utf8'));
    const shots: any[] = Array.isArray(ep.shots) ? ep.shots : [];
    if (!shots.length) throw new Error('episode.json has no shots: run the shotlist step first');
    const byId = new Map(shots.map((s) => [String(s.id).toUpperCase(), s]));
    const unknown: string[] = [];
    let written = 0;

    const epDir = path.dirname(file);
    const boarded = fs.existsSync(path.join(ctx.projectRoot, 'out', slug, 'storyboard.jpg'));
    const refused = new Set<string>();
    try {
      const review = JSON.parse(fs.readFileSync(path.join(ctx.projectRoot, 'out', slug, 'review.json'), 'utf8'));
      if (review && !review.pass) {
        for (const i of review.issues ?? []) {
          if (i?.severity === 'error') {
            for (const id of String(i.shot ?? '').toUpperCase().match(/\bS\d+\b/g) ?? []) refused.add(id);
          }
        }
      }
    } catch {
      /* no review yet */
    }
    const frozen: string[] = [];
    const weakened: string[] = [];
    for (const c of Array.isArray(input.cards) ? input.cards : []) {
      const shot = byId.get(String(c?.id ?? '').toUpperCase());
      if (!shot) {
        unknown.push(String(c?.id ?? '?'));
        continue;
      }
      const { id: _id, ...card } = c;
      const sid = String(shot.id);
      if (boarded && shot.stage && !input.userAsked && JSON.stringify(shot.stage) !== JSON.stringify(card) && shotCoded(epDir, sid)) {
        if (!refused.has(sid.toUpperCase())) {
          frozen.push(sid);
          continue;
        }
        const fewer =
          (Array.isArray(card.people) ? card.people.length : 0) < (shot.stage.people ?? []).length ||
          handledObjects(card.objects).length < handledObjects(shot.stage.objects).length;
        if (fewer) {
          weakened.push(sid);
          continue;
        }
      }
      shot.stage = card;
      written++;
    }
    for (const s of shots) if (!Object.prototype.hasOwnProperty.call(s, 'stage')) s.stage = null;
    fs.writeFileSync(file, `${JSON.stringify(ep, null, 2)}\n`);
    const left = shots.filter((s) => !s.stage).map((s) => String(s.id));
    const problems = left.length
      ? shots.flatMap((s, i) => (s.stage ? stageCardProblems(s.stage, { id: String(s.id), size: s.size, note: s.note }, i === 0) : []))
      : stagingProblems(shots);
    const head = `${written} card(s) saved${unknown.length ? `; unknown shot ids ignored: ${unknown.join(', ')}` : ''}.`;
    if (frozen.length || weakened.length) {
      const lines = [
        frozen.length
          ? `${frozen.join(', ')}: NOT changed. These shots already have prompt specs and a board: when a check says a prompt does not match its card, fix the shot JSON with edit_shot, not the card.`
          : '',
        weakened.length
          ? `${weakened.join(', ')}: NOT changed. The Reviewer refused these shots, so their card may be rewritten, but it may not lose a person or a handled object: put what is missing into the prompt with edit_shot.`
          : '',
      ].filter(Boolean);
      return { ok: false, summary: `${head}\n- ${lines.join('\n- ')}` };
    }
    if (problems.length) return { ok: false, summary: `${head}\nTo fix before writing bible/prompts:\n- ${problems.slice(0, 20).join('\n- ')}` };
    if (left.length) {
      return {
        ok: true,
        summary: `${head} ${left.length}/${shots.length} shots still need a card: ${left.join(', ')}. Send them before write_bible / edit_shot.`,
      };
    }
    return {
      ok: true,
      summary: `${head} All ${shots.length} shots are staged. Next: call write_bible, run step="refs", then write S1..S${shots.length} prompts with edit_shot.`,
    };
  },
};
