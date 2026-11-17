import fs from 'node:fs';
import path from 'node:path';
import type { ChatMessage, ImagePart, Provider } from './providers/types.js';
import { readText } from '../tools/paths.js';
import {
  type DiffusionShotSpec,
  type EpisodeBible,
  type EpisodeShotsBundle,
  extractShotIds,
  hasPositiveDialogueRequest,
  hasPositiveMusicRequest,
  hasPositiveTitleRequest,
  readEpisodeBible,
  readEpisodeShots,
  sortShotIds,
} from '../tools/episode-source.js';
import { detectEpisodeLanguage, directorSizeMismatches } from '../tools/whatif.js';
import { shotWordFrames } from './context.js';
import { formatStageCards, shotCoded } from '../tools/staging.js';

export { extractShotIds as extractReviewShotIds };

export type ReviewIssue = {
  shot: string;
  severity: 'error' | 'warn';
  problem: string;
  fix: string;
};

export type ReviewReport = {
  ok: boolean;
  pass: boolean;
  model: string;
  episode: string;
  summary: string;
  issues: ReviewIssue[];
  createdAt: string;
  /** 'global' on Round 1 (100% of contact sheets); 'delta' on Rounds 2+ (high-res stills + full code of flagged/edited shots) */
  mode?: 'global' | 'delta';
  round?: number;
  targetShots?: string[];
  batches?: number;
  /** false when the model could not take the contact sheets: the verdict comes from the code alone */
  sawImages?: boolean;
  /** true while the board is refused and another review round is still allowed: render waits for it */
  blocking?: boolean;
  /** last round, still refused: the video ships and the unresolved defects are listed in this file */
  defectsFile?: string;
  /** how many reviews in a row refused each shot */
  strikes?: Record<string, number>;
  /** the answer was not valid JSON and its fields were read one by one (unescaped quotes, cut answer) */
  salvaged?: boolean;
  /** the answer as received, kept when nothing could be read from it */
  raw?: string;
  /** the reviewer gave no answer in time, twice: the chain goes on unreviewed */
  timedOut?: boolean;
  /** shots refused three times in a row: the review stops there instead of asking again */
  gaveUpOn?: string[];
  /** a closed review the user came back to: the next board is looked at again, on the shots that changed */
  reopened?: boolean;
};

const MAX_REVIEW_IMAGES = 6;

export function reviewerSystemPrompt(lang: 'fr' | 'en' = 'fr'): string {
  const verdictHint =
    lang === 'en'
      ? `Every shot is clean on screen AND prompts follow all diffusion rules (if images are not attached, verify prompts, character/place consistency, onScreenText language, zero title cards, and Omni Flash motion/SFX rules).`
      : `Chaque plan est propre à l'image ET ses prompts respectent toutes les règles diffusion (si les images ne sont pas jointes, juge sur les prompts, la cohérence bible/personnages, la langue de onScreenText, l'absence de titres et les règles Omni Flash).`;
  const replyLang =
    lang === 'en'
      ? `Write your "summary", "problem", and "fix" fields in ENGLISH (this episode is narrated in English — NEVER complain that the script/audio is English!).`
      : `Rédige tes champs "summary", "problem" et "fix" en FRANÇAIS.`;
  return `Tu es un superviseur artistique et réalisateur INDÉPENDANT pour une série d'animation 2D générée 100% par diffusion (Nano Banana 2 Lite pour les images + Gemini Omni Flash 2.1 pour l'image-to-video + bruitages physiques).
Tu reçois :
1) Le script complet de l'épisode (${lang === 'en' ? 'script EN' : 'script FR'}),
2) La bible visuelle (bible.json) + note de réalisation (DIRECTOR.md) + fiches de mise en scène (stage),
3) Les specs JSON des plans (shots/S*.json : imagePrompt, videoPrompt, sfxHint, onScreenText, size, angle, view, cameraMove, characters, place),
4) Les feuilles de référence (refs/*.png) et planches de contact du storyboard (images jointes).

Ton rôle est d'EMPÊCHER toute sortie visuellement incohérente avant l'étape coûteuse de génération vidéo. ${replyLang} Sois factuel, sévère sur les VRAIS défauts et sobre (aucun compliment).

Ce que tu DOIS vérifier sur chaque plan :
1. **Signature What-If 2D Cartoon Style (BLOQUANT)** : le style doit être un **dessin animé 2D minimaliste à têtes rondes/ovales (~1:3.5), contours noirs épais uniformes, yeux simples expressifs, aplats cel-shading et halo lumineux doré subtil autour des silhouettes** dans un décor 2D atmosphérique éclairé par des sources visibles. Si un plan dérive vers des **proportions réalistes de comics Marvel/DC, du manga/anime réaliste, du photoréalisme ou de la 3D CGI** → \`severity: "error"\` !
2. **Identité des personnages & anti-clones** : chaque personnage dans les vignettes respecte-t-il sa fiche dans \`bible.json\` / \`refs/char_*.png\` (tête ronde, coiffure, couleur exacte des vêtements) ? Aucun personnage en double/clone (ex. "two officers" alors que \`characters\` n'a qu'un ID) ni personnage non listé dans \`characters\` ? Si la fiche indique des mains seules ou un lieu vide (\`characters: []\`), aucun intrus ne doit apparaître.
3. **Placement spatial, axe 180° & continuité décor/lumière** : la position écran gauche/centre/droite et la direction du regard correspondent-elles à la fiche \`stage\` et à \`DIRECTOR.md\` ? Les plafonniers ou lampes d'une même pièce restent-ils cohérents d'un plan voisin à l'autre ? Les pieds et objets ont-ils une ombre de contact (aucun objet flottant ou fusionné à une main) ? Aucun texte parasite aléatoire sur les posters/murs quand \`onScreenText: null\` ?
4. **INTERDICTION STRICTE des titres & sous-titres** : Y a-t-il un écran-titre, une title-card d'épisode, un bandeau d'intro ou des sous-titres/légendes collés sur l'image ? Si OUI → \`severity: "error"\` immédiat !
5. **Textes diégétiques (\`onScreenText\`) uniquement & langue pure** : si un plan montre du texte dans le monde (SMS sur téléphone, document, enseigne), est-il lisible, non charabia, strictement dans la langue de l'épisode (${lang === 'en' ? 'ANGLAIS uniquement' : 'FRANÇAIS uniquement'}), et justifié par le script ?
6. **Cadrage fidèle & compatibilité Frame 0 (\`imagePrompt\` ↔ \`videoPrompt\`)** : respect du format \`insert\` (gros plan objet ≥70% sans plan large extérieur), \`close\`, \`ots\` (épaule floue au 1er plan), \`reverse\` (contrechamp 180°), format 16:9 ou 9:16 sans têtes coupées, et la posture de \`imagePrompt\` correspond-elle exactement à l'image 0 de \`videoPrompt\` (ex. déjà devant le bureau avant de frapper dessus) ?
7. **Audit Omni 1.2 Flash (\`videoPrompt\` & \`sfxHint\`)** :
   - \`videoPrompt\` décrit-il UNE action physique continue et UN mouvement caméra fluide réalisables sur la durée \`durationSec\` (sans changements de décor ni coupes multiples) ?
   - \`sfxHint\` demande-t-il UNIQUEMENT des bruitages physiques/ambiance concrets ? Si \`sfxHint\` ou \`videoPrompt\` demande de la musique, une mélodie ou de la parole/voix/dialogue/questioning → \`severity: "error"\` immédiat !

Règles strictes :
- Ne signale un plan qu'avec une preuve concrète visible sur l'image ou lue dans sa spec JSON.
- Propose TOUJOURS un \`fix\` concret et directement applicable via \`edit_shot\`, \`write_bible\` ou \`stage_shots\`.
- Classe en \`severity: "error"\` tout défaut bloquant (personnage différent/cloné, style cassé, carton-titre/sous-titres présents, texte charabia ou mauvaise langue, musique/dialogue demandé à Omni, incohérence narrative grave) et en \`severity: "warn"\` un simple polissage mineur.
- Limite à 8 problèmes maximum par rapport, triés du plus grave au plus léger.

Réponds UNIQUEMENT avec un objet JSON valide respectant exactement ce schéma :
{
  "pass": true si zéro issue de severity === "error" (${verdictHint}),
  "summary": "Résumé concis en 1-2 phrases sur la qualité visuelle, narrative et la lisibilité.",
  "issues": [
    {
      "shot": "S12",
      "severity": "error" ou "warn",
      "problem": "Description factuelle de ce qui cloche.",
      "fix": "Correction exacte à appliquer avec edit_shot / write_bible."
    }
  ]
}`;
}

export function deltaReviewerSystemPrompt(lang: 'fr' | 'en' = 'fr'): string {
  const replyLang =
    lang === 'en'
      ? `Write your "summary", "problem", and "fix" fields in ENGLISH (this episode is narrated in English — NEVER complain that the audio/script is English!).`
      : `Rédige tes champs "summary", "problem" et "fix" en FRANÇAIS.`;
  return `Tu effectues une **CONTRE-VISITE CIBLÉE (Delta Review)** après correction des plans signalés au tour précédent.

Tu reçois UNIQUEMENT :
1. Les problèmes bloquants exacts signalés au tour précédent sur ces plans,
2. Les **images individuelles pleine résolution** (\`out/<slug>/board/S<N>.png\`) et les feuilles de référence pertinentes,
3. Les specs JSON complètes (\`shots/S*.json\`), \`bible.json\`, \`DIRECTOR.md\`, le script et les repères \`mot@frame\`.

Ta mission stricte (${replyLang}) :
- Vérifier PRIORITAIREMENT si chacun des problèmes bloquants signalés au tour précédent sur ces plans ciblés est résolu (style What-If 2D à tête ronde + contour noir + halo doré, placement spatial gauche/droite & axe 180°, continuité lumière/décor, compatibilité Frame 0 entre image et \`videoPrompt\`, identités/vêtements, absence de carton-titre/sous-titres/posters textuels parasites, langue \`${lang}\` du texte diégétique, \`videoPrompt\` + \`sfxHint\` sans musique ni parole).
- Vérifier qu'aucune nouvelle erreur franche (style comic/anime réaliste, image vide/cassée, clone/personnage disparu, mauvais décor, prop flottant) n'a été introduite sur ces plans.
- N'invente PAS de nouveaux pinaillages mineurs hors périmètre si les défauts signalés sont bien corrigés !

Réponds UNIQUEMENT avec un objet JSON valide respectant ce schéma :
{
  "pass": true,
  "summary": "Résumé concis en 1-2 phrases du résultat de la contre-visite.",
  "issues": [
    {
      "shot": "S3",
      "severity": "error",
      "problem": "Ce qui reste concrètement cassé parmi les points signalés (ou la régression introduite sur ce plan).",
      "fix": "Instruction directe et réalisable via edit_shot sur ce plan."
    }
  ]
}`;
}

export type SummarizeShotsOptions = {
  onlyShots?: string[];
  shotTimings?: Map<string, { len: number; key: number }>;
  perShotChars?: number;
};

/** Summarize diffusion JSON shot specs cleanly for the Reviewer. */
export function summarizeShotPrompts(epSrc: EpisodeShotsBundle, opts: SummarizeShotsOptions = {}): string {
  const onlySet = opts.onlyShots?.length ? new Set(opts.onlyShots.map((s) => s.toUpperCase())) : null;
  const maxPerShot = opts.perShotChars ?? (onlySet ? 3_500 : 1_800);
  const ids = sortShotIds([...epSrc.shots.keys()]).filter((id) => !onlySet || onlySet.has(id));
  if (!ids.length) return '(aucune spec de plan shots/S*.json écrite)';

  return ids
    .map((id) => {
      const sp = epSrc.shots.get(id)!;
      const tm = opts.shotTimings?.get(id);
      const tmTag = tm ? `, timeline=${tm.len}f (${(tm.len / 30).toFixed(1)}s)` : '';
      const ost = sp.onScreenText
        ? `\n  onScreenText: [${sp.onScreenText.lang ?? '?'}/${sp.onScreenText.kind}] "${sp.onScreenText.text}"`
        : '';
      const neg = sp.negativePrompt ? `\n  negativePrompt: ${sp.negativePrompt}` : '';
      const body =
        `### ${id} [place=${sp.place}, chars=${sp.characters.join(',') || 'none'}, size=${sp.size ?? 'medium'}, ${sp.view ?? 'front'}/${sp.angle ?? 'level'}, cam=${sp.cameraMove ?? 'still'}${tmTag}]\n` +
        `  imagePrompt: ${sp.imagePrompt}${ost}${neg}\n` +
        `  videoPrompt: ${sp.videoPrompt}\n` +
        `  sfxHint: ${sp.sfxHint}`;
      return body.slice(0, maxPerShot);
    })
    .join('\n\n');
}

/** Deterministic measured counts over shots/S*.json + bible.json before visual judgment. */
export function diffusionMetrics(epSrc: EpisodeShotsBundle, bible?: EpisodeBible | null, expectedLang?: 'fr' | 'en'): string {
  const specs: DiffusionShotSpec[] = sortShotIds([...epSrc.shots.keys()])
    .map((id) => epSrc.shots.get(id)!)
    .filter(Boolean);
  if (!specs.length) return '0 plan JSON rédigé dans shots/S*.json';

  const moveCounts = new Map<string, number>();
  const viewCounts = new Map<string, number>();
  const placeSet = new Set<string>();
  const charSet = new Set<string>();
  const onScreenShots: string[] = [];
  const ruleViolations: string[] = [];
  let maxPovStreak = 0;
  let curPovStreak = 0;
  let maxStaticStreak = 0;
  let curStaticStreak = 0;

  for (const s of specs) {
    placeSet.add(s.place);
    for (const c of s.characters) charSet.add(c);
    const mv = s.cameraMove ?? 'still';
    const vw = s.view ?? 'front';
    moveCounts.set(mv, (moveCounts.get(mv) ?? 0) + 1);
    viewCounts.set(vw, (viewCounts.get(vw) ?? 0) + 1);
    if (s.size === 'pov') {
      curPovStreak++;
      if (curPovStreak > maxPovStreak) maxPovStreak = curPovStreak;
    } else curPovStreak = 0;
    if (mv === 'still') {
      curStaticStreak++;
      if (curStaticStreak > maxStaticStreak) maxStaticStreak = curStaticStreak;
    } else curStaticStreak = 0;

    if (s.onScreenText) {
      const ostLang = s.onScreenText.lang;
      onScreenShots.push(`${s.id}(${ostLang ?? '?'}:${s.onScreenText.kind})`);
      if (expectedLang && ostLang && ostLang !== expectedLang) {
        ruleViolations.push(`${s.id}: langue onScreenText=${ostLang} != episode=${expectedLang}`);
      }
    }
    if (hasPositiveTitleRequest(`${s.imagePrompt} ${s.videoPrompt}`)) {
      ruleViolations.push(`${s.id}: mention suspecte de title card/sous-titres`);
    }
    if (hasPositiveMusicRequest(`${s.videoPrompt} ${s.sfxHint}`)) {
      ruleViolations.push(`${s.id}: mention suspecte de musique pour Omni`);
    }
    if (hasPositiveDialogueRequest(`${s.videoPrompt} ${s.sfxHint}`)) {
      ruleViolations.push(`${s.id}: mention suspecte de dialogue/voix pour Omni`);
    }
  }

  const movesSummary = [...moveCounts.entries()].map(([k, v]) => `${k}:${v}`).join(', ');
  const viewsSummary = [...viewCounts.entries()].map(([k, v]) => `${k}:${v}`).join(', ');
  const parts = [
    `${specs.length} plan(s) diffusion`,
    `${placeSet.size} lieu(x) utilisés${bible ? `/${bible.places.length} dans bible` : ''}`,
    `${charSet.size} personnage(s) utilisés${bible ? `/${bible.characters.length} dans bible` : ''}`,
    `cadrages: ${viewsSummary}`,
    `mouvements: ${movesSummary}`,
    onScreenShots.length ? `texte diégétique: ${onScreenShots.join(', ')}` : 'aucun texte à l’écran',
  ];
  if (maxPovStreak >= 3) parts.push(`alerte POV: ${maxPovStreak} plans POV consécutifs`);
  if (maxStaticStreak >= 4) parts.push(`alerte caméra statique: ${maxStaticStreak} plans statiques consécutifs`);
  if (ruleViolations.length) parts.push(`violations détectées: ${ruleViolations.join(' ; ')}`);
  return parts.join(' · ');
}

/** Shot lengths from timeline.json: a long held shot is the first thing that kills the pace. */
export function timingMetrics(timeline: any): string {
  const fps = Number(timeline?.fps ?? 30);
  const shots = (Array.isArray(timeline?.shots) ? timeline.shots : []) as { id: string; len: number }[];
  if (!shots.length) return '(pas de timeline)';
  const secs = shots.map((s) => ({ id: String(s.id), s: Number(s.len ?? 0) / fps })).sort((a, b) => a.s - b.s);
  const median = secs[Math.floor(secs.length / 2)]!.s;
  const longest = secs[secs.length - 1]!;
  const over = secs.filter((x) => x.s > 6).map((x) => x.id);
  return `durée des plans : médiane ${median.toFixed(1)} s, la plus longue ${longest.s.toFixed(1)} s (${longest.id}) · ${over.length} plan(s) de plus de 6 s${over.length ? ` (${over.slice(0, 8).join(', ')})` : ''}`;
}

/**
 * Reads a review whose JSON does not parse, field by field. A reviewer that quotes a word of the script inside a text
 * ("il dit "pouces" trop tôt") or whose answer is cut by the output cap gives invalid JSON; the first real launch on
 * the 16 Nov engine lost a whole refused review that way and the video shipped unreviewed. Each value runs from its
 * key to the next known key, so quotes inside a text no longer matter.
 */
export function salvageReview(text: string): { pass?: boolean; summary?: string; issues: ReviewIssue[] } | null {
  const KEY = /"(pass|summary|issues|shot|severity|problem|fix)"\s*:/g;
  const marks: { key: string; start: number; end: number }[] = [];
  for (let m = KEY.exec(text); m; m = KEY.exec(text)) marks.push({ key: m[1]!, start: m.index, end: m.index + m[0].length });
  if (!marks.length) return null;
  const valueOf = (i: number): string => {
    let v = text.slice(marks[i]!.end, marks[i + 1]?.start ?? text.length).trim();
    v = v.replace(/```\s*$/, '').replace(/[\s,{}[\]]+$/, '');
    if (v.startsWith('"')) v = v.slice(1);
    if (v.endsWith('"') && !v.endsWith('\\"')) v = v.slice(0, -1);
    return v.replace(/\\n/g, ' ').replace(/\\"/g, '"').replace(/\s+/g, ' ').trim();
  };
  let pass: boolean | undefined;
  let summary: string | undefined;
  const issues: ReviewIssue[] = [];
  let cur: Partial<ReviewIssue> | null = null;
  const flush = () => {
    if (cur && ((cur.problem ?? '').length >= 12 || (cur.fix ?? '').length >= 12))
      issues.push({ shot: (cur.shot || 'GLOBAL').trim(), severity: cur.severity ?? 'warn', problem: cur.problem ?? '', fix: cur.fix ?? '' });
    cur = null;
  };
  marks.forEach((mk, i) => {
    const v = valueOf(i);
    if (mk.key === 'pass') pass = /^true\b/i.test(v) ? true : /^false\b/i.test(v) ? false : pass;
    else if (mk.key === 'summary') summary = v;
    else if (mk.key === 'issues') flush();
    else {
      const k = mk.key as 'shot' | 'severity' | 'problem' | 'fix';
      if (!cur || cur[k] !== undefined) {
        flush();
        cur = {};
      }
      if (k === 'severity') cur.severity = v.toLowerCase().startsWith('error') ? 'error' : 'warn';
      else cur[k] = v;
    }
  });
  flush();
  if (!issues.length && pass === undefined) return null;
  return { pass, summary, issues };
}

export function parseReviewJson(raw: string, model: string, episode: string): ReviewReport {
  const cleaned = raw.trim();
  const fenceMatch = /```(?:json)?\s*([\s\S]*?)```/i.exec(cleaned);
  const candidate = fenceMatch ? fenceMatch[1]!.trim() : cleaned;
  const firstBrace = candidate.indexOf('{');
  const lastBrace = candidate.lastIndexOf('}');
  const jsonSlice = firstBrace >= 0 && lastBrace > firstBrace ? candidate.slice(firstBrace, lastBrace + 1) : candidate;

  try {
    const parsed = JSON.parse(jsonSlice);
    const rawIssues = Array.isArray(parsed?.issues) ? parsed.issues : [];
    const issues: ReviewIssue[] = rawIssues
      .filter((it: any) => it && (it.problem || it.fix))
      .map((it: any) => ({
        shot: String(it.shot ?? 'GLOBAL').trim(),
        severity: (String(it.severity ?? 'warn').toLowerCase() === 'error' ? 'error' : 'warn') as 'error' | 'warn',
        problem: String(it.problem ?? '').trim(),
        fix: String(it.fix ?? '').trim(),
      }));
    const hasError = issues.some((i) => i.severity === 'error');
    const pass = typeof parsed?.pass === 'boolean' ? parsed.pass && !hasError : !hasError;
    return {
      ok: true,
      pass,
      model,
      episode,
      summary: String(parsed?.summary ?? (pass ? 'Réalisation cohérente.' : `${issues.length} problème(s) détecté(s).`)).trim(),
      issues,
      createdAt: new Date().toISOString(),
    };
  } catch {
    const got = salvageReview(cleaned);
    const hasError = !!got && got.issues.some((i) => i.severity === 'error');
    // a refusal whose blocking issue was cut off is not read as a pass: it stays unreadable, so the reviewer is asked again
    if (got && !(got.pass === false && !hasError)) {
      const pass = !hasError;
      return {
        ok: true,
        pass,
        model,
        episode,
        summary: got.summary || (pass ? 'Réalisation cohérente.' : `${got.issues.length} problème(s) détecté(s).`),
        issues: got.issues,
        createdAt: new Date().toISOString(),
        salvaged: true,
      };
    }
    return {
      ok: false,
      pass: true,
      model,
      episode,
      summary: `Réponse du reviewer non-JSON ignorée (${cleaned.slice(0, 140)})`,
      issues: [],
      createdAt: new Date().toISOString(),
      raw: cleaned.slice(0, 20_000),
    };
  }
}

/** The orientation the episode targets: read from episode.json or bible.json (default 16:9). */
function episodeAspect(_projectRoot: string, epRaw: string, epDir?: string): string {
  try {
    const a = String(JSON.parse(epRaw)?.aspect ?? '');
    if (a === '9:16' || a === '16:9') return a;
  } catch {
    /* fall through */
  }
  if (epDir) {
    try {
      const b = JSON.parse(readText(epDir, 'bible.json', 20_000) ?? '{}');
      if (b?.aspect === '9:16' || b?.aspect === '16:9') return b.aspect;
    } catch {
      /* fall through */
    }
  }
  return '16:9';
}

function readImagePart(file: string, mime: string): ImagePart | null {
  try {
    const st = fs.statSync(file);
    if (st.size > 0 && st.size <= 4_000_000) {
      return { mime, data: fs.readFileSync(file).toString('base64') };
    }
  } catch {
    /* missing file */
  }
  return null;
}

function loadPreviousReview(outDir: string): ReviewReport | null {
  try {
    const raw = fs.readFileSync(path.join(outDir, 'review.json'), 'utf8');
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object') return parsed as ReviewReport;
  } catch {
    /* none */
  }
  return null;
}

function chunkArray<T>(items: T[], size: number): T[][] {
  if (!items.length) return [];
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

/** How long one call to the reviewer may take. One episode waited 31 minutes for a review, another 10 (2026-11-16). */
export const reviewTimeoutMs = (): number => {
  const n = Number(process.env.WHATIF_REVIEW_TIMEOUT_MS);
  return Number.isFinite(n) && n > 0 ? n : 240_000;
};
export class ReviewTimeout extends Error {}
/** provider.chat with a deadline of its own: the call is aborted and, should the provider ignore that, left behind. */
export async function chatWithin(provider: Provider, messages: ChatMessage[], signal: AbortSignal | undefined, ms = reviewTimeoutMs()): Promise<string> {
  const ctl = new AbortController();
  const onUser = () => ctl.abort();
  if (signal?.aborted) ctl.abort();
  signal?.addEventListener('abort', onUser, { once: true });
  let timer: NodeJS.Timeout | undefined;
  let late = false;
  const deadline = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      late = true;
      ctl.abort();
      reject(new ReviewTimeout(`no answer from the reviewer in ${Math.round(ms / 1000)} s`));
    }, ms);
  });
  try {
    const call = provider.chat({ messages, tools: [], signal: ctl.signal }).then((r) => r.text);
    call.catch(() => undefined);
    return await Promise.race([call, deadline]);
  } catch (err) {
    if (late && !signal?.aborted) throw new ReviewTimeout(`no answer from the reviewer in ${Math.round(ms / 1000)} s`);
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
    signal?.removeEventListener('abort', onUser);
  }
}

async function askReviewerWithFallback(
  provider: Provider,
  systemPrompt: string,
  userPrompt: string,
  images: ImagePart[],
  slug: string,
  signal?: AbortSignal,
): Promise<{ report: ReviewReport; sawImages: boolean }> {
  const late = (): ReviewReport => ({
    ok: false,
    pass: true,
    model: provider.model,
    episode: slug,
    summary: `Reviewer indisponible : aucune réponse en ${Math.round(reviewTimeoutMs() / 60_000)} min, deux essais.`,
    issues: [],
    createdAt: new Date().toISOString(),
    timedOut: true,
  });
  // a reviewer that does not answer is asked once more, then the chain goes on unreviewed instead of waiting
  const ask = async (msgs: ChatMessage[]): Promise<ReviewReport> => {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return parseReviewJson(await chatWithin(provider, msgs, signal), provider.model, slug);
      } catch (err) {
        if (!(err instanceof ReviewTimeout)) throw err;
      }
    }
    return late();
  };

  let sawImages = images.length > 0;
  let sent: ChatMessage[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt, ...(images.length ? { images } : {}) },
  ];
  let report: ReviewReport;
  try {
    report = await ask(sent);
  } catch (err) {
    if (!images.length) throw err;
    // Text-only reviewer fallback: audit the code + script + DIRECTOR.md without images
    sawImages = false;
    sent = [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }];
    report = await ask(sent);
  }
  if (!report.ok && !report.timedOut) report = await ask(sent);
  return { report, sawImages };
}

function mergeBatchReports(reports: ReviewReport[], model: string, episode: string): ReviewReport {
  if (reports.length === 1) return reports[0]!;
  const okReports = reports.filter((r) => r.ok);
  if (!okReports.length) return reports[0]!;

  const seen = new Set<string>();
  const mergedIssues: ReviewIssue[] = [];
  for (const r of okReports) {
    for (const iss of r.issues) {
      const key = `${iss.shot.toUpperCase()}::${iss.severity}::${iss.problem.slice(0, 80).toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      mergedIssues.push(iss);
    }
  }
  mergedIssues.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === 'error' ? -1 : 1));
  const cappedIssues = mergedIssues.slice(0, 10);
  const hasError = cappedIssues.some((i) => i.severity === 'error');
  const pass = okReports.every((r) => r.pass) && !hasError;
  const summaries = okReports.map((r, idx) => `Lot ${idx + 1}/${okReports.length}: ${r.summary}`).join(' · ');

  return {
    ok: true,
    pass,
    model,
    episode,
    summary: summaries,
    issues: cappedIssues,
    createdAt: new Date().toISOString(),
  };
}

export type ReviewOptions = {
  final?: boolean;
  round?: number;
  mode?: 'auto' | 'global' | 'delta';
};

function loadStagedShots(epDir: string, ep: any): any[] {
  const shots = Array.isArray(ep?.shots) ? ep.shots.map((s: any) => ({ ...s })) : [];
  let stageFileCards: Record<string, any> = {};
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(epDir, 'shots.stage.json'), 'utf8'));
    stageFileCards = (raw?.cards && typeof raw.cards === 'object' ? raw.cards : raw) ?? {};
  } catch {
    /* optional shots.stage.json */
  }
  const byId = new Map<string, any>();
  for (const s of shots) {
    if (s?.id) byId.set(String(s.id).toUpperCase(), s);
  }
  for (const [k, card] of Object.entries(stageFileCards)) {
    const id = k.toUpperCase();
    if (!/^S\d+$/.test(id) || !card || typeof card !== 'object') continue;
    if (byId.has(id)) byId.get(id)!.stage = byId.get(id)!.stage ?? card;
    else {
      const entry = { id, stage: card };
      shots.push(entry);
      byId.set(id, entry);
    }
  }
  return shots.filter((s: any) => s?.stage);
}

function collectRefPathsForShots(outDir: string, epSrc: EpisodeShotsBundle, shotIds?: string[]): string[] {
  const refsDir = path.join(outDir, 'refs');
  if (!fs.existsSync(refsDir)) return [];
  const wanted = new Set<string>();
  if (shotIds?.length) {
    for (const id of shotIds) {
      const sp = epSrc.shots.get(id);
      if (!sp) continue;
      for (const c of sp.characters) wanted.add(`char_${c}.png`);
      if (sp.place) wanted.add(`place_${sp.place}.png`);
    }
  }
  const files = fs.readdirSync(refsDir).filter((n) => /\.png$/i.test(n)).sort((a, b) => {
    const aChar = a.startsWith('char_') ? 0 : 1;
    const bChar = b.startsWith('char_') ? 0 : 1;
    return aChar !== bChar ? aChar - bChar : a.localeCompare(b);
  });
  const selected = wanted.size ? files.filter((f) => wanted.has(f)) : files;
  return selected.map((f) => path.join(refsDir, f));
}

export async function reviewEpisodeBoard(
  provider: Provider,
  projectRoot: string,
  slug: string,
  signal?: AbortSignal,
  opts: ReviewOptions = {},
): Promise<ReviewReport> {
  const epDir = path.join(projectRoot, 'src', 'episodes', slug);
  const outDir = path.join(projectRoot, 'out', slug);
  const epRaw = readText(epDir, 'episode.json', 3_000_000) ?? '{}';
  const directorRaw = readText(epDir, 'DIRECTOR.md', 200_000) ?? '(aucun DIRECTOR.md)';
  const bibleRaw = readText(epDir, 'bible.json', 20_000) ?? '';
  const bibleRes = readEpisodeBible(epDir);
  let ep: any = {};
  try {
    ep = JSON.parse(epRaw);
  } catch {
    ep = {};
  }
  const stagedShots = loadStagedShots(epDir, ep);
  const cardsBlock = (ids?: string[]): string => {
    const text = formatStageCards(stagedShots, ids);
    return text ? `\n\n## Fiches de mise en scène (ce que CHAQUE plan doit montrer — juge l'image contre sa fiche)\n${text.slice(0, 16_000)}` : '';
  };
  const epSrc = readEpisodeShots(epDir);
  const aspect = episodeAspect(projectRoot, epRaw, epDir);

  let timeline: any = null;
  try {
    timeline = JSON.parse(readText(epDir, 'timeline.json', 600_000) ?? 'null');
  } catch {
    timeline = null;
  }
  const epLang = detectEpisodeLanguage(ep);
  const langUpper = epLang.toUpperCase();

  const shotTimings = new Map<string, { len: number; key: number }>();
  for (const s of Array.isArray(timeline?.shots) ? timeline.shots : []) {
    if (s?.id) {
      const len = Number(s.len ?? 60);
      const key = Math.min(Math.max(0, Number(s.key ?? 18)), Math.max(0, len - 1));
      shotTimings.set(String(s.id).toUpperCase(), { len, key });
    }
  }

  const scriptLines = (Array.isArray(ep?.script) ? ep.script : [])
    .map((b: any, i: number) => `${i + 1}. ${String(b?.text ?? '').trim()}`)
    .join('\n');

  const formatShotBreakdown = (filterIds?: Set<string>) =>
    (Array.isArray(ep?.shots) ? ep.shots : [])
      .filter((s: any) => !filterIds || filterIds.has(String(s?.id ?? '').toUpperCase()))
      .map((s: any) => {
        const id = String(s?.id ?? '').toUpperCase();
        const tm = shotTimings.get(id);
        const tmTag = tm ? `, ${(tm.len / 30).toFixed(1)}s (${tm.len}f)` : '';
        return `- ${s.id} [${s.size ?? 'medium'}, set=${s.set ?? 'custom'}${tmTag}]: "${s.note ?? ''}"`;
      })
      .join('\n');

  const boardDir = path.join(outDir, 'board');
  const boardStillIds = fs.existsSync(boardDir)
    ? sortShotIds(
        fs
          .readdirSync(boardDir)
          .filter((n) => /^S\d+\.png$/i.test(n))
          .map((n) => n.replace(/\.png$/i, '').toUpperCase()),
      )
    : [];

  const sheetsDir = path.join(outDir, 'sheets');
  const sheetFiles = fs.existsSync(sheetsDir)
    ? fs
        .readdirSync(sheetsDir)
        .filter((n) => /^sheet-\d+\.jpe?g$/i.test(n))
        .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }))
    : [];

  // Determine whether this review is Round 1 (Global 100% Audit) or Round 2+ (Targeted High-Res Delta Re-check)
  const prevReview = loadPreviousReview(outDir);
  const prevErrors = prevReview?.ok && !prevReview.pass ? prevReview.issues.filter((i) => i.severity === 'error') : [];
  const prevRound = typeof prevReview?.round === 'number' ? prevReview.round : 0;
  const round = opts.round ?? (prevReview?.ok ? prevRound + 1 : 1);

  const requestedMode = opts.mode ?? 'auto';
  const prevCreatedMs = prevReview?.createdAt ? Date.parse(prevReview.createdAt) || 0 : 0;
  const flaggedShots = sortShotIds(prevErrors.flatMap((i) => extractShotIds(i.shot)));
  const modifiedShots: string[] = [];
  if (prevCreatedMs > 0) {
    for (const [id, mt] of epSrc.shotMtimesMs) {
      if (mt > prevCreatedMs + 500) modifiedShots.push(id);
    }
    for (const id of boardStillIds) {
      try {
        const st = fs.statSync(path.join(boardDir, `${id}.png`));
        if (st.mtimeMs > prevCreatedMs + 500) modifiedShots.push(id);
      } catch {
        /* ignore */
      }
    }
  }
  const deltaTargetShots = sortShotIds([...flaggedShots, ...modifiedShots]);
  const useDeltaMode =
    requestedMode === 'delta' ||
    (requestedMode === 'auto' &&
      Boolean(
        prevReview?.ok &&
          deltaTargetShots.length > 0 &&
          ((!prevReview.pass && (prevReview.blocking || prevReview.reopened) && prevErrors.length > 0) || prevReview.pass),
      ));

  const sizeDrift = directorSizeMismatches(directorRaw, ep);
  const autoMetrics = [
    diffusionMetrics(epSrc, bibleRes.bible, epLang),
    timingMetrics(timeline),
    sizeDrift.length ? `écarts de taille DIRECTOR.md ↔ episode.json (${sizeDrift.length}) : ${sizeDrift.join(', ')}` : '',
  ]
    .filter(Boolean)
    .join('\n');

  let finalReport: ReviewReport;
  let sawImagesOverall = false;
  let batchCount = 1;

  if (useDeltaMode && deltaTargetShots.length > 0) {
    // ── MODE 2: High-Resolution Targeted Delta Re-check (Rounds 2+) ──
    const targetChunks = chunkArray(deltaTargetShots, MAX_REVIEW_IMAGES);
    batchCount = targetChunks.length;
    const chunkReports: ReviewReport[] = [];

    for (let cIdx = 0; cIdx < targetChunks.length; cIdx++) {
      const chunkIds = targetChunks[cIdx]!;
      const chunkSet = new Set(chunkIds);
      const images: ImagePart[] = [];
      for (const id of chunkIds) {
        const img = readImagePart(path.join(boardDir, `${id}.png`), 'image/png');
        if (img) images.push(img);
      }
      if (!images.length) {
        for (const n of sheetFiles.slice(0, MAX_REVIEW_IMAGES)) {
          const img = readImagePart(path.join(sheetsDir, n), 'image/jpeg');
          if (img) images.push(img);
        }
      }
      if (!images.length) {
        const strip = readImagePart(path.join(outDir, 'storyboard.jpg'), 'image/jpeg');
        if (strip) images.push(strip);
      }
      // Attach relevant ref sheets when image slots remain in delta review
      for (const refFile of collectRefPathsForShots(outDir, epSrc, chunkIds)) {
        if (images.length >= MAX_REVIEW_IMAGES) break;
        const img = readImagePart(refFile, 'image/png');
        if (img) images.push(img);
      }

      const relevantPrevErrors = prevErrors.filter((iss) => {
        const ids = extractShotIds(iss.shot);
        return !ids.length || ids.some((id) => chunkSet.has(id));
      });
      const prevIssuesList = (relevantPrevErrors.length ? relevantPrevErrors : prevErrors)
        .map((iss) => `- [${iss.shot}] Problème signalé : ${iss.problem}\n  Correctif demandé : ${iss.fix}`)
        .join('\n');
      const wordMarks = shotWordFrames(projectRoot, slug, chunkIds);

      const deltaUserPrompt = [
        `# Contre-visite ciblée (Tour ${round}) de l'épisode "${slug}" (orientation: ${aspect}, langue: ${langUpper})`,
        targetChunks.length > 1 ? `Lot ciblé ${cIdx + 1}/${targetChunks.length} — plans vérifiés : ${chunkIds.join(', ')}` : `Plans ciblés vérifiés : ${chunkIds.join(', ')}`,
        `## 1. Problèmes bloquants signalés au tour précédent (à vérifier point par point)\n${prevIssuesList}`,
        `## 2. Script complet (narration ${langUpper} — contexte)\n${scriptLines.slice(0, 25_000)}`,
        `## 3. Découpage des plans ciblés (episode.json.shots)\n${formatShotBreakdown(chunkSet) || '(aucune entrée)'}`,
        wordMarks.length ? `## 4. Repères mot@frame sur l'horloge propre de chaque plan ciblé\n${wordMarks.join('\n')}` : '',
        bibleRaw ? `## 5. Bible de continuité (bible.json)\n${bibleRaw.slice(0, 6_000)}` : '',
        `## 6. Note de réalisation (DIRECTOR.md)\n${directorRaw.slice(0, 14_000)}${cardsBlock(chunkIds)}`,
        `## 7. Specs JSON complètes des plans ciblés (shots/S*.json)\n${summarizeShotPrompts(epSrc, { onlyShots: chunkIds, shotTimings, perShotChars: 4_000 })}`,
        images.length
          ? `\n${images.length} image(s) pleine résolution jointe(s) (${chunkIds.join(', ')} + fiches refs éventuelles). Vérifie si les problèmes ci-dessus sont résolus et renvoie ton verdict JSON.`
          : `\nVérifie sur les specs JSON ci-dessus si les problèmes bloquants signalés sur ${chunkIds.join(', ')} sont résolus, puis renvoie ton verdict JSON.`,
      ]
        .filter(Boolean)
        .join('\n\n');

      const { report, sawImages } = await askReviewerWithFallback(
        provider,
        deltaReviewerSystemPrompt(epLang),
        deltaUserPrompt,
        images,
        slug,
        signal,
      );
      if (sawImages) sawImagesOverall = true;
      chunkReports.push(report);
    }

    finalReport = mergeBatchReports(chunkReports, provider.model, slug);
    finalReport.mode = 'delta';
    finalReport.targetShots = deltaTargetShots;
  } else {
    // ── MODE 1: Global 100% Audit across ALL contact sheets in chunks of <= 6 sheets ──
    const sheetChunks = sheetFiles.length ? chunkArray(sheetFiles, MAX_REVIEW_IMAGES) : [[]];
    batchCount = sheetChunks.length;
    const perSheet = aspect === '9:16' ? 8 : 9;
    const orderedAllShots = boardStillIds.length ? boardStillIds : epSrc.exportedShotIds;
    const chunkReports: ReviewReport[] = [];

    for (let cIdx = 0; cIdx < sheetChunks.length; cIdx++) {
      const chunkSheets = sheetChunks[cIdx]!;
      const images: ImagePart[] = [];
      for (const n of chunkSheets) {
        const img = readImagePart(path.join(sheetsDir, n), 'image/jpeg');
        if (img) images.push(img);
      }
      if (!images.length && cIdx === 0) {
        const strip = readImagePart(path.join(outDir, 'storyboard.jpg'), 'image/jpeg');
        if (strip) images.push(strip);
      }
      // On the first batch, append up to 2 ref sheets if image slots remain
      if (cIdx === 0 && images.length < MAX_REVIEW_IMAGES) {
        const refPaths = collectRefPathsForShots(outDir, epSrc);
        for (const refFile of refPaths.slice(0, Math.min(2, MAX_REVIEW_IMAGES - images.length))) {
          const img = readImagePart(refFile, 'image/png');
          if (img) images.push(img);
        }
      }

      let batchShots: string[] | undefined;
      if (sheetChunks.length > 1 && orderedAllShots.length > cIdx * MAX_REVIEW_IMAGES * perSheet) {
        const startShotIdx = cIdx * MAX_REVIEW_IMAGES * perSheet;
        const endShotIdx = startShotIdx + chunkSheets.length * perSheet;
        const slice = orderedAllShots.slice(startShotIdx, endShotIdx);
        if (slice.length) batchShots = slice;
      }

      const sheetRangeNote =
        sheetChunks.length > 1
          ? ` (lot ${cIdx + 1}/${sheetChunks.length} : planches ${cIdx * MAX_REVIEW_IMAGES + 1} à ${cIdx * MAX_REVIEW_IMAGES + chunkSheets.length} sur ${sheetFiles.length}${batchShots ? `, couvrant ${batchShots[0]}..${batchShots[batchShots.length - 1]}` : ''})`
          : '';

      const userPrompt = [
        `# Audit indépendant de l'épisode "${slug}" (orientation: ${aspect}, langue: ${langUpper}, defaultSet: ${JSON.stringify(ep?.defaultSet ?? '')})${sheetRangeNote}`,
        `## 1. Script complet (narration ${langUpper})\n${scriptLines.slice(0, 25_000)}`,
        `## 2. Découpage des plans (episode.json.shots)\n${formatShotBreakdown().slice(0, 25_000)}`,
        bibleRaw ? `## 3. Bible de continuité (bible.json)\n${bibleRaw.slice(0, 8_000)}` : '',
        `## 4. Note de réalisation (DIRECTOR.md)\n${directorRaw.slice(0, 15_000)}${cardsBlock(batchShots)}`,
        `## 5. Specs JSON des plans (shots/S*.json)\n${summarizeShotPrompts(epSrc, { onlyShots: batchShots, shotTimings, perShotChars: 1_800 })}`,
        `## 6. Mesures automatiques (bible, shots/*.json et timeline.json)\n${autoMetrics}`,
        images.length
          ? `\n${images.length} image(s) jointe(s)${sheetRangeNote} (planches de contact + fiches refs éventuelles). Compare chaque vignette générée et ses prompts (imagePrompt, videoPrompt, sfxHint, onScreenText) au récit, à bible.json et à DIRECTOR.md, puis renvoie ton verdict JSON.`
          : `\nInspecte la cohérence entre le récit, bible.json, DIRECTOR.md et les specs JSON des plans, puis renvoie ton verdict JSON.`,
      ]
        .filter(Boolean)
        .join('\n\n');

      const { report, sawImages } = await askReviewerWithFallback(
        provider,
        reviewerSystemPrompt(epLang),
        userPrompt,
        images,
        slug,
        signal,
      );
      if (sawImages) sawImagesOverall = true;
      chunkReports.push(report);
    }

    finalReport = mergeBatchReports(chunkReports, provider.model, slug);
    finalReport.mode = 'global';
  }

  finalReport.round = round;
  finalReport.batches = batchCount;
  finalReport.sawImages = sawImagesOverall;
  // The same shot refused three reviews in a row will not be fixed by a fourth: on 2026-11-13 three projects went to
  // round 8 ("5e signalement, correctif toujours non appliqué") and ended refused anyway. Stop, ship, list the defects.
  const strikes = nextStrikes(prevReview?.ok && !prevReview.pass ? prevReview.strikes : undefined, finalReport);
  finalReport.strikes = strikes;
  const gaveUpOn = finalReport.ok && !finalReport.pass ? Object.keys(strikes).filter((id) => strikes[id]! >= MAX_STRIKES) : [];
  if (gaveUpOn.length) finalReport.gaveUpOn = gaveUpOn;
  const final = Boolean(opts.final) || gaveUpOn.length > 0;
  finalReport.blocking = finalReport.ok && !finalReport.pass && !final;
  if (final && finalReport.ok && !finalReport.pass) {
    const file = writeUnresolvedDefects(projectRoot, slug, finalReport);
    if (file) finalReport.defectsFile = file;
  }

  try {
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(path.join(outDir, 'review.json'), JSON.stringify(finalReport, null, 2) + '\n');
  } catch {
    /* ignore write error */
  }

  return finalReport;
}

/**
 * The cards read before any shot is coded: text only, one short call. A weak picture is cheaper to fix as a sentence
 * than as code, stills and a review round. Advisory: it never blocks, it runs once per set of cards, and an unreadable
 * answer is silence, not a verdict. Returns the note for the agent, or null when there is nothing to say.
 */
export async function reviewStageCards(provider: Provider, projectRoot: string, slug: string, signal?: AbortSignal): Promise<string | null> {
  const epDir = path.join(projectRoot, 'src', 'episodes', slug);
  let ep: any;
  try {
    ep = JSON.parse(readText(epDir, 'episode.json', 3_000_000) ?? 'null');
  } catch {
    return null;
  }
  const rawShots: any[] = Array.isArray(ep?.shots) ? ep.shots : [];
  const shots = loadStagedShots(epDir, ep);
  if (!rawShots.length || shots.length < rawShots.length) return null;
  const cards = formatStageCards(shots);
  const sig = String(cards.length) + ':' + [...cards].reduce((h, c) => (h * 31 + c.charCodeAt(0)) >>> 0, 7).toString(36);
  const marker = path.join(projectRoot, 'out', slug, 'cards-review.json');
  if (fs.existsSync(marker)) return null;
  if (shots.some((s) => shotCoded(epDir, String(s.id)))) return null;
  const script = (Array.isArray(ep.script) ? ep.script : []).map((b: any) => String(b?.text ?? '')).join(' ').slice(0, 9_000);
  const system = `Tu es réalisateur d'animation 2D diffusion. On te donne le script d'un épisode court et la fiche de mise en scène de chaque plan, AVANT de générer les prompts images. Dis seulement ce qui donnera une mauvaise image : un plan sans sujet clair ; une action impossible à représenter sur un still fixe ; un personnage qui change de lieu sans raison entre deux plans voisins ; un objet manquant ou incohérent avec le script ; trois plans de suite au même cadrage et même angle ; un dialogue filmé sans champ/contrechamp ; un format 9:16 mal composé. Ne commente pas ce qui va bien. Réponds en JSON strict : {"issues":[{"shot":"S4","problem":"…","fix":"…"}]} avec 8 remarques au plus, les plus graves d'abord ; {"issues":[]} si tout est bon.`;
  const user = `## Script\n${script}\n\n## Fiches (format ${ep.aspect ?? '16:9'}, ${shots.length} plans)\n${cards.slice(0, 20_000)}`;
  let issues: { shot: string; problem: string; fix: string }[] = [];
  try {
    const text = await chatWithin(provider, [{ role: 'system', content: system }, { role: 'user', content: user }], signal);
    const m = text.match(/\{[\s\S]*\}/);
    const parsed = m ? JSON.parse(m[0]) : null;
    if (!parsed || !Array.isArray(parsed.issues)) return null;
    issues = parsed.issues.filter((i: any) => i && typeof i.problem === 'string' && i.problem.trim()).slice(0, 8).map((i: any) => ({ shot: String(i.shot ?? '?'), problem: String(i.problem), fix: String(i.fix ?? '') }));
  } catch {
    return null;
  }
  try {
    fs.mkdirSync(path.dirname(marker), { recursive: true });
    fs.writeFileSync(marker, JSON.stringify({ sig, model: provider.model, createdAt: new Date().toISOString(), issues }, null, 2) + '\n');
  } catch {
    /* the note still goes to the agent */
  }
  if (!issues.length) return null;
  return [
    `[REVIEWER — LECTURE DES FICHES, avant les prompts] ${issues.length} fiche(s) donneraient une image faible. Corrige-les avec stage_shots (seulement ces plans) AVANT d'écrire les JSON ; c'est un conseil, rien n'est bloqué :`,
    ...issues.map((i) => `- [${i.shot}] ${i.problem}${i.fix ? `\n  → ${i.fix}` : ''}`),
  ].join('\n');
}

export const MAX_STRIKES = 3;
const shotIdsOf = (shot: unknown): string[] => [...new Set((String(shot ?? '').toUpperCase().match(/\bS\d+\b/g) ?? []))];
/** Refusals in a row per shot: +1 for each shot with an error in this review, back to nothing for the others. */
export function nextStrikes(prev: Record<string, number> | undefined, report: { ok: boolean; pass: boolean; issues: { shot?: unknown; severity?: string }[] }): Record<string, number> {
  if (!report.ok || report.pass) return {};
  const out: Record<string, number> = {};
  for (const i of report.issues) {
    if (i.severity !== 'error') continue;
    for (const id of shotIdsOf(i.shot)) out[id] = Math.max(out[id] ?? 0, (prev?.[id] ?? 0) + 1);
  }
  return out;
}

/**
 * The shots the last review refused, when nothing that could fix them has changed since (null otherwise).
 * Asking the reviewer again would cost a paid call and give the same verdict (z8nvytvrw5c, 2026-11-11: 16 minutes for nothing).
 * Changes that count: a refused shot's own file, any shared file of the episode, DIRECTOR.md, episode.json.
 */
export function refusedUnchanged(projectRoot: string, slug: string): string[] | null {
  const prev = loadPreviousReview(path.join(projectRoot, 'out', slug));
  if (!prev?.ok || prev.pass || !(prev.blocking || prev.reopened) || !prev.createdAt) return null;
  const since = (Date.parse(prev.createdAt) || 0) + 500;
  const refused = sortShotIds(prev.issues.filter((i) => i.severity === 'error').flatMap((i) => extractShotIds(i.shot)));
  const epDir = path.join(projectRoot, 'src', 'episodes', slug);
  const newer = (file: string): boolean => {
    try {
      return fs.statSync(file).mtimeMs > since;
    } catch {
      return false;
    }
  };
  const walk = (dir: string, depth = 0): boolean => {
    let names: string[] = [];
    try {
      names = fs.readdirSync(dir);
    } catch {
      return false;
    }
    for (const n of names) {
      const f = path.join(dir, n);
      let st: fs.Stats;
      try {
        st = fs.statSync(f);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        if (depth < 2 && walk(f, depth + 1)) return true;
        continue;
      }
      // a modular shot file only matters when that shot was refused
      const m = /^(S\d+)\.json$/i.exec(n);
      if (m && path.basename(dir) === 'shots' && !refused.includes(m[1]!.toUpperCase())) continue;
      if (/\.(json|md)$/i.test(n) && !/^(timeline|words)\.json$/.test(n) && st.mtimeMs > since) return true;
    }
    return false;
  };
  if (walk(epDir)) return null;
  return refused.length ? refused : null;
}

/** Written next to the video when the last review round still refused it: the video ships, with this list. */
export function writeUnresolvedDefects(projectRoot: string, slug: string, report: ReviewReport): string | null {
  const errors = report.issues.filter((i) => i.severity === 'error');
  if (report.pass || !errors.length) return null;
  const rel = `out/${slug}/DEFAUTS-NON-CORRIGES.md`;
  const body = [
    `# Défauts non corrigés — ${slug}`,
    '',
    `Le Reviewer (${report.model}) a refusé la planche au dernier tour (tour ${report.round ?? '?'}). La vidéo est rendue quand même : voici ce qui n'est pas corrigé.`,
    '',
    `Résumé : ${report.summary}`,
    '',
    ...errors.map((e) => `- **${e.shot}** — ${e.problem}\n  → ${e.fix}`),
    '',
  ].join('\n');
  try {
    fs.mkdirSync(path.join(projectRoot, 'out', slug), { recursive: true });
    fs.writeFileSync(path.join(projectRoot, rel), body);
    return rel;
  } catch {
    return null;
  }
}

/**
 * A review closed with defects (three refusals of a shot, or no round left) stays closed for the run that made it.
 * When the user writes again on that project ("fix S5 and S38, then render"), the fixes deserve a look: the counts
 * start again and the next board is reviewed, on the changed shots only. The render is not held back meanwhile.
 * Returns the shots that were left refused, or null when there is nothing to reopen.
 */
export function reopenReview(projectRoot: string, slug: string): string[] | null {
  const file = path.join(projectRoot, 'out', slug, 'review.json');
  try {
    const report = JSON.parse(fs.readFileSync(file, 'utf8')) as ReviewReport;
    if (!report?.ok || report.pass || report.blocking || report.reopened) return null;
    const refused = sortShotIds(report.issues.filter((i) => i.severity === 'error').flatMap((i) => extractShotIds(i.shot)));
    if (!refused.length) return null;
    const { gaveUpOn: _g, strikes: _s, ...rest } = report;
    fs.writeFileSync(file, JSON.stringify({ ...rest, reopened: true }, null, 2) + '\n');
    return refused;
  } catch {
    return null;
  }
}

/**
 * The agent asked for the render again and again without touching a refused shot: it will not fix them. The review is
 * closed as after three refusals of a shot: the video ships with the list of its defects. Returns the defects file.
 */
export function giveUpReview(projectRoot: string, slug: string): string | null {
  const file = path.join(projectRoot, 'out', slug, 'review.json');
  try {
    const report = JSON.parse(fs.readFileSync(file, 'utf8')) as ReviewReport;
    if (!report?.ok || report.pass || !report.blocking) return null;
    const refused = sortShotIds(report.issues.filter((i) => i.severity === 'error').flatMap((i) => extractShotIds(i.shot)));
    const defectsFile = writeUnresolvedDefects(projectRoot, slug, report) ?? undefined;
    fs.writeFileSync(file, JSON.stringify({ ...report, blocking: false, gaveUpOn: refused, defectsFile }, null, 2) + '\n');
    return defectsFile ?? `out/${slug}/review.json`;
  } catch {
    return null;
  }
}

/** A board that was not reviewed cannot stay refused: the reviewer failed, is off, or has no round left. */
export function clearReviewBlock(projectRoot: string, slug: string): void {
  const file = path.join(projectRoot, 'out', slug, 'review.json');
  try {
    const report = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (report.blocking) fs.writeFileSync(file, JSON.stringify({ ...report, blocking: false }, null, 2) + '\n');
  } catch {
    /* no review yet */
  }
}

export function formatReviewFeedback(report: ReviewReport, slug: string, wordFrames: string[] = []): string {
  if (!report.ok && report.timedOut) {
    return `[reviewer indépendant · ${report.model}] Audit de "${slug}" : NON VÉRIFIÉ. ${report.summary}\nContinue la chaîne : vérifie toi-même les planches de contact.`;
  }
  if (!report.ok) {
    return `[reviewer indépendant · ${report.model}] Audit de "${slug}" : NON VÉRIFIÉ (réponse illisible, deux essais). ${report.summary}\nContinue la chaîne : vérifie toi-même les planches de contact.`;
  }
  const errors = report.issues.filter((i) => i.severity === 'error');
  const warns = report.issues.filter((i) => i.severity === 'warn');
  const affectedShots = sortShotIds(errors.flatMap((i) => extractShotIds(i.shot)));
  const modeLabel =
    report.mode === 'delta'
      ? `Contre-visite ciblée (tour ${report.round ?? 2}${report.targetShots?.length ? ` · ${report.targetShots.join(', ')}` : ''})`
      : `Audit global${report.batches && report.batches > 1 ? ` (${report.batches} lots, 100% des planches)` : ''}`;

  const lines: string[] = [
    `[reviewer indépendant · ${report.model}] ${modeLabel} de "${slug}" : ${report.pass ? 'VALIDÉ ✓' : 'CORRECTIONS REQUISES ✗'}`,
    `Résumé : ${report.summary}`,
    ...(report.sawImages === false ? ["(Ce modèle n'accepte pas les images : verdict établi sur les specs JSON seules.)"] : []),
  ];
  if (errors.length) {
    lines.push(`\nProblèmes bloquants à corriger plan par plan (avec edit_shot) :`);
    for (const err of errors) {
      lines.push(`- [${err.shot}] Problème : ${err.problem}\n  → Correctif : ${err.fix}`);
    }
    if (wordFrames.length) lines.push(`\nMoment de chaque mot dans ces plans (mot@image, horloge du plan) pour caler les actions :\n${wordFrames.join('\n')}`);
    if (report.defectsFile) {
      lines.push(
        `\n⚠ ${report.gaveUpOn?.length ? `ARRÊT DE LA REVUE : ${report.gaveUpOn.join(', ')} refusé(s) ${MAX_STRIKES} fois de suite, une revue de plus ne changerait rien.` : 'DERNIER TOUR :'} il n'y a plus de contre-visite. Le rendu n'est plus bloqué, mais ces défauts restent. Corrige ce que tu peux d'abord ; puis, quoi que tu fasses, termine ton message final par « Vidéo livrée avec ${affectedShots.length || errors.length} défaut(s) non corrigé(s) » et cite-les. La liste est dans ${report.defectsFile}.`,
      );
    }
    const rerunArgs = affectedShots.length ? `, args: ${JSON.stringify(affectedShots)}` : '';
    lines.push(
      `\nApplique ces corrections plan par plan avec edit_shot { episode: "${slug}", id: "S...", imagePrompt: "...", ... } (sans toucher aux autres plans déjà validés), relance uniquement les plans modifiés via whatif_run { episode: "${slug}", step: "stills"${rerunArgs} } puis whatif_run { episode: "${slug}", step: "board" }, et enchaîne ensuite clips -> render.`,
    );
  } else if (warns.length) {
    lines.push(`\nRemarques mineures (non bloquantes) :`);
    for (const w of warns) {
      lines.push(`- [${w.shot}] ${w.problem} (${w.fix})`);
    }
  }
  return lines.join('\n');
}
