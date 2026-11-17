#!/usr/bin/env node
// Everything a real 100% diffusion run needs that only the machine itself can verify. Run before the first episode:
//   docker compose exec studio node pipeline/preflight.mjs
// Checks binaries, Python packages, :8005 server, Image/Video diffusion provider config, and LLM Agent + Reviewer models.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const load = (rel) => import(path.join(ROOT, 'dist', rel));
const { loadConfig } = await load('orchestrator/config.js');
const { Db } = await load('orchestrator/db.js');
const { resolveMediaConfig } = await load('orchestrator/media-providers.js');
const { resolveProvider, resolveReviewProvider, reviewProviderConfig } = await load('orchestrator/providers/index.js');

let failed = 0;
const line = (state, what, detail = '') => {
  if (state === 'ÉCHEC') failed++;
  console.log(`${state.padEnd(9)} ${what}${detail ? ` : ${detail}` : ''}`);
};
const has = (cmd, args) => {
  try {
    return execFileSync(cmd, args, { stdio: 'pipe', encoding: 'utf8' });
  } catch {
    return null;
  }
};

// 32x32 PNG test patch for Reviewer vision check
const PIXEL =
  'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR4nO3NMQEAAAjDMMC/2EnABHypgKaT1GfzegcAAAAAAAAAAACHLaaNAxBlrinYAAAAAElFTkSuQmCC';
const ask = async (provider, images) => {
  const messages = [
    {
      role: 'user',
      content: 'Réponds par le seul mot OK.',
      ...(images ? { images: [{ mime: 'image/png', data: PIXEL }] } : {}),
    },
  ];
  const started = Date.now();
  const res = await provider.chat({ messages, tools: [], signal: AbortSignal.timeout(90_000) });
  return { text: res.text.trim().slice(0, 40), ms: Date.now() - started, finish: res.finishReason };
};

const config = loadConfig();
const db = new Db(config.dbPath);

// ---------------------------------------------------------------- pipeline scripts (self-contained)
const requiredScripts = [
  'source_import.py',
  'voice_fish.py',
  'shotlist.py',
  'plan_run.py',
  'media_client.py',
  'image_gen.py',
  'board_run.py',
  'board_sheets.py',
  'video_gen.py',
  'assemble_run.py',
  'master_run.py',
  'motiongrid_run.py',
  'upscale_run.py',
];
const missingScripts = requiredScripts.filter((f) => !fs.existsSync(path.join(ROOT, 'pipeline', f)));
line(missingScripts.length ? 'ÉCHEC' : 'OK', 'scripts pipeline diffusion', missingScripts.length ? `manquants : ${missingScripts.join(', ')}` : `${requiredScripts.length} scripts présents`);

// ---------------------------------------------------------------- the machine
const ffVer = has('ffmpeg', ['-version']);
line(ffVer ? 'OK' : 'ÉCHEC', 'ffmpeg', ffVer ? ffVer.split('\n')[0] : 'binaire introuvable');
const ffEnc = has('ffmpeg', ['-hide_banner', '-encoders']) || '';
line(ffEnc.includes('libx264') && ffEnc.includes('aac') ? 'OK' : 'ÉCHEC', 'encodeurs libx264 + aac');
line(has('python3', ['-c', 'import PIL, numpy']) !== null ? 'OK' : 'ÉCHEC', 'python : pillow et numpy');
line(fs.existsSync('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') ? 'OK' : 'ATTENTION', 'police DejaVu', 'légendes des planches de contact et composite refs');

try {
  const res = await fetch(`${config.nouveauUrl}/`, { signal: AbortSignal.timeout(5_000) });
  line('OK', 'serveur nouveau (:8005)', `${config.nouveauUrl} répond ${res.status}`);
} catch (e) {
  line('ÉCHEC', 'serveur nouveau (:8005)', `${config.nouveauUrl} injoignable (${e.message}) : source et voix échoueront`);
}

// ---------------------------------------------------------------- diffusion image & video providers
const mediaCfg = resolveMediaConfig(db, process.env, config.dataRoot);
for (const [role, label] of [
  ['image', 'Diffusion Image (Nano Banana 2 Lite / 2.1)'],
  ['video', 'Diffusion Vidéo (Omni 1.2 Flash)'],
]) {
  const c = mediaCfg[role];
  if (c.driver === 'mock') {
    line('ATTENTION', `${label} (${c.driver})`, `mode mock actif (hors-ligne) · modèle=${c.model}`);
  } else if (!c.baseURL) {
    line('ÉCHEC', `${label} (${c.driver})`, 'baseURL manquante : configurez WHATIF_*_BASE_URL, providers.json ou UI ⚙');
  } else if (c.driver === 'gflow') {
    const origin = c.baseURL.replace(/\/+$/, '').replace(/\/v1$/, '');
    try {
      const hr = await fetch(`${origin}/health`, { signal: AbortSignal.timeout(5_000) });
      let creditsTxt = '';
      try {
        const cr = await fetch(`${origin}/v1/credits`, { signal: AbortSignal.timeout(5_000) });
        if (cr.ok) {
          const cj = await cr.json();
          if (cj && cj.credits !== undefined) creditsTxt = `, crédits=${cj.credits}`;
        }
      } catch {
        /* credits check optional */
      }
      const resSuffix = role === 'video' && c.resolution ? `, résolution=${c.resolution}` : '';
      const fbSuffix = role === 'image' && c.fallbackModel ? ` (fallback=${c.fallbackModel})` : '';
      line(
        hr.ok ? 'OK' : 'ÉCHEC',
        `${label} (gflow :8006)`,
        `${origin}/health → ${hr.status} · modèle=${c.model}${fbSuffix}${resSuffix}, refMode=${c.refMode}${creditsTxt}`,
      );
    } catch (e) {
      line('ÉCHEC', `${label} (gflow)`, `${origin}/health injoignable (${e.message})`);
    }
  } else if (!c.apiKey && ['gemini', 'openai'].includes(c.driver)) {
    line('ATTENTION', `${label} (${c.driver})`, `baseURL=${c.baseURL}, modèle=${c.model}, aucune clé API renseignée`);
  } else {
    const resSuffix = role === 'video' && c.resolution ? `, résolution=${c.resolution}` : '';
    line('OK', `${label} (${c.driver})`, `modèle=${c.model}${resSuffix}, endpoint=${c.baseURL}${c.endpointPath || ''}, refMode=${c.refMode}`);
  }
}

const up = mediaCfg.upscaler;
if (!up?.enabled) {
  line('OK', 'Upscaler Anime 2D (optionnel)', 'désactivé (activez-le via UI ⚙, providers.json ou WHATIF_UPSCALER_ENABLED=1)');
} else if (up.driver === 'mock') {
  line('ATTENTION', 'Upscaler Anime 2D (mock)', `cible=${up.target}, modèle=${up.model}`);
} else {
  try {
    const r = await fetch(`${up.baseURL.replace(/\/+$/, '')}/health`, { signal: AbortSignal.timeout(8_000) });
    line(
      r.ok ? 'OK' : 'ATTENTION',
      `Upscaler Anime 2D (${up.model} @ ${up.gpuType})`,
      `${up.baseURL}/health répond ${r.status} · cible=${up.target}, denoise=${up.denoiseLevel}`,
    );
  } catch (e) {
    line('ÉCHEC', 'Upscaler Anime 2D (Modal)', `${up.baseURL}/health injoignable (${e.message})`);
  }
}

// ---------------------------------------------------------------- the LLM agent + reviewer
try {
  const agent = resolveProvider(db);
  if (agent.id === 'mock') line('ÉCHEC', 'modèle de l’agent', 'WHATIF_PROVIDER=mock : aucun vrai modèle configuré');
  else {
    const r = await ask(agent);
    line(r.text ? 'OK' : 'ÉCHEC', `modèle de l’agent (${agent.model})`, `${r.ms} ms, réponse « ${r.text} », fin ${r.finish}`);
    try {
      await ask(agent, true);
      line('OK', 'l’agent voit les images');
    } catch (e) {
      line('ATTENTION', 'l’agent ne prend pas les images', `il ne verra pas ses planches (${String(e.message).split('\n')[0].slice(0, 120)})`);
    }
  }
} catch (e) {
  line('ÉCHEC', 'modèle de l’agent', String(e.message).split('\n')[0].slice(0, 200));
}

const reviewCfg = reviewProviderConfig(db);
const reviewer = reviewCfg ? resolveReviewProvider(db) : null;
if (!reviewer) line('ATTENTION', 'Reviewer', 'désactivé ou sans clé : aucun second regard sur les planches');
else {
  const same = reviewCfg.model === resolveProvider(db).model;
  try {
    const r = await ask(reviewer, true);
    line(
      r.text ? 'OK' : 'ÉCHEC',
      `Reviewer (${reviewer.model})`,
      `voit les images, ${r.ms} ms, réponse « ${r.text} », fin ${r.finish}${r.text ? '' : ' : réponse vide, monter WHATIF_REVIEW_MAX_TOKENS'}`,
    );
  } catch (e) {
    const msg = String(e.message).split('\n')[0].slice(0, 120);
    if (/\b40[13]\b/.test(msg)) line('ÉCHEC', `Reviewer (${reviewer.model})`, `clé API refusée (${msg}) : vérifier WHATIF_REVIEW_API_KEY`);
    else line('ATTENTION', `Reviewer (${reviewer.model})`, `refuse les images (${msg}) : il jugera uniquement sur les specs JSON`);
  }
  if (same) line('ATTENTION', 'Reviewer', 'même modèle que l’agent : régler WHATIF_REVIEW_MODEL pour un regard vraiment indépendant');
}

db.close();
console.log(failed ? `\n${failed} point(s) bloquant(s) à corriger avant un run.` : '\nRien de bloquant : le run diffusion peut partir.');
process.exit(failed ? 1 : 0);
