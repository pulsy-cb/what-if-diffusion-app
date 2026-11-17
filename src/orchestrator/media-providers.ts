import fs from 'node:fs';
import path from 'node:path';
import type { Db } from './db.js';
import { resolveCatalogFilePaths, stripJsonCommentsAndTrailingCommas } from './providers/catalog.js';

export type MediaDriver = 'gflow' | 'openai' | 'gemini' | 'http' | 'mock';
export type MediaRefMode = 'multi' | 'composite' | 'none';
export type VideoResolution = '360p' | '720p' | '1080p';
export type UpscalerTarget = 'auto-uhd' | 'auto-2k' | 'auto-1080p' | '2x' | '3x' | '4x';
export type UpscalerModel = 'real-cugan' | 'realesr-animevideov3' | 'realesrgan-x2plus' | 'realesrgan-x4plus';
export type UpscalerGpu = 'L40S' | 'A100-40GB' | 'L4';
export type UpscalerDriver = 'modal-anime' | 'mock';

export const DEFAULT_UPSCALER_URL = 'https://bkrpulsy--anime-upscaler-t4-fastapi-app.modal.run';

export type MediaEndpointConfig = {
  id: string;
  driver: MediaDriver;
  baseURL: string;
  apiKey: string;
  hasKey: boolean;
  model: string;
  refMode: MediaRefMode;
  maxRefs: number;
  fallbackModel?: string;
  profile?: string;
  resolution?: VideoResolution;
  endpointPath?: string;
  pollPath?: string;
  headers?: Record<string, string>;
  extraBody?: Record<string, unknown>;
  responsePaths?: {
    url?: string;
    b64?: string;
    jobId?: string;
    status?: string;
  };
};

export type UpscalerConfig = {
  enabled: boolean;
  driver: UpscalerDriver;
  baseURL: string;
  apiKey: string;
  hasKey: boolean;
  target: UpscalerTarget;
  model: UpscalerModel;
  denoiseLevel: 0 | 1 | 2;
  gpuType: UpscalerGpu;
};

export type ResolvedMediaConfig = {
  image: MediaEndpointConfig;
  video: MediaEndpointConfig;
  upscaler: UpscalerConfig;
  catalogSlots: Record<string, { role: 'image' | 'video' | 'both'; label: string; driver: MediaDriver; baseURL: string; model?: string; refMode?: MediaRefMode; hasKey: boolean }>;
};

const IS_NODE_TEST_RUNNER =
  Boolean(process.env.NODE_TEST_CONTEXT) || process.execArgv.includes('--test') || process.env.NODE_ENV === 'test';

export const MEDIA_DRIVER_DEFAULTS: Record<
  MediaDriver,
  {
    label: string;
    baseURL: string;
    imageModel: string;
    imageFallbackModel?: string;
    videoModel: string;
    defaultResolution?: VideoResolution;
    refMode: MediaRefMode;
    maxRefs: number;
    imageEndpointPath?: string;
    videoEndpointPath?: string;
  }
> = {
  gflow: {
    label: 'gflow (:8006 Google Flow — Nano Banana & Omni)',
    baseURL: 'http://127.0.0.1:8006',
    imageModel: 'nano2-lite',
    imageFallbackModel: 'nano2.1',
    videoModel: 'omni-1.2-flash',
    defaultResolution: '360p',
    refMode: 'multi',
    maxRefs: 10,
    imageEndpointPath: '/v1/images/generations',
    videoEndpointPath: '/v1/videos/generations',
  },
  gemini: {
    label: 'gemini (Google AI Studio / Gemini API)',
    baseURL: 'https://generativelanguage.googleapis.com/v1beta',
    imageModel: 'gemini-3.1-flash-lite-image',
    videoModel: 'gemini-omni-flash-2.1',
    refMode: 'multi',
    maxRefs: 4,
  },
  openai: {
    label: 'openai (Proxy / OpenAI-compatible)',
    baseURL: 'https://api.openai.com/v1',
    imageModel: 'nano-banana-2-lite',
    videoModel: 'gemini-omni-flash-2.1',
    refMode: 'multi',
    maxRefs: 4,
    imageEndpointPath: '/images/generations',
    videoEndpointPath: '/videos/generations',
  },
  http: {
    label: 'http (Endpoint REST personnalisé)',
    baseURL: 'http://127.0.0.1:8090',
    imageModel: 'nano2-lite',
    videoModel: 'gemini-omni-flash-2.1',
    refMode: 'multi',
    maxRefs: 4,
    imageEndpointPath: '/generate/image',
    videoEndpointPath: '/generate/video',
  },
  mock: {
    label: 'mock (Génération locale offline pour tests)',
    baseURL: '',
    imageModel: 'mock-nano2-lite',
    videoModel: 'mock-omni-flash-2.1',
    refMode: 'multi',
    maxRefs: 4,
  },
};

function normalizeDriver(raw: unknown, fallback: MediaDriver): MediaDriver {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === 'gflow' || s === 'flow') return 'gflow';
  if (s === 'gemini' || s === 'google') return 'gemini';
  if (s === 'openai' || s === 'openai-compatible' || s === 'agnes') return 'openai';
  if (s === 'http' || s === 'custom' || s === 'webhook') return 'http';
  if (s === 'mock') return 'mock';
  return fallback;
}

function normalizeRefMode(raw: unknown, fallback: MediaRefMode): MediaRefMode {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === 'composite' || s === 'sheet' || s === 'single') return 'composite';
  if (s === 'multi' || s === 'array') return 'multi';
  if (s === 'none' || s === 'off') return 'none';
  return fallback;
}

function normalizeResolution(raw: unknown, fallback: VideoResolution = '1080p'): VideoResolution {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === '360p' || s === '360') return '360p';
  if (s === '720p' || s === '720' || s === 'hd') return '720p';
  if (s === '1080p' || s === '1080' || s === 'fhd') return '1080p';
  return fallback;
}

function normalizeUpscalerTarget(raw: unknown): UpscalerTarget {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === 'auto-uhd' || s === 'uhd' || s === '4k' || s === '2160p') return 'auto-uhd';
  if (s === 'auto-2k' || s === '2k' || s === '1440p') return 'auto-2k';
  if (s === 'auto-1080p' || s === '1080p' || s === 'fhd') return 'auto-1080p';
  if (s === '2x' || s === '2') return '2x';
  if (s === '3x' || s === '3') return '3x';
  if (s === '4x' || s === '4') return '4x';
  return 'auto-2k';
}

function normalizeUpscalerModel(raw: unknown): UpscalerModel {
  const s = String(raw ?? '').trim().toLowerCase();
  if (s === 'real-cugan') return 'real-cugan';
  if (s === 'realesr-animevideov3') return 'realesr-animevideov3';
  if (s === 'realesrgan-x2plus') return 'realesrgan-x2plus';
  if (s === 'realesrgan-x4plus') return 'realesrgan-x4plus';
  return 'real-cugan';
}

function normalizeUpscalerGpu(raw: unknown): UpscalerGpu {
  const s = String(raw ?? '').trim().toUpperCase();
  if (s === 'A100-40GB' || s === 'A100') return 'A100-40GB';
  if (s === 'L4') return 'L4';
  return 'L40S';
}

function normalizeDenoise(raw: unknown): 0 | 1 | 2 {
  const n = Number(raw);
  if (n === 0 || n === 1 || n === 2) return n;
  return 1;
}

function resolveEnvRef(val: unknown, env: NodeJS.ProcessEnv): string {
  if (typeof val !== 'string') return '';
  const trimmed = val.trim();
  if (!trimmed) return '';
  const m = /^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$/.exec(trimmed);
  if (m) return (env[m[1]!] ?? '').trim();
  return trimmed;
}

function resolveHeaders(raw: unknown, env: NodeJS.ProcessEnv): Record<string, string> | undefined {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return undefined;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof v !== 'string') continue;
    const resolved = v.replace(/\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?/g, (_full, name: string) => env[name] ?? '');
    if (resolved.trim()) out[k] = resolved;
  }
  return Object.keys(out).length ? out : undefined;
}

type RawMediaCatalog = {
  defaultImageProvider?: string;
  defaultVideoProvider?: string;
  image?: Record<string, unknown>;
  video?: Record<string, unknown>;
  upscaler?: Record<string, unknown>;
  named: Record<string, Record<string, unknown>>;
};

function readMediaCatalog(env: NodeJS.ProcessEnv, dataRoot?: string): RawMediaCatalog {
  const out: RawMediaCatalog = { named: {} };
  for (const filePath of resolveCatalogFilePaths(env, dataRoot)) {
    try {
      if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) continue;
      const raw = fs.readFileSync(filePath, 'utf8');
      if (!raw.trim()) continue;
      const parsed = JSON.parse(stripJsonCommentsAndTrailingCommas(raw));
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) continue;
      if (typeof parsed.defaultImageProvider === 'string' && parsed.defaultImageProvider.trim()) {
        out.defaultImageProvider = parsed.defaultImageProvider.trim();
      }
      if (typeof parsed.defaultVideoProvider === 'string' && parsed.defaultVideoProvider.trim()) {
        out.defaultVideoProvider = parsed.defaultVideoProvider.trim();
      }
      const mp = parsed.mediaProviders;
      if (!mp || typeof mp !== 'object' || Array.isArray(mp)) continue;
      for (const [key, val] of Object.entries(mp as Record<string, unknown>)) {
        if (!val || typeof val !== 'object' || Array.isArray(val)) continue;
        const entry = val as Record<string, unknown>;
        if (key === 'upscaler') {
          out.upscaler = entry;
          continue;
        }
        if (entry.enabled === false) continue;
        if (key === 'image') out.image = entry;
        else if (key === 'video') out.video = entry;
        else out.named[key] = entry;
      }
    } catch {
      /* ignore broken catalog file here — catalog.ts surfaces parse errors */
    }
  }
  return out;
}

function resolveRole(
  role: 'image' | 'video',
  dbSettings: Record<string, string>,
  env: NodeJS.ProcessEnv,
  catalog: RawMediaCatalog,
): MediaEndpointConfig {
  const isImage = role === 'image';
  const dbProv = (isImage ? dbSettings.imageProvider : dbSettings.videoProvider)?.trim() || '';
  const dbBase = (isImage ? dbSettings.imageBaseURL : dbSettings.videoBaseURL)?.trim() || '';
  const dbKey = (isImage ? dbSettings.imageApiKey : dbSettings.videoApiKey)?.trim() || '';
  const dbModel = (isImage ? dbSettings.imageModel : dbSettings.videoModel)?.trim() || '';
  const dbRefMode = isImage ? dbSettings.imageRefMode?.trim() || '' : '';
  const dbEndpoint = (!isImage ? dbSettings.videoEndpointPath?.trim() : dbSettings.imageEndpointPath?.trim()) || '';
  const dbFallbackModel = isImage ? dbSettings.imageFallbackModel?.trim() || '' : '';
  const dbProfile = dbSettings.gflowProfile?.trim() || '';
  const dbResolution = !isImage ? dbSettings.videoResolution?.trim() || '' : '';

  const envProv = (isImage ? env.WHATIF_IMAGE_PROVIDER : env.WHATIF_VIDEO_PROVIDER)?.trim() || '';
  const envBase = (isImage ? env.WHATIF_IMAGE_BASE_URL : env.WHATIF_VIDEO_BASE_URL)?.trim() || '';
  const envKey = (isImage ? env.WHATIF_IMAGE_API_KEY : env.WHATIF_VIDEO_API_KEY)?.trim() || '';
  const envModel = (isImage ? env.WHATIF_IMAGE_MODEL : env.WHATIF_VIDEO_MODEL)?.trim() || '';
  const envFallbackModel = isImage ? env.WHATIF_IMAGE_FALLBACK_MODEL?.trim() || '' : '';
  const envProfile = env.WHATIF_GFLOW_PROFILE?.trim() || '';
  const envRefMode = (isImage ? env.WHATIF_IMAGE_REF_MODE : env.WHATIF_VIDEO_REF_MODE)?.trim() || '';
  const envMaxRefs = Number((isImage ? env.WHATIF_IMAGE_MAX_REFS : env.WHATIF_VIDEO_MAX_REFS) ?? '');
  const envEndpoint = (isImage ? env.WHATIF_IMAGE_ENDPOINT : env.WHATIF_VIDEO_ENDPOINT)?.trim() || '';
  const envPollPath = (!isImage ? env.WHATIF_VIDEO_POLL_PATH : env.WHATIF_IMAGE_POLL_PATH)?.trim() || '';
  const envResolution = !isImage ? env.WHATIF_VIDEO_RESOLUTION?.trim() || '' : '';

  const directCat = isImage ? catalog.image : catalog.video;
  const catDefaultName = isImage ? catalog.defaultImageProvider : catalog.defaultVideoProvider;
  const requestedId = dbProv || envProv || catDefaultName || '';
  const namedCat = requestedId && catalog.named[requestedId] ? catalog.named[requestedId]! : undefined;
  const nestedNamed = namedCat && typeof namedCat[role] === 'object' && !Array.isArray(namedCat[role])
    ? (namedCat[role] as Record<string, unknown>)
    : undefined;
  const catEntry = nestedNamed ?? namedCat ?? (directCat && (!requestedId || requestedId === normalizeDriver(directCat.driver ?? directCat.provider, 'gflow')) ? directCat : undefined);

  const fallbackDriver: MediaDriver = IS_NODE_TEST_RUNNER && !requestedId && !catEntry ? 'mock' : 'gflow';
  const driver = normalizeDriver(
    dbProv && MEDIA_DRIVER_DEFAULTS[dbProv.toLowerCase() as MediaDriver]
      ? dbProv
      : catEntry?.driver ?? catEntry?.provider ?? requestedId ?? envProv,
    fallbackDriver,
  );
  const defaults = MEDIA_DRIVER_DEFAULTS[driver];

  const catBase = typeof catEntry?.baseURL === 'string' ? catEntry.baseURL.trim() : typeof catEntry?.baseUrl === 'string' ? catEntry.baseUrl.trim() : '';
  const catKey = resolveEnvRef(catEntry?.apiKey, env) || (typeof catEntry?.apiKeyEnv === 'string' ? (env[catEntry.apiKeyEnv.trim()] ?? '').trim() : '');
  const catModel = typeof catEntry?.model === 'string' ? catEntry.model.trim() : typeof catEntry?.defaultModel === 'string' ? catEntry.defaultModel.trim() : '';
  const catRefMode = catEntry?.refMode;
  const catMaxRefs = Number(catEntry?.maxRefs);
  const catEndpoint = typeof catEntry?.endpointPath === 'string' ? catEntry.endpointPath.trim() : '';
  const catPoll = typeof catEntry?.pollPath === 'string' ? catEntry.pollPath.trim() : '';
  const catRes = !isImage ? catEntry?.resolution : undefined;

  const baseURL = dbBase || catBase || envBase || defaults.baseURL;
  const apiKey = dbKey || catKey || envKey || '';
  const model = dbModel || catModel || envModel || (isImage ? defaults.imageModel : defaults.videoModel);
  const refMode = normalizeRefMode(dbRefMode || catRefMode || envRefMode, defaults.refMode);
  const maxRefs =
    Number.isFinite(catMaxRefs) && catMaxRefs >= 0
      ? Math.trunc(catMaxRefs)
      : Number.isFinite(envMaxRefs) && envMaxRefs >= 0
        ? Math.trunc(envMaxRefs)
        : defaults.maxRefs;
  const endpointPath = dbEndpoint || catEndpoint || envEndpoint || (isImage ? defaults.imageEndpointPath : defaults.videoEndpointPath);
  const pollPath = catPoll || envPollPath || undefined;
  const fallbackModel = isImage
    ? dbFallbackModel ||
      (typeof catEntry?.fallbackModel === 'string' ? catEntry.fallbackModel.trim() : '') ||
      envFallbackModel ||
      defaults.imageFallbackModel ||
      ''
    : '';
  const profile = dbProfile || (typeof catEntry?.profile === 'string' ? catEntry.profile.trim() : '') || envProfile || '';
  const resolution = !isImage
    ? normalizeResolution(dbResolution || catRes || envResolution, defaults.defaultResolution ?? '1080p')
    : undefined;
  const headers = resolveHeaders(catEntry?.headers, env);
  const extraBody =
    catEntry?.extraBody && typeof catEntry.extraBody === 'object' && !Array.isArray(catEntry.extraBody)
      ? (catEntry.extraBody as Record<string, unknown>)
      : undefined;
  const responsePaths =
    catEntry?.responsePaths && typeof catEntry.responsePaths === 'object' && !Array.isArray(catEntry.responsePaths)
      ? (catEntry.responsePaths as MediaEndpointConfig['responsePaths'])
      : undefined;

  const isLocal = /^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/|$)/i.test(baseURL);
  const hasKey = Boolean(apiKey) || driver === 'mock' || isLocal;

  return {
    id: requestedId || driver,
    driver,
    baseURL,
    apiKey,
    hasKey,
    model,
    refMode,
    maxRefs,
    ...(fallbackModel ? { fallbackModel } : {}),
    ...(profile ? { profile } : {}),
    ...(resolution ? { resolution } : {}),
    ...(endpointPath ? { endpointPath } : {}),
    ...(pollPath ? { pollPath } : {}),
    ...(headers ? { headers } : {}),
    ...(extraBody ? { extraBody } : {}),
    ...(responsePaths ? { responsePaths } : {}),
  };
}

function resolveUpscaler(
  dbSettings: Record<string, string>,
  env: NodeJS.ProcessEnv,
  catalog: RawMediaCatalog,
): UpscalerConfig {
  const cat = catalog.upscaler;
  const parseBool = (v: unknown): boolean | null => {
    if (typeof v === 'boolean') return v;
    const s = String(v ?? '').trim().toLowerCase();
    if (!s) return null;
    if (['1', 'true', 'yes', 'on'].includes(s)) return true;
    if (['0', 'false', 'no', 'off'].includes(s)) return false;
    return null;
  };
  const dbEnabled = parseBool(dbSettings.upscalerEnabled);
  const catEnabled = cat ? parseBool(cat.enabled) : null;
  const envEnabled = parseBool(env.WHATIF_UPSCALER_ENABLED);
  const enabled = dbEnabled ?? catEnabled ?? envEnabled ?? false;

  const rawDriver = (
    dbSettings.upscalerDriver?.trim() ||
    (typeof cat?.driver === 'string' ? cat.driver.trim() : '') ||
    env.WHATIF_UPSCALER_DRIVER?.trim() ||
    ''
  ).toLowerCase();
  const driver: UpscalerDriver =
    rawDriver === 'mock' || (IS_NODE_TEST_RUNNER && !rawDriver) ? 'mock' : 'modal-anime';

  const baseURL = (
    dbSettings.upscalerBaseURL?.trim() ||
    (typeof cat?.baseURL === 'string' ? cat.baseURL.trim() : typeof cat?.baseUrl === 'string' ? cat.baseUrl.trim() : '') ||
    env.WHATIF_UPSCALER_URL?.trim() ||
    DEFAULT_UPSCALER_URL
  ).replace(/\/+$/, '');

  const apiKey =
    dbSettings.upscalerApiKey?.trim() ||
    resolveEnvRef(cat?.apiKey, env) ||
    (typeof cat?.apiKeyEnv === 'string' ? (env[cat.apiKeyEnv.trim()] ?? '').trim() : '') ||
    env.WHATIF_UPSCALER_API_KEY?.trim() ||
    '';

  const target = normalizeUpscalerTarget(
    dbSettings.upscalerTarget || cat?.target || env.WHATIF_UPSCALER_TARGET,
  );
  const model = normalizeUpscalerModel(
    dbSettings.upscalerModel || cat?.model || env.WHATIF_UPSCALER_MODEL,
  );
  const denoiseLevel = normalizeDenoise(
    dbSettings.upscalerDenoise ?? cat?.denoiseLevel ?? env.WHATIF_UPSCALER_DENOISE,
  );
  const gpuType = normalizeUpscalerGpu(
    dbSettings.upscalerGpu || cat?.gpuType || env.WHATIF_UPSCALER_GPU,
  );

  const hasKey = Boolean(apiKey) || driver === 'mock' || baseURL === DEFAULT_UPSCALER_URL;
  return { enabled, driver, baseURL, apiKey, hasKey, target, model, denoiseLevel, gpuType };
}

export function resolveMediaConfig(
  db: Db,
  env: NodeJS.ProcessEnv = process.env,
  dataRoot?: string,
): ResolvedMediaConfig {
  const s = db.settings();
  const root = dataRoot ?? db.dataRoot;
  const catalog = readMediaCatalog(env, root);

  const catalogSlots: ResolvedMediaConfig['catalogSlots'] = {};
  for (const [id, entry] of Object.entries(catalog.named)) {
    const driver = normalizeDriver(entry.driver ?? entry.provider ?? id, 'http');
    const baseURL = String(entry.baseURL ?? entry.baseUrl ?? MEDIA_DRIVER_DEFAULTS[driver].baseURL).trim();
    const key = resolveEnvRef(entry.apiKey, env);
    const roleRaw = String(entry.role ?? '').toLowerCase();
    const role: 'image' | 'video' | 'both' = roleRaw === 'image' ? 'image' : roleRaw === 'video' ? 'video' : 'both';
    catalogSlots[id] = {
      role,
      label: String(entry.label ?? `${id} (${driver})`),
      driver,
      baseURL,
      ...(typeof entry.model === 'string' && entry.model.trim() ? { model: entry.model.trim() } : {}),
      ...(entry.refMode ? { refMode: normalizeRefMode(entry.refMode, MEDIA_DRIVER_DEFAULTS[driver].refMode) } : {}),
      hasKey: Boolean(key) || driver === 'mock' || /^https?:\/\/(127\.0\.0\.1|localhost)/i.test(baseURL),
    };
  }

  return {
    image: resolveRole('image', s, env, catalog),
    video: resolveRole('video', s, env, catalog),
    upscaler: resolveUpscaler(s, env, catalog),
    catalogSlots,
  };
}

export function getMediaPresets(env: NodeJS.ProcessEnv = process.env, dataRoot?: string) {
  const catalog = readMediaCatalog(env, dataRoot);
  const out: Record<
    string,
    {
      label: string;
      driver: MediaDriver;
      baseURL: string;
      imageModel: string;
      imageFallbackModel?: string;
      videoModel: string;
      defaultResolution?: VideoResolution;
      refMode: MediaRefMode;
      fromCatalog?: boolean;
    }
  > = {};
  for (const [id, entry] of Object.entries(catalog.named)) {
    const driver = normalizeDriver(entry.driver ?? entry.provider ?? id, 'http');
    const def = MEDIA_DRIVER_DEFAULTS[driver];
    out[id] = {
      label: String(entry.label ?? `${id} (${driver})`),
      driver,
      baseURL: String(entry.baseURL ?? entry.baseUrl ?? def.baseURL).trim(),
      imageModel: String((entry.image as any)?.model ?? entry.model ?? def.imageModel).trim(),
      ...(typeof entry.fallbackModel === 'string' && entry.fallbackModel.trim()
        ? { imageFallbackModel: entry.fallbackModel.trim() }
        : def.imageFallbackModel
          ? { imageFallbackModel: def.imageFallbackModel }
          : {}),
      videoModel: String((entry.video as any)?.model ?? entry.model ?? def.videoModel).trim(),
      defaultResolution: normalizeResolution(entry.resolution ?? (entry.video as any)?.resolution, def.defaultResolution ?? '1080p'),
      refMode: normalizeRefMode(entry.refMode, def.refMode),
      fromCatalog: true,
    };
  }
  for (const [driver, def] of Object.entries(MEDIA_DRIVER_DEFAULTS) as [MediaDriver, (typeof MEDIA_DRIVER_DEFAULTS)[MediaDriver]][]) {
    if (!out[driver]) {
      out[driver] = {
        label: def.label,
        driver,
        baseURL: def.baseURL,
        imageModel: def.imageModel,
        ...(def.imageFallbackModel ? { imageFallbackModel: def.imageFallbackModel } : {}),
        videoModel: def.videoModel,
        ...(def.defaultResolution ? { defaultResolution: def.defaultResolution } : {}),
        refMode: def.refMode,
      };
    }
  }
  return out;
}

/** Build child-process environment carrying resolved image, video, and upscaler credentials for Python scripts. */
export function mediaEnvForSubprocess(
  db: Db | undefined,
  env: NodeJS.ProcessEnv = process.env,
  dataRoot?: string,
): NodeJS.ProcessEnv {
  const cfg = db
    ? resolveMediaConfig(db, env, dataRoot)
    : (() => {
        const catalog = readMediaCatalog(env, dataRoot);
        return {
          image: resolveRole('image', {}, env, catalog),
          video: resolveRole('video', {}, env, catalog),
          upscaler: resolveUpscaler({}, env, catalog),
        };
      })();

  return {
    ...env,
    WHATIF_MEDIA_CONFIG_JSON: JSON.stringify({ image: cfg.image, video: cfg.video, upscaler: cfg.upscaler }),
    WHATIF_IMAGE_PROVIDER: cfg.image.driver,
    WHATIF_IMAGE_BASE_URL: cfg.image.baseURL,
    WHATIF_IMAGE_API_KEY: cfg.image.apiKey,
    WHATIF_IMAGE_MODEL: cfg.image.model,
    WHATIF_IMAGE_FALLBACK_MODEL: cfg.image.fallbackModel ?? '',
    WHATIF_IMAGE_REF_MODE: cfg.image.refMode,
    WHATIF_IMAGE_MAX_REFS: String(cfg.image.maxRefs),
    WHATIF_GFLOW_PROFILE: cfg.image.profile ?? cfg.video.profile ?? '',
    WHATIF_VIDEO_PROVIDER: cfg.video.driver,
    WHATIF_VIDEO_BASE_URL: cfg.video.baseURL,
    WHATIF_VIDEO_API_KEY: cfg.video.apiKey,
    WHATIF_VIDEO_MODEL: cfg.video.model,
    WHATIF_VIDEO_RESOLUTION: cfg.video.resolution ?? (cfg.video.driver === 'gflow' ? '360p' : '1080p'),
    WHATIF_UPSCALER_ENABLED: cfg.upscaler.enabled ? '1' : '0',
    WHATIF_UPSCALER_DRIVER: cfg.upscaler.driver,
    WHATIF_UPSCALER_URL: cfg.upscaler.baseURL,
    WHATIF_UPSCALER_API_KEY: cfg.upscaler.apiKey,
    WHATIF_UPSCALER_TARGET: cfg.upscaler.target,
    WHATIF_UPSCALER_MODEL: cfg.upscaler.model,
    WHATIF_UPSCALER_DENOISE: String(cfg.upscaler.denoiseLevel),
    WHATIF_UPSCALER_GPU: cfg.upscaler.gpuType,
  };
}
