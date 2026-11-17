import fs from 'node:fs';
import path from 'node:path';

export type Config = {
  host: string;
  /** Exact additional browser origins, e.g. the current public tunnel; local origins remain allowed. */
  allowedOrigins: string[];
  port: number;
  dataRoot: string;
  dbPath: string;
  projectsDir: string;
  /** Kept nullable/optional for backward compatibility; the diffusion studio is fully self-contained. */
  skillDir: string | null;
  webDir: string;
  authToken: string | null;
  maxSteps: number;
  contextChars: number;
  approvalTimeoutMs: number;
  /** 'off' = the agent runs the whole pipeline unattended (default), 'pipeline' = ask before the
   *  tools that mutate a project, 'all' = ask for every 'ask' policy tool. */
  maxTokens: number;
  approvalsMode: 'off' | 'pipeline' | 'all';
  /** Local TTS/translation API (the "nouveau" app), the one the videos and their takes live in. */
  nouveauUrl: string;
  /** Maximum number of independent reviewer feedback rounds per episode before letting the pipeline proceed. */
  reviewMaxRounds: number;
};

function intEnv(raw: string | undefined, fallback: number, min = 0): number {
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.trunc(n));
}

function parseAllowedOrigins(raw: string | undefined): string[] {
  const origins = new Set<string>();
  for (const candidate of raw?.split(',') ?? []) {
    const value = candidate.trim();
    if (!value) continue;
    try {
      const url = new URL(value);
      if (!['http:', 'https:'].includes(url.protocol)) continue;
      if (url.username || url.password || url.pathname !== '/' || url.search || url.hash) continue;
      origins.add(url.origin);
    } catch {
      // Ignore malformed entries; a bad allowlist must never broaden access.
    }
  }
  return [...origins];
}

export function loadConfig(): Config {
  const here = path.dirname(new URL(import.meta.url).pathname);
  const packageRoot = path.resolve(here, '..', '..');
  const dataRoot = path.resolve(process.env.WHATIF_DATA ?? path.join(packageRoot, 'data'));
  const dbPath = process.env.WHATIF_DB ?? path.join(dataRoot, 'app.db');
  fs.mkdirSync(dataRoot, { recursive: true });
  return {
    host: process.env.WHATIF_HOST?.trim() || '127.0.0.1',
    allowedOrigins: parseAllowedOrigins(process.env.WHATIF_ALLOWED_ORIGINS),
    port: intEnv(process.env.WHATIF_PORT, 3020, 0),
    dataRoot,
    dbPath,
    projectsDir: path.join(dataRoot, 'projects'),
    skillDir: null,
    webDir: path.join(packageRoot, 'web', 'public'),
    authToken: process.env.WHATIF_AUTH_TOKEN?.trim() || null,
    maxSteps: intEnv(process.env.WHATIF_MAX_STEPS, 1_000, 1),
    contextChars: intEnv(process.env.WHATIF_CONTEXT_CHARS, 180_000, 2_000),
    approvalTimeoutMs: intEnv(process.env.WHATIF_APPROVAL_TIMEOUT_MS, 30 * 60_000, 100),
    maxTokens: intEnv(process.env.WHATIF_MAX_TOKENS, 24_000, 256),
    approvalsMode: ((): 'off' | 'pipeline' | 'all' => {
      const raw = (process.env.WHATIF_APPROVALS ?? 'off').toLowerCase();
      return raw === 'all' ? 'all' : raw === 'pipeline' || raw === 'on' ? 'pipeline' : 'off';
    })(),
    nouveauUrl: (process.env.WHATIF_NOUVEAU_URL?.trim() || 'http://127.0.0.1:8005').replace(/\/$/, ''),
    reviewMaxRounds: intEnv(process.env.WHATIF_REVIEW_MAX_ROUNDS, 3, 0),
  };
}
