import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { spawnSync } from 'node:child_process';

import { loadConfig } from '../dist/orchestrator/config.js';
import { createApp } from '../dist/orchestrator/main.js';
import { ToolRegistry, fsTools, defaultRegistry } from '../dist/tools/index.js';

// The audio pipeline scripts need numpy. Where python has none, those tests say why they are not run instead of failing
// on an ImportError that has nothing to do with the code (the install is documented in README).
const HAS_NUMPY = spawnSync('python3', ['-c', 'import numpy'], { stdio: 'ignore' }).status === 0;
const NEEDS_NUMPY = HAS_NUMPY ? {} : { skip: 'python3 has no numpy: pip install numpy to run this test' };

function tmpApp() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-'));
  const config = { ...loadConfig(), dataRoot: dir, dbPath: path.join(dir, 'app.db'), projectsDir: path.join(dir, 'projects') };
  return { dir, app: createApp(config) };
}

test('tools: schema validation and unique-match editing', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-tools-'));
  fs.mkdirSync(path.join(dir, 'src', 'episodes', 'e'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'src', 'episodes', 'e', 'a.ts'), 'export const A = 1;\nexport const B = 2;\n');
  const registry = new ToolRegistry(fsTools);
  const ctx = { projectId: 'p', projectRoot: dir, dataRoot: dir, skillDir: null, log: () => {} };

  const bad = await registry.run('read_file', { nope: 1 }, ctx);
  assert.equal(bad.ok, false);
  assert.match(bad.summary, /invalid arguments/);

  const read = await registry.run('read_file', { path: 'src/episodes/e/a.ts' }, ctx);
  assert.equal(read.ok, true);
  assert.match(read.summary, /export const A/);

  const edit = await registry.run('edit_file', { path: 'src/episodes/e/a.ts', edits: [{ oldText: 'A = 1', newText: 'A = 42' }] }, ctx);
  assert.equal(edit.ok, true);
  assert.match(fs.readFileSync(path.join(dir, 'src', 'episodes', 'e', 'a.ts'), 'utf8'), /A = 42/);

  const ambiguous = await registry.run('edit_file', { path: 'src/episodes/e/a.ts', edits: [{ oldText: 'export const', newText: 'x' }] }, ctx);
  assert.equal(ambiguous.ok, false);
  assert.match(ambiguous.summary, /not unique/);

  const escape = await registry.run('read_file', { path: '../../etc/passwd' }, ctx);
  assert.equal(escape.ok, false);
});

test('shell: allowlist refuses operators and unknown commands', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-shell-'));
  const registry = defaultRegistry();
  const ctx = { projectId: 'p', projectRoot: dir, dataRoot: dir, skillDir: null, log: () => {} };
  const { checkCommand } = await import('../dist/tools/shell-policy.js');
  assert.equal(checkCommand('ls -la').ok, true);
  assert.equal(checkCommand('npm install').ok, true);
  assert.equal(checkCommand('rm -rf /').ok, false);
  assert.equal(checkCommand('ls; rm -rf /').ok, false);
  assert.equal(checkCommand('git status').ok, true);
  assert.equal(registry.get('shell').policy, 'ask');
});

test('agent loop end-to-end over HTTP with the mock provider', async () => {
  const { dir, app } = tmpApp();
  app.db.setSetting('provider', 'mock');
  await new Promise((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  const port = app.server.address().port;
  const base = `http://127.0.0.1:${port}`;
  try {
    const health = await (await fetch(`${base}/api/health`)).json();
    assert.equal(health.ok, true);

    const created = await (
      await fetch(`${base}/api/projects`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: 'demo' }) })
    ).json();
    assert.equal(created.project.id, 'demo');

    const thread = await (
      await fetch(`${base}/api/threads`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ projectId: 'demo' }),
      })
    ).json();

    const posted = await fetch(`${base}/api/threads/${thread.id}/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'fais un tour de reconnaissance' }),
    });
    assert.equal(posted.status, 202);

    let done = false;
    for (let i = 0; i < 100 && !done; i++) {
      await new Promise((r) => setTimeout(r, 100));
      const data = await (await fetch(`${base}/api/threads/${thread.id}/messages`)).json();
      const last = data.messages[data.messages.length - 1];
      if (last?.role === 'assistant' && /Terminé/.test(last.content)) done = true;
    }
    assert.equal(done, true, 'the mock agent should finish its script');

    const notes = path.join(dir, 'projects', 'demo', 'NOTES.md');
    assert.ok(fs.existsSync(notes), 'write_file should have created NOTES.md');
    assert.match(fs.readFileSync(notes, 'utf8'), /Agent notes/);

    const served = await fetch(`${base}/api/file?project=demo&path=NOTES.md`);
    assert.equal(served.status, 200);
    assert.match(await served.text(), /Agent notes/);

    const messages = await (await fetch(`${base}/api/threads/${thread.id}/messages`)).json();
    assert.ok(messages.messages.some((m) => m.role === 'tool' && m.payload?.name === 'repo_map'));
    assert.equal(typeof messages.elapsedMs, 'number');
    assert.ok(messages.elapsedMs >= 0);

    const projs = await (await fetch(`${base}/api/projects`)).json();
    const demoProj = projs.find((p) => p.id === 'demo');
    assert.ok(demoProj);
    assert.equal(typeof demoProj.elapsedMs, 'number');
  } finally {
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('approval gate blocks a run until the human answers', async () => {
  const { dir, app } = tmpApp();
  const thread = app.db.createThread((app.db.createProject('p2', 'p2', path.join(dir, 'p2'))).id, 'gate');
  const pending = app.approvals.request(thread.id, 'whatif_run', { step: 'voice' }, 5000);
  const row = app.db.pendingApprovals(thread.id)[0];
  assert.ok(row, 'an approval row must be pending');
  assert.equal(app.approvals.decide(row.id, true), true);
  assert.equal(await pending, 'approved');
  assert.equal(app.db.getApproval(row.id).status, 'approved');

  const denied = app.approvals.request(thread.id, 'shell', { command: 'ls' }, 5000);
  const row2 = app.db.pendingApprovals(thread.id)[0];
  app.approvals.decide(row2.id, false);
  assert.equal(await denied, 'denied');

  const timeout = await app.approvals.request(thread.id, 'shell', { command: 'ls' }, 30);
  assert.equal(timeout, 'timeout');
  await app.close();
  fs.rmSync(dir, { recursive: true, force: true });
});

test('whatif audits: episode formula/language and shots diffusion prompt/aspect checks', async () => {
  const { episodeAudit, shotsAudit, normalizeAspect, syncStudioAspect } = await import('../dist/tools/whatif.js');
  const badFr = {
    title: ['WHAT IF YOU COULD', 'PLIE UNE FEUILLE ?'],
    script: [
      { beat: '1', text: 'Tu plies une feuille en deux dans ta chambre.' },
      { beat: '2', text: 'Tu la replies encore sur le bureau.' },
      { beat: '3', text: 'La pile dépasse déjà la table.' },
      { beat: '4', text: 'What if you could plier une feuille 42 fois ?' },
      { beat: '5', text: 'À 42 plis, elle atteint la Lune : 439 800 km.' },
    ],
    science: [{ claim: '42 plis = 439 800 km', working: '0.1mm * 2^42', source: 'exponentielle' }],
  };
  assert.match(episodeAudit(badFr, 'fold') ?? '', /mixes English/);

  const goodFr = {
    ...badFr,
    title: ['ET SI TU POUVAIS', 'PLIER UNE FEUILLE 42 FOIS ?'],
    script: badFr.script.map((b, i) => (i === 3 ? { beat: '4', text: 'Et si tu pouvais plier une feuille 42 fois ?' } : b)),
  };
  assert.equal(episodeAudit(goodFr, 'fold'), null);

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-shots-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'fold');
    const shotsDir = path.join(epDir, 'shots');
    fs.mkdirSync(shotsDir, { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'bible.json'),
      JSON.stringify({
        stylePrompt: '2D hand-crafted cel animation, clean ink lines, moody warm lighting',
        characters: [{ id: 'HERO', name: 'Hero', appearance: 'Young adult in dark coat, short brown hair' }],
        places: [{ id: 'STUDY', name: 'Study', description: 'Quiet home study with wooden desk and lamp' }],
      }),
    );
    fs.writeFileSync(
      path.join(shotsDir, 'S1.json'),
      JSON.stringify({
        id: 'S1',
        place: 'UNKNOWN_ROOM',
        characters: ['HERO'],
        size: 'wide',
        imagePrompt: 'Wide 2D shot of Hero standing at the study desk holding a sheet of paper.',
        videoPrompt: 'Hero folds the paper gently with slow push camera, strictly no dialogue.',
        sfxHint: 'crisp paper folding sound, quiet room tone, no music',
      }),
    );
    fs.writeFileSync(
      path.join(shotsDir, 'S2.json'),
      JSON.stringify({
        id: 'S2',
        place: 'STUDY',
        characters: [],
        size: 'insert',
        onScreenText: { kind: 'document', text: 'FOLD HERE', lang: 'en' },
        imagePrompt: 'Macro insert of note on desk reading "FOLD HERE" with pen beside it.',
        videoPrompt: 'Camera slowly pushes toward folded document, strictly no speech.',
        sfxHint: 'soft paper creak, no music',
      }),
    );
    const ep = { lang: 'fr', shots: [{ id: 'S1', set: 'study', size: 'wide' }, { id: 'S2', set: 'study', size: 'insert' }] };
    const tl = { shots: [{ id: 'S1', len: 60 }, { id: 'S2', len: 60 }] };
    const err = shotsAudit(dir, 'fold', ep, tl) ?? '';
    assert.match(err, /S1\.place "UNKNOWN_ROOM" is not defined in bible\.json\.places/);
    assert.match(err, /S2\.onScreenText\.lang is "en" on a "fr" episode/);

    assert.equal(normalizeAspect('9:16'), '9:16');
    assert.equal(normalizeAspect('vertical'), '9:16');
    assert.equal(normalizeAspect('16:9'), '16:9');
    fs.writeFileSync(path.join(dir, 'PROJECT.md'), '# Project\n\n- aspect: 9:16 (1080x1920 vertical)\n');
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ id: 'Fold', slug: 'fold', aspect: '16:9' }));
    assert.equal(syncStudioAspect(dir, 'fold'), '9:16');
    assert.equal(JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8')).aspect, '9:16');

    const shortenedSourceEp = {
      title: ['LE PIRE', 'RENDEZ-VOUS'],
      source: { provider: 'nouveau', videoId: '7685825400659774751', words: 288 },
      script: [{ text: 'Un court résumé de quinze mots qui raccourcit complètement la traduction du serveur.' }],
      science: [],
    };
    assert.match(episodeAudit(shortenedSourceEp, 'rdv') ?? '', /shortened or rewrote the :8005 translation proposal/);

    const fullWords = Array.from({ length: 240 }, (_, i) => `mot${i}`).join(' ');
    const fullSourceEp = {
      title: ['LE PIRE', 'RENDEZ-VOUS'],
      source: { provider: 'nouveau', videoId: '7685825400659774751', words: 288 },
      script: [{ text: fullWords }],
      science: [],
    };
    assert.equal(episodeAudit(fullSourceEp, 'rdv'), null);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('context compaction: compacts old write_file payloads, drops orphan tool messages, and protects scripts/package.json', async () => {
  const { buildMessages, countScriptWords } = await import('../dist/orchestrator/context.js');
  const { checkCommand } = await import('../dist/tools/shell-policy.js');

  assert.equal(countScriptWords([{ text: '[excited] Bonjour <break/> le monde !' }]), 4);
  assert.equal(checkCommand('find . -name "*.ts" -delete').ok, false);
  assert.equal(checkCommand('cat /etc/passwd').ok, false);
  assert.equal(checkCommand('cat .env').ok, false);

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-ctx-'));
  try {
    const registry = new ToolRegistry(fsTools);
    const ctx = { projectId: 'p', projectRoot: dir, dataRoot: dir, skillDir: null, log: () => {} };
    const blockedScript = await registry.run('write_file', { path: 'scripts/evil.py', content: 'print(1)' }, ctx);
    assert.equal(blockedScript.ok, false);
    assert.match(blockedScript.summary, /refused by security policy/);

    const blockedPkg = await registry.run('write_file', { path: 'package.json', content: '{}' }, ctx);
    assert.equal(blockedPkg.ok, false);

    const bigContent = Array.from({ length: 120 }, (_, i) => `export const line${i} = ${i};`).join('\n');
    const history = [
      {
        id: 1,
        thread_id: 't',
        role: 'assistant',
        content: 'writing shots batch 1',
        payload: JSON.stringify({ toolCalls: [{ id: 'c1', name: 'write_file', args: { path: 'src/episodes/ep/shots.tsx', content: bigContent } }] }),
        created_at: '',
      },
      {
        id: 2,
        thread_id: 't',
        role: 'tool',
        content: 'ok\ncreated shots.tsx',
        payload: JSON.stringify({ toolCallId: 'c1', name: 'write_file', ok: true }),
        created_at: '',
      },
      { id: 3, thread_id: 't', role: 'user', content: 'continue step 1', payload: null, created_at: '' },
      { id: 4, thread_id: 't', role: 'assistant', content: 'step 1 done', payload: null, created_at: '' },
      { id: 5, thread_id: 't', role: 'user', content: 'continue step 2', payload: null, created_at: '' },
      { id: 6, thread_id: 't', role: 'assistant', content: 'step 2 done', payload: null, created_at: '' },
    ];
    const msgs = buildMessages({ projectRoot: dir, history, maxChars: 90_000 });
    const oldAssistant = msgs.find((m) => m.role === 'assistant' && m.toolCalls?.length);
    assert.ok(oldAssistant, 'old assistant tool call should be present');
    assert.match(String(oldAssistant.toolCalls[0].args.content), /\[compacted:/);

    // Test episodePin: pins DIRECTOR.md, bible.json, authored shots, and remaining shots
    const epDir = path.join(dir, 'src', 'episodes', 'ep');
    fs.mkdirSync(path.join(epDir, 'shots'), { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        id: 'Ep',
        slug: 'ep',
        title: ['TEST'],
        script: [{ text: 'Un deux trois quatre' }],
        shots: [
          { id: 'S1', size: 'wide', set: 'study', note: 'Un deux' },
          { id: 'S2', size: 'close', set: 'study', note: 'trois quatre' },
        ],
      }),
    );
    fs.writeFileSync(
      path.join(epDir, 'timeline.json'),
      JSON.stringify({
        slug: 'ep',
        total: 120,
        segments: [{ shot: 'S1', a: 0, b: 60 }, { shot: 'S2', a: 60, b: 120 }],
        shots: [{ id: 'S1', len: 60, size: 'wide' }, { id: 'S2', len: 60, size: 'close' }],
      }),
    );
    fs.writeFileSync(path.join(epDir, 'DIRECTOR.md'), '# Réalisation\nSalon crème #EFE3CC avec canapé vert #7A8B6F.');
    fs.writeFileSync(
      path.join(epDir, 'bible.json'),
      JSON.stringify({
        stylePrompt: '2D cel-shaded graphic novel style, crisp dark ink outlines',
        characters: [{ id: 'NARR', name: 'NARR', appearance: 'Young adult in bright blue jacket and dark jeans' }],
        places: [{ id: 'STUDY', name: 'Study', description: 'Warm cream living room study with olive green sofa and window' }],
      }),
    );
    fs.writeFileSync(
      path.join(epDir, 'shots', 'S1.json'),
      JSON.stringify({
        id: 'S1',
        place: 'STUDY',
        characters: ['NARR'],
        imagePrompt: 'Wide 2D frame of NARR in bright blue jacket inside the warm cream living room.',
        videoPrompt: 'NARR nods gently in the warm light with slow push camera, strictly no dialogue.',
        sfxHint: 'soft room ambience, fabric rustle, no music',
      }),
    );
    const msgsPinned = buildMessages({ projectRoot: dir, history, maxChars: 90_000 });
    const pinMsg = msgsPinned.find((m) => m.role === 'system' && m.content.startsWith('## Active episode continuity pin'));
    assert.ok(pinMsg, 'continuity pin system message should be present');
    assert.match(pinMsg.content, /Salon crème #EFE3CC/);
    assert.match(pinMsg.content, /NARR/);
    assert.match(pinMsg.content, /Remaining: S2/);

    // Test independent Reviewer (parseReviewJson, formatReviewFeedback, reviewEpisodeBoard)
    const { parseReviewJson, formatReviewFeedback, reviewEpisodeBoard } = await import('../dist/orchestrator/reviewer.js');
    const parsedFail = parseReviewJson(
      '```json\n{"pass":false,"summary":"Incohérence S2","issues":[{"shot":"S2","severity":"error","problem":"Tenue différente","fix":"Respecter veste bleue dans bible.json"}]}\n```',
      'gemini-2.5-flash',
      'ep',
    );
    assert.equal(parsedFail.pass, false);
    assert.equal(parsedFail.issues.length, 1);
    // a reviewer that quotes a word of the script without escaping it, or whose answer is cut, is still read
    const quoted = parseReviewJson(
      '```json\n{\n "pass": false,\n "summary": "Fidèle au script",\n "issues": [\n  {"shot": "S7", "severity": "error", "problem": "Elijah dit "pouces" mais il est hors cadre", "fix": "Recadrer sur Elijah"},\n  {"shot": "S15", "severity": "warn", "problem": "Cadre vide en bas, personnage minuscule", "fix": "Resserrer"}\n ]\n}\n```',
      'm',
      'ep',
    );
    assert.equal(quoted.ok, true, 'unescaped quotes must not lose the review');
    assert.equal(quoted.salvaged, true);
    assert.equal(quoted.pass, false);
    assert.deepEqual(quoted.issues.map((i) => `${i.shot}:${i.severity}`), ['S7:error', 'S15:warn']);
    assert.match(quoted.issues[0].problem, /"pouces" mais il est hors cadre$/);
    assert.equal(quoted.issues[0].fix, 'Recadrer sur Elijah');
    const cut = parseReviewJson('```json\n{"pass": false, "summary": "Deux défauts", "issues": [{"shot": "S2", "severity": "error", "problem": "Le héros est de dos en gros plan", "fix": "Tourner le visage"}, {"shot": "S9", "severity": "err', 'm', 'ep');
    assert.equal(cut.ok, true, 'a cut answer keeps the issues that are complete');
    assert.equal(cut.pass, false);
    assert.deepEqual(cut.issues.map((i) => i.shot), ['S2']);
    const cutEarly = parseReviewJson('```json\n{"pass": false, "summary": "L\'épisode est globalement fidèle au script (casting conforme', 'm', 'ep');
    assert.equal(cutEarly.ok, false, 'a refusal with no readable issue is not a pass: the reviewer is asked again');
    assert.match(cutEarly.raw, /globalement fidèle/, 'the unreadable answer is kept');
    assert.equal(parseReviewJson('je ne peux pas répondre', 'm', 'ep').ok, false);
    const fb = formatReviewFeedback(parsedFail, 'ep');
    assert.match(fb, /CORRECTIONS REQUISES/);
    assert.match(fb, /args: \["S2"\]/);

    const fakeReviewer = {
      label: 'review-mock',
      model: 'review-flash',
      async chat({ messages }) {
        assert.equal(messages.length, 2, 'reviewer must run in a clean 2-message context');
        return {
          text: JSON.stringify({ pass: true, summary: 'Tout est cohérent.', issues: [] }),
          toolCalls: [],
        };
      },
    };
    const report = await reviewEpisodeBoard(fakeReviewer, dir, 'ep');
    assert.equal(report.pass, true);
    assert.ok(fs.existsSync(path.join(dir, 'out', 'ep', 'review.json')), 'review.json should be saved');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('write policy: the agent writes JSON/MD in episodes, root notes and out/, never TSX/scripts/secrets', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-write-'));
  const registry = new ToolRegistry(fsTools);
  const ctx = { projectId: 'p', projectRoot: dir, dataRoot: dir, skillDir: null, log: () => {} };
  const write = (p, c = 'x') => registry.run('write_file', { path: p, content: c }, ctx);
  try {
    assert.equal((await write('src/episodes/crush/DIRECTOR.md', '# Director')).ok, true);
    assert.equal((await write('PROJECT.md')).ok, true);
    assert.equal((await write('out/probe.txt')).ok, true);
    for (const bad of ['src/episodes/crush/shots/S1.tsx', 'pipeline/shotlist.py', 'src/engine/rig.tsx', '.env', 'providers.json', '.gitignore', 'package.json']) {
      const r = await write(bad);
      assert.equal(r.ok, false, `${bad} must be refused`);
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('context: 3 cached leading system blocks inject diffusion director & prompting guides', async () => {
  const { buildMessages } = await import('../dist/orchestrator/context.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-ctx-'));
  try {
    const sys = buildMessages({ projectRoot: dir, history: [], maxChars: 90_000 }).filter((m) => m.role === 'system');
    assert.ok(sys.length >= 3, 'at least 3 leading system prefix messages for Anthropic caching');
    const joined = sys.map((m) => m.content).join('\n');
    assert.match(joined, /2D AI DIFFUSION DIRECTOR GUIDE/);
    assert.match(joined, /NANO BANANA 2 LITE & GEMINI OMNI FLASH 2\.1 PROMPTING GUIDE/);
    assert.match(joined, /NO TITLE CARDS/);
    assert.match(joined, /NO MUSIC IN VIDEO\/AUDIO/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('db: project_runs gains a model column on an old database and records it', async () => {
  const { DatabaseSync } = await import('node:sqlite');
  const { Db } = await import('../dist/orchestrator/db.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-db-'));
  const dbPath = path.join(dir, 'old.db');
  const old = new DatabaseSync(dbPath);
  old.exec(
    `CREATE TABLE project_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, thread_id TEXT NOT NULL, elapsed_ms INTEGER NOT NULL, steps INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);` +
      `INSERT INTO project_runs (project_id, thread_id, elapsed_ms, steps, created_at) VALUES ('p', 't', 5, 1, 'x');`,
  );
  old.close();
  try {
    const db = new Db(dbPath);
    db.addRunDuration('p', 't', 10, 2, 'zai-org/GLM-5.3-Flash');
    const rows = db.raw.prepare('SELECT model FROM project_runs ORDER BY id').all();
    assert.deepEqual(rows.map((r) => r.model), [null, 'zai-org/GLM-5.3-Flash']);
    db.close();
    new Db(dbPath).close(); // reopening an already-migrated database is a no-op
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('image pick: re-rendered shots first, then contact sheets, then an even spread (never S1, S10, S11)', async () => {
  const { pickImages } = await import('../dist/orchestrator/image-pick.js');
  const stills = Array.from({ length: 36 }, (_, i) => `out/e/board/S${i + 1}.png`);
  const board = 'out/e/storyboard.jpg';
  const sheets = [1, 2, 3, 4, 5].map((n) => `out/e/sheets/sheet-0${n}.jpg`);

  assert.deepEqual(pickImages([...stills, board], ['S7', 'S3']), ['out/e/board/S7.png', 'out/e/board/S3.png'], 'asked shots, in the asked order');
  assert.deepEqual(pickImages([...stills, board, ...sheets]), sheets.slice(0, 3), 'sheets win over loose stills');
  const spread = pickImages([...stills, board]);
  assert.deepEqual(spread, ['out/e/board/S1.png', 'out/e/board/S19.png', 'out/e/board/S36.png'], 'even spread (indices 0, 18, 35), natural order, no storyboard strip');
  assert.deepEqual(pickImages(['out/e/board/S2.png', 'out/e/board/S10.png']), ['out/e/board/S2.png', 'out/e/board/S10.png'], 'few images: all, natural order');
});

test('reviewer: reads the contact sheets and the measured diffusion counts, not just the strip', async () => {
  const { reviewEpisodeBoard, diffusionMetrics } = await import('../dist/orchestrator/reviewer.js');
  const { readEpisodeShots } = await import('../dist/tools/episode-source.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-rev-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'e');
    const shotsDir = path.join(epDir, 'shots');
    const outDir = path.join(dir, 'out', 'e');
    fs.mkdirSync(shotsDir, { recursive: true });
    fs.mkdirSync(path.join(outDir, 'sheets'), { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'bible.json'),
      JSON.stringify({
        stylePrompt: '2D cel-shaded graphic novel style, crisp ink outlines',
        characters: [{ id: 'A', name: 'Alice', appearance: 'Young woman in red coat' }],
        places: [{ id: 'ROOM', name: 'Room', description: 'Cozy dim living room with wooden desk' }],
      }),
    );
    for (const [id, mv, vw] of [
      ['S1', 'push', '3/4-left'],
      ['S2', 'pan', 'left'],
      ['S3', 'pull', 'reverse'],
    ]) {
      fs.writeFileSync(
        path.join(shotsDir, `${id}.json`),
        JSON.stringify({
          id,
          place: 'ROOM',
          characters: ['A'],
          view: vw,
          cameraMove: mv,
          imagePrompt: `2D frame ${id} showing Alice inside the cozy dim living room.`,
          videoPrompt: `Alice turns toward the desk with smooth ${mv} camera motion, strictly no dialogue.`,
          sfxHint: 'soft footsteps on wood, no music',
        }),
      );
    }
    const epSrc = readEpisodeShots(epDir);
    assert.match(diffusionMetrics(epSrc), /3 plan\(s\) diffusion/);

    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ aspect: '9:16', script: [{ text: 'Bonjour.' }], shots: [] }));
    fs.writeFileSync(path.join(outDir, 'storyboard.jpg'), 'strip');
    for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) fs.writeFileSync(path.join(outDir, 'sheets', `sheet-0${n}.jpg`), `sheet${n}`);
    const seenCalls = [];
    const provider = {
      model: 'fake',
      label: 'fake',
      chat: async (o) => {
        seenCalls.push(o.messages);
        return { text: '{"pass":true,"summary":"ok","issues":[]}', toolCalls: [], finishReason: 'stop' };
      },
    };
    const report = await reviewEpisodeBoard(provider, dir, 'e');
    assert.equal(report.ok, true);
    assert.equal(seenCalls.length, 2, '8 sheets are audited in 2 batches (6 + 2) so zero sheets are dropped');
    const batch1User = seenCalls[0].find((m) => m.role === 'user');
    const batch2User = seenCalls[1].find((m) => m.role === 'user');
    assert.equal(batch1User.images.length, 6, 'batch 1 gets at most 6 sheets');
    assert.equal(batch2User.images.length, 2, 'batch 2 gets the remaining 2 sheets');
    assert.equal(Buffer.from(batch1User.images[0].data, 'base64').toString(), 'sheet1', 'sheets in natural order, not the strip');
    assert.equal(Buffer.from(batch2User.images[0].data, 'base64').toString(), 'sheet7', 'second batch starts at sheet 7');
    assert.match(batch1User.content, /3 plan\(s\) diffusion/);
    assert.match(batch1User.content, /planches 1 à 6 sur 8/);
    assert.match(batch2User.content, /planches 7 à 8 sur 8/);

    fs.rmSync(path.join(outDir, 'sheets'), { recursive: true });
    seenCalls.length = 0;
    await reviewEpisodeBoard(provider, dir, 'e');
    assert.equal(seenCalls[0].find((m) => m.role === 'user').images.length, 1, 'falls back to storyboard.jpg when there are no sheets');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('reviewer: an unreadable answer is never a validation, a text-only model is still heard, a refusal holds the render back', async () => {
  const { reviewEpisodeBoard, formatReviewFeedback, clearReviewBlock } = await import('../dist/orchestrator/reviewer.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-rev2-'));
  const epDir = path.join(dir, 'src', 'episodes', 'e');
  const outDir = path.join(dir, 'out', 'e');
  fs.mkdirSync(epDir, { recursive: true });
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(epDir, 'shots.tsx'), 'export const S1 = () => null;\n');
  fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ script: [{ text: 'Bonjour.' }], shots: [] }));
  fs.writeFileSync(path.join(outDir, 'storyboard.jpg'), 'strip');
  const replies = (list) => {
    const calls = [];
    return { calls, provider: { model: 'fake', label: 'fake', chat: async (o) => {
      calls.push(o.messages);
      const next = list.shift();
      if (next instanceof Error) throw next;
      return { text: next ?? '', toolCalls: [] };
    } } };
  };
  const refused = '{"pass":false,"summary":"non","issues":[{"shot":"S1","severity":"error","problem":"p","fix":"f"}]}';
  try {
    const twice = replies(['', 'toujours pas du json']);
    const unread = await reviewEpisodeBoard(twice.provider, dir, 'e');
    assert.equal(twice.calls.length, 2, 'asks a second time');
    assert.equal(unread.ok, false);
    const text = formatReviewFeedback(unread, 'e');
    assert.match(text, /NON VÉRIFIÉ/);
    assert.doesNotMatch(text, /VALIDÉ ✓/);
    assert.equal(unread.blocking, false);

    const second = replies(['', '{"pass":true,"summary":"ok","issues":[]}']);
    assert.equal((await reviewEpisodeBoard(second.provider, dir, 'e')).ok, true, 'the second answer counts');

    const textOnly = replies([new Error('400 image input not supported'), '{"pass":true,"summary":"ok","issues":[]}']);
    const code = await reviewEpisodeBoard(textOnly.provider, dir, 'e');
    assert.equal(code.sawImages, false);
    assert.ok(textOnly.calls[0][1].images?.length, 'first with the images');
    assert.equal(textOnly.calls[1][1].images, undefined, 'then without');
    assert.match(formatReviewFeedback(code, 'e'), /specs JSON seules/);

    const down = replies([new Error('503'), new Error('503')]);
    await assert.rejects(reviewEpisodeBoard(down.provider, dir, 'e'), /503/);

    const first = await reviewEpisodeBoard(replies([refused]).provider, dir, 'e');
    assert.equal(first.blocking, true, 'a refusal with rounds left holds the render');
    assert.equal(JSON.parse(fs.readFileSync(path.join(outDir, 'review.json'), 'utf8')).blocking, true);
    clearReviewBlock(dir, 'e');
    assert.equal(JSON.parse(fs.readFileSync(path.join(outDir, 'review.json'), 'utf8')).blocking, false);
    assert.equal((await reviewEpisodeBoard(replies([refused]).provider, dir, 'e', undefined, { final: true })).blocking, false, 'the last round never holds it');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('agent loop: the reviewer speaks after every tool result of the step, a refused board stops the render', async () => {
  const { runTurn } = await import('../dist/orchestrator/agent.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { reviewRefusal } = await import('../dist/tools/whatif.js');
  const { clearReviewBlock } = await import('../dist/orchestrator/reviewer.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-loop-'));
  const root = path.join(dir, 'p');
  const epDir = path.join(root, 'src', 'episodes', 'e');
  fs.mkdirSync(epDir, { recursive: true });
  fs.mkdirSync(path.join(root, 'out', 'e'), { recursive: true });
  fs.writeFileSync(path.join(epDir, 'shots.tsx'), 'export const S1 = () => null;\n');
  fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ script: [{ text: 'Bonjour.' }], shots: [] }));
  fs.writeFileSync(path.join(root, 'out', 'e', 'storyboard.jpg'), 'strip');
  const db = new Db(path.join(dir, 'app.db'));
  const ok = (name) => ({ name, description: name, policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: name }) });
  const registry = new ToolRegistry([{ ...ok('whatif_run'), policy: 'ask' }, ok('whatif_status'), ok('git')]);
  try {
    db.createProject('p', 'p', root);
    const th = db.createThread('p', 't');
    db.addMessage(th.id, 'user', 'go');
    let step = 0;
    const provider = { id: 'x', label: 'x', model: 'm', chat: async () => {
      step++;
      if (step === 1) return { text: '', toolCalls: [{ id: 'a', name: 'whatif_run', args: { episode: 'e', step: 'board' } }, { id: 'b', name: 'whatif_status', args: { episode: 'e' } }] };
      return { text: 'fini', toolCalls: [] };
    } };
    const reviewer = { model: 'r', label: 'r', chat: async () => ({ text: '{"pass":false,"summary":"non","issues":[{"shot":"S1","severity":"error","problem":"p","fix":"f"}]}', toolCalls: [] }) };
    const bus = { emit() {} };
    const config = { ...loadConfig(), maxSteps: 6, reviewMaxRounds: 2, approvalsMode: 'off', contextChars: 90_000 };
    await runTurn({ db, bus, config, registry, approvals: null, provider: () => provider, reviewProvider: () => reviewer }, th.id, new AbortController().signal);
    const rows = db.listMessages(th.id).map((m) => m.role);
    const firstTool = rows.indexOf('tool');
    assert.deepEqual(rows.slice(firstTool, firstTool + 3), ['tool', 'tool', 'user'], 'no user message between the two tool results');

    assert.equal(JSON.parse(fs.readFileSync(path.join(root, 'out', 'e', 'review.json'), 'utf8')).blocking, true);
    assert.match(reviewRefusal(root, 'e'), /independent reviewer refused this board[\s\S]*\[S1\] p/);
    clearReviewBlock(root, 'e');
    assert.equal(reviewRefusal(root, 'e'), null);
  } finally {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('rhythm: words land on frames, the pace follows the length, and timeline timing metrics stay visible', async () => {
  const { paceTarget, paceProblem } = await import('../dist/tools/whatif.js');
  const { shotWordFrames } = await import('../dist/orchestrator/context.js');
  const { timingMetrics, formatReviewFeedback } = await import('../dist/orchestrator/reviewer.js');

  assert.deepEqual([paceTarget(20).minCutsPerMin, paceTarget(89).minCutsPerMin, paceTarget(962).minCutsPerMin], [24, 20, 10]);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-pace-'));
  const ep = path.join(dir, 'src', 'episodes', 'e');
  fs.mkdirSync(ep, { recursive: true });
  try {
    const tl = (n, len) => ({ fps: 30, total: n * len, shots: Array.from({ length: n }, (_, i) => ({ id: `S${i + 1}`, len })), segments: Array.from({ length: n }, (_, i) => ({ shot: `S${i + 1}`, a: i * len, b: (i + 1) * len, t0: 0 })) });
    fs.writeFileSync(path.join(ep, 'timeline.json'), JSON.stringify(tl(36, 801)));
    assert.match(paceProblem(dir, 'e'), /PACE TOO SLOW for a long video .* 36 shots over 961\.2s/, 'the y-school case: 36 shots for 16 min');
    fs.writeFileSync(path.join(ep, 'timeline.json'), JSON.stringify(tl(214, 135)));
    assert.equal(paceProblem(dir, 'e'), null, '4.5 s shots pass for a long video');
    fs.writeFileSync(path.join(ep, 'timeline.json'), JSON.stringify(tl(28, 96)));
    assert.match(paceProblem(dir, 'e'), /short \(under 2\.5 min\)/, '3.2 s shots are too slow for a short now');

    fs.writeFileSync(path.join(ep, 'timeline.json'), JSON.stringify({ fps: 30, total: 120, shots: [{ id: 'S1', len: 60 }, { id: 'S2', len: 60 }], segments: [{ shot: 'S1', a: 0, b: 60, t0: 0 }, { shot: 'S2', a: 60, b: 120, t0: 0 }] }));
    fs.writeFileSync(path.join(ep, 'words.json'), JSON.stringify([{ w: "J'ai", s: 0 }, { w: 'soulevé,', s: 0.4 }, { w: 'Mes', s: 2.1 }, { w: 'ciseaux', s: 2.5 }]));
    assert.deepEqual(shotWordFrames(dir, 'e'), ["S1 (60f): J'ai@0 soulevé@12", 'S2 (60f): Mes@3 ciseaux@15'], 'frames are the shot\'s own clock, punctuation dropped');
    assert.deepEqual(shotWordFrames(dir, 'e', ['S2']), ['S2 (60f): Mes@3 ciseaux@15']);
    const report = { ok: true, pass: false, model: 'm', episode: 'e', summary: 's', issues: [{ shot: 'S2', severity: 'error', problem: 'p', fix: 'f' }], createdAt: '' };
    assert.match(formatReviewFeedback(report, 'e', shotWordFrames(dir, 'e', ['S2'])), /mot@image[\s\S]*S2 \(60f\): Mes@3 ciseaux@15/);
    assert.match(timingMetrics(tl(3, 300)), /médiane 10\.0 s, la plus longue 10\.0 s .* 3 plan\(s\) de plus de 6 s \(S1, S2, S3\)/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('agent loop: the same render failure twice stops the turn instead of retrying for hours', async () => {
  const { runTurn } = await import('../dist/orchestrator/agent.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-stop-'));
  const db = new Db(path.join(dir, 'app.db'));
  let renders = 0;
  const registry = new ToolRegistry([
    { name: 'whatif_run', description: 'r', policy: 'ask', schema: { type: 'object' }, run: async (a) => (a.step === 'render' ? (renders++, { ok: false, summary: `$ modal_render.py\nexit 124 (timeout)\n[modal] sources: ${renders}0.5 MB` }) : { ok: true, summary: 'ok' }) },
    { name: 'read_file', description: 'r', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
    { name: 'git', description: 'g', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
  ]);
  try {
    fs.mkdirSync(path.join(dir, 'p'), { recursive: true });
    db.createProject('p', 'p', path.join(dir, 'p'));
    const th = db.createThread('p', 't');
    db.addMessage(th.id, 'user', 'go');
    let n = 0;
    const provider = { id: 'x', label: 'x', model: 'm', chat: async () => {
      n++;
      // render, a successful read in between (what fooled the old loop-breaker), render again, then render forever
      const name = n % 2 === 0 ? 'read_file' : 'whatif_run';
      return { text: '', toolCalls: [{ id: `c${n}`, name, args: name === 'read_file' ? { path: 'x' } : { episode: 'e', step: 'render' } }] };
    } };
    const config = { ...loadConfig(), maxSteps: 20, reviewMaxRounds: 0, approvalsMode: 'off', contextChars: 90_000 };
    await runTurn({ db, bus: { emit() {} }, config, registry, approvals: null, provider: () => provider, reviewProvider: () => null }, th.id, new AbortController().signal);
    assert.equal(renders, 2, 'stops after the second identical failure');
    const last = db.listMessages(th.id).at(-1);
    assert.match(last.content, /Arrêt : render échoue toujours pour la même raison \(timeout\)/);
  } finally {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('vertical 9:16 pace: paceTarget and shotlist.py cut a 160s vertical episode at short rhythm (>= 20 cuts/min)', async () => {
  const { paceTarget } = await import('../dist/tools/whatif.js');
  const { execFileSync } = await import('node:child_process');
  assert.deepEqual(
    { minCutsPerMin: paceTarget(160, '9:16').minCutsPerMin, maxShotSec: paceTarget(160, '9:16').maxShotSec },
    { minCutsPerMin: 20, maxShotSec: 5 },
  );
  assert.deepEqual(
    { minCutsPerMin: paceTarget(160, '16:9').minCutsPerMin, maxShotSec: paceTarget(160, '16:9').maxShotSec },
    { minCutsPerMin: 10, maxShotSec: 9 },
  );

  // Build a 160-second synthetic French narration + words.json and verify shotlist.py cuts ~60+ shots in 9:16 vs ~35 in 16:9
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-shotlist916-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'vert');
    fs.mkdirSync(epDir, { recursive: true });
    const sentences = Array.from(
      { length: 40 },
      (_, i) => `Ensuite Léa regarde le téléphone posé sur la table numéro ${i + 1}, et Marc répond tout bas devant Luisa.`,
    );
    const words = [];
    let t = 0.2;
    for (const s of sentences) {
      const toks = s.split(/\s+/);
      for (const w of toks) {
        words.push({ w, s: Number(t.toFixed(3)), e: Number((t + 0.22).toFixed(3)) });
        t += 0.25;
      }
      t += 0.25; // clause pause around ~160s total
    }
    fs.writeFileSync(path.join(epDir, 'words.json'), JSON.stringify(words));
    const baseEp = {
      id: 'Vert',
      slug: 'vert',
      title: ['CE QUE J’AI VU', 'SUR SON TÉLÉPHONE'],
      defaultSet: 'salon',
      script: sentences.map((text, i) => ({ beat: `b${i + 1}`, text })),
      shots: [],
    };
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ ...baseEp, aspect: '9:16' }));
    execFileSync('python3', [path.resolve('pipeline/shotlist.py'), 'vert', '--project', dir], { stdio: 'pipe' });
    const ep916 = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));
    assert.ok(ep916.shots.length >= 55, `expected >= 55 shots for 160s 9:16, got ${ep916.shots.length}`);

    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ ...baseEp, aspect: '16:9' }));
    execFileSync('python3', [path.resolve('pipeline/shotlist.py'), 'vert', '--project', dir], { stdio: 'pipe' });
    const ep169 = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));
    assert.ok(ep916.shots.length > ep169.shots.length + 12, `9:16 (${ep916.shots.length}) must cut faster than 16:9 (${ep169.shots.length})`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('reviewer: keeps >25KB episode.json script, reads 9:16 from bible/episode, summarizes shots & DIRECTOR size mismatches', async () => {
  const { summarizeShotPrompts, reviewEpisodeBoard } = await import('../dist/orchestrator/reviewer.js');
  const { readEpisodeShots } = await import('../dist/tools/episode-source.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-bigrev-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'big');
    const shotsDir = path.join(epDir, 'shots');
    const outDir = path.join(dir, 'out', 'big');
    fs.mkdirSync(shotsDir, { recursive: true });
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'bible.json'),
      JSON.stringify({
        stylePrompt: '2D cel-shaded graphic novel style, clean dark ink outlines',
        characters: [{ id: 'NARR', name: 'NARR', appearance: 'Young adult in trench coat' }],
        places: [{ id: 'SALON', name: 'Nail Salon', description: 'Warm urban nail salon interior' }],
      }),
    );
    fs.writeFileSync(
      path.join(shotsDir, 'S1.json'),
      JSON.stringify({
        id: 'S1',
        place: 'SALON',
        characters: ['NARR'],
        imagePrompt: 'Wide 2D establishing shot inside the nail salon with NARR standing near the counter.',
        videoPrompt: 'NARR walks toward the reception desk with slow camera pan, strictly no dialogue.',
        sfxHint: 'soft footsteps on tile, faint door chime, no music',
      }),
    );
    fs.writeFileSync(
      path.join(shotsDir, 'S2.json'),
      JSON.stringify({
        id: 'S2',
        place: 'SALON',
        characters: [],
        imagePrompt: 'Empty salon desk surface with manicure tools arranged cleanly in moody warm light.',
        videoPrompt: 'Gentle slow push onto manicure tools on wooden counter, strictly no dialogue.',
        sfxHint: 'quiet indoor room tone, no music',
      }),
    );

    const summary = summarizeShotPrompts(readEpisodeShots(epDir));
    assert.match(summary, /### S1 \[place=SALON, chars=NARR/);
    assert.match(summary, /### S2 \[place=SALON, chars=none/);

    // Build an episode.json > 28 KB + S6 size mismatch vs DIRECTOR.md
    const bigScript = Array.from({ length: 180 }, (_, i) => ({
      beat: `beat-${i + 1}`,
      text: `Phrase narrative numéro ${i + 1} avec un témoin unique PHRASE_FINALE_${i + 1} dans le salon de manucure.`,
    }));
    const epJson = JSON.stringify(
      {
        id: 'Big',
        slug: 'big',
        aspect: '9:16',
        script: bigScript,
        shots: [
          { id: 'S1', size: 'wide', set: 'nails', note: 'ouverture' },
          { id: 'S6', size: 'insert', set: 'nails', note: 'reclassé en douce' },
        ],
      },
      null,
      2,
    );
    assert.ok(Buffer.byteLength(epJson) > 25_000, `episode.json should exceed 25 KB (got ${Buffer.byteLength(epJson)})`);
    fs.writeFileSync(path.join(epDir, 'episode.json'), epJson);
    fs.writeFileSync(
      path.join(epDir, 'DIRECTOR.md'),
      `# DIRECTOR\n\n| plan | cadrage | action |\n|---|---|---|\n| S1 | large | ouverture |\n| S6 | close | réaction visage de Léa |\n`,
    );
    fs.writeFileSync(path.join(outDir, 'storyboard.jpg'), 'fake');

    let capturedPrompt = '';
    const provider = {
      model: 'fake-rev',
      label: 'fake',
      chat: async ({ messages }) => {
        capturedPrompt = messages.find((m) => m.role === 'user')?.content ?? '';
        return { text: JSON.stringify({ pass: true, summary: 'OK', issues: [] }) };
      },
    };
    await reviewEpisodeBoard(provider, dir, 'big');
    assert.match(capturedPrompt, /orientation:\s*9:16/, 'preserves 9:16 orientation');
    assert.match(capturedPrompt, /Phrase narrative numéro 1\b/, 'script is NOT dropped when episode.json > 25 KB');
    assert.match(capturedPrompt, /écarts de taille DIRECTOR\.md ↔ episode\.json \(1\) : S6 \(DIRECTOR\.md="close" vs episode\.json="insert"\)/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('projects and threads archiving, renaming, and activity metadata over HTTP + SQLite migration', async () => {
  const { dir, app } = tmpApp();
  await new Promise((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  const port = app.server.address().port;
  const base = `http://127.0.0.1:${port}`;
  try {
    const p1 = await (
      await fetch(`${base}/api/projects`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: 'Ancien Projet' }),
      })
    ).json();
    const p2 = await (
      await fetch(`${base}/api/projects`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: 'Projet Actif' }),
      })
    ).json();
    assert.equal(p1.project.archived_at, null);
    assert.equal(p2.project.archived_at, null);

    // Create two threads in p2
    const t1 = await (
      await fetch(`${base}/api/threads`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ projectId: p2.project.id, title: 'Brouillon initial' }),
      })
    ).json();
    const t2 = await (
      await fetch(`${base}/api/threads`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ projectId: p2.project.id, title: 'Conversation principale' }),
      })
    ).json();
    assert.equal(t1.archived_at, null);
    assert.equal(t2.archived_at, null);

    // Archive thread t1 and rename t2
    const t1Arch = await (
      await fetch(`${base}/api/threads/${t1.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ archived: true }),
      })
    ).json();
    assert.equal(typeof t1Arch.archived_at, 'string');

    const t2Ren = await (
      await fetch(`${base}/api/threads/${t2.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ title: ' Montage final ' }),
      })
    ).json();
    assert.equal(t2Ren.title, 'Montage final');

    // Archive project p1
    const p1Arch = await (
      await fetch(`${base}/api/projects/${p1.project.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ archived: true, name: 'Ancien Projet (V1)' }),
      })
    ).json();
    assert.equal(typeof p1Arch.archived_at, 'string');
    assert.equal(p1Arch.name, 'Ancien Projet (V1)');

    // Verify GET /api/projects reflects archive state and thread counts
    const list = await (await fetch(`${base}/api/projects`)).json();
    const gotP1 = list.find((p) => p.id === p1.project.id);
    const gotP2 = list.find((p) => p.id === p2.project.id);
    assert.ok(gotP1.archived_at);
    assert.equal(gotP2.archived_at, null);
    assert.equal(gotP2.threadCount, 1);
    assert.equal(gotP2.archivedThreadCount, 1);
    assert.equal(typeof gotP2.lastActivityAt, 'string');

    // Unarchive project p1 and thread t1
    const p1Restored = await (
      await fetch(`${base}/api/projects/${p1.project.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ archived: false }),
      })
    ).json();
    assert.equal(p1Restored.archived_at, null);

    const t1Restored = await (
      await fetch(`${base}/api/threads/${t1.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ archived: false }),
      })
    ).json();
    assert.equal(t1Restored.archived_at, null);

    // Verify static UI serves theme toggle & archive hooks
    const indexHtml = await (await fetch(`${base}/`)).text();
    assert.match(indexHtml, /theme-toggle/);
    assert.match(indexHtml, /toggle-archived-projects/);
  } finally {
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('loop-breaker: hashes full tool args (>240 chars) without false stops, and emits assistant message on real 5x identical loop', async () => {
  const { runTurn } = await import('../dist/orchestrator/agent.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-loophash-'));
  const db = new Db(path.join(dir, 'app.db'));
  try {
    const pRoot = path.join(dir, 'p');
    fs.mkdirSync(path.join(pRoot, 'src', 'episodes', 'ep'), { recursive: true });
    fs.writeFileSync(
      path.join(pRoot, 'src', 'episodes', 'ep', 'DIRECTOR.md'),
      'LINE_MATCH\nOther text\nLINE_MATCH\n',
    );
    const fsReg = new ToolRegistry(fsTools);
    const ctx = { projectId: 'p', projectRoot: pRoot, dataRoot: dir, skillDir: null, log: () => {} };
    const dup = await fsReg.run('edit_file', { path: 'src/episodes/ep/DIRECTOR.md', edits: [{ oldText: 'LINE_MATCH', newText: 'UNIQUE' }] }, ctx);
    assert.equal(dup.ok, false);
    assert.match(dup.summary, /matches at lines 1, 3/);

    db.createProject('p', 'p', pRoot);

    // Case 1: 6 failed calls that share a 300-char prefix in args, but differ at the end -> must NOT trigger 5x stop!
    const th1 = db.createThread('p', 't1');
    db.addMessage(th1.id, 'user', 'fix shots');
    let calls1 = 0;
    const longPrefix = 'A'.repeat(300);
    const reg1 = new ToolRegistry([
      {
        name: 'edit_file',
        description: 'e',
        policy: 'auto',
        schema: { type: 'object' },
        run: async (a) => (a.tail === 6 ? { ok: true, summary: 'ok' } : { ok: false, summary: 'oldText not found' }),
      },
      { name: 'git', description: 'g', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
    ]);
    const prov1 = {
      id: 'x',
      label: 'x',
      model: 'm',
      chat: async () => {
        calls1++;
        if (calls1 <= 6) {
          return { text: '', toolCalls: [{ id: `c${calls1}`, name: 'edit_file', args: { prefix: longPrefix, tail: calls1 } }] };
        }
        return { text: 'Tout est corrigé.', toolCalls: [] };
      },
    };
    const config = { ...loadConfig(), maxSteps: 20, reviewMaxRounds: 0, approvalsMode: 'off', contextChars: 90_000 };
    await runTurn({ db, bus: { emit() {} }, config, registry: reg1, approvals: null, provider: () => prov1, reviewProvider: () => null }, th1.id, new AbortController().signal);
    assert.equal(calls1, 7, '6 distinct >240-char edit_file calls should not be killed at attempt 5');
    assert.equal(db.listMessages(th1.id).at(-1).content, 'Tout est corrigé.');

    // Case 2: 5 genuinely identical failing calls -> stops cleanly AND records an assistant message explaining why
    const th2 = db.createThread('p', 't2');
    db.addMessage(th2.id, 'user', 'loop');
    let calls2 = 0;
    const prov2 = {
      id: 'x',
      label: 'x',
      model: 'm',
      chat: async () => {
        calls2++;
        return { text: '', toolCalls: [{ id: `k${calls2}`, name: 'edit_file', args: { prefix: longPrefix, tail: 1 } }] };
      },
    };
    await runTurn({ db, bus: { emit() {} }, config, registry: reg1, approvals: null, provider: () => prov2, reviewProvider: () => null }, th2.id, new AbortController().signal);
    assert.equal(calls2, 5);
    const last2 = db.listMessages(th2.id).at(-1);
    assert.equal(last2.role, 'assistant');
    assert.match(last2.content, /Arrêt : l'outil "edit_file" a échoué 5 fois de suite/);
  } finally {
    db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('codex provider: normalizes gpt 6.1 terra/sol, converts Responses payload + images/tools, and auto-refreshes OAuth on 401', async () => {
  const http = await import('node:http');
  const { codexProvider, normalizeCodexModel, toCodexResponsesPayload } = await import('../dist/orchestrator/providers/codex.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { providerConfig, reviewProviderConfig } = await import('../dist/orchestrator/providers/index.js');

  assert.deepEqual(normalizeCodexModel('gpt 6.1 terra'), { model: 'gpt-6.1-terra', effort: 'medium' });
  assert.deepEqual(normalizeCodexModel('GPT 6.1 Sol:low'), { model: 'gpt-6.1-sol', effort: 'low' });

  const converted = toCodexResponsesPayload(
    [
      { role: 'system', content: 'System rule' },
      { role: 'user', content: 'Look at this', images: [{ mime: 'image/png', data: 'AAAA' }] },
      { role: 'assistant', content: 'Checking', toolCalls: [{ id: 'call_1', name: 'read_file', args: { path: 'a.ts' } }] },
      { role: 'tool', toolCallId: 'call_1', name: 'read_file', content: 'ok\n1| const x = 1;' },
    ],
    [{ type: 'function', function: { name: 'read_file', description: 'Read', parameters: { type: 'object' } } }],
    'gpt 6.1 terra',
  );
  assert.equal(converted.model, 'gpt-6.1-terra');
  assert.equal(converted.instructions, 'System rule');
  assert.equal(converted.tools[0].name, 'read_file');
  assert.equal(converted.input[0].content[1].type, 'input_image');
  assert.equal(converted.input[2].type, 'function_call');
  assert.equal(converted.input[3].type, 'function_call_output');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-codex-'));
  const authPath = path.join(dir, 'auth.json');
  fs.writeFileSync(
    authPath,
    JSON.stringify({
      auth_mode: 'chatgpt',
      tokens: { access_token: 'stale-token', refresh_token: 'ref-123', account_id: 'acc_42' },
    }),
  );

  let responsesHits = 0;
  let refreshHits = 0;
  const srv = http.createServer(async (req, res) => {
    if (req.url === '/oauth/token') {
      refreshHits++;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ access_token: 'fresh-token', refresh_token: 'ref-456' }));
      return;
    }
    if (req.url === '/responses') {
      responsesHits++;
      if (req.headers.authorization === 'Bearer stale-token') {
        res.writeHead(401, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'expired' }));
        return;
      }
      assert.equal(req.headers.authorization, 'Bearer fresh-token');
      assert.equal(req.headers['chatgpt-account-id'], 'acc_42');
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write(`data: ${JSON.stringify({ type: 'response.output_text.delta', delta: 'Salut ' })}\n\n`);
      res.write(
        `data: ${JSON.stringify({
          type: 'response.output_item.done',
          item: { type: 'function_call', call_id: 'c_99', name: 'repo_map', arguments: '{"maxFiles":10}' },
        })}\n\n`,
      );
      res.write(`data: ${JSON.stringify({ type: 'response.completed', response: { status: 'completed' } })}\n\n`);
      res.end();
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const port = srv.address().port;
  const db = new Db(path.join(dir, 'app.db'));
  try {
    const prov = codexProvider({
      baseURL: `http://127.0.0.1:${port}`,
      oauthTokenUrl: `http://127.0.0.1:${port}/oauth/token`,
      authPath,
      model: 'gpt 6.1 terra',
    });
    let streamed = '';
    const out = await prov.chat({
      messages: [{ role: 'user', content: 'Go' }],
      tools: [],
      onText: (d) => (streamed += d),
    });
    assert.equal(refreshHits, 1, '401 triggers OAuth refresh');
    assert.equal(responsesHits, 2);
    assert.equal(streamed, 'Salut ');
    assert.equal(out.text, 'Salut ');
    assert.deepEqual(out.toolCalls, [{ id: 'c_99', name: 'repo_map', args: { maxFiles: 10 } }]);
    const savedAuth = JSON.parse(fs.readFileSync(authPath, 'utf8'));
    assert.equal(savedAuth.tokens.access_token, 'fresh-token');

    // Verify DB codex settings override differing .env provider cleanly
    db.setSetting('provider', 'codex');
    db.setSetting('model', 'gpt-6.1-terra');
    db.setSetting('reviewProvider', 'codex');
    db.setSetting('reviewModel', 'gpt-6.1-sol');
    assert.equal(providerConfig(db).provider, 'codex');
    assert.equal(providerConfig(db).model, 'gpt-6.1-terra');
    assert.equal(reviewProviderConfig(db).provider, 'codex');
    assert.equal(reviewProviderConfig(db).model, 'gpt-6.1-sol');
  } finally {
    db.close();
    await new Promise((r) => srv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('agy provider + per-project models: live CLI model parsing, mock agy CLI (tools, images & auth fast-fail), and isolated project overrides', async () => {
  const {
    agyProvider,
    normalizeAgyModel,
    parseAgyModelsOutput,
    providerConfig,
    reviewProviderConfig,
  } = await import('../dist/orchestrator/providers/index.js');
  const { createApp } = await import('../dist/orchestrator/main.js');
  const { loadConfig } = await import('../dist/orchestrator/config.js');

  // 1. Model normalization + canonical IDs + live `agy models` stdout parsing
  assert.equal(normalizeAgyModel('3.8 flash medium'), 'gemini-3.8-flash-medium');
  assert.equal(normalizeAgyModel('carbon low'), 'gemini-4-carbon-low');
  assert.equal(normalizeAgyModel('aragon low'), 'gemini-4-argon-low');
  assert.equal(normalizeAgyModel('claude-sonnet-4-6'), 'claude-sonnet-4-6');
  assert.equal(normalizeAgyModel('claude-opus-4-6-thinking'), 'claude-opus-4-6-thinking');
  assert.equal(normalizeAgyModel('gpt-oss-120b-medium'), 'gpt-oss-120b-medium');

  const parsedAgy = parseAgyModelsOutput(
    [
      'Fetching available models...',
      'gemini-3.8-flash-high\tGemini 3.8 Flash (High)',
      'gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)',
      'gemini-3.8-flash-low      Gemini 3.8 Flash (Low)',
      'claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)',
      'claude-opus-4-6-thinking  Claude Opus 4.6 (Thinking)',
      'gpt-oss-120b-medium\tGPT-OSS 120B (Medium)',
    ].join('\n'),
  );
  assert.deepEqual(
    parsedAgy.map((m) => m.id),
    [
      'gemini-3.8-flash-high',
      'gemini-3.8-flash-medium',
      'gemini-3.8-flash-low',
      'claude-sonnet-4-6',
      'claude-opus-4-6-thinking',
      'gpt-oss-120b-medium',
    ],
  );

  // 2. Mock agy binary: test structured tool calls, reviewer image streaming, and instant auth-prompt abort
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-agy-test-'));
  const fakeAgy = path.join(dir, 'fake-agy');
  fs.writeFileSync(
    fakeAgy,
    `#!/usr/bin/env node
const fs = require('node:fs');
const args = process.argv.slice(2);
const modelIdx = args.indexOf('--model');
const model = modelIdx >= 0 ? args[modelIdx + 1] : 'unknown';
const hasSchema = args.includes('--json-schema');
const printArg = args.find((a) => a.startsWith('--print=')) || '';
const prompt = printArg.slice('--print='.length);
if (process.env.FAKE_AGY_UNAUTH === '1') {
  process.stdout.write('Authentication required. Please visit the URL to log in:\\nhttps://accounts.google.com/...\\n');
  setTimeout(() => process.exit(1), 30000);
} else if (hasSchema) {
  const out = {
    event: 'result',
    result: {
      status: 'SUCCESS',
      response: JSON.stringify({ text: 'Plan prêt (' + model + ')', toolCalls: [{ name: 'whatif_run', argsJson: '{"step":"stills"}' }] }),
      structured_output: { text: 'Plan prêt (' + model + ')', toolCalls: [{ name: 'whatif_run', argsJson: '{"step":"stills"}' }] },
      usage: { total_tokens: 42 },
    },
  };
  process.stdout.write(JSON.stringify(out) + '\\n');
} else {
  const imgMatch = prompt.match(/attached_1\\.png/);
  const imgExists = imgMatch ? fs.existsSync(require('node:path').join(process.cwd(), 'attached_1.png')) : false;
  process.stdout.write(JSON.stringify({ event: 'step_update', step_update: { step_type: 'agent_response', text_delta: '{"pass":true,' } }) + '\\n');
  const finalJson = JSON.stringify({ pass: true, summary: 'Image vue=' + imgExists + ' via ' + model, issues: [] });
  process.stdout.write(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: finalJson } }) + '\\n');
}
`,
    { mode: 0o755 },
  );

  const app = createApp({
    ...loadConfig(),
    dataRoot: dir,
    projectsDir: path.join(dir, 'projects'),
    dbPath: path.join(dir, 'app.db'),
    skillDir: null,
  });
  await new Promise((r) => app.server.listen(0, '127.0.0.1', r));
  const port = app.server.address().port;
  const base = `http://127.0.0.1:${port}`;

  try {
    // Agent call with tools
    const agentProv = agyProvider({ model: 'aragon low', agyBin: fakeAgy });
    assert.equal(agentProv.model, 'gemini-4-argon-low');
    const agentRes = await agentProv.chat({
      messages: [{ role: 'user', content: 'Lance stills' }],
      tools: [{ type: 'function', function: { name: 'whatif_run', description: 'run', parameters: { type: 'object' } } }],
    });
    assert.equal(agentRes.text, 'Plan prêt (gemini-4-argon-low)');
    assert.equal(agentRes.toolCalls.length, 1);
    assert.equal(agentRes.toolCalls[0].name, 'whatif_run');
    assert.deepEqual(agentRes.toolCalls[0].args, { step: 'stills' });

    // Reviewer call with image attachment and no tools
    const revProv = agyProvider({ model: 'carbon medium', agyBin: fakeAgy });
    let streamed = '';
    const revRes = await revProv.chat({
      messages: [{ role: 'user', content: 'Vérifie la planche', images: [{ mime: 'image/png', data: Buffer.from('PNGDATA').toString('base64') }] }],
      tools: [],
      onText: (d) => (streamed += d),
    });
    assert.ok(streamed.includes('{"pass":true,'));
    assert.match(revRes.text, /Image vue=true via gemini-4-carbon-medium/);

    // Fast fail (<1.5s, never waiting 30s) when agy asks for interactive login
    const unauthProv = agyProvider({ model: '3.8 flash medium', agyBin: fakeAgy, env: { FAKE_AGY_UNAUTH: '1' } });
    const t0 = Date.now();
    await assert.rejects(
      () => unauthProv.chat({ messages: [{ role: 'user', content: 'hi' }], tools: [] }),
      /Session Google Antigravity \(agy\) non connectée/,
    );
    assert.ok(Date.now() - t0 < 3000, 'kills unauthenticated agy prompt immediately without hanging');

    // 3. Per-project concurrent model configuration via HTTP API
    await fetch(`${base}/api/settings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        provider: 'codex',
        model: 'gpt-6.1-terra',
        reviewProvider: 'codex',
        reviewModel: 'gpt-6.1-sol',
      }),
    });

    const pCodexRes = await fetch(`${base}/api/projects`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: 'Projet Codex' }),
    });
    const { project: pCodex } = await pCodexRes.json();
    assert.equal(pCodex.effectiveModel.provider, 'codex');
    assert.equal(pCodex.effectiveModel.model, 'gpt-6.1-terra');
    assert.equal(pCodex.effectiveModel.reviewModel, 'gpt-6.1-sol');

    const pAgyRes = await fetch(`${base}/api/projects`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        name: 'Projet AGY',
        provider: 'agy',
        model: 'gemini-4-carbon-low',
        reviewProvider: 'agy',
        reviewModel: 'gemini-4-argon-medium',
      }),
    });
    const { project: pAgy } = await pAgyRes.json();
    assert.equal(pAgy.effectiveModel.provider, 'agy');
    assert.equal(pAgy.effectiveModel.model, 'gemini-4-carbon-low');
    assert.equal(pAgy.effectiveModel.reviewProvider, 'agy');
    assert.equal(pAgy.effectiveModel.reviewModel, 'gemini-4-argon-medium');

    // Verify DB-level resolution stays isolated across both projects simultaneously
    assert.equal(providerConfig(app.db, app.db.getProject(pCodex.id)).provider, 'codex');
    assert.equal(providerConfig(app.db, app.db.getProject(pCodex.id)).model, 'gpt-6.1-terra');
    assert.equal(providerConfig(app.db, app.db.getProject(pAgy.id)).provider, 'agy');
    assert.equal(providerConfig(app.db, app.db.getProject(pAgy.id)).model, 'gemini-4-carbon-low');
    assert.equal(reviewProviderConfig(app.db, app.db.getProject(pAgy.id)).model, 'gemini-4-argon-medium');

    // Patching project back to null resets to global default
    const resetRes = await fetch(`${base}/api/projects/${pAgy.id}`, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ provider: null, model: null, reviewProvider: null, reviewModel: null }),
    });
    const resetProj = await resetRes.json();
    assert.equal(resetProj.effectiveModel.provider, 'codex');
    assert.equal(resetProj.effectiveModel.model, 'gpt-6.1-terra');
  } finally {
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('two-mode reviewer: Round 1 audits 100% of 9 sheets (66 shots) & Round 2 runs high-res targeted delta review', async () => {
  const { reviewEpisodeBoard, formatReviewFeedback } = await import('../dist/orchestrator/reviewer.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-twomode-rev-'));
  try {
    const slug = 'marie66';
    const epDir = path.join(dir, 'src', 'episodes', slug);
    const shotsDir = path.join(epDir, 'shots');
    const outDir = path.join(dir, 'out', slug);
    const boardDir = path.join(outDir, 'board');
    const sheetsDir = path.join(outDir, 'sheets');
    fs.mkdirSync(shotsDir, { recursive: true });
    fs.mkdirSync(boardDir, { recursive: true });
    fs.mkdirSync(sheetsDir, { recursive: true });

    const allIds = Array.from({ length: 66 }, (_, i) => `S${i + 1}`);
    fs.writeFileSync(
      path.join(epDir, 'bible.json'),
      JSON.stringify({
        stylePrompt: '2D cel-shaded graphic novel animation, clean dark ink lines',
        characters: [{ id: 'LEA', name: 'Léa', appearance: 'Young woman in blue coat' }],
        places: [{ id: 'SALON', name: 'Salon', description: 'Warm apartment living room with mustard sofa' }],
      }),
    );
    for (const id of allIds) {
      fs.writeFileSync(
        path.join(shotsDir, `${id}.json`),
        JSON.stringify({
          id,
          place: 'SALON',
          characters: ['LEA'],
          cameraMove: 'push',
          imagePrompt: `Medium shot ${id} of Léa inside warm apartment living room.`,
          videoPrompt: `Léa reacts naturally with gentle push camera motion, strictly no dialogue.`,
          sfxHint: 'soft room ambience, fabric rustle, no music',
        }),
      );
      fs.writeFileSync(path.join(boardDir, `${id}.png`), `PNG_${id}`);
    }
    for (let n = 1; n <= 9; n++) {
      fs.writeFileSync(path.join(sheetsDir, `sheet-0${n}.jpg`), `SHEET_${n}`);
    }
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        aspect: '9:16',
        script: [{ text: 'Je regarde son téléphone sur la table basse.' }],
        shots: allIds.map((id) => ({ id, size: 'medium', set: 'salon', note: `Note ${id}` })),
      }),
    );
    fs.writeFileSync(
      path.join(epDir, 'timeline.json'),
      JSON.stringify({
        fps: 30,
        total: 66 * 60,
        segments: allIds.map((id, i) => ({ shot: id, a: i * 60, b: (i + 1) * 60 })),
        shots: allIds.map((id) => ({ id, len: 60, key: 18 })),
      }),
    );
    fs.writeFileSync(path.join(epDir, 'DIRECTOR.md'), '# DIRECTOR\nSalon chaleureux en 9:16.');

    // ── ROUND 1: Global Audit across 9 contact sheets (must inspect sheet-01..06 AND sheet-07..09!) ──
    const round1Calls = [];
    const round1Provider = {
      model: 'gpt-6.1-sol',
      label: 'Codex Reviewer',
      chat: async ({ messages }) => {
        round1Calls.push(messages);
        if (round1Calls.length === 1) {
          return {
            text: JSON.stringify({
              pass: false,
              summary: 'Défauts sur S3 et S11.',
              issues: [
                { shot: 'S3, S11', severity: 'error', problem: 'Cadrage trop large', fix: 'Passer cam en close' },
              ],
            }),
          };
        }
        return {
          text: JSON.stringify({
            pass: false,
            summary: 'Défaut sur S65 (planche 9).',
            issues: [
              { shot: 'S65', severity: 'error', problem: 'Téléphone absent de la main', fix: 'Ajouter handRProp' },
            ],
          }),
        };
      },
    };

    const r1 = await reviewEpisodeBoard(round1Provider, dir, slug);
    assert.equal(r1.mode, 'global');
    assert.equal(r1.round, 1);
    assert.equal(r1.batches, 2, '9 contact sheets run in 2 batches (6 + 3)');
    assert.equal(r1.pass, false);
    assert.equal(r1.blocking, true);
    assert.equal(round1Calls.length, 2);
    assert.equal(round1Calls[0][1].images.length, 6, 'lot 1 attaches sheets 1..6');
    assert.equal(round1Calls[1][1].images.length, 3, 'lot 2 attaches sheets 7..9');
    assert.match(round1Calls[0][1].content, /couvrant S1\.\.S48/);
    assert.match(round1Calls[1][1].content, /couvrant S49\.\.S66/);
    assert.match(round1Calls[1][1].content, /timeline=60f \(2\.0s\)/);
    assert.equal(r1.issues.length, 2, 'merges issues from both batches including S65 on sheet 9');

    const fb1 = formatReviewFeedback(r1, slug);
    assert.match(fb1, /edit_shot/, 'recommends edit_shot instead of edit_file');
    assert.match(fb1, /args: \["S3","S11","S65"\]/, 'parses multi-shot "S3, S11" + "S65" cleanly');

    // ── ROUND 2: Targeted High-Resolution Delta Re-check (attaches ONLY full-res S3.png, S11.png, S65.png!) ──
    const round2Calls = [];
    const round2Provider = {
      model: 'gpt-6.1-sol',
      label: 'Codex Reviewer',
      chat: async ({ messages }) => {
        round2Calls.push(messages);
        return {
          text: JSON.stringify({ pass: true, summary: 'S3, S11 et S65 sont corrigés.', issues: [] }),
        };
      },
    };

    const r2 = await reviewEpisodeBoard(round2Provider, dir, slug);
    assert.equal(r2.mode, 'delta', 'Round 2 switches automatically to targeted delta mode');
    assert.equal(r2.round, 2);
    assert.deepEqual(r2.targetShots, ['S3', 'S11', 'S65']);
    assert.equal(r2.pass, true);
    assert.equal(r2.blocking, false);
    assert.equal(round2Calls.length, 1, '3 target stills fit in a single high-res delta call');

    const deltaSys = round2Calls[0][0].content;
    const deltaUser = round2Calls[0][1];
    assert.match(deltaSys, /CONTRE-VISITE CIBLÉE/);
    assert.equal(deltaUser.images.length, 3, 'attaches the 3 full-res PNG stills instead of 9 contact sheets');
    assert.deepEqual(
      deltaUser.images.map((im) => Buffer.from(im.data, 'base64').toString()),
      ['PNG_S3', 'PNG_S11', 'PNG_S65'],
    );
    assert.match(deltaUser.content, /Plans ciblés vérifiés : S3, S11, S65/);
    assert.match(deltaUser.content, /Cadrage trop large/);
    assert.match(deltaUser.content, /Téléphone absent de la main/);
    assert.match(deltaUser.content, /### S65/);
    assert.doesNotMatch(deltaUser.content, /### S20/, 'untouched shots are NOT re-audited in delta mode');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('live model discovery & .env custom API: queries agy CLI, Codex models, and OpenAI /models + preserves .env provider alongside codex/agy', async () => {
  const http = await import('node:http');
  const {
    clearModelDiscoveryCache,
    discoverProviderModels,
    getEnvProviderSummary,
    getProviderPresets,
    providerConfig,
    reviewProviderConfig,
  } = await import('../dist/orchestrator/providers/index.js');
  const { createApp } = await import('../dist/orchestrator/main.js');
  const { loadConfig } = await import('../dist/orchestrator/config.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-live-models-'));
  const fakeAgy = path.join(dir, 'fake-agy');
  fs.writeFileSync(
    fakeAgy,
    `#!/usr/bin/env node
if (process.argv.includes('models')) {
  process.stdout.write([
    'gemini-3.8-flash-high     Gemini 3.8 Flash (High)',
    'gemini-3.8-flash-medium   Gemini 3.8 Flash (Medium)',
    'gemini-3.8-flash-low      Gemini 3.8 Flash (Low)',
    'claude-sonnet-4-6         Claude Sonnet 4.6 (Thinking)',
    'claude-opus-4-6-thinking  Claude Opus 4.6 (Thinking)',
    'gpt-oss-120b-medium       GPT-OSS 120B (Medium)',
  ].join('\\n') + '\\n');
  process.exit(0);
}
process.exit(1);
`,
    { mode: 0o755 },
  );

  // Mock Codex cache + auth + custom OpenAI-compatible API server (like river.ai)
  const codexDir = path.join(dir, 'codex');
  fs.mkdirSync(codexDir, { recursive: true });
  const codexAuth = path.join(codexDir, 'auth.json');
  fs.writeFileSync(codexAuth, JSON.stringify({ tokens: { access_token: 'codex-tok', account_id: 'acc_1' } }));

  let customApiAuthHeader = '';
  const apiSrv = http.createServer((req, res) => {
    if (req.url?.startsWith('/v1/models')) {
      customApiAuthHeader = req.headers.authorization || '';
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          object: 'list',
          data: [
            { id: 'GLM-5.3' },
            { id: 'deepseek-ai/DeepSeek-V4.1-Flash-1017' },
            { id: 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4' },
            { id: 'moonshotai/Kimi-K3.1-Pro' },
            { id: 'text-embedding-3-small' }, // should be filtered out
          ],
        }),
      );
      return;
    }
    if (req.url?.startsWith('/codex/models')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          models: [
            { slug: 'gpt-6-astra', display_name: 'GPT-6-Astra', visibility: 'list', input_modalities: ['text', 'image'] },
            { slug: 'gpt-6.1-terra', display_name: 'GPT-6.1 Terra', visibility: 'list', input_modalities: ['text', 'image'] },
            { slug: 'gpt-6-sol', display_name: 'GPT-6-Sol', visibility: 'list', input_modalities: ['text', 'image'] },
            { slug: 'codex-auto-review', display_name: 'Internal Helper', visibility: 'hide', input_modalities: ['text'] },
          ],
        }),
      );
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise((r) => apiSrv.listen(0, '127.0.0.1', r));
  const apiPort = apiSrv.address().port;

  const savedEnv = {
    WHATIF_PROVIDER: process.env.WHATIF_PROVIDER,
    WHATIF_BASE_URL: process.env.WHATIF_BASE_URL,
    WHATIF_MODEL: process.env.WHATIF_MODEL,
    WHATIF_API_KEY: process.env.WHATIF_API_KEY,
    WHATIF_REVIEW_PROVIDER: process.env.WHATIF_REVIEW_PROVIDER,
    WHATIF_REVIEW_BASE_URL: process.env.WHATIF_REVIEW_BASE_URL,
    WHATIF_REVIEW_MODEL: process.env.WHATIF_REVIEW_MODEL,
    WHATIF_REVIEW_API_KEY: process.env.WHATIF_REVIEW_API_KEY,
    WHATIF_AGY_BIN: process.env.WHATIF_AGY_BIN,
    WHATIF_CODEX_AUTH: process.env.WHATIF_CODEX_AUTH,
  };

  process.env.WHATIF_PROVIDER = 'openai';
  process.env.WHATIF_BASE_URL = `http://127.0.0.1:${apiPort}/v1`;
  process.env.WHATIF_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017';
  process.env.WHATIF_API_KEY = 'sk-river-secret';
  process.env.WHATIF_REVIEW_PROVIDER = 'openai';
  process.env.WHATIF_REVIEW_BASE_URL = `http://127.0.0.1:${apiPort}/v1`;
  process.env.WHATIF_REVIEW_MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4';
  process.env.WHATIF_REVIEW_API_KEY = 'sk-river-secret';
  process.env.WHATIF_AGY_BIN = fakeAgy;
  process.env.WHATIF_CODEX_AUTH = codexAuth;

  clearModelDiscoveryCache();
  const app = createApp({
    ...loadConfig(),
    dataRoot: dir,
    projectsDir: path.join(dir, 'projects'),
    dbPath: path.join(dir, 'app.db'),
    skillDir: null,
  });
  await new Promise((r) => app.server.listen(0, '127.0.0.1', r));
  const appBase = `http://127.0.0.1:${app.server.address().port}`;

  try {
    // 1. `.env` custom provider appears as first-class `env` preset
    const envSummary = getEnvProviderSummary();
    assert.equal(envSummary.available, true);
    assert.ok('env' in getProviderPresets(), 'presets expose env custom API provider');

    // 2. Live discovery via HTTP `/api/models` for agy, env, and codex
    const agyRes = await (await fetch(`${appBase}/api/models?provider=agy&refresh=1`)).json();
    assert.equal(agyRes.source, 'live');
    assert.deepEqual(
      agyRes.models.map((m) => m.id),
      [
        'gemini-3.8-flash-high',
        'gemini-3.8-flash-medium',
        'gemini-3.8-flash-low',
        'claude-sonnet-4-6',
        'claude-opus-4-6-thinking',
        'gpt-oss-120b-medium',
      ],
    );
    assert.equal(agyRes.defaultAgentModel, 'gemini-3.8-flash-medium');
    assert.equal(agyRes.defaultReviewerModel, 'claude-sonnet-4-6');

    const envRes = await (await fetch(`${appBase}/api/models?provider=env&refresh=1`)).json();
    assert.equal(envRes.source, 'live');
    assert.equal(customApiAuthHeader, 'Bearer sk-river-secret');
    assert.ok(envRes.models.some((m) => m.id === 'moonshotai/Kimi-K3.1-Pro'), 'discovers all chat models on custom API');
    assert.ok(!envRes.models.some((m) => m.id === 'text-embedding-3-small'), 'filters out embedding models');
    assert.equal(envRes.defaultAgentModel, 'deepseek-ai/DeepSeek-V4.1-Flash-1017');
    assert.equal(envRes.defaultReviewerModel, 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4');

    const codexRes = await discoverProviderModels(app.db, {
      provider: 'codex',
      baseURL: `http://127.0.0.1:${apiPort}/codex`,
      codexAuthPath: codexAuth,
      forceRefresh: true,
    });
    assert.equal(codexRes.source, 'live');
    assert.deepEqual(
      codexRes.models.map((m) => m.id),
      ['gpt-6-astra', 'gpt-6.1-terra', 'gpt-6-sol'],
      'filters out visibility:hide codex-auto-review and returns real subscription models',
    );

    // 3. Even when global DB settings are switched to `agy`, a project selecting `env` (or switching back to `env`)
    // preserves `.env` baseURL, API key, Agent model, and Reviewer model!
    await fetch(`${appBase}/api/settings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        provider: 'agy',
        model: 'gemini-3.8-flash-medium',
        reviewProvider: 'agy',
        reviewModel: 'claude-sonnet-4-6',
      }),
    });

    const pEnvRes = await fetch(`${appBase}/api/projects`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        name: 'Projet River Env',
        provider: 'env',
        model: 'moonshotai/Kimi-K3.1-Pro',
        reviewProvider: 'env',
        reviewModel: 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
      }),
    });
    const { project: pEnv } = await pEnvRes.json();
    const pEnvDb = app.db.getProject(pEnv.id);
    const effAgent = providerConfig(app.db, pEnvDb);
    const effRev = reviewProviderConfig(app.db, pEnvDb);
    assert.equal(effAgent.provider, 'env');
    assert.equal(effAgent.baseURL, `http://127.0.0.1:${apiPort}/v1`);
    assert.equal(effAgent.apiKey, 'sk-river-secret');
    assert.equal(effAgent.model, 'moonshotai/Kimi-K3.1-Pro');
    assert.equal(effRev.provider, 'env');
    assert.equal(effRev.apiKey, 'sk-river-secret');
    assert.equal(effRev.model, 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4');
  } finally {
    for (const [k, v] of Object.entries(savedEnv)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    await app.close();
    await new Promise((r) => apiSrv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('reliability & speed fixes: image deduplication, targeted layoutReport, layout skips expensiveFails, SSE retry, SIGTERM cleanup', async () => {
  const { buildMessages } = await import('../dist/orchestrator/context.js');
  const { layoutReport } = await import('../dist/orchestrator/layout.js');
  const { openAIProvider } = await import('../dist/orchestrator/providers/openai.js');
  const { exec } = await import('../dist/tools/exec.js');
  const { runTurn } = await import('../dist/orchestrator/agent.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { ToolRegistry } = await import('../dist/tools/index.js');
  const { loadConfig } = await import('../dist/orchestrator/config.js');
  const http = await import('node:http');
  const { execFileSync } = await import('node:child_process');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-rel-'));
  let sseSrv = null;
  try {
    // 1. Historical mutable PNG deduplication in buildMessages: only newest turn attaches S8.png
    const boardDir = path.join(dir, 'out', 'ep', 'board');
    fs.mkdirSync(boardDir, { recursive: true });
    // Create S28.png (blank black = fails layoutReport) and S63.png (balanced drawing across left & right halves = passes)
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=320x180', '-frames:v', '1', path.join(boardDir, 'S28.png')]);
    execFileSync('ffmpeg', [
      '-y', '-v', 'error', '-f', 'lavfi',
      '-i', 'color=c=black:s=320x180,drawbox=x=30:y=30:w=110:h=110:color=white:t=fill,drawbox=x=180:y=30:w=110:h=110:color=yellow:t=fill',
      '-frames:v', '1',
      path.join(boardDir, 'S63.png'),
    ]);

    const history = [
      { id: 1, thread_id: 't1', role: 'user', content: 'Ancien rendu S63', payload: JSON.stringify({ images: ['out/ep/board/S63.png'] }), created_at: '2026-11-11T00:01:00Z' },
      { id: 2, thread_id: 't1', role: 'assistant', content: 'Je corrige S63', payload: null, created_at: '2026-11-11T00:02:00Z' },
      { id: 3, thread_id: 't1', role: 'user', content: 'Nouveau rendu S63', payload: JSON.stringify({ images: ['out/ep/board/S63.png'] }), created_at: '2026-11-11T00:03:00Z' },
    ];
    const msgs = buildMessages({ projectRoot: dir, history, maxChars: 90_000 });
    const userMsgs = msgs.filter((m) => m.role === 'user');
    assert.equal(userMsgs.length, 2);
    assert.equal(userMsgs[0].images?.length ?? 0, 0, 'older turn must not re-attach overwritten S63.png');
    assert.match(userMsgs[0].content, /attachés uniquement au message le plus récent/);
    assert.equal(userMsgs[1].images?.length ?? 0, 1, 'newest turn attaches current S63.png');

    // 2. Targeted layoutReport(dir, ['S63']) ignores un-regenerated S28.png, while full layoutReport(dir) flags S28
    const targetedLayout = await layoutReport(boardDir, ['S63']);
    assert.ok(!targetedLayout.includes('EMPTY:'), `targeted S63 check should pass, got: ${targetedLayout}`);
    const fullLayout = await layoutReport(boardDir);
    assert.match(fullLayout, /FAILED_SHOTS:\s*S28/);

    // 3. runTurn: 4 layout/ink checks ("FAILED_SHOTS: S...") on different shots do NOT trigger expensiveFails
    const db = new Db(path.join(dir, 'app.db'));
    db.createProject('lay-test', 'LayTest', dir);
    const thread = db.createThread('lay-test', 'Run');
    db.addMessage(thread.id, 'user', 'Go');
    let stillsCount = 0;
    const reg = new ToolRegistry([
      {
        name: 'whatif_run',
        description: 'mock pipeline',
        policy: 'auto',
        schema: { type: 'object' },
        async run(input) {
          stillsCount++;
          const shot = input.args?.[0] ?? `S${stillsCount}`;
          if (stillsCount <= 4) {
            return {
              ok: false,
              summary: `layout (measured on the stills):\nEMPTY: ${shot} leaves the right half dead\nFAILED_SHOTS: ${shot}\nOne or more shots failed the layout/ink check (see EMPTY above)`,
            };
          }
          return { ok: true, summary: 'all stills balanced' };
        },
      },
      { name: 'git', description: 'g', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
    ]);
    let llmCalls = 0;
    const fakeProv = {
      id: 'fake',
      label: 'fake',
      model: 'fake',
      async chat() {
        llmCalls++;
        if (llmCalls <= 5) {
          return {
            text: '',
            toolCalls: [{ id: `c${llmCalls}`, name: 'whatif_run', args: { episode: 'ep', step: 'stills', args: [`S${llmCalls}`] } }],
            finishReason: 'tool_calls',
          };
        }
        return { text: 'Épisode terminé sans faux arrêt !', toolCalls: [], finishReason: 'stop' };
      },
    };
    const config = { ...loadConfig(), maxSteps: 12, reviewMaxRounds: 0, approvalsMode: 'off', contextChars: 60_000 };
    await runTurn(
      {
        db,
        bus: { emit() {} },
        config,
        registry: reg,
        approvals: null,
        provider: () => fakeProv,
        reviewProvider: () => null,
      },
      thread.id,
      new AbortController().signal,
    );
    assert.equal(stillsCount, 5, '4 FAILED_SHOTS layout warnings did not kill the turn via expensiveFails');
    const finalMsgs = db.listMessages(thread.id);
    assert.equal(finalMsgs.at(-1).content, 'Épisode terminé sans faux arrêt !');
    db.close();

    // 4. openAIProvider retries transient in-stream SSE error ("Inference is temporarily unavailable. Please retry.")
    let sseHits = 0;
    sseSrv = http.createServer((_req, res) => {
      sseHits++;
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      if (sseHits === 1) {
        res.end('data: {"error":{"message":"Inference is temporarily unavailable. Please retry."}}\n\n');
        return;
      }
      res.write('data: {"choices":[{"delta":{"content":"Reprise SSE OK"},"finish_reason":null}]}\n\n');
      res.write('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n');
      res.end('data: [DONE]\n\n');
    });
    await new Promise((r) => sseSrv.listen(0, '127.0.0.1', r));
    const ssePort = sseSrv.address().port;
    const oai = openAIProvider({
      baseURL: `http://127.0.0.1:${ssePort}/v1`,
      apiKey: 'test',
      model: 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
      sseRetryBaseMs: 15,
    });
    const sseRes = await oai.chat({ messages: [{ role: 'user', content: 'ping' }], tools: [] });
    assert.equal(sseHits, 2, 'retried after transient SSE error');
    assert.equal(sseRes.text, 'Reprise SSE OK');

    // 5. exec sends SIGTERM before SIGKILL so child processes can tear down external resources (e.g. Modal app)
    const cleanupMarker = path.join(dir, 'sigterm_cleaned.txt');
    const ac = new AbortController();
    const safety = setTimeout(() => ac.abort(), 3_000);
    await exec(
      [
        process.execPath,
        '-e',
        `const fs = require('node:fs'); process.on('SIGTERM', () => { fs.writeFileSync(${JSON.stringify(cleanupMarker)}, 'cleaned'); process.exit(143); }); console.log('READY'); setInterval(() => {}, 1000);`,
      ],
      {
        cwd: dir,
        timeoutMs: 5_000,
        signal: ac.signal,
        onLine: (line) => {
          if (line.trim() === 'READY') ac.abort();
        },
      },
    );
    clearTimeout(safety);
    assert.equal(fs.readFileSync(cleanupMarker, 'utf8'), 'cleaned', 'child received SIGTERM and ran cleanup handler');
  } finally {
    if (sseSrv) await new Promise((r) => sseSrv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});


test('step 2 — shotlist.py: 3-gear cold-open hook + brisk long 16:9 pace, semantic inserts, looks & mid-episode fx', async () => {
  const { execFileSync } = await import('node:child_process');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-step2-shotlist-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'er-long');
    fs.mkdirSync(epDir, { recursive: true });

    // Build a ~240s (4-minute) realistic French narrative with speech verbs, props, amounts, flashbacks, and ER crisis twists
    const clauses = [];
    for (let i = 0; i < 64; i++) {
      if (i === 4) {
        clauses.push("Trois ans plus tôt je me souviens encore de cette vieille maison calme où mon père souriait.");
      } else if (i === 18) {
        clauses.push("Aux urgences à 3h14 du matin le médecin regarde la facture de 15000 euros posée sur la table.");
      } else if (i === 32) {
        clauses.push("Soudain c'est la panique totale dans le couloir et l'alarme hurle devant nous !");
      } else if (i === 48) {
        clauses.push("Elle ouvre l'enveloppe avec ses mains qui tremblent et lit le message sur son téléphone.");
      } else if (i % 5 === 2) {
        clauses.push(`Ensuite Clara pose la clé et le dossier médical numéro ${i} sur le bureau avant de répondre doucement.`);
      } else {
        clauses.push(`Marc s'approche doucement dans le salon numéro ${i} et demande pourquoi tout le monde garde le silence.`);
      }
    }

    const words = [];
    let t = 0.2;
    for (const c of clauses) {
      const toks = c.split(/\s+/);
      for (let wi = 0; wi < toks.length; wi++) {
        const raw = toks[wi];
        const m = /^(.+?)([.,!?]+)?$/.exec(raw);
        const w = m ? m[1] : raw;
        const p = m && m[2] ? m[2] : wi === toks.length - 1 ? '.' : '';
        words.push({ w, p, s: Number(t.toFixed(3)), e: Number((t + 0.23).toFixed(3)) });
        t += 0.26;
      }
      t += 0.32;
    }

    fs.writeFileSync(path.join(epDir, 'words.json'), JSON.stringify(words));
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        id: 'ErLong',
        slug: 'er-long',
        aspect: '16:9',
        title: ['LA NUIT OÙ', 'TOUT A BASCULÉ'],
        source: { provider: 'nouveau', videoId: 'vid-er-1' },
        defaultSet: 'hospital',
        script: clauses.map((text, idx) => ({ beat: `b${idx + 1}`, text })),
        shots: [],
      }),
    );

    execFileSync('python3', [path.resolve('pipeline/shotlist.py'), 'er-long', '--project', dir], { stdio: 'pipe' });
    const ep = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));

    const totalSec = words[words.length - 1].e;
    const cutsPerMin = ((ep.shots.length - 1) / totalSec) * 60;
    assert.ok(cutsPerMin >= 17.5, `long 16:9 episode cuts briskly (>= 17.5 cuts/min vs old 13.1), got ${cutsPerMin.toFixed(1)}`);

    const inserts = ep.shots.filter((s) => s.size === 'insert');
    assert.ok(inserts.length >= 8, `speech verbs no longer block semantic inserts (expected >= 8 inserts, got ${inserts.length})`);

    const looks = ep.edit.map((e) => e.look).filter(Boolean);
    assert.ok(looks.includes('memory'), `flashback clause tagged with look="memory" (got ${JSON.stringify(looks)})`);
    assert.ok(looks.includes('tension'), `crisis clause tagged with look="tension" (got ${JSON.stringify(looks)})`);

    const fxShots = new Set(ep.fx.map((f) => f.at));
    assert.ok(fxShots.size >= 2, `mid-episode dramatic twists also receive fx, not only the last shot (${[...fxShots].join(', ')})`);

    assert.equal(ep.shots.some((s) => /title|schematic/i.test(String(s.size || '') + String(s.accent || ''))), false, 'no title or schematic shots in 100% diffusion mode');
    assert.deepEqual(ep.sound, { provider: 'omni-flash', music: false, dialogue: false });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('step 5 — post-render speed: master_run.py (-vn + -14 LUFS + mtime cache), motiongrid_run.py fast-seek & non-polluting whatif_status', NEEDS_NUMPY, async () => {
  const { execFileSync } = await import('node:child_process');
  const { defaultRegistry } = await import('../dist/tools/index.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-step5-postrender-'));
  try {
    const slug = 'fast-post';
    const epDir = path.join(dir, 'src', 'episodes', slug);
    const outDir = path.join(dir, 'out', slug);
    const pubDir = path.join(dir, 'public', slug);
    fs.mkdirSync(path.join(outDir, 'board'), { recursive: true });
    fs.mkdirSync(epDir, { recursive: true });
    fs.mkdirSync(pubDir, { recursive: true });

    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ id: 'FastPost', slug, aspect: '16:9', script: [{ beat: 'b1', text: 'Test.' }], shots: [{ id: 'S1' }, { id: 'S2' }] }));
    fs.writeFileSync(
      path.join(epDir, 'timeline.json'),
      JSON.stringify({
        fps: 30,
        total: 120,
        shots: [{ id: 'S1', len: 60, key: 15 }, { id: 'S2', len: 60, key: 15 }],
        segments: [{ shot: 'S1', a: 0, b: 60 }, { shot: 'S2', a: 60, b: 120 }],
      }),
    );
    fs.writeFileSync(path.join(outDir, 'storyboard.jpg'), 'jpg');
    fs.writeFileSync(path.join(outDir, 'board', 'S1.png'), 'png');

    // Synthesize a 4s 1080p MP4 with quiet voice (-19 LUFS) + transient spike so old linear=true clamped at -14.8 LUFS
    const rawMp4 = path.join(outDir, `${slug}-raw.mp4`);
    execFileSync(
      'ffmpeg',
      [
        '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'color=c=navy:s=1920x1080:r=30:d=4',
        '-f', 'lavfi', '-i', "aevalsrc='0.14*sin(2*PI*220*t)+if(between(t,1.0,1.04),0.68*sin(2*PI*880*t),0)':s=48000:d=4",
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', rawMp4,
      ],
      { stdio: 'pipe' },
    );

    // 1. master_run.py --skip-render hits -14 LUFS cleanly using -vn, then hits instant mtime cache on 2nd call
    const out1 = execFileSync('python3', [path.resolve('pipeline/master_run.py'), slug, '--skip-render'], { cwd: dir, encoding: 'utf8' });
    assert.match(out1, /loudness OK/);
    assert.doesNotMatch(out1, /! loudness off target/);
    const rep = JSON.parse(fs.readFileSync(path.join(outDir, 'report.json'), 'utf8'));
    assert.ok(Math.abs(rep.lufs + 14.0) <= 0.8, `expected lufs near -14.0, got ${rep.lufs}`);
    assert.ok(fs.existsSync(path.join(outDir, `${slug}-final.mp4`)));

    const tCache0 = Date.now();
    const out2 = execFileSync('python3', [path.resolve('pipeline/master_run.py'), slug, '--skip-render'], { cwd: dir, encoding: 'utf8' });
    assert.match(out2, /loudness OK \(cached\)/);
    assert.match(out2, /"cached":\s*true/);
    assert.ok(Date.now() - tCache0 < 3000, 'second master call returns cached report without re-encoding');

    // 2. motiongrid_run.py extracts shot grids quickly in one pass per shot without leaving _f.png behind
    const mgOut = execFileSync('python3', [path.resolve('pipeline/motiongrid_run.py'), slug, 'S1', 'S2'], { cwd: dir, encoding: 'utf8' });
    assert.match(mgOut, /RESULT_JSON:\{"ok":\s*true/);
    assert.equal(fs.existsSync(path.join(outDir, 'motion', 'S1.jpg')), true);
    assert.equal(fs.existsSync(path.join(outDir, 'motion', 'S2.jpg')), true);
    assert.equal(fs.existsSync(path.join(outDir, 'motion', '_f.png')), false);

    // 3. whatif_status returns status JSON without re-attaching storyboard/board images
    const statusRes = await defaultRegistry().run('whatif_status', { episode: slug }, { projectId: 'p', projectRoot: dir, dataRoot: dir, skillDir: dir, log: () => {} });
    assert.equal(statusRes.ok, true);
    assert.deepEqual(statusRes.artifacts, [], 'whatif_status must not re-inject old storyboard/still images');
    assert.equal(statusRes.data.final, true);

    // 4. shell-policy blocks accidental local CPU video renders and legacy O(N^2) skill scripts
    const { checkCommand } = await import('../dist/tools/shell-policy.js');
    assert.equal(checkCommand('npx remotion render src/index.ts Ep out.mp4').ok, false);
    assert.equal(checkCommand('python3 /skill/scripts/master.py ep --target -14.3').ok, false);
    assert.equal(checkCommand('python3 /skill/scripts/motiongrid.py ep').ok, false);

    // 5. exec() kills the whole process group on timeout so grandchild processes holding stdio never hang forever
    const { exec } = await import('../dist/tools/exec.js');
    const tTree0 = Date.now();
    const treeRes = await exec(
      [
        process.execPath,
        '-e',
        `const { spawn } = require('node:child_process'); spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'inherit' }); setInterval(() => {}, 1000);`,
      ],
      { cwd: dir, timeoutMs: 150 },
    );
    assert.equal(treeRes.timedOut, true);
    assert.ok(Date.now() - tTree0 < 5_000, 'exec resolved promptly despite grandchild inheriting stdio pipes');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('narrator voice detection (Léa vs Hugo, 1st & 3rd person) and :8005 wrong-gender TTS cache guard', NEEDS_NUMPY, async () => {
  const { execFile, execFileSync } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const execFileAsync = promisify(execFile);
  const http = await import('node:http');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-narrator-voice-'));
  let srv = null;
  try {
    // 1. Verify detect_narrator_voice on tricky real-world French stories
    const pyCheck = execFileSync(
      'python3',
      [
        '-c',
        `
import json, sys
sys.path.insert(0, 'pipeline')
from nouveau_client import detect_narrator_voice

cases = {
  # 1st-person girl with colloquial negation + state/reflexive participles ("the-er")
  "the_er": "[nervous] Je suis assise là aux urgences. Je me suis allongée sur le lit, j'étais pas assez inquiète au début.",
  # 1st-person girl with -ée reflexives + COI invariable ("demandé", "rendu compte") + direct vocative ("not-father" / "clee")
  "not_father": "Je me suis demandé quoi faire et je me suis rendu compte du danger. Je me suis jetée au sol et je me suis accroupie. Il m'a murmuré : « Bonne fille. »",
  # 3rd-person female protagonist ("accenseur")
  "accenseur": "Elle marche seule dans la nuit pour rentrer chez elle. Soudain, elle remarque un homme. Terrifiée, elle accélère mais il la suit. Elle court vers l'ascenseur, elle appuie sur le bouton et elle retient son souffle.",
  # 3rd-person female protagonist ("cancer")
  "cancer": "Tout le monde pensait que sa sœur avait un cancer. Elle rasait ses cheveux en cachette. Quand elle entrait au salon, elle jouait la comédie, mais elle cachait la vérité alors qu'elle allait très bien.",
  # 1st-person male essay talking extensively about women ("ce-que-veulent-les-femmes")
  "male_essay": "Les femmes veulent qu'on leur dise qu'elle est belle. J'ai une amie qui m'a dit ça, tu sais ce qu'elle veut ? Si j'ai raté quelque chose, c'est parce que je ne suis pas une femme, je suis fatigué de deviner.",
  # 3rd-person male story ("prince")
  "male_third": "[urgent] Ce prince croyait toute sa ville muette. Il traversait le palais chaque matin sans entendre un mot.",
}
out = {k: detect_narrator_voice(v) for k, v in cases.items()}
print(json.dumps(out))
`,
      ],
      { encoding: 'utf8' },
    );
    const det = JSON.parse(pyCheck);
    assert.equal(det.the_er.voiceName, 'Léa');
    assert.equal(det.the_er.pov, 'first');
    assert.equal(det.not_father.voiceName, 'Léa', 'COI invariable demandé/rendu compte must not cancel female -ée reflexives');
    assert.equal(det.accenseur.voiceName, 'Léa');
    assert.equal(det.accenseur.pov, 'third');
    assert.equal(det.cancer.voiceName, 'Léa');
    assert.equal(det.cancer.pov, 'third');
    assert.equal(det.male_essay.voiceName, 'Hugo');
    assert.equal(det.male_third.voiceName, 'Hugo');
    assert.equal(det.male_third.pov, 'third');

    // 2. Verify voice_fish.py & source_import.py never let an old Hugo MP3 on :8005 overwrite Léa,
    //    while still reusing user-uploaded ("upload") audio when appropriate.
    const fakeMp3 = path.join(dir, 'sample.mp3');
    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=330:duration=1.2', '-ar', '44100', '-ac', '1', fakeMp3],
      { stdio: 'pipe' },
    );
    const mp3Bytes = fs.readFileSync(fakeMp3);

    let checkVoiceTag = 'elevenlabs:3Tp5mYhT9wrfnry9pzvV'; // Stale Hugo on :8005
    const generatedCalls = [];
    let transcriptExists = true;
    let noProposal = false;
    let deepgramCalls = 0;

    srv = http.createServer((req, res) => {
      const u = req.url || '';
      if (req.method === 'POST' && u.includes('/deepgram')) {
        // a video whose audio track cannot be read
        deepgramCalls++;
        req.resume();
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Failed to extract audio from video.' }));
        return;
      }
      if (req.method === 'GET' && u.startsWith('/api/videos/vid-fem')) {
        if (u.includes('/transcript/check')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ exists: transcriptExists, file_exists: transcriptExists }));
          return;
        }
        if (u.endsWith('/translations') && noProposal) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ proposals: [] }));
          return;
        }
        if (u.includes('/transcript')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ transcript: [{ start: 0, end: 2.0, text: 'I lay down on the stretcher.' }] }));
          return;
        }
        if (u.endsWith('/tts/check')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ exists: true, file_exists: true, id: 11, path: '/downloads/tts/old.mp3', voice: checkVoiceTag, duration_seconds: 1.2 }));
          return;
        }
        if (u.endsWith('/translations')) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(
            JSON.stringify({
              proposals: [
                // custom videos list a shorter variant first: the selected one must still win
                ...(transcriptExists ? [] : [{ id: 8, is_selected: 0, prompt_type: 'short_fiction_alternative', model_name: 'test', title: 'Variante', translation: 'Une autre histoire, plus courte.' }]),
                {
                  id: 7,
                  is_selected: true,
                  prompt_type: 'narration',
                  model_name: 'test',
                  title: 'Urgences',
                  translation: "Je me suis allongée sur le brancard et j'étais pas assez inquiète.",
                },
              ],
            }),
          );
          return;
        }
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ id: 'vid-fem', title: 'Urgences', duration_seconds: 30 }));
        return;
      }
      if (req.method === 'POST' && u.includes('/translations/7/select')) {
        req.resume();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      if (req.method === 'POST' && u.endsWith('/tts/elevenlabs')) {
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          const parsed = JSON.parse(body);
          generatedCalls.push(parsed);
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ id: 99, audioPath: '/downloads/tts/lea_fresh.mp3' }));
        });
        return;
      }
      if (req.method === 'POST' && u.includes('/tts/transcribe')) {
        req.resume();
        req.on('end', () => {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ transcript: [{ start: 0.05, end: 1.1, text: 'Je me suis allongée' }] }));
        });
        return;
      }
      if (req.method === 'GET' && u.startsWith('/downloads/tts/')) {
        res.writeHead(200, { 'content-type': 'audio/mpeg' });
        res.end(mp3Bytes);
        return;
      }
      req.resume();
      res.writeHead(404);
      res.end('not found');
    });
    await new Promise((r) => srv.listen(0, '127.0.0.1', r));
    const port = srv.address().port;
    const env = { ...process.env, WHATIF_NOUVEAU_URL: `http://127.0.0.1:${port}` };

    const slug = 'fem-ep';
    const epDir = path.join(dir, 'src', 'episodes', slug);
    fs.mkdirSync(epDir, { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({ id: 'FemEp', slug, aspect: '9:16', source: { provider: 'nouveau', videoId: 'vid-fem' }, script: [], shots: [] }),
    );

    // 2a. source_import.py keeps Léa even though :8005 has an old Hugo TTS
    await execFileAsync('python3', [path.resolve('pipeline/source_import.py'), slug, '--video', 'vid-fem', '--apply-script'], { cwd: dir, env });
    const epAfterImport = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));
    assert.equal(epAfterImport.voice.voiceName, 'Léa');
    assert.equal(epAfterImport.voice.voiceId, 'KSyQzmsYhFbuOhqj1Xxv');
    assert.equal(epAfterImport.voice.narratorGender, 'female');

    // 2b. voice_fish.py ignores the stale Hugo MP3 and generates fresh ElevenLabs TTS with Léa
    await execFileAsync('python3', [path.resolve('pipeline/voice_fish.py'), slug], { cwd: dir, env });
    assert.equal(generatedCalls.length, 1, 'must regenerate TTS when :8005 cached voice is Hugo instead of Léa');
    assert.equal(generatedCalls[0].voiceId, 'KSyQzmsYhFbuOhqj1Xxv');
    const meta1 = JSON.parse(fs.readFileSync(path.join(dir, 'out', slug, 'voice_meta.json'), 'utf8'));
    assert.equal(meta1.voiceId, 'KSyQzmsYhFbuOhqj1Xxv');
    assert.equal(meta1.audioPath, '/downloads/tts/lea_fresh.mp3');

    // 2c. When :8005 has a user-uploaded audio ("upload"), voice_fish.py reuses it without calling ElevenLabs
    checkVoiceTag = 'upload';
    fs.rmSync(path.join(dir, 'public', slug, 'voice_raw.mp3'), { force: true });
    await execFileAsync('python3', [path.resolve('pipeline/voice_fish.py'), slug], { cwd: dir, env });
    assert.equal(generatedCalls.length, 1, 'uploaded user audio on :8005 is reused without calling ElevenLabs');
    const meta2 = JSON.parse(fs.readFileSync(path.join(dir, 'out', slug, 'voice_meta.json'), 'utf8'));
    assert.equal(meta2.audioPath, '/downloads/tts/old.mp3');

    // 2d. No English transcript and an audio track :8005 cannot read, but the French text is already there:
    //     the import takes it and never asks for a transcription
    transcriptExists = false;
    const ok = await execFileAsync('python3', [path.resolve('pipeline/source_import.py'), slug, '--video', 'vid-fem'], { cwd: dir, env });
    assert.match(ok.stdout, /"ok": ?true/);
    assert.equal(deepgramCalls, 0, 'a French proposal on :8005 makes the English transcription useless');
    assert.match(
      JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8')).script[0].text,
      /allongée/,
      'no "hook" proposal: the one selected on :8005 is taken, not the first of the list',
    );

    // 2e. Neither transcript nor French text: the import fails once, and says it is not worth retrying
    noProposal = true;
    const ko = await execFileAsync('python3', [path.resolve('pipeline/source_import.py'), slug, '--video', 'vid-fem'], { cwd: dir, env }).catch((e) => e);
    assert.equal(ko.code, 4);
    assert.match(ko.stdout, /Failed to extract audio/);
    assert.match(ko.stdout, /do NOT retry/);
  } finally {
    if (srv) {
      srv.closeAllConnections?.();
      await new Promise((r) => srv.close(r));
    }
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('review: the reviewer is not asked again when nothing that could fix the refused shots changed', async () => {
  const { refusedUnchanged } = await import('../dist/orchestrator/reviewer.js');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-stuck-'));
  try {
    const ep = path.join(root, 'src', 'episodes', 'e', 'shots');
    fs.mkdirSync(ep, { recursive: true });
    fs.mkdirSync(path.join(root, 'out', 'e'), { recursive: true });
    for (const f of ['S1.json', 'S6.json']) fs.writeFileSync(path.join(ep, f), '{}');
    fs.writeFileSync(path.join(root, 'src', 'episodes', 'e', 'bible.json'), '{}');
    const old = new Date(Date.now() - 60_000);
    for (const f of [path.join(ep, 'S1.json'), path.join(ep, 'S6.json'), path.join(root, 'src', 'episodes', 'e', 'bible.json')]) fs.utimesSync(f, old, old);
    fs.writeFileSync(path.join(root, 'out', 'e', 'review.json'), JSON.stringify({
      ok: true, pass: false, blocking: true, createdAt: new Date(Date.now() - 30_000).toISOString(),
      issues: [{ shot: 'S6', severity: 'error', problem: 'p', fix: 'f' }],
    }));
    assert.deepEqual(refusedUnchanged(root, 'e'), ['S6']);
    // an edit of a shot that was NOT refused does not count
    fs.writeFileSync(path.join(ep, 'S1.json'), '{"changed":true}');
    assert.deepEqual(refusedUnchanged(root, 'e'), ['S6']);
    // the refused shot edited: the reviewer may look again
    fs.writeFileSync(path.join(ep, 'S6.json'), '{"fixed":true}');
    assert.equal(refusedUnchanged(root, 'e'), null);
    // so does bible.json
    fs.utimesSync(path.join(ep, 'S6.json'), old, old);
    assert.deepEqual(refusedUnchanged(root, 'e'), ['S6']);
    fs.writeFileSync(path.join(root, 'src', 'episodes', 'e', 'bible.json'), '{"updated":true}');
    assert.equal(refusedUnchanged(root, 'e'), null);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('review: the last round still refused ships the video with a list of the unresolved defects', async () => {
  const { writeUnresolvedDefects, formatReviewFeedback } = await import('../dist/orchestrator/reviewer.js');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-defects-'));
  try {
    const report = { ok: true, pass: false, model: 'm', episode: 'e', summary: 's', round: 2, createdAt: new Date().toISOString(), issues: [{ shot: 'S6', severity: 'error', problem: 'JULES_KID after the jump', fix: 'use JULES_TEEN' }] };
    const rel = writeUnresolvedDefects(root, 'e', report);
    assert.equal(rel, 'out/e/DEFAUTS-NON-CORRIGES.md');
    assert.match(fs.readFileSync(path.join(root, rel), 'utf8'), /S6.*JULES_KID after the jump/s);
    assert.equal(writeUnresolvedDefects(root, 'e', { ...report, pass: true, issues: [] }), null);
    assert.match(formatReviewFeedback({ ...report, defectsFile: rel }, 'e'), /DERNIER TOUR.*Vidéo livrée avec 1 défaut/s);
    assert.doesNotMatch(formatReviewFeedback(report, 'e'), /DERNIER TOUR/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('shot_words: the words of a shot with their frames, size and length, without shell or python', async () => {
  const { shotWords } = await import('../dist/tools/shot-words.js');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-words-'));
  try {
    const dir = path.join(root, 'src', 'episodes', 'e');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'episode.json'), JSON.stringify({ shots: [{ id: 'S1', size: 'close' }] }));
    fs.writeFileSync(path.join(dir, 'words.json'), JSON.stringify([{ w: 'Bonjour', s: 0.5, e: 0.9 }, { w: 'toi', s: 1, e: 1.3 }]));
    fs.writeFileSync(path.join(dir, 'timeline.json'), JSON.stringify({ fps: 30, shots: [{ id: 'S1', len: 60 }], segments: [{ shot: 'S1', a: 0, b: 60, t0: 0 }] }));
    const r = await shotWords.run({ episode: 'e', ids: ['s1'] }, { projectRoot: root });
    assert.match(r.summary, /S1 \[close\] \(60f\): Bonjour@15 toi@30/);
    await assert.rejects(() => shotWords.run({ episode: 'nope' }, { projectRoot: root }), /does not exist/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

const stageCard = (id, o = {}) => ({
  id,
  see: `Plan ${id} : la prof retourne le sac de Maya au-dessus du pupitre`,
  subject: 'the bag held upside down',
  people: [{ who: 'PROF', where: 'left of d2', does: `turns the bag upside down ${id}` }],
  objects: [],
  change: 'the teacher has come closer to the desk',
  ...o,
});

test('staging cards: stage_shots saves one card per shot and refuses missing, vague, copied or repeated ones', async () => {
  const { stageShots, stagingProblems, usesStaging, formatStageCards } = await import('../dist/tools/staging.js');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-stage-'));
  try {
    const dir = path.join(root, 'src', 'episodes', 'e');
    fs.mkdirSync(dir, { recursive: true });
    const ids = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'];
    const file = path.join(dir, 'episode.json');
    fs.writeFileSync(file, JSON.stringify({ shots: ids.map((id) => ({ id, size: 'medium', note: 'elle fouille le sac', stage: null })) }));
    assert.equal(usesStaging(JSON.parse(fs.readFileSync(file, 'utf8')).shots), true);
    assert.equal(usesStaging([{ id: 'S1', size: 'wide' }]), false, 'an episode made before the cards is left alone');

    // some shots only: the rest is named
    let r = await stageShots.run({ episode: 'e', cards: [stageCard('S1'), stageCard('S9')] }, { projectRoot: root });
    assert.equal(r.ok, true, 'a first batch is not an error');
    assert.match(r.summary, /1 card\(s\) saved; unknown shot ids ignored: S9/);
    assert.match(r.summary, /5\/6 shots still need a card: S2, S3, S4, S5, S6/);
    // a bad card in a batch is said at once
    r = await stageShots.run({ episode: 'e', cards: [stageCard('S2', { see: 'Maya acquiesce.' })] }, { projectRoot: root });
    assert.equal(r.ok, false);
    assert.match(r.summary, /S2\.see is too short \(15 chars, 50 at least\)/);
    assert.match(stagingProblems(JSON.parse(fs.readFileSync(file, 'utf8')).shots).join('\n'), /4\/6 shots have no staging card \(S3, S4, S5, S6\)/);

    // all of them, each different: accepted and written
    r = await stageShots.run({ episode: 'e', cards: ids.map((id) => stageCard(id)) }, { projectRoot: root });
    assert.equal(r.ok, true, r.summary);
    const saved = JSON.parse(fs.readFileSync(file, 'utf8')).shots;
    assert.equal(saved[2].stage.people[0].who, 'PROF');
    assert.equal(saved[2].stage.id, undefined);
    assert.match(formatStageCards(saved, ['S3']), /^S3 \[medium\] Plan S3 .* — sujet : the bag held upside down\n {3}PROF \(left of d2\): turns the bag upside down S3$/);

    const shots = (over) => ids.map((id, i) => ({ id, size: 'medium', note: 'elle fouille le sac', stage: (({ id: _i, ...c }) => c)(stageCard(id, over(id, i))) }));
    assert.deepEqual(stagingProblems(shots(() => ({}))), []);
    // the voice line is not a picture; a short sentence is not one either
    assert.match(stagingProblems(shots((id) => (id === 'S2' ? { see: 'Elle fouille le sac' } : {}))).join('\n'), /S2\.see is too short/);
    assert.match(stagingProblems([{ id: 'S1', size: 'medium', note: 'La prof retourne le sac de Maya sur le pupitre de la classe', stage: { see: 'la prof retourne le sac de Maya sur le pupitre de la classe !', subject: 'bag', people: [], objects: ['bag'] } }]).join('\n'), /repeats the voice line/);
    // who must be the bible.json constant, the action must be visual, the angle must exist
    const bad = stagingProblems(shots((id) => (id === 'S3' ? { people: [{ who: 'la prof', where: '', does: 'idle' }], angle: 'dutch' } : {}))).join('\n');
    assert.match(bad, /S3\.people\[0\]\.who must be the UPPERCASE character ID/);
    assert.match(bad, /S3\.people\[0\]\.where is missing/);
    assert.match(bad, /S3\.people\[0\]\.does "idle" says nothing visual/);
    // a verb that shows nothing, alone or nearly; the same verb with a how is fine
    const vague = (does) => stagingProblems(shots((id) => (id === 'S3' ? { people: [{ who: 'PROF', where: 'left', does }] } : {}))).join('\n');
    assert.match(vague('réagit'), /S3\.people\[0\]\.does "réagit" says nothing visual/);
    assert.match(vague("change d'expression"), /says nothing visual/);
    assert.match(vague('regarde Maya'), /says nothing visual/);
    assert.match(vague('ok'), /S3\.people\[0\]\.does is too vague/);
    assert.equal(vague('regarde le sac en se penchant sur le pupitre, une main à plat'), '');
    // no subject, no picture
    assert.match(stagingProblems(shots((id) => (id === 'S4' ? { subject: '' } : {}))).join('\n'), /S4\.subject is missing/);
    assert.match(bad, /S3\.angle must be "level", "low" or "high"/);
    // a medium shot of nothing
    assert.match(stagingProblems(shots((id) => (id === 'S4' ? { people: [] } : {}))).join('\n'), /S4 has no person and no object/);
    // nothing says what changed
    assert.match(stagingProblems(shots((id) => (id === 'S5' ? { change: '' } : {}))).join('\n'), /S5\.change is missing/);
    assert.doesNotMatch(stagingProblems(shots((id) => (id === 'S1' ? { change: '' } : {}))).join('\n'), /S1\.change/);
    // two neighbours alike
    assert.match(stagingProblems(shots((id) => (id === 'S2' || id === 'S3' ? { people: [{ who: 'PROF', where: 'left', does: 'points at Maya' }] } : {}))).join('\n'), /S2 and S3 have the same card/);
    // the same action all along
    const rep = stagingProblems(shots((id, i) => ({ see: `Image ${id} : la prof explique encore devant le tableau vert`, people: [{ who: 'PROF', where: i % 2 ? 'left' : 'right', does: 'explains with one arm out' }], ...(i % 2 ? {} : { objects: ['x'] }) }))).join('\n');
    assert.match(rep, /PROF "explains with one arm out" in 6\/6 shots/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('review: a shot refused three reviews in a row ends the review, and the cards are read before the code', async () => {
  const { nextStrikes, MAX_STRIKES, reviewEpisodeBoard, reviewStageCards, formatReviewFeedback } = await import('../dist/orchestrator/reviewer.js');
  assert.equal(MAX_STRIKES, 3);
  const rep = (...shots) => ({ ok: true, pass: false, issues: shots.map((shot) => ({ shot, severity: 'error' })) });
  // refusals in a row: a fixed shot leaves the count, a new one starts at 1, several ids in one remark all count
  assert.deepEqual(nextStrikes(undefined, rep('S3', 'S9')), { S3: 1, S9: 1 });
  assert.deepEqual(nextStrikes({ S3: 1, S9: 1 }, rep('S3')), { S3: 2 });
  assert.deepEqual(nextStrikes({ S3: 2 }, rep('S3, S4')), { S3: 3, S4: 1 });
  assert.deepEqual(nextStrikes({ S3: 2 }, { ok: true, pass: true, issues: [] }), {});
  assert.deepEqual(nextStrikes({ S3: 2 }, { ok: true, pass: false, issues: [{ shot: 'S3', severity: 'warning' }] }), {}, 'a warning is not a refusal');

  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-strikes-'));
  try {
    const dir = path.join(root, 'src', 'episodes', 'e');
    fs.mkdirSync(path.join(dir, 'shots'), { recursive: true });
    const card = (id, see) => ({ see, subject: 'the bag', people: [{ who: 'PROF', where: 'left of d2', does: `turns the bag upside down ${id}` }], objects: ['bag'], change: 'the teacher has come closer' });
    const shots = ['S1', 'S2', 'S3'].map((id) => ({ id, size: 'medium', note: 'n', stage: card(id, `Plan ${id} : la prof retourne le sac de Maya au-dessus du pupitre`) }));
    const writeEp = (list) => fs.writeFileSync(path.join(dir, 'episode.json'), JSON.stringify({ aspect: '9:16', script: [{ text: 'La prof fouille le sac.' }], shots: list }));
    writeEp(shots);
    fs.writeFileSync(path.join(dir, 'bible.json'), JSON.stringify({ stylePrompt: '2D cel-shaded graphic novel animation, clean ink lines', characters: [{ id: 'PROF', name: 'Prof', appearance: 'Teacher in beige coat' }], places: [{ id: 'R', name: 'Classroom', description: 'Dim school classroom with wooden desks' }] }));
    for (const s of shots) fs.writeFileSync(path.join(dir, 'shots', `${s.id}.json`), JSON.stringify({ id: s.id, place: 'R', characters: ['PROF'], imagePrompt: `2D shot ${s.id} of Prof holding bag.`, videoPrompt: 'Prof turns bag upside down slowly, strictly no dialogue.', sfxHint: 'soft fabric rustle, no music' }));
    fs.writeFileSync(path.join(dir, 'timeline.json'), JSON.stringify({ fps: 30, shots: shots.map((s) => ({ id: s.id, len: 60 })), segments: [] }));

    // --- the same shot refused again and again: the third time is the last
    const refuse = { model: 'm', chat: async () => ({ text: JSON.stringify({ pass: false, summary: 's', issues: [{ shot: 'S2', severity: 'error', problem: 'la main est dans le vide', fix: 'rapproche PROF' }] }) }) };
    const r1 = await reviewEpisodeBoard(refuse, root, 'e', undefined, { round: 1 });
    assert.deepEqual(r1.strikes, { S2: 1 });
    assert.equal(r1.blocking, true);
    const r2 = await reviewEpisodeBoard(refuse, root, 'e', undefined, { round: 2 });
    assert.deepEqual(r2.strikes, { S2: 2 });
    assert.equal(r2.blocking, true, 'twice: still another chance');
    const r3 = await reviewEpisodeBoard(refuse, root, 'e', undefined, { round: 3 });
    assert.deepEqual(r3.strikes, { S2: 3 });
    assert.deepEqual(r3.gaveUpOn, ['S2']);
    assert.equal(r3.blocking, false, 'the render is not held back any longer');
    assert.equal(r3.defectsFile, 'out/e/DEFAUTS-NON-CORRIGES.md');
    assert.match(formatReviewFeedback(r3, 'e'), /ARRÊT DE LA REVUE : S2 refusé\(s\) 3 fois de suite/);
    assert.match(fs.readFileSync('src/orchestrator/agent.ts', 'utf8'), /prevRev\.gaveUpOn\.length\) reviewRounds = config\.reviewMaxRounds;/);

    // --- the user comes back to the closed review: the counts start again and the changed shots are looked at
    const { reopenReview, refusedUnchanged } = await import('../dist/orchestrator/reviewer.js');
    const reviewFile = path.join(root, 'out', 'e', 'review.json');
    assert.equal(refusedUnchanged(root, 'e'), null, 'a closed review holds nothing back');
    assert.deepEqual(reopenReview(root, 'e'), ['S2']);
    const open = JSON.parse(fs.readFileSync(reviewFile, 'utf8'));
    assert.equal(open.reopened, true);
    assert.equal(open.gaveUpOn, undefined);
    assert.equal(open.strikes, undefined);
    assert.equal(open.blocking, false, 'the render is not held back by a reopening');
    assert.equal(open.issues.length, 1, 'the defects stay: they say which shots to fix');
    assert.equal(reopenReview(root, 'e'), null, 'reopened once');
    assert.deepEqual(refusedUnchanged(root, 'e'), ['S2'], 'nothing fixed yet: no paid review for the same pictures');
    const later = new Date(Date.now() + 60_000);
    fs.utimesSync(path.join(dir, 'shots', 'S2.json'), later, later);
    assert.equal(refusedUnchanged(root, 'e'), null, 'S2 was edited: worth a look');
    const r4 = await reviewEpisodeBoard(refuse, root, 'e', undefined, { round: 1 });
    assert.deepEqual(r4.strikes, { S2: 1 }, 'three new chances, not a fourth refusal');
    assert.equal(r4.blocking, true);
    assert.equal(r4.gaveUpOn, undefined);
    assert.equal(r4.mode, r3.mode, 'looked at the way a later round is: the changed shots, not the whole board again');
    assert.equal(JSON.parse(fs.readFileSync(reviewFile, 'utf8')).reopened, undefined);
    assert.equal(reopenReview(root, 'e'), null, 'a review still in progress is not reopened');
    fs.writeFileSync(reviewFile, JSON.stringify({ ok: true, pass: true, issues: [], createdAt: new Date().toISOString() }));
    assert.equal(reopenReview(root, 'e'), null, 'an accepted board has nothing to reopen');
    assert.equal(reopenReview(root, 'nope'), null);
    const agentSrc = fs.readFileSync('src/orchestrator/agent.ts', 'utf8');
    assert.match(agentSrc, /lastMsg\?\.role === 'user' && !lastMsg\.content\.startsWith\('\[orchestrateur\]'\)/);
    assert.match(agentSrc, /else if \(prevRev\?\.reopened\) reopenedReview = true;/);

    // --- the cards, read before any prompt spec
    let calls = 0;
    let seen = '';
    const reader = (text) => ({ model: 'm', chat: async ({ messages }) => { calls++; seen = messages.map((m) => m.content).join('\n'); return { text }; } });
    const marker = path.join(root, 'out', 'e', 'cards-review.json');
    assert.equal(await reviewStageCards(reader('{"issues":[{"shot":"S1","problem":"x"}]}'), root, 'e'), null);
    assert.equal(calls, 0, 'no reading once a shot is authored');
    assert.equal(fs.existsSync(marker), false);
    for (const s of shots) fs.rmSync(path.join(dir, 'shots', `${s.id}.json`));
    const note = await reviewStageCards(reader('ok {"issues":[{"shot":"S2","problem":"aucun sujet clair","fix":"cadre sur le sac"},{"shot":"S3","problem":""}]}'), root, 'e');
    assert.match(note, /LECTURE DES FICHES, avant les prompts\] 1 fiche\(s\)/);
    assert.match(note, /- \[S2\] aucun sujet clair\n {2}→ cadre sur le sac/);
    assert.match(note, /rien n'est bloqué/);
    assert.match(seen, /La prof fouille le sac\./, 'the script is given');
    assert.match(seen, /S2 \[medium\] Plan S2 .* — sujet : the bag/, 'and every card');
    assert.match(seen, /format 9:16, 3 plans/);
    assert.equal(JSON.parse(fs.readFileSync(marker, 'utf8')).issues.length, 1);
    // once per episode: neither the same cards nor changed cards are read again
    assert.equal(await reviewStageCards(reader('{"issues":[{"shot":"S1","problem":"x"}]}'), root, 'e'), null);
    writeEp(shots.map((s, i) => (i === 0 ? { ...s, stage: card('S1', 'Plan S1 : gros plan sur le sac retourné, trois crayons tombent sur le pupitre') } : s)));
    assert.equal(await reviewStageCards(reader('{"issues":[{"shot":"S1","problem":"x"}]}'), root, 'e'), null, 'changed cards are not read a second time');
    assert.equal(calls, 1);
    // nothing to say is silence; an unreadable answer is not a verdict and leaves no marker; incomplete cards are not read at all
    fs.rmSync(marker);
    assert.equal(await reviewStageCards(reader('{"issues":[]}'), root, 'e'), null, 'nothing to say is silence');
    assert.equal(calls, 2);
    fs.rmSync(marker);
    assert.equal(await reviewStageCards(reader('je ne sais pas'), root, 'e'), null);
    assert.equal(await reviewStageCards({ model: 'm', chat: async () => { throw new Error('down'); } }, root, 'e'), null);
    assert.equal(fs.existsSync(marker), false);
    writeEp(shots.map((s, i) => (i === 1 ? { ...s, stage: null } : s)));
    const before = calls;
    assert.equal(await reviewStageCards(reader('{"issues":[{"shot":"S1","problem":"x"}]}'), root, 'e'), null);
    assert.equal(calls, before, 'no call while a card is missing');
    assert.match(fs.readFileSync('src/orchestrator/agent.ts', 'utf8'), /call\.name === 'stage_shots' && deps\.reviewProvider\)/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('pipelineTodo mtime invalidation, rerun nudge & read_file loop-breaker', async () => {
  const { episodeStatus, pipelineTodo } = await import('../dist/orchestrator/context.js');
  const { runTurn } = await import('../dist/orchestrator/agent.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { ToolRegistry } = await import('../dist/tools/index.js');
  const { loadConfig } = await import('../dist/orchestrator/config.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-rerun-'));
  try {
    const slug = 'ep1';
    const epDir = path.join(dir, 'src', 'episodes', slug);
    const shotsDir = path.join(epDir, 'shots');
    const outDir = path.join(dir, 'out', slug);
    const refsDir = path.join(outDir, 'refs');
    const boardDir = path.join(outDir, 'board');
    const clipsDir = path.join(outDir, 'clips');
    const pubDir = path.join(dir, 'public', slug);
    for (const d of [shotsDir, refsDir, boardDir, clipsDir, pubDir]) fs.mkdirSync(d, { recursive: true });

    const baseT = Math.floor(Date.now() / 1000) - 100;
    const writeAt = (f, content, sec) => {
      fs.writeFileSync(f, content);
      fs.utimesSync(f, sec, sec);
    };
    const mkShot = (id) =>
      JSON.stringify({
        id,
        place: 'P1',
        characters: ['C1'],
        imagePrompt: `2D still ${id} of C1 in P1.`,
        videoPrompt: `C1 nods in P1, strictly no dialogue.`,
        sfxHint: 'room tone, no music',
      });
    writeAt(path.join(epDir, 'episode.json'), JSON.stringify({ id: slug, script: ['Bonjour à tous voici une histoire'], shots: [{ id: 'S1', stage: {} }, { id: 'S2', stage: {} }] }), baseT);
    writeAt(path.join(epDir, 'words.json'), JSON.stringify([{ w: 'Bonjour', s: 0, e: 1.2 }]), baseT);
    writeAt(path.join(epDir, 'timeline.json'), JSON.stringify({ fps: 30, total: 600, shots: [{ id: 'S1', len: 300 }, { id: 'S2', len: 300 }] }), baseT);
    writeAt(path.join(epDir, 'DIRECTOR.md'), 'Mise en scène détaillée '.repeat(15), baseT);
    writeAt(path.join(epDir, 'bible.json'), JSON.stringify({ stylePrompt: '2D cel-shaded graphic novel style, crisp ink', characters: [{ id: 'C1', name: 'C1', appearance: 'Young adult in dark jacket' }], places: [{ id: 'P1', name: 'Room', description: 'Quiet wooden room with lamp' }] }), baseT);
    writeAt(path.join(epDir, 'shots.stage.json'), JSON.stringify({ cards: { S1: { see: 'x' }, S2: { see: 'y' } } }), baseT);
    writeAt(path.join(refsDir, 'place_P1.png'), 'png', baseT + 5);
    writeAt(path.join(refsDir, 'char_C1.png'), 'png', baseT + 5);
    writeAt(path.join(pubDir, 'voice.wav'), 'wav', baseT);
    writeAt(path.join(pubDir, 'sfx.wav'), 'wav', baseT);
    writeAt(path.join(shotsDir, 'S1.json'), mkShot('S1'), baseT + 10);
    writeAt(path.join(shotsDir, 'S2.json'), mkShot('S2'), baseT + 10);
    writeAt(path.join(boardDir, 'S1.png'), 'png', baseT + 20);
    writeAt(path.join(boardDir, 'S2.png'), 'png', baseT + 20);
    fs.utimesSync(boardDir, baseT + 20, baseT + 20);
    writeAt(path.join(outDir, 'storyboard.jpg'), 'jpg', baseT + 30);
    writeAt(path.join(clipsDir, 'S1.mp4'), 'mp4', baseT + 35);
    writeAt(path.join(clipsDir, 'S2.mp4'), 'mp4', baseT + 35);
    writeAt(path.join(outDir, `${slug}-raw.mp4`), 'mp4', baseT + 40);
    writeAt(path.join(outDir, `${slug}-final.mp4`), 'mp4', baseT + 50);

    assert.deepEqual(pipelineTodo(dir), [], 'finished episode with fresh artifacts has empty todo');

    // Editing S1.json after final.mp4 invalidates stills, board, clips, render, master
    writeAt(path.join(shotsDir, 'S1.json'), mkShot('S1'), baseT + 60);
    assert.deepEqual(pipelineTodo(dir), ['stills', 'board', 'clips', 'render', 'master']);

    // Re-running stills updates board/S1.png -> leaves board, clips, render, master
    writeAt(path.join(boardDir, 'S1.png'), 'png2', baseT + 70);
    assert.deepEqual(pipelineTodo(dir), ['board', 'clips', 'render', 'master']);

    // Re-running board + clips updates storyboard.jpg & S1.mp4 -> leaves render, master
    writeAt(path.join(outDir, 'storyboard.jpg'), 'jpg2', baseT + 80);
    writeAt(path.join(clipsDir, 'S1.mp4'), 'mp4', baseT + 82);
    assert.deepEqual(pipelineTodo(dir), ['render', 'master']);

    // Re-running render + master updates raw.mp4 & final.mp4 -> todo empty again!
    writeAt(path.join(outDir, `${slug}-raw.mp4`), 'mp42', baseT + 90);
    writeAt(path.join(outDir, `${slug}-final.mp4`), 'mp42', baseT + 95);
    assert.deepEqual(episodeStatus(dir)[0].todo, []);

    // 5. Follow-up user prompt asking to re-render / fix when final.mp4 already exists nudges the model if it tries to stop at step 0
    const db = new Db(path.join(dir, 'app.db'));
    try {
      db.createProject('p', 'p', dir);
      const config = { ...loadConfig(), maxSteps: 10, reviewMaxRounds: 0, approvalsMode: 'off', contextChars: 60_000 };
      let rerunTriggered = false;
      const regRerun = new ToolRegistry([
        {
          name: 'whatif_run',
          description: 'w',
          policy: 'auto',
          schema: { type: 'object' },
          run: async () => {
            rerunTriggered = true;
            return { ok: true, summary: 'rendered fresh video' };
          },
        },
        { name: 'git', description: 'g', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
      ]);
      let turnCall = 0;
      const provRerun = {
        id: 'fake',
        label: 'fake',
        model: 'fake',
        chat: async () => {
          turnCall++;
          if (turnCall === 1) return { text: 'La vidéo finale est déjà prête !', toolCalls: [] };
          if (turnCall === 2) return { text: '', toolCalls: [{ id: 'r1', name: 'whatif_run', args: { episode: slug, step: 'render' } }] };
          return { text: 'Nouveau rendu terminé.', toolCalls: [] };
        },
      };
      const thRerun = db.createThread('p', 'rerun');
      db.addMessage(thRerun.id, 'user', 'Recommence quand même le rendu de la vidéo');
      await runTurn({ db, bus: { emit() {} }, config, registry: regRerun, approvals: null, provider: () => provRerun, reviewProvider: () => null }, thRerun.id, new AbortController().signal);
      assert.equal(rerunTriggered, true, 'orchestrator nudges past premature "already rendered" stop on explicit rerun request');
      assert.equal(db.listMessages(thRerun.id).at(-1).content, 'Nouveau rendu terminé.');

      // 6. Single read_file between identical failing whatif_run calls does NOT reset consecutiveFails
      let failRuns = 0;
      const regReadLoop = new ToolRegistry([
        {
          name: 'whatif_run',
          description: 'w',
          policy: 'auto',
          schema: { type: 'object' },
          run: async () => {
            failRuns++;
            return { ok: false, summary: 'cannot run stills yet — S7: missing angle' };
          },
        },
        { name: 'read_file', description: 'r', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'file content' }) },
        { name: 'git', description: 'g', policy: 'auto', schema: { type: 'object' }, run: async () => ({ ok: true, summary: 'ok' }) },
      ]);
      let rlStep = 0;
      const provReadLoop = {
        id: 'fake',
        label: 'fake',
        model: 'fake',
        chat: async () => {
          rlStep++;
          return rlStep % 2 === 1
            ? { text: '', toolCalls: [{ id: `wf${rlStep}`, name: 'whatif_run', args: { episode: slug, step: 'stills' } }] }
            : { text: '', toolCalls: [{ id: `rf${rlStep}`, name: 'read_file', args: { path: 'S7.tsx' } }] };
        },
      };
      const thLoop = db.createThread('p', 'readloop');
      db.addMessage(thLoop.id, 'user', 'go');
      await runTurn({ db, bus: { emit() {} }, config: { ...config, maxSteps: 20 }, registry: regReadLoop, approvals: null, provider: () => provReadLoop, reviewProvider: () => null }, thLoop.id, new AbortController().signal);
      assert.equal(failRuns, 5, 'interleaved single read_file calls must not mask 5 identical whatif_run failures');
      const loopMsgs = db.listMessages(thLoop.id);
      assert.ok(loopMsgs.some((m) => /Lire un fichier \(read_file\) sans rien modifier ne corrige PAS/.test(m.content)));
      assert.match(loopMsgs.at(-1).content, /Arrêt : l'outil "whatif_run" a échoué 5 fois de suite/);
    } finally {
      db.close();
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('CSRF Origin: the configured public tunnel origin passes, while an unlisted origin stays blocked', async () => {
  const previous = process.env.WHATIF_ALLOWED_ORIGINS;
  process.env.WHATIF_ALLOWED_ORIGINS = 'https://chart-llc-intense-pay.trycloudflare.com';
  const { dir, app } = tmpApp();
  await new Promise((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${app.server.address().port}`;
  try {
    const allowed = await fetch(`${base}/api/health`, { headers: { origin: 'https://chart-llc-intense-pay.trycloudflare.com' } });
    assert.equal(allowed.status, 200);
    const denied = await fetch(`${base}/api/health`, { headers: { origin: 'https://evil.example' } });
    assert.equal(denied.status, 403);
    assert.deepEqual(await denied.json(), { error: 'cross-origin request blocked' });
  } finally {
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
    if (previous === undefined) delete process.env.WHATIF_ALLOWED_ORIGINS;
    else process.env.WHATIF_ALLOWED_ORIGINS = previous;
  }
});


test('security hardening (C1-C4): malformed URL 400, CSRF Origin 403, .env block, Range 416/206, C2 path normalization, symlink escape, shell policy, SSR VM sandbox & API key origin guard', async () => {
  const http = await import('node:http');
  const { dir, app } = tmpApp();
  app.db.setSetting('provider', 'mock');
  await new Promise((resolve) => app.server.listen(0, '127.0.0.1', resolve));
  const port = app.server.address().port;
  const base = `http://127.0.0.1:${port}`;
  try {
    // 1. C1: malformed percent-encoding returns 400 instead of crashing the HTTP server
    const badStatus = await new Promise((resolve, reject) => {
      const req = http.request({ host: '127.0.0.1', port, path: '/%E0%A4%A', method: 'GET' }, (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode));
      });
      req.on('error', reject);
      req.end();
    });
    assert.equal(badStatus, 400);
    // Server is still alive right after malformed URL
    assert.equal((await fetch(`${base}/api/health`)).status, 200);

    // 2. Cross-origin CSRF & Host rebinding blocked on /api/*
    const csrfRes = await fetch(`${base}/api/settings`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json', origin: 'https://evil.example' },
      body: JSON.stringify({ model: 'pwned' }),
    });
    assert.equal(csrfRes.status, 403);

    // 3. /api/file blocks .env & handles suffix / out-of-bounds Range headers cleanly
    const created = await (
      await fetch(`${base}/api/projects`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: 'secproj' }),
      })
    ).json();
    const pid = created.project.id;
    const proot = created.project.root;
    fs.writeFileSync(path.join(proot, '.env'), 'SECRET_KEY=topsecret\n');
    fs.mkdirSync(path.join(proot, 'out'), { recursive: true });
    fs.writeFileSync(path.join(proot, 'out', 'sample.bin'), '0123456789');

    assert.equal((await fetch(`${base}/api/file?project=${pid}&path=.env`)).status, 403);
    assert.equal(
      (await fetch(`${base}/api/file?project=${pid}&path=out/sample.bin`, { headers: { range: 'bytes=999-' } })).status,
      416,
    );
    const suffixRes = await fetch(`${base}/api/file?project=${pid}&path=out/sample.bin`, { headers: { range: 'bytes=-4' } });
    assert.equal(suffixRes.status, 206);
    assert.equal(await suffixRes.text(), '6789');

    // 4. C2: write_file normalizes ../ segments before checking WRITABLE_PATHS, read_file blocks .env, safeJoin blocks symlinks
    const registry = defaultRegistry();
    const ctx = { projectId: pid, projectRoot: proot, dataRoot: dir, skillDir: null, log: () => {} };
    const dotGitEscape = await registry.run('write_file', { path: 'src/episodes/../../.git/config', content: 'x' }, ctx);
    assert.equal(dotGitEscape.ok, false);
    const engineEscape = await registry.run('write_file', { path: 'src/episodes/../engine/hack.ts', content: 'x' }, ctx);
    assert.equal(engineEscape.ok, false);
    const envRead = await registry.run('read_file', { path: '.env' }, ctx);
    assert.equal(envRead.ok, false);

    const outsideFile = path.join(os.tmpdir(), `whatif-outside-${Date.now()}.txt`);
    fs.writeFileSync(outsideFile, 'outside-secret');
    try {
      fs.mkdirSync(path.join(proot, 'src', 'episodes', 'sym'), { recursive: true });
      fs.symlinkSync(outsideFile, path.join(proot, 'src', 'episodes', 'sym', 'escape.ts'));
      const symRead = await registry.run('read_file', { path: 'src/episodes/sym/escape.ts' }, ctx);
      assert.equal(symRead.ok, false);
    } finally {
      fs.rmSync(outsideFile, { force: true });
    }

    // 5. Shell policy blocks npx prefix spoofing, ffmpeg file:/network protocols, python out/ scripts, and git -c
    const { checkCommand } = await import('../dist/tools/shell-policy.js');
    assert.equal(checkCommand('npx tsc-evil').ok, false);
    assert.equal(checkCommand('npx tsc --noEmit').ok, true);
    assert.equal(checkCommand('ffmpeg -i file:/etc/passwd out/x.mp4').ok, false);
    assert.equal(checkCommand('python3 out/evil.py').ok, false);
    assert.equal(checkCommand('git -c core.sshCommand=evil status').ok, false);

    // 6. C3: shotsAudit runs SSR inside an isolated VM child process and blocks require('node:child_process') & process globals
    const nm = path.join(proot, 'node_modules');
    for (const [name, body] of [
      ['typescript', `module.exports = require(${JSON.stringify(path.resolve('node_modules/typescript'))});`],
      ['react', 'module.exports = { Fragment: "fragment", createElement: (type, props, ...children) => ({ type, props: { ...(props || {}), children } }) };'],
      ['react-dom', 'module.exports = {};'],
      ['react-dom/server', 'const render = (el) => { if (!el || typeof el !== "object") return ""; if (typeof el.type === "function") return render(el.type(el.props || {})); const ch = el.props && el.props.children; return Array.isArray(ch) ? ch.map(render).join("") : render(ch); }; module.exports = { renderToStaticMarkup: render };'],
      ['react/jsx-runtime', 'module.exports = { Fragment: "fragment", jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) };'],
    ]) {
      fs.mkdirSync(path.join(nm, name), { recursive: true });
      fs.writeFileSync(path.join(nm, name, 'index.js'), body);
    }
    fs.writeFileSync(path.join(proot, 'package.json'), '{}');
    const epDir = path.join(proot, 'src', 'episodes', 'sandboxed');
    fs.mkdirSync(path.join(epDir, 'shots'), { recursive: true });
    fs.mkdirSync(path.join(proot, 'src', 'engine'), { recursive: true });
    fs.writeFileSync(path.join(proot, 'src', 'engine', 'shot.tsx'), `export const shot = () => () => null;\n`);
    fs.writeFileSync(
      path.join(proot, 'src', 'engine', 'index.ts'),
      `import React from 'react';\nexport const Stage = ({ children }) => React.createElement('g', null, children);\nexport { shot } from './shot';\n`,
    );
    const tlObj = {
      schema: 2,
      fps: 30,
      total: 180,
      shots: [
        { id: 'S1', len: 60, size: 'establishing' },
        { id: 'S2', len: 60, size: 'establishing' },
        { id: 'S3', len: 60, size: 'establishing' },
      ],
      segments: [
        { shot: 'S1', a: 0, b: 60 },
        { shot: 'S2', a: 60, b: 120 },
        { shot: 'S3', a: 120, b: 180 },
      ],
    };
    const epObj = {
      slug: 'sandboxed',
      shots: [
        { id: 'S1', size: 'establishing' },
        { id: 'S2', size: 'establishing' },
        { id: 'S3', size: 'establishing' },
      ],
    };
    fs.writeFileSync(path.join(epDir, 'timeline.json'), JSON.stringify(tlObj));
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify(epObj));
    const { shotsAudit } = await import('../dist/tools/whatif.js');
    const missingBible = shotsAudit(proot, 'sandboxed', epObj, tlObj) ?? '';
    assert.match(missingBible, /bible\.json is missing/);

    // 6b. In diffusion mode, TSX files cannot be written in shots/ via write_file (strictly S<N>.json specs)
    const { fsTools } = await import('../dist/tools/index.js');
    const secReg = new ToolRegistry(fsTools);
    const tsxAttempt = await secReg.run(
      'write_file',
      { path: 'src/episodes/sandboxed/shots/S1.tsx', content: 'export const S1 = () => null;' },
      { projectId: pid, projectRoot: proot, dataRoot: dir, log: () => {} },
    );
    assert.equal(tsxAttempt.ok, false, 'write_file refuses TSX in shots/ (S<N>.json only)');
    assert.match(tsxAttempt.summary, /must be a JSON prompt spec named "S<N>\.json"/);

    // 7. C4: isAllowedCodexBase & discoverProviderModels origin check prevent credential exfiltration
    const { isAllowedCodexBase, discoverProviderModels } = await import('../dist/orchestrator/providers/index.js');
    assert.equal(isAllowedCodexBase('https://chatgpt.com/backend-api/codex'), true);
    assert.equal(isAllowedCodexBase('http://127.0.0.1:8123/codex'), true);
    assert.equal(isAllowedCodexBase('https://evil.example/?chatgpt.com'), false);
    assert.equal(isAllowedCodexBase('https://chatgpt.com.evil.example/v1'), false);

    let leakedAuthHeader = null;
    const spySrv = http.createServer((req, res) => {
      leakedAuthHeader = req.headers.authorization ?? null;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'spy-model' }] }));
    });
    await new Promise((r) => spySrv.listen(0, '127.0.0.1', r));
    const spyPort = spySrv.address().port;
    try {
      app.db.setSetting('provider', 'openai');
      app.db.setSetting('apiKey', 'sk-super-secret-stored-key');
      app.db.setSetting('baseURL', 'https://api.openai.com/v1');
      await discoverProviderModels(app.db, {
        provider: 'openai',
        baseURL: `http://127.0.0.1:${spyPort}/v1`,
        forceRefresh: true,
        env: {},
      });
      assert.equal(leakedAuthHeader, null, 'stored DB apiKey must NOT be sent to an untrusted overrideBaseURL');
    } finally {
      await new Promise((r) => spySrv.close(r));
    }

    // 8. Context buildMessages repairs orphan tool_calls, and Bus coalesces assistant_delta without evicting tool_call
    const { buildMessages } = await import('../dist/orchestrator/context.js');
    const { Bus } = await import('../dist/orchestrator/bus.js');
    const th = app.db.createThread(pid, 'orphan');
    app.db.addMessage(th.id, 'user', 'bonjour');
    app.db.addMessage(th.id, 'assistant', '', { toolCalls: [{ id: 'tc_orphan', name: 'read_file', args: { path: 'PROJECT.md' } }] });
    app.db.addMessage(th.id, 'user', 'relance');
    const built = buildMessages({ projectRoot: proot, history: app.db.listMessages(th.id), maxChars: 120_000 });
    assert.ok(
      built.some((m) => m.role === 'tool' && m.toolCallId === 'tc_orphan'),
      'orphan assistant.toolCalls receives a synthetic error tool message so OpenAI/Anthropic never 400',
    );

    const bus = new Bus();
    bus.emit(th.id, 'tool_call', { id: 'tc1', name: 'read_file' });
    for (let i = 0; i < 600; i++) bus.emit(th.id, 'assistant_delta', { text: 'a' });
    const replayed = bus.replay(th.id, 0);
    assert.equal(replayed[0].type, 'tool_call', 'coalesced assistant_delta stream never evicts earlier tool_call events');
    assert.equal(replayed[1].data.text.length, 600);
    assert.equal(bus.replay(th.id, replayed[0].seq).length, 1, 'replay(afterSeq) returns only newer events');
  } finally {
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('providers.json multi-provider & multi-model catalog: JSONC parsing, $ENV keys, hot-reload, concurrent OpenAI/Anthropic routing & secret isolation', async () => {
  const http = await import('node:http');
  const {
    clearModelDiscoveryCache,
    clearProvidersCatalogCache,
    providerConfig,
    resolveCatalogFilePaths,
    resolveProvider,
    resolveReviewProvider,
    reviewProviderConfig,
  } = await import('../dist/orchestrator/providers/index.js');
  const { checkCommand } = await import('../dist/tools/shell-policy.js');
  const { walk } = await import('../dist/tools/paths.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-catalog-'));
  // Empty WHATIF_PROVIDERS_FILE (as passed by docker-compose `${WHATIF_PROVIDERS_FILE:-}`) must still resolve default paths
  assert.deepEqual(resolveCatalogFilePaths({ WHATIF_PROVIDERS_FILE: '' }, dir), [path.join(dir, 'providers.json')]);
  assert.deepEqual(resolveCatalogFilePaths({ WHATIF_PROVIDERS_FILE: 'none' }, dir), []);
  const seenRequests = [];

  const mockGateway = http.createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    seenRequests.push({
      url: req.url,
      auth: req.headers.authorization || '',
      xApiKey: req.headers['x-api-key'] || '',
      body: body ? JSON.parse(body) : null,
    });

    if (req.url === '/proxy/v1/models') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'live-discovered-vl' }, { id: 'text-embedding-3-large' }] }));
      return;
    }
    if (req.url === '/or/v1/chat/completions') {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.end('data: {"choices":[{"delta":{"content":"Réponse OpenRouter OK"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n');
      return;
    }
    if (req.url === '/claude/v1/messages') {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.end(
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Réponse Claude OK"}}\n\n' +
          'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n',
      );
      return;
    }
    res.writeHead(404).end();
  });

  await new Promise((r) => mockGateway.listen(0, '127.0.0.1', r));
  const gwPort = mockGateway.address().port;
  const prevSecret = process.env.WHATIF_TEST_CAT_SECRET;
  process.env.WHATIF_TEST_CAT_SECRET = 'sk-from-env-interpolation-999';

  const catalogPath = path.join(dir, 'providers.json');
  fs.writeFileSync(
    catalogPath,
    `{
      // JSONC comments + trailing commas must work cleanly with URLs containing "https://"
      "defaultProvider": "openrouter-pro",
      "defaultReviewProvider": "proxy-live",
      "providers": {
        "openrouter-pro": {
          "label": "OpenRouter Pro",
          "driver": "openai",
          "baseURL": "http://127.0.0.1:${gwPort}/or/v1",
          "apiKey": "$WHATIF_TEST_CAT_SECRET",
          "defaultModel": "anthropic/claude-sonnet-4.5",
          "reviewModel": "google/gemini-2.5-pro",
          "models": [
            "anthropic/claude-sonnet-4.5",
            { "id": "google/gemini-2.5-pro", "label": "Gemini 2.5 Pro Vision", "supportsVision": true },
            "deepseek/deepseek-chat-v3",
          ],
        },
        "claude-perso": {
          "label": "Claude Direct Perso",
          "driver": "anthropic",
          "baseURL": "http://127.0.0.1:${gwPort}/claude",
          "apiKey": "sk-ant-literal-secret-777",
          "defaultModel": "claude-sonnet-4-5",
          "models": ["claude-sonnet-4-5", "claude-opus-4-5"],
        },
        "proxy-live": {
          "label": "Proxy Live Discovery",
          "driver": "openai",
          "baseURL": "http://127.0.0.1:${gwPort}/proxy/v1",
          "apiKey": "sk-proxy-live-555",
          "defaultModel": "pinned-model-1",
          "discoverModels": true,
          "models": ["pinned-model-1"],
        },
        "disabled-prov": {
          "enabled": false,
          "baseURL": "https://disabled.example.com/v1",
          "apiKey": "sk-disabled"
        }
      }
    }\n`,
  );

  clearProvidersCatalogCache();
  clearModelDiscoveryCache();
  const app = createApp({
    ...loadConfig(),
    dataRoot: dir,
    dbPath: path.join(dir, 'app.db'),
    projectsDir: path.join(dir, 'projects'),
    skillDir: null,
  });
  await new Promise((r) => app.server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${app.server.address().port}`;

  try {
    // 1. /api/settings exposes catalog providers & defaults without ever leaking secret apiKey strings
    const rawSettingsText = await (await fetch(`${base}/api/settings`)).text();
    assert.ok(!rawSettingsText.includes('sk-from-env-interpolation-999'), 'never leak env-interpolated apiKey');
    assert.ok(!rawSettingsText.includes('sk-ant-literal-secret-777'), 'never leak literal apiKey');
    const settings = JSON.parse(rawSettingsText);
    assert.equal(settings.provider, 'openrouter-pro');
    assert.equal(settings.model, 'anthropic/claude-sonnet-4.5');
    assert.equal(settings.hasKey, true);
    assert.equal(settings.reviewer?.provider, 'proxy-live');
    assert.deepEqual(settings.providersCatalog.providerIds, ['openrouter-pro', 'claude-perso', 'proxy-live']);
    assert.equal('disabled-prov' in settings.presets, false, 'enabled:false provider is omitted');

    // 2. Static catalog models return immediately (`source: "config_file"`), while `discoverModels: true` merges pinned + live
    const orModels = await (await fetch(`${base}/api/models?provider=openrouter-pro`)).json();
    assert.equal(orModels.source, 'config_file');
    assert.deepEqual(
      orModels.models.map((m) => m.id),
      ['anthropic/claude-sonnet-4.5', 'google/gemini-2.5-pro', 'deepseek/deepseek-chat-v3'],
    );
    assert.equal(orModels.defaultAgentModel, 'anthropic/claude-sonnet-4.5');
    assert.equal(orModels.defaultReviewerModel, 'google/gemini-2.5-pro');

    const liveModels = await (await fetch(`${base}/api/models?provider=proxy-live`)).json();
    assert.equal(liveModels.source, 'live');
    assert.deepEqual(
      liveModels.models.map((m) => m.id),
      ['pinned-model-1', 'live-discovered-vl'],
      'merges pinned providers.json models with live /models and filters out embeddings',
    );

    // 3. Hot-reload on file edit without restarting server
    await new Promise((r) => setTimeout(r, 15));
    const updatedRaw = fs
      .readFileSync(catalogPath, 'utf8')
      .replace('"deepseek/deepseek-chat-v3",', '"deepseek/deepseek-chat-v3", "qwen/qwen3-max-hot",');
    fs.writeFileSync(catalogPath, updatedRaw);

    const orModelsAfterEdit = await (await fetch(`${base}/api/models?provider=openrouter-pro`)).json();
    assert.ok(
      orModelsAfterEdit.models.some((m) => m.id === 'qwen/qwen3-max-hot'),
      'editing providers.json hot-reloads new models immediately without restart',
    );

    // 4. Concurrent projects use distinct catalog providers & drivers simultaneously without touching .env
    const p1 = app.db.createProject('proj-or', 'Proj OR', path.join(dir, 'projects', 'proj-or'), {
      provider: 'openrouter-pro',
      model: 'qwen/qwen3-max-hot',
      review_provider: 'claude-perso',
      review_model: 'claude-opus-4-5',
    });
    const p2 = app.db.createProject('proj-cl', 'Proj Claude', path.join(dir, 'projects', 'proj-cl'), {
      provider: 'claude-perso',
      model: 'claude-sonnet-4-5',
    });

    assert.equal(providerConfig(app.db, p1).apiKey, 'sk-from-env-interpolation-999');
    assert.equal(reviewProviderConfig(app.db, p1)?.apiKey, 'sk-ant-literal-secret-777');
    assert.equal(reviewProviderConfig(app.db, p1)?.model, 'claude-opus-4-5');

    const prov1 = resolveProvider(app.db, p1);
    const prov2 = resolveProvider(app.db, p2);
    const rev1 = resolveReviewProvider(app.db, p1);
    assert.equal(prov1.id, 'openrouter-pro');
    assert.equal(prov2.id, 'claude-perso');
    assert.equal(rev1?.id, 'claude-perso');

    const res1 = await prov1.chat({ messages: [{ role: 'user', content: 'ping 1' }], tools: [] });
    const res2 = await prov2.chat({ messages: [{ role: 'user', content: 'ping 2' }], tools: [] });
    assert.equal(res1.text, 'Réponse OpenRouter OK');
    assert.equal(res2.text, 'Réponse Claude OK');

    const orReq = seenRequests.find((r) => r.url === '/or/v1/chat/completions');
    const clReq = seenRequests.find((r) => r.url === '/claude/v1/messages');
    assert.equal(orReq?.auth, 'Bearer sk-from-env-interpolation-999');
    assert.equal(orReq?.body?.model, 'qwen/qwen3-max-hot');
    assert.equal(clReq?.xApiKey, 'sk-ant-literal-secret-777');
    assert.equal(clReq?.body?.model, 'claude-sonnet-4-5');

    // 5. Security: providers.json is blocked from /api/file, read_file, write_file, walk(), and shell
    fs.mkdirSync(p1.root, { recursive: true });
    fs.writeFileSync(path.join(p1.root, 'providers.json'), '{"providers":{"leak":{"apiKey":"sk-leak"}}}');
    fs.writeFileSync(path.join(p1.root, 'PROJECT.md'), '# ok\n');

    assert.equal((await fetch(`${base}/api/file?project=${p1.id}&path=providers.json`)).status, 403);
    assert.equal(walk(p1.root).includes('providers.json'), false, 'walk() hides providers.json');

    const reg = defaultRegistry();
    const tctx = { projectId: p1.id, projectRoot: p1.root, dataRoot: dir, skillDir: null, log: () => {} };
    assert.equal((await reg.run('read_file', { path: 'providers.json' }, tctx)).ok, false);
    assert.equal((await reg.run('write_file', { path: 'providers.json', content: '{}' }, tctx)).ok, false);
    assert.equal(checkCommand('cat providers.json').ok, false);
  } finally {
    if (prevSecret === undefined) delete process.env.WHATIF_TEST_CAT_SECRET;
    else process.env.WHATIF_TEST_CAT_SECRET = prevSecret;
    await app.close();
    await new Promise((r) => mockGateway.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('mid-turn compaction preserves recent assistant+tool steps and prepends user anchor instead of wiping kept history', async () => {
  const { buildMessages } = await import('../dist/orchestrator/context.js');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-midturn-'));
  try {
    const history = [
      {
        id: 1,
        thread_id: 't1',
        role: 'user',
        content: 'Nouvel épisode « custom-100-pintes » (format 16:9). Fais tout du début à la fin.',
        payload: null,
        created_at: '',
      },
    ];
    let nextId = 2;
    for (let step = 1; step <= 18; step++) {
      const tcId = `call_${step}`;
      history.push({
        id: nextId++,
        thread_id: 't1',
        role: 'assistant',
        content: `Step ${step}`,
        payload: JSON.stringify({ toolCalls: [{ id: tcId, name: 'read_file', args: { path: `file_${step}.tsx` } }] }),
        created_at: '',
      });
      history.push({
        id: nextId++,
        thread_id: 't1',
        role: 'tool',
        content: `ok\n${'x'.repeat(2500)}\nstep_${step}_done`,
        payload: JSON.stringify({ toolCallId: tcId, name: 'read_file', ok: true }),
        created_at: '',
      });
    }

    // Force tight maxChars (<19k test budget) so older steps + initial user message get compacted mid-turn
    const msgs = buildMessages({ projectRoot: dir, history, maxChars: 16_000 });
    const nonSystem = msgs.filter((m) => m.role !== 'system');
    assert.ok(nonSystem.length >= 5, `expected preserved assistant+tool history, got ${nonSystem.length} messages`);
    assert.equal(nonSystem[0].role, 'user', 'first non-system message is the re-anchored user prompt');
    assert.match(nonSystem[0].content, /custom-100-pintes/, 'original kickoff user goal is preserved in anchor');
    assert.ok(
      nonSystem.some((m) => m.role === 'assistant' && m.content === 'Step 18'),
      'most recent assistant step is preserved (zero amnesia)',
    );
    assert.ok(
      nonSystem.some((m) => m.role === 'tool' && m.content.includes('step_18_done')),
      'most recent tool result is preserved',
    );
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('bilingual FR/EN support: English proposals never block, detect_narrator_voice picks Léa/Hugo in EN, Deepgram uses lang=en, and shotlist/reviewer adapt', async () => {
  const { execFile, execFileSync } = await import('node:child_process');
  const { promisify } = await import('node:util');
  const execFileAsync = promisify(execFile);
  const http = await import('node:http');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-bilingual-en-'));
  let srv = null;
  try {
    // 1. Python detect_text_language & detect_narrator_voice on English + French stories
    const pyCheck = execFileSync(
      'python3',
      [
        '-c',
        `
import json, sys
sys.path.insert(0, 'pipeline')
from nouveau_client import detect_narrator_voice, detect_text_language

the_promise = """[Low, steady voice]
They stole my best friend's phone and read her messages to her dead mother in front of the entire school.
I wanted revenge. And to get it, I decided to become friends with the girl who had destroyed her.
I found her locked in a bathroom stall. She told me she wanted to disappear.
Lena stopped folding. "Don't do anything, Harper."
I told her my plan."""

en_wife_selfid = "I'm a 28-year-old woman and last night my husband left his phone unlocked on the kitchen counter while I was doing my makeup."
en_male = "I'm a 31-year-old guy and my wife asked me to pick up her sister at the airport while I was trimming my beard."
en_third_fem = "She walked out of the hospital alone in the rain. Everyone thought she had forgotten what happened, but she kept the key in her coat pocket and she waited until midnight."
fr_fem = "Je me suis assise dans le salon et mon mari m'a regardée en silence."

print(json.dumps({
  "lang_en": detect_text_language(the_promise),
  "lang_fr": detect_text_language(fr_fem),
  "promise": detect_narrator_voice(the_promise, "The Promise"),
  "wife_selfid": detect_narrator_voice(en_wife_selfid, "The Phone"),
  "en_male": detect_narrator_voice(en_male, "Airport"),
  "en_third_fem": detect_narrator_voice(en_third_fem, "Midnight"),
  "fr_fem": detect_narrator_voice(fr_fem, "Le Salon"),
}))
`,
      ],
      { encoding: 'utf8' },
    );
    const det = JSON.parse(pyCheck);
    assert.equal(det.lang_en, 'en');
    assert.equal(det.lang_fr, 'fr');
    assert.equal(det.promise.lang, 'en');
    assert.equal(det.promise.pov, 'first');
    assert.equal(det.promise.gender, 'female');
    assert.equal(det.promise.voiceName, 'Léa', `expected Léa on "The Promise", got ${JSON.stringify(det.promise)}`);
    assert.equal(det.wife_selfid.voiceName, 'Léa');
    assert.equal(det.en_male.lang, 'en');
    assert.equal(det.en_male.gender, 'male');
    assert.equal(det.en_male.voiceName, 'Hugo');
    assert.equal(det.en_third_fem.pov, 'third');
    assert.equal(det.en_third_fem.voiceName, 'Léa');
    assert.equal(det.fr_fem.lang, 'fr');
    assert.equal(det.fr_fem.voiceName, 'Léa');

    // 2. End-to-end source_import.py -> voice_fish.py -> shotlist.py on an English :8005 proposal
    const fakeMp3 = path.join(dir, 'en-sample.mp3');
    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=330:duration=9.0', '-ar', '44100', '-ac', '1', fakeMp3],
      { stdio: 'pipe' },
    );
    const mp3Bytes = fs.readFileSync(fakeMp3);

    const enProposal =
      'They stole my best friend phone in the bathroom stall. ' +
      'I found her locked in a bathroom stall crying over her messages. ' +
      'Then Sloane walked onto the stage in the gym with a microphone and shouted at everyone. ' +
      'Lena whispered, "Don\'t do anything, Harper." That evening I packed boxes in my bedroom.';

    let transcribeLangSeen = '';
    let generatedVoiceId = '';

    srv = http.createServer((req, res) => {
      const u = req.url || '';
      if (req.method === 'GET' && u === '/api/videos/vid-en-promise') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ id: 'vid-en-promise', title: 'The Promise', duration: 9.0 }));
        return;
      }
      if (req.method === 'GET' && u.startsWith('/api/videos/vid-en-promise/transcript/check')) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ exists: true, file_exists: true }));
        return;
      }
      if (req.method === 'GET' && u.startsWith('/api/videos/vid-en-promise/translations')) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            proposals: [
              {
                id: 1363,
                is_selected: true,
                prompt_type: 'manual',
                model_name: 'manual',
                title: 'The Promise',
                translation: enProposal,
              },
            ],
          }),
        );
        return;
      }
      if (req.method === 'POST' && u.includes('/translations/1363/select')) {
        req.resume();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      if (req.method === 'GET' && u === '/api/videos/vid-en-promise/tts/check') {
        res.writeHead(200, { 'content-type': 'application/json' });
        // Simulate stale wrong-gender Hugo audio on :8005 so voice_fish.py regenerates with Léa
        res.end(JSON.stringify({ exists: true, file_exists: true, id: 55, path: '/downloads/tts/hugo.mp3', voice: 'elevenlabs:3Tp5mYhT9wrfnry9pzvV', duration_seconds: 9.0 }));
        return;
      }
      if (req.method === 'POST' && u === '/api/videos/vid-en-promise/tts/elevenlabs') {
        let body = '';
        req.on('data', (c) => (body += c));
        req.on('end', () => {
          const parsed = JSON.parse(body || '{}');
          generatedVoiceId = parsed.voiceId || '';
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ id: 56, audioPath: '/downloads/tts/lea-en.mp3' }));
        });
        return;
      }
      if (req.method === 'GET' && u === '/downloads/tts/lea-en.mp3') {
        res.writeHead(200, { 'content-type': 'audio/mpeg' });
        res.end(mp3Bytes);
        return;
      }
      if (req.method === 'POST' && u.startsWith('/api/videos/vid-en-promise/tts/transcribe')) {
        const q = new URL(u, 'http://127.0.0.1').searchParams;
        transcribeLangSeen = q.get('lang') || '';
        req.resume();
        req.on('end', () => {
          const toks = enProposal.split(/\s+/).filter(Boolean);
          const step = 8.4 / toks.length;
          const words = toks.map((w, idx) => ({
            word: w,
            start: Number((0.1 + idx * step).toFixed(3)),
            end: Number((0.1 + (idx + 0.85) * step).toFixed(3)),
          }));
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ transcript: [{ start: 0.1, end: 8.5, words }] }));
        });
        return;
      }
      res.writeHead(404);
      res.end('{}');
    });
    await new Promise((r) => srv.listen(0, '127.0.0.1', r));
    const port = srv.address().port;

    const proot = path.join(dir, 'studio');
    const slug = 'the-promise';
    fs.mkdirSync(path.join(proot, 'src', 'episodes', slug), { recursive: true });
    fs.mkdirSync(path.join(proot, 'public', slug), { recursive: true });
    fs.mkdirSync(path.join(proot, 'out', slug), { recursive: true });
    fs.writeFileSync(path.join(proot, 'package.json'), JSON.stringify({ name: 'test-studio' }));
    fs.writeFileSync(
      path.join(proot, 'src', 'episodes', slug, 'episode.json'),
      JSON.stringify({
        slug,
        aspect: '9:16',
        title: ['ET SI TU POUVAIS', '… ?'],
        source: { provider: 'nouveau', videoId: 'vid-en-promise' },
        script: [],
        shots: [],
        edit: [],
      }),
    );

    const env = { ...process.env, WHATIF_NOUVEAU_URL: `http://127.0.0.1:${port}` };
    await execFileAsync('python3', [path.resolve('pipeline/source_import.py'), slug, '--video', 'vid-en-promise', '--pick', 'hook', '--apply'], {
      cwd: proot,
      env,
    });

    const epAfterSource = JSON.parse(fs.readFileSync(path.join(proot, 'src', 'episodes', slug, 'episode.json'), 'utf8'));
    assert.equal(epAfterSource.lang, 'en');
    assert.equal(epAfterSource.voice.voiceName, 'Léa');
    assert.equal(epAfterSource.voice.narratorGender, 'female');
    assert.deepEqual(epAfterSource.title, ['THE', 'PROMISE']);

    await execFileAsync('python3', [path.resolve('pipeline/voice_fish.py'), slug], { cwd: proot, env });
    assert.equal(generatedVoiceId, 'KSyQzmsYhFbuOhqj1Xxv', 'regenerated English TTS with Léa instead of keeping wrong-gender Hugo');
    assert.equal(transcribeLangSeen, 'en', 'passed lang=en (not hardcoded fr) to Deepgram tts_transcribe');

    await execFileAsync('python3', [path.resolve('pipeline/shotlist.py'), slug, '--project', proot], { cwd: proot, env });
    const epAfterShotlist = JSON.parse(fs.readFileSync(path.join(proot, 'src', 'episodes', slug, 'episode.json'), 'utf8'));
    const setsUsed = new Set(epAfterShotlist.shots.map((s) => s.set));
    assert.ok(setsUsed.has('corridor') || setsUsed.has('auditorium'), `expected English school/bathroom/gym sets instead of pure study fallback, got ${[...setsUsed].join(', ')}`);
    assert.ok(epAfterShotlist.shots.some((s) => String(s.size).includes('insert')), 'expected semantic insert on English phone/messages/microphone');

    // 3. TypeScript episodeAudit, detectEpisodeLanguage, prompts & Reviewer language adaptation
    const { detectEpisodeLanguage, episodeAudit } = await import('../dist/tools/whatif.js');
    const { buildMessages, SYSTEM_PROMPT } = await import('../dist/orchestrator/context.js');
    const { DIRECTOR_GUIDE } = await import('../dist/orchestrator/director-guide.js');
    const { reviewerSystemPrompt, deltaReviewerSystemPrompt } = await import('../dist/orchestrator/reviewer.js');

    assert.equal(detectEpisodeLanguage(epAfterShotlist), 'en');
    assert.equal(episodeAudit(epAfterShotlist, slug), null, 'English episode passes episodeAudit cleanly');
    assert.match(
      episodeAudit({ ...epAfterShotlist, title: ['ET SI TU POUVAIS', '… ?'] }, slug) ?? '',
      /French placeholder.*English episode/,
      'catches leftover French stub title on an English episode',
    );

    const pinMsgs = buildMessages({
      projectRoot: proot,
      history: [{ id: 1, thread_id: 't1', role: 'user', content: `Continue ${slug}`, payload: null, created_at: '' }],
    });
    const sysJoined = pinMsgs.filter((m) => m.role === 'system').map((m) => m.content).join('\n');
    assert.match(sysJoined, /Language: English \(en\) — continue the entire pipeline in English/);
    assert.match(SYSTEM_PROMPT, /NEVER block, stop, translate, rewrite, or shorten/);
    assert.match(DIRECTOR_GUIDE, /Language lock/);
    assert.match(reviewerSystemPrompt('en'), /ANGLAIS uniquement/);
    assert.match(reviewerSystemPrompt('fr'), /FRANÇAIS uniquement/);
    assert.match(deltaReviewerSystemPrompt('en'), /langue `en`/);
  } finally {
    if (srv) await new Promise((r) => srv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('audit of 2026-11-16: repo_map lists episodes whatever the cap & compaction leaves no fake code', async () => {
  const { defaultRegistry } = await import('../dist/tools/index.js');
  const { buildMessages } = await import('../dist/orchestrator/context.js');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-a1116-'));
  try {
    // --- repo_map: the cap is on what is listed, not on what is known
    const ep = path.join(root, 'src', 'episodes', 'my-ep');
    fs.mkdirSync(path.join(ep, 'shots'), { recursive: true });
    fs.writeFileSync(path.join(root, 'PROJECT.md'), '# p');
    fs.writeFileSync(path.join(ep, 'episode.json'), '{}');
    for (let i = 1; i <= 12; i++) fs.writeFileSync(path.join(ep, 'shots', `S${i}.json`), JSON.stringify({ id: `S${i}` }));
    for (let i = 0; i < 8; i++) fs.writeFileSync(path.join(root, `a${i}.md`), 'x');
    const repoMap = defaultRegistry().get('repo_map');
    const ctx = { projectRoot: root, log: () => {} };
    const r1 = await repoMap.run({ maxFiles: 3, filter: 'PROJECT.md' }, ctx);
    assert.match(r1.summary, /^1 files\. Episodes: my-ep\./);
    const r2 = await repoMap.run({ maxFiles: 1, filter: 'src/episodes/my-ep' }, ctx);
    assert.match(r2.summary, /^1 files\. Episodes: my-ep\./, 'the filter is applied before the cap');
    const r3 = await repoMap.run({ maxFiles: 5 }, ctx);
    assert.equal(r3.data.files.length, 5);
    assert.deepEqual(r3.data.episodes, ['my-ep']);

    // --- compaction: an old big call leaves one plain line without any code
    const bigSpec = {
      episode: 'my-ep',
      id: 'S3',
      place: 'ROOM',
      characters: ['HERO'],
      imagePrompt: 'Detailed 2D scene '.repeat(90),
      videoPrompt: 'Hero steps forward toward window slowly, strictly no dialogue.',
      sfxHint: 'footsteps, no music',
    };
    const history = [{ id: 1, thread_id: 't', role: 'user', content: 'go', payload: null, created_at: '' }];
    let id = 2;
    const call = (name, args) => { const cid = `c${id}`; history.push({ id: id++, thread_id: 't', role: 'assistant', content: '', payload: JSON.stringify({ toolCalls: [{ id: cid, name, args }] }), created_at: '' }); history.push({ id: id++, thread_id: 't', role: 'tool', content: 'ok', payload: JSON.stringify({ toolCallId: cid, name, ok: true }), created_at: '' }); };
    call('edit_shot', bigSpec);
    for (let i = 0; i < 8; i++) call('read_file', { path: 'PROJECT.md' });
    const old = buildMessages({ projectRoot: root, history, maxChars: 90_000 }).find((m) => m.role === 'assistant' && m.toolCalls?.[0]?.name === 'edit_shot');
    const shown = String(old.toolCalls[0].args.summary ?? '');
    assert.match(shown, /^\[compacted: shot spec \(\d+ chars\) was written to disk by edit_shot\./);
    assert.doesNotMatch(shown, /Detailed 2D scene/, 'nothing redundant left after compaction');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('master: a final that lands under the target is brought up to it under a limiter, the peak stays safe', async () => {
  const { execFileSync } = await import('node:child_process');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-level-'));
  try {
    const slug = 'lvl';
    const outDir = path.join(dir, 'out', slug);
    fs.mkdirSync(outDir, { recursive: true });
    fs.mkdirSync(path.join(dir, 'src', 'episodes', slug), { recursive: true });
    fs.writeFileSync(path.join(dir, 'src', 'episodes', slug, 'episode.json'), JSON.stringify({ id: slug }));
    // A quiet, speech-like bed with a low thump every four seconds: the two-pass normalisation lands it at -14.7 LUFS with
    // the peak already at the ceiling, which is what the real episodes did (-14.5 to -15.0 LUFS, -1.1 dBTP).
    const bed = '0.05*(0.5+0.5*sin(2*PI*3*t))*gt(sin(2*PI*0.4*t),-0.5)*(sin(2*PI*160*t)+0.6*sin(2*PI*480*t)+0.4*sin(2*PI*1400*t)+0.3*sin(2*PI*2600*t))';
    const thump = 'if(lt(mod(t,4.1),0.3),0.5*sin(2*PI*90*t)*exp(-9*mod(t,4.1)),0)';
    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=navy:s=320x180:r=30:d=12', '-f', 'lavfi', '-i', `aevalsrc='${bed}+${thump}':s=48000:d=12`, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', path.join(outDir, `${slug}-raw.mp4`)],
      { stdio: 'pipe' },
    );
    const out = execFileSync('python3', [path.resolve('pipeline/master_run.py'), slug, '--skip-render'], { cwd: dir, encoding: 'utf8' });
    assert.match(out, /level catch-up: \+/, 'the missing gain is applied');
    const rep = JSON.parse(fs.readFileSync(path.join(outDir, 'report.json'), 'utf8'));
    assert.ok(Math.abs(rep.lufs + 14.0) <= 0.3, `expected the final within 0.3 dB of -14 LUFS, got ${rep.lufs}`);
    assert.ok(rep.true_peak_dbtp <= -0.9, `the peak stays under -0.9 dBTP, got ${rep.true_peak_dbtp}`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('whatif-studio MCP server + stdio bridge: exposes Studio tools, translates host paths, enforces Subagent Reviewer gate & Round 2 delta pack', async () => {
  const { spawn } = await import('node:child_process');
  // The test brings its own skill folder: a machine without WHATIF_SKILL_DIR (or a sibling clone) runs it all the same.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-'));
  const skill = path.join(dir, 'skill');
  fs.mkdirSync(path.join(skill, 'studio', 'src', 'episodes'), { recursive: true });
  fs.writeFileSync(path.join(skill, 'studio', 'package.json'), JSON.stringify({ name: 'studio', private: true }));
  fs.writeFileSync(path.join(skill, 'studio', 'src', 'episodes', 'index.ts'), 'export const EPISODES = [];\n');
  const app = createApp({ ...loadConfig(), dataRoot: dir, dbPath: path.join(dir, 'app.db'), projectsDir: path.join(dir, 'projects'), skillDir: skill });
  await new Promise((r) => app.server.listen(0, '127.0.0.1', r));
  const port = app.server.address().port;

  const child = spawn('node', [path.resolve('dist/mcp/stdio.js')], {
    env: {
      ...process.env,
      WHATIF_PORT: String(port),
      WHATIF_STUDIO_HOST: '127.0.0.1',
      WHATIF_HOST_DATA: dir,
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  const pending = new Map();
  let stdoutBuf = '';
  child.stdout.on('data', (chunk) => {
    stdoutBuf += chunk.toString('utf8');
    let nl;
    while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
      const line = stdoutBuf.slice(0, nl).trim();
      stdoutBuf = stdoutBuf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.id !== undefined && pending.has(msg.id)) {
        const resolve = pending.get(msg.id);
        pending.delete(msg.id);
        resolve(msg);
      }
    }
  });

  let nextRpcId = 1;
  const rpc = (method, params = {}) =>
    new Promise((resolve) => {
      const id = nextRpcId++;
      pending.set(id, resolve);
      child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    });

  try {
    const initRes = await rpc('initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'agy-test', version: '1.0' } });
    assert.equal(initRes.result.serverInfo.name, 'whatif-studio');

    const listRes = await rpc('tools/list');
    const toolNames = listRes.result.tools.map((t) => t.name);
    for (const expected of [
      'whatif_create_project',
      'whatif_run',
      'whatif_episode_pin',
      'whatif_shot_words',
      'whatif_write_bible',
      'whatif_stage_shots',
      'whatif_edit_shot',
      'whatif_write_file',
      'whatif_read_file',
      'whatif_review_pack',
      'whatif_save_review',
      'whatif_status',
    ]) {
      assert.ok(toolNames.includes(expected), `missing MCP tool ${expected}`);
    }

    // 1. Create project over MCP & check SQLite + hostProjectRoot path translation
    const createRes = await rpc('tools/call', {
      name: 'whatif_create_project',
      arguments: { episode: 'mcp-ep', aspect: '9:16' },
    });
    assert.equal(createRes.result.isError, false, createRes.result.content[0]?.text);
    assert.match(createRes.result.content[0].text, new RegExp(`hostProjectRoot: ${dir.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/projects/mcp-ep`));
    assert.ok(app.db.getProject('mcp-ep'), 'project registered in Studio SQLite database');

    const projRoot = path.join(dir, 'projects', 'mcp-ep');
    const epDir = path.join(projRoot, 'src', 'episodes', 'mcp-ep');
    const outDir = path.join(projRoot, 'out', 'mcp-ep');
    fs.mkdirSync(path.join(outDir, 'refs'), { recursive: true });
    fs.mkdirSync(path.join(outDir, 'sheets'), { recursive: true });
    fs.mkdirSync(path.join(outDir, 'board'), { recursive: true });
    fs.writeFileSync(path.join(outDir, 'refs', 'char_ELIJAH.png'), 'fake-ref-elijah');
    fs.writeFileSync(path.join(outDir, 'storyboard.jpg'), 'fake-board');
    fs.writeFileSync(path.join(outDir, 'sheets', 'sheet-01.jpg'), 'fake-sheet-1');
    fs.writeFileSync(path.join(outDir, 'board', 'S1.png'), 'fake-s1');

    // Write DIRECTOR.md, bible.json (via whatif_write_bible) & S1.json (via whatif_edit_shot)
    const wrDir = await rpc('tools/call', {
      name: 'whatif_write_file',
      arguments: { episode: 'mcp-ep', path: 'src/episodes/mcp-ep/DIRECTOR.md', content: '# Director Plan\nClassroom exam scene in 2D cel animation with Elijah and his professor.\n'.repeat(4) },
    });
    assert.equal(wrDir.result.isError, false);

    const wbRes = await rpc('tools/call', {
      name: 'whatif_write_bible',
      arguments: {
        episode: 'mcp-ep',
        stylePrompt: '2D hand-crafted cel-shaded animation, crisp ink outlines, dramatic shadows',
        characters: [{ id: 'ELIJAH', name: 'Elijah', appearance: 'Teenager in navy hoodie with short dark curls' }],
        places: [{ id: 'ROOM', name: 'Classroom', description: 'Dim wooden classroom with desks and green chalkboard' }],
      },
    });
    assert.equal(wbRes.result.isError, false, wbRes.result.content[0]?.text);

    const edShot = await rpc('tools/call', {
      name: 'whatif_edit_shot',
      arguments: {
        episode: 'mcp-ep',
        id: 'S1',
        place: 'ROOM',
        characters: ['ELIJAH'],
        size: 'wide',
        imagePrompt: 'Wide 2D shot of Elijah in classroom nervously hiding a thick folded cheat card beside his desk.',
        videoPrompt: 'Elijah glances sideways toward the teacher desk while gripping the paper, slow push camera, strictly no dialogue.',
        sfxHint: 'soft paper crinkle, faint chalk dust silence, no music',
      },
    });
    assert.equal(edShot.result.isError, false, edShot.result.content[0]?.text);

    // 2. BOTH clips & render are mechanically blocked until Independent Reviewer subagent runs!
    const blockedClips = await rpc('tools/call', {
      name: 'whatif_run',
      arguments: { episode: 'mcp-ep', step: 'clips' },
    });
    assert.equal(blockedClips.result.isError, true);
    assert.match(blockedClips.result.content[0].text, /STOP BEFORE CLIPS.*Independent Reviewer subagent/);

    const blockedRender = await rpc('tools/call', {
      name: 'whatif_run',
      arguments: { episode: 'mcp-ep', step: 'render' },
    });
    assert.equal(blockedRender.result.isError, true);
    assert.match(blockedRender.result.content[0].text, /STOP BEFORE RENDER.*Independent Reviewer subagent/);

    // 3. Round 1 Global Review Pack (includes refs/*.png + sheets) -> Save failing review (blocking: true)
    const pack1 = await rpc('tools/call', {
      name: 'whatif_review_pack',
      arguments: { episode: 'mcp-ep' },
    });
    assert.equal(pack1.result.isError, false);
    assert.match(pack1.result.content[0].text, /mode=global/);
    assert.match(pack1.result.content[0].text, /char_ELIJAH\.png/);
    assert.match(pack1.result.content[0].text, /sheet-01\.jpg/);

    const save1 = await rpc('tools/call', {
      name: 'whatif_save_review',
      arguments: {
        episode: 'mcp-ep',
        pass: false,
        summary: 'S1 lacks giant cheat sheet prop.',
        issues: [{ shot: 'S1', severity: 'error', problem: 'Missing cardboard cheat sheet prop', fix: 'Update imagePrompt' }],
        mode: 'global',
      },
    });
    assert.equal(save1.result.isError, false);
    assert.match(save1.result.content[0].text, /CORRECTIONS REQUISES/);

    // 4. Touch S1.json + S1.png after review timestamp -> Round 2 automatically switches to High-Res Delta Mode!
    const futureMs = Date.now() + 5_000;
    fs.utimesSync(path.join(epDir, 'shots', 'S1.json'), futureMs / 1000, futureMs / 1000);
    fs.utimesSync(path.join(outDir, 'board', 'S1.png'), futureMs / 1000, futureMs / 1000);

    const pack2 = await rpc('tools/call', {
      name: 'whatif_review_pack',
      arguments: { episode: 'mcp-ep' },
    });
    assert.equal(pack2.result.isError, false, pack2.result.content[0]?.text);
    assert.match(pack2.result.content[0].text, /mode=delta/);
    assert.match(pack2.result.content[0].text, /board\/S1\.png/);

    const save2 = await rpc('tools/call', {
      name: 'whatif_save_review',
      arguments: { episode: 'mcp-ep', pass: true, summary: 'S1 fixed cleanly.', issues: [], mode: 'delta' },
    });
    assert.equal(save2.result.isError, false);
    assert.match(save2.result.content[0].text, /VALIDÉ ✓/);
    const savedRev = JSON.parse(fs.readFileSync(path.join(outDir, 'review.json'), 'utf8'));
    assert.equal(savedRev.pass, true);
    assert.equal(savedRev.blocking, false);
  } finally {
    child.kill();
    await app.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a reviewer that does not answer: the call has a deadline, is asked once more, then the chain goes on unreviewed', async () => {
  const { chatWithin, ReviewTimeout, reviewEpisodeBoard, formatReviewFeedback, reviewTimeoutMs } = await import('../dist/orchestrator/reviewer.js');
  assert.equal(reviewTimeoutMs(), 240_000, 'four minutes by default');
  let calls = 0;
  let aborted = 0;
  const silent = {
    label: 'silent',
    model: 'never-answers',
    chat({ signal }) {
      calls++;
      return new Promise((_, reject) => signal?.addEventListener('abort', () => { aborted++; reject(new Error('aborted')); }));
    },
  };
  const t0 = Date.now();
  await assert.rejects(() => chatWithin(silent, [{ role: 'user', content: 'x' }], undefined, 40), ReviewTimeout);
  assert.ok(Date.now() - t0 < 1500 && aborted === 1, 'the silent call is aborted at the deadline');
  // a provider that ignores the abort is left behind all the same
  const deaf = { label: 'deaf', model: 'deaf', chat: () => new Promise(() => {}) };
  await assert.rejects(() => chatWithin(deaf, [{ role: 'user', content: 'x' }], undefined, 40), ReviewTimeout);
  // the user's stop is not a timeout
  const ctl = new AbortController();
  setTimeout(() => ctl.abort(), 20);
  await assert.rejects(() => chatWithin(silent, [{ role: 'user', content: 'x' }], ctl.signal, 5000), (e) => !(e instanceof ReviewTimeout));
  // an answer in time goes through
  assert.equal(await chatWithin({ label: 'ok', model: 'ok', chat: async () => ({ text: 'fine', toolCalls: [] }) }, [], undefined, 500), 'fine');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-revtime-'));
  const prev = process.env.WHATIF_REVIEW_TIMEOUT_MS;
  process.env.WHATIF_REVIEW_TIMEOUT_MS = '40';
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'ep');
    fs.mkdirSync(path.join(epDir, 'shots'), { recursive: true });
    fs.mkdirSync(path.join(dir, 'out', 'ep'), { recursive: true });
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ id: 'Ep', slug: 'ep', aspect: '16:9', script: [{ text: 'Bonjour.' }], shots: [{ id: 'S1' }] }));
    fs.writeFileSync(path.join(epDir, 'timeline.json'), JSON.stringify({ fps: 30, total: 60, shots: [{ id: 'S1', len: 60 }], segments: [{ shot: 'S1', a: 0, b: 60 }] }));
    fs.writeFileSync(path.join(epDir, 'shots', 'S1.tsx'), 'export const S1 = () => null;');
    fs.writeFileSync(path.join(epDir, 'shots.tsx'), "export { S1 } from './shots/S1';");
    calls = 0;
    const t1 = Date.now();
    const report = await reviewEpisodeBoard(silent, dir, 'ep');
    assert.equal(report.ok, false);
    assert.equal(report.timedOut, true);
    assert.equal(report.pass, true, 'an absent review does not block the render');
    assert.equal(calls, 2, 'asked twice, not three times');
    assert.ok(Date.now() - t1 < 3000, `the chain waited ${Date.now() - t1} ms`);
    assert.match(formatReviewFeedback(report, 'ep'), /NON VÉRIFIÉ\. Reviewer indisponible : aucune réponse/);
  } finally {
    if (prev === undefined) delete process.env.WHATIF_REVIEW_TIMEOUT_MS; else process.env.WHATIF_REVIEW_TIMEOUT_MS = prev;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

