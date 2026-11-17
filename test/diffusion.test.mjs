import assert from 'node:assert/strict';
import { execFile, execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

const HAS_NUMPY_PIL =
  spawnSync('python3', ['-c', 'import numpy, PIL'], { stdio: 'ignore' }).status === 0;
const NEEDS_PY_MEDIA = HAS_NUMPY_PIL
  ? {}
  : { skip: 'python3 requires numpy + pillow to run diffusion pipeline tests' };

function validBible() {
  return {
    stylePrompt: '2D hand-crafted cel-shaded animation, crisp ink lines, painterly lighting',
    negativePrompt: 'photorealistic, 3D CGI, title card, subtitles, watermark',
    characters: [
      {
        id: 'MAYA',
        name: 'Maya',
        role: 'protagonist',
        appearance: 'Woman late 20s, dark wavy bob, blue denim jacket over cream sweater',
      },
      {
        id: 'LUCAS',
        name: 'Lucas',
        role: 'partner',
        appearance: 'Man late 20s, tousled brown hair, olive wool coat',
      },
    ],
    places: [
      {
        id: 'LIVING',
        name: 'Living room',
        recipe: 'living',
        description: 'Dim urban apartment at night, mustard sofa, walnut table, rain on window',
      },
      {
        id: 'HALL',
        name: 'Hallway',
        recipe: 'corridor',
        description: 'Narrow apartment hallway, coat rack, warm overhead pendant',
      },
    ],
  };
}

function validShot(id, over = {}) {
  return {
    id,
    place: 'LIVING',
    characters: ['MAYA'],
    size: 'medium',
    angle: 'level',
    view: '3/4-left',
    cameraMove: 'push',
    imagePrompt:
      'Medium 3/4 view inside the dimly lit living room. Maya in her blue denim jacket sits on the mustard sofa holding a smartphone.',
    negativePrompt: 'strictly no title card, avoid subtitles, no extra people',
    onScreenText: null,
    videoPrompt: 'Maya slowly raises the phone toward her face as her expression tenses, gentle push in, strictly no dialogue.',
    sfxHint: 'soft fabric rustle, rain tapping glass, quiet room ambience; no music and no voice.',
    ...over,
  };
}

test('diffusion schema validation & negative-clause stripping: bible, edit_shot single/batch, stagingPromptProblems & onScreenText lang', async () => {
  const {
    validateBible,
    validateShotSpec,
    hasPositiveTitleRequest,
    hasPositiveMusicRequest,
    hasPositiveDialogueRequest,
  } = await import('../dist/tools/episode-source.js');
  const { fsTools } = await import('../dist/tools/index.js');
  const { stagingPromptProblems } = await import('../dist/tools/staging.js');
  const { shotsAudit } = await import('../dist/tools/whatif.js');
  const { diffusionMetrics } = await import('../dist/orchestrator/reviewer.js');

  // Negative-clause stripping must pass explicit exclusions and block positive requests
  assert.equal(hasPositiveTitleRequest('2D scene, strictly no title card, zero subtitles, avoid opening title'), false);
  assert.equal(hasPositiveTitleRequest('Add an opening title card reading Chapter 1'), true);
  assert.equal(hasPositiveMusicRequest('Rain tapping glass; strictly no background music and zero piano melody'), false);
  assert.equal(hasPositiveMusicRequest('Add cinematic score and piano melody swelling loudly'), true);
  assert.equal(hasPositiveDialogueRequest('Quiet room tone, no spoken words, strictly no dialogue, avoid voiceover'), false);
  assert.equal(hasPositiveDialogueRequest('Characters speaking aloud with clear voiceover dialogue'), true);

  assert.deepEqual(validateBible(validBible()).errors, []);
  assert.ok(validateBible({ ...validBible(), characters: [{ id: 'bad_id', name: 'x', appearance: 'y' }] }).errors.length > 0);

  const bible = validBible();
  assert.deepEqual(validateShotSpec(validShot('S1'), 'S1').errors, []);
  assert.ok(
    validateShotSpec(validShot('S1', { sfxHint: 'dramatic background music swells' }), 'S1').errors.some((e) =>
      /music/i.test(e),
    ),
  );
  assert.ok(
    validateShotSpec(validShot('S1', { videoPrompt: 'Maya starts characters talking with spoken lines' }), 'S1').errors.some(
      (e) => /dialogue|speech|voice/i.test(e),
    ),
  );
  assert.ok(
    validateShotSpec(validShot('S1', { imagePrompt: 'Wide shot with an opening title card in bold letters' }), 'S1').errors.some(
      (e) => /title card|subtitle/i.test(e),
    ),
  );

  // Tool integration: write_bible + edit_shot (single and batch)
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-diff-val-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'story');
    fs.mkdirSync(epDir, { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        id: 'Story',
        slug: 'story',
        lang: 'fr',
        aspect: '9:16',
        script: [{ text: 'Elle lit le message sur son téléphone.' }],
        shots: [
          { id: 'S1', size: 'medium', set: 'living' },
          { id: 'S2', size: 'insert', set: 'living' },
          { id: 'S3', size: 'close', set: 'living' },
        ],
      }),
    );
    fs.writeFileSync(
      path.join(epDir, 'timeline.json'),
      JSON.stringify({
        fps: 30,
        total: 270,
        shots: [
          { id: 'S1', len: 90 },
          { id: 'S2', len: 90 },
          { id: 'S3', len: 90 },
        ],
        segments: [
          { shot: 'S1', a: 0, b: 90 },
          { shot: 'S2', a: 90, b: 180 },
          { shot: 'S3', a: 180, b: 270 },
        ],
      }),
    );
    const writeBibleTool = fsTools.find((t) => t.name === 'write_bible');
    const editShotTool = fsTools.find((t) => t.name === 'edit_shot');
    const ctx = { projectId: 'p', projectRoot: dir, dataRoot: dir, log: () => {} };

    const wb = await writeBibleTool.run({ episode: 'story', ...bible }, ctx);
    assert.equal(wb.ok, true, wb.summary);

    const es1 = await editShotTool.run({ episode: 'story', ...validShot('S1') }, ctx);
    assert.equal(es1.ok, true, es1.summary);

    // Batch write S2 (valid French SMS text) + S3 (wrong English language onScreenText to test audit!)
    const batch = await editShotTool.run(
      {
        episode: 'story',
        shots: [
          validShot('S2', {
            characters: [],
            size: 'insert',
            angle: 'high',
            view: 'front',
            cameraMove: 'still',
            imagePrompt:
              'Macro insert looking down at a glowing smartphone screen on a wood table showing SMS "Tu rentres ce soir ?".',
            onScreenText: { kind: 'phone', text: 'Tu rentres ce soir ?', lang: 'fr' },
            videoPrompt: 'Phone screen glows softly with incoming notification, subtle camera hold.',
          }),
          validShot('S3', {
            angle: 'low',
            view: 'front',
            cameraMove: 'still',
            imagePrompt: 'Close shot of Maya reading a paper note "DO NOT OPEN" in dim light.',
            onScreenText: { kind: 'document', text: 'DO NOT OPEN', lang: 'en' },
          }),
        ],
      },
      ctx,
    );
    assert.equal(batch.ok, true, batch.summary);
    assert.equal(fs.existsSync(path.join(epDir, 'shots', 'S2.json')), true);

    const auditErr = shotsAudit(dir, 'story', JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8')), JSON.parse(fs.readFileSync(path.join(epDir, 'timeline.json'), 'utf8'))) ?? '';
    assert.match(auditErr, /S3.*onScreenText.*en.*fr/i, `expected FR/EN mismatch on S3, got: ${auditErr}`);

    // stagingPromptProblems checks bilingual props ('sac' -> 'backpack'), cast mismatches & POV monotony
    const cards = [
      {
        id: 'S1',
        size: 'medium',
        stage: {
          see: 'Maya cherche dans son sac devant Lucas dans le salon.',
          people: [{ who: 'MAYA', where: 'left', does: 'searches in backpack' }],
          objects: ['sac'],
          moves: true,
        },
      },
    ];
    const specsMap = new Map([
      [
        'S1',
        validShot('S1', {
          characters: ['MAYA', 'LUCAS'],
          cameraMove: 'still',
          imagePrompt: 'Maya stands alone on the sofa.',
          videoPrompt: 'Camera remains stationary on Maya standing still.',
        }),
      ],
    ]);
    const stProbs = stagingPromptProblems(cards, specsMap, bible, true, '9:16').join('\n');
    assert.match(stProbs, /S1.*LUCAS/i, 'detects un-staged character LUCAS');
    assert.match(stProbs, /S1.*sac/i, 'detects missing bilingual prop sac/backpack');
    assert.match(stProbs, /S1.*moves=true.*videoPrompt/i, 'detects videoPrompt lacking movement when moves=true');

    const { readEpisodeShots } = await import('../dist/tools/episode-source.js');
    const epSrc = readEpisodeShots(epDir);
    const metrics = diffusionMetrics(epSrc, bible, 'fr');
    assert.match(metrics, /3 plan\(s\) diffusion/);
    assert.match(metrics, /violations détectées.*S3/);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('multi-API media providers: config precedence, /api/settings key masking & Python media_client HTTP drivers (gflow/openai/gemini/http polling)', NEEDS_PY_MEDIA, async () => {
  const { resolveMediaConfig } = await import('../dist/orchestrator/media-providers.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { createApp } = await import('../dist/orchestrator/main.js');
  const { loadConfig } = await import('../dist/orchestrator/config.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-media-cfg-'));
  const srvHits = [];
  let pollCounter = 0;

  // Create a valid 16x16 PNG in Python cleanly and an MP4 using ffmpeg to serve from our local mock HTTP server
  const samplePng = path.join(dir, 'sample.png');
  const sampleMp4 = path.join(dir, 'sample.mp4');
  execFileSync(
    'python3',
    [
      '-c',
      `from PIL import Image; Image.new("RGB", (64, 64), (40, 120, 200)).save(${JSON.stringify(samplePng)})`,
    ],
    { stdio: 'pipe' },
  );
  execFileSync(
    'ffmpeg',
    ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=128x128:r=30:d=1.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', sampleMp4],
    { stdio: 'pipe' },
  );
  const pngB64 = fs.readFileSync(samplePng).toString('base64');
  const mp4Bytes = fs.readFileSync(sampleMp4);

  let srv;
  try {
    // 1. Test precedence: .env < providers.json < DB settings
    const pJsonPath = path.join(dir, 'providers.json');
    fs.writeFileSync(
      pJsonPath,
      JSON.stringify({
        mediaProviders: {
          image: {
            driver: 'openai',
            baseURL: 'https://proxy.example/v1',
            apiKey: '$TEST_IMG_KEY',
            model: 'nano2-from-json',
            refMode: 'multi',
          },
          video: {
            driver: 'gemini',
            baseURL: 'https://gemini.example/v1beta',
            apiKey: '$TEST_VID_KEY',
            model: 'gemini-omni-flash-2.1',
          },
        },
      }),
    );

    const db = new Db(path.join(dir, 'test.db'));
    const env = {
      TEST_IMG_KEY: 'secret-from-providers-json',
      TEST_VID_KEY: 'secret-vid-json',
      WHATIF_IMAGE_MODEL: 'env-overridden-by-json',
    };
    let resolved = resolveMediaConfig(db, env, dir);
    assert.equal(resolved.image.driver, 'openai');
    assert.equal(resolved.image.apiKey, 'secret-from-providers-json');
    assert.equal(resolved.image.model, 'nano2-from-json');
    assert.equal(resolved.video.driver, 'gemini');
    assert.equal(resolved.video.model, 'gemini-omni-flash-2.1');

    // DB overrides providers.json
    db.setSetting('imageProvider', 'gflow');
    db.setSetting('imageModel', 'nano2-db-override');
    resolved = resolveMediaConfig(db, env, dir);
    assert.equal(resolved.image.driver, 'gflow');
    assert.equal(resolved.image.model, 'nano2-db-override');
    db.close();

    // 2. Test /api/settings GET & POST key masking (never leak raw keys)
    const app = createApp({
      ...loadConfig(),
      dataRoot: dir,
      dbPath: path.join(dir, 'app.db'),
      projectsDir: path.join(dir, 'projects'),
    });
    await new Promise((r) => app.server.listen(0, '127.0.0.1', r));
    const appPort = app.server.address().port;
    try {
      const postRes = await fetch(`http://127.0.0.1:${appPort}/api/settings`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          imageProvider: 'gflow',
          imageBaseURL: 'http://127.0.0.1:8080/v1',
          imageApiKey: 'sk-top-secret-img',
          imageModel: 'nano-banana-2-lite',
          videoProvider: 'gemini',
          videoBaseURL: 'https://generativelanguage.googleapis.com/v1beta',
          videoApiKey: 'sk-top-secret-vid',
          videoModel: 'gemini-omni-flash-2.1',
        }),
      });
      assert.equal(postRes.status, 200);
      const getRes = await fetch(`http://127.0.0.1:${appPort}/api/settings`);
      const body = await getRes.json();
      assert.equal(body.media.image.driver, 'gflow');
      assert.equal(body.media.image.model, 'nano-banana-2-lite');
      assert.equal(body.media.image.hasKey, true);
      assert.equal(body.media.video.driver, 'gemini');
      assert.equal(body.media.video.hasKey, true);
      assert.doesNotMatch(JSON.stringify(body), /sk-top-secret/, '/api/settings must NEVER return raw API keys');
    } finally {
      await new Promise((r) => app.server.close(r));
    }

    // 3. Test Python pipeline/media_client.py against live HTTP endpoints (gflow, openai, gemini, http polling)
    srv = http.createServer((req, res) => {
      const urlPath = req.url || '';
      if (req.method === 'GET' && urlPath === '/files/vid-job.mp4') {
        res.writeHead(200, { 'content-type': 'video/mp4' });
        res.end(mp4Bytes);
        return;
      }
      if (req.method === 'GET' && urlPath === '/jobs/job-42') {
        pollCounter++;
        res.writeHead(200, { 'content-type': 'application/json' });
        if (pollCounter === 1) {
          res.end(JSON.stringify({ status: 'running' }));
        } else {
          res.end(JSON.stringify({ status: 'succeeded', result: { output: { videoUrl: `http://127.0.0.1:${srv.address().port}/files/vid-job.mp4` } } }));
        }
        return;
      }
      let raw = '';
      req.on('data', (c) => (raw += c));
      req.on('end', () => {
        const parsed = raw ? JSON.parse(raw) : {};
        srvHits.push({ url: urlPath, auth: req.headers.authorization || '', xGoog: req.headers['x-goog-api-key'] || '', body: parsed });
        res.writeHead(200, { 'content-type': 'application/json' });
        if (urlPath.includes(':generateContent')) {
          res.end(JSON.stringify({ candidates: [{ content: { parts: [{ inlineData: { mimeType: 'image/png', data: pngB64 } }] } }] }));
          return;
        }
        if (urlPath === '/v1/custom/video') {
          res.end(JSON.stringify({ meta: { taskId: 'job-42' } }));
          return;
        }
        res.end(JSON.stringify({ data: [{ b64_json: pngB64 }] }));
      });
    });
    await new Promise((r) => srv.listen(0, '127.0.0.1', r));
    const port = srv.address().port;

    const ref1 = path.join(dir, 'place_LIVING.png');
    const ref2 = path.join(dir, 'char_MAYA.png');
    fs.copyFileSync(samplePng, ref1);
    fs.copyFileSync(samplePng, ref2);

    const outGflow = path.join(dir, 'out_gflow.png');
    const outOpenai = path.join(dir, 'out_openai.png');
    const outGemini = path.join(dir, 'out_gemini.png');
    const outHttpVid = path.join(dir, 'out_http.mp4');

    await execFileAsync(
      'python3',
      [
        '-c',
        `
import sys; sys.path.insert(0, "pipeline"); import media_client as mc
from pathlib import Path

base = "http://127.0.0.1:${port}"
refs = [Path(${JSON.stringify(ref1)}), Path(${JSON.stringify(ref2)})]
labels = ["PLACE:LIVING", "CHAR:MAYA"]

mc.generate_image("Maya in room", "", "16:9", Path(${JSON.stringify(outGflow)}), refs, labels, None,
  {"driver": "gflow", "baseURL": base + "/v1", "apiKey": "gflow-k", "model": "nano2-lite", "refMode": "composite", "endpointPath": "/images/generations"})

mc.generate_image("Maya closeup", "", "16:9", Path(${JSON.stringify(outOpenai)}), refs, labels, None,
  {"driver": "openai", "baseURL": base + "/v1", "apiKey": "oai-k", "model": "nano2-lite", "refMode": "multi", "endpointPath": "/images/generations"})

mc.generate_image("Maya gem", "", "9:16", Path(${JSON.stringify(outGemini)}), refs, labels, None,
  {"driver": "gemini", "baseURL": base + "/v1beta", "apiKey": "gem-k", "model": "nano-banana-2-lite", "refMode": "multi"})

mc.generate_video(Path(${JSON.stringify(outGflow)}), "Maya waves hand, physical footsteps only", 2.0, "16:9", Path(${JSON.stringify(outHttpVid)}), None,
  {"driver": "http", "baseURL": base, "apiKey": "http-k", "model": "gemini-omni-flash-2.1", "endpointPath": "/v1/custom/video", "pollPath": "/jobs/{id}",
   "responsePaths": {"jobId": "meta.taskId", "status": "status", "url": "result.output.videoUrl"}})
print("DRIVERS_OK")
`,
      ],
    );

    assert.equal(srvHits.length, 4);
    assert.ok(srvHits[0].body.image_base64 && !srvHits[0].body.images, 'gflow composite packs multi-refs into ONE image_base64');
    assert.equal(srvHits[0].auth, 'Bearer gflow-k');
    assert.equal(srvHits[1].body.images.length, 2, 'openai multi sends array of ref images');
    assert.equal(srvHits[2].xGoog, 'gem-k', 'gemini uses x-goog-api-key header');
    assert.equal(srvHits[2].body.contents[0].parts.filter((p) => p.inlineData).length, 2);
    assert.equal(pollCounter, 2, 'http generic driver polls job until succeeded');
    assert.ok(fs.statSync(outGflow).size > 200);
    assert.ok(fs.statSync(outGemini).size > 200);
    assert.ok(fs.statSync(outHttpVid).size > 500);
  } finally {
    if (srv) await new Promise((r) => srv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('shotlist.py (no titles, omni-flash sound) & plan_run.py standalone 30fps cue resolver', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-plan-'));
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'nontitle');
    fs.mkdirSync(epDir, { recursive: true });
    const scriptText = 'Il ouvre la porte lentement. Soudain une enveloppe glisse sur le parquet. Qui a posé cette lettre ?';
    const toks = scriptText.split(/\s+/);
    let t = 0.1;
    const words = toks.map((raw) => {
      const m = /^(.+?)([.?]+)?$/.exec(raw);
      const item = { w: m ? m[1] : raw, p: m?.[2] ?? '', s: Number(t.toFixed(3)), e: Number((t + 0.35).toFixed(3)) };
      t += 0.48;
      return item;
    });
    fs.writeFileSync(path.join(epDir, 'words.json'), JSON.stringify(words));
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        id: 'NonTitle',
        slug: 'nontitle',
        aspect: '9:16',
        title: ['PAS UN CARTON'],
        script: [{ text: scriptText }],
        shots: [],
      }),
    );

    execFileSync('python3', [path.resolve('pipeline/shotlist.py'), 'nontitle', '--project', dir], { stdio: 'pipe' });
    const ep = JSON.parse(fs.readFileSync(path.join(epDir, 'episode.json'), 'utf8'));
    assert.equal(ep.shots.some((s) => /title|schematic/i.test(String(s.size || '') + String(s.accent || ''))), false, 'shotlist.py creates ZERO title or schematic shots');
    assert.deepEqual(ep.sound, { provider: 'omni-flash', music: false, dialogue: false });

    // Test plan_run.py cue resolver directly (f:, s:, bare word, #nth offset) and rejecting w: prefix
    execFileSync('python3', [path.resolve('pipeline/plan_run.py'), 'nontitle', '--project', dir], { stdio: 'pipe' });
    const tl = JSON.parse(fs.readFileSync(path.join(epDir, 'timeline.json'), 'utf8'));
    assert.ok(tl.total > 60 && Array.isArray(tl.segments) && tl.segments.length >= 2);
    assert.equal(tl.segments[0].a, 0);
    assert.equal(tl.segments[tl.segments.length - 1].b, tl.total);

    ep.edit = [{ shot: 'S1', at: 'f:0' }, { shot: 'S2', at: 'w:enveloppe' }];
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify(ep));
    let failedW = false;
    try {
      execFileSync('python3', [path.resolve('pipeline/plan_run.py'), 'nontitle', '--project', dir], { stdio: 'pipe' });
    } catch (e) {
      failedW = String(e.stderr || e.message).includes('no "w:" prefix') || e.status !== 0;
    }
    assert.equal(failedW, true, 'plan_run.py rejects legacy w: prefix');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('full mock diffusion E2E: refs -> stills (SHA-256 cache + targeted rerun) -> board -> review gate -> clips -> render + -14 LUFS master & ducked SFX', NEEDS_PY_MEDIA, async () => {
  const { defaultRegistry, fsTools } = await import('../dist/tools/index.js');
  const { clearReviewBlock } = await import('../dist/orchestrator/reviewer.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-e2e-diff-'));
  const slug = 'nightcall';
  const reg = defaultRegistry();
  const ctx = {
    projectId: slug,
    projectRoot: dir,
    dataRoot: dir,
    log: () => {},
    mediaEnv: {
      WHATIF_IMAGE_PROVIDER: 'mock',
      WHATIF_IMAGE_MODEL: 'nano2-lite',
      WHATIF_VIDEO_PROVIDER: 'mock',
      WHATIF_VIDEO_MODEL: 'gemini-omni-flash-2.1',
    },
  };

  try {
    // 1. Scaffold project via whatif_new (zero skillDir needed!)
    const scaffold = await reg.run('whatif_new', { episode: slug, aspect: '9:16' }, ctx);
    assert.equal(scaffold.ok, true, scaffold.summary);

    const epDir = path.join(dir, 'src', 'episodes', slug);
    const pubDir = path.join(dir, 'public', slug);
    const outDir = path.join(dir, 'out', slug);
    fs.mkdirSync(pubDir, { recursive: true });

    // Synthesize a 6.6s voice.wav with speech in [0.5..1.8] & [3.8..5.8] and silence in [2.0..3.6]
    const voiceWav = path.join(pubDir, 'voice.wav');
    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=6.6', '-ar', '44100', '-ac', '1', voiceWav],
      { stdio: 'pipe' },
    );

    const words = [
      { w: 'Minuit', s: 0.5, e: 1.0, p: '' },
      { w: 'sonne', s: 1.1, e: 1.7, p: '.' },
      { w: 'Elle', s: 3.9, e: 4.5, p: '' },
      { w: 'écoute', s: 4.6, e: 5.7, p: '.' },
    ];
    fs.writeFileSync(path.join(epDir, 'words.json'), JSON.stringify(words));
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({
        id: 'Nightcall',
        slug,
        aspect: '9:16',
        lang: 'fr',
        script: [
          { beat: '1', text: 'Minuit sonne dans l appartement silencieux sous la pluie.' },
          { beat: '2', text: 'Le téléphone posé sur la table basse se met à vibrer.' },
          { beat: '3', text: 'Maya décroche lentement et écoute une respiration inconnue.' },
        ],
        science: [{ claim: 'nuit', working: 'silence nocturne', source: 'récit' }],
        shots: [
          { id: 'S1', size: 'wide', set: 'living', note: 'Minuit sonne.' },
          { id: 'S2', size: 'insert', set: 'living', note: 'Le téléphone vibre.' },
          { id: 'S3', size: 'close', set: 'living', note: 'Elle écoute.' },
        ],
        edit: [
          { shot: 'S1', at: 'f:0' },
          { shot: 'S2', at: 's:2.2' },
          { shot: 'S3', at: 's:4.4' },
        ],
      }),
    );

    // 2. plan step (30fps timeline = 198 frames = 6.6s)
    const planRes = await reg.run('whatif_run', { episode: slug, step: 'plan' }, ctx);
    assert.equal(planRes.ok, true, planRes.summary);

    // 3. DIRECTOR.md + write_bible + refs
    fs.writeFileSync(
      path.join(epDir, 'DIRECTOR.md'),
      '# DIRECTOR — Nightcall (9:16)\n' +
        '1. Style 2D cel-shading nocturne crisp ink outlines, palette bleu nuit et ambre chaud, sans aucun carton-titre ni sous-titre.\n' +
        '2. Casting invariant: Maya (veste en jean bleue sur pull crème, carré noir) seule dans le salon moutarde face au téléphone posé sur la table en noyer.\n' +
        '3. Découpage en 3 plans dynamiques (large -> insert téléphone SMS -> gros plan visage) avec bruitages physiques Omni Flash uniquement.',
    );
    const writeBible = fsTools.find((t) => t.name === 'write_bible');
    assert.equal((await writeBible.run({ episode: slug, ...validBible() }, ctx)).ok, true);

    const refsRes = await reg.run('whatif_run', { episode: slug, step: 'refs' }, ctx);
    assert.equal(refsRes.ok, true, refsRes.summary);
    assert.equal(fs.existsSync(path.join(outDir, 'refs', 'place_LIVING.png')), true);
    assert.equal(fs.existsSync(path.join(outDir, 'refs', 'char_MAYA.png')), true);

    // 4. Author staging cards & JSON shots S1..S3 -> run stills (1st run + SHA-256 cached 2nd run + targeted rerun)
    const stRes = await reg.run(
      'stage_shots',
      {
        episode: slug,
        cards: [
          { id: 'S1', see: 'Plan large dans le salon sombre où Maya attend assise sur le canapé moutarde.', subject: 'Maya', people: [{ who: 'MAYA', where: 'center', does: 'sits waiting tensely on sofa' }], objects: ['smartphone'], angle: 'level', view: '3/4-left', cam: 'wide', moves: false },
          { id: 'S2', see: 'Gros plan sur le téléphone posé sur la table affichant Appel inconnu.', subject: 'phone', people: [], objects: ['smartphone'], angle: 'high', view: 'front', cam: 'close', moves: false, change: 'camera cuts close to table' },
          { id: 'S3', see: 'Gros plan sur le visage inquiet de Maya décrochant l appel près de la fenêtre.', subject: 'Maya face', people: [{ who: 'MAYA', where: 'center', does: 'holds phone to ear listening closely' }], objects: ['smartphone'], angle: 'level', view: '3/4-left', cam: 'close', moves: true, change: 'closer facial reaction' },
        ],
      },
      ctx,
    );
    assert.equal(stRes.ok, true, stRes.summary);

    const editShot = fsTools.find((t) => t.name === 'edit_shot');
    assert.equal(
      (
        await editShot.run(
          {
            episode: slug,
            shots: [
              validShot('S1', { view: '3/4-left', cameraMove: 'still', imagePrompt: 'Wide nocturnal 2D living room, Maya waiting with smartphone on mustard sofa.' }),
              validShot('S2', {
                characters: [],
                size: 'insert',
                angle: 'high',
                view: 'front',
                cameraMove: 'still',
                imagePrompt: 'Overhead close insert on smartphone sitting on dark table showing "Appel inconnu".',
                onScreenText: { kind: 'phone', text: 'Appel inconnu', lang: 'fr' },
                videoPrompt: 'Smartphone screen pulses faintly with incoming ring light, camera steady.',
              }),
              validShot('S3', { size: 'close', view: '3/4-left', cameraMove: 'push', imagePrompt: 'Close-up of Maya lifting smartphone to ear by rainy window.', videoPrompt: 'Maya steps closer to the rainy window and leans slightly while listening intently, gentle push in, strictly no dialogue.' }),
            ],
          },
          ctx,
        )
      ).ok,
      true,
    );

    const stills1 = await reg.run('whatif_run', { episode: slug, step: 'stills' }, ctx);
    assert.equal(stills1.ok, true, stills1.summary);
    const stillsManifest = JSON.parse(fs.readFileSync(path.join(outDir, 'board', 'manifest.json'), 'utf8'));
    assert.ok(stillsManifest.S1?.hash && stillsManifest.S2?.hash && stillsManifest.S3?.hash);

    // 2nd stills run must use SHA-256 cache (0 generated)
    const stills2 = await reg.run('whatif_run', { episode: slug, step: 'stills' }, ctx);
    assert.equal(stills2.ok, true, stills2.summary);
    assert.match(stills2.summary, /0 generated, 3 cached/);

    // Edit S2 -> targeted stills ['S2'] regenerates only S2
    assert.equal(
      (
        await editShot.run(
          {
            episode: slug,
            ...validShot('S2', {
              characters: [],
              size: 'insert',
              angle: 'high',
              view: 'front',
              cameraMove: 'still',
              imagePrompt: 'Updated closer overhead insert on smartphone screen reading "Appel inconnu".',
              onScreenText: { kind: 'phone', text: 'Appel inconnu', lang: 'fr' },
            }),
          },
          ctx,
        )
      ).ok,
      true,
    );
    const stillsTargeted = await reg.run('whatif_run', { episode: slug, step: 'stills', args: ['S2'] }, ctx);
    assert.equal(stillsTargeted.ok, true, stillsTargeted.summary);
    assert.match(stillsTargeted.summary, /1 generated, 0 cached/);

    // 5. Build storyboard sheets & verify Reviewer blocking gate on BOTH clips & render
    const boardRes = await reg.run('whatif_run', { episode: slug, step: 'board' }, ctx);
    assert.equal(boardRes.ok, true, boardRes.summary);
    assert.equal(fs.existsSync(path.join(outDir, 'storyboard.jpg')), true);
    assert.equal(fs.existsSync(path.join(outDir, 'sheets', 'sheet-01.jpg')), true);

    fs.writeFileSync(
      path.join(outDir, 'review.json'),
      JSON.stringify({ ok: true, pass: false, blocking: true, summary: 'S2 too dark', issues: [{ shot: 'S2', severity: 'error', problem: 'dim', fix: 'brighten' }], createdAt: new Date().toISOString() }),
    );
    const blockedClips = await reg.run('whatif_run', { episode: slug, step: 'clips' }, ctx);
    assert.equal(blockedClips.ok, false);
    assert.match(blockedClips.summary, /refused this board/);
    const blockedRender = await reg.run('whatif_run', { episode: slug, step: 'render' }, ctx);
    assert.equal(blockedRender.ok, false);

    // Clear block (simulate Reviewer pass) -> clips + render + -14 LUFS master!
    clearReviewBlock(dir, slug);
    const clipsRes = await reg.run('whatif_run', { episode: slug, step: 'clips' }, ctx);
    assert.equal(clipsRes.ok, true, clipsRes.summary);
    assert.equal(fs.existsSync(path.join(outDir, 'clips', 'S1.mp4')), true);
    assert.equal(fs.existsSync(path.join(outDir, 'clips', 'S3.mp4')), true);

    const renderRes = await reg.run('whatif_run', { episode: slug, step: 'render' }, ctx);
    assert.equal(renderRes.ok, true, renderRes.summary);
    assert.equal(fs.existsSync(path.join(outDir, `${slug}-raw.mp4`)), true);
    assert.equal(fs.existsSync(path.join(outDir, `${slug}-final.mp4`)), true);
    assert.equal(fs.existsSync(path.join(pubDir, 'sfx.wav')), true);

    // Verify sfx.wav is actually ducked during spoken intervals vs silent gap (t=0.5..1.7 vs t=2.2..3.4)
    const duckCheck = execFileSync(
      'python3',
      [
        '-c',
        `
import sys; sys.path.insert(0, "pipeline"); import assemble_run as ar, numpy as np
s = ar.decode_audio_f64(${JSON.stringify(path.join(pubDir, 'sfx.wav'))})
rms_speech = float(np.sqrt(np.mean(s[int(0.6*44100):int(1.6*44100)]**2)))
rms_gap = float(np.sqrt(np.mean(s[int(2.4*44100):int(3.3*44100)]**2)))
print(f"{rms_speech:.6f},{rms_gap:.6f}")
assert rms_gap > rms_speech * 1.4, (rms_speech, rms_gap)
`,
      ],
      { encoding: 'utf8' },
    );
    assert.ok(duckCheck.includes(','), duckCheck);

    // Verify exact final duration (~6.6s) and loudness report near -14 LUFS
    const finalDur = Number(
      execFileSync(
        'ffprobe',
        ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path.join(outDir, `${slug}-final.mp4`)],
        { encoding: 'utf8' },
      ).trim(),
    );
    assert.ok(Math.abs(finalDur - 6.6) < 0.25, `expected ~6.6s final MP4, got ${finalDur}s`);
    const report = JSON.parse(fs.readFileSync(path.join(outDir, 'report.json'), 'utf8'));
    assert.ok(Number(report.lufs) <= -12.0 && Number(report.lufs) >= -16.5, `unexpected LUFS: ${JSON.stringify(report)}`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('default port 3020 & video resolution + optional Modal Anime 2D upscaler config resolution', async () => {
  const { loadConfig } = await import('../dist/orchestrator/config.js');
  const { Db } = await import('../dist/orchestrator/db.js');
  const { mediaEnvForSubprocess, resolveMediaConfig } = await import('../dist/orchestrator/media-providers.js');

  const prevPort = process.env.WHATIF_PORT;
  delete process.env.WHATIF_PORT;
  try {
    const cfg = loadConfig();
    assert.equal(cfg.port, 3020);
  } finally {
    if (prevPort !== undefined) process.env.WHATIF_PORT = prevPort;
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-upcfg-'));
  try {
    const db = new Db(path.join(tmp, 'studio.db'));
    db.setSetting('videoResolution', '720p');
    db.setSetting('upscalerEnabled', '1');
    db.setSetting('upscalerTarget', 'auto-uhd');
    db.setSetting('upscalerModel', 'realesr-animevideov3');
    db.setSetting('upscalerDenoise', '2');
    db.setSetting('upscalerGpu', 'A100-40GB');
    db.setSetting('upscalerBaseUrl', 'https://bkrpulsy--anime-upscaler-t4-fastapi-app.modal.run');

    const resolved = resolveMediaConfig(db, {}, tmp);
    assert.equal(resolved.video.resolution, '720p');
    assert.equal(resolved.upscaler.enabled, true);
    assert.equal(resolved.upscaler.target, 'auto-uhd');
    assert.equal(resolved.upscaler.model, 'realesr-animevideov3');
    assert.equal(resolved.upscaler.denoiseLevel, 2);
    assert.equal(resolved.upscaler.gpuType, 'A100-40GB');

    const envOut = mediaEnvForSubprocess(db, {}, tmp);
    assert.equal(envOut.WHATIF_VIDEO_RESOLUTION, '720p');
    assert.equal(envOut.WHATIF_UPSCALER_ENABLED, '1');
    assert.equal(envOut.WHATIF_UPSCALER_TARGET, 'auto-uhd');
    const jsonCfg = JSON.parse(envOut.WHATIF_MEDIA_CONFIG_JSON);
    assert.equal(jsonCfg.video.resolution, '720p');
    assert.equal(jsonCfg.upscaler.enabled, true);
    db.close();
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('360p/720p diffusion resolution & Modal Anime 2D upscaler HTTP workflow + render auto-chaining', NEEDS_PY_MEDIA, async () => {
  const { defaultRegistry } = await import('../dist/tools/index.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-upscale-e2e-'));
  let server = null;
  try {
    // 1. Verify native 360p (9:16 -> 360x640) & 720p (16:9 -> 1280x720) helper dimensions
    const dims = execFileSync(
      'python3',
      [
        '-c',
        'import json, sys; sys.path.insert(0, "pipeline"); import media_client as mc; print(json.dumps({"s916_360": mc.target_dimensions("9:16", "360p"), "w169_720": mc.target_dimensions("16:9", "720p")}))',
      ],
      { encoding: 'utf8' },
    );
    const parsedDims = JSON.parse(dims.trim());
    assert.deepEqual(parsedDims.s916_360, [360, 640]);
    assert.deepEqual(parsedDims.w169_720, [1280, 720]);

    // 2. Prepare a 720p (1280x720) mastered MP4 with stereo AAC audio and test Modal Anime 2D API contract
    const slug = 'up-ep';
    const outDir = path.join(dir, 'out', slug);
    const epDir = path.join(dir, 'src', 'episodes', slug);
    fs.mkdirSync(outDir, { recursive: true });
    fs.mkdirSync(epDir, { recursive: true });
    fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ id: 'UpEp', slug, aspect: '16:9', lang: 'fr' }));

    const baseFinal = path.join(outDir, `${slug}-final.mp4`);
    execFileSync(
      'ffmpeg',
      [
        '-y',
        '-v',
        'error',
        '-f',
        'lavfi',
        '-i',
        'color=c=0x1e2c44:s=1280x720:d=1.2:r=30',
        '-f',
        'lavfi',
        '-i',
        'sine=frequency=440:duration=1.2',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        '128k',
        '-shortest',
        baseFinal,
      ],
      { stdio: 'pipe' },
    );

    let postCount = 0;
    let pollCount = 0;
    let downloaded = false;
    let deleted = false;
    let uploadedMp4 = path.join(dir, 'uploaded.mp4');

    server = http.createServer(async (req, res) => {
      const url = new URL(req.url || '/', 'http://127.0.0.1');
      if (req.method === 'POST' && url.pathname === '/upscale') {
        postCount += 1;
        const chunks = [];
        for await (const c of req) chunks.push(c);
        const body = Buffer.concat(chunks);
        const txt = body.toString('latin1');
        assert.match(txt, /name="model_name"\r\n\r\nreal-cugan/);
        assert.match(txt, /name="output_scale"\r\n\r\n3/);
        assert.match(txt, /name="denoise_level"\r\n\r\n1/);
        assert.match(txt, /name="gpu_type"\r\n\r\nL40S/);
        assert.match(txt, /name="file"; filename="up-ep-video\.mp4"/);
        // Save valid raw MP4 bytes from marker onward (or render 3840x2160 directly in /download)
        fs.writeFileSync(uploadedMp4, body.subarray(0, Math.min(body.length, 4096)));
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ task_id: 'task-modal-01', status: 'queued' }));
        return;
      }
      if (req.method === 'GET' && url.pathname === '/status/task-modal-01') {
        pollCount += 1;
        res.writeHead(200, { 'content-type': 'application/json' });
        if (pollCount < 2) {
          res.end(JSON.stringify({ task_id: 'task-modal-01', status: 'processing', progress: 50 }));
        } else {
          res.end(JSON.stringify({ task_id: 'task-modal-01', status: 'completed', progress: 100 }));
        }
        return;
      }
      if (req.method === 'GET' && url.pathname === '/download/task-modal-01') {
        downloaded = true;
        const uhdPath = path.join(dir, 'modal-uhd-out.mp4');
        await execFileAsync('ffmpeg', [
          '-y',
          '-v',
          'error',
          '-f',
          'lavfi',
          '-i',
          'color=c=0x1e2c44:s=3840x2160:d=1.2:r=30',
          '-pix_fmt',
          'yuv420p',
          '-an',
          uhdPath,
        ]);
        const data = fs.readFileSync(uhdPath);
        res.writeHead(200, { 'content-type': 'video/mp4', 'content-length': String(data.length) });
        res.end(data);
        return;
      }
      if (req.method === 'DELETE' && url.pathname === '/task/task-modal-01') {
        deleted = true;
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
      }
      res.writeHead(404);
      res.end();
    });

    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;

    const reg = defaultRegistry();
    const upCtx = {
      projectId: 'up',
      projectRoot: dir,
      dataRoot: dir,
      log: () => {},
      mediaEnv: {
        ...process.env,
        WHATIF_UPSCALER_ENABLED: '1',
        WHATIF_UPSCALER_DRIVER: 'modal-anime',
        WHATIF_UPSCALER_BASE_URL: `http://127.0.0.1:${port}`,
        WHATIF_UPSCALER_TARGET: 'auto-uhd',
        WHATIF_UPSCALER_MODEL: 'real-cugan',
        WHATIF_UPSCALER_DENOISE: '1',
        WHATIF_UPSCALER_GPU: 'L40S',
      },
    };

    const upRes = await reg.run('whatif_run', { episode: slug, step: 'upscale' }, upCtx);
    assert.equal(upRes.ok, true, upRes.summary);
    assert.equal(postCount, 1);
    assert.ok(pollCount >= 2, `expected >= 2 polls, got ${pollCount}`);
    assert.equal(downloaded, true);
    assert.equal(deleted, true);

    const probeFinal = execFileSync(
      'ffprobe',
      ['-v', 'error', '-show_entries', 'stream=codec_type,codec_name,width,height', '-of', 'json', baseFinal],
      { encoding: 'utf8' },
    );
    const streams = JSON.parse(probeFinal).streams || [];
    const vStream = streams.find((s) => s.codec_type === 'video');
    const aStream = streams.find((s) => s.codec_type === 'audio');
    assert.equal(vStream?.width, 3840);
    assert.equal(vStream?.height, 2160);
    assert.equal(aStream?.codec_name, 'aac', 'Master audio track must be preserved untouched alongside upscaled UHD video');
    assert.equal(fs.existsSync(path.join(outDir, `${slug}-base.mp4`)), true);

    const upCached = await reg.run('whatif_run', { episode: slug, step: 'upscale' }, upCtx);
    assert.equal(upCached.ok, true, upCached.summary);
    assert.equal(postCount, 1, 'second upscale run should use SHA-256 cache');

    // 3. Test 360p (9:16) render auto-chaining into mock Anime 2D upscaler (auto-2k -> 1440x2560)
    const shortSlug = 'short-up';
    const shortEp = path.join(dir, 'src', 'episodes', shortSlug);
    const shortPub = path.join(dir, 'public', shortSlug);
    const shortOut = path.join(dir, 'out', shortSlug);
    fs.mkdirSync(shortEp, { recursive: true });
    fs.mkdirSync(shortPub, { recursive: true });
    fs.mkdirSync(path.join(shortOut, 'board'), { recursive: true });
    fs.mkdirSync(path.join(shortOut, 'refs'), { recursive: true });

    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=330:duration=2.0', '-ar', '44100', '-ac', '1', path.join(shortPub, 'voice.wav')],
      { stdio: 'pipe' },
    );
    fs.writeFileSync(
      path.join(shortEp, 'words.json'),
      JSON.stringify([{ w: 'Regarde', s: 0.2, e: 0.7, p: '.' }, { w: 'Vite', s: 0.9, e: 1.5, p: '.' }]),
    );
    fs.writeFileSync(
      path.join(shortEp, 'DIRECTOR.md'),
      `${'Ligne directrice 2D courte 9:16 avec arrière-plan détaillé, ombres marquées et contraste fort. '.repeat(4)}`,
    );
    fs.writeFileSync(
      path.join(shortEp, 'episode.json'),
      JSON.stringify({
        id: 'ShortUp',
        slug: shortSlug,
        aspect: '9:16',
        lang: 'fr',
        script: [
          { beat: '1', text: 'Minuit sonne dans le salon silencieux.' },
          { beat: '2', text: 'Le téléphone se met soudain à vibrer.' },
          { beat: '3', text: 'Maya regarde vite l écran dans l ombre.' },
        ],
        science: [{ claim: 'nuit', working: 'silence nocturne', source: 'récit' }],
        shots: [{ id: 'S1', size: 'medium', set: 'living', note: 'Regarde.' }],
        edit: [{ shot: 'S1', at: 'f:0' }],
      }),
    );
    fs.writeFileSync(
      path.join(shortEp, 'timeline.json'),
      JSON.stringify({ fps: 30, total: 60, shots: [{ id: 'S1', len: 60 }], segments: [{ shot: 'S1', a: 0, b: 60 }] }),
    );
    const shortCtx = {
      projectId: 'short-up',
      projectRoot: dir,
      dataRoot: dir,
      log: () => {},
      mediaEnv: {
        ...process.env,
        NODE_TEST_CONTEXT: '1',
        WHATIF_IMAGE_PROVIDER: 'mock',
        WHATIF_VIDEO_PROVIDER: 'mock',
        WHATIF_VIDEO_RESOLUTION: '360p',
        WHATIF_UPSCALER_ENABLED: '1',
        WHATIF_UPSCALER_DRIVER: 'mock',
        WHATIF_UPSCALER_TARGET: 'auto-2k',
      },
    };
    const wbShort = await reg.run('write_bible', { episode: shortSlug, ...validBible() }, shortCtx);
    assert.equal(wbShort.ok, true, wbShort.summary);
    const stShort = await reg.run(
      'stage_shots',
      {
        episode: shortSlug,
        cards: [
          {
            id: 'S1',
            see: 'Plan moyen dans le salon sombre où Maya tient son téléphone assise sur le canapé moutarde.',
            subject: 'Maya',
            people: [{ who: 'MAYA', where: 'center', does: 'sits holding smartphone on sofa' }],
            objects: ['smartphone'],
            angle: 'level',
            view: '3/4-left',
            cam: 'medium',
            moves: false,
          },
        ],
      },
      shortCtx,
    );
    assert.equal(stShort.ok, true, stShort.summary);
    const s1Spec = validShot('S1');
    delete s1Spec.onScreenText;
    const edShort = await reg.run('edit_shot', { episode: shortSlug, ...s1Spec }, shortCtx);
    assert.equal(edShort.ok, true, edShort.summary);

    const rRefs = await reg.run('whatif_run', { episode: shortSlug, step: 'refs' }, shortCtx);
    assert.equal(rRefs.ok, true, rRefs.summary);
    const rStills = await reg.run('whatif_run', { episode: shortSlug, step: 'stills' }, shortCtx);
    assert.equal(rStills.ok, true, rStills.summary);
    const rBoard = await reg.run('whatif_run', { episode: shortSlug, step: 'board' }, shortCtx);
    assert.equal(rBoard.ok, true, rBoard.summary);
    fs.writeFileSync(
      path.join(shortOut, 'review.json'),
      JSON.stringify({ ok: true, pass: true, blocking: false, summary: 'ok', issues: [], createdAt: new Date().toISOString() }),
    );
    const rAuto = await reg.run('whatif_run', { episode: shortSlug, step: 'render' }, shortCtx);
    assert.equal(rAuto.ok, true, rAuto.summary);

    const rawDim = execFileSync(
      'ffprobe',
      ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', path.join(shortOut, `${shortSlug}-raw.mp4`)],
      { encoding: 'utf8' },
    ).trim();
    assert.equal(rawDim, '360x640', 'raw render should honor 360p (9:16)');

    const finalDim = execFileSync(
      'ffprobe',
      ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', path.join(shortOut, `${shortSlug}-final.mp4`)],
      { encoding: 'utf8' },
    ).trim();
    assert.equal(finalDim, '1440x2560', 'auto-upscaled final render should reach 2K QHD (9:16)');

    const st = await reg.run('whatif_status', { episode: shortSlug }, shortCtx);
    assert.equal(st.ok, true);
    assert.equal(st.data.upscale?.width, 1440);
    assert.equal(st.data.upscale?.height, 2560);
  } finally {
    if (server) await new Promise((resolve) => server.close(resolve));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('gflow :8006 REST v0.101.0 contract: per-project Flow isolation, nano2-lite -> nano2.1 fallback/upgrade & omni-1.2-flash 360p i2v polling', NEEDS_PY_MEDIA, async () => {
  const { resolveMediaConfig } = await import('../dist/orchestrator/media-providers.js');
  const { Db } = await import('../dist/orchestrator/db.js');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-gflow-v101-'));
  let srv = null;
  try {
    // 1. Verify gflow driver defaults point to http://127.0.0.1:8006, nano2-lite (fallback nano2.1), omni-1.2-flash @ 360p
    const db = new Db(path.join(dir, 'cfg.db'));
    const gCfg = resolveMediaConfig(db, { WHATIF_IMAGE_PROVIDER: 'gflow', WHATIF_VIDEO_PROVIDER: 'gflow' }, dir);
    assert.equal(gCfg.image.baseURL, 'http://127.0.0.1:8006');
    assert.equal(gCfg.image.model, 'nano2-lite');
    assert.equal(gCfg.image.fallbackModel, 'nano2.1');
    assert.equal(gCfg.image.refMode, 'multi');
    assert.equal(gCfg.video.baseURL, 'http://127.0.0.1:8006');
    assert.equal(gCfg.video.model, 'omni-1.2-flash');
    assert.equal(gCfg.video.resolution, '360p');
    db.close();

    // Build small real PNG & MP4 fixture files served over /v1/files/*
    const samplePng = path.join(dir, 'sample.png');
    const sampleMp4 = path.join(dir, 'sample.mp4');
    execFileSync(
      'python3',
      ['-c', `from PIL import Image; Image.new("RGB", (64, 64), (30, 90, 160)).save(${JSON.stringify(samplePng)})`],
      { stdio: 'pipe' },
    );
    execFileSync(
      'ffmpeg',
      ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=0x16243a:s=360x640:r=30:d=1.5', '-f', 'lavfi', '-i', 'sine=frequency=180:duration=1.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', sampleMp4],
      { stdio: 'pipe' },
    );
    const pngBytes = fs.readFileSync(samplePng);
    const mp4Bytes = fs.readFileSync(sampleMp4);

    let projSeq = 0;
    let jobSeq = 0;
    const createdProjects = [];
    const imgRequests = [];
    const vidRequests = [];
    const jobs = new Map();

    srv = http.createServer(async (req, res) => {
      const url = new URL(req.url || '/', 'http://127.0.0.1');
      if (req.method === 'GET' && url.pathname === '/health') {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok' }));
        return;
      }
      if (req.method === 'GET' && url.pathname.startsWith('/v1/files/')) {
        const isVid = url.pathname.endsWith('.mp4');
        res.writeHead(200, { 'content-type': isVid ? 'video/mp4' : 'image/png' });
        res.end(isVid ? mp4Bytes : pngBytes);
        return;
      }
      if (req.method === 'GET' && url.pathname.startsWith('/v1/jobs/')) {
        const jid = url.pathname.slice('/v1/jobs/'.length);
        const st = jobs.get(jid);
        if (!st) {
          res.writeHead(404);
          res.end();
          return;
        }
        st.polls += 1;
        res.writeHead(200, { 'content-type': 'application/json' });
        if (st.polls === 1) {
          res.end(JSON.stringify({ job_id: jid, status: 'processing', task_type: st.type, created_at: 1, check_url: `/v1/jobs/${jid}` }));
        } else if (st.fail) {
          res.end(JSON.stringify({ job_id: jid, status: 'failed', task_type: st.type, created_at: 1, data: [], error: 'simulated lite error', check_url: `/v1/jobs/${jid}` }));
        } else {
          res.end(
            JSON.stringify({
              job_id: jid,
              status: 'succeeded',
              task_type: st.type,
              created_at: 1,
              completed_at: 2,
              data: [
                {
                  project_id: st.project,
                  profile: 'testprof',
                  url: isVidUrl(st.type) ? `/v1/files/${jid}.mp4` : `/v1/files/${jid}.png`,
                  local_path: '/tmp/x',
                  media_name: st.mediaName,
                  media_type: st.type,
                },
              ],
              error: null,
              check_url: `/v1/jobs/${jid}`,
            }),
          );
        }
        return;
      }
      let raw = '';
      for await (const chunk of req) raw += chunk;
      const body = raw ? JSON.parse(raw) : {};
      if (req.method === 'POST' && url.pathname === '/v1/projects') {
        projSeq += 1;
        const pid = `proj-uuid-${projSeq}`;
        createdProjects.push({ ...body, project_id: pid });
        res.writeHead(201, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ project_id: pid, title: body.title || 'untitled', profile: body.profile || 'testprof', url: `https://labs.google/fx/tools/flow/project/${pid}`, is_default: false }));
        return;
      }
      if (req.method === 'POST' && url.pathname === '/v1/images/generations') {
        imgRequests.push(body);
        jobSeq += 1;
        const jid = `img_${jobSeq}`;
        const shouldFail = String(body.prompt || '').includes('TRIGGER_FALLBACK') && body.model === 'nano2-lite';
        const mediaName = `00000000-0000-4000-8000-${String(jobSeq).padStart(12, '0')}`;
        jobs.set(jid, { polls: 0, type: 'image', fail: shouldFail, project: body.project, mediaName });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ job_id: jid, status: 'pending', task_type: 'image', created_at: 1, data: [], check_url: `/v1/jobs/${jid}` }));
        return;
      }
      if (req.method === 'POST' && url.pathname === '/v1/videos/generations') {
        vidRequests.push(body);
        jobSeq += 1;
        const jid = `vid_${jobSeq}`;
        const mediaName = `00000000-0000-4000-8000-${String(jobSeq).padStart(12, '0')}`;
        jobs.set(jid, { polls: 0, type: 'video', fail: false, project: body.project, mediaName });
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ job_id: jid, status: 'pending', task_type: 'video', created_at: 1, data: [], check_url: `/v1/jobs/${jid}` }));
        return;
      }
      res.writeHead(404);
      res.end();
    });

    function isVidUrl(t) {
      return t === 'video';
    }

    await new Promise((r) => srv.listen(0, '127.0.0.1', r));
    const port = srv.address().port;
    const base = `http://127.0.0.1:${port}`;

    // Build two isolated Studio episodes -> must create TWO distinct Flow project IDs
    const makeEp = (slug, aspect) => {
      const epDir = path.join(dir, 'src', 'episodes', slug);
      const outDir = path.join(dir, 'out', slug);
      fs.mkdirSync(path.join(epDir, 'shots'), { recursive: true });
      fs.mkdirSync(outDir, { recursive: true });
      fs.writeFileSync(path.join(epDir, 'episode.json'), JSON.stringify({ id: slug, slug, aspect, lang: 'fr' }));
      fs.writeFileSync(path.join(epDir, 'timeline.json'), JSON.stringify({ fps: 30, aspect, total: 90, shots: [{ id: 'S1', len: 90 }], segments: [{ shot: 'S1', a: 0, b: 90 }] }));
      fs.writeFileSync(path.join(epDir, 'bible.json'), JSON.stringify(validBible()));
      fs.writeFileSync(path.join(epDir, 'shots', 'S1.json'), JSON.stringify(validShot('S1', { imagePrompt: 'Maya stands in living room holding coffee cup, 2D cel animation.' })));
    };
    makeEp('ep-a', '9:16');
    makeEp('ep-b', '16:9');

    const pyEnv = {
      ...process.env,
      NODE_TEST_CONTEXT: '1',
      WHATIF_IMAGE_PROVIDER: 'gflow',
      WHATIF_IMAGE_BASE_URL: base,
      WHATIF_IMAGE_MODEL: 'nano2-lite',
      WHATIF_IMAGE_FALLBACK_MODEL: 'nano2.1',
      WHATIF_IMAGE_REF_MODE: 'multi',
      WHATIF_VIDEO_PROVIDER: 'gflow',
      WHATIF_VIDEO_BASE_URL: base,
      WHATIF_VIDEO_MODEL: 'omni-1.2-flash',
      WHATIF_VIDEO_RESOLUTION: '360p',
    };

    // Run refs on ep-a & ep-b via image_gen.py with quick pollInterval override via WHATIF_MEDIA_CONFIG_JSON
    const cfgForEnv = (overrides = {}) =>
      JSON.stringify({
        image: {
          driver: 'gflow',
          baseURL: base,
          apiKey: '',
          model: 'nano2-lite',
          fallbackModel: 'nano2.1',
          refMode: 'multi',
          maxRefs: 10,
          pollInterval: 0.05,
          ...overrides,
        },
        video: {
          driver: 'gflow',
          baseURL: base,
          apiKey: '',
          model: 'omni-1.2-flash',
          resolution: '360p',
          pollInterval: 0.05,
        },
      });

    await execFileAsync('python3', ['pipeline/image_gen.py', 'ep-a', '--mode', 'refs', '--only', 'MAYA,LIVING', '--project', dir], {
      env: { ...pyEnv, WHATIF_MEDIA_CONFIG_JSON: cfgForEnv() },
    });
    await execFileAsync('python3', ['pipeline/image_gen.py', 'ep-b', '--mode', 'refs', '--only', 'LIVING', '--project', dir], {
      env: { ...pyEnv, WHATIF_MEDIA_CONFIG_JSON: cfgForEnv() },
    });

    assert.equal(createdProjects.length, 2, 'each episode must create its own Flow project');
    assert.equal(createdProjects[0].set_default, false);
    assert.equal(createdProjects[1].set_default, false);
    const gpA = JSON.parse(fs.readFileSync(path.join(dir, 'out', 'ep-a', 'gflow-project.json'), 'utf8'));
    const gpB = JSON.parse(fs.readFileSync(path.join(dir, 'out', 'ep-b', 'gflow-project.json'), 'utf8'));
    assert.notEqual(gpA.project_id, gpB.project_id);

    const refsManifestA = JSON.parse(fs.readFileSync(path.join(dir, 'out', 'ep-a', 'refs', 'manifest.json'), 'utf8'));
    assert.ok(refsManifestA.place_LIVING?.mediaName && refsManifestA.char_MAYA?.mediaName);

    // Run stills + video on ep-a -> must reuse gpA.project_id (still 2 projects total!) & use ref_media_ids + initial_frame UUID (no file uploads!)
    await execFileAsync('python3', ['pipeline/image_gen.py', 'ep-a', '--mode', 'stills', '--project', dir], {
      env: { ...pyEnv, WHATIF_MEDIA_CONFIG_JSON: cfgForEnv() },
    });
    const stillReq = imgRequests[imgRequests.length - 1];
    assert.equal(stillReq.ref_media_ids?.length, 2, 'stills should use existing in-project ref_media_ids UUIDs');
    assert.equal(stillReq.images_base64, undefined, 'should NOT upload images_base64 when ref_media_ids exist');
    assert.equal(stillReq.image_base64, undefined);

    await execFileAsync('python3', ['pipeline/video_gen.py', 'ep-a', '--project', dir], {
      env: { ...pyEnv, WHATIF_MEDIA_CONFIG_JSON: cfgForEnv() },
    });
    assert.equal(createdProjects.length, 2, 'subsequent stills & clips on ep-a must reuse gpA.project_id');
    assert.equal(fs.existsSync(path.join(dir, 'out', 'ep-a', 'board', 'S1.png')), true);
    assert.equal(fs.existsSync(path.join(dir, 'out', 'ep-a', 'clips', 'S1.mp4')), true);

    const boardManifestA = JSON.parse(fs.readFileSync(path.join(dir, 'out', 'ep-a', 'board', 'manifest.json'), 'utf8'));
    const lastVid = vidRequests[vidRequests.length - 1];
    assert.equal(lastVid.model, 'omni-1.2-flash');
    assert.equal(lastVid.mode, 'i2v');
    assert.equal(lastVid.resolution, '360p');
    assert.equal(lastVid.duration, 3);
    assert.equal(lastVid.aspect, '9:16');
    assert.equal(lastVid.project, gpA.project_id);
    assert.equal(lastVid.initial_frame, boardManifestA.S1.mediaName, 'i2v clip should reference generated S1 media_name UUID');
    assert.equal(lastVid.image_base64, undefined, 'should NOT upload image_base64 when initial_frame UUID is present');

    // Direct Python test: (a) nano2-lite job failure automatically falls back to nano2.1; (b) >3 refs auto-upgrades nano2-lite -> nano2.1
    const fbOut = path.join(dir, 'fb.png');
    const upgOut = path.join(dir, 'upg.png');
    const refPaths = [samplePng, samplePng, samplePng, samplePng].map((p) => path.resolve(p));
    await execFileAsync('python3', [
      '-c',
      `
import sys; sys.path.insert(0, "pipeline"); import media_client as mc
from pathlib import Path
cfg = {"driver": "gflow", "baseURL": ${JSON.stringify(base)}, "model": "nano2-lite", "fallbackModel": "nano2.1", "refMode": "multi", "maxRefs": 10, "pollInterval": 0.05, "project": "proj-uuid-1"}
r1 = mc.generate_image("TRIGGER_FALLBACK test shot", "", "16:9", Path(${JSON.stringify(fbOut)}), None, None, None, cfg)
assert r1["model"] == "nano2.1", r1
refs4 = [Path(${JSON.stringify(refPaths[0])}) for _ in range(4)]
r2 = mc.generate_image("4 refs test", "", "16:9", Path(${JSON.stringify(upgOut)}), refs4, ["P", "C1", "C2", "C3"], None, cfg)
assert r2["model"] == "nano2.1", r2
print("FALLBACK_AND_UPGRADE_OK")
`,
    ]);
    const fourRefReq = imgRequests[imgRequests.length - 1];
    assert.equal(fourRefReq.model, 'nano2.1');
    assert.equal(fourRefReq.images_base64.length, 4);
  } finally {
    if (srv) await new Promise((r) => srv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('regression guards: comma negative clauses, Codex nullable/empty shots[], plural-clone & left/right staging, refs combine, and codex/agy retries', async () => {
  const { validateShotSpec, hasPositiveTitleRequest, hasPositiveDialogueRequest } = await import('../dist/tools/episode-source.js');
  const { fsTools } = await import('../dist/tools/index.js');
  const { validateSchema } = await import('../dist/tools/types.js');
  const { stagingPromptProblems } = await import('../dist/tools/staging.js');
  const { codexProvider } = await import('../dist/orchestrator/providers/codex.js');
  const { agyProvider } = await import('../dist/orchestrator/providers/agy.js');

  // 1. Comma-separated negative clauses + dialogue verbs
  assert.equal(hasPositiveTitleRequest('No 3D, photorealism, title cards or subtitles.'), false);
  assert.equal(hasPositiveDialogueRequest('Officer stands on porch while questioning TEEN'), true);

  // 2. edit_shot schema accepts onScreenText: null and Codex-style { id, shots: [], shot: { onScreenText: { kind, text: '' }, seed: null } }
  const editShotTool = fsTools.find((t) => t.name === 'edit_shot');
  assert.deepEqual(
    validateSchema(editShotTool.schema, {
      episode: 'story',
      id: 'S1',
      ...validShot('S1', { onScreenText: null, seed: null }),
    }),
    [],
  );

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'whatif-reg-'));
  let codexSrv = null;
  let pySrv = null;
  try {
    const epDir = path.join(dir, 'src', 'episodes', 'reg');
    fs.mkdirSync(epDir, { recursive: true });
    fs.writeFileSync(
      path.join(epDir, 'episode.json'),
      JSON.stringify({ id: 'reg', slug: 'reg', lang: 'fr', aspect: '9:16', script: [{ text: 'Test' }], shots: [{ id: 'S1', size: 'medium' }] }),
    );
    fs.writeFileSync(
      path.join(epDir, 'timeline.json'),
      JSON.stringify({ fps: 30, total: 90, shots: [{ id: 'S1', len: 90 }], segments: [{ shot: 'S1', a: 0, b: 90 }] }),
    );
    const ctx = { projectId: 'reg', projectRoot: dir, dataRoot: dir, log: () => {} };
    const bibleTool = fsTools.find((t) => t.name === 'write_bible');
    assert.equal(
      (
        await bibleTool.run(
          {
            episode: 'reg',
            stylePrompt: 'Minimalist 2D What If explainer cartoon animation, round heads, bold ink lines',
            characters: [{ id: 'BOY', name: 'Boy', appearance: 'Young kid wearing bright red 1950s pajamas with cream piping', refPrompt: 'Full-body neutral standing pose' }],
            places: [{ id: 'BEDROOM', name: 'Bedroom', description: 'Dim 1955 kid bedroom with pine nightstand screen-left and lamp', refPrompt: 'Empty architectural background plate' }],
          },
          ctx,
        )
      ).ok,
      true,
    );

    // Validate edit_shot accept Codex pattern
    const codexCallArgs = {
      episode: 'reg',
      id: 'S1',
      shots: [],
      shot: {
        place: 'BEDROOM',
        characters: ['BOY'],
        size: 'medium',
        angle: 'level',
        view: 'front',
        cameraMove: 'push',
        imagePrompt: 'Medium shot in dim bedroom with BOY in red 1950s pajamas on screen-left reaching for rotary phone.',
        onScreenText: { kind: 'document', text: '', lang: 'fr' },
        videoPrompt: 'Boy lifts handset carefully toward ear while room lamp glows softly, gentle push in.',
        sfxHint: 'plastic handset creak, faint static buzz, quiet room ambience',
        seed: null,
      },
    };
    assert.equal(editShotTool.validate(codexCallArgs), null);
    const edRes = await editShotTool.run(codexCallArgs, ctx);
    assert.equal(edRes.ok, true, edRes.summary);
    const savedS1 = JSON.parse(fs.readFileSync(path.join(epDir, 'shots', 'S1.json'), 'utf8'));
    assert.equal(savedS1.onScreenText, null);
    assert.equal(savedS1.seed, undefined);
    assert.deepEqual(validateShotSpec(savedS1, 'S1').errors, []);

    // 3. Plural-clone and left/right spatial mismatch guards in stagingPromptProblems
    const cards = [
      {
        id: 'S1',
        size: 'medium',
        stage: {
          see: 'Le garçon se tient à gauche près de la fenêtre.',
          people: [{ who: 'BOY', where: 'left, near window', does: 'looks outside nervously' }],
          objects: [],
        },
      },
    ];
    const badSpecs = new Map([
      [
        'S1',
        validShot('S1', {
          place: 'BEDROOM',
          characters: ['BOY'],
          imagePrompt: 'Two boys stand together on the right side of the room staring out.',
        }),
      ],
    ]);
    const probs = stagingPromptProblems(cards, badSpecs, null, true, '16:9').join('\n');
    assert.match(probs, /face cloning|Two boys/i);
    assert.match(probs, /opposite side/i);

    // 4. Python image_gen.py --mode refs combines BOTH appearance+refPrompt and description+refPrompt
    if (HAS_NUMPY_PIL) {
      const samplePng = path.join(dir, 'sample.png');
      execFileSync('python3', ['-c', `from PIL import Image; Image.new("RGB", (64, 64), (50, 90, 140)).save(${JSON.stringify(samplePng)})`]);
      const pngBytes = fs.readFileSync(samplePng);
      const capturedPrompts = [];
      let pSeq = 0;
      pySrv = http.createServer(async (req, res) => {
        const u = new URL(req.url || '/', 'http://127.0.0.1');
        if (req.method === 'GET' && u.pathname.startsWith('/v1/files/')) {
          res.writeHead(200, { 'content-type': 'image/png' });
          res.end(pngBytes);
          return;
        }
        if (req.method === 'GET' && u.pathname.startsWith('/v1/jobs/')) {
          const jid = u.pathname.split('/').pop();
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({
            job_id: jid, status: 'succeeded', task_type: 'image', created_at: 1, completed_at: 2,
            data: [{ project_id: 'p1', url: `/v1/files/${jid}.png`, media_name: `00000000-0000-4000-8000-${String(jid).replace(/\D/g, '').padStart(12, '0')}`, media_type: 'image' }],
            error: null,
          }));
          return;
        }
        let bodyRaw = '';
        for await (const c of req) bodyRaw += c;
        const body = bodyRaw ? JSON.parse(bodyRaw) : {};
        res.writeHead(200, { 'content-type': 'application/json' });
        if (u.pathname === '/v1/projects') {
          res.end(JSON.stringify({ project_id: 'p1', is_default: false }));
          return;
        }
        if (u.pathname === '/v1/images/generations') {
          pSeq += 1;
          capturedPrompts.push(String(body.prompt || ''));
          res.end(JSON.stringify({ job_id: `img_${pSeq}`, status: 'pending' }));
          return;
        }
        res.end('{}');
      });
      await new Promise((r) => pySrv.listen(0, '127.0.0.1', r));
      const pPort = pySrv.address().port;
      await execFileAsync('python3', ['pipeline/image_gen.py', 'reg', '--mode', 'refs', '--project', dir], {
        env: {
          ...process.env,
          NODE_TEST_CONTEXT: '1',
          WHATIF_MEDIA_CONFIG_JSON: JSON.stringify({
            image: { driver: 'gflow', baseURL: `http://127.0.0.1:${pPort}`, model: 'nano2-lite', fallbackModel: 'nano2.1', pollInterval: 0.05 },
            video: { driver: 'gflow', baseURL: `http://127.0.0.1:${pPort}`, model: 'omni-1.2-flash', resolution: '360p', pollInterval: 0.05 },
          }),
        },
      });
      const charPrompt = capturedPrompts.find((p) => p.includes('CHARACTER MODEL SHEET'));
      const placePrompt = capturedPrompts.find((p) => p.includes('REFERENCE ENVIRONMENT'));
      assert.ok(charPrompt && charPrompt.includes('bright red 1950s pajamas') && charPrompt.includes('Full-body neutral standing pose'));
      assert.ok(placePrompt && placePrompt.includes('pine nightstand screen-left') && placePrompt.includes('Empty architectural background plate'));
    }

    // 5. Codex mid-stream socket drop retry & AGY safety filter fallback retry
    let codexHits = 0;
    codexSrv = http.createServer((_req, res) => {
      codexHits += 1;
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      if (codexHits === 1) {
        res.write('data: {"type":"response.output_text.delta","delta":"hello "}\n\n');
        setTimeout(() => res.socket?.destroy(), 25);
        return;
      }
      res.end(
        'data: {"type":"response.output_text.delta","delta":"hello world"}\n\n' +
        'data: {"type":"response.completed","response":{"status":"completed"}}\n\n' +
        'data: [DONE]\n\n',
      );
    });
    await new Promise((r) => codexSrv.listen(0, '127.0.0.1', r));
    const cPort = codexSrv.address().port;
    const authFile = path.join(dir, 'codex-auth.json');
    fs.writeFileSync(authFile, JSON.stringify({ tokens: { access_token: 'tok-test', account_id: 'acc-test' } }));
    const cProv = codexProvider({ baseURL: `http://127.0.0.1:${cPort}`, model: 'gpt-6.1-terra', authPath: authFile });
    const cOut = await cProv.chat({ messages: [{ role: 'user', content: 'hi' }], tools: [] });
    assert.equal(codexHits, 2);
    assert.equal(cOut.text, 'hello world');

    // Fake agy CLI script testing safety-filter auto-retry with framing + fallback model
    const fakeAgy = path.join(dir, 'fake-agy');
    const agyLog = path.join(dir, 'agy-calls.txt');
    fs.writeFileSync(
      fakeAgy,
      `#!/usr/bin/env bash
echo "$@" >> ${JSON.stringify(agyLog)}
if echo "$@" | grep -q "IMPORTANT STUDIO SAFETY FRAMING"; then
  echo '{"event":"result","result":{"status":"OK","response":"recovered cleanly"}}'
else
  echo '{"event":"result","result":{"status":"ERROR","error":"Your previous response was blocked by content safety filters."}}'
fi
`,
      { mode: 0o755 },
    );
    const aProv = agyProvider({ model: 'gemini-4-hydrogen-medium', agyBin: fakeAgy });
    const aOut = await aProv.chat({ messages: [{ role: 'user', content: 'scene' }], tools: [] });
    assert.equal(aOut.text, 'recovered cleanly');
  } finally {
    if (codexSrv) await new Promise((r) => codexSrv.close(r));
    if (pySrv) await new Promise((r) => pySrv.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
