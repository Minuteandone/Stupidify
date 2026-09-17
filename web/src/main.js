import { pipeline, LogitsProcessor } from '@huggingface/transformers';
import './style.css';

const $ = (id) => document.getElementById(id);

const els = {
  model: $('model'), runtime: $('runtime'), method: $('method'), severity: $('severity'), bits: $('bits'),
  seed: $('seed'), tokens: $('tokens'), prompt: $('prompt'), run: $('run'), unload: $('unload'),
  baseline: $('baseline'), damaged: $('damaged'), status: $('status'), statusDot: $('statusDot'),
  progress: $('progress'), progressText: $('progressText'), severityValue: $('severityValue'),
  bitsValue: $('bitsValue'), recipeLabel: $('recipeLabel'), modelWarning: $('modelWarning'),
  randomPrompt: $('randomPrompt'),
};

const prompts = [
  'Once upon a time, a very small computer tried to use Windows and',
  'The robot opened the wrong program for the fifth time, so it decided to',
  'A scientist removed exactly twelve percent of a language model and discovered',
  'The strangest thing about the computer goblin was that it could always',
  'In the year 2042, the least qualified AI in the building was asked to',
];

let generator = null;
let loadedKey = null;

function hash01(index, step, seed) {
  let x = (index ^ Math.imul(step + 1, 0x9e3779b1) ^ seed) >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

class StupidifyLogitsProcessor extends LogitsProcessor {
  constructor({ method, severity, bits, seed }) {
    super();
    this.method = method;
    this.severity = severity;
    this.bits = bits;
    this.seed = seed >>> 0;
    this.step = 0;
  }

  _call(_inputIds, logits) {
    const data = logits.data;
    const sev = this.severity;
    const step = this.step++;
    if (!data?.length || sev <= 0) return logits;

    let sampledAbs = 0;
    let samples = 0;
    const stride = Math.max(1, Math.floor(data.length / 512));
    for (let i = 0; i < data.length; i += stride) {
      const v = Number(data[i]);
      if (Number.isFinite(v)) { sampledAbs += Math.abs(v); samples += 1; }
    }
    const scale = samples ? sampledAbs / samples : 1;

    const zeroPass = (amount) => {
      for (let i = 0; i < data.length; i += 1) {
        if (hash01(i, step, this.seed) < amount) data[i] = -Infinity;
      }
    };

    const noisePass = (amount) => {
      const magnitude = Math.max(scale, 0.25) * amount * 5;
      for (let i = 0; i < data.length; i += 1) {
        if (!Number.isFinite(data[i])) continue;
        const n = (hash01(i, step + 17, this.seed ^ 0xa5a5a5a5) * 2) - 1;
        data[i] += n * magnitude;
      }
    };

    const signFlipPass = (amount) => {
      for (let i = 0; i < data.length; i += 1) {
        if (Number.isFinite(data[i]) && hash01(i, step + 31, this.seed) < amount) data[i] = -data[i];
      }
    };

    const bitcrushPass = (bits) => {
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < data.length; i += stride) {
        const v = Number(data[i]);
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return;
      const levels = (2 ** bits) - 1;
      const width = max - min;
      for (let i = 0; i < data.length; i += 1) {
        const v = Number(data[i]);
        if (!Number.isFinite(v)) continue;
        data[i] = min + (Math.round(((v - min) / width) * levels) / levels) * width;
      }
    };

    if (this.method === 'zero') zeroPass(sev);
    else if (this.method === 'noise') noisePass(sev);
    else if (this.method === 'signflip') signFlipPass(sev);
    else if (this.method === 'bitcrush') bitcrushPass(this.bits);
    else {
      bitcrushPass(this.bits);
      zeroPass(Math.min(0.45, sev * 0.55));
      noisePass(sev * 0.7);
      signFlipPass(Math.min(0.15, sev * 0.15));
    }
    return logits;
  }
}

function seededRandom(seed) {
  let x = (seed >>> 0) || 1;
  return () => {
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

async function withSeed(seed, fn) {
  const original = Math.random;
  Math.random = seededRandom(seed);
  try { return await fn(); }
  finally { Math.random = original; }
}

function runtimeOptions() {
  const runtime = els.runtime.value;
  if (runtime === 'webgpu') return { device: 'webgpu' };
  if (runtime === 'auto' && 'gpu' in navigator) return { device: 'webgpu' };
  return {};
}

function dtypeFor(model) {
  return model.includes('gpt2-xl') ? 'q4' : 'q8';
}

function setStatus(label, kind = '', detail = '', percent = 0) {
  els.status.textContent = label;
  els.statusDot.className = `status-dot ${kind}`;
  els.progressText.textContent = detail;
  els.progress.value = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0;
}

function updateRecipe() {
  els.severityValue.value = Number(els.severity.value).toFixed(2);
  els.bitsValue.value = els.bits.value;
  els.recipeLabel.textContent = `${els.method.value} · ${Number(els.severity.value).toFixed(2)} · ${els.bits.value}-bit`;
}

function updateWarning() {
  const xl = els.model.value.includes('gpt2-xl');
  els.modelWarning.classList.toggle('hidden', !xl);
  els.modelWarning.textContent = xl
    ? 'GPT-2 XL is enormous for a browser. Expect a very large download, high memory use, and possible failure on phones/tablets. DistilGPT-2 is the recommended Pages demo.'
    : '';
}

async function ensureModel() {
  const model = els.model.value;
  const runtime = els.runtime.value;
  const key = `${model}|${runtime}`;
  if (generator && loadedKey === key) return generator;

  if (generator) await unloadModel();
  setStatus('Loading model', 'busy', 'Starting download…', 2);

  const options = {
    ...runtimeOptions(),
    dtype: dtypeFor(model),
    progress_callback: (event) => {
      const p = Number(event?.progress ?? 0);
      const pct = p <= 1 ? p * 100 : p;
      const name = event?.file ?? event?.name ?? event?.status ?? 'model files';
      setStatus('Loading model', 'busy', String(name), Number.isFinite(pct) ? pct : 8);
    },
  };

  generator = await pipeline('text-generation', model, options);
  loadedKey = key;
  els.unload.disabled = false;
  setStatus('Ready', 'good', `${model} loaded with ${runtime === 'auto' ? ('gpu' in navigator ? 'WebGPU' : 'WASM') : runtime}.`, 100);
  return generator;
}

async function unloadModel() {
  if (generator?.dispose) await generator.dispose();
  generator = null;
  loadedKey = null;
  els.unload.disabled = true;
  setStatus('Idle', '', 'Model unloaded.', 0);
}

async function generate(pipe, prompt, options, seed) {
  return withSeed(seed, async () => {
    const result = await pipe(prompt, options);
    return result?.[0]?.generated_text ?? String(result);
  });
}

async function runExperiment() {
  els.run.disabled = true;
  els.baseline.textContent = 'Loading / generating…';
  els.damaged.textContent = 'Waiting to be made worse…';

  try {
    const pipe = await ensureModel();
    const prompt = els.prompt.value || 'The computer said';
    const seed = Number(els.seed.value) || 42;
    const maxNewTokens = Math.max(1, Math.min(128, Number(els.tokens.value) || 48));
    const shared = {
      max_new_tokens: maxNewTokens,
      do_sample: true,
      temperature: 0.9,
      top_p: 0.95,
      repetition_penalty: 1.05,
    };

    setStatus('Generating baseline', 'busy', 'Normal model output…', 100);
    els.baseline.textContent = await generate(pipe, prompt, shared, seed);

    const damage = new StupidifyLogitsProcessor({
      method: els.method.value,
      severity: Number(els.severity.value),
      bits: Number(els.bits.value),
      seed,
    });

    setStatus('Stupidifying', 'busy', 'Generating again with damaged logits…', 100);
    els.damaged.textContent = await generate(pipe, prompt, { ...shared, logits_processor: [damage] }, seed);
    setStatus('Done', 'good', 'Same prompt and sampling seed; second run has deterministic logit damage.', 100);
  } catch (error) {
    console.error(error);
    setStatus('Experiment failed', 'bad', error?.message ?? String(error), 0);
    els.damaged.textContent = `Error: ${error?.message ?? error}`;
  } finally {
    els.run.disabled = false;
  }
}

els.run.addEventListener('click', runExperiment);
els.unload.addEventListener('click', unloadModel);
els.severity.addEventListener('input', updateRecipe);
els.bits.addEventListener('input', updateRecipe);
els.method.addEventListener('change', updateRecipe);
els.model.addEventListener('change', () => { updateWarning(); setStatus('Idle', '', 'Model selection changed; it will load on the next run.', 0); });
els.runtime.addEventListener('change', () => setStatus('Idle', '', 'Runtime changed; the model will reload on the next run.', 0));
els.randomPrompt.addEventListener('click', () => {
  els.prompt.value = prompts[Math.floor(Math.random() * prompts.length)];
});

updateRecipe();
updateWarning();
