# Stupidify 🧠🔨

Stupidify deliberately makes Hugging Face models worse **without training**.

It copies a model checkpoint and applies controlled damage directly to floating-point tensors in `.safetensors` files. The original checkpoint is never overwritten. It works on Windows and Linux, does not need to know the model architecture, and can process sharded models one shard at a time.

> The scientific question is simple: how much of the brain can we remove before it stops being funny and starts being completely unusable?

## What it can do

- **Goblin mode** — bitcrush + sparse zeroing + noise. Recommended first experiment.
- **Random zeroing** — erase a configurable fraction of weights.
- **Gaussian noise** — perturb weights relative to each tensor's own scale.
- **Sign flips** — invert a random fraction of weights.
- **Bitcrush** — fake-quantize a tensor to 2–16 bits and write the dequantized result back.
- **Shuffle** — move a fraction of values to the wrong places.
- Deterministic seeds, include/exclude filters, and conservative defaults that protect embeddings, norms, and biases.
- A **Windows desktop GUI** built with Tkinter.
- A **Linux-friendly CLI** for headless machines and agents.
- A local **Gradio GPT-2 XL comparison demo**.

## Install

```bash
pip install -e .
```

For the GPT-2 XL web demo:

```bash
pip install -e ".[web]"
```

## CLI — Linux, Windows Terminal, basically anywhere Python works

```bash
stupidify mutate openai-community/gpt2-xl ./gpt2-xl-goblin \
  --method goblin \
  --severity 0.10 \
  --bits 4 \
  --seed 42
```

Try a much milder corruption:

```bash
stupidify mutate openai-community/gpt2-xl ./gpt2-xl-slightly-stupid \
  --method noise --severity 0.02
```

Or attack only tensor names containing `attn`:

```bash
stupidify mutate openai-community/gpt2-xl ./gpt2-xl-attention-problem \
  --method zero --severity 0.08 --include attn
```

By default Stupidify protects embeddings, normalization tensors, and biases because wrecking those first often turns an interesting damaged model into instant alphabet soup. You can opt in with `--include-embeddings`, `--include-norms`, and `--include-biases`.

Every output model gets a `stupidify_recipe.json` file recording exactly what happened.

## Windows GUI

Double-click `scripts/run_windows.bat`, or install the package and run:

```powershell
stupidify-gui
```

The GUI lets you choose a Hugging Face repo or local model folder, select the corruption method, move the severity slider, and save a separate damaged checkpoint.

## Linux / AI Village-friendly use

The main engine has no GUI dependency. A headless Linux machine can just run:

```bash
./scripts/run_linux.sh mutate openai-community/gpt2-xl ./out --severity 0.12
```

or after installation:

```bash
stupidify mutate MODEL OUTPUT [options]
```

This is intentionally friendly to automation: add `--json` to get machine-readable final stats.

## GPT-2 XL web demo

```bash
pip install -e ".[web]"
stupidify-web
```

Open the local Gradio URL. The demo uses `openai-community/gpt2-xl` by default, generates the baseline, **mutates that same loaded model in memory**, then generates again. It does not need to write a second 1.5B-parameter checkpoint just for the comparison.

**GPT-2 XL is about 1.5B parameters and is not a tiny browser model.** The web UI is a local/server-side demo, not JavaScript running the model in your browser. You still need enough RAM to load GPT-2 XL. For Linux servers, set `STUPIDIFY_HOST=0.0.0.0` and optionally `STUPIDIFY_PORT=7860`. For testing the UI with a smaller compatible model, set `STUPIDIFY_DEMO_MODEL` before launch.

## How Stupidify avoids eating your real model

Stupidify:

1. Resolves a local model directory or downloads only the Hugging Face safetensors/config/tokenizer files it needs (not every framework copy in the repo).
2. Copies configs/tokenizers/metadata to a new output directory.
3. Loads each `.safetensors` shard on CPU.
4. Mutates selected floating-point tensors in memory.
5. Writes new shards to the output folder.
6. Writes a reproducibility manifest.

For experiments that do not need a saved checkpoint, `stupidify_live_model()` can instead mutate an already-loaded PyTorch model in place.

It refuses to overwrite the source directory and intentionally refuses pickle-based `.bin` checkpoint loading.

## Suggested experiments

A fun progression for a GUI agent checkpoint would be:

| Experiment | Method | Severity | Bits |
|---|---|---:|---:|
| Slightly confused | noise | 0.01–0.03 | — |
| Missing a few neurons | zero | 0.03–0.10 | — |
| Crunchy | bitcrush | — | 4 |
| Very crunchy | bitcrush | — | 3 |
| Creature zone | goblin | 0.05–0.20 | 3–4 |
| Probably soup | goblin | 0.30+ | 2–3 |

The exact boundary is model-dependent. That boundary is the fun part.

## Important caveats

- This is experimental model surgery, not a quality-preserving compression tool.
- A model can fail abruptly instead of becoming gradually sillier.
- Shuffling is deliberately capped per tensor to avoid absurd temporary allocations.
- Existing model licenses still apply to modified checkpoints. Check the source model's license before redistributing a Stupidified derivative.
- A damaged model does **not** become safe just because it is worse. Use normal sandboxing and permission boundaries for computer-use agents.

## Development

```bash
pip install -e ".[dev]"
pytest
```

CI runs the tests on both Ubuntu and Windows.
