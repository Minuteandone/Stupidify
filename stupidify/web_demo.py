from __future__ import annotations

import gc
import os

from .config import DamageConfig
from .core import stupidify_live_model

DEFAULT_MODEL = os.environ.get("STUPIDIFY_DEMO_MODEL", "openai-community/gpt2-xl")


def _require_web_deps():
    try:
        import gradio as gr
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the web extras first: pip install 'stupidify[web]'") from exc
    return gr, AutoModelForCausalLM, AutoTokenizer


def build_demo():
    gr, AutoModelForCausalLM, AutoTokenizer = _require_web_deps()

    def run_demo(prompt: str, method: str, severity: float, bits: int, seed: int, max_new_tokens: int):
        notes: list[str] = [f"Loading {DEFAULT_MODEL} once…"]
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
        inputs = tokenizer(prompt, return_tensors="pt")
        generator = dict(max_new_tokens=int(max_new_tokens), do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tokenizer.eos_token_id)
        original_out = model.generate(**inputs, **generator)
        original_text = tokenizer.decode(original_out[0], skip_special_tokens=True)
        notes.append("Stupidifying the loaded model in memory…")
        cfg = DamageConfig(method=method, severity=float(severity), bits=int(bits), seed=int(seed))
        stats = stupidify_live_model(model, cfg, progress=notes.append)
        damaged_out = model.generate(**inputs, **generator)
        damaged_text = tokenizer.decode(damaged_out[0], skip_special_tokens=True)
        notes.append(f"Touched {stats.tensors_changed} tensors. Reloading is the reset button.")
        del model
        gc.collect()
        return original_text, damaged_text, "\n".join(notes)

    with gr.Blocks(title="Stupidify GPT-2 XL") as demo:
        gr.Markdown("# Stupidify 🧠🔨\nCompare normal **GPT-2 XL** with the same loaded model after deliberate weight damage — no training involved. The model is mutated in memory, so the demo doesn't need a second checkpoint copy.")
        prompt = gr.Textbox(label="Prompt", value="The weirdest thing about computers is")
        with gr.Row():
            method = gr.Dropdown(["goblin", "zero", "noise", "signflip", "bitcrush", "shuffle"], value="goblin", label="Damage method")
            severity = gr.Slider(0, 0.75, value=0.10, step=0.01, label="Severity")
            bits = gr.Slider(2, 8, value=4, step=1, label="Bit depth")
            seed = gr.Number(value=42, precision=0, label="Seed")
            max_tokens = gr.Slider(8, 128, value=48, step=1, label="New tokens")
        run = gr.Button("Make GPT-2 XL dumber", variant="primary")
        with gr.Row():
            normal = gr.Textbox(label="Original GPT-2 XL", lines=12)
            stupid = gr.Textbox(label="Stupidified GPT-2 XL", lines=12)
        log = gr.Textbox(label="What happened", lines=8)
        run.click(run_demo, [prompt, method, severity, bits, seed, max_tokens], [normal, stupid, log])
    return demo


def main() -> None:
    demo = build_demo()
    host = os.environ.get("STUPIDIFY_HOST", "127.0.0.1")
    port = int(os.environ.get("STUPIDIFY_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
