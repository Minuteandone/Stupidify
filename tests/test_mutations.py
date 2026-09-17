from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from stupidify.config import DamageConfig
from stupidify.core import stupidify_live_model, stupidify_model
from stupidify.mutations import mutate_tensor


def test_zero_is_deterministic():
    a = torch.ones(10_000)
    b = torch.ones(10_000)
    cfg = DamageConfig(method="zero", severity=0.2, seed=123, min_ndim=1, min_numel=1)
    mutate_tensor("layer.weight", a, cfg)
    mutate_tensor("layer.weight", b, cfg)
    assert torch.equal(a, b)
    assert 1500 < int((a == 0).sum()) < 2500


def test_signflip_actually_changes_source():
    tensor = torch.ones(1000)
    cfg = DamageConfig(method="signflip", severity=1.0, seed=3, min_ndim=1, min_numel=1)
    mutate_tensor("layer.weight", tensor, cfg)
    assert torch.all(tensor == -1)


def test_bitcrush_changes_values():
    tensor = torch.linspace(-1, 1, 1000)
    before = tensor.clone()
    cfg = DamageConfig(method="bitcrush", bits=3, min_ndim=1, min_numel=1)
    mutate_tensor("layer.weight", tensor, cfg)
    assert not torch.equal(before, tensor)
    assert torch.unique(tensor).numel() <= 7


def test_live_model_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False))
    cfg = DamageConfig(method="zero", severity=1.0, seed=1)
    stats = stupidify_live_model(model, cfg)
    assert torch.count_nonzero(model[0].weight) == 0
    assert stats.tensors_changed == 1


def test_local_safetensors_repo_roundtrip(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"fake"}', encoding="utf-8")
    save_file({"transformer.h.0.attn.weight": torch.ones(32, 32), "transformer.wte.weight": torch.ones(32, 32)}, str(source / "model.safetensors"), metadata={"format": "pt"})
    cfg = DamageConfig(method="zero", severity=1.0, seed=1)
    stats = stupidify_model(str(source), str(output), cfg, progress=None)
    result = load_file(str(output / "model.safetensors"))
    assert torch.count_nonzero(result["transformer.h.0.attn.weight"]) == 0
    assert torch.all(result["transformer.wte.weight"] == 1)
    assert (output / "config.json").exists()
    manifest = json.loads((output / "stupidify_recipe.json").read_text(encoding="utf-8"))
    assert manifest["config"]["method"] == "zero"
    assert stats.tensors_changed == 1


def test_refuses_nonempty_output(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (output / "mine.txt").write_text("keep me", encoding="utf-8")
    save_file({"layer.weight": torch.ones(16, 16)}, str(source / "model.safetensors"))
    cfg = DamageConfig(method="zero", severity=0.1)
    try:
        stupidify_model(str(source), str(output), cfg, progress=None)
    except ValueError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("expected a refusal to overwrite a non-empty output folder")
