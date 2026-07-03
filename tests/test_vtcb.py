"""Tests for Phase 7: VTCBRunner and MedVLM.rebuild_projector_at_budget."""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml
import pytest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ── Minimal mock dataset ──────────────────────────────────────────────────────

class _MockDataset:
    """Tiny dataset of (vol, report, label_dict, patient_id) tuples."""

    def __init__(self, n: int = 4, D: int = 8, H: int = 224, W: int = 224):
        self.n = n
        self.D = D
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            vol = torch.randn(self.D, self.H, self.W)
            report = f"No acute findings. Study {i}."
            label_dict = {f"label_{j:02d}": float((i + j) % 2) for j in range(18)}
            patient_id = f"patient_{i:04d}"
            yield (vol, report, label_dict, patient_id)


# ── Module-scoped fixtures (load opt-125m once for the whole module) ──────────

@pytest.fixture(scope="module")
def small_model():
    _skip_if_no_cuda()
    from aadp.models.vlm import MedVLM

    return MedVLM(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
        llm_model_name="facebook/opt-125m",
        num_latents=8,
        num_tokens=16,
        use_film=True,
        max_depth=64,
        device=DEVICE,
    )


@pytest.fixture(scope="module")
def mock_dataset():
    return _MockDataset(n=4, D=8, H=224, W=224)


# ── MedVLM.rebuild_projector_at_budget ───────────────────────────────────────

class TestRebuildProjectorAtBudget:

    def test_changes_num_tokens(self, small_model):
        _skip_if_no_cuda()
        small_model.rebuild_projector_at_budget(8)
        assert small_model.projector.num_tokens == 8
        assert small_model._num_tokens == 8

    def test_stage1_weights_preserved(self, small_model):
        _skip_if_no_cuda()
        old_latents = small_model.projector.stage1.latents.data.clone()
        small_model.rebuild_projector_at_budget(32)
        torch.testing.assert_close(
            small_model.projector.stage1.latents.data, old_latents
        )
        small_model.rebuild_projector_at_budget(16)   # restore for later tests

    def test_forward_works_after_rebuild(self, small_model):
        _skip_if_no_cuda()
        small_model.rebuild_projector_at_budget(16)
        vols = torch.randn(2, 8, 224, 224, device=DEVICE)
        with torch.no_grad():
            out = small_model(vols, ["Describe this CT."] * 2,
                              report_tokens=None, max_new_tokens=4)
        assert out.shape[0] == 2   # (B, generated_length)

    def test_unsupported_projector_raises(self):
        _skip_if_no_cuda()
        from aadp.models.vlm import MedVLM
        import torch.nn as nn

        class _DummyProjector(nn.Module):
            num_tokens = 64
            def forward(self, *a, **k):
                return torch.zeros(1, 64, 192)

        from aadp.models.vlm import MedVLM
        model_dummy = MedVLM(
            vit_model_name="vit_tiny_patch16_224",
            vit_pretrained=False,
            llm_model_name="facebook/opt-125m",
            num_latents=8,
            num_tokens=16,
            device=DEVICE,
        )
        model_dummy.projector = _DummyProjector()
        with pytest.raises(NotImplementedError):
            model_dummy.rebuild_projector_at_budget(8)


# ── VTCBRunner ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vtcb_runner(small_model, mock_dataset, tmp_path_factory):
    from aadp.evaluation.benchmarks.vtcb import VTCBRunner

    results_dir = str(tmp_path_factory.mktemp("vtcb_results"))
    return VTCBRunner(
        model=small_model,
        val_dataset=mock_dataset,
        radgenome_dataset=None,
        totalseg_dataset=None,
        token_budgets=[16],
        primary_budget=16,
        batch_size=2,
        max_samples=4,
        max_new_tokens=8,       # keep test fast
        device=str(DEVICE),
        results_dir=results_dir,
    )


@pytest.fixture(scope="module")
def vtcb_results(vtcb_runner):
    return vtcb_runner.run(model_name="test_model")


class TestVTCBRunner:

    def test_run_returns_dict_with_budget_key(self, vtcb_results):
        _skip_if_no_cuda()
        assert 16 in vtcb_results

    def test_run_contains_report_generation(self, vtcb_results):
        _skip_if_no_cuda()
        assert "report_generation" in vtcb_results[16]

    def test_run_skips_t3_t4_when_none(self, vtcb_results):
        _skip_if_no_cuda()
        keys = set(vtcb_results[16].keys())
        assert "lesion_recall" not in keys
        assert "anatomical_localisation" not in keys

    def test_json_written_to_disk(self, vtcb_runner, vtcb_results):
        _skip_if_no_cuda()
        json_path = os.path.join(vtcb_runner.results_dir, "test_model_vtcb.json")
        assert os.path.exists(json_path), "VTCBRunner should write a JSON file"
        with open(json_path) as fh:
            data = json.load(fh)
        assert "results" in data
        assert "16" in data["results"]

    def test_compare_identical_jsons_equal(self, vtcb_runner, vtcb_results):
        _skip_if_no_cuda()
        from aadp.evaluation.benchmarks.vtcb import VTCBRunner

        json_path = os.path.join(vtcb_runner.results_dir, "test_model_vtcb.json")
        comparison = VTCBRunner.compare({"modelA": json_path, "modelB": json_path})

        for metric, values in comparison.items():
            if "modelA" in values and "modelB" in values:
                vA, vB = values["modelA"], values["modelB"]
                # Skip NaN (NaN != NaN in IEEE 754)
                if not (isinstance(vA, float) and math.isnan(vA)):
                    assert vA == vB, f"Mismatch on {metric}: {vA} != {vB}"

    def test_plot_compression_curves_saves_png(self, vtcb_runner, vtcb_results, tmp_path_factory):
        _skip_if_no_cuda()
        from aadp.evaluation.benchmarks.vtcb import VTCBRunner

        json_path = os.path.join(vtcb_runner.results_dir, "test_model_vtcb.json")
        plot_dir = str(tmp_path_factory.mktemp("vtcb_plots"))
        VTCBRunner.plot_compression_curves(
            {"test_model": json_path},
            metric_names=["radgraph_f1"],
            save_dir=plot_dir,
        )
        assert os.path.exists(os.path.join(plot_dir, "radgraph_f1.png"))

    def test_report_parameter_counts_positive(self, small_model):
        _skip_if_no_cuda()
        from aadp.evaluation.benchmarks.vtcb import VTCBRunner
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedAADP

        projectors = {
            "aadp": small_model.projector,
            "task_cond": TaskConditionedAADP(
                embed_dim=192, num_latents=8, num_tokens=16, device=DEVICE
            ),
        }
        counts = VTCBRunner.report_parameter_counts(projectors)
        for name, c in counts.items():
            assert c["total"] > 0, f"Expected positive total params for {name}"


# ── Smoke test: scripts/evaluate.py ──────────────────────────────────────────

class TestEvaluateScript:

    def test_smoke_evaluate_script(self, small_model, tmp_path_factory):
        """Smoke test: evaluate.py --config ... --checkpoint ... exits 0."""
        _skip_if_no_cuda()

        # Write a minimal projector checkpoint
        ckpt_dir = str(tmp_path_factory.mktemp("ckpt"))
        ckpt_path = os.path.join(ckpt_dir, "smoke_ckpt.pt")
        torch.save({"projector": small_model.projector.state_dict()}, ckpt_path)

        # Write a minimal config
        cfg_dir = str(tmp_path_factory.mktemp("cfg"))
        cfg_path = os.path.join(cfg_dir, "smoke_config.yaml")
        config = {
            "vit_model_name": "vit_tiny_patch16_224",
            "llm_model_name": "facebook/opt-125m",
            "num_latents": 8,
            "num_tokens": 16,
            "max_depth": 64,
            "experiment_name": "smoke_test",
        }
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(config, fh)

        results_dir = str(tmp_path_factory.mktemp("results"))

        import os as _os
        env = _os.environ.copy()
        project_root = str(Path(__file__).parent.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (project_root + _os.pathsep + existing) if existing else project_root

        proc = subprocess.run(
            [
                sys.executable, "scripts/evaluate.py",
                "--config", cfg_path,
                "--checkpoint", ckpt_path,
                "--budgets", "16",
                "--max_samples", "0",    # empty dataset → instant
                "--batch_size", "2",
                "--max_new_tokens", "4",
                "--results_dir", results_dir,
                "--device", str(DEVICE),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        assert proc.returncode == 0, (
            f"evaluate.py exited with code {proc.returncode}\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
