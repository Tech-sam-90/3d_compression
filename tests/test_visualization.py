"""Tests for visualization/attention_maps.py and visualization/compression_curves.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ── visualize_attention ────────────────────────────────────────────────────────

class TestVisualizeAttention:
    def test_produces_png(self, tmp_path):
        from visualization.attention_maps import visualize_attention

        D = 64
        vol = torch.rand(D, 224, 224)
        attn = torch.rand(D)
        save_path = str(tmp_path / "attn.png")
        visualize_attention(vol, attn, "Describe any findings", save_path)
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 1000

    def test_gt_slice_indices_highlighted(self, tmp_path):
        from visualization.attention_maps import visualize_attention

        vol = torch.rand(32, 224, 224)
        attn = torch.rand(32)
        save_path = str(tmp_path / "attn_gt.png")
        visualize_attention(vol, attn, "Find nodules", save_path,
                            gt_slice_indices=[3, 7, 15])
        assert Path(save_path).exists()

    def test_gt_slice_indices_none_graceful(self, tmp_path):
        from visualization.attention_maps import visualize_attention

        vol = torch.rand(16, 224, 224)
        attn = torch.rand(16)
        save_path = str(tmp_path / "attn_none.png")
        visualize_attention(vol, attn, "Any findings?", save_path,
                            gt_slice_indices=None)
        assert Path(save_path).exists()

    def test_single_slice_volume(self, tmp_path):
        from visualization.attention_maps import visualize_attention

        vol = torch.rand(1, 224, 224)
        attn = torch.tensor([1.0])
        save_path = str(tmp_path / "single.png")
        # Must not raise on D=1
        visualize_attention(vol, attn, "Single slice test", save_path)
        assert Path(save_path).exists()

    def test_creates_nested_parent_dirs(self, tmp_path):
        from visualization.attention_maps import visualize_attention

        vol = torch.rand(8, 64, 64)
        attn = torch.rand(8)
        save_path = str(tmp_path / "nested" / "sub" / "attn.png")
        visualize_attention(vol, attn, "Test", save_path)
        assert Path(save_path).exists()

    def test_attn_on_gpu_tensor(self, tmp_path):
        _skip_if_no_cuda()
        from visualization.attention_maps import visualize_attention

        vol = torch.rand(16, 64, 64, device="cuda")
        attn = torch.rand(16, device="cuda")
        save_path = str(tmp_path / "gpu.png")
        # Should move to CPU internally without error
        visualize_attention(vol, attn, "GPU tensor test", save_path)
        assert Path(save_path).exists()


# ── compare_instructions ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_model():
    _skip_if_no_cuda()
    from aadp.models.vlm import MedVLM

    model = MedVLM(
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
        llm_model_name="facebook/opt-125m",
        llm_frozen=True,
        vit_frozen=True,
        num_latents=8,
        num_tokens=16,
        device=DEVICE,
    ).to(DEVICE)
    model.eval()
    return model


class TestCompareInstructions:
    def test_produces_png(self, small_model, tmp_path):
        _skip_if_no_cuda()
        from visualization.attention_maps import compare_instructions

        vol = torch.rand(16, 224, 224)
        save_path = str(tmp_path / "compare.png")
        compare_instructions(
            vol,
            ["Describe findings"],
            [(small_model, "A-ADP")],
            save_path,
        )
        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 1000

    def test_two_instructions_two_panels(self, small_model, tmp_path):
        _skip_if_no_cuda()
        from visualization.attention_maps import compare_instructions

        vol = torch.rand(8, 224, 224)
        save_path = str(tmp_path / "two_panel.png")
        compare_instructions(
            vol,
            ["Describe findings", "Are there nodules?"],
            [(small_model, "A-ADP")],
            save_path,
        )
        assert Path(save_path).exists()

    def test_single_slice_volume(self, small_model, tmp_path):
        _skip_if_no_cuda()
        from visualization.attention_maps import compare_instructions

        vol = torch.rand(1, 224, 224)
        save_path = str(tmp_path / "single_compare.png")
        compare_instructions(
            vol,
            ["Any finding?"],
            [(small_model, "A-ADP")],
            save_path,
        )
        assert Path(save_path).exists()

    def test_two_models_same_instruction(self, small_model, tmp_path):
        _skip_if_no_cuda()
        from visualization.attention_maps import compare_instructions

        vol = torch.rand(8, 224, 224)
        save_path = str(tmp_path / "two_models.png")
        compare_instructions(
            vol,
            ["Findings?"],
            [(small_model, "Model-A"), (small_model, "Model-B")],
            save_path,
        )
        assert Path(save_path).exists()


# ── plot_paper_figures ─────────────────────────────────────────────────────────

def _write_mock_vtcb(path: Path, model_name: str, budgets=None) -> None:
    if budgets is None:
        budgets = [16, 32, 64, 128]
    data = {
        "model_name": model_name,
        "primary_budget": 64,
        "token_budgets": budgets,
        "results": {
            str(M): {
                "report_generation": {
                    "radgraph_f1": round(0.1 * (M / 16), 4),
                    "ratescore_mean": round(0.2 + 0.01 * (M / 16), 4),
                },
                "abnormality_classification": {
                    "auroc_macro": round(0.5 + 0.05 * (M / 16), 4),
                    "f1_macro": round(0.3 + 0.04 * (M / 16), 4),
                },
            }
            for M in budgets
        },
    }
    path.write_text(json.dumps(data, indent=2))


class TestPlotPaperFigures:
    @pytest.fixture
    def results_dir(self, tmp_path):
        rd = tmp_path / "results"
        rd.mkdir()
        _write_mock_vtcb(rd / "aadp_vtcb.json", "A-ADP")
        _write_mock_vtcb(rd / "perceiver_vtcb.json", "Perceiver")
        return str(rd)

    def test_produces_png_and_pdf(self, results_dir, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        save_dir = str(tmp_path / "figs")
        plot_paper_figures(results_dir, save_dir)
        assert len(list(Path(save_dir).glob("*.png"))) >= 1
        assert len(list(Path(save_dir).glob("*.pdf"))) >= 1

    def test_summary_grid_produced(self, results_dir, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        save_dir = str(tmp_path / "figs2")
        plot_paper_figures(results_dir, save_dir)
        assert (Path(save_dir) / "summary_grid.png").exists()
        assert (Path(save_dir) / "summary_grid.pdf").exists()

    def test_per_metric_files_named_correctly(self, results_dir, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        save_dir = str(tmp_path / "figs3")
        plot_paper_figures(results_dir, save_dir)
        assert (Path(save_dir) / "radgraph_f1.png").exists()
        assert (Path(save_dir) / "radgraph_f1.pdf").exists()
        assert (Path(save_dir) / "auroc_macro.png").exists()

    def test_empty_results_dir_no_error(self, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        empty = str(tmp_path / "empty")
        os.makedirs(empty)
        save_dir = str(tmp_path / "figs_empty")
        # No JSON files → should not raise
        plot_paper_figures(empty, save_dir)

    def test_single_budget_json(self, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        rd = tmp_path / "single_b"
        rd.mkdir()
        _write_mock_vtcb(rd / "test_vtcb.json", "test", budgets=[32])
        save_dir = str(tmp_path / "figs_single")
        plot_paper_figures(str(rd), save_dir)
        assert (Path(save_dir) / "radgraph_f1.png").exists()

    def test_creates_save_dir(self, results_dir, tmp_path):
        from visualization.compression_curves import plot_paper_figures

        save_dir = str(tmp_path / "new" / "nested" / "dir")
        # Directory does not exist yet — must be created
        plot_paper_figures(results_dir, save_dir)
        assert Path(save_dir).is_dir()
