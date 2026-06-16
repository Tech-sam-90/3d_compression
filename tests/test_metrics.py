"""Tests for aadp/evaluation/metrics/*.

All tensor operations on CUDA.  Tests that require optional packages
(radgraph, ratescore / bert_score) are skipped when those packages are absent.
"""

import math
import pytest
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.1  RadGraph F1
# ═══════════════════════════════════════════════════════════════════════════════

def _radgraph_available():
    try:
        import radgraph  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _radgraph_available(), reason="radgraph not installed")
class TestRadGraphF1:
    def test_import_error_without_package(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "radgraph", None)
        from importlib import import_module
        import importlib
        import aadp.evaluation.metrics.radgraph_f1 as mod
        importlib.reload(mod)
        with pytest.raises(ImportError, match="radgraph not installed"):
            mod.RadGraphF1()

    def test_returns_expected_keys(self):
        from aadp.evaluation.metrics.radgraph_f1 import compute_radgraph_f1
        preds = ["No pneumonia detected.", "Mild pleural effusion."]
        refs  = ["No acute cardiopulmonary process.", "Small pleural effusion."]
        result = compute_radgraph_f1(preds, refs)
        assert set(result.keys()) == {"precision", "recall", "f1"}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_perfect_match(self):
        from aadp.evaluation.metrics.radgraph_f1 import compute_radgraph_f1
        text = ["No acute findings."]
        result = compute_radgraph_f1(text, text)
        assert result["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_length_mismatch_raises(self):
        from aadp.evaluation.metrics.radgraph_f1 import compute_radgraph_f1
        with pytest.raises(ValueError, match="same length"):
            compute_radgraph_f1(["a"], ["b", "c"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6.2  RaTEScore
# ═══════════════════════════════════════════════════════════════════════════════

def _any_ratescore_backend():
    for pkg in ("ratescore", "bert_score"):
        try:
            __import__(pkg)
            return True
        except ImportError:
            pass
    return False


class TestRaTEScoreDetect:
    def test_no_backend_raises(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ratescore", None)
        monkeypatch.setitem(sys.modules, "bert_score", None)
        from aadp.evaluation.metrics.ratescore import RaTEScore
        with pytest.raises(ImportError, match="ratescore"):
            RaTEScore()

    def test_length_mismatch_raises(self):
        if not _any_ratescore_backend():
            pytest.skip("no ratescore backend available")
        from aadp.evaluation.metrics.ratescore import RaTEScore
        rs = RaTEScore()
        with pytest.raises(ValueError, match="same length"):
            rs.compute(["a"], ["b", "c"])


@pytest.mark.skipif(not _any_ratescore_backend(), reason="no ratescore backend")
class TestRaTEScoreCompute:
    def test_keys_and_range(self):
        from aadp.evaluation.metrics.ratescore import compute_ratescore
        preds = ["No pneumonia.", "Pleural effusion."]
        refs  = ["No acute findings.", "Small pleural effusion."]
        result = compute_ratescore(preds, refs)
        assert "ratescore_mean" in result
        assert "ratescore_std" in result
        assert 0.0 <= result["ratescore_mean"] <= 1.0
        assert result["ratescore_std"] >= 0.0

    def test_identical_texts_high_score(self):
        from aadp.evaluation.metrics.ratescore import compute_ratescore
        text = ["Lungs are clear. No acute findings."]
        result = compute_ratescore(text, text)
        assert result["ratescore_mean"] > 0.8

    def test_std_zero_single_item(self):
        from aadp.evaluation.metrics.ratescore import compute_ratescore
        text = ["Single sentence."]
        result = compute_ratescore(text, text)
        assert result["ratescore_std"] == pytest.approx(0.0, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.3  AUROC / F1
# ═══════════════════════════════════════════════════════════════════════════════

class TestAurocF1:
    def test_keys_present(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        names = ["pneumonia", "effusion"]
        N = 20
        preds  = torch.rand(N, 2, device=DEVICE)
        labels = torch.randint(0, 2, (N, 2), device=DEVICE).float()
        # Ensure at least one positive per label
        labels[0, 0] = 1.0
        labels[1, 1] = 1.0
        result = compute_auroc_f1(preds, labels, names)
        for name in names:
            assert f"auroc_{name}" in result
            assert f"f1_{name}" in result
        assert "auroc_macro" in result
        assert "f1_macro" in result

    def test_perfect_classifier(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        N = 10
        labels = torch.zeros(N, 1, device=DEVICE)
        labels[:5] = 1.0
        preds = labels.clone()
        result = compute_auroc_f1(preds, labels, ["lung"])
        assert result["auroc_lung"] == pytest.approx(1.0)
        assert result["f1_lung"] == pytest.approx(1.0)

    def test_no_positive_examples_auroc_nan(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        N = 10
        labels = torch.zeros(N, 2, device=DEVICE)
        labels[:5, 0] = 1.0  # label "a": 5 pos / 5 neg → valid AUROC
        # label "b": all negative → NaN AUROC
        preds = torch.rand(N, 2, device=DEVICE)
        result = compute_auroc_f1(preds, labels, ["a", "b"])
        assert math.isnan(result["auroc_b"])
        # macro excludes NaN labels, so it should be finite (from label "a")
        assert not math.isnan(result["auroc_macro"])

    def test_length_mismatch_raises(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        preds = torch.rand(5, 3, device=DEVICE)
        labels = torch.zeros(5, 3, device=DEVICE)
        with pytest.raises(ValueError, match="label_names length"):
            compute_auroc_f1(preds, labels, ["a", "b"])

    def test_accepts_gpu_tensors(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import compute_auroc_f1
        N = 8
        labels = torch.zeros(N, 1, device=DEVICE)
        labels[:4] = 1.0
        preds = torch.rand(N, 1, device=DEVICE)
        result = compute_auroc_f1(preds, labels, ["cls"])
        assert isinstance(result["auroc_cls"], float)


class TestAbnormalityClassificationHead:
    def test_output_shape_and_range(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import AbnormalityClassificationHead
        head = AbnormalityClassificationHead(embed_dim=128, num_labels=18).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        out = head(x)
        assert out.shape == (4, 18)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_custom_num_labels(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.auroc_f1 import AbnormalityClassificationHead
        head = AbnormalityClassificationHead(embed_dim=64, num_labels=5).to(DEVICE)
        x = torch.randn(3, 64, device=DEVICE)
        out = head(x)
        assert out.shape == (3, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.4  Recall@K
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecallAtK:
    def test_perfect_recall(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k
        attn = torch.tensor([[0.9, 0.05, 0.05]], device=DEVICE)
        gt   = [[0]]
        result = compute_recall_at_k(attn, gt, k=1)
        assert result["recall_at_k"] == pytest.approx(1.0)
        assert result["n_samples"] == 1

    def test_zero_recall(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k
        attn = torch.tensor([[0.05, 0.9, 0.05]], device=DEVICE)
        gt   = [[2]]
        result = compute_recall_at_k(attn, gt, k=1)
        assert result["recall_at_k"] == pytest.approx(0.0)

    def test_empty_gt_skipped(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k
        attn = torch.rand(3, 10, device=DEVICE)
        gt   = [[2], [], [5]]
        result = compute_recall_at_k(attn, gt, k=3)
        assert result["n_samples"] == 2

    def test_k_clamped_to_d(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k
        attn = torch.rand(2, 5, device=DEVICE)
        gt   = [[0, 1], [3]]
        result = compute_recall_at_k(attn, gt, k=100)
        assert result["k"] == 5

    def test_batch_mixed_recall(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k
        attn = torch.tensor(
            [[0.9, 0.9, 0.05, 0.05],
             [0.05, 0.05, 0.9, 0.9]],
            device=DEVICE
        )
        gt = [[0], [1]]
        result = compute_recall_at_k(attn, gt, k=2)
        assert result["recall_at_k"] == pytest.approx(0.5)
        assert result["n_samples"] == 2

    def test_k_curve_keys(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k_curve
        attn = torch.rand(4, 20, device=DEVICE)
        gt   = [[5], [10, 11], [3], [18]]
        ks   = [1, 5, 10]
        result = compute_recall_at_k_curve(attn, gt, k_values=ks)
        assert set(result.keys()) == {"recall_at_1", "recall_at_5", "recall_at_10"}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_k_curve_monotone(self):
        """Recall@K is non-decreasing in K."""
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.recall_at_k import compute_recall_at_k_curve
        torch.manual_seed(0)
        attn = torch.rand(8, 30, device=DEVICE)
        gt   = [[5, 15]] * 8
        ks = [1, 3, 5, 10]
        result = compute_recall_at_k_curve(attn, gt, k_values=ks)
        vals = [result[f"recall_at_{k}"] for k in ks]
        for a, b in zip(vals, vals[1:]):
            assert a <= b + 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# 6.5  Dice Overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiceOverlap:
    def _make_masks(self, B, D, H, W, device):
        masks = torch.zeros(B, D, H, W, dtype=torch.bool, device=device)
        masks[:, : D // 2, :, :] = True
        return masks

    def test_keys_present(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap
        B, D, H, W = 2, 10, 16, 16
        attn = torch.ones(B, D, device=DEVICE) / D
        masks = self._make_masks(B, D, H, W, DEVICE)
        result = compute_dice_overlap(attn, masks)
        assert "dice_mean" in result
        assert "dice_std" in result
        assert "dice_per_sample" in result
        assert len(result["dice_per_sample"]) == B

    def test_all_zero_mask_nan(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap
        B, D, H, W = 2, 8, 8, 8
        attn = torch.rand(B, D, device=DEVICE)
        attn = attn / attn.sum(dim=1, keepdim=True)
        masks = torch.zeros(B, D, H, W, dtype=torch.bool, device=DEVICE)
        result = compute_dice_overlap(attn, masks)
        for v in result["dice_per_sample"]:
            assert math.isnan(v)
        assert math.isnan(result["dice_mean"])

    def test_explicit_threshold_perfect(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap
        B, D, H, W = 2, 10, 8, 8
        attn = torch.zeros(B, D, device=DEVICE)
        attn[:, 0] = 1.0
        masks = torch.zeros(B, D, H, W, dtype=torch.bool, device=DEVICE)
        masks[:, 0, :, :] = True
        result = compute_dice_overlap(attn, masks, threshold=0.5)
        for d in result["dice_per_sample"]:
            assert not math.isnan(d)
            assert d == pytest.approx(1.0, abs=1e-4)

    def test_depth_mismatch_raises(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap
        attn  = torch.rand(2, 8, device=DEVICE)
        masks = torch.zeros(2, 10, 8, 8, dtype=torch.bool, device=DEVICE)
        with pytest.raises(ValueError, match="D"):
            compute_dice_overlap(attn, masks)

    def test_dice_range(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_overlap
        torch.manual_seed(42)
        B, D, H, W = 4, 16, 16, 16
        attn = torch.rand(B, D, device=DEVICE)
        attn = attn / attn.sum(dim=1, keepdim=True)
        masks = torch.rand(B, D, H, W, device=DEVICE) > 0.5
        result = compute_dice_overlap(attn, masks)
        for v in result["dice_per_sample"]:
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0 + 1e-6

    def test_dice_per_structure_missing_file(self):
        """Missing masks return NaN, not a crash."""
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_per_structure

        class _FakeDataset:
            def load_mask(self, pid, structure):
                raise FileNotFoundError(f"no mask for {pid}/{structure}")

        attn = torch.rand(2, 10, device=DEVICE)
        result = compute_dice_per_structure(
            attn,
            _FakeDataset(),
            patient_ids=["p1", "p2"],
            structure_names=["liver", "spleen"],
            device=DEVICE,
        )
        assert math.isnan(result["dice_liver"])
        assert math.isnan(result["dice_spleen"])
        assert math.isnan(result["dice_macro"])

    def test_dice_per_structure_macro(self):
        _skip_if_no_cuda()
        from aadp.evaluation.metrics.dice_overlap import compute_dice_per_structure

        class _FakeDataset:
            def load_mask(self, pid, structure):
                D, H, W = 10, 8, 8
                mask = torch.zeros(D, H, W, dtype=torch.bool)
                mask[:5] = True
                return mask

        attn = torch.zeros(2, 10, device=DEVICE)
        attn[:, :5] = 0.2
        result = compute_dice_per_structure(
            attn,
            _FakeDataset(),
            patient_ids=["p1", "p2"],
            structure_names=["liver", "spleen"],
            device=DEVICE,
        )
        assert "dice_liver" in result
        assert "dice_spleen" in result
        assert "dice_macro" in result
        assert not math.isnan(result["dice_macro"])
