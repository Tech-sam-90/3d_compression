"""Tests for A1 ablation: TaskConditionedIntraSliceDistiller + TaskConditionedAADP."""
import torch
import pytest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ── TaskConditionedIntraSliceDistiller ────────────────────────────────────────

class TestTaskConditionedIntraSliceDistiller:
    def test_output_shape(self):
        _skip_if_no_cuda()
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedIntraSliceDistiller

        B, D, N, C, K, cond_dim = 2, 4, 196, 192, 32, 768
        s1 = TaskConditionedIntraSliceDistiller(
            embed_dim=C, num_latents=K, num_heads=8, cond_dim=cond_dim, device=DEVICE
        )
        x = torch.randn(B * D, N, C, device=DEVICE)
        etext = torch.randn(B, cond_dim, device=DEVICE)
        with torch.no_grad():
            out = s1(x, 14, 14, etext, B, D)
        assert out.shape == (B * D, K, C)

    def test_film_is_identity_at_init(self):
        _skip_if_no_cuda()
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedIntraSliceDistiller

        B_D, K, C, cond_dim = 8, 32, 192, 768
        s1 = TaskConditionedIntraSliceDistiller(
            embed_dim=C, num_latents=K, num_heads=8, cond_dim=cond_dim, device=DEVICE
        )
        q = torch.randn(B_D, K, C, device=DEVICE)
        etext_exp = torch.randn(B_D, cond_dim, device=DEVICE)
        with torch.no_grad():
            film_out = s1.film(q, etext_exp)
        # gamma=1, beta=0 at init → film(q, *) == q
        torch.testing.assert_close(film_out, q)

    def test_output_matches_baseline_at_init(self):
        """At init (FiLM identity), TC Stage 1 must equal baseline Stage 1."""
        _skip_if_no_cuda()
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedIntraSliceDistiller
        from aadp.models.projector.stage1 import IntraSliceDistiller

        B, D, N, C, K, cond_dim = 2, 4, 196, 192, 32, 768

        task_s1 = TaskConditionedIntraSliceDistiller(
            embed_dim=C, num_latents=K, num_heads=8, cond_dim=cond_dim, device=DEVICE
        )
        base_s1 = IntraSliceDistiller(embed_dim=C, num_latents=K, num_heads=8, device=DEVICE)

        # Copy all shared parameters so the only difference is the FiLM layer
        base_s1.latents.data.copy_(task_s1.latents.data)
        base_s1.cross_attn.load_state_dict(task_s1.cross_attn.state_dict())
        base_s1.norm_q.load_state_dict(task_s1.norm_q.state_dict())
        base_s1.norm_kv.load_state_dict(task_s1.norm_kv.state_dict())
        base_s1.pos_enc.load_state_dict(task_s1.pos_enc.state_dict())

        x = torch.randn(B * D, N, C, device=DEVICE)
        etext = torch.randn(B, cond_dim, device=DEVICE)
        with torch.no_grad():
            out_task = task_s1(x, 14, 14, etext, B, D)
            out_base = base_s1(x, 14, 14)

        torch.testing.assert_close(out_task, out_base, rtol=1e-5, atol=1e-5)

    def test_invalid_head_divisibility_raises(self):
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedIntraSliceDistiller

        with pytest.raises(ValueError, match="divisible"):
            TaskConditionedIntraSliceDistiller(embed_dim=100, num_heads=8, device=DEVICE)


# ── TaskConditionedAADP ───────────────────────────────────────────────────────

class TestTaskConditionedAADP:

    @pytest.fixture(scope="class")
    def model(self):
        _skip_if_no_cuda()
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedAADP

        return TaskConditionedAADP(
            embed_dim=192,
            num_latents=16,
            num_tokens=32,
            num_heads_stage1=8,
            num_heads_stage2=8,
            cond_dim=768,
            device=DEVICE,
        )

    def test_output_shape(self, model):
        _skip_if_no_cuda()
        B, D, N, C, M = 2, 4, 196, 192, 32
        patch_tokens = torch.randn(B, D, N, C, device=DEVICE)
        etext = torch.randn(B, 768, device=DEVICE)
        with torch.no_grad():
            out = model(patch_tokens, etext, 14, 14)
        assert out.shape == (B, M, C)

    def test_num_tokens_property(self, model):
        assert model.num_tokens == 32

    def test_num_parameters_positive(self, model):
        counts = model.num_parameters()
        assert counts["stage1"] > 0
        assert counts["stage2"] > 0
        assert counts["total"] == counts["stage1"] + counts["stage2"]

    def test_get_slice_attention_shape(self, model):
        _skip_if_no_cuda()
        B, D, N, C = 2, 6, 196, 192
        patch_tokens = torch.randn(B, D, N, C, device=DEVICE)
        etext = torch.randn(B, 768, device=DEVICE)
        with torch.no_grad():
            model(patch_tokens, etext, 14, 14)
        attn = model.get_slice_attention()
        assert attn.shape == (B, D)
        assert torch.all(attn >= 0)

    def test_rebuild_at_budget_changes_num_tokens(self, model):
        _skip_if_no_cuda()
        old_latents = model.stage1.latents.data.clone()
        model.rebuild_at_budget(8)
        assert model.num_tokens == 8
        # Stage 1 must be untouched
        torch.testing.assert_close(model.stage1.latents.data, old_latents)

    def test_rebuild_at_budget_forward_still_works(self, model):
        _skip_if_no_cuda()
        model.rebuild_at_budget(4)
        B, D, N, C = 2, 4, 196, 192
        patch_tokens = torch.randn(B, D, N, C, device=DEVICE)
        etext = torch.randn(B, 768, device=DEVICE)
        with torch.no_grad():
            out = model(patch_tokens, etext, 14, 14)
        assert out.shape == (B, 4, C)

    def test_get_slice_attention_before_forward_raises(self):
        _skip_if_no_cuda()
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedAADP

        fresh = TaskConditionedAADP(embed_dim=192, num_latents=16, num_tokens=32, device=DEVICE)
        with pytest.raises(RuntimeError, match="forward"):
            fresh.get_slice_attention()


# ── Factory registration ──────────────────────────────────────────────────────

class TestFactoryEntry:
    def test_factory_builds_aadp_task_cond_stage1(self):
        _skip_if_no_cuda()
        from aadp.training.factory import build_projector
        from aadp.ablations.task_conditioned_stage1 import TaskConditionedIntraSliceDistiller

        config = {
            "projector": "aadp_task_cond_stage1",
            "num_latents": 16,
            "num_tokens": 32,
            "use_film": True,
            "max_depth": 128,
        }
        proj = build_projector(config, embed_dim=192, cond_dim=768, device=DEVICE)
        assert proj.num_tokens == 32
        assert hasattr(proj, "stage1")
        assert hasattr(proj, "stage2")
        assert isinstance(proj.stage1, TaskConditionedIntraSliceDistiller)

    def test_factory_unknown_raises(self):
        from aadp.training.factory import build_projector

        with pytest.raises(ValueError, match="Unknown projector type"):
            build_projector({"projector": "does_not_exist"}, embed_dim=192, cond_dim=768)
