#!/usr/bin/env python3
"""Pre-cache every model this project downloads at runtime.

Run this on the Narval LOGIN NODE (has internet) before submitting any SLURM
job. GPU compute nodes have no internet — all models must already be sitting
in HF_HOME / NLTK_DATA before a job starts, or `from_pretrained()` hangs
indefinitely trying to reach the network instead of failing fast (see
HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE in smoke_test_narval.sh / train_ictc.sh).

Usage:
    python scripts/cache_all_models.py --hf_token YOUR_TOKEN

Covers (found by searching the full codebase for from_pretrained /
hf_hub_download / nltk.download / hardcoded model-name strings):
  - HF transformer models: stanford-crfm/BioMedLM (main LLM + instruction
    encoder, configs/ctclip_stage2.yaml), facebook/opt-125m (smoke test +
    most of tests/), facebook/opt-1.3b (hardcoded default in
    aadp/models/ctclip_vlm.py, aadp/training/factory.py,
    scripts/train_ctclip.py), StanfordAIMI/RadBERT (bert_score fallback
    backend in aadp/evaluation/metrics/ratescore.py).
  - NLTK data: punkt, punkt_tab, wordnet, omw-1.4 (BLEU/METEOR scoring in
    scripts/vtcb_sweep_ctclip.py, colab_smoke_test.py).
  - Evaluation-metric model downloads (radgraph, ratescore, bert_score) —
    only triggered if those optional packages are installed; this script
    does not install them. If missing, it prints the pip install command
    from requirements.txt's "Evaluation metrics" section and moves on.

NOT covered here: CT-RATE report/label CSVs — those are dataset files
downloaded by scripts/download_ctrate_labels.py, not model weights.

No GREEN metric (StanfordAIMI/GREEN) and no pycocoevalcap/CIDEr
implementation exist anywhere in this codebase — nothing to cache for either.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_CACHE_DIR = "/project/def-uanazodo-ab/sadeniji/hf_cache"

# (repo_id, "causal" | "encoder", why it's needed)
HF_MODELS = [
    ("stanford-crfm/BioMedLM", "causal",
     "main LLM + instruction encoder — configs/ctclip_stage2.yaml"),
    ("facebook/opt-125m", "causal",
     "smoke-test LLM + instruction encoder, and default across tests/"),
    ("facebook/opt-1.3b", "causal",
     "hardcoded default in aadp/models/ctclip_vlm.py, aadp/training/factory.py, scripts/train_ctclip.py"),
    ("StanfordAIMI/RadBERT", "encoder",
     "bert_score fallback backend in aadp/evaluation/metrics/ratescore.py"),
]

NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]


def _set_hf_home(cache_dir: str) -> None:
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)


def _resolve_token(token: Optional[str]) -> Optional[str]:
    return token or os.environ.get("HF_TOKEN")


def _cache_hf_model(name: str, kind: str, why: str, token: Optional[str], results: Dict[str, str]) -> None:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_cls = AutoModelForCausalLM if kind == "causal" else AutoModel
    print(f"\n[model] {name} ({why})")
    try:
        print(f"  downloading tokenizer...", flush=True)
        AutoTokenizer.from_pretrained(name, token=token, trust_remote_code=True)
        print(f"  downloading weights...", flush=True)
        model_cls.from_pretrained(name, token=token, trust_remote_code=True)
        print(f"  OK")
        results[name] = "OK"
    except Exception as e:
        print(f"  FAILED: {e}")
        results[name] = f"FAILED: {e}"


def _cache_nltk(results: Dict[str, str]) -> None:
    import nltk

    for pkg in NLTK_PACKAGES:
        print(f"\n[nltk] {pkg}")
        try:
            nltk.download(pkg, quiet=True)
            print(f"  OK")
            results[f"nltk:{pkg}"] = "OK"
        except Exception as e:
            print(f"  FAILED: {e}")
            results[f"nltk:{pkg}"] = f"FAILED: {e}"


def _cache_radgraph(results: Dict[str, str]) -> None:
    print("\n[metric] radgraph")
    try:
        from radgraph import F1RadGraph
    except ImportError:
        print("  SKIPPED: package not installed (pip install radgraph==0.1.18)")
        results["radgraph"] = "SKIPPED: not installed"
        return
    try:
        F1RadGraph(reward_level="all")
        print("  OK (model cached)")
        results["radgraph"] = "OK"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["radgraph"] = f"FAILED: {e}"


def _cache_ratescore(results: Dict[str, str]) -> None:
    print("\n[metric] ratescore")
    try:
        import ratescore  # noqa: F401
    except ImportError:
        print(
            "  SKIPPED: package not installed (pip install ratescore). "
            "Falls back to bert_score/StanfordAIMI-RadBERT, which is "
            "cached separately above regardless."
        )
        results["ratescore"] = "SKIPPED: not installed"
        return
    print(
        "  package is installed but exposes no explicit pre-cache hook — "
        "its model downloads lazily on the first .compute() call."
    )
    results["ratescore"] = "PACKAGE PRESENT (not pre-triggered)"


def _cache_bert_score(results: Dict[str, str]) -> None:
    print("\n[metric] bert_score")
    try:
        import bert_score  # noqa: F401
    except ImportError:
        print("  SKIPPED: package not installed (pip install bert-score==0.3.13)")
        results["bert_score"] = "SKIPPED: not installed"
        return
    print("  package installed; backing model (StanfordAIMI/RadBERT) is cached above.")
    results["bert_score"] = "PACKAGE PRESENT"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (else HF_TOKEN env/.env)")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR,
                         help="HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE target directory")
    args = parser.parse_args()

    # Line-buffer stdout even when redirected to a file/log — otherwise
    # print() output sits in a block buffer while subprocess.run's du call
    # (which writes to the inherited fd directly) lands out of order, and
    # progress is invisible to anyone tailing the log while this runs.
    sys.stdout.reconfigure(line_buffering=True)

    _set_hf_home(args.cache_dir)
    token = _resolve_token(args.hf_token)

    results: Dict[str, str] = {}

    for name, kind, why in HF_MODELS:
        _cache_hf_model(name, kind, why, token, results)

    _cache_nltk(results)
    _cache_radgraph(results)
    _cache_ratescore(results)
    _cache_bert_score(results)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    print(f"\n=== Cache size ({args.cache_dir}) ===")
    subprocess.run(["du", "-sh", args.cache_dir])

    n_failed = sum(1 for v in results.values() if str(v).startswith("FAILED"))
    if n_failed:
        print(f"\n{n_failed} item(s) FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
