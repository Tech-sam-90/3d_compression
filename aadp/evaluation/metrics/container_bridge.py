"""container_bridge.py — subprocess bridge to the pinned Apptainer container.

GREEN and RadGraph-XL both depend on ML packages with real version
incompatibilities against this project's main training environment (see the
"KNOWN LIMITATION" notes in green.py / radgraph_f1.py). Rather than risk
shimming around those incompatibilities in-process — which for
RadGraph-XL's second broken API would mean reimplementing model-specific
special-token wrapping by hand, with real risk of silently-wrong scores —
both metrics run inside a separate, exactly-pinned Apptainer container
(pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime + transformers==4.40.0),
built by scripts/build_metrics_container.sh, via this subprocess bridge.

If the container hasn't been built, container_available() returns False and
callers fall back to their original in-process attempt — which raises in
this project's main environment, already caught by
aadp/evaluation/metrics/compute_all.py and reported as NaN with a warning.
So nothing breaks for anyone who hasn't run the container build step.

Requires `apptainer` on PATH (module load apptainer/1.3.5) in whatever
process calls this — see the SLURM scripts for the sweep/eval jobs.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

CONTAINER_SIF = os.environ.get(
    "ICTC_METRICS_CONTAINER",
    "/project/def-uanazodo-ab/sadeniji/containers/pytorch_2.2.2_cu121.sif",
)
CONTAINER_SITE = os.environ.get(
    "ICTC_METRICS_CONTAINER_SITE",
    "/project/def-uanazodo-ab/sadeniji/container_site",
)
HF_CACHE = os.environ.get(
    "ICTC_HF_CACHE",
    "/project/def-uanazodo-ab/sadeniji/hf_cache",
)
_WORKER_SCRIPT = str(Path(__file__).resolve().parents[3] / "scripts" / "container_metrics_worker.py")


def container_available() -> bool:
    """True if the pinned metrics container has been built."""
    return Path(CONTAINER_SIF).exists() and Path(CONTAINER_SITE).exists()


def run_in_container(
    metric: str,
    predictions: List[str],
    references: List[str],
    timeout: int = 1800,
) -> Dict:
    """Run ``metric`` ("green" | "radgraph_xl") inside the pinned container.

    Args:
        metric:      "green" or "radgraph_xl".
        predictions: Generated report strings.
        references:  Ground-truth report strings.
        timeout:     Max seconds to wait for the container process.

    Returns:
        The worker's result dict — either the metric's own keys, or
        ``{"error": "..."}`` on failure (never raises for a worker-side
        failure, only for infrastructure problems like a missing container
        or `apptainer` not being on PATH).

    Raises:
        RuntimeError: If the container isn't built, or the subprocess itself
            fails to run (as opposed to the metric computation failing
            inside it, which is reported via the returned dict instead).
    """
    if not container_available():
        raise RuntimeError(
            f"Metrics container not found at {CONTAINER_SIF} (or site dir "
            f"{CONTAINER_SITE} missing). Run scripts/build_metrics_container.sh "
            "on the login node first."
        )

    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "input.json"
        out_path = Path(tmp) / "output.json"
        in_path.write_text(json.dumps({
            "metric": metric,
            "predictions": predictions,
            "references": references,
        }))

        cmd = [
            "apptainer", "exec", "--cleanenv", "--nv",
            "--bind", f"{CONTAINER_SITE}:/container_site",
            "--bind", f"{tmp}:/io",
            "--env", "PYTHONPATH=/container_site",
            "--env", f"HF_HOME={HF_CACHE}",
            "--env", f"TRANSFORMERS_CACHE={HF_CACHE}",
            "--env", f"HF_DATASETS_CACHE={HF_CACHE}",
            "--env", "HF_HUB_OFFLINE=1",
            "--env", "TRANSFORMERS_OFFLINE=1",
            CONTAINER_SIF,
            "python", _WORKER_SCRIPT, "--input", "/io/input.json", "--output", "/io/output.json",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`apptainer` not found on PATH — run `module load apptainer/1.3.5` "
                "before invoking a metric backed by the container."
            ) from exc

        if out_path.exists():
            return json.loads(out_path.read_text())

        raise RuntimeError(
            f"Container metric worker produced no output (metric={metric}, "
            f"returncode={result.returncode}): "
            f"stdout={result.stdout[-1000:]!r} stderr={result.stderr[-1000:]!r}"
        )
