#!/usr/bin/env python3
"""container_metrics_worker.py — runs INSIDE the pinned Apptainer container.

Not meant to be imported or run directly on the host — invoked via
`apptainer exec --nv <sif> python container_metrics_worker.py` by
aadp/evaluation/metrics/container_bridge.py, using the packages installed
into the container's bind-mounted site directory (see
scripts/build_metrics_container.sh).

Reads {"metric": "green"|"radgraph_xl", "predictions": [...], "references": [...]}
from --input, computes the requested metric using the container's real
(non-shimmed) green-score / radgraph packages, writes the result to --output
as JSON. Errors are written to the output JSON as {"error": "..."} and also
printed to stderr with a non-zero exit code, so the host-side bridge can
detect the failure either way.
"""

import argparse
import json
import sys


def _compute_green(predictions, references):
    import torch
    from green_score import GREEN

    if not torch.cuda.is_available():
        return {"error": "GREEN requires a GPU but none is detected inside the container."}

    model = GREEN(model_name="StanfordAIMI/GREEN-radllama2-7b", output_dir="/tmp/green_output")
    mean_score, _, _, _, _ = model(references, predictions)
    return {"score": float(mean_score)}


def _compute_radgraph_xl(predictions, references):
    from radgraph import F1RadGraph

    scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    mean_reward, _, _, _ = scorer(refs=references, hyps=predictions)
    precision, recall, f1 = mean_reward
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to write result JSON")
    args = parser.parse_args()

    with open(args.input) as fh:
        payload = json.load(fh)

    metric = payload["metric"]
    predictions = payload["predictions"]
    references = payload["references"]

    try:
        if metric == "green":
            result = _compute_green(predictions, references)
        elif metric == "radgraph_xl":
            result = _compute_radgraph_xl(predictions, references)
        else:
            result = {"error": f"Unknown metric: {metric!r}"}
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}

    with open(args.output, "w") as fh:
        json.dump(result, fh)

    if "error" in result:
        print(f"WORKER ERROR ({metric}): {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
