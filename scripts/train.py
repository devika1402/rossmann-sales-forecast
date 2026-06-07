#!/usr/bin/env python
"""Train and cross-validate a model. Thin CLI over ``rossmann.pipeline``.

Examples
--------
    python scripts/train.py --model lgbm
    python scripts/train.py --model baseline
    python scripts/train.py --model mstl
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rossmann import config, pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def _write_report(result: dict) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Results — {result['model']}",
        "",
        "| Fold | RMSPE | MAE | MAPE |",
        "|------|-------|-----|------|",
    ]
    for f in result["folds"]:
        lines.append(
            f"| {f['fold']} | {f['rmspe']:.4f} | {f['mae']:.1f} | {f['mape']:.4f} |"
        )
    s = result["summary"]
    lines += [
        f"| **mean±std** | **{s['rmspe_mean']:.4f} ± {s['rmspe_std']:.4f}** "
        f"| {s['mae_mean']:.1f} | {s['mape_mean']:.4f} |",
        "",
    ]
    if result.get("tuned_params"):
        lines += ["## Tuned params (Optuna, Fold 1)", "```json",
                  json.dumps(result["tuned_params"], indent=2), "```", ""]
    if result.get("feature_importance") is not None:
        lines += ["## Top 15 features (gain)", "```",
                  result["feature_importance"].head(15).to_string(), "```", ""]

    report = config.OUTPUTS_DIR / "metrics_report.md"
    report.write_text("\n".join(lines))
    logger.info("Wrote %s", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Rossmann sales forecaster.")
    parser.add_argument("--model", choices=["lgbm", "baseline", "naive", "mstl"], default="lgbm")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna (lgbm only).")
    args = parser.parse_args()

    logger.info("Loading data and building features...")
    feats = pipeline.load_and_build(with_test=False)

    logger.info("Running 3-fold walk-forward CV for model=%s", args.model)
    result = pipeline.run_cv(
        feats, model_name=args.model, tune_on_fold1=not args.no_tune
    )

    s = result["summary"]
    logger.info(
        "DONE %s | RMSPE %.4f ± %.4f | MAE %.1f | MAPE %.4f",
        args.model, s["rmspe_mean"], s["rmspe_std"], s["mae_mean"], s["mape_mean"],
    )
    _write_report(result)

    # Persist tuned params so predict.py can reuse them without re-tuning.
    if result.get("tuned_params"):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.MODELS_DIR / "lgbm_params.pkl", "wb") as fh:
            pickle.dump(result["tuned_params"], fh)


if __name__ == "__main__":
    main()
