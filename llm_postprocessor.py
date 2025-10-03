"""Utilities for preparing secondary diagnosis prompts and feature summaries."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass
class BatchFeatureSummary:
    """Container for the PCA- and SPE-based summary of a batch."""

    batch_name: str
    sample_count: int
    spe_values: Sequence[float]
    t2_values: Sequence[float]
    pca_scores: np.ndarray

    def to_serializable(self) -> Dict[str, Any]:
        """Serialize numeric fields into JSON-safe types."""
        percentile_levels = [5, 25, 50, 75, 90, 95, 99]
        spe_percentiles = {
            f"p{lvl}": float(np.percentile(self.spe_values, lvl)) for lvl in percentile_levels
        }
        t2_percentiles = {
            f"p{lvl}": float(np.percentile(self.t2_values, lvl)) for lvl in percentile_levels
        }
        component_stats: List[Dict[str, Any]] = []
        for idx in range(self.pca_scores.shape[1]):
            component_scores = self.pca_scores[:, idx]
            component_stats.append(
                {
                    "component": int(idx + 1),
                    "mean": float(np.mean(component_scores)),
                    "std": float(np.std(component_scores)),
                    "max": float(np.max(component_scores)),
                    "min": float(np.min(component_scores)),
                }
            )

        # Identify the top anomalous samples using SPE ranking.
        spe_array = np.asarray(self.spe_values)
        top_indices = np.argsort(spe_array)[-5:][::-1]
        top_samples = [
            {
                "index": int(idx),
                "spe": float(spe_array[idx]),
                "t2": float(self.t2_values[idx]),
                "score_rank": int(rank + 1),
            }
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "batch_name": self.batch_name,
            "sample_count": int(self.sample_count),
            "spe_statistics": {
                "mean": float(np.mean(self.spe_values)),
                "std": float(np.std(self.spe_values)),
                "max": float(np.max(self.spe_values)),
                "min": float(np.min(self.spe_values)),
                "percentiles": spe_percentiles,
            },
            "t2_statistics": {
                "mean": float(np.mean(self.t2_values)),
                "std": float(np.std(self.t2_values)),
                "max": float(np.max(self.t2_values)),
                "min": float(np.min(self.t2_values)),
                "percentiles": t2_percentiles,
            },
            "principal_component_summary": component_stats,
            "top_anomalies": top_samples,
        }


def build_feature_summary(
    batch_name: str,
    spe_values: Sequence[float],
    t2_values: Sequence[float],
    pca_scores: np.ndarray,
) -> BatchFeatureSummary:
    """Generate a summarized view of PCA/SPE metrics for the batch."""
    return BatchFeatureSummary(
        batch_name=batch_name,
        sample_count=len(spe_values),
        spe_values=np.asarray(spe_values, dtype=float),
        t2_values=np.asarray(t2_values, dtype=float),
        pca_scores=np.asarray(pca_scores, dtype=float),
    )


def load_prompt_template(path: Path) -> str:
    """Load the markdown prompt template from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found at {path}")
    return path.read_text(encoding="utf-8")


def render_secondary_prompt(
    template: str,
    *,
    initial_label: str,
    feature_summary: Mapping[str, Any],
    candidate_faults: Iterable[str],
) -> str:
    """Render the secondary diagnosis prompt with the collected context."""
    context = {
        "initial_label": initial_label,
        "feature_summary": json.dumps(feature_summary, ensure_ascii=False, indent=2),
        "candidate_faults": ", ".join(candidate_faults),
    }
    return template.format(**context)


def format_feature_digest(summary: Mapping[str, Any]) -> str:
    """Create a short textual digest used inside evidence blocks."""
    spe_stats = summary.get("spe_statistics", {})
    t2_stats = summary.get("t2_statistics", {})
    anomalies = summary.get("top_anomalies", [])

    def _extract_stat(block: Mapping[str, Any], field: str) -> str:
        value = block.get(field)
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return "N/A"
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)

    lines = [
        f"SPE mean/std: {_extract_stat(spe_stats, 'mean')} / {_extract_stat(spe_stats, 'std')}",
        f"SPE p95: {_extract_stat(spe_stats.get('percentiles', {}), 'p95')}",
        f"T² mean/std: {_extract_stat(t2_stats, 'mean')} / {_extract_stat(t2_stats, 'std')}",
        f"T² p95: {_extract_stat(t2_stats.get('percentiles', {}), 'p95')}",
    ]

    if anomalies:
        top_line = ", ".join(
            f"#{item['score_rank']}: idx={item['index']}, SPE={item['spe']:.3f}, T²={item['t2']:.3f}"
            for item in anomalies
        )
        lines.append(f"Top anomalies: {top_line}")

    return "\n".join(lines)


__all__ = [
    "BatchFeatureSummary",
    "build_feature_summary",
    "load_prompt_template",
    "render_secondary_prompt",
    "format_feature_digest",
]
