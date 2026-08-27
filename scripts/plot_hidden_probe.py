#!/usr/bin/env python3
"""Render the hidden-state probe summary as a dependency-free SVG."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
    return "<line " + " ".join(f'{key}="{value}"' for key, value in values.items()) + "/>"


def text(x: float, y: float, value: object, **attrs: object) -> str:
    values = {"x": x, "y": y, **attrs}
    return (
        "<text "
        + " ".join(f'{key}="{item}"' for key, item in values.items())
        + f">{html.escape(str(value))}</text>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    contextual = summary["selected_contextual"]
    baseline = summary["stage0_baseline"]
    interval = summary["primary_paired_protein_bootstrap"]

    width, height = 1240, 560
    left = (70, 115, 540, 360)
    right = (680, 115, 500, 360)
    y_min, y_max = 0.50, 0.70
    colors = {1e-5: "#8fb8d8", 1e-4: "#176b9c", 1e-3: "#45a6a1"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<metadata>summary_sha256={hashlib.sha256(raw).hexdigest()}</metadata>",
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        text(60, 48, "ProGen2 contextual hidden states improve contact discrimination", **{"font-family": "sans-serif", "font-size": 24, "font-weight": 700, "fill": "#17202a"}),
        text(60, 78, f"Held-out mean protein AUC gain: +{interval['mean_auc_difference']:.3f}  (95% protein bootstrap CI +{interval['lower']:.3f} to +{interval['upper']:.3f})", **{"font-family": "sans-serif", "font-size": 15, "fill": "#40505c"}),
    ]

    def y_position(value: float, panel: tuple[int, int, int, int]) -> float:
        return panel[1] + panel[3] * (y_max - value) / (y_max - y_min)

    # Panel 1: validation AUC at every stage and alpha.
    px, py, pw, ph = left
    svg.append(text(px, py - 24, "Validation: signal strengthens in late layers", **{"font-family": "sans-serif", "font-size": 17, "font-weight": 700, "fill": "#17202a"}))
    for value in (0.50, 0.55, 0.60, 0.65, 0.70):
        y = y_position(value, left)
        svg.append(line(px, y, px + pw, y, stroke="#d9dee2", **{"stroke-width": 1}))
        svg.append(text(px - 12, y + 5, f"{value:.2f}", **{"text-anchor": "end", "font-family": "sans-serif", "font-size": 12, "fill": "#52616b"}))
    for stage in (0, 5, 10, 15, 20, 25, 27):
        x = px + pw * stage / 27
        svg.append(line(x, py + ph, x, py + ph + 6, stroke="#52616b"))
        svg.append(text(x, py + ph + 23, stage, **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 12, "fill": "#52616b"}))
    svg.extend(
        [
            line(px, py, px, py + ph, stroke="#52616b", **{"stroke-width": 1.2}),
            line(px, py + ph, px + pw, py + ph, stroke="#52616b", **{"stroke-width": 1.2}),
            text(px + pw / 2, py + ph + 48, "ProGen2 representation stage", **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 13, "fill": "#34434d"}),
            text(px - 48, py + ph / 2, "Mean protein AUC", transform=f"rotate(-90 {px - 48} {py + ph / 2})", **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 13, "fill": "#34434d"}),
        ]
    )
    grid = summary["validation_grid"]
    for alpha in (1e-5, 1e-4, 1e-3):
        rows = sorted((row for row in grid if float(row["alpha"]) == alpha), key=lambda row: row["stage"])
        points = " ".join(
            f"{px + pw * row['stage'] / 27:.2f},{y_position(row['validation']['mean_per_protein_auc'], left):.2f}"
            for row in rows
        )
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[alpha]}" stroke-width="2.3"/>')
    selected_x = px + pw * contextual["stage"] / 27
    selected_y = y_position(contextual["validation"]["mean_per_protein_auc"], left)
    svg.append(f'<circle cx="{selected_x}" cy="{selected_y}" r="5.5" fill="#e45756" stroke="#ffffff" stroke-width="2"/>')
    svg.append(text(selected_x - 5, selected_y - 13, f"selected: stage {contextual['stage']}", **{"text-anchor": "end", "font-family": "sans-serif", "font-size": 12, "font-weight": 700, "fill": "#b23b3a"}))
    legend_x = px + 12
    for offset, alpha in enumerate((1e-5, 1e-4, 1e-3)):
        lx = legend_x + offset * 112
        svg.append(line(lx, py + 18, lx + 22, py + 18, stroke=colors[alpha], **{"stroke-width": 3}))
        svg.append(text(lx + 29, py + 23, f"α={alpha:g}", **{"font-family": "sans-serif", "font-size": 11, "fill": "#40505c"}))

    # Panel 2: untouched test performance by exact sequence-separation bin.
    px, py, pw, ph = right
    svg.append(text(px, py - 24, "Held-out test: contextual signal persists at long range", **{"font-family": "sans-serif", "font-size": 17, "font-weight": 700, "fill": "#17202a"}))
    for value in (0.50, 0.55, 0.60, 0.65, 0.70):
        y = y_position(value, right)
        svg.append(line(px, y, px + pw, y, stroke="#d9dee2", **{"stroke-width": 1}))
        svg.append(text(px - 12, y + 5, f"{value:.2f}", **{"text-anchor": "end", "font-family": "sans-serif", "font-size": 12, "fill": "#52616b"}))
    svg.extend(
        [
            line(px, py, px, py + ph, stroke="#52616b", **{"stroke-width": 1.2}),
            line(px, py + ph, px + pw, py + ph, stroke="#52616b", **{"stroke-width": 1.2}),
            text(px - 48, py + ph / 2, "Pooled AUC", transform=f"rotate(-90 {px - 48} {py + ph / 2})", **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 13, "fill": "#34434d"}),
        ]
    )
    contextual_bins = contextual["test"]["distance_bins"]
    baseline_bins = baseline["test"]["distance_bins"]
    group_width = pw / len(contextual_bins)
    bar_width = 25
    baseline_color, contextual_color = "#e7a28e", "#176b9c"
    for index, (contextual_row, baseline_row) in enumerate(zip(contextual_bins, baseline_bins)):
        center = px + group_width * (index + 0.5)
        for x, row, color in (
            (center - bar_width - 2, baseline_row, baseline_color),
            (center + 2, contextual_row, contextual_color),
        ):
            y = y_position(row["auc"], right)
            bottom = y_position(0.50, right)
            svg.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bottom - y:.2f}" rx="2" fill="{color}"/>')
        svg.append(text(center, py + ph + 22, contextual_row["label"], **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 11, "fill": "#52616b"}))
    for offset, (label, color) in enumerate((("Stage 0 embedding", baseline_color), (f"Stage {contextual['stage']} contextual", contextual_color))):
        lx = px + 15 + offset * 185
        svg.append(f'<rect x="{lx}" y="{py + 10}" width="16" height="12" rx="2" fill="{color}"/>')
        svg.append(text(lx + 23, py + 21, label, **{"font-family": "sans-serif", "font-size": 11, "fill": "#40505c"}))
    svg.append(text(px + pw / 2, py + ph + 48, "Sequence separation (residues)", **{"text-anchor": "middle", "font-family": "sans-serif", "font-size": 13, "fill": "#34434d"}))
    svg.append(text(width - 60, height - 18, "Frozen 90/30/30 protein split · 18,166 matched test pairs", **{"text-anchor": "end", "font-family": "sans-serif", "font-size": 11, "fill": "#687780"}))
    svg.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
