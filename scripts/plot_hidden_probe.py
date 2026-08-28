#!/usr/bin/env python3
"""Render the hidden-state probe result as a simple, dependency-free SVG."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


FONT = "Arial, sans-serif"
INK = "#24313a"
MUTED = "#60717c"
GRID = "#dfe5e8"
BLUE = "#2774a5"
SALMON = "#e79b83"
RED = "#c84b49"


def element(name: str, **attrs: object) -> str:
    values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f"<{name} {values}/>"


def label(x: float, y: float, value: object, **attrs: object) -> str:
    values = {"x": x, "y": y, "font-family": FONT, **attrs}
    rendered = " ".join(
        f'{key.replace("_", "-")}="{item}"' for key, item in values.items()
    )
    return f"<text {rendered}>{html.escape(str(value))}</text>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    selected = summary["selected_contextual"]
    baseline = summary["stage0_baseline"]
    interval = summary["primary_paired_protein_bootstrap"]

    width, height = 1120, 520
    left = (70, 125, 455, 300)
    right = (625, 125, 455, 300)
    y_min, y_max = 0.50, 0.70

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<metadata>summary_sha256={hashlib.sha256(raw).hexdigest()}</metadata>",
        element("rect", width="100%", height="100%", fill="#ffffff"),
        label(50, 42, "Hidden-state probe", font_size=23, font_weight=700, fill=INK),
        label(
            50,
            70,
            f"Test AUC: {baseline['test']['mean_per_protein_auc']:.3f} before context → "
            f"{selected['test']['mean_per_protein_auc']:.3f} at stage {selected['stage']} "
            f"(+{interval['mean_auc_difference']:.3f})",
            font_size=14,
            fill=MUTED,
        ),
    ]

    def y_position(value: float, panel: tuple[int, int, int, int]) -> float:
        return panel[1] + panel[3] * (y_max - value) / (y_max - y_min)

    def axes(panel: tuple[int, int, int, int], y_title: str) -> list[str]:
        px, py, pw, ph = panel
        items: list[str] = []
        for value in (0.50, 0.55, 0.60, 0.65, 0.70):
            y = y_position(value, panel)
            items.append(element("line", x1=px, y1=y, x2=px + pw, y2=y, stroke=GRID))
            items.append(
                label(
                    px - 10,
                    y + 4,
                    f"{value:.2f}",
                    text_anchor="end",
                    font_size=11,
                    fill=MUTED,
                )
            )
        items.extend(
            [
                element("line", x1=px, y1=py, x2=px, y2=py + ph, stroke=MUTED),
                element(
                    "line",
                    x1=px,
                    y1=py + ph,
                    x2=px + pw,
                    y2=py + ph,
                    stroke=MUTED,
                ),
                label(
                    px - 48,
                    py + ph / 2,
                    y_title,
                    transform=f"rotate(-90 {px - 48} {py + ph / 2})",
                    text_anchor="middle",
                    font_size=12,
                    fill=INK,
                ),
            ]
        )
        return items

    # Left: validation performance for the selected classifier strength only.
    px, py, pw, ph = left
    svg.append(label(px, 105, "Choose the ProGen2 stage", font_size=16, font_weight=700, fill=INK))
    svg.extend(axes(left, "Validation AUC"))

    rows = sorted(
        (
            row
            for row in summary["validation_grid"]
            if float(row["alpha"]) == float(selected["alpha"])
        ),
        key=lambda row: row["stage"],
    )
    points = " ".join(
        f"{px + pw * row['stage'] / 27:.2f},"
        f"{y_position(row['validation']['mean_per_protein_auc'], left):.2f}"
        for row in rows
    )
    svg.append(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')

    selected_x = px + pw * selected["stage"] / 27
    selected_y = y_position(selected["validation"]["mean_per_protein_auc"], left)
    svg.append(element("circle", cx=selected_x, cy=selected_y, r=5, fill=RED))
    svg.append(
        label(
            selected_x - 8,
            selected_y - 11,
            f"stage {selected['stage']}",
            text_anchor="end",
            font_size=12,
            font_weight=700,
            fill=RED,
        )
    )
    for stage in (0, 5, 10, 15, 20, 25, 27):
        x = px + pw * stage / 27
        svg.append(label(x, py + ph + 20, stage, text_anchor="middle", font_size=11, fill=MUTED))
    svg.append(
        label(
            px + pw / 2,
            py + ph + 45,
            "ProGen2 stage",
            text_anchor="middle",
            font_size=12,
            fill=INK,
        )
    )

    # Right: final test performance by sequence separation.
    px, py, pw, ph = right
    svg.append(label(px, 105, "Test performance by sequence separation", font_size=16, font_weight=700, fill=INK))
    svg.extend(axes(right, "Test AUC"))

    contextual_bins = selected["test"]["distance_bins"]
    baseline_bins = baseline["test"]["distance_bins"]
    group_width = pw / len(contextual_bins)
    bar_width = 24
    for index, (contextual_row, baseline_row) in enumerate(
        zip(contextual_bins, baseline_bins)
    ):
        center = px + group_width * (index + 0.5)
        for x, row, color in (
            (center - bar_width - 2, baseline_row, SALMON),
            (center + 2, contextual_row, BLUE),
        ):
            y = y_position(row["auc"], right)
            bottom = y_position(0.50, right)
            svg.append(
                element(
                    "rect",
                    x=f"{x:.2f}",
                    y=f"{y:.2f}",
                    width=bar_width,
                    height=f"{bottom - y:.2f}",
                    rx=2,
                    fill=color,
                )
            )
        svg.append(
            label(
                center,
                py + ph + 20,
                contextual_row["label"],
                text_anchor="middle",
                font_size=10,
                fill=MUTED,
            )
        )

    legend_y = py + 18
    legend_items = (
        (px + 16, "Stage 0", SALMON),
        (px + 130, f"Stage {selected['stage']}", BLUE),
    )
    for x, legend_label, color in legend_items:
        svg.append(element("rect", x=x, y=legend_y - 11, width=15, height=11, rx=2, fill=color))
        svg.append(label(x + 22, legend_y, legend_label, font_size=11, fill=INK))
    svg.append(
        label(
            px + pw / 2,
            py + ph + 45,
            "Positions apart in the sequence",
            text_anchor="middle",
            font_size=12,
            fill=INK,
        )
    )

    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
