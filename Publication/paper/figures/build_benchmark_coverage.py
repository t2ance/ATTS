"""Build a 2-ring donut chart showing benchmark coverage:
inner ring = 4 benchmarks (HLE-Verified Gold, LiveCodeBench v6, BabyVision, GPQA-Diamond);
outer ring = per-benchmark subcategories with item counts.

Counts verified 2026-05-04 via direct dataset reads against:
  - skylenage/HLE-Verified (Gold subset filter)
  - livecodebench/code_generation_lite, release_v6
  - UnipatAI/BabyVision
  - Idavidrein/gpqa, gpqa_diamond
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Wedge
from PIL import Image


HLE_CATEGORIES = [
    ("Math", 271),
    ("Biology/Medicine", 97),
    ("Computer Science/AI", 86),
    ("Other", 62),
    ("Humanities/Social Science", 57),
    ("Physics", 50),
    ("Engineering", 25),
    ("Chemistry", 20),
]

LCB_DIFFICULTIES = [
    ("Medium", 383),
    ("Hard", 350),
    ("Easy", 322),
]

BABYVISION_SUBTYPES = [
    ("Count Same Patterns", 35),
    ("3D Views", 27),
    ("Find the shadow", 23),
    ("Recognize numbers and letters", 23),
    ("Count 3D blocks", 22),
    ("2D Pattern Completion", 20),
    ("Pattern and Color Completion", 20),
    ("Maze", 20),
    ("Connect the lines", 19),
    ("Count Clusters", 18),
    ("3D Pattern Completion", 18),
    ("Find the same", 17),
    ("Overlay Patterns", 17),
    ("Find the different", 16),
    ("Reconstruction", 14),
    ("Logic Patterns", 14),
    ("Metro map", 12),
    ("3D Cube Unfold", 12),
    ("Paper Folding", 12),
    ("Rotation Patterns", 10),
    ("Mirroring Patterns", 10),
    ("Lines Observation", 9),
]

GPQA_SUBDOMAINS = [
    ("Organic Chemistry", 72),
    ("Quantum Mechanics", 25),
    ("Chemistry (general)", 20),
    ("Physics (general)", 19),
    ("Molecular Biology", 15),
    ("High-energy particle physics", 14),
    ("Astrophysics", 13),
    ("Relativistic Mechanics", 7),
    ("Electromagnetism and Photonics", 6),
    ("Genetics", 4),
    ("Inorganic Chemistry", 1),
    ("Optics and Acoustics", 1),
    ("Condensed Matter Physics", 1),
]

BENCHMARKS = [
    # Wrapped to 2 lines so each label fits inside its 90 deg wedge without
    # crossing the radial separator drawn at the wedge boundary. The fourth
    # tuple element is the Lucide icon stem (file at icons/<stem>.svg).
    ("HLE-Verified\nGold", HLE_CATEGORIES, "#3B6FB6", "graduation-cap"),
    ("LiveCodeBench\nv6", LCB_DIFFICULTIES, "#2E8B57", "code"),
    ("Baby\nVision", BABYVISION_SUBTYPES, "#7E57C2", "eye"),
    ("GPQA-\nDiamond", GPQA_SUBDOMAINS, "#E07B39", "atom"),
]

ICON_DIR = Path(__file__).parent / "icons"


def _load_icon_white(svg_stem: str, size_px: int = 256) -> np.ndarray:
    """Render a Lucide stroke-icon SVG to RGBA with white stroke.

    Lucide SVGs declare ``stroke="currentColor"``; rewrite to white before
    rasterizing so the icon reads on the coloured benchmark wedge.
    """
    svg_path = ICON_DIR / f"{svg_stem}.svg"
    raw = svg_path.read_text(encoding="utf-8")
    raw = raw.replace('stroke="currentColor"', 'stroke="#FFFFFF"')
    png_bytes = cairosvg.svg2png(
        bytestring=raw.encode("utf-8"),
        output_width=size_px,
        output_height=size_px,
    )
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


def _shade(hex_color: str, factor: float) -> tuple[float, float, float]:
    """Lighten/darken a hex color. factor in [-1, 1]; positive = lighter."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if factor >= 0:
        r = r + int((255 - r) * factor)
        g = g + int((255 - g) * factor)
        b = b + int((255 - b) * factor)
    else:
        f = 1 + factor
        r = int(r * f)
        g = int(g * f)
        b = int(b * f)
    return (r / 255, g / 255, b / 255)


def build():
    bench_totals = [sum(c for _, c in items) for _, items, _, _ in BENCHMARKS]
    grand_total = sum(bench_totals)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"aspect": "equal"})

    # Hollow donut: empty center hole, then two annular rings.
    hole_r = 0.22  # inner edge of the inner ring; center is empty
    inner_r0, inner_r1 = hole_r, 0.55
    outer_r0, outer_r1 = 0.55, 1.00
    leaf_label_r = 0.78  # mid-radius of outer ring

    # Equal 90 degrees per benchmark; leaves within each benchmark are
    # equal-angle so that small-count subcategories remain readable.
    bench_span_each = 360.0 / len(BENCHMARKS)
    angle = 90.0  # start at top

    for (name, items, base_color, icon_stem), bench_total in zip(BENCHMARKS, bench_totals):
        bench_span = bench_span_each
        theta_start = angle
        theta_end = angle - bench_span

        # Inner ring wedge: now an annulus from hole_r to inner_r1, so the
        # center stays hollow.
        ax.add_patch(
            Wedge(
                center=(0, 0),
                r=inner_r1,
                theta1=theta_end,
                theta2=theta_start,
                width=inner_r1 - inner_r0,
                facecolor=base_color,
                edgecolor="white",
                linewidth=1.5,
            )
        )

        # Inner annulus contents: a Lucide icon near the outer edge of the
        # annulus, with the wrapped benchmark name centered just inside it.
        mid = math.radians((theta_start + theta_end) / 2)
        annulus_thickness = inner_r1 - inner_r0
        icon_r = inner_r0 + 0.72 * annulus_thickness
        name_r = inner_r0 + 0.32 * annulus_thickness

        ax.text(
            name_r * math.cos(mid),
            name_r * math.sin(mid),
            name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            linespacing=1.05,
        )

        icon_img = _load_icon_white(icon_stem)
        # zoom is image-pixel scale: figsize is 12in -> data-units are
        # roughly 0.0915 in/unit at the 1.0 outer radius. We want the icon
        # to occupy ~0.13 data units; with a 256 px image that's ~0.0005.
        icon_box = OffsetImage(icon_img, zoom=0.13)
        ax.add_artist(
            AnnotationBbox(
                icon_box,
                (icon_r * math.cos(mid), icon_r * math.sin(mid)),
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
        )

        # outer ring leaves: equal angular split within the 90 deg sector
        n_items = len(items)
        leaf_span_each = bench_span / n_items
        for i, (leaf_name, leaf_count) in enumerate(items):
            leaf_span = leaf_span_each
            leaf_start = angle
            leaf_end = angle - leaf_span

            # alternating shade for visual rhythm; slight gradient by index
            shade_factor = 0.10 + 0.45 * (i / max(n_items - 1, 1))
            color = _shade(base_color, shade_factor)

            ax.add_patch(
                Wedge(
                    center=(0, 0),
                    r=outer_r1,
                    theta1=leaf_end,
                    theta2=leaf_start,
                    width=outer_r1 - outer_r0,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )

            # leaf label: placed INSIDE the outer ring, oriented radially
            # so each label reads along its slice from inner to outer edge.
            mid_leaf = (leaf_start + leaf_end) / 2
            mid_rad = math.radians(mid_leaf)
            x = leaf_label_r * math.cos(mid_rad)
            y = leaf_label_r * math.sin(mid_rad)

            # Radial orientation: rotate so text runs along the slice's
            # bisector. On the right half (cos>0) text reads outward;
            # on the left half flip 180 so it isn't upside down.
            rot = mid_leaf
            if math.cos(mid_rad) < 0:
                rot += 180

            # Pick text colour for contrast against the wedge fill.
            r_, g_, b_ = _shade(base_color, shade_factor)
            luminance = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            txt_color = "#FFFFFF" if luminance < 0.55 else "#222222"

            ax.text(
                x,
                y,
                leaf_name,
                ha="center",
                va="center",
                rotation=rot,
                rotation_mode="anchor",
                fontsize=8.0,
                color=txt_color,
            )

            angle = leaf_end

        # the leaves above accumulated angle; nothing extra needed

    # Outer rim and inner-hole outline for clean donut silhouette.
    ax.add_patch(
        plt.Circle((0, 0), outer_r1, fill=False, edgecolor="#888", linewidth=0.8)
    )
    ax.add_patch(
        plt.Circle((0, 0), hole_r, fill=False, edgecolor="#888", linewidth=0.8)
    )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    out_dir = Path(__file__).parent
    pdf_path = out_dir / "benchmark_coverage.pdf"
    png_path = out_dir / "benchmark_coverage.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05, dpi=180)
    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"Total items: {grand_total}")
    print(
        f"Per-benchmark: "
        f"HLE Gold {bench_totals[0]}, LCB v6 {bench_totals[1]}, "
        f"BabyVision {bench_totals[2]}, GPQA-Diamond {bench_totals[3]}"
    )


if __name__ == "__main__":
    build()
