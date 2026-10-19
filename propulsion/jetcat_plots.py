"""Plot engine weight vs. static thrust and weight vs. TSFC.

Usage:
    python jetcat_plots.py                      # reads engine_catalogue.csv next to this script
    python jetcat_plots.py path/to/file.csv     # custom CSV
    python jetcat_plots.py --metric             # kg / N / kg/(N*h) instead of lb / lbf / lb/(lbf*h)
    python jetcat_plots.py --no-show            # just save the PNGs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TREND = "#8a8984"
# One color + marker shape per manufacturer, assigned in this fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def load(csv_path, metric):
    df = pd.read_csv(csv_path)
    if not metric:
        df["weight"] = df["weight_lb"]
        df["thrust"] = df["thrust_max_lbf"]
        df["tsfc"] = df["tsfc_lb_per_lbf_h"]
        units = {"weight": "lb", "thrust": "lbf", "tsfc": "lb/(lbf·h)"}
    else:
        df["weight"] = df["weight_g"] / 1000.0
        df["thrust"] = df["thrust_max_N"]
        df["tsfc"] = df["sfc_kg_per_N_h"]
        units = {"weight": "kg", "thrust": "N", "tsfc": "kg/(N·h)"}
    if "manufacturer" not in df.columns:
        df["manufacturer"] = "JetCat"

    # Rows missing weight, thrust or TSFC can't be plotted (and would break the fit).
    missing = df[["weight", "thrust", "tsfc"]].isna().any(axis=1)
    for name in df.loc[missing, "engine"]:
        print(f"Skipping {name}: missing weight, thrust or TSFC")
    df = df[~missing]
    return df.sort_values("weight").reset_index(drop=True), units


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


# Candidate label positions (offset in points, horizontal alignment), tried in order.
CANDIDATES = [((0, 9), "center"), ((0, -15), "center"), ((8, -3), "left"),
              ((-8, -3), "right"), ((8, 8), "left"), ((-8, 8), "right"),
              ((8, -14), "left"), ((-8, -14), "right"), ((0, 20), "center")]


def label_points(ax, df, ycol):
    # Short names ("P400") keep labels compact. Each label takes the first candidate
    # spot that doesn't overlap a marker or an already-placed label.
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(df[["weight", ycol]].to_numpy())
    marker_r = 6 * fig.dpi / 72  # ~marker radius in pixels
    taken = []

    def clear(bbox):
        if any(bbox.overlaps(b) for b in taken):
            return False
        return not any(bbox.x0 - marker_r < x < bbox.x1 + marker_r and
                       bbox.y0 - marker_r < y < bbox.y1 + marker_r for x, y in pts)

    for (_, row) in df.iterrows():
        name = row["engine"].split("-")[0]
        for offset, ha in CANDIDATES:
            text = ax.annotate(name, (row["weight"], row[ycol]), xytext=offset,
                               textcoords="offset points", ha=ha, fontsize=8,
                               color=TEXT_SECONDARY)
            bbox = text.get_window_extent(renderer).expanded(1.05, 1.1)
            if clear(bbox):
                break
            if (offset, ha) != CANDIDATES[-1]:
                text.remove()
        taken.append(bbox)


def plot(df, ycol, ylabel, title, units, out_path, fit_label):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    style_axes(ax)

    slope, intercept = np.polyfit(df["weight"], df[ycol], 1)
    xs = np.linspace(df["weight"].min(), df["weight"].max(), 100)
    ax.plot(xs, slope * xs + intercept, color=TREND, linewidth=1.5, linestyle="--",
            label=fit_label(slope, intercept))

    for i, (maker, group) in enumerate(df.groupby("manufacturer", sort=True)):
        ax.scatter(group["weight"], group[ycol], s=64,
                   color=SERIES_COLORS[i % len(SERIES_COLORS)],
                   marker=SERIES_MARKERS[i % len(SERIES_MARKERS)],
                   edgecolors=SURFACE, linewidths=2, zorder=3, label=maker)

    ax.set_xlabel(f"Engine weight ({units['weight']})")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=12)
    ax.margins(x=0.06, y=0.1)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="best")

    # Labels go last, once the layout (and so the data->pixel transform) is final.
    fig.tight_layout()
    label_points(ax, df, ycol)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    print(f"Saved {out_path}")
    return fig


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default=here / "engine_catalogue.csv", type=Path)
    parser.add_argument("--metric", action="store_true", help="use kg / N units instead of lb / lbf")
    parser.add_argument("--no-show", action="store_true", help="save PNGs without opening windows")
    args = parser.parse_args()

    df, u = load(args.csv, args.metric)
    # PNGs are named after the CSV, e.g. engine_catalogue_weight_vs_thrust.png
    out_base = args.csv.resolve().parent / args.csv.stem

    makers = sorted(df["manufacturer"].unique())
    who = makers[0] if len(makers) == 1 else ", ".join(makers[:-1]) + " & " + makers[-1]

    plot(df, "thrust", f"Max static thrust ({u['thrust']})",
         f"{who} engines: weight vs. static thrust", u,
         f"{out_base}_weight_vs_thrust.png",
         lambda m, b: f"Linear fit: T = {m:.1f}·W {b:+.1f}")

    plot(df, "tsfc", f"TSFC at max thrust ({u['tsfc']})",
         f"{who} engines: weight vs. TSFC", u,
         f"{out_base}_weight_vs_tsfc.png",
         lambda m, b: f"Linear fit: TSFC = {m:.4f}·W {b:+.3f}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
