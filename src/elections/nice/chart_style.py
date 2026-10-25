"""Shared look and data loading for the turnout charts.

Every chart script imports this module, so colors, fonts, titles and the data
loading stay identical across model2.py, hourly.py, normalized.py and heatmap.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- Data -------------------------------------------------------------------
DATA_PATH = "src/elections/data.csv"
OUTPUT_DIR = Path("charts")
TARGET = "Parlamentarni 2026"
EXCLUDE = ["Predsednički 2000.", "Parlamentarni 2000."]  # too little hourly data
POLLS_OPEN = 7
FINAL_HOUR = 20
SOURCE = "Izvor: studenti_u_blokadi"

# --- Palette (studenti_u_blokadi: crimson on light blue, white shapes) -------
SURFACE = "#ffffff"  # page background
PANEL = "#ffffff"  # plot area
INK = "#1a1a1a"  # key numbers
INK_2 = "#3d3d3d"  # labels, subtitles
INK_3 = "#5f6675"  # footnotes, tick labels
GRID = "#ebeef4"
CONTEXT = "#c4c8d0"  # past elections when they are background

ACCENT = "#ae1e37"  # crimson: the current election (2026) and titles
ACCENT_SOFT = "#e5b3bc"
PAST = "#2b2b2b"  # past elections when in focus
PAST_SOFT = "#b9bec8"

# Sequential crimson ramp (light -> dark) for magnitudes, e.g. the heatmap
RAMP = [
    "#fbf0f2",
    "#f3cfd6",
    "#e7a3b0",
    "#d67185",
    "#c4445d",
    "#ae1e37",
    "#8c1629",
    "#68101e",
    "#430913",
]
RAMP_EDGE = PAST  # outline for the highlighted row in the heatmap


def _pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]:
        if name in available:
            return name
    return "sans-serif"


def apply():
    """Set the shared matplotlib look. Call once at the top of each script."""
    plt.rcParams.update(
        {
            "font.family": _pick_font(),
            "font.size": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": PANEL,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "lines.solid_capstyle": "round",
            "hatch.linewidth": 0.8,
        }
    )


def load_data():
    """Return the turnout table (numbers, '~' removed) and the hour of each column."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8").set_index("Izbori")
    df = df.apply(
        lambda col: pd.to_numeric(col.astype(str).str.replace("~", ""), errors="coerce")
    )
    df = df.drop(index=EXCLUDE, errors="ignore")
    hours = np.array([int(c.rstrip("h")) for c in df.columns])
    return df, hours


def readings(row, hours):
    """Known (hour, turnout) points of one election, with 0% at poll opening."""
    known = row.notna().values
    return np.r_[POLLS_OPEN, hours[known]], np.r_[0.0, row.values[known]]


def short(name):
    """Compact election name for tight labels."""
    return (
        name.replace("Parlamentarni", "Parl.")
        .replace("Predsednički", "Pred.")
        .replace(" II krug", " II")
        .rstrip(".")
    )


def hour_axis(ax, hours=None, every=1):
    hours = range(POLLS_OPEN, FINAL_HOUR + 1) if hours is None else hours
    ticks = [h for h in hours if (h - POLLS_OPEN) % every == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{h}h" for h in ticks])


def _from_top(fig, inches):
    return 1 - inches / fig.get_figheight()


def _left(fig):
    return 0.22 / fig.get_figwidth()


def header(fig, title, subtitle=None):
    """Left-aligned title block; positions are in inches so every figure size matches."""
    fig.patches.append(
        plt.Rectangle(
            (0, _from_top(fig, 0.07)),
            1,
            0.07 / fig.get_figheight(),
            transform=fig.transFigure,
            color=ACCENT,
            zorder=10,
        )
    )
    fig.text(
        _left(fig),
        _from_top(fig, 0.24),
        title.upper(),
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=ACCENT,
    )
    if subtitle:
        fig.text(
            _left(fig),
            _from_top(fig, 0.64),
            subtitle,
            ha="left",
            va="top",
            fontsize=10.5,
            color=INK_2,
        )


def header_legend(fig, handles, ncol=None):
    """Legend in one row directly under the subtitle."""
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=ncol or len(handles),
        bbox_to_anchor=(_left(fig) - 0.004, _from_top(fig, 0.92)),
        handlelength=1.6,
        columnspacing=1.8,
        borderaxespad=0,
    )


def footer(fig, note=None):
    text = SOURCE if not note else f"{note}    ·    {SOURCE}"
    fig.text(
        _left(fig),
        0.16 / fig.get_figheight(),
        text,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK_3,
    )


def save(fig, filename, header_in=1.05, footer_in=0.45):
    """Save into OUTPUT_DIR, leaving room (in inches) for the header and footer."""
    top = _from_top(fig, header_in)
    bottom = footer_in / fig.get_figheight()
    fig.tight_layout(rect=[0, bottom, 1, top])
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")
    return path
