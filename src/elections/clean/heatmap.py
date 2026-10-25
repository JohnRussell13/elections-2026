import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Settings ---------------------------------------------------------------
CSV_PATH = "src/elections/data.csv"
TARGET = "Parlamentarni 2026"  # shown in the top row
EXCLUDE = ["Predsednički 2000.", "Parlamentarni 2000."]  # too little hourly data
POLLS_OPEN = 7
FINAL_HOUR = 20
OUTPUT = "izlaznost_heatmapa.png"

# --- Load data --------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8").set_index("Izbori")
df = df.apply(
    lambda col: pd.to_numeric(col.astype(str).str.replace("~", ""), errors="coerce")
)
df = df.drop(index=EXCLUDE, errors="ignore")
hours = np.array([int(c.rstrip("h")) for c in df.columns])

rows = [TARGET] + [
    n for n in df.index[::-1] if n != TARGET
]  # target, then newest first
n_slots = FINAL_HOUR - POLLS_OPEN  # one cell per hour: 7–8h ... 19–20h

# --- Gain per hour; gaps are spread evenly over the hours they cover ---------
gains = np.full((len(rows), n_slots), np.nan)
estimated = np.zeros_like(gains, dtype=bool)

for i, name in enumerate(rows):
    row = df.loc[name]
    known = row.notna().values
    hh = np.r_[POLLS_OPEN, hours[known]]
    vv = np.r_[0.0, row.values[known]]
    for j in range(1, len(hh)):
        start, end = hh[j - 1], hh[j]
        rate = (vv[j] - vv[j - 1]) / (end - start)
        for h in range(start, end):
            gains[i, h - POLLS_OPEN] = rate
            estimated[i, h - POLLS_OPEN] = (end - start) > 1

# --- Plot -------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list(
    "turnout", ["#f4f1e8", "#e9c46a", "#e76f51", "#7a1f2b"]
)
fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.5))
im = ax.imshow(gains, cmap=cmap, vmin=1.5, vmax=7.5, aspect="auto")

for i in range(len(rows)):
    for j in range(n_slots):
        v = gains[i, j]
        if np.isnan(v):
            continue
        est = estimated[i, j]
        ax.text(
            j,
            i,
            f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if v > 5.8 else "#222",
            style="italic" if est else "normal",
            alpha=0.65 if est else 1,
            fontweight="normal" if est else "bold",
        )

ax.set_xticks(range(n_slots))
ax.set_xticklabels([f"{h}–{h + 1}h" for h in range(POLLS_OPEN, FINAL_HOUR)], fontsize=9)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.axhline(0.5, color="black", lw=1.5)  # separate target row

# White grid between cells
ax.set_xticks(np.arange(-0.5, n_slots), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="minor", length=0)

cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("p.p. po satu")
ax.set_title("Porast izlaznosti po satu (p.p.)")
fig.text(
    0.01,
    0.01,
    "Podebljano = izmereno za taj sat.  Svetlo kurzivom = prosek preko perioda "
    "bez merenja (uklj. 7h do prvog merenja).",
    fontsize=9,
    color="#555",
)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT, dpi=140)
plt.show()
