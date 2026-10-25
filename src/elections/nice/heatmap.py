"""Heatmap: turnout gained in each hour, one row per election."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import chart_style as cs

cs.apply()
df, hours = cs.load_data()
rows = [cs.TARGET] + [n for n in df.index[::-1] if n != cs.TARGET]
n_slots = cs.FINAL_HOUR - cs.POLLS_OPEN

# Gain per hour; gaps are spread evenly over the hours they cover
gains = np.full((len(rows), n_slots), np.nan)
estimated = np.zeros_like(gains, dtype=bool)
for i, name in enumerate(rows):
    hh, vv = cs.readings(df.loc[name], hours)
    for j in range(1, len(hh)):
        start, end = hh[j - 1], hh[j]
        for h in range(start, end):
            gains[i, h - cs.POLLS_OPEN] = (vv[j] - vv[j - 1]) / (end - start)
            estimated[i, h - cs.POLLS_OPEN] = (end - start) > 1

VMIN, VMAX = 1.5, 7.5
cmap = LinearSegmentedColormap.from_list("red", cs.RAMP)

fig, ax = plt.subplots(figsize=(14, 0.52 * len(rows) + 2.4))
ax.grid(False)
im = ax.imshow(gains, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

for i in range(len(rows)):
    for j in range(n_slots):
        v = gains[i, j]
        if np.isnan(v):
            continue
        dark_cell = (v - VMIN) / (VMAX - VMIN) > 0.55
        est = estimated[i, j]
        ax.text(
            j,
            i,
            f"{v:.1f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if dark_cell else cs.INK,
            alpha=0.6 if est else 1,
            fontweight="normal" if est else "bold",
        )

# Accent outline around the current election's row
ax.add_patch(
    Rectangle(
        (-0.5, -0.5),
        n_slots,
        1,
        fill=False,
        edgecolor=cs.RAMP_EDGE,
        linewidth=2.2,
        clip_on=False,
    )
)

ax.set_xticks(range(n_slots))
ax.set_xticklabels(
    [f"{h}–{h + 1}h" for h in range(cs.POLLS_OPEN, cs.FINAL_HOUR)], fontsize=9
)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r.rstrip(".") for r in rows])
ax.get_yticklabels()[0].set_color(cs.ACCENT)
ax.get_yticklabels()[0].set_fontweight("bold")

# Surface-colored gaps between cells
ax.set_xticks(np.arange(-0.5, n_slots), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color=cs.PANEL, linewidth=2)
ax.tick_params(which="both", length=0)
for side in ax.spines.values():
    side.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelcolor=cs.INK_2)
cb.set_label("p.p. po satu", color=cs.INK_2)

cs.header(
    fig,
    "Porast izlaznosti po satu",
    "Tamnije = više birača u tom satu. Svaki red je jedni izbori, "
    f"{cs.TARGET} je na vrhu.",
)
cs.footer(
    fig,
    "Podebljano = izmereno za taj sat; svetlije = prosek preko perioda "
    "bez merenja (uklj. 7h do prvog merenja)",
)
cs.save(fig, "heatmap.png")
plt.show()
