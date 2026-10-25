"""Share of the final turnout reached by each hour (every election ends at 100%)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs

LABEL_HOURS = [9, 11, 13, 15, 17]

cs.apply()
df, hours = cs.load_data()
grid = np.arange(cs.POLLS_OPEN, cs.FINAL_HOUR + 1)
final_col = f"{cs.FINAL_HOUR}h"
finished = [n for n in df.index[::-1] if pd.notna(df.loc[n, final_col])]
unfinished = [n for n in df.index if pd.isna(df.loc[n, final_col])]

fig, ax = plt.subplots(figsize=(13, 7.2))
shares = {}
for name in finished:
    x, y = cs.readings(df.loc[name], hours)
    share = y / y[-1] * 100
    shares[name] = pd.Series(np.interp(grid, x, share), index=grid)
    is_target = name == cs.TARGET  # 2026 appears once its 20h value is in
    ax.plot(
        x,
        share,
        color=cs.ACCENT if is_target else cs.CONTEXT,
        lw=2.6 if is_target else 1.2,
        zorder=4 if is_target else 2,
        marker="o" if is_target else None,
        ms=5,
        label=cs.TARGET if is_target else None,
    )

past = pd.DataFrame({k: v for k, v in shares.items() if k != cs.TARGET})
median = past.median(axis=1)
lo, hi = past.min(axis=1), past.max(axis=1)
ax.fill_between(
    grid,
    lo,
    hi,
    color=cs.PAST_SOFT,
    alpha=0.25,
    lw=0,
    zorder=1,
    label="raspon prošlih izbora",
)
ax.plot(grid, median, color=cs.PAST, lw=2.6, zorder=3, label="medijana prošlih izbora")

for h in LABEL_HOURS:
    ax.plot(h, median[h], "o", ms=8, color=cs.PAST, mec=cs.PANEL, mew=2, zorder=5)
    ax.annotate(
        f"{median[h]:.0f}%",
        (h, median[h]),
        xytext=(-10, 12),
        textcoords="offset points",
        ha="right",
        fontsize=11,
        fontweight="bold",
        color=cs.INK,
    )
    ax.annotate(
        f"do {h}h",
        (h, median[h]),
        xytext=(-10, 26),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color=cs.INK_3,
    )

cs.hour_axis(ax)
ax.set_xlim(cs.POLLS_OPEN, cs.FINAL_HOUR + 0.15)
ax.set_ylim(0, 104)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
ax.legend(loc="lower right")

subtitle = (
    "Svaki izbori su podeljeni svojom konačnom izlaznošću, pa svi završavaju "
    "na 100%. Crno: tipičan tok dana."
)
cs.header(fig, "Koliki deo konačne izlaznosti je glasao do svakog sata", subtitle)
note = (
    f"{', '.join(unfinished)} nije prikazan dok konačna izlaznost ne bude poznata"
    if unfinished
    else "Nedostajući sati linearno interpolirani"
)
cs.footer(fig, note)
cs.save(fig, "normalized.png")
plt.show()
