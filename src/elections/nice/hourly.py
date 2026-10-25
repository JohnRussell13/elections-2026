"""Izlaznost po satu: turnout gained in each hour, one small panel per election."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import chart_style as cs

cs.apply()
df, hours = cs.load_data()
order = [cs.TARGET] + [n for n in df.index[::-1] if n != cs.TARGET]

ncols = 3
nrows = int(np.ceil(len(order) / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(15, 3.1 * nrows + 1.6), sharex=True, sharey=True
)
axes = np.atleast_2d(axes)

for ax, name in zip(axes.flat, order):
    hh, vv = cs.readings(df.loc[name], hours)
    is_target = name == cs.TARGET
    solid = cs.ACCENT if is_target else cs.PAST
    soft = cs.ACCENT_SOFT if is_target else cs.PAST_SOFT

    for i in range(1, len(hh)):
        start, end = hh[i - 1], hh[i]
        span = end - start
        rate = (vv[i] - vv[i - 1]) / span
        lead = i == 1 and span > 1  # from poll opening to the first reading
        gap = span > 1 and not lead

        ax.bar(
            start + span / 2,
            rate,
            width=span - 0.14,
            color=cs.PANEL if lead else (soft if gap else solid),
            edgecolor=soft if lead else "none",
            hatch="////" if lead else None,
            linewidth=1.0,
        )
        ax.text(
            start + span / 2,
            rate + 0.18,
            f"{rate:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=cs.INK if is_target else cs.INK_2,
            fontweight="bold" if is_target else "normal",
        )

    final = df.loc[name, f"{cs.FINAL_HOUR}h"]
    ax.set_title(name.rstrip("."), color=cs.ACCENT if is_target else cs.INK)
    status = f"{final:.1f}% ukupno" if pd.notna(final) else "u toku"
    ax.text(
        1,
        1.035,
        status,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=cs.INK_3,
    )
    ax.set_xlim(cs.POLLS_OPEN, cs.FINAL_HOUR)

for ax in axes.flat[len(order) :]:
    ax.set_visible(False)
for ax in axes[-1]:
    cs.hour_axis(ax, every=1)
    ax.tick_params(axis="x", labelsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("p.p. po satu")
axes.flat[0].set_ylim(0, 10.5)

cs.header(
    fig,
    "Izlaznost po satu",
    "Koliko je procentnih poena izlaznosti dodato u svakom satu. "
    f"{cs.TARGET} je prikazan crveno.",
)
cs.header_legend(
    fig,
    [
        Patch(facecolor=cs.PAST, label="izmereno za taj sat"),
        Patch(facecolor=cs.PAST_SOFT, label="prosek po satu, nema merenja između"),
        Patch(
            facecolor=cs.PANEL,
            edgecolor=cs.PAST_SOFT,
            hatch="////",
            label="prosek po satu od otvaranja (7h) do prvog merenja",
        ),
    ],
)
cs.footer(fig, "Prazni periodi su raspodeljeni ravnomerno po satima")
cs.save(fig, "hourly.png", header_in=1.3)
plt.show()
